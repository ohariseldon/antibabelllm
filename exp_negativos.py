"""O detector sem chave reconhece IA ou só o assunto/época?

Os humanos da régua pt são da Folha (2015-2017); os textos de IA são de 2026, sobre dados
da economia brasileira. Aqui entram notícias humanas de economia da Agência Brasil de
2025-2026 (mesmo período, mesmos temas), em dois testes:
  A. o detector como estava (sem Agência no treino): quanto delas ele acusa de IA?
  B. metade das notícias da Agência no treino como humano, a outra metade no teste:
     o falso positivo volta ao nominal? e quanto do babelllm ainda é pego?

  python exp_negativos.py estilo emb
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

import numpy as np

import estilo
from avalia import divisao
from config import RAW
from extrair import DADOS, carregar, mascarar_digitos
from textos import janelas, limpar_simples
from treina import _clf, _score, matriz

AGENCIA = os.path.join(RAW, "agencia_brasil.jsonl")
BASE = DADOS.rsplit(".", 1)[0]


def _u(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def janelas_agencia():
    rng = random.Random(3)
    out = []
    for l in open(AGENCIA, encoding="utf-8"):
        d = json.loads(l)
        js = janelas(limpar_simples(d["texto"]), 200, minimo=120)
        if len(js) > 1:
            js = rng.sample(js, 1)
        for j in js:
            out.append(dict(lingua="pt", rotulo=0, fonte="agencia_brasil", ataque="nenhum",
                            familia="ab:" + d["url"], texto=j))
    return out


def features_agencia(linhas, blocos):
    return features(linhas, blocos, f"{BASE}.agencia.npz")


def features(linhas, blocos, cache):
    try:
        z = np.load(cache)
        if len(z["estilo"]) == len(linhas):
            return np.concatenate([z[b] for b in blocos], 1)
    except FileNotFoundError:
        pass
    E = np.array([estilo.vetor(l["texto"], "pt") for l in linhas], dtype=np.float32)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    m = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
    m.max_seq_length = 320
    M = m.encode(["passage: " + mascarar_digitos(l["texto"]) for l in linhas], batch_size=16,
                 normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    np.savez(cache, estilo=E, emb=M)
    return np.concatenate([{"estilo": E, "emb": M}[b] for b in blocos], 1)


def main():
    blocos = [a for a in sys.argv[1:] if not a.startswith("--")] or ["estilo", "emb"]
    modelo = "hgb" if "--hgb" in sys.argv else "lr"
    linhas = carregar()
    split = divisao(linhas)
    y = np.array([l["rotulo"] for l in linhas])
    pt = np.array([l["lingua"] == "pt" for l in linhas])
    X = matriz(blocos, linhas, "pt")
    ok = ~np.isnan(X).any(1) & pt
    fonte = np.array([l["fonte"] for l in linhas])
    atk = np.array([l["ataque"] for l in linhas])
    babel = np.array([f.startswith("babel") for f in fonte])

    ag = janelas_agencia()
    Xa = features_agencia(ag, blocos)
    oka = ~np.isnan(Xa).any(1)
    ag = [l for l, k in zip(ag, oka) if k]
    Xa = Xa[oka]
    lado_a = np.array(["treino" if _u(l["familia"]) < 0.4 else ("calib" if _u(l["familia"]) < 0.55 else "teste")
                       for l in ag])
    print(f"Agência Brasil: {len(ag)} janelas (treino {np.sum(lado_a == 'treino')}, calib "
          f"{np.sum(lado_a == 'calib')}, teste {np.sum(lado_a == 'teste')})")

    def rodar(com_agencia: bool, com_babel_fams: set | None):
        """com_babel_fams: famílias do babelllm que podem entrar no treino (None = nenhuma)."""
        fam = np.array([l["familia"] for l in linhas])
        tr = ok & (split == "treino") & ~babel
        if com_babel_fams:
            tr |= ok & babel & np.isin(fam, list(com_babel_fams))
        Xtr, ytr = X[tr], y[tr]
        if com_agencia:
            Xtr = np.concatenate([Xtr, Xa[lado_a == "treino"]])
            ytr = np.concatenate([ytr, np.zeros(np.sum(lado_a == "treino"), int)])
        clf = _clf(modelo).fit(Xtr, ytr)
        s = np.full(len(X), np.nan)
        s[ok] = _score(clf, X[ok], modelo)
        sa = _score(clf, Xa, modelo)
        hc = s[ok & (split == "calib") & (y == 0)]
        if com_agencia:
            hc = np.concatenate([hc, sa[lado_a == "calib"]])
        return s, sa, np.quantile(hc, 0.99), np.quantile(hc, 0.95)

    fams_b = sorted({l["familia"] for l in linhas if l["fonte"].startswith("babel")})
    rng = np.random.RandomState(0)
    rng.shuffle(fams_b)
    dobras = [set(fams_b[i::5]) for i in range(5)]
    fam = np.array([l["familia"] for l in linhas])
    for com_agencia in (False, True):
        for com_babel in (False, True):
            fp_folha, fp_ag, tp = [], [], {}
            for d in dobras:
                s, sa, l1, l5 = rodar(com_agencia, (set(fams_b) - d) if com_babel else None)
                hf = s[ok & (split == "teste") & (y == 0) & ~np.isin(fonte, ["folha_traduzidas"])]
                at = sa[lado_a == "teste"]
                fp_folha.append(((hf > l1).mean(), (hf > l5).mean()))
                fp_ag.append(((at > l1).mean(), (at > l5).mean()))
                for i in np.where(ok & babel & np.isin(fam, list(d)))[0]:
                    tp.setdefault(atk[i], []).append((s[i] > l1, s[i] > l5))
                for i in np.where(ok & (fonte == "wasemu_claude") & (split == "teste"))[0]:
                    tp.setdefault("wasemu_original", []).append((s[i] > l1, s[i] > l5))
            fpf, fpa = np.mean(fp_folha, 0), np.mean(fp_ag, 0)
            tb = np.concatenate([np.array(v) for k, v in tp.items() if k != "wasemu_original"])
            tw = np.array(tp["wasemu_original"])
            print(f"== treino {'COM' if com_agencia else 'sem'} Agência, {'COM' if com_babel else 'sem'} babelllm "
                  f"({'+'.join(blocos)}, {modelo})")
            print(f"   falso+ Folha teste: 1%→{fpf[0]:.3f} 5%→{fpf[1]:.3f} | falso+ Agência teste: "
                  f"1%→{fpa[0]:.3f} 5%→{fpa[1]:.3f}")
            print(f"   pega babelllm: 1%→{tb[:, 0].mean():.3f} 5%→{tb[:, 1].mean():.3f} (n={len(tb)}) | "
                  f"wasemu original: 1%→{tw[:, 0].mean():.3f}")
            print("   por ataque @1%: " + " ".join(f"{k}={np.mean([a for a, _ in v]):.2f}"
                                               for k, v in sorted(tp.items()) if k != "wasemu_original"), flush=True)


if __name__ == "__main__":
    main()
