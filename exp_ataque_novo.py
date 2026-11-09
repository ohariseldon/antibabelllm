"""Mede uma versão nova do ataque (uma pasta de saídas do babelllm) no detector sem chave.

Mesmas 5 dobras por artigo do resto da régua: o teste é sempre nos artigos cujas saídas
o detector não viu. Três detectores, todos com a Agência Brasil no treino como humano
(controle de assunto e época):
  - nunca viu babelllm;
  - sem retreinar: viu as saídas antigas do babelllm dos outros artigos, não o ataque novo;
  - retreinando: viu também o ataque novo nos outros artigos.

  python exp_ataque_novo.py <pasta> <nome> [--comparar babel_prosa]
"""
from __future__ import annotations

import glob
import os
import random
import sys

import numpy as np

from avalia import divisao
from config import DADOS
from exp_negativos import _u, features, janelas_agencia, features_agencia
from extrair import carregar
from textos import janelas, limpar_markdown
from treina import _clf, _score, matriz

BASE = DADOS.rsplit(".", 1)[0]


def janelas_pasta(pasta, nome):
    rng = random.Random(5)
    out = []
    for f in sorted(glob.glob(os.path.join(pasta, "*.md"))):
        js = janelas(limpar_markdown(open(f, encoding="utf-8").read()), 200, minimo=120)
        if len(js) > 4:
            js = rng.sample(js, 4)
        for j in js:
            out.append(dict(lingua="pt", rotulo=1, fonte="babel_novo", ataque=nome,
                            familia="wasemu:" + os.path.basename(f)[:-3], texto=j))
    return out


def main():
    pasta, nome = sys.argv[1], sys.argv[2]
    comparar = sys.argv[sys.argv.index("--comparar") + 1] if "--comparar" in sys.argv else "babel_prosa"
    blocos = sys.argv[sys.argv.index("--blocos") + 1].split(",") if "--blocos" in sys.argv else ["estilo", "emb"]
    linhas = carregar()
    split = divisao(linhas)
    y = np.array([l["rotulo"] for l in linhas])
    X = matriz(blocos, linhas, "pt")
    ok = ~np.isnan(X).any(1) & np.array([l["lingua"] == "pt" for l in linhas])
    fonte = np.array([l["fonte"] for l in linhas])
    atk = np.array([l["ataque"] for l in linhas])
    fam = np.array([l["familia"] for l in linhas])
    babel = np.array([f.startswith("babel") for f in fonte])

    ag = janelas_agencia()
    Xa = features_agencia(ag, blocos)
    oka = ~np.isnan(Xa).any(1)
    ag = [l for l, k in zip(ag, oka) if k]
    Xa = Xa[oka]
    lado_a = np.array(["treino" if _u(l["familia"]) < 0.4 else ("calib" if _u(l["familia"]) < 0.55 else "teste")
                       for l in ag])

    nv = janelas_pasta(pasta, nome)
    Xn = features(nv, blocos, f"{BASE}.novo_{nome}.npz")
    okn = ~np.isnan(Xn).any(1)
    nv = [l for l, k in zip(nv, okn) if k]
    Xn = Xn[okn]
    famn = np.array([l["familia"] for l in nv])
    print(f"ataque novo '{nome}': {len(nv)} janelas de {len(set(famn))} artigos")

    fams_b = sorted({l["familia"] for l in linhas if l["fonte"].startswith("babel")})
    rng = np.random.RandomState(0)
    rng.shuffle(fams_b)
    dobras = [set(fams_b[i::5]) for i in range(5)]
    res = {}
    for modo in ("nunca viu babelllm", "sem retreinar", "retreinando"):
        acerto_novo, acerto_cmp, fpa = [], [], []
        for d in dobras:
            tr = ok & (split == "treino") & ~babel
            if modo != "nunca viu babelllm":
                tr |= ok & babel & ~np.isin(fam, list(d))
            Xtr = [X[tr], Xa[lado_a == "treino"]]
            ytr = [y[tr], np.zeros(np.sum(lado_a == "treino"), int)]
            if modo == "retreinando":
                m = ~np.isin(famn, list(d))
                Xtr.append(Xn[m])
                ytr.append(np.ones(m.sum(), int))
            clf = _clf("lr").fit(np.concatenate(Xtr), np.concatenate(ytr))
            s = np.full(len(X), np.nan)
            s[ok] = _score(clf, X[ok], "lr")
            sa, sn = _score(clf, Xa, "lr"), _score(clf, Xn, "lr")
            hc = np.concatenate([s[ok & (split == "calib") & (y == 0)], sa[lado_a == "calib"]])
            l1, l5 = np.quantile(hc, 0.99), np.quantile(hc, 0.95)
            fpa.append((sa[lado_a == "teste"] > l1).mean())
            mt = np.isin(famn, list(d))
            acerto_novo += [(v > l1, v > l5) for v in sn[mt]]
            mc = ok & np.isin(fam, list(d)) & (atk == comparar)
            acerto_cmp += [(v > l1, v > l5) for v in s[mc]]
        a, c = np.array(acerto_novo), np.array(acerto_cmp)
        res[modo] = (a[:, 0].mean(), a[:, 1].mean(), c[:, 0].mean(), np.mean(fpa))
        print(f"== {modo:20s} [{'+'.join(blocos)}] {nome}: tpr@1%={a[:, 0].mean():.3f} tpr@5%={a[:, 1].mean():.3f} (n={len(a)}) | "
              f"{comparar}: tpr@1%={c[:, 0].mean():.3f} | falso+ Agência teste @1%={np.mean(fpa):.3f}", flush=True)


if __name__ == "__main__":
    main()
