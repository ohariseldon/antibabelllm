"""Experimento 1 da marca: candidatos reais do Haiku, sem gastar chamada.

Para cada um dos 20 artigos, cada frase do original ganha um grupo de versões: ela mesma
e a frase mais parecida em cada reescrita do babelllm (Haiku). O texto marcado escolhe,
frase a frase, a versão de nota maior pela chave; o texto de controle escolhe ao acaso.
Depois os dois passam por ataques de estrutura (embaralhar, fundir, dividir, cortar) e
o detector mede z e p.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys

import numpy as np

from config import ARTIGOS, BABEL, resultado
from selo import Selo
from textos import frases, limpar_markdown

ORIG = ARTIGOS
REESC = ["out_haiku20_prosa", "out_haiku20_forte", "out_haiku20_nomes", "out_haiku20"]
BASE = BABEL


def grupos(selo: Selo, nome: str, piso=0.70):
    fo = frases(limpar_markdown(open(f"{ORIG}/{nome}.md", encoding="utf-8").read()))
    Vo = selo.vetores(fo)
    pools = [[f] for f in fo]
    for d in REESC:
        p = f"{BASE}/{d}/{nome}.md"
        if not os.path.exists(p):
            continue
        fr = frases(limpar_markdown(open(p, encoding="utf-8").read()))
        if not fr:
            continue
        Vr = selo.vetores(fr)
        S = Vo @ Vr.T
        for i in range(len(fo)):
            j = int(S[i].argmax())
            if S[i, j] >= piso and fr[j] not in pools[i]:
                pools[i].append(fr[j])
    return pools


def ataques(fs: list[str], rng: random.Random) -> dict[str, str]:
    out = {"nenhum": " ".join(fs)}
    e = fs[:]
    rng.shuffle(e)
    out["embaralhar"] = " ".join(e)
    fund = []
    i = 0
    while i < len(fs):
        if i + 1 < len(fs) and rng.random() < 0.7:
            a, b = fs[i].rstrip("."), fs[i + 1]
            fund.append(a + ", e " + b[0].lower() + b[1:])
            i += 2
        else:
            fund.append(fs[i])
            i += 1
    out["fundir_70%"] = " ".join(fund)
    div = []
    for f in fs:
        partes = [p.strip() for p in f.split(", ") if len(p.split()) >= 3]
        if len(partes) > 1:
            div += [p.rstrip(".") + "." for p in partes]
        else:
            div.append(f)
    out["dividir_virgulas"] = " ".join(div)
    keep = [f for f in fs if rng.random() > 0.3]
    out["cortar_30%"] = " ".join(keep)
    mix = fund[:]
    rng.shuffle(mix)
    out["fundir+embaralhar"] = " ".join(mix)
    return out


def main():
    chave = next((x for x in sys.argv[1:] if not x.startswith("--")), "chave-de-teste-luiz")
    selo = Selo(chave)
    cal = resultado("calib_chave_teste.npz")
    if "--calib" in sys.argv and os.path.exists(cal):
        selo.carregar_calibracao(cal)
        print("com calibração por chave", flush=True)
    rng = random.Random(1)
    nomes = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{ORIG}/*.md"))
    res = {"marcado": {}, "controle": {}}
    textos_salvos = {}
    tam = []
    for nome in nomes:
        pools = grupos(selo, nome)
        tam.append(np.mean([len(p) for p in pools]))
        esc = selo.escolher(pools)
        marcado = [p[k] for p, k in zip(pools, esc)]
        controle = [rng.choice(p) for p in pools]
        textos_salvos[nome] = {"marcado": marcado, "controle": controle, "original": [p[0] for p in pools]}
        for rot, fs in (("marcado", marcado), ("controle", controle)):
            for atk, txt in ataques(fs, rng).items():
                r = selo.detectar(txt)
                res[rot].setdefault(atk, []).append((r.z, r.p))
        print(nome[:40], "versões/frase", round(tam[-1], 2),
              "z marcado", round(res["marcado"]["nenhum"][-1][0], 2),
              "z controle", round(res["controle"]["nenhum"][-1][0], 2), flush=True)
    print("\nversões por frase (média):", round(float(np.mean(tam)), 2))
    for rot in ("marcado", "controle"):
        print(f"== {rot}")
        for atk, v in res[rot].items():
            v = np.array(v)
            print(f"   {atk:20s} z médio={v[:, 0].mean():5.2f}  detectado a 1%={(v[:, 1] <= 0.01).mean():.2f}"
                  f"  a 5%={(v[:, 1] <= 0.05).mean():.2f}")
    sufixo = ("" if selo.W is not None else "_cru") + ("_calib" if selo.calib else "")
    json.dump({k: {a: v for a, v in d.items()} for k, d in res.items()},
              open(resultado(f"exp_selo_pool{sufixo}.json"), "w"))
    json.dump(textos_salvos, open(resultado(f"exp_selo_pool_textos{sufixo}.json"), "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
