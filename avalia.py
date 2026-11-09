"""Régua do detector sem chave: acerto (TPR) com 1% de falso positivo, por idioma e por ataque.

Divisão por família (o mesmo artigo e as reescritas dele ficam do mesmo lado):
  - pt: as 20 famílias que passaram pelo babelllm vão inteiras para o teste,
    junto com 30% das outras; o resto é treino (55%) e calibração (15%).
  - en: 55/15/30 por título do RAID.
O limiar é escolhido nos humanos da calibração (1% de falso positivo) e aplicado ao teste.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from extrair import DADOS, carregar


def _u(s: str) -> float:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def divisao(linhas) -> np.ndarray:
    babel = {l["familia"] for l in linhas if l["fonte"].startswith("babel")}
    out = []
    for l in linhas:
        f = l["familia"]
        if f in babel:
            out.append("teste")
            continue
        u = _u(f)
        out.append("treino" if u < 0.55 else ("calib" if u < 0.70 else "teste"))
    return np.array(out)


def limiar_fpr(scores_humanos: np.ndarray, fpr: float) -> float:
    return float(np.quantile(scores_humanos, 1 - fpr))


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1
    rp = r[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def relatorio(linhas, score, split, fprs=(0.01, 0.05), titulo=""):
    """score: maior = mais IA."""
    linhas = np.array(linhas, dtype=object)
    res = {"titulo": titulo, "linguas": {}}
    for lg in ("pt", "en"):
        sel = np.array([l["lingua"] == lg for l in linhas])
        hc = score[sel & (split == "calib") & np.array([l["rotulo"] == 0 for l in linhas])]
        ht = score[sel & (split == "teste") & np.array([l["rotulo"] == 0 for l in linhas])]
        if len(hc) == 0 or len(ht) == 0:
            continue
        d = {"n_humanos_teste": int(len(ht)), "fpr": {}, "grupos": {}}
        lim = {f: limiar_fpr(hc, f) for f in fprs}
        for f in fprs:
            d["fpr"][f"{f:.0%}"] = round(float((ht > lim[f]).mean()), 4)
        grupos = {}
        for i in np.where(sel & (split == "teste"))[0]:
            l = linhas[i]
            if l["rotulo"] == 1:
                grupos.setdefault((l["fonte"], l["ataque"]), []).append(score[i])
        todos = []
        for g, v in sorted(grupos.items()):
            v = np.array(v)
            todos.append(v)
            d["grupos"][f"{g[0]}|{g[1]}"] = {"n": int(len(v)), "auroc": round(float(auroc(v, ht)), 3),
                                              **{f"tpr@{f:.0%}": round(float((v > lim[f]).mean()), 3) for f in fprs}}
        v = np.concatenate(todos)
        d["ia_toda"] = {"n": int(len(v)), "auroc": round(float(auroc(v, ht)), 3),
                        **{f"tpr@{f:.0%}": round(float((v > lim[f]).mean()), 3) for f in fprs}}
        res["linguas"][lg] = d
    return res


def imprimir(res):
    print(f"== {res['titulo']}")
    for lg, d in res["linguas"].items():
        print(f"  [{lg}] humanos no teste: {d['n_humanos_teste']}; falso positivo real: {d['fpr']}")
        print(f"   {'TODA A IA':40s} n={d['ia_toda']['n']:4d} auroc={d['ia_toda']['auroc']:.3f} "
              f"tpr@1%={d['ia_toda']['tpr@1%']:.3f} tpr@5%={d['ia_toda']['tpr@5%']:.3f}")
        for g, x in d["grupos"].items():
            print(f"   {g:40s} n={x['n']:4d} auroc={x['auroc']:.3f} tpr@1%={x['tpr@1%']:.3f} tpr@5%={x['tpr@5%']:.3f}")


if __name__ == "__main__":
    linhas = carregar()
    s = divisao(linhas)
    import collections
    print(collections.Counter(zip([l["lingua"] for l in linhas], [l["rotulo"] for l in linhas], s)))
