"""Falso positivo por chave, para MIL chaves de uma vez.

Cada isca vira, por sua vez, a "chave verdadeira" e é comparada com as outras 999. Assim se
vê a distribuição do falso positivo entre chaves (não só a média), em três versões:
sem branqueamento, com branqueamento e com branqueamento + calibração por chave
(calibração feita em textos humanos do lado "treino"; medida no lado "teste").
"""
from __future__ import annotations

import json

import numpy as np

from avalia import divisao
from config import resultado
from exp_selo_fpr import docs_humanos
from extrair import carregar
from selo import BRANQUEADOR_PADRAO, Selo


def zmax_docs(selo: Selo, docs):
    out = []
    for d in docs:
        zs, _, _ = selo.zs(d)
        Z = np.stack(list(zs.values()))
        if selo.calib:
            Z = np.stack([(zs[k] - selo.calib[k][0]) / selo.calib[k][1] for k in zs])
        out.append(Z.max(0))
    return np.array(out)  # (docs, 1+R)


def fpr_por_chave(ZM):
    # posto de cada chave dentro do documento: p_j = (1 + nº de chaves com z maior) / total
    r = (-ZM).argsort(1).argsort(1)
    p = (1 + r) / ZM.shape[1]
    return (p <= 0.01).mean(0), (p <= 0.05).mean(0)


def main():
    linhas = carregar()
    split = divisao(linhas)
    ref = docs_humanos(linhas, split, "treino", "pt", 400) + docs_humanos(linhas, split, "treino", "en", 400)
    teste = {lg: docs_humanos(linhas, split, "teste", lg, 250) for lg in ("pt", "en")}
    base = Selo("chave-de-teste-luiz", branqueador=None)
    W = np.load(BRANQUEADOR_PADRAO)
    res = {}
    for nome, w, cal in (("cru", None, False), ("branqueado", W, False), ("branq+calib", W, True)):
        s = Selo("chave-de-teste-luiz", modelo=base.modelo, branqueador=w)
        s._cache = base._cache
        if cal:
            s.calibrar(ref)
            s.salvar_calibracao(resultado("calib_chave_teste.npz"))
        for lg, docs in teste.items():
            f1, f5 = fpr_por_chave(zmax_docs(s, docs))
            res[f"{nome}|{lg}"] = dict(media1=float(f1.mean()), media5=float(f5.mean()),
                                       p90_1=float(np.quantile(f1, .9)), p99_1=float(np.quantile(f1, .99)),
                                       p90_5=float(np.quantile(f5, .9)), p99_5=float(np.quantile(f5, .99)),
                                       chaves_acima_2pct=float((f1 > 0.02).mean()),
                                       chaves_acima_8pct_a5=float((f5 > 0.08).mean()),
                                       chave_luiz=[float(f1[0]), float(f5[0])])
            r = res[f"{nome}|{lg}"]
            print(f"{nome:12s} {lg} falso+ a 1%: média {r['media1']:.3f}, p90 {r['p90_1']:.3f}, p99 {r['p99_1']:.3f},"
                  f" chaves>2% {r['chaves_acima_2pct']:.2f} | a 5%: média {r['media5']:.3f}, p90 {r['p90_5']:.3f},"
                  f" p99 {r['p99_5']:.3f} | chave luiz {r['chave_luiz']}", flush=True)
    json.dump(res, open(resultado("exp_selo_fpr2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
