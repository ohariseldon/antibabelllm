"""Pesa as direções do espaço de sentido pela estabilidade sob paráfrase.

C_w = covariância das diferenças entre paráfrases da mesma frase (reescritas do Haiku).
A transformação C_w^(-1/2) aumenta as direções que a paráfrase mexe pouco e encolhe as que
ela mexe muito. Confiabilidade medida em artigos que não entraram na estimativa (2 dobras).
"""
from __future__ import annotations

import glob
import os

import numpy as np

from config import ARTIGOS
from exp_embedder import confiabilidade
from exp_selo_pool import grupos
from selo import BRANQUEADOR_PADRAO, Selo


def raiz_inv(C, encolher=0.1):
    d = len(C)
    C = (1 - encolher) * C + encolher * np.trace(C) / d * np.eye(d)
    w, U = np.linalg.eigh(C)
    return ((U / np.sqrt(w)) @ U.T * np.sqrt(np.trace(C) / d)).astype(np.float32)


def cov_dentro(selo, docs, T=None):
    D = []
    for doc in docs:
        for g in doc:
            if len(g) < 2:
                continue
            V = selo.vetores(g)
            V = V - V.mean(0)
            if T is not None:
                V = V @ T
            D.append(V)
    D = np.concatenate(D)
    return D.T @ D / len(D)


def main():
    nomes = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{ARTIGOS}/*.md"))
    base = Selo("x", branqueador=None, n_iscas=1)
    pools = [grupos(base, n) for n in nomes]
    Wf = np.load(BRANQUEADOR_PADRAO)
    res = {}
    for dobra in (0, 1):
        est = [p for i, p in enumerate(pools) if i % 2 == dobra]
        tst = [p for i, p in enumerate(pools) if i % 2 != dobra]
        Tw = raiz_inv(cov_dentro(base, est))
        Twf = Wf @ raiz_inv(cov_dentro(base, est, Wf))
        for nome, T in (("nenhuma", None), ("branqueador Folha", Wf), ("estável C_w^-1/2", Tw),
                        ("branqueador + estável", Twf)):
            base.W = T
            m, _ = confiabilidade(base, tst, chaves=30)
            res.setdefault(nome, []).append(m)
    base.W = None
    for k, v in res.items():
        print(f"{k:24s} confiabilidade={np.mean(v):.3f} (dobras {np.round(v, 3).tolist()})", flush=True)


if __name__ == "__main__":
    main()
