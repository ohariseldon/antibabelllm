"""Qual modelo de vetor deixa a nota do selo mais estável sob paráfrase?

Usa as reescritas do Haiku: para cada frase original, as versões dela nas reescritas são
paráfrases. Confiabilidade = parte da variação da nota que vem do conteúdo (entre frases)
e não das palavras (entre paráfrases da mesma frase). Quanto maior, mais a marca feita na
hora de gerar sobrevive a uma paráfrase. Mede com 50 chaves para tirar a média.
"""
from __future__ import annotations

import sys

import numpy as np

from config import ARTIGOS
from exp_selo_pool import grupos
from selo import Selo
import glob
import os


def confiabilidade(selo_base: Selo, pools, chaves=50):
    """pools: lista de documentos; cada documento é lista de grupos (frase + paráfrases)."""
    rs = []
    for k in range(chaves):
        s = Selo(f"k{k}", modelo=selo_base.modelo, branqueador=None, n_iscas=1, m_eixos=128)
        s._cache = selo_base._cache
        s.W = selo_base.W
        entre, dentro = [], []
        for doc in pools:
            ref = s.vetores([g[0] for g in doc])
            media = ref.mean(0)
            for g in doc:
                if len(g) < 3:
                    continue
                q = s.nota(s.residuos(s.vetores(g), media))
                entre.append(q.mean())
                dentro.append(q.var(ddof=1))
        vb = np.var(entre) - np.mean(dentro) / 4  # corrige a média de ~4 versões
        rs.append(vb / (vb + np.mean(dentro)))
    return float(np.mean(rs)), float(np.std(rs))


def main():
    nomes = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{ARTIGOS}/*.md"))
    for emb in sys.argv[1:] or ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]:
        base = Selo("x", embedder=emb, branqueador=None, n_iscas=1)
        pools = [grupos(base, n) for n in nomes]
        m, sd = confiabilidade(base, pools)
        print(f"{emb:70s} confiabilidade={m:.3f} ± {sd:.3f}  versões/frase={np.mean([len(g) for d in pools for g in d]):.2f}",
              flush=True)


if __name__ == "__main__":
    main()
