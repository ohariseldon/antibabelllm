"""Falso positivo real da marca para uma chave FIXA, em texto humano que ela nunca viu.

O p-valor pelo posto entre iscas é exato na média das chaves. Mas o dono usa UMA chave
em milhares de textos do mesmo gênero; se essa chave tiver um viés fixo, o falso positivo
dela passa do nominal. Aqui: 5 chaves, textos humanos da Folha (pt) e do RAID (en) do lado
"teste", com e sem branqueamento (ajustado só no lado "treino").
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np

from avalia import divisao
from config import resultado
from extrair import carregar
from selo import BRANQUEADOR_PADRAO, Selo, ajustar_branqueador
from textos import frases


def docs_humanos(linhas, split, lado, lg, limite):
    fam = collections.OrderedDict()
    for l, s in zip(linhas, split):
        if l["rotulo"] == 0 and l["lingua"] == lg and s == lado:
            fam.setdefault(l["familia"], []).append(l["texto"])
    docs = [" ".join(v) for v in fam.values()]
    rng = np.random.RandomState(0)
    rng.shuffle(docs)
    return [d for d in docs if len(frases(d)) >= 8][:limite]


def main():
    linhas = carregar()
    split = divisao(linhas)
    base = Selo("x", branqueador=None)
    if "--ajustar" in sys.argv:
        treino = [frases(d) for lg in ("pt", "en") for d in docs_humanos(linhas, split, "treino", lg, 1500)]
        print("ajustando com", len(treino), "documentos,", sum(map(len, treino)), "frases", flush=True)
        W = ajustar_branqueador(treino, base.modelo)
        np.save(BRANQUEADOR_PADRAO, W.astype(np.float32))
    W = np.load(BRANQUEADOR_PADRAO)
    teste = {lg: docs_humanos(linhas, split, "teste", lg, 250) for lg in ("pt", "en")}
    print({k: len(v) for k, v in teste.items()}, flush=True)
    out = {}
    for chave in ["chave-de-teste-luiz", "k1", "k2", "k3", "k4"]:
        for nome, w in (("cru", None), ("branqueado", W)):
            s = Selo(chave, modelo=base.modelo, branqueador=w)
            s._cache = base._cache
            for lg, docs in teste.items():
                r = [s.detectar(d) for d in docs]
                p = np.array([x.p for x in r])
                z = np.array([x.z for x in r])
                out[f"{chave}|{nome}|{lg}"] = dict(fpr1=float((p <= 0.01).mean()), fpr5=float((p <= 0.05).mean()),
                                                   z=float(z.mean()))
                print(f"{chave:22s} {nome:10s} {lg} falso+ a 1%={(p <= 0.01).mean():.3f} a 5%={(p <= 0.05).mean():.3f}"
                      f" z médio={z.mean():5.2f}", flush=True)
    json.dump(out, open(resultado("exp_selo_fpr.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
