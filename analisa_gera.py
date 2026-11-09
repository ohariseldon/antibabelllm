"""Relê os textos de gera_marcado.py com branqueamento + calibração da chave."""
from __future__ import annotations

import json
import sys

import numpy as np

from config import resultado
from selo import Selo

TIPOS = ("marcado", "controle", "marcado_parafrase", "controle_parafrase")


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else resultado("gera_marcado.json")
    d = json.load(open(caminho))
    s = Selo("chave-de-teste-luiz")
    s.carregar_calibracao(resultado("calib_chave_teste.npz"))
    textos = {t: [" ".join(r[t]) for r in d.values()] for t in TIPOS}
    print(f"{len(d)} temas; frases médias marcado={np.mean([len(r['marcado']) for r in d.values()]):.1f}")
    for t in TIPOS:
        rs = [s.detectar(x) for x in textos[t]]
        z = np.array([r.z for r in rs]); p = np.array([r.p for r in rs])
        print(f"{t:20s} z médio={z.mean():5.2f}  z={np.round(z, 1).tolist()}  p≤1%={np.mean(p <= .01):.2f} p≤5%={np.mean(p <= .05):.2f}")


if __name__ == "__main__":
    main()
