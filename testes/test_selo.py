"""Testes rápidos da marca, com um modelo falso (sem baixar nada).

  python -m unittest discover -s testes
"""
import hashlib
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from selo import Selo, ajustar_branqueador  # noqa: E402
from textos import frases  # noqa: E402


class ModeloFalso:
    """Vetor pseudoaleatório por texto, com uma direção comum forte (imita o tema)."""

    def __init__(self, d=48):
        self.d = d
        self.tema = np.random.default_rng(7).standard_normal(d)

    def get_sentence_embedding_dimension(self):
        return self.d

    def encode(self, textos, **_):
        out = []
        for t in textos:
            s = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            v = np.random.default_rng(s).standard_normal(self.d) + 3 * self.tema
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


def texto(n, semente):
    rng = np.random.default_rng(semente)
    return [f"Frase número {i} sobre o assunto {rng.integers(1e9)}." for i in range(n)]


class TestSelo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = ModeloFalso()

    def selo(self, chave="k", **kw):
        return Selo(chave, modelo=self.m, m_eixos=16, n_iscas=199, branqueador=None, **kw)

    def test_frases(self):
        self.assertEqual(frases("O Sr. Silva chegou. Depois saiu! E voltou?"),
                         ["O Sr. Silva chegou.", "Depois saiu!", "E voltou?"])

    def test_p_valor_sob_nula_e_uniforme(self):
        # sem marca, p deve cair abaixo de 5% em ~5% dos textos (chaves diferentes)
        ps = [self.selo(f"chave-{i}").detectar(" ".join(texto(20, i))).p for i in range(60)]
        self.assertLess(np.mean(np.array(ps) <= 0.05), 0.2)
        self.assertGreater(np.mean(ps), 0.3)

    def test_marcar_e_detectar(self):
        s = self.selo()
        pools = [[f"Versão {j} da frase {i}." for j in range(8)] for i in range(30)]
        esc = s.escolher(pools, piso_sentido=-1.0)
        marcado = " ".join(p[k] for p, k in zip(pools, esc))
        controle = " ".join(p[0] for p in pools)
        self.assertLess(s.detectar(marcado).p, 0.01)
        self.assertGreater(s.detectar(marcado).z, s.detectar(controle).z)
        # outra chave não vê a marca
        self.assertGreater(self.selo("outra").detectar(marcado).p, 0.01)

    def test_ordem_nao_importa(self):
        # na escala de frases a ordem não muda nada (a de pares, sim, por construção)
        s = self.selo()
        fs = texto(15, 3)
        a = s.detectar(" ".join(fs), escalas=("frases",)).z
        b = s.detectar(" ".join(fs[::-1]), escalas=("frases",)).z
        self.assertAlmostEqual(a, b, places=4)

    def test_branqueador(self):
        docs = [texto(12, i) for i in range(40)]
        W = ajustar_branqueador(docs, self.m)
        self.assertEqual(W.shape, (self.m.d, self.m.d))
        self.assertTrue(np.allclose(W, W.T, atol=1e-6))
        s = Selo("k", modelo=self.m, m_eixos=16, n_iscas=99, branqueador=W)
        self.assertTrue(0 < s.detectar(" ".join(texto(20, 99))).p <= 1)

    def test_cache(self):
        s = self.selo()
        v1 = s.vetores(["a b c d.", "e f g h."])
        v2 = s.vetores(["e f g h.", "a b c d."])
        self.assertTrue(np.allclose(v1[0], v2[1]))


if __name__ == "__main__":
    unittest.main()
