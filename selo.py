"""Marca d'água semântica de frase, sem ordem, com p-valor exato.

Ideia em quatro linhas:
  1. Cada frase vira um vetor de sentido multilíngue e = E(s), normalizado.
  2. Tira-se a média do próprio documento: r = e - média(e). Sobra o que cada frase
     diz de diferente das outras (o tema comum sai, e com ele a correlação entre frases).
  3. A chave define eixos sorteados g_1..g_m, metade "positivos" e metade "negativos".
     A nota da frase é q(r) = Σ s_j <r, g_j>² / (m ||r||²): alta quando a frase varia
     ao longo dos eixos positivos da chave. É uma função suave do sentido, então uma
     paráfrase fiel mexe pouco nela, e não depende da posição da frase.
  4. Marcar = para cada frase, entre N versões com o mesmo conteúdo, ficar com a de nota
     maior. Detectar = média das notas, comparada com a de 1.000 chaves-isca. Sem marca,
     a chave verdadeira é só mais uma chave sorteada, e o p-valor (posto entre as iscas)
     é exato, mesmo com todas as frases falando do mesmo tema.

Fusões e divisões de frase: a detecção mede em três escalas (orações, frases, pares de
frases vizinhas) e usa o máximo; as iscas passam pelo mesmo máximo, então o p-valor
continua exato.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass, field

import numpy as np

from textos import frases

EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_SEMENTE_ISCAS = 20261109
BRANQUEADOR_PADRAO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados", "branqueador_minilm.npy")


def _rng_da_chave(chave: str, rotulo: str) -> np.random.Generator:
    h = hmac.new(chave.encode(), rotulo.encode(), hashlib.sha256).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _eixos(rng: np.random.Generator, d: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    g = rng.standard_normal((d, m)).astype(np.float32)
    g /= np.linalg.norm(g, axis=0, keepdims=True)
    s = np.ones(m, dtype=np.float32)
    s[rng.permutation(m)[: m // 2]] = -1.0
    return g, s


_ORACAO = re.compile(r"(?<=[,;:—–])\s+(?=(?:e|mas|porém|enquanto|que|pois|porque|and|but|while|which|because)\b)|;\s+|\s+[—–]\s+")


def oracoes(f: str) -> list[str]:
    partes = [p.strip(" ,;—–") for p in _ORACAO.split(f)]
    partes = [p for p in partes if len(p.split()) >= 4]
    return partes if len(partes) > 1 else [f]


@dataclass
class Resultado:
    z: float  # quantos desvios-padrão a chave fica acima das iscas
    p: float  # p-valor exato pelo posto entre as iscas
    escala: str  # escala que deu o máximo
    n_unidades: dict = field(default_factory=dict)
    notas: list = field(default_factory=list)  # nota por frase (escala "frases")

    @property
    def marcado(self) -> bool:
        return self.p <= 0.01


class Selo:
    def __init__(self, chave: str, embedder: str = EMBEDDER, m_eixos: int = 128, n_iscas: int = 999,
                 modelo=None, branqueador: str | np.ndarray | None = "padrao"):
        if modelo is None:
            from sentence_transformers import SentenceTransformer
            modelo = SentenceTransformer(embedder, device="cpu")
        self.modelo = modelo
        self.d = self.modelo.get_sentence_embedding_dimension()
        self._cache: dict[str, np.ndarray] = {}
        self.calib: dict = {}  # escala -> (média, desvio) do z de cada chave em texto humano
        # Branqueamento: o espaço de sentido é muito desigual (poucas direções concentram a
        # variação). Sem corrigir, cada chave tem um viés fixo que se repete em todo texto do
        # mesmo gênero: o p-valor só é exato na média das chaves, não para uma chave fixa.
        # W = C^(-1/2), com C estimada em texto humano público, iguala as direções.
        if isinstance(branqueador, str):
            caminho = BRANQUEADOR_PADRAO if branqueador == "padrao" else branqueador
            branqueador = np.load(caminho) if os.path.exists(caminho) else None
        self.W = None if branqueador is None else branqueador.astype(np.float32)
        self.m = m_eixos
        self.g, self.s = _eixos(_rng_da_chave(chave, "eixos"), self.d, m_eixos)
        rng = np.random.default_rng(_SEMENTE_ISCAS)
        self.iscas = [_eixos(rng, self.d, m_eixos) for _ in range(n_iscas)]
        # todas as chaves numa matriz só: (d, (1+R)*m)
        self._G = np.concatenate([self.g] + [g for g, _ in self.iscas], 1)
        self._S = np.stack([self.s] + [s for _, s in self.iscas])  # (1+R, m)

    # -- vetores --------------------------------------------------------------
    def vetores(self, textos: list[str]) -> np.ndarray:
        if not textos:
            return np.zeros((0, self.d), np.float32)
        novos = list(dict.fromkeys(t for t in textos if t not in self._cache))
        if len(self._cache) + len(novos) > 200_000:
            self._cache.clear()
            novos = list(dict.fromkeys(textos))
        if novos:
            V = self.modelo.encode(novos, batch_size=32, normalize_embeddings=True,
                                   show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
            self._cache.update(zip(novos, V))
        return np.stack([self._cache[t] for t in textos])

    def residuos(self, V: np.ndarray, media: np.ndarray | None = None) -> np.ndarray:
        """Tira a média do documento e, se houver, branqueia."""
        R = V - (V.mean(0, keepdims=True) if media is None else media)
        return R if self.W is None else R @ self.W

    def _notas_todas(self, R: np.ndarray) -> np.ndarray:
        """R: (n, d) resíduos -> notas (1+iscas, n). Linha 0 = chave verdadeira."""
        nr = (R * R).sum(1) + 1e-9
        P = (R @ self._G) ** 2  # (n, (1+R)*m)
        P = P.reshape(len(R), -1, self.m)  # (n, 1+R, m)
        q = np.einsum("nkm,km->kn", P, self._S) / self.m
        return q / nr[None, :] * self.d  # escala ~1 sob a hipótese nula

    def nota(self, R: np.ndarray) -> np.ndarray:
        nr = (R * R).sum(1) + 1e-9
        return ((R @ self.g) ** 2 @ self.s) / self.m / nr * self.d

    # -- marcar ---------------------------------------------------------------
    def escolher(self, candidatos: list[list[str]], media_doc: np.ndarray | None = None,
                 originais: list[str] | None = None, piso_sentido: float = 0.80) -> list[int]:
        """Para cada frase, o índice da versão com nota maior. `candidatos[i][0]` deve ser a frase
        original (ou a de referência); versões com sentido longe dela (cosseno < piso) são descartadas."""
        planos = [c for cs in candidatos for c in cs]
        V = self.vetores(planos)
        if media_doc is None:
            ref = self.vetores([cs[0] for cs in candidatos])
            media_doc = ref.mean(0)
        esc, k = [], 0
        for cs in candidatos:
            v = V[k:k + len(cs)]
            k += len(cs)
            q = self.nota(self.residuos(v, media_doc))
            ok = (v @ v[0]) >= piso_sentido
            ok[0] = True
            q[~ok] = -np.inf
            esc.append(int(np.argmax(q)))
        return esc

    # -- detectar -------------------------------------------------------------
    def zs(self, texto: str, escalas=("oracoes", "frases", "pares")):
        """z de todas as chaves (a verdadeira é a coluna 0) em cada escala."""
        fs = frases(texto)
        unidades = {}
        if "frases" in escalas:
            unidades["frases"] = fs
        if "oracoes" in escalas:
            unidades["oracoes"] = [o for f in fs for o in oracoes(f)]
        if "pares" in escalas:
            unidades["pares"] = [fs[i] + " " + fs[i + 1] for i in range(0, len(fs) - 1)]
        zs, notas_frases, ns = {}, [], {}
        for nome, us in unidades.items():
            if len(us) < 3:
                continue
            V = self.vetores(us)
            Q = self._notas_todas(self.residuos(V))  # (1+R, n)
            T = Q.mean(1)
            mu, sd = T[1:].mean(), T[1:].std() + 1e-9
            zs[nome] = (T - mu) / sd
            ns[nome] = len(us)
            if nome == "frases":
                notas_frases = Q[0].round(3).tolist()
        return zs, ns, notas_frases

    def calibrar(self, docs: list[str], escalas=("oracoes", "frases", "pares")) -> dict:
        """Calibração por chave em texto humano de referência: cada chave (a verdadeira e as
        iscas, todas pelo mesmo caminho) ganha média e desvio do próprio z. Tira o viés fixo
        que uma chave tem num gênero de texto; como iscas e chave passam pela mesma correção,
        o p-valor pelo posto continua exato."""
        acc: dict[str, list[np.ndarray]] = {}
        for d in docs:
            z, _, _ = self.zs(d, escalas)
            for k, v in z.items():
                acc.setdefault(k, []).append(v)
        self.calib = {k: (np.mean(v, 0), np.std(v, 0) + 1e-6) for k, v in acc.items() if len(v) >= 30}
        return self.calib

    def salvar_calibracao(self, caminho: str):
        np.savez(caminho, **{f"{k}_mu": m for k, (m, _) in self.calib.items()},
                 **{f"{k}_sd": s for k, (_, s) in self.calib.items()})

    def carregar_calibracao(self, caminho: str):
        z = np.load(caminho)
        self.calib = {k[:-3]: (z[k], z[k[:-3] + "_sd"]) for k in z.files if k.endswith("_mu")}

    def detectar(self, texto: str, escalas=("oracoes", "frases", "pares")) -> Resultado:
        zs, ns, notas_frases = self.zs(texto, escalas)
        if not zs:
            return Resultado(0.0, 1.0, "curto", ns)
        nomes = list(zs)
        Z = np.stack([zs[k] for k in nomes])  # (escalas, 1+R)
        if self.calib:
            for i, k in enumerate(nomes):
                if k in self.calib:
                    mu, sd = self.calib[k]
                    Z[i] = (Z[i] - mu) / sd
        zmax = Z.max(0)
        k = int(Z[:, 0].argmax())
        p = (1 + (zmax[1:] >= zmax[0]).sum()) / len(zmax)
        return Resultado(float(zmax[0]), float(p), nomes[k], ns, notas_frases)


def ajustar_branqueador(docs: list[list[str]], modelo, encolher: float = 0.1) -> np.ndarray:
    """docs: lista de documentos, cada um uma lista de frases (texto humano público).
    Devolve W = C^(-1/2), onde C é a covariância dos resíduos normalizados."""
    planos = [f for d in docs for f in d]
    V = modelo.encode(planos, batch_size=64, normalize_embeddings=True, show_progress_bar=False,
                      convert_to_numpy=True).astype(np.float64)
    d = V.shape[1]
    C, n, k = np.zeros((d, d)), 0, 0
    for doc in docs:
        v = V[k:k + len(doc)]
        k += len(doc)
        if len(v) < 3:
            continue
        r = v - v.mean(0)
        r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-9
        C += r.T @ r
        n += len(r)
    C /= n
    C = (1 - encolher) * C + encolher * np.trace(C) / d * np.eye(d)
    w, U = np.linalg.eigh(C)
    return (U / np.sqrt(w)) @ U.T * np.sqrt(np.trace(C) / d)
