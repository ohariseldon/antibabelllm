"""Extrai os sinais de cada janela e grava em .npy ao lado do JSONL.

  python extrair.py estilo   (segundos)
  python extrair.py emb      (vetor de sentido multilíngue, dígitos mascarados)
  python extrair.py lm       (Binoculars e cia. com o par Qwen2.5; o mais lento)
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np

from config import DADOS


def carregar(caminho=DADOS):
    return [json.loads(l) for l in open(caminho, encoding="utf-8")]


def mascarar_digitos(t: str) -> str:
    return re.sub(r"\d", "0", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tipo", choices=["estilo", "emb", "lm"])
    ap.add_argument("--dados", default=DADOS)
    ap.add_argument("--par", default="qwen0.5")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    linhas = carregar(a.dados)
    base = a.dados.rsplit(".", 1)[0]
    t0 = time.time()
    if a.tipo == "estilo":
        import estilo
        pt = np.array([estilo.vetor(l["texto"], "pt") if l["lingua"] == "pt" else [np.nan] * len(estilo.nomes("pt"))
                       for l in linhas], dtype=np.float32)
        en = np.array([estilo.vetor(l["texto"], "en") if l["lingua"] == "en" else [np.nan] * len(estilo.nomes("en"))
                       for l in linhas], dtype=np.float32)
        np.save(base + ".estilo_pt.npy", pt)
        np.save(base + ".estilo_en.npy", en)
    elif a.tipo == "emb":
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(a.threads)
        m = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
        m.max_seq_length = 320
        X = m.encode(["passage: " + mascarar_digitos(l["texto"]) for l in linhas], batch_size=16,
                     normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        np.save(base + ".emb.npy", X.astype(np.float32))
    else:
        from lm import ParLM
        m = ParLM(a.par, max_tokens=256, threads=a.threads)
        out = np.full((len(linhas), 9), np.nan, dtype=np.float32)
        for i, l in enumerate(linhas):
            s = m.sinais(l["texto"])
            out[i] = [s.n_tokens] + s.vetor()
            if i % 200 == 0:
                print(i, round(time.time() - t0), flush=True)
                np.save(base + f".lm_{a.par}.npy", out)
        np.save(base + f".lm_{a.par}.npy", out)
    print(a.tipo, "ok", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
