"""Sinais de modelo de linguagem, sem chave: Binoculars, Fast-DetectGPT e estatísticas por token.

Par observador/executor da mesma família (mesmo tokenizador):
  observador = modelo base, executor = versão instruct.
Binoculars (Hans et al. 2024): log-PPL do executor / perplexidade cruzada observador→executor.
Valor baixo = texto que um LLM acharia "natural de escrever" = provável IA.
Fast-DetectGPT (Bao et al. 2024), forma analítica: curvatura da log-prob do texto
em relação ao que o modelo amostraria em cada posição.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PARES = {
    "qwen0.5": ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen1.5": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
}


@dataclass
class Sinais:
    n_tokens: int
    binoculars: float  # menor = mais IA
    fastdetect: float  # maior = mais IA
    logp_obs: float  # log-prob média por token (observador)
    logp_exe: float
    entropia_obs: float
    rank1: float  # fração de tokens que são o top-1 do observador
    rank10: float
    logp_std: float  # variação da surpresa (burstiness por token)

    def vetor(self) -> list[float]:
        return [self.binoculars, self.fastdetect, self.logp_obs, self.logp_exe,
                self.entropia_obs, self.rank1, self.rank10, self.logp_std]

    NOMES = ["binoculars", "fastdetect", "logp_obs", "logp_exe", "entropia_obs", "rank1", "rank10", "logp_std"]


class ParLM:
    def __init__(self, par: str = "qwen0.5", max_tokens: int = 512, threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        obs, exe = PARES[par]
        self.tok = AutoTokenizer.from_pretrained(obs)
        self.obs = AutoModelForCausalLM.from_pretrained(obs, dtype=torch.bfloat16).eval()
        self.exe = AutoModelForCausalLM.from_pretrained(exe, dtype=torch.bfloat16).eval()
        self.max_tokens = max_tokens

    @torch.inference_mode()
    def sinais(self, texto: str) -> Sinais:
        """Um texto por vez: o vocabulário do Qwen tem 152 mil tokens, e um lote
        de logits em float32 passa de 1 GB por texto."""
        ids = self.tok(texto, return_tensors="pt", truncation=True,
                       max_length=self.max_tokens)["input_ids"]
        n = ids.shape[1] - 1
        if n < 8:
            return Sinais(n, *([float("nan")] * 8))
        alvo = ids[0, 1:]
        lo = self.obs(input_ids=ids).logits[0, :-1].float()
        tok_lo = lo.gather(-1, alvo[:, None])[:, 0]
        maior = (lo > tok_lo[:, None]).sum(-1)
        lp_o = F.log_softmax(lo, -1)
        del lo
        p_o = lp_o.exp()
        tok_lp_o = lp_o.gather(-1, alvo[:, None])[:, 0]
        ent_o = -(p_o * lp_o).sum(-1)
        del lp_o
        lp_e = F.log_softmax(self.exe(input_ids=ids).logits[0, :-1].float(), -1)
        tok_lp_e = lp_e.gather(-1, alvo[:, None])[:, 0]
        mu = (p_o * lp_e).sum(-1)  # = -entropia cruzada observador→executor
        var = (p_o * lp_e.pow(2)).sum(-1) - mu.pow(2)
        del lp_e, p_o
        ppl_e = -float(tok_lp_e.mean())
        xppl = -float(mu.mean())
        fd = float((tok_lp_e.sum() - mu.sum()) / torch.sqrt(var.sum().clamp_min(1e-6)))
        return Sinais(
            n_tokens=n, binoculars=ppl_e / max(xppl, 1e-6), fastdetect=fd,
            logp_obs=float(tok_lp_o.mean()), logp_exe=-ppl_e,
            entropia_obs=float(ent_o.mean()),
            rank1=float((maior == 0).float().mean()), rank10=float((maior < 10).float().mean()),
            logp_std=float(tok_lp_o.std()),
        )

    def sinais_lote(self, textos: list[str]) -> list[Sinais]:
        return [self.sinais(t) for t in textos]
