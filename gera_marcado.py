"""Marca na hora de gerar (o caso "nós somos o fornecedor").

Frase a frase, o modelo sorteia N continuações; o texto marcado fica com a de nota maior
pela chave, e o de controle com a primeira (geração normal). Como as N continuações dizem
coisas diferentes, a marca fica no que a frase diz, não nas palavras, e uma paráfrase fiel
não tem como tirá-la sem mudar o conteúdo.

Depois, ataque de paráfrase com o mesmo modelo local (trechos de 3 frases reescritos)
e a detecção nos quatro textos: marcado, controle, marcado parafraseado, controle parafraseado.

  python gera_marcado.py --n 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import resultado
from selo import Selo
from textos import frases

MODELO = "Qwen/Qwen2.5-1.5B-Instruct"

TEMAS_PT = [
    "o envelhecimento da população brasileira e a Previdência",
    "a geada de 2021 e o preço do café",
    "o ciclo do boi e o preço da carne",
    "a frota de veículos e o consumo de combustíveis no Brasil",
    "o preço do arroz e do feijão na mesa do brasileiro",
    "o petróleo como principal produto de exportação do Brasil",
    "o consumo de cimento como termômetro da construção civil",
    "etanol ou gasolina: quando compensa abastecer com cada um",
    "a inadimplência das famílias e os juros do cartão de crédito",
    "o Pix e a concorrência entre bancos",
    "o desmatamento e a expansão da fronteira agrícola",
    "o mercado de trabalho informal depois da pandemia",
]
TEMAS_EN = [
    "the aging of the population and pension systems",
    "why coffee prices rise after a frost in Brazil",
    "electric cars and the future of gasoline demand",
    "credit card debt and high interest rates",
    "the role of cement consumption as an economic indicator",
    "how instant payments changed retail banking",
    "deforestation and agricultural expansion",
    "informal work after the pandemic",
]

PEDIDO = {
    "pt": ("Escreva um artigo de opinião em português do Brasil, em prosa corrida, sem títulos, "
           "sem listas e sem markdown, com cerca de {k} frases, sobre: {tema}."),
    "en": ("Write an opinion article in English, in plain prose, with no headings, no lists and no "
           "markdown, about {k} sentences long, on: {tema}."),
}
PARAFRASE = {
    "pt": ("Reescreva o trecho abaixo com outras palavras e outra estrutura de frase, mantendo todos "
           "os fatos e números. Responda só com o trecho reescrito.\n\n{t}"),
    "en": ("Rewrite the passage below in different words and with different sentence structure, "
           "keeping every fact and number. Reply with the rewritten passage only.\n\n{t}"),
}


class Gerador:
    def __init__(self, nome=MODELO, threads=2):
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(nome)
        self.tok.padding_side = "left"
        self.m = AutoModelForCausalLM.from_pretrained(nome, dtype=torch.bfloat16).eval()

    def _prompt(self, pedido: str, prefixo: str = "") -> str:
        msgs = [{"role": "user", "content": pedido}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefixo

    @torch.inference_mode()
    def continuacoes(self, pedido: str, prefixo: str, n: int, max_novos=56, temp=0.9, sem_fim=False) -> list[str]:
        ids = self.tok(self._prompt(pedido, prefixo), return_tensors="pt")
        extra = {}
        if sem_fim:  # o texto ainda está curto: não deixa o modelo encerrar
            fins = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|endoftext|>")}
            extra["suppress_tokens"] = [t for t in fins if t is not None and t >= 0]
        out = self.m.generate(**ids, do_sample=True, temperature=temp, top_p=0.95, max_new_tokens=max_novos,
                              num_return_sequences=n, pad_token_id=self.tok.eos_token_id, **extra)
        txts = self.tok.batch_decode(out[:, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        res = []
        for t in txts:
            t = t.strip()
            fs = frases(t + " X") if t else []  # o "X" força o corte depois da última frase completa
            if t and len(fs) >= 2:
                res.append(fs[0])
            elif t and t[-1:] in ".!?":
                res.append(t)
            else:
                res.append("")  # fim do texto ou frase cortada
        return res

    @torch.inference_mode()
    def responder(self, pedidos: list[str], max_novos=400, temp=0.7) -> list[str]:
        ps = [self._prompt(p) for p in pedidos]
        ids = self.tok(ps, return_tensors="pt", padding=True)
        out = self.m.generate(**ids, do_sample=True, temperature=temp, top_p=0.95, max_new_tokens=max_novos,
                              pad_token_id=self.tok.eos_token_id)
        return [t.strip() for t in self.tok.batch_decode(out[:, ids["input_ids"].shape[1]:], skip_special_tokens=True)]


def gerar(g: Gerador, selo: Selo | None, lingua: str, tema: str, n: int, k: int = 18, rng=None,
          minimo: int = 0, registro: list | None = None) -> list[str]:
    """registro, se dado, recebe as candidatas de cada passo e a escolhida (para análise)."""
    pedido = PEDIDO[lingua].format(k=k, tema=tema)
    v_tema = selo.vetores([tema])[0] if selo else None
    texto: list[str] = []
    while len(texto) < k + 4:
        prefixo = " ".join(texto) + (" " if texto else "")
        for _tentativa in range(3):  # frase cortada ou vazia: tenta de novo antes de desistir
            cands = g.continuacoes(pedido, prefixo, n if selo else 1, sem_fim=len(texto) < minimo)
            cands = [c for c in cands if len(c.split()) >= 4 and c not in texto]
            if cands or len(texto) >= minimo:
                break
        if not cands:
            break
        if selo is None:
            texto.append(cands[0])
            continue
        V = selo.vetores(cands)
        ja = selo.vetores(texto) if texto else np.zeros((0, selo.d), np.float32)
        # média do documento ainda desconhecida: usa o que já foi escrito + o tema (peso de 2 frases)
        media = (ja.sum(0) + 2 * v_tema + V.mean(0)) / (len(ja) + 3)
        q = selo.nota(selo.residuos(V, media))
        texto.append(cands[int(np.argmax(q))])
        if registro is not None:
            registro.append({"candidatas": cands, "escolhida": int(np.argmax(q))})
    return texto


def parafrasear(g: Gerador, fs: list[str], lingua: str, bloco=3) -> list[str]:
    trechos = [" ".join(fs[i:i + bloco]) for i in range(0, len(fs), bloco)]
    out = []
    for i in range(0, len(trechos), 4):
        out += g.responder([PARAFRASE[lingua].format(t=t) for t in trechos[i:i + 4]])
    return frases(" ".join(o.replace("\n", " ") for o in out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--chave", default="chave-de-teste-luiz")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--minimo", type=int, default=0, help="frases mínimas antes de deixar o modelo encerrar")
    ap.add_argument("--saida", default=resultado("gera_marcado.json"))
    a = ap.parse_args()
    selo = Selo(a.chave)
    g = Gerador(threads=a.threads)
    feitos = json.load(open(a.saida)) if os.path.exists(a.saida) else {}
    rng = random.Random(0)
    temas = [("pt", t) for t in TEMAS_PT] + [("en", t) for t in TEMAS_EN]
    for lingua, tema in temas:
        if tema in feitos:
            continue
        t0 = time.time()
        passos: list = []
        marcado = gerar(g, selo, lingua, tema, a.n, rng=rng, minimo=a.minimo, registro=passos)
        controle = gerar(g, None, lingua, tema, a.n, rng=rng, minimo=a.minimo)
        r = {"lingua": lingua, "marcado": marcado, "controle": controle, "passos": passos,
             "marcado_parafrase": parafrasear(g, marcado, lingua),
             "controle_parafrase": parafrasear(g, controle, lingua)}
        r["det"] = {k: vars(selo.detectar(" ".join(r[k]))) for k in
                    ("marcado", "controle", "marcado_parafrase", "controle_parafrase")}
        for k in r["det"]:
            r["det"][k].pop("notas", None)
        feitos[tema] = r
        json.dump(feitos, open(a.saida, "w"), ensure_ascii=False, indent=1)
        print(f"{lingua} {tema[:40]:40s} {round(time.time() - t0)}s frases={len(marcado)}/{len(controle)} "
              + " ".join(f"{k}: z={v['z']:.2f} p={v['p']:.3f}" for k, v in r["det"].items()), flush=True)


if __name__ == "__main__":
    main()
