"""Monta o conjunto de janelas rotuladas (humano / IA / IA atacada) em pt-BR e inglês.

Saída: JSONL com uma janela de ~200 palavras por linha:
  id, lingua, rotulo (0 humano, 1 IA), fonte, ataque, familia, texto
`familia` agrupa o que não pode ser separado entre treino e teste (o mesmo artigo,
as reescritas dele, o mesmo título no RAID).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import pandas as pd
import pyarrow.parquet as pq

from config import BABEL, DADOS, RAW, WASEMU
from textos import janelas, limpar_markdown, limpar_simples

ECON = ["viniciustorres", "samuelpessoa", "alexandreschwartsman", "laura-carvalho", "antoniodelfim",
        "marcostroyjo", "mercadoaberto", "vaivem", "carodinheiro", "marciadessen", "raquellandim",
        "celso-rocha-de-barros", "helioschwartsman", "nelsondesa"]
GERAIS = ["bernardomellofranco", "clovisrossi", "ruycastro", "eliogaspari", "janiodefreitas",
          "marilizpereirajorge", "joaopereiracoutinho", "marcelocoelho", "claudiacollucci",
          "leaoserva", "luizfelipeponde", "antonioprata", "vladimirsafatle", "ronaldolemos",
          "contardocalligaris", "viniciusmota", "demetriomagnoli", "matiasspektor", "andresinger",
          "igorgielow", "patriciacamposmello", "lucianacoelho"]
TRADUZIDAS = ["paulkrugman", "martinwolf", "lucy-kellaway"]

BABEL_DIRS = {
    "out_haiku20_prosa": ("haiku", "babel_prosa"), "out_haiku20_forte": ("haiku", "babel_forte"),
    "out_haiku20_nomes": ("haiku", "babel_nomes"), "out_haiku20": ("haiku", "babel_padrao"),
    "out_haiku4": ("haiku", "babel_v04"), "out_haiku4_marc": ("haiku", "babel_marcadores"),
    "out_haiku4_semsig": ("haiku", "babel_semsig"), "out_haiku4_wm": ("haiku", "babel_marca_propria"),
    "out_gemini4": ("gemini", "babel_v04"), "out_gemini4_marc": ("gemini", "babel_marcadores"),
    "out_gemini4_semsig": ("gemini", "babel_semsig"),
}


def _jan(txt, n, maxj, rng):
    js = janelas(txt, n, minimo=int(n * 0.6))
    if len(js) > maxj:
        js = rng.sample(js, maxj)
    return js


def pt_humano(rng, n_jan, por_cat):
    df = pq.read_table(os.path.join(RAW, "folha.parquet"),
                       columns=["title", "text", "category", "subcategory", "link"]).to_pandas()
    df = df[df.text.str.split().str.len() >= n_jan + 40]
    grupos = {
        "folha_colunas_econ": df[df.subcategory.isin(ECON)],
        "folha_mercado": df[(df.category == "mercado")],
        "folha_colunas_gerais": df[df.subcategory.isin(GERAIS)],
        "folha_traduzidas": df[df.subcategory.isin(TRADUZIDAS)],
    }
    out = []
    for fonte, g in grupos.items():
        g = g.sample(min(len(g), por_cat[fonte]), random_state=rng.randint(0, 10**6))
        for _, r in g.iterrows():
            t = limpar_simples(r.text)
            for j in _jan(t, n_jan, 1, rng):
                out.append(dict(lingua="pt", rotulo=0, fonte=fonte, ataque="nenhum",
                                familia="folha:" + r.link, texto=j))
    return out


def _fam_wasemu(caminho):
    rel = os.path.relpath(caminho, WASEMU)
    return "wasemu:" + rel[:-3].replace("/", "_")


def pt_ia(rng, n_jan):
    out = []
    for f in sorted(glob.glob(os.path.join(WASEMU, "**", "*.md"), recursive=True)):
        if os.path.basename(f) in ("README.md", "TOPICOS.md") or "/codigo/" in f:
            continue
        t = limpar_markdown(open(f, encoding="utf-8").read())
        for j in _jan(t, n_jan, 4, rng):
            out.append(dict(lingua="pt", rotulo=1, fonte="wasemu_claude", ataque="nenhum",
                            familia=_fam_wasemu(f), texto=j))
    base = BABEL
    for d, (redator, ataque) in BABEL_DIRS.items():
        for f in sorted(glob.glob(f"{base}/{d}/*.md")):
            fam = "wasemu:" + os.path.basename(f)[:-3]
            t = limpar_markdown(open(f, encoding="utf-8").read())
            for j in _jan(t, n_jan, 4, rng):
                out.append(dict(lingua="pt", rotulo=1, fonte=f"babel_{redator}", ataque=ataque,
                                familia=fam, texto=j))
    return out


def en_raid(rng, n_jan, n_humano, n_por_celula, modelos, ataques):
    df = pq.read_table(os.path.join(RAW, "raid_news.parquet")).to_pandas()
    out = []
    h = df[(df.model == "human") & (df.attack == "none")]
    h = h.sample(min(len(h), n_humano), random_state=1)
    for _, r in h.iterrows():
        for j in _jan(limpar_simples(r.generation), n_jan, 1, rng):
            out.append(dict(lingua="en", rotulo=0, fonte="raid_human", ataque="nenhum",
                            familia="raid:" + str(r.title), texto=j))
    for m in modelos:
        for a in ataques:
            g = df[(df.model == m) & (df.attack == a) & (df.decoding == "sampling") & (df.repetition_penalty == "no")]
            g = g.sample(min(len(g), n_por_celula), random_state=rng.randint(0, 10**6))
            for _, r in g.iterrows():
                for j in _jan(limpar_simples(r.generation), n_jan, 1, rng):
                    out.append(dict(lingua="en", rotulo=1, fonte=f"raid_{m}", ataque=a,
                                    familia="raid:" + str(r.title), texto=j))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=DADOS)
    ap.add_argument("--palavras", type=int, default=200)
    ap.add_argument("--semente", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.semente)
    linhas = []
    linhas += pt_humano(rng, a.palavras, {"folha_colunas_econ": 800, "folha_mercado": 500,
                                          "folha_colunas_gerais": 300, "folha_traduzidas": 120})
    linhas += pt_ia(rng, a.palavras)
    linhas += en_raid(rng, a.palavras, 1500, 80,
                      ["gpt4", "chatgpt", "llama-chat", "mistral-chat", "cohere-chat"],
                      ["none", "paraphrase", "synonym", "article_deletion", "perplexity_misspelling"])
    for i, l in enumerate(linhas):
        l["id"] = i
    with open(a.o, "w", encoding="utf-8") as fh:
        for l in linhas:
            fh.write(json.dumps(l, ensure_ascii=False) + "\n")
    d = pd.DataFrame(linhas)
    print(d.groupby(["lingua", "rotulo", "fonte", "ataque"]).size().to_string())
    print("famílias:", d.familia.nunique(), "janelas:", len(d))


if __name__ == "__main__":
    main()
