"""Limpeza de texto, divisão em frases e janelas. Igual para todas as fontes.

Tudo que é formato (markdown, links, tabelas, linha de assinatura) sai antes de
qualquer medida, para o detector não aprender "markdown = IA".
"""
from __future__ import annotations

import re
import unicodedata

_INVIS = re.compile("[​-‏‪-‮⁠-⁤﻿­]")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_CODE = re.compile(r"```.*?```", re.S)
_HTML = re.compile(r"<[^>]+>")
_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_ITAL = re.compile(r"(?<![\w*])[*_]([^*_\n]+)[*_](?![\w*])")


def limpar_markdown(md: str) -> str:
    """Markdown -> só os parágrafos de prosa, um por linha."""
    md = unicodedata.normalize("NFC", md)
    md = _INVIS.sub("", md)
    md = _CODE.sub("\n", md)
    md = _HTML.sub(" ", md)
    paras: list[str] = []
    for bloco in re.split(r"\n\s*\n", md):
        linhas = [l.strip() for l in bloco.strip().splitlines() if l.strip()]
        if not linhas:
            continue
        if all(l.startswith("|") for l in linhas):  # tabela
            continue
        if linhas[0].startswith("#"):  # título ou intertítulo
            linhas = linhas[1:]
            if not linhas:
                continue
        txt = []
        for l in linhas:
            l = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", l)  # item de lista vira frase
            l = re.sub(r"^>\s?", "", l)
            txt.append(l)
        p = " ".join(txt)
        p = _LINK.sub(r"\1", p)
        p = _URL.sub("", p)
        p = _BOLD.sub(r"\2", p)
        p = _ITAL.sub(r"\1", p)
        p = p.replace("`", "")
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        # linha de assinatura / metadados curtos: "wasemu · Estudo · ... · leitura de 9 min"
        if p.count("·") >= 2 and len(p.split()) < 25:
            continue
        if len(p.split()) < 4 and not p.endswith((".", "?", "!")):
            continue
        paras.append(p)
    return "\n".join(paras)


def limpar_simples(txt: str) -> str:
    txt = unicodedata.normalize("NFC", txt)
    txt = _INVIS.sub("", txt)
    txt = _URL.sub("", txt)
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n+|\s{2,}", txt)]
    return "\n".join(p for p in paras if p)


_ABREV = re.compile(
    r"(?:Sr|Sra|Dr|Dra|Prof|Profa|etc|pp?|ex|nº|art|Mr|Mrs|Ms|vs|e\.g|i\.e|U\.S|Inc|Ltd|Jr|St|No|Fig|cap|vol|[A-Z])\.$"
)
_CORTE = re.compile(r"(?<=[.!?…])[\"'”’»)]*\s+(?=[\"'“‘«(]?[A-ZÁÉÍÓÚÂÊÔÃÕÇÀ])")


def frases(texto: str) -> list[str]:
    """Divisão simples em frases (pt e en). Não quebra em 8,09% nem em 3.5."""
    out: list[str] = []
    for par in texto.split("\n"):
        par = par.strip()
        if not par:
            continue
        buf = ""
        for p in _CORTE.split(par):
            buf = f"{buf} {p}".strip() if buf else p.strip()
            if _ABREV.search(buf.rsplit(" ", 1)[-1]):
                continue  # o ponto era de abreviação: junta com a próxima
            out.append(buf)
            buf = ""
        if buf:
            out.append(buf)
    return [f for f in out if len(f.split()) >= 2]


def palavras(texto: str) -> int:
    return len(texto.split())


def janelas(texto: str, n_palavras: int = 250, minimo: int = 150) -> list[str]:
    """Janelas de ~n palavras que respeitam o fim das frases."""
    fs = frases(texto)
    out, cur, k = [], [], 0
    for f in fs:
        cur.append(f)
        k += palavras(f)
        if k >= n_palavras:
            out.append(" ".join(cur))
            cur, k = [], 0
    if cur and k >= minimo:
        out.append(" ".join(cur))
    return out
