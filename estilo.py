"""Sinais de estilo sem modelo, em pt e en. Baratos e difíceis de apagar só trocando palavras.

Cada valor é por 100 palavras ou por frase, para não depender do tamanho da janela.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from textos import frases

_PAL = re.compile(r"[A-Za-zÀ-ÿ]+(?:[-'’][A-Za-zÀ-ÿ]+)*")

FUNCIONAIS = {
    "pt": ("a o as os de da do das dos em na no nas nos um uma por para com sem sobre entre até "
           "que se e ou mas porém contudo entretanto todavia pois porque como quando onde enquanto "
           "já ainda também só apenas mais menos muito pouco bem mesmo não nem nunca sempre "
           "é são foi foram era eram ser está estão tem têm há havia isso isto esse essa este esta "
           "ele ela eles elas seu sua seus suas lhe nos quem qual quais cujo assim então logo "
           "depois antes agora aqui ali lá tão tanto cada todo toda todos todas outro outra").split(),
    "en": ("the a an of in on at to for with without by from about into over under between "
           "and or but yet so because although though while when where as if than that which who "
           "whom whose this these those it its they them their he she his her we our you your "
           "is are was were be been being has have had do does did not no nor never always "
           "also only just even still more most less very much many such each every all any "
           "some other another then thus therefore however here there now").split(),
}

# Fórmulas que LLMs repetem; cada lista é curta de propósito (só o que aparece muito).
FORMULAS = {
    "pt": [
        r"\bnão (?:é|são|foi|se trata de)\b[^.;]{1,60}?[,;—–-]\s*(?:mas|e sim|e)?\s*(?:é|são|foi)\b",
        r"\b(?:vale|cabe) (?:ressaltar|destacar|lembrar|notar)\b", r"\bé importante (?:notar|destacar|lembrar)\b",
        r"\bem (?:suma|resumo|síntese)\b", r"\bno fim das contas\b", r"\balém disso\b", r"\bpor outro lado\b",
        r"\bnesse sentido\b", r"\bdesse modo\b", r"\bdessa forma\b", r"\bem outras palavras\b",
        r"\bou seja\b", r"\bisso significa\b", r"\bo que (?:mostra|revela|indica|sugere)\b",
        r"\bem última análise\b", r"\bpanorama\b", r"\bcenário\b", r"\bdesafios?\b", r"\bcrucial\b",
        r"\bfundamental\b", r"\bsignificativ[oa]s?\b", r"\bmarcante\b", r"\bnão apenas\b[^.]{1,80}\bmas também\b",
        r"\bmais do que\b", r"\b\w+, \w+ e \w+\b(?=[.;])", r"^[^:]{3,40}: [A-ZÀ-Ú]",
    ],
    "en": [
        r"\b(?:is|are|was) not\b[^.;]{1,60}?[,;—–-]\s*(?:but|it)?\s*(?:is|are|was)\b",
        r"\bit is (?:important|worth) (?:to note|noting)\b", r"\bin (?:summary|conclusion|short)\b",
        r"\bultimately\b", r"\bmoreover\b", r"\bfurthermore\b", r"\badditionally\b", r"\bon the other hand\b",
        r"\bin other words\b", r"\bthis (?:means|shows|highlights|underscores)\b", r"\bdelve\b",
        r"\btapestry\b", r"\blandscape\b", r"\bcrucial\b", r"\bpivotal\b", r"\bsignificant(?:ly)?\b",
        r"\bnotabl[ey]\b", r"\bnot only\b[^.]{1,80}\bbut also\b", r"\bhighlight(?:s|ed|ing)?\b",
        r"\bunderscore[sd]?\b", r"\bfoster(?:s|ed|ing)?\b", r"\bnavigat(?:e|es|ing)\b", r"\brobust\b",
        r"\bseamless(?:ly)?\b", r"\bkey\b", r"\boverall\b", r"\bremains?\b", r"\bcomprehensive\b",
    ],
}
_FORM = {l: [re.compile(p, re.I) for p in ps] for l, ps in FORMULAS.items()}
_INICIO_CONECTIVO = {
    "pt": re.compile(r"^(?:mas|e|já|além disso|no entanto|porém|assim|por isso|ainda assim|em|com|para|na|no|o|a)\b", re.I),
    "en": re.compile(r"^(?:but|and|yet|however|moreover|still|so|in|with|for|the|this|that)\b", re.I),
}


def nomes(lingua: str) -> list[str]:
    base = ["fr_media", "fr_dp", "fr_cv", "fr_min", "fr_max", "fr_curta", "fr_longa", "pal_media", "ttr_mattr",
            "virg", "pontovirg", "doispontos", "travessao", "parenteses", "aspas", "interrog", "exclam",
            "digitos", "maiusc", "bigramas_rep", "inicio_conectivo", "inicio_repetido", "formulas_total"]
    return base + [f"form_{i}" for i in range(len(FORMULAS[lingua]))] + [f"fw_{w}" for w in FUNCIONAIS[lingua]]


def _mattr(ws, w=50):
    if len(ws) < w:
        return len(set(ws)) / max(1, len(ws))
    vals = [len(set(ws[i:i + w])) / w for i in range(0, len(ws) - w + 1, 10)]
    return sum(vals) / len(vals)


def vetor(texto: str, lingua: str) -> list[float]:
    fs = frases(texto)
    ws = [w.lower() for w in _PAL.findall(texto)]
    n = max(1, len(ws))
    lens = [len(f.split()) for f in fs] or [n]
    mu = sum(lens) / len(lens)
    sd = math.sqrt(sum((x - mu) ** 2 for x in lens) / len(lens))
    p100 = lambda c: 100.0 * c / n
    bigr = Counter(zip(ws, ws[1:]))
    rep = sum(c - 1 for c in bigr.values() if c > 1) / max(1, len(ws) - 1)
    inic = [f.split()[0].lower() for f in fs if f.split()]
    ini_rep = (len(inic) - len(set(inic))) / max(1, len(inic))
    ini_con = sum(bool(_INICIO_CONECTIVO[lingua].match(f)) for f in fs) / max(1, len(fs))
    forms = [len(r.findall(texto)) for r in _FORM[lingua]]
    cont = Counter(ws)
    v = [mu, sd, sd / max(mu, 1e-6), min(lens), max(lens),
         sum(x < 10 for x in lens) / len(lens), sum(x > 30 for x in lens) / len(lens),
         sum(len(w) for w in ws) / n, _mattr(ws),
         p100(texto.count(",")), p100(texto.count(";")), p100(texto.count(":")),
         p100(texto.count("—") + texto.count("–") + texto.count(" - ")), p100(texto.count("(")),
         p100(texto.count('"') + texto.count("“") + texto.count("«")), p100(texto.count("?")), p100(texto.count("!")),
         p100(len(re.findall(r"\d+", texto))), p100(len(re.findall(r"\b[A-ZÀ-Ú][a-zà-ÿ]+", texto))),
         rep, ini_con, ini_rep, p100(sum(forms))]
    v += [p100(c) for c in forms]
    v += [p100(cont[w]) for w in FUNCIONAIS[lingua]]
    return v
