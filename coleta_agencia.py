"""Negativos difíceis em pt: notícias de economia da Agência Brasil (licença CC BY 3.0 BR),
do mesmo período e dos mesmos temas dos artigos de IA (2025-2026). Sem isto, o detector
poderia estar só reconhecendo o assunto (Folha 2015-2017 contra textos de 2026).

  python coleta_agencia.py --paginas 40
Uma requisição por segundo.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import os
import time
import urllib.request

from config import RAW

BASE = "https://agenciabrasil.ebc.com.br"
UA = {"User-Agent": "Mozilla/5.0 (pesquisa antibabelllm)"}


def baixar(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", "replace")


def texto_da_noticia(h: str) -> str:
    i = h.find('class="conteudo-noticia"')
    if i < 0:
        return ""
    corpo = h[i:]
    fim = corpo.find("</article>")
    corpo = corpo[: fim if fim > 0 else 200_000]
    paras = []
    for p in re.findall(r"<p[^>]*>(.*?)</p>", corpo, re.S):
        t = html.unescape(re.sub(r"<[^>]+>", "", p)).strip()
        if len(t.split()) < 8 or t.lower().startswith(("leia mais", "confira", "* ", "matéria ampliada")):
            continue
        paras.append(re.sub(r"\s+", " ", t))
    return "\n".join(paras)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secao", default="economia")
    ap.add_argument("--paginas", type=int, default=40)
    ap.add_argument("--saida", default=os.path.join(RAW, "agencia_brasil.jsonl"))
    a = ap.parse_args()
    links = []
    for p in range(a.paginas):
        try:
            h = baixar(f"{BASE}/{a.secao}?page={p}")
        except Exception as e:  # noqa: BLE001
            print("página", p, e, flush=True)
            continue
        for l in re.findall(rf'href="(/{a.secao}/noticia/20(?:25|26)-\d\d/[^"]+)"', h):
            if l not in links:
                links.append(l)
        time.sleep(1)
    print(len(links), "links", flush=True)
    n = 0
    with open(a.saida, "w", encoding="utf-8") as f:
        for l in links:
            try:
                t = texto_da_noticia(baixar(BASE + l))
            except Exception as e:  # noqa: BLE001
                print(l, e, flush=True)
                continue
            if len(t.split()) >= 220:
                f.write(json.dumps({"url": BASE + l, "texto": t}, ensure_ascii=False) + "\n")
                n += 1
            time.sleep(1)
    print(n, "notícias salvas", flush=True)


if __name__ == "__main__":
    main()
