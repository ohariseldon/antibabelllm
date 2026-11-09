"""Onde ficam os dados (nada disto vai para o repositório). Tudo por variável de ambiente.

  ANTIBABEL_TRABALHO  pasta de trabalho (dados baixados, features, resultados)
  ANTIBABEL_WASEMU    pasta com os artigos de IA em .md (no wasemu: docs/artigos-opiniao)
  ANTIBABEL_BABEL     pasta com as saídas do babelllm (subpastas out_*)
  ANTIBABEL_ARTIGOS   pasta com os 20 artigos originais que passaram pelo babelllm
"""
import os

TRABALHO = os.environ.get("ANTIBABEL_TRABALHO", os.path.expanduser("~/antibabel_trabalho"))
RAW = os.path.join(TRABALHO, "data", "raw")
DADOS = os.path.join(TRABALHO, "data", "janelas.jsonl")
WASEMU = os.environ.get("ANTIBABEL_WASEMU", os.path.join(TRABALHO, "wasemu_artigos"))
BABEL = os.environ.get("ANTIBABEL_BABEL", os.path.join(TRABALHO, "babel-teste"))
ARTIGOS = os.environ.get("ANTIBABEL_ARTIGOS", os.path.join(BABEL, "artigos"))


def resultado(nome: str) -> str:
    os.makedirs(os.path.join(TRABALHO, "resultados"), exist_ok=True)
    return os.path.join(TRABALHO, "resultados", nome)
