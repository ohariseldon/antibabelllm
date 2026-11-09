#!/usr/bin/env bash
# Baixa os dados públicos usados na régua (nada disto vai para o repositório).
#   pt humano: notícias e colunas da Folha (2015-2017), dataset emdemor/news-of-the-brazilian-newspaper
#   en: RAID (Dugan et al. 2024), só o domínio "news" (humanos e geradores, com ataques)
# Os textos de IA em pt (artigos do wasemu e saídas do babelllm) vêm de pastas locais;
# veja montar_dados.py.
set -euo pipefail
DEST=${1:-/home/user/work/data/raw}
PY=${PYTHON:-python}
mkdir -p "$DEST"
HF=https://huggingface.co/datasets
curl -L -o "$DEST/folha.parquet" "$HF/emdemor/news-of-the-brazilian-newspaper/resolve/main/brazilian-news.parquet"
for i in 0006 0007 0008 0009; do
  curl -L -o "$DEST/raid_$i.parquet" "$HF/liamdugan/raid/resolve/refs%2Fconvert%2Fparquet/raid/partial-train/$i.parquet"
done
"$PY" - "$DEST" <<'EOF'
import sys
import pandas as pd
import pyarrow.parquet as pq
d = sys.argv[1]
df = pd.concat([pq.read_table(f"{d}/raid_{i}.parquet").to_pandas() for i in ["0006", "0007", "0008", "0009"]])
df[df.domain == "news"].to_parquet(f"{d}/raid_news.parquet")
EOF
echo "ok: $DEST"
