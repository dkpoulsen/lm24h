#!/usr/bin/env bash
# Resumable parallel download of FineWeb-Edu sample-10BT parquet shards.
cd /home/dkp/lm24h/data/raw
cat ../files.txt | xargs -P 4 -I{} sh -c 'f=$(basename {}); curl -sSL -C - --retry 10 --retry-delay 5 -o $f "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/{}" && echo done $f'
