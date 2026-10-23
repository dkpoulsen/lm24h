#!/usr/bin/env bash
# Usage: scripts/torchrun.sh <gpus csv> <script.py> [args]  -> torchrun with nproc = #gpus
GPUS="$1"; shift; N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
exec "$(dirname "$0")/run.sh" "$GPUS" -m torch.distributed.run --standalone --nproc_per_node=$N "$@"
