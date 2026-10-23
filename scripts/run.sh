#!/usr/bin/env bash
# Usage: scripts/run.sh <gpus e.g. 4 or 0,1,2,3,4> <cmd...>   (runs inside the training container)
set -eu
GPUS="$1"; shift
exec podman run --rm --replace --name "${CNAME:-lm24h}" \
  --device nvidia.com/gpu=all --network=host --ipc=host \
  --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e PYTHONPATH=/work/src:/lm24h/pylib -e HF_HOME=/lm24h/hf \
  -e TORCHINDUCTOR_CACHE_DIR=/lm24h/inductor -e TRITON_CACHE_DIR=/lm24h/triton \
  -v /home/dkp/Projects/lm24h:/work -v /home/dkp/lm24h:/lm24h -w /work \
  --entrypoint /opt/venv/bin/python localhost/glm53-engine:1.8.0 "$@"
