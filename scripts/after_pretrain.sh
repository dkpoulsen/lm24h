#!/usr/bin/env bash
# Unattended: wait for pretrain -> SFT -> export base+chat HF dirs -> sample generations.  Log: /home/dkp/lm24h/runs/pipeline.log
cd "$(dirname "$0")/.."; R=/home/dkp/lm24h/runs; L=$R/pipeline.log; log(){ echo "[$(date)] $*" >> $L; }
while [ ! -e $R/lm24h-500m/ckpt/final.pt ]; do
  pgrep -f launch_stage.sh >/dev/null || { log "pretrain launcher gone and no final.pt; abort"; exit 1; }; sleep 120; done
log "pretrain done"; sleep 20
CNAME=lm24h-gen scripts/run.sh 0 -m lm24h.export_hf /lm24h/runs/lm24h-500m/ckpt/final.pt /lm24h/export/base >> $L 2>&1
mkdir -p $R/lm24h-500m-sft; scripts/launch_stage.sh configs/sft_500m.toml $R/lm24h-500m-sft/train.log
[ -e $R/lm24h-500m-sft/ckpt/final.pt ] || { log "SFT no final.pt"; exit 1; }
log "SFT done"
CNAME=lm24h-gen scripts/run.sh 0 -m lm24h.export_hf /lm24h/runs/lm24h-500m-sft/ckpt/final.pt /lm24h/export/chat --chat >> $L 2>&1
CNAME=lm24h-gen scripts/run.sh 0 -m lm24h.generate /lm24h/runs/lm24h-500m/ckpt/final.pt > $R/samples_base.txt 2>&1
CNAME=lm24h-gen scripts/run.sh 0 -m lm24h.generate /lm24h/runs/lm24h-500m-sft/ckpt/final.pt --chat > $R/samples_chat.txt 2>&1
log "ALL DONE"
