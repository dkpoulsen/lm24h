#!/usr/bin/env bash
# Usage: scripts/launch_stage.sh <config.toml> <logfile>   -- restarts from the latest checkpoint on crash (exit!=0, not SIGTERM stop)
CFG="$1"; LOG="$2"; cd "$(dirname "$0")/.."
while true; do
  scripts/torchrun.sh 0,1,2,3,4 -m lm24h.train --config "$CFG" >> "$LOG" 2>&1; rc=$?
  echo "[launcher] exit rc=$rc at $(date)" >> "$LOG"
  [ $rc -eq 0 ] && break; [ -f /home/dkp/lm24h/STOP ] && break
  sleep 30
done
