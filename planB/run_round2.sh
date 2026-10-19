#!/usr/bin/env bash
# Chains the remaining round-2 steps so they finish unattended. Log: out/full/run_round2_log.txt
cd "$(dirname "$0")"
LOG=out/full/run_round2_log.txt
BW=data/hg38.phyloP100way.bw
FULLSIZE=9870053206
export PYTHONIOENCODING=utf-8 TF_CPP_MIN_LOG_LEVEL=2
log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

log "waiting for full_2dD training (3 seeds)"
until grep -q '"full_2dD_s2"' out/full/train_full_summary.json 2>/dev/null; do sleep 60; done
log "full_2dD done"

log "ensuring phyloP download is complete"
while [ "$(stat -c %s "$BW" 2>/dev/null || echo 0)" -lt "$FULLSIZE" ]; do
  if ! pgrep -f "phyloP100way" >/dev/null 2>&1 && ! tasklist 2>/dev/null | grep -qi curl; then
    log "resuming phyloP download at $(stat -c %s "$BW" 2>/dev/null) bytes"
    curl -sS -C - -o "$BW" https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw || sleep 30
  else
    sleep 60
  fi
done
log "phyloP download complete"

python phylop_extract.py >> "$LOG" 2>&1 || { log "phylop_extract FAILED"; exit 1; }
log "phyloP extracted; training hybrid"
python train_full.py full_2dD_phylop 3 > out/full/train_full_phylop_log.txt 2>&1 || { log "hybrid training FAILED"; exit 1; }
log "hybrid done"

log "waiting for validation-set VEP scores"
until [ -f out/vep_scores_val.csv ]; do sleep 60; done
python evaluate_round2.py >> "$LOG" 2>&1 || { log "evaluate_round2 FAILED"; exit 1; }
log "ROUND 2 COMPLETE"
