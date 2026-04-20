#!/usr/bin/env bash
cd /home/grey/dev/graiai/malleus
source /home/grey/dev/graiai/sherpa-onnx/build_venv/bin/activate
OUT=results/primock57/learned_sweep.tsv
echo -e "mode\tdrop_thr\teps_del\teps_ins\twer\tdelta" > "$OUT"

run() {
    local mode=$1 drop=$2 ed=$3 ei=$4
    tmp=$(mktemp)
    python3 -u scripts/learned_fusion.py \
        --mode $mode --drop-thr $drop \
        --eps-del $ed --eps-ins $ei > "$tmp" 2>&1
    wer=$(grep -oP '5-fold CV learned CNC: \K[0-9.]+' "$tmp")
    delta=$(grep -oP 'Δ\s+:\s+\K[+-][0-9.]+' "$tmp" | tail -1)
    rm -f "$tmp"
    echo -e "$mode\t$drop\t$ed\t$ei\t$wer\t$delta" | tee -a "$OUT"
}

# hybrid / gate sweep with fixed eps near hand-tuned sweet spot
for drop in 0.10 0.20 0.30 0.40; do
    run hybrid $drop 0.60 0.70
done
