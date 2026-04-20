#!/usr/bin/env bash
# Sweep (eps_del_scale, eps_ins_scale) grid. Writes incrementally to stdout.
cd /home/grey/dev/graiai/malleus
source /home/grey/dev/graiai/sherpa-onnx/build_venv/bin/activate

DELS="${DELS:-0.6 1.0 1.4}"
INSS="${INSS:-0.4 0.7 1.0}"
OUT="${OUT:-results/primock57/eps_grid_sweep.tsv}"
echo -e "eps_del\teps_ins\twer\tdelta" > "$OUT"

for del in $DELS; do
    for ins in $INSS; do
        tmp=$(mktemp)
        python3 -u scripts/cnc_fusion.py \
            --models parakeet-tdt-0.6b-v2,whisper-distil-v3.5 \
            --dataset primock57 \
            --eps-del-scale $del \
            --eps-ins-scale $ins > "$tmp" 2>&1
        wer=$(grep -oP 'CNC fused WER\s+:\s+\K[0-9.]+' "$tmp")
        delta=$(grep -oP 'Delta\s+:\s+\K[+-][0-9.]+' "$tmp")
        rm -f "$tmp"
        echo -e "$del\t$ins\t$wer\t$delta" | tee -a "$OUT"
    done
done
