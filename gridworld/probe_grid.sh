#!/usr/bin/env bash
# Probe all 18 trained models. Writes per-checkpoint JSON to probe_results/.
set -euo pipefail

mkdir -p probe_results

start_time=$(date +%s)
total=18
i=0

for n in 10 30 50; do
  for size in small large; do
    for seed in 0 1 2; do
      out="probe_results/${size}_n${n}_seed${seed}.json"
      ckpt="checkpoints/model_${size}_n${n}_seed${seed}.pt"
      i=$((i + 1))

      if [ -f "$out" ]; then
        echo "[$i/$total] SKIP $out (exists)"
        continue
      fi

      echo "[$i/$total] PROBE size=$size n=$n seed=$seed"
      python probe.py \
        --checkpoint "$ckpt" \
        --size "$size" \
        --data_path "dataset_${n}.pkl" \
        --output "$out"
    done
  done
done

elapsed=$(( $(date +%s) - start_time ))
printf "Done. Elapsed: %dm %ds\n" $((elapsed / 60)) $((elapsed % 60))
