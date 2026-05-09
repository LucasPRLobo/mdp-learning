#!/usr/bin/env bash
# Train 18 models: {small, large} x {seed 0,1,2} x {dataset 10, 30, 50}
set -euo pipefail

mkdir -p checkpoints logs

start_time=$(date +%s)
total=18
done_count=0

for n in 10 30 50; do
  for size in small large; do
    for seed in 0 1 2; do
      ckpt="checkpoints/model_${size}_n${n}_seed${seed}.pt"
      log="logs/${size}_n${n}_seed${seed}.txt"
      done_count=$((done_count + 1))

      if [ -f "$ckpt" ]; then
        echo "[$done_count/$total] SKIP $ckpt (exists)"
        continue
      fi

      echo "[$done_count/$total] RUN  size=$size n=$n seed=$seed -> $ckpt"
      python train.py \
        --seed "$seed" \
        --size "$size" \
        --data_path "dataset_${n}.pkl" \
        --output "$ckpt" \
        2>&1 | tee "$log"
    done
  done
done

elapsed=$(( $(date +%s) - start_time ))
printf "Done. Elapsed: %dm %ds\n" $((elapsed / 60)) $((elapsed % 60))
