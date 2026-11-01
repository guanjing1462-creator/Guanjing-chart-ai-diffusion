#!/bin/bash
# A/B 配对实验：同一初始权重 + 同一验证切分 + 同一验证噪声，只差音频条件分支
# A = 有音频条件（--audio-cond），B = 无条件。各 3 个随机种子，12 轮。
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p out/models out/diffusion/logs
DATA=out/diff-data/audio-cond-labels.npz
COND=out/diff-data/cond-windows.npz

for s in 0 1 2; do
  echo "===== seed $s · A（有音频条件）====="
  python3 tools/train-diffusion.py --data "$DATA" \
    --audio-cond "$COND" --cond-dim 32 \
    --epochs 12 --batch 16 --val-frac 0.1 --threads 1 \
    --seed "$s" --init-seed "$s" \
    --out "out/models/diff-cond-seed$s.pt" \
    --log "out/diffusion/logs/cond-seed$s.jsonl" || exit 1

  echo "===== seed $s · B（无音频条件）====="
  python3 tools/train-diffusion.py --data "$DATA" \
    --epochs 12 --batch 16 --val-frac 0.1 --threads 1 \
    --seed "$s" --init-seed "$s" \
    --out "out/models/diff-nocond-seed$s.pt" \
    --log "out/diffusion/logs/nocond-seed$s.jsonl" || exit 1
done
echo "===== 全部完成 ====="
