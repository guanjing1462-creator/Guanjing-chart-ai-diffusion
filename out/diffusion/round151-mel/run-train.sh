#!/bin/bash
set -u
cd /root/work/yamero-ai/proj
COMMON="--cond-dim 32 --cond-drop 0.1 --epochs 9 --batch 32 --pred x0 \
 --threads 1 --init-seed 7 --group-cap 300 --time-budget 2100"
echo "=== [1/2] mel onset 通量条件 ==="
python3 tools/train-diffusion32-mel.py --data out/diff-data-mel/dataset.npz --cond-key M $COMMON \
  --out out/models/r151-M-melflux.pt \
  --log out/diffusion/round151-mel/train-M.jsonl \
  > out/diffusion/round151-mel/train-M.log 2>&1
echo "mel exit=$?"
echo "=== [2/2] 4 带条件（同训练器基线）==="
python3 tools/train-diffusion32-mel.py --data out/diff-data-audio-fixed/dataset.npz --cond-key C $COMMON \
  --out out/models/r151-C-band4.pt \
  --log out/diffusion/round151-mel/train-C.jsonl \
  > out/diffusion/round151-mel/train-C.log 2>&1
echo "band4 exit=$?"
