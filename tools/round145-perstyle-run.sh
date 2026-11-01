#!/usr/bin/env bash
# 第 145 轮：四风格分模型，串行训练 + 就地验收（同一时刻只有一个训练进程）
set -euo pipefail
cd /root/work/yamero-ai/proj
ROOT=out/diffusion/round145-perstyle
mkdir -p "$ROOT"
for s in tech jack speed stamina; do
  echo "==== 训练 $s 开始 $(date +%H:%M:%S) ===="
  python3 tools/train-diffusion32-perstyle.py \
    --data out/diff-data-grouped/dataset.npz --only-style "$s" \
    --pred x0 --base 48 --batch 64 --lr 2e-4 --threads 1 \
    --max-windows 4000 --epochs 9 --time-budget 1000 --min-epochs 0 \
    --group-cap 300 --init-seed 0 --seed 0 --measure-ab 0 \
    --out "$ROOT/$s.pt" --log "$ROOT/$s-train.jsonl" >> "$ROOT/$s-train.log" 2>&1
  echo "==== 训练 $s 结束 / 验收采样开始 $(date +%H:%M:%S) ===="
  python3 tools/eval-perstyle.py --root "$ROOT" --styles "$s" --num 5 \
    --diff 0.5 --seed 0 --threads 1 --out "$ROOT/eval-$s.json" >> "$ROOT/$s-eval.log" 2>&1
  echo "==== 验收 $s 结束 $(date +%H:%M:%S) ===="
done
echo "==== 全流程完成 $(date +%H:%M:%S) ===="
