#!/bin/bash
# round154 建集流水线：4 带 onset → 32 带 mel 通量 → 质量自证 → 报告
set -o pipefail
cd /root/work/yamero-ai/proj || exit 1
D=out/diffusion/round154-data
mkdir -p "$D"

echo "== [1/4] 4 带 onset 条件集 $(date -u) =="
python3 tools/build-audio-cond-fixed.py --dl out/audio-dl \
    --out out/diff-data-audio-fixed2 > "$D/build-fixed.log" 2>&1
echo "fixed 退出码 $?"

echo "== [2/4] 32 带 mel onset 通量 $(date -u) =="
python3 tools/build-mel-cond.py --src out/diff-data-audio-fixed2/dataset.npz \
    --dl out/audio-dl --out out/diff-data-mel2 --feature flux --agg mean \
    > "$D/build-mel.log" 2>&1
echo "mel 退出码 $?"

echo "== [3/4] 质量自证 $(date -u) =="
python3 tools/round154-quality.py > "$D/quality.log" 2>&1
echo "quality 退出码 $?"

echo "== [4/4] 报告 $(date -u) =="
python3 tools/round154-report.py > "$D/report.log" 2>&1
echo "report 退出码 $?"
echo "ALL DONE $(date -u)"
