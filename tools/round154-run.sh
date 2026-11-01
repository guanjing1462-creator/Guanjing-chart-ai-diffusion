#!/bin/bash
# round154 一键流水线：下载(限时) → 4 带条件集 → mel 通量集 → 质量自证 → 报告
set -o pipefail
cd /root/work/yamero-ai/proj || exit 1
D=out/diffusion/round154-data
mkdir -p "$D"

echo "== [1/5] 下载 $(date -u) =="
python3 tools/round154-dl.py --minutes "${DL_MIN:-10}" --workers 2 \
    --cand-file /tmp/cand154.json --target-new 62 > "$D/dl-final.log" 2>&1
echo "下载退出码 $?"

echo "== [2/5] 4 带 onset 条件集 $(date -u) =="
python3 tools/build-audio-cond-fixed.py --dl out/audio-dl \
    --out out/diff-data-audio-fixed2 > "$D/build-fixed.log" 2>&1
echo "fixed 退出码 $?"

echo "== [3/5] 32 带 mel onset 通量 $(date -u) =="
python3 tools/build-mel-cond.py --src out/diff-data-audio-fixed2/dataset.npz \
    --dl out/audio-dl --out out/diff-data-mel2 --feature flux --agg mean \
    > "$D/build-mel.log" 2>&1
echo "mel 退出码 $?"

echo "== [4/5] 质量自证 $(date -u) =="
python3 tools/round154-quality.py > "$D/quality.log" 2>&1
echo "quality 退出码 $?"

echo "== [5/5] 报告 $(date -u) =="
python3 tools/round154-report.py > "$D/report.log" 2>&1
echo "report 退出码 $?"
echo "ALL DONE $(date -u)"
