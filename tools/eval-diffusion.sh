#!/usr/bin/env bash
# eval-diffusion.sh —— 每次扩散 ckpt 更新后，自动采样 + 量踩音率，追加到日志
#
# 为什么需要它：训练 loss 和「出谱对音」不是一回事（loss 0.108→0.10 时踩音率
# 没动）。每 epoch 存盘后必须实测，否则会拿 loss 自欺。
#
# 用法：
#   ./tools/eval-diffusion.sh            # 跑一次
#   WATCH=1 ./tools/eval-diffusion.sh    # 守着 ckpt 变化，每变一次测一次
set -uo pipefail
cd "$(dirname "$0")/.."
CKPT=out/diffusion/diff.pt
MEL=out/audio_features/INTERNET_YAMERO.npz
LOG=out/diffusion-eval.jsonl
[ -f "$MEL" ] || python3 tools/make-mel.py --audio "ref/INTERNET YAMERO.ogg" -o "$MEL"

measure_once() {
  local tag="$1" out=/tmp/ev-$1.osu
  python3 tools/train_diffusion.py --sample --ckpt "$CKPT" --vae out/vae/vae.pt \
      --audio-npz "$MEL" --feature '{"ett":30,"technical":1,"hb":1}' -o "$out" >/dev/null 2>&1 || {
      echo "{\"epoch\":\"$tag\",\"error\":\"sample failed\"}" >> "$LOG"; return; }
  python3 tools/onset-align.py --osu "$out" -o "$out.a" >/dev/null 2>&1
  python3 - "$tag" "$out" "$out.a" <<'PY' >> "$LOG"
import json, subprocess, sys
tag, raw, ali = sys.argv[1], sys.argv[2], sys.argv[3]
def m(p):
    r = subprocess.run([sys.executable, 'tools/measure-onset.py', '--osu', p],
                       capture_output=True, text=True)
    try: return json.loads(r.stdout)
    except Exception: return {}
a, b = m(raw), m(ali)
rec = {'epoch': tag,
       'notes': a.get('notes'),
       'raw_lt41': a.get('rate<41ms'), 'raw_median': a.get('medianMs'),
       'aligned_lt41': b.get('rate<41ms'), 'aligned_lt10': b.get('rate<10ms'),
       'aligned_median': b.get('medianMs')}
print(json.dumps(rec, ensure_ascii=False))
PY
  tail -1 "$LOG"
}

if [ "${WATCH:-0}" = "1" ]; then
  last=""
  while true; do
    cur=$(md5sum "$CKPT" 2>/dev/null | cut -d' ' -f1)
    if [ -n "$cur" ] && [ "$cur" != "$last" ]; then
      last="$cur"
      echo "[$(date +%H:%M)] ckpt 变了，开始采样…"
      measure_once "$(date +%H%M)"
    fi
    sleep 120
  done
fi
measure_once "$(date +%H%M)"
