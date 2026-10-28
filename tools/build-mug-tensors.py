#!/usr/bin/env python3
"""把 .osu 编码成 Mug-Diffusion 的谱面张量（16ch × 4096 帧 @ 46.44ms）。

    python3 tools/build-mug-tensors.py out/charts -o out/mug_tensors

通道布局（mug/data/convertor.py Feature Layout，4K）：
  ch [0..4)   is_start      该帧该列是否有音符头
  ch [4..8)   offset_start  头在本帧内的亚帧偏移（0-1）
  ch [8..12)  is_holding    LN 身体（头之后含尾帧）
  ch [12..16) offset_end    尾的亚帧偏移（仅末帧有效）

额外输出：
  timing: [4096, 2]  半拍打点（1 = 拍线，2 = 亚帧偏移）
  valid:  [4096]     有效长度掩码

帧率 46.4399ms（= 2048/22050/2*1000，与 Mug 一致），4096 帧 = 190.2s 上限，
超长谱截断（Mug 同款行为）。续传：单文件 npz，存在即跳过。
"""
import argparse
import glob
import os

import numpy as np

FRAME_MS = 2048 / 22050 / 2 * 1000     # 46.4399
MAX_FRAME = 4096


def read_timing(path):
    """第一根红线 → (bpm, offset_ms)。"""
    sec = None
    for line in open(path, encoding='utf-8', errors='ignore'):
        line = line.strip()
        if line == '[TimingPoints]':
            sec = 'tp'
            continue
        if line.startswith('[') and line != '[TimingPoints]':
            sec = None
        if sec != 'tp' or not line or line.startswith('//'):
            continue
        p = line.split(',')
        if len(p) < 2:
            continue
        try:
            beat_len = float(p[1])
        except ValueError:
            continue
        if beat_len > 0:
            return 60000.0 / beat_len, float(p[0])
    return None, None


def encode(path, keys=4):
    bpm, offset = read_timing(path)
    if bpm is None:
        return None
    arr = np.zeros((MAX_FRAME, keys * 4), dtype=np.float16)
    max_idx = 0
    lns = []          # (t, end, col)
    sec = None
    for line in open(path, encoding='utf-8', errors='ignore'):
        line = line.strip()
        if line == '[HitObjects]':
            sec = 'ho'
            continue
        if line.startswith('[') and line != '[HitObjects]':
            sec = None
        if sec != 'ho' or not line or line.startswith('//'):
            continue
        p = line.split(',')
        if len(p) < 5:
            continue
        try:
            t = float(p[2])
            x = int(p[0])
            is_ln = len(p) > 5 and p[5].split(':')[0] not in ('0', '')
            end = float(p[5].split(':')[0]) if is_ln else None
        except ValueError:
            continue
        col = x * keys // 512
        if not (0 <= col < keys):
            continue
        idx = int(t // FRAME_MS)
        if idx >= MAX_FRAME or idx < 0:   # 坏谱里有负时间戳
            continue
        off = (t - idx * FRAME_MS) / FRAME_MS
        arr[idx, col] = 1
        arr[idx, col + keys] = off
        max_idx = max(max_idx, idx)
        if is_ln:
            e_idx = min(MAX_FRAME - 1, int(end // FRAME_MS))
            if e_idx < 0:
                continue
            e_off = (end - e_idx * FRAME_MS) / FRAME_MS
            for i in range(idx + 1, e_idx + 1):
                arr[i, col + keys * 2] = 1
            arr[e_idx, col + keys * 3] = e_off
            max_idx = max(max_idx, e_idx)

    # timing 通道：半拍打点。BPM 钳制到 [150,300]（Mug convertor 同款）——
    # 畸形红线条目会把 half 推到 ≈0，t += half 永不前进 = 死循环。
    timing = np.zeros((MAX_FRAME, 2), dtype=np.float16)
    bpm_c = bpm
    while bpm_c < 150:
        bpm_c *= 2
    while bpm_c >= 300:
        bpm_c /= 2
    half = 60000.0 / bpm_c / 2
    t = offset
    it = 0
    while t < MAX_FRAME * FRAME_MS and it < MAX_FRAME * 4:
        idx = int(t // FRAME_MS)
        if 0 <= idx < MAX_FRAME:
            timing[idx, 0] = 1
            timing[idx, 1] = (t - idx * FRAME_MS) / FRAME_MS
        t += half
        it += 1

    valid = np.zeros(MAX_FRAME, dtype=np.int8)
    valid[:max_idx + 1] = 1
    return arr.transpose(), timing, valid, bpm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('charts')
    ap.add_argument('-o', '--out', default='out/mug_tensors')
    ap.add_argument('--max', type=int, default=0, help='最多处理多少张（0=全量）')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.charts, '*.osu')))
    done = 0, None
    made = skip = bad = 0
    for i, f in enumerate(files):
        cid = os.path.splitext(os.path.basename(f))[0]
        outp = os.path.join(args.out, f'{cid}.npz')
        if os.path.exists(outp):
            skip += 1
            continue
        r = encode(f)
        if r is None:
            bad += 1
            continue
        arr, timing, valid, bpm = r
        np.savez_compressed(outp, note=arr, timing=timing, valid=valid, bpm=bpm)
        made += 1
        if args.max and made >= args.max:
            break
        if (i + 1) % 1000 == 0:
            print(f'  {i+1}/{len(files)}  编码 {made}  跳过 {skip}  坏 {bad}')
    print(f'完成：编码 {made}  已存在 {skip}  无法解析 {bad}')


if __name__ == '__main__':
    main()
