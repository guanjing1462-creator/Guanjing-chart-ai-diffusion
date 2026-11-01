#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""accent-audit: 量「该有的重音有没有」，以及「有没有空音」。

用户听感反馈的三条，前两条可以直接量化：

  空音（打在没声音的地方）
    silentNotes = 音符时刻的局部 onset 能量 < 全曲 p15 的比例
    （连续音/静音段里还硬塞音符）

  重音没采（真正响的地方空着 / 只给单音）
    accentCover = 音乐强 onset（env 的局部极大值，前 10%）在 ±40ms 内有音符的比例
    accentChord = 被采到的强 onset 里，落在和弦（≥2 键）上的比例
    强段空窗   = 高能量 2s 窗里行数 < 2 的比例（"单调空虚"）

  叠（tech 只堆和弦）
    chordRowRate = 和弦行占比
    flowRate     = 和弦行之间连续单音的占比（"叠"的反面是"有流动"）

用法：
  python3 tools/accent-audit.py --osu X.osu --audio "ref/INTERNET YAMERO.ogg"
"""
import argparse
import bisect
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('ac', os.path.join(HERE, 'audit-chart.py'))
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)


def build_env(audio, hop=256):
    import librosa
    y, sr = librosa.load(audio, sr=22050, mono=True)
    env = np.maximum(librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop), 0)
    return env, sr / hop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--osu', required=True)
    ap.add_argument('--audio', required=True)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    env, fps = build_env(args.audio)
    ms_per_frame = 1000.0 / fps
    dur_ms = len(env) * ms_per_frame
    notes, _ = ac.parse_osu(args.osu)
    if not notes:
        print('没有音符')
        return
    rows = ac.rows_of(notes)

    def env_at(t_ms, half=60.0):
        f0 = max(0, int((t_ms - half) / ms_per_frame))
        f1 = min(len(env), int((t_ms + half) / ms_per_frame) + 1)
        return float(env[f0:f1].max()) if f1 > f0 else 0.0

    # ---- 空音 ----
    note_env = np.array([env_at(t) for t, c, e in notes])
    p15 = float(np.percentile(env, 15))
    p90 = float(np.percentile(env, 90))
    silent = float((note_env < p15).mean())

    # ---- 强 onset（局部极大 + 高于 p90 + 间隔 >=100ms）----
    idx = []
    last = -10**9
    for i in range(1, len(env) - 1):
        if env[i] >= env[i - 1] and env[i] > env[i + 1] and env[i] >= p90:
            t = i * ms_per_frame
            if t - last >= 100:
                idx.append(t)
                last = t
    accents = np.array(idx)
    times = np.array([t for t, c, e in notes])
    chord_flag = np.array([len(c) >= 2 for t, c in rows])
    row_times = np.array([t for t, c in rows])

    def near(arr, t, tol):
        j = bisect.bisect_left(arr, t - tol)
        return j < len(arr) and arr[j] <= t + tol

    cov = sum(1 for a in accents if near(row_times, a, 40))
    cov_tight = sum(1 for a in accents if near(row_times, a, 22))
    acc_chord = 0
    for a in accents:
        j = bisect.bisect_left(row_times, a - 40)
        if j < len(row_times) and row_times[j] <= a + 40:
            k = j
            best = abs(row_times[j] - a)
            while k + 1 < len(row_times) and abs(row_times[k + 1] - a) <= best:
                k += 1
                best = abs(row_times[k] - a)
            if chord_flag[k]:
                acc_chord += 1

    # ---- 强段空窗（单调空虚）----
    win = 2000.0
    nwin = int(dur_ms / win) + 1
    wen = np.zeros(nwin)
    cnt = np.zeros(nwin)
    for i, v in enumerate(env):
        k = int(i * ms_per_frame / win)
        if k < nwin:
            wen[k] += v
            cnt[k] += 1
    wen = wen / np.maximum(cnt, 1)
    nrows = np.zeros(nwin)
    for t, c in rows:
        k = int(t / win)
        if 0 <= k < nwin:
            nrows[k] += 1
    loud = wen >= np.percentile(wen, 70)
    dead = float((nrows[loud] < 2).mean()) if loud.any() else 0.0

    # ---- 流动 vs 叠 ----
    sizes = np.array([len(c) for t, c in rows])
    chord_rate = float((sizes >= 2).mean())
    flow = 0
    run = 0
    for s in sizes:
        if s == 1:
            run += 1
        else:
            flow += run
            run = 0
    flow += run
    flow_rate = flow / len(sizes)

    rep = {
        'file': os.path.basename(args.osu),
        'notes': len(notes), 'rows': len(rows),
        'silentNoteRate': round(silent, 4),
        'accentCount': int(len(accents)),
        'accentCover40': round(cov / max(1, len(accents)), 4),
        'accentCover22': round(cov_tight / max(1, len(accents)), 4),
        'accentChordRate': round(acc_chord / max(1, cov), 4),
        'loudWindowDeadRate': round(dead, 4),
        'chordRowRate': round(chord_rate, 4),
        'flowRate': round(flow_rate, 4),
    }
    print(f"{rep['file'][:40]:42s} 空音 {silent*100:4.1f}%  "
          f"重音覆盖(±40) {rep['accentCover40']*100:5.1f}%  (±22) {rep['accentCover22']*100:5.1f}%  "
          f"覆盖中带和弦 {rep['accentChordRate']*100:5.1f}%  "
          f"强段空窗 {dead*100:4.1f}%  和弦行 {chord_rate*100:4.1f}%  单音占比 {flow_rate*100:5.1f}%")
    if args.json:
        import json
        json.dump(rep, open(args.json, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
