#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""human-pitch-check: 用人类谱 + 它们自己的 mel，量「音高方向 → 列移动」的真实基准。

动机：v11 加了音高引导，但产物上的相关系数一直只有 0.01-0.03。到底是生成器
没做到，还是人类谱本来就这样？没有基准就只能瞎调。本脚本对每张有音频的
人类谱算同一个指标（行到行的音高差 vs 列均值差），给出人类分布。

mel 是 [128,T] 的 log-mel（46.44ms/帧）。音高代理 = mel 频带加权质心，
限制在 40-90 号 mel 带（中高频，避开底鼓），再 log2。

用法：
  python3 tools/human-pitch-check.py --charts out/charts --audio out/audio_features \
      --n 200 -o out/human-pitchdir.json
"""
import argparse
import glob
import importlib.util
import json
import os
import random
import statistics as st

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('audit_chart', os.path.join(HERE, 'audit-chart.py'))
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)

FRAME_MS = 2048 / 22050 / 2 * 1000


def mel_pitch(mel, lo=40, hi=90):
    """log-mel [128,T] → 逐帧音高代理（log2 相对质心）。"""
    band = mel[lo:hi].astype(np.float64)
    band = np.maximum(band, 0.0)
    idx = np.arange(lo, hi, dtype=np.float64)[:, None]
    w = band.sum(axis=0) + 1e-9
    c = (band * idx).sum(axis=0) / w          # 平均 mel 带位置
    c = np.log2(np.maximum(c, 1.0))
    if len(c) >= 5:
        c = np.convolve(c, np.ones(5) / 5.0, mode='same')
    return c


def metric(osu_path, mel_path):
    notes, _ = ac.parse_osu(osu_path)
    if len(notes) < 300:
        return None
    mel = np.load(mel_path)['mel']
    p = mel_pitch(mel)
    rows = ac.rows_of(notes)
    t0 = notes[0][0]
    pairs = []
    prev = None
    for t, cs in rows:
        i = int((t - t0) / FRAME_MS)
        if i < 2 or i >= len(p):
            continue
        mean_c = sum(cs) / len(cs)
        if prev is not None and t - prev[0] <= 260:
            pairs.append((p[i] - prev[1], mean_c - prev[2]))
        prev = (t, p[i], mean_c)
    if len(pairs) < 100:
        return None
    r = ac.spearman([x for x, y in pairs], [y for x, y in pairs])
    return {'id': os.path.basename(osu_path)[:-4], 'n': len(pairs), 'pitchDir': round(r, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--charts', default='out/charts')
    ap.add_argument('--audio', default='out/audio_features')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('-o', default='out/human-pitchdir.json')
    ap.add_argument('--seed', type=int, default=3)
    args = ap.parse_args()
    ids = {os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(os.path.join(args.audio, '*.npz'))}
    pairs = []
    for p in glob.glob(os.path.join(args.charts, '*.osu')):
        cid = os.path.splitext(os.path.basename(p))[0]
        if cid in ids:
            pairs.append((p, os.path.join(args.audio, cid + '.npz')))
    random.Random(args.seed).shuffle(pairs)
    out = []
    for p, m in pairs[:args.n]:
        try:
            r = metric(p, m)
        except Exception:      # noqa: BLE001
            continue
        if r:
            out.append(r)
    vs = sorted(x['pitchDir'] for x in out)
    if not vs:
        print('没有可用样本')
        return
    res = {'n': len(vs),
           'p10': round(vs[int(len(vs) * .1)], 3), 'p50': round(vs[len(vs) // 2], 3),
           'p90': round(vs[int(len(vs) * .9)], 3), 'mean': round(st.mean(vs), 3),
           'fracPositive': round(sum(1 for v in vs if v > 0) / len(vs), 3),
           'samples': out}
    json.dump(res, open(args.o, 'w'), ensure_ascii=False, indent=1)
    print(f'人类谱音高→列相关（n={len(vs)}）: p10 {res["p10"]}  p50 {res["p50"]}  '
          f'p90 {res["p90"]}  正相关占比 {res["fracPositive"]*100:.0f}%')
    print('写出', args.o)


if __name__ == '__main__':
    main()
