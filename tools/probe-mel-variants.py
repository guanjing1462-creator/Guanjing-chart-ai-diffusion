#!/usr/bin/env python3
"""探针：mel 条件的不同聚合口径，逐谱算 X–C 相关，挑出 ≥ 4 带基线的那个。

候选变体（都是 32 通道，mel 参数 n_mels=32/n_fft=1024/hop=256/sr=22050）：
  energy_mean  log1p(mel) → 格内均值                       （= round151 第一版，已知 0.287）
  fl_mean      log1p(mel) → 正向时间差分 → 格内均值
  fl_max       log1p(mel) → 正向时间差分 → 格内最大
  fl_mean_h   同上，但通量时刻整体前移半帧（对齐起音中心）
  fl_max_h    同上 + 格内最大
归一化：每首歌逐通道 P99 → clip[0,1]；指标：占用 vs 32 通道求和，全局时间轴 Pearson。

用法： python3 tools/probe-mel-variants.py [--limit-charts 8]
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BM = load_module('tools/build-mel-cond.py', 'bm')
BFX = BM.BFX
WIN = BM.WIN


def cell_reduce(mel, tf, q, step, n, mode):
    """格内均值/最大。[n, ch]"""
    n = int(n)
    F = mel.shape[1]
    ch = mel.shape[0]
    out = np.zeros((n, ch), np.float32)
    if n <= 0 or F == 0:
        return out
    t0 = q + np.arange(n, dtype=np.float64) * step
    i0 = np.clip(np.searchsorted(tf, t0, 'left'), 0, F)
    i1 = np.clip(np.searchsorted(tf, t0 + step, 'left'), 0, F)
    ok = i1 > i0
    if not ok.any():
        return out
    cs = np.concatenate([np.zeros((ch, 1)), np.cumsum(mel.astype(np.float64), 1)],
                        axis=1)
    a, b = i0[ok], i1[ok]
    if mode == 'mean':
        out[ok] = ((cs[:, b] - cs[:, a]) / (b - a)[None, :]).T.astype(np.float32)
    else:
        # 分段最大：逐格切片（格很窄，循环可接受）
        j = np.flatnonzero(ok)
        for k in j:
            out[k] = mel[:, i0[k]:i1[k]].max(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='out/diff-data-audio-fixed/dataset.npz')
    ap.add_argument('--dl', default='out/audio-dl')
    ap.add_argument('--limit-charts', type=int, default=0)
    ap.add_argument('--out', default='out/diff-data-mel/variant-probe.json')
    a = ap.parse_args()

    src = os.path.join(ROOT, a.src)
    d = np.load(src, allow_pickle=True)
    X, C, cid, st = d['X'], d['C'], d['chart_id'].astype(str), d['start_cell']
    song = d['song'].astype(np.int64)
    man = json.load(open(os.path.join(os.path.dirname(src), 'manifest.json'),
                        encoding='utf-8'))
    by_cid = {m['chart_id']: m for m in man if m.get('verdict') == 'kept'}
    charts = list(np.unique(cid))
    if a.limit_charts:
        charts = charts[:a.limit_charts]

    VARIANTS = [('energy_mean', 'energy', 'mean', 0.0),
                ('fl_mean', 'flux', 'mean', 0.0),
                ('fl_max', 'flux', 'max', 0.0),
                ('fl_mean_h', 'flux', 'mean', BM.MEL_HOP / 2.0),
                ('fl_max_h', 'flux', 'max', BM.MEL_HOP / 2.0)]
    cache = BM.MelCache()
    raw = {v[0]: {} for v in VARIANTS}
    valid = {}
    t_all = time.time()
    for ci, c in enumerate(charts):
        m = by_cid[c]
        set_dir = os.path.join(ROOT, a.dl, str(m['SetID']))
        ap_ = BM.resolve_audio(set_dir, m.get('audio') or '')
        L, tf, dur_ms = cache.get(ap_)
        q = float(m['offset_ms']) + float(m['shift_ms'])
        step = float(m['step_ms'])
        n = int(m['n_cells'])
        t0 = q + np.arange(n, dtype=np.float64) * step
        valid[c] = (t0 >= 0.0) & (t0 < min(float(dur_ms), float(tf[-1])))
        flux = np.maximum(np.diff(L, axis=1, prepend=L[:, :1]), 0.0).astype(np.float32)
        for name, kind, agg, sh in VARIANTS:
            arr = L if kind == 'energy' else flux
            raw[name][c] = cell_reduce(arr, tf - sh, q, step, n, agg)
        if (ci + 1) % 20 == 0 or ci == len(charts) - 1:
            print(f'  [{ci+1}/{len(charts)}] 格内聚合 {time.time()-t_all:.0f}s', flush=True)

    song_of = {c: int(song[np.flatnonzero(cid == c)[0]]) for c in charts}

    def norm_variant(name):
        scale = {}
        for s in sorted(set(song_of.values())):
            cs = [c for c in charts if song_of[c] == s]
            cat = np.concatenate([raw[name][c][valid[c]] for c in cs], 0)
            p = np.percentile(cat, 99, axis=0) if len(cat) else np.ones(32)
            scale[s] = np.where(p > 1e-9, p, 1.0)
        out = {}
        for c in charts:
            out[c] = np.clip(raw[name][c] / scale[song_of[c]][None, :],
                             0.0, 1.0).astype(np.float32)
        return out

    def per_chart_r(mats, key):
        rows = []
        for c in charts:
            sel = np.flatnonzero(cid == c)
            cr = [(int(st[i]), j) for j, i in enumerate(sel)]   # 局部行号
            occ = BM.occupancy_timeline(cr, X[sel])
            if len(occ) < 20:
                continue
            if isinstance(mats, np.ndarray):        # 4 带：直接按窗口取行
                wins = mats[sel]
            else:                                  # 变体：按格号从整首格特征切窗
                cells = mats[c]
                wins = np.stack([cells[int(st[i]):int(st[i]) + WIN] for i in sel])
            ml = BM.timeline(cr, wins)
            e = ml.sum(axis=1) if key != 'C' else ml.mean(axis=1)
            rows.append((str(c), BFX.pearson(occ, e)))
        return rows

    res, per_chart = {}, {}
    band_rows = dict(per_chart_r(np.asarray(C, np.float32), 'C'))
    for name, *_ in VARIANTS:
        mats = norm_variant(name)
        rows = per_chart_r(mats, name)
        per_chart[name] = {c: round(r, 4) for c, r in rows}
        rr = np.array([r for _, r in rows])
        beats = sum(1 for c, r in rows if r > band_rows.get(c, 0))
        res[name] = dict(median=round(float(np.median(rr)), 4),
                         mean=round(float(rr.mean()), 4),
                         frac_ge_025=round(float((rr >= 0.25).mean()), 4),
                         frac_ge_040=round(float((rr >= 0.40).mean()), 4),
                         min=round(float(rr.min()), 4),
                         beats_band4=int(beats), n=int(len(rr)))
        print(f'{name:12s} 中位 {res[name]["median"]:+.3f}  '
              f'r≥0.25 {res[name]["frac_ge_025"]*100:3.0f}%  '
              f'r≥0.40 {res[name]["frac_ge_040"]*100:3.0f}%  '
              f'优于4带的谱 {beats}/{len(rr)}', flush=True)
    rb = np.array(list(band_rows.values()))
    print(f'{"4带(对照)":12s} 中位 {np.median(rb):+.3f}  '
          f'r≥0.25 {(rb >= 0.25).mean()*100:3.0f}%')
    best = max(res, key=lambda k: res[k]['median'])
    print(f'\n最佳变体：{best}（中位 {res[best]["median"]:+.3f}）', flush=True)
    outp = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(dict(variants=res, band4_median=round(float(np.median(rb)), 4),
                   band4_frac_ge_025=round(float((rb >= 0.25).mean()), 4),
                   best=best, n_charts=len(charts),
                   seconds=round(time.time() - t_all, 1),
                   per_chart=per_chart, band4_per_chart={
                       c: round(r, 4) for c, r in band_rows.items()}),
              open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'写出 {a.out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
