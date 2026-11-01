#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 采样器端到端验收：音符密度 vs 音频强度（250/1000ms）、到最近强起音的距离、
以及「条件平移 ±1/±4 格后指标是否退化」的对照。

输入是 tools/sample-diffusion32align.py 的产物（rolls-*.npz，含 X=[n,4,256]、
P=[n,4,256]、C=[n,256,ch]（喂给模型的条件）、以及同目录 sample-meta-*.json 里的
start_cell）。所以「喂给模型的条件」与「真条件」可以分开算相关：
  - r(对给定C)：roll vs 实际喂进去的条件（跟着平移 = 模型在用条件）
  - r(对真C)  ：roll vs 歌曲原始条件（平移后应明显下降）

统计口径
--------
1. 占位率 = 二值 roll 的均值（与采样器 roll_stats 同口径）；
2. 强起音：librosa onset_strength(hop=256) → peak_pick(pre/post_max=3, pre/post_avg=8,
   delta=0.12, wait=2)（与 tools/paint-gj.py 单路检测同参数），再取强度前 30% 的峰；
   音符时间 = (start_cell + 窗内偏移) * step_ms（.osu 由 --concat 从 0 起排）。
3. r250 / r1000：1/32 格。音符密度按 250/1000ms 分箱 vs 条件 32 带求和同分箱，
   逐窗算 Pearson 再报中位；同时报整段（6 窗拼起来）池化值。
4. 强起音距离：每个音符到最近强起音的时间差，报中位 / 均值 / ±1 格内比例。

用法
----
  python3 tools/check-align-sampler.py --audio "ref/multisong/Heart Fraction.ogg" \\
      --bpm 136.3 --true-cond out/diffusion/round154-sampler/Heart-Fraction-mel.npz \\
      --run paired=out/diffusion/round154-sampler/paired \\
      --run shift1=out/diffusion/round154-sampler/shift1 \\
      --run shift4=out/diffusion/round154-sampler/shift4 \\
      --out out/diffusion/round154-sampler/acceptance.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

SR = 22050
HOP = 256
WIN = 256


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def load_run(path):
    """path = 目录或 rolls-*.npz → dict(X, P, C, start_cell, meta)。"""
    if os.path.isdir(path):
        npz = sorted(glob.glob(os.path.join(path, 'rolls-*.npz')))
        if len(npz) != 1:
            raise SystemExit(f'[错误] {path} 下应有 1 个 rolls-*.npz，实际 {len(npz)}')
        npz = npz[0]
    else:
        npz = path
    z = np.load(npz, allow_pickle=True)
    for k in ('X', 'P'):
        if k not in z.files:
            raise SystemExit(f'[错误] {npz} 缺 {k}')
    meta_p = os.path.join(os.path.dirname(os.path.abspath(npz)),
                          'sample-meta-' +
                          os.path.basename(npz)[len('rolls-'):-len('.npz')] + '.json')
    meta = json.load(open(meta_p, encoding='utf-8')) if os.path.exists(meta_p) else {}
    start = int(meta.get('start_cell', 0))
    C = z['C'].astype(np.float32) if 'C' in z.files else None
    return dict(path=npz, X=z['X'].astype(np.float32), P=z['P'].astype(np.float32),
                C=C, start_cell=start, meta=meta)


def strong_onsets(audio, pct=70.0):
    """强起音时刻（ms）：onset_strength → peak_pick → 取强度前 (100-pct)% 的峰。"""
    import librosa
    y, sr = librosa.load(audio, sr=SR, mono=True)
    env = np.maximum(librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP), 0)
    pk = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=8, post_avg=8,
                                delta=0.12, wait=2)
    if len(pk) == 0:
        return np.zeros(0), np.zeros(0), env
    val = env[pk]
    thr = np.percentile(val, pct)
    keep = pk[val >= thr]
    return (keep * 1000.0 * HOP / sr).astype(np.float64), val[val >= thr], env


def binned(roll_flat, cond_tot, cpb, m):
    nb = m // cpb
    dens = roll_flat[:nb * cpb].reshape(nb, cpb).sum(axis=1)
    cb = cond_tot[:nb * cpb].reshape(nb, cpb).sum(axis=1)
    return dens, cb


def occupancy(roll):
    return float(roll.mean())


def onset_metrics(note_ms, so_ms, step_ms):
    if len(note_ms) == 0 or len(so_ms) < 2:
        return dict(音符数=int(len(note_ms)), 中位_ms=None, 均值_ms=None,
                    p25_ms=None, p75_ms=None, 一格外占比=None)
    idx = np.searchsorted(so_ms, note_ms)
    idx = np.clip(idx, 1, len(so_ms) - 1)
    d1 = np.abs(note_ms - so_ms[idx - 1])
    d2 = np.abs(note_ms - so_ms[idx])
    d = np.minimum(d1, d2)
    return dict(音符数=int(len(note_ms)),
                中位_ms=round(float(np.median(d)), 2),
                均值_ms=round(float(d.mean()), 2),
                p25_ms=round(float(np.percentile(d, 25)), 2),
                p75_ms=round(float(np.percentile(d, 75)), 2),
                一格外占比=round(float((d > step_ms).mean()), 4))


def metrics(run, true_cond_all, step_ms, so_ms):
    X, P, C = run['X'], run['P'], run['C']
    n = X.shape[0]
    start = run['start_cell']
    # 条件总强度（32 带求和）；喂给模型的与真条件各一份
    c_given = C.sum(axis=2) if C is not None else None
    c_true = np.stack([true_cond_all[start + i * WIN:start + (i + 1) * WIN].sum(axis=1)
                       for i in range(n)]) if true_cond_all is not None else None
    fp = np.full((n, 4, WIN), -1e9, np.float32) if n == P.shape[0] else None

    out = dict(命令=run['path'], start_cell=start, 窗口数=int(n),
               占位率=round(occupancy(X), 4),
               音符数=[int(v) for v in X.sum(axis=(1, 2))],
               每窗占位率=[round(float(v), 4) for v in X.mean(axis=(1, 2))])
    # 音符时间（.osu 由 --concat 从 0 起连续排）
    offs = np.arange(n * WIN)
    rows, lanes = np.nonzero(X.transpose(0, 2, 1).reshape(-1, 4))
    cells = offs[rows] + start
    note_ms = cells * step_ms
    out['强起音距离'] = onset_metrics(note_ms, so_ms, step_ms)
    out['强起音数'] = int(len(so_ms))

    # 250 / 1000ms 分箱相关（先逐窗，再整段池化）
    for tag, bin_ms in (('250', 250.0), ('1000', 1000.0)):
        cpb = max(1, int(round(bin_ms / step_ms)))
        rs_t, rs_g = [], []
        for i in range(n):
            rf = X[i].sum(axis=0)            # 每格音符数（4 键求和）
            if c_true is not None:
                d, c = binned(rf, c_true[i], cpb, WIN)
                rs_t.append(pearson(d, c))
            if c_given is not None:
                d, c = binned(rf, c_given[i], cpb, WIN)
                rs_g.append(pearson(d, c))
        # 整段池化（窗首尾相接）
        rf_all = X.sum(axis=1).reshape(-1)   # [n*WIN]
        pool_t = pool_g = None
        if c_true is not None:
            pool_t = pearson(*binned(rf_all, c_true.reshape(-1), cpb, n * WIN))
        if c_given is not None:
            pool_g = pearson(*binned(rf_all, c_given.reshape(-1), cpb, n * WIN))
        out[f'cpb{tag}'] = cpb
        out[f'r{tag}_对真C'] = dict(
            逐窗中位=round(float(np.nanmedian(rs_t)), 4) if rs_t else None,
            池化=round(pool_t, 4) if pool_t is not None and np.isfinite(pool_t) else None)
        out[f'r{tag}_对给定C'] = dict(
            逐窗中位=round(float(np.nanmedian(rs_g)), 4) if rs_g else None,
            池化=round(pool_g, 4) if pool_g is not None and np.isfinite(pool_g) else None)
        out[f'逐窗_r{tag}_对真C'] = [round(v, 4) if np.isfinite(v) else None for v in rs_t]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('--bpm', type=float, required=True)
    ap.add_argument('--true-cond', required=True, help='原始 mel 条件 npz（M=[T,32]）')
    ap.add_argument('--run', action='append', required=True,
                    help='tag=采样输出目录（或 rolls-*.npz），可重复')
    ap.add_argument('--out', default='out/diffusion/round154-sampler/acceptance.json')
    a = ap.parse_args(argv)

    step_ms = 60000.0 / float(a.bpm) / 8.0
    z = np.load(a.true_cond, allow_pickle=True)
    true_cond = np.asarray(z['M'], dtype=np.float32)
    so_ms, so_val, env = strong_onsets(a.audio)
    print(f'1/32 格 {step_ms:.2f}ms；强起音 {len(so_ms)} 个 '
          f'（{len(so_ms) / (len(env) * HOP / SR):.2f}/s）')

    runs, results = {}, {}
    for spec in a.run:
        if '=' not in spec:
            raise SystemExit(f'[错误] --run 需要 tag=路径，收到 {spec!r}')
        tag, path = spec.split('=', 1)
        run = load_run(path)
        runs[tag] = run
        m = metrics(run, true_cond, step_ms, so_ms)
        results[tag] = m
        print(f"{tag:8s} 占位 {m['占位率']:.4f}  音符 {sum(m['音符数'])}  "
              f"起音距离中位 {m['强起音距离']['中位_ms']}ms  "
              f"r250(真C) {m['r250_对真C']['池化']}  r1000(真C) {m['r1000_对真C']['池化']}  "
              f"r250(给定C) {m['r250_对给定C']['池化']}")

    # ---- 输出层面的对照：同一 seed 下，平移条件是否真的改变了输出 ----
    cmp = {}
    base = runs.get('paired')
    if base is not None and base['C'] is not None:
        bX, bP = base['X'], base['P']
        for tag, run in runs.items():
            if tag == 'paired' or run['C'] is None:
                continue
            X2, P2 = run['X'], run['P']
            if X2.shape != bX.shape:
                continue
            inter = float((bX * X2).sum())
            union = float(((bX + X2) > 0).sum())
            cmp[tag] = dict(与配对IoU=round(inter / max(1.0, union), 4),
                            与配对概率相关=round(pearson(bP.reshape(-1), P2.reshape(-1)), 4),
                            占用变化=round(float(X2.mean() - bX.mean()), 4))
    out = dict(bpm=float(a.bpm), step_ms=round(step_ms, 4), win=WIN,
               音频=a.audio, 真条件文件=a.true_cond,
               强起音=dict(个数=int(len(so_ms)),
                           每秒=round(len(so_ms) / (len(env) * HOP / SR), 3),
                           定义='onset_strength+peak_pick(paint-gj 同参) 取强度前 30%',
                           强度中位=round(float(np.median(so_val)), 4) if len(so_val) else None),
               runs=results, 平移对照=cmp,
               说明='r(对真C) 用歌曲原始条件算；r(对给定C) 用实际喂给模型（可能已平移）的条件算。'
                    '平移若被模型跟随，则「对给定C」保持而「对真C」下降。')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n写出 {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
