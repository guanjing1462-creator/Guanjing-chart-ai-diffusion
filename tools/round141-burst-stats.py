#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""round141（成簇/邻接加权）的统一统计口径：数据集参考值 + 生成卷 pooled 指标。

只做统计，不训练、不采样。

  数据集侧：按 style 在 train(split==0) 上算占位率与 **p132（格口径、按间隔池化）**，
  与 tools/sample-diffusion32occ.py 的 `1/32占比_p132`、out/diff-data-grouped/report.json
  的 `p132` 定义逐位一致（相邻占用行间隔 == 1 的间隔数 / 全部间隔数，跨窗口累加后相除）。

  生成侧：读采样器写出的 rolls-*.npz（X=二值卷 [n,4,256]），同样口径算 pooled 占位率、
  p132，另外给两个诊断量：
    - 邻接覆盖 lane：同键列相邻的正样本格 / 正样本格（任务书原文的 adj 口径）
    - 邻接覆盖 row ：所在行左右邻行有占用的正样本格 / 正样本格（p132 度量的口径）
  这两个数解释「为什么加权没动 p132」或「为什么动了」。

用法：
  python3 tools/round141-burst-stats.py --data out/diff-data-grouped/dataset.npz \
      --gen baseline=/tmp/occ_0,/tmp/occ_1,/tmp/occ_2,/tmp/occ_3 \
      --gen lane-a2=out/diffusion/round141-burst/samples/lane-a2 \
      --out out/diffusion/round141-burst/stats.json
"""
import argparse
import glob
import json
import os

import numpy as np

STYLES = ['tech', 'jack', 'speed', 'stamina']


def pooled_stats(A, P=None):
    """A: [n,4,L] 0/1（已解码的卷）。P: 可选 [n,4,L] 原始概率，用于「对齐前」的密度/规整度。

    返回 pooled 占位率 / p132 / 两个邻接覆盖诊断；给了 P 再加「固定阈值 0.5 的原始占位率与 p132」
    —— 后者就是**密度漂移**的证据：row 加权如果把原始占位率推高，p132 的改善里就混着密度贡献。
    """
    A = np.asarray(A, dtype=np.int32)
    row = (A.sum(1) > 0).astype(np.int32)                 # [n,L] 行占用
    row_next = np.zeros_like(row)
    row_next[:, 1:] += row[:, :-1]
    row_next[:, :-1] += row[:, 1:]
    adj_row = A * (row_next > 0)[:, None, :]              # 行口径邻接（只标有音符的格）

    lane_next = np.zeros_like(A)
    lane_next[:, :, 1:] += A[:, :, :-1]
    lane_next[:, :, :-1] += A[:, :, 1:]
    adj_lane = A * (lane_next > 0)

    g1 = tot = 0
    for r in row:
        idx = np.where(r > 0)[0]
        if len(idx) < 2:
            continue
        d = np.diff(idx)
        g1 += int((d == 1).sum())
        tot += int(d.size)
    pos = float(A.sum())
    extra = {}
    if P is not None:
        raw = (np.asarray(P) > 0.5).astype(np.int32)
        r_row = (raw.sum(1) > 0).astype(np.int32)
        rg = rt = 0
        for r in r_row:
            idx = np.where(r > 0)[0]
            if len(idx) < 2:
                continue
            d = np.diff(idx)
            rg += int((d == 1).sum())
            rt += int(d.size)
        extra = {'原始占位率@0.5': round(float(raw.mean()), 4),
                 '原始p132@0.5': round(rg / max(1, rt), 4)}
    return {
        '窗口数': int(A.shape[0]),
        '占位率': round(float(A.mean()), 4),
        'p132': round(g1 / max(1, tot), 4),
        '间隔数': int(tot),
        '非空行比例': round(float(row.mean()), 4),
        '每窗音符数': round(pos / max(1, A.shape[0]), 1),
        '邻接覆盖_lane': round(float(adj_lane.sum()) / max(1.0, pos), 4),
        '邻接覆盖_row': round(float(adj_row.sum()) / max(1.0, pos), 4),
        **extra,
    }


def dataset_stats(path):
    d = np.load(path)
    X = np.asarray(d['X'])
    sp, st = np.asarray(d['split']), np.asarray(d['style'])
    out = {}
    tr = sp == 0
    out['_all_train'] = pooled_stats(X[tr])
    for s, name in enumerate(STYLES):
        out[name] = pooled_stats(X[tr & (st == s)])
    return out


def load_rolls(dirs):
    """读一个配置目录下全部 rolls-*.npz，按 style 分组堆叠。返回 {style_name: [n,4,256]}。"""
    xs, ps = {}, {}
    for f in sorted(glob.glob(os.path.join(dirs, 'rolls-*.npz'))):
        z = np.load(f)
        s = int(np.asarray(z['style']).reshape(-1)[0]) if 'style' in z.files else None
        if s is None:
            base = os.path.basename(f)
            s = int(base.split('s')[1].split('d')[0])
        xs.setdefault(STYLES[s], []).append(np.asarray(z['X']))
        if 'P' in z.files:
            ps.setdefault(STYLES[s], []).append(np.asarray(z['P']))
    if not xs:
        raise SystemExit(f'[错误] {dirs} 里没有 rolls-*.npz')
    out = {k: np.concatenate(v) for k, v in xs.items()}
    prb = {k: np.concatenate(v) for k, v in ps.items()}
    return out, prb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='out/diff-data-grouped/dataset.npz')
    ap.add_argument('--gen', action='append', default=[],
                    help='label=dir[,dir...]，dir 里是采样器写的 rolls-*.npz')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    res = {'数据集': dataset_stats(a.data), '生成': {}}
    print('=== 数据集（train split==0，按 style）===')
    for name in STYLES:
        r = res['数据集'][name]
        print(f'  {name:8s} 窗 {r["窗口数"]:6d}  占位率 {r["占位率"]:.4f}  '
              f'p132 {r["p132"]:.4f}  邻接覆盖 lane {r["邻接覆盖_lane"]:.4f} / row {r["邻接覆盖_row"]:.4f}')
    r = res['数据集']['_all_train']
    print(f'  {"全 train":8s} 窗 {r["窗口数"]:6d}  占位率 {r["占位率"]:.4f}  p132 {r["p132"]:.4f}')

    for spec in a.gen:
        label, dirs = spec.split('=', 1)
        dirs = [x for x in dirs.split(',') if x]
        blocks = [load_rolls(x) for x in dirs]
        merged, mprobs = {}, {}
        for b, pr in blocks:
            for k, v in b.items():
                merged.setdefault(k, []).append(v)
            for k, v in pr.items():
                mprobs.setdefault(k, []).append(v)
        merged = {k: np.concatenate(v) for k, v in merged.items()}
        mprobs = {k: np.concatenate(v) for k, v in mprobs.items()}
        conf = {name: pooled_stats(merged.get(name, np.zeros((1, 4, 256), np.int32)),
                                   mprobs.get(name))
                for name in STYLES}
        res['生成'][label] = conf
        print(f'\n=== 生成 {label} ===')
        for name in STYLES:
            r = conf[name]
            ds = res['数据集'][name]
            print(f'  {name:8s} 窗 {r["窗口数"]:3d}  占位率 {r["占位率"]:.4f}（目标 {ds["占位率"]:.4f}）  '
                  f'p132 {r["p132"]:.4f}（数据集 {ds["p132"]:.4f}，差 {r["p132"] - ds["p132"]:+.4f}）'
                  f'  原始@0.5 占位 {r.get("原始占位率@0.5", float("nan")):.4f} / '
                  f'p132 {r.get("原始p132@0.5", float("nan")):.4f}')

    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f'\n写出 {a.out}')


if __name__ == '__main__':
    main()
