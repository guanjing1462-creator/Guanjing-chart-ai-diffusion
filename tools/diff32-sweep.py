#!/usr/bin/env python3
"""1/32 采样结果的阈值扫描 + 奇偶格概率诊断（一次性诊断脚本）。

读 sample-diffusion32.py 存下的 rolls-*.npz（含 P = 未阈值化概率 [n,4,256]），
输出每个阈值下的占位率 / p132 / 非空行比例 / 每行音符数，以及奇偶格的平均概率。

用法：python3 tools/diff32-sweep.py out/diffusion32/samples/rolls-*.npz [--json out.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

DATA_OCC = 0.2145      # 1/32 训练集占位率（decode 阈值的一个自然锚点）


def stats_multi(X, P):
    """X: [n,4,W] 0/1；P: [n,4,W]。返回跨窗口聚合的指标（p132 按 gap 总数加权，
    与 sample-diffusion32.py / eval-generated.py 的口径一致，不能只看单窗口）。"""
    n = X.shape[0]
    gap1 = gap_tot = 0
    per_p132 = []
    for i in range(n):
        occ = np.where(X[i].sum(axis=0) > 0)[0]
        if len(occ) < 2:
            per_p132.append(0.0); continue
        g = np.diff(occ)
        gap1 += int((g == 1).sum()); gap_tot += int(g.size)
        per_p132.append(round(float((g == 1).mean()), 3))
    rows_nz = float((X.sum(axis=1) > 0).mean())
    notes = int(X.sum())
    npr = notes / max(1, int((X.sum(axis=1) > 0).sum()))
    odd = np.arange(X.shape[2]) % 2 == 1
    return {
        '占位率': round(float(X.mean()), 4),
        '每窗口占位率': [round(float(X[i].mean()), 3) for i in range(n)],
        'p132': round(gap1 / max(1, gap_tot), 4),
        'p132_每窗口': per_p132,
        '非空行比例': round(rows_nz, 4),
        '非空行每行音符数': round(npr, 3),
        '音符数': notes,
        '每窗口音符数': [int(X[i].sum()) for i in range(n)],
        'P均值_奇格': round(float(P[:, :, odd].mean()), 4),
        'P均值_偶格': round(float(P[:, :, ~odd].mean()), 4),
    }


def stats(X, P):
    """X: [4,W] 0/1；P: [4,W]。返回单窗口指标 dict（保留给 --emit 用）。"""
    occ = np.where(X.sum(axis=0) > 0)[0]
    rowsum = X.sum(axis=0)
    nz_rows = float((rowsum > 0).mean())
    npr = float(X.sum() / max(1, int((rowsum > 0).sum())))
    out = {
        '占位率': round(float(X.mean()), 4),
        '非空行比例': round(nz_rows, 4),
        '非空行每行音符数': round(npr, 3),
        '音符数': int(X.sum()),
    }
    if len(occ) > 1:
        g = np.diff(occ)
        out['p132'] = round(float((g == 1).mean()), 4)
        out['gap_中位_us'] = int(np.median(g)) * 55   # 以 55.03ms/格 记
        hist = {}
        for u in g:
            k = f'{int(u)}格'
            hist[k] = hist.get(k, 0) + 1
        out['gap_hist'] = {k: round(v / len(g), 3) for k, v in sorted(
            hist.items(), key=lambda z: int(z[0][:-1]))[:8]}
        # 连打段长度分布（连续被占用的格数）
        runs, cur = [], 0
        for k in range(X.shape[1]):
            if rowsum[k] > 0:
                cur += 1
            elif cur:
                runs.append(cur); cur = 0
        if cur:
            runs.append(cur)
        out['连续段长度均值'] = round(float(np.mean(runs)), 2) if runs else 0.0
        out['连续段占比_>=2'] = round(float((np.array(runs) >= 2).mean()), 3) if runs else 0.0
    else:
        out['p132'] = 0.0
    # 奇偶格概率（按窗口内多数占用奇偶性归一）
    odd = np.arange(X.shape[1]) % 2 == 1
    p_odd = float(P[:, odd].mean())
    p_even = float(P[:, ~odd].mean())
    out['P均值_奇格'] = round(p_odd, 4)
    out['P均值_偶格'] = round(p_even, 4)
    return out


def decode(P, mode, t, rng):
    """把 [n,4,W] 概率卷解成二值卷。

    threshold  —— X = P > t（传统做法；对「条件概率 < t 的稀有结构」会系统性丢掉）
    bernoulli  —— X ~ Bernoulli(P)，逐格独立抽样。
                  这是二值钢琴卷的标准采样口径：MSE 训练下 p̂ 就是 P(x0=1|上下文)，
                  按它抽样才是「按模型学到的概率生成」，阈值化只是它的硬判决近似。
    """
    if mode == 'bernoulli':
        return (rng.random(P.shape) < P).astype(np.uint8)
    return (P > t).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', help='rolls-*.npz（由 sample-diffusion32.py 写出）')
    ap.add_argument('--json', default='')
    ap.add_argument('--emit-dir', default='',
                    help='把候选解码结果存成 .npy（[n,4,W] uint8），便于直接喂 roll-to-osu32.py')
    ap.add_argument('--decode', choices=['threshold', 'bernoulli'], default='',
                    help='除扫描外，额外用一种解码整体导出（配合 --decode-threshold）')
    ap.add_argument('--decode-threshold', type=float, default=0.3)
    ap.add_argument('--tag', default='x')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    z = np.load(a.npz)
    P = z['P'].astype(np.float32)                  # [n,4,W]
    n = P.shape[0]
    print(f'文件 {a.npz}  窗口 {n}  P 形状 {P.shape}')
    p = P.reshape(-1)
    print(f'P 分布: 均值 {p.mean():.4f}  sd {p.std():.4f}  q10 {np.quantile(p,0.1):.3f} '
          f'q50 {np.quantile(p,0.5):.3f} q90 {np.quantile(p,0.9):.3f} '
          f'q99 {np.quantile(p,0.99):.3f} max {p.max():.3f}')
    print(f'1/32 训练集占位率参考 {DATA_OCC}')

    rows = []
    for t in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        X = (P > t).astype(np.uint8)
        st = stats_multi(X, P)
        rows.append({'阈值': t, **st})
        print(f"t={t:.2f}  占位率 {st['占位率']:.4f}  **p132 {st['p132']:.4f}**  "
              f"非空行 {st['非空行比例']:.3f}  行内音符 {st['非空行每行音符数']:.2f}  "
              f"音符/窗口 {st['每窗口音符数']}")
        print(f'          p132 每窗口 {st["p132_每窗口"]}')
    Xb = (np.random.default_rng(0).random(P.shape) < P).astype(np.uint8)
    st = stats_multi(Xb, P)
    rows.append({'阈值': 'bernoulli', **st})
    print(f"bernoulli  占位率 {st['占位率']:.4f}  **p132 {st['p132']:.4f}**  "
          f"非空行 {st['非空行比例']:.3f}  行内音符 {st['非空行每行音符数']:.2f}  "
          f"音符/窗口 {st['每窗口音符数']}")
    print(f'           p132 每窗口 {st["p132_每窗口"]}')
    # 额外：按指定解码整体导出
    if a.decode and a.emit_dir:
        os.makedirs(a.emit_dir, exist_ok=True)
        rng = np.random.default_rng(a.seed)
        X = np.stack([decode(P[i], a.decode, a.decode_threshold, rng) for i in range(n)])
        st = stats_multi(X, P)
        print(f"\n[导出] decode={a.decode} t={a.decode_threshold} → 占位率 "
              f"{X.mean():.4f} p132 {st['p132']:.4f} 非空行 {st['非空行比例']:.3f}")
        for i in range(n):
            f = os.path.join(a.emit_dir, f'{a.tag}-{i:02d}.npy')
            np.save(f, X[i])
        print(f'  {n} 个窗口 → {a.emit_dir}/{a.tag}-*.npy')
        if a.json:
            with open(a.json, 'w', encoding='utf-8') as fh:
                json.dump({'npz': a.npz, 'decode': a.decode,
                           'threshold': a.decode_threshold,
                           '占位率': round(float(X.mean()), 4),
                           'p132': st['p132'], '扫描': rows,
                           '每窗口p132': st['p132_每窗口']},
                          fh, ensure_ascii=False, indent=1)
            print('JSON →', a.json)
            return

    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump({'npz': a.npz, 'P分布': {'均值': round(float(p.mean()), 4),
                                              'sd': round(float(p.std()), 4),
                                              'q50': round(float(np.quantile(p, 0.5)), 4),
                                              'q90': round(float(np.quantile(p, 0.9)), 4)},
                       'data_occ': DATA_OCC, '扫描': rows}, fh, ensure_ascii=False, indent=1)
        print('JSON →', a.json)


if __name__ == '__main__':
    main()
