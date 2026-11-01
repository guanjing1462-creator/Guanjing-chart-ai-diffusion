#!/usr/bin/env python3
"""用真实 beatLength 重建 1/32 网格数据集（上一版用中位间隔当网格，是错的）。

真实网格：
  1/16 步长 = beatLength / 4
  1/32 步长 = beatLength / 8
窗口：[4, 256] 个 1/32 格（= 32 个 1/16 = 8 小节 4/4）

输出 out/diff-data-true/dataset.npz（X / style / diff），并打印统计：
  占位率、相邻格都填的比例（真实细分工）、偶数格/奇数格占位（检测格纹）
"""
import argparse, glob, os
import numpy as np

WIN = 256
STYLES = ['tech', 'jack', 'speed', 'stamina']


def timing(fn):
    """返回第一个非继承 timing point 的 beatLength。"""
    for ln in open(fn, encoding='utf-8', errors='ignore'):
        p = ln.strip().split(',')
        if len(p) >= 2:
            try:
                bl = float(p[1])
                un = int(p[6]) if len(p) > 6 else 1
            except ValueError:
                continue
            if un == 1 and bl > 0:
                return bl
    return None


def rows_of(fn):
    xs = {64: 0, 192: 1, 320: 2, 448: 3}
    by_t = {}
    for ln in open(fn, encoding='utf-8', errors='ignore'):
        p = ln.split(',')
        if len(p) < 5:
            continue
        try:
            x, t, typ = int(p[0]), int(p[2]), int(p[3])
        except ValueError:
            continue
        if x not in xs:
            continue
        if typ & 1 or typ & 128:
            by_t.setdefault(t, set()).add(xs[x])
    return sorted((t, sorted(c)) for t, c in by_t.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='out/diff-data-true/dataset.npz')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    X, S, D = [], [], []
    stats = {'charts': 0, 'variable_bpm': 0}
    for si, style in enumerate(STYLES):
        files = sorted(glob.glob(f'out/packs/{style}/*.osu'))
        if a.limit:
            files = files[:a.limit]
        for f in files:
            bl = timing(f)
            if not bl:
                continue
            # 变速检测：统计非继承 timing point 数量
            nred = sum(1 for ln in open(f, encoding='utf-8', errors='ignore')
                       if ln.count(',') >= 2 and ln.strip().split(',')[-1].strip() == '1')
            if nred > 3:
                stats['variable_bpm'] += 1
                continue
            rows = rows_of(f)
            if len(rows) < 64:
                continue
            s32 = bl / 8.0
            if s32 <= 5:
                continue
            nps = len(rows) / max(1e-6, (rows[-1][0] - rows[0][0]) / 1000.0)
            db = 0 if nps < 8 else 1 if nps < 12 else 2 if nps < 16 else 3
            span = int(round((rows[-1][0] - rows[0][0]) / s32)) + 1
            if span < WIN:
                continue
            grid = np.zeros((4, span), dtype=np.uint8)
            for t, cs in rows:
                k = int(round(t / s32))
                if 0 <= k < span:
                    for c in cs:
                        grid[c, k] = 1
            for st in range(0, span - WIN, WIN // 2):
                w = grid[:, st:st + WIN]
                if w.sum() < 16:
                    continue
                X.append(w); S.append(si); D.append(db)
            stats['charts'] += 1
    X = np.stack(X); S = np.array(S, np.uint8); D = np.array(D, np.uint8)
    np.savez_compressed(a.out, X=X, style=S, diff=D)
    adj = (X[:, :, :-1] & X[:, :, 1:]).any(axis=1).mean()
    print(f'数据集（真实 beatLength 网格）：{X.shape[0]} 窗口 [4,{WIN}]')
    print(f'  来源谱面 {stats["charts"]} 张（跳过变速 {stats["variable_bpm"]} 张）')
    print(f'  占位率 {X.mean():.3f}')
    print(f'  相邻两格都填 {adj:.3f}（旧中位间隔版 0.012）')
    print(f'  偶数格 {X[:, :, ::2].mean():.3f} / 奇数格 {X[:, :, 1::2].mean():.3f}')
    print(f'  风格 {np.bincount(S)}  难度 {np.bincount(D)}')


if __name__ == '__main__':
    main()
