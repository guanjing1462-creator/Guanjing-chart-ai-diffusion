#!/usr/bin/env python3
"""把人类谱转成钢琴卷窗口数据集，供扩散模型训练。

表示：4 键 × 128 行（1/16 网格），二值。
条件：风格（4 类）+ 难度档（按音符密度分 4 档）
输出：out/diff-data/dataset.npz
"""
import argparse, glob, json, os
import numpy as np

WIN = 128          # 128 个 1/16 = 8 小节
STYLES = ['tech', 'jack', 'speed', 'stamina']
SID = {s: i for i, s in enumerate(STYLES)}


def parse_roll(path):
    """返回 [(time_ms, set(cols))]。"""
    xs = {64: 0, 192: 1, 320: 2, 448: 3}
    by_t = {}
    for ln in open(path, encoding='utf-8', errors='ignore'):
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


def roll_from_rows(rows, step_ms, offset_ms, n=WIN):
    """把行转成 [4, n] 的 0/1 矩阵（对齐到网格起点）。"""
    m = np.zeros((4, n), dtype=np.float32)
    for t, cols in rows:
        k = int(round((t - offset_ms) / step_ms))
        if 0 <= k < n:
            for c in cols:
                m[c, k] = 1.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='out/diff-data/dataset.npz')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    # 每张谱的"1/16 步长"没有真实 BPM，用相邻间隔的中位数估算网格
    xs_list, style_list, diff_list = [], [], []
    for style in STYLES:
        files = sorted(glob.glob(f'out/packs/{style}/*.osu'))
        if a.limit:
            files = files[:a.limit]
        for f in files:
            rows = parse_roll(f)
            if len(rows) < WIN:
                continue
            gaps = np.diff([t for t, _ in rows]).astype(float)
            gaps = gaps[(gaps > 40) & (gaps < 400)]
            if len(gaps) < 20:
                continue
            step = float(np.median(gaps))
            nps = len(rows) / max(1e-6, (rows[-1][0] - rows[0][0]) / 1000.0)
            # 难度档：按音符率四分
            dbucket = 0 if nps < 8 else 1 if nps < 12 else 2 if nps < 16 else 3
            # 滑窗切片
            for start in range(0, len(rows) - 32, WIN // 2):
                seg = rows[start:start + 400]
                base_t = seg[0][0]
                # 该窗口内够 32 行就切
                sub = [r for r in seg if r[0] < base_t + step * WIN]
                if len(sub) < 32:
                    continue
                m = roll_from_rows(sub, step, base_t)
                if m.sum() < 12:      # 太空的窗口不要
                    continue
                xs_list.append(m)
                style_list.append(SID[style])
                diff_list.append(dbucket)
    X = np.stack(xs_list).astype(np.uint8)
    S = np.array(style_list, dtype=np.uint8)
    D = np.array(diff_list, dtype=np.uint8)
    np.savez_compressed(a.out, X=X, style=S, diff=D)
    print(f'数据集：{X.shape[0]} 个窗口  {X.shape[1]}键 × {X.shape[2]}行')
    print(f'  风格分布 {np.bincount(S)}   难度分布 {np.bincount(D)}')
    print(f'  平均占位率 {X.mean():.3f}')
    print('  保存到', a.out)


if __name__ == '__main__':
    main()
