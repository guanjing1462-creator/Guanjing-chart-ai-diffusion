#!/usr/bin/env python3
"""把人类谱转成 **1/32 网格** 的钢琴卷窗口数据集，供扩散模型训练。

与 tools/diff-data.py 的关系
---------------------------
解析逻辑完全一致（同一个 parse_roll：只认 4 键 x=64/192/320/448、type&1|type&128、
按时间聚成「行」；二值钢琴卷；风格 4 类 + 按音符率分的 4 档难度），只改一件事：

    网格步长 = 估计出的 1/16 步长的一半（即 1/32），窗口 = [4, 256] 格，
    覆盖的音乐时长与旧的 [4, 128]@1/16 完全相同（256 × step/2 = 128 × step）。

这样做的直接目的：旧数据集在 1/16 网格上**根本没有半格位置**，所以扩散模型
输出里 1/32 占比恒为 0（人类 0.300、规则生成器 0.5~0.7），Koishi 就判「没有密集
连打」→ 难度 < 11（BelowScope）。换到 1/32 网格后，半格位置变成合法格点，
模型才有可能学会 1/32。

步长估计（与旧脚本同一把尺子）
-----------------------------
没有可信 BPM 时，旧脚本用该谱相邻行间隔（40~400ms）的中位数当 1/16 步长；
本脚本原样沿用这个估计 `step16`，然后取 `step32 = step16 / 2`。
（实测 60 张谱的中位数 / (60000/bpm/4) 中位数 = 1.00，即该估计基本等于真实 1/16。）

输出：out/diff-data32/dataset.npz，键名与旧数据一致
-------------------------------------------------
    X     [N,4,256] uint8   二值钢琴卷
    style [N] uint8         0=tech 1=jack 2=speed 3=stamina
    diff  [N] uint8         按音符率 4 档（<8 / 8~12 / 12~16 / >16 nps）

报告里给 1/32 占比的两种口径
--------------------------
  p132_gap  ：相邻占用行间隔 = **1 个 1/32 格**（= 半个 1/16）的比例。
              与 tools/eval-generated.py 的 `p132` 完全同口径（人类 0.300），
              旧数据集按定义恒为 0（1/16 网格上不存在半格间隔）。
  off_parity：占用格中「偏离多数奇偶性」的比例 —— 即真正落在 1/16 之间半格上的
              内容占比（奇偶性按窗口内多数派归一，避免窗口起点落在半格上时整体翻转）。

用法
----
  python3 tools/diff-data32.py                      # 全量 4 个包
  python3 tools/diff-data32.py --limit 20           # 每个包只取前 20 张（冒烟）
  python3 tools/diff-data32.py --out out/diff-data32/dataset.npz
"""
import argparse
import glob
import json
import os

import numpy as np

STEP_DIV = 2       # 1/32 = 1/16 的一半
WIN = 256          # 256 个 1/32 格 = 128 个 1/16 格 = 8 小节（与旧窗口同时长）
STYLES = ['tech', 'jack', 'speed', 'stamina']
SID = {s: i for i, s in enumerate(STYLES)}
LANE_X = {64: 0, 192: 1, 320: 2, 448: 3}


def parse_roll(path):
    """返回 [(time_ms, [cols])]，与 tools/diff-data.py 的 parse_roll 逐行一致。"""
    by_t = {}
    for ln in open(path, encoding='utf-8', errors='ignore'):
        p = ln.split(',')
        if len(p) < 5:
            continue
        try:
            x, t, typ = int(p[0]), int(p[2]), int(p[3])
        except ValueError:
            continue
        c = LANE_X.get(x)
        if c is None:
            continue
        if typ & 1 or typ & 128:
            by_t.setdefault(t, set()).add(c)
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


def gap_stats(m):
    """单窗口的行间隔统计（单位 = 1/32 格）。

    返回 (占用行索引数组, 间隔数组的 1/32 格数)。
    """
    occ = np.where(m.sum(axis=0) > 0)[0]
    if len(occ) < 2:
        return occ, np.array([], dtype=float)
    return occ, np.diff(occ).astype(float)


def parity_stats(m):
    """返回 (多数奇偶性, 偏离多数奇偶性的占用格数, 占用格数)。

    占用格 k 的奇偶性决定了它是 1/16 位置（偶数，若窗口起点在 1/16 上）还是
    1/16 之间的半格（奇数）。窗口起点 base_t 是某个真实音符，可能本身就在半格上，
    所以用窗口内的多数派当基准，只看「少数派」的比例。
    """
    occ = np.where(m.sum(axis=0) > 0)[0]
    if len(occ) == 0:
        return 0, 0, 0
    n_odd = int((occ % 2 == 1).sum())
    n_even = len(occ) - n_odd
    if n_odd <= n_even:
        return 0, n_odd, len(occ)          # 多数派是偶数，奇数格 = 半格内容
    return 1, n_even, len(occ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='每个风格包只取前 N 张（冒烟）')
    ap.add_argument('--out', default='out/diff-data32/dataset.npz')
    ap.add_argument('--stride', type=int, default=64,
                    help='滑窗步长（单位=音符行）。旧脚本 WIN//2=64；这里保持 64 行，'
                         '即同样的时间重叠度 → 窗口数与旧数据集可比')
    ap.add_argument('--seg-rows', type=int, default=512,
                    help='每个窗口最多看多少行原始音符（旧脚本 400，1/32 下留点余量）')
    ap.add_argument('--min-rows', type=int, default=128,
                    help='整张谱至少多少行才用（旧脚本要求 >= WIN=128）')
    ap.add_argument('--min-fill', type=int, default=20,
                    help='窗口内 1 的个数下限（旧脚本 12；1/32 格数是旧的两倍，取 20）')
    ap.add_argument('--old', default='out/diff-data/dataset.npz',
                    help='用于对比的旧数据集（只读）')
    ap.add_argument('--report', default='out/diff-data32/report.json')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)

    xs_list, style_list, diff_list = [], [], []
    # 诊断
    n_files = n_skip_rows = n_skip_gaps = 0
    occ_all = []            # 每个窗口的占位率
    gap1 = gap_tot = 0      # p132_gap：间隔 == 1 格 / 间隔总数
    off_par = par_tot = 0   # off_parity：偏离多数奇偶性的格 / 格总数
    per_style = {}
    gap_hist = {}
    steps = []              # 每张谱的估计 step32（ms）
    resid = []              # 量化残差（占一格的比例）
    nper = []               # 每窗口音符数

    for style in STYLES:
        files = sorted(glob.glob(f'out/packs/{style}/*.osu'))
        if a.limit:
            files = files[:a.limit]
        cnt = 0
        for f in files:
            n_files += 1
            rows = parse_roll(f)
            if len(rows) < a.min_rows:
                n_skip_rows += 1
                continue
            gaps = np.diff([t for t, _ in rows]).astype(float)
            gaps = gaps[(gaps > 40) & (gaps < 400)]
            if len(gaps) < 20:
                n_skip_gaps += 1
                continue
            step16 = float(np.median(gaps))
            step = step16 / STEP_DIV          # ← 1/32 格
            steps.append(step)
            nps = len(rows) / max(1e-6, (rows[-1][0] - rows[0][0]) / 1000.0)
            dbucket = 0 if nps < 8 else 1 if nps < 12 else 2 if nps < 16 else 3

            # 量化残差诊断：音符相对窗口起点落在网格上的偏差（单位=格）
            t0 = rows[0][0]
            kk = np.round((np.array([t for t, _ in rows]) - t0) / step)
            rr = np.abs((np.array([t for t, _ in rows]) - t0) / step - kk)
            resid.append(float(np.median(rr)))

            for start in range(0, len(rows) - 32, a.stride):
                seg = rows[start:start + a.seg_rows]
                base_t = seg[0][0]
                sub = [r for r in seg if r[0] < base_t + step * WIN]
                if len(sub) < 32:
                    continue
                m = roll_from_rows(sub, step, base_t)
                if m.sum() < a.min_fill:      # 太空的窗口不要
                    continue
                xs_list.append(m)
                style_list.append(SID[style])
                diff_list.append(dbucket)
                cnt += 1

                # --- 统计 ---
                nz = float(m.mean())
                occ_all.append(nz)
                nper.append(float(m.sum()))
                _, g = gap_stats(m)
                if g.size:
                    gap_tot += int(g.size)
                    gap1 += int((g == 1).sum())
                    for u in g:
                        key = ('1/32' if u == 1 else '1/16' if u == 2 else
                               '1/8' if u == 4 else '1/4' if u == 8 else
                               '1/32*2' if u < 1 else '其它(非整除)')
                        gap_hist[key] = gap_hist.get(key, 0) + 1
                _, off, tot = parity_stats(m)
                off_par += off
                par_tot += tot
        per_style[style] = cnt

    if not xs_list:
        raise SystemExit('[错误] 一个窗口都没切出来，检查 --min-fill / --min-rows')

    X = np.stack(xs_list).astype(np.uint8)
    S = np.array(style_list, dtype=np.uint8)
    D = np.array(diff_list, dtype=np.uint8)
    np.savez_compressed(a.out, X=X, style=S, diff=D)

    occ = float(X.mean())
    p132_gap = gap1 / max(1, gap_tot)
    off_parity = off_par / max(1, par_tot)
    rep = {
        '数据': a.out,
        '窗口数': int(X.shape[0]),
        '形状': list(X.shape),
        '占位率': round(occ, 4),
        'p132_gap': round(p132_gap, 4),
        'off_parity': round(off_parity, 4),
        '间隔直方图(单位1/32格)': {k: round(v / max(1, gap_tot), 4)
                                for k, v in sorted(gap_hist.items())},
        '风格分布': np.bincount(S, minlength=4).tolist(),
        '难度分布': np.bincount(D, minlength=4).tolist(),
        '每个风格的窗口': per_style,
        '每窗口音符数_均值': round(float(np.mean(nper)), 2),
        '估计step32_ms_中位数': round(float(np.median(steps)), 2),
        '量化残差_中位数': round(float(np.median(resid)), 4),
        '跳过_行数不足': n_skip_rows,
        '跳过_间隔不足': n_skip_gaps,
        '用到的谱数': n_files,
        '滑窗步长': a.stride,
    }

    # ---------------- 与旧数据集对比（只读）----------------
    cmp_rows = []
    if os.path.exists(a.old):
        od = np.load(a.old)
        OX = od['X']
        old_occ = float(OX.mean())
        # 旧数据是 1/16 网格：间隔必为整数格 >=1，半格间隔按定义 = 0
        old_p132 = 0.0
        cmp_rows = [
            {'数据集': '旧 ' + os.path.basename(os.path.dirname(a.old)),
             '网格': '1/16', '窗口': [int(OX.shape[1]), int(OX.shape[2])],
             '窗口数': int(OX.shape[0]), '占位率': round(old_occ, 4),
             'p132_gap': old_p132, '1/32 可表示': '否'},
            {'数据集': '新 diff-data32', '网格': '1/32',
             '窗口': [int(X.shape[1]), int(X.shape[2])],
             '窗口数': int(X.shape[0]), '占位率': round(occ, 4),
             'p132_gap': round(p132_gap, 4), '1/32 可表示': '是'},
        ]
        rep['对比旧数据集'] = cmp_rows
        rep['旧数据集占位率'] = round(old_occ, 4)

    with open(a.report, 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)

    print(f'数据集：{X.shape[0]} 个窗口  {X.shape[1]}键 × {X.shape[2]}行（1/32 网格）')
    print(f'  风格分布 {np.bincount(S, minlength=4).tolist()}   '
          f'难度分布 {np.bincount(D, minlength=4).tolist()}')
    print(f'  平均占位率 {occ:.4f}')
    print(f'  1/32 占比 p132_gap = {p132_gap:.4f}（间隔=1 个 1/32 格的相邻行比例）')
    print(f'  半格内容 off_parity = {off_parity:.4f}（落在 1/16 之间的占用格比例）')
    print(f'  间隔直方图 ' + '  '.join(f'{k}:{v/gap_tot:.3f}'
                                   for k, v in sorted(gap_hist.items())))
    print(f'  估计 step32 中位数 {np.median(steps):.2f}ms，量化残差中位数 '
          f'{np.median(resid):.3f} 格')
    print(f'  用谱 {n_files} 张（跳行数不足 {n_skip_rows} / 间隔不足 {n_skip_gaps}）')
    if cmp_rows:
        print('  对比旧数据集：')
        for r in cmp_rows:
            print(f"    {r['数据集']:18s} 网格{r['网格']:5s} 窗口{r['窗口']} "
                  f"窗口数 {r['窗口数']:6d} 占位率 {r['占位率']:.4f} "
                  f"p132 {r['p132_gap']:.4f}")
    print('  保存到', a.out)
    print('  报告 →', a.report)


if __name__ == '__main__':
    main()
