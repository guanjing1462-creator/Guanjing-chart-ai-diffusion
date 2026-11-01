#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round138 · 采样卷统计（占位率 / p132 / 难度条件单调性）。

只读采样脚本落盘的 `rolls-s*dd-*.npz`（X=[n,4,256] 二值卷、P=概率卷），
按「格口径」算占位率与 p132，与数据集报告（占位率 0.1986 / p132 0.2637）对齐。

p132 口径（与 tools/diff-data32-grouped.py 的 gap_p132 / sample-diffusion32c.py 的
roll_stats 一致）：窗口内相邻占用行（至少 1 个键有音符的行）间隔 == 1 格的比例，
跨窗口池化。数据集另有「毫秒口径」0.2545（间隔 / step16 ∈ [0.35,0.75)）；
生成卷是严格等格长、无 timing 抖动，1 格间隔恒等于 0.5 step16，因此两个口径在
生成侧**必然相等**，毫秒口径不提供额外信息 —— 这一点写进报告，不能拿它冒充差异。

用法：
  python3 tools/round138-sample-stats.py --dir /tmp/r138-samples \
      --out out/diffusion/round138-train/sample-stats.json
"""
import argparse, glob, json, os, re
import numpy as np

DATASET = {'占位率': 0.1986, 'p132': 0.2637, '窗口数': 95574,
           '每split占位率': {'train': 0.1985, 'val': 0.1845, 'test': 0.2195},
           '每split_p132': {'train': 0.26, 'val': 0.2893, 'test': 0.3175}}


def roll_stats(X):
    """X: [n,4,L] 0/1 → 占位率、池化 p132、非空行比例、每窗音符数。"""
    n = X.shape[0]
    row_sum = X.sum(axis=1)                      # [n,L]
    occ = float(X.mean())
    gaps1 = tot = 0
    nonempty = float((row_sum > 0).mean())
    for i in range(n):
        oc = np.where(row_sum[i] > 0)[0]
        if len(oc) < 2:
            continue
        d = np.diff(oc)
        gaps1 += int((d == 1).sum())
        tot += int(d.size)
    return {'窗口数': int(n), '占位率': round(occ, 4),
            'p132': round(gaps1 / max(1, tot), 4),
            'p132_间隔数': int(tot),
            '非空行比例': round(nonempty, 4),
            '每窗音符数': round(float(X.sum(axis=(1, 2)).mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='采样输出目录（含 rolls-*.npz）')
    ap.add_argument('--out', default='', help='统计 json 输出路径')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, 'rolls-s*d*.npz')))
    if not files:
        raise SystemExit(f'[错误] {a.dir} 里没有 rolls-*.npz')
    rows = []
    for f in files:
        d = np.load(f)
        X = d['X'].astype(np.uint8)
        st = roll_stats(X)
        st['style'] = int(d['style'])
        st['diff'] = float(d['diff'])
        st['阈值'] = float(d['threshold']) if 'threshold' in d else None
        st['文件'] = os.path.basename(f)
        rows.append(st)

    rows.sort(key=lambda r: (r['style'], r['diff']))
    print('=== 逐条件采样统计（格口径）===')
    print(f'{"style":>5} {"diff":>5} {"占位率":>8} {"p132":>7} {"非空行":>7} {"每窗音符":>8}')
    for r in rows:
        print(f'{r["style"]:>5} {r["diff"]:>5.1f} {r["占位率"]:>8.4f} {r["p132"]:>7.4f} '
              f'{r["非空行比例"]:>7.4f} {r["每窗音符数"]:>8.1f}')

    # 难度条件单调性：每个 style 上 0.1 → 0.5 → 0.7 占位率/每窗音符数是否递增
    mono = {}
    for s in sorted({r['style'] for r in rows}):
        seq = [r for r in rows if r['style'] == s]
        seq.sort(key=lambda r: r['diff'])
        occ = [r['占位率'] for r in seq]
        nps = [r['每窗音符数'] for r in seq]
        mono[s] = {'diff': [r['diff'] for r in seq], '占位率': occ, '每窗音符数': nps,
                   '占位率递增': all(b > a_ for a_, b in zip(occ, occ[1:])),
                   '每窗音符数递增': all(b >= a_ for a_, b in zip(nps, nps[1:]))}
    ok = sum(1 for s in mono if mono[s]['占位率递增'])
    print(f'\n占位率单调（diff 0.1→0.5→0.7 严格递增）：{ok}/{len(mono)} 个 style')
    for s, m in mono.items():
        print(f'  style {s}: 占位率 {m["占位率"]}  每窗音符 {m["每窗音符数"]}  '
              f'递增={m["占位率递增"]}')

    allX = np.concatenate([np.load(f)['X'].astype(np.uint8) for f in files])
    pooled = roll_stats(allX)
    print(f'\n全部 {len(files)} 个条件 × {pooled["窗口数"]} 窗池化：占位率 {pooled["占位率"]:.4f}'
          f' / p132 {pooled["p132"]:.4f}')
    print(f'数据集口径：占位率 {DATASET["占位率"]} / p132 {DATASET["p132"]}（格口径，95,574 窗）')
    print(f'  差（采样-数据集）：占位率 {pooled["占位率"] - DATASET["占位率"]:+.4f}'
          f' / p132 {pooled["p132"] - DATASET["p132"]:+.4f}')

    res = {'逐条件': rows, '单调性': mono, '池化': pooled,
           '数据集': DATASET,
           '口径说明': '占位率=X 全均值；p132=相邻占用行间隔==1 格的池化比例；'
                      '生成卷等格长，毫秒口径(p132_ms)与格口径恒等，不另算',
           '采样目录': os.path.abspath(a.dir)}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
