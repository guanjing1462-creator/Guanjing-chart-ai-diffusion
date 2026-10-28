#!/usr/bin/env python3
"""把采音对齐基线画成图：对齐度 vs 谱面密度，标出生成器的位置。

用法：
    python3 tools/plot-alignment.py /tmp/osz_x/alignment-baseline.json \
        -o out/alignment-baseline.png --gen 1.45 --gen-grid 1.39
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BINS = [(0, 600, '<600 notes\n(sparse)'), (600, 1500, '600-1500'),
        (1500, 3000, '1500-3000'), (3000, 10 ** 9, '>3000 notes\n(dense)')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json')
    ap.add_argument('-o', '--out', default='out/alignment-baseline.png')
    ap.add_argument('--gen', type=float, default=1.45, help='onset 生成器')
    ap.add_argument('--gen-grid', type=float, default=1.39, help='BPM 网格生成器')
    args = ap.parse_args()

    d = json.load(open(args.json))
    rows = [r for r in d['rows'] if abs(r['offset']) < 140]   # 排除 offset 未收敛的
    x = np.array([r['notes'] for r in rows], dtype=float)
    y = np.array([r['aligned'] for r in rows], dtype=float)
    rnd = d['randomAvg']

    fig = plt.figure(figsize=(15, 6.2), facecolor='#0d1420')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.22)

    # ---------- 左：散点 + 分组均值 ----------
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor('#111a28')
    ax.scatter(x, y, s=14, color='#38e8ff', alpha=0.45, edgecolors='none', label='human chart')

    centers, means, p90s = [], [], []
    for lo, hi, label in BINS:
        m = (x >= lo) & (x < hi)
        if m.sum() == 0:
            continue
        centers.append(np.sqrt(max(lo, 1) * min(hi, 6000)))
        means.append(y[m].mean())
        p90s.append(np.percentile(y[m], 90))

    ax.plot(centers, means, 'o-', color='#ff4fd8', lw=2.2, ms=8,
            label='bin mean', zorder=5)
    ax.plot(centers, p90s, 's--', color='#ffc23d', lw=1.4, ms=5,
            label='bin p90', zorder=5, alpha=0.85)

    ax.axhline(rnd, color='#7d90ad', lw=1.3, ls=':', label=f'random x{rnd:.2f}')
    ax.axhline(args.gen, color='#4dffa6', lw=2, ls='--',
               label=f'onset gen x{args.gen:.2f}')
    ax.axhline(args.gen_grid, color='#9b8cff', lw=1.6, ls='-.',
               label=f'grid gen x{args.gen_grid:.2f}')

    ax.set_xscale('log')
    ax.set_xlabel('notes per chart (log)', color='#c8d8ee', fontsize=10)
    ax.set_ylabel('accent alignment  (note-frame RMS / global mean)',
                  color='#c8d8ee', fontsize=10)
    ax.set_title(f'Accent alignment vs chart density   n={len(rows)} 4K/7K ranked',
                 color='#e8f2ff', fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(colors='#8fa5c0', labelsize=9)
    ax.grid(alpha=0.15, color='#5a7ba0')
    for s in ax.spines.values():
        s.set_color('#2a3a52')
    leg = ax.legend(fontsize=8, loc='upper right', framealpha=0.25,
                    facecolor='#0d1420', edgecolor='#2a3a52')
    for t in leg.get_texts():
        t.set_color('#c8d8ee')

    # ---------- 右：分组均值柱状 + 生成器对照 ----------
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#111a28')
    labels = [b[2] for b in BINS if ((x >= b[0]) & (x < b[1])).sum() > 0]
    vals = [v for v in means]
    colors = ['#38e8ff', '#5ce1ff', '#7fb2ff', '#9b8cff']

    bars = ax2.bar(range(len(vals)), vals, color=colors[:len(vals)],
                   edgecolor='#0d1420', alpha=0.92, width=0.62)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.03, f'x{v:.2f}',
                 ha='center', color='#e8f2ff', fontsize=10, fontweight='bold')
        n = ((x >= BINS[i][0]) & (x < BINS[i][1])).sum()
        ax2.text(b.get_x() + b.get_width() / 2, 0.06, f'n={n}',
                 ha='center', color='#0d1420', fontsize=8)

    ax2.axhline(args.gen, color='#4dffa6', lw=2, ls='--')
    ax2.axhline(rnd, color='#7d90ad', lw=1.2, ls=':')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8, color='#8fa5c0')
    ax2.set_ylabel('mean alignment', color='#c8d8ee', fontsize=10)
    ax2.set_title('Denser chart = lower alignment\n(physics, not a flaw)',
                  color='#e8f2ff', fontsize=11, fontweight='bold', pad=10)
    ax2.set_ylim(0, max(max(vals), args.gen) * 1.22)
    ax2.tick_params(colors='#8fa5c0', labelsize=9)
    ax2.grid(axis='y', alpha=0.15, color='#5a7ba0')
    for s in ax2.spines.values():
        s.set_color('#2a3a52')
    ax2.text(0.98, args.gen + 0.03, f'onset gen x{args.gen:.2f}', ha='right',
             color='#4dffa6', fontsize=8, transform=ax2.get_yaxis_transform())

    plt.tight_layout()
    plt.savefig(args.out, dpi=140, facecolor='#0d1420', bbox_inches='tight')
    print('saved', args.out)

    print(f"\n分组均值：")
    for (lo, hi, label), v in zip([b for b in BINS if ((x >= b[0]) & (x < b[1])).sum() > 0], vals):
        print(f"  {label.split(chr(10))[0]:<18} x{v:.2f}")


if __name__ == '__main__':
    main()
