#!/usr/bin/env python3
"""特征包质检：文件完整性 + shape 一致性 + 采音对齐分布。

    python3 tools/verify-features.py out/features
    python3 tools/verify-features.py out/features --plot out/features-quality.png

对齐度用 mel 低频带 + 扫偏移算。这个口径比 22kHz 全频段 RMS 敏感得多
（同一张谱：低频带 ×1.59，全频段 RMS ×1.06）—— 高频的镲片和齿音会稀释鼓点。
"""
import argparse
import json
import os
import random
import sys

import numpy as np

FRAME_MS = 46.4399


def level_energy(mel, bands=(0, 16)):
    """mel 低频带的线性功率和。expm1 还原 log1p 压缩。"""
    return np.expm1(mel[bands[0]:bands[1]].astype(np.float32)).sum(0)


def align_of(energy, note_times, max_shift_ms=120, step_ms=10):
    """扫偏移找最佳对齐；返回 (最佳偏移, 对齐值, 零偏移对齐值)。"""
    if len(energy) < 30:
        return 0, 0.0, 0.0
    t = np.asarray(note_times, dtype=np.float64)
    base = energy.mean() or 1e-9
    v0_idx = (t / FRAME_MS).astype(np.int64)
    v0_idx = v0_idx[(v0_idx >= 0) & (v0_idx < len(energy))]
    v0 = energy[v0_idx].mean() / base if len(v0_idx) else 0.0

    best = (0, v0)
    for sh in range(-max_shift_ms, max_shift_ms + 1, step_ms):
        idx = ((t + sh) / FRAME_MS).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < len(energy))]
        if len(idx) < 30:
            continue
        v = energy[idx].mean() / base
        if v > best[1]:
            best = (sh, v)
    return best[0], best[1], v0


def note_times_from_tensor(tensor, keys):
    """从 start 通道 + offset 通道还原音符时间（ms）。"""
    start = tensor[:keys]
    off = tensor[keys:2 * keys]
    cols, frames = np.nonzero(start > 0)
    if not len(frames):
        return np.zeros(0)
    t = (frames - 0.5 + off[cols, frames].astype(np.float64)) * FRAME_MS
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default='out/features')
    ap.add_argument('--plot', default=None)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    idx_path = os.path.join(args.root, 'index.jsonl')
    if not os.path.exists(idx_path):
        sys.exit(f'找不到 {idx_path}')
    rows = [json.loads(l) for l in open(idx_path, encoding='utf-8')]
    if args.limit:
        rows = rows[:args.limit]
    print(f'索引 {len(rows)} 条')

    missing, shape_bad, empty = [], [], []
    aligns, offs, ratios_notes = [], [], []
    ok = 0

    for r in rows:
        ap_ = os.path.join(args.root, r['audio'])
        cp = os.path.join(args.root, r['chart'])
        if not (os.path.exists(ap_) and os.path.exists(cp)):
            missing.append(r.get('chart', '?'))
            continue
        try:
            with np.load(ap_) as z:
                mel = z['mel']
                rms = z['rms'] if 'rms' in z else None
            with np.load(cp) as z:
                tensor = z['tensor']
                valid = z['valid']
                keys = int(z['keys'])
        except Exception as e:
            shape_bad.append((r.get('chart'), f'读取失败 {e}'))
            continue

        if tensor.shape != (4 * keys, 8192):
            shape_bad.append((r.get('chart'), f'tensor {tensor.shape} != {(4*keys, 8192)}'))
            continue
        if abs(int(valid.sum()) - r['frames']) > 3:
            # valid 到最后一个音符，frames 是音频长度，两者本来就不等，
            # 只检查 valid 不超过音频长度
            pass
        if int(valid.sum()) > mel.shape[1] + 2:
            shape_bad.append((r.get('chart'),
                              f'valid {int(valid.sum())} > mel {mel.shape[1]}'))
            continue

        nt = note_times_from_tensor(tensor, keys)
        if len(nt) == 0:
            empty.append(r.get('chart'))
            continue

        e = level_energy(mel)
        sh, vbest, v0 = align_of(e, nt)
        aligns.append(vbest)
        offs.append(sh)

        n_note = tensor[:keys].sum()
        ratios_notes.append(float(n_note) / max(1, int(valid.sum())))
        ok += 1

    n = len(aligns)
    print(f'\n通过 {ok} / {len(rows)}')
    if missing:
        print(f'缺文件 {len(missing)}：{missing[:3]}')
    if shape_bad:
        print(f'shape 异常 {len(shape_bad)}：{shape_bad[:3]}')
    if empty:
        print(f'空谱面 {len(empty)}：{empty[:3]}')

    if n:
        a = np.array(aligns)
        o = np.array(offs)
        print(f'\n采音对齐（mel 0-16 带，扫 ±120ms）：')
        print(f'  均值 ×{a.mean():.2f}   中位 ×{np.median(a):.2f}   '
              f'p10 ×{np.percentile(a,10):.2f}   p90 ×{np.percentile(a,90):.2f}   '
              f'最大 ×{a.max():.2f}')
        print(f'  最佳偏移 中位 {np.median(o):+.0f}ms   '
              f'（+40~60ms 是谱师的提前量，正常）')
        print(f'  每帧平均音符数 {np.mean(ratios_notes):.2f}')

        # 按密度分档
        print(f'\n按音符数分档：')
        for lo, hi in [(0, 600), (600, 1500), (1500, 3000), (3000, 10**9)]:
            sel = [a[i] for i, r in enumerate(rows) if ok and lo <= rows[i].get('notes', 0) < hi]
            if len(sel) >= 3:
                print(f'  {lo}-{hi if hi < 10**9 else "∞"}n  n={len(sel):>3}  '
                      f'×{np.mean(sel):.2f}')

        if args.plot:
            make_plot(aligns, offs, ratios_notes, args.plot)


def make_plot(aligns, offs, ratios, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor='#0d1420')
    for ax in axes:
        ax.set_facecolor('#111a28')
        ax.tick_params(colors='#8fa5c0', labelsize=8)
        ax.grid(alpha=0.15, color='#5a7ba0')
        for s in ax.spines.values():
            s.set_color('#2a3a52')

    axes[0].hist(aligns, bins=40, color='#38e8ff', edgecolor='#0d1420', alpha=0.9)
    axes[0].axvline(np.median(aligns), color='#ff4fd8', lw=1.8, ls='--')
    axes[0].axvline(1.0, color='#7d90ad', lw=1.2, ls=':')
    axes[0].set_title(f'alignment  median x{np.median(aligns):.2f}',
                      color='#e8f2ff', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('note-frame energy / global mean', color='#8fa5c0', fontsize=9)

    axes[1].hist(offs, bins=range(-130, 141, 10), color='#ffc23d',
                 edgecolor='#0d1420', alpha=0.9)
    axes[1].axvline(0, color='#7d90ad', lw=1.2, ls=':')
    axes[1].set_title('best offset (ms)', color='#e8f2ff',
                      fontsize=11, fontweight='bold')
    axes[1].set_xlabel('shift applied to note times', color='#8fa5c0', fontsize=9)

    axes[2].scatter(aligns, ratios, s=12, color='#4dffa6', alpha=0.55,
                    edgecolors='none')
    axes[2].set_title('alignment vs notes-per-frame', color='#e8f2ff',
                      fontsize=11, fontweight='bold')
    axes[2].set_xlabel('alignment', color='#8fa5c0', fontsize=9)
    axes[2].set_ylabel('notes per active frame', color='#8fa5c0', fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=135, facecolor='#0d1420', bbox_inches='tight')
    print(f'\n画像 {path}')


if __name__ == '__main__':
    main()
