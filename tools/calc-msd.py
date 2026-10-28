#!/usr/bin/env python3
"""用 MinaCalc（Etterna 官方难度计算器）给谱面算 7 维技能值。

这是 Mug-Diffusion 训练时用的同一套标签（见 scripts/prepare_beatmap_features.py
里的 `import minacalc`）。7 个技能维度就是「键型」的权威定义：
    stream / jumpstream / handstream / stamina / jackspeed / chordjack / technical

用法：
    python3 tools/calc-msd.py out/charts -o out/msd.jsonl
    python3 tools/calc-msd.py out/charts -o out/msd.jsonl --plot out/msd-profile.png

依赖：
    先编译 MinaCalc 共享库（源码在 Mug-Diffusion 的 scripts/MinaCalc-1.0.tar.gz）：
        g++ -std=c++17 -O2 -fPIC -shared -I. -o libminacalc.so wrap.cpp \
            MinaCalc/MinaCalc.cpp
    wrap.cpp 见本仓库 tools/minacalc_wrap.cpp。
"""
import argparse
import ctypes
import glob
import json
import os
import sys

SKILLS = ['overall', 'stream', 'jumpstream', 'handstream',
          'stamina', 'jackspeed', 'chordjack', 'technical']

# Etterna 的判定目标是 93%（"good" 全连），Mug-Diffusion 用默认值
DEFAULT_GOAL = 0.93


def load_lib(path):
    lib = ctypes.CDLL(path)
    lib.mc_calc.restype = ctypes.c_int
    lib.mc_calc.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint),
        ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_float),
    ]
    return lib


def parse_osu(path):
    """返回 (keys, [(time_ms, mask)], n_hold)。同一毫秒的多个音符合并成一行。"""
    keys = 4
    rows = {}
    holds = 0
    sec = None
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('['):
                sec = line
                continue
            if sec == '[Difficulty]' and line.startswith('CircleSize:'):
                try:
                    keys = int(float(line.split(':')[1]))
                except ValueError:
                    pass
            elif sec == '[HitObjects]' and ',' in line and line[:1].isdigit():
                p = line.split(',')
                if len(p) < 4:
                    continue
                try:
                    x = int(float(p[0]))
                    t = int(float(p[2]))
                except ValueError:
                    continue
                if int(float(p[3])) & 128:
                    holds += 1
                col = min(keys - 1, max(0, x * keys // 512))
                rows[t] = rows.get(t, 0) | (1 << col)
    return keys, sorted(rows.items()), holds


def calc(lib, rows, rate=1.0, goal=DEFAULT_GOAL):
    n = len(rows)
    if n == 0:
        return None
    times = (ctypes.c_double * n)(*[t / 1000.0 for t, _ in rows])
    masks = (ctypes.c_uint * n)(*[m for _, m in rows])
    out = (ctypes.c_float * 8)()
    got = lib.mc_calc(times, masks, n, rate, goal, out)
    if got <= 0:
        return None
    return {SKILLS[i]: round(float(out[i]), 2) for i in range(min(8, got))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('charts')
    ap.add_argument('-o', '--out', default='out/msd.jsonl')
    ap.add_argument('--lib', default=os.environ.get('MINACALC_LIB', './libminacalc.so'))
    ap.add_argument('--plot', default=None)
    ap.add_argument('--rate', type=float, default=1.0,
                    help='音乐倍速；1.0 是原速。1.1 对应常见的 1.1x 图')
    args = ap.parse_args()

    if not os.path.exists(args.lib):
        sys.exit(f'找不到 MinaCalc 库：{args.lib}\n先按 README 编译，或用 --lib 指定路径')

    lib = load_lib(args.lib)
    files = sorted(glob.glob(os.path.join(args.charts, '**', '*.osu'), recursive=True))
    if not files:
        sys.exit(f'{args.charts} 下没有 .osu')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows_out = []
    skipped = 0

    for i, f in enumerate(files):
        try:
            keys, rows, holds = parse_osu(f)
        except Exception:
            skipped += 1
            continue
        # MinaCalc 是为 4K 设计的；其它键数算出来的值没有可比性，直接跳过
        if keys != 4 or len(rows) < 20:
            skipped += 1
            continue
        msd = calc(lib, rows, rate=args.rate)
        if not msd or msd.get('overall', 0) <= 0:
            skipped += 1
            continue
        dur = (rows[-1][0] - rows[0][0]) / 1000.0
        rows_out.append({
            'id': os.path.splitext(os.path.basename(f))[0],
            'rows': len(rows),
            'notes': sum(bin(m).count('1') for _, m in rows),
            'holds': holds,
            'durSec': round(dur, 1),
            'nps': round(sum(bin(m).count('1') for _, m in rows) / dur, 2) if dur > 0 else 0,
            **msd,
        })
        if (i + 1) % 50 == 0:
            print(f'\r{i + 1}/{len(files)}  有效 {len(rows_out)}  跳过 {skipped}', end='')

    print(f'\r完成：{len(rows_out)} 张有效 / {skipped} 跳过（非 4K 或过短）')

    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'写出 {args.out}')

    if not rows_out:
        return

    # 汇总
    print(f"\n{'技能':<12}{'均值':>7}{'中位':>7}{'最大':>7}")
    for k in SKILLS:
        vals = sorted(r[k] for r in rows_out if k in r)
        if not vals:
            continue
        print(f'{k:<12}{sum(vals)/len(vals):>7.2f}{vals[len(vals)//2]:>7.2f}{vals[-1]:>7.2f}')

    if args.plot:
        make_plot(rows_out, args.plot)


def make_plot(rows, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        print('没装 matplotlib，跳过绘图')
        return

    # 找一个能显示中文的字体，找不到就退回英文标签
    zh = None
    for name in ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            zh = name
            break
        except Exception:
            continue
    if zh:
        plt.rcParams['font.sans-serif'] = [zh]
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle('4K beatmap skill profile (Etterna MinaCalc)  n=%d' % len(rows),
                 fontsize=14, fontweight='bold')

    for ax, k in zip(axes.flat, SKILLS):
        vals = [r[k] for r in rows if k in r]
        ax.hist(vals, bins=28, color='#38e8ff', edgecolor='#0a1120', alpha=0.85)
        ax.set_title(k, fontsize=11, fontweight='bold')
        ax.axvline(sorted(vals)[len(vals) // 2], color='#ff4fd8', lw=1.6, ls='--')
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=8)

    # 最后一张换成技能雷达式的相关性：各技能 vs overall
    ax = axes.flat[-1]
    ax.clear()
    ov = [r['overall'] for r in rows]
    for k, c in zip(SKILLS[1:], ['#ff4fd8', '#ffc23d', '#4dffa6', '#9b8cff', '#ff7a5c', '#5ce1ff', '#d6ff3d']):
        ax.scatter(ov, [r[k] for r in rows], s=5, alpha=0.45, color=c, label=k)
    ax.plot([0, max(ov)], [0, max(ov)], color='#888', lw=1, ls=':')
    ax.set_xlabel('overall', fontsize=9)
    ax.legend(fontsize=6, loc='upper left', ncol=2, framealpha=0.3)
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=130, facecolor='#0d1420')
    print(f'画像 {path}')


if __name__ == '__main__':
    main()
