#!/usr/bin/env python3
"""音频 → 4K 谱面，一条命令。

    python3 tools/chartgen.py song.ogg --target 21.5 --title "INTERNET YAMERO" \\
        --artist KOTOKO --out out/yamero

流程（每步都能单独跑，这里串起来）：

    1. analyze-audio.mjs   解码 → BPM / offset / onset 峰 / 11.6ms 细包络
    2. gen-notes.mjs       onset 定时刻 → 段落能量配额 → 键型步进器定列
    3. chartgen.py         clean_jacks（同列 <100ms 换列/删）→ add_holds → 写 .osu

难度不靠「几星」猜：MinaCalc 算 MSD，按 MSD = 2.02·nps^0.88 反解初值再迭代。
这条曲线不是拍脑袋的——用 1990 张真实 4K ranked 谱回归出来的（R²=0.956），
见 out/nps-msd.json 与 README「密度-难度曲线」一节。

    --target 21.5    给难度（Etterna Dan 段位：Delta ≈ 21–22）
    --nps 11.4       或者直接给密度，跳过拟合
"""
import argparse
import ctypes
import json
import math
import os
import shutil
import subprocess
import sys
import re
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKILLS = ['overall', 'stream', 'jumpstream', 'handstream',
          'stamina', 'jackspeed', 'chordjack', 'technical']

# 5463 个 4K ranked 难度（2026-10 全量抓取），MinaCalc 算出 5254 个 MSD 后
# 做幂律回归：
#   MSD = 1.9661 * nps^0.8921      R² = 0.9074
# 离群的基本都是新上传的异常谱（2.96 nps 却算出 31.3 MSD，×5.7 倍），
# 大概率是整活/转谱/时间轴错乱，不该让它们带偏「密度→难度」这条线。
# 注：语料从 1978 扩到 5254 后 R² 从 0.9625 降到 0.9074——小样本时
# 同质化程度高（都是热门 ranked），全量后尾部分散度真实显现，属于预期。
FIT_A, FIT_B, FIT_R2 = 1.9661, 0.8921, 0.9074


# --------------------------------------------------------------------------
# MinaCalc
# --------------------------------------------------------------------------
def load_minacalc(path):
    lib = ctypes.CDLL(path)
    lib.mc_calc.restype = ctypes.c_int
    lib.mc_calc.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint),
                            ctypes.c_int, ctypes.c_double, ctypes.c_double,
                            ctypes.POINTER(ctypes.c_float)]
    return lib


def msd_of(lib, notes, rate=1.0, goal=0.93):
    """notes: [{t, col}] → 7 维技能值。同一毫秒的音符合并成一行（和弦）。"""
    if not notes:
        return None
    rows = {}
    for n in notes:
        t = int(n['t'])
        rows[t] = rows.get(t, 0) | (1 << int(n['col']))
    keys = sorted(rows.items())
    n = len(keys)
    times = (ctypes.c_double * n)(*[t / 1000.0 for t, _ in keys])
    masks = (ctypes.c_uint * n)(*[m for _, m in keys])
    out = (ctypes.c_float * 8)()
    got = lib.mc_calc(times, masks, n, rate, goal, out)
    if got <= 0:
        return None
    return {SKILLS[i]: round(float(out[i]), 2) for i in range(min(8, got))}


# --------------------------------------------------------------------------
# 后处理
# --------------------------------------------------------------------------
def clean_jacks(notes, keys=4, min_gap_ms=76):
    """同列间隔 <100ms 的音符换列，四列都挤就删掉。

    这套生成器先定时刻再定列，步进器不检查「这列上一步刚用过」。
    结果 stream/jumpstream 相邻两拍会撞同列，隔 40–80ms —— 物理上按不出来，
    MinaCalc 会判成极速纵连，jackspeed 直接 20+，整张谱的难度画像失真。
    真实谱的叠键是谱面按拍子精确排的，不会是 50ms 的假纵连。
    """
    notes = sorted(notes, key=lambda z: (z['t'], z['col']))
    last = {}
    fixed = dropped = 0
    out = []
    for n in notes:
        col = int(n['col'])
        t = int(n['t'])
        prev = last.get(col)
        if prev is not None and t - prev < min_gap_ms:
            cands = [c for c in range(keys)
                     if last.get(c) is None or t - last[c] >= min_gap_ms]
            if cands:
                cands.sort(key=lambda c: (abs(c - col), c))
                n['col'] = cands[0]
                fixed += 1
                col = cands[0]
            else:
                dropped += 1
                continue
        last[col] = t
        out.append(n)
    return out, fixed, dropped


def add_holds(notes, bpm, ratio=0.17, offset_ms=0.0, seed=0):
    """LN 层生成器 v2：按人类谱挖掘规律放置（语料 5463 张 4K ranked，D13-16 档）。

    旧版（v13 之前）的病根——用户实测「长条位置和数量不符合人类」：
      1. 长度 = 同列间隔×0.62：尾永远提前 38% 松手，全长出 0.6 拍上下的
         中长条（人类长度 p50 只有 0.42 拍，61% 是 <0.5 拍的短条）
      2. 按时间均匀分桶：8s 窗 LN 数 cv 只有 0.24——人类 0.93，LN 是
         成段的（LN sections），不是全曲均匀撒
      3. LN 头同刻和弦率 0.32：人类 0.53，一半的 LN 头踩在和弦行里
         （LN+rice 层：一根手指按住，其余手指还在打）

    v2 规则（全部来自挖掘中位数）：
      - 占比目标 ratio（站点 Koishi 硬上限 18%，默认 0.17 贴着上限）
      - 长度按格量化：{1,2,3,4,6,8} 格 = {0.25,0.5,0.75,1,1.5,2} 拍，
        短条为主（p50≈2 格=0.5 拍）
      - 释放后同列留 ≥1 格再起压（人类 p10 0.249 拍 ≈ 1 格）
      - 8s 窗激活制：约 38% 的窗激活并可续一段（cv≈1.0），激活窗内
        和弦行优先转 LN
      - 爆发段照放（人类 LN 处局部密度 = 全曲均值 1.01，不挑疏处），
        但行速 >14 行/s 时长度压到 ≤2 格——1/32 密流里锁一根长手指
        是另一回事
    """
    if not notes:
        return notes, 0
    import random
    from bisect import bisect_left, bisect_right
    from collections import Counter
    beat = 60000.0 / (bpm or 120)
    cell = beat / 4.0
    rng = random.Random((seed or 0) * 2654435761 + 99991)

    last_t = max(n['t'] for n in notes)
    row_ts = sorted(set(n['t'] for n in notes))

    def local_rowrate(t):
        return (bisect_right(row_ts, t + 500) - bisect_left(row_ts, t - 500)) / 1.0

    by_col = {}
    for i, n in enumerate(notes):
        by_col.setdefault(int(n['col']), []).append(i)
    nxt_same = {}
    for c, idxs in by_col.items():
        idxs.sort(key=lambda i: notes[i]['t'])
        for j, i in enumerate(idxs):
            nxt_same[i] = notes[idxs[j + 1]]['t'] if j + 1 < len(idxs) else last_t

    # 候选：网格对齐的 tap（1/24、1/32 细分位不做 LN 头，保证尾也在格上）
    cands = []
    for i, n in enumerate(notes):
        if n.get('end'):
            continue
        ph = (n['t'] - offset_ms) % cell
        ph = min(ph, cell - ph)
        if ph > 2.0:
            continue
        gap = nxt_same[i] - n['t']
        len_max = int((gap - 0.9 * cell) // cell)          # 释放后留 ~1 格
        rate = local_rowrate(n['t'])
        cap = 2 if rate > 14 else 4 if rate > 10 else 8     # 密流锁短、疏处放长
        len_max = min(len_max, cap, int((last_t + cell - n['t']) // cell))
        if len_max >= 1:
            cands.append((i, len_max))

    target = int(len(notes) * ratio)

    # LN 成段：8s 窗激活（人类 8s 窗 LN 数 cv≈0.93），激活可续窗
    win = 8000.0
    n_win = int(last_t / win) + 2
    active = [False] * n_win
    w = 0
    while w < n_win:
        if rng.random() < 0.38:
            active[w] = True
            if w + 1 < n_win and rng.random() < 0.55:
                active[w + 1] = True
                w += 1
        w += 1

    def win_of(t):
        return min(n_win - 1, int(t / win))

    row_size = Counter(n['t'] for n in notes)
    def is_chord(c):
        return row_size[notes[c[0]]['t']] > 1

    act = [c for c in cands if active[win_of(notes[c[0]]['t'])]]
    inact = [c for c in cands if not active[win_of(notes[c[0]]['t'])]]
    rng.shuffle(act)
    rng.shuffle(inact)
    act.sort(key=lambda c: 0 if is_chord(c) else 1)    # 稳定排序：和弦行优先
    inact.sort(key=lambda c: 0 if is_chord(c) else 1)

    # 两段式选位（v2.2）：本图同列间隔里 2 格占大头（轮转键型的必然），
    # 均匀抽取会让一半 LN 只有 1 格（p50 0.25 拍，人类 0.42）。
    # 人类长度分布是 p10=1 格 / p50≈1.7 格 / p90≈3.3 格——所以先从
    # 空间 ≥2 格的候选里抽 2-6 格条（和弦行优先），不够再补 1 格短条，
    # 正好复现人类的 p10 尾巴。
    long_pool = [c for c in (act + inact) if c[1] >= 2]
    short_pool = [c for c in (act + inact) if c[1] == 1]

    made = 0
    used_rows = set()
    # 长条抽样：2 格为主（人类 p50 1.7 格），3/4 格次之
    for i, len_max in long_pool:
        if made >= target:
            break
        t = notes[i]['t']
        if t in used_rows:
            continue
        r = rng.random()
        L = 2 if r < 0.42 else 3 if r < 0.70 else 4 if r < 0.94 else 6
        notes[i]['end'] = int(round(t + min(L, len_max) * cell))
        used_rows.add(t)
        made += 1
    # 补 1 格短条（人类 p10 的尾巴）
    for i, len_max in short_pool:
        if made >= target:
            break
        t = notes[i]['t']
        if t in used_rows:
            continue
        notes[i]['end'] = int(round(t + cell))
        used_rows.add(t)
        made += 1
    return notes, made


# --------------------------------------------------------------------------
# 写 .osu
# --------------------------------------------------------------------------
def build_osu(notes, meta, audio_name, title, artist, creator, version,
              keys=4, od=8.0, source='', tags='ai generated scene3d'):
    bpm = meta.get('bpm') or 120
    beat = 60000.0 / bpm
    offset = int(meta.get('offsetMs') or 0)
    L = [
        'osu file format v14', '',
        '[General]',
        f'AudioFilename: {audio_name}',
        'AudioLeadIn: 0',
        f'PreviewTime: {offset + 60000}',
        'Countdown: 0',
        'SampleSet: Soft',
        'StackLeniency: 0.7',
        'Mode: 3',
        'LetterboxInBreaks: 0',
        'SpecialStyle: 0',
        'WidescreenStoryboard: 0', '',
        '[Editor]',
        'DistanceSpacing: 1',
        'BeatDivisor: 4',
        'GridSize: 4',
        'TimelineZoom: 1', '',
        '[Metadata]',
        f'Title:{title}',
        f'TitleUnicode:{title}',
        f'Artist:{artist}',
        f'ArtistUnicode:{artist}',
        f'Creator:{creator}',
        f'Version:{version}',
        f'Source:{source}',
        f'Tags:{tags}',
        'BeatmapID:0',
        'BeatmapSetID:-1', '',
        '[Difficulty]',
        'HPDrainRate:8',
        f'CircleSize:{keys}',
        f'OverallDifficulty:{od}',
        'ApproachRate:5',
        'SliderMultiplier:1.4',
        'SliderTickRate:1', '',
        '[Events]',
        '//Background and Video events',
        '//Break Periods',
        '//Storyboard Layer 0 (Background)',
        '//Storyboard Layer 1 (Fail)',
        '//Storyboard Layer 2 (Pass)',
        '//Storyboard Layer 3 (Foreground)',
        '//Storyboard Sound Samples', '',
        '[TimingPoints]',
        f'{offset},{beat:.4f},4,2,1,70,1,0', '',
        '[HitObjects]',
    ]
    for n in sorted(notes, key=lambda z: (z['t'], z['col'])):
        # mania 的 x 就是列位置；按 keys 均分 512 宽
        x = int(512 * (int(n['col']) + 0.5) / keys)
        t = int(n['t'])
        if n.get('end') and n['end'] > t:
            L.append(f'{x},192,{t},128,0,{int(n["end"])}:0:0:0:0:')
        else:
            L.append(f'{x},192,{t},1,0,0:0:0:0:')
    L.append(f'// AIMode: scene3d chartgen (onset-driven, MSD-fitted)')
    return '\n'.join(L)


# --------------------------------------------------------------------------
# 子进程
# --------------------------------------------------------------------------
def run_node(script, *a):
    r = subprocess.run(['node', os.path.join(HERE, script), *map(str, a)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f'{script} 失败：\n{r.stderr[-1500:]}')
    return r.stdout


def analyze(audio, analysis_path, sr=22050, delta=0.6, bpm=None, offset=None):
    a = [audio, '-o', analysis_path, '--sr', sr, '--delta', delta]
    if bpm:
        a += ['--bpm', bpm]
    if offset is not None:
        a += ['--offset', offset]
    out = run_node('analyze-audio.mjs', *a)
    return json.loads(out.strip().splitlines()[-1])


def calibrate(analysis_path, bpm, sub=16):
    """给定 BPM，反解第一拍相位（见 tools/calibrate-grid.mjs）。

    自动测 BPM 在本曲上会把 185 判成 93.8（倍频错），网格一错小节线全歪。
    所以 BPM 由人来给，offset 让机器解：在 [0, beat) 里搜一个相位，
    让最多的强 onset 落在 1/4 拍上；再在 4 个相位里挑重拍当小节起点。
    """
    out = run_node('calibrate-grid.mjs', analysis_path, '--bpm', bpm, '--sub', sub)
    return json.loads(out.strip().splitlines()[-1])


def gen_notes(analysis_path, nps, seed, keys, notes_path, section=8, chord=None,
              weights=None, onsets=False, dyn=None, grid_min=None, seg_mul_cap=None):
    a = [analysis_path, '--nps', f'{nps:.3f}', '--seed', seed, '--keys', keys,
         '--section', section, '-o', notes_path]
    if onsets:
        a.append('--onsets')
    if dyn is not None:
        a += ['--dyn', dyn]
    if grid_min is not None:
        a += ['--grid-min', grid_min]
    if seg_mul_cap is not None:
        a += ['--seg-mul-cap', seg_mul_cap]
    if chord is not None:
        a += ['--chord', chord]
    if weights:
        a += ['--weights', weights]
    run_node('gen-notes.mjs', *a)
    return json.load(open(notes_path, encoding='utf-8'))


def density_check(nps, msd):
    """拿 5254 个真实 4K 难度的分桶做体检：同密度下这张谱算难还是算简单。"""
    p = os.path.join(ROOT, 'out', 'nps-msd.json')
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    # 桶宽 2 nps，对齐到偶数边界
    lo = math.floor(nps / 2) * 2
    g = sorted(r['overall'] for r in rows if lo <= r['nps'] < lo + 2)
    if len(g) < 10:
        return None
    pct = sum(1 for x in g if x < msd) / len(g) * 100
    q = lambda f: g[min(len(g) - 1, int(len(g) * f))]
    return {
        'bucket': f'{lo}-{lo + 2} nps',
        'n': len(g),
        'p25': round(q(0.25), 2), 'median': round(q(0.5), 2), 'p75': round(q(0.75), 2),
        'percentile': round(pct, 1),
        'verdict': ('偏硬' if pct > 90 else '偏软' if pct < 10 else '正常'),
    }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='音频 → 4K 谱面（MinaCalc 定难度）')
    ap.add_argument('audio', help='音频文件（ffmpeg 支持的任意格式）')
    ap.add_argument('--out', default='out/chart', help='输出目录')
    ap.add_argument('--target', type=float, default=None,
                    help='目标 Etterna MSD（Delta≈21.5）；与 --nps 二选一')
    ap.add_argument('--nps', type=float, default=None, help='直接指定密度，跳过拟合')
    ap.add_argument('--keys', type=int, default=4)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--title', default=None)
    ap.add_argument('--artist', default=None)
    ap.add_argument('--version', default=None, help='难度名，默认按 MSD 取 Dan 名')
    ap.add_argument('--creator', default='scene3d chartgen')
    ap.add_argument('--source', default='')
    ap.add_argument('--od', type=float, default=8.0)
    ap.add_argument('--ln', type=float, default=0.17,
                    help='长条占比目标（人类 13-15 档 27-31%；站点 Koishi 硬上限 0.18，'
                         '超过会 UnsupportedLN，默认 0.17 贴上限）')
    ap.add_argument('--section', type=float, default=8.0, help='段落长度（秒）')
    ap.add_argument('--bpm', type=float, default=None,
                    help='强制 BPM（自相关会把 185 测成 93.8，建议手动给）')
    ap.add_argument('--offset', type=float, default=None,
                    help='强制 offset(ms)；不给则按 BPM 自动校准相位')
    ap.add_argument('--dyn', type=float, default=1.6,
                    help='段落密度指数：1=线性(平)，1.6≈人类起伏，2.2=强对比')
    ap.add_argument('--onsets', action='store_true',
                    help='退回「贴 onset 峰」模式（默认走 1/16 网格）')
    ap.add_argument('--weights', default=None,
                    help='键型权重 stream=6.5,jumpstream=2.6,chordjack=3.0（其余用默认）')
    ap.add_argument('--chord', type=float, default=None,
                    help='手动指定和弦率 notes/row（默认按密度查人类分档表）')
    ap.add_argument('--grid-min', type=float, default=None,
                    help='网格事件的能量门槛（默认 0.04，调低=供给更多行）')
    ap.add_argument('--seg-mul-cap', type=float, default=None,
                    help='段落能量乘数上限（默认 2.8，调高=峰值段更密）')
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--lib', default=os.path.join(ROOT, 'libminacalc.so'))
    ap.add_argument('--reanalyze', action='store_true', help='强制重跑音频分析')
    ap.add_argument('--osz', action='store_true', help='顺手打包 .osz')
    ap.add_argument('--keep', action='store_true', help='保留中间文件')
    args = ap.parse_args()

    if not os.path.exists(args.lib):
        sys.exit(f'找不到 MinaCalc 库：{args.lib}（编译方法见 README）')
    if args.target is None and args.nps is None:
        args.target = 21.5
    os.makedirs(args.out, exist_ok=True)

    # 上传文件常带 "1792920093161178607-" 这种数字前缀，osu 里写着难看
    raw = os.path.basename(args.audio)
    audio_name = re.sub(r'^\d{6,}-', '', raw)
    title = args.title or os.path.splitext(audio_name)[0]
    artist = args.artist or 'Unknown'

    # ---- 1. 分析 ----
    analysis_path = os.path.join(args.out, 'analysis.json')
    if args.reanalyze or not os.path.exists(analysis_path):
        # 先跑一遍（BPM/offset 用指定的），再校准相位，最后带 offset 重跑。
        # 注意：第一遍也必须把 --offset 传进去——旧代码只在自动校准分支里
        # 传 offset，显式给定 offset 时被静默忽略，网格整体偏 51ms
        # （1/16 格 81ms 的 0.63），音符全部落在分音之间的"影子格"上。
        info = analyze(args.audio, analysis_path, bpm=args.bpm, offset=args.offset)
        if args.offset is None and args.bpm:
            cal = calibrate(analysis_path, args.bpm)
            args.offset = cal['offsetMs']
            info = analyze(args.audio, analysis_path, bpm=args.bpm, offset=args.offset)
            print(f"[1/3] 分析 {info['durationSec']}s  BPM {args.bpm}（手动）  "
                  f"offset {cal['offsetMs']:.0f}ms（校准）")
            print(f"      落网率：1/4 {cal['snap']['quarter']}%  "
                  f"1/8 {cal['snap']['eighth']}%  1/16 {cal['snap']['sixteenth']}%  "
                  f"（共 {cal['bars']} 小节）")
        else:
            print(f"[1/3] 分析 {info['durationSec']}s  BPM {info['bpm']} "
                  f"(conf {info['confidence']})  offset {info['offsetMs']}ms  "
                  f"峰 {info['peaks']} 个 ({info['peakPerSec']}/s)")
            if info['confidence'] < 0.3:
                print(f"      注意：BPM 置信度只有 {info['confidence']:.0%}，"
                      f"建议用 --bpm 手动指定（自相关会把 185 测成 93.8 这种倍频错）")
    else:
        d = json.load(open(analysis_path))
        info = {'durationSec': d['durationSec'], 'bpm': d['bpm'],
                'confidence': d.get('bpmConfidence', 0), 'offsetMs': d['offsetMs'],
                'peaks': len(d['peaks']), 'peakPerSec': 0}
        print(f"[1/3] 复用分析 {info['durationSec']}s  BPM {info['bpm']}  offset {info['offsetMs']}ms")

    lib = load_minacalc(args.lib)
    notes_path = os.path.join(args.out, 'notes.json')

    def build(nps):
        data = gen_notes(analysis_path, nps, args.seed, args.keys, notes_path,
                         section=args.section, onsets=args.onsets, dyn=args.dyn,
                         weights=args.weights, chord=args.chord,
                         grid_min=args.grid_min, seg_mul_cap=args.seg_mul_cap)
        data['notes'], fix, drop = clean_jacks(data['notes'], keys=args.keys)
        data['notes'], ln = add_holds(data['notes'], data.get('bpm') or 120,
                                      ratio=args.ln,
                                      offset_ms=(args.offset if args.offset is not None
                                                 else info['offsetMs']),
                                      seed=args.seed)
        m = msd_of(lib, data['notes'])
        return data, m, fix, drop, ln

    # ---- 2. 拟合 ----
    if args.nps is not None:
        nps = args.nps
        data, m, fix, drop, ln = build(nps)
        history = [(round(nps, 2), m['overall'], len(data['notes']))]
        print(f"[2/3] 指定 {nps} nps → MSD {m['overall']}")
    else:
        # 用回归曲线给初值，比从 12 nps 猜快一倍
        nps = (args.target / FIT_A) ** (1 / FIT_B)
        print(f"[2/3] 目标 MSD {args.target}，回归初值 {nps:.2f} nps")
        history = []
        best = None
        for i in range(args.iters):
            data, m, fix, drop, ln = build(round(nps, 2))
            over = m['overall']
            history.append((round(nps, 2), over, len(data['notes'])))
            print(f"      第 {i + 1} 轮  nps={nps:5.2f}  notes={len(data['notes']):>5}  "
                  f"MSD={over:5.2f}  换列={fix:<4} 删={drop:<3} LN={ln:<4}"
                  f"(stream {m['stream']}, jack {m['jackspeed']})")
            # MSD 对 nps 不是严格单调的（clean_jacks 换列会改变结构），
            # 最后落地的那轮未必最接近目标。记下所有轮次，挑最准的那张。
            if best is None or abs(over - args.target) < abs(best[1]['overall'] - args.target):
                best = (data, m, fix, drop, ln)
            if abs(args.target - over) <= 0.3:
                print(f"      收敛：MSD {over} vs 目标 {args.target}")
                break
            ratio = (args.target / over) ** (1 / FIT_B) if over > 0 else 1.15
            ratio = max(0.65, min(1.35, ratio))
            nps *= ratio
        data, m, fix, drop, ln = best
        print(f"      采用：MSD {m['overall']}（误差 {abs(m['overall'] - args.target):.2f}）")

    notes = data['notes']
    duration = data['meta']['durationSec']
    final_nps = len(notes) / duration
    holds = sum(1 for n in notes if n.get('end'))

    # ---- 3. 写盘 ----
    version = args.version or dan_name(m['overall'])
    osu_text = build_osu(notes, data, audio_name, title, artist, args.creator,
                         version, keys=args.keys, od=args.od, source=args.source)
    osu_name = f'{artist} - {title} ({args.creator}) [{version}].osu'
    osu_path = os.path.join(args.out, osu_name)
    open(osu_path, 'w', encoding='utf-8').write(osu_text)

    audio_dst = os.path.join(args.out, audio_name)
    if os.path.abspath(args.audio) != os.path.abspath(audio_dst):
        shutil.copy2(args.audio, audio_dst)

    check = density_check(final_nps, m['overall'])

    # 技能向量定向建议：MinaCalc 的技能占比就是"键型配方"的成绩单。
    # 人类 14–16 nps 带的中位占比（out/msd-full.jsonl, n=487）：
    #   stream .891 / js .947 / hs .815 / stamina .949 /
    #   jackspeed .585 / chordjack .734 / technical .927
    HUMAN_RATIO = {'stream': 0.891, 'jumpstream': 0.947, 'handstream': 0.815,
                   'stamina': 0.949, 'jackspeed': 0.585, 'chordjack': 0.734,
                   'technical': 0.927}
    # 默认权重（与 gen-notes.mjs 的 DEFAULT_W 保持一致），建议值按比例放大
    WEIGHT_SUGGEST = {'stream': 5.5, 'jumpstream': 3.6, 'handstream': 2.4,
                      'chordjack': 5.0, 'trill': 0.8, 'roll': 1.5,
                      'ladder': 1.5, 'staircase': 1.5}
    hints = []
    for sk, hr in HUMAN_RATIO.items():
        if m.get(sk, 0) <= 0:
            continue
        ratio = m[sk] / m['overall']
        gap = hr - ratio
        if gap > 0.08:
            hints.append({'skill': sk, 'ratio': round(ratio, 3), 'human': hr,
                          'hint': f'低于人类带 {gap:.2f}，试 --weights {sk}='
                                  f'{round(WEIGHT_SUGGEST.get(sk, 1) * (1 + gap), 1)}'})
    if hints:
        print("      键型建议：")
        for h in hints:
            print(f"        {h['skill']:12s} {h['ratio']:.3f} vs 人类 {h['human']:.3f}  →  {h['hint']}")

    report = {
        'title': title, 'artist': artist, 'version': version,
        'audio': audio_name, 'osu': osu_name,
        'keys': args.keys, 'notes': len(notes), 'holds': holds,
        'holdRatio': round(holds / max(1, len(notes)) * 100, 1),
        'durationSec': round(duration, 1), 'nps': round(final_nps, 2),
        'bpm': data.get('bpm'), 'offsetMs': data.get('offsetMs'),
        'alignRatio': data['meta'].get('alignRatio'),
        'sections': data['sections'], 'msd': m,
        'history': history, 'densityCheck': check,
        'hand': data['meta'].get('hand'),
        'skillHints': hints,
        'fit': {'a': FIT_A, 'b': FIT_B, 'r2': FIT_R2},
    }
    json.dump(report, open(os.path.join(args.out, 'report.json'), 'w'),
              ensure_ascii=False, indent=1)

    print(f"\n[3/3] {osu_name}")
    print(f"      {len(notes)} notes  长条 {holds} ({report['holdRatio']}%)  "
          f"{duration:.0f}s  {final_nps:.2f} nps  BPM {data.get('bpm')}")
    print(f"      MSD {m['overall']}  stream {m['stream']} js {m['jumpstream']} "
          f"hs {m['handstream']} stamina {m['stamina']}")
    print(f"              jack {m['jackspeed']} cj {m['chordjack']} tech {m['technical']}")
    if check:
        print(f"      密度体检：{check['bucket']} 桶内真实谱中位 {check['median']} "
              f"(P25 {check['p25']} / P75 {check['p75']})，"
              f"这张在第 {check['percentile']} 百分位 → {check['verdict']}")
    print(f"      写出 {osu_path}")

    if args.osz:
        osz_path = os.path.join(args.out, f'{title} [{version}].osz')
        with zipfile.ZipFile(osz_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.write(osu_path, osu_name)
            z.write(audio_dst, audio_name)
        print(f"      打包 {osz_path} ({os.path.getsize(osz_path) / 1048576:.1f} MB)")

    if not args.keep:
        for f in ['notes.json']:
            p = os.path.join(args.out, f)
            if os.path.exists(p):
                os.remove(p)


def dan_name(msd):
    """默认难度名直接写 MSD。

    不去猜「这个 MSD 该叫哪一档 Dan」——Etterna 各段位的 MSD 区间随版本变，
    与其编个看着专业其实会错的名字，不如把 MinaCalc 的原值摆出来。
    要自定义就传 --version。
    """
    return f'MSD {msd:.1f}'


if __name__ == '__main__':
    main()
