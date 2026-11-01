#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval-generated: 生成谱面的客观质量评估（扩散 vs 规则生成器 vs 人类基线）。

为什么要这个脚本
----------------
训练损失（MSE/CE）低 ≠ 谱面好。损失只保证「模型复现了训练集的钢琴卷分布」，
不保证谱面踩在音上、有段落结构、键型像某个流派、能弹。这个脚本用**可测量**的
指标把「好不好」拆成四组：

  1. 踩音对齐  —— 音符/行到音频起音的距离（需要谱面对应 out/multisong 里的 6 首歌）
  2. 结构指标  —— 音符率、npr、jack 率、1/32 占比、手位移、行重复率、占位率
  3. 键型相似度—— 与 tech/jack/speed/stamina 四个包质心的余弦相似度
  4. 难度 + 可弹性 —— 观景台 Koishi diff、同列 <55ms 冲突、单侧 3 音/55ms 冲突

用法
----
  # 评估扩散采样谱
  python3 tools/eval-generated.py out/diffusion/test-*.osu --label diffusion \
      --json out/diffusion/eval-diffusion.json

  # 规则生成器对照组
  python3 tools/eval-generated.py out/all/Heart_Fraction--*-D22.osu --label rule-gen

  # 人类基线：每个包随机 20 张（--seed 固定，可复现）
  python3 tools/eval-generated.py --pack-sample 20 --seed 0 --label human \
      --json out/diffusion/eval-human.json

  # 一次评估多组（组名=文件前缀映射），输出对比表
  python3 tools/eval-generated.py --groups groups.json --json out/diffusion/eval-all.json

依赖（只读，不改）
------------------
  - tools/audit-chart.py   parse_osu() / rows_of()：全仓库统一解析器（importlib 加载）
  - tools/gj/batch-koishi.mjs  难度 + 5 维键型概率（node 子进程，一次最多 20 张）
  - /tmp/d_{tech,jack,speed,stamina}.json  人类包质心（batch-koishi 输出数组，取 probs 平均；
                                           文件缺失时自动用包里 20 张重算）
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics as st
import subprocess
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKS = ['tech', 'jack', 'speed', 'stamina']
PACK_DIR = os.path.join(ROOT, 'out/packs')
MULTISONG_DIR = os.path.join(ROOT, 'out/multisong')
KOISHI = os.path.join(ROOT, 'tools/gj/batch-koishi.mjs')
KOISHI_CACHE = '/tmp/eval-koishi-cache.json'
BATCH_MAX = 20           # 硬性要求：一次最多 20 张（内存峰值）
KOISHI_MODE = 'batch'    # batch=整批（快）；per-file=逐张（稳，可捕捉病态谱）
KOISHI_TIMEOUT = 120     # 逐张模式的单张墙钟上限（秒）
DEFAULT_BPM = 136.3
STEP_MS = 60000.0 / DEFAULT_BPM / 4.0     # 1/16 格 = 110.05ms


# ---------------------------------------------------------------------------
# 统一解析器（只 import，不改 audit-chart.py）
# ---------------------------------------------------------------------------
def _load_audit():
    spec = importlib.util.spec_from_file_location('ac_eval', os.path.join(ROOT, 'tools/audit-chart.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ac = _load_audit()


# ---------------------------------------------------------------------------
# 人类包质心
# ---------------------------------------------------------------------------
def _probs_mean(rows):
    keys = PACKS + ['ln'] if any('ln' in (r.get('probs') or {}) for r in rows) else None
    if keys is None:
        keys = sorted({k for r in rows for k in (r.get('probs') or {})})
    n = 0
    acc = {k: 0.0 for k in keys}
    for r in rows:
        p = r.get('probs') or {}
        if not p:
            continue
        n += 1
        for k in keys:
            acc[k] += float(p.get(k, 0.0))
    if not n:
        raise SystemExit('质心计算失败：没有有效的 probs')
    return {k: v / n for k, v in acc.items()}, n


def load_centroids(rebuild=False):
    """返回 {pack: {probs dict, n, source}}。"""
    cents, missing = {}, []
    for p in PACKS:
        f = f'/tmp/d_{p}.json'
        if os.path.exists(f) and not rebuild:
            try:
                rows = json.load(open(f))
                vec, n = _probs_mean(rows)
                cents[p] = {'probs': vec, 'n': n, 'source': f}
                continue
            except Exception:
                pass
        missing.append(p)
    if missing:
        print(f'[centroid] 质心缓存缺失 {missing}，用每包 20 张重算…', file=sys.stderr)
        for p in missing:
            files = sorted(glob.glob(os.path.join(PACK_DIR, p, '*.osu')))
            rng = random.Random(0)
            pick = rng.sample(files, min(20, len(files)))
            res = run_koishi(pick, label=f'centroid-{p}')
            vec, n = _probs_mean(res)
            cents[p] = {'probs': vec, 'n': n, 'source': 'recomputed'}
            json.dump({**vec}, open(f'/tmp/d_{p}.json', 'w'), indent=1)
    return cents


def cosine(a, b, keys):
    num = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    da = math.sqrt(sum(a.get(k, 0.0) ** 2 for k in keys))
    db = math.sqrt(sum(b.get(k, 0.0) ** 2 for k in keys))
    return num / (da * db) if da * db > 1e-12 else float('nan')


# ---------------------------------------------------------------------------
# Koishi（node 子进程，批量 ≤20，带磁盘缓存）
# ---------------------------------------------------------------------------
def _cache_key(path):
    stt = os.stat(path)
    h = hashlib.md5(f'{os.path.abspath(path)}|{stt.st_mtime_ns}|{stt.st_size}'.encode()).hexdigest()
    return h


def _load_cache():
    if os.path.exists(KOISHI_CACHE):
        try:
            return json.load(open(KOISHI_CACHE))
        except Exception:
            return {}
    return {}


def _save_cache(c):
    tmp = KOISHI_CACHE + '.tmp'
    json.dump(c, open(tmp, 'w'))
    os.replace(tmp, KOISHI_CACHE)


def _run_node(files, timeout):
    """跑一次 batch-koishi，返回 JSON 列表；超时/解析失败抛异常。"""
    proc = subprocess.run(['node', KOISHI] + files, cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-800:])
        raise RuntimeError(f'batch-koishi 退出码 {proc.returncode}')
    return json.loads(proc.stdout)


def _one_fallback(p, timeout=None):
    """逐张跑：超时/失败写进 error 字段，不让整批陪葬。"""
    timeout = timeout or KOISHI_TIMEOUT
    base = os.path.basename(p)
    t0 = time.time()
    try:
        r = _run_node([p], timeout)[0]
        print(f'[koishi]   {base[:34]:34s} {time.time() - t0:5.1f}s diff={r.get("diff")}',
              file=sys.stderr)
        return r
    except subprocess.TimeoutExpired:
        print(f'[koishi]   {base[:34]:34s} 超时(>{timeout:.0f}s) → error=timeout',
              file=sys.stderr)
        return {'file': base, 'diff': None, 'tier': None, 'pattern': None,
                'probs': None, 'error': 'timeout'}
    except Exception as e:
        print(f'[koishi]   {base[:34]:34s} 失败 {e}', file=sys.stderr)
        return {'file': base, 'diff': None, 'tier': None, 'pattern': None,
                'probs': None, 'error': f'fail:{e}'}


def _set_koishi_mode(mode, timeout=None):
    global KOISHI_MODE, KOISHI_TIMEOUT
    KOISHI_MODE = mode
    if timeout:
        KOISHI_TIMEOUT = timeout


def run_koishi(paths, label=''):
    """跑 batch-koishi（一次 ≤20 张），返回与输入同序的结果。

    两个坑（实测踩过）：
      1. batch-koishi 内部对每张谱最多等 400s，而它的分析是「同步触发 + 异步完成」，
         被放弃的那张谱仍在后台烧 CPU —— 一张病态谱就能把整批拖到几十分钟；
      2. 整批被拖死时，前面已经算完的谱也一起丢掉。
    所以：batch 模式给整批设墙钟预算，超时就杀掉整批、逐张重跑；
    per-file 模式（--koishi-mode per-file）直接逐张跑，单张有独立上限，最稳。
    """
    cache = _load_cache()
    out = {}
    todo, keys = [], {}
    for p in paths:
        k = _cache_key(p)
        keys[p] = k
        if k in cache:
            out[p] = cache[k]
        else:
            todo.append(p)
    if todo:
        print(f'[koishi] {label}: 需跑 {len(todo)}/{len(paths)} 张（模式 {KOISHI_MODE}）',
              file=sys.stderr)
        for i in range(0, len(todo), BATCH_MAX):
            chunk = todo[i:i + BATCH_MAX]
            t0 = time.time()
            if KOISHI_MODE == 'per-file':
                res = [_one_fallback(p) for p in chunk]
                source = 'per-file'
            else:
                budget = 45 + 12 * len(chunk)
                try:
                    res = _run_node(chunk, budget)
                    source = 'batch'
                except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as e:
                    print(f'[koishi]   批 {i // BATCH_MAX + 1} 失败（{type(e).__name__}），'
                          f'改为逐张重跑 {len(chunk)} 张', file=sys.stderr)
                    res = [_one_fallback(p) for p in chunk]
                    source = 'per-file'
                    # 学到教训：这一批能把 batch 跑死，后面别再试整批了
                    _set_koishi_mode('per-file')
                    print('[koishi]   本轮剩余批次自动切到 per-file 模式', file=sys.stderr)
            base = {os.path.basename(p): p for p in chunk}
            for r in res:
                p = base.get(r['file'])
                if p:
                    out[p] = r
                    cache[keys[p]] = r
            _save_cache(cache)
            print(f'[koishi]   批 {i // BATCH_MAX + 1}: {len(chunk)} 张 '
                  f'{time.time() - t0:.1f}s ({source})', file=sys.stderr)
    return [out[p] for p in paths]


# ---------------------------------------------------------------------------
# 谱面元信息：bpm / offset / 对应歌曲
# ---------------------------------------------------------------------------
def chart_meta(notes, meta):
    """从 TimingPoints 取 bpm/offset；从 Title/AudioFilename 推断对应 multisong 歌。"""
    tps = meta.get('TimingPoints', [])
    bpm, offset = DEFAULT_BPM, None
    for t, b in tps:
        if b > 0:                      # 非继承点
            bpm = 60000.0 / b
            offset = t
            break
    if tps and offset is None:
        offset = tps[0][0]
    beat = 60000.0 / bpm

    song, song_src = None, 'no-audio-link'
    af = (meta.get('AudioFilename') or '').strip()
    stem = os.path.splitext(af)[0]
    cands = []
    if stem:
        cands.append(stem.replace(' ', '_'))
    for fx in glob.glob(os.path.join(MULTISONG_DIR, '*-fx.json')):
        s = os.path.basename(fx)[:-len('-fx.json')]
        for c in cands:
            if c and (c.lower() == s.lower() or c.lower().replace('-', '_') == s.lower()):
                song, song_src = s, af
    return {'bpm': round(bpm, 3), 'beat_ms': round(beat, 2),
            'offset_ms': (round(offset, 1) if offset is not None else None),
            'song': song, 'song_src': song_src}


# ---------------------------------------------------------------------------
# 结构指标
# ---------------------------------------------------------------------------
def structural(notes, rows, step):
    """音符率 / npr / jack 率 / 1/32 占比 / 手位移 / 行重复率 / 占位率。"""
    r = {}
    if not notes or not rows:
        return {k: None for k in ('duration_s', 'note_rate', 'row_rate', 'npr', 'chord_rate',
                                  'ln_ratio', 'jack_rate', 'p132', 'hand_shift', 'repeat8',
                                  'occupancy', 'gap_hist')}
    t0, t1 = notes[0][0], notes[-1][0]
    dur = max(1e-6, (t1 - t0) / 1000.0)
    r['duration_s'] = round(dur, 1)
    r['note_rate'] = round(len(notes) / dur, 2)          # 音符率 nps
    r['row_rate'] = round(len(rows) / dur, 2)            # 行率 rps
    r['npr'] = round(len(notes) / len(rows), 3)          # 每行音符数
    r['chord_rate'] = round(sum(1 for _, cs in rows if len(cs) > 1) / len(rows), 3)
    ln = sum(1 for _, _, e in notes if e is not None)
    r['ln_ratio'] = round(ln / len(notes), 3)

    # jack 率：相邻行间隔 ≤2 个 1/16 格 且 有共列
    gaps = [rows[i + 1][0] - rows[i][0] for i in range(len(rows) - 1)]
    jacks = 0
    for i in range(1, len(rows)):
        if gaps[i - 1] <= 2 * step * 1.05 and (set(rows[i - 1][1]) & set(rows[i][1])):
            jacks += 1
    r['jack_rate'] = round(jacks / max(1, len(gaps)), 3)

    # 1/32 占比：行间隔归一化成「1/16 格数」后 ≈0.5 的比例
    units = [g / step for g in gaps]
    n = max(1, len(units))
    r['p132'] = round(sum(1 for u in units if 0.35 <= u < 0.75) / n, 3)
    hist = Counter()
    for u in units:
        if u < 0.35:
            hist['<1/32'] += 1
        elif u < 0.75:
            hist['1/32'] += 1
        elif u < 1.4:
            hist['1/16'] += 1
        elif u < 2.4:
            hist['1/8'] += 1
        elif u < 3.6:
            hist['3/16'] += 1
        elif u < 4.6:
            hist['1/4'] += 1
        else:
            hist['>=1/2'] += 1
    r['gap_hist'] = {k: round(v / n, 3) for k, v in hist.items()}

    # 手位移均值：相邻行「重心列」的绝对差（单位=列宽）
    shifts = [abs(st.fmean(rows[i][1]) - st.fmean(rows[i - 1][1])) for i in range(1, len(rows))]
    r['hand_shift'] = round(st.fmean(shifts), 3) if shifts else 0.0

    # 行重复率：8 行滑窗签名重复比例
    sigs = [tuple(cs) for _, cs in rows]
    W = 8
    if len(sigs) >= W:
        wins = [tuple(sigs[i:i + W]) for i in range(len(sigs) - W + 1)]
        r['repeat8'] = round(1 - len(set(wins)) / len(wins), 3)
    else:
        r['repeat8'] = None

    # 占位率：占用格 / (格行数 × 4)
    ncell = int(round((t1 - t0) / step)) + 1
    r['occupancy'] = round(len(notes) / (ncell * 4), 4)
    return r


# ---------------------------------------------------------------------------
# 可弹性（参照 tools/playability-check.py 的口径，按任务要求收紧到 55ms）
# ---------------------------------------------------------------------------
def playability(notes, rows):
    """同列 <55ms 冲突 + 单侧 55ms 内 ≥3 音冲突。返回 (冲突数, 每分钟冲突数, 细节)。"""
    fast = []       # 同列极速连打
    side = []       # 单侧过载
    # 1) 同列 <55ms
    bycol = {}
    for t, c in [(t, c) for t, c, _ in notes]:
        bycol.setdefault(c, []).append(t)
    for c, ts in bycol.items():
        ts.sort()
        for i in range(1, len(ts)):
            if ts[i] - ts[i - 1] < 55:
                fast.append((ts[i], c, ts[i] - ts[i - 1]))
    # 2) 单侧 55ms 内 ≥3 音（左手 col0/1，右手 col2/3）
    for name, cols in (('L', (0, 1)), ('R', (2, 3))):
        ts = sorted(t for t, c, _ in notes if c in cols)
        i = 0
        while i < len(ts):
            j = i
            while j + 1 < len(ts) and ts[j + 1] - ts[i] < 55:
                j += 1
            if j - i + 1 >= 3:
                side.append((ts[i], name, j - i + 1))
                i = j + 1            # 跳过整段，避免同一段被重复计数
            else:
                i += 1
    dur = max(1e-6, (notes[-1][0] - notes[0][0]) / 1000.0)
    n = len(fast) + len(side)
    # 补充：单列最高连打频率（Hz）——1/16 网格下 55ms 冲突不可能触发，
    # 但「单指每秒要按几下」才是真正决定卡不卡手的量。
    max_hz = 0.0
    for c, ts in bycol.items():
        for i in range(1, len(ts)):
            g = ts[i] - ts[i - 1]
            if g > 0:
                max_hz = max(max_hz, 1000.0 / g)   # Hz = 1000ms / 间隔ms
    return {'fast_same_col': len(fast), 'side_overload': len(side),
            'conflicts': n, 'conflicts_per_min': round(n / dur * 60, 2),
            'max_col_hz': round(max_hz, 2)}


# ---------------------------------------------------------------------------
# 踩音对齐（用 out/multisong/*-fx.json 的起音）
# ---------------------------------------------------------------------------
_fx_cache = {}


def load_fx(song):
    if song not in _fx_cache:
        p = os.path.join(MULTISONG_DIR, f'{song}-fx.json')
        d = json.load(open(p))
        hop = d['fine']['hopMs']
        peaks_ms = sorted(int(pk) * hop for pk in d['peaks'])
        onset = d['onset']
        accent = d['accent']
        order = sorted(range(len(onset)), key=lambda i: onset[i])
        rank = [0.0] * len(onset)
        for r, i in enumerate(order):
            rank[i] = r / max(1, len(onset) - 1)
        order = sorted(range(len(accent)), key=lambda i: accent[i])
        rank_acc = [0.0] * len(accent)
        for r, i in enumerate(order):
            rank_acc[i] = r / max(1, len(accent) - 1)
        _fx_cache[song] = {'path': p, 'hop_ms': hop, 'peaks_ms': peaks_ms,
                           'offset': d['offsetMs'], 'step': d['stepMs'], 'bpm': d['bpm'],
                           'dur': d['durationSec'], 'onset': onset, 'rank': rank,
                           'accent': accent, 'rank_acc': rank_acc}
    return _fx_cache[song]


def onset_cell_metric(notes, cm):
    """起音格位命中：谱面用的 1/16 格位上，音频起音强度的百分位。

    这是本工程生成链路真正能用的口径——生成器吃的是每一步的 onset/accent 曲线，
    不是原始谱峰；峰距离（nearest peak）在这首歌上被 1559 个峰「压平」了
    （随机撒点也有 ~36ms），所以必须两个口径一起看。
      - cell_rank_mean：所用格位起音强度的平均百分位，0.5 = 与随机撒点等价
      - cell_strong_rate：落在起音强度前 30% 格位的比例，随机基线 0.30
    """
    fx = load_fx(cm['song'])
    off, step = fx['offset'], fx['step']
    ts = [x[0] for x in notes]
    out = {}
    for name, rkey in (('onset', 'rank'), ('accent', 'rank_acc')):
        rk_arr = fx[rkey]
        n = len(rk_arr)
        cells = [max(0, min(n - 1, int(round((t - off) / step)))) for t in ts]
        out[name + '_rank_mean'] = round(st.fmean(rk_arr[c] for c in cells), 3)
        out[name + '_strong_rate'] = round(
            sum(1 for c in cells if rk_arr[c] >= 0.70) / len(cells), 3)
    return out


def chance_alignment(notes, cm, seed=0):
    """对照组：如果「完全不看音频」会得到多少踩音误差？

    两个蒙特卡洛基线（同一张谱的音符数、同样的时间跨度，固定种子可复现）：
      - grid  ：随机撒在该曲的 1/16 网格上（等价于「只用节拍、不用 onset」）
      - uniform：随机撒在整个时间轴上（完全不看音频）
    再给一个整体平移扫描：最好的平移是多少 ms。
    生成谱如果和 grid 基线差不多，说明它其实没在用音频起音。
    """
    fx = load_fx(cm['song'])
    peaks = fx['peaks_ms']
    import random as _rnd
    rng = _rnd.Random(seed)
    ts = [t for t, c, e in notes]
    t0, t1 = ts[0], ts[-1]
    n = len(ts)
    actual = st.fmean(ac.nearest_peaks(peaks, ts))
    off, step = fx['offset'], fx['step']
    k0, k1 = int((t0 - off) / step), int((t1 - off) / step) + 1
    grid_rand = [off + rng.randint(k0, k1) * step for _ in range(n)]
    unif_rand = [rng.uniform(t0, t1) for _ in range(n)]
    shifts = {}
    for d in range(-80, 81, 4):
        shifts[d] = st.fmean(ac.nearest_peaks(peaks, [t + d for t in ts]))
    best_d = min(shifts, key=lambda k: shifts[k])
    return {
        'mean_ms': round(actual, 1),
        'chance_grid_ms': round(st.fmean(ac.nearest_peaks(peaks, grid_rand)), 1),
        'chance_uniform_ms': round(st.fmean(ac.nearest_peaks(peaks, unif_rand)), 1),
        'best_shift_ms': best_d,
        'mean_at_best_shift_ms': round(shifts[best_d], 1),
        'gain_vs_grid': round(st.fmean(ac.nearest_peaks(peaks, grid_rand)) - actual, 1),
    }


def alignment(notes, rows, cm):
    """返回对齐指标；谱面不对应 6 首歌则返回 {'skipped': 原因}。"""
    if not cm['song']:
        return {'skipped': '谱面未关联 out/multisong 的 6 首歌（AudioFilename 无法匹配），'
                           '踩音对齐不适用'}
    fx = load_fx(cm['song'])
    peaks = fx['peaks_ms']
    err = ac.nearest_peaks(peaks, [t for t, c, e in notes])
    rerr = ac.nearest_peaks(peaks, [t for t, cs in rows])
    if not err:
        return {'skipped': '空谱'}
    es = sorted(err)
    # 网格贴合：音符是否落在歌曲 1/16 网格上
    off, step = fx['offset'], fx['step']
    ongrid = sum(1 for t in (x[0] for x in notes)
                 if abs(((t - off) / step) - round((t - off) / step)) < 0.12)
    return {
        'song': cm['song'],
        'n_onsets': len(peaks),
        'mean_ms': round(st.fmean(err), 1),
        'median_ms': round(st.median(err), 1),
        'p90_ms': round(es[int(len(es) * 0.9)], 1),
        'within10': round(sum(1 for x in err if x <= 10) / len(err), 3),
        'within20': round(sum(1 for x in err if x <= 20) / len(err), 3),
        'within40': round(sum(1 for x in err if x <= 40) / len(err), 3),
        'row_median_ms': round(st.median(rerr), 1),
        'row_within15': round(sum(1 for x in rerr if x <= 15) / len(rerr), 3),
        'grid_fit': round(ongrid / len(notes), 3),
        'onset_cell': onset_cell_metric(notes, cm),
        'chance': chance_alignment(notes, cm),
    }


# ---------------------------------------------------------------------------
# 单张谱评估
# ---------------------------------------------------------------------------
def eval_chart(path, k, cents):
    notes, meta = ac.parse_osu(path)
    rows = ac.rows_of(notes)
    cm = chart_meta(notes, meta)
    step = cm['beat_ms'] / 4.0
    m = {'file': path, 'name': os.path.basename(path),
         'title': meta.get('Title'), 'version': meta.get('Version'),
         'notes': len(notes), 'rows': len(rows), 'bpm': cm['bpm'],
         'song': cm['song']}
    m.update(structural(notes, rows, step))
    m.update(playability(notes, rows) if notes else
             {'fast_same_col': None, 'side_overload': None, 'conflicts': None,
              'conflicts_per_min': None, 'max_col_hz': None})
    m['align'] = alignment(notes, rows, cm) if notes else {'skipped': '空谱'}

    # Koishi：难度 + 键型
    m['diff'] = k.get('diff')
    m['tier'] = k.get('tier')
    m['pattern'] = k.get('pattern')          # 不可靠（人类一致率 42~45%），仅参考
    probs = k.get('probs')
    m['probs'] = probs
    if probs:
        keys = sorted(probs)
        sims = {p: round(cosine(probs, cents[p]['probs'], keys), 3) for p in cents}
        m['cos'] = sims
        best = max(sims, key=lambda p: sims[p])
        m['best_pack'] = best
        m['best_cos'] = sims[best]
    else:
        m['cos'] = {p: None for p in cents}
        m['best_pack'], m['best_cos'] = None, None
    if k.get('error'):
        m['koishi_error'] = k['error']
    return m


# ---------------------------------------------------------------------------
# 聚合 + 表格
# ---------------------------------------------------------------------------
METRICS = [
    ('note_rate', '音符率 nps', 2),
    ('row_rate', '行率 rps', 2),
    ('npr', 'npr（每行音符）', 3),
    ('chord_rate', '和弦率', 3),
    ('ln_ratio', 'LN 占比', 3),
    ('jack_rate', 'jack 率', 3),
    ('p132', '1/32 占比', 3),
    ('hand_shift', '手位移（列）', 3),
    ('repeat8', '行重复率(8行窗)', 3),
    ('occupancy', '占位率', 4),
    ('conflicts_per_min', '冲突/分', 2),
    ('max_col_hz', '单列最高频率 Hz', 2),
    ('diff', 'Koishi diff', 2),
    ('best_cos', '最佳质心余弦', 3),
    ('cos_tech', 'cos·tech', 3),
    ('cos_jack', 'cos·jack', 3),
    ('cos_speed', 'cos·speed', 3),
    ('cos_stamina', 'cos·stamina', 3),
]


def _val(m, key):
    if key.startswith('cos_'):
        return (m.get('cos') or {}).get(key[4:])
    return m.get(key)


def aggregate(charts):
    agg = {}
    for key, label, nd in METRICS:
        vals = [v for v in (_val(c, key) for c in charts) if isinstance(v, (int, float))]
        if not vals:
            agg[key] = {'n': 0}
            continue
        vs = sorted(vals)
        agg[key] = {
            'n': len(vals),
            'mean': round(st.fmean(vals), nd + 1),
            'std': round(st.pstdev(vals), nd + 1) if len(vals) > 1 else 0.0,
            'median': round(st.median(vals), nd + 1),
            'p10': round(vs[int(len(vs) * 0.1)], nd + 1),
            'p90': round(vs[min(len(vs) - 1, int(len(vs) * 0.9))], nd + 1),
        }
    # 对齐指标单独聚合（不是所有谱都有）
    al = [c['align'] for c in charts if isinstance(c.get('align'), dict) and 'mean_ms' in c['align']]
    if al:
        agg['align'] = {
            'n': len(al),
            'mean_ms': round(st.fmean(x['mean_ms'] for x in al), 1),
            'median_ms': round(st.fmean(x['median_ms'] for x in al), 1),
            'within20': round(st.fmean(x['within20'] for x in al), 3),
            'within40': round(st.fmean(x['within40'] for x in al), 3),
            'row_within15': round(st.fmean(x['row_within15'] for x in al), 3),
            'grid_fit': round(st.fmean(x['grid_fit'] for x in al), 3),
            'onset_rank': round(st.fmean(x['onset_cell']['onset_rank_mean']
                                         for x in al if 'onset_cell' in x), 3),
            'accent_rank': round(st.fmean(x['onset_cell']['accent_rank_mean']
                                          for x in al if 'onset_cell' in x), 3),
            'accent_strong': round(st.fmean(x['onset_cell']['accent_strong_rate']
                                            for x in al if 'onset_cell' in x), 3),
        }
    agg['pattern_vote'] = dict(Counter(c.get('pattern') for c in charts if c.get('pattern')))
    agg['best_pack_vote'] = dict(Counter(c.get('best_pack') for c in charts if c.get('best_pack')))
    agg['diff_missing'] = sum(1 for c in charts if c.get('diff') is None)
    agg['n'] = len(charts)
    return agg


def _w(s):
    """显示宽度：CJK 记 2。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def _pad(s, width, right=False):
    n = width - _w(s)
    return (' ' * n + s) if right else (s + ' ' * n)


def fmt_table(groups, title=''):
    """groups: [(name, agg)] → 中文对齐表格（按显示宽度对齐）。"""
    W1, W2 = 22, 22
    head = _pad('指标', W1) + ''.join(_pad(n, W2, True) for n, _ in groups)
    lines = [title, head, '-' * _w(head)]
    for key, label, _ in METRICS:
        cells = []
        for _, a in groups:
            s = a.get(key) or {}
            if not s.get('n'):
                cells.append(_pad('—', W2, True))
            elif key == 'diff':
                cells.append(_pad(f"{s['mean']:.2f} ±{s['std']:.2f}", W2, True))
            else:
                cells.append(_pad(f"{s['mean']:.3f} ±{s['std']:.3f}", W2, True))
        lines.append(_pad(label, W1) + ''.join(cells))
    cells = []
    for _, a in groups:
        s = a.get('align') or {}
        cells.append(_pad((f"{s['mean_ms']:.0f}ms w20={s['within20']:.2f}" if s else '—'), W2, True))
    lines.append(_pad('踩音峰距 均值/w20', W1) + ''.join(cells))
    cells = []
    for _, a in groups:
        s2 = a.get('align') or {}
        cells.append(_pad((f"accent {s2['accent_rank']:.2f}" if s2 else '—'), W2, True))
    lines.append(_pad('起音格位百分位(0.5随机)', W1) + ''.join(cells))
    cells = []
    for _, a in groups:
        s2 = a.get('align') or {}
        cells.append(_pad((f"onset {s2['onset_rank']:.2f}" if s2 else '—'), W2, True))
    lines.append(_pad('onset格位百分位(0.5随机)', W1) + ''.join(cells))
    cells = []
    for _, a in groups:
        cells.append(_pad(f"diff空 {a.get('diff_missing', 0)}/{a.get('n', 0)}", W2, True))
    lines.append(_pad('Koishi diff 缺失', W1) + ''.join(cells))
    return '\n'.join(lines)


def compare_mode(args):
    """把多次评估的 JSON 合并成一张对比表。"""
    payloads = []
    for f in args.compare:
        try:
            payloads.append(json.load(open(f)))
        except Exception as e:
            raise SystemExit(f'读不了 {f}: {e}')
    cols = []
    for p in payloads:
        label = p['meta']['label']
        gs = p.get('groups') or {}
        if len(gs) > 1:
            for g, a in gs.items():
                cols.append((f'{label}:{g}', a))
        else:
            cols.append((label, p['total']))
    print(fmt_table(cols, '=== 对比表 ==='))
    # 分组合并版（所有组平铺，便于看每组）
    flat = []
    for p in payloads:
        for g, a in (p.get('groups') or {}).items():
            flat.append((f"{p['meta']['label']}:{g}", a))
    if len(flat) > len(cols):
        print()
        print(fmt_table(flat, '=== 分组明细 ==='))
    if args.json:
        json.dump({'columns': {n: a for n, a in cols}}, open(args.json, 'w'),
                  ensure_ascii=False, indent=1)
        print(f'\n合并 JSON → {args.json}')
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='.osu 路径（glob 已由 shell 展开）')
    ap.add_argument('--glob', action='append', default=[], help='额外 glob 模式')
    ap.add_argument('--pack-sample', type=int, default=0,
                    help='人类基线：每个包随机取 N 张（与 files 互斥）')
    ap.add_argument('--only-pack', default=None, choices=PACKS,
                    help='只对某一个人类包做 --pack-sample（用于分片并行跑基线）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label', default='gen')
    ap.add_argument('--json', default=None)
    ap.add_argument('--rebuild-centroid', action='store_true')
    ap.add_argument('--koishi-mode', choices=['batch', 'per-file'], default='batch',
                    help='batch=整批快跑；per-file=逐张跑（慢一点，但病态谱只赔一张）')
    ap.add_argument('--koishi-timeout', type=float, default=120,
                    help='per-file 模式单张墙钟上限（秒）')
    ap.add_argument('--extra-diff-ge', type=float, default=None,
                    help='额外输出一个「Koishi diff >= 该值」子集聚合（人类基线分难度段对照用）')
    ap.add_argument('--merge', nargs='+', default=[],
                    help='把多个 eval JSON 的 charts 合并重算聚合并写出（--json）')
    ap.add_argument('--compare', nargs='+', default=[],
                    help='只做对比：读入之前生成的 eval JSON，打印合并表（可多次给）')
    args = ap.parse_args()

    if args.merge:
        args.merge = args.merge + [f for f in args.files if f.endswith('.json')]
        charts = []
        for f in args.merge:
            charts.extend(json.load(open(f))['charts'])
        groups = {}
        for c in charts:
            groups.setdefault(c.get('group'), []).append(c)
        aggs = [(g, aggregate(cs)) for g, cs in groups.items()]
        total = aggregate(charts)
        if args.extra_diff_ge is not None:
            sub = [c for c in charts if isinstance(c.get('diff'), (int, float))
                   and c['diff'] >= args.extra_diff_ge]
            aggs.append((f'diff≥{args.extra_diff_ge:g}', aggregate(sub)))
        print(fmt_table(aggs + [('全部', total)], f'=== 合并表 · {args.label} ==='))
        payload = {'meta': {'script': 'tools/eval-generated.py', 'label': args.label,
                            'n_charts': len(charts), 'merged_from': args.merge,
                            'pack_sample': args.pack_sample or None, 'seed': args.seed,
                            'notes': []},
                   'charts': charts, 'groups': {g: a for g, a in aggs}, 'total': total}
        if args.json:
            json.dump(payload, open(args.json, 'w'), ensure_ascii=False, indent=1)
            print(f'\nJSON → {args.json}')
        return 0

    if args.compare:
        args.compare = args.compare + [f for f in args.files if f.endswith('.json')]
        return compare_mode(args)

    t_start = time.time()
    files, pack_of = [], {}
    if args.pack_sample:
        rng = random.Random(args.seed)
        for p in ([args.only_pack] if args.only_pack else PACKS):
            allf = sorted(glob.glob(os.path.join(PACK_DIR, p, '*.osu')))
            pick = rng.sample(allf, min(args.pack_sample, len(allf)))
            for f in pick:
                files.append(f)
                pack_of[f] = p
    for g in args.glob:
        files.extend(sorted(glob.glob(g)))
    files.extend(args.files)
    files = [f for f in dict.fromkeys(files) if os.path.exists(f)]
    if not files:
        raise SystemExit('没有输入谱面')

    _set_koishi_mode(args.koishi_mode, args.koishi_timeout)
    cents = load_centroids(rebuild=args.rebuild_centroid)
    ks = run_koishi(files, label=args.label)
    charts = []
    for f, k in zip(files, ks):
        c = eval_chart(f, k, cents)
        c['group'] = pack_of.get(f, args.label)
        charts.append(c)
        print(f"[eval] {c['name'][:34]:34s} n={c['notes']:5d} nps={c['note_rate'] or 0:6.2f} "
              f"diff={c['diff'] if c['diff'] is not None else '—':>6} "
              f"cos={c['best_cos'] if c['best_cos'] is not None else '—'}"
              f"({c['best_pack'] or '—'})", file=sys.stderr)

    groups = {}
    for c in charts:
        groups.setdefault(c['group'], []).append(c)
    aggs = [(g, aggregate(cs)) for g, cs in groups.items()]
    total = aggregate(charts)
    if args.extra_diff_ge is not None:
        sub = [c for c in charts if isinstance(c.get('diff'), (int, float))
               and c['diff'] >= args.extra_diff_ge]
        aggs.append((f'{args.label}(diff≥{args.extra_diff_ge:g})', aggregate(sub)))

    print()
    print(fmt_table(aggs + [('全部', total)], f'=== 评估表 · {args.label} ==='))

    payload = {
        'meta': {
            'script': 'tools/eval-generated.py',
            'label': args.label,
            'n_charts': len(charts),
            'pack_sample': args.pack_sample or None,
            'seed': args.seed,
            'elapsed_s': round(time.time() - t_start, 1),
            'centroids': {p: {'n': cents[p]['n'], 'probs': {k: round(v, 4) for k, v in cents[p]['probs'].items()},
                              'source': cents[p]['source']} for p in cents},
            'notes': [
                'Koishi pattern 标签不可靠（人类谱自身一致率仅 42~45%），只看 probs/余弦与 diff',
                'diff 为 null 表示 Koishi 判为低于 11 级（难度模型下限）',
                '1/32 占比 = 行间隔落在 1/32 拍（±50%）的比例，按谱面 bpm 归一到 1/16 格计数',
                '手位移 = 相邻行重心列索引差的绝对值均值（单位：列宽）',
                '行重复率 = 8 行滑窗签名中「重复出现」的比例',
                '可弹性冲突口径：同列 <55ms；单侧（0/1 或 2/3）55ms 内 ≥3 音；'
        '另给「单列最高频率 Hz」——1/16 网格(110ms)下 55ms 冲突不可能触发，只有 1/32 才有，'
        '所以 0 冲突不代表好弹，还要看单指要按多快',
            ],
        },
        'charts': charts,
        'groups': {g: a for g, a in aggs},
        'total': total,
    }
    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        json.dump(payload, open(args.json, 'w'), ensure_ascii=False, indent=1)
        print(f'\nJSON → {args.json}')
    print(f'耗时 {time.time() - t_start:.1f}s，{len(charts)} 张')


if __name__ == '__main__':
    main()
