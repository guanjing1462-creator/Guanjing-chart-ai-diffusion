#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paint-v11: 人类节奏词汇表 + 精确对音 + 旋律引导列位。

v10 的病（本机 audit 实测，Tech D15.8）：
  - 对音：音符到最近 onset 的平均误差 17.0ms、p90 35.4ms（因为落在
    「1/16 格中心」而不是真实 onset 上）→ 听着不贴音
  - 节奏：20% 的行间隔 <40ms（同格/半格挤爆），41% 在 81ms，行间隔分布
    与人类 tech/high 桶完全不是一回事 → 机枪感
  - 和弦：和弦行落在强 onset 的比例 43.4%，单音行 41.0%（没有区别）
    → 和弦不强调任何东西，听感平
  - 列位：谱质心（音高）与列移动的相关系数 ≈ -0.01（等于无关）
    → 旋律上下行不映射到左右移动，看着像随机撒点（"一坨"的主因）

v11 的四个替换：
  1. 节奏层：小节级模板（从 3.6 万张人类谱学来的 16 格 occupancy+chord
     模板），按「当前小节音频的 onset 能量/重音」打分选模板 → 节奏是
     人类的词汇表，不是任意撒点
  2. 时间层：每行时间取**最近的真实 onset 峰**（±半格内），不再用格中心
     → 对音误差从 17ms 降到 ~5ms
  3. 和弦层：和弦只在「该小节强 onset/强拍」上出现，弱格降级为单音
     → 和弦真正强调东西
  4. 列位层：音高（谱质心）方向引导列移动，叠加人类列转移矩阵、手交替、
     防意外 jack（≤2 格内禁止同列）→ 旋律可听、手型可打
  5. 结构层：4 小节乐句为动机单元，重复乐段的第二次出现复用第一次的
     列模式（可镜像/旋转变奏）→ 有呼应、有逻辑

用法：
  python3 tools/paint-v11.py --analysis out/final14-s50/analysis.json \
      --audio "INTERNET YAMERO.ogg" --style rice/technical/high --dps 10.5 \
      --version "v11 Tech" --seed 22 -o out.osu
"""
import argparse
import sys
import importlib.util
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np

CELL_PER_BAR = 16           # 1/16 基准网格；1/32 只由音频真实的连续击打触发
JACK_MODE = ['intent']
STYLE_CLASS = ['']          # rice / stream / tech / ln —— 决定和弦与连流策略      # 'intent'=信任人类转移矩阵 / 'forbid'=禁止同列连打
PITCH_THR = [0.03]
PATTERN = [None]          # (type_name, corpus_dict) 人类键型转移模型
SHAPE2 = [None]           # 二阶形状模型（train-shape-model.py 导出）
MOVE_BOOST = [0.0]        # 位移权重：>0 时偏好「离上一行远」的列（高难 tech 的难度来源）
DISJOINT_P = [0.30]       # 和弦→和弦强制「完全不共列」的概率（可通过 --disjoint-p 改）
RAW_TABLE = [0]           # 1=命中转移表时不叠加纠偏
ANCHOR_RPS = [0.0]        # 锚点稀释后的目标行频（0=未稀释）
ONSET_STR = [None]        # 与 _onset_ms 对齐的强度数组（保底用）
ONSET_ALL = [None]        # 稀释前的全量起音（补音要落在独立起音上）
COL_USE = [0, 0, 0, 0]    # 各列使用次数（用于均衡）
COL_BAL = [0.0]           # 列均衡强度（0=关）
# 人谱的列偏好（tech 包实测）：单音爱用中列、双押爱用外列
PREF_SINGLE = [0.212, 0.289, 0.287, 0.211]
PREF_CHORD = [0.273, 0.226, 0.226, 0.274]
UPGRADE_FAR = [0.0]       # 升级和弦时选最远列的概率（提高 单→和 位移）
ANCHOR_P = [0.0]          # 和弦锚定强度（共列的候选加权，0=关）
NEAR_PEN = [0.0]          # 单↔和 距离惩罚强度（0=关，1=按上表打折）
COL_PRIOR = [0.0]         # 列偏好先验强度（0=关）
SRC = {}                  # 诊断：各路径产出的行/单音数
BARSTAT = {'want': 0, 'got': 0, 'bars': 0, 'capped_by_occ': 0, 'capped_by_env': 0}
REJ = {}   # 拆音被拒的原因计数
SRC_COL = {}              # 诊断：各路径的列直方图
BAR_COL = [None]          # {bar: [4 次计数]} 小节级列配额
BAR_MS = [1297.4]         # 小节时长（ms），由 cell_ms*16 设定


def bar_of(t):
    return int(float(t) // max(1.0, BAR_MS[0]))


def bar_col_get(b):
    if BAR_COL[0] is None:
        BAR_COL[0] = {}
    return BAR_COL[0].setdefault(b, [0, 0, 0, 0])


def pick_col_balanced(cands, t, prev_c=None, fallback_random=True):
    """在所有放置路径里统一使用：优先本小节用得最少的列。"""
    if not cands:
        return None
    b = bar_of(t)
    cnt = bar_col_get(b)
    # 小节计数 + 全局计数一起看：拆音占 54% 的行，只靠小节计数在 1-2 个候选间没有区分度
    _gmax = max(1, max(COL_USE))
    # 人类单音偏好中列（0.212/0.289/0.287/0.211）：把偏好折进评分，
    # 否则「不重复前后列」的强制交替会把四列均匀轮转，永远对不上
    # 拆音占 54% 的行，但它只能在「前后行没占用的列」里挑 → 天然偏外侧（实测 0.33/0.20/0.19/0.29）。
    # 加一条「靠近上一行」的偏好，把拆音往中间带。
    _score = {c: cnt[c] * 2.0 + COL_USE[c] * 4.0 / _gmax
                 - 2.2 * float(np.log(max(0.05, PREF_SINGLE[c] / 0.25)))
                 + ((0.55 - MOVE_BOOST[0]) * abs(c - prev_c) if prev_c is not None else 0.0)
              for c in cands}
    mn = min(_score[c] for c in cands)
    best = [c for c in cands if _score[c] <= mn + 0.35]
    ch = best[int(np.random.randint(len(best)))] if len(best) > 1 else best[0]
    cnt[ch] += 1      # 自计数：这个函数每选一次就记一次，否则配额永远是 0
    return ch


def bar_col_add(t, c):
    bar_col_get(bar_of(t))[c] += 1
SAME_HAND_T = [0.0]      # 同手双押目标占比（0=关）
SAME_HAND_K = [1.0]      # 校准力度
PRIOR_ALPHA = [0.0]
SHAPE_PRIOR = [None]      # {size: {shape_tuple: prob}} 人谱形状边际（教采样「人的双押长什么样」）
TARGETS = [None]          # 该类型的硬指标（jack 率 / 手位移），用于实时纠偏
JACK_P = [0.005]          # 旋律方向门控阈值（按本曲 |ptrend| p55 自适应）
CHORD_PAIRS_SAME = [(0, 1), (2, 3)]
CHORD_PAIRS_ALT = [(0, 2), (1, 3), (0, 3), (1, 2)]
# 人类 D15-15.7 实测双押形状权重（40302 个和弦）：
#   同手相邻 (0,1)/(2,3) 占 32%，宽跨 (0,3) 12%，其余跨手 26%
# 我们原来是反的：跨手 (0,2)/(1,3) 占 27%，同手只有 21% —— 听感就是「一叠一叠很重」
PAIR_W = {(0, 1): 0.161, (2, 3): 0.159, (0, 3): 0.120,
          (1, 3): 0.087, (1, 2): 0.086, (0, 2): 0.086}
CHORD_TRI = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
STRONG_W = [3.0, 0, 0, 0, 2.0, 0, 0, 0, 2.6, 0, 0, 0, 2.0, 0, 0, 0]


# ---------------------------------------------------------------- 音频特征
def audio_features(ogg, hop_ms, cache_dir='out/audio-cache'):
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(f'{os.path.abspath(ogg)}|{os.path.getsize(ogg)}|{hop_ms}'.encode()).hexdigest()[:16]
    cp = os.path.join(cache_dir, key + '.npz')
    if os.path.exists(cp):
        z = np.load(cp)
        return {k: z[k] for k in z.files}
    import librosa
    y, sr = librosa.load(ogg, sr=22050, mono=True)
    hop = int(round(sr * hop_ms / 1000.0))
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    pm = (freqs >= 200) & (freqs <= 2500)
    ptot = S[pm].sum(axis=0) + 1e-9
    pitch = np.log2(np.maximum((S[pm] * freqs[pm, None]).sum(axis=0) / ptot, 100.0))
    def band(a, b):
        m = (freqs >= a) & (freqs < b)
        return np.log1p(S[m].sum(axis=0))
    low, mid, high = band(20, 160), band(160, 1500), band(4000, 12000)
    logmag = np.log1p(S.sum(axis=0))
    flux = np.maximum(0.0, np.diff(logmag, prepend=logmag[:1]))
    os_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    out = {'pitch': pitch.astype(np.float32), 'low': low.astype(np.float32),
           'onsetlib': np.maximum(os_env, 0).astype(np.float32),
           'mid': mid.astype(np.float32), 'high': high.astype(np.float32),
           'flux': flux.astype(np.float32), 'hopMs': np.float32(hop_ms)}
    np.savez_compressed(cp, **out)
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def cell_table(an, af):
    """每个 1/16 格（stepMs）的聚合特征。"""
    off, step16 = an['offsetMs'], an['stepMs']
    step = step16 * 16.0 / CELL_PER_BAR          # 1/32 格 = 40.54ms
    dur = an['durationSec'] * 1000.0
    n = int((dur - off) / step) + 1
    hop = float(af['hopMs'])
    fine_e = np.asarray(an.get('fine', {}).get('energy', []), dtype=np.float64)
    peaks_ms = sorted(p * hop for p in an['peaks'])
    import bisect
    acc = an['accent']
    ons_arr = an.get('onset') or [0.0] * len(acc)
    # 半格能量：从 fine.energy（11.6ms）聚合，比 1/16 accent 细一倍
    def cell_energy(t0, t1):
        f0 = max(0, int(t0 / hop))
        f1 = max(f0, min(len(fine_e), int(t1 / hop)))
        if fine_e.size and f1 > f0:
            return float(np.mean(fine_e[f0:f1]))
        return 0.0
    acc_med_pre = 1.0
    rows = []
    nfr = len(af['pitch'])
    for i in range(n):
        t0 = off + i * step
        f0 = max(0, min(nfr - 1, int(t0 / hop)))
        f1 = max(f0 + 1, min(nfr, int((t0 + step) / hop)))
        sl = slice(f0, f1)
        rows.append({
            'i': i, 't0': t0, 'center': t0 + step / 2,
            'peak': None, 'peakErr': 99.0,
            # 直接用分析器的 1/16 格数组（同网格）：accent=电平，onset=起音强度
            'accent': float(acc[i]) if i < len(acc) else 0.0,
            'onset': float(ons_arr[i]) if i < len(ons_arr) else 0.0,
            'flux': float(np.max(af['flux'][sl])) if f1 > f0 else 0.0,
            'low': float(np.mean(af['low'][sl])) if f1 > f0 else 0.0,
            'mid': float(np.mean(af['mid'][sl])) if f1 > f0 else 0.0,
            'high': float(np.mean(af['high'][sl])) if f1 > f0 else 0.0,
            # 分频带起音（格子内正向差分最大值）：「这一格有没有鼓/旋律/镲的击打」
            'on_low': float(np.max(np.maximum(np.diff(af['low'][max(0, f0 - 1):f1]), 0))) if f1 > f0 else 0.0,
            'on_mid': float(np.max(np.maximum(np.diff(af['mid'][max(0, f0 - 1):f1]), 0))) if f1 > f0 else 0.0,
            'on_high': float(np.max(np.maximum(np.diff(af['high'][max(0, f0 - 1):f1]), 0))) if f1 > f0 else 0.0,
            'on_lib': float(np.max(af['onsetlib'][sl])) if f1 > f0 else 0.0,
            'pitch': float(np.median(af['pitch'][sl])) if f1 > f0 else 0.0,
            'pitchStd': float(np.std(af['pitch'][sl])) if f1 > f0 else 0.0,
        })
    # 旋律走向（ptrend）：人类是「乐句层面的音高上行 → 手位整体右移」，
    # 不是每 40ms 一个方向。所以先用 5 格滑动平均取轮廓，再取与 2 格前的差
    # 作为方向信号——单步 delta 太小太抖，之前门控几乎不触发。
    raw_p = np.array([r['pitch'] for r in rows], dtype=np.float64)
    if len(raw_p) >= 5:
        ker = np.ones(5) / 5.0
        sm = np.convolve(raw_p, ker, mode='same')
        vals = []
        for k, r in enumerate(rows):
            back = max(0, k - 4)          # 4 个 1/32 格 ≈ 162ms，一个乐句步进
            v = float(sm[k] - sm[back]) if k - back >= 2 else 0.0
            r['ptrend'] = v
            vals.append(abs(v))
        vals = np.array(vals)
        # 门控阈值按本曲分布定（固定阈值 0.06 在这首歌上几乎不触发）
        thr = float(np.percentile(vals, 55)) if len(vals) else 0.03
        PITCH_THR[0] = max(0.02, thr)
        print(f'旋律方向阈值 {PITCH_THR[0]:.3f} log2Hz（|ptrend| p55）')
    # 每格定「出谱时间 + 显著度」：
    #   时间：格内 onset 强度足够 → 取格内 flux 峰值（真实起音，ms 级）；
    #        否则用网格线（off + i*step，节拍器级）。
    #   显著度：onset 强度（真实起音）为主，电平为辅——**不能只看电平**，
    #        电平高但没有起音的格（持续音）不该无脑放音符。
    ons = np.array([r['onset'] for r in rows], dtype=np.float64)
    lev = np.array([r['accent'] for r in rows], dtype=np.float64)
    ons_p90 = float(np.percentile(ons, 90)) + 1e-9
    lev_p50 = float(np.percentile(lev, 50)) + 1e-9
    flux_arr = af['flux']
    for r in rows:
        t0 = r['t0']
        f0 = max(0, min(len(flux_arr) - 1, int(t0 / hop)))
        f1 = max(f0 + 1, min(len(flux_arr), int((t0 + step) / hop)))
        seg = flux_arr[f0:f1]
        # 时间一律用网格线：网格已经过独立 onset 包络联合拟合验证
        # （信噪 0.784→2.655，折叠 profile 0/4/8/12 拍点最强）。
        # 不要用格内 flux 峰：flux 的相对格起点滞后 +38ms（STFT 窗+差分造成），
        # 用它落点会让所有音符迟到——这正是「重音不在准拍子上」的来源。
        r['time'] = float(t0)
        r['is_onset'] = bool(seg.size and r['onset'] >= 0.30 * ons_p90)
        r['peakErr'] = 0.0 if r['is_onset'] else float(step / 2)
        r['salience'] = 0.72 * (r['onset'] / ons_p90) + 0.28 * (r['accent'] / lev_p50)
    return rows, peaks_ms


def bar_phase(cells):
    """小节相位：让强拍格能量最大（音频侧，与词汇表学到的相位同口径）。"""
    best, bp = -1.0, 0
    for ph in range(CELL_PER_BAR):
        sc = sum(c['accent'] * STRONG_W[(c['i'] - ph) % CELL_PER_BAR] for c in cells)
        if sc > best:
            best, bp = sc, ph
    return bp


# ---------------------------------------------------------------- 段落
def detect_sections(cells, phase, bars_per_sec=4):
    """4 小节一段，按音频特征给段落打等价组。返回 [(start_bar, end_bar, gid, feats)]。"""
    nb = (cells[-1]['i'] - phase) // CELL_PER_BAR + 1
    sec_bars = bars_per_sec
    secs = []
    for s0 in range(0, nb, sec_bars):
        s1 = min(nb, s0 + sec_bars)
        sel = [c for c in cells if s0 <= (c['i'] - phase) // CELL_PER_BAR < s1]
        if not sel:
            secs.append({'s0': s0, 's1': s1, 'feat': [0.0] * 6})
            continue
        f = [float(np.mean([c['accent'] for c in sel])),
             float(np.mean([c['flux'] for c in sel])),
             float(np.mean([c['low'] for c in sel])),
             float(np.mean([c['high'] for c in sel])),
             float(np.mean([c['pitch'] for c in sel])),
             float(np.std([c['pitch'] for c in sel]))]
        secs.append({'s0': s0, 's1': s1, 'feat': f})
    # 归一化后分组（余弦 > 0.93 视为同一段音乐）
    F = np.array([s['feat'] for s in secs], dtype=np.float64)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    gids, gi = [], 0
    gid_of = {}
    for k in range(len(secs)):
        hit = None
        for j in range(k):
            if k - j < 1:
                continue
            if float(F[k] @ F[j]) > 0.80:
                hit = gids[j]
                break
        if hit is None:
            gids.append(gi)
            gi += 1
        else:
            gids.append(hit)
    for k, s in enumerate(secs):
        s['gid'] = gids[k]
        gid_of.setdefault(gids[k], []).append(k)
    return secs, gid_of


def section_targets(secs, cells, phase, src_rps, dyn=0.45, onset_ms=None, cell_ms=81.0898, fill_mult=1.0, onset_str=None, str_floor=0.0):
    """段能量 → 段目标行频（相对 dps 的调制，钳 ±dyn）。"""
    e = np.array([s['feat'][0] for s in secs])
    lo, hi = np.percentile(e, 15), np.percentile(e, 90)
    lo = min(lo, hi - 1e-9)
    w = (e - lo) / (hi - lo)
    w = np.clip(w, -0.2, 1.4)
    rel = 1.0 + dyn * (w - w.mean()) / (w.std() + 1e-9) * 0.5
    rel = np.clip(rel, 1 - dyn, 1 + dyn)
    # —— 密度由音频决定 ——
    # 之前 rps 是「按能量比例乘一个基准」（src_rps），等于凭感觉定密度。
    # 实测这首歌的 onset 间隔分布是 1/8 占 54%、1/16 占 35%、1/32 只占 3%，
    # 也就是音乐本身的脉冲约 7.7 行/s；而我们一直在写 1/16（10.5 行/s），
    # 比音乐快 36% —— 听感就是「碎」和「采音不对」。
    _ot = onset_ms if onset_ms is not None else np.array([])
    for k, s in enumerate(secs):
        _t_base = cells[0]['time'] if cells else 0.0
        t0 = _t_base + s['s0'] * CELL_PER_BAR * cell_ms
        t1 = _t_base + (s['s0'] + (s.get('nb') or 4)) * CELL_PER_BAR * cell_ms
        # 只统计「够强」的起音：之前按全部起音算密度，遇到「弱音很多」的段落（镲片密）
        # 会把密度抬上去 → 音符落在最弱 30% 的起音上（实测 30-38s 段有 49%）
        _mw = (_ot >= t0) & (_ot < t1)
        if len(_ot) and onset_str is not None and str_floor > 0:
            _thr_s = float(np.percentile(onset_str, str_floor))
            _mw &= (onset_str >= _thr_s)
        n_on = int(np.sum(_mw))
        dur_s = max(0.3, (t1 - t0) / 1000.0)
        audio_rps = n_on / dur_s
        # 音乐脉冲 → 目标行频：clamp 在 4~13，并保留能量带来的相对起伏
        # 观景台「大拍技」实测是音乐 onset 率的 1.8 倍（16.7 行/s vs 9.3 onset/s），
        # 所以段落目标 = 音频 onset 率 × style 的补音倍数（--fill-mult）
        # 段落起伏：你的谱段间行数 CV 是 0.41，我们原来只有 0.28（太均匀=单调）。
        # 用段能量的百分位排名做 0.55~1.45 倍的调制，把 CV 拉到人类水平。
        _rank = float((w < w[k]).mean()) if len(w) > 1 else 0.5
        s['rps'] = float(np.clip(audio_rps * fill_mult, 3.0, 22.0)) \
            * (0.30 + 1.30 * _rank)   # 加强起伏：安静段 0.30×、最响段 1.60×
                                      # （原来 0.55-1.45 太窄，30-38s 那种弱音段仍被写得很密）
        s['audio_rps'] = audio_rps
        s['quiet'] = float(w[k]) < 0.12
        s['intensity'] = float(min(1.0, max(0.0, w[k])))   # 0=最弱段 1=最强段
    # —— 难度定标：音频只决定「段间起伏的形状」，总密度由 target-ett 决定 ——
    # 之前 src_rps 完全没用上，导致 ETT 14/18/24 生成出完全相同的谱（难度参数是死的）。
    if src_rps and src_rps > 0 and secs:
        _cur = float(np.mean([s['rps'] for s in secs]))
        if _cur > 0.3:
            _k = float(src_rps) / _cur
            _k = float(np.clip(_k, 0.35, 2.5))
            for s in secs:
                s['rps'] = float(np.clip(s['rps'] * _k, 2.5, 24.0))
    return secs


# ---------------------------------------------------------------- 节奏模板
def pick_template(rnd, bar_cells, templates, want_rows, quiet=False):
    """在词汇表里挑最贴「本小节音频 onset 分布」的人类节奏模板。

    关键修正（v11.1）：模板必须**落在真实 onset 上**。否则模板会在没有音的
    格子上放行，随后被对音硬约束丢掉 → 密度塌、节奏和音乐脱节。
    """
    if quiet and rnd.random() < 0.5:
        return None                     # 气口：直接空小节
    best, bestsc = None, -1e9
    for tp in templates:
        occ, ch = tp['occ'], tp['chord']
        nr = len(occ)
        if abs(nr - want_rows) > max(2, want_rows * 0.5):
            continue
        bmean = (sum(bar_cells[p]['accent'] for p in occ
                     if p < len(bar_cells) and bar_cells[p]) / max(1, nr))
        bref = sum(c['accent'] for c in bar_cells.values() if c) / max(1, len(bar_cells))
        support = sum(1 for p in occ
                      if p < len(bar_cells) and bar_cells[p]
                      and bar_cells[p]['accent'] >= bref * 0.85) if occ else 0
        sup = support / max(1, nr)
        align = 0.0
        for p in occ:
            if p < len(bar_cells) and bar_cells[p]:
                c = bar_cells[p]
                align += c['salience']
        align /= max(1, nr)
        chsc = sum(STRONG_W[p] / 3.0 for p in ch) / max(1, len(ch)) if ch else 0.0
        dens = -abs(nr - want_rows) / max(1.0, want_rows)
        sc = 3.4 * (sup - 0.62) + 0.8 * math.log1p(align) + 0.3 * chsc + 1.0 * dens \
             + 0.5 * math.log(tp['w'] + 1e-6) + rnd.random() * 0.2
        if sc > bestsc:
            bestsc, best = sc, tp
    if best is not None:
        bref2 = sum(c['accent'] for c in bar_cells.values() if c) / max(1, len(bar_cells))
        support = sum(1 for p in best['occ']
                      if p < len(bar_cells) and bar_cells[p]
                      and bar_cells[p]['accent'] >= bref2 * 0.85)
        if support / max(1, len(best['occ'])) < 0.5:
            return None                 # 支撑太差 → 交给 onset 兜底
    return best


# ---------------------------------------------------------------- 列位
def norm_trans(fp):
    tr = np.array(fp['col_trans'], dtype=np.float64)
    tr = np.maximum(tr, 0.0) + 0.5
    return tr / tr.sum(axis=1, keepdims=True)


def sample_cols(rnd, state, trans, gap, pitch_d, chord_size, fp, accent_rel,
                pitch_w=1.0, thr=0.06, cell_ms=81.0898):
    """给一行选列。gap=距上一行的格数；pitch_d=音高变化（log2 Hz）。"""
    prev = state['prev_col']
    recent = state['recent']
    load = state['load']
    tot = sum(load.values()) + 1e-9
    pref_dir = 0
    if pitch_d is not None:
        if pitch_d > thr:
            pref_dir = 1
        elif pitch_d < -thr:
            pref_dir = -1
    def base_score(c):
        s = trans[prev][c]
        # 同列：intent 模式下不做特别惩罚（人类转移矩阵的对角线就是 jack 率），
        # 只压「三连击以上」，避免变成死板列锤
        if c == prev:
            if len(recent) >= 2 and recent[-2] == c and gap <= 2:
                s *= 0.25
            elif gap <= 1:
                s *= 0.85
        # 手交替：1 格流的跨手优先
        if gap <= 1 and ((c < 2) != (prev < 2)):
            s *= 1.55
        # 连流手感：stream 档偏好「单调阶梯」（同方向 ±1），做出连续流的形状
        if STYLE_CLASS[0] == 'stream' and gap <= 1 and len(recent) >= 2:
            step_dir = prev - recent[-2]
            if step_dir != 0 and (c - prev) == step_dir:
                s *= 1.7
            elif (c - prev) == -step_dir:
                s *= 0.45
        elif gap <= 1 and len(recent) >= 2:
            step_dir = prev - recent[-2]
            if step_dir != 0 and (c - prev) == step_dir:
                s *= 1.25
        # 旋律方向：上行偏右、下行偏左
        if pref_dir != 0:
            span = (c - prev) * pref_dir
            if span > 0:
                s *= 1.0 + pitch_w * (1.0 + 0.5 * min(2, span))
            elif span < 0:
                s *= max(0.05, 1.0 - pitch_w * (0.45 + 0.2 * min(2, -span)))
            else:
                s *= max(0.1, 1.0 - pitch_w * 0.55)
        # 负载均衡
        s *= 1.0 + 0.55 * (1.0 - load[c] / tot * 4 / 4)
        # 2 行前同列 → 3 连击惩罚
        if len(recent) >= 2 and recent[-2] == c and gap <= 2:
            s *= 0.25
        return s
    forbid = set()
    if gap <= 2:
        prev_cols = set(state.get('prev_cols') or ())
        if chord_size <= 1 and JACK_MODE[0] != 'forbid':
            # 单音同列连打 = 「叠」。人类 D15-15.7 实测只有 **2.7%**，
            # 我们原来是 21-31%（为了拉站点 D 放开过，听感就是一路叠）。
            # 这里默认禁掉，只保留 3% 概率做少量真正的 jack 味。
            if rnd.random() < JACK_P[0]:
                pass
            else:
                forbid = prev_cols
        if JACK_MODE[0] == 'forbid':
            forbid = prev_cols
        elif chord_size <= 1 and len(prev_cols) >= 2:
            # 和弦之后的那一下：不能复用和弦已经按下的列。人类 D13-14 的
            # 和弦→单音同列只有 0.025，我们原来是 0.166（6.6 倍），听感是
            # 「手指黏在同一个键上」
            forbid = prev_cols
        elif chord_size >= 2:
            # 和弦**不要**禁掉上一行的列：人类 D15-15.7 的「和弦→和弦」
            # 有 70% 是共用列的（锚和弦/叠和弦），禁掉就变成一堆散和弦
            forbid = set()
    if PATTERN[0] is not None:
        ptype, corpus = PATTERN[0]
        dt = corpus['types'][ptype]
        ms = gap * cell_ms
        bucket = ('32' if ms <= 60 else '16' if ms <= 110 else
                  '8' if ms <= 190 else '4' if ms <= 330 else
                  '2' if ms <= 650 else '1' if ms <= 1200 else 'slow')
        previous = sorted(set(state.get('prev_cols') or ()))
        _tab_used = False
        # 二阶模型（训练得到）：上下文 = 前两行形状 | 上一行形状 | 节奏档
        _t2 = SHAPE2[0]
        _tab2 = {}
        if _t2:
            _p2 = tuple(sorted(set(state.get('prev2_cols') or ())))
            # 与 train-shape-model.py 导出时的键格式保持一致：'(0, 1)|(2,)|16'
            _key = f'{_p2}|{tuple(previous)}|{bucket}'
            _tab2 = _t2.get(_key) or {}
        tables = [_tab2,
                  dt['trans'].get(f'{previous}|{bucket}', {}),
                  dt['marg'].get(bucket, {})]
        for table in tables:
            # 这一轮候选是否来自「真正的转移表」（而不是不看上一行的边际分布）
            _tab_used = bool(table) and table in (tables[0], tables[1])
            candidates, weights = [], []
            for encoded, weight in table.items():
                shape = json.loads(encoded)
                if (len(shape) != chord_size or len(set(shape)) != len(shape)
                        or any(c not in range(4) for c in shape)
                        or not np.isfinite(weight) or weight <= 0):
                    continue
                if ptype not in ('jack', 'chordjack') and set(shape) & forbid:
                    continue
                candidates.append(shape); weights.append(weight)
            if candidates:
                # 和弦→和弦：包里只有 69% 共列，我们是 94% → 三成情况要求完全不共列
                if (len(previous) > 1 and chord_size > 1
                        and state.get('cluster_chord')):
                    # 簇内和弦：必须与上一和弦共列（人类簇内平均位移 0.71，我们 1.1+）
                    anchored = [c for c in candidates if set(c) & set(previous)]
                    if anchored:
                        keep = [(c, w) for c, w in zip(candidates, weights) if set(c) & set(previous)]
                        candidates = [k[0] for k in keep]
                        weights = [k[1] for k in keep]
                elif (len(previous) > 1 and chord_size > 1
                        and rnd.random() < DISJOINT_P[0]):
                    pairs = [(c, w) for c, w in zip(candidates, weights)
                             if not (set(c) & set(previous))]
                    if pairs:
                        candidates = [p0[0] for p0 in pairs]
                        weights = [p0[1] for p0 in pairs]
                # —— 目标保持器 ——
                # 人类局部转移拼接起来会漂：实测 jack 类单音同列 0.324 vs 人类 0.228，
                # stream 手位移 1.61 vs 1.44。这里按「人类目标 vs 当前累计」实时修正权重。
                tgt = TARGETS[0] or {}
                stat = state.setdefault('stat', {'ss': 0, 'jack': 0, 'move': 0.0, 'n': 0})
                prev_single = len(previous) == 1
                r_jack = stat['jack'] / max(1, stat['ss'])
                r_move = stat['move'] / max(1, stat['n'])
                t_jack = tgt.get('jackRate')
                t_move = tgt.get('handMove')
                # 硬上限：累计已经比人类更叠时，直接不许再出同列（软权重压不住
                # 「该 context 里 jack 是唯一候选」的情况，实测软权重无效）
                if (t_jack is not None and prev_single and r_jack > t_jack * 1.05
                        and len(candidates) > 1):
                    keep = [(sh, w) for sh, w in zip(candidates, weights)
                            if not (len(sh) == 1 and sh[0] == previous[0])]
                    if keep:
                        candidates = [k[0] for k in keep]
                        weights = [k[1] for k in keep]
                # 双向纠偏：低于目标时也要往上拉（V18 只压不拉，Tech 手位移卡在 1.06/1.34）
                want_more_jack = (t_jack is not None and r_jack < t_jack * 0.90)
                want_more_move = (t_move is not None and r_move < t_move * 0.92)
                adj = []
                for shape, w in zip(candidates, weights):
                    w = float(w)
                    is_jack = prev_single and len(shape) == 1 and shape[0] == previous[0]
                    if is_jack and t_jack is not None:
                        # 比人类更叠 → 压制；比人类更松 → 放宽
                        w *= float(np.clip((t_jack / max(1e-6, r_jack)) ** 1.6, 0.15, 4.0))
                        if want_more_jack:
                            w *= 1.9           # 明显欠叠时主动挑同列
                    elif t_jack is not None and r_jack > t_jack * 1.15 and len(shape) == 1:
                        w *= 1.25          # 已经太叠时，非 jack 的单音加权
                    if t_move is not None and previous:
                        mv = float(np.mean([min(abs(c - q) for q in previous) for c in shape]))
                        # 只在「位移低于目标」时开一个轻微的行间激励（形状先验管行内，两者可分开）
                        if RAW_TABLE[0] and _tab_used:
                            expo = 1.0 if want_more_move else 0.0
                        else:
                            expo = 0.90 if want_more_move else 0.35
                        w *= float(np.exp(-expo * (mv - t_move)))
                        if r_move > t_move * 1.10 and mv > t_move:
                            w *= 0.75
                    # 形状先验：把人谱的「形状边际」乘进权重，纠正条件分布长时间漂移
                    _pr_all = SHAPE_PRIOR[0]
                    if _pr_all and PRIOR_ALPHA[0] > 0:
                        _pr = _pr_all.get(len(shape))
                        if _pr:
                            _pw = _pr.get(tuple(sorted(shape)), 1e-4)
                            w *= float(_pw) ** PRIOR_ALPHA[0]
                    adj.append(max(1e-9, w))
                # 和弦锚定：人谱 stamina 的相邻和弦几乎不动（和→和 0.31），
                # 我们每拍换手（0.73）。给「与上一个和弦共列」的候选加权。
                if len(previous) >= 2 and chord_size >= 2 and ANCHOR_P[0] > 0:
                    _an = []
                    for _sh in candidates:
                        _share = len(set(_sh) & set(previous))
                        _an.append(1.0 + ANCHOR_P[0] * _share)
                    adj = [w * f for w, f in zip(adj, _an)]
                # 单↔和 距离约束：人谱 单→和 位移 1.08，我们只有 0.82 —— 和弦贴着相邻单音。
                # 按最小列距做软惩罚（0 列距最重、1 列距次之），把这个过渡拉回人类水平。
                if len(previous) == 1 and chord_size >= 2 and NEAR_PEN[0] > 0:
                    _pp = []
                    for _sh in candidates:
                        _dm = min(abs(c - previous[0]) for c in _sh)
                        _f = (0.20, 0.65, 1.0, 1.0)[min(3, int(_dm))]
                        _pp.append(1.0 + (NEAR_PEN[0] - 1.0) * (1.0 - _f))
                    adj = [w * f for w, f in zip(adj, _pp)]
                # 列偏好先验：单音偏中列、双押偏外列（人类谱的稳定习惯，实测差 0.06-0.08）
                if COL_PRIOR[0] > 0:
                    _pref = PREF_SINGLE if chord_size == 1 else PREF_CHORD
                    _cp = []
                    for _sh in candidates:
                        _w0 = 1.0
                        for _c0 in _sh:
                            _w0 *= _pref[_c0]
                        _cp.append(_w0)
                    _m = sum(_cp) / max(1, len(_cp))
                    if _m > 0:
                        adj = [w * (f / _m) ** COL_PRIOR[0] for w, f in zip(adj, _cp)]
                # 列使用率均衡：用得少的列加权（实测基座采样让第 0 列偏多 3.5%）
                if COL_BAL[0] > 0 and sum(COL_USE) > 60:
                    _mean_use = sum(COL_USE) / 4.0
                    _cb = []
                    for _sh in candidates:
                        _w0 = 1.0
                        for _c0 in _sh:
                            _w0 *= (max(1.0, _mean_use) / max(1.0, COL_USE[_c0])) ** COL_BAL[0]
                        _cb.append(_w0)
                    adj = [w * f for w, f in zip(adj, _cb)]
                # 同手形状校准（在候选集整体上做）：把「同手双押」的质量调到目标占比。
                # 人谱 tech 46.7%，我们自然漂到 ~35%（上一行的形状会自洽地锁住）。
                _sht = SAME_HAND_T[0]
                if _sht > 0 and chord_size == 2 and len(candidates) > 1:
                    _sh = [(a <= 1) == (b <= 1) for a, b in candidates]
                    _m_same = sum(w for w, m in zip(adj, _sh) if m)
                    _m_oth = sum(w for w, m in zip(adj, _sh) if not m)
                    if _m_same > 0 and _m_oth > 0:
                        _need = _sht / max(1e-6, 1 - _sht)
                        _cur = _m_same / _m_oth
                        _k = (_need / _cur) ** SAME_HAND_K[0]
                        adj = [w * (_k if m else 1.0) for w, m in zip(adj, _sh)]
                pick = list(rnd.choices(candidates, weights=adj)[0])
                for _c0 in pick:
                    COL_USE[_c0] += 1
                stat['n'] += 1
                stat['move'] += float(np.mean([min(abs(c - q) for q in previous) for c in pick])) if previous else 0.0
                if prev_single and len(pick) == 1:
                    stat['ss'] += 1
                    stat['jack'] += int(pick[0] == previous[0])
                state['pattern_hits'] = state.get('pattern_hits', 0) + 1
                _k = 'pat_single' if len(pick) == 1 else 'pat_chord'
                SRC[_k] = SRC.get(_k, 0) + 1
                _h = SRC_COL.setdefault(_k, [0, 0, 0, 0])
                for _c0 in pick:
                    _h[_c0] += 1
                return pick
        state['pattern_fallbacks'] = state.get('pattern_fallbacks', 0) + 1
    if chord_size <= 1:
        cands = [c for c in range(4) if c not in forbid] or list(range(4))
        if pitch_w > 0 and pref_dir != 0:
            gated = [c for c in cands if ((c - prev) * pref_dir) > 0]
            if len(gated) >= 1 and rnd.random() < 0.93:
                cands = gated
        sc = np.maximum(np.array([base_score(c) for c in cands]), 1e-9)
        return [int(rnd.choices(cands, weights=sc.tolist())[0])]
    if chord_size >= 4:
        return [0, 1, 2, 3]
    cands = list(CHORD_TRI) if chord_size == 3 else CHORD_PAIRS_SAME + CHORD_PAIRS_ALT
    free = [p for p in cands if not (set(p) & forbid)]
    if free:
        cands = free
    def cscore(pair):
        s = 1.0
        if chord_size == 2:
            s *= PAIR_W.get(tuple(pair), 0.07) / 0.086
        # 和弦应该包含旋律列附近（prev 的 ±1）——锚和弦
        if prev in pair:
            s *= 0.55 if gap <= 2 else 1.0
        if (prev + 1) % 4 in pair or (prev - 1) % 4 in pair:
            s *= 1.35
        # 音高方向：和弦重心顺着旋律走
        if pitch_w > 0 and pref_dir != 0:
            com = sum(pair) / len(pair) - prev
            if com * pref_dir > 0:
                s *= 1.0 + pitch_w * 0.7
            elif com * pref_dir < 0:
                s *= max(0.08, 1.0 - pitch_w * 0.6)
        # 强 onset 优先级更高
        s *= 1.0 + 0.5 * accent_rel
        s *= 1.0 + 0.2 * sum(1.0 - load[c] / tot * 4 for c in pair)
        return s
    if chord_size == 2:
        # 双押形状直接按人类分布抽（只过滤掉与上一行冲突的形状），
        # 保证形状分布与人类一致；其他打分只用于「允许集合为空」时的兜底。
        pv = set(state.get('prev_cols') or ())
        ok = [p for p in cands if not (set(p) & forbid)]
        pool = ok or cands
        # 人类 D15-15.7「和弦→和弦」共列 0.70（不是 100%）：三成情况下要求
        # 与上一和弦完全不共列，避免变成一路锚死
        if pv and rnd.random() < 0.30:
            dis = [p for p in pool if not (set(p) & pv)]
            if dis:
                pool = dis
        w = np.array([PAIR_W.get(tuple(p), 0.05)
                      for p in pool], dtype=np.float64)
        if w.sum() <= 0:
            w = np.ones(len(pool))
        return list(pool[int(rnd.choices(range(len(pool)), weights=(w / w.sum()).tolist())[0])])
    sc = np.array([cscore(p) for p in cands])
    sc = np.maximum(sc, 1e-9)
    return list(cands[int(rnd.choices(range(len(cands)), weights=sc.tolist())[0])])


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', required=True)
    ap.add_argument('--audio', required=True)
    ap.add_argument('--style', required=True, help='指纹桶，如 rice/technical/high')
    ap.add_argument('--fingerprints', default='out/style-fingerprints.json')
    ap.add_argument('--lexicon', default='out/rhythm-lexicon.json')
    ap.add_argument('--density-target', type=float, default=0.0,
                    help='行频硬目标（行/s）；>0 时会削薄多余的行（低难度用）')
    ap.add_argument('--npr-from-ett', type=int, default=0,
                    help='1=用难度反推 npr（会牺牲档位 npr）；0=固定为档位语料值（人谱中 npr 不随难度变）')
    ap.add_argument('--upgrade-far', type=float, default=0.0,
                    help='升级和弦时选最远列的概率（人谱 单→和 1.62，我们 1.42）')
    ap.add_argument('--anchor-p', type=float, default=0.0,
                    help='和弦锚定强度：与上一个和弦共列的候选加权（人谱和→和位移 0.31）')
    ap.add_argument('--near-pen', type=float, default=0.0,
                    help='和弦贴相邻单音的惩罚强度（0=关，1=最强）')
    ap.add_argument('--anchor-group', type=float, default=0.0,
                    help='连续和弦段的锚定概率（共享一列，模拟人谱的和弦锚定）')
    ap.add_argument('--spread-mirror', type=float, default=1.0,
                    help='和弦紧贴相邻单音时尝试镜像拉开（0=关）')
    ap.add_argument('--loop-bars', type=int, default=0,
                    help='循环式动机：每 N 小节重复一次键型并吸附到最近起音格（0=关，实验性）')
    ap.add_argument('--bar-motif', type=float, default=0.0,
                    help='小节级动机重复概率（行数相同的小节抄前面小节；Jack 档建议 0.7）')
    ap.add_argument('--motif-cluster', type=float, default=0.0,
                    help='把特征距离小于该阈值的段落也当作可复用组（0=只用等价组）')
    ap.add_argument('--motif-mirror', type=float, default=0.25,
                    help='段落复用时的左右镜像概率（0=完全照抄，重复度最高）')
    ap.add_argument('--section-motif', type=float, default=0.0,
                    help='段落级动机复用概率（同等价组抄列；0=关，建议 0.6）')
    ap.add_argument('--motif-len', type=int, default=16,
                    help='动机重复的窗口行数（人谱多为 1-2 小节）')
    ap.add_argument('--motif-repeat', type=float, default=0.0,
                    help='动机重复概率（人谱 tech 0.3 / jack 0.6；0=关）')
    ap.add_argument('--run-fill', type=float, default=0.0,
                    help='成串补音（宽空档里补 2 个 1/32，不贴和弦）0=关')
    ap.add_argument('--chord-slot', type=float, default=0.0,
                    help='生成阶段预选和弦槽，拆音绕开（0=关）')
    ap.add_argument('--chord-room', type=float, default=0.0,
                    help='给和弦清出 ±45ms 空位（0=关）')
    ap.add_argument('--chain-rewrite', type=float, default=1.0,
                    help='连续单音段按人谱单音转移模型重排列走向（0=关）')
    ap.add_argument('--col-prior', type=float, default=0.0,
                    help='列偏好先验强度（单音偏中列/双押偏外列；0=关，1=按人谱概率直接加权）')
    ap.add_argument('--col-balance', type=float, default=0.0,
                    help='列使用率均衡强度（0=关；0.3-0.5 可消除第 0 列偏多）')
    ap.add_argument('--same-hand-target', type=float, default=0.0,
                    help='把「同手双押」占比校准到该目标（人谱 tech 0.467）；0=关')
    ap.add_argument('--shape-prior', type=float, default=0.0,
                    help='形状边际先验的强度（0=关；0.5-0.8 会明显把人谱的双押形状拉回来）')
    ap.add_argument('--chord-shape-fix', type=float, default=0.0,
                    help='按人谱边际重排双押形状的概率（0=关闭）')
    ap.add_argument('--move-boost', type=float, default=0.0,
                    help='位移权重（>0 偏好远离上一行的列，用来给高难 tech 增加难度来源）')
    ap.add_argument('--quantize', type=int, default=0, choices=[0, 16, 32],
                    help='音符时间固化到 1/16 或 1/32 网格')
    ap.add_argument('--grid-snap-ms', type=float, default=0.0,
                    help='把补音时间吸附到 1/32 网格的容差（ms），0=不吸附；降低节奏混沌度')
    ap.add_argument('--fill-snap-ms', type=float, default=0.0,
                    help='补音吸附到真实起音的容差（ms）；0=不吸附（默认，'
                         '实测吸附会把 1/32 从 0.52 打到 0.35，精度收益不值）')
    ap.add_argument('--onset-strength-floor', type=float, default=45.0,
                    help='段落密度只统计能量高于该分位的起音（0=不过滤）。防止弱音段落被写密')
    ap.add_argument('--anchor-mult', type=float, default=1.0,
                    help='锚点间隔倍率：>1 更稀（低难度用），<1 更密')
    ap.add_argument('--dps-fit', choices=['pack', 'feature'], default='feature',
                    help='难度→密度拟合来源：feature=档位保真（默认，36 张跨歌实测最好）、'
                         'pack=难度保真（ETT 12-25 误差 ≤1.0）')
    ap.add_argument('--auto-dps', type=int, default=1,
                    help='1=按 --target-ett 从 3.6 万份人谱拟合线自动定密度（行/s = 0.411*ETT+0.57）')
    ap.add_argument('--dps', type=float, default=0.0, help='目标行频（行/s）；0=按 ETT 自动')
    ap.add_argument('--ln-open-space', type=float, default=0.0,
                    help='为长条主动腾空间的概率（删掉体内的同列音符）')
    ap.add_argument('--npr-scale', type=float, default=0.0,
                    help='npr 目标整体缩放（难度阶梯用：低难度 0.85、高难度 1.15）')
    ap.add_argument('--ln-len-dist', type=str, default='',
                    help='覆盖 LN 长度分布，如 1:0.35,2:0.3,4:0.2,8:0.15')
    ap.add_argument('--ln-cap', type=float, default=None)
    ap.add_argument('--dyn', type=float, default=0.45, help='段间密度起伏幅度')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--version', required=True)
    ap.add_argument('--title', default='INTERNET YAMERO')
    ap.add_argument('--artist', default='KOTOKO')
    ap.add_argument('--creator', default='mug-prototype')
    ap.add_argument('--audio-name', default='', help='留空=用 --audio 的文件名（打包 osz 时必须一致）')
    ap.add_argument('--structure-out', default=None)
    ap.add_argument('--no-pitch', action='store_true', help='关闭音高引导列位（消融用）')
    ap.add_argument('--ev-thr', type=float, default=0.30, help='音频节奏模式的绝对阈值下限')
    ap.add_argument('--ev-mult', type=float, default=1.05, help='音频节奏模式的相对中位阈值倍数')
    ap.add_argument('--song-id', default=None, help='音频特征 id（默认从 --audio 文件名推）')
    ap.add_argument('--model', default=None,
                    help='人类谱训练出的落点模型（out/placement.pt），给了就用它排序候选格')
    ap.add_argument('--target-ett', type=float, default=15.0,
                    help='用模型时假定的目标难度（ett 量级），决定它会选多密的落点')
    ap.add_argument('--jack-p', type=float, default=0.005,
                    help='有意制造同列连打（jack）的概率；观景台 Jack 向谱约 0.023')
    ap.add_argument('--jack-target', type=float, default=0.0,
                    help='单音同列连打的目标比例（分类包实测值）；低于它就主动把下一音压到同列')
    ap.add_argument('--onset-multiband', type=int, default=1,
                    help='1=分三频带检峰再合并（能在密集段找到更多真实击打）')
    ap.add_argument('--section-swing', type=float, default=0.0,
                    help='段落级 npr 摆动幅度：0=逐行均匀；>0 时按 8 小节块交替偏单音/偏和弦，'
                         '让单音成段出现（人类的 jack 也是这样形成的）')
    ap.add_argument('--fix-playable', type=int, default=1,
                    help='修复物理上弹不了的模式（同列 40ms 连打、单侧连续和弦）')
    ap.add_argument('--accent-cover', type=float, default=0.0,
                    help='强起音覆盖率目标（0=不处理）；缺口会在 ±60ms 内补一个音符')
    ap.add_argument('--raw-table', type=int, default=0,
                    help='1=命中转移表时不再叠加手位移/jack 纠偏（表本身已是人类分布）')
    ap.add_argument('--chord-cluster-p', type=float, default=0.0,
                    help='上一行是和弦时，这一行继续出和弦的概率（人类谱的和弦是成簇的）')
    ap.add_argument('--disjoint-p', type=float, default=0.30,
                    help='和弦→和弦时强制「完全不共列」的概率（会拉高手位移；0=尽量锚列）')
    ap.add_argument('--npr-mode', choices=['quota', 'prop'], default='quota',
                    help='npr 控制方式：quota=按配额（不振荡）、prop=按比例（Stamina 类更灵）')
    ap.add_argument('--size-boost', type=float, default=1.0,
                    help='基座行大小采样时和弦的额外权重（>1 更爱出和弦，用来补 npr）')
    ap.add_argument('--fill-prob', type=float, default=1.0,
                    help='每个拆音位置实际插入的概率（用来精确控制 1/32 占比）')
    ap.add_argument('--fill-chord-p', type=float, default=0.30,
                    help='拆出来的音里有多大概率是双押（用来把 npr 拉回包的水平）')
    ap.add_argument('--fill-thr-pct', type=float, default=35.0,
                    help='拆音门限的全曲能量分位（越低拆得越多；包里 1/32 多就调低）')
    ap.add_argument('--fill-spacing', type=float, default=0.0,
                    help='把间隔拆细到的目标格数（0.5=1/32，1.0=1/16）；0=不拆')
    ap.add_argument('--sub32', type=float, default=0.0,
                    help='在相邻 1/16 行之间插 1/32 补音的概率（观景台大拍技以 1/32 为主）')
    ap.add_argument('--fill-max-per-gap', type=int, default=0,
                    help='每个 1/16 空档最多补几个音（1=只允许 1/32，禁止 1/3 三连）')
    ap.add_argument('--fill-mult', type=float, default=1.6,
                    help='在「音频 onset 锚点」之间补音的程度：1.0=只写 onset，1.8≈观景台大拍技')
    ap.add_argument('--gap-bias', type=float, default=0.0,
                    help='间隔模型的密度偏置：>0 偏向更短的间隔（更密）')
    ap.add_argument('--gap-model', default=None,
                    help='节奏间隔模型（out/gap.pt）：用「下一个音符隔几格」走完全曲')
    ap.add_argument('--ptype', default=None,
                    help='键型类型（jack/chordjack/stream/jumpstream/handstream/stamina/technical）：'
                         '直接按人类该类型的转移统计采样下一行的键型')
    ap.add_argument('--shape-model', default='',
                    help='二阶形状模型（train-shape-model.py 导出）')
    ap.add_argument('--pattern-corpus', default='out/pattern-corpus.json')
    ap.add_argument('--strict', action='store_true',
                    help='严格采音：只在音频有起音的格子上放音符，不补音/不保底/不插 1/32')
    ap.add_argument('--rhythm', choices=['gap', 'anchor', 'onset', 'audio', 'template'], default='anchor',
                    help='audio=节奏由音频分频带起音决定（默认，采音优先）；template=旧的人类模板选格')
    ap.add_argument('--style-class', choices=['auto', 'rice', 'stream', 'tech', 'ln'],
                    default='auto',
                    help='键型大类：rice/stream=纯单音流（和弦极少），tech=随行大小分布出和弦，'
                         'ln=同 tech + 长条')
    ap.add_argument('--jack-mode', choices=['intent', 'forbid'], default='intent',
                    help='intent=按人类列转移矩阵的允许同列连打（默认）；'
                         'forbid=旧行为（≤2 格内禁止同列，实测会把 jack 逼成跳键）')
    ap.add_argument('--chord-boost', type=float, default=0.0,
                    help='和弦升音概率（>0 时强 onset 上和弦加一个音，用来拉 D）')
    ap.add_argument('-o', required=True)
    args = ap.parse_args()
    # 小节时长：1/16 步长 × 16 格
    if not getattr(args, 'audio_name', ''):
        args.audio_name = os.path.basename(args.audio)
    # 拆音比例随难度自动标定（实测：ETT12 fp0.3→7.2 nps，fp0.8→11.1；ETT20 fp0.8→14.7）
    if getattr(args, 'fill_prob', 1.0) < 0:
        args.fill_prob = float(np.clip(0.021 * float(args.target_ett) + 0.30, 0.15, 1.0))
        print(f'拆音比例自动：ETT {args.target_ett} → fill-prob {args.fill_prob:.2f}')
    if args.dps <= 0:
        # 人谱拟合（feature-db 36k+ 张图）：行/s = 0.411*ETT + 0.57
        if getattr(args, 'dps_fit', 'pack') == 'pack':
            args.dps = float(max(2.0, 0.094 * float(args.target_ett) + 7.76))
        else:
            args.dps = float(max(2.0, 0.411 * float(args.target_ett) + 0.57))
        print(f'自动密度：ETT {args.target_ett} → {args.dps:.2f} 行/s')
    JACK_P[0] = args.jack_p
    DISJOINT_P[0] = args.disjoint_p
    RAW_TABLE[0] = args.raw_table
    PRIOR_ALPHA[0] = args.shape_prior
    SAME_HAND_T[0] = args.same_hand_target
    COL_BAL[0] = args.col_balance
    COL_PRIOR[0] = args.col_prior
    NEAR_PEN[0] = args.near_pen
    ANCHOR_P[0] = args.anchor_p
    UPGRADE_FAR[0] = args.upgrade_far
    if args.ptype:
        _pc = json.load(open(args.pattern_corpus, encoding='utf-8'))
        PATTERN[0] = (args.ptype, _pc)
        _dt = _pc['types'].get(args.ptype, {})
        _sz = _dt.get('sizes') or {}
        _npr = (sum(int(k) * v for k, v in _sz.items()) / sum(_sz.values())) if _sz else None
        TARGETS[0] = {'jackRate': _dt.get('jackRate'),
                      'handMove': (_dt.get('hand') or {}).get('mean'),
                      'nPerRow': _npr}
        try:
            from collections import Counter as _C
            _agg = {}
            for _bk, _tab in (_dt.get('marg') or {}).items():
                for _k, _v in _tab.items():
                    _sh = tuple(json.loads(_k))
                    _agg.setdefault(len(_sh), _C())[_sh] += _v
            _pr = {}
            for _sz, _c in _agg.items():
                _t = sum(_c.values()) or 1
                _pr[_sz] = {k: v / _t for k, v in _c.items()}
            SHAPE_PRIOR[0] = _pr
            print(f'形状先验：{ {k: len(v) for k, v in _pr.items()} }（按尺寸的形状数）')
        except Exception as _e:
            SHAPE_PRIOR[0] = None
        print(f'键型模型：{args.ptype}（{_pc["types"].get(args.ptype, {}).get("n", 0)} 行样本的转移统计）')
        if getattr(args, 'shape_model', ''):
            try:
                _sm = json.load(open(args.shape_model))
                SHAPE2[0] = _sm.get('trans2') or None
                _hl = (_sm.get('loss_holdout') or {})
                _best = _sm.get('best', '')
                print(f'二阶形状模型：{len(SHAPE2[0] or {})} 个上下文'
                      f'（留出损失 {_hl.get(_best, 0):.3f} bits/样本，最优 {_best}）')
            except Exception as _e:
                SHAPE2[0] = None
                print(f'二阶模型加载失败：{_e}')
    MOVE_BOOST[0] = float(getattr(args, 'move_boost', 0.0) or 0.0)

    fps = json.load(open(args.fingerprints))
    if args.style not in fps:
        s, p = args.style.split('/', 1)
        for d in ('high', 'mid', 'low'):
            if f'{s}/{p}/{d}' in fps:
                args.style = f'{s}/{p}/{d}'
                break
    assert args.style in fps, f'指纹库没有 {args.style}'
    fp = fps[args.style]
    lex = json.load(open(args.lexicon))
    templates = (lex['buckets'].get(args.style) or lex['global'])['templates']
    JACK_MODE[0] = args.jack_mode
    if args.style_class != 'auto':
        STYLE_CLASS[0] = args.style_class
    else:
        _cls, _pat = (args.style.split('/') + [''])[:2]
        STYLE_CLASS[0] = 'ln' if _cls == 'ln' else ('stream' if _pat == 'stream' else 'tech')
    print(f'键型大类: {STYLE_CLASS[0]}')
    an = json.load(open(args.analysis))
    bpm = float(an['bpm'])
    off = float(an['offsetMs'])
    step = float(an['stepMs'])
    BAR_MS[0] = step * 16.0
    cell_ms = step * 16.0 / CELL_PER_BAR          # 出谱网格（1/32 = 40.54ms）
    rnd = random.Random(args.seed)

    af = audio_features(args.audio, an['fine']['hopMs'])
    cells, peaks_ms = cell_table(an, af)
    phase = bar_phase(cells)
    print(f'音频特征: {len(cells)} 格 / 小节相位 {phase}')
    # 每格所在小节与格内位置
    for c in cells:
        c['bar'] = (c['i'] - phase) // CELL_PER_BAR
        c['pos'] = (c['i'] - phase) % CELL_PER_BAR
    bar_of = defaultdict(dict)
    for c in cells:
        if c['bar'] >= 0:
            bar_of[c['bar']][c['pos']] = c

    secs, gid_of = detect_sections(cells, phase)
    import librosa as _lr
    _y, _sr = _lr.load(args.audio, sr=22050, mono=True)
    _env = np.maximum(_lr.onset.onset_strength(y=_y, sr=_sr, hop_length=256), 0)
    _fr = _lr.util.peak_pick(_env, pre_max=3, post_max=3, pre_avg=8, post_avg=8,
                             delta=0.12, wait=2)
    _onset_ms = _fr * 1000.0 * 256 / _sr
    if getattr(args, 'onset_multiband', True):
        # 单路检测在密集段会漏掉一半击打（实测全曲只有 1857 个峰 = 7.7/s，
        # 而 tech 包里 60% 的音符间隔 <60ms）。这里分三频带各自检峰再合并去重。
        _S = np.abs(_lr.stft(_y, n_fft=2048, hop_length=256))
        _freqs = _lr.fft_frequencies(sr=_sr, n_fft=2048)
        _bands = [(0, 200), (200, 2000), (2000, 8000)]
        _merged = list(_onset_ms)
        for _lo, _hi in _bands:
            _m = (_freqs >= _lo) & (_freqs < _hi)
            if not _m.any():
                continue
            _band_env = np.maximum(np.diff(np.log1p(_S[_m].sum(0)), prepend=0), 0)
            _bmed = float(np.median(_band_env)) + 1e-9
            _bp = _lr.util.peak_pick(_band_env, pre_max=2, post_max=2, pre_avg=5, post_avg=5,
                                     delta=0.55 * _bmed, wait=1)
            _merged.extend((_bp * 1000.0 * 256 / _sr).tolist())
        _merged.sort()
        _dedup = []
        for _t in _merged:
            if not _dedup or _t - _dedup[-1] >= 22.0:
                _dedup.append(_t)
        _onset_ms = np.array(_dedup)
        ONSET_ALL[0] = _onset_ms.copy()   # 稀释前的全量起音（补音要用它判断「独立起音」）
    _fps_s0 = float(_sr) / 256.0
    _str0 = np.array([_env[min(len(_env) - 1, int(t / 1000.0 * _fps_s0))] for t in _onset_ms]) \
        if len(_onset_ms) else None
    ONSET_STR[0] = _str0
    secs = section_targets(secs, cells, phase, args.dps, args.dyn,
                           onset_ms=_onset_ms, cell_ms=step, fill_mult=args.fill_mult,
                           onset_str=_str0, str_floor=float(args.onset_strength_floor))
    # —— npr 作为难度杠杆 ——
    # 行数被「必须落在起音上」限制在 ~0.85×起音率，所以更高难度只能靠和弦：
    #   nps_target = 0.777*ETT + 0.23（3.6 万份人谱拟合）
    #   npr_target = nps_target / 实际行频
    if TARGETS[0] and args.target_ett and getattr(args, 'npr_from_ett', 0):
        _rows = ANCHOR_RPS[0] if ANCHOR_RPS[0] > 0 else \
            (float(np.mean([x['rps'] for x in secs])) if secs else args.dps)
        _nps_t = 0.777 * float(args.target_ett) + 0.23
        _npr_t = float(np.clip(_nps_t / max(0.5, _rows), 1.0, 4.0))
        print(f'难度杠杆：ETT {args.target_ett:.0f} → nps 目标 {_nps_t:.1f}，'
              f'行频目标 {_rows:.2f}/s → npr 目标 {_npr_t:.2f}（原 {TARGETS[0].get("nPerRow")}）')
        TARGETS[0]['nPerRow'] = _npr_t
    # 难度 → npr 缩放：低难度应该「又稀又少和弦」（人类低难度 npr 1.1-1.3），
    # 之前 npr 由 nps/行频反推，低难度反而变成高 npr（因为行频被音频卡住）。
    if TARGETS[0] and getattr(args, 'npr_scale', 0) > 0:
        _base = float(TARGETS[0].get('nPerRow') or 1.33)
        TARGETS[0]['nPerRow'] = _base * args.npr_scale
        print(f'npr 缩放 ×{args.npr_scale:.2f}：{_base:.2f} → {TARGETS[0]["nPerRow"]:.2f}')
    print(f'[诊断] 检测起音 {len(_onset_ms)} 个'
          f'（{len(_onset_ms) / max(1.0, (_onset_ms[-1] - _onset_ms[0]) / 1000.0):.1f}/s）'
          f'，时长 {(_onset_ms[-1] - _onset_ms[0]) / 1000.0:.0f}s')
    _dbg_rps = float(np.mean([s['rps'] for s in secs])) if secs else 0.0
    print(f'[诊断] 段落目标行频平均 {_dbg_rps:.2f}/s  （--dps={args.dps}, anchor_mult={args.anchor_mult}）')
    # —— 难度 = 锚点密度（在采样前决定，而不是事后删行） ——
    # 上轮教训：事后删行会把相邻行并成大间隔 → 1/32 从 0.6 掉到 0.35、手位移从 1.3 掉到 0.9。
    # 正确做法：只保留最强的 X% 起音作为锚点，图案文法在更少的锚点上正常运行。
    ANCHOR_RPS[0] = 0.0
    _onset_all = _onset_ms.copy()   # 稀释前的全量起音（强起音补采用这个，而不是锚点集）
    if args.auto_dps and args.target_ett and len(_onset_ms) > 8:
        # 直接测量拟合（57 张 pack 谱 × 各自 ETT）：行/s = 0.094*ETT + 7.76
        if getattr(args, 'dps_fit', 'pack') == 'pack':
            _tgt_rps = 0.094 * float(args.target_ett) + 7.76
        else:
            _tgt_rps = 0.411 * float(args.target_ett) + 0.57
        ANCHOR_RPS[0] = _tgt_rps
        _mid = 1000.0 / max(0.5, _tgt_rps) * float(args.anchor_mult)
        _fps_l = float(_sr) / 256.0
        _env_s = np.array([_env[min(len(_env) - 1, int(t / 1000.0 * _fps_l))]
                           for t in _onset_ms]) if _env is not None else np.zeros(len(_onset_ms))
        _order = np.argsort(-_env_s)
        _kept = []
        _sorted_kept = []
        for _ix in _order:
            _t = float(_onset_ms[_ix])
            _k = int(np.searchsorted(_sorted_kept, _t))
            _ok = True
            if _k > 0 and abs(_t - _sorted_kept[_k - 1]) < _mid:
                _ok = False
            if _ok and _k < len(_sorted_kept) and abs(_sorted_kept[_k] - _t) < _mid:
                _ok = False
            if _ok:
                _sorted_kept.insert(_k, _t)
                _kept.append(_t)
        if len(_kept) >= 8:
            print(f'锚点稀释：ETT {args.target_ett:.0f} → 目标 {_tgt_rps:.2f} 行/s，'
                  f'onset {len(_onset_ms)} → {len(_kept)}')
            _onset_ms = np.array(sorted(_kept))
    # 锚点稀释之后重算 npr 目标（口径必须与锚点行频一致，否则低难度偏低）
    if TARGETS[0] and ANCHOR_RPS[0] > 0 and args.target_ett and getattr(args, 'npr_from_ett', 0):
        if getattr(args, 'dps_fit', 'pack') == 'pack':
            _nps2 = (0.094 * float(args.target_ett) + 7.76) * \
                    max(1.05, 0.062 * float(args.target_ett) + 0.16)
        else:
            _nps2 = 0.777 * float(args.target_ett) + 0.23
        TARGETS[0]['nPerRow'] = float(np.clip(_nps2 / ANCHOR_RPS[0], 1.0, 4.0))
        print(f'npr 目标按锚点行频重算：nps {_nps2:.1f} / 行频 {ANCHOR_RPS[0]:.2f} '
              f'→ {TARGETS[0]["nPerRow"]:.2f}')
    onset_cells = set(int(round((t - (cells[0]['time'] - (cells[0]['i']) * step)) / step))
                      for t in _onset_ms) if len(_onset_ms) else set()
    # —— 人类谱训练的「间隔模型」：走位生成 ——
    gap_cells = set()
    if args.gap_model:
        import torch as _th2
        _gs = importlib.util.spec_from_file_location(
            'bdg', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build-note-dataset.py'))
        _bdg = importlib.util.module_from_spec(_gs); _gs.loader.exec_module(_bdg)
        _gt = importlib.util.spec_from_file_location(
            'tg', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train-gap.py'))
        _tg = importlib.util.module_from_spec(_gt); _gt.loader.exec_module(_tg)
        _melg = np.load(f'out/audio_features/{args.song_id}.npz')['mel']
        _ncg = len(cells)
        _fxt, _ = _bdg.chart_features(_melg, an['offsetMs'], step * 4, _ncg,
                                      [(0, 16), (16, 32), (32, 48), (48, 64),
                                       (64, 80), (80, 96), (96, 112), (112, 128)])
        _gck = _th2.load(args.gap_model, map_location='cpu', weights_only=False)
        _gnet = _tg.GapNet(_gck['d_in'])
        _gnet.load_state_dict(_gck['state']); _gnet.eval()
        _prev = 8.0
        _i = 0
        _rng = random.Random(args.seed + 77)
        while _i < _ncg:
            gap_cells.add(_i)
            _en = min(40.0, max(0.0, args.target_ett)) / 40.0
            _x = np.concatenate([_fxt[_i].astype(np.float32),
                                 np.array([_prev / 8.0, _en, _en * _en], dtype=np.float32)])
            with _th2.no_grad():
                _lg = _gnet(_th2.from_numpy(_x[None]))[0]
                _pr = _th2.softmax(_lg / 0.9, dim=0).numpy()
            _pr[0] = 0.0
            if args.gap_bias:
                _kk = np.arange(len(_pr), dtype=np.float64)
                _pr = _pr * np.where(_kk > 0, _kk ** (-args.gap_bias), 0.0)
            _pr = _pr / _pr.sum()
            _k = int(_rng.choices(range(len(_pr)), weights=_pr.tolist())[0])
            _k = max(1, _k)
            # 落点吸附到最近的音频 onset（±1 格内），兼顾「节奏像人」和「采得到音」
            _j = _i + _k
            if _j not in onset_cells:
                for _d in (1, -1):
                    if (_j + _d) in onset_cells and _j + _d > _i:
                        _j += _d
                        break
            _k = max(1, _j - _i)
            _prev = min(8, _k)
            _i = _j
        print(f'间隔模型走位：{len(gap_cells)} 个落点（{len(gap_cells) / (len(cells) * step / 1000.0):.1f} 行/s）')
        onset_cells = set(gap_cells) | onset_cells

    # —— 人类谱训练的落点模型 ——
    if args.model:
        import torch as _th
        _spec = importlib.util.spec_from_file_location(
            'bd', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build-note-dataset.py'))
        _bd = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_bd)
        _ck = _th.load(args.model, map_location='cpu', weights_only=False)
        _net = _th.nn.Sequential()
        # 复用训练脚本的网络结构
        _ts = importlib.util.spec_from_file_location(
            'tp', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train-placement.py'))
        _tp = importlib.util.module_from_spec(_ts)
        _ts.loader.exec_module(_tp)
        _net = _tp.Net(_ck['d_in'])
        _net.load_state_dict(_ck['state'])
        _net.eval()
        # 换歌时没有 mel 特征文件是正常的：降级为不使用落点模型，而不是崩掉
        _mel = None
        if args.song_id:
            _melp = f'out/audio_features/{args.song_id}.npz'
            if os.path.exists(_melp):
                try:
                    _mel = np.load(_melp)['mel']
                except Exception:
                    _mel = None
        if _mel is not None:
            _beat = step * 4
            _ncf = len(cells)
            _fx, _ = _bd.chart_features(_mel, an['offsetMs'], _beat, _ncf,
                                        [(0, 16), (16, 32), (32, 48), (48, 64),
                                         (64, 80), (80, 96), (96, 112), (112, 128)])
            _e = np.clip(args.target_ett, 0, 40) / 40.0
            _X = np.concatenate([_fx.astype(np.float32),
                                 np.full((_ncf, 1), _e, dtype=np.float32),
                                 np.full((_ncf, 1), _e ** 2, dtype=np.float32)], axis=1)
            with _th.no_grad():
                _lo, _ls = _net(_th.from_numpy(_X))
                _p = _th.sigmoid(_lo).numpy()
                _ps = _th.softmax(_ls, dim=1).numpy()
            for _i, _c in enumerate(cells):
                _c['model_p'] = float(_p[_i]) if _i < len(_p) else 0.0
                _c['model_size'] = _ps[_i] if _i < len(_ps) else np.array([1, 0, 0, 0, 0])
            print(f'落点模型已载入：目标难度 ett≈{args.target_ett}，'
                  f'全曲 P 中位 {float(np.median(_p)):.3f}')
    # 每小节的「平均起音强度」：电平正常但起音极弱 = 持续音/渐弱段，
    # 那里应该留长音或休息，不能一路铺 16 分（实测尾声 220-240s 的「空音」全在这）
    _t_base0 = cells[0]['time'] if cells else 0.0
    _bar_env = {}
    _nb_total = int(len(cells) / CELL_PER_BAR) + 1
    for _b in range(_nb_total):
        _t0 = _t_base0 + _b * CELL_PER_BAR * step
        _t1 = _t0 + CELL_PER_BAR * step
        _f0 = max(0, int(_t0 / 1000.0 * (_sr / 256.0)))
        _f1 = min(len(_env), int(_t1 / 1000.0 * (_sr / 256.0)) + 1)
        _bar_env[_b] = float(_env[_f0:_f1].mean()) if _f1 > _f0 else 0.0
    _bar_env_med = float(np.median(list(_bar_env.values()))) if _bar_env else 1.0
    print(f'段落: {len(secs)} 段（{len(gid_of)} 个等价组，重复组 '
          f'{sum(1 for v in gid_of.values() if len(v) > 1)} 个）')

    # 段 → 小节映射
    bar_sec = {}
    for k, s in enumerate(secs):
        for b in range(s['s0'], s['s1']):
            bar_sec[b] = k

    # ---- 段落驱动生成：动机单元（1/2/4 小节）在段内 + 重复段间复用 ----
    cells_by_i = {c['i']: c for c in cells}
    trans_n = norm_trans(fp)
    acc_vals = [c['accent'] for c in cells]
    acc_med = float(np.median(acc_vals)) + 1e-9
    # 「空音」闸门：全曲电平 p20 以下的格子一律不放音符。人类谱在低能量处
    # 几乎不放音符（实测低能量处音符比例 0.000），我们原来是 1%。
    acc_floor = float(np.percentile(acc_vals, 20))
    # 和弦只能落在全曲最响的那批时刻：人类谱（观景台雪月花）和弦平均在第 86 分位，
    # 我们原来是 75 分位 —— 这就是「重音没踩准」的量化来源
    acc_p78 = float(np.percentile(acc_vals, 78))
    low_vals = [c['low'] for c in cells]
    low_p90 = float(np.percentile(low_vals, 90)) + 1e-9
    mid_vals = [c['mid'] for c in cells]
    mid_p90 = float(np.percentile(mid_vals, 90)) + 1e-9
    high_vals = [c['high'] for c in cells]
    high_p90 = float(np.percentile(high_vals, 90)) + 1e-9
    # 分频带起音：鼓(低频) / 旋律(中频) / 镲(高频)。用户说的「真正需要重音」
    # 多数是中频的旋律重音和低频的鼓点；只看宽带 flux 会把镲片当成重音。
    for c0 in cells:
        c0['band'] = max(c0['low'] / low_p90, c0['mid'] / mid_p90, c0['high'] / high_p90)
    band_p = np.array([c['band'] for c in cells])
    band_cut = float(np.percentile(band_p, 22))
    # 全曲最强的起音（前 12%）：这些是「真正需要重音」的地方，
    # 一律不能跳过、而且要砸和弦。之前只按「强拍/分频带」判断，
    # 最强的那批反而被跳过（实测覆盖 0.95 → 0.81）。
    ons_p88 = float(np.percentile([c['onset'] for c in cells], 88))
    ol_p90 = float(np.percentile([c['on_low'] for c in cells], 90))
    olib_p90 = float(np.percentile([c['on_lib'] for c in cells], 90))
    olib_p30 = float(np.percentile([c['on_lib'] for c in cells], 30))
    olib_p80 = float(np.percentile([c["on_lib"] for c in cells], 90))
    om_p90 = float(np.percentile([c['on_mid'] for c in cells], 90))
    oh_p90 = float(np.percentile([c['on_high'] for c in cells], 90))
    ons_p70 = float(np.percentile([c['onset'] for c in cells], 70))
    # 和弦/密度看的是「这一格有没有强起音」，不是电平（电平在持续段一直是高的）
    _ons = [c['onset'] for c in cells if c['onset'] > 0]
    ons_med = float(np.median(_ons)) + 1e-9 if _ons else 1.0
    row_size = fp['row_size']
    ln_ratio_target = args.ln_cap if args.ln_cap is not None else fp['ln_ratio']
    ln_len_dist = fp.get('ln_len_dist') or {'1': 1.0}

    state = {'prev_col': rnd.randrange(4), 'prev_cols': set(), 'recent': [],
             'load': {0: 0, 1: 0, 2: 0, 3: 0}}
    out_rows = []
    motif_mem = {}
    occ_counter = defaultdict(int)
    n_reuse = 0
    n_drop = [0]

    # 人类图案块库（4/6 个连续单音的列序列）：stream 的「流」是有形状的，
    # 逐格马尔科夫会变成随机游走（实测 60-66s 一段全是 3-1-0-3-1-3-0…）
    patterns = fp.get('patterns') or []
    def _is_ladder(g):
        """四列循环空间里的单调阶梯（0123 / 3210 / 3012 …）——stream 的骨架。"""
        if len(g) < 3:
            return False
        steps = [((g[i + 1] - g[i]) % 4) for i in range(len(g) - 1)]
        return all(st in (1, 3) for st in steps) and len(set(steps)) == 1
    _all = [(list(p['g']), float(p['w'])) for p in patterns
            if p.get('n') in (4, 6) and p.get('g')]
    _lad = [(g, w) for g, w in _all if _is_ladder(g)]
    # stream 档只走阶梯（干净上行/下行），tech/ln 用全部图案块
    pat_pool = _lad if (STYLE_CLASS[0] == 'stream' and _lad) else _all
    pat_total = sum(w for _, w in pat_pool) or 1.0
    print(f'图案块库: {len(pat_pool)} 个（阶梯 {len(_lad)}）')

    # 段落等价组 → 固定图案块子集：同一段音乐第二次出现时用同样的块库，
    # 不同段用不同块库。这是「动机呼应」的机器实现（规则引擎原来没有）。
    gid_pools = {}

    def pool_for(gid):
        if gid not in gid_pools:
            r2 = random.Random(1000 + (gid if gid is not None else 0))
            k = max(4, min(len(pat_pool), 6))
            gid_pools[gid] = r2.sample(pat_pool, k) if len(pat_pool) > k else list(pat_pool)
        return gid_pools[gid]

    def place_block(prev_col, max_len, gid=None):
        """取一个图案块并平移到当前手位（保持形状），返回列序列（不含 prev）。"""
        pool = pool_for(gid) if gid is not None else pat_pool
        tot = sum(w for _, w in pool) or 1.0
        if not pool:
            return []
        r = rnd.random() * tot
        cum = 0.0
        for g, w in pool:
            cum += w
            if r <= cum:
                # 关键：块的第一个音**不能**等于上一行的列（那就是「叠」）。
                # 取块的走向，从 prev_col 往那个方向走一格作为块的起点。
                d = 1 if g[-1] > g[0] else -1
                start = (prev_col + d) % 4
                off = (start - g[0]) % 4
                return [((c + off) % 4) for c in g][:max_len]
        g = pool[-1][0]
        d = 1 if g[-1] > g[0] else -1
        off = ((prev_col + d) % 4 - g[0]) % 4
        return [((c + off) % 4) for c in g][:max_len]

    def pick_size(dist):
        keys = sorted(int(k) for k in dist)
        tot = sum(dist[str(k)] for k in keys) or 1.0
        r = rnd.random()
        cum = 0.0
        for k in keys:
            cum += dist[str(k)] / tot
            if r <= cum:
                return k
        return keys[-1]

    def transform(cols, tf):
        if tf == 'id':
            return list(cols)
        if tf == 'mirror':
            return sorted(3 - c for c in cols)
        if isinstance(tf, str) and tf.startswith('shift_'):
            k = int(tf.split('_')[1])
            return sorted((c + k) % 4 for c in cols)
        if tf == 'shift':
            return sorted((c + 1) % 4 for c in cols)
        return sorted((c + 2) % 4 for c in cols)

    dbg = []          # (pitch_delta, column_delta, size) 诊断音高引导是否生效
    dbg_want, dbg_got = [0], [0]

    def advance(cols):
        for cc in cols:
            state['load'][cc] += 1
        state['prev_col'] = cols[-1]
        state['prev2_cols'] = state.get('prev_cols') or ()
        state['prev_cols'] = set(cols)
        state['recent'] = (state['recent'] + [cols[-1]])[-3:]

    def gen_piece(b0, b1, base_i, quiet, rps, intensity=0.5, gid=None):
        """生成 [b0,b1) 小节的行并配列；返回 [(dp, cols)]"""
        rows = []
        want = int(round(rps * CELL_PER_BAR * cell_ms / 1000.0))
        want = max(0, min(CELL_PER_BAR, want))
        for b in range(b0, b1):
            BARSTAT['bars'] += 1
            BARSTAT['want'] += want
            ps = bar_of.get(b)
            if not ps:
                continue
            if args.rhythm in ('anchor', 'audio', 'onset'):
                tp = None          # 节奏由音频决定，模板不参与选格
            else:
                tp = pick_template(rnd, ps, templates, want, quiet)
            bref3 = sum(c['accent'] for c in ps.values()) / max(1, len(ps))
            avail = {p: c for p, c in ps.items() if c['accent'] >= bref3 * 0.55}
            if not avail:
                continue
            # 持续音段（本小节音频 onset 很少）不许硬填：那里应该留长音或休息，
            # 而不是一路 1/16 —— 这是「空音」的主要来源（实测 228.5s 一带 17 行
            # 却几乎没有起音）。
            _n_on_bar = sum(1 for p2 in range(CELL_PER_BAR)
                            if (b * CELL_PER_BAR + p2) in onset_cells)
            if _n_on_bar <= 3:
                if max(2, _n_on_bar) < want:
                    BARSTAT['capped_by_occ'] += 1
                want = min(want, max(2, _n_on_bar))
            if _bar_env.get(b, 1.0) < 0.30 * _bar_env_med:
                if want > 2:
                    BARSTAT['capped_by_env'] += 1
                want = min(want, 2)      # 起音极弱的小节：只留骨架
            if args.strict:
                # 严格采音：这一小节只放在检测到 onset 的格子上，一个补音都不加
                occ = [p for p in range(CELL_PER_BAR)
                       if p in ps and ps[p]['i'] in onset_cells]
                if len(occ) > want:
                    occ = sorted(occ, key=lambda q: -ps[q]['onset'])[:want]
                chords = [p for p in occ if p in avail
                          and (avail[p]['on_low'] >= ol_p90 * 0.85 or STRONG_W[p] >= 2.6)]
            elif args.rhythm == 'gap':
                # 间隔模型走位：这一小节里模型给出的落点（不额外补音）
                occ = [p for p in range(CELL_PER_BAR)
                       if (b * CELL_PER_BAR + p) in gap_cells and p in ps]
                if not occ:
                    occ = sorted(ps, key=lambda p: -ps[p]['salience'])[:1]
                chords = [p for p in occ if p in avail
                          and (avail[p]['on_low'] >= ol_p90 * 0.9 or STRONG_W[p] >= 2.6)]
            elif args.rhythm == 'anchor':
                # —— 锚点 + 补音 ——
                # 锚点 = 音频检测到的 onset 吸附到 1/16 格（这些必须有音符，采音）。
                # 补音 = 在锚点之间按 style 的密度倍数补格子（人类的「大拍技」
                # 实测是音乐 onset 率的 1.8 倍：16.7 行/s vs 音频 9.3 onset/s）。
                if 'model_p' in next(iter(ps.values())):
                    # —— 模型主导选格：按「人类谱会不会在这一格放音符」排序 ——
                    # 音频 onset 命中给一点加成（保证重音不漏），但排序权在模型。
                    score_p = {}
                    for p2 in range(CELL_PER_BAR):
                        if p2 not in ps:
                            continue
                        bp = ps[p2]['model_p']
                        if (b * CELL_PER_BAR + p2) in onset_cells:
                            bp = min(1.0, bp * 1.15 + 0.05)
                        score_p[p2] = bp
                    top = sorted(score_p, key=lambda q: -score_p[q])
                    # 锚点必须先保住「音频 onset 命中」的格子（采音底线），
                    # 再用模型分补齐到本小节的目标条数。
                    onset_here = [q for q in range(CELL_PER_BAR)
                                  if (b * CELL_PER_BAR + q) in onset_cells and q in score_p]
                    need = max(len(onset_here), int(round(len(onset_here) + 1)))
                    anchors = list(onset_here)
                    for q in top:
                        if len(anchors) >= need:
                            break
                        if q not in anchors:
                            anchors.append(q)
                    anchors = sorted(anchors)
                else:
                    anchors = [p for p in range(CELL_PER_BAR)
                               if (b * CELL_PER_BAR + p) in onset_cells and p in ps
                               and ps[p]['accent'] >= acc_floor * 0.7]
                if not anchors:
                    anchors = sorted(avail, key=lambda p: -avail[p]['salience'])[:2]
                # 补音倍数也按段落强度变：强段铺密、弱段留白，段间行数 CV 才上得来
                target = int(round(len(anchors) * args.fill_mult
                                   * (0.50 + 1.00 * intensity)))
                target = max(len(anchors), min(want, target))
                occ = list(anchors)
                # 贪心补「离已有行最远」的格子，补到 target
                while len(occ) < target:
                    # 补音格：有模型就按「人类谱会不会在这格放音符」的概率选，
                    # 没有就退回「有起音 + 最大间隙」的规则。
                    if 'model_p' in next(iter(avail.values())):
                        pool = [p for p in avail if p not in occ]
                        if not pool:
                            break
                        best = max(pool, key=lambda p: avail[p]['model_p'])
                    else:
                        pool = [p for p in avail if p not in occ
                                and avail[p]['on_lib'] >= olib_p30]
                        if not pool:
                            pool = [p for p in avail if p not in occ]
                        if not pool:
                            break
                        best, bd = None, -1
                        for p in pool:
                            d = min(abs(p - q) for q in occ)
                            w2 = avail[p]['salience']
                            if d > bd or (d == bd and best is not None and w2 > avail[best]['salience']):
                                best, bd = p, d
                    if best is None:
                        break
                    occ.append(best)
                chords = [p for p in anchors if p in avail
                          and avail[p]['on_low'] >= ol_p90 * 0.85
                          or (STRONG_W[p] >= 2.0 and avail[p]['salience'] > 1.1 * float(np.median([c['salience'] for c in avail.values()])))]
            elif args.rhythm == 'onset':
                # —— 最直接的采音：把音频检测到的 onset 时刻吸附到 1/16 格 ——
                # 音乐里有什么时刻发声，谱就在那一格；间隔分布自然跟音乐一致
                # （这首歌 1/8 占 54%、1/16 占 35%），不会凭空写成满屏 16 分。
                occ = [p for p in range(CELL_PER_BAR)
                       if (b * CELL_PER_BAR + p) in onset_cells and p in avail]
                if not occ:
                    occ = sorted(avail, key=lambda p: -avail[p]['salience'])[:2]
                if len(occ) > want:
                    occ = sorted(occ, key=lambda p: (-(0 if STRONG_W[p] >= 2.0 else 1),
                                                     -avail[p]['salience']))[:want]
                chords = [p for p in occ
                          if avail[p]['on_low'] >= ol_p90 * 0.9 or STRONG_W[p] >= 2.6]
            elif args.rhythm == 'audio':
                # —— 采音优先：音符只落在「真的有击打」的格子上 ——
                # ev = 分频带起音（鼓/旋律/镲，各自按 p90 归一）取最大。
                # 阈值按本小节自适应：quiet 段落自然变稀，不再用全局密度目标硬填。
                ev = {p: max(c2['on_low'] / (ol_p90 + 1e-9),
                             c2['on_mid'] / (om_p90 + 1e-9),
                             c2['on_high'] / (oh_p90 + 1e-9),
                             1.25 * c2['on_lib'] / (olib_p90 + 1e-9))
                      for p, c2 in avail.items()}
                evs = sorted(ev.values())
                bar_med = evs[len(evs) // 2] if evs else 0.0
                # 本小节「有多少个真实击打」由音频自己决定：
                # 密集段（底鼓+军鼓+踩镲都有）自然多，持续音段自然少。
                hit_cells = [p for p, v in ev.items() if v >= max(args.ev_thr, 0.9 * bar_med)]
                n_hit = len(hit_cells)
                # 行数 = 本地击打数，只被 section 目标和上限约束
                target_n = min(want, max(3, n_hit + (1 if STRONG_W[0] else 0)))
                occ = sorted(hit_cells, key=lambda p: -ev[p])[:target_n]
                # 拍点保底（有轻微击打就补）
                for p, v in ev.items():
                    if STRONG_W[p] >= 2.0 and v >= 0.25 and p not in occ and len(occ) < want:
                        occ.append(p)
                # 和弦：低频/中频击打明显强于本小节中位 → 这才是「重音」
                chords = [p for p in occ
                          if max(avail[p]['on_low'] / (ol_p90 + 1e-9),
                                 avail[p]['on_mid'] / (om_p90 + 1e-9)) >= max(0.75, 1.6 * bar_med)]
            elif tp:
                ev = {p: max(c2['on_low'] / (ol_p90 + 1e-9),
                             c2['on_mid'] / (om_p90 + 1e-9),
                             c2['on_high'] / (oh_p90 + 1e-9),
                             1.25 * c2['on_lib'] / (olib_p90 + 1e-9))
                      for p, c2 in avail.items()}
                occ = [p for p in tp['occ'] if p in avail]
                # 「采音」硬约束：这一格完全没有起音事件（任何频带都没有）→ 删掉
                occ = [p for p in occ if ev.get(p, 0.0) >= 0.12]
                # 这一格有强事件但模板没选 → 补上（重音必须采）
                for p, v in ev.items():
                    if v >= 0.85 and p not in occ:
                        occ.append(p)
                chords = [p for p in tp.get('chord', ()) if p in avail] if isinstance(tp, dict) else []
                if len(occ) < want:
                    extra = sorted((p for p in avail if p not in occ),
                                   key=lambda p: -avail[p]['salience'] * (1.3 if STRONG_W[p] >= 2 else 1.0))
                    for p in extra[:max(0, want - len(occ))]:
                        occ.append(p)
            else:
                ranked = sorted(avail, key=lambda p: -(avail[p]['salience']
                                                       * (1.6 if STRONG_W[p] >= 2.0 else 1.0)))
                occ = ranked[:want]
                chords = [p for p in occ if STRONG_W[p] >= 2.0
                          and avail[p]['accent'] > acc_med * 1.15]
            # 乐句收尾 fill：段落最后一小节，末拍（格 12-15）铺满 16 分，
            # 并在格 12 上放一个和弦当「收句」。人类谱的段落边界几乎都有这种动作，
            # 没有 fill 的谱听起来就是一格一格往前推、没有呼吸。
            if not args.strict and b == b1 - 1 and (b1 - b0) > 1 and rnd.random() < 0.6:
                occ = sorted(set(occ) | {p2 for p2 in (12, 13, 14, 15) if p2 in avail})
                fill_set = {12, 13, 14, 15}
            else:
                fill_set = set()
            # 顶层重音强制入谱：模板没覆盖到的强起音格子要补上，
            # 否则「真正需要重音的地方」会空着（实测最强 2% 只覆盖 0.71）
            forced = {p2 for p2, c2 in ps.items()
                      if c2['onset'] >= ons_p88 and c2['accent'] >= acc_floor}
            if args.strict:
                occ = [p for p in occ if ps[p]['i'] in onset_cells]
            else:
                occ = sorted(set(occ) | forced)
            prev_chord = False
            for p in occ:
                c = ps.get(p)
                if c is None:
                    continue

                if c['accent'] < acc_floor:
                    continue          # 空音闸门
                ar = c['onset'] / ons_med
                strong = STRONG_W[p] >= 2.0
                # 人类只覆盖 ~72% 的重音，会主动放掉最弱的那批（留白/呼吸）。
                # 不强拍、分频带起音又最弱 → 跳过。
                top_accent = c['onset'] >= ons_p88
                # 强起音（前 30%）一律不许跳过——「真正需要重音」的地方不能空，
                # 之前用 p88 当门槛，导致最强的那批反而被跳过（覆盖只有 0.71）
                if (c['onset'] < ons_p70) and (not strong) and c['band'] < band_cut \
                        and rnd.random() < 0.55:
                    continue
                # 行大小：**按该桶人类的 row_size 分布抽**（1/2/3/4 押的真实比例），
                # chord_boost 只提高「这一行要不要出和弦」的概率，不改大小配比。
                # 之前用 boost 直接 +1 尺寸，结果 3 押占 43%、4 押 0.2%（人类 8%/4%）。
                # —— 行大小直接取自该分类包的人类分布 ——
                # V18 实测：tech 包 74% 单音 / 20% 双 / 5% 三 / 0.7% 四，
                # 而我们的产物是 46%/40%/11%/3.5% —— 和弦多一倍，手位移因此被压低。
                if PATTERN[0] is not None:
                    _dt = PATTERN[0][1]['types'].get(PATTERN[0][0], {})
                    _sz = _dt.get('sizes') or {}
                    if _sz:
                        _k = sorted(int(x) for x in _sz)
                        _w = np.array([float(_sz[str(x)]) for x in _k])
                        if strong or ar > 1.1:
                            _boost = {1: 1.0, 2: 1.45, 3: 1.7, 4: 1.5}
                            _w = _w * np.array([_boost.get(x, 1.0) for x in _k])
                        _w = _w * np.array([1.0 if x == 1 else args.size_boost for x in _k])
                        # npr 目标保持器：累计 npr 高于包目标就压和弦，低于就抬
                        # 配额控制（替代比例控制）：按「到当前行为止应当多出的音符数」决定
                        # 这一行要不要和弦。比例控制会让 npr 在目标上下摆动，配额控制不会。
                        _npr_t = (TARGETS[0] or {}).get('nPerRow')
                        _quota_chord = None
                        # 段落级摆动：按 8 小节一块决定这块偏单音还是偏和弦。
                        # 人类谱的单音是成段的（所以段内 jack 机会多），逐行均匀采样会把它们摊平。
                        _swing = getattr(args, 'section_swing', 0.0)
                        if _npr_t and _swing > 0:
                            _blk = int(float(c['time']) / (cell_ms * 16 * 8))
                            _sign = 1.0 if (hash((_blk, 7)) % 2 == 0) else -1.0
                            _npr_t = max(1.05, _npr_t + _sign * _swing)
                        if _npr_t and getattr(args, 'npr_mode', 'quota') == 'quota':
                            _st = state.setdefault('nprstat', {'notes': 0.0, 'rows': 0})
                            _want_excess = (_st['rows'] + 1) * (_npr_t - 1.0)
                            _have_excess = _st['notes'] - _st['rows']
                            _quota_chord = _have_excess < _want_excess - 0.25
                        elif _npr_t:
                            _st = state.setdefault('nprstat', {'notes': 0.0, 'rows': 0})
                            _cur = _st['notes'] / max(1, _st['rows'])
                            _k2 = float(np.clip((_npr_t / max(1e-6, _cur)) ** 2.0, 0.35, 3.0))
                            _w = _w * np.array([1.0 if x == 1 else _k2 for x in _k])
                        _w = _w / _w.sum()
                        # 和弦成簇（按簇发放配额）：人类 tech 包 12.6% 的相邻对是「和弦→和弦」，
                        # 且这些和弦共列（位移 0.71）。逐行判断会把簇打断，所以一次给 2-3 行。
                        _run = int(state.get('chord_run', 0))
                        if _run > 0 and _quota_chord is not None:
                            state['chord_run'] = _run - 1
                            _clust = True
                        else:
                            _clust = False
                        if _quota_chord is False and not _clust:
                            size = 1
                        elif _quota_chord is True or _clust:
                            # 簇内必须真的出和弦（V40 的 bug：这里落到 else 分支，
                            # 又按全分布采样，簇内一半变成单音，CC 自然不涨）
                            _k3 = [x for x in _k if x >= 2]
                            _w3 = [w for x, w in zip(_k, _w.tolist()) if x >= 2]
                            # 差目标还远时优先用 3-4 键：基座行 4 列全空，
                            # 不受「拆音只剩 1-2 空列」的限制（npr 1.55 封顶的主因）
                            if _npr_t and _k3:
                                _cn = _st['notes'] / max(1.0, _st['rows']) if _st['rows'] else 1.0
                                if _cn < _npr_t - 0.15:
                                    _w3 = [w * (3.0 if x >= 3 else 1.0) for x, w in zip(_k3, _w3)]
                            size = int(rnd.choices(_k3, weights=_w3)[0]) if _k3 else 1
                        else:
                            size = int(rnd.choices(_k, weights=_w.tolist())[0])
                        _st = state.setdefault('nprstat', {'notes': 0.0, 'rows': 0})
                        _st['rows'] += 1
                        _st['notes'] += size
                        state['prev_chord_row'] = size >= 2
                        state['cluster_chord'] = bool(size >= 2 and _clust)
                        if size >= 2 and state.get('chord_run', 0) == 0:
                            _cp = getattr(args, 'chord_cluster_p', 0.0)
                            if _cp > 0 and rnd.random() < _cp:
                                state['chord_run'] = rnd.choices([1, 2], [0.62, 0.38])[0]
                        rows.append({'cell': c, 'size': size, 'ar': ar, 'p': p,
                                     'bar': b, 'time': c['time']})
                        continue
                p1 = row_size.get('1', 0.63)
                p_chord = 1.0 - p1
                if strong or ar > 0.8:
                    p_chord *= (1.0 + 1.2 * args.chord_boost)
                # 段落强度调制：弱段（主歌/过渡）少和弦，强段（副歌）多和弦，
                # 让「同一套块库」在曲子里随情绪变厚变薄，而不是一条道铺到底
                p_chord *= (0.40 + 1.25 * intensity)
                # 和弦要压在「有鼓」的重音上（低频能量），不是宽带 flux 的杂音上
                p_chord *= (0.45 + 1.1 * min(1.0, c['low'] / low_p90))
                p_chord = min(0.95, p_chord)
                if p in chords and c['on_lib'] < olib_p80 and not strong:
                    size = 1        # 起音不够强的时刻不许给和弦
                elif p in chords and 'model_size' in c and STYLE_CLASS[0] in ('tech', 'ln'):
                    # 用人类谱训练的大小头决定和弦大小（2/3/4 押的比例）
                    ms = np.asarray(c['model_size'], dtype=np.float64)
                    if ms.sum() <= 0:
                        ms = np.array([0.5, 0.35, 0.1, 0.04, 0.01])
                    ms = ms / ms.sum()
                    ms[0] = 0.0
                    if ms.sum() <= 0:
                        size = 2
                    else:
                        size = int(rnd.choices(range(5), weights=(ms / ms.sum()).tolist())[0])
                        size = max(2, size)
                elif top_accent and STYLE_CLASS[0] in ('tech', 'ln'):
                    size = 2 if c['onset'] < ons_p88 * 1.6 else 3
                elif p in fill_set and p == 12:
                    # 收句和弦：即使是 stream 档也允许一个（收句要砸）
                    size = 2 if STYLE_CLASS[0] != 'tech' else 3
                elif STYLE_CLASS[0] == 'stream':
                    size = 1
                elif STYLE_CLASS[0] == 'rice':
                    size = 1
                    if strong and ar > 0.9 and rnd.random() < 0.12:
                        size = 2
                elif rnd.random() < p_chord:
                    # 在 2/3/4 里按人类比例重归一化
                    w2, w3, w4 = (row_size.get('2', 0.25), row_size.get('3', 0.08),
                                  row_size.get('4', 0.04))
                    tot = w2 + w3 + w4 or 1.0
                    r = rnd.random() * tot
                    size = 2 if r <= w2 else (3 if r <= w2 + w3 else 4)
                    # 和弦是强调：弱起音 + 弱拍上不许出
                    if not (ar > 0.5 or strong):
                        size = 1
                else:
                    size = 1
                # 流动钳制：上一行已经是和弦 → 这一行默认降成单音（除非是顶层重音）。
                # 人类 D15-15.7 的「和弦→和弦」只占 26%，我们原来是 41-47%，
                # 听感就是「一叠一叠地砸」，没有单音流把和弦串起来。
                if prev_chord and size >= 2 and not top_accent:
                    if rnd.random() < 0.42:
                        size = 1
                prev_chord = size >= 2
                rows.append({'cell': c, 'size': size, 'ar': ar, 'p': p, 'bar': b, 'time': c['time']})
        dbg_want[0] += want * (b1 - b0)
        dbg_got[0] += len(rows)
        if os.environ.get('PV11_BARDBG'):
            print(f'[bar] {b0}-{b1-1} want={want} rps={rps:.2f} quiet={quiet} '
                  f'rows={len(rows)} avail={len(avail)} tp={None if not tp else len(tp["occ"])}')
        rows.sort(key=lambda r: r['cell']['i'])
        piece = []
        prev_i = None
        prev_pitch = None
        block_buf = []          # 图案块剩余列（形状感的来源）
        for r in rows:
            c = r['cell']
            gap = 99 if prev_i is None else c['i'] - prev_i
            pd = c.get('ptrend')
            prev_c_now = state['prev_col']
            pthr = PITCH_THR[0]
            # —— 单音连流用图案块铺，而不是逐格独立采样 ——
            # 逐格马尔科夫出来的是随机游走（实测 3-1-0-3-1-3-0…），
            # 以「块」为单位接龙才有可辨认的形状（阶梯/折返）。
            if r['size'] == 1 and (prev_i is None or gap <= 2):
                if block_buf:
                    nxt = block_buf.pop(0)
                    if nxt == state['prev_col'] and gap <= 2:
                        # 块内相邻音撞上一行同列 → 挪到相邻空列（人耳就是「叠」）
                        free = [c for c in range(4) if c != state['prev_col']
                                and c not in (state.get('prev_cols') or set())]
                        nxt = free[0] if free else (nxt + 1) % 4
                    cols = [nxt]
                    advance(cols)
                    piece.append((c['i'] - base_i, tuple(cols)))
                    prev_i = c['i']
                    prev_pitch = c['pitch']
                    continue
                if pat_pool and PATTERN[0] is None and rnd.random() < 0.95:
                    seq = place_block(state['prev_col'], 12, gid)
                    if seq:
                        cols = [seq[0]]
                        block_buf = seq[1:]
                        advance(cols)
                        piece.append((c['i'] - base_i, tuple(cols)))
                        prev_i = c['i']
                        prev_pitch = c['pitch']
                        continue
            cols = sample_cols(rnd, state, trans_n, gap, pd, r['size'], fp, r['ar'],
                               pitch_w=(0.0 if args.no_pitch else 1.6),
                               thr=pthr, cell_ms=cell_ms)
            if pd is not None:
                dbg.append((pd, (sum(cols) / len(cols)) - prev_c_now))
            cols = sorted(set(cols))
            # 配额要求了 size 键，但采样器可能只返回更少的列（表里没有该尺寸的形状）
            # → 这里是 npr 从 1.79 掉到 1.53 的真正原因。补足到目标尺寸。
            if r['size'] >= 2 and len(cols) < int(r['size']):
                _extra = [cc for cc in range(4) if cc not in cols]
                rnd.shuffle(_extra)
                for _c2 in _extra:
                    if len(cols) >= int(r['size']):
                        break
                    cols = sorted(cols + [_c2])
            # 音高方向硬约束：单音行且乐句在走音高时，强制朝旋律方向移动
            # （软加权会被「禁卡列」和转移矩阵吃掉，实测产物相关只剩 0.02）
            if (not args.no_pitch) and PATTERN[0] is None and pd is not None and len(cols) == 1 \
                    and abs(pd) > pthr:
                want = 1 if pd > 0 else -1
                prev_c = state['prev_col']
                if (cols[0] - prev_c) * want <= 0:
                    free = [x for x in range(4)
                            if (x - prev_c) * want > 0 and x not in state['prev_cols']]
                    if free:
                        cols = [min(free, key=lambda x: abs(x - prev_c))]
            advance(cols)
            piece.append((c['i'] - base_i, tuple(cols)))
            prev_i = c['i']
            prev_pitch = c['pitch']
        return piece

    for k, s in enumerate(secs):
        base_i = s['s0'] * CELL_PER_BAR + phase
        gid = s['gid']
        occ = occ_counter[gid]
        occ_counter[gid] += 1
        reuse_p = 0.0 if occ == 0 else (0.70 if occ == 1 else (0.50 if occ == 2 else 0.30))
        # 注意：这里必须保留 `not args.strict`。
        # 第 96 轮试过在 strict 下打开「等价段复用」→ 复用的行不在目标段的 onset 格上，
        # 触发 strict onset invariant 断言；加了 ±2 格吸附后能跑通，但基座行数掉到 700
        # （正常 2000+），整体密度崩掉。动机复用需要改在「拆音之后」而不是基座层。
        if not args.strict and gid in motif_mem and rnd.random() < reuse_p:
            n_reuse += 1
            src_piece, src_pitch, src_mlen = motif_mem[gid]
            # 音高区域对齐：目标段与源段平均音高差 → 整体平移列位。
            # 随机 shift/rot 会把「音高↑→列右移」冲掉（实测产物相关 0.02），
            # 平移保留方向、只调区域，这才是人类「同段变奏」的做法。
            # 只有音高区域差得明显（>0.35 log2Hz ≈ 半个八度）才整体平移；
            # 否则原样复现——同段音乐用同一套键位才是「呼应」
            kk = int(round((s['feat'][4] - src_pitch) / 0.35))
            kk = max(-1, min(1, kk))
            tf = ('shift_%d' % kk) if kk != 0 else 'id'
            if rnd.random() < 0.12:
                tf = 'mirror'
            # 动机单元可能只有 1-2 小节，目标段有 4 小节 → 必须铺满，
            # 否则整段 3/4 是空的（实测空小节 12.4% vs 人类 2%，
            # 而且空在最响的副歌条上）
            nb_t = max(1, s['s1'] - s['s0'])
            span_src = max(1, src_mlen) * CELL_PER_BAR
            reps = (nb_t + max(1, src_mlen) - 1) // max(1, src_mlen)
            for rp in range(reps):
                for (dp, cols) in src_piece:
                    ci = base_i + dp + rp * span_src
                    if not (base_i <= ci < base_i + nb_t * CELL_PER_BAR):
                        continue
                    # 严格模式：复用的格子必须也是这一段的 onset 格，否则顺移 ±1 格找最近的
                    if args.strict and ci not in onset_cells:
                        _alt = None
                        for _d2 in (1, -1, 2, -2):
                            if (ci + _d2) in onset_cells and (ci + _d2) in cells_by_i:
                                _alt = ci + _d2
                                break
                        if _alt is None:
                            _n_drop = globals().get('_N_REUSE_DROP', 0) + 1
                            globals()['_N_REUSE_DROP'] = _n_drop
                            continue
                        ci = _alt
                    c = cells_by_i.get(ci)
                    if c is None:
                        continue
                    cols2 = transform(list(cols), tf)
                    advance(cols2)
                    out_rows.append({'i': c['i'], 't': c['time'],
                                     'ar': c['accent'] / acc_med, 'cols': cols2})
            continue
        nb = max(1, s['s1'] - s['s0'])
        mlen = rnd.choices([1, 2, 4], [0.12, 0.5, 0.38])[0] if nb > 2 else min(2, nb)
        mlen = nb if args.strict else max(1, min(mlen, nb))
        piece = gen_piece(s['s0'], s['s0'] + mlen, base_i, s['quiet'], s['rps'],
                          s.get('intensity', 0.5), gid)
        piece_pitch = s['feat'][4]
        if piece:
            motif_mem[gid] = (piece, s['feat'][4], mlen)
        rep = 0
        span = mlen * CELL_PER_BAR
        while rep * mlen < nb:
            rep_off = rep * span
            tf = 'id'
            if rep > 0 and rnd.random() < 0.35:
                kk = int(round((s['feat'][4] - piece_pitch) / 0.15))
                kk = max(-1, min(1, kk))
                tf = ('shift_%d' % kk) if kk != 0 else rnd.choices(
                    ['id', 'shift', 'rot'], [0.6, 0.25, 0.15])[0]
            for (dp, cols) in piece:
                ci = base_i + dp + rep_off
                c = cells_by_i.get(ci)
                if c is None:
                    continue
                # 只接受落在本段范围内的小节
                if not (s['s0'] * CELL_PER_BAR + phase <= ci
                        < s['s1'] * CELL_PER_BAR + phase):
                    continue
                cols2 = transform(list(cols), tf)
                advance(cols2)
                out_rows.append({'i': ci, 't': c['time'], 'ar': c['accent'] / acc_med,
                                 'cols': cols2})
            rep += 1

    if os.environ.get('PV11_DEBUG') and len(dbg) > 30:
        import numpy as _np
        dd = _np.array(dbg)
        c = _np.corrcoef(dd[:, 0], dd[:, 1])[0, 1]
        up = dd[dd[:, 0] > 0.1]; dn = dd[dd[:, 0] < -0.1]
        print(f'[dbg] 生成期音高/列相关 {c:.3f}  上行均位移 '
              f'{up[:,1].mean() if len(up) else 0:.2f} (n{len(up)})  下行均位移 '
              f'{dn[:,1].mean() if len(dn) else 0:.2f} (n{len(dn)})')
    if os.environ.get('PV11_DEBUG'):
        import collections as _c
        got = _c.Counter()
        for r in out_rows:
            b = (r['i'] - phase) // CELL_PER_BAR
            try:
                got[bar_sec[b]] += 1
            except KeyError:
                pass
        for k2, sec in enumerate(secs):
            n_bar = max(1, sec['s1'] - sec['s0'])
            print(f"[dbg] sec{k2:02d} e={sec['feat'][0]:.3f} 目标 {sec['rps']:.2f}/s "
                  f"实际 {got[k2] / (n_bar * CELL_PER_BAR * cell_ms / 1000):.2f}/s quiet={sec['quiet']}")
    print(f'[dbg] 目标行 {dbg_want[0]}  实际 {dbg_got[0]}  (产物行 {len(out_rows)})')
    print(f'行: {len(out_rows)}（动机复用 {n_reuse} 段，对音硬约束丢 {n_drop[0]} 行）')

    # ---- 小节级密度整形：让密度曲线真正贴住音乐（4 小节的段落目标太粗，
    #      实测 4s 窗密度/能量相关只有 0.21，人类 0.8+） ----
    bar_e = defaultdict(float)
    bar_n = defaultdict(int)
    for c in cells:
        if c['bar'] >= 0:
            bar_e[c['bar']] += c['accent']
            bar_n[c['bar']] += 1
    bar_e = {b: bar_e[b] / max(1, bar_n[b]) for b in bar_e}
    by_bar = defaultdict(list)
    for r in out_rows:
        by_bar[(r['i'] - phase) // CELL_PER_BAR].append(r)
    shaped = []
    for b, rs in by_bar.items():
        rs.sort(key=lambda r: r['i'])
        sec_e = sum(bar_e.get(x, 0.0) for x in range(b - 1, b + 2)) / 3.0 or 1e-9
        f = max(0.45, min(1.25, bar_e.get(b, sec_e) / sec_e))
        target = len(rs) if f > 0.92 else max(0, int(round(len(rs) * f)))
        if target <= 0:
            continue
        if target < len(rs):
            keep = [rs[0]]
            rest = sorted(rs[1:], key=lambda r: -r['ar'])
            keep += rest[:max(0, target - 1)]
            keep.sort(key=lambda r: r['i'])
            shaped += keep
        else:
            shaped += rs
    out_rows = shaped

    # ---- 强段最低密度保底 ----
    # 问题：某些小节被「空音闸门 + 模板未覆盖」双重清空，只剩 fill 的 4 个格子，
    # 结果在响亮的段落里出现整小节（1297ms）空洞 —— 听感就是「单调空虚」。
    # 这里对「能量 ≥ 全曲中位数 0.75 倍」的小节做保底：行数不足目标一半就补，
    # 补的格子按该小节 accent 从高到低取（必要时无视空音闸门，因为这里是响段）。
    # 保底行数也必须由音频决定：这一小节音乐里有几个 onset，就保到它的 ~0.8 倍
    _bar_sec = CELL_PER_BAR * cell_ms / 1000.0
    _t_base = cells[0]['time'] if cells else 0.0
    # 段落密度 → 每小节目标行数（之前直接用全局 args.dps，段落/难度差异全被抹平）
    _sec_rps = {}
    for _s in secs:
        for _b in range(_s['s0'], _s['s0'] + (_s.get('nb') or 4)):
            _sec_rps[_b] = float(_s.get('rps') or args.dps)
    def _bar_need_for(b):
        t0 = _t_base + b * CELL_PER_BAR * cell_ms
        t1 = t0 + CELL_PER_BAR * cell_ms
        if len(_onset_ms):
            _m = (_onset_ms >= t0) & (_onset_ms < t1)
            # 只用「够强」的起音计数：弱音密集段（镲片多、无重音）不应该被保底写密
            if ONSET_STR[0] is not None and args.onset_strength_floor > 0:
                _thr = float(np.percentile(ONSET_STR[0], args.onset_strength_floor))
                _m &= (ONSET_STR[0] >= _thr)
            n_on = int(np.sum(_m))
        else:
            n_on = 4
        _want = _sec_rps.get(b, args.dps)
        return max(3, min(int(round(_want * _bar_sec)), int(round(n_on * 0.85))))
    _bar_need = 4
    _be = {}
    for c0 in cells:
        if c0['bar'] >= 0:
            _be.setdefault(c0['bar'], []).append(c0['accent'])
    _be = {b: float(np.mean(v)) for b, v in _be.items()}
    _be_med = float(np.median(list(_be.values()))) if _be else 0.0
    _by_bar = defaultdict(list)
    for r in out_rows:
        _by_bar[(r['i'] - phase) // CELL_PER_BAR].append(r)
    _added = 0
    if args.strict:
        _be_med = float('inf')      # strict：保底全部失效
    # 注意：要遍历**所有有小节号的小节**，不能只遍历「已经有行的小节」——
    # 一整小节被清空时它不会出现在 _by_bar 里，正是空洞的主要来源
    for b in sorted(_be):
        rs = _by_bar.get(b, [])
        if _be.get(b, 0.0) < 0.75 * _be_med:
            continue                      # 真正安静的小节允许空
        _bn = _bar_need_for(b)
        if len(rs) >= _bn:
            continue
        have = {r['i'] for r in rs}
        bar_cells = [c for c in cells if c['bar'] == b]
        # 贪心「最大间隙优先」：每次补到离已有行最远的格子，专治小节内部的空洞
        need = max(0, _bn - len(rs))
        pool = [c for c in bar_cells if c['i'] not in have]
        while need > 0 and pool:
            best, bd = None, -1
            for c2 in pool:
                d = min(abs(c2['i'] - x) for x in have) if have else 99
                if d > bd or (d == bd and c2['accent'] > (best or c2)['accent']):
                    best, bd = c2, d
            if best is None:
                break
            loads = {k: sum(1 for r0 in out_rows if k in r0['cols']) for k in range(4)}
            # 保底行也不能制造「单音同列连打」：看上下 2 格内已用的列
            near = set()
            for r0 in out_rows:
                if abs(r0['i'] - best['i']) <= 2:
                    near |= set(r0['cols'])
            cand_cols = [k for k in sorted(range(4), key=lambda k: loads[k]) if k not in near]
            col = cand_cols[0] if cand_cols else min(range(4), key=lambda k: loads[k])
            out_rows.append({'i': best['i'], 't': best['time'],
                             'ar': best['accent'] / acc_med, 'cols': [col]})
            have.add(best['i'])
            pool.remove(best)
            need -= 1
            _added += 1
    if _added:
        print(f'  强段保底补了 {_added} 行')

    # ---- 1/32 补音：在相邻 1/16 行中间插半格 ----
    # 观景台的大拍技谱以 1/32 为主（Exoplanetary Mirage 16.7 行/s、间距中位 39ms），
    # 我们原来只在 1/16 格上写，密度和间隔特征都对不上。
    _env = None
    _env_fps = 0.0
    _sub32_thr = 0.0
    if args.sub32 > 0 or args.fill_spacing > 0:
        import librosa as _lr2
        _yy, _ss = _lr2.load(args.audio, sr=22050, mono=True)
        _env = np.maximum(_lr2.onset.onset_strength(y=_yy, sr=_ss, hop_length=256), 0)
        _env_fps = _ss / 256.0
        # 分段自适应门限：按 1s 时间桶算本地分位，避免「全局门限在密集段太松、
        # 在稀疏段太紧」。仍保留一个全局地板，防止在安静段凭空补音。
        _sub32_thr = float(np.percentile(_env, args.fill_thr_pct))
        _bin_local = {}
        _fps2 = _env_fps
        _nbin = int(len(_env) / _fps2) + 1
        _acc = {}
        for _i2, _v in enumerate(_env):
            _b2 = int(_i2 / _fps2)
            _acc.setdefault(_b2, []).append(float(_v))
        for _b2, _vals in _acc.items():
            if len(_vals) >= 8:
                _bin_local[_b2] = float(np.percentile(_vals, args.fill_thr_pct))
    # 严格模式仍然允许 1/32，但必须过音频门限（下方 env >= p35 的检查）
    if (args.sub32 > 0 or args.fill_spacing > 0) and len(out_rows) > 4:
        out_rows.sort(key=lambda r: (r['t'], r['i']))
        step_ms = cells[1]['time'] - cells[0]['time'] if len(cells) > 1 else 81.09
        # ---- 和弦槽（生成阶段的前置约束）----
        # 人谱和弦前后间隔 1.02 个 1/16 格；后置清空位会损失 24% 密度（第 82 轮证明）。
        # 改成先生成阶段就把「和弦要占的位置」圈出来，拆音时绕开它们。
        _npr0 = float((TARGETS[0] or {}).get('nPerRow') or 1.35)
        _n_slots = 0
        if args.chord_slot > 0 and len(out_rows) > 8:
            _n_slots = int(round(len(out_rows) * max(0.0, _npr0 - 1.0)))
            _gaps = []
            for _i3 in range(len(out_rows)):
                _gp = (out_rows[_i3]['t'] - out_rows[_i3 - 1]['t']) if _i3 > 0 else 9999.0
                _gn = (out_rows[_i3 + 1]['t'] - out_rows[_i3]['t']) if _i3 + 1 < len(out_rows) else 9999.0
                _gaps.append((min(_gp, _gn), _i3))
            _gaps.sort(key=lambda x: -x[0])
            for _g0, _i3 in _gaps[:_n_slots]:
                out_rows[_i3]['slot'] = True
            print(f'和弦槽：预选 {_n_slots} 个位置')
        new_rows = []
        # 当前密度是否低于目标：低于时每个空档尝试补 2 个点（而不是 1 个）
        _span0 = max(1.0, (out_rows[-1]['t'] - out_rows[0]['t']) / 1000.0)
        _cur_rps = len(out_rows) / _span0
        _tgt_rps = float((TARGETS[0] or {}).get('rowsPerSec') or args.dps or 10.0)
        _need_dense = _cur_rps < _tgt_rps * 0.92
        print(f'[诊断] 补音前密度 {_cur_rps:.2f} 行/s，目标 {_tgt_rps:.2f} → '
              f'{"双点补音" if _need_dense else "单点补音"}')
        prev_t = None
        prev_cols = set()
        _mv_sum, _mv_n = 0.0, 0
        _n_sum = float(sum(len(r0['cols']) for r0 in out_rows))
        _r_sum = len(out_rows)
        _sorted0 = sorted(out_rows, key=lambda r0: r0['t'])
        for _a, _b in zip(_sorted0, _sorted0[1:]):
            _mv_sum += float(np.mean([min(abs(c - q) for q in _a['cols']) for c in _b['cols']]))
            _mv_n += 1
        for idx, r in enumerate(out_rows):
            new_rows.append(r)
            if idx + 1 >= len(out_rows):
                continue
            nxt = out_rows[idx + 1]
            gap = nxt['t'] - r['t']
            # 和弦槽两侧留空：如果这一格的一端是和弦槽，只在空档 >=2 格时才补，
            # 且补在离槽 >=0.95 格的位置（否则和弦会被 1/32 挤住）
            _slot_guard = 0.0
            if r.get('slot') or nxt.get('slot'):
                _slot_guard = 0.95 * step_ms * 2.0
            # 不只填 1 格：2-3 格的空档同样按比例拆细（tech 包里 60% 是 <60ms 的间隔，
            # 我们 40% 的间隔是 110-190ms，全靠这一步拉回来）
            span = gap / step_ms
            if 0.75 <= span <= 4.4 and (args.fill_spacing > 0 or args.sub32 > 0):
                if args.fill_spacing > 0:
                    # 「拆到目标间距」：包里的 tech/speed 是 60-73% 的 <60ms 间隔，
                    # 也就是 1/32 密度；jack 包 66% 是 1/16，所以按包设不同间距
                    n_ins = max(0, int(round(span / args.fill_spacing)) - 1)
                    n_ins = min(n_ins, 7)
                else:
                    want = (span - 1.0) * args.sub32
                    if getattr(args, 'fill_max_per_gap', 0):
                        want = min(want, float(args.fill_max_per_gap))
                    n_ins = int(want) + (1 if rnd.random() < (want - int(want)) else 0)
                _last_col = r['cols'][0] if len(r['cols']) == 1 else None
                for k in range(n_ins):
                    if args.fill_prob < 1.0 and rnd.random() >= args.fill_prob:
                        continue
                    frac = (k + 1.0) / (n_ins + 1.0)
                    mid = r['t'] + gap * frac
                    # 关键：补音必须落在**独立起音**上（±30ms 内有一个起音）。
                    # 之前只看包络强度，而离起音 40ms 处包络仍然很高 →
                    # 补出来的音和相邻音共用同一个起音（实测 Tech 40%、Jack 93%），这就是「塞」。
                    _oa = ONSET_ALL[0]
                    if _oa is not None and len(_oa):
                        # 补音**吸附到空档里真实存在的起音**（而不是落在几何中点）：
                        # 既保住 1/32，又让补出来的音精确踩在声音上
                        _k = int(np.searchsorted(_oa, mid))
                        _cand = []
                        if _k < len(_oa):
                            _cand.append(float(_oa[_k]))
                        if _k > 0:
                            _cand.append(float(_oa[_k - 1]))
                        # 只在「离几何中点足够近」时吸附：否则补音偏到一侧会把
                        # 1/32 分布破坏掉（实测 0.52 → 0.38）
                        _cand = [c for c in _cand if r['t'] + 12 < c < r['t'] + gap - 12]
                        if not _cand:
                            REJ['no_onset'] = REJ.get('no_onset', 0) + 1
                            continue
                        # 网格吸附：把补音落在 1/32 网格上（人类谱的节奏混沌度 0.42-0.71，我们 0.84+）
                        _gs = float(getattr(args, 'grid_snap_ms', 0.0))
                        if _gs > 0:
                            try:
                                _st = float(an['stepMs']); _of = float(an['offsetMs'])
                            except Exception:
                                _st, _of = 0.0, 0.0
                            if _st > 0:
                                _k = round((mid - _of) / (_st / 2.0))
                                _sn = _of + _k * (_st / 2.0)
                                if abs(_sn - mid) <= _gs:
                                    mid = _sn
                        _tol = float(getattr(args, 'fill_snap_ms', 0.0))
                        if _tol > 0:
                            _near = [c for c in _cand if abs(c - mid) <= _tol]
                            if _near:
                                mid = min(_near, key=lambda c: abs(c - mid))
                        # _tol == 0：仍要求附近有独立起音，但音符放在中点（保持 1/32 分布）
                    if _env is not None:
                        _f = int(mid / 1000.0 * _env_fps)
                        if not (0 <= _f < len(_env)):
                            continue
                        _thr_here = _bin_local.get(int(_f / _env_fps), _sub32_thr)
                        if _env[_f] < max(_sub32_thr * 0.5, _thr_here):
                            REJ['weak_env'] = REJ.get('weak_env', 0) + 1
                            continue
                    if _slot_guard > 0:
                        _d_end = min(abs(mid - r['t']), abs(nxt['t'] - mid))
                        if _d_end < _slot_guard:
                            continue
                    used = set(r['cols']) | set(nxt['cols'])

                    # 补音之间也不能同列连打（否则 1/32 会变成一片 jack）
                    free = [k2 for k2 in range(4) if k2 not in used and k2 != _last_col]
                    if not free:
                        REJ['no_col'] = REJ.get('no_col', 0) + 1
                        continue
                    # 补音的列也要服从手位移目标：超了就贴最近列，欠了就挑最远列
                    _tm = (TARGETS[0] or {}).get('handMove')
                    if _tm is not None and _last_col is not None:
                        _run = _mv_sum / max(1, _mv_n)
                        _prev_c = r['cols'][0] if len(r['cols']) == 1 else _last_col
                        _next_c = nxt['cols'][0] if len(nxt['cols']) == 1 else _last_col
                        if _run > _tm:
                            # 超目标：同时贴近前后两行（只贴前一行会把下一段距离拉大）
                            _pick = min(free, key=lambda k2:
                                        abs(k2 - _prev_c) + abs(k2 - _next_c))
                        else:
                            # 需要更多位移时，原来直接挑「最远列」= 一律选外侧 0/3，
                            # 这正是单音列分布偏外的直接原因（实测拆音列 0.33/0.20/0.19/0.29）。
                            # 改成：在**能增加位移的候选**里按列偏好挑，位移需求交给和弦去补。
                            # 需要更多位移就挑最远列（第 78 轮回测：改成列均衡后
                            # 总位移从 1.18 掉到 1.06，列分布也没变好 → 保持原样）
                            _pick = max(free, key=lambda k2:
                                        abs(k2 - _prev_c) + abs(k2 - _next_c))
                        _mv_sum += abs(_pick - _last_col)
                        _mv_n += 1
                    else:
                        # 拆音也是单音：按人谱的单音列偏好加权。
                        # 单音只有约 42% 走 pattern 采样器，其余来自拆音/升级/补采，
                        # 不改这些路径的话列分布永远对不上。
                        # 小节级列配额：所有放置路径统一优先补本小节用得最少的列
                        _pick = pick_col_balanced(free, mid, prev_c=_last_col)
                    _last_col = _pick
                    _cols_new = [_pick]
                    _is_fill_row = True
                    SRC['fill'] = SRC.get('fill', 0) + 1
                    _h = SRC_COL.setdefault('fill', [0, 0, 0, 0])
                    _h[_pick] += 1
                    # 补音里也要有一部分双押，否则 npr 被单音稀释（tech 包 1.33，我们 1.22）
                    _rest = [k2 for k2 in free if k2 != _pick]
                    # npr 保持器（拆音端）：累计 npr 低于包目标就多出双押，高于就少出
                    _npr_t = (TARGETS[0] or {}).get('nPerRow')
                    if _npr_t:
                        # 拆音是 npr 被稀释的主因（首轮 npr 1.80，拆完掉到 1.53）：
                        # 低于目标就**一定**出和弦，高于就出单音（配额式，不是按比例）
                        _cur_npr = _n_sum / max(1.0, _r_sum)
                        _pc = 1.0 if _cur_npr < _npr_t else 0.0
                    else:
                        _pc = args.fill_chord_p
                    if _rest and rnd.random() < _pc:
                        _cols_new.append(_rest[int(rnd.random() * len(_rest))])
                    _fin = sorted(set(_cols_new))
                    _n_sum += len(_fin)
                    _r_sum += 1
                    new_rows.append({'i': r['i'], 't': mid,
                                     'ar': (r['ar'] + nxt['ar']) / 2,
                                     'cols': _fin, 'sub': True, 'is_fill': True})
        out_rows = new_rows

    # ---- 时间轴去抖：相邻行间隔 <45ms 视为同一拍点，丢弃后一行（防机枪） ----
    out_rows.sort(key=lambda r: (r['t'], r['i']))
    kept = []
    for r in out_rows:
        if kept and r['t'] - kept[-1]['t'] < 34:
            continue
        kept.append(r)
    out_rows = kept

    if args.strict:
        assert all(r['i'] in onset_cells for r in out_rows), 'strict onset invariant failed'
    if PATTERN[0] is not None:
        print('pattern hits', state.get('pattern_hits', 0), 'fallbacks', state.get('pattern_fallbacks', 0))

    # ---- LN 层：8s 窗激活 + 两段式（先长条、再短条）+ 释放留 1 格 ----
    ln_map = {}
    if ln_ratio_target > 0.02 and out_rows:
        keys = [int(x) for x in ln_len_dist]
        vals = [ln_len_dist[x] for x in ln_len_dist]
        sub = CELL_PER_BAR // 16            # 1/16 格 → 1/32 格
        cand = []
        for idx, r in enumerate(out_rows):
            c = cells_by_i[r['i']]
            if r['ar'] < 0.5:
                continue
            for col in r['cols']:
                cand.append((idx, r, col, c))
        # 时间序推进（乱序会让「列占用前沿」一步跳到曲尾，实测只放出 31 条）
        cand.sort(key=lambda x: x[3]['i'])
        # cap 要按「音符数」算：一个行只有一个列会变成 LN（不是整行），
        # 按行数算会严重低估（实测目标 16% 只出到 10.5%）
        n_notes_est = sum(len(r['cols']) for r in out_rows)
        cap = int(n_notes_est * min(ln_ratio_target, 0.175))
        col_until = {}
        next_same = {}
        # 关键：next_same 必须用**全部**行算，不能只用候选行——候选被
        # ar>=0.5 过滤过，漏掉低能量行会让 LN 盖住后面的音符（实测每张
        # 4-15 处「LN 体内有音符」）。
        by_col = {}
        for r0 in out_rows:
            for c0_ in r0['cols']:
                by_col.setdefault(c0_, []).append(r0['i'])
        for col, its in by_col.items():
            its.sort()
            next_same[col] = {its[k]: (its[k + 1] if k + 1 < len(its) else None)
                              for k in range(len(its))}
        for idx, r, col, c0 in cand:
            if len(ln_map) >= cap:
                break
            i = c0['i']
            if i < col_until.get(col, -99) + 1:
                continue
            nxt = next_same[col].get(i)
            maxL = (nxt - i - 1) if nxt is not None else 12
            if maxL < 2:
                continue
            L = int(rnd.choices(keys, weights=vals)[0]) * sub
            # 主动腾空间：若目标长度 > 当前可用空间，就把体内的同列音符删掉
            # （之前是「被空间压缩」，导致所有 LN 都被压到同样的长度区间）
            if args.ln_open_space > 0 and L > maxL and rnd.random() < args.ln_open_space:
                _del = [r0 for r0 in out_rows
                        if i < r0['i'] <= i + min(12, L) and col in r0['cols']]
                _del_ids = {id(x) for x in _del}
                out_rows[:] = [r0 for r0 in out_rows if id(r0) not in _del_ids]
                for r0 in _del:
                    if r0['i'] in by_col.get(col, []):
                        by_col[col].remove(r0['i'])
                _del2 = [r0 for r0 in out_rows
                         if i < r0['i'] <= i + min(12, L) and col in r0['cols']]
                maxL = min(12, L)
            L = max(2, min(12, L, maxL))
            seg = [cells_by_i[i + k]['accent'] for k in range(L + 1)
                   if (i + k) in cells_by_i]
            if seg and (sum(seg) / len(seg)) < 0.45 * c0['accent']:
                continue
            ln_map[i] = (col, L, float(r['t']))
            col_until[col] = i + L
    n_ln = len(ln_map)
    if out_rows and n_ln / len(out_rows) > 0.175:
        ln_map = {k: ln_map[k] for k in list(ln_map)[: int(len(out_rows) * 0.175)]}

    # LN 占用清理：1/32 补音可能落进 LN 体内（结构校验会报「LN 体内有音符」），
    # 这里把落在 [start, start+L] 区间内的同列音符删掉。
    _ln_span = {}
    for _ci, _ln in ln_map.items():
        _c, _L = _ln[0], _ln[1]
        _t0 = _ln[2] if len(_ln) > 2 else None
        if _t0 is None:
            _r0 = next((r for r in out_rows if r['i'] == _ci), None)
            _t0 = _r0['t'] if _r0 else 0.0
        _ln_span.setdefault(_c, []).append((_t0, _t0 + _L * cell_ms))
    # 不再在生成期删掉「LN 体内的音符」——那会把拆音吃掉（1/32 掉到 0.41）。
    # 改为输出期把 LN 收窄到下一个音符之前，两边都保住。

    # ---- 成串补音（run-based fill）----
    # 第 82/83 轮证明：只在「1 格空档中点」补单点的话，
    # 一旦要给和弦留空，密度就必然掉 27%。
    # 人谱的密度来自成串的 1/32 连音（2-3 个连在一起），所以按串补。
    if args.run_fill > 0 and out_rows:
        out_rows.sort(key=lambda r0: r0['t'])
        _chordt = [r0['t'] for r0 in out_rows if len(r0['cols']) >= 2]
        _guard = 0.55 * float(cell_ms) * 2.0
        _added = 0
        _out = []
        for _i4 in range(len(out_rows) - 1):
            _a, _b = out_rows[_i4], out_rows[_i4 + 1]
            _out.append(_a)
            _gap = _b['t'] - _a['t']
            if _gap < 1.9 * float(cell_ms) * 2.0 or _gap > 3.2 * float(cell_ms) * 2.0:
                continue
            # 空档内补 2 个，形成 1/32 连串
            for _fr in (0.4, 0.75):
                _t4 = _a['t'] + _gap * _fr
                if any(abs(_t4 - _ct) < _guard for _ct in _chordt):
                    continue
                if _env is not None and _env_fps > 0:
                    _f4 = int(_t4 / 1000.0 * _env_fps)
                    if not (0 <= _f4 < len(_env)) or _env[_f4] < _sub32_thr:
                        continue
                _c4 = [c for c in range(4) if c not in _a['cols'] and c not in _b['cols']]
                if not _c4:
                    continue
                _out.append({'i': int(round(_t4 / cell_ms)), 't': float(_t4),
                             'ar': (_a.get('ar', 1.0) + _b.get('ar', 1.0)) / 2.0,
                             'cols': [pick_col_balanced(_c4, _t4)], 'sub': True,
                             'is_fill': True})
                _added += 1
        if out_rows:
            _out.append(out_rows[-1])
        out_rows = _out
        if _added:
            print(f'成串补音：补了 {_added} 个音符')

    # ---- 小节级音符配额（替代逐行判断，解决 npr 在两端漂移） ----
    # 每个小节给定目标音符数 = 行数 × npr，不够就把单音升级成和弦，多了就降级。
    # 这比「逐行判断要不要和弦」稳定：局部配额不会累积误差。
    for _r0 in out_rows:
        for _c0 in _r0['cols']:
            COL_USE[_c0] += 1
            bar_col_add(_r0['t'], _c0)   # 小节级列配额要靠这个计数才能起作用
    _npr_goal = float((TARGETS[0] or {}).get('nPerRow') or 0)
    if _npr_goal > 1.0 and out_rows:
        out_rows.sort(key=lambda r0: r0['t'])
        _bar_ms = float(cell_ms) * CELL_PER_BAR
        _bars = {}
        for _i, _r in enumerate(out_rows):
            _bars.setdefault(int(_r['t'] // _bar_ms), []).append(_i)
        _up = _dn = 0
        # 配额按小节强度分配：响段和弦更厚、安静段更薄（人类谱的常态）。
        # 之前每小节用同一个 npr 目标，把强度差异抹平了（安静 1.27 / 响 1.40 太接近）。
        _bacc = {}
        for _b, _idxs in _bars.items():
            _accs = [float(cells_by_i.get(out_rows[_i]['i'], {}).get('accent', 0.0)) for _i in _idxs]
            _bacc[_b] = float(np.mean(_accs)) if _accs else 0.0
        _keys = list(_bacc)
        _vals = np.array([_bacc[k] for k in _keys])
        if len(_vals) > 1 and _vals.std() > 1e-9:
            _rank = np.argsort(np.argsort(_vals)) / max(1.0, len(_vals) - 1.0)
        else:
            _rank = np.full(len(_keys), 0.5)
        _factor = {k: 0.80 + 0.40 * r for k, r in zip(_keys, _rank)}
        _meanf = float(np.mean(list(_factor.values()))) or 1.0
        for _b, _idxs in _bars.items():
            # 和弦优先放在「周围宽松」的位置：人谱和弦前后间隔中位 82ms（1/16），
            # 我们原来只 55-64ms（塞进 1/32 密集处）。按行间距离降序排优先级，
            # 先给宽的位置升级，配额不够再依次往下补 —— 而不是放弃升级。
            _room = {}
            for _i2 in _idxs:
                _gp = (out_rows[_i2]['t'] - out_rows[_i2 - 1]['t']) if _i2 > 0 else 999.0
                _gn = (out_rows[_i2 + 1]['t'] - out_rows[_i2]['t']) if _i2 + 1 < len(out_rows) else 999.0
                _room[_i2] = min(_gp, _gn)
            # 先升级预选的「和弦槽」，再按宽松度补
            # 优先升级「两边都是单音」的行：人谱 tech 的 单→单 占 40%、和→和 只 14%，
            # 我们分别是 34%/16% —— 说明我们的和弦扎堆，人类的和弦是分散孤立的。
            _iso = {}
            for _q in _idxs:
                _l = (len(out_rows[_q - 1]['cols']) == 1) if _q > 0 else True
                _r = (len(out_rows[_q + 1]['cols']) == 1) if _q + 1 < len(out_rows) else True
                _iso[_q] = 0 if (_l and _r) else (1 if (_l or _r) else 2)
            _idxs = sorted(_idxs, key=lambda q: (0 if out_rows[q].get('slot') else 1,
                                                 _iso[q], -_room[q]))
            # 用 floor：round 会让每小节系统性地多给一点，导致全局 npr 偏高约 0.04
            _tgt_notes = int(np.floor(len(_idxs) * _npr_goal * _factor[_b] / _meanf))
            _cur = sum(len(out_rows[_i]['cols']) for _i in _idxs)
            _ptr = 0
            while _cur < _tgt_notes and _ptr < len(_idxs) * 2:
                _i = _idxs[_ptr % len(_idxs)]
                _r = out_rows[_i]
                # 允许一路补到 4 键（Jack 档 npr 2.33 需要三键/四键，补到 2 就停会卡在 2.18）
                if len(_r['cols']) < 4:
                    _near = set()
                    if _i > 0:
                        _near |= set(out_rows[_i - 1]['cols'])
                    if _i + 1 < len(out_rows):
                        _near |= set(out_rows[_i + 1]['cols'])
                    _free = [c for c in range(4) if c not in _r['cols'] and c not in _near]
                    # 同手优先：即使会与前后行共列也允许（jack 由后面的补齐/拆散机制管），
                    # 否则同手配对几乎拿不到（这是双押形状一直停在 35% 的原因）
                    if len(_r['cols']) == 1 and args.same_hand_target > 0:
                        _c0 = _r['cols'][0]
                        _same = 1 - _c0 if _c0 <= 1 else 5 - _c0
                        if _same not in _r['cols']:
                            _free = [_same] + [c for c in _free if c != _same]
                    if _free:
                        # 形状偏好：人谱的双押里「同一只手」（0-1 / 2-3）占 46.7%。
                        # （试过「升级时按位移优先挑最远列」：单↔和 位移只从 0.82 到 0.85，
                        #   和→和 反而从 0.35 涨到 0.48，总位移掉到 1.05 → 已回退）
                        # 原位加一列（人谱形状偏好：同手双押 46.7%）。
                        # 试过两种「升级时整体重挑形状 / 避开密集处」的改法：
                        # 单↔和 位移没改善（0.85），npr 掉到 1.29、同手率掉到 41% → 已回退。
                        _pick = None
                        # （试过「升级时要求和弦距相邻单音 >=1 列」：单→和 无变化 0.95→0.95，
                        #   因为和弦的基准列是采样器定的，加一列不影响最小距 → 已回退）
                        _free_use = _free
                        if len(_r['cols']) == 1:
                            _c0 = _r['cols'][0]
                            _same = 1 - _c0 if _c0 <= 1 else 5 - _c0   # 0↔1, 2↔3
                            if _same in _free_use:
                                _pick = _same
                        if _pick is None:
                            _pick = _free_use[int(rnd.random() * len(_free_use))]
                        _r['cols'] = sorted(_r['cols'] + [_pick])
                        COL_USE[_pick] += 1     # 每补一次就计数，均衡才有意义
                        _cur += 1
                        _up += 1
                        _ptr -= 1          # 同一行可以继续补，直到满足配额
                _ptr += 1
            _ptr = 0
            while _cur > _tgt_notes and _ptr < len(_idxs) * 2:
                _i = _idxs[_ptr % len(_idxs)]
                _r = out_rows[_i]
                if len(_r['cols']) >= 2:
                    _r['cols'] = _r['cols'][:-1]
                    _cur -= 1
                    _dn += 1
                _ptr += 1
        if _up or _dn:
            print(f'小节配额：升级 {_up} 处、降级 {_dn} 处（npr 目标 {_npr_goal:.2f}）')

    # ---- 双押形状对齐人谱边际 ----
    # 人谱 tech 的双押形状边际：0-1 23.6% / 2-3 23.1% / 0-3 20.0% / 1-2·1-3·0-2 各 11.1%
    # （同手双押合计 46.7%）。我们生成出来只有 31.8%，跨手的偏多 —— 观感就不是人写的。
    if args.chord_shape_fix > 0 and out_rows:
        _SHAPES = [((0, 1), 0.236), ((2, 3), 0.231), ((0, 3), 0.200),
                   ((1, 2), 0.111), ((1, 3), 0.111), ((0, 2), 0.111)]
        _keys = [k for k, _ in _SHAPES]; _ws = [w for _, w in _SHAPES]
        _fix = 0
        for _i, _r in enumerate(out_rows):
            if len(_r['cols']) != 2:
                continue
            if rnd.random() >= args.chord_shape_fix:
                continue
            _near = set()
            if _i > 0:
                _near |= set(out_rows[_i - 1]['cols'])
            if _i + 1 < len(out_rows):
                _near |= set(out_rows[_i + 1]['cols'])
            _pairs = list(zip(_keys, _ws))
            rnd.shuffle(_pairs)
            _new = None
            # 先试「完全不共列」，再退一步只允许共 1 列（jack 由后面的补齐/拆散机制管）
            for _k, _ in _pairs:
                if all(c not in _near for c in _k):
                    _new = _k; break
            if _new is None:
                for _k, _ in _pairs:
                    if sum(1 for c in _k if c in _near) <= 1:
                        _new = _k; break
            if _new:
                _r['cols'] = list(_new)
                _fix += 1
        if _fix:
            print(f'双押形状重排：{_fix} 行')

    # ---- 连续单音段重排（结构修法） ----
    # 起因：拆音占 54% 的行，而它只能在「前后行都未占用的列」里挑 → 被挤到外侧，
    # 整体列分布因此偏外（第 0 列 0.28 vs 人谱 0.24）。
    # 做法：不碰时间、密度、和弦，只把「连续单音段」的列走向按人谱的单音转移模型重排。
    if args.chain_rewrite > 0 and PATTERN[0] is not None and out_rows:
        _pt, _corp = PATTERN[0]
        _dt = _corp['types'][_pt]
        _tabs = _dt.get('trans') or {}
        _n_run = _n_rw = 0
        _rw_mv = 0.0
        _rw_n = 0
        _i = 0
        while _i < len(out_rows):
            if len(out_rows[_i]['cols']) != 1:
                _i += 1
                continue
            _j = _i
            while (_j + 1 < len(out_rows) and len(out_rows[_j + 1]['cols']) == 1
                   and out_rows[_j + 1]['t'] - out_rows[_j]['t'] <= 2.2 * cell_ms):
                _j += 1
            _run_len = _j - _i + 1
            if _run_len >= 3:
                _n_run += 1
                _prev = (out_rows[_i - 1]['cols'][0] if _i > 0 and out_rows[_i - 1]['cols'] else None)
                for _k in range(_i, _j + 1):
                    _r2 = out_rows[_k]
                    _g = 0.0 if _k + 1 >= len(out_rows) else (out_rows[_k + 1]['t'] - _r2['t'])
                    _b = ('32' if _g <= 60 else '16' if _g <= 110 else '8' if _g <= 190
                          else '4' if _g <= 330 else '2' if _g <= 650 else '1' if _g <= 1200 else 'slow')
                    _tab = _tabs.get(f'{sorted([_prev]) if _prev is not None else []}|{_b}') or {}
                    _cand = []
                    for _enc, _w in _tab.items():
                        _sh = json.loads(_enc)
                        if len(_sh) == 1 and _w > 0:
                            _cand.append((_sh[0], _w))
                    if not _cand:
                        _marg = (_dt.get('marg') or {}).get(_b) or {}
                        _cand = [(json.loads(k)[0], v) for k, v in _marg.items()
                                 if len(json.loads(k)) == 1 and v > 0]
                    if _cand:
                        _cs = [c for c, _ in _cand]
                        # 位移补偿：手位移低于目标时，偏向离上一行更远的列
                        _tmv = (TARGETS[0] or {}).get('handMove')
                        _need_move = False
                        if _tmv is not None and _prev is not None:
                            _rv = _rw_mv / max(1, _rw_n)
                            _need_move = _rv < _tmv * 0.98
                        _cs = [c for c, _ in _cand]
                        _ws = [w * (PREF_SINGLE[c] / 0.25)
                                 * (float(np.exp(0.35 * abs(c - _prev))) if (_need_move and _prev is not None) else 1.0)
                               for c, w in _cand]
                        _new = _cs[int(np.random.choice(len(_cs), p=np.array(_ws) / sum(_ws)))]
                        if _new != _r2['cols'][0]:
                            _r2['cols'] = [_new]
                            _n_rw += 1
                        if _prev is not None:
                            _rw_mv += abs(_new - _prev)
                            _rw_n += 1
                        _prev = _new
                    else:
                        _prev = _r2['cols'][0]
            _i = _j + 1
        if _n_rw:
            print(f'单音段重排：{_n_run} 段、改列 {_n_rw} 行')

    # ---- 和弦清出空位（跨歌一致性的修法） ----
    # 人谱和弦前后间隔中位 82ms；在密集歌里所有行都是 1/32 间隔，
    # 光靠「按宽松度排序」找不到空位（实测 Tech 档仍有 63-65ms）。
    # 做法：给每个和弦行清掉 ±45ms 内的补音（补音本来就是补出来的，删掉不损失采音）。
    if args.chord_room > 0 and out_rows:
        out_rows.sort(key=lambda r0: r0['t'])
        # 用「时间不在 1/16 网格上」判定补音（补音落在半格中点），
        # 不依赖 is_fill 标记（标记在合并时可能丢）
        def _is_offgrid(t):
            _q = t / float(cell_ms)
            return abs(_q - round(_q)) > 0.25
        _drop = set()
        for _i, _r in enumerate(out_rows):
            if len(_r['cols']) < 2:
                continue
            for _j in (_i - 1, _i + 1):
                if 0 <= _j < len(out_rows):
                    _o = out_rows[_j]
                    # 相邻单音只要比 0.75 格还近就删（不管是不是补音）——
                    # 实测 Flash/seatrus 的和弦间隔只有 0.50 格（人谱 1.02）
                    if (len(_o['cols']) == 1
                            and abs(_o['t'] - _r['t']) < 1.15 * float(cell_ms)):
                        _drop.add(_o['t'])
        if _drop:
            out_rows = [r0 for r0 in out_rows if r0['t'] not in _drop]
            print(f'和弦清空位：删掉 {len(_drop)} 个补音')
            # ---- 补位：把清空造成的密度损失补回来（避开和弦周围 0.6 格） ----
            out_rows.sort(key=lambda r0: r0['t'])
            _chord_t = [r0['t'] for r0 in out_rows if len(r0['cols']) >= 2]
            _guard = 0.6 * float(cell_ms) * 2.0
            _added = 0
            _newr = []
            for _i in range(len(out_rows) - 1):
                _a, _b = out_rows[_i], out_rows[_i + 1]
                _newr.append(_a)
                _gap = _b['t'] - _a['t']
                if _gap < 0.90 * float(cell_ms) * 2.0:
                    continue
                _mid = _a['t'] + _gap / 2.0
                if _env is not None and _env_fps > 0:
                    _f = int(_mid / 1000.0 * _env_fps)
                    if not (0 <= _f < len(_env)) or _env[_f] < _sub32_thr:
                        continue
                if any(abs(_mid - _ct) < _guard for _ct in _chord_t):
                    continue
                _cand = [c for c in range(4)
                         if c not in _a['cols'] and c not in _b['cols']]
                if not _cand:
                    continue
                _newr.append({'i': int(round(_mid / cell_ms)), 't': float(_mid),
                              'ar': (_a.get('ar', 1.0) + _b.get('ar', 1.0)) / 2.0,
                              'cols': [pick_col_balanced(_cand, _mid)], 'sub': True,
                              'is_fill': True})
                _added += 1
            if out_rows:
                _newr.append(out_rows[-1])
            out_rows = _newr
            print(f'补位：补回 {_added} 个音符')

    # ---- 动机重复（人谱的核心特征） ----
    # 实测：人谱 tech 有 31.9% 的 8 行片段是重复的（最高重复 33 次），jack 档 69.1%；
    # 我们只有 0.1% / 0.0% —— 几乎从不重复，听起来就是「随机塞」。
    # 做法：找出节奏骨架相同的两段，把后面那段的列照抄前面那段（可做整体镜像）。
    if args.motif_repeat > 0 and len(out_rows) > 32:
        out_rows.sort(key=lambda r0: r0['t'])
        _W = int(getattr(args, 'motif_len', 16))
        _sig = []
        for _i in range(1, len(out_rows)):
            _g = (out_rows[_i]['t'] - out_rows[_i - 1]['t']) / float(cell_ms)
            _sig.append(round(_g * 4) / 4)
        _groups = {}
        for _i in range(0, max(0, len(_sig) - _W)):
            _k = (tuple(_sig[_i:_i + _W]),
                  tuple(len(r0['cols']) for r0 in out_rows[_i:_i + _W]))
            _groups.setdefault(_k, []).append(_i)
        _cands = [v for v in _groups.values() if len(v) > 1]
        _n_rep = 0
        for _grp in _cands:
            for _j in _grp[1:]:
                if rnd.random() >= args.motif_repeat:
                    continue
                _src = _grp[0]
                _mirror = rnd.random() < float(getattr(args, 'motif_mirror', 0.25))
                for _q in range(_W):
                    _a0 = out_rows[_src + _q]
                    _b0 = out_rows[_j + _q]
                    if len(_a0['cols']) != len(_b0['cols']):
                        _feasible = False
                        break
                    if _mirror:
                        _cols0 = [3 - c for c in _a0['cols']]
                        if len(set(_cols0)) != len(_cols0):
                            continue
                    else:
                        _cols0 = list(_a0['cols'])
                    _b0['cols'] = sorted(_cols0)
                _n_rep += 1
        if _n_rep:
            print(f'动机重复：复用 {_n_rep} 段（候选 {len(_cands)} 组）')

    # ---- 段落级动机复用（整段替换，只平移时间） ----
    # 人谱 tech 23.3% 的 8 行片段重复（jack 69.1%），我们 0.1%：
    # 跨段复用原本写在 `if not args.strict` 下，而我们的谱全用 --strict → 从未生效。
    # 局部抄列试过（整段行数不等、只有部分能对上）→ 重复度没起来。
    # 这里改为：同一等价组，把后续段落的行**整段替换**为首次出现的行（时间平移过去）。
    if args.section_motif > 0 and secs and out_rows:
        out_rows.sort(key=lambda r0: r0['t'])
        _t_base = cells[0]['time'] if cells else 0.0
        _bar_ms = CELL_PER_BAR * float(cell_ms)

        def _span(s0, s1):
            return (_t_base + s0 * _bar_ms, _t_base + s1 * _bar_ms)

        _groups = {}
        for _s in secs:
            _groups.setdefault(_s['gid'], []).append(_s)
        # 放宽：把「音频特征相近」的段落也归为一组（Jack 这类等价组少的歌靠这个才有复用）
        if float(getattr(args, 'motif_cluster', 0.0)) > 0:
            _tol = float(args.motif_cluster)
            _keys = []
            for _g, _ss in _groups.items():
                _f = np.array(_ss[0]['feat'][:5], dtype=float)
                _merged = False
                for _k, (_rep, _lst) in enumerate(_keys):
                    if float(np.linalg.norm(_f - _rep)) <= _tol:
                        _lst.extend(_ss)
                        _merged = True
                        break
                if not _merged:
                    _keys.append((_f, list(_ss)))
            _groups = {i: v for i, (_f, v) in enumerate(_keys)}
            _groups = {k: sorted(v, key=lambda z: z['s0']) for k, v in _groups.items()}
        _n_sec = _n_row = 0
        for _g, _ss in _groups.items():
            if len(_ss) < 2:
                continue
            _a, _b = _span(_ss[0]['s0'], _ss[0]['s1'])
            _src = [r0 for r0 in out_rows if _a <= r0['t'] < _b]
            if len(_src) < 12:
                continue
            for _tg in _ss[1:]:
                if rnd.random() >= args.section_motif:
                    continue
                _c, _d = _span(_tg['s0'], _tg['s1'])
                if _d - _c < 1500.0:
                    continue
                _scale = max(0.5, min(2.0, (_d - _c) / max(1.0, _b - _a)))
                _mirror = rnd.random() < float(getattr(args, 'motif_mirror', 0.25))
                _new = []
                for _r0 in _src:
                    _t_new = _c + (_r0['t'] - _a) * _scale
                    if not (_c <= _t_new < _d):
                        continue
                    _cols0 = [3 - c for c in _r0['cols']] if _mirror else list(_r0['cols'])
                    if len(set(_cols0)) != len(_cols0):
                        _cols0 = list(_r0['cols'])
                    _new.append({'i': _r0['i'], 't': float(_t_new),
                                 'ar': _r0.get('ar', 1.0), 'cols': sorted(_cols0),
                                 'sub': True})
                if len(_new) < 12:
                    continue
                out_rows = [r0 for r0 in out_rows if not (_c <= r0['t'] < _d)]
                out_rows.extend(_new)
                out_rows.sort(key=lambda r0: r0['t'])
                _n_sec += 1
                _n_row += len(_new)
        if _n_sec:
            print(f'段落动机复用：{_n_sec} 段 / {_n_row} 行（整段替换）')

    # ---- 小节级动机重复（Jack 档的关键） ----
    # 人谱 jack 有 69.1% 的 8 行片段重复，这是「短动机反复」而不是整段复用。
    # 做法：按小节切分，行数相同的小节直接抄前面某小节的列。
    if args.bar_motif > 0 and out_rows:
        out_rows.sort(key=lambda r0: r0['t'])
        _bm = float(cell_ms) * CELL_PER_BAR
        _bars = {}
        for _r0 in out_rows:
            _bars.setdefault(int((_r0['t'] - (cells[0]['time'] if cells else 0)) // _bm), []).append(_r0)
        _keys = sorted(_bars)
        _pools = {}
        for _k in _keys:
            _pools.setdefault(len(_bars[_k]), []).append(_k)
        _n = 0
        for _k in _keys:
            _cand = [q for q in _pools.get(len(_bars[_k]), []) if q < _k]
            if not _cand or rnd.random() >= args.bar_motif:
                continue
            _src = _bars[_cand[int(rnd.random() * len(_cand))]]
            for _a0, _b0 in zip(_src, _bars[_k]):
                if _a0['t'] >= _b0['t']:
                    break
                # 注意：只在节奏骨架相同的两小节之间抄才有意义，
                # 否则窗口序列对不上，重复度反而下降（实测 Jack 16.4%→12.6%）
                _b0['cols'] = sorted(_a0['cols'])
                _n += 1
        if _n:
            print(f'小节动机重复：改写 {_n} 行')

    # ---- 循环式动机（实验：Jack 档专用） ----
    # 人谱 jack 69% 的重复来自「同一小节键型反复几十次」。
    # 做法：每 m 小节循环一次键型，并把音符时间**吸附到最近的起音格**，
    # 这样图案是循环的、采音仍尽量贴着音频（代价：可能偏 ±1 格）。
    if args.loop_bars > 0 and out_rows and cells:
        out_rows.sort(key=lambda r0: r0['t'])
        _bm = float(cell_ms) * CELL_PER_BAR
        _t0 = cells[0]['time']
        _bars = {}
        for _r0 in out_rows:
            _bars.setdefault(int((_r0['t'] - _t0) // _bm), []).append(_r0)
        _keys = sorted(_bars)
        _m = max(1, int(args.loop_bars))
        _grid = sorted(onset_cells)
        _moved = _dropped = 0
        for _idx, _k in enumerate(_keys):
            if _idx < _m:
                continue
            _src = _bars.get(_keys[_idx % _m])
            if not _src:
                continue
            _rows = _bars[_k]
            _n = min(len(_src), len(_rows))
            for _q in range(_n):
                _want_t = _t0 + _k * _bm + (_src[_q]['t'] - (_t0 + _keys[_idx % _m] * _bm))
                _ci = int(round((_want_t - _t0) / float(cell_ms)))
                _best = None
                for _d in (0, 1, -1):
                    if (_ci + _d) in onset_cells:
                        _best = _ci + _d
                        break
                if _best is None:
                    _dropped += 1
                    continue
                _rows[_q]['i'] = _best
                _rows[_q]['t'] = _t0 + _best * float(cell_ms)
                _rows[_q]['cols'] = sorted(_src[_q]['cols'])
                if _best != _ci:
                    _moved += 1
        seen = set()
        _ded = []
        for _k in _keys:
            for _r0 in _bars[_k]:
                _key = round(_r0['t'], 1)
                if _key in seen:
                    continue
                seen.add(_key)
                _ded.append(_r0)
        out_rows = sorted(_ded, key=lambda r0: r0['t'])
        print(f'循环式动机：{len(_keys)} 小节、吸附偏移 {_moved} 行、丢弃 {_dropped} 行')

    # ---- 单↔和 交界展开（镜像法） ----
    # 人谱 单→和 1.07 / 和→单 1.00，我们 0.81 / 0.90：和弦紧贴着相邻单音。
    # 镜像（0↔3, 1↔2）不改变形状类型（同手仍是同手、外列仍是外列），
    # 但能把和弦挪到另一侧，从而拉开与相邻单音的距离。
    if args.spread_mirror > 0 and out_rows:
        out_rows.sort(key=lambda r0: r0['t'])
        _n = 0
        for _i in range(len(out_rows)):
            _r0 = out_rows[_i]
            if len(_r0['cols']) < 2:
                continue
            _nb = []
            if _i > 0 and len(out_rows[_i - 1]['cols']) == 1:
                _nb.append(out_rows[_i - 1]['cols'][0])
            if _i + 1 < len(out_rows) and len(out_rows[_i + 1]['cols']) == 1:
                _nb.append(out_rows[_i + 1]['cols'][0])
            if not _nb:
                continue
            _d_now = min(abs(c - q) for c in _r0['cols'] for q in _nb)
            if _d_now > 1:
                continue
            _mir = [3 - c for c in _r0['cols']]
            if len(set(_mir)) != len(_mir):
                continue
            _d_new = min(abs(c - q) for c in _mir for q in _nb)
            if _d_new > _d_now:
                if rnd.random() < args.spread_mirror:
                    _r0['cols'] = sorted(_mir)
                    _n += 1
        if _n:
            print(f'单↔和 展开：镜像 {_n} 个和弦')

    # ---- 双手锚定组（和弦段） ----
    # 人谱 stamina 的相邻和弦几乎不动（逐音符 和→和 0.31），我们 0.71：
    # 我们每拍换手（左和弦→右和弦），人谱则是「双手一起锚定」。
    # 做法：对长度 >=3 的连续和弦段，按概率挑一个锚定列，让该段每个和弦都含它。
    if args.anchor_group > 0 and out_rows:
        out_rows.sort(key=lambda r0: r0['t'])
        _i = 0
        _n_run = _n_row = 0
        while _i < len(out_rows):
            if len(out_rows[_i]['cols']) < 2:
                _i += 1
                continue
            _j = _i
            while _j + 1 < len(out_rows) and len(out_rows[_j + 1]['cols']) >= 2:
                _j += 1
            if _j - _i + 1 >= 3 and rnd.random() < args.anchor_group:
                # 和弦组：锚定列固定，其余列在两侧来回跳
                # → 相邻和弦共享 1-2 列（和→和 低），但没有被按住的列照样移动（hand 不塌）
                _col = int(rnd.random() * 4)
                _sz = max(len(out_rows[_k]['cols']) for _k in range(_i, _j + 1))
                _ok = True
                _flip = 0
                for _k in range(_i, _j + 1):
                    _sh = list(out_rows[_k]['cols'])
                    _need = _sz - 1
                    _pool = [c for c in range(4) if c != _col]
                    _flip += 1
                    _pool.sort(key=lambda c: -abs(c - _col) if _flip % 2 == 0 else abs(c - _col))
                    _new = sorted([_col] + _pool[:_need])
                    if len(set(_new)) != len(_new) or len(_new) != len(_sh):
                        _ok = False
                        break
                    out_rows[_k]['cols'] = _new
                if _ok:
                    _n_run += 1
                    _n_row += (_j - _i + 1)
            _i = _j + 1
        if _n_run:
            print(f'双手锚定组：{_n_run} 段 / {_n_row} 行')

    # ---- 可弹性修复（用户硬约束：绝对不能卡手） ----
    # 实测 V51 每张谱有 18-25 处「同列 40ms 连打」（单指 25 次/秒，弹不了）
    # 和 130-220 处「单侧连续和弦」（同一只手 40ms 内两个双押）。
    if args.fix_playable:
        out_rows.sort(key=lambda r0: r0['t'])
        _fixed_jack = _fixed_side = 0
        for _i in range(1, len(out_rows)):
            _a, _b = out_rows[_i - 1], out_rows[_i]
            _dt = _b['t'] - _a['t']
            # 1) 同列极速连打
            if _dt < 55 and len(_a['cols']) == 1 and len(_b['cols']) == 1:
                if _a['cols'][0] == _b['cols'][0]:
                    _try = [c for c in range(4) if c not in _a['cols']]
                    _nxt = out_rows[_i + 1]['cols'] if _i + 1 < len(out_rows) else []
                    _try = [c for c in _try if c not in _nxt] or _try
                    if _try:
                        _b['cols'] = [_try[0]]
                        _fixed_jack += 1
            # 2) 单侧连续和弦
            # 单侧过载：同一只手在 55ms 内要按 >=3 个键（两只手指不可能完成）
            if _dt < 55 and len(_a['cols']) + len(_b['cols']) >= 3:
                _side_a = 0 if all(c <= 1 for c in _a['cols']) else (1 if all(c >= 2 for c in _a['cols']) else -1)
                _side_b = 0 if all(c <= 1 for c in _b['cols']) else (1 if all(c >= 2 for c in _b['cols']) else -1)
                if _side_a == _side_b and _side_a >= 0:
                    _mir = sorted(3 - c for c in _b['cols'])
                    _side_m = 0 if all(c <= 1 for c in _mir) else 1
                    if _side_m != _side_a:
                        _b['cols'] = _mir
                    else:
                        _b['cols'] = _b['cols'][:1]   # 实在不行就退成单音
                    _fixed_side += 1
        if _fixed_jack or _fixed_side:
            print(f'可弹性修复：同列连打 {_fixed_jack} 处、单侧连续和弦 {_fixed_side} 处')

    # ---- 强起音补采 ----
    # 实测：最强 10% 的起音有 ~21% 没有音符覆盖（用户最早反馈的「重音没采到」）。
    # 这里对未被覆盖的强起音，在 ±60ms 内最近的已有行上补一个空列；没有行就新建一行。
    if args.accent_cover > 0 and _env is not None and len(_onset_ms) > 0:
        _fps = float(_sr) / 256.0   # 该阶段 _env_fps 还没初始化（bug：会用成 0）
        # 用**稀释前**的全量起音来判断强音（否则强起音集合会被锚点稀释缩小十倍）
        _base = _onset_all if len(_onset_all) else _onset_ms
        _str = np.array([_env[min(len(_env) - 1, int(_t / 1000.0 * _fps))] for _t in _base])
        _thr = float(np.percentile(_str, 90))
        _strong = [_t for _t, _v in zip(_base, _str) if _v >= _thr]
        out_rows.sort(key=lambda r0: r0['t'])
        _times = np.array([r0['t'] for r0 in out_rows]) if out_rows else np.array([])
        _used = set((int(round(r0['t'])), c) for r0 in out_rows for c in r0['cols'])
        _added = 0
        for _st in _strong:
            if len(_times) and np.min(np.abs(_times - _st)) <= 40:
                continue
            if len(_times):
                _j = int(np.argmin(np.abs(_times - _st)))
                # 窗口从 60ms 放到 140ms：优先「给已有的行加一键」而不是「新建一行」。
                # 新建行会改变相邻间隔分布（实测 1/32 偏差从 0.022 退到 0.030）。
                # 窗口按档自适应：和弦密度越低（npr 越高），越该「贴到已有行」而不是新建行
                _npr_now = float((TARGETS[0] or {}).get('nPerRow') or 1.4)
                _win = 60.0 + 80.0 * float(np.clip((_npr_now - 1.4) / 0.9, 0.0, 1.0))
                if abs(_times[_j] - _st) <= _win:
                    _r = out_rows[_j]
                    _free = [c for c in range(4) if c not in _r['cols']]
                    # 补的列不能和前后行同列（否则补重音反而造出同列连打，Jack 档 0.113→0.140）
                    if len(free_bak := _free) > 1:
                        _near = set()
                        if _j > 0:
                            _near |= set(out_rows[_j - 1]['cols'])
                        if _j + 1 < len(out_rows):
                            _near |= set(out_rows[_j + 1]['cols'])
                        _ok = [c for c in _free if c not in _near]
                        if _ok:
                            _free = _ok
                    if _free:
                        _r['cols'] = sorted(_r['cols'] + [_free[int(rnd.random() * len(_free))]])
                        _added += 1
                    continue
            _free = [c for c in range(4) if (int(round(_st)), c) not in _used]
            if len(_free) > 1 and len(_times):
                _j2 = int(np.argmin(np.abs(_times - _st)))
                _ok2 = [c for c in _free if c not in out_rows[_j2]['cols']]
                if _ok2:
                    _free = _ok2
            if _free:
                out_rows.append({'i': int(round(_st / cell_ms)), 't': float(_st),
                                 'ar': 0.5, 'cols': [_free[int(rnd.random() * len(_free))]], 'sub': True})
                _used.add((int(round(_st)), _free[0]))
                _times = np.append(_times, float(_st))
                _added += 1
        if _added:
            out_rows.sort(key=lambda r0: r0['t'])
        print(f'强起音补采：覆盖缺口补了 {_added} 个音符（强起音 {len(_strong)} 个）')

    # ---- jack 目标补齐：低于包的实测 jack 率时，主动把相邻单音压到同列 ----
    # V20 实测 jack 档只有 0.023，而 jack 包是 0.113 —— 纠偏器只压不升，
    # 这里补一个「正向」机制：只在 1 格间隔的单→单上做，且不超过目标。
    if args.jack_target > 0 and len(out_rows) > 4:
        out_rows.sort(key=lambda r0: r0['t'])
        _ss = _jk = 0
        for _i in range(1, len(out_rows)):
            _a, _b = out_rows[_i - 1], out_rows[_i]
            if len(_a['cols']) != 1 or len(_b['cols']) != 1:
                continue
            if _b['t'] - _a['t'] < 81:   # 口径一致：>=1/16 都算（之前卡了 200ms 上限，和度量不一致）
                continue
            _ss += 1
            _is = (_b['cols'][0] == _a['cols'][0])
            _jk += int(_is)
            if not _is and _ss > 0 and _jk / _ss < args.jack_target:
                _b['cols'] = [_a['cols'][0]]
                _jk += 1
        # 反向：jack 高于目标时主动拆散（Stamina 档实测 0.090 vs 目标 0.049）
        _brk = 0
        if _ss and _jk / _ss > args.jack_target:
            for _i in range(1, len(out_rows)):
                if _jk / max(1, _ss) <= args.jack_target:
                    break
                _a, _b = out_rows[_i - 1], out_rows[_i]
                if len(_a['cols']) != 1 or len(_b['cols']) != 1:
                    continue
                if _b['t'] - _a['t'] < 81:
                    continue
                if _b['cols'][0] != _a['cols'][0]:
                    continue
                _nxt = out_rows[_i + 1]['cols'] if _i + 1 < len(out_rows) else []
                _alt = [c for c in range(4) if c not in _a['cols'] and c not in _nxt]
                if _alt:
                    _b['cols'] = [_alt[0]]
                    _jk -= 1
                    _brk += 1
        print(f'jack 补齐后：{_jk}/{_ss} = {_jk / max(1, _ss):.3f}（目标 {args.jack_target}，拆散 {_brk} 处）')

    # ---- 密度削薄（低难度的下界） ----
    # 「必须落在起音上」把行频下限锁在起音率附近，所以 ETT 12 只能到 14 音符/s。
    # 这里按目标行频主动丢弃**弱起音**行（保留 LN 起点与和弦行），让低难度真的低。
    _row_t = float(args.density_target or 0)
    if _row_t > 0 and out_rows:
        _span = max(1.0, (out_rows[-1]['t'] - out_rows[0]['t']) / 1000.0)
        _cur = len(out_rows) / _span
        if _cur > _row_t * 1.05:
            _ln_starts = set(ln_map.keys())
            _keep_n = int(round(_row_t * _span))
            _order = sorted(range(len(out_rows)), key=lambda i: (
                0 if (out_rows[i]['i'] in _ln_starts or len(out_rows[i]['cols']) >= 2) else 1,
                float(cells_by_i.get(out_rows[i]['i'], {}).get('accent', 0.0)),
                i))
            _drop = set(_order[_keep_n:])
            _before = len(out_rows)
            out_rows = [r for i, r in enumerate(out_rows) if i not in _drop]
            print(f'密度削薄：目标 {_row_t:.2f} 行/s，{_before} → {len(out_rows)} 行')
    _nst = state.get('nprstat')
    if _nst and _nst['rows']:
        print(f'[诊断] 配额累计 npr = {_nst["notes"] / _nst["rows"]:.3f}'
              f'（{_nst["rows"]:.0f} 行）')
    # ---- 网格固化 ----
    # 人类谱间隔几乎严格是 1/16 的整数倍（残差 0.01），我们 0.16（落在音频起音上）。
    # 固化后间隔词汇回到 1×/2×/0.5×，也更接近人类。
    if getattr(args, 'quantize', 0):
        _st = float(an['stepMs']); _of = float(an['offsetMs'])
        _unit = _st / (2.0 if args.quantize == 32 else 1.0)
        _q = {}
        for _r in out_rows:
            _k = int(round((_r['t'] - _of) / _unit))
            _qt = _of + _k * _unit
            _slot = _q.setdefault(round(_qt, 3), {'t': _qt, 'cols': set(), 'i': _r['i']})
            _slot['cols'] |= set(_r['cols'])
        _before = len(out_rows)
        out_rows = [{'t': v['t'], 'cols': sorted(v['cols']), 'i': v['i']} for _, v in sorted(_q.items())]
        print(f'网格固化 1/{args.quantize}：{_before} 行 → {len(out_rows)} 行')
    # ---- 输出 ----
    merged = {}
    cell_of_t = {}
    for r in out_rows:
        t = int(round(r['t']))
        merged.setdefault(t, set()).update(r['cols'])
        cell_of_t.setdefault(t, r['i'])
    # 最终防线：按「输出实际用的时间（取整后）」重建 LN 区间，把落在体内的音符剔掉。
    # 前面那次清理用的是生成期的浮点时间，取整后仍可能差 1ms。
    _i2t = {}
    for r in out_rows:
        _i2t.setdefault(r['i'], int(round(r['t'])))
    _ln_rt = {}
    for _ci, _ln in ln_map.items():
        _c, _L = _ln[0], _ln[1]
        _t0 = _i2t.get(_ci, _ln[2] if len(_ln) > 2 else 0)
        _ln_rt.setdefault(_c, []).append((_t0, int(_t0 + _L * cell_ms + 0.5)))

    _ln_last_end = {}
    lines = []
    for t in sorted(merged):
        ci = cell_of_t.get(t)
        item = ln_map.get(ci) if ci is not None else None
        for col in sorted(merged[t]):
            if any(_a < t < _b for (_a, _b) in _ln_rt.get(col, ())) and not (item and item[0] == col):
                continue
            x = int((col + 0.5) * 128)
            if item and item[0] == col:
                if t >= _ln_last_end.get(col, -1):
                    L = item[1]
                    _end = int(t + L * cell_ms + 0.5)
                    _ln_last_end[col] = _end
                    lines.append(f'{x},192,{t},128,0,{_end}:0:0:0:0:')
                # 与同列上一条 LN 重叠 → 整条丢弃（不能退化成 tap，那会落进体内）
            else:
                lines.append(f'{x},192,{t},1,0,0:0:0:0:')
    head = f'''osu file format v14

[General]
AudioFilename: {args.audio_name}
Mode: 3

[Metadata]
Title: {args.title}
Artist: {args.artist}
Creator: {args.creator}
Version: {args.version}
Tags: GJ algorithm rhythm-lexicon{('-pitch' if not args.no_pitch else '')} {args.style}

[Difficulty]
CircleSize: 4
OverallDifficulty: 8

[TimingPoints]
{off:.2f},{60000.0 / bpm},4,2,1,20,1,0

[HitObjects]
'''
    # LN 与拆音冲突的第二种解法：**收窄 LN**，而不是删掉音符
    # （V24 里 LN 档因为「删音符」导致 1/32 只有 0.41，包是 0.60）
    _tap_t = {}
    for _ln in lines:
        _f = _ln.split(',')
        if len(_f) >= 4 and _f[3] == '1':
            _tap_t.setdefault(_f[0], []).append(int(_f[2]))
    if _tap_t:
        _fixed = []
        for _ln in lines:
            _f = _ln.split(',')
            if len(_f) >= 6 and _f[3] == '128':
                _c, _t0 = _f[0], int(_f[2])
                _end = int(_f[5].split(':')[0])
                _nxt = [x for x in _tap_t.get(_c, ()) if x > _t0]
                if _nxt and _end >= min(_nxt):
                    _end = min(_nxt) - 1
                if _end - _t0 < 40:
                    continue          # 太短就整条丢掉，避免 1ms 长的假 LN
                _f[5] = f'{_end}:0:0:0:0:'
                _ln = ','.join(_f)
            _fixed.append(_ln)
        lines = _fixed
    # 输出前最后一道：按「真实写入的毫秒值」清掉落在同列 LN 体内的 tap。
    # （前面的清理用的是生成期时间，取整后仍可能差 1ms，实测 6 张里会漏 1 个）
    _ln_iv = []
    for _ln in lines:
        _f = _ln.split(',')
        if len(_f) >= 6 and _f[3] == '128':
            _ln_iv.append((_f[0], int(_f[2]), int(_f[5].split(':')[0])))
    if _ln_iv:
        _out = []
        for _ln in lines:
            _f = _ln.split(',')
            if len(_f) >= 4 and _f[3] == '1':
                _c, _t = _f[0], int(_f[2])
                if any(a[0] == _c and a[1] < _t < a[2] for a in _ln_iv):
                    continue
            _out.append(_ln)
        lines = _out
    open(args.o, 'w', encoding='utf-8').write(head + '\n'.join(lines) + '\n')
    span = (out_rows[-1]['t'] - out_rows[0]['t']) / 1000.0 if len(out_rows) > 1 else 1.0
    print('拆音拒绝原因:', dict(sorted(REJ.items(), key=lambda kv: -kv[1])))
    print('路径统计:', dict(sorted(SRC.items(), key=lambda kv: -kv[1])))
    for _k, _h in SRC_COL.items():
        _t = sum(_h) or 1
        print('  %-12s n=%4d  列 %s' % (_k, _t, [round(v / _t, 3) for v in _h]))
    print(f'写出 {args.o}: 行 {len(merged)}  音符 {len(lines)}  LN {len(ln_map)} '
          f'({len(ln_map) / max(1, len(merged)) * 100:.1f}%)  行频 {len(merged) / span:.2f}/s  '
          f'音符/行 {len(lines) / max(1, len(merged)):.2f}')
    if args.structure_out:
        json.dump({'phase': phase, 'seg_cells': CELL_PER_BAR * 4,
                   'sections': [{'s0': s['s0'], 's1': s['s1'], 'gid': s['gid'],
                                 'rps': round(s['rps'], 2)} for s in secs]},
                  open(args.structure_out, 'w'))


def _run_batch(path):
    """批量模式：一个进程里连续生成多张谱，省掉每次 ~13s 的导入开销。"""
    import json as _json
    jobs = _json.load(open(path, encoding='utf-8'))
    t_all = __import__('time').time()
    for k, job in enumerate(jobs):
        argv = []
        for kk, vv in job.items():
            if vv is True:
                argv.append(kk)
            elif vv is not False and vv is not None:
                argv += [kk, str(vv)]
        sys.argv = ['paint-gj.py'] + argv
        t0 = __import__('time').time()
        try:
            main()
            print(f'[{k + 1}/{len(jobs)}] {job.get("-o", "")} OK  {__import__("time").time() - t0:.1f}s', flush=True)
        except SystemExit:
            print(f'[{k + 1}/{len(jobs)}] {job.get("-o", "")} EXIT  {__import__("time").time() - t0:.1f}s', flush=True)
    print(f'批量完成 {len(jobs)} 张，总耗时 {__import__("time").time() - t_all:.1f}s', flush=True)


if __name__ == '__main__':
    if '--batch' in sys.argv:
        _run_batch(sys.argv[sys.argv.index('--batch') + 1])
    else:
        main()
