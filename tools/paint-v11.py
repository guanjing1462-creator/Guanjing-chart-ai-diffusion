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
PITCH_THR = [0.03]          # 旋律方向门控阈值（按本曲 |ptrend| p55 自适应）
CHORD_PAIRS_SAME = [(0, 1), (2, 3)]
CHORD_PAIRS_ALT = [(0, 2), (1, 3), (0, 3), (1, 2)]
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
    out = {'pitch': pitch.astype(np.float32), 'low': low.astype(np.float32),
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


def section_targets(secs, cells, phase, src_rps, dyn=0.45):
    """段能量 → 段目标行频（相对 dps 的调制，钳 ±dyn）。"""
    e = np.array([s['feat'][0] for s in secs])
    lo, hi = np.percentile(e, 15), np.percentile(e, 90)
    lo = min(lo, hi - 1e-9)
    w = (e - lo) / (hi - lo)
    w = np.clip(w, -0.2, 1.4)
    rel = 1.0 + dyn * (w - w.mean()) / (w.std() + 1e-9) * 0.5
    rel = np.clip(rel, 1 - dyn, 1 + dyn)
    for k, s in enumerate(secs):
        s['rps'] = src_rps * rel[k]
        s['quiet'] = float(w[k]) < 0.12
        s['intensity'] = float(min(1.0, max(0.0, w[k])))   # 0=最弱段 1=最强段
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
                pitch_w=1.0, thr=0.06):
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
        if JACK_MODE[0] == 'forbid':
            forbid = prev_cols
        elif chord_size <= 1 and len(prev_cols) >= 2:
            # 和弦之后的那一下：不能复用和弦已经按下的列。人类 D13-14 的
            # 和弦→单音同列只有 0.025，我们原来是 0.166（6.6 倍），听感是
            # 「手指黏在同一个键上」
            forbid = prev_cols
        elif chord_size >= 2:
            # 和弦不要复用上一行的列：人类在 D13-14 档 rowJack 只有 0.07，
            # 单音连打是音乐性的 jack，和弦踩同列才是「卡住」的来源
            forbid = set(state.get('prev_cols') or ())
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
            s *= (1 - fp['hand_alt']) if pair in CHORD_PAIRS_SAME else fp['hand_alt']
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
    ap.add_argument('--dps', type=float, required=True, help='目标行频（行/s）')
    ap.add_argument('--ln-cap', type=float, default=None)
    ap.add_argument('--dyn', type=float, default=0.45, help='段间密度起伏幅度')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--version', required=True)
    ap.add_argument('--title', default='INTERNET YAMERO')
    ap.add_argument('--artist', default='KOTOKO')
    ap.add_argument('--creator', default='mug-prototype')
    ap.add_argument('--audio-name', default='INTERNET YAMERO.ogg')
    ap.add_argument('--structure-out', default=None)
    ap.add_argument('--no-pitch', action='store_true', help='关闭音高引导列位（消融用）')
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
    secs = section_targets(secs, cells, phase, args.dps, args.dyn)
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

    def place_block(prev_col, max_len):
        """取一个图案块并平移到当前手位（保持形状），返回列序列（不含 prev）。"""
        if not pat_pool:
            return []
        r = rnd.random() * pat_total
        cum = 0.0
        for g, w in pat_pool:
            cum += w
            if r <= cum:
                off = (prev_col - g[0]) % 4
                return [((c + off) % 4) for c in g][:max_len]
        g = pat_pool[-1][0]
        off = (prev_col - g[0]) % 4
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
        state['prev_cols'] = set(cols)
        state['recent'] = (state['recent'] + [cols[-1]])[-3:]

    def gen_piece(b0, b1, base_i, quiet, rps, intensity=0.5):
        """生成 [b0,b1) 小节的行并配列；返回 [(dp, cols)]"""
        rows = []
        want = int(round(rps * CELL_PER_BAR * cell_ms / 1000.0))
        want = max(0, min(CELL_PER_BAR, want))
        for b in range(b0, b1):
            ps = bar_of.get(b)
            if not ps:
                continue
            tp = pick_template(rnd, ps, templates, want, quiet)
            bref3 = sum(c['accent'] for c in ps.values()) / max(1, len(ps))
            avail = {p: c for p, c in ps.items() if c['accent'] >= bref3 * 0.55}
            if not avail:
                continue
            if tp:
                occ = [p for p in tp['occ'] if p in avail]
                chords = [p for p in tp.get('chord', ()) if p in avail] if isinstance(tp, dict) else []
                if len(occ) < want:
                    # 兜底：模板覆盖不足 → 按能量补 onset 格（仍只在真实 onset 上）
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
            if b == b1 - 1 and (b1 - b0) > 1 and rnd.random() < 0.6:
                occ = sorted(set(occ) | {p2 for p2 in (12, 13, 14, 15) if p2 in avail})
                fill_set = {12, 13, 14, 15}
            else:
                fill_set = set()
            occ = sorted(occ)
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
                if (not top_accent) and (not strong) and c['band'] < band_cut \
                        and rnd.random() < 0.55:
                    continue
                # 行大小：**按该桶人类的 row_size 分布抽**（1/2/3/4 押的真实比例），
                # chord_boost 只提高「这一行要不要出和弦」的概率，不改大小配比。
                # 之前用 boost 直接 +1 尺寸，结果 3 押占 43%、4 押 0.2%（人类 8%/4%）。
                p1 = row_size.get('1', 0.63)
                p_chord = 1.0 - p1
                if strong or ar > 0.8:
                    p_chord *= (1.0 + 1.2 * args.chord_boost)
                # 段落强度调制：弱段（主歌/过渡）少和弦，强段（副歌）多和弦，
                # 让「同一套块库」在曲子里随情绪变厚变薄，而不是一条道铺到底
                p_chord *= (0.55 + 0.9 * intensity)
                # 和弦要压在「有鼓」的重音上（低频能量），不是宽带 flux 的杂音上
                p_chord *= (0.45 + 1.1 * min(1.0, c['low'] / low_p90))
                p_chord = min(0.95, p_chord)
                if top_accent and STYLE_CLASS[0] in ('tech', 'ln'):
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
                    cols = [block_buf.pop(0)]
                    advance(cols)
                    piece.append((c['i'] - base_i, tuple(cols)))
                    prev_i = c['i']
                    prev_pitch = c['pitch']
                    continue
                if pat_pool and rnd.random() < 0.95:
                    seq = place_block(state['prev_col'], 12)
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
                               thr=pthr)
            if pd is not None:
                dbg.append((pd, (sum(cols) / len(cols)) - prev_c_now))
            cols = sorted(set(cols))
            # 音高方向硬约束：单音行且乐句在走音高时，强制朝旋律方向移动
            # （软加权会被「禁卡列」和转移矩阵吃掉，实测产物相关只剩 0.02）
            if (not args.no_pitch) and pd is not None and len(cols) == 1 \
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
        if gid in motif_mem and rnd.random() < reuse_p:
            n_reuse += 1
            src_piece, src_pitch, src_mlen = motif_mem[gid]
            # 音高区域对齐：目标段与源段平均音高差 → 整体平移列位。
            # 随机 shift/rot 会把「音高↑→列右移」冲掉（实测产物相关 0.02），
            # 平移保留方向、只调区域，这才是人类「同段变奏」的做法。
            kk = int(round((s['feat'][4] - src_pitch) / 0.15))
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
        mlen = max(1, min(mlen, nb))
        piece = gen_piece(s['s0'], s['s0'] + mlen, base_i, s['quiet'], s['rps'],
                          s.get('intensity', 0.5))
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

    # ---- 时间轴去抖：相邻行间隔 <45ms 视为同一拍点，丢弃后一行（防机枪） ----
    out_rows.sort(key=lambda r: (r['t'], r['i']))
    kept = []
    for r in out_rows:
        if kept and r['t'] - kept[-1]['t'] < 34:
            continue
        kept.append(r)
    out_rows = kept

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
            L = max(2, min(12, L, maxL))
            seg = [cells_by_i[i + k]['accent'] for k in range(L + 1)
                   if (i + k) in cells_by_i]
            if seg and (sum(seg) / len(seg)) < 0.45 * c0['accent']:
                continue
            ln_map[i] = (col, L)
            col_until[col] = i + L
    n_ln = len(ln_map)
    if out_rows and n_ln / len(out_rows) > 0.175:
        ln_map = {k: ln_map[k] for k in list(ln_map)[: int(len(out_rows) * 0.175)]}

    # ---- 输出 ----
    merged = {}
    cell_of_t = {}
    for r in out_rows:
        t = int(round(r['t']))
        merged.setdefault(t, set()).update(r['cols'])
        cell_of_t.setdefault(t, r['i'])
    lines = []
    for t in sorted(merged):
        ci = cell_of_t.get(t)
        item = ln_map.get(ci) if ci is not None else None
        for col in sorted(merged[t]):
            x = int((col + 0.5) * 128)
            if item and item[0] == col:
                L = item[1]
                lines.append(f'{x},192,{t},128,0,{int(t + L * cell_ms + 0.5)}:0:0:0:0:')
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
Tags: ai v11 rhythm-lexicon{('-pitch' if not args.no_pitch else '')} {args.style}

[Difficulty]
CircleSize: 4
OverallDifficulty: 8

[TimingPoints]
{off:.2f},{60000.0 / bpm},4,2,1,20,1,0

[HitObjects]
'''
    open(args.o, 'w', encoding='utf-8').write(head + '\n'.join(lines) + '\n')
    span = (out_rows[-1]['t'] - out_rows[0]['t']) / 1000.0 if len(out_rows) > 1 else 1.0
    print(f'写出 {args.o}: 行 {len(merged)}  音符 {len(lines)}  LN {len(ln_map)} '
          f'({len(ln_map) / max(1, len(merged)) * 100:.1f}%)  行频 {len(merged) / span:.2f}/s  '
          f'音符/行 {len(lines) / max(1, len(merged)):.2f}')
    if args.structure_out:
        json.dump({'phase': phase, 'seg_cells': CELL_PER_BAR * 4,
                   'sections': [{'s0': s['s0'], 's1': s['s1'], 'gid': s['gid'],
                                 'rps': round(s['rps'], 2)} for s in secs]},
                  open(args.structure_out, 'w'))


if __name__ == '__main__':
    main()
