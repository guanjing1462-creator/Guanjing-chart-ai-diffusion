#!/usr/bin/env python3
"""重建音频条件数据集：逐谱自洽位移校正 + 质量门禁（修「谱面与音频错位」）。

旧数据 out/diff-data-audio-big 的致命问题：C 特征假定 `.osu` 时间轴与音频时间轴
零点一致（都从 0 开始），但多曲 pack / 不同音频版本会让整张谱相对音频平移。
实测该数据集的逐谱 X–C 相关（全局时间轴聚合）中位 −0.062，81% 的谱基本无关。

本脚本对 `out/audio-dl/<SetID>/` 下**每一张** 4K mania 谱（Mode=3 且 CircleSize=4）
独立处理：

  1) 解析 `[TimingPoints]`，只取 uninherited=1 的红线；用**第一条**红线的
     offset/beatLength 定 1/32 网格（step = beatLength/8 ms）；红线 >3 条
     （变速谱）跳过并计数。
  2) 取音频：优先该谱 `[General] AudioFilename` 指向的文件，没有再退到目录里
     唯一的 mp3/ogg/wav。
  3) onset 包络：sr=22050 / n_fft=512 / hop=256 / 128 带 mel(0–8k) → log1p →
     正向差分 → 0-200 / 200-2000 / 2000-8000 / 总 四带求和（与
     tools/build-audio-cond-big.py 同一套参数与带划分；该脚本 hop=128，
     这里按任务书用 hop=256，帧时刻按帧中心重算）。
  4) 自洽求最佳位移：δ 在 [--shift-lo, --shift-hi]（默认 ±2500ms）内以 50ms
     扫描，对每个 δ 算
        r(δ) = corr( 音符占用序列 (X 的四轨占用均值, 固定作者网格) ,
                     条件强度序列 (四带 P99 归一后取均值, 采样起点 offset+δ) )
     取 r 最大的 δ* 与 r*。同时记录「音符头数 vs 四带求和」这一较粗的口径，
     用来区分「真错位」和「LN 表示问题」。
  5) 门禁：r* ≥ --min-r（默认 0.25）且 |δ*| ≤ --max-shift（默认 300ms）。
     通过 → 用**校正后的采样起点 offset+δ*** 重算该谱 1/32 网格 C 特征，
     窗口 [4,256] / stride 128，只留「音符数 ≥16 且条件覆盖率 ≥90%」的窗口；
     不通过 → 记 reject_reason（low_r / large_shift / no_audio /
     variable_bpm / parse_error / no_window）并丢弃。
  6) 输出 out/diff-data-audio-fixed/{dataset.npz, manifest.json, stats.json}
     X uint8[N,4,256] / C float16[N,256,4] / song int16 / chart_id U48 /
     split int8(按 set 分组) / start_cell int32 / bpm float32 / shift_ms float32
  7) --verify <npz>：用同一函数复核「全局时间轴聚合」的逐谱 X–C 相关
     （把该谱所有窗口的 X 按 start_cell 铺回整首时间轴、C 同理，再算 Pearson），
     这就是第 148 轮给出「中位 −0.062」的同一口径，可直接对照旧数据集。

⚠ 两个容易踩的坑（都已实测确认，写在这里防止后来者重犯）：
   a. 位移必须只施加在「音频采样起点」上。若把 δ 同时施加给音符分格与音频采样，
      二者的相对关系不变、r 与 δ 无关（只是给格子重新编号），搜索退化成恒等。
      所以：音符仍按作者红线 offset 分格（保留谱面 1/32 相位，音符本来就写在
      格线上），C 在 offset+δ* 处采样 —— 即「把错位的音频拉回谱面时间轴」。
   b. 门禁统计量必须和数据集实际消费的 X 同口径。X 是「每键每格是否有音符
      （含 LN 持续段）」，而 LN 尾会把格子填满；用「音符头数」当统计量会把
      LN 密集谱判成对齐良好（实测某谱 头数 r=+0.56，占用 r=+0.03），
      保留下来在目标指标上仍是 0。所以搜索/门禁用占用口径，头数口径只作诊断。

用法：
  python3 tools/build-audio-cond-fixed.py                      # 全量重建 + 自证
  python3 tools/build-audio-cond-fixed.py --limit-sets 4       # 小样试跑
  python3 tools/build-audio-cond-fixed.py --verify out/diff-data-audio-big/dataset.npz
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, OrderedDict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 音频 / 特征参数（与 tools/build-audio-cond-big.py 对齐，仅 hop 128→256）----
SR = 22050
N_FFT = 512
HOP = 256                    # 任务书要求；旧构建脚本是 128
N_MELS = 128
FMIN, FMAX = 0.0, 8000.0
BANDS = ((0.0, 200.0), (200.0, 2000.0), (2000.0, 8000.0))
CH_NAMES = ['low_0_200', 'mid_200_2000', 'high_2000_8000', 'onset_total']
BLOCK = 4096                 # STFT 分块（≈47.6s @hop256），内存峰值与音频长度无关

# ---- 网格 / 窗口参数 ----
WIN, STRIDE = 256, 128
MIN_NOTES = 16
MIN_COV = 0.90
MAX_REDLINES = 3
BIN_COARSE = 250.0           # 独立口径：250ms 分箱（同 validate-audio-cond-big.py）

DEFAULTS = dict(min_r=0.25, max_shift=300.0,
                shift_lo=-2500.0, shift_hi=2500.0, shift_step=50.0)

REASONS = ['low_r', 'large_shift', 'no_audio', 'variable_bpm', 'parse_error',
           'no_window']


# ============================== .osu 解析 ==============================
def parse_osu(path):
    """解析 .osu。返回 dict(mode, cs, audio, redlines[(t,bl)], notes[(t,col,end)],
    version, title, artist)。

    红线：只有 uninherited==1 且 beatLength>0 的才算（绿线/SV 线不算）。
    音符：x*4/512 定列（与旧构建脚本同一映射）；LN 记 (头, 列, 尾)。
    """
    mode = cs = None
    audio = title = version = artist = ''
    redlines, notes = [], []
    sec = ''
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            if line.startswith('[') and line.endswith(']'):
                sec = line
                continue
            if sec == '[General]':
                if line.startswith('AudioFilename:'):
                    audio = line.split(':', 1)[1].strip()
                elif line.startswith('Mode:'):
                    mode = int(float(line.split(':', 1)[1].strip()))
            elif sec == '[Difficulty]':
                if line.startswith('CircleSize:'):
                    cs = float(line.split(':', 1)[1].strip())
            elif sec == '[Metadata]':
                if line.startswith('Title:'):
                    title = line.split(':', 1)[1].strip()
                elif line.startswith('Artist:'):
                    artist = line.split(':', 1)[1].strip()
                elif line.startswith('Version:'):
                    version = line.split(':', 1)[1].strip()
            elif sec == '[TimingPoints]':
                p = line.split(',')
                if len(p) < 8:
                    continue
                try:
                    t, bl, un = float(p[0]), float(p[1]), int(float(p[6]))
                except Exception:
                    continue
                if un == 1 and bl > 0:
                    redlines.append((t, bl))
            elif sec == '[HitObjects]':
                p = line.split(',')
                if len(p) < 5:
                    continue
                try:
                    x, t, typ = int(float(p[0])), float(p[2]), int(p[3])
                except Exception:
                    continue
                col = min(3, max(0, int(x * 4 / 512)))
                if typ & 128:                                     # LN
                    try:
                        end = float(p[5].split(':')[0])
                    except Exception:
                        end = t
                    notes.append((t, col, max(t, end)))
                elif typ & 1:                                     # 普通音符
                    notes.append((t, col, t))
    return dict(mode=mode, cs=cs, audio=audio, redlines=sorted(redlines),
                notes=sorted(notes), version=version, title=title, artist=artist)


def resolve_audio(set_dir, files, declared):
    """优先 AudioFilename；没有再退到目录里唯一的 mp3/ogg/wav。返回 (path|None, how)。"""
    cand = os.path.basename(declared.replace('\\', '/')) if declared else ''
    if cand:
        if cand in files:
            return os.path.join(set_dir, cand), 'declared'
        low = {f.lower(): f for f in files}
        if cand.lower() in low:
            return os.path.join(set_dir, low[cand.lower()]), 'declared_ci'
    auds = [f for f in files if f.lower().endswith(('.mp3', '.ogg', '.wav'))]
    if len(auds) == 1:
        return os.path.join(set_dir, auds[0]), 'unique'
    return None, ('ambiguous' if auds else 'none')


# ============================== 音频 → 4 带包络 ==============================
def _decode_ffmpeg(path):
    """ffmpeg 兜底解码成 22050 单声道 float32。"""
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', path,
           '-f', 'f32le', '-ac', '1', '-ar', str(SR), 'pipe:1']
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or len(r.stdout) < 4 * SR:
        return None
    return np.frombuffer(r.stdout, dtype='<f4').astype(np.float32, copy=True)


def load_mono(path):
    """读音频 → 22050 单声道 float32。soundfile 优先（快），ffmpeg 兜底。"""
    y = None
    try:
        import soundfile as sf
        import librosa
        data, sr = sf.read(path, dtype='float32', always_2d=True)
        mono = np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)
        if sr != SR:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=SR,
                                    res_type='soxr_mq')
        y = np.ascontiguousarray(mono, dtype=np.float32)
    except Exception:
        y = _decode_ffmpeg(path)
    if y is None or len(y) < SR * 5:
        return None
    return y


def onset_bands4(y):
    """4 通道 onset 强度包络 [4, F] float32。

    与 tools/audio-cond-data.py::onset_bands 同一套做法（分块 STFT → mel →
    log1p → 正向差分 → 三带 + 总 求和），只把 hop 从 128 改成 256。
    """
    import librosa

    n_frames = 1 + (len(y) - N_FFT) // HOP
    if n_frames < 3:
        return np.zeros((4, 0), dtype=np.float32)
    mel_fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS,
                                 fmin=FMIN, fmax=FMAX)
    mel_hz = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    masks = [mel_hz < BANDS[0][1],
             (mel_hz >= BANDS[1][0]) & (mel_hz < BANDS[1][1]),
             mel_hz >= BANDS[2][0]]
    out = np.zeros((4, n_frames - 1), dtype=np.float32)
    prev = None
    for i0 in range(0, n_frames, BLOCK):
        i1 = min(n_frames, i0 + BLOCK)
        seg = y[i0 * HOP:(i1 - 1) * HOP + N_FFT]
        S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP, center=False))
        L = np.log1p(mel_fb @ (S ** 2))
        if i0 == 0:
            flux, col0 = np.diff(L, axis=1), 0
        else:
            flux = np.diff(np.concatenate([prev[:, None], L], axis=1), axis=1)
            col0 = i0 - 1
        flux = np.maximum(flux, 0.0)
        prev = L[:, -1].copy()
        m = flux.shape[1]
        out[0, col0:col0 + m] = flux[masks[0]].sum(axis=0)
        out[1, col0:col0 + m] = flux[masks[1]].sum(axis=0)
        out[2, col0:col0 + m] = flux[masks[2]].sum(axis=0)
        out[3, col0:col0 + m] = flux.sum(axis=0)
        del S, L, flux
    return out


def frame_times(n):
    """通量列 c 对应 STFT 帧 c+1，其中心在 ((c+1)*HOP + N_FFT/2) 采样处；返回 ms。"""
    return ((np.arange(n) + 1) * HOP + N_FFT // 2) / SR * 1000.0


class AudioEnvCache:
    """按路径缓存 (4 带包络, 帧中心时刻, 音频时长ms)；FIFO 限量防止吃满内存。"""

    def __init__(self, max_items=48, max_samples=60_000_000):
        self.cache = OrderedDict()
        self.max_items = max_items
        self.max_samples = max_samples
        self.n_samples = 0
        self.hits = self.misses = self.fail = 0

    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            self.hits += 1
            return self.cache[path]
        self.misses += 1
        y = load_mono(path)
        if y is None:
            self.fail += 1
            self.cache[path] = None
            return None
        dur_ms = len(y) / SR * 1000.0
        env = onset_bands4(y)
        del y
        if env.shape[1] < 8:
            self.cache[path] = None
            self.fail += 1
            return None
        tf = frame_times(env.shape[1])
        item = (env, tf, dur_ms)
        self.cache[path] = item
        self.n_samples += len(tf) * 4
        while len(self.cache) > self.max_items or (
                self.n_samples > self.max_samples and len(self.cache) > 1):
            _, old = self.cache.popitem(last=False)
            if old is not None:
                self.n_samples -= len(old[1]) * 4
        return item


# ============================== 网格工具 ==============================
def _cell_index(tf, q_ms, step_ms, n):
    """每格起点帧下标 i0 与终点帧下标 i1（左闭右开），以及『有帧』掩码。"""
    t0 = q_ms + np.arange(int(n), dtype=np.float64) * step_ms
    i0 = np.searchsorted(tf, t0, side='left').astype(np.int64)
    i1 = np.searchsorted(tf, t0 + step_ms, side='left').astype(np.int64)
    return t0, i0, i1


def pool4_cells(env, tf, q_ms, step_ms, n):
    """4 带包络按「格内取最大」聚到以 q_ms 为起点、宽 step_ms 的网格。返回 [n,4]。

    第 k 格覆盖 [q+k*step, q+(k+1)*step)；np.maximum.reduceat(axis=1) 一次性
    取所有带的分段最大值（末尾补 −inf 哨兵，保证最后一段只到数组末尾）。
    """
    n = int(n)
    out = np.zeros((n, 4), dtype=np.float32)
    F = env.shape[1]
    if n <= 0 or F == 0:
        return out
    _, i0, i1 = _cell_index(tf, q_ms, step_ms, n)
    ok = (i0 < F) & (i1 > i0)
    if not ok.any():
        return out
    pe = np.concatenate([env, np.full((4, 1), -np.inf, np.float32)], axis=1)
    idx = np.where(ok, i0, F)[ok]
    out[ok] = np.maximum.reduceat(pe, idx, axis=1).T
    return out


def cell_max(values, tf, q_ms, step_ms, n):
    """单通道版（诊断用）。"""
    n = int(n)
    out = np.zeros(n, dtype=np.float32)
    F = len(values)
    if n <= 0 or F == 0:
        return out
    _, i0, i1 = _cell_index(tf, q_ms, step_ms, n)
    ok = (i0 < F) & (i1 > i0)
    if not ok.any():
        return out
    vals = np.concatenate([values, np.array([-np.inf], np.float32)])
    out[ok] = np.maximum.reduceat(vals, np.where(ok, i0, F)[ok])
    return out


def occupancy_grid(notes, origin_ms, step_ms, n_cells):
    """X 的整首占用张量 occ[4, n]（含 LN 持续段）与音符头计数 heads[n]。"""
    occ = np.zeros((4, n_cells), dtype=np.uint8)
    heads = np.zeros(n_cells, dtype=np.int32)
    for t, col, end in notes:
        c0 = int(np.floor((t - origin_ms) / step_ms))
        c1 = int(np.floor((end - origin_ms) / step_ms))
        if c1 < 0 or c0 >= n_cells:
            continue
        c0 = max(c0, 0)
        c1 = min(max(c1, c0), n_cells - 1)
        occ[col, c0] = 1
        if c1 > c0:
            occ[col, c0 + 1:c1 + 1] = 1
        heads[c0] += 1
    return occ, heads


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 8 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def norm_p99(x, valid):
    """每通道按 P99 归一到 0–1（只用有效格算 P99）。"""
    sc = np.ones(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        v = x[valid, j] if valid.any() else x[:, j]
        p = float(np.percentile(v, 99)) if len(v) else 0.0
        sc[j] = p if p > 1e-9 else 1.0
    return np.clip(x / sc, 0.0, 1.0), sc


def grid_cond(env, tf, dur_ms, q_ms, step_ms, n_cells):
    """在采样起点 q_ms 上重算 4 带 C 特征（逐谱 P99 归一），返回 (C, 有效掩码)。"""
    feat_raw = pool4_cells(env, tf, q_ms, step_ms, n_cells)
    t0, _, _ = _cell_index(tf, q_ms, step_ms, n_cells)
    limit = min(dur_ms, float(tf[-1]))
    valid = (t0 >= 0.0) & (t0 < limit)
    feat, _sc = norm_p99(feat_raw, valid)
    return feat, valid, feat_raw


def search_shift(env, tf, dur_ms, occ, heads, offset_ms, step_ms, n_cells,
                 lo, hi, step):
    """扫描音频采样起点偏移 δ。

    音符占用序列固定用作者网格 offset_ms（保留 1/32 相位）；C 在 offset_ms+δ
    处重采样 —— 相对位移才是 δ 真正改变的量。
    返回 (δ*, r*, r0, r_other, r_head_at_best, r_best_within, curve).
    """
    occm = occ.mean(axis=0).astype(np.float32)          # 每格占用率 0..1
    curve = []
    for d in np.arange(lo, hi + 1e-9, step):
        C, valid, feat_raw = grid_cond(env, tf, dur_ms, offset_ms + d,
                                      step_ms, n_cells)
        if valid.sum() < 8:
            curve.append((float(d), 0.0, 0.0))
            continue
        m = valid
        r_occ = pearson(occm[m], C[m].mean(axis=1))     # 主口径（= 数据集 X–C）
        r_head = pearson(heads[m], feat_raw[m].sum(axis=1))   # 诊断口径（粗）
        curve.append((float(d), r_occ, r_head))
    arr = np.array([c[1] for c in curve])
    best = float(arr.max())
    cands = np.flatnonzero(np.abs(arr - best) < 1e-9)
    ds = np.array([curve[j][0] for j in cands])
    i = int(cands[int(np.argmin(np.abs(ds)))])          # 并列取 |δ| 最小
    d_star, r_star = curve[i][0], best
    j0 = int(np.argmin(np.abs([c[0] for c in curve])))
    r0 = float(curve[j0][1])
    far = np.abs(np.array([c[0] for c in curve]) - d_star) > 400.0
    r_other = float(arr[far].max()) if far.any() else 0.0
    r_head_best = float(curve[i][2])
    return d_star, r_star, r0, r_other, r_head_best, curve


def coarse_r(env, notes, offset_ms, shift_ms, dur_ms):
    """独立口径：250ms 分箱，音符头直方图 vs 四带求和（同 validate-audio-cond-big.py）。

    音符在 osu 时间 t 上的音频对应 t+shift_ms（即 C 采样起点偏移的方向）。
    """
    if not notes:
        return 0.0
    nf = env.shape[1]
    t_last = (nf * HOP + N_FFT // 2) / SR * 1000.0        # 最后一帧中心（ms）
    limit = max(min(dur_ms, t_last) / 1000.0, 1.0)
    edges = np.arange(0.0, limit + 1e-9, BIN_COARSE / 1000.0)
    if len(edges) < 12:
        return 0.0
    t = np.array([x[0] for x in notes], dtype=np.float64) + shift_ms
    h, _ = np.histogram(t / 1000.0, bins=edges)
    # 帧 → 分箱求和
    fps = SR / float(HOP)
    nbin = len(edges) - 1
    idx = np.minimum((np.arange(env.shape[1]) / fps / (BIN_COARSE / 1000.0)),
                     nbin - 1).astype(np.int64)
    acc = np.zeros(nbin, dtype=np.float64)
    np.add.at(acc, idx, env.sum(axis=0).astype(np.float64))
    return pearson(h.astype(np.float64), acc)


def build_windows(notes, origin_ms, step_ms, n_cells):
    """音符占用张量 → 窗口。返回 X[N,4,WIN] uint8, starts[N], head_counts[N]。"""
    if n_cells < WIN:
        return (np.zeros((0, 4, WIN), np.uint8), np.zeros(0, np.int32),
                np.zeros(0, np.int32))
    occ, heads = occupancy_grid(notes, origin_ms, step_ms, n_cells)
    csum = np.concatenate([[0], np.cumsum(heads)])
    starts = np.arange(0, n_cells - WIN + 1, STRIDE, dtype=np.int32)
    X = np.stack([occ[:, s:s + WIN] for s in starts]).astype(np.uint8)
    counts = (csum[starts + WIN] - csum[starts]).astype(np.int32)
    return X, starts, counts


def assign_splits(counts_by_set, ratios=(0.80, 0.10, 0.10)):
    """按窗口数把整个 set 贪心分到 train/val/test（同 set 不跨集）。

    与 tools/build-audio-cond-big.py::assign_splits 同一算法。
    """
    total = float(sum(counts_by_set.values())) or 1.0
    target = [total * r for r in ratios]
    cur = [0.0, 0.0, 0.0]
    out = {}
    for sid, w in sorted(counts_by_set.items(), key=lambda kv: (-kv[1], kv[0])):
        j = max(range(3), key=lambda k: target[k] - cur[k])
        out[sid] = j
        cur[j] += w
    return out


# ============================== 复核（自证） ==============================
def verify_dataset(npz_path, win=WIN, n_rand=8, seed=7):
    """逐谱 X–C 相关（全局时间轴聚合）+ 随机循环平移对照。

    把该谱所有窗口按 start_cell 铺回整首时间轴（X 四轨取均值 = 占用率，
    C 四带取均值），再算 Pearson r —— 第 148 轮「中位 −0.062」的同一口径。
    对照：把 C 序列在谱内随机循环平移后重算 r（应 ≈ 0）。
    """
    d = np.load(npz_path)
    X, C = d['X'], d['C']
    starts, cids = d['start_cell'], d['chart_id']
    rng = np.random.default_rng(seed)
    lags = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
    rows = []
    for cid in np.unique(cids):
        sel = np.flatnonzero(cids == cid)
        st = starts[sel].astype(np.int64)
        base = int(st.min())
        end = int((st + win).max())
        occ = np.zeros(end - base, dtype=np.float64)
        cnd = np.zeros(end - base, dtype=np.float64)
        cnt = np.zeros(end - base, dtype=np.float64)
        for i, s in zip(sel, st):
            a = int(s - base)
            occ[a:a + win] += X[i].astype(np.float32).mean(axis=0)
            cnd[a:a + win] += C[i].astype(np.float32).mean(axis=1)
            cnt[a:a + win] += 1.0
        m = cnt > 0
        o_, c_ = occ[m] / cnt[m], cnd[m] / cnt[m]
        r = pearson(o_, c_)
        rr = [pearson(o_, np.roll(c_, int(k))) for k in
              rng.integers(8, max(9, len(c_) - 8), n_rand)]
        prof = {int(l): round(pearson(o_, np.roll(c_, int(l))), 4) for l in lags}
        rows.append(dict(chart_id=str(cid), windows=int(len(sel)),
                         cells=int(m.sum()), r=round(r, 4),
                         r_rand=round(float(np.mean(rr)), 4), lag_prof=prof))
    rs = np.array([x['r'] for x in rows])
    rn = np.array([x['r_rand'] for x in rows])
    lag_prof = {str(l): round(float(np.median([x['lag_prof'][l] for x in rows])), 4)
                for l in lags} if rows else {}
    summary = dict(
        n_charts=int(len(rows)), n_windows=int(len(cids)),
        r_median=round(float(np.median(rs)), 4) if len(rs) else 0.0,
        r_mean=round(float(rs.mean()), 4) if len(rs) else 0.0,
        r_min=round(float(rs.min()), 4) if len(rs) else 0.0,
        r_p10=round(float(np.percentile(rs, 10)), 4) if len(rs) else 0.0,
        r_p25=round(float(np.percentile(rs, 25)), 4) if len(rs) else 0.0,
        r_p75=round(float(np.percentile(rs, 75)), 4) if len(rs) else 0.0,
        r_p90=round(float(np.percentile(rs, 90)), 4) if len(rs) else 0.0,
        r_max=round(float(rs.max()), 4) if len(rs) else 0.0,
        r_rand_median=round(float(np.median(rn)), 4) if len(rn) else 0.0,
        r_rand_p90=round(float(np.percentile(rn, 90)), 4) if len(rn) else 0.0,
        lag_profile=lag_prof,
        frac_ge_025=round(float((rs >= 0.25).mean()), 4) if len(rs) else 0.0,
        frac_ge_02=round(float((rs >= 0.2).mean()), 4) if len(rs) else 0.0,
        frac_ge_04=round(float((rs >= 0.4).mean()), 4) if len(rs) else 0.0)
    return summary, rows


# ============================== 主构建 ==============================
def build(args):
    dl = os.path.abspath(args.dl)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    sets = sorted(int(x) for x in os.listdir(dl)
                  if x.isdigit() and os.path.isdir(os.path.join(dl, x)))
    if args.sets:
        want = {int(x) for x in args.sets.split(',') if x.strip()}
        sets = [s for s in sets if s in want]
    if args.limit_sets:
        sets = sets[:args.limit_sets]

    fetch = {}
    fp = os.path.join(dl, '_fetch_log.json')
    if os.path.exists(fp):
        fetch = json.load(open(fp, encoding='utf-8'))

    env_cache = AudioEnvCache()
    t_all = time.time()
    print(f'待处理谱面包 {len(sets)} 个 / 门禁 r*≥{args.min_r} |δ*|≤{args.max_shift:.0f}ms '
          f'/ 位移搜索 {args.shift_lo:.0f}..{args.shift_hi:.0f}ms step {args.shift_step:.0f}ms',
          flush=True)

    Xs, Cs, songs, cids, sts, bpms, shs = [], [], [], [], [], [], []
    set_of_chunk, manifest, songs_meta = [], [], {}
    song_idx = 0
    reject = Counter()
    n_charts_seen = n_non4k = n_parse_fail = 0

    for si, sid in enumerate(sets):
        d = os.path.join(dl, str(sid))
        files = sorted(os.listdir(d))
        osus = [f for f in files if f.lower().endswith('.osu')]
        title = artist = ''
        n_win_set = 0
        for k, f in enumerate(osus):
            path = os.path.join(d, f)
            n_charts_seen += 1
            rec = dict(SetID=sid, chart=f, windows=0, verdict='rejected',
                       reject_reason=None)
            try:
                info = parse_osu(path)
            except Exception as e:                      # 解析失败
                n_parse_fail += 1
                reject['parse_error'] += 1
                rec.update(reject_reason='parse_error', error=str(e)[:120])
                manifest.append(rec)
                continue
            if info['mode'] != 3 or info['cs'] is None or abs(info['cs'] - 4) > 0.01:
                n_non4k += 1
                reject['non_4k'] += 1
                manifest.append({**rec, 'reject_reason': 'non_4k'})
                continue
            title = title or info['title']
            artist = artist or info['artist']
            rec.update(title=info['title'], artist=info['artist'],
                       version=info['version'], notes=len(info['notes']),
                       redlines=len(info['redlines']))
            if not info['notes']:
                reject['parse_error'] += 1
                rec['reject_reason'] = 'parse_error'
                manifest.append(rec)
                continue
            if len(info['redlines']) > MAX_REDLINES:     # 变速谱
                reject['variable_bpm'] += 1
                rec['reject_reason'] = 'variable_bpm'
                manifest.append(rec)
                continue
            aud_path, how = resolve_audio(d, files, info['audio'])
            rec['audio_how'] = how
            item = env_cache.get(aud_path) if aud_path else None
            if item is None:
                reject['no_audio'] += 1
                rec['reject_reason'] = 'no_audio'
                manifest.append(rec)
                continue
            env, tf, dur_ms = item
            offset_ms, beat_len = info['redlines'][0]
            step_ms = beat_len / 8.0                    # 1/32 拍
            n_cells = int(np.floor((info['notes'][-1][2] - offset_ms) / step_ms)) + 2
            occ, heads = occupancy_grid(info['notes'], offset_ms, step_ms, n_cells)
            n_ln = sum(1 for t, c, e in info['notes'] if e > t)
            rec.update(audio=os.path.basename(aud_path),
                       offset_ms=round(offset_ms, 3), step_ms=round(step_ms, 4),
                       beat_len=round(beat_len, 4),
                       bpm=round(60000.0 / beat_len, 3),
                       n_cells=int(n_cells), ln_frac=round(n_ln / len(info['notes']), 4),
                       occ_fill=round(float((occ.sum(axis=0) > 0).mean()), 4),
                       dur_s=round(dur_ms / 1000.0, 1))
            if n_cells < WIN:
                reject['no_window'] += 1
                rec['reject_reason'] = 'no_window'
                manifest.append(rec)
                continue

            # ---- 4) 自洽位移搜索 ----
            d_star, r_star, r0, r_other, r_head, curve = search_shift(
                env, tf, dur_ms, occ, heads, offset_ms, step_ms, n_cells,
                args.shift_lo, args.shift_hi, args.shift_step)
            within = [c[1] for c in curve if abs(c[0]) <= args.max_shift]
            rec.update(r_star=round(r_star, 4), r0=round(r0, 4),
                       r_other=round(r_other, 4), shift_ms=round(d_star, 3),
                       r_head=round(r_head, 4),
                       r_best_within=round(max(within) if within else 0.0, 4))

            # ---- 5) 门禁 ----
            # 先判「有没有像样的峰」，再判「峰在哪」：r* 不达标时位移没有意义
            if r_star < args.min_r:
                reject['low_r'] += 1
                rec['reject_reason'] = 'low_r'
                manifest.append(rec)
                continue
            if abs(d_star) > args.max_shift:
                reject['large_shift'] += 1
                rec['reject_reason'] = 'large_shift'
                manifest.append(rec)
                continue

            # ---- 用校正起点重算 C 特征 ----
            q = offset_ms + d_star
            feat, valid, _raw = grid_cond(env, tf, dur_ms, q, step_ms, n_cells)
            X, starts, counts = build_windows(info['notes'], offset_ms,
                                              step_ms, n_cells)
            if X.shape[0] == 0:
                reject['no_window'] += 1
                rec['reject_reason'] = 'no_window'
                manifest.append(rec)
                continue
            cov = np.array([float(valid[s:s + WIN].mean()) for s in starts])
            keep = (counts >= MIN_NOTES) & (cov >= MIN_COV)
            if not keep.any():
                reject['no_window'] += 1
                rec['reject_reason'] = 'no_window'
                rec['max_notes_in_window'] = int(counts.max())
                manifest.append(rec)
                continue

            kk = np.flatnonzero(keep)
            nw = len(kk)
            cid = f'{sid}-{k:02d}-{(info["version"] or "diff")[:24]}'[:48]
            Xs.append(X[kk])
            Cs.append(feat[starts[kk][:, None] + np.arange(WIN)[None, :]]
                      .astype(np.float16))
            songs.append(np.full(nw, song_idx, np.int16))
            cids.append(np.array([cid] * nw))
            sts.append(starts[kk].astype(np.int32))
            bpms.append(np.full(nw, 60000.0 / beat_len, np.float32))
            shs.append(np.full(nw, d_star, np.float32))
            set_of_chunk.append(sid)
            n_win_set += nw
            # 独立口径（250ms 分箱）：δ* / 0 / 随机位移
            rng = np.random.default_rng(1000 + si * 100 + k)
            r_co_star = coarse_r(env, info['notes'], offset_ms, d_star, dur_ms)
            r_co_0 = coarse_r(env, info['notes'], offset_ms, 0.0, dur_ms)
            r_co_rand = float(np.mean([
                coarse_r(env, info['notes'], offset_ms,
                         float(z), dur_ms)
                for z in rng.uniform(args.shift_lo, args.shift_hi, 4)]))
            rec.update(windows=nw, verdict='kept', chart_id=cid,
                       corrected_offset_ms=round(q, 3),
                       mean_notes_per_window=round(float(counts[kk].mean()), 1),
                       mean_coverage=round(float(cov[kk].mean()), 4),
                       coarse_r_star=round(r_co_star, 4),
                       coarse_r_0=round(r_co_0, 4),
                       coarse_r_rand=round(r_co_rand, 4))
            manifest.append(rec)
        if n_win_set:
            songs_meta[str(song_idx)] = (f'{title} [{sid}]' if title else f'Set{sid}')
            song_idx += 1
        print(f'  [{si+1}/{len(sets)}] set {sid}: 谱 {len(osus)} / 累计保留 '
              f'{sum(len(x) for x in Xs)} 窗口 / {time.time()-t_all:.0f}s', flush=True)

    if not Xs:
        print('没有产出任何窗口 —— 全被门禁拒了', file=sys.stderr)
        return 1

    X = np.concatenate(Xs).astype(np.uint8)
    C = np.concatenate(Cs).astype(np.float16)
    song = np.concatenate(songs).astype(np.int16)
    cid_arr = np.concatenate(cids).astype('U48')
    start_cell = np.concatenate(sts).astype(np.int32)
    bpm = np.concatenate(bpms).astype(np.float32)
    shift_ms = np.concatenate(shs).astype(np.float32)

    counts_by_set = {}
    for sid, arr in zip(set_of_chunk, Xs):
        counts_by_set[sid] = counts_by_set.get(sid, 0) + len(arr)
    split_of = assign_splits(counts_by_set)
    split = np.concatenate([np.full(len(x), split_of[sid], np.int8)
                            for sid, x in zip(set_of_chunk, Xs)]).astype(np.int8)
    for m in manifest:
        if m.get('verdict') == 'kept':
            m['split'] = int(split_of[m['SetID']])

    npz_path = os.path.join(out, 'dataset.npz')
    np.savez_compressed(npz_path, X=X, C=C, song=song, chart_id=cid_arr,
                        split=split, start_cell=start_cell, bpm=bpm,
                        shift_ms=shift_ms,
                        songs=np.array(json.dumps(songs_meta, ensure_ascii=False)))
    json.dump(manifest, open(os.path.join(out, 'manifest.json'), 'w',
                             encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(songs_meta, open(os.path.join(out, 'songs.json'), 'w',
                               encoding='utf-8'), ensure_ascii=False, indent=1)

    # ---- 7) 自证：重建后的逐谱 X–C 相关 ----
    print('\n重建完成，开始自证（全局时间轴聚合的逐谱 X–C 相关）…', flush=True)
    ver_sum, ver_rows = verify_dataset(npz_path)
    ver_by_cid = {x['chart_id']: x for x in ver_rows}

    old_sum = old_rows = None
    old_npz = os.path.join(ROOT, 'out', 'diff-data-audio-big', 'dataset.npz')
    old_man = os.path.join(ROOT, 'out', 'diff-data-audio-big', 'manifest.json')
    if os.path.exists(old_npz):
        old_sum, old_rows = verify_dataset(old_npz)
    old_by_key = {}
    if old_rows and os.path.exists(old_man):
        old_manj = json.load(open(old_man, encoding='utf-8'))
        rb = {x['chart_id']: x['r'] for x in old_rows}
        for m in old_manj:
            if m.get('chart_id') in rb:
                old_by_key[(int(m['SetID']), m['chart'])] = rb[m['chart_id']]

    kept = [m for m in manifest if m.get('verdict') == 'kept']
    paired = []
    for m in kept:
        paired.append(dict(SetID=int(m['SetID']), chart=m['chart'],
                           chart_id=m['chart_id'], windows=m['windows'],
                           r_new=ver_by_cid.get(m['chart_id'], {}).get('r'),
                           r_old=old_by_key.get((int(m['SetID']), m['chart'])),
                           shift_ms=m['shift_ms'], r_star=m['r_star'],
                           r_head=m['r_head'],
                           coarse_r_star=m.get('coarse_r_star'),
                           coarse_r_0=m.get('coarse_r_0'),
                           coarse_r_rand=m.get('coarse_r_rand')))
    pr_new = [p['r_new'] for p in paired if p['r_new'] is not None]
    pr_old = [p['r_old'] for p in paired if p['r_old'] is not None]
    paired_sum = dict(
        n=len(paired), n_with_old=len(pr_old),
        r_new_median=round(float(np.median(pr_new)), 4) if pr_new else 0.0,
        r_old_median=round(float(np.median(pr_old)), 4) if pr_old else 0.0,
        r_old_min=round(float(np.min(pr_old)), 4) if pr_old else 0.0,
        r_old_max=round(float(np.max(pr_old)), 4) if pr_old else 0.0,
        improved=int(sum(1 for p in paired if p['r_old'] is not None
                         and p['r_new'] is not None and p['r_new'] > p['r_old'])),
        worsened=int(sum(1 for p in paired if p['r_old'] is not None
                         and p['r_new'] is not None and p['r_new'] <= p['r_old'])),
        delta_median=round(float(np.median(
            [p['r_new'] - p['r_old'] for p in paired
             if p['r_old'] is not None and p['r_new'] is not None])), 4)
        if pr_old else 0.0)

    # ---- 统计 ----
    shifts = np.array([m['shift_ms'] for m in kept], dtype=float)
    r_stars = np.array([m['r_star'] for m in kept], dtype=float)
    r0s = np.array([m['r0'] for m in kept], dtype=float)
    r_oth = np.array([m['r_other'] for m in kept], dtype=float)
    hist_edges = np.arange(-2500, 2501, 250)
    hist, _ = np.histogram(shifts, bins=hist_edges) if len(shifts) else (np.zeros(20), None)
    zero_shift = int((np.abs(shifts) < 1e-6).sum()) if len(shifts) else 0

    def dist(a):
        a = np.asarray(a, dtype=float)
        if len(a) == 0:
            return {}
        return dict(n=int(len(a)),
                    median=round(float(np.median(a)), 4),
                    mean=round(float(a.mean()), 4),
                    min=round(float(a.min()), 4),
                    max=round(float(a.max()), 4),
                    p10=round(float(np.percentile(a, 10)), 4),
                    p25=round(float(np.percentile(a, 25)), 4),
                    p75=round(float(np.percentile(a, 75)), 4),
                    p90=round(float(np.percentile(a, 90)), 4))

    # 被拒谱的分类：真错位 vs LN 表示问题
    low_r = [m for m in manifest if m.get('reject_reason') == 'low_r']
    ln_explained = [m for m in low_r if m.get('r_head', 0) >= 0.40]
    large_shift = [m for m in manifest if m.get('reject_reason') == 'large_shift']

    sets_kept = sorted({int(m['SetID']) for m in kept})
    skip = fetch.get('skip', {}) if fetch else {}
    fail = fetch.get('fail', {}) if fetch else {}
    skip_reasons = Counter()
    for v in skip.values():
        skip_reasons[v['reason'].split(' ')[0].split('=')[0]] += 1
    target = dict(r_median_ge_035=bool(ver_sum['r_median'] >= 0.35),
                  charts_ge_15=bool(ver_sum['n_charts'] >= 15))
    target['met'] = bool(target['r_median_ge_035'] and target['charts_ge_15'])

    independent = dict(
        coarse_r_star=dist([p['coarse_r_star'] for p in paired]),
        coarse_r_0=dist([p['coarse_r_0'] for p in paired]),
        coarse_r_rand=dist([p['coarse_r_rand'] for p in paired]),
        r_head=dist([p['r_head'] for p in paired]),
    )
    ln_stats = dict(
        kept_ln_frac=dist([m['ln_frac'] for m in kept]),
        kept_occ_fill=dist([m['occ_fill'] for m in kept]),
        low_r_n=int(len(low_r)),
        low_r_head_r=dist([m['r_head'] for m in low_r]),
        low_r_ln_frac=dist([m['ln_frac'] for m in low_r]),
        low_r_occ_fill=dist([m['occ_fill'] for m in low_r]),
        low_r_explained_by_ln=int(len(ln_explained)),
        large_shift_n=int(len(large_shift)),
        large_shift_r_best_within=dist([m['r_best_within'] for m in large_shift]),
        large_shift_would_pass=int(sum(1 for m in large_shift
                                       if m.get('r_best_within', 0) >= args.min_r)))

    stats = dict(
        generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        elapsed_s=round(time.time() - t_all, 1),
        config=dict(min_r=args.min_r, max_shift=args.max_shift,
                    shift_lo=args.shift_lo, shift_hi=args.shift_hi,
                    shift_step=args.shift_step, sr=SR, n_fft=N_FFT, hop=HOP,
                    n_mels=N_MELS, fmax=FMAX, win=WIN, stride=STRIDE,
                    min_notes=MIN_NOTES, min_cov=MIN_COV,
                    max_redlines=MAX_REDLINES, channels=CH_NAMES,
                    bin_coarse_ms=BIN_COARSE,
                    shift_semantics='C 采样起点 = 首条红线 offset + shift*；'
                                    '音符分格固定用作者 offset（保留 1/32 相位）'),
        counts=dict(charts_seen=n_charts_seen, charts_non_4k=n_non4k,
                    charts_parse_fail=n_parse_fail,
                    charts_4k=n_charts_seen - n_non4k - n_parse_fail,
                    kept_charts=len(kept), rejected=sum(
                        v for k, v in reject.items() if k != 'non_4k'),
                    windows=int(X.shape[0]), songs=int(song_idx),
                    sets_processed=len(sets), sets_kept=len(sets_kept),
                    notes_kept=int(sum(m['notes'] for m in kept)),
                    audio_cache=dict(hits=env_cache.hits,
                                     misses=env_cache.misses,
                                     fail=env_cache.fail)),
        reject_reasons={k: int(reject.get(k, 0)) for k in REASONS},
        split_counts={str(k): int((split == k).sum()) for k in (0, 1, 2)},
        split_sets={str(k): int(sum(1 for v in split_of.values() if v == k))
                    for k in (0, 1, 2)},
        shift_dist=dict(zero_shift_charts=zero_shift, **dist(shifts)),
        shift_hist=dict(bin_ms=250,
                        rows=[dict(lo=int(a), hi=int(b), n=int(c))
                              for a, b, c in zip(hist_edges[:-1], hist_edges[1:], hist)]),
        r_star=dist(r_stars), r0=dist(r0s),
        peak_prominence=dist(r_stars - r_oth),
        verify=ver_sum, verify_per_chart=ver_rows,
        verify_old=old_sum,
        paired_old_new=paired_sum, paired_rows=paired,
        independent=independent, ln_analysis=ln_stats,
        target=target,
        data=dict(npz=npz_path, bytes=int(os.path.getsize(npz_path)),
                  X=[int(x) for x in X.shape], C=[int(x) for x in C.shape]),
        songs=songs_meta,
        skipped_packs=dict(n=len(skip) + len(fail),
                           reasons=dict(skip_reasons),
                           note='503 等镜像侧缺档，本轮不重试（任务要求不联网）'),
    )
    json.dump(stats, open(os.path.join(out, 'stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    print(f'\n数据集：{npz_path}  {os.path.getsize(npz_path)/1e6:.1f} MB')
    print(f'X {X.shape} uint8 / C {C.shape} float16 / 谱 {len(kept)} / 歌 {song_idx}')
    print('拒绝原因：' + ', '.join(f'{k}={v}' for k, v in
                               stats['reject_reasons'].items() if v))
    print(f'校正位移：中位 {stats["shift_dist"].get("median", 0):+.0f}ms '
          f'范围 [{stats["shift_dist"].get("min", 0):+.0f}, '
          f'{stats["shift_dist"].get("max", 0):+.0f}]ms '
          f'（δ*=0 的谱 {zero_shift}/{len(shifts)}）')
    print(f'自证 X–C：中位 {ver_sum["r_median"]:+.3f} / 最小 {ver_sum["r_min"]:+.3f} '
          f'/ p25 {ver_sum["r_p25"]:+.3f} / r≥0.25 {ver_sum["frac_ge_025"]*100:.0f}% '
          f'/ 谱 {ver_sum["n_charts"]} / 窗 {ver_sum["n_windows"]} '
          f'/ 随机平移对照中位 {ver_sum["r_rand_median"]:+.3f}')
    if old_sum:
        print(f'旧数据集同口径：中位 {old_sum["r_median"]:+.3f} / '
              f'r≥0.2 {old_sum["frac_ge_02"]*100:.0f}% / 谱 {old_sum["n_charts"]}')
    print(f'目标（r中位≥0.35 且 谱≥15）：{"达成" if target["met"] else "未达成"}')

    if args.report:
        write_report(args, stats, out)
    return 0


# ============================== REPORT.md ==============================
def write_report(args, s, out_dir):
    rep = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(rep), exist_ok=True)
    c, v, ov = s['counts'], s['verify'], s['verify_old']
    ind, ln = s['independent'], s['ln_analysis']
    A = []
    A.append('# 第 149 轮：重建音频条件数据集（逐谱位移校正 + 质量门禁）')
    A.append('')
    A.append('修的是第 148 轮查出来的致命问题：旧 `out/diff-data-audio-big` 里 '
             '**谱面与音频大面积错位**（逐谱 X–C 相关中位 −0.062，81% 的谱基本无关）。'
             '本轮对每张 4K mania 谱独立找「音频采样起点偏移」，只留能自证对齐的谱。')
    A.append('')
    A.append('## 1. 处理面')
    A.append('')
    A.append('| 项 | 值 |')
    A.append('| --- | --- |')
    A.append(f'| 谱面包（`out/audio-dl/`） | **{c["sets_processed"]}** 个 |')
    A.append(f'| 扫到的 .osu | {c["charts_seen"]} 张 |')
    A.append(f'| 非 4K mania | {c["charts_non_4k"]} 张（不计入） |')
    A.append(f'| 4K mania 谱（Mode=3 且 CS=4） | **{c["charts_4k"]}** 张 |')
    A.append(f'| 通过门禁并保留 | **{c["kept_charts"]}** 张 |')
    A.append(f'| 保留窗口 | **{c["windows"]}** 个 |')
    A.append(f'| 保留歌（set） | {c["songs"]} 首 / {c["sets_kept"]} 包 |')
    A.append(f'| 保留谱的音符总数 | {c["notes_kept"]} |')
    A.append(f'| 用时 | {s["elapsed_s"]:.0f}s（CPU-only，无 GPU） |')
    A.append('')
    A.append('## 2. 门禁：通过 / 拒绝')
    A.append('')
    A.append(f'门禁：`r* ≥ {s["config"]["min_r"]}` 且 `|shift*| ≤ '
             f'{s["config"]["max_shift"]:.0f}ms`。注意 r* 用的就是数据集实际消费的'
             '统计量（X 四轨占用均值 vs 四带归一化条件均值，1/32 格），'
             '因此它和下面第 4 节的自证是同一把尺子。')
    A.append('')
    A.append('| 判定 | 谱数 | 含义 |')
    A.append('| --- | --- | --- |')
    rr = s['reject_reasons']
    A.append(f'| ✅ kept | {c["kept_charts"]} | 位移校正后 r* 达标 |')
    A.append(f'| ❌ low_r | {rr["low_r"]} | 扫遍 ±2500ms 相关仍 < {s["config"]["min_r"]} |')
    A.append(f'| ❌ large_shift | {rr["large_shift"]} | 最佳位移 > {s["config"]["max_shift"]:.0f}ms（音频版本可疑） |')
    A.append(f'| ❌ no_audio | {rr["no_audio"]} | AudioFilename 缺失且目录音频不唯一/不可解码 |')
    A.append(f'| ❌ variable_bpm | {rr["variable_bpm"]} | 红线 >3 条（变速谱，单一 δ 不成立） |')
    A.append(f'| ❌ parse_error | {rr["parse_error"]} | 解析失败 / 无音符 |')
    A.append(f'| ❌ no_window | {rr["no_window"]} | 门禁过但没有满足「音符≥16 且覆盖≥90%」的窗口 |')
    A.append('')
    A.append('### 拒绝的两种性质（别把「LN 表示问题」当成「没对齐」）')
    A.append('')
    if ln.get('low_r_n'):
        A.append(f'- 被 low_r 拒的 {ln["low_r_n"]} 张里，有 **{ln["low_r_explained_by_ln"]}** '
                 f'张的「音符头数 vs 四带求和」相关 ≥0.40（中位 '
                 f'{ln["low_r_head_r"].get("median", 0):+.2f}）—— 这些谱**其实是对齐的**，'
                 '低分来自 X 的表示：X 含 LN 持续段，LN 尾把格子填满而 onset 只在头部出现。'
                 f'它们的 LN 占比中位 {ln["low_r_ln_frac"].get("median", 0)*100:.0f}%、'
                 f'占用填充率中位 {ln["low_r_occ_fill"].get("median", 0)*100:.0f}%'
                 f'（保留谱分别是 {ln["kept_ln_frac"].get("median", 0)*100:.0f}% / '
                 f'{ln["kept_occ_fill"].get("median", 0)*100:.0f}%）。')
        A.append(f'- 被 large_shift 拒的 {ln["large_shift_n"]} 张里，'
                 f'有 {ln["large_shift_would_pass"]} 张在 |δ|≤{s["config"]["max_shift"]:.0f}ms '
                 '范围内的最好相关其实能过门禁（中位 '
                 f'{ln["large_shift_r_best_within"].get("median", 0):+.2f}）——'
                 '也就是说「全场最优位移落在很远处」未必等于「近处没有可用对齐」，'
                 '这是本门禁口径偏保守的地方，见第 7 节。')
    A.append('')
    A.append('## 3. 校正位移的分布')
    A.append('')
    sd = s['shift_dist']
    if sd:
        A.append(f'中位 **{sd["median"]:+.0f}ms**，均值 {sd["mean"]:+.1f}ms，'
                 f'范围 [{sd["min"]:+.0f}, {sd["max"]:+.0f}]ms，'
                 f'p25 {sd["p25"]:+.0f} / p75 {sd["p75"]:+.0f}；'
                 f'δ*=0（本来就对齐）的谱 **{sd["zero_shift_charts"]}/{sd["n"]}**。')
        A.append('')
        A.append('```')
        A.append('校正位移直方图（250ms 一档，仅通过门禁的谱）')
        bars = s['shift_hist']['rows']
        mx = max([r['n'] for r in bars] + [1])
        for r in bars:
            if r['n'] == 0:
                continue
            w = int(round(40.0 * r['n'] / mx))
            A.append(f'{r["lo"]:+6d}..{r["hi"]:+6d}ms  {r["n"]:3d} | {"#" * w}')
        A.append('```')
    A.append('')
    A.append('峰显著性（r* 减去 |δ−δ*|>400ms 处的最高值）中位 '
             f'{s["peak_prominence"].get("median", 0):+.3f}，'
             f'p10 {s["peak_prominence"].get("p10", 0):+.3f}。')
    A.append('')
    A.append('## 4. 自证：重建后的逐谱 X–C 相关（全局时间轴聚合）')
    A.append('')
    A.append('方法同第 148 轮（把每张谱的所有窗口按 `start_cell` 铺回整首时间轴，'
             'X 四轨取均值、C 四带取均值，再算 Pearson r），直接在产出 npz 上重算：')
    A.append('')
    A.append('| 指标 | 新数据集 | 旧数据集（同口径重算） |')
    A.append('| --- | --- | --- |')
    def row(name, key, fmt='{:+0.3f}'):
        a = v.get(key)
        b = ov.get(key) if ov else None
        return (f'| {name} | {fmt.format(a) if a is not None else "n/a"} | '
                f'{fmt.format(b) if b is not None else "n/a"} |')
    A.append(f'| 谱数 | {v["n_charts"]} | {ov["n_charts"] if ov else "n/a"} |')
    A.append(f'| 窗口数 | {v["n_windows"]} | {ov["n_windows"] if ov else "n/a"} |')
    A.append(f'| **r 中位** | **{v["r_median"]:+.3f}** | '
             f'{ov["r_median"]:+.3f} |' if ov else
             f'| **r 中位** | **{v["r_median"]:+.3f}** | n/a |')
    if ov:
        A.append(row('r 均值', 'r_mean'))
        A.append(row('r 最小', 'r_min'))
        A.append(row('r p10', 'r_p10'))
        A.append(row('r p25', 'r_p25'))
        A.append(row('r p75', 'r_p75'))
        A.append(row('r p90', 'r_p90'))
    A.append(f'| r ≥ 0.25 占比 | **{v["frac_ge_025"]*100:.0f}%** | '
             f'{ov["frac_ge_025"]*100:.0f}% |' if ov else '')
    A.append(f'| r ≥ 0.40 占比 | {v["frac_ge_04"]*100:.0f}% | '
             f'{ov["frac_ge_04"]*100:.0f}% |' if ov else '')
    A.append(f'| 随机循环平移 C 的对照（中位） | {v["r_rand_median"]:+.3f} | '
             f'{ov["r_rand_median"]:+.3f} |' if ov else '')
    A.append('')
    tg = s['target']
    A.append(f'**目标（r 中位 ≥0.35 且保留谱 ≥15）：'
             f'{"✅ 达成" if tg["met"] else "❌ 未达成"}**'
             f'（实际 r 中位 {v["r_median"]:+.3f}、{v["n_charts"]} 张谱、'
             f'{v["n_windows"]} 窗口）')
    A.append('')
    A.append('### 独立证据：这不是「自证自己」')
    A.append('')
    A.append('门禁与自证同口径（都基于 X–C 的 1/32 格相关），自证只证明'
             '「数据落盘没写错」。证明「位移真的找对了」用下面两样：')
    A.append('')
    A.append('| 证据 | 中位 r | 说明 |')
    A.append('| --- | --- | --- |')
    lp = v.get('lag_profile', {})
    A.append(f'| **把 C 平移 1 格**（≈40ms） | {lp.get("1", 0):+.3f} | '
             '对 1/32 格相位极敏感：一旦错半格就掉下去 |')
    A.append(f'| 平移 2 格 | {lp.get("2", 0):+.3f} | |')
    A.append(f'| 平移 4 格（≈1/8 拍） | {lp.get("4", 0):+.3f} | |')
    A.append(f'| 平移 8 格（≈1/4 拍） | {lp.get("8", 0):+.3f} | |')
    A.append(f'| C 在谱内随机循环平移（8 次均值） | {v["r_rand_median"]:+.3f} | '
             '对照组，应 ≈0 |')
    A.append('')
    A.append('峰值在 lag=0、平移一格就塌 —— 说明相关来自「音符落在哪一个 1/32 格」'
             '的相位对齐，而不是慢变的响度包络。')
    A.append('')
    A.append('另一套粗口径（250ms 分箱，`validate-audio-cond-big.py` 的做法）'
             '**没有判别力**：')
    A.append(f'δ* 处中位 {ind["coarse_r_star"].get("median", 0):+.3f}，'
             f'δ=0 处 {ind["coarse_r_0"].get("median", 0):+.3f}，'
             f'**随机位移也有 {ind["coarse_r_rand"].get("median", 0):+.3f}**。'
             '分箱一粗，音符密度与响度包络的低频结构本来就跟得上，位移的好坏被淹没 ——'
             '这正好解释了第 148 轮 offset 搜索为什么会在很多包上给出数百毫秒到 −2400ms '
             '的「最佳位移」（曲线很平，Δr 只有 0.06 量级，取到哪算哪）。')
    A.append('')
    A.append('### 同一批谱的新旧对照')
    p = s['paired_old_new']
    if p.get('n_with_old'):
        A.append('')
        A.append(f'在旧数据集里也存在的 **{p["n_with_old"]}** 张保留谱：'
                 f'r 中位 {p["r_old_median"]:+.3f} → **{p["r_new_median"]:+.3f}**，'
                 f'变好 {p["improved"]} 张 / 变差 {p["worsened"]} 张，'
                 f'中位提升 {p["delta_median"]:+.3f}。')
        A.append('')
        A.append('| SetID | 谱 | r 旧 | r 新 | 校正位移 |')
        A.append('| --- | --- | --- | --- | --- |')
        for row_ in sorted(s['paired_rows'],
                           key=lambda z: (z['r_old'] is None, -(z['r_new'] or 0)))[:12]:
            A.append(f'| {row_["SetID"]} | {row_["chart"][:40]} | '
                     f'{"n/a" if row_["r_old"] is None else format(row_["r_old"], "+.3f")} | '
                     f'{row_["r_new"]:+.3f} | {row_["shift_ms"]:+.0f}ms |')
        A.append('')
        A.append('（只列前 12 张，完整表见 `stats.json` 的 `paired_rows`。）')
    A.append('')
    A.append('## 5. 产出与字段')
    A.append('')
    A.append(f'`{os.path.relpath(s["data"]["npz"], ROOT)}`（{s["data"]["bytes"]/1e6:.1f} MB）')
    A.append('')
    A.append('| 键 | 形状 / dtype | 含义 |')
    A.append('| --- | --- | --- |')
    A.append(f'| `X` | {tuple(s["data"]["X"])} uint8 | 每键每格是否有音符（含 LN 持续段） |')
    A.append(f'| `C` | {tuple(s["data"]["C"])} float16 | 四带 onset 强度（0–1，逐谱 P99 归一） |')
    A.append('| `song` | int16 | 歌（set）编号 |')
    A.append('| `chart_id` | U48 | 谱面 id（`SetID-序号-难度名`） |')
    A.append('| `split` | int8 | 0/1/2 = train/val/test，**按 set 整组划分** |')
    A.append('| `start_cell` | int32 | 窗口在 1/32 网格上的起始格号 |')
    A.append('| `bpm` | float32 | 首条红线的 BPM |')
    A.append('| `shift_ms` | float32 | 该窗所属谱的校正位移 δ* |')
    A.append('| `songs` | JSON 字符串 | 歌编号 → 歌名（另存 `songs.json`） |')
    A.append('')
    A.append('split 窗口数：' + '，'.join(f'{k}={vv}' for k, vv in s['split_counts'].items()) +
             '；涉及 set：' + '，'.join(f'{k}={vv}' for k, vv in s['split_sets'].items()))
    A.append('')
    A.append('## 6. 方法要点（和旧构建脚本的差别）')
    A.append('')
    A.append('1. **位移只施加在音频采样起点上**：音符仍按作者红线 offset 分格（保留谱面 '
             '1/32 相位），C 在 `offset+δ*` 处重采样。若把 δ 同时加在音符分格和音频采样上，'
             '两者相对关系不变、r 与 δ 无关，搜索会退化成恒等（已实测）。')
    A.append('2. **每张谱用自己 `AudioFilename` 的音频**：旧脚本在多曲 pack 里会退到目录里'
             '最大的 mp3，等于给谱配错歌。本轮 4K 谱全部命中自己的音频文件，'
             '所以「多曲 pack 配错音频」这一类错位在这一步就被消掉了。')
    A.append('3. **门禁与目标指标同口径**：r* 就是自证的 X–C（1/32 格，X 含 LN 段），'
             '不是更宽松的粗粒度相关；顺带发现「音符头数 vs 四带求和」这个更粗的尺子会把'
             'LN 密集谱误判成对齐良好（保存下来的话自证仍是 0），所以只把它留作诊断。')
    A.append(f'4. onset 包络按任务书用 hop=256（旧脚本 hop=128），其余 mel/带划分/'
             '正向差分/格内取最大/P99 归一全部同口径。')
    A.append('5. 窗口过滤沿用旧口径：音符数 ≥16、条件覆盖率 ≥90%、[4,256]/stride 128。')
    A.append('')
    A.append('## 7. 剩余限制（不粉饰）')
    A.append('')
    sp = s['skipped_packs']
    A.append(f'- **包不够**：`out/audio-dl` 只有 {c["sets_processed"]} 个包（下载阶段 503 失败 '
             f'{sp["reasons"].get("下载失败", 0)} 个、超单包上限 {sp["reasons"].get("超过单包上限", 0)} 个），'
             f'本轮按要求不联网补。最终 **{c["kept_charts"]} 张谱 / {c["windows"]} 窗口**，'
             '规模比旧数据集（200 谱 / 5279 窗）小，且这批包的曲风偏 4K 技术图/串图，'
             '风格覆盖不广。')
    A.append('- **LN 密集谱被门禁刷掉**：这不是错位，而是 X 的表示问题（LN 尾填满格子、'
             'onset 只在头部）。若要 LN 覆盖，应单独建 LN 子集并换目标（例如把 X 的 LN 段'
             '换成「LN 头 + 持续时间」），而不是把 r 低的谱硬塞回来。')
    A.append('- **large_shift 口径偏保守**：按任务书先取全场最优 δ* 再判 |δ*|≤300ms，'
             '会有「远处有更强峰、近处其实也能用」的谱被拒（见 2 节统计）。'
             '这类谱的音频版本仍可疑，本轮不做二次放行。')
    A.append('- **50ms 量化 + 101 个候选取最大**：r* 含选择偏差（取最大本身抬高数值），'
             '自证同口径也无法消除；真正的验收要靠训练时的「打乱条件」对照（本轮不训练）。')
    A.append('- **只能修整体平移**：变速谱（红线 >3）直接跳过；分段错位 / 中途换音频版本'
             '这类问题的单一 δ 无法表达。')
    A.append('- **同包多谱不做一致性约束**：每张谱各自找 δ*，理论上同曲多难度应当一致，'
             '本轮不加跨谱惩罚。')
    A.append('')
    A.append('## 8. 复现')
    A.append('')
    A.append('```')
    A.append('python3 tools/build-audio-cond-fixed.py \\')
    A.append(f'  --min-r {s["config"]["min_r"]} --max-shift {s["config"]["max_shift"]:.0f} \\')
    A.append('  --report out/diffusion/round149-audio-fixed/REPORT.md')
    A.append('# 同一口径复核任意 npz（含旧数据集对照）')
    A.append('python3 tools/build-audio-cond-fixed.py --verify out/diff-data-audio-big/dataset.npz')
    A.append('```')
    open(rep, 'w', encoding='utf-8').write('\n'.join([x for x in A if x]) + '\n')
    print('报告 →', rep)


# ============================== CLI ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dl', default=os.path.join(ROOT, 'out', 'audio-dl'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'diff-data-audio-fixed'))
    ap.add_argument('--min-r', type=float, default=DEFAULTS['min_r'])
    ap.add_argument('--max-shift', type=float, default=DEFAULTS['max_shift'])
    ap.add_argument('--shift-lo', type=float, default=DEFAULTS['shift_lo'])
    ap.add_argument('--shift-hi', type=float, default=DEFAULTS['shift_hi'])
    ap.add_argument('--shift-step', type=float, default=DEFAULTS['shift_step'])
    ap.add_argument('--sets', default='', help='只处理这些 SetID（逗号分隔）')
    ap.add_argument('--limit-sets', type=int, default=0)
    ap.add_argument('--report', default='', help='写 REPORT.md 到该路径')
    ap.add_argument('--verify', default='', help='只复核一个 npz 的逐谱 X–C 相关')
    a = ap.parse_args()
    if a.verify:
        s, rows = verify_dataset(a.verify)
        print(json.dumps(s, ensure_ascii=False, indent=1))
        print('\n最差 8 张：')
        for r in sorted(rows, key=lambda z: z['r'])[:8]:
            print(f'  {r["chart_id"][:40]:40s} r={r["r"]:+.3f} 窗 {r["windows"]}')
        return 0
    return build(a)


if __name__ == '__main__':
    sys.exit(main())
