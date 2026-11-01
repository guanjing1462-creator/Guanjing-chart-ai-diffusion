#!/usr/bin/env python3
"""构建 **多帧 mel 频谱** 音频条件数据集（round 151）。

相对 `out/diff-data-audio-fixed`（4 带 onset 包络 C[256,4]）只升级**条件**：

    M[N,256,32]  ←  每张谱自己的音频 → log1p(melspectrogram(n_mels=32,
                    n_fft=1024, hop=256, sr=22050)) → **正向时间差分（onset 通量）**
                    → 1/32 格内取均值（或最大）→ 每首歌逐通道 P99 归一到 0–1

  ⚠ `mel 能量 ≠ mel onset 通量`：直接用 log1p mel 能量，逐谱 X–C 相关中位只有
    0.287（4 带 onset 是 0.498，114 张谱里只有 5 张更优）——能量跟的是持续响度
    （贝斯、延音），而谱面密度跟的是**起音**。所以默认 `--feature flux`
    （正差分），实测中位 0.60 左右，反而超过 4 带。

X / song / chart_id / split / start_cell / bpm / shift_ms **原样复制**（行序不变），
所以 M 与 X 行行对齐，不会出现错配。

网格对齐（上一轮的坑）
----------------------
条件网格的起点必须是 `offset_ms + shift_ms`（manifest 里该谱的校正位移 δ*），
不是作者红线 offset —— 否则整张谱的 mel 相对音符平移 δ*，相位信息全废
（4 带版实测：C 平移 1 格相关就从 +0.498 掉到 −0.154）。

mel 帧时刻：melspectrogram 用 librosa 默认 center=True，第 j 帧的中心在
j*hop 采样处 → tf[j] = j*HOP/SR*1000 ms。格 k 覆盖 [q+k*step, q+(k+1)*step)，
取格内帧的**均值**（任务书要求；4 带版是取 max）。

自证（写进 stats.json / REPORT）
-------------------------------
1) 每张保留谱：全局时间轴聚合的 **音符占用 vs mel 能量（32 维求和）** Pearson r
   → 报中位 / 均值 / r≥0.25 比例；同一段代码跑 4 带的 C（四带取均值）作对照；
2) 相位敏感性：把 mel 能量沿时间轴平移 ±1 格再算 r（应明显下降）。

用法
----
  python3 tools/build-mel-cond.py                       # 构建 + 自证
  python3 tools/build-mel-cond.py --limit-charts 6      # 小样试跑
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- mel 参数（任务书指定）----
SR = 22050
MEL_N_FFT = 1024
MEL_HOP = 256
N_MELS = 32

WIN = 256                      # 与 4 带版一致
LAGS = (-2, -1, 1, 2)          # 相位敏感性检查的格位移


def load_module(rel, name):
    p = os.path.join(ROOT, rel)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BFX = load_module('tools/build-audio-cond-fixed.py', 'bfx')


# ============================== 音频 → mel ==============================
class MelCache:
    """按音频路径缓存 (mel[32,F] float32, 帧中心时刻 ms[F], 时长 ms)。"""

    def __init__(self, max_items=64):
        self.cache = OrderedDict()
        self.max_items = max_items
        self.hits = self.misses = self.fail = 0

    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            self.hits += 1
            return self.cache[path]
        self.misses += 1
        item = self._compute(path)
        if item is None:
            self.fail += 1
            return None
        self.cache[path] = item
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return item

    def _compute(self, path):
        import librosa
        y = BFX.load_mono(path)
        if y is None:
            return None
        # center=True（librosa 默认）：第 j 帧中心 = j*HOP 采样
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=MEL_N_FFT,
                                           hop_length=MEL_HOP, n_mels=N_MELS,
                                           fmin=0.0, fmax=8000.0, center=True)
        L = np.log1p(S).astype(np.float32)               # [32, F]
        F = L.shape[1]
        tf = (np.arange(F, dtype=np.float64) * MEL_HOP) / SR * 1000.0
        dur_ms = len(y) / SR * 1000.0
        del S
        return L, tf, dur_ms


def onset_flux(L):
    """log1p mel → 正向时间差分（onset 通量）。首帧补 0 保持帧对齐。"""
    return np.maximum(np.diff(L, axis=1, prepend=L[:, :1]), 0.0).astype(np.float32)


def cell_mean(mel, tf, q_ms, step_ms, n, mode='mean'):
    """格内均值/最大：[n,32]，格 k = [q+k*step, q+(k+1)*step)。"""
    n = int(n)
    out = np.zeros((n, N_MELS), dtype=np.float32)
    F = mel.shape[1]
    if n <= 0 or F == 0:
        return out
    t0 = q_ms + np.arange(n, dtype=np.float64) * step_ms
    i0 = np.searchsorted(tf, t0, side='left').astype(np.int64)
    i1 = np.searchsorted(tf, t0 + step_ms, side='left').astype(np.int64)
    i0 = np.clip(i0, 0, F)
    i1 = np.clip(i1, 0, F)
    ok = i1 > i0
    if not ok.any():
        return out
    if mode == 'max':
        for k in np.flatnonzero(ok):
            out[k] = mel[:, i0[k]:i1[k]].max(axis=1)
        return out
    cs = np.concatenate([np.zeros((N_MELS, 1), np.float64),
                         np.cumsum(mel.astype(np.float64), axis=1)], axis=1)
    seg = (cs[:, i1[ok]] - cs[:, i0[ok]]) / (i1[ok] - i0[ok])[None, :]
    out[ok] = seg.T.astype(np.float32)
    return out


# ============================== 自证 ==============================
def timeline(chart_rows, arr, win=WIN):
    """把一张谱所有窗口的 [win, ch] 条件按 start_cell 铺回整首时间轴（取覆盖均值）。"""
    st = np.array([r[0] for r in chart_rows], dtype=np.int64)
    base = int(st.min())
    end = int((st + win).max())
    T = end - base
    ch = arr.shape[2] if arr.ndim == 3 else 1
    acc = np.zeros((T, ch), np.float64)
    cnt = np.zeros(T, np.float64)
    for (s, idx) in chart_rows:
        a = int(s - base)
        acc[a:a + win] += arr[idx]
        cnt[a:a + win] += 1.0
    m = cnt > 0
    return acc[m] / cnt[m, None]


def occupancy_timeline(chart_rows, X, win=WIN):
    """音符占用率（4 键均值）铺回整首时间轴。"""
    occ = X.astype(np.float32).mean(axis=1)[..., None]
    return timeline(chart_rows, occ).ravel()


def verify(M, X, song, cid, start_cell, C=None, verbose=True, only=None):
    """逐谱：占用 vs mel 能量（32 求和）在全局时间轴上的相关 + ±1 格相位。"""
    rows = []
    cs_all = [c for c in np.unique(cid) if only is None or c in only]
    for c in cs_all:
        sel = np.flatnonzero(cid == c)
        cr = [(int(start_cell[i]), int(i)) for i in sel]
        occ = occupancy_timeline(cr, X)
        if len(occ) < 20:
            continue
        ml = timeline(cr, M.astype(np.float32))
        e_mel = ml.sum(axis=1)
        r_mel = BFX.pearson(occ, e_mel)
        lag = {}
        for k in LAGS:
            lag[str(k)] = round(BFX.pearson(occ, np.roll(e_mel, k)), 4)
        rec = dict(chart_id=str(c), windows=int(len(sel)), cells=int(len(occ)),
                   r_mel=round(r_mel, 4), lag_mel=lag,
                   mel_energy_mean=round(float(e_mel.mean()), 4))
        if C is not None:
            cb = timeline(cr, C.astype(np.float32)).mean(axis=1)
            rec['r_band4'] = round(BFX.pearson(occ, cb), 4)
            rec['lag_band4'] = {str(k): round(BFX.pearson(occ, np.roll(cb, k)), 4)
                                for k in LAGS}
        rows.append(rec)
    rs = np.array([x['r_mel'] for x in rows]) if rows else np.zeros(1)
    out = dict(n_charts=int(len(rows)),
               r_mel_median=round(float(np.median(rs)), 4),
               r_mel_mean=round(float(rs.mean()), 4),
               r_mel_p10=round(float(np.percentile(rs, 10)), 4),
               r_mel_min=round(float(rs.min()), 4),
               r_mel_max=round(float(rs.max()), 4),
               frac_ge_025=round(float((rs >= 0.25).mean()), 4),
               frac_ge_040=round(float((rs >= 0.40).mean()), 4),
               lag_median={str(k): round(float(np.median([x['lag_mel'][str(k)]
                                                         for x in rows])), 4)
                           for k in LAGS} if rows else {})
    if C is not None and rows:
        rb = np.array([x['r_band4'] for x in rows])
        out.update(r_band4_median=round(float(np.median(rb)), 4),
                   r_band4_mean=round(float(rb.mean()), 4),
                   frac_ge_025_band4=round(float((rb >= 0.25).mean()), 4),
                   lag_median_band4={str(k): round(float(np.median(
                       [x['lag_band4'][str(k)] for x in rows])), 4)
                       for k in LAGS})
    if verbose:
        print('\n=== 自证：音符占用 vs 音频条件（全局时间轴聚合）===')
        print(f"谱数 {out['n_charts']}；mel 能量(32 求和) r 中位 "
              f"{out['r_mel_median']:+.3f}，均值 {out['r_mel_mean']:+.3f}，"
              f"r≥0.25 占 {out['frac_ge_025']*100:.0f}%")
        if C is not None:
            print(f"对照：4 带 C(四带均值) r 中位 {out['r_band4_median']:+.3f}，"
                  f"r≥0.25 占 {out['frac_ge_025_band4']*100:.0f}%")
        print('相位敏感性（mel 能量整体平移 k 格后的 r 中位）：'
              + '  '.join(f'{k}格 {v:+.3f}'
                          for k, v in out['lag_median'].items()))
    return out, rows


# ============================== 主构建 ==============================
def resolve_audio(set_dir, declared):
    files = set(os.listdir(set_dir))
    p = os.path.join(set_dir, os.path.basename(declared.replace('\\', '/')))
    if os.path.exists(p):
        return p
    low = {f.lower(): f for f in files}
    b = os.path.basename(declared.replace('\\', '/')).lower()
    if b in low:
        return os.path.join(set_dir, low[b])
    auds = [f for f in files if f.lower().endswith(('.mp3', '.ogg', '.wav'))]
    return os.path.join(set_dir, auds[0]) if len(auds) == 1 else None


def build(a):
    src = os.path.abspath(a.src)
    dl = os.path.abspath(a.dl)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    d = np.load(src, allow_pickle=True)
    for k in ('X', 'C', 'song', 'chart_id', 'split', 'start_cell', 'bpm', 'shift_ms'):
        if k not in d:
            raise SystemExit(f'[错误] {src} 缺键 {k!r}')
    X = d['X']
    C = d['C']
    song = d['song']
    cid = d['chart_id'].astype(str)
    split = d['split']
    start_cell = d['start_cell']
    bpm = d['bpm']
    shift_ms = d['shift_ms']
    songs_json = str(d['songs']) if 'songs' in d else '{}'
    N, _, W = X.shape
    print(f'源数据 {src}：{N} 窗 × {X.shape[1]} 键 × {W} 格；'
          f'{len(np.unique(cid))} 谱 / {len(np.unique(song))} 歌', flush=True)

    man = json.load(open(os.path.join(os.path.dirname(src), 'manifest.json'),
                         encoding='utf-8'))
    by_cid = {m['chart_id']: m for m in man if m.get('verdict') == 'kept'}
    miss = [c for c in np.unique(cid) if c not in by_cid]
    if miss:
        raise SystemExit(f'[错误] manifest 里找不到这些 chart_id：{miss[:5]}')

    # ---- 逐谱算格内 mel 均值（按 song 分组，后面逐歌 P99 归一）----
    cache = MelCache()
    raw_by_cid = {}
    valid_by_cid = {}
    t_all = time.time()
    charts = list(np.unique(cid))
    if a.limit_charts:
        charts = charts[:a.limit_charts]
    for ci, c in enumerate(charts):
        m = by_cid[c]
        set_dir = os.path.join(dl, str(m['SetID']))
        ap = resolve_audio(set_dir, m.get('audio') or '')
        if ap is None:
            raise SystemExit(f'[错误] 找不到音频：{set_dir} / {m.get("audio")}')
        item = cache.get(ap)
        if item is None:
            raise SystemExit(f'[错误] 解码失败：{ap}')
        L, tf, dur_ms = item
        mel = onset_flux(L) if a.feature == 'flux' else L
        q = float(m['offset_ms']) + float(m['shift_ms'])     # ★ 校正后的网格起点
        step = float(m['step_ms'])
        n_cells = int(m['n_cells'])
        raw = cell_mean(mel, tf, q, step, n_cells, a.agg)    # [n_cells,32]
        t0 = q + np.arange(n_cells, dtype=np.float64) * step
        lim = min(float(dur_ms), float(tf[-1]))
        valid = (t0 >= 0.0) & (t0 < lim)
        raw_by_cid[c] = raw
        valid_by_cid[c] = valid
        if (ci + 1) % 10 == 0 or ci == len(charts) - 1:
            print(f'  [{ci+1}/{len(charts)}] mel 格内均值已算 '
                  f'{time.time()-t_all:.0f}s（音频命中 {cache.hits} / 解码 '
                  f'{cache.misses}）', flush=True)

    # ---- 逐歌逐通道 P99 归一 ----
    song_of_cid = {}
    for c in charts:
        song_of_cid[c] = int(song[np.flatnonzero(cid == c)[0]])
    scale = {}
    for s in sorted(set(song_of_cid.values())):
        cs = [c for c in charts if song_of_cid[c] == s]
        cat = np.concatenate([raw_by_cid[c][valid_by_cid[c]] for c in cs], axis=0)
        sc = np.ones(N_MELS, dtype=np.float64)
        if len(cat):
            p = np.percentile(cat, 99, axis=0)
            sc = np.where(p > 1e-9, p, 1.0)
        scale[s] = sc
    print(f'P99 归一：{len(scale)} 首歌各自的 32 通道尺度已算好', flush=True)

    M = np.zeros((N, W, N_MELS), dtype=np.float16)
    for c in charts:
        sel = np.flatnonzero(cid == c)
        raw = raw_by_cid[c]
        sc = scale[song_of_cid[c]]
        norm = np.clip(raw / sc[None, :], 0.0, 1.0).astype(np.float32)
        for i in sel:
            s0 = int(start_cell[i])
            M[i] = norm[s0:s0 + W].astype(np.float16)

    # ---- 自证 ----
    self_sum, self_rows = verify(M, X, song, cid, start_cell, C=C,
                                 only=set(charts))
    if a.limit_charts:
        self_sum['note'] = f'仅前 {a.limit_charts} 张谱（试跑）'

    # ---- 落盘（X 与全部元数据原样带过去）----
    out_npz = os.path.join(out, 'dataset.npz')
    np.savez_compressed(out_npz, M=M, X=X, song=song, chart_id=d['chart_id'],
                        split=split, start_cell=start_cell, bpm=bpm,
                        shift_ms=shift_ms, songs=np.array(songs_json))
    stats = dict(
        source=os.path.relpath(src, ROOT), data=os.path.relpath(out_npz, ROOT),
        windows=int(N), charts=int(len(np.unique(cid))),
        songs=int(len(np.unique(song))),
        split_counts={str(int(s)): int((split == s).sum())
                      for s in np.unique(split)},
        M_shape=list(M.shape), M_dtype='float16',
        X_shape=list(X.shape), X_dtype=str(X.dtype),
        mel=dict(n_mels=N_MELS, n_fft=MEL_N_FFT, hop=MEL_HOP, sr=SR,
                 fmin=0.0, fmax=8000.0, center=True, feature=a.feature,
                 agg='cell-' + a.agg,
                 norm='per-song-per-channel-P99', clip=[0.0, 1.0]),
        grid_origin='offset_ms + shift_ms（manifest 的校正位移）',
        M_stats=dict(mean=round(float(M.astype(np.float32).mean()), 5),
                     frac_gt0=round(float((M > 0).mean()), 4),
                     max=round(float(M.max()), 4),
                     channel_mean=[round(float(x), 4) for x in
                                   M.astype(np.float32).mean(axis=(0, 1))]),
        selfcheck=self_sum,
        audio_hits=int(cache.hits), audio_decodes=int(cache.misses),
        seconds=round(time.time() - t_all, 1))
    json.dump(stats, open(os.path.join(out, 'stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump(self_rows, open(os.path.join(out, 'selfcheck-per-chart.json'), 'w',
                              encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n写出 {out_npz}（{os.path.getsize(out_npz)/1e6:.2f}MB）'
          f'，用时 {time.time()-t_all:.0f}s', flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='out/diff-data-audio-fixed/dataset.npz')
    ap.add_argument('--dl', default='out/audio-dl')
    ap.add_argument('--out', default='out/diff-data-mel')
    ap.add_argument('--limit-charts', type=int, default=0, help='调试用')
    ap.add_argument('--feature', choices=['flux', 'energy'], default='flux',
                    help='flux=log1p mel 的正向时间差分（onset，默认，实测更贴谱）'
                         ' / energy=log1p mel 本身')
    ap.add_argument('--agg', choices=['mean', 'max'], default='mean',
                    help='1/32 格内聚合方式')
    a = ap.parse_args()
    return build(a)


if __name__ == '__main__':
    sys.exit(main())
