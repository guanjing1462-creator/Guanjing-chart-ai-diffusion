#!/usr/bin/env python3
"""验证音频条件数据集：音符密度 vs 音频强度相关性 + 随机平移对照。

口径同 /tmp/dens_corr.py（250ms 分箱、onset_strength hop=256、z 标准化后 Pearson r），
另外加三样对照：
  1) 随机平移：把谱面音符时间在整首歌内随机循环平移，重算 r（预期 ≈0）
  2) 网格落点自检：音符头相对首红线 offset 的相位，看有多少落在 1/32 格线上
  3) 用已构建的 C 特征（1/32 格）聚到 250ms 再算 r，与原始 onset 口径互证
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 22050
HOP_V = 256
BIN = 0.25


def env_250ms(y):
    """onset_strength → 250ms 分箱求和 → z 标准化。"""
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_V)
    fps = SR / HOP_V
    nb = int(len(env) / fps / BIN) + 1
    grid = np.arange(nb + 1) * BIN
    out = np.array([env[int(a * fps):max(int(a * fps) + 1, int(b * fps))].sum()
                    for a, b in zip(grid, grid[1:])], dtype=float)
    return out, grid


def dens_corr(note_times_ms, audio, grid):
    h, _ = np.histogram(note_times_ms / 1000.0, bins=grid)
    h = h.astype(float)
    a = audio[:len(h)]
    if h.std() < 1e-9 or a.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(h, a)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dl', default=os.path.join(ROOT, 'out', 'audio-dl'))
    ap.add_argument('--data', default=os.path.join(ROOT, 'out', 'diff-data-audio-big'))
    ap.add_argument('--n-songs', type=int, default=14)
    ap.add_argument('--n-shift', type=int, default=20)
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'diffusion',
                                                  'round146-audio-big', 'validation.json'))
    a = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'bacb', os.path.join(ROOT, 'tools', 'build-audio-cond-big.py'))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)

    man = json.load(open(os.path.join(a.data, 'manifest.json'), encoding='utf-8'))
    # 每 set 只取一张谱，挑窗口最多的，保证跨 set 分布
    best = {}
    for m in man:
        k = m['SetID']
        if k not in best or m['windows'] > best[k]['windows']:
            best[k] = m
    picks = sorted(best.values(), key=lambda m: -m['windows'])[:a.n_songs]
    print(f'抽样 {len(picks)} 首（每 set 一张谱，按窗口数降序）', flush=True)

    d = np.load(os.path.join(a.data, 'dataset.npz'))
    C, cid_arr = d['C'], d['chart_id']
    rows = []
    import librosa
    for i, m in enumerate(picks):
        sid, chart = m['SetID'], m['chart']
        dpath = os.path.join(a.dl, str(sid), chart)
        npath = os.path.join(a.dl, str(sid), m['audio'])
        if not (os.path.exists(dpath) and os.path.exists(npath)):
            print(f'  [{i+1}] 缺文件 {sid}/{chart}', flush=True)
            continue
        ts, offset_ms, beat_len = B.note_times(dpath)
        if len(ts) < 32:
            continue
        y, _ = librosa.load(npath, sr=SR, mono=True)
        audio, grid = env_250ms(y)
        dur = len(y) / SR
        r = dens_corr(ts, audio, grid)
        # 随机平移对照：循环平移整条谱
        rng = np.random.default_rng(1000 + i)
        shifts = rng.uniform(0, dur, a.n_shift)
        rs = []
        for s in shifts:
            rs.append(dens_corr(np.mod(ts + s * 1000.0, dur * 1000.0), audio, grid))
        r_shift = float(np.mean(rs))
        # 网格落点自检：音符头在 1/32 格上的相位
        step = beat_len / 8.0
        ph = np.mod(ts - offset_ms, step) / step
        on_grid = float(np.mean((ph < 0.10) | (ph > 0.90)))
        # C 特征口径互证：把该谱窗口的 C 四通道求平均聚到 250ms
        sel = cid_arr == m['chart_id']
        c_ok = 'n/a'
        if sel.sum() > 0:
            # 把窗口 C 重建回整首时间轴（同一格可能被多个窗口覆盖 → 取均值）
            Cs_ = C[sel].astype(np.float32)
            st_ = d['start_cell'][sel]
            ncell = int(st_.max()) + Cs_.shape[1] + 1
            acc0 = np.zeros(ncell)
            cnt0 = np.zeros(ncell)
            for wi in range(len(Cs_)):
                a0 = int(st_[wi])
                acc0[a0:a0 + Cs_.shape[1]] += Cs_[wi].mean(axis=1)   # 4 通道均值
                cnt0[a0:a0 + Cs_.shape[1]] += 1.0
            cc = acc0 / np.maximum(cnt0, 1.0)
            step_s = step / 1000.0
            t_c = offset_ms / 1000.0 + np.arange(len(cc)) * step_s
            nb = len(audio)
            acc = np.zeros(nb)
            idx = np.floor(t_c / BIN).astype(int)
            msk = (idx >= 0) & (idx < nb)
            np.add.at(acc, idx[msk], cc[msk])
            h, _ = np.histogram(ts / 1000.0, bins=grid)
            if h.astype(float).std() > 1e-9 and acc[:len(h)].std() > 1e-9:
                c_ok = round(float(np.corrcoef(h.astype(float), acc[:len(h)])[0, 1]), 3)
        rows.append({'SetID': sid, 'chart': chart, 'title': m['title'][:40],
                     'bpm': m['bpm'], 'notes': int(m.get('notes', 0)),
                     'dur_s': round(dur, 1), 'r': round(r, 4),
                     'r_shift': round(r_shift, 4),
                     'on_grid': round(on_grid, 3), 'r_C250': c_ok,
                     'windows': int(sel.sum())})
        print(f'  [{i+1}/{len(picks)}] {sid} {m["title"][:28]:28s} '
              f'r={r:+.3f}  平移基线={r_shift:+.3f}  格点={on_grid:.2f}  '
              f'C口径={c_ok}  窗口={int(sel.sum())}', flush=True)

    if not rows:
        print('抽样失败', file=sys.stderr)
        return 1
    rs = np.array([x['r'] for x in rows])
    rb = np.array([x['r_shift'] for x in rows])
    og = np.array([x['on_grid'] for x in rows])
    summary = {
        'n_songs': len(rows), 'bin_ms': 250, 'n_shift': a.n_shift,
        'r_mean': round(float(rs.mean()), 4),
        'r_median': round(float(np.median(rs)), 4),
        'r_min': round(float(rs.min()), 4),
        'r_max': round(float(rs.max()), 4),
        'r_shift_mean': round(float(rb.mean()), 4),
        'r_shift_median': round(float(np.median(rb)), 4),
        'r_shift_abs_mean': round(float(np.abs(rb).mean()), 4),
        'on_grid_median': round(float(np.median(og)), 3),
        'on_grid_min': round(float(og.min()), 3),
        'sample_rows': rows,
    }
    if summary['r_median'] < 0.20:
        summary['warning'] = '数据对齐可能有问题（中位相关 < 0.20），需检查网格起点/1-32换算/音频 offset'
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n=== 汇总 ===')
    print(f'抽样 {len(rows)} 首  r 均值 {rs.mean():+.3f}  中位 {np.median(rs):+.3f}  '
          f'最小 {rs.min():+.3f}  最大 {rs.max():+.3f}')
    print(f'随机平移基线 均值 {rb.mean():+.3f}  中位 {np.median(rb):+.3f}  '
          f'|r| 均值 {np.abs(rb).mean():+.3f}')
    print(f'1/32 格点落点率 中位 {np.median(og):.3f}  最小 {og.min():.3f}')
    if summary.get('warning'):
        print('⚠ ' + summary['warning'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
