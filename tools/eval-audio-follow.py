#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采音（audio-following）验收：模型生成的谱面密度是否跟着条件 C 走。

回答的问题不是「val 损失谁低」，而是：
  **给模型一段音频特征 C，它生成的音符在时间上是否与该 C 对齐？**

协议
----
1. 从数据集指定 split（默认 1=val；2=test 作独立复现）抽 `--num` 个窗口，
   按占位率分层、每歌最多 `--per-song` 个，保留 (song, start_cell, bpm, 人类 X, 条件 C)。
2. 六组采样，**同一套初始噪声 seed**（配对设计，只差条件）：
   - `A-无条件`  ：模型 A（cond_dim=0），难度 = 本窗人类占位率归一化
   - `B-配对`    ：模型 B，条件 = 本窗 C
   - `B-错配`    ：模型 B，条件 = 另一窗（打乱 song/start_cell）的 C 与难度
   - `B-零条件`  ：模型 B，条件全零（同权重、同噪声，隔离「用没用条件」）
   - `B-条件+1格/-1格`：模型 B，条件 = 本窗 C 整体平移 ±1 格（1 格≈40ms）
     → 真学到相位的话指标应明显退化；把 C 平移一格还好 = 只学到密度先验
3. 每个生成窗口算两个指标（人类 X 同口径，作参照）：
   ① 分箱后 **音符密度 vs C 强度** 的 Pearson r（250ms / 1000ms；主用「总」带，四带都算）
   ② **1/32 格踩格率**：音符落在 1/32 细分格（(start_cell+k) 为奇数）且该格
      C总 > 窗内中位 的比例（含保奇偶性平移对照）
4. 额外：`scan_cells` 给出把 C 平移 k 格后的 r（相位敏感度曲线）；
   条件因果效应用固定噪声下「换条件」的输出差衡量。

DDPM 采样公式复用 tools/sample-diffusion32c.py 的 prediction_to_x0 + 后验均值，
本文件只是把 `--diff` 从标量扩成逐窗口向量、`--cond` 扩成逐窗口。
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = 256
BANDS = ['低', '中', '高', '总']
CELL_SHIFTS = (-8, -4, -2, -1, 0, 1, 2, 4, 8)


def load_sampler():
    """按路径导入采样器模块（文件名带连字符，不能直接 import）。"""
    p = os.path.join(ROOT, 'tools', 'sample-diffusion32c.py')
    spec = importlib.util.spec_from_file_location('smp32c', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@torch.no_grad()
def sample_windows(model, diffmod, cond, diffv, pred, seed, prediction_to_x0):
    """与 sampler 的 ddpm_sample 同一套公式，但允许逐窗口 diff / 逐窗口 cond。"""
    n = len(diffv)
    torch.manual_seed(seed)
    x = torch.randn(n, 4, WIN)
    sty = torch.zeros(n, dtype=torch.long)
    dv = torch.as_tensor(diffv, dtype=torch.float32)
    if cond is not None:
        cond = torch.as_tensor(np.ascontiguousarray(cond), dtype=torch.float32)
        assert cond.shape == (n, WIN, 4), cond.shape
    T = diffmod.T
    for t in reversed(range(T)):
        tb = torch.full((n,), t, dtype=torch.long)
        out = model(x, tb, sty, dv, cond)
        a_t, ab = diffmod.alphas[t], diffmod.abar[t]
        x0 = prediction_to_x0(x, out, ab, pred)
        if pred == 'x0':
            x0 = x0.clamp(-1.0, 2.0)
        if t > 0:
            abp = diffmod.abar[t - 1]
            mean = (abp.sqrt() * diffmod.betas[t] * x0
                    + a_t.sqrt() * (1 - abp) * x) / (1 - ab)
            sigma = (diffmod.betas[t] * (1 - abp) / (1 - ab)).sqrt()
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean
    return x.numpy()


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def binned(roll, C, cells_per_bin):
    """把 [4,256] 占用卷与 [256,4] 条件按 cells_per_bin 格分箱求和。"""
    nb = WIN // cells_per_bin
    m = nb * cells_per_bin
    dens = roll[:, :m].sum(axis=0).reshape(-1, cells_per_bin).sum(axis=1)
    cb = C[:m].reshape(nb, cells_per_bin, 4).sum(axis=1)
    return dens, cb


def _count_odd_hi(roll, C, start_cell):
    med = float(np.median(C[:, 3]))
    cells = np.arange(WIN) + int(start_cell)
    odd = (cells % 2) == 1          # 1/32 细分格（相对首红线的全局格号）
    high = C[:, 3] > med
    n_odd = n_odd_hi = n_all = n_all_hi = 0
    for k in range(WIN):
        cnt = int(roll[:, k].sum())
        if cnt == 0:
            continue
        n_all += cnt
        if high[k]:
            n_all_hi += cnt
        if odd[k]:
            n_odd += cnt
            if high[k]:
                n_odd_hi += cnt
    return n_odd, n_odd_hi, n_all, n_all_hi


def metrics(roll_bin, C, bpm, start_cell):
    """单窗口指标。roll_bin [4,256] 0/1，C [256,4]（评估所对的音频条件）。"""
    step_ms = 60000.0 / max(1e-6, float(bpm)) / 8.0
    out = {'step_ms': round(step_ms, 3), '占用': float(roll_bin.mean()),
           '音符数': int(roll_bin.sum())}
    for tag, bin_ms in (('250', 250.0), ('1000', 1000.0)):
        cpb = max(1, int(round(bin_ms / step_ms)))
        dens, cb = binned(roll_bin.astype(np.float64), C, cpb)
        out[f'cpb{tag}ms'] = cpb
        out[f'r{tag}'] = {}
        for j in range(4):
            r = pearson(dens, cb[:, j])
            out[f'r{tag}'][BANDS[j]] = round(r, 4) if np.isfinite(r) else None
        out[f'dens_cv{tag}'] = round(float(dens.std() / max(1e-9, dens.mean())), 4)
    # 相位敏感度：把 C 平移 k 格后再算 r250（k=0 为真对齐）
    cpb250 = max(1, int(round(250.0 / step_ms)))
    scan = {}
    for k in CELL_SHIFTS:
        dens, cb = binned(roll_bin.astype(np.float64), np.roll(C, k, axis=0), cpb250)
        r = pearson(dens, cb[:, 3])
        scan[str(k)] = round(r, 4) if np.isfinite(r) else None
    out['scan_cells'] = scan
    no, nohi, na, nahi = _count_odd_hi(roll_bin, C, start_cell)
    # 平移对照：同一批音符 + 保奇偶性（偶数格）位移后的 C
    _, nohi_s, _, _ = _count_odd_hi(roll_bin, np.roll(C, 64, axis=0), start_cell)
    out.update({'odd_n': no, 'odd_hi': nohi, 'all_n': na, 'all_hi': nahi,
                'p_odd_hi': round(nohi / no, 4) if no else None,
                'p_all_hi': round(nahi / na, 4) if na else None,
                'p_odd_hi_shift': round(nohi_s / no, 4) if no else None})
    return out


def pick_windows(split, song, occ, n_target, seed, which=1, per_song=4):
    """按占位率分层 + 每歌上限抽窗口（确定性）。"""
    rng = np.random.default_rng(seed)
    va = np.where(split == which)[0]
    if len(va) == 0:
        raise SystemExit(f'[错误] split=={which} 没有窗口')
    order = va[np.argsort(occ[va], kind='stable')]
    buckets = np.array_split(order, 5)
    cnt, picked = {}, []

    def take(i):
        s = int(song[i])
        if i in picked or cnt.get(s, 0) >= per_song:
            return False
        picked.append(int(i))
        cnt[s] = cnt.get(s, 0) + 1
        return True

    lists = [list(rng.permutation(b)) for b in buckets]
    while len(picked) < n_target and any(lists):
        for l in lists:
            if len(picked) >= n_target:
                break
            while l:
                if take(int(l.pop(0))):
                    break
    if len(picked) < n_target:            # 每歌上限挡住时放宽
        for i in rng.permutation(va):
            if len(picked) >= n_target:
                break
            if int(i) not in picked:
                picked.append(int(i))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='out/diff-data-audio-fixed/dataset.npz')
    ap.add_argument('--ckpt-a', default='out/models/r148-A-fixed.pt')
    ap.add_argument('--ckpt-b', default='out/models/r148-B-fixed.pt')
    ap.add_argument('--num', type=int, default=24, help='评估窗口数（≥20）')
    ap.add_argument('--per-song', type=int, default=4)
    ap.add_argument('--which-split', type=int, default=1, choices=[1, 2],
                    help='1=val（默认）/ 2=test（独立复现）')
    ap.add_argument('--seed', type=int, default=20251)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--cond-shift-cells', type=int, default=1,
                    help='重采样对照里把条件 C 平移几格（1 格≈40ms）')
    ap.add_argument('--out', default='out/diffusion/round148-audio/collect-val.json')
    ap.add_argument('--from-probs', default='',
                    help='复用某个 collect-*-probs.npz 里已采样的概率卷（跳过采样，只重算指标）')
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    smp = load_sampler()
    p2x0 = smp.prediction_to_x0

    d = np.load(os.path.join(ROOT, a.data), allow_pickle=True)
    X = d['X']
    C = np.asarray(d['C'], dtype=np.float32)
    song = d['song'].astype(np.int64)
    split = d['split'].astype(np.int64)
    start = d['start_cell'].astype(np.int64)
    bpm = d['bpm'].astype(np.float64)
    occ = X.reshape(len(X), -1).mean(axis=1).astype(np.float64)

    idx = pick_windows(split, song, occ, a.num, a.seed, a.which_split, a.per_song)
    C_ev = C[idx]
    s_ev = song[idx]
    print(f'评估窗口：{len(idx)} 个（split=={a.which_split}，{len(set(s_ev.tolist()))} 首歌，'
          f'占位率 {occ[idx].min():.3f}–{occ[idx].max():.3f}）', flush=True)

    modelA, diffA, infoA = smp.load_model(os.path.join(ROOT, a.ckpt_a), False,
                                          48, 200, a.threads, 0, pred=None)
    modelB, diffB, infoB = smp.load_model(os.path.join(ROOT, a.ckpt_b), False,
                                          48, 200, a.threads, 0, pred=None)
    print(f'  A：{a.ckpt_a}（cond_dim={infoA["cond_dim"]}, pred={infoA["pred"]}, '
          f'val={infoA.get("训练验证损失")}）', flush=True)
    print(f'  B：{a.ckpt_b}（cond_dim={infoB["cond_dim"]}, pred={infoB["pred"]}, '
          f'val={infoB.get("训练验证损失")}）', flush=True)
    if infoB['cond_dim'] <= 0:
        raise SystemExit('[错误] B 权重没有音频分支')

    dc = infoB.get('difficulty_condition') or infoA.get('difficulty_condition')
    lo, hi = float(dc['lo']), float(dc['hi'])
    diffv = np.clip((occ[idx] - lo) / max(1e-9, hi - lo), 0, 1).astype(np.float32)
    print(f'  难度条件归一化：占位率 {lo:.3f}–{hi:.3f} → [0,1]', flush=True)

    # 错配：打乱 song/start_cell（全排列且每首歌都换掉）
    rng = np.random.default_rng(a.seed + 1)
    perm = None
    for _ in range(4000):
        p = rng.permutation(len(idx))
        if np.all(p != np.arange(len(idx))) and np.all(s_ev[p] != s_ev):
            perm = p
            break
    if perm is None:
        raise SystemExit('[错误] 构造不出打乱 song/start_cell 的错配排列')

    sh = int(a.cond_shift_cells)          # 1 格 ≈ 40ms：条件分支的「半格」扰动
    sh4 = 4                               # 4 格 = 一个 stride-4 下采样单位（≈160ms）
    shifted = {'+%d格' % sh: np.roll(C_ev, sh, axis=1),
               '-%d格' % sh: np.roll(C_ev, -sh, axis=1),
               '+%d格' % sh4: np.roll(C_ev, sh4, axis=1),
               '-%d格' % sh4: np.roll(C_ev, -sh4, axis=1)}

    if a.from_probs:
        zp = np.load(os.path.join(ROOT, a.from_probs), allow_pickle=True)
        perm = zp['perm']
        probs = {k[2:]: zp[k].astype(np.float32) for k in zp.files if k.startswith('P_')}
        rolls = {k: (v > a.threshold).astype(np.uint8) for k, v in probs.items()}
        print(f'▶ 复用 {a.from_probs} 里已采样的 {len(probs)} 组概率卷（不重新采样）', flush=True)
    else:
        plan = [
            ('A-无条件', modelA, diffA, None, diffv),
            ('B-配对', modelB, diffB, C_ev, diffv),
            ('B-错配', modelB, diffB, C_ev[perm], diffv[perm]),
            ('B-零条件', modelB, diffB, None, diffv),
            ('B-条件+%d格' % sh, modelB, diffB, shifted['+%d格' % sh], diffv),
            ('B-条件-%d格' % sh, modelB, diffB, shifted['-%d格' % sh], diffv),
            ('B-条件+%d格' % sh4, modelB, diffB, shifted['+%d格' % sh4], diffv),
            ('B-条件-%d格' % sh4, modelB, diffB, shifted['-%d格' % sh4], diffv),
        ]
        probs, rolls = {}, {}
        for tag, model, dm, cond, dv in plan:
            print(f'▶ 采样 {tag}（{len(idx)} 窗，{dm.T} 步）…', flush=True)
            raw = sample_windows(model, dm, cond, dv, infoB['pred'], a.seed, p2x0)
            probs[tag] = raw
            rolls[tag] = (raw > a.threshold).astype(np.uint8)
            print(f'  完成，平均占用 {rolls[tag].mean():.3f}', flush=True)

    # 概率卷先落盘（后面任何一步出错都不必重采）
    outp0 = os.path.join(ROOT, a.out).replace('.json', '-probs.npz')
    os.makedirs(os.path.dirname(outp0) or '.', exist_ok=True)
    np.savez_compressed(outp0, idx=np.array(idx), perm=perm, C=C_ev,
                        X=X[idx].astype(np.uint8),
                        **{f'P_{k}': v.astype(np.float32) for k, v in probs.items()})

    # 占用校准阈值：同一规则对全部模型（把平均占用拉到人类同批窗口）
    occ_target = float(np.mean([X[i].mean() for i in idx]))
    calib_t = {tag: float(np.quantile(probs[tag], 1.0 - occ_target)) for tag in probs}
    print(f'  人类平均占用 {occ_target:.3f}；校准阈值 '
          + '，'.join(f'{k}={v:.3f}' for k, v in calib_t.items()), flush=True)

    # 结构零假设：若音符在奇数格/偶数格上均匀分布，p_odd_hi 会是多少
    null_odd, null_even = [], []
    for j in range(len(idx)):
        cj = C_ev[j]
        med = float(np.median(cj[:, 3]))
        hi = cj[:, 3] > med
        cells = np.arange(WIN) + int(start[idx[j]])
        odd = (cells % 2) == 1
        null_odd.append(float(hi[odd].mean()))
        null_even.append(float(hi[~odd].mean()))
    print(f'  结构零假设（均匀音符）：p_odd_hi 基线 {np.mean(null_odd):.3f}'
          f'（偶数格 {np.mean(null_even):.3f}）', flush=True)

    # ---- 组装「样本 × 所评估的 C」的行 ----
    row_specs = [
        ('人类（同 split 真谱）', [X[i].astype(np.uint8) for i in idx], C_ev),
        ('A-无条件', rolls['A-无条件'], C_ev),
        ('B-配对', rolls['B-配对'], C_ev),
        ('B-错配', rolls['B-错配'], C_ev[perm]),
        ('B-零条件', rolls['B-零条件'], C_ev),
        ('B-条件+%d格·对其给定C' % sh, rolls['B-条件+%d格' % sh], shifted['+%d格' % sh]),
        ('B-条件+%d格·对真C' % sh, rolls['B-条件+%d格' % sh], C_ev),
        ('B-条件-%d格·对其给定C' % sh, rolls['B-条件-%d格' % sh], shifted['-%d格' % sh]),
        ('B-条件-%d格·对真C' % sh, rolls['B-条件-%d格' % sh], C_ev),
        ('B-条件+%d格·对其给定C' % sh4, rolls['B-条件+%d格' % sh4], shifted['+%d格' % sh4]),
        ('B-条件+%d格·对真C' % sh4, rolls['B-条件+%d格' % sh4], C_ev),
        ('B-条件-%d格·对其给定C' % sh4, rolls['B-条件-%d格' % sh4], shifted['-%d格' % sh4]),
        ('B-条件-%d格·对真C' % sh4, rolls['B-条件-%d格' % sh4], C_ev),
        ('B-错配·对真C', rolls['B-错配'], C_ev),
        ('B-配对·事后错配C', rolls['B-配对'], C_ev[perm]),
        ('A-无条件·事后错配C', rolls['A-无条件'], C_ev[perm]),
    ]
    for tag in ('A-无条件', 'B-配对', 'B-错配'):
        rl = (probs[tag] > calib_t[tag]).astype(np.uint8)
        cc = C_ev[perm] if tag == 'B-错配' else C_ev
        row_specs.append((tag + '·占用校准', rl, cc))

    rows = {}
    for name, rl, cc in row_specs:
        rows[name] = [metrics(rl[j], cc[j], bpm[idx[j]], start[idx[j]])
                      for j in range(len(idx))]

    # ---- 条件因果效应（固定初始噪声，只改条件）----
    def _eff(p1, p0):
        return {'平均|ΔP|': round(float(np.abs(p1 - p0).mean()), 5),
                '二值翻转率': round(float(((p1 > a.threshold) != (p0 > a.threshold)).mean()), 4),
                '占用差': round(float((p1 > a.threshold).mean() - (p0 > a.threshold).mean()), 4)}

    eff = {'B-配对 vs B-零条件': _eff(probs['B-配对'], probs['B-零条件']),
           'B-配对 vs B-错配（换内容）': _eff(probs['B-配对'], probs['B-错配']),
           'B-配对 vs B-条件+%d格（相位）' % sh: _eff(probs['B-配对'],
                                                   probs['B-条件+%d格' % sh]),
           'A-无条件 vs B-零条件（仅权重）': _eff(probs['A-无条件'], probs['B-零条件'])}

    def band_agg(rs, binf, band):
        v = [r[binf][band] for r in rs if r[binf][band] is not None]
        return (float(np.mean(v)), float(np.median(v)), len(v)) if v else (None, None, 0)

    summary = {}
    for tag, rs in rows.items():
        n_odd = sum(r['odd_n'] for r in rs)
        n_all = sum(r['all_n'] for r in rs)
        summary[tag] = {
            '窗口数': len(rs),
            'r250_总': dict(zip(('mean', 'median', 'n'), band_agg(rs, 'r250', '总'))),
            'r1000_总': dict(zip(('mean', 'median', 'n'), band_agg(rs, 'r1000', '总'))),
            'r250_四带中位': {b: round(band_agg(rs, 'r250', b)[1], 4) for b in BANDS},
            'r1000_四带中位': {b: round(band_agg(rs, 'r1000', b)[1], 4) for b in BANDS},
            'r250_格移位扫描中位': {
                str(k): (round(float(np.nanmedian([r['scan_cells'][str(k)] for r in rs
                                                   if r['scan_cells'][str(k)] is not None])), 4)
                         if any(r['scan_cells'][str(k)] is not None for r in rs) else None)
                for k in CELL_SHIFTS},
            'p_odd_hi_池化': round(sum(r['odd_hi'] for r in rs) / n_odd, 4) if n_odd else None,
            'p_odd_hi_平移对照': (
                round(sum((r['p_odd_hi_shift'] or 0.0) * r['odd_n'] for r in rs)
                      / max(1, sum(r['odd_n'] for r in rs
                                   if r['p_odd_hi_shift'] is not None)), 4)
                if any(r['p_odd_hi_shift'] is not None for r in rs) else None),
            'p_all_hi_池化': round(sum(r['all_hi'] for r in rs) / n_all, 4) if n_all else None,
            '占用': round(float(np.mean([r['占用'] for r in rs])), 4),
            '音符总数': int(n_all),
            '1/32格音符数': int(n_odd),
        }

    def paired(x, y, key='r250', band='总'):
        xs, ys = [], []
        for rx, ry in zip(rows[x], rows[y]):
            vx, vy = rx[key][band], ry[key][band]
            if vx is not None and vy is not None and np.isfinite(vx) and np.isfinite(vy):
                xs.append(vx)
                ys.append(vy)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 4:
            return None
        diff = xs - ys
        res = {'n': int(len(xs)), 'diff_mean': round(float(diff.mean()), 4),
               'diff_median': round(float(np.median(diff)), 4),
               'wins': int((diff > 0).sum()), 'losses': int((diff < 0).sum())}
        try:
            from scipy.stats import wilcoxon
            if np.any(diff != 0):
                res['wilcoxon_p'] = float(f'{wilcoxon(xs, ys).pvalue:.4g}')
        except Exception:
            pass
        return res

    comp = {}
    for key in ('r250', 'r1000'):
        for x, y in (('B-配对', 'B-错配'), ('B-配对', 'A-无条件'),
                     ('B-配对', 'B-零条件'), ('B-配对', 'B-配对·占用校准'),
                     ('B-配对·占用校准', 'A-无条件·占用校准'),
                     ('B-配对·占用校准', 'B-错配·占用校准'),
                     ('B-配对', 'B-配对·事后错配C'), ('A-无条件', 'A-无条件·事后错配C'),
                     ('B-配对', 'B-错配·对真C'),
                     ('B-配对', '人类（同 split 真谱）')):
            comp[f'{x} − {y} @{key}'] = paired(x, y, key)
        for s_k in (sh, sh4):
            comp[f'B-条件+{s_k}格·对其给定C − B-配对 @{key}'] = paired(
                'B-条件+%d格·对其给定C' % s_k, 'B-配对', key)

    res = {'config': {'data': a.data, 'which_split': a.which_split,
                      'num_windows': len(idx), 'seed': a.seed,
                      'threshold': a.threshold, 'ckpt_a': a.ckpt_a, 'ckpt_b': a.ckpt_b,
                      'diff_norm': [lo, hi], 'cond_shift_cells': sh,
                      'occ_target_human': round(occ_target, 4),
                      'null_p_odd_hi_uniform': round(float(np.mean(null_odd)), 4),
                      'null_p_odd_hi_even': round(float(np.mean(null_even)), 4),
                      'calib_threshold': {k: round(v, 4) for k, v in calib_t.items()},
                      'note': '错配=打乱 song/start_cell 后的条件（全排列、每首歌都换）；'
                              '「事后错配C」= 不重采样，只把样本与错配的 C 算相关。'},
           'windows': [{'idx': int(i), 'song': int(song[i]), 'start_cell': int(start[i]),
                        'bpm': round(float(bpm[i]), 2), 'occ': round(float(occ[i]), 4),
                        'diff': round(float(diffv[j]), 4),
                        'cond_from_idx': int(idx[perm[j]])}
                       for j, i in enumerate(idx)],
           'summary': summary, 'compare': comp, 'cond_effect': eff,
           'per_window': {k: v for k, v in rows.items()}}
    outp = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(outp) or '.', exist_ok=True)
    def _san(o):                       # 兜底：numpy 标量/数组 → Python 原生
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f'不可序列化 {type(o)!r}')

    with open(outp, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=_san)

    # ---- 控制台 ----
    order = ['人类（同 split 真谱）', 'A-无条件', 'B-配对', 'B-错配', 'B-零条件',
             'A-无条件·占用校准', 'B-配对·占用校准', 'B-错配·占用校准',
             'B-条件+%d格·对其给定C' % sh, 'B-条件+%d格·对真C' % sh,
             'B-条件-%d格·对其给定C' % sh, 'B-条件-%d格·对真C' % sh,
             'B-条件+%d格·对其给定C' % sh4, 'B-条件+%d格·对真C' % sh4,
             'B-条件-%d格·对其给定C' % sh4, 'B-条件-%d格·对真C' % sh4,
             'B-错配·对真C', 'B-配对·事后错配C', 'A-无条件·事后错配C']
    print('\n=== 采音指标（r 中位；p_odd_hi 池化，随机≈0.5）===')
    print(f'{"组":28s} {"r250总":>7s} {"r1000总":>7s} {"p_odd_hi":>8s} {"占全部符":>7s} '
          f'{"占用":>6s} {"音符":>7s}')
    for tag in order:
        if tag not in summary:
            continue
        s = summary[tag]
        p_all = round(sum(r['odd_hi'] for r in rows[tag]) / max(1, s['音符总数']), 4)
        print(f'{tag:28s} {s["r250_总"]["median"]:>7.3f} {s["r1000_总"]["median"]:>7.3f} '
              f'{str(s["p_odd_hi_池化"]):>8s} {p_all:>7.3f} {s["占用"]:>6.3f} '
              f'{s["音符总数"]:>7d}')
    print('\n=== r250（总带）格移位扫描中位：把 C 平移 k 格后算 r（k=0 为真对齐，1 格≈40ms）===')
    print(f'{"组":28s} ' + ' '.join(f'{k:>7d}' for k in CELL_SHIFTS))
    for tag in order:
        if tag not in summary:
            continue
        sc = summary[tag]['r250_格移位扫描中位']
        print(f'{tag:28s} ' + ' '.join(
            (f'{sc[str(k)]:>7.3f}' if sc[str(k)] is not None else f'{"-":>7s}')
            for k in CELL_SHIFTS))
    print('\n=== 逐窗口配对比较（均值差 / n / 胜-负 / Wilcoxon p）===')
    for k, v in comp.items():
        if v:
            print(f'  {k}: {v["diff_mean"]:+.4f}（n={v["n"]}，{v["wins"]}胜{v["losses"]}负'
                  + (f'，p={v["wilcoxon_p"]:.3g}' if 'wilcoxon_p' in v else '') + ')')
    print('\n=== 条件因果效应（固定初始噪声，只改条件）===')
    for k, v in eff.items():
        print(f'  {k}: 平均|ΔP| {v["平均|ΔP|"]:.4f}  二值翻转率 {v["二值翻转率"]:.3f}  '
              f'占用差 {v["占用差"]:+.4f}')
    print(f'\n写出 {a.out}')


if __name__ == '__main__':
    main()
