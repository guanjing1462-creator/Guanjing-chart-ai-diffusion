#!/usr/bin/env python3
"""采音（audio-following）验收 —— **mel 条件版**（round 151）。

协议沿用 `tools/eval-audio-follow.py`（round 148/150），只把条件从 4 带换成
任意通道数（mel=32 / 4 带=4），并把「条件强度」定义为：

    条件能量 = 通道求和（mel 32 维求和；4 带用第 4 通道「总带」，与上轮同口径）

同一批 24 个验证窗（`pick_windows(split=1, seed=20251, per_song=4)`，与上轮
完全一致的抽样），同一套初始噪声，只改**喂给模型的条件**：

    paired   本窗条件                     ← 应当最好
    mismatch 另一窗的条件（打乱歌/起点，难度也一起换）
    zero     全零条件
    ±1格     本窗条件沿时间轴平移 1 格     ← 相位敏感性

主判据：**paired − mismatch 的逐窗差**（真正的对齐信号）；
另外给出「占用校准」变体（阈值把平均占用拉到人类同批窗口），与上轮可对照。

用法
----
  python3 tools/eval-mel-follow.py --tag mel  --ckpt out/models/r151-M-mel.pt
  python3 tools/eval-mel-follow.py --tag band4 --ckpt out/models/r151-C-band4.pt
"""
import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = 256


def load_module(rel, name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TRAIN = load_module('tools/train-diffusion32-mel.py', 't32mel')


def pearson(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def sample_windows(model, diff, cond, diffv, pred, seed, n):
    """逐窗口 diff / 逐窗口 cond 的 DDPM 采样（公式同 sample-diffusion32c.py）。"""
    torch.manual_seed(seed)
    x = torch.randn(n, 4, WIN)
    sty = torch.zeros(n, dtype=torch.long)
    dv = torch.as_tensor(diffv, dtype=torch.float32)
    cc = None if cond is None else torch.as_tensor(
        np.ascontiguousarray(cond), dtype=torch.float32)
    if cc is not None:
        assert cc.shape[0] == n and cc.shape[1] == WIN, cc.shape
    T = diff.T
    for t in reversed(range(T)):
        tb = torch.full((n,), t, dtype=torch.long)
        out = model(x, tb, sty, dv, cc)
        a_t, ab = diff.alphas[t], diff.abar[t]
        x0 = _p2x0(x, out, ab, pred)
        if pred == 'x0':
            x0 = x0.clamp(-1.0, 2.0)
        if t > 0:
            abp = diff.abar[t - 1]
            mean = (abp.sqrt() * diff.betas[t] * x0
                    + a_t.sqrt() * (1 - abp) * x) / (1 - ab)
            sigma = (diff.betas[t] * (1 - abp) / (1 - ab)).sqrt()
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean
    return x.numpy()


def _p2x0(x, out, ab, pred):
    if pred == 'eps':
        return (x - (1 - ab).sqrt() * out) / ab.sqrt()
    if pred == 'x0':
        return out
    return ab.sqrt() * x - (1 - ab).sqrt() * out


def cond_energy(C):
    """[...,ch] → [...] 条件强度：mel=32 维求和；4 带用「总带」通道。"""
    return C[..., 3] if C.shape[-1] == 4 else C.sum(axis=-1)


def binned(roll, C, cpb):
    nb = WIN // cpb
    m = nb * cpb
    dens = roll[:, :m].sum(axis=0).reshape(-1, cpb).sum(axis=1)
    cb = cond_energy(C[:m]).reshape(nb, cpb).sum(axis=1)
    return dens, cb


def metrics(roll, C, step_ms):
    out = {'占用': float(roll.mean()), '音符数': int(roll.sum())}
    e = cond_energy(C)
    for tag, bin_ms in (('250', 250.0), ('1000', 1000.0)):
        cpb = max(1, int(round(bin_ms / step_ms)))
        dens, cb = binned(roll.astype(np.float64), C, cpb)
        r = pearson(dens, cb)
        out[f'r{tag}'] = round(r, 4) if np.isfinite(r) else None
        out[f'dens_cv{tag}'] = round(float(dens.std() / max(1e-9, dens.mean())), 4)
    # 相位敏感度曲线：把条件平移 k 格后与同一批音符算 r250
    step_cpb = max(1, int(round(250.0 / step_ms)))
    scan = {}
    for k in (-2, -1, 0, 1, 2):
        dens, cb = binned(roll.astype(np.float64), np.roll(C, k, axis=0), step_cpb)
        r = pearson(dens, cb)
        scan[str(k)] = round(r, 4) if np.isfinite(r) else None
    out['scan_cells'] = scan
    out['energy_mean'] = round(float(e.mean()), 4)
    return out


def pick_windows(split, song, occ, n_target, seed, which=1, per_song=4):
    """按占位率分层 + 每歌上限抽窗口（与 round148/150 完全同一算法）。"""
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
    if len(picked) < n_target:
        for i in rng.permutation(va):
            if len(picked) >= n_target:
                break
            if int(i) not in picked:
                picked.append(int(i))
    return picked


def sign_test(delta):
    """paired 差值的符号检验（双侧精确二项）。"""
    d = np.asarray([x for x in delta if np.isfinite(x)], np.float64)
    w = int((d > 0).sum())
    n = len(d)
    if n == 0:
        return dict(n=0, win=0, p=None, median=None, mean=None)
    k = min(w, n - w)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * 2.0 / (2 ** n)
    return dict(n=n, win=w, median=round(float(np.median(d)), 4),
                mean=round(float(d.mean()), 4), p=round(min(1.0, p), 6))


def load_ckpt(path, threads=1):
    ck = torch.load(os.path.join(ROOT, path), map_location='cpu')
    model = TRAIN.UNet1D(base=ck.get('base', 48), cond_dim=ck['cond_dim'],
                         cond_ch=ck.get('cond_channels', 4),
                         cond_stride=ck.get('cond_stride', 4))
    model.load_state_dict(ck['state'], strict=True)
    model.eval()
    diff = TRAIN.Diffusion(ck.get('timesteps', 200))
    return model, diff, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='mel / band4')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-mel', default='out/diff-data-mel/dataset.npz')
    ap.add_argument('--data-band', default='out/diff-data-audio-fixed/dataset.npz')
    ap.add_argument('--num', type=int, default=24)
    ap.add_argument('--per-song', type=int, default=4)
    ap.add_argument('--which-split', type=int, default=1, choices=[1, 2])
    ap.add_argument('--seed', type=int, default=20251)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    t_all = time.time()

    dm = np.load(os.path.join(ROOT, a.data_mel), allow_pickle=True)
    db = np.load(os.path.join(ROOT, a.data_band), allow_pickle=True)
    X = dm['X']
    assert np.array_equal(X, db['X']), '两个数据集的 X 不一致（行序变了）'
    song = dm['song'].astype(np.int64)
    split = dm['split'].astype(np.int64)
    start = dm['start_cell'].astype(np.int64)
    bpm = dm['bpm'].astype(np.float64)
    occ = X.reshape(len(X), -1).mean(axis=1).astype(np.float64)
    Cond = {'M': np.asarray(dm['M'], dtype=np.float32),
            'C': np.asarray(db['C'], dtype=np.float32)}

    idx = pick_windows(split, song, occ, a.num, a.seed, a.which_split, a.per_song)
    s_ev = song[idx]
    print(f'评估窗口 {len(idx)} 个（split=={a.which_split}，'
          f'{len(set(s_ev.tolist()))} 首歌，占位率 {occ[idx].min():.3f}–'
          f'{occ[idx].max():.3f}）', flush=True)

    model, diff, ck = load_ckpt(a.ckpt, a.threads)
    ck_key = ck.get('cond_key') or ('M' if ck.get('cond_channels') == 32 else 'C')
    Cev = Cond[ck_key][idx]
    ch = Cev.shape[-1]
    print(f'  权重 {a.ckpt}（cond_key={ck_key}, 通道 {ch}, cond_dim={ck["cond_dim"]}, '
          f'pred={ck["pred"]}, val={ck.get("val"):.5f}）', flush=True)
    if ch != ck.get('cond_channels'):
        raise SystemExit(f'[错误] 数据集条件通道 {ch} 与权重 {ck.get("cond_channels")} 不符')

    dc = ck['difficulty_condition']
    lo, hi = float(dc['lo']), float(dc['hi'])
    diffv = np.clip((occ[idx] - lo) / max(1e-9, hi - lo), 0, 1).astype(np.float32)

    # 错配：打乱 song/start_cell（每窗都换掉，且歌也不一样）
    rng = np.random.default_rng(a.seed + 1)
    perm = None
    for _ in range(4000):
        p = rng.permutation(len(idx))
        if np.all(p != np.arange(len(idx))) and np.all(s_ev[p] != s_ev):
            perm = p
            break
    if perm is None:
        raise SystemExit('[错误] 构造不出错配排列')

    plan = [('配对', Cev, diffv),
            ('错配', Cev[perm], diffv[perm]),
            ('零条件', None, diffv),
            ('条件+1格', np.roll(Cev, 1, axis=1), diffv),
            ('条件-1格', np.roll(Cev, -1, axis=1), diffv)]
    probs = {}
    for tag, cond, dv in plan:
        t0 = time.time()
        print(f'▶ 采样 {tag}（{len(idx)} 窗 × {diff.T} 步）…', flush=True)
        probs[tag] = sample_windows(model, diff, cond, dv, ck['pred'], a.seed,
                                    len(idx))
        print(f'  完成 {time.time()-t0:.0f}s，平均占用 '
              f'{(probs[tag] > a.threshold).mean():.3f}', flush=True)

    occ_target = float(np.mean([X[i].mean() for i in idx]))
    calib = {t: float(np.quantile(probs[t], 1.0 - occ_target)) for t in probs}
    print(f'  人类平均占用 {occ_target:.3f}；校准阈值 '
          + '，'.join(f'{k}={v:.3f}' for k, v in calib.items()), flush=True)

    rolls = {t: (probs[t] > a.threshold).astype(np.uint8) for t in probs}
    cal = {t: (probs[t] > calib[t]).astype(np.uint8) for t in probs}

    # ---- 行：样本 × 所评估的条件 ----
    L = lambda n: np.roll(Cev, n, axis=1)
    row_specs = [
        ('人类', [X[i].astype(np.uint8) for i in idx], Cev),
        ('配对', rolls['配对'], Cev),
        ('错配', rolls['错配'], Cev[perm]),
        ('零条件', rolls['零条件'], Cev),
        ('条件+1格·对其给定条件', rolls['条件+1格'], L(1)),
        ('条件+1格·对真条件', rolls['条件+1格'], Cev),
        ('条件-1格·对其给定条件', rolls['条件-1格'], L(-1)),
        ('条件-1格·对真条件', rolls['条件-1格'], Cev),
        ('配对·事后错配条件', rolls['配对'], Cev[perm]),
        ('配对·占用校准', cal['配对'], Cev),
        ('错配·占用校准', cal['错配'], Cev[perm]),
        ('零条件·占用校准', cal['零条件'], Cev),
    ]
    step_ms = 60000.0 / np.maximum(1e-6, bpm[idx]) / 8.0
    rows = {name: [metrics(r[j], cc[j], float(step_ms[j]))
                   for j in range(len(idx))]
            for name, r, cc in row_specs}

    def agg(rs, key):
        v = [r[key] for r in rs if r[key] is not None]
        return round(float(np.median(v)), 4) if v else None

    summary = {name: {'r250': agg(rs, 'r250'), 'r1000': agg(rs, 'r1000'),
                      '占用': round(float(np.mean([r['占用'] for r in rs])), 4),
                      '音符数': int(np.mean([r['音符数'] for r in rs]))}
               for name, rs in rows.items()}

    def delta(name_a, name_b, key='r250'):
        d = []
        for i in range(len(idx)):
            x, y = rows[name_a][i].get(key), rows[name_b][i].get(key)
            d.append(np.nan if (x is None or y is None) else x - y)
        return sign_test(d)

    deltas = {
        '配对 − 错配 @r250': delta('配对', '错配', 'r250'),
        '配对 − 错配 @r1000': delta('配对', '错配', 'r1000'),
        '配对 − 零条件 @r250': delta('配对', '零条件', 'r250'),
        '配对·校准 − 错配·校准 @r250': delta('配对·占用校准', '错配·占用校准', 'r250'),
        '配对·校准 − 错配·校准 @r1000': delta('配对·占用校准', '错配·占用校准', 'r1000'),
        '配对·校准 − 零条件·校准 @r250': delta('配对·占用校准', '零条件·占用校准', 'r250'),
        '配对 − 人类 @r250': delta('配对', '人类', 'r250'),
        '配对·校准 − 人类 @r250': delta('配对·占用校准', '人类', 'r250'),
    }
    # 相位敏感性：把给定条件平移 1 格后再评估，与其自身给定条件比
    for tag in ('条件+1格·对其给定条件', '条件-1格·对其给定条件'):
        deltas[f'{tag} − 配对 @r250'] = delta(tag, '配对', 'r250')
    deltas['条件+1格·对真条件 − 配对 @r250'] = delta('条件+1格·对真条件', '配对', 'r250')
    deltas['条件-1格·对真条件 − 配对 @r250'] = delta('条件-1格·对真条件', '配对', 'r250')

    print(f'\n=== 采音指标（{a.tag}；r 中位，n={len(idx)}）===')
    print(f'{"组":28s} {"r250":>7s} {"r1000":>7s} {"占用":>6s} {"音符":>6s}')
    for name, s in summary.items():
        print(f'{name:28s} {s["r250"]:7.3f} {s["r1000"]:7.3f} '
              f'{s["占用"]:6.3f} {s["音符数"]:6d}')
    print('\n=== 配对差值（逐窗；主判据是「配对−错配」）===')
    for k, v in deltas.items():
        if v['n']:
            print(f'  {k}: 中位 {v["median"]:+.4f}，均值 {v["mean"]:+.4f}，'
                  f'{v["win"]}/{v["n"]} 胜，p={v["p"]:.2e}')
    print(f'\n总用时 {time.time()-t_all:.0f}s', flush=True)

    out = a.out or f'out/diffusion/round151-mel/eval-{a.tag}.json'
    outp = os.path.join(ROOT, out)
    os.makedirs(os.path.dirname(outp) or '.', exist_ok=True)
    json.dump(dict(tag=a.tag, ckpt=a.ckpt, cond_key=ck_key, cond_channels=ch,
                   cond_dim=int(ck['cond_dim']), pred=ck['pred'],
                   val=ck.get('val'), epochs=ck.get('epoch'),
                   data_mel=a.data_mel, data_band=a.data_band,
                   idx=[int(i) for i in idx], perm=[int(x) for x in perm],
                   occ_target=occ_target, threshold=a.threshold,
                   calib=calib, summary=summary, deltas=deltas,
                   rows=rows, seconds=round(time.time() - t_all, 1)),
              open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'写出 {out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
