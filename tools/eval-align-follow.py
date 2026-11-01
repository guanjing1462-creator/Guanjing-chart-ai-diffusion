#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""round 153：**多尺度全分辨率**条件注入（S1）vs **stride=4 单点**注入（S4）的采音验收。

协议与 tools/eval-audio-follow.py 逐字段对齐（同一套窗口抽样、同一初始噪声、
同样的 250/1000ms 分箱 Pearson、同样的 ±格平移与 p_odd_hi 定义），只改三处：

  1. 条件换成 32 带 **mel onset 通量**（`M`），「总强度」= 32 带逐格求和
     （round151 的 mel 口径；4 带版里的「总」通道在这里没有对应物）；
  2. 一次加载**两个权重**（S1 / S4），每组采样都喂同一初始噪声，
     因此「S1 与 S4 的差」也不会混入采样噪声；
  3. 除配对/错配/零条件外，对**两个**权重都做 ±1 格（≈40ms）与
     ±4 格（≈160ms，旧分支一个 stride 单位）的条件平移。

指标（全部沿用旧脚本的定义，便于与 round148/151 的数字直接对照）
------------------------------------------------------------------
  ① r250 / r1000：生成窗口的 1/32 格占用按 250ms / 1000ms 分箱后，
     与条件总强度的 Pearson r（逐窗算，报中位）；
  ② `p_odd_hi`：音符落在 1/32 细分格（全局格号为奇数）**且**该格条件总强度
     高于窗内中位的比例（池化 = 总命中/总奇数格音符）。这是「1/32 踩格」的直接
     指标 —— 人类 val 0.813 / test 0.785，结构零假设（均匀音符）0.639 / 0.604。
  ③ `scan_cells`：把条件平移 k 格后重算 r250（相位敏感度曲线，峰越尖 = 相位越细）。

显著性：逐窗配对差 + Wilcoxon 符号秩；`p_odd_hi` 另用**按窗口 bootstrap**
（2000 次重采样窗口，池化计数）给 S1−S4 的置信区间与单侧 p。
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = 256
CELL_SHIFTS = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
SHIFTS = (-4, -1, 1, 4)          # 报告用的条件平移（1 格 ≈ 40ms，4 格 ≈ 160ms）


def load_module(relpath, name):
    """按路径导入模块（文件名带连字符，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@torch.no_grad()
def sample_windows(model, diffmod, cond, diffv, pred, seed, prediction_to_x0):
    """与 eval-audio-follow.py 的采样完全相同：同 seed → 同初始噪声。"""
    n = len(diffv)
    torch.manual_seed(seed)
    x = torch.randn(n, 4, WIN)
    sty = torch.zeros(n, dtype=torch.long)
    dv = torch.as_tensor(diffv, dtype=torch.float32)
    if cond is not None:
        cond = torch.as_tensor(np.ascontiguousarray(cond), dtype=torch.float32)
        assert cond.shape[0] == n and cond.shape[1] == WIN, cond.shape
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


def load_align_model(trainer, ckpt, threads):
    """按 checkpoint 元数据复原 S1/S4 模型（strict 加载，结构不符直接报错）。"""
    torch.set_num_threads(max(1, threads))
    ck = torch.load(os.path.join(ROOT, ckpt), map_location='cpu', weights_only=False)
    state = ck.get('state', ck)
    base = int(ck.get('base', 48))
    pred = ck.get('pred', 'x0')
    cond_dim = int(ck.get('cond_dim', 0))
    cond_ch = int(ck.get('cond_channels', 0))
    inject = str(ck.get('inject', 'mid4'))
    cond_stride = int(ck.get('cond_stride', 4 if inject == 'mid4' else 1))
    model = trainer.UNet1D(base=base, cond_dim=cond_dim, cond_ch=cond_ch,
                           cond_stride=cond_stride, inject=inject)
    model.load_state_dict(state, strict=True)
    model.eval()
    info = {'路径': ckpt, 'pred': pred, 'base': base, 'cond_dim': cond_dim,
            'cond_channels': cond_ch, 'inject': inject, 'cond_stride': cond_stride,
            'val': ck.get('val'), 'epoch': ck.get('epoch'),
            'cond_key': ck.get('cond_key'), 'cond_drop': ck.get('cond_drop'),
            'params': int(sum(p.numel() for p in model.parameters())),
            'difficulty_condition': ck.get('difficulty_condition')}
    return model, trainer.Diffusion(int(ck.get('timesteps', 200))), info


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def binned(roll, ctot, cells_per_bin):
    nb = WIN // cells_per_bin
    m = nb * cells_per_bin
    dens = roll[:, :m].sum(axis=0).reshape(-1, cells_per_bin).sum(axis=1)
    cb = ctot[:m].reshape(nb, cells_per_bin).sum(axis=1)
    return dens, cb


def _count_odd_hi(roll, ctot, start_cell):
    med = float(np.median(ctot))
    cells = np.arange(WIN) + int(start_cell)
    odd = (cells % 2) == 1
    high = ctot > med
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


def metrics(roll_bin, ctot, bpm, start_cell):
    """单窗口指标（与 eval-audio-follow.py 同口径，条件用 1 维总强度）。"""
    step_ms = 60000.0 / max(1e-6, float(bpm)) / 8.0
    out = {'step_ms': round(step_ms, 3), '占用': float(roll_bin.mean()),
           '音符数': int(roll_bin.sum())}
    for tag, bin_ms in (('250', 250.0), ('1000', 1000.0)):
        cpb = max(1, int(round(bin_ms / step_ms)))
        dens, cb = binned(roll_bin.astype(np.float64), ctot.astype(np.float64), cpb)
        r = pearson(dens, cb)
        out['cpb' + tag] = cpb
        out['r' + tag] = round(r, 4) if np.isfinite(r) else None
        out['dens_cv' + tag] = round(float(dens.std() / max(1e-9, dens.mean())), 4)
    cpb250 = max(1, int(round(250.0 / step_ms)))
    scan = {}
    for k in CELL_SHIFTS:
        dens, cb = binned(roll_bin.astype(np.float64),
                          np.roll(ctot, k).astype(np.float64), cpb250)
        r = pearson(dens, cb)
        scan[str(k)] = round(r, 4) if np.isfinite(r) else None
    out['scan_cells'] = scan
    no, nohi, na, nahi = _count_odd_hi(roll_bin, ctot, start_cell)
    out.update({'odd_n': no, 'odd_hi': nohi, 'all_n': na, 'all_hi': nahi,
                'p_odd_hi': round(nohi / no, 4) if no else None,
                'p_all_hi': round(nahi / na, 4) if na else None})
    return out


def pick_windows(split, song, occ, n_target, seed, which=1, per_song=4):
    """按占位率分层 + 每歌上限抽窗口（确定性；与旧脚本逐行相同）。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='out/diff-data-mel/dataset.npz')
    ap.add_argument('--cond-key', default='M')
    ap.add_argument('--ckpt-s1', default='out/models/r153-S1-multi.pt')
    ap.add_argument('--ckpt-s4', default='out/models/r153-S4-mid4.pt')
    ap.add_argument('--label-s1', default='S1')
    ap.add_argument('--label-s4', default='S4')
    ap.add_argument('--num', type=int, default=24, help='评估窗口数')
    ap.add_argument('--per-song', type=int, default=4)
    ap.add_argument('--which-split', type=int, default=1, choices=[1, 2])
    ap.add_argument('--seed', type=int, default=20251)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--boot', type=int, default=2000, help='p_odd_hi 的窗口 bootstrap 次数')
    ap.add_argument('--out', default='out/diffusion/round153-align/collect-val.json')
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    trainer = load_module('tools/train-diffusion32-align.py', 'tr153')
    smp = load_module('tools/sample-diffusion32c.py', 'smp32c')
    p2x0 = smp.prediction_to_x0

    d = np.load(os.path.join(ROOT, a.data), allow_pickle=True)
    X = d['X']
    M = np.asarray(d[a.cond_key], dtype=np.float32)      # [N,256,ch] 32 带
    song = d['song'].astype(np.int64)
    split = d['split'].astype(np.int64)
    start = d['start_cell'].astype(np.int64)
    bpm = d['bpm'].astype(np.float64)
    occ = X.reshape(len(X), -1).mean(axis=1).astype(np.float64)
    cin = int(M.shape[2])
    print(f'数据 {a.data}：条件键 {a.cond_key} → {M.shape[1:]}（{cin} 带），'
          f'X {tuple(X.shape)}', flush=True)

    idx = pick_windows(split, song, occ, a.num, a.seed, a.which_split, a.per_song)
    C_ev = M[idx]                                        # [n,256,ch]
    ctot_ev = C_ev.sum(axis=2)                           # [n,256] 总强度
    s_ev = song[idx]
    print(f'评估窗口：{len(idx)} 个（split=={a.which_split}，'
          f'{len(set(s_ev.tolist()))} 首歌，占位率 {occ[idx].min():.3f}–{occ[idx].max():.3f}）',
          flush=True)

    m1, d1, i1 = load_align_model(trainer, a.ckpt_s1, a.threads)
    m4, d4, i4 = load_align_model(trainer, a.ckpt_s4, a.threads)
    for tag, info in ((a.label_s1, i1), (a.label_s4, i4)):
        if info['cond_channels'] != cin:
            raise SystemExit(f'[错误] {tag} 权重条件通道 {info["cond_channels"]} ≠ 数据 {cin}')
        print(f'  {tag}：{info["路径"]}（inject={info["inject"]}, cond_stride='
              f'{info["cond_stride"]}, pred={info["pred"]}, val={info["val"]}, '
              f'epoch={info["epoch"]}, 参数 {info["params"]/1e6:.4f}M）', flush=True)

    dc = i1.get('difficulty_condition') or i4.get('difficulty_condition')
    if not dc:
        raise SystemExit('[错误] 权重里没有 difficulty_condition（无法复现难度条件）')
    lo, hi = float(dc['lo']), float(dc['hi'])
    diffv = np.clip((occ[idx] - lo) / max(1e-9, hi - lo), 0, 1).astype(np.float32)
    print(f'  难度条件归一化：占位率 {lo:.3f}–{hi:.3f} → [0,1]', flush=True)

    # ---- 错配条件：打乱 song/start_cell（每首歌都换成别的歌的窗口）----
    rng = np.random.default_rng(a.seed + 1)
    perm = None
    for _ in range(4000):
        p = rng.permutation(len(idx))
        if np.all(p != np.arange(len(idx))) and np.all(s_ev[p] != s_ev):
            perm = p
            break
    if perm is None:
        raise SystemExit('[错误] 构造不出打乱 song/start_cell 的错配排列')
    ctot_mm = ctot_ev[perm]

    shift_arr = {}
    for k in SHIFTS:
        shift_arr[k] = np.roll(C_ev, k, axis=1)
        shift_arr[k + 100] = np.roll(ctot_ev, k, axis=1)   # 下标 100+k 存「给定条件的总强度」

    # ---- 每组 = 一次采样（同 seed → 14 组共用同一初始噪声）----
    groups = [('配对', C_ev, diffv),
              ('错配', C_ev[perm], diffv[perm]),
              ('零条件', None, diffv)]
    for k in SHIFTS:
        groups.append(('条件%+d格' % k, shift_arr[k], diffv))

    rolls, probs = {}, {}
    for tag, model, dm, info in ((a.label_s1, m1, d1, i1), (a.label_s4, m4, d4, i4)):
        for gname, cond, dv in groups:
            print(f'▶ {tag}·{gname}（{len(idx)} 窗 × {dm.T} 步）…', flush=True)
            raw = sample_windows(model, dm, cond, dv, info['pred'], a.seed, p2x0)
            key = '%s-%s' % (tag, gname)
            probs[key] = raw
            rolls[key] = (raw > a.threshold).astype(np.uint8)
            print(f'  完成，平均占用 {rolls[key].mean():.3f}', flush=True)

    outp0 = os.path.join(ROOT, a.out).replace('.json', '-probs.npz')
    os.makedirs(os.path.dirname(outp0) or '.', exist_ok=True)
    np.savez_compressed(outp0, idx=np.array(idx), perm=perm, C=C_ev,
                        X=X[idx].astype(np.uint8),
                        **{f'P_{k}': v.astype(np.float32) for k, v in probs.items()})
    print(f'  概率卷已落盘 {outp0}', flush=True)

    # ---- 占用校准（同一规则，把平均占用拉到人类同批窗口）----
    occ_target = float(np.mean([X[i].mean() for i in idx]))
    calib_t = {k: float(np.quantile(v, 1.0 - occ_target)) for k, v in probs.items()}
    print(f'  人类平均占用 {occ_target:.3f}；校准阈值 '
          + '，'.join(f'{k}={v:.3f}' for k, v in calib_t.items()), flush=True)

    # ---- 结构零假设：均匀音符的 p_odd_hi ----
    null_odd, null_even = [], []
    for j in range(len(idx)):
        med = float(np.median(ctot_ev[j]))
        hiv = ctot_ev[j] > med
        cells = np.arange(WIN) + int(start[idx[j]])
        odd = (cells % 2) == 1
        null_odd.append(float(hiv[odd].mean()))
        null_even.append(float(hiv[~odd].mean()))
    null_p = float(np.mean(null_odd))
    print(f'  结构零假设（均匀音符）：p_odd_hi 基线 {null_p:.3f}'
          f'（偶数格 {np.mean(null_even):.3f}）', flush=True)

    # ---- 组装行：每行 = 一批生成谱 + 它所对的「条件总强度」 ----
    rows = {}
    rows['人类（同 split 真谱）'] = [metrics(X[i].astype(np.uint8), ctot_ev[j],
                                       bpm[idx[j]], start[idx[j]])
                                 for j, i in enumerate(idx)]
    for tag in (a.label_s1, a.label_s4):
        rows[f'{tag}-配对'] = [metrics(rolls[f'{tag}-配对'][j], ctot_ev[j],
                                    bpm[idx[j]], start[idx[j]]) for j in range(len(idx))]
        rows[f'{tag}-错配·对真C'] = [metrics(rolls[f'{tag}-错配'][j], ctot_ev[j],
                                        bpm[idx[j]], start[idx[j]]) for j in range(len(idx))]
        rows[f'{tag}-错配·对其给定C'] = [metrics(rolls[f'{tag}-错配'][j], ctot_mm[j],
                                           bpm[idx[j]], start[idx[j]]) for j in range(len(idx))]
        rows[f'{tag}-配对·事后错配C'] = [metrics(rolls[f'{tag}-配对'][j], ctot_mm[j],
                                           bpm[idx[j]], start[idx[j]]) for j in range(len(idx))]
        for k in SHIFTS:
            sk = shift_arr[k + 100]
            rows[f'{tag}-条件{k:+d}格·对其给定C'] = [
                metrics(rolls[f'{tag}-条件{k:+d}格'][j], sk[j],
                        bpm[idx[j]], start[idx[j]]) for j in range(len(idx))]
            rows[f'{tag}-条件{k:+d}格·对真C'] = [
                metrics(rolls[f'{tag}-条件{k:+d}格'][j], ctot_ev[j],
                        bpm[idx[j]], start[idx[j]]) for j in range(len(idx))]
        rows[f'{tag}-零条件'] = [metrics(rolls[f'{tag}-零条件'][j], ctot_ev[j],
                                    bpm[idx[j]], start[idx[j]]) for j in range(len(idx))]

    # 占用校准行
    for tag in (a.label_s1, a.label_s4):
        for gname in ('配对', '错配', '零条件'):
            key = f'{tag}-{gname}'
            rl = (probs[key] > calib_t[key]).astype(np.uint8)
            cc = ctot_mm if gname == '错配' else ctot_ev
            rows[f'{tag}-{gname}·占用校准'] = [
                metrics(rl[j], cc[j], bpm[idx[j]], start[idx[j]]) for j in range(len(idx))]

    # ---- 池化汇总 ----
    def band_med(rs, key):
        v = [r[key] for r in rs if r[key] is not None]
        return float(np.median(v)) if v else None

    summary = {}
    for tag, rs in rows.items():
        n_odd = sum(r['odd_n'] for r in rs)
        n_all = sum(r['all_n'] for r in rs)
        # 输出「位移估计」：逐窗取 scan_cells 的峰位（r250 最大处），再取中位。
        # 条件平移 k 格后若输出真的跟着移动 k 格，这个峰位就应该 ≈ k。
        est = []
        for r in rs:
            vals = [(int(k), v) for k, v in r['scan_cells'].items() if v is not None]
            if vals:
                est.append(max(vals, key=lambda t: t[1])[0])
        summary[tag] = {
            '窗口数': len(rs),
            '峰值位移_格': int(np.median(est)) if est else None,
            'r250_中位': round(band_med(rs, 'r250'), 4) if band_med(rs, 'r250') is not None else None,
            'r250_均值': round(float(np.nanmean([r['r250'] for r in rs if r['r250'] is not None])), 4),
            'r1000_中位': round(band_med(rs, 'r1000'), 4) if band_med(rs, 'r1000') is not None else None,
            'r1000_均值': round(float(np.nanmean([r['r1000'] for r in rs if r['r1000'] is not None])), 4),
            'r250_格移位扫描中位': {
                str(k): (round(float(np.nanmedian([r['scan_cells'][str(k)] for r in rs
                                                   if r['scan_cells'][str(k)] is not None])), 4)
                         if any(r['scan_cells'][str(k)] is not None for r in rs) else None)
                for k in CELL_SHIFTS},
            'p_odd_hi_池化': round(n_odd_hi := sum(r['odd_hi'] for r in rs) / n_odd, 4) if n_odd else None,
            'p_all_hi_池化': round(sum(r['all_hi'] for r in rs) / n_all, 4) if n_all else None,
            '占用': round(float(np.mean([r['占用'] for r in rs])), 4),
            '音符总数': int(n_all),
            '1/32格音符数': int(n_odd),
            '1/32格高强音符数': int(sum(r['odd_hi'] for r in rs)),
        }

    # ---- 逐窗配对统计 ----
    def paired(x, y, key='r250'):
        xs, ys = [], []
        for rx, ry in zip(rows[x], rows[y]):
            vx, vy = rx[key], ry[key]
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

    def paired_podd(x, y):
        """p_odd_hi 的逐窗配对差（窗口内比例差，等权），外加池化差。"""
        dx, nx = [], []
        for rx, ry in zip(rows[x], rows[y]):
            px, py = rx['p_odd_hi'], ry['p_odd_hi']
            if px is not None and py is not None:
                dx.append(px - py)
                nx.append(rx['odd_n'])
        if len(dx) < 4:
            return None
        dx = np.array(dx)
        res = {'n': int(len(dx)), 'diff_mean': round(float(dx.mean()), 4),
               'wins': int((dx > 0).sum()), 'losses': int((dx < 0).sum()),
               'pooled_x': summary[x]['p_odd_hi_池化'],
               'pooled_y': summary[y]['p_odd_hi_池化']}
        res['pooled_diff'] = round(res['pooled_x'] - res['pooled_y'], 4)
        try:
            from scipy.stats import wilcoxon
            if np.any(dx != 0):
                res['wilcoxon_p'] = float(f'{wilcoxon(dx).pvalue:.4g}')
        except Exception:
            pass
        return res

    def boot_podd(x, y, n_boot):
        """按窗口 bootstrap：重采样窗口后重算池化 p_odd_hi 之差（x−y）。"""
        g = np.random.default_rng(12345)
        nx = np.array([r['odd_n'] for r in rows[x]])
        hx = np.array([r['odd_hi'] for r in rows[x]])
        ny = np.array([r['odd_n'] for r in rows[y]])
        hy = np.array([r['odd_hi'] for r in rows[y]])
        n = len(nx)
        diffs = []
        for _ in range(n_boot):
            sel = g.integers(0, n, n)
            px = hx[sel].sum() / max(1, nx[sel].sum())
            py = hy[sel].sum() / max(1, ny[sel].sum())
            diffs.append(px - py)
        diffs = np.array(diffs)
        return {'boot_mean': round(float(diffs.mean()), 4),
                'ci95': [round(float(np.percentile(diffs, 2.5)), 4),
                         round(float(np.percentile(diffs, 97.5)), 4)],
                'p_one_sided': round(float((diffs <= 0).mean()), 4),
                'n_boot': n_boot}

    L1, L4 = a.label_s1, a.label_s4
    comp = {}
    for key in ('r250', 'r1000'):
        for x, y in ((f'{L1}-配对', f'{L1}-错配·对真C'),
                     (f'{L4}-配对', f'{L4}-错配·对真C'),
                     (f'{L1}-配对', f'{L1}-零条件'),
                     (f'{L4}-配对', f'{L4}-零条件'),
                     (f'{L1}-配对', f'{L4}-配对'),
                     (f'{L1}-配对', '人类（同 split 真谱）'),
                     (f'{L4}-配对', '人类（同 split 真谱）'),
                     (f'{L1}-配对·占用校准', f'{L1}-错配·占用校准'),
                     (f'{L4}-配对·占用校准', f'{L4}-错配·占用校准'),
                     (f'{L1}-配对·占用校准', f'{L4}-配对·占用校准')):
            comp[f'{x} − {y} @{key}'] = paired(x, y, key)
        for tag in (L1, L4):
            for k in SHIFTS:
                comp[f'{tag}-条件{k:+d}格·对其给定C − {tag}-配对 @{key}'] = paired(
                    f'{tag}-条件{k:+d}格·对其给定C', f'{tag}-配对', key)
                comp[f'{tag}-条件{k:+d}格·对真C − {tag}-配对 @{key}'] = paired(
                    f'{tag}-条件{k:+d}格·对真C', f'{tag}-配对', key)

    podd = {}
    for x, y in ((f'{L1}-配对', f'{L4}-配对'),
                 (f'{L1}-配对', '人类（同 split 真谱）'),
                 (f'{L4}-配对', '人类（同 split 真谱）'),
                 (f'{L1}-配对·占用校准', f'{L4}-配对·占用校准')):
        podd[f'{x} − {y}'] = {'paired': paired_podd(x, y), 'bootstrap': boot_podd(x, y, a.boot)}

    # ---- 条件因果效应（固定噪声，只改条件）----
    def _eff(p1, p0):
        return {'平均|ΔP|': round(float(np.abs(p1 - p0).mean()), 5),
                '二值翻转率': round(float(((p1 > a.threshold) != (p0 > a.threshold)).mean()), 4),
                '占用差': round(float((p1 > a.threshold).mean() - (p0 > a.threshold).mean()), 4)}

    eff = {}
    for tag in (L1, L4):
        eff[f'{tag}: 配对 vs 零条件'] = _eff(probs[f'{tag}-配对'], probs[f'{tag}-零条件'])
        eff[f'{tag}: 配对 vs 错配（换内容）'] = _eff(probs[f'{tag}-配对'], probs[f'{tag}-错配'])
        for k in SHIFTS:
            eff[f'{tag}: 配对 vs 条件{k:+d}格（相位）'] = _eff(
                probs[f'{tag}-配对'], probs[f'{tag}-条件{k:+d}格'])

    res = {'config': {'data': a.data, 'cond_key': a.cond_key, 'cond_channels': cin,
                      'which_split': a.which_split, 'num_windows': len(idx),
                      'seed': a.seed, 'threshold': a.threshold,
                      'ckpt_s1': a.ckpt_s1, 'ckpt_s4': a.ckpt_s4,
                      'label_s1': a.label_s1, 'label_s4': a.label_s4,
                      'diff_norm': [lo, hi], 'occ_target_human': round(occ_target, 4),
                      'null_p_odd_hi_uniform': round(null_p, 4),
                      'null_p_odd_hi_even': round(float(np.mean(null_even)), 4),
                      'calib_threshold': {k: round(v, 4) for k, v in calib_t.items()},
                      'note': ('条件=32 带 mel onset 通量，总强度=32 带求和；'
                               '错配=打乱 song/start_cell 的条件；'
                               '「对其给定C」= 用平移/错配后喂给模型的那个条件来算相关，'
                               '「对真C」= 用本窗真条件算相关。')},
           'model_info': {L1: i1, L4: i4},
           'windows': [{'idx': int(i), 'song': int(song[i]), 'start_cell': int(start[i]),
                        'bpm': round(float(bpm[i]), 2), 'occ': round(float(occ[i]), 4),
                        'diff': round(float(diffv[j]), 4),
                        'cond_from_idx': int(idx[perm[j]])}
                       for j, i in enumerate(idx)],
           'summary': summary, 'compare': comp, 'p_odd_hi': podd,
           'cond_effect': eff}

    outp = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(outp) or '.', exist_ok=True)

    def _san(o):
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
    order = ['人类（同 split 真谱）']
    for tag in (L1, L4):
        order += [f'{tag}-配对', f'{tag}-错配·对真C', f'{tag}-错配·对其给定C',
                  f'{tag}-零条件', f'{tag}-配对·占用校准',
                  f'{tag}-配对·事后错配C']
        for k in SHIFTS:
            order += [f'{tag}-条件{k:+d}格·对其给定C', f'{tag}-条件{k:+d}格·对真C']
    print('\n=== 采音指标（r 中位；p_odd_hi 池化；1 格≈40ms）===')
    print(f'{"组":34s} {"r250":>7s} {"r1000":>7s} {"p_odd_hi":>9s} {"占用":>6s} {"音符":>6s}')
    for tag in order:
        s = summary[tag]
        print(f'{tag:34s} {s["r250_中位"]:>7.3f} {s["r1000_中位"]:>7.3f} '
              f'{str(s["p_odd_hi_池化"]):>9s} {s["占用"]:>6.3f} {s["音符总数"]:>6d}')

    print('\n=== r250 格移位扫描中位（k=0 为真对齐）===')
    print(f'{"组":34s} ' + ' '.join(f'{k:>7d}' for k in CELL_SHIFTS))
    for tag in order:
        sc = summary[tag]['r250_格移位扫描中位']
        if tag != '人类（同 split 真谱）' and '对其给定C' in tag:
            continue
        print(f'{tag:34s} ' + ' '.join(
            (f'{sc[str(k)]:>7.3f}' if sc[str(k)] is not None else f'{"-":>7s}')
            for k in CELL_SHIFTS))

    print('\n=== 逐窗配对比较（均值差 / n / 胜-负 / Wilcoxon p）===')
    for k, v in comp.items():
        if v:
            print(f'  {k}: {v["diff_mean"]:+.4f}（n={v["n"]}，{v["wins"]}胜{v["losses"]}负'
                  + (f'，p={v["wilcoxon_p"]:.3g}' if 'wilcoxon_p' in v else '') + ')')

    print('\n=== p_odd_hi（1/32 踩格）===')
    for k, v in podd.items():
        p, b = v['paired'], v['bootstrap']
        if p:
            print(f'  {k}: 池化 {p["pooled_x"]} vs {p["pooled_y"]}（差 {p["pooled_diff"]:+.4f}）；'
                  f'逐窗 = {p["diff_mean"]:+.4f}（{p["wins"]}胜{p["losses"]}负'
                  + (f'，p={p["wilcoxon_p"]:.3g}' if 'wilcoxon_p' in p else '') + '）')
            print(f'      bootstrap: 均值 {b["boot_mean"]:+.4f}，95% CI '
                  f'[{b["ci95"][0]:+.4f}, {b["ci95"][1]:+.4f}]，单侧 p={b["p_one_sided"]:.4f}')

    print('\n=== 条件因果效应（固定初始噪声，只改条件）===')
    for k, v in eff.items():
        print(f'  {k}: 平均|ΔP| {v["平均|ΔP|"]:.4f}  二值翻转率 {v["二值翻转率"]:.3f}  '
              f'占用差 {v["占用差"]:+.4f}')
    print(f'\n写出 {a.out}')


if __name__ == '__main__':
    main()
