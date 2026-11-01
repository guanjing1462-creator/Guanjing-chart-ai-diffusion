#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音频条件分支是否真的起作用？—— 三组客观检验（只用验证窗口，不碰训练集）。

直接复用 `tools/sample-diffusion.py` 里的 UNet1D / Diffusion / ddpm_sample（同一份代码，
避免结构漂移），对比两个权重的 checkpoint：
    A = 训练时加过 --audio-cond 的（有音频分支）
    B = 同配置、没加 --audio-cond 的（无条件）

检验 1（分布指标）：多噪声抽样验证损失
    在验证窗口上换 10 组固定 (t, 噪声)，分别算 A / B 的去噪 MSE。
    配对比较（同一组噪声、同一批窗口）：A 是否系统性更低。

检验 2（单步去噪）：条件是否把 x0 预测拉向真实谱面
    固定 t、固定噪声，x_t = √ā·x0 + √(1-ā)·ε。
    对 A：用「本窗口自己的条件」vs「别的窗口的条件（错配）」各反推一次 x0，
    和真实谱面比 F1 / MSE。B 没有条件分支，错配前后必须**逐位相同**（负对照）。

检验 3（真采样）：条件换掉，生成的谱面会不会变；条件强度是否传到密度上
    固定初始噪声，分别用匹配条件 / 置换条件采样（T=200 DDPM）。
    - A：两次结果应明显不同；B：必须完全相同（负对照）。
    - 再把 4 个 onset 最强的窗口条件 vs 4 个最弱的窗口条件做一批采样，
      看生成音符密度是否跟着条件强度走（这是「条件有语义」的直接证据）。

输出：out/diffusion/eval-audio-cond.json + 终端表格。

用法：
  python3 tools/eval-audio-cond.py \
      --cond-ckpt out/models/diff-cond-seed0.pt \
      --nocond-ckpt out/models/diff-nocond-seed0.pt \
      --threads 1
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

TOOLS = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('sample_diffusion',
                                             os.path.join(TOOLS, 'sample-diffusion.py'))
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)          # 只加载定义，main() 有 __main__ 保护不会跑

WIN = sd.WIN


def load_ckpt(path, threads):
    model, diff, info = sd.load_model(path, False, 48, 200, threads)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return model, diff, info, ck


def val_loss_multi(model, diff, X, S, D, cond, draws=10, seed=999):
    """[draws] 个固定噪声下的验证 MSE（与训练脚本的 val-fixed 同一把尺子，只是多抽几次）。"""
    g = torch.Generator().manual_seed(seed)
    T = diff.T
    vals = []
    for _ in range(draws):
        t = torch.randint(0, T, (len(X),), generator=g)
        noise = torch.randn(X.shape, generator=g)
        a = diff.abar[t][:, None, None]
        xt = a.sqrt() * X + (1 - a).sqrt() * noise
        with torch.no_grad():
            pred = model(xt, t, S, D, cond)
        vals.append(F.mse_loss(pred, noise).item())
    return vals


def sample_batch(model, diff, cond, seed, pred='eps', style=0, diff_id=1):
    """固定种子采一批（同样的种子 ⇒ 同样的初始噪声）。"""
    torch.manual_seed(int(seed))
    if cond is None:
        return sd.ddpm_sample(model, diff, 1, style, diff_id, verbose=False, pred=pred).numpy()
    return sd.ddpm_sample(model, diff, len(cond), style, diff_id, verbose=False,
                          pred=pred, cond=cond).numpy()


def roll_metrics(gen, gt, thr=0.5):
    """gen: [n,4,128] 连续值；gt: [n,4,128] 0/1。"""
    b = gen > thr
    g = gt > 0.5
    ep = 1e-9
    tp = float((b & g).sum())
    fp = float((b & ~g).sum())
    fn = float((~b & g).sum())
    out = {
        'f1': 2 * tp / (2 * tp + fp + fn + ep),
        'iou': tp / (tp + fp + fn + ep),
        'mse': float(((gen - gt) ** 2).mean()),
        'occ': float(b.mean()),
        'occ_gt': float(g.mean()),
    }
    # 行剖面相关：每格音符数的 128 维形状（只看节奏轮廓是否对上）
    rg = b.sum(axis=1).astype(np.float64)      # [n,128]
    rt = g.sum(axis=1).astype(np.float64)
    cors = []
    for i in range(len(rg)):
        if rg[i].std() > 1e-6 and rt[i].std() > 1e-6:
            cors.append(float(np.corrcoef(rg[i], rt[i])[0, 1]))
    out['row_corr'] = float(np.mean(cors)) if cors else 0.0
    return out


def pair_stats(x, y):
    """配对差值 x - y：均值 / 标准差 / 变好比例。"""
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {'mean': float(d.mean()), 'sd': float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            'n_better': int((d > 0).sum()), 'n': int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond-ckpt', default='out/models/diff-cond-seed0.pt')
    ap.add_argument('--nocond-ckpt', default='out/models/diff-nocond-seed0.pt')
    ap.add_argument('--data', default='out/diff-data/audio-cond-labels.npz')
    ap.add_argument('--windows', default='out/diff-data/cond-windows.npz')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--draws', type=int, default=10, help='检验 1 的噪声抽样次数')
    ap.add_argument('--t-onestep', type=int, default=100, help='检验 2 用的加噪步 t')
    ap.add_argument('--n-sample', type=int, default=8, help='检验 3 用几个验证窗口')
    ap.add_argument('--n-extreme', type=int, default=4, help='每组取几个 onset 极值窗口')
    ap.add_argument('--seed', type=int, default=0, help='采样用的初始噪声种子')
    ap.add_argument('--out', default='out/diffusion/eval-audio-cond.json')
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    t_start = time.time()

    d = np.load(a.data, allow_pickle=True)
    X = torch.from_numpy(d['X'].astype(np.float32))
    S = torch.from_numpy(d['style'].astype(np.int64))
    D = torch.from_numpy(d['diff'].astype(np.int64))
    C = torch.from_numpy(d['cond'].astype(np.float32))          # [N,128,4]
    Sg = d['song'] if 'song' in d else None
    St = d['start'] if 'start' in d else None

    modelA, diffA, infoA, ckA = load_ckpt(a.cond_ckpt, a.threads)
    modelB, diffB, infoB, ckB = load_ckpt(a.nocond_ckpt, a.threads)
    print(f'A = {a.cond_ckpt}  cond_dim={infoA["cond_dim"]}  ckpt_val={infoA.get("训练验证损失")}')
    print(f'B = {a.nocond_ckpt}  cond_dim={infoB["cond_dim"]}  ckpt_val={infoB.get("训练验证损失")}')

    te = ckA.get('val_idx')
    if te is None:
        te = list(range(len(X)))[-max(4, len(X) // 10):]
    te = list(map(int, te))
    print(f'验证窗口 {len(te)} 个（从 A 的 ckpt 里读 val_idx；两模型同一切分）')
    Xte, Ste, Dte, Cte = X[te], S[te], D[te], C[te]

    res = {'A': {'ckpt': a.cond_ckpt, 'cond_dim': infoA['cond_dim'],
                 'val': infoA.get('训练验证损失')},
           'B': {'ckpt': a.nocond_ckpt, 'cond_dim': infoB['cond_dim'],
                 'val': infoB.get('训练验证损失')},
           'val_windows': te}

    # ---------------- 检验 1：多噪声验证损失 ----------------
    print('\n[1] 验证损失（%d 组固定噪声，配对）' % a.draws)
    la = np.array(val_loss_multi(modelA, diffA, Xte, Ste, Dte, Cte, draws=a.draws))
    lb = np.array(val_loss_multi(modelB, diffB, Xte, Ste, Dte, None, draws=a.draws))
    st = {'A_mean': float(la.mean()), 'A_sd': float(la.std(ddof=1)),
          'B_mean': float(lb.mean()), 'B_sd': float(lb.std(ddof=1)),
          'delta': pair_stats(lb, la),        # 正数 = A 更低（更好）
          'A_draws': la.tolist(), 'B_draws': lb.tolist()}
    print(f'  A {la.mean():.4f} ± {la.std(ddof=1):.4f}   B {lb.mean():.4f} ± {lb.std(ddof=1):.4f}')
    print(f'  配对差 (B−A) = {st["delta"]["mean"]:+.4f} ± {st["delta"]["sd"]:.4f}，'
          f'A 更低的噪声组 {st["delta"]["n_better"]}/{st["delta"]["n"]}')
    res['val_loss_multi'] = st

    # ---------------- 检验 2：单步去噪（条件 vs 错配条件）----------------
    print(f'\n[2] t={a.t_onestep} 单步反推 x0：匹配条件 vs 置换条件')
    g = torch.Generator().manual_seed(4242)
    tval = torch.full((len(te),), int(a.t_onestep), dtype=torch.long)
    noise = torch.randn(Xte.shape, generator=g)
    ab_t = diffA.abar[int(a.t_onestep)]
    x_t = ab_t.sqrt() * Xte + (1 - ab_t).sqrt() * noise
    # 置换：按条件强度排序后错开一半 —— 尽量把「最像」的条件换成「最不像」的
    cm_te = Cte.mean(dim=(1, 2)).numpy()
    ordr = np.argsort(cm_te)
    perm = torch.from_numpy(np.roll(ordr, len(ordr) // 2).copy())
    Cperm = Cte[perm]
    pred_mode_A = ckA.get('pred', 'eps')

    def to_x0(out, x):
        if pred_mode_A == 'x0':
            return out.clamp(-1.0, 2.0)
        if pred_mode_A == 'v':
            return ab_t.sqrt() * x - (1 - ab_t).sqrt() * out
        return (x - (1 - ab_t).sqrt() * out) / ab_t.sqrt()

    with torch.no_grad():
        outA_match = modelA(x_t, tval, Ste, Dte, Cte)
        outA_perm = modelA(x_t, tval, Ste, Dte, Cperm)
        outB_match = modelB(x_t, tval, Ste, Dte, None)
        outB_perm = modelB(x_t, tval, Ste, Dte, None)
    x0A_match, x0A_perm = to_x0(outA_match, x_t), to_x0(outA_perm, x_t)
    x0B = to_x0(outB_match, x_t)
    mA = roll_metrics(x0A_match.numpy(), Xte.numpy())
    mA_p = roll_metrics(x0A_perm.numpy(), Xte.numpy())
    mB = roll_metrics(x0B.numpy(), Xte.numpy())
    same_B = float((outB_match - outB_perm).abs().max())
    print(f'  A 匹配条件 F1 {mA["f1"]:.4f}  MSE {mA["mse"]:.4f}    '
          f'置换条件 F1 {mA_p["f1"]:.4f}  MSE {mA_p["mse"]:.4f}')
    print(f'  B（无条件，负对照）匹配/置换两次输出最大差 {same_B:.2e}（应为 0）')
    # 逐窗口配对 F1
    f1a, f1p = [], []
    for i in range(len(te)):
        f1a.append(roll_metrics(x0A_match[i:i + 1].numpy(), Xte[i:i + 1].numpy())['f1'])
        f1p.append(roll_metrics(x0A_perm[i:i + 1].numpy(), Xte[i:i + 1].numpy())['f1'])
    res['one_step'] = {
        't': int(a.t_onestep),
        'A_match': mA, 'A_perm': mA_p, 'B': mB,
        'B_max_abs_diff': same_B,
        'A_f1_paired': pair_stats(f1a, f1p),
    }
    print(f'  逐窗口 F1（匹配 − 置换）= {np.mean(f1a)-np.mean(f1p):+.4f}，'
          f'匹配更好的窗口 {sum(np.array(f1a)>np.array(f1p))}/{len(te)}')

    # ---------------- 检验 2b：因果干预（固定 x_t，只换条件强度）----------------
    # 同一批 x_t（验证窗口 @ t），把所有窗口的条件统一换成「最弱」/「最强」的几段，
    # 看预测 x0 的占位率有没有被条件推动 —— x_t 完全不变，所以差异只可能来自条件。
    print(f'\n[2b] 因果干预：x_t 不变，把条件统一换成最弱 / 最强的几段（t={a.t_onestep}）')
    cmean_all = C.mean(dim=(1, 2))
    order = torch.argsort(cmean_all)
    weak = C[order[:a.n_extreme]].mean(dim=0, keepdim=True).repeat(len(te), 1, 1)
    strong = C[order[-a.n_extreme:]].mean(dim=0, keepdim=True).repeat(len(te), 1, 1)
    with torch.no_grad():
        x0_weak = to_x0(modelA(x_t, tval, Ste, Dte, weak), x_t)
        x0_strong = to_x0(modelA(x_t, tval, Ste, Dte, strong), x_t)
        x0B_weak = to_x0(modelB(x_t, tval, Ste, Dte, weak), x_t)
        x0B_strong = to_x0(modelB(x_t, tval, Ste, Dte, strong), x_t)
    occ_weak = float((x0_weak > 0.5).float().mean())
    occ_strong = float((x0_strong > 0.5).float().mean())
    occB_weak = float((x0B_weak > 0.5).float().mean())
    occB_strong = float((x0B_strong > 0.5).float().mean())
    d_stack = [(x0_strong[i] - x0_weak[i]).abs().mean().item() for i in range(len(te))]
    print(f'  A：最弱条件 → 预测占位率 {occ_weak:.4f}；最强条件 → {occ_strong:.4f}'
          f'（{occ_strong-occ_weak:+.4f}，相对 {100*(occ_strong-occ_weak)/occ_weak:+.1f}%）')
    print(f'  B（负对照）：{occB_weak:.4f} → {occB_strong:.4f}（差值必须为 0）')
    print(f'  每个窗口上两种条件的 x0 预测平均 |Δ| = {np.mean(d_stack):.4f}')
    res['causal_density'] = {
        't': int(a.t_onestep),
        'A_occ_weak': occ_weak, 'A_occ_strong': occ_strong,
        'A_delta': occ_strong - occ_weak,
        'A_rel': (occ_strong - occ_weak) / occ_weak if occ_weak else 0.0,
        'B_occ_weak': occB_weak, 'B_occ_strong': occB_strong,
        'per_window_mean_abs_diff': float(np.mean(d_stack)),
        'cond_mean_weak': float(cmean_all[order[:a.n_extreme]].mean()),
        'cond_mean_strong': float(cmean_all[order[-a.n_extreme:]].mean()),
    }

    # ---------------- 检验 3：真采样 ----------------
    ns = min(a.n_sample, len(te))
    print(f'\n[3a] 真采样（T={diffA.T}）：前 {ns} 个验证窗口，匹配条件 vs 置换条件（同一种子）')
    idx = list(range(ns))
    Cm = Cte[idx]
    # 置换：按条件强度排序后错开一半，保证「换掉的条件」和原来差得远
    ordr = np.argsort(Cm.mean(dim=(1, 2)).numpy())
    Cp = Cm[torch.from_numpy(np.roll(ordr, len(ordr) // 2).copy())]
    Xs = Xte[idx]
    t0 = time.time()
    genA_m = sample_batch(modelA, diffA, Cm, a.seed)
    genA_p = sample_batch(modelA, diffA, Cp, a.seed)
    costA = time.time() - t0
    print(f'  A 采样耗时 {costA:.0f}s')
    dA = np.abs(genA_m - genA_p)
    mA_m = roll_metrics(genA_m, Xs.numpy())
    mA_p = roll_metrics(genA_p, Xs.numpy())
    print(f'  A：匹配条件 F1 {mA_m["f1"]:.4f} / 置换 {mA_p["f1"]:.4f}；'
          f'两次生成的平均 |Δ| = {dA.mean():.4f}（最大 {dA.max():.3f}）')

    t0 = time.time()
    genB_1 = sample_batch(modelB, diffB, Cm, a.seed)
    genB_2 = sample_batch(modelB, diffB, Cp, a.seed)
    costB = time.time() - t0
    dB = np.abs(genB_1 - genB_2)
    mB_s = roll_metrics(genB_1, Xs.numpy())
    print(f'  B（负对照）：两次生成的平均 |Δ| = {dB.mean():.2e}（必须为 0）；'
          f'F1 {mB_s["f1"]:.4f}，耗时 {costB:.0f}s')

    # 密度是否跟着条件走：生成占位率 vs 真实占位率（配对），以及 vs 条件强度
    occ_m = (genA_m > 0.5).mean(axis=(1, 2))
    occ_p = (genA_p > 0.5).mean(axis=(1, 2))
    occ_gt = Xs.numpy().mean(axis=(1, 2))
    cm_m = Cm.mean(dim=(1, 2)).numpy()
    cm_p = Cp.mean(dim=(1, 2)).numpy()
    den_err_m = np.abs(occ_m - occ_gt)
    den_err_p = np.abs(occ_p - occ_gt)
    flip = float(((genA_m > 0.5) != (genA_p > 0.5)).mean())
    r_m = float(np.corrcoef(occ_m, occ_gt)[0, 1]) if np.std(occ_m) > 1e-9 else 0.0
    r_p = float(np.corrcoef(occ_p, occ_gt)[0, 1]) if np.std(occ_p) > 1e-9 else 0.0
    r_cond = float(np.corrcoef(occ_m, cm_m)[0, 1]) if np.std(occ_m) > 1e-9 else 0.0
    res['sample_match_vs_perm'] = {
        'n_windows': ns, 'seed': int(a.seed),
        'A': {'match': mA_m, 'perm': mA_p,
              'mean_abs_diff': float(dA.mean()), 'max_abs_diff': float(dA.max()),
              'cells_flipped_frac': flip,
              'occ_gen': occ_m.tolist(), 'occ_gen_perm': occ_p.tolist(),
              'occ_gt': occ_gt.tolist(),
              'density_err_match': float(den_err_m.mean()),
              'density_err_perm': float(den_err_p.mean()),
              'density_err_paired': pair_stats(-den_err_m, -den_err_p),
              'corr_occ_gt_match': r_m, 'corr_occ_gt_perm': r_p,
              'corr_occ_cond': r_cond,
              'seconds': round(costA, 1)},
        'B': {'f1': mB_s, 'mean_abs_diff': float(dB.mean()),
              'max_abs_diff': float(dB.max()), 'seconds': round(costB, 1)},
    }
    print(f'  A 两次生成中「阈值化后翻面的格子比例」= {flip*100:.1f}%')
    print(f'  A 生成占位率（匹配）{np.round(occ_m,3)}')
    print(f'  A 真实占位率              {np.round(occ_gt,3)}')
    print(f'  A 生成占位率（置换）{np.round(occ_p,3)}')
    print(f'  密度误差 |生成−真实|：匹配 {den_err_m.mean():.4f} / 置换 {den_err_p.mean():.4f}'
          f'；匹配更好的窗口 {int((den_err_m < den_err_p).sum())}/{ns}')
    print(f'  生成占位率 vs 真实占位率 r：匹配 {r_m:+.3f} / 置换 {r_p:+.3f}；'
          f'生成占位率 vs 条件强度 r {r_cond:+.3f}')

    # 3b：onset 极值条件 → 生成密度
    cmean = C.mean(dim=(1, 2)).numpy()             # 用 199 个窗口里条件最强的/最弱的
    order = np.argsort(cmean)
    lo_idx = torch.from_numpy(order[:a.n_extreme].copy())
    hi_idx = torch.from_numpy(order[-a.n_extreme:].copy())
    print(f'\n[3b] 条件强度 → 生成密度：最弱 {a.n_extreme} 个 vs 最强 {a.n_extreme} 个条件窗口')
    t0 = time.time()
    gen_lo = sample_batch(modelA, diffA, C[lo_idx], a.seed + 100)
    gen_hi = sample_batch(modelA, diffA, C[hi_idx], a.seed + 100)
    costE = time.time() - t0
    occ_lo = float((gen_lo > 0.5).mean())
    occ_hi = float((gen_hi > 0.5).mean())
    occ_lo_w = [float((gen_lo[i] > 0.5).sum()) for i in range(len(lo_idx))]
    occ_hi_w = [float((gen_hi[i] > 0.5).sum()) for i in range(len(hi_idx))]
    print(f'  条件均值：弱 {cmean[lo_idx.numpy()].mean():.3f} / 强 {cmean[hi_idx.numpy()].mean():.3f}')
    print(f'  生成占位率：弱 {occ_lo:.4f} / 强 {occ_hi:.4f}；'
          f'音符数 弱 {occ_lo_w} 强 {occ_hi_w}（耗时 {costE:.0f}s）')
    res['cond_strength_to_density'] = {
        'n_each': int(a.n_extreme),
        'cond_mean_lo': float(cmean[lo_idx.numpy()].mean()),
        'cond_mean_hi': float(cmean[hi_idx.numpy()].mean()),
        'occ_lo': occ_lo, 'occ_hi': occ_hi,
        'notes_lo': occ_lo_w, 'notes_hi': occ_hi_w,
        'seconds': round(costE, 1),
    }

    # 3c：同一首歌两个不同 start 的条件
    song_arr = np.asarray(Sg) if Sg is not None else None
    if song_arr is not None:
        # 挑窗口最多的那首歌里条件差异最大的两个 start
        best = None
        for s in np.unique(song_arr):
            ii = np.where(song_arr == s)[0]
            if len(ii) < 2:
                continue
            sub = cmean[ii]
            pair = (ii[int(np.argmin(sub))], ii[int(np.argmax(sub))])
            score = float(cmean[pair[1]] - cmean[pair[0]])
            if best is None or score > best[0]:
                best = (score, s, pair)
        if best:
            score, s, (i0, i1) = best
            print(f'\n[3c] 同一首歌 {d["songs"][s] if "songs" in d else s} 的两个 start：'
                  f'{int(St[i0])} vs {int(St[i1])}（条件均值 {cmean[i0]:.3f} / {cmean[i1]:.3f}）')
            cc = torch.stack([C[i0], C[i1]])
            g2 = sample_batch(modelA, diffA, cc, a.seed + 200)
            dd = float(np.abs(g2[0] - g2[1]).mean())
            print(f'  两次生成平均 |Δ| = {dd:.4f}；音符数 {int((g2[0]>0.5).sum())} vs '
                  f'{int((g2[1]>0.5).sum())}')
            res['same_song_two_starts'] = {
                'song': str(d['songs'][s]) if 'songs' in d else str(s),
                'start_a': int(St[i0]), 'start_b': int(St[i1]),
                'cond_mean_a': float(cmean[i0]), 'cond_mean_b': float(cmean[i1]),
                'mean_abs_diff': dd,
                'notes_a': int((g2[0] > 0.5).sum()), 'notes_b': int((g2[1] > 0.5).sum()),
            }
    res['seconds'] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f'\n结果写出 → {a.out}（总耗时 {res["seconds"]}s）')


if __name__ == '__main__':
    main()
