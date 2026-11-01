#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""密度修复版扩散训练脚本（只做「提高采样占位率」这一件事）。

为什么单独开一个文件
--------------------
1. `tools/train-diffusion.py` 正被别的子代理同时修改（加音频条件分支），
   直接在它上面改会互相覆盖。
2. 更重要的是：**旧脚本有一个会改变结论的 bug** ——
       diff.loss(model, X[idx], S[idx], D[idx])          # 训练用默认 mode='eps'
   `Diffusion.loss` 的 mode 默认是 'eps'，训练循环没有把 `a.pred` 传进去，
   而验证循环（eval_loss）传了。于是：
       `--pred x0` 时 → 训练目标是 eps、验证损失量的是 x0-MSE；
       checkpoint 里却写着 pred='x0'，采样器就按 x0 去解释一个 eps 网络的输出。
   这就是「训练损失 0.29 / 验证损失 0.85 且在涨」的真正原因（两个损失量的
   根本不是同一个量），并不是过拟合。本脚本把 mode 显式传进去。

与官方训练脚本的兼容性
----------------------
- 模型结构（UNet1D/Block）与改动前**逐字节等价**：dropout 是 nn.Identity 或
  nn.Dropout（无参数），不改变 state_dict，`tools/sample-diffusion.py` 仍然能
  strict=True 加载。
- 不引入 cond 分支（那是另一个子代理在做的事）。
- checkpoint 里存 pred/base/temb_dim/timesteps，采样器直接读。

新增的能力
----------
- `--dropout`：Block 里加 dropout（无参数，兼容旧采样器）
- `--wd`：weight decay
- `--ema`：权重 EMA（CPU 上代价 ~0，通常对采样质量有正收益）
- `--eval-samples N`：**每个 epoch 真的跑一遍 DDPM 采样**，直接量「采样占位率」
  在 0.30/0.35/0.40/0.45/0.50 五条阈值上的值。这是唯一的验收指标，
  比验证 MSE 更能说明问题；num=2 时一次约 13s，可接受。
- `--det-tail K`：采样后 K 步（t<K）用确定性均值推进，便于在训练期对比。
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 128
NLANES = 4
# 阈值下探到 0.05：x0 回归在初期会「收缩」（预测均值偏低），需要低阈值才看得见结构
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


# ---------------------------------------------------------------------------
# 模型（与 tools/train-diffusion.py 改动前的 UNet1D 等价）
# ---------------------------------------------------------------------------
def temb_fn(dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)

    def f(t):
        a = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=1)
    return f


class Block(nn.Module):
    def __init__(self, cin, cout, temb_dim, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cin)
        self.norm2 = nn.GroupNorm(8, cout)
        self.temb = nn.Linear(temb_dim, cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, te):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(te))[:, :, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, cin=4, base=48, temb_dim=128, n_style=4, n_diff=4,
                 dropout=0.0):
        super().__init__()
        self.temb_fn = temb_fn(temb_dim)
        self.c_style = nn.Embedding(n_style, temb_dim)
        self.c_diff = nn.Embedding(n_diff, temb_dim)
        self.temb = nn.Sequential(nn.Linear(temb_dim, temb_dim), nn.SiLU(),
                                  nn.Linear(temb_dim, temb_dim))
        self.inp = nn.Conv1d(cin, base, 3, padding=1)
        self.d1 = Block(base, base, temb_dim, dropout)
        self.d2 = Block(base, base * 2, temb_dim, dropout)
        self.d3 = Block(base * 2, base * 2, temb_dim, dropout)
        self.mid = Block(base * 2, base * 2, temb_dim, dropout)
        self.u3 = Block(base * 4, base * 2, temb_dim, dropout)
        self.u2 = Block(base * 3, base, temb_dim, dropout)
        self.out = nn.Conv1d(base, cin, 3, padding=1)
        self.down = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, style, diff):
        te = self.temb(self.temb_fn(t) + self.c_style(style) + self.c_diff(diff))
        h = self.inp(x)
        e0 = self.d1(h, te)
        e1 = self.d2(self.down(e0), te)
        e2 = self.d3(self.down(e1), te)
        m = self.mid(e2, te)
        u3 = self.u3(torch.cat([self.up(m), e1], dim=1), te)
        u2 = self.u2(torch.cat([self.up(u3), e0], dim=1), te)
        return self.out(u2)


class Diffusion:
    def __init__(self, steps=200):
        self.T = steps
        self.betas = torch.linspace(1e-4, 0.02, steps)
        self.alphas = 1 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)

    def q(self, x0, t, noise):
        a = self.abar[t].to(x0.device)[:, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def target(self, x0, t, noise, mode):
        a = self.abar[t].to(x0.device)[:, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        if mode == 'x0':
            return xt, x0
        if mode == 'v':
            return xt, a.sqrt() * noise - (1 - a).sqrt() * x0
        return xt, noise

    def loss(self, model, x0, style, diff, mode='eps'):
        # ★ 关键：mode 必须由调用方显式传入（旧脚本缺这一步）
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt, tgt = self.target(x0, t, noise, mode)
        pred = model(xt, t, style, diff)
        return F.mse_loss(pred, tgt)


# ---------------------------------------------------------------------------
# 采样（与 tools/sample-diffusion.py 的 ddpm_sample 数学一致，只多了 det-tail）
# ---------------------------------------------------------------------------
@torch.no_grad()
def ddpm_sample(model, diff, n, style_id, diff_id, pred='eps', det_tail=0,
                x0_clamp=(-1.0, 2.0)):
    model.eval()
    x = torch.randn(n, NLANES, WIN)
    sty = torch.full((n,), int(style_id), dtype=torch.long)
    dif = torch.full((n,), int(diff_id), dtype=torch.long)
    T = diff.T
    for t in reversed(range(T)):
        tb = torch.full((n,), t, dtype=torch.long)
        out = model(x, tb, sty, dif)
        a_t, ab = diff.alphas[t], diff.abar[t]
        if pred == 'x0':
            x0 = out.clamp(*x0_clamp)
        elif pred == 'v':
            x0 = ab.sqrt() * x - (1 - ab).sqrt() * out
        else:
            x0 = (x - (1 - ab).sqrt() * out) / ab.sqrt()
        if t > 0:
            ab_prev = diff.abar[t - 1]
            mean = (ab_prev.sqrt() * diff.betas[t] * x0
                    + a_t.sqrt() * (1 - ab_prev) * x) / (1 - ab)
            if t < det_tail:
                x = mean                     # 确定性尾巴：不加噪
            else:
                sigma = (diff.betas[t] * (1 - diff.abar[t - 1]) / (1 - ab)).sqrt()
                x = mean + sigma * torch.randn_like(x)
        else:
            x = x0                            # t=0 直接取 x0 预测
    return x


def occ_at(P, th):
    return float((P > th).mean())


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='out/diff-data/dataset.npz')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--wd', type=float, default=0.01, help='weight decay')
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--ema', type=float, default=0.999, help='0=不用 EMA')
    ap.add_argument('--base', type=int, default=48)
    ap.add_argument('--timesteps', type=int, default=200)
    ap.add_argument('--pred', choices=['eps', 'x0', 'v'], default='x0')
    ap.add_argument('--max-windows', type=int, default=0)
    ap.add_argument('--out', default='out/models/diffusion4k-x0full.pt')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--val-frac', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-samples', type=int, default=2,
                    help='每轮真的采样几个窗口量占位率（0=不量）')
    ap.add_argument('--eval-style', type=int, default=1)
    ap.add_argument('--eval-diff', type=int, default=2)
    ap.add_argument('--det-tail', type=int, default=0,
                    help='采样时 t<该步数用确定性均值（0=纯祖先采样）')
    ap.add_argument('--time-budget', type=int, default=0)
    ap.add_argument('--min-epochs', type=int, default=0)
    ap.add_argument('--log', default='')
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print(f'线程 {torch.get_num_threads()}｜pred={a.pred} base={a.base} '
          f'batch={a.batch} lr={a.lr} wd={a.wd} dropout={a.dropout} ema={a.ema}',
          flush=True)

    d = np.load(a.data)
    X = torch.from_numpy(d['X'].astype(np.float32))
    S = torch.from_numpy(d['style'].astype(np.int64))
    D = torch.from_numpy(d['diff'].astype(np.int64))
    n_all = len(X)
    gsub = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(n_all, generator=gsub)
    if a.max_windows and a.max_windows < n_all:
        order = order[:a.max_windows]
    n = len(order)
    k = int(n * (1 - a.val_frac))
    tr, te = order[:k], order[k:]
    print(f'数据 全集 {n_all}，本配置用 {n} 窗口，训练 {len(tr)} / 验证 {len(te)}', flush=True)
    print(f'  训练集风格 {torch.bincount(S[tr], minlength=4).tolist()}'
          f' 难度 {torch.bincount(D[tr], minlength=4).tolist()}', flush=True)
    print(f'  数据占位率 {X.mean().item():.4f}（参考目标）', flush=True)

    model = UNet1D(base=a.base, dropout=a.dropout)
    nparam = sum(p.numel() for p in model.parameters())
    print('参数量 %.3fM' % (nparam / 1e6), flush=True)
    diff = Diffusion(a.timesteps)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    ema = None
    if a.ema > 0:
        ema = {kk: v.detach().clone().float() for kk, v in model.state_dict().items()
               if v.dtype.is_floating_point}

    gval = torch.Generator().manual_seed(1234)
    vt = torch.randint(0, a.timesteps, (len(te),), generator=gval)
    vn = torch.randn(len(te), 4, WIN, generator=gval)
    Xte, Ste, Dte = X[te], S[te], D[te]

    def eval_loss(mdl):
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(te), 512):
                m = min(512, len(te) - i)
                sl = slice(i, i + m)
                xt = diff.q(Xte[sl], vt[sl], vn[sl])
                pred = mdl(xt, vt[sl], Ste[sl], Dte[sl])
                _, tgt = diff.target(Xte[sl], vt[sl], vn[sl], a.pred)
                tot += F.mse_loss(pred, tgt).item() * m
        return tot / len(te)

    # EMA 权重装回同一个结构，方便每轮量它自己的验证损失
    ema_model = None
    if ema is not None:
        ema_model = UNet1D(base=a.base, dropout=a.dropout)

    def sync_ema():
        ema_model.load_state_dict({kk: vv for kk, vv in ema.items()}, strict=False)
        ema_model.eval()
        return ema_model

    def eval_density(tag_model):
        """真采样 → 连续 x0 → 各阈值下的占位率。返回 dict。

        采样用固定种子，但**先把全局 RNG 存起来、量完恢复**，
        否则后面每轮的 batch 排列与噪声会完全相同（等于每轮重复同一批数据）。
        """
        rng_backup = torch.get_rng_state()
        torch.manual_seed(777)
        P = ddpm_sample(tag_model, diff, a.eval_samples, a.eval_style, a.eval_diff,
                        pred=a.pred, det_tail=a.det_tail).numpy()
        torch.manual_seed(777)
        r = {'占位率': {f'{th:.2f}': round(occ_at(P, th), 4) for th in THRESHOLDS},
             '每行音符数': {f'{th:.2f}': round(float((P > th).sum(1).mean()), 3)
                          for th in THRESHOLDS},
             'x0均值': round(float(P.mean()), 4), 'x0标准差': round(float(P.std()), 4)}
        torch.set_rng_state(rng_backup)
        return r

    gperm = torch.Generator().manual_seed(a.seed + 991)   # 每轮重新打乱用
    best = 1e9
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        p = tr[torch.randperm(len(tr), generator=gperm)]
        tot, nb, t_first = 0.0, 0, 0.0
        for i in range(0, len(p), a.batch):
            idx = p[i:i + a.batch]
            tb = time.time()
            opt.zero_grad()
            loss = diff.loss(model, X[idx], S[idx], D[idx], mode=a.pred)   # ★ 修复点
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    msd = model.state_dict()
                    for kk, vv in ema.items():
                        vv.mul_(a.ema).add_(msd[kk].float(), alpha=1 - a.ema)
            tot += loss.item() * len(idx)
            nb += 1
            if nb == 1:
                t_first = (time.time() - tb) * 1000
        sched.step()
        model.eval()
        vl = eval_loss(model)
        if ema_model is not None:
            vl_ema = eval_loss(sync_ema())
        else:
            vl_ema = vl
        tr_l = tot / len(p)
        ep_s = time.time() - t0

        # 采样评估跑「将被存盘的那份权重」（有 EMA 就是 EMA）
        eval_model = sync_ema() if ema_model is not None else model
        t_s = time.time()
        dens = eval_density(eval_model) if a.eval_samples else {}
        ds = time.time() - t_s

        print(f'epoch {ep}: 训练 {tr_l:.4f}  验证 {vl:.4f}（EMA {vl_ema:.4f}）  '
              f'用时 {ep_s:.0f}s（{nb} batch，{t_first:.0f}ms/batch）'
              + (f'｜采样 {ds:.0f}s 占位率 {dens.get("占位率")} '
                 f'x0均值 {dens.get("x0均值")}' if dens else ''), flush=True)
        if a.log:
            os.makedirs(os.path.dirname(a.log) or '.', exist_ok=True)
            with open(a.log, 'a') as f:
                f.write(json.dumps({'epoch': ep, 'train': tr_l, 'val': vl,
                                    'val_ema': vl_ema,
                                    'sec': round(ep_s, 1), 'batches': nb,
                                    'ms_per_batch': round(t_first),
                                    'lr': round(sched.get_last_lr()[0], 8),
                                    'density': dens,
                                    'cfg': {'pred': a.pred, 'base': a.base,
                                            'batch': a.batch, 'lr': a.lr, 'wd': a.wd,
                                            'dropout': a.dropout, 'ema': a.ema,
                                            'windows': n, 'det_tail': a.det_tail}},
                                   ensure_ascii=False) + '\n')
        if vl < best:
            best = vl
            os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
            state = {kk: vv.clone() for kk, vv in ema.items()} if ema is not None \
                else model.state_dict()
            torch.save({'state': state, 'state_raw': model.state_dict(),
                        'pred': a.pred, 'base': a.base,
                        'temb_dim': 128, 'timesteps': a.timesteps, 'val': vl,
                        'val_ema': vl_ema,
                        'epoch': ep, 'batch': a.batch, 'rows': X.shape[2],
                        'windows': n, 'params': nparam, 'dropout': a.dropout,
                        'ema': a.ema, 'data': a.data, 'val_frac': a.val_frac,
                        'seed': a.seed, 'density_at_save': dens,
                        'train_seconds': round(time.time() - t_start, 1)}, a.out)
            print(f'  ↳ 保存最优（验证 {vl:.4f}）', flush=True)
        if a.time_budget and ep >= a.min_epochs and (time.time() - t_start) > a.time_budget:
            print(f'时间预算 {a.time_budget}s 到，停在 epoch {ep}', flush=True)
            break
    print(f'完成，最优验证损失 {best:.4f}，总用时 {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
