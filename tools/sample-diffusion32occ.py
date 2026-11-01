#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4K 谱面扩散模型的 DDPM 采样器（CPU 版）—— **1/32 网格版**。

本文件是 tools/sample-diffusion.py 的副本（不改共享文件），改动：
  1. WIN 128 → 256（1/32 网格，与 1/16 窗口同样时长）；
  2. 格时长 = 60000/bpm/**8**（1/32 格 = 1/16 的一半），密度统计按 256 格算；
  3. 统计里直接给 **1/32 占比**（相邻行间隔 = 1 格的相对比例，与
     tools/eval-generated.py 的 p132 同口径）和半格内容占比，便于和人类 0.300 对比。

把 `tools/train-diffusion32.py` 训出来的 U-Net 从纯噪声开始逐步去噪，
产出 [4,256] 二值钢琴卷（1/32 网格），并打印基本统计。

设计取舍说明
------------
1. **不 import 训练脚本**：`tools/train-diffusion.py` 正在被别人改（加速训练），
   所以这里把 UNet1D / Block / Diffusion / temb_fn 按「当前版本」整段复制过来，
   训练脚本再怎么动都不会把采样器改坏。
2. **超参从 ckpt 里读**：`base` / `temb_dim` / `timesteps` 优先取 checkpoint 里存的，
   取不到就从 state_dict 的权重形状反推（训练脚本改宽度也能救回来）。
3. **内存**：一次最多采 8 个窗口（--num 1..8），模型只有 0.2~0.4M 参数，
   128 行的窗口在 CPU 上 200 步采样是秒级。

用法
----
  # 没有训练好的权重时，用随机初始化权重把整条流程测通
  python3 tools/sample-diffusion.py --random-weights --style 0 --diff 1 --num 4

  # 有 out/models/diffusion4k.pt 时（默认路径）
  python3 tools/sample-diffusion.py --style 2 --diff 3 --num 8 --outdir out/diffusion/samples

  # 带音频条件的权重（训练时用了 --audio-cond）：同一个起点的音频段做条件
  python3 tools/sample-diffusion.py --ckpt out/models/diffusion-cond.pt \
      --cond out/multisong/Heart_Fraction-cond.npz --cond-index 64 --num 4

  # 也可以用 cond-windows.npz 的第 N 个窗口（自动找歌 + 起始格）
  python3 tools/sample-diffusion.py --ckpt out/models/diffusion-cond.pt \
      --cond out/diff-data/cond-windows.npz --cond-index 7 --num 4

  # 强制无条件（条件全零）采样，用来和「有条件」对比
  python3 tools/sample-diffusion.py --ckpt out/models/diffusion-cond.pt --cond-zero --num 4

输出（默认 out/diffusion/samples/）
  roll-00.npy ...      每个窗口一个 [4,128] uint8 钢琴卷（>0.5 → 1）
  rolls.npz            X=[n,4,128] uint8（阈值化）、P=[n,4,128] float32（原始概率）、
                       style/diff/seed 元数据
  sample-meta.json     本次采样的参数与统计

音频条件（可选，默认关闭）
-------------------------
训练脚本 `tools/train-diffusion.py --audio-cond ...` 训出来的权重里多一个
`AudioCondNet`（`[B,128,4] → [B,cond_dim,128] → 加进 U-Net 中间层`）。
本脚本按 checkpoint 里的 `cond_dim` 决定要不要建这个分支：
    cond_dim=0（旧权重）→ 结构与以前完全一致，strict 加载照旧；
    cond_dim>0          → 建分支，--cond 给条件；不给就喂全零（等于无条件采样）。
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 256            # 钢琴卷长度：256 个 1/32 格 = 8 小节（与旧 1/16 窗口同时长）
NLANES = 4           # 4K
MAX_NUM = 8          # 一次最多采 8 个窗口（内存/耗时上限）


# ---------------------------------------------------------------------------
# 以下四段从 tools/train-diffusion.py 原样复制（只读该文件，不 import、不改它）
# ---------------------------------------------------------------------------
def temb_fn(dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)

    def f(t):
        a = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=1)
    return f


class Block(nn.Module):
    def __init__(self, cin, cout, temb_dim, up=False):
        super().__init__()
        self.up = up
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cin)
        self.norm2 = nn.GroupNorm(8, cout)
        self.temb = nn.Linear(temb_dim, cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(te))[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AudioCondNet(nn.Module):
    """音频条件分支（从 tools/train-diffusion.py 复制，结构必须逐字段一致）。

    [B,L,4] → 逐位置特征 [B,cond_dim,L] → kernel=stride=4 卷积 → [B,mid_ch,L//4]
    加进 U-Net 中间层。结构与 tools/train-diffusion.py 的 AudioCondNet 逐字段一致
    （初始化用 PyTorch 默认，加载权重时会被覆盖，见训练脚本里的注释）。
    """

    def __init__(self, cin=4, cond_dim=32, mid_ch=96, downsample=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cond_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(cond_dim, cond_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.proj = nn.Conv1d(cond_dim, mid_ch, kernel_size=downsample,
                              stride=downsample)
        # 投影层初始化：默认 std≈0.051 时，条件分支对中间层的贡献只有 8%，
        # 144 步训练根本推不动（见 AUDIO-BRANCH.md 的消融表）。std=0.3 让
        # 初始贡献约 28%，换条件就能让输出动 ~4.7%，训练里条件才学得起来。
        nn.init.normal_(self.proj.weight, std=0.3)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond):
        """cond: [B,L,4] → [B,mid_ch,L//4]"""
        f = self.net(cond.transpose(1, 2))    # [B,cond_dim,L]
        return self.proj(f)                    # [B,mid_ch,L//4]


class UNet1D(nn.Module):
    def __init__(self, cin=4, base=48, temb_dim=128, n_style=4, n_diff=4,
                 cond_dim=0):
        super().__init__()
        self.temb_fn = temb_fn(temb_dim)
        self.c_style = nn.Embedding(n_style, temb_dim)
        # 与训练侧同名持久 buffer；禁止丢掉 checkpoint 的 Fourier 基底。
        self.register_buffer('freq', torch.randn(temb_dim // 2) * 3.0)
        self.c_diff = nn.Sequential(nn.Linear(temb_dim, temb_dim), nn.SiLU(),
                                    nn.Linear(temb_dim, temb_dim))
        self.temb = nn.Sequential(nn.Linear(temb_dim, temb_dim), nn.SiLU(),
                                  nn.Linear(temb_dim, temb_dim))
        self.inp = nn.Conv1d(cin, base, 3, padding=1)
        self.d1 = Block(base, base, temb_dim)          # 128
        self.d2 = Block(base, base * 2, temb_dim)      # 64
        self.d3 = Block(base * 2, base * 2, temb_dim)  # 32
        self.mid = Block(base * 2, base * 2, temb_dim)
        self.u3 = Block(base * 4, base * 2, temb_dim)  # cat(m↑, e1)
        self.u2 = Block(base * 3, base, temb_dim)      # cat(u3↑, e0)
        self.out = nn.Conv1d(base, cin, 3, padding=1)
        self.down = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # cond_dim=0 → 不注册音频分支参数，旧权重 strict 加载不受影响
        self.cond_dim = int(cond_dim)
        self.cond_net = (AudioCondNet(cin=4, cond_dim=self.cond_dim,
                                      mid_ch=base * 2, downsample=4)
                         if self.cond_dim > 0 else None)

    def forward(self, x, t, style, diff, cond=None):
        d = diff.float()[:, None]
        feats = torch.cat([torch.sin(d * self.freq[None, :]), torch.cos(d * self.freq[None, :])], dim=1)
        te = self.temb(self.temb_fn(t) + self.c_style(style) + self.c_diff(feats))
        h = self.inp(x)                                # [B,b,128]
        e0 = self.d1(h, te)                            # [B,b,128]
        e1 = self.d2(self.down(e0), te)                # [B,2b,64]
        e2 = self.d3(self.down(e1), te)                # [B,2b,32]
        m = self.mid(e2, te)                           # [B,2b,32]
        if self.cond_net is not None:
            if cond is None:
                cond = x.new_zeros(x.shape[0], x.shape[2], 4)   # 没有条件 → 全零
            m = m + self.cond_net(cond)                # [B,2b,32]，逐位置相加
        u3 = self.u3(torch.cat([self.up(m), e1], dim=1), te)   # [B,2b,64]
        u2 = self.u2(torch.cat([self.up(u3), e0], dim=1), te)  # [B,b,128]
        return self.out(u2)


class Diffusion:
    def __init__(self, steps=200, device='cpu'):
        self.T = steps
        self.betas = torch.linspace(1e-4, 0.02, steps)
        self.alphas = 1 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)

    def q(self, x0, t, noise):
        a = self.abar[t].to(x0.device)[:, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, model, x0, style, diff):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q(x0, t, noise)
        pred = model(xt, t, style, diff)
        return F.mse_loss(pred, noise)


def prediction_to_x0(xt, output, abar, pred):
    """反解训练参数化；不裁剪，以便独立验证公式。abar 可为标量或可广播 Tensor。"""
    if pred == 'x0':
        return output
    if pred == 'v':
        return abar.sqrt() * xt - (1 - abar).sqrt() * output
    if pred == 'eps':
        return (xt - (1 - abar).sqrt() * output) / abar.sqrt()
    raise ValueError(f'未知预测目标 {pred!r}，必须是 eps/x0/v')


# ---------------------------------------------------------------------------
# 采样（本文件新增）
# ---------------------------------------------------------------------------
@torch.no_grad()
def ddpm_sample(model, diff, n, style_id, diff_id, device='cpu', verbose=True, pred=None,
                cond=None):
    """标准 DDPM 祖先采样：x_T ~ N(0,I) → x_0。

    pred 与训练一致：eps（预测噪声）/ x0（直接预测干净谱面）/ v（速度）
    cond：可选 [n,128,4] 音频条件（模型的 cond_dim>0 时才有意义；None = 全零条件）
    返回 [n, 4, 128] 的「近似 x0」（连续值，未阈值化）。
    """
    if not 0.0 <= float(diff_id) <= 1.0:
        raise ValueError('连续难度条件必须是 [0,1] 内的有限浮点数')
    stored_pred = getattr(model, 'prediction_type', None)
    pred = stored_pred if pred is None else pred
    if pred not in ('eps', 'x0', 'v'):
        raise ValueError('必须明确指定预测目标 eps/x0/v（或从 checkpoint 恢复）')
    if stored_pred is not None and pred != stored_pred:
        raise ValueError(f'预测目标 {pred!r} 与 checkpoint {stored_pred!r} 不一致')
    model.eval()
    x = torch.randn(n, NLANES, WIN, device=device)
    sty = torch.full((n,), int(style_id), dtype=torch.long, device=device)
    dif = torch.full((n,), float(diff_id), dtype=torch.float32, device=device)
    if cond is not None:
        cond = torch.as_tensor(cond, dtype=torch.float32, device=device)
        if cond.shape != (n, WIN, 4):
            raise SystemExit(f'[错误] 条件形状 {tuple(cond.shape)}，期望 {(n, WIN, 4)}')
    T = diff.T
    t_start = time.time()
    report_at = max(1, T // 10)
    for t in reversed(range(T)):
        tb = torch.full((n,), t, dtype=torch.long, device=device)
        out = model(x, tb, sty, dif, cond)
        a_t = diff.alphas[t]
        ab = diff.abar[t]
        x0 = prediction_to_x0(x, out, ab, pred)
        if pred == 'x0':
            x0 = x0.clamp(-1.0, 2.0)
        # 标准 DDPM 后验均值：μ = (√ᾱ_{t-1}β_t x0 + √α_t(1-ᾱ_{t-1}) x_t) / (1-ᾱ_t)
        if t > 0:
            ab_prev = diff.abar[t - 1]
            mean = (ab_prev.sqrt() * diff.betas[t] * x0
                    + a_t.sqrt() * (1 - ab_prev) * x) / (1 - ab)
        else:
            mean = x0
        if t > 0:
            sigma = (diff.betas[t] * (1 - diff.abar[t - 1]) / (1 - ab)).sqrt()
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean
        if verbose and (t % report_at == 0 or t == T - 1):
            print(f'  去噪 t={t:4d}/{T}  用时 {time.time() - t_start:5.1f}s', flush=True)
    return x


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def roll_stats(roll_bin, probs=None, step_ms=None):
    """roll_bin: [n,4,128] 0/1。返回统计 dict。"""
    n = roll_bin.shape[0]
    row_sum = roll_bin.sum(axis=1)                     # [n,128] 每格（时间行）音符数
    lane_occ = roll_bin.mean(axis=(0, 2))              # [4] 每列占位率
    occ = float(roll_bin.mean())
    rows_nonempty = float((row_sum > 0).mean())
    chord = row_sum[row_sum > 0]
    st = {
        '窗口数': int(n),
        '占位率': round(occ, 4),
        '非空行比例': round(rows_nonempty, 4),
        '每行平均音符数': round(float(row_sum.mean()), 3),
        '非空行平均和弦': round(float(chord.mean()) if chord.size else 0.0, 3),
        '最大和弦': int(row_sum.max()) if row_sum.size else 0,
        '各列占位率': [round(float(v), 4) for v in lane_occ],
        '每窗口音符数': [int(v) for v in roll_bin.sum(axis=(1, 2))],
    }
    # 1/32 占比：相邻占用行间隔 = 1 格的比例（= 半个 1/16，与 eval-generated 的 p132 同口径）
    occ_idx = [np.where(row_sum[i] > 0)[0] for i in range(n)]
    gaps1 = tot_g = 0
    off_par = par_tot = 0
    for idx in occ_idx:
        if len(idx) < 2:
            continue
        d = np.diff(idx)
        gaps1 += int((d == 1).sum())
        tot_g += int(d.size)
        n_odd = int((idx % 2 == 1).sum())
        n_even = len(idx) - n_odd
        off_par += min(n_odd, n_even)      # 偏离窗口多数奇偶性的格 = 半格内容
        par_tot += len(idx)
    st['1/32占比_p132'] = round(gaps1 / max(1, tot_g), 4)
    st['半格内容_off_parity'] = round(off_par / max(1, par_tot), 4)
    if step_ms:
        # 1 个格 = step_ms 毫秒 → 音符密度（每秒）
        sec = WIN * step_ms / 1000.0
        st['每窗口时长_s'] = round(sec, 3)
        st['音符密度_nps'] = round(float(roll_bin.sum(axis=(1, 2)).mean()) / sec, 3)
        st['行密度_rps'] = round(float((row_sum > 0).sum(axis=1).mean()) / sec, 3)
    if probs is not None:
        p = probs.reshape(-1)
        st['原始概率'] = {
            '均值': round(float(p.mean()), 4),
            '标准差': round(float(p.std()), 4),
            '最小': round(float(p.min()), 4),
            '最大': round(float(p.max()), 4),
            'q10': round(float(np.quantile(p, 0.10)), 4),
            'q50': round(float(np.quantile(p, 0.50)), 4),
            'q90': round(float(np.quantile(p, 0.90)), 4),
        }
    return st


def ascii_roll(roll, rows=None):
    """把 [4,128] 二值卷打成 4 列 ASCII，方便肉眼看键型。"""
    out = []
    lanes = 'ABCD'
    out.append('      ' + ' '.join(lanes))
    n = roll.shape[1] if rows is None else min(rows, roll.shape[1])
    for k in range(n):
        cell = ['#' if roll[c, k] > 0 else '.' for c in range(NLANES)]
        out.append(f'{k:4d}  ' + ' '.join(cell))
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# 权重加载
# ---------------------------------------------------------------------------
def load_model(ckpt, random_weights, base, timesteps, threads, cond_dim=0, pred=None):
    """返回 (model, Diffusion, info dict)，只允许可完整恢复的连续 Fourier 权重。"""
    torch.set_num_threads(max(1, threads))
    if pred is not None and pred not in ('eps', 'x0', 'v'):
        raise ValueError(f'未知预测目标 {pred!r}，必须是 eps/x0/v')
    if random_weights:
        if timesteps <= 0:
            raise ValueError('timesteps 必须 > 0')
        model = UNet1D(cin=NLANES, base=base, cond_dim=cond_dim)
        model.prediction_type = pred or 'eps'
        diff = Diffusion(timesteps)
        info = {'权重': '随机初始化（--random-weights）', 'base': base,
                'timesteps': timesteps, 'cond_dim': int(cond_dim),
                'pred': model.prediction_type}
        return model, diff, info

    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError(f'{ckpt}: 找不到 state_dict')
    state = ck.get('state', ck)
    meta = ck if 'state' in ck else {}
    if not isinstance(state, dict) or not state:
        raise ValueError(f'{ckpt}: state 必须是非空 state_dict')
    if meta.get('model_type', 'unet1d-continuous-fourier-v1') != 'unet1d-continuous-fourier-v1':
        raise ValueError(f'{ckpt}: 未知模型结构 {meta["model_type"]!r}')
    if 'freq' not in state:
        raise ValueError(f'{ckpt}: 缺少 freq，无法恢复训练时的 Fourier 条件；不允许随机补齐')
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f'{ckpt}: {key} 不是浮点权重 Tensor')
        if not torch.isfinite(value).all():
            raise ValueError(f'{ckpt}: {key} 含 NaN/Inf，拒绝损坏权重')

    def shape(key, ndim):
        if key not in state or state[key].ndim != ndim:
            raise ValueError(f'{ckpt}: 未知或损坏的连续模型结构：{key}')
        return state[key].shape

    def integer(key, default, minimum=1):
        value = meta.get(key, default)
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError(f'{ckpt}: {key} 必须是整数') from e
        if isinstance(value, bool) or result != value or result < minimum:
            raise ValueError(f'{ckpt}: {key} 必须是 >= {minimum} 的整数')
        return result

    inferred_base, cin, _ = shape('inp.weight', 3)
    n_style, inferred_temb = shape('c_style.weight', 2)
    base = integer('base', inferred_base)
    temb_dim = integer('temb_dim', inferred_temb)
    inferred_cond = shape('cond_net.proj.weight', 3)[1] if 'cond_net.proj.weight' in state else 0
    cond_dim = integer('cond_dim', inferred_cond, 0)
    if (base != inferred_base or temb_dim != inferred_temb or cond_dim != inferred_cond
            or cin != NLANES or n_style != 4 or base % 8 or temb_dim % 2):
        raise ValueError(f'{ckpt}: 模型元数据与权重结构不一致，或不是支持的 4K 连续模型')
    if shape('freq', 1) != (temb_dim // 2,):
        raise ValueError(f'{ckpt}: freq 形状与 temb_dim 不一致')
    if integer('rows', WIN) != WIN or meta.get('grid', '1/32') != '1/32':
        raise ValueError(f'{ckpt}: 窗口/网格元数据不适用于 1/32、{WIN} 格采样器')
    if integer('cond_channels', 4 if cond_dim else 0, 0) != (4 if cond_dim else 0):
        raise ValueError(f'{ckpt}: 音频条件元数据与分支结构不一致')
    timesteps = integer('timesteps', timesteps)
    stored_pred = meta.get('pred')
    if stored_pred is not None and stored_pred not in ('eps', 'x0', 'v'):
        raise ValueError(f'{ckpt}: 未知 checkpoint 预测目标 {stored_pred!r}')
    if stored_pred is not None and pred is not None and pred != stored_pred:
        raise ValueError(f'{ckpt}: --pred {pred!r} 与 checkpoint {stored_pred!r} 不一致')
    resolved_pred = stored_pred if stored_pred is not None else pred
    if resolved_pred is None:
        raise ValueError(f'{ckpt}: 缺少 pred 元数据，请明确提供 --pred eps/x0/v；不猜测')

    model = UNet1D(cin=cin, base=base, temb_dim=temb_dim, n_style=n_style, cond_dim=cond_dim)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise ValueError(f'{ckpt}: strict 加载失败，禁止跳过任何权重/缓冲区：{e}') from e
    model.prediction_type = resolved_pred
    diff = Diffusion(timesteps)
    info = {'权重': str(ckpt), '加载方式': 'strict', 'base': base,
            'temb_dim': temb_dim, 'timesteps': timesteps, 'cond_dim': cond_dim,
            'pred': resolved_pred, 'difficulty_condition': meta.get('difficulty_condition')}
    if 'val' in meta:
        info['训练验证损失'] = round(float(meta['val']), 5)
    return model, diff, info


def load_cond_segment(path, index, cond_root='out/multisong', win=WIN):
    """读出 [win,4] float32 的一段音频条件。两种文件都支持：

      - 单歌条件文件 out/multisong/<歌名>-cond.npz：键 feat [T,4]，index = 起始格；
      - cond-windows.npz：键 song/start/songs，index = 窗口序号（自带歌名与起始格）。

    越界/不够长在尾部补零，采样不会因为边界报错。
    """
    cd = np.load(path, allow_pickle=True)
    keys = set(cd.keys())
    if 'feat' in keys:
        f = cd['feat'].astype(np.float32)
        start = int(index)
        tag = f'{os.path.basename(path)} 起始格 {start}'
    elif {'song', 'start', 'songs'} <= keys:
        i = int(index)
        n_win = len(cd['song'])
        if not (0 <= i < n_win):
            print(f'[错误] --cond-index {i} 超出 {path} 的窗口数 {n_win}', file=sys.stderr)
            sys.exit(2)
        name = str(cd['songs'][int(cd['song'][i])])
        start = int(cd['start'][i])
        f = np.load(os.path.join(cond_root, f'{name}-cond.npz'))['feat'].astype(np.float32)
        tag = f'窗口 {i} → {name} 起始格 {start}'
    else:
        print(f'[错误] {path} 里没有 feat，也不是 cond-windows 格式', file=sys.stderr)
        sys.exit(2)

    seg = f[start:start + win]
    if len(seg) < win:
        seg = np.concatenate([seg, np.zeros((win - len(seg), 4), np.float32)])
    return seg.astype(np.float32), tag


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description='4K 谱面扩散模型 DDPM 采样器')
    ap.add_argument('--ckpt', default='out/models/diffusion4k.pt', help='权重路径')
    ap.add_argument('--random-weights', action='store_true',
                    help='不加载权重，用随机初始化权重采样（没有训练产物时测通流程用）')
    ap.add_argument('--style', type=int, default=0, choices=[0, 1, 2, 3],
                    help='风格条件：0=tech 1=jack 2=speed 3=stamina')
    ap.add_argument('--diff', type=float, default=0.5, help='连续占位率条件 0..1（不是已标定的 Koishi 难度）')
    ap.add_argument('--pred', choices=['eps', 'x0', 'v'], default=None,
                    help='缺少 pred 元数据时必须指定；有元数据时仅允许一致的目标')
    ap.add_argument('--num', type=int, default=4, choices=list(range(1, MAX_NUM + 1)),
                    help=f'采样窗口数 1..{MAX_NUM}（内存上限）')
    ap.add_argument('--timesteps', type=int, default=200, help='扩散步数（ckpt 里有时以 ckpt 为准）')
    ap.add_argument('--base', type=int, default=48, help='U-Net 基础通道（--random-weights 时生效）')
    ap.add_argument('--cond-dim', type=int, default=0,
                    help='音频分支通道数（只配合 --random-weights 测形状用；真权重从 ckpt 读）')
    ap.add_argument('--threshold', type=float, default=0.5, help='阈值化门限，>该值视为有音符')
    ap.add_argument('--target-occ', type=float, default=0.0,
                    help='目标占位率（0=用固定阈值）。给了就按概率分位数反解阈值，使解码占位率≈该值')
    ap.add_argument('--seed', type=int, default=0, help='随机种子')
    ap.add_argument('--threads', type=int, default=1, help='torch 线程数（默认 CPU 单线程）')
    ap.add_argument('--bpm', type=float, default=136.3, help='算步长/密度用，1/32 格 = 60/bpm/8 秒')
    ap.add_argument('--outdir', default='out/diffusion32/samples', help='输出目录')
    ap.add_argument('--ascii', type=int, default=0, help='打印前 N 个窗口的 ASCII 钢琴卷（0=不打印）')
    ap.add_argument('--quiet', action='store_true', help='不打印去噪进度')
    # ---- 音频条件（只有 cond_dim>0 的权重才用得上的可选参数）----
    ap.add_argument('--cond', default='',
                    help='音频条件文件：单歌 <歌名>-cond.npz（用 feat）或 cond-windows.npz（按窗口序号）')
    ap.add_argument('--cond-index', type=int, default=0,
                    help='条件下的起始格（feat 文件）或窗口序号（cond-windows.npz）')
    ap.add_argument('--cond-root', default='out/multisong',
                    help='cond-windows.npz 配对时单歌 cond 文件所在目录')
    ap.add_argument('--cond-zero', action='store_true',
                    help='强制零条件（无条件采样），用于和「有条件」对比')
    a = ap.parse_args(argv)
    if not 0.0 <= a.diff <= 1.0:
        ap.error('--diff 必须是 [0,1] 内的有限浮点数')

    if a.num > MAX_NUM:
        print(f'[错误] --num 最大 {MAX_NUM}', file=sys.stderr)
        sys.exit(2)
    if not (0.0 < a.threshold < 1.0):
        print('[错误] --threshold 要在 (0,1) 之间', file=sys.stderr)
        sys.exit(2)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    try:
        model, diff, info = load_model(a.ckpt, a.random_weights, a.base, a.timesteps,
                                       a.threads, a.cond_dim, pred=a.pred)
    except (OSError, ValueError, RuntimeError) as e:
        ap.error(str(e))
    n_par = sum(p.numel() for p in model.parameters())
    step_ms = 60000.0 / max(1e-6, a.bpm) / 8.0      # 1/32 格 = 1/16 的一半

    print(f'▶ 采样：style={a.style} diff={a.diff} num={a.num} T={diff.T} '
          f'base={info["base"]} 参数 {n_par / 1e6:.2f}M')
    print(f'  权重：{info["权重"]}')
    print(f'  1/32 格 = {step_ms:.2f} ms（bpm {a.bpm}），窗口 {WIN} 格 = '
          f'{WIN * step_ms / 1000:.2f}s')

    # ---------------- 音频条件（可选）----------------
    cdim = int(info.get('cond_dim', 0))
    cond_t, cond_info, cond_tag = None, None, ''
    if cdim > 0:
        if a.cond and not a.cond_zero:
            seg, cond_tag = load_cond_segment(a.cond, a.cond_index, a.cond_root)
            cond_t = torch.from_numpy(seg)[None].repeat(a.num, 1, 1)
            cond_info = {'文件': a.cond, 'index': int(a.cond_index),
                         '定位': cond_tag,
                         '通道均值': [round(float(v), 4) for v in seg.mean(0)]}
            print(f'  音频条件：{cond_tag}；通道均值 {cond_info["通道均值"]}')
        else:
            cond_t = torch.zeros(a.num, WIN, 4)
            cond_info = {'模式': '全零' + ('（--cond-zero）' if a.cond_zero else '（没给 --cond）')}
            print('  音频条件：全零（无条件模式）')
    elif a.cond or a.cond_zero:
        print('[错误] 权重没有音频分支（cond_dim=0），--cond 用不上；'
              '请用训练时加过 --audio-cond 的权重', file=sys.stderr)
        sys.exit(2)

    cond_sfx = '' if cond_t is None else (f'-c{a.cond_index}' if a.cond and not a.cond_zero else '-czero')

    t0 = time.time()
    pred_mode = info['pred']
    raw = ddpm_sample(model, diff, a.num, a.style, a.diff,
                      verbose=not a.quiet, pred=pred_mode, cond=cond_t).numpy()
    cost = time.time() - t0

    probs = raw.astype(np.float32)               # [n,4,128] 连续值
    thr_used = a.threshold
    if a.target_occ > 0:
        if not (0.0 < a.target_occ < 0.95):
            print('[错误] --target-occ 要在 (0,0.95) 之间', file=sys.stderr)
            sys.exit(2)
        # 按全局分位数反解：解码后占位率 ≈ target_occ
        thr_used = float(np.quantile(probs, 1.0 - a.target_occ))
        print(f'[占位率标定] 目标 {a.target_occ:.4f} → 阈值 {thr_used:.4f}（固定阈值 {a.threshold:.2f} 作废）')
    roll_bin = (probs > thr_used).astype(np.uint8)

    st = roll_stats(roll_bin, probs=probs, step_ms=step_ms)
    st['耗时_s'] = round(cost, 2)
    st['每窗口耗时_s'] = round(cost / max(1, a.num), 2)
    st['扩散步数'] = diff.T
    st['阈值'] = thr_used
    st['阈值来源'] = 'target-occ' if a.target_occ > 0 else 'fixed'
    if a.target_occ > 0:
        st['目标占位率'] = a.target_occ

    print('\n=== 采样统计 ===')
    for k, v in st.items():
        print(f'  {k}: {v}')

    # ---------------- 落盘 ----------------
    os.makedirs(a.outdir, exist_ok=True)
    # 文件名里带 rnd / 条件标记，避免不同权重/条件互相覆盖
    stem = (f's{a.style}d{a.diff}-seed{a.seed}' + cond_sfx
            + ('-rnd' if a.random_weights else ''))
    files = []
    for i in range(a.num):
        p = os.path.join(a.outdir, f'roll-{stem}-{i:02d}.npy')
        np.save(p, roll_bin[i])
        files.append(p)
    npz_path = os.path.join(a.outdir, f'rolls-{stem}.npz')
    extra = {'C': cond_t[:1].numpy()} if cond_t is not None else {}
    np.savez_compressed(npz_path, X=roll_bin, P=probs,
                        style=np.int64(a.style), diff=np.float64(a.diff),
                        pred=np.str_(pred_mode),
                        seed=np.int64(a.seed), threshold=np.float64(thr_used),
                        **extra)
    meta = {
        'style': a.style, 'diff': a.diff, 'num': a.num, 'seed': a.seed,
        'threshold': thr_used, 'target_occ': a.target_occ, 'bpm': a.bpm, 'step_ms': round(step_ms, 4),
        'win': WIN, 'timesteps': diff.T, 'random_weights': bool(a.random_weights),
        'pred': pred_mode, 'weights': info, 'cond': cond_info, 'stats': st,
        'files': [os.path.basename(f) for f in files],
    }
    meta_path = os.path.join(a.outdir, f'sample-meta-{stem}.json')
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    if a.ascii > 0:
        for i in range(min(a.ascii, a.num)):
            print(f'\n=== 窗口 {i}（A/B/C/D = x 64/192/320/448）===')
            print(ascii_roll(roll_bin[i]))

    print(f'\n写出 {len(files)} 个 [4,{WIN}] 钢琴卷 → {a.outdir}/roll-{stem}-*.npy')
    print(f'        打包概率卷 → {npz_path}')
    print(f'        统计元数据 → {meta_path}')
    print(f'\n下一步：python3 tools/roll-to-osu32.py {a.outdir}/roll-{stem}-*.npy '
          f'--concat --bpm {a.bpm} -o out/diffusion32/{stem}.osu   # 1/32 转换器')


if __name__ == '__main__':
    main()
