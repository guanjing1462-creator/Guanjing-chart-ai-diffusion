#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 采样器：**多尺度全分辨率**条件注入（--inject multi）+ **32 带 mel onset 通量**条件。

本文件是 tools/sample-diffusion32c.py 的副本（旧采样器一字未改），改动只在
「条件分支」这一块：

  1. 内联 `MultiScaleCondNet`（从 tools/train-diffusion32-align.py 逐字段复制）：
     两层 Conv1d(k=3,s=1) 保持 256 位置，然后每层一个 `projs[i]` =
     `AvgPool1d(level_pool[i]) + Conv1d(cond_dim→level_ch[i], k=1)`，
     把条件加到 e0(256)/e1(128)/e2(64)/mid(64) 四个层级（最浅层 1 格 ≈40ms）。
  2. 结构/超参**全部从 checkpoint 元数据 + 权重形状**读：
     `inject / cond_key / cond_dim / cond_channels / cond_stride / base / timesteps`。
  3. **严格加载**：`load_state_dict(strict=True)`；`inject=='multi'` 必须与权重里的
     `cond_net.projs.*` 一致；`cond_key` 必须与给定条件一致；缺键/多键直接报错退出。
     绝不允许 strict=False —— 那会静默丢掉 cond_net.projs.*，条件失效（136 轮踩过）。

用法
----
  # S1 + 单首歌的 mel 条件（[T,32] 从 --start-cell 切 256 格，一次 6 个连续窗口）
  python3 tools/sample-diffusion32align.py --ckpt out/models/r153-S1-multi.pt \
      --cond-mel out/diffusion/round154-sampler/Heart-Fraction-mel.npz \
      --num 6 --start-cell 0 --style 0 --diff 0.35 --outdir out/diffusion/round154-sampler/paired

  # 旧的 4 带条件（S4 / 更早的 4 带权重）：--cond out/multisong/<歌>-cond.npz --cond-index N
  python3 tools/sample-diffusion32align.py --ckpt out/models/r153-S4-mid4.pt \
      --cond-mel <mel.npz> --num 4

  # 无条件（条件全零）对照
  python3 tools/sample-diffusion32align.py --ckpt out/models/r153-S1-multi.pt --cond-zero

条件形状（严格校验，不静默补零）
------------------------------
  mel：单个窗口 `[256,32]`，或多窗口 `[N,256,32]`，或整首 `[T,32]`（按 --start-cell 切）。
       通道数必须等于 ckpt 的 cond_channels；fin16/fp32 都行。
  4 带：`[256,4]`（单歌 feat 从 --cond-index 起切 256 格）。
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



class MultiScaleCondNet(nn.Module):
    """多尺度全分辨率条件分支（round 153 / S1）。

    与 tools/train-diffusion32-align.py 的 MultiScaleCondNet **逐字段一致**：

        cond [B,256,cin]
          → Conv1d(cin→cond_dim, k=3, s=1) → SiLU       # 保持 256 位置
          → Conv1d(cond_dim→cond_dim, k=3, s=1) → SiLU  # 保持 256 位置
          → 逐层 AvgPool1d(k) 降到该层长度 + 1×1 卷积对齐通道
          → 加到 e0(256)/e1(128)/e2(64)/mid(64)

    最浅层是 **256 位置**（1 格 = 1/32 格 ≈ 40ms），相位分辨率 ≈ 1 格。
    投影层 std=0.3 / bias 置零，与训练侧一致（加载时会被权重覆盖）。
    """

    def __init__(self, cin=32, cond_dim=32, level_ch=(48, 96, 96, 96),
                 level_pool=(1, 2, 4, 4)):
        super().__init__()
        self.cin = int(cin)
        assert len(level_ch) == len(level_pool)
        self.level_ch = tuple(int(c) for c in level_ch)
        self.level_pool = tuple(int(k) for k in level_pool)
        self.net = nn.Sequential(
            nn.Conv1d(cin, cond_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(cond_dim, cond_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.projs = nn.ModuleList([nn.Conv1d(cond_dim, c, 1) for c in self.level_ch])
        for p in self.projs:
            nn.init.normal_(p.weight, std=0.3)
            nn.init.zeros_(p.bias)

    def forward(self, cond):
        """cond: [B,L,cin] → list of [B,level_ch[i],L//level_pool[i]]"""
        f = self.net(cond.transpose(1, 2))          # [B,cond_dim,256]
        outs = []
        for k, proj in zip(self.level_pool, self.projs):
            g = f if k == 1 else F.avg_pool1d(f, k)
            outs.append(proj(g))
        return outs


class UNet1D(nn.Module):
    def __init__(self, cin=4, base=48, temb_dim=128, n_style=4, n_diff=4,
                 cond_dim=0, cond_ch=4, cond_stride=4, inject='mid4'):
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
        self.cond_ch = int(cond_ch)
        self.cond_stride = int(cond_stride)
        self.inject = str(inject)
        if self.cond_dim > 0 and self.inject == 'multi':
            self.cond_net = MultiScaleCondNet(
                cin=self.cond_ch, cond_dim=self.cond_dim,
                level_ch=(base, base * 2, base * 2, base * 2),
                level_pool=(1, 2, 4, 4))
        elif self.cond_dim > 0:
            self.cond_net = AudioCondNet(cin=self.cond_ch, cond_dim=self.cond_dim,
                                         mid_ch=base * 2,
                                         downsample=self.cond_stride)
        else:
            self.cond_net = None

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
                cond = x.new_zeros(x.shape[0], x.shape[2], self.cond_ch)  # 无条件 → 全零
            if self.inject == 'multi':
                c0, c1, c2, c3 = self.cond_net(cond)
                e0 = e0 + c0                       # [B,b,256]   ← 1 格分辨率
                e1 = e1 + c1                       # [B,2b,128]
                e2 = e2 + c2                       # [B,2b,64]
                m = m + c3                         # [B,2b,64]
            else:
                cv = self.cond_net(cond)
                if self.cond_stride == 4:
                    m = m + cv                     # [B,2b,64]
                elif self.cond_stride == 2:
                    e1 = e1 + cv                   # [B,2b,128]
                else:
                    e0 = e0 + cv                   # [B,b,256]
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
        cch = int(getattr(model, 'cond_ch', 4))
        if getattr(model, 'cond_net', None) is not None:
            cch = int(model.cond_net.cin)
        if cond.shape != (n, WIN, cch):
            raise SystemExit(f'[错误] 条件形状 {tuple(cond.shape)}，期望 {(n, WIN, cch)}')
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
def load_model(ckpt, random_weights, base, timesteps, threads, cond_dim=0, pred=None,
               cond_ch=None, inject=None, cond_key=None):
    """返回 (model, Diffusion, info dict)，只允许可完整恢复的连续 Fourier 权重。

    S1（inject='multi'）的 cond_net.projs.* 必须逐个被严格加载：先逐键比对
    `模型键集合 vs checkpoint 键集合`，缺键/多键直接报错，再 `strict=True` 加载。
    绝不使用 strict=False —— 那会静默丢掉 projs.*，条件失效而采样照跑（136 轮的坑）。
    """
    torch.set_num_threads(max(1, threads))
    if pred is not None and pred not in ('eps', 'x0', 'v'):
        raise ValueError(f'未知预测目标 {pred!r}，必须是 eps/x0/v')
    if random_weights:
        if timesteps <= 0:
            raise ValueError('timesteps 必须 > 0')
        model = UNet1D(cin=NLANES, base=base, cond_dim=cond_dim,
                       cond_ch=(cond_ch or 4), inject=(inject or 'mid4'))
        model.prediction_type = pred or 'eps'
        diff = Diffusion(timesteps)
        info = {'权重': '随机初始化（--random-weights）', 'base': base,
                'timesteps': timesteps, 'cond_dim': int(cond_dim),
                'cond_ch': model.cond_ch, 'inject': model.inject,
                'cond_key': cond_key, 'pred': model.prediction_type}
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

    # ---- 条件分支结构：只从权重形状 + 元数据推，不接受「猜」----
    if 'cond_net.projs.0.weight' in state:
        inferred_inject = 'multi'
        inferred_cond = shape('cond_net.projs.0.weight', 3)[1]
        inferred_ch = shape('cond_net.net.0.weight', 3)[1]
        level_ch = [shape(f'cond_net.projs.{i}.weight', 3)[0] for i in range(4)]
        if tuple(level_ch) != (base, base * 2, base * 2, base * 2):
            raise ValueError(f'{ckpt}: projs 通道 {level_ch} 与 base={base} 的 '
                             f'MultiScaleCondNet 结构不符')
        inferred_stride = 1
        expect_scales = [256, 128, 64, 64]
    elif 'cond_net.proj.weight' in state:
        inferred_inject = 'mid4'
        inferred_cond = shape('cond_net.proj.weight', 3)[1]
        inferred_ch = shape('cond_net.net.0.weight', 3)[1]
        inferred_stride = shape('cond_net.proj.weight', 3)[2]
        level_ch = [base * 2]
        expect_scales = [64]
    else:
        inferred_inject = 'none'
        inferred_cond, inferred_ch, inferred_stride = 0, 0, 0
        level_ch, expect_scales = [], []
    if meta.get('inject', inferred_inject) != inferred_inject:
        raise ValueError(f'{ckpt}: 元数据 inject={meta.get("inject")!r} 与权重结构 '
                         f'{inferred_inject!r} 不一致（结构必须与训练侧逐字段一致）')
    cond_dim = integer('cond_dim', inferred_cond, 0)
    cond_ch = integer('cond_channels', inferred_ch, 0)
    cond_stride = integer('cond_stride', inferred_stride, 0 if inferred_cond == 0 else 1)
    if (cond_dim != inferred_cond or cond_ch != inferred_ch
            or (inferred_cond and cond_stride != inferred_stride)):
        raise ValueError(f'{ckpt}: 条件元数据与权重结构不一致 '
                         f'(cond_dim={cond_dim}/{inferred_cond}, '
                         f'cond_channels={cond_ch}/{inferred_ch}, '
                         f'cond_stride={cond_stride}/{inferred_stride})')
    if inferred_inject == 'multi' and meta.get('inject_scales') not in (None, expect_scales):
        raise ValueError(f'{ckpt}: inject_scales={meta.get("inject_scales")!r} '
                         f'与 multi 的 {expect_scales} 不一致')
    stored_cb = meta.get('cond_key')
    if inferred_cond > 0:
        if stored_cb is None:
            raise ValueError(f'{ckpt}: 条件权重缺少 cond_key 元数据，无法校验条件来源')
        if cond_key is not None and str(stored_cb) != str(cond_key):
            raise ValueError(f'{ckpt}: checkpoint 的 cond_key={stored_cb!r} 与给定条件 '
                             f'{cond_key!r} 不一致（拒绝用错条件的权重采样）')
    elif cond_key is not None:
        raise ValueError(f'{ckpt}: 权重没有条件分支（cond_dim=0），却给了 --cond-key {cond_key!r}')

    if (base != inferred_base or temb_dim != inferred_temb or cin != NLANES
            or n_style != 4 or base % 8 or temb_dim % 2):
        raise ValueError(f'{ckpt}: 模型元数据与权重结构不一致，或不是支持的 4K 连续模型')
    if shape('freq', 1) != (temb_dim // 2,):
        raise ValueError(f'{ckpt}: freq 形状与 temb_dim 不一致')
    if integer('rows', WIN) != WIN or meta.get('grid', '1/32') != '1/32':
        raise ValueError(f'{ckpt}: 窗口/网格元数据不适用于 1/32、{WIN} 格采样器')
    timesteps = integer('timesteps', timesteps)
    stored_pred = meta.get('pred')
    if stored_pred is not None and stored_pred not in ('eps', 'x0', 'v'):
        raise ValueError(f'{ckpt}: 未知 checkpoint 预测目标 {stored_pred!r}')
    if stored_pred is not None and pred is not None and pred != stored_pred:
        raise ValueError(f'{ckpt}: --pred {pred!r} 与 checkpoint {stored_pred!r} 不一致')
    resolved_pred = stored_pred if stored_pred is not None else pred
    if resolved_pred is None:
        raise ValueError(f'{ckpt}: 缺少 pred 元数据，请明确提供 --pred eps/x0/v；不猜测')

    model = UNet1D(cin=cin, base=base, temb_dim=temb_dim, n_style=n_style,
                   cond_dim=cond_dim, cond_ch=(cond_ch or 4),
                   cond_stride=(cond_stride or 4), inject=inferred_inject)
    # ---- 逐键比对（缺键/多键在这里就报错，给出具体键名）----
    want, have = set(model.state_dict()), set(state)
    missing, extra = sorted(want - have), sorted(have - want)
    if missing or extra:
        raise ValueError(
            f'{ckpt}: 权重键不匹配，拒绝 strict=False 静默丢条件分支。'
            f'缺键 {missing[:8]}{"…" if len(missing) > 8 else ""}；'
            f'多键 {extra[:8]}{"…" if len(extra) > 8 else ""}')
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise ValueError(f'{ckpt}: strict 加载失败，禁止跳过任何权重/缓冲区：{e}') from e
    model.prediction_type = resolved_pred
    diff = Diffusion(timesteps)
    info = {'权重': str(ckpt), '加载方式': 'strict（逐键比对 + strict=True）',
            'base': base, 'temb_dim': temb_dim, 'timesteps': timesteps,
            'cond_dim': cond_dim, 'cond_ch': cond_ch, 'cond_stride': cond_stride,
            'inject': inferred_inject, 'cond_key': stored_cb,
            'inject_scales': meta.get('inject_scales'),
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

    if f.ndim != 2 or f.shape[1] != 4:
        print(f'[错误] {path}: 4 带条件必须是 [T,4]，实际 {tuple(f.shape)}；'
              f'mel 条件请用 --cond-mel', file=sys.stderr)
        sys.exit(2)
    seg = f[start:start + win]
    if len(seg) < win:
        seg = np.concatenate([seg, np.zeros((win - len(seg), 4), np.float32)])
    return seg.astype(np.float32), tag


def load_mel_cond(path, num, start_cell, cond_ch, win=WIN, keys=('M', 'cond', 'feat')):
    """mel onset 通量条件 → [num, win, cond_ch] float32 + 定位串。**严格形状校验**。

    支持三种输入（fin16/fp32 均可，逐通道必须与 ckpt 的 cond_channels 一致）：
      - [win,32]      单窗口条件，--num 必须为 1（否则报错，不静默重复）；
      - [N,win,32]    每窗口一条条件，取第 0..num-1 窗（N 必须 >= num）；
      - [T,32]        整首歌，窗口 i 从 --start-cell + i*win 切；不够长直接报错。
    不做补零：条件缺失就是错配，必须显式修 --start-cell / --num。
    """
    if not os.path.exists(path):
        raise ValueError(f'{path}: 文件不存在')
    z = np.load(path, allow_pickle=True)
    arr = None
    for k in keys:
        if k in z.files:
            arr = np.asarray(z[k])
            break
    if arr is None:
        raise ValueError(f'{path}: 没有 {" / ".join(keys)} 键（有 {list(z.files)}）')
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f'{path}: 条件必须是浮点数组，实际 {arr.dtype}')
    arr = arr.astype(np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f'{path}: 条件含 NaN/Inf，拒绝采样')
    if arr.min() < 0.0 or arr.max() > 1.0 + 1e-4:
        raise ValueError(f'{path}: 条件应在 [0,1]（P99 归一后），实际 '
                         f'[{arr.min():.3f},{arr.max():.3f}]；口径不对不要硬喂')
    num = int(num)
    if arr.ndim == 3:
        if tuple(arr.shape[1:]) != (win, cond_ch):
            raise ValueError(f'{path}: 条件形状 {tuple(arr.shape)}，期望 [N,{win},{cond_ch}]')
        if arr.shape[0] < num:
            raise ValueError(f'{path}: 只有 {arr.shape[0]} 个窗口条件，--num={num}')
        return arr[:num].copy(), f'{os.path.basename(path)} 每窗口条件 [0..{num}]'
    if arr.ndim == 2:
        if arr.shape[1] != cond_ch:
            raise ValueError(f'{path}: 条件通道 {arr.shape[1]}，但权重要 {cond_ch}'
                             f'（mel=32，4 带=4；两条路径不能混用）')
        if arr.shape[0] < win:
            raise ValueError(f'{path}: 只有 {arr.shape[0]} 格 < 窗口 {win}')
        if arr.shape[0] == win and num == 1:
            base = int(start_cell)
            if base != 0:
                raise ValueError(f'{path}: 单窗口条件只有 [{win},{cond_ch}] 格，'
                                 f'无法按 --start-cell={base} 切片')
            return arr[None].copy(), f'{os.path.basename(path)} 单窗口'
        out = np.zeros((num, win, cond_ch), np.float32)
        for i in range(num):
            b = int(start_cell) + i * win
            if b + win > arr.shape[0]:
                raise ValueError(f'{path}: 第 {i} 窗要从格 {b} 起切 {win} 格，'
                                 f'但文件只有 {arr.shape[0]} 格')
            out[i] = arr[b:b + win]
        return out, (f'{os.path.basename(path)} [T={arr.shape[0]},{cond_ch}] '
                     f'起始格 {int(start_cell)}，{num} 窗连续')
    raise ValueError(f'{path}: 条件维度 {arr.ndim}，期望 2（[T,ch]）或 3（[N,win,ch]）')


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
                    help='目标占位率（0=固定阈值）。给了就按概率分位数反解阈值，使解码占位率≈该值')
    ap.add_argument('--seed', type=int, default=0, help='随机种子')
    ap.add_argument('--threads', type=int, default=1, help='torch 线程数（默认 CPU 单线程）')
    ap.add_argument('--bpm', type=float, default=136.3, help='算步长/密度用，1/32 格 = 60/bpm/8 秒')
    ap.add_argument('--outdir', default='out/diffusion32/samples', help='输出目录')
    ap.add_argument('--ascii', type=int, default=0, help='打印前 N 个窗口的 ASCII 钢琴卷（0=不打印）')
    ap.add_argument('--quiet', action='store_true', help='不打印去噪进度')
    # ---- 音频条件（只有 cond_dim>0 的权重才用得上的可选参数）----
    ap.add_argument('--cond', default='',
                    help='4 带音频条件文件（S1/mel 权重不接受）：单歌 <歌名>-cond.npz 或 cond-windows.npz')
    ap.add_argument('--cond-mel', default='',
                    help='mel onset 通量条件（32 带）：[256,32] / [N,256,32] / 整首 [T,32]')
    ap.add_argument('--start-cell', type=int, default=0,
                    help='mel 条件在整首歌里的起始 1/32 格（窗口 i 取 start+i*256 起）')
    ap.add_argument('--cond-key', default='M',
                    help="期望的 checkpoint cond_key（默认 'M'，必须与权重元数据一致）")
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
    cch = int(info.get('cond_ch', 4))
    inject = str(info.get('inject', 'none'))
    cond_t, cond_info, cond_tag = None, None, ''
    if cdim > 0:
        if a.cond and a.cond_mel:
            print('[错误] --cond（4 带）与 --cond-mel（32 带 mel）只能给一个', file=sys.stderr)
            sys.exit(2)
        if cch == 32:
            if a.cond:
                print(f'[错误] 这个权重要 {cch} 带 mel onset 通量（inject={inject}，'
                      f'cond_key={info.get("cond_key")}），--cond 是 4 带路径，会形/义都不对',
                      file=sys.stderr)
                sys.exit(2)
            if a.cond_mel and not a.cond_zero:
                try:
                    arr, cond_tag = load_mel_cond(a.cond_mel, a.num, a.start_cell, cch)
                except (OSError, ValueError) as e:
                    print(f'[错误] mel 条件：{e}', file=sys.stderr)
                    sys.exit(2)
                cond_t = torch.from_numpy(arr)
                cm = arr.mean(axis=(0, 1))
                cond_info = {'文件': a.cond_mel, 'cond_key': info.get('cond_key'),
                             'start_cell': int(a.start_cell), '定位': cond_tag,
                             '形状': list(arr.shape),
                             '通道均值': [round(float(v), 4) for v in cm]}
                print(f'  音频条件（mel {cch} 带）：{cond_tag}；'
                      f'通道均值前8 {cond_info["通道均值"][:8]}')
            else:
                cond_t = torch.zeros(a.num, WIN, cch)
                cond_info = {'模式': '全零' + ('（--cond-zero）' if a.cond_zero else '（没给 --cond-mel）'),
                             'cond_key': info.get('cond_key')}
                print('  音频条件：全零（无条件模式）')
        elif cch == 4:
            if a.cond_mel:
                print(f'[错误] 这个权重是 4 带条件（cond_channels=4），--cond-mel 的 32 带用不了；'
                      f'请用 --cond', file=sys.stderr)
                sys.exit(2)
            if a.cond and not a.cond_zero:
                seg, cond_tag = load_cond_segment(a.cond, a.cond_index, a.cond_root)
                cond_t = torch.from_numpy(seg)[None].repeat(a.num, 1, 1)
                cond_info = {'文件': a.cond, 'index': int(a.cond_index),
                             '定位': cond_tag,
                             '通道均值': [round(float(v), 4) for v in seg.mean(0)]}
                print(f'  音频条件（4 带）：{cond_tag}；通道均值 {cond_info["通道均值"]}')
            else:
                cond_t = torch.zeros(a.num, WIN, 4)
                cond_info = {'模式': '全零' + ('（--cond-zero）' if a.cond_zero else '（没给 --cond）')}
                print('  音频条件：全零（无条件模式）')
        else:
            print(f'[错误] 不支持的条件通道数 cond_channels={cch}', file=sys.stderr)
            sys.exit(2)
    elif a.cond or a.cond_mel or a.cond_zero:
        print('[错误] 权重没有音频分支（cond_dim=0），--cond/--cond-mel 用不上；'
              '请用训练时加过音频条件的权重', file=sys.stderr)
        sys.exit(2)

    if cond_t is not None:
        if a.cond_zero:
            cond_sfx = '-czero'
        elif a.cond_mel:
            cond_sfx = f'-m{a.start_cell}'
        else:
            cond_sfx = f'-c{a.cond_index}'
    else:
        cond_sfx = ''

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
    extra = {'C': cond_t.numpy().astype(np.float16)} if cond_t is not None else {}
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
        'start_cell': int(a.start_cell), 'inject': inject, 'cond_ch': cch,
        'cond_sfx': cond_sfx,
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
