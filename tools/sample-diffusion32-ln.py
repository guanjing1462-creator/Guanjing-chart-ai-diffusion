#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4K 谱面扩散模型 DDPM 采样器 —— **1/32 网格 · LN 版（8 通道）**。

本文件是 tools/sample-diffusion32occ.py 的副本（不改共享文件），为
tools/train-diffusion32-ln.py 训出的 **8 通道**权重服务：

    输入/输出 = 4 键 × 2：前 4 通道 = 占用 X，后 4 通道 = LN 体 L

采样后解码：
    X = P > thr_occ           （thr_occ 可由 --target-occ 按概率分位数反解）
    L = (Q > --ln-thresh) & X （LN 体必须是占用）
    H = L 的连续段起点         （窗口内能看到的 LN 头）
LN 头数 = 每键 L 连续段的段数；LN 长度 = 段的格数（1/32 格，按窗口裁剪）。
这与 tools/diff-data-ln.py 的统计口径一致，可直接与人类数据对照。

用法
----
  # 5 窗，占位率标定到人类均值 0.2223，风格 3(stamina)
  python3 tools/sample-diffusion32-ln.py --ckpt out/diffusion/round147-ln/l1.0/model.pt \
      --style 3 --diff 0.455 --target-occ 0.2223 --num 5 --seed 147 \
      --outdir out/diffusion/round147-ln/samples/l1.0-s3

  # LN 体阈值放大（诊断「概率有没有信息」时用）
  python3 tools/sample-diffusion32-ln.py ... --ln-thresh 0.3

输出（--outdir）
  roll-<stem>-<i>.npy   每个窗口的 X [4,256] uint8
  rolls-<stem>.npz      X/L/H [n,4,256] uint8 + P/Q 概率 + 阈值元数据
  sample-meta-<stem>.json  参数与统计（LN 头数/体格/长度分布）
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

WIN = 256            # 1/32 格窗口长（与 out/diff-data-ln 一致）
NLANES = 4           # 4K
NCH = 8              # 4 键 ×（占用, LN 体）
MAX_NUM = 8


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
# 采样 / 解码 / 统计（本文件新增）
# ---------------------------------------------------------------------------
@torch.no_grad()
def ddpm_sample(model, diff, n, style_id, diff_id, device='cpu', verbose=True,
                pred=None):
    """DDPM 祖先采样：[n,8,256] 近似 x0（前 4 通道占用，后 4 通道 LN 体）。"""
    if not 0.0 <= float(diff_id) <= 1.0:
        raise ValueError('连续难度条件必须是 [0,1] 内的有限浮点数')
    stored_pred = getattr(model, 'prediction_type', None)
    pred = stored_pred if pred is None else pred
    if pred not in ('eps', 'x0', 'v'):
        raise ValueError('必须明确指定预测目标 eps/x0/v（或从 checkpoint 恢复）')
    if stored_pred is not None and pred != stored_pred:
        raise ValueError(f'预测目标 {pred!r} 与 checkpoint {stored_pred!r} 不一致')
    model.eval()
    x = torch.randn(n, NCH, WIN, device=device)
    sty = torch.full((n,), int(style_id), dtype=torch.long, device=device)
    dif = torch.full((n,), float(diff_id), dtype=torch.float32, device=device)
    T = diff.T
    t_start = time.time()
    report_at = max(1, T // 10)
    for t in reversed(range(T)):
        tb = torch.full((n,), t, dtype=torch.long, device=device)
        out = model(x, tb, sty, dif, None)
        a_t = diff.alphas[t]
        ab = diff.abar[t]
        x0 = prediction_to_x0(x, out, ab, pred)
        if pred == 'x0':
            x0 = x0.clamp(-1.0, 2.0)
        if t > 0:
            ab_prev = diff.abar[t - 1]
            mean = (ab_prev.sqrt() * diff.betas[t] * x0
                    + a_t.sqrt() * (1 - ab_prev) * x) / (1 - ab)
            sigma = (diff.betas[t] * (1 - diff.abar[t - 1]) / (1 - ab)).sqrt()
            x = mean + sigma * torch.randn_like(x)
        else:
            x = x0
        if verbose and (t % report_at == 0 or t == T - 1):
            print(f'  去噪 t={t:4d}/{T}  用时 {time.time() - t_start:5.1f}s', flush=True)
    return x


def ln_runs(ln_window):
    """[4,256] 的 LN 体 → 每段 (lane, start, end_inclusive, len)。"""
    runs = []
    for lane in range(NLANES):
        row = ln_window[lane]
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row, [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1
        for s, e in zip(starts.tolist(), ends.tolist()):
            runs.append((lane, s, e, e - s + 1))
    return runs


def runs_to_H(ln_window):
    """从 L 的连续段反推 LN 头（每段起点）。"""
    h = np.zeros_like(ln_window)
    for lane, s, _e, _len in ln_runs(ln_window):
        h[lane, s] = 1
    return h


def roll_stats(Xb, Lb, Pb, Qb, step_ms=None):
    """统计采样窗口：占用基础指标 + LN 头/体格/长度分布。"""
    n = Xb.shape[0]
    row_sum = Xb.sum(axis=1)
    occ = float(Xb.mean())
    st = {
        '窗口数': int(n),
        '占位率': round(occ, 4),
        '非空行比例': round(float((row_sum > 0).mean()), 4),
        '每窗口音符数': [int(v) for v in Xb.sum(axis=(1, 2))],
        'LN体占位率': round(float(Lb.mean()), 6),
    }
    heads, cells, lens, per_win_heads, per_win_cells, per_win_lens = [], [], [], [], [], []
    for i in range(n):
        runs = ln_runs(Lb[i])
        ln_cells = int(Lb[i].sum())
        heads.append(len(runs))
        cells.append(ln_cells)
        per_win_heads.append(len(runs))
        per_win_cells.append(ln_cells)
        per_win_lens.append([r[3] for r in runs])
        for r in runs:
            lens.append(r[3])
    st['LN头数'] = int(sum(heads))
    st['LN体格数'] = int(sum(cells))
    st['LN头数_每窗'] = heads
    st['LN体格数_每窗'] = cells
    st['LN长度_每窗'] = per_win_lens
    st['LN头数_每窗均值'] = round(float(np.mean(heads)), 3) if n else 0.0
    st['LN体格数_每窗均值'] = round(float(np.mean(cells)), 3) if n else 0.0
    st['有LN窗口占比'] = round(float(np.mean([h > 0 for h in heads])), 4) if n else 0.0
    if lens:
        arr = np.asarray(lens, dtype=float)
        st['LN长度分布'] = {
            'n': int(arr.size),
            '均值': round(float(arr.mean()), 2),
            '中位数': round(float(np.median(arr)), 2),
            'p25': round(float(np.percentile(arr, 25)), 2),
            'p75': round(float(np.percentile(arr, 75)), 2),
            'p95': round(float(np.percentile(arr, 95)), 2),
            '最大': int(arr.max()),
        }
    else:
        st['LN长度分布'] = {'n': 0}
    if step_ms:
        st['每窗口时长_s'] = round(WIN * step_ms / 1000.0, 3)
    # 概率诊断：LN 体的连续值分布（判断「学到的概率有没有信息」）
    st['LN概率'] = {
        '均值': round(float(Qb.mean()), 4),
        'q50': round(float(np.quantile(Qb, 0.50)), 4),
        'q90': round(float(np.quantile(Qb, 0.90)), 4),
        'q99': round(float(np.quantile(Qb, 0.99)), 4),
        'q999': round(float(np.quantile(Qb, 0.999)), 4),
        '最大': round(float(Qb.max()), 4),
        '>0.5格数': int((Qb > 0.5).sum()),
        '>0.3格数': int((Qb > 0.3).sum()),
    }
    return st


# ---------------------------------------------------------------------------
# 权重加载（8 通道 LN 版：严格加载，不猜结构）
# ---------------------------------------------------------------------------
def load_model(ckpt, random_weights, base, timesteps, threads, cond_dim=0, pred=None):
    """返回 (model, Diffusion, info)。只接受 8 通道（占用 4 + LN 体 4）的连续模型。"""
    torch.set_num_threads(max(1, threads))
    if pred is not None and pred not in ('eps', 'x0', 'v'):
        raise ValueError(f'未知预测目标 {pred!r}，必须是 eps/x0/v')
    if random_weights:
        model = UNet1D(cin=NCH, base=base, cond_dim=0)
        model.prediction_type = pred or 'x0'
        info = {'权重': '随机初始化（--random-weights）', 'base': base,
                'timesteps': timesteps, 'cond_dim': 0, 'pred': model.prediction_type}
        return model, Diffusion(timesteps), info

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
        raise ValueError(f'{ckpt}: 缺少 freq，无法恢复训练时的 Fourier 条件')
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
    if 'out.weight' in state and state['out.weight'].shape[0] != cin:
        raise ValueError(f'{ckpt}: 输出通道 {state["out.weight"].shape[0]} 与输入 {cin} 不一致')
    if (base != inferred_base or temb_dim != inferred_temb
            or cin != NCH or n_style != 4 or base % 8 or temb_dim % 2):
        raise ValueError(f'{ckpt}: 模型元数据与权重结构不一致（期望 cin={NCH}，'
                         f'实际 cin={cin}）；本采样器只认 LN 版 8 通道权重')
    if 'cond_net.proj.weight' in state:
        raise ValueError(f'{ckpt}: 这份权重带音频条件分支，本采样器不支持')
    if shape('freq', 1) != (temb_dim // 2,):
        raise ValueError(f'{ckpt}: freq 形状与 temb_dim 不一致')
    if integer('rows', WIN) != WIN or meta.get('grid', '1/32') != '1/32':
        raise ValueError(f'{ckpt}: 窗口/网格元数据不适用于 1/32、{WIN} 格采样器')
    if 'channels' in meta and integer('channels', NCH) != NCH:
        raise ValueError(f'{ckpt}: channels={meta["channels"]} 不是 8 通道 LN 模型')
    timesteps = integer('timesteps', timesteps)
    stored_pred = meta.get('pred')
    if stored_pred is not None and stored_pred not in ('eps', 'x0', 'v'):
        raise ValueError(f'{ckpt}: 未知 checkpoint 预测目标 {stored_pred!r}')
    if stored_pred is not None and pred is not None and pred != stored_pred:
        raise ValueError(f'{ckpt}: --pred {pred!r} 与 checkpoint {stored_pred!r} 不一致')
    resolved_pred = stored_pred if stored_pred is not None else pred
    if resolved_pred is None:
        raise ValueError(f'{ckpt}: 缺少 pred 元数据，请明确提供 --pred eps/x0/v；不猜测')

    model = UNet1D(cin=NCH, base=base, temb_dim=temb_dim, n_style=n_style, cond_dim=0)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise ValueError(f'{ckpt}: strict 加载失败，禁止跳过任何权重/缓冲区：{e}') from e
    model.prediction_type = resolved_pred
    diff = Diffusion(timesteps)
    info = {'权重': str(ckpt), '加载方式': 'strict', 'base': base,
            'temb_dim': temb_dim, 'timesteps': timesteps, 'cond_dim': 0,
            'pred': resolved_pred, 'channels': NCH,
            'difficulty_condition': meta.get('difficulty_condition'),
            'ln': meta.get('ln')}
    if 'val' in meta:
        info['训练验证损失'] = round(float(meta['val']), 5)
    return model, diff, info


def main(argv=None):
    ap = argparse.ArgumentParser(description='4K 谱面扩散模型（LN 版）采样器')
    ap.add_argument('--ckpt', required=True, help='8 通道权重路径')
    ap.add_argument('--style', type=int, default=3, choices=[0, 1, 2, 3],
                    help='风格条件：0=tech 1=jack 2=speed 3=stamina')
    ap.add_argument('--diff', type=float, default=0.5,
                    help='连续占位率条件 0..1（用 ckpt 里 train 拟合的 lo/hi 从人类占位率换算）')
    ap.add_argument('--pred', choices=['eps', 'x0', 'v'], default=None)
    ap.add_argument('--num', type=int, default=5, choices=list(range(1, MAX_NUM + 1)))
    ap.add_argument('--timesteps', type=int, default=200)
    ap.add_argument('--base', type=int, default=48)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='占用阈值（--target-occ 给了就作废）')
    ap.add_argument('--target-occ', type=float, default=0.0,
                    help='目标占位率（0=固定阈值）；按概率分位数反解阈值')
    ap.add_argument('--ln-thresh', type=float, default=0.5,
                    help='LN 体阈值（与占用同一固定口径 0.5）')
    ap.add_argument('--ln-target-ratio', type=float, default=0.0,
                    help='可选诊断：把 LN 体/占位 比例标定到该值（人类=0.0632），'
                         '只在报告里另给一份解码，不替代固定阈值')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bpm', type=float, default=200.0, help='密度统计用；默认数据集中位 BPM')
    ap.add_argument('--outdir', default='out/diffusion/round147-ln/samples')
    ap.add_argument('--ascii', type=int, default=0)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--json-out', default='', help='额外把统计写成 JSON（报告用）')
    a = ap.parse_args(argv)
    if not 0.0 <= a.diff <= 1.0:
        ap.error('--diff 必须是 [0,1] 内的有限浮点数')
    if not (0.0 < a.threshold < 1.0) or not (0.0 < a.ln_thresh < 1.0):
        ap.error('--threshold / --ln-thresh 要在 (0,1) 之间')

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    try:
        model, diff, info = load_model(a.ckpt, False, a.base, a.timesteps,
                                       a.threads, 0, pred=a.pred)
    except (OSError, ValueError, RuntimeError) as e:
        ap.error(str(e))
    n_par = sum(p.numel() for p in model.parameters())
    step_ms = 60000.0 / max(1e-6, a.bpm) / 8.0
    print(f'▶ 采样（LN 版 8 通道）：style={a.style} diff={a.diff} num={a.num} '
          f'T={diff.T} base={info["base"]} 参数 {n_par / 1e6:.2f}M')
    print(f'  权重：{info["权重"]}')
    if 'ln' in info:
        print(f'  ckpt LN 元数据：{info["ln"]}')

    t0 = time.time()
    raw = ddpm_sample(model, diff, a.num, a.style, a.diff,
                      verbose=not a.quiet, pred=info['pred']).numpy()
    cost = time.time() - t0

    P = raw[:, :NLANES].astype(np.float32)     # 占用概率
    Q = raw[:, NLANES:].astype(np.float32)     # LN 体概率
    thr_occ = a.threshold
    if a.target_occ > 0:
        if not (0.0 < a.target_occ < 0.95):
            print('[错误] --target-occ 要在 (0,0.95) 之间', file=sys.stderr)
            sys.exit(2)
        thr_occ = float(np.quantile(P, 1.0 - a.target_occ))
        print(f'[占位率标定] 目标 {a.target_occ:.4f} → 占用阈值 {thr_occ:.4f}')
    Xb = (P > thr_occ).astype(np.uint8)
    Lb = ((Q > a.ln_thresh) & (Xb > 0)).astype(np.uint8)     # LN 体必须是占用
    Hb = np.stack([runs_to_H(Lb[i]) for i in range(len(Lb))])

    st = roll_stats(Xb, Lb, P, Q, step_ms=step_ms)
    st['耗时_s'] = round(cost, 2)
    st['扩散步数'] = diff.T
    st['占用阈值'] = thr_occ
    st['占用阈值来源'] = 'target-occ' if a.target_occ > 0 else 'fixed'
    st['LN阈值'] = a.ln_thresh
    st['target_occ'] = a.target_occ
    st['style'] = a.style
    st['diff条件'] = a.diff
    st['ckpt'] = a.ckpt
    if a.ln_target_ratio > 0:
        # 诊断解码：让 LN 体/占用 的比例 ≈ 人类值，检验长度分布形状而不仅是密度
        occ_cells = max(1, int(Xb.sum()))
        want = a.ln_target_ratio * occ_cells
        mask_in = Xb > 0
        vals = Q[mask_in]
        need = min(1, max(0, int(want)))
        thr_ln2 = float(np.quantile(vals, 1.0 - need / max(1, vals.size))) if vals.size else 1.0
        Lb2 = ((Q > thr_ln2) & (Xb > 0)).astype(np.uint8)
        st['诊断解码_比例标定'] = {'目标LN体比': a.ln_target_ratio, '阈值': round(thr_ln2, 4),
                                  'LN体格数': int(Lb2.sum()),
                                  'LN头数': int(sum(len(ln_runs(Lb2[i])) for i in range(len(Lb2)))),
                                  'LN长度分布': roll_stats(Xb, Lb2, P, Q)['LN长度分布']}
    print('\n=== 采样统计（LN 版）===')
    print(json.dumps({k: v for k, v in st.items()
                      if k not in ('LN头数_每窗', 'LN体格数_每窗', 'LN长度_每窗')},
                     ensure_ascii=False, indent=1))
    print(f'  LN 头数/窗 {st["LN头数_每窗"]}')
    print(f'  LN 体格/窗 {st["LN体格数_每窗"]}')
    print(f'  LN 长度/窗 {st["LN长度_每窗"]}')

    os.makedirs(a.outdir, exist_ok=True)
    stem = f'ln-s{a.style}d{a.diff}-t{a.ln_thresh}-seed{a.seed}'
    for i in range(a.num):
        np.save(os.path.join(a.outdir, f'roll-{stem}-{i:02d}.npy'), Xb[i])
    npz_path = os.path.join(a.outdir, f'rolls-{stem}.npz')
    np.savez_compressed(npz_path, X=Xb, L=Lb, H=Hb, P=P, Q=Q,
                        style=np.int64(a.style), diff=np.float64(a.diff),
                        pred=np.str_(info['pred']), seed=np.int64(a.seed),
                        thr_occ=np.float64(thr_occ), ln_thresh=np.float64(a.ln_thresh),
                        target_occ=np.float64(a.target_occ))
    meta = {'ckpt': a.ckpt, 'style': a.style, 'diff': a.diff, 'num': a.num,
            'seed': a.seed, 'thr_occ': thr_occ, 'ln_thresh': a.ln_thresh,
            'target_occ': a.target_occ, 'bpm': a.bpm, 'win': WIN,
            'timesteps': diff.T, 'pred': info['pred'], 'weights': info,
            'stats': st}
    meta_path = os.path.join(a.outdir, f'sample-meta-{stem}.json')
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or '.', exist_ok=True)
        with open(a.json_out, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
    if a.ascii > 0:
        for i in range(min(a.ascii, a.num)):
            print(f'\n=== 窗口 {i}：占用 X（#）/ LN 体 L（=）===')
            for k in range(WIN):
                occ = [('#' if Xb[i, c, k] else '.') for c in range(NLANES)]
                ln = [('=' if Lb[i, c, k] else '.') for c in range(NLANES)]
                if any(v != '.' for v in occ) or any(v != '.' for v in ln):
                    print(f'{k:4d}  ' + ' '.join(occ) + '   ' + ' '.join(ln))
    print(f'\n钢琴卷 → {a.outdir}/roll-{stem}-*.npy')
    print(f'打包 → {npz_path}')
    print(f'统计 → {meta_path}')


if __name__ == '__main__':
    main()
