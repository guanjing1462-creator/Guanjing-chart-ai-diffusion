#!/usr/bin/env python3
"""4K 谱面扩散模型（CPU 小型 1D U-Net DDPM）—— **多尺度全分辨率条件注入（round 153）**。

本文件复制自 tools/train-diffusion32-mel.py（旧文件一律不改），唯一改动是
**条件注入方式**，用 `--inject` 在两档之间切换（其余超参/数据/种子完全相同）：

  --inject multi （新，S1）
      条件编码器保持 256 位置（1 格 = 1/32 格 ≈ 40ms），不加任何 stride：
        cond [B,256,32] → Conv1d(32→cond_dim,k=3,s=1) ×2 → [B,cond_dim,256]
      再对每个 U-Net 层级用 AvgPool1d(k) 降到该层长度，1×1 卷积对齐通道后相加：
        e0 256 位置 (base ch)   ← AvgPool1d(1)
        e1 128 位置 (2*base ch) ← AvgPool1d(2)
        e2  64 位置 (2*base ch) ← AvgPool1d(4)
        mid 64 位置 (2*base ch) ← AvgPool1d(4)
      → 最浅层是 256 位置，相位分辨率 ≈ 1 格（≈40ms @190BPM）。

  --inject mid4  （旧，S4；与 train-diffusion32-mel.py 逐字段一致）
        cond [B,256,32] → Conv1d(32→cond_dim)×2 → Conv1d(cond_dim→2*base,k=4,s=4)
             → [B,2*base,64] 只加到 mid 层 → 相位分辨率 ≈ 4 格（≈160ms）。

两档共享 `--init-seed`（torch.manual_seed 在建模型前）：U-Net 主干权重逐元素相同，
差别只在条件分支的接线方式，因此「谁学到细相位」不会混入初始化差异。

其余口径（x0 参数化、base 48、Fourier 难度条件、固定验证 t/噪声、未加权 MSE、
--threads 1、--group-cap 300、--cond-drop 0.1、split 严格按数据集）保持不变。
数据默认 out/diff-data-mel/dataset.npz 的 `M`（32 带 mel onset 通量）。

用法
----
  # S1：多尺度全分辨率
  python3 tools/train-diffusion32-align.py --inject multi --cond-key M --cond-dim 32 \\
      --cond-drop 0.1 --epochs 15 --batch 32 --pred x0 --threads 1 --init-seed 7 \\
      --out out/models/r153-S1-multi.pt
  # S4：stride=4 单点（旧结构）
  python3 tools/train-diffusion32-align.py --inject mid4  --cond-key M --cond-dim 32 \\
      --cond-drop 0.1 --epochs 15 --batch 32 --pred x0 --threads 1 --init-seed 7 \\
      --out out/models/r153-S4-mid4.pt

原始说明（train-diffusion32-mel.py 头）保留如下：

  0. **条件通道数由数据决定**：`--cond-key M` 取 [256,32] 多帧 mel 频谱，
     `--cond-key C` 取 [256,4] 四带 onset（**同一训练器跑 4 带基线**，
     两组实验只差条件内容，差异不可能来自代码路径）。
  1. **条件直接从数据集取**：`cond[idx]`（[256,ch] float16），
     不再按「歌名 + 起始格」去 out/multisong 查文件（旧路径不适用于
     out/diff-data-audio-big/dataset.npz，它把条件与窗口放在同一个 npz 里，
     行与行一一对齐，配对不可能错）。
  2. **严格按数据集 split 取数**：`split==0` → train，`split==1` → val
     （0=train/1=val/2=test）。不做任何随机窗口划分；启动时校验 train/val
     的 song 与 chart_id 交集为空，不空直接退出。
  3. **每 epoch 按 song 分组限流**：`--group-cap`（组 = `song`），避免窗口多的
     歌主导一个 epoch；train 池每轮重新抽样。
  4. 数据集没有 `style` 键 → 风格嵌入喂常数 0（该嵌入退化成偏置项，仅为了
     和采样器的 `n_style==4` 结构保持一致）。难度条件仍是连续占位率
     （5/95 分位归一化，分位只在 train 上算，不碰 val）。

其余（x0 参数化、base 48、Fourier 难度编码、固定验证 t/噪声、未加权 MSE、
--threads 1、checkpoint 字段）与 train-diffusion32c.py 一致。

音频条件分支（4 带版与 sample-diffusion32c.py 逐字段一致，cin 随数据变化）

    cond [B,256,ch] → Conv1d(ch→cond_dim)×2 → Conv1d(cond_dim→2*base, k=4, s=4)
                → [B,2*base,64] 加到 U-Net 中间层

`--cond-dim 0` → 不注册任何音频参数（无条件基线 A，state_dict 与旧权重同构）。
`--cond-drop p` → 训练时以概率 p 把条件置零，为 classifier-free guidance 铺路。

用法
----
  # A：无条件基线
  python3 tools/train-diffusion32-audio.py --cond-dim 0 --epochs 9 --batch 32 \
      --pred x0 --threads 1 --init-seed 7 --out out/models/r148-A-uncond.pt
  # mel 条件（主实验）
  python3 tools/train-diffusion32-mel.py --cond-key M --cond-dim 32 --cond-drop 0.1 \
      --epochs 9 --batch 32 --pred x0 --threads 1 --init-seed 7 \
      --out out/models/r151-M-mel.pt
  # 4 带条件（同训练器基线）
  python3 tools/train-diffusion32-mel.py --cond-key C --cond-dim 32 --cond-drop 0.1 \
      --epochs 9 --batch 32 --pred x0 --threads 1 --init-seed 7 \
      --out out/models/r151-C-band4.pt
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """音频条件分支：[B,L,4] → 逐位置特征 [B,cond_dim,L] → 适配到 U-Net 中间层。

    结构与 tools/sample-diffusion32c.py 的 AudioCondNet 逐字段一致（strict 加载）。
    投影层 std=0.3：默认初始化下条件对中间层贡献太小，短训练推不动
    （见 out/diffusion/AUDIO-BRANCH.md 的消融表）。
    """

    def __init__(self, cin=4, cond_dim=32, mid_ch=96, downsample=4):
        super().__init__()
        self.cin = int(cin)
        self.downsample = int(downsample)
        # stride=4 → [B,mid_ch,64] 加到 mid；2 → [B,mid_ch,128] 加到 e1；
        # 1 → [B,mid_ch//2,256] 加到 e0（相位分辨率 4→2→1 格）
        out_ch = mid_ch if self.downsample >= 2 else mid_ch // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cond_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(cond_dim, cond_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.proj = nn.Conv1d(cond_dim, out_ch,
                              kernel_size=(1 if self.downsample == 1
                                           else self.downsample),
                              stride=self.downsample)
        nn.init.normal_(self.proj.weight, std=0.3)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond):
        """cond: [B,L,cin] → [B,out_ch,L//downsample]"""
        f = self.net(cond.transpose(1, 2))    # [B,cond_dim,L]
        return self.proj(f)                    # [B,mid_ch,L//4]


class MultiScaleCondNet(nn.Module):
    """多尺度全分辨率条件分支（round 153 / S1）：256 位置条件注入到**每个** U-Net 层级。

        cond [B,256,cin]
          → Conv1d(cin→cond_dim, k=3, s=1) → SiLU       # 保持 256 位置
          → Conv1d(cond_dim→cond_dim, k=3, s=1) → SiLU  # 保持 256 位置
          → 逐层 AvgPool1d(k) 降到该层长度 + 1×1 卷积对齐通道
          → 加到 e0(256)/e1(128)/e2(64)/mid(64)

    最浅层是 **256 位置**（1 格 = 1/32 格 ≈ 40ms），所以相位分辨率 ≈ 1 格；
    深层的平均池化只是让「同一个条件」在各尺度上都能被看到，
    和旧版「一次性 stride=4 投影到 mid」相比，浅层不再丢采样点（不混叠）。

    投影层 std=0.3 / bias 置零，沿用 AUDIO-BRANCH.md 的消融结论
    （默认初始化在短训练里推不动条件分支）。
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
        self.register_buffer('freq', torch.randn(temb_dim // 2) * 3.0)
        self.c_diff = nn.Sequential(nn.Linear(temb_dim, temb_dim), nn.SiLU(),
                                    nn.Linear(temb_dim, temb_dim))
        self.temb = nn.Sequential(nn.Linear(temb_dim, temb_dim), nn.SiLU(),
                                  nn.Linear(temb_dim, temb_dim))
        self.inp = nn.Conv1d(cin, base, 3, padding=1)
        self.d1 = Block(base, base, temb_dim)          # 256
        self.d2 = Block(base, base * 2, temb_dim)      # 128
        self.d3 = Block(base * 2, base * 2, temb_dim)  # 64
        self.mid = Block(base * 2, base * 2, temb_dim)  # 64
        self.u3 = Block(base * 4, base * 2, temb_dim)
        self.u2 = Block(base * 3, base, temb_dim)
        self.out = nn.Conv1d(base, cin, 3, padding=1)
        self.down = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # cond_dim=0 → 不注册音频分支参数（A 基线，state_dict 与旧权重同构）
        self.cond_dim = int(cond_dim)
        self.cond_ch = int(cond_ch)
        self.cond_stride = int(cond_stride)
        self.inject = str(inject)
        if self.cond_dim > 0 and self.inject == 'multi':
            # 层级通道：e0=base, e1=2*base, e2=2*base, mid=2*base
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
        feats = torch.cat([torch.sin(d * self.freq[None, :]),
                           torch.cos(d * self.freq[None, :])], dim=1)
        te = self.temb(self.temb_fn(t) + self.c_style(style) + self.c_diff(feats))
        h = self.inp(x)                                # [B,b,256]
        e0 = self.d1(h, te)                            # [B,b,256]
        e1 = self.d2(self.down(e0), te)                # [B,2b,128]
        e2 = self.d3(self.down(e1), te)                # [B,2b,64]
        m = self.mid(e2, te)                           # [B,2b,64]
        if self.cond_net is not None:
            if cond is None:
                cond = x.new_zeros(x.shape[0], x.shape[2], self.cond_net.cin)
            if self.inject == 'multi':
                c0, c1, c2, c3 = self.cond_net(cond)
                e0 = e0 + c0                           # [B,b,256]   ← 1 格分辨率
                e1 = e1 + c1                           # [B,2b,128]
                e2 = e2 + c2                           # [B,2b,64]
                m = m + c3                             # [B,2b,64]
            else:
                cv = self.cond_net(cond)
                if self.cond_stride == 4:
                    m = m + cv                         # [B,2b,64]
                elif self.cond_stride == 2:
                    e1 = e1 + cv                       # [B,2b,128]
                else:
                    e0 = e0 + cv                       # [B,b,256]
        u3 = self.u3(torch.cat([self.up(m), e1], dim=1), te)
        u2 = self.u2(torch.cat([self.up(u3), e0], dim=1), te)
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

    def target(self, x0, t, noise, mode):
        if mode not in ('eps', 'x0', 'v'):
            raise ValueError(f'未知预测目标 {mode!r}，必须是 eps/x0/v')
        a = self.abar[t].to(x0.device)[:, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        if mode == 'x0':
            return xt, x0, None
        if mode == 'v':
            return xt, a.sqrt() * noise - (1 - a).sqrt() * x0, None
        return xt, noise, None

    def loss(self, model, x0, style, diff, mode='eps', cond=None, pos_weight=1.0):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt, tgt, _ = self.target(x0, t, noise, mode)
        pred = model(xt, t, style, diff, cond)
        if pos_weight != 1.0 and mode == 'x0':
            w = 1.0 + (pos_weight - 1.0) * x0
            return (w * (pred - tgt) ** 2).sum() / w.sum()
        return F.mse_loss(pred, tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='out/diff-data-mel/dataset.npz')
    ap.add_argument('--cond-key', default='', 
                    help='条件键：M=多帧 mel / C=四带 onset（空=有 M 用 M，否则 C）')
    ap.add_argument('--epochs', type=int, default=9)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--base', type=int, default=48)
    ap.add_argument('--timesteps', type=int, default=200)
    ap.add_argument('--pred', choices=['eps', 'x0', 'v'], default='x0',
                    help='训练目标参数化（本轮固定 x0：稀疏二值占用直接回归）')
    ap.add_argument('--out', default='out/models/r148-audio.pt')
    ap.add_argument('--threads', type=int, default=1,
                    help='torch 线程数（铁律：1）')
    ap.add_argument('--min-epochs', type=int, default=0)
    ap.add_argument('--time-budget', type=int, default=0, help='秒；超了就停（0=不限）')
    ap.add_argument('--log', default='', help='逐 epoch 追加 jsonl')
    ap.add_argument('--group-cap', type=int, default=300,
                    help='每 epoch 单个 song 最多贡献的窗口数（0=不限）')
    ap.add_argument('--max-train-windows', type=int, default=0,
                    help='调试用：只用前 N 个 train 窗口')
    # ---- 音频条件 ----
    ap.add_argument('--cond-dim', type=int, default=32,
                    help='音频条件分支逐位置通道数；0 = 无条件基线（不建分支）')
    ap.add_argument('--inject', type=str, default='multi', choices=['multi', 'mid4'],
                    help='条件注入方式：multi=多尺度全分辨率（256/128/64/64，'
                         '最浅层 1 格≈40ms）/ mid4=旧结构 stride=4 单点注入 '
                         'mid（64 格≈160ms）')
    ap.add_argument('--cond-drop', type=float, default=0.0,
                    help='训练时把条件置零的概率（CFG 用；A 档无意义）')
    ap.add_argument('--pos-weight', type=float, default=1.0,
                    help='x0 参数化时正样本权重（默认 1=不加权，val 始终未加权）')
    ap.add_argument('--init-seed', type=int, default=-1,
                    help='>=0 时先 torch.manual_seed(N)：A/B 共享同一初始权重与噪声流')
    ap.add_argument('--seed', type=int, default=0, help='池抽样/调试子集种子')
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print(f'线程 {torch.get_num_threads()}（interop {torch.get_num_interop_threads()}）', flush=True)

    d = np.load(a.data, allow_pickle=True)
    for k in ('X', 'song', 'split', 'chart_id', 'start_cell', 'bpm'):
        if k not in d:
            raise SystemExit(f'[错误] {a.data} 缺少键 {k!r}（本训练器要求音频版数据集）')
    ckey = a.cond_key or ('M' if 'M' in d else 'C')
    if ckey not in d:
        raise SystemExit(f'[错误] {a.data} 里没有条件键 {ckey!r}（有 {list(d.files)}）')
    Xraw = d['X']
    Wall = int(Xraw.shape[2])
    if Wall % 4 != 0:
        raise SystemExit(f'[错误] 窗口长度 {Wall} 不能被 4 整除（U-Net 两次 2× 下采样）')
    song = d['song'].astype(np.int64)
    split = d['split'].astype(np.int64)
    chart = d['chart_id'].astype(str)
    C_all = np.asarray(d[ckey], dtype=np.float32)      # [N,256,ch]
    cond_ch = int(C_all.shape[2]) if C_all.ndim == 3 else 0
    if C_all.ndim != 3 or C_all.shape[1] != Wall or cond_ch < 1:
        raise SystemExit(f'[错误] 条件 {ckey} 形状 {C_all.shape}，期望 [N,{Wall},ch]')
    print(f'数据 {a.data}：{len(Xraw)} 窗口 × {Xraw.shape[1]}键 × {Wall} 格；'
          f'条件键 {ckey} → {C_all.shape[1:]}', flush=True)

    # ---- 严格按 split 取数（0=train / 1=val / 2=test，test 一律不碰）----
    tr_g = np.where(split == 0)[0]
    va_g = np.where(split == 1)[0]
    n_te = int((split == 2).sum())
    if len(tr_g) == 0 or len(va_g) == 0:
        raise SystemExit('[错误] split 里没有 train 或 val 窗口')
    # 分组泄漏校验：song / chart 不许跨 train-val
    if set(song[tr_g].tolist()) & set(song[va_g].tolist()):
        raise SystemExit('[错误] song 在 train 与 val 之间重叠，split 不是按组划分')
    if set(chart[tr_g].tolist()) & set(chart[va_g].tolist()):
        raise SystemExit('[错误] chart_id 在 train 与 val 之间重叠')
    print(f'split：train {len(tr_g)} 窗 / {len(set(song[tr_g].tolist()))} 歌，'
          f'val {len(va_g)} 窗 / {len(set(song[va_g].tolist()))} 歌，'
          f'test {n_te} 窗（不使用）；train∩val song=0 chart=0', flush=True)

    # ---- 连续难度条件：占位率 5/95 分位归一化（分位只在 train 上算）----
    occ_all = Xraw.reshape(len(Xraw), -1).mean(axis=1).astype(np.float32)
    lo, hi = np.percentile(occ_all[tr_g], 5), np.percentile(occ_all[tr_g], 95)
    cont_all = np.clip((occ_all - lo) / max(1e-6, hi - lo), 0, 1).astype(np.float32)
    print(f'连续难度条件（train 分位）占位率 {lo:.3f}–{hi:.3f} → [0,1]，'
          f'train 均值 {cont_all[tr_g].mean():.3f}', flush=True)

    # ---- 只保留 train+val 的窗口，转 float32（内存 ×4，本数据集 ~40MB）----
    use = np.concatenate([tr_g, va_g])
    pos = {int(g): i for i, g in enumerate(use)}
    tr = np.array([pos[int(g)] for g in tr_g], dtype=np.int64)
    va = np.array([pos[int(g)] for g in va_g], dtype=np.int64)
    if a.max_train_windows and a.max_train_windows < len(tr):
        tr = tr[:a.max_train_windows]
    W = Wall
    X = torch.from_numpy(np.ascontiguousarray(Xraw[use]).astype(np.float32))
    D = torch.from_numpy(cont_all[use])
    S = torch.zeros(len(use), dtype=torch.long)        # 无 style 标签 → 常数 0
    C = (torch.from_numpy(np.ascontiguousarray(C_all[use]))
         if a.cond_dim > 0 else None)
    if C is not None:
        print(f'音频条件：直接取数据集 {ckey} 键 → {tuple(C.shape)}，'
              f'行均值 {float(C.mean()):.4f}，非零格占比 '
              f'{float((C.abs().sum(-1) > 0).float().mean()):.3f}', flush=True)

    tr_t = torch.from_numpy(tr)
    va_t = torch.from_numpy(va)
    # ---- 每 epoch 按 song 分组的抽样池 ----
    gid = song[use]
    songs_tr = sorted(set(gid[tr_t].tolist()))
    grp_idx = [tr_t[torch.from_numpy(np.where(gid[tr_t.numpy()] == s)[0])].numpy()
               for s in songs_tr]

    def draw_pool(seed):
        """每组最多 --group-cap 窗；再打乱。A/B 用同一套抽样（同 seed 同池）。"""
        g = torch.Generator().manual_seed(seed)
        parts = []
        for arr in grp_idx:
            m = len(arr)
            take = m if (not a.group_cap or m <= a.group_cap) else a.group_cap
            if take >= m:
                parts.append(torch.from_numpy(arr))
            else:
                sel = torch.randperm(m, generator=g)[:take]
                parts.append(torch.from_numpy(arr[sel.numpy()]))
        pool = torch.cat(parts)
        return pool[torch.randperm(len(pool), generator=g)]

    sizes = sorted([len(x) for x in grp_idx], reverse=True)
    print(f'train 池：{len(tr)} 窗 / {len(grp_idx)} 歌，单歌最多 {sizes[0]} 窗 '
          f'（--group-cap {a.group_cap or "无"} → 每轮约 '
          f'{sum(min(s, a.group_cap) if a.group_cap else s for s in sizes)} 窗）', flush=True)

    if a.init_seed >= 0:
        torch.manual_seed(a.init_seed)       # A/B 配对：共享初始权重与噪声流
        print(f'初始种子 {a.init_seed}（配对实验）', flush=True)
    model = UNet1D(base=a.base, cond_dim=(a.cond_dim if a.cond_dim > 0 else 0),
                   cond_ch=(cond_ch if a.cond_dim > 0 else 0),
                   cond_stride=(4 if a.inject == 'mid4' else 1),
                   inject=a.inject)
    nparam = sum(p.numel() for p in model.parameters())
    ncond = sum(p.numel() for p in model.cond_net.parameters()) if model.cond_net else 0
    print('参数量 %.3fM（cond_dim=%d，条件通道 %d，注入 %s，条件分支 %.3fM）'
          % (nparam / 1e6, model.cond_dim, model.cond_ch, model.inject, ncond / 1e6),
          flush=True)
    diff = Diffusion(a.timesteps)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    # 验证用固定 t/噪声（同一把尺子量，A/B 曲线可比）
    gval = torch.Generator().manual_seed(1234)
    vt = torch.randint(0, a.timesteps, (len(va),), generator=gval)
    vn = torch.randn(len(va), 4, W, generator=gval)
    Xte, Ste, Dte = X[va_t], S[va_t], D[va_t]
    Cte = C[va_t] if C is not None else None

    def eval_loss():
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(va), 512):
                sl = slice(i, min(512, len(va) - i) + i)
                cb = Cte[sl] if Cte is not None else None
                xt = diff.q(Xte[sl], vt[sl], vn[sl])
                pred = model(xt, vt[sl], Ste[sl], Dte[sl], cb)
                _, tgt_v, _ = diff.target(Xte[sl], vt[sl], vn[sl], a.pred)
                tot += F.mse_loss(pred, tgt_v).item() * (sl.stop - sl.start)
        return tot / len(va)

    best = 1e9
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        pool = draw_pool(1000 + ep)
        tot = 0.0
        nb = 0
        t_first = 0.0
        for i in range(0, len(pool), a.batch):
            idx = pool[i:i + a.batch]
            tb = time.time()
            opt.zero_grad()
            cb = None
            if C is not None:
                cb = C[idx]
                if a.cond_drop > 0:
                    keep = (torch.rand(len(idx)) >= a.cond_drop).float()[:, None, None]
                    cb = cb * keep
            loss = diff.loss(model, X[idx], S[idx], D[idx], mode=a.pred, cond=cb,
                             pos_weight=a.pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
            nb += 1
            if nb == 1:
                t_first = (time.time() - tb) * 1000
        sched.step()
        model.eval()
        vl = eval_loss()
        ep_s = time.time() - t0
        print(f'epoch {ep}: 训练 {tot / len(pool):.4f}  验证 {vl:.4f}  '
              f'用时 {ep_s:.0f}s（{nb} batch，{t_first:.0f}ms/batch）', flush=True)
        if a.log:
            os.makedirs(os.path.dirname(a.log) or '.', exist_ok=True)
            with open(a.log, 'a') as f:
                f.write(json.dumps({'epoch': ep, 'train': tot / len(pool), 'val': vl,
                                    'sec': round(ep_s, 1), 'batches': nb,
                                    'ms_per_batch': round(t_first),
                                    'pool': len(pool),
                                    'lr': round(sched.get_last_lr()[0], 8)},
                                   ensure_ascii=False) + '\n')
        if vl < best:
            best = vl
            os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
            torch.save({'state': model.state_dict(), 'pred': a.pred, 'base': a.base,
                        'model_type': 'unet1d-continuous-fourier-v1',
                        'difficulty_condition': {'type': 'occupancy-percentile-v1',
                                                 'lo': float(lo), 'hi': float(hi),
                                                 'percentiles': [5, 95],
                                                 'range': [0.0, 1.0]},
                        'temb_dim': 128, 'timesteps': a.timesteps, 'val': vl,
                        'epoch': ep, 'batch': a.batch, 'rows': W, 'grid': '1/32',
                        'pos_weight': a.pos_weight,
                        'windows': len(use), 'params': nparam,
                        'cond_dim': model.cond_dim,
                        'cond_channels': (cond_ch if model.cond_dim else 0),
                        'cond_key': ckey,
                        'cond_stride': model.cond_stride,
                        'inject': model.inject,
                        'inject_scales': ([256, 128, 64, 64]
                                          if model.inject == 'multi' else [64]),
                        'audio_cond_source': f'dataset.npz:{ckey}',
                        'cond_drop': a.cond_drop,
                        'data': a.data,
                        'split': {'train': int(len(tr)), 'val': int(len(va)),
                                  'test_unused': int(n_te)},
                        'train_songs': int(len(grp_idx)),
                        'val_source_idx': [int(x) for x in use[va]],
                        'val_idx': [int(x) for x in va],
                        'group_cap': a.group_cap,
                        'seed': a.seed, 'init_seed': a.init_seed,
                        'train_seconds': round(time.time() - t_start, 1)},
                       a.out)
            print(f'  ↳ 保存最优（验证 {vl:.4f}）', flush=True)
        if a.time_budget and ep >= a.min_epochs and (time.time() - t_start) > a.time_budget:
            print(f'时间预算 {a.time_budget}s 到，停在 epoch {ep}', flush=True)
            break
    print(f'完成，最优验证损失 {best:.4f}，总用时 {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
