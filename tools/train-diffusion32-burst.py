#!/usr/bin/env python3
"""4K 谱面扩散模型（CPU 可训练的小型一维 U-Net DDPM）—— **1/32 网格版 · 邻接加权（成簇）**。

本文件是 tools/train-diffusion32-grouped.py 的副本（不改共享文件），**唯一改动是把 x0 的
MSE 换成「邻接加权 MSE」**，让生成的 1/32 从「均匀撒点」偏向人类数据里的「成簇爆发」：

    邻接标记  adj[b,c,k] = X[b,c,k] * (X[b,c,k-1] | X[b,c,k+1])   # 同键列、时间相邻
    权重      w = 1 + adj_weight * adj          （--adj-weight，0 = 基线）
    损失      L = Σ w * (pred - x0)^2 / Σ w     （按权重和归一化，loss 尺度不随 w 漂移）

只给「与相邻格成对」的格加权；验证损失**始终是未加权 MSE**，所以不同 --adj-weight 的
val 可以直接相减。其余全部与 grouped 版一致：
  1. 读 `split` 键：train 只用 split==0（87,009 窗 / 2,299 谱 / 141 组），
     val 只用 split==1（5,025 窗 / 148 谱 / 17 组）。**禁止**再对窗口 randperm 切分；
  2. 训练集每个 epoch 从 split==0 里**无放回**随机抽 N 个窗口（--max-windows，默认 8000）
     控时长；val 用完整 5,025 窗（不抽）；
  3. 连续难度条件 = 占位率按 **train 集** 5%/95% 分位归一化到 [0,1]（lo/hi 存进
     checkpoint），采样器 --diff 直接吃这个归一化值，接口不变；
  4. checkpoint 里记录 split_policy / adj_weight / burst_loss 等元数据；
  5. --measure-ab：训完后用**同一份权重**在 A（split==1，无泄漏）与 B（从 split==0
     随机抽同样多窗，等价旧的泄漏式验证）上算损失，直接量化泄漏虚高多少。
其余（模型/条件/日志格式）与原训练器逐字节相同。

数据：out/diff-data-grouped/dataset.npz（[N,4,256] 钢琴卷窗口，1/32 网格 = 旧 1/16 的一半步长）
条件：风格（4 类）+ 连续占位率条件（5%/95% 分位归一化，不等于 Koishi 定数）+ 可选音频
训练目标：--pred eps/x0/v；Fourier 基底持久化，与连续采样器 strict 恢复一致。

音频条件（--audio-cond，默认关闭）
---------------------------------
Mug-Diffusion 用 mel 频谱当主条件；这里做同构的小样本版本：
每格 4 个 onset 强度（低/中/高频 + 总），形状 [128, 4]，取值 0–1。

    音频 [B,128,4] → Conv1d(4→cond_dim,3) → SiLU → Conv1d(cond_dim→cond_dim,3) → SiLU
                   → 逐位置特征 [B,cond_dim,128]
                   → Conv1d(cond_dim→2*base, kernel=4, stride=4)  ← 零初始化
                   → [B,2*base,32] 加到 U-Net 中间层 mid 上（通道相加）

- 不加 --audio-cond：模型结构与旧权重完全一致（cond_dim=0，不注册任何新参数），
  训练/验证/采样行为与改动前逐字节相同。
- 加了 --audio-cond 但某个窗口没有条件：喂全零（模型仍保留无条件能力）。
- --cond-drop p：训练时以概率 p 把条件置零（classifier-free guidance 用，默认 0）。
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

    中间层形状是 [B, 2*base, L/4]（两次 2× 池化），所以最后用 kernel=stride=4 的
    卷积同时完成（下采样 + 通道投影）。

    初始化用 PyTorch 默认（不是零初始化）：本仓只有 199 个窗口 / 144 步训练，
    零初始化的投影层在这么短的训练里长不起来（实测 12 轮后 proj 权重 |w|≈0.0026，
    对中间层幅度的贡献只有 1.7%，换条件后输出只动 0.35%，验证损失和无条件模型
    差 0.0005 —— 等于白插一个分支）。默认初始化让条件从第一步就参与计算。
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
        # 连续难度条件：用 Fourier 特征编码一个 [0,1] 标量（而不是离散档位）
        # 持久化固定基底；沿用旧 Parameter 的 state 键，旧权重可 strict 加载。
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
        # cond_dim=0 → 不注册任何音频分支参数，state_dict 与旧权重完全一致
        self.cond_dim = int(cond_dim)
        self.cond_net = (AudioCondNet(cin=4, cond_dim=self.cond_dim,
                                      mid_ch=base * 2, downsample=4)
                         if self.cond_dim > 0 else None)

    def forward(self, x, t, style, diff, cond=None):
        # diff 现在是 [B] 的浮点难度值（0..1）；转成 Fourier 特征再进 MLP
        d = diff.float()[:, None]
        feats = torch.cat([torch.sin(d * self.freq[None, :]),
                           torch.cos(d * self.freq[None, :])], dim=1)
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


def row_occupied(x0, thresh=0.5):
    """行占用 occ[b,k] = 该时间格上任一键有音符（4 键里 max）。返回 [B,L] 0/1。"""
    return (x0.max(dim=1).values > thresh).to(x0.dtype)


def adjacent_mask(x0, thresh=0.5, scope='lane'):
    """邻接标记（两种口径，时间轴两端不平移、不环绕）。

    scope='lane'（任务书原文）：adj[b,c,k] = X[b,c,k] & (X[b,c,k-1] | X[b,c,k+1])
        —— 同键列、时间相邻；只覆盖「同列 1/32 连续」这种极少见的内容。
    scope='row'（与 p132 的行口径对齐）：
        occ[b,k] = 任一键在 k 有音符；adj[b,c,k] = X[b,c,k] & (occ[b,k-1] | occ[b,k+1])
        —— 跨键列的 1/32 交替（左右手/楼梯）也算成对，而 p132 统计的正是这种东西
        （实测同列相邻只占 0.13–0.63% 的格，行口径占 12–34%，见报告 §2）。
    两种口径都只标记「有音符的格」，不会给空格的损失加权重。
    """
    xb = (x0 > thresh).to(x0.dtype)
    if scope == 'row':
        occ = row_occupied(x0, thresh)
        nb = torch.zeros_like(occ)
        nb[:, 1:] = nb[:, 1:] + occ[:, :-1]          # 左邻
        nb[:, :-1] = nb[:, :-1] + occ[:, 1:]         # 右邻
        return xb * ((nb > 0).to(x0.dtype))[:, None, :]
    nb = torch.zeros_like(xb)
    nb[:, :, 1:] = nb[:, :, 1:] + xb[:, :, :-1]      # 左邻
    nb[:, :, :-1] = nb[:, :, :-1] + xb[:, :, 1:]     # 右邻
    return xb * (nb > 0).to(x0.dtype)


def burst_weight(x0, adj_weight, pos_weight=1.0, scope='lane', style=None,
                 style_scale=None):
    """逐格损失权重 w = 1 + adj_weight * style_scale[style] * adj（再乘 pos_weight 分支）。

    返回 None 表示「不加权」，调用方直接退回未加权 MSE（与 grouped 版逐位一致）。
    style_scale：按风格缩放邻接强度（[4] 或 None=全 1）。全局加权会把 jack/stamina
    这种「本来就不成簇」的风格也推成成簇（row-a2 实测 0.092→0.465），所以用数据集里
    各风格的自然行邻接率做系数，让 tech/speed 保留强加权、jack/stamina 基本不推。
    """
    if adj_weight <= 0 and pos_weight == 1.0:
        return None
    w = torch.ones_like(x0)
    if adj_weight > 0:
        # 成簇目标：只有「与相邻格成对」的格被放大，孤立格保持原权重。
        a = adj_weight
        if style_scale is not None:
            if style is None:
                raise ValueError('给了 style_scale 就必须同时给 style（[B] 的整型风格标签）')
            a = a * style_scale.to(x0.dtype)[style][:, None, None]
        w = w + a * adjacent_mask(x0, scope=scope)
    if pos_weight != 1.0:
        # 正样本加权可与邻接加权叠加：1/32 网格正样本只占 21.5%
        w = w * (1.0 + (pos_weight - 1.0) * x0)
    return w


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
        """返回 (xt, 训练目标, 权重)。eps=噪声，x0=干净谱面，v=速度。"""
        if mode not in ('eps', 'x0', 'v'):
            raise ValueError(f'未知预测目标 {mode!r}，必须是 eps/x0/v')
        a = self.abar[t].to(x0.device)[:, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        if mode == 'x0':
            return xt, x0, None
        if mode == 'v':
            return xt, a.sqrt() * noise - (1 - a).sqrt() * x0, None
        return xt, noise, None

    def loss(self, model, x0, style, diff, mode='eps', cond=None, pos_weight=1.0,
             adj_weight=0.0, scope='lane', style_scale=None):
        """x0 目标 + 邻接加权 MSE。

        L = Σ w * (pred - x0)^2 / Σ w，w = 1 + adj_weight * adj（* 可选 pos_weight 项）。
        按权重和归一化而不是直接求和：否则 adj_weight 一变大，损失的绝对尺度也跟着变大，
        等效于偷偷调高学习率，两档就不可比了（--adj-weight 0 时与未加权 MSE 完全相等）。
        """
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt, tgt, _ = self.target(x0, t, noise, mode)
        pred = model(xt, t, style, diff, cond)
        w = burst_weight(x0, adj_weight, pos_weight, scope=scope,
                         style=style, style_scale=style_scale)
        if mode == 'x0' and w is not None:
            return (w * (pred - tgt) ** 2).sum() / w.sum()
        return F.mse_loss(pred, tgt)


def load_audio_cond(path, n_data, song=None, start=None, X=None,
                    cond_root='out/multisong', win=128):
    """把 --audio-cond 指向的 npz 读成 [N,128,4] float32，与 --data 的行一一对齐。

    支持三种文件格式：
      1. `{'cond': [N,128,4]}`：已经按数据行排好（最省事，见 tools/build-audio-cond-labels.py）
      2. `{'X','song','start','songs'}`（cond-windows.npz 格式）：按 (song,start) 配对，
         数据里没有 song/start 时退化成「行数相同 + X 相同」的按序对齐
      3. `{'feat': [T,4]}`（单首歌的 cond 文件）：只做校验，训练请用前两种

    对齐不上的窗口（没有音频的歌）填全零 —— 模型退回无条件行为，而不是报错。
    """
    cd = np.load(path, allow_pickle=True)
    keys = set(cd.keys())

    if 'cond' in keys:
        C = np.asarray(cd['cond'], dtype=np.float32)
        if C.ndim != 3 or C.shape[1:] != (win, 4):
            raise SystemExit(f'[错误] {path} 的 cond 形状 {C.shape}，期望 [N,{win},4]')
        if len(C) != n_data:
            raise SystemExit(f'[错误] {path} 的 cond 行数 {len(C)} 与数据 {n_data} 不一致')
        print(f'音频条件：{path}（cond 键，按行对齐）→ {C.shape}')
        return C

    if {'song', 'start', 'songs'} <= keys:
        songs = [str(s) for s in cd['songs']]
        cs, cr = np.asarray(cd['song']), np.asarray(cd['start'])
        rows = len(cs)
        C = np.zeros((n_data, win, 4), dtype=np.float32)
        cache = {}
        if song is not None and start is not None:
            # 按 (歌, 起始格) 精确配对：数据是别的歌时自然 0 命中（全零条件）
            lut = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(song, start))}
            hit = 0
            for j in range(rows):
                i = lut.get((int(cs[j]), int(cr[j])))
                if i is None:
                    continue
                C[i] = _cond_seg(songs[int(cs[j])], int(cr[j]), cond_root, win, cache)
                hit += 1
            print(f'音频条件：{path} 按 (song,start) 配对，命中 {hit}/{n_data} 行'
                  f'（未命中的窗口用全零）')
            if hit == 0:
                print('[注意] 一个都没配上：--data 与 --audio-cond 不是同一批窗口，'
                      '条件全为零，等于无条件训练')
            return C
        if rows != n_data:
            # 数据来自没有音频的曲子（比如全量 dataset.npz）：条件全零，模型学无条件去噪
            print(f'[警告] 条件窗口 {rows} 行 ≠ 数据 {n_data} 行，且 --data 没有 song/start 可配对'
                  f' → 全部用全零条件（等价于无条件训练，模型仍保留无条件能力）')
            return C
        if X is not None and 'X' in keys and not np.array_equal(np.asarray(cd['X']), X):
            print('[警告] --data 的 X 与 --audio-cond 的 X 不同：只按行号对齐，请确认是同一批窗口')
        for j in range(rows):
            C[j] = _cond_seg(songs[int(cs[j])], int(cr[j]), cond_root, win, cache)
        print(f'音频条件：{path} 按行对齐（{rows} 行）→ {C.shape}')
        return C

    if 'feat' in keys:
        raise SystemExit(f'[错误] {path} 是单首歌的 feat [T,4] 条件文件；'
                         '训练请用 cond-windows.npz 或带 cond 键的对齐文件')

    raise SystemExit(f'[错误] {path} 里没有可识别的条件键（cond / song+start+songs / feat）')


def _cond_seg(song_name, start, cond_root, win=128, cache=None):
    if cache is None:
        cache = {}
    if song_name not in cache:
        f = os.path.join(cond_root, f'{song_name}-cond.npz')
        cache[song_name] = np.load(f)['feat'].astype(np.float32)
    f = cache[song_name]
    seg = f[start:start + win]
    if len(seg) < win:                      # 越界补零，不让训练崩
        seg = np.concatenate([seg, np.zeros((win - len(seg), 4), np.float32)])
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='out/diff-data-grouped/dataset.npz')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--base', type=int, default=48)
    ap.add_argument('--timesteps', type=int, default=200)
    ap.add_argument('--pred', choices=['eps', 'x0', 'v'], default='eps',
                    help='训练目标参数化：eps（噪声）/ x0（干净谱面，适合稀疏二值）/ v（速度）')
    ap.add_argument('--max-windows', type=int, default=8000,
                    help='每个 epoch 从 train(split==0) 里无放回抽多少个窗口（0=全量 87,009）；'
                         'val 始终用完整 split==1 的 5,025 窗')
    ap.add_argument('--out', default='out/models/diffusion32.pt')
    ap.add_argument('--threads', type=int, default=1,
                    help='torch 线程数。CPU 被别的进程抢占时 1 线程反而最快（实测 617ms vs 2197ms/step）')
    ap.add_argument('--val-frac', type=float, default=-1.0,
                    help='[已废弃] 旧版随机切分比例；本文件按 split 键取 val，给正数会直接报错')
    ap.add_argument('--seed', type=int, default=0, help='每 epoch 抽窗 / 验证噪声的随机种子')
    ap.add_argument('--val-fixed', type=int, default=1,
                    help='验证用固定的 t/噪声（同一把尺子量，曲线才可比）')
    ap.add_argument('--min-epochs', type=int, default=0, help='时间预算生效前至少跑几轮')
    ap.add_argument('--time-budget', type=int, default=0, help='秒；超了就停（0=不限）')
    ap.add_argument('--log', default='', help='逐 epoch 追加 jsonl（便于出报告）')
    ap.add_argument('--group-cap', type=int, default=300,
                    help='同源组每 epoch 最多贡献的窗口数（0=不设上限，退化成按窗口均匀抽）。'
                         'train 最大组占 32,513 窗（37%%），不设上限会被少数大组主导')
    ap.add_argument('--measure-ab', type=int, default=1,
                    help='训完后用同一份权重做 A/B 泄漏量化（A=split==1；B=从 split==0 随机抽）；默认 1')
    ap.add_argument('--ab-n', type=int, default=5025,
                    help='B 集（泄漏式验证）窗口数，默认 5025 = val 集大小，便于直接对比')
    # ---- 音频条件（默认关闭，行为与旧版完全一致）----
    ap.add_argument('--audio-cond', default='',
                    help='音频条件文件（如 out/diff-data/cond-windows.npz）；不给=无条件分支')
    ap.add_argument('--cond-dim', type=int, default=32,
                    help='音频分支的逐位置特征通道数（--audio-cond 时生效）')
    ap.add_argument('--cond-drop', type=float, default=0.0,
                    help='训练时把条件置零的概率（classifier-free guidance 用，默认 0）')
    ap.add_argument('--cond-root', default='out/multisong',
                    help='单首歌 <歌名>-cond.npz 所在目录（cond-windows 配对时用）')
    ap.add_argument('--pos-weight', type=float, default=1.0,
                    help='x0 参数化时给正样本的权重倍数（默认 1=不加权）。'
                         '数据正样本率 p 时 (1-p)/p 是「平衡」参考值（1/32 数据 p≈0.21 → 3.6）；'
                         '验证损失始终用未加权的 MSE，保证不同配置的 val 可比')
    ap.add_argument('--init-seed', type=int, default=-1,
                    help='>=0 时先 torch.manual_seed(N)，让同 seed 的两次训练共享同一套初始权重'
                         '与噪声流（A/B 配对实验用；默认 -1 = 不动全局 RNG，保持旧行为）')
    # ---- 本轮唯一改动：邻接加权（成簇）----
    ap.add_argument('--adj-weight', type=float, default=0.0,
                    help='邻接加权强度：w = 1 + adj_weight * adj，'
                         'adj = 该格与本键列左/右邻格同时有音符。0 = 与 grouped 版完全相同的基线；'
                         '建议对比 2.0 与 5.0')
    ap.add_argument('--adj-style-scale', default='',
                    help='按风格缩放邻接强度：四风格逗号分隔（顺序 tech,jack,speed,stamina）。'
                         '数据集的自然行邻接率是 tech 0.577 / jack 0.049 / speed 0.590 / '
                         'stamina 0.117，归一到 tech 就是 1.0,0.085,1.02,0.20。'
                         '空=全 1（全局加权，会把 jack/stamina 也推成成簇）')
    ap.add_argument('--adj-scope', choices=['lane', 'row'], default='lane',
                    help='邻接口径：lane=任务书原文「同键列时间相邻」；'
                         'row=行口径（该时间格邻行有任一键占用），与 p132 度量的对象一致。'
                         '实测 lane 只覆盖 0.13–0.63%% 的格（项均权重 1.003–1.013），row 覆盖 12–34%%')
    a = ap.parse_args()
    style_scale = None
    if a.adj_style_scale.strip():
        try:
            vals = [float(x) for x in a.adj_style_scale.replace(' ', '').split(',') if x != '']
        except ValueError:
            raise SystemExit('[错误] --adj-style-scale 需要 4 个逗号分隔的数字')
        if len(vals) != 4 or any(v < 0 for v in vals):
            raise SystemExit('[错误] --adj-style-scale 需要 4 个 >=0 的数字（tech,jack,speed,stamina）')
        style_scale = torch.tensor(vals, dtype=torch.float32)
        print(f'按风格缩放邻接强度：{dict(zip(["tech","jack","speed","stamina"], vals))}', flush=True)
    if a.adj_weight < 0:
        raise SystemExit('[错误] --adj-weight 必须 >= 0（0 = 未加权的基线）')
    if a.adj_weight > 0 and a.pred != 'x0':
        raise SystemExit('[错误] 邻接加权只定义在 x0 目标上（权重取自干净的 x0），'
                         f'当前 --pred {a.pred}；请用 --pred x0')

    torch.set_num_threads(max(1, a.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print(f'线程 {torch.get_num_threads()}（internop {torch.get_num_interop_threads()}）', flush=True)

    if a.val_frac > 0:
        raise SystemExit('[错误] --val-frac 已废弃：本训练器按数据集 split 键取 val；'
                         '随机窗口切分会让同曲/同组窗口落在两侧（旧 val 0.0811 就是这么来的）')
    d = np.load(a.data)
    Xraw = d['X']                                  # uint8 [N,4,W]
    Wall = int(Xraw.shape[2])
    if Wall % 4 != 0:
        raise SystemExit(f'[错误] 窗口长度 {Wall} 不能被 4 整除：U-Net 有两次 2× 下采样')
    if 'split' not in d:
        raise SystemExit('[错误] 数据里没有 split 键：本训练器只认数据集自带的分组 split，'
                         '拒绝回退到随机窗口切分')
    print(f'数据 {a.data}：{Xraw.shape[0]} 窗口 × {Xraw.shape[1]}键 × {Wall} 行'
          f'（{Wall // 4} 格的 mid 层）', flush=True)
    split = np.asarray(d['split']).astype(np.int8)
    S = torch.from_numpy(np.asarray(d['style']).astype(np.int64))
    tr_np = np.where(split == 0)[0]                # train 只取 split==0
    va_np = np.where(split == 1)[0]                # val 只取 split==1
    if len(tr_np) == 0 or len(va_np) == 0:
        raise SystemExit('[错误] split 里 train/val 有一侧为空；'
                         f'分布 {np.bincount(np.clip(split, 0, 2), minlength=3).tolist()}')
    cid = np.asarray(d['chart_id']) if 'chart_id' in d else None
    gid = np.asarray(d['group_id']) if 'group_id' in d else None
    # 连续难度条件：用窗口占位率（≈密度）归一化到 [0,1]；lo/hi **只在 train 上拟合**
    occ = Xraw.reshape(len(Xraw), -1).mean(axis=1).astype(np.float32)
    lo, hi = np.percentile(occ[tr_np], 5), np.percentile(occ[tr_np], 95)
    cont = np.clip((occ - lo) / max(1e-6, hi - lo), 0, 1).astype(np.float32)
    D = torch.from_numpy(cont)
    print(f'连续难度条件（占位率，train 拟合）：{lo:.3f}–{hi:.3f} → [0,1]；'
          f'train 均值 {cont[tr_np].mean():.3f} / val 均值 {cont[va_np].mean():.3f}', flush=True)
    n_all = len(Xraw)
    # 音频条件：[N,128,4]，与数据行对齐；没给就不建（模型也少一个分支）
    C = None
    if a.audio_cond:
        if a.cond_dim <= 0:
            raise SystemExit('[错误] --cond-dim 必须 > 0')
        Cnp = load_audio_cond(a.audio_cond, n_all,
                              song=d['song'].astype(np.int64) if 'song' in d else None,
                              start=d['start'].astype(np.int64) if 'start' in d else None,
                              X=d['X'], cond_root=a.cond_root, win=Wall)
        C = torch.from_numpy(Cnp)
        nz = (C.abs().sum(-1) > 0).float().mean().item()
        print(f'  条件覆盖：{nz*100:.1f}% 的窗口有条件（其余全零）；'
              f'通道均值 {C.mean(dim=(0,1)).tolist()}', flush=True)
    elif 'cond' in d:
        print('[注意] 数据文件里有 cond，但没给 --audio-cond → 按无条件训练', flush=True)
    # ---- 按 split 键划分：train=split==0（每 epoch 抽 N），val=split==1（完整）----
    W = Wall
    tr = torch.from_numpy(tr_np)
    te = torch.from_numpy(va_np)
    # 同源组索引：最大组 32,513 窗（占 train 37%），纯按窗口均匀抽会被少数大组主导
    if gid is not None:
        _, invv = np.unique(gid[va_np], return_inverse=True)
        gval_lbl = torch.from_numpy(invv.astype(np.int64))
        ug, inv_tr = np.unique(gid[tr_np], return_inverse=True)
        grp_idx = [tr_np[inv_tr == j] for j in range(len(ug))]
    else:
        gval_lbl = None
        grp_idx = [tr_np]
    sizes = sorted((len(x) for x in grp_idx), reverse=True)
    print(f'组规模（train）：{len(grp_idx)} 组，最大 {sizes[0]} 窗 / 中位 '
          f'{sizes[len(sizes) // 2]} 窗 / 最小 {sizes[-1]} 窗；前 3 组占 '
          f'{100 * sum(sizes[:3]) / max(1, sum(sizes)):.0f}% 窗口', flush=True)

    def draw_pool(n_target, seed):
        """抽一批窗口：每组最多 group_cap（--group-cap），再从合并池里均匀抽到 n_target。

        train 的每个 epoch 与 B 集（泄漏量化）都走这一条路径 —— 两边窗口来源策略一致，
        否则 B 会被大组拉得过于乐观、把 A/B 差值夸大。
        """
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
        if len(pool) > n_target:
            pool = pool[torch.randperm(len(pool), generator=g)[:n_target]]
        return pool

    n_ep = min(a.max_windows, len(tr)) if a.max_windows else len(tr)
    # B 集（泄漏式验证的等价物）：同一套按组上限抽样，从 split==0 抽 ab_n 窗
    ab_n = min(a.ab_n, len(tr))
    ab = draw_pool(ab_n, a.seed + 7919)
    if gid is not None:
        _, invab = np.unique(gid[ab.numpy()], return_inverse=True)
        gab_lbl = torch.from_numpy(invab.astype(np.int64))
    else:
        gab_lbl = None
    # 只把 val / B 转 float32（全量 train 转 float32 要 356MB，2GB 机器吃不下）
    Xte = torch.from_numpy(np.ascontiguousarray(Xraw[va_np])).float()
    Ste, Dte = S[te], D[te]
    Xab = torch.from_numpy(np.ascontiguousarray(Xraw[ab.numpy()])).float()
    Sab, Dab = S[ab], D[ab]
    n_grp = (lambda arr, idx: len(np.unique(arr[idx])) if arr is not None else -1)
    print(f'split 口径：train = split==0 → {len(tr)} 窗 / {n_grp(cid, tr_np)} 谱 / '
          f'{n_grp(gid, tr_np)} 组；val = split==1 → {len(te)} 窗 / {n_grp(cid, va_np)} 谱 / '
          f'{n_grp(gid, va_np)} 组；test = split==2 → {int((split == 2).sum())} 窗（本次不碰）', flush=True)
    print(f'每个 epoch 从 train 无放回抽 {n_ep} 窗（共 {len(tr)}）；val 用完整 {len(te)} 窗（不抽）', flush=True)
    print(f'  风格分布 train {torch.bincount(S[tr], minlength=4).tolist()}  '
          f'val {torch.bincount(Ste, minlength=4).tolist()}', flush=True)
    print(f'  连续难度 [train min {float(D[tr].min()):.2f} max {float(D[tr].max()):.2f} 均值 '
          f'{float(D[tr].mean()):.2f}] [val 均值 {float(Dte.mean()):.2f}]', flush=True)

    if a.init_seed >= 0:
        torch.manual_seed(a.init_seed)       # A/B 配对：同 seed 共享初始权重与噪声流
        print(f'初始种子 {a.init_seed}（配对实验用）', flush=True)
    model = UNet1D(base=a.base, cond_dim=(a.cond_dim if a.audio_cond else 0))
    nparam = sum(p.numel() for p in model.parameters())
    print('参数量 %.3fM（cond_dim=%d）' % (nparam / 1e6, model.cond_dim), flush=True)
    diff = Diffusion(a.timesteps)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    # 邻接加权诊断：val 正样本里「与同键列邻格成对」的占比 + 全格平均权重，
    # 用来核对 --adj-weight 的实际强度（adj_weight 只放大这一小部分格）。
    with torch.no_grad():
        adj_v = adjacent_mask(Xte, scope=a.adj_scope)
        frac_adj = float(adj_v.sum()) / max(1.0, float(Xte.sum()))
        mean_w = 1.0 + a.adj_weight * float(adj_v.mean())
    print(f'邻接加权 adj_weight={a.adj_weight} scope={a.adj_scope}：'
          f'val 正样本里成对格占 {frac_adj * 100:.1f}%，'
          f'全格平均损失权重 {mean_w:.3f}（1.000 = 未加权 MSE）', flush=True)
    if style_scale is not None:
        per = []
        for s_ in range(4):
            m = torch.from_numpy(np.asarray(d['style'])[va_np] == s_)
            if int(m.sum()):
                per.append(f'style{s_}={1.0 + a.adj_weight * float(style_scale[s_]) * float(adj_v[m].mean()):.3f}')
        print('  按风格的平均损失权重：' + ' / '.join(per), flush=True)

    # 验证固定 t 与噪声：否则每轮用不同的噪声量同一把尺子，曲线抖到看不清趋势。
    # A/B 两边都用 seed 1234、同样长度 → t 与噪声逐元素相同，两个损失可直接相减。
    def eval_loss(X_e, S_e, D_e, lbl=None):
        """返回 (池化损失, 组均损失, 逐窗损失)。

        池化 = 全部窗口的未加权 MSE（与旧版 val 同口径）；
        组均 = 先按同源组算组内均值、再对组取平均（更能反映跨组泛化，
               不会被大组/多窗歌曲拉偏）。两套 t/噪声固定（seed 1234），可跨集相减。
        """
        n = len(X_e)
        g = torch.Generator().manual_seed(1234)
        vt_ = torch.randint(0, a.timesteps, (n,), generator=g)
        vn_ = torch.randn(n, 4, W, generator=g)
        if not a.val_fixed:                  # 非要随机尺子就每轮换一套（不推荐）
            g = torch.Generator().manual_seed(int(time.time()))
            vt_ = torch.randint(0, a.timesteps, (n,), generator=g)
            vn_ = torch.randn(n, 4, W, generator=g)
        vec = torch.zeros(n)
        with torch.no_grad():
            for i in range(0, n, 512):
                m = min(512, n - i)          # 末块真实长度（不能用 sl.stop-sl.start）
                sl = slice(i, i + m)
                xt = diff.q(X_e[sl], vt_[sl], vn_[sl])
                pred = model(xt, vt_[sl], S_e[sl], D_e[sl], None)
                _, tgt_v, _ = diff.target(X_e[sl], vt_[sl], vn_[sl], a.pred)
                vec[sl] = ((pred - tgt_v) ** 2).mean(dim=(1, 2))
        pooled = float(vec.mean())
        gm = None
        if lbl is not None and len(lbl) == n:
            k = int(lbl.max().item()) + 1
            per = torch.zeros(k).scatter_add_(0, lbl, vec)
            cnt = torch.zeros(k).scatter_add_(0, lbl, torch.ones_like(vec))
            gm = float((per / cnt.clamp(min=1.0)).mean())
        return pooled, gm, vec

    def build_payload(ep, vl, saw_windows):
        return {'state': model.state_dict(), 'pred': a.pred, 'base': a.base,
                'model_type': 'unet1d-continuous-fourier-v1',
                'difficulty_condition': {'type': 'occupancy-percentile-v1',
                                         'lo': float(lo), 'hi': float(hi),
                                         'percentiles': [5, 95], 'range': [0.0, 1.0],
                                         'fit_on': 'train split only (split==0)'},
                'split_policy': {'source': f"{a.data}['split']", 'train': 0, 'val': 1,
                                 'test': 2, 'random_window_split': False,
                                 'train_window_source': 'per-epoch random sample without '
                                                        'replacement from split==0',
                                 'train_per_epoch': int(n_ep),
                                 'val_window_source': 'all windows with split==1 (no sampling)'},
                'train_groups': int(n_grp(gid, tr_np)), 'val_groups': int(n_grp(gid, va_np)),
                'train_charts': int(n_grp(cid, tr_np)), 'val_charts': int(n_grp(cid, va_np)),
                'val_n_groups': int(len(np.unique(gid[va_np]))) if gid is not None else -1,
                'train_windows': int(len(tr)), 'val_windows': int(len(te)),
                'train_windows_seen': int(saw_windows),
                'temb_dim': 128, 'timesteps': a.timesteps, 'val': vl,
                'epoch': ep, 'batch': a.batch, 'rows': W, 'grid': '1/32',
                'pos_weight': a.pos_weight,
                'adj_weight': a.adj_weight, 'adj_scope': a.adj_scope,
                'adj_style_scale': (style_scale.tolist() if style_scale is not None else None),
                'burst_loss': ('adj-weighted-x0-mse-v1' if a.adj_weight > 0
                               else 'plain-x0-mse (= grouped 基线)'),
                'val_loss_note': '验证损失恒为未加权 MSE，跨 adj_weight 可直接相减',
                'windows': int(len(tr)), 'params': nparam,
                'cond_dim': model.cond_dim,
                'cond_channels': 4 if model.cond_dim else 0,
                'audio_cond': a.audio_cond or None,
                'cond_drop': a.cond_drop,
                'data': a.data,
                'val_idx': te.tolist(),
                'val_frac': a.val_frac, 'seed': a.seed,
                'init_seed': a.init_seed,
                'train_seconds': round(time.time() - t_start, 1)}

    best = 1e9
    best_payload = None
    last_payload = None
    seen = torch.zeros(n_all, dtype=torch.bool)     # 累计被抽到的窗口（报告用）
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        # 每 epoch 从 train(split==0) 抽 n_ep 窗（按组上限，见 draw_pool）；
        # 种子固定 → 每轮抽样可复现
        pool = draw_pool(n_ep, a.seed * 100003 + ep)
        seen[pool] = True
        if ep == 1:
            if gid is not None:
                _, cc = np.unique(gid[pool.numpy()], return_counts=True)
                print(f'  本 epoch 抽到 {len(pool)} 窗，覆盖 {len(cc)} 组，单组最多 {int(cc.max())} 窗'
                      f'（上限 {a.group_cap or "无"}）；对照组规模最大 {sizes[0]} 窗', flush=True)
            else:
                print(f'  本 epoch 抽到 {len(pool)} 窗（数据无 group_id，未做组上限）', flush=True)
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
                    # classifier-free guidance：按概率把条件置零，让模型同时学会「无条件」去噪
                    keep = (torch.rand(len(idx)) >= a.cond_drop).float()[:, None, None]
                    cb = cb * keep
            xb = torch.from_numpy(np.ascontiguousarray(Xraw[idx.numpy()])).float()
            loss = diff.loss(model, xb, S[idx], D[idx], mode=a.pred, cond=cb,
                             pos_weight=a.pos_weight, adj_weight=a.adj_weight,
                             scope=a.adj_scope, style_scale=style_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
            nb += 1
            if nb == 1:
                t_first = (time.time() - tb) * 1000
        sched.step()
        model.eval()
        vl, vgm, _ = eval_loss(Xte, Ste, Dte, gval_lbl)
        ep_s = time.time() - t0
        print(f'epoch {ep}: 训练 {tot/len(pool):.4f}  验证 {vl:.4f}（组均 {vgm:.4f}）  '
              f'用时 {ep_s:.0f}s（{nb} batch，{t_first:.0f}ms/batch）', flush=True)
        if a.log:
            os.makedirs(os.path.dirname(a.log) or '.', exist_ok=True)
            with open(a.log, 'a') as f:
                f.write(json.dumps({'epoch': ep, 'train': tot / len(pool), 'val': vl,
                                    'val_group_mean': vgm,
                                    'sec': round(ep_s, 1), 'batches': nb,
                                    'ms_per_batch': round(t_first),
                                    'train_windows': int(len(pool)),
                                    'lr': round(sched.get_last_lr()[0], 8)},
                                   ensure_ascii=False) + '\n')
        last_payload = build_payload(ep, vl, int(seen.sum()))
        if vl < best:
            best = vl
            best_payload = build_payload(ep, vl, int(seen.sum()))
            os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
            torch.save(best_payload, a.out)
            print(f'  ↳ 保存最优（验证 {vl:.4f}）→ {a.out}', flush=True)
        if a.time_budget and ep >= a.min_epochs and (time.time() - t_start) > a.time_budget:
            print(f'时间预算 {a.time_budget}s 到，停在 epoch {ep}', flush=True)
            break
    # ---------------- A/B 泄漏量化（同一份权重，同样窗口数，同样的固定 t/噪声）----------------
    ab_res = None
    if a.measure_ab and last_payload is not None:
        model.eval()
        a_loss, a_gm, _ = eval_loss(Xte, Ste, Dte, gval_lbl)   # A：split==1，无泄漏
        b_loss, b_gm, _ = eval_loss(Xab, Sab, Dab, gab_lbl)    # B：split==0 按组上限抽（泄漏式）
        ab_res = {'A_split1_leakfree': round(a_loss, 6),
                  'A_group_mean': round(a_gm, 6) if a_gm is not None else None,
                  'B_split0_sampled': round(b_loss, 6),
                  'B_group_mean': round(b_gm, 6) if b_gm is not None else None,
                  'B_minus_A': round(b_loss - a_loss, 6),
                  'B_over_A': round(b_loss / a_loss, 4) if a_loss > 0 else None,
                  'Bgm_minus_Agm': round(b_gm - a_gm, 6) if (a_gm is not None and b_gm is not None) else None,
                  'n_A': int(len(te)), 'n_B': int(len(ab)),
                  'A_groups': int(n_grp(gid, va_np)), 'B_groups': int(n_grp(gid, ab.numpy())),
                  'protocol': f'同一份权重；A {len(te)} 窗 / B {len(ab)} 窗，两侧抽取都走'
                              ' draw_pool（组上限 ' + str(a.group_cap) + '）；固定 t/噪声（seed 1234，'
                              '逐元素相同）；未加权 MSE；pred=' + a.pred,
                  'note': 'B 是旧做法的等价物：窗口从训练池里抽，与训练窗口同曲/可能已见过。'
                          'B 的组均按 141 个 train 组平均、A 的组均按 17 个 val 组平均，'
                          '两者的组构成不同，组均只作方向参考；池化差值是主证据'}
        print('\n=== A/B 泄漏量化（final epoch 权重）===')
        print(f'  A split==1（无泄漏） 池化 {a_loss:.4f} / 组均 {a_gm if a_gm is None else round(a_gm, 4)}'
              f'   [{len(te)} 窗 / {n_grp(gid, va_np)} 组]')
        print(f'  B split==0 抽样      池化 {b_loss:.4f} / 组均 {b_gm if b_gm is None else round(b_gm, 4)}'
              f'   [{len(ab)} 窗 / {n_grp(gid, ab.numpy())} 组]')
        print(f'  差值 B-A（池化）      : {b_loss - a_loss:+.4f}'
              f'（相对 A {100 * (b_loss - a_loss) / max(1e-9, a_loss):+.1f}%）', flush=True)
        if a_gm is not None and b_gm is not None:
            print(f'  差值 B-A（组均）      : {b_gm - a_gm:+.4f}', flush=True)
        if last_payload is not None:
            last_payload['leak_ab'] = ab_res
        if best_payload is not None:
            best_payload['leak_ab'] = ab_res
            torch.save(best_payload, a.out)
    last_path = os.path.splitext(a.out)[0] + '-last.pt'
    if last_payload is not None:
        os.makedirs(os.path.dirname(last_path) or '.', exist_ok=True)
        torch.save(last_payload, last_path)
        print(f'末轮权重（未被 val 选过，无选择偏差）→ {last_path}', flush=True)
    print(f'完成，最优验证损失 {best:.4f}（epoch {best_payload["epoch"] if best_payload else "-"}），'
          f'总用时 {time.time()-t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
