#!/usr/bin/env python3
"""4K 谱面扩散模型（CPU 小型一维 U-Net DDPM）—— **1/32 网格 · LN 感知（8 通道）**。

本文件是 tools/train-diffusion32-burst.py 的副本（不改共享文件），**唯一结构性改动**：

    输出/输入从 4 通道 → **8 通道** = 4 键 × 2（占用 + LN 体）
        Y = [X(4), L(4)]      X: LN 整段覆盖 + 普通音符起点；L: LN 覆盖格（长条体）
    损失 = 占用 MSE + λ · LN 体 MSE       （--ln-lambda，λ=1.0 / 2.0 两档对比）

理由：旧数据（out/diff-data-grouped）只保留 LN 起点，模型不可能学会「人写的 LN 有多长」。
本文件吃 out/diff-data-ln/dataset.npz（同一套 group 划分 + 1/32 真实红线网格），
把「长条体」当成第二个预测头，用 λ 控制它在总损失里的权重。

两档对照 + 基线
--------------
- λ=1.0 / λ=2.0：LN 体 MSE 权重不同，看 LN 召回（LN 头数/长度）是否变好；
- --ln-mode baseline：占用目标换成「LN 头当普通音符」（只有 tap 格 + LN 头，
  LN 体目标全 0），这正是旧数据集的口径。它**必然**学不出 LN 长度（L 输出≈0），
  用来证明「是 LN 体监督（而不是 8 通道结构本身）带来了 LN 能力」。

保持不变（与 burst 版逐项一致）
------------------------------
x0 参数化、base 48、batch 32、threads 1、按 group 严格切 split（train=split==0 /
val=split==1，禁止窗口随机划分）、group-cap 300、每 epoch 从 train 无放回抽
--max-windows 窗、验证用完整 val 集、固定 t/噪声（seed 1234）便于跨配置相减、
连续难度条件 = 占位率按 train 5%/95% 分位归一化（lo/hi 存 ckpt）。

验证损失为**未加权 MSE**，占用与 LN 体分开报（val_occ / val_ln，池化 + 组均）。
最优权重按 occ+ln（未加权和）选，保证两档 λ 用同一把尺子挑模型。

数据：out/diff-data-ln/dataset.npz（X/L/H 各 [N,4,256] uint8）
用法：
  python3 tools/train-diffusion32-ln.py --ln-lambda 1.0 --epochs 9 --batch 32 \
      --threads 1 --time-budget 1700 --out out/diffusion/round147-ln/l1.0/model.pt
"""
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

NCH = 8          # 4 键 ×（占用, LN 体）
NLANES = 4


def build_targets(d, mode):
    """把 npz 的 X/L/H 拼成 [N,8,W] 训练目标：前 4 通道=占用，后 4 通道=LN 体。

    mode='full'     : 占用 = X（LN 整段覆盖），LN 体 = L（真实长条体）
    mode='baseline' : 占用 = (tap 格 ∪ LN 头)，LN 体 = 全 0
                      —— 旧数据集口径「把 LN 头当普通音符」，LN 长度信息为零
    """
    Xr = np.asarray(d['X'])
    N, _, W = Xr.shape
    Y = np.empty((N, NCH, W), dtype=np.uint8)
    Y[:, :4] = Xr
    if mode == 'full':
        Y[:, 4:] = np.asarray(d['L'])
    else:
        Lr = np.asarray(d['L'])
        Hr = np.asarray(d['H'])
        Y[:, :4] = (Xr & (1 - Lr)) | Hr      # 去掉 LN 体，只留头（= 普通音符）
        Y[:, 4:] = 0
    return Y


def ln_loss(diff, model, y0, style, dcond, mode, lam, pos_weight=1.0,
            adj_weight=0.0, scope='lane', style_scale=None):
    """占用 MSE + λ · LN 体 MSE（x0 目标；邻接/正样本加权只作用在占用通道）。"""
    t = torch.randint(0, diff.T, (y0.shape[0],), device=y0.device)
    noise = torch.randn_like(y0)
    xt, tgt, _ = diff.target(y0, t, noise, mode)
    pred = model(xt, t, style, dcond, None)
    w = burst_weight(y0[:, :4], adj_weight, pos_weight, scope=scope,
                     style=style, style_scale=style_scale)
    d_occ = (pred[:, :4] - tgt[:, :4]) ** 2
    if w is not None:
        loss_occ = (w * d_occ).sum() / w.sum()
    else:
        loss_occ = d_occ.mean()
    loss_ln = ((pred[:, 4:] - tgt[:, 4:]) ** 2).mean()
    return loss_occ + lam * loss_ln, float(loss_occ), float(loss_ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='out/diff-data-ln/dataset.npz')
    ap.add_argument('--epochs', type=int, default=9)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--base', type=int, default=48)
    ap.add_argument('--timesteps', type=int, default=200)
    ap.add_argument('--pred', choices=['eps', 'x0', 'v'], default='x0',
                    help='训练目标参数化（x0 默认，与 burst 版一致）')
    ap.add_argument('--max-windows', type=int, default=8000,
                    help='每个 epoch 从 train(split==0) 里无放回抽多少个窗口（0=全量）')
    ap.add_argument('--out', default='out/models/diffusion32-ln.pt')
    ap.add_argument('--threads', type=int, default=1, help='torch 线程数（本机固定 1）')
    ap.add_argument('--val-frac', type=float, default=-1.0,
                    help='[已废弃] 本文件按 split 键取 val，给正数直接报错')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-fixed', type=int, default=1,
                    help='验证用固定 t/噪声（seed 1234），跨配置可直接相减')
    ap.add_argument('--min-epochs', type=int, default=0)
    ap.add_argument('--time-budget', type=int, default=0, help='秒；超了就停（0=不限）')
    ap.add_argument('--log', default='')
    ap.add_argument('--group-cap', type=int, default=300,
                    help='同源组每 epoch 最多贡献的窗口数（0=不设上限）')
    ap.add_argument('--measure-ab', type=int, default=0,
                    help='训完后做 A/B 泄漏量化（A=split==1；B=从 split==0 抽）；默认关')
    ap.add_argument('--ab-n', type=int, default=4149)
    ap.add_argument('--cond-dim', type=int, default=32)
    ap.add_argument('--pos-weight', type=float, default=1.0)
    ap.add_argument('--init-seed', type=int, default=-1)
    ap.add_argument('--adj-weight', type=float, default=0.0)
    ap.add_argument('--adj-style-scale', default='')
    ap.add_argument('--adj-scope', choices=['lane', 'row'], default='lane')
    # ---- 本轮唯一的结构性改动：LN 体输出 + λ ----
    ap.add_argument('--ln-lambda', type=float, default=1.0,
                    help='LN 体 MSE 的权重 λ（总损失 = 占用 MSE + λ · LN 体 MSE）')
    ap.add_argument('--ln-mode', choices=['full', 'baseline'], default='full',
                    help='full=LN 整段覆盖 + 真实 LN 体；'
                         'baseline=把 LN 头当普通音符、LN 体目标全 0（旧口径基线）')
    a = ap.parse_args()
    if a.ln_lambda < 0:
        raise SystemExit('[错误] --ln-lambda 必须 >= 0')
    style_scale = None
    if a.adj_style_scale.strip():
        try:
            vals = [float(x) for x in a.adj_style_scale.replace(' ', '').split(',') if x != '']
        except ValueError:
            raise SystemExit('[错误] --adj-style-scale 需要 4 个逗号分隔的数字')
        if len(vals) != 4 or any(v < 0 for v in vals):
            raise SystemExit('[错误] --adj-style-scale 需要 4 个 >=0 的数字')
        style_scale = torch.tensor(vals, dtype=torch.float32)

    torch.set_num_threads(max(1, a.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print(f'线程 {torch.get_num_threads()}（internop {torch.get_num_interop_threads()}）', flush=True)
    if a.val_frac > 0:
        raise SystemExit('[错误] --val-frac 已废弃：本训练器按数据集 split 键取 val')

    d = np.load(a.data)
    if 'L' not in d or 'H' not in d:
        raise SystemExit('[错误] 数据里没有 L/H 通道：请先用 tools/diff-data-ln.py 建 LN 数据集')
    Xraw = np.asarray(d['X'])
    W = int(Xraw.shape[2])
    if W % 4 != 0:
        raise SystemExit(f'[错误] 窗口长度 {W} 不能被 4 整除：U-Net 有两次 2× 下采样')
    if 'split' not in d:
        raise SystemExit('[错误] 数据里没有 split 键：拒绝回退到随机窗口切分')
    Y = build_targets(d, a.ln_mode)
    del Xraw
    print(f'数据 {a.data}：{Y.shape[0]} 窗口 × {Y.shape[1]} 通道 × {W} 行；'
          f'LN 模式 {a.ln_mode}，λ={a.ln_lambda}', flush=True)
    print(f'  目标占位率(前4通道) {float(Y[:, :4].mean()):.4f}；'
          f'LN 体占用率(后4通道) {float(Y[:, 4:].mean()):.4f}', flush=True)
    split = np.asarray(d['split']).astype(np.int8)
    S = torch.from_numpy(np.asarray(d['style']).astype(np.int64))
    tr_np = np.where(split == 0)[0]
    va_np = np.where(split == 1)[0]
    if len(tr_np) == 0 or len(va_np) == 0:
        raise SystemExit('[错误] split 里 train/val 有一侧为空；'
                         f'分布 {np.bincount(np.clip(split, 0, 2), minlength=3).tolist()}')
    cid = np.asarray(d['chart_id']) if 'chart_id' in d else None
    gid = np.asarray(d['group_id']) if 'group_id' in d else None
    # 连续难度条件：占用通道的窗口占位率；lo/hi 只在 train 上拟合
    occ = Y[:, :4].reshape(len(Y), -1).mean(axis=1).astype(np.float32)
    lo, hi = np.percentile(occ[tr_np], 5), np.percentile(occ[tr_np], 95)
    cont = np.clip((occ - lo) / max(1e-6, hi - lo), 0, 1).astype(np.float32)
    D = torch.from_numpy(cont)
    print(f'连续难度条件（占位率，train 拟合）：{lo:.3f}–{hi:.3f} → [0,1]；'
          f'train 均值 {cont[tr_np].mean():.3f} / val 均值 {cont[va_np].mean():.3f}', flush=True)

    tr = torch.from_numpy(tr_np)
    te = torch.from_numpy(va_np)
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
          f'{sizes[len(sizes) // 2]} 窗 / 最小 {sizes[-1]} 窗', flush=True)

    def draw_pool(n_target, seed):
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
    Xte = torch.from_numpy(np.ascontiguousarray(Y[va_np])).float()
    Ste, Dte = S[te], D[te]
    n_grp = (lambda arr, idx: len(np.unique(arr[idx])) if arr is not None else -1)
    print(f'split 口径：train = split==0 → {len(tr)} 窗 / {n_grp(cid, tr_np)} 谱 / '
          f'{n_grp(gid, tr_np)} 组；val = split==1 → {len(te)} 窗 / {n_grp(cid, va_np)} 谱 / '
          f'{n_grp(gid, va_np)} 组；test = split==2 → {int((split == 2).sum())} 窗（本次不碰）',
          flush=True)
    print(f'每个 epoch 从 train 无放回抽 {n_ep} 窗（共 {len(tr)}）；val 用完整 {len(te)} 窗', flush=True)
    print(f'  风格分布 train {torch.bincount(S[tr], minlength=4).tolist()}  '
          f'val {torch.bincount(Ste, minlength=4).tolist()}', flush=True)

    if a.init_seed >= 0:
        torch.manual_seed(a.init_seed)
        print(f'初始种子 {a.init_seed}（配对实验用）', flush=True)
    model = UNet1D(cin=NCH, base=a.base, cond_dim=0)
    nparam = sum(p.numel() for p in model.parameters())
    print('参数量 %.3fM（8 通道：占用 4 + LN 体 4）' % (nparam / 1e6), flush=True)
    diff = Diffusion(a.timesteps)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def eval_loss(Y_e, S_e, D_e, lbl=None):
        """返回 dict：占用/LN 体分开的池化与组均 MSE（同一把固定 t/噪声尺子）。"""
        n = len(Y_e)
        g = torch.Generator().manual_seed(1234)
        vt_ = torch.randint(0, a.timesteps, (n,), generator=g)
        vn_ = torch.randn(n, NCH, W, generator=g)
        if not a.val_fixed:
            g = torch.Generator().manual_seed(int(time.time()))
            vt_ = torch.randint(0, a.timesteps, (n,), generator=g)
            vn_ = torch.randn(n, NCH, W, generator=g)
        vec_occ = torch.zeros(n)
        vec_ln = torch.zeros(n)
        with torch.no_grad():
            for i in range(0, n, 512):
                m = min(512, n - i)
                sl = slice(i, i + m)
                xt = diff.q(Y_e[sl], vt_[sl], vn_[sl])
                pred = model(xt, vt_[sl], S_e[sl], D_e[sl], None)
                _, tgt_v, _ = diff.target(Y_e[sl], vt_[sl], vn_[sl], a.pred)
                vec_occ[sl] = ((pred[:, :4] - tgt_v[:, :4]) ** 2).mean(dim=(1, 2))
                vec_ln[sl] = ((pred[:, 4:] - tgt_v[:, 4:]) ** 2).mean(dim=(1, 2))

        def gm(v):
            if lbl is None or len(lbl) != n:
                return None
            k = int(lbl.max().item()) + 1
            per = torch.zeros(k).scatter_add_(0, lbl, v)
            cnt = torch.zeros(k).scatter_add_(0, lbl, torch.ones_like(v))
            return float((per / cnt.clamp(min=1.0)).mean())

        return {'occ': float(vec_occ.mean()), 'ln': float(vec_ln.mean()),
                'occ_gm': gm(vec_occ), 'ln_gm': gm(vec_ln),
                'occ_vec': vec_occ, 'ln_vec': vec_ln}

    t_start = time.time()

    def build_payload(ep, vl, ev):
        return {'state': model.state_dict(), 'pred': a.pred, 'base': a.base,
                'model_type': 'unet1d-continuous-fourier-v1',
                'channels': NCH, 'channel_layout': 'X(4)|L(4)',
                'ln': {'lambda': a.ln_lambda, 'mode': a.ln_mode,
                       'loss': 'occ_mse + lambda * ln_body_mse',
                       'val_occ': round(ev['occ'], 6), 'val_ln': round(ev['ln'], 6)},
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
                'train_windows': int(len(tr)), 'val_windows': int(len(te)),
                'temb_dim': 128, 'timesteps': a.timesteps, 'val': vl,
                'epoch': ep, 'batch': a.batch, 'rows': W, 'grid': '1/32',
                'pos_weight': a.pos_weight, 'adj_weight': a.adj_weight,
                'adj_scope': a.adj_scope,
                'adj_style_scale': (style_scale.tolist() if style_scale is not None else None),
                'val_loss_note': '验证恒为未加权 MSE；val_occ 与 val_ln 分开报，'
                                 '最优权重按 occ+ln 选',
                'windows': int(len(tr)), 'params': nparam, 'cond_dim': 0,
                'cond_channels': 0, 'data': a.data,
                'val_idx': te.tolist(), 'seed': a.seed, 'init_seed': a.init_seed,
                'train_seconds': round(time.time() - t_start, 1)}

    best = 1e9
    best_payload = None
    last_payload = None
    seen = torch.zeros(len(Y), dtype=torch.bool)
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        pool = draw_pool(n_ep, a.seed * 100003 + ep)
        seen[pool] = True
        if ep == 1:
            if gid is not None:
                _, cc = np.unique(gid[pool.numpy()], return_counts=True)
                print(f'  本 epoch 抽到 {len(pool)} 窗，覆盖 {len(cc)} 组，单组最多 {int(cc.max())} 窗'
                      f'（上限 {a.group_cap or "无"}）', flush=True)
        tot = tot_occ = tot_ln = 0.0
        nb = 0
        t_first = 0.0
        for i in range(0, len(pool), a.batch):
            idx = pool[i:i + a.batch]
            tb = time.time()
            opt.zero_grad()
            yb = torch.from_numpy(np.ascontiguousarray(Y[idx.numpy()])).float()
            loss, lo_, ln_ = ln_loss(diff, model, yb, S[idx], D[idx], a.pred,
                                     a.ln_lambda, pos_weight=a.pos_weight,
                                     adj_weight=a.adj_weight, scope=a.adj_scope,
                                     style_scale=style_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
            tot_occ += lo_ * len(idx)
            tot_ln += ln_ * len(idx)
            nb += 1
            if nb == 1:
                t_first = (time.time() - tb) * 1000
        sched.step()
        model.eval()
        ev = eval_loss(Xte, Ste, Dte, gval_lbl)
        ep_s = time.time() - t0
        vl = ev['occ'] + ev['ln']
        print(f'epoch {ep}: 训练 总 {tot/len(pool):.4f}（占用 {tot_occ/len(pool):.4f} / '
              f'LN {tot_ln/len(pool):.4f}）  验证 占用 {ev["occ"]:.4f} / LN {ev["ln"]:.4f}'
              f'（组均 {ev["occ_gm"]:.4f} / {ev["ln_gm"]:.4f}）  '
              f'用时 {ep_s:.0f}s（{nb} batch，{t_first:.0f}ms/batch）', flush=True)
        if a.log:
            os.makedirs(os.path.dirname(a.log) or '.', exist_ok=True)
            with open(a.log, 'a') as f:
                f.write(json.dumps({'epoch': ep, 'train': tot / len(pool),
                                    'train_occ': tot_occ / len(pool),
                                    'train_ln': tot_ln / len(pool),
                                    'val_occ': ev['occ'], 'val_ln': ev['ln'],
                                    'val_occ_group_mean': ev['occ_gm'],
                                    'val_ln_group_mean': ev['ln_gm'],
                                    'sec': round(ep_s, 1), 'batches': nb,
                                    'ms_per_batch': round(t_first),
                                    'train_windows': int(len(pool)),
                                    'ln_lambda': a.ln_lambda, 'ln_mode': a.ln_mode,
                                    'lr': round(sched.get_last_lr()[0], 8)},
                                   ensure_ascii=False) + '\n')
        last_payload = build_payload(ep, vl, ev)
        if vl < best:
            best = vl
            best_payload = build_payload(ep, vl, ev)
            os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
            torch.save(best_payload, a.out)
            print(f'  ↳ 保存最优（占用+LN {vl:.4f}）→ {a.out}', flush=True)
        if a.time_budget and ep >= a.min_epochs and (time.time() - t_start) > a.time_budget:
            print(f'时间预算 {a.time_budget}s 到，停在 epoch {ep}', flush=True)
            break

    if a.measure_ab and last_payload is not None and gid is not None:
        model.eval()
        ab = draw_pool(min(a.ab_n, len(tr)), a.seed + 7919)
        _, invab = np.unique(gid[ab.numpy()], return_inverse=True)
        gab_lbl = torch.from_numpy(invab.astype(np.int64))
        Xab = torch.from_numpy(np.ascontiguousarray(Y[ab.numpy()])).float()
        eva = eval_loss(Xte, Ste, Dte, gval_lbl)
        evb = eval_loss(Xab, S[ab], D[ab], gab_lbl)
        ab_res = {'A_occ': round(eva['occ'], 6), 'A_ln': round(eva['ln'], 6),
                  'B_occ': round(evb['occ'], 6), 'B_ln': round(evb['ln'], 6),
                  'B_minus_A_occ': round(evb['occ'] - eva['occ'], 6),
                  'B_minus_A_ln': round(evb['ln'] - eva['ln'], 6),
                  'n_A': int(len(te)), 'n_B': int(len(ab))}
        print(f'\\n=== A/B 泄漏量化 ===\\n  A split==1 占用 {eva["occ"]:.4f} / LN {eva["ln"]:.4f}\\n'
              f'  B split==0 占用 {evb["occ"]:.4f} / LN {evb["ln"]:.4f}', flush=True)
        if last_payload is not None:
            last_payload['leak_ab'] = ab_res
        if best_payload is not None:
            best_payload['leak_ab'] = ab_res
            torch.save(best_payload, a.out)

    last_path = os.path.splitext(a.out)[0] + '-last.pt'
    if last_payload is not None:
        os.makedirs(os.path.dirname(last_path) or '.', exist_ok=True)
        torch.save(last_payload, last_path)
        print(f'末轮权重 → {last_path}', flush=True)
    print(f'完成，最优 占用+LN {best:.4f}（epoch '
          f'{best_payload["epoch"] if best_payload else "-"}），'
          f'总用时 {time.time()-t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
