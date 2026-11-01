#!/usr/bin/env python3
"""4K 谱面扩散模型（CPU 可训练的小型一维 U-Net DDPM）—— **1/32 网格版**。

本文件是 tools/train-diffusion.py 的副本（不改共享文件），只做三处改动：
  1. 默认数据换成 out/diff-data32/dataset.npz（[N,4,256]，1/32 网格，同时长窗口）；
  2. 窗口长度从数据里读（W = X.shape[2]），验证噪声/位置编码都按 W 生成，
     不再硬编码 128；并检查 W 能被 4 整除（U-Net 有两次 2× 下采样：
     256→128→64，mid 层 64 格，卷积长度自适应，结构无需改）；
  3. 先按 seed 抽子集再转 float32（1/32 全量 88787 窗口转 float32 要 364MB，
     抽 6000 窗口只要 24MB，给这台 2GB 机器留余量）。
其余（条件、损失、时间预算、日志格式）与训练脚本完全一致。

数据：out/diff-data32/dataset.npz（[N, 4, 256] 钢琴卷窗口，1/32 网格 = 旧 1/16 的一半步长）
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

    def loss(self, model, x0, style, diff, mode='eps', cond=None, pos_weight=1.0):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt, tgt, _ = self.target(x0, t, noise, mode)
        pred = model(xt, t, style, diff, cond)
        if pos_weight != 1.0 and mode == 'x0':
            # 正样本（有音符的格）加权：1/32 网格上正样本只占 21.5%，
            # 不加权时模型倾向「大多数格是空的」，稀有的奇格（1/32 位置）被压到阈值下。
            w = 1.0 + (pos_weight - 1.0) * x0
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
    ap.add_argument('--data', default='out/diff-data32/dataset.npz')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--base', type=int, default=48)
    ap.add_argument('--timesteps', type=int, default=200)
    ap.add_argument('--pred', choices=['eps', 'x0', 'v'], default='eps',
                    help='训练目标参数化：eps（噪声）/ x0（干净谱面，适合稀疏二值）/ v（速度）')
    ap.add_argument('--max-windows', type=int, default=0, help='只用前 N 个窗口（调试用）')
    ap.add_argument('--out', default='out/models/diffusion32.pt')
    ap.add_argument('--threads', type=int, default=1,
                    help='torch 线程数。CPU 被别的进程抢占时 1 线程反而最快（实测 617ms vs 2197ms/step）')
    ap.add_argument('--val-frac', type=float, default=0.03, help='验证集比例')
    ap.add_argument('--seed', type=int, default=0, help='切分随机种子')
    ap.add_argument('--val-fixed', type=int, default=1,
                    help='验证用固定的 t/噪声（同一把尺子量，曲线才可比）')
    ap.add_argument('--min-epochs', type=int, default=0, help='时间预算生效前至少跑几轮')
    ap.add_argument('--time-budget', type=int, default=0, help='秒；超了就停（0=不限）')
    ap.add_argument('--log', default='', help='逐 epoch 追加 jsonl（便于出报告）')
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
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print(f'线程 {torch.get_num_threads()}（internop {torch.get_num_interop_threads()}）', flush=True)

    d = np.load(a.data)
    Xraw = d['X']                                  # uint8 [N,4,W]
    Wall = int(Xraw.shape[2])
    if Wall % 4 != 0:
        raise SystemExit(f'[错误] 窗口长度 {Wall} 不能被 4 整除：U-Net 有两次 2× 下采样')
    print(f'数据 {a.data}：{Xraw.shape[0]} 窗口 × {Xraw.shape[1]}键 × {Wall} 行'
          f'（{Wall // 4} 格的 mid 层）', flush=True)
    S = torch.from_numpy(d['style'].astype(np.int64))
    # 连续难度条件：用窗口占位率（≈密度）归一化到 [0,1]
    occ = Xraw.reshape(len(Xraw), -1).mean(axis=1).astype(np.float32)
    lo, hi = np.percentile(occ, 5), np.percentile(occ, 95)
    cont = np.clip((occ - lo) / max(1e-6, hi - lo), 0, 1).astype(np.float32)
    D = torch.from_numpy(cont)
    print(f'连续难度条件：占位率 {lo:.3f}–{hi:.3f} → [0,1]，均值 {cont.mean():.3f}', flush=True)
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
    # 随机抽子集（带种子）：直接取前 N 个会按歌曲顺序偏置，风格/难度分布会歪
    gsub = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(n_all, generator=gsub)
    if a.max_windows and a.max_windows < n_all:
        order = order[:a.max_windows]
    n = len(order)
    # 只把用到的那批窗口转 float32（省内存），之后 tr/te 用子集内的局部下标
    X = torch.from_numpy(np.ascontiguousarray(Xraw[order.numpy()]).astype(np.float32))
    W = int(X.shape[2])                     # 实际窗口长度（1/32 数据 = 256）
    S = S[order]
    D = D[order]
    if C is not None:
        C = C[order]                 # 条件必须与 X/S/D 同源，包含子集随机重排
    k = int(n * (1 - a.val_frac))
    tr, te = torch.arange(k), torch.arange(k, n)
    print(f'数据 全集 {n_all}，本配置用 {n} 窗口，训练 {len(tr)} / 验证 {len(te)}', flush=True)
    print(f'  子集风格分布 {torch.bincount(S[tr], minlength=4).tolist()}'
          f'  连续难度 [min {float(D[tr].min()):.2f}, max {float(D[tr].max()):.2f}] 均值 {float(D[tr].mean()):.2f}', flush=True)

    if a.init_seed >= 0:
        torch.manual_seed(a.init_seed)       # A/B 配对：同 seed 共享初始权重与噪声流
        print(f'初始种子 {a.init_seed}（配对实验用）', flush=True)
    model = UNet1D(base=a.base, cond_dim=(a.cond_dim if a.audio_cond else 0))
    nparam = sum(p.numel() for p in model.parameters())
    print('参数量 %.3fM（cond_dim=%d）' % (nparam / 1e6, model.cond_dim), flush=True)
    diff = Diffusion(a.timesteps)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    # 验证固定 t 与噪声：否则每轮用不同的噪声量同一把尺子，曲线抖到看不清趋势
    gval = torch.Generator().manual_seed(1234)
    vt = torch.randint(0, a.timesteps, (len(te),), generator=gval)
    vn = torch.randn(len(te), 4, W, generator=gval)
    Xte, Ste, Dte = X[te], S[te], D[te]   # X 已是 [n,4,W]
    Cte = C[te] if C is not None else None

    def eval_loss():
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(te), 512):
                m = min(512, len(te) - i)          # 末块真实长度（不能用 sl.stop-sl.start）
                sl = slice(i, i + m)
                cb = Cte[sl] if Cte is not None else None
                if a.val_fixed:
                    xt = diff.q(Xte[sl], vt[sl], vn[sl])
                    pred = model(xt, vt[sl], Ste[sl], Dte[sl], cb)
                    _, tgt_v, _ = diff.target(Xte[sl], vt[sl], vn[sl], a.pred)
                    l = F.mse_loss(pred, tgt_v)
                else:
                    l = diff.loss(model, Xte[sl], Ste[sl], Dte[sl], a.pred, cb)
                tot += l.item() * m
        return tot / len(te)

    best = 1e9
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        p = tr[torch.randperm(len(tr))]
        tot = 0.0
        nb = 0
        t_first = 0.0
        for i in range(0, len(p), a.batch):
            idx = p[i:i + a.batch]
            tb = time.time()
            opt.zero_grad()
            cb = None
            if C is not None:
                cb = C[idx]
                if a.cond_drop > 0:
                    # classifier-free guidance：按概率把条件置零，让模型同时学会「无条件」去噪
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
        print(f'epoch {ep}: 训练 {tot/len(p):.4f}  验证 {vl:.4f}  '
              f'用时 {ep_s:.0f}s（{nb} batch，{t_first:.0f}ms/batch）', flush=True)
        if a.log:
            os.makedirs(os.path.dirname(a.log) or '.', exist_ok=True)
            with open(a.log, 'a') as f:
                f.write(json.dumps({'epoch': ep, 'train': tot / len(p), 'val': vl,
                                    'sec': round(ep_s, 1), 'batches': nb,
                                    'ms_per_batch': round(t_first),
                                    'lr': round(sched.get_last_lr()[0], 8)},
                                   ensure_ascii=False) + '\n')
        if vl < best:
            best = vl
            os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
            torch.save({'state': model.state_dict(), 'pred': a.pred, 'base': a.base,
                        'model_type': 'unet1d-continuous-fourier-v1',
                        'difficulty_condition': {'type': 'occupancy-percentile-v1',
                                                 'lo': float(lo), 'hi': float(hi),
                                                 'percentiles': [5, 95], 'range': [0.0, 1.0]},
                        'temb_dim': 128, 'timesteps': a.timesteps, 'val': vl,
                        'epoch': ep, 'batch': a.batch, 'rows': X.shape[2], 'grid': '1/32', 'pos_weight': a.pos_weight,
                        'windows': n, 'params': nparam,
                        'cond_dim': model.cond_dim,
                        'cond_channels': 4 if model.cond_dim else 0,
                        'audio_cond': a.audio_cond or None,
                        'cond_drop': a.cond_drop,
                        'data': a.data,
                        'val_idx': te.tolist(), 'val_source_idx': order[te].tolist(),
                        'val_frac': a.val_frac, 'seed': a.seed,
                        'init_seed': a.init_seed,
                        'train_seconds': round(time.time() - t_start, 1)},
                       a.out)
            print(f'  ↳ 保存最优（验证 {vl:.4f}）', flush=True)
        if a.time_budget and ep >= a.min_epochs and (time.time() - t_start) > a.time_budget:
            print(f'时间预算 {a.time_budget}s 到，停在 epoch {ep}', flush=True)
            break
    print(f'完成，最优验证损失 {best:.4f}，总用时 {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
