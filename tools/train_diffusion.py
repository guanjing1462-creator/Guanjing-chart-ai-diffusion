#!/usr/bin/env python3
"""条件扩散模型：在 VAE 潜空间 [16,512] 上生成谱面，双条件（音频 mel + 控制 token）。

训练：
    python3 tools/train_diffusion.py --data out/mug_tensors --audio out/audio_features \
        --features out/feature-db.jsonl --vae out/vae/vae.pt --epochs 40

采样（生成 .osu）：
    python3 tools/train_diffusion.py --sample --ckpt out/diffusion/diff.pt \
        --audio-npz out/audio_features/<id>.npz \
        --feature '{"ett":25,"technical":1,"hb":1}' -o out/samples/gen.osu

架构对齐 Mug-Diffusion（见 MUG-DIFFUSION-剖析.md）：
  - 潜空间 [16,512]（KL-VAE 先训好，本脚本冻结加载，只训扩散侧）
  - 音频 mel [128,4096] → 1D 编码器 8× 下采样 → [64,512]，与潜变量通道拼接
  - 控制条件：ln_ratio + ett×7 + rc/hb/ln + pattern bool×7 → 量化 id →
    Embedding [128,20] → 瓶颈 cross-attention；训练时 10% 概率整包丢弃
    （id 全 0 = uncond），采样时 CFG（scale 3）
  - 1000 步线性 β，eps 参数化，smooth_l1；采样 = DDIM 确定性（eta 0）
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
_STEP = [0]
def step_seed():
    return _STEP[0]
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 控制特征词表（Mug mania_beatmap_features 等价物） ----------------
# sr 缺官方口径（osu API 拿不到），用 ett 代难度主口径；rank_status 无信息源，略。
NUMERIC = [('ln_ratio', 0.0, 1.0, 0.1)] + \
          [(k, 5.0, 35.0, 1.0) for k in
           ('ett', 'stream_ett', 'jumpstream_ett', 'handstream_ett',
            'jackspeed_ett', 'chordjack_ett', 'technical_ett')]
BOOL_KEYS = ['rc', 'hb', 'ln', 'stream', 'jumpstream', 'handstream',
             'jackspeed', 'chordjack', 'technical', 'stamina']
TOKEN_KEYS = [k for k, *_ in NUMERIC] + BOOL_KEYS
NULL_ID = 0


def build_vocab():
    """特征名 → {量化值: id}。id 0 保留给 null（CFG uncond）。"""
    vocab, next_id = {}, 1
    for key, lo, hi, step in NUMERIC:
        vals = {}
        v = lo
        while v <= hi + 1e-9:
            vals[round(v, 2)] = next_id
            next_id += 1
            v = round(v + step, 4)
        vocab[key] = vals
    for key in BOOL_KEYS:
        vocab[key] = {0: next_id, 1: next_id + 1}
        next_id += 2
    return vocab


VOCAB = build_vocab()
N_TOKENS = len(TOKEN_KEYS)
NUMERIC_SPEC = {k: (lo, hi, step) for k, lo, hi, step in NUMERIC}


def feature_to_ids(feat, dropout=0.0, rng=None):
    """特征字典 → 20 个 embedding id；dropout 时该 token 置 null。"""
    ids = [NULL_ID] * N_TOKENS
    for i, key in enumerate(TOKEN_KEYS):
        if dropout > 0 and rng is not None and rng.random() < dropout:
            continue
        if key in NUMERIC_SPEC:
            lo, hi, step = NUMERIC_SPEC[key]
            v = float(feat.get(key, lo))
            q = round(round((v - lo) / step) * step + lo, 2)
            ids[i] = VOCAB[key].get(q, NULL_ID)
        else:
            ids[i] = VOCAB[key].get(int(feat.get(key, 0)), NULL_ID)
    return ids


# ---------------- 模型块 ----------------
def gn(ch):
    return nn.GroupNorm(min(8, ch), ch)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    ang = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, temb_dim):
        super().__init__()
        self.n1, self.c1 = gn(ch_in), nn.Conv1d(ch_in, ch_out, 3, padding=1)
        self.n2, self.c2 = gn(ch_out), nn.Conv1d(ch_out, ch_out, 3, padding=1)
        self.temb = nn.Linear(temb_dim, ch_out * 2)
        self.skip = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.temb(F.silu(temb)).chunk(2, 1)
        h = self.n2(h) * (1 + scale[..., None]) + shift[..., None]
        return self.skip(x) + self.c2(F.silu(h))


class CrossAttention(nn.Module):
    """Q = 潜特征帧，K/V = 控制 token [B, dim, 20]。"""

    def __init__(self, ch, ctx_dim=128, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Conv1d(ch, ch, 1)
        self.kv = nn.Linear(ctx_dim, ch * 2)
        self.proj = zero_init_conv(ch, ch)

    def forward(self, x, ctx):
        B, C, T = x.shape
        H = self.heads
        q = self.q(x).reshape(B, H, C // H, T).permute(0, 1, 3, 2)      # [B,H,T,C/H]
        kv = self.kv(ctx.transpose(1, 2))                                # [B,20,2C]
        k, v = kv.chunk(2, 2)                                            # [B,20,C]
        k = k.reshape(B, -1, H, C // H).permute(0, 2, 1, 3)              # [B,H,20,C/H]
        v = v.reshape(B, -1, H, C // H).permute(0, 2, 1, 3)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(C // H), dim=-1)
        out = (att @ v).permute(0, 2, 1, 3).reshape(B, C, T)
        return x + self.proj(out)


def zero_init_conv(ch_in, ch_out):
    c = nn.Conv1d(ch_in, ch_out, 1)
    nn.init.zeros_(c.weight)
    nn.init.zeros_(c.bias)
    return c


class UNet1D(nn.Module):
    """入参 x = cat(潜[16], 音频[64]) = 80ch，512 帧 → eps [16,512]。

    结构：conv_in → 2 级（ResBlock+下采样）→ 瓶颈（ResBlock+cross-attn+ResBlock）
    → 2 级（上采样+ResBlock，跳连拼接）→ conv_out（零初始化）。"""

    def __init__(self, z_ch=16, audio_ch=64, base=128, mult=(1, 2, 2), ctx_dim=128):
        super().__init__()
        temb_dim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, temb_dim), nn.SiLU(),
                                      nn.Linear(temb_dim, temb_dim))
        chs = [base * m for m in mult]
        self.conv_in = nn.Conv1d(z_ch + audio_ch, chs[0], 3, padding=1)
        self.down = nn.ModuleList()
        for i in range(len(mult) - 1):
            ch_in = chs[i - 1] if i > 0 else chs[0]
            self.down.append(nn.ModuleDict({
                'block': ResBlock(ch_in, chs[i], temb_dim),
                'down': nn.Conv1d(chs[i], chs[i], 4, stride=2, padding=1)}))
        self.mid1 = ResBlock(chs[-1], chs[-1], temb_dim)
        self.attn = CrossAttention(chs[-1], ctx_dim)
        self.mid2 = ResBlock(chs[-1], chs[-1], temb_dim)
        self.up = nn.ModuleList()
        for i in range(len(mult) - 1, 0, -1):
            # LDM 标准：先上采样，再与同级 skip 拼接 → 块入 chs[i]+chs[i-1]
            self.up.append(nn.ModuleDict({
                'up': nn.Upsample(scale_factor=2, mode='nearest'),
                'block': ResBlock(chs[i] + chs[i - 1], chs[i - 1], temb_dim)}))
        self.conv_out = zero_init_conv(chs[0], z_ch)

    def forward(self, x, t, ctx):
        temb = self.time_mlp(timestep_embedding(t, 128))
        h = self.conv_in(x)
        skips = []
        for d in self.down:
            h = d['block'](h, temb)
            skips.append(h)
            h = d['down'](h)
        h = self.mid2(self.attn(self.mid1(h, temb), ctx), temb)
        for u in self.up:
            h = u['up'](h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = u['block'](h, temb)
        return self.conv_out(h)


class AudioEncoder(nn.Module):
    """mel [128,4096]（46.44ms/帧）→ [64,512]（3 次 2× 平均池化下采样）。"""

    def __init__(self, n_mels=128, chs=(64, 128, 256, 64)):
        super().__init__()
        self.conv_in = nn.Conv1d(n_mels, chs[0], 3, padding=1)
        stages = []
        for i in range(3):
            stages.append(nn.Sequential(
                gn(chs[i]), nn.SiLU(), nn.Conv1d(chs[i], chs[i + 1], 3, padding=1),
                gn(chs[i + 1]), nn.SiLU(), nn.Conv1d(chs[i + 1], chs[i + 1], 3, padding=1),
                nn.AvgPool1d(2)))
        self.stages = nn.ModuleList(stages)

    def forward(self, mel):
        h = self.conv_in(mel)
        for s in self.stages:
            h = s(h)
        return h


class ControlEmbedder(nn.Module):
    def __init__(self, vocab_size, dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, ids):
        return self.emb(ids).transpose(1, 2)      # [B, 128, 20]


class Mug(nn.Module):
    """音频条件扩散。v11 增补：**onset 支路**。

    为什么加（本机诊断）：mel 编码器下采样 8× 后，一拍内的 onset 信息被
    平均掉；而「鼓点在哪」正是出谱对音的唯一硬依据。原模型只能从 mel 的
    缓慢起伏里猜节奏 → 骨架踩音率卡在 70%。
    这里把「onset 强度 / 低频 onset / 高频 onset」三条逐帧通道用卷积
    8× 下采样后直接加到音频特征上；分支做零初始化，所以加载旧 ckpt
    时行为与原来完全一致，训练中再逐步学会使用它。
    """

    def __init__(self, vocab_size, onset_ch=3):
        super().__init__()
        self.audio_enc = AudioEncoder()
        self.onset_enc = zero_init_conv_seq(onset_ch, 64)
        self.control = ControlEmbedder(vocab_size)
        self.unet = UNet1D()

    def forward(self, z_t, t, ctrl_ids, mel, ons=None):
        ctx = self.control(ctrl_ids)
        w = self.audio_enc(mel)
        if ons is not None:
            w = w + self.onset_enc(ons)
        return self.unet(torch.cat([z_t, w], dim=1), t, ctx)


def zero_init_conv_seq(ch_in, ch_out, stride=8, k=7):
    c = nn.Conv1d(ch_in, ch_out, k, stride=stride, padding=k // 2)
    nn.init.zeros_(c.weight)
    nn.init.zeros_(c.bias)
    return c


def onset_channels(mel):
    """mel [B,128,T] → [B,3,T]（全分辨率）：总 onset 强度 + 低频 + 高频。"""
    eps = 1e-6
    logm = torch.log1p(mel.clamp(min=0))
    def flux(band):
        m = logm[:, band[0]:band[1]]
        d = m[:, :, 1:] - m[:, :, :-1]
        f = F.relu(d).sum(dim=1)
        return F.pad(f, (1, 0))
    tot = flux((0, 128))
    low = flux((0, 24))       # ~0-250Hz：底鼓
    high = flux((80, 128))    # ~3k+：镲/军鼓泛音
    out = torch.stack([tot, low, high], dim=1)
    scale = out.flatten(2).amax(dim=2, keepdim=True) + eps
    return out / scale


# ---------------- 扩散 ----------------
BETAS = torch.linspace(1e-4, 0.02, 1000)
ALPHAS = 1 - BETAS
ACUM = torch.cumprod(ALPHAS, 0)


def q_sample(x0, t, noise):
    a = ACUM[t].view(-1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


def training_loss(model, x0, ctrl, mel, t, ons=None):
    noise = torch.randn_like(x0)
    z_t = q_sample(x0, t, noise)
    eps = model(z_t, t, ctrl, mel, ons)
    return F.smooth_l1_loss(eps, noise)


@torch.no_grad()
def sample(model, mel, ctrl, scale=3.0, steps=50, uncond_ctrl=None, device='cpu', ons=None):
    """DDIM 确定性采样 + CFG。返回 z [1,16,512]（还需 VAE.decode 到帧域）。"""
    model.eval()
    ts = torch.linspace(999, 0, steps).long()
    x = torch.randn(1, 16, 512, device=device)
    for i in range(len(ts)):
        t = ts[i].expand(1)
        eps_c = model(x, t, ctrl, mel, ons)
        eps_u = model(x, t, uncond_ctrl, mel, ons) if scale != 1.0 else eps_c
        eps = eps_u + scale * (eps_c - eps_u)
        a = ACUM[t].view(-1, 1, 1)
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-10, 10)
        if i == len(ts) - 1:
            x = x0
            break
        a_prev = ACUM[ts[i + 1]].view(-1, 1, 1)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    model.train()
    return x


# ---------------- 数据 ----------------
class DiffusionDataset(torch.utils.data.Dataset):
    """(tensor npz, audio npz, feature row) 三元组；仅取有音频的谱。"""

    def __init__(self, tensor_dir, audio_dir, feature_path):
        self.pairs = []
        feats = {}
        for l in open(feature_path):
            d = json.loads(l)
            feats[d['id']] = d
        for ap in glob.glob(os.path.join(audio_dir, '*.npz')):
            cid = os.path.splitext(os.path.basename(ap))[0]
            tp = os.path.join(tensor_dir, f'{cid}.npz')
            if cid in feats and os.path.exists(tp):
                self.pairs.append((tp, ap, feats[cid]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        tp, ap, feat = self.pairs[i]
        note = np.load(tp)['note'].astype(np.float32)          # [16,4096]
        mel = np.load(ap)['mel'].astype(np.float32)            # [128,4096]
        # rate 增强（Mug 同款）：50% 概率 0.9/1.1 变速，音频与谱面
        # 同步缩放时间轴——一份真值变三份，SR/MSD 条件也随 rate 换算。
        r = np.random.RandomState(i + step_seed())
        if r.rand() < 0.5:
            rate = 0.9 if r.rand() < 0.5 else 1.1
            idx = (np.arange(4096) / rate).clip(0, 4095)
            i0 = idx.astype(int); i1 = np.minimum(i0 + 1, 4095); frac = (idx - i0)[None, :]
            mel = (mel[:, i0] * (1 - frac) + mel[:, i1] * frac).astype(np.float32)
            f1 = (idx - i0)                                     # 一维帧插值系数
            for ch in list(range(4)) + list(range(8, 12)):      # 二值通道近邻
                note[ch] = note[ch][i0]
            for ch in list(range(4, 8)) + list(range(12, 16)):  # 偏移通道线性
                note[ch] = np.clip(note[ch][i0] * (1 - f1) + note[ch][i1] * f1, 0, 1)
            feat = dict(feat)
            if feat.get('ett'):
                feat['ett'] = round(feat['ett'] * rate, 2)      # SR/MSD 随 rate 换算
        ons = onset_channels(torch.from_numpy(mel).unsqueeze(0))[0].numpy().astype(np.float32)
        return note, mel, ons, np.array(feature_to_ids(feat), dtype=np.int64)


# ---------------- .osu 写出 ----------------
def sample_to_osu(vae, z, path, bpm=185.0, offset=1263.28, keys=4):
    """z [1,16,512] → VAE.decode → [16,4096] → 音符提取 → 写 .osu。"""
    with torch.no_grad():
        full = vae.dec(z)[0].cpu().numpy()
    FRAME_MS = 2048 / 22050 / 2 * 1000
    objs = []
    for col in range(keys):
        starts = np.where(full[col] > 0.5)[0]
        for s in starts:
            off = float(np.clip(full[col + keys][s], 0, 1))
            t = (s + off) * FRAME_MS
            i = s + 1
            while (i < full.shape[1] and full[col + 2 * keys][i] > 0.5
                   and full[col][i] <= 0.5):
                i += 1
            end = None
            if i - 1 > s:
                e_off = float(np.clip(full[col + 3 * keys][i - 1], 0, 1))
                end = (i - 1 + e_off) * FRAME_MS
            objs.append((t, col, end))
    objs.sort()
    L = ['osu file format v14', '', '[General]', 'AudioFilename: audio.mp3', 'Mode: 3',
         '', '[Metadata]', 'Title: generated', 'Artist: mug-prototype', 'Version: AI', '',
         '[Difficulty]', f'CircleSize: {keys}', '', '[TimingPoints]',
         f'{offset},{60000 / bpm},4,2,1,20,1,0', '', '[HitObjects]']
    colw = 512 // keys
    for t, col, end in objs:
        x = int((col + 0.5) * colw)
        if end:
            L.append(f'{x},192,{int(t)},128,0,{int(end)}:0:0:0:0:')
        else:
            L.append(f'{x},192,{int(t)},1,0,0:0:0:0:')
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    return len(objs)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='out/mug_tensors')
    ap.add_argument('--audio', default='out/audio_features')
    ap.add_argument('--features', default='out/feature-db.jsonl')
    ap.add_argument('--vae', default='out/vae/vae.pt')
    ap.add_argument('--out', default='out/diffusion')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--cap-steps', type=int, default=800,
                    help='每 epoch 步数上限（沙箱活跃窗口 ~10 分钟适配）')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--ckpt', default='out/diffusion/diff.pt')
    ap.add_argument('--audio-npz', default=None)
    ap.add_argument('--feature', default=None)
    ap.add_argument('-o', default='out/samples/gen.osu')
    args = ap.parse_args()

    vocab_size = max(max(v.values()) for v in VOCAB.values()) + 1
    model = Mug(vocab_size)

    if args.sample:
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=False)
        vae = VAE()
        vae.load_state_dict(torch.load(args.vae, map_location='cpu'))
        mel = torch.from_numpy(np.load(args.audio_npz)['mel'].astype(np.float32)).unsqueeze(0)
        ons = onset_channels(mel)
        feat = json.loads(args.feature) if args.feature else {}
        ctrl = torch.tensor([feature_to_ids(feat)], dtype=torch.int64)
        uncond = torch.tensor([feature_to_ids({})], dtype=torch.int64)
        z = sample(model, mel, ctrl, scale=3.0, uncond_ctrl=uncond, ons=ons)
        os.makedirs(os.path.dirname(args.o) or '.', exist_ok=True)
        n = sample_to_osu(vae, z, args.o)
        print(f'写出 {args.o}（{n} 音符）')
        return

    ds = DiffusionDataset(args.data, args.audio, args.features)
    print(f'音频条件训练对: {len(ds)}')
    if len(ds) == 0:
        sys.exit('没有 (tensor, audio, feature) 三元组——先跑 build-audio-features.py')
    n_val = max(1, len(ds) // 20)
    train_ds = torch.utils.data.Subset(ds, range(n_val, len(ds)))
    val_ds = torch.utils.data.Subset(ds, range(n_val))

    ckpt = os.path.join(args.out, 'diff.pt')
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location='cpu')
        miss, unexp = model.load_state_dict(sd, strict=False)
        print('续训：载入', ckpt, f'（新增 {len(miss)} 个 key：{list(miss)[:3]}）')
    # VAE 冻结加载：训练目标是"VAE 潜空间上的 eps"，不是帧域
    vae = VAE()
    vae.load_state_dict(torch.load(args.vae, map_location='cpu'))
    vae.eval()
    for p_ in vae.parameters():
        p_.requires_grad_(False)

    def encode_notes(x):
        with torch.no_grad():
            mu, _ = vae.enc(x)
        return mu

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.RandomState(7)
    step = 0
    for ep in range(args.epochs):
        perm = rng.permutation(len(train_ds))
        n_steps = min(args.cap_steps, len(perm) // args.batch)
        tot = 0.0
        model.train()
        for it in range(n_steps):
            idx = perm[it * args.batch:(it + 1) * args.batch]
            xs, ms, cs, os_ = [], [], [], []
            for j in idx:
                note, mel, ons, ids = train_ds[j]
                xs.append(note)
                ms.append(mel)
                os_.append(ons)
                feat = train_ds.dataset.pairs[train_ds.indices[j]][2]
                cs.append(feature_to_ids(feat, dropout=0.1, rng=np.random))
            x0 = torch.from_numpy(np.stack(xs))
            mel = torch.from_numpy(np.stack(ms))
            ctrl = torch.tensor(np.stack(cs), dtype=torch.int64)
            ons = torch.from_numpy(np.stack(os_))
            t = torch.randint(0, 1000, (x0.shape[0],))
            x0_lat = encode_notes(x0)
            loss = training_loss(model, x0_lat, ctrl, mel, t, ons)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            step += 1
            _STEP[0] = step
        print(f'epoch {ep}: loss {tot/max(1,n_steps):.4f} ({n_steps} 步)', flush=True)
        torch.save(model.state_dict(), ckpt)


from train_vae import VAE  # noqa: E402  （sample 模式需要 VAE decode）

if __name__ == '__main__':
    main()
