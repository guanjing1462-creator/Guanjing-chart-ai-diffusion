# 写谱 AI 训练手册（RUNBOOK）

> 目标：音频条件 + 人类分布 + 键型可控的可训练写谱模型（Mug-Diffusion 等价物，自研实现）
> 数据集：36837 张谱 / 36598 特征 / 36837 张量 / ~5.5k 音频对（见 out/dataset-final.json）
> 日期：2026-10-27

## 三阶段训练（顺序不可换）

### 阶段 1：KL-VAE（谱面 → 潜空间）

```sh
python3 tools/train_vae.py --data out/mug_tensors --epochs 20 --batch 8
```

- 输入 [16,4096] → 潜 [16,512]（时间 8×）
- **验收：start 通道 F1 ≥ 0.99**（日志里每 epoch 打印）。不到 0.99 别进阶段 2
- CPU 约 8 分钟/epoch（cap-steps 1400）；GPU 几分钟一轮
- 断点续训：重跑同命令自动载入 out/vae/vae.pt

### 阶段 2：音频特征全量提取（阶段 3 的数据）

```sh
python3 tools/build-audio-features.py --charts out/charts --pool /tmp/osz_x --out out/audio_features
```

- 预计 ~5500 张有音频（2498 个音频 × 多难度）
- 每张 ~1.5s（解码 + mel），约 2.5 小时
- 增量续传，随时可断

### 阶段 3：条件扩散（核心模型）

```sh
python3 tools/train_diffusion.py --epochs 40 --batch 4
```

- 结构：VAE 潜空间 [16,512] 上 1D UNet（eps 预测）；音频 mel 编码后通道拼接；
  21 token 控制（ln_ratio / ett×7 / rc-hb-ln / pattern bool×7）cross-attention；
  训练 10% 条件丢弃，采样 CFG scale 3
- GPU（4090 级）：batch 16、全量步，半天-一天；CPU：cap-steps 适配活跃窗口
- **验收**：生成谱跑 `run-koishi.js`，D 值随控制 token 单调、pattern 判定随
  token 变化（把 technical=1 换 chordjack=1，判定应跟着变）

### 采样（生成谱面）

```sh
python3 tools/train_diffusion.py --sample --ckpt out/diffusion/diff.pt \
    --vae out/vae/vae.pt --audio-npz out/audio_features/<id>.npz \
    --feature '{"ett":25,"technical":1,"hb":1}' -o out/samples/gen.osu
```

- 特征键：ln_ratio / ett / stream_ett… / rc-hb-ln / stream…stamina（bool）
- 生成后建议过一遍规则引擎后处理：`chartgen.py` 的 clean_jacks + grid 吸附
  （模型输出在 46.44ms 轴上，吸附到分音更可玩）

## 数据资产（out/）

| 文件 | 内容 |
|---|---|
| charts/*.osu | 36837 张 4K 谱（Mug 官方清单 31523 + 原有 5463） |
| feature-db.jsonl | 36598 行 21 维控制特征（minacalc 在线口径） |
| mug_tensors/*.npz | 36837 个 [16,4096] 谱面张量 + timing 通道 |
| audio_features/*.npz | 有音频谱的 mel [128,4096]（持续补齐中） |
| msd-full.jsonl | 5254 张历史 MSD（对拍用） |

## 已知约束

- 沙箱空闲会休眠杀后台进程：长训练必须常驻环境（自己的 GPU 机器最合适），
  或在本会话里分多个活跃窗口推进（所有脚本都有断点续传）
- 音频只覆盖 ~15% 谱面：音频条件的扩散阶段训练集 ~5.5k 张（rate 增强后 ~1.2 万），
  比 Mug 的 3.3 万少；先验证管线，扩音频需要带宽
- VAE 当前只训了几轮，F1 未到 0.99：扩散阶段的潜空间会随 VAE 重训而失效，
  **VAE 定稿前不要长训扩散**
