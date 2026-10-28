# Mug-Diffusion 剖析报告 —— 每一层怎么工作，以及能偷什么

> 仓库：https://github.com/Keytoyze/Mug-Diffusion（4K VSRG 谱面生成，Stable Diffusion 改）
> 剖析方式：模型配置 YAML + 全部源码逐文件读（本地 `mugsrc/` 3020 行 = 数据管线全部，
> `mug/` 模型包经 GitHub 网页端逐文件抓取）。权重本体不可得（见 §9），但代码 100% 可读，
> 架构、张量形状、控制接口、增强体系全部还原如下。
> 日期：2026-10-26

---

## 0. 一句话总结

**把谱面当成一段 16 通道的"时间序列图像"，用 KL-VAE 压 8 倍，再用 1D UNet 扩散模型
在音频多尺度特征（通道拼接）+ 21 个风格/难度 token（cross-attention）双条件下
去噪生成，最后解码回 .osu。** 它不用音乐网格——整曲铺在固定 46.44ms 时间轴上。

---

## 1. 端到端管线

```
.osu + 音频
   │ scripts_prepare_beatmap_features.py（离线，一次）
   ▼
feature.db（每张谱一行：sr/rank_status/ln_ratio/rc/hb/ln/ett+7×MSD+7×bool）
   │
   │ 训练时在线（ds.py OsuDataset.__getitem__）
   ▼
┌─────────────────────────────────────────────────────────────┐
│ note:   [16, 4096]  谱面张量（46.44ms/帧）                    │
│ audio:  [128, 32768] mel 频谱（5.805ms/帧）                   │
│ feature: [21]     量化后的 embedding id 序列                  │
└─────────────────────────────────────────────────────────────┘
   │                    │                    │
   │ AutoencoderKL      │ Melspectrogram     │ BeatmapFeature
   │ (时间 8× 压缩)      │ ScaleEncoder1D     │ Embedder
   ▼                    ▼                    ▼
z: [16,512]      w: 多尺度 [256,512,512,512]   c: [128, 21]
   │                    │（逐级通道拼接）          │（cross-attention）
   └──────► 1D UNet（eps 预测，1000 步 DDPM，推理用 DDIM 100 步 + CFG 5）
            │
            ▼
        解码 z → [16,4096] → 逐列还原音符 → gridify 对齐网格
        → remove_intractable_mania_mini_jacks(90ms) → .osu
```

---

## 2. 谱面表示：16 通道 × 4096 帧（mug/data/convertor.py）

这是全仓库最值得抄的设计。**不用拍子网格，用固定时间轴**：

| 参数 | 值 | 来源 |
|---|---|---|
| 音频帧 | n_fft=512, hop=128 样本 @22050Hz → **5.805ms/帧** | configs/mug/mug_diffusion.yaml |
| 谱面帧 | audio_note_window_ratio=8 → **46.44ms/帧** | 同上 |
| 序列长 | max_audio_frame=32768 帧 → **4096 note 帧 = 190.2s** | 同上 |
| 通道 | **key_count × 4 = 16**（4K） | convertor.py Feature Layout |

每列 4 个通道的语义（这是它表达"毫秒精度"的钥匙）：

```
ch [0..K)   is_start      该帧该列是否有音符头（0/1）
ch [K..2K)  offset_start  头在本帧内的亚帧偏移（0-1，×46.44ms = 毫秒精度）
ch [2K..3K) is_holding    LN 身体：头之后、尾之前，含尾帧不含头帧
ch [3K..4K) offset_end    尾的亚帧偏移（仅"这帧还按着、下帧松开"时有效）
```

**LN 的语义是"按住直到同列下一个头出现"**——解码时从 head 沿 is_holding 走到
断掉的那帧。这把"LN 长度生成"这个难点转化成了"同列下一个头何时来"，与
tap 共用同一套生成机制。副作用：同帧起止的 hold 会退化为 tap（最小 LN = 2 帧
= 92.9ms），以及所有 LN 尾都挂在"下一个头"上，无法表达"长按后留白"。

另有 timing 通道（2 通道）：红色 timing line 按 **半拍**（60000/bpm/2）打点，
记录每拍所在帧 + 亚帧偏移，SV 自动归一到 150-300bpm 区间。生成的谱
`gridify()` 时用它反推 BPM/offset。

`valid_flag`：实际谱长（末音 + 边界）之前的帧置 1，之后为 0——损失和采样
都只在有效区算。

**对照我们的管线**：我们此前用"1/16 音乐网格 + 爆发段 1/32/1/24/1/48 细分"，
优点是分音可读，缺点是音符时刻被格子钳制。Mug 的 46ms 固定轴 + 亚帧偏移
是更彻底的"贴音频"，代价是输出必须靠 gridify 事后吸附回分音（它的
`ut.timing()` 用 beat 检测 + `test_timing` 暴力搜相位）。我们目前的
"听写式选格"相当于把这两步合并了，保留。

---

## 3. 音频条件：多尺度 mel 编码器（mug/cond/wave.py）

- 输入 mel `[128, 32768]`（128 mel 带 × 190s）
- `MelspectrogramScaleEncoder1D`：conv_in 1D → 10 级下采样
  （channel_mult `[1,1,1,1,2,2,2,4,4,4]`，每级 2 个 ResBlock，
  空洞卷积 dilation `(1,2)` / `(4,8)` 交替，GroupNorm 32）
- 分辨率 ∈ {128,256,512} 时插 `ContextualTransformer`（自注意力）
- **输出是多尺度特征列表**：每级一个 `[C_level, T_level]`，最细 512ch×32768
  → 最粗 512ch×64。UNet 按分辨率各取所需。

为什么不用一个池化向量？因为谱面和音频是**逐帧对齐**的——UNet 在第 i 级
处理时间位置 t 时，需要"此刻音频在响什么"，通道拼接提供了这种逐位置条件。

---

## 4. 控制条件：21 个量化 token（mug/cond/feature.py + yaml）

| token | 类型 | 量化 |
|---|---|---|
| sr | numeric | 1-8，步 0.2（osu 星数） |
| rank_status | category | ranked / loved / graveyard |
| ln_ratio | numeric | 0-1，步 0.1 |
| rc / hb / ln | bool | rice图 / 混合图 / LN图 |
| ett + 7×`*_ett` | numeric | 5-35，步 1（Etterna overall + 6 技能维） |
| 7×`*`（stream 等） | bool | "这张图是不是该技能主导" |

- 每个特征是一个 token，量化值 → `nn.Embedding(总词表, 128)` → **[128, 21]**，
  作为 UNet cross-attention 的 K/V。
- **标签不是预存的，是训练时在线算的**：`ds.py load_feature()` 每次取样本都
  现场跑 minacalc（`minacalc.calc_skill_set(rate, notes)`）重算 7 维 MSD，
  bool 判据 = "该技能分与最高技能分差 ≤1"。
- 训练时 **feature dropout 0.5**：每个特征独立以 50% 概率被丢——这是它
  推理时能做 CFG 的根基（推理的 uncond 就是空字典 `{}` 映射到的全零 id）。

**rate 增强的联动换算**（很精妙）：以 0.9-1.1× 变速播放音频（cv2.resize mel），
谱面时间戳同步缩放，然后 SR 按 `star_ratio = 0.8184×(rate-1)+1` 换算，
MSD 由 minacalc 按 rate 重算——一份谱免费变成变速后的多份"真值对"。

**webui 的控制语义**：难度滑条（SR 或 MSD 二选一）、map type 三选一
（rice/hybrid/LN）、ln_ratio 滑条、7 个 pattern 各有 "more/less" 开关 +
MSD 滑条。**不勾选 = 自由发挥**（特征被 dropout 的效果反过来用）。

---

## 5. 扩散主体（mug/diffusion/diffusion.py + unet.py）

| 组件 | 配置 |
|---|---|
| first stage | AutoencoderKL：输入 16ch，channel_mult [1,2,4,4]，kl_weight 1e-6，**时间压 8×**（4096→512 帧/62.5s×8） |
| 潜变量 | z: **[16, 512]**，encode 返回分布取 `.mode()`（不是确定性张量） |
| UNet | 1D，model_channels=128，channel_mult [1,2,3,4]，attn @ [8,4,2]，heads 8，context_dim 128，dropout 0 |
| 扩散 | 1000 步 linear β (1e-4→0.02)，**eps 参数化，smooth_l1 + 0.01** |
| 音频注入 | **通道拼接**（AudioConcatBlock）：每级 ResBlock 前 `cat([h, audio_level])`，各级通道 [256,512,512,512] |
| 特征注入 | **cross-attention**（ContextualTransformer，K/V = 21 token） |
| S4 层 | `s4_layer: true`：每级每个 ResBlock 后挂 S4（Structured State Space），残差 + zero_module 初始化——补卷积/局部 attention 的长程感受野 |
| 训练 | lr 1e-5，batch 48，AdamW，loss_ratio 按分布 std 归一化监控 |

**训练时没有 CFG dropout 逻辑**（LDM 的 cond_dropout 没搬过来）——条件丢弃
发生在 dataset 层（feature dropout 0.5）。推理时 CFG 用空特征字典当 uncond，
**音频永远有**（无法生成"无音频"的谱面，这是合理的先验绑定）。

**采样即落盘**：`log_beatmap` 用祖先采样（训练日志用），webui 用
DDIMSampler（eta=0，默认 100 步，CFG scale 默认 5），z clamp 到 [-10,10]，
解码后 `convertor.array_to_objects` 逐列还原 → `gridify()` 吸附网格 →
`remove_intractable_mania_mini_jacks(90ms)` 清理 <90ms 的迷你纵连 → 写 .osu。

---

## 6. 后处理的网格吸附（mug/data/utils.py）

`gridify()`：先用 `timing()` 从谱面音符间隔里**猜 BPM/offset**
（`test_timing` 对候选 BPM 逐个验证吻合率），再把所有音符吸附到
1/16 网格。也就是说 Mug 生成的谱面本来不在分音上，**分音是事后算出来的**。
`remove_intractable_mania_mini_jacks(jack_interval=90)`：把 <90ms 的同列
连击重排——90ms 是 166.67bpm 的 1/4 拍。

---

## 7. 增强体系（ds.py __getitem__）

| 增强 | 概率 | 做法 |
|---|---|---|
| mirror | 0.5 | 列镜像 |
| random column | 可选 | 列随机置换（破坏手型先验，默认关） |
| rate | 0.2 | 音频 mel 时间缩放 0.9-1.1× + 谱面时间同步 + SR/MSD 换算重算 |
| pitch | 0.2 | mel 频谱上下平移 1-4 bin（半音近似） |
| freq mask | 0 | 频带置零（默认关） |
| feature dropout | **0.5** | 每个控制特征独立丢弃 |

---

## 8. 能偷什么 → 已偷 / 待偷清单

**已落地到我们算法（本轮）**：
1. **LN 层生成规则全部重写**（add_holds v2.2）：从我们自己的 5463 张
   人类语料挖出「LN 占比 27-31%、61% 是 <0.5 拍短条、8s 窗 cv≈0.93 成段、
   53% LN 头踩和弦行、释放后留 ≥1 格」五条规律，替代旧的
   「长度=间隔×0.62 + 均匀分桶」。修完站点 D 不掉（14.13→14.13，tech 不变）。
   —— Mug 没有教这个，是它的**数据思路**教的：控制信号要从人类数据里量出来。

**对齐结论（已确认的设计差异，不抄）**：
2. Mug 的 46ms 固定轴 + 亚帧偏移 vs 我们的 1/16 格 + 四分音细分：
   我们要"分音可读"，它要"贴音频自由度"。两者已验证都能到 D14；
   我们的 grid 约束反而是可玩性资产，保留。
3. 音频多尺度 concat：我们不做扩散，规则引擎用 accent 逐格量能量
   等价于"最细尺度的音频条件"。若走学习路线（§10），照抄这个多尺度结构。

**待偷（写进 TRAINING-PLAN）**：
4. **rate 增强换算公式**：0.9-1.1× 变速 + SR 换算 + minacalc 在线重算——
   我们的 5463 张语料可免费扩 ~2.2 倍，且天然获得"同曲多难度"对。
5. **控制接口设计**：ln_ratio / ett / 7 pattern bool 的量化 embedding +
   "不勾选 = 自由"语义——我们生成器的 `--weights/--chord/--ln` 旋钮
   应该统一成同一套特征向量，为学习模型做准备。
6. **S4/长程层**：规则引擎的段落调度（8s 段 + 能量排序发牌）是手工版
   "长程一致性"；若训练，bottleneck 加 S4 或在 512 潜长上用线性注意力。
7. **LN = hold-until-next-head** 的表示法：若训练我们自己的模型，
   直接用它的 4 通道/列编码（含 timing 通道），比我们的 token 序列
   （chart-tokenizer.js）更省解码复杂度。

---

## 9. 权重获取结论（为什么拿不到，以及影响）

- 仓库 `models/ckpt` 已被作者删除（commit "Update: remove ckpt"），
  权重只随 **Windows 打包 exe** 发布（Google Drive + hiosu 两个链接）。
- 本沙箱实测：hiosu 直链无响应，drive.google.com 连接失败（000），
  GitHub 直连也被墙（git clone 133s 超时）——只有网页代理能通。
- 即使拿到：本机无 GPU（torch 2.10 CPU-only），latent diffusion 推理
  3 分钟音频在 CPU 上预计 30-60 分钟/张。
- **实际影响为零**：架构、编码、控制接口、后处理全部已从代码还原（本文档）。
  Mug 的真正资产是"谱面=16ch 时间序列 + 双条件 + 在线 MSD 标签"这套设计，
  不是那 1GB 权重——我们的语料（5463 张谱 + 2498 个音频/4.3GB + 34000 张
  官方数据集清单）足以自己训练。

## 10. 下一步：用这套设计训练我们自己的模型

数据已就位：`out/dataset.jsonl`（406 张已 token 化）+ `out/charts/*.osu`
（5463 张）+ `/tmp/osz_x`（694 套、2498 个音频）+ `mugdiffusion-trainset.jsonl`
（34000 张官方清单）。

分三步（见 TRAINING-PLAN.md 更新）：
1. **补数据**：跑 scripts_prepare_beatmap_features.py 等价物，把 5463 张谱
   的 feature.db 建出来（minacalc 批量 + sr 抓取），音频解码缓存对齐
   46.44ms 帧。
2. **先训 autoencoder**：16ch×4096 → [16,512]，CPU 可跑（一晚上量级），
   重建损失用 Mug 注释掉的 ManiaReconstructLoss 思路（头/保持/尾分段加权）。
3. **再训条件扩散**：UNet 1D 缩小版（model_channels 64），音频条件先用
   我们现成的 accent/energy 特征替代 mel 多尺度（管线已通），特征 token
   照抄 §4 的 21 维——推理时我们的规则引擎做 gridify 后处理，
   模型只负责"人类的图案与密度调度"这一层。
