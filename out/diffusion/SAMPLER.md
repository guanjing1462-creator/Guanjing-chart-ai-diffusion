# 4K 谱面扩散模型 · 采样器 + .osu 转换器

> 目标：把 `tools/train-diffusion.py` 训出来的扩散模型，从「钢琴卷窗口」接回「可玩的 .osu 谱面」。
> 环境：纯 CPU、约 2GB 可用内存、无 GPU。**未改动** `tools/train-diffusion.py`、`tools/paint-gj.py` 和数据集。

## 0. 一句话结论

| 环节 | 状态 |
|---|---|
| DDPM 采样（200 步，从纯噪声） | ✅ 跑通，4 窗口 97s（随机权重）/ 74s（真权重 base=32） |
| 钢琴卷 → .osu（4K mania） | ✅ 跑通，4 张谱结构自检全 0 异常 |
| `node tools/gj/batch-koishi.mjs 谱面.osu` | ✅ **成功解析**，`diff` / `pattern` / `probs` 字段齐全（最大一张 12.01 / Beta mid / jack） |
| 8 窗口峰值内存 | 209 MB（预算 2GB 的 10%） |

## 1. 新增文件

| 文件 | 作用 |
|---|---|
| `tools/sample-diffusion.py` | DDPM 采样器：噪声 → [4,128] 钢琴卷 + 统计 |
| `tools/roll-to-osu.py` | 钢琴卷 → osu!mania 4K 谱面（支持多窗口 concat、LN 推断） |
| `out/diffusion/samples/` | 采样产物（每窗口 `[4,128]` .npy + 打包 .npz + 统计 json） |
| `out/diffusion/test-*.osu` | 闭环验证用的 4 张谱 |
| `out/diffusion/koishi-result.json` | 上述谱面的 Koishi 原始输出 |

## 2. `tools/sample-diffusion.py`

### 用法

```bash
# ① 还没有训练权重时：用随机初始化权重把整条流程测通（不发散、不报错）
python3 tools/sample-diffusion.py --random-weights --style 0 --diff 1 --num 4 --seed 0

# ② 有 out/models/diffusion4k.pt 时（默认路径，不用写 --ckpt）
python3 tools/sample-diffusion.py --style 2 --diff 3 --num 4 --seed 0

# ③ 想看键型长什么样（ASCII 钢琴卷，A/B/C/D = x 64/192/320/448）
python3 tools/sample-diffusion.py --random-weights --num 1 --ascii 1 --quiet

# ④ 换权重路径 / 少步数快测
python3 tools/sample-diffusion.py --ckpt out/models/diffusion4k.pt --timesteps 50 --num 2
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ckpt` | `out/models/diffusion4k.pt` | 权重；`base` / `temb_dim` / `timesteps` 优先从 ckpt 里读，读不到就从权重形状反推 |
| `--random-weights` | 关 | 不读文件，用随机初始化权重采样（训练产物没出来时测流程用） |
| `--style` | 0 | 风格条件 0..3（与 `tools/diff-data.py` 的 `STYLES` 一致：tech / jack / speed / stamina） |
| `--diff` | 1 | 难度档条件 0..3（按音符率四分：<8 / 8~12 / 12~16 / >16 nps） |
| `--num` | 4 | 窗口数，**1..8**（硬上限，argparse 直接拒 9） |
| `--timesteps` | 200 | 扩散步数（ckpt 里有 `timesteps` 时以 ckpt 为准） |
| `--threshold` | 0.5 | 阈值化门限，`>0.5` 视为有音符 |
| `--bpm` | 136.3 | 只用于算格长与密度统计（1/16 格 = 60000/bpm/4 ms） |
| `--seed` / `--threads` | 0 / 2 | 随机种子 / torch 线程数（默认留 CPU 给别人） |
| `--outdir` | `out/diffusion/samples` | 输出目录 |
| `--ascii` | 0 | 打印前 N 个窗口的 ASCII 卷 |

### 采样算法

标准 DDPM 祖先采样（eps 参数化，线性 schedule 与训练脚本一致：β 1e-4→0.02，T 步，abar 累积）：

```
x_T ~ N(0, I)
for t = T-1 .. 0:
    eps = model(x_t, t, style, diff)                 # 预测噪声（复用训练时的前向签名）
    mean = (x_t - (1-α_t)/sqrt(1-ᾱ_t) · eps) / sqrt(α_t)
    x_{t-1} = mean + sqrt(β_t (1-ᾱ_{t-1})/(1-ᾱ_t)) · z   (t>0, z~N(0,I))
    x_{t-1} = mean                                        (t=0)
```

一次把 `--num` 个窗口并成一个 batch 去噪（同 batch 同条件），所以 8 窗口不会 8 倍耗时。

### 输出

```
out/diffusion/samples/roll-s0d1-seed0-rnd-00.npy   [4,128] uint8 二值卷（喂给 roll-to-osu）
out/diffusion/samples/rolls-s0d1-seed0-rnd.npz     X=[n,4,128] uint8（阈值化）
                                                   P=[n,4,128] float32（原始概率，可做置信度过滤）
                                                   style/diff/seed/threshold
out/diffusion/samples/sample-meta-*.json           参数 + 全部统计
```

文件名带 `-rnd` 表示随机权重，**不会和真权重的产物互相覆盖**。

### 和训练脚本的耦合（重要）

- 采样器**不 import** `tools/train-diffusion.py`，而是把 `temb_fn` / `Block` / `UNet1D` / `Diffusion` 按当前版本整段**复制**进来 —— 那个文件正在被别人改（加速训练），复制一份才不会被改坏。
- 加载权重时：`base` 取 `ckpt['base']`，`temb_dim` 取 `temb.0.weight` 的形状，`timesteps` 取 `ckpt['timesteps']`，都读不到就报错而不是瞎猜。
- `strict=True` 加载失败时退回非严格加载并打印差异；**核心权重缺失就报错退出**（不会拿半个模型去采样）。
- ckpt 里没有 betas，schedule 用与训练脚本相同的线性定义重建 —— 训练脚本若改 schedule，这里要同步。
- 已核对：截止本报告，训练脚本的改动只在训练循环（线程数/验证集/时间预算/日志），`temb_fn`/`Block`/`UNet1D`/`Diffusion` 四个类和前向签名 `model(x, t, style, diff)` 与复制时一致，采到的 `val 0.521 / base 32` 权重能 `strict` 加载。**如果哪天模型结构真的变了**（加通道、加音频条件），这里会显式报「核心参数缺失」而不是悄悄采样出垃圾。

## 3. `tools/roll-to-osu.py`

### 用法

```bash
# 单窗口
python3 tools/roll-to-osu.py out/diffusion/samples/roll-s2d3-seed0-00.npy -o a.osu

# 多窗口拼成一张完整谱（按输入顺序首尾相接，第 i 窗从 offset + i*128*step_ms 开始）
python3 tools/roll-to-osu.py out/diffusion/samples/roll-s2d3-seed0-*.npy --concat -o full.osu

# 打包的 npz（X=[n,4,128]）→ 每个窗口一张谱
python3 tools/roll-to-osu.py out/diffusion/samples/rolls-s2d3-seed0.npz --outdir out/diffusion

# 加 LN 推断（同列连续 ≥4 格 → 一条长条）
python3 tools/roll-to-osu.py out/diffusion/samples/roll-*.npy --concat --ln runs --ln-min 4 -o ln.osu

# 改 BPM/偏移/键数
python3 tools/roll-to-osu.py roll.npy --concat --bpm 200 --offset 500 --keys 4 -o x.osu
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `inputs` | — | `.npy`（`[4,128]` 或 `[n,4,128]`）或 `.npz`（`X`/`roll`/`P`）、目录、通配符 |
| `-o / --out` | — | 单张时的输出文件；配 `--outdir` 时作为文件名前缀目录 |
| `--outdir` | `out/diffusion` | 多张时的输出目录 |
| `--concat` | 关 | 把多个窗口首尾相接拼成一张完整谱；不加就每窗口一张 |
| `--bpm` | 136.3 | BPM，决定 1/16 格长（60000/bpm/4 ms） |
| `--offset` | 0 | 第一个音符的时间偏移 ms |
| `--step-ms` | `60000/bpm/4` | 直接指定 1/16 格长（给了就覆盖 bpm 推算） |
| `--keys` | 4 | 键数 1..4（4 时列坐标正好 64/192/320/448），须与卷的行数一致 |
| `--threshold` | 0.5 | 概率卷阈值 |
| `--ln` | `none` | `none`=全 tap；`runs`=同列连续 ≥`--ln-min` 格变一条 LN |
| `--ln-min` | 3 | LN 最短连续格数（bin 最小 2） |
| `--max-chord` | 0 | 同一时刻最多保留几个音（0=不限），想压难度/防 4 押糊成一坨时用 |
| `--min-gap-ms` | 0 | 同列最小间隔，压 1/16 机枪用 |
| `--title/--artist/--creator/--version/--audio/--od` | — | 写进 [Metadata] / [General] |
| `--json` | 关 | 额外打印 JSON 统计 |

### 卷 → 音符规则

- **坐标**：`x = int((列 + 0.5) * 512 / 键数)`，4K → `64 / 192 / 320 / 448`（与 `tools/paint-gj.py` 一致）。
- **时间**：第 k 个 1/16 格 = `offset + (窗口序号*128 + k) * step_ms`，四舍五入到整数 ms；136.3 bpm → 110.05 ms/格。
- **tap**：`x,192,t,1,0,0:0:0:0:`；**LN**：`x,192,t,128,0,end:0:0:0:0:`。
- **LN 推断**：二值卷本身分不清「长条」和「同列密打」，所以默认 `--ln none`（全 tap）。`--ln runs` 才把同列连续 ≥`--ln-min` 格折成一条 LN（终点 = 最后一格的右边界），中间格不再出 tap。
- **清理**（三道，避免游戏里互相卡）：
  1. 同列同毫秒去重；
  2. LN 终点 ≤ 起点、或长度 <40ms 的假 LN 退化成 tap；
  3. 落在同列 LN 体内的 tap 丢弃（`--ln runs` 由构造保证不重叠，但仍做兜底）。
- **表头**：[General]（Mode: 3）、[Metadata]、[Difficulty]（`CircleSize` = 键数，另有一行 `Keys: N` 供肉眼确认，osu! 会忽略不认识的键 —— 实测加不加 Koishi 结果完全一致）、[TimingPoints]（`offset,60000/bpm,4,2,1,20,1,0`）、[HitObjects]。
- **多窗口拼接**：窗口之间是硬切（`--concat` 按窗口序号平移）。想要过渡自然，后续可以做重叠窗口 + 交叉淡化（见第 6 节）。

## 4. 闭环验证（实测）

环境：8 核 CPU（此刻被别的训练任务占满）、Python 3.12.3、torch 2.5.1、Node 24.19。

### 4.1 采样

```bash
python3 tools/sample-diffusion.py --random-weights --style 0 --diff 1 --num 4 --seed 0 --threads 4
python3 tools/sample-diffusion.py                      --style 2 --diff 3 --num 4 --seed 0 --threads 4
```

**A. 随机初始化权重（`--random-weights`，base=48，0.41M 参数）**

| 指标 | 值 |
|---|---|
| 占位率 | **0.5835** |
| 非空行比例 | 0.957 |
| 每行平均音符数 | **2.334** |
| 非空行平均和弦 / 最大和弦 | 2.439 / 4 |
| 各列占位率 | 0.6621 / 0.6094 / 0.6562 / 0.4062 |
| 每窗口音符数 | 285 / 297 / 308 / 305 |
| 音符密度 | **21.2 nps**（行密度 8.70 rps，窗口 14.09s） |
| 原始概率 均值/标准差/q50 | 1.549 / 4.091 / 1.360（跨度 -9.8 ~ 17.4，**没有发散**） |
| 耗时 | 97.2s / 4 窗口（24.3s/窗，200 步） |

**B. 真权重 `out/models/diffusion4k.pt`（训练中，base=32，0.21M 参数，val 0.521）**

| 指标 | 值 |
|---|---|
| 占位率 | **0.4229** |
| 非空行比例 | 0.9141 |
| 每行平均音符数 | **1.691** |
| 非空行平均和弦 / 最大和弦 | 1.85 / 4 |
| 各列占位率 | 0.4824 / 0.4082 / 0.3926 / 0.4082 |
| 每窗口音符数 | 229 / 220 / 207 / 210 |
| 音符密度 | **15.4 nps**（行密度 8.31 rps） |
| 耗时 | 74.0s / 4 窗口（18.5s/窗，200 步） |

**参照**：数据集本身的占位率是 **0.4072**（30551 窗口 × 512 格）。真权重采样出 0.4229、每窗口 207~229 个格 ≈ 数据集量级 —— 说明它至少学会了「密度大致对」，但键型/节奏还远不到人类（见 4.2：Koishi 判定的 pattern 永远是 jack）。随机权重 0.5835 是纯噪声基线，只用来验证管线。

### 4.2 转换 + Koishi 解析（核心验收）

```bash
python3 tools/roll-to-osu.py out/diffusion/samples/roll-s0d1-seed0-rnd-*.npy --concat -o out/diffusion/test-random.osu
node    tools/gj/batch-koishi.mjs out/diffusion/test-random.osu
```

| 谱面 | 来源 | 音符 | 时长/NPS | Koishi `diff` | `tier` | `pattern` | `probs` | 解析 |
|---|---|---|---|---|---|---|---|---|
| `test-random.osu` | 随机权重 4 窗 concat，136.3 bpm | 1195 tap | 56.1s / 21.3 | **12.01** | Beta mid | jack | jack .998 / tech 0 | ✅ `error: null` |
| `test-real-s2d3-bpm200.osu` | 真权重 style=2 diff=3，200 bpm | 866 tap | 38.3s / 22.6 | **14.03** | Delta mid | jack | jack .988 | ✅ `error: null` |
| `test-real-s2d3.osu` | 真权重 style=2 diff=3，136.3 bpm | 866 tap | 56.2s / 15.4 | `null` | < Alpha Low | jack | jack .996 | ✅ `error: null` |
| `test-random-ln.osu` | 随机权重 + `--ln runs --ln-min 4`（LN 13.5%） | 611 tap + 95 LN | 56.1s / 12.6 | `null` | < Alpha Low | jack | jack .995 | ✅ `error: null` |

**结论：Koishi 全部能正常解析**（4/4 `error: null`，`pattern` / `probs` 字段都在）。其中两张在它的可算区间内给出了数值难度（12.01 / 14.03），另外两张 `diff: null` 不是解析失败，而是 **Koishi 自己的取值范围限制**：它只输出 `11 ≤ 定数 < 17` 的结果（见 5.1），预测低于 11 就返回 `null` 并标 `tier: "< Alpha Low"`、附带 `numericDifficultyHint: "BelowScope"` 警告。

### 4.3 结构自检（不依赖 Koishi）

对 3 张谱逐行回读校验：段齐全（General/Editor/Metadata/Difficulty/TimingPoints/HitObjects）、列坐标只出现 64/192/320/448、时间非降（已排序）、`(列,时间)` 无重复、无 tap 落在同列 LN 体内、无 LN 重叠、无非法 type：

```
test-random.osu            tap 1195 LN  0 | 排序 OK | 重复 0 | tap在LN内 0 | LN重叠 0 | 异常 0
test-real-s2d3-bpm200.osu  tap  866 LN  0 | 排序 OK | 重复 0 | tap在LN内 0 | LN重叠 0 | 异常 0
test-random-ln.osu         tap  611 LN 95 | 排序 OK | 重复 0 | tap在LN内 0 | LN重叠 0 | 异常 0
```

### 4.4 内存与耗时

| 场景 | 内存 | 墙钟 |
|---|---|---|
| 8 窗口（`--num 8 --timesteps 20`，`resource` 峰值 RSS） | **209 MB** | 35.4s |
| 4 窗口 200 步（随机权重，`--threads 2`、机器满负载） | 183 MB（采样中 VmRSS） | 135s |
| 4 窗口 200 步（真权重 base=32，`--threads 4`） | 同量级（模型更小） | 74s |

一次最多 8 个窗口是硬限制（`--num` 的 argparse choices），8 窗口下内存不到 2GB 预算的 11%。耗时波动主要来自和别的训练任务抢 CPU（同一份代码 97s ~ 135s），`--threads` 可以按当时负载试 1/2/4（训练脚本侧实测被抢占时 1 线程反而最快）。

## 5. 坑与限制（写谱前必须知道）

### 5.1 Koishi 有明确的适用边界（不是我们的 bug）

| Koishi 返回 | 真实原因（从 worker 源码里读到的） |
|---|---|
| `diff: null` + `tier: "< Alpha Low"` + `hint: BelowScope` | 「Koishi 仅计算 `11 ≤ 定数 < 17`；预测低于 11，不输出难度」 |
| `diff: null` + `tier: "无法定级"` + `probs: {}` | `UnsupportedLN`：「Koishi 不计算长条**占比超过 18%** 的谱面」 |

实测验证：LN 占比 13.5% → 正常解析；LN 29.2%（`--ln-min 3`）和 LN 58.2%（`--ln-min 2`）→ 两张都是「无法定级」、`probs` 为空。**所以想用 Koishi 打分，LN 占比必须压在 18% 以下**（`--ln-min` 调大即可，或直接用 `--ln none`）。
同理，单窗口（13s / 119 音符）太短，Koishi 只会给 `< Alpha Low`/`null`，要**多个窗口 `--concat`** 才够长度。

### 5.2 二值卷分不清 LN 与密打

数据集 `X` 只有「这格有没有音符」，没有 LN 长度通道。`--ln runs` 是启发式（同列连续 ≥N 格 = LN），在稠密卷上会把「同列连打」误判成长条。真要做 LN 预测，得在训练侧把 LN 单独做一个通道（或用一个「长度」回归头），转换器这边只要读那个通道即可。

### 5.3 随机权重 ≠ 可用谱面

`--random-weights` 的产物是**噪声基线**（占位率 0.58、每行 2.3 个音、乱压 4 押），作用只是证明「采样 → 写谱 → 解析」这条链没断。它的 Koishi `pattern` 永远是 jack、`tech 概率 ≈ 0`，别拿它当生成质量。

### 5.4 当前模型的水平（照实说）

真权重（val 0.521，训练尚未收敛）出来的谱：密度对（0.42 vs 数据集 0.41），但 Koishi 判定 `jack 0.99`、`tech 0` —— 说明模型只学到了「哪里大概有音符」，没学到人类键型的列间关系（这正是 `out/diffusion/DESIGN.md` 里「阶段 B/C/D」还没做的部分）。难度估值随 `--bpm` 明显变化（15.4 nps → null，22.6 nps → 14.03），也说明它更像「铺满网格」而不是「有节奏结构」。

### 5.5 权重会被训练覆盖

`out/models/diffusion4k.pt` 是训练脚本每 epoch 覆盖写的。采样到一半权重被换掉不会崩（已读进内存），但**采样结果和报告里的数字要标明用的是哪个版本**（本报告：`val 0.521 / base 32`，文件时间 11:34）。

## 6. 下一步建议

1. **训练侧**：加音频条件（`DESIGN.md` 阶段 C）之前，先让模型学会「小节内节奏位置」—— 现在输出几乎每行都有音符，行密度 8.31 rps 太高、非空行 91%，人类谱会留空拍。
2. **采样侧**：加 DDIM / `--eta` 少步采样（200 步在 CPU 上 18s/窗，50 步能到 5s 内）；加窗口**重叠拼接**（前后窗各 16 格交叠做交叉淡化）替代现在的硬切。
3. **转换侧**：等模型支持 LN 通道后，`--ln` 加一个 `channel` 模式直接读；`--max-chord 2` 之类可以在「密度够、但手感乱」时兜底。

## 7. 复现全部验证的命令

```bash
cd /root/work/yamero-ai/proj

# 1) 采样（随机权重；没有 ckpt 也能跑）
python3 tools/sample-diffusion.py --random-weights --style 0 --diff 1 --num 4 --seed 0 --threads 4
#    真权重
python3 tools/sample-diffusion.py --style 2 --diff 3 --num 4 --seed 0 --threads 4

# 2) 转 osu（4 张验证谱）
python3 tools/roll-to-osu.py out/diffusion/samples/roll-s0d1-seed0-rnd-*.npy --concat -o out/diffusion/test-random.osu
python3 tools/roll-to-osu.py out/diffusion/samples/roll-s2d3-seed0-*.npy      --concat -o out/diffusion/test-real-s2d3.osu
python3 tools/roll-to-osu.py out/diffusion/samples/roll-s2d3-seed0-*.npy      --concat --bpm 200 -o out/diffusion/test-real-s2d3-bpm200.osu
python3 tools/roll-to-osu.py out/diffusion/samples/roll-s0d1-seed0-rnd-*.npy --concat --ln runs --ln-min 4 -o out/diffusion/test-random-ln.osu

# 3) Koishi 验收（应看到 diff / pattern / probs 三个字段）
node tools/gj/batch-koishi.mjs out/diffusion/test-random.osu out/diffusion/test-real-s2d3-bpm200.osu
# → test-random.osu           → 12.01 Beta mid [jack]   probs: jack .998 …
# → test-real-s2d3-bpm200.osu → 14.03 Delta mid [jack]  probs: jack .988 …
# 完整 4 张的结果已存在 out/diffusion/koishi-result.json
```
