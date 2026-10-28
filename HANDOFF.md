# 4K 谱面生成 AI —— 完整交接文档（HANDOFF）

> 交接日期：2026-10-29 02:30（GMT+8）
> 交接目标：把本项目全部训练成果、数据、代码、经验交给下一个 AI 继续训练
> 阅读顺序：本文档 → RUNBOOK.md → TRAINING-PLAN.md → MUG-DIFFUSION-剖析.md → FEATURES.md

---

## 0. 一句话现状

以 Mug-Diffusion（github.com/cv277/Mug-Diffusion）为蓝本的自研实现：**音频条件扩散模型 + KL-VAE 谱面压缩 + 21 token 键型控制**，数据集 36837 张 4K 谱面；同时并行维护一条**规则管线**（音频骨架→风格指纹着色→站点闭环锚定）作为交付线，已产出 10 轮 INTERNET YAMERO 谱面。

## 1. 包清单（5 个 tar.gz）

| 包 | 内容 | 体积 |
|---|---|---|
| 01-code-docs-models | 代码 + 文档 + 模型权重 + 指纹库 + 参考谱 + 日志 | ~150M |
| 02-site-analyzer | 站点 Koishi 分析器本地复现（难度/键型判定校验） | ~35M |
| 03-dataset-charts | 36837 张 .osu 谱面 + 36598 张 21 维控制特征 | ~3G |
| 04-dataset-tensors-audio | 36837 张 Mug 张量 + 9500+ 张音频 mel | ~2.9G |
| 05-audio-pool | 全部谱面集原始音频（osz 解压池 /tmp/osz_x） | ~4.3G |

解压后目录结构：
```
06-chart-ai/
├── HANDOFF.md RUNBOOK.md TRAINING-PLAN.md FEATURES.md MUG-DIFFUSION-剖析.md
├── README.md  notes.json  generator.html  preview.html  libminacalc.so
├── src/               # 规则管线核心 JS（chart-patterns.js 等）
├── tools/             # 全部 Python 工具（见 §3）
└── out/
    ├── charts/        # 36837 张 .osu（训练集谱面）
    ├── features/      # 21 维控制特征（chart/ + audio/ + index.jsonl）
    ├── mug_tensors/   # [16,4096] 谱面张量 npz
    ├── audio_features/# mel [128,4096] npz + _index.jsonl
    ├── vae/           # vae.pt（VAE 权重）、vae-kl1e6.pt
    ├── diffusion/     # diff.pt（扩散权重）
    ├── style-fingerprints.json   # 36 桶键型指纹
    ├── dataset-final.json / dataset.jsonl / msd*.jsonl / nps-msd.json
    ├── final*/ sweep6-s50/       # 历轮参考谱产物
    └── *.log          # vae-train / diffusion-train / audio-extract / pipeline
site-analysis/         # 站点分析器（node，见 §4）
```

## 2. 项目背景与硬约束

- 目标曲：KOTOKO - INTERNET YAMERO，**185 BPM，offset 1263.28ms**，242s
- 验收器是某站点的 **Koishi/Azusa 分析器**（Etterna 系）：D 值（Alpha→Zeta）+ pattern 判定（speed/tech/jack/ln）
- 用户核心诉求：**"看着音乐写，写出来符合人类，我要什么键型就要什么"**——键型可控、对音准确、难度可锚定
- 4K 模式（Mode 3，4 列），列坐标 x=64/192/320/448

## 3. 代码地图（tools/ 全部脚本）

### 数据构建链
| 脚本 | 作用 | 要点 |
|---|---|---|
| fetch-mug-list.mjs | 从 ppy 官方 + osu.direct 双源抓谱面集 | 8 线程、AbortSignal.timeout(20000) |
| build-feature-db.py | 21 维控制特征（ett×7/rc/hb/ln/pattern bool×7） | minacalc 传秒非毫秒；返回值是技能维数 |
| build-mug-tensors.py | 谱面 → [16,4096] 张量（头/偏移/LN体/尾偏移 × 4 列） | BPM 钳 150-300；负时间戳防护 |
| build-audio-features.py | 音频 → mel [128,4096] @46.44ms 帧对齐 | **--workers 1 串行防死锁**；已加音频索引 O(1) 查找 |

### 训练链（Mug-Diffusion 等价物）
| 脚本 | 阶段 | 当前状态 |
|---|---|---|
| train_vae.py | KL-VAE：[16,4096]→[16,512]（时间 8× 压缩） | **start F1 0.975 / holding F1 0.921**，续训中（断点续训：重跑同命令自动载入 out/vae/vae.pt） |
| train_diffusion.py | 音频 mel 通道拼接 + 21 token cross-attention + CFG 3 + DDIM 50 | **epoch0 loss 0.1083（9166 音频对，rate 增强 0.9-1.1 等效 ~1.3 万对）**；断点续训自动载入 out/diffusion/diff.pt |

训练参数：`--kl 0 --lr 1.2e-4`（VAE 已用 BCE+pos_weight=64）；lr 阶梯衰减每 4 epoch ×0.7；torch.set_num_threads(4)（4 核沙箱）。

### 规则管线（交付线，10 轮迭代）
| 脚本 | 环节 |
|---|---|
| skeleton-from-audio.py | 段落能量配额选格 → 节奏骨架（100% 踩 onset，信噪比 1.25×，检出 49 音乐段/5 组重复 → structure.json 供段落记忆） |
| build-style-fingerprints.py | 从 36837 张谱面提 36 桶指纹（cls=rice/ln × arch=stream/jumpstream/technical/chordjack/handstream/jackspeed × diff=low/mid/high）：列转移、行大小、行间隔、LN 分布、手交替、4/6-gram 图案块 |
| pattern-painter.py | 指纹着色：图案块接龙 + 段落记忆 + 同列惩罚 + 1/32 半格细分补 nps + --dps 显式行频 + 段落钳制 |
| anchor-difficulty.py | **站点闭环锚定**：内部调 painter --dps 迭代 + run-koishi.js 实测 D，割线法收敛（±0.35） |
| gen-ai-chart.py | 风格系统（rice/ln × 8 子键型 token）+ LN 三层钳制（<60ms 转 tap / >1 拍截断 / 4s 窗墙 >55% 压制） |

## 4. 站点校验体系（site-analysis/）

```sh
cd site-analysis && node run-koishi.js <osu路径>
# stdout 末行 JSON：numericDifficulty / estDiff / pattern / noteCount / enduranceScore / anchorScore
```
- run-koishi.js 特征 dump 注入版，另存 report-*.osu.json（含 patternDetails.probabilities 完整分布）
- **刻度经验（三代 v7/v9/v10 实测）**：
  - `pattern=tech` 需要 **D≥15 且 tech 概率>speed**（中低难度无论如何判 speed）
  - stream 型谱面 D 响应斜率 ~0.4 D/dps，tech 型 ~1.0-1.3 D/dps（锚定时割线法初值要分开）
  - 低于站点量程下限（D<12）返回 `numericDifficulty: null, hint: "BelowScope"`
  - 行密度有断层：dps 低时 1/32 半格细分不触发，行频断崖
- minacalc（libminacalc.so）与站点 D **非线性**，不要用 minacalc 结果外推站点 D

## 5. 规则管线完整出谱命令（v10 实测可用）

```sh
cd 06-chart-ai
# 1) 骨架（对音 100%）
python3 tools/skeleton-from-audio.py --analysis out/final14-s50/analysis.json -o /tmp/sk.osu
# 产出同时写 /tmp/sk.structure.json（音乐段+重复组）

# 2) 着色（键型 × 难度旋钮）
python3 tools/pattern-painter.py --skeleton /tmp/sk.osu --structure /tmp/sk.structure.json \
  --style rice/stream/high --dps 10.3 --seed 11 --version "vX Stream" -o out.osu

# 3) 站点锚定（推荐直接跑，内部迭代 painter+koishi）
python3 tools/anchor-difficulty.py --skeleton /tmp/sk.osu --structure /tmp/sk.structure.json \
  --style rice/technical/high --target 15.8 --seed 22 --max-iters 6 --version "vX Tech" -o final.osu
# 注意：--skeleton 必须传原始骨架！传着色成品会导致二次着色、dps 失效（v10 踩过）

# 4) 打包
zip "INTERNET YAMERO [vX].osz" "INTERNET YAMERO.ogg" KOTOKO*.osu
```

**v10 终版四档实测**（锚定目标 → 站点实测）：
| 版本 | style 桶 | dps | 站点 D | pattern | jack(<70ms) | LN |
|---|---|---|---|---|---|---|
| Easy D12.5 | rice/technical/mid | 7.25 | 12.64 | speed | 3.7% | 1.7% |
| Stream D14.8 | rice/stream/high | 10.3 | 14.87 | speed | 12.9% | 0.5% |
| Tech D15.8 | rice/technical/high | ~9.4 | 15.64 | **tech** | 14.1% | — |
| LN-Tech D15.7 | ln/technical/high | 10.1 | 15.60 | **tech** | 14.2% | 12.7% |

四张 1/16 落网率全部 100%；LN 版 4s 窗 LN 峰值 18（4.5/s，无墙）。

## 6. 训练坑全记录（重要，别重踩）

1. **极端稀疏目标 + L1 损失 = 全零局部最优**：VAE 初版 loss 收敛到 0.011（≈音符密度），F1 卡 0.18。改 **BCE + pos_weight=64** 后 F1 直线上升。
2. **torch 线程超配**：4 核开 16 线程自旋互踩，慢 7-50 倍。`torch.set_num_threads(4)`，0.25s/步。
3. **pkill 自匹配自杀**：`pkill -f "xxx"` 的命令行里含 xxx 会杀掉自己的 shell。用正则点号 `pkill -f "build.audio[.]features"`。
4. **LN 位修正**：砍长条后 hitSound/type 位 128 未清会被站点拒绝。
5. **双重放大**：指纹 nps 当行频用又叠和弦 → 行频 = nps÷和弦率。
6. **半格补偿全曲均摊致 D 爆表** → 段落钳制（每段上限，structure.json 的 seg_cells）。
7. **扩散骨架对音差**：扩散只训 ~2000 步时输出是"训练集平均节奏+噪声"，吸附网格也不对音 → 规则管线接管交付，扩散线继续训练（当前 loss 0.108 仍未到接管标准）。
8. **音频提取器假死**：每任务全池 os.walk（2.6 万文件）+ 4 核被训练占满 → 启动建一次索引 O(1) 查找（已修）。ProcessPoolExecutor 死锁时 `--workers 1` 串行。
9. **offset 显式传参被静默忽略**：v13 前全部偏 51ms。offset 必须 trace 到执行层。
10. **for→while 改写漏 i+=1 死循环**；指纹库被 `head -3` 管道截断。
11. **minacalc 秒/毫秒**：传毫秒结果全错；返回值是技能维数不是错误码。
12. **zsh 不做无引号分词**：`set -- $cfg` 整串进 $1，循环传参全空。

## 7. 数据格式速查

- **Mug 张量** [16,4096]：4 列 × {头, 头偏移, LN 体, 尾偏移}，帧率 46.44ms（21.53fps），时间 8× 压缩相对 5.805ms 音频帧
- **21 控制 token**：ett×7（one-hot 5-35）+ rc/hb/ln 三类 + pattern bool×7（stream/jumpstream/technical/chordjack/stamina/jackspeed/ln_ratio）
- **mel** [128,4096]：sr 22050 / n_fft 512 / hop 128 / n_mels 128，log1p，float16 存储，8× 平均池化到谱面帧
- **指纹桶**（style-fingerprints.json）：`{cls}/{arch}/{diff}` 36 桶，含 nps、和弦率、手交替、LN 率、列转移矩阵、4/6-gram patterns
- **.osu 4K 列映射**：x=64/192/320/448 → 列 0-3；LN type 位 128，尾部在 `5:flags` 前的 endtime

## 8. 当前状态快照（交接时刻）

| 线 | 进度 | 下一步 |
|---|---|---|
| VAE | F1 0.975/0.921，it~960 续训中 | 爬向 0.98+（验收 0.99；holding recall 已 100% 可用） |
| 扩散 | epoch0 loss 0.1083（9166 对） | loss 稳定 <0.10 后跑 v10 踩音率对比：用扩散 ckpt 采样骨架，基线 70.4%，目标 85%+ 接管交付线 |
| 音频对 | 9498/36837 提取中 | 向万级推进，渐饱和 |
| 规则管线 | v10 交付（D12.6-15.6 四档） | 可选：难度桶间插值（任意 D 精确锚）、chordjack/jackspeed 桶出谱验证 |

## 9. 给下一个 AI 的建议

1. 先跑通 §5 的规则管线（15 分钟见效），再碰扩散线
2. 站点校验是唯一真理（§4 的刻度经验直接用），minacalc 只做参考
3. 扩散线想接管交付必须过"踩音率 85%"关，否则只当风格先验用
4. 所有续训都是断点续训：重跑同命令即可，权重路径 out/vae/vae.pt、out/diffusion/diff.pt
5. 用户对谱面的敏感点排序：**对音 > 键型手感（jack/LN 墙）> 难度分配 > 排列新颖度**
