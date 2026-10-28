# 特征包格式

`tools/extract-features.py` 的产出。这是喂给训练脚本的最终输入，格式一旦定了就别轻易改。

## 目录

```
out/features/
├── audio/
│   └── {setId}__{audioStem}.npz     mel + onset + rms
├── chart/
│   └── {setId}__{n}.npz             tensor + valid + keys
├── index.jsonl                      每张谱一行，训练时读这个
└── skipped.jsonl                    跳过的原因（排查数据问题用）
```

`{audioStem}` 是音频文件名去掉扩展名后 sanitize 的结果。**为什么要带上音频名**：13% 的谱面是倍速图或合集包，一个 set 里有多个 mp3，不同难度的音频不一样。

## audio/{setId}__{stem}.npz

| 键 | shape | dtype | 说明 |
|---|---|---|---|
| `mel` | `[128, T]` | float16 | `log1p(melspectrogram)`，T 最多 8192 |
| `onset` | `[T]` | float16 | mel 频带能量总和的**上升沿**，半波整流后归一化到 [0,1] |
| `rms` | `[N]` | float32 | 全频段 RMS，10ms 一格（N ≈ 100×T） |
| `rmsHopMs` | scalar | float32 | 固定 10.0 |

**mel 参数**（严格对齐 Mug-Diffusion 的 `mug/util.py`）：

```
sr       = 22050
n_fft    = 2048
hop      = 1024          -> 46.4399 ms/帧
n_mels   = 128
f_min    = 0, f_max = None
norm     = 'slaney'      <- librosa 默认，torchaudio 默认是 None，必须显式指定
mel_scale= 'slaney'      <- librosa 默认 htk=False，torchaudio 默认是 htk
power    = 2.0
pad_mode = 'reflect'
压缩     = log1p，存 float16
```

`norm` 和 `mel_scale` 这两项如果漏掉，同一段音频算出来的 mel 会和 Mug-Diffusion 差一个常数标度 —— 模型能训，但没法复用它的权重，也没法和它的结果对比。

**T 的上限 8192** ≈ 380 秒。超过的裁掉，不足的由 dataloader 补零（这里不补，省空间）。

## chart/{setId}__{n}.npz

| 键 | shape | dtype | 说明 |
|---|---|---|---|
| `tensor` | `[4*keys, 8192]` | float16 | 4 通道编码，见下 |
| `valid` | `[8192]` | uint8 | 最后一个音符之前的帧为 1，之后为 0（算 loss 时 mask 掉） |
| `keys` | scalar | uint8 | 键数 |

**4 通道语义**（通道布局照 Mug-Diffusion 的 `objects_to_array`）：

```
ch  col                     : note start 标志（0/1）
ch  col + keys              : 该音符在帧内的相对位置，[0,1)
ch  col + 2*keys            : 是否处于 LN（长条）体内
ch  col + 3*keys            : LN 尾部的帧内相对位置，[0,1)
```

**解码成时间**：

```python
t_ms = (idx - 0.5 + tensor[col + keys, idx]) * 46.4399
```

那个 `-0.5` 是因为我们用 `round` 而不是 `floor` 定帧号，见下。

## 和 Mug-Diffusion 的唯一差异：`round` 而不是 `floor`

它的 `read_time` 是 `index = int(t / frame_ms)`（向下取整），offset 落在 `[0,1)`。

我们用 `index = round(t / frame_ms)`，offset 相应平移 0.5 也落在 `[0,1)`。

**为什么改**：STFT 第 i 帧的中心在 `i * 46.44ms`。一个落在 1000ms 的脉冲，
`1000/46.44 = 21.53`，最强响应出现在帧 22（中心 1021.7ms）。
`floor` 会编到帧 21（中心 975.3ms），和能量峰错开近一帧。

用 1430228 实测：floor 编码的音符帧平均能量是全局的 ×1.02，round 是 ×1.37。
信息量两者相同（offset 通道无损），但 round 让 `start` 通道直接压在能量峰上，
模型不用先学一个固定的半帧偏移。

**代价**：不能直接套它的 dataloader 的 `array_to_objects`。如果要互转，
记得补偿这 0.5 帧。

## index.jsonl

```json
{
  "setId": "1430228",
  "osu": "Mili - World.execute(me); (Beatmap) [world.unite(me, you);].osu",
  "audio": "audio/1430228__Audio.npz",
  "audioFile": "Audio.mp3",
  "chart": "chart/1430228__4.npz",
  "keys": 4,
  "notes": 2707,
  "holds": 121,
  "bpm": 130.0,
  "title": "World.execute(me);",
  "version": "world.unite(me, you);",
  "frames": 4574,
  "durSec": 212.4,
  "offsetMs": 0.0
}
```

`offsetMs` 恒为 0，保留字段是为了将来手工补偿个别谱面的主观偏移。

## 加载示例

```python
import json, numpy as np, torch

rows = [json.loads(l) for l in open('out/features/index.jsonl')]

def load_feature(row, max_frames=8192):
    with np.load('out/features/' + row['audio']) as z:
        mel = z['mel'].astype(np.float32)          # [128, T]
    with np.load('out/features/' + row['chart']) as z:
        chart = z['tensor'].astype(np.float32)     # [4K, 8192]
        valid = z['valid'].astype(np.float32)
    T = min(mel.shape[1], max_frames)
    mel_p = np.zeros((128, max_frames), dtype=np.float32)
    mel_p[:, :T] = mel[:, :T]
    return torch.from_numpy(mel_p), torch.from_numpy(chart), torch.from_numpy(valid)
```

## 已知坑（都踩过）

1. **每个 .osu 必须配它自己的 AudioFilename**。一个 set 里有多个 mp3 时
   （倍速图 `audio 1.1x.mp3`、合集包 `1.mp3` / `decoy.mp3`），
   拿第一个音频套全部难度，音符时间轴和音频错位一个倍速，数据直接废掉。
   影响面：13%（57/438）。

2. **缓存 key 必须含 set 目录**。几百个 set 里有一大半音频都叫 `audio.mp3`，
   只用文件名做缓存键会让第二个 set 起全部命中第一个的缓存。
   我在这上面把整批基线从 ×1.41 拉到 ×1.07。

3. **评估不能用 46ms 帧的能量**。2707 音符 / 4574 帧 = 59% 的帧有音符，
   取"音符帧的均值"必然接近全局均值，测出来永远是 ×1.0。
   评估要用 10ms 的 `rms` 或 mel 低频带，并且**必须扫偏移**。

4. **评估不能用 22.05kHz 全频段 RMS**。高频（镲片、齿音、噪声）会稀释鼓点，
   同一张谱 mel 低频带给 ×1.59，全频段 RMS 只给 ×1.06。
   measure-alignment.mjs 用 8kHz 降采样，正好把高频滤掉了，所以它给 ×1.34 是对的。

5. **`np.ascontiguousarray` 后 `torch.from_numpy` 会警告不可写**。
   无害，但如果后面要 in-place 操作记得 `.clone()`。
