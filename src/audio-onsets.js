/*!
 * audio-onsets.js — 音频分析：BPM 检测 + 每步强度（accent）
 * 版本 1.0.0 · 零依赖（Web Audio API）
 *
 * 生成谱面需要两样东西：
 *   1. BPM / offset —— 决定节拍网格
 *   2. 每 1/16 拍的强度 —— 决定「哪一拍值得放音符」
 *
 * 强度用「谱通量（spectral flux）」的简化版：能量包络的一阶差分累积。
 * 比起纯 RMS，它对鼓点更敏感，不容易把长音当成重拍。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.AudioOnsets = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  /* UMD 的 root 只在外层包装函数的作用域里，factory 内部访问不到。
     之前 decode() 直接写了 root.AudioContext，导致「上传真实 mp3」必崩
     （ReferenceError: root is not defined），而合成音频走 analyzeBuffer，
     从不经过这里，所以一直没被发现。 */
  var G = typeof globalThis !== 'undefined' ? globalThis
        : (typeof window !== 'undefined' ? window : {});

  /* ------------------------------------------------------------------ */
  /* 解码                                                                */
  /* ------------------------------------------------------------------ */
  function decode(arrayBuffer, ctx) {
    var AC = G.AudioContext || G.webkitAudioContext;
    var audioCtx = ctx || new AC();
    return audioCtx.decodeAudioData(arrayBuffer.slice(0)).then(function (buf) {
      return { buffer: buf, ctx: audioCtx };
    });
  }

  /* 混合成单声道 Float32 */
  function mono(buffer) {
    var n = buffer.length;
    var ch = buffer.numberOfChannels;
    if (ch === 1) return buffer.getChannelData(0);
    var out = new Float32Array(n);
    for (var c = 0; c < ch; c++) {
      var d = buffer.getChannelData(c);
      for (var i = 0; i < n; i++) out[i] += d[i] / ch;
    }
    return out;
  }

  /* ------------------------------------------------------------------ */
  /* 能量包络 + 谱通量                                                    */
  /* ------------------------------------------------------------------ */
  /**
   * @param {Float32Array} data 单声道采样
   * @param {number} sampleRate
   * @param {number} hopMs 每帧毫秒（通常 = stepMs）
   * @returns {{ flux: Float32Array, energy: Float32Array, hopMs }}
   */
  function envelope(data, sampleRate, hopMs) {
    var hop = Math.max(1, Math.round(sampleRate * hopMs / 1000));
    var frames = Math.floor(data.length / hop);
    var energy = new Float32Array(frames);
    var flux = new Float32Array(frames);

    // 每帧 RMS
    for (var f = 0; f < frames; f++) {
      var s = 0, start = f * hop;
      for (var i = 0; i < hop; i++) {
        var v = data[start + i] || 0;
        s += v * v;
      }
      energy[f] = Math.sqrt(s / hop);
    }

    // 谱通量近似：正向能量差（只取上升沿，抑制衰减）
    for (var g = 1; g < frames; g++) {
      var d = energy[g] - energy[g - 1];
      flux[g] = d > 0 ? d : 0;
    }

    return { flux: flux, energy: energy, hopMs: hopMs };
  }

  /* 归一化到 0..1（用 95 分位避免个别爆音拉满） */
  function normalize(arr) {
    var tmp = Array.prototype.slice.call(arr).sort(function (a, b) { return a - b; });
    var p95 = tmp[Math.floor(tmp.length * 0.95)] || 1;
    var out = new Float32Array(arr.length);
    for (var i = 0; i < arr.length; i++) {
      out[i] = Math.max(0, Math.min(1, arr[i] / (p95 || 1)));
    }
    return out;
  }

  /* ------------------------------------------------------------------ */
  /* BPM 检测：对谱通量做自相关                                          */
  /* ------------------------------------------------------------------ */
  function detectBPM(flux, hopMs, opt) {
    var o = opt || {};
    var minBpm = o.minBpm || 60;
    var maxBpm = o.maxBpm || 220;

    var framesPerSec = 1000 / hopMs;
    var lagMin = Math.max(2, Math.round(framesPerSec * 60 / maxBpm));
    var lagMax = Math.min(flux.length - 1, Math.round(framesPerSec * 60 / minBpm));
    if (lagMax <= lagMin) return { bpm: 0, confidence: 0 };

    var best = { lag: 0, score: -1 };
    var scores = [];
    for (var lag = lagMin; lag <= lagMax; lag++) {
      var s = 0;
      for (var i = 0; i + lag < flux.length; i++) s += flux[i] * flux[i + lag];
      scores.push({ lag: lag, score: s });
      if (s > best.score) best = { lag: lag, score: s };
    }

    // 倍频消歧：如果 2×lag 处也很强，说明真实拍子可能更慢
    var half = scores.find(function (x) { return x.lag === Math.round(best.lag * 2); });
    if (half && half.score > best.score * 0.75) best = half;

    var mean = scores.reduce(function (a, x) { return a + x.score; }, 0) / scores.length;
    var bpm = 60000 / (best.lag * hopMs);
    var lag = best.lag;

    /* 倍频消歧后必须把 BPM 规整回「人耳感知区间」。
       底鼓在 1、3 拍、军鼓在 2、4 拍时，2 拍周期（= BPM/2）的自相关往往最强，
       于是 128 BPM 的鼓点会被判成 64 —— 一半的谱面会因此步数减半、密度腰斩。
       这里用 90–180 作为感知区间，超出就折半/翻倍。 */
    while (bpm < 90 && lag / 2 >= lagMin) { bpm *= 2; lag /= 2; }
    while (bpm >= 180 && lag * 2 <= lagMax) { bpm /= 2; lag *= 2; }

    return {
      bpm: Math.round(bpm * 10) / 10,
      confidence: mean > 0 ? Math.min(1, Math.max(0, (best.score / mean - 1) / 4)) : 0,
      lag: Math.round(lag),
      rawBpm: Math.round((60000 / (best.lag * hopMs)) * 10) / 10
    };
  }

  /* 找第一个强 onset 当 offset（毫秒） */
  function detectOffset(flux, hopMs) {
    var norm = normalize(flux);
    var threshold = 0.5;
    // 取前 20% 时长里第一个超过阈值的峰
    var limit = Math.floor(norm.length * 0.2);
    for (var i = 1; i < limit; i++) {
      if (norm[i] > threshold && norm[i] >= norm[i - 1] && norm[i] >= norm[i + 1]) {
        return Math.round(i * hopMs);
      }
    }
    return 0;
  }

  /* ------------------------------------------------------------------ */
  /* 重采样到固定的 1/16 拍网格                                          */
  /* ------------------------------------------------------------------ */
  /**
   * 把任意 hop 的 flux 对齐到「每 stepMs 一格」的网格上，
   * 这样生成器可以直接按 index 取强度。
   */
  function toGrid(flux, hopMs, stepMs, totalSteps) {
    var out = new Float32Array(totalSteps);
    var ratio = stepMs / hopMs;
    for (var s = 0; s < totalSteps; s++) {
      var center = s * ratio;
      var lo = Math.max(0, Math.floor(center - ratio / 2));
      var hi = Math.min(flux.length - 1, Math.ceil(center + ratio / 2));
      var best = 0;
      for (var j = lo; j <= hi; j++) {
        if (flux[j] > best) best = flux[j];
      }
      out[s] = best;
    }
    return normalize(out);
  }

  function ceil(num) { return Math.ceil(num); }

  /* ------------------------------------------------------------------ */
  /* 主入口                                                              */
  /* ------------------------------------------------------------------ */
  /**
   * @param {ArrayBuffer} arrayBuffer 音频文件字节
   * @param {object} opt { stepMs, bpm(可选，给了就跳过检测) }
   * @returns Promise<{ bpm, offsetMs, stepMs, steps, accent, durationSec, confidence }>
   */
  function analyze(arrayBuffer, opt) {
    return decode(arrayBuffer).then(function (res) {
      return analyzeBuffer(res.buffer, opt);
    });
  }

  /* 直接分析一个 AudioBuffer（合成音频、或已经解码过的场景用这条） */
  function analyzeBuffer(buf, opt) {
    var o = opt || {};
    var stepMs = o.stepMs || 80;
    var data = mono(buf);
    var fineHop = o.fineHopMs || 11.6;
    var fineEnv = envelope(data, buf.sampleRate, fineHop);
      var durSec = buf.duration;

      var env = envelope(data, buf.sampleRate, Math.min(20, stepMs));
      var bpmInfo = o.bpm ? { bpm: o.bpm, confidence: 1 } : detectBPM(env.flux, env.hopMs);
      var bpm = bpmInfo.bpm || 150;

      // 用检测到的 BPM 重新算 step（保证 1/16 拍对齐）
      var realStepMs = 60000 / bpm / 4;
      var offsetMs = o.offsetMs != null ? o.offsetMs : detectOffset(env.flux, env.hopMs);
      var usable = Math.max(0, durSec * 1000 - offsetMs);
      var steps = Math.floor(usable / realStepMs);

      var accent = toGrid(env.flux, env.hopMs, realStepMs, steps);

      return {
        bpm: bpm,
        bpmConfidence: Math.round((bpmInfo.confidence || 0) * 100) / 100,
        offsetMs: offsetMs,
        stepMs: Math.round(realStepMs * 100) / 100,
        steps: steps,
        accent: accent,
        durationSec: Math.round(durSec * 10) / 10,
        sampleRate: buf.sampleRate,
        /* 高分辨率包络：给「音符时间吸附」用。
           音乐网格（1/16 拍）太粗，吸附不到真实的鼓点/重音上。
           Mug-Diffusion 之所以能用固定 46ms 网格还采得准，靠的是亚帧 offset 通道；
           这里用 11.6ms（512/44100）的细包络做同样的事。 */
        fine: {
          energy: fineEnv.energy,
          flux: fineEnv.flux,
          hopMs: fineEnv.hopMs
        }
      };
  }

  /* 峰值检测：自适应阈值 + 局部最大值。
     阈值 = 滑动均值 + k·标准差（librosa onset_detect 的经典做法），
     窗口取 ~200ms，能跟上快速的 stream 段而不会被单个重音带偏。 */
  function detectPeaks(energy, hopMs, opt) {
    var o = opt || {};
    var win = Math.max(3, Math.round((o.winMs || 200) / hopMs));
    var k = o.delta != null ? o.delta : 0.9;
    var minGap = Math.max(1, Math.round((o.minGapMs || 40) / hopMs));
    var n = energy.length;
    var peaks = [];
    var lastIdx = -1e9;

    for (var i = 1; i < n - 1; i++) {
      // 局部最大值
      if (energy[i] < energy[i - 1] || energy[i] < energy[i + 1]) continue;
      // 滑动窗口的均值与标准差
      var lo = Math.max(0, i - win), hi = Math.min(n, i + win + 1);
      var sum = 0, cnt = 0;
      for (var j = lo; j < hi; j++) { sum += energy[j]; cnt++; }
      var mean = cnt ? sum / cnt : 0;
      var varSum = 0;
      for (var j2 = lo; j2 < hi; j2++) { var d = energy[j2] - mean; varSum += d * d; }
      var sd = cnt > 1 ? Math.sqrt(varSum / cnt) : 0;
      if (energy[i] < mean + k * sd) continue;
      if (i - lastIdx < minGap) {
        // 太密：保留更强的那个
        if (peaks.length && energy[i] > energy[peaks[peaks.length - 1]]) {
          peaks[peaks.length - 1] = i;
          lastIdx = i;
        }
        continue;
      }
      peaks.push(i);
      lastIdx = i;
    }
    return peaks;
  }

  /* 单独算一条细包络（不改 BPM 网格的场景用） */
  function fineEnvelope(buf, hopMs) {
    return envelope(mono(buf), buf.sampleRate, hopMs || 11.6);
  }

  return {
    analyze: analyze,
    analyzeBuffer: analyzeBuffer,
    fineEnvelope: fineEnvelope,
    detectPeaks: detectPeaks,
    decode: decode,
    mono: mono,
    envelope: envelope,
    normalize: normalize,
    detectBPM: detectBPM,
    toGrid: toGrid,
    version: '1.0.0'
  };
});
