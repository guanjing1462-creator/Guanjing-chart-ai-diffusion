/*!
 * chart-tokenizer.js — 谱面 ↔ token 序列（双向可逆）
 * 版本 1.0.0 · 零依赖 · 浏览器与 Node 通用
 *
 * 表示方式（业界常见做法）：
 *   把时间按 1/16 拍量化成固定步长，每一步用一个整数表示"这一帧按了哪几列"，
 *   第 c 列对应 bit c。4K 的词汇表就是 0..15 共 16 个 token，7K 是 0..127。
 *   空帧为 0。这样谱面就变成一条普通的整数序列，可以直接喂给语言模型。
 *
 *   step = 60000 / bpm / 4    （1/16 拍）
 *   180 BPM 时一步 ≈ 83ms，3 分钟的谱约 2100 帧。
 *
 * 为什么不用"事件序列"（dt + mask）：
 *   固定步长的序列长度可预测、可以整批 padding，训练代码最简单；
 *   代价是空闲段会填很多 0（可以用 run-length 压缩缓解）。
 *
 * 限制：
 *   - keys ≤ 8（4K/7K 够用）。9K 以上 mask 词汇表会涨到 512+，需要换编码方式。
 *   - hold 只记录头，长度信息存在 holds 数组里，不参与 token 序列。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ChartTokenizer = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  var SUB_DIV = 4;   // 每拍切 4 份 → 1/16 拍

  function stepMsOf(bpm) {
    return 60000 / (bpm || 160) / SUB_DIV;
  }

  /* notes: [{t, col, end?}] -> {tokens, stepMs, frames, keys, bpm, holds} */
  function tokenize(notes, opts) {
    opts = opts || {};
    var keys = Math.max(1, Math.min(8, opts.keys || 4));
    var bpm = opts.bpm || 160;
    var stepMs = stepMsOf(bpm);
    var maxBars = opts.maxFrames || 20000;   // 安全上限，约 28 分钟

    if (!notes || !notes.length) {
      return { tokens: [], stepMs: stepMs, frames: 0, keys: keys, bpm: bpm, holds: [] };
    }

    var lastT = 0;
    for (var i = 0; i < notes.length; i++) {
      var e = notes[i].end || notes[i].t;
      if (e > lastT) lastT = e;
    }
    var frames = Math.min(maxBars, Math.ceil(lastT / stepMs) + 1);
    var tokens = new Array(frames);
    for (var f = 0; f < frames; f++) tokens[f] = 0;

    var holds = [];
    for (var n = 0; n < notes.length; n++) {
      var note = notes[n];
      var col = Math.max(0, Math.min(keys - 1, note.col | 0));
      var idx = Math.round(note.t / stepMs);
      if (idx >= 0 && idx < frames) tokens[idx] |= (1 << col);
      if (note.end && note.end > note.t) {
        holds.push({
          at: idx,
          col: col,
          // 长度也用"帧"表示，和 token 同一套坐标系
          len: Math.max(1, Math.round((note.end - note.t) / stepMs))
        });
      }
    }
    return { tokens: tokens, stepMs: stepMs, frames: frames, keys: keys, bpm: bpm, holds: holds };
  }

  /* tokens -> notes（用于把模型输出还原成可渲染的谱面） */
  function detokenize(seq, opts) {
    opts = opts || {};
    var keys = Math.max(1, Math.min(8, opts.keys || 4));
    var stepMs = opts.stepMs || stepMsOf(opts.bpm || 160);
    var notes = [];

    for (var i = 0; i < seq.length; i++) {
      var mask = seq[i] | 0;
      if (!mask) continue;
      for (var c = 0; c < keys; c++) {
        if (mask & (1 << c)) notes.push({ t: Math.round(i * stepMs), col: c });
      }
    }

    // 把 hold 挂回去（如果生成时保留了 holds）
    if (opts.holds && opts.holds.length) {
      var index = {};
      for (var k = 0; k < notes.length; k++) index[notes[k].t + ':' + notes[k].col] = notes[k];
      for (var h = 0; h < opts.holds.length; h++) {
        var ho = opts.holds[h];
        var t = Math.round(ho.at * stepMs);
        var target = index[t + ':' + ho.col];
        if (target) target.end = t + Math.round(ho.len * stepMs);
      }
    }
    return notes;
  }

  /* ------------------------------------------------------------------ */
  /* 统计工具：不训练也能给出可用的分布画像                                */
  /* ------------------------------------------------------------------ */
  function profile(seq, keys) {
    var total = 0, frames = seq.length, counts = {};
    var chord = { 1: 0, 2: 0, 3: 0, 4: 0 };
    var lanes = new Array(keys).fill(0);

    for (var i = 0; i < frames; i++) {
      var m = seq[i] | 0;
      if (!m) continue;
      var bits = 0;
      for (var c = 0; c < keys; c++) if (m & (1 << c)) { bits++; lanes[c]++; }
      total += bits;
      if (bits >= 4) chord[4]++; else chord[bits]++;
      counts[m] = (counts[m] || 0) + 1;
    }

    var density = new Array(Math.ceil(frames / 16)).fill(0);
    for (var j = 0; j < frames; j++) {
      var mm = seq[j] | 0;
      if (!mm) continue;
      var d = Math.floor(j / 16);
      for (var c2 = 0; c2 < keys; c2++) if (mm & (1 << c2)) density[d]++;
    }

    var top = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; }).slice(0, 8);
    return {
      notes: total,
      frames: frames,
      occupancy: frames ? total / frames / keys : 0,
      chord: chord,
      lanes: lanes,
      density: density,
      topPatterns: top.map(function (k) { return { mask: +k, n: counts[k] }; })
    };
  }

  /* ------------------------------------------------------------------ */
  /* 基线模型：二元转移 + 密度约束                                        */
  /* ------------------------------------------------------------------ */
  /* 这不是"AI"，只是一个统计基线。作用有两个：
     1) 验证 token 表示是否可学（能生成结构相似的序列说明表示没坏）
     2) 给真模型提供下界——真模型必须显著超过它才值得训 */
  function createBigram(keys) {
    var V = 1 << keys;
    var trans = [];
    for (var i = 0; i < V; i++) trans.push(new Array(V).fill(0));
    var start = new Array(V).fill(0);

    return {
      learn: function (seq) {
        if (!seq.length) return;
        start[seq[0]]++;
        for (var i = 1; i < seq.length; i++) trans[seq[i - 1]][seq[i]]++;
      },
      sample: function (len, opts) {
        opts = opts || {};
        var rnd = opts.rng || Math.random;
        var out = new Array(len);
        // 起点按出现频率抽样
        var cur = weightedPick(start, rnd);
        out[0] = cur;
        for (var i = 1; i < len; i++) {
          var row = trans[cur];
          var sum = 0;
          for (var j = 0; j < V; j++) sum += row[j];
          if (!sum) { cur = 0; }
          else {
            var r = rnd() * sum, acc = 0;
            cur = 0;
            for (var k = 0; k < V; k++) { acc += row[k]; if (acc >= r) { cur = k; break; } }
          }
          out[i] = cur;
        }
        return out;
      }
    };
  }

  function weightedPick(arr, rnd) {
    var sum = 0, i;
    for (i = 0; i < arr.length; i++) sum += arr[i];
    if (!sum) return 0;
    var r = rnd() * sum, acc = 0;
    for (i = 0; i < arr.length; i++) { acc += arr[i]; if (acc >= r) return i; }
    return 0;
  }

  /* 可复现随机数（mulberry32），方便对比不同 seed 的生成结果 */
  function makeRng(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  return {
    SUB_DIV: SUB_DIV,
    stepMsOf: stepMsOf,
    tokenize: tokenize,
    detokenize: detokenize,
    profile: profile,
    createBigram: createBigram,
    makeRng: makeRng,
    version: '1.0.0'
  };
});
