#!/usr/bin/env node
/* 把任意音频解码成「BPM + onset 峰 + 细包络」，给谱面生成器用。
 *
 *   node tools/analyze-audio.mjs song.ogg -o analysis.json
 *
 * 浏览器里跑的是 audio-onsets.js 的 analyze()（靠 decodeAudioData）；
 * 命令行没有 AudioContext，所以这里用 ffmpeg 解成 22.05k 单声道 f32，
 * 再伪造一个 AudioBuffer 形状的对象喂给 analyzeBuffer()——两者走的是同一套
 * 包络/BPM/网格代码，网页版和 CLI 出的谱不会有两套结果。
 */
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const AO = require('../src/audio-onsets.js');

function parseArgs(argv) {
  const o = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const isFlag = a.startsWith('--') || (a.length === 2 && a.startsWith('-'));
    if (!isFlag) continue;
    const k = a.replace(/^--?/, '');
    const v = argv[i + 1];
    if (v && !v.startsWith('-')) { o[k] = v; i++; } else { o[k] = true; }
  }
  return o;
}

const args = parseArgs(process.argv.slice(2));
const audio = args._ || args.audio || process.argv[2];
if (!audio || audio.startsWith('--')) {
  console.error('用法: node tools/analyze-audio.mjs <音频> [-o analysis.json] [--sr 22050] [--delta 0.6] [--gap 25] [--bpm 185] [--offset 1263]');
  process.exit(1);
}
const out = args.o || path.join(path.dirname(audio), 'analysis.json');
const SR = parseInt(args.sr || '22050', 10);
// 22.05k 对 RMS 包络足够；用 44.1k 会让包络点数翻倍、JSON 大一倍而信息量不变

const t0 = Date.now();
const pcm = execFileSync('ffmpeg', [
  '-v', 'error', '-i', audio,
  '-ac', '1', '-ar', String(SR),
  '-f', 'f32le', '-acodec', 'pcm_f32le', '-',
], { maxBuffer: 1 << 29 });

const n = Math.floor(pcm.length / 4);
const data = new Float32Array(pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + n * 4));

// 伪造 AudioBuffer：analyzeBuffer 只用到这五个成员
const fake = {
  sampleRate: SR,
  length: n,
  numberOfChannels: 1,
  duration: n / SR,
  getChannelData: () => data,
};

/* BPM 和 offset 可以人工指定：
   自相关在本曲上把 185 测成 93.8（倍频错），网格一错小节线全歪。
   --bpm 给定后，stepMs 直接按 60000/bpm/4 算（1/16 拍），
   --offset 给定后不再做相位检测。 */
const an = AO.analyzeBuffer(fake, {
  stepMs: 80,
  fineHopMs: 11.6,
  bpm: args.bpm ? parseFloat(args.bpm) : undefined,
  offsetMs: args.offset != null ? parseFloat(args.offset) : undefined,
});
const peaks = AO.detectPeaks(an.fine.energy, an.fine.hopMs, {
  winMs: 200,
  delta: args.delta != null ? parseFloat(args.delta) : 0.6,
  minGapMs: args.gap != null ? parseFloat(args.gap) : 25,
});

const r4 = (x) => Math.round(x * 10000) / 10000;
const doc = {
  source: path.basename(audio),
  sampleRate: SR,
  durationSec: an.durationSec,
  bpm: an.bpm,
  bpmConfidence: an.bpmConfidence,
  offsetMs: an.offsetMs,
  stepMs: an.stepMs,
  steps: an.steps,
  peaks,
  accent: Array.from(an.accent, r4),
  fine: { hopMs: an.fine.hopMs, energy: Array.from(an.fine.energy, r4) },
  generatedAt: new Date().toISOString(),
};

fs.writeFileSync(out, JSON.stringify(doc));
const kb = (fs.statSync(out).size / 1024).toFixed(0);
console.log(JSON.stringify({
  out, kb, bpm: an.bpm, confidence: an.bpmConfidence,
  offsetMs: an.offsetMs, durationSec: an.durationSec,
  peaks: peaks.length, peakPerSec: +(peaks.length / an.durationSec).toFixed(2),
  ms: Date.now() - t0,
}));
