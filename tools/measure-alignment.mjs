#!/usr/bin/env node
/**
 * measure-alignment.mjs — 测「人类谱面的采音对齐度」基线
 *
 * 用法：
 *   node tools/measure-alignment.mjs <目录>        # 目录下每个子目录含 .osu + 音频
 *
 * 指标定义（与 generator.html 里的完全一致，可直接对比）：
 *   对齐度 = 音符所在帧的平均音频能量 ÷ 全曲平均能量
 *   1.0 = 和随机铺格子一样；越高说明音符越贴着有声的地方
 *
 * 为什么用这个定义：
 *   「落在前 N% 强拍的比例」在高密度下会被钉死，看不出好坏；
 *   用均值比值可以直接和随机基线（=1.0）比较。
 */
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const BP = require('../src/beatmap-parser.js');

const dir = process.argv[2];
if (!dir) { console.error('用法: node measure-alignment.mjs <目录>'); process.exit(1); }

const HOP_MS = 10;
const SR = 8000;   // 降采样到 8k 足够算能量包络

/* ---------- wav 读取（16-bit PCM 单声道） ---------- */
function readWav(file) {
  const buf = fs.readFileSync(file);
  if (buf.toString('ascii', 0, 4) !== 'RIFF') throw new Error('not wav');
  let off = 12, dataOff = -1, dataLen = 0;
  while (off < buf.length - 8) {
    const id = buf.toString('ascii', off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    if (id === 'data') { dataOff = off + 8; dataLen = size; break; }
    off += 8 + size + (size % 2);
  }
  if (dataOff < 0) throw new Error('no data chunk');
  const n = Math.floor(dataLen / 2);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = buf.readInt16LE(dataOff + i * 2) / 32768;
  return out;
}

/* ---------- 能量包络（每 HOP_MS 一帧 RMS） ---------- */
function envelope(samples, sampleRate) {
  const hop = Math.round(sampleRate * HOP_MS / 1000);
  const frames = Math.floor(samples.length / hop);
  const env = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    let s = 0;
    const start = f * hop;
    for (let i = 0; i < hop; i++) { const v = samples[start + i] || 0; s += v * v; }
    env[f] = Math.sqrt(s / hop);
  }
  return env;
}

/* 只保留上升沿，抑制衰减（更接近 onset 强度） */
function fluxOf(env) {
  const out = new Float32Array(env.length);
  for (let i = 1; i < env.length; i++) {
    const d = env[i] - env[i - 1];
    out[i] = d > 0 ? d : 0;
  }
  return out;
}

function mean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return a.length ? s / a.length : 0;
}

/* ---------- 主流程 ----------
 * 两个必须做对的地方：
 *
 * 1) 每个 .osu 配它自己的音频。谱包里有 13% 是「倍速图」或「合集包」，
 *    一个 set 里散着 audio.mp3 / audio 1.1x.mp3 / 1.mp3 / decoy.mp3 …
 *    拿第一个音频套全部难度，音符时间轴和音频直接错位一个倍速，
 *    测出来的对齐度是垃圾（我在这上面栽过一次，数值虚高到 ×1.55）。
 *
 * 2) 同一份音频只解码一次。合集包里 15 张谱可能共用 1 个音频，
 *    也可能各用各的 —— 按文件名缓存，两种情况都不会重复跑 ffmpeg。
 */
/* key 必须带目录名：几百个 set 里有一大半的音频都叫 audio.mp3，
   只用文件名做 key 会让第二个 set 起全部命中第一个 set 的缓存，
   测出来的结果全是错的（我在这上面栽过一次，整批 ×1.39 掉到 ×1.07）。 */
const envCache = new Map();

function envelopeFor(subDir, audioFile) {
  const key = path.basename(subDir) + '/' + audioFile;
  if (envCache.has(key)) return envCache.get(key);
  const wavOut = path.join(subDir, `_analysis_${Math.abs(hashStr(audioFile))}.wav`);
  try {
    execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', path.join(subDir, audioFile),
      '-ac', '1', '-ar', String(SR), '-f', 'wav', wavOut], { stdio: 'ignore' });
  } catch (e) {
    envCache.set(key, null);
    return null;
  }
  const pcm = readWav(wavOut);
  fs.rmSync(wavOut, { force: true });
  const env = envelope(pcm, SR);
  const flux = fluxOf(env);
  const avgEnv = mean(env) || 1e-9;
  const avgFlux = mean(flux) || 1e-9;
  const same = [];
  for (let k = 0; k < 3000; k++) same.push(env[Math.floor(Math.random() * env.length)]);
  const rec = { env, flux, avgEnv, avgFlux, randRatio: (mean(same) || 1e-9) / avgEnv };
  envCache.set(key, rec);
  return rec;
}

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  return h;
}

function audioOf(text, files) {
  const name = (text.match(/^AudioFilename:\s*(.+)$/m) || [])[1] || '';
  const base = path.basename(name.trim());
  if (files.includes(base)) return base;
  const low = base.toLowerCase();
  const hit = files.find((f) => f.toLowerCase() === low);
  if (hit) return hit;
  const cands = files.filter((f) => /\.(mp3|ogg|wav|m4a)$/i.test(f));
  return cands.length === 1 ? cands[0] : null;
}

const rows = [];
for (const name of fs.readdirSync(dir)) {
  const sub = path.join(dir, name);
  if (!fs.statSync(sub).isDirectory()) continue;
  const files = fs.readdirSync(sub);
  const osuFiles = files.filter((f) => f.toLowerCase().endsWith('.osu'));
  if (!osuFiles.length) { console.log(`跳过 ${name}: 没有 .osu`); continue; }

  for (const osuFile of osuFiles) {
    let text;
    try { text = fs.readFileSync(path.join(sub, osuFile), 'utf8'); } catch (e) { continue; }
    const audioFile = audioOf(text, files);
    if (!audioFile) continue;

    const sig = envelopeFor(sub, audioFile);
    if (!sig) continue;

    let parsed;
    try { parsed = BP.parse(text); } catch (e) { continue; }
    if (!parsed.notes.length || parsed.notes.length < 30) continue;

    /* 用 RMS 而不是 flux 当强度：flux 只捕捉「能量上升」，
       落在长音/旋律/人声上的音符会被判成没采到音，
       算出来人类谱面反而不如随机 —— 那是指标错了，不是谱面错了。 */
    const ratioAt = (source, avg, offsetMs) => {
      let sum = 0, cnt = 0;
      for (const n of parsed.notes) {
        const idx = Math.floor((n.t + offsetMs) / HOP_MS);
        if (idx >= 0 && idx < source.length) { sum += source[idx]; cnt++; }
      }
      return cnt ? (sum / cnt) / avg : 0;
    };

    let best = { off: 0, r: -1 };
    for (let off = -150; off <= 150; off += 10) {
      const r = ratioAt(sig.env, sig.avgEnv, off);
      if (r > best.r) best = { off, r };
    }

    rows.push({
      name: name + '/' + osuFile.replace(/\.osu$/i, '').slice(-30),
      setId: name, keys: parsed.keys, notes: parsed.objectCount,
      bpm: Math.round(parsed.bpm || 0), aligned: best.r,
      alignedFlux: ratioAt(sig.flux, sig.avgFlux, best.off),
      offset: best.off, random: sig.randRatio,
      audioFile,
      dur: Math.round(sig.env.length * HOP_MS / 1000),
    });
    console.log(`${String(parsed.keys) + 'K'} ${String(parsed.objectCount).padStart(5)}n ` +
      `${String(Math.round(parsed.bpm || 0)).padStart(3)}bpm ` +
      `对齐 ×${best.r.toFixed(2)}  off ${String(best.off).padStart(4)}ms  ` +
      `rand ×${sig.randRatio.toFixed(2)}   ${name}/${osuFile.slice(0, 26)}`);
  }
}

if (rows.length) {
  const avg = rows.reduce((a, r) => a + r.aligned, 0) / rows.length;
  const avgRand = rows.reduce((a, r) => a + r.random, 0) / rows.length;
  console.log(`\n=== 汇总 ===`);
  console.log(`样本 ${rows.length} 张`);
  console.log(`人类谱面平均采音对齐（RMS） ×${avg.toFixed(2)}`);
  console.log(`随机基线 ×${avgRand.toFixed(2)}`);
  const keys = {};
  rows.forEach((r) => { keys[r.keys] = (keys[r.keys] || 0) + 1; });
  console.log(`键数分布 ${JSON.stringify(keys)}`);
  console.log(`生成器现状：onset 驱动 ×1.45 / BPM 网格 ×1.39（同口径，337 秒真实音频实测）`);
  fs.writeFileSync(path.join(dir, 'alignment-baseline.json'), JSON.stringify({
    samples: rows.length, humanAvg: +avg.toFixed(3), randomAvg: +avgRand.toFixed(3), rows
  }, null, 1));
  console.log(`结果写入 ${path.join(dir, 'alignment-baseline.json')}`);
}
