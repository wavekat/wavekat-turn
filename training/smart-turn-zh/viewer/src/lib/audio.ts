export interface LODLevel {
  min: Float32Array;
  max: Float32Array;
  bucketSize: number;
}

interface WavInfo {
  channels: number;
  sampleRate: number;
  bitsPerSample: number;
  dataOffset: number;
  frames: number;
}

export class AudioStore {
  info: WavInfo | null = null;
  raw: Int16Array | null = null;
  channelCount = 0;
  sampleRate = 16000;
  duration = 0;
  totalFrames = 0;

  private lodCache = new Map<number, LODLevel[]>();

  async load(file: File) {
    const buffer = await file.arrayBuffer();
    this.info = parseWavHeader(buffer);
    this.raw = new Int16Array(
      buffer,
      this.info.dataOffset,
      this.info.frames * this.info.channels,
    );
    this.channelCount = this.info.channels;
    this.sampleRate = this.info.sampleRate;
    this.totalFrames = this.info.frames;
    this.duration = this.totalFrames / this.sampleRate;
    this.lodCache.clear();
  }

  /** Read a single sample as float [-1, 1]. channel=-1 for merged average. */
  sample(frame: number, channel: number): number {
    if (!this.raw || frame < 0 || frame >= this.totalFrames) return 0;
    const nch = this.info!.channels;
    if (channel === -1) {
      let sum = 0;
      for (let c = 0; c < nch; c++) sum += this.raw[frame * nch + c];
      return sum / nch / 32768;
    }
    return this.raw[frame * nch + channel] / 32768;
  }

  getLOD(channel: number): LODLevel[] {
    const cached = this.lodCache.get(channel);
    if (cached) return cached;
    const levels = this.buildLOD(channel);
    this.lodCache.set(channel, levels);
    return levels;
  }

  /** Create a mono AudioBuffer for playback of one channel (or merged). */
  createAudioBuffer(channel: number): AudioBuffer | null {
    if (!this.raw || !this.info) return null;
    const ctx = new OfflineAudioContext(1, this.totalFrames, this.sampleRate);
    const buf = ctx.createBuffer(1, this.totalFrames, this.sampleRate);
    const out = buf.getChannelData(0);
    const raw = this.raw;
    const nch = this.info.channels;

    if (channel === -1) {
      for (let i = 0; i < this.totalFrames; i++) {
        let sum = 0;
        for (let c = 0; c < nch; c++) sum += raw[i * nch + c];
        out[i] = sum / nch / 32768;
      }
    } else {
      for (let i = 0; i < this.totalFrames; i++) {
        out[i] = raw[i * nch + channel] / 32768;
      }
    }
    return buf;
  }

  private buildLOD(channel: number): LODLevel[] {
    if (!this.raw || !this.info) return [];
    const raw = this.raw;
    const frames = this.totalFrames;
    const nch = this.info.channels;
    const levels: LODLevel[] = [];

    const B0 = 256;
    const n0 = Math.ceil(frames / B0);
    const min0 = new Float32Array(n0);
    const max0 = new Float32Array(n0);

    for (let b = 0; b < n0; b++) {
      const start = b * B0;
      const end = Math.min(start + B0, frames);
      let lo = Infinity, hi = -Infinity;

      if (channel === -1) {
        for (let i = start; i < end; i++) {
          let sum = 0;
          for (let c = 0; c < nch; c++) sum += raw[i * nch + c];
          const v = sum / nch / 32768;
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      } else {
        for (let i = start; i < end; i++) {
          const v = raw[i * nch + channel] / 32768;
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
      min0[b] = lo;
      max0[b] = hi;
    }
    levels.push({ min: min0, max: max0, bucketSize: B0 });

    while (levels[levels.length - 1].min.length > 512) {
      const prev = levels[levels.length - 1];
      const F = 4;
      const count = Math.ceil(prev.min.length / F);
      const mn = new Float32Array(count);
      const mx = new Float32Array(count);
      for (let b = 0; b < count; b++) {
        const s = b * F;
        const e = Math.min(s + F, prev.min.length);
        let lo = Infinity, hi = -Infinity;
        for (let j = s; j < e; j++) {
          if (prev.min[j] < lo) lo = prev.min[j];
          if (prev.max[j] > hi) hi = prev.max[j];
        }
        mn[b] = lo;
        mx[b] = hi;
      }
      levels.push({ min: mn, max: mx, bucketSize: prev.bucketSize * F });
    }

    return levels;
  }
}

function parseWavHeader(buffer: ArrayBuffer): WavInfo {
  const view = new DataView(buffer);
  const channels = view.getUint16(22, true);
  const sampleRate = view.getUint32(24, true);
  const bitsPerSample = view.getUint16(34, true);

  let offset = 12;
  while (offset < buffer.byteLength - 8) {
    const id = String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3),
    );
    const size = view.getUint32(offset + 4, true);
    if (id === 'data') {
      const dataOffset = offset + 8;
      const frames = size / (channels * (bitsPerSample / 8));
      return { channels, sampleRate, bitsPerSample, dataOffset, frames };
    }
    offset += 8 + size;
    if (offset % 2 !== 0) offset++;
  }
  throw new Error('No data chunk found in WAV file');
}
