/** Web Worker — computes STFT for a single spectrogram tile. */

self.onmessage = (e: MessageEvent) => {
  const { tileKey, samples, fftSize, hopSize } = e.data as {
    tileKey: string;
    samples: Float32Array;
    fftSize: number;
    hopSize: number;
  };

  const freqBins = (fftSize >>> 1) + 1;
  const frames =
    samples.length >= fftSize
      ? Math.floor((samples.length - fftSize) / hopSize) + 1
      : 0;
  const magnitudes = new Float32Array(frames * freqBins);

  // Hann window
  const win = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
  }

  const re = new Float64Array(fftSize);
  const im = new Float64Array(fftSize);

  for (let f = 0; f < frames; f++) {
    const off = f * hopSize;
    for (let i = 0; i < fftSize; i++) {
      re[i] = samples[off + i] * win[i];
      im[i] = 0;
    }

    fft(re, im);

    const base = f * freqBins;
    for (let k = 0; k < freqBins; k++) {
      const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]) / fftSize;
      magnitudes[base + k] = 20 * Math.log10(Math.max(mag, 1e-10));
    }
  }

  postMessage(
    { type: 'result', tileKey, magnitudes, frames, freqBins },
    { transfer: [magnitudes.buffer] },
  );
};

/* ---- Radix-2 Cooley–Tukey FFT (in-place) ---- */

function fft(re: Float64Array, im: Float64Array) {
  const n = re.length;

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }

  // Butterfly stages
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (-2 * Math.PI) / len;
    const wR = Math.cos(ang);
    const wI = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cR = 1,
        cI = 0;
      for (let j = 0; j < half; j++) {
        const a = i + j,
          b = a + half;
        const tR = cR * re[b] - cI * im[b];
        const tI = cR * im[b] + cI * re[b];
        re[b] = re[a] - tR;
        im[b] = im[a] - tI;
        re[a] += tR;
        im[a] += tI;
        const nR = cR * wR - cI * wI;
        cI = cR * wI + cI * wR;
        cR = nR;
      }
    }
  }
}
