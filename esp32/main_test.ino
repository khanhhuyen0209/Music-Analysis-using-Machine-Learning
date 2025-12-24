#include <Arduino.h>
#include <driver/i2s.h>
#include <arduinoFFT.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  float arousal;
  float valence;
} EmotionAV;


// ==================== WEIGHTS ====================
#include "mlp_baseline_weights.h"   // Pitch
#include "tempo_lr_model.h"         // Tempo LR
#include "mlp_weights.h"            // Chord
#include "genre_mlp_weights.h"      // Genre
#include "emotion_mlp.h"            // Emotion

// ==================== CONFIG =====================
#define SAMPLE_RATE     22050
#define SEGMENT_SEC     1
#define NUM_SEGMENTS    8

#define FFT_SIZE        1024
#define AUDIO_LEN       (SAMPLE_RATE * SEGMENT_SEC)

// Pitch label config
#define MIDI_MIN        40
#define MIDI_MAX        88
#define N_VOICED        (MIDI_MAX - MIDI_MIN + 1)   // 49

// Mel config
#define N_MELS          64
#define FMIN_HZ         80.0f
#define FMAX_HZ         2000.0f

// Shift for Pitch MLP integer
#define L0_SHIFT  6
#define L1_SHIFT  8
#define L2_SHIFT  7
#define L3_SHIFT  8

// Gate unvoiced
#define RMS_UNVOICED_TH 0.010f

// Tempo config
#define TEMPO_MIN_BPM   40
#define TEMPO_MAX_BPM   200

// Feature config
#define N_ACF           200
#define HOP_LENGTH      256
#define WIN_LENGTH      1024

#define MAX_ONSET_FRAMES ((NUM_SEGMENTS * AUDIO_LEN) / HOP_LENGTH + 8)
#define MAX_ONSETS      32

// Genre config
#define GENRE_N_FEATURES 58
#define GENRE_HIDDEN_UNITS1 128
#define GENRE_HIDDEN_UNITS2 64
#define GENRE_N_CLASSES 10

// Emotion config
#define EMO_N_FEATURES 72
#define EMO_N_CLASSES 2
#define EMO_N_HIDDEN_LAYERS 2
#define EMO_HIDDEN_UNITS1 256
#define EMO_HIDDEN_UNITS2 128


// ==================== GLOBAL BUFFERS =====================
// Audio
static int16_t audio_buffer[AUDIO_LEN];

// FFT buffers
static float vReal[FFT_SIZE];
static float vImag[FFT_SIZE];
ArduinoFFT<float> FFT(vReal, vImag, FFT_SIZE, (float)SAMPLE_RATE);

// Chroma
static float chroma[12];

// Mel filterbank bins
static uint16_t mel_left[N_MELS];
static uint16_t mel_center[N_MELS];
static uint16_t mel_right[N_MELS];

// Tempo onset env
static float g_onset_env[MAX_ONSET_FRAMES];
static int   g_onset_len = 0;
static float g_prev_energy = 0.0f;

// State across loops (smoothing)
static int g_prev_chord = -1;
static int g_prev_pitch = -1;

// Chord accumulation
static float g_chord_logmel_acc[64];
static int   g_chord_votes = 0;
static const char* CHORD_LABELS[8] = {"Am","Bb","Bdim","C","Dm","Em","F","G"};

// Song-level BPM result (for genre feature 57)
static int g_bpm_final = 0;

// ==================== SCALERS (Paste from your training) =====================
// Pitch scaler
static const float PITCH_MEAN[64] = {
 -31.47932516f, -31.61972600f, -34.58918788f, -31.43537355f, -29.15595713f, -29.38718413f, -31.50365041f, -33.58297018f,
 -34.13655313f, -34.96699259f, -36.43684129f, -37.82619215f, -39.41558779f, -40.53129804f, -42.09631899f, -42.42878572f,
 -43.50092310f, -46.13483026f, -48.68916782f, -49.24920747f, -49.13421082f, -51.26572491f, -52.63556974f, -52.29285503f,
 -53.11413171f, -52.14604688f, -52.12687569f, -51.38349772f, -52.05142022f, -50.65060592f, -50.57307541f, -52.74523542f,
 -54.91302744f, -54.85145829f, -56.98773544f, -53.41452484f, -53.41389753f, -51.60816136f, -52.39674190f, -52.68249512f,
 -51.82719158f, -52.67357656f, -53.26030450f, -55.27142623f, -54.17020182f, -57.95720256f, -56.94443876f, -58.20819817f,
 -57.87664787f, -57.94370646f, -57.26015645f, -57.07221509f, -58.74828378f, -57.61877876f, -57.02966454f, -56.83571400f,
 -56.72278572f, -56.50173418f, -55.72662125f, -56.03672070f, -56.04366191f, -55.51121848f, -54.87711850f, -56.70047583f
};
static const float PITCH_SCALE[64] = {
 13.35365875f, 13.69935291f, 15.74900158f, 15.64776590f, 15.31074646f, 15.52688770f, 15.59203407f, 14.83102993f,
 14.54253457f, 15.16858525f, 15.53326505f, 15.28363189f, 15.51706467f, 15.90098960f, 15.71509448f, 14.95779785f,
 15.02571186f, 15.47151084f, 15.92347857f, 16.13870384f, 15.70969493f, 15.90199158f, 14.94252739f, 14.36264607f,
 14.80975917f, 14.97840065f, 13.95066576f, 15.16707188f, 15.17213327f, 15.78063226f, 15.05557797f, 14.43866076f,
 14.38127501f, 14.50584364f, 14.73796606f, 14.81222148f, 15.33217285f, 15.19824461f, 16.32967383f, 16.63241848f,
 16.05629822f, 16.57244621f, 15.69292181f, 16.22750633f, 14.86925324f, 14.76680832f, 14.20115910f, 14.49847376f,
 14.95698867f, 13.57675454f, 15.03158274f, 13.55913212f, 14.21718687f, 13.24097547f, 14.76442546f, 14.82053051f,
 14.83564059f, 15.87173663f, 14.41766203f, 15.88039520f, 14.30585901f, 16.48745214f, 15.72730351f, 14.79999728f
};

static const char* NOTE_NAMES[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};

// Chord scaler
static const float CHORD_MEAN[16] = {
  -2.03650455f, -2.26095502f, -2.45058618f, -4.14766748f, -4.66261261f, -4.88741472f, -5.50021994f, -5.58704452f,
  -6.32571979f, -6.25388386f, -6.89863926f, -7.02966055f, -6.83384801f, -6.77439976f, -6.81123306f, -7.58411825f
};
static const float CHORD_SCALE[16] = {
  2.53589825f, 1.97801876f, 2.00723825f, 2.00208736f, 2.12831373f, 2.20478826f, 2.28146183f, 2.42331475f,
  2.13611270f, 2.04229769f, 2.11201004f, 2.17831763f, 2.17883348f, 2.29924031f, 2.48667369f, 2.30689137f
};

// Chord transitions and key masks (same as your code)
static const float CHORD_TRANS[8][8] = {
  {0.20, 0.05, 0.05, 0.15, 0.15, 0.15, 0.10, 0.15}, // Am
  {0.05, 0.20, 0.05, 0.15, 0.10, 0.05, 0.20, 0.20}, // Bb
  {0.10, 0.05, 0.10, 0.30, 0.10, 0.10, 0.10, 0.15}, // Bdim
  {0.15, 0.10, 0.05, 0.15, 0.15, 0.10, 0.15, 0.15}, // C
  {0.15, 0.10, 0.05, 0.15, 0.15, 0.10, 0.10, 0.20}, // Dm
  {0.20, 0.05, 0.05, 0.15, 0.15, 0.15, 0.10, 0.15}, // Em
  {0.10, 0.15, 0.05, 0.20, 0.15, 0.05, 0.10, 0.20}, // F
  {0.15, 0.10, 0.05, 0.25, 0.15, 0.10, 0.10, 0.10}  // G
};
static const bool CHORD_IN_KEY[12][8] = {
  {1,0,1,1,1,1,1,1}, // C
  {1,1,1,0,0,1,0,0}, // C#
  {1,0,1,1,1,1,1,1}, // D
  {1,1,0,1,0,1,1,0}, // Eb
  {1,0,1,0,1,1,0,1}, // E
  {0,1,1,1,0,1,1,0}, // F
  {1,0,1,0,1,0,1,1}, // F#
  {0,1,1,1,0,1,1,0}, // G
  {1,0,0,0,1,1,0,1}, // Ab
  {1,0,1,1,0,1,1,0}, // A
  {0,1,0,0,1,1,0,1}, // Bb
  {1,0,1,0,1,0,1,1}  // B
};

// Key profiles
static const int major_profile[12] = { 6,2,3,2,5,4,2,6,2,3,2,4 };
static const int minor_profile[12] = { 6,2,3,5,2,4,2,6,3,2,4,2 };

// ==================== GENRE ACCUMULATORS (FIXED) =====================
// We accumulate sums and sumsq for every feature index used in finalize.
static float genre_sum[GENRE_N_FEATURES];
static float genre_sumsq[GENRE_N_FEATURES];
static int   genre_count = 0;

// ==================== HELPERS =====================
static inline float hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}
static inline float mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}
static inline uint16_t hz_to_fftbin(float hz) {
  float bin = (hz * (FFT_SIZE + 1)) / (float)SAMPLE_RATE;
  int b = (int)floorf(bin);
  if (b < 0) b = 0;
  int maxb = FFT_SIZE / 2;
  if (b > maxb) b = maxb;
  return (uint16_t)b;
}

static inline float compute_rms(const int16_t *x, int n) {
  double sum = 0;
  for (int i = 0; i < n; i++) {
    float v = x[i] / 32768.0f;
    sum += v * v;
  }
  return sqrt(sum / n);
}

static inline int detect_dynamics(float rms) {
  if (rms < 0.01f) return 0;
  else if (rms < 0.03f) return 1;
  else return 2;
}

static inline bool detect_onset(float prev, float curr) {
  return curr > prev * 1.5f;
}

static inline int freq_to_chroma(float freq) {
  if (freq < 50.0f) return -1;
  int midi = (int)roundf(69.0f + 12.0f * log2f(freq / 440.0f));
  return (midi % 12 + 12) % 12;
}

static inline void midi_to_note_name(int midi, char* out, size_t n) {
  int note = ((midi % 12) + 12) % 12;
  int octave = (midi / 12) - 1;
  snprintf(out, n, "%s%d", NOTE_NAMES[note], octave);
}

// ==================== I2S RECORD =====================
static void record_audio_1s() {
  size_t bytes_read = 0;
  i2s_read(I2S_NUM_0, audio_buffer, sizeof(audio_buffer), &bytes_read, portMAX_DELAY);
}

// ==================== MEL FILTERBANK INIT =====================
static void init_mel_filterbank() {
  float mel_min = hz_to_mel(FMIN_HZ);
  float mel_max = hz_to_mel(FMAX_HZ);

  for (int m = 0; m < N_MELS; m++) {
    float mel_l = mel_min + (mel_max - mel_min) * (m + 0) / (N_MELS + 1);
    float mel_c = mel_min + (mel_max - mel_min) * (m + 1) / (N_MELS + 1);
    float mel_r = mel_min + (mel_max - mel_min) * (m + 2) / (N_MELS + 1);

    float hz_l = mel_to_hz(mel_l);
    float hz_c = mel_to_hz(mel_c);
    float hz_r = mel_to_hz(mel_r);

    mel_left[m]   = hz_to_fftbin(hz_l);
    mel_center[m] = hz_to_fftbin(hz_c);
    mel_right[m]  = hz_to_fftbin(hz_r);

    if (mel_center[m] <= mel_left[m])   mel_center[m] = mel_left[m] + 1;
    if (mel_right[m]  <= mel_center[m]) mel_right[m]  = mel_center[m] + 1;
    if (mel_right[m]  > FFT_SIZE/2)     mel_right[m]  = FFT_SIZE/2;
  }
}

static inline float mel_energy_from_mag2(const float* mag, int l, int c, int r) {
  if (r <= l) return 0.0f;
  float sum = 0.0f;

  float denom1 = (float)(c - l);
  for (int i = l; i < c; i++) {
    float w = (denom1 > 0) ? ((float)(i - l) / denom1) : 0.0f;
    float p = mag[i] * mag[i];
    sum += w * p;
  }
  float denom2 = (float)(r - c);
  for (int i = c; i <= r; i++) {
    float w = (denom2 > 0) ? ((float)(r - i) / denom2) : 0.0f;
    float p = mag[i] * mag[i];
    sum += w * p;
  }
  return sum;
}

// ==================== FFT PROCESSING =====================
static void process_fft_magnitude_from_segment_head() {
  for (int i = 0; i < FFT_SIZE; i++) {
    vReal[i] = (float)audio_buffer[i] / 32768.0f;
    vImag[i] = 0.0f;
  }
  FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
  FFT.compute(FFTDirection::Forward);
  FFT.complexToMagnitude(); // vReal becomes magnitude
}

static void accumulate_chroma_from_current_fft() {
  for (int i = 1; i < FFT_SIZE / 2; i++) {
    float freq = (i * (float)SAMPLE_RATE) / (float)FFT_SIZE;
    int c = freq_to_chroma(freq);
    if (c >= 0) chroma[c] += vReal[i];
  }
}

// ==================== PITCH INPUT + MLP =====================
static void build_pitch_input_q(int16_t x_q[64]) {
  float e[64];
  float maxe = 1e-12f;

  for (int m = 0; m < 64; m++) {
    int l = mel_left[m], c = mel_center[m], r = mel_right[m];
    float em = mel_energy_from_mag2(vReal, l, c, r);
    e[m] = em;
    if (em > maxe) maxe = em;
  }

  const float DBK = 4.34294482f; // 10*log10(x) = DBK*ln(x)

  for (int m = 0; m < 64; m++) {
    float ratio = (e[m] + 1e-12f) / maxe;
    float db = DBK * logf(ratio); // <= 0
    float z  = (db - PITCH_MEAN[m]) / PITCH_SCALE[m];

    int q = (int)lrintf(z * MLP_BASELINE_INPUT_SCALE);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    x_q[m] = (int16_t)q;
  }
}

static inline int16_t clamp_i16(int32_t x) {
  if (x > 32767) return 32767;
  if (x < -32768) return -32768;
  return (int16_t)x;
}

static int pitch_predict_idx_fast(const int16_t x0[64]) {
  static int16_t h0[256];
  static int16_t h1[128];
  static int16_t h2[64];

  for (int o = 0; o < 256; o++) {
    int32_t acc = MLP_BASELINE_B0[o];
    for (int i = 0; i < 64; i++) acc += (int32_t)MLP_BASELINE_W0[o][i] * (int32_t)x0[i];
    acc >>= L0_SHIFT;
    if (acc < 0) acc = 0;
    h0[o] = clamp_i16(acc);
  }

  for (int o = 0; o < 128; o++) {
    int32_t acc = MLP_BASELINE_B1[o];
    for (int i = 0; i < 256; i++) acc += (int32_t)MLP_BASELINE_W1[o][i] * (int32_t)h0[i];
    acc >>= L1_SHIFT;
    if (acc < 0) acc = 0;
    h1[o] = clamp_i16(acc);
  }

  for (int o = 0; o < 64; o++) {
    int32_t acc = MLP_BASELINE_B2[o];
    for (int i = 0; i < 128; i++) acc += (int32_t)MLP_BASELINE_W2[o][i] * (int32_t)h1[i];
    acc >>= L2_SHIFT;
    if (acc < 0) acc = 0;
    h2[o] = clamp_i16(acc);
  }

  int best = 0;
  int32_t bestv = INT32_MIN;
  for (int o = 0; o < 49; o++) {
    int32_t acc = MLP_BASELINE_B3[o];
    for (int i = 0; i < 64; i++) acc += (int32_t)MLP_BASELINE_W3[o][i] * (int32_t)h2[i];
    acc >>= L3_SHIFT;
    if (acc > bestv) { bestv = acc; best = o; }
  }
  return best;
}

static int pitch_smooth_octave(int raw_midi, int prev_midi) {
  if (prev_midi < 0) return raw_midi;
  int diff = abs(raw_midi - prev_midi);
  if (diff >= 12) {
    int candidates[3] = {raw_midi, raw_midi - 12, raw_midi + 12};
    int best = raw_midi;
    int min_diff = diff;
    for (int i = 0; i < 3; i++) {
      int c = candidates[i];
      if (c < MIDI_MIN || c > MIDI_MAX) continue;
      int d = abs(c - prev_midi);
      if (d < min_diff) { min_diff = d; best = c; }
    }
    return best;
  }
  return raw_midi;
}

// ==================== TEMPO (ONSET -> ACF -> LR) =====================
static inline float fast_frame_energy_abs(const int16_t* x, int start) {
  int32_t sumAbs = 0;
  for (int i = 0; i < WIN_LENGTH; i++) {
    int16_t v = x[start + i];
    sumAbs += (v >= 0) ? v : (int16_t)(-v);
  }
  return (float)sumAbs * (1.0f / (WIN_LENGTH * 32768.0f));
}

static inline void tempo_onset_reset() {
  g_onset_len = 0;
  g_prev_energy = 0.0f;
}

static inline void tempo_onset_push_segment(const int16_t* audio, int n) {
  for (int start = 0; start + WIN_LENGTH <= n; start += HOP_LENGTH) {
    float e = fast_frame_energy_abs(audio, start);
    float diff = e - g_prev_energy;
    if (diff < 0.0f) diff = 0.0f;
    if (g_onset_len < MAX_ONSET_FRAMES) {
      g_onset_env[g_onset_len++] = diff;
    }
    g_prev_energy = e;
  }
}

static void tempo_build_acf_norm(float out_acf[N_ACF]) {
  for (int lag = 0; lag < N_ACF; lag++) {
    float s = 0.0f;
    int limit = g_onset_len - lag;
    if (limit > 0) {
      for (int t = 0; t < limit; t++) s += g_onset_env[t] * g_onset_env[t + lag];
    }
    out_acf[lag] = s;
  }

  float mx = 0.0f;
  for (int i = 0; i < N_ACF; i++) if (out_acf[i] > mx) mx = out_acf[i];
  float inv = 1.0f / (mx + 1e-6f);
  for (int i = 0; i < N_ACF; i++) out_acf[i] *= inv;
}

static inline float tempo_lr_predict_class() {
  float acf[N_ACF];
  tempo_build_acf_norm(acf);

  float z[N_ACF];
  for (int i = 0; i < N_ACF; i++) {
    float sc = (TEMPO_LR_SCALE[i] == 0.0f) ? 1.0f : TEMPO_LR_SCALE[i];
    z[i] = (acf[i] - TEMPO_LR_MEAN[i]) / sc;
  }

  int best_k = 0;
  float best_s = -1e30f;
  for (int k = 0; k < TEMPO_LR_N_CLASS; k++) {
    float s = TEMPO_LR_B[k];
    for (int i = 0; i < N_ACF; i++) s += TEMPO_LR_W[k][i] * z[i];
    if (s > best_s) { best_s = s; best_k = k; }
  }
  return (float)best_k;
}

static int tempo_lr_predict_bpm(float coarse_bpm) {
  int cls = (int)lrintf(tempo_lr_predict_class());
  int bpm_lin = TEMPO_MIN_BPM + cls;
  if (bpm_lin < TEMPO_MIN_BPM) bpm_lin = TEMPO_MIN_BPM;
  if (bpm_lin > TEMPO_MAX_BPM) bpm_lin = TEMPO_MAX_BPM;

  if (coarse_bpm > 0.0f) {
    int bpm_c = (int)lrintf(coarse_bpm);
    int cand[3] = { bpm_lin, bpm_lin * 2, bpm_lin / 2 };

    int best = cand[0];
    int bestd = abs(cand[0] - bpm_c);

    for (int i = 1; i < 3; i++) {
      int v = cand[i];
      if (v < TEMPO_MIN_BPM || v > TEMPO_MAX_BPM) continue;
      int d = abs(v - bpm_c);
      if (d < bestd) { bestd = d; best = v; }
    }

    int bpm_mix = (int)lrintf(0.7f * (float)best + 0.3f * (float)bpm_c);
    if (bpm_mix < TEMPO_MIN_BPM) bpm_mix = TEMPO_MIN_BPM;
    if (bpm_mix > TEMPO_MAX_BPM) bpm_mix = TEMPO_MAX_BPM;
    return bpm_mix;
  }
  return bpm_lin;
}

static int bpm_adaptive_fusion(float coarse_bpm, float lr_bpm, int onset_count) {
  if (coarse_bpm < 0 || onset_count < 3) {
    int r = (int)lrintf(lr_bpm);
    if (r < TEMPO_MIN_BPM) r = TEMPO_MIN_BPM;
    if (r > TEMPO_MAX_BPM) r = TEMPO_MAX_BPM;
    return r;
  }

  float beat_regularity = 1.0f / (1.0f + (float)onset_count * 0.05f);
  float weight_onset = 0.3f + beat_regularity * 0.4f;
  float weight_lr = 1.0f - weight_onset;

  float bpm = weight_onset * coarse_bpm + weight_lr * lr_bpm;
  int result = (int)lrintf(bpm);
  if (result < TEMPO_MIN_BPM) result = TEMPO_MIN_BPM;
  if (result > TEMPO_MAX_BPM) result = TEMPO_MAX_BPM;
  return result;
}

// ==================== CHORD ACC + MLP =====================
static inline void chord_reset_acc() {
  memset(g_chord_logmel_acc, 0, sizeof(g_chord_logmel_acc));
  g_chord_votes = 0;
}

static inline void chord_accumulate_from_current_fft() {
  for (int m = 0; m < 64; m++) {
    int l = mel_left[m], c = mel_center[m], r = mel_right[m];
    float em = mel_energy_from_mag2(vReal, l, c, r);
    g_chord_logmel_acc[m] += logf(em + 1e-6f);
  }
  g_chord_votes++;
}

static inline float deq_w0(int8_t q) { return (float)q / MLP_L0_W_SCALE; }
static inline float deq_b0(int32_t q){ return (float)q / MLP_L0_B_SCALE; }
static inline float deq_w1(int8_t q) { return (float)q / MLP_L1_W_SCALE; }
static inline float deq_b1(int32_t q){ return (float)q / MLP_L1_B_SCALE; }
static inline float deq_w2(int8_t q) { return (float)q / MLP_L2_W_SCALE; }
static inline float deq_b2(int32_t q){ return (float)q / MLP_L2_B_SCALE; }

static int chord_apply_music_logic(float logits[8], int key_idx, int prev_chord) {
  for (int i = 0; i < 8; i++) {
    if (!CHORD_IN_KEY[key_idx][i]) logits[i] -= 1.5f;
  }
  if (prev_chord >= 0 && prev_chord < 8) {
    for (int i = 0; i < 8; i++) logits[i] += CHORD_TRANS[prev_chord][i] * 1.0f;
  }
  int best = 0;
  float bestv = logits[0];
  for (int i = 1; i < 8; i++) {
    if (logits[i] > bestv) { bestv = logits[i]; best = i; }
  }
  return best;
}

static int chord_predict_from_acc(float out_logits[8]) {
  if (g_chord_votes <= 0) return -1;

  float mel64[64];
  float invN = 1.0f / (float)g_chord_votes;
  for (int i = 0; i < 64; i++) mel64[i] = g_chord_logmel_acc[i] * invN;

  float x16[16];
  for (int g = 0; g < 16; g++) {
    float s = mel64[g*4 + 0] + mel64[g*4 + 1] + mel64[g*4 + 2] + mel64[g*4 + 3];
    float feat = 0.25f * s;
    float sc = (CHORD_SCALE[g] == 0.0f) ? 1.0f : CHORD_SCALE[g];
    x16[g] = (feat - CHORD_MEAN[g]) / sc;
  }

  static float h0[128];
  static float h1[64];

  for (int o = 0; o < 128; o++) {
    float acc = deq_b0(MLP_B0[o]);
    for (int i = 0; i < 16; i++) acc += x16[i] * deq_w0(MLP_W0[o][i]);
    h0[o] = (acc > 0.0f) ? acc : 0.0f;
  }

  for (int o = 0; o < 64; o++) {
    float acc = deq_b1(MLP_B1[o]);
    for (int i = 0; i < 128; i++) acc += h0[i] * deq_w1(MLP_W1[o][i]);
    h1[o] = (acc > 0.0f) ? acc : 0.0f;
  }

  for (int o = 0; o < 8; o++) {
    float acc = deq_b2(MLP_B2[o]);
    for (int i = 0; i < 64; i++) acc += h1[i] * deq_w2(MLP_W2[o][i]);
    out_logits[o] = acc;
  }

  int best = 0;
  float bestv = out_logits[0];
  for (int k = 1; k < 8; k++) {
    if (out_logits[k] > bestv) { bestv = out_logits[k]; best = k; }
  }
  return best;
}

// ==================== KEY DETECTION =====================
static float key_score(const int *profile, int shift) {
  float score = 0;
  for (int i = 0; i < 12; i++) score += chroma[i] * profile[(i + shift) % 12];
  return score;
}

// ==================== GENRE (SAFE, NO VLA, WDT-FRIENDLY) =====================
static inline void genre_reset_accum() {
  memset(genre_sum, 0, sizeof(genre_sum));
  memset(genre_sumsq, 0, sizeof(genre_sumsq));
  genre_count = 0;
}

static inline float minmax_scale(float x, int i) {
  float denom = GENRE_MAX[i] - GENRE_MIN[i];
  if (denom < 1e-6f) return 0.0f;   // tránh divide by zero
  float y = (x - GENRE_MIN[i]) / denom;

  // clip để ổn định
  if (y < 0.0f) y = 0.0f;
  else if (y > 1.0f) y = 1.0f;

  return y;
}

static inline void genre_add_feat(int idx, float v) {
  if (idx < 0 || idx >= GENRE_N_FEATURES) return;
  genre_sum[idx] += v;
  genre_sumsq[idx] += v * v;
}

static inline void compute_segment_mfccs_mean(const int16_t* audio, int n_samples, float mfccs_out[20]) {
  const int frame_length = 1024;
  const int hop_length = 512;
  int n_frames = (n_samples - frame_length) / hop_length + 1;
  if (n_frames <= 0) n_frames = 1;

  float mfcc_sum[20] = {0};

  for (int f = 0; f < n_frames; f++) {
    int start = f * hop_length;
    if (start + frame_length > n_samples) break;

    // frame -> vReal/vImag with Hamming
    for (int i = 0; i < frame_length; i++) {
      float x = (float)audio[start + i] / 32768.0f;
      float w = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * (float)i / (float)(frame_length - 1));
      vReal[i] = x * w;
      vImag[i] = 0.0f;
    }
    // zero pad if any (shouldn't be needed since frame_length == FFT_SIZE)
    for (int i = frame_length; i < FFT_SIZE; i++) {
      vReal[i] = 0.0f;
      vImag[i] = 0.0f;
    }

    FFT.compute(FFTDirection::Forward);
    FFT.complexToMagnitude();

    float log_mel[64];
    for (int m = 0; m < 64; m++) {
      int l = mel_left[m], c = mel_center[m], r = mel_right[m];
      float em = mel_energy_from_mag2(vReal, l, c, r);
      log_mel[m] = logf(em + 1e-6f);
    }

    for (int k = 0; k < 20; k++) {
      float s = 0.0f;
      for (int n = 0; n < 64; n++) s += log_mel[n] * cosf((float)M_PI * (float)k * ((float)n + 0.5f) / 64.0f);
      mfcc_sum[k] += s;
    }

    // WDT friendly
    if ((f & 0x3) == 0) { yield(); delay(1); }
  }

  for (int k = 0; k < 20; k++) mfccs_out[k] = mfcc_sum[k] / (float)n_frames;
}

static inline void accumulate_genre_features_per_segment(const int16_t* audio, int n_samples, float sr) {
  // Feature 0: length
  genre_add_feat(0, (float)n_samples);

  // Chroma mean/var in your original mapping used indices:
  // 1..12 (chroma mean) and 13..24 (chroma var proxy)
  // We'll store chroma values as "observations" per segment and finalize by mean/var later.
  for (int i = 0; i < 12; i++) {
    genre_add_feat(1 + i, chroma[i]);
    genre_add_feat(13 + i, chroma[i]); // keep same style; finalize uses sumsq anyway
  }

  float rms_val = compute_rms(audio, n_samples);
  genre_add_feat(25, rms_val);
  genre_add_feat(26, rms_val); // finalize uses sumsq too

  // Spectral centroid from current vReal (already has magnitude spectrum from segment-head FFT)
  float spec_cent = 0.0f, total_energy = 0.0f;
  for (int i = 1; i < FFT_SIZE / 2; i++) {
    float freq = (i * sr) / FFT_SIZE;
    float mag = vReal[i];
    float e = mag * mag;
    spec_cent += freq * e;
    total_energy += e;
  }
  if (total_energy > 0) spec_cent /= total_energy;
  genre_add_feat(27, spec_cent);
  genre_add_feat(28, spec_cent);

  // ZCR (compute without huge float array)
  int zcr_count = 0;
  for (int i = 1; i < n_samples; i++) {
    int16_t a = audio[i - 1];
    int16_t b = audio[i];
    if ((a >= 0 && b < 0) || (a < 0 && b >= 0)) zcr_count++;
    if ((i & 0x1FFF) == 0) { yield(); } // WDT friendly in big loops
  }
  float zcr = (float)zcr_count / (float)n_samples;
  genre_add_feat(35, zcr);
  genre_add_feat(36, zcr);

  // MFCC mean (20) mapped to 37..56
  float mfccs[20];
  compute_segment_mfccs_mean(audio, n_samples, mfccs);
  for (int k = 0; k < 20; k++) {
    genre_add_feat(37 + k, mfccs[k]);
    // var slots (57 would be tempo, so we do not use 37+20+k here in this 58-dim scheme)
    // Your original code tried 37+20+k but that would exceed 58 if done naively.
  }

  genre_count++;
}

static inline void finalize_genre_features(float out_features[GENRE_N_FEATURES]) {
  if (genre_count <= 0) {
    // Safe fallback: output standardized zeros
    for (int i = 0; i < GENRE_N_FEATURES; i++) out_features[i] = 0.0f;
    // tempo feature
    out_features[57] = (g_bpm_final - GENRE_INPUT_MEAN[57]) / GENRE_INPUT_SCALE[57];
    return;
  }

  // Standardize using provided GENRE_INPUT_MEAN/SCALE
  for (int i = 0; i < GENRE_N_FEATURES; i++) {
    float mean = genre_sum[i] / (float)genre_count;
    float minmax = minmax_scale(mean, i);
    out_features[i] = (minmax - GENRE_INPUT_MEAN[i]) / GENRE_INPUT_SCALE[i];
  }

  // tempo feature is song-level
  float g_bpm = minmax_scale((float)g_bpm_final, 57);
  out_features[57] = (g_bpm - GENRE_INPUT_MEAN[57]) / GENRE_INPUT_SCALE[57];
}

// Genre MLP dequant
static inline float deq_mlp_w1(int8_t q) { return (float)q / GENRE_W1_SCALE; }
static inline float deq_mlp_b1(int8_t q) { return (float)q / GENRE_B1_SCALE; }
static inline float deq_mlp_w2(int8_t q) { return (float)q / GENRE_W2_SCALE; }
static inline float deq_mlp_b2(int8_t q) { return (float)q / GENRE_B2_SCALE; }
static inline float deq_mlp_w3(int8_t q) { return (float)q / GENRE_W3_SCALE; }
static inline float deq_mlp_b3(int8_t q) { return (float)q / GENRE_B3_SCALE; }

static inline int predict_genre_mlp(const float features[GENRE_N_FEATURES]) {
  static float hidden1[GENRE_HIDDEN_UNITS1];
  static float hidden2[GENRE_HIDDEN_UNITS2];
  float logits[GENRE_N_CLASSES];

  for (int h = 0; h < GENRE_HIDDEN_UNITS1; h++) {
    float acc = deq_mlp_b1(GENRE_B1[h]);
    for (int f = 0; f < GENRE_N_FEATURES; f++) acc += features[f] * deq_mlp_w1(GENRE_W1[f][h]);
    hidden1[h] = (acc > 0) ? acc : 0;
    if ((h & 0x1F) == 0) yield();
  }

  for (int h = 0; h < GENRE_HIDDEN_UNITS2; h++) {
    float acc = deq_mlp_b2(GENRE_B2[h]);
    for (int h1 = 0; h1 < GENRE_HIDDEN_UNITS1; h1++) acc += hidden1[h1] * deq_mlp_w2(GENRE_W2[h1][h]);
    hidden2[h] = (acc > 0) ? acc : 0;
    if ((h & 0x0F) == 0) yield();
  }

  for (int c = 0; c < GENRE_N_CLASSES; c++) {
    float acc = deq_mlp_b3(GENRE_B3[c]);
    for (int h2 = 0; h2 < GENRE_HIDDEN_UNITS2; h2++) acc += hidden2[h2] * deq_mlp_w3(GENRE_W3[h2][c]);
    logits[c] = acc;
  }

  int best = 0;
  float bestv = logits[0];
  for (int c = 1; c < GENRE_N_CLASSES; c++) {
    if (logits[c] > bestv) { bestv = logits[c]; best = c; }
  }
  return best;
}

// ==================== EMOTION ======================
// ----- Dequant helpers -----
static inline float deq_emo_w1(int8_t q) { return (float)q / EMO_W1_SCALE; }
static inline float deq_emo_b1(int8_t q) { return (float)q / EMO_B1_SCALE; }
static inline float deq_emo_w2(int8_t q) { return (float)q / EMO_W2_SCALE; }
static inline float deq_emo_b2(int8_t q) { return (float)q / EMO_B2_SCALE; }
static inline float deq_emo_w3(int8_t q) { return (float)q / EMO_W3_SCALE; }
static inline float deq_emo_b3(int8_t q) { return (float)q / EMO_B3_SCALE; }

// ----- Emotion MLP forward -----
static inline EmotionAV predict_emotion_mlp(const float features[EMO_N_FEATURES]) {
  static float h1[EMO_HIDDEN_UNITS1];
  static float h2[EMO_HIDDEN_UNITS2];
  float out[2];

  // Hidden layer 1
  for (int i = 0; i < EMO_HIDDEN_UNITS1; i++) {
    float acc = deq_emo_b1(EMO_B1[i]);
    for (int f = 0; f < EMO_N_FEATURES; f++) {
      acc += features[f] * deq_emo_w1(EMO_W1[f][i]);
    }
    h1[i] = (acc > 0.0f) ? acc : 0.0f;
    if ((i & 0x1F) == 0) yield();
  }

  // Hidden layer 2
  for (int i = 0; i < EMO_HIDDEN_UNITS2; i++) {
    float acc = deq_emo_b2(EMO_B2[i]);
    for (int j = 0; j < EMO_HIDDEN_UNITS1; j++) {
      acc += h1[j] * deq_emo_w2(EMO_W2[j][i]);
    }
    h2[i] = (acc > 0.0f) ? acc : 0.0f;
    if ((i & 0x0F) == 0) yield();
  }

  // Output layer (linear)
  for (int o = 0; o < 2; o++) {
    float acc = deq_emo_b3(EMO_B3[o]);
    for (int j = 0; j < EMO_HIDDEN_UNITS2; j++) {
      acc += h2[j] * deq_emo_w3(EMO_W3[j][o]);
    }
    out[o] = acc;
  }

  EmotionAV r;
  r.arousal = out[0];
  r.valence = out[1];
  return r;
}

// ----- Map Arousal/Valence to label -----
static const char* map_av_to_emotion(float a, float v, float th = 0.05f) {
  if (fabs(a) < th && fabs(v) < th) return "Neutral";
  if (a >= 0 && v >= 0) return "Happy / Excited";
  if (a >= 0 && v < 0)  return "Angry / Tense";
  if (a < 0 && v >= 0)  return "Relaxed / Calm";
  return "Sad / Depressed";
}

// ==================== SETUP =====================
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("ESP32 Music Analyzer + Pitch (FAST) [REWRITE SAFE]");

  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 512,
    .use_apll = false
  };

  i2s_pin_config_t pin_cfg = {
    .bck_io_num = 26,
    .ws_io_num = 25,
    .data_out_num = -1,
    .data_in_num = 33
  };

  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_cfg);

  init_mel_filterbank();
}

EmotionAV predict_emotion_mlp(const float features[EMO_N_FEATURES]);

// ==================== LOOP =====================
void loop() {
  // reset per-loop
  memset(chroma, 0, sizeof(chroma));
  genre_reset_accum();
  chord_reset_acc();
  tempo_onset_reset();

  float prev_rms = 0.0f;

  float onset_times[MAX_ONSETS];
  int onset_count = 0;

  int dyn_hist[3] = {0};
  int pitch_hist[49] = {0};
  int pitch_votes = 0;

  // ====== collect segments ======
  for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
    record_audio_1s();

    // onset env for LR
    tempo_onset_push_segment(audio_buffer, AUDIO_LEN);

    float rms = compute_rms(audio_buffer, AUDIO_LEN);
    dyn_hist[detect_dynamics(rms)]++;

    if (seg > 0 && detect_onset(prev_rms, rms)) {
      if (onset_count < MAX_ONSETS) onset_times[onset_count++] = millis() / 1000.0f;
    }

    // FFT (segment-head)
    process_fft_magnitude_from_segment_head();
    accumulate_chroma_from_current_fft();

    // chord + pitch only when voiced
    if (rms >= RMS_UNVOICED_TH) {
      chord_accumulate_from_current_fft();

      int16_t x_q[64];
      build_pitch_input_q(x_q);
      int idx = pitch_predict_idx_fast(x_q);
      pitch_hist[idx]++;
      pitch_votes++;
    }

    prev_rms = rms;

    // genre per segment (uses current vReal for spectral centroid)
    accumulate_genre_features_per_segment(audio_buffer, AUDIO_LEN, (float)SAMPLE_RATE);
    // emotion per segment 

    // WDT friendly
    yield();
    delay(1);
  }

  // ====== BPM coarse ======
  float coarse_bpm = -1.0f;
  if (onset_count > 1) {
    float sum = 0;
    for (int i = 1; i < onset_count; i++) sum += onset_times[i] - onset_times[i - 1];
    float bpm = 60.0f / (sum / (float)(onset_count - 1));
    Serial.print("BPM (coarse): ");
    Serial.println(bpm);
    coarse_bpm = bpm;
  }

  // Tempo LR + fusion
  int bpm_lr = tempo_lr_predict_bpm(coarse_bpm);
  g_bpm_final = bpm_adaptive_fusion(coarse_bpm, (float)bpm_lr, onset_count);
  Serial.print("BPM (LR): ");
  Serial.println(bpm_lr);
  Serial.print("BPM (final): ");
  Serial.println(g_bpm_final);

  // ====== Key detection ======
  float best_score = -1e30f;
  int best_key = 0;
  bool is_minor = false;

  for (int k = 0; k < 12; k++) {
    float sM = key_score(major_profile, k);
    float sm = key_score(minor_profile, k);
    if (sM > best_score) { best_score = sM; best_key = k; is_minor = false; }
    if (sm > best_score) { best_score = sm; best_key = k; is_minor = true; }
  }
  Serial.print("Detected Key: ");
  Serial.print(NOTE_NAMES[best_key]);
  Serial.println(is_minor ? " minor" : " major");

  // ====== Pitch vote ======
  if (pitch_votes == 0) {
    Serial.println("Pitch: UNVOICED");
    g_prev_pitch = -1;
  } else {
    int best = 0;
    int bestc = pitch_hist[0];
    for (int i = 1; i < 49; i++) {
      if (pitch_hist[i] > bestc) { bestc = pitch_hist[i]; best = i; }
    }
    int raw_midi = MIDI_MIN + best;
    int midi = pitch_smooth_octave(raw_midi, g_prev_pitch);
    g_prev_pitch = midi;

    char name[8];
    midi_to_note_name(midi, name, sizeof(name));

    Serial.print("Pitch: ");
    Serial.print(name);
    Serial.print("  (MIDI ");
    Serial.print(midi);
    Serial.print(") votes ");
    Serial.println(bestc);
  }

  // ====== Chord ======
  float chord_logits[8];
  int chord_idx = chord_predict_from_acc(chord_logits);
  if (chord_idx < 0) {
    Serial.println("Chord: (no vote)");
    g_prev_chord = -1;
  } else {
    chord_idx = chord_apply_music_logic(chord_logits, best_key, g_prev_chord);
    g_prev_chord = chord_idx;
    Serial.print("Chord: ");
    Serial.println(CHORD_LABELS[chord_idx]);
  }

  // ====== Beat grid + rhythm hints ======
  if (onset_count >= 2 && g_bpm_final > 0) {
    float beat_interval = 60.0f / (float)g_bpm_final;
    Serial.print("Beat Grid: ");
    for (int i = 0; i < onset_count; i++) {
      Serial.print(onset_times[i], 2);
      Serial.print("s ");
    }
    Serial.println();
    Serial.print("Suggested next beat at: ");
    Serial.print(onset_times[onset_count - 1] + beat_interval, 2);
    Serial.println("s");
  }

  if (onset_count >= 3 && g_bpm_final > 0) {
    Serial.print("Rhythm Pattern: ");
    float beat = 60.0f / (float)g_bpm_final;
    for (int i = 1; i < onset_count; i++) {
      float ratio = (onset_times[i] - onset_times[i - 1]) / beat;
      if (ratio < 0.6f) Serial.print("S ");
      else if (ratio < 1.2f) Serial.print("Q ");
      else Serial.print("H ");
    }
    Serial.println();
  }

  if (onset_count >= 4) {
    Serial.print("Measure structure: |");
    for (int i = 0; i < onset_count; i++) {
      Serial.print(i % 4 == 0 ? "1" : "-");
      Serial.print("|");
    }
    Serial.println();
  }

  if (onset_count >= 4) {
    float intervals[MAX_ONSETS];
    float avg = 0;
    for (int i = 1; i < onset_count; i++) {
      intervals[i - 1] = onset_times[i] - onset_times[i - 1];
      avg += intervals[i - 1];
    }
    avg /= (float)(onset_count - 1);

    float variance = 0;
    for (int i = 0; i < onset_count - 1; i++) {
      float d = intervals[i] - avg;
      variance += d * d;
    }
    float std = sqrtf(variance / (float)(onset_count - 1));
    float cv = (avg > 1e-6f) ? (std / avg) : 0.0f;

    if (cv > 0.15f) Serial.println("Swing/Shuffle detected!");
    else Serial.println("Straight timing");
  }

  // ====== Genre ======
  float genre_features[GENRE_N_FEATURES];
  finalize_genre_features(genre_features);
  int genre_idx = predict_genre_mlp(genre_features);
  const char* GENRE_LABELS[10] = {"blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"};
  Serial.print("Genre: ");
  Serial.println(GENRE_LABELS[genre_idx]);

  // ====== Emotion ======
  float genre_onehot[10] = {0};
  genre_onehot[genre_idx] = 1.0f;

  // Tạo mảng feature cuối cùng cho Emotion MLPRegressor
  float final_features[GENRE_N_FEATURES + 12];
  for(int i=0; i<GENRE_N_FEATURES; i++) final_features[i] = genre_features[i];
  final_features[GENRE_N_FEATURES ] = random(0, 100) / 100.0f; // giả sử đây là feature ngẫu nhiên cho ví dụ
  final_features[GENRE_N_FEATURES + 1] = NUM_SEGMENTS; // số segments đã xử lý
  for(int i=0; i<10; i++) final_features[GENRE_N_FEATURES + 2 + i] = genre_onehot[i];
  EmotionAV emo = predict_emotion_mlp(final_features);
  const char* label = map_av_to_emotion(emo.arousal, emo.valence);
  Serial.println(label);

  Serial.println("------------------------");
  delay(2000); // shorter, safer
}

// Note chỉnh sửa Genre
// 1. Thêm minmax_scale() để chuẩn hóa DL trước khi train MLP, cập nhật finalize_genre_features()
// 2. Kqua chuyển thành one-hot vector để đưa vào emotion MLPRegressor
// 3. Các feature giả định bổ sung cho emotion MLPRegressor: song_id (random), num_segments (NUM_SEGMENTS)
//    => Do model ban đầu accumulate qua từng segment nhưng mà như vậy khá nặng nên dùng features_final để lấy kqua cuối luôn