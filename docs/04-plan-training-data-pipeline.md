# Training Data Pipeline — LCCC-based Turn Detection Dataset

## Goal

Generate a labeled audio dataset for turn detection (complete vs. incomplete)
from Chinese conversational text, using the following pipeline:

    LCCC conversations
      → split by conversation (train/eval/test)
      → LLM generates complete + incomplete text variants
      → TTS synthesizes speech audio (WAV)
      → labeled audio dataset

## 1. Source Dataset

**LCCC-base** (`thu-coai/lccc`, base split — ~500K strictly filtered dialogues)

Each sample is a multi-turn dialogue: `["你好", "你好呀", "最近怎么样", ...]`

Why LCCC-base over LCCC-large:
- Strictly filtered — less noise, fewer broken sentences
- 500K dialogues is already more than enough (each produces multiple samples)
- Cleaner text → better TTS output

## 2. Data Split Strategy (No Leakage)

**Split at the conversation level, not the utterance level.**

    ┌─────────────────────────────────┐
    │        LCCC conversations       │
    │  (each = multi-turn dialogue)   │
    └──────────┬──────────────────────┘
               │ deterministic hash split
               ▼
       ┌───────────────────────┐
       │  train  80%  (~400K)  │
       │  eval   10%  (~ 50K)  │
       │  test   10%  (~ 50K)  │
       └───────────────────────┘
               │
               ▼  per-utterance expansion
       each utterance → 1 complete + 1 incomplete sample

**Why conversation-level split:**
- Utterances in the same conversation share topic, vocabulary, and speaker style
- Splitting at utterance level would leak conversational context across splits
- A model could memorize topic patterns rather than learning turn boundaries

**Implementation:** Hash the full conversation text (SHA-256) → use last byte to
assign split deterministically. This is reproducible without storing split IDs.

```python
import hashlib

def assign_split(conversation: list[str]) -> str:
    key = "\n".join(conversation).encode()
    h = int(hashlib.sha256(key).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "eval"
    else:
        return "test"
```

## 3. Text Generation (LLM)

For each utterance in a conversation, generate two variants with TTS instructions:

| Label        | Fields             | Example                                          |
|--------------|--------------------|--------------------------------------------------|
| **complete** | text + tts_instruct | `"我明天要去北京出差。"` + `"Calm declarative tone, falling intonation at the end."` |
| incomplete   | text + tts_instruct | `"我明天要去"` + `"Hesitant, trailing off mid-sentence, rising intonation."` |

### What the LLM does

The LLM takes each utterance and produces:
1. **Complete version** — a self-contained sentence. Minor rewording is OK to
   make it sound natural when spoken aloud (remove internet slang, add proper
   sentence-final particles like 啊/吧/呢). Plus a TTS instruct describing
   the speaking style for a completed turn.
2. **Incomplete version** — a natural truncation: cut mid-clause, trail off,
   or pause at a point where the speaker clearly has more to say. Plus a TTS
   instruct describing the speaking style for an interrupted/unfinished turn.

TTS instructs must be in **English** (Qwen3-TTS instruct language).
Text content must be in **Chinese** (target language).

### Model choice

**OpenRouter API** — use any strong Chinese-capable model without local GPU.

- Access to Qwen, DeepSeek, Gemini, Claude, etc. via a single API
- No local GPU needed for this step — offload compute to the cloud
- Pay per token, swap models easily to compare quality
- Recommended starting model: `qwen/qwen3-235b-a22b` (best Chinese quality)

### Prompt design

System prompt:

> You are a speech data labelling assistant. Given a Chinese conversational
> utterance, produce two versions:
>
> 1. **complete**: A natural, spoken-style complete sentence in Chinese, plus
>    a short English TTS instruction describing how a speaker would say a
>    finished thought (e.g. tone, intonation, emotion).
> 2. **incomplete**: A naturally truncated version of the sentence in Chinese
>    (cut mid-clause, trail off), plus a short English TTS instruction
>    describing how a speaker would sound when interrupted or still thinking.
>
> Output JSON only, no explanation. The `text` fields must be Chinese.
> The `tts_instruct` fields must be English.

User input: `我明天要去北京出差`

Expected output:

```json
{
  "complete": {
    "text": "我明天要去北京出差。",
    "tts_instruct": "Calm and steady, declarative tone with falling intonation at the end."
  },
  "incomplete": {
    "text": "我明天要去",
    "tts_instruct": "Hesitant, trailing off mid-sentence as if thinking about what to say next."
  }
}
```

### Batch processing

Use OpenRouter with async HTTP for throughput:

```python
import openai

client = openai.AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

response = await client.chat.completions.create(
    model="qwen/qwen3-235b-a22b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": utterance},
    ],
    response_format={"type": "json_object"},
    temperature=0.7,
    max_tokens=128,
)
```

### Filtering

Post-generation quality checks:
- Parse JSON; discard malformed outputs (must have complete/incomplete with text + tts_instruct)
- Discard if `incomplete.text` is longer than `complete.text`
- Discard if `incomplete.text` is identical to `complete.text`
- Discard if any `text` is empty or too short (< 2 chars)
- Discard if `text` contains non-speech characters (URLs, hashtags, emojis)
- Discard if `tts_instruct` is empty or not in English

## 4. Speech Synthesis (TTS)

### Model choice

**Qwen3-TTS VoiceDesign** (`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`)

- Apache 2.0 license
- 1.7B params, ~8 GB VRAM (bfloat16)
- **Voice design from natural language** — describe any voice persona in text
  (age, gender, tone, emotion, speaking style) and the model creates it
- No preset speakers or reference audio needed — unlimited speaker diversity
- Runs on T4 (16 GB VRAM)

### Voice design for speaker diversity + turn labels

VoiceDesign takes a `voice_description` that defines the speaker persona, plus
an `instruct` that controls per-utterance prosody. This gives us two levers:

1. **`voice_description`** — defines who is speaking (generated once per speaker)
2. **`instruct`** — defines how they speak this particular utterance (per sample)

#### Speaker personas

Pre-generate a pool of ~20 voice descriptions covering diverse demographics:

```python
VOICE_POOL = [
    "年轻女性，声音清亮活泼，语速偏快",
    "中年男性，声音低沉稳重，语速适中",
    "年轻男性，声音明朗有活力，略带磁性",
    "中年女性，声音温柔沉稳，语调平和",
    "老年男性，声音沙哑浑厚，语速较慢",
    # ... more variations
]
```

Randomly assign a voice to each sample; track voice ID in metadata.

#### Prosody instruct per sample

The `tts_instruct` is **not hardcoded** — it is generated by the LLM in step 2
alongside the text, tailored to each specific utterance. This produces more
natural and varied prosody than a single fixed instruction per label.

The combination of voice_description + per-sample instruct makes synthesized
audio more realistic: diverse speakers with prosody that matches the content.

### Audio format

- **Sample rate:** 16 kHz (matches Whisper input, resample from TTS native rate)
- **Channels:** mono
- **Bit depth:** 16-bit PCM
- **Format:** WAV
- **Max duration:** 8 seconds (matching existing training data spec)
- Trim trailing silence to ~200 ms (matching existing data convention)

### Batch processing

```python
from qwen_tts import Qwen3TTSModel
import torch
import soundfile as sf

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

voice = "年轻女性，声音清亮活泼，语速偏快"

# Complete sample — instruct from LLM
wavs, sr = model.generate_voice_design(
    text=sample["complete"]["text"],
    language="Chinese",
    voice_description=voice,
    instruct=sample["complete"]["tts_instruct"],  # e.g. "Calm and steady, falling intonation."
)
sf.write("complete.wav", wavs[0], sr)

# Incomplete sample — instruct from LLM
wavs, sr = model.generate_voice_design(
    text=sample["incomplete"]["text"],
    language="Chinese",
    voice_description=voice,
    instruct=sample["incomplete"]["tts_instruct"],  # e.g. "Hesitant, trailing off mid-sentence."
)
sf.write("incomplete.wav", wavs[0], sr)
```

## 5. Output Format

### Directory structure

Follows the existing raw data convention from upstream:

    output/
    ├── train/
    │   ├── zh/
    │   │   ├── complete-nofiller/
    │   │   │   ├── {uuid}.wav
    │   │   │   └── ...
    │   │   └── incomplete-nofiller/
    │   │       ├── {uuid}.wav
    │   │       └── ...
    ├── eval/
    │   └── zh/
    │       ├── complete-nofiller/
    │       └── incomplete-nofiller/
    └── test/
        └── zh/
            ├── complete-nofiller/
            └── incomplete-nofiller/

### Metadata

Each split gets a `metadata.jsonl` with one line per sample:

```json
{"uuid": "...", "text": "...", "tts_instruct": "...", "label": "complete", "voice_id": "V03", "source_conv_hash": "a3f2..."}
```

## 6. Pipeline Steps (Execution Order)

| Step | Script               | Input                  | Output                 | Compute |
|------|----------------------|------------------------|------------------------|---------|
| 1    | `01_split_lccc.py`   | LCCC-base from HF      | `splits/{train,eval,test}.jsonl` | CPU |
| 2    | `02_generate_text.py`| `splits/*.jsonl`        | `text/{split}.jsonl` with complete/incomplete | OpenRouter API |
| 3    | `03_synthesize.py`   | `text/*.jsonl`          | `audio/{split}/zh/{label}/*.wav` | GPU (Qwen3-TTS) |
| 4    | `04_build_dataset.py`| `audio/` tree           | HuggingFace Dataset (Arrow) | CPU |

Each step is idempotent — it checks for existing outputs and skips them,
so you can resume after failures.

## 7. Scale Estimate

Starting conservative, can scale up later:

| Parameter         | Value       | Notes                       |
|-------------------|-------------|-----------------------------|
| Conversations     | 10K (of 500K) | Start small, validate quality |
| Utterances/conv   | ~4 avg      | ~40K utterances             |
| Samples           | ~80K        | 40K × 2 (complete + incomplete) |
| Audio duration    | ~3s avg     | ~67 hours total             |
| Storage           | ~8 GB       | 16-bit WAV at 16 kHz        |

For comparison, upstream Smart Turn v3.2 uses 270K samples (~41 GB).
We can scale to full 500K conversations (~4M samples) once quality is validated.

## 8. Open Questions

- [ ] Should we include filler words (嗯, 啊, 那个) as a separate category?
      Upstream data has `midfiller` and `endfiller` labels.
- [ ] Do we want to add background noise augmentation, or keep it clean and
      let training handle augmentation?
- [ ] How many voice descriptions should we pre-generate for the VoiceDesign
      speaker pool? Starting with ~20, but more may help generalization.
- [ ] Which OpenRouter model gives the best cost/quality tradeoff for text
      generation? Start with Qwen3-235B, compare with cheaper alternatives.
- [ ] Should we add guardrails on the LLM-generated TTS instructs (e.g., max
      length, banned keywords) to keep them within Qwen3-TTS's capabilities?
