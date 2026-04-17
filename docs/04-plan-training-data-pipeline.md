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

For each utterance in a conversation, generate two variants:

| Label        | Description                          | Example input      | Example output           |
|--------------|--------------------------------------|--------------------|--------------------------|
| **complete** | Full, natural sentence (end of turn) | `"我明天要去北京出差"` | `"我明天要去北京出差。"`    |
| incomplete   | Truncated mid-thought (mid-turn)     | `"我明天要去北京出差"` | `"我明天要去北京……"`       |

### What the LLM does

The LLM takes each utterance and produces:
1. **Complete version** — a self-contained sentence. Minor rewording is OK to
   make it sound natural when spoken aloud (remove internet slang, add proper
   sentence-final particles like 啊/吧/呢).
2. **Incomplete version** — a natural truncation: cut mid-clause, trail off,
   or pause at a point where the speaker clearly has more to say.

### Model choice

**OpenRouter API** — use any strong Chinese-capable model without local GPU.

- Access to Qwen, DeepSeek, Gemini, Claude, etc. via a single API
- No local GPU needed for this step — offload compute to the cloud
- Pay per token, swap models easily to compare quality
- Recommended starting model: `qwen/qwen3-235b-a22b` (best Chinese quality)

### Prompt design

System prompt:

> 你是一个语音数据标注助手。给定一句中文对话，生成两个版本：
> 1. 完整版：自然、口语化的完整句子，适合朗读。
> 2. 不完整版：在句子中间自然截断，让人感觉说话者还没说完。
>
> 只输出JSON，不要解释。

User input: `我明天要去北京出差`

Expected output:

```json
{"complete": "我明天要去北京出差。", "incomplete": "我明天要去"}
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
- Parse JSON; discard malformed outputs
- Discard if `incomplete` is longer than `complete`
- Discard if `incomplete` is identical to `complete`
- Discard if either text is empty or too short (< 2 chars)
- Discard if text contains non-speech characters (URLs, hashtags, emojis)

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

#### Prosody instruct per label

| Label        | TTS instruct                                   | Effect                         |
|--------------|-------------------------------------------------|--------------------------------|
| **complete** | `"用自然平稳的语气说完整句话，句尾语调下降。"`       | Declarative falling intonation |
| incomplete   | `"说到一半停下来，语气未完，像是还想继续说。"`       | Rising/suspended intonation    |

The combination of voice_description + instruct makes synthesized audio more
realistic: diverse speakers with prosody that matches complete vs. incomplete.

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

# Complete sample — falling intonation
wavs, sr = model.generate_voice_design(
    text=sample["complete"],
    language="Chinese",
    voice_description=voice,
    instruct="用自然平稳的语气说完整句话，句尾语调下降。",
)
sf.write("complete.wav", wavs[0], sr)

# Incomplete sample — suspended intonation
wavs, sr = model.generate_voice_design(
    text=sample["incomplete"],
    language="Chinese",
    voice_description=voice,
    instruct="说到一半停下来，语气未完，像是还想继续说。",
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
{"uuid": "...", "text": "...", "label": "complete", "speaker_id": "F01", "source_conv_hash": "a3f2..."}
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
- [ ] Should we vary the TTS `instruct` prompts per sample (e.g., different
      emotions) or keep them consistent per label?
