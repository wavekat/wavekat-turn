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

**Qwen2.5-7B-Instruct** (via `vllm` or `transformers`)

- Best-in-class Chinese language support at the 7B scale
- Apache 2.0 license
- Runs on a single T4 GPU (16 GB VRAM) with vLLM
- Quantized (AWQ/GPTQ) variants available if VRAM is tight

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

Use vLLM for high-throughput offline batch inference:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
params = SamplingParams(temperature=0.7, max_tokens=128)
outputs = llm.generate(prompts, params)
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

**CosyVoice 2** (Alibaba, `FunAudioLLM/CosyVoice2-0.5B`)

- Apache 2.0 license
- Best open-source Chinese conversational TTS quality
- Supports multi-speaker with voice cloning (zero-shot)
- Streaming-capable, but we use offline batch mode here

Alternatives considered:
- **ChatTTS** — good quality but license is restrictive (CC-BY-NC 4.0)
- **Fish Speech** — good but less mature
- **MeloTTS** — fast but less natural prosody

### Speaker diversity

To avoid the model learning to detect turns by speaker voice rather than
linguistic content, we need speaker diversity:

- Use 10-20 reference voice clips (mix of male/female, age ranges)
- Randomly assign a speaker to each sample
- Track speaker ID in metadata for analysis

### Audio format

- **Sample rate:** 16 kHz (matches Whisper input)
- **Channels:** mono
- **Bit depth:** 16-bit PCM
- **Format:** WAV
- **Max duration:** 8 seconds (matching existing training data spec)
- Trim trailing silence to ~200 ms (matching existing data convention)

### Batch processing

```python
from cosyvoice import CosyVoice2

model = CosyVoice2("FunAudioLLM/CosyVoice2-0.5B")

for sample in dataset:
    # zero-shot with random reference speaker
    audio = model.inference_zero_shot(
        sample["text"],
        prompt_text=speaker["prompt_text"],
        prompt_speech=speaker["prompt_audio"],
    )
    save_wav(audio, sample["output_path"], sr=16000)
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

| Step | Script               | Input                  | Output                 | GPU? |
|------|----------------------|------------------------|------------------------|------|
| 1    | `01_split_lccc.py`   | LCCC-base from HF      | `splits/{train,eval,test}.jsonl` | No |
| 2    | `02_generate_text.py`| `splits/*.jsonl`        | `text/{split}.jsonl` with complete/incomplete | Yes |
| 3    | `03_synthesize.py`   | `text/*.jsonl` + speaker refs | `audio/{split}/zh/{label}/*.wav` | Yes |
| 4    | `04_build_dataset.py`| `audio/` tree           | HuggingFace Dataset (Arrow) | No |

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
- [ ] How many reference speakers are enough for good diversity?
- [ ] Should we use Qwen2.5-7B or a larger model (14B/72B) for better
      text generation quality? Depends on available VRAM and time budget.
