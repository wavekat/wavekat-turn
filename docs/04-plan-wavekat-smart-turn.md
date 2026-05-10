# Plan: Distribute WaveKat Smart Turn Fine-tunes via `wavekat-turn`

**Status:** Draft for review
**Date:** 2026-05-11
**Branch:** `feat/wavekat-smart-turn`

> Scope: a language-agnostic distribution path for our own Smart Turn
> fine-tunes. Mandarin (`zh`) is the first language we ship; the design must
> let us add more languages without breaking changes or new HF repos.

---

## What we are (and are not) shipping

**Smart Turn is Pipecat's project.** The architecture (Whisper-Tiny encoder +
binary classification head), the training recipe, the ONNX export pipeline,
and the original `smart-turn-v3.2-cpu.onnx` weights all belong to
[pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn) (BSD 2-Clause).

What WaveKat contributes is **language-specialized weights** that drop into the
same architecture, exported to the **same ONNX interface** Pipecat already
defines. Concretely:

- **Same input tensor:** `input_features`, shape `[B, 80, 800]`, float32.
- **Same output tensor:** `logits`, shape `[B, 1]`, float32 (sigmoid fused).
- **Same audio pipeline:** 16 kHz mono, 8-second window, Whisper-style log-mel
  features (Slaney, n_fft=400, hop=160, 80 mels).

The implication that runs through this whole plan: **anything compatible with
upstream Pipecat Smart Turn must remain compatible with our weights, and vice
versa.** That includes:

1. **Pipecat's own Python loader** (`smart-turn` repo) must be able to consume
   our ONNX files with no code changes. Validated by running the upstream
   Python inference script against our exported ONNX.
2. **Our `wavekat-turn` Rust loader** picks them up via the same `from_file()`
   path used today for the upstream model.
3. **Future ports** (e.g. a Pipecat Python integration, or a third-party
   loader) can use the same files.

Practical consequences for this design:

- The HF repo must be **architecture-named, not crate-named**. It is "ONNX
  weights for Pipecat Smart Turn, fine-tuned by WaveKat", not "weights for
  wavekat-turn".
- The model card must lead with **strong attribution to upstream Pipecat**
  (link to the GitHub repo, the upstream HF org, and the BSD 2-Clause notice)
  before describing our fine-tunes.
- Tensor names, shapes, and feature-extraction parameters are **frozen** to
  match Pipecat. If Pipecat ever revs the architecture (e.g. a v4 with
  different input shape), we add a new family of repos rather than mutating
  the existing one.
- We should not rename `PipecatSmartTurn` to `SmartTurnDetector` in
  `wavekat-turn` (previously suggested as a follow-up). The name correctly
  identifies the *architecture* we are wrapping; both upstream and our
  weights are instances of it. That decision is now reversed — see
  Decision 9.

---

## Context

- `training/smart-turn-zh/` produced a Mandarin fine-tune of Pipecat Smart Turn
  v3 (same architecture: Whisper-Tiny encoder + binary classification head).
- The trained model lives in **wavekat-platform**, which is an **internal-only**
  model registry today (no public anonymous read access).
- `wavekat-turn` is a public OSS crate published to crates.io. Its build script
  downloads `smart-turn-v3.2-cpu.onnx` from HuggingFace at build time and embeds
  the bytes via `include_bytes!()`.
- Goal: make our Chinese model usable from `wavekat-turn` with the same
  zero-setup experience the upstream Pipecat model gets today.

### Sibling-repo precedent: `wavekat-tts`

`wavekat-tts` already publishes WaveKat-owned ONNX weights publicly under the
existing HuggingFace org **[`wavekat`](https://huggingface.co/wavekat)**:

| Repo | Layout | Loading mechanism |
|------|--------|-------------------|
| [`wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX`](https://huggingface.co/wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX) | `fp32/*.onnx`, `int4/*.onnx`, `config.json`, `embeddings/*.npy`, `tokenizer/*` | Runtime download via `hf-hub` crate, cached at `$HF_HOME/hub/` |
| [`wavekat/Qwen3-TTS-0.6B-Base-ONNX`](https://huggingface.co/wavekat/Qwen3-TTS-0.6B-Base-ONNX) | Same shape, plus `speaker_encoder.onnx` / `tokenizer_encoder.onnx` | Same |

Conventions established by `wavekat-tts` that we should follow:
- **HF org name:** `wavekat` (already confirmed live).
- **Repo naming:** `wavekat/<ModelName>-ONNX` with the `-ONNX` suffix.
- **Multi-precision layout:** `fp32/` and `int4/` subdirs inside one repo, so
  users pick precision at runtime instead of at build time.
- **Revision pinning:** the consuming crate pins a dated revision string in
  code (e.g. `REVISION: &str = "2026-04-06"`), so model updates ship via a
  crate release, not silently when a user re-pulls.
- **Local override env var:** `WAVEKAT_MODEL_DIR` (TTS uses
  `WAVEKAT_MODEL_DIR` / `WAVEKAT_CLONE_MODEL_DIR`) lets users point at a
  pre-populated directory and skip downloads entirely — needed for offline
  builds and CI.
- **License:** Apache 2.0 on the consuming crate; the model files inherit
  their upstream license.

---

## Question: HuggingFace first, or load from wavekat-platform?

**Recommendation: HuggingFace first.** Use wavekat-platform as the
source-of-truth training registry; treat HF as the **public distribution
mirror** for snapshots we have explicitly chosen to release. This matches what
`wavekat-tts` already does — the `wavekat` HF org is established and the
pattern is proven across the ecosystem.

| Concern                       | wavekat-platform               | HuggingFace                                       |
|-------------------------------|--------------------------------|---------------------------------------------------|
| Public anonymous access       | No (internal)                  | Yes                                               |
| Works in OSS user's `cargo build` | Would require auth tokens   | Anonymous HTTP GET, no auth                       |
| CDN / global cache            | None                           | Built-in                                          |
| Matches upstream Pipecat path | No                             | Yes — same host, same URL shape as Pipecat        |
| Build-script complexity       | Auth, secrets, rate limits     | A single `ureq::get(url)` call (already in place) |
| Versioning / reproducibility  | Internal version IDs           | Git revisions on the model repo                   |

### What the workflow looks like

1. Train on `wavekat-lab`, push artifact to **wavekat-platform** (already done).
2. When a checkpoint is ready for public release, **export an ONNX snapshot
   to a HF model repo** under a WaveKat org (e.g. `wavekat/smart-turn-zh`).
3. `wavekat-turn`'s build script downloads from HF, the same way it does for
   Pipecat. The platform stays internal; HF carries the public bits only.

This keeps two clear roles:
- **Platform = training registry** (private, includes raw checkpoints, eval
  artifacts, experiments).
- **HF = release channel** (public, only the ONNX files we have decided to
  ship, tagged and immutable).

### Open questions before publishing to HF

- ~~HF org/account name.~~ **Resolved**: use the existing `wavekat` org (same
  as `wavekat-tts` models).
- **HF repo name (language-agnostic).** zh is just the first of many planned
  languages, so the repo name must not bake the language in. Two viable
  shapes:

  **A. One repo, language subdirs** *(recommended)*
  ```
  wavekat/smart-turn-ONNX
    ├── zh/smart-turn-cpu.onnx
    ├── ja/smart-turn-cpu.onnx        (future)
    ├── yue/smart-turn-cpu.onnx       (future)
    └── README.md
  ```
  Mirrors the TTS precedent of per-axis subdirs (`fp32/`, `int4/`).
  Adding a language later is a file push, not a new repo + new model card +
  new revision string.

  **B. Per-language repos with a stable parent pattern**
  ```
  wavekat/smart-turn-zh-ONNX        (this branch)
  wavekat/smart-turn-ja-ONNX        (future)
  ```
  Cleanest model card per language, but every new language is a new repo +
  new constants in `wavekat-turn`, and the repo name still encodes a
  language — exactly what we want to avoid.

  **Decision proposed: A.** Single repo `wavekat/smart-turn-ONNX` with
  `<lang>/` subdirs. Future expansion is additive and never requires a new
  HF repo.

- License. Pipecat upstream is BSD 2-Clause. Our fine-tunes inherit that
  unless we add separate ToS. Confirm we are comfortable publishing under
  BSD 2-Clause.
- Model card content: per-language sections (training data sources, eval
  numbers, intended use, known limitations — dialect coverage, SNR
  conditions). Keep a single top-level model card with a section per
  language.
- Revision convention. Pin a single dated `REVISION = "YYYY-MM-DD"` in
  `wavekat-turn` code, same as `wavekat-tts`. Updates to any language
  bump the same revision.

---

## Architecture: how to add the model to wavekat-turn

The Chinese model is **the same architecture** as upstream Pipecat — only the
weights differ. Mel feature extraction, ring-buffer logic, tensor shapes,
output interpretation, and the 0.5 threshold are all identical.

That means we have three real options for how the public API surfaces it.

### Option A — Variant on the existing `PipecatSmartTurn` struct *(recommended)*

Add a `Variant` enum and constructors that select which set of weights to load.
Inference code is unchanged.

```rust
/// Language for the WaveKat fine-tune. Extend as we ship more languages.
#[non_exhaustive]
pub enum SmartTurnLang {
    /// Mandarin Chinese (first WaveKat fine-tune).
    Zh,
    // Ja, Yue, ... (future)
}

#[non_exhaustive]
pub enum SmartTurnVariant {
    /// Upstream multilingual Pipecat Smart Turn v3.
    PipecatV3,
    /// WaveKat fine-tune for a specific language.
    Wavekat(SmartTurnLang),
}

impl PipecatSmartTurn {
    pub fn new() -> Result<Self, TurnError> {                // unchanged: PipecatV3
    pub fn with_variant(v: SmartTurnVariant) -> Result<Self, TurnError>;
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TurnError>;  // unchanged
}
```

`#[non_exhaustive]` on both enums is deliberate: adding a new language must
not be a breaking change.

The build/load layer resolves `Wavekat(lang)` to `<lang>/smart-turn-cpu.onnx`
inside the single `wavekat/smart-turn-ONNX` HF repo, so adding a new language
is a one-line variant addition + a file in the HF repo — no new constants,
no new feature flag.

**Pros**
- Zero code duplication; the feature is purely "different bytes".
- Honest naming: the *backend* is "Pipecat Smart Turn v3 architecture"; both
  models are instances of it.
- Users on a strict binary-size budget can disable one variant via features.

**Cons**
- `PipecatSmartTurn` is no longer a single-model thing; the type name suggests
  "Pipecat" even when running our weights. We can rename the struct to
  `SmartTurnDetector` and keep `PipecatSmartTurn` as a deprecated type alias.

### Option B — Separate `WavekatSmartTurnZh` struct

A new struct in `audio/smart_turn_zh.rs` that mostly re-exports the same mel
extractor and inference logic.

**Pros**
- Clearer in API docs: "for Chinese, use this struct".

**Cons**
- The mel extractor, ring buffer, and inference path would be copy-pasted or
  factored into a shared inner type — extra plumbing for no behavioral
  difference.
- Long-term, every additional fine-tune (Cantonese, Japanese, domain-specific)
  needs its own struct. Not scalable.

### Option C — No automatic download; rely on `from_file()` only

Publish to HF; expect users to download the file themselves and pass the path
to the existing `from_file()` constructor. Document the URL in the README.

**Pros**
- Zero changes to `wavekat-turn` code.
- Lowest friction to ship.

**Cons**
- Worse UX than the upstream Pipecat path, which is `new()` and "just works".
- Asymmetric: Pipecat users get build-time download, our own users don't.

**Recommendation: Option A.** Single backend type, two (eventually N) variants.
Same UX as upstream Pipecat. Option C is a reasonable v0 if we want to publish
to HF before doing any Rust work.

---

## Model loading strategy

This is the most significant new question now that we've seen the
`wavekat-tts` precedent. The two crates have diverged:

| Crate | Mechanism | Pros | Cons |
|-------|-----------|------|------|
| `wavekat-turn` (today) | `build.rs` downloads, `include_bytes!()` embeds | Zero runtime setup, model lives in the binary, offline-friendly after first build | Bloats binary per variant; no precision choice at runtime; build needs network unless `*_MODEL_PATH` is set |
| `wavekat-tts` (today)  | `hf-hub` runtime download to `$HF_HOME/hub/`, override with `WAVEKAT_MODEL_DIR` | Supports large models, runtime precision selection, no binary bloat, easy to update models without rebuilding | First-run network dependency; cache lives outside the build artefact |

The Chinese model is ~8 MB int8 — small enough to embed under the existing
**< 30 MB → embed** rule in [`02-plan-backends.md`](02-plan-backends.md). So
both options are technically viable.

### Option 1 — Keep embedding (consistent with `wavekat-turn` today)

Add the zh ONNX as a second `include_bytes!()` blob, downloaded by `build.rs`
under feature `pipecat-zh`. Identical pattern to upstream Pipecat.

**Pros**: zero new dependencies; consistent with the current backend; works
offline at runtime; reproducible via the existing version-marker caching.

**Cons**: ecosystem-inconsistent — a `wavekat-tts` user knows
`WAVEKAT_MODEL_DIR` and `~/.cache/huggingface/hub/`, but a `wavekat-turn` user
has to learn `PIPECAT_SMARTTURN_MODEL_PATH` and a build-time recompile to swap
weights.

### Option 2 — Switch to `hf-hub` runtime download (align with `wavekat-tts`)

Add `hf-hub` as a runtime dep gated on `pipecat-zh`. On first `new()` for the
zh variant, download `smart-turn-zh-cpu.onnx` to `$HF_HOME/hub/`. Honor
`WAVEKAT_MODEL_DIR` and `HF_TOKEN`.

**Pros**: unified ecosystem story across `wavekat-vad` / `wavekat-turn` /
`wavekat-tts`; trivially supports future fine-tunes (Cantonese, domain-specific
etc.) without re-publishing the crate; no binary bloat; users can swap model
revisions by setting `WAVEKAT_MODEL_DIR` without rebuilding.

**Cons**: divergence within `wavekat-turn` itself — upstream Pipecat stays
embedded, zh model downloads at runtime. Two mental models for the same crate.
And it introduces first-run network dependency for the zh variant.

### Option 3 — Switch both variants to `hf-hub` (full alignment)

Migrate the upstream Pipecat variant off `include_bytes!()` too, so the whole
crate uses `hf-hub` like `wavekat-tts`. Out of scope for this branch — would
need its own migration plan and a major-version bump.

### Recommendation

**Option 2** for this branch, with **Option 3 as a follow-up** in a separate
migration plan.

Reasoning:
- The ecosystem-consistency win is real: a user who already runs `wavekat-tts`
  doesn't have to learn a second set of env vars.
- The zh variant is the natural place to introduce `hf-hub` because it's
  greenfield — no existing users to migrate.
- Once `hf-hub` is in the dep tree under a feature, migrating the upstream
  Pipecat variant later is a localized change behind the same trait.
- We get **runtime precision selection** for free if/when we publish an fp16
  variant — no rebuild required.

If we're cautious about adding `hf-hub`, Option 1 is a perfectly fine
fallback. The variant API stays the same either way; only the body of
`with_variant(WavekatZh)` changes.

---

## Implementation plan (phased)

### Phase 0 — Publish to HuggingFace (out-of-repo)

1. ~~Decide HF org name and create the org if it does not exist.~~ Use
   existing `wavekat` org.
2. Create one language-agnostic model repo: **`wavekat/smart-turn-ONNX`**.
3. Write a model card. **Lead with attribution**:
   - First section: "WaveKat fine-tunes of [Pipecat Smart Turn v3](https://github.com/pipecat-ai/smart-turn)
     ([upstream HF](https://huggingface.co/pipecat-ai/smart-turn-v3),
     BSD 2-Clause)". State explicitly that the architecture, training recipe,
     and ONNX export contract are Pipecat's; WaveKat contributes
     language-specialized weights only.
   - Followed by a per-language section (data, eval, limitations).
   - Reproduce the BSD 2-Clause notice.
4. Export the ONNX from the wavekat-platform checkpoint we want to ship.
   **Compatibility checks before push** (block on these):
   - Tensor names match Pipecat: input `input_features` `[B, 80, 800]`
     float32, output `logits` `[B, 1]` float32 (sigmoid fused).
   - Loads in the upstream **Pipecat Python** inference pipeline with no
     code changes — just swap the model path. Capture a reference inference
     output for our fixture clips from Python.
   - Loads in our **Rust** pipeline via `from_file()` and matches the
     Python reference within the existing accuracy tolerance.
5. Push the ONNX to **`zh/smart-turn-cpu.onnx`** in the HF repo. Optionally
   add `zh/smart-turn-fp32.onnx` if/when we want to ship higher precision.
6. Sanity check:
   `curl -L https://huggingface.co/wavekat/smart-turn-ONNX/resolve/main/zh/smart-turn-cpu.onnx`
   returns the expected bytes anonymously.

Future-language workflow (e.g. Japanese): push `ja/smart-turn-cpu.onnx` to
the same repo, add a `Ja` variant to `SmartTurnLang`, ship a crate release.
No new HF repo. No new feature flag.

**Python usability note**: because the ONNX matches Pipecat's contract, a
Python user can consume it directly from the Pipecat `smart-turn` repo with
something like `SmartTurnAnalyzer(model_path=hf_hub_download("wavekat/smart-turn-ONNX", "zh/smart-turn-cpu.onnx"))`.
The HF repo README should include this one-liner so the audience is
explicitly "Pipecat users (Python or Rust) who want non-English support",
not just `wavekat-turn` users.

### Phase 1 — Add the variant to `wavekat-turn`

Assuming **Option 2** (hf-hub runtime loading) is chosen:

- Add `SmartTurnLang` and `SmartTurnVariant` enums (both `#[non_exhaustive]`).
  Default constructor `new()` keeps using `PipecatV3` for backwards compat.
- Add `PipecatSmartTurn::with_variant(variant)` constructor.
- Add `hf-hub` as an optional dep gated on the feature flag.
- New module `src/audio/wavekat_download.rs` mirroring `wavekat-tts`'
  `download.rs`:
  - `REPO_ID = "wavekat/smart-turn-ONNX"`, dated `REVISION`.
  - Map `SmartTurnLang::Zh → "zh/smart-turn-cpu.onnx"`. The path lookup is
    the single point that knows about languages — adding a language is one
    match arm.
  - Honor `WAVEKAT_TURN_MODEL_DIR` and `HF_TOKEN` exactly as TTS does for
    `WAVEKAT_MODEL_DIR`. Use a turn-specific name to avoid collision with
    the TTS env var.
  - Return a path that `onnx::session_from_file` consumes.
- In `with_variant(Wavekat(lang))`, call the download helper, then build a
  session from the resolved path. The Pipecat variant continues to use the
  embedded bytes path — no change.

Alternative (Option 1, kept for fallback):

- Extend `build.rs` with a download step per language and per-language
  `include_bytes!()` blobs. Each new language requires a recompile and a
  crate release — strictly worse for the multi-language future, but
  acceptable if we want to avoid the `hf-hub` dependency.

### Phase 2 — Cross-validation

- Add fixture clips in Mandarin (a "finished" clip, an "unfinished" clip,
  a silence/no-speech clip) under `tests/fixtures/`.
- Regenerate the Python reference (`scripts/gen_reference.py`) against the zh
  checkpoint and add `*.zh.mel.npy` / expected probabilities.
- Extend `tests/pipecat.rs` (or add `tests/smart_turn_zh.rs`) with the same
  9-test matrix from Phase 4 of `02-plan-backends.md`, plus parity checks
  against the Python reference.

### Phase 3 — README and example updates

- README: add a row to the Backends table, document the `pipecat-zh` feature,
  show a one-line example with `with_variant(SmartTurnVariant::WavekatZh)`.
- `examples/controller.rs`: optional second example with the zh model.
- Update `02-plan-backends.md` to reflect "model variants" as a concept.

### Phase 4 — Optional follow-ups

- ~~Consider renaming `PipecatSmartTurn` to `SmartTurnDetector`.~~ **Reversed**:
  keep `PipecatSmartTurn`. Pipecat owns the Smart Turn architecture; the
  type name correctly identifies what we are wrapping. Our weights are
  *instances* of Pipecat Smart Turn, not a separate detector.
- Decide whether `TurnController` should expose the variant in its API
  surface. Probably not: it is detector-agnostic by design.
- Once the HF repo exists, open a small PR / issue on
  [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn)
  pointing Python users at our weights for non-English support. Coordinate
  on whether they want to list the WaveKat repo from their README.

---

## Risks and tradeoffs

| Risk                                                          | Mitigation                                                                  |
|---------------------------------------------------------------|-----------------------------------------------------------------------------|
| Two embedded models double the crate's compiled size          | Feature-gate each variant; default features enable only `pipecat`.          |
| HF revision drift between platform and HF                     | Pin the revision in `build.rs` (not just the URL) — same pattern as today.  |
| Model card not ready for public release                       | Phase 0 gates the rest; do not start Phase 1 until the HF repo is signed off. |
| License compatibility (Pipecat is BSD 2-Clause)               | Confirm before publishing; include upstream attribution in the model card.  |
| Discoverability — users may not know there is a zh variant    | README table + variant docstring + a one-liner in the crate-level rustdoc.  |
| Pipecat reves the architecture (v4 with different tensor shape) | Frozen contract is documented in Phase 0. A breaking change upstream means a new HF repo family (e.g. `wavekat/smart-turn-v4-ONNX`), not mutating the existing one. |
| Our ONNX silently diverges from Pipecat's contract            | Phase 0 compatibility checks (Python pipeline + Rust pipeline) are gating. CI in the training repo should re-run them per checkpoint. |
| Python users can't easily find/use our weights                | Model card includes a Python one-liner; consider a PR to upstream pointing at the WaveKat repo. |

---

## Decisions to confirm before implementation

1. **Distribution channel:** HuggingFace `wavekat` org as the public mirror, platform stays internal? *(strongly recommended: yes — sibling `wavekat-tts` already does this)*
2. **HF repo name:** `wavekat/smart-turn-ONNX` (language-agnostic, with per-language subdirs like `zh/`) — recommended over `wavekat/smart-turn-zh-ONNX` because more languages are coming.
3. **License:** ship under BSD 2-Clause to match upstream Pipecat? *(default: yes)*
4. **API shape:** variant enum on `PipecatSmartTurn` (Option A), separate struct (Option B), or `from_file()`-only (Option C)? *(recommended: A)*
5. **Loading mechanism:** keep `build.rs` + `include_bytes!()` (Option 1), or switch to `hf-hub` runtime download (Option 2)? *(recommended: 2 — aligns with `wavekat-tts`)*
6. **Feature flag name:** language-agnostic, e.g. `wavekat-smart-turn` or `smart-turn-wavekat`, rather than `pipecat-zh` / `smart-turn-zh` (one flag gates *all* WaveKat fine-tunes, language is chosen at runtime via `SmartTurnLang`).
7. **Default features:** does the zh variant ship in default features, or stay opt-in? *(recommended: opt-in — keeps default install lean)*
8. **Env var name:** `WAVEKAT_TURN_MODEL_DIR` (recommended, turn-specific, no collision with `WAVEKAT_MODEL_DIR` from TTS). Applies to whichever language is selected.
9. ~~Rename `PipecatSmartTurn` → `SmartTurnDetector`?~~ **Resolved (keep `PipecatSmartTurn`)** — the architecture is Pipecat's; renaming would obscure that.
10. **Future migration:** do we want to plan now for moving the upstream Pipecat variant off `include_bytes!()` to `hf-hub` too (Option 3), or leave that for later?
11. **Coordination with Pipecat upstream:** do we want to proactively notify pipecat-ai/smart-turn maintainers (issue / PR linking our HF repo) so Python users discover the weights? *(recommended: yes, after Phase 0 ships)*
