# Datasets — Mandarin Conversational Audio for Turn Detection

Survey of candidate corpora for training a Mandarin Smart Turn detector.
Sources reviewed: [OpenSLR](http://www.openslr.org/resources.php) (full list,
SLR1–SLR162) and HuggingFace.

## What we need

Spontaneous conversational audio with **per-speaker turn boundaries**:

- Real conversation — readers don't yield turns naturally; read-speech corpora
  cannot supervise turn detection as a primary signal.
- Turn timestamps, or per-speaker channels we can VAD ourselves.
- Permissive license — we want commercial-friendly model weights.

## Recommended basket (Mandarin, commercial-OK)

About 338 h of pre-annotated Mandarin conversation across three corpora.

| SLR | Name | Hours | Per-speaker channels | License | Why |
|---|---|---|---|---|---|
| [119](http://www.openslr.org/119/) | AliMeeting | 118 | yes (headset) | CC BY-SA 4.0 | 2–4 spk meetings, clean per-speaker tracks, diarization ready. Best fit. |
| [159](http://www.openslr.org/159/) | AISHELL-5 | 100 | partial (training-set near-field) | CC BY-SA 4.0 | In-car free conversation, 2–4 spk. Closest to voice-agent setting. |
| [111](http://www.openslr.org/111/) | AISHELL-4 | 120 | no (8-ch array only) | CC BY-SA 4.0 | 4–8 spk meetings, more overlap; needs careful filtering. |

For scale once the base pipeline works:

| SLR | Name | Hours | License | Catch |
|---|---|---|---|---|
| [121](http://www.openslr.org/121/) | WenetSpeech | 10,000+ | CC BY 4.0 | Multi-domain (podcasts, interviews, audiobooks). Not pre-segmented — needs diarization (e.g. pyannote-audio). |

## Reserve / research-only

| SLR | Name | Hours | License | Why on hold |
|---|---|---|---|---|
| [123](http://www.openslr.org/123/) | MAGICDATA Conversational | 180 | **CC BY-NC-ND 4.0** | Non-commercial only. Use only if scope stays research. |
| [155](http://www.openslr.org/155/) | SBCSAE | 20 | **CC BY-ND 3.0** | No-derivatives blocks publishing processed clips. Small. American English. |

## Optional English coverage (multilingual model)

| SLR | Name | Hours | License | Notes |
|---|---|---|---|---|
| [16](http://www.openslr.org/16/) | AMI Corpus | ~100 | CC BY 4.0 | Classic meeting corpus, headset + array. |
| [150](http://www.openslr.org/150/) | CHiME-6 | ~50 | CC BY-SA 4.0 | Dinner-party recordings, JSON annotations. |

## Not useful for the primary turn signal

**Read speech** (no natural turn yielding): SLR18 THCHS-30, SLR33 AISHELL-1,
SLR38 Free ST, SLR47 Primewords, SLR62 aidatatang_200zh, SLR68 MAGICDATA Read
755h, SLR93 AISHELL-3, SLR138 SHALCAS22A. Could be reused as text/voice
sources for synthetic-truncation augmentation if we go that route.

**Other excluded** (full sweep): software mirrors (SLR2/3/4/9/11/15/23/48/50/56),
impulse-response and noise databases (SLR13/17/20/26/28), wake-word and
hotword (SLR85/87/120), speaker verification (SLR82 CN-Celeb, SLR156),
pronunciation dictionaries (SLR8/14/21/29/34), text-only corpora (SLR55/153),
emotional read-speech TTS (SLR88/110/115/136/161), scripture (SLR129/132),
handwriting (SLR84), nonverbal vocalizations (SLR99), test fixtures
(SLR1/81), forensics (SLR162), whistled language (SLR137), single-speaker TTS
data for low-resource languages (most of the "Crowdsourced high-quality"
series, Thorsten Müller, Sinhala TTS, etc.), and endangered-language
documentation (SLR89/92/107/124/133/147/148/158/149).

## License notes

**CC BY-SA 4.0 share-alike** applies to *derived datasets* — if we publish
processed clips, the clip dataset itself must be CC BY-SA. Trained model
weights are generally not treated as derivatives of training data in most
jurisdictions, so a permissive weight release is usually fine, but worth
confirming with whoever handles licensing if we plan to ship weights.

**CC BY-NC-ND** (SLR123) blocks both commercial use and derivative datasets —
incompatible with any pipeline that publishes processed clips.

## Open questions

- WenetSpeech subsets: which of its domains (podcast / interview / meeting /
  audiobook / vlog) are spontaneous enough to be worth diarizing? Need to
  inspect the metadata before committing GPU time.
- Whether to include SLR123 (RAMC, 180h) under a research-only carve-out —
  doubles available Mandarin conversation hours but adds license-tracking
  burden across artifacts.
- Multilingual scope: is English (AMI + CHiME-6) in scope for v1, or
  Mandarin-only?
