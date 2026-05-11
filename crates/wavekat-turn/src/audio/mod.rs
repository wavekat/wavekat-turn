//! Audio-based turn detection backends.
//!
//! These backends operate directly on raw audio frames and do not
//! require an upstream ASR transcript.
//!
//! [`PipecatSmartTurn`] is the entry point; [`SmartTurnVariant`] selects
//! which set of weights to load (upstream Pipecat vs WaveKat fine-tunes).
//! When the `wavekat-smart-turn` feature is enabled, [`SmartTurnLang`]
//! enumerates the language-specialized fine-tunes available on
//! HuggingFace.

#[cfg(feature = "pipecat")]
mod pipecat;

#[cfg(feature = "wavekat-smart-turn")]
pub(crate) mod wavekat_download;

#[cfg(feature = "pipecat")]
pub use pipecat::{PipecatSmartTurn, SmartTurnVariant};

#[cfg(feature = "wavekat-smart-turn")]
pub use pipecat::SmartTurnLang;
