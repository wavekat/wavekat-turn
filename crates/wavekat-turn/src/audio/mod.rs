//! Audio-based turn detection backends.
//!
//! These backends operate directly on raw audio frames and do not
//! require an upstream ASR transcript.

#[cfg(feature = "pipecat")]
mod pipecat;

#[cfg(feature = "pipecat")]
pub use pipecat::PipecatSmartTurn;
