//! Text-based turn detection backends.
//!
//! These backends operate on ASR transcript text and optionally
//! use conversation history for context-aware predictions.

#[cfg(feature = "livekit")]
mod livekit;

#[cfg(feature = "livekit")]
pub use livekit::LiveKitEou;
