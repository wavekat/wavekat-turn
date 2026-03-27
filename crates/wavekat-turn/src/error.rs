use thiserror::Error;

/// Errors that can occur during turn detection.
#[derive(Debug, Error)]
pub enum TurnError {
    #[error("backend error: {0}")]
    BackendError(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("model not loaded: {0}")]
    ModelNotLoaded(String),
}
