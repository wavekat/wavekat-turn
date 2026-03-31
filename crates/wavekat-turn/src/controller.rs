use crate::{AudioFrame, AudioTurnDetector, TurnError, TurnPrediction, TurnState};

/// Orchestration wrapper around any [`AudioTurnDetector`].
///
/// Tracks prediction state across calls and provides convenience methods
/// like [`reset_if_finished`](TurnController::reset_if_finished) for
/// correct VAD integration without manual state bookkeeping.
///
/// # Usage
///
/// ```ignore
/// let detector = PipecatSmartTurn::new()?;
/// let mut ctrl = TurnController::new(detector);
///
/// // Audio arrives continuously
/// ctrl.push_audio(&frame);
///
/// // VAD speech start — soft reset (keeps buffer if turn was unfinished)
/// ctrl.reset_if_finished();
///
/// // VAD speech end — predict
/// let result = ctrl.predict()?;
/// ```
///
/// See [`reset_if_finished`](TurnController::reset_if_finished) for details
/// on when to use soft vs hard reset.
pub struct TurnController<T: AudioTurnDetector> {
    inner: T,
    last_state: Option<TurnState>,
}

impl<T: AudioTurnDetector> TurnController<T> {
    /// Create a new controller wrapping the given detector.
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            last_state: None,
        }
    }

    /// Feed audio into the detector.
    pub fn push_audio(&mut self, frame: &AudioFrame) {
        self.inner.push_audio(frame);
    }

    /// Run prediction on buffered audio.
    ///
    /// Tracks the result state internally for [`reset_if_finished`](Self::reset_if_finished).
    pub fn predict(&mut self) -> Result<TurnPrediction, TurnError> {
        let result = self.inner.predict()?;
        self.last_state = Some(result.state);
        Ok(result)
    }

    /// Hard reset — always clears the buffer.
    ///
    /// Use when you know a new turn is starting (e.g. after the assistant
    /// finishes responding).
    pub fn reset(&mut self) {
        self.inner.reset();
        self.last_state = None;
    }

    /// Soft reset — clears the buffer only if the last prediction was
    /// [`Finished`](TurnState::Finished) or no prediction has been made
    /// since the last reset.
    ///
    /// Returns `true` if a reset occurred, `false` if skipped.
    ///
    /// Call this on VAD speech-start when you don't know whether the user
    /// is continuing the same turn or starting a new one. If the previous
    /// prediction was [`Unfinished`](TurnState::Unfinished), the buffer is
    /// preserved so the next [`predict`](Self::predict) runs on the full
    /// accumulated audio.
    pub fn reset_if_finished(&mut self) -> bool {
        match self.last_state {
            Some(TurnState::Unfinished) => false,
            _ => {
                self.reset();
                true
            }
        }
    }

    /// Returns the state from the last [`predict`](Self::predict) call,
    /// or `None` if no prediction has been made since the last reset.
    pub fn last_state(&self) -> Option<TurnState> {
        self.last_state
    }

    /// Returns a mutable reference to the inner detector.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Unwrap the controller, returning the inner detector.
    pub fn into_inner(self) -> T {
        self.inner
    }
}
