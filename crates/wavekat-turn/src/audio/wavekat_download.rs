//! Runtime download of WaveKat-trained Smart Turn weights from HuggingFace.
//!
//! Mirrors the `wavekat-tts` pattern: one language-agnostic HF repo with
//! per-language subdirectories, a dated `REVISION` pinned in code so that
//! model updates ship via a crate release, and a `WAVEKAT_TURN_MODEL_DIR`
//! escape hatch for offline / CI builds.

use std::path::PathBuf;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};

use super::pipecat::SmartTurnLang;
use crate::error::TurnError;

/// HuggingFace repo holding all WaveKat Smart Turn fine-tunes.
const REPO_ID: &str = "wavekat/smart-turn-ONNX";

/// Pinned model revision. Bumping this string is the way to ship updated
/// weights to consumers — same pattern as `wavekat-tts`.
const REVISION: &str = "main";

/// Env var that lets callers point at a local directory containing
/// `<lang>/smart-turn-cpu.onnx`, skipping the HuggingFace download entirely.
const LOCAL_DIR_ENV: &str = "WAVEKAT_TURN_MODEL_DIR";

/// Map a language to its file path inside the HF repo.
fn relative_path(lang: SmartTurnLang) -> &'static str {
    match lang {
        SmartTurnLang::Zh => "zh/smart-turn-cpu.onnx",
    }
}

/// Resolve the on-disk path for `lang`, downloading from HuggingFace if needed.
pub(crate) fn resolve_model(lang: SmartTurnLang) -> Result<PathBuf, TurnError> {
    let rel = relative_path(lang);

    if let Some(dir) = std::env::var_os(LOCAL_DIR_ENV) {
        let candidate = PathBuf::from(dir).join(rel);
        if !candidate.exists() {
            return Err(TurnError::ModelNotLoaded(format!(
                "{LOCAL_DIR_ENV} is set but {} does not exist",
                candidate.display()
            )));
        }
        return Ok(candidate);
    }

    let api = ApiBuilder::new()
        .with_token(std::env::var("HF_TOKEN").ok())
        .build()
        .map_err(|e| TurnError::BackendError(format!("failed to build hf-hub client: {e}")))?;

    let repo = api.repo(Repo::with_revision(
        REPO_ID.to_string(),
        RepoType::Model,
        REVISION.to_string(),
    ));

    repo.get(rel).map_err(|e| {
        TurnError::BackendError(format!(
            "failed to download {REPO_ID}@{REVISION}:{rel} from HuggingFace: {e}. \
             Set {LOCAL_DIR_ENV} to a directory containing {rel} to skip the download."
        ))
    })
}
