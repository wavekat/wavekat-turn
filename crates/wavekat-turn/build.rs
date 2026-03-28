//! Build script for wavekat-turn.
//!
//! Downloads ONNX model(s) at build time when the corresponding feature flag
//! is enabled. Models are written to `OUT_DIR` and embedded via `include_bytes!`.
//!
//! # Environment variables
//!
//! | Variable                         | Effect                                     |
//! |----------------------------------|--------------------------------------------|
//! | `PIPECAT_SMARTTURN_MODEL_PATH`   | Use this local file instead of downloading |
//! | `PIPECAT_SMARTTURN_MODEL_URL`    | Override the download URL                  |
//! | `DOCS_RS`                        | Write zero-byte placeholder (no network)   |

#[allow(unused_imports)]
use std::env;
#[allow(unused_imports)]
use std::fs;
#[allow(unused_imports)]
use std::path::Path;

fn main() {
    // docs.rs builds with --network none; write empty placeholders so
    // include_bytes! compiles without downloading anything.
    if env::var("DOCS_RS").is_err() {
        #[cfg(feature = "pipecat")]
        setup_pipecat_model();
    } else {
        #[cfg(feature = "pipecat")]
        {
            let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
            let model_path = Path::new(&out_dir).join("smart-turn-v3.2-cpu.onnx");
            if !model_path.exists() {
                fs::write(&model_path, b"").expect("failed to write placeholder model");
            }
        }
    }
}

#[cfg(feature = "pipecat")]
fn setup_pipecat_model() {
    const DEFAULT_MODEL_URL: &str =
        "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx";
    const MODEL_FILE: &str = "smart-turn-v3.2-cpu.onnx";
    // Bump this string when updating the default model URL so cached builds
    // re-download the new version.
    const MODEL_VERSION: &str = "v3.2-cpu";

    println!("cargo:rerun-if-env-changed=PIPECAT_SMARTTURN_MODEL_PATH");
    println!("cargo:rerun-if-env-changed=PIPECAT_SMARTTURN_MODEL_URL");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let model_dest = Path::new(&out_dir).join(MODEL_FILE);
    let version_marker = Path::new(&out_dir).join("smart-turn.version");

    // Option 1: caller provides a local model file
    if let Ok(local_path) = env::var("PIPECAT_SMARTTURN_MODEL_PATH") {
        let local_path = Path::new(&local_path);
        if !local_path.exists() {
            panic!(
                "PIPECAT_SMARTTURN_MODEL_PATH points to a non-existent file: {}",
                local_path.display()
            );
        }
        println!(
            "cargo:warning=Using local Pipecat Smart Turn model: {}",
            local_path.display()
        );
        fs::copy(local_path, &model_dest).expect("failed to copy local model file");
        println!("cargo:rerun-if-changed={}", local_path.display());
        return;
    }

    // Skip download if a matching version is already cached
    let cached_version = fs::read_to_string(&version_marker).unwrap_or_default();
    if model_dest.exists() && cached_version.trim() == MODEL_VERSION {
        return;
    }

    // Option 2: download (caller may override the URL)
    let url =
        env::var("PIPECAT_SMARTTURN_MODEL_URL").unwrap_or_else(|_| DEFAULT_MODEL_URL.to_string());

    println!("cargo:warning=Downloading Pipecat Smart Turn model ({MODEL_VERSION}) from {url}");

    let response = ureq::get(&url)
        .call()
        .unwrap_or_else(|e| panic!("failed to download Pipecat Smart Turn model from {url}: {e}"));

    let bytes = response
        .into_body()
        .read_to_vec()
        .expect("failed to read model bytes");

    fs::write(&model_dest, &bytes).expect("failed to write model file");
    fs::write(&version_marker, MODEL_VERSION).expect("failed to write version marker");

    println!(
        "cargo:warning=Pipecat Smart Turn model ({MODEL_VERSION}) downloaded to {}",
        model_dest.display()
    );
}
