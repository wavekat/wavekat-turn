.PHONY: help check test fmt lint doc ci accuracy mel hf-smoke example-controller

help:
	@echo "Available targets:"
	@echo "  check     Check workspace compiles"
	@echo "  test      Run all tests"
	@echo "  accuracy  Cross-validate Rust pipeline against Python reference"
	@echo "  mel       Compare Rust vs Python mel spectrograms element-wise"
	@echo "  hf-smoke  Download wavekat/smart-turn-ONNX from HF and run zh fixtures"
	@echo "  fmt       Format code"
	@echo "  lint      Run clippy with warnings as errors"
	@echo "  doc       Build and open docs in browser"
	@echo "  ci        Run all CI checks locally (fmt, clippy, test, doc, features)"
	@echo "  example-controller  Run TurnController example"

# Check workspace compiles
check:
	cargo check --workspace

# Run all tests
test:
	cargo test --workspace

# Cross-validate Rust mel+ONNX pipeline against Python reference probabilities.
# Builds with `wavekat-smart-turn` so the zh fine-tune rows are also emitted;
# WaveKat weights are fetched from HuggingFace on first run (cached in $HF_HOME).
accuracy:
	cargo test --features wavekat-smart-turn --test accuracy -- --ignored accuracy_report --nocapture

# Compare Rust vs Python mel spectrograms element-wise (requires .npy fixtures)
mel:
	cargo test --features pipecat -- mel_report --ignored --nocapture

# Download wavekat/smart-turn-ONNX from HuggingFace and assert the zh fine-tune
# correctly classifies the Mandarin fixtures. Requires network on first run;
# subsequent runs hit the HF cache under $HF_HOME/hub/.
hf-smoke:
	cargo test --features wavekat-smart-turn --test pipecat \
	    -- --ignored wavekat_hf_download_smoke --nocapture

# Run TurnController example
example-controller:
	cargo run --features pipecat --example controller

# Format code
fmt:
	cargo fmt --all

# Lint
lint:
	cargo clippy --workspace -- -D warnings

# Build and open docs in browser
doc:
	cargo doc --no-deps -p wavekat-turn --all-features --open

# Run all CI checks locally (mirrors .github/workflows/ci.yml)
ci:
	cargo fmt --all -- --check
	cargo clippy --workspace -- -D warnings
	cargo test --workspace
	cargo doc --no-deps -p wavekat-turn --all-features
	cargo test -p wavekat-turn --no-default-features --features ""
	cargo test -p wavekat-turn --no-default-features --features "pipecat"
	cargo test -p wavekat-turn --no-default-features --features "livekit"
	cargo test -p wavekat-turn --no-default-features --features "pipecat,livekit"
