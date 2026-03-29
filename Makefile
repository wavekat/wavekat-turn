.PHONY: help check test fmt lint doc ci accuracy mel

help:
	@echo "Available targets:"
	@echo "  check     Check workspace compiles"
	@echo "  test      Run all tests"
	@echo "  accuracy  Cross-validate Rust pipeline against Python reference"
	@echo "  mel       Compare Rust vs Python mel spectrograms element-wise"
	@echo "  fmt       Format code"
	@echo "  lint      Run clippy with warnings as errors"
	@echo "  doc       Build and open docs in browser"
	@echo "  ci        Run all CI checks locally (fmt, clippy, test, doc, features)"

# Check workspace compiles
check:
	cargo check --workspace

# Run all tests
test:
	cargo test --workspace

# Cross-validate Rust mel+ONNX pipeline against Python reference probabilities
accuracy:
	cargo test --features pipecat --test accuracy -- --ignored accuracy_report --nocapture

# Compare Rust vs Python mel spectrograms element-wise (requires .npy fixtures)
mel:
	cargo test --features pipecat -- mel_report --ignored --nocapture

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
