.PHONY: help check test fmt lint doc ci

help:
	@echo "Available targets:"
	@echo "  check   Check workspace compiles"
	@echo "  test    Run all tests"
	@echo "  fmt     Format code"
	@echo "  lint    Run clippy with warnings as errors"
	@echo "  doc     Build and open docs in browser"
	@echo "  ci      Run all CI checks locally (fmt, clippy, test, doc, features)"

# Check workspace compiles
check:
	cargo check --workspace

# Run all tests
test:
	cargo test --workspace

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
