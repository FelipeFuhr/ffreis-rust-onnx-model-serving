.DEFAULT_GOAL := help

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

SHELL := /usr/bin/env bash

CARGO ?= cargo
PYTHON ?= python3

COVERAGE_MIN ?= 90
ROOT_DIR ?= $(abspath ..)
COVERAGE_XML ?= $(ROOT_DIR)/coverage/cobertura.xml
CHECK_COVERAGE_SCRIPT ?= $(ROOT_DIR)/scripts/check_coverage.py
COVERAGE_DIR := $(dir $(COVERAGE_XML))

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ------------------------------------------------------------------------------
# Rust quality
# ------------------------------------------------------------------------------

.PHONY: fmt
fmt: ## Format Rust code
	$(CARGO) fmt --all

.PHONY: fmt-check
fmt-check: ## Check Rust formatting
	$(CARGO) fmt --all -- --check

.PHONY: clippy
clippy: ## Run clippy lints
	$(CARGO) clippy --all-targets --all-features -- -D warnings

.PHONY: lint
lint: fmt-check clippy ## Run format check and clippy

# ------------------------------------------------------------------------------
# gRPC + parity
# ------------------------------------------------------------------------------

.PHONY: grpc-check
grpc-check: ## Verify protobuf/gRPC contract compiles
	$(CARGO) check --verbose

.PHONY: test-grpc-parity
test-grpc-parity: ## Run HTTP/gRPC parity integration tests
	$(CARGO) test --test integration_tests --verbose

# ------------------------------------------------------------------------------
# Build & test
# ------------------------------------------------------------------------------

.PHONY: build
build: ## Build the app
	$(CARGO) build --verbose

.PHONY: test
test: ## Run tests
	$(CARGO) test --verbose

.PHONY: test-unit
test-unit: ## Run unit tests only (src/*)
	$(CARGO) test --lib --verbose

.PHONY: test-integration
test-integration: ## Run integration tests only (tests/integration/*)
	$(CARGO) test --test integration_tests --verbose

.PHONY: test-property
test-property: ## Run property-based tests (proptest) in unit test module
	@if [ -n "$(PROPTEST_CASES)" ]; then \
		PROPTEST_CASES="$(PROPTEST_CASES)" $(CARGO) test --lib property_ --verbose; \
	else \
		$(CARGO) test --lib property_ --verbose; \
	fi

.PHONY: test-e2e
test-e2e: ## Run e2e/ML scaffold tests (ignored tests are not run by default)
	$(CARGO) test --test e2e_tests --verbose

.PHONY: test-e2e-ignored
test-e2e-ignored: ## Run ignored e2e/ML tests (requires external fixtures)
	$(CARGO) test --test e2e_tests -- --ignored --nocapture

.PHONY: clean
clean: ## Clean Rust build artifacts
	$(CARGO) clean

# ------------------------------------------------------------------------------
# Coverage
# ------------------------------------------------------------------------------

.PHONY: coverage
coverage: ## Generate coverage report (Cobertura XML)
	command -v cargo-llvm-cov >/dev/null 2>&1 || { \
		echo "cargo-llvm-cov not installed. Install with: cargo install cargo-llvm-cov --locked"; \
		exit 1; \
	}
	mkdir -p "$(COVERAGE_DIR)"
	$(CARGO) llvm-cov --all-features --workspace --all-targets --cobertura --output-path "$(COVERAGE_XML)"

.PHONY: coverage-check
coverage-check: ## Fail if coverage is below COVERAGE_MIN
	command -v cargo-llvm-cov >/dev/null 2>&1 || { \
		echo "cargo-llvm-cov not installed. Install with: cargo install cargo-llvm-cov --locked"; \
		exit 1; \
	}
	$(MAKE) coverage
	COVERAGE_MIN="$(COVERAGE_MIN)" COVERAGE_XML="$(COVERAGE_XML)" \
		"$(PYTHON)" "$(CHECK_COVERAGE_SCRIPT)"
