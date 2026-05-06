.DEFAULT_GOAL := help

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

SHELL := /usr/bin/env bash

CARGO ?= cargo
CONTAINER_COMMAND ?= podman

LEFTHOOK_VERSION ?= 1.7.10
LEFTHOOK_DIR ?= $(CURDIR)/.bin
LEFTHOOK_BIN ?= $(LEFTHOOK_DIR)/lefthook

PREFIX ?= ffreis
IMAGE_PROVIDER ?=
IMAGE_TAG ?= api-grpc-smoke
SMOKE_TIMEOUT ?= 20m
BASE_DIR ?= .
CONTAINER_DIR ?= container

IMAGE_PREFIX := $(if $(IMAGE_PROVIDER),$(IMAGE_PROVIDER)/,)$(PREFIX)
IMAGE_ROOT := $(IMAGE_PREFIX)

# ------------------------------------------------------------------------------
# Image names
# ------------------------------------------------------------------------------

BASE_IMAGE := $(IMAGE_PREFIX)/base
BASE_BUILDER_IMAGE := $(IMAGE_PREFIX)/base-builder
BUILDER_IMAGE := $(IMAGE_PREFIX)/builder
BASE_RUNNER_IMAGE := $(IMAGE_PREFIX)/base-runner
RUNNER_IMAGE := $(IMAGE_PREFIX)/runner

# ------------------------------------------------------------------------------
# Derived values
# ------------------------------------------------------------------------------

# Extract app name from Cargo.toml (computed once)
APP_NAME := $(shell grep '^name' app/Cargo.toml | sed 's/name[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/')

# Extract digests from digests.env (computed once)
BASE_IMAGE_VALUE := $(shell grep '^BASE_IMAGE=' $(CONTAINER_DIR)/digests.env | cut -d= -f2)
BASE_DIGEST_VALUE := $(shell grep '^BASE_DIGEST=' $(CONTAINER_DIR)/digests.env | cut -d= -f2)

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ------------------------------------------------------------------------------
# Meta targets
# ------------------------------------------------------------------------------

.PHONY: all
all: lint build run coverage ## Lint, build, run, and generate coverage

# ------------------------------------------------------------------------------
# Tooling / setup
# ------------------------------------------------------------------------------

.PHONY: get-rust
get-rust: ## Download rustup installer script (used by container builds)
	curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o scripts/install-rust.sh

.PHONY: install-rust-local
install-rust-local: ## Install Rust locally if missing
	@if command -v cargo >/dev/null 2>&1; then \
		echo "cargo already installed: $$(command -v cargo)"; \
		exit 0; \
	fi
	curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y
	@if [ -f "$$HOME/.cargo/env" ]; then \
		. "$$HOME/.cargo/env"; \
	fi

.PHONY: install-podman-local
install-podman-local: ## Install Podman locally if missing
	@if command -v podman >/dev/null 2>&1; then \
		echo "podman already installed: $$(command -v podman)"; \
		exit 0; \
	fi
	sudo apt-get update
	sudo apt-get install -y podman

.PHONY: local-setup
local-setup: install-rust-local install-podman-local lefthook-install ## Install local dev prerequisites

# ------------------------------------------------------------------------------
# Container builds
# ------------------------------------------------------------------------------

.PHONY: build-base
build-base: ## Build base image (pinned by digest env)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base -t $(BASE_IMAGE) $(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE_VALUE)" \
		--build-arg BASE_DIGEST="$(BASE_DIGEST_VALUE)"

.PHONY: build-base-builder
build-base-builder: build-base get-rust ## Build base-builder image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-builder -t $(BASE_BUILDER_IMAGE) $(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE)"

.PHONY: build-builder
build-builder: build-base build-base-builder ## Build builder image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.builder -t $(BUILDER_IMAGE) $(BASE_DIR) \
		--build-arg BASE_BUILDER_IMAGE="$(BASE_BUILDER_IMAGE)"

.PHONY: build-base-runner
build-base-runner: build-base ## Build base-runner image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-runner -t $(BASE_RUNNER_IMAGE) $(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE)"

.PHONY: build-runner
build-runner: build-base-runner run-builder ## Build runner image (needs built binary)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.runner -t $(RUNNER_IMAGE) $(BASE_DIR) \
		--build-arg BASE_RUNNER_IMAGE="$(BASE_RUNNER_IMAGE)" \
		--build-arg APP_NAME="$(APP_NAME)"

.PHONY: build-images
build-images: get-rust build-base build-base-builder build-builder build-base-runner build-runner ## Build all images (may be slow)

.PHONY: run-builder
run-builder: build-builder ## Run builder container to produce release artifact
	$(CONTAINER_COMMAND) run --rm \
	--user "$(id -u):$(id -g)" \
	-e CARGO_TARGET_DIR=/build/target \
	-v "$(CURDIR)/build:/build/target" \
	-v "$(CURDIR)/app:/build" \
	$(BUILDER_IMAGE)

.PHONY: build
build: build-runner ## Build everything (images + app artifact + runner)

# ------------------------------------------------------------------------------
# App (local) targets
# ------------------------------------------------------------------------------

.PHONY: build-app-local
build-app-local: ## Build app locally (no containers)
	$(MAKE) -C app build

.PHONY: run-app
run-app: ## Run the runner container
	$(CONTAINER_COMMAND) run $(RUNNER_IMAGE)

.PHONY: run
run: run-app ## Alias: run the app

.PHONY: fmt
fmt: ## Format Rust code
	$(MAKE) -C app fmt

.PHONY: fmt-check
fmt-check: ## Check Rust formatting
	$(MAKE) -C app fmt-check

.PHONY: clippy
clippy: ## Run clippy lints
	$(MAKE) -C app clippy

.PHONY: test
test: ## Run tests
	$(MAKE) -C app test

.PHONY: test-property
test-property: ## Run property-based tests (proptest)
	$(MAKE) -C app test-property

.PHONY: grpc-check
grpc-check: ## Verify protobuf/gRPC contract compiles
	$(MAKE) -C app grpc-check

.PHONY: openapi-check
openapi-check: ## Validate OpenAPI contract
	uv run --with openapi-spec-validator --with pyyaml python scripts/check_openapi.py

.PHONY: openapi-drift-check
openapi-drift-check: ## Ensure API changes are accompanied by OpenAPI updates
	@test -n "$(BASE_SHA)" || (echo "BASE_SHA is required" && exit 1)
	@test -n "$(HEAD_SHA)" || (echo "HEAD_SHA is required" && exit 1)
	python3 scripts/check_openapi_drift.py --base "$(BASE_SHA)" --head "$(HEAD_SHA)"

.PHONY: test-grpc-parity
test-grpc-parity: ## Run HTTP/gRPC parity tests
	$(MAKE) -C app test-grpc-parity

.PHONY: smoke-api-grpc
smoke-api-grpc: ## Run docker-compose HTTP + gRPC smoke test
	@set -euo pipefail; \
	cleanup() { \
		IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" $(CONTAINER_COMMAND) compose -f examples/docker-compose.api-grpc.yml down --remove-orphans || true; \
	}; \
	trap cleanup EXIT; \
	IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" timeout --foreground "$(SMOKE_TIMEOUT)" $(CONTAINER_COMMAND) compose -f examples/docker-compose.api-grpc.yml up --build --abort-on-container-exit --exit-code-from smoke

.PHONY: lint
lint: fmt-check clippy ## Run formatting check + clippy

# ------------------------------------------------------------------------------
# Coverage
# ------------------------------------------------------------------------------

.PHONY: coverage
coverage: ## Generate coverage report (Cobertura XML) into ./coverage/
	$(MAKE) -C app coverage

.PHONY: coverage-check
coverage-check: ## Fail if coverage is below COVERAGE_MIN
	$(MAKE) -C app coverage-check

# ------------------------------------------------------------------------------
# Lefthook
# ------------------------------------------------------------------------------

.PHONY: lefthook-bootstrap
lefthook-bootstrap: ## Download lefthook binary into ./.bin
	LEFTHOOK_VERSION="$(LEFTHOOK_VERSION)" BIN_DIR="$(LEFTHOOK_DIR)" ./scripts/bootstrap_lefthook.sh

.PHONY: lefthook-install
lefthook-install: lefthook-bootstrap ## Install git hooks if missing
	@if [ -x "$(LEFTHOOK_BIN)" ] && [ -x ".git/hooks/pre-commit" ] && [ -x ".git/hooks/pre-push" ]; then \
		echo "lefthook hooks already installed"; \
		exit 0; \
	fi
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" install

.PHONY: lefthook-run
lefthook-run: lefthook-bootstrap ## Run hooks (pre-commit + pre-push)
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" run pre-commit
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" run pre-push

.PHONY: lefthook
lefthook: lefthook-bootstrap lefthook-install lefthook-run ## Install hooks and run them

.PHONY: install-lefthook-local
install-lefthook-local: ## Print path to lefthook binary (for debugging)
	@if command -v lefthook >/dev/null 2>&1; then \
		echo "lefthook already installed: $$(command -v lefthook)"; \
		exit 0; \
	fi
	curl -1sLf 'https://dl.cloudsmith.io/public/evilmartians/lefthook/setup.deb.sh' | sudo -E bash
	sudo apt install lefthook

lefthook-local: install-lefthook-local lefthook-install lefthook-run

# ------------------------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------------------------

.PHONY: clean-app
clean-app: ## Clean Rust build artifacts
	$(MAKE) -C app clean

.PHONY: clean-repo
clean-repo: clean-app ## Clean repo build outputs
	rm -rf build coverage
	rm -f scripts/install-rust.sh

.PHONY: clean-base
clean-base: ## Remove base image
	$(CONTAINER_COMMAND) rmi $(BASE_IMAGE) || true

.PHONY: clean-base-builder
clean-base-builder: ## Remove base-builder image
	$(CONTAINER_COMMAND) rmi $(BASE_BUILDER_IMAGE) || true

.PHONY: clean-builder
clean-builder: ## Remove builder image
	$(CONTAINER_COMMAND) rmi $(BUILDER_IMAGE) || true

.PHONY: clean-base-runner
clean-base-runner: ## Remove base-runner image
	$(CONTAINER_COMMAND) rmi $(BASE_RUNNER_IMAGE) || true

.PHONY: clean-runner
clean-runner: ## Remove runner image
	$(CONTAINER_COMMAND) rmi $(RUNNER_IMAGE) || true

.PHONY: clean-all
clean-all: clean-repo clean-base clean-base-builder clean-builder clean-base-runner clean-runner ## Clean everything

# ------------------------------------------------------------------------------
# Security
# ------------------------------------------------------------------------------

GITLEAKS ?= gitleaks

.PHONY: secrets-scan-staged
secrets-scan-staged: ## Scan staged diff for secrets
	@command -v $(GITLEAKS) >/dev/null 2>&1 || (echo "Missing tool: $(GITLEAKS). Install: https://github.com/gitleaks/gitleaks#installing" && exit 1)
	$(GITLEAKS) protect --staged --redact
