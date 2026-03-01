#!/usr/bin/env sh
set -eu

LEFTHOOK_VERSION="${LEFTHOOK_VERSION:-1.7.10}"
BIN_DIR="${BIN_DIR:-.bin}"
BIN="$BIN_DIR/lefthook"

mkdir -p "$BIN_DIR"

if [ ! -x "$BIN" ]; then
  echo "Downloading lefthook v$LEFTHOOK_VERSION ..."

  OS="$(uname -s)"
  ARCH="$(uname -m)"

  case "$OS" in
    Linux)  OS=Linux ;;
    Darwin) OS=Darwin ;;
    *)
      echo "Unsupported OS: $OS" >&2
      exit 2
      ;;
  esac

  case "$ARCH" in
    x86_64|amd64) ARCH=x86_64 ;;
    aarch64|arm64) ARCH=arm64 ;;
    *)
      echo "Unsupported arch: $ARCH" >&2
      exit 2
      ;;
  esac

  URL="https://github.com/evilmartians/lefthook/releases/download/v${LEFTHOOK_VERSION}/lefthook_${LEFTHOOK_VERSION}_${OS}_${ARCH}"

  curl --fail --show-error --silent --location \
     --proto '=https' \
     --tlsv1.2 \
     "$URL" -o "$BIN"
  chmod +x "$BIN"
fi

echo "Lefthook available at: $BIN"
