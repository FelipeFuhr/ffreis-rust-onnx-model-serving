#!/usr/bin/env bash
set -euo pipefail

MAX_SIZE_BYTES="${MAX_SIZE_BYTES:-1048576}"
has_error=0

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This hook must run inside a Git repository." >&2
  exit 1
fi

is_allowlisted() {
  local path="$1"
  local ext="${path##*.}"
  ext="${ext,,}"

  case "$path" in
    testdata/*|*/testdata/*|examples/*|*/examples/*)
      return 0
      ;;
  esac

  case "$ext" in
    png|jpg|jpeg|svg|webp|ico|woff|woff2|ttf)
      return 0
      ;;
  esac

  return 1
}

while IFS= read -r -d '' file; do
  size="$(git cat-file -s ":${file}")"
  if [[ -z "$size" ]]; then
    continue
  fi

  if [[ "$size" -le "$MAX_SIZE_BYTES" ]]; then
    continue
  fi

  if is_allowlisted "$file"; then
    continue
  fi

  echo "Staged file exceeds ${MAX_SIZE_BYTES} bytes: ${file} (${size} bytes)" >&2
  has_error=1
done < <(git diff --cached --name-only --diff-filter=ACM -z)

if [[ "$has_error" -ne 0 ]]; then
  echo "Large staged files must be split, compressed, or explicitly allowlisted." >&2
  exit 1
fi
