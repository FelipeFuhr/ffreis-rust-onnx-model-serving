#!/usr/bin/env bash
set -euo pipefail

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

while IFS= read -r -d '' entry; do
  IFS=$'\t' read -r added deleted file <<<"$entry"

  if [[ "$added" != "-" || "$deleted" != "-" ]]; then
    continue
  fi

  if is_allowlisted "$file"; then
    continue
  fi

  echo "Unexpected staged binary file: ${file}" >&2
  has_error=1
done < <(git diff --cached --numstat --diff-filter=ACM -z)

if [[ "$has_error" -ne 0 ]]; then
  echo "Binary files are blocked unless they match allowlisted paths/extensions." >&2
  exit 1
fi
