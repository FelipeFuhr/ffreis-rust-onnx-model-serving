#!/usr/bin/env bash
set -euo pipefail

has_error=0
found_any=0
tmp_output="$(mktemp)"
trap 'rm -f "${tmp_output}"' EXIT

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This hook must run inside a Git repository." >&2
  exit 1
fi

while IFS= read -r -d '' file; do
  found_any=1
  if git show ":${file}" | \
      grep -nE '^(<<<<<<< |=======|>>>>>>> )' >"${tmp_output}" 2>/dev/null; then
    echo "Merge conflict markers detected in staged file: ${file}" >&2
    sed 's/^/  /' "${tmp_output}" >&2
    has_error=1
  fi
done < <(git diff --cached --name-only --diff-filter=ACM -z)

if [[ "$found_any" -eq 0 ]]; then
  exit 0
fi

if [[ "$has_error" -ne 0 ]]; then
  echo "Resolve conflict markers before committing." >&2
  exit 1
fi
