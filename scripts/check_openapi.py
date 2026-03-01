#!/usr/bin/env python3
"""Validate checked-in OpenAPI contract."""

from __future__ import annotations

from pathlib import Path

import yaml
from openapi_spec_validator import validate_spec


def _load_spec(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Expected OpenAPI mapping at {path}")
    return loaded


def _assert_unique_operation_ids(spec: dict[str, object]) -> None:
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        raise RuntimeError("OpenAPI 'paths' must be an object")
    seen: set[str] = set()
    for path_item in paths.values():
        if not isinstance(path_item, dict):
            continue
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            operation_id = operation.get("operationId")
            if isinstance(operation_id, str):
                if operation_id in seen:
                    raise RuntimeError(f"Duplicate operationId: {operation_id}")
                seen.add(operation_id)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec_path = repo_root / "docs" / "openapi.yaml"
    spec = _load_spec(spec_path)
    validate_spec(spec)
    _assert_unique_operation_ids(spec)
    print("OpenAPI contract is valid.")


if __name__ == "__main__":
    main()

