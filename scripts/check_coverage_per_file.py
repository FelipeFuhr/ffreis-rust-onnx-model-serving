import os
import sys
from defusedxml import ElementTree as ET


def _normalize(path: str) -> str:
    return path.replace("\\", "/").strip()


def _target_variants(target: str) -> set[str]:
    normalized = _normalize(target).lstrip("./")
    variants = {normalized}
    if normalized.startswith("app/"):
        variants.add(normalized[len("app/") :])
    return variants


def _matches_target(filename: str, target: str) -> bool:
    file_norm = _normalize(filename)
    for variant in _target_variants(target):
        if file_norm == variant or file_norm.endswith(f"/{variant}"):
            return True
    return False


def _parse_targets(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def main() -> int:
    min_cov = float(os.environ.get("COVERAGE_PER_FILE_MIN", "90"))
    path = os.environ.get("COVERAGE_XML", "../coverage/cobertura.xml")
    targets_raw = os.environ.get(
        "COVERAGE_FILES",
        "app/src/main.rs,app/src/lib.rs,app/src/config.rs,app/src/telemetry.rs",
    )
    targets = _parse_targets(targets_raw)
    if not targets:
        print("COVERAGE_FILES is empty; nothing to check.", file=sys.stderr)
        return 2

    try:
        tree = ET.parse(path)
    except FileNotFoundError:
        print(f"Coverage report not found: {path}", file=sys.stderr)
        return 2
    except ET.ParseError as exc:
        print(f"Could not parse coverage XML: {path}: {exc}", file=sys.stderr)
        return 2

    root = tree.getroot()
    class_nodes = root.findall(".//class")
    if not class_nodes:
        print("Coverage XML has no <class> entries.", file=sys.stderr)
        return 2

    stats: dict[str, dict[str, int]] = {
        target: {"covered": 0, "total": 0} for target in targets
    }

    for class_node in class_nodes:
        filename = class_node.attrib.get("filename", "")
        if not filename:
            continue
        matched_targets = [target for target in targets if _matches_target(filename, target)]
        if not matched_targets:
            continue
        for line in class_node.findall("./lines/line"):
            hits_raw = line.attrib.get("hits")
            if hits_raw is None:
                continue
            try:
                hits = int(hits_raw)
            except ValueError:
                continue
            for target in matched_targets:
                stats[target]["total"] += 1
                if hits > 0:
                    stats[target]["covered"] += 1

    missing_targets = [target for target, data in stats.items() if data["total"] == 0]
    if missing_targets:
        print(
            "Coverage XML missing requested file(s): "
            + ", ".join(sorted(missing_targets)),
            file=sys.stderr,
        )
        return 2

    failed = False
    for target in targets:
        covered = stats[target]["covered"]
        total = stats[target]["total"]
        rate = (covered / total) * 100.0
        if rate + 1e-9 < min_cov:
            print(
                f"{target}: {rate:.2f}% is below minimum {min_cov:.2f}% ({covered}/{total})",
                file=sys.stderr,
            )
            failed = True
        else:
            print(f"{target}: {rate:.2f}% ({covered}/{total})")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
