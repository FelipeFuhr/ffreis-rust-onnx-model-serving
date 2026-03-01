import os
import sys
from defusedxml import ElementTree as ET

min_cov = float(os.environ.get("COVERAGE_MIN", "90"))
max_drift_raw = os.environ.get("COVERAGE_MAX_DRIFT", "").strip()
max_drift = float(max_drift_raw) if max_drift_raw else None
path = os.environ.get("COVERAGE_XML", "../coverage/cobertura.xml")
label = os.environ.get("COVERAGE_LABEL", "coverage")

try:
    tree = ET.parse(path)
except FileNotFoundError:
    print(f"Coverage report not found: {path}", file=sys.stderr)
    raise SystemExit(2)
except ET.ParseError as e:
    print(f"Could not parse coverage XML: {path}: {e}", file=sys.stderr)
    raise SystemExit(2)

root = tree.getroot()
rate_attr = root.attrib.get("line-rate")
if rate_attr is None:
    print("Coverage XML missing 'line-rate' attribute on root element", file=sys.stderr)
    raise SystemExit(2)

rate = float(rate_attr) * 100.0
if rate + 1e-9 < min_cov:
    print(f"{label}: got {rate:.2f}% | wanted >= {min_cov:.2f}%")
    raise SystemExit(1)

if max_drift is not None and rate - min_cov > max_drift + 1e-9:
    print(
        f"{label}: got {rate:.2f}% | wanted <= {min_cov + max_drift:.2f}% "
        f"(min {min_cov:.2f}% + drift {max_drift:.2f}%). Raise COVERAGE_MIN.",
    )
    raise SystemExit(1)

print(f"{label}: got {rate:.2f}% | wanted >= {min_cov:.2f}%")
