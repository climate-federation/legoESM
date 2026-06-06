"""Tier-JSON tolerance loader + schema validator.

Each tier owns one JSON file under
``tests/ocean/fidelity/fixtures/tier{N}_<name>.json``. The loader is
intentionally schema-aware so a typo or missing key in a fixture fails
loudly instead of silently degrading a test to a no-op.

The schema (version 1) is intentionally hand-rolled — no ``jsonschema``
dependency — because every check here is structurally trivial and we do
not want to add a runtime dep at the foundation stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

CI_MARKERS: tuple[str, ...] = ("fast", "nightly", "manual_only")
REFERENCE_KINDS: tuple[str, ...] = (
    "analytical",
    "literature_scalar",
    "veros",
    "observation",
)
SCHEMA_VERSION: int = 1
TIER_RANGE: tuple[int, int] = (0, 8)


class TolerancesSchemaError(ValueError):
    """Raised when a tier JSON payload violates the v1 schema."""


@dataclass(frozen=True)
class CaseTolerance:
    metric: str
    tolerance_window: tuple[float, float]
    reference: dict[str, Any]
    ci_marker: Literal["fast", "nightly", "manual_only"]
    expected_runtime_s: float | None = None
    subcase_filters: dict[str, Any] | None = None


@dataclass(frozen=True)
class TierTolerances:
    schema_version: int
    tier: int
    description: str
    common: dict[str, Any]
    cases: dict[str, CaseTolerance]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TolerancesSchemaError(message)


def _validate_case(name: str, raw: Any) -> CaseTolerance:
    _require(
        isinstance(raw, dict),
        f"case {name!r}: expected object, got {type(raw).__name__}",
    )
    for required in ("metric", "tolerance_window", "reference", "ci_marker"):
        _require(required in raw, f"case {name!r}: missing required key {required!r}")

    metric = raw["metric"]
    _require(
        isinstance(metric, str) and metric,
        f"case {name!r}: 'metric' must be a non-empty string",
    )

    win = raw["tolerance_window"]
    _require(
        isinstance(win, list)
        and len(win) == 2
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in win),
        f"case {name!r}: 'tolerance_window' must be a [lo, hi] pair of numbers",
    )
    _require(
        float(win[0]) <= float(win[1]),
        f"case {name!r}: tolerance_window lo must be <= hi (got {win})",
    )

    ref = raw["reference"]
    _require(
        isinstance(ref, dict) and "kind" in ref,
        f"case {name!r}: 'reference' must be an object with a 'kind' field",
    )
    _require(
        ref["kind"] in REFERENCE_KINDS,
        f"case {name!r}: reference.kind {ref['kind']!r} not in {REFERENCE_KINDS}",
    )

    marker = raw["ci_marker"]
    _require(
        marker in CI_MARKERS,
        f"case {name!r}: ci_marker {marker!r} not in {CI_MARKERS}",
    )

    runtime = raw.get("expected_runtime_s")
    if runtime is not None:
        _require(
            isinstance(runtime, (int, float))
            and not isinstance(runtime, bool)
            and runtime > 0,
            f"case {name!r}: expected_runtime_s must be a positive number",
        )

    filters = raw.get("subcase_filters")
    if filters is not None:
        _require(
            isinstance(filters, dict),
            f"case {name!r}: subcase_filters must be an object when present",
        )

    return CaseTolerance(
        metric=metric,
        tolerance_window=(float(win[0]), float(win[1])),
        reference=dict(ref),
        ci_marker=marker,  # type: ignore[arg-type]
        expected_runtime_s=float(runtime) if runtime is not None else None,
        subcase_filters=dict(filters) if filters is not None else None,
    )


def validate_tier_payload(raw: Any) -> TierTolerances:
    """Validate a parsed tier-JSON payload and return a frozen dataclass."""
    _require(
        isinstance(raw, dict),
        f"tier payload must be an object, got {type(raw).__name__}",
    )
    for required in ("schema_version", "tier", "description", "common", "cases"):
        _require(required in raw, f"missing required key {required!r}")

    version = raw["schema_version"]
    _require(
        isinstance(version, int) and version == SCHEMA_VERSION,
        f"schema_version {version!r} does not match supported {SCHEMA_VERSION}",
    )

    tier = raw["tier"]
    lo, hi = TIER_RANGE
    _require(
        isinstance(tier, int) and lo <= tier <= hi,
        f"tier {tier!r} must be an integer in [{lo}, {hi}]",
    )

    desc = raw["description"]
    _require(
        isinstance(desc, str) and desc,
        "description must be a non-empty string",
    )

    common = raw["common"]
    _require(isinstance(common, dict), "common must be an object")

    cases = raw["cases"]
    _require(
        isinstance(cases, dict) and cases,
        "cases must be a non-empty object",
    )

    parsed = {name: _validate_case(name, body) for name, body in cases.items()}
    return TierTolerances(
        schema_version=version,
        tier=tier,
        description=desc,
        common=dict(common),
        cases=parsed,
    )


def load_tier_file(path: Path | str) -> TierTolerances:
    """Read and validate a tier-JSON file from disk.

    Errors are re-raised with the file path prepended so CI logs show
    exactly which fixture is malformed.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    try:
        return validate_tier_payload(raw)
    except TolerancesSchemaError as exc:
        raise TolerancesSchemaError(f"{path}: {exc}") from exc
