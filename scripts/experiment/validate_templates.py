#!/usr/bin/env python
"""Validate every experiment template under ``config/templates/`` and emit the
run-status table (``project_status.md``) — the lesommer ``validate_templates`` +
``project_status`` tools, built on legoESM's existing config loader.

A template is a versioned ``legoesm run`` config YAML carrying an extra
``experiment:`` metadata block (tier / complexity / extent / maturity /
description / data / conservation_gates).  Validation = the metadata is
well-formed AND the config resolves + passes the production
``ExperimentConfig.validate_strict`` (reusing the *same* loader ``legoesm run``
uses, so a template that validates here is runnable):

    Config.from_yaml(path).to_experiment_config().validate_strict()

This does NOT reinvent provenance/reproduce — that is the driver's
``run_manifest.json`` + ``legoesm reproduce`` (a superset of the lesommer
``experiment.tag``).  This tool only guards the *template library*.

Usage::

    python scripts/experiment/validate_templates.py                # validate, print table
    python scripts/experiment/validate_templates.py --write-status  # also regen project_status.md
    python scripts/experiment/validate_templates.py --templates-dir config/templates

Exit code is non-zero if any template fails to load/resolve/validate or has a
malformed ``experiment:`` block, so CI gates on the template library directly.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo root = two levels up from this file (scripts/experiment/<this>).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TEMPLATES = _REPO_ROOT / "config" / "templates"
_DEFAULT_STATUS = _REPO_ROOT / "project_status.md"
_CATALOG = _REPO_ROOT / "config" / "data_catalog.yaml"

_VALID_TIERS = {"tier0", "tier1", "tier2", "tier3"}
_VALID_MATURITY = {"run_tested", "init_only", "structurally_validated"}
# Atmosphere complexity rungs whose value must equal the resolved dycore
# model_type (codex review HIGH-1: a template can declare `complexity:
# hydrostatic` while the config silently resolves to shallow_water).
_ATM_RUNGS = {"shallow_water", "hydrostatic", "nonhydrostatic"}


def _catalog_ids(catalog: Path | None = None) -> set[str]:
    """Dataset ids declared in config/data_catalog.yaml (empty if absent)."""
    import yaml
    catalog = catalog or _CATALOG
    if not catalog.is_file():
        return set()
    doc = yaml.safe_load(catalog.read_text()) or {}
    return set((doc.get("datasets") or {}).keys())


@dataclass
class TemplateReport:
    rel_path: str
    category: str
    tier: str
    complexity: str
    extent: str
    maturity: str
    description: str
    ok: bool
    error: str = ""


def _meta_errors(meta: dict[str, Any]) -> list[str]:
    """Validate the ``experiment:`` metadata block; return human-readable errors."""
    errs: list[str] = []
    if not isinstance(meta, dict):
        return ["missing or non-mapping 'experiment:' block"]
    tier = str(meta.get("tier", ""))
    if tier not in _VALID_TIERS:
        errs.append(f"tier={tier!r} not in {sorted(_VALID_TIERS)}")
    maturity = str(meta.get("maturity", ""))
    if maturity not in _VALID_MATURITY:
        errs.append(f"maturity={maturity!r} not in {sorted(_VALID_MATURITY)}")
    if not str(meta.get("complexity", "")).strip():
        errs.append("missing 'complexity'")
    if not str(meta.get("description", "")).strip():
        errs.append("missing 'description'")
    data = meta.get("data", [])
    if not isinstance(data, list):
        errs.append("'data' must be a list (empty = idealized / no external data)")
    return errs


def validate_template(path: Path) -> TemplateReport:
    """Load + resolve + strict-validate one template; never raises.

    Mode-aware: dispatches to the atmosphere or ocean YAML adapter through
    ``legoesm.experiment_registry`` so an ocean template validates through the
    same path ``run_omip_core2.py --config`` consumes.
    """
    from legoesm import experiment_registry

    rel = str(path.relative_to(_DEFAULT_TEMPLATES.parent))
    category = path.parent.name
    meta: dict[str, Any] = {}
    try:
        mode, cfg = experiment_registry.load_adapter(str(path))
        meta = cfg.get_meta()
        errs = _meta_errors(meta)
        # Resolve + strict-validate through the SAME path the runner uses.
        cfg.validate_strict()
        # codex HIGH-1: the declared atmosphere complexity rung MUST equal the
        # resolved dycore model_type — else a 'hydrostatic' template that left
        # the canonical `atmosphere.dynamics` at its shallow_water default would
        # validate while silently describing the wrong experiment.  Ocean
        # templates use a different complexity vocabulary, so this atmosphere-
        # specific cross-check only applies to atmosphere modes.
        complexity = str(meta.get("complexity", ""))
        # A ``setup:`` template (#388) names an idealized matrix case and does
        # NOT resolve a dycore from a recipe, so the complexity↔model_type
        # cross-check does not apply (its `complexity` describes the matrix
        # case, not a recipe-resolved dycore).
        is_setup_template = bool(getattr(cfg, "get", lambda *_: None)("setup"))
        if (not is_setup_template
                and experiment_registry.is_atmosphere_mode(mode)
                and complexity in _ATM_RUNGS):
            model_type = cfg.to_experiment_config().dycore.model_type
            if model_type != complexity:
                errs.append(
                    f"complexity={complexity!r} but resolved dycore.model_type="
                    f"{model_type!r} (set 'atmosphere.dynamics: {complexity}')"
                )
        # codex MEDIUM-2: every experiment.data id must exist in the catalog.
        known = _catalog_ids()
        for ds in (meta.get("data") or []):
            if ds not in known:
                errs.append(f"data id {ds!r} not in config/data_catalog.yaml")
    except Exception as exc:  # noqa: BLE001 — report, don't crash the sweep
        return TemplateReport(
            rel, category, str(meta.get("tier", "?")),
            str(meta.get("complexity", "?")), str(meta.get("extent", "?")),
            str(meta.get("maturity", "?")), str(meta.get("description", "")).strip()[:60],
            ok=False, error=f"{type(exc).__name__}: {exc}",
        )
    return TemplateReport(
        rel, category, str(meta.get("tier", "")), str(meta.get("complexity", "")),
        str(meta.get("extent", "")), str(meta.get("maturity", "")),
        str(meta.get("description", "")).strip().replace("\n", " ")[:80],
        ok=not errs, error="; ".join(errs),
    )


def collect_templates(templates_dir: Path) -> list[Path]:
    return sorted(p for p in templates_dir.rglob("*.yaml")
                  if not p.name.startswith("_"))


def render_status_md(reports: list[TemplateReport]) -> str:
    lines = [
        "# legoESM experiment templates — run-status",
        "",
        "Auto-generated by `scripts/experiment/validate_templates.py "
        "--write-status`. Do not edit by hand.",
        "",
        "Maturity: **run_tested** (integration completed clean) · "
        "**init_only** (constructs, not run end-to-end) · "
        "**structurally_validated** (config passes `validate_strict`, no run).",
        "",
        "| Template | Tier | Complexity | Extent | Maturity | Valid | Description |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(reports, key=lambda r: (r.tier, r.rel_path)):
        valid = "✅" if r.ok else f"❌ {r.error}"
        desc = r.description.replace("|", "\\|")
        lines.append(
            f"| `{r.rel_path}` | {r.tier} | {r.complexity} | {r.extent} | "
            f"{r.maturity} | {valid} | {desc} |"
        )
    n_ok = sum(r.ok for r in reports)
    lines += ["", f"**{n_ok}/{len(reports)} templates valid.**", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--templates-dir", default=str(_DEFAULT_TEMPLATES))
    p.add_argument("--write-status", action="store_true",
                   help="(re)generate project_status.md from the results")
    p.add_argument("--status-path", default=str(_DEFAULT_STATUS))
    args = p.parse_args(argv)

    templates_dir = Path(args.templates_dir)
    if not templates_dir.is_dir():
        print(f"ERROR: templates dir not found: {templates_dir}")
        return 2
    paths = collect_templates(templates_dir)
    if not paths:
        print(f"ERROR: no templates (*.yaml) under {templates_dir}")
        return 2

    reports = [validate_template(p) for p in paths]
    for r in reports:
        mark = "  PASS" if r.ok else "**FAIL"
        print(f"  {mark} | {r.tier:5s} | {r.maturity:22s} | {r.rel_path}"
              + ("" if r.ok else f"  -> {r.error}"))
    n_ok = sum(r.ok for r in reports)
    print(f"\n  {n_ok}/{len(reports)} templates valid.")

    if args.write_status:
        Path(args.status_path).write_text(render_status_md(reports))
        print(f"  Wrote {args.status_path}")

    return 0 if n_ok == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
