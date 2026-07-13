"""Config-driven evaluation recipes (ESMValTool-style) for the land pipeline.

A *recipe* is a YAML file describing one or more evaluation *cases*: a model
run (a directory of ``.out`` files) scored against one or more references,
over a chosen set of variables and metrics.  Running the recipe produces a
JSON scorecard (``legoesm.land.evaluation.scorecard``) and printed tables —
no bespoke per-experiment script.

Example recipe (see ``config/land_eval/chats7_parity.yaml``)::

    name: chats7_parity
    output_dir: validation_output
    cases:
      - name: chats7_adapter_vs_fortran
        model: {label: legoESM adapter, path: results/chats7_adapter_2007-05,
                tag: CHATS7_2007-05}
        references: [fortran_v2, jax_standalone]
        variables: [flux:shflx, flux:lhflx, flux:gpp, fsun:tl_sun]
        metrics: [bias, rmse, nrmse, corr, bias_score, rmse_score, taylor_score]

Scope of the row-aligned scorer here: references of format ``clm_ml_out``
(the Fortran and JAX ``.out`` triplets), which share the model's schema and
timestep grid.  Observation references (``fluxnet_obs`` / ``chats_obs``)
require site-specific time-alignment and are produced through the plotting
stage (``scripts/plot/plot_obs_vs_adapter_4sites.py``,
``plot_chats7_profile_e2.py``); pointing a recipe case at an obs reference
raises with that guidance rather than silently mis-scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from legoesm.land.evaluation import fluxio, references
from legoesm.land.evaluation.metrics import METRIC_REGISTRY
from legoesm.land.evaluation.scorecard import (
    Scorecard,
    VariableResult,
    score_variable,
)

# Build a variable-key -> [tags that contain it] index once.
_KEY_TO_TAGS: dict[str, list[str]] = {}
for _tag, _specs in fluxio.SCHEMAS.items():
    for _spec in _specs:
        _KEY_TO_TAGS.setdefault(_spec[0], []).append(_tag)

_LABEL_UNIT: dict[tuple[str, str], tuple[str, str]] = {
    (tag, spec[0]): (spec[1], spec[2])
    for tag, specs in fluxio.SCHEMAS.items()
    for spec in specs
}


@dataclass
class ModelSpec:
    """The model run being evaluated: a directory of ``<tag>_<schema>.out``."""

    label: str
    path: str
    tag: str


@dataclass
class ReferenceRef:
    """A reference selection in a case (id + optional path/label override)."""

    ref_id: str
    path: str | None = None
    label: str | None = None


@dataclass
class EvalCase:
    name: str
    model: ModelSpec
    references: list[ReferenceRef]
    variables: list[str]  # "tag:key" or bare "key" (must be unambiguous)
    metrics: list[str]
    score_weights: dict[str, float] | None = None


@dataclass
class EvalRecipe:
    name: str
    cases: list[EvalCase]
    output_dir: str = "validation_output"
    description: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------


def _resolve_variable(entry: str) -> tuple[str, str]:
    """Map a recipe variable entry to ``(tag, key)``, raising if invalid."""
    if ":" in entry:
        tag, key = entry.split(":", 1)
        if tag not in fluxio.SCHEMAS:
            raise ValueError(
                f"variable {entry!r}: unknown schema tag {tag!r}; "
                f"known: {sorted(fluxio.SCHEMAS)}"
            )
        if key not in {s[0] for s in fluxio.SCHEMAS[tag]}:
            raise ValueError(
                f"variable {entry!r}: key {key!r} not in schema {tag!r}"
            )
        return tag, key
    tags = _KEY_TO_TAGS.get(entry)
    if not tags:
        raise ValueError(
            f"variable {entry!r}: unknown key; not in any schema "
            f"{sorted(fluxio.SCHEMAS)}"
        )
    if len(tags) > 1:
        raise ValueError(
            f"variable {entry!r}: ambiguous (in schemas {tags}); qualify it "
            f"as e.g. {tags[0]}:{entry}"
        )
    return tags[0], entry


def parse_recipe(data: dict) -> EvalRecipe:
    """Validate a parsed-YAML dict into an ``EvalRecipe``.

    Fails fast on unknown metric names, unknown reference ids, and
    malformed variable entries — a typo in a recipe must raise before any
    scoring runs, never silently drop a variable/metric.
    """
    if not isinstance(data, dict):
        raise ValueError("recipe must be a mapping at top level")
    name = str(data.get("name") or "unnamed_recipe")
    cases_raw = data.get("cases")
    if not cases_raw:
        raise ValueError(f"recipe {name!r}: no 'cases' defined")

    cases: list[EvalCase] = []
    for i, c in enumerate(cases_raw):
        cname = str(c.get("name") or f"case_{i}")
        model_raw = c.get("model")
        if not model_raw or "path" not in model_raw or "tag" not in model_raw:
            raise ValueError(
                f"case {cname!r}: 'model' needs 'path' and 'tag'"
            )
        model = ModelSpec(
            label=str(model_raw.get("label") or cname),
            path=str(model_raw["path"]),
            tag=str(model_raw["tag"]),
        )

        refs_raw = c.get("references") or []
        if not refs_raw:
            raise ValueError(f"case {cname!r}: no 'references' listed")
        refs: list[ReferenceRef] = []
        for r in refs_raw:
            if isinstance(r, str):
                refs.append(ReferenceRef(ref_id=r))
            elif isinstance(r, dict):
                if "id" not in r:
                    raise ValueError(
                        f"case {cname!r}: reference dict needs 'id'"
                    )
                refs.append(ReferenceRef(
                    ref_id=str(r["id"]),
                    path=r.get("path"),
                    label=r.get("label"),
                ))
            else:
                raise ValueError(
                    f"case {cname!r}: reference must be a string id or a "
                    f"mapping, got {type(r).__name__}"
                )
            # Validate the id now (raises on unknown).
            references.get_reference_spec(refs[-1].ref_id)

        variables = [str(v) for v in (c.get("variables") or [])]
        if not variables:
            raise ValueError(f"case {cname!r}: no 'variables' listed")
        for v in variables:
            _resolve_variable(v)  # validate, raise on bad entry

        metric_names = [str(mn) for mn in (c.get("metrics") or [])]
        if not metric_names:
            raise ValueError(f"case {cname!r}: no 'metrics' listed")
        for mn in metric_names:
            if mn not in METRIC_REGISTRY:
                raise ValueError(
                    f"case {cname!r}: unknown metric {mn!r}; known: "
                    f"{sorted(METRIC_REGISTRY)}"
                )

        cases.append(EvalCase(
            name=cname, model=model, references=refs,
            variables=variables, metrics=metric_names,
            score_weights=c.get("score_weights"),
        ))

    return EvalRecipe(
        name=name,
        cases=cases,
        output_dir=str(data.get("output_dir") or "validation_output"),
        description=str(data.get("description") or ""),
        metadata=dict(data.get("metadata") or {}),
    )


def load_recipe(path: str | Path) -> EvalRecipe:
    """Load and validate a recipe YAML file."""
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return parse_recipe(data)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _load_clm_ml_out(dir_path: Path, tag: str) -> dict[str, dict[str, "object"]]:
    """Load {schema_tag: {var_key: column}} for a CLM-ML ``.out`` triplet."""
    out: dict[str, dict] = {}
    for schema_tag in fluxio.SCHEMAS:
        fpath = dir_path / f"{tag}_{schema_tag}.out"
        if fpath.exists():
            out[schema_tag] = fluxio.load_out_named(fpath, schema_tag)
        else:
            out[schema_tag] = {}
    return out


def run_case(
    case: EvalCase, repo_root: Path
) -> list[Scorecard]:
    """Score one case's model against each reference; return scorecards."""
    model_dir = Path(case.model.path)
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir
    model_cols = _load_clm_ml_out(model_dir, case.model.tag)

    var_tags = [_resolve_variable(v) for v in case.variables]

    cards: list[Scorecard] = []
    for ref in case.references:
        spec = references.get_reference_spec(ref.ref_id)
        if spec.fmt != "clm_ml_out":
            raise ValueError(
                f"case {case.name!r}: reference {ref.ref_id!r} has format "
                f"{spec.fmt!r}, which the row-aligned recipe scorer does not "
                f"handle. Observation references are scored through the "
                f"plotting stage (see scripts/plot/plot_obs_vs_adapter_4sites"
                f".py). Use only clm_ml_out references in a scoring recipe."
            )
        ref_dir = references.resolve_reference_dir(
            ref.ref_id, repo_root, ref.path
        )
        ref_cols = _load_clm_ml_out(ref_dir, case.model.tag)

        results: list[VariableResult] = []
        for tag, key in var_tags:
            mcol = model_cols.get(tag, {}).get(key)
            rcol = ref_cols.get(tag, {}).get(key)
            if mcol is None or rcol is None:
                # Variable absent in this model/reference: record n=0 rather
                # than fabricate a score.
                label, unit = _LABEL_UNIT.get((tag, key), (key, ""))
                results.append(VariableResult(
                    key=f"{tag}:{key}", label=label, unit=unit, group=tag,
                    n=0, metrics={}, score=float("nan"),
                ))
                continue
            n = min(len(mcol), len(rcol))
            label, unit = _LABEL_UNIT.get((tag, key), (key, ""))
            results.append(score_variable(
                rcol[:n], mcol[:n],
                key=f"{tag}:{key}", label=label, unit=unit, group=tag,
                metric_names=case.metrics,
                score_weights=case.score_weights,
            ))

        cards.append(Scorecard(
            case=case.name,
            model=case.model.label,
            reference=ref.label or spec.label,
            variables=results,
            metadata={"reference_dir": str(ref_dir),
                      "model_dir": str(model_dir)},
        ))
    return cards


def run_recipe(
    recipe: EvalRecipe, repo_root: Path
) -> list[Scorecard]:
    """Run every case in a recipe; return the flat list of scorecards."""
    cards: list[Scorecard] = []
    for case in recipe.cases:
        cards.extend(run_case(case, repo_root))
    return cards
