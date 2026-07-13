"""Reference-dataset resolution for the land evaluation pipeline.

A *reference* is whatever a model run is scored against: the Fortran CLM-ML
v2 output, the JAX-standalone output, or tower observations.  Because the
observation/staged datasets are gitignored (a fresh clone has no data), this
module implements the "both / per-dataset" access policy documented in
``docs/land/evaluation_data_manifest.md``:

    1. an explicit ``path`` given in the recipe wins;
    2. else ``$LEGO_LAND_EVAL_DATA`` (a shared bundle root) + the reference's
       documented relative path;
    3. else the in-tree default (only the Fortran/JAX references ship in-tree);
    4. else a hard error naming the manifest + fetch script — never a silent
       "reference missing, scored 0".

``KNOWN_REFERENCES`` is the registry; naming an unknown reference id is a
``ValueError`` (dispatch hardening), not a silent default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Env var naming a shared data bundle root (HPC filesystem / cloud mount).
DATA_ROOT_ENV = "LEGO_LAND_EVAL_DATA"


@dataclass(frozen=True)
class ReferenceSpec:
    """How to find and read one reference dataset.

    ``fmt`` selects the reader downstream: ``"clm_ml_out"`` for the
    whitespace ``.out`` triplet (flux/fsun/aux), ``"obs_csv"`` for a
    per-timestep observation CSV, ``"chats_obs"`` for the multi-level
    Zenodo CHATS tower CSV.
    """

    ref_id: str
    label: str
    fmt: str
    in_tree_default: str | None  # repo-relative dir, or None if data-blocked
    bundle_relpath: str | None  # path under $LEGO_LAND_EVAL_DATA
    doc: str = ""


# Built-in references.  in_tree_default=None means "not in git" -> must come
# from the bundle root or an explicit recipe path.
KNOWN_REFERENCES: dict[str, ReferenceSpec] = {
    "fortran_v2": ReferenceSpec(
        ref_id="fortran_v2",
        label="Fortran CLM-ML v2",
        fmt="clm_ml_out",
        in_tree_default="docs/output_files_clm_ml-v2",
        bundle_relpath="clm_ml_v2_fortran",
        doc="Reference Fortran CLM-ML v2 offline outputs (in-tree).",
    ),
    "jax_standalone": ReferenceSpec(
        ref_id="jax_standalone",
        label="CLM-ML-JAX standalone",
        fmt="clm_ml_out",
        in_tree_default="clm-ml-jax/src/output_files/JAX_outputs_05_2007_31days",
        bundle_relpath="clm_ml_jax_standalone",
        doc="JAX-standalone CLM-ML outputs, the Fortran-parity reference.",
    ),
    "zenodo_obs": ReferenceSpec(
        ref_id="zenodo_obs",
        label="CHATS tower obs (Zenodo 17426258)",
        fmt="chats_obs",
        in_tree_default=None,  # gitignored under docs/MLC_experiment_plan/Data
        bundle_relpath="Zenodo_17426258_CHATS_30-min",
        doc="Bonan/Burns/Patton 2026 CHATS 30-min multi-level obs.",
    ),
    "fluxnet_obs": ReferenceSpec(
        ref_id="fluxnet_obs",
        label="FLUXNET tower obs",
        fmt="obs_csv",
        in_tree_default=None,  # obs_targets.csv is produced under results/
        bundle_relpath="fluxnet",
        doc="Per-timestep obs_targets.csv written by run_fluxnet_offline.py.",
    ),
}


def get_reference_spec(ref_id: str) -> ReferenceSpec:
    """Look up a built-in reference, raising on an unknown id."""
    try:
        return KNOWN_REFERENCES[ref_id]
    except KeyError:
        raise ValueError(
            f"Unknown reference id {ref_id!r}; known references: "
            f"{sorted(KNOWN_REFERENCES)}. Add one to KNOWN_REFERENCES or give "
            f"an explicit 'path' in the recipe."
        ) from None


def resolve_reference_dir(
    ref_id: str,
    repo_root: Path,
    explicit_path: str | None = None,
) -> Path:
    """Resolve a reference to a concrete directory following the policy.

    ``explicit_path`` (from the recipe) may be absolute or repo-relative
    and may contain ``$LEGO_LAND_EVAL_DATA`` / ``~``.  Raises
    ``FileNotFoundError`` with actionable guidance when nothing resolves.
    """
    spec = get_reference_spec(ref_id)

    # 1) explicit path from the recipe.
    if explicit_path:
        p = Path(os.path.expandvars(os.path.expanduser(explicit_path)))
        if not p.is_absolute():
            p = repo_root / p
        if p.is_dir():
            return p
        raise FileNotFoundError(
            f"Reference {ref_id!r}: explicit path {p} does not exist."
        )

    # 2) shared bundle root.
    root = os.environ.get(DATA_ROOT_ENV)
    if root and spec.bundle_relpath:
        p = Path(os.path.expanduser(root)) / spec.bundle_relpath
        if p.is_dir():
            return p

    # 3) in-tree default.
    if spec.in_tree_default:
        p = repo_root / spec.in_tree_default
        if p.is_dir():
            return p

    # 4) give up loudly.
    raise FileNotFoundError(
        f"Could not locate reference {ref_id!r} ({spec.label}).\n"
        f"  Tried: explicit path (none), ${DATA_ROOT_ENV}/"
        f"{spec.bundle_relpath} (root="
        f"{os.environ.get(DATA_ROOT_ENV, '<unset>')}), in-tree "
        f"{spec.in_tree_default or '<not in git>'}.\n"
        f"  See docs/land/evaluation_data_manifest.md and "
        f"scripts/data/fetch_land_eval_data.py to obtain it, or set "
        f"${DATA_ROOT_ENV} to a bundle containing '{spec.bundle_relpath}'."
    )
