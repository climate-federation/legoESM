"""Deploy a saved correction-campaign output as a production turbulence override.

Closes the practical loop of ``docs/COMPARE_REANALYSIS.md`` ("updating these
parameters in the AMIP/CMIP simulations"): a campaign job
(``scripts/run/run_correction_campaign.py``) writes the corrected per-column
``clubb_lite`` coefficient field(s) to JSON, and a fresh PRODUCTION run loads them
as an :class:`~legoesm.atmosphere.physics.turbulence.config.TurbulenceConfig` for
``ExperimentConfig.turbulence_override`` — so the bias-reducing parameters are
applied to a new simulation.

The per-column override is RUNTIME-ONLY (the standard config serializers null it,
iter 35), so this JSON + loader is the deploy vehicle.  Usage::

    from legoesm.training.deploy_correction import corrected_turbulence_override
    override = corrected_turbulence_override("campaign_out.json")
    cfg = ExperimentConfig(..., turbulence="clubb_lite", turbulence_override=override)

The base config's ``turbulence`` MUST be ``"clubb_lite"`` (``validate_strict``
requires the override scheme to match).
"""

from __future__ import annotations

import json
from typing import Any

# CLUBB-lite promotion key → CLUBBLiteConfig field name (the campaign output's
# multi "fields" dict is keyed by promotion_key; the single output by field name).
_CLUBB_PROMOTION_TO_FIELD = {
    "clubb_lite_C_K": "C_K",
    "clubb_lite_Pr_t": "Pr_t",
    "clubb_lite_C_eps": "C_eps",
}
_CLUBB_FIELDS = ("C_K", "Pr_t", "C_eps")


def corrected_clubb_config(data: dict):
    """Build a per-column :class:`CLUBBLiteConfig` from a campaign-output dict.

    Handles both campaign output shapes:

    * MULTI (``--coefficients``): a ``"fields"`` dict ``{promotion_key: [...]}`` —
      each clubb promotion key maps to its config field.
    * SINGLE (``--diagnosis-method``): a top-level ``"C_K"`` / ``"Pr_t"`` /
      ``"C_eps"`` array (exactly one).

    Raises on an empty / unrecognized / non-CLUBB output (dispatch hardening).
    """
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig

    if not isinstance(data, dict):
        raise ValueError(
            f"campaign output must be a JSON object; got {type(data).__name__}.")
    defaults = CLUBBLiteConfig()
    overrides: dict[str, Any] = {}
    if "fields" in data:
        fields = data["fields"]
        if not isinstance(fields, dict):
            raise ValueError(
                "multi-coefficient campaign output 'fields' must be a dict keyed "
                f"by promotion_key; got {type(fields).__name__}.")
        for key, vals in fields.items():
            field = _CLUBB_PROMOTION_TO_FIELD.get(key)
            if field is None:
                raise ValueError(
                    f"deploy loader is CLUBB-lite-specific; campaign output has an "
                    f"unsupported promotion_key {key!r} "
                    f"(expected one of {tuple(_CLUBB_PROMOTION_TO_FIELD)}).")
            overrides[field] = _clean_field_array(field, vals, defaults)
    else:
        for field in _CLUBB_FIELDS:
            if field in data:
                overrides[field] = _clean_field_array(field, data[field], defaults)
        if len(overrides) != 1:
            raise ValueError(
                "single-coefficient campaign output must carry exactly one of "
                f"{_CLUBB_FIELDS}; got {sorted(overrides)}.")
    if not overrides:
        raise ValueError(
            "no corrected CLUBB-lite coefficient field found in the campaign "
            "output (expected a 'fields' dict or a C_K/Pr_t/C_eps array).")
    return CLUBBLiteConfig(**overrides)


def _clean_field_array(field: str, vals: Any, defaults):
    """Validate + dtype-normalize a corrected per-column coefficient array.

    Casts to the scheme field's default (float) dtype so a hand-edited / integer
    JSON array can never enter ``CLUBBLiteConfig`` as a non-differentiable int
    array, and rejects an empty / non-1-D / non-finite field LOUDLY at deploy
    time rather than silently feeding NaN turbulence (or a rank-2 broadcast
    crash) into a production run.
    """
    import jax.numpy as jnp

    dtype = jnp.asarray(getattr(defaults, field)).dtype
    arr = jnp.asarray(vals, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(
            f"corrected '{field}' field must be a 1-D per-column array; "
            f"got ndim={arr.ndim} (shape {tuple(arr.shape)}).")
    if arr.size == 0:
        raise ValueError(f"corrected '{field}' field is empty.")
    if not bool(jnp.all(jnp.isfinite(arr))):
        raise ValueError(f"corrected '{field}' field has non-finite values.")
    return arr


def corrected_turbulence_override(source: dict | str | Any):
    """Load a campaign output into a deployable ``TurbulenceConfig``.

    ``source`` is the campaign-output dict, a JSON path (str/``os.PathLike``), or
    an open file object.  Returns ``TurbulenceConfig(scheme="clubb_lite",
    clubb_lite=<per-column CLUBBLiteConfig>)`` for ``ExperimentConfig.turbulence_
    override`` (the base config's ``turbulence`` must be ``"clubb_lite"``).
    """
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    if isinstance(source, dict):
        data = source
    elif hasattr(source, "read"):
        data = json.load(source)
    else:
        with open(source) as f:
            data = json.load(f)
    return TurbulenceConfig(
        scheme="clubb_lite", clubb_lite=corrected_clubb_config(data)
    )
