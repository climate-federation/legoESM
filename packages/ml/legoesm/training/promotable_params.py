"""Registry of GCM coefficients the LES feedback can promote to per-column fields.

Stage 7 of ``docs/COMPARE_REANALYSIS.md`` (the per-scheme **promotion**): records
which production scheme ``*Config`` fields can accept a spatially-varying
``(ncol,)`` field — and applies one — closing the loop into a real scheme without
risky body surgery.

A field is *promotable* when the scheme body already consumes it element-wise
over the column dimension, so a per-column array broadcasts identically to the
scalar default (the production path stays byte-identical).  The gray-radiation
optical depth is the cleanest example: ``tau_ref = tau_e + (tau_p−tau_e)·sin²lat``
is already ``(ncol,)``-vectorised, so promoting ``tau_equator`` / ``tau_pole``
needs NO body change.  (Gray ``sfc_albedo`` is NOT promotable via the config
field — the driver passes a per-column ``sfc_albedo`` *argument* that shadows
``config.sfc_albedo``; see the NOTE by the registry.)  The CLUBB-lite
eddy-diffusivity coefficient ``C_K`` is the **physically-coherent** target — the
LES eddy-diffusivity diagnosis maps directly onto it; its body use is wrapped in
:func:`legoesm.atmosphere.physics._shared.broadcast_column_param` so a per-column
``C_K`` broadcasts over the vertical with the scalar path byte-identical.

The field is applied through :func:`legoesm.training.feedback.apply_column_parameter_field`
with the registered name as the ``promoted_fields`` allowlist — which avoids
adding a ``shape``-keyed ``__param_spec__`` entry (that would force the scalar
trainable-param collector to thread a ``dims={'n_col': N}`` everywhere).

NOTE on the physical mapping: this records the promotion *mechanism*.  Which LES
diagnosis (eddy diffusivity / entrainment) maps onto which GCM coefficient is the
scientific choice deferred in §6; a turbulence-diffusivity or convection-
entrainment field promotion follows the SAME pattern once its scheme body is
made ``(ncol,)``-safe (wrap the coefficient use in a column-broadcast).
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm.training.feedback import apply_column_parameter_field


class PromotableField(NamedTuple):
    """A scheme ``*Config`` field that accepts a per-column feedback field."""

    field: str            # the NamedTuple field name on the scheme config
    scheme: str           # human-readable scheme name
    units: str            # field units
    body_safe: bool       # True ⇒ scheme body already broadcasts (ncol,) (no edit)
    note: str = ""


# Coefficients verified per-column-capable today (gray radiation is already
# vectorised over columns).  Extend as schemes are made ``(ncol,)``-safe.
PROMOTABLE_FIELDS: dict[str, PromotableField] = {
    "gray_tau_equator": PromotableField(
        field="tau_equator", scheme="gray radiation", units="1", body_safe=True,
        note="tau_ref = tau_e + (tau_p-tau_e)*sin2lat is element-wise (ncol,).",
    ),
    "gray_tau_pole": PromotableField(
        field="tau_pole", scheme="gray radiation", units="1", body_safe=True,
        note="same element-wise optical-depth path as tau_equator.",
    ),
    "clubb_lite_C_K": PromotableField(
        field="C_K", scheme="CLUBB-lite turbulence", units="1", body_safe=True,
        note="K_m = C_K·l·√(wp2); the body wraps C_K in broadcast_column_param "
             "so a per-column field broadcasts over the vertical. PHYSICALLY "
             "maps the LES eddy-diffusivity diagnosis onto the GCM diffusivity.",
    ),
}
# NOTE: gray ``sfc_albedo`` is deliberately NOT registered — the driver passes a
# per-column ``sfc_albedo`` as an explicit argument to ``gray_radiation`` that
# SHADOWS ``config.sfc_albedo`` (``alpha = config.sfc_albedo if sfc_albedo is
# None else sfc_albedo``), so a config-field promotion would be silently dead in
# production.  A per-column albedo feedback must thread the explicit argument
# instead — a different mechanism, out of scope here.


def promotable_field_names() -> tuple[str, ...]:
    """Registry keys of the currently per-column-promotable coefficients."""
    return tuple(PROMOTABLE_FIELDS)


def apply_feedback_to_scheme(
    config, registry_key: str, field, *, expected_ncol: int | None = None
):
    """Splice a per-column feedback ``field`` into a scheme config (traced-in-loss).

    ``registry_key`` selects the promotable coefficient (raises on unknown —
    dispatch hardening); the field is applied via
    :func:`apply_column_parameter_field` with the registered field name as the
    authorization allowlist, so it can never silently overwrite an
    un-registered scalar parameter.  Pass ``expected_ncol`` (the scheme's column
    count) to validate the field length at the splice point rather than letting
    a mismatched field fail or mis-align deep in the physics.  Returns the
    updated config.
    """
    if registry_key not in PROMOTABLE_FIELDS:
        raise ValueError(
            f"Unknown promotable coefficient {registry_key!r}; choose from "
            f"{tuple(PROMOTABLE_FIELDS)}."
        )
    promotion = PROMOTABLE_FIELDS[registry_key]
    return apply_column_parameter_field(
        config, promotion.field, field,
        promoted_fields={promotion.field}, expected_ncol=expected_ncol,
    )
