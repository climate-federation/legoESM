"""Shared Veros-faithful free-run stepping composition (#429 / #433).

The five Veros free-run recipes (``acc``, ``acc_basic``, ``global_4deg``,
``global_flexible``, ``global_1deg``) all select the SAME time-stepping
composition and differ only in the per-setup ``dt_mom:dt_tracer`` ratio:

  * ``outer_integrator="ab2"``          — Veros AB2 outer integrator
  * ``ab2_scope="advective"``           — dissipative tendencies at weight 1.0
  * ``barotropic_solver="rigid_lid"``   — Veros enable_streamfunction
  * ``coriolis_scheme="explicit_ab2"``  — tend_coriolisf in du (the energy lever)
  * ``momentum_friction_additive=True`` — Veros solve_stream.py du_mix placement
  * ``implicit_vmix_dzw_slot=True``     — Veros dzw implicit-diffusion slot (#428)

This composition used to live as FIVE copy-pasted blocks, so the next
composition-level fix needed five synchronised edits.  PR #432 caught exactly
that drift: ``build_acc_basic_model_config`` was a copy that did not pick up
PR #429's faithful-stepping defaults, so acc_basic silently shipped the leaky
``matsuno_split`` / ``"total"`` composition for a day.

This module is the single source of truth.  :data:`VEROS_FAITHFUL_STEPPING_SIGNATURE`
is the composition (everything except the per-setup ratio); :func:`veros_faithful_stepping`
returns the kwargs to splat into a :class:`~legoesm.ocean.state.LatLonCGridOceanConfig`.
The ratchet ``tests/ocean/unit/test_veros_faithful_stepping_shared.py`` asserts that
EVERY Veros free-run recipe's config carries this exact signature — a recipe that
copy-pastes a divergent value (or omits a future composition field) goes red.

This is config-only (it selects existing canonical model blocks); it does not
introduce numerics and slots into the #376 recipe architecture, not a parallel
system.  The acc / acc_basic recipes gate the composition on the free-run path
(``with_surface_forcing``); on the frozen-state tendency-probe path every field
above equals its ``LatLonCGridOceanConfig`` default, so those recipes simply omit
the splat there (bit-identical).
"""

from __future__ import annotations

from typing import Any

# The stepping composition, mapping LatLonCGridOceanConfig field -> Veros-faithful
# value.  EXCLUDES the per-setup ``dt_mom_ratio`` (the one field that varies across
# setups; supplied by :func:`veros_faithful_stepping`).
VEROS_FAITHFUL_STEPPING_SIGNATURE: dict[str, Any] = {
    "outer_integrator": "ab2",
    "ab2_scope": "advective",
    "barotropic_solver": "rigid_lid",
    "coriolis_scheme": "explicit_ab2",
    "momentum_friction_additive": True,
    "implicit_vmix_dzw_slot": True,
}


def veros_faithful_stepping(*, dt_mom_ratio: float) -> dict[str, Any]:
    """Veros-faithful free-run stepping kwargs for ``LatLonCGridOceanConfig``.

    Splat into the model-config constructor::

        LatLonCGridOceanConfig(
            ..., **veros_faithful_stepping(dt_mom_ratio=R), physics=...)

    Returns the shared composition (:data:`VEROS_FAITHFUL_STEPPING_SIGNATURE`)
    plus ``dt_mom_ratio`` — the only field that varies across setups (acc 9.0,
    global_4deg 48, global_flexible 8, global_1deg 1, acc_basic
    ``DT_TRACER_S/DT_MOM_S``).
    """
    return {**VEROS_FAITHFUL_STEPPING_SIGNATURE, "dt_mom_ratio": float(dt_mom_ratio)}
