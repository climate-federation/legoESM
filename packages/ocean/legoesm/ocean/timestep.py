"""Grid-derived timestep selection.

Authored by GLM (design and function body), 2026-09-11, at the user's request
that the timestep be made grid-size dependent rather than a hand-set literal
per run card. Reviewed by codex and Claude.

WHY THIS EXISTS. The OMIP cards carried unexplained timestep literals -- 150 s
on the tripole and MPAS, 1800 s on FESOM2 -- while NEMO ORCA1 runs rn_Dt=3600
at the same nominal resolution. Measuring the tripole mesh showed the minimum
cell spacing is 1.00 km but that every cell that small is DRY LAND; the
smallest WET cell is 23.3 km. Excluding dry cells, the rule below returns
about 3490 s, which is essentially the oracle's own timestep, and feeding the
dry cells back in reproduces the 150 s literal exactly. So 150 s looks like a
Courant bound taken without a wet mask.

WHAT THIS FUNCTION DOES AND DOES NOT ESTABLISH. It returns the HORIZONTAL
advective/internal-wave Courant bound and nothing more. Codex's review
(2026-09-11) showed that is NECESSARY BUT NOT SUFFICIENT here, on two counts,
and neither is handled by this function:

  * TRACERS ARE STILL EXPLICIT on this path -- superbee vertical fluxes feed a
    forward-Euler update -- so the VERTICAL TRACER CFL is NOT removed by the
    adaptive-implicit vertical advection, which applies to momentum. Clipping
    the reconstruction's Courant number does not bound the transported flux.
    RK3 also freezes density across stages, so its stability region cannot by
    itself certify the coupled internal-wave step.
  * THE DRY-CELL EXCLUSION IS NOT SAFE WHEN SEA ICE IS ACTIVE. Tripole ice
    transport carries no coastal mask and can advect ice onto nominally dry
    cells (ice/transport.py), with explicit donor-cell transport and a single
    substep, so an ocean-dry sliver can still carry an ice CFL. Pass a mask
    that reflects everything that actually carries dynamics, not just the
    ocean wet mask, and prefer the RESOLVED mask since cyclic-overlap setup
    reconnects nominally dry halo columns.

MEASURED CONSEQUENCE, 2026-09-11: the vertical tracer condition does NOT just
theoretically survive, it BINDS FIRST on this configuration. From the baseline
day-30 tripole state, the largest |w| is 2.25e-03 m/s against a thinnest layer
of 1.02 m, so the explicit vertical tracer step needs dt <= about 456 s at unit
Courant -- roughly EIGHT TIMES SMALLER than the 3488 s horizontal bound this
function returns. The card's 150 s sits comfortably under that, at a vertical
Courant of about 0.33. So the horizontal number below must NEVER be adopted on
its own: on this configuration it is not the binding constraint.

Treat the returned value as an UPPER BOUND from one condition, to be combined
with the vertical tracer bound and confirmed by a short integration, never as
a certificate of stability.

WHICH CONDITION THIS ONE IS. With the barotropic mode solved implicitly the
external-wave CFL is removed, so the signal speed here is the first baroclinic
mode plus the advective velocity, NOT sqrt(g*H).

A COURANT NUMBER IS A MAX-NORM QUANTITY, so the only valid reduction over a
mesh is the MINIMUM over cells. Volume-weighted or harmonic means measure
accuracy, not stability, and would understate the constraint. If a vanishing
fraction of wet cells sets the minimum, the mesh wants repairing rather than
the timestep shrinking -- which is why the provenance below reports how many
cells bind.

This is a SETUP-TIME helper and is deliberately not jittable: it takes concrete
arrays, and tracing it raises, which is the intended loud failure.
"""
from __future__ import annotations

import numpy as np


def derive_timestep(dx, wet_mask, *, c_courant, c1_baroclinic, u_max,
                    dt_cap=None):
    """Largest timestep the explicit baroclinic step admits on this mesh.

    ``dx`` is the cell-to-cell spacing and ``wet_mask`` selects the cells that
    actually carry dynamics. Dry cells are excluded, which is the whole point:
    on a tripolar mesh the convergent cells near the pivot are land, and
    letting them set the timestep costs a factor of twenty for nothing.

    The signal speed is ``c1_baroclinic + u_max``. Pass the total speed as
    ``c1_baroclinic`` with ``u_max=0`` if it has already been combined.

    Returns ``(dt, provenance)``; the provenance records why this timestep was
    chosen so a run can log it instead of leaving a bare number in a card.
    """
    dx = np.asarray(dx, dtype=float)
    wet = np.asarray(wet_mask, dtype=bool)

    if dx.shape != wet.shape:
        raise ValueError(
            f"shape mismatch: dx {dx.shape} vs wet_mask {wet.shape}")
    if not wet.any():
        raise ValueError("wet_mask selects no cells: nothing constrains dt")

    dx_wet = dx[wet]
    if not np.isfinite(dx_wet).all():
        bad = np.flatnonzero(~np.isfinite(dx_wet))
        raise ValueError(
            f"non-finite dx at {bad.size} wet cell(s), first flat index "
            f"{bad[0]}")
    if not (dx_wet > 0).all():
        raise ValueError("non-positive dx at wet cell(s)")

    for _name, _v in (("c_courant", c_courant),
                      ("c1_baroclinic", c1_baroclinic), ("u_max", u_max)):
        if not np.isfinite(_v):
            raise ValueError(f"{_name} must be finite; got {_v!r}")
    if not c_courant > 0:
        raise ValueError("c_courant must be > 0")
    if c1_baroclinic < 0 or u_max < 0:
        raise ValueError("c1_baroclinic and u_max must be >= 0")
    if dt_cap is not None and not (np.isfinite(dt_cap) and dt_cap > 0):
        # A NaN cap used to fall through the comparison below and silently
        # leave the Courant value in place (codex review).
        raise ValueError(f"dt_cap must be finite and > 0; got {dt_cap!r}")

    speed = c1_baroclinic + u_max
    if speed <= 0:
        raise ValueError("c1_baroclinic + u_max must be > 0")

    i_min = int(np.argmin(dx_wet))
    dx_min = float(dx_wet[i_min])
    dt_courant = c_courant * dx_min / speed

    if dt_cap is not None and dt_cap < dt_courant:
        dt, binding = float(dt_cap), "dt_cap"
    else:
        dt, binding = dt_courant, "courant"

    flat_idx = int(np.flatnonzero(wet.reshape(-1))[i_min])
    provenance = {
        "dt": dt,
        "binding_constraint": binding,
        "dx_min": dx_min,
        "dx_min_index": tuple(int(i) for i in
                              np.unravel_index(flat_idx, dx.shape)),
        # Only meaningful when the Courant condition is what binds; a cap
        # binding says nothing about how many cells set the spacing minimum.
        "n_binding_cells": (int(np.count_nonzero(dx_wet == dx_min))
                            if binding == "courant" else None),
        "n_dx_min_cells": int(np.count_nonzero(dx_wet == dx_min)),
        "n_wet_cells": int(dx_wet.size),
        "wave_speed": float(speed),
        "c_courant_target": float(c_courant),
        "courant_resulting": dt * speed / dx_min,
        "dt_courant": dt_courant,
        "dt_cap": None if dt_cap is None else float(dt_cap),
    }
    return dt, provenance
