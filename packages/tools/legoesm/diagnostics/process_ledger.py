"""Per-process column water/energy tendency ledger (budget attribution).

Attributes the model's column WATER and dry-enthalpy ENERGY tendencies to
the physical operator that produced them, so a global budget non-closure
(e.g. the 2-yr AMIP pilot's 1.4 mm/day vapor sink and ~190 W/m²
unexplained atmospheric heating, 2026-07-22) lands in a NAMED row instead
of an inferred one.

Ledger semantics (sign convention, stated once here and binding for every
row): each row is the process's CONTRIBUTION to d/dt of the atmospheric
column store — positive = the process ADDS water [kg/m²/s] / dry enthalpy
[W/m²] to the column, area-mean over the globe.  Consequences:

* Surface evaporation enters through the TURBULENCE row (the implicit
  bottom-BC moistening), NOT as a separate E row.
* Precipitation leaves through the CONVECTION / MICROPHYSICS rows (their
  ∫dq dp/g is negative by the amount precipitated) — there is no separate
  P row; a conserving scheme's water row equals −(its surface precip).
* ``sum(rows) == d/dt(column total water)`` between the segment snapshots
  by construction ("other_physics" and "clips"/"dynamics" are computed as
  residual/deltas, see below) — the closure check is that the TOTAL
  matches E − P − dW/dt from the independent MoistureBudgetTracker.
* The energy column is DRY ENTHALPY ONLY (c_pd·∫dT dp/g): latent,
  kinetic and geopotential terms are excluded BY DESIGN — this ledger
  attributes heating, it is not a closed moist-static-energy budget
  (that lives in ``energy_budget.EnergyBudgetTracker``).

Rows (fixed order, ``LEDGER_PROCESSES``):

* ``turbulence``    — BL scheme tendencies incl. its implicit surface flux
  bottom BC (production: louis).
* ``convection``    — the convection scheme's dT/dq contributions.
* ``microphysics``  — the microphysics scheme's contributions (its surface
  precip appears as the negative column ∫dq of removal/sedimentation).
* ``radiation``     — the held radiative heating ``dT_dt_rad`` (water 0).
* ``other_physics`` — RESIDUAL of the assembled PhysicsOutput totals minus
  the four rows above: GWD heating, the joint vapour donor clamp, the
  bulk-BL kick on no-turbulence configs, … .  Residual by construction so
  the five physics rows always sum to the PhysicsOutput totals.
* ``clips``         — the state-update positivity floors (q ≥ 0 across all
  species), the energy-consistent moisture clip, and the operator-split
  tail (saturation adjustment / moisture fixer / q_v smoothing floor).
  Positive water = floors CREATED water.
* ``dynamics``      — the dycore step's column-store delta (includes the
  dry-mass fixer's p_s adjustment and, with ``advect_moisture``, the
  flux-form tracer transport).

All helpers are pure ``jnp`` (JIT/scan-safe, no host transfers).  Global
means are UNWEIGHTED column means (documented limitation: exact closure
holds for any weighting since every row uses the same mean; on the
cubed-sphere production grid cells are near-equal-area so the numbers read
as global means to a few %).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.diagnostics.column_integrals import column_mass_integral

# Fixed row order — the (n_process, 2) ledger arrays index rows by this
# tuple and columns by ``LEDGER_COLUMNS``.
LEDGER_PROCESSES = (
    "turbulence",
    "convection",
    "microphysics",
    "radiation",
    "other_physics",
    "clips",
    "dynamics",
)
N_LEDGER = len(LEDGER_PROCESSES)
LEDGER_COLUMNS = ("water", "energy")  # [kg/m²/s], [W/m²]

# Row indices (module constants so instrumentation sites cannot drift
# against the tuple order).
ROW_TURBULENCE = LEDGER_PROCESSES.index("turbulence")
ROW_CONVECTION = LEDGER_PROCESSES.index("convection")
ROW_MICROPHYSICS = LEDGER_PROCESSES.index("microphysics")
ROW_RADIATION = LEDGER_PROCESSES.index("radiation")
ROW_OTHER = LEDGER_PROCESSES.index("other_physics")
ROW_CLIPS = LEDGER_PROCESSES.index("clips")
ROW_DYNAMICS = LEDGER_PROCESSES.index("dynamics")


def ledger_entry(dq_total_dt, dT_dt, p_s, dsigma):
    """One ledger row ``[water, energy]`` from per-level tendencies.

    Parameters
    ----------
    dq_total_dt : jax.Array or None
        Sum of the process's water-species tendencies [kg/kg/s]
        (..., nlev).  ``None`` ⇒ water row 0 (e.g. radiation).
    dT_dt : jax.Array or None
        The process's temperature tendency [K/s] (..., nlev).
        ``None`` ⇒ energy row 0.
    p_s : jax.Array
        Surface pressure [Pa] (...).
    dsigma : array-like
        Sigma layer thicknesses (nlev,).

    Returns
    -------
    jax.Array, shape (2,)
        ``[global-mean column water rate kg/m²/s,
           global-mean column dry-enthalpy rate W/m²]``.
    """
    zero = jnp.zeros((), dtype=jnp.result_type(p_s.dtype, jnp.float32))
    water = (
        jnp.mean(column_mass_integral(dq_total_dt, p_s, dsigma))
        if dq_total_dt is not None else zero
    )
    energy = (
        constants.c_pd * jnp.mean(column_mass_integral(dT_dt, p_s, dsigma))
        if dT_dt is not None else zero
    )
    return jnp.stack([water, energy]).astype(water.dtype)


def column_store_snapshot(p_s, dsigma, T, *species):
    """Snapshot ``[water_store, enthalpy_store]`` global means.

    ``water_store`` = mean ∫(Σ species) dp/g [kg/m²]; ``enthalpy_store`` =
    c_pd · mean ∫T dp/g [J/m²].  ``None`` species are skipped.  Pair two
    snapshots as ``(after − before)/dt`` to fill the ``dynamics`` /
    ``clips`` rows.
    """
    total_q = None
    for s in species:
        if s is None:
            continue
        total_q = s if total_q is None else total_q + s
    water = (
        jnp.mean(column_mass_integral(total_q, p_s, dsigma))
        if total_q is not None else jnp.zeros(())
    )
    enthalpy = constants.c_pd * jnp.mean(column_mass_integral(T, p_s, dsigma))
    return jnp.stack([water, jnp.asarray(enthalpy, dtype=water.dtype)])


def zero_ledger(dtype=jnp.float64):
    """A zeros ``(N_LEDGER, 2)`` ledger array (accumulator seed)."""
    return jnp.zeros((N_LEDGER, 2), dtype=dtype)


def format_ledger_table(rates, header="Per-process column budget ledger"):
    """Human-readable table from an ``(N_LEDGER, 2)`` array of MEAN RATES
    (water [kg/m²/s], energy [W/m²]) — host-side (NumPy) only."""
    import numpy as np
    r = np.asarray(rates)
    lines = [header, "=" * 58,
             f"  {'process':<14s} {'water [mm/day]':>16s} {'energy [W/m²]':>14s}"]
    for i, name in enumerate(LEDGER_PROCESSES):
        lines.append(
            f"  {name:<14s} {r[i, 0] * 86400.0:>+16.4f} {r[i, 1]:>+14.3f}")
    tot = r.sum(axis=0)
    lines.append("  " + "-" * 46)
    lines.append(
        f"  {'TOTAL':<14s} {tot[0] * 86400.0:>+16.4f} {tot[1]:>+14.3f}")
    return "\n".join(lines)
