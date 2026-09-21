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

#: Water MASS species [kg/kg] the ledger's water column sums over — the FV
#: reference list (compiled_segments.py ``_led_q_names``).  NUMBER
#: concentrations (N_c/N_i/N_r/N_s/N_g, [#/kg] or [#/m³]) are DELIBERATELY
#: excluded: adding them to a [kg/kg] water budget is a units error (the same
#: exclusion whose omission produced the 2026-07 conserving-borrow defect).
LEDGER_WATER_SPECIES = ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g")

# Row indices (module constants so instrumentation sites cannot drift
# against the tuple order).
ROW_TURBULENCE = LEDGER_PROCESSES.index("turbulence")
ROW_CONVECTION = LEDGER_PROCESSES.index("convection")
ROW_MICROPHYSICS = LEDGER_PROCESSES.index("microphysics")
ROW_RADIATION = LEDGER_PROCESSES.index("radiation")
ROW_OTHER = LEDGER_PROCESSES.index("other_physics")
ROW_CLIPS = LEDGER_PROCESSES.index("clips")
ROW_DYNAMICS = LEDGER_PROCESSES.index("dynamics")


def sigma_band_weight(sigma_half, sigma_lo, sigma_hi):
    """Per-layer weight (nlev,) for the sigma band ``[sigma_lo, sigma_hi]``.

    A layer straddling a band edge gets the FRACTION of its thickness inside
    the band, so the band boundary need not land on a layer interface and the
    weights of two adjacent bands sum to one everywhere. Sigma increases
    downward, so ``sigma_lo`` is the upper (lower-pressure) edge.

    numpy/jnp arithmetic only; the mesh and coordinate are static, so this is
    evaluated once at setup and never inside a traced region.
    """
    sh = jnp.asarray(sigma_half)
    if sh.ndim != 1 or sh.shape[0] < 2:
        raise ValueError(f"sigma_half must be 1-D with >=2 entries, got {sh.shape}")
    if not (0.0 <= sigma_lo < sigma_hi <= 1.0):
        raise ValueError(
            f"need 0 <= sigma_lo < sigma_hi <= 1, got ({sigma_lo}, {sigma_hi})")
    top, bot = sh[:-1], sh[1:]
    overlap = jnp.clip(jnp.minimum(bot, sigma_hi) - jnp.maximum(top, sigma_lo),
                       0.0, None)
    return overlap / jnp.maximum(bot - top, 1e-30)


def apply_level_weight(field, level_weight):
    """Mask a (..., nlev) tendency to a vertical band before integrating.

    WHY THIS EXISTS: the column ledger integrates over the WHOLE column, which
    makes it structurally blind to a vertical-REDISTRIBUTION bias. Measured on
    the production AMIP, convection's column water row is exactly zero — correct
    for a scheme that moves water up and down without removing it — while the
    tropical free troposphere is twice as moist as observed. The column budget
    closes and says nothing about the defect.

    Restricting the integral to a band answers "which process supplies THIS
    layer" without changing the accumulator's shape, so the hot-loop state that
    carries it is untouched. ``None`` is the byte-identical full-column default.

    ``level_weight`` is a (nlev,) array of per-layer weights, normally 1 inside
    the band and 0 outside; fractional values at the band edges are honoured so
    a band boundary need not fall on a layer interface.
    """
    if level_weight is None or field is None:
        return field
    w = jnp.asarray(level_weight, dtype=jnp.asarray(field).dtype)
    if w.shape[-1] != jnp.asarray(field).shape[-1]:
        raise ValueError(
            f"level_weight has {w.shape[-1]} levels but the tendency has "
            f"{jnp.asarray(field).shape[-1]}")
    return field * w


def ledger_entry_column(dq_total_dt, dT_dt, p_s, dsigma, dp=None,
                        level_weight=None):
    """PER-COLUMN ledger row ``[water, energy]`` — no global reduction.

    Same quantity and sign convention as :func:`ledger_entry`, but keeping
    the horizontal dimension.  This is the form the MPAS lean loop needs:
    a global mean cannot answer "which term sustains THIS column" when the
    event of interest is one cell in 10242 (the 2026-07-31 day-311 / day-315
    single-grid-point detonations), because the signal is ~1e-4 of the
    global mean.  Reduce to the global row afterwards with
    :func:`reduce_ledger_global`, which takes AREA WEIGHTS — on the SCVT
    mesh ``areaCell`` varies by ~47% (measured max/min 1.471), so an
    unweighted mean is not a global mean.

    Parameters
    ----------
    dq_total_dt : jax.Array or None
        Sum of the process's water-species tendencies [kg/kg/s]
        (..., nlev).  ``None`` ⇒ water column 0 (e.g. radiation).
    dT_dt : jax.Array or None
        The process's temperature tendency [K/s] (..., nlev).
        ``None`` ⇒ energy column 0.
    p_s : jax.Array
        Surface pressure [Pa] (...).
    dsigma : array-like
        Sigma layer thicknesses (nlev,).  Used only when ``dp`` is None.
    dp : jax.Array, optional
        Layer pressure thickness [Pa].  REQUIRED for correctness on a hybrid
        column (#1400): the ``p_s * dsigma`` form is wrong by ``dA*(p_s -
        p_ref)`` there.  Pass ``VerticalCoordProtocol.layer_thickness_dp``.

    Returns
    -------
    jax.Array, shape (..., 2)
        ``[column water rate kg/m²/s, column dry-enthalpy rate W/m²]``
        per column.
    """
    ref = jnp.asarray(p_s)
    zero = jnp.zeros(ref.shape, dtype=jnp.result_type(ref.dtype, jnp.float32))
    _dq = apply_level_weight(dq_total_dt, level_weight)
    _dT = apply_level_weight(dT_dt, level_weight)
    water = (column_mass_integral(_dq, p_s, dsigma, dp=dp)
             if _dq is not None else zero)
    energy = (constants.c_pd
              * column_mass_integral(_dT, p_s, dsigma, dp=dp)
              if _dT is not None else zero)
    return jnp.stack([water, jnp.asarray(energy, dtype=water.dtype)], axis=-1)


def column_store_snapshot_column(p_s, dsigma, T, *species, dp=None,
                                 level_weight=None):
    """PER-COLUMN ``[water_store, enthalpy_store]`` — no global reduction.

    Per-column analogue of :func:`column_store_snapshot`; pair two snapshots
    as ``(after − before)/dt`` to fill the ``dynamics`` / ``clips`` rows of a
    per-column ledger.  ``None`` species are skipped.
    """
    total_q = None
    for s in species:
        if s is None:
            continue
        total_q = s if total_q is None else total_q + s
    ref = jnp.asarray(p_s)
    total_q = apply_level_weight(total_q, level_weight)
    water = (column_mass_integral(total_q, p_s, dsigma, dp=dp)
             if total_q is not None
             else jnp.zeros(ref.shape,
                            dtype=jnp.result_type(ref.dtype, jnp.float32)))
    enthalpy = constants.c_pd * column_mass_integral(
        apply_level_weight(T, level_weight), p_s, dsigma, dp=dp)
    return jnp.stack([water, jnp.asarray(enthalpy, dtype=water.dtype)],
                     axis=-1)


# ---------------------------------------------------------------------------
# TOTAL energy (#1354)
# ---------------------------------------------------------------------------
#
# The ledger's ``energy`` column is DRY ENTHALPY, ``c_pd * int T dp/g``.  That
# is the right quantity for attributing HEATING, and the wrong one for
# attributing a LEAK, because dry enthalpy is not conserved by adiabatic
# dynamics: a perfectly conserving core converts enthalpy into geopotential
# and kinetic energy all the time, and every joule of that conversion lands in
# the ``dynamics`` row -- which is computed as a store delta and therefore
# cannot tell conversion from leakage.  Measured on one level-5 artifact
# (#1354): the dynamics row reads 33 W/m^2 with no leak required to explain it.
#
# The helpers below supply the column the attribution actually needs.  For a
# HYDROSTATIC atmosphere the column total energy is
#
#     TE = (1/g) int_0^{p_s} ( c_p T + 1/2 |v|^2 + L_v q_v - L_f q_ice ) dp
#          + Phi_s p_s / g
#
# The surface term and the c_p (rather than c_v) are the same statement: the
# geopotential integral ``int Phi dp`` is integrated by parts using the
# hydrostatic relation, giving ``Phi_s p_s + int R_d T dp``, and
# ``c_v + R_d = c_p``.  So the ``c_pd * int T dp/g`` the ledger already
# computes is exactly the internal-plus-geopotential part, and what is MISSING
# from it is kinetic energy, the latent terms, and the surface geopotential
# term that moves when the mass fixer moves ``p_s``.


def column_total_energy(p_s, dsigma, T, *, u=None, v=None, phis=None,
                        q_v=None, q_ice=None, dp=None, level_weight=None):
    """PER-COLUMN total energy [J/m^2] for a hydrostatic column (#1354).

    Parameters mirror :func:`column_store_snapshot_column`.  Every optional
    term defaults to absent rather than zero-filled, so a caller that has no
    winds gets the enthalpy+latent energy and knows it: ``None`` means "this
    term was not supplied", and the docstring of whatever quotes the number
    has to say which terms were in it.

    ``phis`` is the SURFACE GEOPOTENTIAL [m^2/s^2]; it enters as
    ``phis * p_s / g``, the integrated-by-parts boundary term, and it is the
    term that makes the total energy respond to a mass-fixer ``p_s``
    adjustment.  Omitting it over flat ground costs nothing; omitting it over
    topography does not.
    """
    ref = jnp.asarray(p_s)
    acc = jnp.result_type(ref.dtype, jnp.float32)

    def _col(x, scale):
        if x is None:
            return jnp.zeros(ref.shape, dtype=acc)
        return scale * column_mass_integral(
            apply_level_weight(x, level_weight), p_s, dsigma, dp=dp)

    total = _col(T, constants.c_pd)
    if u is not None or v is not None:
        ke = None
        for w in (u, v):
            if w is None:
                continue
            ke = w * w if ke is None else ke + w * w
        total = total + _col(ke, 0.5)
    total = total + _col(q_v, constants.L_v) - _col(q_ice, constants.L_f)
    if phis is not None:
        total = total + jnp.asarray(phis) * ref / constants.g
    return jnp.asarray(total, dtype=acc)


def total_energy_entry_column(p_s, dsigma, *, dT_dt=None, du_dt=None,
                              dv_dt=None, u=None, v=None, dq_v_dt=None,
                              dq_ice_dt=None, dp_s_dt=None, phis=None,
                              dp=None, level_weight=None):
    """PER-COLUMN total-energy rate [W/m^2] for ONE process (#1354).

    The time derivative of :func:`column_total_energy` holding the layer-mass
    weights fixed, plus the surface term's own rate:

        dTE/dt = c_p int dT/dt dp/g
               + int (u du/dt + v dv/dt) dp/g
               + L_v int dq_v/dt dp/g  -  L_f int dq_ice/dt dp/g
               + Phi_s (dp_s/dt) / g

    The kinetic term needs the CURRENT winds as well as their tendency --
    d/dt(|v|^2/2) = v . dv/dt -- so a process that returns a momentum
    tendency must hand over the state it was computed against, not just the
    tendency.  A process with no momentum tendency contributes nothing there,
    which is the common case and is why ``u``/``v`` are optional.

    Holding the layer masses fixed is exact for every physics process (they
    do not move ``p_s``) and is NOT exact for the dycore, which is why the
    dynamics row must be filled from a pair of
    :func:`column_total_energy` snapshots rather than from this function.
    """
    ref = jnp.asarray(p_s)
    acc = jnp.result_type(ref.dtype, jnp.float32)
    rate = jnp.zeros(ref.shape, dtype=acc)

    def _col(x, scale):
        if x is None:
            return jnp.zeros(ref.shape, dtype=acc)
        return scale * column_mass_integral(
            apply_level_weight(x, level_weight), p_s, dsigma, dp=dp)

    rate = rate + _col(dT_dt, constants.c_pd)
    dke = None
    for w, dw in ((u, du_dt), (v, dv_dt)):
        if w is None or dw is None:
            continue
        dke = w * dw if dke is None else dke + w * dw
    rate = rate + _col(dke, 1.0)
    rate = rate + _col(dq_v_dt, constants.L_v) - _col(dq_ice_dt, constants.L_f)
    if phis is not None and dp_s_dt is not None:
        rate = rate + jnp.asarray(phis) * jnp.asarray(dp_s_dt) / constants.g
    return jnp.asarray(rate, dtype=acc)


def zero_ledger_column(n_columns, dtype=jnp.float64):
    """A zeros ``(n_columns, N_LEDGER, 2)`` per-column accumulator seed."""
    return jnp.zeros((n_columns, N_LEDGER, 2), dtype=dtype)


def reduce_ledger_global(ledger_column, area=None):
    """Reduce a per-column ledger ``(ncol, N_LEDGER, 2)`` to ``(N_LEDGER, 2)``.

    ``area`` (ncol,) supplies AREA WEIGHTS; the result is
    ``sum(row*area)/sum(area)``.  ``None`` falls back to an unweighted mean
    and is correct ONLY on an equal-area mesh — on the production SCVT mesh
    it is not (areaCell max/min = 1.471), so pass ``mesh.areaCell``.
    """
    if area is None:
        return jnp.mean(ledger_column, axis=0)
    w = jnp.asarray(area, dtype=ledger_column.dtype)
    return (jnp.sum(ledger_column * w[:, None, None], axis=0)
            / jnp.sum(w))


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
