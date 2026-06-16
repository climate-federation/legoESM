"""Kessler warm-rain microphysics as an MPAS operator-split physics forcing.

Adapter that applies the shared column-local Kessler scheme
(:func:`legoesm.atmosphere.physics.microphysics.kessler.kessler_microphysics`)
to an MPAS / Voronoi hydrostatic state and returns physics tendencies in the
operator-split ``physics_fn(state, mesh, sigma_coord, *, phys_state, forcing)``
convention consumed by ``MPASPrimitiveEquationModel._step_jit`` (single rank)
and ``make_voronoi_mpi_step`` (multi rank).  No radiation, no convection — this
is the moisture + condensation forcing for a *moist baroclinic wave*.

Why an adapter and not a new scheme: Kessler itself
(``physics/microphysics/kessler.py``) is purely COLUMN-local — it operates on
``(ncol, nlev)`` arrays with no horizontal coupling.  So this wrapper only has
to (1) pull ``q_v``/``q_c``/``q_r`` out of ``state.tracers``, (2) build the
column thermodynamic inputs Kessler needs (``p_full``, ``p_half``, ``rho``,
``dz``) from the SHARED helpers in ``atmosphere.physics._shared`` and
``grids.vertical`` (nothing re-derived), and (3) repackage the column output as
``MPASHydrostaticTendencies``.  Because the physics is column-local, ``mesh`` is
accepted for signature compliance but unused, and the forcing is correct
per-rank under MPI with NO extra halo exchange — the dycore already
halo-exchanges and advects the tracers mass-consistently.

Operator-split timing: the MPAS step evaluates ``physics_fn`` once on the
post-dynamics state and applies the tendencies forward over ``dt``
(``state += dt · tendency``; see ``_step_jit``).  Kessler defines its
saturation adjustment as an increment/``dt`` rate, so a single forward
application over the SAME ``dt`` recovers the intended adjustment.  ``dt`` is
not part of the ``physics_fn`` signature, so it is bound at build time via
:func:`make_kessler_forcing_mpas`.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASHydrostaticTendencies
from legoesm.grids.vertical import (
    HybridSigmaPressureCoordinate,
    pressure_from_sigma,
)
from legoesm.atmosphere.physics._shared import compute_layer_dz, compute_rho
from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.output import make_zero_hydrometeors

_DIMS_CELL = ("nCells", "level")
_DIMS_EDGE = ("nEdges", "level")
_DIMS_CELL_2D = ("nCells",)
_REQUIRED_TRACERS = ("q_v", "q_c", "q_r")


def make_kessler_forcing_mpas(dt, config: KesslerConfig | None = None):
    """Build an operator-split MPAS ``physics_fn`` applying Kessler over ``dt``.

    Parameters
    ----------
    dt : float
        Physics step [s].  Bound into the returned closure because the
        dycore's ``physics_fn`` convention passes no timestep.  Must be > 0.

        CONTRACT: the MPAS step applies these tendencies forward with the step's
        OWN ``dt`` (``state += dt_step * tendency``).  Kessler's saturation
        adjustment is an increment/``dt``-here rate, so the caller MUST step the
        model with the SAME ``dt`` used here, or the adjustment is mis-scaled by
        ``dt_step / dt_here``.  (The non-saturation processes are true
        dt-independent rates and are unaffected.)  The benchmark builds and
        steps with one ``dt`` per case, so they always match; the returned
        callable carries ``_bound_dt`` so a wrapper can assert this.
    config : KesslerConfig, optional
        Microphysics configuration (defaults to ``KesslerConfig()``).

    Returns
    -------
    callable
        ``physics_fn(state, mesh, sigma_coord, *, phys_state=None,
        forcing=None) -> MPASHydrostaticTendencies`` carrying ``dT_dt`` (latent
        heating) and ``tracer_tendencies`` for ``q_v``/``q_c``/``q_r``.  Wind
        and surface-pressure tendencies are zero (warm-rain microphysics has no
        direct momentum or surface-pressure source).
    """
    cfg = config if config is not None else KesslerConfig()
    if not (dt > 0.0):
        raise ValueError(f"kessler forcing dt must be > 0, got {dt!r}")
    dt = float(dt)

    def kessler_forcing_mpas(
        state, mesh, sigma_coord, *, phys_state=None, forcing=None,
    ):
        if state.tracers is None or any(
            k not in state.tracers for k in _REQUIRED_TRACERS
        ):
            have = None if state.tracers is None else sorted(state.tracers)
            raise ValueError(
                "kessler_forcing_mpas requires state.tracers with keys "
                f"{_REQUIRED_TRACERS}; got {have}. Initialize the state with "
                "baroclinic_wave_init_mpas(..., moist=True) (or attach the "
                "tracers before stepping)."
            )
        if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
            raise NotImplementedError(
                "kessler_forcing_mpas supports sigma coordinates only "
                "(half-level hybrid pressure is not wired); got "
                f"{type(sigma_coord).__name__}."
            )

        T = state.T.data                  # (nCells, nlev)
        p_s = state.p_s.data              # (nCells,)
        # Tracer advection is NOT positive-definite and the dycore clamps
        # tracers to >=0 only AFTER physics (primitive_eq_mpas.py / voronoi_mpi.py
        # apply the floor post-physics).  So a tiny post-advection undershoot
        # could feed a negative q_c into Kessler's accretion (kessler.py uses
        # raw q_c), reversing the process.  Clip to non-negative HERE before the
        # column scheme reads them (codex review).  These are diagnostics-only
        # clips on the physics INPUT; the prognostic floor still happens in the
        # dycore after the tendencies are applied.
        q_v = jnp.maximum(state.tracers["q_v"].data, 0.0)
        q_c = jnp.maximum(state.tracers["q_c"].data, 0.0)
        q_r = jnp.maximum(state.tracers["q_r"].data, 0.0)

        # Column thermodynamic inputs from the SHARED helpers (TOA-first
        # half levels per SigmaCoordinate convention).
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)   # (nCells, nlev)
        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)   # (nCells, nlev+1)
        rho = compute_rho(T, p_full, q_v)        # moist ideal-gas density
        dz = compute_layer_dz(T, p_half, q_v)    # moist hypsometric thickness

        ncol, nlev = T.shape
        hydro = make_zero_hydrometeors(ncol, nlev, dtype=T.dtype)._replace(
            q_c=q_c, q_r=q_r,
        )
        out = kessler_microphysics(
            T=T, q_v=q_v, hydrometeors=hydro,
            p_full=p_full, p_half=p_half, rho=rho, dz=dz,
            dt=dt, config=cfg,
        )

        zeros_edge = jnp.zeros_like(state.u.data)
        zeros_ps = jnp.zeros_like(p_s)
        return MPASHydrostaticTendencies(
            du_dt=Field(data=zeros_edge, name="du_dt_kessler",
                        dims=_DIMS_EDGE, units="m/s^2"),
            dT_dt=Field(data=out.dT_dt, name="dT_dt_kessler",
                        dims=_DIMS_CELL, units="K/s"),
            dp_s_dt=Field(data=zeros_ps, name="dp_s_dt_kessler",
                          dims=_DIMS_CELL_2D, units="Pa/s"),
            dphis_dt=Field(data=zeros_ps, name="dphis_dt_kessler",
                           dims=_DIMS_CELL_2D, units="m^2/s^3"),
            tracer_tendencies={
                "q_v": Field(data=out.dq_v_dt, name="dq_v_dt_kessler",
                             dims=_DIMS_CELL, units="kg/kg/s"),
                "q_c": Field(data=out.dq_c_dt, name="dq_c_dt_kessler",
                             dims=_DIMS_CELL, units="kg/kg/s"),
                "q_r": Field(data=out.dq_r_dt, name="dq_r_dt_kessler",
                             dims=_DIMS_CELL, units="kg/kg/s"),
            },
        )

    # Introspectable bound timestep so a step wrapper can assert the caller
    # steps with the same dt the saturation-adjustment rate was scaled by.
    kessler_forcing_mpas._bound_dt = dt
    return kessler_forcing_mpas
