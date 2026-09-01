"""Advective bottom boundary layer on the MPAS Voronoi mesh (NEMO trabbl
``nn_bbl_adv=2``) — the edge-indexed port of :mod:`.bbl_adv`.

Same physics, same NEMO trabbl formula and 3-leg exactly-conserving exchange
as the structured lat-lon/tripole implementation; the ONLY difference is the
face bookkeeping: a structured grid enumerates i/j faces by array slicing,
the Voronoi mesh enumerates edges by ``cellsOnEdge`` gathers (one unambiguous
cell pair per edge; the icosahedral sphere is closed, so every edge has two
real cells and land is handled by the mask).

Discretization choices (GLM design review 2026-09-01, pinned by tests):
  * face WIDTH = ``dvEdge`` (the transverse extent of the face between the
    two cells — NEMO's ``e2u``/``e1v`` analogue).  ``dcEdge`` is the
    cell-centre spacing and plays NO role here: the trabbl transport formula
    ``tr = width * e3_bbl * g*gamma * max(0, zgdrho)`` has no along-slope
    distance in it (the slope enters only through the SIGN/selection).
  * down-slope direction from the STATIC reference bottom depth (mid-cell
    cumulative ``h_ref`` depth), exactly like the structured version — NEMO
    also builds mgrh once in ``tra_bbl_init`` from gdept_0.
  * alpha/beta at each column's OWN bottom pressure, averaged over the edge
    (thermobaricity preserved — same eos_rab gating as the structured code).

Reference: Campin & Goosse (1999) Tellus; NEMO 5.0.1 TRA/trabbl.F90.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants

__physics_contract__ = {
    "summary": (
        "Advective bottom boundary layer (Campin & Goosse 1999; NEMO trabbl "
        "nn_bbl_adv=2) on the MPAS Voronoi mesh: edge-indexed port of "
        "bbl_adv.py — down-slope transport tr = dvEdge*e3_bbl*g*gamma*"
        "max(0, zgdrho) with the same exactly-conserving 3-leg exchange."
    ),
    "inputs": {
        "T": "degC", "S": "PSU", "h_ref": "m", "land_mask": "1",
        "area": "m^2", "dv_edge": "m", "gamma_s": "s", "rho_0": "kg/m^3",
        "dt": "s",
    },
    "outputs": {"tr_bbl": "m^3/s", "dT_dt": "degC/s", "dS_dt": "PSU/s"},
    "sign_convention": (
        "Transport signed +1 when cell 2 of the edge is the deeper column "
        "(mgrh convention); non-zero only when the shelf bottom is DENSER."
    ),
    "conserves": ["tracer", "salt"],
    "differentiable": True,
    "reference": (
        "Campin & Goosse (1999) Tellus 51A 412-430; NEMO 5.0.1 "
        "TRA/trabbl.F90 (ORCA1: nn_bbl_adv=2, rn_gambbl=20 s); "
        "structured twin: ocean/physics/bbl_adv.py"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_bbl_adv_mpas.py: two-cell analytic overflow "
        "matching the structured implementation's transport exactly, "
        "sign-flip ridge, exact conservation on a random staircase, "
        "flat-bottom/land inactivity."
    ),
}


class BBLGeometryMPAS(NamedTuple):
    """Static edge-indexed BBL geometry (NEMO tra_bbl_init on Voronoi)."""
    mgrh: jnp.ndarray      # (nEdges,) +1 deeper at cell2, -1 deeper at cell1, 0 flat
    k_s: jnp.ndarray       # (nEdges,) SHELF (shallow) bottom level index
    k_d: jnp.ndarray       # (nEdges,) DEEP bottom level index
    e3_bbl: jnp.ndarray    # (nEdges,) min bottom thickness of the 2 columns
    dep_bot: jnp.ndarray   # (nCells,) bottom mid-cell depth [m]
    active: jnp.ndarray    # (nEdges,) both-wet AND sloped (float 0/1)
    bot_k: jnp.ndarray     # (nCells,) bottom level index
    h_ref: jnp.ndarray     # (nCells, nlev) reference thicknesses
    c1: jnp.ndarray        # (nEdges,) cell 1 index
    c2: jnp.ndarray        # (nEdges,) cell 2 index


def bbl_static_geometry_mpas(h_ref: jnp.ndarray, land_mask: jnp.ndarray,
                             cells_on_edge: jnp.ndarray) -> BBLGeometryMPAS:
    """Static geometry from per-cell reference thicknesses ``h_ref``
    (nCells, nlev; 0 below the seafloor — partial-cell aware), the ocean
    ``land_mask`` (nCells,), and ``cellsOnEdge`` (2, nEdges)."""
    h = jnp.asarray(h_ref, dtype=jnp.float64)
    mask = jnp.asarray(land_mask, dtype=jnp.float64)
    c1 = jnp.asarray(cells_on_edge[0], dtype=jnp.int32)
    c2 = jnp.asarray(cells_on_edge[1], dtype=jnp.int32)

    wet3 = h > 1.0e-3  # coeff-ok: wet-cell thickness floor [m]
    n_active = jnp.sum(wet3.astype(jnp.int32), axis=-1)
    bot_k = jnp.maximum(n_active - 1, 0)
    cum = jnp.cumsum(h, axis=-1)
    z_mid = cum - 0.5 * h
    dep_bot = jnp.take_along_axis(z_mid, bot_k[:, None], axis=-1)[:, 0]
    e3_bot = jnp.take_along_axis(h, bot_k[:, None], axis=-1)[:, 0]

    d1, d2 = dep_bot[c1], dep_bot[c2]
    mgrh = jnp.sign(d2 - d1)
    k_s = jnp.where(mgrh >= 0, bot_k[c1], bot_k[c2])
    k_d = jnp.maximum(bot_k[c1], bot_k[c2])
    e3_bbl = jnp.minimum(e3_bot[c1], e3_bot[c2])
    active = ((mask[c1] > 0.5) & (mask[c2] > 0.5)
              & (mgrh != 0)).astype(jnp.float64)

    return BBLGeometryMPAS(
        mgrh=mgrh, k_s=k_s.astype(jnp.int32), k_d=k_d.astype(jnp.int32),
        e3_bbl=e3_bbl, dep_bot=dep_bot, active=active,
        bot_k=bot_k.astype(jnp.int32), h_ref=h, c1=c1, c2=c2,
    )


def bbl_transports_mpas(T: jnp.ndarray, S: jnp.ndarray,
                        geom: BBLGeometryMPAS, dv_edge: jnp.ndarray, *,
                        gamma_s: float, rho_0: float) -> jnp.ndarray:
    """Campin-Goosse down-slope transport per edge [m^3/s] — the same
    formula and eos_rab gating as :func:`.bbl_adv.bbl_transports`."""
    from legoesm.ocean.eos import (
        haline_contraction_coeff, thermal_expansion_coeff,
    )
    g_gamma = constants.g * gamma_s

    Tb = jnp.take_along_axis(T, geom.bot_k[:, None], axis=-1)[:, 0]
    Sb = jnp.take_along_axis(S, geom.bot_k[:, None], axis=-1)[:, 0]
    p_bot = rho_0 * constants.g * geom.dep_bot
    alpha = thermal_expansion_coeff(Tb, Sb, p_bot)
    beta = haline_contraction_coeff(Tb, Sb, p_bot)

    c1, c2, mgrh = geom.c1, geom.c2, geom.mgrh
    a_bar = 0.5 * (alpha[c1] + alpha[c2])
    b_bar = 0.5 * (beta[c1] + beta[c2])
    T_sh = jnp.where(mgrh >= 0, Tb[c1], Tb[c2])
    T_dp = jnp.where(mgrh >= 0, Tb[c2], Tb[c1])
    S_sh = jnp.where(mgrh >= 0, Sb[c1], Sb[c2])
    S_dp = jnp.where(mgrh >= 0, Sb[c2], Sb[c1])
    zgdrho = jnp.maximum(a_bar * (T_dp - T_sh) - b_bar * (S_dp - S_sh), 0.0)
    return jnp.asarray(dv_edge, dtype=T.dtype) * geom.e3_bbl * g_gamma \
        * zgdrho * mgrh * geom.active


def apply_bbl_adv_tendency_mpas(dT_dt, dS_dt, T, S, h_k, area,
                                geom: BBLGeometryMPAS, tr, *, nlev: int):
    """NEMO ``tra_bbl_adv`` 3-leg exchange on cells — ``.at[].add`` scatter
    over the edge's two cells; ``nlev`` static (unrolled loop)."""
    a3 = area[:, None] * jnp.maximum(h_k, 1.0e-3)  # coeff-ok: thickness floor [m]
    inv_v = 1.0 / a3

    c1, c2 = geom.c1, geom.c2
    zu = jnp.abs(tr) * geom.active
    up = geom.mgrh >= 0                     # True: cell1 shelf, cell2 deep
    sh = jnp.where(up, c1, c2)              # (nEdges,) shelf cell index
    dp = jnp.where(up, c2, c1)              # (nEdges,) deep cell index
    ks_, kd_ = geom.k_s, geom.k_d

    def _apply(dpt, pt):
        pt_sh = pt[sh]                      # (nEdges, nlev)
        pt_dp = pt[dp]
        iv_sh = inv_v[sh]
        iv_dp = inv_v[dp]
        pt_sh_b = jnp.take_along_axis(pt_sh, ks_[:, None], axis=-1)[:, 0]
        pt_dp_at_ks = jnp.take_along_axis(pt_dp, ks_[:, None], axis=-1)[:, 0]
        pt_dp_b = jnp.take_along_axis(pt_dp, kd_[:, None], axis=-1)[:, 0]

        # (1) shelf bottom: exchange with the deep column at shelf level
        ztra_sh = zu * (pt_dp_at_ks - pt_sh_b) * jnp.take_along_axis(
            iv_sh, ks_[:, None], axis=-1)[:, 0]
        dpt = dpt.at[sh, ks_].add(ztra_sh.astype(dpt.dtype))
        # (2) deep-column upward return flow ks <= k < kd (static unroll)
        for k in range(nlev - 1):
            in_rng = ((k >= ks_) & (k < kd_)).astype(pt.dtype)
            ztra = zu * in_rng * (pt_dp[:, k + 1] - pt_dp[:, k]) * iv_dp[:, k]
            dpt = dpt.at[dp, k].add(ztra.astype(dpt.dtype))
        # (3) deep bottom: receives the shelf bottom water
        ztra_dp = zu * (pt_sh_b - pt_dp_b) * jnp.take_along_axis(
            iv_dp, kd_[:, None], axis=-1)[:, 0]
        dpt = dpt.at[dp, kd_].add(ztra_dp.astype(dpt.dtype))
        return dpt

    return _apply(dT_dt, T), _apply(dS_dt, S)


def apply_bbl_adv_step_mpas(state, geom: BBLGeometryMPAS, dt: float, *,
                            gamma_s: float, rho_0: float, area, dv_edge,
                            nlev: int):
    """HOST post-step BBL application on the Voronoi mesh — the exact twin
    of :func:`.bbl_adv.apply_bbl_adv_step` (recompute transports from the
    CURRENT bottom T/S, cap the exchange fraction, forward-Euler step)."""
    T = jnp.asarray(state.T.data, dtype=jnp.float64)
    S = jnp.asarray(state.S.data, dtype=jnp.float64)
    h_k = geom.h_ref
    tr = bbl_transports_mpas(T, S, geom, dv_edge,
                             gamma_s=gamma_s, rho_0=rho_0)
    # Exchange cap — the structured twin's 0.25*V_min/dt bound, additionally
    # divided by the number of ACTIVE incident edges on the busier of the
    # edge's two cells: a hexagonal cell can receive ~6 converging edges,
    # and 6 x 0.25 = 1.5 cell volumes per step would destabilize (codex
    # 2026-09-01 MAJOR). Inactive at production scales.
    area_ = jnp.asarray(area, dtype=jnp.float64)
    e3_bot = jnp.take_along_axis(h_k, geom.bot_k[:, None], axis=-1)[:, 0]
    V_bot = area_ * jnp.maximum(e3_bot, 1.0e-3)  # coeff-ok: thickness floor [m]
    n_inc = jnp.zeros(area_.shape[0], dtype=jnp.float64)
    n_inc = n_inc.at[geom.c1].add(geom.active).at[geom.c2].add(geom.active)
    div = jnp.maximum(jnp.maximum(n_inc[geom.c1], n_inc[geom.c2]), 1.0)
    cap = 0.25 * jnp.minimum(V_bot[geom.c1], V_bot[geom.c2]) / (dt * div)
    tr = jnp.sign(tr) * jnp.minimum(jnp.abs(tr), cap)
    dT, dS = apply_bbl_adv_tendency_mpas(
        jnp.zeros_like(T), jnp.zeros_like(S), T, S, h_k, area_, geom, tr,
        nlev=nlev)
    T_new = (T + dt * dT).astype(state.T.data.dtype)
    S_new = (S + dt * dS).astype(state.S.data.dtype)
    return state._replace(
        T=state.T.replace(data=T_new),
        S=state.S.replace(data=S_new),
    )


__all__ = [
    "BBLGeometryMPAS",
    "apply_bbl_adv_step_mpas",
    "apply_bbl_adv_tendency_mpas",
    "bbl_static_geometry_mpas",
    "bbl_transports_mpas",
]
