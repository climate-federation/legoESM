"""NEMO ORCA1 tendency-matching comparator (campaign step c; run via SLURM).

Consumes the RUN_TRD hourly trend re-run (``keep_run3_1h_*`` /
``ORCA1_1h_*_trd1h_*``: state + avt/avs + ttrd_zdf/strd_zdf + stress) and
compares the legoESM zdftke card against NEMO **per process, at a matched
state** — no time integration, so the IC/spin-up/seasonal confounds of
state-matching are absent. Two column-local stages (no model construction):

* **Stage A — closure test**: run the grid-agnostic TKE kernel
  (:func:`tke_vertical_mixing`, the ORCA1 card in the quasi-steady
  diagnostic mode) on NEMO's OWN hour-``r-1`` state (T/S, U/V averaged to
  T-points, rho from the NEMO S-EOS, e3t thicknesses, taum stress) and
  compare the resulting tracer diffusivity ``K_H`` against NEMO's ``avt``
  for hour ``r``.  Tests the CLOSURE alone (what the SST campaign tuned).
  Mode-B is an approximation to NEMO's prognostic en at equilibrium —
  expect agreement in pattern/magnitude, not bitwise.
* **Stage B — operator test**: one backward-Euler implicit solve of the
  hour-``r-1`` T/S with NEMO's OWN ``avt``/``avs`` (hour ``r``) and compare
  ``(T' - T)/dt`` against ``ttrd_zdf``/``strd_zdf`` for hour ``r``.  Tests
  the implicit-diffusion OPERATOR with the closure removed.

Record alignment (rn_Dt=3600 -> ONE step per hourly record).  MEASURED
2026-08-12 (``nemo_trend_closure.py`` pairing scan, 24-record RUN_TRD2 file):
the trend at record ``r`` matches the state change ``r -> r+1`` better than
``r-1 -> r`` — T corr 0.952 vs 0.897, S corr 0.891 vs 0.751 — so record
``r+1`` is the POST-operator state for trend ``r``.  Stage B2 anchors there.
Stage A/B keep using state ``r-1`` as the closure input (a diffusivity is a
function of the state it is computed from, not of the pairing).

Caveats carried from the trend-run findings: NEMO ``ttrd_zdf`` under this
key_RK3 build INCLUDES the EVD convective enhancement via ``avt`` (no
separate ttrd_evd), and any isoneutral-K33 implicit contribution rides in
``avt`` too — Stage B uses NEMO's avt so both effects cancel there; Stage A
compares against the same avt, so convection-scheme differences show up as
K disagreement in convecting columns (reported per-region, not hidden).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# The Stage-B implicit solve MUST run in float64 (see _fill); FORCE the JAX
# flag before any jax import (codex: setdefault let a pre-existing
# JAX_ENABLE_X64=0 silently demote jnp.asarray(float64) to f32 -- the exact
# bug class this module just had).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"


REGIONS = (
    ("arctic", 66.0, 90.0),
    ("nh_mid", 30.0, 66.0),
    ("tropics", -15.0, 15.0),
    ("sh_mid", -66.0, -30.0),
    ("antarctic", -90.0, -66.0),
)


def _fill(a):
    a = a.filled(np.nan) if np.ma.isMaskedArray(a) else np.asarray(a)
    # FLOAT64 ALWAYS.  The netCDF variables are float32, and the Stage-B
    # backward-Euler solve has diagonal terms dt*K/(dz*dz_half) ~ 3e5 in EVD
    # columns (avs = 100, dz ~ 1 m); solving that system in float32 loses
    # ~0.1 PSU to roundoff and FABRICATES |dS/dt| ~ 1e-4 PSU/s exactly where
    # EVD fires.  Measured on the worst column (j=153, i=280, rec 1): f32
    # gives +1.25e-4 PSU/s in an already-homogenised block (8000x the
    # gradient-limited bound, max principle violated) while f64 gives -8e-8,
    # the NEMO strd_zdf magnitude.  The '46x EVD-salt mismatch' of job
    # 9208553 was this instrument bug, not a model defect.
    return a.astype(np.float64)


IWM_VARS = ("power_bot", "power_cri", "power_nsq", "power_sho",
            "scale_bot", "scale_cri")


def load_iwm_forcing(path: str):
    """ORCA1's OWN de Lavergne wave-power maps (namzdf_iwm sn_mpb..sn_dsc).

    ORCA1 runs ``ln_zdfiwm = .true.`` (namelist_cfg:434), so NEMO's avt and
    avm both carry a wave-driven diffusivity that a TKE-only probe does not.
    Comparing the two Prandtl numbers without it compares different things.

    The six variable names are the ones the namelist itself lists against
    ``zdfiwm_forcing_TRA.nc``; a missing one RAISES rather than falling back
    to the uniform-power defaults, because that fallback is a different
    physical field and would silently answer a different question.
    """
    import netCDF4 as nc

    ds = nc.Dataset(path)
    out = {}
    for v in IWM_VARS:
        if v not in ds.variables:
            raise KeyError(
                f"{path} has no {v!r}; ORCA1's namzdf_iwm names all of "
                f"{IWM_VARS} in zdfiwm_forcing_TRA.nc. Refusing to substitute "
                "the uniform-power fallback, which is a different field.")
        a = _fill(ds.variables[v][:])
        if a.ndim == 3:
            # monthly climatology (namelist frequency -12). The Mode-A state
            # is NEMO's step-8760 restart = 1 January, so record 0.
            a = a[0]
        out[v] = a.reshape(-1)
    return out


def load_pair(tfile: str, ufile: str, vfile: str, rec: int):
    """State at record ``rec-1`` + target avt/trends at record ``rec``."""
    import netCDF4 as nc

    dsT = nc.Dataset(tfile)
    dsU = nc.Dataset(ufile)
    dsV = nc.Dataset(vfile)
    r0, r1 = rec - 1, rec
    out = {
        "T": _fill(dsT.variables["votemper"][r0]),      # (z, y, x) degC
        "S": _fill(dsT.variables["vosaline"][r0]),
        "e3t": _fill(dsT.variables["e3t"][r0]),          # (z, y, x) m
        "taum": _fill(dsT.variables["taum"][r0]),        # (y, x) N/m2
        "u": _fill(dsU.variables["uoce"][r0]),           # (z, y, x) U-points
        "v": _fill(dsV.variables["voce"][r0]),           # V-points
        "avt": _fill(dsT.variables["avt"][r1]),          # (z+? , y, x) W-points
        "avt_rec0": _fill(dsT.variables["avt"][0]),      # step-8761 avt (A2 target)
        # MOMENTUM diffusivity at the same record as avt. NEMO's own Prandtl
        # number is avm/avt (nn_pdl=1), so carrying both makes the tracer-vs-
        # momentum split of the Stage-A excess a read of NEMO's own output
        # rather than an inference. ORCA1 sets nn_evdm=0, so avm carries no EVD.
        "avm": _fill(dsT.variables["avm"][r1]),
        "avs": _fill(dsT.variables["avs"][r1]),
        "ttrd_zdf": _fill(dsT.variables["ttrd_zdf"][r1]),
        "strd_zdf": _fill(dsT.variables["strd_zdf"][r1]),
        # POST-operator state for the trend at r1 (see the pairing note in
        # run_stage_b): record r1+1, clamped to the last record.
        "T_post": _fill(dsT.variables["votemper"][
            min(r1 + 1, dsT.variables["votemper"].shape[0] - 1)]),
        "S_post": _fill(dsT.variables["vosaline"][
            min(r1 + 1, dsT.variables["vosaline"].shape[0] - 1)]),
        # XIOS one_file splits coords per grid: T fields on nav_*_grid_T,
        # avt/avm on the W grid (identical horizontal positions on eORCA1).
        "lat": _fill(dsT.variables["nav_lat_grid_T"][:]),
        "lon": _fill(dsT.variables["nav_lon_grid_T"][:]),
        "mld": _fill(dsT.variables["mldr10_1"][r0]),
    }
    dsT.close(); dsU.close(); dsV.close()
    return out


def u_to_T(u):
    """U-point -> T-point average along x (NEMO C-grid: u(i) sits between
    T(i) and T(i+1); T(i) gets 0.5*(u(i-1)+u(i)))."""
    uT = np.full_like(u, np.nan)
    uu = np.nan_to_num(u, nan=0.0)
    w = np.isfinite(u).astype(u.dtype)
    num = uu + np.roll(uu, 1, axis=-1)
    den = w + np.roll(w, 1, axis=-1)
    with np.errstate(invalid="ignore"):
        uT = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return uT


def v_to_T(v):
    vT = np.full_like(v, np.nan)
    vv = np.nan_to_num(v, nan=0.0)
    w = np.isfinite(v).astype(v.dtype)
    num = vv + np.roll(vv, 1, axis=-2)
    den = w + np.roll(w, 1, axis=-2)
    with np.errstate(invalid="ignore"):
        vT = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return vT


def shear_order_squared(un, vn):
    """Shear-squared under NEMO's order vs the centred-average order.

    Both are built from the SAME native C-grid velocities and, critically,
    from the SAME set of faces with the SAME denominator, so the ONLY thing
    that differs is where the square is taken:

      NEMO (zdfsh2)   square the vertical difference AT EACH FACE, then
                      average over the contributing faces
      ours            average the velocity over those faces FIRST, then
                      difference and square

    Jensen then gives ratio >= 1 pointwise, with equality where the two faces
    carry the identical vertical difference (including every one-face coastal
    column). The caller asserts that floor.

    THE FACE SET IS MASK-AWARE, AND THE FIRST VERSION OF THIS FUNCTION WAS NOT.
    It averaged the NEMO side over 2 faces unconditionally while ``u_to_T``
    divides by the number of FINITE neighbours, so a column with one dry face
    returned exactly 0.5 and tripped the assertion on the first run. A face
    counts here only when BOTH of its levels are finite, and both sides then
    divide by that same count. (NEMO's real zdfsh2 instead DOUBLES production
    next to coasts via ``2 - umask*umask``; that compensation is deliberately
    NOT reproduced, because this probe isolates the squaring order alone.)

    The common 1/e3w^2 multiplies both sides identically and is omitted, so
    the ratio needs no depth ladder and inherits no e3w approximation.

    Returns ``(sh2_nemo, sh2_ours)`` as ``(ncol, z-1)`` on the interior
    interfaces, matching the wet_pair/lat_col layout the reports use.
    """
    z, ny, nx = un.shape

    def _pair(vel, axis):
        v = np.nan_to_num(vel, nan=0.0)
        fin = np.isfinite(vel).astype(np.float64)
        face = fin[:-1] * fin[1:]                 # both levels finite
        dv = (v[:-1] - v[1:]) * face
        den = face + np.roll(face, 1, axis=axis)
        den_s = np.maximum(den, 1.0)
        # NEMO order: square at the face, then average over live faces
        sq = (dv ** 2 + np.roll(dv ** 2, 1, axis=axis)) / den_s
        # our order: average the two levels over the SAME live faces, then
        # difference and square
        up = (v[:-1] * face + np.roll(v[:-1] * face, 1, axis=axis)) / den_s
        lo = (v[1:] * face + np.roll(v[1:] * face, 1, axis=axis)) / den_s
        ct = (up - lo) ** 2
        live = den > 0
        return np.where(live, sq, 0.0), np.where(live, ct, 0.0)

    su_n, su_o = _pair(un, -1)
    sv_n, sv_o = _pair(vn, -2)
    sh2_n = su_n + sv_n
    sh2_o = su_o + sv_o

    def _cols(a):
        return np.transpose(a.reshape(z - 1, ny * nx), (1, 0))

    return _cols(sh2_n), _cols(sh2_o)


def shear_order_report(sh2_n, sh2_o, wet, lat_col, regions_def):
    """Median sh2_nemo/sh2_ours by band, plus the share of shear we discard.

    Ri = N^2 / shear^2, so a ratio of R here means the centred order inflates
    the Richardson number by R relative to NEMO's on the same velocities. That
    is the number which says whether the Prandtl ceiling we sit on is a
    discretisation artifact or a real deficit in the resolved flow.
    """
    rows = []
    for name, lo, hi in regions_def:
        band = (lat_col >= lo) & (lat_col <= hi)
        m = wet & band[:, None] & (sh2_o > 1e-20) & np.isfinite(sh2_n)
        n = int(m.sum())
        if n < 10:
            continue
        r = sh2_n[m] / sh2_o[m]
        rows.append({
            "region": name, "n": n,
            "sh2_ratio_median": float(np.median(r)),
            "sh2_ratio_p90": float(np.percentile(r, 90)),
            "sh2_ratio_mean": float(r.mean()),
            "sh2_ratio_min": float(r.min()),
            "frac_ratio_above_2": float((r > 2.0).mean()),
        })
    return rows


def run_stage_a(d, cfg):
    """Closure test: legoESM TKE K_H(state) vs NEMO avt. Returns (K_H, avt_i)
    both at the legoESM interior interfaces (nlev-1 per column)."""
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing

    T = d["T"]; S = d["S"]; e3t = d["e3t"]
    z, ny, nx = T.shape
    wet = np.isfinite(T)
    ncol = ny * nx

    def cols(a):        # (z,y,x) -> (ncol, z)
        return np.transpose(a.reshape(z, ncol), (1, 0))

    T_c = np.nan_to_num(cols(T), nan=0.0)
    S_c = np.where(np.isfinite(cols(S)), cols(S), 35.0)
    u_c = np.nan_to_num(cols(u_to_T(d["u"])), nan=0.0)
    v_c = np.nan_to_num(cols(v_to_T(d["v"])), nan=0.0)
    dz_c = np.where(np.isfinite(cols(e3t)), cols(e3t), 1.0)
    dz_half = 0.5 * (dz_c[:, :-1] + dz_c[:, 1:])            # centre-to-centre
    # In-situ density from the NEMO S-EOS at cell-centre depth (the same EOS
    # family the tripole recipe selects; kernel only needs rho for insitu N2).
    from legoesm.ocean.eos import nemo_seos_eos
    zc = np.cumsum(dz_c, axis=1) - 0.5 * dz_c                # depth >0 down [m]
    # nemo_seos_eos takes PRESSURE [Pa]; depth enters as zh = p/(rho0*g).
    p_pa = constants.rho_ocean * constants.g * zc
    rho = np.asarray(nemo_seos_eos(jnp.asarray(T_c), jnp.asarray(S_c),
                                   jnp.asarray(p_pa)))
    taum = np.nan_to_num(d["taum"].reshape(ncol), nan=0.0)
    lat_deg = d["lat"].reshape(ncol)
    # Stress components: the card's Dirichlet BC + etau need |tau| only —
    # feed taum through tau_x with tau_y=0 (|.| identical), plus the explicit
    # taum channel for exactness.
    # nn_mxl=3 needs (dz_ref, jacobian): the kernel builds e3t as their
    # rank-1 product (same approximation the production C-grid path makes).
    # Reference profile = per-level max over columns (full-cell thickness;
    # partial bottom cells see a slightly long |dl/dz| cap — bottom-cell-
    # local, negligible for the upper-ocean K comparison).
    dz_ref_1d = np.nanmax(dz_c, axis=0)
    jac1 = np.ones((ncol,), dtype=dz_c.dtype)
    # Interior interface reference heights (negative-down, surface+bottom
    # dropped) — the lc/etau surface terms read interface depths from these
    # (the k_profiles caller passes z_coord.z_half_ref[1:-1]).
    z_interface = -np.cumsum(dz_ref_1d)[:-1]
    out = tke_vertical_mixing(
        jnp.asarray(u_c), jnp.asarray(v_c), jnp.asarray(T_c),
        jnp.asarray(S_c), jnp.asarray(rho), jnp.asarray(dz_half),
        tke_old=None,
        tau_x_surface=jnp.asarray(taum), tau_y_surface=jnp.zeros_like(jnp.asarray(taum)),
        taum_surface=jnp.asarray(taum),
        dt=86400.0, cfg=cfg, rho_0=constants.rho_ocean, g=constants.g,
        n_iterations=3,
        z_interface=jnp.asarray(z_interface),
        lat_deg=jnp.asarray(lat_deg),
        ice_frac=None,
        dz_ref=jnp.asarray(dz_ref_1d), jacobian=jnp.asarray(jac1),
        # Geometric depth ladders for n2_mode="nemo_bn2". These are built
        # from NEMO's OWN e3t at this record, so they carry the live z*
        # stretch. NOTE this is a RECONSTRUCTION of gdept (cumsum(e3t)-e3t/2),
        # not NEMO's analytic gdept_1d; measured against ORCA1 L75's own
        # e3w_1d the divisor differs by 6.6e-4 median / 2.9e-3 max (job
        # 9411507). An earlier version of this comment called it "strictly
        # better" than the z_coord ladder, which was never measured -- the
        # honest statement is that both are approximations and this one's
        # error is bounded above. Without them the closure
        # raises rather than silently falling back to S-EOS, which is how
        # this gap was found (job 9411221).
        t_depth=jnp.asarray(zc),
        w_depth=jnp.asarray(np.cumsum(dz_c, axis=1)[:, :-1]),
        # e3w_int: the same centre-to-centre spacing already handed in above,
        # bounded by the 6.6e-4 divisor error noted there. Required since the
        # card selects n2_mode="nemo_bn2"; without it the closure raises.
        e3w_int=jnp.asarray(dz_half),
    )
    K_H = np.asarray(out.K_H).reshape(ncol, z - 1)
    K_M = np.asarray(out.K_M).reshape(ncol, z - 1)
    # NEMO avt lives at W-points (levels 1..z-1 are the interior interfaces
    # legoESM computes; level 0 is the surface). Slice to interior.
    avt_i = np.transpose(d["avt"].reshape(z, ncol), (1, 0))[:, 1:]
    avm_i = np.transpose(d["avm"].reshape(z, ncol), (1, 0))[:, 1:]
    wet_i = np.transpose(wet.reshape(z, ncol), (1, 0))
    wet_pair = wet_i[:, :-1] & wet_i[:, 1:]
    return K_H, avt_i, wet_pair, (z, ny, nx), K_M, avm_i


def _bathy_kwargs(T_raw, e3t_raw, cfg):
    """Per-column seafloor index and wet-interface mask, from the state itself.

    NEMO's land cells arrive as NaN, so the wet-cell count per column IS the
    bathymetry on this ladder -- no separate bathymetry file is needed and
    none can disagree with the state being stepped.

    ``w_active`` is the interior wmask: an interface is live only when the
    T-cells on both sides of it are wet, the same pairing the reports use.
    """
    import jax.numpy as jnp   # module scope has numpy only; see load_pair

    # T_raw and e3t_raw MUST still carry their NaNs. run_stage_a2_mode_a fills
    # them (nan_to_num on T, 35.0 on S, 1.0 on e3t) before stepping, and an
    # earlier version of this helper read the FILLED arrays: every column then
    # measured 75 of 75 levels wet and the mask was all ones, which is why the
    # first --use-bathy arm came back bit-identical. The print below exists to
    # catch exactly that, and did.
    wet_cell = np.isfinite(T_raw) & np.isfinite(e3t_raw) & (e3t_raw > 1e-6)
    n_wet = wet_cell.sum(axis=1)
    n_iface = wet_cell.shape[1] - 1
    w_active = (wet_cell[:, :-1] & wet_cell[:, 1:]).astype(np.float64)
    print(f"[bathy] wet interfaces {w_active.mean():.4f} of the array; "
          f"median column depth {int(np.median(n_wet))} of {wet_cell.shape[1]} "
          f"levels; shallowest wet column {int(n_wet[n_wet > 0].min())}")
    # bottom_level places NEMO's bottom TKE pin at the PER-COLUMN seafloor
    # instead of the array floor. The closure refuses it without a Dirichlet
    # VALUE, and NEMO's is MAX(0.001875 * CdU_bot * |u_bot|, rn_emin) — the
    # first term needs a bottom drag coefficient a diagnostic has no business
    # choosing, so the FLOOR alone is used. That is not an invented number:
    # cfg.tke_background is 1.0e-6 and ORCA1's rn_emin is 1.e-6
    # (namelist_ref:1228), the same value, and NEMO's expression can never
    # fall below it. So this is a LOWER BOUND on NEMO's boundary condition,
    # and the arm tests WHERE the pin lands rather than its exact magnitude.
    n_iface = wet_cell.shape[1] - 1
    bottom_level = np.clip(n_wet - 1, 0, n_iface - 1).astype(np.int32)
    return {"w_active": jnp.asarray(w_active),
            "bottom_level": jnp.asarray(bottom_level),
            "bottom_dirichlet": float(cfg.tke_background)}


def run_stage_a2_mode_a(d, rst, cfg_prog, iwm_maps=None,
                        ice_frac=None, use_bathy=False,
                        n_iterations=1, dt_s=3600.0,
                        budget=False, en_next=None):
    # The closure gates bottom_dirichlet behind TKEConfig.bottom_tke_bc and
    # raises if the value is supplied with the gate off -- a second
    # silent-no-op guard, and it fired. Enabling it is part of THIS arm's one
    # variable ("give the closure the bottom boundary"), not a separate knob.
    if use_bathy:
        cfg_prog = cfg_prog._replace(bottom_tke_bc=True)
    """EXACT Mode-A closure test: NEMO's own restart state INCLUDING the
    prognostic ``en`` -> ONE legoESM en-step (dt=3600, n_iterations=1) ->
    K_H vs NEMO's avt of the very next step (hourly record 0). Same state,
    same carry — closure implementation vs closure implementation."""
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing
    from legoesm.ocean.eos import nemo_seos_eos

    T3 = rst["tn"]; S3 = rst["sn"]
    z, ny, nx = T3.shape
    ncol = ny * nx

    def cols(a):
        return np.transpose(a.reshape(z, ncol), (1, 0))

    e3t = d["e3t"]  # rec r0 geometry (ssh drift over 1 h is negligible)
    T_raw = cols(T3)            # KEEP the NaNs: they ARE the land mask
    T_c = np.nan_to_num(T_raw, nan=0.0)
    S_c = np.where(np.isfinite(cols(S3)), cols(S3), 35.0)
    u_c = np.nan_to_num(cols(u_to_T(rst["un"])), nan=0.0)
    v_c = np.nan_to_num(cols(v_to_T(rst["vn"])), nan=0.0)
    dz_c = np.where(np.isfinite(cols(e3t)), cols(e3t), 1.0)
    dz_half = 0.5 * (dz_c[:, :-1] + dz_c[:, 1:])
    zc = np.cumsum(dz_c, axis=1) - 0.5 * dz_c
    rho = np.asarray(nemo_seos_eos(
        jnp.asarray(T_c), jnp.asarray(S_c),
        jnp.asarray(constants.rho_ocean * constants.g * zc)))
    taum = np.nan_to_num(d["taum"].reshape(ncol), nan=0.0)
    lat_deg = d["lat"].reshape(ncol)
    dz_ref_1d = np.nanmax(dz_c, axis=0)
    z_interface = -np.cumsum(dz_ref_1d)[:-1]
    # en on W-levels (0=surface); interior interfaces 1..z-1 seed the carry.
    en_i = np.nan_to_num(cols(rst["en"])[:, 1:], nan=0.0)
    out = tke_vertical_mixing(
        jnp.asarray(u_c), jnp.asarray(v_c), jnp.asarray(T_c),
        jnp.asarray(S_c), jnp.asarray(rho), jnp.asarray(dz_half),
        tke_old=jnp.asarray(en_i),
        tau_x_surface=jnp.asarray(taum),
        tau_y_surface=jnp.zeros_like(jnp.asarray(taum)),
        taum_surface=jnp.asarray(taum),
        # One 3600 s step is the closure-vs-closure test. Many steps on the
        # SAME frozen state, seeded from NEMO's own en, ask a different
        # question: does our TKE EQUATION hold the oracle's energy on the
        # oracle's state, or drain it -- the prognostic equation, not the
        # diagnostic K(en), which is all one step can see.
        dt=float(dt_s), cfg=cfg_prog, rho_0=constants.rho_ocean, g=constants.g,
        n_iterations=int(n_iterations),
        z_interface=jnp.asarray(z_interface),
        lat_deg=jnp.asarray(lat_deg),
        # NEMO's OWN ice concentration at this instant when supplied. Passing
        # None here runs the polar ocean with NO SEA ICE: the closure damps
        # its surface TKE flux by (1 - ice fraction) at tke.py:2090/2171, and
        # None turns that damping identically off, injecting full wind-driven
        # TKE into water the oracle has under ice. In January that is the
        # entire Arctic and the Antarctic coast -- the two bands where the
        # Prandtl comparison fails.
        ice_frac=(None if ice_frac is None else jnp.asarray(ice_frac)),
        # WHERE EACH COLUMN'S SEAFLOOR IS. Left None, the closure pins its
        # bottom TKE boundary condition at the LAST row of the array
        # unconditionally -- exact on a flat-bottom column and wrong on every
        # shelf, where the real seafloor is hundreds of metres above it -- and
        # without w_active there is no wet-interface mask to stop the sweeps
        # at the bottom either. ORCA1's Antarctic shelf is a few hundred
        # metres deep on a 75-level ladder reaching ~5000 m.
        **(_bathy_kwargs(T_raw, cols(e3t), cfg_prog) if use_bathy else {}),
        dz_ref=jnp.asarray(dz_ref_1d),
        jacobian=jnp.asarray(np.ones((ncol,), dtype=dz_c.dtype)),
        # Same ladders as Stage A, for the same reason -- built from NEMO's
        # own e3t so they carry the live z* stretch.
        t_depth=jnp.asarray(zc),
        w_depth=jnp.asarray(np.cumsum(dz_c, axis=1)[:, :-1]),
        # e3w_int: the same centre-to-centre spacing already handed in above,
        # bounded by the 6.6e-4 divisor error noted there. Required since the
        # card selects n2_mode="nemo_bn2"; without it the closure raises.
        e3w_int=jnp.asarray(dz_half),
        return_budget=bool(budget),
    )
    K_H = np.asarray(out.K_H).reshape(ncol, z - 1)
    # K_M too: Stage A (Mode-B, our own equilibrium TKE) measures K_M/avm 0.47
    # in the equatorial upper 100 m, while the zero-step closure HANDED NEMO's
    # own en reproduces avm_k to 0.999. Those two differ in the TKE and in the
    # state, so neither alone says whether our viscosity deficit is the closure
    # or the equilibrium. Mode-A takes NEMO's own en on NEMO's own state and is
    # the control that separates them.
    K_M = np.asarray(out.K_M).reshape(ncol, z - 1)
    # N^2 on the SAME state, from the SAME shared helper the closure calls
    # (legoesm.ocean.eos via _shared), with the SAME arguments -- not a second
    # implementation. TKEOutput does not expose N^2, and Ri = N^2/shear^2 is
    # the quantity the Prandtl ceiling is really reporting, so it has to be
    # recomputed here to be seen at all.
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2
    n2 = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T_c), jnp.asarray(S_c), jnp.asarray(zc),
        jnp.asarray(np.cumsum(dz_c, axis=1)[:, :-1]),
        g=constants.g, eos_form=getattr(cfg_prog, "n2_eos_form", "seos"),
        e3w_int=jnp.asarray(dz_half)))
    if iwm_maps is not None:
        # NEMO's zdfphy ordering: the closure runs, then zdf_iwm ADDS onto
        # avt/avs/avm. k_profiles.py:461-462 does exactly this, adding the
        # SAME K_iwm to both the tracer and the momentum total -- which is
        # why leaving it out drives our Prandtl ratio away from NEMO's.
        from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
            IWMConfig, IWMForcing, compute_iwm_diffusivity)
        H = np.sum(np.where(np.isfinite(cols(e3t)), cols(e3t), 0.0), axis=1)
        forcing = IWMForcing(
            ebot=jnp.asarray(iwm_maps["power_bot"]),
            ecri=jnp.asarray(iwm_maps["power_cri"]),
            ensq=jnp.asarray(iwm_maps["power_nsq"]),
            esho=jnp.asarray(iwm_maps["power_sho"]),
            hbot=jnp.asarray(iwm_maps["scale_bot"]),
            # zdfiwm.F90:421 stores the INVERSE of the file's scale_cri.
            hcri_inv=jnp.asarray(
                1.0 / np.maximum(iwm_maps["scale_cri"], 1e-6)))
        # ORCA1 namzdf_iwm: ln_mevar=.false., ln_tsdiff=.false. (both read
        # from namelist_cfg:459-460, and both are the IWMConfig defaults).
        K_wave = np.asarray(compute_iwm_diffusivity(
            forcing, jnp.asarray(zc), jnp.asarray(dz_half), jnp.asarray(H),
            jnp.asarray(n2),
            cfg=IWMConfig(enabled=True, mevar=False, tsdiff=False),
            rho_0=constants.rho_ocean)[0])
        print(f"[iwm] K_wave median {np.median(K_wave):.3e} "
              f"p90 {np.percentile(K_wave, 90):.3e} m2/s; added to BOTH K_H "
              "and K_M, as zdf_iwm does to avt and avm")
        K_H = K_H + K_wave
        K_M = K_M + K_wave
    # K = c_k * l * sqrt(e), and we SEED NEMO's own en but then step it, so a
    # diffusivity excess is either the LENGTH or the stepped ENERGY. Returning
    # both decomposes it instead of leaving it to argument.
    _leps = np.asarray(out.l_eps).reshape(ncol, z - 1)
    _enew = np.asarray(out.tke_new).reshape(ncol, z - 1)
    bud = None
    if budget:
        bud = _seed_budget_terms(out.budget, rst, cols, dz_c, dz_half, n2,
                                 u_c, v_c, cfg_prog, float(dt_s), ncol, z,
                                 en_next=en_next)
    return K_H, K_M, n2, dz_half, _leps, _enew, en_i, bud


def _seed_budget_terms(b, rst, cols, dz_c, dz_half, n2, u_c, v_c, cfg,
                       dt_s, ncol, z, en_next=None):
    """Per-term TKE rates [m2/s3] at NEMO's seed: OURS from the solver's own
    :class:`TKEStepBudget` (one step, divided by dt), NEMO's from the restart
    with the formulas of ``zdftke.F90`` (5.0.1) evaluated on the same en.

    Interface index j of our arrays is NEMO W-level array index a = j+1
    (``en_i = en[:, 1:]`` above), so a-1 = j is the row above (the z=0
    surface row for j=0) and a+1 = j+2 the row below.
    """
    # NEMO's rn_ediss (namelist_ref:1226) must be the coefficient our
    # dissipation used, or an eps mismatch is a constant, not an operator.
    if abs(float(cfg.c_eps) - 0.7) > 1e-12:
        raise SystemExit(f"cfg.c_eps={cfg.c_eps} != rn_ediss=0.7; the NEMO "
                         "dissipation rebuild below assumes NEMO's own value")
    r = lambda x: np.asarray(x).reshape(ncol, z - 1) / dt_s
    ours = {k: r(getattr(b, k)) for k in b._fields}
    # nn_etau is an INCREMENT per call with no dt (zdftke.F90:492-495), applied
    # once per NEMO step: present it as increment / 3600 s, not / dt_s.
    ours["etau"] = np.asarray(b.etau).reshape(ncol, z - 1) / 3600.0
    en_f = np.nan_to_num(cols(rst["en"]), nan=0.0)        # (ncol, z) W-levels
    avm_f = np.nan_to_num(cols(rst["avm_k"]), nan=0.0)
    avt_f = np.nan_to_num(cols(rst["avt_k"]), nan=0.0)
    dsl_f = np.nan_to_num(cols(rst["dissl"]), nan=0.0)
    pad0 = lambda x: np.concatenate([x, np.zeros_like(x[:, :1])], axis=1)
    e_prev, e_cur, e_next = en_f[:, :-1], en_f[:, 1:], pad0(en_f[:, 2:])
    avm_prev, avm_cur, avm_next = avm_f[:, :-1], avm_f[:, 1:], pad0(avm_f[:, 2:])
    # zdftke.F90:414 (1.5*rn_Dt*rn_ediss*dissl on the diagonal) and :419
    # (+0.5*rn_ediss*dissl*en on the rhs): net rate -rn_ediss*dissl*en at the seed.
    eps_n = -cfg.c_eps * dsl_f[:, 1:] * e_cur
    # zdftke.F90:418  - p_avt * rn2 (rn2 from the same nemo_bn2 helper the
    # closure ran on this state; the restart carries no rn2)
    b_n = -avt_f[:, 1:] * n2
    # zdftke.F90:417  + p_sh2. zdfsh2.F90:67-94 builds it from the U/V-point
    # shear weighted by avm; here avm_k times the CENTRED T-point shear our
    # closure also uses (production is ~1% of dissipation in the 2-8 m band).
    du = np.diff(u_c, axis=1) / dz_half
    dv = np.diff(v_c, axis=1) / dz_half
    p_n = avm_cur * (du * du + dv * dv)
    # zdftke.F90:404-414: zzd_up/zzd_lw = -0.5*dt*max(avm_a+avm_a+-1, 2e-5)
    # / (e3t * e3w); rate = 0.5*[lw*(e_{a-1}-e_a) + up*(e_{a+1}-e_a)] with
    # e3t[a-1] = dz_c[:, :-1], e3t[a] = dz_c[:, 1:], e3w[a] = dz_half.
    lw = np.maximum(avm_cur + avm_prev, 2.0e-5) / (dz_c[:, :-1] * dz_half)
    up = np.maximum(avm_cur + avm_next, 2.0e-5) / (dz_c[:, 1:] * dz_half)
    t_n = 0.5 * (lw * (e_prev - e_cur) + up * (e_next - e_cur))
    nemo = {"production": p_n, "buoyancy": b_n, "dissipation": eps_n,
            "transport": t_n}
    # NEMO's four reconstructed terms summed (its Langmuir and etau are NOT
    # in the restart, so this is PARTIAL and not a closure test); the +24 h
    # en change is printed only as the scale of NEMO's real tendency.
    nemo["sum4_partial"] = p_n + b_n + eps_n + t_n
    if en_next is not None:
        e_next_f = np.nan_to_num(cols(en_next), nan=0.0)
        nemo["d_en_dt_24h"] = (e_next_f[:, 1:] - e_cur) / 86400.0
    # Dissipation length at matched energy: ours from the closure, NEMO's
    # from dissl = sqrt(en)/zmxld (zdftke.F90:717) inverted on the same en.
    with np.errstate(divide="ignore", invalid="ignore"):
        l_nemo = np.where(dsl_f[:, 1:] > 0, np.sqrt(e_cur) / dsl_f[:, 1:], np.nan)
    return {"ours": ours, "nemo": nemo, "e3w": dz_half, "l_eps_nemo": l_nemo,
            "avm_sfc": avm_f[:, 0], "avm_1": avm_f[:, 1],
            "en_sfc": en_f[:, 0],
            "dtype": str(np.asarray(b.transport).dtype)}


def run_stage_b(d):
    """Operator test: BE solve with NEMO's avt/avs vs ttrd_zdf/strd_zdf."""
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        implicit_vertical_diffusion_ocean,
    )

    T = d["T"]; S = d["S"]; e3t = d["e3t"]
    z, ny, nx = T.shape
    ncol = ny * nx

    def cols(a):
        return np.transpose(a.reshape(z, ncol), (1, 0))

    T_c = np.nan_to_num(cols(T), nan=0.0)
    S_c = np.where(np.isfinite(cols(S)), cols(S), 35.0)
    dz_c = np.where(np.isfinite(cols(e3t)), cols(e3t), 1.0)
    dz_half = 0.5 * (dz_c[:, :-1] + dz_c[:, 1:])
    K_t = np.nan_to_num(cols(d["avt"]), nan=0.0)[:, 1:]      # interior interfaces
    K_s = np.nan_to_num(cols(d["avs"]), nan=0.0)[:, 1:]
    dt = 3600.0
    T_new = np.asarray(implicit_vertical_diffusion_ocean(
        jnp.asarray(T_c), jnp.asarray(K_t), jnp.asarray(dz_c),
        jnp.asarray(dz_half), dt))
    S_new = np.asarray(implicit_vertical_diffusion_ocean(
        jnp.asarray(S_c), jnp.asarray(K_s), jnp.asarray(dz_c),
        jnp.asarray(dz_half), dt))
    dT = (T_new - T_c) / dt
    dS = (S_new - S_c) / dt
    ttrd = cols(d["ttrd_zdf"])
    strd = cols(d["strd_zdf"])
    wet_c = cols(np.isfinite(T).astype(float)) > 0.5

    # --- Stage B2: solve on the reconstructed PRE-ZDF state ----------------
    # NEMO's implicit zdf acts on the RHS-UPDATED field (after the step's
    # explicit trends: advection, ldf, sbc, qsr...), not on the saved r-1
    # state -- in convecting columns the surface fluxes create the very
    # gradients zdf then removes, so Stage B on the r-1 state systematically
    # undershoots (measured f64: ours 2-10x smaller, corr ~0.1-0.6).
    # ttrd_tot/strd_tot are enabled in the XIOS defs but EMPTY in every
    # RUN_TRD file (finite count 0, measured 2026-08-11), so the budget
    # cannot be closed through the total trend.  Instead the RHS field is
    # reconstructed backwards from the saved post-step state:
    #     S_pre = S(r) - dt*strd_zdf.
    # HONEST SCOPE (codex tendency-instrument review, 2026-08-11): this is a
    # CONDITIONAL RIGHT-INVERSE test, not an independent operator
    # certification -- the input is built from the oracle target, then the
    # candidate operator must map it back toward that target.  The
    # counterfactual table shows it still discriminates: identity, explicit
    # diffusion, and one-interface K shifts all fail badly (nrmse 1..5e5)
    # while the BE operator scores 0.05-0.34.  Known non-exactness, all
    # measured small at rec 1 but not zero: (a) the *_zdf fields advertise
    # interval_operation = 7200 s vs 3600 s state fields -- the dt-scan
    # below settles the pairing empirically; (b) key_qco variable-volume
    # trends carry e3t stage weights (endpoint effect: rms 2.2e-6 relative,
    # Antarctic EVD); (c) NEMO's tra_zdf_imp matrix also contains the
    # ln_zad_Aimp implicit vertical-advection terms and the isoneutral MSC
    # akz addition, which this pure-K solve does not model.  So residual !=
    # pure discretization; it is 'operator + unmodelled matrix terms'.
    # TIME PAIRING (measured 2026-08-12, nemo_trend_closure.py pairing scan
    # on the 24-record RUN_TRD2 file): the trend written at record r matches
    # the state change r -> r+1 BETTER than r-1 -> r, for both tracers
    #   T: corr 0.952 vs 0.897, residual/state 0.305 vs 0.443
    #   S: corr 0.891 vs 0.751, residual/state 0.475 vs 0.709
    # so the post-operator state for trend r is record r+1, not record r.
    # Stage B2's right-inverse construction is insensitive to this (it builds
    # its input FROM the trend and checks the operator maps it back), which is
    # why its 0.98-0.999 correlations were not disturbed by the mispairing --
    # but the anchor is corrected here so the reconstructed state is the one
    # NEMO's operator actually produced.
    T_r1 = np.nan_to_num(cols(d["T_post"]), nan=0.0)
    S_r1 = np.where(np.isfinite(cols(d["S_post"])), cols(d["S_post"]), 35.0)
    T_pre = np.where(np.isfinite(ttrd), T_r1 - dt * np.nan_to_num(ttrd), T_c)
    S_pre = np.where(np.isfinite(strd), S_r1 - dt * np.nan_to_num(strd), S_c)

    # dt-PAIRING SCAN (codex RED 1): if the emitted trends were really
    # 7200 s two-step diagnostics, reconstructing and solving at dt=3600
    # would show a systematic ~2x mismatch.  Score the worst-region proxy at
    # both dts and print; the better one is the operative pairing.
    for dt_try in (3600.0, 7200.0):
        Sp = np.where(np.isfinite(strd), S_r1 - dt_try * np.nan_to_num(strd), S_c)
        S2t = np.asarray(implicit_vertical_diffusion_ocean(
            jnp.asarray(Sp), jnp.asarray(K_s), jnp.asarray(dz_c),
            jnp.asarray(dz_half), dt_try))
        d2t = (S2t - Sp) / dt_try
        m = wet_c & np.isfinite(strd)
        err = float(np.sqrt(np.mean((d2t[m] - strd[m]) ** 2)))
        ref = float(np.sqrt(np.mean(strd[m] ** 2)))
        print(f"[dt-scan] dt={dt_try:.0f}s: global S nrmse {err / ref:.4f}")
    T2 = np.asarray(implicit_vertical_diffusion_ocean(
        jnp.asarray(T_pre), jnp.asarray(K_t), jnp.asarray(dz_c),
        jnp.asarray(dz_half), dt))
    S2 = np.asarray(implicit_vertical_diffusion_ocean(
        jnp.asarray(S_pre), jnp.asarray(K_s), jnp.asarray(dz_c),
        jnp.asarray(dz_half), dt))
    dT2 = (T2 - T_pre) / dt
    dS2 = (S2 - S_pre) / dt
    return dT, dS, ttrd, strd, wet_c, (z, ny, nx), dT2, dS2


def d2_for_a2(d):
    """Stage A2 uses the RESTART state; geometry/stress must be the values in
    effect DURING step 8761 — d already holds rec r0=0 state-row fields when
    --rec 1, which IS the post-step-8761 snapshot; the 1-hour drift in e3t/
    taum is negligible for K (documented approximation)."""
    return d


def avt_a2(d):
    """A2 target = avt RECORD 1 (not 0): the restart stores no avt, so
    NEMO's first-step avt (rec 0) is an initialization transient — measured
    2026-07-27: calm-column rms(avt) 9-22 m2/s at rec 0 vs 0.003-0.1 at
    rec 1, same columns. Scoring one kernel en-step against rec 1 leaves a
    one-step en lag (small: surface TKE e-folds in minutes-hours)."""
    z = d["avt"].shape[0]
    ncol = d["avt"].shape[1] * d["avt"].shape[2]
    return np.transpose(d["avt"].reshape(z, ncol), (1, 0))[:, 1:]


# Lon-boxed regions (lon in NEMO's -180..180 convention on this grid).
# The latitude REGIONS above cannot see the equatorial Pacific: "tropics" is
# -15..15 over ALL longitudes, so a cold-tongue diffusivity error is averaged
# against the Indian and Atlantic basins. Nino3 is where the +3.19 C SST bias
# and the 26 m too-deep thermocline both live.
BOX_REGIONS = (
    ("nino3", -5.0, 5.0, -150.0, -90.0),
    ("nino4", -5.0, 5.0, 160.0, -150.0),      # crosses the dateline
    ("eqpac", -2.0, 2.0, -180.0, -80.0),
)


def box_report(name, ours, theirs, wet, lat_col, lon_col, evd_cols=None):
    """Same metrics as region_report but over lon-boxed equatorial regions.

    Kept as a separate function rather than folded into REGIONS because the
    dateline wrap needs an OR where the latitude bands need an AND, and
    silently special-casing that inside the existing loop is how a region ends
    up selecting the wrong cells.
    """
    rows = []
    subsets = [("", None)] if evd_cols is None else [
        ("/calm", ~evd_cols), ("/evd", evd_cols)]
    for tag0, la, lb, lo, hi in BOX_REGIONS:
        inlon = ((lon_col >= lo) & (lon_col <= hi) if lo <= hi
                 else (lon_col >= lo) | (lon_col <= hi))
        for suff, colsel in subsets:
            m = (wet & np.isfinite(theirs) & np.isfinite(ours)
                 & (lat_col[:, None] >= la) & (lat_col[:, None] <= lb)
                 & inlon[:, None])
            if colsel is not None:
                m = m & colsel[:, None]
            if not m.any():
                continue
            o, t = ours[m], theirs[m]
            ro = float(np.sqrt(np.mean(o ** 2)))
            rt = float(np.sqrt(np.mean(t ** 2)))
            rows.append({"region": tag0 + suff, "n": int(m.sum()),
                         "rms_ours": ro, "rms_nemo": rt,
                         "rms_diff": float(np.sqrt(np.mean((o - t) ** 2))),
                         "ratio": ro / rt if rt > 0 else float("nan")})
    print(f"\n=== {name} (lon-boxed) ===")
    for r in rows:
        print(f"  {r['region']:14s} n={r['n']:>7d}  ours {r['rms_ours']:.4e}  "
              f"nemo {r['rms_nemo']:.4e}  ours/nemo {r['ratio']:.3f}")
    return rows


def prandtl_split_report(K_H, K_M, avt_i, avm_i, wet, lat_col, lon_col,
                         evd_cols=None, z_iface=None, z_cuts_m=(300.0, 100.0),
                         stage="Stage A"):
    """Split the Stage-A K_H excess into a MOMENTUM part and a PRANDTL part.

    Stage A scores K_H against ``avt``, which carries BOTH the closure
    amplitude and the Prandtl reduction (nn_pdl=1).  NEMO writes ``avm`` at
    the same record, so the two factors separate with no model run and no
    inference:

        K_H/avt  =  (K_M/avm)  x  (Pr_nemo/Pr_ours),
        Pr_nemo = avm/avt,      Pr_ours = K_M/K_H.

    A K_M/avm near 1 with the whole excess in the Prandtl ratio means the
    closure amplitude and length are right and our tracer/momentum SPLIT is
    wrong -- a different defect, in a different place, from an over-energetic
    closure.  The PRINTED columns do not compose into the identity: the K
    columns are ratios of means, matching how the Stage-A rms ratios are read,
    and the Pr columns are pointwise medians.  They are printed together to be
    read against each other, not multiplied.

    The runtime control is NOT the identity -- that is a tautology for any
    four arrays and cannot fail.  It is NEMO's own Prandtl sign (avt <= avm
    off the floors, since pdlr <= 1), which a level or record misalignment
    does break; see the comment at the check.

    FLOORED INTERFACES ARE EXCLUDED, and they have to be.  NEMO applies the
    tracer floor ``avtb`` AFTER the Prandtl reduction and the momentum floor
    ``avmb`` to avm, so on a floored interface avm/avt is the ratio of two
    constants and has nothing to do with nn_pdl.  Both floors are detected as
    the field minimum over wet interfaces (the same way the zero-step closure
    detects avmb) rather than hardcoded from the namelist, so a rebuilt oracle
    with different backgrounds cannot silently poison the ratio.

    DEPTH SPLIT, and why it is not optional.  The first run of this report gave
    a Pr_ours MEDIAN of exactly 10.00 -- the clamp ceiling -- in all sixteen
    region rows at once, which is a statement about the abyss (shear ~ 0 makes
    Ri enormous and Pr saturates legitimately) and says nothing about the
    thermocline the cold-tongue bias lives in.  Rows are therefore emitted for
    the full column AND for interfaces above ``z_cut_m``, and the fraction of
    interfaces sitting ON each clamp is printed beside the median so a
    saturated statistic can never again be read as a physical Prandtl number.

    REDUCTION, matched on purpose.  ``ratio_rms`` is rms(ours)/rms(theirs), the
    SAME reduction ``box_report`` uses for the 2.311 this is meant to explain;
    ``ratio_mean`` is the ratio of means beside it.  They differ by ~4x in the
    equatorial boxes because the K distribution is heavy-tailed, and quoting
    one against the other is the confound this docstring exists to prevent.

    WHAT ``avt`` IS NOT.  ORCA1 runs ln_zdfiwm=.true., so NEMO's avt/avm carry
    an internal-wave (tidal) mixing contribution our TKE closure does not model
    at all, plus EVD on the tracer (rn_evd=100, nn_evdm=0 so momentum is
    spared).  And the hourly file writes both with ``cell_methods = time:
    mean``, i.e. an average over the hour rather than an instantaneous field --
    which is also why the detected "floor" is not the namelist background.
    None of that is fixable here; it is stated so the numbers are read as
    "our TKE K vs NEMO's TOTAL diffusivity", which is what they are.
    """
    fin = wet & np.isfinite(avt_i) & np.isfinite(avm_i) \
        & np.isfinite(K_H) & np.isfinite(K_M)
    avtb = float(np.min(avt_i[fin & (avt_i > 0)])) if (fin & (avt_i > 0)).any() else 0.0
    avmb = float(np.min(avm_i[fin & (avm_i > 0)])) if (fin & (avm_i > 0)).any() else 0.0
    free = fin & (avt_i > avtb * 1.01) & (avm_i > avmb * 1.01) & (K_H > 0)
    print(f"\n=== {stage}: momentum vs Prandtl split ===")
    print(f"  NEMO floors detected: avtb {avtb:.3e}, avmb {avmb:.3e} m2/s; "
          f"off-floor on {100.0 * free.sum() / max(fin.sum(), 1):.1f}% of wet "
          f"interfaces ({int(free.sum())} of {int(fin.sum())})")
    if free.sum() == 0:
        raise SystemExit("FATAL: every interface sits on a NEMO floor; the "
                         "Prandtl split has no domain to measure on.")
    # CONTROL, with a known answer and able to FAIL.  (An earlier revision
    # asserted the factorisation identity here; that is worthless --
    # (a/b)*((b/c)/(a/d)) == d/c for ANY four arrays, so it passes on
    # misaligned fields, on shuffled fields, on noise.  A check that cannot
    # fail is not a control.)
    #
    # What IS falsifiable is NEMO's own Prandtl sign: nn_pdl=1 sets
    # avt = max(avtb, pdlr*zav) with pdlr = 1/max(1, ...) <= 1 and
    # avm = max(avmb, zav), so OFF THE FLOORS avt <= avm always.  A level
    # slice or W-point offset between the two breaks that, and so does
    # reading the wrong record.  EVD is the one legitimate inversion (rn_evd
    # sets avt=10 while nn_evdm=0 leaves avm alone), so the abort is scoped to
    # the CALM columns and the global fraction is reported beside it.
    _inv = float(np.mean(avt_i[free] > avm_i[free]))
    if evd_cols is not None:
        _calm = free & (~evd_cols)[:, None]
        _inv_calm = float(np.mean(avt_i[_calm] > avm_i[_calm])) \
            if _calm.any() else 0.0
    else:
        _inv_calm = _inv
    print(f"  [control] NEMO avt > avm (Prandtl sign violated) on "
          f"{100.0 * _inv:.3f}% of off-floor interfaces, "
          f"{100.0 * _inv_calm:.3f}% in calm columns — expected ~0 in calm")
    if _inv_calm > 0.01:
        raise SystemExit(
            f"FATAL: NEMO's own avt exceeds its avm on {100.0 * _inv_calm:.2f}% "
            "of off-floor CALM interfaces. nn_pdl=1 makes that impossible, so "
            "the two fields are not on the same interfaces (or not the same "
            "record) and no Prandtl number below is meaningful.")
    rows = []
    subsets = [("", None)] if evd_cols is None else [
        ("/calm", ~evd_cols), ("/evd", evd_cols)]
    regions = [(n, (lat_col >= lo) & (lat_col <= hi)) for n, lo, hi in REGIONS]
    for n, la, lb, lo, hi in BOX_REGIONS:
        inlon = ((lon_col >= lo) & (lon_col <= hi) if lo <= hi
                 else (lon_col >= lo) | (lon_col <= hi))
        regions.append((n, (lat_col >= la) & (lat_col <= lb) & inlon))
    if z_iface is None:
        depths = [("", None)]
    else:
        # 300 m brackets the equatorial thermocline; 100 m is the EUC core and
        # the depth the Z20 bias is measured at. Both, because the full-column
        # rms turned out to be a DEEP signal (nino3/calm 2.311 whole column,
        # 1.054 above 300 m) and one cut cannot show that.
        depths = [("", None)] + [(f"<{c:g}m", z_iface <= c) for c in z_cuts_m]

    def _rms(x):
        return float(np.sqrt(np.mean(x ** 2)))

    def _pr_at_our_ceiling(pr_o, pr_n):
        """NEMO's Prandtl ON THE INTERFACES WHERE OURS IS CLAMPED.

        ``_ri_ratio`` below selects interfaces where BOTH Prandtl numbers are
        free, i.e. exactly the ones where we are NOT clamped -- a subset
        defined by the absence of the effect, which cannot measure it. In the
        Antarctic that subset is 8114 of 310244 interfaces and it reported
        Ri_ours/Ri_nemo = 1.085, which reads as agreement while 95% of the
        basin sits on our ceiling.

        This is the pairing that discriminates. Where OUR Pr is at the
        ceiling: NEMO's Pr also high means our Richardson number is right and
        the difference lives in the Prandtl formula; NEMO's Pr near its median
        means our Ri is too large on exactly those interfaces, and the first
        suspect is the shear -- this probe feeds the closure cell-centred
        u/v from u_to_T/v_to_T, which removes sub-cell shear and inflates Ri.
        """
        ours_ceil = pr_o > 9.99
        n = int(ours_ceil.sum())
        if n < 10:
            return {"n_ours_at_ceiling": n,
                    "Pr_nemo_median_where_ours_ceil": float("nan"),
                    "Pr_nemo_frac_ceil_where_ours_ceil": float("nan")}
        pn = pr_n[ours_ceil]
        return {"n_ours_at_ceiling": n,
                "Pr_nemo_median_where_ours_ceil": float(np.median(pn)),
                "Pr_nemo_frac_ceil_where_ours_ceil": float((pn > 9.99).mean())}

    def _ri_ratio(pr_o, pr_n):
        """Ri(ours)/Ri(NEMO) on the both-unclamped subset, or NaN if empty."""
        free_both = ((pr_o > 1.001) & (pr_o < 9.99)
                     & (pr_n > 1.001) & (pr_n < 9.99))
        n = int(free_both.sum())
        return {"n_both_unclamped": n,
                "Ri_ratio_median": (float(np.median(pr_o[free_both]
                                                    / pr_n[free_both]))
                                    if n >= 10 else float("nan"))}

    for tag0, incol in regions:
        for suff, colsel in subsets:
            for dsuff, dsel in depths:
                m = free & incol[:, None]
                if colsel is not None:
                    m = m & colsel[:, None]
                if dsel is not None:
                    m = m & dsel
                if m.sum() < 10:
                    continue
                pr_n = avm_i[m] / avt_i[m]
                pr_o = K_M[m] / K_H[m]
                rows.append({
                    "region": tag0 + suff + dsuff, "n": int(m.sum()),
                    # MATCHED to box_report's reduction; the mean sits beside it
                    # because the two disagree by ~4x on a heavy-tailed K field.
                    "K_H_over_avt_rms": _rms(K_H[m]) / _rms(avt_i[m]),
                    "K_M_over_avm_rms": _rms(K_M[m]) / _rms(avm_i[m]),
                    "K_H_over_avt_mean": float(K_H[m].mean() / avt_i[m].mean()),
                    "K_M_over_avm_mean": float(K_M[m].mean() / avm_i[m].mean()),
                    **_pr_at_our_ceiling(pr_o, pr_n),
                    "Pr_nemo_median": float(np.median(pr_n)),
                    "Pr_ours_median": float(np.median(pr_o)),
                    "Pr_ours_p10": float(np.percentile(pr_o, 10)),
                    "Pr_ours_p90": float(np.percentile(pr_o, 90)),
                    # A median that equals a clamp is not a Prandtl number.
                    "Pr_ours_frac_at_ceiling": float(np.mean(pr_o > 9.99)),
                    "Pr_ours_frac_at_floor": float(np.mean(pr_o < 1.001)),
                    "Pr_nemo_frac_at_ceiling": float(np.mean(pr_n > 9.99)),
                    "Pr_nemo_frac_at_floor": float(np.mean(pr_n < 1.001)),
                    # Both closures use Pr = clamp(4.5*Ri, 1, 10) -- the card
                    # sets prandtl_ri_coeff = 1/ri_cri = 4.5 to match nn_pdl=1
                    # -- so ON THE INTERFACES WHERE NEITHER IS CLAMPED the Pr
                    # ratio IS the Richardson-number ratio, with the shared
                    # coefficient cancelling. That subset is the only place the
                    # comparison means anything, and its size is reported so a
                    # ratio taken on a handful of points is visible as such.
                    **_ri_ratio(pr_o, pr_n),
                })
    print("  %-22s %8s %9s %9s %8s %8s %8s %8s %8s" % (
        "region", "n", "KH/avt_r", "KM/avm_r",
        "Pr_nemo", "Pr_ours", "at_ceil", "Ri_o/Ri_n", "n_free"))
    for r in rows:
        print("  %-22s %8d %9.3f %9.3f %8.2f %8.2f %8.3f %8.2f %8d" % (
            r["region"], r["n"], r["K_H_over_avt_rms"], r["K_M_over_avm_rms"],
            r["Pr_nemo_median"], r["Pr_ours_median"],
            r["Pr_ours_frac_at_ceiling"], r["Ri_ratio_median"],
            r["n_both_unclamped"]))
    print("  READ: KH/avt_r is the SAME reduction as the Stage-A box ratio, so "
          "it is the number that has to be explained; KM/avm_r is the same "
          "reduction on momentum. KM/avm_r ~ 1 with KH/avt_r >> 1 means the "
          "closure amplitude and length are right and the tracer/momentum "
          "split is wrong. at_ceil is the fraction of interfaces where OUR Pr "
          "sits on its 10.0 clamp -- where that is large the Pr median is the "
          "clamp, not a physical Prandtl number, and the row says nothing.")
    return rows


def region_report(name, ours, theirs, wet, lat_col, top_k=None,
                  evd_cols=None):
    """Per-region metrics; when ``evd_cols`` (bool, ncol) is given each
    region is split calm/evd — NEMO's EVD convection sets avt=rn_evd=10
    m2/s, which the Mode-B card does not model, so mixing the two regimes
    scores the convection SCHEME difference, not the closure."""
    rows = []
    subsets = [("", None)] if evd_cols is None else [
        ("/calm", ~evd_cols), ("/evd", evd_cols)]
    for tag0, lo, hi in REGIONS:
      for suff, colsel in subsets:
        tag = tag0 + suff
        m = (wet & np.isfinite(theirs) & np.isfinite(ours)
             & (lat_col[:, None] >= lo) & (lat_col[:, None] < hi))
        if colsel is not None:
            m = m & colsel[:, None]
        if top_k is not None:
            mm = np.zeros_like(m); mm[:, :top_k] = True; m = m & mm
        if not m.any():
            continue
        a, b = ours[m], theirs[m]
        denom = float(np.sqrt(np.mean(b ** 2)))
        rows.append({
            "region": tag, "n": int(m.sum()),
            "rms_ours": float(np.sqrt(np.mean(a ** 2))),
            "rms_nemo": denom,
            "rms_diff": float(np.sqrt(np.mean((a - b) ** 2))),
            "nrmse": float(np.sqrt(np.mean((a - b) ** 2)) / max(denom, 1e-30)),
            "corr": float(np.corrcoef(a, b)[0, 1]) if a.size > 3 else np.nan,
        })
    print(f"--- {name} ---")
    for r in rows:
        print(f"  {r['region']:16s} n={r['n']:8d} rms(le)={r['rms_ours']:.3e} "
              f"rms(nemo)={r['rms_nemo']:.3e} nrmse={r['nrmse']:.3f} "
              f"corr={r['corr']:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfile", required=True)
    ap.add_argument("--ufile", required=True)
    ap.add_argument("--vfile", required=True)
    ap.add_argument("--rec", type=int, default=1,
                    help="target record r (state r-1 vs avt/trends r); 1..11")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--skip-maps", action="store_true")
    ap.add_argument("--iwm-backgrounds", action="store_true",
                    help="Build the card with iwm_enabled=True, i.e. the "
                         "MOLECULAR backgrounds NEMO's zdfiwm_init substitutes "
                         "for &namzdf's rn_avm0/rn_avt0 when ln_zdfiwm=.true. "
                         "(avmb=1.4e-6, avtb=1e-10; zdfiwm.F90:375-379). ORCA1 "
                         "runs ln_zdfiwm=.true., so this is the MATCHED card "
                         "and the default is the mismatched one -- kept as the "
                         "default only so the standing 2.311 stays "
                         "reproducible. Note it matches the FLOORS only: the "
                         "de Lavergne wave field that supplies NEMO's actual "
                         "interior background is not modelled here at all, so "
                         "the deep rows go from 'our floor vs their wave "
                         "mixing' to 'nothing vs their wave mixing'.")
    ap.add_argument("--mxl-choice", type=int, default=None, choices=[1, 2, 3, 4],
                    help="Override tke_mxl_choice. Stage A's correlation is "
                         "0.63 and did NOT move when the amplitude convention "
                         "changed -- an amplitude error cannot change "
                         "correlation. So the SPATIAL PATTERN of our K_H is "
                         "wrong independently of its size, and the length is "
                         "the remaining term (the energy already matches NEMO "
                         "to 4 digits). If corr moves with this, the pattern "
                         "defect is in the length.")
    ap.add_argument("--prandtl-mode", default=None,
                    choices=["unit", "constant", "richardson", "nemo_ri"],
                    help="Override prandtl_mode. The other term that can "
                         "reshape K_H at fixed energy. 'nemo_ri' is NEMO's "
                         "exact nn_pdl=1 form and was missing from this list "
                         "while being implemented in tke.py:1467 -- so the "
                         "one option a stable-column comparison most needs "
                         "was unreachable from the validator.")
    ap.add_argument("--kappa-convention", default=None,
                    choices=["gaspar_sqrt2e", "veros_sqrte"],
                    help="Override the K-from-TKE amplitude convention. The "
                         "run's default is 'gaspar_sqrt2e', which our own "
                         "config docstring says DOUBLE-COUNTS the sqrt(2) on "
                         "the buoyancy-length path (tke_mxl_choice=3), giving "
                         "K_M x1.414. Stage A measured our K_H at 4.8x NEMO's "
                         "avt in calm Arctic columns; rerunning with "
                         "'veros_sqrte' tests how much of that factor the "
                         "double-count carries -- on CPU, with no model run.")
    ap.add_argument("--n2-mode", default=None,
                    choices=["insitu", "insitu_signed", "adiabatic",
                             "nemo_bn2"],
                    help="Override the stratification the closure sees. The "
                         "card now selects 'nemo_bn2' (NEMO's own eosbn2 "
                         "assembly). Passing 'insitu' reverts ONLY that, "
                         "which is the one-variable control for the signed "
                         "TEOS-10 change -- the in-situ density gradient "
                         "carries a +g^2/c^2 = 4.27e-5 s^-2 compressibility "
                         "bias that the adiabatic form does not.")
    ap.add_argument("--n2-eos-form", default=None, choices=["seos", "teos10"],
                    help="Which alpha/beta the nemo_bn2 assembly uses. Inert "
                         "under every other --n2-mode.")
    ap.add_argument("--cfg-override", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="repeatable TKEConfig override on the ORCA1 card, "
                         "for one-variable ablation of any field the named "
                         "flags do not cover (e.g. etau_frac=0.0, "
                         "tke_shear_production=nemo_face_native). Values "
                         "parse as bool/int/float/str in that order. An "
                         "unknown FIELD raises, and so does an override that "
                         "EQUALS the card value -- a no-op arm reported as "
                         "'no effect' is the worst outcome here. Ported from "
                         "nemo_zero_step_closure.py, same semantics.")
    ap.add_argument("--iwm-forcing", default=None,
                    help="Path to ORCA1's zdfiwm_forcing_TRA.nc. ORCA1 runs "
                         "ln_zdfiwm=.true., so its avt AND avm both carry a "
                         "wave-driven diffusivity; without this the Stage-A2 "
                         "Prandtl comparison is TKE-only on our side and "
                         "TKE+wave on NEMO's. Default off keeps every "
                         "existing number byte-identical.")
    ap.add_argument("--ice-restart-npz", default=None,
                    help="rebuild_nemo_restart.py output carrying 'a_i' from "
                         "NEMO's ice restart at the SAME step as the ocean "
                         "restart. Without it Stage A2 runs with no sea ice, "
                         "so the surface TKE flux is undamped in every "
                         "ice-covered column. Default off keeps existing "
                         "numbers byte-identical.")
    ap.add_argument("--variance-budget", default=None, metavar="MESH_MASK",
                    help="path to eORCA1 mesh_mask; enables the tracer-"
                         "variance comparison chi = 2*int(theta' dtheta/dt)dV "
                         "between OUR Mode-A closure and NEMO's ttrd_zdf on "
                         "NEMO's OWN state. Needs the mesh for e1t*e2t: "
                         "an unweighted integral on a tripole grid is wrong.")
    ap.add_argument("--mode-a-iterations", type=int, default=1,
                    help="Mode-A: number of backward-Euler TKE steps taken on "
                         "the frozen restart state from NEMO's own en "
                         "(default 1 = the closure-vs-closure test). Large "
                         "values integrate to the equation's own fixed point.")
    ap.add_argument("--mode-a-dt", type=float, default=3600.0,
                    help="Mode-A step length [s] (default 3600 = NEMO's).")
    ap.add_argument("--mode-a-budget", action="store_true",
                    help="Mode-A: print the per-term TKE budget of ONE step at "
                         "NEMO's seed (TKEStepBudget) beside NEMO's own terms "
                         "from the restart, 2-8 m, cold-tongue calm columns. "
                         "Requires --mode-a-iterations 1.")
    ap.add_argument("--restart-next-npz", default=None,
                    help="rebuild_nemo_restart.py output (field en) at the "
                         "restart +24 h; bounds NEMO's own closure residual "
                         "in the --mode-a-budget table.")
    ap.add_argument("--use-bathy", action="store_true",
                    help="Tell the closure where each column's seafloor is "
                         "(bottom_level + w_active, derived from the state's "
                         "own land mask). Without it the bottom TKE boundary "
                         "condition is pinned at the array floor on every "
                         "column, which is wrong by kilometres on a shelf. "
                         "Default off keeps existing numbers byte-identical.")
    ap.add_argument("--restart-npz", default=None,
                    help="rebuild_nemo_restart.py output; enables the EXACT "
                         "Mode-A closure test (Stage A2, forces --rec 1: the "
                         "restart state pairs with avt record 0)")
    args = ap.parse_args()

    from scripts.run.run_omip_core2 import orca1_zdftke_config
    # --iwm-backgrounds: ORCA1 runs ln_zdfiwm=.true., and NEMO's zdfiwm_init
    # then OVERWRITES the &namzdf backgrounds it just read --
    #     avmb(:) = rnu (1.4e-6, molecular);  avtb(:) = 1e-10
    # (zdfiwm.F90:375-379, "background avt is specified in zdf_iwm"). So the
    # rn_avm0/rn_avt0 pair the card carries by default, 1.2e-4/1.2e-5, is NOT
    # what this oracle ran. Measured on its own output: the restart's avm_k
    # floor is 1.400e-06, exactly rnu; the deep equatorial avt runs down to
    # 3.5e-07 with 89% of interfaces BELOW rn_avt0. Comparing our floored K
    # against that is a confound, not a closure test, and it is the obvious
    # candidate for the whole-column-only 2.3x.
    cfg = orca1_zdftke_config(prognostic=False,
                              iwm_enabled=args.iwm_backgrounds)
    # Mode-B quasi-steady for the
    def _apply_overrides(c, label):
        """Every CLI override, applied through ONE path.

        Stage A2 used to rebuild the card with a bare
        ``orca1_zdftke_config()``, so EVERY override -- --mxl-choice,
        --prandtl-mode, --kappa-convention and later --n2-mode -- was
        silently discarded there. Two Stage A2 arms differing only in
        --n2-mode came back bit-identical in all ten regions (job 9411271),
        which reads as "the change has no effect in Mode-A" when it actually
        means the flag never arrived. Both stages now go through here.
        """
        for spec in args.cfg_override:
            if "=" not in spec:
                raise SystemExit(f"--cfg-override {spec!r} must be FIELD=VALUE")
            k, _, raw = spec.partition("=")
            if k not in c._fields:
                raise SystemExit(
                    f"--cfg-override {k!r} is not a TKEConfig field. Silently "
                    "ignoring it would make an ablation look like a null "
                    "result.")
            if raw in ("True", "False"):
                v = (raw == "True")
            else:
                try:
                    v = int(raw)
                except ValueError:
                    try:
                        v = float(raw)
                    except ValueError:
                        v = raw
            if getattr(c, k) == v:
                raise SystemExit(
                    f"--cfg-override {k}={v!r} EQUALS the card value, so this "
                    "arm is a no-op and would be reported as 'no effect'.")
            c = c._replace(**{k: v})
            print(f"[cfg/{label}] {k} OVERRIDE -> {v!r}")
        for flag, field, cast in (
                ("mxl_choice", "tke_mxl_choice", int),
                ("prandtl_mode", "prandtl_mode", str),
                ("kappa_convention", "kappa_convention", str),
                ("n2_mode", "n2_mode", str),
                ("n2_eos_form", "n2_eos_form", str)):
            v = getattr(args, flag)
            if v is not None:
                c = c._replace(**{field: cast(v)})
                print(f"[cfg/{label}] {field} OVERRIDE -> {cast(v)}")
        print(f"[cfg/{label}] EFFECTIVE: n2_mode={c.n2_mode!r} "
              f"n2_eos_form={c.n2_eos_form!r} "
              f"mxl={c.tke_mxl_choice} kappa={c.kappa_convention!r}")
        return c

    cfg = _apply_overrides(cfg, "stage_a")
    # state-function closure test (NEMO's en is prognostic; Mode-B is its
    # documented equilibrium approximation — see module docstring).

    d = load_pair(args.tfile, args.ufile, args.vfile, args.rec)
    lat_col = d["lat"].reshape(-1)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    result = {"rec": args.rec}

    K_H, avt_i, wet_pair, shp, K_M, avm_i = run_stage_a(d, cfg)
    # EVD flag: any interface in the column at/above NEMO's rn_evd scale.
    evd_cols = np.nanmax(np.where(wet_pair, avt_i, 0.0), axis=1) > 1.0
    print(f"[evd] convecting columns: {int(evd_cols.sum())} "
          f"({100.0 * evd_cols.mean():.1f}% of columns)")
    result["stage_a_boxes"] = box_report(
        "Stage A: closure K_H vs NEMO avt, equatorial Pacific boxes",
        K_H, avt_i, wet_pair, lat_col, d["lon"].reshape(-1),
        evd_cols=evd_cols)
    result["stage_a"] = region_report(
        "Stage A: closure K_H (legoESM TKE card) vs NEMO avt [m2/s]",
        K_H, avt_i, wet_pair, lat_col, evd_cols=evd_cols)
    # Interface depths for the depth split: NEMO's own e3t at this record,
    # cumulated to the W-points, then sliced to the interior interfaces the
    # K arrays live on (drop the surface, same [:, 1:] slice avt/avm get).
    _z, _ny, _nx = shp
    _e3t_c = np.transpose(
        np.where(np.isfinite(d["e3t"]), d["e3t"], 0.0).reshape(_z, _ny * _nx),
        (1, 0))
    _z_iface = np.cumsum(_e3t_c, axis=1)[:, :-1]
    result["stage_a_prandtl"] = prandtl_split_report(
        K_H, K_M, avt_i, avm_i, wet_pair, lat_col, d["lon"].reshape(-1),
        evd_cols=evd_cols, z_iface=_z_iface, z_cuts_m=(300.0, 100.0))

    if args.restart_npz:
        if args.rec != 1:
            raise SystemExit("--restart-npz requires --rec 1 (the restart "
                             "state pairs with avt record 0)")
        rst = dict(np.load(args.restart_npz))
        # prognostic=True card, THEN the same overrides Stage A got.
        cfg_a2 = _apply_overrides(
            orca1_zdftke_config(iwm_enabled=args.iwm_backgrounds), "stage_a2")
        _iwm = load_iwm_forcing(args.iwm_forcing) if args.iwm_forcing else None
        _ice = None
        if args.ice_restart_npz:
            _ir = dict(np.load(args.ice_restart_npz))
            if "a_i" not in _ir:
                raise KeyError(
                    f"{args.ice_restart_npz} has no 'a_i'; NEMO's ice restart "
                    "stores concentration per category as a_i.")
            _ai = np.asarray(_ir["a_i"], dtype=np.float64)
            # (numcat, y, x) -> total concentration per column. ORCA1's ice
            # restart at this step carries a single category, but summing is
            # correct for any number of them.
            _ice = np.nan_to_num(_ai, nan=0.0).reshape(-1, _ai.shape[-2]
                                                       * _ai.shape[-1]).sum(0)
            _ice = np.clip(_ice, 0.0, 1.0)
            print(f"[ice] a_i: mean {_ice.mean():.4f}, "
                  f"frac>0.15 {(_ice > 0.15).mean():.4f}")
        if args.mode_a_budget:
            if args.mode_a_iterations != 1:
                raise SystemExit("--mode-a-budget reads ONE step at the seed; "
                                 "pass --mode-a-iterations 1")
            # The closure identity is checked at 1e-12 relative; the solver
            # must run in float64, not the policy default (Rule 1c).
            from legoesm.core.precision import PrecisionPolicy, set_policy
            set_policy(PrecisionPolicy.fp64())
        _en_next = (dict(np.load(args.restart_next_npz))["en"]
                    if args.restart_next_npz else None)
        (K_H2, K_M2, n2_ours, e3w_a2, leps2, enew2,
         eseed2, bud2) = run_stage_a2_mode_a(
            d2_for_a2(d), rst, cfg_a2, iwm_maps=_iwm, ice_frac=_ice,
            use_bathy=args.use_bathy,
            n_iterations=args.mode_a_iterations, dt_s=args.mode_a_dt,
            budget=args.mode_a_budget, en_next=_en_next)
        # SEED vs STEPPED turbulent energy, per interface, cold-tongue calm
        # columns. Under one step this is a near-identity; under many steps
        # it is the prognostic-equation test: a ratio near 1 through the upper
        # 30 m means our equation holds NEMO's energy on NEMO's own state.
        _lon_col = d["lon"].reshape(-1) % 360.0
        _box = (wet_pair.any(axis=1) & (np.abs(lat_col) <= 2.0)
                & (_lon_col >= 220.0) & (_lon_col <= 240.0) & ~evd_cols)
        print(f"\n[mode-a-en] {int(_box.sum())} calm columns 220-240E |lat|<=2; "
              f"{args.mode_a_iterations} step(s) x {args.mode_a_dt:g} s = "
              f"{args.mode_a_iterations * args.mode_a_dt / 3600.0:.1f} h on the "
              "frozen restart state, seeded from NEMO's own en")
        print("[mode-a-en]  depth m   en_seed(NEMO)   en_stepped(ours)   ratio")
        _zi_med = np.nanmedian(np.where(_box[:, None], _z_iface, np.nan), axis=0)
        for k in range(min(30, eseed2.shape[1])):
            _sel = _box & np.isfinite(eseed2[:, k]) & (eseed2[:, k] > 0)
            if not _sel.any():
                continue
            _s = float(np.median(eseed2[_sel, k])); _e = float(np.median(enew2[_sel, k]))
            print(f"[mode-a-en]  {_zi_med[k]:7.1f}   {_s:12.3e}   {_e:14.3e}   "
                  f"{_e / _s if _s > 0 else float('nan'):7.3f}")
        if bud2 is not None:
            # Per-term rates at the seed, medians over the calm box, plus the
            # median of per-column ratios (paired, not a ratio of medians).
            print(f"\n[mode-a-budget] one {args.mode_a_dt:g} s step at NEMO's "
                  f"seed; budget dtype {bud2['dtype']}; rates in m2/s3; "
                  "median over the box; ratio = median of per-column ours/NEMO; "
                  "nbias = median [IQR] of (ours-NEMO)/max|terms| per column")
            print("[mode-a-budget] READ WITH: ours are the step's IMPLICIT terms "
                  "(on e_new); NEMO's are explicit on the seed. They coincide "
                  "only as dt -> 0 (use --mode-a-dt 1 for seed rates). NEMO P "
                  "is avm_k*centred-shear, not p_sh2, so k=0 is not scoreable "
                  "on P. 'NEMO den/dt24h' is a 24 h evolving-forcing tendency: "
                  "a BOUND on NEMO's closure residual, not a closure check.")
            _o, _n = bud2["ours"], bud2["nemo"]
            _chk = _box[:, None] & (eseed2 > 0)
            if not _chk.any():
                raise SystemExit("[mode-a-budget] empty box selection")
            for _side in (_o, _n):
                for _f, _v in _side.items():
                    if not np.all(np.isfinite(_v[_chk])):
                        raise SystemExit(f"[mode-a-budget] non-finite {_f} inside the box")
            _res_gate = np.abs(_o["residual"]) / np.maximum(
                sum(np.abs(_o[f]) for f in ("production", "buoyancy",
                                             "dissipation", "transport")), 1e-300)
            if float(np.max(_res_gate[_chk])) > 1e-9:
                raise SystemExit("[mode-a-budget] budget does not close: max "
                                 f"residual/sum|terms| = {float(np.max(_res_gate[_chk])):.2e}")
            print(f"[mode-a-budget] surface operands, box medians: en(1) "
                  f"{float(np.median(bud2['en_sfc'][_box])):.3e}  avm_k(1) "
                  f"{float(np.median(bud2['avm_sfc'][_box])):.3e}  avm_k(2) "
                  f"{float(np.median(bud2['avm_1'][_box])):.3e}")
            _sig = np.maximum.reduce([np.abs(_n[f]) for f in
                                      ("production", "buoyancy", "dissipation",
                                       "transport")] + [np.abs(_o["external"])])
            _sig = np.maximum(_sig, 1e-300)
            _res_rel = np.abs(_o["residual"]) / np.maximum(
                np.abs(_o["dissipation"]) + np.abs(_o["transport"]), 1e-300)
            print(f"[mode-a-budget] closure residual / (|eps|+|T|), box max: "
                  f"{float(np.nanmax(np.where(_box[:, None], _res_rel, 0.0))):.2e}")
            _rows = [("production", "production"), ("buoyancy", "buoyancy"),
                     ("dissipation", "dissipation"), ("transport", "transport"),
                     ("external(LC)", "external"), ("etau", "etau"),
                     ("floor", "floor"), ("pin", "pin")]
            for k in range(min(8, eseed2.shape[1])):
                _sel = _box & np.isfinite(eseed2[:, k]) & (eseed2[:, k] > 0)
                if not _sel.any():
                    continue
                print(f"[mode-a-budget] --- interface {k} at {_zi_med[k]:.1f} m, "
                      f"en_seed median {float(np.median(eseed2[_sel, k])):.3e}, "
                      f"n={int(_sel.sum())} ---")
                print("[mode-a-budget]  term            ours          NEMO       ratio   nbias")
                for lab, f in _rows:
                    _ov = _o[f][_sel, k]
                    if f in _n:
                        _nv = _n[f][_sel, k]
                        _ok = _nv != 0.0
                        _ratio = (float(np.median(_ov[_ok] / _nv[_ok]))
                                  if _ok.any() else float("nan"))
                        _nb = (_ov - _nv) / _sig[_sel, k]
                        _q = np.percentile(_nb, [25, 50, 75])
                        _sd = float(np.mean(np.sign(_ov) != np.sign(_nv)))
                        _nae = float(np.sum(np.abs(_ov - _nv)) / max(np.sum(np.abs(_nv)), 1e-300))
                        print(f"[mode-a-budget]  {lab:13s} {float(np.median(_ov)):12.3e} "
                              f"{float(np.median(_nv)):12.3e} {_ratio:9.3f}   "
                              f"nbias {_q[1]:+7.3f} [{_q[0]:+7.3f},{_q[2]:+7.3f}]  "
                              f"signdis {_sd:.2f}  nae {_nae:.3f}")
                    else:
                        print(f"[mode-a-budget]  {lab:13s} {float(np.median(_ov)):12.3e} "
                              f"{'-':>12s} {'-':>9s}")
                _os = sum(_o[f][_sel, k] for f in
                          ("production", "external", "buoyancy", "dissipation",
                           "transport", "pin", "floor", "etau"))
                _ln = bud2["l_eps_nemo"][_sel, k]; _lo = leps2[_sel, k]
                _okl = np.isfinite(_ln) & (_ln > 0)
                print(f"[mode-a-budget]  {'l_eps [m]':13s} {float(np.median(_lo)):12.3e} "
                      f"{float(np.median(_ln[_okl])):12.3e} "
                      f"{float(np.median(_lo[_okl] / _ln[_okl])):9.3f}   (ours = closure; NEMO = sqrt(en)/dissl)")
                print(f"[mode-a-budget]  {'net (ours)':13s} {float(np.median(_os)):12.3e}")
                print(f"[mode-a-budget]  {'NEMO sum4 part':13s} "
                      f"{float(np.median(_n['sum4_partial'][_sel, k])):12.3e}")
                if "d_en_dt_24h" in _n:
                    print(f"[mode-a-budget]  {'NEMO den/dt24h':13s} "
                          f"{float(np.median(_n['d_en_dt_24h'][_sel, k])):12.3e}")
            _bk = slice(1, 4)   # interfaces 1..3 = 2.1, 3.3, 4.5 m
            _w = bud2["e3w"][:, _bk]
            _selb = _box & np.all(eseed2[:, _bk] > 0, axis=1)
            _tb_o = np.sum(_o["transport"][:, _bk] * _w, axis=1)[_selb]
            _tb_n = np.sum(_n["transport"][:, _bk] * _w, axis=1)[_selb]
            print(f"[mode-a-budget] transport INTO 2-5 m (sum_k T_k e3w_k, m3/s3): "
                  f"ours {float(np.median(_tb_o)):.3e}  NEMO {float(np.median(_tb_n)):.3e}  "
                  f"paired-median ratio {float(np.median(_tb_o / np.where(_tb_n == 0, np.nan, _tb_n))):.3f}  "
                  f"signdis {float(np.mean(np.sign(_tb_o) != np.sign(_tb_n))):.2f}  n={int(_selb.sum())}")
            result["stage_a2_mode_a_budget"] = {
                "dt_s": args.mode_a_dt, "dtype": bud2["dtype"],
                "box_n": int(_box.sum()),
                "medians": {
                    side: {f: [float(np.median(v[_box & (eseed2[:, k] > 0), k]))
                               for k in range(min(8, eseed2.shape[1]))]
                           for f, v in terms.items()}
                    for side, terms in (("ours", _o), ("nemo", _n))},
            }
        result["stage_a2_mode_a"] = region_report(
            # LABEL FIX 2026-08-13: this said "rec 0", but avt_a2() returns
            # d["avt"], and load_pair(--rec 1) puts NEMO's RECORD 1 avt there —
            # deliberately, because the restart stores no avt so rec 0 is an
            # initialisation transient (calm-column rms 9-22 m2/s at rec 0 vs
            # 0.003-0.1 at rec 1).  The label contradicted the code it names.
            "Stage A2: MODE-A closure — kernel(restart state + NEMO en, one "
            "3600s en-step) K_H vs NEMO avt REC 1 [m2/s]",
            K_H2, avt_a2(d), wet_pair, lat_col, evd_cols=evd_cols)
        # THE CONTROL THAT SEPARATES CLOSURE FROM EQUILIBRIUM.  Stage A builds
        # its own Mode-B equilibrium TKE and lands K_M/avm = 0.47 in the
        # equatorial upper 100 m; the zero-step closure, HANDED NEMO's own en,
        # reproduces avm_k to 0.999. Those two differ in the TKE *and* in the
        # state, so neither says which. Mode-A takes NEMO's own en on NEMO's own
        # state and scores the SAME split with the SAME reduction and the SAME
        # depth cuts, so the only thing left between it and Stage A is the TKE.
        #   A2 K_M/avm ~ 1 above 100 m -> the closure is fine and our
        #        EQUILIBRIUM TKE is ~4x too small at the equator (K ~ sqrt(e)).
        #   A2 K_M/avm ~ 0.47 as well  -> the TKE is not the difference and the
        #        deficit is in the closure or the length after all.
        print("\n--- Stage A2 (Mode-A, NEMO's own en) momentum/Prandtl split "
              "--- read against the Stage-A table above; the ONLY difference "
              "is where the TKE came from.")
        result["stage_a2_prandtl"] = prandtl_split_report(
            K_H2, K_M2, avt_a2(d), avm_i, wet_pair, lat_col,
            d["lon"].reshape(-1), evd_cols=evd_cols, z_iface=_z_iface,
            z_cuts_m=(300.0, 100.0), stage="Stage A2 (Mode-A, NEMO's own en)")
        # WHY OUR RICHARDSON NUMBER IS LARGER THAN NEMO'S, measured rather
        # than argued. The Prandtl split above shows our Pr pinned at its
        # ceiling of 10 across ~95% of the Antarctic while NEMO sits near 1.7
        # on those SAME interfaces, which says our Ri = N^2/shear^2 is too
        # large there. Ri has exactly two inputs, and this isolates the shear
        # one: the same native velocities squared in NEMO's order and in ours.
        # No model step, no oracle diffusivity, no depth ladder -- pure
        # discretisation arithmetic, so it cannot be confounded by the closure.
        sh2_n, sh2_o = shear_order_squared(rst["un"], rst["vn"])
        _fin = np.isfinite(sh2_n) & np.isfinite(sh2_o) & (sh2_o > 1e-20)
        _worst = float((sh2_n[_fin] / sh2_o[_fin]).min()) if _fin.any() else 1.0
        # Jensen makes ratio >= 1 pointwise; below 1 means this probe is wrong,
        # so it fails loudly instead of reporting a number nobody can trust.
        assert _worst > 1.0 - 1e-9, (
            f"shear_order_squared violated Jensen (min ratio {_worst:.6f}); "
            "the face/centre pairing or an axis is wrong")
        result["shear_order"] = shear_order_report(
            sh2_n, sh2_o, wet_pair, lat_col, REGIONS)
        print("\n--- Shear-squared: NEMO's face-native order / our centred "
              "order, SAME velocities. Ri = N^2/shear^2, so ratio R means the "
              "centred order inflates Ri by R. Jensen floor 1.0 asserted; "
              f"observed min {_worst:.4f}.")
        for _r in result["shear_order"]:
            print(f"  {_r['region']:12s} n={_r['n']:8d} "
                  f"median={_r['sh2_ratio_median']:7.3f} "
                  f"p90={_r['sh2_ratio_p90']:8.3f} "
                  f"frac>2={_r['frac_ratio_above_2']:5.3f}")

        # WHICH INPUT TO Ri IS ANOMALOUS *WHERE THE DEFECT IS*?
        #
        # The shear-order ratio was reported as a WHOLE-BAND median and came
        # out 1.026 in the Antarctic, which retired the squaring order. But
        # the defect is a coastal rim, and coastal columns are exactly where
        # the centred average loses the most: a T-column with one dry face
        # keeps a single velocity difference where the open ocean averages
        # two. The band median is dominated by open water and can hide that
        # completely -- the same p90 is 1.799 with 5.6% of interfaces above 2.
        #
        # Two reviewers split on what to do next. Codex wants NEMO's own bn2
        # (a rerun) to test N^2. GLM argued a 4.5x N^2 error from IDENTICAL
        # T and S is implausible and something must be adding to avt but not
        # avm; its candidate was double diffusion, which ORCA1's namelist
        # refutes outright (ln_zdfddm = .false., namelist_cfg:431). The
        # disagreement is still the signal, and this is the measurement that
        # settles it from data already in hand, BEFORE paying for a rerun:
        #   ratio large on the shelf  -> the shear explains Ri, no rerun
        #   ratio ~1 on the shelf     -> N^2 is the suspect, rerun justified
        _shallow = _z_iface < 100.0
        _nonevd = (~evd_cols)[:, None]
        _sh_rows = []
        for _nm, _lo, _hi in REGIONS:
            _bd = (lat_col >= _lo) & (lat_col <= _hi)
            # NEMO'S OWN TKE MUST BE PHYSICAL IN THE CELL, or the comparison
            # is against a mask value. NEMO floors wet en at rn_emin = 1e-6
            # (namelist_ref:1228); measured over the whole restart, 69.3% of
            # its finite en is <= 1e-10 and the median is exactly 0.0 — those
            # are dry rows, and 1e-10 is avtb, a DIFFUSIVITY background, not a
            # TKE floor. The antarctic<100m seed median was exactly 1.000e-10,
            # i.e. MOST of that set was dry in NEMO's accounting while passing
            # the T-file wet mask, so its avm was a background and the 20x
            # ratio was measured against it.
            _en_phys = eseed2 > 1.0e-6
            _m = (wet_pair & _bd[:, None] & _shallow & _nonevd & _en_phys
                  & (sh2_o > 1e-20) & np.isfinite(sh2_n))
            if int(_m.sum()) < 10:
                continue
            _r = sh2_n[_m] / sh2_o[_m]
            _sh_rows.append({
                "region": _nm + "/calm<100m", "n": int(_m.sum()),
                "n_dropped_en_unphysical": int(
                    (wet_pair & _bd[:, None] & _shallow & _nonevd
                     & ~_en_phys).sum()),
                "sh2_ratio_median": float(np.median(_r)),
                "sh2_ratio_p90": float(np.percentile(_r, 90)),
                "frac_above_2": float((_r > 2.0).mean()),
                "n2_ours_median": float(np.median(n2_ours[_m])),
                "shear2_ours_median": float(np.median(
                    sh2_o[_m] / np.maximum(e3w_a2, 1e-12)[_m] ** 2)),
                # The four diffusivities on the SAME cells. If our Ri is huge
                # because the shelf is quiescent, and NEMO's Prandtl is
                # nonetheless ~2, then NEMO's avt must carry something its avm
                # does not -- and these four numbers say so directly instead
                # of by inference.
                "K_H_ours_median": float(np.median(K_H2[_m])),
                "K_M_ours_median": float(np.median(K_M2[_m])),
                "avt_nemo_median": float(np.median(avt_a2(d)[_m])),
                "avm_nemo_median": float(np.median(avm_i[_m])),
                # THE DECOMPOSITION. K ~ l * sqrt(e). en_seed is NEMO's own
                # TKE as handed in; tke_new is what one 3600 s step made of
                # it. If l_eps is the outlier the length is the problem; if
                # tke_new/en_seed is, the step is amplifying energy NEMO's
                # does not have.
                "l_eps_median": float(np.median(leps2[_m])),
                "en_seed_median": float(np.median(eseed2[_m])),
                "tke_new_median": float(np.median(enew2[_m])),
                "tke_growth_median": float(np.median(
                    enew2[_m] / np.maximum(eseed2[_m], 1e-30))),
            })
        result["shear_order_shallow"] = _sh_rows
        print("\n--- Shear order RESTRICTED to the defect geometry "
              "(<100 m, nonconvecting). If the Antarctic/Arctic rows are far "
              "above the 1.026 whole-band median, the coastal shear "
              "discretisation explains the Prandtl gap and no NEMO rerun is "
              "needed.")
        for _r in _sh_rows:
            print(f"  {_r['region']:22s} n={_r['n']:8d} "
                  f"l_eps={_r['l_eps_median']:9.3f} m  "
                  f"en_seed={_r['en_seed_median']:9.3e} "
                  f"tke_new={_r['tke_new_median']:9.3e} "
                  f"growth={_r['tke_growth_median']:8.3f} "
                  f"K_M/avm={_r['K_M_ours_median']/_r['avm_nemo_median']:8.2f}")

        # THE OTHER INPUT TO Ri. With the shear order worth only ~2-7% and the
        # Mode-A state being NEMO's OWN velocities, the remaining way for our
        # Ri to exceed NEMO's is N^2. NEMO's rn2 is not in the files this
        # comparator reads, but it does not need to be: where NEMO's Prandtl is
        # strictly inside its bounds the pdl transform is INVERTIBLE, so
        #     Ri_nemo = Pr_nemo / prandtl_ri_coeff,  Pr_nemo = avm/avt
        # and NEMO's own N^2 follows as Ri_nemo * shear^2_nemo.
        # The unclamped restriction here is forced by invertibility -- outside
        # it Pr carries no Ri information at all -- unlike the _ri_ratio subset,
        # which was merely convenient and was therefore vacuous.
        _coeff = float(getattr(cfg_a2, "prandtl_ri_coeff", 4.5))
        _e3w2 = np.maximum(e3w_a2, 1e-12) ** 2
        _sh2_n_real = sh2_n / _e3w2          # NEMO's order, real units s^-2
        _sh2_o_real = sh2_o / _e3w2          # centred order, what we feed
        _avt2, _avm2 = avt_a2(d), avm_i
        with np.errstate(divide="ignore", invalid="ignore"):
            _pr_n = _avm2 / _avt2
            _ri_n = _pr_n / _coeff
            _n2_nemo = _ri_n * _sh2_n_real
            _ri_o = n2_ours / np.maximum(_sh2_o_real, 1e-20)
        # END-TO-END CONTROL: the Pr this predicts from our own N^2 and shear
        # must reproduce the Pr the closure actually produced (K_M/K_H). If it
        # does not, the inference chain below is broken and its numbers mean
        # nothing, so it is reported next to them rather than assumed.
        _pr_pred = np.clip(_coeff * _ri_o, 1.0, 10.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            _pr_act = K_M2 / K_H2
        _rows = []
        for _nm, _lo, _hi in REGIONS:
            _band = (lat_col >= _lo) & (lat_col <= _hi)
            _inv = (_pr_n > 1.001) & (_pr_n < 9.99)     # NEMO invertible
            _m = (wet_pair & _band[:, None] & _inv
                  & np.isfinite(_n2_nemo) & np.isfinite(n2_ours)
                  & (np.abs(_n2_nemo) > 1e-12))
            if int(_m.sum()) < 10:
                continue
            _ctl = np.isfinite(_pr_pred) & np.isfinite(_pr_act) & _m
            _rows.append({
                "region": _nm, "n": int(_m.sum()),
                "n2_ours_over_nemo_median": float(np.median(
                    n2_ours[_m] / _n2_nemo[_m])),
                "n2_ours_median": float(np.median(n2_ours[_m])),
                "n2_nemo_implied_median": float(np.median(_n2_nemo[_m])),
                "ri_ours_median": float(np.median(_ri_o[_m])),
                "ri_nemo_median": float(np.median(_ri_n[_m])),
                "control_pr_pred_minus_actual_median": float(np.median(
                    _pr_pred[_ctl] - _pr_act[_ctl])) if _ctl.any() else float("nan"),
            })
        result["n2_split"] = _rows
        print("\n--- Ri = N^2/shear^2, the N^2 half. NEMO's N^2 is INFERRED by "
              f"inverting its own pdl (Pr=avm/avt, coeff {_coeff}) on the "
              "interfaces where that inverse exists.  CONTROL: pr_pred-pr_act "
              "must be ~0 or the inference is broken.")
        for _r in _rows:
            print(f"  {_r['region']:12s} n={_r['n']:8d} "
                  f"N2_ours/N2_nemo={_r['n2_ours_over_nemo_median']:8.3f} "
                  f"Ri_ours={_r['ri_ours_median']:8.4f} "
                  f"Ri_nemo={_r['ri_nemo_median']:7.4f} "
                  f"CTL={_r['control_pr_pred_minus_actual_median']:+8.4f}")

    dT, dS, ttrd, strd, wet_c, _, dT2, dS2 = run_stage_b(d)
    result["stage_b_T"] = region_report(
        "Stage B: operator dT_zdf (BE solve w/ NEMO avt) vs ttrd_zdf [K/s]",
        dT, ttrd, wet_c & np.isfinite(ttrd), lat_col, evd_cols=evd_cols)
    result["stage_b_S"] = region_report(
        "Stage B: operator dS_zdf vs strd_zdf [PSU/s]",
        dS, strd, wet_c & np.isfinite(strd), lat_col, evd_cols=evd_cols)
    result["stage_b2_T"] = region_report(
        "Stage B2: operator on PRE-ZDF state (tot-closure) dT vs ttrd_zdf [K/s]",
        dT2, ttrd, wet_c & np.isfinite(ttrd), lat_col, evd_cols=evd_cols)
    result["stage_b2_S"] = region_report(
        "Stage B2: operator on PRE-ZDF state dS vs strd_zdf [PSU/s]",
        dS2, strd, wet_c & np.isfinite(strd), lat_col, evd_cols=evd_cols)

    # --- Variance budget: OUR closure vs NEMO's, on NEMO's OWN state ------
    # The operator-resolved budget on NEMO's archived trends showed vertical
    # mixing is its dominant tracer-variance destroyer. The question that
    # matters for us is whether OUR closure destroys variance at the same
    # rate. Because Mode-A already evaluated our K on NEMO's state, the state
    # is shared by construction and there is NO state-difference confound --
    # the single cleanest comparison available.
    #
    # chi = 2 * integral(theta' * dtheta/dt) dV, theta' about the
    # VOLUME-WEIGHTED mean. Ratio near 1 means our closure removes tracer
    # variance at NEMO's rate; below 1 means we under-mix, above 1 over-mix.
    if args.variance_budget:
        import jax.numpy as jnp
        from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
            implicit_vertical_diffusion_ocean,
        )
        from global_tracer_content import load_mesh_metrics
        e1t, e2t, _e3ref, _tm = load_mesh_metrics(args.variance_budget)
        zc, nyc, nxc = d["T"].shape
        ncolc = nyc * nxc

        def _cols(a):
            return np.transpose(np.asarray(a).reshape(zc, ncolc), (1, 0))

        dz_c = np.where(np.isfinite(_cols(d["e3t"])), _cols(d["e3t"]), 1.0)
        dz_h = 0.5 * (dz_c[:, :-1] + dz_c[:, 1:])
        T_c = np.nan_to_num(_cols(d["T"]), nan=0.0)
        area = (e1t * e2t).reshape(ncolc)[:, None]
        dVc = area * dz_c
        wetv = (_cols(np.isfinite(d["T"]).astype(float)) > 0.5) & np.isfinite(
            _cols(d["ttrd_zdf"]))

        # DRY CELLS MUST NOT DIFFUSE. `T_c` carries 0.0 below the sea floor
        # (nan_to_num), so an unmasked solve sees a ~20 K jump at the last wet
        # interface and manufactures an enormous tendency in the deepest wet
        # cell. The first run of this budget returned a ratio of 414, which no
        # closure whose diffusivity is 0.89-1.49x NEMO's could produce -- an
        # instrument defect, not a result. K is therefore zeroed at every
        # interface whose two adjacent cells are not both wet.
        wet_cell = _cols(np.isfinite(d["T"]).astype(float)) > 0.5
        iface_wet = wet_cell[:, :-1] & wet_cell[:, 1:]
        K_use = np.where(iface_wet, np.nan_to_num(K_H2, nan=0.0), 0.0)
        print(f"[variance] K interfaces kept {int(iface_wet.sum())} of "
              f"{iface_wet.size}; K_use max {K_use.max():.4e} m2/s")

        T_ours = np.asarray(implicit_vertical_diffusion_ocean(
            jnp.asarray(T_c), jnp.asarray(K_use), jnp.asarray(dz_c),
            jnp.asarray(dz_h), 3600.0))
        dT_ours = (T_ours - T_c) / 3600.0
        _nm = _cols(d["ttrd_zdf"])
        print(f"[variance] |dT| ours max {np.abs(np.where(wetv, dT_ours, 0)).max():.4e}"
              f"  NEMO max {np.abs(np.where(wetv, np.nan_to_num(_nm), 0)).max():.4e} K/s")

        def _chi(tend):
            w = np.where(wetv, dVc, 0.0)
            vol = w.sum()
            mu = float((np.where(wetv, T_c, 0.0) * w).sum() / vol)
            return float(2.0 * (np.where(wetv, T_c - mu, 0.0)
                                * np.where(wetv, tend, 0.0) * w).sum())

        chi_ours = _chi(dT_ours)
        chi_nemo = _chi(_cols(d["ttrd_zdf"]))
        ratio = chi_ours / chi_nemo if chi_nemo else float("nan")
        print(f"\n[variance] chi_zdf(theta) ours {chi_ours:+.6e}  "
              f"NEMO {chi_nemo:+.6e}  ratio {ratio:.4f}")
        print("[variance] ratio < 1 = we under-mix, > 1 = we over-mix; "
              "state is NEMO's own, so this is not confounded by the state.")
        result["variance_budget_zdf_theta"] = {
            "chi_ours": chi_ours, "chi_nemo": chi_nemo, "ratio": ratio,
            "n_cells": int(wetv.sum()),
            "note": "our K from the Mode-A closure, NEMO's from ttrd_zdf, "
                    "same state, same volumes, same BE solver",
        }

    with open(out_dir / f"tendency_match_rec{args.rec}.json", "w") as f:
        json.dump(result, f, indent=1)

    if not args.skip_maps:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z, ny, nx = shp
        k10 = 3   # ~10-30 m interface on the 75-level grid
        fig, axes = plt.subplots(3, 2, figsize=(13, 12))
        panels = [
            ("legoESM K_H (iface k=3)", K_H[:, k10].reshape(ny, nx), 0, 1e-2),
            ("NEMO avt (iface k=3)", avt_i[:, k10].reshape(ny, nx), 0, 1e-2),
            ("dT_zdf legoESM (k=0)", dT[:, 0].reshape(ny, nx), -2e-5, 2e-5),
            ("ttrd_zdf NEMO (k=0)", ttrd[:, 0].reshape(ny, nx), -2e-5, 2e-5),
            ("Stage B T residual (k=0)", (dT - ttrd)[:, 0].reshape(ny, nx),
             -2e-6, 2e-6),
            ("Stage A K residual (iface k=3)",
             (K_H - avt_i)[:, k10].reshape(ny, nx), -5e-3, 5e-3),
        ]
        for ax, (title, fld, vmin, vmax) in zip(axes.flat, panels):
            wm = np.isfinite(d["T"][0])
            fld = np.where(wm, fld, np.nan)
            im = ax.pcolormesh(fld, vmin=vmin, vmax=vmax,
                               cmap="RdBu_r" if vmin < 0 else "viridis")
            ax.set_title(title, fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"NEMO tendency match — rec {args.rec} "
                     f"(state r-1 vs avt/trends r)", fontsize=11)
        fig.tight_layout()
        png = out_dir / f"tendency_match_rec{args.rec}.png"
        fig.savefig(png, dpi=110)
        print(f"[maps] {png}")

        if args.restart_npz:
            # WHERE THE PRANDTL DEFECT LIVES. The band table says our effective
            # tracer-to-momentum ratio matches NEMO within 1-9% in ten bands
            # and is 4.5-5.7x too large in the polar upper 100 m. A band mean
            # cannot say whether that is the whole polar cap or a rim -- e.g.
            # the marginal ice zone, or shelves -- and those imply different
            # causes, so it is plotted.
            # MASK THE MAP THE WAY THE TABLE IS MASKED. The band rows quoted
            # as "calm" EXCLUDE columns where NEMO's EVD fired; an unmasked
            # map beside a masked table compares two different reductions and
            # invites exactly the misreading this session has already made
            # twice. January is northern winter, so the northern basins are
            # full of convecting columns whose avt carries an EVD our probe
            # has no counterpart for -- unmasked, they dominate the picture
            # and look like a defect the table explicitly set aside.
            # SAME GATE AS THE TABLE: NEMO's own en must exceed its rn_emin
            # floor, or the cell is one NEMO treats as dry and its avm/avt are
            # background fills. Without this the figure showed a bright
            # Antarctic shelf band that was entirely our real closure divided
            # by NEMO's mask values (job 9877224: the Antarctic momentum ratio
            # goes from 20.16 to 0.92 once the gate is applied).
            _en_ok = (eseed2 > 1.0e-6)
            wm2 = np.isfinite(d["T"][0]) & (~evd_cols).reshape(ny, nx)
            with np.errstate(divide="ignore", invalid="ignore"):
                _pro = K_M2 / np.maximum(K_H2, 1e-30)
                _prn = avm_i / np.maximum(avt_a2(d), 1e-30)
                _rat = np.log10(np.maximum(_pro, 1e-6)
                                / np.maximum(_prn, 1e-6))
            # ABSOLUTE ratios, not just the Prandtl one. Reading the ratio
            # alone inverted the physics once already: the Prandtl number is
            # off because K_M is excessive by MORE than K_H is, not because
            # tracer mixing is weak. Both diffusivities are plotted against
            # the oracle's own so the sign cannot be misread again.
            with np.errstate(divide="ignore", invalid="ignore"):
                _rkm = np.log10(np.maximum(K_M2, 1e-30)
                                / np.maximum(avm_i, 1e-30))
                _rkh = np.log10(np.maximum(K_H2, 1e-30)
                                / np.maximum(avt_a2(d), 1e-30))
            fig2, ax2 = plt.subplots(2, 2, figsize=(13, 8.4))
            for _ax, _fld, _kk, _ttl in (
                    (ax2[0][0], _rkm, 3, "log10(K_M ours / avm NEMO)  ~10-30 m"),
                    (ax2[0][1], _rkm, 8, "log10(K_M ours / avm NEMO)  ~100 m"),
                    (ax2[1][0], _rkh, 3, "log10(K_H ours / avt NEMO)  ~10-30 m"),
                    (ax2[1][1], _rkh, 8, "log10(K_H ours / avt NEMO)  ~100 m")):
                _f = np.where(wm2 & _en_ok[:, _kk].reshape(ny, nx),
                              _fld[:, _kk].reshape(ny, nx), np.nan)
                _im = _ax.pcolormesh(_f, vmin=-2, vmax=2, cmap="RdBu_r")
                _ax.set_title(_ttl, fontsize=9)
                plt.colorbar(_im, ax=_ax, shrink=0.85)
            fig2.suptitle(
                "Vertical diffusivity vs NEMO, Mode-A (NEMO's own state, ice, "
                "wave mixing and backgrounds; convecting columns masked). "
                "RED = WE MIX MORE THAN NEMO, 0 = match, +2 is a hundredfold "
                "excess. Top momentum, bottom tracer. White = no data or NEMO "
                "inactive there.", fontsize=10)
            fig2.tight_layout()
            png2 = out_dir / f"prandtl_ratio_rec{args.rec}.png"
            fig2.savefig(png2, dpi=110)
            print(f"[maps] {png2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
