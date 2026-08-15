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


def run_stage_a2_mode_a(d, rst, cfg_prog):
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
    T_c = np.nan_to_num(cols(T3), nan=0.0)
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
        dt=3600.0, cfg=cfg_prog, rho_0=constants.rho_ocean, g=constants.g,
        n_iterations=1,
        z_interface=jnp.asarray(z_interface),
        lat_deg=jnp.asarray(lat_deg),
        ice_frac=None,
        dz_ref=jnp.asarray(dz_ref_1d),
        jacobian=jnp.asarray(np.ones((ncol,), dtype=dz_c.dtype)),
        # Same ladders as Stage A, for the same reason -- built from NEMO's
        # own e3t so they carry the live z* stretch.
        t_depth=jnp.asarray(zc),
        w_depth=jnp.asarray(np.cumsum(dz_c, axis=1)[:, :-1]),
    )
    K_H = np.asarray(out.K_H).reshape(ncol, z - 1)
    # K_M too: Stage A (Mode-B, our own equilibrium TKE) measures K_M/avm 0.47
    # in the equatorial upper 100 m, while the zero-step closure HANDED NEMO's
    # own en reproduces avm_k to 0.999. Those two differ in the TKE and in the
    # state, so neither alone says whether our viscosity deficit is the closure
    # or the equilibrium. Mode-A takes NEMO's own en on NEMO's own state and is
    # the control that separates them.
    K_M = np.asarray(out.K_M).reshape(ncol, z - 1)
    return K_H, K_M


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
                         evd_cols=None, z_iface=None, z_cuts_m=(300.0, 100.0)):
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
    print(f"\n=== Stage A: momentum vs Prandtl split ===")
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
        K_H2, K_M2 = run_stage_a2_mode_a(d2_for_a2(d), rst, cfg_a2)
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
            z_cuts_m=(300.0, 100.0))

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
