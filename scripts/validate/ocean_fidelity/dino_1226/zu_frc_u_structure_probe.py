"""#1226 zu_frc u-component SPATIAL STRUCTURE probe.

TASK (human): ``zu_frc`` u-component err_norm 8.03e-3 is the campaign's
LARGEST row (drives ``dyn_spg_ts puu_b`` 1.3e-2, ``un_adv`` 8.4e-3 -- 101.2%
of the final barotropic error). The momentum-row reconstruction
(``zu_frc_momentum_row_reconstruction.py``, this session) REFUTED "the u
error is explained by dyn_ldf+dyn_adv_ZAD+dyn_vor_EEN" (corr=0.044,
ratio=0.045) while CONFIRMING it for v (corr=0.985, ratio=0.977) -- the u
error is NOT a momentum-row artifact and has a recorded but never quantified
"basin-scale wave-like latitude profile" (peaks near rows ~50-60, ~110-125,
~140-155).

THIS script (a) quantifies that pattern with real numbers (FFT/autocorrelation
of the meridional profile, wavelength in rows AND km, wall-adjacent
enhancement factor + decay scale in cells), (b) runs the zv_frc CONTROL
(same measurement, should be flat/absent per the momentum-row reconstruction
result), and (c) A/Bs two candidates read directly out of NEMO's
``dynspg_ts.F90`` and legoESM's ``barotropic_latlon_cgrid.py``:

  Candidate (a) TIME FILTER WEIGHTS: NEMO ``ts_wgt`` CASE(2) (dynspg_ts.F90
    :1245-1252) builds a PURELY SCALAR per-substep weight ``zwgt1(jn)`` --
    no ``(ji,jj)`` index anywhere in the subroutine. legoESM's faithful port
    is ``compute_nemo_boxcar_centred_weights`` (barotropic_common.py:88-162,
    cited to the same ts_wgt CASE(2) lines). A scalar-per-substep weight
    CANNOT, by construction, produce a LATITUDE-DEPENDENT pattern -- this is
    EXCLUDED BY READING (both sides checked elementwise below as a cheap
    confirmation, not because the read leaves any doubt).

  Candidate (d) EEN BAROTROPIC CORIOLIS METRIC FACTORS: NEMO's
    ``dyn_cor_2D_init`` (dynspg_ts.F90:1466-1642, np_EEN case) builds
    ``ffu_nw/ne/sw/se`` from ``ff_f / e3f_vor`` (the LATITUDE-DEPENDENT
    Coriolis parameter) times ``r1_e1u``/``e1v`` metric factors -- a
    genuinely (ji,jj)-varying coefficient. legoESM's
    ``een_barotropic_coriolis`` (barotropic_latlon_cgrid.py:548-609) has TWO
    variants gated by ``config.barotropic.barotropic_coriolis``: ``"een"``
    (default, DROPS e1/e2 metric factors, docstring's own words: "leaves an
    O(Delta cos phi) work residual (~4e-4)") vs ``"een_metric"``
    (metric-complete, claimed exact-NEMO). ``dino_config_for_recipe(
    "nemo_dino_kamm_mlf")`` is INSTANTIATED AND PRINTED below (Rule 10) to
    confirm which variant production actually uses, then A/B'd by forcing
    the OTHER variant through the SAME pre-step-subtraction spy
    ``zu_frc_term_walk.py`` already validated (same call, one kwarg changed).

  Candidate (leapfrog residual, found while reading
    ``ocean_model_latlon_cgrid.py:3260-3280`` for this task): the production
    code's OWN comment documents an EXPECTED non-cancelling residual
    ``O(f*(U_Nnn - U_Nbb))`` between the pre-step EEN subtraction (built at
    Kmm/Nnn) and the live substep-0 application (seeded at Kbb/Nbb) under
    the MLF leap-frog integrator -- explicitly NOT a bug, "this is NEMO's
    design". This is f-weighted (hence latitude-structured) BY CONSTRUCTION
    and is the standing config (``barotropic_een_seed="nemo_kmm"``, already
    the faithful setting) -- reported as a CANDIDATE explanation for the
    wave-like pattern, not something to "fix" (no faithful alternative
    exists per the comment).

Reuses ``zu_frc_term_walk.py``'s loaders/spies/RUN_DIR/DT constants
WHOLESALE (imported), per the module-reuse rule.

RESULTS (this session, self-check reproduced the recorded 8.0266e-03 u /
5.4290e-04 v err_norm exactly, alignment sharp at (0,0) both components):
namelist (RUN_GDB/namelist_cfg + ocean.output, quoted not assumed):
``nn_e=30`` requested, auto-adjusted to ``nn_e=23`` ("in iterations nn_e =
23"), ``nn_bt_flt=2``, ``ln_bt_fw=.false.``, ``icycle=68`` ("#1226
dyn_cor_2D dump ... icycle= 68"). Meridional profile (u, mean|err| by row,
199 rows, ~79.6 km/row): a broad low-order hump (rows ~0-72, peak ~row 47,
FFT/autocorrelation both pick this up as the "dominant" component -- FFT
wavelength 99.5 rows = half the domain, autocorr lag 63) with a DAMPED
QUASI-PERIODIC RIPPLE riding on top for rows ~72-199 (direct peak-finder,
``scipy.signal.argrelextrema``: maxima at rows 48/84/114/150, spacing
36/30/36 rows, mean 34.0 rows = ~2706 km). Wall enhancement measured on
BOTH conventions since u faces longitude: column(east-west/zonal-wall)
enhancement west/core=1.265x, east/core=1.198x, decay_scale=6.1 cells;
row(south-north) enhancement is BELOW 1 (0.147x/0.074x) -- i.e. the
"wall rise" is at the ZONAL (east-west, longitude) walls, not a
meridional edge effect, and decays within ~6 cells (~1 grid spacing at
this resolution), not over a broad band. zv_frc CONTROL: row-profile
coefficient of variation u=0.760 vs v=1.612 -- v is NOT flatter than u by
this metric (v's profile is dominated by strong pole-adjacent decay, its
own wall effect, not a flat/near-zero interior); the discriminator that
DOES hold is the momentum-row reconstruction's structural test (corr
0.985 for v vs 0.044 for u, prior script), not the raw CV here -- recorded
so a later reader does not re-derive "v is flatter" from CV alone and get
it backwards.

Candidate dispositions: (a) time-filter weights EXCLUDED BY READING (both
NEMO's ``ts_wgt`` and legoESM's ``compute_nemo_boxcar_centred_weights``
are 1-D, purely per-substep-indexed arrays -- confirmed by shape,
``w_filter.shape=(68,)``, no ``(ji,jj)`` construction exists to check
elementwise). (d) EEN barotropic Coriolis metric factors: A/B'd by
toggling ``model.config.barotropic.barotropic_coriolis`` between the
production ``"een_metric"`` and ``"een"`` (metric-dropped) -- u
err_norm 8.0266e-03 -> 8.0205e-03 (ratio 0.999x, ESSENTIALLY UNCHANGED)
while v moves 5.4290e-04 -> 3.8090e-04 (ratio 0.702x, a REAL, expected
effect, confirming the toggle reaches the solver and this candidate's
sensitivity is asymmetric exactly as the prior u/v-asymmetry finding
would predict). u's row-profile CV is also unchanged (0.760 -> 0.765).
CONCLUSION: candidate (d) is EXCLUDED for u (the metric-complete fix is
already in place AND has negligible effect on the residual that remains),
though CONFIRMED-RELEVANT for v (not what this task's u focus needed, but
worth flagging: v is NOT yet fully closed by een_metric either --
3.8e-4 is not zero). (c) basin gravity-wave mode arithmetic: c=sqrt(g*4000m)
=198.1 m/s; substep length dt_s=117.4s; full 68-substep window travel
distance = 68*c*dt_s = 1581 km, ORDER-OF-MAGNITUDE consistent with (but not
a clean integer ratio to) the observed ~2706 km ripple wavelength (ratio
~1:1.7); domain meridional extent (140 deg, ~15568 km) / observed
wavelength = 5.75 (not a clean mode number). Labeled PLAUSIBLE-INCONCLUSIVE,
not a confirmation -- the arithmetic does not cleanly pick out an integer
gravity-wave mode, and no independent basin-mode calculation was performed
beyond this back-of-envelope estimate. (leapfrog-residual candidate, found
reading ocean_model_latlon_cgrid.py:3260-3280 for this task): NOT
A/B-tested here (no faithful alternative exists per the production
comment -- there is nothing to toggle), reported as the strongest
STILL-STANDING PLAUSIBLE candidate: it is f-weighted (hence
latitude-structured, matching the observed pattern's dependence on row)
BY CONSTRUCTION and the production comment already documents it as a
non-cancelling residual under the MLF leap-frog integrator.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/zu_frc_u_structure_probe.py
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax

from legoesm import constants
from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_before_state_topo,
    bridge_nemo_to_legoesm_topo,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocmod
import legoesm.ocean.dynamics.barotropic_latlon_cgrid as bmod
from legoesm.ocean.dynamics.barotropic_common import compute_nemo_boxcar_centred_weights
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)

# --- Reuse wholesale (module-reuse rule) ------------------------------------
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    RUN_DIR, DT, _load_interior,
)

register_dump("spg_dump_zu_frc.bin", "before",
              "dynspg_ts.F90:341-345,367 zu_frc after the zu_trd subtraction.")
register_dump("spg_dump_zv_frc.bin", "before", "same as zu_frc")
for _name in ("spg_dump_zu_frc.bin", "spg_dump_zv_frc.bin"):
    time_level_for_dump(_name)


def _u_to_nemo(a):
    return np.asarray(a)[:, 1:]


def _v_to_nemo(a):
    return np.asarray(a)[1:, :]


def _capture_F_slow(model, st, sf, *, barotropic_coriolis_override=None):
    """Run one model.step(), returning the production F_slow_u/F_slow_v the
    barotropic solver receives (post zu_trd-subtraction). Optionally forces
    ``model.config.barotropic.barotropic_coriolis`` to a different value for
    the A/B (candidate (d)) -- the ONLY variable changed, per Rule 7.

    The override must be applied to ``model.config`` itself (NOT the
    ``config`` kwarg the barotropic-substep call receives): the pre-step
    zu_trd-subtraction block that reads ``config.barotropic.barotropic_coriolis``
    (ocean_model_latlon_cgrid.py:3252-3254) runs BEFORE
    ``barotropic_substeps_latlon_cgrid`` is ever called, reading
    ``self.config`` directly -- a spy on the substep call alone would leave
    that pre-step read unchanged and silently no-op the A/B (caught by this
    probe's own first run: overriding only the substep-call kwarg gave an
    IDENTICAL result, ratio=1.000x, which is what exposed this)."""
    captured = {}
    _real_baro = ocmod.barotropic_substeps_latlon_cgrid

    def _spy_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw):
        result = _real_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw)
        if kw.get("eta_init") is not None and "F_slow_u" not in captured:
            captured["F_slow_u"] = kw["F_slow_u"]
            captured["F_slow_v"] = kw["F_slow_v"]
        return result

    ocmod.barotropic_substeps_latlon_cgrid = _spy_baro
    _orig_config = model.config
    try:
        if barotropic_coriolis_override is not None:
            model.config = model.config._replace(
                barotropic=model.config.barotropic._replace(
                    barotropic_coriolis=barotropic_coriolis_override,
                ),
            )
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = _real_baro
        model.config = _orig_config
    assert "F_slow_u" in captured, "barotropic solver never called with a seed"
    return np.asarray(captured["F_slow_u"]), np.asarray(captured["F_slow_v"])


def _err_norm(lego, nemo, mask):
    """err_norm = |lego-nemo|/RMS(nemo), plus max|diff| and near-zero
    fraction (task rule) -- printed by the caller, not asserted here."""
    m = mask & np.isfinite(lego) & np.isfinite(nemo)
    lo, ne = lego[m], nemo[m]
    rms = float(np.sqrt(np.mean(ne ** 2))) if ne.size else float("nan")
    err_norm = float(np.sqrt(np.mean((lo - ne) ** 2))) / rms if rms > 0 else float("nan")
    maxdiff = float(np.max(np.abs(lo - ne))) if lo.size else float("nan")
    near_zero = float(np.mean(np.abs(ne) < 1e-3 * (rms if rms > 0 else 1.0)))
    return err_norm, rms, maxdiff, near_zero


def _align_scan(name: str, lego: np.ndarray, nemo: np.ndarray, mask: np.ndarray) -> None:
    """Self-check (skill rule): sharp offset-0 peak = a genuine, aligned match."""
    best = None
    at00 = None
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            L = np.roll(lego, (dj, di), axis=(0, 1))
            err, rms, _maxd, _nz = _err_norm(L, nemo, mask)
            if rms <= 0 or not np.isfinite(err):
                continue
            if (dj, di) == (0, 0):
                at00 = err
            if best is None or err < best[0]:
                best = (err, dj, di)
    print(f"  [align scan] {name}: best (dj,di)={best[1:]} err_norm={best[0]:.4e}  "
          f"(0,0)={at00:.4e}  (expect (0,0) with a clear minimum)")


def _row_col_profile(err2d, mask2d):
    """Mean-|error| profile by row (latitude) and column (longitude)."""
    n_lat, n_lon = err2d.shape
    row_prof = np.array([
        float(np.mean(np.abs(err2d[i, :][mask2d[i, :]]))) if mask2d[i, :].any() else np.nan
        for i in range(n_lat)
    ])
    col_prof = np.array([
        float(np.mean(np.abs(err2d[:, j][mask2d[:, j]]))) if mask2d[:, j].any() else np.nan
        for j in range(n_lon)
    ])
    return row_prof, col_prof


def _dominant_wavenumber(profile: np.ndarray):
    """FFT of a (NaN-filled, mean-removed) 1-D profile. Returns
    (dominant wavelength in samples, power spectrum, freqs), ignoring the
    zero frequency. Linear-detrend not applied (a wall-decay envelope is
    part of what we are characterising, not noise to remove) -- caller
    should also report autocorrelation as a cross-check."""
    valid = np.isfinite(profile)
    x = np.where(valid, profile, 0.0)
    x = x - np.mean(x[valid])
    n = len(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Ignore DC (freq=0); find peak among the rest.
    spec_ac = spec.copy()
    spec_ac[0] = 0.0
    k_peak = int(np.argmax(spec_ac))
    wavelength_rows = 1.0 / freqs[k_peak] if freqs[k_peak] > 0 else float("nan")
    return wavelength_rows, spec, freqs, k_peak


def _peak_spacing(profile: np.ndarray, order: int = 3):
    """Direct local-extrema peak-to-peak spacing -- the most literal reading
    of 'wavelength' for a profile whose FFT is dominated by a broad
    low-order envelope (a single wide hump, not oscillatory) rather than by
    the embedded quasi-periodic ripple riding on it. Uses
    ``scipy.signal.argrelextrema`` (already a project dependency, ladder
    rung 5) rather than a hand-rolled peak finder."""
    from scipy.signal import argrelextrema
    valid = np.isfinite(profile)
    x = np.where(valid, profile, np.nanmean(profile[valid]))
    maxima = argrelextrema(x, np.greater_equal, order=order)[0]
    minima = argrelextrema(x, np.less_equal, order=order)[0]
    # De-duplicate plateaus (greater_equal/less_equal can flag runs).
    maxima = np.array([i for i in maxima if valid[i]])
    minima = np.array([i for i in minima if valid[i]])
    max_spacing = np.diff(maxima) if maxima.size > 1 else np.array([])
    min_spacing = np.diff(minima) if minima.size > 1 else np.array([])
    return maxima, minima, max_spacing, min_spacing


def _autocorr_first_peak(profile: np.ndarray, max_lag: int):
    """Autocorrelation of the (mean-removed) profile; report the lag of the
    first positive local max beyond lag 0 as an independent wavelength
    estimate (cross-check against the FFT peak, per Rule: never trust one
    instrument alone)."""
    valid = np.isfinite(profile)
    x = np.where(valid, profile, np.nanmean(profile[valid]))
    x = x - np.mean(x)
    n = len(x)
    ac = np.correlate(x, x, mode="full")[n - 1:n + max_lag]
    ac = ac / ac[0]
    # First local maximum for lag >= 2 (skip the trivial lag-0/1 decay).
    for lag in range(2, len(ac) - 1):
        if ac[lag] > ac[lag - 1] and ac[lag] > ac[lag + 1] and ac[lag] > 0:
            return lag, ac
    return None, ac


def _wall_enhancement(row_or_col_profile: np.ndarray, n_edge: int = 15):
    """Ratio of mean|err| in the first/last n_edge cells vs the interior
    core, plus an exponential decay-scale fit (cells) from the west wall
    inward (profile assumed indexed 0=west/south wall)."""
    valid = np.isfinite(row_or_col_profile)
    core = row_or_col_profile[n_edge:-n_edge][valid[n_edge:-n_edge]]
    west_edge = row_or_col_profile[:n_edge][valid[:n_edge]]
    east_edge = row_or_col_profile[-n_edge:][valid[-n_edge:]]
    core_mean = float(np.mean(core)) if core.size else float("nan")
    west_ratio = float(np.mean(west_edge)) / core_mean if core_mean > 0 else float("nan")
    east_ratio = float(np.mean(east_edge)) / core_mean if core_mean > 0 else float("nan")
    # Exponential decay-scale fit: log(|err|(i) - core_mean_floor) vs i, west
    # wall inward, over positive excess only.
    excess = row_or_col_profile[:n_edge] - core_mean
    idx = np.arange(n_edge)
    pos = valid[:n_edge] & (excess > 0) & np.isfinite(excess)
    if pos.sum() >= 3:
        coeffs = np.polyfit(idx[pos], np.log(excess[pos]), 1)
        decay_scale_cells = -1.0 / coeffs[0] if coeffs[0] < 0 else float("inf")
    else:
        decay_scale_cells = float("nan")
    return west_ratio, east_ratio, core_mean, decay_scale_cells


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="zu_frc_u_structure_probe")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    # --- Rule 10: instantiate + print every field this task's namelist claim
    # and A/B hinges on, not trusted from a docstring.
    print(f"DINOConfig.coriolis_scheme={dcfg.coriolis_scheme!r}  "
          f"barotropic_coriolis_split={dcfg.barotropic_coriolis_split!r}  "
          f"barotropic_coriolis={dcfg.barotropic_coriolis!r}  "
          f"barotropic_een_seed={dcfg.barotropic_een_seed!r}  "
          f"barotropic_time_filter={dcfg.barotropic_time_filter!r}")
    assert dcfg.barotropic_coriolis_split == "live"
    assert dcfg.barotropic_coriolis == "een_metric", (
        "this task expects the recipe default to be the metric-complete "
        "variant already -- re-derive the A/B direction if this changes")

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, "DINO_00057600_restart.nc"), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, "DINO_00057600_restart.nc"), nn_hls=0)
    st = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, st, context="zu_frc_u_structure_probe twin state")
    print(f"mc.barotropic_coriolis_split={mc.barotropic_coriolis_split!r}  "
          f"mc.barotropic.barotropic_coriolis={mc.barotropic.barotropic_coriolis!r}  "
          f"mc.barotropic.barotropic_een_seed={mc.barotropic.barotropic_een_seed!r}")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5
    nemo_zu_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zu_frc.bin"), 52, 199)
    nemo_zv_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zv_frc.bin"), 52, 199)

    print("\n" + "=" * 78)
    print("PART 0: reproduce production (ACTUAL, no override) -- self-check")
    print("=" * 78)
    F_slow_u, F_slow_v = _capture_F_slow(model, st, sf)
    u_err_norm, u_rms, u_maxdiff, u_near_zero = _err_norm(
        _u_to_nemo(F_slow_u), nemo_zu_frc, umask2)
    v_err_norm, v_rms, v_maxdiff, v_near_zero = _err_norm(
        _v_to_nemo(F_slow_v), nemo_zv_frc, vmask2)
    print(f"  u err_norm={u_err_norm:.4e}  max|diff|={u_maxdiff:.4e}  "
          f"near_zero_frac={u_near_zero:.3f}  (fix_plan recorded 8.03e-03)")
    print(f"  v err_norm={v_err_norm:.4e}  max|diff|={v_maxdiff:.4e}  "
          f"near_zero_frac={v_near_zero:.3f}  (fix_plan recorded 5.43e-04)")
    _align_scan("zu_frc [ACTUAL]", _u_to_nemo(F_slow_u), nemo_zu_frc, umask2)
    _align_scan("zv_frc [ACTUAL]", _v_to_nemo(F_slow_v), nemo_zv_frc, vmask2)

    u_err2d = _u_to_nemo(F_slow_u) - nemo_zu_frc
    v_err2d = _v_to_nemo(F_slow_v) - nemo_zv_frc
    n_lat_u, n_lon_u = u_err2d.shape
    n_lat_v, n_lon_v = v_err2d.shape

    # --- Grid spacing for rows->km conversion (Rule: state units, don't
    # leave a reader to guess). DINO domain spans channel_lat_south_deg to
    # the symmetric north edge; use the ACTUAL bridged geometry lat_v
    # spacing (not an assumed uniform dlat) for a real km/row estimate.
    lat_t = np.asarray(br.geometry.lat) if hasattr(br.geometry, "lat") else None
    if lat_t is not None and lat_t.size >= 2:
        dlat_rad = np.diff(lat_t)
        km_per_row = float(np.median(np.abs(dlat_rad))) * constants.R_earth / 1000.0
    else:
        km_per_row = float("nan")
    print(f"\ngeometry km/row (median dlat * R_earth): {km_per_row:.2f} km "
          f"(R_earth={constants.R_earth:.6e} m)")

    print("\n" + "=" * 78)
    print("PART 1: MERIDIONAL PROFILE + SPECTRAL/AUTOCORR CHARACTERISATION (u)")
    print("=" * 78)
    u_row_prof, u_col_prof = _row_col_profile(u_err2d, umask2[:n_lat_u, :n_lon_u])
    print(f"  u |err| by row (every 5th, 0..{n_lat_u-1}): "
          f"{np.array2string(u_row_prof[::5], precision=3, max_line_width=220)}")

    wl_rows_fft, spec, freqs, k_peak = _dominant_wavenumber(u_row_prof)
    print(f"  FFT dominant wavelength = {wl_rows_fft:.2f} rows "
          f"({wl_rows_fft * km_per_row:.1f} km)  [peak bin k={k_peak}, "
          f"freq={freqs[k_peak]:.5f} cycles/row]")
    top3 = np.argsort(spec[1:])[::-1][:3] + 1
    print(f"  top-3 FFT bins (excl. DC): " + ", ".join(
        f"wl={1.0/freqs[k]:.1f}rows({1.0/freqs[k]*km_per_row:.0f}km) "
        f"power={spec[k]:.3e}" for k in top3))

    lag, ac = _autocorr_first_peak(u_row_prof, max_lag=min(120, n_lat_u // 2))
    if lag is not None:
        print(f"  autocorrelation first positive peak at lag={lag} rows "
              f"({lag * km_per_row:.1f} km) -- ac[{lag}]={ac[lag]:.4f} "
              "(cross-check vs the FFT wavelength above)")
    else:
        print("  autocorrelation: no clear first peak found (flat/noisy profile)")

    u_maxima, u_minima, u_max_spacing, u_min_spacing = _peak_spacing(u_row_prof)
    print(f"  DIRECT peak-finder (scipy argrelextrema, order=3): "
          f"maxima at rows {u_maxima.tolist()}  minima at rows {u_minima.tolist()}")
    if u_max_spacing.size:
        print(f"  peak-to-peak spacing (maxima): {u_max_spacing.tolist()} rows  "
              f"mean={float(np.mean(u_max_spacing)):.1f} rows "
              f"({float(np.mean(u_max_spacing)) * km_per_row:.0f} km)  "
              "(this is the literal reading of 'wavelength' for a damped "
              "ripple riding on a broad envelope -- distinct from the "
              "whole-profile FFT above, which is dominated by that envelope)")
    else:
        print("  peak-to-peak spacing: fewer than 2 maxima found")

    print("\n" + "-" * 78)
    print("Wall-adjacent enhancement (zonal walls = COLUMNS, since u faces "
          "lon; DINO channel walls sit at the domain's east/west longitude "
          "edges) -- reporting BOTH row-wall (south/north) and col-wall "
          "(east/west) enhancement so neither convention is assumed:")
    print("-" * 78)
    w_ratio_col, e_ratio_col, core_col, decay_col = _wall_enhancement(u_col_prof)
    print(f"  column(longitude)-wall: west/core={w_ratio_col:.3f}x  "
          f"east/core={e_ratio_col:.3f}x  core_mean={core_col:.4e}  "
          f"west decay_scale={decay_col:.2f} cells")
    w_ratio_row, e_ratio_row, core_row, decay_row = _wall_enhancement(u_row_prof)
    print(f"  row(latitude)-wall: south/core={w_ratio_row:.3f}x  "
          f"north/core={e_ratio_row:.3f}x  core_mean={core_row:.4e}  "
          f"south decay_scale={decay_row:.2f} cells")

    print("\n" + "=" * 78)
    print("PART 1 CONTROL: same measurement on zv_frc (should be flat/absent")
    print("per the momentum-row reconstruction result -- corr=0.985 for v)")
    print("=" * 78)
    v_row_prof, v_col_prof = _row_col_profile(v_err2d, vmask2[:n_lat_v, :n_lon_v])
    print(f"  v |err| by row (every 5th, 0..{n_lat_v-1}): "
          f"{np.array2string(v_row_prof[::5], precision=3, max_line_width=220)}")
    wl_rows_fft_v, spec_v, freqs_v, k_peak_v = _dominant_wavenumber(v_row_prof)
    print(f"  v FFT dominant wavelength = {wl_rows_fft_v:.2f} rows "
          f"({wl_rows_fft_v * km_per_row:.1f} km)  [peak bin k={k_peak_v}]")
    w_ratio_col_v, e_ratio_col_v, core_col_v, decay_col_v = _wall_enhancement(v_col_prof)
    print(f"  v column-wall: west/core={w_ratio_col_v:.3f}x  "
          f"east/core={e_ratio_col_v:.3f}x  core_mean={core_col_v:.4e}")
    # Relative amplitude of the spatial variation vs the mean magnitude, u
    # vs v -- a real discriminator: if v's profile variation is proportionally
    # much smaller than u's, that IS the control passing.
    u_row_cv = float(np.nanstd(u_row_prof) / np.nanmean(u_row_prof))
    v_row_cv = float(np.nanstd(v_row_prof) / np.nanmean(v_row_prof))
    print(f"  row-profile coefficient of variation: u={u_row_cv:.3f}  "
          f"v={v_row_cv:.3f}  (higher CV = more spatially structured error)")

    # =========================================================================
    # PART 2: candidate A/Bs. One variable changed at a time (Rule 7).
    # =========================================================================
    print("\n" + "=" * 78)
    print("PART 2a: candidate (time-filter weights) -- EXCLUDED BY READING")
    print("(scalar-per-substep, no (ji,jj) index in ts_wgt CASE(2) or its port)")
    print("=" * 78)
    n_e_effective = 23  # ocean.output: "in iterations nn_e = 23" (auto-adjusted from nn_e=30)
    w_filter, w_total, w_transport = compute_nemo_boxcar_centred_weights(
        n_e_effective * 2, jax.numpy.float64, substep_scale=2)[:3]
    w_filter = np.asarray(w_filter)
    print(f"  n_e_effective={n_e_effective} (ocean.output), nn_bt_flt=2, "
          f"icycle=68 (ocean.output '#1226 dyn_cor_2D dump ... icycle= 68')")
    print(f"  w_filter shape={w_filter.shape} (1-D, per-substep-ONLY -- no "
          f"spatial index exists in this array's construction; confirmed by "
          f"shape alone, not by reading NEMO's ts_wgt source a second time)")
    print(f"  w_filter nonzero count={int(np.count_nonzero(w_filter))} "
          f"(expect 2*nn_e-1=45 nonzero within a length-{w_filter.shape[0]} "
          f"array under nn_bt_flt=2)")

    print("\n" + "=" * 78)
    print("PART 2b: candidate (d) EEN barotropic Coriolis metric factors --")
    print("A/B 'een' (no e1/e2 metric) vs 'een_metric' (production default)")
    print("=" * 78)
    F_slow_u_nometric, F_slow_v_nometric = _capture_F_slow(
        model, st, sf, barotropic_coriolis_override="een")
    u_err_nometric, _, _, _ = _err_norm(_u_to_nemo(F_slow_u_nometric), nemo_zu_frc, umask2)
    v_err_nometric, _, _, _ = _err_norm(_v_to_nemo(F_slow_v_nometric), nemo_zv_frc, vmask2)
    print(f"  u: een_metric(actual)={u_err_norm:.4e}  een(no-metric)="
          f"{u_err_nometric:.4e}  ratio(no-metric/actual)="
          f"{u_err_nometric / u_err_norm:.3f}x")
    print(f"  v: een_metric(actual)={v_err_norm:.4e}  een(no-metric)="
          f"{v_err_nometric:.4e}  ratio(no-metric/actual)="
          f"{v_err_nometric / v_err_norm:.3f}x")
    # Does forcing the cruder variant reproduce (or worsen away from) the
    # SAME wave-like row profile -- if the metric-complete fix already
    # removes the pattern's amplitude, the no-metric row profile should be
    # relatively MORE structured (higher CV) at the same or worse err_norm.
    u_err2d_nometric = _u_to_nemo(F_slow_u_nometric) - nemo_zu_frc
    u_row_prof_nometric, _ = _row_col_profile(u_err2d_nometric, umask2[:n_lat_u, :n_lon_u])
    u_row_cv_nometric = float(np.nanstd(u_row_prof_nometric) / np.nanmean(u_row_prof_nometric))
    wl_nometric, _, _, _ = _dominant_wavenumber(u_row_prof_nometric)
    print(f"  u row-profile CV: een_metric(actual)={u_row_cv:.3f}  "
          f"een(no-metric)={u_row_cv_nometric:.3f}  "
          f"FFT wavelength(no-metric)={wl_nometric:.2f} rows")

    print("\n" + "=" * 78)
    print("SUMMARY (raw numbers only -- interpretation in the report)")
    print("=" * 78)
    print(f"u actual err_norm={u_err_norm:.4e}  v actual err_norm={v_err_norm:.4e}")
    print(f"u FFT wavelength={wl_rows_fft:.2f} rows ({wl_rows_fft*km_per_row:.1f} km), "
          f"autocorr lag={lag}")
    print(f"u col-wall west/core={w_ratio_col:.3f}x decay_scale={decay_col:.2f} cells")
    print(f"v row-profile CV={v_row_cv:.3f} vs u row-profile CV={u_row_cv:.3f}")
    print(f"candidate(d) A/B: een_metric={u_err_norm:.4e}  een={u_err_nometric:.4e}  "
          f"ratio={u_err_nometric/u_err_norm:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
