#!/usr/bin/env python
"""#1455 Phase 1/2a: OFFLINE alignment of the DINO 90-day twin's SURFACE-HEAT
forcing clock against NEMO's, and the arithmetic that says whether the
mis-alignment can own the southern-band surface warm bias.

NO model run, no GPU, no NEMO run.  Every number here is arithmetic on the
two analytic forcing formulas plus one NEMO output file used as a control.

WHAT IS BEING COMPARED
----------------------
NEMO's DINO surface heat forcing (``cfgs/DINO/MY_SRC/usrdef_sbc.F90``,
``nn_forcingtype=4``) is, per step ``kt``:

    usrdef_sbc.F90:536-547   ztime  = REAL(kt)*rn_Dt/3600 - (nyear-1)*24*360   [hours]
                             cos1   = COS( (ztime - 171*24)/4320 * pi )   ! 21 June  phase
                             cos2   = COS( (ztime - 201*24)/4320 * pi )   ! 21 July  phase
    usrdef_sbc.F90:216-217   T*_s   = rn_tstar_s - 0.5*cos2   ;   T*_n = rn_tstar_n + 3.0*cos2
    usrdef_sbc.F90:234-238   T*(phi)= T*_ns + (rn_tstar_eq - T*_ns)
                                      * SIN( pi*(phi + rn_phi_max)/(rn_phi_max - rn_phi_min) )
    usrdef_sbc.F90:268       Qsr    = MAX( 230*COS( pi*(phi - 23.5*cos1)/180 ), 0 )
    usrdef_sbc.F90:257-258   qtot   = rn_trp*( SST - T* )            ! rn_trp = -40 W/m2/K
    usrdef_sbc.F90:279       qns    = qtot - Qsr

    ==> the NET column heat input is qns + Qsr = qtot = A*(T* - SST) with
        A = |rn_trp| = 40 W/m2/K.  Qsr is a REDISTRIBUTION of part of that
        fixed total down the column (traqsr.F90:665-712), NOT an extra source.
        A Qsr error therefore only moves heat WITHIN the column; only a T*
        error changes the column's net heat input.  This distinction is what
        the two blocks of output below are kept separate for.

The twin harness (``kamm_twin_90d.py:433-439``) starts from NEMO's kt=5760
restart (RUN_90D_TWIN/ocean.output:711 -- kt 5761 is 0001/07/01, nday_year
181) but passes ``t_seconds=(k+1)*DT`` with k counting from 0, i.e. it
evaluates the SAME formulas at model day 0.03..90 while NEMO evaluates them
at day 180.03..270.

Because both seasonal cosines have a 360-day period, a 180-day offset is
EXACTLY an antiphase flip (cos(x+pi) = -cos(x)); this probe measures the
resulting T* and Qsr differences rather than asserting them.

OUTPUT BLOCKS
-------------
  [SELF-CHECKS]  three fatal controls, run before any number is printed
  [T-STAR]       per-band mean dT* over the 90-day window, the implied net
                 surface heat-flux difference A*dT*, and a 1-box mixed-layer
                 response integrated with the ACTUAL dT*(t)
  [Q-SR]         per-band mean dQsr and the fraction of it absorbed BELOW the
                 mixed layer (NEMO 2-band, rn_abs/rn_si0/rn_si1 read from
                 RUN_90D_TWIN/ocean.output:942-944), i.e. the part of the Qsr
                 error that is NOT pure in-mixed-layer redistribution

This probe prints numbers and the already-published measured values side by
side.  It does not print a verdict.

Usage
-----
    python twin_forcing_clock_alignment.py [--selftest]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Configuration read from the ORACLE's resolved run, not from memory.
# RUN_90D_TWIN/namelist_cfg:22-46 (namusr_def) and :117 (namdom rn_Dt);
# RUN_90D_TWIN/ocean.output:296 (nn_it000), :711 (kt 5761 = 0001/07/01),
# :942-944 (rn_abs / rn_si0 / rn_si1).
# ---------------------------------------------------------------------------
DINO_ROOT = os.environ.get(
    "DINO_ORACLE_ROOT", "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO")
MESH_MASK = f"{DINO_ROOT}/RUN_TRAJ/mesh_mask.nc"
OCEAN_OUTPUT = f"{DINO_ROOT}/RUN_90D_TWIN/ocean.output"
RUN_1Y_GLOB = f"{DINO_ROOT}/RUN_1Y/DINO_1y_00010101_00011230_grid_T_*.nc"

DT = 2700.0            # namdom rn_Dt
KT0 = 5760             # twin restart step (DINO_00005760_restart)
N_STEPS = 2880         # 90 d x 32 steps/d
STEPS_PER_YEAR = 11520 # 360 d x 32

RN_PHI_MIN, RN_PHI_MAX = -70.0, 70.0
RN_TSTAR_S, RN_TSTAR_N, RN_TSTAR_EQ = -0.5, 5.0, 27.0
AMP_S, AMP_N = 0.5, 3.0
A_THETA = 40.0         # |rn_trp| [W/m2/K]
QSR_AMP, DECL_AMP = 230.0, 23.5
RN_ABS, RN_SI0, RN_SI1 = 0.58, 0.35, 23.0

RHO0, CP = 1026.0, 3991.86   # DINOConfig.rho_0 / .c_p (= NEMO rho0 / rcp)
H_ML = 74.0            # depth over which the measured anomaly is flat (+-1.5%)

# Gate window (acceptance_gate_90d southern-band metric, rows 14..31).
SOUTH_ROWS = slice(14, 32)
# A northern mirror band, same |lat| span, for the meridional-dipole column.
NORTH_ROWS = slice(199 - 32, 199 - 14)

# Published measured values this probe is read against (commit a5183778c).
# thermal part of the day-N sigma gap, converted there to a temperature:
MEASURED_DT_K = {30: 0.098, 60: 0.159, 90: 0.170}


# ---------------------------------------------------------------------------
# Independent transcription of usrdef_sbc.F90 (NOT a call into legoesm).
# ---------------------------------------------------------------------------
def nemo_seasonal_cosines(kt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """usrdef_sbc.F90:536-547.  ``kt`` absolute NEMO step index (year 1)."""
    ztime = kt.astype(np.float64) * DT / 3600.0        # hours since 0001-01-01
    half = 24.0 * 360.0 / 2.0                          # 4320 h
    c1 = np.cos((ztime - 171.0 * 24.0) / half * np.pi)
    c2 = np.cos((ztime - 201.0 * 24.0) / half * np.pi)
    return c1, c2


def nemo_t_star(lat_deg: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """usrdef_sbc.F90:216-217, 233-239.  Shapes broadcast (nt,1) x (1,nlat)."""
    t_s = RN_TSTAR_S - AMP_S * c2
    t_n = RN_TSTAR_N + AMP_N * c2
    t_ns = np.where(lat_deg <= 0.0, t_s, t_n)
    prof = np.sin(np.pi * (lat_deg + RN_PHI_MAX) / (RN_PHI_MAX - RN_PHI_MIN))
    return t_ns + (RN_TSTAR_EQ - t_ns) * prof


def nemo_qsr(lat_deg: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """usrdef_sbc.F90:268 (daily mean; ln_diu_cyc=.false. -> qsr = this)."""
    return np.maximum(QSR_AMP * np.cos(np.pi * (lat_deg - DECL_AMP * c1) / 180.0), 0.0)


def qsr_fraction_below(depth_m: float) -> float:
    """NEMO 2-band transmission at ``depth_m`` (traqsr.F90:665-712 zatt)."""
    return RN_ABS * np.exp(-depth_m / RN_SI0) + (1.0 - RN_ABS) * np.exp(-depth_m / RN_SI1)


# ---------------------------------------------------------------------------
def read_gphit() -> np.ndarray:
    return _read_mesh()[0]


def _read_mesh() -> tuple[np.ndarray, np.ndarray]:
    """(1-D T-point latitude, surface-level tmask) from the oracle mesh_mask."""
    import netCDF4 as nc
    with nc.Dataset(MESH_MASK) as d:
        g = np.asarray(d.variables["gphit"][:], dtype=np.float64).squeeze()
        tm = np.asarray(d.variables["tmask"][:], dtype=np.float64).squeeze()[0] > 0.5
    if not np.isfinite(g).all():
        raise SystemExit("mesh_mask gphit carries non-finite values")
    if float(np.max(np.abs(g - g[:, :1]))) != 0.0:
        raise SystemExit("gphit is not zonally constant; the 1-D reduction is invalid")
    if tm.shape != g.shape:
        raise SystemExit(f"tmask surface level {tm.shape} != gphit {g.shape}")
    return g[:, 0], tm


def stitch_1y_qsr() -> np.ndarray:
    """Annual-mean ``soshfldo`` (= qsr) stitched from the 16 RUN_1Y grid_T tiles.

    The wet set comes from mesh_mask ``tmask(k=1)``, NOT from the netCDF fill
    mask: XIOS writes the two meridional wall columns as a genuine 0.0 rather
    than as missing, so a fill-mask reduction silently averages 50 ocean cells
    with 2 zeros and reports 50/52 = 0.9615 of the true value on every row that
    HAS walls (the channel rows, which have none, came out exact) -- an
    instrument artifact that was measured, not assumed.
    """
    import netCDF4 as nc
    files = sorted(glob.glob(RUN_1Y_GLOB))
    if not files:
        raise SystemExit(f"no RUN_1Y grid_T tiles matched {RUN_1Y_GLOB}")
    full = None
    for f in files:
        with nc.Dataset(f) as d:
            if len(d.dimensions["time_counter"]) != 1:
                raise SystemExit(f"{f}: expected exactly 1 time record")
            ny, nx = (int(v) for v in d.getncattr("DOMAIN_size_global")[::-1])
            i0, j0 = (int(v) for v in d.getncattr("DOMAIN_position_first"))
            i1, j1 = (int(v) for v in d.getncattr("DOMAIN_position_last"))
            v = np.asarray(d.variables["soshfldo"][0], dtype=np.float64)
            if full is None:
                full = np.full((ny, nx), np.nan)
            full[j0 - 1:j1, i0 - 1:i1] = v
    if full is None or not np.isfinite(full).all():
        raise SystemExit("stitched soshfldo carries non-finite values")
    return full


def calendar_control() -> tuple[int, int]:
    """Check ``ztime = kt*rn_Dt`` against NEMO's own logged day-of-year.

    ``ocean.output`` prints one header per new day::

        ======>> time-step =    5761      New day, DATE Y/M/D = 0001/07/01      nday_year = 181

    ``usrdef_sbc.F90:536`` makes the seasonal phase a function of
    ``REAL(kt)*rn_Dt``, so ``floor(kt*rn_Dt/86400) + 1`` must equal the logged
    ``nday_year`` on every one of them.  Returns (n_headers, max |offset|).
    """
    import re
    pat = re.compile(r"time-step\s*=\s*(\d+).*?nday_year\s*=\s*(\d+)")
    offs = []
    with open(OCEAN_OUTPUT, "r", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                kt, doy = int(m.group(1)), int(m.group(2))
                offs.append(int(np.floor(kt * DT / 86400.0)) + 1 - doy)
    if not offs:
        raise SystemExit(f"no 'New day' headers parsed from {OCEAN_OUTPUT}")
    return len(offs), int(np.max(np.abs(offs)))


# ---------------------------------------------------------------------------
def band_mean(x: np.ndarray, rows: slice) -> float:
    return float(np.mean(x[..., rows]))


def one_box_response(dtstar_t: np.ndarray, h: float) -> np.ndarray:
    """Linear 1-box mixed-layer response to the T* difference.

    dT'/dt = (A/(rho0*cp*h)) * (dT*(t) - T'),  T'(0)=0, explicit Euler at DT.
    This is a FIRST-ORDER estimate of the SST difference the T* mis-phasing
    would produce with no advection and a fixed mixing depth; it is not a
    model run and is labelled as such wherever quoted.
    """
    lam = A_THETA / (RHO0 * CP * h)
    out = np.empty_like(dtstar_t)
    tp = 0.0
    for i, ds in enumerate(dtstar_t):
        tp = tp + DT * lam * (ds - tp)
        out[i] = tp
    return out


def selftest() -> None:
    kt = np.arange(1, 100, dtype=np.float64)
    c1a, c2a = nemo_seasonal_cosines(kt)
    c1b, c2b = nemo_seasonal_cosines(kt + 180.0 * 32.0)
    assert np.max(np.abs(c1a + c1b)) < 1e-12, "180 d is not antiphase for cos1"
    assert np.max(np.abs(c2a + c2b)) < 1e-12, "180 d is not antiphase for cos2"

    # The legoESM production formula must equal this transcription at the SAME
    # absolute time -- if it does not, the mis-alignment below is not the only
    # difference and the rest of the probe is unreadable.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax.numpy as jnp
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe, dino_Q_sr_seasonal, dino_T_star_seasonal)
    cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    lat = np.linspace(-69.0, 69.0, 47)
    for k in (1, 977, 5761, 8640):
        t = k * DT
        c1, c2 = nemo_seasonal_cosines(np.array([float(k)]))
        d_t = float(np.max(np.abs(
            np.asarray(dino_T_star_seasonal(jnp.asarray(lat), t, cfg))
            - nemo_t_star(lat, c2[0]))))
        d_q = float(np.max(np.abs(
            np.asarray(dino_Q_sr_seasonal(jnp.asarray(lat), t, cfg))
            - nemo_qsr(lat, c1[0]))))
        assert d_t < 1e-10, f"T* transcription mismatch at kt={k}: {d_t:.3e}"
        assert d_q < 1e-10, f"Qsr transcription mismatch at kt={k}: {d_q:.3e}"

    # Non-vacuity: with a ZERO restart offset the two clocks coincide and every
    # difference this probe reports must be exactly 0.
    lat1 = _read_mesh()[0][None, :]
    ks = np.arange(1, 33, dtype=np.float64)[:, None]
    _, c2r = nemo_seasonal_cosines(ks)
    _, c2a0 = nemo_seasonal_cosines(ks + 0.0)
    assert float(np.max(np.abs(nemo_t_star(lat1, c2r) - nemo_t_star(lat1, c2a0)))) == 0.0
    _, c2a5 = nemo_seasonal_cosines(ks + KT0)
    assert float(np.max(np.abs(nemo_t_star(lat1, c2r) - nemo_t_star(lat1, c2a5)))) > 0.1, (
        "the KT0 offset produced no T* difference -- the probe cannot detect one")
    print("[SELF-CHECKS] antiphase identity, legoESM-vs-transcription agreement "
          "(4 steps, max err < 1e-10), and zero-offset non-vacuity: all PASS")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    selftest()

    lat, tmask = _read_mesh()               # (199,), (199,52)
    lat1 = lat[None, :]
    k = np.arange(1, N_STEPS + 1, dtype=np.float64)[:, None]

    c1_rel, c2_rel = nemo_seasonal_cosines(k)          # harness clock
    c1_abs, c2_abs = nemo_seasonal_cosines(k + KT0)    # NEMO's clock

    ts_rel = nemo_t_star(lat1, c2_rel)
    ts_abs = nemo_t_star(lat1, c2_abs)
    dts = ts_rel - ts_abs                              # (nt, nlat)

    qs_rel = nemo_qsr(lat1, c1_rel)
    qs_abs = nemo_qsr(lat1, c1_abs)
    dqs = qs_rel - qs_abs

    print()
    print("[CLOCKS] twin harness t = (k+1)*dt  ->  seasonal day "
          f"{float(k[0,0])*DT/86400:.3f} .. {float(k[-1,0])*DT/86400:.3f}")
    print(f"         NEMO         t = kt*dt      ->  seasonal day "
          f"{float(k[0,0]+KT0)*DT/86400:.3f} .. {float(k[-1,0]+KT0)*DT/86400:.3f}"
          f"   (restart kt={KT0})")

    print()
    print("[T-STAR]  dT* = harness-clock T* minus NEMO-clock T*  [K]"
          "   (this is the ONLY term that changes the column's net heat input)")
    print(f"  {'band':22s} {'lat span':>18s} {'dT* d0':>9s} {'dT* d90':>9s} "
          f"{'dT* mean':>9s} {'A*dT* [W/m2]':>13s}")
    for name, rows in (("gate south 14..31", SOUTH_ROWS),
                       ("north mirror", NORTH_ROWS)):
        span = f"{lat[rows][0]:+.2f}..{lat[rows][-1]:+.2f}"
        d0 = float(np.mean(dts[0, rows]))
        d90 = float(np.mean(dts[-1, rows]))
        dm = band_mean(dts, rows)
        print(f"  {name:22s} {span:>18s} {d0:+9.3f} {d90:+9.3f} {dm:+9.3f} {A_THETA*dm:+13.1f}")

    dts_south = np.mean(dts[:, SOUTH_ROWS], axis=1)
    resp = one_box_response(dts_south, H_ML)
    print()
    print(f"  1-box mixed-layer response to dT*(t), h = {H_ML:.0f} m "
          f"(tau = rho0*cp*h/A = {RHO0*CP*H_ML/A_THETA/86400:.1f} d), vs the "
          "measured thermal gap:")
    print(f"  {'day':>5s} {'predicted dSST [K]':>20s} {'measured dSST [K]':>19s} {'ratio':>7s}")
    for day, meas in sorted(MEASURED_DT_K.items()):
        pred = float(resp[day * 32 - 1])
        print(f"  {day:5d} {pred:20.4f} {meas:19.4f} {pred/meas:7.2f}")

    frac_below = qsr_fraction_below(H_ML)
    print()
    print("[Q-SR]   dQsr = harness-clock Qsr minus NEMO-clock Qsr  [W/m2]."
          "  Qsr is INSIDE qtot, so only the")
    print(f"         part absorbed BELOW the {H_ML:.0f} m mixed layer "
          f"({100*frac_below:.2f}% by the NEMO 2-band profile) leaves it.")
    print(f"  {'band':22s} {'dQsr d0':>9s} {'dQsr d90':>9s} {'dQsr mean':>10s} "
          f"{'below-ML [W/m2]':>16s} {'90 d dSST_eq [K]':>17s}")
    for name, rows in (("gate south 14..31", SOUTH_ROWS),
                       ("north mirror", NORTH_ROWS)):
        d0 = float(np.mean(dqs[0, rows]))
        d90 = float(np.mean(dqs[-1, rows]))
        dm = band_mean(dqs, rows)
        below = dm * frac_below
        # heat exported below the ML over 90 d, expressed as an ML temperature
        dsst = -below * (N_STEPS * DT) / (RHO0 * CP * H_ML)
        print(f"  {name:22s} {d0:+9.1f} {d90:+9.1f} {dm:+10.1f} {below:+16.2f} {dsst:+17.4f}")

    # ---- control: the transcription against the oracle's OWN annual-mean qsr
    obs = stitch_1y_qsr()
    wet = tmask
    kt_year = np.arange(1, STEPS_PER_YEAR + 1, dtype=np.float64)[:, None]
    c1_year, _ = nemo_seasonal_cosines(kt_year)
    pred_year = np.mean(nemo_qsr(lat1, c1_year), axis=0)          # (nlat,)
    obs_zonal = np.array([
        np.mean(obs[j][wet[j]]) if wet[j].any() else np.nan for j in range(obs.shape[0])])
    ok = np.isfinite(obs_zonal)
    err = np.abs(pred_year[ok] - obs_zonal[ok])
    print()
    print("[CONTROL] transcription vs the oracle's OWN RUN_1Y annual-mean "
          "soshfldo (=qsr), zonal mean over wet cells:")
    print(f"          rows compared {int(ok.sum())}  max|err| {float(err.max()):.4f} W/m2  "
          f"mean|err| {float(err.mean()):.4f} W/m2  "
          f"peak observed {float(np.nanmax(obs_zonal)):.2f} W/m2")
    if not np.isfinite(err).all():
        raise SystemExit("non-finite error in the RUN_1Y control")

    n_days_ck, max_off = calendar_control()
    print()
    print("[CONTROL] the clock convention itself, against NEMO's OWN printed "
          "calendar in RUN_90D_TWIN/ocean.output:")
    print(f"          '(kt*rn_Dt/86400) + 1' vs the logged nday_year over "
          f"{n_days_ck} day headers: max offset {max_off} day")


if __name__ == "__main__":
    main()
