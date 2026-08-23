"""Sea-surface-height WAVE-FIELD twin: legoESM vs NEMO, sampled every model step.

WHY THIS PROBE EXISTS
---------------------
The DINO twin has been certified step-by-step and compared at the CLIMATE end
(90-day means).  Nothing in between has ever been looked at, and the saved twin
output is DAILY while a barotropic gravity wave crosses the basin in roughly
four hours.  That was the motivation.

WHAT THE FIRST RUN ACTUALLY FOUND, and it corrects the motivation: there is no
wave in the free run to alias.  NEMO's own free surface at day 180 changes by
1.8e-4 m per step against an 0.83 m field, and 99.99% of its temporal variance
sits at periods longer than a day.  The basin radiated its free gravity-wave
energy away during 180 days of spin-up under smooth seasonal forcing.  The
split-explicit time-averaging is NOT the reason -- the 90-minute boxcar passes
94% at 8 h -- so per-step sampling of a free run cannot answer a wave question
however finely it samples.

Hence two lanes, and the second is the one that answers the question asked:
  FREE      -- both models from the same restart, no perturbation.  Measures
               whether legoESM injects free-surface structure the oracle does
               not have, and where.
  IMPULSE   -- an IDENTICAL Gaussian sea-surface bump added to both models'
               initial condition (see ``make_impulse_restart``), which excites
               the barotropic response on purpose and turns a null-signal
               experiment into a controlled one.

A further caveat that no sampling choice can remove: at 2700 s a wave at
sqrt(gH) = 210 m/s crosses one 64 km cell in 0.11 of a step, so cell-to-cell
PHASE LAG is unresolvable here in either direction.  Propagation shows up
SPATIALLY (where the front has got to at each sample), not temporally, and
``propagation_lag`` exists to make that limit explicit rather than assumed.

WHAT IT COMPARES
----------------
Both models are started from the SAME NEMO day-180 restart
(DINO_00005760_restart.nc) and integrated 160 steps (5 days at rn_Dt=2700 s).
Sea surface height is sampled EVERY step on both sides -- 45-minute sampling,
finer than the "hourly" the task asked for, because the NEMO time step is
2700 s and an hourly cadence is not an integer number of steps.

TIME CENTERING (the trap this probe had to get right)
-----------------------------------------------------
NEMO writes its restart in stpmlf.F90 at

    stpmlf.F90:621-624   Nrhs = Nbb ; Nbb = Nnn ; Nnn = Naa ; Naa = Nrhs   ! swap
    stpmlf.F90:634       IF( lrst_oce ) CALL rst_write( kstp, Nbb, Nnn )

i.e. the swap happens BEFORE the write, and rst_write is called WITHOUT a Kaa
argument, so the MLF branch of restart.F90 runs:

    restart.F90:191      iom_rstput( ... 'sshb', ssh(:,:,Kbb) )   ! before
    restart.F90:197      iom_rstput( ... 'sshn', ssh(:,:,Kmm) )   ! now

With the POST-swap indices, Kmm == the pre-swap Naa.  Therefore

    sshn in DINO_<kt>_restart.nc  ==  sea surface height AFTER kt steps,

and that -- not sshb -- is the field that must be lined up with legoESM's eta
after the same number of steps.  (The same argument makes 'un'/'vn' the
after-level 3-D velocities; 'ub'/'vb' are the Asselin-filtered previous level.)

The identity is checked mechanically at extraction time: the sshb of dump
kt+1 must equal a filtered combination of neighbouring levels, and -- the
cheap decisive one -- dump kt's sshn is NOT equal to dump kt+1's sshn.

MASKING
-------
NEMO fills land with exact 0.0 in the restart.  A comparison that treats those
as data reports a difference of zero over half the grid and dilutes every
normalized statistic.  The wet mask is taken from the model grid (not from
"where the field is zero", which is a structural-zero trap), and the probe
plants a synthetic violation in a dry cell to prove the mask is actually
applied -- a masked statistic that does not move when a dry cell is poisoned
is not masking anything.

NaN IS FATAL.  Nothing here uses nanmean/nanmax.

THIS PROBE PRINTS NUMBERS, NOT VERDICTS.  The interpretation lives in the
findings document, after the controls below have passed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import netCDF4
import numpy as np

# NEMO free-surface sampling: DINO runs rn_Dt = 2700 s, so one dump per step
# is a 45-minute sample.  Recorded here so the probe never has to guess it;
# it is CHECKED against the 'rdt' variable in every restart it reads.
NEMO_RDT_S = 2700.0
# NEMO namelist_ref:73 rn_atfp -- the Asselin filter weight, needed for the
# time-level identity below.
#
# That identity is EXACT here, and that is CONFIGURATION-DEPENDENT rather than
# a property of NEMO: ssh_atf (MY_SRC/sshwzv.F90:519-528) also subtracts a
# freshwater-forcing term, which vanishes in DINO because rn_emp_prop = 0 and
# ln_rnf / ln_isf are false.  A DINO variant with a real emp would leave a
# smooth residual of order rn_atfp*rn_Dt*d(emp)/rho0 and would be wrongly
# accused by the gate below of a broken dump-to-time-level mapping.
NEMO_ATFP = 0.1


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def provenance(extra: dict) -> dict:
    """Stamp git SHA, wall clock, dtype and input identity onto every output."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(
            ["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", here, "status", "--porcelain"], text=True).strip()
    except Exception as exc:                      # pragma: no cover - env dep
        sha, dirty = f"UNKNOWN({exc})", "UNKNOWN"
    prov = {
        "git_sha": sha,
        "git_dirty": bool(dirty),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        # of the ANALYSIS.  The dtype each artifact was STORED at is recorded
        # separately, per side, by load_side -- a hardcoded stamp here would
        # certify a precision nothing checked.
        "dtype": "float64",
        "argv": sys.argv,
    }
    prov.update(extra)
    return prov


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# NEMO side: stitch the per-rank restart tiles, one dump per step
# --------------------------------------------------------------------------
def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def _load_rebuild():
    """Reuse the committed tile stitcher rather than re-deriving one."""
    sys.path.insert(0, os.path.join(_repo_root(), "scripts", "validate",
                                    "ocean_fidelity"))
    from rebuild_nemo_restart import rebuild  # noqa: E402,PLC0415
    return rebuild


def extract_nemo(run_dir: str, kt0: int, nsteps: int, out: str,
                 fields: tuple[str, ...] = ("sshn", "sshb")) -> dict:
    """Stitch `sshn` (and `sshb`) from every per-step NEMO restart dump.

    Returns the assembled dict; also writes `out` (.npz).

    kt0 is the restart step the run STARTED from (5760), so dump kt0+n holds
    the state after n steps.  Time coordinate t[n] = n * rdt seconds.
    """
    if nsteps < 2:
        raise SystemExit("nsteps must be >= 2 (the time-level identity needs "
                         "two consecutive dumps)")
    exe = os.path.join(run_dir, "nemo.exe")
    binary_md5 = md5(exe) if os.path.exists(exe) else "ABSENT"
    rebuild = _load_rebuild()
    eta = None
    kts = []
    rdt_seen = set()
    for n in range(1, nsteps + 1):
        kt = kt0 + n
        pattern = os.path.join(run_dir, f"DINO_{kt:08d}_restart_*.nc")
        tiles = sorted(glob.glob(pattern))
        if not tiles:
            raise SystemExit(f"missing NEMO dump for kt={kt}: {pattern}")
        got = rebuild(pattern, list(fields))
        for name in fields:
            if name not in got:
                raise SystemExit(f"{name} absent from NEMO dump kt={kt}")
        if eta is None:
            ny, nx = got["sshn"].shape
            eta = {name: np.empty((nsteps, ny, nx), dtype=np.float64)
                   for name in fields}
        for name in fields:
            a = got[name]
            if not np.isfinite(a).all():
                raise SystemExit(
                    f"NaN/Inf in stitched {name} at kt={kt} -- a tile did not "
                    f"cover the whole domain; refusing to continue")
            eta[name][n - 1] = a
        kts.append(kt)
        # rdt sanity: read once per file group, cheap and catches a wrong run
        # The Asselin identity below is a RECURRENCE IN THE ARRAY INDEX, so it
        # is exactly invariant to a uniform shift of the whole window: it pins
        # sshb-against-sshn and the direction of time, NOT the kt0 <-> step-N
        # registration legoESM is lined up against.  That registration is
        # pinned here instead, from each dump's OWN step counter.
        with netCDF4.Dataset(tiles[0]) as ds:
            if "kt" in ds.variables:
                kt_in_file = int(np.squeeze(ds.variables["kt"][:]))
                if kt_in_file != kt:
                    raise SystemExit(
                        f"dump named kt={kt} carries kt={kt_in_file} inside; "
                        "the file<->step registration is wrong")
            elif n == 1:
                raise SystemExit(
                    "the NEMO dumps carry no 'kt' variable, so the "
                    "file<->step registration cannot be pinned; refusing "
                    "rather than trusting the filename")
            if n in (1, nsteps):
                rdt_seen.add(float(np.squeeze(ds.variables["rdt"][:])))

    if rdt_seen != {NEMO_RDT_S}:
        raise SystemExit(f"NEMO rdt {sorted(rdt_seen)} != expected "
                         f"{NEMO_RDT_S}; the run is not the certified twin")

    # geometry + the wet mask donor, taken from tile 0 of the first dump
    geom = rebuild(os.path.join(run_dir, f"DINO_{kt0+1:08d}_restart_*.nc"),
                   ["nav_lat", "nav_lon"])
    for name in ("nav_lat", "nav_lon"):
        if name not in geom or not np.isfinite(geom[name]).all():
            raise SystemExit(f"{name} missing or non-finite in the NEMO dumps")

    # ---- TIME-LEVEL CONTROL, and it has to be one that can FAIL.
    # "consecutive dumps differ" and "sshn differs from sshb" both pass just
    # as happily under a one-dump offset, so they prove nothing on their own.
    # The discriminating check is NEMO's own Asselin filter, written out
    # EXACTLY.  At step n the filter (ssh_atf) acts on the pre-swap now level
    # and its result is what the swap turns into the written `sshb`:
    #
    #     sshb[n] = sshn[n-1] + rn_atfp * ( sshb[n-1] - 2 sshn[n-1] + sshn[n] )
    #
    # The BEFORE level entering the filter at step n is the already-FILTERED
    # now level of step n-1, i.e. sshb[n-1] -- NOT the unfiltered sshn[n-2].
    # Getting that wrong leaves a residual that looks like a small physical
    # term (it is second order in the filter weight and grows with how fast
    # the field is changing: 1.3% of a step's change on a quiet run, 16% on an
    # excited one) and invites exactly the wrong conclusion.  In the correct
    # form the identity is EXACT -- measured 0.0 to the last bit -- which is
    # what makes it a real control: a one-dump offset cannot hide inside a
    # tolerance that is machine zero.
    sn, sb = eta["sshn"], eta["sshb"]
    resid = sb[1:] - (sn[:-1] + NEMO_ATFP * (sb[:-1] - 2 * sn[:-1] + sn[1:]))
    step_change = float(np.max(np.abs(sn[1:] - sn[:-1])))
    resid_max = float(np.max(np.abs(resid)))
    scale = float(np.max(np.abs(sn)))
    if step_change <= 0.0:
        raise SystemExit("the free surface never changes; the time-level "
                         "identity cannot be tested")
    if resid_max > 1.0e-9 * scale:
        raise SystemExit(
            f"TIME-LEVEL IDENTITY FAILED: |sshb - Asselin(sshb,sshn)| = "
            f"{resid_max:.3e} m against a field of {scale:.3e} m. The "
            "identity is exact for THIS configuration (see NEMO_ATFP), so a "
            "residual above roundoff means "
            "sshb and sshn are misaligned or time runs backwards.")
    # NB: a zero residual does NOT on its own prove the mapping is right --
    # the identity is invariant to a uniform shift of the window.  The kt
    # check in the loop above pins the absolute registration; the two together
    # are what close the question.
    d_consec = float(np.max(np.abs(sn[1] - sn[0])))
    d_bn = float(np.max(np.abs(sn[0] - sb[0])))

    payload = {
        "eta": eta["sshn"],
        "eta_before": eta["sshb"],
        "kt": np.asarray(kts, dtype=np.int64),
        "t_seconds": np.arange(1, nsteps + 1, dtype=np.float64) * NEMO_RDT_S,
        "nav_lat": geom["nav_lat"],
        "nav_lon": geom["nav_lon"],
    }
    prov = provenance({
        "source": "NEMO",
        "run_dir": run_dir,
        "binary_md5": binary_md5,
        "kt0": kt0,
        "nsteps": nsteps,
        "rdt_s": NEMO_RDT_S,
        "time_level": ("sshn == ssh(Kmm) with POST-swap indices "
                       "(stpmlf.F90:621-624 swap, :634 rst_write) == state "
                       "AFTER kt steps"),
        "control_max_abs_consecutive_sshn_diff_m": d_consec,
        "control_max_abs_sshn_minus_sshb_m": d_bn,
        "control_asselin_identity_residual_m": resid_max,
        "control_asselin_identity_vs_step_change": resid_max / step_change,
        "rn_atfp": NEMO_ATFP,
    })
    np.savez_compressed(out, provenance=json.dumps(prov), **payload)
    print(json.dumps(prov, indent=2))
    print(f"[shape] eta {payload['eta'].shape}  "
          f"absmax {np.max(np.abs(payload['eta'])):.6e} m")
    print(f"[out] {out}")
    return payload


# --------------------------------------------------------------------------
# controlled excitation
# --------------------------------------------------------------------------
# PRE-REGISTERED impulse, fixed before the perturbed pair was run.  The free
# 5-day run turned out to contain no barotropic wave energy at all (the basin
# radiated it away during 180 days of spin-up), so "do the wave patterns
# match" cannot be answered by watching a quiet ocean.  This puts an
# IDENTICAL free-surface bump into BOTH models' initial condition and compares
# the radiating response, which converts a null-signal experiment into a
# controlled one.
IMPULSE_AMPLITUDE_M = 0.05
IMPULSE_RADIUS_M = 300.0e3
IMPULSE_CENTRE_LAT = -30.0
IMPULSE_CENTRE_LON = 25.0


def make_impulse_restart(restart_in: str, restart_out: str,
                         amplitude_m: float = IMPULSE_AMPLITUDE_M,
                         radius_m: float = IMPULSE_RADIUS_M,
                         centre_lat: float = IMPULSE_CENTRE_LAT,
                         centre_lon: float = IMPULSE_CENTRE_LON,
                         mesh_mask: str | None = None) -> dict:
    """Copy a NEMO restart and add a Gaussian sea-surface bump to it.

    The bump goes on BOTH ``sshn`` and ``sshb``, identically, so the leapfrog
    starts from a consistent pair and the impulse is a displacement rather
    than an artificial time derivative.  Only wet cells are touched.

    This is coordinate-safe on both sides: the DINO restart carries ONLY
    ``sshb``/``sshn`` for the free surface (no ``e3t``/``gdept``/``r3t``
    fields -- verified), and both models derive the quasi-Eulerian stretching
    from the ssh they read against a static ladder taken from the mesh.  So a
    bump in ssh cannot leave one model's vertical coordinate inconsistent
    with the other's.
    """
    import shutil
    mesh_mask = mesh_mask or MESH_MASK
    wet = wet_mask(mesh_mask)
    # EVERY refusal happens before a single byte is written.  A guard that
    # fires after the mutation leaves the exact coordinate-inconsistent
    # restart it exists to prevent sitting on disk, where a retry or an
    # unchecked exit code runs the model from it.
    with netCDF4.Dataset(restart_in) as ds:
        for name in ("e3t", "gdept", "r3t", "e3t_n"):
            if name in ds.variables:
                raise SystemExit(
                    f"{restart_in} carries {name}, which depends on ssh -- "
                    "bumping ssh alone would leave the vertical coordinate "
                    "inconsistent; refusing")
        for name in ("sshn", "sshb"):
            if name not in ds.variables:
                raise SystemExit(f"{name} absent from {restart_in}")
        raw = ds.variables["sshn"][:]
        if np.asarray(raw).squeeze().shape != wet.shape:
            raise SystemExit(
                f"restart ssh {np.asarray(raw).squeeze().shape} does not "
                f"match the mesh {wet.shape}")
        if np.ma.isMaskedArray(raw):
            restart_wet = ~np.ma.getmaskarray(raw).squeeze()
            if not np.array_equal(restart_wet & wet, wet):
                raise SystemExit(
                    "the restart masks cells the mesh calls wet; a bump there "
                    "would be written on top of a fill value")
    shutil.copyfile(restart_in, restart_out)
    with netCDF4.Dataset(mesh_mask) as ds:
        lat = np.asarray(ds.variables["nav_lat"][:]).squeeze()
        lon = np.asarray(ds.variables["nav_lon"][:]).squeeze()
        e1t = np.asarray(ds.variables["e1t"][0]).squeeze()
        e2t = np.asarray(ds.variables["e2t"][0]).squeeze()
    # Distance from the centre using the CENTRE CELL's own e1t/e2t, i.e. a
    # locally-Cartesian approximation on the model's metric rather than a
    # re-derived great-circle formula.  At 300 km against 64 km cells the
    # error is small, but it IS an approximation, and it does not wrap in x.
    cost = np.where(wet, (lat - centre_lat) ** 2 + (lon - centre_lon) ** 2,
                    np.inf)
    jc, ic = np.unravel_index(np.argmin(cost), cost.shape)
    if not wet[jc, ic]:
        raise SystemExit("impulse centre landed on a dry cell")
    jj, ii = np.meshgrid(np.arange(lat.shape[0]), np.arange(lat.shape[1]),
                         indexing="ij")
    dy = (jj - jc) * e2t[jc, ic]
    dx = (ii - ic) * e1t[jc, ic]
    bump = amplitude_m * np.exp(-(dx ** 2 + dy ** 2) / radius_m ** 2)
    bump = np.where(wet, bump, 0.0)
    if not np.isfinite(bump).all():
        raise SystemExit("impulse field is not finite")

    with netCDF4.Dataset(restart_out, "r+") as ds:
        for name in ("sshn", "sshb"):
            a = np.asarray(ds.variables[name][:])
            ds.variables[name][:] = a + bump[None, ...]
    info = {"centre_j": int(jc), "centre_i": int(ic),
            "centre_lat": float(lat[jc, ic]), "centre_lon": float(lon[jc, ic]),
            "amplitude_m": float(amplitude_m), "radius_m": float(radius_m),
            "bump_max_m": float(bump.max()),
            "bump_wet_cells_above_1pct": int((bump > 0.01 * amplitude_m).sum()),
            "restart_in": restart_in, "restart_out": restart_out}
    print(json.dumps(provenance(info), indent=2))
    return info


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
MESH_MASK = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
             "RUN_TRAJ/mesh_mask.nc")

# PRE-REGISTERED comparison targets, in HOURS.  With rn_Dt = 2700 s an hourly
# cadence is not an integer number of steps, so each target is realised at the
# NEAREST step and the probe reports the step's true time next to it.
TARGET_HOURS = (1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 120.0)

# PRE-REGISTERED verdict thresholds (fixed before either model was compared):
#   the per-step floor is measured IN THIS EXPERIMENT as the wet-cell max
#   |eta_lego - eta_nemo| after ONE step.  A scheme difference that does not
#   amplify accumulates at most linearly, so over n steps the difference is
#   bounded by n * floor.  Exceeding that is amplification, not transcription.
GROWTH_BAR_LINEAR = 1.0     # multiplier on (n_steps * one_step_floor)
# NOTE: ``spectra`` returns an AMPLITUDE spectrum, so this is an amplitude
# ratio, not a power ratio.  It is EVALUATED per probe (`exceeds_spectral_bar`
# below), not merely stamped -- a pre-registered threshold that no code
# compares against gets applied by eye after the numbers are seen, which makes
# it post-hoc.
SPECTRAL_AMPLITUDE_BAR = 2.0
# A band is compared only if one of the two models has at least this fraction
# of its peak amplitude there.  See the live-band comment in `compare`.
SPECTRAL_LIVE_FRACTION = 0.1


def wet_mask(mesh_mask: str = MESH_MASK) -> np.ndarray:
    """Surface wet mask (ny, nx) from the model's OWN mesh, never from zeros.

    Inferring land from "where the field is exactly 0" is the structural-zero
    trap: a genuinely zero ocean value (and the equator, where several fields
    vanish by symmetry) would be silently deleted from every statistic.
    """
    with netCDF4.Dataset(mesh_mask) as ds:
        tm = np.asarray(ds.variables["tmask"][0]).squeeze()   # (nk, ny, nx)
    return tm[0] > 0.5


def load_side(path: str, name: str) -> dict:
    d = np.load(path, allow_pickle=False)
    raw_dtype = str(np.asarray(d["eta"]).dtype)
    eta = np.asarray(d["eta"], dtype=np.float64)
    if eta.ndim != 3:
        raise SystemExit(f"{name}: eta must be (nt, ny, nx), got {eta.shape}")
    if not np.isfinite(eta).all():
        raise SystemExit(f"{name}: eta contains NaN/Inf -- fatal, not masked")
    t = np.asarray(d["t_seconds"], dtype=np.float64)
    if t.shape[0] != eta.shape[0]:
        raise SystemExit(f"{name}: t_seconds {t.shape} vs eta {eta.shape}")
    if raw_dtype != "float64":
        raise SystemExit(
            f"{name}: eta was stored as {raw_dtype}. The per-step agreement "
            "this probe measures sits below float32's ~1e-7 relative quantum, "
            "so a float32 artifact cannot support the comparison. Re-run the "
            "producer at full precision.")
    out = {"eta": eta, "t": t, "name": name, "dtype_on_disk": raw_dtype,
           "keys": sorted(k for k in d.files)}
    if "eta_before" in d.files:
        out["eta_before"] = np.asarray(d["eta_before"], dtype=np.float64)
    return out


def masked_stats(diff: np.ndarray, wet: np.ndarray,
                 area: np.ndarray | None = None) -> dict:
    """max/rms of a difference field over WET cells only, plus its locus.

    The rms is AREA-WEIGHTED.  DINO spans 70S-70N, so a cell at the channel
    covers about a third of the area of one at the equator and an unweighted
    mean silently over-counts the poles.  ``area`` may be omitted for
    synthetic tests, where uniform weights are the honest choice.
    """
    d = diff[wet]
    w = np.ones_like(d) if area is None else area[wet]
    amax = float(np.max(np.abs(d)))
    rms = float(np.sqrt(np.sum(w * d ** 2) / np.sum(w)))
    flat = np.argmax(np.where(wet, np.abs(diff), -np.inf))
    j, i = np.unravel_index(flat, diff.shape)
    return {"max_abs_m": amax, "rms_m": rms, "argmax_j": int(j),
            "argmax_i": int(i)}


def locus_partition(wet: np.ndarray, lat: np.ndarray,
                    periodic_i: bool = True,
                    equator_half_width_deg: float = 2.0,
                    wall_cells: int = 1) -> dict:
    """Three disjoint wet regions the difference can live in.

    WALL   -- wet cells within ``wall_cells`` of land, plus the northern and
              southern outermost rows (the free-slip walls and the
              topographic staircase).
    EQUATOR-- |lat| <= ``equator_half_width_deg``, away from the walls.
    INTERIOR - everything else.

    ``periodic_i`` (default True, which is what DINO is: the bridge builds the
    geometry with ``periodic_i=True``, the namelist sets ``ln_Iperio``, and
    the basin's east/west walls are LAND columns rather than domain edges)
    makes the zonal neighbour test WRAP.  Forcing the first and last columns
    to "wall" would mislabel the re-entrant ACC channel's periodic seam -- 70
    wet cells here -- as wall, and the wall share is a number this probe
    reports.

    ``equator_half_width_deg`` serves TWO different arguments and they give
    different answers, so the caller must choose knowingly: a few rows is the
    right width for "f -> 0, so the Coriolis term has a structural zero",
    while the equatorial WAVEGUIDE in this basin is ~26 degrees for the
    barotropic mode and ~3 degrees for the first baroclinic one.  ``compare``
    reports the split at several widths for exactly this reason.

    ``wall_cells`` widens the wall band; 1 is the bare adjacency, 2 covers the
    reach of the isoneutral operator and the EEN vorticity triads.
    """
    dry = ~wet
    nb = np.zeros_like(wet)
    nb[1:, :] |= dry[:-1, :]
    nb[:-1, :] |= dry[1:, :]
    if periodic_i:
        nb |= np.roll(dry, 1, axis=1)
        nb |= np.roll(dry, -1, axis=1)
    else:
        nb[:, 1:] |= dry[:, :-1]
        nb[:, :-1] |= dry[:, 1:]
        nb[:, 0] = nb[:, -1] = True
    nb[0, :] = nb[-1, :] = True
    for _ in range(max(0, int(wall_cells) - 1)):
        grow = np.zeros_like(nb)
        grow[1:, :] |= nb[:-1, :]
        grow[:-1, :] |= nb[1:, :]
        if periodic_i:
            grow |= np.roll(nb, 1, axis=1)
            grow |= np.roll(nb, -1, axis=1)
        else:
            grow[:, 1:] |= nb[:, :-1]
            grow[:, :-1] |= nb[:, 1:]
        nb = nb | grow
    wall = wet & nb
    eq = wet & (~wall) & (np.abs(lat) <= equator_half_width_deg)
    interior = wet & (~wall) & (~eq)
    return {"wall": wall, "equator": eq, "interior": interior}


def locus_shares(diff: np.ndarray, regions: dict,
                 area: np.ndarray | None = None) -> dict:
    """Area-weighted share of the squared difference PER REGION, and its
    ENRICHMENT relative to how much of the domain that region occupies.

    The raw share alone is close to a tautology: the interior is 93% of the
    wet cells here, so it wins every panel regardless of the physics, and a
    genuine three-fold concentration on the walls would still read as
    "the difference lives in the interior".  The enrichment (share divided by
    area fraction, 1.0 = no localisation) is the number that can actually
    name a locus, so both are returned.
    """
    def wsum(m):
        w = np.ones(int(m.sum())) if area is None else area[m]
        return float(np.sum(w * diff[m] ** 2)), float(np.sum(w))
    parts = {k: wsum(m) for k, m in regions.items()}
    tot = sum(v[0] for v in parts.values())
    tot_a = sum(v[1] for v in parts.values())
    if tot <= 0.0 or tot_a <= 0.0:
        return {k: {"share": 0.0, "area_fraction": 0.0, "enrichment": 0.0}
                for k in regions}
    out = {}
    for k, (num, ar) in parts.items():
        share, frac = num / tot, ar / tot_a
        out[k] = {"share": share, "area_fraction": frac,
                  "enrichment": (share / frac) if frac > 0 else 0.0}
    return out


def plant_dry_violation(eta: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Poison every DRY cell with a huge value.

    Every masked statistic must be bit-identical on the poisoned copy.  A
    statistic that moves is not masking; it is averaging land.
    """
    if wet.all():
        raise SystemExit("no dry cells: the mask control cannot fire, so the "
                         "masking claim would be vacuous")
    out = eta.copy()
    out[..., ~wet] = 1.0e6
    return out


def spectra(series: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum of a de-meaned, Hann-windowed series.

    Two corrections that are easy to get wrong and both bite here:
    the mean removed is the WINDOW-WEIGHTED mean, otherwise the window
    re-injects a DC component that then leaks across the low bins; and the
    coherent-gain factor 2/sum(w) is right only for INTERIOR bins -- the zero
    and (for even n) Nyquist bins are not paired, so they take 1/sum(w).
    Leaving the factor at 2 there inflates the Nyquist amplitude two-fold,
    and the Nyquist bin is exactly where a computational mode would sit.
    """
    w = np.hanning(series.size)
    x = (series - np.average(series, weights=w)) * w
    amp = np.abs(np.fft.rfft(x)) * 2.0 / w.sum()
    amp[0] *= 0.5
    if series.size % 2 == 0:
        amp[-1] *= 0.5
    freq = np.fft.rfftfreq(series.size, d=dt)
    return freq, amp


# Period bands used for the domain-wide energy comparison.  The four
# single-cell spectra answer "do the peaks line up"; they cannot answer
# "is there a band where legoESM carries excess energy, and WHERE", which is
# the under-damped-computational-mode / spurious-wall-reflection question.
PERIOD_BANDS_H = (("lt_6h", 0.0, 6.0), ("6_24h", 6.0, 24.0),
                  ("gt_24h", 24.0, np.inf))


def variance_bands(eta: np.ndarray, wet: np.ndarray, dt: float,
                   area: np.ndarray | None = None) -> dict:
    """Share of TEMPORAL variance in each period band, summed over wet cells.

    Window-weighted-demeaned and Hann-windowed per cell, the same way
    ``spectra`` does it, so the single-cell spectra and this domain-wide
    number really are the same estimator.

    Interior bins of a one-sided spectrum each stand for a CONJUGATE PAIR and
    carry twice the variance of the unpaired zero and Nyquist bins, so they
    are weighted 2 here.  Weighting every bin equally inflates the Nyquist end
    -- which is exactly where a leapfrog computational mode sits, i.e. the
    band this probe is hunting.  The zero bin is then excluded outright: the
    series are demeaned, so it holds only window leakage, and dividing by a
    total that includes it would make every share depend on the leakage.
    """
    x = eta[:, wet]
    w = np.hanning(x.shape[0])[:, None]
    nt = x.shape[0]
    x = (x - np.sum(x * w, axis=0, keepdims=True) / w.sum()) * w
    aw = (np.ones(x.shape[1]) if area is None else area[wet])
    power = (np.abs(np.fft.rfft(x, axis=0)) ** 2) * aw[None, :]
    pair = np.full(power.shape[0], 2.0)
    pair[0] = 1.0
    if nt % 2 == 0:
        pair[-1] = 1.0
    power = power * pair[:, None]
    freq = np.fft.rfftfreq(x.shape[0], d=dt)
    period_h = np.full(freq.shape, np.inf)
    period_h[1:] = 1.0 / (freq[1:] * 3600.0)
    total = float(power[1:].sum())
    if total <= 0.0:
        raise SystemExit("variance_bands: the field has no temporal variance "
                         "at all -- a share of zero total is undefined, not 0")
    out = {}
    for name, lo, hi in PERIOD_BANDS_H:
        m = (period_h >= lo) & (period_h < hi)
        m[0] = False
        out[name] = float(power[m].sum()) / total
    return out


def step_noise_ratio(lego: np.ndarray, nemo: np.ndarray, wet: np.ndarray,
                     regions: dict, area: np.ndarray | None = None) -> dict:
    """Per-cell ratio of mean-squared STEP-TO-STEP change, legoESM / NEMO.

    The highest frequency either model can represent is the 2-step mode, and
    a step-to-step variance ratio is the cheapest map of "where does legoESM
    carry high-frequency energy the oracle does not".  A ratio near 1
    everywhere means the two free surfaces jitter alike; a large ratio
    concentrated in one region NAMES that region.

    The denominator IS floored, at the 10th percentile of the oracle's own
    step-noise over wet cells: a near-static cell would otherwise produce an
    enormous ratio that dominates the headline ``max`` without meaning
    anything.  Cells where NEMO is exactly static are excluded outright.
    Read ``median`` and ``p90`` first; ``max`` is a single cell.
    """
    dl = np.diff(lego, axis=0)
    dn = np.diff(nemo, axis=0)
    num = (dl ** 2).mean(axis=0)
    den = (dn ** 2).mean(axis=0)
    live = wet & (den > 0.0)
    if not live.any():
        raise SystemExit("step_noise_ratio: no wet cell where NEMO moves")
    floor = float(np.percentile(den[live], 10.0))
    den_f = np.maximum(den, floor) if floor > 0.0 else den
    ratio = np.where(live, num / np.where(live, den_f, 1.0), np.nan)
    vals = ratio[live]
    j, i = np.unravel_index(
        np.argmax(np.where(live, ratio, -np.inf)), ratio.shape)
    # which locus does the excess live in?  Share of the EXCESS (num - den,
    # clipped at zero) per region, so a region with many mildly-noisy cells
    # cannot be hidden by one extreme cell elsewhere.
    # AREA-WEIGHTED, like every other global statistic here: the wall band
    # includes the outermost rows and is latitude-biased by construction, and
    # this grid spans 70S-70N.
    aw = np.ones_like(num) if area is None else area
    excess = np.where(live, np.maximum(num - den, 0.0), 0.0) * aw
    tot = float(excess.sum())
    if tot > 0:
        shares = {}
        n_live = float((aw * live).sum())
        for k, m in regions.items():
            frac = float((aw * (m & live)).sum()) / n_live
            sh = float(excess[m].sum()) / tot
            shares[k] = {"share": sh, "area_fraction": frac,
                         "enrichment": (sh / frac) if frac > 0 else 0.0}
    else:
        shares = {k: {"share": 0.0, "area_fraction": 0.0, "enrichment": 0.0}
                  for k in regions}
    return {
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "max": float(vals.max()),
        "argmax_j": int(j), "argmax_i": int(i),
        "n_live_cells": int(live.sum()),
        "denominator_floor_m2": floor,
        "excess_share_by_locus": shares,
    }


def highpass(x: np.ndarray, dt: float, cut_hours: float = 24.0) -> np.ndarray:
    """Keep only periods SHORTER than ``cut_hours`` (de-meaned, sharp cut)."""
    freq = np.fft.rfftfreq(x.shape[0], d=dt)
    period_h = np.full(freq.shape, np.inf)
    period_h[1:] = 1.0 / (freq[1:] * 3600.0)
    spec = np.fft.rfft(x - x.mean(axis=0, keepdims=True), axis=0)
    spec[period_h >= cut_hours] = 0.0
    return np.fft.irfft(spec, n=x.shape[0], axis=0)


def propagation_lag(eta: np.ndarray, wet: np.ndarray, jrow: int, dx_m: float,
                    dt: float, max_lag: int = 20) -> dict:
    """Median time lag between ADJACENT cells along one row -- a phase speed.

    This is the "do the two models propagate their surface signal at the same
    speed" measurement.  It answers a second question for free: if the median
    lag is 0 for BOTH models, the sampling cannot resolve propagation at all,
    and no wave-speed claim may be made from this data set in either
    direction.

    SIGN CONVENTION: a POSITIVE lag means the EASTERN neighbour receives the
    signal later, i.e. eastward propagation, and the reported phase speed is
    then positive eastward.  The raw cross-correlation argmax has the opposite
    sign (it is the shift applied to the WESTERN series), so it is negated
    below -- stated here because a silently flipped propagation direction is
    exactly the kind of error that survives every finite check.
    """
    a = highpass(eta[:, jrow, :], dt)
    cols = [i for i in range(a.shape[1]) if wet[jrow, i]]
    lags = []
    for i0, i1 in zip(cols[:-1], cols[1:]):
        x, y = a[:, i0], a[:, i1]
        if x.std() == 0.0 or y.std() == 0.0:
            continue
        best, best_c = 0, -np.inf
        for lag in range(-max_lag, max_lag + 1):
            xs = x[max(0, lag):x.size + min(0, lag)]
            ys = y[max(0, -lag):y.size + min(0, -lag)]
            if xs.size < 8:
                continue
            c = float(np.corrcoef(xs, ys)[0, 1])
            if np.isfinite(c) and c > best_c:
                best, best_c = lag, c
        lags.append(-best)   # see the SIGN CONVENTION note above
    if not lags:
        raise SystemExit("propagation_lag: no usable adjacent wet pair")
    lags = np.asarray(lags, dtype=np.float64)
    med = float(np.median(lags))
    return {
        "median_lag_steps": med,
        "median_lag_hours": med * dt / 3600.0,
        "phase_speed_m_s": (dx_m / (med * dt)) if med != 0.0 else None,
        "frac_pairs_within_one_step": float((np.abs(lags) <= 1).mean()),
        "n_pairs": int(lags.size),
        "row": int(jrow),
        "dx_m": float(dx_m),
    }


def distance_field_km(wet: np.ndarray, e1t: np.ndarray, e2t: np.ndarray,
                      centre_j: int, centre_i: int,
                      periodic_i: bool = True) -> np.ndarray:
    """Distance from a centre cell, integrating the model's ACTUAL metric.

    Using the centre cell's e1t/e2t for the whole grid is wrong by up to 1.46x
    at the far corners of this basin, and not wrapping in x is wrong by about
    two cells for the far columns of a zonally periodic domain.  Both are
    invisible in a model-to-model RATIO and both make the distances themselves
    -- which get read as kilometres -- wrong.
    """
    ny, nx = wet.shape
    ycum = np.concatenate([[0.0], np.cumsum(e2t[:-1, centre_i])])
    dy = ycum - ycum[centre_j]
    xcum = np.concatenate([[0.0], np.cumsum(e1t[centre_j, :-1])])
    dx = xcum - xcum[centre_i]
    if periodic_i:
        lx = float(e1t[centre_j, :].sum())
        dx = np.where(np.abs(dx) > 0.5 * lx, np.abs(dx) - lx, dx)
    return np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2) / 1.0e3


def radial_spread(field: np.ndarray, wet: np.ndarray, area: np.ndarray,
                  r_km: np.ndarray) -> list[float]:
    """Energy-weighted mean distance of a field from a centre, per sample [km].

    This is how wave propagation IS resolvable here.  Cell-to-cell phase lag is
    not -- a barotropic wave crosses one cell in 0.11 of a time step -- but the
    radius the response has spread to at each sample is, because the wave
    covers about nine cells per step.  Comparing the two models' spreading
    rates is therefore the honest form of "do they propagate at the same
    speed".

    SATURATION is the trap here.  Once the response has filled the basin this
    measure stops moving and any model-to-model ratio is forced toward 1, so
    the caller must compare it against the saturation value (the same measure
    on a uniform field) and read only the samples well below it.  ``r_km``
    comes from ``distance_field_km``.
    """
    out = []
    for n in range(field.shape[0]):
        w = area * field[n] ** 2 * wet
        tot = float(w.sum())
        if tot <= 0.0:
            raise SystemExit("radial_spread: a sample carries no energy at "
                             "all, so its mean radius is undefined")
        out.append(float((w * r_km).sum() / tot))
    return out


def two_dt_mode(field: np.ndarray, wet: np.ndarray,
                area: np.ndarray | None = None,
                regions: dict | None = None) -> dict:
    """Amplitude of the 2-step (Nyquist) component, per sample.

    Leapfrog carries a computational mode that alternates sign every step, and
    an impulsive displacement excites it.  "Is legoESM's version of that mode
    under-damped relative to the oracle's" is the concrete form of the
    excess-energy question, and a single-cell spectrum cannot answer it
    because the mode is basin-wide and decays within a few steps -- far too
    short to resolve as a spectral peak.

    The alternating part of a series at sample n is
    ``x[n] - (x[n-1] + x[n+1]) / 2``.  A smooth series annihilates it; for
    ``x[n] = A*(-1)**n`` it returns -2A, so the operator is halved here and
    the reported number IS the amplitude A, not twice it.  Getting that factor
    wrong would not change any ratio between the two models but would make
    every absolute value quoted from this function wrong by two.  Interior
    samples only.
    """
    if field.shape[0] < 3:
        raise SystemExit("two_dt_mode needs at least 3 samples")
    alt = two_dt_component(field)
    w = np.ones(int(wet.sum())) if area is None else area[wet]
    amp = [float(np.sqrt(np.sum(w * alt[n][wet] ** 2) / np.sum(w)))
           for n in range(alt.shape[0])]
    out = {"amplitude_by_sample_m": amp,
           "first_8_mean_m": float(np.mean(amp[:8])),
           "last_half_mean_m": float(np.mean(amp[len(amp) // 2:]))}
    if regions is not None:
        # A basin rms cannot tell a genuine basin-wide computational mode from
        # a transient stuck on two boundary rows, and those are very different
        # findings.  The locus of the FIRST sample is reported so the reader
        # can see which one it is.
        out["locus_first_sample"] = locus_shares(alt[0], regions, area)
    return out


def two_dt_component(field: np.ndarray) -> np.ndarray:
    """The alternating (2-step) part of a series, interior samples only.

    ``0.5*(x[n] - (x[n-1]+x[n+1])/2)``: for ``x[n] = A*(-1)**n`` this is -A,
    so the result is an amplitude and not twice one.
    """
    if field.shape[0] < 3:
        raise SystemExit("two_dt_component needs at least 3 samples")
    return 0.5 * (field[1:-1] - 0.5 * (field[:-2] + field[2:]))


def two_dt_leakage_floor(field: np.ndarray, wet: np.ndarray,
                         area: np.ndarray | None = None) -> float:
    """How much of the reported 2-step amplitude is just a SLOW field leaking.

    The 2-step operator is a curvature high-pass, not a notch: about 29% of a
    6-hour signal passes it.  Running it on a copy of the field with the
    2-step component REMOVED gives the part of the answer that is leakage
    rather than mode, and that floor belongs next to any ratio quoted from it.

    The smoother is the symmetric [1,2,1]/4 kernel, which annihilates the
    2-step mode EXACTLY (1-2+1 = 0) while passing 85% of a 6-hour signal.  An
    odd boxcar does not: a 5-point mean leaves an alternating series at 1/5 of
    its amplitude, so the "floor" would be a fifth of the mode itself.
    """
    k = np.array([1.0, 2.0, 1.0]) / 4.0
    flat = field.reshape(field.shape[0], -1)
    sm = np.apply_along_axis(
        lambda v: np.convolve(v, k, mode="same"), 0, flat).reshape(field.shape)
    alt = two_dt_component(sm)
    w = np.ones(int(wet.sum())) if area is None else area[wet]
    amp = [float(np.sqrt(np.sum(w * alt[n][wet] ** 2) / np.sum(w)))
           for n in range(min(8, alt.shape[0]))]
    return float(np.mean(amp))


def two_dt_phase_agreement(a: np.ndarray, b: np.ndarray, wet: np.ndarray,
                           area: np.ndarray | None = None,
                           n_samples: int = 8) -> list[float]:
    """Signed spatial correlation of two fields' 2-step components.

    Amplitude alone cannot tell "both models ring the SAME mode" from "each
    rings its own": a ratio near 1 is consistent with both.  A positive
    correlation says the same mode; near zero says they are unrelated.
    """
    aa, bb = two_dt_component(a), two_dt_component(b)
    w = np.ones(int(wet.sum())) if area is None else area[wet]
    out = []
    for n in range(min(n_samples, aa.shape[0])):
        x, y = aa[n][wet], bb[n][wet]
        den = np.sqrt(np.sum(w * x ** 2) * np.sum(w * y ** 2))
        out.append(float(np.sum(w * x * y) / den) if den > 0 else 0.0)
    return out


def alternation_ratio(series: list[float], n_early: int = 8) -> float:
    """Mean over ODD STEPS divided by mean over EVEN STEPS.

    Sample index 0 is step 1, so the odd STEPS are the even INDICES -- hence
    ``series[0::2]`` in the numerator.  A value near 1 means no step-to-step
    alternation; a large value means the quantity rings at the 2-step period.
    """
    a = np.asarray(series[:n_early], dtype=np.float64)
    if a.size < 2:
        raise SystemExit("alternation_ratio needs at least 2 samples")
    ev = a[1::2].mean()
    return float(a[0::2].mean() / ev) if ev > 0 else float("inf")


def time_level_discriminator(lego_eta, sshn, sshb, wet) -> dict:
    """Which NEMO time level does legoESM's eta actually line up with?

    The source says ``sshn`` (see the module docstring: the swap at
    stpmlf.F90:621-624 precedes rst_write at :634, so the written "now" level
    is the pre-swap AFTER level).  This is the EMPIRICAL check of that reading:
    the registered pairing is scored against the three obvious alternatives.

    Resolvability is decided by BOOTSTRAPPING over samples, not by comparing
    the margin to NEMO's inter-level spread.  That spread was the wrong yard
    stick: in the impulse lane both levels are responses, so their difference
    is largest exactly when the N-vs-N-1 question is EASIEST, and the test
    could never fire.  Here the samples are resampled with replacement and the
    fraction of draws in which the registered pairing still wins is reported --
    a number that means the same thing in both lanes.

    All four candidates are scored on the SAME nt-1 window so the two shifted
    ones are not compared on a different amount of data.
    """
    n = lego_eta.shape[0]

    def rms(a):
        return float(np.sqrt(np.mean(a[:, wet] ** 2)))

    def per_sample(a):
        return np.array([np.mean(a[k][wet] ** 2) for k in range(a.shape[0])])

    pairs = {
        "registered_lego_N_vs_sshn_N": (lego_eta - sshn)[1:],
        "alt_lego_N_vs_sshb_N": (lego_eta - sshb)[1:],
        "alt_lego_N_vs_sshn_Nminus1": lego_eta[1:] - sshn[:-1],
        "alt_lego_Nminus1_vs_sshn_N": lego_eta[:-1] - sshn[1:],
    }
    ms = {k: per_sample(v) for k, v in pairs.items()}
    out = {k: float(np.sqrt(v.mean())) for k, v in ms.items()}
    out["nemo_own_sshn_minus_sshb"] = rms(sshn - sshb)

    rng = np.random.default_rng(0)
    keys = list(ms)
    wins = 0
    draws = 500
    for _ in range(draws):
        idx = rng.integers(0, n - 1, size=n - 1)
        scores = {k: ms[k][idx].mean() for k in keys}
        if min(scores, key=scores.get) == "registered_lego_N_vs_sshn_N":
            wins += 1
    out["registered_wins_bootstrap_fraction"] = wins / draws
    order = sorted(keys, key=lambda k: out[k])
    out["closest"] = order[0]
    out["margin_m"] = float(out[order[1]] - out[order[0]])
    out["registered_is_closest"] = bool(
        order[0] == "registered_lego_N_vs_sshn_N")
    # Resolvable means the sampling itself agrees, not that the margin beats
    # some other quantity.
    out["resolvable"] = bool(out["registered_wins_bootstrap_fraction"] > 0.95
                             or out["registered_wins_bootstrap_fraction"]
                             < 0.05)
    return out


def compare(nemo_npz: str, lego_npz: str, outdir: str,
            mesh_mask: str = MESH_MASK,
            nemo_free_npz: str | None = None,
            lego_free_npz: str | None = None) -> dict:
    """Compare two eta time series cell by cell, band by band.

    With ``nemo_free_npz``/``lego_free_npz`` supplied, the observable becomes
    the IMPULSE RESPONSE -- each model's perturbed run minus its OWN free run.
    That is the lane that can actually say something about waves: it removes
    the balanced background both models are merely carrying along, leaving the
    radiating signal the bump excited.  Each side is differenced against its
    own free run, never against the other model's, so a background difference
    cannot leak into the response.
    """
    os.makedirs(outdir, exist_ok=True)
    nemo = load_side(nemo_npz, "NEMO")
    lego = load_side(lego_npz, "legoESM")
    lane = "free"
    dtypes = {"NEMO": nemo["dtype_on_disk"], "legoESM": lego["dtype_on_disk"]}
    if (nemo_free_npz is None) != (lego_free_npz is None):
        raise SystemExit("the impulse lane needs BOTH free runs or neither -- "
                         "differencing one side only would compare a response "
                         "against a full field")
    if nemo_free_npz is not None:
        nf = load_side(nemo_free_npz, "NEMO free")
        for side in (nemo, nf):
            if "eta_before" not in side:
                raise SystemExit(
                    f"{side['name']} carries no eta_before, so the response "
                    "cannot keep the before-level in step with the after "
                    "one. Re-extract with sshb.")
        lf = load_side(lego_free_npz, "legoESM free")
        for a, b in ((nemo, nf), (lego, lf)):
            if a["eta"].shape != b["eta"].shape or not np.array_equal(
                    a["t"], b["t"]):
                raise SystemExit("a perturbed run and its free run disagree "
                                 "in shape or clock")
        dtypes["NEMO free"] = nf["dtype_on_disk"]
        dtypes["legoESM free"] = lf["dtype_on_disk"]
        nemo = dict(nemo, eta=nemo["eta"] - nf["eta"],
                    eta_before=nemo["eta_before"] - nf["eta_before"])
        lego = dict(lego, eta=lego["eta"] - lf["eta"])
        lane = "impulse_response"
    wet = wet_mask(mesh_mask)

    if nemo["eta"].shape != lego["eta"].shape:
        raise SystemExit(f"frame mismatch: NEMO {nemo['eta'].shape} vs "
                         f"legoESM {lego['eta'].shape}")
    if nemo["eta"].shape[1:] != wet.shape:
        raise SystemExit(f"mask {wet.shape} does not match frame "
                         f"{nemo['eta'].shape[1:]}")
    if not np.array_equal(nemo["t"], lego["t"]):
        raise SystemExit("the two time axes differ -- a wave comparison across "
                         "mismatched clocks is a confound, not a result")

    with netCDF4.Dataset(mesh_mask) as ds:
        lat = np.asarray(ds.variables["nav_lat"][:]).squeeze()
        lon = np.asarray(ds.variables["nav_lon"][:]).squeeze()
        e1t = np.asarray(ds.variables["e1t"][0]).squeeze()   # zonal cell width
        e2t = np.asarray(ds.variables["e2t"][0]).squeeze()
    area = np.where(wet, e1t * e2t, 0.0)
    for path in (nemo_npz, lego_npz, nemo_free_npz, lego_free_npz):
        if path is None:
            continue
        with np.load(path) as nav:
            for name, ref in (("nav_lat", lat), ("nav_lon", lon)):
                if name not in nav.files:
                    continue
                if not np.allclose(np.asarray(nav[name]), ref, atol=1e-4):
                    raise SystemExit(
                        f"{name} in {path} does not match the mesh at "
                        f"{mesh_mask}. The equator band and all four probe "
                        "points would be placed on the wrong cells.")

    t = nemo["t"]
    dt = float(t[1] - t[0])
    nt = t.size
    if abs(t[0] - dt) > 1e-9 * dt:
        raise SystemExit(
            f"the first sample is at t={t[0]} s but the cadence is {dt} s. "
            "Every 'step N' label and the one-step floor below assume one "
            "frame per model step; refusing to mislabel them.")
    if not np.allclose(np.diff(t), dt):
        raise SystemExit("the time axis is not uniformly sampled")
    diff = lego["eta"] - nemo["eta"]

    # ---- CONTROL 1: the dry-cell violation must not move a single number ----
    poisoned = plant_dry_violation(lego["eta"], wet)
    a = masked_stats(diff[-1], wet, None)
    b = masked_stats((poisoned - nemo["eta"])[-1], wet, None)
    if a != b:
        raise SystemExit(f"MASK CONTROL FAILED: poisoning dry cells changed a "
                         f"masked statistic ({a} vs {b})")
    # ...and it must MOVE the unmasked one, or the control itself is vacuous
    if float(np.max(np.abs(diff[-1]))) == float(
            np.max(np.abs((poisoned - nemo["eta"])[-1]))):
        raise SystemExit("MASK CONTROL VACUOUS: the poison did not change even "
                         "the UNmasked statistic")

    regions = locus_partition(wet, lat)
    # Sensitivity of the locus split to the two width choices a reader could
    # reasonably dispute (see locus_partition's docstring), reported so the
    # headline locus is never read as if the widths were free of choice.
    locus_sensitivity = {}
    for eq_deg in (2.0, 5.0, 10.0, 26.0):
        for wall_n in (1, 2):
            r = locus_partition(wet, lat, equator_half_width_deg=eq_deg,
                                wall_cells=wall_n)
            locus_sensitivity[f"eq{eq_deg:g}deg_wall{wall_n}cell"] = {
                k: {"cells": int(v.sum()),
                    "area_fraction": float(v.sum()) / float(wet.sum())}
                for k, v in r.items()}

    # ---- growth curve + locus ------------------------------------------
    signal_change = np.array(
        [np.sqrt(np.sum(area[wet] * (nemo["eta"][n] - nemo["eta"][0])[wet] ** 2)
                 / np.sum(area[wet])) for n in range(nt)])
    growth = []
    for n in range(nt):
        st = masked_stats(diff[n], wet, area)
        st["share"] = locus_shares(diff[n], regions, area)
        # The bar that actually measures FIDELITY rather than amplification:
        # the error relative to how much the oracle's own field moved over
        # the same window.  The linear bar below is proportional to the
        # measured error itself, so it can only see super-linear growth.
        st["rms_over_signal_change"] = (float(st["rms_m"] / signal_change[n])
                                        if signal_change[n] > 0 else None)
        # For the impulse lane this is the number that matters: the
        # disagreement as a fraction of the RESPONSE the bump excited.
        ref_rms = float(np.sqrt(np.sum(area[wet] * nemo["eta"][n][wet] ** 2)
                                / np.sum(area[wet])))
        st["rms_over_reference_rms"] = (float(st["rms_m"] / ref_rms)
                                        if ref_rms > 0 else None)
        st["t_hours"] = float(t[n] / 3600.0)
        st["step"] = n + 1
        growth.append(st)
    floor = growth[0]["max_abs_m"]
    # The linear bar only means something where the error GROWS from a small
    # first-step floor.  In the impulse lane the first sample is the LARGEST
    # value in the series (the response then decays), so n*floor cannot be
    # exceeded by construction and the criterion is a test that cannot fail.
    # It is reported as null with a reason instead of as a spectacular pass.
    growing = floor < max(g["max_abs_m"] for g in growth)
    bar = GROWTH_BAR_LINEAR * nt * floor if growing else None

    # ---- pre-registered snapshot times ---------------------------------
    snaps = []
    for h in TARGET_HOURS:
        if h * 3600.0 > t[-1] + 0.5 * dt:
            continue      # never silently clamp a target onto the last frame
        n = int(np.argmin(np.abs(t / 3600.0 - h)))
        s = dict(growth[n])
        s["target_hours"] = h
        snaps.append(s)
    if not snaps:
        raise SystemExit("no pre-registered target hour lies inside the run")

    # ---- spectra at four named probe points ----------------------------
    probes = named_probes(lat, lon, wet)

    # ---- propagation speed along the channel row, both sides -----------
    jrow = probes["channel"][0]
    dx = float(np.median(e1t[jrow, wet[jrow]]))
    prop = {"NEMO": propagation_lag(nemo["eta"], wet, jrow, dx, dt),
            "legoESM": propagation_lag(lego["eta"], wet, jrow, dx, dt),
            "difference": propagation_lag(diff, wet, jrow, dx, dt)}
    spec = {}
    for pname, (j, i) in probes.items():
        fn, an = spectra(nemo["eta"][:, j, i], dt)
        fl, al = spectra(lego["eta"][:, j, i], dt)
        if not np.allclose(fn, fl):
            raise SystemExit("spectral frequency axes differ")
        # A band counts as LIVE if EITHER model has real power there.  A cut
        # defined from NEMO alone would exclude exactly the failure mode this
        # comparison exists to find -- a band where legoESM rings and the
        # oracle is silent.  Bin 0 is excluded (the series are de-meaned, so
        # it holds only leakage), and the threshold is 10% of the peak, not
        # 0.1%: the looser cut admits noise-floor bins whose ratio is
        # meaningless and then takes a max over ~80 of them, which is an
        # extreme-value generator rather than a measurement.
        ref = max(float(an[1:].max()), float(al[1:].max()))
        live = np.maximum(an, al) > SPECTRAL_LIVE_FRACTION * ref
        live[0] = False
        if not live.any():
            raise SystemExit(f"probe {pname}: no live spectral band")
        ratio = np.where(live, al / np.maximum(an, 1e-30), np.nan)
        two_sided = np.where(live, np.maximum(
            ratio, 1.0 / np.maximum(ratio, 1e-30)), -np.inf)
        worst = int(np.argmax(two_sided))
        spec[pname] = {
            "j": int(j), "i": int(i),
            "lat": float(lat[j, i]), "lon": float(lon[j, i]),
            "peak_freq_nemo_cph": float(fn[1:][np.argmax(an[1:])] * 3600.0),
            "peak_freq_lego_cph": float(fl[1:][np.argmax(al[1:])] * 3600.0),
            "peak_amp_nemo_m": float(an[1:].max()),
            "peak_amp_lego_m": float(al[1:].max()),
            "worst_band_freq_cph": float(fn[worst] * 3600.0),
            "worst_band_period_h": float(1.0 / (fn[worst] * 3600.0))
            if fn[worst] > 0 else float("inf"),
            "worst_band_ratio_lego_over_nemo": float(ratio[worst]),
            "worst_band_two_sided_ratio": float(two_sided[worst]),
            # the pre-registered bar, EVALUATED rather than merely stamped
            "exceeds_spectral_bar": bool(
                two_sided[worst] > SPECTRAL_AMPLITUDE_BAR),
            "n_live_bands": int(live.sum()),
            "_freq_cph": fn * 3600.0, "_amp_nemo": an, "_amp_lego": al,
        }

    if "eta_before" not in nemo:
        raise SystemExit(
            "the NEMO artifact carries no eta_before, so the time-level "
            "pairing cannot be discriminated. Re-extract with sshb.")
    levels = time_level_discriminator(lego["eta"], nemo["eta"],
                                      nemo["eta_before"], wet)
    # A pairing that is not the registered one is a FINDING, not a silent
    # field buried in a nested dict -- it means legoESM's state lines up
    # better with a different NEMO time level, i.e. a sub-step timing offset.
    # It is surfaced at the top of the result either way.
    time_level_warning = None
    if not levels["registered_is_closest"]:
        time_level_warning = (
            f"legoESM lines up better with {levels['closest']} than with the "
            f"source-cited sshn pairing "
            f"({levels[levels['closest']]:.3e} vs "
            f"{levels['registered_lego_N_vs_sshn_N']:.3e} m rms, registered "
            f"wins in {levels['registered_wins_bootstrap_fraction']:.0%} of "
            "bootstrap draws). In the impulse lane this is a SUB-STEP TIMING "
            "OFFSET in the response, not an extractor bug -- the extractor's "
            "mapping is pinned independently by each dump's own kt.")
        print("WARNING: " + time_level_warning)

    bands = {"NEMO": variance_bands(nemo["eta"], wet, dt, area),
             "legoESM": variance_bands(lego["eta"], wet, dt, area)}
    # A field with no temporal variance has no band SHARES -- 0/0, not zeros.
    # That is a legitimate state here (two identical models) and must not
    # crash, but it must also not be reported as if the shares were measured.
    bands["difference"] = (variance_bands(diff, wet, dt, area)
                           if np.any(diff[:, wet]) else None)
    noise = step_noise_ratio(lego["eta"], nemo["eta"], wet, regions, area)

    # Spatial spreading of the response.  Only meaningful in the impulse lane,
    # where there is a response with a centre; in the free lane it would be
    # the mean radius of the whole balanced field, which answers nothing.
    spread = None
    if lane == "impulse_response":
        # The PRE-REGISTERED bump centre, not the argmax of the response: by
        # the first sample the peak sits on the radiating ring and the centre
        # has already reversed sign, so the argmax lands three rows away.
        cj, ci = named_cell(lat, lon, wet, IMPULSE_CENTRE_LAT,
                            IMPULSE_CENTRE_LON)
        r_km = distance_field_km(wet, e1t, e2t, cj, ci)
        sp_n = radial_spread(nemo["eta"], wet, area, r_km)
        sp_l = radial_spread(lego["eta"], wet, area, r_km)
        ratio = [b / a_ for a_, b in zip(sp_n, sp_l)]
        # A UNIFORM field's mean radius is where this measure saturates. Past
        # about half of it the response has filled the basin, both models say
        # the same thing by construction, and the ratio is forced to 1 -- a
        # median over the whole run is therefore a saturated number, not a
        # measurement.
        sat = float((area * r_km * wet).sum() / (area * wet).sum())
        usable = [n for n, v in enumerate(sp_n) if v < 0.5 * sat]
        spread = {
            "centre_j": int(cj), "centre_i": int(ci),
            "saturation_radius_km": sat,
            "nemo_km": sp_n, "lego_km": sp_l,
            "ratio_lego_over_nemo": ratio,
            "n_samples_below_half_saturation": len(usable),
            "ratio_unsaturated_samples": [ratio[n] for n in usable],
            "ratio_unsaturated_median": (float(np.median([ratio[n]
                                                          for n in usable]))
                                         if usable else None),
            "ratio_first_4_samples": ratio[:4],
            # kept for continuity, but SATURATED -- do not quote it alone
            "ratio_all_samples_median_SATURATED": float(np.median(ratio)),
        }

    two_dt = {
        "difference": two_dt_mode(diff, wet, area, regions),
        "NEMO": two_dt_mode(nemo["eta"], wet, area, regions),
        "legoESM": two_dt_mode(lego["eta"], wet, area, regions),
        "leakage_floor_m": {
            "NEMO": two_dt_leakage_floor(nemo["eta"], wet, area),
            "legoESM": two_dt_leakage_floor(lego["eta"], wet, area),
        },
        "phase_agreement_first8": two_dt_phase_agreement(
            lego["eta"], nemo["eta"], wet, area),
    }
    # The alternation of the DIFFERENCE itself: if the two models' leapfrog
    # modes were identical the difference would not alternate at all.
    diff_rms_by_step = [g["rms_m"] for g in growth]
    two_dt["difference_alternation_ratio_first8"] = alternation_ratio(
        diff_rms_by_step, 8)
    two_dt["difference_alternation_ratio_rest"] = alternation_ratio(
        diff_rms_by_step[8:], min(16, max(2, len(diff_rms_by_step) - 8)))

    result = {
        "time_level_warning": time_level_warning,
        "radial_spread": spread,
        "two_dt_mode": two_dt,
        "lane": lane,
        "locus_sensitivity": locus_sensitivity,
        "propagation": prop,
        "time_levels_rms_m": levels,
        "variance_bands": bands,
        "step_noise_ratio": noise,
        "one_step_floor_max_abs_m": floor,
        "n_steps": nt,
        "dt_seconds": dt,
        "linear_bar_m": bar,
        "linear_bar_applicable": bool(growing),
        "linear_bar_note": (None if growing else
                            "the first sample is the largest in the series, "
                            "so n*floor cannot be exceeded; this criterion is "
                            "vacuous in this lane"),
        "final_max_abs_m": growth[-1]["max_abs_m"],
        "growth_over_linear_bar": ((growth[-1]["max_abs_m"] / bar)
                                   if (growing and bar > 0) else None),
        "snapshots": snaps,
        "growth": [{k: v for k, v in g.items()} for g in growth],
        "spectra": {k: {kk: vv for kk, vv in v.items()
                        if not kk.startswith("_")}
                    for k, v in spec.items()},
        "provenance": provenance({
            "nemo_npz": nemo_npz, "lego_npz": lego_npz,
            "mesh_mask": mesh_mask, "lane": lane,
            "nemo_free_npz": nemo_free_npz, "lego_free_npz": lego_free_npz,
            "dtype_on_disk": dtypes,
            "wet_cells": int(wet.sum()),
            "dry_cells": int((~wet).sum()),
            "mask_control": "PASSED (dry poison inert on masked stats, "
                            "active on unmasked)",
            "growth_bar_linear": GROWTH_BAR_LINEAR,
            "spectral_amplitude_bar": SPECTRAL_AMPLITUDE_BAR,
            "spectral_live_fraction": SPECTRAL_LIVE_FRACTION,
        }),
    }
    with open(os.path.join(outdir, "eta_wave_twin.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    make_figures(nemo, lego, diff, wet, lat, lon, t, snaps, growth, spec,
                 regions, outdir)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("growth",)}, indent=2, default=float))
    print(f"[out] {outdir}")
    return result


def named_cell(lat, lon, wet, target_lat, target_lon) -> tuple[int, int]:
    """Nearest WET cell to a target position.

    A degree of longitude is cos(lat) of a degree of latitude, so an unscaled
    degree-space distance biases every mid-latitude choice zonally.
    """
    dlon = (lon - target_lon) * np.cos(np.deg2rad(target_lat))
    cost = np.where(wet, (lat - target_lat) ** 2 + dlon ** 2, np.inf)
    j, i = np.unravel_index(np.argmin(cost), cost.shape)
    if not wet[j, i]:
        raise SystemExit("cell placement landed on a dry cell")
    return int(j), int(i)


def named_probes(lat, lon, wet) -> dict:
    """Four wet probe points, named for the physics they are meant to expose."""
    def pick(target_lat, target_lon):
        return named_cell(lat, lon, wet, target_lat, target_lon)
    return {
        "channel": pick(-55.0, 25.0),     # the ACC channel
        "equator": pick(0.0, 25.0),       # f -> 0, structural-zero territory
        "west_wall": pick(-30.0, 3.0),    # western boundary, reflection site
        "mid_basin": pick(30.0, 25.0),    # quiet interior reference
    }


def _safe_scale(a: np.ndarray) -> float:
    """Positive colour-scale limit.  ``np.nanmax(x) or 1e-30`` LOOKS like a
    guard but is not: NaN is truthy, so an all-NaN slice returns NaN and hands
    matplotlib vmin=vmax=nan."""
    m = np.abs(a)
    v = float(np.nanmax(m)) if np.isfinite(m).any() else 0.0
    return v if v > 0.0 else 1e-30


def make_figures(nemo, lego, diff, wet, lat, lon, t, snaps, growth, spec,
                 regions, outdir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nan = np.where(wet, 0.0, np.nan)

    # (a) difference maps at the pre-registered hours
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    for ax, s in zip(axes.ravel(), snaps):
        n = s["step"] - 1
        m = np.abs(diff[n] + nan)
        v = _safe_scale(m)
        im = ax.pcolormesh(lon, lat, diff[n] + nan, cmap="RdBu_r",
                           vmin=-v, vmax=v, shading="auto")
        ax.set_title(f"t={s['t_hours']:.2f} h (target {s['target_hours']:g} h)\n"
                     f"max|d|={s['max_abs_m']:.2e} m", fontsize=9)
        plt.colorbar(im, ax=ax)
    fig.suptitle("eta difference  legoESM - NEMO  [m]  (dry cells blanked)")
    fig.savefig(os.path.join(outdir, "fig_diff_maps.png"), dpi=110)
    plt.close(fig)

    # (b) Hovmoller: time-longitude at the channel latitude, time-latitude at
    #     a mid-basin longitude, both sides and the difference
    jc = spec["channel"]["j"]
    ic = spec["mid_basin"]["i"]
    th = t / 3600.0
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for row, (lab, sl, xax, xname) in enumerate([
            ("time-longitude @ channel lat "
             f"{lat[jc, 0]:.1f}", (slice(None), jc, slice(None)),
             lon[jc, :], "longitude"),
            ("time-latitude @ lon "
             f"{lon[0, ic]:.1f}", (slice(None), slice(None), ic),
             lat[:, ic], "latitude")]):
        wl = (wet[jc, :] if row == 0 else wet[:, ic])
        blank = np.where(wl, 0.0, np.nan)
        for col, (nm, fld) in enumerate([("NEMO", nemo["eta"]),
                                         ("legoESM", lego["eta"]),
                                         ("lego - NEMO", diff)]):
            a = fld[sl] + blank
            v = _safe_scale(a)
            ax = axes[row, col]
            im = ax.pcolormesh(xax, th, a, cmap="RdBu_r", vmin=-v, vmax=v,
                               shading="auto")
            ax.set_title(f"{nm}\n{lab}", fontsize=9)
            ax.set_xlabel(xname)
            ax.set_ylabel("hours")
            plt.colorbar(im, ax=ax)
    fig.suptitle("eta Hovmoller [m] -- same colour scale within a panel only")
    fig.savefig(os.path.join(outdir, "fig_hovmoller.png"), dpi=110)
    plt.close(fig)

    # (c) growth curve + locus shares
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].semilogy(th, [g["max_abs_m"] for g in growth], label="max|d eta|")
    axes[0].semilogy(th, [g["rms_m"] for g in growth], label="rms|d eta|")
    floor = growth[0]["max_abs_m"]
    axes[0].semilogy(th, floor * np.arange(1, len(growth) + 1), "k--",
                     label="linear accumulation of the 1-step floor")
    axes[0].set_xlabel("hours")
    axes[0].set_ylabel("m")
    axes[0].legend()
    axes[0].set_title("difference growth (wet cells only)")
    for key in ("wall", "equator", "interior"):
        axes[1].plot(th, [g["share"][key]["enrichment"] for g in growth],
                     label=key)
    axes[1].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel("hours")
    axes[1].set_ylabel("enrichment (share / area fraction)")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_title("locus")
    fig.savefig(os.path.join(outdir, "fig_growth_locus.png"), dpi=110)
    plt.close(fig)

    # (d) spectra at the four probe points
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    for ax, (nm, sp) in zip(axes, spec.items()):
        f = sp["_freq_cph"][1:]
        ax.loglog(f, sp["_amp_nemo"][1:], label="NEMO")
        ax.loglog(f, sp["_amp_lego"][1:], label="legoESM")
        ax.set_title(f"{nm}  lat {sp['lat']:.1f} lon {sp['lon']:.1f}",
                     fontsize=9)
        ax.set_xlabel("cycles / hour")
        ax.set_ylabel("amplitude [m]")
        ax.legend(fontsize=8)
    fig.suptitle("eta frequency spectra at the four pre-registered probes")
    fig.savefig(os.path.join(outdir, "fig_spectra.png"), dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract-nemo",
                       help="stitch per-step NEMO ssh into one .npz")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--kt0", type=int, default=5760)
    e.add_argument("--nsteps", type=int, default=160)
    e.add_argument("--out", required=True)

    m = sub.add_parser("make-impulse-restart",
                       help="copy a NEMO restart and add the pre-registered "
                            "Gaussian sea-surface bump")
    m.add_argument("--restart-in", required=True)
    m.add_argument("--restart-out", required=True)
    m.add_argument("--amplitude-m", type=float, default=IMPULSE_AMPLITUDE_M)
    m.add_argument("--radius-m", type=float, default=IMPULSE_RADIUS_M)
    m.add_argument("--mesh-mask", default=MESH_MASK)

    c = sub.add_parser("compare", help="the pre-registered wave comparison")
    c.add_argument("--nemo", required=True)
    c.add_argument("--lego", required=True)
    c.add_argument("--outdir", required=True)
    c.add_argument("--mesh-mask", default=MESH_MASK)
    c.add_argument("--nemo-free", default=None,
                   help="NEMO's UNPERTURBED run; supplying it (with "
                        "--lego-free) switches the observable to the impulse "
                        "RESPONSE, each model minus its own free run")
    c.add_argument("--lego-free", default=None)

    a = ap.parse_args(argv)
    if a.cmd == "extract-nemo":
        extract_nemo(a.run_dir, a.kt0, a.nsteps, a.out)
    elif a.cmd == "make-impulse-restart":
        make_impulse_restart(a.restart_in, a.restart_out, a.amplitude_m,
                             a.radius_m, mesh_mask=a.mesh_mask)
    elif a.cmd == "compare":
        compare(a.nemo, a.lego, a.outdir, a.mesh_mask,
                a.nemo_free, a.lego_free)


if __name__ == "__main__":
    main()
