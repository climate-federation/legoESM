"""#1226 canonical state-initialized 90-day twin runner.

A "twin" run initializes legoESM directly from a *developed* NEMO restart
(bridged onto legoESM's staggered C-grid via
:func:`bridge_nemo_to_legoesm_topo`) rather than from the analytic paper rest
state, then integrates forward so the two models can be compared step-for-step
/ day-for-day starting from an identical, dynamically-active IC. This isolates
tendency/scheme mismatches from spin-up-trajectory divergence.

DAY-0 GATE (mandatory, not optional): a twin is only a twin if the legoESM
state at step 0 is bit-identical (to 1e-8) to the NEMO restart it was bridged
from, on wet cells, AND the velocity field is non-trivially nonzero. On
2026-07-24 a defect let a twin instrument silently fall back to the analytic
rest-state IC (``dino_lat_lon_state(...)``) instead of the bridged restart
(``br.state``) -- producing a "twin" that was actually a from-rest spin-up,
which invalidated every day-0..90 comparison built on top of it. This module
hardens that check into an assert-and-raise gate (`verify_day0_matches_restart`)
that runs before any integration and is unit-tested directly (see
``tests/ocean/unit/test_dino_1226_instruments.py``): a rest-state start must be
IMPOSSIBLE to smuggle through un-flagged.

START MODE (#1455, default flipped 2026-08-24): the twin continues NEMO's
LEAP-FROG by default -- ``state.{T,S,u,v,eta}_before`` is seeded from the NEMO
restart's own ``tb/sb/ub/vb``/``sshb`` (the Modified-Leap-Frog integrator's
third time level), so the twin's step-0 entry state is EXACTLY NEMO's. See
:func:`resolve_start_mode`; the resolved mode is stamped into the npz as
``twin_start_mode`` and printed by the acceptance gate.

The old default was a forward-EULER start, and it was not a small thing. It
does THREE things, and the difference between them matters (adversarial physics
review, 2026-08-24, corrected an earlier statement here that named only the
first):

  1. a HALF-STEP DELAY at every frequency. This is the running-mean identity,
     and it is EXACT only for a reference whose before level equals its now
     level. It does not grow.
  2. a PERMANENT STATE PERTURBATION of half a leap-frog step of tendency,
     because a developed NEMO restart's before level is NOT its now level. In
     the gate's own units: one leap-frog step of circumpolar transport is
     4.11e-3 Sv here (ACC of the restart's now level 61.946220 Sv against its
     before level 61.950334 Sv), so the Euler start injects about 2.06e-3 Sv at
     t=0. This one DOES grow, and it is the dominant term at long lead.
  3. it USED to skip NEMO's step-1 after-level reconciliation (#1640 finding
     3) and to drop the whole surface tracer tendency on that step. #1729
     closed both: the Euler start is now a parameterisation of the ordinary
     step, not an early return. CONSEQUENCE, so nobody reads an old number as
     comparable -- ``--legacy-euler-start`` still selects the Euler START
     MODE, but its step 1 is no longer the step 1 those pre-2026-08-24
     artifacts were produced with.

RETRACTED: an earlier version of this docstring said the Euler trajectory "IS
the two-point running mean of the true one" full stop. That both OVERSTATES the
exactness (with the card's Asselin gamma=0.1 the residual is 0.8-6.4% of the
signal, and against a real before level the running mean buys essentially
nothing) and UNDERSTATES the defect (it omits term 2 entirely). Anyone
"correcting" an Euler-start artifact by shifting it half a step would be fixing
the smaller half.
``--legacy-euler-start`` (equivalently ``--no-bridge-before``) selects it back
with a loud banner, for reproducing artifacts recorded before the flip.

WHO ACTUALLY RAN THE EULER START -- AUDITED, because the claim that started
this work ("every recorded twin ran the Euler start") is REFUTED. Every twin
build recorded in this repo's run logs passed the before-level bridge
explicitly: 18 of 18, including all four 90-day acceptance-gate arms, the
staircase/divisor arms, the viscosity ablation and the EEN-metric pair. The
acceptance gate has defaulted the flag ON since 2026-08-09, and the
verdict-year pre-registration specifies it. What ran the Euler start were
ad-hoc lego runs launched straight off this CLI without the flag -- which is
exactly the footgun this default removes, and is where the retracted
"half-step model lag" came from. The recorded GATE and VERDICT numbers were
never Euler-start.

WHAT THE EULER START DOES ON THIS CARD, measured rather than inferred: it does
NOT raise. An earlier version of this docstring said the card's
``tke_n2_time_level="nemo_before"`` axis makes ``model.step`` raise
``ValueError`` at step 0 without the bridge; that was true before the #1317
fix, which now seeds a local ``before := now`` on the Euler-start branch
(NEMO's own cold-start convention, ``istate.F90:97-99``). RETRACTED: a 1-day
``--legacy-euler-start`` run of the shipped card completes normally. The
difference between the two starts is therefore not a crash, it is that the
Euler arm's before level is its own now level instead of NEMO's ``tb``.

INTEGRATOR-MEMORY HANDSHAKE CAVEAT, what is LEFT of it: ``--bridge-tke`` (TKE
closure memory) is a SEPARATE, independent flag and is still cold-start by
default, so days 1-4 of a twin still run on a legoESM-native cold start for
THAT memory while NEMO continues from its own warmed-up state. Do not read
days 1-4 of a non-``--bridge-tke`` run as a scheme-fidelity signal.

SEASONAL CLOCK (#1455): the analytic DINO surface forcing follows the DAY OF
YEAR, so a twin bridged from a mid-year NEMO restart must continue NEMO's own
seasonal clock, not restart the year at zero. This harness reads the offset
from the restart's ``adatrj`` BY DEFAULT. Setting ``DINO_TWIN_SEASONAL_KT0=0``
selects the old relative clock (which forces the day-180 twin exactly antiphase
to NEMO) and prints a loud banner; it exists only to reproduce numbers recorded
before 2026-08-20, none of which are comparable to NEMO.

VERTICAL LADDER (#1455): a twin only isolates SCHEME differences if both
models stand on the same grid, so :func:`run_twin` defaults to NEMO's OWN
vertical ladders -- ``LEGOESM_NEMO_E3T=both``, i.e. NEMO's thickness ladder AND
its T-point depths (see :func:`resolve_ladder_mode`). Two things this does NOT
touch: the MODEL-WIDE bridge default, which still resolves to the 1-D reference
ladder for every other caller; and :func:`_build_twin_state`, which a dozen
sibling probes import directly and which therefore keeps the grid those probes
were recorded on. Setting ``LEGOESM_NEMO_E3T`` explicitly still wins,
``--legacy-1d-ladder`` selects the old 1-D ladder with a loud banner, and the
two disagreeing is fatal rather than one silently winning. The resolved mode is
stamped into the npz as ``nemo_ladder_mode``.

Usage
-----
    python kamm_twin_90d.py <recipe> <out.npz> [--days 90] [--save-3d]
                            [--fp64-3d]

SNAPSHOT PRECISION.  NOTE, so the claim is not read wider than it is: no
scorer READS the reduced series yet -- ``verdict360`` and ``floor90_ensemble``
still reduce the 3-D block, so the series removes the storage quantum from the
ARTIFACT, not yet from any number the campaign prints. Wiring a consumer is a
separate change (both reviews).

The 3-D block is stored float32 BY DEFAULT and that has
not changed; ``--fp64-3d`` stores it float64 and roughly doubles the artifact.
Independently of that flag, every snapshot day also carries an fp64 REDUCED
series -- the five acceptance-gate metrics, the six section/band/latitude-group
transports, and the per-latitude-row transport profile -- computed from the
LIVE model state before the storage cast.  Those are the quantities the scorers
reduce the 3-D block to anyway, so storing them directly removes the float32
storage quantum from all of them for a few kB.  Per-field storage dtypes are
stamped as ``storage_dtypes``; ``control_dtype`` remains the precision the arm
was BUILT at, which is a different thing.

NEMO artifact paths default to the machine-local oracle-build tree and can be
overridden via env vars (``DINO_NEMO_RUN_TRAJ``, ``DINO_NEMO_RUN_STEPDUMP``) or
CLI flags, for portability off this box.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.field import Field
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    apply_dino_lat_lon_surface_forcing,
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
    read_nemo_restart_en,
    read_nemo_restart_tke_coefficients,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    NEMO_E3T_MODES,
    bridge_before_state_topo,
    bridge_nemo_to_legoesm_topo,
)

from legoesm import constants

# NEMO oracle-build artifact roots (mesh/restart donors). Override via env var
# or --run-traj/--run-stepdump for a different machine/build layout.
RUN_TRAJ = os.environ.get(
    "DINO_NEMO_RUN_TRAJ",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ",
)
RUN_STEPDUMP = os.environ.get(
    "DINO_NEMO_RUN_STEPDUMP",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP",
)
RESTART_FILE = "DINO_00005760_restart.nc"  # developed day-180 state

# Relative amplitude of the --perturb-seed kick.  1e-14 is the value every
# recorded noise-floor ensemble used and is the DEFAULT so those artifacts stay
# byte-comparable; it is a knob only because a 1e-14 perturbation is still
# GROWING at day 90 (measured: ACC max-pairwise 8.75e-8 / 3.26e-7 / 1.15e-5 Sv
# at days 30/60/90), so it is not a valid null for a comparison whose own
# perturbation has already saturated.  Matching the amplitude to the difference
# under test is what makes a within-arm ensemble a null instead of a floor.
PERTURB_EPS_DEFAULT = 1e-14

DT = float(__import__("os").environ.get("DINO_DT", "2700.0"))
STEPS_PER_DAY = 32  # 32 * 2700s = 86400s = 1 day
SNAP_DAYS = (0, 30, 60, 90)  # full 3-D T/S snapshot days when --save-3d
BRIDGE_OMEGA_MODES = ("nemo", "legacy-rounded")


def resolve_bridge_omega(mode: str) -> tuple[float, str]:
    """Return the bridge-only Omega and its fail-closed reference mode."""
    if mode == "nemo":
        return NEMO_CONSTANTS_CONFIG.Omega, "nemo"
    if mode == "legacy-rounded":
        return constants.Omega, "selected_omega"
    raise ValueError(
        f"bridge_omega must be one of {BRIDGE_OMEGA_MODES}, got {mode!r}")


def _array_content_sha256(name: str, value) -> str:
    """Content identity for one materialized array, including shape/dtype."""
    digest = hashlib.sha256()
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(name.encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()

# The scalar reductions every recorded scorer takes of a 3-D snapshot, in the
# order verdict360.KEYS declares them: the five acceptance-gate metrics, then
# the six section/band/latitude-group transports the verdict adds.  They are
# stored as fp64 TIME SERIES next to the snapshots (see
# :func:`build_snapshot_reducer`) so a future ensemble can measure a spread
# that the float32 3-D storage quantum would otherwise swallow.
REDUCED_KEYS = ("acc", "up", "deep", "smax", "smean",
                "acc_mean", "band", "band_c", "g_south", "g_band", "g_north")
# Extra per-latitude-row transport profile stored alongside them.  One
# full-section profile covers the channel band and both flanking groups (the
# consumer slices it), so there is exactly one array and no second spelling of
# the row reduction.
REDUCED_ROW_KEY = "row_sv"
# Which 3-D fields a legacy (unstamped) artifact stored in float32.  Read by
# the quantum gates in verdict360/floor90_ensemble: an artifact written before
# the storage stamp existed gets today's behaviour, unchanged.
LEGACY_STORAGE_DTYPES = {"T3d": "float32", "S3d": "float32", "eta3d": "float32",
                         "u3d": "float32", "v3d": "float32",
                         "eta": "float32", "sst": "float32",
                         "u": "float32", "v": "float32"}


def _stress_content_sha256(tau_x, tau_y) -> str:
    """Content identity for the two T-point before-stress arrays."""
    digest = hashlib.sha256()
    for name, value in (("tau_x", tau_x), ("tau_y", tau_y)):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _initial_state_sha256(state) -> str:
    """Compact bit identity of the actual prognostic/before state at t=0."""
    digest = hashlib.sha256()
    names = (
        "T", "S", "u", "v", "eta",
        "T_before", "S_before", "u_before", "v_before", "eta_before",
        "tau_x_prev", "tau_y_prev", "land_mask", "u_mask", "v_mask",
    )
    for name in names:
        value = getattr(state, name, None)
        if value is None:
            digest.update(f"{name}:None|".encode("ascii"))
            continue
        value = getattr(value, "data", value)
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_content_sha256(path: str | None) -> str | None:
    """Hash an optional forcing/perturbation file for run-config identity."""
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _analytic_dino_tpoint_stress(grid, cfg, *, t_seconds: float):
    """Load analytic DINO T-point stress and bind it to one seasonal time."""
    if not np.isfinite(t_seconds) or t_seconds < 0.0:
        raise ValueError(
            f"before-stress reconstruction time must be finite and >=0, got "
            f"{t_seconds!r}")
    forcing = dino_lat_lon_surface_forcing_arrays(grid, cfg)
    surface = dino_step_surface_forcing(forcing)
    if surface is None or surface.tau_x is None or surface.tau_y is None:
        raise SystemExit("DINO analytic T-point stress loader returned no stress")
    tau_x = jnp.asarray(surface.tau_x)
    tau_y = jnp.asarray(surface.tau_y)
    return tau_x, tau_y, _stress_content_sha256(tau_x, tau_y)


def reconstruct_dino_before_stress_tpoint(state, grid, cfg, *, t_seconds: float):
    """Replace only the bridged before-stress carry with analytic T stress.

    This is oracle-mimicry glue: NEMO restart ``utau_b/vtau_b`` are already
    U/V-point fields, whereas legoESM's canonical carry is T-point.  Rebuild
    the prior T field through DINO's existing forcing loaders; never invert
    the face field.  DINO wind is time independent, but ``t_seconds`` binds
    this reconstruction to the restart's actual prior seasonal clock.
    """
    tau_x, tau_y, content_hash = _analytic_dino_tpoint_stress(
        grid, cfg, t_seconds=t_seconds)
    if state.tau_x_prev is None or state.tau_y_prev is None:
        raise SystemExit(
            "--bridge-before-stress-tpoint requires bridged before stress")
    if (tau_x.shape != state.tau_x_prev.shape
            or tau_y.shape != state.tau_y_prev.shape):
        raise SystemExit(
            "T-point before-stress shape mismatch: "
            f"analytic={tau_x.shape}/{tau_y.shape} "
            f"bridge={state.tau_x_prev.shape}/{state.tau_y_prev.shape}")
    rebuilt = state._replace(tau_x_prev=tau_x, tau_y_prev=tau_y)
    receipt = {
        "bridge_before_stress_stagger": "T",
        "bridge_before_stress_reconstruction_seconds": float(t_seconds),
        "bridge_before_stress_sha256": content_hash,
    }
    print("BEFORE-STRESS BRIDGE: reconstructed analytic T-point carry "
          f"at t={t_seconds:.0f}s sha256={content_hash}", flush=True)
    return rebuilt, receipt


def verify_day0_matches_restart(st, restart_state, land_mask, *, tol: float = 1e-8) -> None:
    """Day-0 gate: raise SystemExit unless ``st`` == the NEMO restart on wet cells.

    Guards against the 2026-07-24 defect where a "twin" silently started from
    the analytic rest-state IC instead of the bridged restart. Checks:
      (1) max|dT| = max|d_eta| = max|du| = max|dv| = 0 (to `tol`) vs the raw
          NEMO restart, restricted to wet cells;
      (2) max|u0| > 0.1 -- a rest-state start has u == 0 identically, so any
          twin claiming a developed IC must show real velocity.

    Parameters
    ----------
    st : legoESM ocean state (post-bridge, pre-integration)
    restart_state : NemoState (the raw restart read by ``read_nemo_restart``)
    land_mask : (n_lat, n_lon) wet mask (``br.land_mask``)
    """
    wet = np.asarray(land_mask) > 0.5
    t0_lego = np.asarray(st.T.data[:, :, 0])
    t0_nemo = np.asarray(restart_state.T[:, :, 0])
    eta0_lego = np.asarray(st.eta.data)
    eta0_nemo = np.asarray(restart_state.ssh)
    u0_lego = np.asarray(st.u.data[:, 1:, 0])   # east face of cell i -> NEMO un[i]
    u0_nemo = np.asarray(restart_state.u[:, :, 0])
    v0_lego = np.asarray(st.v.data[1:, :, 0])   # north face of cell j -> NEMO vn[j]
    v0_nemo = np.asarray(restart_state.v[:, :, 0])

    d_t = float(np.max(np.abs(t0_lego[wet] - t0_nemo[wet])))
    d_eta = float(np.max(np.abs(eta0_lego[wet] - eta0_nemo[wet])))
    d_u = float(np.max(np.abs(u0_lego[wet] - u0_nemo[wet])))
    d_v = float(np.max(np.abs(v0_lego[wet] - v0_nemo[wet])))
    max_u0 = float(np.max(np.abs(u0_lego[wet])))
    max_v0 = float(np.max(np.abs(v0_lego[wet])))

    print(f"DAY-0 VERIFY vs NEMO restart (wet cells): "
          f"max|dT|={d_t:.3e}  max|d_eta|={d_eta:.3e}  max|du|={d_u:.3e}  max|dv|={d_v:.3e}  "
          f"max|u0|={max_u0:.4f}  max|v0|={max_v0:.4f}", flush=True)

    failures = []
    if not d_t < tol:
        failures.append(f"T mismatch vs NEMO restart: {d_t:.3e} >= {tol:.1e}")
    if not d_eta < tol:
        failures.append(f"eta mismatch vs NEMO restart: {d_eta:.3e} >= {tol:.1e}")
    if not d_u < tol:
        failures.append(f"u mismatch vs NEMO restart: {d_u:.3e} >= {tol:.1e}")
    if not d_v < tol:
        failures.append(f"v mismatch vs NEMO restart: {d_v:.3e} >= {tol:.1e}")
    if not max_u0 > 0.1:
        failures.append(
            f"max|u0|={max_u0:.4f} <= 0.1 -- looks like a REST-STATE start, not a "
            "developed-restart twin (2026-07-24 defect: verify the caller passed "
            "br.state, not dino_lat_lon_state(...))"
        )
    if not max_v0 > 0.1:
        failures.append(
            f"max|v0|={max_v0:.4f} <= 0.1 -- looks like a REST-STATE start, not a "
            "developed-restart twin (2026-07-24 defect: verify the caller passed "
            "br.state, not dino_lat_lon_state(...))"
        )
    if failures:
        raise SystemExit(
            "DAY-0 GATE FAILED -- refusing to run a twin that is not verifiably "
            "initialized from the NEMO restart:\n  " + "\n  ".join(failures)
        )


def bridge_tke_from_restart(st, restart_en, land_mask, *,
                            restart_avm=None, restart_avt=None,
                            restart_dissl=None):
    """Seed ``st.tke`` from a NEMO restart's ``en`` (TKE closure integrator memory).

    ``restart_en`` is the raw ``(n_lat, n_lon, jpk)`` array from
    :func:`read_nemo_restart_en` (index 0 = surface w-level, matching
    ``gdepw_1d``). legoESM's ``state.tke`` is ``(n_lat, n_lon, nlev-1)`` at
    the interior interfaces (dims ``("lat","lon","level")``); since
    ``jpk == nlev`` (NEMO w/T-levels share one ``nav_lev`` axis), dropping the
    surface w-level index (``restart_en[..., 1:]``) leaves exactly ``nlev-1``
    levels aligned index-for-index with lego's interior interfaces. Masked to
    wet columns (matches the cold-start seed's ``land_mask``-gated fill).
    """
    en_interior = np.asarray(restart_en, dtype=np.float64)[..., 1:]  # drop w-level 0 (surface)
    wet = (np.asarray(land_mask) > 0.5)[:, :, None]
    tke_data = jnp.asarray(np.where(wet, en_interior, 0.0), dtype=st.T.data.dtype)
    updates = dict(tke=Field(data=tke_data, name="tke",
                            dims=("lat", "lon", "level"), units="m^2/s^2"))
    supplied = (restart_avm, restart_avt, restart_dissl)
    if any(x is not None for x in supplied) and any(x is None for x in supplied):
        raise ValueError(
            "restart_avm, restart_avt, and restart_dissl must be supplied "
            "together")
    if restart_avm is not None:
        avm = np.asarray(restart_avm, dtype=np.float64)
        avt = np.asarray(restart_avt, dtype=np.float64)
        dissl = np.asarray(restart_dissl, dtype=np.float64)
        if (avm.shape != restart_en.shape or avt.shape != restart_en.shape
                or dissl.shape != restart_en.shape):
            raise ValueError(
                "restart avm_k/avt_k/dissl must match en shape; got "
                f"{avm.shape}/{avt.shape}/{dissl.shape} vs {restart_en.shape}")
        updates.update(
            tke_avm=Field(
                data=jnp.asarray(np.where(wet, avm[..., 1:], 0.0),
                                 dtype=st.T.data.dtype),
                name="tke_avm", dims=("lat", "lon", "level"), units="m^2/s"),
            tke_avt=Field(
                data=jnp.asarray(np.where(wet, avt[..., 1:], 0.0),
                                 dtype=st.T.data.dtype),
                name="tke_avt", dims=("lat", "lon", "level"), units="m^2/s"),
            tke_avm_surface=Field(
                data=jnp.asarray(np.where(wet[..., 0], avm[..., 0], 0.0),
                                 dtype=st.T.data.dtype),
                name="tke_avm_surface", dims=("lat", "lon"), units="m^2/s"),
            tke_dissl=Field(
                data=jnp.asarray(np.where(wet, dissl[..., 1:], 0.0),
                                 dtype=st.T.data.dtype),
                name="tke_dissl", dims=("lat", "lon", "level"), units="s^-1"),
        )
    return st._replace(**updates)


def _print_before_bridge_verify(st, before, grid) -> None:
    """Print max|d_tb|/max|d_sb|/max|d_ub|/max|d_vb| vs the raw restart
    before-level (wet cells) -- the --bridge-before day-0 gate companion to
    ``verify_day0_matches_restart``'s now-level check.

    T/S use the FULL 3-D ``tmask`` (not the 2-D surface ``land_mask``): under
    full-step topography a wet surface column still has dry cells below
    ``k_bot``, where NEMO stores a raw 0.0 but the bridge's Neumann-fill
    extrapolates a nonzero value (matches the now-level bridge's own T/S
    fill) -- indexing those cells with the 2-D mask would spuriously flag
    the intentional fill as a mismatch.
    """
    tmask3 = np.asarray(grid.tmask) > 0.5
    d_tb = float(np.max(np.abs(np.asarray(st.T_before.data)[tmask3] - before.T[tmask3])))
    d_sb = float(np.max(np.abs(np.asarray(st.S_before.data)[tmask3] - before.S[tmask3])))
    umask3 = np.asarray(grid.umask) > 0.5
    vmask3 = np.asarray(grid.vmask) > 0.5
    d_ub = float(np.max(np.abs(
        np.asarray(st.u_before.data)[:, 1:, :][umask3] - before.u[umask3])))
    d_vb = float(np.max(np.abs(
        np.asarray(st.v_before.data)[1:, :, :][vmask3] - before.v[vmask3])))
    print(f"BEFORE-LEVEL BRIDGE VERIFY vs NEMO restart tb/sb/ub/vb (wet cells): "
          f"max|d_tb|={d_tb:.3e}  max|d_sb|={d_sb:.3e}  max|d_ub|={d_ub:.3e}  "
          f"max|d_vb|={d_vb:.3e}", flush=True)


def assert_nemo_seasonal_clock(stamped, path: str) -> tuple[float, float]:
    """Refuse an artifact whose seasonal clock is not the one NEMO is on.

    ``stamped`` is anything with ``__contains__`` and ``__getitem__`` over the
    stamp names -- an ``npz`` handle or a plain dict.  Returns the pair
    ``(t0_used, t0_nemo)`` in seconds.

    #1455 and its follow-up.  The first version of this guard rejected only
    ``t0 == 0``, which let every other manual offset through: an explicit step
    offset of one stamps 2700 s, still most of a year out of phase with the
    day-180 restart, and the gate called it a NEMO comparison.  Comparing
    against the offset read from the restart itself rejects ALL of them.
    """
    for name in ("seasonal_t0_seconds", "seasonal_t0_reference_seconds"):
        if name not in stamped:
            raise SystemExit(
                f"{path} carries no {name} stamp -- it predates the seasonal "
                "clock guard (#1455) and its forcing phase cannot be checked "
                "against NEMO's. Re-run the twin with the current "
                "kamm_twin_90d.py.")
    t0 = float(stamped["seasonal_t0_seconds"])
    t0_nemo = float(stamped["seasonal_t0_reference_seconds"])
    if abs(t0 - t0_nemo) > 0.5:
        raise SystemExit(
            f"{path} ran with its seasonal forcing at {t0 / 86400.0:.2f} d of "
            f"the 360-day year, but the NEMO run it is scored against is at "
            f"{t0_nemo / 86400.0:.2f} d "
            f"({abs(t0 - t0_nemo) / 86400.0:.2f} d out of phase). That is a "
            "cross-season comparison, not a fidelity measurement.")
    return t0, t0_nemo


# The ladder mode and precision this campaign's headline claims are about.
# A candidate on anything else is scoreable but NOT certifiable against that
# claim -- see :func:`certifiable_grid_and_precision`.
CLAIM_LADDER_MODE = "both"
CLAIM_CONTROL_DTYPE = "float64"
CLAIM_START_MODE = "bridged"


def vertical_ladder_sha256(z_coord) -> str:
    """Content hash of the vertical-coordinate arrays ACTUALLY in memory.

    GLM's #1640 point, and it is the right one: **a label is a taxonomy, not
    an identity**.  ``nemo_ladder_mode`` records which ladder the runner MEANT
    to build; it cannot tell two runs apart that carry the same label and
    different numbers (a changed bridge, a changed reference profile, a
    silently-different dtype).  Hashing the arrays makes "same grid" DECIDABLE
    instead of asserted: equal hash => bit-identical ladders, full stop.

    SCOPE -- THIS IS A LADDER IDENTITY, NOT A FULL GRID IDENTITY, and the
    name of the stamp should be read that way.  It hashes the four 1-D
    reference arrays, which is what the ladder MODE moves: the DINO bridge
    collapses NEMO's 3-D thickness/depth fields to a 1-D mean profile before
    they reach ``z_coord`` (``nemo_state_bridge.effective_vertical_scale_
    factors``; it raises if the card has genuine horizontal ``e3t_0``
    variation).  It deliberately does NOT cover ``h_partial`` /
    ``bottom_level`` / ``is_active`` -- the bathymetry and the wet/dry
    staircase -- so TWO RUNS ON DIFFERENT TOPOLOGIES HASH IDENTICALLY.  An
    earlier draft justified that exclusion by claiming the extra arrays
    "cannot distinguish the thing this hash exists to distinguish"; that is
    backwards -- extra content never reduces a hash's discriminating power --
    and is RETRACTED.  The honest reason is narrower: the ladder question is
    the one the gate's claim turns on, and keeping the input small keeps the
    stamp comparable across the runs already recorded.  Widening it to the
    topology is a strict improvement whenever someone wants it.

    Dtype is folded in deliberately: the same numbers at fp32 and fp64 are NOT
    the same grid for a claim that depends on precision.  A MISSING attribute
    and a present-but-``None`` one are also distinguished, so a future field
    rename degrades loudly rather than into a stable-looking weaker value.
    """
    import hashlib
    _MISSING = object()
    h = hashlib.sha256()
    for name in ("z_full_ref", "z_half_ref", "dz_ref", "t_depth_ref"):
        a = getattr(z_coord, name, _MISSING)
        if a is _MISSING:
            h.update(f"{name}:ABSENT|".encode())
            continue
        if a is None:
            h.update(f"{name}:None|".encode())
            continue
        a = np.asarray(a)
        h.update(f"{name}:{a.dtype.str}:{a.shape}|".encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def certifiable_grid_and_precision(stamped) -> tuple:
    """Report whether a candidate is on the claim's grid/precision.

    NOT named ``assert_*``: its sibling ``assert_nemo_seasonal_clock`` in this
    module RAISES, and this one deliberately does not -- an off-claim
    candidate is still worth scoring, so the caller decides between a verdict
    and UNCERTIFIED.  (It was briefly called ``assert_certifiable_...`` with an
    unused ``path`` argument; review caught both.)

    #1640, the reviewer's minimum ask.  The gate deliberately SCORES any
    ladder -- both are legitimately scoreable and the gate's job is to say
    which grid a score was earned on.  But this campaign's headline is
    specifically about the reference's ladders at double precision, and a gate
    that will score anything cannot certify that.  So the two are separated:
    the score is always printed, and the PASS/FAIL VERDICT is withheld
    (UNCERTIFIED) when the candidate is not on the claim's grid/precision.

    THE START MODE IS DELIBERATELY *NOT* A CRITERION HERE (#1455,
    2026-08-24), and the reasoning is recorded so it is a decision rather than
    an omission, and review corrected its shape.  It is ASYMMETRIC:

      * ``twin_start_mode`` MISSING  -> printed, never a reason.  The stamp is
        new, so NO recorded artifact carries it, the verdict-year baseline
        included.  Refusing them all for being unstamped would orphan the
        campaign's own baseline while catching nothing -- audited, every
        recorded twin build was in fact bridged.
      * ``twin_start_mode`` present and NOT ``"bridged"`` -> a reason, exactly
        like an off-ladder arm.  There is no bootstrap problem here: a run
        that STAMPED "euler" was deliberately started off the trajectory the
        thresholds were earned on, which is the same test the ladder criterion
        applies.  An earlier revision of this docstring argued the whole
        dimension out on the unstamped case's harm; that harm attaches only to
        ``None``, and the argument is RETRACTED for the stamped case.

    The consequence, stated because it is a behaviour change: the legacy
    ``--legacy-euler-start`` arm of the registered 90-day A/B scores in full
    and returns UNCERTIFIED (exit 3) rather than a PASS/FAIL tally.  Its
    numbers are unchanged and still printed as neutral (within Nx)/(over Nx)
    markers, which is the same information without the verdict token.

    Returns ``(ok, reasons, ladder, dtype)``.  Raises nothing -- the caller
    decides between a verdict and UNCERTIFIED, because an off-claim candidate
    is still worth scoring and printing.
    """
    ladder = (str(stamped["nemo_ladder_mode"])
              if "nemo_ladder_mode" in stamped else None)
    dtype = (str(stamped["control_dtype"])
             if "control_dtype" in stamped else None)
    reasons = []
    if ladder is None:
        reasons.append(
            "no nemo_ladder_mode stamp -- this artifact predates the stamp and "
            "does not record which vertical ladders it ran on")
    elif ladder != CLAIM_LADDER_MODE:
        reasons.append(
            f"vertical ladder is {ladder!r}, but the claim is about "
            f"{CLAIM_LADDER_MODE!r} (NEMO's own thickness AND depth ladders)")
    start = str(stamped["twin_start_mode"]) if "twin_start_mode" in stamped else None
    if start is not None and start != CLAIM_START_MODE:
        reasons.append(
            f"twin start mode is {start!r}, but the claim is about "
            f"{CLAIM_START_MODE!r} (NEMO's own leap-frog before level). A "
            "missing stamp is NOT a reason -- see this function's docstring.")
    if dtype is None:
        reasons.append(
            "no control_dtype stamp -- the precision this arm was built at is "
            "not recorded")
    elif dtype != CLAIM_CONTROL_DTYPE:
        reasons.append(
            f"control dtype is {dtype!r}, but the claim is about "
            f"{CLAIM_CONTROL_DTYPE!r}")
    return (not reasons), reasons, ladder, dtype


def restart_elapsed_seconds(path: str) -> float:
    """Model seconds elapsed at the restart, read from the restart ITSELF.

    NEMO writes both ``adatrj`` (elapsed days) and ``kt`` (step index) into the
    restart, and ``usrdef_sbc.F90:536`` makes the seasonal phase a function of
    ``REAL(kt)*rn_Dt``.  Reading ``adatrj`` makes the offset independent of the
    run's timestep; cross-checking it against ``kt*DT`` catches a restart whose
    timestep differs from this harness's ``DT``.
    """
    import netCDF4 as nc
    with nc.Dataset(path) as d:
        for name in ("adatrj", "kt"):
            if name not in d.variables:
                raise SystemExit(f"{path}: restart has no '{name}' variable")
        adatrj = float(np.asarray(d.variables["adatrj"][:]).ravel()[0])
        kt = float(np.asarray(d.variables["kt"][:]).ravel()[0])
    t_from_days, t_from_kt = adatrj * 86400.0, kt * DT
    if not np.isfinite([t_from_days, t_from_kt]).all():
        raise SystemExit(f"{path}: non-finite adatrj/kt ({adatrj}, {kt})")
    if abs(t_from_days - t_from_kt) > 0.5 * DT:
        raise SystemExit(
            f"{path}: restart adatrj={adatrj} d ({t_from_days:.0f} s) disagrees "
            f"with kt={kt:.0f} x DT={DT:.0f} s ({t_from_kt:.0f} s) -- the "
            "restart was written at a different timestep than this harness runs")
    return t_from_days


# Compatibility for branch-local committed probes written before main made the
# helper public.  Keep the old spelling until those frozen scorers are retired.
_restart_elapsed_seconds = restart_elapsed_seconds


def seasonal_t0_seconds(restart_path: str) -> float:
    """Absolute seasonal-clock offset [s] for a twin bridged from ``restart_path``.

    #1455.  DINO's analytic surface forcing is a function of the DAY OF YEAR
    through the ABSOLUTE step index (``usrdef_sbc.F90:536-547``:
    ``ztime = REAL(kt)*rn_Dt``), so a twin that restarts its own seasonal year
    at zero forces legoESM out of phase with the NEMO run it is compared
    against.  For the canonical ``DINO_00005760_restart.nc`` (day 180 of a
    360-day year) that offset is EXACTLY antiphase.  Correcting it collapsed
    the day-30 southern surface-density gap from -0.013224 to -0.000125
    kg/m3, i.e. from 139x the noise floor to 1.3x, AT THE TWO SAMPLED PHASES
    (commits 1c03f8311, 076217667, afd8e06b6).

    RETRACTED, 2026-08-23 (GLM review on PR #1634): this docstring previously
    said the offset "owns 99.1% of" that gap.  THE PERCENTAGE IS WITHDRAWN and
    no percentage replaces it.  Two sampled phases cannot yield a fraction --
    that needs a dose-response curve through intermediate offsets -- and the
    decomposition silently assumed error = season + ocean with no interaction
    term.  Report the two measured multipliers (139x -> 1.3x), never a share.
    BLIND SPOT, also unquantified: an exact 180/360-day antiphase flips only
    the ODD harmonics, so the SEMIANNUAL component is IN PHASE in both arms
    and this experiment cannot see semiannual error, including whatever part
    of the residual is semiannual.  Settling it needs a phase sweep
    (0/45/90/135/180 days) plus a control in which the REFERENCE model is run
    with artificially antiphased forcing -- named as follow-up, not run.

    DEFAULT (env unset) is therefore the NEMO clock, read from the restart
    ITSELF (``adatrj``, cross-checked against ``kt*DT``) -- never hardcoded and
    never scraped from a filename.  ``DINO_TWIN_SEASONAL_KT0`` (the same knob
    the clock A/B lane used) remains available to override it:

      unset / "restart"  -> t0 from the restart's own ``adatrj``   [DEFAULT]
      "0"                -> t0 = 0, the LEGACY relative clock; reproduces
                            historical (antiphase) numbers ONLY
      <integer>          -> t0 = <integer> * DT, explicit step offset

    Any non-default selection prints a loud banner, so a log can never be read
    without knowing which clock produced it.
    """
    env = os.environ.get("DINO_TWIN_SEASONAL_KT0")
    if env is None or env == "restart":
        t0_sec = restart_elapsed_seconds(restart_path)
        source = "restart adatrj" + ("" if env is None else " (explicit)")
    else:
        try:
            kt0 = int(env)
        except ValueError:
            raise SystemExit(
                f"Unknown DINO_TWIN_SEASONAL_KT0={env!r}: expected 'restart' "
                "(lowercase) or an integer step index") from None
        if kt0 < 0:
            raise SystemExit(
                f"DINO_TWIN_SEASONAL_KT0={kt0} is negative; expected >= 0")
        t0_sec = kt0 * DT
        banner = ("LEGACY RELATIVE CLOCK" if kt0 == 0
                  else f"MANUAL STEP OFFSET kt0={kt0}")
        source = f"OVERRIDE {banner}"
        print("\n" + "!" * 78, flush=True)
        print(f"!! {banner}: DINO_TWIN_SEASONAL_KT0={env}", flush=True)
        print("!! The seasonal forcing does NOT follow the restart's own "
              "day-of-year.", flush=True)
        print("!! This is a HISTORICAL-REPRODUCTION mode (#1455). Numbers "
              "produced here are", flush=True)
        print("!! NOT comparable to the NEMO run this twin scores against.",
              flush=True)
        print("!" * 78 + "\n", flush=True)
    print(f"seasonal clock: t_seconds = {t0_sec:.0f}s + (k+1)*{DT:.0f}s  "
          f"[source: {source}]  "
          f"(restart is day {(t0_sec / 86400.0) % 360.0:.2f} of the 360-day "
          f"year, {t0_sec / 86400.0:.2f} d elapsed in total; NEMO logs "
          f"nday_year = {int(t0_sec // 86400.0) % 360 + 1} at its next step)",
          flush=True)
    return t0_sec


# The accepted selections come from the bridge itself (NEMO_E3T_MODES), never
# re-listed here: a harness that accepted a mode the bridge rejects three calls
# later would fail deep inside grid construction instead of at the CLI.
NEMO_LADDER_TWIN_DEFAULT = "both"
_LADDER_ANNOUNCED: set[str] = set()      # modes whose loud banner already fired


def resolve_snap_days(snap_days, n_days: int, save_3d: bool) -> tuple[int, ...]:
    """Which days get a full 3-D snapshot.

    ``snap_days=None`` keeps the recorded :data:`SNAP_DAYS` grid, so every
    artifact produced before the argument existed is reproduced exactly.  Days
    beyond ``n_days`` are dropped rather than raising: a caller asking for a
    year-long grid on a 90-day run gets the 90-day prefix, which is what the
    filter did before this was a function.  Without ``save_3d`` there are no
    snapshots at all and the answer is empty regardless of what was asked for.
    """
    if not save_3d:
        return ()
    grid = SNAP_DAYS if snap_days is None else tuple(int(d) for d in snap_days)
    return tuple(d for d in grid if d <= n_days)


def snapshot_dtype(fp64_3d: bool):
    """Storage dtype for the 3-D snapshot block.

    DEFAULT IS FLOAT32 AND STAYS FLOAT32.  The 3-D block is by far the largest
    thing in the artifact and doubling every recorded twin would buy nothing
    for the scores that are already resolved -- the fp64 REDUCED SERIES
    (:func:`build_snapshot_reducer`) is the always-on cheap win.  ``--fp64-3d``
    is for the runs where measuring the SPREAD between near-identical members
    is the point, which is exactly where float32 storage has capped the
    campaign (2 differing cells of 342134 at day 10; members tying at the
    storage quantum on max-type metrics).
    """
    return np.float64 if fp64_3d else np.float32


def storage_stamp(snap_name: str, reduced_resolution: str) -> str:
    """The per-field storage stamp, as the ONE json spelling every writer uses.

    ``snap_name`` is the 3-D block's storage dtype, or ``"absent"`` when no
    block was written -- a field with no snapshot on disk must not be stamped
    with the dtype it would have had.

    ``reduced_resolution`` is deliberately NOT "float64 because we stored it in
    a float64 array".  The series is always stored float64, but its RESOLUTION
    is the precision the arm was BUILT at: on a deliberate ``FP64=0`` arm the
    series is float64-stored and float32-resolved, and a consumer that read
    "float64" here would credit it with resolution it does not have (physics
    review). ``"absent"`` when no series was written.

    Factored out because the verification instrument used to hand-write this
    same json -- a second spelling of the very thing it exists to check, which
    could not notice a change to the real one (code review).
    """
    return json.dumps(
        {"T3d": snap_name, "S3d": snap_name, "eta3d": snap_name,
         "u3d": snap_name, "v3d": snap_name,
         "eta": "float32", "sst": "float32", "u": "float32", "v": "float32",
         "reduced": reduced_resolution},
        sort_keys=True)


def reduced_series_kwargs(reduced, days) -> dict:
    """The npz keys for the fp64 reduced series, restricted to ``days``.

    Separated from :func:`run_twin` so the stamp and the CONTENT are computed
    from ONE day list.  They were computed thirty lines apart, and a run whose
    snapshot grid excluded day 0 but which then blew up before its first
    requested day stamped a series as present while writing none -- exactly the
    "missing key with no stated reason" this change exists to remove (code
    review).
    """
    days = [d for d in sorted(days) if d in reduced]
    if not days:
        return {}
    out = {"reduced_days": np.asarray(days, dtype=np.int32)}
    for k in REDUCED_KEYS:
        out[f"reduced_{k}"] = np.asarray([reduced[d][k] for d in days],
                                         dtype=np.float64)
    out[f"reduced_{REDUCED_ROW_KEY}"] = np.asarray(
        [reduced[d][REDUCED_ROW_KEY] for d in days], dtype=np.float64)
    return out


def snapshot_storage_dtypes(stamped) -> dict:
    """Per-field STORAGE dtypes of an artifact, honestly.

    ``control_dtype`` records the precision the arm was BUILT at; it says
    nothing about what was written to disk, and for every artifact recorded
    before this stamp the two differ (fp64 compute, fp32 storage).  This reads
    the per-field ``storage_dtypes`` stamp; an artifact that predates it gets
    :data:`LEGACY_STORAGE_DTYPES`, i.e. today's behaviour unchanged.
    """
    raw = stamped["storage_dtypes"] if "storage_dtypes" in stamped else None
    if raw is None:
        return dict(LEGACY_STORAGE_DTYPES)
    return json.loads(str(raw))


def build_snapshot_reducer(run_traj: str):
    """``(reducer, status)`` -- the fp64 scalar reductions of a 3-D snapshot.

    Every reduction here is IMPORTED from the recorded scorers
    (``acceptance_gate_90d``, ``acc_driver_decomp``, ``floor90_ensemble``,
    ``basin_seasonal_decomp``) and none is re-spelled.  A second spelling of
    any of them is a reduction drift, which is the defect class this campaign
    has spent the most time on; the ACC pair in :func:`run_twin` imports its
    reducer for the same reason.

    WHAT THE RESOLUTION ADVANTAGE ACTUALLY IS, stated because it is
    conditional: the reduction runs on the LIVE model state before the storage
    cast, so it carries whatever precision the arm was built at.  On an fp64
    arm (the campaign default, ``FP64=1``) that is strictly more resolution
    than reducing the stored float32 field.  On a deliberate ``FP64=0`` fp32
    arm the stored field is already lossless and the series buys nothing --
    ``control_dtype`` is the stamp that says which.

    The imports are deferred: ``acceptance_gate_90d`` imports THIS module, and
    ``acc_thermal_wind`` opens a hard-coded mesh at import time.  If the
    scorers cannot be loaded, or describe a different mesh than this run, the
    reducer is ``None`` and ``status`` says why -- the reason is STAMPED into
    the artifact so a consumer reads it instead of guessing at a missing key.
    """
    import sys
    d = os.path.dirname(os.path.abspath(__file__))
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        import acc_thermal_wind as A
        import acceptance_gate_90d as G
        import acc_driver_decomp as D
        import floor90_ensemble as F
        import basin_seasonal_decomp as B
    except Exception as exc:                      # pragma: no cover - env-dependent
        return None, f"scorers unimportable: {type(exc).__name__}: {exc}"
    mesh = os.path.abspath(f"{run_traj}/mesh_mask.nc")
    got = os.path.abspath(A.mm.filepath())
    if got != mesh:
        # Same refusal the daily-ACC reducer makes: two meshes are two
        # geometries, and a series reduced on the wrong one is a confidently
        # wrong number rather than a missing one.
        return None, f"scorer mesh {got} is not this run's mesh {mesh}"

    def reducer(f64):
        """{metric: float} + the per-row profile, from live fp64 arrays."""
        # The u slice is load_candidate's, verbatim: faces 1..52 map to NEMO's
        # u columns.  (load_candidate then writes u[:,47,:] = lU[:,48,:], which
        # is a NO-OP after this slice -- index 47 already IS original column 48
        # -- and is deliberately not reproduced, same call as _acc_pair's.)
        u = f64["u"][:, 1:53, :]
        # The GATE's mask (acceptance_gate_90d.main:356), not A.tmask alone:
        # both sides of the recorded gate are scored on this intersection, and
        # on these artifacts the two are bit-identical (verdict360.lego_state
        # asserts exactly that).  Using the gate's spelling means the stored
        # series IS the gate's quantity on any card, including one whose land
        # differs.
        wet = A.tmask & (f64["land_mask"] > 0.5)[:, :, None]
        st = {"T": f64["T"], "S": f64["S"], "u": u,
              "land_mask": f64["land_mask"]}
        m = dict(G.metrics(st, wet))
        m["acc_mean"] = D._avg(D.section_total(u, A.umask))
        m["band"] = F.band_transport(u, A.umask)
        m["band_c"] = F.band_transport_campaign(u, A.umask)
        for key, (_name, rows) in zip(("g_south", "g_band", "g_north"),
                                      D.LAT_GROUPS):
            m[key] = D._avg(D.group_transport(u, A.umask, rows))
        resid = abs(m["g_south"] + m["g_band"] + m["g_north"] - m["acc_mean"])
        if resid > 1e-9:
            raise SystemExit(
                f"FATAL: the three latitude groups do not partition the full "
                f"section: residual {resid:.3e} Sv")
        missing = [k for k in REDUCED_KEYS if k not in m]
        if missing:
            raise SystemExit(f"FATAL: reducer produced no {missing}")
        m[REDUCED_ROW_KEY] = B.row_transport(u, A.umask, slice(0, A.NY))
        return m

    return reducer, "ok"


def capture_snapshot(fields, *, snap_dtype, reducer, land_mask):
    """``(stored, reduced, status)`` for one snapshot day.

    ``stored`` is the 3-D block cast to ``snap_dtype`` (what goes in the npz);
    ``reduced`` is the scalar series computed from float64 views of the SAME
    arrays BEFORE that cast, so the series keeps the trajectory's own
    resolution even when the block is stored float32.  That ordering is the
    whole point and it is why this is one function rather than two call sites.

    ``land_mask`` is a reducer INPUT, not a snapshot field: the reductions all
    need the wet domain, but the mask is time-invariant and is already written
    once per run, so it is never cast to ``snap_dtype`` and never appears in
    ``stored``.  It is REQUIRED, deliberately: it was optional for exactly one
    commit and a call site that forgot it shipped, crashing every snapshot run
    (both reviews asked for this).  A default would re-arm that bug and let a
    future caller reduce on no mask at all.

    ``reduced`` is ``None`` exactly when ``status`` says why -- the return
    shape never changes with the outcome, so a caller cannot accidentally
    unpack two different things.

    ``reducer`` is injected so the storage contract is testable without the
    NEMO mesh :func:`build_snapshot_reducer` loads.
    """
    stored = {k: np.asarray(v, dtype=snap_dtype) for k, v in fields.items()}
    if reducer is None:
        return stored, None, "no reducer"
    live = {k: np.asarray(v, dtype=np.float64) for k, v in fields.items()}
    live["land_mask"] = np.asarray(land_mask, dtype=np.float64)
    red, status = safe_reduce(reducer, live)
    return stored, red, status


def safe_reduce(reducer, live):
    """``(metrics or None, status)`` -- run one day's reduction, and NEVER let
    it kill the integration.

    The reduction is a DIAGNOSTIC written alongside the primary data, and the
    npz is only written after the whole time loop.  A reducer that raised at
    day 90 of a 90-day twin would therefore delete 90 days of compute to
    protect a few kB of annotation -- a diagnostic destroying the data it
    annotates (both reviews).  So a failing day loses its series and says why;
    the status is stamped into the artifact.
    """
    try:
        return reducer(live), "ok"
    except (Exception, SystemExit) as exc:         # noqa: BLE001 - see below
        # SystemExit is a BaseException, NOT an Exception, and it is exactly
        # what the reducer raises on a partition-residual failure -- a bare
        # `except Exception` here caught nothing and the run still died. Found
        # by the unit test, after the first version of this fix. KeyboardInterrupt
        # is deliberately still allowed through: a human asking the run to stop
        # must not be turned into a dropped diagnostic.
        return None, f"reduction failed: {type(exc).__name__}: {exc}"


def resolve_ladder_mode(legacy_1d_ladder: bool = False) -> str:
    """Which of NEMO's vertical ladders this TWIN hands legoESM, and why.

    #1455.  The bridge (:func:`legoesm.ocean.fidelity.nemo_state_bridge.
    effective_vertical_scale_factors`) resolves ``LEGOESM_NEMO_E3T`` to ``"off"``
    when nothing sets it -- i.e. legoESM is built on a 1-D reference ladder that
    is NOT the ladder the NEMO run integrates with.  That model-wide default is
    left alone; this changes the default for THIS RUNNER'S twin path only
    (:func:`run_twin`), where the whole point of the run is that the two models
    share a grid.  It is deliberately NOT called from :func:`_build_twin_state`,
    because a dozen sibling probes import that helper directly and must keep the
    grid they were recorded on.

    Measured cost of the 1-D ladder, two 90-day twin arms from the day-180
    restart differing ONLY in this mode, bit-identical at day 0, both stable,
    both under an fp64 precision policy (corrected clock, ``--bridge-before``):
    day-90 circumpolar channel-band transport error vs NEMO +2.93 Sv on ``"off"``
    against +0.29 Sv on ``"both"``, and full-section ACC error +1.87 Sv against
    -0.60 Sv.

    PLAUSIBLE, not confirmed: that it takes BOTH halves because the thickness
    ladder carries the barotropic component and the depth ladder the baroclinic
    one.  That split comes from the two half-ladder arms, which were measured at
    the fp32 control dtype and have NOT been re-run at fp64 -- unlike the two
    arms quoted above.

    Resolution order, highest first:

      ``LEGOESM_NEMO_E3T`` set  -> that mode, whatever it is    [override kept]
      ``--legacy-1d-ladder``    -> ``"off"``, with a loud banner
      nothing                   -> ``"both"``                   [TWIN DEFAULT]

    Setting the variable to something the flag contradicts is FATAL rather than
    silently letting one win.

    THIS FUNCTION IS PURE apart from its printing: it RETURNS the mode and does
    NOT write ``LEGOESM_NEMO_E3T``.  It used to write it, because that was the
    only channel the bridge read, and that write-back was the source of a whole
    class of defect: a second caller could not tell the operator's setting from
    the value the first call had planted, so an in-process two-arm sweep silently
    ran the same grid twice.  No check can separate those two cases -- they are
    the same string in the same slot -- so the write-back is gone instead, and
    the resolved mode travels to the bridge as an ARGUMENT (``e3t_mode``).  Two
    different ladders in one process are now simply two calls.
    """
    env = os.environ.get("LEGOESM_NEMO_E3T")
    if env is not None:
        if env not in NEMO_E3T_MODES:
            raise SystemExit(
                f"Unknown LEGOESM_NEMO_E3T={env!r}: expected one of "
                f"{', '.join(NEMO_E3T_MODES)}")
        if legacy_1d_ladder and env != "off":
            raise SystemExit(
                f"--legacy-1d-ladder selects the 1-D ladder ('off') but "
                f"LEGOESM_NEMO_E3T={env!r} selects {env!r}; they disagree. "
                "Pass one or the other, not both.")
        source = ("LEGOESM_NEMO_E3T + --legacy-1d-ladder (agreeing)"
                  if legacy_1d_ladder else "LEGOESM_NEMO_E3T (explicit override)")
        mode = env
    elif legacy_1d_ladder:
        mode, source = "off", "--legacy-1d-ladder"
    else:
        mode, source = NEMO_LADDER_TWIN_DEFAULT, "twin default"
    # The banner is once per MODE per process (repeating it trains people to
    # ignore it); the one-line statement of the grid is EVERY call, so no twin
    # can ever be built without its log saying which ladder it stands on.
    # Keyed to the GRID, not to the default: if someone flips the default back,
    # the warning must survive rather than vanish with it.
    if mode != "both" and mode not in _LADDER_ANNOUNCED:
        _LADDER_ANNOUNCED.add(mode)
        print("\n" + "!" * 78, flush=True)
        print(f"!! NON-DEFAULT VERTICAL LADDER: {mode!r}  [{source}]", flush=True)
        if mode == "off":
            print("!! This is the 1-D REFERENCE ladder, NOT the grid the NEMO "
                  "run integrates", flush=True)
            print("!! with. Measured cost at day 90: +2.93 Sv of circumpolar "
                  "transport error", flush=True)
            print("!! against +0.29 Sv on NEMO's own ladders (#1455). Numbers "
                  "produced here", flush=True)
            print("!! score legoESM on a grid NEMO does not have.", flush=True)
        elif mode == "gdept_only":
            print("!! Mixed ladders: cells from the 1-D thickness ladder, "
                  "T-points from NEMO's.", flush=True)
            print("!! Worst level of each grid: this one puts T-points 110.2 m "
                  "from the centre", flush=True)
            print("!! of the cell they sit in (at k=32), against 11.3 m on "
                  "'both' (at k=34).", flush=True)
            print("!! NEMO's own T-points are not cell centres either, so "
                  "'both' is offset too --", flush=True)
            print("!! just 10x less (#1455).", flush=True)
        else:   # e3t_only
            print("!! Mixed ladders: cells from NEMO's thickness ladder, "
                  "T-points from the 1-D one.", flush=True)
            print("!! Worst WET level of each grid: this one puts T-points "
                  "97.2 m from the centre", flush=True)
            print("!! of the cell they sit in (at k=32), against 11.3 m on "
                  "'both' (at k=34).", flush=True)
            print("!! It also carries only the barotropic half of the geometry "
                  "fix, and that", flush=True)
            print("!! split was measured at fp32 -- PLAUSIBLE, not confirmed "
                  "(#1455).", flush=True)
        print("!" * 78 + "\n", flush=True)
    print(f"vertical ladder: LEGOESM_NEMO_E3T={mode}  [source: {source}]  "
          f"(twin default {NEMO_LADDER_TWIN_DEFAULT!r} = NEMO's own thickness "
          f"AND T-depth ladders)", flush=True)
    return mode


# The start mode this runner's twin path takes when nothing selects one.
TWIN_START_MODE_DEFAULT = "bridged"
_START_ANNOUNCED: set[str] = set()      # modes whose loud banner already fired


def resolve_start_mode(bridge_before: bool) -> str:
    """Which time-integration START this twin takes, and why -- ``"bridged"``
    (NEMO's own before level, the DEFAULT) or ``"euler"`` (legacy).

    #1455.  This is the ladder pattern applied to the OTHER harness default
    that was silently deciding results.  It is pure apart from its printing:
    it RETURNS the mode and writes nothing.

    WHY THE DEFAULT IS ``"bridged"``.  A twin exists to compare two models on
    the same state; an Euler start makes legoESM integrate a DIFFERENT
    trajectory from step 1 for a reason that has nothing to do with the
    physics under test.  It does three separable things:

      1. HALF-STEP DELAY.  Unfiltered, and ONLY when the reference's before
         level equals its now level, an Euler-started leap-frog trajectory is
         exactly the two-point running mean of the true one, and a two-point
         running mean is a half-step delay at every frequency.  This term does
         not grow.  With the card's Asselin gamma=0.1 the identity is
         approximate (residual 0.8-6.4% of signal) and the lag rises with
         period toward ~0.5/(1-gamma) = 0.556 rather than sitting at 0.500.
      2. A PERMANENT STATE PERTURBATION of half a leap-frog step of tendency,
         because a DEVELOPED restart's before level is not its now level.
         Measured on the day-180 restart in the gate's own metric: one
         leap-frog step of circumpolar transport is 4.11e-3 Sv, so the Euler
         start injects ~2.06e-3 Sv at t=0.  This term GROWS and dominates at
         long lead.  The running-mean identity says nothing about it.
      3. The branch returns before NEMO's step-1 after-level reconciliation
         (#1640 finding 3), so step 1 commits a depth-mean deposit NEMO
         removes.

    RETRACTED here and at four sibling sites: the claim that the trajectory
    simply IS the running mean of the true one.  It overstates the exactness
    and omits term 2.

    CARRIED, NOT REPRODUCED: the impulse-lag pair 0.470 (Euler) / 0.000142
    (bridged) comes from the ``fidelity/dino-eta-waves`` harness, which is not
    on this branch; no committed probe in this tree regenerates it, and 0.470
    sits below the filtered prediction of 0.544-0.556, reconciled only by its
    own bootstrap interval.  PLAUSIBLE until that probe lands.  The flip does
    not rest on those values -- it rests on terms 2 and 3, which are measured
    from the restart itself.

    WHAT IT IS NOT.  The 5-day evidence says the effect is LAUNCH-concentrated
    (impulse response 65x better bridged at launch; only 7.6x better late, and
    the day-5 worst cell marginally WORSE bridged).  So this is not a claim
    that recorded endpoint metrics are wrong -- that is a measurement, and the
    90-day A/B that bounds it is registered separately.

    Resolution order, highest first:

      ``--legacy-euler-start`` / ``--no-bridge-before``  -> ``"euler"``, loud
      nothing                                            -> ``"bridged"``

    There is no environment channel and therefore no disagreement case: unlike
    the ladder, the only way to select the legacy start is the flag.
    """
    mode = TWIN_START_MODE_DEFAULT if bridge_before else "euler"
    # Banner once per MODE per process (repeating it trains people to ignore
    # it); the one-line statement is EVERY call, so no twin can be built
    # without its log saying which start it took.  Keyed to the MODE, not to
    # the default: if someone flips the default back, the warning survives.
    if mode != "bridged" and mode not in _START_ANNOUNCED:
        _START_ANNOUNCED.add(mode)
        print("\n" + "!" * 78, flush=True)
        print("!! LEGACY FORWARD-EULER START: this twin does NOT continue "
              "NEMO's leap-frog.", flush=True)
        print("!! It DELAYS the trajectory by about half a step at every "
              "frequency (the", flush=True)
        print("!! two-point RUNNING MEAN identity, exact only when the "
              "reference's before", flush=True)
        print("!! level equals its now level -- which a developed restart's "
              "does NOT) AND it", flush=True)
        print("!! INJECTS a permanent perturbation of half a leap-frog step "
              "of tendency:", flush=True)
        print("!! about 2.06e-3 Sv of circumpolar transport at t=0, measured "
              "on this very", flush=True)
        print("!! restart. The injection GROWS; the delay does not. Any "
              "phase, lag, wave,", flush=True)
        print("!! fast-response or 2-dt number produced here is that "
              "artifact, and shifting", flush=True)
        print("!! a result half a step does NOT undo it.", flush=True)
        print("!! It ALSO skips NEMO's step-1 after-level reconciliation "
              "(#1640 finding 3):", flush=True)
        print("!! step 1 commits a depth-mean deposit NEMO removes.",
              flush=True)
        print("!! Historical-reproduction mode ONLY. NOTE: this is NOT how "
              "the recorded gate", flush=True)
        print("!! or verdict-year artifacts were produced -- audited, 18 of "
              "18 recorded twin", flush=True)
        print("!! builds passed the bridge explicitly. The flag exists for "
              "ad-hoc runs that", flush=True)
        print("!! did not, which is where the retracted half-step lag came "
              "from.", flush=True)
        print("!" * 78 + "\n", flush=True)
    print(f"twin start: {mode}  [bridge_before={bool(bridge_before)}]  "
          f"(default {TWIN_START_MODE_DEFAULT!r} = NEMO's own leap-frog "
          f"before level)", flush=True)
    return mode


def start_mode_of(stamped) -> str | None:
    """The ``twin_start_mode`` an artifact records, or ``None`` if unstamped.

    Shared so every scorer reads the stamp the same way instead of inferring
    the start from a filename or from whatever the default was on the day.
    ``None`` means the artifact PREDATES the stamp and therefore does not
    record its start mode.  It does NOT mean "Euler": every recorded twin
    build audited in this repo's run logs (18 of 18) was bridged, so guessing
    either way from the date would be wrong more often than right.  Read the
    run log.
    """
    return (str(stamped["twin_start_mode"])
            if "twin_start_mode" in stamped else None)


def _build_twin_state(recipe: str, run_traj: str, run_stepdump: str, *,
                       bridge_tke: bool = False, bridge_before: bool = True,
                       bridge_before_stress_tpoint: bool = True,
                       vmix_scheme: str | None = None,
                       use_gm_redi: bool | None = None,
                       surface_stress_implicit: bool | None = None,
                       surface_tendency_placement: str | None = None,
                       barotropic_continuity_evaluation: str | None = None,
                       vface_zonal_metric_evaluation: str | None = None,
                       tke_preclosure_coeff_source: str | None = None,
                       tke_matrix_evaluation: str | None = None,
                       tke_solver_evaluation: str | None = None,
                       zdf_implicit_solver_evaluation: str | None = None,
                       tke_etau_exponential_evaluation: str | None = None,
                       tke_htau_evaluation: str | None = None,
                       tke_mxl_raw_evaluation: str | None = None,
                       tke_langmuir_evaluation: str | None = None,
                       tke_shear_evaluation_stage: str | None = None,
                       tke_shear_metric_source: str | None = None,
                       tke_n2_evaluation_stage: str | None = None,
                       dino_wind_profile_evaluation: str | None = None,
                       gm_redi_slope_n2_evaluation: str | None = None,
                       gm_redi_slope_prd_evaluation: str | None = None,
                       gm_redi_slope_metric_evaluation: str | None = None,
                       gm_redi_slope_face_thickness_evaluation: str | None = None,
                       gm_redi_slope_depth_evaluation: str | None = None,
                       u_m: float | None = None,
                       restart_file: str = RESTART_FILE,
                       e3t_mode: str | None = None,
                       bridge_omega: str = "nemo"):
    """Bridge the NEMO restart into a legoESM state and run the day-0 gate.

    ``restart_file``: basename of the (rebuilt, single-file) NEMO restart
    inside ``run_stepdump``. Default is the day-180 spin-up state; pass e.g.
    "DINO_00057600_restart.nc" to twin from NEMO's *year-5* state, which turns
    the twin into a matched-state GROWTH-RATE test (same developed state, same
    window, one variable = the model) rather than a from-rest comparison.

    ``vmix_scheme``: optional override of ``DINOConfig.vmix_scheme`` (e.g.
    "constant" for the #1317 TKE-vs-constant-mixing discriminator). ``None``
    (default) leaves the recipe's own vmix_scheme untouched.

    ``use_gm_redi``: optional override of ``DINOConfig.use_gm_redi`` (GM
    ablation discriminator: False zeroes the eddy-induced/bolus coefficient
    -- kappa_GM and the Treguier/Visbeck adaptive-kappa diagnostics -- while
    leaving kappa_Redi / the isoneutral slope machinery untouched and active,
    since dino.py's isoneutral GMRediConfig branch builds Redi unconditionally
    and only gates kappa_GM/treguier.enabled/visbeck.enabled on this flag;
    see dino.py:2437-2503). ``None`` (default) leaves the recipe's own value.

    ``surface_tendency_placement``: optional override of
    ``DINOConfig.surface_tendency_placement`` (#1492 A/B: "applied_now" is
    the legacy defect path -- surface_forcing mutates T/S directly before
    model.step(); "leapfrog_rhs" is the NEMO-faithful fix -- the surface
    tendency rate is folded into the Nnn RHS instead, see dino.py:262-281).
    ``None`` (default) leaves the recipe's own value.

    ``surface_stress_implicit``: optional one-variable override of the wind
    boundary-condition placement. ``None`` leaves the recipe unchanged;
    True routes the same centred stress through the implicit vertical solve.

    ``bridge_before_stress_tpoint``: bridge-only correction for NEMO's U/V
    restart stress, enabled by default. Requires ``bridge_before`` and
    replaces only legoESM's T-point previous-stress carry with the existing
    analytic DINO forcing at the restart's own time. ``False`` is the
    known-wrong historical U-as-T reproduction path, not a model-config
    selector.

    ``u_m``: optional override of ``DINOConfig.U_M`` (NEMO ``rn_Uv``, the
    lateral viscous velocity scale [m/s], card default 0.27).  It is the
    ONLY input to the lateral-viscosity coefficient on this card --
    ``A_h_base = 0.5*U_M*R*dlon`` (dino.py:3020) and nothing else reads it
    (dino.py:3494 is the MPAS builder, not this lat-lon lane; B_h is 0 and
    the barotropic diffusion is off), so scaling it is a genuine
    one-variable viscosity ablation.  ``None`` (default) leaves 0.27.

    Returns (br, cfg, mc, model, forcing, sf, st) ready to integrate.
    """
    # NOTE: this helper deliberately does NOT resolve the vertical ladder. A
    # dozen sibling probes import it directly and must keep the grid they were
    # recorded on; the twin default is applied in run_twin, one level up and
    # handed down through e3t_mode. e3t_mode=None keeps the bridge's own
    # resolution (LEGOESM_NEMO_E3T, else the 1-D ladder) exactly as before.
    if bridge_before_stress_tpoint and not bridge_before:
        raise SystemExit(
            "--bridge-before-stress-tpoint requires --bridge-before; a "
            "T-point prior-stress carry cannot be attached to an Euler start")
    # Union merge (PR #1695 x cf/main): the bridge is built ONCE, carrying both
    # main's metric/Coriolis-placement selectors AND the PR's oracle-Omega /
    # f_reference_mode selection.  cfg is built first because the bridge reads
    # cfg.vface_zonal_metric_evaluation / cfg.coriolis_placement /
    # cfg.tke_htau_evaluation.
    bridge_omega_value, f_reference_mode = resolve_bridge_omega(bridge_omega)
    cfg = dataclasses.replace(dino_config_for_recipe(recipe),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    g = read_nemo_mesh_mask(f"{run_traj}/mesh_mask.nc", nn_hls=0)
    s = read_nemo_restart(f"{run_stepdump}/{restart_file}", nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(
        g, s, periodic_i=True, full_step=True, e3t_mode=e3t_mode,
        omega=bridge_omega_value,
        f_reference_mode=f_reference_mode,
        vface_zonal_metric_evaluation=(
            vface_zonal_metric_evaluation
            if vface_zonal_metric_evaluation is not None
            else cfg.vface_zonal_metric_evaluation),
        coriolis_placement=cfg.coriolis_placement,
        carry_native_lat_deg=(cfg.tke_htau_evaluation == "nemo_literal"))
    print(f"ARM: bridge_omega={bridge_omega} "
          f"omega={bridge_omega_value:.17g} "
          f"f_reference_mode={f_reference_mode}", flush=True)
    if vmix_scheme is not None:
        cfg = dataclasses.replace(cfg, vmix_scheme=vmix_scheme)
    if use_gm_redi is not None:
        cfg = dataclasses.replace(cfg, use_gm_redi=use_gm_redi)
    if surface_stress_implicit is not None:
        cfg = dataclasses.replace(
            cfg, surface_stress_implicit=bool(surface_stress_implicit))
    if surface_tendency_placement is not None:
        cfg = dataclasses.replace(cfg, surface_tendency_placement=surface_tendency_placement)
    if barotropic_continuity_evaluation is not None:
        if barotropic_continuity_evaluation not in ("generic", "nemo_literal"):
            raise ValueError("barotropic_continuity_evaluation must be "
                             "'generic' or 'nemo_literal'")
        cfg = dataclasses.replace(
            cfg, barotropic_continuity_evaluation=barotropic_continuity_evaluation)
    if vface_zonal_metric_evaluation is not None:
        if vface_zonal_metric_evaluation not in (
                "legacy_tracer_midpoint", "nemo_vpoint"):
            raise ValueError(
                "vface_zonal_metric_evaluation must be "
                "'legacy_tracer_midpoint' or 'nemo_vpoint'")
        cfg = dataclasses.replace(
            cfg, vface_zonal_metric_evaluation=vface_zonal_metric_evaluation)
    if tke_preclosure_coeff_source is not None:
        if tke_preclosure_coeff_source not in (
                "carried_previous_step", "current_subiteration"):
            raise ValueError(
                "tke_preclosure_coeff_source must be "
                "'carried_previous_step' or 'current_subiteration'")
        cfg = dataclasses.replace(
            cfg, tke_preclosure_coeff_source=tke_preclosure_coeff_source)
    if tke_matrix_evaluation is not None:
        if tke_matrix_evaluation not in ("nemo_literal", "factored"):
            raise ValueError(
                "tke_matrix_evaluation must be 'nemo_literal' or 'factored'")
        cfg = dataclasses.replace(
            cfg, tke_matrix_evaluation=tke_matrix_evaluation)
    if tke_solver_evaluation is not None:
        if tke_solver_evaluation not in ("nemo_literal", "shared_thomas"):
            raise ValueError(
                "tke_solver_evaluation must be 'nemo_literal' or "
                "'shared_thomas'")
        cfg = dataclasses.replace(
            cfg, tke_solver_evaluation=tke_solver_evaluation)
    if zdf_implicit_solver_evaluation is not None:
        if zdf_implicit_solver_evaluation not in (
                "nemo_literal", "shared_thomas"):
            raise ValueError(
                "zdf_implicit_solver_evaluation must be 'nemo_literal' or "
                "'shared_thomas'")
        cfg = dataclasses.replace(
            cfg,
            zdf_implicit_solver_evaluation=zdf_implicit_solver_evaluation)
    for value, field, choices in (
        (tke_etau_exponential_evaluation,
         "tke_etau_exponential_evaluation", ("nemo_literal", "jax_expression")),
        (tke_htau_evaluation, "tke_htau_evaluation",
         ("nemo_literal", "jax_expression")),
        (tke_mxl_raw_evaluation, "tke_mxl_raw_evaluation",
         ("nemo_literal", "factored")),
        (gm_redi_slope_n2_evaluation, "gm_redi_slope_n2_evaluation",
         ("carried_step_entry", "recompute")),
        (gm_redi_slope_prd_evaluation, "gm_redi_slope_prd_evaluation",
         ("nemo_literal", "density_roundtrip")),
        (gm_redi_slope_metric_evaluation,
         "gm_redi_slope_metric_evaluation", ("nemo_reciprocal", "division")),
        (gm_redi_slope_face_thickness_evaluation,
         "gm_redi_slope_face_thickness_evaluation",
         ("nemo_qco_live", "static_face")),
        (gm_redi_slope_depth_evaluation, "gm_redi_slope_depth_evaluation",
         ("nemo_qco_live_literal", "legacy_jacobian_t_surface")),
    ):
        if value is not None:
            if value not in choices:
                raise ValueError(f"{field} must be one of {choices}; got {value!r}")
            cfg = dataclasses.replace(cfg, **{field: value})
    if tke_langmuir_evaluation is not None:
        if tke_langmuir_evaluation not in ("nemo_literal", "vectorized"):
            raise ValueError(
                "tke_langmuir_evaluation must be 'nemo_literal' or "
                "'vectorized'")
        cfg = dataclasses.replace(
            cfg, tke_langmuir_evaluation=tke_langmuir_evaluation)
    if tke_shear_evaluation_stage is not None:
        if tke_shear_evaluation_stage not in (
                "step_entry", "implicit_solve_state"):
            raise ValueError(
                "tke_shear_evaluation_stage must be 'step_entry' or "
                "'implicit_solve_state'")
        cfg = dataclasses.replace(
            cfg, tke_shear_evaluation_stage=tke_shear_evaluation_stage)
    if tke_shear_metric_source is not None:
        if tke_shear_metric_source not in (
                "nemo_qco_live_face", "tpoint_jacobian"):
            raise ValueError(
                "tke_shear_metric_source must be 'nemo_qco_live_face' or "
                "'tpoint_jacobian'")
        cfg = dataclasses.replace(
            cfg, tke_shear_metric_source=tke_shear_metric_source)
    if tke_n2_evaluation_stage is not None:
        if tke_n2_evaluation_stage not in (
                "step_entry", "implicit_solve_state"):
            raise ValueError(
                "tke_n2_evaluation_stage must be 'step_entry' or "
                "'implicit_solve_state'")
        cfg = dataclasses.replace(
            cfg, tke_n2_evaluation_stage=tke_n2_evaluation_stage)
    if dino_wind_profile_evaluation is not None:
        if dino_wind_profile_evaluation not in (
                "nemo_literal", "factored_smoothstep"):
            raise ValueError(
                "dino_wind_profile_evaluation must be 'nemo_literal' or "
                "'factored_smoothstep'")
        cfg = dataclasses.replace(
            cfg, dino_wind_profile_evaluation=dino_wind_profile_evaluation)
    if u_m is not None:
        if not (u_m > 0.0):
            raise ValueError(f"u_m (rn_Uv) must be > 0, got {u_m!r}")
        cfg = dataclasses.replace(cfg, U_M=float(u_m))
    _em = os.environ.get("DINO_EEN_METRIC")
    if _em:
        # #1455: NEMO's vor_een weights the meridional transport by e1v and
        # divides the u-tendency by e1u (dynvor.F90:791-792, :804); legoESM's
        # AL81 triad uses neither. "nemo" selects NEMO's form. The card leaves
        # it "off" while its wall-row consequence is disputed, so this is the
        # arm switch for the controlled A/B.
        if _em not in ("off", "nemo"):
            raise ValueError(
                f"DINO_EEN_METRIC must be 'off' or 'nemo', got {_em!r}")
        cfg = dataclasses.replace(cfg, een_metric_weighting=_em)
        print(f"ARM: een_metric_weighting={_em}")
    print(f"resolved EEN metric weighting={cfg.een_metric_weighting}", flush=True)
    _ba = os.environ.get("DINO_BOLUS_ADV")
    if _ba:
        # #1226: "through_fct" folds the GM bolus into the ADVECTING MASS FLUX;
        # "centred" applies it as a tendency instead. On NEMO's true vertical
        # grid the bolus transport is 37% larger, so this isolates whether the
        # instability arrives through the advecting velocity.
        cfg = dataclasses.replace(cfg, gm_bolus_advection=_ba)
        print(f"ABLATION: gm_bolus_advection={_ba}")
    _rt = os.environ.get("DINO_RECONCILE_TARGET")
    if _rt:
        # NEMO dyn_spg_ts N6 (dynspg_ts.F90:1170): which time-averaged barotropic
        # mean the 3-D momentum depth-mean is reconciled onto ("velocity_avg" =
        # primary boxcar; "transport_avg" = un_adv/hu secondary/transport mean).
        # The confirmation A/B for the barotropic-reconcile term: one variable.
        cfg = dataclasses.replace(cfg, barotropic_reconcile_target=_rt)
        print(f"ABLATION: barotropic_reconcile_target={_rt}")
    _ar = os.environ.get("DINO_AFTER_RECONCILE")
    if _ar:
        # NEMO mlf_baro_corr (cfgs/DINO/MY_SRC/stpmlf.F90:754-765): the SECOND
        # depth-mean reconciliation, run after dyn_zdf on the committed AFTER
        # level. "nemo_mlf_baro_corr" is what NEMO does.
        #
        # CORRECTED 2026-08-21 (#1455 R6): "off" is NO LONGER the card default.
        # nemo_dino_kamm_mlf now SHIPS the faithful pair, so the twin's default
        # composition is velocity_avg + nemo_mlf_baro_corr. To reproduce any
        # arm recorded BEFORE that flip, set BOTH:
        #     DINO_RECONCILE_TARGET=transport_avg DINO_AFTER_RECONCILE=off
        #
        # CORRECTED 2026-08-21: an earlier version of this comment called it
        # "orthogonal to DINO_RECONCILE_TARGET" and said "setting both is two
        # variables and the A/B must not". BOTH HALVES WERE WRONG, and the
        # second was an instruction against running the only faithful
        # configuration. They COMPOSE: NEMO commits uu_b(Kaa), its PRIMARY
        # velocity-weighted boxcar, so matching NEMO needs
        # DINO_RECONCILE_TARGET=velocity_avg AND this set to
        # nemo_mlf_baro_corr. Setting both is the FAITHFUL arm of a 2x2, not a
        # two-variable mistake -- see PHASE2_R6_alignment_and_prereg.md.
        cfg = dataclasses.replace(cfg, barotropic_after_reconcile=_ar)
        print(f"ABLATION: barotropic_after_reconcile={_ar}")

    # CRITICAL: st MUST be the NEMO-restart-carrying bridged state (br.state)
    # -- NOT dino_lat_lon_state(...) (the analytic paper-IC rest state), which
    # would silently spin up from rest instead of twinning the restart. See
    # the 2026-07-24 defect note in the module docstring.
    st = br.state
    print("twin from developed NEMO state (br.state, NOT dino_lat_lon_state)")
    verify_day0_matches_restart(st, s, br.land_mask)

    # DEFAULT ON (#1455, 2026-08-24): bridge NEMO's leap-frog BEFORE-level
    # state (tb/sb/ub/vb, the MLF integrator's THIRD time level) onto
    # state.{T,S,u,v,eta}_before, so the twin's leap-frog entry state is
    # EXACTLY NEMO's -- not a forward-Euler cold start. Skipping it delays the
    # trajectory about half a step AND injects a permanent perturbation of half
    # a leap-frog step of tendency (~2.06e-3 Sv of circumpolar transport on this
    # restart), and skips NEMO's step-1 after-level reconciliation; see
    # resolve_start_mode for the three terms and their receipts. Every direct
    # caller of this helper already passed bridge_before=True explicitly, so
    # the flipped default changes NO recorded sibling probe -- unlike the
    # ladder default, which is deliberately NOT resolved here for that reason.
    # nemo_dino_kamm_mlf sets tke_n2_time_level="nemo_before" +
    # tke_shear_production="nemo_burchard", both of which READ these fields
    # every step -- bridging is what makes those axes correct from step 0
    # instead of only after the model's own Euler-start populates them.
    restart_path = f"{run_stepdump}/{restart_file}"
    if bridge_before:
        before = read_nemo_restart_before(restart_path, nn_hls=0)
        st = bridge_before_state_topo(br._replace(state=st), g, before, periodic_i=True)
        _print_before_bridge_verify(st, before, g)
        if bridge_before_stress_tpoint:
            prior_time = restart_elapsed_seconds(restart_path)
            st, _ = reconstruct_dino_before_stress_tpoint(
                st, br.geometry, cfg, t_seconds=prior_time)

    # OPTIONAL: bridge NEMO's developed TKE closure memory (`en`) onto lego's
    # cold-start `state.tke` -- isolates whether the TKE cold-start (vs the
    # bridged prognostic T/S/eta/u/v) drives the day 0-4 surface-layer
    # handshake divergence (#1317). Off by default (matches the module
    # docstring's documented cold-start caveat) so `--bridge-tke` is additive,
    # not a silent behavior change.
    if bridge_tke:
        en_restart = read_nemo_restart_en(f"{run_stepdump}/{restart_file}", nn_hls=0)
        _carry_coeffs = (cfg.tke_preclosure_coeff_source
                         == "carried_previous_step")
        if _carry_coeffs:
            avm_restart, avt_restart, dissl_restart = (
                read_nemo_restart_tke_coefficients(
                f"{run_stepdump}/{restart_file}", nn_hls=0)
            )
        else:
            avm_restart = avt_restart = dissl_restart = None
        st = bridge_tke_from_restart(
            st, en_restart, br.land_mask,
            restart_avm=avm_restart, restart_avt=avt_restart,
            restart_dissl=dissl_restart)
        wet = np.asarray(br.land_mask) > 0.5
        d_en = float(np.max(np.abs(
            np.asarray(st.tke.data)[wet] - en_restart[..., 1:][wet])))
        print(f"TKE BRIDGE: seeded state.tke from NEMO restart en "
              f"(w-level 1..{en_restart.shape[-1]-1} -> interior interface "
              f"0..{en_restart.shape[-1]-2})  max|d_en|={d_en:.3e}", flush=True)
        if _carry_coeffs:
            d_avm = float(np.max(np.abs(
                np.asarray(st.tke_avm.data)[wet] - avm_restart[..., 1:][wet])))
            d_avt = float(np.max(np.abs(
                np.asarray(st.tke_avt.data)[wet] - avt_restart[..., 1:][wet])))
            d_dissl = float(np.max(np.abs(
                np.asarray(st.tke_dissl.data)[wet]
                - dissl_restart[..., 1:][wet])))
            print("TKE COEFFICIENT BRIDGE: restart avm_k/avt_k/dissl -> carried "
                  f"closure pair max|d_avm|={d_avm:.3e} "
                  f"max|d_avt|={d_avt:.3e} max|d_dissl|={d_dissl:.3e}",
                  flush=True)

    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    if os.environ.get("DINO_NEMO_KMM_DIVISOR") is not None:
        raise SystemExit(
            "DINO_NEMO_KMM_DIVISOR is GONE. NEMO's e3w(Kmm) implicit-solve "
            "divisor (trazdf.F90:219-221, dynzdf.F90:200-203) is no longer a "
            "flag: it is unbranched inside the NEMO identity "
            'zdf_implicit_solver_evaluation="nemo_literal", which this card '
            "already selects. Compare across commits, not across this knob "
            "(docs/ocean/fidelity/dino_zdf_divisor_arm_receipt.md).")
    # #1492 P2: NEMO-faithful step-composition A/B (docs/ocean/fidelity/
    # nemo_mlf_step_transcription_spec.md resolved decision 2). Default "" =
    # legacy (outer_integrator="leapfrog", unchanged); "nemo_mlf" routes to
    # the single-pass stpmlf.F90 transcription via the REAL dispatch. Opt-in
    # measurement knob only -- NOT a recipe/kamm default (same pattern as
    # dino_year_screen_fullframe.py). Unknown value raises.
    _OI = os.environ.get("DINO_OUTER_INTEGRATOR", "")
    if _OI:
        if _OI not in ("leapfrog", "nemo_mlf"):
            raise SystemExit(
                f"Unknown DINO_OUTER_INTEGRATOR={_OI!r}: expected "
                "'leapfrog' or 'nemo_mlf'")
        # nemo_mlf HARD-REQUIRES the NEMO implicit-ZDF identity at
        # construction (spec resolved decision 4; it carries NEMO's e3w(Kmm)
        # divisor) -- auto-force it so the env knob alone is sufficient.
        mc = mc._replace(
            outer_integrator=_OI,
            zdf_implicit_solver_evaluation=(
                "nemo_literal" if _OI == "nemo_mlf"
                else mc.zdf_implicit_solver_evaluation))
        print(f"ABLATION: outer_integrator={mc.outer_integrator} "
              "zdf_implicit_solver_evaluation="
              f"{mc.zdf_implicit_solver_evaluation}")
    print(f"barotropic_diffusion_alpha={mc.barotropic.barotropic_diffusion_alpha} "
          f"barotropic_face_depth={mc.barotropic.barotropic_face_depth} "
          f"zdf_drag_in_matrix={mc.zdf_drag_in_matrix} "
          f"zdf_baroclinic_only={mc.zdf_baroclinic_only} "
          f"barotropic_drag_substep={mc.barotropic_drag_substep}", flush=True)

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    # NEMO evaluates the analytic wind at its stored gphiu operand.  Keep that
    # raw degree-valued mesh field through the bridge: radians->degrees would
    # perturb 154 row-8 columns beyond the registered 1e-15 bar.
    forcing = dino_lat_lon_surface_forcing_arrays(
        br.geometry, cfg, wind_lat_deg=g.gphit[:, 0])
    sf = dino_step_surface_forcing(forcing)
    print(f"slope_scheme={mc.gm_redi.slope_scheme} "
          f"kappa_GM_max={float(jnp.max(jnp.abs(mc.gm_redi.kappa_GM))):.1f}")
    print(f"tau_x[Pa] min/max = {float(jnp.min(sf.tau_x)):.3f}/{float(jnp.max(sf.tau_x)):.3f}")
    return br, cfg, mc, model, forcing, sf, st


def run_twin(recipe: str, out_path: str, *, n_days: int = 90, save_3d: bool = False,
             run_traj: str = RUN_TRAJ, run_stepdump: str = RUN_STEPDUMP,
             bridge_tke: bool = False, bridge_before: bool = True,
             bridge_before_stress_tpoint: bool = True,
             vmix_scheme: str | None = None,
             use_gm_redi: bool | None = None,
             surface_stress_implicit: bool | None = None,
             surface_tendency_placement: str | None = None,
             barotropic_continuity_evaluation: str | None = None,
             vface_zonal_metric_evaluation: str | None = None,
             tke_preclosure_coeff_source: str | None = None,
             tke_matrix_evaluation: str | None = None,
             tke_solver_evaluation: str | None = None,
             zdf_implicit_solver_evaluation: str | None = None,
             tke_etau_exponential_evaluation: str | None = None,
             tke_htau_evaluation: str | None = None,
             tke_mxl_raw_evaluation: str | None = None,
             tke_langmuir_evaluation: str | None = None,
             tke_shear_evaluation_stage: str | None = None,
             tke_shear_metric_source: str | None = None,
             tke_n2_evaluation_stage: str | None = None,
             dino_wind_profile_evaluation: str | None = None,
             gm_redi_slope_n2_evaluation: str | None = None,
             gm_redi_slope_prd_evaluation: str | None = None,
             gm_redi_slope_metric_evaluation: str | None = None,
             gm_redi_slope_face_thickness_evaluation: str | None = None,
             gm_redi_slope_depth_evaluation: str | None = None,
             u_m: float | None = None,
             restart_file: str = RESTART_FILE,
             perturb_seed: int | None = None,
             perturb_eps: float = PERTURB_EPS_DEFAULT,
             perturb_baro: str | None = None,
             perturb_baro_key: str = "dU_avg",
             perturb_baro_scale: float = 1.0,
             daily_acc: bool = False,
             save_step_eta: bool = False,
             snap_days: tuple[int, ...] | None = None,
             fp64_3d: bool = False,
             legacy_1d_ladder: bool = False,
             bridge_omega: str = "nemo") -> bool:
    """Run the state-initialized twin for ``n_days`` and save an npz. Returns stable.

    ``perturb_seed``: optional #1492 item-2.2 noise-control lane -- if set,
    applies a 1e-14-relative multiplicative perturbation to the bridged
    now-level T (``T *= 1 + 1e-14 * N(0,1)`` per grid point,
    ``numpy.random.default_rng(perturb_seed)``), mirroring
    ``scripts/tmp/_perturb_restart_ensemble.py``'s documented ensemble
    pattern (same eps, same draw shape) but applied post-bridge to the
    legoESM state's now-level T only.  NOTE (#1455, 2026-08-24): the
    justification for perturbing only the now level used to be that the
    default state carries no before-level T.  That is no longer true -- the
    default now bridges it -- so the now-only perturbation is a deliberate
    (and unchanged, so every recorded ensemble stays comparable) choice, not a
    forced one.  A member's tb and tn therefore differ by the kick at step 0.
    Whether that floor is the same size as the old one is a MEASUREMENT, not
    an inference, and it is not claimed here -- but it is also not a live
    concern: ``floor90_ensemble.py`` already passed the bridge explicitly, so
    the recorded floor that sets this gate's thresholds was built this way.

    ``perturb_baro``/``perturb_baro_key``/``perturb_baro_scale``: #1455
    Phase-2 Measurement 1.  Adds ``scale *`` a depth-uniform barotropic
    velocity field, once at t=0, to the now-level u.  The field is the
    per-step barotropic deposit measured by ``substep_traj_compare.py`` and
    saved by its ``DINO_1455_DEPOSIT_MAP`` block, so the injected section
    transport at t=0+ IS that deposit and the response divided by it is the
    retention factor -- the quantity whose absence voided every "x times the
    budget row" ratio this campaign published (eb3f6d23d).

    ``daily_acc``: store the ACC transport every day under two reducers (the
    deposit's own mean-over-longitudes one and the recorded gate's median
    one).  Off by default; adds two float64 arrays to the npz.

    ``snap_days``: which days get a full 3-D T/S/eta/u/v snapshot under
    ``save_3d``.  ``None`` keeps the recorded ``SNAP_DAYS`` grid so every
    artifact produced before this argument existed is reproduced exactly;
    days past ``n_days`` are dropped either way.  #1455 verdict360 needs a
    10-day grid out to a year so an ensemble spread can be read at the same
    days NEMO's own ``nn_stock`` restarts land on -- the alternative,
    growing the module-level tuple, would silently change every sibling
    probe's artifact.

    ``save_step_eta``: replace the default daily/float32 ``eta`` payload with
    every-step/float64 eta plus relative ``t_seconds`` for the campaign's
    Nyquist-safe 2dt scorer. The daily field remains under ``eta_daily``.

    ``bridge_before_stress_tpoint`` defaults on and reconstructs only the
    initial T-point previous-stress carry. ``False`` reproduces the historical
    U-as-T bridge defect. Later-step carry behavior remains the model's
    ordinary ``_seed_centred_forcing_carry`` path.
    """
    _producer_sha_entry, _producer_dirty_entry = _git_provenance()
    if (_producer_dirty_entry
            and os.environ.get("LEGOESM_ALLOW_DIRTY") != "1"):
        raise SystemExit("REFUSING run_twin from a dirty tracked tree; the "
                         "artifact producer identity would be ambiguous")
    # Resolved here and handed DOWN as an argument -- nothing is written into the
    # environment, so two ladders can be built in one process without either
    # inheriting the other's setting.
    ladder_mode = resolve_ladder_mode(legacy_1d_ladder)
    requested_start = resolve_start_mode(bridge_before)
    bridge_omega_value, _ = resolve_bridge_omega(bridge_omega)
    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        recipe, run_traj, run_stepdump, bridge_tke=bridge_tke,
        bridge_before=bridge_before, vmix_scheme=vmix_scheme,
        bridge_before_stress_tpoint=bridge_before_stress_tpoint,
        use_gm_redi=use_gm_redi, restart_file=restart_file,
        surface_stress_implicit=surface_stress_implicit,
        surface_tendency_placement=surface_tendency_placement,
        barotropic_continuity_evaluation=barotropic_continuity_evaluation,
        vface_zonal_metric_evaluation=vface_zonal_metric_evaluation,
        tke_preclosure_coeff_source=tke_preclosure_coeff_source,
        tke_matrix_evaluation=tke_matrix_evaluation,
        tke_solver_evaluation=tke_solver_evaluation,
        zdf_implicit_solver_evaluation=zdf_implicit_solver_evaluation,
        tke_etau_exponential_evaluation=tke_etau_exponential_evaluation,
        tke_htau_evaluation=tke_htau_evaluation,
        tke_mxl_raw_evaluation=tke_mxl_raw_evaluation,
        tke_langmuir_evaluation=tke_langmuir_evaluation,
        tke_shear_evaluation_stage=tke_shear_evaluation_stage,
        tke_shear_metric_source=tke_shear_metric_source,
        tke_n2_evaluation_stage=tke_n2_evaluation_stage,
        dino_wind_profile_evaluation=dino_wind_profile_evaluation,
        gm_redi_slope_n2_evaluation=gm_redi_slope_n2_evaluation,
        gm_redi_slope_prd_evaluation=gm_redi_slope_prd_evaluation,
        gm_redi_slope_metric_evaluation=gm_redi_slope_metric_evaluation,
        gm_redi_slope_face_thickness_evaluation=(
            gm_redi_slope_face_thickness_evaluation),
        gm_redi_slope_depth_evaluation=gm_redi_slope_depth_evaluation,
        u_m=u_m, e3t_mode=ladder_mode, bridge_omega=bridge_omega)

    # #1455 review: the stamp must be a RECEIPT, not a restatement of the flag.
    # ``resolve_start_mode`` reads the CLI; what actually seeds the before level
    # is ``_build_twin_state`` one call later, and nothing linked the two -- so
    # an artifact could have said "bridged" for a run that was not, exactly the
    # label-vs-identity failure ``vertical_ladder_sha256`` exists to close for
    # the ladder. This OBSERVES the built state instead, and refuses a
    # disagreement rather than stamping either answer.
    #
    # SCOPE, stated so the next reader does not over-read it: this is a
    # STRUCTURAL receipt (are the before-level fields populated at all), not a
    # VALUE receipt. The value receipt -- max|d_tb| against NEMO's own tb -- is
    # computed and printed by ``_print_before_bridge_verify`` inside
    # ``_build_twin_state``; carrying it out to here would change that helper's
    # return arity at fifteen sibling call sites, which is not worth it for a
    # number that is already in every bridged run's log.
    observed_start = "bridged" if st.T_before is not None else "euler"
    if observed_start != requested_start:
        raise SystemExit(
            f"START-MODE MISMATCH: resolve_start_mode said "
            f"{requested_start!r} but the built state is {observed_start!r} "
            f"(state.T_before is "
            f"{'populated' if st.T_before is not None else 'None'}). Refusing "
            "to stamp either answer -- the twin's start mode and the state it "
            "actually starts from must agree.")
    start_mode = observed_start

    # Observe the built geometry and model config only after the pre-existing
    # start-mode receipt has passed. This preserves that gate's priority while
    # making the new selector a content receipt rather than a CLI restatement.
    built_bridge_omega = float(br.geometry.omega)
    config_omega = float(mc.constants.Omega)
    if built_bridge_omega != float(bridge_omega_value):
        raise SystemExit(
            f"bridge Omega receipt mismatch: built={built_bridge_omega:.17g} "
            f"selected={bridge_omega_value:.17g}")
    if config_omega != float(NEMO_CONSTANTS_CONFIG.Omega):
        raise SystemExit(
            "bridge-Omega arm changed model-config Omega: "
            f"{config_omega:.17g} != NEMO {NEMO_CONSTANTS_CONFIG.Omega:.17g}")
    bridge_f_t_sha256 = _array_content_sha256("f_T", br.geometry.f_T)
    bridge_f_u_sha256 = _array_content_sha256("f_u", br.geometry.f_u)
    bridge_f_v_sha256 = _array_content_sha256("f_v", br.geometry.f_v)
    print("BRIDGE OMEGA RECEIPT: "
          f"mode={bridge_omega} bridge={built_bridge_omega:.17g} "
          f"config={config_omega:.17g} f_T_sha256={bridge_f_t_sha256} "
          f"f_u_sha256={bridge_f_u_sha256} "
          f"f_v_sha256={bridge_f_v_sha256}", flush=True)

    # Receipt, not a restatement of the selector: reconstruct the expected
    # analytic T field through the same loaders and compare it to the carry
    # actually present on the built state before stamping "T".
    restart_path = f"{run_stepdump}/{restart_file}"
    if bridge_before_stress_tpoint:
        reconstruction_time = restart_elapsed_seconds(restart_path)
        expected_x, expected_y, expected_hash = _analytic_dino_tpoint_stress(
            br.geometry, cfg, t_seconds=reconstruction_time)
        if (not np.array_equal(np.asarray(st.tau_x_prev), np.asarray(expected_x))
                or not np.array_equal(
                    np.asarray(st.tau_y_prev), np.asarray(expected_y))):
            raise SystemExit(
                "BEFORE-STRESS STAMP MISMATCH: selector requested T-point "
                "reconstruction but the built carry differs from the "
                "existing DINO analytic forcing loader")
        bridge_before_stress_stagger = "T"
        bridge_before_stress_sha256 = expected_hash
    else:
        reconstruction_time = float("nan")
        bridge_before_stress_stagger = (
            "U_AS_T_LEGACY" if bridge_before else "NONE")
        bridge_before_stress_sha256 = (
            _stress_content_sha256(st.tau_x_prev, st.tau_y_prev)
            if st.tau_x_prev is not None and st.tau_y_prev is not None else "")
    print("BEFORE-STRESS RECEIPT: "
          f"stagger={bridge_before_stress_stagger} "
          f"reconstruction_seconds={reconstruction_time} "
          f"sha256={bridge_before_stress_sha256}", flush=True)

    # #1455 512517fdc + review a0cd04b8: the BINDING precision check, on the
    # materialized geometry and state (arrays cannot lie about their dtype the
    # way the policy global can). Shared gate, not a re-derivation.
    from legoesm.ocean.fidelity.precision_gate import require_fp64
    if _fp64_requested():
        require_fp64(br.geometry, st, context="kamm_twin_90d oracle twin")
    control_dtype_stamp = str(np.asarray(st.T.data).dtype)
    print(f"PRECISION: materialized state dtype = {control_dtype_stamp}")

    if perturb_seed is not None:
        rng = np.random.default_rng(perturb_seed)
        t0 = np.asarray(st.T.data, dtype=np.float64)
        factor = 1.0 + perturb_eps * rng.standard_normal(t0.shape)
        t_pert = jnp.asarray(t0 * factor, dtype=st.T.data.dtype)
        d_t = float(np.max(np.abs(np.asarray(t_pert) - t0)))
        rel = float(np.max(np.abs(np.asarray(t_pert) - t0) / np.maximum(np.abs(t0), 1e-30)))
        print(f"PERTURB seed={perturb_seed} eps={perturb_eps:.3e}: "
              f"max|dT|={d_t:.3e}  max_rel|dT/T|={rel:.3e}", flush=True)
        st = st._replace(T=st.T.replace(data=t_pert))

    initial_state_sha256 = _initial_state_sha256(st)

    nsteps = STEPS_PER_DAY * n_days

    # #1492: "leapfrog_rhs" placement REQUIRES return_rate=True + threading
    # the rate into model.step(external_tracer_rate=...) (run_dino.py's
    # driver wiring, mirrored; _check_surface_tendency_placement raises on a
    # mismatch, so the legacy applied_now loop cannot silently run a
    # leapfrog_rhs config).
    _sf_placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    if _sf_placement == "leapfrog_rhs":
        dyn = jax.jit(lambda st, ext: model.step(st, DT, surface_forcing=sf,
                                                  external_tracer_rate=ext))
    else:
        dyn = jax.jit(lambda st: model.step(st, DT, surface_forcing=sf))

    # #1455 SEASONAL CLOCK. Absolute (NEMO's own day-of-year, read from the
    # bridged restart) by DEFAULT; see seasonal_t0_seconds() for the override.
    t0_sec = seasonal_t0_seconds(f"{run_stepdump}/{restart_file}")
    # The clock the NEMO run this twin is scored against is actually on, read
    # from the same restart.  Stamping it next to the clock the twin USED lets
    # a scorer reject ANY offset that is not NEMO's, not merely t0=0.
    t0_reference_sec = restart_elapsed_seconds(f"{run_stepdump}/{restart_file}")
    resolved_snap_days = resolve_snap_days(snap_days, n_days, save_3d)
    # Everything about this run a comparison must hold fixed.  A two-arm A/B
    # that changes the clock and something else is a confound, and nothing in
    # the artifact could see it before this stamp existed.
    run_config = json.dumps({
        "recipe": recipe, "n_days": int(n_days),
        "run_traj": run_traj, "run_stepdump": run_stepdump,
        "restart_file": restart_file,
        "bridge_tke": bool(bridge_tke), "bridge_before": bool(bridge_before),
        "bridge_before_stress_tpoint": bool(bridge_before_stress_tpoint),
        "bridge_omega": bridge_omega,
        "vmix_scheme": vmix_scheme, "use_gm_redi": use_gm_redi,
        "surface_stress_implicit": surface_stress_implicit,
        "surface_tendency_placement": surface_tendency_placement,
        "barotropic_continuity_evaluation":
            cfg.barotropic_continuity_evaluation,
        "vface_zonal_metric_evaluation":
            cfg.vface_zonal_metric_evaluation,
        "save_3d": bool(save_3d),
        "snap_days": list(resolved_snap_days),
        "save_step_eta": bool(save_step_eta),
        # Resolved production selectors, not merely the optional CLI
        # overrides.  These receipts make the faithful and legacy climate
        # arms distinguishable even when both are launched from defaults.
        "tke_preclosure_coeff_source": cfg.tke_preclosure_coeff_source,
        "tke_matrix_evaluation": cfg.tke_matrix_evaluation,
        "tke_solver_evaluation": cfg.tke_solver_evaluation,
        "zdf_implicit_solver_evaluation":
            cfg.zdf_implicit_solver_evaluation,
        "tke_etau_exponential_evaluation":
            cfg.tke_etau_exponential_evaluation,
        "tke_htau_evaluation": cfg.tke_htau_evaluation,
        "tke_mxl_raw_evaluation": cfg.tke_mxl_raw_evaluation,
        "tke_langmuir_evaluation": cfg.tke_langmuir_evaluation,
        "tke_shear_evaluation_stage": cfg.tke_shear_evaluation_stage,
        "tke_shear_metric_source": cfg.tke_shear_metric_source,
        "tke_n2_evaluation_stage": cfg.tke_n2_evaluation_stage,
        "dino_wind_profile_evaluation":
            cfg.dino_wind_profile_evaluation,
        "gm_redi_slope_n2_evaluation": cfg.gm_redi_slope_n2_evaluation,
        "gm_redi_slope_prd_geometry_stage":
            cfg.gm_redi_slope_prd_geometry_stage,
        "gm_redi_slope_prd_evaluation": cfg.gm_redi_slope_prd_evaluation,
        "gm_redi_slope_metric_evaluation": cfg.gm_redi_slope_metric_evaluation,
        "gm_redi_slope_face_thickness_evaluation":
            cfg.gm_redi_slope_face_thickness_evaluation,
        "gm_redi_flux_face_thickness_evaluation":
            cfg.gm_redi_flux_face_thickness_evaluation,
        "gm_redi_horizontal_evaluation": cfg.gm_redi_horizontal_evaluation,
        "gm_redi_vertical_skew_evaluation":
            cfg.gm_redi_vertical_skew_evaluation,
        "gm_redi_a33_evaluation": cfg.gm_redi_a33_evaluation,
        "gm_redi_w_slope_stage_evaluation":
            cfg.gm_redi_w_slope_stage_evaluation,
        "gm_redi_slope_depth_evaluation": cfg.gm_redi_slope_depth_evaluation,
        "gm_treguier_vertical_reduction_evaluation":
            cfg.gm_treguier_vertical_reduction_evaluation,
        "gm_treguier_sqrt_evaluation": cfg.gm_treguier_sqrt_evaluation,
        "barotropic_transport_accumulation_evaluation":
            cfg.barotropic_transport_accumulation_evaluation,
        "barotropic_seed_evaluation": cfg.barotropic_seed_evaluation,
        "barotropic_een_coefficient_evaluation":
            cfg.barotropic_een_coefficient_evaluation,
        "barotropic_pgf_evaluation": cfg.barotropic_pgf_evaluation,
        "zad_qco_evaluation": cfg.zad_qco_evaluation,
        "wzv_call2_evaluation": cfg.wzv_call2_evaluation,
        "perturb_seed": perturb_seed, "perturb_eps": float(perturb_eps),
        "perturb_baro": perturb_baro,
        "perturb_baro_sha256": _file_content_sha256(perturb_baro),
        "perturb_baro_key": perturb_baro_key,
        "perturb_baro_scale": float(perturb_baro_scale),
        "daily_acc": bool(daily_acc),
        "u_m": u_m,
        # DELIBERATELY NOT recorded here: --fp64-3d. run_config is compared
        # BYTE-FOR-BYTE between two arms by twin_seasonal_clock_ab.py, which
        # hard-aborts on any difference as a confound; a new key would make
        # every recorded-arm-vs-new-arm A/B abort forever (code review). The
        # flag changes STORAGE only, never the trajectory, and it is recorded
        # in the storage_dtypes stamp where a storage fact belongs.
    }, sort_keys=True)

    land_mask = np.asarray(st.land_mask.data)
    n_lat, n_lon = br.geometry.n_lat, br.geometry.n_lon

    # #1455 PHASE-2: the ACC transport reducer, built ONCE and used for both
    # the daily readout and the injected-transport stamp.  Two reducers here
    # would be two chances for a staggering drift, which is the defect class
    # this campaign has spent the most time on.
    #
    # Two reductions are stored:
    #   acc_dep  -- the DEPOSIT's own (full 2-D e2u, e3t_1d ladder, rows
    #               1..197, MEAN over longitudes 2..-2).  R(t) is a ratio of
    #               this functional to itself, so no cross-metric staggering
    #               enters the retention number.
    #   acc_gate -- acc_thermal_wind.acc_full, the recorded gate metric
    #               (MEDIAN over longitudes), for context against the gap.
    _acc_dep_daily = _acc_gate_daily = None
    _acc_pair = None
    _injected_sv = 0.0
    if daily_acc or perturb_baro is not None:
        import acc_thermal_wind as _A
        import netCDF4 as _nc
        _mmp = f"{run_traj}/mesh_mask.nc"
        # The gate reducer loads its OWN mesh at import time from a hardcoded
        # path.  If that is not the mesh this run was built on, the two
        # reductions describe different geometries -- checked, not assumed.
        if os.path.abspath(getattr(_A, "mm", None).filepath()) != os.path.abspath(_mmp):
            raise SystemExit(
                f"FATAL: acc_thermal_wind loaded {_A.mm.filepath()} but this "
                f"run uses {_mmp}; the two reducers would describe different "
                "geometries")
        _mmd = _nc.Dataset(_mmp)
        _e2u2d = np.asarray(_mmd.variables["e2u"][0]).squeeze()          # (j,i)
        _e3t1d = np.asarray(_mmd.variables["e3t_1d"][:]).squeeze()       # (k,)
        _um3 = np.moveaxis(np.asarray(_mmd.variables["umask"][0]).squeeze(),
                           0, -1) > 0.5                                  # (j,i,k)
        _mmd.close()

        def _acc_pair(u_full):
            """(deposit-reducer Sv, gate-metric Sv) from the model's u faces."""
            u = np.asarray(u_full, dtype=np.float64)[:, 1:53, :]
            u = np.where(_um3, u, 0.0)
            # rows 1..197 then MEAN over longitudes 2..-2, matching acc_sv
            col = np.einsum("jik,k,ji->ji", u, _e3t1d, _e2u2d)[1:198, :].sum(axis=0)
            dep = float(col[2:-2].mean()) / 1.0e6
            # NOTE: acc_thermal_wind.load_lego carries a line
            #   u[:, 47, :] = lU[:, 48, :]
            # which is a NO-OP after the [:, 1:53] slice (index 47 already IS
            # original column 48), traced back to an editing leftover in
            # compare_fullframe.py.  It is deliberately NOT reproduced here.
            gate = _A.acc_full(u, _A.umask)
            return dep, gate

    if daily_acc:
        _acc_dep_daily = np.full(n_days, np.nan, dtype=np.float64)
        _acc_gate_daily = np.full(n_days, np.nan, dtype=np.float64)
        print(f"daily ACC enabled (mesh {run_traj}/mesh_mask.nc)", flush=True)

    # #1455 PHASE-2 MEASUREMENT 1, the RETENTION FACTOR.  The per-step deposit
    # is an INJECTION; the ACC gap is an ACCUMULATION, and converting one to
    # the other needs the trajectory's retention of a one-step injection --
    # never measured on this campaign, and the thing that voided every
    # "x times the budget row" ratio (89dadfeb4, eb3f6d23d).
    #
    # This injects the MEASURED deposit's own barotropic velocity field, once,
    # at t=0, on top of an otherwise byte-identical free run.  The field is
    # depth-UNIFORM, so it adds exactly that barotropic increment and zero
    # baroclinic shear, and its section transport at t=0+ is the deposit
    # itself.  Column 0 of lego's u is the periodic wrap of NEMO's last column
    # -- the same mapping the forcing substitution established.
    if perturb_baro is not None:
        _pb = np.load(perturb_baro, allow_pickle=True)
        _dU = np.asarray(_pb[perturb_baro_key], dtype=np.float64)   # (jpj,jpi)
        _u0 = np.asarray(st.u.data, dtype=np.float64)               # (jpj,jpi+1,nz)
        if _dU.shape != (_u0.shape[0], _u0.shape[1] - 1):
            raise SystemExit(
                f"FATAL: perturbation {_dU.shape} does not match the u grid "
                f"{_u0.shape}; refusing to broadcast a guess")
        if not np.all(np.isfinite(_dU)):
            raise SystemExit("FATAL: non-finite values in the perturbation field")
        _add = np.zeros_like(_u0)
        _add[:, 1:, :] = (perturb_baro_scale * _dU)[:, :, None]
        _add[:, 0, :] = perturb_baro_scale * _dU[:, -1][:, None]
        # A dry u-cell must stay exactly 0.  dU_avg is already masked by NEMO's
        # surface umask, but the column below the bathymetry is not, so mask
        # against the model's own 3-D wet u-faces rather than a rebuilt guess.
        import netCDF4 as _nc0
        _mm0 = _nc0.Dataset(f"{run_traj}/mesh_mask.nc")
        _umask3 = np.moveaxis(
            np.asarray(_mm0.variables["umask"][0]).squeeze(), 0, -1) > 0.5
        _mm0.close()
        _add[:, 1:, :] = np.where(_umask3, _add[:, 1:, :], 0.0)
        _add[:, 0, :] = np.where(_umask3[:, -1, :], _add[:, 0, :], 0.0)
        _u_pert = jnp.asarray(_u0 + _add, dtype=st.u.data.dtype)
        # The BEFORE level gets the SAME increment.  Perturbing only the now
        # level of a leapfrog state plants a splitting (computational) mode
        # that the Asselin filter then damps over the first few steps -- which
        # would contaminate R(day 1), the load-bearing number here, with an
        # artifact of the time scheme rather than the ocean's response.  The
        # T-perturbation path documents exactly this hazard and this one
        # inherited none of it.
        if getattr(st, "u_before", None) is not None:
            _ub = np.asarray(st.u_before.data, dtype=np.float64)
            st = st._replace(u_before=st.u_before.replace(
                data=jnp.asarray(_ub + _add, dtype=st.u_before.data.dtype)))
        # THE NORMALISER, MEASURED.  Every retention factor divides by this
        # number, so it is computed from the state BEFORE and AFTER the
        # injection with the run's own reducer and STAMPED into the artifact.
        # It was previously a hand-typed default on the scoring side, which
        # would have divided an in-loop-pattern response by the TOTAL
        # deposit the moment --perturb-baro-key changed.
        _dep_before, _ = _acc_pair(_u0)
        _dep_after, _ = _acc_pair(np.asarray(_u_pert, dtype=np.float64))
        _injected_sv = _dep_after - _dep_before
        st = st._replace(u=st.u.replace(data=_u_pert))
        print(f"PERTURB-BARO {perturb_baro}:{perturb_baro_key} scale="
              f"{perturb_baro_scale:g}  max|du|={np.abs(_add).max():.4e} m/s  "
              f"nonzero={int((_add != 0).sum())}  "
              f"INJECTED TRANSPORT={_injected_sv:+.6e} Sv", flush=True)
    eta_daily = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    sst_daily = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    u_daily = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    v_daily = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    eta_step = None
    if save_step_eta:
        if control_dtype_stamp != "float64":
            raise SystemExit(
                "--save-step-eta requires a materialized float64 control "
                f"state, got {control_dtype_stamp}; casting fp32 output to "
                "float64 would forge precision")
        eta_step = np.full((nsteps, n_lat, n_lon), np.nan, dtype=np.float64)
        print(f"per-step eta enabled: {nsteps} samples at float64", flush=True)

    snaps = resolved_snap_days
    snap_np = snapshot_dtype(fp64_3d)
    snap_name = np.dtype(snap_np).name
    t3d, s3d, eta3d, u3d, v3d = {}, {}, {}, {}, {}
    # The fp64 reduced series is ALWAYS built (it is ~2 kB per snapshot day and
    # is what makes an ensemble spread measurable below the storage quantum);
    # --fp64-3d only changes the big 3-D block.
    _reducer, _reduced_status = (build_snapshot_reducer(run_traj) if snaps
                                 else (None, "no snapshot days requested"))
    _reduced = {}
    print(f"3-D snapshot storage dtype = {snap_name}; fp64 reduced series: "
          f"{_reduced_status}", flush=True)
    if snaps and _reducer is None:
        print("WARNING: no fp64 reduced series will be stored for this run "
              f"({_reduced_status}); the reason is stamped as "
              "reduced_series_status", flush=True)

    _reduce_fail = {}

    def _snap(day):
        """Store the day's 3-D block and reduce the LIVE state at fp64."""
        stored, red, why = capture_snapshot(
            {"T": st.T.data, "S": st.S.data, "eta": st.eta.data,
             "u": st.u.data, "v": st.v.data},
            snap_dtype=snap_np, reducer=_reducer, land_mask=land_mask)
        t3d[day], s3d[day] = stored["T"], stored["S"]
        eta3d[day], u3d[day], v3d[day] = stored["eta"], stored["u"], stored["v"]
        if _reducer is None:
            return
        if red is None:
            _reduce_fail[day] = why
            print(f"  WARNING day {day}: {why} -- no reduced series for this "
                  f"day; the 3-D block is unaffected", flush=True)
        else:
            _reduced[day] = red

    if save_3d:
        _snap(0)
        # _snap stores full-depth u/v faces (NOT just the surface level
        # captured by u_daily/v_daily below) -- required for ACC
        # (acc_thermal_wind.py's acc_full integrates over all NZ levels), so a
        # snapshot day's u/v must carry the whole water column, matching
        # T3d/S3d's full depth.
        print("captured day-0 3-D T/S/u/v snapshot", flush=True)

    blew_up_at = None
    t0 = time.time()
    for k in range(nsteps):
        if _sf_placement == "leapfrog_rhs":
            st, _ext_rate = apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, DT,
                t_seconds=t0_sec + (k + 1) * DT, return_rate=True)
            st = dyn(st, _ext_rate)
        else:
            st = apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, DT, t_seconds=t0_sec + (k + 1) * DT)
            st = dyn(st)

        if eta_step is not None:
            eta_now64 = np.asarray(st.eta.data, dtype=np.float64)
            if not np.isfinite(eta_now64).all():
                raise SystemExit(
                    f"FATAL: non-finite per-step eta at step {k + 1}")
            eta_step[k] = eta_now64

        if (k + 1) % STEPS_PER_DAY == 0:
            day_idx = (k + 1) // STEPS_PER_DAY - 1
            day_num = day_idx + 1
            t_now3d = np.asarray(st.T.data)
            m = land_mask > 0.5
            finite = bool(np.isfinite(t_now3d[m]).all())
            tmax = float(t_now3d[m].max()) if finite else float("nan")
            tmin = float(t_now3d[m].min()) if finite else float("nan")
            unstable = not (finite and tmax < 60.0)

            eta_now = np.asarray(st.eta.data, dtype=np.float32)
            t_now = np.asarray(st.T.data[:, :, 0], dtype=np.float32)
            u_full = np.asarray(st.u.data[:, :, 0], dtype=np.float32)
            v_full = np.asarray(st.v.data[:, :, 0], dtype=np.float32)
            u_now = u_full[:, 1:]
            v_now = v_full[1:, :]

            if daily_acc:
                _d, _gt = _acc_pair(st.u.data)
                if not (np.isfinite(_d) and np.isfinite(_gt)):
                    raise SystemExit(
                        f"FATAL: non-finite ACC at day {day_num} "
                        f"(dep={_d!r} gate={_gt!r}) -- a NaN transport must "
                        "never be averaged away into a retention curve")
                _acc_dep_daily[day_idx] = _d
                _acc_gate_daily[day_idx] = _gt

            eta_daily[day_idx] = eta_now
            sst_daily[day_idx] = t_now
            u_daily[day_idx] = u_now
            v_daily[day_idx] = v_now

            if save_3d and day_num in snaps:
                _snap(day_num)
                print(f"  captured day {day_num} full 3-D T/S/u/v snapshot", flush=True)

            print(f"  day {(k+1)*DT/86400:6.1f}  T[{tmin:.1f},{tmax:.1f}] finite={finite} "
                  f"max|eta|={np.nanmax(np.abs(eta_now)):.4f} "
                  f"max|u|={np.nanmax(np.abs(u_now)):.4f} "
                  f"max|v|={np.nanmax(np.abs(v_now)):.4f} wall={time.time()-t0:.0f}s", flush=True)

            if unstable:
                blew_up_at = k + 1
                print(f"BLOWUP detected at step {k+1} (day {(k+1)*DT/86400:.2f}) -- "
                      f"T range [{tmin:.2f},{tmax:.2f}] finite={finite}", flush=True)
                break

    stable = blew_up_at is None
    print(f"DONE {'blew up at step ' + str(blew_up_at) if blew_up_at else 'nsteps=' + str(nsteps)} "
          f"STABLE={stable}  wall={time.time()-t0:.0f}s", flush=True)

    # Which snapshot days actually have a 3-D block on disk, resolved ONCE:
    # the storage stamp, the write loop and the size line must all agree, and
    # three spellings of "day 0 is captured whenever --save-3d but only saved
    # if it is in the requested grid" would be three chances to disagree.
    _stored_days = [d for d in snaps if d in t3d]
    _snap_stamp = snap_name if _stored_days else "absent"
    # THE SERIES KEYS AND THE STAMP ARE BUILT FROM THE SAME CALL, so the stamp
    # can never promise a series the artifact does not carry (code review: the
    # two used to be computed thirty lines apart).
    _red_kwargs = reduced_series_kwargs(_reduced, snaps)
    _red_days = [int(d) for d in _red_kwargs.get("reduced_days", ())]
    # The series is stored float64 but RESOLVED at the precision the arm was
    # built at -- stamp the resolution, not the container (physics review).
    _red_stamp = control_dtype_stamp if _red_kwargs else "absent"
    if _reduce_fail:
        _reduced_status = (f"{_reduced_status}; reduction failed on days "
                           f"{sorted(_reduce_fail)}: "
                           f"{_reduce_fail[sorted(_reduce_fail)[0]]}")

    _producer_sha_exit, _producer_dirty_exit = _git_provenance()
    if (_producer_sha_exit != _producer_sha_entry
            or _producer_dirty_exit != _producer_dirty_entry):
        raise SystemExit(
            "REFUSING to save: producing checkout changed during integration "
            f"(entry={_producer_sha_entry}/{_producer_dirty_entry}, "
            f"exit={_producer_sha_exit}/{_producer_dirty_exit})")
    save_kwargs = dict(
        eta=eta_daily, sst=sst_daily, u=u_daily, v=v_daily,
        land_mask=land_mask.astype(np.float32),
        day=np.arange(1, n_days + 1, dtype=np.int32),
        blew_up_at_step=(blew_up_at if blew_up_at is not None else -1),
        stable=stable,
        # #1455: stamp the seasonal-clock offset INTO the artifact so a scorer
        # can read the one variable under test instead of trusting a filename.
        seasonal_t0_seconds=np.float64(t0_sec),
        # #1455: same reason -- stamp WHICH vertical ladders the bridge handed
        # legoESM, so the gate can say which grid a score was earned on instead
        # of inferring it from a filename. This is the value resolve_ladder_mode
        # RETURNED at the top of this function, not a re-read of the environment
        # -- provenance must not come from mutable process state.
        nemo_ladder_mode=np.str_(ladder_mode),
        # #1455: stamp WHICH time-integration START the twin took. Same reason
        # as the ladder: a scorer must be able to read the one variable under
        # test off the artifact instead of trusting a filename or a default
        # that has since moved. This is the value OBSERVED on the built state
        # (checked against what resolve_start_mode requested), not a re-read of
        # the flag -- a label with no identity behind it is what the ladder
        # content hash exists to prevent, and this dimension gets the same
        # treatment.
        twin_start_mode=np.str_(start_mode),
        # Round-3 bridge-only stagger receipt. "T" is stamped only after the
        # built carry was compared byte-for-byte with the existing analytic
        # DINO forcing loader at the restart's own time.
        bridge_before_stress_stagger=np.str_(bridge_before_stress_stagger),
        bridge_before_stress_reconstruction_seconds=np.float64(
            reconstruction_time),
        bridge_before_stress_sha256=np.str_(bridge_before_stress_sha256),
        # Bridge-only two-Earth counterfactual receipts. The geometry's own
        # scalar and Coriolis content hashes make the selector's effect
        # decidable; config_omega proves the model card stayed on NEMO Earth.
        bridge_omega_mode=np.str_(bridge_omega),
        bridge_omega_rad_s=np.float64(built_bridge_omega),
        bridge_f_T_sha256=np.str_(bridge_f_t_sha256),
        bridge_f_u_sha256=np.str_(bridge_f_u_sha256),
        bridge_f_v_sha256=np.str_(bridge_f_v_sha256),
        config_omega_rad_s=np.float64(config_omega),
        een_metric_weighting=np.str_(cfg.een_metric_weighting),
        # #1455 512517fdc: stamp the precision the arm was built at.
        control_dtype=np.str_(control_dtype_stamp),
        # #1455: stamp the lateral viscous velocity the arm ran at (NEMO's
        # rn_Uv). It is the one variable in the viscosity ablation, and a
        # scorer that reads it from the filename can be handed a swapped file
        # and produce a confidently wrong sign. Taken from the BUILT config,
        # not from the CLI argument, so it records what the model used.
        rn_Uv=np.float64(cfg.U_M),
        # The wind-placement arm must be selected and scored from artifact
        # content, never inferred from a filename or CLI transcript.
        surface_stress_implicit=np.bool_(mc.surface_stress_implicit),
        # #1640 (GLM): a label is a taxonomy, not an identity. Hash the
        # vertical-coordinate arrays ACTUALLY in memory so "same grid" is
        # decidable rather than asserted -- nemo_ladder_mode records intent,
        # this records the numbers.
        vertical_ladder_sha256=np.str_(vertical_ladder_sha256(br.z_coord)),
        # PER-FIELD STORAGE dtypes, stamped honestly.  control_dtype above
        # records the precision the arm was BUILT at and says nothing about
        # what was written to disk -- for every artifact recorded before this
        # stamp the two differ (fp64 compute, fp32 storage), which is exactly
        # the gap that made the quantum machinery in verdict360/floor90 read
        # the wrong thing off control_dtype.  Consumers read this through
        # snapshot_storage_dtypes(); an artifact without it gets
        # LEGACY_STORAGE_DTYPES, i.e. today's behaviour unchanged.
        storage_dtypes=np.str_(storage_stamp(_snap_stamp, _red_stamp)),
        # Why a reduced series is or is not present.  A missing key with no
        # reason is a consumer guessing; this makes it readable.  NOT named
        # reduced_* : the data series all share that prefix and a consumer
        # globbing it would get a string mixed in with the floats (code
        # review) -- the size accounting already had to special-case it.
        snapshot_reduction_status=np.str_(_reduced_status),
        # #1455 follow-up: the clock NEMO is on, and the rest of the recipe.
        seasonal_t0_reference_seconds=np.float64(t0_reference_sec),
        run_config=np.str_(run_config),
        # Bind the producing checkout to the artifact, not only its run log.
        # ``main`` has already made a dirty tracked tree fatal; programmatic
        # callers still stamp their dirt count so downstream gates can refuse.
        producer_git_sha=np.str_(_producer_sha_entry),
        producer_dirty_tracked_files=np.int32(_producer_dirty_entry),
        codex_session_id=np.str_(os.environ.get("CODEX_SESSION_ID", "")),
        initial_state_sha256=np.str_(initial_state_sha256),
    )
    if daily_acc:
        # #1455 Phase-2: stamp the perturbation next to the response it caused,
        # so a retention curve can never be assembled from a mislabelled arm.
        save_kwargs["acc_dep_daily"] = _acc_dep_daily
        save_kwargs["acc_gate_daily"] = _acc_gate_daily
        save_kwargs["perturb_baro"] = np.str_(perturb_baro or "")
        save_kwargs["perturb_baro_key"] = np.str_(perturb_baro_key)
        save_kwargs["perturb_baro_scale"] = np.float64(
            perturb_baro_scale if perturb_baro else 0.0)
        # The MEASURED injected transport, the number every retention factor
        # divides by.  Stamped so the scorer never has to be told it.
        save_kwargs["injected_sv"] = np.float64(_injected_sv)
    if eta_step is not None:
        # eta_wave_twin.load_side expects relative elapsed time on both model
        # artifacts. seasonal_t0_seconds separately stamps the absolute DINO
        # clock used by the forcing, so no phase information is lost.
        save_kwargs["eta_daily"] = save_kwargs["eta"]
        save_kwargs["eta"] = eta_step
        save_kwargs["t_seconds"] = (
            np.arange(1, nsteps + 1, dtype=np.float64) * DT)
        save_kwargs["capture_every_steps"] = np.int32(1)
        save_kwargs["dt_seconds"] = np.float64(DT)
    for d in _stored_days:
        save_kwargs[f"T3d_day{d}"] = t3d[d]
        save_kwargs[f"S3d_day{d}"] = s3d[d]
        save_kwargs[f"eta3d_day{d}"] = eta3d[d]
        save_kwargs[f"u3d_day{d}"] = u3d[d]
        save_kwargs[f"v3d_day{d}"] = v3d[d]

    # THE ALWAYS-ON CHEAP WIN: the gate/verdict scalars and the per-row
    # transport profile, at float64, on every snapshot day.  These are the
    # quantities the scorers reduce the 3-D block down to anyway, so storing
    # them directly removes the storage quantum from every one of them at a
    # cost of a few kB -- the fp32 3-D block stays the default.
    save_kwargs.update(_red_kwargs)

    np.savez(out_path, **save_kwargs)
    # SIZE ACCOUNTING, printed rather than argued: the fp64 3-D flag roughly
    # doubles the artifact, the reduced series is noise next to it.
    _b3d = sum(int(a.nbytes) for d in _stored_days
               for a in (t3d[d], s3d[d], eta3d[d], u3d[d], v3d[d]))
    _bred = sum(int(v.nbytes) for v in _red_kwargs.values())
    _itemsize = np.dtype(snap_np).itemsize
    print(f"SIZE: 3-D block {_b3d / 1e6:.1f} MB at {snap_name} "
          f"({len(_stored_days)} days x 5 fields; would be "
          f"{_b3d * 4 / _itemsize / 1e6:.1f} MB at float32, "
          f"{_b3d * 8 / _itemsize / 1e6:.1f} MB at float64)  |  "
          f"fp64 reduced series {_bred / 1e3:.1f} kB", flush=True)
    print(f"SAVED {out_path}  stable={stable}  "
          f"3-D snapshots at days={sorted(snaps)}  "
          f"fp64 reduced series at days={_red_days}", flush=True)
    return stable


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("recipe", help="dino recipe name, e.g. nemo_dino_kamm_mlf")
    p.add_argument("out", help="output .npz path")
    p.add_argument("--days", type=int, default=90, help="twin length in days (default 90)")
    p.add_argument("--save-3d", action="store_true",
                    help="also save full 3-D T/S/eta/u/v at days 0/30/60/90")
    p.add_argument("--run-traj", default=RUN_TRAJ, help="NEMO RUN_TRAJ dir (mesh_mask donor)")
    p.add_argument("--run-stepdump", default=RUN_STEPDUMP,
                    help="NEMO RUN_STEPDUMP dir (restart donor)")
    p.add_argument("--bridge-tke", action="store_true",
                    help="seed state.tke from the NEMO restart's en (#1317 TKE "
                         "cold-start isolation experiment); default off (cold start)")
    p.add_argument("--bridge-before", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="seed state.{T,S,u,v,eta}_before from the NEMO restart's "
                         "tb/sb/ub/vb/sshb (#1317 leap-frog before-level bridge), "
                         "so the twin CONTINUES NEMO's leap-frog. DEFAULT ON "
                         "since 2026-08-24 (#1455). The card's "
                         "tke_n2_time_level=nemo_before axis then reads NEMO's "
                         "own before-state from step 0 instead of a copy of "
                         "the now level (it does NOT raise without this -- an "
                         "earlier help string said it did; retracted). "
                         "--no-bridge-before is the legacy forward-Euler start "
                         "-- see --legacy-euler-start")
    stress_carry = p.add_mutually_exclusive_group()
    stress_carry.add_argument(
        "--bridge-before-stress-tpoint",
        dest="bridge_before_stress_tpoint", action="store_true",
        help="use the faithful analytic DINO T-point prior-stress reconstruction "
             "at the restart's own prior seasonal time (DEFAULT ON since "
             "2026-08-28). Requires --bridge-before; stamps stagger/time/content "
             "identity. Does not change model config or later-step carry behavior")
    stress_carry.add_argument(
        "--bridge-before-stress-legacy-u-as-t",
        dest="bridge_before_stress_tpoint", action="store_false",
        help="opt into the known-wrong historical bridge that stores NEMO's "
             "U/V-point prior stress as a T-point carry. Reproduction only; "
             "artifacts stamp U_AS_T_LEGACY")
    # Resolve the default after parsing ``bridge_before``: a bridged start
    # faithfully reconstructs the T-point carry by default, while an Euler
    # start has no prior-stress carry to reconstruct.  An explicit
    # ``--bridge-before-stress-tpoint --legacy-euler-start`` remains True and
    # is refused by _build_twin_state's fail-closed compatibility gate.
    p.set_defaults(bridge_before_stress_tpoint=None)
    p.add_argument("--bridge-omega", choices=BRIDGE_OMEGA_MODES, default="nemo",
                   help="bridge-geometry Earth rotation only: 'nemo' is the "
                        "default NEMO sidereal rate; 'legacy-rounded' "
                        "reproduces the historical rounded-constants bridge "
                        "while leaving model-config Omega on NEMO. Both modes "
                        "stamp scalar and f_T/f_u/f_v content identities")
    p.add_argument("--legacy-euler-start", dest="bridge_before",
                   action="store_false",
                   help="start the twin from a forward-Euler step instead of "
                        "NEMO's before level (the pre-2026-08-24 default; "
                        "spelled --no-bridge-before too). HISTORICAL "
                        "REPRODUCTION ONLY: it delays the trajectory about "
                        "half a step at every frequency AND injects a "
                        "permanent perturbation of half a leap-frog step of "
                        "tendency (~2.06e-3 Sv of circumpolar transport on the "
                        "day-180 restart). Prints a loud banner. NOTE "
                        "(#1729): it no longer skips NEMO's step-1 after-level "
                        "reconciliation -- the Euler start now runs the same "
                        "body every other step runs -- so this flag reproduces "
                        "the START MODE of pre-2026-08-24 arms, NOT their "
                        "numbers.")
    p.add_argument("--vmix-scheme", default=None,
                    help="override DINOConfig.vmix_scheme (e.g. 'constant' for "
                         "the #1317 TKE-vs-constant-mixing discriminator); "
                         "default None leaves the recipe's own scheme untouched")
    p.add_argument("--use-gm-redi", dest="use_gm_redi", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="override DINOConfig.use_gm_redi (GM ablation "
                         "discriminator: --no-use-gm-redi zeroes kappa_GM / "
                         "disables the Treguier/Visbeck adaptive-kappa "
                         "diagnostics while leaving kappa_Redi / isoneutral "
                         "slopes untouched -- see dino.py:2437-2503); "
                         "default None leaves the recipe's own value")
    p.add_argument("--surface-tendency-placement", default=None,
                    choices=("applied_now", "leapfrog_rhs"),
                    help="override DINOConfig.surface_tendency_placement "
                         "(#1492 A/B: 'applied_now' legacy defect vs "
                         "'leapfrog_rhs' NEMO-faithful fix); default None "
                         "leaves the recipe's own value")
    p.add_argument(
        "--barotropic-continuity-evaluation", default=None,
        choices=("generic", "nemo_literal"),
        help="one-variable QCO continuity association selector; default None "
             "uses the recipe (NEMO cards use nemo_literal)")
    p.add_argument(
        "--vface-zonal-metric-evaluation", default=None,
        choices=("legacy_tracer_midpoint", "nemo_vpoint"),
        help="one-variable V-face Mercator e1v selector; default None uses "
             "the recipe (NEMO cards use nemo_vpoint)")
    p.add_argument(
        "--tke-preclosure-coeff-source", default=None,
        choices=("carried_previous_step", "current_subiteration"),
        help="override the TKE pre-solve avm_k/avt_k lifetime; default None "
             "uses the recipe (DINO NEMO cards carry the previous-step pair; "
             "current_subiteration is historical reproduction)")
    p.add_argument(
        "--tke-matrix-evaluation", default=None,
        choices=("nemo_literal", "factored"),
        help="override the TKE matrix construction; default None uses the "
             "recipe (DINO NEMO cards use literal zdftke source order; "
             "factored is historical reproduction)")
    p.add_argument(
        "--tke-solver-evaluation", default=None,
        choices=("nemo_literal", "shared_thomas"),
        help="override the TKE tridiagonal recurrence; default None uses "
             "the recipe (DINO NEMO cards use the literal zdftke scans; "
             "shared_thomas is historical reproduction)")
    p.add_argument(
        "--zdf-implicit-solver-evaluation", default=None,
        choices=("nemo_literal", "shared_thomas"),
        help="override the final dynzdf/trazdf matrix and recurrence "
             "evaluation; default None uses the recipe (DINO NEMO cards "
             "use literal source order; shared_thomas is historical "
             "reproduction)")
    p.add_argument("--tke-etau-exponential-evaluation", default=None,
                   choices=("nemo_literal", "jax_expression"))
    p.add_argument("--tke-htau-evaluation", default=None,
                   choices=("nemo_literal", "jax_expression"))
    p.add_argument("--tke-mxl-raw-evaluation", default=None,
                   choices=("nemo_literal", "factored"))
    p.add_argument(
        "--tke-langmuir-evaluation", default=None,
        choices=("nemo_literal", "vectorized"),
        help="override the Langmuir source construction; default None uses "
             "the recipe (DINO NEMO cards use literal zdftke source order; "
             "vectorized is historical reproduction)")
    p.add_argument(
        "--tke-shear-evaluation-stage", default=None,
        choices=("step_entry", "implicit_solve_state"),
        help="override the zdf_sh2 evaluation lifetime; default None uses "
             "the recipe (complete DINO NEMO cards freeze step-entry p_sh2; "
             "implicit_solve_state is historical reproduction)")
    p.add_argument(
        "--tke-shear-metric-source", default=None,
        choices=("nemo_qco_live_face", "tpoint_jacobian"),
        help="override zdf_sh2 vertical face metrics; default None uses the "
             "recipe (DINO NEMO cards use live NOW/BEFORE QCO metrics; "
             "tpoint_jacobian is historical reproduction)")
    p.add_argument(
        "--tke-n2-evaluation-stage", default=None,
        choices=("step_entry", "implicit_solve_state"),
        help="override rn2/rn2b/live-geometry evaluation lifetime; default "
             "None uses the recipe (complete DINO NEMO cards retain the "
             "step-entry bundle; implicit_solve_state is historical "
             "reproduction)")
    p.add_argument(
        "--dino-wind-profile-evaluation", default=None,
        choices=("nemo_literal", "factored_smoothstep"),
        help="override DINO's wind-profile evaluation; default None uses the "
             "recipe (complete DINO NEMO cards use the literal Fortran "
             "association; factored_smoothstep is historical reproduction)")
    p.add_argument("--gm-redi-slope-n2-evaluation", default=None,
                   choices=("carried_step_entry", "recompute"))
    p.add_argument("--gm-redi-slope-prd-evaluation", default=None,
                   choices=("nemo_literal", "density_roundtrip"))
    p.add_argument("--gm-redi-slope-metric-evaluation", default=None,
                   choices=("nemo_reciprocal", "division"))
    p.add_argument("--gm-redi-slope-face-thickness-evaluation", default=None,
                   choices=("nemo_qco_live", "static_face"))
    p.add_argument("--gm-redi-slope-depth-evaluation", default=None,
                   choices=("nemo_qco_live_literal",
                            "legacy_jacobian_t_surface"))
    p.add_argument("--surface-stress-implicit", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="override DINOConfig.surface_stress_implicit for the "
                        "wind-placement A/B; default None leaves the recipe "
                        "unchanged. Artifacts stamp the resolved model value")
    p.add_argument("--u-m", dest="u_m", type=float, default=None,
                   help="override DINOConfig.U_M (NEMO rn_Uv, the lateral "
                        "viscous velocity [m/s]; card default 0.27). The "
                        "lateral viscosity coefficient is A_h = 0.5*U_M*dx, "
                        "so --u-m 0.54 DOUBLES the lateral viscosity and "
                        "nothing else (#1455 Munk ablation). Default None "
                        "leaves the recipe's own value.")
    p.add_argument("--legacy-1d-ladder", action="store_true",
                   help="build legoESM on the 1-D REFERENCE vertical ladder "
                        "(LEGOESM_NEMO_E3T=off) instead of NEMO's own "
                        "thickness+T-depth ladders. Historical-reproduction "
                        "mode: it costs +2.93 Sv of day-90 circumpolar "
                        "transport error against +0.29 Sv on the default "
                        "(#1455), and prints a loud banner. Fatal if "
                        "LEGOESM_NEMO_E3T is also set to anything but 'off'.")
    p.add_argument("--perturb-seed", type=int, default=None,
                    help="#1492 item-2.2 noise control: apply a relative "
                         "multiplicative perturbation to the bridged now-level "
                         "T using numpy.random.default_rng(seed); default None "
                         "= no perturbation. Amplitude from --perturb-eps.")
    p.add_argument("--perturb-eps", type=float, default=PERTURB_EPS_DEFAULT,
                   help="relative amplitude of the --perturb-seed kick "
                        f"(default {PERTURB_EPS_DEFAULT:g}, the value every "
                        "recorded noise-floor ensemble used -- unchanged so "
                        "those artifacts stay comparable). Raise it to build a "
                        "within-arm null whose amplitude MATCHES the difference "
                        "under test: a 1e-14 kick is still growing at day 90 "
                        "and is a floor, not a null (#1455).")
    p.add_argument("--perturb-baro", default=None,
                   help="#1455 Phase-2 Measurement 1: npz holding a "
                        "depth-uniform barotropic velocity field, injected ONCE "
                        "at t=0 into the now-level u. Written by "
                        "substep_traj_compare.py's DINO_1455_DEPOSIT_MAP block.")
    p.add_argument("--perturb-baro-key", default="dU_avg",
                   help="which array in --perturb-baro to inject: dU_avg (the "
                        "TOTAL per-step deposit) or dU_sub (the in-loop share)")
    p.add_argument("--perturb-baro-scale", type=float, default=1.0,
                   help="multiplier on the injected field; the linearity "
                        "control arm uses 0.5, the sign arm -1.0")
    p.add_argument("--snap-days", default=None,
                   help="comma-separated days for the --save-3d 3-D snapshots "
                        "(default: the recorded 0,30,60,90 grid). Days past "
                        "--days are dropped.")
    p.add_argument("--fp64-3d", dest="fp64_3d", action="store_true",
                   help="store the 3-D T/S/eta/u/v snapshot block at float64 "
                        "instead of float32. Roughly DOUBLES the artifact, so "
                        "it is off by default; turn it on for ensemble runs "
                        "where measuring the SPREAD between near-identical "
                        "members is the point, which is where float32 storage "
                        "has capped this campaign (members tying at the "
                        "storage quantum on max-type metrics). The fp64 "
                        "REDUCED time series is stored either way and is the "
                        "cheap win -- this flag only matters for a metric "
                        "nobody has reduced yet.")
    p.add_argument("--daily-acc", action="store_true",
                   help="#1455 Phase-2: store the ACC transport EVERY day under "
                        "both the deposit's reducer and the recorded gate's. "
                        "The 0/30/60/90 snapshot grid cannot resolve a decay "
                        "timescale of days, which is what the retention "
                        "measurement is pre-registered to discriminate.")
    p.add_argument("--save-step-eta", action="store_true",
                   help="store every-step eta at float64 under the scorer's "
                        "eta/t_seconds contract; preserve daily eta as "
                        "eta_daily. Required for 2dt/Nyquist scoring")
    args = p.parse_args(argv)
    if args.bridge_before_stress_tpoint is None:
        args.bridge_before_stress_tpoint = bool(args.bridge_before)
    return args


def _smoke_check_vmix_scheme_override():
    """Runnable check: --vmix-scheme actually overrides cfg.vmix_scheme.

    No NEMO bridge/restart required -- exercises the same
    dataclasses.replace path _build_twin_state uses.
    """
    base = dino_config_for_recipe("nemo_dino_kamm_mlf")
    assert base.vmix_scheme == "tke", (
        f"expected nemo_dino_kamm_mlf default vmix_scheme='tke', got {base.vmix_scheme!r}")
    overridden = dataclasses.replace(base, vmix_scheme="constant")
    assert overridden.vmix_scheme == "constant"
    assert base.vmix_scheme == "tke", "override must not mutate the original cfg"
    print("OK: --vmix-scheme override changes cfg.vmix_scheme (tke -> constant)")

    assert base.use_gm_redi is True, (
        f"expected nemo_dino_kamm_mlf default use_gm_redi=True, got {base.use_gm_redi!r}")
    gm_off = dataclasses.replace(base, use_gm_redi=False)
    assert gm_off.use_gm_redi is False
    assert base.use_gm_redi is True, "override must not mutate the original cfg"
    print("OK: --use-gm-redi override changes cfg.use_gm_redi (True -> False)")

    # Base-AGNOSTIC by design: this self-check proves the OVERRIDE MECHANISM
    # works; it must not pin the card's default.  It previously asserted
    # 'applied_now' and therefore went RED the moment the card was CORRECTED
    # to 'leapfrog_rhs' (#1492 -- 'applied_now' under the leap-frog discards
    # ~56% of every applied surface flux, retention (1-2g)/(2(1-g)) = 4/9 at
    # g=0.1; the recorded 10-yr A/B closes 98.7% of the ACC gap and takes
    # sigma>1.6 dense water from 0.000x to 1.031x NEMO).  A harness self-check
    # that fails BECAUSE the model was fixed is a broken tripwire.
    _placements = ("applied_now", "leapfrog_rhs")
    assert base.surface_tendency_placement in _placements, (
        f"unknown surface_tendency_placement "
        f"{base.surface_tendency_placement!r} on nemo_dino_kamm_mlf")
    _other = _placements[1 - _placements.index(base.surface_tendency_placement)]
    flipped = dataclasses.replace(base, surface_tendency_placement=_other)
    assert flipped.surface_tendency_placement == _other
    assert base.surface_tendency_placement != _other, (
        "override must not mutate the original cfg")
    print("OK: --surface-tendency-placement override changes "
          f"cfg.surface_tendency_placement "
          f"({base.surface_tendency_placement} -> {_other})")

    assert base.surface_stress_implicit is False
    wind_implicit = dataclasses.replace(base, surface_stress_implicit=True)
    assert wind_implicit.surface_stress_implicit is True
    assert base.surface_stress_implicit is False
    print("OK: --surface-stress-implicit override changes "
          "cfg.surface_stress_implicit (False -> True)")

    # --u-m: the #1455 Munk ablation knob. Assert BOTH that the field moves
    # and that the quantity it feeds (the lateral-viscosity coefficient the
    # dycore actually reads) moves by the same factor -- a field that changed
    # while A_h did not would be a vacuous knob.
    from legoesm.ocean.experiments.dino import dino_lat_lon_grid, dino_lat_lon_model_config
    doubled = dataclasses.replace(base, U_M=2.0 * base.U_M)
    assert doubled.U_M == 2.0 * base.U_M
    assert base.U_M == 0.27, f"expected card rn_Uv=0.27, got {base.U_M}"
    _g = dino_lat_lon_grid(base)
    _ah1 = dino_lat_lon_model_config(_g, base, physics=False)[0].lateral_viscosity.A_h
    _ah2 = dino_lat_lon_model_config(_g, doubled, physics=False)[0].lateral_viscosity.A_h
    assert abs(_ah2 / _ah1 - 2.0) < 1e-12, (
        f"--u-m doubling must double A_h; got {_ah1} -> {_ah2}")
    print(f"OK: --u-m override doubles the lateral viscosity "
          f"(U_M {base.U_M} -> {doubled.U_M}, A_h {_ah1:.4f} -> {_ah2:.4f} m2/s)")

    # "changes A_h" is only half the claim the pre-registration makes; the other
    # half is "and nothing else". Diff every leaf of the two built configs and
    # require exactly one to move, so a future edit that quietly routes U_M into
    # a second consumer (a CFL-derived substep count, a diagnostic coefficient)
    # turns this red instead of silently making the ablation two-variable.
    def _leaves(obj, path=""):
        if hasattr(obj, "_fields"):
            for f in obj._fields:
                yield from _leaves(getattr(obj, f), f"{path}.{f}" if path else f)
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for f in dataclasses.fields(obj):
                yield from _leaves(getattr(obj, f.name),
                                   f"{path}.{f.name}" if path else f.name)
        else:
            yield path, obj

    _l1 = dict(_leaves(dino_lat_lon_model_config(_g, base, physics=False)[0]))
    _l2 = dict(_leaves(dino_lat_lon_model_config(_g, doubled, physics=False)[0]))
    assert set(_l1) == set(_l2), "the two configs do not have the same leaves"
    _moved = sorted(k for k in _l1
                    if not _eq_leaf(_l1[k], _l2[k]))
    assert _moved == ["lateral_viscosity.A_h"], (
        f"--u-m must move exactly lateral_viscosity.A_h and nothing else; it "
        f"moved {_moved} (of {len(_l1)} leaves)")
    print(f"OK: --u-m moves exactly one of the {len(_l1)} built-config leaves "
          f"({_moved[0]}) -- the ablation is one variable")


def _eq_leaf(a, b) -> bool:
    """Leaf equality that tolerates arrays and None."""
    import numpy as _np
    if a is None or b is None:
        return a is b
    try:
        return bool(_np.array_equal(_np.asarray(a), _np.asarray(b)))
    except Exception:
        return a is b or a == b


def _git_provenance() -> tuple[str, int]:
    """Return producing HEAD and tracked-dirt count for log/artifact stamps."""
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True).stdout.strip()
    return sha, len(dirt.splitlines()) if dirt else 0


def provenance_gate() -> None:
    """Stamp source provenance and REFUSE to run from a dirty tracked tree.

    Added after the 2026-08-19/20 reconciliation (#1455, a009c6812): two runs
    of this harness at byte-identical committed source differed by 2.5 Sv in
    day-90 ACC, and the second review's log forensics left "an uncommitted
    working-tree edit" as the sole surviving candidate -- the difference is
    unrecoverable because nothing stamped the tree state. Every future run
    prints the HEAD sha and the tracked-file dirt count; a dirty tree aborts
    unless LEGOESM_ALLOW_DIRTY=1 is set explicitly (and then the dirt list is
    printed so the log carries what the tree carried).
    """
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE: HEAD={sha} dirty_tracked_files={len(dirt.splitlines())}")
    if dirt:
        print("PROVENANCE: dirty tracked files:")
        for line in dirt.splitlines():
            print(f"  {line}")
        if os.environ.get("LEGOESM_ALLOW_DIRTY") != "1":
            raise SystemExit(
                "REFUSING to run from a dirty tracked tree (see #1455 "
                "a009c6812: an uncommitted edit produced an unattributable "
                "2.5 Sv shift). Commit or stash, or set LEGOESM_ALLOW_DIRTY=1 "
                "to run anyway with the dirt list stamped in the log.")


def _fp64_requested() -> bool:
    return os.environ.get("FP64", "1") == "1"


def _precision_gate() -> None:
    """Set the fp64 policy at entry (the materialized check runs later).

    Added after the 0.213 Sv "baseline discrepancy" (#1455, 512517fdc): arms
    run without the run_fp64.py wrapper silently built the whole oracle
    comparison at the fp32 policy default (JAX_ENABLE_X64 alone does not set
    the policy -- skill Rule 1c). This only WRITES the policy; writing and
    re-reading the same global proves nothing (review a0cd04b8: a backend
    without f64 hardware clamps the real arrays to f32 while the policy
    still reads f64). The binding check is require_fp64() on the BUILT
    geometry+state inside run_twin -- the shared gate every sibling probe
    uses (ocean/fidelity/precision_gate.py). FP64=0 = deliberate fp32 arm.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy
    if _fp64_requested():
        set_policy(PrecisionPolicy.fp64())


def main(argv=None):
    args = _parse_args(argv)
    provenance_gate()
    _precision_gate()
    if args.recipe == "smoke-check":
        _smoke_check_vmix_scheme_override()
        return
    run_twin(args.recipe, args.out, n_days=args.days, save_3d=args.save_3d,
              run_traj=args.run_traj, run_stepdump=args.run_stepdump,
              bridge_tke=args.bridge_tke, bridge_before=args.bridge_before,
              bridge_before_stress_tpoint=args.bridge_before_stress_tpoint,
              vmix_scheme=args.vmix_scheme, use_gm_redi=args.use_gm_redi,
              surface_stress_implicit=args.surface_stress_implicit,
              surface_tendency_placement=args.surface_tendency_placement,
              barotropic_continuity_evaluation=(
                  args.barotropic_continuity_evaluation),
              vface_zonal_metric_evaluation=(
                  args.vface_zonal_metric_evaluation),
              tke_preclosure_coeff_source=args.tke_preclosure_coeff_source,
              tke_matrix_evaluation=args.tke_matrix_evaluation,
              tke_solver_evaluation=args.tke_solver_evaluation,
              zdf_implicit_solver_evaluation=(
                  args.zdf_implicit_solver_evaluation),
              tke_etau_exponential_evaluation=(
                  args.tke_etau_exponential_evaluation),
              tke_htau_evaluation=args.tke_htau_evaluation,
              tke_mxl_raw_evaluation=args.tke_mxl_raw_evaluation,
              tke_langmuir_evaluation=args.tke_langmuir_evaluation,
              tke_shear_evaluation_stage=args.tke_shear_evaluation_stage,
              tke_shear_metric_source=args.tke_shear_metric_source,
              tke_n2_evaluation_stage=args.tke_n2_evaluation_stage,
              dino_wind_profile_evaluation=args.dino_wind_profile_evaluation,
              gm_redi_slope_n2_evaluation=args.gm_redi_slope_n2_evaluation,
              gm_redi_slope_prd_evaluation=args.gm_redi_slope_prd_evaluation,
              gm_redi_slope_metric_evaluation=(
                  args.gm_redi_slope_metric_evaluation),
              gm_redi_slope_face_thickness_evaluation=(
                  args.gm_redi_slope_face_thickness_evaluation),
              gm_redi_slope_depth_evaluation=(
                  args.gm_redi_slope_depth_evaluation),
              u_m=args.u_m,
              perturb_seed=args.perturb_seed, perturb_eps=args.perturb_eps,
              perturb_baro=args.perturb_baro,
              perturb_baro_key=args.perturb_baro_key,
              perturb_baro_scale=args.perturb_baro_scale,
              daily_acc=args.daily_acc,
              save_step_eta=args.save_step_eta,
              snap_days=(None if args.snap_days is None else
                         tuple(int(x) for x in args.snap_days.split(","))),
              fp64_3d=args.fp64_3d,
              legacy_1d_ladder=args.legacy_1d_ladder,
              bridge_omega=args.bridge_omega)


if __name__ == "__main__":
    main()
