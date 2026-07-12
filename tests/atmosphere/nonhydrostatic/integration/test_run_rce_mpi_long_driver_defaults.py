"""Regression test for the plane CRM production driver argparse defaults.

iter-59 found ``scripts/run/run_rce_mpi_long.py`` argparse defaults still
shipping with ``--n-acoustic-substeps`` defaulting to ``24`` (set
for iter-1's ``dt=1.0 s`` config) while the driver's ``--dt`` default
is ``5.0 s`` (iter-12/14 production). The combination produces an
over-stable but 2x-slower substep vs iter-14's measured production
contract (N=12 at dt=5).

This test parses the driver source via ``ast`` to find every
``p.add_argument`` call and asserts the production-contract defaults
match iter-14 + iter-38 measurements. Catches future silent reverts
in < 1 s before any of the slow nightly tests would.

Mirrors the iter-58 pattern (``test_run_rce_30day_wrapper_defaults.py``)
which locks the wrapper-script env-var defaults.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER = REPO_ROOT / "scripts" / "run" / "run_rce_mpi_long.py"


def _parse_argparse_defaults(source: str) -> dict[str, object]:
    """Walk the AST for every ``p.add_argument(...)`` call and return
    a dict mapping CLI flag → default value literal. Returns only
    flags with literal defaults (skips ones using ``default=None``
    or non-constant defaults).
    """
    tree = ast.parse(source)
    out: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant):
            continue
        flag = first.value
        if not (isinstance(flag, str) and flag.startswith("--")):
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                out[flag] = kw.value.value
                break
    return out


@pytest.fixture(scope="module")
def driver_defaults() -> dict[str, object]:
    assert DRIVER.exists(), f"driver script missing: {DRIVER}"
    return _parse_argparse_defaults(DRIVER.read_text())


def test_dt_default_matches_iter183(driver_defaults):
    """iter-180/183 refresh: Van Leer (iter-183, after iter-180's
    WENO5 walked back on a wall-time regression) + beta=0.2 lets the
    production driver run at dt=20.0 s (4x iter-14's dt=5.0 + 3x
    wall-time speedup vs dt=10+upwind1). The driver default moved
    in lockstep with the wrapper. Locks the new production
    contract."""
    assert driver_defaults["--dt"] == 20.0


def test_n_acoustic_substeps_default_matches_iter14(driver_defaults):
    """iter-14 + iter-38 production use N_ACOUSTIC=12 at dt=5.0 s.
    iter-59 refreshed the driver default from the stale 24
    (iter-1 dt=1.0 s value) to 12."""
    assert driver_defaults["--n-acoustic-substeps"] == 12, (
        f"--n-acoustic-substeps default = "
        f"{driver_defaults['--n-acoustic-substeps']!r}, expected 12 "
        f"(iter-14 + iter-38 production contract). With dt=5.0 + "
        f"N=24 the acoustic CFL ratio halves — over-stable but "
        f"2x-cost substep, silently differs from the iter-14 "
        f"measurement."
    )


def test_hyperdiff_default_matches_iter14(driver_defaults):
    """iter-15 + iter-38 production use hyperdiff=5e6."""
    assert driver_defaults["--hyperdiff"] == 5.0e6


def test_smag_cs_default_matches_iter14(driver_defaults):
    """iter-3 + iter-38 production use Smag c_s=0.2."""
    assert driver_defaults["--smag-cs"] == 0.2


def test_no_bubble_default(driver_defaults):
    """F7/F10 contract: production runs on clean Wing IC (no bubble)."""
    assert driver_defaults["--bubble-theta-pert"] == 0.0


def test_no_qv_noise_default(driver_defaults):
    """F7/F10 contract: production runs without qv noise."""
    assert driver_defaults["--qv-noise-amp"] == 0.0


def test_nx_ny_production_defaults(driver_defaults):
    """iter-12+ production grid is 132x132."""
    assert driver_defaults["--nx"] == 132
    assert driver_defaults["--ny"] == 132


def test_nlev_production_default(driver_defaults):
    """iter-12 + iter-14 production uses nlev=30."""
    assert driver_defaults["--nlev"] == 30


# iter-61 Codex MEDIUM coverage-gap fix: assert the production-anchored
# defaults that iter-59 missed. These are values that affect either the
# stability envelope (sponge_*, advection, semi-implicit) or the
# observability (log_every_steps, rad_call_interval_s) of the iter-14
# production measurement. Each is grounded in CRM_implementation.md.


def test_dx_production_default(driver_defaults):
    """iter-12 production grid spacing: dx=2 km."""
    assert driver_defaults["--dx"] == 2000.0


def test_H_production_default(driver_defaults):
    """iter-12 production domain height: H=33 km."""
    assert driver_defaults["--H"] == 33_000.0


def test_dz_sfc_production_default(driver_defaults):
    """iter-12 surface dz=100 m."""
    assert driver_defaults["--dz-sfc"] == 100.0


def test_vertical_grid_production_default(driver_defaults):
    """iter-14/38 production: --vertical-grid uniform (dz=H/nlev ≈ 1100m
    at nlev=30, H=33km). Stretched grid is opt-in via CLI."""
    assert driver_defaults["--vertical-grid"] == "uniform"


def test_n_physics_substeps_production_default(driver_defaults):
    """iter-14/38 production: 10 physics substeps per dycore step."""
    assert driver_defaults["--n-physics-substeps"] == 10


def test_rad_call_interval_default(driver_defaults):
    """Driver default rad cadence = 600 s. Production runs without
    --no-radiation. iter-39 added --no-radiation flag with default
    False (production unaffected)."""
    assert driver_defaults["--rad-call-interval-s"] == 600.0
    assert driver_defaults["--no-radiation"] is False


def test_no_mass_fixer_default(driver_defaults):
    """iter-95 added --no-mass-fixer to disable
    fix_moist_mass_plane{,_mpi}. Default MUST stay False so that:

    * Gravity-wave / hydrostatic smokes (where total water IS
      conserved) keep the legacy fixer to catch mass drift bugs.
    * Existing 30-day production runs are unaffected (backward-
      compat).
    * RCE spinup callers must explicitly opt-in via the flag —
      otherwise iter-95 Bug 2 returns (surface flux is removed
      every step → CWV pinned at IC → no convection).

    If someone silently flips this to True ``default=True``,
    every existing wave-mode regression that relies on the fixer
    would start drifting.
    """
    assert driver_defaults["--no-mass-fixer"] is False, (
        f"--no-mass-fixer default = "
        f"{driver_defaults['--no-mass-fixer']!r}, expected False. "
        f"Flipping the default to True breaks every existing "
        f"gravity-wave / hydrostatic smoke that asserts total "
        f"water is conserved. iter-95 added the flag as an "
        f"opt-in for RCE spinup, not as the new default."
    )


def test_adaptive_dt_default(driver_defaults):
    """iter-228: --adaptive-dt opt-in flag parses but is a stub
    until the F11 fix-path-4 time-loop restructure lands. Default
    False keeps the iter-183 production contract unchanged."""
    assert driver_defaults["--adaptive-dt"] is False


def test_use_dd_default(driver_defaults):
    """iter-5 added --use-dd; default False keeps the legacy
    rank-0-broadcast F8-stable path as the canonical entry."""
    assert driver_defaults["--use-dd"] is False


def test_advection_production_default(driver_defaults):
    """iter-183 wall-time refresh: Van Leer TVD is the production
    default at dt=20. iter-180 had set WENO5 as the new default
    based on a stability check alone, but the iter-183 wall-time
    measurement showed WENO5 at dt=20 actually runs SLOWER than
    iter-14's dt=10+upwind1 (2x fewer steps but 2.4x per-step
    cost → net 1.22x slower). Van Leer at dt=20 is 3x faster than
    the baseline at the same sim time. WENO5 stays available as
    opt-in for sharp-front problems."""
    assert driver_defaults["--advection"] == "van_leer"


def test_vertical_tracer_advection_default_van_leer(driver_defaults):
    """codex CRM-dycore review: --vertical-tracer-advection defaults to
    van_leer, matching the serial run_rcemip_plane default so serial and MPI
    RCE use the SAME (positive-definite) vertical tracer scheme. Now that the
    MPI halo path HONORS the config (previously it silently used centered), a
    flip to 'centered' would re-introduce non-monotone vertical tracer
    transport (negative q at sharp convective gradients) and re-diverge the
    MPI RCE driver from serial."""
    assert driver_defaults["--vertical-tracer-advection"] == "van_leer"


def test_sponge_production_defaults(driver_defaults):
    """iter-1 sponge layer config."""
    assert driver_defaults["--sponge-coeff"] == 0.05
    assert driver_defaults["--sponge-width"] == 10_000.0


def test_implicit_buoyancy_default(driver_defaults):
    """F2/F10: KW78 substep is inert at production dt; default off."""
    assert driver_defaults["--implicit-buoyancy"] is False


def test_qv_noise_seed_default(driver_defaults):
    """Deterministic seed for qv-noise (when amp > 0)."""
    assert driver_defaults["--qv-noise-seed"] == 0


def test_driver_argparse_advection_choices_derived_from_shared_map():
    """iter-192 Codex MEDIUM#2: argparse ``--advection choices`` MUST
    be derived from the shared HORIZONTAL_ADVECTION_HALO_REQUIREMENT
    map (or its alias) so a future fourth scheme automatically
    surfaces in --help. A regression that hardcodes
    ``choices=["upwind1", "van_leer", "weno5"]`` inline would
    re-introduce the iter-186/187 dual-registry drift hazard the
    fix collapsed.

    AST-based check: locate the ``p.add_argument("--advection", ...)``
    call and verify the ``choices`` keyword argument is a call to
    ``sorted(...)`` with a Name argument that resolves to the same
    alias used elsewhere in the file for HORIZONTAL_ADVECTION_HALO_REQUIREMENT.
    """
    import ast
    tree = ast.parse(DRIVER.read_text())
    # Find what name HORIZONTAL_ADVECTION_HALO_REQUIREMENT is bound to.
    aliases = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == (
            "legoesm.atmosphere.dynamics.compressible_euler_plane"
        ):
            for alias in node.names:
                if alias.name == "HORIZONTAL_ADVECTION_HALO_REQUIREMENT":
                    aliases.add(alias.asname or alias.name)
    assert aliases, (
        "Driver missing the iter-187 shared map import (precondition "
        "for the iter-192 choices derivation)."
    )
    # Walk every ``p.add_argument("--advection", ...)`` call.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and func.attr == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--advection"):
            continue
        # Found the --advection add_argument call. Inspect the
        # choices kwarg.
        choices_kw = next(
            (kw for kw in node.keywords if kw.arg == "choices"),
            None,
        )
        assert choices_kw is not None, (
            "--advection add_argument call missing choices kwarg "
            "(iter-192 contract: derived from sorted(_ADV_HALO_REQ))."
        )
        # Expect: sorted(<one of the aliases>)
        val = choices_kw.value
        assert (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Name)
            and val.func.id == "sorted"
            and len(val.args) == 1
            and isinstance(val.args[0], ast.Name)
            and val.args[0].id in aliases
        ), (
            "--advection choices must be sorted(<shared-map-alias>); "
            f"got ast.dump(val)={ast.dump(val)} with known aliases "
            f"{aliases!r}. A hardcoded list re-introduces the dual-"
            f"registry drift hazard."
        )
        return
    pytest.fail(
        "Driver source has no ``p.add_argument(\"--advection\", ...)`` "
        "call — argparse contract broken."
    )


def test_driver_consults_shared_halo_requirement_map():
    """iter-187/192: the driver MUST source its per-scheme halo
    requirement from the shared
    HORIZONTAL_ADVECTION_HALO_REQUIREMENT map in
    compressible_euler_plane, not a hardcoded inline dict.

    Pre iter-187 the driver hardcoded
    ``{"upwind1": 1, "van_leer": 2, "weno5": 3}`` inline and the
    halo dispatch hardcoded the same values separately — classic
    drift hazard. iter-187 collapsed to a single source of truth.

    iter-192 (Codex LOW): the iter-187 substring scrape could
    false-pass if the import lived in a comment, dead branch, or
    unused helper. Switched to an AST walk that verifies the
    import is a top-level executable statement.
    """
    import ast
    tree = ast.parse(DRIVER.read_text())
    # Find a top-level ImportFrom that brings in
    # HORIZONTAL_ADVECTION_HALO_REQUIREMENT (under any local alias).
    import_found = False
    aliases = set()
    for node in tree.body:  # top-level only — no commented or
        # inside-function code paths.
        if isinstance(node, ast.ImportFrom) and node.module == (
            "legoesm.atmosphere.dynamics.compressible_euler_plane"
        ):
            for alias in node.names:
                if alias.name == "HORIZONTAL_ADVECTION_HALO_REQUIREMENT":
                    import_found = True
                    aliases.add(alias.asname or alias.name)
    assert import_found, (
        "Driver must have a TOP-LEVEL import of "
        "HORIZONTAL_ADVECTION_HALO_REQUIREMENT from "
        "compressible_euler_plane (iter-192 AST check)."
    )
    # The runtime call must look up the required halo via the
    # imported name (or its alias). Walk the AST for a Subscript
    # like ``<imported>[args.advection]``.
    lookup_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in aliases:
                # Verify the subscript expression is args.advection
                # (don't allow a hardcoded scheme name).
                idx = node.slice
                if isinstance(idx, ast.Attribute) and isinstance(idx.value, ast.Name):
                    if idx.value.id == "args" and idx.attr == "advection":
                        lookup_found = True
                        break
    assert lookup_found, (
        "Driver must look up the required halo via "
        "<imported>[args.advection] using a name imported from "
        "HORIZONTAL_ADVECTION_HALO_REQUIREMENT (iter-186 Codex HIGH "
        "fix). Found neither a direct nor an aliased subscript."
    )


def test_driver_imports_build_smooth_k1_pattern_from_package():
    """iter-218: iter-208 moved build_smooth_k1_pattern from
    scripts/run/run_rce_mpi_long.py to
    legoesm.atmosphere.idealized.rcemip_initial_conditions. The
    driver now imports it via the standard package path.

    Pre iter-218 no test asserted this import wiring. If someone
    reverted the iter-208 move by inlining the function back in
    the driver, the iter-204/207 unit test would still pass (it
    imports from the package); the unit-test-vs-driver path
    coverage would silently diverge.

    AST-based check: assert the driver has a TOP-LEVEL ImportFrom
    that brings in build_smooth_k1_pattern from the rcemip_initial_conditions
    module.
    """
    import ast
    tree = ast.parse(DRIVER.read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == (
            "legoesm.atmosphere.idealized.rcemip_initial_conditions"
        ):
            for alias in node.names:
                if alias.name == "build_smooth_k1_pattern":
                    return
    pytest.fail(
        "Driver must have a top-level import of "
        "build_smooth_k1_pattern from "
        "legoesm.atmosphere.idealized.rcemip_initial_conditions "
        "(iter-208 helper-move contract). A future revert to an "
        "inline driver-local implementation would silently bypass "
        "the iter-204/207 unit test."
    )


def test_driver_argparse_theta_noise_mode_choices_locked():
    """iter-211 Codex LOW: lock the exact ``--theta-noise-mode``
    choices list via AST so a future PR that adds a 'smooth_k2'
    (or any other) mode without updating the iter-204
    test_smooth_k1_pattern_zero_mean_and_bounded grid sweep
    (or adding a sibling pattern test) surfaces here.

    Unlike --advection (iter-193) the theta-noise-mode choices
    are NOT yet derived from a shared registry — both modes are
    hand-coded in the driver dispatch. Lock the EXACT list ``[
    "white", "smooth_k1"]`` so introducing a new mode forces a
    coordinated update in tests AND in build_smooth_k1_pattern's
    sibling pattern functions.
    """
    import ast
    tree = ast.parse(DRIVER.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and func.attr == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--theta-noise-mode"):
            continue
        # Found the --theta-noise-mode add_argument call.
        choices_kw = next(
            (kw for kw in node.keywords if kw.arg == "choices"),
            None,
        )
        assert choices_kw is not None, (
            "--theta-noise-mode add_argument missing choices kwarg "
            "(iter-211 contract: explicit list with paired tests)."
        )
        val = choices_kw.value
        # Expect a List of string Constants: ["white", "smooth_k1"]
        assert isinstance(val, ast.List), (
            f"--theta-noise-mode choices must be a list literal; "
            f"got ast.dump(val)={ast.dump(val)}"
        )
        names = [
            elt.value for elt in val.elts
            if isinstance(elt, ast.Constant)
        ]
        assert names == ["white", "smooth_k1"], (
            f"--theta-noise-mode choices = {names}, expected "
            f"['white', 'smooth_k1']. iter-211 contract: any new "
            f"mode must come with paired pattern + unit test "
            f"updates so the iter-204 four-grid coverage isn't left "
            f"stale."
        )
        return
    pytest.fail(
        "Driver source has no ``p.add_argument(\"--theta-noise-mode\", ...)`` "
        "call — iter-203 dispatch broken."
    )


def test_theta_noise_defaults(driver_defaults):
    """iter-181 added --theta-noise-amp / --theta-noise-seed for the
    RCEMIP / Wing 2018 standard theta' symmetry-breaker. Default
    0.0 = clean Wing IC (no perturbation). F11 documents why even
    non-zero theta' noise is not yet a recommended path. Lock the
    defaults so the iter-181 alternative-symmetry-breaker entry
    point cannot silently activate at production scale.

    iter-208: --theta-noise-mode (added iter-203) defaults to
    'white' — the RCEMIP / Wing 2018 uniform random pattern. The
    'smooth_k1' alternative is the F11 fix-path-3 candidate; the
    default must stay 'white' so the production wrapper's
    THETA_NOISE_MODE=white pass-through is consistent.
    """
    assert driver_defaults["--theta-noise-amp"] == 0.0
    assert driver_defaults["--theta-noise-seed"] == 0
    assert driver_defaults["--theta-noise-mode"] == "white"


# iter-64: cover the snapshot/profile/log cadence defaults. iter-61
# added the wrapper-side hardcoded values (--snapshot-hours 24.0,
# --profile-days 5.0, --log-every-steps 100) as parametric
# assertions but the driver-side defaults were uncovered. The
# wrapper relies on these matching: a driver default change would
# now be silently different from the wrapper claim.


def test_snapshot_hours_default(driver_defaults):
    """24-hour snapshot cadence (daily). Wrapper also passes 24.0
    explicitly — driver+wrapper should agree."""
    assert driver_defaults["--snapshot-hours"] == 24.0


def test_snapshot_3d_hours_default(driver_defaults):
    """Driver default 0.0 = disabled. Wrapper overrides to 1.0
    (hourly 3D snapshots for GIF generation) — this is the only
    legitimate wrapper override of a driver default in this
    cadence group."""
    assert driver_defaults["--snapshot-3d-hours"] == 0.0


def test_profile_days_default(driver_defaults):
    """5-day profile cadence. Wrapper also passes 5.0 explicitly."""
    assert driver_defaults["--profile-days"] == 5.0


def test_log_every_steps_default(driver_defaults):
    """Production driver: log every 100 outer steps. Wrapper also
    passes 100 explicitly. iter-15/38 short smokes / production-
    scale tests override to 20/15/60 for tighter inspection."""
    assert driver_defaults["--log-every-steps"] == 100


# iter-153: pin the three surface-flux + acoustic-damping defaults
# that drive the iter-14 measured production envelope (max|w| = 6.1e-3
# m/s, MSE drift = 1.7e-4 over 725 steps). Silent changes here would
# alter CWV equilibrium or mask instabilities while still passing
# every other test in this file.


def test_c_h_production_default(driver_defaults):
    """Surface bulk heat exchange coefficient.  c_h = 1.5e-3 is the
    iter-14 production value (typical ocean ABL transfer coefficient).
    A silent change to e.g. 1e-3 would shift the CWV equilibrium
    outside the Wing 2018 45-60 mm plateau without tripping any
    other test."""
    assert driver_defaults["--c-h"] == 1.5e-3


def test_acoustic_off_centering_production_default(driver_defaults):
    """iter-180/183 refresh: beta=0.2 is the production default
    pairing with dt=20 + Van Leer TVD (iter-183 wall-time fix after
    iter-180's WENO5 walked back). Stronger off-centering damps the
    acoustic mode at the larger outer dt without needing more
    substeps. iter-14's beta=0.0 (neutral) was sufficient at
    dt=5; the 4x dt increase needs the extra damping."""
    assert driver_defaults["--acoustic-off-centering"] == 0.2


def test_vertical_theta_diffusion_production_default(driver_defaults):
    """Explicit vertical Laplacian on theta' [m^2/s]. 0.0 = off
    (iter-14 contract).  A silent flip to 1e4-5e4 would smear
    convection vertically and inflate the cloud-fraction plateau
    above the 0.4-0.5 range the iter-105 30-day run sits in."""
    assert driver_defaults["--vertical-theta-diffusion"] == 0.0


def test_n_outer_split_default_preserves_iter183_contract(driver_defaults):
    """iter-233: --n-outer-split default MUST be '1' (the string
    '1', not int 1 — argparse default for the str-typed flag).

    Any non-'1' default would silently change the iter-183
    production contract: --n-outer-split=2 means 2 dycore substeps
    per outer step at dt/2, which is bit-different from --n-outer-split=1
    (different acoustic-substep cadence, different roundoff
    accumulation). The iter-183 30-day production run was verified
    at --n-outer-split=1 (implicit default); a default-flip would
    invalidate iter-229's DOD PASS without anyone noticing.
    """
    assert driver_defaults["--n-outer-split"] == "1"


def test_max_wind_safe_default_300(driver_defaults):
    """iter-233: --max-wind-safe default = 300.0 m/s. Covers the
    iter-223 F11 cascade ceiling (max|w|=225 m/s before NaN) with
    33%% safety margin. Used by ``--n-outer-split auto`` to pick
    the static n_split from the conservative max-wind CFL."""
    assert driver_defaults["--max-wind-safe"] == 300.0


def test_cfl_safe_default_0p4(driver_defaults):
    """iter-233: --cfl-safe default = 0.4 (SK08/FV3 conservative
    target, 2.5x margin under the formal CFL=1 limit)."""
    assert driver_defaults["--cfl-safe"] == 0.4


def test_driver_rejects_n_outer_split_non_numeric_string(tmp_path):
    """iter-234 (Codex iter-233 round-1 LOW): the driver must
    SystemExit cleanly when --n-outer-split receives an invalid
    non-numeric, non-'auto' string. Unit test of
    select_n_outer_split covers the helper-level ValueError but
    NOT the driver-level coercion path; this closes the gap.
    """
    import subprocess
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "12", "--ny", "12", "--nlev", "20",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", "0.001",
        "--semi-implicit-acoustic",
        "--acoustic-off-centering", "0.1",
        "--n-acoustic-substeps", "6",
        "--no-radiation",
        "--n-outer-split", "foo",
        "--output", str(tmp_path / "out"),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0, (
        f"Driver did not reject --n-outer-split=foo (returncode "
        f"{result.returncode}). stdout: {result.stdout[-500:]}"
    )
    combined_err = (result.stderr + result.stdout).lower()
    assert "n-outer-split" in combined_err or "n_outer_split" in combined_err, (
        f"Driver rejection message missing 'n-outer-split' marker. "
        f"stderr: {result.stderr[-500:]}"
    )


def test_driver_rejects_n_outer_split_zero(tmp_path):
    """iter-234 (Codex iter-233 round-1 LOW): the driver must
    SystemExit on --n-outer-split=0 (running zero inner steps
    would advance the state by zero per outer step — silent no-op).
    """
    import subprocess
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "12", "--ny", "12", "--nlev", "20",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", "0.001",
        "--semi-implicit-acoustic",
        "--acoustic-off-centering", "0.1",
        "--n-acoustic-substeps", "6",
        "--no-radiation",
        "--n-outer-split", "0",
        "--output", str(tmp_path / "out"),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0, (
        f"Driver did not reject --n-outer-split=0 (returncode "
        f"{result.returncode}). stdout: {result.stdout[-500:]}"
    )
    combined_err = (result.stderr + result.stdout).lower()
    assert ">= 1" in combined_err or "must be" in combined_err, (
        f"Driver rejection message for n-outer-split=0 missing "
        f"the '>= 1' / 'must be' marker. stderr: {result.stderr[-500:]}"
    )


def test_driver_rejects_n_outer_split_auto_bad_max_wind_safe(tmp_path):
    """iter-235 (Codex iter-234 round-2 LOW#1): --n-outer-split=auto
    must SystemExit cleanly when --max-wind-safe is 0 or negative.

    Pre iter-235 the helper raised raw ValueError from
    select_n_outer_split (the existing CLI finite/range validator
    runs AFTER the auto-resolve block). iter-235 wraps the helper
    in try/except ValueError -> SystemExit so a post-run analyst
    gets a clean error message.
    """
    import subprocess
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "12", "--ny", "12", "--nlev", "20",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", "0.001",
        "--semi-implicit-acoustic",
        "--acoustic-off-centering", "0.1",
        "--n-acoustic-substeps", "6",
        "--no-radiation",
        "--n-outer-split", "auto",
        "--max-wind-safe", "0.0",  # invalid
        "--output", str(tmp_path / "out"),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0, (
        f"Driver did not reject --n-outer-split=auto with "
        f"--max-wind-safe=0 (returncode {result.returncode}). "
        f"stdout: {result.stdout[-500:]}"
    )
    combined_err = (result.stderr + result.stdout).lower()
    # Either a clean iter-235 SystemExit (preferred) or the
    # iter-67/68 generic-finite-validator message; both are
    # acceptable as long as a stack trace is NOT shown.
    assert ("auto" in combined_err or "max-wind-safe" in combined_err
            or "max_wind_safe" in combined_err), (
        f"Driver auto-mode rejection should mention 'auto' or "
        f"'max-wind-safe'. stderr: {result.stderr[-500:]}"
    )
    assert "traceback" not in combined_err, (
        f"Driver should SystemExit cleanly, not raise a Python "
        f"traceback. stderr: {result.stderr[-500:]}"
    )


def test_driver_imports_select_n_outer_split_from_package():
    """iter-233: the driver must import ``select_n_outer_split``
    from ``legoesm.timestepping.split_explicit`` rather than
    re-implementing the FV3-style n_split formula inline. Locking
    the import path here so a future revert to an inline
    implementation surfaces immediately (mirrors iter-218
    build_smooth_k1_pattern pattern).

    Note: the import is FUNCTION-SCOPE deferred (inside main(),
    only fires when --n-outer-split=auto) to keep parse_args
    import-time lean. The AST check therefore looks for the
    deferred import inside any function body, not at module top.
    """
    import ast
    tree = ast.parse(DRIVER.read_text())
    target_module = "legoesm.timestepping.split_explicit"
    target_name = "select_n_outer_split"
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == target_module
        ):
            for alias in node.names:
                if alias.name == target_name:
                    return
    pytest.fail(
        f"Driver must import {target_name} from {target_module} "
        f"(iter-233 contract). A future revert to an inline "
        f"n_split formula would silently bypass the iter-233 unit "
        f"tests."
    )
