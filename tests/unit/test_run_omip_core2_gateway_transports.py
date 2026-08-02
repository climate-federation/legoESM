"""CLI round-trip + writer tests for ``--gateway-transports``.

Repo rule: a new user-tunable flag needs the flag, the wiring, AND a round-trip
test in the same change.  These also pin the OUTPUT CONTRACT (the keys written
to transports.txt) so a downstream reader cannot be broken silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parser():
    from scripts.run import run_omip_core2 as R
    return R._build_arg_parser()


def test_flag_defaults_off():
    args = _parser().parse_args(["--grid", "tripole"])
    assert hasattr(args, "gateway_transports")
    assert args.gateway_transports is False, (
        "the diagnostic must be OPT-IN so existing runs are untouched")


def test_flag_round_trips_on():
    args = _parser().parse_args(["--grid", "tripole", "--gateway-transports"])
    assert args.gateway_transports is True


def test_flag_does_not_collide_with_other_transport_options():
    """Parsing with the flag must leave every other setting at its default."""
    p = _parser()
    base = p.parse_args(["--grid", "tripole"])
    with_flag = p.parse_args(["--grid", "tripole", "--gateway-transports"])
    diffs = {k for k in vars(base)
             if getattr(base, k) != getattr(with_flag, k)}
    assert diffs == {"gateway_transports"}, f"unexpected side effects: {diffs}"


def test_writer_is_a_noop_when_accumulator_is_none(tmp_path):
    """Flag off => no accumulator => transports.txt must not be created."""
    from scripts.run import run_omip_core2 as R
    R._gateway_transport_diag(None, tmp_path, io_proc=True)
    assert not (tmp_path / "transports.txt").exists()


def test_writer_is_a_noop_when_no_steps_accumulated(tmp_path):
    from legoesm.ocean.diagnostics_sections import new_gateway_accumulator
    from scripts.run import run_omip_core2 as R
    acc = new_gateway_accumulator(("bering_pacific", "davis_caa"))
    assert acc.n == 0
    R._gateway_transport_diag(acc, tmp_path, io_proc=True)
    assert not (tmp_path / "transports.txt").exists()


def test_writer_appends_expected_keys_and_preserves_existing_content(tmp_path):
    """It must APPEND (like the ACC/MHT diags), never truncate the AMOC line."""
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import GatewayAccumulator
    from scripts.run import run_omip_core2 as R

    (tmp_path / "transports.txt").write_text("amoc26N_Sv 1.2345\n")
    names = ("bering_pacific", "atlantic_nordic")
    # 2 accumulated steps; volume sums 2e6 and 6e6 m^3/s -> means 1 and 3 Sv
    acc = GatewayAccumulator(names,
                             jnp.asarray([2.0e6, 6.0e6]),
                             jnp.asarray([4.0e6, 8.0e6]), 2)
    R._gateway_transport_diag(acc, tmp_path, io_proc=True)
    txt = (tmp_path / "transports.txt").read_text()
    assert "amoc26N_Sv 1.2345" in txt, "the existing content was truncated"
    assert "gateway_n_steps 2" in txt
    assert "gateway_bering_pacific_vol_Sv 1.000000" in txt
    assert "gateway_atlantic_nordic_vol_Sv 3.000000" in txt
    assert "gateway_bering_pacific_salt_psu_m3s" in txt


def test_writer_respects_io_proc_false(tmp_path):
    """Non-IO ranks must not write (mirrors the sibling transport diags)."""
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import GatewayAccumulator
    from scripts.run import run_omip_core2 as R

    acc = GatewayAccumulator(("davis_caa",), jnp.asarray([1.0e6]),
                             jnp.asarray([1.0]), 1)
    R._gateway_transport_diag(acc, tmp_path, io_proc=False)
    assert not (tmp_path / "transports.txt").exists()


def test_writer_never_raises_on_a_malformed_accumulator(tmp_path):
    """Like its siblings, the diagnostic must not be able to kill a run."""
    from scripts.run import run_omip_core2 as R

    class _Broken:
        n = 3

        def as_dict(self):
            raise RuntimeError("boom")

    R._gateway_transport_diag(_Broken(), tmp_path, io_proc=True)  # must not raise


@pytest.mark.parametrize("name", [nm for nm, _ in __import__(
    "legoesm.ocean.diagnostics_sections", fromlist=["x"]).ARCTIC_GATEWAYS])
def test_every_gateway_name_is_filename_safe(name):
    """Names become transports.txt keys, so they must be plain identifiers."""
    assert name.replace("_", "").isalnum(), name
    assert name == name.lower()


# =====================================================================
# codex M10 — the ONE-STEP BUILT-MODEL test.
#
# The two bugs that 36 green unit tests could not see (guard placed AFTER the
# `use_scan` early return; app_grid_type string that never matched) were both
# WIRING bugs.  These exercise the wiring itself.
# =====================================================================


def _tiny_latlon():
    """Smallest workable lat-lon C-grid model + state (same API the driver
    uses via run_omip._create_setup)."""
    from scripts.run import run_omip
    grid, z_coord, config, model, _kind = run_omip._create_setup(
        grid_type="latlon", resolution="16x32", nlev=4, H_max=1000.0,
        physics_preset="minimal", water_type="II")
    state = run_omip._init_rest_state("latlon", grid, z_coord, 1000.0)
    return grid, z_coord, model, state


def test_one_step_driver_chain_writes_n_steps_1_and_every_gateway_key(tmp_path):
    """Built model -> setup -> ONE model.step -> gateway_step -> writer.

    Asserts `gateway_n_steps 1` and EVERY gateway output key, exactly as codex
    required.  This is the chain main() runs, entered through the same helper
    functions main() imports.
    """
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import (
        ARCTIC_GATEWAYS, gateway_step, setup_gateway_accumulator,
    )
    from scripts.run import run_omip_core2 as R

    grid, z_coord, model, state = _tiny_latlon()
    lat2d = jnp.asarray(np.degrees(np.asarray(grid.lat))[:, None]
                        * np.ones((1, grid.n_lon)))
    lon2d = jnp.asarray(np.degrees(np.asarray(grid.lon))[None, :]
                        * np.ones((grid.n_lat, 1)))
    acc, stack, faces = setup_gateway_accumulator(
        lat2d, lon2d, state.land_mask.data, lat_min_deg=0.0)
    assert int(jnp.sum(faces.u_sel)) + int(jnp.sum(faces.v_sel)) > 0, (
        "the test grid produced no boundary faces -- it cannot detect anything")
    assert acc.n == 0

    state1 = model.step(state, 600.0)
    acc = gateway_step(acc, stack, state1, z_coord, grid)
    assert acc.n == 1, "the accumulator did not advance -- it is not wired"

    R._gateway_transport_diag(acc, tmp_path, io_proc=True)
    txt = (tmp_path / "transports.txt").read_text()
    assert "gateway_n_steps 1" in txt
    for name, _bins in ARCTIC_GATEWAYS:
        assert f"gateway_{name}_vol_Sv" in txt, f"missing volume key for {name}"
        assert f"gateway_{name}_salt_psu_m3s" in txt, f"missing salt key {name}"
    # every value must be finite -- a NaN would still "write all the keys"
    for line in txt.splitlines():
        if line.startswith("gateway_") and "n_steps" not in line:
            assert np.isfinite(float(line.split()[1])), f"non-finite: {line}"


def test_one_step_chain_is_a_pure_diagnostic(tmp_path):
    """gateway_step must not perturb the state it is handed."""
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import (
        gateway_step, setup_gateway_accumulator,
    )

    grid, z_coord, model, state = _tiny_latlon()
    lat2d = jnp.asarray(np.degrees(np.asarray(grid.lat))[:, None]
                        * np.ones((1, grid.n_lon)))
    lon2d = jnp.asarray(np.degrees(np.asarray(grid.lon))[None, :]
                        * np.ones((grid.n_lat, 1)))
    acc, stack, _f = setup_gateway_accumulator(
        lat2d, lon2d, state.land_mask.data, lat_min_deg=0.0)
    s1 = model.step(state, 600.0)
    S_before = np.asarray(s1.S.data).copy()
    u_before = np.asarray(s1.u.data).copy()
    gateway_step(acc, stack, s1, z_coord, grid)
    np.testing.assert_array_equal(np.asarray(s1.S.data), S_before)
    np.testing.assert_array_equal(np.asarray(s1.u.data), u_before)
    # and stepping again from s1 is unaffected
    a = np.asarray(model.step(s1, 600.0).S.data)
    gateway_step(acc, stack, s1, z_coord, grid)
    b = np.asarray(model.step(s1, 600.0).S.data)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------- the lane guards


def test_lane_guard_rejects_scan_block():
    """H3: the scan path returns before the loop, so the flag must be refused."""
    from legoesm.ocean.diagnostics_sections import validate_gateway_lanes
    with pytest.raises(ValueError, match="scan-block"):
        validate_gateway_lanes(use_scan=True, spmd_persistent=False,
                               app_grid_type="tripole")


def test_lane_guard_rejects_spmd_persistent():
    from legoesm.ocean.diagnostics_sections import validate_gateway_lanes
    with pytest.raises(ValueError, match="spmd-persistent-state"):
        validate_gateway_lanes(use_scan=False, spmd_persistent=True,
                               app_grid_type="tripole")


@pytest.mark.parametrize("bad", ["cubed_sphere", "mpas", "latlon_bathy"])
def test_lane_guard_rejects_unsupported_app_grid_types(bad):
    """H4: 'latlon_bathy' is the CLI value, NOT the resolved app_grid_type --
    accepting it silently disabled the real lat-lon grid."""
    from legoesm.ocean.diagnostics_sections import validate_gateway_lanes
    with pytest.raises(ValueError, match="app_grid_type"):
        validate_gateway_lanes(use_scan=False, spmd_persistent=False,
                               app_grid_type=bad)


@pytest.mark.parametrize("good", ["tripole", "latlon"])
def test_lane_guard_accepts_the_grids_the_driver_actually_sets(good):
    """Pins the strings against run_omip_core2's own app_grid_type literals."""
    from legoesm.ocean.diagnostics_sections import validate_gateway_lanes
    validate_gateway_lanes(use_scan=False, spmd_persistent=False,
                           app_grid_type=good)


def test_driver_sets_exactly_the_app_grid_types_the_guard_accepts():
    """If the driver gains a new app_grid_type, the guard must be updated.

    Reads the DRIVER SOURCE for its `app_grid_type = "..."` literals so a new
    grid cannot silently fall outside the supported set.
    """
    import re

    from legoesm.ocean.diagnostics_sections import SUPPORTED_APP_GRIDS
    src = (Path(_ROOT) / "scripts/run/run_omip_core2.py").read_text()
    literals = set(re.findall(r'app_grid_type = "([a-z_]+)"', src))
    assert literals, "no app_grid_type literals found -- the regex rotted"
    assert set(SUPPORTED_APP_GRIDS) <= literals, (
        f"the guard accepts {set(SUPPORTED_APP_GRIDS) - literals}, which the "
        "driver never sets -- the flag would be dead for those grids")


def test_guard_call_precedes_the_use_scan_early_return_in_the_driver():
    """H3 was an ORDERING bug: the guard sat AFTER `if use_scan:` returns.

    Naming the exact symbols and asserting their ORDER in the driver source is
    the only cheap way to pin an ordering constraint inside a monolithic
    main().  It fails if the guard is ever moved back below the early return.
    """
    src = (Path(_ROOT) / "scripts/run/run_omip_core2.py").read_text()
    lines = src.splitlines()
    guard = next(i for i, ln in enumerate(lines)
                 if "validate_gateway_lanes(" in ln and "import" not in ln)
    scan = next(i for i, ln in enumerate(lines) if ln == "    if use_scan:")
    assert guard < scan, (
        f"validate_gateway_lanes is called at line {guard + 1}, AFTER the "
        f"`if use_scan:` early return at line {scan + 1} -- it would never "
        "execute and --gateway-transports would be silently ignored")


def test_driver_calls_the_accumulator_inside_the_step_loop():
    """The per-step call must be INSIDE `for step in range(...)`, after it."""
    src = (Path(_ROOT) / "scripts/run/run_omip_core2.py").read_text()
    lines = src.splitlines()
    # Match the loop by its END bound, not its start: the start is
    # `start_step + 1` once --restart-from can resume mid-integration, and
    # pinning the literal `range(1, ...)` made this test fail on a change that
    # did not move the accumulator at all.
    loop = next(i for i, ln in enumerate(lines)
                if ln.strip().startswith("for step in range(")
                and "n_steps + 1)" in ln)
    call = next(i for i, ln in enumerate(lines)
                if "_gw_acc = gateway_step(" in ln)
    write = next(i for i, ln in enumerate(lines)
                 if "_gateway_transport_diag(_gw_acc" in ln)
    assert loop < call < write, (
        "the accumulation must sit inside the step loop and before the writer")
