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
        ARCTIC_GATEWAYS, gateway_step, promote_gateway_geometry,
        setup_gateway_accumulator,
    )
    from scripts.run import run_omip_core2 as R

    grid, z_coord, model, state = _tiny_latlon()
    geom = promote_gateway_geometry(getattr(model, "grid", grid))
    lat2d = jnp.asarray(np.degrees(np.asarray(grid.lat))[:, None]
                        * np.ones((1, grid.n_lon)))
    lon2d = jnp.asarray(np.degrees(np.asarray(grid.lon))[None, :]
                        * np.ones((grid.n_lat, 1)))
    acc, stack, faces = setup_gateway_accumulator(
        lat2d, lon2d, state.land_mask.data, lat_min_deg=0.0)
    assert int(jnp.sum(faces.u_sel)) + int(jnp.sum(faces.v_sel)) > 0, (
        "the test grid produced no boundary faces -- it cannot detect anything")
    assert acc.n == 0

    # The rest state has u = v = 0, so EVERY gateway transport would be
    # exactly zero and a finiteness-only assertion would pass without the
    # transport code doing anything at all (codex round-3 finding 2).  Seed a
    # known uniform NORTHWARD v so the expected SIGN is known a priori: the
    # region is lat >= 0, so northward flow is INTO it, i.e. POSITIVE.
    state1 = model.step(state, 600.0)
    v0 = 0.05
    state1 = state1._replace(
        v=state1.v.replace(data=jnp.full_like(state1.v.data, v0)))
    acc = gateway_step(acc, stack, state1, z_coord, geom)
    assert acc.n == 1, "the accumulator did not advance -- it is not wired"

    total = float(jnp.sum(acc.volume))
    assert total > 0.0, (
        f"uniform northward v={v0} m/s must give a POSITIVE (into-region) "
        f"total volume transport; got {total:.6e} m^3/s")
    assert float(jnp.max(jnp.abs(acc.volume))) > 0.0, (
        "no single gateway carried any transport")

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
    # ... and at least one WRITTEN volume must be non-zero, so the file itself
    # (not just the in-memory accumulator) carries a real number.
    vols = [float(ln.split()[1]) for ln in txt.splitlines()
            if ln.startswith("gateway_") and ln.split()[0].endswith("_vol_Sv")]
    assert any(v != 0.0 for v in vols), f"every written volume was zero: {vols}"


def test_gateway_volume_is_signed_and_linear_in_velocity():
    """Flip v -> -v flips every gateway sign; 2v doubles every gateway.

    A transport diagnostic that returned |flux|, dropped the orientation, or
    normalised by something velocity-dependent would pass the smoke checks
    above and fail here.
    """
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import (
        gateway_step, promote_gateway_geometry, setup_gateway_accumulator,
    )

    grid, z_coord, model, state = _tiny_latlon()
    geom = promote_gateway_geometry(getattr(model, "grid", grid))
    lat2d = jnp.asarray(np.degrees(np.asarray(grid.lat))[:, None]
                        * np.ones((1, grid.n_lon)))
    lon2d = jnp.asarray(np.degrees(np.asarray(grid.lon))[None, :]
                        * np.ones((grid.n_lat, 1)))
    acc0, stack, _f = setup_gateway_accumulator(
        lat2d, lon2d, state.land_mask.data, lat_min_deg=0.0)
    s1 = model.step(state, 600.0)

    def _vol(scale):
        st = s1._replace(
            v=s1.v.replace(data=jnp.full_like(s1.v.data, 0.05 * scale)))
        return np.asarray(gateway_step(acc0, stack, st, z_coord, geom).volume)

    base = _vol(1.0)
    assert np.any(base != 0.0), "the seeded flow produced no transport at all"
    np.testing.assert_allclose(_vol(-1.0), -base, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(_vol(2.0), 2.0 * base, rtol=1e-10, atol=0.0)


def test_gateway_step_rejects_an_unpromoted_grid():
    """The per-step promotion was removed; a raw LatLonGrid must RAISE.

    Non-vacuity: ``_tiny_latlon`` returns exactly such a raw grid, and it is
    what the driver used to pass.  Silently re-promoting it every step both
    rebuilt every metric array and defaulted metric_convention to "exact".
    """
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import (
        gateway_step, setup_gateway_accumulator,
    )

    grid, z_coord, model, state = _tiny_latlon()
    assert not hasattr(grid, "dy_u"), (
        "this test needs an UNPROMOTED grid to be meaningful")
    lat2d = jnp.asarray(np.degrees(np.asarray(grid.lat))[:, None]
                        * np.ones((1, grid.n_lon)))
    lon2d = jnp.asarray(np.degrees(np.asarray(grid.lon))[None, :]
                        * np.ones((grid.n_lat, 1)))
    acc, stack, _f = setup_gateway_accumulator(
        lat2d, lon2d, state.land_mask.data, lat_min_deg=0.0)
    with pytest.raises(TypeError, match="promoted LatLonCGridGeometry"):
        gateway_step(acc, stack, state, z_coord, grid)


def test_one_step_chain_is_a_pure_diagnostic(tmp_path):
    """gateway_step must not perturb the state it is handed."""
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import (
        gateway_step, promote_gateway_geometry, setup_gateway_accumulator,
    )

    grid, z_coord, model, state = _tiny_latlon()
    geom = promote_gateway_geometry(getattr(model, "grid", grid))
    lat2d = jnp.asarray(np.degrees(np.asarray(grid.lat))[:, None]
                        * np.ones((1, grid.n_lon)))
    lon2d = jnp.asarray(np.degrees(np.asarray(grid.lon))[None, :]
                        * np.ones((grid.n_lat, 1)))
    acc, stack, _f = setup_gateway_accumulator(
        lat2d, lon2d, state.land_mask.data, lat_min_deg=0.0)
    s1 = model.step(state, 600.0)
    S_before = np.asarray(s1.S.data).copy()
    u_before = np.asarray(s1.u.data).copy()
    gateway_step(acc, stack, s1, z_coord, geom)
    np.testing.assert_array_equal(np.asarray(s1.S.data), S_before)
    np.testing.assert_array_equal(np.asarray(s1.u.data), u_before)
    # and stepping again from s1 is unaffected
    a = np.asarray(model.step(s1, 600.0).S.data)
    gateway_step(acc, stack, s1, z_coord, geom)
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
    loop = next(i for i, ln in enumerate(lines)
                if ln.strip().startswith("for step in range(1, n_steps"))
    call = next(i for i, ln in enumerate(lines)
                if "_gw_acc = gateway_step(" in ln)
    write = next(i for i, ln in enumerate(lines)
                 if "_gateway_transport_diag(_gw_acc" in ln)
    assert loop < call < write, (
        "the accumulation must sit inside the step loop and before the writer")
    # The geometry must be promoted ONCE, BEFORE the loop, and the per-step
    # call must pass that promoted object -- not the raw grid.  Promoting per
    # step rebuilt every metric array and defaulted the metric convention to
    # "exact" regardless of the run's config (codex round-3 finding 1).
    promote = next(i for i, ln in enumerate(lines)
                   if "_gw_geom = promote_gateway_geometry(" in ln)
    assert promote < loop, (
        "promote_gateway_geometry must run BEFORE the step loop, not in it")
    assert "_gw_geom" in "".join(lines[call:call + 6]), (
        "the per-step gateway_step call must be handed the promoted _gw_geom")
    # The promotion must take the MODEL's convention, not the "exact" default.
    # A live one-step test cannot catch a regression here because the test
    # model's convention IS "exact" (codex round-4 finding 3), so pin the
    # argument itself -- and test_promote_geometry_honours_metric_convention
    # below proves the argument is load-bearing rather than decorative.
    promo_src = "".join(lines[promote:promote + 5])
    assert ("model.grid" in promo_src
            or 'getattr(model, "grid"' in promo_src), (
        "promotion must start from model.grid (already built with "
        f"config.metric_convention), not the raw grid; got: {promo_src!r}")
    assert "metric_convention" in promo_src, (
        "promotion must forward the model's metric_convention")
    # The per-step call must sit inside a non-fatal boundary: a DIAGNOSTIC
    # must never abort a production run (codex round-4 RED).
    #
    # Located by SEARCHING for the enclosing try/except rather than by a fixed
    # +/-N line window: the window version broke the moment the call gained one
    # more keyword argument (#1442's `source="stored"`), reporting a missing
    # non-fatal boundary that was still right there.  A structural assertion
    # must not be sensitive to the LENGTH of the call it brackets.
    opens = [i for i in range(call, -1, -1) if lines[i].strip() == "try:"]
    assert opens, "gateway_step is not inside any try: block"
    try_at = opens[0]
    indent = len(lines[try_at]) - len(lines[try_at].lstrip())
    closes = [i for i in range(call, len(lines))
              if lines[i].strip().startswith("except")
              and len(lines[i]) - len(lines[i].lstrip()) == indent]
    assert closes, (
        "gateway_step must be wrapped in a non-fatal try/except; it raises on "
        "a bad geometry and that raise sits inside the step loop")
    assert try_at < call < closes[0], (
        f"gateway_step (line {call + 1}) is not between its try: "
        f"(line {try_at + 1}) and its except (line {closes[0] + 1})")


def test_promote_geometry_honours_metric_convention():
    """The metric_convention argument must change the metrics it produces.

    Without this, `test_driver_calls_the_accumulator_inside_the_step_loop`'s
    assertion that the driver forwards `metric_convention` would be checking
    a decorative keyword.
    """
    import numpy as _np

    from legoesm.ocean.diagnostics_sections import promote_gateway_geometry

    grid, _z, _model, _state = _tiny_latlon()
    assert not hasattr(grid, "dy_u"), (
        "the convention is only applied when converting a RAW LatLonGrid, so "
        "this test needs an unpromoted grid to be meaningful")
    exact = promote_gateway_geometry(grid, metric_convention="exact")
    nemo = promote_gateway_geometry(grid, metric_convention="nemo_isotropic")
    # dy_u and dx_v are precisely the two metrics section_transport integrates
    # with, so it is those that have to respond to the convention.
    differs = [nm for nm in ("dy_u", "dx_v")
               if not _np.allclose(_np.asarray(getattr(exact, nm)),
                                   _np.asarray(getattr(nemo, nm)))]
    assert differs, (
        "neither dy_u nor dx_v responded to metric_convention -- the "
        "convention the driver forwards would be decorative for this "
        "diagnostic, and codex round-3 finding 1 would be moot")


# =====================================================================
# CUMULATIVE per-cadence dump (gateway_transports.csv)
#
# The run-end transports.txt block is a WHOLE-RUN mean and cannot separate the
# cold-start adjustment from the settled window.  The CSV dumps the running
# TOTALS so any window is (cumsum_b - cumsum_a) / (n_b - n_a).
#
# Every test below either (a) asserts its own fixture is non-degenerate before
# asserting the property, or (b) computes the value the defect it guards would
# produce and asserts the real value differs.
# =====================================================================


def _read_csv(path):
    """(header, [row dicts]) with the ragged-row check the parse depends on."""
    import csv as _csv
    with open(path, newline="") as fh:
        rd = _csv.reader(fh)
        header = next(rd)
        rows = []
        for r in rd:
            assert len(r) == len(header), (
                f"ragged CSV row: {len(r)} cells for {len(header)} columns")
            rows.append(dict(zip(header, r)))
    return header, rows


def _seeded_states(model, state, values, dt=600.0):
    """One post-step state per entry in ``values``, with v pinned to it.

    The rest state has u = v = 0, so EVERY gateway transport would be exactly
    zero and every assertion below would hold trivially (codex round-3 finding
    2).  Pinning v per step both makes the transports non-zero and makes the
    two windows carry DIFFERENT, known signals.
    """
    import jax.numpy as jnp
    out = []
    s = state
    for val in values:
        s = model.step(s, dt)
        out.append(s._replace(
            v=s.v.replace(data=jnp.full_like(s.v.data, val))))
    return out


def _gw_setup(lat_min_deg=0.0):
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import (
        promote_gateway_geometry, setup_gateway_accumulator,
    )
    grid, z_coord, model, state = _tiny_latlon()
    geom = promote_gateway_geometry(getattr(model, "grid", grid))
    lat2d = jnp.asarray(np.degrees(np.asarray(grid.lat))[:, None]
                        * np.ones((1, grid.n_lon)))
    lon2d = jnp.asarray(np.degrees(np.asarray(grid.lon))[None, :]
                        * np.ones((grid.n_lat, 1)))
    acc, stack, faces = setup_gateway_accumulator(
        lat2d, lon2d, state.land_mask.data, lat_min_deg=lat_min_deg)
    assert int(jnp.sum(faces.u_sel)) + int(jnp.sum(faces.v_sel)) > 0, (
        "the test grid produced no boundary faces -- it cannot detect anything")
    return grid, z_coord, model, state, geom, acc, stack


# ---------------------------------------------------------------- open/close


def test_open_writes_a_header_naming_every_gateway(tmp_path):
    from legoesm.ocean.diagnostics_sections import ARCTIC_GATEWAYS
    from scripts.run import run_omip_core2 as R

    names = tuple(nm for nm, _b in ARCTIC_GATEWAYS)
    csv = R._gateway_cumulative_open(tmp_path, names, io_proc=True)
    assert csv is not None
    R._gateway_cumulative_close(csv)
    header, rows = _read_csv(tmp_path / R.GATEWAY_CUMULATIVE_CSV)
    assert rows == [], "no dump was requested, so there must be no data rows"
    assert header[:3] == ["step", "day", "n_steps"]
    for nm in names:
        assert f"{nm}_vol_cumsum_m3s" in header
        assert f"{nm}_salt_cumsum_psu_m3s" in header


def test_open_is_a_noop_on_a_non_io_rank(tmp_path):
    """Mirrors the sibling diags: N processes must not clobber one file."""
    from scripts.run import run_omip_core2 as R
    assert R._gateway_cumulative_open(tmp_path, ("davis_caa",),
                                      io_proc=False) is None
    assert not (tmp_path / R.GATEWAY_CUMULATIVE_CSV).exists()


def test_row_and_close_are_noops_when_the_csv_was_never_opened(tmp_path):
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import GatewayAccumulator
    from scripts.run import run_omip_core2 as R

    acc = GatewayAccumulator(("davis_caa",), jnp.asarray([1.0]),
                             jnp.asarray([2.0]), 1)
    R._gateway_cumulative_row(None, acc, 5, 1.0)     # must not raise
    R._gateway_cumulative_close(None)                # must not raise
    assert not (tmp_path / R.GATEWAY_CUMULATIVE_CSV).exists()


def test_row_is_a_noop_when_the_accumulator_was_disabled_mid_run(tmp_path):
    """gateway_step failing sets _gw_acc=None; later dumps must be skipped."""
    from scripts.run import run_omip_core2 as R
    csv = R._gateway_cumulative_open(tmp_path, ("davis_caa",), io_proc=True)
    R._gateway_cumulative_row(csv, None, 7, 2.0)
    R._gateway_cumulative_close(csv)
    _h, rows = _read_csv(tmp_path / R.GATEWAY_CUMULATIVE_CSV)
    assert rows == [], "a row was written for a disabled accumulator"


def test_row_refuses_to_write_columns_that_would_be_mis_assigned(tmp_path):
    """Header built from names A, accumulator carrying names B => NO row.

    Writing it anyway would put davis_caa's transport under bering_pacific's
    column -- a defect that reads as a physics result.
    """
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import GatewayAccumulator
    from scripts.run import run_omip_core2 as R

    csv = R._gateway_cumulative_open(tmp_path, ("bering_pacific", "davis_caa"),
                                     io_proc=True)
    swapped = GatewayAccumulator(("davis_caa", "bering_pacific"),
                                 jnp.asarray([1.0, 2.0]),
                                 jnp.asarray([3.0, 4.0]), 1)
    R._gateway_cumulative_row(csv, swapped, 1, 0.1)   # must not raise
    R._gateway_cumulative_close(csv)
    _h, rows = _read_csv(tmp_path / R.GATEWAY_CUMULATIVE_CSV)
    assert rows == [], "a mis-assigned row was written"


def test_row_never_raises_on_a_malformed_accumulator(tmp_path):
    """Like its siblings, the dump must not be able to kill a run."""
    from scripts.run import run_omip_core2 as R

    class _Broken:
        names = ("davis_caa",)
        n = 3

        @property
        def volume(self):
            raise RuntimeError("boom")

    csv = R._gateway_cumulative_open(tmp_path, ("davis_caa",), io_proc=True)
    R._gateway_cumulative_row(csv, _Broken(), 3, 1.0)   # must not raise
    R._gateway_cumulative_close(csv)
    _h, rows = _read_csv(tmp_path / R.GATEWAY_CUMULATIVE_CSV)
    assert rows == []


# ------------------------------------------------- the windowed-mean recovery


def test_differencing_two_dumps_equals_a_fresh_accumulator_over_that_window(
        tmp_path):
    """THE load-bearing test: the reason the dump is cumulative.

    Drives the REAL model + the REAL gateway_step over two windows with
    different flow, dumps a row at the window boundary and at the end, then
    checks that differencing the two rows reproduces, to round-off, a SECOND
    accumulator that was started fresh at the boundary and fed the identical
    states.

    Three non-degeneracy assertions run BEFORE the property assertion:
      * the seeded flow produces a non-zero transport at all,
      * the two dumps DIFFER,
      * the windowed mean differs from the whole-run mean by a wide margin
        (otherwise a writer that emitted the whole-run mean would pass).
    """
    from legoesm.ocean.diagnostics_sections import (
        gateway_step, new_gateway_accumulator,
    )
    from scripts.run import run_omip_core2 as R

    grid, z_coord, model, state, geom, acc, stack = _gw_setup()
    # Window A: 3 steps of northward (INTO the region) flow.
    # Window B: 2 steps of weaker SOUTHWARD flow -- a different sign and a
    # different magnitude, so the two windows are unmistakably distinct.
    win_a = _seeded_states(model, state, [0.05, 0.05, 0.05])
    win_b = _seeded_states(model, win_a[-1], [-0.02, -0.02])
    n_a, n_b = len(win_a), len(win_b)

    csv = R._gateway_cumulative_open(tmp_path, acc.names, io_proc=True)
    for s in win_a:
        acc = gateway_step(acc, stack, s, z_coord, geom, source="auto")
    assert acc.n == n_a
    R._gateway_cumulative_row(csv, acc, n_a, float(n_a))
    # A fresh accumulator over window B ONLY -- the independent reference.
    ref = new_gateway_accumulator(acc.names)
    for s in win_b:
        acc = gateway_step(acc, stack, s, z_coord, geom, source="auto")
        ref = gateway_step(ref, stack, s, z_coord, geom, source="auto")
    R._gateway_cumulative_row(csv, acc, n_a + n_b, float(n_a + n_b))
    R._gateway_cumulative_close(csv)

    _h, rows = _read_csv(tmp_path / R.GATEWAY_CUMULATIVE_CSV)
    assert len(rows) == 2, f"expected two dumps, got {len(rows)}"
    ra, rb = rows
    assert int(ra["n_steps"]) == n_a and int(rb["n_steps"]) == n_a + n_b

    ref_mean = np.asarray(ref.volume) / n_b
    ref_salt_mean = np.asarray(ref.tracer) / n_b
    assert np.any(ref_mean != 0.0), (
        "the seeded flow produced no transport -- every assertion below would "
        "be trivially satisfied")
    assert np.any(ref_salt_mean != 0.0), (
        "the seeded flow carried no salt -- the salt columns would be checked "
        "against zero and a corrupted salt column would pass")
    # Differencing two cumulative sums cancels the shared prefix ANALYTICALLY
    # but not in floating point, so the tolerance has to track the storage
    # precision: new_gateway_accumulator falls back to float32 whenever x64 is
    # off (e.g. an --fp32 run), where eps is 1.2e-7, not 2.2e-16.  Stating it
    # here beats a magic rtol that silently passes for the wrong reason.
    _f64 = np.asarray(acc.volume).dtype == np.float64
    rtol = 1e-9 if _f64 else 1e-4

    n_win = int(rb["n_steps"]) - int(ra["n_steps"])
    assert n_win == n_b, f"the window spans {n_win} steps, expected {n_b}"
    checked = 0
    for i, nm in enumerate(acc.names):
        # BOTH columns (codex round-1 YELLOW 5): a corruption confined to the
        # salt column would pass a volume-only check, and the salt transport is
        # the quantity the Arctic budget question actually needs.
        for key, ref_arr in ((f"{nm}_vol_cumsum_m3s", ref_mean),
                             (f"{nm}_salt_cumsum_psu_m3s", ref_salt_mean)):
            if ref_arr[i] == 0.0:
                continue                 # dry gateway on this tiny grid
            ca, cb = float(ra[key]), float(rb[key])
            # non-degeneracy 1: the dumps actually moved
            assert ca != cb, f"{key}: the two dumps are identical"
            window = (cb - ca) / n_win
            whole = cb / int(rb["n_steps"])
            # non-degeneracy 2: the window is not the whole run in disguise
            assert abs(window - whole) > 0.1 * max(abs(whole), abs(window)), (
                f"{key}: window {window:.6e} and whole-run {whole:.6e} means "
                "are too close for this test to discriminate them")
            # the property
            np.testing.assert_allclose(
                window, ref_arr[i], rtol=rtol, atol=0.0,
                err_msg=f"{key}: differenced window mean != a fresh "
                        "accumulator over the same window")
            checked += 1
    assert checked >= 2, (
        f"only {checked} columns were non-zero -- the test did not exercise "
        "both the volume and the salt path")


def test_the_final_row_reproduces_the_transports_txt_whole_run_mean(tmp_path):
    """The CSV and transports.txt must be two views of the SAME numbers.

    cumsum / n_steps / 1e6 is exactly the Sv the run-end block prints, so a
    reader can cross-check one against the other.  Non-vacuity: the run-end
    value is asserted non-zero first.
    """
    from legoesm.ocean.diagnostics_sections import gateway_step
    from scripts.run import run_omip_core2 as R

    grid, z_coord, model, state, geom, acc, stack = _gw_setup()
    for s in _seeded_states(model, state, [0.05, -0.03, 0.04]):
        acc = gateway_step(acc, stack, s, z_coord, geom, source="auto")
    csv = R._gateway_cumulative_open(tmp_path, acc.names, io_proc=True)
    R._gateway_cumulative_row(csv, acc, acc.n, float(acc.n))
    R._gateway_cumulative_close(csv)
    R._gateway_transport_diag(acc, tmp_path, io_proc=True)

    _h, rows = _read_csv(tmp_path / R.GATEWAY_CUMULATIVE_CSV)
    txt = dict(ln.split() for ln in
               (tmp_path / "transports.txt").read_text().splitlines())
    assert int(txt["gateway_n_steps"]) == int(rows[-1]["n_steps"]) == acc.n
    nonzero = 0
    for nm in acc.names:
        sv_txt = float(txt[f"gateway_{nm}_vol_Sv"])
        sv_csv = float(rows[-1][f"{nm}_vol_cumsum_m3s"]) / acc.n / 1.0e6
        # transports.txt writes %.6f, so compare at that resolution
        assert sv_csv == pytest.approx(sv_txt, abs=5e-7), (
            f"{nm}: CSV {sv_csv} vs transports.txt {sv_txt}")
        nonzero += int(sv_txt != 0.0)
    assert nonzero, "every gateway was zero -- the comparison proves nothing"


def test_the_cumulative_dump_leaves_transports_txt_byte_identical(tmp_path):
    """The run-end output contract must not move: same bytes, CSV or no CSV."""
    from legoesm.ocean.diagnostics_sections import gateway_step
    from scripts.run import run_omip_core2 as R

    grid, z_coord, model, state, geom, acc, stack = _gw_setup()
    states = _seeded_states(model, state, [0.05, -0.03])

    control = tmp_path / "control"
    control.mkdir()
    a = acc
    for s in states:
        a = gateway_step(a, stack, s, z_coord, geom, source="auto")
    R._gateway_transport_diag(a, control, io_proc=True)

    treated = tmp_path / "treated"
    treated.mkdir()
    b = acc
    csv = R._gateway_cumulative_open(treated, acc.names, io_proc=True)
    for i, s in enumerate(states, start=1):
        b = gateway_step(b, stack, s, z_coord, geom, source="auto")
        R._gateway_cumulative_row(csv, b, i, float(i))
    R._gateway_cumulative_close(csv)
    R._gateway_transport_diag(b, treated, io_proc=True)

    ctl = (control / "transports.txt").read_bytes()
    trt = (treated / "transports.txt").read_bytes()
    assert ctl, "the control wrote nothing -- the comparison is vacuous"
    assert ctl == trt, "the cumulative dump changed transports.txt"
    _h, rows = _read_csv(treated / R.GATEWAY_CUMULATIVE_CSV)
    assert len(rows) == len(states), "the treated arm did not actually dump"


# ---------------------------------------------------- non-perturbation (SHA)


_MASS_FLUX_SLOTS = ("mass_flux_u", "mass_flux_v", "mass_flux_w")


def _digest(state) -> str:
    """SHA-256 of the state with the #1442 mass-flux slots blanked.

    --gateway-transports turns ``store_mass_flux`` ON, which populates three
    slots that are ``None`` with the flag off.  Blanking them makes the two
    arms structurally identical, so the digest compares the TRAJECTORY -- the
    question this test asks -- rather than trivially reporting the presence of
    the new slots.  ``store_mass_flux``'s own non-perturbation is pinned
    separately by tests/ocean/unit/test_mass_flux_store.py.
    """
    from legoesm.io.state_digest import pytree_state_digest
    return pytree_state_digest(
        state._replace(**{k: None for k in _MASS_FLUX_SLOTS}))


def test_the_digest_used_below_is_actually_value_sensitive():
    """Without this, the non-perturbation test could pass on a constant hash."""
    import jax.numpy as jnp

    _g, _z, model, state = _tiny_latlon()
    s1 = model.step(state, 600.0)
    s2 = s1._replace(S=s1.S.replace(data=s1.S.data + jnp.asarray(1e-9)))
    assert _digest(s1) == _digest(s1), "the digest is not deterministic"
    assert _digest(s1) != _digest(s2), (
        "a 1e-9 perturbation did not change the digest -- it cannot detect "
        "a perturbed trajectory either")


def test_accumulating_and_dumping_does_not_perturb_the_trajectory(tmp_path):
    """Feature OFF vs ON: identical SHA-256 state digests after N steps.

    ON exercises the FULL chain the driver runs -- store_mass_flux, the
    per-step gateway_step, and a cumulative CSV dump after every step -- so a
    dump that pulled, cast, or wrote back any leaf would show up here.  N > 1
    so a per-step perturbation compounds instead of cancelling.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.diagnostics_sections import gateway_step
    from scripts.run import run_omip, run_omip_core2 as R

    n_steps, dt = 3, 600.0

    def _build(store):
        grid, z_coord, config, model, _kind = run_omip._create_setup(
            grid_type="latlon", resolution="16x32", nlev=4, H_max=1000.0,
            physics_preset="minimal", water_type="II")
        if store:
            config = config._replace(store_mass_flux=True)
            model = LatLonCGridOceanModel(grid, z_coord, config)
        state = run_omip._init_rest_state("latlon", grid, z_coord, 1000.0)
        return grid, z_coord, model, state

    # --- OFF: no flag at all
    _g0, _z0, m_off, s_off = _build(store=False)
    # The two arms must be compared from the SAME initial condition, so capture
    # the INITIAL digest before stepping.  (An earlier draft compared the OFF
    # arm's FINAL state against the ON arm's INITIAL state and went red for
    # that reason alone -- a fixture bug that looked exactly like the defect
    # the test hunts.)
    d_init_off = _digest(s_off)
    for _ in range(n_steps):
        s_off = m_off.step(s_off, dt)
    assert _digest(s_off) != d_init_off, (
        "the OFF arm did not move over 3 steps -- an unperturbed trajectory "
        "would then be trivially reproduced by any ON arm")

    # --- ON: store_mass_flux + gateway_step + a CSV dump every step
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import (
        promote_gateway_geometry, setup_gateway_accumulator,
    )
    g1, z1, m_on, s_on = _build(store=True)
    geom = promote_gateway_geometry(getattr(m_on, "grid", g1))
    lat2d = jnp.asarray(np.degrees(np.asarray(g1.lat))[:, None]
                        * np.ones((1, g1.n_lon)))
    lon2d = jnp.asarray(np.degrees(np.asarray(g1.lon))[None, :]
                        * np.ones((g1.n_lat, 1)))
    acc, stack, _f = setup_gateway_accumulator(
        lat2d, lon2d, s_on.land_mask.data, lat_min_deg=0.0)
    assert _digest(s_on) == d_init_off, "the two arms started apart"
    csv = R._gateway_cumulative_open(tmp_path, acc.names, io_proc=True)
    for k in range(1, n_steps + 1):
        s_on = m_on.step(s_on, dt)
        acc = gateway_step(acc, stack, s_on, z1, geom, source="stored")
        R._gateway_cumulative_row(csv, acc, k, k * dt / 86400.0)
    R._gateway_cumulative_close(csv)

    assert acc.n == n_steps, "the ON arm did not actually accumulate"
    _h, rows = _read_csv(tmp_path / R.GATEWAY_CUMULATIVE_CSV)
    assert len(rows) == n_steps, "the ON arm did not actually dump"
    d_off, d_on = _digest(s_off), _digest(s_on)
    assert d_off == d_on, (
        f"the gateway accumulator/dump perturbed the trajectory over "
        f"{n_steps} steps: {d_off} != {d_on}")


# ------------------------------------------------------------- driver wiring


def _driver_lines():
    return (Path(_ROOT) / "scripts/run/run_omip_core2.py").read_text().splitlines()


def _code(line: str) -> str:
    """The CODE part of a source line, with any trailing comment removed.

    A source-scanning test that matches a COMMENTED-OUT call passes while the
    call does nothing (codex round-1 YELLOW 6).  Splitting on ``#`` is naive --
    it would also cut a ``#`` inside a string literal -- but none of the lines
    these tests match contain one, and cutting too much can only make an
    assertion FAIL, never pass spuriously, which is the safe direction.

    RESIDUAL LIMIT, stated rather than papered over: this still cannot see that
    a call is REACHABLE.  A call at the right indentation inside a block whose
    condition is never true would pass.  The live coverage for that is
    test_differencing_two_dumps_equals_a_fresh_accumulator_over_that_window and
    test_accumulating_and_dumping_does_not_perturb_the_trajectory, which run
    the real accumulator and the real writer; these source tests only pin the
    ORDERING constraints inside a monolithic main() that no in-memory test can
    reach.
    """
    return line.split("#", 1)[0]


def test_driver_opens_the_csv_before_the_loop_and_dumps_at_the_snapshot_cadence():
    """Ordering constraints inside a monolithic main(), pinned by symbol name.

    A dump BEFORE the loop or a missing run-end row both leave a window
    unrecoverable, and neither is visible to any in-memory unit test.
    """
    lines = _driver_lines()
    loop = next(i for i, ln in enumerate(lines)
                if ln.strip().startswith("for step in range(1, n_steps"))
    opens = [i for i, ln in enumerate(lines)
             if "_gw_csv = _gateway_cumulative_open(" in _code(ln)]
    assert len(opens) == 1, f"expected one CSV open, found {opens}"
    assert opens[0] < loop, (
        "the CSV must be opened BEFORE the step loop; opening it inside would "
        "truncate the file at every cadence")
    rows = [i for i, ln in enumerate(lines)
            if "_gateway_cumulative_row(" in _code(ln) and "def " not in ln]
    in_loop = [i for i in rows if i > loop]
    assert len(in_loop) >= 2, (
        "expected a cadence row inside the loop AND a run-end row after it; "
        f"found row calls at {rows}")
    # The cadence row must sit in the SAME block as the snapshot write, so the
    # dump cadence really is --snapshot-every-days.
    snap = next(i for i, ln in enumerate(lines)
                if ln.strip().startswith("if snap_every > 0 and step % snap_every")
                and i > loop)
    cadence = next(i for i in in_loop if i > snap)
    snap_indent = len(lines[snap]) - len(lines[snap].lstrip())
    assert (len(lines[cadence]) - len(lines[cadence].lstrip())
            > snap_indent), (
        f"the cadence dump at line {cadence + 1} is not inside the "
        f"`if snap_every ...` block at line {snap + 1}")


def test_driver_writes_a_run_end_row_and_closes_after_the_loop():
    """`step != n_steps` excludes the last step, so the final window is only
    recoverable if the run-end row is written."""
    lines = _driver_lines()
    write = next(i for i, ln in enumerate(lines)
                 if "_gateway_transport_diag(_gw_acc" in ln)
    final_row = [i for i, ln in enumerate(lines)
                 if "_gateway_cumulative_row(_gw_csv, _gw_acc, n_steps" in _code(ln)]
    assert len(final_row) == 1, (
        "the run-end cumulative row is missing -- the last window (e.g. days "
        "60-90 of a 90-day run) could not be recovered")
    closes = [i for i, ln in enumerate(lines)
              if "_gateway_cumulative_close(" in _code(ln) and "def " not in ln]
    assert closes, "the CSV is never closed"
    assert final_row[0] < write, (
        "the run-end row must be written before/with the whole-run block")
    after = [i for i in closes if i > final_row[0]]
    assert after, "the CSV is not closed after the run-end row is written"


def test_driver_dumps_and_closes_on_the_blowup_abort_path():
    """The abort `return 1` skips the run-end writer; without a dump there the
    accumulated totals since the last cadence are lost.

    There are TWO `[ABORT] non-finite state` sites -- the --scan-block lane
    (which the gateway lane guard REFUSES outright) and the host loop.  This
    must target the host loop's, i.e. the one after `for step in range(...)`;
    picking the first match would assert against the lane the flag can never
    reach and would pass no matter what the host loop does.
    """
    lines = _driver_lines()
    loop = next(i for i, ln in enumerate(lines)
                if ln.strip().startswith("for step in range(1, n_steps"))
    aborts = [i for i, ln in enumerate(lines)
              if 'print("[ABORT] non-finite state", flush=True)' in ln]
    assert len(aborts) >= 2, (
        "expected a scan-lane and a host-loop abort; the anchor rotted")
    abort = next(i for i in aborts if i > loop)
    tail = "\n".join(_code(ln) for ln in lines[abort:abort + 14])
    assert "_gateway_cumulative_row(" in tail, (
        "the blowup path does not dump the accumulator before returning")
    assert "_gateway_cumulative_close(" in tail, (
        "the blowup path leaves the CSV open")
    assert "return 1" in tail


def test_the_csv_filename_is_pinned():
    """A downstream reader keys off this name; renaming it is a breaking change."""
    from scripts.run import run_omip_core2 as R
    assert R.GATEWAY_CUMULATIVE_CSV == "gateway_transports.csv"
    assert R.GATEWAY_CUMULATIVE_CSV != "transports.txt", (
        "the per-cadence series must NOT share the run-end scalar file")


# =====================================================================
# codex round-1 findings.  Each of these exists because an earlier assertion
# was shown to be VACUOUS, either by the reviewer or by the mutation battery
# (scripts/cluster/omip_nemo/_mutation_gw_cumulative.sbatch).
# =====================================================================


def test_n_steps_is_the_accumulator_count_not_the_step_index(tmp_path):
    """MUTATION M5 STAYED GREEN because every other test had step == acc.n.

    The dumps carry the model step index AND the accumulator's own count, and
    only the latter may be used as the divisor: the driver disables the
    diagnostic on a gateway_step failure WITHOUT stopping the run, so the two
    can diverge, and dividing by the step difference would then report a
    scaled-down rate.  Every other row test passes step == acc.n, which makes
    a writer that emitted `step` indistinguishable -- this one does not.
    """
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import GatewayAccumulator
    from scripts.run import run_omip_core2 as R

    step, n_acc = 900, 617          # deliberately unequal
    assert step != n_acc, "the fixture must make the two distinguishable"
    acc = GatewayAccumulator(("davis_caa",), jnp.asarray([1.5e6]),
                             jnp.asarray([5.1e7]), n_acc)
    csv = R._gateway_cumulative_open(tmp_path, acc.names, io_proc=True)
    R._gateway_cumulative_row(csv, acc, step, 6.25)
    R._gateway_cumulative_close(csv)
    _h, rows = _read_csv(tmp_path / R.GATEWAY_CUMULATIVE_CSV)
    assert len(rows) == 1
    assert int(rows[0]["n_steps"]) == n_acc, (
        f"n_steps is {rows[0]['n_steps']}, the accumulator counted {n_acc} -- "
        "a reader dividing by this gets the wrong rate")
    assert int(rows[0]["step"]) == step, "the step index column is wrong"


def test_a_repeated_dump_point_is_not_written_twice(tmp_path):
    """Two rows with the same n_steps give a reader a ZERO-step window.

    Dump points genuinely coincide: a run ending exactly on a cadence
    boundary, or a gateway_step failure on the step right after a cadence
    dump.  Both then call the writer with the same accumulator.
    """
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import GatewayAccumulator
    from scripts.run import run_omip_core2 as R

    acc = GatewayAccumulator(("davis_caa",), jnp.asarray([2.0e6]),
                             jnp.asarray([3.0e6]), 42)
    csv = R._gateway_cumulative_open(tmp_path, acc.names, io_proc=True)
    R._gateway_cumulative_row(csv, acc, 42, 1.0)
    R._gateway_cumulative_row(csv, acc, 42, 1.0)      # same n -> suppressed
    grown = acc._replace(volume=jnp.asarray([3.0e6]), n=43)
    R._gateway_cumulative_row(csv, grown, 43, 1.1)    # n moved -> written
    R._gateway_cumulative_close(csv)
    _h, rows = _read_csv(tmp_path / R.GATEWAY_CUMULATIVE_CSV)
    ns = [int(r["n_steps"]) for r in rows]
    assert ns == [42, 43], f"expected one row per dump point, got {ns}"


def test_a_write_failure_disables_the_writer_instead_of_retrying_it(tmp_path):
    """A broken handle must not be retried at every later cadence.

    Non-vacuity: the SAME accumulator sequence is first shown to produce rows
    through a healthy writer, so the emptiness below is caused by the failure
    and not by the fixture.
    """
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import GatewayAccumulator
    from scripts.run import run_omip_core2 as R

    def _accs():
        return [GatewayAccumulator(("davis_caa",), jnp.asarray([float(k)]),
                                   jnp.asarray([float(k)]), k)
                for k in (1, 2, 3)]

    healthy = tmp_path / "healthy"
    healthy.mkdir()
    csv = R._gateway_cumulative_open(healthy, ("davis_caa",), io_proc=True)
    for k, a in enumerate(_accs(), start=1):
        R._gateway_cumulative_row(csv, a, k, float(k))
    R._gateway_cumulative_close(csv)
    _h, ok_rows = _read_csv(healthy / R.GATEWAY_CUMULATIVE_CSV)
    assert len(ok_rows) == 3, "the control did not write -- the test is vacuous"

    broken = tmp_path / "broken"
    broken.mkdir()
    csv = R._gateway_cumulative_open(broken, ("davis_caa",), io_proc=True)
    csv.fh.close()                    # simulate a dead handle mid-run
    accs = _accs()
    R._gateway_cumulative_row(csv, accs[0], 1, 1.0)   # fails -> DISABLE
    assert csv.fh is None, "the writer was left live after a write failure"
    R._gateway_cumulative_row(csv, accs[1], 2, 2.0)   # must be a silent no-op
    R._gateway_cumulative_close(csv)                  # must not raise
    _h, rows = _read_csv(broken / R.GATEWAY_CUMULATIVE_CSV)
    assert rows == [], "a row survived a disabled writer"


def test_transports_txt_content_is_pinned_exactly(tmp_path):
    """Pin the WHOLE file, not just 'the two arms agree'.

    The byte-identical test compares two arms through the SAME writer, so a
    change to that writer moves both and passes (codex round-1 YELLOW 3).
    This pins the literal bytes a downstream reader parses, so any change to
    the key names, the ordering, or the number formats is a deliberate,
    visible edit here.
    """
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import GatewayAccumulator
    from scripts.run import run_omip_core2 as R

    acc = GatewayAccumulator(("bering_pacific", "atlantic_nordic"),
                             jnp.asarray([2.0e6, 6.0e6]),
                             jnp.asarray([4.0e6, 8.0e6]), 2)
    R._gateway_transport_diag(acc, tmp_path, io_proc=True)
    assert (tmp_path / "transports.txt").read_text() == (
        "gateway_n_steps 2\n"
        "gateway_bering_pacific_vol_Sv 1.000000\n"
        "gateway_bering_pacific_salt_psu_m3s 2.000000e+06\n"
        "gateway_atlantic_nordic_vol_Sv 3.000000\n"
        "gateway_atlantic_nordic_salt_psu_m3s 4.000000e+06\n"
    ), "the transports.txt output contract moved"


def test_driver_salvages_the_accumulated_tail_when_gateway_step_fails():
    """codex round-1 RED: `_gw_acc = None` alone throws away the tail.

    gateway_step is pure, so on a raise `_gw_acc` still holds the last-good
    accumulator (n = step - 1).  Everything since the previous cadence row --
    and the whole-run block, since _gateway_transport_diag(None) is a no-op --
    is lost unless it is dumped before the handle is dropped.
    """
    lines = _driver_lines()
    # Anchor on the log line (unique) and read FORWARD, rather than anchoring
    # on `_gw_acc = None` and reading back over a fixed window -- the comment
    # block between them is longer than any window worth hard-coding, and a
    # too-small one made this test raise StopIteration instead of asserting.
    log = [i for i, ln in enumerate(lines) if "[gateway] DISABLED at step" in ln]
    assert len(log) == 1, f"expected one disable site, found {log}"
    tail = lines[log[0]:log[0] + 20]
    drop = next(i for i, ln in enumerate(tail)
                if _code(ln).strip() == "_gw_acc = None")
    window = [_code(ln) for ln in tail[:drop]]
    assert any("_gateway_cumulative_row(" in ln for ln in window), (
        "the accumulator is discarded without dumping what it reached")
    assert any("step - 1" in ln for ln in window), (
        "the salvage row must be labelled with the LAST GOOD step (step - 1); "
        "gateway_step raised on `step`, so that step never accumulated")
