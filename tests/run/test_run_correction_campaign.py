"""Smoke test for the LES-informed correction CAMPAIGN driver composition.

Exercises the genuinely-new orchestration (``make_clubb_build_driver`` +
``make_les_diagnose_fn`` + ``build_correction_campaign``) WITHOUT a real model
run: a mock driver supplies the column state and a mock LES supplies the plane
state, so the REAL process_column forcing-extract + diagnose path runs (the real
model run + real LES are covered by iter 35/37 and iter 20 respectively).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.column_les import ColumnLESConfig  # noqa: E402
from legoesm.atmosphere.dynamics.les_regime import (  # noqa: E402
    LESRegimeConfig,
    LESResolutionConfig,
)
from legoesm.training.compare_reanalysis import ColumnState  # noqa: E402

from scripts.run.run_correction_campaign import (  # noqa: E402
    _build_arg_parser,
    _campaign_knobs_from_args,
    _capture_initial_record,
    _distributed_campaign_kwargs,
    _env_kernel_export_note,
    _format_per_variable_bias,
    _per_variable_bias_dict,
    _per_variable_bias_from_dict,
    _per_variable_to_json,
    _resolve_era5_n_times,
    build_campaign_output_dict,
    build_correction_campaign,
    build_distributed_multi_correction_campaign,
    build_multi_correction_campaign,
    compose_compare_fn,
    grid_latlon_deg,
    load_base_config_and_grid,
    make_base_driver_builder,
    make_clubb_build_driver,
    make_les_diagnose_fn,
    maybe_env_grid_fn,
    refuse_unsupported_multirank,
    resolve_orographic_phis,
)


def test_env_kernel_export_note():
    """The env-kernel export note: WARN only when --feedback-strategy environment
    produced NO kernel; silent (None) when a kernel exists or for static/non-CLUBB."""
    # Kernel produced ⇒ no note (it gets written).
    assert _env_kernel_export_note("environment", has_kernel=True) is None
    # static campaign with no kernel ⇒ silent (correctly has nothing to transfer).
    assert _env_kernel_export_note("static", has_kernel=False) is None
    # environment-strategy with NO kernel ⇒ a WARNING explaining the absent artifact.
    note = _env_kernel_export_note("environment", has_kernel=False)
    assert note is not None
    assert "feedback-strategy environment" in note
    assert "no <out>.env_kernel.json" in note.lower() or "env_kernel.json" in note


def test_per_variable_bias_dict_roundtrip():
    """The raw-PerVariableBias (campaign-start baseline) checkpoint serializer round-trips
    losslessly, with NaN precip <-> JSON null (preserving precip-not-compared)."""
    import json

    from legoesm.training.bias_metrics import PerVariableBias

    assert _per_variable_bias_dict(None) is None
    assert _per_variable_bias_from_dict(None) is None
    pv = PerVariableBias(jnp.asarray(4.0), jnp.asarray(1e-3), jnp.asarray(2.0),
                         jnp.asarray(float("nan")))
    d = _per_variable_bias_dict(pv)
    assert d["T_rmse_K"] == pytest.approx(4.0)
    assert d["precip_err_mm_day"] is None              # NaN -> null
    json.loads(json.dumps(d, allow_nan=False))         # valid strict JSON
    back = _per_variable_bias_from_dict(d)
    assert float(back.global_T_rmse_K) == pytest.approx(4.0)
    assert float(back.global_wind_rmse_m_s) == pytest.approx(2.0)
    assert bool(jnp.isnan(back.global_precip_err_mm_day))   # null -> NaN


def test_capture_initial_record():
    """_capture_initial_record: the FIRST fresh round captures the campaign-start
    baseline (combined + per-variable); later/resumed rounds PRESERVE it; a non-finite
    baseline is NOT stored (it must not poison every future resume's reported start)."""
    from types import SimpleNamespace

    from legoesm.training.bias_metrics import (
        BiasImprovement,
        PerVariableBias,
        compare_per_variable_bias,
    )

    def _res(base, t_base, n_diag=2, n_valid=1):
        bias = BiasImprovement(
            baseline_bias=jnp.asarray(base), updated_bias=jnp.asarray(base * 0.9),
            absolute_reduction=jnp.asarray(0.0), fractional_improvement=jnp.asarray(0.0),
            improved=jnp.asarray(True))
        pvb = compare_per_variable_bias(
            PerVariableBias(jnp.asarray(t_base), jnp.asarray(1e-3), jnp.asarray(2.0),
                            jnp.asarray(float("nan"))),
            PerVariableBias(jnp.asarray(1.0), jnp.asarray(1e-3), jnp.asarray(2.0),
                            jnp.asarray(float("nan"))))
        return SimpleNamespace(bias=bias, per_variable_bias=pvb,
                               n_diagnosed=n_diag, n_diagnoses_valid=n_valid)

    box = {"initial_bias": None, "initial_per_variable": None}
    _capture_initial_record(box, _res(5.0, 8.0))       # first fresh round captures
    assert box["initial_bias"] == pytest.approx(5.0)
    assert box["initial_per_variable"]["T_rmse_K"] == pytest.approx(8.0)
    assert box["n_diag_seg"] == 2 and box["n_valid_seg"] == 1   # counts accumulate
    _capture_initial_record(box, _res(0.6, 3.0, n_diag=3, n_valid=2))  # later round
    assert box["initial_bias"] == pytest.approx(5.0)            # original PRESERVED
    assert box["initial_per_variable"]["T_rmse_K"] == pytest.approx(8.0)
    assert box["n_diag_seg"] == 5 and box["n_valid_seg"] == 3   # SEGMENT running sum
    # A non-finite baseline is NOT stored (the next round retries) — never poison resumes.
    box2 = {"initial_bias": None, "initial_per_variable": None}
    _capture_initial_record(box2, _res(float("nan"), 8.0))
    assert box2["initial_bias"] is None


def test_per_variable_to_json_nan_precip_is_null():
    """The persisted per-variable JSON: None->None; a NaN precip (not compared)
    serializes as JSON null (valid JSON, round-trips), NEVER a misleading 0/NaN."""
    import json

    from legoesm.training.bias_metrics import (
        PerVariableBias,
        compare_per_variable_bias,
    )

    assert _per_variable_to_json(None) is None
    b = PerVariableBias(jnp.asarray(4.0), jnp.asarray(1e-3), jnp.asarray(2.0),
                        jnp.asarray(float("nan")))
    u = PerVariableBias(jnp.asarray(1.0), jnp.asarray(1e-3), jnp.asarray(5.0),
                        jnp.asarray(float("nan")))
    d = _per_variable_to_json(compare_per_variable_bias(b, u))
    assert d["baseline"]["T_rmse_K"] == pytest.approx(4.0)
    assert d["final"]["wind_rmse_m_s"] == pytest.approx(5.0)
    assert d["baseline"]["precip_err_mm_day"] is None     # NaN -> null, not 0
    assert d["improved"] == {"T": True, "qv": False, "wind": False, "precip": False}
    json.loads(json.dumps(d, allow_nan=False))            # valid strict JSON (no NaN token)


def test_format_per_variable_bias():
    """The per-round per-variable line: None -> '' (mock round); a real
    PerVariableBiasImprovement -> baseline->updated per variable + which improved."""
    from legoesm.training.bias_metrics import (
        PerVariableBias,
        PerVariableBiasImprovement,
    )

    assert _format_per_variable_bias(None) == ""
    base = PerVariableBias(jnp.asarray(4.0), jnp.asarray(1e-3), jnp.asarray(2.0),
                           jnp.asarray(float("nan")))
    upd = PerVariableBias(jnp.asarray(1.0), jnp.asarray(1e-3), jnp.asarray(5.0),
                          jnp.asarray(float("nan")))
    pvb = PerVariableBiasImprovement(
        baseline=base, updated=upd,
        T_improved=jnp.asarray(True), qv_improved=jnp.asarray(False),
        wind_improved=jnp.asarray(False), precip_improved=jnp.asarray(False))
    line = _format_per_variable_bias(pvb)
    assert "per-var RMSE" in line
    assert "T 4->1K" in line and "wind 2->5m/s" in line
    assert "improved: T" in line and "wind" not in line.split("improved:")[1]


def test_resolve_orographic_phis_off_skips_provider():
    """'off' returns None WITHOUT calling the provider (so no driver/probe is built
    — the legacy flat path constructs nothing)."""
    calls = []

    def provider():
        calls.append(1)
        return jnp.full((4, 8), 5000.0)

    assert resolve_orographic_phis("off", provider) is None
    assert calls == []                                   # provider NOT called for 'off'


def test_resolve_orographic_phis_auto_terrain_and_flat():
    """'auto' returns the model's terrain when present, and None (safe) when the
    model is flat (identically-zero phis) — passing 'auto' is always safe."""
    terrain = jnp.zeros((4, 8)).at[1, 1].set(3000.0)
    out = resolve_orographic_phis("auto", lambda: terrain)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(terrain))
    assert resolve_orographic_phis("auto", lambda: jnp.zeros((4, 8))) is None


def test_resolve_orographic_phis_on_requires_terrain():
    """'on' returns terrain when present but FAILS LOUD (SystemExit) on a flat model
    — an explicit terrain request must not silently run flat."""
    terrain = jnp.full((4, 8), 2000.0)
    np.testing.assert_array_equal(
        np.asarray(resolve_orographic_phis("on", lambda: terrain)),
        np.asarray(terrain))
    with pytest.raises(SystemExit, match="requires model topography"):
        resolve_orographic_phis("on", lambda: jnp.zeros((4, 8)))


def test_resolve_orographic_phis_unknown_mode_raises():
    """Dispatch hardening: an unknown mode raises ValueError (argparse choices=
    constrains the CLI, but the helper must not silently accept a typo)."""
    with pytest.raises(ValueError, match="unknown orographic_forcing mode"):
        resolve_orographic_phis("terrain", lambda: jnp.zeros((4, 8)))


def test_orographic_forcing_flag_parsed():
    """The --orographic-forcing flag parses to the expected choices (default auto)."""
    p = _build_arg_parser()
    base = ["--config", "c.json", "--era5-zarr", "z"]
    assert p.parse_args(base).orographic_forcing == "auto"
    assert p.parse_args(base + ["--orographic-forcing", "on"]).orographic_forcing == "on"
    with pytest.raises(SystemExit):                       # argparse rejects bad choice
        p.parse_args(base + ["--orographic-forcing", "terrain"])


def test_era5_time_mean_flags_parsed():
    """The iter-140 time-mean reference flags parse to the expected defaults+values
    (wiring guard: a rename/removal of --era5-n-times / --era5-time-idx is caught)."""
    p = _build_arg_parser()
    base = ["--config", "c.json", "--era5-zarr", "z"]
    d = p.parse_args(base)
    assert d.era5_time_idx == 0           # default: first time
    assert d.era5_n_times == 1            # default: a single snapshot (old behaviour)
    v = p.parse_args(base + ["--era5-time-idx", "12", "--era5-n-times", "30"])
    assert v.era5_time_idx == 12
    assert v.era5_n_times == 30


def test_resolve_era5_n_times_validates_loudly():
    """--era5-n-times must be >= 1; a value < 1 fails LOUDLY (no silent max(1,...)
    clamp that would mask a fat-fingered 0 / negative) — CLAUDE.md fail-loud."""
    assert _resolve_era5_n_times(1) == 1          # the default single-snapshot window
    assert _resolve_era5_n_times(30) == 30        # a real climatology window
    assert _resolve_era5_n_times(2.0) == 2        # int-coerces a clean whole float
    for bad in (0, -1, -7):
        with pytest.raises(SystemExit, match=r"--era5-n-times must be >= 1"):
            _resolve_era5_n_times(bad)
    # a non-integral float is rejected too (no silent truncation 2.9 -> 2).
    with pytest.raises(SystemExit, match=r"must be a whole number"):
        _resolve_era5_n_times(2.9)


def test_refuse_unsupported_multirank_guards_cli():
    """The single-process CLI refuses an mpirun -np >1 launch LOUDLY (it wires none
    of the distributed hooks); a single rank (or no MPI) is allowed (iter 88)."""
    refuse_unsupported_multirank(comm_size=1)          # single rank: allowed
    with pytest.raises(SystemExit, match="SINGLE-PROCESS CLI"):
        refuse_unsupported_multirank(comm_size=2)      # multi-rank: refused

_SMALL_RES = LESResolutionConfig(
    dx_m=50.0, nx=8, ny=8, nlev=8, domain_top_m=2000.0, dz_sfc_m=50.0)
_SMALL_REGIME = LESRegimeConfig(shallow=_SMALL_RES, deep=_SMALL_RES)


def _full_grid_state(nlat=8, nlon=16, nlev=5):
    shp, sfc = (nlat, nlon, nlev), (nlat, nlon)
    return ColumnState(
        T=jnp.full(shp, 280.0), q_v=jnp.full(shp, 5e-3),
        u=jnp.full(shp, 5.0), v=jnp.zeros(shp),
        p_s=jnp.full(sfc, 1.0e5), sst_K=jnp.full(sfc, 290.0))


class _FakeDriver:
    def __init__(self, state):
        self.state = state

    def run(self, segment_callback, **kwargs):  # noqa: ARG002
        segment_callback(self, 0.0, 1.0)        # one diagnostic segment
        return "OK"


def _mock_run_les(setup):
    """Mock plane-LES result: a synthetic state shaped to the setup's grid/hc."""
    from legoesm.atmosphere.dynamics.compressible_euler_plane import make_rest_state

    grid, hc = setup.grid, setup.height_coord
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    ny, nx, nlev = grid.ny, grid.nx, hc.n_levels
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    s = jnp.asarray(np.where((ii + jj) % 2 == 0, 1.0, -1.0))
    w = 2.0 * s[:, :, None] * jnp.ones((ny, nx, nlev + 1))
    thp = 0.5 * s[:, :, None] * jnp.ones((ny, nx, nlev))
    tr = jnp.zeros((ny, nx, nlev, 3)).at[..., 0].set(0.01)
    return state._replace(
        w=state.w.replace(data=w),
        theta_prime=state.theta_prime.replace(data=thp),
        tracers=state.tracers.replace(data=tr))


def _base_config():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite")


def test_grid_latlon_deg_defaults_to_grid_centre_degrees():
    """grid_latlon_deg defaults lat/lon to the grid's centre coords CONVERTED to
    DEGREES (radians·180/π); explicit values pass through UNCHANGED (no double
    conversion). A dropped rad2deg would tag the manifest with radian 'latitudes'."""
    from types import SimpleNamespace

    grid = SimpleNamespace(grid_lat=np.array([0.0, np.pi / 2]),
                           grid_lon=np.array([0.0, np.pi]))
    lat, lon = grid_latlon_deg(grid, None, None)
    np.testing.assert_allclose(np.asarray(lat), [0.0, 90.0])
    np.testing.assert_allclose(np.asarray(lon), [0.0, 180.0])
    # Explicit values pass through (caller already supplied degrees).
    lat2, lon2 = grid_latlon_deg(grid, jnp.array([12.0]), jnp.array([34.0]))
    np.testing.assert_array_equal(np.asarray(lat2), [12.0])
    np.testing.assert_array_equal(np.asarray(lon2), [34.0])


def test_load_base_config_and_grid_roundtrip(tmp_path):
    """load_base_config_and_grid loads the ExperimentConfig JSON + builds the SAME grid
    + vertical coord the run uses (latlon res 8, nlev 5), WITHOUT the heavy setup() —
    the config-loading entry shared by the campaign CLI AND the OSSE CLI, so a
    regression here breaks both launches."""
    import json

    from legoesm.driver.config import experiment_config_to_dict

    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(experiment_config_to_dict(_base_config())))
    cfg, grid, sigma = load_base_config_and_grid(str(p))
    assert cfg.grid.grid_type == "latlon" and cfg.grid.resolution == 8
    assert cfg.turbulence == "clubb_lite"             # the campaign-required scheme survives
    assert grid is not None and hasattr(grid, "grid_lat")
    assert int(np.asarray(sigma.sigma_full).shape[0]) == 5   # nlev levels built


def test_maybe_env_grid_fn_dispatch():
    """maybe_env_grid_fn returns the env-grid producer ONLY for 'environment' (else
    None — the static-scatter feedback needs no env grid), with ``sigma`` bound."""
    from legoesm.training.feedback_assembly import column_environment_grid

    sigma = object()
    fn = maybe_env_grid_fn("environment", sigma)
    assert callable(fn)
    assert fn.func is column_environment_grid        # partial of the right producer
    assert fn.keywords.get("sigma") is sigma         # sigma bound into the partial
    assert maybe_env_grid_fn("static", sigma) is None
    assert maybe_env_grid_fn("anything_else", sigma) is None


def test_main_cmip_rejects_unknown_coupled_preset(monkeypatch):
    """--mode cmip with an unknown --coupled-preset fails LOUD at launch (a typo'd preset)
    with the valid choices, NOT a cryptic KeyError deeper in the coupled-driver build.
    Fires before the reference loading, so only load_base_config_and_grid is stubbed."""
    from types import SimpleNamespace

    import scripts.run.run_correction_campaign as rcc

    monkeypatch.setattr(
        rcc, "load_base_config_and_grid",
        lambda path: (SimpleNamespace(
            grid=SimpleNamespace(grid_type="latlon", resolution=8, nlev=5)),
            object(), object()))
    with pytest.raises(SystemExit, match="unknown --coupled-preset"):
        rcc.main(["--config", "c.json", "--era5-zarr", "z", "--mode", "cmip",
                  "--coupled-preset", "not_a_preset"])


def test_main_resume_rejects_mismatched_corrected_field(tmp_path, monkeypatch):
    """The HOT resume path (multi-day SLURM restarts): resuming a checkpoint whose
    corrected coefficient (Pr_t) differs from --diagnosis-method's field (C_K) fails LOUD
    at launch, NOT silently loading the field into the WRONG config slot (a garbage
    correction). Drives main() with the heavy preamble stubbed up to the resume guard."""
    import json
    from types import SimpleNamespace

    import legoesm.training.compare_reanalysis as cr
    import legoesm.training.era5_to_state as e2s

    import scripts.run.run_correction_campaign as rcc
    import scripts.validate.compare_amip_era5 as cae

    fake_cfg = SimpleNamespace(
        grid=SimpleNamespace(grid_type="latlon", resolution=8, nlev=5))
    fake_grid = SimpleNamespace(grid_shape_2d=(2, 3))
    monkeypatch.setattr(rcc, "load_base_config_and_grid",
                        lambda path: (fake_cfg, fake_grid, object()))
    monkeypatch.setattr(
        rcc, "make_base_driver_builder",
        lambda mode, coupled_preset=None, ocean_grid=None: ((lambda c: None),
                                                            (lambda d, day, dt: None)))
    monkeypatch.setattr(e2s, "load_era5_time_mean", lambda cfg, idx: object())
    monkeypatch.setattr(cae, "select_era5_regrid", lambda canon: (lambda slc, g, s: object()))
    monkeypatch.setattr(cr, "column_state_from_carry", lambda carry: object())
    monkeypatch.setattr(rcc, "resolve_orographic_phis", lambda forcing, provider: None)

    ckpt = tmp_path / "ckpt.json"
    ckpt.write_text(json.dumps({"corrected_field": "Pr_t", "field": [[0.4]], "round": 0}))
    with pytest.raises(SystemExit, match="checkpoint corrects"):
        rcc.main(["--config", "c.json", "--era5-zarr", "z", "--resume", str(ckpt),
                  "--diagnosis-method", "clubb_coefficient"])    # corrected_field "C_K" != "Pr_t"


def test_maybe_write_env_kernel_writes_when_kernel_present(tmp_path, monkeypatch, capsys):
    """``_maybe_write_env_kernel`` exports the RAW env-kernel JSON (the grid-AGNOSTIC
    cross-resolution deploy PRODUCER) when the last ACCEPTED round produced one — the
    write/skip glue was untested (its components are)."""
    import json
    from types import SimpleNamespace

    import legoesm.training.correction_loop as cl
    import legoesm.training.deploy_correction as dc

    import scripts.run.run_correction_campaign as rcc

    monkeypatch.setattr(cl, "last_accepted_env_kernel", lambda result: "KERNEL")
    monkeypatch.setattr(dc, "env_kernel_to_dict", lambda k: {"field": "C_K", "k": k})
    out = str(tmp_path / "camp.json")
    rcc._maybe_write_env_kernel(
        SimpleNamespace(out=out, feedback_strategy="environment"), object())
    with open(f"{out}.env_kernel.json") as f:
        assert json.load(f) == {"field": "C_K", "k": "KERNEL"}     # last-accepted kernel serialized
    assert "wrote RAW environment kernel" in capsys.readouterr().out


def test_maybe_write_env_kernel_skips_and_warns_when_no_kernel(tmp_path, monkeypatch, capsys):
    """No kernel (a rejected/no-op round) → NO file; an 'environment'-strategy campaign
    that produced none WARNS (the cross-grid artifact the user expected is absent), while
    'static' is SILENT (it has no env→coefficient regression to transfer)."""
    import os
    from types import SimpleNamespace

    import legoesm.training.correction_loop as cl

    import scripts.run.run_correction_campaign as rcc

    monkeypatch.setattr(cl, "last_accepted_env_kernel", lambda result: None)
    out = str(tmp_path / "camp.json")
    rcc._maybe_write_env_kernel(
        SimpleNamespace(out=out, feedback_strategy="environment"), object())
    assert not os.path.exists(f"{out}.env_kernel.json")            # nothing written
    assert "WARNING" in capsys.readouterr().out                    # env-strategy + no kernel ⇒ warn
    rcc._maybe_write_env_kernel(
        SimpleNamespace(out=out, feedback_strategy="static"), object())
    assert "WARNING" not in capsys.readouterr().out                # static ⇒ silent


def test_print_deploy_hint_verifies_output_deploys_on_own_grid(tmp_path, capsys):
    """``_print_deploy_hint`` round-trips the JUST-WRITTEN campaign output through the
    production deploy loader WITH its own grid, so a non-deployable / grid-inconsistent
    output fails LOUD at WRITE time — not silently when the HPC user deploys it days
    later. A valid output passes + prints the deploy hint; an output verified against a
    DIFFERENT grid raises (the deploy guard's rejection, propagated not swallowed)."""
    import json

    from legoesm.grids.latlon import create_latlon_grid

    import scripts.run.run_correction_campaign as rcc

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    output = {"C_K": [0.4] * (8 * 16),
              "grid": rcc._grid_provenance(_base_config(), grid)}
    path = str(tmp_path / "out.json")
    with open(path, "w") as f:
        json.dump(output, f)
    rcc._print_deploy_hint(path, grid)                  # deploys onto its OWN grid → no raise
    assert "deploy into a production run" in capsys.readouterr().out
    # An output verified against a DIFFERENT grid fails LOUD at write-time.
    other = create_latlon_grid(4, 8, dtype=jnp.float64)
    with pytest.raises(ValueError):
        rcc._print_deploy_hint(path, other)


def test_grid_provenance_produces_deploy_compatible_fingerprint():
    """``_grid_provenance`` is the campaign-side deploy grid-identity fingerprint PRODUCER
    embedded in the output JSON; the deploy CONSUMER (``assert_deploy_compatible``) is
    tested but the producer was NOT (a private helper the def-vs-test sweep skips). Lock
    (a) the recorded metadata, (b) the producer→consumer ROUND-TRIP (a provenanced output
    deploys onto its OWN grid), and (c) non-vacuity (a DIFFERENT grid → DIFFERENT coord
    fingerprint) — so a producer regression cannot silently break the grid-safety guard
    that stops per-column coefficients landing on the wrong cells (clause-6 deploy)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.training.deploy_correction import (
        assert_deploy_compatible,
        corrected_turbulence_override,
    )

    import scripts.run.run_correction_campaign as rcc

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    base_cfg = _base_config()                           # latlon, resolution 8, nlev 5
    prov = rcc._grid_provenance(base_cfg, grid)
    assert prov["grid_type"] == "latlon"
    assert prov["resolution"] == 8 and prov["nlev"] == 5
    assert "coord_sha256" in prov                       # the coordinate fingerprint

    output = {"C_K": [0.4] * (8 * 16), "grid": prov}
    assert_deploy_compatible(output, grid)              # PRODUCER → CONSUMER round-trip OK
    assert corrected_turbulence_override(output, grid=grid).scheme == "clubb_lite"

    # Non-vacuity: a DIFFERENT grid yields a DIFFERENT coord fingerprint (so the guard
    # would catch a cross-grid deploy — the consumer's tested job; here we prove the
    # PRODUCER distinguishes grids rather than emitting a constant).
    other = create_latlon_grid(4, 8, dtype=jnp.float64)
    assert rcc._grid_provenance(base_cfg, other)["coord_sha256"] != prov["coord_sha256"]


def test_compose_compare_fn_threads_valid_mask_and_manifest_reducer(monkeypatch):
    """compose_compare_fn must pass ``valid_mask`` + ``manifest_reducer`` THROUGH to
    make_compare_fn, so an ocean/land mask or the distributed owned-cell mask actually
    restricts the worst-column ranking. A dropped pass-through would silently rank +
    correct masked (e.g. non-owned halo / land) cells — otherwise caught ONLY by the
    MPI e2e tests (which do not run in the default fast suite)."""
    from types import SimpleNamespace

    import scripts.run.run_correction_campaign as rcc

    captured = {}

    def fake_make_compare_fn(**kwargs):
        captured.update(kwargs)
        return lambda config: None

    def reducer(manifest):                           # the distributed top-k post-processor
        return manifest

    monkeypatch.setattr(rcc, "make_compare_fn", fake_make_compare_fn)
    vm = jnp.array([True, False, True])
    sigma = SimpleNamespace(sigma_full=jnp.zeros(5), sigma_half=jnp.zeros(6))
    compose_compare_fn(
        base_atm_config=_base_config(),          # turbulence="clubb_lite" (required)
        build_base_driver=(lambda cfg: object()),
        extract_column_state=(lambda d, day, dt: None),
        reference=object(), sigma=sigma, area_weights=jnp.ones(3), n_worst=2,
        lat_deg=jnp.zeros(3), lon_deg=jnp.zeros(3),
        valid_mask=vm, manifest_reducer=reducer,
    )
    assert captured["valid_mask"] is vm              # the mask is threaded, not dropped
    assert captured["manifest_reducer"] is reducer   # the distributed top-k reducer too
    assert captured["n_worst"] == 2                  # (sanity: other args also threaded)


def test_make_clubb_build_driver_injects_override():
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig

    captured = {}

    def build_base(cfg):
        captured["cfg"] = cfg
        return _FakeDriver(_full_grid_state())

    build_driver = make_clubb_build_driver(_base_config(), build_base)
    build_driver(CLUBBLiteConfig(C_K=0.9))
    cfg = captured["cfg"]
    assert cfg.turbulence == "clubb_lite"
    assert cfg.turbulence_override is not None
    assert float(cfg.turbulence_override.clubb_lite.C_K) == 0.9


def test_make_les_diagnose_fn_runs_process_column():
    """diagnose_fn(record, model_ctx) drives the REAL process_column + a mock LES."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les)

    class _Env:
        cape_J_kg = 200.0  # noqa: N815  (mirrors ColumnEnvironment field)

    class _Rec:
        grid_index = (4, 8)
        lat_deg = 20.0
        environment = _Env()

    out = diagnose_fn(_Rec(), _full_grid_state())
    assert out.K.shape == (_SMALL_RES.nlev - 1,)   # eddy-K profile
    assert bool(jnp.all(jnp.isfinite(out.K)))


def test_make_les_diagnose_fn_threads_phis_to_process_column(monkeypatch):
    """iter-118 wiring: make_les_diagnose_fn(phis=...) forwards the static model
    topography to process_column (→ the orographic geostrophic term); the default
    forwards None (flat). The forcing extraction inside process_column is stubbed so
    the test pins ONLY the phis threading."""
    import legoesm.atmosphere.dynamics.column_les as cl
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    captured = {}

    def fake_process_column(record, **kw):
        captured["phis"] = kw.get("phis", "MISSING")
        return "DIAG"

    # make_les_diagnose_fn imports process_column at call time (function-scope), so
    # patch the module attribute BEFORE building the diagnose_fn.
    monkeypatch.setattr(cl, "process_column", fake_process_column)

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)

    class _Env:
        cape_J_kg = 200.0  # noqa: N815

    class _Rec:
        grid_index = (4, 8)
        lat_deg = 20.0
        environment = _Env()

    phis_grid = jnp.zeros((8, 16))
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, phis=phis_grid)
    assert diagnose_fn(_Rec(), _full_grid_state()) == "DIAG"
    assert captured["phis"] is phis_grid          # forwarded to process_column

    diagnose_flat = make_les_diagnose_fn(
        grid, sigma, les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les)                 # no phis → flat
    diagnose_flat(_Rec(), _full_grid_state())
    assert captured["phis"] is None


def test_make_les_diagnose_fn_mpas_routes_edge_velocity():
    """An MPAS worst column spins off its LES end-to-end (iter 76): the model_ctx
    carries the native u_edge, make_les_diagnose_fn routes it to the Voronoi
    forcing extractor (grid=mesh, v=None), and the mock LES yields a diagnosis."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    diagnose_fn = make_les_diagnose_fn(
        mesh, sigma, les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les)

    # An MPAS ColumnState: cell T/q_v/p_s + the NATIVE edge velocity in u_edge.
    u_edge = 6.0 * jnp.cos(jnp.asarray(mesh.angleEdge))[:, None] * jnp.ones((1, nlev))
    mpas_ctx = ColumnState(
        T=jnp.full((mesh.nCells, nlev), 285.0),
        q_v=jnp.full((mesh.nCells, nlev), 6e-3),
        u=jnp.zeros((mesh.nCells, nlev)),     # cell winds present but unused (routed)
        v=jnp.zeros((mesh.nCells, nlev)),
        p_s=jnp.full((mesh.nCells,), 1.0e5),
        u_edge=u_edge)

    cell = 40

    class _Env:
        cape_J_kg = 200.0  # noqa: N815

    class _Rec:
        grid_index = (cell,)                  # arity-1 cell index for MPAS
        lat_deg = float(np.rad2deg(np.asarray(mesh.latCell)[cell]))
        environment = _Env()

    out = diagnose_fn(_Rec(), mpas_ctx)
    assert out.K.shape == (_SMALL_RES.nlev - 1,)
    assert bool(jnp.all(jnp.isfinite(out.K)))


def test_make_les_diagnose_fn_mpas_u_edge_on_wrong_grid_raises():
    """u_edge set but a non-Voronoi grid → loud grid/state mismatch (Codex)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    diagnose_fn = make_les_diagnose_fn(
        grid, create_sigma_coordinate(5),
        les_config=ColumnLESConfig(regime=_SMALL_REGIME), run_les_fn=_mock_run_les)
    ctx = _full_grid_state()._replace(u_edge=jnp.zeros((10, 5)))

    class _Rec:
        grid_index = (4, 8)
        lat_deg = 20.0

    with pytest.raises(ValueError, match="not a VoronoiMesh"):
        diagnose_fn(_Rec(), ctx)


@pytest.mark.slow
def test_build_correction_campaign_wiring_one_round():
    """WIRING test: the whole campaign composes + runs one round with a mock
    driver + mock LES — a per-column clubb C_K is produced and the bias is a
    finite measurement.  Does NOT validate that the C_K injection changes the
    model output (the mock driver returns a fixed state) — that mechanism is
    covered by iter 35/37."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    # reference = model with a localized +6 K bias → one deterministic worst column.
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    def build_base(cfg):           # mock: fixed state regardless of C_K
        return _FakeDriver(model_state)

    def extract(driver, day, dt):  # noqa: ARG001
        return driver.state

    result = build_correction_campaign(
        base_atm_config=_base_config(), build_base_driver=build_base,
        extract_column_state=extract, reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, n_worst=1,
        # Test the config→model wiring (a per-column C_K is applied); the mock
        # compare has no real improvement signal, so disable the monotonic gate
        # that would otherwise reject this round (gate tested in test_correction_loop).
        accept_only_if_improved=False)

    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert it.n_diagnosed == 1
    # The campaign produced a per-column clubb C_K (the loop closed).
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 1
    assert result.final_config.C_K.shape == (8 * 16,)


def _mpas_base_config(nlev=5):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=2, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="mpas"),
        radiation="gray", turbulence="clubb_lite")


def _mpas_full_state(mesh, nlev=5, *, bias_cell=None, bias_dt=0.0):
    """An MPAS comparison ColumnState: cell T/q_v/p_s + the native u_edge."""
    temp = np.full((mesh.nCells, nlev), 285.0)
    if bias_cell is not None:
        temp[bias_cell] += bias_dt
    u_edge = 6.0 * np.cos(np.asarray(mesh.angleEdge))[:, None] * np.ones((1, nlev))
    return ColumnState(
        T=jnp.asarray(temp), q_v=jnp.full((mesh.nCells, nlev), 6e-3),
        u=jnp.zeros((mesh.nCells, nlev)), v=jnp.zeros((mesh.nCells, nlev)),
        p_s=jnp.full((mesh.nCells,), 1.0e5),
        sst_K=jnp.full((mesh.nCells,), 290.0),
        u_edge=jnp.asarray(u_edge))


@pytest.mark.parametrize("feedback_strategy", ["static", "environment"])
def test_build_correction_campaign_mpas_one_round(feedback_strategy):
    """CAPSTONE (iters 73-76): the FULL MPAS pipeline composes through the REAL
    build_correction_campaign — cell compare/rank → Voronoi forcing extract (via
    the native u_edge) → mock LES → per-CELL clubb C_K feedback → re-run → finite
    bias. NON-VACUOUS: a SHEARED mock LES gives a VALID clubb_coefficient
    diagnosis, so the diagnosed C_K (clamped to bounds) actually REPLACES the
    background at the worst cell (static) / spreads via the env regression +
    produces a deploy kernel (environment) — not a no-op. Mock driver + mock LES;
    the C_K-changes-MODEL-output mechanism is iter 35/37."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    bias_cell = 37
    bg = float(CLUBBLiteConfig().C_K)              # production default (0.4)
    model_state = _mpas_full_state(mesh, nlev)
    # reference = model with one cold-biased cell → a deterministic worst cell.
    reference = _mpas_full_state(mesh, nlev, bias_cell=bias_cell, bias_dt=-6.0)

    def build_base(cfg):           # mock: fixed MPAS state regardless of C_K
        return _FakeDriver(model_state)

    def extract(driver, day, dt):  # noqa: ARG001
        return driver.state

    result = build_correction_campaign(
        base_atm_config=_mpas_base_config(nlev), build_base_driver=build_base,
        extract_column_state=extract, reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        # clubb_coefficient + a SHEARED mock → a VALID, non-background C_K diagnosis.
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        feedback_strategy=feedback_strategy,
        accept_only_if_improved=False)

    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert it.n_diagnosed == 1                 # the worst CELL spun off its LES
    assert bool(np.isfinite(float(it.bias.updated_bias)))
    # A per-CELL clubb C_K reached the config (the loop closed on the 1-D cell axis).
    ck = np.asarray(result.final_config.C_K)
    assert ck.ndim == 1 and ck.shape == (mesh.nCells,)
    assert np.all(np.isfinite(ck))
    # NON-VACUOUS: the correction actually changed C_K away from the background.
    assert not np.allclose(ck, bg), "the diagnosed C_K never reached the cells"
    if feedback_strategy == "static":
        # The diagnosed value landed on the worst CELL; the rest stay background.
        assert not np.isclose(float(ck[bias_cell]), bg)
        np.testing.assert_allclose(np.delete(ck, bias_cell), bg)
    else:
        # The env-strategy round produced a transferable deploy kernel (iter 70).
        assert it.env_kernel is not None and it.env_kernel.field == "C_K"


def test_build_correction_campaign_owned_cell_mask_excludes_halo():
    """WIRING (iter 86): the ``valid_mask`` param threads through
    build_correction_campaign → compose_compare_fn → make_compare_fn, so a
    DISTRIBUTED-MPAS owned-cell mask keeps a HALO cell out of the ranking. A halo
    cell carrying the globally-LARGEST bias is masked out; the campaign instead
    spins off + corrects the worst OWNED cell — proving the mask both threads
    through AND changes the outcome (non-vacuous)."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.training.compare_reanalysis import owned_cell_valid_mask

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    n_owned = 120                          # owned = [0, 120); halo = [120, 162)
    halo_worst, owned_worst = 150, 50
    bg = float(CLUBBLiteConfig().C_K)
    model_state = _mpas_full_state(mesh, nlev)
    # reference: TWO cold-biased cells — the halo one colder (worst), the owned one
    # second.  Without the mask the halo cell is worst; with it, the owned cell is.
    ref = _mpas_full_state(mesh, nlev)
    rt = np.asarray(ref.T).copy()
    rt[halo_worst] -= 12.0
    rt[owned_worst] -= 8.0
    reference = ref._replace(T=jnp.asarray(rt))
    owned_mask = owned_cell_valid_mask(jnp.arange(mesh.nCells) < n_owned)

    def build_base(cfg):
        return _FakeDriver(model_state)

    def extract(driver, day, dt):  # noqa: ARG001
        return driver.state

    result = build_correction_campaign(
        base_atm_config=_mpas_base_config(nlev), build_base_driver=build_base,
        extract_column_state=extract, reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False, valid_mask=owned_mask)

    it = result.iterations[0]
    assert it.n_diagnosed == 1
    ck = np.asarray(result.final_config.C_K)
    # The masked HALO cell was NEVER diagnosed (stays background); the worst OWNED
    # cell got the correction.
    assert np.isclose(float(ck[halo_worst]), bg), "a halo cell was wrongly corrected"
    assert not np.isclose(float(ck[owned_worst]), bg), "the owned worst cell was skipped"


def test_build_correction_campaign_mpas_les_budget_clusters_cells():
    """The LES-cost reduction (env clustering) works on the MPAS cell layout: 4
    worst cells in 2 distinct-SST environments + les_budget=2 → only 2 LES run
    (the cluster representatives), but all 4 cells are corrected. Exercises
    cluster_columns_by_environment over the 1-D cell manifest."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    # 4 worst cells split into 2 SST environments (2 cells each) → 2 clusters.
    cold_cells, warm_cells = [10, 20], [120, 130]
    worst = cold_cells + warm_cells

    def _state(*, biased):
        temp = np.full((mesh.nCells, nlev), 285.0)
        if biased:
            for c in worst:
                temp[c] -= 6.0
        sst = np.full((mesh.nCells,), 285.0)
        sst[warm_cells] = 300.0                    # the env tag that splits clusters
        u_edge = 6.0 * np.cos(np.asarray(mesh.angleEdge))[:, None] * np.ones((1, nlev))
        return ColumnState(
            T=jnp.asarray(temp), q_v=jnp.full((mesh.nCells, nlev), 6e-3),
            u=jnp.zeros((mesh.nCells, nlev)), v=jnp.zeros((mesh.nCells, nlev)),
            p_s=jnp.full((mesh.nCells,), 1.0e5), sst_K=jnp.asarray(sst),
            u_edge=jnp.asarray(u_edge))

    model_state = _state(biased=False)
    reference = _state(biased=True)               # the 4 biased cells are the worst

    result = build_correction_campaign(
        base_atm_config=_mpas_base_config(nlev),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=4, les_budget=2,
        accept_only_if_improved=False)

    it = result.iterations[0]
    assert it.n_corrected == 4                    # all 4 worst cells corrected
    assert it.n_diagnosed == 2                    # but only 2 LES (cluster reps)
    assert np.isfinite(float(it.bias.baseline_bias))
    assert np.isfinite(float(it.bias.updated_bias))


def _gaussian_full_state(grid, nlev=5, *, bias_col=None, bias_dt=0.0):
    """A Gaussian (spectral) comparison ColumnState on the (n_lat, n_lon) grid.

    The zonal wind is DIVERGENT (``u = U cos λ``, varying with longitude) so the SH
    continuity chain in the iter-83 Gaussian extractor produces a genuinely NONZERO
    ω — a solid-body (``u = U cos φ``) field is non-divergent (∇·v = 0) and would
    leave ω ≈ 0, making the spectral path vacuously exercised.  T/q_v/p_s are
    uniform with one optionally cold-biased column for a deterministic worst column;
    the winds are identical in model and reference, so the worst-column ranking is
    driven purely by the T bias.  ALL arrays are float64 (the Gaussian extractor
    hard-raises on a float32 ``T``)."""
    n_lat, n_lon = grid.n_lat, grid.n_lon
    temp = np.full((n_lat, n_lon, nlev), 285.0)
    if bias_col is not None:
        temp[bias_col] += bias_dt
    lon = np.asarray(grid.grid_lon)[:, :, None]    # (n_lat, n_lon, 1)
    u = 15.0 * np.cos(lon) * np.ones((n_lat, n_lon, nlev))   # divergent → ω ≠ 0
    return ColumnState(
        T=jnp.asarray(temp, dtype=jnp.float64),
        q_v=jnp.full((n_lat, n_lon, nlev), 6e-3, dtype=jnp.float64),
        u=jnp.asarray(u, dtype=jnp.float64),
        v=jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64),
        p_s=jnp.full((n_lat, n_lon), 1.0e5, dtype=jnp.float64),
        sst_K=jnp.full((n_lat, n_lon), 290.0, dtype=jnp.float64))


def test_build_correction_campaign_gaussian_one_round(monkeypatch):
    """CAPSTONE (iters 83-84): the FULL Gaussian/spectral pipeline composes through
    the REAL build_correction_campaign — 2-D (n_lat, n_lon) compare/rank → the
    iter-83 SH-divergence forcing extract (``extract_column_forcing_gaussian`` via
    the GaussianGrid dispatch branch, col_index=(i_lat, i_lon)) → sheared mock LES →
    per-COLUMN clubb C_K feedback → re-run → finite bias.

    NON-VACUOUS, each property MACHINE-CHECKED rather than inferred:
      1. A SPY wraps ``extract_column_forcing_gaussian`` and asserts the spectral
         branch fired EXACTLY ONCE, for the BIASED worst column, producing a FINITE,
         NONZERO ω (the SH continuity chain genuinely ran — the divergent wind makes
         ω ≠ 0).  The lat-lon FD extractor is monkeypatched to RAISE, so a silent
         FD fallback is impossible (Codex iter-85 issues 1+2).
      2. A SHEARED mock LES gives a VALID clubb_coefficient diagnosis, so the
         diagnosed C_K actually REPLACES the background at the worst column and the
         OTHER columns stay at the background (Codex iter-85 issue 4).
      3. The float64 ``T`` survives the campaign into the extractor's float64 guard
         (a float32 ``T`` would raise; the other fields are widened to T's dtype).
    Mock driver + mock LES; the C_K-changes-model-output mechanism is iter 35/37."""
    from legoesm.atmosphere.dynamics import column_large_scale_extract as clse
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_gaussian_grid(n_max=21, dealiasing="linear")
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    # mid-grid, off the poles AND off the u=15cos(λ) divergence nodes (λ=0,π at
    # i_lon 0,22) so the SH continuity chain produces a genuinely nonzero ω there.
    bias_col = (11, 10)
    bg = float(CLUBBLiteConfig().C_K)              # production default (0.4)
    model_state = _gaussian_full_state(grid, nlev)
    # reference = model with one cold-biased column → a deterministic worst column.
    reference = _gaussian_full_state(grid, nlev, bias_col=bias_col, bias_dt=-6.0)

    # ROUTING + FORCING probe (Codex iter-85): the dispatcher resolves both
    # extractors as module globals, so patch them on the module the dispatcher reads.
    real_gaussian = clse.extract_column_forcing_gaussian
    calls: dict = {"n": 0}

    def _spy_gaussian(*, col_index, **kw):
        out = real_gaussian(col_index=col_index, **kw)
        omega = np.asarray(out.omega)
        assert np.all(np.isfinite(omega)), "spectral extractor produced non-finite ω"
        assert float(np.max(np.abs(omega))) > 1e-4, "ω is vacuously ~0 (no divergence)"
        calls["n"] += 1
        calls["col_index"] = tuple(int(c) for c in col_index)
        return out

    def _no_latlon(**kw):  # noqa: ARG001
        raise AssertionError("lat-lon FD extractor called for a GaussianGrid")

    monkeypatch.setattr(clse, "extract_column_forcing_gaussian", _spy_gaussian)
    monkeypatch.setattr(clse, "extract_column_forcing_latlon", _no_latlon)

    def build_base(cfg):           # mock: fixed Gaussian state regardless of C_K
        return _FakeDriver(model_state)

    def extract(driver, day, dt):  # noqa: ARG001
        return driver.state

    result = build_correction_campaign(
        base_atm_config=_base_config(), build_base_driver=build_base,
        extract_column_state=extract, reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.asarray(grid.grid_area), n_iterations=1,
        # clubb_coefficient + a SHEARED mock → a VALID, non-background C_K diagnosis.
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False)

    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert it.n_diagnosed == 1                 # the worst COLUMN spun off its LES
    assert bool(np.isfinite(float(it.bias.updated_bias)))
    # ROUTING: the SPECTRAL branch fired exactly once, for the biased worst column
    # (the lat-lon FD extractor would have raised — proven not taken).
    assert calls["n"] == 1, "the Gaussian/spectral extractor was not the path taken"
    assert calls["col_index"] == bias_col, "ranked the wrong column as worst"
    # A per-column clubb C_K reached the config, flattened on the (n_lat*n_lon) axis.
    ck = np.asarray(result.final_config.C_K)
    assert ck.ndim == 1 and ck.shape == (grid.n_lat * grid.n_lon,)
    assert np.all(np.isfinite(ck))
    # NON-VACUOUS: the diagnosed C_K landed on the worst column, others stay at bg.
    flat_worst = bias_col[0] * grid.n_lon + bias_col[1]
    assert not np.isclose(ck[flat_worst], bg), "diagnosed C_K never reached the column"
    np.testing.assert_allclose(np.delete(ck, flat_worst), bg)  # rest untouched


def _mock_run_les_sheared(setup):
    """Mock plane-LES with a mean-wind shear so the clubb_coefficient diagnosis
    yields a VALID dimensionless C_K (the rest-state mock has no shear)."""
    state = _mock_run_les(setup)
    z = jnp.asarray(setup.height_coord.z_full)
    ny, nx = setup.grid.ny, setup.grid.nx
    u = (0.01 * z)[None, None, :] * jnp.ones((ny, nx, z.shape[0]))   # constant shear
    return state._replace(u=state.u.replace(data=u))


def test_build_correction_campaign_clubb_coefficient_method():
    """The dimensionless clubb_coefficient diagnosis is wired end-to-end: the
    campaign auto-populates l_mix_max from the GCM config, the LES diagnoses a
    DIMENSIONLESS C_K, the loop reduces it (same method), and a per-column,
    in-bounds C_K is produced — proving the units-correct path runs."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False)            # wiring test (no improvement signal)

    it = result.iterations[0]
    assert it.n_diagnosed == 1
    ck = np.asarray(result.final_config.C_K).reshape(-1)
    assert ck.shape == (8 * 16,)
    # clip_to_bounds default ON ⇒ the dimensionless C_K stays in (0.1, 1.2).
    assert float(ck.min()) >= 0.1 and float(ck.max()) <= 1.2


def test_build_correction_campaign_rejects_multi_methods():
    """build_correction_campaign is single-coefficient: a diagnosis_methods config
    (which makes process_column return a dict) is rejected up front, not crashed
    downstream (the simultaneous multi-coefficient campaign is not wired here)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    with pytest.raises(ValueError, match="diagnosis_methods"):
        build_correction_campaign(
            base_atm_config=_base_config(),
            build_base_driver=lambda cfg: _FakeDriver(model_state),
            extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
            reference=model_state, sigma=sigma, grid=grid,
            area_weights=jnp.ones((8, 16)), n_iterations=1,
            les_config=ColumnLESConfig(
                regime=_SMALL_REGIME,
                diagnosis_methods=("clubb_coefficient", "prandtl_number"),
                clubb_l_mix_max=100.0),
            run_les_fn=_mock_run_les_sheared, n_worst=1)


def test_build_multi_correction_campaign_mpas_three_coefficients():
    """The SIMULTANEOUS multi-coefficient campaign composes on the MPAS cell
    layout: C_K + Pr_t + C_eps co-corrected from ONE Voronoi LES spin-off per
    worst CELL (via the native u_edge through the SAME make_les_diagnose_fn) →
    three in-bounds per-CELL (nCells,) fields. Exercises run_multi_correction_*
    + the {method: diagnosis} dict path for MPAS."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    model_state = _mpas_full_state(mesh, nlev)
    reference = _mpas_full_state(mesh, nlev, bias_cell=37, bias_dt=-6.0)

    result = build_multi_correction_campaign(
        base_atm_config=_mpas_base_config(nlev),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_K", "Pr_t", "C_eps"), accept_only_if_improved=False)

    assert set(result.final_fields) == {
        "clubb_lite_C_K", "clubb_lite_Pr_t", "clubb_lite_C_eps"}
    ck = np.asarray(result.final_config.C_K).reshape(-1)
    prt = np.asarray(result.final_config.Pr_t).reshape(-1)
    ceps = np.asarray(result.final_config.C_eps).reshape(-1)
    assert ck.shape == (mesh.nCells,) and prt.shape == (mesh.nCells,)
    assert ceps.shape == (mesh.nCells,)
    assert float(ck.min()) >= 0.1 and float(ck.max()) <= 1.2       # C_K bounds
    assert float(prt.min()) >= 0.3 and float(prt.max()) <= 1.5     # Pr_t bounds
    assert float(ceps.min()) >= 0.06 and float(ceps.max()) <= 0.6  # C_eps bounds


def test_build_correction_campaign_prandtl_number_method():
    """The prandtl_number diagnosis is wired end-to-end: the campaign selects the
    clubb_lite_Pr_t promotion + Pr_t background by method, the LES diagnoses a
    DIMENSIONLESS Pr_t, and an in-bounds per-column Pr_t is produced."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="prandtl_number"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False)

    # The corrected coefficient is Pr_t (not C_K); C_K stays the scalar default.
    prt = np.asarray(result.final_config.Pr_t).reshape(-1)
    assert prt.shape == (8 * 16,)
    assert float(prt.min()) >= 0.3 and float(prt.max()) <= 1.5   # Pr_t bounds
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 0   # C_K untouched
    assert result.final_config.C_K == CLUBBLiteConfig().C_K


def test_build_correction_campaign_bias_tol_early_stops():
    """build_correction_campaign forwards bias_tol/patience: a mock driver that
    never improves the bias is rejected every round, so the campaign stops early
    ('converged') instead of running all n_iterations. Uses diagnosis_method=
    'clubb_coefficient' (which the sheared mock LES diagnoses VALIDLY) so the rounds
    genuinely RUN + get rejected — not the dry-LES abort path (iter 101)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),  # fixed → never improves
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=10,
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        bias_tol=1e-9, patience=2)            # default gate ON
    assert result.stop_reason == "converged"
    # sanity: the rounds genuinely produced VALID diagnoses (so this exercises the
    # convergence path, not the dry-LES abort).
    assert all(it.n_diagnoses_valid > 0 for it in result.iterations)
    assert len(result.iterations) == 2        # 2 rejected rounds → early stop


def test_build_correction_campaign_default_gate_rejects_non_improving():
    """build_correction_campaign defaults the monotonic gate ON: the mock driver
    returns a fixed state regardless of C_K, so the round does not lower the bias
    and is REJECTED — the config stays the uncorrected scalar default."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, n_worst=1)            # default gate ON

    assert result.accepted == (False,)                  # non-improving → rejected
    assert not bool(result.iterations[0].bias.improved)
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 0  # unchanged default


def test_make_base_driver_builder_dispatch():
    """The run-mode dispatch returns the right SST extractor + raises on a bad
    mode / a CMIP call missing the coupled pieces (dispatch hardening, iter 40)."""
    from legoesm.training.run_to_column_mean import (
        amip_column_state,
        cmip_column_state,
    )

    _build_amip, extract_amip = make_base_driver_builder("amip")
    assert extract_amip is amip_column_state
    assert callable(_build_amip)

    # cmip needs ONLY coupled_preset; ocean_grid is optional (None ⇒ the coupled
    # driver uses its own atm grid, same-grid coupling).
    _build_cmip, extract_cmip = make_base_driver_builder(
        "cmip", coupled_preset=object())
    assert extract_cmip is cmip_column_state
    assert callable(_build_cmip)

    with pytest.raises(ValueError, match="unknown mode"):
        make_base_driver_builder("xyz")
    with pytest.raises(ValueError, match="requires coupled_preset"):
        make_base_driver_builder("cmip")  # missing coupled_preset


@pytest.mark.slow
def test_make_base_driver_builder_amip_builds_real_driver():
    """The AMIP builder constructs + sets up a real ModelDriver (the campaign's
    build_base_driver path that main() uses)."""
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.model_driver import ModelDriver

    build_amip, _ = make_base_driver_builder("amip")
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite")
    driver = build_amip(cfg)
    assert isinstance(driver, ModelDriver)
    assert driver.grid is not None and driver.sigma is not None


@pytest.mark.slow
def test_make_base_driver_builder_cmip_builds_real_driver():
    """The CMIP builder constructs + sets up a real CoupledESMDriver with
    ocean_grid=None (same-grid coupling: ocean on the atm grid), exposing the
    state/q_v/ocean_state that cmip_column_state reads."""
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    build_cmip, _ = make_base_driver_builder(
        "cmip", coupled_preset=PRESETS["aquaplanet"](), ocean_grid=None)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite")
    driver = build_cmip(cfg)
    assert isinstance(driver, CoupledESMDriver)
    # cmip_column_state reads these; same-grid ⇒ SST on the atm column shape.
    assert driver.state is not None and driver.q_v is not None
    assert tuple(driver.ocean_state.T_sfc.data.shape) == (8, 16)


@pytest.mark.slow
def test_build_correction_campaign_checkpoint_passthrough():
    """build_correction_campaign forwards checkpoint_callback + start_round to
    run_correction_campaign (the restart wiring main() uses)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    calls = []

    def ckpt(round_idx, res, field):
        calls.append((round_idx, np.asarray(field).copy(), res))

    # initial_field (resume base): every non-worst column keeps this value, so the
    # checkpointed accumulated field proves initial_field was forwarded + used.
    # Gate OFF: this is the forwarding/wiring test (the gate is tested separately),
    # so the round is accepted and the worst column gets a correction on top of base.
    init_field = jnp.full((8, 16), 0.55)
    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, n_worst=1,
        initial_field=init_field, start_round=7, checkpoint_callback=ckpt,
        accept_only_if_improved=False)

    assert len(calls) == 1
    round_idx, field, res = calls[0]
    assert round_idx == 7            # start_round forwarded; callback fired once
    flat = field.reshape(-1)
    worst = 4 * 16 + 8               # the +6 K worst column (row-major)
    others = np.delete(flat, worst)
    # Non-worst columns keep initial_field (0.55), NOT the default background 0.4
    # — so initial_field was genuinely forwarded + used as the round's base.
    np.testing.assert_allclose(others, 0.55)
    # Resume contract (the CLI checkpoint fix relies on this): the persisted
    # accumulated field IS the accepted state and its flattened form equals the
    # accepted config's per-column C_K — so rebuilding the config FROM the field on
    # resume cannot desync. (Accepted round ⇒ res.updated_config.C_K matches too.)
    np.testing.assert_allclose(flat, np.asarray(res.updated_config.C_K))
    np.testing.assert_allclose(
        flat, np.asarray(result.final_config.C_K).reshape(-1))


@pytest.mark.slow
def test_build_correction_campaign_mpas_resume_accumulates_on_cells():
    """The restartable campaign (§1) works on the MPAS cell layout: an
    initial_field (nCells,) is the round-0 base, so every non-worst CELL keeps it
    and the worst CELL is corrected ON TOP — the (ncol,)-field accumulation +
    grid.grid_shape_2d=(nCells,) resume path is grid-agnostic."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    bias_cell = 37
    model_state = _mpas_full_state(mesh, nlev)
    reference = _mpas_full_state(mesh, nlev, bias_cell=bias_cell, bias_dt=-6.0)

    calls = []
    init_field = jnp.full((mesh.nCells,), 0.55)       # the resume base
    result = build_correction_campaign(
        base_atm_config=_mpas_base_config(nlev),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        initial_field=init_field, start_round=7,
        checkpoint_callback=lambda r, res, f: calls.append((r, np.asarray(f).copy())),
        accept_only_if_improved=False)

    assert len(calls) == 1 and calls[0][0] == 7       # start_round forwarded
    ck = np.asarray(result.final_config.C_K)
    assert ck.shape == (mesh.nCells,)
    # Non-worst cells keep the resume base; the worst cell is corrected on top.
    np.testing.assert_allclose(np.delete(ck, bias_cell), 0.55)
    assert not np.isclose(float(ck[bias_cell]), 0.55)


@pytest.mark.slow
def test_build_correction_campaign_environment_strategy():
    """feedback_strategy='environment' runs end-to-end through the campaign
    (column_environment_grid computes the full-grid env each round) → a
    per-column C_K."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, n_worst=1, feedback_strategy="environment",
        accept_only_if_improved=False)  # wiring test; gate tested separately

    assert len(result.iterations) == 1
    ck = np.asarray(result.final_config.C_K)
    assert ck.shape == (8 * 16,)             # per-column (env-generalized) field
    assert np.all(np.isfinite(ck))


def test_build_multi_correction_campaign_corrects_both_coefficients():
    """build_multi_correction_campaign co-corrects C_K AND Pr_t from ONE LES run
    per column: it sets diagnosis_methods, auto-populates l_mix_max, and the multi
    campaign produces in-bounds per-column fields for BOTH coefficients."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_multi_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_K", "Pr_t"), accept_only_if_improved=False)

    assert set(result.final_fields) == {"clubb_lite_C_K", "clubb_lite_Pr_t"}
    ck = np.asarray(result.final_config.C_K).reshape(-1)
    prt = np.asarray(result.final_config.Pr_t).reshape(-1)
    assert ck.shape == (8 * 16,) and prt.shape == (8 * 16,)
    assert float(ck.min()) >= 0.1 and float(ck.max()) <= 1.2     # C_K bounds
    assert float(prt.min()) >= 0.3 and float(prt.max()) <= 1.5   # Pr_t bounds


def test_build_correction_campaign_c_eps_single_method():
    """Single-coefficient c_eps is wired: build_correction_campaign selects the
    clubb_lite_C_eps promotion + C_eps background + auto-populates l_mix_max."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME, diagnosis_method="c_eps"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False)
    ceps = np.asarray(result.final_config.C_eps).reshape(-1)
    assert ceps.shape == (8 * 16,)
    assert float(ceps.min()) >= 0.06 and float(ceps.max()) <= 0.6
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 0   # C_K untouched


def test_build_multi_correction_campaign_three_coefficients_with_c_eps():
    """C_K + Pr_t + C_eps co-corrected from one LES run; C_eps closes the wp2-
    identification gap. All three end up in-bounds per-column fields."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_multi_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_K", "Pr_t", "C_eps"), accept_only_if_improved=False)

    assert set(result.final_fields) == {
        "clubb_lite_C_K", "clubb_lite_Pr_t", "clubb_lite_C_eps"}
    ceps = np.asarray(result.final_config.C_eps).reshape(-1)
    assert ceps.shape == (8 * 16,)
    assert float(ceps.min()) >= 0.06 and float(ceps.max()) <= 0.6   # C_eps bounds


def test_build_multi_correction_campaign_sequential_staged():
    """sequential=True routes the staged (block-coordinate-descent) mode through
    build_multi_correction_campaign; per-coefficient fractions are reported."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_multi_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_eps", "C_K", "Pr_t"), sequential=True,
        accept_only_if_improved=False)
    assert set(result.final_fields) == {
        "clubb_lite_C_K", "clubb_lite_Pr_t", "clubb_lite_C_eps"}
    assert result.iterations[0].step_fractions_by_key is not None


def test_build_multi_correction_campaign_rejects_unknown_coefficient():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    with pytest.raises(ValueError, match="unknown coefficient"):
        build_multi_correction_campaign(
            base_atm_config=_base_config(),
            build_base_driver=lambda cfg: _FakeDriver(model_state),
            extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
            reference=model_state, sigma=sigma, grid=grid,
            area_weights=jnp.ones((8, 16)), n_iterations=1,
            les_config=ColumnLESConfig(regime=_SMALL_REGIME),
            run_les_fn=_mock_run_les_sheared, n_worst=1,
            coefficients=("C_K", "bogus"))


# --- Distributed multi-coefficient campaign wrapper (iter 95) ----------------
def _mock_layout(owned, local_cells, n_global=None):
    lc = np.asarray(local_cells)
    if n_global is None:                              # default: tightest mesh spanning lc
        n_global = int(lc.max()) + 1 if lc.size else 0
    return SimpleNamespace(
        owned_mask_cells=jnp.asarray(owned),
        partition=SimpleNamespace(local_cells=lc, nCells_global=n_global))


def _global_cell_state(ncells, nlev=4):
    # PHYSICALLY plausible values (the distributed kwargs now pre-validate the global
    # reference via validate_reference_physical) — distinct per cell for slice checks.
    rng = np.arange(ncells * nlev, dtype=float).reshape(ncells, nlev)
    return ColumnState(
        T=jnp.asarray(280.0 + 0.01 * rng), q_v=jnp.full((ncells, nlev), 5.0e-3),
        u=jnp.zeros((ncells, nlev)), v=jnp.zeros((ncells, nlev)),
        p_s=jnp.full((ncells,), 1.0e5) + jnp.arange(float(ncells)),
        sst_K=jnp.arange(float(ncells)) + 290.0)


def test_distributed_campaign_kwargs_composes_hooks_and_slices():
    """``_distributed_campaign_kwargs`` (the shared composition for BOTH the single +
    multi distributed wrappers) builds the three hooks from the layout AND slices the
    GLOBAL reference + area_weights to the rank's local cells — order-preserving."""
    from functools import partial

    from legoesm.parallel.reductions import global_sum_mpi
    from legoesm.training.distributed_manifest import gather_global_worst_columns

    # 4 global cells; this rank's local_cells = [2, 3, 1] (owned 2,3; halo 1).
    layout = _mock_layout([True, True, False], [2, 3, 1])
    ref = _global_cell_state(4)
    area = jnp.arange(4.0) + 1.0
    kw = _distributed_campaign_kwargs(layout, ref, area, n_worst=1, base_valid_mask=None)
    np.testing.assert_array_equal(np.asarray(kw["valid_mask"]), [True, True, False])
    assert kw["global_reduce"] is global_sum_mpi
    assert isinstance(kw["manifest_reducer"], partial)
    assert kw["manifest_reducer"].func is gather_global_worst_columns
    assert kw["n_worst"] == 1
    # reference + area sliced to local_cells [2,3,1] (order preserved).
    np.testing.assert_array_equal(
        np.asarray(kw["reference"].T), np.asarray(ref.T)[[2, 3, 1]])
    np.testing.assert_array_equal(np.asarray(kw["area_weights"]), [3.0, 4.0, 2.0])
    # the GLOBAL reference is pre-validated here, so the inner builder skips the
    # per-slice re-validation (collective-safe — iter 99).
    assert kw["validate_reference"] is False


def test_distributed_campaign_kwargs_validates_global_reference_units():
    """``_distributed_campaign_kwargs`` pre-validates the GLOBAL reference for a
    units/sign error (identical on every rank ⇒ collective-safe) BEFORE slicing, so a
    distributed run fails fast instead of correcting against a fake bias (iter 99)."""
    layout = _mock_layout([True, True, False], [2, 3, 1])
    bad = _global_cell_state(4)._replace(
        p_s=jnp.full((4,), 1013.0))                  # hPa, not Pa
    with pytest.raises(ValueError, match=r"global reference.*p_s outside"):
        _distributed_campaign_kwargs(
            layout, bad, jnp.ones(4), n_worst=1, base_valid_mask=None)


def test_distributed_campaign_kwargs_shape_check_precedes_physical():
    """The DETERMINISTIC reference-shape check runs BEFORE the value-dependent physical
    check: a mis-passed rank-LOCAL reference (wrong cell count) AND a bad physical value
    fails on the SHAPE (identical on every rank ⇒ collective-safe), not the physical
    bound (which could differ per rank) — iter 99 Codex ordering fix."""
    layout = _mock_layout([True, True, False], [2, 3, 1])   # n_global defaults to 4
    # a 3-cell reference (wrong shape) that ALSO has a units-bad p_s.
    local_ref = _global_cell_state(3)._replace(p_s=jnp.full((3,), 1013.0))
    with pytest.raises(ValueError, match="has 3 cells but the partitioned mesh has 4"):
        _distributed_campaign_kwargs(
            layout, local_ref, jnp.ones(4), n_worst=1, base_valid_mask=None)


def test_distributed_campaign_kwargs_rejects_mesh_mismatch():
    """A GLOBAL area_weights / reference whose cell-count ≠ the partitioned mesh is
    REJECTED LOUDLY — the JAX gather would otherwise silently clamp out-of-range cell
    ids (too-short) or mis-align (wrong mesh), corrupting the rank's compare/weights
    on a multi-day run (iter 97; EXACT-N vs nCells_global also catches too-LONG)."""
    layout = _mock_layout([True, True, False], [2, 3, 4], n_global=5)
    ref5 = _global_cell_state(5)
    # area_weights shorter than the global mesh (5).
    with pytest.raises(ValueError, match="area_weights has 4 cells.*global mesh has 5"):
        _distributed_campaign_kwargs(
            layout, ref5, jnp.ones(4), n_worst=1, base_valid_mask=None)
    # area_weights LONGER than the global mesh — exact check (a bounds check misses this).
    with pytest.raises(ValueError, match="area_weights has 6 cells.*global mesh has 5"):
        _distributed_campaign_kwargs(
            layout, ref5, jnp.ones(6), n_worst=1, base_valid_mask=None)
    # a non-1-D area_weights is rejected (must be per-cell).
    with pytest.raises(ValueError, match="must be 1-D"):
        _distributed_campaign_kwargs(
            layout, ref5, jnp.ones((5, 2)), n_worst=1, base_valid_mask=None)
    # reference cell-count ≠ the mesh (here a reference for 6 cells, mesh has 5) —
    # caught by the deterministic shape check that precedes the physical check.
    with pytest.raises(ValueError, match="reference has 6 cells.*partitioned mesh has 5"):
        _distributed_campaign_kwargs(
            layout, _global_cell_state(6), jnp.ones(5), n_worst=1, base_valid_mask=None)


def test_build_distributed_multi_forwards_composed_kwargs(monkeypatch):
    """``build_distributed_multi_correction_campaign`` forwards the composed
    distributed kwargs (the three hooks + the rank-local reference/area slice) AND the
    multi-coefficient ``campaign_kwargs`` (e.g. ``coefficients``) to
    ``build_multi_correction_campaign`` — the multi sibling of the iter-89 single
    wrapper, sharing the SAME composition (no MPI: only the wiring is exercised)."""
    captured = {}

    def _fake_multi(**kwargs):
        captured.update(kwargs)
        return "MULTI_RESULT"

    monkeypatch.setattr(
        "scripts.run.run_correction_campaign.build_multi_correction_campaign",
        _fake_multi)
    layout = _mock_layout([True, True, False], [2, 3, 1])
    ref = _global_cell_state(4)
    out = build_distributed_multi_correction_campaign(
        layout=layout, reference=ref, area_weights=jnp.arange(4.0) + 1.0, n_worst=1,
        base_atm_config="CFG", coefficients=("C_K", "Pr_t", "C_eps"))
    assert out == "MULTI_RESULT"
    # the distributed hooks + rank-local slice reached build_multi...
    np.testing.assert_array_equal(np.asarray(captured["valid_mask"]), [True, True, False])
    assert captured["manifest_reducer"] is not None and captured["global_reduce"] is not None
    np.testing.assert_array_equal(
        np.asarray(captured["reference"].T), np.asarray(ref.T)[[2, 3, 1]])
    # ...and the multi-only kwargs pass through.
    assert captured["coefficients"] == ("C_K", "Pr_t", "C_eps")
    assert captured["base_atm_config"] == "CFG"


def test_build_distributed_multi_rejects_duplicate_hook_kwarg():
    """A caller cannot set the distributed hooks inconsistently — passing one in
    campaign_kwargs collides with the supplied one (TypeError)."""
    layout = _mock_layout([True, True], [0, 1])
    ref = _global_cell_state(2)
    with pytest.raises(TypeError):
        build_distributed_multi_correction_campaign(
            layout=layout, reference=ref, area_weights=jnp.ones(2), n_worst=1,
            valid_mask=jnp.ones(2, dtype=bool))   # duplicate of the supplied hook


@pytest.mark.parametrize(
    "multi, target",
    [(False, "build_distributed_correction_campaign"),
     (True, "build_distributed_multi_correction_campaign")])
def test_build_distributed_mpas_campaign_wires_local_mesh(monkeypatch, multi, target):
    """``build_distributed_mpas_campaign`` (iter 96, the one-call RUNNABLE entry point)
    partitions the GLOBAL mesh and forwards the rank-LOCAL mesh as BOTH the campaign
    ``grid`` and the bound ``build_base_driver`` — and dispatches to the single vs
    multi distributed wrapper on ``multi`` (no MPI: the layout build + both wrappers
    are monkeypatched, only the wiring is exercised)."""
    import scripts.run.run_correction_campaign as rcc

    layout = SimpleNamespace(local_mesh="LOCAL_MESH")
    seen_global = {}

    def _fake_make_layout(global_mesh, rank, n_ranks):
        seen_global.update(global_mesh=global_mesh, rank=rank, n_ranks=n_ranks)
        return layout

    monkeypatch.setattr(
        "legoesm.parallel.voronoi_mpi.make_voronoi_partition_layout", _fake_make_layout)

    captured = {}

    def _fake_wrapper(**kwargs):
        captured.update(kwargs)
        return "WRAPPED"

    # patch BOTH wrappers; only ``target`` should actually be called.
    for name in ("build_distributed_correction_campaign",
                 "build_distributed_multi_correction_campaign"):
        monkeypatch.setattr(rcc, name,
                            _fake_wrapper if name == target else _boom_wrapper)

    driver_calls = []

    def _build_local_driver(cfg, local_mesh):
        driver_calls.append((cfg, local_mesh))
        return "DRIVER"

    out = rcc.build_distributed_mpas_campaign(
        global_mesh="GMESH", rank=0, n_ranks=1, reference="REF", area_weights="AREA",
        n_worst=2, build_local_driver=_build_local_driver, multi=multi,
        validate_partition=False,                    # mock layout + no MPI collective
        base_atm_config="CFG")

    assert out == "WRAPPED"
    assert seen_global == dict(global_mesh="GMESH", rank=0, n_ranks=1)
    # the rank-LOCAL mesh (not the global one) is wired as the grid + into the driver.
    assert captured["grid"] == "LOCAL_MESH"
    assert captured["layout"] is layout
    assert captured["reference"] == "REF" and captured["n_worst"] == 2
    assert captured["base_atm_config"] == "CFG"        # campaign_kwargs pass through
    # the bound build_base_driver binds local_mesh and forwards the user's builder.
    assert captured["build_base_driver"]("the_cfg") == "DRIVER"
    assert driver_calls == [("the_cfg", "LOCAL_MESH")]


def _boom_wrapper(**kwargs):                            # the wrapper that must NOT run
    raise AssertionError("wrong distributed wrapper dispatched")


def test_build_arg_parser_defaults():
    """The CLI parser (the HPC entry point, previously untested) has the documented
    defaults: abort/clamp/gate are ON, mode=amip (iter 102)."""
    args = _build_arg_parser().parse_args(["--config", "base.json", "--era5-zarr", "era5.zarr"])
    assert args.mode == "amip"
    assert args.iterations == 3 and args.n_worst == 20 and args.patience == 2
    assert args.bias_tol is None
    assert args.keep_dry_rounds is False          # default → dry-abort ON
    assert args.allow_unphysical_coeff is False   # default → bounds clamp ON
    assert args.keep_worsening_rounds is False    # default → monotonic gate ON
    assert args.feedback_strategy == "static"
    # --les-dt default 0.5 keeps the acoustic Courant < 1 at dx=50 m (iter 103); the
    # old 1.0 default gave C_a≈1.16 and is now rejected by the LES CFL pre-flight.
    assert args.les_dt == 0.5


def test_build_arg_parser_required_and_choices():
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])                                   # --config required
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "x", "--era5-zarr", "z", "--mode", "bogus"])  # bad choice


def test_campaign_knobs_from_args_maps_flags():
    """The shared arg→build-kwargs mapping handles the boolean NEGATIONS + the CSV
    step-fractions parse correctly (an inverted flag = a silent HPC bug; iter 102)."""
    parser = _build_arg_parser()
    knobs = _campaign_knobs_from_args(parser.parse_args(["--config", "x", "--era5-zarr", "z"]))
    assert knobs["stop_on_no_valid_diagnoses"] is True    # NOT --keep-dry-rounds
    assert knobs["clip_to_bounds"] is True                # NOT --allow-unphysical-coeff
    assert knobs["accept_only_if_improved"] is True       # NOT --keep-worsening-rounds
    assert knobs["step_fractions"] is None
    assert knobs["patience"] == 2 and knobs["n_worst"] == 20 and knobs["bias_tol"] is None

    flagged = _campaign_knobs_from_args(parser.parse_args([
        "--config", "x", "--era5-zarr", "z", "--keep-dry-rounds", "--allow-unphysical-coeff",
        "--keep-worsening-rounds", "--step-fractions", "1.0,0.5,0.25",
        "--bias-tol", "1e-3", "--patience", "5", "--n-worst", "8",
        "--feedback-strategy", "environment", "--les-budget", "4"]))
    assert flagged["stop_on_no_valid_diagnoses"] is False
    assert flagged["clip_to_bounds"] is False
    assert flagged["accept_only_if_improved"] is False
    assert flagged["step_fractions"] == [1.0, 0.5, 0.25]
    assert flagged["bias_tol"] == pytest.approx(1e-3) and flagged["patience"] == 5
    assert flagged["n_worst"] == 8 and flagged["les_budget"] == 4
    assert flagged["feedback_strategy"] == "environment"


def test_campaign_knobs_are_valid_kwargs_for_both_builders():
    """Every knob the CLI maps MUST be a real kwarg of BOTH builders — so a renamed /
    removed builder param fails LOUDLY here, not silently on an HPC launch (this is
    the test that would have caught the iter-101 'CLI forgot to forward the flag'
    drift; iter 102)."""
    import inspect

    knobs = set(_campaign_knobs_from_args(
        _build_arg_parser().parse_args(["--config", "x", "--era5-zarr", "z"])))
    assert "stop_on_no_valid_diagnoses" in knobs            # the iter-101 flag
    for fn in (build_correction_campaign, build_multi_correction_campaign):
        params = set(inspect.signature(fn).parameters)
        missing = knobs - params
        assert not missing, f"{fn.__name__} missing CLI knobs: {missing}"


def test_builders_forward_campaign_knobs_to_run_loop(monkeypatch):
    """Closes the builder→run-loop hop (the subset test only proves CLI→builder): each
    builder must FORWARD the run-bound knobs to run_*correction_campaign. The loop is
    monkeypatched to capture kwargs; sentinel values are asserted. (n_worst is NOT here
    — it is consumed by compose_compare_fn, a different hop.) iter 102 Codex."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    import scripts.run.run_correction_campaign as rcc

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    ms = _full_grid_state()                                   # physical → passes validate
    common = dict(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(ms),        # noqa: ARG005
        extract_column_state=lambda d, day, dt: d.state,      # noqa: ARG005
        reference=ms, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=4, n_worst=1,
        run_les_fn=_mock_run_les_sheared)
    # sentinel knobs (non-default so a dropped forward is visible).
    knobs = dict(les_budget=3, feedback_strategy="environment",
                 accept_only_if_improved=False, step_fractions=[0.5],
                 clip_to_bounds=False, bias_tol=0.123, patience=9,
                 stop_on_no_valid_diagnoses=False)

    # run_correction_campaign is a module-top import (patch on rcc);
    # run_multi_correction_campaign is a function-scope import (patch the SOURCE).
    cap_single = {}
    monkeypatch.setattr(rcc, "run_correction_campaign",
                        lambda *a, **k: (cap_single.update(k), "R")[1])
    assert build_correction_campaign(
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        **common, **knobs) == "R"
    for key, val in knobs.items():
        assert cap_single[key] == val, f"single builder dropped {key}"

    cap_multi = {}
    monkeypatch.setattr(
        "legoesm.training.correction_loop.run_multi_correction_campaign",
        lambda *a, **k: (cap_multi.update(k), "M")[1])
    assert build_multi_correction_campaign(
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        coefficients=("C_K", "Pr_t"), **common, **knobs) == "M"
    for key, val in knobs.items():
        assert cap_multi[key] == val, f"multi builder dropped {key}"


def _bias_imp(base, upd):
    from legoesm.training.bias_metrics import BiasImprovement
    frac = (base - upd) / base if base != 0.0 else 0.0
    return BiasImprovement(
        baseline_bias=jnp.asarray(base), updated_bias=jnp.asarray(upd),
        absolute_reduction=jnp.asarray(base - upd),
        fractional_improvement=jnp.asarray(frac), improved=jnp.asarray(upd < base))


def test_campaign_output_dict_single_round_trips_into_deploy():
    """The single-coefficient campaign OUTPUT dict, after a JSON round-trip, loads
    through the DEPLOY path (corrected_clubb_config) into a per-column CLUBBLiteConfig
    with the SAME C_K — locking the write/read format of the clause-5 'update the
    parameters' plumbing against drift (iter 105)."""
    import json

    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import CampaignResult, CorrectionResult
    from legoesm.training.deploy_correction import corrected_clubb_config

    field = jnp.array([0.42, 0.55, 0.61, 0.73])
    res = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=field),
        iterations=(CorrectionResult(
            updated_config=None, bias=_bias_imp(1.0, 0.6),
            worst_column_change=jnp.asarray(0.0), feedback_field=field.reshape(2, 2),
            n_corrected=2, n_diagnosed=2, n_diagnoses_valid=2),),
        final_field=field.reshape(2, 2), accepted=(True,), stop_reason="converged")
    summary = summarize_campaign(res, promotion_key="clubb_lite_C_K")
    health = campaign_health(summary)

    out = build_campaign_output_dict(
        res, grid_provenance={"grid_type": "latlon", "n_columns": 4},
        summary=summary, health=health, corrected_field="C_K")
    loaded = json.loads(json.dumps(out))                  # the REAL on-disk round trip
    cfg = corrected_clubb_config(loaded)
    np.testing.assert_allclose(np.asarray(cfg.C_K).reshape(-1), np.asarray(field))
    # the diagnostics survive too (the file is also the campaign's human report).
    assert loaded["summary"]["stop_reason"] == "converged"
    assert loaded["health"]["status"] == health.status
    assert loaded["grid"]["grid_type"] == "latlon"


def test_campaign_output_dict_multi_round_trips_into_deploy():
    """The multi-coefficient ``"fields"`` shape round-trips into corrected_clubb_config
    recovering BOTH C_K and Pr_t per column (iter 105)."""
    import json

    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import (
        MultiCampaignResult,
        MultiCorrectionResult,
    )
    from legoesm.training.deploy_correction import corrected_clubb_config

    ck = jnp.array([0.42, 0.55, 0.61, 0.73])
    prt = jnp.array([0.78, 0.81, 0.83, 0.80])
    res = MultiCampaignResult(
        final_config=CLUBBLiteConfig(C_K=ck, Pr_t=prt),
        iterations=(MultiCorrectionResult(
            updated_config=None, bias=_bias_imp(1.0, 0.5),
            worst_column_change=jnp.asarray(0.0), feedback_fields={},
            n_corrected=2, n_diagnosed=2, n_diagnoses_valid=2),),
        final_fields={"clubb_lite_C_K": ck.reshape(2, 2),
                      "clubb_lite_Pr_t": prt.reshape(2, 2)},
        accepted=(True,), stop_reason="max_iterations")
    summary = summarize_campaign(res)
    health = campaign_health(summary)

    out = build_campaign_output_dict(
        res, grid_provenance={"grid_type": "latlon", "n_columns": 4},
        summary=summary, health=health, coefficients=("C_K", "Pr_t"))
    loaded = json.loads(json.dumps(out))
    assert loaded["coefficients"] == ["C_K", "Pr_t"]
    cfg = corrected_clubb_config(loaded)
    np.testing.assert_allclose(np.asarray(cfg.C_K).reshape(-1), np.asarray(ck))
    np.testing.assert_allclose(np.asarray(cfg.Pr_t).reshape(-1), np.asarray(prt))


def test_campaign_output_dict_requires_exactly_one_shape():
    """Dispatch hardening: pass exactly one of corrected_field / coefficients."""
    res = SimpleNamespace(iterations=(), accepted=(), final_config=None,
                          final_fields={})
    summary = SimpleNamespace()
    health = SimpleNamespace(status="x", message="y")
    with pytest.raises(ValueError, match="EXACTLY one"):
        build_campaign_output_dict(res, grid_provenance={}, summary=summary,
                                   health=health)                       # neither
    with pytest.raises(ValueError, match="EXACTLY one"):
        build_campaign_output_dict(res, grid_provenance={}, summary=summary,
                                   health=health,
                                   corrected_field="C_K", coefficients=("C_K",))  # both


def test_campaign_output_dict_scalar_single_field_is_rejected_by_deploy():
    """A no-op / zero-round campaign that corrected NOTHING leaves a SCALAR
    final_config field; the output writes it as a bare float (NO reshape), so the
    deploy loader's 1-D assertion REJECTS it loudly instead of silently accepting a
    1-column array (Codex iter 105 — the un-masked failure path)."""
    import json

    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import CampaignResult, CorrectionResult
    from legoesm.training.deploy_correction import corrected_clubb_config

    res = CampaignResult(
        final_config=CLUBBLiteConfig(),                  # scalar default C_K (no-op)
        iterations=(CorrectionResult(
            updated_config=None, bias=_bias_imp(1.0, 1.0),  # never improved
            worst_column_change=jnp.asarray(0.0), feedback_field=jnp.zeros((2, 2)),
            n_corrected=0, n_diagnosed=0, n_diagnoses_valid=0),),
        final_field=jnp.zeros((2, 2)), accepted=(True,), stop_reason="max_iterations")
    summary = summarize_campaign(res, promotion_key="clubb_lite_C_K")
    out = build_campaign_output_dict(
        res, grid_provenance={"grid_type": "latlon"}, summary=summary,
        health=campaign_health(summary), corrected_field="C_K")
    loaded = json.loads(json.dumps(out))
    assert not isinstance(loaded["C_K"], list)            # a bare float, NOT [0.4]
    with pytest.raises(ValueError, match="1-D per-column"):
        corrected_clubb_config(loaded)
