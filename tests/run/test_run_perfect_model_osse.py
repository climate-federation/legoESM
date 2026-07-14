"""Smoke test for the perfect-model OSSE CLI wiring (scripts/validate).

``build_perfect_model_osse`` is exercised end-to-end with a mock C_K-sensitive
driver + a mock plane-LES: it must generate the pseudo-truth from ``true_clubb``,
run the real run->time-mean->compare->real-process_column->campaign chain from
``biased_clubb``, and return an :class:`OSSEResult` whose bias obeys the monotonic
gate.  The RECOVERY logic itself is unit-tested with controllable diagnoses in
``tests/unit/test_perfect_model_osse.py``; here the LES diagnosis is the real
(mock-fed) path, so only the wiring + gate invariant are asserted.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig  # noqa: E402
from legoesm.atmosphere.dynamics.les.les_regime import (  # noqa: E402
    LESRegimeConfig,
    LESResolutionConfig,
)
from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig  # noqa: E402
from legoesm.training.compare_reanalysis import ColumnState  # noqa: E402
from legoesm.training.perfect_model_osse import osse_verdict  # noqa: E402

from scripts.validate.run_perfect_model_osse import (  # noqa: E402
    _build_argparser,
    build_multi_perfect_model_osse,
    build_perfect_model_osse,
)

_SMALL_RES = LESResolutionConfig(
    dx_m=50.0, nx=8, ny=8, nlev=8, domain_top_m=2000.0, dz_sfc_m=50.0)
_SMALL_REGIME = LESRegimeConfig(shallow=_SMALL_RES, deep=_SMALL_RES)


class _FakeDriver:
    def __init__(self, state):
        self.state = state

    def run(self, segment_callback, **kwargs):  # noqa: ARG002
        segment_callback(self, 0.0, 1.0)
        return "OK"


def _full_grid_state(temp, nlat=8, nlon=16, nlev=5):
    shp, sfc = (nlat, nlon, nlev), (nlat, nlon)
    return ColumnState(
        T=temp, q_v=jnp.full(shp, 5e-3), u=jnp.full(shp, 5.0), v=jnp.zeros(shp),
        p_s=jnp.full(sfc, 1.0e5), sst_K=jnp.full(sfc, 290.0))


def _ck_sensitive_base(cfg):
    """A driver whose column temperature depends on the injected C_K, so the true
    and biased configs produce DIFFERENT states (a real bias to correct)."""
    ck = jnp.asarray(cfg.turbulence_override.clubb_lite.C_K)
    nlat, nlon, nlev = 8, 16, 5
    ck2d = (jnp.full((nlat, nlon), float(ck)) if ck.ndim == 0
            else ck.reshape(nlat, nlon))
    temp = 280.0 + 10.0 * ck2d[:, :, None] * jnp.ones((nlat, nlon, nlev))
    return _FakeDriver(_full_grid_state(temp))


def _mock_run_les_sheared(setup):
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import make_rest_state

    grid, hc = setup.grid, setup.height_coord
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    ny, nx, nlev = grid.ny, grid.nx, hc.n_levels
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    s = jnp.asarray(np.where((ii + jj) % 2 == 0, 1.0, -1.0))
    w = 2.0 * s[:, :, None] * jnp.ones((ny, nx, nlev + 1))
    thp = 0.5 * s[:, :, None] * jnp.ones((ny, nx, nlev))
    tr = jnp.zeros((ny, nx, nlev, 3)).at[..., 0].set(0.01)
    z = jnp.asarray(hc.z_full)
    u = (0.01 * z)[None, None, :] * jnp.ones((ny, nx, z.shape[0]))
    return state._replace(
        w=state.w.replace(data=w), theta_prime=state.theta_prime.replace(data=thp),
        tracers=state.tracers.replace(data=tr), u=state.u.replace(data=u))


def _base_config():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite")


def test_orographic_forcing_flag_parsed():
    """The OSSE CLI exposes --orographic-forcing (default auto) so a terrain
    go/no-go uses the SAME forcing the real campaign does (iter 126 parity)."""
    import pytest

    p = _build_argparser()
    base = ["--config", "c.json", "--true-ck", "0.2", "--biased-ck", "0.1"]
    assert p.parse_args(base).orographic_forcing == "auto"
    assert p.parse_args(base + ["--orographic-forcing", "off"]).orographic_forcing == "off"
    with pytest.raises(SystemExit):                       # argparse rejects bad choice
        p.parse_args(base + ["--orographic-forcing", "terrain"])


def test_surface_flux_flag_parsed():
    """The OSSE CLI exposes --surface-flux (default off, iter 471) so the go/no-go can use the
    SAME SST-driven LES surface BC the real campaign does — it changes HOW the closure is
    diagnosed, so the OSSE must match it to faithfully predict the realistic recovery."""
    p = _build_argparser()
    base = ["--config", "c.json", "--true-ck", "0.2", "--biased-ck", "0.1"]
    assert p.parse_args(base).surface_flux is False       # off by default (back-compat)
    assert p.parse_args(base + ["--surface-flux"]).surface_flux is True


def test_quick_flag_parsed_and_defaults_off():
    """The OSSE CLI exposes --quick (iter 504): the fast single-coefficient WIRING smoke that
    swaps the production LES for the tiny shared validation regime. Off by default so the
    normal invocation stays the production go/no-go (back-compat)."""
    p = _build_argparser()
    base = ["--config", "c.json", "--true-ck", "0.2", "--biased-ck", "0.1"]
    assert p.parse_args(base).quick is False
    assert p.parse_args(base + ["--quick"]).quick is True


def test_quick_rejected_with_multi_or_cross_resolution():
    """--quick is single-coefficient only: combining it with --coefficients (multi) or
    --fine-resolution (cross-resolution) fails LOUD rather than silently running the slow
    production LES the operator meant to skip. Guard is pure + runs before the heavy setup."""
    import pytest

    from scripts.validate.run_perfect_model_osse import _reject_quick_with_multi_or_cross

    p = _build_argparser()
    base = ["--config", "c.json"]
    # single-coefficient --quick is allowed (no raise)
    _reject_quick_with_multi_or_cross(
        p.parse_args(base + ["--true-ck", "0.2", "--biased-ck", "0.1", "--quick"]))
    with pytest.raises(SystemExit, match="incompatible"):
        _reject_quick_with_multi_or_cross(
            p.parse_args(base + ["--coefficients", "C_K,Pr_t", "--quick"]))
    with pytest.raises(SystemExit, match="incompatible"):
        _reject_quick_with_multi_or_cross(p.parse_args(
            base + ["--true-ck", "0.2", "--biased-ck", "0.1",
                    "--fine-resolution", "8", "--quick"]))
    # WITHOUT --quick the same multi/cross args are fine (the guard is --quick-scoped)
    _reject_quick_with_multi_or_cross(p.parse_args(base + ["--coefficients", "C_K,Pr_t"]))


def test_build_osse_harness_returns_the_shared_wiring():
    """_build_osse_harness (iter 291) factors the driver/LES harness shared by the single
    + multi OSSE builders (byte-identical before): a callable run_fn / build_compare_fn /
    diagnose_fn + the grid_shape, so the wiring lives in ONE place — not the copy-paste
    CLAUDE.md forbids (and reusable per-grid for a future cross-resolution build)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    from scripts.validate.run_perfect_model_osse import _build_osse_harness

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    run_fn, build_compare_fn, diagnose_fn, grid_shape = _build_osse_harness(
        base_atm_config=_base_config(), build_base_driver=_ck_sensitive_base,
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        sigma=sigma, grid=grid, area_weights=jnp.ones((8, 16)),
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1, lat_deg=None, lon_deg=None, phis=None)
    assert callable(run_fn) and callable(build_compare_fn) and callable(diagnose_fn)
    assert grid_shape == (8, 16)


def test_build_perfect_model_osse_wiring():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)

    res = build_perfect_model_osse(
        base_atm_config=_base_config(),
        build_base_driver=_ck_sensitive_base,
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        sigma=sigma, grid=grid, area_weights=jnp.ones((8, 16)),
        true_clubb=CLUBBLiteConfig(C_K=0.9),
        biased_clubb=CLUBBLiteConfig(C_K=0.4),
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1, n_iterations=1)

    # Pseudo-truth generated from true_clubb; biased start recorded.
    assert res.true_value == 0.9
    assert res.initial_value == 0.4
    # A real bias existed (true and biased states differ by construction).
    assert res.initial_bias > 0.0
    # Monotonic gate invariant: the OSSE never reports a worsened bias.
    assert res.final_bias <= res.initial_bias + 1e-9
    assert res.n_rounds == 1
    # The full chain produced finite recovery metrics (real numbers, not NaN).
    assert np.isfinite(res.recovered_value)
    assert np.isfinite(res.final_param_error)
    # The verdict is one of the gated outcomes (the recovered/no_change outcomes
    # are asserted non-vacuously with controllable diagnoses in the unit test).
    assert osse_verdict(res).status in {"recovered", "bias_only", "no_change"}


def test_build_cross_resolution_osse_wiring(monkeypatch):
    """build_cross_resolution_osse (iter 296) wires the coarse + fine driver/LES harnesses
    and the env_grid_fns into run_cross_resolution_osse.  Capture the call (monkeypatching the
    heavy deploy): BOTH grids get a callable run + build-compare harness; the COARSE leg also
    a diagnose_fn; BOTH legs a callable env_grid_fn (the 'environment' strategy the kernel
    transfer REQUIRES — a None there is the iter-69/70 failure mode); the promotion_key/field
    match the method; clip_to_bounds defaults True (production parity).  The deploy/transfer
    MATH itself is unit-tested with analytic harnesses in tests/unit/test_perfect_model_osse."""
    import legoesm.training.perfect_model_osse as pmo
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    from scripts.run.run_correction_campaign import METHOD_PROMOTION
    from scripts.validate.run_perfect_model_osse import build_cross_resolution_osse

    captured = {}

    def _fake_run_cross(**kwargs):
        captured.update(kwargs)
        return "SENTINEL"

    # The builder imports run_cross_resolution_osse from this module at call time, so patch
    # the source — capturing the wiring WITHOUT running the heavy coarse campaign + fine deploy.
    monkeypatch.setattr(pmo, "run_cross_resolution_osse", _fake_run_cross)

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    cfg = _base_config()
    out = build_cross_resolution_osse(
        build_base_driver=_ck_sensitive_base,
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        base_atm_config_coarse=cfg, coarse_grid=grid, coarse_sigma=sigma,
        area_weights_coarse=jnp.ones((8, 16)),
        base_atm_config_fine=cfg, fine_grid=grid, fine_sigma=sigma,
        area_weights_fine=jnp.ones((8, 16)),
        true_clubb=CLUBBLiteConfig(C_K=0.9), biased_clubb=CLUBBLiteConfig(C_K=0.4),
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1, n_iterations=1)

    assert out == "SENTINEL"
    promotion_key, field_name = METHOD_PROMOTION["clubb_coefficient"]
    assert captured["coefficient_field"] == field_name
    assert captured["promotion_key"] == promotion_key
    assert captured["diagnosis_method"] == "clubb_coefficient"
    assert captured["coarse_grid_shape"] == (8, 16)
    assert captured["fine_grid_shape"] == (8, 16)
    for key in ("coarse_run_fn", "coarse_build_compare_fn", "coarse_diagnose_fn",
                "coarse_env_grid_fn", "fine_run_fn", "fine_build_compare_fn",
                "fine_env_grid_fn"):
        assert callable(captured[key]), f"{key} not wired as a callable"
    assert captured["clip_to_bounds"] is True       # coarse campaign clips (production parity)


def test_resolve_fine_resolution_validates_positive_and_distinct():
    """_resolve_fine_resolution (iter 297) fail-louds the two --fine-resolution misuses:
    a non-positive resolution (would crash deep in grid construction) and fine == coarse
    (the SAME grid — the kernel trivially 'transfers' to itself, a falsely-reassuring
    no-op). A distinct positive resolution passes through unchanged."""
    import pytest

    from scripts.validate.run_perfect_model_osse import _resolve_fine_resolution

    assert _resolve_fine_resolution(32, 16) == 32        # finer, distinct -> ok
    assert _resolve_fine_resolution(8, 16) == 8          # coarser but distinct -> ok
    for bad in (0, -4):
        with pytest.raises(SystemExit, match="positive grid resolution"):
            _resolve_fine_resolution(bad, 16)
    with pytest.raises(SystemExit, match="SAME grid"):
        _resolve_fine_resolution(16, 16)                 # degenerate same-grid


def test_run_cross_resolution_main_exit_code_gates_on_transfer(monkeypatch):
    """The --fine-resolution mode is exit-code-gateable (iter 287-290 automation contract):
    exit 0 ONLY when the kernel TRANSFERRED (well-covered AND the fine bias fell); a
    below-threshold-coverage 'out_of_hull' verdict exits non-zero so a chained pipeline
    (smoke && osse --fine-resolution && ...) HALTS on an untrustworthy transfer.  Stubs only
    the heavy pieces (fine-grid build, phis, the OSSE run) — the real _resolve_fine_resolution
    + cross_res_osse_verdict + exit logic are exercised."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.perfect_model_osse import CrossResOSSEResult

    import scripts.run.run_correction_campaign as rcc
    import scripts.validate.run_perfect_model_osse as rpo

    coarse_grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    # The fn imports these from run_correction_campaign at call time → patch the source.
    monkeypatch.setattr(rcc, "_build_grid_for_config",
                        lambda *a, **k: (create_latlon_grid(16, 32, dtype=jnp.float64),
                                         create_sigma_coordinate(5)))
    monkeypatch.setattr(rcc, "resolve_orographic_phis", lambda *a, **k: None)

    def _result(in_hull, reduced, unc=1.0, corr=0.5):
        return CrossResOSSEResult(
            coarse_initial_bias=1.0, coarse_final_bias=0.6, coarse_bias_reduced=True,
            fine_bias_uncorrected=unc, fine_bias_corrected=corr,
            fine_bias_reduction=unc - corr, fine_bias_reduced=reduced,
            fraction_covered=in_hull, fraction_in_hull=in_hull,
            coverage_threshold=0.8, n_fine_columns=512, kernel_field="C_K")

    args = rpo._build_argparser().parse_args(
        ["--config", "c.json", "--true-ck", "0.9", "--biased-ck", "0.4",
         "--fine-resolution", "16"])
    base_cfg = _base_config()           # grid.resolution == 8, distinct from fine 16
    common = dict(base_cfg=base_cfg, coarse_grid=coarse_grid, coarse_sigma=sigma,
                  build_base_driver=lambda *a, **k: None, extract_fn=lambda *a, **k: None,
                  run_les=lambda *a, **k: None, phis_coarse=None)

    # well-covered + the fine bias fell -> transferred -> exit 0
    monkeypatch.setattr(rpo, "build_cross_resolution_osse",
                        lambda **kw: _result(in_hull=1.0, reduced=True))
    assert rpo._run_cross_resolution_main(args, **common) == 0

    # coverage below threshold -> out_of_hull -> exit 1 (untrustworthy; halt the pipeline)
    monkeypatch.setattr(rpo, "build_cross_resolution_osse",
                        lambda **kw: _result(in_hull=0.1, reduced=True, corr=0.9))
    assert rpo._run_cross_resolution_main(args, **common) == 1


def test_run_multi_osse_main_gates_exit_and_forwards_the_config(monkeypatch):
    """The multi-coefficient OSSE CLI (iter 472) builds true=defaults / biased=defaults*factor,
    runs build_multi_perfect_model_osse, and exit-code-gates on multi_osse_verdict (0 = all
    recovered + bias fell, else 1); forwards the coefficients + --surface-flux. Wires the
    previously CLI-unreachable build_multi_perfect_model_osse."""
    from types import SimpleNamespace

    from legoesm.training.perfect_model_osse import CoefRecovery, MultiOSSEResult

    import scripts.validate.run_perfect_model_osse as rpo

    captured = {}

    def _coef(field, fin_err):
        return CoefRecovery(field=field, true_value=0.4, initial_value=0.6,
                            recovered_value=0.42, initial_param_error=0.2,
                            final_param_error=fin_err, param_error_reduced=fin_err < 0.2)

    def _result(final, all_rec):
        per = {"clubb_lite_C_K": _coef("C_K", 0.05 if all_rec else 0.3),
               "clubb_lite_Pr_t": _coef("Pr_t", 0.05 if all_rec else 0.3)}
        return MultiOSSEResult(initial_bias=1.0, final_bias=final,
                               bias_reduction=1.0 - final, bias_reduced=final < 1.0,
                               per_coefficient=per, all_recovered=all_rec,
                               n_rounds=3, n_accepted=2, summary=None)

    args = SimpleNamespace(coefficients="C_K,Pr_t", multi_bias_factor=1.5, surface_flux=True,
                           n_worst=4, iterations=1)
    common = dict(
        base_cfg=object(),
        grid=SimpleNamespace(grid_shape_2d=(2, 2), grid_area=jnp.ones(4)),
        sigma=object(), build_base_driver=lambda *a, **k: None,
        extract_fn=lambda *a, **k: None, run_les=lambda *a, **k: None, phis=None)

    # all recovered + bias fell -> exit 0; the spy captures the forwarded config
    def _spy_ok(**kw):
        captured.update(kw)
        return _result(0.4, all_rec=True)
    monkeypatch.setattr(rpo, "build_multi_perfect_model_osse", _spy_ok)
    assert rpo._run_multi_osse_main(args, **common) == 0
    assert captured["coefficients"] == ("C_K", "Pr_t")        # forwarded
    assert captured["les_config"].surface_flux is True        # --surface-flux forwarded
    assert captured["true_clubb"].C_K != captured["biased_clubb"].C_K   # biased = default*factor

    # bias fell but NOT all recovered -> bias_only -> exit 1
    monkeypatch.setattr(rpo, "build_multi_perfect_model_osse",
                        lambda **kw: _result(0.4, all_rec=False))
    assert rpo._run_multi_osse_main(args, **common) == 1


def test_main_single_dispatch_builds_clubb_and_gates_exit(monkeypatch):
    """The TOP-LEVEL OSSE main() SINGLE-coefficient path (iter 492) — what
    preflight_osse.sbatch's default invokes (main() is pragma:no-cover heavy I/O). It wires
    --config → _build_run_setup → build_perfect_model_osse(true/biased CLUBB from
    --true-ck/--biased-ck) → osse_verdict → exit code. The heavy setup + the 4 model runs are
    monkeypatched; the DISPATCH + the clubb-config building + the verdict→exit wiring run for
    real, so a launcher-breaking regression (wrong builder args, broken dispatch) fails in CI
    rather than on a multi-day HPC batch node."""
    from types import SimpleNamespace

    import pytest
    from legoesm.training.perfect_model_osse import OSSEResult

    import scripts.run.run_correction_campaign as rcc
    import scripts.validate.run_perfect_model_osse as rpo

    # _build_run_setup + _area_weights are FUNCTION-scope-imported in main() from
    # run_correction_campaign, so patch them THERE; build_perfect_model_osse is module-level.
    dummy_grid = SimpleNamespace(grid_shape_2d=(2, 2), grid_area=jnp.ones(4))
    monkeypatch.setattr(
        rcc, "_build_run_setup",
        lambda args: (object(), dummy_grid, object(), (lambda *a, **k: None),
                      (lambda *a, **k: None), (lambda *a, **k: None), None))
    monkeypatch.setattr(rcc, "_area_weights", lambda grid: jnp.ones(4))

    def _osse_result(recovered):
        return OSSEResult(
            initial_bias=2.0, final_bias=1.0, bias_reduction=1.0, bias_reduced=True,
            true_value=0.4, initial_value=0.6,
            recovered_value=0.42 if recovered else 0.6,
            initial_param_error=0.2, final_param_error=0.02 if recovered else 0.3,
            param_error_reduced=recovered, n_rounds=1, n_accepted=1, summary=None)

    captured = {}

    def _spy(**kw):
        captured.update(kw)
        return _osse_result(recovered=True)
    monkeypatch.setattr(rpo, "build_perfect_model_osse", _spy)

    rc = rpo.main(["--config", "ignored.json", "--true-ck", "0.4", "--biased-ck", "0.6",
                   "--iterations", "1", "--n-worst", "2"])
    assert rc == 0                                         # recovered → GO (exit 0)
    assert float(captured["true_clubb"].C_K) == 0.4        # config-derived TRUE coefficient
    assert float(captured["biased_clubb"].C_K) == 0.6      # the biased start
    assert captured["n_iterations"] == 1 and captured["n_worst"] == 2
    assert captured["les_config"].surface_flux is False    # default off

    # NOT recovered → bias_only/no_change → exit 1 (the campaign-gating no-go).
    monkeypatch.setattr(rpo, "build_perfect_model_osse",
                        lambda **kw: _osse_result(recovered=False))
    assert rpo.main(["--config", "x.json", "--true-ck", "0.4", "--biased-ck", "0.6"]) == 1

    # Single mode REQUIRES --true-ck/--biased-ck (fail-loud dispatch); --coefficients +
    # --fine-resolution is an unsupported combination.
    with pytest.raises(SystemExit, match="requires --true-ck"):
        rpo.main(["--config", "x.json"])
    with pytest.raises(SystemExit, match="cross-resolution"):
        rpo.main(["--config", "x.json", "--coefficients", "C_K", "--fine-resolution", "8"])


def test_assert_pseudo_truth_finite_gates_on_columnstate():
    """_assert_pseudo_truth_finite (iter 302) fails loud on a DIVERGED ColumnState pseudo-truth
    (run_fn(true_config) blew up → NaN), but SKIPS a non-ColumnState reference (the analytic
    ARRAY run_fns the unit tests use) so they are unaffected — the OSSE analog of the iter-301
    baseline divergence guard."""
    import pytest
    from legoesm.training.perfect_model_osse import _assert_pseudo_truth_finite

    # Non-ColumnState references (an analytic grid array; None) are skipped — no raise.
    assert _assert_pseudo_truth_finite(jnp.ones((4, 3)), "x") is None
    assert _assert_pseudo_truth_finite(None, "x") is None
    # A finite ColumnState pseudo-truth passes; a NaN one raises with the DIVERGED hint.
    finite = _full_grid_state(jnp.full((8, 16, 5), 285.0))
    assert _assert_pseudo_truth_finite(finite, "pseudo-truth run") is None
    diverged = _full_grid_state(jnp.full((8, 16, 5), 285.0).at[0, 0, 0].set(jnp.nan))
    with pytest.raises(ValueError, match=r"pseudo-truth run.*DIVERGED"):
        _assert_pseudo_truth_finite(diverged, "pseudo-truth run")


def test_build_cross_resolution_osse_rejects_multi_method():
    """The cross-res builder deploys ONE env->coefficient kernel (single-coefficient): a
    diagnosis_methods list is rejected loudly, mirroring build_perfect_model_osse."""
    import pytest
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    from scripts.validate.run_perfect_model_osse import build_cross_resolution_osse

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    cfg = _base_config()
    with pytest.raises(ValueError, match="single-coefficient"):
        build_cross_resolution_osse(
            build_base_driver=_ck_sensitive_base,
            extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
            base_atm_config_coarse=cfg, coarse_grid=grid, coarse_sigma=sigma,
            area_weights_coarse=jnp.ones((8, 16)),
            base_atm_config_fine=cfg, fine_grid=grid, fine_sigma=sigma,
            area_weights_fine=jnp.ones((8, 16)),
            true_clubb=CLUBBLiteConfig(C_K=0.9), biased_clubb=CLUBBLiteConfig(C_K=0.4),
            les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                       diagnosis_method="clubb_coefficient",
                                       diagnosis_methods=("clubb_coefficient",)),
            run_les_fn=_mock_run_les_sheared, n_worst=1, n_iterations=1)


def test_build_perfect_model_osse_threads_phis(monkeypatch):
    """iter-120 symmetry with the campaign (iter 118): BOTH OSSE builders
    (single + multi) forward the optional static topography `phis` to
    make_les_diagnose_fn, so the OSSE go/no-go can use real terrain.
    make_les_diagnose_fn is stubbed (raising to short-circuit the heavy OSSE run)
    to pin ONLY the phis threading; the default forwards None (flat)."""
    import pytest
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    import scripts.validate.run_perfect_model_osse as ro

    captured = {}

    class _StopError(Exception):
        pass

    def fake_make(grid, sigma, *, les_config, run_les_fn, phis=None):  # noqa: ARG001
        captured["phis"] = phis
        raise _StopError  # short-circuit before the heavy OSSE run

    monkeypatch.setattr(ro, "make_les_diagnose_fn", fake_make)

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    common = dict(
        base_atm_config=_base_config(), build_base_driver=_ck_sensitive_base,
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        sigma=sigma, grid=grid, area_weights=jnp.ones((8, 16)),
        true_clubb=CLUBBLiteConfig(C_K=0.9), biased_clubb=CLUBBLiteConfig(C_K=0.4),
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1, n_iterations=1)

    phis_grid = jnp.zeros((8, 16))
    with pytest.raises(_StopError):
        build_perfect_model_osse(**common, phis=phis_grid)
    assert captured["phis"] is phis_grid          # forwarded to make_les_diagnose_fn
    with pytest.raises(_StopError):
        build_perfect_model_osse(**common)        # default → None (flat)
    assert captured["phis"] is None

    # build_multi_perfect_model_osse threads phis identically (Codex coverage).
    captured.clear()
    multi = {**common,
             "true_clubb": CLUBBLiteConfig(C_K=0.9, Pr_t=0.5, C_eps=0.3),
             "biased_clubb": CLUBBLiteConfig(C_K=0.4, Pr_t=0.33, C_eps=0.1),
             "les_config": ColumnLESConfig(regime=_SMALL_REGIME)}
    with pytest.raises(_StopError):
        build_multi_perfect_model_osse(
            **multi, coefficients=("C_K", "Pr_t", "C_eps"), phis=phis_grid)
    assert captured["phis"] is phis_grid


def test_build_multi_perfect_model_osse_wiring():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)

    res = build_multi_perfect_model_osse(
        base_atm_config=_base_config(),
        build_base_driver=_ck_sensitive_base,
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        sigma=sigma, grid=grid, area_weights=jnp.ones((8, 16)),
        true_clubb=CLUBBLiteConfig(C_K=0.9, Pr_t=0.5, C_eps=0.3),
        biased_clubb=CLUBBLiteConfig(C_K=0.4, Pr_t=0.33, C_eps=0.1),
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1, n_iterations=1,
        coefficients=("C_K", "Pr_t", "C_eps"))

    # All three coefficients are tracked; the monotonic gate invariant holds.
    assert set(res.per_coefficient) == {
        "clubb_lite_C_K", "clubb_lite_Pr_t", "clubb_lite_C_eps"}
    assert res.initial_bias > 0.0
    assert res.final_bias <= res.initial_bias + 1e-9
    assert all(np.isfinite(c.final_param_error)
               for c in res.per_coefficient.values())


def test_build_multi_perfect_model_osse_rejects_unknown_coefficient():
    import pytest
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    with pytest.raises(ValueError, match="unknown coefficient"):
        build_multi_perfect_model_osse(
            base_atm_config=_base_config(),
            build_base_driver=_ck_sensitive_base,
            extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
            sigma=sigma, grid=grid, area_weights=jnp.ones((8, 16)),
            true_clubb=CLUBBLiteConfig(C_K=0.9),
            biased_clubb=CLUBBLiteConfig(C_K=0.4),
            les_config=ColumnLESConfig(regime=_SMALL_REGIME),
            run_les_fn=_mock_run_les_sheared, n_worst=1, n_iterations=1,
            coefficients=("C_K", "bogus"))


def test_build_perfect_model_osse_rejects_unknown_method():
    import pytest
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    with pytest.raises(ValueError, match="unknown diagnosis_method"):
        build_perfect_model_osse(
            base_atm_config=_base_config(),
            build_base_driver=_ck_sensitive_base,
            extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
            sigma=sigma, grid=grid, area_weights=jnp.ones((8, 16)),
            true_clubb=CLUBBLiteConfig(C_K=0.9),
            biased_clubb=CLUBBLiteConfig(C_K=0.4),
            les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                       diagnosis_method="bogus_method"),
            run_les_fn=_mock_run_les_sheared, n_worst=1, n_iterations=1)


def test_osse_main_wiring_monkeypatched(monkeypatch):
    """Drive the OSSE CLI main() with all heavy I/O stubbed: assert it wires the args
    (true_ck/biased_ck → CLUBBLiteConfig, n_iterations) into build_perfect_model_osse,
    computes the REAL verdict, and returns 0 on 'recovered' / 1 otherwise. The go/no-go
    pre-flight's main() was previously untested (pragma: no cover); a wiring regression
    would surface only when an HPC user runs the pre-flight."""
    import pytest
    from legoesm.training.perfect_model_osse import OSSEResult

    import scripts.run.run_correction_campaign as rcc
    import scripts.validate.run_perfect_model_osse as osse_cli

    # Stub the function-scoped imports (resolved from rcc at call time) + the
    # module-level build_perfect_model_osse, so no real driver/twin runs.
    monkeypatch.setattr(rcc, "load_base_config_and_grid",
                        lambda path: (object(), object(), object()))
    monkeypatch.setattr(rcc, "make_base_driver_builder",
                        lambda mode, coupled_preset=None, ocean_grid=None: ((lambda c: None),  # noqa: ARG005
                                                           (lambda d, day, dt: None)))
    monkeypatch.setattr(rcc, "_area_weights", lambda grid: jnp.ones((1, 1)))
    monkeypatch.setattr(rcc, "resolve_orographic_phis",
                        lambda forcing, provider: None)

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return OSSEResult(
            initial_bias=1.0, final_bias=0.4, bias_reduction=0.6, bias_reduced=True,
            true_value=0.9, initial_value=0.4, recovered_value=0.9,
            initial_param_error=0.5, final_param_error=0.05, param_error_reduced=True,
            n_rounds=2, n_accepted=2, summary=None)

    monkeypatch.setattr(osse_cli, "build_perfect_model_osse", fake_build)

    rc = osse_cli.main(["--config", "c.json", "--true-ck", "0.9", "--biased-ck", "0.4",
                        "--iterations", "2", "--n-worst", "1"])
    assert rc == 0                                       # the REAL 'recovered' verdict is ok
    assert float(captured["true_clubb"].C_K) == pytest.approx(0.9)   # default field = C_K
    assert float(captured["biased_clubb"].C_K) == pytest.approx(0.4)
    assert captured["n_iterations"] == 2                # CLI --iterations threaded through


def test_osse_main_returns_nonzero_when_not_recovered(monkeypatch):
    """main() returns 1 when the verdict is NOT ok (e.g. no_change) — the CLI exit code
    is the machine-readable go/no-go an automated pre-flight reads."""
    from legoesm.training.perfect_model_osse import OSSEResult

    import scripts.run.run_correction_campaign as rcc
    import scripts.validate.run_perfect_model_osse as osse_cli

    monkeypatch.setattr(rcc, "load_base_config_and_grid",
                        lambda path: (object(), object(), object()))
    monkeypatch.setattr(rcc, "make_base_driver_builder",
                        lambda mode, coupled_preset=None, ocean_grid=None: ((lambda c: None),  # noqa: ARG005
                                                           (lambda d, day, dt: None)))
    monkeypatch.setattr(rcc, "_area_weights", lambda grid: jnp.ones((1, 1)))
    monkeypatch.setattr(rcc, "resolve_orographic_phis",
                        lambda forcing, provider: None)
    # A no_change result (bias did not fall, no param recovery) → verdict not ok → rc 1.
    monkeypatch.setattr(osse_cli, "build_perfect_model_osse", lambda **kw: OSSEResult(
        initial_bias=1.0, final_bias=1.0, bias_reduction=0.0, bias_reduced=False,
        true_value=0.9, initial_value=0.4, recovered_value=0.4,
        initial_param_error=0.5, final_param_error=0.5, param_error_reduced=False,
        n_rounds=2, n_accepted=0, summary=None))
    rc = osse_cli.main(["--config", "c.json", "--true-ck", "0.9", "--biased-ck", "0.4"])
    assert rc == 1


def test_production_loop_defaults_matches_production():
    """The OSSE twin must use the SAME loop defaults as the production campaign so the
    go/no-go is not FALSELY OPTIMISTIC: clip_to_bounds defaults True (production clips a
    diagnosed coefficient to its registered bounds) and env_grid_fn is wired for
    'environment' (else the loop raises on a None). Crucially setdefault must NOT override
    an EXPLICIT value (an explicit no-clip twin must stay no-clip)."""
    from functools import partial

    from legoesm.training.feedback_assembly import column_environment_grid

    import scripts.validate.run_perfect_model_osse as osse

    # Default static campaign: clip ON, no env grid.
    kw = {}
    osse._production_loop_defaults(kw, sigma=object())
    assert kw["clip_to_bounds"] is True
    assert kw["env_grid_fn"] is None

    # 'environment' strategy → env_grid_fn is the producer bound to sigma.
    sigma = object()
    kw2 = {"feedback_strategy": "environment"}
    osse._production_loop_defaults(kw2, sigma=sigma)
    assert isinstance(kw2["env_grid_fn"], partial)
    assert kw2["env_grid_fn"].func is column_environment_grid
    assert kw2["env_grid_fn"].keywords.get("sigma") is sigma

    # setdefault must NOT override an EXPLICIT clip_to_bounds / env_grid_fn.
    kw3 = {"clip_to_bounds": False, "env_grid_fn": "EXPLICIT"}
    osse._production_loop_defaults(kw3, sigma=object())
    assert kw3["clip_to_bounds"] is False
    assert kw3["env_grid_fn"] == "EXPLICIT"


def test_osse_main_prints_realism_breakdown(monkeypatch, capsys):
    """The OSSE go/no-go reports the LES-realism breakdown (iter 520): when _build_run_setup's
    _RealismCapture accumulated breakdowns, main() prints them with the [osse] prefix — so the
    operator sees WHY any spin-off LES was rejected right at the go/no-go, not just the verdict.
    """
    from types import SimpleNamespace

    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import LESRealismBreakdown
    from legoesm.training.perfect_model_osse import OSSEResult

    import scripts.run.run_correction_campaign as rcc
    import scripts.validate.run_perfect_model_osse as rpo

    t, f = jnp.asarray(True), jnp.asarray(False)
    cap = rcc._RealismCapture(lambda setup: None)
    cap.breakdowns = [
        LESRealismBreakdown(turbulent=t, finite=t, thermo_consistent=t,
                            moisture_physical=t, rh_ok=t, overall=t),
        LESRealismBreakdown(turbulent=f, finite=t, thermo_consistent=t,
                            moisture_physical=t, rh_ok=t, overall=f)]   # 1 laminar
    dummy_grid = SimpleNamespace(grid_shape_2d=(2, 2), grid_area=jnp.ones(4))
    monkeypatch.setattr(rcc, "_build_run_setup",
                        lambda args: (object(), dummy_grid, object(),
                                      (lambda *a, **k: None), (lambda *a, **k: None), cap, None))
    monkeypatch.setattr(rcc, "_area_weights", lambda grid: jnp.ones(4))
    monkeypatch.setattr(rpo, "build_perfect_model_osse", lambda **kw: OSSEResult(
        initial_bias=2.0, final_bias=1.0, bias_reduction=1.0, bias_reduced=True,
        true_value=0.4, initial_value=0.6, recovered_value=0.42,
        initial_param_error=0.2, final_param_error=0.02, param_error_reduced=True,
        n_rounds=1, n_accepted=1, summary=None))

    rpo.main(["--config", "x.json", "--true-ck", "0.4", "--biased-ck", "0.6",
              "--iterations", "1"])
    out = capsys.readouterr().out
    assert "[osse] LES realism: 1/2 realistic" in out and "1x laminar" in out
