"""Direct tests for SWG native-C-grid coarse-graining."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from legoesm.ocean.fidelity import swg_coarse_graining as coarse
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    SWG_MADDISON_NAMELIST,
)


def test_masked_gaussian_preserves_constant_and_handles_walls(monkeypatch):
    calls = 0
    real_filter = coarse.gaussian_filter

    def counted_filter(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_filter(*args, **kwargs)

    monkeypatch.setattr(coarse, "gaussian_filter", counted_filter)
    mask = np.zeros((41, 41), dtype=bool)
    mask[5:36, 7:34] = True
    field = np.where(mask, 3.25, 1.0e20)
    filtered = coarse.masked_gaussian_filter(field, mask, sigma_cells=4.0)
    assert calls == 2
    assert np.allclose(filtered[mask], 3.25, rtol=0.0, atol=2.0e-15)
    assert np.all(filtered[~mask] == 0.0)
    # The planted unnormalised wall treatment cannot preserve this constant.
    planted = coarse.gaussian_filter(
        np.where(mask, 3.25, 0.0), 4.0, mode="constant", cval=0.0)
    assert np.max(np.abs(planted[mask] - 3.25)) > 0.1


def test_gaussian_attenuates_plane_wave_by_analytic_factor():
    n = 512
    sigma = 8.0
    mode = 4
    x = np.arange(n, dtype=np.float64)
    wave = np.cos(2.0 * np.pi * mode * x / n)[None, :] * np.ones((n, 1))
    mask = np.ones((n, n), dtype=bool)
    filtered = coarse.masked_gaussian_filter(wave, mask, sigma_cells=sigma)
    interior = np.s_[48:-48, 48:-48]
    measured = np.sum(filtered[interior] * wave[interior]) / np.sum(
        wave[interior] ** 2)
    expected = np.exp(-0.5 * (sigma * 2.0 * np.pi * mode / n) ** 2)
    assert np.isclose(measured, expected, rtol=2.0e-5, atol=2.0e-5)


def test_round1_diagnostic_is_explicitly_distinct_from_conservative_target(
        monkeypatch):
    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    rng = np.random.default_rng(2)
    constants = {kind: np.full(source["mask"][kind].shape, 7.0)
                 for kind in ("T", "U", "V")}
    constant_out = coarse.coarse_grain_cgrid(
        constants, source, target,
        gaussian_length_m=float(target["spacing_m"]))
    for kind in ("T", "U", "V"):
        assert np.allclose(constant_out[kind][target["mask"][kind]], 7.0,
                           rtol=0.0, atol=2.0e-15)
    fields = {kind: rng.normal(size=source["mask"][kind].shape)
              for kind in ("T", "U", "V")}
    diagnostic = coarse.coarse_grain_cgrid(
        fields, source, target,
        gaussian_length_m=float(target["spacing_m"]))
    conservative = coarse.conservative_restrict_cgrid(fields, source, target)
    for kind in ("T", "U", "V"):
        assert np.all(diagnostic[kind][~target["mask"][kind]] == 0.0)
    assert any(not np.array_equal(diagnostic[kind], conservative[kind])
               for kind in ("T", "U", "V"))
    # Removing the round-1 mean patch must measurably change the diagnostic;
    # this proves the test does not merely assert a value forced by the code.
    monkeypatch.setattr(
        coarse, "_restore_mean",
        lambda field, mask, area, target_mean: np.where(mask, field, 0.0))
    unpatched = coarse.coarse_grain_cgrid(
        fields, source, target,
        gaussian_length_m=float(target["spacing_m"]))
    assert any(
        abs(np.mean(diagnostic[kind][target["mask"][kind]])
            - np.mean(unpatched[kind][target["mask"][kind]])) > 1.0e-8
        for kind in ("T", "U", "V"))


def _synthetic_state(source, seed=8):
    rng = np.random.default_rng(seed)
    return {
        kind: rng.normal(scale=0.01, size=source["mask"][kind].shape)
        for kind in ("T", "U", "V")
    }


def test_conservative_restriction_preserves_volume_and_continuity_at_walls():
    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    fields = _synthetic_state(source)
    output = coarse.conservative_restrict_cgrid(fields, source, target)
    source_wet = source["mask"]["T"]
    target_wet = target["mask"]["T"]
    source_volume = np.sum(
        source["area"]["T"][source_wet]
        * (source["reference_thickness"]["T"][source_wet]
           + fields["T"][source_wet]), dtype=np.float64)
    target_volume = np.sum(
        target["area"]["T"][target_wet]
        * (target["reference_thickness"]["T"][target_wet]
           + output["T"][target_wet]), dtype=np.float64)
    assert np.isclose(source_volume, target_volume, rtol=2.0e-16, atol=0.0)
    result = coarse.require_continuity_commutes(
        fields, source, target, atol_m_s=1.0e-15)
    assert result["interior"]["max_abs_m_s"] <= 1.0e-15
    assert result["outer_four_cells"]["max_abs_m_s"] <= 1.0e-15


def test_jax_restrictor_matches_numpy_eager_jit_and_has_a_tangent():
    import jax
    import jax.numpy as jnp

    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    fields = _synthetic_state(source, seed=10)
    expected = coarse.conservative_restrict_cgrid(fields, source, target)
    restrict = coarse.build_jax_conservative_restrictor(source, target)
    arguments = tuple(jnp.asarray(fields[kind]) for kind in ("T", "U", "V"))
    eager = restrict(*arguments)
    compiled = jax.jit(restrict)(*arguments)
    for kind, eager_value, compiled_value in zip(
            ("T", "U", "V"), eager, compiled, strict=True):
        # T subtracts the 500 m reference depth after a differently ordered
        # block sum in NumPy and XLA; the measured roundoff is 5.12e-13 m.
        atol = 6.0e-13 if kind == "T" else 2.0e-15
        np.testing.assert_allclose(
            np.asarray(eager_value), expected[kind], rtol=2.0e-15, atol=atol)
        np.testing.assert_allclose(
            np.asarray(compiled_value), expected[kind],
            rtol=2.0e-15, atol=atol)
        planted = np.asarray(eager_value).copy()
        planted[target["mask"][kind]] += 2.0e-12
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                planted, expected[kind], rtol=2.0e-15, atol=atol)
    tangents = tuple(jnp.ones_like(value) for value in arguments)
    _, tangent = jax.jvp(restrict, arguments, tangents)
    assert all(np.all(np.isfinite(np.asarray(value))) for value in tangent)
    assert any(np.any(np.asarray(value) != 0.0) for value in tangent)


def test_filtered_extensives_commute_with_continuity_at_walls_and_corners():
    rng = np.random.default_rng(904)
    ny, nx = 73, 79
    volume = rng.normal(size=(ny, nx))
    u = np.zeros((ny, nx + 1), dtype=np.float64)
    v = np.zeros((ny + 1, nx), dtype=np.float64)
    u[:, 1:-1] = rng.normal(size=(ny, nx - 1))
    v[1:-1, :] = rng.normal(size=(ny - 1, nx))
    divergence = u[:, 1:] - u[:, :-1] + v[1:] - v[:-1]
    filtered_divergence, _, _ = coarse.filter_closed_cgrid_extensives(
        divergence, np.zeros_like(u), np.zeros_like(v), sigma_cells=4.0)
    filtered_volume, filtered_u, filtered_v = (
        coarse.filter_closed_cgrid_extensives(
            volume, u, v, sigma_cells=4.0))
    diagnosed = (filtered_u[:, 1:] - filtered_u[:, :-1]
                 + filtered_v[1:] - filtered_v[:-1])
    residual = diagnosed - filtered_divergence
    wall = np.zeros_like(volume, dtype=bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    assert np.max(np.abs(residual[~wall])) < 2.0e-15
    assert np.max(np.abs(residual[wall])) < 2.0e-15
    assert np.all(filtered_u[:, (0, -1)] == 0.0)
    assert np.all(filtered_v[(0, -1), :] == 0.0)
    assert np.isclose(np.sum(filtered_volume), np.sum(volume),
                      rtol=0.0, atol=2.0e-13)

    # Plant the tempting even-reflection treatment on a normal transport.
    kernel = coarse.gaussian_kernel_1d(sigma_cells=4.0)
    radius = kernel.size // 2
    wrong_u = coarse._convolve_valid_axis(
        coarse._pad_even_cell_centres(u, radius, 1), kernel, 1)
    wrong = wrong_u[:, 1:] - wrong_u[:, :-1]
    assert np.max(np.abs(wrong[:, (0, -1)]
                         - filtered_divergence[:, (0, -1)])) > 1.0e-3


def test_shifted_filter_lowering_has_no_convolution_and_guard_catches_plant():
    import jax
    import jax.numpy as jnp
    from legoesm.training.swg_online import assert_no_convolution_operations

    kernel = jnp.asarray(coarse.gaussian_kernel_1d(sigma_cells=1.0))
    volume = jnp.arange(64.0, dtype=jnp.float64).reshape(8, 8)
    u = jnp.zeros((8, 9), dtype=jnp.float64)
    v = jnp.zeros((9, 8), dtype=jnp.float64)
    lowered = jax.jit(lambda a, b, c:
        coarse.filter_closed_cgrid_extensives_array(
            a, b, c, kernel=kernel)).lower(volume, u, v)
    assert_no_convolution_operations(lowered, label="shifted filter plant control")

    dense_kernel = (kernel[:, None] * kernel[None, :])[None, None]
    padded = jnp.pad(volume, ((3, 3), (3, 3)), mode="symmetric")
    planted = jax.jit(lambda value: jax.lax.conv_general_dilated(
        value[None, None], dense_kernel, (1, 1), "VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"))[0, 0]).lower(padded)
    with pytest.raises(RuntimeError, match="forbidden convolution"):
        assert_no_convolution_operations(planted, label="planted dense filter")


def test_numpy_filter_refuses_silent_float64_downcast_without_jax_x64():
    program = """
import numpy as np
from legoesm.ocean.fidelity.swg_coarse_graining import filter_closed_cgrid_extensives
volume = np.zeros((7, 7), dtype=np.float64)
u = np.zeros((7, 8), dtype=np.float64)
v = np.zeros((8, 7), dtype=np.float64)
try:
    filter_closed_cgrid_extensives(volume, u, v, sigma_cells=1.0, truncate=2.0)
except RuntimeError as exc:
    assert "JAX_ENABLE_X64=1" in str(exc)
else:
    raise AssertionError("float64 restriction silently ran with JAX x64 disabled")
"""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [sys.executable, "-c", program], env=environment,
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_filtered_extensive_transfer_function_and_temporal_continuity_plant():
    n = 256
    sigma = 8.0
    wavelength = 64.0
    x = np.arange(n, dtype=np.float64)
    volume = np.cos(2.0 * np.pi * x / wavelength)[None, :] * np.ones((n, 1))
    u = np.zeros((n, n + 1), dtype=np.float64)
    v = np.zeros((n + 1, n), dtype=np.float64)
    filtered, _, _ = coarse.filter_closed_cgrid_extensives(
        volume, u, v, sigma_cells=sigma)
    interior = np.s_[64:-64, 64:-64]
    measured = np.sum(filtered[interior] * volume[interior]) / np.sum(
        volume[interior] ** 2)
    analytic = np.exp(-2.0 * np.pi**2 * sigma**2 / wavelength**2)
    assert measured == pytest.approx(analytic, rel=3.0e-5)

    rng = np.random.default_rng(611)
    u[:, 1:-1] = rng.normal(size=(n, n - 1))
    v[1:-1] = rng.normal(size=(n - 1, n))
    divergence = u[:, 1:] - u[:, :-1] + v[1:] - v[:-1]
    end = volume - divergence
    start_f, _, _ = coarse.filter_closed_cgrid_extensives(
        volume, np.zeros_like(u), np.zeros_like(v), sigma_cells=sigma)
    end_f, u_f, v_f = coarse.filter_closed_cgrid_extensives(
        end, u, v, sigma_cells=sigma)
    residual = end_f - start_f + u_f[:, 1:] - u_f[:, :-1] \
        + v_f[1:] - v_f[:-1]
    assert np.max(np.abs(residual)) < 2.0e-15
    wrong_end = end_f + 1.0
    planted = wrong_end - start_f + u_f[:, 1:] - u_f[:, :-1] \
        + v_f[1:] - v_f[:-1]
    assert np.min(np.abs(planted)) > 0.99


def test_filtered_restriction_jax_matches_numpy_and_uses_native_staggering():
    import jax
    import jax.numpy as jnp

    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    fields = _synthetic_state(source, seed=505)
    expected = coarse.filtered_restrict_cgrid(
        fields, source, target, sigma_source_cells=8.0)
    restrict = coarse.build_jax_filtered_restrictor(
        source, target, sigma_source_cells=8.0)
    arguments = tuple(jnp.asarray(fields[kind]) for kind in ("T", "U", "V"))
    actual = jax.jit(restrict)(*arguments)
    jax.block_until_ready(actual[0])
    for kind, value in zip(("T", "U", "V"), actual, strict=True):
        # The shifted filter plus compiled block restriction differs from the
        # shared offline path by at most 2.0473e-12 m in the final subtraction.
        tolerance = 2.1e-12 if kind == "T" else 3.0e-15
        np.testing.assert_allclose(
            np.asarray(value), expected[kind], rtol=3.0e-15, atol=tolerance)
    # Swapping native face components is a staggering violation, not a
    # symmetry of this anisotropic manufactured state.
    swapped = coarse.filtered_restrict_cgrid(
        {"T": fields["T"], "U": fields["V"], "V": fields["U"]},
        source, target, sigma_source_cells=8.0)
    assert np.max(np.abs(swapped["U"] - expected["U"])) > 1.0e-5


def test_filtered_restriction_temporal_continuity_including_coarse_walls():
    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    rng = np.random.default_rng(606)
    transports = {
        kind: np.where(source["mask"][kind],
                       rng.normal(scale=2.0e3,
                                  size=source["mask"][kind].shape), 0.0)
        for kind in ("U", "V")}
    interval = 3_600.0
    eta0 = np.zeros_like(source["mask"]["T"], dtype=np.float64)
    eta1 = eta0 - interval * coarse.transport_divergence(transports, source)

    def state(eta):
        result = {"T": eta}
        for kind in ("U", "V"):
            cross = (source["face_length"][kind]
                     * coarse.live_face_thickness(eta, source, kind))
            result[kind] = np.where(
                source["mask"][kind], transports[kind] / cross, 0.0)
        return result

    restricted0 = coarse.filtered_restrict_cgrid(
        state(eta0), source, target, sigma_source_cells=8.0)
    restricted1 = coarse.filtered_restrict_cgrid(
        state(eta1), source, target, sigma_source_cells=8.0)
    restricted_q = coarse.filtered_restrict_transports(
        transports, source, target, sigma_source_cells=8.0)
    residual = coarse.temporal_continuity_residual(
        restricted0["T"], restricted1["T"], restricted_q, target, interval)
    wet = target["mask"]["T"]
    jj, ii = np.where(wet)
    wall = np.zeros_like(wet)
    wall[jj.min(), ii.min():ii.max() + 1] = True
    wall[jj.max(), ii.min():ii.max() + 1] = True
    wall[jj.min():jj.max() + 1, ii.min()] = True
    wall[jj.min():jj.max() + 1, ii.max()] = True
    assert np.max(np.abs(residual[wet & ~wall])) < 2.0e-16
    assert np.max(np.abs(residual[wall])) < 2.0e-16
    planted = restricted1["T"].copy()
    planted[wet] += 1.0
    wrong = coarse.temporal_continuity_residual(
        restricted0["T"], planted, restricted_q, target, interval)
    assert np.max(np.abs(wrong[wet])) >= 1.0 / interval


def test_daily_mean_restricts_mean_transport_not_product_of_means():
    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    first = _synthetic_state(source, seed=31)
    second = _synthetic_state(source, seed=32)
    # Amplify eta/velocity covariance so the forbidden product of means is
    # decisively different from mean(h*u).
    first["T"] += 2.0
    first["U"] += 1.0
    first["V"] -= 1.0
    second["T"] -= 2.0
    second["U"] -= 1.0
    second["V"] += 1.0
    transports = [coarse.native_face_transports(state, source)
                  for state in (first, second)]
    mean_transport = {
        kind: 0.5 * (transports[0][kind] + transports[1][kind])
        for kind in ("U", "V")
    }
    mean_fields = {kind: 0.5 * (first[kind] + second[kind])
                   for kind in ("T", "U", "V")}
    exact = coarse.conservative_restrict_daily_mean(
        mean_fields["T"], mean_transport, source, target)
    forbidden = coarse.conservative_restrict_cgrid(mean_fields, source, target)
    assert np.max(np.abs(exact["U"] - forbidden["U"])) > 1.0e-4
    assert np.max(np.abs(exact["V"] - forbidden["V"])) > 1.0e-4


def test_temporal_continuity_detects_wrong_ssh_restriction():
    geometry = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    rng = np.random.default_rng(84)
    transports = {
        kind: np.where(geometry["mask"][kind],
                       rng.normal(size=geometry["mask"][kind].shape), 0.0)
        for kind in ("U", "V")
    }
    interval = 86_400.0
    start = np.zeros_like(geometry["mask"]["T"], dtype=np.float64)
    divergence = coarse.transport_divergence(transports, geometry)
    end = start - interval * divergence
    residual = coarse.temporal_continuity_residual(
        start, end, transports, geometry, interval)
    assert np.max(np.abs(residual)) <= 2.0e-22
    wrong = end.copy()
    wrong[geometry["mask"]["T"]] += 1.0
    planted = coarse.temporal_continuity_residual(
        start, wrong, transports, geometry, interval)
    assert np.max(np.abs(planted)) >= 1.0 / interval


def test_jax_live_transport_diagnostic_matches_numpy():
    import jax
    import jax.numpy as jnp

    geometry = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    fields = _synthetic_state(geometry, seed=45)
    expected = coarse.native_face_transports(fields, geometry)
    diagnostic = jax.jit(coarse.build_jax_native_transport_diagnostic(
        SWG_DG50_NAMELIST))
    actual = diagnostic(*(jnp.asarray(fields[kind])
                          for kind in ("T", "U", "V")))
    for kind, value in zip(("U", "V"), actual, strict=True):
        np.testing.assert_allclose(
            np.asarray(value), expected[kind], rtol=2.0e-15, atol=2.0e-15)


def test_truth_source_decks_are_explicit_and_nested_at_dg50():
    fine = coarse.truth_source_namelist("dg625")
    sensitivity = coarse.truth_source_namelist("dg3125")
    assert fine.nn_SWG == 16 and sensitivity.nn_SWG == 32
    assert fine.rn_Dt_s == 50.0 and sensitivity.rn_Dt_s == 25.0
    assert (fine.nn_SWG // 2, sensitivity.nn_SWG // 2) == (8, 16)
    with pytest.raises(ValueError, match="unsupported truth source"):
        coarse.truth_source_namelist("typo")


def test_analytic_cgrid_face_staggering_selects_tiling_faces():
    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    rows, columns = np.indices(source["mask"]["T"].shape)
    fields = {
        "T": np.zeros_like(rows, dtype=np.float64),
        "U": rows.astype(np.float64) + 0.01 * columns,
        "V": 2.0 * columns.astype(np.float64) + 0.01 * rows,
    }
    output = coarse.conservative_restrict_cgrid(fields, source, target)
    # First coarse east face tiles fine native-U column 9 over T rows 2..9.
    assert np.isclose(output["U"][1, 1], np.mean(fields["U"][2:10, 9]),
                      rtol=0.0, atol=2.0e-15)
    # First coarse north face tiles fine native-V row 9 over T columns 2..9.
    assert np.isclose(output["V"][1, 1], np.mean(fields["V"][9, 2:10]),
                      rtol=0.0, atol=2.0e-15)
    # Adjacent-face plants are deliberately distinguishable.
    assert output["U"][1, 1] != np.mean(fields["U"][2:10, 8])
    assert output["V"][1, 1] != np.mean(fields["V"][8, 2:10])


def test_gaussian_vorticity_path_spreads_impulse_without_mean_patch():
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    wet = target["mask"]["T"]
    field = np.zeros(wet.shape, dtype=np.float64)
    at = tuple(np.argwhere(wet)[0])
    field[at] = 1.0
    smoothed = coarse.gaussian_smooth_restricted_scalar(
        field, wet, sigma_cells=1.0)
    assert 0.0 < smoothed[at] < 1.0
    assert np.count_nonzero(smoothed[wet]) > 1
    assert np.isclose(np.sum(smoothed[wet]), 1.0, rtol=0.0, atol=2.0e-15)
    fields = {"T": field, "U": np.zeros_like(field), "V": np.zeros_like(field)}
    u_at = tuple(np.argwhere(target["mask"]["U"])[100])
    fields["U"][u_at] = 1.0
    raw = coarse.restricted_relative_vorticity(fields, target)
    filtered = coarse.restricted_relative_vorticity(
        fields, target, gaussian_sigma_cells=1.0)
    assert raw.shape == filtered.shape == wet.shape
    assert not np.array_equal(raw, filtered)


def test_planted_round1_nonconservative_state_is_refused():
    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    fields = _synthetic_state(source, seed=9)
    planted = coarse.coarse_grain_cgrid(
        fields, source, target,
        gaussian_length_m=float(target["spacing_m"]))
    with pytest.raises(ValueError, match="non-conservative restriction"):
        coarse.require_continuity_commutes(
            fields, source, target, atol_m_s=1.0e-15,
            coarse_fields=planted)


_REAL_FINE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/"
    "legoesm_runs/dg_gpu_1095d_daily_mean")
_REAL_FINE_SNAPSHOT_ROOT = _REAL_FINE_ROOT.with_name("dg_gpu_1095d")


@pytest.mark.skipif(not (_REAL_FINE_SNAPSHOT_ROOT / "manifest.json").is_file(),
                    reason="complete three-year fine snapshots not mounted")
def test_real_day1095_shifted_filter_matches_round5_dense_result():
    import jax
    import jax.numpy as jnp

    manifest = json.loads(
        (_REAL_FINE_SNAPSHOT_ROOT / "manifest.json").read_text())
    path = _REAL_FINE_SNAPSHOT_ROOT / "day1095.npz"
    assert manifest["daily_sha256"][path.name] == coarse.sha256(path)
    with np.load(path, allow_pickle=False) as saved:
        fields = {kind: np.asarray(saved[name], dtype=np.float64)
                  for kind, name in (("T", "ssh_m"), ("U", "u_m_s"),
                                     ("V", "v_m_s"))}
    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    layout = coarse._restriction_layout(source, target)
    local = coarse._local_closed_extensives(fields, source, layout)
    kernel = jnp.asarray(coarse.gaussian_kernel_1d(sigma_cells=8.0))
    radius = int(kernel.size // 2)
    kernel2 = (kernel[:, None] * kernel[None, :])[None, None]

    def old_dense(value, parities):
        for axis, parity in enumerate(parities):
            if parity == "even":
                padding = [(0, 0), (0, 0)]
                padding[axis] = (radius, radius)
                value = jnp.pad(value, padding, mode="symmetric")
            else:
                indices = jnp.arange(1, radius + 1)
                value = jnp.concatenate((
                    -jnp.take(value, indices[::-1], axis=axis), value,
                    -jnp.take(value, value.shape[axis] - 1 - indices,
                              axis=axis)), axis=axis)
        return jax.lax.conv_general_dilated(
            value[None, None], kernel2, (1, 1), "VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"))[0, 0]

    dense = [np.asarray(jax.jit(
        lambda value, parity=parities: old_dense(value, parity))(
            jnp.asarray(value))).copy()
        for value, parities in zip(
            local, (("even", "even"), ("even", "odd"),
                    ("odd", "even")), strict=True)]
    dense[1][:, (0, -1)] = 0.0
    dense[2][(0, -1), :] = 0.0
    sj0, sj1 = layout["source_j0"], layout["source_j1"]
    si0, si1 = layout["source_i0"], layout["source_i1"]
    volume = np.zeros_like(source["area"]["T"])
    volume[sj0:sj1 + 1, si0:si1 + 1] = dense[0]
    transports = {"U": np.zeros_like(volume), "V": np.zeros_like(volume)}
    transports["U"][sj0:sj1 + 1, si0:si1 + 1] = dense[1][:, 1:]
    transports["V"][sj0:sj1 + 1, si0:si1 + 1] = dense[2][1:, :]
    old_state = coarse._state_from_restricted_extensives(
        coarse._restrict_t_cell_sum(volume, source, target, layout),
        coarse.restrict_native_transports(
            transports, source, target, layout), target)
    shifted_state = coarse.filtered_restrict_cgrid(
        fields, source, target, sigma_source_cells=8.0)
    tolerances = {"T": 1.3e-12, "U": 3.0e-15, "V": 2.0e-15}
    measured = {
        kind: float(np.max(np.abs(shifted_state[kind] - old_state[kind])))
        for kind in ("T", "U", "V")}
    assert measured == pytest.approx({
        "T": 1.1937117960769683e-12,
        "U": 2.6645352591003757e-15,
        "V": 1.887379141862766e-15,
    }, rel=1.0e-12, abs=0.0)
    for kind in ("T", "U", "V"):
        np.testing.assert_allclose(
            shifted_state[kind], old_state[kind], rtol=0.0,
            atol=tolerances[kind])
    planted = shifted_state["T"].copy()
    planted[target["mask"]["T"]] += 2.0e-11
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            planted, old_state["T"], rtol=0.0, atol=tolerances["T"])


@pytest.mark.skipif(not (_REAL_FINE_ROOT / "manifest.json").is_file(),
                    reason="complete three-year fine run not mounted")
def test_real_day1095_continuity_commutes_interior_and_walls():
    manifest_path = _REAL_FINE_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    name = "day1095.npz"
    path = _REAL_FINE_ROOT / name
    assert manifest["status"] == "complete"
    assert manifest["daily_sha256"][name] == coarse.sha256(path)
    report = coarse.audit_snapshot(
        path, manifest_path, commit="a" * 40)
    assert report["status"] == "PASS"
    assert report["regions"]["interior"]["max_abs_m_s"] <= 1.0e-15
    assert report["regions"]["outer_four_cells"]["max_abs_m_s"] <= 1.0e-15


@pytest.mark.skipif(not (_REAL_FINE_ROOT / "manifest.json").is_file(),
                    reason="complete three-year fine run not mounted")
def test_real_day1095_filtered_continuity_commutes_on_coarse_walls():
    manifest = json.loads((_REAL_FINE_ROOT / "manifest.json").read_text())
    path = _REAL_FINE_ROOT / "day1095.npz"
    assert manifest["daily_sha256"][path.name] == coarse.sha256(path)
    with np.load(path, allow_pickle=False) as saved:
        fields = {kind: np.asarray(saved[name], dtype=np.float64)
                  for kind, name in (("T", "ssh_m"), ("U", "u_m_s"),
                                     ("V", "v_m_s"))}
    source = coarse.cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = coarse.cgrid_geometry(SWG_DG50_NAMELIST)
    layout = coarse._restriction_layout(source, target)
    volume, u, v = coarse._local_closed_extensives(fields, source, layout)
    divergence = u[:, 1:] - u[:, :-1] + v[1:] - v[:-1]
    filtered_divergence, _, _ = coarse.filter_closed_cgrid_extensives(
        divergence, np.zeros_like(u), np.zeros_like(v), sigma_cells=8.0)
    _, filtered_u, filtered_v = coarse.filter_closed_cgrid_extensives(
        volume, u, v, sigma_cells=8.0)
    sj0, sj1 = layout["source_j0"], layout["source_j1"]
    si0, si1 = layout["source_i0"], layout["source_i1"]
    embedded_divergence = np.zeros_like(source["area"]["T"])
    embedded_divergence[sj0:sj1 + 1, si0:si1 + 1] = filtered_divergence
    embedded_transport = {
        "U": np.zeros_like(embedded_divergence),
        "V": np.zeros_like(embedded_divergence),
    }
    embedded_transport["U"][sj0:sj1 + 1, si0:si1 + 1] = filtered_u[:, 1:]
    embedded_transport["V"][sj0:sj1 + 1, si0:si1 + 1] = filtered_v[1:, :]
    restricted_divergence = coarse._restrict_t_cell_sum(
        embedded_divergence, source, target, layout) / target["area"]["T"]
    coarse_transport = coarse.restrict_native_transports(
        embedded_transport, source, target, layout)
    residual = coarse.transport_divergence(coarse_transport, target) \
        - restricted_divergence
    wet = target["mask"]["T"]
    jj, ii = np.where(wet)
    wall = np.zeros_like(wet)
    wall[jj.min(), ii.min():ii.max() + 1] = True
    wall[jj.max(), ii.min():ii.max() + 1] = True
    wall[jj.min():jj.max() + 1, ii.min()] = True
    wall[jj.min():jj.max() + 1, ii.max()] = True
    assert np.max(np.abs(residual[wet & ~wall])) < 1.0e-15
    assert np.max(np.abs(residual[wall])) < 1.0e-15
