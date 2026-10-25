"""SI3 rung O5 model preparation: ORCA2 five-category H79 strength + aEVP on the tripole.

Synthetic ORCA2-shaped fixtures (cyclic seam, T-pivot fold, closed south,
coast) exercise the exact-input functions; the lane-3 cards are controlled
against this round's parent through their production dispatchers.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice.dynamics import (
    SI3CGridAEVPConfig,
    SI3CGridAEVPForcing,
    SI3CGridAEVPState,
    SI3CGridMetrics,
    SI3ORCA2FoldProgram,
    SI3ORCA2StrengthConfig,
    si3_orca2_aevp_trace,
    si3_orca2_aevp_solver,
    si3_orca2_ice_strength,
    si3_var_sshdyn,
)
from legoesm.ice.si3_orca2_state import si3_lbc_lnk_orca2

from legoesm import constants

REPO_ROOT = Path(__file__).resolve().parents[3]
O5_PARENT_COMMIT = "1bb17df9ddcc35037c72f81e9dba1de853a46798"
NN_HLS = 2
JPI, JPJ = 16, 12  # Ni0glo = 12 (even, as the T-pivot fold needs)
# ORCA2 namelist_ice_ref:76-78,83 and :110-114 (H79, smoothing, aEVP).
STRENGTH_CFG = SI3ORCA2StrengthConfig(ln_str_h79=True, rn_pstar=2.0e4, rn_crhg=20.0, ln_str_smooth=True)
TEST_SUBCYCLES = 4  # test-local; ORCA2 resolves nn_nevp = 100
LABELS = ("fimask", "zdelta", "zp_delt", "u_ice", "v_ice", "zs1", "zs2", "zs12")


@pytest.fixture(autouse=True)
def _libm():
    """DECISION 162: SI3 rungs certify under the glibc exp arm."""
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(previous)


def _sha(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(np.asarray(array, dtype=np.float64)).tobytes())
    return digest.hexdigest()


def _lbc(field, nature, sign=1.0):
    return si3_lbc_lnk_orca2(jnp.asarray(field), nature, sign, nn_hls=NN_HLS, ldfull=True)


def _fold(full: bool) -> SI3ORCA2FoldProgram:
    return SI3ORCA2FoldProgram(nn_hls=NN_HLS, nfd_full_when_absent=full)


def _tmask() -> np.ndarray:
    tmask = np.ones((JPI, JPJ))
    tmask[5:8, 4:6] = 0.0  # an island
    return np.asarray(_lbc(tmask, "T"))


def _uv_masks(tmask):
    umask = tmask * np.roll(tmask, -1, axis=0)
    vmask = tmask * np.roll(tmask, -1, axis=1)
    return np.asarray(_lbc(umask, "U")), np.asarray(_lbc(vmask, "V"))


def _categories(seed: int):
    rng = np.random.default_rng(seed)
    a_i = rng.uniform(0.05, 0.19, size=(JPI, JPJ, 5))
    h_i = rng.uniform(0.3, 3.7, size=(JPI, JPJ, 5))
    tmask = _tmask()
    a_i = a_i * tmask[..., None]
    a_i[2, 3, :] = 1.0e-12  # one ice-free wet column (at_i <= epsi10 branch)
    v_i = a_i * h_i
    a_i = np.stack([np.asarray(_lbc(a_i[..., k], "T")) for k in range(5)], axis=-1)
    v_i = np.stack([np.asarray(_lbc(v_i[..., k], "T")) for k in range(5)], axis=-1)
    return a_i, v_i, tmask


def _metrics(rng) -> SI3CGridMetrics:
    def metric(nature, base):
        raw = base * (1.0 + 0.1 * rng.uniform(size=(JPI, JPJ)))
        # domain_cfg metrics are finite in the closed south halo; only seam/fold apply.
        return _lbc(raw, nature).at[:, :NN_HLS].set(raw[:, :NN_HLS])

    e1t, e2t = metric("T", 1.9e5), metric("T", 2.1e5)
    e1u, e2u = metric("U", 1.95e5), metric("U", 2.05e5)
    e1v, e2v = metric("V", 1.85e5), metric("V", 2.15e5)
    e1f, e2f = metric("F", 1.9e5), metric("F", 2.0e5)
    return SI3CGridMetrics(
        e1t=e1t, e2t=e2t, e1u=e1u, e2u=e2u, e1v=e1v, e2v=e2v, e1f=e1f, e2f=e2f,
        area_t=e1t * e2t, area_u=e1u * e2u, area_v=e1v * e2v, area_f=e1f * e2f,
    )


def _aevp_config(n_subcycles: int) -> SI3CGridAEVPConfig:
    return SI3CGridAEVPConfig(
        scheme="si3_aevp", staggering="si3_c_grid", dt_s=10800.0, n_subcycles=n_subcycles,
        eccentricity=2.0, creep_limit_s_inv=2.0e-9, strength_parameter_pa=2.0e4,
        strength_decay=20.0, rho_snow=constants.rho_snow, rho_ice=constants.rho_ice,
        rho_water=constants.rho_water, rho_ocean=constants.rho_ocean_nemo,
        gravity=constants.g_nemo, rn_ishlat=2.0, halo_width=NN_HLS, category_count=5,
        landfast=False, landfast_depth_fraction=0.125, landfast_basal_friction_n_m3=15.0,
        landfast_relaxation_s_inv=1.0e-5, landfast_tensile_fraction=0.05, convergence_check=0,
    )


def _aevp_inputs(seed: int = 7):
    rng = np.random.default_rng(seed)
    a_i, v_i, tmask = _categories(seed)
    umask, vmask = _uv_masks(tmask)
    strength = si3_orca2_ice_strength(a_i=a_i, v_i=v_i, tmask=tmask, config=STRENGTH_CFG, fold=_fold(True))
    zero = jnp.zeros((JPI, JPJ))

    def field(nature, scale, sign=1.0, mask=None):
        value = scale * rng.standard_normal((JPI, JPJ))
        if mask is not None:
            value = value * mask
        return _lbc(value, nature, sign)

    vt_i = jnp.sum(jnp.asarray(v_i), axis=-1)
    forcing = SI3CGridAEVPForcing(
        concentration_t=strength.at_i, ice_volume_t=vt_i,
        snow_volume_t=0.2 * vt_i, pond_volume_t=0.01 * vt_i, lid_volume_t=0.001 * vt_i,
        air_stress_u_t=field("T", 0.1), air_stress_v_t=field("T", 0.1),
        drag_io_t=_lbc(5.0e-3 * (1.0 + rng.uniform(size=(JPI, JPJ))), "T"),
        ocean_u_u=field("U", 0.05, -1.0, umask), ocean_v_v=field("V", 0.05, -1.0, vmask),
        ssh_t=None, coriolis_t=_lbc(1.4e-4 * np.ones((JPI, JPJ)), "T"),
        tmask_t=jnp.asarray(tmask), umask_u=jnp.asarray(umask), vmask_v=jnp.asarray(vmask),
        fast_tmask=zero, depth_t=zero, depth_u=zero, depth_v=zero,
        iceberg_tmask=zero, iceberg_umask=zero, iceberg_vmask=zero,
    )
    state = SI3CGridAEVPState(
        u_ice_u=field("U", 0.1, -1.0, umask), v_ice_v=field("V", 0.1, -1.0, vmask),
        stress1_t=field("T", 1.0e3), stress2_t=field("T", 1.0e3), stress12_f=field("F", 1.0e3),
    )
    entry = dict(
        ssh_m=field("T", 0.3), snwice_mass=field("T", 300.0), snwice_mass_b=field("T", 300.0),
        # previous-step velocities, deliberately different from the entry state
        u_ice_b=field("U", 0.1, -1.0, umask), v_ice_b=field("V", 0.1, -1.0, vmask),
    )
    return state, forcing, _metrics(rng), strength.strength, entry


def _run_aevp(*, n=TEST_SUBCYCLES, full=False, differentiable=False, **plants):
    state, forcing, metrics, strength, entry = _aevp_inputs()
    return si3_orca2_aevp_solver(
        state, forcing, metrics, _aevp_config(n), strength_t=strength, ln_ice_embd=False,
        fold=_fold(full), differentiable=differentiable, **entry, **plants,
    )


# --- fold: the F nature added to O1's shared lbc ------------------------------


def test_f_fold_matches_lbcnfd_index_formulas():
    """lbcnfd.f90:722-747 written as explicit 1-based loops, independent of the model."""
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((JPI, JPJ))
    for full in (True, False):
        got = np.asarray(si3_lbc_lnk_orca2(jnp.asarray(raw), "F", -1.0, nn_hls=NN_HLS, ldfull=full))
        ref = raw.copy()
        ref[:NN_HLS, :] = raw[JPI - 2 * NN_HLS : JPI - NN_HLS, :]
        ref[JPI - NN_HLS :, :] = ref[NN_HLS : 2 * NN_HLS, :]
        ipi, ipj, ihls, ni0 = JPI, JPJ, NN_HLS, JPI - 2 * NN_HLS
        for jj in range(1, ihls + 2 if full else ihls + 1):
            ij1, ij2 = ipj - jj + 1, ipj - 2 * ihls + jj - 2
            src = ref[:, ij2 - 1].copy()
            for ji in range(1, ihls + 1):
                ref[ji - 1, ij1 - 1] = -src[2 * ihls + 1 - ji - 1]
            for ji in range(1, ni0 + 1):
                ref[ihls + ji - 1, ij1 - 1] = -src[ipi - ihls - ji + 1 - 1]
            for ji in range(1, ihls + 1):
                ref[ipi - ihls + ji - 1, ij1 - 1] = -src[ipi - ihls - ji + 1 - 1]
        ref[:, :NN_HLS] = 0.0
        np.testing.assert_array_equal(got, ref)


# --- (a) category area/volume -> strength (+ smoothing) ------------------------


def test_strength_matches_independent_numpy_transcription():
    a_i, v_i, tmask = _categories(3)
    out = si3_orca2_ice_strength(a_i=a_i, v_i=v_i, tmask=tmask, config=STRENGTH_CFG, fold=_fold(True))
    at = a_i[..., 0]
    vt = v_i[..., 0]
    for k in range(1, 5):
        at = at + a_i[..., k]
        vt = vt + v_i[..., k]
    raw = np.vectorize(lambda a, v: (2.0e4 * v) * math.exp(-20.0 * (1.0 - a)) if a > 1.0e-10 else 0.0)(at, vt)
    np.testing.assert_array_equal(np.asarray(out.at_i), at)
    np.testing.assert_array_equal(np.asarray(out.strength_raw), raw)
    smoothed = raw.copy()  # every interior cell, coast included (icedyn_rdgrft.f90:1100-1113)
    for i in range(NN_HLS, JPI - NN_HLS):
        for j in range(NN_HLS, JPJ - NN_HLS):
            if at[i, j] > 1.0e-10:
                num = 4.0 * raw[i, j] + (
                    (raw[i - 1, j] * tmask[i - 1, j] + raw[i + 1, j] * tmask[i + 1, j])
                    + (raw[i, j - 1] * tmask[i, j - 1] + raw[i, j + 1] * tmask[i, j + 1])
                )
                den = 4.0 + tmask[i - 1, j] + tmask[i + 1, j] + tmask[i, j - 1] + tmask[i, j + 1]
                smoothed[i, j] = num / den
            else:
                smoothed[i, j] = 0.0
    np.testing.assert_array_equal(np.asarray(out.strength), np.asarray(_lbc(smoothed, "T")))
    assert float(out.strength[2, 3]) == 0.0  # at_i <= epsi10
    assert np.any(tmask[NN_HLS:-NN_HLS, NN_HLS:-NN_HLS] == 0.0)


def test_strength_inexact_value_pins_and_plants():
    a_i, v_i, tmask = _categories(3)

    def run(**kw):
        return si3_orca2_ice_strength(a_i=a_i, v_i=v_i, tmask=tmask, config=STRENGTH_CFG, fold=_fold(False), **kw)

    out = run()
    pinned = _sha(out.at_i, out.strength_raw, out.strength)
    assert pinned == STRENGTH_PIN, pinned
    for plant in (
        dict(_plant_unmasked_smooth=True),
        dict(_plant_exchange_before_smooth=True),
        dict(_plant_fold_sign=frozenset({"strength"})),
    ):
        planted = run(**plant)
        assert not np.array_equal(np.asarray(planted.strength), np.asarray(out.strength)), plant


def test_strength_jit_equals_eager_and_fd_gradient():
    a_i, v_i, tmask = _categories(3)

    def total(v):
        out = si3_orca2_ice_strength(a_i=a_i, v_i=v, tmask=tmask, config=STRENGTH_CFG, fold=_fold(True))
        return jnp.sum(out.strength[NN_HLS:-NN_HLS, NN_HLS:-NN_HLS])

    eager = si3_orca2_ice_strength(a_i=a_i, v_i=v_i, tmask=tmask, config=STRENGTH_CFG, fold=_fold(True))
    jitted = jax.jit(
        lambda a, v, t: si3_orca2_ice_strength(a_i=a, v_i=v, tmask=t, config=STRENGTH_CFG, fold=_fold(True))
    )(a_i, v_i, tmask)
    for e, j in zip(eager, jitted):
        np.testing.assert_array_equal(np.asarray(e), np.asarray(j))
    grad = np.asarray(jax.grad(total)(jnp.asarray(v_i)))
    assert np.all(np.isfinite(grad))
    k = (9, 6, 2)
    eps = 1.0e-6 * v_i[k]
    plus, minus = v_i.copy(), v_i.copy()
    plus[k] += eps
    minus[k] -= eps
    fd = (float(total(jnp.asarray(plus))) - float(total(jnp.asarray(minus)))) / (2.0 * eps)
    assert grad[k] != 0.0
    assert abs(fd / grad[k] - 1.0) < 1.0e-6


# --- (b) post-strength field -> aEVP -------------------------------------------


def test_aevp_inexact_value_pins():
    out = _run_aevp()
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in out)
    pinned = _sha(*out)
    assert pinned == AEVP_PIN, pinned


@pytest.mark.parametrize("label", LABELS)
def test_aevp_wrong_fold_sign_plant_fires(label):
    base = _run_aevp()
    planted = _run_aevp(_plant_fold_sign=frozenset({label}))
    assert any(not np.array_equal(np.asarray(b), np.asarray(p)) for b, p in zip(base, planted)), label


def test_aevp_parity_plant_fires():
    base = _run_aevp()
    assert _sha(*_run_aevp(_plant_swap_parity=True)) != _sha(*base)


def test_aevp_trace_is_the_shared_solver_and_exposes_every_inner_frame():
    state, forcing, metrics, strength, entry = _aevp_inputs()
    config = _aevp_config(TEST_SUBCYCLES)
    common = dict(
        strength_t=strength,
        ln_ice_embd=False,
        fold=_fold(False),
        differentiable=False,
        **entry,
    )
    trace = si3_orca2_aevp_trace(state, forcing, metrics, config, **common)
    solved = si3_orca2_aevp_solver(state, forcing, metrics, config, **common)
    for actual, expected in zip(trace.final, solved):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    for frames in trace[2:]:
        assert frames.shape == (TEST_SUBCYCLES, JPI, JPJ)


def test_aevp_exchange_program_site_by_site(monkeypatch):
    """Nature, fold sign and ldfull at every lbc_lnk site of icedyn_rhg_evp.f90."""
    import legoesm.ice.si3_orca2_state as o1

    calls = []
    real = o1.si3_lbc_lnk_orca2

    def recording(field, cd_nat, psgn, nn_hls, ldfull, **kw):
        calls.append((cd_nat, float(psgn), bool(ldfull)))
        return real(field, cd_nat, psgn, nn_hls=nn_hls, ldfull=ldfull, **kw)

    state, forcing, metrics, strength, entry = _aevp_inputs()
    monkeypatch.setattr(o1, "si3_lbc_lnk_orca2", recording)
    si3_orca2_aevp_solver(
        state, forcing, metrics, _aevp_config(TEST_SUBCYCLES), strength_t=strength,
        ln_ice_embd=False, fold=_fold(False), differentiable=False, **entry,
    )
    expected = (
        [("F", 1.0, True)]  # :238 fimask, kt == nit000
        + [("T", 1.0, False), ("T", 1.0, False), ("U", -1.0, True), ("V", -1.0, True)]  # :440, :647/:754 (traced once)
        + [("T", 1.0, False), ("T", 1.0, False), ("F", 1.0, False)]  # :845
    )
    assert calls == expected



def test_partial_vs_full_pivot_line_fold():
    """Fold-symmetric inputs give mirror-identical pivot rows (NEMO's "NP repro"
    parentheses), so the full/partial fold only binds when the pivot row is not
    symmetric; break it in one right-half strength cell and the two programs differ."""
    state, forcing, metrics, strength, entry = _aevp_inputs()
    config = _aevp_config(TEST_SUBCYCLES)

    def solve(strength_t, full):
        return si3_orca2_aevp_solver(
            state, forcing, metrics, config, strength_t=strength_t, ln_ice_embd=False,
            fold=_fold(full), differentiable=False, **entry,
        )

    assert _sha(*solve(strength, True)) == _sha(*solve(strength, False))
    pivot = JPJ - NN_HLS - 1  # 1-based ipj - ihls
    broken = strength.at[JPI - NN_HLS - 2, pivot].multiply(1.5)
    assert _sha(*solve(broken, True)) != _sha(*solve(broken, False))


def test_aevp_jit_equals_eager_and_fd_gradient():
    """Eager = jax.disable_jit (one primitive at a time, nothing fused); nonzero
    Coriolis, so an unbarriered Coriolis statement fused into FMAs under jit fails."""
    state, forcing, metrics, strength, entry = _aevp_inputs()
    config = _aevp_config(TEST_SUBCYCLES)
    assert bool(jnp.all(forcing.coriolis_t[:, NN_HLS:] != 0.0))

    def solve(strength_t, *, differentiable, coriolis_t=forcing.coriolis_t):
        return si3_orca2_aevp_solver(
            state, forcing._replace(coriolis_t=coriolis_t), metrics, config, strength_t=strength_t,
            ln_ice_embd=False, fold=_fold(False), differentiable=differentiable, **entry,
        )

    with jax.disable_jit():
        eager = solve(strength, differentiable=False)
    jitted = jax.jit(lambda s: solve(s, differentiable=False))(strength)
    for e, j in zip(eager, jitted):
        np.testing.assert_array_equal(np.asarray(e), np.asarray(j))
    no_coriolis = solve(strength, differentiable=False, coriolis_t=jnp.zeros_like(forcing.coriolis_t))
    assert _sha(*no_coriolis) != _sha(*jitted)  # Coriolis binds in this fixture

    def loss(scale):
        out = solve(strength * scale, differentiable=True)
        return jnp.sum(out.u_ice_u[NN_HLS:-NN_HLS, NN_HLS:-NN_HLS] ** 2)

    grad = float(jax.grad(loss)(1.0))
    assert np.isfinite(grad) and grad != 0.0
    eps = 1.0e-6
    fd = (float(loss(1.0 + eps)) - float(loss(1.0 - eps))) / (2.0 * eps)
    assert abs(fd / grad - 1.0) < 1.0e-5


def test_physical_inputs_have_no_defaults():
    state, forcing, metrics, strength, entry = _aevp_inputs()
    config = _aevp_config(TEST_SUBCYCLES)
    common = dict(strength_t=strength, ln_ice_embd=False, fold=_fold(False), differentiable=False, **entry)
    with pytest.raises(TypeError):
        si3_orca2_aevp_solver(state, forcing, metrics, config, **{k: v for k, v in common.items() if k != "ssh_m"})
    with pytest.raises(ValueError, match="no default"):
        si3_orca2_aevp_solver(state, forcing, metrics, config, **{**common, "snwice_mass": None})
    with pytest.raises(ValueError, match="no default"):
        si3_orca2_aevp_solver(state, forcing, metrics, config, **{**common, "strength_t": None})
    with pytest.raises(ValueError, match="ice_var_sshdyn"):
        si3_orca2_aevp_solver(state, forcing._replace(ssh_t=entry["ssh_m"]), metrics, config, **common)
    with pytest.raises(ValueError, match="levitating"):
        si3_orca2_aevp_solver(state, forcing, metrics, config, **{**common, "ln_ice_embd": True})
    with pytest.raises(ValueError, match="Frankenstein"):
        si3_orca2_aevp_solver(state, forcing, metrics, config._replace(category_count=1), **common)
    with pytest.raises(TypeError):
        si3_orca2_aevp_solver(state, forcing, metrics, config, **{**common, "fold": (2, False)})
    a_i, v_i, tmask = _categories(3)
    with pytest.raises(ValueError, match="no default"):
        si3_orca2_ice_strength(a_i=a_i, v_i=None, tmask=tmask, config=STRENGTH_CFG, fold=_fold(True))
    with pytest.raises(ValueError, match=r"\(jpi, jpj, 5\)"):
        si3_orca2_ice_strength(a_i=a_i[..., :1], v_i=v_i[..., :1], tmask=tmask, config=STRENGTH_CFG, fold=_fold(True))
    for name in ("u_ice_b", "v_ice_b"):
        with pytest.raises(TypeError):
            si3_orca2_aevp_solver(state, forcing, metrics, config, **{k: v for k, v in common.items() if k != name})
        with pytest.raises(ValueError, match="no default"):
            si3_orca2_aevp_solver(state, forcing, metrics, config, **{**common, name: None})
    masses = {k: entry[k] for k in ("ssh_m", "snwice_mass", "snwice_mass_b")}
    assert si3_var_sshdyn(ln_ice_embd=False, **masses) is not None


def test_aevp_reads_recorded_previous_velocity_not_entry_state():
    """u_ice_b/v_ice_b (icedyn_rhg_evp.f90:574,625) are inputs of their own:
    substituting the entry u_ice/v_ice for them changes both velocities."""
    state, forcing, metrics, strength, entry = _aevp_inputs()
    assert not np.array_equal(np.asarray(entry["u_ice_b"]), np.asarray(state.u_ice_u))
    assert not np.array_equal(np.asarray(entry["v_ice_b"]), np.asarray(state.v_ice_v))
    common = dict(strength_t=strength, ln_ice_embd=False, fold=_fold(False), differentiable=False)
    config = _aevp_config(TEST_SUBCYCLES)
    recorded = si3_orca2_aevp_solver(state, forcing, metrics, config, **common, **entry)
    substituted = si3_orca2_aevp_solver(
        state, forcing, metrics, config, **common, **{**entry, "u_ice_b": state.u_ice_u, "v_ice_b": state.v_ice_v}
    )
    for leaf in ("u_ice_u", "v_ice_v"):
        assert not np.array_equal(np.asarray(getattr(recorded, leaf)), np.asarray(getattr(substituted, leaf))), leaf


# --- lane-3 cards: parent vs current through the production dispatchers --------

_RHG_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d_rhg/final")
_RHEO_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final")


def _load_parent(names_paths):
    rev = O5_PARENT_COMMIT
    saved = {name: sys.modules.get(name) for name, _ in names_paths}
    loaded = {}
    try:
        for name, rel in names_paths:
            src = subprocess.check_output(["git", "show", f"{rev}:{rel}"], cwd=REPO_ROOT, text=True)
            mod = types.ModuleType(name)
            mod.__file__ = f"<git:{rev}:{rel}>"
            sys.modules[name] = mod
            exec(compile(src, mod.__file__, "exec"), mod.__dict__)
            loaded[name] = mod
    finally:
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig
    return loaded


def _gate_module():
    import importlib.util

    path = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_oracle_gate.py"
    spec = importlib.util.spec_from_file_location("o5_oracle_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_trees_identical(a, b, label):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=label)


def _parent_lane3():
    return _load_parent(
        (
            ("legoesm.ice.dynamics", "packages/ice/legoesm/ice/dynamics.py"),
            ("legoesm.ice.fidelity.nemo_adv2d_rhg_testcase_recipe",
             "packages/ice/legoesm/ice/fidelity/nemo_adv2d_rhg_testcase_recipe.py"),
            ("legoesm.ice.fidelity.nemo_rheo_testcase_recipe",
             "packages/ice/legoesm/ice/fidelity/nemo_rheo_testcase_recipe.py"),
        )
    )


@pytest.mark.skipif(not _RHG_ROOT.exists(), reason="lane-3 ICE_ADV2D_RHG record absent")
def test_lane3_rhg_card_parent_vs_current_bit_identical():
    import netCDF4
    from legoesm.ice.fidelity import nemo_adv2d_rhg_testcase_recipe as rhg_cur

    parent = _parent_lane3()
    gate = _gate_module()

    with netCDF4.Dataset(_RHG_ROOT / "output.init_ice.nc") as ds:
        sst = np.asarray(ds["sst"][0]).T
    _, entry = gate.read_frame(_RHG_ROOT / "oracle_ice_step_entry_kt00000001.bin")
    rhg_par = parent["legoesm.ice.fidelity.nemo_adv2d_rhg_testcase_recipe"]
    card_c, card_p = rhg_cur.build_ice_adv2d_rhg_card(sst, entry), rhg_par.build_ice_adv2d_rhg_card(sst, entry)
    sc, sp = card_c.initial_state, card_p.initial_state
    for step in range(2):
        sc = rhg_cur.step_ice_adv2d_rhg_card(card_c, sc, completed_steps=step)
        sp = rhg_par.step_ice_adv2d_rhg_card(card_p, sp, completed_steps=step)
        _assert_trees_identical(sc, sp, f"ICE_ADV2D_RHG step {step + 1}")
    planted = rhg_par.step_ice_adv2d_rhg_card(
        card_p, card_p.initial_state, completed_steps=0, stress_divergence_outer_weight=0.5001
    )
    first = rhg_cur.step_ice_adv2d_rhg_card(card_c, card_c.initial_state, completed_steps=0)
    assert not np.array_equal(np.asarray(planted.dynamics.u_ice_u), np.asarray(first.dynamics.u_ice_u))


@pytest.mark.slow
@pytest.mark.skipif(not _RHEO_ROOT.exists(), reason="lane-3 ICE_RHEO record absent")
def test_lane3_rheo_card_parent_vs_current_bit_identical():
    """One ICE_RHEO step (rhg -> adv -> rdgrft -> cor): ~4 min per arm eager on 1004x1004."""
    import netCDF4
    from legoesm.ice.fidelity import nemo_rheo_testcase_recipe as rheo_cur

    parent = _parent_lane3()
    gate = _gate_module()
    _, frame = gate.read_frame(_RHEO_ROOT / "oracle_ice_step_entry_kt00000001.bin")
    names = ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f", "tmask", "umask", "vmask")
    with netCDF4.Dataset(_RHEO_ROOT / "mesh_mask.nc") as ds:
        mesh = {name: np.asarray(ds[name][:]) for name in names}
    with netCDF4.Dataset(_RHEO_ROOT / "output.init_ice.nc") as ds:
        sst_k = np.asarray(ds["sst"][0]) + constants.T_freeze
    rheo_par = parent["legoesm.ice.fidelity.nemo_rheo_testcase_recipe"]
    rc, rp = rheo_cur.build_ice_rheo_card(frame, mesh, sst_k), rheo_par.build_ice_rheo_card(frame, mesh, sst_k)
    sc = rheo_cur.step_ice_rheo_card(rc, rc.initial_state, completed_steps=0)
    sp = rheo_par.step_ice_rheo_card(rp, rp.initial_state, completed_steps=0)
    _assert_trees_identical(sc, sp, "ICE_RHEO step 1")


STRENGTH_PIN = "097712318bf93aaa1444855ebddff76db8ff21b3c59cd9672dddff9830391223"
AEVP_PIN = "cf17eb690a4e2345eb1a3b4d6a650c9ccd951f5684744c5de776ef8d78f1c147"
