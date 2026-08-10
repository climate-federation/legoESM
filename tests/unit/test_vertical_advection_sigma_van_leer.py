"""Monotone (van-Leer TVD) SIGMA vertical advection — the UTLS warm-bias option.

``vertical_advection(..., scheme="van_leer")`` is the opt-in replacement for the
first-order upwind gradient in the sigma lane.  Upwind's leading truncation
error is a diffusion ``K_σ = |σ̇|·Δσ/2``; on the cldF_fsd AMIP run that term was
measured at **+0.822 K/day** at the tropical UTLS maximum (15S-15N, 91.4 hPa,
N=37 checkpoints) — larger than the entire production temperature tendency
there and 2.1x the radiative cooling.

These tests pin, in order: the DEFAULT path is bit-identical, the dispatch
raises, the analytic properties (consistency, order, the diffusion identity,
monotonicity, boundary treatment), jit parity and differentiability, and the
config/factory wiring.

Which tests detect a silent ``van_leer -> upwind`` fallback, MEASURED by
mutating ``_vertical_advection_van_leer_sigma`` to return the upwind result
(codex rounds 5-6 — the earlier list here was wrong).  Exactly these 7 fail:

  test_theta_default_is_bit_identical
  test_convergence_order_upwind_is_first_and_van_leer_is_second
  test_implicit_diffusion_is_removed_at_utls_config
  test_tropopause_slope_break_is_advected_without_overshoot
  test_monotone_at_the_measured_production_courant_and_not_above_the_bound
  test_mpas_tendencies_default_unchanged_and_van_leer_moves_all_three_paths
  test_convergence_at_a_fixed_physical_pressure_and_on_a_refined_grid

The linear-profile and stretched-grid tests do NOT (upwind's divided difference
is also exact on a linear field), and neither do the reconstruction tests
(``van_leer_face_values_sigma`` is called directly, bypassing the dispatch) —
each says so in its own docstring.

SIGN CONVENTION (stated at the term): σ increases DOWNWARD, index 0 = model
top; σ̇ > 0 = DESCENT.  Tendencies are sources in df/dt.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.vertical import (
    VERTICAL_ADVECTION_SCHEMES,
    create_sigma_coordinate,
    vertical_advection,
    vertical_advection_theta,
)

jax.config.update("jax_enable_x64", True)

# Tropical UTLS configuration of the measured bias (cldF_fsd, 2.5 deg MPAS):
# 30 sigma levels, sigma_top = 0.01, dt = 75 s.
NLEV = 30
SIGMA_TOP = 0.01


def _coord(nlev=NLEV, refine=1.0):
    return create_sigma_coordinate(nlev, sigma_top=SIGMA_TOP,
                                   dtype=jnp.float64, tropopause_refine=refine)


def _theta_smooth(sigma):
    """Smooth, strictly MONOTONE θ(σ) — a stably stratified column.

    Monotone on purpose: θ has no extremum in a stable atmosphere (the tropical
    cold point is an extremum of T, not of θ), so this is the regime the
    limiter must NOT clip.
    """
    return 300.0 + 200.0 * jnp.exp(-3.0 * sigma) + 10.0 * sigma ** 2


def _dtheta_dsigma(sigma):
    return -600.0 * jnp.exp(-3.0 * sigma) + 20.0 * sigma


def _sigma_dot(sigma_half, amp=1.0e-5):
    """σ̇(σ) vanishing at both interfaces — the dycore's rigid lid + surface."""
    return amp * jnp.sin(
        jnp.pi * (sigma_half - SIGMA_TOP) / (1.0 - SIGMA_TOP))


def _legacy_upwind(field, sigma_dot, coord):
    """The pre-change operator, transcribed from git history (the byte-identity
    reference — an independent expression, not a call into the new code)."""
    sdf = 0.5 * (sigma_dot[..., :-1] + sigma_dot[..., 1:])
    df_over = jnp.diff(field, axis=-1) / coord.dsigma_full
    pad = ((0, 0),) * (df_over.ndim - 1)
    grad_bwd = jnp.pad(df_over, (*pad, (1, 0)))
    grad_fwd = jnp.pad(df_over, (*pad, (0, 1)))
    return -sdf * jnp.where(sdf > 0, grad_bwd, grad_fwd)


# ---------------------------------------------------------------- default path


def test_default_is_bit_identical_to_legacy_upwind():
    """Default (and explicit "upwind") == the pre-change operator, EXACTLY."""
    coord = _coord()
    rng = np.random.default_rng(20260806)
    f = jnp.asarray(rng.normal(size=(11, NLEV)) * 20.0 + 250.0)
    sd = jnp.asarray(rng.normal(size=(11, NLEV + 1)) * 1e-5)
    sd = sd.at[..., 0].set(0.0).at[..., -1].set(0.0)
    ref = _legacy_upwind(f, sd, coord)
    for got in (vertical_advection(f, sd, coord),
                vertical_advection(f, sd, coord, scheme="upwind")):
        assert jnp.array_equal(got, ref), "default sigma path perturbed"


def test_theta_default_is_bit_identical():
    coord = _coord()
    rng = np.random.default_rng(7)
    T = jnp.asarray(200.0 + rng.normal(size=(5, NLEV)) * 10.0)
    p_s = jnp.asarray(1.0e5 + rng.normal(size=(5,)) * 500.0)
    sd = jnp.asarray(rng.normal(size=(5, NLEV + 1)) * 1e-5)
    sd = sd.at[..., 0].set(0.0).at[..., -1].set(0.0)
    a = vertical_advection_theta(T, sd, p_s, coord)
    b = vertical_advection_theta(T, sd, p_s, coord, scheme="upwind")
    assert jnp.array_equal(a, b)
    c = vertical_advection_theta(T, sd, p_s, coord, scheme="van_leer")
    assert not jnp.allclose(a, c), "van_leer arm is a no-op on the theta path"


# -------------------------------------------------------------------- dispatch


@pytest.mark.parametrize("bad", ["upwid", "vanleer", "centered", "", "VAN_LEER"])
def test_unknown_scheme_raises(bad):
    coord = _coord()
    f = jnp.zeros((2, NLEV))
    sd = jnp.zeros((2, NLEV + 1))
    with pytest.raises(ValueError, match="unknown vertical advection scheme"):
        vertical_advection(f, sd, coord, scheme=bad)
    with pytest.raises(ValueError, match="unknown vertical advection scheme"):
        vertical_advection_theta(f + 250.0, sd, jnp.full((2,), 1e5), coord,
                                 scheme=bad)


@pytest.mark.parametrize("nlev", [1, 2, 3])
def test_van_leer_raises_below_the_stencil_width(nlev):
    """A selected scheme must never silently become another one (codex P2):
    below 4 levels the 4-cell stencil is undefined, so it raises rather than
    returning the upwind result."""
    coord = create_sigma_coordinate(nlev, sigma_top=SIGMA_TOP, dtype=jnp.float64)
    f = jnp.arange(nlev, dtype=jnp.float64)[None, :]
    sd = jnp.ones((1, nlev + 1))
    vertical_advection(f, sd, coord)  # upwind still works
    with pytest.raises(ValueError, match="at least 4 vertical levels"):
        vertical_advection(f, sd, coord, scheme="van_leer")


def test_scheme_tuple_is_the_dispatch_domain():
    assert VERTICAL_ADVECTION_SCHEMES == ("upwind", "van_leer")


# ------------------------------------------------------------------ consistency


@pytest.mark.parametrize("scheme", VERTICAL_ADVECTION_SCHEMES)
def test_constant_field_zero_tendency(scheme):
    coord = _coord()
    sd = _sigma_dot(coord.sigma_half)[None, :] * jnp.ones((3, NLEV + 1))
    t = vertical_advection(jnp.full((3, NLEV), 271.35), sd, coord, scheme=scheme)
    assert float(jnp.max(jnp.abs(t))) == 0.0


def test_linear_field_is_exact_for_van_leer_in_the_interior():
    """A linear profile is reproduced EXACTLY by a 2nd-order reconstruction.

    WHAT THIS PINS: the reconstruction itself.  It caught the first
    implementation's ``mode="edge"`` ghost pad, which zeroed the upwind slope
    at face 1, collapsed the limiter to donor cell there and produced a 50%
    tendency error at level 1.  It does NOT detect a wholesale fallback to
    upwind (upwind's divided difference is also exact on a linear field) —
    the order / diffusion / slope-break tests do that, verified by mutation.

    The two boundary levels are excluded: BOTH schemes use the first-order
    zero-gradient BC there by design, and upwind's error at k=0 is 1.3e-06.
    """
    coord = _coord()
    slope = 2.5
    f = (slope * coord.sigma_full + 7.0)[None, :]
    sd = _sigma_dot(coord.sigma_half)[None, :]
    sd_full = 0.5 * (sd[..., :-1] + sd[..., 1:])
    exact = -sd_full * slope
    vl = vertical_advection(f, sd, coord, scheme="van_leer")
    up = vertical_advection(f, sd, coord, scheme="upwind")
    assert float(jnp.max(jnp.abs((vl - exact)[0, 1:-1]))) < 1e-15
    # BOUNDARY CONTRACT: the two end levels are the first-order tendency.
    assert jnp.array_equal(vl[..., 0], up[..., 0])
    assert jnp.array_equal(vl[..., -1], up[..., -1])
    # ... and upwind is NOT exact at k=0 (the zero-gradient BC), which is why
    # k=0 is excluded from every interior claim.
    assert float(jnp.abs((up - exact)[0, 0])) > 1e-8


# ------------------------------------------------------------------------ order


def _order_slope(scheme, levels=(30, 60, 120, 240, 480)):
    """Convergence slope of the L-inf interior error on a manufactured
    solution with a known analytic tendency."""
    errs = []
    for n in levels:
        coord = _coord(n)
        f = _theta_smooth(coord.sigma_full)[None, :]
        sd = _sigma_dot(coord.sigma_half)[None, :]
        sd_full = 0.5 * (sd[..., :-1] + sd[..., 1:])
        exact = -sd_full * _dtheta_dsigma(coord.sigma_full)[None, :]
        got = vertical_advection(f, sd, coord, scheme=scheme)
        # Exclude the levels whose stencil touches a boundary: the BC is a
        # separate (zero-gradient) statement, scored by its own test below.
        # The excluded fraction shrinks with n, so it is not an order cheat.
        err = float(jnp.max(jnp.abs((got - exact)[0, 3:-3])))
        errs.append((1.0 / n, err))
    h = np.log(np.array([e[0] for e in errs]))
    e = np.log(np.array([e[1] for e in errs]))
    return float(np.polyfit(h, e, 1)[0]), errs


def test_convergence_order_upwind_is_first_and_van_leer_is_second():
    s_up, e_up = _order_slope("upwind")
    s_vl, e_vl = _order_slope("van_leer")
    assert 0.85 <= s_up <= 1.15, f"upwind slope {s_up:.3f} (expected ~1)"
    assert 1.85 <= s_vl <= 2.15, f"van_leer slope {s_vl:.3f} (expected ~2)"
    # And it is strictly more accurate at every resolution tested.
    for (_, a), (_, b) in zip(e_up, e_vl):
        assert b < a


# ----------------------------------------------------- the diffusion identity


def test_implicit_diffusion_is_removed_at_utls_config():
    """``scheme - centred`` is upwind's ``K_σ = |σ̇|Δσ/2`` diffusion.

    Measured on the 30-level tropical UTLS configuration against the analytic
    identity, then again for van-Leer, which must be an order of magnitude
    smaller.  Comparable to the +0.822 K/day measured on the run.
    """
    coord = _coord()
    f = _theta_smooth(coord.sigma_full)[None, :]
    sd = _sigma_dot(coord.sigma_half)[None, :]
    sd_full = 0.5 * (sd[..., :-1] + sd[..., 1:])
    # centred reference stencil (identical in every other respect)
    df_over = jnp.diff(f, axis=-1) / coord.dsigma_full
    pad = ((0, 0),) * (df_over.ndim - 1)
    g_b = jnp.pad(df_over, (*pad, (1, 0)))
    g_f = jnp.pad(df_over, (*pad, (0, 1)))
    ctr = -sd_full * 0.5 * (g_b + g_f)

    up = vertical_advection(f, sd, coord, scheme="upwind")
    vl = vertical_advection(f, sd, coord, scheme="van_leer")
    # Score the INTERIOR (levels 2..nlev-3): k=0/k=nlev-1 are the first-order
    # boundary contract for both schemes, and k=1/k=nlev-2 read a ghost-based
    # reconstruction whose O(Δσ²) constant is larger than an interior one.
    sl = slice(2, -2)
    num_up = np.asarray(up - ctr)[0, sl]
    num_vl = np.asarray(vl - ctr)[0, sl]

    # The published identity: upwind - centred == |σ̇|·(Δσ/2)·∂²f/∂σ².
    d2 = np.asarray((g_f - g_b) / coord.dsigma_full.mean())[0, sl]
    k_sigma = np.asarray(jnp.abs(sd_full))[0, sl] * float(
        coord.dsigma_full.mean()) / 2.0
    assert np.allclose(num_up, k_sigma * d2, rtol=2e-2), "K_sigma identity broke"

    ratio = np.max(np.abs(num_vl)) / np.max(np.abs(num_up))
    assert ratio < 0.10, (
        f"van_leer residual diffusion {np.max(np.abs(num_vl)):.3e} is not "
        f"<< upwind {np.max(np.abs(num_up)):.3e} (ratio {ratio:.3f})")


# ----------------------------------------------------------------- monotonicity


def test_tropopause_slope_break_is_advected_without_overshoot():
    """Advect a tropopause-like KINK IN THE GRADIENT (not an extremum).

    θ is monotone in a stable column, so the tropical cold point is a sharp
    change of ``∂θ/∂σ``, NOT a θ extremum — this is the feature the scheme has
    to carry.  Uniform σ̇ makes the exact solution a pure translate, so the
    error against it is a clean skill measure.  Scored on the interior levels
    the boundary fallback cannot reach in the time advected.
    """
    coord = _coord()
    sig = np.asarray(coord.sigma_full)
    brk = 0.10  # tropical cold point

    def prof(x):
        # monotone decreasing in sigma, slope jumps by 15x at the break
        return np.where(x < brk, 350.0 + 1500.0 * (brk - x),
                        350.0 - 100.0 * (x - brk))

    f0 = jnp.asarray(prof(sig)[None, :])
    c = 2.0e-5
    sd = jnp.full((1, NLEV + 1), c)
    dt = 75.0
    nstep = int(round(1.5 * float(coord.dsigma[0]) / (dt * c)))
    assert 10 <= nstep <= 200, f"step count {nstep} out of the intended range"
    exact = jnp.asarray(prof(sig - c * dt * nstep)[None, :])
    lo0, hi0 = float(f0.min()), float(f0.max())

    out = {}
    for scheme in VERTICAL_ADVECTION_SCHEMES:
        f = f0
        over = under = 0.0
        for _ in range(nstep):
            f = f + dt * vertical_advection(f, sd, coord, scheme=scheme)
            over = max(over, float(f.max()) - hi0)
            under = max(under, lo0 - float(f.min()))
        out[scheme] = (over, under,
                       float(jnp.max(jnp.abs((f - exact)[0, 3:-3]))))
    over, under, err_vl = out["van_leer"]
    assert over < 1e-9 and under < 1e-9, (
        f"van_leer overshoot {over:.3e} K / undershoot {under:.3e} K")
    err_up = out["upwind"][2]
    assert err_vl < 0.5 * err_up, (
        f"van_leer did not carry the slope break better: L-inf error "
        f"{err_vl:.3f} K vs upwind {err_up:.3f} K over {nstep} steps")


def test_uniform_grid_matches_the_shared_van_leer_face_values():
    """On a uniform grid the kernel's reconstruction must EQUAL the shared
    ``van_leer_face_values``, ELEMENT BY ELEMENT for both returned arrays.

    Codex round 3: the earlier version seeded the reference face 1 and
    reconstructed later faces by inverting the tendency, which only pins
    successive DIFFERENCES — a constant offset on every interior face would
    have passed.  This compares the arrays directly.  The HD-1 smoothness-ratio
    sign convention is what this pins to the one shared implementation.
    """
    from legoesm.core.flux_limiters import van_leer_face_values
    from legoesm.grids.vertical import van_leer_face_values_sigma
    coord = _coord()
    rng = np.random.default_rng(99)
    for scale in (15.0, 1e-8, 1e4):
        f = jnp.asarray(220.0 + rng.normal(size=(4, NLEV)) * scale)
        got_pos, got_neg = van_leer_face_values_sigma(f, coord)
        f0, f1 = f[..., 0:1], f[..., 1:2]
        fm1, fm2 = f[..., -1:], f[..., -2:-1]
        fp = jnp.concatenate([3 * f0 - 2 * f1, 2 * f0 - f1, f,
                              2 * fm1 - fm2, 3 * fm1 - 2 * fm2], axis=-1)
        n = NLEV
        ref_pos, ref_neg = van_leer_face_values(
            fp[..., 0:n + 1], fp[..., 1:n + 2],
            fp[..., 2:n + 3], fp[..., 3:n + 4])
        for name, got, ref in (("q_pos", got_pos, ref_pos),
                               ("q_neg", got_neg, ref_neg)):
            d = float(jnp.max(jnp.abs(got - ref)))
            tol = 1e-12 * max(scale, 1.0)
            assert d <= tol, (
                f"{name} differs from the shared helper by {d:.3e} "
                f"(scale {scale}, tol {tol:.3e})")


@pytest.mark.parametrize("refine", [1.0, 3.0])
def test_kernel_face_values_never_leave_the_local_range(refine):
    """THIS kernel's own reconstruction is bounded by its two neighbours.

    Codex round 1 (P2): the earlier version probed
    ``flux_limiters.van_leer_face_values`` instead, so it could not see this
    kernel's stretched-grid weights or its clip.  ``refine=3`` is the grid on
    which the MUSCL weight exceeds 1/2 and the clip becomes load-bearing.
    """
    from legoesm.grids.vertical import van_leer_face_values_sigma
    coord = _coord(refine=refine)
    # (interior faces only: 0 and nlev are bounded against a GHOST cell and
    # never enter a tendency -- both boundary levels take the first-order
    # result; codex round 2.)
    rng = np.random.default_rng(int(refine * 10) + 5)
    for scale in (30.0, 1e-10, 1e5):
        f = jnp.asarray(200.0 + rng.normal(size=(4, NLEV)) * scale)
        q_pos, q_neg = van_leer_face_values_sigma(f, coord)
        # interfaces 1..nlev-1 sit between real cells j-1 and j
        lo = jnp.minimum(f[..., :-1], f[..., 1:])
        hi = jnp.maximum(f[..., :-1], f[..., 1:])
        for name, q in (("q_pos", q_pos), ("q_neg", q_neg)):
            qi = q[..., 1:NLEV]
            assert bool(jnp.all(qi >= lo)) and bool(jnp.all(qi <= hi)), (
                f"{name} left the local range (refine={refine}, scale={scale})")
    # And the clip is LOAD-BEARING on the stretched grid: recompute the RAW
    # (unclipped) MUSCL value and show it leaves the range there while the
    # returned value does not.  On the uniform grid the raw value is already
    # in range, i.e. the clip is provably non-binding.
    from legoesm.core.flux_limiters import grad_safe_ratio, van_leer_limiter
    f = jnp.asarray(200.0 + np.random.default_rng(77).normal(size=(6, NLEV)) * 30.0)
    q_pos, q_neg = van_leer_face_values_sigma(f, coord)
    f0, f1 = f[..., 0:1], f[..., 1:2]
    fm1, fm2 = f[..., -1:], f[..., -2:-1]
    fp = jnp.concatenate([3 * f0 - 2 * f1, 2 * f0 - f1, f,
                          2 * fm1 - fm2, 3 * fm1 - 2 * fm2], axis=-1)
    n = NLEV
    f_jm2, f_jm1 = fp[..., 0:n + 1], fp[..., 1:n + 2]
    f_j = fp[..., 2:n + 3]
    dcp = jnp.pad(coord.dsigma_full, (2, 2), mode="edge")
    dsp = jnp.pad(coord.dsigma, (1, 1), mode="edge")
    delta = f_j - f_jm1
    s_loc = delta / dcp[1:n + 2]
    ok = jnp.abs(s_loc) > 0.0
    r = grad_safe_ratio((f_jm1 - f_jm2) / dcp[0:n + 1],
                        jnp.where(jnp.abs(s_loc) > 1e-30, s_loc, 1e-30), ok)
    w = dsp[:-1] / (dsp[:-1] + dsp[1:])
    r_neg = grad_safe_ratio((fp[..., 3:n + 4] - f_j) / dcp[2:n + 3],
                            jnp.where(jnp.abs(s_loc) > 1e-30, s_loc, 1e-30), ok)
    w_pos = dsp[:-1] / (dsp[:-1] + dsp[1:])
    w_neg = dsp[1:] / (dsp[:-1] + dsp[1:])
    raw = {"q_pos": f_jm1 + w_pos * van_leer_limiter(r) * delta,
           "q_neg": f_j - w_neg * van_leer_limiter(r_neg) * delta}
    hi_j = jnp.maximum(f[..., :-1], f[..., 1:])
    lo_j = jnp.minimum(f[..., :-1], f[..., 1:])

    def out_of_range(a):
        return bool(jnp.any(a[..., 1:n] > hi_j) or jnp.any(a[..., 1:n] < lo_j))

    # BOTH returned arrays are in range, and BOTH raw ones show the clip is
    # load-bearing on the stretched grid / non-binding on the uniform one.
    for name, ret in (("q_pos", q_pos), ("q_neg", q_neg)):
        assert not out_of_range(ret), f"{name} left the range after the clip"
        if refine > 1.0:
            assert out_of_range(raw[name]), (
                f"clip should be load-bearing for {name} on the stretched grid")
        else:
            assert not out_of_range(raw[name]), (
                f"clip should be non-binding for {name} on the uniform grid")


def test_monotone_at_the_measured_production_courant_and_not_above_the_bound():
    """The monotonicity claim, with its CFL condition stated and BOTH sides
    measured (codex round 1, P1).

    The update is TVD while ~``nu_k + nu_{k+1} <= 1``; it is NOT above that.
    The measured statistic is that PAIR, in the RAW INTERFACE velocities the
    update multiplies (codex round 2 — the half->full average the first-order
    path uses cancels opposite-signed interfaces): global max 0.0642 over all
    cells, levels and the 37 year-1 checkpoint snapshots at dt = 75 s.  With a
    uniform sigma_dot the pair is 2x the per-face Courant, so the rollout below
    is driven at ``PAIR/2`` per face.  This test pins (a) no new extremum at 3x
    the measured PAIR maximum, and (b) that an overshoot DOES appear at
    per-face Courant 0.75 — so the limit is documented, not hidden.
    """
    MEASURED_PAIR_MAX = 0.0642
    coord = _coord()
    sig = np.asarray(coord.sigma_full)
    dsig = float(coord.dsigma[0])
    # A smooth Gaussian blob is the field that exposes the bound: its flanks
    # carry an uncompensated anti-diffusive face term while the limiter is
    # clipped at the peak.  A top-hat does NOT trigger it -- measured.
    fields = {"gauss": np.exp(-((sig - 0.5) / 0.05) ** 2),
              "tophat": ((sig > 0.4) & (sig < 0.6)).astype(float)}

    def rollout(q0, courant, nstep):
        sd = jnp.full((1, NLEV + 1), courant * dsig)  # |sd|*dt/dsigma = courant
        q, lo, hi = q0, 0.0, 1.0
        for _ in range(nstep):
            q = q + vertical_advection(q, sd, coord, scheme="van_leer")
            lo = min(lo, float(q.min()))
            hi = max(hi, float(q.max()))
        return lo, hi

    for name, arr in fields.items():
        q0 = jnp.asarray(arr[None, :])
        # 3x the measured PAIR maximum, expressed as a per-face Courant
        lo, hi = rollout(q0, 3.0 * MEASURED_PAIR_MAX / 2.0, 300)
        assert lo >= -1e-12 and hi <= 1.0 + 1e-12, (
            f"{name}: new extremum at 3x the measured production Courant: "
            f"[{lo}, {hi}]")
    # The other side of the claim: at Courant 0.75 it DOES overshoot, so the
    # limit is documented rather than hidden (codex round 1 counterexample,
    # reproduced here: one Euler step, [-0.0173, 0.9911] at nlev=30).
    lo_bad, hi_bad = rollout(jnp.asarray(fields["gauss"][None, :]), 0.75, 1)
    assert lo_bad < -1e-3, (
        "the documented CFL limit no longer bites - re-derive the bound "
        f"before widening the claim (got [{lo_bad}, {hi_bad}])")


def test_stretched_grid_stays_exact_on_linear_and_monotone():
    """A ``tropopause_refine`` grid (the sibling proposed fix) must not degrade
    the scheme — it is the combination a refined run would use.

    WHAT THIS PINS: the slope-ratio + Δσ-weighted face position.  With raw
    difference ratios and a 0.5 weight the error here was 9.2e-07, i.e. WORSE
    than upwind (exact, 1.3e-17) on the same profile.  Like the uniform-grid
    linear test it does not detect a wholesale fallback to upwind.
    """
    coord = _coord(refine=3.0)
    spread = float(coord.dsigma.max() / coord.dsigma.min()) - 1.0
    assert spread > 0.5, "refine=3 did not stretch the grid; test is vacuous"
    slope = -4.0
    f = (slope * coord.sigma_full + 300.0)[None, :]
    sd = _sigma_dot(coord.sigma_half)[None, :]
    sd_full = 0.5 * (sd[..., :-1] + sd[..., 1:])
    err_vl = float(jnp.max(jnp.abs(
        (vertical_advection(f, sd, coord, scheme="van_leer")
         - (-sd_full * slope))[0, 1:-1])))
    # Exact on a linear profile at NON-UNIFORM spacing: this is what the slope
    # ratios + Δσ-weighted face position buy.  Raw-difference ratios and a 0.5
    # weight gave 9.2e-07 here, i.e. WORSE than upwind (which is exact).
    assert err_vl < 1e-14, f"stretched-grid linear error {err_vl:.3e}"
    # Monotone: a rollout on a random column creates no new extremum.
    rng = np.random.default_rng(3)
    f0 = jnp.asarray(200.0 + rng.normal(size=(1, NLEV)) * 30.0)
    sdc = jnp.full((1, NLEV + 1), 1.0e-5)
    sdc = sdc.at[..., 0].set(0.0).at[..., -1].set(0.0)
    lo0, hi0 = float(f0.min()), float(f0.max())
    g = f0
    for _ in range(500):
        g = g + 10.0 * vertical_advection(g, sdc, coord, scheme="van_leer")
    assert float(g.max()) <= hi0 + 1e-9 and float(g.min()) >= lo0 - 1e-9


# --------------------------------------------------------------- boundary + BC


def test_boundary_levels_are_the_first_order_tendency_under_both_schemes():
    """The boundary CONTRACT: at k=0 and k=nlev-1 van-Leer returns exactly the
    upwind tendency, for any σ̇ including a perturbed boundary interface.

    NOT "σ̇ at a boundary interface does not enter the tendency" — it does,
    through the half->full average (the earlier name was wrong; codex round 5).
    What is pinned is equality between the two schemes at BOTH endpoints.
    """
    coord = _coord()
    rng = np.random.default_rng(11)
    f = jnp.asarray(200.0 + rng.normal(size=(4, NLEV)) * 5.0)
    sd = jnp.asarray(rng.normal(size=(4, NLEV + 1)) * 1e-5)
    sd0 = sd.at[..., 0].set(0.0).at[..., -1].set(0.0)
    sd1 = sd.at[..., 0].set(9.9e-4).at[..., -1].set(-9.9e-4)
    for scheme in VERTICAL_ADVECTION_SCHEMES:
        a = vertical_advection(f, sd0, coord, scheme=scheme)
        b = vertical_advection(f, sd1, coord, scheme=scheme)
        # Both schemes give the SAME boundary tendency (van_leer defers to the
        # first-order path there), and both DO see sigma_dot at the boundary
        # interface through the half->full average — so the contract pinned
        # here is equality between the schemes, not independence from sd.
        assert jnp.array_equal(a[..., 0],
                               vertical_advection(f, sd0, coord)[..., 0])
        assert jnp.array_equal(a[..., -1],
                               vertical_advection(f, sd0, coord)[..., -1])
        assert jnp.array_equal(b[..., 0],
                               vertical_advection(f, sd1, coord)[..., 0])
        assert jnp.array_equal(b[..., -1],
                               vertical_advection(f, sd1, coord)[..., -1])


# --------------------------------------------------------------- JAX discipline


def test_jit_parity_matches_eager():
    coord = _coord()
    f = _theta_smooth(coord.sigma_full)[None, :]
    sd = _sigma_dot(coord.sigma_half)[None, :]
    for scheme in VERTICAL_ADVECTION_SCHEMES:
        eager = vertical_advection(f, sd, coord, scheme=scheme)
        jitted = jax.jit(vertical_advection, static_argnums=(3,))(
            f, sd, coord, scheme)
        assert jnp.allclose(eager, jitted, rtol=0, atol=1e-13)


def test_gradients_are_finite_and_second_order_correct():
    coord = _coord(12)
    from jax._src.public_test_util import check_grads

    def loss(x, s):
        return jnp.sum(vertical_advection(x, s, coord, scheme="van_leer") ** 2)

    # (a) 2nd-order check_grads on an O(1)-SCALED problem with a single-signed
    # sigma_dot.  Scaling matters: check_grads' default step (1e-6) is a 10%
    # perturbation of a physical sigma_dot ~ 1e-5 and flips the donor-cell
    # select at the lid, so the finite-difference REFERENCE (not the AD) is
    # what breaks there.  One-signed sigma_dot removes the select kink.
    f1 = _theta_smooth(coord.sigma_full)[None, :] * jnp.ones((2, 1))
    sd1 = 0.5 + 0.25 * jnp.cos(
        jnp.pi * coord.sigma_half)[None, :] * jnp.ones((2, 1))
    check_grads(loss, (f1, sd1), order=2, modes=("rev", "fwd"))

    # (b) At the PRODUCTION scaling (sigma_dot ~ 1e-5, zero at the lid), AD is
    # checked against a hand-stepped central difference instead.
    f2 = _theta_smooth(coord.sigma_full)[None, :] * jnp.ones((2, 1))
    sd2 = _sigma_dot(coord.sigma_half)[None, :] * jnp.ones((2, 1))
    g_f, g_s = jax.grad(loss, argnums=(0, 1))(f2, sd2)
    for arr, gref, eps in ((f2, g_f, 1e-5), (sd2, g_s, 1e-9)):
        num = np.zeros(arr.shape)
        for i in range(arr.shape[0]):
            for k in range(arr.shape[1]):
                num[i, k] = (float(loss(*( (arr.at[i, k].add(eps), sd2)
                                          if arr is f2 else
                                          (f2, arr.at[i, k].add(eps)) )))
                             - float(loss(*( (arr.at[i, k].add(-eps), sd2)
                                            if arr is f2 else
                                            (f2, arr.at[i, k].add(-eps)) )))
                             ) / (2 * eps)
        err = float(np.max(np.abs(np.asarray(gref) - num)))
        scale = float(np.max(np.abs(num)))
        assert err < 1e-8 * max(scale, 1.0), f"AD vs FD {err:.3e} (scale {scale:.3e})"

    # (c) ADVERSARIAL data (random extrema -> phi kinks at r=0, and
    # near-uniform columns -> the grad_safe_ratio floor): no finite-difference
    # reference is valid at a kink, but the gradient must still be FINITE.
    rng = np.random.default_rng(5)
    for scale in (8.0, 0.0, 1e-14):
        fa = jnp.asarray(250.0 + rng.normal(size=(2, 12)) * scale)
        sa = jnp.asarray(rng.normal(size=(2, 13)) * 1e-5)
        sa = sa.at[..., 0].set(0.0).at[..., -1].set(0.0)
        g = jax.grad(loss, argnums=(0, 1))(fa, sa)
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in g), (
            f"non-finite gradient at scale {scale}")


def test_scan_compatible():
    """Usable as a lax.scan body (static scheme captured in the closure)."""
    coord = _coord()
    f0 = _theta_smooth(coord.sigma_full)[None, :]
    sd = _sigma_dot(coord.sigma_half)[None, :]

    def body(f, _):
        return f + 1.0 * vertical_advection(f, sd, coord, scheme="van_leer"), None

    out, _ = jax.lax.scan(body, f0, None, length=8)
    assert bool(jnp.all(jnp.isfinite(out)))


# ------------------------------------------------------------------ MPAS wiring


def test_mpas_config_default_and_dispatch():
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
    )
    assert MPASPrimitiveEquationConfig().vert_advection_scheme == "upwind"
    assert (MPASPrimitiveEquationConfig(vert_advection_scheme="van_leer")
            .vert_advection_scheme == "van_leer")


def test_mpas_tendencies_reject_unknown_and_hybrid():
    """Fail-early on the STATIC config value, before any tracing."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, mpas_hydrostatic_tendencies,
    )
    from legoesm.grids.vertical import standard_hybrid_levels

    coord = _coord()
    hyb = standard_hybrid_levels(NLEV)
    cfg_bad = MPASPrimitiveEquationConfig(vert_advection_scheme="nope")
    cfg_vl = MPASPrimitiveEquationConfig(vert_advection_scheme="van_leer")
    with pytest.raises(ValueError, match="unknown vert_advection_scheme"):
        mpas_hydrostatic_tendencies(None, None, coord, cfg_bad)
    with pytest.raises(ValueError, match="sigma vertical coordinate only"):
        mpas_hydrostatic_tendencies(None, None, hyb, cfg_vl)


def test_driver_config_and_validate_strict():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig

    assert DycoreConfig().mpas_vert_advection_scheme == "upwind"
    base = dict(discretization="mpas", model_type="hydrostatic")
    ok = ExperimentConfig(
        dycore=DycoreConfig(mpas_vert_advection_scheme="van_leer", **base),
        grid=ExperimentConfig().grid._replace(grid_type="voronoi",
                                              vertical_coord="sigma"))
    ok.validate_strict()  # must not raise

    with pytest.raises(ValueError, match="mpas_vert_advection_scheme must be"):
        ExperimentConfig(dycore=DycoreConfig(
            mpas_vert_advection_scheme="bogus", **base)).validate_strict()
    with pytest.raises(ValueError, match="sigma vertical coordinate only"):
        ExperimentConfig(
            dycore=DycoreConfig(mpas_vert_advection_scheme="van_leer", **base),
            grid=ExperimentConfig().grid._replace(
                grid_type="voronoi", vertical_coord="hybrid")).validate_strict()
    # nlev below the 4-cell stencil must be rejected AT CONFIG VALIDATION, not
    # deep inside the traced kernel (codex round 3).
    with pytest.raises(ValueError, match="at least 4 .*vertical levels"):
        ExperimentConfig(
            dycore=DycoreConfig(mpas_vert_advection_scheme="van_leer", **base),
            grid=ExperimentConfig().grid._replace(
                grid_type="voronoi", vertical_coord="sigma",
                nlev=3)).validate_strict()
    # ... and the DEFAULT scheme is unaffected by the level count.
    ExperimentConfig(
        dycore=DycoreConfig(**base),
        grid=ExperimentConfig().grid._replace(
            grid_type="voronoi", vertical_coord="sigma",
            nlev=3)).validate_strict()


def test_tracer_blob_stays_positive_and_bounded_under_varying_sigma_dot():
    """The SAME operator carries q_v/q_c/q_i, which DO have extrema.

    The advective form is not automatically TVD when σ̇ varies (the anti-
    diffusive face term is uncompensated at the cell edge), so this is checked
    on adversarial tracer data rather than argued: a Gaussian blob and a
    top-hat, driven by a sign-changing σ̇ whose per-face Courant is ~2.1x the
    measured production raw-face maximum of 0.0322 (the assertion below pins
    that ratio, so the stated coverage cannot drift — codex round 5).
    """
    coord = _coord()
    sig = np.asarray(coord.sigma_full)
    sh = np.asarray(coord.sigma_half)
    q0 = np.zeros((2, NLEV))
    q0[0] = np.exp(-((sig - 0.5) / 0.05) ** 2)
    q0[1] = ((sig > 0.4) & (sig < 0.6)).astype(float)
    q0 = jnp.asarray(q0)
    sd = jnp.asarray(
        (3.0e-5 * np.sin(2 * np.pi * (sh - SIGMA_TOP) / (1.0 - SIGMA_TOP)))[None, :]
        * np.ones((2, 1)))
    dt = 75.0
    cour = float(jnp.max(jnp.abs(0.5 * (sd[..., :-1] + sd[..., 1:])))) \
        * dt / float(coord.dsigma.min())
    # 0.0322 = measured global max RAW-FACE Courant on the target run.
    assert 2.0 <= cour / 0.0322 <= 2.5, (
        f"Courant {cour:.4f} is {cour/0.0322:.2f}x the production raw-face "
        f"maximum; the docstring claims ~2.1x")
    for scheme in VERTICAL_ADVECTION_SCHEMES:
        q = q0
        lo = hi = None
        for _ in range(400):
            q = q + dt * vertical_advection(q, sd, coord, scheme=scheme)
            lo = float(q.min()) if lo is None else min(lo, float(q.min()))
            hi = float(q.max()) if hi is None else max(hi, float(q.max()))
        assert lo >= -1e-12, f"{scheme}: tracer went negative ({lo:.3e})"
        assert hi <= 1.0 + 1e-12, f"{scheme}: tracer overshot 1.0 ({hi:.9f})"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_float32_and_float64_finite_including_uniform_columns(dtype):
    """float32 is the non-x64 default coordinate dtype; a near-uniform column
    is the regime where an ungated limiter-ratio derivative NaNs in f32 (the
    ``grad_safe_ratio`` failure mode)."""
    coord = create_sigma_coordinate(NLEV, sigma_top=SIGMA_TOP, dtype=dtype)
    f = jnp.asarray(_theta_smooth(coord.sigma_full)[None, :], dtype)
    sd = jnp.asarray(_sigma_dot(coord.sigma_half)[None, :], dtype)
    for scheme in VERTICAL_ADVECTION_SCHEMES:
        out = vertical_advection(f, sd, coord, scheme=scheme)
        assert out.dtype == dtype and bool(jnp.all(jnp.isfinite(out)))

        def loss(x):
            return jnp.sum(vertical_advection(x, sd, coord, scheme=scheme) ** 2)

        assert bool(jnp.all(jnp.isfinite(jax.grad(loss)(f))))
        # exactly-uniform column: the guarded-denominator regime
        assert bool(jnp.all(jnp.isfinite(
            jax.grad(loss)(jnp.full((3, NLEV), 250.0, dtype)))))


def test_vmap_matches_direct_and_jit_does_not_retrace():
    """The retrace check (a cache-size assertion) lives HERE, not in the
    jit-parity test above, whose name previously over-claimed it."""
    coord = _coord()
    f = _theta_smooth(coord.sigma_full)[None, :] * jnp.ones((5, 1))
    sd = _sigma_dot(coord.sigma_half)[None, :] * jnp.ones((5, 1))
    direct = vertical_advection(f, sd, coord, scheme="van_leer")
    mapped = jax.vmap(
        lambda a, b: vertical_advection(a[None], b[None], coord,
                                        scheme="van_leer")[0])(f, sd)
    assert bool(jnp.allclose(mapped, direct, rtol=0, atol=1e-14))
    fn = jax.jit(lambda a, b: vertical_advection(a, b, coord, scheme="van_leer"))
    for _ in range(6):
        fn(f, sd)
    assert fn._cache_size() == 1, "hot-loop retrace"


def _mpas_state_and_mesh(nlev=NLEV, level=2):
    """A small real MPAS mesh + non-trivial hydrostatic state WITH tracers."""
    from legoesm.core.field import Field
    from legoesm.core.state import MPASHydrostaticState
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(level)
    nC, nE = mesh.nCells, mesh.nEdges
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    lat = jnp.asarray(mesh.latCell)
    # stably stratified, latitude-dependent, plus noise -> non-zero sigma_dot
    sig = _coord(nlev).sigma_full
    T = (200.0 + 90.0 * sig[None, :] + 20.0 * jnp.cos(lat)[:, None]
         + 0.5 * jax.random.normal(k1, (nC, nlev)))
    state = MPASHydrostaticState(
        u=Field(data=10.0 * jax.random.normal(k2, (nE, nlev)), name="u",
                dims=("nEdges", "nlev"), units="m/s"),
        T=Field(data=T, name="T", dims=("nCells", "nlev"), units="K"),
        p_s=Field(data=1.0e5 + 500.0 * jax.random.normal(k3, (nC,)),
                  name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(data=jnp.zeros((nC,)), name="phis", dims=("nCells",),
                   units="m²/s²"),
        tracers={
            "q_v": Field(data=jnp.abs(1e-3 * jax.random.normal(k4, (nC, nlev))),
                         name="q_v", dims=("nCells", "nlev"), units="kg/kg"),
        },
    )
    return mesh, state


def test_mpas_tendencies_default_unchanged_and_van_leer_moves_all_three_paths():
    """End-to-end on a real MPAS state (codex round 1, P2): the earlier
    dispatch test passed ``state=None`` and so only exercised the guard.

    Pins that (a) the DEFAULT config is bit-identical to an explicit "upwind"
    config through the whole tendency, and (b) selecting "van_leer" changes
    ALL THREE transported quantities the one selector is documented to
    cover — theta (dT_dt), the tracers, and the edge winds (du_dt).
    """
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, mpas_hydrostatic_tendencies,
    )
    mesh, state = _mpas_state_and_mesh()
    coord = _coord()
    base = MPASPrimitiveEquationConfig()
    assert base.vert_advection_scheme == "upwind"
    t_def = mpas_hydrostatic_tendencies(state, mesh, coord, base)
    t_up = mpas_hydrostatic_tendencies(
        state, mesh, coord, base._replace(vert_advection_scheme="upwind"))
    t_vl = mpas_hydrostatic_tendencies(
        state, mesh, coord, base._replace(vert_advection_scheme="van_leer"))

    assert jnp.array_equal(t_def.dT_dt.data, t_up.dT_dt.data)
    assert jnp.array_equal(t_def.du_dt.data, t_up.du_dt.data)
    assert jnp.array_equal(t_def.tracer_tendencies["q_v"].data,
                           t_up.tracer_tendencies["q_v"].data)

    for name, a, b in (
            ("dT_dt (theta path)", t_def.dT_dt.data, t_vl.dT_dt.data),
            ("du_dt (edge winds)", t_def.du_dt.data, t_vl.du_dt.data),
            ("q_v (tracers)", t_def.tracer_tendencies["q_v"].data,
             t_vl.tracer_tendencies["q_v"].data)):
        assert bool(jnp.all(jnp.isfinite(b))), f"{name}: non-finite"
        d = float(jnp.max(jnp.abs(a - b)))
        assert d > 0.0, f"{name}: van_leer did not reach this transport path"
    # the surface-pressure tendency does NOT go through vertical advection
    assert jnp.array_equal(t_def.dp_s_dt.data, t_vl.dp_s_dt.data)


def test_convergence_at_a_fixed_physical_pressure_and_on_a_refined_grid():
    """Order at the 91.4 hPa target itself, and on a stretched grid.

    Codex round 1: the L-inf study excludes levels [3:-3], so re-measure the
    error at the FIXED physical level this option exists for (interpolated to
    the same sigma on every grid), and repeat on ``tropopause_refine=3``.
    """
    sigma_target = 91.4e2 / 1.0e5  # 91.4 hPa at p_s = 1000 hPa

    def slope(scheme, refine):
        errs = []
        for n in (30, 60, 120, 240, 480):
            coord = _coord(n, refine=refine)
            f = _theta_smooth(coord.sigma_full)[None, :]
            sd = _sigma_dot(coord.sigma_half)[None, :]
            sd_full = 0.5 * (sd[..., :-1] + sd[..., 1:])
            exact = -sd_full * _dtheta_dsigma(coord.sigma_full)[None, :]
            err = np.abs(np.asarray(
                vertical_advection(f, sd, coord, scheme=scheme) - exact))[0]
            # linear interpolation of the ERROR to the fixed physical sigma
            errs.append(float(np.interp(sigma_target,
                                        np.asarray(coord.sigma_full), err)))
        h = np.log(1.0 / np.array([30, 60, 120, 240, 480], float))
        return float(np.polyfit(h, np.log(np.array(errs)), 1)[0]), errs

    for refine in (1.0, 3.0):
        s_up, e_up = slope("upwind", refine)
        s_vl, e_vl = slope("van_leer", refine)
        assert 0.85 <= s_up <= 1.15, f"refine={refine} upwind slope {s_up:.3f}"
        assert s_vl >= 1.85, f"refine={refine} van_leer slope {s_vl:.3f}"
        assert e_vl[0] < e_up[0], (
            f"refine={refine}: van_leer not more accurate at nlev=30 "
            f"({e_vl[0]:.3e} vs {e_up[0]:.3e})")


def test_factory_forwards_the_scheme_to_the_built_mpas_model():
    """The FACTORY hand-off, end to end (codex round 2, P2).

    The MPAS-state test injects ``MPASPrimitiveEquationConfig`` directly and
    the CLI test stops at ``DycoreConfig``, so deleting the
    ``component_factory`` mapping would leave both green.  This closes that
    gap: build the dycore from an ``ExperimentConfig`` and read the scheme off
    the constructed model.
    """
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.driver.config import DycoreConfig, ExperimentConfig
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(1)
    coord = _coord(6)
    base = dict(discretization="mpas", model_type="hydrostatic", dt=300.0)
    for scheme in VERTICAL_ADVECTION_SCHEMES:
        cfg = ExperimentConfig(
            dycore=DycoreConfig(mpas_vert_advection_scheme=scheme, **base),
            grid=ExperimentConfig().grid._replace(grid_type="voronoi",
                                                  vertical_coord="sigma"))
        model = create_atmosphere_dycore(cfg, mesh, coord)
        assert model.config.vert_advection_scheme == scheme, (
            f"factory dropped the scheme: asked {scheme!r}, model has "
            f"{model.config.vert_advection_scheme!r}")
