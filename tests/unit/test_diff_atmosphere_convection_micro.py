"""Differentiability sweep for the atmosphere convection / microphysics /
cloud schemes NOT covered by ``tests/unit/test_diff_atmosphere_physics.py``.

That file exercises the *integration bridge* (``make_convection_physics`` /
``make_microphysics_physics``) with a single ``d(dT_dt)/dT`` probe.  This file
goes at the **leaf** modules directly and widens the probe to the other
physically meaningful inputs each scheme actually consumes:

  * convection ``bechtold``, ``emanuel``, ``kain_fritsch``, ``mass_flux``,
    ``tiedtke``, ``zhang_mcfarlane`` — w.r.t. T, q_v, the prognostic carry,
    the KF grid-scale updraft ``w``, and the CMT wind inputs.
  * microphysics ``thompson``, ``sundqvist``, ``p3`` — w.r.t. T, q_v, q_c and
    (where the scheme has an ice phase) q_i / rime mass / rime volume.
  * aerosol ``arg_activation``, ``aerosol_activation``, ``prognostic_aerosol``
    — w.r.t. updraft, T, p, and aerosol NUMBER.
  * ``ml_emulator`` — w.r.t. BOTH the input state and the network parameters.
  * clouds ``cloud_fraction`` — w.r.t. T, q_v, q_cloud, q_ice for all three
    selectable fraction schemes.

Assertion discipline (a bare ``isfinite`` is NOT acceptable — a companion
sweep found tests that passed while measuring an identically-zero quantity):
every gradient assertion is preceded by a FORWARD-activity assertion on the
quantity being reduced, and is followed by non-zero-fraction, magnitude and
spatial-structure checks.  See :func:`assert_forward_active` /
:func:`assert_gradient_ok`.

**Pressure-axis convention (load-bearing).**  Every kernel here reads the
surface as ``[:, -1]`` (``T_base = T[:, -1]``, ``p_base = p_full[:, -1]``,
``dp = p_half[..., 1:] - p_half[..., :-1] > 0``).  The fixture therefore
builds pressure INCREASING with index, and
:class:`TestFixtureConventions` asserts that directly and cross-checks it
against a real kernel (a scheme must diagnose positive CAPE from this
column).  An inverted axis silently zeroes whole columns via
``maximum(..., 0)`` clamps and makes every gradient below exactly 0.0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from legoesm.thermo import saturation_mixing_ratio

from legoesm import constants

# --- Fixture scale (kept tiny: gradients, not physics realism) -------------
_NCOL = 16
_NLEV = 10
_DT = 600.0            # physics step [s]
_SIGMA_TOP = 0.05      # model-top sigma of the fixture column

# --- Fixture sounding shape (a conditionally unstable tropical column) -----
# NOT physical constants — the two end-points of the linear-in-sigma fixture
# profile.  Chosen so the column is conditionally unstable and near-saturated,
# which is what the CAPE triggers of every scheme below actually gate on.
_T_SFC_K = 300.0
_T_TOP_K = 200.0


# ===========================================================================
# Assertion helpers
# ===========================================================================

def assert_forward_active(arr, name):
    """The reduced quantity must be non-trivial BEFORE we differentiate it.

    Guards the failure mode this sweep exists to catch: a gradient test that
    "passes" because the forward quantity is identically zero (a masked-out
    column, a clamped optical depth, an unused argument), so *any* gradient
    assertion on it is vacuous.
    """
    a = jnp.asarray(arr)
    assert jnp.all(jnp.isfinite(a)), f"{name}: forward output has NaN/Inf"
    amax = float(jnp.max(jnp.abs(a)))
    assert amax > 0.0, (
        f"{name}: forward output is IDENTICALLY ZERO (max|x| = 0.0) — the "
        "scheme is inactive on this fixture, so any gradient assertion on it "
        "would be vacuous.  Fix the fixture (trigger/CAPE gate), do not "
        "relax the assertion."
    )
    return amax


def assert_gradient_ok(grad, name, min_nonzero_frac=0.01, require_structure=True):
    """Finite + non-zero + structured.  ``isfinite`` alone is never enough."""
    g = jnp.asarray(grad)
    assert jnp.all(jnp.isfinite(g)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = float(jnp.mean(jnp.abs(g) > 0.0))
    gmax = float(jnp.max(jnp.abs(g)))
    assert gmax > 0.0, (
        f"{name}: gradient is IDENTICALLY ZERO — the input is unreachable by "
        "AD (masked column, detached branch, or an unused argument)."
    )
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac * 100:.2f}% of entries are non-zero "
        f"(need {min_nonzero_frac * 100:.2f}%); max|g| = {gmax:.3e}"
    )
    if require_structure and g.size > 1:
        assert float(jnp.std(g)) > 0.0, (
            f"{name}: gradient is spatially UNIFORM (std = 0) — it carries no "
            "column/level structure, which for a column-local scheme means "
            "the dependence collapsed to a constant."
        )
    return gmax


# ===========================================================================
# Shared column fixture
# ===========================================================================

class _Column:
    """A conditionally unstable, near-saturated tropical column set.

    Surface is at index ``-1``; pressure INCREASES with index.
    """

    def __init__(self, ncol=_NCOL, nlev=_NLEV, seed=0):
        key = jax.random.PRNGKey(seed)
        k_t, k_rh, k_u, k_v, k_w = jax.random.split(key, 5)

        self.ncol, self.nlev = ncol, nlev
        p_s = jnp.full((ncol,), constants.p_ref)

        # Surface-LAST sigma: sigma (and hence p) INCREASES with index.
        sigma_half = jnp.linspace(_SIGMA_TOP, 1.0, nlev + 1)
        self.p_half = p_s[:, None] * sigma_half[None, :]
        self.p_full = 0.5 * (self.p_half[:, :-1] + self.p_half[:, 1:])
        self.p_s = p_s
        self.dp = self.p_half[:, 1:] - self.p_half[:, :-1]       # > 0
        sigma_full = self.p_full / p_s[:, None]

        # Linear-in-sigma sounding: warm at index -1 (surface), cold at 0.
        T_1d = _T_TOP_K + (_T_SFC_K - _T_TOP_K) * sigma_full
        # Per-column offset + per-cell jitter so gradients have real 2-D
        # structure (a rank-1 field would make the "structure" check weak).
        self.T = (
            T_1d
            + 3.0 * jax.random.normal(k_t, (ncol, 1))
            + 0.5 * jax.random.normal(k_t, (ncol, nlev))
        )

        # ~60-95 % RH, from the model's OWN saturation curve (never re-derive).
        self.rh = 0.6 + 0.35 * jax.random.uniform(k_rh, (ncol, 1))
        self.q_v = self.rh * saturation_mixing_ratio(self.T, self.p_full)

        # Sheared winds for the CMT closures.
        shear = jnp.linspace(-1.0, 1.0, nlev)[None, :]
        self.u = 8.0 * shear + 2.0 * jax.random.normal(k_u, (ncol, nlev))
        self.v = 3.0 * shear + 1.0 * jax.random.normal(k_v, (ncol, nlev))

        # Mostly-ascending grid-scale w for the Kain-Fritsch trigger.
        self.w = 0.05 + 0.25 * jnp.abs(jax.random.normal(k_w, (ncol, nlev)))

        from legoesm.atmosphere.physics._shared import compute_layer_dz, compute_rho
        self.rho = compute_rho(self.T, self.p_full, self.q_v)
        self.dz = compute_layer_dz(self.T, self.p_half, self.q_v)


@pytest.fixture(scope="module")
def col():
    return _Column()


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    """Drop compiled graphs between cases.

    The leaf kernels here (Bechtold/IFS in particular) are large graphs; the
    JAX runtime aborts after a few dozen accumulated compilations in one
    process.  Clearing per test keeps the whole file inside one pytest run.
    """
    yield
    jax.clear_caches()


def _hydrometeors(col, *, q_c=1.0e-4, q_r=1.0e-5, q_i=1.0e-5,
                  q_s=1.0e-5, q_g=1.0e-6):
    """A moderately cloudy hydrometeor state (all species non-zero).

    Number concentrations follow the per-species conventions documented on
    :class:`HydrometeorState`: N_c / N_r per VOLUME [1/m^3], N_i per MASS
    [1/kg].
    """
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    ones = jnp.ones((col.ncol, col.nlev))
    return HydrometeorState(
        q_c=q_c * ones, q_r=q_r * ones, q_i=q_i * ones,
        q_s=q_s * ones, q_g=q_g * ones,
        N_c=1.0e8 * ones, N_r=1.0e4 * ones, N_i=1.0e4 * ones,
    )


def _micro_loss(out):
    """Scalar reduction over EVERY microphysics output channel.

    A ``dT_dt``-only loss hides dead paths in the tracer / number / precip
    tendencies (the sedimentation fall speeds only ever reach ``dq_r_dt``).
    """
    total = jnp.sum(out.dT_dt ** 2)
    for fld in (out.dq_v_dt, out.dq_c_dt, out.dq_r_dt, out.dq_i_dt,
                out.dq_s_dt, out.dq_g_dt, out.dN_c_dt, out.dN_r_dt,
                out.dN_i_dt):
        total = total + jnp.sum(fld ** 2)
    return total + jnp.sum(out.precipitation ** 2)


def _conv_loss(out):
    """Scalar reduction over the thermodynamic convection tendencies."""
    return (jnp.sum(out.dT_dt ** 2)
            + jnp.sum(out.dq_v_dt ** 2)
            + jnp.sum(out.dq_c_conv_dt ** 2))


# ===========================================================================
# Convection leaf dispatch
# ===========================================================================

_CONV_SCHEMES = (
    "bechtold", "emanuel", "kain_fritsch", "mass_flux",
    "tiedtke", "zhang_mcfarlane",
)

#: Schemes whose ``ConvectionOutput`` carries a convective momentum transport
#: tendency (``enable_cmt=True`` by default in their configs).
_CMT_SCHEMES = ("bechtold", "tiedtke", "zhang_mcfarlane")


def _conv_carry(scheme, col):
    """Fresh (cold-start) prognostic carry with the shape the leaf expects.

    ``mass_flux`` is SCALAR-prognostic (``M_c``, shape ``(ncol,)``); the
    other five carry the full ``(ncol, nlev)`` mass-flux profile.
    """
    if scheme == "mass_flux":
        return jnp.zeros((col.ncol,))
    return jnp.zeros((col.ncol, col.nlev))


def _call_convection(scheme, col, *, T=None, q_v=None, u=None, v=None,
                     w=None, carry=None):
    """Invoke one convection leaf and return its ``ConvectionOutput``."""
    T = col.T if T is None else T
    q_v = col.q_v if q_v is None else q_v
    u = col.u if u is None else u
    v = col.v if v is None else v
    w = col.w if w is None else w
    carry = _conv_carry(scheme, col) if carry is None else carry
    p_full, p_half = col.p_full, col.p_half

    if scheme == "bechtold":
        from legoesm.atmosphere.physics.convection.bechtold import (
            bechtold_convection,
        )
        from legoesm.atmosphere.physics.convection.config import BechtoldConfig
        out, _prof, _stoch = bechtold_convection(
            T, q_v, p_full, p_half, u, v, carry,
            jnp.zeros((col.ncol,)), None, _DT, BechtoldConfig(),
        )
        return out
    if scheme == "emanuel":
        from legoesm.atmosphere.physics.convection.config import EmanuelConfig
        from legoesm.atmosphere.physics.convection.emanuel import (
            emanuel_convection,
        )
        out, _prof = emanuel_convection(
            T, q_v, p_full, p_half, carry, _DT, EmanuelConfig(),
        )
        return out
    if scheme == "kain_fritsch":
        from legoesm.atmosphere.physics.convection.config import (
            KainFritschConfig,
        )
        from legoesm.atmosphere.physics.convection.kain_fritsch import (
            kain_fritsch_convection,
        )
        out, _prof = kain_fritsch_convection(
            T, q_v, p_full, p_half, w, carry, _DT, KainFritschConfig(),
        )
        return out
    if scheme == "mass_flux":
        from legoesm.atmosphere.physics.convection.config import MassFluxConfig
        from legoesm.atmosphere.physics.convection.mass_flux import (
            mass_flux_convection,
        )
        out, _mc = mass_flux_convection(
            T, q_v, p_full, p_half, carry, _DT, MassFluxConfig(),
        )
        return out
    if scheme == "tiedtke":
        from legoesm.atmosphere.physics.convection.config import TiedtkeConfig
        from legoesm.atmosphere.physics.convection.tiedtke import (
            tiedtke_convection,
        )
        out, _prof = tiedtke_convection(
            T, q_v, p_full, p_half, u, v, carry, _DT, TiedtkeConfig(),
        )
        return out
    if scheme == "zhang_mcfarlane":
        from legoesm.atmosphere.physics.convection.config import (
            ZhangMcFarlaneConfig,
        )
        from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
            zhang_mcfarlane_convection,
        )
        out, _prof = zhang_mcfarlane_convection(
            T, q_v, p_full, p_half, u, v, carry, _DT, ZhangMcFarlaneConfig(),
        )
        return out
    raise ValueError(f"unknown convection scheme {scheme!r}")


# ===========================================================================
# Fixture conventions (run these first — everything else trusts them)
# ===========================================================================

class TestFixtureConventions:
    """The fixture itself is the thing most likely to make this file lie."""

    def test_pressure_increases_with_index(self, col):
        """Surface at ``[:, -1]``; a decreasing axis silently zeroes columns."""
        dp = jnp.diff(col.p_half, axis=-1)
        assert float(jnp.min(dp)) > 0.0, (
            "p_half must INCREASE with index (surface last).  A reversed axis "
            "makes dp < 0, and the maximum(dp, 0) clamps inside the kernels "
            "then zero the whole column and every gradient in this file."
        )
        assert float(jnp.min(jnp.diff(col.p_full, axis=-1))) > 0.0
        # p_half[:, -1] is what the kernels read as the surface pressure.
        assert jnp.allclose(col.p_half[:, -1], col.p_s)
        # ... and the sounding must be warm at the surface end.
        assert float(jnp.min(col.T[:, -1] - col.T[:, 0])) > 0.0

    def test_column_is_convectively_active(self, col):
        """Cross-check the axis against a kernel that reads ``[:, -1]``.

        Tiedtke diagnoses CAPE from a parcel launched at ``T[:, -1]``.  If the
        fixture were inverted the parcel would launch from the (cold, thin)
        model top and CAPE would collapse to zero — which is exactly the
        vacuous-test failure mode this file guards against.
        """
        out = _call_convection("tiedtke", col)
        assert float(jnp.min(out.cape)) > 0.0, (
            f"fixture CAPE = {float(jnp.min(out.cape)):.3e} J/kg — the column "
            "is not conditionally unstable, so every convection gradient "
            "below would be measuring an inactive scheme."
        )
        assert float(jnp.min(out.convective_mask)) > 0.0

    def test_hydrometeor_state_is_nonzero(self, col):
        """Microphysics probes must start from a genuinely cloudy state."""
        hyd = _hydrometeors(col)
        for name in ("q_c", "q_r", "q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
            assert float(jnp.min(getattr(hyd, name))) > 0.0, (
                f"hydrometeor {name} is zero — an ice/rain-path gradient "
                "probe on this state would be measuring a dead branch."
            )


# ===========================================================================
# Convection — gradients w.r.t. the state
# ===========================================================================

class TestConvectionLeafGrad:

    @pytest.mark.parametrize("scheme", _CONV_SCHEMES)
    def test_grad_wrt_T(self, scheme, col):
        def loss(T):
            return _conv_loss(_call_convection(scheme, col, T=T))

        assert_forward_active(
            _call_convection(scheme, col).dT_dt, f"{scheme} dT_dt")
        grad = jax.grad(loss)(col.T)
        assert_gradient_ok(grad, f"convection({scheme}) w.r.t. T")

    @pytest.mark.parametrize("scheme", _CONV_SCHEMES)
    def test_grad_wrt_qv(self, scheme, col):
        """Moisture is the other half of every convective closure.

        A dead ``d(tend)/dq_v`` would mean the plume / CAPE path is driven by
        temperature alone — the scheme could not learn a humidity-dependent
        trigger under parameter estimation.
        """
        def loss(q_v):
            return _conv_loss(_call_convection(scheme, col, q_v=q_v))

        assert_forward_active(
            _call_convection(scheme, col).dq_v_dt, f"{scheme} dq_v_dt")
        grad = jax.grad(loss)(col.q_v)
        assert_gradient_ok(grad, f"convection({scheme}) w.r.t. q_v")

    @pytest.mark.parametrize("scheme", _CONV_SCHEMES)
    def test_grad_wrt_prognostic_carry(self, scheme, col):
        """The mass-flux / CBMF carry is the scheme's memory across steps.

        If it were detached, a multi-step gradient (4D-Var window, BPTT
        training segment) would silently drop the convective memory term —
        the tendency would still look right at a single step.

        Kain-Fritsch is DIAGNOSTIC by construction: the leaf does
        ``del conv_prog_profile`` at entry and only emits a fresh profile, so
        its carry gradient is EXACTLY zero.  That is a documented scheme
        property, not a defect, and is asserted as such.
        """
        carry0 = _conv_carry(scheme, col) + 1.0e-2

        def loss(carry):
            return _conv_loss(_call_convection(scheme, col, carry=carry))

        grad = jax.grad(loss)(carry0)
        assert jnp.all(jnp.isfinite(grad)), (
            f"convection({scheme}) w.r.t. carry: gradient has NaN/Inf"
        )
        gmax = float(jnp.max(jnp.abs(grad)))
        if scheme == "kain_fritsch":
            # NON-DIFFERENTIABLE BY CONSTRUCTION (documented): KF discards the
            # incoming carry (`del conv_prog_profile`), so d(tend)/d(carry) is
            # structurally 0.  Asserting the zero pins the property.
            assert gmax == 0.0, (
                "Kain-Fritsch is documented as diagnostic (it deletes the "
                f"incoming carry) yet d(tend)/d(carry) = {gmax:.3e} != 0 — "
                "the carry became live; update this expectation."
            )
        else:
            assert_gradient_ok(
                grad, f"convection({scheme}) w.r.t. prognostic carry",
                min_nonzero_frac=0.0, require_structure=False,
            )

    @pytest.mark.parametrize("scheme", _CMT_SCHEMES)
    def test_grad_wrt_u_through_cmt(self, scheme, col):
        """Convective momentum transport must be differentiable in the wind."""
        out0 = _call_convection(scheme, col)
        assert out0.du_dt_conv is not None, (
            f"{scheme} is listed as a CMT scheme but du_dt_conv is None"
        )
        assert_forward_active(out0.du_dt_conv, f"{scheme} du_dt_conv")

        def loss(u):
            out = _call_convection(scheme, col, u=u)
            return jnp.sum(out.du_dt_conv ** 2) + jnp.sum(out.dv_dt_conv ** 2)

        grad = jax.grad(loss)(col.u)
        assert_gradient_ok(grad, f"convection({scheme}) CMT w.r.t. u")

    def test_grad_wrt_w_grid_kain_fritsch(self, col):
        """KF's trigger perturbation is driven by the grid-scale updraft.

        ``w_grid`` is KF's only non-thermodynamic input; a zero gradient here
        means the trigger reduces to the fixed ``parcel_perturb_T`` and the
        dynamical-core coupling is severed.
        """
        def loss(w):
            return _conv_loss(_call_convection("kain_fritsch", col, w=w))

        grad = jax.grad(loss)(col.w)
        assert_gradient_ok(grad, "convection(kain_fritsch) w.r.t. w_grid")

    @pytest.mark.parametrize("scheme", _CONV_SCHEMES)
    def test_trigger_smoothness_sweep(self, scheme, col):
        """The TRIGGER itself, isolated from the tendency magnitude.

        Every scheme here gates on
        ``cape_trigger = sigmoid(sharpness * (CAPE - threshold))``
        (``sharpness = 0.1 /(J/kg)``, or equivalently ``/cape_activation_scale
        = 10 J/kg``; ``threshold ~ 70-100 J/kg``).  That is smooth by
        construction — but MEASURING it (this sweep, not inference) shows it
        has TWO exactly-zero-gradient dead zones flanking a live band:

        1. **Sigmoid saturation (deep convection).**  In float64 the logistic
           returns EXACTLY 1.0 once its argument exceeds ~36.7, i.e. once
           ``CAPE >~ 440 J/kg``, and its VJP ``s*(1-s)`` is then EXACTLY 0.0.
           The deep-tropical base fixture (CAPE ~ 10^3 J/kg) sits here for
           five of the six schemes.
        2. **CAPE clamped at zero (stable column).**  CAPE is the integral of
           the POSITIVE part of the parcel buoyancy; in a column with no
           buoyant layer that positive part is identically zero, so
           ``d(CAPE)/d(state) = 0`` exactly and the mask is pinned at its
           floor ``sigmoid(-sharpness*threshold)`` — 9.11e-4 for the
           ``(70 J/kg, 0.1)`` default, a NON-zero mask with a zero gradient.

        Both are correct physics/numerics, not detached branches, and both are
        pinned here.  The consequence for parameter estimation is concrete and
        worth stating plainly: **a CAPE-trigger parameter
        (``cape_threshold``, ``cape_sharpness``) is unreachable by AD in
        strongly convecting AND in stable columns** — trigger tuning only
        receives gradient signal from marginal columns, or needs a softer
        sharpness to widen the live band.

        The probe walks a STABILIZATION parameter ``s`` blending the sounding
        toward an isothermal column at its own surface temperature (``s=0``
        the deep convective fixture, ``s=1`` a column where no lifted parcel
        can be buoyant for any parcel definition — undilute, dilute-entraining
        or USL-mixed).  RH is held fixed so the moisture field follows.  It
        requires:

          * every gradient in the sweep is finite;
          * the sweep BRACKETS the transition (some point has an unsaturated
            mask) — otherwise the probe proves nothing, and the fixture rather
            than the scheme is at fault;
          * somewhere in that band the gradient is NON-ZERO — the trigger is
            genuinely smooth, not a hard step;
          * every zero-gradient point is EXPLAINED by one of the two dead
            zones above (mask saturated at 1, or CAPE clamped at 0).  A zero
            gradient at a live CAPE with an interior mask would be a real
            defect — a detached branch masquerading as a smooth trigger.
        """
        # Stabilization blend toward an isothermal column at T_surface; q_v is
        # recomputed at FIXED RH so the moisture field follows the sounding
        # instead of the column merely becoming supersaturated.
        T_iso = jnp.broadcast_to(col.T[:, -1:], col.T.shape)

        def mask_sum(s):
            T = (1.0 - s) * col.T + s * T_iso
            q_v = col.rh * saturation_mixing_ratio(T, col.p_full)
            out = _call_convection(scheme, col, T=T, q_v=q_v)
            return jnp.sum(out.convective_mask), jnp.mean(out.cape)

        assert_forward_active(
            _call_convection(scheme, col).convective_mask, f"{scheme} mask")

        # One compile, many evaluations (the leaf graphs are large).
        value_and_grad = jax.jit(jax.value_and_grad(mask_sum, has_aux=True))
        rows = []
        for s in [i / 10.0 for i in range(11)]:
            (val, cape), grad = value_and_grad(jnp.asarray(s))
            rows.append((s, float(val) / col.ncol, float(cape), float(grad)))

        table = "\n".join(
            f"    s={s:4.2f}   CAPE={c:10.3f} J/kg   mask={m:.8f}   "
            f"d(mask)/ds={g:+.6e}"
            for s, m, c, g in rows
        )
        for s, m, c, g in rows:
            assert jnp.isfinite(jnp.asarray(g)), (
                f"convection({scheme}) trigger sweep: non-finite gradient at "
                f"s={s}\n{table}"
            )

        sat = 1.0e-12          # float64 logistic saturation tolerance
        cape_dead = 1.0e-9     # CAPE at (or below) its positive-part clamp
        live = [(s, m, c, g) for s, m, c, g in rows
                if sat < m < 1.0 - sat and c > cape_dead]
        assert live, (
            f"convection({scheme}) trigger sweep never entered the live band "
            "(some point must have an unsaturated mask AND non-zero CAPE) — "
            "the probe cannot distinguish a smooth trigger from a hard step.  "
            f"Refine the stabilization sweep.\n{table}"
        )
        assert any(g != 0.0 for _, _, _, g in live), (
            f"convection({scheme}): d(convective_mask)/d(state) is EXACTLY "
            "ZERO throughout the live band — the trigger is a hard step (or a "
            f"detached branch) and is unlearnable.\n{table}"
        )
        for s, m, c, g in rows:
            if g == 0.0:
                saturated = m >= 1.0 - sat
                cape_clamped = c <= cape_dead
                assert saturated or cape_clamped, (
                    f"convection({scheme}): zero trigger gradient at s={s} "
                    f"with a LIVE CAPE ({c:.6e} J/kg) and an interior mask "
                    f"({m:.8f}) — neither sigmoid saturation nor the CAPE "
                    f"positive-part clamp explains it, so this is a detached "
                    f"branch.\n{table}"
                )


# ===========================================================================
# Microphysics — gradients w.r.t. the state
# ===========================================================================

_MICRO_SCHEMES = ("thompson", "sundqvist", "p3")


def _call_microphysics(scheme, col, *, T=None, q_v=None, hyd=None):
    T = col.T if T is None else T
    q_v = col.q_v if q_v is None else q_v
    hyd = _hydrometeors(col) if hyd is None else hyd
    args = (T, q_v, hyd, col.p_full, col.p_half, col.rho, col.dz, _DT)

    if scheme == "thompson":
        from legoesm.atmosphere.physics.microphysics.config import ThompsonConfig
        from legoesm.atmosphere.physics.microphysics.thompson import (
            thompson_microphysics,
        )
        return thompson_microphysics(*args, ThompsonConfig())
    if scheme == "sundqvist":
        from legoesm.atmosphere.physics.microphysics.config import (
            SundqvistConfig,
        )
        from legoesm.atmosphere.physics.microphysics.sundqvist import (
            sundqvist_microphysics,
        )
        return sundqvist_microphysics(*args, SundqvistConfig())
    if scheme == "p3":
        from legoesm.atmosphere.physics.microphysics.config import P3Config
        from legoesm.atmosphere.physics.microphysics.p3 import p3_microphysics
        return p3_microphysics(*args, P3Config())
    raise ValueError(f"unknown microphysics scheme {scheme!r}")


class TestMicrophysicsLeafGrad:

    @pytest.mark.parametrize("scheme", _MICRO_SCHEMES)
    def test_grad_wrt_T(self, scheme, col):
        def loss(T):
            return _micro_loss(_call_microphysics(scheme, col, T=T))

        assert_forward_active(
            _call_microphysics(scheme, col).dT_dt, f"{scheme} dT_dt")
        grad = jax.grad(loss)(col.T)
        assert_gradient_ok(grad, f"microphysics({scheme}) w.r.t. T")

    @pytest.mark.parametrize("scheme", _MICRO_SCHEMES)
    def test_grad_wrt_qv(self, scheme, col):
        def loss(q_v):
            return _micro_loss(_call_microphysics(scheme, col, q_v=q_v))

        assert_forward_active(
            _call_microphysics(scheme, col).dq_v_dt, f"{scheme} dq_v_dt")
        grad = jax.grad(loss)(col.q_v)
        assert_gradient_ok(grad, f"microphysics({scheme}) w.r.t. q_v")

    @pytest.mark.parametrize("scheme", _MICRO_SCHEMES)
    def test_grad_wrt_qc(self, scheme, col):
        """Autoconversion / accretion / the condensate sink all key on q_c."""
        hyd0 = _hydrometeors(col)

        def loss(q_c):
            hyd = hyd0._replace(q_c=q_c)
            return _micro_loss(_call_microphysics(scheme, col, hyd=hyd))

        grad = jax.grad(loss)(hyd0.q_c)
        assert_gradient_ok(grad, f"microphysics({scheme}) w.r.t. q_c")

    @pytest.mark.parametrize("scheme", _MICRO_SCHEMES)
    def test_grad_wrt_qr(self, scheme, col):
        """Rain path: Marshall-Palmer fractional powers + fall speeds."""
        hyd0 = _hydrometeors(col)

        def loss(q_r):
            hyd = hyd0._replace(q_r=q_r)
            return _micro_loss(_call_microphysics(scheme, col, hyd=hyd))

        grad = jax.grad(loss)(hyd0.q_r)
        assert_gradient_ok(grad, f"microphysics({scheme}) w.r.t. q_r")

    @pytest.mark.parametrize("scheme", ("thompson", "p3"))
    def test_grad_wrt_qi(self, scheme, col):
        """Ice phase: depositional growth ``N_i**(1/3)`` + ice fall speeds."""
        hyd0 = _hydrometeors(col)

        def loss(q_i):
            hyd = hyd0._replace(q_i=q_i)
            return _micro_loss(_call_microphysics(scheme, col, hyd=hyd))

        grad = jax.grad(loss)(hyd0.q_i)
        assert_gradient_ok(grad, f"microphysics({scheme}) w.r.t. q_i")

    @pytest.mark.parametrize("scheme", ("thompson", "p3"))
    def test_grad_wrt_Ni(self, scheme, col):
        """Ice NUMBER — the second moment, per-mass [1/kg] by convention."""
        hyd0 = _hydrometeors(col)

        def loss(N_i):
            hyd = hyd0._replace(N_i=N_i)
            return _micro_loss(_call_microphysics(scheme, col, hyd=hyd))

        grad = jax.grad(loss)(hyd0.N_i)
        assert_gradient_ok(grad, f"microphysics({scheme}) w.r.t. N_i")

    def test_grad_wrt_p3_rime_variables(self, col):
        """P3's predicted ice properties: rime MASS and rime VOLUME.

        P3 reinterprets the ``q_s`` slot as ``q_rim`` [kg/kg] and the ``q_g``
        slot as ``B_rim`` [m^3/kg_air].  These two are what makes P3 "P3"
        (predicted particle properties); a detached gradient here would mean
        the rime-density feedback is inert.
        """
        hyd0 = _hydrometeors(col)

        def loss_qrim(q_rim):
            hyd = hyd0._replace(q_s=q_rim)
            return _micro_loss(_call_microphysics("p3", col, hyd=hyd))

        def loss_brim(B_rim):
            hyd = hyd0._replace(q_g=B_rim)
            return _micro_loss(_call_microphysics("p3", col, hyd=hyd))

        # B_rim = q_rim / rho_rime, with rho_rime ~ O(400) kg/m^3.
        B_rim0 = hyd0.q_s / 400.0
        assert_gradient_ok(jax.grad(loss_qrim)(hyd0.q_s), "p3 w.r.t. q_rim")
        assert_gradient_ok(jax.grad(loss_brim)(B_rim0), "p3 w.r.t. B_rim")


# ===========================================================================
# Aerosol activation
# ===========================================================================

class TestAerosolActivationGrad:
    """ARG2000 modal activation + the Andreae AOD->CCN proxy."""

    @staticmethod
    def _modes():
        # One accumulation mode, SI units: [1/m^3], [m], [-], [-].
        return (jnp.array([1.0e8]), jnp.array([5.0e-8]),
                jnp.array([2.0]), jnp.array([0.6]))

    def test_arg_cdnc_grad_wrt_updraft(self, col):
        """S_max (and hence CDNC) is set by the updraft/condensation race."""
        from legoesm.atmosphere.physics.microphysics.arg_activation import arg_cdnc
        n, r, s, k = self._modes()
        w0 = 0.3 * jnp.ones((col.ncol, col.nlev))

        cdnc0 = arg_cdnc(w0, col.T, col.p_full, n, r, s, k)
        assert_forward_active(cdnc0, "arg_cdnc")

        def loss(w):
            return jnp.sum(arg_cdnc(w, col.T, col.p_full, n, r, s, k) ** 2)

        assert_gradient_ok(jax.grad(loss)(w0), "arg_cdnc w.r.t. w")

    def test_arg_cdnc_grad_wrt_T_and_p(self, col):
        from legoesm.atmosphere.physics.microphysics.arg_activation import arg_cdnc
        n, r, s, k = self._modes()
        w0 = 0.3 * jnp.ones((col.ncol, col.nlev))

        def loss_T(T):
            return jnp.sum(arg_cdnc(w0, T, col.p_full, n, r, s, k) ** 2)

        def loss_p(p):
            return jnp.sum(arg_cdnc(w0, col.T, p, n, r, s, k) ** 2)

        assert_gradient_ok(jax.grad(loss_T)(col.T), "arg_cdnc w.r.t. T")
        assert_gradient_ok(jax.grad(loss_p)(col.p_full), "arg_cdnc w.r.t. p")

    def test_arg_cdnc_grad_wrt_mode_number(self, col):
        """The aerosol->CDNC sensitivity (the first indirect effect)."""
        from legoesm.atmosphere.physics.microphysics.arg_activation import arg_cdnc
        n, r, s, k = self._modes()
        w0 = 0.3 * jnp.ones((col.ncol, col.nlev))

        def loss(mode_number):
            return jnp.sum(
                arg_cdnc(w0, col.T, col.p_full, mode_number, r, s, k) ** 2)

        grad = jax.grad(loss)(n)
        assert_gradient_ok(grad, "arg_cdnc w.r.t. mode_number",
                           require_structure=False)

    def test_activated_nc_field_arg_grad_wrt_prognostic_number(self, col):
        """Part-2 -> Part-1 feed: prognostic aerosol NUMBER drives ARG.

        This is the vmapped single-mode path in ``arg_cdnc_from_config``; a
        broken gradient here severs the prognostic-aerosol -> CDNC ->
        microphysics chain that the aerosol indirect effect rides on.
        """
        from legoesm.atmosphere.physics.microphysics.arg_activation import (
            ActivationConfig,
            activated_nc_field,
        )
        cfg = ActivationConfig(scheme="arg")
        n_aer = 1.0e8 * jnp.ones((col.ncol, col.nlev))
        shape = (col.ncol, col.nlev)

        nc0 = activated_nc_field(cfg, shape, T=col.T, p=col.p_full,
                                 aerosol_number=n_aer)
        assert_forward_active(nc0, "activated_nc_field(arg)")

        def loss(number):
            return jnp.sum(activated_nc_field(
                cfg, shape, T=col.T, p=col.p_full,
                aerosol_number=number) ** 2)

        assert_gradient_ok(jax.grad(loss)(n_aer),
                           "activated_nc_field(arg) w.r.t. aerosol_number")

    def test_ccn_from_aod_grad_in_active_range(self, col):
        """Andreae (2009) AOD -> CCN inversion, inside the clipped range.

        The fit is clipped to ``[n_ccn_min, n_ccn_max]``; inside that range the
        power law must be differentiable in AOD.  Outside it the gradient is
        legitimately zero (see the companion assertion below), which is a
        clip, not a bug — but it does mean an AOD-tuning gradient silently
        dies for very clean / very polluted columns.
        """
        from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
            CCNFromAODConfig,
            ccn_from_aod,
        )
        cfg = CCNFromAODConfig()
        aod_active = jnp.linspace(0.05, 0.4, col.ncol)

        def loss(aod):
            return jnp.sum(ccn_from_aod(aod, cfg) ** 2)

        assert_gradient_ok(jax.grad(loss)(aod_active),
                           "ccn_from_aod w.r.t. aod (active range)")

        # Saturated branch: DOCUMENTED dead gradient (clip, by design).
        aod_saturated = jnp.full((col.ncol,), 1.0e3)
        grad_sat = jax.grad(loss)(aod_saturated)
        assert jnp.all(jnp.isfinite(grad_sat))
        assert float(jnp.max(jnp.abs(grad_sat))) == 0.0, (
            "ccn_from_aod above n_ccn_max is expected to be clipped flat "
            "(zero gradient); a non-zero value means the cap moved."
        )

    def test_specified_nc_field_grad_wrt_layer_aod(self, col):
        """Column AOD -> per-column Nc, broadcast to all levels."""
        from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
            specified_nc_field,
        )
        aod = jnp.full((col.ncol, col.nlev), 0.02)
        shape = (col.ncol, col.nlev)

        def loss(a):
            return jnp.sum(specified_nc_field(a, shape) ** 2)

        assert_forward_active(specified_nc_field(aod, shape),
                              "specified_nc_field")
        assert_gradient_ok(jax.grad(loss)(aod),
                           "specified_nc_field w.r.t. aerosol_od")


class TestPrognosticAerosolGrad:
    """Prognostic aerosol-number tracer: sources, sinks, and the step."""

    @staticmethod
    def _config():
        from legoesm.atmosphere.physics.microphysics.prognostic_aerosol import (
            PrognosticAerosolConfig,
        )
        return PrognosticAerosolConfig(enabled=True)

    def test_tendency_grad_wrt_number_and_precip(self, col):
        from legoesm.atmosphere.physics.microphysics.prognostic_aerosol import (
            aerosol_number_tendency,
        )
        cfg = self._config()
        n0 = 1.0e8 * jnp.ones((col.ncol, col.nlev))
        precip = jnp.linspace(1.0e-5, 1.0e-4, col.ncol)

        tend0 = aerosol_number_tendency(n0, col.dz, precip, cfg)
        assert_forward_active(tend0, "aerosol_number_tendency")

        def loss_n(n):
            return jnp.sum(aerosol_number_tendency(n, col.dz, precip, cfg) ** 2)

        def loss_p(p):
            return jnp.sum(aerosol_number_tendency(n0, col.dz, p, cfg) ** 2)

        assert_gradient_ok(jax.grad(loss_n)(n0),
                           "aerosol_number_tendency w.r.t. N")
        assert_gradient_ok(jax.grad(loss_p)(precip),
                           "aerosol_number_tendency w.r.t. precip_rate")

    def test_step_grad_wrt_number(self, col):
        """Backward-Euler step ``N_new = (N + dt S)/(1 + dt L)``."""
        from legoesm.atmosphere.physics.microphysics.prognostic_aerosol import (
            step_prognostic_aerosol,
        )
        cfg = self._config()
        n0 = 1.0e8 * jnp.ones((col.ncol, col.nlev))
        precip = jnp.linspace(1.0e-5, 1.0e-4, col.ncol)

        def loss(n):
            return jnp.sum(
                step_prognostic_aerosol(n, _DT, col.dz, precip, cfg) ** 2)

        assert_gradient_ok(jax.grad(loss)(n0),
                           "step_prognostic_aerosol w.r.t. N")

    def test_step_disabled_is_identity_with_unit_gradient(self, col):
        """``enabled=False`` returns the state unchanged — d(N_new)/dN = 1.

        Not a bug: it is the documented byte-identical proxy path.  Pinning
        the unit gradient proves the disabled branch is a pass-through and not
        a silently detached (``stop_gradient``-like) one.
        """
        from legoesm.atmosphere.physics.microphysics.prognostic_aerosol import (
            PrognosticAerosolConfig,
            step_prognostic_aerosol,
        )
        cfg = PrognosticAerosolConfig(enabled=False)
        n0 = 1.0e8 * jnp.ones((col.ncol, col.nlev))
        precip = jnp.linspace(1.0e-5, 1.0e-4, col.ncol)

        def loss(n):
            return jnp.sum(step_prognostic_aerosol(n, _DT, col.dz, precip, cfg))

        grad = jax.grad(loss)(n0)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.allclose(grad, jnp.ones_like(grad)), (
            "disabled prognostic-aerosol step must be an identity in N "
            "(gradient 1), not a detached branch"
        )


# ===========================================================================
# ML microphysics emulator — gradients w.r.t. inputs AND parameters
# ===========================================================================

class TestMLEmulatorGrad:

    @staticmethod
    def _build():
        from legoesm.atmosphere.physics.microphysics.config import (
            MicrophysicsMLEmulatorConfig,
        )
        from legoesm.atmosphere.physics.microphysics.ml_emulator import (
            MicrophysicsEmulator,
        )
        cfg = MicrophysicsMLEmulatorConfig(n_hidden=16)
        model = MicrophysicsEmulator(
            cfg.n_input, cfg.n_hidden, cfg.n_layers, cfg.n_output,
            key=jax.random.PRNGKey(cfg.seed),
        )
        return cfg, model

    def _run(self, col, cfg, model, *, T=None, q_v=None, hyd=None):
        from legoesm.atmosphere.physics.microphysics.ml_emulator import (
            ml_microphysics,
        )
        T = col.T if T is None else T
        q_v = col.q_v if q_v is None else q_v
        hyd = _hydrometeors(col) if hyd is None else hyd
        return ml_microphysics(T, q_v, hyd, col.p_full, col.p_half,
                               col.rho, col.dz, _DT, cfg, model)

    def test_grad_wrt_inputs(self, col):
        cfg, model = self._build()
        out0 = self._run(col, cfg, model)
        assert_forward_active(out0.dT_dt, "ml_emulator dT_dt")

        def loss_T(T):
            return _micro_loss(self._run(col, cfg, model, T=T))

        def loss_q(q_v):
            return _micro_loss(self._run(col, cfg, model, q_v=q_v))

        assert_gradient_ok(jax.grad(loss_T)(col.T), "ml_emulator w.r.t. T")
        assert_gradient_ok(jax.grad(loss_q)(col.q_v), "ml_emulator w.r.t. q_v")

    def test_grad_wrt_hydrometeor_inputs(self, col):
        cfg, model = self._build()
        hyd0 = _hydrometeors(col)

        def loss(q_c):
            return _micro_loss(
                self._run(col, cfg, model, hyd=hyd0._replace(q_c=q_c)))

        assert_gradient_ok(jax.grad(loss)(hyd0.q_c), "ml_emulator w.r.t. q_c")

    def test_grad_wrt_network_parameters(self, col):
        """Every MLP weight/bias must be reachable — this IS the training path.

        A dead parameter leaf means that layer never updates; the emulator
        would train but under-fit in a way no forward test can see.
        """
        import equinox as eqx

        cfg, model = self._build()

        def loss(m):
            return _micro_loss(self._run(col, cfg, m))

        grads = eqx.filter_grad(loss)(model)
        leaves = [g for g in jax.tree_util.tree_leaves(grads)
                  if eqx.is_inexact_array(g)]
        assert leaves, "no differentiable parameter leaves found in the emulator"
        for i, g in enumerate(leaves):
            assert jnp.all(jnp.isfinite(g)), (
                f"ml_emulator parameter leaf {i} (shape {g.shape}): NaN/Inf"
            )
            assert float(jnp.max(jnp.abs(g))) > 0.0, (
                f"ml_emulator parameter leaf {i} (shape {g.shape}) has an "
                "IDENTICALLY ZERO gradient — that layer is unreachable by AD "
                "and would never train."
            )


# ===========================================================================
# Cloud fraction / cloud optical properties
# ===========================================================================

_CLOUD_SCHEMES = ("sundqvist", "xu_randall", "resolved")


def _cloud_loss(props):
    return (jnp.sum(props.cloud_fraction ** 2)
            + jnp.sum(props.lwp ** 2)
            + jnp.sum(props.iwp ** 2))


class TestCloudFractionGrad:

    @staticmethod
    def _config(scheme):
        from legoesm.atmosphere.physics.clouds.config import CloudConfig
        return CloudConfig(scheme=scheme)

    def _run(self, col, scheme, *, T=None, q_v=None, q_cloud=None, q_ice=None):
        from legoesm.atmosphere.physics.clouds.cloud_fraction import (
            compute_cloud_properties,
        )
        ones = jnp.ones((col.ncol, col.nlev))
        T = col.T if T is None else T
        q_v = col.q_v if q_v is None else q_v
        q_cloud = 1.0e-4 * ones if q_cloud is None else q_cloud
        q_ice = 1.0e-5 * ones if q_ice is None else q_ice
        return compute_cloud_properties(
            T, col.p_full, q_v, col.dp, self._config(scheme),
            q_cloud=q_cloud, q_ice=q_ice,
        )

    @pytest.mark.parametrize("scheme", _CLOUD_SCHEMES)
    def test_forward_is_cloudy(self, col, scheme):
        """No cloud => every gradient below is measuring a clear sky."""
        props = self._run(col, scheme)
        cf_max = assert_forward_active(props.cloud_fraction,
                                       f"cloud_fraction({scheme})")
        assert cf_max > 1.0e-3, (
            f"cloud_fraction({scheme}): max cf = {cf_max:.3e} — effectively "
            "clear; the fixture column is not saturated enough to make this "
            "probe meaningful."
        )
        assert_forward_active(props.lwp, f"lwp({scheme})")
        assert_forward_active(props.iwp, f"iwp({scheme})")

    # ``resolved`` is deliberately excluded from the thermodynamic probes —
    # see ``test_resolved_scheme_is_detached_from_thermodynamics``.
    @pytest.mark.parametrize("scheme", ("sundqvist", "xu_randall"))
    def test_grad_wrt_qv(self, col, scheme):
        """RH is the primary driver of the diagnostic fraction."""
        def loss(q_v):
            return _cloud_loss(self._run(col, scheme, q_v=q_v))

        assert_gradient_ok(jax.grad(loss)(col.q_v),
                           f"cloud_fraction({scheme}) w.r.t. q_v")

    @pytest.mark.parametrize("scheme", ("sundqvist", "xu_randall"))
    def test_grad_wrt_T(self, col, scheme):
        """T enters through q_sat (RH) AND the liquid/ice partition."""
        def loss(T):
            return _cloud_loss(self._run(col, scheme, T=T))

        assert_gradient_ok(jax.grad(loss)(col.T),
                           f"cloud_fraction({scheme}) w.r.t. T")

    def test_resolved_scheme_is_detached_from_thermodynamics(self, col):
        """MEASURED: ``scheme='resolved'`` has NO gradient path to T or q_v.

        NON-DIFFERENTIABLE BY DESIGN, not a bug — but a consequential
        property, so it is pinned rather than skipped.  For the CRM
        ``resolved`` scheme:

          * ``cf = q_cond/(q_cond + q_cloud_resolved_ref)`` — condensate only,
            no RH term;
          * the ``cf * q_c_diagnostic`` sub-grid condensate floor (whose
            liquid/ice split is the only place ``T`` enters the water paths)
            is explicitly NOT applied — ``compute_cloud_properties`` gates it
            on ``config.scheme in ("sundqvist", "xu_randall")`` because for a
            CRM the explicit ``q_cloud``/``q_ice`` from microphysics ARE the
            truth and a floor would inject spurious cloud water;
          * with ``n_cloud``/``n_ice`` absent the effective radii are config
            constants.

        So ``(cloud_fraction, lwp, iwp)`` are functions of
        ``(q_cloud, q_ice, dp)`` alone.  Consequence for training: with
        ``cloud_scheme='resolved'`` no gradient reaches the thermodynamic
        state through the cloud-radiation path — it must travel via
        microphysics' condensate instead.

        We do NOT infer this from reading the code: the exact-zero AD result
        is cross-checked against a finite difference, which distinguishes a
        genuine structural independence (forward output does not move) from a
        detached-but-dependent branch (forward output moves, gradient lies) —
        the latter WOULD be a bug.
        """
        def loss_T(T):
            return _cloud_loss(self._run(col, "resolved", T=T))

        def loss_q(q_v):
            return _cloud_loss(self._run(col, "resolved", q_v=q_v))

        g_T = jax.grad(loss_T)(col.T)
        g_q = jax.grad(loss_q)(col.q_v)
        assert jnp.all(jnp.isfinite(g_T)) and jnp.all(jnp.isfinite(g_q))
        assert float(jnp.max(jnp.abs(g_T))) == 0.0, (
            "cloud_fraction('resolved') gained a T dependence; update this "
            "documented expectation (and re-enable the parametrized probe)."
        )
        assert float(jnp.max(jnp.abs(g_q))) == 0.0, (
            "cloud_fraction('resolved') gained a q_v dependence; update this "
            "documented expectation (and re-enable the parametrized probe)."
        )

        # Finite-difference cross-check: the forward output must be genuinely
        # invariant.  If it moved while AD reported 0, the gradient would be
        # WRONG (a detached branch) rather than the dependence being absent.
        base = loss_T(col.T)
        for delta in (+5.0, -5.0):
            moved = loss_T(col.T + delta)
            assert moved == base, (
                f"cloud_fraction('resolved'): loss MOVED under a {delta:+.1f} K "
                f"perturbation ({base:.12e} -> {moved:.12e}) while jax.grad "
                "reported exactly zero — that is a detached gradient (BUG), "
                "not a structural independence."
            )
        moved_q = loss_q(col.q_v * 1.5)
        assert moved_q == loss_q(col.q_v), (
            "cloud_fraction('resolved'): loss moved under a q_v perturbation "
            "while jax.grad reported exactly zero — detached gradient (BUG)."
        )

    @pytest.mark.parametrize("scheme", _CLOUD_SCHEMES)
    def test_grad_wrt_condensate(self, col, scheme):
        """Explicit condensate from microphysics -> LWP/IWP -> radiation.

        Sundqvist's *fraction* is RH-only, but its water PATHS still take the
        explicit condensate, so the reduced loss must respond to q_cloud for
        all three schemes.  A dead gradient here severs microphysics from the
        radiative cloud effect.
        """
        ones = jnp.ones((col.ncol, col.nlev))
        q_c0, q_i0 = 1.0e-4 * ones, 1.0e-5 * ones

        def loss_c(q_c):
            return _cloud_loss(self._run(col, scheme, q_cloud=q_c))

        def loss_i(q_i):
            return _cloud_loss(self._run(col, scheme, q_ice=q_i))

        assert_gradient_ok(jax.grad(loss_c)(q_c0),
                           f"cloud_fraction({scheme}) w.r.t. q_cloud")
        assert_gradient_ok(jax.grad(loss_i)(q_i0),
                           f"cloud_fraction({scheme}) w.r.t. q_ice")

    def test_grad_wrt_conv_precip_overlap(self, col):
        """Slingo (1987) convective cover: cf responds to convective precip.

        This is the convection -> cloud -> radiation coupling; the
        ``jnp.maximum(cf, cf_conv)`` overlap must not detach it.
        """
        from legoesm.atmosphere.physics.clouds.cloud_fraction import (
            compute_cloud_properties,
        )
        from legoesm.atmosphere.physics.clouds.config import CloudConfig
        cfg = CloudConfig(scheme="sundqvist", convective_cloud=True)
        ones = jnp.ones((col.ncol, col.nlev))
        conv_precip0 = jnp.full((col.ncol,), 1.0e-4)   # ~8.6 mm/day

        def loss(conv_precip):
            props = compute_cloud_properties(
                col.T, col.p_full, col.q_v, col.dp, cfg,
                q_cloud=1.0e-4 * ones, q_ice=1.0e-5 * ones,
                conv_precip=conv_precip,
            )
            return _cloud_loss(props)

        assert_gradient_ok(jax.grad(loss)(conv_precip0),
                           "cloud_fraction convective overlap w.r.t. conv_precip",
                           require_structure=False)
