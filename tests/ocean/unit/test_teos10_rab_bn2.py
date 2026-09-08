"""NEMO TEOS-10 rab / bn2 — the coefficient set ORCA1 actually runs.

ORCA1's namelist_cfg sets ``ln_teos10 = .true.`` (line 308, with ``ln_eos80``
commented out) and NEMO then sets ``l_useCT = .TRUE.``. Our card had been
computing stratification from a clipped in-situ density gradient, and job
9407791 showed that is the whole remaining mixing-length deficit: swapping
only the N2 that seeds the buoyancy length moved the zero-step ratio against
NEMO's own length from 0.897 to 0.997 (Southern Ocean), 0.914 to 0.998
(tropics) and 0.792 to 1.004 (Arctic).

These tests guard the transcription, which is where a silent wrong number
would enter: 122 coefficients across three families.
"""
from __future__ import annotations

import math

import jax.numpy as jnp

import numpy as np
import pytest


def _eos():
    from legoesm.ocean import eos
    return eos


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def test_coefficient_table_is_bit_pinned():
    """Counts and 'differs from EOS-80' cannot catch a ONE-DIGIT edit.

    Codex 9408213 flagged that gap. This hashes the whole table, so any future
    change to any of the 126 entries goes red and has to be justified against
    eosbn2.F90 rather than slipping through.
    """
    import hashlib
    import json
    c = _eos()._ROQUET_TEOS10
    items = sorted((k, repr(v)) for k, v in c.items())
    h = hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]
    assert len(c) == 126, len(c)
    assert h == "fffd0a0f90450c83", (
        f"TEOS-10 coefficient table changed (hash {h}). If deliberate, "
        "re-derive it with the mechanical extractor against eosbn2.F90:"
        "1926-2110 and update this hash -- do not hand-edit coefficients.")


def test_coefficient_families_are_complete():
    """52 density + 35 thermal + 35 haline, matching NEMO's block exactly.

    A missing coefficient would silently evaluate as a KeyError only on the
    branch that reads it, so count them up front.
    """
    c = _eos()._ROQUET_TEOS10
    fams = {}
    for k in c:
        if k[:3] in ("EOS", "ALP", "BET"):
            fams[k[:3]] = fams.get(k[:3], 0) + 1
    assert fams == {"EOS": 52, "ALP": 35, "BET": 35}, fams


def test_normalization_differs_from_eos80_where_nemo_says_it_does():
    """The two easy-to-miss differences, pinned.

    NEMO's TEOS-10 branch sets rdeltaS = 32 (EOS-80 uses 20) and
    r1_S0 = 0.875/35.16504, the Absolute-Salinity scaling (EOS-80 uses 1/40).
    Copying the EOS-80 normalization onto TEOS-10 coefficients would be a
    plausible-looking, entirely wrong EOS.
    """
    t = _eos()._ROQUET_TEOS10
    e = _eos()._ROQUET_EOS80
    assert t["rdeltaS"] == 32.0 and e["rdeltaS"] == 20.0
    assert t["r1_S0"] == pytest.approx(0.875 / 35.16504, rel=1e-15)
    assert e["r1_S0"] == pytest.approx(1.0 / 40.0, rel=1e-15)
    # shared
    assert t["r1_T0"] == pytest.approx(1.0 / 40.0, rel=1e-15)
    assert t["r1_Z0"] == 1.0e-4


def test_coefficients_are_not_the_eos80_set():
    """Non-vacuity: the two tables must actually differ.

    If a copy-paste had left the EOS-80 numbers under the TEOS-10 name every
    other test here would still pass.
    """
    t, e = _eos()._ROQUET_TEOS10, _eos()._ROQUET_EOS80
    shared = [k for k in t if k.startswith("EOS") and k in e]
    assert len(shared) == 52
    differing = [k for k in shared if t[k] != e[k]]
    assert len(differing) == 52, (
        f"only {len(differing)}/52 density coefficients differ between the "
        "TEOS-10 and EOS-80 sets; they should all differ")


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def test_alpha_beta_physical_magnitudes():
    """Sanity-check against known seawater values before anything is built on it.

    At 10 degC, 35 psu, near-surface: thermal expansion is ~1.7e-4 /K and
    haline contraction ~7.6e-4 /psu for seawater. Getting the /zs on beta
    wrong (it is inside NEMO's expression, not a normalization) lands beta
    around 5x off, which this catches.
    """
    import jax.numpy as jnp
    alpha, beta = _eos().nemo_roquet_alpha_beta(
        jnp.array([10.0]), jnp.array([35.0]), jnp.array([0.0]))
    a, b = float(alpha[0]), float(beta[0])
    assert 1.0e-4 < a < 2.5e-4, f"alpha {a:.3e} outside the seawater range"
    assert 7.0e-4 < b < 8.2e-4, f"beta {b:.3e} outside the seawater range"


def test_alpha_grows_with_temperature():
    """Thermal expansion increases with temperature in seawater."""
    import jax.numpy as jnp
    T = jnp.array([0.0, 10.0, 20.0, 30.0])
    S = jnp.full_like(T, 35.0)
    z = jnp.zeros_like(T)
    alpha, _ = _eos().nemo_roquet_alpha_beta(T, S, z)
    a = np.asarray(alpha)
    assert np.all(np.diff(a) > 0), a


def test_bn2_teos10_differs_from_seos_and_is_signed():
    """The wiring is live, and it keeps the signed convention.

    Signed matters: the whole finding is that clipping at zero suppresses the
    long mixing lengths NEMO produces in convectively neutral water.
    """
    import jax.numpy as jnp
    e = _eos()
    nlev = 12
    gdept = jnp.asarray(np.linspace(5.0, 500.0, nlev))
    gdepw = 0.5 * (gdept[:-1] + gdept[1:])
    # stably stratified column with ONE inverted pair
    _t = np.linspace(18.0, 4.0, nlev)
    _t[5] = _t[6] - 0.5                          # cell 5 colder than 6 -> unstable
    T = jnp.asarray(_t)
    S = jnp.full((nlev,), 35.0)
    n2_seos = e.compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept, gdepw, e3w_source="depth_difference")
    n2_teos = e.compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept, gdepw, eos_form="teos10",
        e3w_source="depth_difference")
    a, b = np.asarray(n2_seos), np.asarray(n2_teos)
    assert a.shape == b.shape == (nlev - 1,)
    assert not np.allclose(a, b), "teos10 produced the S-EOS answer"
    # same sign structure: both must see the inverted pair as unstable
    # Interior interface i sits between cell i (upper) and cell i+1 (lower),
    # so inverting cells 5 and 6 makes interface 5 unstable -- not 4. The
    # first revision asserted 4 and failed; the INDEX was wrong, not the code.
    assert b[5] < 0.0, f"teos10 lost the unstable interface: {b[5]:.3e}"
    assert a[5] < 0.0
    # and agree to within a modest factor elsewhere (same physics, better fit)
    stable = np.arange(nlev - 1) != 5
    r = b[stable] / a[stable]
    assert np.all((r > 0.5) & (r < 2.0)), r


def test_unknown_eos_form_raises():
    """Dispatch hardening: a typo must not silently pick the S-EOS."""
    import jax.numpy as jnp
    e = _eos()
    T = jnp.full((4,), 10.0)
    S = jnp.full((4,), 35.0)
    gd = jnp.asarray([5.0, 15.0, 30.0, 50.0])
    gw = jnp.asarray([10.0, 22.0, 40.0])
    with pytest.raises(ValueError, match="eos_form"):
        e.compute_buoyancy_frequency_nemo_bn2(
            T, S, gd, gw, eos_form="teos-10",
            e3w_source="depth_difference")


def test_default_stays_seos_bit_identical():
    """The default path must not move: every prior arm scored on it."""
    import jax.numpy as jnp
    e = _eos()
    T = jnp.asarray(np.linspace(15.0, 3.0, 8))
    S = jnp.asarray(np.linspace(34.5, 34.9, 8))
    gd = jnp.asarray(np.linspace(5.0, 300.0, 8))
    gw = 0.5 * (gd[:-1] + gd[1:])
    a = np.asarray(e.compute_buoyancy_frequency_nemo_bn2(
        T, S, gd, gw, e3w_source="depth_difference"))
    b = np.asarray(e.compute_buoyancy_frequency_nemo_bn2(
        T, S, gd, gw, eos_form="seos",
        e3w_source="depth_difference"))
    assert np.array_equal(a, b)


def test_alpha_beta_is_differentiable():
    """Both are used inside the TKE closure, which must stay grad-safe."""
    import jax
    import jax.numpy as jnp
    e = _eos()

    def f(T):
        alpha, beta = e.nemo_roquet_alpha_beta(
            T, jnp.full_like(T, 35.0), jnp.full_like(T, 100.0))
        return jnp.sum(alpha) + jnp.sum(beta)

    g = jax.grad(f)(jnp.array([5.0, 15.0, 25.0]))
    assert np.all(np.isfinite(np.asarray(g))), g
    assert np.any(np.asarray(g) != 0.0)


def test_alpha_beta_match_the_derivative_of_the_density_polynomial():
    """The decisive check on NEMO's ``zn / zs`` on beta.

    NEMO writes ``pab(jp_sal) = zn / zs * r1_rho0`` with zs = sqrt(scaled S),
    and there is no explicit r1_S0/2 chain-rule factor in that line -- it is
    absorbed into the BET coefficients. If it were NOT absorbed, our beta
    would be off by r1_S0/(2*zs), i.e. a factor of several.

    Rather than reason about it, differentiate the density polynomial these
    coefficients belong to. NEMO's convention is

        alpha = -(1/rho0) d(rho)/dT ,   beta = +(1/rho0) d(rho)/dS

    so a central difference on ``nemo_roquet_eos`` evaluated with the SAME
    TEOS-10 coefficient set must reproduce ``nemo_roquet_alpha_beta``. This
    also catches a wrong normalization constant, since both sides would have
    to be wrong identically to agree.
    """
    import jax.numpy as jnp
    from legoesm import constants
    e = _eos()
    # MUST match nemo_roquet_alpha_beta's default, which is NEMO's
    # rho0 = 1026 -- not legoESM's constants.rho_ocean = 1025. Using
    # the wrong one makes the derivative identity fail by 1026/1025.
    rho0 = e._NEMO_RHO0
    T0, S0, depth = 10.0, 35.0, 500.0
    # Pressure MUST be built with the SAME rho0 the EOS inverts it with
    # (docstring: depth is recovered as p/(rho0*g)). Using
    # constants.rho_ocean = 1025 here while passing rho0 = 1026 below
    # shifts the recovered depth by 0.0975%, which is 4.9 m at 5000 m
    # and was the entire measured alpha mismatch (job 9410757).
    p = rho0 * constants.g * depth                     # nemo_roquet_eos takes Pa

    def rho(T, S):
        return float(e.nemo_roquet_eos(
            jnp.array([T]), jnp.array([S]), jnp.array([p]),
            coeffs=e._ROQUET_TEOS10, rho0=rho0)[0])

    dT, dS = 1.0e-3, 1.0e-3
    alpha_fd = -(rho(T0 + dT, S0) - rho(T0 - dT, S0)) / (2 * dT) / rho0
    beta_fd = (rho(T0, S0 + dS) - rho(T0, S0 - dS)) / (2 * dS) / rho0

    alpha, beta = e.nemo_roquet_alpha_beta(
        jnp.array([T0]), jnp.array([S0]), jnp.array([depth]))
    a, b = float(alpha[0]), float(beta[0])

    assert a == pytest.approx(alpha_fd, rel=2e-4), (
        f"alpha {a:.6e} != d(rho)/dT {alpha_fd:.6e}")
    assert b == pytest.approx(beta_fd, rel=2e-4), (
        f"beta {b:.6e} != d(rho)/dS {beta_fd:.6e} -- the /zs factor on "
        "NEMO's beta line is wrong (dropped, doubled, or double-counted)")


def test_alpha_beta_match_the_density_derivative_ACROSS_THE_OCEAN():
    """The same identity as above, swept over the ocean's real T/S/p range.

    The single-point version (T=10, S=35, 500 m) is the decisive check on
    NEMO's zn/zs beta line, but it is a weak check on the COEFFICIENT TABLES:
    126 numbers were transcribed by hand, and a typo in a high-order term is
    invisible at one point while changing the answer at cold/fresh or
    warm/deep. Nothing else validates the transcription -- the hash pin only
    locks in whatever was typed.

    What makes this an independent check rather than a tautology: alpha/beta
    come from the 35-entry ALP_ and BET_ tables, while the density comes from
    the SEPARATE 52-entry EOS_ table. They are different columns of Roquet et
    al. (2015) transcribed separately, so a typo in one has to be matched by
    an exactly compensating typo in the other to survive.
    """
    import itertools
    import jax
    import jax.numpy as jnp
    from legoesm import constants
    e = _eos()
    rho0 = e._NEMO_RHO0

    def rho_j(T, S, p):
        return e.nemo_roquet_eos(
            jnp.atleast_1d(T), jnp.atleast_1d(S), jnp.atleast_1d(p),
            coeffs=e._ROQUET_TEOS10, rho0=rho0)[0]

    # Corners plus interior of the real ocean envelope: polar-freezing to
    # tropical-surface, brackish-shelf to Red-Sea, surface to full depth.
    temps = (-2.0, 0.0, 10.0, 20.0, 30.0)
    sals = (30.0, 34.0, 35.0, 37.0, 40.0)
    depths = (0.0, 100.0, 1000.0, 5000.0)
    # jax.grad, NOT a central difference. Differencing this polynomial has a
    # truncation floor of ~1.3e-7 in alpha, the SAME size as the signal from a
    # mistyped coefficient (measured, job 9410697), so the differenced version
    # could not separate a real typo from its own error.
    d_dT = jax.grad(rho_j, argnums=0)
    d_dS = jax.grad(rho_j, argnums=1)
    worst_a = worst_b = 0.0
    worst_at = None
    for T0, S0, depth in itertools.product(temps, sals, depths):
        # Pressure built with the SAME rho0 the EOS inverts it with. Using
        # constants.rho_ocean (1025) against rho0 = 1026 shifts the recovered
        # depth by 0.0975% -- 4.9 m at 5000 m -- and that alone WAS the entire
        # apparent mismatch (job 9410799: floor 1.32e-7 -> 1.44e-13).
        p = rho0 * constants.g * depth
        alpha_fd = -float(d_dT(T0, S0, p)) / rho0
        beta_fd = float(d_dS(T0, S0, p)) / rho0
        alpha, beta = e.nemo_roquet_alpha_beta(
            jnp.array([T0]), jnp.array([S0]), jnp.array([depth]))
        ra = abs(float(alpha[0]) - alpha_fd)
        rb = abs(float(beta[0]) - beta_fd)
        if max(ra, rb) > max(worst_a, worst_b):
            worst_at = (T0, S0, depth, float(alpha[0]), alpha_fd,
                        float(beta[0]), beta_fd)
        worst_a, worst_b = max(worst_a, ra), max(worst_b, rb)

    # TOLERANCE CALIBRATED, NOT GUESSED. scripts/validate/ocean_fidelity/
    # teos10_coeff_sensitivity.py measures the floor at 1.44e-13 (alpha) and
    # 3.16e-12 (beta) over this envelope -- pure float64 round-off on a
    # 52-term polynomial. 1e-11 is 3.2x that, and a perturbation of ONE EOS
    # coefficient by 1 part in 1e4 is caught for 47 of the 52 coefficients.
    #
    # The 5 it cannot catch are EOS000/001/002/003 and EOS103: the constant
    # and pure-pressure terms have no T or S dependence, so they vanish under
    # d/dT and d/dS and NO derivative check can see them, by construction.
    # That is a property of the method, not slack in the tolerance.
    #
    # ABSOLUTE, not relative: alpha passes through ZERO near the temperature
    # of maximum density in cold fresh water, so a relative error divides by
    # ~0 there and one meaningless point would set the whole gate. Both
    # quantities are O(1e-4) in their own units.
    assert worst_a < 1.0e-11, f"alpha: worst |diff| {worst_a:.3e} at {worst_at}"
    assert worst_b < 1.0e-11, f"beta: worst |diff| {worst_b:.3e} at {worst_at}"


def test_density_stays_in_the_physical_range_over_that_envelope():
    """A transcription typo large enough to matter usually leaves the range.

    Derived, not remembered: seawater over -2..30 C, 30..40 g/kg, 0..5000 dbar
    spans roughly 1015-1070 kg/m3. This is a coarse tripwire, deliberately --
    its job is to fail loudly on a mistyped exponent, not to certify accuracy.
    That is what the derivative sweep above is for.
    """
    import itertools
    import jax.numpy as jnp
    from legoesm import constants
    e = _eos()
    lo, hi = 1e9, -1e9
    for T0, S0, depth in itertools.product(
            (-2.0, 10.0, 30.0), (30.0, 35.0, 40.0), (0.0, 1000.0, 5000.0)):
        r = float(e.nemo_roquet_eos(
            jnp.array([T0]), jnp.array([S0]),
            jnp.array([e._NEMO_RHO0 * constants.g * depth]),
            coeffs=e._ROQUET_TEOS10, rho0=e._NEMO_RHO0)[0])
        lo, hi = min(lo, r), max(hi, r)
    assert 1015.0 < lo < 1030.0, f"min density {lo:.3f} kg/m3 is unphysical"
    assert 1030.0 < hi < 1070.0, f"max density {hi:.3f} kg/m3 is unphysical"


class TestProductionWiring:
    """The TEOS-10 path must be REACHABLE from the card, not just importable.

    Codex 9408213 #6: compute_N2's nemo_bn2 branch called the bn2 helper
    without eos_form, so n2_mode="nemo_bn2" silently took the S-EOS branch and
    the whole port was dead code from production's point of view. These pin
    the chain card -> TKEConfig -> tke kernel -> compute_N2 -> bn2.
    """

    def test_card_selects_signed_teos10_stratification(self):
        import scripts.run.run_omip_core2 as core2
        cfg = core2.orca1_zdftke_config()
        assert cfg.n2_mode == "nemo_bn2", (
            "the card is clipping N2 again; NEMO's rn2 is signed and the "
            "clipped form is the whole mixing-length deficit (job 9407791)")
        assert cfg.n2_eos_form == "teos10", (
            "ORCA1 runs ln_teos10=.true.; setting n2_mode without "
            "n2_eos_form leaves the S-EOS alpha/beta in place")

    def test_compute_n2_forwards_the_eos_form(self):
        """The forward that was missing. Must CHANGE the answer, not just pass."""
        import numpy as np
        from legoesm.ocean.physics.vertical_mixing._shared import compute_N2
        nlev = 10
        T = jnp.asarray(np.linspace(16.0, 2.0, nlev))
        S = jnp.asarray(np.linspace(34.4, 34.9, nlev))
        gd = jnp.asarray(np.linspace(5.0, 400.0, nlev))
        gw = 0.5 * (gd[:-1] + gd[1:])
        common = dict(
            T_cell=T, S_cell=S, t_depth=gd, w_depth=gw,
            e3w_int=jnp.diff(gd), n2_mode="nemo_bn2")
        a = np.asarray(compute_N2(None, None, 1026.0, **common,
                                  n2_eos_form="seos"))
        b = np.asarray(compute_N2(None, None, 1026.0, **common,
                                  n2_eos_form="teos10"))
        assert not np.allclose(a, b), (
            "compute_N2 ignored n2_eos_form -- the forward is missing again "
            "and the TEOS-10 port is unreachable from production")

    def test_mpas_accepts_nemo_bn2_and_still_refuses_adiabatic(self):
        """MPAS shares the card, so it must run the same stratification.

        The old guard rejected every non-insitu mode citing a hydrostatic
        pressure requirement that only applies to 'adiabatic'. If MPAS cannot
        run what the card sets, the three-grid comparison measures the code.
        """
        import inspect
        from legoesm.ocean.physics.vertical_mixing import mpas_integration as m
        src = inspect.getsource(m)
        assert '("insitu", "nemo_bn2")' in src, (
            "the MPAS n2_mode guard no longer admits nemo_bn2; the card sets "
            "it, so MPAS would fail loud and the cross-grid arms could not run")
        assert "_bn2_ladder_kwargs" in src, (
            "MPAS stopped threading the depth ladders nemo_bn2 needs")
        assert "nemo_bn2_live_ladders" in src, (
            "MPAS reverted to STATIC depth ladders. NEMO evaluates bn2 on "
            "gdept_0*(1+eta/ht_0) under z*, and the C-grid path uses the live "
            "helper -- static ones agree only at eta=0, so the two grids would "
            "run different stratification and the cross-grid comparison would "
            "measure the code, not the physics (codex 9408814 #2).")
        assert "nemo_bn2_depth_ladders" not in src, (
            "the static ladder helper is back in the MPAS bridge")


class TestCrossGridAndEndToEnd:
    """The two gaps codex 9409347 named after the TEOS-10 wiring landed.

    #4: nothing proved the two grid bridges see the SAME N^2. That equality
    is the claim this whole fidelity branch exists to make, and it rested on
    reading two call sites rather than on a measurement.

    #5: nothing proved PRODUCTION consumes a negative bn2 as buoyancy
    production. The direct-EOS sign test would still pass if someone
    re-inserted a clip inside ``_shared.compute_N2``, which is the one place
    a clip would plausibly come back.
    """

    def test_ladders_are_live_not_static(self):
        """The MPAS bridge must CONSUME eta, not just be handed a ladder.

        My first attempt at this test called ``nemo_bn2_live_ladders`` twice
        with the same arguments and asserted the two results matched. That is
        trivially true and proves nothing about either bridge -- the exact
        cannot-fail pattern this file exists to avoid. The defect codex 9408814
        actually found was MPAS calling the STATIC ladder helper, which agrees
        with the live one only at eta = 0.

        So the discriminating measurement is the one the defect would fail:
        raise eta and require the diffusivity to MOVE. A static-ladder
        regression makes the two runs identical.
        """
        import numpy as np
        from legoesm.core.field import Field
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        from legoesm.ocean.physics.vertical_mixing.config import (
            TKEConfig, VerticalMixingConfig)
        from legoesm.ocean.physics.vertical_mixing.mpas_integration import (
            make_tke_profiles_mpas)
        from legoesm.ocean.vertical import create_ocean_z_star

        mesh = create_voronoi_mesh(subdivision_level=1)
        z = create_ocean_z_star(n_levels=6, H_max=4000.0)
        st = rest_state_mpas_ocean(
            mesh, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=4000.0, land_lat_threshold=85.0)
        pf = make_tke_profiles_mpas(VerticalMixingConfig(
            scheme="tke",
            tke=TKEConfig(n2_mode="nemo_bn2", n2_eos_form="teos10")))

        flat = pf(st, mesh, z)[1]
        # A large but physical z* excursion, so the stretch 1+eta/H_bathy is
        # unambiguously different from 1.
        # Field is a plain pytree class, not a NamedTuple -- rebuild it.
        raised = st._replace(eta=Field(
            data=jnp.full_like(st.eta.data, 40.0),
            name=st.eta.name, dims=st.eta.dims, units=st.eta.units))
        moved = pf(raised, mesh, z)[1]

        # CHALLENGED AND UPHELD. An adversarial review argued this
        # perturbation is two-variable, because eta also enters the z*
        # Jacobian and so moves dz_half by ~1% independently of the ladders.
        # Reasonable a priori, but the injection experiment already
        # discriminates it: the regression was injected by zeroing eta INSIDE
        # _bn2_ladder_kwargs ONLY, leaving the Jacobian at
        # mpas_integration.py:273 reading the real eta. Under that injection
        # the assertion below FAILED. If dz_half carried the signal it would
        # have passed. So the difference is attributable to the ladders.
        assert np.all(np.isfinite(np.asarray(moved)))
        assert not np.allclose(np.asarray(flat), np.asarray(moved)), (
            "eta did not change the MPAS diffusivity -- the bridge is on "
            "STATIC depth ladders, so it runs a different N^2 from the "
            "tripole path and the cross-grid comparison is invalid")

    def test_both_bridges_share_one_ladder_helper(self):
        """Cross-grid equality reduces to both bridges calling ONE helper.

        Asserting the helper equals itself is vacuous, so what is checked
        here is the thing that can actually drift: that neither bridge has
        grown its own ladder construction. Source-level, and deliberately
        narrow -- the execution check above is what proves MPAS uses eta.
        """
        import inspect
        from legoesm.ocean.physics.vertical_mixing import (
            k_profiles, mpas_integration)
        for mod in (k_profiles, mpas_integration):
            src = inspect.getsource(mod)
            assert "nemo_bn2_live_ladders" in src, (
                f"{mod.__name__} does not use the shared live-ladder helper")
            assert "nemo_bn2_depth_ladders(" not in src, (
                f"{mod.__name__} calls the STATIC ladder helper; under z* it "
                f"agrees with the live one only at eta = 0")

    def test_production_consumes_negative_bn2_as_buoyancy_production(self):
        """End-to-end: an inverted column must gain TKE from the signed N^2.

        Runs the real ``tke_vertical_mixing`` with the card's stratification
        against a control whose N^2 is clipped at zero. Signed must produce
        MORE TKE, because -K_H*N^2 > 0 is a source only when N^2 < 0. A clip
        re-inserted anywhere between the config and the kernel kills the
        difference and fails this test.
        """
        import numpy as np
        from legoesm.ocean.eos import nemo_bn2_live_ladders
        from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
        from legoesm.ocean.physics.vertical_mixing.tke import (
            tke_vertical_mixing)
        from legoesm.ocean.vertical import create_ocean_z_star

        nlev = 8
        z = create_ocean_z_star(n_levels=nlev, H_max=1000.0)
        # Statically UNSTABLE: cold and fresh over warm and salty.
        T = jnp.asarray(np.linspace(2.0, 20.0, nlev))[None, :]
        S = jnp.asarray(np.linspace(33.0, 36.0, nlev))[None, :]
        eta = jnp.zeros((1,))
        H = jnp.full((1,), 1000.0)
        t_depth, w_depth = nemo_bn2_live_ladders(z, eta, H)
        dz_half = jnp.asarray(np.diff(np.asarray(t_depth)[0]))[None, :]
        u = jnp.zeros((1, nlev))
        v = jnp.zeros((1, nlev))
        rho = jnp.full((1, nlev), 1026.0)

        def _run(cfg):
            # Positional order is (u, v, T, S, rho, dz_half, tke_old,
            # tau_x, tau_y, dt, cfg) -- T/S are ALREADY positional, so
            # passing T_cell=/S_cell= again is a duplicate-argument error.
            return tke_vertical_mixing(
                u, v, T, S, rho, dz_half,
                jnp.full((1, nlev - 1), 1e-4),
                None, None, 3600.0, cfg,
                t_depth=t_depth, w_depth=w_depth,
                e3w_int=jnp.diff(t_depth, axis=-1), dz_ref=z.dz_ref,
            )

        cfg = TKEConfig(prognostic=True, n2_mode="nemo_bn2",
                        n2_eos_form="teos10")
        signed = _run(cfg)
        assert np.all(np.isfinite(np.asarray(signed.tke_new)))

        # THE CLIPPED CONTROL. There is no n2_mode that means "nemo_bn2 with
        # a clip", and "insitu" is not a control (different N^2 formula AND
        # a clip -- two variables). So the control is built from the physics
        # instead: with CONSTANT T and S the bn2 assembly's dT and dS are
        # identically zero, so N^2 == 0 exactly. That is precisely what a
        # re-inserted max(N2, 0) would turn the unstable column into. One
        # variable (the T/S profile), and the comparison is
        # tolerance-independent: under a clip the two runs coincide.
        T_flat = jnp.full_like(T, 11.0)
        S_flat = jnp.full_like(S, 34.5)
        clipped = tke_vertical_mixing(
            u, v, T_flat, S_flat, rho, dz_half,
            jnp.full((1, nlev - 1), 1e-4),
            None, None, 3600.0, cfg,
            t_depth=t_depth, w_depth=w_depth,
            e3w_int=jnp.diff(t_depth, axis=-1), dz_ref=z.dz_ref,
        )

        n2 = _bn2_signed(T, S, t_depth, w_depth)
        n2_flat = _bn2_signed(T_flat, S_flat, t_depth, w_depth)
        assert float(np.min(np.asarray(n2))) < 0.0, (
            "the fixture is not statically unstable -- the test cannot fail "
            "for the reason it claims")
        assert np.allclose(np.asarray(n2_flat), 0.0, atol=1e-18), (
            "the control column is not neutral, so it is not a stand-in for "
            "the clipped answer")

        e_signed = float(np.max(np.asarray(signed.tke_new)))
        e_clipped = float(np.max(np.asarray(clipped.tke_new)))
        assert e_signed > e_clipped, (
            f"unstable column produced no more TKE than the neutral one "
            f"({e_signed:.4e} vs {e_clipped:.4e}) -- the negative bn2 is "
            f"not reaching the -K_H*N^2 buoyancy source, which is what a "
            f"clip re-inserted in _shared.compute_N2 would do")


def _bn2_signed(T, S, t_depth, w_depth):
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2
    return compute_buoyancy_frequency_nemo_bn2(
        T, S, t_depth, w_depth, eos_form="teos10",
        e3w_source="depth_difference")
