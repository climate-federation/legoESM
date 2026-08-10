"""Field-level ownership between legacy AIMIP leaves and the spec route.

The class-level subtraction excluded EVERY spec parameter of a class the
legacy route touches, so spec-only fields (Sundqvist ``qc_crit``, McFarlane
``fcrit2``, most of ``CloudConfig``, EDMF entirely under the old key
arithmetic) trained nowhere. Ownership is now per-FIELD: the spec collector
excludes exactly the fields the legacy ``to_*_config`` methods write.
"""
import pytest

pytest.importorskip("jax")


@pytest.fixture(scope="module")
def owned():
    from legoesm.training.aimip_params import aimip_legacy_owned_fields
    return aimip_legacy_owned_fields()


def test_legacy_written_fields_are_owned(owned):
    # One sentinel per legacy family (values written by to_*_config).
    for q in (
        "atm.turb.LouisConfig.l_mix_max",
        "atm.turb.SurfaceLayerConfig.z0",
        "atm.gwd.McFarlaneConfig.G_0",
        "atm.clouds.CloudConfig.rh_crit",
        "atm.micro.SundqvistConfig.auto_rate",
        "atm.conv.TiedtkeConfig.tau_M_u_relax",
    ):
        assert q in owned, q


def test_spec_only_fields_are_free(owned):
    # Fields the legacy route never writes must NOT be excluded — these are
    # exactly the ones the class-level rule wrongly suppressed.
    for q in (
        "atm.micro.SundqvistConfig.qc_crit",
        "atm.gwd.McFarlaneConfig.fcrit2",
        "atm.conv.ConvectiveEDMFConfig.tau_a",
        "atm.clouds.CloudConfig.conv_cloud_coeff",
    ):
        assert q not in owned, q


def test_collector_excluding_owned_is_disjoint_and_nonempty(owned):
    from legoesm.training.aimip_params import aimip_scheme_keys_for
    from legoesm.training.param_collector import build_trainable_params

    keys = aimip_scheme_keys_for(
        convection="edmf", turbulence="louis", gwd="mcfarlane",
        microphysics="sundqvist", radiation="rrtmgp", cloud="xu_randall")
    assert "atm.clouds.CloudConfig" in keys  # the explicit clouds routing
    assert "atm.conv.ConvectiveEDMFConfig" in keys

    p = build_trainable_params(
        active_scheme_keys=keys, tier="extended",
        exclude=tuple(sorted(owned)))
    collected = {
        f"{k}.{f}" for k, v in p.to_overrides().items() for f in v}
    assert collected, "extended tier collected nothing"
    assert not (collected & owned), sorted(collected & owned)
    # EDMF must be trainable now (it trained NOTHING before this change).
    assert any(q.startswith("atm.conv.ConvectiveEDMFConfig.")
               for q in collected)


def test_cloud_ownership_conditional_on_xu_randall():
    """Under a non-Xu cloud scheme the factory builds CloudConfig from
    defaults, so its fields (rh_crit above all) must stay collectable —
    the sundqvist-cloud swap arm's primary control (codex)."""
    from legoesm.training.aimip_params import aimip_legacy_owned_fields

    owned_sq = aimip_legacy_owned_fields(cloud_scheme="sundqvist")
    assert "atm.clouds.CloudConfig.rh_crit" not in owned_sq
    owned_xr = aimip_legacy_owned_fields(cloud_scheme="xu_randall")
    assert "atm.clouds.CloudConfig.rh_crit" in owned_xr


def test_nonvacuous_without_exclude_the_overlap_exists(owned):
    """Partner check: dropping the exclusion recreates the collision the
    ownership rule prevents — proves the exclusion is load-bearing."""
    from legoesm.training.aimip_params import aimip_scheme_keys_for
    from legoesm.training.param_collector import build_trainable_params

    keys = aimip_scheme_keys_for(
        convection="edmf", turbulence="louis", gwd="mcfarlane",
        microphysics="sundqvist", radiation="rrtmgp", cloud="xu_randall")
    p = build_trainable_params(active_scheme_keys=keys, tier="extended")
    collected = {
        f"{k}.{f}" for k, v in p.to_overrides().items() for f in v}
    assert collected & owned, "exclusion is vacuous — nothing would collide"
