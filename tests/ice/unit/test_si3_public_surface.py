"""What ``legoesm.ice`` exports for the two SI3 lanes."""
import legoesm.ice as ice


def test_kept_jpl1_ridging_arm_stays_public():
    for name in ("apply_si3_jpl1_ridging", "SI3JPL1RidgingConfig", "SI3JPL1RidgingLosses", "SI3JPL1RidgingState"):
        assert name in ice.__all__ and hasattr(ice, name)


def test_no_builder_for_a_selector_the_step_refuses():
    assert not hasattr(ice, "build_si3_orca2_o4_config")
    assert hasattr(ice, "si3_o4_step")
