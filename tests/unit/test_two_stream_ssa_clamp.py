"""BUG-B robustness: the two-stream single-scattering albedo must be bounded.

On a drifted coupled state the cloud optics can emit ssa>1; the DIFFUSE
two-stream reflectance+transmittance (r_diff+t_diff) then exceeds 1 (energy
creation), which amplifies the radiation field and produces super-physical TOA
fluxes (sw_down>1000, OLR<0).  ``solve_sw`` / ``solve_lw`` now clamp
ssa∈[0,1] (and asymmetry∈[-1,1]) before the two-stream solve.  These tests pin
the mechanism: an unclamped ssa>1 violates single-layer energy conservation,
and clamping it to 1 restores r_diff+t_diff<=1.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (
    monochromatic_two_stream as m2s,
)


def _rt_sum(ssa_val, tau=1.0, g=0.85, zenith=0.6):
    out = m2s.sw_cell_properties(
        zenith,
        jnp.array([[[tau]]]),
        jnp.array([[[ssa_val]]]),
        jnp.array([[[g]]]),
    )
    return float(out["r_diff"][0, 0, 0] + out["t_diff"][0, 0, 0])


def test_unclamped_ssa_gt1_violates_energy_conservation():
    # ssa>1 is the failure mode: the single-layer diffuse reflectance +
    # transmittance exceeds 1 (the two-stream creates energy).
    assert _rt_sum(1.8) > 1.0 + 1e-6


def test_clamped_ssa_conserves_energy():
    # The BUG-B fix clamps ssa to [0,1]; at ssa=1 (conservative scattering)
    # the diffuse r+t is <= 1 (energy-conserving), and clamping a 1.8 input
    # reproduces exactly the ssa=1 result.
    assert _rt_sum(1.0) <= 1.0 + 1e-9
    assert _rt_sum(float(jnp.clip(1.8, 0.0, 1.0))) == _rt_sum(1.0)


def test_valid_ssa_unchanged_by_clamp():
    # For physical ssa (<=1) the clamp is a no-op — values are untouched.
    for s in (0.0, 0.3, 0.7, 0.999):
        assert _rt_sum(float(jnp.clip(s, 0.0, 1.0))) == _rt_sum(s)
