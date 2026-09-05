"""#1521: is the droplet-size distribution's SHAPE parameter evaluated at the
GRID-MEAN droplet number where the fit it comes from expects the IN-CLOUD one?


The effective radius comes from the Morrison/SAM gamma PSD,

    r_eff = (pgam + 3) / (2 * lamc) ,
    lamc  = ( cons26 * nc_permass * (pgam+1)(pgam+2)(pgam+3) / q_c ) ** (1/3) ,
    pgam  = f( nc_cm3 )        [Martin et al. fit]

`lamc` carries the droplet number TWICE: once through the factor
`nc_permass / q_c`, where a common cloud-fraction dilution cancels so grid-mean
inputs give the in-cloud ratio, and once inside `pgam(nc_cm3)`, where it does
NOT -- `pgam` is a function of the CONCENTRATION alone, and the Martin fit is
defined on the in-cloud concentration while the prognostic-Nc branch passes the
grid-mean number straight in.  (An earlier revision of this file called `lamc`
ratio-only. That was wrong and codex caught it; the conclusion is unchanged,
since the dilution still reaches the radius only through `pgam`.)

CONTROLLED COMPARISON, one variable = whether the number handed to pgam is
diluted by cloud fraction.  The in-cloud state (q_c_in, N_c_in) is held FIXED;
only the representation changes:

    arm A (what the model does): q_c = cf*q_c_in , N_c = cf*N_c_in
    arm B (in-cloud):            q_c =    q_c_in , N_c =    N_c_in

Both go through the SAME `compute_cloud_properties`, so the ratio isolates the
pgam bias.  At cf = 1 the two arms are identical by construction, which is the
built-in control.

Prints numbers only.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    compute_cloud_properties)
from legoesm.atmosphere.physics.clouds.config import CloudConfig

# Production values (config/amip/amip_production.yaml): clouds: sundqvist,
# microphysics: morrison, cloud_q_c_diagnostic: 5.0e-6.
CFG = CloudConfig(scheme="sundqvist", rh_crit=0.85, q_c_diagnostic=5.0e-6)
NCOL, NLEV = 1, 1
T = jnp.full((NCOL, NLEV), 283.0)
P = jnp.full((NCOL, NLEV), 85000.0)
QV = jnp.full((NCOL, NLEV), 6.0e-3)
DP = jnp.full((NCOL, NLEV), 2500.0)


def reff(q_c, n_c_per_m3, cf):
    out = compute_cloud_properties(
        T, P, QV, DP, CFG,
        q_cloud=jnp.full((NCOL, NLEV), q_c),
        q_ice=jnp.zeros((NCOL, NLEV)),
        n_cloud=jnp.full((NCOL, NLEV), n_c_per_m3),
        cloud_fraction_override=jnp.full((NCOL, NLEV), cf),
    )
    return float(np.asarray(out.r_eff_liq)[0, 0]) * 1e6   # microns


def arms(q_c_in, n_c_in, cf):
    """(r_eff as the model computes it, r_eff with the in-cloud number)."""
    return reff(q_c_in * cf, n_c_in * cf, cf), reff(q_c_in, n_c_in, cf)


def worst_over_reachable_box():
    """Largest bias over the REACHABLE (q_c, N_c, cf) box, not two archetypes.

    Quoting a bound from two hand-picked columns says nothing about the states
    the model actually visits (codex review, HIGH).  This sweeps the range a
    warm cloud can occupy -- in-cloud water 0.02 to 2.0 g/kg, in-cloud droplet
    number 20 to 1000 cm^-3, cover 0.05 to 1 -- and returns the worst case
    found, so the bound is over a space rather than over two points.

    It is still NOT the radiatively weighted bias in a run: that needs the
    model's own joint distribution of (q_c, N_c, cf), which only output can
    supply.  What this bounds is how large the effect can get anywhere in the
    box, which is the bound the negative result actually needs.
    """
    worst, where = 0.0, None
    for q_c in (2.0e-5, 1.0e-4, 3.0e-4, 5.0e-4, 1.0e-3, 2.0e-3):
        for n_c in (20.0e6, 50.0e6, 100.0e6, 300.0e6, 600.0e6, 1000.0e6):
            for cf in (1.0, 0.9, 0.7, 0.5, 0.3, 0.15, 0.05):
                a, b = arms(q_c, n_c, cf)
                if abs(b - a) > worst:
                    worst, where = abs(b - a), (q_c, n_c, cf, a, b)
    return worst, where


def main():
    print("in-cloud q_c = 0.30 g/kg, in-cloud N_c = 100 cm^-3 (marine stratocumulus)")
    print(f"{'cf':>5s} {'r_eff grid-mean [um]':>21s} {'r_eff in-cloud [um]':>20s} "
          f"{'ratio':>7s} {'bias [um]':>10s}")
    for label, qc, nc in (("", 3.0e-4, 100.0e6),
                          ("in-cloud q_c = 0.50 g/kg, in-cloud N_c = 300 cm^-3"
                           " (polluted continental)", 5.0e-4, 300.0e6)):
        if label:
            print()
            print(label)
        for cf in (1.0, 0.9, 0.7, 0.5, 0.3, 0.1):
            a, b = arms(qc, nc, cf)
            print(f"{cf:5.2f} {a:21.3f} {b:20.3f} {a / b:7.4f} {a - b:10.3f}")
    print()
    worst, (q_c, n_c, cf, a, b) = worst_over_reachable_box()
    print("worst bias over the reachable box "
          "[q_c 0.02-2.0 g/kg, N_c 20-1000 cm^-3, cf 0.05-1]:")
    print(f"  {worst:.3f} um at q_c={q_c * 1e3:.3f} g/kg, "
          f"N_c={n_c / 1e6:.0f} cm^-3, cf={cf:.2f} "
          f"(grid-mean {a:.3f} um vs in-cloud {b:.3f} um)")
    print("NB the pgam clip binds at low diluted concentrations, so this is the "
          "bias the MODEL carries -- smaller than the unclipped Martin-fit "
          "bias would be, and it is the model's that matters here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
