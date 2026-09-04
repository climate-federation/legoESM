"""#1521: is the droplet-size distribution's SHAPE parameter evaluated at the
GRID-MEAN droplet number where the fit it comes from expects the IN-CLOUD one?


The effective radius comes from the Morrison/SAM gamma PSD,

    r_eff = (pgam + 3) / (2 * lamc) ,
    lamc  = ( cons26 * nc_permass * (pgam+1)(pgam+2)(pgam+3) / q_c ) ** (1/3) ,
    pgam  = f( nc_cm3 )        [Martin et al. fit]

`lamc` depends on droplet number and cloud water only through the RATIO nc/q_c,
so if both are grid-mean the ratio is the in-cloud one and that part is right.
`pgam` is NOT a ratio: it is a function of the droplet CONCENTRATION alone, and
the Martin fit is defined on the in-cloud concentration.  On the prognostic-Nc
branch the code passes the grid-mean number straight in.

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
