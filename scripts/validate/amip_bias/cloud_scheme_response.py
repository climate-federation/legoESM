"""Cloud cover vs RH and temperature: Sundqvist against Xu-Randall.

Sundqvist is a pure RH closure with a hard onset at rh_crit; Xu-Randall also
reads the CONDENSATE, so an ice-bearing layer can be cloudy below rh_crit. This
prints both on the same axes before any scheme is swapped.
"""
import os, sys, glob
sys.path[:0] = [os.getcwd()+"/src"] + glob.glob(os.getcwd()+"/packages/*") + [os.getcwd()]
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice

RH_GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
CASES = [("Arctic winter low", 230.0, 900e2, 2.0e-5),
         ("mixed-phase stratus", 258.0, 800e2, 5.0e-5),
         ("anvil / TTL", 215.0, 200e2, 2.0e-5),
         ("warm marine", 285.0, 950e2, 1.0e-4)]

def cover(scheme, T, p, rh, q_i, sat="mixed_phase"):
    Tj = jnp.array([[T]]); pj = jnp.array([[p]]); dp = jnp.array([[5e3]])
    # RH is defined against the curve the scheme uses, so set q_v from it
    qsl = float(saturation_mixing_ratio(Tj, pj)[0, 0])
    qsi = float(saturation_mixing_ratio_ice(Tj, pj)[0, 0])
    qs = qsi if (sat == "mixed_phase" and T < 233.15) else qsl
    q_v = jnp.array([[rh * qs]])
    cfg = CloudConfig(scheme=scheme, rh_crit=0.85, saturation_scheme=sat,
                      q_c_diagnostic=5e-6)
    kw = dict(q_ice=jnp.array([[q_i]]), q_cloud=jnp.zeros((1, 1)))
    out = compute_cloud_properties(Tj, pj, q_v, dp, cfg, **kw)
    return float(out.cloud_fraction[0, 0])

for name, T, p, q_i in CASES:
    print(f"\n{name}: T = {T:.0f} K, p = {p/100:.0f} hPa, ice = {q_i*1e6:.0f} mg/kg")
    print("   RH   " + "".join(f"{r:8.2f}" for r in RH_GRID))
    for scheme in ("sundqvist", "xu_randall"):
        row = [cover(scheme, T, p, r, q_i) for r in RH_GRID]
        print(f"  {scheme:11s}" + "".join(f"{v:8.3f}" for v in row))
    # what the ice alone buys at LOW RH, which is the mixed-phase question
    dry = 0.70
    a = cover("sundqvist", T, p, dry, q_i); b = cover("xu_randall", T, p, dry, q_i)
    b0 = cover("xu_randall", T, p, dry, 0.0)
    print(f"  at RH {dry}: sundqvist {a:.3f} | xu_randall {b:.3f} "
          f"(with no ice: {b0:.3f})")
