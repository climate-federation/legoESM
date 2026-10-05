"""Sub-layer refinement of the IFS convection trigger: what the conservative
reconstruction changed.

Left: the humidity cross-section the refined departure search actually sees --
the L30 parent layer means, the finite-volume minmod reconstruction at r = 4,
and the analytic profile the fixture is built from.
Right: diagnosed cloud top against effective vertical resolution, for the
refinement before and after the fix and for native grids.

Usage: plot_ifs_refinement.py OUT.png
"""
import sys, os
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests", "unit"))
import test_ifs_test_ascent as tref                      # the frozen fixtures
from legoesm.atmosphere.physics.convection import _ifs_test_ascent as ta
from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import compute_moist_adiabat
from legoesm.thermo import saturation_specific_humidity

out_path = sys.argv[1] if len(sys.argv) > 1 else "ifs_refinement.png"
R = 4
NLEV = 30

T30, q30, p_full30, p_half30 = tref.column("deep", nlev=NLEV)
ph = jnp.asarray(p_half30)
p_half_r = np.asarray(ta._refine_half_levels(ph, R))
p_full_r = 0.5 * (p_half_r[:, 1:] + p_half_r[:, :-1])
c_par = 0.5 * (ph[:, :-1] + ph[:, 1:])
h_dp = ph[:, 1:] - ph[:, :-1]
k_par = jnp.repeat(jnp.arange(NLEV), R)
q_recon = np.asarray(ta._fv_minmod_refine(jnp.asarray(q30), c_par, h_dp,
                                          jnp.asarray(p_full_r), k_par))[0]


def analytic(p, ps=101300.0):
    T0, q0 = 300.0, 0.017
    rcpl = constants.R_d / constants.c_pd
    Tad = np.asarray(compute_moist_adiabat(jnp.array([T0]), jnp.array(p)[None, :],
                                           jnp.array([q0])))[0]
    T = np.where(p > 95000, T0 * (p / ps) ** rcpl, Tad - 1.0)
    T = np.where(p < 15000, np.maximum(T, 200.0), T)
    qs = np.asarray(saturation_specific_humidity(jnp.array(T), jnp.array(p)))
    return np.minimum(np.where(p > 95000, q0, 0.8 * qs), q0)


p_fine = np.linspace(4.0e4, 1.01e5, 400)
q_fine = analytic(p_fine)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 6.2))

ax.plot(q_fine * 1e3, p_fine / 100.0, color="0.55", lw=1.4,
        label="analytic profile the fixture samples")
ax.step(np.asarray(q30)[0] * 1e3, np.asarray(p_full30)[0] / 100.0, where="mid",
        color="C0", lw=1.6, label="L30 parent layer means")
ax.plot(q_recon * 1e3, p_full_r[0] / 100.0, color="C3", lw=1.2, marker=".",
        ms=3, label=f"conservative reconstruction, r = {R}")
ax.set_xlabel("specific humidity [g/kg]")
ax.set_ylabel("pressure [hPa]")
ax.set_ylim(1010.0, 400.0)
ax.set_title("what the refined search sees")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25)

# measured cloud tops (hPa) against effective level count
eff = np.array([30, 60, 120, 240])
before = np.array([894.8, 557.2, 523.4, 489.6])
after = np.array([894.8, 658.5, 624.7, 590.9])
nat_eff = np.array([30, 60, 120])
native = np.array([894.8, 616.2, 536.1])
bx.plot(eff, before, "o--", color="0.6", label="refinement, clipped ramp (before)")
bx.plot(eff, after, "o-", color="C3", label="refinement, conservative (after)")
bx.plot(nat_eff, native, "s-", color="C0", label="native grid")
bx.set_xscale("log", base=2)
bx.set_xticks(eff)
bx.set_xticklabels([str(int(e)) for e in eff])
bx.set_xlabel("effective vertical levels")
bx.set_ylabel("diagnosed cloud top [hPa]")
bx.invert_yaxis()
bx.set_title("cloud top against resolution")
bx.legend(fontsize=8, loc="lower left")
bx.grid(alpha=0.25)

fig.suptitle("IFS trigger sub-layer refinement: conservative reconstruction, "
             "frozen deep sounding", fontsize=11)
fig.tight_layout()
fig.savefig(out_path, dpi=130)
print("wrote", out_path)
print(f"parent-layer water conservation: max error "
      f"{100 * np.max(np.abs(np.add.reduceat(q_recon * np.diff(p_half_r[0]), np.arange(0, NLEV * R, R)) - np.asarray(q30)[0] * np.diff(np.asarray(p_half30)[0])) / np.maximum(np.abs(np.asarray(q30)[0] * np.diff(np.asarray(p_half30)[0])), 1e-30)):.5f}%")
