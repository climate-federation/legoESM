"""Vertical profiles of the faithful IFS convection chain vs the legacy Bechtold
reduction on the frozen deep sounding: cloud-base mass flux, heating and
moistening against pressure.

Single column, so this is the cross-section that exists before the scheme runs
in a model: it shows WHICH LAYERS each scheme moves, which a column-integrated
number hides.  Usage:
    python scripts/plot/plot_ifs_faithful_profiles.py [out.png]
"""
import sys
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "tests/unit")
from test_ifs_faithful import _deep_column, _kwargs          # noqa: E402
from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection  # noqa: E402
from legoesm.atmosphere.physics.convection.config import BechtoldConfig  # noqa: E402

out_path = sys.argv[1] if len(sys.argv) > 1 else "ifs_faithful_profiles.png"

T, q, p_full, p_half = _deep_column()
ncol, nlev = T.shape
kw = _kwargs(ncol, nlev)
kw["dq_dt_dyn"] = jnp.full((ncol, nlev), 1e-8)      # a real sub-cloud supply
rad = jnp.full((ncol, nlev), -1.5 / 86400.0)

runs = {}
for label, cfg in (("legacy Bechtold", BechtoldConfig()),
                   ("faithful IFS chain", BechtoldConfig(use_ifs_ascent=True))):
    out, M_u, _ = bechtold_convection(T, q, p_full, p_half, config=cfg,
                                      dT_dt_rad=rad, **kw)
    runs[label] = (np.asarray(out.dT_dt[0]) * 86400.0,
                   np.asarray(out.dq_v_dt[0]) * 86400.0 * 1e3,
                   np.asarray(M_u[0]))

p = np.asarray(p_full[0]) / 100.0
fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True)
colors = {"legacy Bechtold": "0.45", "faithful IFS chain": "#c1440e"}

axes[0].plot(np.asarray(T[0]), p, color="#26456e")
axes[0].set_xlabel("T [K]")
axes[0].set_title("environment")
ax_q = axes[0].twiny()
ax_q.plot(np.asarray(q[0]) * 1e3, p, color="#1b7a5a", ls="--")
ax_q.set_xlabel("q [g/kg]", color="#1b7a5a")

for label, (dT, dq, mu) in runs.items():
    axes[1].plot(dT, p, color=colors[label], label=label, lw=2)
    axes[2].plot(dq, p, color=colors[label], label=label, lw=2)
    axes[3].plot(mu, p, color=colors[label], label=label, lw=2)

axes[1].set_xlabel("dT/dt [K/day]"); axes[1].set_title("convective heating")
axes[2].set_xlabel("dq/dt [g/kg/day]"); axes[2].set_title("convective moistening")
axes[3].set_xlabel("M_u [kg m$^{-2}$ s$^{-1}$]"); axes[3].set_title("updraught mass flux")
for ax in axes[1:]:
    ax.axvline(0.0, color="0.8", lw=0.8)
    ax.legend(fontsize=8, loc="lower right")
axes[0].invert_yaxis()
axes[0].set_ylabel("pressure [hPa]")
fig.suptitle("Faithful IFS convection chain vs legacy Bechtold — frozen deep sounding, "
             f"{nlev} levels, dt {kw['dt']:g} s", fontsize=11)
fig.tight_layout()
fig.savefig(out_path, dpi=130)
print("wrote", out_path)
for label, (dT, dq, mu) in runs.items():
    print(f"{label:20s} peak heating {dT[np.argmax(np.abs(dT))]:+9.2f} K/day   "
          f"peak M_u {mu.max():.4f}")
