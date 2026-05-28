# Phase G.0c — legoESM-Veros ACC tier-2 tendency comparison

Source recipe: ``legoesm.ocean.fidelity.veros_acc_recipe.build_acc_recipe``.
Veros snapshot: ``veros.setups.acc.acc.ACCSetup`` via
``legoesm.ocean.fidelity.veros_runner.run_veros``.

Constants override: ``override_constants(**VEROS_CONSTANTS)``

## legoESM probe summary

- ``pgf_ke_u``: shape (44, 31, 15), min=-3.103e-03, max=1.975e-07, L1=1.940e-05
- ``pgf_ke_v``: shape (45, 30, 15), min=-2.269e-03, max=2.269e-03, L1=3.437e-05
- ``coriolis_u``: shape (44, 31, 15), min=-4.243e-06, max=1.056e-06, L1=1.447e-07
- ``coriolis_v``: shape (45, 30, 15), min=-4.473e-06, max=3.937e-06, L1=1.864e-07
- ``vortcor_u``: shape (44, 31, 15), min=-6.043e-09, max=7.133e-09, L1=7.090e-11
- ``vortcor_v``: shape (45, 30, 15), min=-6.588e-09, max=6.002e-09, L1=6.151e-11
- ``vertadv_u``: shape (44, 31, 15), min=-8.513e-09, max=1.015e-08, L1=6.428e-11
- ``vertadv_v``: shape (45, 30, 15), min=-8.909e-09, max=1.067e-08, L1=8.288e-11
- ``ah_lap_u``: shape (44, 31, 15), min=-2.870e-07, max=3.176e-07, L1=4.908e-09
- ``ah_lap_v``: shape (45, 30, 15), min=-3.396e-07, max=4.475e-07, L1=4.725e-09
- ``bh_bilap_u``: shape (44, 31, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``bh_bilap_v``: shape (45, 30, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``botdrag_u``: shape (44, 31, 15), min=-1.411e-10, max=1.229e-10, L1=3.087e-12
- ``botdrag_v``: shape (45, 30, 15), min=-1.881e-10, max=6.796e-11, L1=1.597e-12
- ``av_vert_u``: shape (44, 31, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``av_vert_v``: shape (45, 30, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``phys_u``: shape (44, 31, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``phys_v``: shape (45, 30, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``total_u``: shape (44, 31, 15), min=-3.103e-03, max=3.174e-07, L1=1.940e-05
- ``total_v``: shape (45, 30, 15), min=-2.269e-03, max=2.269e-03, L1=3.437e-05
- ``dT_dt_total``: shape (44, 30, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``dS_dt_total``: shape (44, 30, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``rho``: shape (44, 30, 15), min=9.973e+02, max=1.034e+03, L1=1.025e+03

## Per-region per-process metrics (legoESM vs Veros)

| Veros field | mapped to legoESM | interior L2 | boundary L2 | equator L2 | ML L2 | abyssal L2 | pattern corr (interior) | sign-match (interior) |
|---|---|---|---|---|---|---|---|---|
| coriolis_u | coriolis_u (shape mismatch / aggregate; deferred) | — | — | — | — | — | — | — |
| coriolis_v | coriolis_v (shape mismatch / aggregate; deferred) | — | — | — | — | — | — | — |
| veros_du_adv | (aggregate) (shape mismatch / aggregate; deferred) | — | — | — | — | — | — | — |
| veros_dv_adv | (aggregate) (shape mismatch / aggregate; deferred) | — | — | — | — | — | — | — |
| veros_du_mix | av_vert_u (shape mismatch / aggregate; deferred) | — | — | — | — | — | — | — |
| veros_dv_mix | av_vert_v (shape mismatch / aggregate; deferred) | — | — | — | — | — | — | — |
| veros_dT_hmix | (aggregate) (shape mismatch / aggregate; deferred) | — | — | — | — | — | — | — |
| veros_dT_vmix | (aggregate) (shape mismatch / aggregate; deferred) | — | — | — | — | — | — | — |
| veros_dT_iso | (aggregate) (shape mismatch / aggregate; deferred) | — | — | — | — | — | — | — |
| veros_dS_hmix | (aggregate) (shape mismatch / aggregate; deferred) | — | — | — | — | — | — | — |
| veros_dS_vmix | (aggregate) (shape mismatch / aggregate; deferred) | — | — | — | — | — | — | — |
| veros_dS_iso | (aggregate) (shape mismatch / aggregate; deferred) | — | — | — | — | — | — | — |
| rho | rho | 5.099e+00 | 1.682e+01 | 4.418e+00 | 7.033e+00 | nan | 0.6768 | 1.0000 |
