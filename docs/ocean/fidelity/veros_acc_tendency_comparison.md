# Phase G.0c — legoESM-Veros ACC tier-2 tendency comparison

Source recipe: ``legoesm.ocean.fidelity.veros_acc_recipe.build_acc_recipe``.
Veros snapshot: ``veros.setups.acc.acc.ACCSetup`` via
``legoesm.ocean.fidelity.veros_runner.run_veros``.

Constants: pinned via config (``ConstantsConfig`` / ``VEROS_CONSTANTS_CONFIG``).

## legoESM probe summary

- ``pgf_ke_u``: shape (44, 31, 15), min=-5.312e-08, max=1.955e-07, L1=1.812e-09
- ``pgf_ke_v``: shape (45, 30, 15), min=-1.356e-07, max=1.805e-07, L1=3.209e-08
- ``coriolis_u``: shape (44, 31, 15), min=-3.855e-06, max=9.892e-07, L1=1.326e-07
- ``coriolis_v``: shape (45, 30, 15), min=-4.239e-06, max=3.626e-06, L1=1.754e-07
- ``vortcor_u``: shape (44, 31, 15), min=-1.082e-08, max=2.052e-08, L1=9.696e-11
- ``vortcor_v``: shape (45, 30, 15), min=-1.596e-08, max=1.881e-08, L1=1.458e-10
- ``vertadv_u``: shape (44, 31, 15), min=-8.454e-09, max=6.624e-09, L1=6.280e-11
- ``vertadv_v``: shape (45, 30, 15), min=-1.731e-08, max=1.758e-08, L1=8.762e-11
- ``ah_lap_u``: shape (44, 31, 15), min=-2.829e-07, max=4.636e-07, L1=3.939e-09
- ``ah_lap_v``: shape (45, 30, 15), min=-2.055e-07, max=1.956e-07, L1=3.464e-09
- ``bh_bilap_u``: shape (44, 31, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``bh_bilap_v``: shape (45, 30, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``botdrag_u``: shape (44, 31, 15), min=-1.411e-10, max=1.229e-10, L1=3.076e-12
- ``botdrag_v``: shape (45, 30, 15), min=-1.881e-10, max=6.796e-11, L1=1.562e-12
- ``av_vert_u``: shape (44, 31, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``av_vert_v``: shape (45, 30, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``phys_u``: shape (44, 31, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``phys_v``: shape (45, 30, 15), min=0.000e+00, max=0.000e+00, L1=0.000e+00
- ``total_u``: shape (44, 31, 15), min=-2.710e-07, max=4.667e-07, L1=5.419e-09
- ``total_v``: shape (45, 30, 15), min=-2.091e-07, max=1.924e-07, L1=3.380e-08
- ``dT_dt_total``: shape (44, 30, 15), min=-3.557e-09, max=3.860e-09, L1=1.206e-10
- ``dS_dt_total``: shape (44, 30, 15), min=-2.540e-20, max=2.574e-20, L1=1.288e-21
- ``dT_gm_redi``: shape (44, 30, 15), min=-3.557e-09, max=3.860e-09, L1=1.206e-10
- ``dS_gm_redi``: shape (44, 30, 15), min=-2.540e-20, max=2.574e-20, L1=1.288e-21
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
| rho | rho | 3.756e-02 | 3.758e-02 | 3.754e-02 | 2.344e-02 | nan | 1.0000 | 1.0000 |

## Per-process MOMENTUM comparison at cell centres (Q2)

legoESM components aggregated to Veros groupings (du_adv=vortcor+vertadv, du_mix=av_vert+botdrag, du_cor 1:1), both interpolated to cell centres. The recipe now uses flux-form momentum advection (adopted, matching Veros). Observed (developed flow, 864000 s): du_cor/dv_cor corr ~0.96/0.98 (strong); du_adv/dv_adv corr ~0.65/0.71 (MODERATE, R²≈0.43/0.51). The advection residual is a per-process MAPPING floor — legoESM's vector-invariant decomposition (vortcor+vertadv) does not align cleanly with Veros's flux-form ∇·(uu) split, so the NET momentum tendency agrees better than any single per-process row; a total-tendency / energy-budget check (roadmap T2-3) side-steps this. NOT a bug, but NOT a clean per-process match either — see the strategy doc §8 ledger. `wsign` = sign-match over cells with `|veros| > 0.1·max` (dynamically significant; the plain sign-match is near-zero-cell noise for these processes).

| process | interior L2 | interior corr | interior sign | interior wsign |
|---|---|---|---|---|
| coriolis_u | 1.497e-07 | 0.9590 | 0.9168 | 0.9810 |
| coriolis_v | 1.137e-07 | 0.9764 | 0.9774 | 0.9818 |
| du_adv | 4.896e-10 | 0.6537 | 0.7006 | 0.9328 |
| dv_adv | 5.126e-10 | 0.7148 | 0.8015 | 0.8716 |
| du_mix | 7.157e-07 | 0.0011 | 0.0651 | 0.0000 |
| dv_mix | 3.616e-07 | 0.0077 | 0.0636 | 0.0000 |

## Per-process TRACER comparison at cell centres (Q2)

legoESM per-process tracer tendencies isolated by differencing probe runs (full - scheme-off); tracers are cell-centred so no interpolation is needed. iso = GM/Redi isoneutral, vmix = vertical mixing. vmix is IMPLICIT in the ACC recipe, so legoESM produces no explicit vmix tendency (documented delta — see the strategy doc §8 ledger).

| process | interior L2 | interior corr | interior sign |
|---|---|---|---|
| T_iso | 1.523e-09 | 0.1710 | 0.9451 |
| S_iso | 8.541e-20 | 0.0452 | 0.7872 |
