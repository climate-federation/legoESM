F3’s low-density floor is fixed: `maximum(rho_air, 0.1)` matches Morrison’s floor behavior, including the genuine 15 hPa regime. No new unit, sign, conservation, differentiability, or MPAS-wiring issue found.

Minor precision caveat only: Morrison is passed moist/virtual-temperature density via [`compute_rho`](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/_shared.py:191), while the seed uses dry `p/(R_d T)`. Above the floor this makes the seed cap slightly tighter (~`0.608 q_v`), not looser; at the low-density floor they are exactly identical. This is not substantive.

OVERALL VERDICT: no substantive findings
