# Snow soot probe — round 2 (final instrument + conclusion), 2026-11-22

Branch probe/snow-soot-snicar-2026-11-22 (worktree /glade/derecho/scratch/pg2328/snow_soot_wt), commits
6ff395f471..416f19b1de. Round-1 fixes applied (see commit 6ff395f471 message). Since then: soil node = lower
layer T (vote a/a/a); snow temperatures now a LINEAR profile from the surface T our age clock sees
(t1 = Ts + g dz1/2, t2 = Ts + g(dz1 + dz2/2)); CLM's diagnosed top gradient is then g/2 (reported: 10, 5, 0 K/m).

Result (git 416f19b1de, ifx 2025.2.1; CESM2 CMIP6 historical.001 MAM 1979-83, 45-70N land, FSNO>=0.9,
landfrac>=0.5, cell-month H2OSNO<=1000; 28557 cell-months):
- history loading: BC top 110 ppb, lower 43 ppb; dust top 9.6 ppm, lower 2.7 ppm; SWE 209, top 5.3 kg/m2;
  FSDS 143.6 W/m2; BC phi fraction 0.82; dust bins [0.012, 0, 0.988, 0] (file bins 2,4 are zero).
- history proxy snow albedo (diagnostic): VIS dir/dif 0.916/0.913, NIR 0.588/0.666, broadband 0.774.
- oracle spring mean (days 61-105, cosz 0.45, history SW weights): clean bb 0.818 (VIS 0.972/0.967, NIR 0.624/0.697);
  dirty 0.793 (VIS 0.932/0.919, NIR 0.619/0.691); BC-only 0.797; dust-only 0.812; all-phi 0.792; all-pho 0.798;
  cosz 0.3/0.6 dirty 0.802/0.784. r_top 140 -> 278 um.
- ours (production scalars, 150 d, 5000 K; spectrally flat): spring mean 0.755 (model-internal broadband incl.
  2% UV at 0.05: 0.741). Day 0: ours 0.808, SNICAR clean 0.841, dirty 0.819. Day 105: ours 0.729, dirty 0.767.
- darkening clean-dirty 0.026 (3.7 W/m2 equiv); gap ours-dirty -0.037 (-5.4 W/m2 equiv).
- Pre-registered verdict: INCONCLUSIVE (darkening < 0.03; |gap| > 0.02, sign opposite to the hypothesis).

Proposed conclusion: (1) on matched synthetic snowfall-free histories, our bare snow albedo is DARKER than
CLM5.0 SNICAR with CLM5.0's own spring BC+dust, at every day and every cosz tested, so the missing soot/dust
term does not make our snow brighter than CAM6's; (2) CLM5.0's soot+dust darkening is ~0.026 broadband
(mostly BC, VIS); (3) ours is spectrally flat: VIS ~0.18 too dark, NIR ~0.07-0.13 too bright vs SNICAR;
(4) the 1979 TOA-albedo excess over Siberia/Canada must therefore come from elsewhere (cover fraction /
canopy masking / fresh-snow frequency) — PLAUSIBLE, not measured here.

Review: (A) any remaining instrument defect that would change sign or move numbers > 0.01; (B) is each
conclusion line supported, overstated, or wrong? One line each. Do not edit files.
