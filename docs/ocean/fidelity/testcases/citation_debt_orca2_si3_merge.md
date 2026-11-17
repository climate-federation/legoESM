# Known citation debt in the ORCA2 + SI3 merge

Review found this debt on 2026-11-17. By user decision, the citations are
recorded here and are not fixed in this merge. No receipt or citation-gate map
entry is changed by this ledger.

| # | cited path:span left in place | what it should cite on this tree | receipt:line | cited span pinned by gate map? |
|---:|---|---|---|:---:|
| 1 | `ocean_model_latlon_cgrid.py:6813-6911` | WZV's full-step `eta_before`/`eta_after` operands, `ocean_model_latlon_cgrid.py:8880-8907` | `nemo_testcases_l2_gyre_phase3_round8_receipt.md:2602` | yes |
| 2 | `ocean_model_latlon_cgrid.py:14638` | the top-stress divisor and division, `ocean_model_latlon_cgrid.py:18993-19000` | `nemo_testcases_l4_orca2_round315_differentiability_receipt.md:188` | yes |
| 3 | `ocean_model_latlon_cgrid.py:2174` | the vertical flux-divergence division, `ocean_model_latlon_cgrid.py:2172` | `nemo_testcases_l4_orca2_round315_differentiability_receipt.md:190` | yes |
| 4 | `ocean_model_latlon_cgrid.py:2160` | the horizontal flux-divergence divisor and division, `ocean_model_latlon_cgrid.py:2158-2159` | `nemo_testcases_l4_orca2_round315_differentiability_receipt.md:191` | yes |
| 5 | `ocean_model_latlon_cgrid.py:2160` | the horizontal flux-divergence divisor and division, `ocean_model_latlon_cgrid.py:2158-2159` | `nemo_testcases_l4_orca2_round315_differentiability_receipt.md:197` | yes |
| 6 | `ocean_model_latlon_cgrid.py:2174` | the vertical flux-divergence division, `ocean_model_latlon_cgrid.py:2172` | `nemo_testcases_l4_orca2_round315_differentiability_receipt.md:197` | yes |
| 7 | `ocean_model_latlon_cgrid.py:2160` | the horizontal flux-divergence divisor and division, `ocean_model_latlon_cgrid.py:2158-2159` | `nemo_testcases_l4_orca2_round315_differentiability_receipt.md:238` | yes |
| 8 | `ocean_model_latlon_cgrid.py:2174` | the vertical flux-divergence division, `ocean_model_latlon_cgrid.py:2172` | `nemo_testcases_l4_orca2_round315_differentiability_receipt.md:238` | yes |
| 9 | `ocean_model_latlon_cgrid.py:14638` | the top-stress divisor and division, `ocean_model_latlon_cgrid.py:18993-19000` | `nemo_testcases_l4_orca2_round315_differentiability_receipt.md:242` | yes |
| 10 | `ocean_pe_latlon_cgrid.py:4294-4298` | the five drag-config arguments, `ocean_pe_latlon_cgrid.py:4481-4485` | `nemo_testcases_l4_orca2_round315_differentiability_receipt.md:249` | no |
| 11 | `ocean_pe_latlon_cgrid.py:4375-4379` | the generic path's five drag-config arguments, `ocean_pe_latlon_cgrid.py:4566-4570` | `nemo_testcases_l4_orca2_round315_differentiability_receipt.md:251` | no |
| 12 | `ocean_model_latlon_cgrid.py:2160` | the horizontal flux-divergence divisor and division, `ocean_model_latlon_cgrid.py:2158-2159` | `nemo_testcases_l4_orca2_round316_gradient_guards_receipt.md:25` | yes |
| 13 | `ocean_model_latlon_cgrid.py:2174` | the vertical flux-divergence division, `ocean_model_latlon_cgrid.py:2172` | `nemo_testcases_l4_orca2_round316_gradient_guards_receipt.md:25` | yes |
| 14 | `ocean_model_latlon_cgrid.py:2930-2932` | the active source-rounded QCO combine, `ocean_model_latlon_cgrid.py:2925-2927` | `nemo_testcases_l4_orca2_round316_gradient_guards_receipt.md:26` | yes |
| 15 | `ocean_model_latlon_cgrid.py:2938` | the mask for that QCO combine, `ocean_model_latlon_cgrid.py:2933` | `nemo_testcases_l4_orca2_round316_gradient_guards_receipt.md:26` | yes |
| 16 | `ocean_model_latlon_cgrid.py:14638` | the top-stress divisor and division, `ocean_model_latlon_cgrid.py:18993-19000` | `nemo_testcases_l4_orca2_round316_gradient_guards_receipt.md:28` | yes |
| 17 | `ocean_pe_latlon_cgrid.py:4294-4298` | the five drag-config arguments, `ocean_pe_latlon_cgrid.py:4481-4485` | `nemo_testcases_l4_orca2_round316_gradient_guards_receipt.md:31` | no |
| 18 | `ocean_pe_latlon_cgrid.py:4375-4379` | the generic path's five drag-config arguments, `ocean_pe_latlon_cgrid.py:4566-4570` | `nemo_testcases_l4_orca2_round316_gradient_guards_receipt.md:31` | no |
| 19 | `ocean_model_latlon_cgrid.py:11770-11774` | the traced drag-config arguments in the TKE path, `ocean_model_latlon_cgrid.py:12708-12712` | `nemo_testcases_l4_orca2_round319_traced_drag_receipt.md:28` | no |
| 20 | `ocean_model_latlon_cgrid.py:11770` | the first traced drag-config argument, `ocean_model_latlon_cgrid.py:12708` | `nemo_testcases_l4_orca2_round319_traced_drag_receipt.md:53` | no |
| 21 | `nemo_testcase_recipe.py:236-238` | the GYRE Decision-36 shear selectors, `nemo_testcase_recipe.py:638-644` | `nemo_testcases_l2_gyre_decision36_nemo_face_shear_receipt.md:39` | no |
| 22 | `ocean_model_latlon_cgrid.py:9181-9198` | the static `now2` selection of step-entry eta, `ocean_model_latlon_cgrid.py:13196-13208` | `nemo_testcases_l2_gyre_decision36_nemo_face_shear_receipt.md:40` | no |
| 23 | `ocean_model_latlon_cgrid.py:9079-9152` | the leapfrog arm's carried before-level requirements, `ocean_model_latlon_cgrid.py:13132-13140,13203-13208` | `nemo_testcases_l2_gyre_decision36_nemo_face_shear_receipt.md:40` | no |

Debt rows: **23**.
