# NEMO-testcases L2 GYRE round 67: stage-3 LDF order receipt

Date: 2026-09-12. Final disposition: **HELD / REFUTED; no production physics
change remains**. Oracle producer `3b3b045bd9e03b60330204e7590e4c4470b7a0ca`;
132 admitted fields, including all 63 Krhs splits (43 exact, 20 changed).

## Source order

| Boundary | legoESM parent | compiled NEMO GYRE |
|---|---|---|
| SBC and QSR | Builds the stage-3 physical rate and replaces the Kbb QSR by its Kmm evaluation (`ocean_model_latlon_cgrid.py:6277-6346`). | Krhs already contains advection/SBC; stage 3 calls QSR (`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:917-965`). |
| LDF | Computes GM/Redi and adds `dt*dT_gm/dS_gm` to `T_mid/S_mid` in concentration form (`ocean_model_latlon_cgrid.py:7082-7514`). | Calls `tra_ldf` after QSR and before ZDF (`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:917-965`). |
| Content | The final WS call restarts from Kbb/saved stage 2 (`ocean_model_latlon_cgrid.py:7813-7851`); its content is advection content plus only `stage_source_rates[2]` (`ocean_model_latlon_cgrid.py:1862-1912`). The earlier LDF concentration update is therefore replaced, not post-solve. | ZDF forms `h(Kbb)*T(Kbb)+p2dt*h(Kmm)*Krhs` (`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565`). |
| Implicit ZDF | K33 is added to the tracer matrix (`ocean_model_latlon_cgrid.py:10188-10194`), then the captured content is passed to the literal solve (`ocean_model_latlon_cgrid.py:8212-8324`, `ocean_model_latlon_cgrid.py:10612-10633`). | `avt/avs + ah_wslp2` is assembled first (`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:416-443`), then the tridiagonal matrix (`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:463-479`) and content/solve. |

Thus moving the already-computed LDF rate into pre-solve Krhs is NEMO's order,
not a tuning choice. The final tree deliberately retains the parent behavior
because the frozen exact landing gate below failed.

## Substitution result and owner

The clean pre-edit report is `round67_ldf_order_before.json` (SHA-256
`5934bb4bc1f19d87e6f9d8e356a0d2398ac962e84e6d7c03c96f69e09770c8d9`).
Its all-oracle rebuild is bit-exact and its one-ULP content plant fires on one
cell.

| arm, T maximum | Krhs (K/s) | content (K m) |
|---|---:|---:|
| routed live | 6.198289884215e-11 | 5.954039670542e-5 |
| NEMO post-SBC + live QSR/LDF | 3.901914601258e-12 | 5.767504376308e-7 |
| NEMO post-QSR + live LDF | 3.901917581490e-12 | 5.767508639565e-7 |
| NEMO post-LDF | 0 | 1.359694579151e-10 (Kmm association) |

The isolated component maxima are advection `6.196948294061e-11`, SBC
`1.052042188174e-12`, QSR `1.168556066120e-16`, and LDF
`3.901917581501e-12` K/s. Complete FCT advection therefore owns the remaining
`5.954e-5`; K33 cannot own a pre-solve difference, and surface, solar, LDF,
and thickness are measured smaller. The preregistered phrase that post-SBC
“does not remove the floor” is **REFUTED** because that cumulative boundary
also replaces advection.

## Candidate, frozen gate, and Rule 12

Routing alone improved production T content `1.679392691671e-3 ->
5.954039670542e-5` (28.2x). Its kt3 T/S became
`8.916073106491e-7/7.235656340754e-8`, but missed the exact override through
association rounding. The universal NEMO-form Krhs association then produced
T/S `8.916073070964e-7/7.235655630211e-8`, yet its complete metric rows still
differed from the frozen recomputed-FCT override (T: 17,998 versus 17,999
unequal wet cells; RMS differs `1.18e-17`). Report
`round67_ldf_order_after.json` is status **REFUTED**, SHA-256
`986f11aa44d3cca2af9d9ea03cbd80fd0d5eb18ecab669a2e24c51582e484e75`.

| card | statement/scope | Rule-12 disposition |
|---|---|---|
| GYRE | executes LDF + shared WS content | **HELD** at exact kt3 gate; kt1--10/day 1--30 after arms unreached |
| LOCK_EXCHANGE | same WS helper; NEMO LDF OFF and legoESM `gm_redi=None` (`lock_kt1_10/ocean.output:578`, `nemo_testcase_recipe.py:576-663`) | association trajectory unreached |
| OVERFLOW | same; LDF OFF/`gm_redi=None` (`overflow_kt1_10/ocean.output:690`, `nemo_testcase_recipe.py:576-663`) | association trajectory unreached |
| DINO | separate modified-leapfrog program | source-inert; execution gate unreached |
| ORCA2 | topology/record absent | **UNMEASURED WITH SPEC**: resolve native card, record stage-3 post-SBC/QSR/LDF Krhs and pre/post-ZDF T/S at kt1--10, run this cumulative gate plus trajectory comparison; require exact statement replay, registered moves, and no earlier first-over-bar |

Decision-36 after-arm day-30 baseline is T/S RMS
`1.239756827231e-2/2.195296277634e-3`; no after value is claimed because the
landing gate stopped first. The final source diff against round-66 tip is empty.

## Review, tests, and disposition

Pre/post self-review: HOLD was enforced and both physics commits were reversed.
The requested `codex exec --sandbox read-only` was invoked twice pre-code: the
first failed to initialize read-only state; an isolated-state retry ended
`ERROR: Reconnecting... waiting for network` and produced no verdict. The final
post-diff invocation on clean commit `16be2d8fe287` also timed out with exit
124. Its terminal result, verbatim, was **“ERROR: Reconnecting... waiting for
network”**. There is no `VERDICT:` line to quote; the separate-review
requirement is **UNMET/BLOCKED**, and absent a verdict is not approval. Full
terminal output is `postdiff_codex_review.log` in the round-67 evidence root.

Focused tests: round-66/67 instrument tests 5 passed; helper splits 3 passed
and 4 passed; the final citation/provenance/stamp/instrument batch passed
44/44 after adding the missing worktree stamp to the pre-existing round-63
admission report. Python compile and diff checks passed. The candidate one-ULP
plant exits nonzero. GYRE/tank/day-30 trajectory gates are **UNREACHED**, not
passed.

Writable-clone history includes instruments, both candidate physics commits,
and `3ebdc93d7082` withdrawing them. The original checkout cannot commit and
retains only the documentation/instrument/citation changes listed in the final
handoff; its production ocean source is identical to `86c15c256496`.

ASKED/UNASKED: no model choice was encountered or made; no card field was
added. Open question: acquire a pre-edit prediction from the production
stage-3 FCT divergence itself (not the separately recomputed divergence), then
preregister its exact content/kt3 row before reconsidering the source-derived
route. The complete-FCT advection debt remains next.
