# Issue #1455 closure reconciliation

Date: 2026-08-31. Session: `01a04e34-d1fb-73e0-b25a-177641f0a246`.
Source branch: `origin/fidelity/dino-zdf-sweep-codex` at
`a2ca7e97c8555c6bc16a3399ff245a8bd21c1740`.

This is a receipt audit of all 15 boxes that remain unchecked in issue #1455.
`gh issue view 1455 --repo climate-federation/legoESM --json body` was attempted
three times on this host and failed each time with `error connecting to
api.github.com`. The box texts below are normalized from the preserved full
issue export/status sweeps and the request's confirmed count. The closure
comment must be compared once against the rendered live body before posting;
this audit does not claim a successful live API read.

The disposition vocabulary is strict:

- **DONE** means a committed receipt or measurement answers the box as written.
- **SUPERSEDED** means a later registered result or an explicit scope decision
  removes the premise; it does not mean the proposed work was run.
- **STILL-LIVE** means the question has no closing receipt. It moves intact to
  the successor tracker.

## Reconciliation

| # | unchecked box (normalized) | disposition | receipt and reasoning |
|---:|---|---|---|
| 1 | Take deep-southern-box `traldf_iso` / Redi to the bar. | **DONE** | The ordered Redi T/S walk closes temperature `zfu/zfv/zfw` and salinity `zfu/zfv` at the strict bar. Salinity `zfw` is explicitly cleared under Rule 1b as proven oracle arithmetic (`6.690652e-15`, below the frozen `2e-14` ceiling), not mislabeled AT-BAR. The ZDF/Asselin tail is also released. Receipt: `docs/ocean/fidelity/dino_split_explicit_momentum_chain_round93_result.md`, commit `7852bf22d031693b067f891cf63d224a5a4a2bde`, final artifact SHA-256 `1bac2857f8d0e78f5a51ae1834654773f539bbcac5a3093fe016d5bb5f2b486e`. |
| 2 | Decide whether the zero legoESM K33 budget bucket versus nonzero NEMO `zdf-zdfp` is an instrument failure or physics/wiring. | **STILL-LIVE** | The fp64 reruns rule out the original quantization explanation, and the production implicit K33/Redi coefficient path is now source-certified, but no current like-for-like budget split proves whether the zero *bucket* is absent physics or different accounting. The state record still labels this the parked K33 split/wiring hole (`docs/ocean/fidelity/dino_1226_state.md`, K33 entries around the deep-box budget). |
| 3 | Separate the advective-form mismatch from snapshot sampling in the `+0.404 W m-2` face-flux-versus-bucket residual. | **STILL-LIVE** | The committed state record says the residual is confounded by snapshot-versus-accumulated sampling and calls for a step-accumulated face-flux dump. The ordered per-step tracer walk does not answer that interval-budget question (`docs/ocean/fidelity/dino_1226_state.md`, `+0.404` table and caveat). |
| 4 | Reconcile the old “bolus 37% larger” statement with the y20 1–2% agreement. | **DONE** | The statement was retracted by its own originating lane: the maximum included inactive sub-seafloor garbage, the wet-mask result was `+3.8%`, and the comparison was between two legoESM grid modes rather than legoESM versus NEMO. The standing y20 delivery result remains 1–2%. Receipt: `docs/ocean/fidelity/dino_1226_state.md`, “OPEN CONTRADICTION” retraction; commit `8f99f22690a` records the reconciliation. |
| 5 | Re-walk MXL with the true `rn2` dump. | **DONE** | The true-`rn2` chain walk fixed the proxy harness and reduced MXL isolation `0.00992 -> 0.00069`; the later ordered ZDF walk certified the production MXL scan/composite, including exact `zmxlm/zmxld` closure after the operand fixes. Receipts: `docs/ocean/fidelity/dino_1226_state.md` (“MXL isolation”), and `docs/ocean/fidelity/dino_zdf_chain_sweep_result.md` rows 19–20. |
| 6 | Close the `zdftke` composite residual. | **DONE** | The ordered ZDF ladder certifies Richardson `zri/pdlr` at `0/9,920` with maximum `2.877796e-16`, closes the row-19/20 production composite, and records red controls. Receipt: `docs/ocean/fidelity/dino_zdf_chain_sweep_result.md`; the earlier shared Prandtl owner is commit `4aeeb867d`, and the current sweep result is carried by the zdf-sweep branch. |
| 7 | Disposition all remaining fidelity-gate DEBT rows as bias versus arithmetic/tail. | **DONE** | This was an accounting/disposition task, not a promise that the legacy 53-row gate would become all-green. A fresh run at the source SHA reports `AT BAR 15 | CEILING 5 | DEBT 26 | NEAR-CLASS 3 | UNMEASURED 3 | WAIVED 1` (53 total); those rows now have ordered-chain receipts, explicit ceiling/near-class/Rule-1b handling, or named residual registry status. The nonzero DEBT count is retained honestly. Fresh output: `/tmp/dino1455_fidelity_bar_gate.txt`; master accounting: `docs/ocean/fidelity/dino_full_step_coverage_round94_result.md`, commit `6dfdf66d88d8ef31fb7709d11564513869bcf45a`. |
| 8 | Close the nine uncovered `stp_MLF` calls. | **DONE** | The source-parsed registry supersedes the old partial call list: all 116 `stp_MLF` calls are enumerated, with 34 measured, 3 active UNMEASURED, 79 explicitly WAIVED, and **0 UNCOVERED**. Strict active coverage is `34/37 = 91.891892%`; complete accounting including waivers is `116/116 = 100%`. A synthetic unaccounted-call plant fails. Receipt: `docs/ocean/fidelity/dino_full_step_coverage_round94_result.md`, commit `6dfdf66d88d8ef31fb7709d11564513869bcf45a`, rebattery binding `7e7ad4df90dd3eb911c4ea6405f2aa9d6e73cd8f`. The three active UNMEASURED calls remain `ldf_dyn` coefficient isolation, `tra_sbc` RHS application, and `tra_qsr` two-band redistribution. |
| 9 | Explain the Rule-8 result in which the true 3-D thickness ladder tracked NEMO worse. | **STILL-LIVE** | The original global “7.4 Sv worse for 40 years” premise was retracted after the surface-placement repair (`dbb0977454f`, `78ba0632208`), but the post-fix, band-additive remeasurement found a narrower faithful-but-worse result: south `-1.484 -> -1.652 Sv`, channel `-0.146 -> +1.971 Sv`, and north `-0.110 -> -0.999 Sv`, while the apparent total improvement came from cross-band cancellation. No protocol-matched post-fix 40-year owner was measured. The obsolete global claim stays retracted; the localized compensation/restart question remains live (`docs/ocean/fidelity/dino_1226_state.md`, Rule-8 and southern-budget sections). |
| 10 | Resolve true-ladder restart-start instability. | **STILL-LIVE** | Four 90-day day-180 restart arms were stable (`0.633–0.635 m s-1` peak), so the broad “known broken” statement was retracted (commit `78ba0632208cff6ef2694718989163c4a8b76772`). That is not a root-cause receipt for the historical restart failure and does not prove stability at the former multi-year restart horizon. The narrow restart-protocol question remains open. |
| 11 | Distinguish dense-water formation failure from erosion/ventilation lag. | **STILL-LIVE** | No receipt observes the winter formation event at the historical from-rest y40 epoch. The state record explicitly leaves “erosion” versus “incomplete ventilation” unresolved (`docs/ocean/fidelity/dino_1226_state.md`, formation/ventilation section). The capstone T/S census establishes statistical distinction, not this causal direction. |
| 12 | Put `cfgs/DINO/MY_SRC` under version control and recover provenance for the vanished RUN_20Y binary. | **DONE** | Oracle commit chain begins at `491ac8d`; `MY_SRC` is version-controlled, the certified binary is preserved as `nemo.exe.certified_d3cf9242`, and `.binary_provenance.txt` binds it. The year-20 control reproduced all 1,520 restart variables across 16 tiles bit-for-bit and ACC exactly `142.8098167694`. Repo receipt: commit `ff07aa0364c` (“oracle provenance gap CLOSED”); the deterministic instrumented binary is separately SHA-256 pinned in the later ordered-chain receipts. This closes reproducibility without pretending the original pre-version-control tree can be reconstructed independently of its certified binary receipt. |
| 13 | Add sub-annual output capable of observing the winter dense-water formation event. | **STILL-LIVE** | The old y40 archives contain annual `T,S,eta,u,v,land_mask` only. The current 20-year capstone has its own registered snapshots/monthly series, but it starts from the matched twin epoch and does not reconstruct the historical from-rest y40 winter event. Receipt of absence: `docs/ocean/fidelity/dino_1226_state.md`, “no MLD, no sub-annual field.” |
| 14 | Run the 20-year post-capstone attribution hunt. | **SUPERSEDED** | The preregistered 20-year statistical-equivalence question has been answered `DISTINGUISHABLE_AT_20Y`; the user explicitly decided not to launch a second 20-year attribution hunt in this lane. Receipt: `docs/ocean/fidelity/PREREG_multi_year_climate_equivalence.md` plus score-fix commit `a2ca7e97c8555c6bc16a3399ff245a8bd21c1740`; score artifact SHA-256 `7ad72da798f754ff011c08af6f16cf44fdf4aa5aa7a682d458c9335c93c64bc0`. This is an explicit scope decision, not a claim of causal attribution. |
| 15 | Extend the historical integration from y40 to y60. | **STILL-LIVE** | The y40 restart exists, but no y60 arm or receipt exists. The capstone stops at the registered 20-year horizon and cannot be relabeled as this extension (`docs/ocean/fidelity/dino_1226_state.md`, y60 note). |

Disposition tally: **DONE 7 | SUPERSEDED 1 | STILL-LIVE 7 | total 15**.

## What closes, and what does not

The DINO headline closes. On current faithful defaults, the registered climate
rebattery reports day-360 basin transport gap `-0.0232 Sv` rather than the
historical `-0.9519 Sv` (improvement `+0.929 Sv`, 15 times the floor), southern
MLD RMS `1.04e-4 m` rather than `22.5 m`, and wall flicker
ratio/share `1.1997/0.1091`. Commit `7e7ad4df90dd3eb911c4ea6405f2aa9d6e73cd8f`
binds that battery. The 20-year capstone then gives the separate, longer-horizon
verdict **DISTINGUISHABLE_AT_20Y** for the `nemo_dino_kamm_mlf` twin bridge:
ACC and density contrast are UNRESOLVED, while row/basin transport, MLD, T/S
census, and variability REFUTE statistical indistinguishability. That verdict
does not undo the day-360 closure and does not make a standalone or cross-recipe
claim.

The registry closes its accounting contract, not every active strict row:
`116/116` calls are accounted for, `34/37` active calls are measured, and the
three active UNMEASURED calls are named. The legacy fidelity gate likewise
retains its nonzero DEBT count. Those qualifications must remain beside any
coverage number.

## STILL-LIVE rows for the successor tracker

These seven rows must move, without being silently checked off:

1. K33 deep-box budget split: current fp64 instrument versus physical/wiring
   ownership.
2. Advective-form versus flux-form `+0.404 W m-2`: accumulated face-flux dump
   to remove the snapshot-sampling confound.
3. Rule-8 faithful-ladder compensation: reconcile the post-fix band-localized
   channel/south/north response at a protocol-matched long horizon.
4. Historical true-thickness-ladder restart-start instability: reproduce or
   bound under an epoch-matched restart protocol.
5. Dense-water formation versus ventilation/erosion: winter-resolving causal
   diagnostic.
6. Sub-annual output for the historical from-rest formation epoch.
7. Optional y40-to-y60 extension, with its own horizon noise floor and frozen
   verdict bars.

The successor is the NEMO testcase ladder leading to ORCA1 OMIP at the DINO
bar, branch `fidelity/nemo-testcases-l1-codex`. Its coverage contract starts in
`docs/ocean/fidelity/testcases/nemo_testcases_l1_preregistered_dossier.md` and
the phase receipt/manifest matrix under `docs/ocean/fidelity/testcases/`.
Standalone-recipe and cross-recipe transfer remain explicit successor scope.

## Proposed closure comment for issue #1455

> The DINO tracker can now close, with its residual work transferred rather
> than erased. The current faithful-default rebattery confirms the original
> climate headline is repaired: at day 360 the basin transport gap is
> **-0.0232 Sv** versus the historical **-0.9519 Sv** (`+0.929 Sv`, 15x the
> floor), southern MLD RMS is **1.04e-4 m** versus **22.5 m**, and wall flicker
> is **1.1997 / 0.1091**. The separate preregistered 20-year capstone is honest
> about the horizon: **DISTINGUISHABLE_AT_20Y** for the
> `nemo_dino_kamm_mlf` twin bridge—ACC and density contrast UNRESOLVED;
> row/basin transport, MLD, T/S census, and variability REFUTE statistical
> indistinguishability. That is not a standalone or cross-recipe claim.
>
> | # | box | disposition | decisive receipt |
> |---:|---|---|---|
> | 1 | `traldf_iso` / Redi to bar | DONE | round-93 ordered T/S Redi receipt; S.zfw explicitly Rule-1b at `6.69e-15` |
> | 2 | K33 zero bucket: instrument or physics | STILL-LIVE | fp64 rules out quantization; budget split/wiring remains unowned |
> | 3 | `+0.404 W m-2` advective form vs sampling | STILL-LIVE | accumulated face-flux dump still absent |
> | 4 | “bolus 37% larger” contradiction | DONE | originating result retracted; wet-mask `+3.8%`, wrong comparison pair |
> | 5 | MXL true-`rn2` re-walk | DONE | `0.00992 -> 0.00069`; production scan later closed |
> | 6 | `zdftke` composite | DONE | `zri/pdlr` `0/9,920`, max `2.88e-16`; composite closed |
> | 7 | DEBT bias/tail disposition | DONE | fresh gate tally recorded, including 26 residual DEBT rows; not claimed zero |
> | 8 | nine uncovered calls | DONE | parsed registry: 116 calls, 0 UNCOVERED, 34/37 active measured, 79 waived |
> | 9 | Rule-8 ladder tracking-worse result | STILL-LIVE | old global premise retracted, but post-fix band-localized compensation remains unowned |
> | 10 | restart-start instability | STILL-LIVE | 90-day non-reproduction does not close the historical mechanism |
> | 11 | formation vs ventilation/erosion | STILL-LIVE | winter event remains unobserved |
> | 12 | `MY_SRC` and RUN_20Y provenance | DONE | `ff07aa0364c`; certified binary + 1,520-variable/16-tile identity control |
> | 13 | sub-annual formation output | STILL-LIVE | historical y40 archive remains annual-only |
> | 14 | 20-year attribution hunt | SUPERSEDED | capstone verdict recorded; follow-on hunt explicitly descoped |
> | 15 | y60 extension | STILL-LIVE | no y60 arm exists |
>
> Full receipts and qualifications are in
> `docs/ocean/fidelity/dino_1455_closure_reconciliation.md`. Coverage is
> **34/37 = 91.891892%** for active calls and **116/116 = 100%** for accounting
> with 79 explicit waivers; the three active UNMEASURED calls are `ldf_dyn`
> coefficient isolation, `tra_sbc` RHS application, and `tra_qsr` two-band
> redistribution. The seven STILL-LIVE research rows are transferred to the
> successor campaign, not waived.
>
> Successor: NEMO testcase ladder -> ORCA1 OMIP at the DINO bar, branch
> `fidelity/nemo-testcases-l1-codex`, with the coverage matrix in
> `docs/ocean/fidelity/testcases/` (starting from
> `nemo_testcases_l1_preregistered_dossier.md`). That tracker owns the explicit
> standalone/cross-recipe transfer that this DINO twin-bridge closure does not
> claim.
