# ORCA2-DECKS round 12 receipt — rung 1 admitted, rung-0 handoff stopped

Date: 2026-10-02

Disposition: **STOPPED_FOR_DECISION**.  The independent rung-1 oracle record is
admitted, completing the side lane's requested NEMO decks and records for
rungs 1 through 10.  The rung-1 to main-rung-0 boundary is not a one-module
damping edge: it also changes the active horizontal shape of the background
tracer diffusivity.  This side lane makes no configuration choice for it.

Base: `5b80c1b970`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round12.md`, commit `f1f8a1328`,
before rerunning the admission, reading record payloads, or measuring the
cross-lane background-diffusivity signal.  Every run claim is labelled
**independent**: NEMO starts from that rung's own from-rest initialization.

## Rung inventory

| rung | NEMO deck | independent record | status |
|---:|---|---|---|
| 10 | shipped one-category ocean-ice deck | 480 frames + 240-step month | **ADMITTED** round 2 |
| 9 | rung 10 without ice | 480 frames + 240-step month | **ADMITTED** round 3 |
| 8 | rung 9 without DDM, river-mouth diffusivity, differential T/S mixing | 480 frames + 240-step month | **ADMITTED** round 4 |
| 7 | rung 8 without internal-wave mixing/background reset | 480 frames + 240-step month | **ADMITTED** round 5 |
| 6 | rung 7 with constant mixing replacing TKE | 480 frames + 240-step month | **ADMITTED** round 6 |
| 5 | rung 6 without runoff | 480 version-2 frames + 240-step month | **ADMITTED** round 8 |
| 4 | rung 5 without penetrative chlorophyll shortwave | 480 version-2 frames + 240-step month | **ADMITTED** round 9 |
| 3 | rung 4 with exact-zero surface fluxes and no restoring/freshwater budget | 480 version-2 frames + 240-step month | **ADMITTED** round 10 |
| 2 | rung 3 without GM eddy-induced velocity or mixed-layer eddies | 480 version-2 frames + 240-step month | **ADMITTED** round 11 |
| 1 | rung 2 without BBL or geothermal heating | 480 version-2 frames + 240-step month | **ADMITTED** this round |
| 0 | main-lane-owned dynamics core | main-lane record | **OUT OF SIDE-LANE SCOPE; boundary unresolved** |

## Rung-1 admission

The operator's round-11 launcher completed at producer commit `5b80c1b970`.
This round reran its admit-only path against the existing target.  All twenty
named plants printed `STATUS PLANT-FIRED`; the clean gate then printed
`PASS_RUNG1_RECORD`.  The admitted record contains:

- 480/480 self-describing frames, 3,840 finite PRESENT fields, and 960 ABSENT
  fields, exactly the owner-off runoff fields in every frame;
- an fp64 exact-zero five-field surface-flux file;
- eight finite T/U/V/W month files with 86 floating variables;
- two finite fp64 step-240 ocean restart shards and no ice product; and
- a complete 549-regular-file SHA-256 inventory.

The resolved output proves BBL and geothermal heating are both off, the active
BBL prints are absent, the geothermal input is unread, and every inherited
rung-2 through rung-10 predicate still passes.  HD12-P1 and HD12-P2 are
**CONFIRMED**.

## Complete rung-1 versus rung-0 classification

The mechanically parsed 21-assignment diff is completely classified.  The
four restart rows are protocol-only, the five file operands name two separately
generated but bit-equal zero-flux files, and the two damping rows are the
intended module boundary.

| class | count | assignments | disposition |
|---|---:|---|---|
| restart protocol | 4 | `ln_rst_list`, `nn_itend`, `nn_stock`, `nn_stocklist` | protocol, not physics |
| zero-file name | 5 | `sn_emp`, `sn_qsr`, `sn_qtot`, `sn_utau`, `sn_vtau` | inputs are independently pinned exact zero |
| intended damping boundary | 2 | `ln_tradmp`, `ln_tsd_dmp` | rung-1 module removal |
| owner-off | 3 | `ln_spc_dyn`, `ln_sssr_bnd`, `nn_chldta` | no compiled/active consumer in either deck |
| explicit reference default | 6 | `ln_zdfric`, `ln_zdfgls`, `ln_zdfosm`, `ln_zdfnpc`, `ln_zdfmfc`, `ln_zdfswm` | absent in rung 1 equals explicit false in rung 0 |
| active extra | 1 | `nn_havtb` | **configuration decision required** |

The owner-off result is source-ordered.  The reused binary has no `key_agrif`
(`cpp_ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE.fcm:1`) and the gate finds no
`ln_spc_dyn` symbol in any compiled source.  Both decks resolve `ln_ssr=false`,
so NEMO skips the restoring initializer and runtime call
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:365-369`,
`:517-521`).  Both resolve `ln_traqsr=false`, so NEMO skips the shortwave
initializer and stage call
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/nemogcm.f90:428-434`,
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:523-529`).
The six absent vertical-mixing selectors are all false in the pinned reference
namelist (`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/EXP00/namelist_ref:1180-1201`).
HD12-P4 is **CONFIRMED**.

`nn_havtb` is different.  The compiled initializer reads it with the constant-
mixing parameters, makes the background tracer diffusivity one tenth of its
nominal value in the equatorial band, and writes that shaped background into
every level (`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:205-228`).
Rung 1 resolves `nn_havtb=1`; main rung 0 resolves `0`.  On their bit-identical
rank-0 mesh, the compiled conditions affect 2,590 wet surface cells, reach the
factor 0.1, and change the nominal `1.2e-5 m2/s` background by as much as
`1.08e-5 m2/s`.  The gate's zero-signal plant fires.  HD12-P3 is **CONFIRMED**.

## Decision

The existing instructions conflict at this boundary.  The side-lane rule keeps
every unnamed switch at the shipped deck value, which is `nn_havtb=1`; the
main-lane rung 0 uses the GYRE-physics value `0`.  Changing either is a
configuration choice and is not authorized by Decision 80.

Recommended resolution: set `nn_havtb=0` on rungs 1 through 6 and make its
activation part of rung 7's shipped TKE settings.  That preserves rung 0's
"ORCA2 geometry with GYRE physics" definition and keeps the rung-1 damping edge
pure, but it overrides note B23 and requires new records for rungs 1 through 6.

## Gates, review, and scope

- Round-11 admission: all 20 plants fire; clean status `PASS_RUNG1_RECORD`.
- Round-12 handoff gate: all five plants fire; clean status
  `PASS_RUNG1_RECORD_ACTIVE_RUNG0_DIFFERENCE`; focused controls **6 passed**.
- The round receipt citation gate passes 7 citations with zero failures and
  zero unmapped citations; its shifted `zdfphy` span plant fires.  The
  cumulative default-receipt gate also passes with zero failures or unmapped
  citations.
- Hierarchy rounds 1 through 12 plus citation controls: **119 passed, 2 failed**
  because the citation-map edit made the worktree-stamp ratchet fire.  After
  committing that edit, exactly those two IDs passed in isolation and the
  final clean focused battery passed **121/121**.
- The one allowed `tests/ocean/fidelity -n 12` battery selected 2,281 tests and
  reached 99%, then entered the same late-suite no-output stall as rounds 10
  and 11 and was interrupted.  Thirty-four failure markers were visible, but
  xdist emitted no IDs before the missing summary; they are therefore
  **UNATTRIBUTED**, not labelled known reds, and no second broad battery ran.
- The required separate review command returned before reading the diff:
  `failed to initialize in-process app-server client: Read-only file system`.
  Verdict: **independent review unavailable in-sandbox**.
- No file under `packages/` or `src/` changed.  GYRE is byte-identical by
  construction; its year gate was not rerun.  No NEMO source or record changed.

Search-before-build: the existing round-11 semantic diff and namelist parser
were reused; no second parser was added.  ASKED choices: all constructed rungs
and module removals follow Decisions 79/80 and note B23.  UNASKED choices:
none.

## Prediction ledger

| prediction | status |
|---|---|
| HD12-P1 rung-1 record | **CONFIRMED**; clean admission over the complete existing record |
| HD12-P2 admission non-vacuity | **CONFIRMED**; all twenty plants fire |
| HD12-P3 active cross-lane difference | **CONFIRMED**; `nn_havtb` changes 2,590 wet rank-0 surface cells by up to `1.08e-5 m2/s` |
| HD12-P4 remaining cross-lane differences | **CONFIRMED**; three owner-off and six reference-default assignments |

## OPEN

1. User decision: keep main rung 0 at `nn_havtb=0` and move rungs 1 through 6
   to zero (recommended), or change main rung 0 to the shipped value 1.
2. After that decision, the owning lane regenerates every affected deck and
   record; no acquisition is requested before the choice.
3. Rung-0 card construction and cross-model scoring remain main-lane work.

## UNVERIFIED

- The broad ocean-fidelity battery did not produce a terminal summary, so its
  thirty-four visible failure markers have no test IDs and were not isolated.
- No post-decision deck or oracle record exists.
