# GRIDLADDER-VMIX L3 round 20 receipt — ORCA2 eddy-induced velocity

Status: **ACQUISITION_NEEDED**.  L3 is preregistered, implemented, selected,
and preflighted.  This round did not build or run NEMO, acquire a record,
score a trajectory, or make a certification claim.

The base import is merge `b3e5011190` with certified lane-A G9 parent
`7a31b67ef9`; it retains the lane-B L2 parent at `51b2582f5e`.  The frozen
prediction contract is `698ee4d7d8`.  Implementation is `55ef11b7b3`, with
preflight-control corrections `5f41de9724` and `fbc77d660c`.  Lane A's
arithmetic was merged, not reimplemented.

## ORCA2 survey

The complete resolved `namtra_eiv` group has exactly five keys.  ORCA2 states
`ln_ldfeiv=T` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:317`, `nn_aei_ijk_t=21` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:319`, `rn_Ue=0.03 m/s` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:322`, and `rn_Le=200000 m` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:323`.  The only unresolved member is
`ln_eke_equ`, whose reference value is false at
`cfgs/SHARED/namelist_ref:980`.  The compiled namelist declaration contains
those five and no others at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:572-574`; NEMO reads reference first
and configuration second at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:583-584`.

The DINO/ORCA2 compiled key set is `key_si3 key_qco key_vco_1d3d key_RK3`, as
pinned by `cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm:1`.  The running arm is
therefore the RK3 arm quoted below.  EIV is rejected without laplacian
isoneutral or triad diffusion at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:611-612`; L3 supplies the already
certified L2 standard-isoneutral selection.

The initialization computes `zUfac=0.5*rn_Ue` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:622` and
`aei0=zUfac*rn_Le` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:624`, hence the ORCA2 cap is exactly
`3000 m2/s`.  Selection 21 is the time-varying two-dimensional case at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:658` and sets the time-dependent flag
at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:663`.

`ldf_eiv` is the compiled Treguier coefficient routine at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:707`.  Its non-triad arm thickness-
integrates stratification and isoneutral slope magnitude at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:758-760`, bounds the Rossby radius to
`2–40 km` at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:767`, forms the growth-
rate coefficient at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:769`, applies the
tropical taper and `aei0` cap at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:775-776`, and averages the coefficient
independently to U and V faces at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:780-781`.  The once-per-step slope is
built at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:189`; selection 21 then
recomputes the time-varying coefficient at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:193`.

The RK3 transport routine begins at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:892`, constructs the U/V bolus
streamfunctions at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:947-950`, and adds
their vertical differences to `pFu/pFv` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:954-955`.  `tra_adv` invokes that
transport only for EIV with the non-triad scheme at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:255-257`.  Thus the bolus reaches the
existing G9 FCT transport and does not enter momentum or continuity.

No extra EIV input file is active: selection 21 derives the coefficient from
the live stratification, slopes, metrics, Coriolis field, and masks named in
the cited routine.  `ln_eke_equ=F` also leaves the GEOMETRIC EKE input family
inactive.  legoESM already had one GM/Redi implementation, its Treguier
coefficient, NEMO face averaging, and bolus-through-FCT transport; L3 extends
that path with RK3 stage-3 routing and adds no duplicate physics operator.

## L3 card and controls

`ORCA2L3-eiv` is exactly certified G9 tracer advection plus certified L2
isoneutral diffusion plus the five-key EIV group.  The reader refuses every
single-key plant.  The selector requires the complete composition:
Treguier-21, the `3000 m2/s` cap with zero non-NEMO floor, NEMO U/V face
averaging, NEMO left-fold vertical reduction, source-ordered square root, and
bolus-through-FCT.  Stages 1 and 2 retain G9 transport; only stage 3 receives
the existing bolus transport before the FCT limiter.

The focused composition run reports **19 passed in 27.81 s**.  The full L3
file reports **16 passed**, including a traced production step.  Against the
pre-L3 commit, normalized one-step jaxprs are identical for all four named
hard controls:

- G9: 176859 lines, SHA-256
  `f7ef15ccd26baece4d929817134f614dc76e2df26a82d45016324b1800addb3d`;
- L2: 195287 lines, SHA-256
  `3ab35572c486b79cbe71626a439efc8972d606b76071990771aee539ce4145c5`;
- full ORCA2: 134776 lines, SHA-256
  `661743dd1c710b733fae48aad45fdcd924d19f178eab2637bd58a0a41056713c`;
- DINO: 123235 lines, SHA-256
  `7861f31e1cd582021e35a896e76827c01e53395434795b1517b50b1fa2ad3fcc`.

Each comparison emitted `expected_identical: true`.  The selected-only table
tests cover every remaining pre-existing card; none can select nonzero EIV.

## Acquisition preflight and boundary

The committed entry point defaults to read-only preflight and reserves
`--run` for the operator.  Preflight reports
`L3_PREFLIGHT_OK checklist=7 plants=all_refused` and proves:

1. the exact `ldftra` insertion context, a refusal after a continued `&`, and
   gfortran syntax of the instrumented preprocessed file;
2. `ORCA2_ICE_PISCES` is a real `makenemo -r` reference and a planted name is
   refused;
3. the rendered two-step deck has `nn_fsbc=2` and every end/stock time is a
   multiple of 2, while a planted stock at 1 is refused;
4. the six-frame synthetic writer matches the strict reader byte-for-byte,
   truncated/tag plants fail, and every character literal is exactly 16
   bytes;
5. all six expected NEMO frames have a writer and deleting the terminal frame
   fails;
6. output creation works below an absent parent and a regular-file parent is
   refused; and
7. both target configuration and run root are absent, with planted existing
   targets refused.

The operator acquisition creates three deterministic two-step twins, requires
all six observer frames, requires nonzero EIV transport deltas, and refuses an
L3 stage-3 tracer record equal to either admitted G9 or admitted L2.  No
record exists yet, so no score or bar verdict is permitted.

## Landing gates

- explicit L3 receipt citation gate: **PASS**, 25 citations, zero failures,
  zero unmapped citations, and zero map-audit failures;
- citation-gate unit suite after re-anchoring every merge-shifted pin:
  **17 passed in 5.30 s**;
- exact default ORCA2 landing list: **145 passed in 1019.36 s**;
- private-import ratchet, rerun separately: **6 passed in 5.07 s**; and
- acquisition preflight at `ab0978659197`: **PASS**, all seven plants refused.

The first default-list attempt correctly stopped at **144 passed, 1 failed**:
the default receipt still held ten stale bare continuations after the lane
merge.  Those citations were re-anchored in `ab0978659197`; the complete list
was then rerun to the 145-test result above.  No NEMO executable was built or
run.

## Continuation (Claude, 2026-10-20) — review of codex's uncommitted resume, items 1-5

Status stays **ACQUISITION_NEEDED**.  No NEMO was built or run and nothing was
scored.  The first acquisition (kept aside as `oracle_l3_r20_failed1`) ran NEMO
correctly with the eddy-induced velocity on; only the tool's `rn_Ue` text grep
refused it.

Codex left eleven files uncommitted.  Every hunk was kept; five defects in
them were fixed first, and nothing was stashed:

1. The bolus read the slope ghost row with sign +1; NEMO folds `wslpj` as
   `W` with sign -1 at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfslp.f90:338`.
2. The new fold also ran in L2's zero-coefficient centred bolus, changing
   L2's program.  It is now selected only with the through-FCT bolus, which
   exists only for `ln_ldfeiv` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:255-257`).
3. The shared ratio exchange emitted a dead slice in every traced program, so
   every ladder card's jaxpr had moved; it now emits no op.
4. A signed-zero `jnp.where` around the stage-3 add was replaced by NEMO's
   own association, `pFu - (zpsi(jk) - zpsi(jk+1))`
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:954-955`).
5. The acquisition's numeric log check anchored the name at line start, but
   NEMO prints a description first, so it would have refused the same log
   again; and its clock check demanded every restart-list entry be even,
   which refuses twin C (restarts 3,5,7,9, as L2's admitted twin C ran).
   NEMO's rule is `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:342-347`.

The five items, as landed:

1. **L1 restored.**  The merge let G9's table switch off L1's end-of-stage
   pivot fold (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:795-796`).  L1's
   two-step values now equal lane B's pre-merge tip `51b2582f5e`, 36 of 36
   leaves bitwise; the L1 tests, including the kt 1-4 record test, pass.
2. **Stage-3 routing.**  The bolus is added only to `pFu/pFv`, then `ww` is
   recomputed by `wzv` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:296`).  With
   the bolus forced to zero, T, S, u, v and ssh of the G9+L2 step are
   bit-identical (full depth); under the old route the same plant moves T in
   430,550 cells and S in 430,552 (up to 0.016 K and 0.14 psu), and the new
   test fails there.
3. **EIV north fold.**  The face coefficients take NEMO's `U`/`V` +1 exchange
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:783`), the slope ghost row its
   -1 fold, and the open exchanged V pivot mask, all through the shared fold
   helpers and L3-selected.  The exact unit test (jit, traced inputs) fails
   with any one of the three removed.
4. **One fold helper.**  Lane A's private `_folded_north` is gone;
   `north_neighbor_T` serves both.  The face builder takes one `grid=` with
   explicit `north_fold=` and `fold_exchange_ratios=` choices.
5. **Acquisition.**  Log values are parsed as binary64 and compared exactly
   to the deck.  Plants: NEMO's exact print lines (the old grep refuses them,
   the new check accepts them), a neighbouring float (refused), and the first
   acquisition's own log (accepted).  Preflight renders every deck the run
   uses.  The record follows the ladder standard: smoke, writer-free boundary
   run to NEMO's own stop or 100 steps, then three twins to at most 10 steps.

Every other card is unchanged.  Normalized one-step jaxprs equal merge
`d68c94cd66` for G5-G9, V1-V5, L2, GYRE, DINO, full ORCA2, SMT1-4, VORTEX,
OMT0-4 and rung 0.  Merge `d68c94cd66` itself reproduces both lane tips'
two-step values bitwise (G5, V1-V5, L2 against `51b2582f5e`; G9 against
`7a31b67ef9`; G6-G8 jaxprs equal lane A's).  Only L1 (restored) and L3 move.

Evidence, outside git:
`gridladder_vmix_rounds/round20/claude_r20c/` (jaxpr hashes, two-step arrays,
zero-bolus probes, pytest logs, preflight log); the inherited review dumps are
under `round20/independent_review_failed1/` and `claude_r20c/st_ref/`.

One pre-existing failure is unrelated to this diff: a source-text test of the
Treguier dispatcher (`test_treguier_kappa_uses_the_same_n2_variant_as_the_slopes`)
fails identically at merge `d68c94cd66`.

Landing gates for this continuation: recorded in the next commit.

independent review: operator (Claude)

codex review: operator (Claude-authored continuation; codex review when capacity allows)

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r20/repo
ACQUISITION_NEEDED: scripts/validate/ocean_fidelity/testcases/nemo_testcase_l3_orca2eiv/run.sh --run
DECISION_NEEDED: NONE
ROUND_STATUS: ACQUISITION_NEEDED — L3 reviewed, fixed and preflighted; L1 restored; record not acquired or scored
