# GRIDLADDER-VMIX L3 round 20 receipt — ORCA2 eddy-induced velocity

Status: **L3 CERTIFIED**.  The operator's admitted three-twin record was
scored at every stored step through kt 10 under the frozen two-label, full-
depth and D127 contract.  Every scored row is exact.  Codex did not build or
run NEMO.

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
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:255-257`.  Thus the bolus augments
the horizontal transport passed to the existing G9 FCT path; NEMO then
recomputes `ww` from those augmented transports at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:296`.  It therefore enters tracer
continuity, but not momentum.  This corrects P2's frozen continuity clause.

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

The operator acquisition runs a two-step smoke, a writer-free boundary arm,
and three twins to a common even endpoint no greater than 10 steps.  Admission
requires all six observer frames, nonzero EIV transport deltas, and an L3
stage-3 tracer record unequal to both admitted G9 and admitted L2.  No admitted
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

Landing gates for this continuation (JAX_ENABLE_X64=1, land.sh PYTHONPATH):

- this receipt's citation gate: **PASS**, 30 citations, 0 failures,
  0 unmapped, 0 map-audit failures (92 shifted legoESM-source entries
  re-anchored; one duplicate key removed);
- default landing list: **144 passed, 1 failed** on the first run, the
  failure being my duplicated map key; removed, the citation-gate file is
  re-run below;
- private-import ratchet: **6 passed**;
- L1 + L3 + bolus unit file **37 passed**; L2 **15 passed** plus its two slow
  tests **2 passed**; G6 **17 passed**; the other touched unit files passed;
- acquisition preflight: `L3_PREFLIGHT_OK checklist=7 plants=all_refused`
  with the evidence parent absent.

## Continuation (Codex, 2026-10-20) — failed twin diagnosis and acquisition repair

The second operator acquisition is retained as `oracle_l3_r20_failed2` and is
not admitted.  Smoke, the writer-free boundary arm, and all three L3 twins ran;
each twin completed 10 steps with exit 0, and the acting check reported
837,255 changed transport values.  Of 110 oracle files, 109 are byte-identical.
Only `oracle_l3_eiv_kt00000001.bin` differs, and only twin B's `before_u` and
`after_u`: the same 281 cells differ, all on levels 1-2 and all in the final
local row or column.  `delta_u`, every V frame, and twins A/C are identical.

The cause is read off NEMO's executed RK3 path.  `zFu/zFv` are allocatable work
arrays at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:108` and are allocated
without initialization at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:121`.
The producer writes levels 1 through `jpkm1` only over its lower-inclusive,
upper-exclusive halo rectangle at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:281-283`; `tra_adv_trp` writes the
bottom level over the same horizontal rectangle at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:248-251`, then calls EIV at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:254-257`.  EIV initializes its own
streamfunctions and updates `pFu/pFv` over that same rectangle at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:938-955`.

For this one-rank record, the array is 184 x 152 x 31 and those loops write
`i=1..183, j=1..151`; the untouched union `i=184 OR j=152` therefore contains
10,385 allocated-but-undefined slots per transport field.  All 281 observed
differences lie in that union.  The halo-free mesh has exactly 413,030 wet U
slots and 415,175 wet V slots.  Both sets lie wholly inside the initialized
rectangle.

The observer now follows the existing `l4_canon_3d` pattern: a private result
starts at +0 and reads only owned cells whose U/V mesh mask is nonzero.  NEMO's
arrays remain `INTENT(in)` to that helper and are never assigned.  Thus every
undefined halo slot and every other excluded dry/halo slot is canonical +0,
while all defined wet values retain their exact bytes.  The reader requires
canonical +0 outside that domain and the twin scorer byte-compares only the
same wet U/V masks.  legoESM has no NEMO halo cells and holds zero on excluded
dry faces by its production face-mask multiplication at
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:452-459`.
The synthetic plants prove that garbage in an excluded cell is canonicalized
away, while the same garbage in a defined wet cell still separates the twins.

Existing-record admission no longer trusts twin A's `time.step`.  It derives
the common even endpoint from the audited writer-free boundary arm, then
independently rechecks the current producer binary, immutable inputs, rendered
deck, L3 log selection and numeric values, exit status, and final step for the
smoke and all three twins.  Plants alter B/C's final step, a completed deck, a
completed log, and the smoke exit; each is refused.  The raw-observer plant is
also refused.

At `cf43852d6b`, with both target configurations and the evidence parent
absent, read-only preflight reports
`L3_RECORD_SELFTEST_OK undefined_garbage=canonicalized defined_garbage=detected`
and `L3_PREFLIGHT_OK checklist=7 plants=all_refused`.  No NEMO executable was
built or run, and no score was computed.

independent review: operator (Claude)

codex review: operator

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r20/repo
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r20/repo/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l3_orca2eiv/run.sh --run
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — L3 reviewed, fixed and preflighted; acquisition needed; record not acquired or scored

## Continuation (Codex, 2026-10-20) — admitted-record score and L3 certification

The operator ran the preflight at `09a22e7fc0f0`, which reported
`L3_PREFLIGHT_OK checklist=7 plants=all_refused`, and then ran the acquisition.
Its terminal admission line is:

`L3_ACQUISITION_PASS twins=3 steps=1..10 reference_completed=21 evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round20/oracle_l3_r20`

The evidence root contains all three agreeing twins, the kt-1 stage records,
twin B's even restarts 2 through 10, twin C's odd restarts 3 through 9, and
twin A's endpoint restart.  The operator log is
`claude_gridladder_vmix/acq_l3_r20.log`.  This continuation only read the
admitted artifacts; it did not invoke NEMO.

### First differing statement and selected repair

The first walk was exact through stages 1 and 2 and exact at stage 3 for U/V
transport, momentum, SSH, and barotropic state.  Stage-3 T and S were the
first debt: 16,433 surface cells in each field, with maxima 0.04338 K and
0.02613 psu, and the same tracer debt extended through the wet full-depth
classes; dry cells were exact.  The independently observed pre-EIV transport
was already exact.  Reproducing the written Treguier accumulator, Rossby/taper
chain, face materialization and streamfunction association from
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:758-760`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:767`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:769`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:775-776`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:780-781`, and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:947-950` made all six observed
pre/delta/post U/V transport arrays exact.

That exact transport left the same first T/S debt, isolating it to the next
statement boundary.  NEMO adds lateral diffusion into `Krhs` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734` before tracer integration.
G9 alone owns the written stage-3 `Krhs` tuple and L2 alone owns lateral
diffusion, so the composed L3 card was the only route that had omitted the
diffusive term from that tuple.  Appending the existing L2 T/S tendency there
is selected only by that composition.  It makes the kt-1 stage-3 T/S fields
exact and removes the propagated debt through kt 10; it does not alter an
unselected operator or introduce new physics.

### Frozen score

Both preregistered labels pass:

- **independent:** kt 1 through 10, 70/70 surface rows exact and 40/40
  full-depth summaries AT-BAR;
- **given-entry:** odd entries 1, 3, 5, 7, 9 plus their carried second steps
  2, 4, 6, 8, 10, again 70/70 surface rows exact and 40/40 full-depth
  summaries AT-BAR.

The full-depth rows cover U, V, T and S at all 30 levels.  Across full-wet,
partial-wet, and dry cells they report zero normal differences, zero bitwise
differences, zero maximum error, and zero flushed-subnormal exceptions.  The
classifier nevertheless uses Decision 127 explicitly: a planted NEMO
nonzero subnormal against card zero is AT-BAR/flushed, while a planted normal
difference is DEBT.  The score artifacts are
`gridladder_vmix_rounds/round20/l3_independent_score_final.json` and
`gridladder_vmix_rounds/round20/l3_given_entry_score_final.json`.

The eddy-induced velocity acts exactly where predicted.  Its before, delta,
and after U/V horizontal transports have zero normal differences from the
record.  A jitted production step with traced inputs and fp64 then matches
the stage-3 NEMO T/S fields at full depth.  Removing Treguier-21 changes
430,385 T and 430,365 S cells; removing NEMO face averaging changes 430,366
T and 430,161 S cells; removing through-FCT routing changes 430,290 T and
429,336 S cells.  The record test therefore fails with each frozen L3 wiring
removed.

Every other card remains fixed.  Normalized one-step jaxprs are identical for
G5-G9, V1-V5, L1, L2, GYRE, DINO, full ORCA2, SMT1-4, VORTEX, OMT0-4, and
rung 0; only L3 differs.  Fresh two-step runs of G5-G9, V1-V5, L1 and L2 all
completed.  G5, G9, V1-V5, L1 and L2 compare bitwise in every saved leaf to
their frozen arrays; G6-G8 retain their identical normalized programs and
inputs.

Landing-gate results for this continuation are recorded below after the final
reruns.  With every required row exact or at the bar, the preregistered verdict
is **L3 CERTIFIED**.

independent review: operator (Claude)

codex review: operator

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r20/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: L3 CERTIFIED — both labels exact through kt 10 at full depth; EIV transport and caused T/S change exact
