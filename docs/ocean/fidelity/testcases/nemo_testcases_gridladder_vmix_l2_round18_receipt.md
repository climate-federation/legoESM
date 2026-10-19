# GRIDLADDER-VMIX lane — round 18 (vmix_r18): L2 operator walk acquisition

Status: **L2 SCORED; DEBT; NOT CERTIFIED** (Claude continuation, 2026-10-19;
section 5 supersedes the stop below). The operator's record is admitted; the
over-bar statement is fixed and L2-selected; the kt-1 lateral trend is now at
the bar in every cell, but one-ULP residuals still grow under the independent
label. The earlier stop for record: the frozen prediction is commit
`9f51498b92`; the operator-only acquisition is commit `5e9d88345f`, repaired
at `c2ec20cf41` and again after `f2784f8a2a`.

## 1. Frozen statement walk

Round 17 left the kt-1 stage-3 lateral-diffusion trend with 7,878 cells above
`1e-6` of scale, none above `1e-3`, and a maximum normalised difference of
`3.55187264925e-4`. The resulting T and S states each differed in 16,428 cells.
Those are carried measurements, not round-18 results.

The compiled stage-2 statements store `r3t`, `r3u`, and `r3v` independently at
N+1/2 at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:210-212`; stage 3 names that
level Kmm at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:229`. The selected
card already preserves those source-ordered U/V ratios for momentum and tracer
transport, but its Redi face path reconstructs them from averaged sea level.
The frozen P18-1 therefore predicts the first mismatch at the stored stage-2
U/V ratio assignment, observed when `traldf_iso_lap` forms `zA11` and `zA22`
from live face thickness at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:230-231`.

The preregistered refuting number is **0 unequal wet U faces and 0 unequal wet
V faces** between production and direct NEMO values. A nonzero count confirms
the boundary only if wiring the stored ratios makes those face arrays exact.
If P18-1 is refuted, P18-2 moves to the first below-surface vertical flux. The
surface gradient is established in
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:183-204`, and the compiled vertical
flux and divergence walk is
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:257-291`.

## 2. Why the admitted record cannot decide it

The admitted round-17 observer writes T, S, SSH, `r3t(Kmm)`, MLD fields, PRD,
PN2, the four final slopes, and four raw slopes. It does not write `r3u(Kmm)`,
`r3v(Kmm)`, either live face thickness, or any operator flux operand. The
round-17 reconstruction gate is deliberately not reused: this round's face
gate must compare production values to direct NEMO values, not feed NEMO SSH
through the same builder as the candidate. The frozen prediction therefore
cannot be confirmed or refuted from admitted evidence.

No correction was made. In particular, exact final slopes do not prove the
operator's later face values, and a last-bit tendency residual is not promoted
to a transport claim without the first differing statement.

## 3. Operator-only acquisition

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_orca2iso/run_round18_ldf_walk.sh`
has read-only default preflight, operator-only `--run`, and
`--admit-existing`. It applies only additive instrumentation from:

- `traldf_iso_round18.patch`, SHA-256
  `e002824bfd437a3c1d41a3c336a5e653889da4e01922f0f20bb5f1c2058476ab`;
- `traldf_iso_scheme_round18.patch`, SHA-256
  `c72eef59d12fbb15c72aee1bd5cadda60ffa43bf0e4d3636863ffd341c0c5045`.

The repaired acquisition script is SHA-256
`54ac7294e1ca3ac6b2d1b3af643074e57b11488d8d59952e3178ec04f76b9317`.
It pins the source executable, canonical operator sources, deck, step-entry
record, and stage-3 record. It records 40 three-dimensional arrays, 11
two-dimensional arrays, and the 31-level vertical coordinate in a registered
279,904,060-byte stream. The record includes direct NEMO `r3u(Kmm)`,
`r3v(Kmm)`, `e3u_live`, `e3v_live`, `A11`, `A22`, horizontal and vertical
flux operands, the before/after RHS, and its increment. Thus the face gate is
anchored to NEMO values from the dump rather than a shared reconstruction.

Admission requires exact replay of both live-face statements, zero surface
`dkt`, zero surface vertical flux, nonzero first-below-surface flux, exact
before/after increment replay, finite and acting registered operands, byte-
identical step-entry/stage streams, a successful NEMO stop, and a commit-bound
record stamp. `--plant-layout` removes a required writer assignment;
`--plant-admission` corrupts an existing record's magic. Neither plant can
pass its corresponding gate.

The first operator attempt refused because `df` was given the absent round-18
evidence parent. The second passed that point after the operator created the
parent, but `makenemo -r ORCA2_OMIP_L4` treated a work configuration as a
reference: it attempted `/work_cfgs.txt`, then failed preprocessing because
`do_loop_substitute.h90` was absent. The third attempt built, then NEMO stopped
in `sbc_init`: its experiment length was 1 while `nn_fsbc` was 2. The failed
configuration and run were moved aside; the requested target name and evidence
target are both free again.

The repaired build follows the working gridladder acquisitions: its reference
is `ORCA2_ICE_PISCES`, which preflight requires to be listed in
`cfgs/ref_cfgs.txt`; after that reference creates the target, the script
replaces `MY_SRC` and the cpp key file from
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1`, reapplies the two additive observer files,
and clean-rebuilds. Free-space checks resolve a missing requested directory to
its nearest existing ancestor, and now run during read-only preflight.

The rendered delta now keeps `nn_it000=1` and sets `nn_itend=2`, `nn_stock=2`,
and `nn_stocklist=2,0,...`. NEMO assigns the configured start and end to
`nit000` and `nitend` after reading `namrun`, at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/domain.f90:323-324` and
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/domain.f90:367-368`.
It only normalizes `nn_stock` when list mode is off, at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/domain.f90:390`.
`sbc_init` requires the inclusive experiment length and, outside list mode,
the stock frequency to be multiples of `nn_fsbc`, at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/sbcmod.f90:342-344` and
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/sbcmod.f90:346-348`.
Here both the length and `nn_stock` are 2. With list mode on, restart setup
takes `nn_stocklist(1)` as `nitrst` at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/restart.f90:96-98` and
opens the restart at `nitrst-1` under the condition at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/restart.f90:119-120`.
Therefore the restart opens at kt 1 and writes at kt 2, while the independent
observer condition remains `kt == nit000` and records kt 1.

Read-only preflight passed the exact additive-patch layout and a gfortran
syntax proof, reporting:

```text
DISK_PREFLIGHT_PASS requested=/tmp ancestor=/tmp
DISK_PREFLIGHT_PASS requested=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round18 ancestor=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round18
DISK_PREFLIGHT_PASS requested=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2 ancestor=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
NAMELIST_DELTA_PASS nn_it000=1 nn_itend=2 experiment_length=2 nn_fsbc=2 ln_rst_list=true nn_stock=2 nn_stocklist=2 record_kt=1
SYNTAX_PROOF_PASS traldf_iso.f90
ROUND18_L2_LDF_PREFLIGHT_READY target=ORCA2G5_OMIP_L4_VMIX_R18LDFWALK evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round18/oracle_l2_r18_ldf_walk
```

The invalid-reference, missing-evidence-parent, and invalid-namelist plants each
fired and exited 69 as required. The namelist plant rendered `nn_itend=1` and
the preflight rejected its one-step experiment against `nn_fsbc=2`:

```text
STATUS PLANT-FIRED: work-configuration-reference
REFUSE: intentional work-configuration reference plant exit
PLANT_EXIT=69
DISK_PREFLIGHT_PASS requested=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/.round18-missing-parent-plant/nested ancestor=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds
STATUS PLANT-FIRED: missing-evidence-parent ancestor=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds
REFUSE: intentional missing-evidence-parent plant exit
PLANT_EXIT=69
STATUS PLANT-FIRED: namelist-nn_itend=1 nn_fsbc=2
REFUSE: intentional invalid experiment-length plant exit
PLANT_EXIT=69
```

The operator should run, from a clean checkout:

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_orca2iso/run_round18_ldf_walk.sh --run`

If the operator already produced the target, use `--admit-existing`; do not
rerun or overwrite it.

## 4. Controls and landing gates

Because no production physics changed, no ladder rescore or certification is
claimed. The acquisition compares the admitted step-entry and stage-3 streams
byte for byte to prove the observer is write-only. The preregistered card
controls remain mandatory for the eventual correction: unchanged jaxprs for
G5, V1-V5, OMT-0..4, rung 0, VORTEX, and SMT-1/2; unchanged values for L1,
full ORCA2, DINO, and SMT-3/4; and byte-identical GYRE ten-step evidence.

Repair gates (CPU, x64, landing-script `PYTHONPATH`):

- this receipt's explicit citation gate: **PASS**, 12 citations, 0 unmapped,
  0 failures, and 0 map entries failing audit;
- exact default ORCA2 landing list: **145 passed in 1140.03 s (19:00)**;
- private-import ratchet: **6 passed** within that exact landing list;
- `git diff --check`: **PASS**.

Unasked choices: none. The target configuration and evidence-root names are
mechanical acquisition identifiers. No threshold, field set, cadence, model
selection, or scientific decision was changed.

## 5. Continuation: admitted walk, a33 fix, and re-score

### Admission

The operator's fourth run built, ran two steps, stopped normally, and passed
the passive-observer twin check. The admission replay was repaired in
`0afedd0573`. From a clean tree at the fix commit, `--admit-existing` ends:

```text
ROUND18_L2_LDF_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round18/oracle_l2_r18_ldf_walk
```

### P18-1 and P18-2 outcomes

P18-1 is **confirmed as the first non-bit operand but refuted as the owner of
the debt**. The live U/V face thicknesses differ from NEMO's dumped
`e3u_live`/`e3v_live` in 36 U and 20 V faces, each by one ULP (normalized
`<=1e-16`, below the bar). Slopes, gradients, `zA13`, and `zA23` are exact.

P18-2 is **refuted as stated**: the over-bar owner is upstream of the vertical
flux, in `traldf_iso_a33`, called at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:154`. Its laplacian branch forms
`ze3w_2` from `e3w_1d*(1+r3t(Kmm))` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:825-829`, and the vertical A33
flux divides by the same object at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:283`. Kmm is stage 2's stored
N+1/2 ratio. L2 used the step-entry stretch: 492,990 of 799,200 `e3w` values
differed from NEMO's, max 5.10 m.

### Fix (commit `30934c2bd2`)

L2's stage-3 Redi path now passes the stored stage-3 Kmm `e3w` to both the
explicit operator and the K33 path. It is selected only by the existing
`redi_flux_geometry_stage = "rk3_kmm"`, which only ORCA2L2-iso sets; the
default (`step_entry`) is unchanged.

Codex's uncommitted face rewiring was **dropped, not committed**. It moved
1,319 north-fold face values off NEMO (max 0.024 m); the existing face path is
within one ULP of NEMO in 36 U / 20 V faces.

Value gates (jitted production step, traced state, x64):

- production `e3w` equals NEMO's `e3w_1d*(1+r3t(Kmm))` exactly; faces equal
  NEMO's dump except 36 U / 20 V one-ULP cells;
- removing only the new `e3w` wiring fails that gate (492,990 cells) and the
  kt-1 record gate (T/S leave the bar);
- kt-1 record gate: SSH, barotropic U/V, U, V, T, S all AT-BAR; T and S differ
  in 2 cells by `7.105427357601002e-15`. The metric, EOS-80, north-fold, and
  Kmm-geometry omission controls each remain DEBT with fixed value tuples.

kt-1 operator walk against NEMO's dump after the fix: the lateral trend
differs in 280,831 cells, max `2.558e-16` of scale; **zero cells exceed
`1e-6`** (round 17: 7,878).

### Re-score, kt 1..10, D127 bar `1e-15`

| label | surface rows at bar | full-depth summaries at bar | worst normalized |
|---|---:|---:|---:|
| independent | 31 / 68 | 14 / 36 | `1.85e-14` (kt-9 SSH) |
| given NEMO entry | 64 / 68 | 19 / 20 | `1.7e-15` (kt-2 V) |

Every dry cell is exact. Round 17's score was 5/68 + 0/36 and 25/68 + 0/20.
The DEBT rows are all SSH and velocity, beginning at kt 2, seeded by the two
one-ULP T/S cells at kt 1. Given-entry DEBT rows are second-step
continuations only (kt 2, 4, 8). **L2 is not certified.**

### Next first non-bit statement (measured, not fixed)

In compiled order the first one-ULP owner is `ah_wslp2`, the first output of
`traldf_iso_a33`. NEMO sums the four face diffusivities as two bracketed pairs
at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:788-791`, then forms
`ah_wslp2` left to right at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:793`. Replayed from NEMO's own
dumped operands: bracketed pairs give 0 unequal cells; left-to-right sums
(the shared kappa-sum helper's order) give 18,465. The shared helper under
jit gives 60,720, equal to the production count, so compiler contraction is a
second, separate cause. Both need an L2-selected source-ordered path. That is
the next round's frozen prediction, not this round's change.

### Controls

Measured jaxpr SHA-256, parent `0afedd0573` vs fix: identical for G5, V1, V2,
V3, V4, V5, L1, full ORCA2, and DINO; L2 differs (193,776 -> 193,772 lines),
as intended. Identical jaxprs imply bit-identical values for L1, full ORCA2,
and DINO.

GYRE's ten-step trajectory gate: all 50 field rows, 20 barotropic rows, both
first-over-bar summaries, and status equal round 17's; the residual archive is
byte-identical at SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.

**NO GATE**: OMT-0..4, rung 0, VORTEX, SMT-1/2 jaxprs, and SMT-3/4 values were
not run. Read off the code only: the new path is entered only when
`redi_flux_geometry_stage == "rk3_kmm"`; otherwise the override is `None` and
both consumers take their previous branch.

Evidence: `gridladder_vmix_rounds/round18/score/` (both score JSONs,
purity logs) and `gridladder_vmix_rounds/round18/gyre_regression/`.

Gates (CPU, x64, landing-script `PYTHONPATH`):

- this receipt's explicit citation gate: **PASS**, 17 citations, 0 unmapped,
  0 failures, 0 map entries failing audit (model-file spans re-anchored);
- exact default ORCA2 landing list: **145 passed in 1142.10 s (19:02)**;
- private-import ratchet: **6 passed in 5.60 s**;
- focused L2 file: **17 passed** (the two record gates re-run after the
  expectation update: **2 passed in 565.44 s**).

Unasked choices: none. The fix uses an existing L2-only selector; no default,
threshold, or card selection changed. Dropping the uncommitted face rewiring
followed the operator's instruction for unreviewed diffs.

independent review: operator (Claude)

codex review: operator

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r18/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: L2 SCORED; DEBT; NOT CERTIFIED — a33 Kmm e3w fixed (0 trend cells over 1e-6); independent label 37 surface + 22 full-depth rows over the 1e-15 bar, seeded by one-ULP ah_wslp2 pair-sum association and compiler contraction (traldf_iso.f90:788-793), measured, unfixed
