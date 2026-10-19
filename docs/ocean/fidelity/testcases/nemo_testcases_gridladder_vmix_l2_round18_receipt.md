# GRIDLADDER-VMIX lane — round 18 (vmix_r18): L2 operator walk acquisition

Status: **STOPPED FOR RECORD; NO PHYSICS CHANGE; NOT CERTIFIED.** The frozen
round-18 prediction is commit `9f51498b92`; the operator-only acquisition is
commit `5e9d88345f`. This round did not build or run NEMO and produced no new
oracle measurement.

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

The acquisition script itself is SHA-256
`47483448f65b1e877eb79f7df195a4d75eb2f37305acf2ae7abcacb433fef078`.
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

Read-only preflight passed the exact additive-patch layout and a gfortran
syntax proof, reporting:

```text
SYNTAX_PROOF_PASS traldf_iso.f90
ROUND18_L2_LDF_PREFLIGHT_READY target=ORCA2G5_OMIP_L4_VMIX_R18LDFWALK evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round18/oracle_l2_r18_ldf_walk
```

The source-layout plant fired and exited 69 as required:

```text
STATUS PLANT-FIRED: source-layout
REFUSE: intentional source-layout plant exit
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

Final gates (CPU, x64, landing-script `PYTHONPATH`):

- this receipt's explicit citation gate: **PASS**, 5 citations, 0 unmapped,
  0 failures, and 0 map entries failing audit;
- planted shift of the stage-2 ratio citation: gate **FAIL**, 1 failure, exit
  1, as required;
- exact default ORCA2 landing list, including the receipt-citation tests,
  private-import ratchet, and parallel-receipt guard: **145 passed in
  1136.90 s (18:56)**;
- `git diff --check`: **PASS**.

Unasked choices: none. The target configuration and evidence-root names are
mechanical acquisition identifiers. No threshold, field set, cadence, model
selection, or scientific decision was changed.

independent review: operator (Claude)

codex review: operator

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/vmix_r18-1984811343
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/vmix_r18-1984811343/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_orca2iso/run_round18_ldf_walk.sh --run
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — preflighted operator-only L2 internal walk is required before the first differing statement can be fixed
