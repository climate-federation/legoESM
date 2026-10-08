# ORCA2 round 167 — kt=8 exit-depth hold

Date: 2026-10-07. Base `7ad068658`; preregistration `642a676dc`;
final measurement instrument `73c60ec1f`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round167/`.
Verdict: **HELD**. The first source-ordered non-bit boundary before the
round-166 reciprocal overflow is the exit U-face depth sum. The reciprocal and
the following seven-array boundary association do not own the non-finite
transition. The depth sum's two operands remain to be split before any source
change or atomic halo/V-transport landing.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into the table. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple are
unchanged.

## Admitted record and source statement

The operator-run round-166 record re-admits with exactly two rank slabs,
exactly-once global coverage, 65 external substeps and 2,106 self-described
groups per rank. All 20 instrumented terminal restarts are byte-identical to
the admitted round-96 baseline. Rank-record SHA-256 values are
`c2c4c27410e6c9ba3e6a0606cbdc28c77946398401203cadcb59912bce6ff4b9`
and `06e1f3afe622e955f23ebd7f9654a42afc7130e52dc3312d37f387fb897f2555`.

The record's compiled executable constructs `hu_e = hu_0 + zsshu_a` and then
the masked reciprocal at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:763-766`. It next
associates velocities, depths, reciprocals and SSH in one call at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:770-779`. Those branches
execute here: the admitted header says `Kcycle=65`, the rung-0 deck is
non-linear free surface, and the recorder contains all 65 post-call depth and
reciprocal frames on both ranks.

## Measurement

The complete private arm from rounds 164-166 again completes independent
steps 1-7. Its admitted round-166 trace first has 42 non-finite compact U
reciprocals at kt=8 external substep 2. One is the compact grid's redundant
periodic closure; the self-describing native NEMO layout therefore contains 41
unique registered faces. The first maps from compact `[j=7,i=114]` to native
`[j=7,i=113]`. This coordinate conversion corrects the preregistration's
implicit 42-native-face reading; it does not change the admitted compact
census.

| substep-2 boundary | active differing faces | candidate non-finite active faces | finite maximum absolute difference |
|---|---:|---:|---:|
| exit U depth, before association | 15,789 | 0 | 7.880736370873552e53 m |
| exit U reciprocal, before association | 15,789 | 40 | 1.4073748835532803e14 m^-1 |
| exit U depth, after association | 15,789 | 0 | 7.880736370873552e53 m |
| exit U reciprocal, after association | 15,789 | 40 | 1.4073748835532803e14 m^-1 |

Across all 41 unique registered faces, legoESM's pre-association depth is
finite and exactly `0.0` m. NEMO's recorded depth is finite and spans
36.60196376160646 to 5392.468449038692 m; its recorded reciprocal is finite.
Replaying only NEMO's recorded depth at the reciprocal consumer removes all
41 non-finites. One replay bit remains unequal because the oracle frame is
post-association while this replay deliberately stops before association; it
does not revive a non-finite. Replacing only the reciprocal leaves all 41
depth differences, so it cannot own an earlier boundary.

Thus the first statement named by this walk is the exit face-depth sum, not
the division and not `lbc_lnk`. This is a boundary attribution, not permission
to edit the sum: its raw reference-depth and face-SSH operands have not yet
been separated. Production packages, configuration and sea ice remain
unchanged, and the adverse atomic halo/V-transport unit remains private and
HELD.

## Predictions, controls and retractions

R167-P1, P2, P3 and P4 are **CONFIRMED**. P2's 42 compact faces correspond to
41 unique native record faces; all have a finite wrong depth before the
reciprocal. The three report plants fire on the compact census, compiled source
order and depth replay. Focused unit coverage also perturbs a nonzero depth,
not a zero multiplier.

Four failed instrument attempts remain evidence. The first evaluated the
known-failing full-step companion after the barotropic trace. The second added
seven arrays to the JIT output and changed the census, so it was rejected as a
non-passive observer. The third passed host NumPy arrays to a JAX-only helper.
The fourth exposed the compact/native count mismatch above. None produced a
reported science number; their logs are retained.

The separate `codex exec --sandbox read-only` review attempt returned
**independent review unavailable in-sandbox** before reading the diff:
`failed to initialize in-process app-server client: Read-only file system`.

ASKED choices: continue the compiled-source independent rung-0 walk. UNASKED
choices: empty. No configuration, forcing, carried-state policy, stabiliser,
sea-ice selector or production model statement changed.

## Mechanical validation

Focused coverage passes 59/59, including the new record-layout, compact/native
index, nonzero perturbation and report-plant controls. The cumulative citation
gate and this receipt's gate both pass with zero unmapped spans; shifting the
compiled `dynspg_ts.f90:763-766` citation makes the planted gate refuse.

The one required `tests/ocean/fidelity -n 12` invocation collected 2,708
tests and reached 99% before its 900-second controller cap during
`test_prediction_plant_is_fail_closed`. Its retained log contains 2,688 passed,
7 skipped and 4 failed terminal outcomes, leaving 9 without a terminal outcome;
no second full battery was started. The four failures are the registered
pre-existing reds also present in the round-166 battery: the moved GYRE
certified-year spread record, unscoped allow-dirty drivers, unstamped report
emitters and SI3 scalar-math provenance. No round-167 test failed, and the log
contains neither a symbol-materialisation failure nor a `MemoryError`.

## OPEN

1. At kt=8 external substep 2, split the zero exit-depth sum into the carried
   raw `hu_0` operand and recorded `j002_sshu_a` face SSH, one variable at a
   time. The present record contains the required face-SSH frame.
2. If face SSH is first non-bit, walk its area-weighted two-cell construction
   back to `j002_ssha_e`; then walk continuity transport/divergence in compiled
   order. No acquisition is currently required.
3. Keep the complete raw-depth/no-extra-V-mask/seven-array/materialised-`zhV`
   unit private and HELD until the upstream compensating statement is exact
   and Decision 96's full shared gates pass.
