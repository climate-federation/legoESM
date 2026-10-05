# Receipt — VORTEX_SMT round 17 (lane round 229): LDF tensor magnitude walk

**Status: HELD.**  No physics, card, carried state, trajectory, or certified
number changed.  The corrected R16 record is admitted and all six repaired
tensor-factor groups calibrate bit-for-bit.  The first source-exact candidate,
closing NEMO's deepest W-mask level, removes only 33.48% of the production-JIT
LDF RHS error and therefore fails the preregistered 90% magnitude criterion.
Production is restored and the candidate is preserved under `manifests/`.
The remaining 66.72% is now closed to the final partial-cell tracer-thickness
divisor, which is the next round's statement.

Base: `be95a1a1685adcdb24ceb1a0276fea81c600c836` (round 228).
Preregistration: `5367387bd`.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round229/`.

## 1. Admission and frozen aggregate

The operator-run R16 acquisition ends in
`ROUND228_LDF_INTERNAL_RECORD_READY`.  Its step-10 restart is byte-identical
to the plain build; the two self-describing records parse 16 slope groups and
50 ISO groups to EOF; `rhs_after-rhs_before` rebuilds `rhs_increment` with
zero unequal cells; its header, field-name and truncation plants all exit
nonzero; and both record stamps pass.  Unlike the retracted R15 groups, the
R16 writer copies every scalar inside its producing loop at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:260-266`.

The clean production-JIT replay reproduces the frozen boundaries exactly:

| boundary | maximum absolute T difference |
|---|---:|
| pre-LDF | `7.418332614861356e-11 K` |
| post-LDF | `2.0915088416728622e-07 K` |
| additional across LDF | `2.090767008411376e-07 K` |

R17-P1 and R17-P2 are confirmed.  Direct reconstruction from the full
halo-bearing recorded operands gives zero unequal cells for `A11`, `A22`,
`hmsku`, `hmskv`, `A13`, and `A23`; the production row plant on `hmsku`
exits 1 and prints `STATUS PLANT-FIRED`.

## 2. Compiled-order tensor walk

NEMO forms the two live-thickness diagonal coefficients, four-mask
reciprocals, slope cross-terms, and horizontal fluxes in that order at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-259`.
The production-JIT comparison is:

| row | unequal cells | maximum absolute difference |
|---|---:|---:|
| `e3u_flux` / `e3v_flux` | 8,071 / 7,993 | 170.383902028 / 170.383902114 m |
| `dit` / `djt` / `dkt` | 0 / 0 / 0 | 0 / 0 / 0 |
| `A11` / `A22` | 36,032 / 36,012 | 170.383900019 / 170.383895118 |
| `hmsku` / `hmskv` | 3,590 / 3,590 | 0.25 / 0.25 |
| `A13` / `A23` | 626 / 626 | 0.0263429385 / 0.0264142613 |
| `fu` / `fv` | 6,656 / 6,656 | 23.9209085204 / 24.6183175282 |
| LDF RHS increment | 8,641 | `7.2621621143171395e-11 K s-1` |

Thus R17-P3 is **REFUTED**: the six factors are valid record fields but are
not bit-exact in production.  The first large statement is the four-mask
reciprocal.  NEMO reads its explicit closed `jk+1` W level in the cited
horizontal block; legoESM's periodic vertical roll wraps the surface W mask
onto the floor.

The one-variable source-exact arm closes that bottom level only.  It makes
`hmsku`, `hmskv`, `A13`, and `A23` bit-exact and reduces the LDF RHS maximum
from `7.2621621143171395e-11` to `4.8307864472774789e-11 K s-1`: 33.4800522%
removed, 66.5199478% remaining.  Its stage increment falls from
`2.090767008411376e-07` to `1.390524424493833e-07 K`.  The old periodic form
fails the focused regression test, while the corrected production row's ULP
plant fires.

R17-P4 required at least 90% closure.  It is **REFUTED**, so the candidate did
not enter any trajectory landing gate.  The exact patch and its non-vacuity
test are retained as
`manifests/nemo_testcase_l1_vortex_smt_round229_closed_bottom_wmask_held.patch`.

## 3. Remaining magnitude owner

A post-hoc family substitution, explicitly not promoted to a production
proof, gives:

| recorded family substituted | remaining RHS maximum |
|---|---:|
| horizontal fluxes | `4.8452277849849912e-11 K s-1` |
| vertical fluxes | `7.2621621143193471e-11 K s-1` |
| all horizontal and vertical fluxes | `4.8452277850122817e-11 K s-1` |
| all fluxes plus NEMO's final volume divisor | `2.9778502051908996e-22 K s-1` |

The vertical family is exonerated at this magnitude.  Recorded horizontal
fluxes remove 33.28%, consistent with the held mask candidate, but replacing
every flux still leaves 66.72%.  The residual closes by 11 orders of magnitude
only when the exact compiled final statement is replayed: NEMO multiplies the
three flux differences by stored `r1_e1e2t` and divides by live
`e3t_3d*(1+r3t(Kmm)*tmask)` at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`;
the deepest-level branch uses the same divisor at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331`.

The stored horizontal metric reciprocal is already bit-exact.  The effective
tracer thickness is not: 36,494 wet cells differ, by as much as
`449.32957223715755 m`, because the current operator constructs
`dz_ref*jacobian` rather than consuming the card's live partial-cell
`e3t_3d*(1+r3t(Kmm))`.  This is the named first magnitude statement for round
230.  The `2.98e-22 K s-1` remainder is the compiled-rounding floor, not a
second magnitude owner.

## 4. Landing and blast radius

R17-P5 is not reached.  Production at round end is byte-for-byte the round-228
physics; no certified trajectory row can move.  Accordingly the SMT/flat
VORTEX registries, tanks, GYRE ladder/year, generic GYRE recipe, DINO month,
and ORCA2 ladders are not re-baselined.  The committed changes are an extended
measurement gate, citation mappings, documentation, and the held manifest.

The next divisor statement is shared by every card selecting this
`nemo_iso_lap` path.  If its production arm closes, its landing round must
measure both SMT cards, all flat VORTEX cards, tanks, the certified GYRE ladder
and year, the generic GYRE card, the private-work-directory DINO month gate,
and write the ORCA2 merge pointer.  No card-specific scope is inferred here.

**UNASKED list: EMPTY.**  No physical/configuration/default choice was made.

## 5. Tests, review, and citations

The clean production walk and its production-JIT plant are recorded in
`final_tensor_walk_v2.json` and `candidate_plant.log`.  The old-form unit
non-vacuity failure is `nonvacuity_old_guard.log`.  Focused test and citation
gate summaries are recorded below after the final clean-tree pass.

The required separate read-only Codex review result is recorded below after
the final diff review.

## 6. OPEN

Round 230 stays on SMT-3.  Add one private production-JIT discriminator that
threads the card's already-verified live Kmm T-point thickness into only the
final Redi divergence divisor.  Reproduce the `4.8452277850122817e-11` residual
with recorded fluxes first, then require the production arm to reach the
`2.9778502051908996e-22 K s-1` reconstruction floor without moving upstream
flux rows.  If it closes, transcribe NEMO's live partial-cell divisor and run
the complete shared-statement landing gates named above.  Only after SMT-3 is
closed does Decision 93 proceed to SMT-4 lateral momentum diffusion.

No NEMO acquisition and no configuration decision is required.
