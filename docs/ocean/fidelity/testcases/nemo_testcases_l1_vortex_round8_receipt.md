# NEMO testcase fidelity: round 192 / VORTEX round 8 receipt

**Status: STOPPED_FOR_RECORD.** The first non-bit stage-2/3 momentum statement
cannot be named from the admitted VORTEX records: they contain stage entry and
stage output, but no within-stage momentum accumulator. This round therefore
adds only a fail-closed, paired-build acquisition arm to the existing VORTEX
harness. It changes no model code, recipe, card, restart state, or certified
trajectory. The operator must run the acquisition; round 193 will perform the
one-variable production-JIT walk.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round192/`.
The committed preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_round8.md`.

## 1. Frozen residual and why a record is required

Round 191 leaves the VORTEX-vector first-over-bar row at `kt=2`:

| field | normalised max residual |
|---|---:|
| T | `3.625489e-09` |
| S | `4.060244e-16` |
| u | `3.3693297863401916e-06` |
| v | `3.337024e-06` |
| ssh | `3.709010e-08` |

The previously admitted stage walk bounds the momentum discrepancy to the
live RK3 stages: stage 2 is non-bit at
`u = 1.6701813777553198e-06` and stage 3 at
`u = 3.3737007034684297e-06`. Those records expose neither the accumulator
between operators nor the vertical velocity used by vertical advection. A
first non-bit *statement* is therefore unmeasured, not inferred.

## 2. Compiled execution order

The record follows the vector card's own compiled program.

- Stages 2 and 3 first form their cross-level vertical velocity at
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:287-293`.
- The stage accumulator then receives HPG, VOR/EEN, and vector advection, in
  that order, at
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:318-332`.
- The vector-advection call is itself KEG followed by ZAD at
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:134-138`.
- Stage 2 applies its explicit velocity update at
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:363-370`.
- Stage 3 alone adds LDF and then applies the implicit ZDF solve at
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:387-405`.
- Both stages then apply the common barotropic/SPG correction at
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:408-419`.

That order fixes the record boundaries. No operator was selected by a model
configuration guess.

## 3. Acquisition implementation

The new `stage23` arm extends the existing VORTEX `run.sh`; it is not a second
trajectory harness. It creates a new paired target:

- uninstrumented: `VORTEX_VEC_R8_OMIP_L1`;
- instrumented: `VORTEX_VEC_R8_OMIP_L1_P3`;
- evidence: `round192/oracle_stage23_terms`.

The instrument adds calls only after the compiled statements above. It writes
one self-describing file for stage 2 and one for stage 3 at `kt=1`. Each file
contains the stage `Kmm` u/v/ssh, `ww`, the starting accumulator, every ordered
operator boundary, the pre-correction stage result, and the post-correction
output. Stage 2 requires exactly 18 named groups; stage 3 requires exactly 20.

The checker predicts neither byte size nor a complete header tuple. It parses
each `(name, rank, n1, n2, n3, payload)` group to EOF, derives the payload size
from that group's own declaration, and refuses an unknown, missing, duplicate,
or malformed group. The stage layouts are distinct: stage 2 requires
`update_u/v` and forbids LDF/ZDF groups; stage 3 requires `ldf_u/v` and
`zdf_u/v` and forbids the stage-2 update groups.

Passivity is the binding note-AS test: the instrumented step-10 NEMO restart
must be byte-identical to the simultaneously built uninstrumented restart.
The run also refuses a dirty legoESM worktree and records its exact Git SHA,
hashes both NEMO binaries and every source/tool input, verifies the compiled
writer calls, and rejects vector-math symbols. Every otherwise-unhandled shell
failure prints a named `REFUSE` line.

## 4. Preflight, syntax proof, and plants

The committed clean-tree preflight at `cc9f5eee3` prints:

> `GFORTRAN_SYNTAX_PASS .../vortex_r8_stage_terms.F90`
>
> `PREFLIGHT_OK variant stage23: instrument and deck patches apply to the shipped sources`

The full output is `round192/preflight.log`. The proof compiles the new writer
with gfortran's `-fsyntax-only`; it does not claim that NEMO was built in this
sandbox. The acquire arm verifies the calls again in the eventual build's
compiled `ppsrc` before running NEMO.

The parser tests cover both legal stage layouts, missing groups, a false group
count, duplicate names, and a corrupted first-group extent. The extent plant
raises a refusal. The acquisition runs the same plant against the real stage-3
record and accepts only its nonzero exit.

## 5. Rule table and trajectory disposition

| required scope | disposition this round |
|---|---|
| VORTEX-vector kt=1..10 | unchanged; no model/card code changed; first debt remains kt=2 with the values in section 1 |
| VORTEX-flux | unchanged; acquisition selects the vector deck only |
| GYRE ladder/month/year | unchanged at the round-191 certified arm; no shared implementation changed |
| DINO month | not triggered: no file under `packages/` or `src/` changed |
| LOCK_EXCHANGE / OVERFLOW | unchanged; no shared implementation changed |
| generic NEMO-GYRE recipe | unchanged; no recipe or model file changed |
| ORCA2 | UNMEASURED here and unchanged; this is an additive VORTEX-only writer |

No moved trajectory row exists to register. The current first-over-bar row is
not claimed improved, and no landing criterion is invoked.

## 6. Independent review

The required read-only `codex exec` pass was attempted against commits
`37d5d4659` and `cc9f5eee3`, with instructions to refute source order, record
coverage, passivity, layout, plants, and target naming. It emitted no
scientific verdict. Verbatim result:

> independent review unavailable in-sandbox — `Error: failed to initialize
> in-process app-server client: Read-only file system (os error 30)`

The complete terminal output is `round192/codex_review.log`. There is no
`SHIP` or `DO NOT SHIP` verdict to override.

## 7. Tests and citation gate

- Shell syntax, Python byte compilation, patch application, and Fortran
  syntax-only preflight: PASS.
- Focused VORTEX checker/walk plus receipt-citation suite:
  `46 passed in 2.69s`.
- Round-specific citation gate: PASS, six citations, zero failures, zero
  unmapped citations, and zero map-audit failures. Shifting
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:318-332` by two lines
  exits 1 with `SYMBOL-NOT-AT-LINE`.

## 8. Verdict

STOPPED_FOR_RECORD. The additive acquisition is ready and locally proven, but
the stage-2/stage-3 term values do not exist until NEMO is run. No physics or
configuration change is landed. The operator entry point is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_round8_stage_terms/run.sh`;
it binds `--variant stage23 --run` and delegates to the one shared VORTEX
harness.

## 9. OPEN — round 193

1. Run the committed `stage23` acquisition and admit it only if the paired
   restart is byte-identical and the real extent plant exits nonzero.
2. Compare the ordinary step/stage records with the existing vector record as
   an informational compiled-rounding check; note AS leaves restart identity
   as the binding passivity criterion.
3. From NEMO's own stage entry, substitute HPG, VOR/EEN, KEG, ZAD, LDF, ZDF,
   the `ww`/stage-velocity input, update, and common correction in compiled
   order under production JIT, with one live plant per arm. Name the first
   non-bit statement and its u/v magnitude.
4. If one statement is a candidate, land it only under Decisions
   43/45/55/59 with VORTEX, GYRE, generic-card, DINO-month, and tank coverage.
