# SI3 Ladder Round 25 (`si3_r25`) Preregistration — O4 Prather Acquisition and Exact-Input Score

## Scope frozen before measurement

This round builds the operator-only acquisition for rung O4. It does not build
or run NEMO. The operator will run the committed launcher after this round.

The selected program is the approved O4 row: `ORCA2_SAS_ICE`, five categories,
Prather advection on, all other dynamics and thermodynamics off
(`docs/ice/fidelity/si3_ladder_design.md:224`). DECISION 170 fixes the sole
deck overlay to `ln_dynADV2D=.true.` and `rn_uice=rn_vice=0.5` m/s. The record
uses one rank and the two ice steps `kt={1,2}`, which execute the two alternating
Prather sweep orders. No other namelist value changes.

The acquisition observes the executing `ice_dyn_adv_pra` call at its entry and
exit boundaries. NEMO passes the complete five-category state at
`icedyn_adv.f90:149-150`; the routine executes the x-then-y arm on the first
step and y-then-x on the second (`icedyn_adv_pra.f90:267-367`), exchanges every
transported field and all five moments across the cyclic seam and north fold
(`icedyn_adv_pra.f90:450-494`), and returns after restart bookkeeping at
`icedyn_adv_pra.f90:501-510`. Observers are outside every category, layer,
subcycle, and grid loop.

## Frozen record and score contract

For each of `kt={1,2}`, the record contains:

1. Exact declared extents and structural metadata (`jpi`, `jpj`, `jpl`,
   `nlay_i`, `nlay_s`, `nn_hls`, interior bounds, rank, step, boundary).
2. The nine horizontal metric/mask arrays read by the production Prather path,
   prescribed `u_ice` and `v_ice`, the current and delayed CFL values,
   `icycle`, `rDt_ice`, and the top HFN category bound.
3. Entry and exit values of `ato_i`, all ten three-dimensional transported
   inventories, the three layered inventories, the six residual/diagnostic
   accumulators, and all 55 Prather moment arrays. Full-halo arrays retain their
   real full allocation; `A2D(0)` accumulators retain their real interior-only
   allocation. No padding convention substitutes for a NEMO allocation.
4. The exit includes the completed fold/seam exchange. The scorer reports the
   full domain and separately the cyclic seam plus north-fold cells for every
   exit field, including every moment.

The scorer feeds each recorded entry to the production
`si3_dyn_adv_pra_orca2` path from round 8, with explicit ORCA2 configuration,
and compares every recorded exit at exactly zero ULP. Signed-zero differences
count as nonzero ULP differences. D127 applies only to a NEMO nonzero subnormal
versus a legoESM signed zero at its producing statement; such cells are counted
separately and do not excuse any normal-range difference.

The raw records, manifest, reference/record binary hashes, source hashes,
resolved deck, same-program proof, reference/record restart hashes, logs, and
resource usage are retained. Admission accepts only relative manifest paths and
requires the manifest's producer commit and both binary digests. Refused runs
and builds are retained; rerun requires the explicit move-aside mode.

## Predictions and falsifiers

### P1 — Source-derived coverage and real extents

Prediction: the shared NEMO read-set scanner derives every external array and
scalar read by `ice_dyn_adv_pra`; the O4 entry boundary covers all of them. The
shared allocation gate resolves every writer actual argument from the real
preprocessed `ALLOCATE` or declaration and accepts no alternate extent.

Falsifiers: removing one moment family, one metric, `hi_max`, or a residual
accumulator makes the read-set gate fail. Replacing an interior-only
`A2D(0)` allocation with `(jpi,jpj)`, changing a writer mode, or deleting a
source allocation makes the shared extent gate fail.

### P2 — Passive, correctly placed observers

Prediction: with NEMO's real `-O3 -funroll-all-loops -fno-tree-vectorize`
flags, the pristine and instrumented `ice_dyn_adv_pra` functions have identical
per-mnemonic floating-point instruction counts, and the pristine function has
a nonzero floating-point census. The normalized same-program proof finds only
the nonblank observer insertions between their markers. Entry precedes the CFL
calculation; CFL metadata follows `icycle`; exit follows the last fold/seam
exchange and precedes routine return.

Falsifiers: an injected floating-point add changes the census; moving either
boundary across its source-order bracket fails placement; adding an unmarked
statement or comparing against an instrumented baseline fails same-program.

### P3 — Reader, admission, and generated-driver controls

Prediction: the reader rejects bad magic/version, truncation, checksum drift,
nonfinite values, wrong step/boundary, missing frames, absolute manifest paths,
binary/producer drift, and a one-ULP change in a wet cell. The preflight creates
every script that `--run` can execute, checks each with `bash -n`, and exercises
success, refusal, retention, move-aside, and `--admit-existing` end to end with
a stub executable. The stub emits exactly the twelve files (six kinds at each
of two steps) named by the real observer list. Its analytic zero-transport
payload is independent of the production Prather path; its grid seed and
NEMO log/namelist outputs derive from the admitted O7 SAS record.

Falsifiers: each listed reader plant exits nonzero; a generated-script syntax
error exits nonzero; a stub-only fabricated filename or namelist fails the
observer-list gate; forced stub failure retains evidence and both builds.

### P4 — Exact production replay

Prediction: for both sweep orders, every normal-range cell in every exit field
and all 55 moment arrays is zero ULP from the production Prather replay. The
seam/fold subset is also zero ULP. Both steps execute different registered
sweep orders and have nonempty wet, seam, and fold comparison masks.

Falsifiers: inverting the production sweep order, changing any moment's
north-fold sign, omitting the subcycle boundary exchange, or changing one wet
exit bit makes the score fail. A scorer that omits any recorded exit field or
accepts an empty regional mask fails its schema census.

### P5 — Resource and retention gates

Prediction: raw records remain within the design's 10–20 GiB envelope, total
evidence remains below 25 GiB, peak preflight RSS stays below 20 GiB, and the
operator run finishes within one hour. The pre-run manifest exists before the
reference integration starts. No test allocates full acquisition records;
unit tests use small synthetic shapes.

Falsifiers: the launcher refuses before execution when free RAM is below
150 GiB, free disk is below 30 GiB, the schema estimate exceeds 20 GiB, the
evidence exceeds 25 GiB, elapsed time exceeds one hour, or the pre-run manifest
is absent.

## Review and choices

Independent review for this Codex-authored round is the operator (Claude), per
the round note. No nested Codex review will be run.

Choices made: none. The rung, deck, overlay, two steps, one rank, zero-ULP bar,
scored quantities, resource envelope, retention route, and operator-only run
are all fixed by the prompt, design, standing decisions, or the model-prep
record contract.
