# ORCA2 round 145 — initial finite-growth statement

Date: 2026-10-04. Base: `9d8a8c901`. Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round145.md` at
`87381a64d`. Verdict: **HELD**.

Every science value below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity, and zero sea surface. Decision 52's
recorded NEMO entry is not used. The rung-7 ladder remains separately labelled
**given NEMO's entry**. The card, deck, sea-ice selectors, and
`unmeasured_features = ("linear_implicit_bottom_drag",)` did not change.

## Result

The existing round-144 record admits unchanged: 20 self-describing files cover
steps 1 through 10 on both ranks, 20 calibration restarts are byte-identical,
and all nine plants fire. Admission status is
`PASS_R144_INITIAL_GROWTH_RECORD`; JSON SHA-256 is
`5f9d66af9adfca2492b9e669cda7c3d6d9c038d174037890aba520fd7fb887b0`.

At `(j,i)=(87,159)`, step-1 `ssh_entry` and `r3t_entry` are bit-exact. The
mechanical first over-floor row is step-1 `ssh_after`:

| boundary | legoESM | NEMO | absolute error |
|---|---:|---:|---:|
| step 1 `ssh_after` | 0.007427686976947723 m | -0.0004113873760105851 m | 0.007839074352958308 m |
| step 10 `ssh_after` | 0.8128679887999751 m | 0.8479530808471768 m | 0.03508509204720167 m |

The step-10 result is the round-144 step-11 entry error, closing the two record
windows without a gap. All values remain finite and the instrumented and
ordinary states are array-identical at every step. The walk status is
`PASS_ROUND145_INITIAL_GROWTH_WALK`; JSON SHA-256 is
`1ff91267b300de68f97494cf2b8941bbf4ba06afb7776da12da649dbd1555b61`.

The current-tree round-94 replay makes depth averaging, drag, wind, final slow
forcing, and SSH forcing bit-exact globally. Its first non-bit output is the
split-explicit `ssh_after`; JSON SHA-256 is
`c88f528fd93e9236a9b453c55d046a09a36cb9cd10902f63ef1be661a5520411`.
The round-97 replay, with NEMO's recorded slow forcing and carried external
mode, makes every registered substep-1 boundary bit-exact; JSON SHA-256 is
`3395b2b7b5bb576459fb159fb697702777bb2e8e6676568c9b1363793e3ab995`.

The remaining first active boundary is substep-2 `continuity_du`: 64 cells,
maximum `205276.59075050754` transport units at `(j,i)=(37,0)`, followed by
132 unequal `after_ssh` cells with maximum `0.003203816535399729 m`. The first
full-domain precursor is the 68-cell northern-fold entry/carry seam. NEMO
updates U/V velocity, U/V live depth, both reciprocals, and SSH in one batched
T-pivot association after substep 1 in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`. This is the
first source statement that can feed the otherwise exact substep-1 result into
the non-bit substep-2 continuity stencil.

## One-statement arms

Two narrower readings of that call were retained as failed predictions, then
reverted explicitly:

1. Associating only the live V-face depth leaves all ten registered steps
   exactly unchanged. The canonical `.steps` payload has the same SHA-256,
   `484a4bbf8dd1f79c74c2fb912ee86bc82cf0d257ce75b1c72125082326e9b5ff`,
   before and after. Its round-129 replay JSON SHA-256 is
   `14fa61b44190e41413bc050b9fe07cac588e372dfcc28603e9c2189e8a76541b`.
2. Associating only the initial V reciprocal also leaves the active substep-2
   residual unchanged. The existing observer publishes that fold row through
   the active V mask, so it cannot claim the unmasked operand closed. Its JSON
   SHA-256 is
   `993a701fc7847c6c2a23da029d483f8047bc3dd6bf0b1312dc554cf86e07473f`.

The complete batched association is the already measured round-129 held arm,
re-tested after the shared face-thickness merge in round 132. It closes the
local 68-cell barotropic residual but exposes the registered `31.363` salinity
debt; splitting the call into independently landed pieces is not NEMO's
program. Therefore no physics statement lands in this round. The final tree
has no `packages/` diff from the base.

## Frozen prediction ledger

| ID | Verdict | Evidence |
|---|---|---|
| R145-P1 | CONFIRMED | Existing record admits unchanged; all nine plants fire. |
| R145-P2 | CONFIRMED | Step-1 `ssh_entry` and `r3t_entry` are bit-exact. |
| R145-P3 | CONFIRMED | First over-floor row is step-1 `ssh_after`, 0.007839074352958308 m. |
| R145-P4 | CONFIRMED | All registered values are finite through step 10 and every passivity comparison is exact. |
| R145-P5 | REFUTED | Existing streams name the batched fold association, but the observer masks the unassociated entry reciprocal and cannot split that operand faithfully. |
| R145-P6 | CONFIRMED | Neither narrow arm closes the bracketed boundary; both are reverted and the round is measurement-only. |

## Gates, controls, and review

The CPU run used fp64, libm transcendentals, x64-enabled JAX, and production
JIT. The clean baseline walk took 461.5127155780792 s; the live-depth arm took
440.18091583251953 s. No NEMO build or MPI run was attempted.

The final net model/card/deck diff is empty, so the certified GYRE, DINO,
tanks, rung-0, and rung-7 trajectories cannot move. The citation gate checks
the record build itself; shifting the cited span by two lines is a planted
failure. Focused round-145 and citation tests pass 23/23. The one required
`tests/ocean/fidelity -n 12` invocation collected 2,585 tests and reached 99%
before reproducing the registered xdist-controller tail stall; it is
**INCOMPLETE**, not PASS. It recorded 2,553 passes and four pre-existing reds:
SI3 scalar-math provenance, round-35 escape scope, worktree-stamp scope, and
the GYRE round-129 spread-record stamp. No round-145 test failed.

The required separate `codex exec --sandbox read-only` review could not
initialize its in-process app-server client because the read-only filesystem
prevented setup. Verdict: **independent review unavailable in-sandbox**.

ASKED choices: none. UNASKED choices: empty.

## OPEN

1. Rebuild the existing passive substep observer so it publishes the unmasked
   pre-association and post-association U/V velocity, U/V depth, reciprocals,
   and SSH for the single batched call; prove the observed restart is
   byte-identical to the unobserved restart before using it.
2. Re-run the complete call as one NEMO statement on the step-1..10 growth
   gate. It may land only if the rung-0 boundary closes without reviving the
   round-132 salinity debt and all shared-card gates pass.
3. The bottom-drag divisor and EVD composition remain downstream merge items;
   neither can own this earlier fold-association boundary.
