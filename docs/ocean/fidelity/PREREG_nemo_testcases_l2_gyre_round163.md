# Preregistration — round 163, landing the second continuity solve

Committed before any round-163 measurement ran.  This is a LANDING round, not
a discriminator round: the statement was built and fully measured in round
160, and Decision 55 (note AT, 2026-09-24) already answered the question
round 160 HELD on.  Round 163's job is to flip the production default, run
the gate the decision names, and register the numbers -- not to re-derive
them.

## What Decision 55 says, verbatim rule

A statement that is (1) NEMO's own, cited from the compiled ppsrc, (2) proven
one-variable, and (3) takes at least one certified trajectory row from DEBT to
AT-BAR, LANDS even if day-240 or day-360 T rms move by less than 1e-3
relative; the year change is REGISTERED with the 2e-10 K floor quoted and the
day-240 owners re-ranked in the landed arm.  Every other Decision 43/45 row
still binds (day-30 decreases, first-over-bar not earlier, kt1 at-bar rows
stay, moved rows registered, DINO/generic/tanks measured if shared).

Round 160's own numbers satisfy the relative-move test: day 240 moves
+9.98e-05 relative and day 360 +1.97e-04 relative, both under the 1e-3 bar.

## The flip

`nemo_stage_momentum_wzv_split` (the private one-variable arm
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` defines
next to `_NEMOWSRK3TestHooks`) changes its default from `False` to `True`.
`nemo_stage_momentum_wzv_executes(config, hooks=None)`'s `getattr` fallback
changes from `False` to `True` to match, so the census function agrees with
the model with no hooks passed at all -- which is every production card.  No
card constructs the private hooks object today (grepped: the only
`_NEMOWSRK3TestHooks(` call in `packages/` is the model's own
`_nemo_ws_test_hooks or _NEMOWSRK3TestHooks()` default), so this two-line flip
is the whole production change; the arm itself is kept, not deleted, because
an explicit `nemo_stage_momentum_wzv_split=False` is still the one-variable
way to reach the pre-round-160 single shared solve, and three of this round's
own structural tests use exactly that to prove the opt-out still works.

Card census (round 160's, re-verified from each card's resolved
configuration, not re-derived): the route needs the RK3-WS momentum
integrator, the vector-invariant momentum advection and the literal
continuity solve together.  GYRE-zco has all three and is the only card that
resolves it.  LOCK_EXCHANGE-zco and OVERFLOW-zps take the flux-form momentum
advection.  The generic NEMO-GYRE recipe takes the generic continuity solve.
Both DINO cards take the Euler momentum integrator, so the RK3 stage program
never runs.  This round re-runs the admission gate's `_card_execution` and
each card's own resolved-config check rather than trusting round 160's table.

## Expected numbers (round 160's measurement, now the production numbers)

| row | before (production, pre-flip) | after (landed) |
|---|---:|---:|
| developed stage-2 momentum vertical velocity vs oracle | 1.2326857e-08 m/s | 2.3346825e-13 m/s |
| GYRE-zco.kt2.before.u | DEBT (2.7377e-12) | AT-BAR |
| GYRE-zco.kt2.before.v | DEBT (3.2849e-12) | AT-BAR |
| first row over the bar | step 2 (u, v) | step 3 (T, S, u, v, ssh) |
| day 30 T rms | 6.888193513796918e-05 K | 6.572574374770603e-05 K |
| day 240 T rms | 1.6446718648e-02 K | 1.6448360701e-02 K |
| day 360 T rms | 1.1223450850e-02 K | 1.1225660019e-02 K |

The day-240 and day-360 moves are REGISTERED under note AT, not refused: both
sit under the 1e-3 relative bar (9.98e-05 and 1.97e-04) while thousands of
times the ~2e-10 K run-to-run floor in absolute terms -- both statements are
true at once and both are reported.

## Note-AT rows checked this round

1. Cited from compiled ppsrc: `stprk3_stg.f90:360` (velocity indicator) and
   `divhor.f90:126-130` (its continuity solve) for momentum, `traadv.f90:274`
   for the tracer re-solve -- re-quoted in the receipt, not just referenced.
2. Proven one-variable: round 160's own controls (split liveness both
   directions, tracer-identity plant, authority reproduction) re-run this
   round rather than assumed.
3. Takes a certified row from DEBT to AT-BAR: GYRE-zco.kt2.before.u and .v,
   re-measured this round on the full 70-row ladder.
4. Relative year move under 1e-3, floor quoted: re-measured, not copied from
   round 160, in case the flip's blast radius differs from the private-arm
   measurement (it should not, since the arm selected the identical code
   path, but the landing rule requires the number be re-run at the tip that
   ships, not the tip that measured it).
5. Every other Decision 43/45 row still binds: day-30 decrease, first-over-bar
   not earlier, no kt1 at-bar row leaves, every moved row registered, DINO
   measured (not executed, per census), generic/tanks measured.

## The full gate this round runs

Certified ladder (70 rows), day 30/240/360 T rms, the day-240 owner ranking
in the landed arm (note AB method: `year_owners.py --decompose` at the new
tip) reported alongside the pre-flip production ranking so the two can be read
side by side, the other cards' cheapest gates (generic GYRE recipe, both
tanks, both DINO recipes), the six-file push gate, and a Claude reviewer's
verdict quoted verbatim.

## Predictions, each with its falsifier

1. **The flip reproduces round 160's candidate-arm numbers exactly**, because
   it is the same code path selected a different way.  Stage-2 momentum
   vertical velocity 2.3346825e-13 m/s to every digit; day 30/240/360 T rms
   identical to round 160's `after` column.  REFUTED if any digit differs,
   which would mean the flip reaches a different code path than the private
   arm did.
2. **The 70-row ladder reproduces round 160's `moved_rows` table**: 60 rows
   moved, GYRE-zco.kt2.before.{u,v} go DEBT to AT-BAR, first-over-bar moves
   from step 2 to step 3, no kt1 at-bar row leaves.  REFUTED otherwise.
3. **The other cards do not move**: LOCK_EXCHANGE-zco, OVERFLOW-zps, the
   generic NEMO-GYRE recipe and both DINO recipes are bit-identical to their
   pre-flip numbers, because the census says none of them resolves the route.
   REFUTED if any cell moves on any of them, which would mean the flip's
   blast radius is wider than the census.
4. **The day-240 owner ranking names vertical diffusion as the largest term
   in the landed arm**, as it was in every prior ranking (rounds 123-128,
   152, 161) -- this is NOT the "exposure" reading round 162 refuted (a
   one-step cancellation mechanism); it is only the re-ranking note AT asks
   for, on the same instrument used before.  REFUTED if a different process
   ranks first, which is itself a finding to report, not a failure of this
   round.
5. **The push gate is green**: the citation gate plus the five push-gate test
   files, and this round's own updated structural tests (the round-160 file's
   three assertions that flip with the default).

## Before arm

The tip this round starts from is `b18cfc276e39`.  The lane's inherited year
rows before this round's flip are day 30 `6.88819351379691829e-05` K, day 240
`1.64467186440671112e-02` K and day 360 `1.12234508615602115e-02` K -- the
SAME numbers round 160 and round 162 used, because nothing between round 160
and this round changed a certified row (round 161 and 162 are both HELD with
zero files touched under `packages/`).

## Landing rule

Lands if: the flip reproduces round 160's numbers exactly (prediction 1), the
70-row ladder matches round 160's table (prediction 2), no other card moves
(prediction 3), every Decision 43/45 row still binds except the amended year
rows (which are registered per note AT rather than required to not-worsen),
and the push gate is green.  Any binding-row failure is a HOLD: restore
production, prove the ladder is bit-identical to the base commit (0 rows
moved), and report exactly which row failed.
