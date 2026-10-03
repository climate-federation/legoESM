# VORTEX round 22 (lane round 209) — the transport divisor's ASSOCIATION

**ROUND_STATUS: HELD.**  The transcription is correct and its pin is
non-vacuous, but it trips the per-cell ratchet on the flux card at ALL
THREE resolutions (largest worsening 20.5 / 17.5 / 17.5 row-scale ULP
against a 2-ULP bar) while buying NOTHING at row level: the flux card's
`kt=2` u and v are unchanged to ten significant figures.  Under the round's
own stop condition and Decisions 43/45/55/59 a red cellwise ratchet is a
HOLD, not a judgement call, so the change ships as
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l1_vortex_round209_reciprocal_divisor_held.patch`
and the lane tree keeps round 206's divide.

## 1. The statement, cited

`tests/VORTEX_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:270`, inside
`CASE ( np_LIN, np_HYB )` (`:267`), the arm the module default
`n_baro_upd = np_HYB` at `:48` selects:

```fortran
zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0(ji,jj) /(1._wp+r3u(ji,jj,Kmm))) - uu_b(ji,jj,Kmm)
zvb(ji,jj) = vn_adv(ji,jj)*(r1_hv_0(ji,jj) /(1._wp+r3v(ji,jj,Kmm))) - vv_b(ji,jj,Kmm)
```

NEMO MULTIPLIES by a reciprocal it stores once.  The operands:

| operand | compiled definition |
|---|---|
| `r1_hu_0` | `domain.f90:213`  `r1_hu_0(:,:) = ssumask(:,:) / ( hu_0(:,:) + 1._wp -  ssumask(:,:) )` |
| `hu_0` | `domain.f90:194,199`  `hu_0(:,:) = 0` then `hu_0 = hu_0 + e3t_1d(jk)*umask(:,:,jk)` over `jk` |
| `r3u(Kmm)` | `domqco.f90:266-267`, inside `dom_qco_r3c_RK3` (`:237-291`) — the `e1e2t`-weighted ssh mean times `r1_hu_0*r1_e1e2u`.  **NOT** the character-identical `:214-215`, which is the MLF `dom_qco_r3c` (`:188-234`); the reviewer caught that mis-citation and it is corrected here and in the held patch. |

Round 206 landed the algebraically equal DIVIDE by
`SUM_k e3u_0*(1 + r3u*umask)`.  Round 209 transcribes the association.

**Re-anchored per the citation rule, not carried over.**  The same
statement with the same `np_HYB` default is compiled into every build:

| compiled build | `n_baro_upd = np_HYB` | the `zub` statement |
|---|---:|---:|
| `tests/VORTEX_OMIP_L1_P3` | `stprk3_stg.f90:48` | `stprk3_stg.f90:270` |
| `tests/VORTEX_VEC_R8_OMIP_L1_P3` | `stprk3_stg.f90:50` | `stprk3_stg.f90:272` |
| `cfgs/GYRE_OMIP_L2_P3_SM` | `stprk3_stg.f90:49` | `stprk3_stg.f90:276` |
| `tests/LOCK_EXCHANGE_OMIP_L1_P3` | `stprk3_stg.f90:48` | `stprk3_stg.f90:273` |
| `tests/OVERFLOW_OMIP_L1_P3` | `stprk3_stg.f90:48` | `stprk3_stg.f90:274` |

So the change belongs in the SHARED RK3 path, exactly where round 206 put
the depth rule, and every card is measured rather than exempted.

## 2. The operand was verified before it was used

Note CA required the already-built reciprocal to be shown bit-faithful to
NEMO's before being wired in.  `vertical.py:243,253` build

```
r1_hu0 = wet_u / (hu_0 + 1 - wet_u)        # domain.f90:213, ssumask -> wet_u
r1_hu  = r1_hu0 / (1 + r3u)                # the parenthesised quotient of :270
```

over `hu_0 = SUM(e3u_0*umask)` (`vertical.py:653`, `domain.f90:194-199`).
Term for term, same masking rule, same two operations in the same order,
with no extra rounding step.

## 3. The two forms differ only in the last bits — MEASURED, not assumed

Probe `phase3/round209/probe_divisor.py`, run on each card's step-entry
state before the edit (`max |1/SUM_k e3u - r1_hu| / r1_hu` over wet U
faces):

| card | wet U faces | max relative gap |
|---|---:|---:|
| `VORTEX-zco` | 3660 | 4.065759e-16 |
| `OVERFLOW-zps` | 199 | 0.000000e+00 |
| `LOCK_EXCHANGE-zco` | 127 | 0.000000e+00 |

This REFUTES a prediction in the frozen preregistration.  P1 predicted that
cards with dry levels would see a REAL change, because
`e3u = e3u_0*(1 + r3u*umask)` returns `e3u_0`, not zero, on a dry level,
while `hu_0 = SUM(e3u_0*umask)` excludes it — so the summed depth would
pick up reference thicknesses below the seabed.  The independent reviewer
raised the SAME objection as a blocker, from the code, after the
measurement was in hand.  It is refuted by counting: on all three cards
there are **zero** dry levels with a non-zero `e3u_0`, because the card's
own reference ladder is already zero below the seabed.  So the two depths
agree exactly and the change is PURE ASSOCIATION everywhere.  Recorded as a
refuted prediction, and as the one reviewer blocker that measurement
overturned.

## 4. What it did to every card

Before and after at the SAME tip `90308df37`, `--max-step 10
--continue-after-first`, 50 certified rows per card, scored with
`nemo_testcase_offline_compare.py`.  Evidence under `phase3/round209/`
(`traj_<case>_{before,after}.json`, `ulp_<case>.json`,
`registry_summary.txt`).

**The before-baseline is itself controlled**: this round's own
`VORTEX-zco` BEFORE run compares to round 208's committed ladder at
`rows=50 max_worsening_ulps=0` — the baseline is the certified one, not a
re-measurement that drifted.

| card | rows moved | toward | away | ratchet | worst cell |
|---|---:|---:|---:|---|---:|
| `VORTEX-zco` | 24/50 | 17 | 7 | **FAIL** | 20.5 ulp |
| `VORTEX-15km-zco` | 28/50 | 9 | 19 | **FAIL** | 17.5 ulp |
| `VORTEX-10km-zco` | 24/50 | 10 | 14 | **FAIL** | 17.5 ulp |
| `VORTEX_VEC-zco` | 0/50 | 0 | 0 | PASS | 0.0 ulp |
| `VORTEX_VEC-15km-zco` | 0/50 | 0 | 0 | PASS | 0.0 ulp |
| `VORTEX_VEC-10km-zco` | 0/50 | 0 | 0 | PASS | 0.0 ulp |
| `LOCK_EXCHANGE-zco` | 0/50 | 0 | 0 | PASS | 0.0 ulp |
| `OVERFLOW-zps` | 0/50 | 0 | 0 | PASS | 0.00195 ulp |

**No row changed status.**  No DEBT became AT-BAR, nothing at bar fell to
debt, and every card's first-over-bar row is where it was.

The headline rows, which is why this is a HOLD and not a landing:

| row | before | after |
|---|---:|---:|
| `VORTEX-zco.kt2.u` | 1.216831066e-08 | 1.216831066e-08 |
| `VORTEX-zco.kt2.v` | 1.239432196e-08 | 1.239432196e-08 |
| `VORTEX-zco.kt10.u` | 9.953975759e-09 | 9.953974631e-09 |
| `VORTEX-zco.kt10.v` | 8.874215355e-09 | 8.874215990e-09 |

The `kt=2` pair — the row this whole walk is about — does not move at all.
What moves is the last bit of individual cells in later steps, in both
directions, and at 15 km and 10 km the majority of the moved rows move
AWAY.  The worst offenders are named in `ulp_VORTEX-*.json`; the three
largest at 30 km are `kt4.v` cell 17980 (5.000 ulp), `kt7.u` cell 1030
(3.752 ulp) and `kt5.ssh` cell 1554 (3.688 ulp).

The two tanks, which round 206 moved substantially, are inert here — as
section 3 predicted once it was measured: their two divisors are the same
BITS, so there is nothing for an association change to do.

## 5. GYRE, DINO and the rest of the gate

**WHAT LANDS: the receipt and the held patch only.**  No file under
`packages/` or `src/` changes in this commit, so GYRE and DINO are unchanged
by construction.  They were re-run anyway, on the landed tree.

| gate | result |
|---|---|
| GYRE certified ladder | **954 rows, 0 moved, `max_worsening_ulps=0`**, first-over-bar `{T,S,u,v,ssh}` at `kt=3` unchanged |
| ladder ratchet plants | `worsen-3ulp` exit 1, `at-bar-to-debt` exit 1, `improve` exit 0 — the comparison can fail |
| GYRE 360-day from-rest year | **byte-identical on all eight certified days** |
| GYRE snapshot digests | `day030 4e36c106403b495e…`, `day240 8b9cd60475626373…`, `day360 e3e0a068346c7866…` — note BZ's to every character |

| day | note BZ | this round | delta |
|---:|---|---|---:|
| 30 | `2.3432465132112266e-06` | `2.3432465132112266e-06` | 0 |
| 60 | `1.4793247973304582e-05` | `1.4793247973304582e-05` | 0 |
| 90 | `1.633271203963844e-05` | `1.633271203963844e-05` | 0 |
| 120 | `0.00010965907352116351` | `0.00010965907352116351` | 0 |
| 180 | `6.115335539300055e-05` | `6.115335539300055e-05` | 0 |
| 240 | `6.58170609494473e-05` | `6.58170609494473e-05` | 0 |
| 300 | `5.466049845187051e-05` | `5.466049845187051e-05` | 0 |
| 360 | `5.4077419367442036e-05` | `5.4077419367442036e-05` | 0 |

So note BZ's certified GYRE year stands and is not re-pinned.  DINO's month
gate runs inside `land.sh` against the pinned `2.053801168e-03`.

**GYRE UNDER THE HELD STATEMENT IS UNMEASURED, and that is said out loud
rather than left to be inferred.**  The GYRE gates have no `--allow-dirty`
flag and refuse a modified tree; committing the statement in order to
measure it is exactly what the HOLD forbids.  One attempt was made with the
environment escape `LEGOESM_GATE_ALLOW_DIRTY=1`, and when the model revert
landed mid-ladder it made that run's arm ambiguous from the outside, so it
was killed and discarded rather than quoted, and the gates were restarted on
the clean landed tree.  If the operator decides to land the held patch, the
ladder and the year must be run inside that round.  Open item 8.

## 6. The pin, and why it is not vacuous

`tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round209_reciprocal_divisor.py`
asserts two things.  First, that `1/SUM_k e3u` and the stored `r1_hu` agree
to 1e-14 relative and yet are NOT the same bits on this card — without
that, the second assertion could not fail for any reason.  Second, that
production multiplies the stored reciprocal, proven by substituting the
divide form into the kernel at the statement's own call site and requiring
the step to move.

**The first version of this pin was VACUOUS and the control caught it.**
Substituting the kernel globally also substitutes
`_nemo_ws_stage_transport`'s own reciprocal, which round 209 does not
touch, so the step moved on the reverted tree too and the test passed when
it should have failed.  The substitution is now filtered by caller frame to
`_step_impl`, the enclosing function of the `transport_target_u/v` lines.
Measured both ways: **2 passed** on the landed tree, and with the model
edit alone reverted, `assert 0.0 > 0.0` — `1 failed, 1 passed`.

## 7. Independent review

One fresh `code-reviewer` subagent on the diff, told to refute it.
**Verdict: DO NOT SHIP**, four blockers and seven minors.  The round was
already going to HOLD on the ratchet; the review decides whether the HELD
PATCH is right for a later attempt, so every finding was adjudicated by
measurement or by reading the compiled source, not by argument.

1. **"Not an association change — a RULE change on staircase cards."**
   REFUTED BY MEASUREMENT (section 3): zero dry levels carry a non-zero
   `e3u_0` on any of the three cards, so the summed depth IS
   `hu_0*(1+r3u)` and the gap is 4.07e-16 / 0 / 0.  The reviewer was right
   that the equality is not algebraic; it is measured, and the code comment
   now says so instead of implying algebra.
2. **`domqco.f90:215` is the MLF entry, not the RK3 one.**  TAKEN — correct
   and verified in the compiled source: `dom_qco_r3c` is `:188-234` and
   `dom_qco_r3c_RK3` is `:237-291`, `pr3u` at `:266-267`.  Fixed in the
   receipt, the code comment and the test docstring.
3. **`hu_0` is transcribed from the wrong operand.**  ACCEPTED AS A REAL,
   PRE-EXISTING FINDING, and NOT fixed here.  NEMO builds
   `hu_0 = SUM_k e3t_1d(jk)*umask` from the 1-D ladder (`domain.f90:199`);
   legoESM sums the min-rule FACE thickness `e3u_0*umask`
   (`vertical.py:656`).  Identical on a full-step `zco` mesh — which every
   card measured here is — and different at every partial cell, so it is
   a live question for DINO and ORCA2, not for VORTEX.  New OPEN item 6.
   The reviewer also found the companion docstring citation
   (`domain.F90:145`) is not a statement in 5.0.2: also pre-existing, also
   OPEN item 6.
4. **The dropped guard's zero set is the 3-D live mask, while the product
   is masked by the 2-D `u_mask`.**  REFUTED BY COUNTING on the cards that
   exist: zero faces have `u_mask = 1` and `r1_hu = 0`.  The reviewer's
   point that these are different objects stands, so the comment now says
   the agreement is counted rather than argued, and the test asserts the
   two masks are equal in BOTH directions.

Minors taken: the test's one-sided `r1_hu[~wet] == 0` assertion was true by
the definition of `wet` and proved nothing — replaced by the two-way mask
equality; the `vertical.py:243` citation now also names `:241` for
`wet_u`.  Minors recorded and NOT taken, each for a reason: the missing
`nemo_source_round` on `Hu_avg * r1_hu` would be a SECOND statement in one
diff (OPEN item 7); the test covering only `VORTEX-zco` is answered by the
section-3 count over all three cards; `_qco_tr_faces[0:4]` becoming unread
is dead-code-eliminated by XLA and keeps the call one line; and
`_transport_target_legacy`'s docstring is unaffected because the model
edit is not landing.

## 8. Choices made this round

| choice | ASKED / UNASKED |
|---|---|
| HOLD rather than land over a red cellwise ratchet | ASKED — the round's own stop condition and Decision 59 |
| Put the change in the shared RK3 path, not card-scoped | ASKED — note CA, and the branch table in section 1 |
| Keep the `* u_mask` factor and add no `nemo_source_round` | forced — one variable; adding a rounding barrier would be a second statement |
| Drop the divide's dry-column guard | forced by the statement: NEMO's `r1_hu_0` is exactly 0 on a closed face (`ssumask = 0`), and `r1_hu` is 0 on the same faces |

**UNASKED list: empty.**

## 9. OPEN, in order

1. **This statement, awaiting a decision.**  The held patch is correct and
   NEMO-literal.  It cannot be landed under the present ratchet because it
   perturbs cells by up to 20.5 row-scale ULP for no row-level gain.  The
   decision is the operator's: land it as a FIDELITY statement and register
   the cellwise moves as admitted noise, or keep the algebraically equal
   divide and close the item as transcribed-and-measured.
2. Carried from round 206: the time level of the transport target (once per
   step from the step-entry ssh here, per stage with `r3u(Kmm)` in NEMO).
3. Carried from round 206: why GYRE is byte-identical under this statement.
   Section 3 now explains the TANKS (their two divisors are the same bits);
   GYRE's zero is still measured and unexplained.
4. The flux card's `kt=2` residual, `1.216831e-08` u / `1.239432e-08` v,
   now with its last cheap named owner eliminated.
5. Carried from round 206: two pre-existing wrong citations in the gated
   receipt.
6. **NEW, from the review**: `hu_0` is built from the min-rule face
   thickness, not NEMO's 1-D ladder `e3t_1d` (`domain.f90:199`), and the
   docstring citing `domain.F90:145` names a statement 5.0.2 does not
   have.  Inert on every full-step card; a first-order multiplier of the
   transport at a partial cell, so this is a DINO/ORCA2 item.
7. **NEW, from the review**: the composition `Hu_avg * r1_hu` carries no
   `nemo_source_round`, while the sibling that forms the same product
   (`_nemo_stage_corrected_velocity`) does.  One statement at a time.
8. **GYRE under this statement**: see section 5.

## 10. VORTEX status at the stopping rule

**Where the two cards stand.**  Both cards are exact at `kt=1` (every field
at bar; `u`/`v` at 1-2e-16) and both first cross the bar at `kt=2`.

| | 30 km | 15 km | 10 km |
|---|---:|---:|---:|
| flux card, first over bar | `kt2` T/u/v/ssh | `kt2` T/u/v/ssh | `kt2` T/u/v/ssh |
| flux `kt2` u | 1.2168e-08 | 7.1166e-09 | 5.0191e-09 |
| flux `kt2` v | 1.2394e-08 | 7.3089e-09 | 5.1248e-09 |
| flux `kt2` T | 2.8233e-10 | 8.5748e-11 | 3.9930e-11 |
| vector card, first over bar | `kt2` u/v/ssh | `kt2` u/v/ssh | `kt2` u/v/ssh |
| vector `kt2` u | 1.3044e-15 | 1.6863e-15 | 1.5717e-15 |
| vector `kt2` v | 1.3357e-15 | 1.9106e-15 | 1.6943e-15 |

**What is at the bar.**  The vector card is AT THE BAR: its `kt=2` u and v
sit at 1.3-1.9e-15 against a 1e-15 bar — one to two units in the last
place of the oracle's own numbers, at every resolution, and its tracer rows
are at bar outright.  That milestone is met and the ladder confirms it is
not a single-resolution accident.

**What debt remains, and why it is not walked further.**  The flux card
carries 1.2e-08 at `kt=2`, four orders above bar, shrinking with resolution
(the ladder's p ~ -0.9 in velocity).  Round 204 put the whole of it inside
the flux-form momentum advection; round 205 split that; round 206 landed
the depth rule inside it, dropping the row by 9.3x (1.1355e-07 ->
1.2168e-08); round 208 proved it is resolution-independent in character;
round 209 eliminates the last cheap named owner — the divisor's
association is NEMO's, and it is worth under one part in 1e8 of the
remaining row.  There is no further item on this card that is both named
and cheap: what is left is a search inside the UP3 advection's own
arithmetic, which is a bit-walk, and the user's stopping rule retires those
while the row is invisible at a year.  The later-step T/S rows
(`kt10` ssh 1.23e-08, `kt10` u 9.95e-09) are RECORDED here and not walked,
for the same reason.  The VORTEX walk stops at these milestones.
