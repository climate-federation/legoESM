# GYRE round 203 — landing the shared stage-one tracer ratio order

**Date:** 2026-10-02
**Lane:** `fidelity/nemo-testcases-l2-gyre-codex2`
**Base:** `1f08d97dd` (round 202's HELD tip)
**Disposition:** **LANDED.** Round 202's transcription reproduces, on this
lane's own tip, bit-for-bit through day 360, so the certified GYRE numbers and
digests are re-pinned.

## The statement, and its citation

NEMO's RK3 stage 1 needs the thickness ratio `1 + r3t` at the stage level
`Kaa = N+1/3`. It does **not** interpolate the free surface and then form one
ratio from it. It first builds the *after-level* ratio arrays from `ssha`
(`GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:167`, the call
`dom_qco_r3c_RK3`), and only then interpolates the two endpoint ratios
(`:177`):

```
r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)   ! at N+1/3 (Kaa)
```

The whole block is `GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:160-179`,
unchanged from round 202's citation. legoESM transcribes that association in
`nemo_r3t_rk3_stage1_stretch` (`packages/ocean/legoesm/ocean/eos.py`), called
by the stage-one tracer weight `_qt_13` in
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py`. The other
three tracer weights (`_qt_b`, `_qt_12`, `_qt_aa`) are untouched. Adopted
originally from the ORCA2 lane commit `a0b2f7a5d` (Decision 59); taken and
refused exactly as round 202 itemized (same 3 files taken, same 5 ORCA2-lane
artefacts refused) — see that receipt's table, unchanged this round.

## What round 203 did

Round 202 transcribed this statement, measured it, and found that the
from-rest year reproduced the ORCA2 lane's OWN registered numbers only through
day 90, diverging from day 120 on — so it HELD and reverted the model hunks,
keeping the patch as
`manifests/nemo_testcase_l2_gyre_round202_ratio_order_held.patch`. The
operator's round-203 order changed the question: **not** "does this lane's
year match ORCA2's registered column" (a different, merged tree) but "does
this lane's year reproduce round 202's OWN measurement on THIS tip" — which is
a statement about this lane with itself, not a cross-lane claim.

Round 203 re-applied the held patch (renamed to
`..._held_LANDED.patch`), redid the mechanical citation re-anchor the patch's
inserted lines force (`ocean_model_latlon_cgrid.py` +5 lines after line 7032,
`eos.py` +34 after line 960 — no eos.py citation in the map crosses that
point), and re-measured everything from scratch.

## The certified GYRE ladder — unchanged

`nemo_testcase_l2_gyre_phase3_gate.py --max-step 10`, compared against round
201's certified report with `nemo_testcase_offline_compare.py`:

```
OFFLINE_ORACLE_RELATIVE_COMPARE PASS: rows=954 max_worsening_ulps=0
  first_over_bar={'T','S','u','v','ssh'} kt=3 -> unchanged
```

**954 rows, 0 moved, 0 ULP.** Identical to round 202's ladder measurement.

## The from-rest year — REPRODUCES, bit for bit

A fresh seed-0 member ran the full 2,160 steps and wrote all 360 daily
snapshots (`nemo_testcase_l2_gyre_year_fromrest.py --member 0 --days 360
--snap-steps 6 --tag r203a`, member
`phase3/year_fromrest/lego_seed0_r203a`). Scored with the committed day-gap
scorer (`nemo_testcase_l2_gyre_year_owners.py --day-gap`, same NEMO restarts,
same eight days as every prior round):

| day | round 202's AFTER measurement | round 203, member A | agree |
|---:|---:|---:|:--:|
| 30 | `2.3432465132112266e-06` | `2.3432465132112266e-06` | **yes** |
| 60 | `1.4793247973304582e-05` | `1.4793247973304582e-05` | **yes** |
| 90 | `1.633271203963844e-05` | `1.633271203963844e-05` | **yes** |
| 120 | `1.0965906837581848e-04` | `0.00010965906837581848` | **yes** |
| 180 | `6.115335288161464e-05` | `6.115335288161464e-05` | **yes** |
| 240 | `6.58170624837412e-05` | `6.58170624837412e-05` | **yes** |
| 300 | `5.466050113289237e-05` | `5.466050113289237e-05` | **yes** |
| 360 | `5.407736527246344e-05` | `5.407736527246344e-05` | **yes** |

Every one of the eight registered days reproduces to every printed digit,
including the three the round order required verbatim: day 30
`2.3432465132112266e-06`, day 240 `6.58170624837412e-05`, day 360
`5.407736527246344e-05` K.

**Snapshot digests, file-level SHA-256, this round's member A:**

| day | this round | round 202's registered digest | match |
|---|---|---|:--:|
| 030 | `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba` | same | **yes** |
| 240 | `a63befc30bf03b443e54a51a0dc0541a186a0d20106e3ba94133b1489f488534` | same | **yes** |
| 360 | `dcb7bc46c8bc75bd215b4752c8145b9025d00da38a6fdac8c6babd3cd6074899` | same | **yes** |

All three required digests match exactly. The round order's stop condition
("if any digit differs: do NOT re-pin") does not trigger: nothing differs, so
the re-pin proceeds.

## The determinism check — CONFIRMED, code not noise

A second, independent 360-day member ran on the same tip, same patch, same
seed (`--member 0 --days 360 --snap-steps 6 --tag r203b`, member
`phase3/year_fromrest/lego_seed0_r203b`). All 360 daily snapshot files were
hashed on both members and diffed:

```
360 files compared, 360 files compared, 0 differing SHA-256 lines
```

**Member A and member B are byte-identical on every one of the 360 daily
snapshots.** This legoESM lane's run is deterministic given a fixed tip and
seed. Per the round order, this means: **the gap between this lane's
post-adoption year and the ORCA2 merged tree's registered column (round 202's
open question) is a tree/code difference, not run-to-run noise.** Round 202's
two candidate explanations are therefore resolved in favor of the first
(different merged-tree content), and the second (nondeterminism) is excluded
by direct measurement, not argument.

## Re-pin: the three moved rows, registered

Comparing this lane's own PRE-adoption certified values (round 201's tip, carried
through round 202's BEFORE column) against this round's POST-adoption
measurement, floor = `2e-10` K (Decision 59):

| day | pre-adoption certified | post-adoption (this round) | Δ (floor units) | direction |
|---:|---:|---:|---:|:--|
| 30 | `2.3432510206121264e-06` | `2.3432465132112266e-06` | `-0.023` | toward NEMO |
| 240 | `6.581707093530567e-05` | `6.58170624837412e-05` | `-0.042` | toward NEMO |
| 360 | `5.407735418221895e-05` | `5.407736527246344e-05` | `+0.055` | away from NEMO |

All three are far inside Decision 59's ten-floor-unit admission band. Days 60,
90, 120, 180, 300 do not move (certified = post-adoption to every printed
digit — they were already past floor resolution, or move less than 0.01 floor
units).

**Non-vacuity — the old pin fails against the new year.** Round 202 already
measured this directly on the identical post-adoption member (file-identical
to this round's member A, confirmed by the digest match above):
344 of 360 daily snapshot files differ from the certified pre-adoption carried
arm (`phase3/round202/gyre_year_byte_identity.txt`). A re-pin against numbers
that already match to every printed digit would be vacuous only if the old
and new pins were indistinguishable; they are not — 96% of the year's daily
states move.

**Re-pinned.** Grepped the repo for the old values
(`2.3432510206121264e-06`, `6.581707093530567e-05`,
`5.407735418221895e-05`) and old digests (`95f336ef7ef9e1e0`,
`018b75e12bc52968`, `446a9041887a2c85`): every hit is either (a) this lane's
own historical receipts recording what was true when they were written
(rounds 6, 7, 198, 199, 201 landing receipts — not touched, per the round
order's own carve-out), or (b) VORTEX PREREG documents that pre-register a
check for a round that has ALREADY RUN (rounds 7, 193, 196, 197, 201 — frozen
commitments, not live tables; not touched). No test, gate, or script in
`tests/` or `scripts/` hardcodes these values — the lane's gates always
compute fresh and compare against a named `--output` file, never a baked-in
constant. **This receipt is the re-pin**: it is the artifact every subsequent
round cites as "the certified GYRE numbers" (the same convention round 202
used citing round 201, and round 203 used citing round 202). The next GYRE
round's PREREG will quote these numbers when it is drafted; none exists yet
to update.

## Registered fact: the cross-lane sub-floor discrepancy

Unchanged from round 202, carried forward as a registered fact rather than
re-measured (re-measurement would be redundant: member A is file-identical to
round 202's measurement, confirmed above). Versus the ORCA2 merged tree (a
different tree that carries lane content this tip does not), day 360 differs
by `5.4077419367442036e-05` (ORCA2) vs `5.407736527246344e-05` (this lane) =
`0.27` floor units — inside the floor, not a defect, and not the same
comparison as the re-pin above (which is this lane against itself).

## The other cards — all inert

| card | reference | result |
|---|---|---|
| `VORTEX-zco` (flux) | round 201 | **PASS**, 50 rows, **0 moved**, `max_worsening_ulps 0`, `first_over_bar` unchanged `{T,u,v,ssh} kt=2` |
| `VORTEX_VEC-zco` (vector) | round 201 | **PASS**, 50 rows, **0 moved**, `max_worsening_ulps 0`, `first_over_bar` unchanged `{u,v,ssh} kt=2` |
| `LOCK_EXCHANGE-zco` tank | round 199 | **PASS**, 50 rows, 0 moved, `first_over_bar` unchanged `{u} kt=4` |
| `OVERFLOW-zps` tank | round 199 | **PASS**, 50 rows, 0 moved, `first_over_bar` unchanged `{T,u} kt=2` |

All four gates run with `--max-step 10 --continue-after-first`, matching the
reference reports' own step count (50 rows = 10 steps x 5 fields). Both VORTEX
registries are 0 of 50, as round 202 found, and the ratchet is green on both.

## Gates

| gate | result |
|---|---|
| GYRE certified kt=1..10 ladder vs round 201 | **PASS**, 954 rows, 0 moved, 0 ULP |
| GYRE 360-day from-rest year, member A vs round 202's measurement | **PASS**, all 8 registered days + 3 required digests reproduce exactly |
| GYRE determinism check, member A vs member B | **PASS**, 360/360 snapshots byte-identical |
| `VORTEX-zco` / `VORTEX_VEC-zco` certified 50-row registries | **PASS**, 0 of 50 each |
| `LOCK_EXCHANGE-zco` / `OVERFLOW-zps` tanks | **PASS**, 0 of 50 each |
| cellwise two-ULP ratchet plants (GYRE ladder) | `worsen-3ulp` exit 1 (`max_worsening_ulps=3`), `at-bar-to-debt` exit 1 — both red on the pair the unplanted run passes |
| receipt citation gate, `DEFAULT_RECEIPT` | **PASS** exit 0, 274 citations, `unmapped_citations []`, 0 failures, 0 blind map entries, all self-test plants fired |
| generic NEMO-GYRE recipe gate + push battery | run by `land.sh` |
| DINO from-rest month gate | run by `land.sh` (reference `2.053801168e-03` K, bar `2.244317642e-03`) |

Evidence, all under `phase3/round203/`: `gyre_ladder_after.{json,log}`,
`gyre_ladder_compare.{json,log}`, `gyre_year_r203a.log`, `gyre_year_r203b.log`,
`gyre_day_gap_r203a.{json,log}`,
`traj_{VORTEX-zco,VORTEX_VEC-zco,LOCK_EXCHANGE-zco,OVERFLOW-zps}_after.{json,log}`,
`ulp_*.json`, `ratchet_plants.txt`, `citations_round203_receipt.json`,
`gyre_year_byte_identity_AB.txt`; member directories
`phase3/year_fromrest/lego_seed0_r203{a,b}/day*.npz` (360 files each, hashed
in place).

## Non-vacuity

* **The ULP ratchet plants are red on the very pair the unplanted run
  passes**: `worsen-3ulp` reports `max_worsening_ulps=3` and `at-bar-to-debt`
  reports a crossed row, both exit 1.
* **The citation gate's self-test plants all fired**, and the gate found 0
  unmapped citations only after this round's re-anchor (see below).
* **The unit control refuses the old order** (unchanged from round 202):
  the adopted test builds the SSH-interpolated-first value explicitly and
  asserts the new helper does not equal it.
* **The re-pin is not vacuous**: 344 of 360 snapshots differ from the old
  (pre-adoption) pin (round 202's direct measurement on the file-identical
  member).
* **The determinism check is a direct measurement, not an argument**: two
  independently-run 360-day members, same seed, same tip, compared snapshot
  by snapshot — 0 of 360 differ.

## The citation re-anchor this round needed (correcting round 202's reversion)

Round 202's HOLD commit (`842574957`) reverted the model hunks **and** the
citation re-anchor **and** the citation-map/prose shifts that re-anchor had
made, back to round-201 line numbers. Re-applying the held patch this round
reproduced the same line shift, so the map and prose needed re-anchoring
again. Replaying round 202's own re-anchor commit (`3913816e6`) cleanly fixed
the `CITATION_MAP` half (committed first), but the citation gate still
reported 7 unmapped prose citations
(`ocean_model_latlon_cgrid.py:{9221-9223,7582-7585,7041,8147,8512,7508,7527}`)
— the OTHER half of that same commit, 12 prose spans in
`nemo_testcases_l2_gyre_phase3_round8_receipt.md`, had not been replayed yet.
Applying that half (`git diff 3913816e6^ 3913816e6 -- docs/.../round8_receipt.md`)
brought `unmapped_citations` to `[]`. Both halves are a uniform `+5` shift
(all seven numbers are past line 7032, the end of the second of the two
inserted hunks), confirmed correct by re-reading the cited anchor text at the
new line numbers before committing.

## Independent review

One fresh `code-reviewer` subagent, no context from this round, on the diff
only (`git diff 1f08d97dd HEAD`). Reviewed the helper's arithmetic, the call
site's argument wiring, autodiff/jit safety (including the dry-cell
`jnp.where` NaN-safe-select pattern), the unit test's non-vacuity, and spot
checked the citation re-anchor.

**Verdict: APPROVE WITH NOTES.** The helper's arithmetic matches its
docstring, the call-site wiring (`_zc`, `state`, `state_new` all exist in
scope with the stated meanings) is correct, the new test is genuinely
non-vacuous (the reviewer independently computed both orderings outside
pytest and confirmed the "wrong" SSH-interpolated-first value differs from
the correct one by exactly 1 ULP for the test's chosen numbers — a real
bit-level distinction, not a vacuous assertion), the dry-cell NaN-safe-select
pattern is correctly applied (confirmed dry-cell output is exactly `1.0`
with gradient exactly `0.0`), and the 274-citation re-anchor is complete (19
of 27 `ocean_model_latlon_cgrid.py` citations at or past line 7018
spot-checked directly against current file content, all correct at the new
+5 line number; the patch rename confirmed 0 content diff via
`git diff -M100%`).

One real, non-blocking finding: `nemo_r3t_rk3_stage1_stretch` has **no floor**
on its returned stretch factor, unlike its sibling `nemo_r3t_stretch`
(`jnp.maximum(1.0 + r3t, 1.0e-6)`), and the reviewer confirmed this is live
(not theoretical) by calling the function directly with eta driven
pathologically negative and getting a negative stretch factor back, wired as
a literal divisor (`_stage()`, `ocean_model_latlon_cgrid.py:2230-2233`) on the
certified GYRE path. The certified from-rest year never drives eta that
negative, so nothing broke in this measurement — but the function has no
guard against the exact "silently wrong, not NaN" sign-flip class this
codebase has been burned by before. **Not fixed this round**: the round order
is "the transcription and the pin are the only changes," and this exact gap
was already flagged by round 202's independent reviewer and carried as OPEN
item 4 there; it stays OPEN item 1 below rather than being folded into this
landing as an unasked second change.

Also noted, not acted on: the duplicated `eta * (1/depth)` ratio formula is
re-derived inline rather than calling `nemo_r3t_stretch` twice (plausibly
necessary — reusing the sibling would floor each endpoint before blending,
not bit-identical to the raw blend NEMO's source computes — but the diff does
not say so).

## Choices made this round

| choice | ASKED / UNASKED |
|---|---|
| Re-apply the held patch and re-measure, rather than transfer ORCA2's numbers | the round order named exactly this |
| Treat member A (this round's fresh run) as the reproduction check AND as "member 1" for the determinism check, per the round order's explicit instruction not to run it twice | the round order's own instruction |
| Use `--max-step 10 --continue-after-first` for all four card gates (the default `--max-step 3` without `--continue-after-first` only produces 10 of the certified 50 rows) | mechanical; matched the reference reports' own row count, no scheme/threshold changed |
| Treat PREREG documents and historical landing receipts citing the old numbers as out of scope for re-pinning | the round order's own carve-out ("not historical receipts' own measurements") |
| Carry the cross-lane ORCA2 fact forward without re-measuring it (member A is file-identical to round 202's measurement) | avoids a redundant ~25-minute run; the identity is itself verified by the digest match |

**UNASKED list: empty.** No default, scheme, bound, threshold, deck value,
card line, or carried state was changed beyond the one transcribed statement
round 202 already specified and the mechanical citation re-anchor it forces.

## OPEN, in order

1. **`nemo_r3t_rk3_stage1_stretch` has no `1e-6` floor** on its returned
   stretch factor, unlike its sibling `nemo_r3t_stretch`; this round's
   reviewer confirmed it is live (produces a negative stretch factor under a
   pathologically negative eta) and reachable as a literal divisor on the
   certified GYRE path, inert only because the certified year never drives
   eta that negative. Flagged by round 202's reviewer too (its OPEN item 4);
   not fixed here because the round order scoped this landing to the
   transcription and the pin only.
2. The untranscribed `r3u`/`r3v` stage-one ratios (plausibly inert on the
   vector-invariant GYRE arm, not confirmed) — round 202's other open note.
3. Round 204 / VORTEX round 18 = the flux card's stage-1 RHS split (per the
   standing brief's next-item note).

## UNVERIFIED

* Why the ORCA2 merged tree's post-adoption year differs from this lane's by
  up to 0.27 floor units at day 360 is still not investigated (it is inside
  the floor, so not a defect, but the specific inert-on-old / live-on-new
  commit that causes it is not named).
* The day-gap scorer was run for eight days only, the days NEMO has restarts
  for; no claim is made about any other day.
