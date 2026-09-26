# Round 170 — the MPAS TKE surface-boundary red the merge left

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round170_mpas_tke_bc.md`
(commit `30ab7662d`, written before any edit existed).
Change: commit `8cc499c5c`.  Range `30ab7662d..HEAD`.

## Verdict

The red test was NOT a closure defect.  Both merge parents answer the NEMO
z=0 surface boundary IDENTICALLY; what they disagree about is the sea-ice
attenuation law, and `main` is the one that has it wrong.  No production
source, card value, default or configuration is touched by this round — the
diff is two test files — so no certified number can move, and none did.

Decision 60 is honoured: the mesh-based OMIP card keeps `main`'s
`nemo_z0`, unchanged, and the lane side is reconciled to it.

## The mechanism, in one paragraph

NEMO holds the surface turbulent energy at the z=0 water surface and solves
its tridiagonal system from the first interior level down.  Under that
boundary the shallowest coefficient this bridge returns therefore belongs to
the first INTERIOR w-level, which on the test's six-level fixture sits at
63.5 m.  From rest, with the wave-breaking and Langmuir inputs switched off
by ice, the turbulent energy there settles at NEMO's own minimum and NEMO's
background clamps
`MAX(zav, avmb)` / `MAX(zav, avtb)`
(`GYRE_OMIP_L2/BLD/ppsrc/nemo/zdftke.f90:684-685`) pin both viscosity and
diffusivity to the background value at every cell and level.  The previous
assertion compared two such clamped profiles, so it could only pass while the
surface value was held one w-level too deep — which is precisely the defect
`nemo_z0` removes.

## Diagnosis — the pre-registered discriminator, run both ways

The same read-only probe (the mesh card, the same fixture, the same 0.15 Pa
wind, the ice modes the test uses) was run in a CLEAN worktree of each merge
parent — lane parent `36cc29f0a`, main parent `d3f624841` — and on the merged
tip.

| arm | `nemo_z0`, mode 3 at quarter ice, first four interior levels of the first column |
|---|---|
| lane parent | `1.2e-04, 1.2e-04, 1.2e-04, 1.2e-04` |
| main parent | `1.2e-04, 1.2e-04, 1.2e-04, 1.2e-04` |

Identical, to the digit, including the tridiagonal row the boundary builds:
the surface coupling `-4.43975821e-02`, the diagonal `441.13881422`, the
right-hand side `1.34272768e-04` and the solved `1.30339568e-06` are the same
numbers in both trees.  That REFUTES mechanism M1 (a diverging closure
statement), which is what the merge receipt's OPEN entry had assumed.  The
merge round's own one-variable check — force the card back to
`interior_pinned` and the test passes — is consistent with BOTH mechanisms,
so it never discriminated between them.

The surface row itself was checked against NEMO rather than against the test.
NEMO's `zzd_lw` at the first solved level is
`-0.5*rn_Dt*MAX(avm(2)+avm(1),2e-5)/(e3t(1)*e3w(2))`
(`GYRE_OMIP_L2/BLD/ppsrc/nemo/zdftke.f90:414-415`); evaluating it by hand on
this fixture — surface viscosity `1.18946868e-02` from
`MAX(rn_ediff*zmxlm(1)*sqrt(en(1)), avmb)`, top cell 63.49206349 m, first
interior spacing 184.13 m — reproduces the `-4.43975821e-02` the code builds.
The boundary is NEMO's.

What DOES differ between the parents is the ice law.  NEMO's
`SELECT CASE (nn_eice)` reads
`CASE(1) zice_fra = TANH(fr_i*10)`, `CASE(2) zice_fra = fr_i`,
`CASE(3) zice_fra = MIN(4*fr_i, 1)`
(`GYRE_OMIP_L2/BLD/ppsrc/nemo/zdftke.f90:260,261,262`).  `main`'s mesh bridge
passed the RAW fraction for mode 1 — that is NEMO's mode 2 — while this lane
routes every mode through the shared transcription
(`nemo_tke_effective_ice_fraction`), which this lane added on 2026-09-05 in
`2efc0ce157`, "restore NEMO nn_eice1 attenuation".  Measured on the merged
tree with the correct law, a quarter ice cover attenuates 98.7% under mode 1
and completely under mode 3; on `main`'s tree the same quarter cover
attenuates only 25% under mode 1, which left its profiles far above the
clamp.  That is the whole reason `main`'s copy of this test could separate the
two modes and this lane's cannot.

Both parents also edited this test, and the union merge kept the lane's
assertion beside `main`'s card value:

| tree | what the test asserts about mode 3 at quarter ice vs mode 1 at full ice |
|---|---|
| main parent | `allclose(K3q, K1f, rtol=1e-12)` — they AGREE |
| lane parent | `not array_equal(K3q, K1f)` — they DIFFER bitwise |

Both claims are true of NEMO; neither is observable through a clamped
profile.  The lane's stricter form was added while the card still held the
surface value one level too deep.

## The reconciliation

No closure statement was edited; inventing one would have been a fit to the
test.  The two claims were moved to where the clamp cannot mask them.

- `tests/ocean/unit/test_mpas_tke.py` — the bridge test now pins what the
  bridge HANDS the kernel for each ice mode: exactly 1.0 for mode 3 at a
  quarter cover, and `TANH(2.5)` — not the raw 0.25 — for mode 1, strictly
  below one.  This is a bridge test in a bridge-contract class, and it is
  strictly stronger than either profile comparison it replaces, both of which
  are vacuous under the clamp.
- `tests/ocean/unit/test_tke_nemo_terms.py` — the shared-transcription test
  gains full ice in its bit-exact vector and the strict inequality
  `mode 1 at full ice < mode 3 at quarter ice = 1`.  That is the literal
  reading of `TANH` that no coefficient profile can carry.

## Non-vacuity — both directions shown

| plant | result |
|---|---|
| restore `main`'s raw-fraction mode 1 in the mesh bridge | `1 failed` — `assert bool(jnp.allclose(seen[1], math.tanh(2.5), ...))`, reported `0.9866142981514303 = tanh(2.5)` against the planted 0.25 |
| make mode 1 saturate at one in the shared transcription | `1 failed` — the bit-exact `TANH` vector comparison |
| make mode 3 double instead of quadruple the fraction | `1 failed` — `assert quarter_mode3 == 1.0`, `assert 0.5 == 1.0`, i.e. the NEW assertion alone |
| the change as landed | green, quoted below |

Every plant was reverted with `cp -f` and the tree verified with
`git status --porcelain`: only the two test files modified, no source file
left dirty.

## Gates

All run at `8cc499c5c`, binary64 on, CPU, one battery at a time.

| gate | result |
|---|---|
| `tests/ocean/unit/test_mpas_tke.py` (the red file, in full) | `37 passed, 6 warnings in 45.28s` |
| the GYRE push gate — the six files the autopilot pushes on | `135 passed in 966.30s (0:16:06)` |
| card gates, the same five files rounds 163 and the merge round ran | `160 passed, 9 warnings in 340.76s` plus `10 passed in 7.56s` = **170 passed**, the same count as both |
| receipt citation gate | `status PASS`, 274 citations, 0 failures, 0 unmapped, 0 map entries failing audit, every self-test plant fired, worktree `clean: true` |

## Invariants

**GYRE certified ladder — 70 rows, 0 moved.**  `git diff --name-only 30ab7662d..HEAD`
returns exactly

```
tests/ocean/unit/test_mpas_tke.py
tests/ocean/unit/test_tke_nemo_terms.py
```

Neither file is imported by any model, driver, card or recipe — nothing the
GYRE run executes changed, so the ladder digest `cf06a8fc7d0e90f2` and the
certified rows (day 30 `6.572574374770603e-05` K, day 240
`1.644836070117868e-02` K, day 360 `1.1225660018551306e-02` K) cannot move,
and the ladder was not re-run.  This is the "the code cannot move it" claim,
and for a diff with no executable file in it that is the stronger of the two.

**Other cards.**  DINO, both L1 tanks, the lock-exchange and the overflow
gates: 170 passed, unchanged.

**ORCA2 does not execute the changed line.**  Its resolved configuration
selects NEMO `nn_eice=1` and the recipe REFUSES anything else
(`nemo_testcase_recipe.py:1497`), so it runs the shared transcription that
this round did not touch; and both changed files are tests, which no card
imports.  GYRE itself sets `nn_etau = 0` in its own `namelist_cfg` and has no
sea ice, so it never evaluates the attenuation at all.

## Reviews — two, independent

**codex `exec --sandbox read-only` — SHIP, no findings.**  It read the
`nemo_z0` surface row and the shared ice-fraction transcription against the
compiled NEMO source and found no divergence; it judged the new assertions
non-vacuous and stronger at suite level, noting that end-to-end attenuation
stays covered by `test_eice_full_ice_attenuates_vs_eice0`; and it confirmed
mechanically that no card executes a changed line and that no production
source, card value, default or configuration changed.

**Claude code-reviewer subagent — SHIP WITH CHANGES.**  It re-derived the NEMO
citations itself and confirmed all of them verbatim, including the consumer
`MAX(0, 1-zice_fra)` at `zdftke.f90:364,500,506,514` against the code's
`jnp.maximum(0.0, 1.0 - ice_frac)`.  It checked the spy cannot KeyError
(`ice_frac` is keyword-only and always passed), cannot leak between
iterations (function-scoped fixture) and is not a pure mock (the real kernel
still runs.)  It named ONE required change and one accounting note:

| # | finding | disposition |
|---|---|---|
| 1 | the receipt carrying the cross-parent measurement was untracked, so the evidence that refutes the closure-defect hypothesis was not part of the diff | **FIXED** — this file is committed with the change |
| 2 | what is genuinely lost versus the pre-merge test is an end-to-end check that a wrong ice fraction moves the FINAL coefficients | **ACCEPTED, and it is the point**: that path is provably vacuous under this boundary on this fixture — both arms clamp to the background at every cell and level, identically on both merge parents — so the check it replaces could not fail.  The end-to-end direction is still covered by `test_eice_full_ice_attenuates_vs_eice0`, which compares full ice against no ice at all |

## Choices

ASKED (Decision 60, standing brief note AX): keep `main`'s `nemo_z0` on the
mesh-based OMIP card; reconcile the lane side; do not move the certified
numbers.  All three honoured.

UNASKED: one, offered for revert.  The brief's order was to find and fix a
diverging closure statement.  The pre-registered discriminator refutes that
there is one, so no closure statement was edited and the reconciliation lands
in the instrument instead.  Revert = leave the test red and carry the
question; inventing a closure edit that makes this test pass is the one thing
that is NOT on offer, because it would be a fit to the test.

## OPEN

- **DECISION_NEEDED.** The merge receipt's OPEN entry offered two ways to
  close this and recommended making the lane's closure "answer `nemo_z0` the
  way main's does".  That option does not exist: measured, the two closures
  already answer it identically, and `main`'s apparent difference came from
  its mesh bridge reading the raw ice fraction as NEMO mode 1.  Confirm that
  resolving the collision in the instrument, as done here, closes the item.
- `main` still carries the raw-fraction mode-1 mapping on its mesh bridge.
  This lane's merge already fixed it here, but until the lane lands, any
  `main`-side run selecting `nn_eice=1` attenuates by the raw fraction
  instead of `TANH(10*fr_i)`.  Worth a note to that card's owner.
- Carried forward, untouched: round 163's question of whether ORCA2 should
  take the second per-stage continuity solve.
