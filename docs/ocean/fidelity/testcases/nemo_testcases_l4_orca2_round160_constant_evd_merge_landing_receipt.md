# ORCA2 round 160 — constant-background EVD repair and merge qualification

Date: 2026-10-06. Base `427639e60`; preregistration `b3ad96f31`;
measurement tip `5a32fec1f`. Verdict: **LANDED**.

Every rung-0 number below is **independent**: the hierarchy card starts from
its own climatological T/S, zero velocity, and zero sea surface. Every rung-7
number is **given NEMO's entry** under the Decision-52 bridge. The two labels
are not mixed in a table. Sea ice, the six sea-ice selectors, and the card's
`unmeasured_features` tuple are unchanged.

## Result and compiled statement

Round 159 merged the GYRE lane through round 236, then correctly refused the
rung-0 card: its constant vertical closure plus NEMO-replacement EVD reached a
fast path that had already summed the two owners. This round repairs that
source-defined composition.

The compiled rung-0 program initializes the constant tracer and momentum
backgrounds in `ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfphy.f90:205-228`,
copies those closure fields into the live coefficients, and only then calls
EVD at `ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfphy.f90:347-359`. EVD
replaces tracer diffusivity at unstable interfaces in
`ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfevd.f90:107-110`; momentum is
replaced only when `nn_evdm == 1` at
`ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfevd.f90:120-135`. Rung 0 has
`nn_evdm=0`, so its constant momentum background remains live everywhere.

legoESM now routes this exact constant-plus-replacement combination through
the existing separated-profile path. It produces `K=100.0` at the planted
unstable wet interface, `K=1.2e-5` on stable wet interfaces, and
`A=1.2e-4` on every wet interface. The additive known-wrong control produces
`100.000012`, and deleting the route makes construction refuse.

The first implementation attempt exposed a second boundary and **REFUTED**
R160-P2 as originally worded: rung 0 refused because the NEMO NOW/BEFORE N²
operand was absent. The compiled RK3 sequence computes `rn2b` from the entry
`Nbb` tracers, copies it to `rn2`, then calls vertical physics at
`ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:172-181`. The final repair
therefore carries entry T/S for replacement EVD even when the vertical
closure is constant rather than TKE. The failed prediction is retained; it
was not silently rewritten.

## Independent rung-0 ladder

The canonical rung-0 gate passes all 200 rows. The first non-bit statement
remains kt=1 stage-1 T; no exact row leaves the bar and the first debt does not
move earlier. Against round 159's pre-merge control, 183 rows move and 17 are
unchanged. By RMS, 174 move toward NEMO and 9 away; by maximum absolute error,
119 move toward, 55 away, and 9 are equal.

At kt=10 stage 3, every headline RMS and maximum moves toward NEMO:

| field | RMS before -> after | maximum before -> after |
|---|---:|---:|
| T | `5.963034121233243e-3` -> `5.963034072069885e-3` | `0.8633902735293519` -> `0.8633902030298275` |
| S | `1.6088940285535393e-3` -> `1.6088940282822529e-3` | `0.4156673855360964` -> `0.4156673855238111` |
| u | `4.343087152949231e-3` -> `4.343087142442177e-3` | `0.36400841231870196` -> `0.3640084122829801` |
| v | `4.356804437803066e-3` -> `4.356804380110146e-3` | `0.514414600238386` -> `0.5144145634893218` |
| ssh | `2.8026523929348814e-2` -> `2.802652392771078e-2` | `0.4283251766668493` -> `0.42832517646246693` |

The gate artifact is `round160/rung0.json`, SHA-256
`09d9165b63c8cfe91f361a99eb2a8f8ec4bc2f114181175e1c40f63650b74d65`.

## Given-NEMO-entry rung-7 ladder

The Decision-52 ladder also passes all 200 rows. Its first non-bit statement
remains kt=1 stage-1 T, with no exact-row loss and no earlier debt. Against the
last pre-merge certified rung-7 artifact, 179 rows move and 21 are unchanged.
By RMS, 147 move toward and 32 away; by maximum absolute error, 112 move
toward and 67 away. The kt=10 stage-3 salinity veto improves from
`0.28803500879531185` to `0.28126856934547106`.

One large compensation is registered rather than hidden: kt=10 stage-3 V
maximum moves away from `0.4363833806287545` to `1.2990882244341995`, even
though its RMS moves from `5.33557178482116e-3` to
`5.584658859749724e-3` and the majority predicates above pass. This comparison
qualifies the round-159 GYRE merge plus the repair as one batch; the merged
tree could not produce a rung-0 control before this repair, so the row moves
are not assigned to the repair alone.

The rung-7 artifact is `round160/rung7.json`, SHA-256
`a3dd8b38041ab03014a728d2a777c72fadab3121afd3d23f9419eab805d22a49`.
The combined comparison is `round160/ladder_compare.json`, SHA-256
`e44b8a5d099c7d511f409192d261a04313a96a037595ecaa35da29a733c618c5`.
Its planted exact-row loss exits nonzero.

## Shared-model gates

GYRE's 70 certified ten-step rows are array-identical to the round-223 merged
reference: 210 named oracle/candidate/residual arrays differ on zero arrays,
and the first debt remains kt=3 for T/S/u/v/ssh. The in-gate comparison against
the 954-row full reference is retained as a **REFUSED, mismatched-scope
comparison** because that reference also contains private-arm rows; the
name-mapped certified-row comparison is the valid result.

The full 360-day member then reproduces all eight registered round-223
checkpoints byte-for-byte:

| day | snapshot SHA-256 |
|---:|---|
| 30 | `461f6647104ef9f2ad83dfe42b7a3fd5fd50c7489c0e1664290fd25871d6157a` |
| 60 | `1df73a4e5ba702a032bae059f39fc4ce828511036a1cb1ec2c070a4fa81f71f7` |
| 90 | `916a377bb556dc8f3d1ef8e43fd710cb4641349ba17e001b82f848a9351b570e` |
| 120 | `fa0ae4111bedaa0a0a1c38592c8cb8279440b49d8e22ff1973cc8aa915be3c4b` |
| 180 | `2606e52ca91fc103d019a031a8e00aef1d0eeb7a6521e640fd3c0c0f0e978a0a` |
| 240 | `6504303e83b5d94e344d3981c71686e08fd9c24fd576acd621bc0106aaf5510c` |
| 300 | `23c0f17c0d637d97cdda753e0c4126578f224010fb40fe690b4e95618922845a` |
| 360 | `0e11428690189139a9ba71910fa43bbc0ce3fdc2f20c5b27ba206a670fac9641` |

DINO's CPU/fp64 from-rest month passes: day-30 wet 3-D T RMS is
`2.053801168e-03 K` against the fixed `2.244317642e-03 K` bar. The planted
`6.981690958e-03 K` value fails. The score artifact SHA-256 is
`9e4e8005250eaf479d1ff6d7ca92471431cbc52b245cadb829bc1a42e387863d`.

The canonical eleven-card registry set is unchanged on 550/550 rows and
1,650 named arrays. It includes LOCK_EXCHANGE, OVERFLOW, three SMT cards, and
the flat scalar/vector VORTEX cards at all three resolutions. Every registered
first debt is unchanged. The generic shared-card construction gate also
passes. Thus GYRE, DINO, VORTEX, both tanks, and generic cards introduce no
new red.

## Prediction disposition

| prediction | disposition |
|---|---|
| R160-P1 separated composition matches the compiled statement | **CONFIRMED** by exact coefficient values and the additive known-wrong control. |
| R160-P2 rung 0 becomes executable | **REFUTED_FIRST_ATTEMPT, THEN_CLOSED**: the missing entry-level N² operand was the next refusal; the compiled RK3 association was transcribed and the ladder passes. |
| R160-P3 rung 0 keeps its exact rows and first debt | **CONFIRMED**, 183/200 rows moved, zero exact losses. |
| R160-P4 rung 7 keeps its exact rows and first debt | **CONFIRMED**, 179/200 rows moved, zero exact losses; the V maximum regression is registered above. |
| R160-P5 shared cards do not regress | **CONFIRMED**: GYRE ladder/year exact, DINO passes, and 550 cross-card rows are exact. |
| R160-P6 halo work remains separate | **CONFIRMED**: no halo hook is enabled or changed. |

## Review, tests, and citation controls

The separate `codex exec --sandbox read-only` review is **independent review
unavailable in-sandbox**: `failed to initialize in-process app-server client:
Read-only file system`. No independent verdict is claimed.

Focused tests pass 51/51 (two deselected). The one required
`tests/ocean/fidelity -n 12` invocation reached the slow tail after 99%, with
one failure marker, then made no progress for more than ten minutes and was
interrupted; it is **INCOMPLETE**, not PASS. A targeted rerun of the four
registered pre-existing red files produced exactly the known four failures
(SI3 scalar-math provenance, GYRE round-129 spread-record stamp, round-35
escape scope, and worktree-stamp scope) with 27 companion tests passing. No
round-160 test failed.

The default citation receipt passes with zero unmapped citations and zero
failures after both mechanical re-anchors. Its two-line-shift plant fails.
This receipt's compiled citations are separately gated, with its own plant.

ASKED choices: none. UNASKED choices: empty. No configuration choice,
stabiliser, carried-state policy, sea-ice selector, or NEMO source changed.

## OPEN

1. Re-run the rung-0 independent month on this qualified merged tree and name
   its new first non-finite boundary, if any; then resume the hierarchy's
   source-order growth walk.
2. Keep the round-159 U-cyclic/V-fold halo pair separate and HELD. It remains
   exact at its local boundary but failed the independent rung-0 salinity veto;
   this round does not reland or reinterpret it.
3. The rung-7 kt=10 stage-3 V maximum regression is registered above and must
   be included in the next magnitude ranking; do not hide it behind the
   majority-toward result.
