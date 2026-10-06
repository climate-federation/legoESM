# NEMO testcase L4 ORCA2 round 158 receipt — exact cyclic and fold operands

Date: 2026-10-05

Status: **HELD** — the source-ordered U cyclic exchange and V north-fold
associations are now exact in the private arm; no production physics,
configuration, carried state, or sea-ice selector changed.

## Result

All numbers in this receipt are **independent**: the hierarchy rung-0 card
starts from its own climatological T/S, zero velocity, and zero sea surface.
The preregistration was committed as `290aea565` before measurement.

The admitted two-rank record proves the source program directly:

* all 520 rank/side/substep U halo comparisons are bit-exact, with maximum
  absolute difference 0;
* the compact U closure is exact on the 147 non-pivot points for all 65
  barotropic substeps; its pivot bit is not final at this boundary, because
  the later T-pivot fold overwrites it, after which it is exact on 65/65 rows;
* the V source row, permutation, sign `-1`, formula, and final 180-cell row
  are each exact on 65/65 substeps;
* U-cyclic-then-fold and V-cyclic-then-fold both compose exactly to the
  complete seven-field helper, and all seven observed state fields are
  array-identical to the unobserved step.

The source-ordered statement is NEMO's west/east send at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:1961-1979`, followed by
the blocking receive and halo write at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:2060-2068`. The north
fold is dispatched at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:2105-2113`.

For the no-gather fold, the compiled program selects the active extra lines
at `ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1561-1577`, prepares
the field-dependent exchange at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1629-1669`, selects the
V-point neighbour and applies `psgn` at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1712-1738`, then performs
the partial-line overwrite at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1747-1766`. The topology
table marks the T-pivot half-lines and the already-periodic edge cells at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/mppini.f90:1412-1440`.

This closes the two associations as transcriptions. It does not by itself
promote the complete seven-field association arm: round 157 showed that arm
eventually exposes a non-finite rung-0 trajectory, so the source-exact pair
must still pass the production ladder together before landing.

## Predictions

| prediction | verdict | measurement |
|---|---|---|
| R158-P1, literal two-rank west/east order | **CONFIRMED** | 520/520 comparisons exact; maximum 0 |
| R158-P2, U cyclic exact on the whole compact closure before fold | **REFUTED** | 147 non-pivot values exact, but the pivot differs only by signed zero on every substep and is overwritten exactly by the later fold |
| R158-P3, V source/permutation/sign/result exact | **CONFIRMED** | every operand and all 65 completed rows exact |
| R158-P4, complete and passive split | **CONFIRMED** | both compositions and all seven ordinary fields exact |
| R158-P5, production unchanged | **CONFIRMED** | rung 0 moves 0/200 rows and retains kt=1 stage-1 T as first debt |

The R158-P2 falsifier fired exactly as written; it is retained, not repaired
post hoc. The refined statement is that U cyclic is exact on every value it
owns at that source boundary, while the T-pivot operation owns the final
pivot bit.

## Gates and controls

The authoritative `halo_exact.json` has SHA-256
`daa083b148d676ef014cd78204f189c2d616499a881c9fcf75a1869508b5bdf9`.
The two rank shards are self-describing, 218,151,872 bytes each, carry 65
substeps, and have SHA-256 values
`d24710b945b007f8ee6823e0554274ca3d29c9f3d26e548b2694f275f4b7e7cf`
and `fe8a930a74f094a16f8d9bcab37e31b1a0c128fc2fbeb162e5b9cee8363cadb1`.

Seven independent plants fire and exit nonzero: U rank source, U pivot sign,
V source, V permutation, V sign, composition, and unknown selector. The U
rank plant moves one consumed value by one float64 step; the composition
plant moves exactly one completed value. Missing/wrong rank metadata is
covered by the admitted reader inherited from round 97.

The rung-0 production comparison is unchanged on 200/200 rows, has no
bit-identical loss, and keeps kt=1 stage-1 T first. Its comparison artifact
has SHA-256
`e30f82f3f434c01769bed2b5950a504790424ce7734ec7180d7a7156e0d113eb`.

The shared GYRE gate also proves the private helper is inert outside its
observer arm: 70/70 certified rows move by exactly zero, first debt remains
kt=3, and every array in the residual archives is `np.array_equal`. All 30
daily snapshots in the required member are byte-identical to round 157;
day-30 SHA-256 remains
`b017623dea468af4f4ba31761148aa52e478d60196828443b56ceeee36534180`.

No DINO, VORTEX, tank, generic-card, or rung-7 landing gate was run. This is
a preregistered measurement-only round with no production candidate; those
gates belong to the next round's combined landing attempt.

## Tests, citations, and review

Focused round-157/158 and boundary-association tests pass 22/22 before the
receipt. The one required `tests/ocean/fidelity -n 12` invocation is
**INCOMPLETE**, not PASS: it collected 2,637 tests, reached 99%, and showed
only the four already registered failures (SI3 scalar-math provenance, GYRE
round-129 spread-record stamp, round-35 escape scope, and worktree stamp).
After pytest exited the PTY again did not close or emit a summary; it was
interrupted once and was not rerun.

The default citation gate passes with 274 citations, zero failures, zero
unmapped citations, and zero map-audit failures before this receipt. Its
planted two-line shift fires with `SYMBOL-NOT-AT-LINE`. The cumulative
receipt and model citations were re-anchored mechanically with
`difflib.SequenceMatcher`; `reanchor_map.json` has SHA-256
`98887cda53d6045a29d2c3070926ace903a1929e722f1e1d3a02d76626b12ae4`.

The required separate verdict is **independent review unavailable
in-sandbox**. Exact failure: `failed to initialize in-process app-server
client: Read-only file system (os error 30)`. No independent PASS is claimed.

ASKED choices: none. UNASKED choices: empty. ACQUISITION_NEEDED: none.

## OPEN — round 159

1. Combine only these exact U-cyclic and V-north-fold associations into the
   complete seven-field private arm and score both independent ORCA2 ladders.
2. If the pair passes, land it together under rung-0/rung-7, GYRE year,
   DINO, VORTEX, tank, generic-card, citation, and push gates. If it exposes
   the prior non-finite salinity row, retain the source-exact attribution and
   name that exact red row rather than weakening the association.
3. This is the second round of the batch: if the halo landing is not complete,
   perform the pending GYRE-lane merge before any further halo walk.
