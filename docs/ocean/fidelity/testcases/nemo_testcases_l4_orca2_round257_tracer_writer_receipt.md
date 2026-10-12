# ORCA2 round 257 — physical-half tracer-writer receipt

Date: 2026-10-12. Base: `1f5678ebb`. Measurement commit:
`df9685ac40e27e60ea401e41c57c032fd9386e46`. Disposition: **HELD**.

This round changes no model, card, deck, selector, carried-state contract, or
sea-ice declaration. It is an offline, CPU fp64/libm production-JIT
measurement on OMT-4 with the complete private fold/transport unit. Every
number below is reported separately as **independent** or **given NEMO's
entry**; the two labels agree in value.

## Reconciliation of rounds 249 and 256

Round 249 measured tree `5ebe69712`; round 256 measured `f29c9ff8e`. They used
the same two labels, three-row fold band, physical T-pivot half, and named
private hooks, but not the same executable. Between those trees the unit
gained resolved slow-forcing masks (`920f52b3f`), the literal drag addition
(`4a2195086`), and its selector-scope repair (`ca909a37a`). The round-249
"T/S exact" row and round-256 `0.17735225873138316 K` /
`3.283688696376231 PSU` row therefore do not conflict. On the current tree,
round 257 reproduces round 256 to every printed digit under both labels. The
current-tree round-256 row stands; the old tree's exact row remains historical.

## Compiled program and instrument

The executing record build is
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo`. NEMO writes its stage metric U/V
transports in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`, then the
stage-1 CEN2 path writes the U/V tracer face products and their horizontal
divergence in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/traadv_cen.f90:150-160`. The
completed stage subsequently associates U, V, T, and S at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802`.

The committed gate reuses the admitted round-230 rank-complete fold record,
the admitted stage tracer record, the production literal CEN2 helper, and the
existing passive completed-stage trace. Traced and untraced completed states
are array-identical. The offline helper reproduces the production CEN2 result
within `5.421010862427522e-19`, below the `2e-10` floor. NEMO's unused
after-advection work slots below bottom are non-finite in the record; they are
reported by the JSON but never used for the finite-value owner claim. The
completed stage is finite everywhere.

## Source-ordered result

The table is the physical half of the T pivot (`j=147`, `i=0..89`,
`k=0..29`). `unequal` includes signed-zero bit differences; the owner is the
first row with a nonzero finite magnitude. Values are identical under both
labels except the noted signed-zero counts.

| statement | unequal: independent / given entry | max abs | argmax `[i,k]` |
|---|---:|---:|---:|
| entry T | 1,801 / 0 | 0 K | — |
| entry S | 0 / 0 | 0 PSU | — |
| metric `zFu` | 0 / 0 | 0 m3/s | — |
| metric `zFv` | 668 / 668 | 339,003.55099822127 m3/s | `[49,25]` |
| metric `zFw` | 668 / 668 | 2,210,212.2824837444 m3/s | `[49,0]` |
| U face product T | 1,768 / 0 | 0 | — |
| V face product T | 2,410 / 668 | 328,646.63273938664 | `[49,25]` |
| U face product S | 0 / 0 | 0 | — |
| V face product S | 668 / 668 | 11,846,933.212118356 | `[49,25]` |
| horizontal T divergence | 2,685 / 2,685 | 1.8064924491158724e-7 K/s | `[38,3]` |
| vertical T divergence | 2,685 / 2,685 | 1.0333358509465832e-7 K/s | `[38,4]` |
| after advection T / S | 2,698 / 2,695 | 4.9264481114847534e-5 K / 9.121350972737249e-4 PSU | `[49,0]` |
| after SBC T / S | 2,698 / 2,695 | 4.9264481114847534e-5 K / 9.121350972737249e-4 PSU | `[49,0]` |
| QCO stage T / S | 668 / 668 valued | 0.17735225873138316 K / 3.283688696376231 PSU | `[49,0]` |
| post-stage association | UNMEASURED here | — | round 256 already named it |

Thus the first numerical statement is NEMO's stage-1 V transport `zFv`, not
the later CEN2 multiply or a double tracer update. On all 668 unequal finite
slots the candidate transport is zero while NEMO's is nonzero: ratio
min/median/max is signed zero and 647 slots also differ in sign bit. This is
the fingerprint of the missing pivot-row V transport, not an arithmetic-order
residual in `0.5*pV*(T_j+T_j+1)`.

The causal control replaces the admitted `zFu/zFv/zFw` triplet only for the
stage-1 tracer consumer. Under **independent**, after-advection T/S then have
zero finite-valued maximum difference and the completed QCO stage is 0/0 bits
unequal. Under **given NEMO's entry**, the same maxima are zero and completed
S is 0/0 bits unequal; T differs only in 1,801 signed-zero bits. Therefore the
transport substitution closes the `0.177352 K / 3.283689 PSU` physical-half
error exactly in value under both labels. R257-P1 through P6 are confirmed.

## Controls, gates, tests, and review

All seven classifier plants refuse: missing label, wrong support, reordered
source registry, failed round-256 reproduction, false earlier owner, failed
recorded closure, and disabled one-bit sensitivity. The classifier status is
`HELD_R257_TRACER_WRITER_NAMED`; evidence SHA-256 values are:

- independent: `da1f0d25049bcf24993b8cc246d5e4b0efcea15907717edced61b40df945389b`
- given NEMO's entry: `e5870d9739a523e2e26e0cad450db3b2fa5441dcc3eb9416167a146e76d70c95`
- classification: `b8029c4c0df6de273bcf58e6fc5c858648707b21ea44567e9b413b51f82f3aa5`

This round has no `packages/` diff, so no OMT, rung-0, rung-10, GYRE, DINO,
or tank trajectory can move and their landing gates are not activated. The
focused round-231/249/256/257 gate set passed `29/29`; the citation-gate unit
set passed `17/17`. The round receipt citation gate passed all three compiled
anchors with zero failures or unmapped citations, the cumulative default
receipt passed with zero failures or unmapped citations, and a rigidly shifted
`stprk3_stg.f90:282-285` plant refused with exit 1.

The once-only `tests/ocean/fidelity -n 12` battery collected 3,337 tests and
reached 99% before an xdist worker stopped producing output; the stale run was
terminated once and was not repeated. Its preserved log SHA-256 is
`84ba589d59044e7b74098f7719904bfa76ab5e7b5f7688c8a73ae8ac58d45b6f`.
Four observed reds reproduced serially in isolation (`4 failed in 7.77 s`;
isolated-log SHA-256
`344ae50ac9fce34fcbf03c4a471b19c551a5981ac6f650199461a525ea1a3dd4`):

- `test_nemo_si3_scalarmath_v2_gate.py::test_full_v2_gate_and_plants` — known
  SI3 scalar-math provenance red (`A MY_SRC is not verbatim`);
- `test_nemo_testcase_l2_gyre_round129_spread_floor_gate.py::test_record_backed_gate_passes`
  — known registered-record/harness-stamp red;
- `test_nemo_testcase_round35_stamp_scope.py::test_every_driver_that_arms_the_escape_scopes_it`
  — known allow-dirty scope ratchet; and
- `test_nemo_testcase_worktree_stamp.py::test_every_report_emitter_stamps_the_worktree`
  — known worktree-stamp ratchet.

The requested separate `codex exec --sandbox read-only` invocation failed
before review with `failed to initialize in-process app-server client:
Read-only file system`; **independent review unavailable in-sandbox**. Its
preserved stderr/stdout SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN

Decision 116 remains pending and is not re-emitted. The next round must split
the now-named `zFv` writer's already-recorded operands in NEMO's written order:
`e1v`, live `e3v`, `vv(Kmm)`, `zvb`, and `vmask`. It must use the current
production D114 route and distinguish the private unit's zero transport from
NEMO's nonzero pivot transport before any landing. The post-stage association
remains a necessary invariant-restoring statement, not this magnitude owner.
No return to kt=8 is warranted until the kt=1 source is closed.

Round disposition: **HELD**.
