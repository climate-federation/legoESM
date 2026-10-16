# ORCA2 round 298 — kt=7 stage-1 transport boundary

Date: 2026-10-16

Status: **HELD**. The admitted round-297 stream and a passive, source-ordered
replay agree under both **independent OMT-4** and **given NEMO's entry**: the
active-tracer transport checkpoint, `trp`, is the first recorded boundary
that causally carries the kt=7 stage-1 growth. Replacing NEMO's three recorded
transports reduces the fold-band salinity result from
12,046.543040268563 PSU to 325.50387971291514 PSU, below the frozen 10x entry
threshold. The preregistered `adv_t`/`adv_s` prediction is **REFUTED**.

This is a boundary attribution, not yet a leaf-statement attribution. The
complete fold unit remains private and atomic. This round changes no
production package, card, deck, selector, carried state, stabiliser, sea-ice
field, or `unmeasured_features` entry.

## Provenance and admission

- Lane base: `e3cba1315`; frozen preregistration: `d3d68350f`.
- Operator acquisition log SHA-256:
  `0c1e08be6528b47a94e26913f96dbb0e01b204a9ffbabfc888555a3ad02fedbc`.
- Admitted record:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round297/acquisition/orca2_omt4_stage7_source_10step_np2`.
- Admission status: `PASS_R297_OMT4_KT7_STAGE1_ADMISSION`; schema
  `NEMO_L4_R297S7`; two rank records; 23 named fields; exactly-once 148x180
  coverage; all 80 inherited entry/stage frames and both kt=10 terminal
  restarts byte-identical to the admitted round-280 build.
- Raw measurement SHA-256:
  `43243d0e783334089486d055cdbe775db0cc555452d147694f98fec25ea35509`.
- Classification SHA-256:
  `a53fdddde073d871f0a51257fce804facb4d10d945f67cf900a6a1cd8f6a5647`.

The two labels were advanced independently from their own kt=7 entry state.
They produce identical numbers and independently name `trp`; they are not
mixed in the tables below.

## Compiled statement and source order

The resolved OMT-4 program applies the velocity update at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480`, computes
and applies the barotropic correction at `:521-528` and `:547-549`, then calls
the active-tracer transport builder at `:565-592`. The recorded `trp_u`,
`trp_v`, and `trp_w` checkpoint is immediately after that call. Tracer
advection follows at `:617-662`, the inactive surface-source boundary at
`:664-672`, the QCO stage update at `:693-720`, and the completed-state
association at `:812-820`.

Therefore the first causal boundary is NEMO's active-tracer transport call at
`:565-592`. This round does not claim which internal horizontal or vertical
transport statement owns it; substituting all three together is the smallest
record-supported causal arm.

## Source-ordered measurement

All maxima below are absolute candidate-minus-NEMO differences on the
three-row northern fold band at kt=7 stage 1. The values are identical for
the two labels, but are reported separately as required.

| Label | entry S (PSU) | trp U (m3/s) | trp V (m3/s) | trp W (m3/s) | adv S (PSU) | qco/final S (PSU) |
|---|---:|---:|---:|---:|---:|---:|
| independent | 325.6037473733219 | 79,097,748.84596531 | 173,585,937.46527478 | 1,484,398,612.4373436 | 3.30026542696305 | 12,046.543040268563 |
| given NEMO's entry | 325.6037473733219 | 79,097,748.84596531 | 173,585,937.46527478 | 1,484,398,612.4373436 | 3.30026542696305 | 12,046.543040268563 |

The final-to-entry fold-band salinity growth is 36.9975565006522x. The
causal arms discriminate transport from advection:

| Label | candidate transport + candidate advection | candidate transport + NEMO advection | NEMO transport + recomputed advection | Frozen 10x threshold |
|---|---:|---:|---:|---:|
| independent | 12,046.543040268563 | 325.37701095571407 | 325.50387971291514 | 3,256.037473733219 |
| given NEMO's entry | 12,046.543040268563 | 325.37701095571407 | 325.50387971291514 | 3,256.037473733219 |

Both NEMO-advection and NEMO-transport arms return below the threshold, but
the candidate-advection arm remains at the bad result. In source order the
transport inputs already carry the error that advection consumes; advection
is not the first owner. No surface forcing is active: candidate and NEMO
`adv_t`/`adv_s` each equal their corresponding `sbc_t`/`sbc_s` arrays.

## Instrument controls and prediction ledger

The passive live trace is array-identical for every carried state field with
and without the hook. The offline shared tracer-stage replay reproduces the
candidate completed T and S arrays exactly (`max_abs = 0.0`, zero raw unequal
cells) under both labels. Decision 127 is bound mechanically: zero versus the
smallest positive subnormal is one separately counted subnormal match, while
`1.0` versus its next representable normal value remains one unequal cell and
fires the normal one-bit control.

All eight plants refuse with exit 2 and `STATUS PLANT-FIRED`: admission,
label coverage, source order, passivity, replay, unforced equality,
first-carrier selection, and the normal one-bit control.

| ID | Verdict | Result |
|---|---|---|
| R298-P1 | **CONFIRMED** | The two-rank, 23-field record admits with exact coverage and unchanged inherited products. |
| R298-P2 | **CONFIRMED** | Both labels independently name `trp`. |
| R298-P3 | **REFUTED** | `trp`, not `adv_t`/`adv_s`, is the first causal boundary. The failed prediction is retained. |
| R298-P4 | **CONFIRMED** | Passive state, exact candidate replay, unforced equality, Decision-127 control, and normal one-bit control all pass. |
| R298-P5 | **CONFIRMED** | Measurement only; no private constituent or production physics lands. |

## Validation and review

- Citation gate: **PASS**, eight citations, zero failures and zero unmapped
  citations; the shifted `:565-592` plant refuses. The default cumulative
  receipt also passes with zero unmapped citations. Citation-log SHA-256:
  `ce10c8f1a318618ee8c7c2bea4cbf1438af92433ba2d157e0e2e1a5d78093378`.
- Focused round-295/296/297/298 plus citation-gate battery: **28/28 pass** in
  5.31 s. Log SHA-256:
  `d37849f801ad842d19ca25a474d48f54b9d86ad55131f8b35e76629946577369`.
- The required single `tests/ocean/fidelity -n 12` battery reached 99% with
  3,594 pass lines, 19 failure lines, and then hung after its G1 worker
  stopped. Ten failures were G1/JAX cases. Eight pass serially before that
  serial process reached the documented per-process compiler abort; the two
  remaining IDs each pass in fresh isolated processes (170.11 s and 171.61
  s). This is the campaign's known JAX compiler-resource class, not a
  round-298 failure. Full-battery, first serial, and fresh-process SHA-256:
  `5d398741302087151db28225c8bd65502baeb27a6f386690c1d894677b4b82fd`,
  `9f406189e2c9afb2e1f9061c302d94b5c59620ae23ca1e7e376d6219d8481bdc`,
  and `e777dee97ca30c276004776bc6e454c3529770654b9b428b6fc3e31e1f071b1d`.
- The other nine IDs reproduce serially before the receipt commit. Eight are
  the branch-tip reds already registered by round 297: SI3 `MY_SRC`
  provenance; the GYRE year-harness stamp; two stale round-33 mutation
  anchors; OMT-1's added required barotropic field; round-228's missing map
  key; 14 allow-dirty scope ratchets; and 13 worktree-stamp ratchets. The
  ninth is round-179's clean-tree preflight observing this uncommitted
  receipt; it is rechecked after the receipt commit. Serial log SHA-256:
  `8675a31fec925efad3475ef4a26dd6933bc35a38c7fc4d81ae29503c774b8d9f`.
- Independent review: pending the required read-only `codex exec` attempt;
  if the in-sandbox client cannot start, the final receipt records
  independent review unavailable in-sandbox and hands review to the
  operator.

## Configuration choices

- **ASKED:** none.
- **UNASKED:** none.
- Decision 126 remains pending; runoff is unchanged and is not re-raised.
- Sea ice and the ORCA2 card's `unmeasured_features` remain unchanged.

## OPEN

1. Split the kt=7 stage-1 active-tracer transport boundary in compiled order:
   first the carried horizontal `trp_u`, then `trp_v`, then the diagnosed
   vertical `trp_w`, with one recorded component substituted at a time.
2. For the first causal component, replay its operands in the compiled
   transport builder and name the first non-bit leaf statement. A new NEMO
   acquisition is required only if the existing admitted operand streams do
   not cover kt=7.
3. Keep the complete fold unit private and atomic. Nothing lands until OMT-4
   completes under both labels and the standing cross-card gates pass.
