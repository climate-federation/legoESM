# ORCA2 round 215 — OMT-1 vector fold/mask pair held

Date: 2026-10-09. Base: `058dc7680`. Measurement tip: `e5257859f`.
Restored production tip: `c7d3c7c09`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round215/`.
Every OMT-1 number below is separately labelled **independent OMT-1** or
**given NEMO's entry OMT-1**. The final tree has no model-file difference
from the frozen base. No configuration, carried-state, stabiliser, sea-ice
selector, or `unmeasured_features` entry changed.

## First source-ordered statement

The corrected rank-complete record confirms one cancelling pair in NEMO's
vector-form external mode. The compiled executable evaluates the complete V
update and multiplies it by raw `ssvmask` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:669-682`, before the
seven-array north-fold association at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:747-756`. The active
T-pivot V mapping and sign are in
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:684-718`.

The preceding continuity-forcing difference is numerical zero and therefore
remains AT-BAR, not bit-exact. The first numerical statement is the atomic
pair: legoESM compact-remasks the incoming slow V forcing and its complete V
update, while NEMO preserves the fold-halo `Ve_rhs` and uses raw `ssvmask` at
the update. There is no NEMO selector between those operands.

## Record admission and offline replay

Round 214's corrected V3 record admits without another NEMO run: two
self-described rank files cover the global 148x180 domain exactly once, all
payloads are finite, all 64 inherited stage frames are byte-identical, and
both step-8 restarts are byte-identical. Admission status is
`PASS_R213_OMT1_VECTOR_PRE_LBC_ADMISSION`.

The preregistered 35-cell prediction is **REFUTED** as a global count: it was
the old rank-0-only census. The rank-complete record has 68 differing cells in
each operand. Substituting either `zv_frc` alone or `ssvmask` alone leaves all
68 target cells unequal, with maximum error
`4.0579965574751963e-4 m s-1`. Substituting both closes `va_pre_lbc`
bit-for-bit on every cell. The reference-operand replay is also bit-exact.
The report is `vector_pair_replay.json`, SHA-256
`ab6842d1cc2b8d870c5a6dd003b778f3972003bd7c82c8bea423f6e358b70f5b`.

## OMT-1 Decision-96 census

The candidate pair qualifies on OMT-1 itself, identically under both labels.
Of 160 rows, 155 move at the bit level: by RMS, 145 move toward NEMO and 10
away; by maximum, 127 move toward, 12 away, and 16 are score-equal. No exact
row is lost and the first debt (kt=1 stage-1 T) moves toward.

The kt=1 stage-1 SSH RMS/maximum improves from
`0.006243172975388979 / 0.13141649093684893 m` to
`0.006221601064607342 / 0.1308097181688451 m`. The registered final SSH
maximum also improves: kt=8 stage-3 `0.7583316946619745` to
`0.7252558498384503 m`. The independent and given-entry candidate artifacts
have SHA-256 `476e274d981daeee803b9ae06b54278911f42197558425c5393e667da2aba314`
and `71c5c1647887eb33bde473b35494907c9ac9ad5d752a21e2678bec5958d322b4`.

## Binding rung-0 veto

The same ordinary production statement fails Decision 96 on independent
hierarchy rung 0. It moves 195/200 rows; by RMS, 181 move toward and 14 away,
and no exact row is lost. The first debt moves toward. kt=1 stage-1 SSH also
improves from `0.006243384623854269 / 0.1314585958201272 m` RMS/maximum to
`0.006221800053172924 / 0.1308518230521234 m`.

The binding final-SSH veto nevertheless fires: kt=10 stage-3 SSH RMS improves
`0.02802652392771078 -> 0.027986522738908982 m`, but its maximum moves away
`0.42832517646246693 -> 0.42832541665889723 m`. The committed gate exits 2:
`STATUS REFUSE none: atomic pair is not Decision-96 eligible`. This is the
first red landing predicate, so the candidate is **HELD** and production was
restored. The candidate rung-0 artifact is `rung0_after.json`, SHA-256
`9252b91d693bb72a8bf9515679b27a3f2e3eeb31886d72b3fbab8ca5f90a9f8f`.

The shipped rung-10 candidate replay was started but produced no checkpoint
or artifact before it was stopped after the rung-0 veto. It is explicitly
**UNMEASURED**, not PASS. Because the landing predicate had already failed,
GYRE year, DINO month, and tank candidate gates were not run. The restored
tip's package tree is byte-for-byte the frozen base, so their admitted base
results remain unchanged; this transfer is not represented as candidate-arm
evidence.

## Preregistered predictions

| ID | disposition |
|---|---|
| R215-P1 | **CONFIRMED**: corrected two-rank record and every inherited passive stream admit. |
| R215-P2 | **PARTLY REFUTED**: the pair closes exactly and neither half closes, but the frozen 35-cell global prediction is replaced by the measured 68-cell rank-complete census. |
| R215-P3 | **CONFIRMED**: the preceding statement remains signed-zero-only; this pair is the first numerical boundary. |
| R215-P4 | **REFUTED** as a landing claim: OMT-1 qualifies, but rung 0 fails the registered final-SSH maximum veto. |
| R215-P5 | **UNMEASURED for the candidate** after P4 stopped the landing. The restored production tree is identical to the base. |
| R215-P6 | **CONFIRMED**: pair closure, both one-half, source-order, exact-loss, false-majority, and admission controls refuse. |

## Validation and review

Focused round-215 tests pass 7/7. The Decision-96 gate accepts both OMT-1
labels and refuses rung 0; its pair-closure, exact-loss and false-majority
plants all fire. The replay gate's pair, each-half, and source-order plants
all fire. The receipt citation gate passes all three compiled citations with
no failures or unmapped entries; its rigid-shift plant exits 1. The cumulative
default citation gate also passes with no failures or unmapped entries.

The one prescribed `tests/ocean/fidelity -n 12` battery collected 3,040
tests and reached 99% before the remaining compiler-heavy workers stopped
making progress. It was bounded rather than called a PASS: 3,013 passed,
seven skipped, 16 remained unclassified, and exactly four registered
pre-existing failures had emitted verdicts: the GYRE round-129 spread-floor
record stamp, allow-dirty scope, worktree-stamp ratchet, and SI3 scalar-math
provenance gate. The retained log is `round215/pytest_fidelity.log`.

Independent review was attempted separately with `codex exec --sandbox
read-only` and exited 1 before reading the diff: `failed to initialize
in-process app-server client: Read-only file system (os error 30)`.
Independent review is unavailable in-sandbox; this is not a PASS.

## OPEN

1. Attribute the `2.401964303011539e-7 m` rung-0 kt=10 stage-3 SSH-maximum
   regression before revisiting this pair. The next experiment must split the
   consumer downstream of the bit-exact pre-LBC pair, not either half of the
   already-proved cancelling unit.
2. Re-score rung 10, GYRE, DINO and tanks only after a complete compensating
   unit removes the rung-0 veto; no candidate-arm verdict exists for them in
   this round.
3. OMT-2 waits. No acquisition or configuration decision is requested.

No NEMO acquisition is needed. No user decision is pending.
