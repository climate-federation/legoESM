# ORCA2 round 260 — stage-2 V-transport hold

Date: 2026-10-12. Base: `f844b0ff2`. Preregistration commit:
`bc238e314`. First measurement commit: `cd67d23e3`. Rank-scoped transport
measurement commit: `40872736d`. Private-arm removal commit: `958225e99`.
Disposition: **HELD**.

This round changes no production model, card, deck, selector, carried-state
contract, stabiliser, sea-ice declaration, or `unmeasured_features` entry at
its tip. It measures OMT-4 separately as **independent** and **given NEMO's
entry**, on CPU fp64/libm production JIT. No NEMO acquisition was needed.

## Compiled statements

NEMO forms the stage correction in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279` and writes
the U/V metric stage transports in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`. After the
completed RK3 update, NEMO associates U, V, T, and S together at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802`; the
compiled T-pivot V fold rule is
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721`.

The measurement uses the already-passive completed-stage trace, the admitted
round-222 OMT-4 stage frames, and that record's rank-0 stage-2 tracer stream.
The tracer stream is rank-0 only: the transport split scores its admitted
90-column half and makes no claim about an invented complementary rank.

## Frozen predictions and verdicts

R260-P1 is **CONFIRMED**. Both labels reproduce round 259: stage-1 fold-band
T/S maxima are exactly 0.0, and the unassociated stage-2 fold-band salinity
maximum is `6.817836451535307` PSU. The live trace is array-identical to the
ordinary completed state for every carried field.

R260-P2 and R260-P3 are **CONFIRMED**. At the stage-2 entry, the unassociated
pivot-row V differs in 4,955/5,400 slots with maximum
`0.027227136128767482` m/s. The literal post-stage V fold association reduces
the maximum to `3.469446951953614e-18` m/s under both labels. U, T, S, and SSH
are already at the campaign floor there.

R260-P4 is **REFUTED** as preregistered. The associated entry collapses the
stage-2 fold-band tracer errors by about eight orders of magnitude, but does
not close them: T is `6.435500132795369e-9` K and S is
`1.3842782919937235e-8` PSU, both above the `2e-10` floor. The frozen
classifier refuses exactly on that prediction; the refusal is retained.

The prescribed continuation names the first remaining source-ordered
transport boundary. On the admitted rank-0 fold band, both labels produce:

| statement output | unequal/support | maximum absolute error | RMS error |
|---|---:|---:|---:|
| `zFu` | 0/8,100 | 0.0 | 0.0 |
| `zFv` | 849/8,100 | 0.7981690085143782 m3/s | 0.027014656486472527 m3/s |
| `zFw` | 666/8,370 | 1.7749975034384988 m3/s | 0.08531910357837941 m3/s |

Therefore the first non-floor statement after the exact stage entry is the
compiled `zFv` product, not a later tracer update. `zFw` is downstream of the
non-bit horizontal transport. The first `zFv` argmax is local
`[j=2,i=41,k=25]` on the scored three-row band. The result is identical under
both claim labels.

R260-P5 and R260-P6 are **NOT REACHED**. The complete unit is not integrated
or scored through kt=8..10, Decision 96 is not invoked, and nothing lands.
R260-P7 is confirmed: all seven continuation plants refuse (missing label,
wrong provenance, moved exact entry, wrong record identity, non-exact record
replay, erased transport boundary, and one-bit blindness).

## Evidence and validation

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round260/`.
The principal SHA-256 values are:

- independent/given stage-2 measurements:
  `dcffde46c0dc02b15091094868c66c76a2dcf79e15626908e47f85819aab9fd1` /
  `4bd8211d98286460f03764bdb80f0bd0bd0b9de9fd27c4b155947fe4e6650ef0`;
- retained preregistered-classifier refusal:
  `9730cb8520fe6cc2624240e8709cf3ebad19c50a6bd797541f1898b43ed56a82`;
- independent/given rank-scoped transport measurements:
  `e52c757d2150cb98155606ed7b02176c3d1212006c3f91cd6a64beed8e1f3e65` /
  `e4df91d86961364450fe8656a0a24e27c2f2ca869a094ab895ae9623182006f7`;
- transport classification:
  `cb1544d9a3bb2fb7a5961e8f30a15c1d3b25ace70ce9f8f491f900e7baaafc4c`.

The focused round-256/260 tests pass 28/28; log SHA-256 is
`de8feeeeab69bbb6378a1751a5b01834411c1e50774e6c69d186559e074fcd3a`.
The once-only `tests/ocean/fidelity -n 12` battery collected 3,378 tests and
reached 97%. Its four visible reds are exactly the registered pre-existing
GYRE round-129 harness-stamp, allow-dirty scope, worktree-stamp, and SI3
scalar-math provenance failures; all four reproduce serially. The battery
then stopped producing output on
`test_nemo_testcase_l1_vortex_round204_stage1_split.py::test_the_split_hook_at_its_default_changes_nothing`;
that exact ID passes alone in 266.07 s. Battery, known-red isolation, and hung
ID isolation SHA-256 values are
`77ad88a2451b8f8392df749370dc75a1f470ce6ffea805583f8d2f7c6864dff4`,
`635393d7a371fe4a2aae885fa2d4e7f1175552e88be6bb470ba6f2c72d4b75cd`,
and `e63c24ca5cc27e71584ecfe00a1f9c0510785b7d76d4d8bc5a644d492ec0750c`.

The package candidate and its temporary citation re-anchor were removed
before final validation. The final model file is byte-identical to the base,
so this HELD round cannot move OMT, rung-0, rung-10, GYRE, DINO, or tank
trajectories.

The cumulative default citation gate and this receipt's citation gate pass
with no failures or unmapped citations; the planted two-line shift of the
`zFv` statement fails. Their SHA-256 values are
`0ae376908534c8ee4991453e887b51d78d9522ca3f99aba112553e19b37b6d86`,
`6b263762e4efc7ea2a9e62f00e74bbc57c10f96926c7e95553938ca20a8ea30f`,
and `7c7470893f59ed7234640618333a57f833d2b3b581a9beae287fe2162e4c8a91`.

The requested separate `codex exec --sandbox read-only` review failed before
review because its in-process app-server could not initialize on the
read-only filesystem. **Independent review unavailable in-sandbox.** Its log
SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN

The complete atomic unit remains HELD and private. Round 261 starts inside
the cited stage-2 `zFv` product, in written operand order: `e1v`, live `e3v`,
associated `vv(Kmm)`, `zvb`, and resolved `vmask`. The existing stage frame,
D114 geometry, and admitted barotropic-average streams are used first; only a
genuinely missing operand stream warrants an acquisition. No kt=5/8 damage
field is revisited until this rank-0 stage-2 boundary reaches the floor.

ASKED choices: none. Unasked configuration choices made: none. Round
disposition: **HELD**.
