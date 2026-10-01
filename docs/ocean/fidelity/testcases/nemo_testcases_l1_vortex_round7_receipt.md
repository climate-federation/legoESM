# NEMO testcase fidelity: round 191 / VORTEX round 7 receipt

**Status: LANDED.** Decision 78 is implemented exactly as scoped: `GYRE-zco`
and `VORTEX_VEC-zco` now select NEMO's carried after-SSH slot;
`ORCA2-zps` remains explicitly on `rk3_extrapolated` pending its own ladder;
the other certified cards do not execute the route. The GYRE ladder and year
reproduce round 6's carried measurement arm exactly. No new owner walk was
started.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round191/`.
The immutable before arms are round 6's `gates2/gyre_card.json`,
`gates2/vec_card.json`, and `gyre_day_gap_card.json`; the preregistered after
arms are round 6's corresponding carried artifacts. Round 191 regenerated the
after arms rather than treating those predictions as results.

## 1. Compiled-source statement

This is NEMO's own cross-step state, not a new stabiliser or a configuration
guess.

- After the `Nbb <=> Naa` rotation, NEMO writes
  `ssh(:,:,Naa) = 2 * ssh(:,:,Nbb) - ssh(:,:,Naa)` in both compiled programs:
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:221-225` and
  `GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3.f90:222-226`.
- The next step binds that stored `Naa` field to `r3t(:,:,Kaa)` and consumes it
  in the first `wzv` call:
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:149`,
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:153`,
  `GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stp2d.f90:152`, and
  `GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stp2d.f90:156`.
- The field enters the scale-factor part of the vertical-velocity statement at
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90:295-298` and
  `GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/sshwzv.f90:295-298`.
- NEMO persists the same slot as `ssha` at
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:184`, then reads it or
  applies its named missing-slot fallback at
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:362-370`.

The code change therefore selects an already transcribed and measured value;
it does not alter the arithmetic added in round 6. The OMIP resuming driver now
passes the model's shared carried-slot predicate to `load_run_restart`.

## 2. Recipe-derived execution census

The census in `restart_gate_final4.json` imports the production predicate
`nemo_rk3_after_ssh_is_carried`; it does not re-derive a parallel condition.

| card / recipe | stated after-SSH form | executes carried route | disposition |
|---|---|---:|---|
| `GYRE-zco` | `rk3_extrapolated_carried` | yes | switched and fully measured |
| `VORTEX_VEC-zco` | `rk3_extrapolated_carried` | yes | switched and fully measured |
| `ORCA2-zps` | `rk3_extrapolated` | no | explicit hold; ladder still unmeasured on this lane |
| `VORTEX-zco` | unset / route unreachable | no | fingerprint measured |
| `LOCK_EXCHANGE-zco` | unset / route unreachable | no | fingerprint measured |
| `OVERFLOW-zps` | unset / route unreachable | no | fingerprint measured |
| generic `NEMO-GYRE-recipe` | unset / route unreachable | no | own three-step gate PASS |
| DINO `nemo_dino_kamm`, `nemo_dino_kamm_mlf` | `leapfrog_continuity` | no | month gate measured |

This closes the scope trap: ORCA2 inherits the GYRE base recipe, so it receives
an explicit `rk3_extrapolated` override rather than silently switching without
its ladder.

## 3. GYRE certified ladder: full row registry

Before is the round-6 card arm; after is the round-191 default card. Values are
the gate's normalised max absolute residual. The full-precision mechanical
registry is the pair `vortex/round6/gates2/gyre_card.json` and
`round191/gyre_ladder.json`.

| kt | T before -> after | S before -> after | u before -> after | v before -> after | ssh before -> after |
|---:|---|---|---|---|---|
| 1 | `0` -> `0` | `0` -> `0` | `0` -> `0` | `0` -> `0` | `0` -> `0` |
| 2 | `6.054358e-16` -> `6.054358e-16` | `5.786374e-16` -> `5.786374e-16` | `8.326673e-17` -> `8.326673e-17` | `9.714451e-17` -> `9.714451e-17` | `0` -> `0` |
| 3 | `4.605980e-09` -> `5.861944e-10` | `2.378116e-10` -> `1.624493e-11` | `1.048051e-06` -> `4.434812e-10` | `1.088992e-06` -> `5.758083e-10` | `1.954856e-07` -> `6.695688e-11` |
| 4 | `9.291852e-09` -> `8.196478e-10` | `4.815164e-10` -> `1.296683e-11` | `9.007236e-07` -> `9.383797e-10` | `2.709247e-06` -> `1.312899e-09` | `2.696911e-07` -> `2.059097e-10` |
| 5 | `1.535994e-08` -> `9.433474e-10` | `6.978573e-10` -> `2.065596e-11` | `2.307717e-06` -> `3.749105e-10` | `8.673631e-07` -> `1.356928e-09` | `1.971394e-07` -> `2.107059e-10` |
| 6 | `7.684889e-09` -> `1.591710e-09` | `1.614955e-09` -> `3.561116e-11` | `1.354558e-06` -> `9.373022e-10` | `2.033023e-06` -> `2.174386e-09` | `2.387170e-07` -> `4.389195e-10` |
| 7 | `2.572803e-08` -> `2.427215e-09` | `4.523441e-09` -> `7.982616e-11` | `1.926351e-06` -> `1.547945e-09` | `1.015968e-06` -> `5.385081e-09` | `2.330476e-07` -> `5.288256e-10` |
| 8 | `1.155374e-08` -> `3.421273e-09` | `1.128758e-09` -> `1.332139e-10` | `1.000667e-06` -> `1.101391e-09` | `1.854612e-06` -> `4.191806e-09` | `3.712568e-07` -> `7.810069e-10` |
| 9 | `8.035019e-09` -> `4.563650e-09` | `1.099317e-09` -> `1.034879e-10` | `1.891744e-06` -> `1.726474e-09` | `1.199127e-06` -> `4.394961e-09` | `3.003036e-07` -> `9.218335e-10` |
| 10 | `8.236508e-09` -> `5.320815e-09` | `2.770495e-09` -> `8.265885e-10` | `1.237543e-06` -> `1.789976e-09` | `1.625698e-06` -> `2.373581e-09` | `2.881333e-07` -> `1.274981e-09` |

Forty rows move and all forty move toward NEMO; the ten kt=1/2 rows are
identical. No kt=1 row leaves the bar and the first-over-bar row remains kt=3.
The regenerated round-191 JSON is exactly equal, row for row, to round 6's
preregistered carried arm.

## 4. VORTEX-vector ladder: full row registry

Before is round 6's card arm and after is the round-191 default. The
full-precision registry is `vortex/round6/gates2/vec_card.json` versus
`round191/vortex_vec_ladder.json`.

| kt | T before -> after | S before -> after | u before -> after | v before -> after | ssh before -> after |
|---:|---|---|---|---|---|
| 1 | `0` -> `0` | `0` -> `0` | `2.220446e-16` -> `2.220446e-16` | `2.220446e-16` -> `2.220446e-16` | `1.355253e-20` -> `1.355253e-20` |
| 2 | `3.625489e-09` -> `3.625489e-09` | `4.060244e-16` -> `4.060244e-16` | `3.369330e-06` -> `3.369330e-06` | `3.337024e-06` -> `3.337024e-06` | `3.709010e-08` -> `3.709010e-08` |
| 3 | `3.845508e-08` -> `3.261103e-08` | `6.090366e-16` -> `6.090366e-16` | `5.983222e-06` -> `4.010084e-06` | `5.957174e-06` -> `3.983892e-06` | `6.505721e-06` -> `6.505721e-06` |
| 4 | `1.142105e-07` -> `1.140348e-07` | `8.120488e-16` -> `8.120488e-16` | `4.247932e-06` -> `2.372739e-06` | `4.221828e-06` -> `2.329281e-06` | `8.220041e-06` -> `8.230604e-06` |
| 5 | `1.594994e-07` -> `1.691849e-07` | `8.120488e-16` -> `8.120488e-16` | `6.210743e-06` -> `3.712188e-06` | `6.193649e-06` -> `3.595141e-06` | `8.002092e-06` -> `8.026674e-06` |
| 6 | `1.508892e-07` -> `1.724755e-07` | `1.015061e-15` -> `1.015061e-15` | `8.915647e-06` -> `4.743071e-06` | `8.769416e-06` -> `4.596829e-06` | `8.019861e-06` -> `8.026496e-06` |
| 7 | `1.307086e-07` -> `1.582145e-07` | `1.015061e-15` -> `1.015061e-15` | `1.107294e-05` -> `5.458853e-06` | `1.094573e-05` -> `5.314326e-06` | `5.230767e-06` -> `5.232806e-06` |
| 8 | `1.138413e-07` -> `1.475226e-07` | `1.015061e-15` -> `1.015061e-15` | `1.274480e-05` -> `5.939296e-06` | `1.274030e-05` -> `5.903662e-06` | `6.008865e-06` -> `6.015781e-06` |
| 9 | `1.474023e-07` -> `1.455210e-07` | `1.218073e-15` -> `1.218073e-15` | `1.294412e-05` -> `5.592987e-06` | `1.298733e-05` -> `5.483716e-06` | `4.593279e-06` -> `4.593345e-06` |
| 10 | `1.925638e-07` -> `1.495331e-07` | `1.218073e-15` -> `1.218073e-15` | `1.254250e-05` -> `4.865476e-06` | `1.248034e-05` -> `4.724707e-06` | `5.298993e-06` -> `5.356565e-06` |

Thirty-two rows move: twenty improve and twelve worsen, all registered above.
The first-over-bar row remains kt=2 and every kt=1 row is unchanged. The
round-191 JSON exactly reproduces round 6's carried arm.

## 5. GYRE month and year

The round-191 360-day run was freshly generated with the production card and
scored against `phase3/year_fromrest`. Every scored field, not just T, is
bit-for-bit equal to round 6's carried arm at days 30, 240 and 360.

| day | before T rms K | after T rms K | direction |
|---:|---:|---:|---|
| 30 | `2.343969031963881e-06` | `2.3432510206121264e-06` | toward NEMO |
| 240 | `6.582552471436041e-05` | `6.581707093530567e-05` | toward NEMO |
| 360 | `5.408462882803986e-05` | `5.407735418221895e-05` | toward NEMO |

The after values are the new certified GYRE numbers. The day-30 +0.7%
regression introduced in VORTEX round 5 remains mostly unexplained: this switch
recovers only `7.18e-10 K`, as round 6 already measured.

## 6. Restart regeneration and resume proof

`restart_gate_final4.json` regenerated format-5 archives for both switched
cards. `eta_rk3_after` is present in each inventory; every persisted slot
loads exactly; and continuous versus resumed states are exact after both the
first and second post-load steps. Archive SHA-256 values are:

- GYRE: `a437d2cd89fb41da5d1c65a6e113626585ffc2681d57e2d1f0187b2198f074e7`.
- VORTEX-vector: `13556e4991e0c54636ca739edcdf93b828f042c6b0041b7d740735b49753d301`.

VORTEX's derived `w` differs in 37,208 cells immediately after loading because
it is not a persisted restart slot; both resumed production steps are still
exact, so this is registered and is not hidden as a load-equality claim.

The final planted comparison perturbs the recovered `eta_rk3_after` field with
a non-constant pattern. It prints
`STATUS PLANT-FIRED: persisted after-SSH restart-slot comparison` and exits 1.
Two earlier trajectory-output controls are **REFUTED and retained**: a
single-cell `1e-9` perturbation and then a `1e-3` spatial perturbation were
absorbed at the sampled prognostic outputs and therefore could not calibrate
the persisted-slot assertion. The production ladder arm, rather than those
failed plants, establishes that the selected branch moves live numerics.

## 7. Unswitched-card fingerprints

| gate | result |
|---|---|
| `VORTEX-zco` kt=1..10 | all 50 rows exactly equal to round 6 |
| `LOCK_EXCHANGE-zco` kt=1..3 | all rows and residual artifact exactly equal to round 6; AT-BAR |
| `OVERFLOW-zps` kt=1..3 | all rows and residual artifact exactly equal to round 6; first debt remains kt=2, T `7.815970093361103e-15`, u `7.069220209210414e-12` |
| generic NEMO-GYRE recipe | own three-step certified gate PASS; all 15 rows finite |
| DINO month | `2.040288957e-03 K` against bar `2.244317642e-03 K`, PASS |

The GYRE config digest is deliberately re-pinned from
`da52bd90a40f71fd` to `308af4c536cbfd9a`. LOCK_EXCHANGE and OVERFLOW remain
`d794c4c5cb3dd880` and `2bb9d9be75fd924d` respectively.

The DINO number is `1.9e-10 K` from its pinned certified value, the known
harness floor, and DINO does not execute this RK3 carried branch.

## 8. Gates and tests

- Citation gate before this receipt: `16 passed`; all four recipe citations
  shifted by the new ORCA2 override were rigidly re-anchored and the default
  receipt has no unmapped citations.
- Focused suite: `196 passed, 4 failed in 181.85s`. The same four
  `test_run_omip_core2_restart.py` provenance probe-cap failures reproduce at
  the starting commit `c075e6fd5` as `4 failed in 0.56s`; they are pre-existing
  environmental debt. The new driver loader assertion passes.
- Round-specific citation gate: PASS, 10 citations, zero failures, zero
  unmapped citations, zero map-audit failures. Shifting
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:221-225` by two lines exits
  1, as required.
- Full `tests/ocean/fidelity tests/ocean/unit` was attempted as required. The
  `-n 12` run reached 96% but ten JAX/XLA compiler aborts killed and replaced
  workers, then the run stopped advancing; it was interrupted and emitted no
  summary line. The rule-prescribed lower-concurrency retry at `-n 4` reached
  31%, suffered the same abort in two unrelated compilation-heavy tests, and
  was interrupted rather than extending an oversized compiler batch; it too
  emitted no summary line. Exact logs are `full_ocean_tests.log` and
  `full_ocean_tests_n4.log`. These incomplete attempts are not represented as
  passes. The completed focused suite and all numerical gates above are the
  landing evidence.

## 9. Independent Codex review

The required command was run with a prompt instructed to refute the source
derivation, card census, restart proof, full moved-row registry, and
Decision-78 scope. It returned no scientific verdict because the read-only
sandbox blocked its in-process client. Verbatim terminal verdict/error:

> independent review unavailable in-sandbox — `Error: failed to initialize
> in-process app-server client: Read-only file system (os error 30)`

The preceding warning was also retained in `codex_review.log`:
`WARNING: proceeding, even though we could not create PATH aliases: Read-only
file system (os error 30)`. It emitted neither `SHIP` nor `DO NOT SHIP`.

## 10. Landing verdict

LAND. The switch is NEMO's cited state handoff, both switched cards reproduce
their preregistered carried arms, GYRE improves every moved ladder and year row,
VORTEX's worsened rows are all registered without moving the first-over-bar
row, no kt=1 bar row moves, and every non-executing card measured here retains
its fingerprint. ORCA2 remains explicitly unswitched and UNMEASURED rather
than inheriting an unsupported value.

## 11. OPEN

1. Walk the still-open GYRE day-30 +0.7% owner; round 6 refuted the carried
   after-SSH slot as that owner.
2. Re-measure ORCA2 on its own lane before switching it to the carried form.
3. Read the non-MLF DINO card's required after-SSH form from its own compiled
   build.
4. Run Decision 74's VORTEX 30/15/10-km resolution ladder.
