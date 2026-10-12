# Certified SWG reverse-mode gradient gate — round 7 receipt

Date: 2026-10-12

Status: **LANDED; LANE CLOSED**. Wall-clock cost is now a repeated measurement
with uncertainty, never a CONFIRMED linear law. All three real CPU suites are
MEASURED, their five mean times increase strictly, and the directional
gradient and memory results remain valid. This round changes no physics, card
input, loss, random direction, agreement bar, checkpoint schedule,
configuration, stabilizer, clip, guard, or `stop_gradient`.

The certified SWG card may be used for training only through the measured
per-step Python loop crossing the production jitted `swe_rk3` step. No linear
wall-time scaling guarantee is claimed. The uniform scan composition remains
excluded by its retained last-bit gap.

## Preregistration and corrected historical record

Commit `a58fec2e4` preregistered the complete Round-7 protocol before any
measurement. It also records the Round-6 intercept change accurately: commit
`0bc492a1d` preregistered `a >= -tolerance * mean(time)`, then commit
`ad02afb13` changed the rule after measurement to `a >= 0`. The stricter rule
was motivated by interpreting the intercept as fixed overhead, but the timing
made it an unauthorized **DEVIATION**. It did not change Round 6's held outcome
because every schedule independently failed the residual rule. Round 7 removes
the intercept, residual-band, and incremental-band acceptance rules.

The only executable timing condition is strict increase of the five window
mean times. Its limited reason is that a longer adjoint window cannot have
negative incremental work. This condition does not certify linearity.

## Fresh-process CPU timing observations

Every cell below gives the three blocked warm-reverse times in seconds. Each
row was produced in a separate clean process at commit `a58fec2e4`, from the
admitted kt=2 stage-1 entry under
`PrecisionPolicy.fp64(transcendentals="libm")`. The timer brackets only blocked
execution of the already-jitted reverse transform.

| steps | none repeats (s) | uniform repeats (s) | binomial repeats (s) |
|---:|---:|---:|---:|
| 24 | `0.940038 / 0.926732 / 1.010019` | `0.632933 / 0.646483 / 0.638005` | `0.810273 / 0.857669 / 0.865670` |
| 96 | `3.459323 / 4.052426 / 3.951082` | `2.009747 / 2.362796 / 2.475734` | `3.801785 / 3.684256 / 3.604272` |
| 360 | `15.006208 / 11.563077 / 11.416196` | `9.061061 / 8.481368 / 9.158058` | `14.958292 / 14.975894 / 15.105171` |
| 720 | `21.895631 / 26.479507 / 25.387069` | `17.847710 / 17.870070 / 18.153049` | `30.238662 / 31.303729 / 31.361905` |
| 1440 | `54.123683 / 52.817045 / 53.197103` | `37.217041 / 42.692032 / 20.847929` | `81.537215 / 78.309193 / 76.485656` |

All 45 individual rows are finite, nonzero, and CONFIRMED by the unchanged
resource-row classifier. Their repeat labels are exactly `1, 2, 3`; all 45
process IDs are distinct.

## Per-step cost with uncertainty

The fitted model `time = a + b * steps` is descriptive. The reported
uncertainty is one OLS standard error of `b` over all fifteen observations per
schedule. Its residual sum of squares is explicitly decomposed into
within-window repeat scatter plus three times the five-mean fit residual. It is
not a confidence interval or a linearity verdict.

| schedule | b ± standard error (s/step) | repeat RMS (s) | fit-residual RMS (s) | max repeat relative spread | verdict |
|---|---:|---:|---:|---:|---|
| none | `0.03674211 ± 0.00077916` | `1.178385` | `0.850008` | `28.353%` | **MEASURED** |
| uniform | `0.02338319 ± 0.00223997` | `4.153767` | `0.440645` | `65.040%` | **MEASURED** |
| binomial | `0.05476449 ± 0.00176784` | `0.963400` | `3.152723` | `6.560%` | **MEASURED** |

The fitted intercepts are `-0.318159 / 0.326613 / -3.055672 s` and R2 values
are `0.998003 / 0.998674 / 0.987762` for none/uniform/binomial. They are
diagnostics only. Negative fitted intercepts are retained rather than gated or
explained away. The direct evidence therefore supports measured costs with
the stated uncertainty, not a certified affine law.

## Executable controls

The only binding timing plants return **REFUTED**, exit 1:

- constant time;
- a decreasing 1440-step endpoint; and
- the exact nonmonotonic series `[1, 100, 1, 1000, 2000]`.

The old affine-gate controls are retained as descriptive artifacts, not
passes. The review counterexample
`[0.2451153271, 1.4510041507, 4.6135624796, 6.2600150596, 16.3588091356]`
returns **MEASURED** with maximum relative fit residual `0.261666`; the earlier
monotone nonlinear series returns MEASURED with residual `3.540504`; the dead
interval returns MEASURED with residual `0.604516`; and the R2≈0.95 series
returns MEASURED with residual `1.765195`. Each artifact says
`LARGE_RESIDUAL_OLD_GATE_PLANT; descriptive only` and explicitly disables the
current monotonic-trend verdict. MEASURED here is a recorded observation, not
a pass or linearity claim.

## Direct 60-day gradient cost

CPU time is mean ± one sample standard deviation across the three 1440-step
processes; the range is shown to expose the shared-node spread. CPU memory is
the maximum observed total process peak RSS. GPU values are the retained
single operator-acquired native-policy rows at clean `1d7fae766`; they carry
no repeat-derived uncertainty and are not a CPU/GPU speedup comparison.

| schedule | CPU warm time mean ± SD (s), range | peak host RSS (MiB) | GPU warm time (s) | GPU device peak growth (MiB) |
|---|---:|---:|---:|---:|
| none | `53.3793 ± 0.6721`, `52.8170..54.1237` | `25756.438` | `4.219477` | `15483.843` |
| uniform | `33.5857 ± 11.3658`, `20.8479..42.6920` | `2189.617` | `2.452661` | `510.466` |
| binomial | `78.7774 ± 2.5581`, `76.4857..81.5372` | `1969.641` | `6.252818` | `366.080` |

The requested true 60-day uniform CPU cost is therefore
`33.5857 ± 11.3658 s`, with observed range `20.8479..42.6920 s` and maximum
peak host RSS `2189.617 MiB`. Linearity is not certifiable on this shared node.

### Round-8 GPU-repeat amendment (2026-10-12)

After the Round-7 review, the operator acquired three fresh-process native-
policy GPU repeats at both 720 and 1440 steps for every schedule. The evidence
is under `round7/operator_rows/gpu_repeats/`; all 18 JSONs have distinct
process IDs, clean commit `859c56935`, preallocation disabled, the default
allocator, and JAX allocator memory statistics. Time below is mean ± one
sample standard deviation; the range exposes the observed spread. This
supersedes the single-observation GPU timing column above without rewriting
that historical observation.

| schedule | 720-step GPU mean ± SD (s), range | 1440-step GPU mean ± SD (s), range | 1440-step device peak growth (MiB) |
|---|---:|---:|---:|
| none | `2.1109 ± 0.0027`, `2.1086..2.1139` | `4.2269 ± 0.0030`, `4.2238..4.2298` | `15483.843` |
| uniform | `1.2272 ± 0.0022`, `1.2257..1.2297` | `2.4520 ± 0.0007`, `2.4514..2.4527` | `510.466` |
| binomial | `3.0159 ± 0.0138`, `3.0067..3.0317` | `6.2294 ± 0.0181`, `6.2125..6.2486` | `366.080` |

The GPU 60-day costs are therefore measured, not projected: none
`4.2269 ± 0.0030 s`, uniform `2.4520 ± 0.0007 s`, and binomial
`6.2294 ± 0.0181 s`. They remain non-certified native-policy measurements
and are not CPU/GPU speedup claims.

## Retained directional agreement and composition

The native-policy operator GPU agreement rows remain CONFIRMED:

| steps | reverse/forward relative error | reverse/central-FD relative error |
|---:|---:|---:|
| 1 | `2.27064171926424e-16` | `3.57798841572499e-7` |
| 24 | `9.30692734647596e-16` | `2.1980290555407674e-6` |
| 720 | `2.3070259307585205e-16` | `9.041792746430404e-7` |
| 1440 | `3.02727997319582e-15` | `6.852553986852776e-6` |

The broken-VJP plant remains REFUTED. Training composition remains the
per-step Python loop through the production jitted `swe_rk3` step. The rejected
uniform scan composition differs after 24 steps by SSH
`8.881784197001252e-16`, U `1.3877787807814457e-17`, and V
`6.071532165918825e-18`; those exact fields live in the Round-6 JSON. The
bitwise composition prediction remains REFUTED and the gap is not closed.

## Evidence and hygiene

Round-7 evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swggg_rounds/round7/`.

- The shared gate direct suite passes **37/37**.
- All three real repeated timing suites exit 0 MEASURED; all three binding
  plants exit 1 REFUTED; all four old-gate diagnostics emit MEASURED with
  their fit residuals.
- The isolated SWG production-step gradient-matrix test passes **1/1**.
- The independent and stage SWG ladders remain AT-BAR and bit-identical; the
  stage one-ULP plant exits 1.
- Fresh fp64 hashes remain GYRE `f02d2f42a570f413`, TSUNAMI
  `8bb5ed6c1bc031bd`, VORTEX `f7489467bc156303`, and VORTEX_VEC
  `5041b91c67bd1443`.
- The inherited VORTEX-card module failures remain exactly the three stale
  digest pins; no pin is changed here.
- The citation gate is run explicitly on this receipt; its shifted-citation
  plant exits 1.

The committed GPU operator script remains
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`.
No GPU acquisition is needed in this round.

## OPEN

- GYRE six-step and GPU rows belong to `fidelity/gyre-gradient-gate`, not this
  lane.
- The scan composition's last-bit gap is retained and not closed.
- The deck-parameterised card lives on `fidelity/swg-double-gyre`, not this
  branch.
- The inherited stale GYRE/LOCK/OVERFLOW digest pins remain GYRE-lane debt.
  Expected values are `5850bbcfcd351c29` / `4ec82201f0fae7bf` /
  `7f1ec89b3e6578f5`; actual values are `f02d2f42a570f413` /
  `d5dfdf83b3a1e0ab` / `83de0010e1170032`. They are not re-pinned here.
- Wall-time linearity is not certifiable on a shared node.

No additional work or acquisition is scoped in this lane.

## Source citations

The certified compiled SWG dispatcher is the three-stage program at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:122-321`; its executed stage-3 momentum
block contains vector advection, ENS vorticity, constant Laplacian viscosity,
centered wind, and linear drag at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:260-273`. The total-ENS operator and its
live F-thickness divisions are in
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:639-675`.
