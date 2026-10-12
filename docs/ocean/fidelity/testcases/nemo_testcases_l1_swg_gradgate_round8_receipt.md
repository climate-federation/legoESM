# Certified SWG reverse-mode gradient gate — round 8 receipt

Date: 2026-10-12

Status: **LANDED; LANE CLOSED**. This closing round corrects the contradictory
Round-6 preregistration sentence and adds repeat-derived uncertainty to the
GPU timing deliverable. It changes no Python, physics, card input, loss,
random direction, agreement bar, checkpoint schedule, configuration,
stabilizer, clip, guard, or `stop_gradient`.

## Round-6 record correction

The Round-6 body no longer says commit `ad02afb13` encoded an “identical”
gate. It now records the actual deviation: commit `0bc492a1d` preregistered
`a >= -tolerance * mean(time)`, while `ad02afb13` changed the rule after
measurement to `a >= 0`. The change was motivated by interpreting `a` as
fixed overhead. It was unauthorized and did not alter Round 6's held verdict
because every schedule also failed the residual rule.

## Operator GPU repeats

The operator acquired three fresh-process resource rows per schedule at 720
and 1440 steps with the committed gate at `859c56935`. All 18 JSONs report the
native transcendental policy, GPU backend, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
the default allocator, valid JAX allocator memory statistics, clean provenance,
and status `CONFIRMED`. Their 18 recorded process IDs are distinct. Evidence is
under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swggg_rounds/round7/operator_rows/gpu_repeats/`.

| schedule / steps | three warm reverse times (s) | mean ± sample SD (s), range | device peak growth (MiB) |
|---|---:|---:|---:|
| none / 720 | `2.113894 / 2.108635 / 2.110169` | `2.1109 ± 0.0027`, `2.1086..2.1139` | `7750.288` |
| none / 1440 | `4.229789 / 4.223786 / 4.227229` | `4.2269 ± 0.0030`, `4.2238..4.2298` | `15483.843` |
| uniform / 720 | `1.229745 / 1.226009 / 1.225709` | `1.2272 ± 0.0022`, `1.2257..1.2297` | `254.466` |
| uniform / 1440 | `2.452690 / 2.452013 / 2.451360` | `2.4520 ± 0.0007`, `2.4514..2.4527` | `510.466` |
| binomial / 720 | `3.031688 / 3.009303 / 3.006660` | `3.0159 ± 0.0138`, `3.0067..3.0317` | `366.080` |
| binomial / 1440 | `6.248551 / 6.227110 / 6.212484` | `6.2294 ± 0.0181`, `6.2125..6.2486` | `366.080` |

Time uses the same Round-7 convention as CPU: arithmetic mean ± one sample
standard deviation across the three fresh processes, with the full observed
range shown. The GPU 60-day gradient costs are therefore none
`4.2269 ± 0.0030 s`, uniform `2.4520 ± 0.0007 s`, and binomial
`6.2294 ± 0.0181 s`. These are measured native-policy times, not projected
values or certified CPU/GPU speedups. The device peak growth is identical
across repeats for each schedule/window.

## Retained gradient and memory deliverable

The native-policy GPU agreement rows remain CONFIRMED:

| steps | reverse/forward relative error | reverse/central-FD relative error |
|---:|---:|---:|
| 1 | `2.27064171926424e-16` | `3.57798841572499e-7` |
| 24 | `9.30692734647596e-16` | `2.1980290555407674e-6` |
| 720 | `2.3070259307585205e-16` | `9.041792746430404e-7` |
| 1440 | `3.02727997319582e-15` | `6.852553986852776e-6` |

The broken-VJP plant remains REFUTED. Peak host RSS at 1440 steps remains
`25756.438 / 2189.617 / 1969.641 MiB` for none/uniform/binomial; repeated GPU
device peak growth is `15483.843 / 510.466 / 366.080 MiB`. Wall-clock
linearity remains uncertifiable on the shared CPU node. Training composition
remains the measured per-step Python loop through the production jitted
`swe_rk3` step; the scan composition's last-bit gap remains retained.

## Evidence and hygiene

- The GPU-repeat arithmetic was recomputed directly from all 18 operator JSONs;
  every row has clean provenance, a distinct process ID, and a usable JAX
  allocator-memory record.
- No `.py` file changed in this round, so no new direct Python test is needed.
  The existing shared gate direct tests and isolated production-step gradient
  test remain the executable owners.
- The citation gate is run explicitly on this receipt; its shifted-citation
  plant exits 1.
- The certified card sources, ladders, digests, and inherited stale digest pins
  are untouched.

## OPEN

- GYRE six-step and GPU rows belong to `fidelity/gyre-gradient-gate`.
- The scan composition's last-bit gap is retained and not closed.
- The deck-parameterised card lives on `fidelity/swg-double-gyre`.
- The inherited stale GYRE/LOCK/OVERFLOW digest pins remain GYRE-lane debt.
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
