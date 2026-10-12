# Certified SWG reverse-mode gradient gate — round 1 receipt

Date: 2026-10-12

Status: **LANDED, GPU ROWS OPEN**. Reverse mode agrees with forward mode and
central finite differences through the production SWG step for one hour, one
day, and thirty days. All six CPU checkpoint-resource rows completed. No
physics, card choice, stabilizer, clipping, guard, or `stop_gradient` changed.

## Provenance and gate import

The incoming SWG tip was `292504bc1`. The existing GYRE gate lineage was
fetched read-only and cherry-picked through local commit `a36c459f2`, including
its direct test, production matrix row, GPU runner, TKE transpose treatments,
process isolation, and active-cell CEN2 denominator treatment. The shared gate
was extended rather than copied. The frozen SWG protocol is `79f086a51`; the
production-boundary amendment is `d5292ab5c`.

All measured JSON files are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swggg_rounds/round1/` and stamp
clean producer commit `d5292ab5c032a9188f8478538a1de458220c3855`, CPU,
fp64/libm, float64 state and geometry, the admitted kt=2 stage-1 boundary, and
`LatLonCGridOceanModel._step_jitted`.

## Unchanged-card controls

The complete SWG ladder was run before and after the GYRE import. The
given-entry and chained stage ladders each remained 338/338 bit-identical; the
independent, carried-state, and thirty-day ladders remained 50/50, 29/29, and
690/690 bit-identical. Direct comparison of every `per_kt` and control row was
equal. No package file changed after that post-import control.

The committed card digest helper also returned the same values on both sides:

| card | digest before | digest after |
|---|---|---|
| GYRE-zco | `f02d2f42a570f413` | `f02d2f42a570f413` |
| TSUNAMI-zco | `8bb5ed6c1bc031bd` | `8bb5ed6c1bc031bd` |
| VORTEX-zco | `f7489467bc156303` | `f7489467bc156303` |
| VORTEX_VEC-zco | `5041b91c67bd1443` | `5041b91c67bd1443` |

The frozen preregistration's `3b2b358145ced147` GYRE value was a transcription
error and is explicitly withdrawn in its appended erratum. The table above is
the clean helper output replayed independently at incoming `292504bc1` and at
the final lane tip.

The controls are retained as `before_*.json` and `after_import_*.json` in the
round evidence directory. No NEMO executable or source was changed or run.

## Directional agreement

The scalar objective is the mean of variance-normalized wet-cell SSH, U, and V
MSE. PRNG key 0 supplies a masked three-field perturbation with combined norm
`1e-3`; key 42 supplies an independent masked unit direction. Every step enters
the production JIT through an explicit Python window with per-step
rematerialization. Its zero-perturbation primal is byte-identical to the direct
production window.

The frozen bars are reverse/forward relative error at most `1e-9` and
reverse/central-FD relative error at most `0.05`, with central FD step `1e-3`.

| steps | reverse | forward | central FD | rev/fwd relative | rev/FD relative | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `-3.055917560677806e-02` | `-3.0559175606777962e-02` | `-3.0559186540819565e-02` | `3.1788984069699327e-15` | `3.577988403236465e-07` | **CONFIRMED** |
| 24 | `-2.1842644706797177e-05` | `-2.1842644706797268e-05` | `-2.1842692717673664e-05` | `4.188117305913815e-15` | `2.198029203985421e-06` | **CONFIRMED** |
| 720 | `-3.5854846776002614e-09` | `-3.5854846769147900e-09` | `-3.5854814359568625e-09` | `1.9117954763975566e-10` | `9.041018691902848e-07` | **CONFIRMED** |

All values are finite and nonzero; no NaN appeared. The one-step broken-VJP
plant left forward and FD unchanged, doubled reverse to
`-6.111835121355612e-02`, classified REFUTED, and exited 1. Thus the gate can
fail on a wrong pullback.

Artifact SHA-256 values are: one step
`e797e83cc567a0c5d703e6d6165ee3113a8a410c254528e87a0d391966aa43ea`, plant
`e59c03289828637bac882eb95e36977100be2507c32041deb943fce8c0292b7d`, 24 steps
`7c92914d8daeff3f18be328f2cb1e716ba658880e9d96887d87a265700099598`, and 720
steps `868ef32c2734a6b945b1a49dac47681307f1e5de75a52f37afb114e424f8cee7`.

## Retained refutation: scan composition

The initially preregistered 24-step `uniform` scan stopped before derivative
measurement because its primal was not byte-identical to the explicit
production sequence. The maximum finite differences were
`1.3877787807814457e-17` in U, `6.071532165918825e-18` in V, and
`8.881784197001252e-16` in SSH. That bitwise prediction remains **REFUTED**.

Before any 24/720 derivative or resource number was acquired, the committed
amendment restored GYRE's explicit per-step production boundary for agreement.
Resource rows necessarily measure the shared checkpoint schedule and therefore
use its pre-existing value contract (`rtol=1e-9`, `atol=1e-11`) while recording
bit identity and maximum absolute difference. No derivative bar, physical
input, objective, perturbation, or FD step was changed.

## Host memory and wall time

Each row ran in a fresh process. A cold compiled reverse pass preceded one
blocked warm reverse pass while host RSS was sampled every 0.05 seconds.

| schedule | steps | cold compile + reverse (s) | warm reverse (s) | baseline (MiB) | peak (MiB) | increment (MiB) | samples | schedule primal | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| none | 24 | `26.1809` | `24.9504` | `2512.875` | `3486.109` | `973.234` | 462 | within contract, max `8.88e-16` | **CONFIRMED** |
| uniform | 24 | `13.7324` | `10.5710` | `1739.145` | `1973.242` | `234.098` | 205 | within contract, max `8.88e-16` | **CONFIRMED** |
| binomial | 24 | `23.6142` | `17.4020` | `2005.582` | `2466.270` | `460.688` | 335 | bitwise | **CONFIRMED** |
| none | 720 | `52.7089` | `52.3562` | `14004.328` | `21904.895` | `7900.566` | 979 | within contract, max `7.82e-14` | **CONFIRMED** |
| uniform | 720 | `33.8951` | `27.5407` | `1935.355` | `2206.734` | `271.379` | 535 | within contract, max `7.82e-14` | **CONFIRMED** |
| binomial | 720 | `72.1908` | `59.3202` | `2104.953` | `2485.801` | `380.848` | 1154 | bitwise | **CONFIRMED** |

The preregistered 1440-step estimates derived only from successful 720-step
rows are:

| schedule | estimated peak RSS (MiB) | estimated warm reverse (s) |
|---|---:|---:|
| none | `29805.461` | `104.7124` |
| uniform | `2206.734` | `55.0815` |
| binomial | `2640.961` | `118.6403` |

The six resource JSON SHA-256 values, in `(none24, uniform24, binomial24,
none720, uniform720, binomial720)` order, are
`0655a45807170e836742d969e42d3374802c594990bed1bfb17fa6f5ff9e9f34`,
`82dcbbd64dc1fd75762a9468fd352dd3257c0eb2910bedc06fd75fb9344c7c17`,
`3a11aa4996d024f265d88c65361097648fabb17e6da78bb18a26c67787a5987a`,
`1018a4c91186eebbcc77581291232dfcdb715e0b2309a3111ced44f7d8e62e8b`,
`f35bfffb357700f3256ef795b448840a7ebcdb7025cd308ca90252ac9201a228`,
and `c5a04f37918f956e07d7827e1cb58f3b997c3f341884104d9d91279836d4cb78`.

## Gate, tests, and operator work

- Shared gate/direct schedule suite: **32 passed**.
- Imported active-cell CEN2 source-bit/VJP control: **1 passed**.
- Isolated SWG production matrix row: **1 passed, 11 deselected**.
- Python compilation, Ruff for the shared gate/direct test, shell syntax, and
  `git diff --check`: PASS.
- Citation gate and planted citation shift: PASS.

The committed GPU entry point is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`.
It pins physical GPU 1, uses fp64/native, runs the same three agreement and six
resource rows serially, refuses dirty state and overwrite, and validates the
non-certified native-policy label. This sandbox cannot acquire those rows.

The lane is **UNREVIEWED**: independent reviewer agents were not invoked in
this run. The numeric claims remain tied to committed probes and fail-closed
controls rather than reviewer judgment.

## OPEN

1. The operator must run the committed GPU-1 script from a clean final tip and
   retain all nine native-policy JSON rows in the round evidence directory.
2. Complete the campaign's independent reviewer pair before citing this lane
   outside the SWG gradient gate.

## Source citations

The certified compiled SWG dispatcher is the three-stage program at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:122-321`; its executed stage-3 momentum
block contains vector advection, ENS vorticity, constant Laplacian viscosity,
centered wind, and linear drag at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:260-273`. The total-ENS operator and its
live F-thickness divisions are in
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:639-675`.
