# Lane 3b round-10 preregistration — SI3 bulk bit identity and review debt

Date: 2026-09-04
Tracker: `climate-federation/legoESM#1699`
Starting commit: `4e29046372739009e473f8ebe053844ee63819a9`

## Scope and fixed protocol

This round changes no physical selector, forcing, timestep, grid, or oracle
trajectory.  It remains the ORCA1-resolved, constant-coefficient SI3 ice-side
bulk arm, fp64, CPU, 8,760 hourly C1D steps, against scalar-math oracle V2 A.
Rung 3.6 remains design-only and is not implemented.

The pre-implementation search covered `packages/core/legoesm/core`,
`packages/ocean/legoesm/ocean/eos.py`, `packages/core/legoesm/thermo.py`,
`packages/core/legoesm/core/bulk_flux.py`, `packages/ice/legoesm/ice`, and the
SI3 fidelity gates/tests.  The current branch has no source-statement rounding
helper.  The imported GYRE history contains one private
`ocean.eos._nemo_source_round`; this round will promote that exact mechanism to
the shared core precision module and use one implementation.  Existing bulk,
albedo, saturation, config, card, stream readers, and gates will be extended;
no second bulk model, saturation helper, or gate will be created.

## Hypothesis 1 — source-statement association

Baseline oracle-V2 census from the committed round-9 gate is 227,760 field
rows: 211,248 bit-identical and 16,512 non-bit.  The 16,494 rows below are
preregistered **DEBT**, not an accepted roundoff class:

| executing output | baseline non-bit rows | NEMO source statement |
|---|---:|---|
| `utau_ice` | 2,821 | `sbcblk.F90:1145-1147` |
| `vtau_ice` | 2,797 | `sbcblk.F90:1145-1147` |
| `evap_ice` | 3,879 | `sbcblk.F90:1279,1288` |
| `devap_ice` | 3,852 | `sbcblk.F90:1279,1289` |
| `emp_ice` | 1,900 | `sbcblk.F90:1298` |
| `emp_tot` | 1,232 | `sbcblk.F90:1299` |
| `fhld` | 13 | `icesbc.F90:358-360,364-366,391-405` |

Before editing production numerics, the committed gate extension will replay
kt6236 `emp_ice` with scalar NumPy operations materialized in the literal NEMO
statement order.  **CONFIRM order ownership** if that replay is bit-identical
to the oracle while the baseline legoESM value is not.  **REFUTE** if the
literal replay remains non-bit.

Then every active source assignment in `blk_ice_1`, `blk_ice_2`,
`ice_alb`, the Goff-ice `sbc_phy` helpers, and `ice_flx_other` will retain
Fortran grouping and pass its result through the shared source-round helper.
A private, non-config hook disabling only those statement boundaries is the
one-variable arm.  **CONFIRM the fix** if the new census has exactly 18
non-bit rows, all in `albedo`, `qsr_ice`, and `qsr_tot`, and the private arm
restores the 16,494 operation-order rows.  **REFUTE** if any ordinary-arithmetic
row remains non-bit or if a previously bit-identical non-transcendental row
becomes non-bit.

The 18 `icealb.F90:167-169` runtime-`EXP` rows are outside this arithmetic
hypothesis and remain `AWAITING_LIBM_POLICY`; this lane will not implement an
exp/tanh replacement.

## Hypothesis 2 — compiler-folded values

The active derived compile-time values are `reps0` (`sbc_phy.F90:33-35`),
`rgamma_dry` (`:37,39-43`), `rDg_i=LOG10(6.1071_wp)` (`:74-79`), and the
constant `LOG(10._wp)` consumed at `:709-711`.  A new config-local copy will
write these values once as binary64 and will not alter the shipped tree or any
retained V1/V2 root.  Two builds/runs must produce the same dump hash.

**CONFIRM** if at least `rDg_i` differs in bits from JAX runtime `log10(6.1071)`
or if replacing runtime evaluation with its oracle-dumped bit pattern reduces
non-bit rows without changing the formula.  **REFUTE** if all four runtime
values are already bit-identical and the gate census is unchanged.  Every
measured folded bit pattern will be pinned in `legoesm.constants` with its NEMO
line and hexadecimal representation; no NEMO constant will be evaluated by a
runtime transcendental.

## Review and hygiene acceptance

- Every scored row will report normalized error, conventional relative error,
  row-scale ULP error using `spacing(max(max(abs(oracle)), 1))`, and bit counts.
  A nextafter plant must be non-bit with exactly one row-scale ULP.
- ORCA1 namelist choices will leave `constants.py`; existing/new
  `SeaIceConfig` fields will carry complete `__param_spec__` entries, including
  `Ce_ice`, and the C1D card will select every value explicitly.  The SI3
  validator must reject mixed combinations.
- The 227,760-row JSON and any other runtime-scale artifact will live under
  `/data/abyssal/dbalwada/nemo-testcases-l3/round10_si3_bulk/`; git retains only
  hashes and summary evidence.  No data artifact is deleted.
- Runtime versions are always stamped.  An unregistered stack may execute and
  report normalized/relative/ULP metrics, but the gate must withhold the
  bit-identity verdict and exit nonzero only if asked to certify bit identity.
  The accepted bit claim remains Python 3.13.0, JAX/jaxlib 0.10.0, NumPy 2.4.4.
- Plants cover source rounding, folded-constant pins, ULP scoring, config
  coverage, external-artifact hash, and runtime claim withholding.  Each must
  exit nonzero without modifying retained evidence.

## Decisions

| choice | state | disposition |
|---|---|---|
| Treat 16,494 operation-order rows as fixable debt | ASKED | This round's primary acceptance target. |
| Promote the existing GYRE rounding mechanism to shared core | ASKED | User permitted import or promotion; promotion avoids an ice-to-ocean private dependency. |
| Dump and pin compile-folded constants | ASKED | New config-local copy and data root; shipped tree untouched. |
| Move ORCA1 selections into config and add parameter specs | ASKED | Card will select all explicitly. |
| Move large runtime JSON out of git | ASKED | Retained under the requested `/data` evidence tree. |
| Make runtime a stamp with conditional bit-claim refusal | ASKED | CI can still execute non-bit metrics on other stacks. |
| Implement rung 3.6 | UNASKED | Explicitly prohibited this round. |
| Add lane-local exp/tanh | UNASKED | Explicitly prohibited. |
| Change physical selectors/default behavior outside `nemo_si3_constant` | UNASKED | No such change planned. |
| Delete retained evidence | UNASKED | Forbidden; nothing will be deleted. |
| Push | UNASKED | Forbidden; no push. |
