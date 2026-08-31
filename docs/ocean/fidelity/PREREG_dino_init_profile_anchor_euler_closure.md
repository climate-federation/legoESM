# Preregistration: DINO init profile/anchor and Euler closure

Date: 2026-08-30. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

Status: **FROZEN BEFORE IMPLEMENTATION OR NEW SCORE.** This continues the
ordered ladder in `PREREG_dino_standalone_init_geometry_repair.md`. Its clean
v3 artifact confirmed wet topology, T latitude, and T depth with zero bit
differences, then stopped at `common_depth_T_profile`.

No GPU, NEMO execution, or MPI execution is part of the repair/score phase.
The final human-owned GPU arm is emitted only after the CPU ladder admits it.

## Executed oracle source and current owners

DINO selects CASE(4) in
`cfgs/DINO/MY_SRC/usrdef_istate.F90`. The horizontally uniform profiles are
array expressions at lines 135--148. They contain `TANH`, multiplication,
addition, subtraction and division; they contain no `EXP` or `SIN`. The
executed NEMO binary imports scalar `tanh@GLIBC_2.2.5` and does not import a
`_ZGV*vector_tanh` symbol. The campaign's pure-JAX glibc-2.34 vector `EXP` and
`SIN` transcriptions are therefore precedents for explicit rounding and AD,
but are not directly reusable profile operators.

The live anchors and blend are owned by:

- `usrdef_istate.F90:151`: `zphiMAX=MAXVAL(gphit)` over NEMO's full
  construction array, not the stripped 195 x 48 score core and not nominal
  `rn_phi_max=70`;
- lines 153--166: `zTbot`/`zSbot` are `MINVAL(profile + 100*(1-tmask))`, with
  the domain reduction completed before the meridional blend; and
- lines 171--174: one reciprocal is formed, then the literal association is
  `((profile-bot)*(phi_max-ABS(phi))*inv_phi_max+bot)*mask`.

legoESM currently evaluates the profiles with `jnp.tanh` at
`packages/ocean/legoesm/ocean/experiments/dino.py:2479-2529`, defaults to the
nominal 70-degree and last-reference-level anchors at lines 2597--2608, and
factors the latitude ratio before multiplying at lines 2610--2613.

## Frozen ordered peel

One selector, owned by the existing DINO config and explicit on both oracle
cards, may choose the initialization evaluation. The legacy/default behavior
of every non-oracle recipe remains byte-identical. Unknown values and any
faithful selector used without the NEMO-faithful grid/masked-zco card fail at
construction.

Repair in this order, stopping at the first failure:

1. **P0 profile association.** Reassociate the existing JAX operations exactly
   as source lines 135--148, preserving each written division and multiply.
   The current factored form is the planted control.
2. **P1 TANH lowering.** Compare the source-associated JAX result with the
   committed NumPy/source-order oracle. If association alone passes, no new
   transcendental is added. If it fails, distinguish scalar glibc `tanh` from
   JAX `tanh`; implement a pure-JAX, fp64-only oracle lowering only if the
   scalar result owns the mismatch. Float32 and non-faithful paths retain
   `jnp.tanh`. The lowering must pass eager/JIT equality and finite forward and
   reverse AD; a host callback is illegal.
3. **A0 anchor domain.** Derive `MAXVAL(gphit)` from the same analytic
   full-construction Mercator indices already used by the faithful bathymetry,
   and derive `MINVAL` T/S anchors from the faithful profile and wet-level
   domain. No mesh/restart input may enter standalone state construction.
4. **A1 blend association.** Apply the single reciprocal and literal source
   association at lines 171--174. Nominal-70/last-level anchors and the current
   factored ratio are independent planted controls.
5. **E0 Euler.** Only after every initialization admission row passes, run the
   existing `_step_rows` against the registered raw `RUN_KT2` dumps. No Euler
   row may be emitted conditionally after an initialization failure.

## Frozen rows, bars, and disposition

The existing v3 geometry rows remain mandatory zero-difference prerequisites.
The following are POINTWISE rows and retain the frozen absolute bar `1e-15`:

| row | active set | bar |
|---|---|---:|
| `common_depth_T_profile` | every NEMO-wet T cell | max abs <= 1e-15 degC |
| `common_depth_S_profile` | every NEMO-wet T cell | max abs <= 1e-15 PSU |
| `resolved_T_nemo_wet` | every NEMO-wet T cell | max abs <= 1e-15 degC |
| `resolved_S_nemo_wet` | every NEMO-wet T cell | max abs <= 1e-15 PSU |
| each raw Euler/carry row | its registered NEMO wet face/cell set | max abs <= 1e-15 |

Repository search found no executable or cited Rule-1b definition governing
this new IC row. Therefore this probe cannot self-authorize a waiver:
`RULE_1B_CLEARED` is legal only with a pre-existing signed-off rule identifier
recorded in the artifact. Without one, `1e-15` is binding.

Required controls are the old JAX/factored profile, nominal-70 anchor,
last-reference-level bottom anchors, factored blend, one-ULP field plant,
wrong-core plant, and initialization-gate plant. Each must move a nonzero
population or hard-fail. Config identity between oracle and catalog cards must
remain zero diff rows.

Outcome labels:

- `INIT_CONFIRMED`: all geometry, profile and resolved-field rows pass;
- `INIT_PARTIAL_<FIRST_FAILURE>`: stop before Euler;
- `EULER_AT_BAR`: init admitted and every Euler/carry row passes;
- `EULER_DEBT_<FIRST_FAILURE>`: init admitted but an Euler row fails.

## Frozen independent-year prediction and launch gate

The user-supplied discriminator is the prior: independent day-360 SST RMS was
`0.39 degC`; a bit-exact step-2 bridge followed by 359 free days was
`0.0104 degC` (with a 0.94-day endpoint offset), and the day-180 twin result
was `0.005 degC`.

**Prediction:** after initialization and Euler are at bar, a fresh standalone
member-0 integration through day 360 will collapse toward the bridge class,
with a point expectation near `0.01 degC` SST RMS against the horizon-matched
NEMO from-rest day-360 state.

Frozen discriminator:

- `CONFIRM`: horizon-matched final 3-D SST RMS <= `0.02 degC`;
- `REFUTE`: final 3-D SST RMS >= `0.10 degC`;
- `(0.02, 0.10) degC`: partial/inconclusive, reported without relabeling.

The arm is exactly 11,520 steps (`360 d * 32 step/d`) in fp64, standalone
member 0, public `nemo_dino_kamm_mlf` construction, no restart/bridge inputs,
and the existing diagnostic-only reducer mesh. It must request `--snap-final`
and the manifest must stamp day 360 as an fp64 3-D capture. The launch block is
withheld unless the clean CPU artifact says `INIT_CONFIRMED` and either
`EULER_AT_BAR` or a separately identified, pre-existing Rule-1b clearance.
