# ORCA2 round 336 — `tra_zdf` operand audit and acquisition handoff

Date: 2026-10-26

Base: `ba9f39512595`; preregistration: `a7513dc929`; offline audit:
`4cd085a1ec` plus target-support correction `619630bfd0`. Disposition:
**ACQUISITION_NEEDED**. This round changes no model statement or configuration
value, and Codex did not run NEMO.

## Executing source order

The retained round-333 record build calls `tra_zdf` after the recorded LDF,
BBC/BBL and damping boundaries at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/stprk3_stg.f90:745-771`.
For this deck, `ln_zad_Aimp=F`, `ln_zdfmfc=F`, `ln_zdfevd=T`,
`ln_zdfddm=T`, `ln_zdfiwm=T`, and `ln_traldf_msc=T`; these values were read
from the retained `ocean.output` and are enforced by the audit and acquisition
plants.

At every time step, the record build calls the vertical-physics composition at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/stprk3.f90:183`. That
composition first copies the closure result, then applies river-mouth, EVD,
DDM and IWM changes to the final `avt`/`avs` at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfphy.f90:365-392`.
Inside the executing non-Aimp tracer solve, temperature forms `avt+akz` and
salinity forms `avs+akz` at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:180-216`, then
forms the lower, upper and diagonal rows at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:230-235`.
The eliminated diagonal is evaluated at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:268-273`, the
thickness RHS and forward recurrence at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:283-291`, and
back-substitution at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:293-299`.

The shared legoESM production caller reaches its literal tracer-pair dispatch
at `ocean_model_latlon_cgrid.py:14824-14849`. The paired solve consumes the
content RHS and matrix at `implicit_solver.py:478-539`; matrix construction is
at `implicit_solver.py:542-641`, and its eliminated-diagonal, forward and
reverse recurrences are at `implicit_solver.py:329-392`. The stage call that
supplies these operands is at `ocean_model_latlon_cgrid.py:11706-11738`.

## Offline admitted-input audit

Both independently constructed rung-10 labels were run through ordinary and
traced production JIT on CPU under fp64/libm. The trace changed zero prognostic
leaves and was bit-identical to the ordinary state. The round-333 tracer and
internal-LDF inputs were re-admitted before use.

| operand | support | given NEMO entry unequal bits | independent unequal bits | verdict |
|---|---:|---:|---:|---|
| thickness-form T RHS | 18 target words | 0 | 0 | exact |
| `e3t(Kaa)` | 18 target words | 0 | 0 | exact |
| `e3w(Kmm)` | 203 interfaces in the seven target columns | 0 | 0 | exact |

All maxima and RMS values are exactly zero. The admitted records do not contain
the final kt-2 `avt` or `avs` after the full `zdf_phy` composition, and they do
not contain any matrix or recurrence row. The older closure-level `avt_k`
cannot substitute for that final coefficient.

Therefore no first differing operand or statement is **CONFIRMED**. The first
unmeasured operand in executed order is the effective `avt+akz` temperature
coefficient (and `avs+akz` for salinity), so ownership remains **PLAUSIBLE**.
The following matrix, eliminated-diagonal, forward and back-substitution rows
are also unmeasured. Decision 122 and the conditional M1/M2 production landing
path do not open; no DINO month, grid-ladder, OMIP-driver or gradient claim is
made.

The admitted-result artifacts are:

- `round336/offline/given_nemo_entry.json`, SHA-256
  `c479fe246442c3a1f09963a3f68f1e32be46c5f43c3755ca9c5b6209f4d2d616`;
- `round336/offline/independent.json`, SHA-256
  `0fbb4e944a7629c433c7b92d7d999e3cede938e52cd9f843ac8db680da399ced`.

Both report `PASS_R336_TRAZDF_INPUT_AUDIT_ACQUISITION_NEEDED`. Preregistered
R336-P1 and P2 are confirmed, P3 is triggered, and P4 stops for the missing
record.

## Passive acquisition extension

The additions-only operator tool imports the complete reviewed round-333
driver. Immediately before `tra_zdf_imp`, it preserves `Krhs` because the kt-2
stage-3 level tuple is `(Kbb,Kmm,Krhs,Kaa)=(3,2,1,1)` and the live solve
overwrites that slot. One capture call after `tra_zdf_imp`, outside all its
loops, records final `avt`, `avs`, `akz`, geometry, RK levels and live `Kaa`,
and replays NEMO's exact statement association into lower/diagonal/upper,
eliminated, forward and back-substitution fields. Admission requires the
outside-loop back-substitution to equal live `Kaa` bit-for-bit on every
executed cell.

The clean read-only preflight passed every inherited round-333 control,
including 77 parent tests, both inherited writer suites, 140-file deterministic
passivity, syntax checks, synthetic in-place and retained admission routes,
and all inherited plants. The new two-rank fixture passed bounds, signed-zero,
source replay and live-result checks for all 25 fields. All selector,
placement, header, field-order, truncation, checksum, extent, finite-value,
signed-zero, wet-coverage, replay and actual-result plants fired. The actual
record-build `trazdf` object had the same normalized 1,294 floating-point
instructions with and without the observer; a changed matrix statement failed
that census. The extended same-program proof reports 407 identical ppsrc and
419 identical normalized objects, exactly seven observer and five writer
units. Preflight log SHA-256 is
`ef0003f750306f5068b4d3b4513e63db31965a4e5163b408392f4ae263eee543`.

Operator command:

```text
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/orca2_r336-10964668/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_trazdf_acquisition/run.sh --run
```

Expected evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/acquisition/orca1ice_trazdf_internal_kt2_np2`.

No configuration value changed: **ASKED**. Independent review: operator
(Claude).

The mandatory receipt-citation test passes 17 tests. The round-336 receipt
gate finds 13 citations with zero failures, zero unmapped citations and zero
map-audit failures; all nine self-test controls fire. Its JSON SHA-256 is
`9d778eb9257c55c09674e423acab16072e356b8f143df76d23b7615d998b7b8f`.

The default ORCA2 landing battery passes 146 tests in 1,462.44 seconds; log
SHA-256 is
`8ddc3ec81940098fa911966fcd181d4b003a3a1522bcfaee1a107e8bc5c42f9b`.
The all-package private-import ratchet separately passes 6 tests in 6.42
seconds; log SHA-256 is
`50cb70694b877bb2a88d4cf1a09018216f131123bea992f586ce03de4cdd9546`.
No DINO month gate is required because no model or package file changed.
