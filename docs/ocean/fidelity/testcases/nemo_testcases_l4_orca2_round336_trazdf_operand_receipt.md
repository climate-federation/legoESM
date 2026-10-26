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
the final kt-2 `avt` or `avs` after the full `zdf_phy` composition, nor does any
admitted record contain the evaluated `avt+akz` or `avs+akz` coefficient. The
inherited inner-LDF record contains `akz` and the older closure-level `avt_k`;
those fields cannot reconstruct the post-river/EVD/DDM/IWM coefficient. Thus
this is **not** an “every input exact” result. The acquisition below records
final `avt`, final `avs`, `akz`, and their evaluated effective coefficients.

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
loops, records final `avt`, `avs`, `akz`, geometry, RK levels and live `Kaa`.
Five additional calls outside `zdf_phy`'s loops record the kt-2 boundaries
after the TKE `avt_k` copy, runoff-mouth addition, EVD, DDM and IWM in the
executing order at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfphy.f90:365-392`.
This distinguishes a coefficient departure at each executed contributor
instead of attributing from final `avt`/`avs` alone. The passive writer replays
NEMO's exact statement association into lower/diagonal/upper, eliminated,
forward and back-substitution fields. Admission requires the
outside-loop back-substitution to equal live `Kaa` bit-for-bit on every
executed cell. The RHS replay uses NEMO's literal
`p2dt*(e3t_3d*(1+r3t(Kmm)*tmask))*Krhs` association at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:283-291`.
The rejected `((p2dt*e3t_3d)*(1+...))*Krhs` association is both a gate plant
and a required wet-cell discriminator in each admitted rank.

The clean read-only preflight passed every inherited round-333 control,
including 77 parent tests, both inherited writer suites, 140-file deterministic
passivity, 11 syntax checks, synthetic in-place and retained admission routes,
and all inherited plants. The new two-rank fixture compiled and ran the actual
patched `trazdf` source under bounds checking and compared it bitwise with a
separately compiled unpatched module using rank-distinct data and signed-zero
cells. Its passive 36-field replay matched the live patched-source result.
Every derived field now retains `-0.0` outside exactly the ranges filled at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:180-216` and
`:230-235`; a filled-range plant fires. The record header carries the literal
RK tuple and tile bounds, and a time-level plant perturbs the tuple and fires.

The production-flag disassembly census is per function and nonzero:
`tra_zdf` has 60 floating-point instructions and `tra_zdf_imp` has 1,234,
unchanged by the observer; an arithmetic plant injected into `tra_zdf` fires.
The observer-free versus instrumented `zdf_phy` census is likewise unchanged
at 121 instructions, with its own in-function plant. All selector, placement,
header, field-order, truncation, checksum, extent, finite-value, signed-zero,
wet-coverage, exact-RHS-association, contribution-boundary, replay and
actual-result plants fired. All 147 staged `namelist_cfg` assignments belong
to the same group in NEMO's `namelist_ref`; misspelled and misplaced plants
both refuse. A synthetic `ocean.output` containing `E R R O R` refuses while
printing NEMO's matching line. The extended same-program proof reports 407
identical ppsrc and 419 identical normalized objects, exactly seven observer
and five writer units. Both in-place and retained `--admit-existing` routes were rechecked
against each run's manifest: the producer must be a local ancestor and the
binary must match the manifest SHA-256; producer and binary plants fired on
both routes. The refreshed preflight passes; its log is stored as
`round336/resume2_preflight.log`.

Operator command:

```text
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r336/repo/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_trazdf_acquisition/run.sh --run
```

Expected evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/acquisition/orca1ice_trazdf_internal_kt2_np2`.

No configuration value changed: **ASKED**. Independent Argon review verdict:
**HOLD** on fill ranges, function-scoped passivity, checklist items 6/12, and
coefficient sufficiency; every listed item is addressed above. Codex review:
operator.

The mandatory receipt-citation test passes 17 tests. The round-336 receipt
gate finds 13 citations with zero failures, zero unmapped citations and zero
map-audit failures; all nine self-test controls fire. Its report is stored as
`round336/resume2_citation_gate.json`.

The default ORCA2 landing battery passes 146 tests; its log is stored as
`round336/resume2_default_gate.log`. The all-package private-import ratchet
separately passes 6 tests; its log is stored as
`round336/resume2_ratchet.log`.
No DINO month gate is required because no model or package file changed.

## Resume 3: declared extents and refusal-path repair

Claude Opus 4.6 returned RUN. Argon's re-review held the acquisition on five
remaining tool defects; all five are repaired without running NEMO or changing
a model statement, configuration value, observer placement, or score.

The decisive extent is now read from NEMO rather than inferred from the tracer
grid. NEMO allocates `avs`, `avt`, `avt_k`, and `en` with `A2D(0),jpk` at
`zdf_oce.F90:70-72`. Therefore final `avt`/`avs`, saved `avt_k`, and all eleven
recorded `zdf_phy` phase fields are written and parsed at
`(ntsi:ntei,ntsj:ntej,jpk)`, which is `(90,148,31)` in this build; full-grid
geometry, `akz`, masks, tracer state, and replay arrays retain their own
declared extents. The bounds fixture allocates the actual A2D fields at those
lower bounds and runs the patched NEMO `trazdf` source on two rank-distinct
inputs. A source-allocation plant and record-metadata extent plant both fire.

Every value plant now selects a cell from the live wet/owned mask instead of
the former fixed `(2,2,1)` slot. Each saved phase has its own named wet-cell
checksum plant; the gate also enforces the TKE copy, the unchanged salinity
across runoff and EVD, and the final IWM-to-`tra_zdf` handoff. Thus none of
`avt_after_runoff`, `avs_after_runoff`, `avt_after_evd`, `avs_after_evd`,
`avt_after_ddm`, `avs_after_ddm`, `avt_after_iwm`, or `avs_after_iwm` is merely
parsed and ignored.

The inherited parent now scans `ocean.output` and prints matching
`E R R O R` lines before refusing a nonzero `mpirun` status. Its preflight
executes that refusal plant at the reachable parent check, and a structural
test pins the check before the status refusal. Failed target builds are kept
in place rather than deleted. The acquisition README gives the exact
move-aside-before-rerun procedure; refused run evidence remains under
`refused/`.

Both admission recovery routes remain live. The synthetic in-place call now
explicitly removes any operator-provided `R328_EXISTING_RUN` before entering
the inherited route; the retained call supplies its own path. The round-333
and round-336 retention traps arm before invoking inherited
`--admit-existing`, so any later refusal retains a run already moved to its
canonical name. The preflight executes both complete synthetic routes plus
producer-commit and binary-digest plants for each.

Finally, the same-program setup preprocesses the pristine `TRAZDF_SOURCE`
with the build's own `cpp`, requires its nonblank output to equal the real
record-build ppsrc, and only then applies the observer patch. A nonblank
same-program plant fires in both the direct and inherited checks. The refreshed
preflight again reaches `ORCA2_ROUND336_TRAZDF_PREFLIGHT_READY`, with 140-file
deterministic passivity, 11 syntax proofs, the nonzero 60/1,234/121 function
censuses, 407 identical ppsrc units, 419 identical objects, both admission
routes, the real-extent bounds fixture, and every new plant green/fired as
appropriate. Its final log is stored as `round336/resume3_preflight.log`.

The per-receipt citation gate finds 14 citations with zero failures, zero
unmapped citations, and zero map-audit failures; all nine self-tests fire. The
default ORCA2 landing battery passes 146 tests in 710.51 seconds, and the
all-package private-import ratchet passes 6 tests. Their logs are stored as
`round336/resume3_citation_gate.json`, `round336/resume3_default_gate.log`, and
`round336/resume3_ratchet.log`.

The acquisition disposition is unchanged: **ACQUISITION_NEEDED** at the same
operator command and evidence directory above. No unasked choice was made;
configuration choices remain empty. Codex review: operator. Argon findings
are dispositioned here; operator re-review remains required before acquisition.

## Resume 4: phase-content admission repair

Argon's re-review found two remaining verification defects. The phase plant had
selected the first wet owned cell at Fortran `jk=1`, outside the captured
`2:jpkm1` process range, and therefore proved only the unfilled-sentinel check.
It now selects a wet cell from the process range. Every phase plant runs after
the stream checksum has passed and asserts its exact expected refusal message;
the preflight log shows all fifteen named phase refusals rather than accepting
any refusal as sufficient.

Admission now checks the actual process-range content between every recorded
boundary. The closure copy is exact; runoff cannot decrease `avt` and leaves
`avs` unchanged; EVD leaves `avs` unchanged and any changed `avt` value must be
the executed deck's `rn_evd*wmask`; DDM cannot decrease `avt` and cannot set
`avs` below its incoming `avt`; IWM cannot decrease either tracer coefficient;
and both last-phase fields must equal the final solver inputs bit-for-bit. These
properties follow the executed composition at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfphy.f90:365-392`, the EVD
assignment at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfevd.f90:107-110`, the DDM
updates at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfddm.f90:163-172`, and the
non-negative IWM clamp and additions at
`ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfiwm.f90:293-317`.
All eleven phase fields must also be finite and non-negative over the process
range. The preflight pins the four deck values used by these implications and
plants each numeric log echo.

The refreshed read-only preflight reaches
`ORCA2_ROUND336_TRAZDF_PREFLIGHT_READY`. Its two-rank bounds/replay fixture and
26 direct gate tests pass; every phase plant reports the intended message; the
60/1,234/121 nonzero per-function floating-point censuses remain unchanged;
both admission recovery routes pass; and the inherited suites pass 6, 5, and
78 tests. The log is `round336/resume4_preflight.log`, SHA-256
`b7f61a382dbac8eed67ac53a38857fe7d854628b4563707e1dc3abb3a2915ce6`.

The acquisition disposition remains **ACQUISITION_NEEDED** at the operator
command above. No NEMO executable was run, no model or configuration changed,
and no unasked choice was made. Codex review: operator.
