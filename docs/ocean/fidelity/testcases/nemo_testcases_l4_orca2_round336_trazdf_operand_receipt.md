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

The per-receipt citation gate reports PASS for 17 citations with zero failures,
zero unmapped citations, zero map-audit failures, and all nine self-tests fired.
The exact default ORCA2 landing battery passes 152 tests in 1,095.21 seconds;
its log SHA-256 is
`eb3651e27a73de3514417f9f64b0c8f8fdaa10fdc20dddae364b7a4ba15e5333`.
The private-import ratchet also passes independently, 6 tests in 4.86 seconds;
its log SHA-256 is
`43965bf37d47a88cf243b415e98cb6dcfb35b76c1d222723fee50d1a093276b1`.
The logs are `round336/resume4_citation_gate.json`,
`round336/resume4_default_gate.log`, and `round336/resume4_ratchet.log`.

## Resume 5: generated-driver syntax and flow closure

The operator's reviewed `--run` invocation compiled the target and then stopped
before NEMO started. The generated round-333 inner driver had split an injected
writer-presence check after `>/dev/null` without preserving the shell line
continuation, leaving a bare `||` at generated line 1046. The full failure log
is `round336/acquisition_refused_20261026T145001Z/acq_r336.log`. The retained
build at
`cfgs/ORCA2_ORCA1ICE_OMIP_L4_R336TZDF.retained_20261026T145001Z_driver_syntax_error`
was not touched.

The injection is now one syntactically atomic line. The preflight generates the
adapted parent and the exact inner script that `--run` will execute and runs
`bash -n` on each before any build. A plant appends invalid shell syntax to each
generated layer and both checks refuse. Both generated scripts are then run
end to end against a stub NEMO through four cases: quoted `E R R O R` refusal,
nonzero-exit retention, manifest-pinned `--admit-existing`, and `STOP 0`
success. Both layers report all four cases passing.

ACQUISITION TOOL CHECKLIST item 15: every generated script is produced and
syntax-checked and its control flow run with a stub NEMO in the preflight.

The refreshed read-only preflight reaches
`ORCA2_ROUND336_TRAZDF_PREFLIGHT_READY`; it also retains all earlier fixture,
passivity, same-program, admission, and plant results. Its log is
`round336/resume5_preflight.log`, SHA-256
`cc4100530129237927644929ef3c9f50d1b362b54795b2693411403e5747ae1c`.
No NEMO executable was run, no build was started, no model or configuration
changed, and no unasked choice was made. Codex review: operator.

The disposition remains **ACQUISITION_NEEDED** at the operator command above.
Before retrying, the operator must keep the retained failed build untouched and
use the already-clear canonical target name, as documented in the acquisition
README.

The final per-receipt citation gate reports PASS for 17 citations with zero
failures, zero unmapped citations, zero map-audit failures, and all nine
self-tests fired. The exact default ORCA2 landing battery passes 152 tests in
1,062.72 seconds; its log SHA-256 is
`150d8e977e2605dfd88ab8f12149b1c12b076744d3b13778fce92bb826d36011`.
The private-import ratchet separately passes 6 tests in 7.48 seconds; its log
SHA-256 is
`208fb3acdcdd60c220457676d81bb9c0ea34a5bb5c82d189f82a95f8b90fd133`.
The reports are `round336/resume5_citation_gate.json`,
`round336/resume5_default_gate.log`, and `round336/resume5_ratchet.log`.

## Resume 6: generated-driver real-flow closure

Argon's re-check confirmed the resume-5 syntax repair and held checklist item
15 because the preflight still intercepted each generated script near its top.
The former helper independently mimicked retention and admission, then exited;
therefore a syntactically valid truncation of either real driver body remained
green.

The interception is removed. The flow test now invokes each generated driver
itself. Its isolated writable build and evidence roots replace `makenemo` and
`mpirun` only at their executing call sites, and the resulting `nemo.exe` is
the scenario stub. The inherited mode dispatch, source and selector checks,
syntax and same-program proofs, build/run creation, `E R R O R` scan, nonzero
exit handling, cleanup trap, refused-run movement, manifest identity checks,
and canonical retained-run movement all execute in the generated scripts.
The direct inner-driver copy retains the exact combined writer and exported
placement helper that its production parent supplies while the child runs.

For both the generated parent and generated inner driver, the preflight now
passes five cases: successful `--run`, quoted `E R R O R` refusal, retention
after a nonzero NEMO exit, retained `--admit-existing`, and in-place
`--admit-existing`. The stub admission still verifies a local ancestor
producer and the copied NEMO binary's SHA-256. Each layer also has a
syntactically valid body-truncation plant: the parent is cut before its
post-child admission tail and the inner before its admission/build/run body;
both are rejected by the generated-body contract.

The refreshed read-only preflight reaches
`ORCA2_ROUND336_TRAZDF_PREFLIGHT_READY`. Both generated layers report their
five-case flow pass and body-truncation plant. All inherited 78-test, 6-test,
and 5-test batteries, 140-file deterministic passivity, eleven Fortran syntax
proofs, 1,367- and 1,966-instruction inherited censuses, the 60/1,234/121
round-336 function censuses, the two-rank bounds fixture, and the 407/419
same-program proof remain green. The log is `round336/resume6_preflight.log`,
SHA-256 `3a01f0800dc2360c75f49a2c567c6dd756c2e91a92ee1de07bfc86df3d2af832`.

The disposition remains **ACQUISITION_NEEDED** at the operator command above.
No real NEMO executable was run, no model or configuration changed, and no
unasked choice was made. Codex review: operator.

The final per-receipt citation gate reports PASS for 17 citations with zero
failures, zero unmapped citations, zero map-audit failures, and all nine
self-tests fired. The exact default ORCA2 landing battery passes 152 tests in
1,098.95 seconds; its log SHA-256 is
`3be6e54c9f8fd431d27e02f3b32bc14c10bc86b539bf8794c6d09f539c456fcf`.
The private-import ratchet separately passes 6 tests in 6.11 seconds; its log
SHA-256 is
`1742d0160ea02bfb39c1243f53c27353a25868b9a893b6e9c285559b7f172ce9`.
The reports are `round336/resume6_citation_gate.json`,
`round336/resume6_default_gate.log`, and `round336/resume6_ratchet.log`.

## Resume 7: admitted `tra_zdf` statement walk

The manifest-pinned retained-record route admitted producer
`5e9bf388bd0ab4f08b16af86517e4adbe55f6204` with copied executable SHA-256
`24420ae44505ece369af01fcb8303d9e5881ae4ab70e5bd128153ffee3188328`.
Its final line was exactly:

```text
ORCA2_ROUND336_TRAZDF_ACQUISITION_PASS /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/acquisition/orca1ice_trazdf_internal_kt2_np2
```

The admitted build's actual `zdf_phy` copies the TKE coefficient and then
executes runoff, EVD, DDM, and IWM in that order at
`ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/zdfphy.f90:369-404`.
Its non-Aimp `tra_zdf` builds effective K and the lower, upper, and diagonal
rows at
`ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90:185-240`, then
executes eliminated-diagonal, forward, and back-substitution statements at
`ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90:273-278`,
`ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90:288-296`, and
`ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90:298-304`.

The production replay observes legoESM's closure handoff at
`k_profiles.py:329-348`; the shared tracer call, matrix construction, and
three recurrences are the actual production functions at
`implicit_solver.py:478-539`, `implicit_solver.py:542-641`, and
`implicit_solver.py:329-392`. The extra callbacks are passive: each trace-graph
observation equals the corresponding ordinary-reference observation bit for
bit, and both returned production states change zero leaves relative to their
ordinary calls.

Both labels produced the following identical kt-2 walk. “Target faces” are
the W faces adjacent to round 333's 18 registered T words; because that
adjacency includes dry faces, the wet-face column is the ownership census.

| executed output | wet unequal words | wet maximum | target unequal words | target maximum |
|---|---:|---:|---:|---:|
| TKE `avt_k` copy | 17 | `6.361019669726387e-13` | 16 | `1.0e-10` |
| runoff `avt` add | 17 | `6.361019669726387e-13` | 16 | `1.0e-10` |
| EVD `avt` replace | 17 | `6.361019669726387e-13` | 16 | `1.0e-10` |
| DDM `avt` update | 17 | `6.361019670744853e-13` | 16 | `1.0e-10` |
| DDM `avs` write | 17 | `6.361019669725353e-13` | 16 | `1.0e-10` |
| IWM `avt` add | 17 | `6.361019670744853e-13` | 16 | `1.0e-10` |
| IWM `avs` add | 17 | `6.361019669719149e-13` | 16 | `1.0e-10` |
| effective temperature K | 17 | `6.361019670744853e-13` | 9 | `6.361019670744853e-13` |
| lower matrix row | 17 | `6.874397878199601e-10` | 9 | `6.874397878199601e-10` |
| upper matrix row | 17 | `6.874397878199601e-10` | 8 | `6.874397878199601e-10` |
| diagonal matrix row | 25 | `6.874447677773787e-10` | 14 | `6.874447677773787e-10` |
| eliminated diagonal | 25 | `6.874447677773787e-10` | 14 | `6.874447677773787e-10` |
| content RHS | 0 | `0.0` | 0 | `0.0` |
| forward recurrence | 13 | `1.0226131053059362e-09` | 8 | `1.0226131053059362e-09` |
| back substitution | 29 | `7.678080393702658e-12` | 18 | `7.678080393702658e-12` |
| Kaa update | 29 | `7.678080393702658e-12` | 18 | `7.678080393702658e-12` |

Thus the first measured wet mismatch is **CONFIRMED** at the first boundary,
the TKE `avt_k` copy: 17 wet W words, maximum
`6.361019669726387e-13 m2 s-1`. It precedes every runoff/EVD/DDM/IWM and
`tra_zdf` arithmetic statement. The walk nevertheless recovers round 333's
registered terminal signature exactly: 18 T words after back substitution and
the Kaa update, maximum `7.678080393702658e-12 K`, whose unique maximum is
`(i,j,k,rank)=(32,147,5,0)` on the north fold under ice.

Ownership by a particular statement *inside* the shared TKE closure is only
**PLAUSIBLE**: this record starts at the TKE-to-`zdf_phy` handoff and therefore
cannot distinguish the upstream TKE statements. No single cited shared
statement owns the departure, so Decision 122 forbids a literal fix and the
conditional M1--M6, seven-pair, other-caller, and gradient landing work does
not open. No model statement or configuration value changed.

Proposal: extend the same manifest-pinned kt-2 record with passive boundaries
through the shared TKE closure, then replay its source order up to the 17-word
`avt_k` handoff. Only a single first internal statement whose literal repair is
bit-identical or strictly better on all seven Decision-122 pairs and both
labels may proceed to the production-call-site landing path.

The two `PASS_R336_TRAZDF_STATEMENT_REPLAY` artifacts are
`round336/resume7/given.json` (SHA-256
`0e40d493f2fd3e48ba1e0d6fe07dcb2000a395c138cb1cf644bbb3ca284bc578`)
and `round336/resume7/independent.json` (SHA-256
`d6329d6f6a13a45ca929833af2ae5512d4e577c8ebe9117bc92dbbd7cab9a17c`).
The statement replay's seven fail-closed classifier tests pass. Codex review:
operator.

The final per-receipt citation gate reports PASS for 9 citations with zero
failures, zero unmapped citations, zero map-audit failures, and all nine
self-tests fired; its report SHA-256 is
`e73b9ee91280adc6dd2191242799bd36fb3a1ece2f962a5254f3789ab427c75d`.
The exact default ORCA2 landing battery, extended by the seven new replay
controls, passes 159 tests in 1,027.08 seconds; its log SHA-256 is
`c0e64332517d8584ef01eb859f4b7875a235671f6e5fdc4a9f5479534d6da49e`.
The private-import ratchet separately passes 6 tests in 4.85 seconds; its log
SHA-256 is
`357552e211123f746071b1da46e34b619513dc21077812ffe469614a71dd6b28`.
The reports are `round336/resume7_citation_gate.json`,
`round336/resume7_default_gate.log`, and `round336/resume7_ratchet.log`.

## Resume 8: wet-target repair and kt-2 TKE record census

The replay target is now the intersection of the registered adjacent W faces
and NEMO's actual W mask. This removes seven dry bottom faces from the target
census; it does not remove any of the 17 genuine wet `avt_k` differences. The
`first` control is no longer vacuous: its planted record clears the metric of
the first source-ordered row while retaining the old first-row marker, and the
classifier refuses that inconsistent record. Reverting either repair makes its
focused production-call-site test fail.

Both labels reran the production trace and returned
`PASS_R336_TRAZDF_STATEMENT_REPLAY`. Their corrected, identical kt-2 table is:

| executed output | all-wet unequal/support | all-wet maximum | target unequal/support | target maximum |
|---|---:|---:|---:|---:|
| TKE `avt_k` copy | 17/414119 | `6.361019669726387e-13` | 9/18 | `6.361019669726387e-13` |
| runoff `avt` add | 17/414119 | `6.361019669726387e-13` | 9/18 | `6.361019669726387e-13` |
| EVD `avt` replace | 17/414119 | `6.361019669726387e-13` | 9/18 | `6.361019669726387e-13` |
| DDM `avt` update | 17/414119 | `6.361019670744853e-13` | 9/18 | `6.361019670744853e-13` |
| DDM `avs` write | 17/414119 | `6.361019669725353e-13` | 9/18 | `6.361019669725353e-13` |
| IWM `avt` add | 17/414119 | `6.361019670744853e-13` | 9/18 | `6.361019670744853e-13` |
| IWM `avs` add | 17/414119 | `6.361019669719149e-13` | 9/18 | `6.361019669719149e-13` |
| effective temperature K | 17/414119 | `6.361019670744853e-13` | 9/18 | `6.361019670744853e-13` |
| lower matrix row | 17/430552 | `6.874397878199601e-10` | 9/18 | `6.874397878199601e-10` |
| upper matrix row | 17/430552 | `6.874397878199601e-10` | 8/18 | `6.874397878199601e-10` |
| diagonal matrix row | 25/430552 | `6.874447677773787e-10` | 14/18 | `6.874447677773787e-10` |
| eliminated diagonal | 25/430552 | `6.874447677773787e-10` | 14/18 | `6.874447677773787e-10` |
| content RHS | 0/430552 | `0.0` | 0/18 | `0.0` |
| forward recurrence | 13/430552 | `1.0226131053059362e-09` | 8/18 | `1.0226131053059362e-09` |
| back substitution | 29/430552 | `7.678080393702658e-12` | 18/18 | `7.678080393702658e-12` |
| Kaa update | 29/430552 | `7.678080393702658e-12` | 18/18 | `7.678080393702658e-12` |

The artifacts are `round336/resume8/given.json` (SHA-256
`b442f8fd7e198396b01a481585141f357f17cc97e58be846b3911f26b059f74e`)
and `round336/resume8/independent.json` (SHA-256
`5d022117c4e7d8999ab1426cf1df1b09f8669b9864abfb662b7114ea59507490`).
Their raw traced companions are SHA-256
`87061db067300a33d36365c685daae9047139ee4c526d9e3a52340e67391a3cc`
and `cc147b4b77229ff9a823b502b7dd0a3cb8ec944b54009f0c32c446c17f0d6808`.

The cheapest-next-step census found that the admitted round-326 kt-2 files
contain only `hm_i`, `fr_i`, `taum`, and the surface T mask, exactly the four
fields written by `zdftke_round326_writer.F90:34-69`. Round 331 has TKE
internal records only for kt 1; round 332 likewise has only its kt-1
`zmxlm`/`zmxld`/`pdlr` record. There is no admitted
`oracle_tke331_kt00000002_*` or `oracle_r332_source_kt00000002_*` file. Those
records therefore cannot replay the executing `tke_tke` followed by `tke_avn`
calls at
`ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/zdftke.f90:197-200`.
No internal TKE statement can be labeled **CONFIRMED** from existing evidence,
so Decision 122 does not permit a literal production fix or M1--M6 landing.

The acquisition-only extension activates the already reviewed round-331
statement writer at both kt 1 and kt 2, with exact kt-2 levels
`(Kbb,Kmm,Krhs)=(3,3,1)`, at `tke_round331_writer.F90:51-75`. It also records
the full kt-2 mixing-length and Prandtl fields through the existing round-332
in-loop hook at `round332_writer.F90:82-125`; the separate three-field output
is defined at `round332_writer.F90:127-145`. Its production flags, decomposition,
array bounds, record headers and manifests remain pinned, and it changes no
NEMO arithmetic.

The read-only preflight passes 88 focused tests, its inherited 78-test battery,
all refusal plants, 140-file deterministic passivity, eight Fortran syntax
proofs, a 1,367-instruction stage census, the nonzero 783-instruction
`tke_avn` census, and the 421-object/409-ppsrc same-program proof. The log is
`round336/resume8_tke_preflight.log`, SHA-256
`c37cf567690e99692b433c5272586c510ebd5ec0dcaec1f6fbf41e730b248b8b`.
It ends at `ORCA2_ROUND336_TKE_KT2_PREFLIGHT_READY`; no NEMO run or build was
started. No configuration or shared model statement changed. Codex review:
operator.

The required operator handoff is:

```text
ACQUISITION_NEEDED: scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh
```

## Resume 11: passive boundary repair and admitted-recorder census

Argon item 3 is resolved for the new acquisition. When the target is
`tke336`, the shared driver neither stages nor applies the round-328
`dynzdf`/stage patches, and its actual target-build branch has the same
exclusion
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:764-771`,
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1475-1479`).
The same-program result consequently contains only observer unit `zdftke` and
writer unit `tke_round336_writer`: 425 inherited objects and 413 inherited
preprocessed sources are identical. The round-328 writer is compiled only in
an isolated historical-audit scratch directory to supply the checksum module
needed to compile the historical round-331 writer; it is not linked into the
new target.

The final patch has three calls, all in outer `zdf_tke`, and 17 assignments
after completed loop nests. The state-changing Langmuir `en` nest, the matrix
construction, each of the five Thomas phases, the pre-sweep state, `lup`,
`ldown`, the merge, and the pre-Prandtl state now have distinct copy-outs
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/zdftke_round336.patch:13-52`).
The 51-field record writes those boundaries separately, including
`post_langmuir_en`, `solve_diag`, `solve_rhs`, `solve_bottom`, `solve_back`,
`after_lup_mxld`, and `after_ldown_mxl`
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/tke_round336_writer.F90:232-284`).
`entry_avm_k` retains its full `jpi,jpj,jpk` allocation and the semantic plant
explicitly probes the north halo at `jj=ntej`; therefore the 17 observed
`avt_k` differences at `jj=150` have the required `avm_k(:,151,:)` input.

Direct copy-outs of the six Langmuir-local scratch arrays were also tested.
They changed the inlined `zdf_tke` FP multiset (1,007 production instructions
versus 1,012 observed), so retaining them would violate the passivity rule.
Removing those reads restores exact 1,007/1,007 equality; the complete entry
operand set plus the post-Langmuir `en` boundary remains sufficient to replay
the local scratch loops. The Thomas and `lup`/`ldown` split boundaries do not
change the census. This is a measured limitation: no claim is made that a
directly recorded Langmuir scratch value is production arithmetic.

### Historical admitted-recorder audit

The preflight compiles observer-free sources at the real record flags and
compares each named function's normalized FP-opcode multiset with the
SHA-pinned executable that actually produced the admitted record. It also
pins zero-FP/inlined topology instead of silently omitting such functions
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1074-1149`).

| admitted tool | patched/touched function | production FP | producer FP | verdict |
|---|---|---:|---:|---|
| round 5 full-entry | `stp_rk3` | 30 | 30 | equal |
| round 326 HMI | `zdf_tke`; `tke_avn` | 1,007; 783 | 1,007; 783 | equal; equal |
| round 328 dyn walk | `dyn_zdf`; `stp_rk3_stg` | 2,446; 1,175 | 2,365; 1,367 | **different; different** |
| round 328 ZDF terms | `zdf_phy`; `zdf_ddm`; `zdf_iwm` | 121; 58; 423 | 121; 58; 423 | equal; equal; equal |
| round 331 TKE | `zdf_phy`; `zdf_tke`; `tke_tke`; `tke_avn` | 121; 1,007; inlined; 783 | 121; 783; 1,054; inlined | equal; **different; different; different** |
| round 332 source order/raw V | `zdf_phy`; `vor_een`; `dyn_vor`; `stp_rk3_stg`; `zdf_tke`; `tke_tke`; `tke_avn` | 121; 555; zero; 1,175; 1,007; inlined; 783 | 121; 555; zero; 1,367; zero; 1,054; 783 | equal; equal; equal; **different; different; different**; equal |
| round 333 tracer | `stp_rk3_stg` | 1,175 | 1,367 | **different** |
| round 333 internal LDF | `stp_rk3_stg`; `traldf_iso_lap`; `traldf_iso_a33` | 1,175; 526; 538 | 1,367; 526; 538 | **different**; equal; equal |

Thus the admitted round-328 `dyn_zdf` records came from a different arithmetic
program, as did the stage-bearing round-333 records. “Different” is a
provenance result, not an automatic rewrite of an earlier receipt. The exact
earlier conclusions that rest on the affected records are listed here:

- Round 328's ten-boundary exact `dyn_zdf` replay, its statement-localized
  input/drag/matrix/Thomas claims, and the conclusion that the first base-lower
  matrix statement consumes rather than creates the coefficient departure all
  depend on the non-production `dyn_zdf` record. Its later ZDF-term record has
  equal censuses for all three newly instrumented ZDF functions, but it was
  produced by a whole executable that still carried the differing inherited
  dyn/stage program.
- Round 331's first-TKE-mismatch family, exact recurrence/returned-energy
  claims, seven residual `avm/avt` cells, and stored-`Kbb` bottom ownership
  depend on the differing TKE program. Round 332's seven-cell mixing walk,
  `lup` anchor, viscosity-order exclusion, Prandtl conclusion, and raw-V/fold
  stage record likewise depend on differing TKE or stage functions. This is
  the same scope identified in Resume 9, now backed by the actual producer
  binaries rather than reconstructed historical objects alone.
- Round 333's admitted stage-boundary counts and `tra_ldf` call-boundary
  attribution, and every later internal-LDF/divisor or post-LDF/`tra_zdf`
  conclusion that consumes those stage records, come from a whole program
  whose `stp_rk3_stg` differs. The two directly instrumented
  `traldf_iso_lap/a33` functions themselves have equal FP multisets. The
  receipt's reverse-safe landing also has independent Decision-122, DINO, and
  gradient evidence; this audit records its NEMO-record dependency without
  rewriting the receipt.
- Round 5, round 326, and the directly instrumented round-328 ZDF-term
  functions are census-equal, so this audit identifies no affected conclusion
  from their touched functions.

### Final preflight and handoff

The final read-only preflight passes 22 focused tests and the inherited
48-test acquisition battery, 20 placement plants, two syntax proofs, the
51-field two-rank bounds fixture, exact new-tool FP counts
(`zdf_tke=1007`, `tke_avn=783`), the complete historical table above, all 18
admission plants, same-program identity, and the five stub flows (success,
in-place, quoted refusal, crash retention, retained admission). Its log is
`round336/resume11_tke_preflight.log`, SHA-256
`03235710d1e270e5497843b388a2d72fcfac9846406034116c8395dd72f7195f`.
It ends at `ORCA2_ROUND336_TKE_KT2_PREFLIGHT_READY`. No NEMO build or
production run was started. Codex review: operator.

The Resume-11 per-receipt citation gate passes all 15 citations with zero
failures, zero unmapped citations, zero map-audit failures, and all nine
self-tests fired. The exact default ORCA2 landing battery passes all 152 tests
in 1,118.70 seconds; its log SHA-256 is
`0c419e9d0774f09bd85713c1978d7a43f0fc7160aa96ea1290dd80fb0a08c8ec`.
The explicit private-import ratchet passes all 6 tests in 5.55 seconds; its log
SHA-256 is
`4457c438a4ea2ef2ae5630501eaba3193e5fd0762710c36e2464d5b3c160d399`.
The reports are `round336/resume11_citation_gate.json`,
`round336/resume11_default_gate.log`, and `round336/resume11_ratchet.log`.
No DINO month gate is required because no production package or model source
changed.

The round stops for the corrected production-arithmetic record:

```text
ACQUISITION_NEEDED: scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh
```

The final per-receipt citation gate reports PASS for 8 citations with zero
failures, zero unmapped citations, zero map-audit failures, and all nine
self-tests fired; its report SHA-256 is
`f164f0630d018b7dcc6cae5a27fe403c310fb5f27c246e8b724631670e231ff7`.
The exact default ORCA2 landing battery passes 152 tests in 1,101.26 seconds;
its log SHA-256 is
`465bc3ef6b2e0c08751f9fef778fa03ff9a7c552b2e8904af62ff64a395af206`.
The private-import ratchet separately passes 6 tests in 5.20 seconds; its log
SHA-256 is
`f0d5ceafde69f4e360158e199bcba4d954a930ce4d84356b74b9041464fd256c`.
The reports are `round336/resume9_citation_gate.json`,
`round336/resume9_default_gate.log`, and `round336/resume9_ratchet.log`.

The final per-receipt citation gate reports PASS for 5 citations with zero
failures, zero unmapped citations, zero map-audit failures, and all nine
self-tests fired; its report SHA-256 is
`d246220382f843dce91fe9fd6bc090b962d67e57b36a4d6d86f32d9ad45c1552`.
The exact default ORCA2 landing battery, extended by the two replay repairs
and eleven acquisition controls, passes 172 tests in 1,026.90 seconds; its log
SHA-256 is
`696c4359925ef2dec4ce77b02547fd20e8bc5d141d13b7de58a5ad85d3817c2d`.
The private-import ratchet separately passes 6 tests in 4.99 seconds; its log
SHA-256 is
`a199421a3a3c3466f4b0cf403655ab82284f7e63e97b56a4c121389f811fd16d`.
The reports are `round336/resume8_citation_gate.json`,
`round336/resume8_default_gate.log`, and `round336/resume8_ratchet.log`.

## Resume 9: historical observer audit and production-arithmetic kt-2 TKE handoff

The admitted round-331 and round-332 TKE records did **not** come from the
production arithmetic program.  Their calls are inside NEMO's `jj` loop:
round 331 inserts four row observers across `tke_tke` at
`zdftke_round331.patch:8-24`, and round 332 retains those calls and adds its
mixing-length row call at `zdftke_round332.patch:12-23`.

At the real record flags (`-fdefault-real-8 -O3 -funroll-all-loops
-fcray-pointer -ffree-line-length-none -fallow-argument-mismatch
-fno-tree-vectorize`), the per-function FP census is:

| source program | `zdf_tke` FP | `tke_tke` FP | `tke_avn` FP |
|---|---:|---:|---:|
| production | 1,007 | fully inlined into `zdf_tke` | 783 |
| admitted round 331 | 783 | 1,054 | inlined into `zdf_tke` |
| admitted round 332 | 0 | 1,054 | 783 |
| new round-336 copy-out observer | 1,007 | fully inlined into `zdf_tke` | 783 |

The normalized opcode multisets of both FP-bearing functions in the new build
equal production exactly; the arithmetic plant changes the `zdf_tke` census.
The historical build independently pins every nonzero count above and refuses
if round 332 unexpectedly regains FP arithmetic in `zdf_tke`
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:996-1072`).
Thus the 1,007-to-1,054 `tke_tke` change is plain evidence that the admitted
round-331/332 TKE arithmetic differs from production, not merely a symbol-name
or record-layout difference.

Earlier-receipt impact is listed here without rewriting those receipts:

- Round 331's record-dependent attribution of the first TKE mismatch to the
  RHS family, its exact recorded recurrence/returned-energy claims, the seven
  residual `avm/avt` cells, and the stored-`Kbb` bottom-boundary ownership all
  rest on the non-production round-331 TKE record. Its independent D122 card,
  DINO, and gradient measurements still describe the landed model candidate,
  but they do not repair the NEMO statement-localization premise.
- Round 332's seven-cell `zmxlm`/`zmxld`/`p_pdlr` walk, `lup` surface-anchor
  ownership, exclusion of literal viscosity order, and literal Prandtl
  conclusion rest on the combined round-331/332 TKE record. The separate
  raw-V/fold acquisition and its later V landing do not.
- Round 333 cites rounds 331/332 only as contextual refinement of earlier
  vertical/TKE boundaries. Its kt-2 tracer records, divisor ownership, and
  landed tracer statement do not consume their TKE arithmetic.
- Round 334 mentions equality to a round-332 rescore artifact; it makes no TKE
  statement-localization inference from the admitted 331/332 programs.
- Round 336 resumes 7--8 do not use a kt-1 TKE record to score their admitted
  kt-2 `zdf_phy`/`tra_zdf` walk. However, resume 8's claim that activating the
  inherited TKE observers “changes no NEMO arithmetic,” and therefore its
  proposed kt-2 acquisition tool, is refuted and superseded here.

The replacement tool has three procedure calls, all in outer `zdf_tke`, and
only direct assignments at the twelve statement boundaries inside
`tke_tke`/`tke_avn`
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/zdftke_round336.patch:13-52`).
The writer allocates full-halo `p_avm/rn2/rn2b/wmask` and interior-only
`A2D(0)` energy/diffusivity/source arrays with exact kt-1/kt-2 level checks
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/tke_round336_writer.F90:80-128`).
Its 46 ordered fields now include entry `en/p_avt`, `rn2/rn2b`, post-Langmuir
`en`, pre-sweep `zmxlm`, and pre-Prandtl `zav`, as well as the previous
boundary, matrix, solve, final-mixing, and final-coefficient values
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/tke_round336_writer.F90:232-284`).

The read-only preflight passes 17 focused tests and the inherited 43-test
battery; syntax-compiles all five relevant Fortran units; fires all 15 real
preprocessed-source misplacement plants; runs the actual patched NEMO routines
at ranks 0 and 1 with bounds checking, real full-halo versus `A2D(0)` extents,
rank-distinct data, and signed zeros; parses 31 fields and fires all parser and
semantic plants; and passes same-program identity for 423 objects and 411
preprocessed sources. The wrapper has its own `EXIT` retention trap
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh:35-53`)
and its stub NEMO exercises success, quoted refusal, crash retention, in-place
admission, and retained admission through the actual driver branches
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh:116-177`).

The preflight log is `round336/resume9_tke_preflight.log`, SHA-256
`d4eeb86f56d810e166fd399f997fb5a8ced37990fb49e231ea067b64bbb35b3d`.
It ends at `ORCA2_ROUND336_TKE_KT2_PREFLIGHT_READY`; no NEMO production run or
build was started. Codex review: operator. The round therefore stops for the
new production-arithmetic kt-2 TKE record.

## Resume 10: real extents, replay sufficiency, and real stub admission

Decision 182's hold is resolved without running NEMO or changing a model
statement or configuration value. The five formerly sequence-associated
fields now use their actual allocations: full `jpi,jpj` for both drag arrays
and `fr_i`, and `A2D(2)` for `nmln/hmlp`. Every other recorder dummy was
checked against its allocating source as part of the same repair. The fixture
now uses the real NEMO extent class for every one of the 46 fields, with
rank-, halo-, and coordinate-dependent values, so the former interior-only
fixture can no longer conceal a full-array association defect. The ordered
record declarations and metadata are at
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/tke_round336_writer.F90:80-128`
and
`:232-284`.

The record now contains `htau`, bottom `uu/vv` at `Kbb`, evaluated `gdepw`,
`e3t`, and `e3w` at `Kmm`, all masks and top/bottom levels, and the three
background coefficient inputs used by `tke_avn`. Its executable source-read
census classifies 30 external array reads into 31 recorded fields, 34 pinned
scalar or selector reads, and seven reads belonging only to inactive branches
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:73-100`).
Removing `entry_htau` makes the read-set plant refuse. Admission additionally
requires full-domain `fr_i` to equal the admitted kt-2 ice-cover source,
`hm_i/taum` to equal their admitted sources, evaluated metrics to equal the
admitted kt-2 phase record, and bottom drag to equal the linear-drag namelist
constant `-0.0004*ssmask`; the source-value plant perturbs the latter and
refuses
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:310-358`).

The stub no longer bypasses `admit()`. Its success and retained-success paths
run all 16 parser/semantic plants followed by the real two-rank admission;
success, in-place re-admission, quoted NEMO refusal, crash retention, and
retained re-admission all pass end to end
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1301-1326`).
The refreshed read-only preflight passes 19 focused tests and the 45-test
inherited battery, real-source bounds execution on both ranks, five syntax
proofs, all 15 placement plants, the read-set plant, the source-value plant,
and same-program identity for 423 objects and 411 preprocessed sources. The
production FP counts remain `zdf_tke=1007`, `tke_avn=783`, with `tke_tke`
inlined in both builds.

The opcode proof is deliberately described as a multiset: it cannot detect a
MIN/MAX operand swap. The admission backstop is the independent byte compare
over every inherited `oracle_*.bin` and every step-10 restart file
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:497-511`).
The real stub admission reports `DETERMINISTIC_PASSIVITY_PASS files=160`, a
set that includes all four ocean/ice rank restart files, before the TKE gate
can pass. Thus no passivity claim relies on the opcode multiset alone.

The per-receipt citation gate passes with zero failures, zero unmapped
citations, zero map-audit failures, and all nine self-tests fired. The exact
default ORCA2 landing battery passes 152 tests, and the explicit
private-import ratchet passes all 6 tests. No DINO month gate is required
because no production package or model source changed. Codex review: operator.

The required operator handoff remains:

```text
ACQUISITION_NEEDED: scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh
```
