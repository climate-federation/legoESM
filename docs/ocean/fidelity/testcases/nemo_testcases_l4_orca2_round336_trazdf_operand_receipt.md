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
at `ocean_model_latlon_cgrid.py:14833-14858`. The paired solve consumes the
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

## Resume 16: production kt-2 TKE boundary walk

The retained production-arithmetic record is admitted.  Rank 0 contains
233,341 wet words (224,547 process-interface words), rank 1 contains 197,211
(189,572), for 430,552 full and 414,119 process-interface wet words.  The
walk uses the shared production implementation by advancing the actual ORCA2
comparison card through kt=1 and kt=2, then compares its kt=2 statement trace
and state in NEMO source order.  Only the four Thomas intermediates are
source-order replays from the exact recorded live matrix/RHS; they are not
presented as live JIT observers
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_walk.py:136-238`).

The complete wet-word result is:

| boundary | unequal / wet | maximum absolute difference |
| --- | ---: | ---: |
| after BCs | 0 / 430,552 | 0 |
| after Langmuir | 0 / 430,552 | 0 |
| matrix RHS | 0 / 430,552 | 0 |
| matrix diagonal | 0 / 414,119 | 0 |
| matrix upper | 0 / 414,119 | 0 |
| matrix lower | 0 / 414,119 | 0 |
| Thomas diagonal recurrence | 0 / 430,552 | 0 |
| Thomas RHS recurrence | 0 / 430,552 | 0 |
| Thomas terminal division | 0 / 430,552 | 0 |
| Thomas reverse sweep | 0 / 430,552 | 0 |
| after solve | 17 / 430,552 | 6.589800151726445e-13 |
| after TKE copyout | 17 / 414,119 | 6.589800151726445e-13 |
| pre-sweep mixing length | 17 / 430,552 | 1.4305320720289783e-06 |
| after `lup` | 17 / 430,552 | 1.4305320720289783e-06 |
| after `ldown` | 17 / 430,552 | 1.4305320720289783e-06 |
| final `mxl` | 17 / 430,552 | 1.4305320720289783e-06 |
| final `mxld` | 17 / 430,552 | 1.4305320720289783e-06 |
| `zav` | 17 / 414,119 | 6.3610196697257665e-12 |
| pre-`pdlr` `avt` | 17 / 414,119 | 6.3610196697257665e-12 |
| final `avm` | 0 / 414,119 | 0 |
| final `avt_k` | 17 / 414,119 | 6.361019669726387e-13 m2/s |

Thus the first differing recorded boundary is **CONFIRMED** as the live
post-solve boundary: 17 of 430,552 wet words, maximum
6.589800151726445e-13.  The requested endpoint is independently recovered as
17 of 414,119 wet words with maximum 6.361019669726387e-13 m2/s.  The JSON
receipt is
`round336/walk/tke_production_walk.json`, SHA-256
`b2fd42fcefd2c78a721dded4877f6ce7d311680bc8cfeba7a186be86e2666eb1`.

The record-build source performs the diagonal recurrence, RHS recurrence,
terminal division, reverse sweep, and only then the `MAX`/mask statement at
`ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:480-503`.
legoESM's shared solver expresses the terminal division and reverse scan and
returns the floor/mask expression at
`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:1193-1224`.
Because replay of all four recorded Thomas nests is exact but the compiled
production result first differs after the solve, exact ownership by one
source statement is only **PLAUSIBLE**, not confirmed: compilation can change
materialization across that region.

Three controlled literal probes were therefore tried and reverted: a
rounding barrier around the terminal division, one around the reverse-sweep
quotient, and separated rounding barriers around `maximum` then mask
multiplication.  Each left the same 17-word after-solve and `avt_k`
departures.  No single cited statement owns a literal fix, so the Decision
122 seven-pair/two-label eligibility condition is unopened; there is no fix
proposal and no model code is landed in this round.  The focused walk controls
pass 2 tests.  Codex review: operator.

## Resume 17: compiled production TKE solver intermediate walk

### Preregistration (frozen before measurement)

Resume 16 scored a NumPy source-order replay for `solve_diag`, `solve_rhs`,
`solve_bottom`, and `solve_back` rather than the compiled intermediates of
legoESM's production solver `_nemo_literal_tke_solve`. This resume instruments
a debug-only observer of the exact compiled production solver under the ORCA2
comparison card's `jax.jit` and `PrecisionPolicy.fp64(transcendentals="libm")`
settings (verified bit-identical to the uninstrumented production output),
fed with the admitted kt-2 recorded matrix/RHS from
`round336/acquisition/orca1ice_tke_internal_kt2_np2`:

- **R336-R17-P1 (first compiled intermediate)**: Feeding the admitted kt-2
  recorded matrix/RHS into the compiled production solver reproduces the
  17 / 430,552 wet-word `after_solve` departure (`max_abs =
  6.589800151726445e-13`) and identifies the first compiled solver stage
  (`solve_diag`, `solve_rhs`, `solve_bottom`, or `solve_back`) that departs
  from NEMO's recorded stage. *Falsifier*: if all four compiled stages are
  0 / 430,552 unequal on the recorded matrix/RHS, the departure enters via
  inputs constructed in `_solve_tke_backward_euler` rather than inside
  `_nemo_literal_tke_solve`.
- **R336-R17-P2 (code and HLO mechanism)**: Inspecting the solver source and
  compiled XLA CPU HLO identifies the exact unbarriered expression or
  compiler rewrite (reassociation, reciprocal multiplication, or contraction)
  responsible for the first stage departure (**CONFIRMED** if present in the
  compiled HLO and eliminated by barriering the NEMO statement order;
  otherwise **PLAUSIBLE**).
- **R336-R17-P3 (non-landed literal fix proposal)**: Applying
  `nemo_source_round` barriers to each binary operation in NEMO's written
  statement order removes all 17 wet-word departures (`0 / 430,552` unequal
  across all four Thomas stages and `after_solve`, and `0 / 414,119` on
  `final_avt`). *Falsifier*: if any wet word remains unequal under the
  barriered NEMO statement order, the residual is not owned by intra-statement
  XLA reassociation/contraction alone.

### 1. Instrumented compiled production TKE solver stage walk

A debug-only observer `_instrumented_production_tke_solve`
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_walk.py:245-317`)
executes the exact expressions of legoESM's production solver
`_nemo_literal_tke_solve`
(`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:1111-1224`)
under the ORCA2 comparison card's `jax.jit(..., static_argnames=("source_rounded",))`
(`source_rounded=True`) and `PrecisionPolicy.fp64(transcendentals="libm")`
settings, returning `(solve_diag, solve_rhs, solve_bottom, solve_back,
after_solve, e_new)` while asserting bit-for-bit equality of `e_new` against
uninstrumented `_nemo_literal_tke_solve` (`0 / 772,560` interior words unequal;
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_walk.py:333-547`).

1. **Fed with the admitted kt-2 recorded matrix/RHS** (`matrix_lower`,
   `matrix_diag`, `matrix_upper`, `matrix_rhs_en` from
   `round336/acquisition/orca1ice_tke_internal_kt2_np2`): the compiled
   production solver reproduces NEMO's recorded stages with **0 unequal words**
   (`max_abs = 0.0`) at every stage (`solve_diag`, `solve_rhs`, `solve_bottom`,
   `solve_back`, and `after_solve`) on `wet30` (`0 / 430,552`), on the 11,676
   in-solve seafloor Dirichlet slots `jk = mbkt + 1` (`0 / 11,676`), on
   `active_solve30` (`0 / 442,228`), and on `all30` (`0 / 799,200`). The
   preregistered falsifier of **R336-R17-P1** therefore fired: the compiled
   production solver `_nemo_literal_tke_solve` is bit-identical to NEMO's four
   Thomas recurrences and post-solve floor/mask
   (`ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:480-503`).
2. **Fed with the production card's live kt-2 matrix/RHS**
   (`trace.matrix_lower`, `trace.matrix_diag`, `trace.matrix_upper`,
   `trace.rhs_pre_sweep`, `trace.surface_dirichlet`, verified bit-identical to
   `trace.en_post_sweep` on all `799,200` words):
   - On `wet30` (`tke_wmask != 0`, 430,552 words), `solve_diag`, `solve_rhs`,
     and `solve_bottom` are bit-identical (`0 / 430,552`), and the **first
     differing wet solver intermediate is `solve_back` (Thomas reverse sweep)**
     with **114 / 430,552** unequal wet words (60 on rank 0, 54 on rank 1;
     maximum `6.589800151726445e-13`), which `after_solve`
     (`MAX(en, rn_emin) * wmask`, `rn_emin = 1.0e-10`) clamps to
     **17 / 430,552** wet words (11 on rank 0, 6 on rank 1; maximum
     `6.589800151726445e-13`).
   - On `active_solve30` (`wet30 | bottom_bc30`, 442,228 words = 430,552 wet +
     11,676 in-solve seafloor Dirichlet boundary slots at `jk = mbkt + 1` where
     `1 <= mbkt <= 29`, `ssmask != 0`, and `tke_wmask == 0`), `matrix_rhs`,
     `solve_rhs`, and `solve_bottom` already differ at **51 / 442,228** words
     (`51 / 11,676` on `bottom_bc30`: 26 on rank 0, 25 on rank 1, all on the
     north-fold row `jj = 150`, `j = 147`, at `jk = mbkt + 1`, maximum
     `3.1840720976217427e-08`), which `solve_back` couples into the 114 wet
     words above (`165 / 442,228` active-solve words total).

| stage | recorded matrix/RHS (`wet30` / `active_solve30`) | live card matrix/RHS `wet30` (430,552) | live card `bottom_bc30` (11,676) | live card `active_solve30` (442,228) |
| --- | ---: | ---: | ---: | ---: |
| `matrix_rhs` | 0 / 0 (max 0) | 0 / 430,552 (max 0) | 51 / 11,676 (26 r0, 25 r1; max 3.1840720976217427e-08) | 51 / 442,228 (max 3.1840720976217427e-08) |
| `solve_diag` | 0 / 0 (max 0) | 0 / 430,552 (max 0) | 0 / 11,676 (max 0) | 0 / 442,228 (max 0) |
| `solve_rhs` | 0 / 0 (max 0) | 0 / 430,552 (max 0) | 51 / 11,676 (26 r0, 25 r1; max 3.1840720976217427e-08) | 51 / 442,228 (max 3.1840720976217427e-08) |
| `solve_bottom` | 0 / 0 (max 0) | 0 / 430,552 (max 0) | 51 / 11,676 (26 r0, 25 r1; max 3.1840720976217427e-08) | 51 / 442,228 (max 3.1840720976217427e-08) |
| `solve_back` | 0 / 0 (max 0) | **114 / 430,552** (60 r0, 54 r1; max 6.589800151726445e-13) | 51 / 11,676 (26 r0, 25 r1; max 3.1840720976217427e-08) | 165 / 442,228 (86 r0, 79 r1; max 3.1840720976217427e-08) |
| `after_solve` | 0 / 0 (max 0) | **17 / 430,552** (11 r0, 6 r1; max 6.589800151726445e-13) | 0 / 11,676 (max 0) | 17 / 442,228 (11 r0, 6 r1; max 6.589800151726445e-13) |

### 2. Code and compiled HLO explanation (**CONFIRMED** vs **PLAUSIBLE**)

- **Compiled XLA CPU HLO of `_nemo_literal_tke_solve` — CONFIRMED exact**:
  In `jit__nemo_literal_tke_solve`
  (`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:1111-1224`),
  the three `jax.lax.scan` phases compile to sequential `while` loops
  (`%while.16` trip count 29 for `diagonal_step`, `%while.21` trip count 29 for
  `rhs_step`, and `%while.26` trip count 28 for `reverse_step`) rather than
  associative parallel scans:
  - `%fused_computation.1` (`diagonal_step`) emits `%mul.8 = multiply`,
    `%div.10 = divide`, and `%sub.6 = subtract`
    (`diagonal - (lower * previous_upper) / previous`), using IEEE-754 hardware
    division with no reciprocal-multiply rewrite and no FMA contraction.
  - `%fused_computation.3` (`rhs_step`) emits `%div.11 = divide`,
    `%mul.9 = multiply`, the `nemo_source_round` `is-finite`/`select` barrier
    (`%select_n.14`), and `%sub.7 = subtract`
    (`source - rp((lower / previous_diagonal) * previous)`), preserving
    division-before-multiply association and blocking FMA contraction.
  - `%fused_computation.10` (`terminal`) emits `%div.13 = divide`
    (`work[..., jpkm1] / diagonal[..., jpkm1]`).
  - `%fused_computation.5` (`reverse_step`) emits `%mul.10 = multiply`, the
    `nemo_source_round` barrier (`%select_n.16`), `%sub.8 = subtract`, and
    `%div.12 = divide` (`(source - rp(upper * next_value)) / current_diagonal`).
  - `%fused_computation.6` (post-solve floor/mask) emits `%max.0 = maximum` and
    `%mul.11 = multiply`.
  Any hypothesis that XLA reassociation, reciprocal multiplication, or FMA
  contraction inside `_nemo_literal_tke_solve` causes the 17-word departure is
  **REFUTED** (`0 / 799,200` unequal words on the recorded kt-2 matrix/RHS).
- **North-fold V-face mask omission in `_tke_bottom_dirichlet` and reverse-sweep
  coupling — CONFIRMED**:
  In NEMO (`ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:303-309`),
  the bottom-friction TKE boundary condition evaluates
  `zmskv = 2. - vmask(ji,jj-1,mbkt) * vmask(ji,jj,mbkt)` and `zebot` from
  `vv(ji,jj,mbkt,Kbb) + vv(ji,jj-1,mbkt,Kbb)`, writing
  `en(ji,jj,mbkt(ji,jj)+1) = MAX( zebot, rn_emin ) * ssmask(ji,jj)` at the
  seafloor interface `jk = mbkt + 1` where `tke_wmask == 0`. In legoESM,
  `_tke_bottom_dirichlet`
  (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:12466-12495`)
  calls `compute_face_masks_3d(_zc.is_active, _grid)`, which sets the
  northernmost V-face mask row `north = jnp.zeros_like(south)`
  (`packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py:5450-5451`),
  zeroing `vm3[-1]` on the north-fold row `jj = 150` (`j = 147`) instead of
  restoring NEMO's raw `vmask[-1]` as `_tke_step_entry_p_sh2` already does when
  `nemo_sh2_fold_top_v=True`
  (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:12881-12898`).
  Because `compute_tke_step` records `_statement_after_boundaries`
  (`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:3596-3597`)
  before `_solve_tke_backward_euler` scatters `bottom_dirichlet` into `rhs` at
  `is_bottom` (`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:1826-1834`),
  and because `wet30` (`tke_wmask != 0`) excludes `jk = mbkt + 1`, those 51
  north-fold departures on `rhs_pre_sweep` (`jk = mbkt + 1`) are invisible on
  `wet30` in `matrix_rhs`, `solve_rhs`, and `solve_bottom`. At the deepest wet
  interface `jk = mbkt`, however, NEMO's upper diagonal `zd_up(ji,mbkt)`
  (`ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:431-436`) is
  non-zero (`zcof * 2.e-5 / (e3t * e3w)`), whereas at `jk = mbkt + 1` the
  identity row has `zd_lw = 0`, `zd_up = 0`, and `zdiag = 1`. During the
  reverse sweep (`ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:480-503`),
  `zd_up(ji,mbkt) * en(ji,150,mbkt+1)` couples the 51 `jk = mbkt + 1` Dirichlet
  departures into 114 wet words in `solve_back`, of which 17 exceed
  `rn_emin = 1.0e-10` and survive `after_solve` and `final_avt`.

### 3. Non-landed literal fix proposal and verification on recorded inputs

- **Proposal (not landed in model code)**: in `_tke_bottom_dirichlet`
  (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:12466-12495`),
  when `getattr(vmix.tke, "nemo_sh2_fold_top_v", False)` is active and
  `_zc.nemo_een_barotropic.vmask` is present, restore the north-fold V-face
  mask row `vm3 = vm3.at[-1].set(jnp.asarray(_raw.vmask[-1], dtype=h_k.dtype))`
  (matching `_tke_step_entry_p_sh2` at
  `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:12881-12898`).
- **Verification on the admitted kt-2 inputs**
  (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_walk.py:333-547`):
  - `bottom_dirichlet` across all 11,676 in-solve seafloor columns drops from
    `51 / 11,676` (26 on rank 0, 25 on rank 1; maximum
    `3.1840720976217427e-08`) to **`0 / 11,676` (maximum `0.0`)**.
  - Substituting the fixed `bottom_dirichlet` into the live card
    `rhs_pre_sweep` and executing the compiled production solver
    `_nemo_literal_tke_solve` yields **`0 / 430,552` wet words**,
    **`0 / 11,676` bottom-BC words**, and **`0 / 442,228` active-solve words**
    (`max_abs = 0.0`) at `matrix_rhs`, `solve_diag`, `solve_rhs`,
    `solve_bottom`, `solve_back`, and `after_solve`.
  - Propagating the fixed `e_new` through `nemo_etau_injection`,
    `_length_boundaries`, and the diffusivity closure yields **`0` unequal wet
    words (`max_abs = 0.0`)** at every downstream boundary: `after_tke`
    (`0 / 414,119`), `pre_sweep_mxl` (`0 / 430,552`), `after_lup`
    (`0 / 430,552`), `after_ldown` (`0 / 430,552`), `final_mxl`
    (`0 / 430,552`), `final_mxld` (`0 / 430,552`), `zav` (`0 / 414,119`),
    `pre_pdlr_avt` (`0 / 414,119`), and `final_avt` (`0 / 414,119`).

Focused unit controls in `tests/ocean/fidelity/test_orca2_round336_tke_walk.py`
pass 4 tests. independent review: operator.


## Resume 11: passive boundary repair and admitted-recorder census

Argon item 3 is resolved for the new acquisition. When the target is
`tke336`, the shared driver neither stages nor applies the round-328
`dynzdf`/stage patches, and its actual target-build branch has the same
exclusion
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:881-888`,
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1691-1705`).
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
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/tke_round336_writer.F90:236-288`).
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
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1235-1310`).

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
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1157-1233`).
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
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/tke_round336_writer.F90:236-288`).

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
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh:131-205`).

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
`:236-288`.

The record now contains `htau`, bottom `uu/vv` at `Kbb`, evaluated `gdepw`,
`e3t`, and `e3w` at `Kmm`, all masks and top/bottom levels, and the three
background coefficient inputs used by `tke_avn`. Its executable source-read
census classifies 30 external array reads into 31 recorded fields, 34 pinned
scalar or selector reads, and seven reads belonging only to inactive branches
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:92-119`).
Removing `entry_htau` makes the read-set plant refuse. Admission additionally
requires full-domain `fr_i` to equal the admitted kt-2 ice-cover source,
`hm_i/taum` to equal their admitted sources, evaluated metrics to replay from
the admitted round-333 tracer/LDF records, and bottom drag to equal the
linear-drag namelist constant `-0.0004*ssmask`; the source-value plant
perturbs the latter and refuses
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:301-373`).

The stub no longer bypasses `admit()`. Its success and retained-success paths
run all 19 parser/semantic plants followed by the real two-rank admission;
success, in-place re-admission, quoted NEMO refusal, crash retention, and
retained re-admission all pass end to end
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1503-1535`).
The refreshed read-only preflight passes 19 focused tests and the 45-test
inherited battery, real-source bounds execution on both ranks, five syntax
proofs, all 15 placement plants, the read-set plant, the source-value plant,
and same-program identity for 423 objects and 411 preprocessed sources. The
production FP counts remain `zdf_tke=1007`, `tke_avn=783`, with `tke_tke`
inlined in both builds.

The opcode proof is deliberately described as a multiset: it cannot detect a
MIN/MAX operand swap. The admission backstop is the independent byte compare
over every inherited `oracle_*.bin` and every step-10 restart file
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:576-590`).
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

## Resume 15: retained TKE admission and comparison repair

The kept refusal is case **(a), a comparison defect**, not a production-value
departure.  The executing step driver deliberately calls `zdf_phy` with
`(Kbb,Kmm)=(Nbb,Nbb)` at
`ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/stprk3.f90:153-155`.
The TKE observer then records the exact `r3t(:,:,Kmm)` operand passed to the
live TKE arithmetic at
`ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:199-204`.
At kt 2 that live header is `(Kbb,Kmm)=(3,3)`, whereas the admitted
round-333 stage-3 header is `(Kbb,Kmm,Krhs,Kaa)=(3,2,1,1)`.  Therefore the
live TKE `Kmm` slot is round 333's `Kbb` slot; the refused check had replayed
round 333's different `Kmm` slot.

Across the two 90-by-148-by-31 rank-local arrays (825,840 words), replay with
round-333 `r3t_Kbb` gives **0 unequal, max absolute 0** for both live
`e3t(Kmm)` and live `e3w(Kmm)`.  The refused `r3t_Kmm` replay gives 430,552
unequal `e3t` words (rank 0: 233,341, max 0.2327149741621497 m; rank 1:
197,211, max 0.23877390803374965 m) and 509,423 unequal `e3w` words (rank 0:
272,614, max 4.735874793313712 m; rank 1: 236,809, max
11.872214721130604 m).  `r3t_Kaa` has the same unequal counts and exactly
twice those maxima.  Thus staggering, extent and formula are exonerated:
the correct T-grid expression and the independently checked W-grid expression
both match bit-for-bit over identical extents.  The gate now asserts the
header-slot relation and replays `r3t_Kbb` without changing its exact bar
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:344-361`).

The repaired replay exposed one further comparison-only extent defect.  The
round-326 ice recorder zero-filled its halo and copied only the range TKE
executes at `zdftke_round326_writer.F90:34-69`, while round 336 records the
live full-grid `fr_i`.  Full-grid comparison found 101/14,288 unequal words
on rank 0 and 99/14,288 on rank 1 (maxima 0.9971085262668478 and
0.9970634354455692); all 200 are in the unrecorded halo.  The common executed
range is 0/13,320 unequal per rank, max 0.  The check now compares only that
range, again at the exact bit bar
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:362-373`).

The focused suite passes 28 tests.  The committed, clean-tree retained route
then passed 54 inherited/focused tests, 140-file deterministic passivity, the
two-rank 51-field patched-source fixture, all semantic plants, the 425-object
same-program census (815 FP-bearing functions, 205,193 FP instructions), the
five observer-free twin outputs, and both rank-local TKE records.  Admission
reports `PASS_R336_TKE_KT2_ADMISSION`, with 233,341/197,211 wet interfaces
and 224,547/189,572 wet process interfaces.  The admission JSON SHA-256 is
`c86e812a18dcc30d7fc531b747b107cfb99ba51a5c595cba9ca20597157031c5`; the
complete retained-admission log SHA-256 is
`690fe5a9a62f63e355e76c6ff5d1e42fdea8c7955cb134c9dd36bf661d938a00`.
No NEMO executable was run in this resume, and no model source,
configuration, score or fidelity bar changed.

The requested admission line is:

```text
R328_RETAINED_OUTPUTS_ADMITTED /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/acquisition/orca1ice_tke_internal_kt2_np2
```

## Resume 12: producer-value discrimination

Resume 11's unequal FP counts are provenance flags, not value-difference
measurements. This section supersedes only that section's value implication;
the counts and producer identities remain valid. The offline audit recompiles
the observer-free functions with the record flags, compares register-free
ordered FP mnemonics with each admitted producer binary, scans every affected
function for FMA instructions, and byte-compares the already-written twin
witnesses. It does not run NEMO and does not modify either acquisition tool.

All available producer build Makefiles carry the identical flags
`-fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer
-ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`.
None carries `-ffast-math`, `-Ofast`, `-mfma`, `-ffp-contract=fast`, or the
other unsafe-math flags checked by the audit. The affected-function FMA census
is zero in every producer.

The observer-free twin is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2`.
For it and every producer below, the ordered four-restart manifest has SHA-256
`9e09b85d488f60846b94a46687dfe9d757b1e09433cbcab0d60293c7d7161f7a`;
the 30 complete stage files have manifest SHA-256
`9bebfc965ea298c5521e278ac07b41417edf2ba5f9cac8b0f704d59065f7ef15`;
and the post-`zdf_phy` return file has SHA-256
`209369523aafa70ddbaa17695908b6da30929dae6bce7ee225c9ee353c993ad8`.
Here a manifest digest is SHA-256 over filename-ordered `SHA-256  basename`
lines. Thus every row has 4/4 restart files, 30/30 completed-stage files, and
the post-ZDF file byte-identical to the same-input twin.

| admitted record (path below `orca2_rounds`) | census differs? | restart bytes equal to twin? | per-function value test | VERDICT |
|---|---|---|---|---|
| round 328 dyn walk — `round328/acquisition/orca1ice_dynzdf_kt1_np2` | yes: `dyn_zdf` 2,446/2,365; `stp_rk3_stg` 1,175/1,367 | yes, 4/4; common restart digest above | ordered sequences differ for both; all 30 returned stages are byte-identical; no internal-boundary twin | **values production-identical PLAUSIBLE** |
| round 328 ZDF terms — `round328/acquisition/orca1ice_zdfphy_kt1_np2` | direct ZDF functions no; inherited `dyn_zdf`/stage yes | yes, 4/4; common restart digest above | direct censuses remain equal; inherited ordered sequences differ; post-ZDF and stage outputs are byte-identical | **values production-identical PLAUSIBLE** |
| round 331 TKE — `round331/acquisition/orca1ice_tke331_kt1_np2` | yes: `zdf_tke` 1,007/783; `tke_tke` inlined/1,054 | yes, 4/4; common restart digest above | inlining prevents an ordered one-function comparison; the admitted-input post-ZDF return and all stages are byte-identical | **values production-identical PLAUSIBLE** |
| round 332 source order — `round332/acquisition/orca1ice_source_order_kt1_np2` | yes: `zdf_tke` 1,007/0; `tke_tke` inlined/1,054; stage 1,175/1,367 | yes, 4/4; common restart digest above | inlining prevents a TKE sequence match and stage sequence differs; post-ZDF return and all stages are byte-identical | **values production-identical PLAUSIBLE** |
| round 332 raw V — `round332/acquisition/orca1ice_raw_postfold_v_kt1_np2` | yes: stage 1,175/1,367 | yes, 4/4; common restart digest above | stage ordered sequence differs; every completed-stage value is byte-identical | **values production-identical PLAUSIBLE** |
| round 333 tracer stage — `round333/acquisition/orca1ice_tracer_stage3_kt2_np2` | yes: stage 1,175/1,367 | yes, 4/4; common restart digest above | stage ordered sequence differs; every completed-stage value is byte-identical | **values production-identical PLAUSIBLE** |
| round 333 internal LDF — `round333/acquisition/orca1ice_traldf_iso_internal_kt2_np2` | direct LDF counts no; inherited stage yes | yes, 4/4; common restart digest above | stage sequence differs; the 526/526 and 538/538 LDF sequences reorder; all returned stages are byte-identical | **values production-identical PLAUSIBLE** |
| round 336 `tra_zdf` — `round336/acquisition/orca1ice_trazdf_internal_kt2_np2` | direct `tra_zdf` 60/60 and `tra_zdf_imp` 1,234/1,234 no; inherited `zdf_phy`/stage yes | yes, 4/4; common restart digest above | `tra_zdf` and `tra_zdf_imp` ordered sequences are exactly equal (sequence digests `3c2bac50...` and `da401374...`); inherited `zdf_phy` reorders and stage differs; all return witnesses are byte-identical | **values production-identical PLAUSIBLE** |

`PLAUSIBLE` is deliberate. The runtime twins confirm every persistent restart,
every completed stage, and the available direct post-ZDF return. They do not
contain an observer-free copy of each internal boundary later used for
statement localization, and the differing functions' ordered opcode sequences
do not close that gap. No row shows `values differ`. Within the round-336 row,
the two directly instrumented tracer-solve functions are individually
production-identical **CONFIRMED** by exact ordered opcode sequences; the whole
record remains `PLAUSIBLE` because its inherited `zdf_phy` and stage inputs do
not have function-local internal twins at kt 2.

### Superseding earlier-receipt impact list

- Round 328's dyn-walk localization, its exact recorded recurrence statements,
  and the later ZDF-term conclusions are no longer classified as results of
  demonstrated different values. They remain **PLAUSIBLE** pending an
  observer-free internal-boundary twin or a function harness on the admitted
  inputs.
- Round 331's first-TKE-mismatch family, recurrence/returned-energy claims,
  seven residual coefficient cells, and stored-bottom ownership remain
  **PLAUSIBLE**, not refuted. Round 332's mixing walk, surface anchor,
  viscosity-order exclusion, Prandtl conclusion, and raw-V/fold localization
  likewise remain **PLAUSIBLE**. Their independent Decision-122, DINO, and
  gradient results are unchanged.
- Round 333's stage counts, `tra_ldf` call-boundary attribution, internal-LDF
  divisor walk, and later post-LDF/`tra_zdf` conclusions remain **PLAUSIBLE**.
  The reverse-safe model landing still has its independent evidence and is not
  changed by this audit.
- Round 336's admitted `tra_zdf` and `tra_zdf_imp` arithmetic is individually
  **CONFIRMED** production-identical. Conclusions that also depend on inherited
  kt-2 `zdf_phy` or internal stage values remain **PLAUSIBLE**. The new
  production-arithmetic TKE acquisition remains needed because this audit does
  not supply its missing internal kt-2 boundaries.

The committed audit is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_producer_value_audit.py`.
Its report is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/resume12_producer_value_audit.json`,
SHA-256 `5f6679afb2934a41e35c9a5d31352486b06bf7ff7fa12df0cef23f402dc1ae42`.
It reports `PASS_R336_PRODUCER_VALUE_AUDIT`, eight records, zero FMA
instructions in affected functions, zero `values differ`, and eight
`PLAUSIBLE` record verdicts. Its restart- and opcode-mutation plants both
refuse, its self-test passes, and the four focused tests pass. Codex review:
operator.

At evidence commit `7b4d73266bd33668ab05bbcd322cd9585594c755`, the
receipt citation gate passes all 43 citations with zero failures, zero
unmapped claims, and zero map-audit failures; report SHA-256
`ff9198f1b25ff23505f988506b3251718a7596bcb34f9f2ccbed1b175921733f`.
The prescribed offline default battery, including the four new producer-value
tests and the private-import ratchet, passes 156 tests in 1,141.51 seconds;
its complete log SHA-256 is
`fbbb1198e1af7684593eb56bb032c1368db4169a1006682f058fac5da8d23914`.

## Resume 13: exact drag fill and pristine executable census

Sonnet's two HOLD findings are resolved without running NEMO and without
changing a model statement, configuration value, or score. In NEMO,
`rCdU_top` is updated inside `DO_2D(1,1,1,1)` at
`ZDF/zdfdrg.F90:117-127`; both nonlinear bottom-drag arms use that same loop
at `ZDF/zdfdrg.F90:171-190`, and the linear bottom-drag assignment does so at
`ZDF/zdfdrg.F90:213-221`. The macro expands to
`ntsi-1:ntei+1,ntsj-1:ntej+1`, exactly one halo ring, at
`do_loop_substitute.h90:117-125`. Thus the real two-rank layout fills
`2:93,2:151` (92 by 150), not the outermost `jpi`/`jpj` ring.

The writer now allocates `T2D(1)` and copies precisely that filled slice
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/tke_round336_writer.F90:116-135`).
The parser requires both the 92-by-150 extent and origin `(ntsi-1,ntsj-1)`,
with a `drag-range` plant that perturbs the origin and refuses
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:203-229`).
The bottom-drag source-value check is restricted to the identical one-halo
mask slice, so no unwritten outer-ring value is demanded
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:319-324`).

The stub builder now reads admitted real tracer/LDF, ice, and round-328 drag
records. Its drag arrays copy the real NEMO overlap directly; the newly
exposed left/bottom halo is derived from the admitted real mask and the
source-literal linear coefficient for bottom drag, while top drag extends the
nearest real halo value. It then emits the same exact metadata path consumed
by admission
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/make_stub_records.py:99-135`).
Consequently the stub no longer fabricates a full-domain drag record that can
conceal this extent class, and the real parser/admission plus `drag-range`
plant exercises the repaired contract.

The acquisition build no longer copies any inherited round-326 `MY_SRC` unit.
It installs pristine NEMO `zdftke.F90`, the round-336 writer, and only the
round-336 patch
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1689-1737`).
Preflight reconstructs a pristine baseline by replacing all 15 inherited NEMO
units and removing the inherited writer, producing 425 NEMO objects
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1019-1060`).
It compares the FP opcode multiset for every function in every one of those
objects, excluding only the new write-only private writer. The passing census
is 815 FP-bearing functions and 205,193 FP instructions; `stp_rk3_stg` is the
pristine 1,175-FP body
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1460-1472`).
The negative plant recompiles the actual inherited stage source, pins its
1,367-FP census, substitutes it for the clean stage, and makes the whole-build
census refuse
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1473-1492`).

The clean committed preflight exits zero. It passes 23 focused tests and the
49-test inherited battery, the two-rank 51-field bounds/source fixture, all 20
placement plants, the exact drag-range and source-value plants, pristine
same-program identity, the 425-object whole-executable census, and the
inherited-stage census plant. The stub runs the real `admit()` path for
success, in-place admission, quoted NEMO refusal, crash retention, and retained
admission; all five cases pass. It ends with
`ORCA2_ROUND336_TKE_KT2_PREFLIGHT_READY`. Argon's RUN remains satisfied and
Sonnet's HOLDs are resolved. Codex review: operator.

At evidence commit `967a912a690d2f7a0a0ada106ea6b1a7d63f6682`, the
per-receipt citation gate passes all 55 citations with zero failures, zero
unmapped citations, zero map-audit failures, and all nine self-tests fired;
the report SHA-256 is
`4a6b7eb8bfdccad6b37da923ac1b36c881a984415c8bb4230aaf7c3d6eea1776`.
The exact default ORCA2 battery passes all 156 tests in 1,049.58 seconds; its
log SHA-256 is
`3e432bfc275b94f933500d148c56f61efe47d314b6e01392b40de1562203442c`.
The explicit private-import ratchet passes all 6 tests in 4.85 seconds; its log
SHA-256 is
`357552e211123f746071b1da46e34b619513dc21077812ffe469614a71dd6b28`.

The required operator handoff remains:

```text
ACQUISITION_NEEDED: scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh
```

## Resume 14: build-derived output admission

The TKE admission now derives its admissible observer files from the
same-program observer/writer census and the compiled writer's sole `OPEN`
format.  It requires exactly the four terminal restarts, `ocean.output`, and
the two rank-local TKE records for the observed run, while the twin must have
only the five standard outputs; every restart and `ocean.output` is then
byte-compared between the two runs
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:440-483`;
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:593-608`).
Thus round 336 no longer requires this pristine-plus-TKE build to regenerate
the 156 inherited `oracle_*.bin` files from round 326.

For a real acquisition, the driver first saves a pristine executable before
installing the `zdftke` observer and writer
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1696-1699`).
It builds the output contract from the compiled observed build, runs the
observer-free twin first and the observed executable second from identical
deck/input manifests, checks both selector echoes, and only then admits
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1771-1814`).

The replay inputs are now the admitted round-333 tracer and LDF records plus
the admitted round-326 ice records.  Both ranks of all three inputs are pinned
to six literal SHA-256 digests
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:197-204`).
Admission verifies those digests, reconstructs `e3t(Kmm)` and `e3w(Kmm)` from
the round-333 records, checks the round-326 ice fields, and checks final `avt`
against the recorded NEMO `pdlr` statement rather than consuming a
round-336 `tra_zdf` record
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:337-430`).

The stub copies the four explicit restarts, derives its observer records from
the build-generated contract, and never copies an inherited round's
`oracle_*.bin` set
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh:84-97`).
Its new negative scenario writes `oracle_impossible.bin`; preflight requires
the exact unexpected-output refusal before reporting the plant fired
(`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh:166-177`).

The clean prescribed preflight exits zero, the impossible-output plant fires,
all six stub-flow scenarios pass, and the final marker is
`ORCA2_ROUND336_TKE_KT2_PREFLIGHT_READY`.  Its complete log is
`round336/resume14_tke_preflight.log`, SHA-256
`8de71e5e4cac0a6d4a4d3157f6d31eb7d6e5b9d8e86fd9a48dda78a0b38edb3f`.
No model source, configuration value, or score changed.  Codex review:
operator.

At evidence commit `cc6e6b36f613e0cecc345e3c75886219bcef6994`, the
per-receipt citation gate passes all 63 citations with zero failures, zero
unmapped citations, zero map-audit failures, and all nine self-tests fired;
its log SHA-256 is
`95276d63d9c763de2e8dbcd34ea1d41cf80f309a51e729d99b5db753c6ee4cb6`.
The exact default ORCA2 battery passes all 156 tests in 1,076.03 seconds; its
log SHA-256 is
`d178cd62618f9277ba2a45d789fc8b76e5813ef29eaaa072bf4e2417668122bd`.
The explicit private-import ratchet passes all 6 tests in 5.44 seconds; its
log SHA-256 is
`1835a92948bcc36dbddb1ce8d40daf33d8d9bf09a84abe0117f775cf3f3ae3b8`.

The required operator handoff remains:

```text
ACQUISITION_NEEDED: scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh
```

## Resume 18: landing north-fold V-face mask in `_tke_bottom_dirichlet`

### Preregistration (frozen before model edit and measurement)

Both independent reviews (Codex and Gemini Flash) re-ran the Resume 17 probe
and confirmed the single-statement owner of the kt-2 `zdftke`/`tra_zdf`
departure: in NEMO, `dommsk` applies `lbc_lnk` to `vmask`
(`src/OCE/DOM/dommsk.F90:146-172`) and `tke_tke` uses that `vmask` directly in
`zmskv = 2. - vmask(ji,jj-1,mbkt) * vmask(ji,jj,mbkt)` when forming `zebot` at
`jk = mbkt + 1`
(`ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:303-309`), whereas
legoESM's `_tke_bottom_dirichlet` leaves the northernmost V-face mask row at
`compute_face_masks_3d`'s zero wall instead of restoring the card's raw NEMO
`vmask[-1]` under `nemo_sh2_fold_top_v=True` (already set on the ORCA2 NEMO
recipe at `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2063`).

- **R336-R18-P1 (production-call-site test and `eager == jit`)**: Restoring
  `vm3[-1]` from `_zc.nemo_een_barotropic.vmask[-1]` under the existing
  `nemo_sh2_fold_top_v` branch in `_tke_bottom_dirichlet` (no new knob) makes
  the production vertical-mixing call (`LatLonCGridOceanModel._apply_implicit_vertical_mixing`
  and `_tke_bottom_dirichlet`) match NEMO's `zebot` formula on the north-fold
  row, satisfy `eager == jit` (with `jax.disable_jit()` around the eager arm
  and inputs passed as JIT arguments), and fail when the one-line fix is
  reverted.
- **R336-R18-P2 (Decision 122 on all seven pairs, both labels)**: Across
  `rung0/independent`, `omt1/independent`, `omt2/independent`,
  `omt3/independent`, `omt4/independent`, `rung10/given_nemo_entry`, and
  `rung10/independent`, zero exact rows are lost (`exact_row_losses == []`) and
  every row is bit-identical or strictly better vs parent `254fec16a822`.
  *Falsifier*: any exact-row loss or any row moving away (`away > 0`) stops the
  landing immediately.
- **R336-R18-P3 (other callers and DINO month gate)**: Every other caller of
  `_tke_bottom_dirichlet` (where `nemo_sh2_fold_top_v=False` or
  `bottom_tke_bc=False`, including faithful DINO, `V1`, `GYRE`, `VORTEX`,
  `ORCA2G1..G8`, and non-NEMO configurations) is bit-identical parent vs
  current (normalized JAXPR and 3-step faithful DINO leaf-step SHA-256 hashes),
  and the 30-day GPU DINO from-rest month gate passes within bar
  (`<= 2.244317642e-03 K`).
- **R336-R18-P4 (rung-10 N=1 gradients)**: Rung-10 N=1 forward parity has 0
  non-bit leaves, reverse cotangents (`T@1`, `ssh@1`) are finite across all
  799,200 cells with 0 nonzero dry cells, and reverse equals forward (JVP) for
  `T_dir0` and `rn_ud` within `1e-10`.
- **R336-R18-P5 (rung-10 kt 1--10 rescore)**: Round 333's 18-word `kt=2 stage3 T`
  departure (and 13-word `kt=2 stage3 S` departure) becomes `0` unequal wet
  words (`BIT_EXACT`) under both `given_nemo_entry` and `independent`, and the
  first non-bit / first wet-value / first over-bar row across `kt=1..10` is
  reported.

### 1. Landed production change and revert-sensitive call-site test

- **Model fix**: In `_tke_bottom_dirichlet`
  (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:12466-12495`),
  under the existing `bool(getattr(vmix.tke, "nemo_sh2_fold_top_v", False))`
  selector (no new configuration knob; already set on the ORCA2 NEMO recipe at
  `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2063`),
  `_tke_bottom_dirichlet` validates that `_grid.fold` is active and
  `_zc.nemo_een_barotropic.vmask` is present and restores the north-fold V-face
  mask row `vm3 = vm3.at[-1].set(jnp.asarray(_raw.vmask[-1], dtype=h_k.dtype))`
  before forming `v_sum = vm3[:-1] * v_b[:-1] + vm3[1:] * v_b[1:]`. This matches
  NEMO's `lbc_lnk`-processed `vmask` (`src/OCE/DOM/dommsk.F90:146-172`) in
  `tke_tke` (`ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:303-309`)
  and matches `_tke_step_entry_p_sh2`
  (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:12881-12898`).
- **Production-call-site test and `eager == jit` (`M2`, `M4`)**:
  `tests/ocean/unit/test_nemo_sh2_fold_top_v.py:118-310` exercises
  `model._tke_bottom_dirichlet` and
  `LatLonCGridOceanModel._apply_implicit_vertical_mixing` on a partial-cell
  north-fold staircase (`mbkt = 2 < nlev - 1 = 3`, so the in-solve seafloor
  Dirichlet slot `k = 2` couples into wet interface `k = 1` during the Thomas
  reverse sweep) with depth-uniform velocity (`dv == 0`, isolating
  `_tke_bottom_dirichlet` from `_tke_step_entry_p_sh2`). Eager execution
  (`with jax.disable_jit():`) and `jax.jit(run_vmix)(*args)` with
  `(eta, u, v, avm, tke0)` passed as JIT arguments agree bit-for-bit on
  `_tke_bottom_dirichlet` and within `rtol=1e-13` on `K_v`/`A_v`. When the
  fix in `_tke_bottom_dirichlet` is reverted,
  `test_production_bottom_dirichlet_restores_north_fold_vmask_and_matches_nemo`
  fails (`EXIT_CODE=1`, 6/6 elements mismatched on `bot_on[-1]` vs
  `nemo_ref_top`, `max_abs = 2.04939015e-04`, relative error `0.5`; log
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r336_resume18/revert_test_failure.log`),
  and passes (`5 passed`) when enabled.
- **Admitted kt-2 21-boundary TKE walk on the landed model**:
  `tests/ocean/fidelity/test_orca2_round336_tke_walk.py:126-207` and
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r336_resume18/round336_tke_walk_landed.json`
  (SHA-256 `8102ffa2e46fceb0b3e5b12f263929f08658501f38748adb2a0ddc60d8b229ed`)
  report `PASS_R336_TKE_WALK` with `first_non_bit_boundary = None`,
  `first_non_bit_active_solve30_stage = None`,
  `first_non_bit_wet30_solver_stage = None`, and `0` unequal wet words across
  all 21 recorded `zdftke`/`zdf_phy` boundaries at `kt = 2`.

### 2. Decision 122 census across all seven pairs and both labels

Measured with `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_d122_gate.py:217-457`
(unit-tested in `tests/ocean/fidelity/test_orca2_round336_d122_gate.py:20-50`;
self-test artifact `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r336_resume18/d122_self_test.json`,
SHA-256 `a93149a47b89842a1dbc99157ccd56f18705d19795eeb36860935777793bc4fa`;
both `exact-loss` and `away-row` plants fire `STATUS PLANT-FIRED`). Both the
same-tree control comparison (`decision.json`, SHA-256
`807888ca3e096627bcfce9c1eac4085a6de8560e740aed8852bd529dc045b687`) and the
cross-tree parent comparison against `254fec16a822` (`cross_tree_decision.json`,
SHA-256 `893dc811ab135fc8e27bc8d548ce3d7f9d3b7f9a8cc8473f0d6a272a19df7b68`)
return `PASS_R336_D122_DECISION` with identical numbers:

| Pair (`rung / label`) | Rows | `exact_row_losses` | `unequal_increases` | RMS / `max_abs` (`toward / equal / away`) | Parent classes (`BIT / AT_BAR / OVER_BAR`) | Current classes (`BIT / AT_BAR / OVER_BAR`) | Old first `OVER_BAR` -> New at old target | New first `OVER_BAR` | Candidate artifact SHA-256 |
|---|---:|---:|---:|---|---|---|---|---|---|
| `rung0 / independent` | 200 | 0 | 0 | `0 / 200 / 0` | `35 / 15 / 150` | `35 / 15 / 150` | `(1, stage3, T)` `430,551` -> `430,551` (`equal`) | `(1, stage3, T)` | `8e6bc77cd63658eeb8ec14632a328dfe472c15b316cc30ab7c71cd2f8e389b0c` |
| `omt1 / independent` | 160 | 0 | 0 | `0 / 160 / 0` | `62 / 0 / 98` | `62 / 0 / 98` | `(4, stage3, T)` `7` -> `7` (`equal`) | `(4, stage3, T)` | `60c1e963d8ae8fee27cadc2e765fee0a837d577bf723bf83e917f9e85415b28d` |
| `omt2 / independent` | 200 | 0 | 0 | `0 / 200 / 0` | `62 / 0 / 138` | `62 / 0 / 138` | `(4, stage3, T)` `7` -> `7` (`equal`) | `(4, stage3, T)` | `5c5b571952541a12e265b328f257518dadcacbaee85c9ca278baa8da8db6bb26` |
| `omt3 / independent` | 200 | 0 | 0 | `0 / 200 / 0` | `36 / 12 / 152` | `36 / 12 / 152` | `(3, stage3, T)` `5` -> `5` (`equal`) | `(3, stage3, T)` | `2df03a4460a0f870ebc5c93264dfdbcbd5a2ac288daa033d72890f83ef269895` |
| `omt4 / independent` | 200 | 0 | 0 | `0 / 200 / 0` | `16 / 4 / 180` | `16 / 4 / 180` | `(2, stage1, u)` `433,735` -> `433,735` (`equal`) | `(2, stage1, u)` | `9dbd287f7686a73bfa32eb330f42f5db9b6fb05adca39848a6fee47126f7af43` |
| `rung10 / given_nemo_entry` | 200 | 0 | 0 | `159 / 41 / 0` | `31 / 11 / 158` | `151 / 49 / 0` | `(2, stage3, T)` `18` (`max_abs=7.678e-12`) -> `0` (`BIT_EXACT`, `toward`) | `None` (`0` `OVER_BAR`) | `25f1bc86c762c9500b9881c72a16c274d7f875bd759d3a439c5bd95583cfc7fe` |
| `rung10 / independent` | 200 | 0 | 0 | `159 / 41 / 0` | `31 / 11 / 158` | `151 / 49 / 0` | `(2, stage3, T)` `18` (`max_abs=7.678e-12`) -> `0` (`BIT_EXACT`, `toward`) | `None` (`0` `OVER_BAR`) | `e8ddde4e7e16e73fd84326dd468c22db4653d6144eb472ad966348e60f036dcf` |

The same-tree `rung10` control artifacts (`rung10_given_nemo_entry_control.json`
SHA-256 `2939abb87017acc312d2b7a5ffaa1b0572f7ef10cb3a25dc5df40536d9b0de0c`;
`rung10_independent_control.json` SHA-256
`744b1db8a099c114dbe250097c753071533609009f10a03e186764b1d3c0dc46`) are
bit-identical to the Round-333 parent candidate rows on all 200 rows. On both
`rung10` labels, `120` rows move from `OVER_BAR` directly to `BIT_EXACT` (`31 -> 151`
`BIT_EXACT`), `39` rows move from `OVER_BAR` to `AT_BAR_NOT_EXACT` with strictly
smaller RMS, `max_abs`, and `unequal` counts (`158 -> 0` `OVER_BAR`), and the
remaining `41` rows (`31 BIT_EXACT` + `10 AT_BAR_NOT_EXACT`) are bit-identical
to the parent.

### 3. Caller audit and DINO fidelity gates (`M6`)

- **All callers of `_tke_bottom_dirichlet`**:
  1. `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:13927-13929`
     (production caller inside `_apply_implicit_vertical_mixing`).
  2. `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2t_tke_entry_gate.py:93`
     (ORCA2 `kt=1` rest-state entry check where `u = v = 0`; `0` unequal words
     vs parent, SHA-256 `23e5a7ad14db9c9df3463d102a5d064780b4939bad3b5b8643012909e1d1585b`).
- **Non-NEMO and non-ORCA2 card audit**
  (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_d122_gate.py:461-608`;
  artifact `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r336_resume18/caller_audit.json`,
  SHA-256 `66310b27c31186a6c704d7e545f068be8bf59061911a2153652856420850c438`,
  `STATUS PASS_R336_CALLER_AUDIT`):
  - Non-NEMO prognostic TKE with `bottom_tke_bc=True, nemo_sh2_fold_top_v=False`
    (both folded and unfolded grids): `0` unequal words vs parent on both
    `_tke_bottom_dirichlet` and `_apply_implicit_vertical_mixing` (`K_v`, `A_v`).
  - Non-ORCA2 NEMO testcase cards (`GYRE-zco`, `LOCK_EXCHANGE-zco`,
    `OVERFLOW-zps`, `VORTEX-zco`, `ORCA2-rung0`): `nemo_sh2_fold_top_v=False`,
    `bottom_tke_bc=False`, `_tke_bottom_dirichlet` returns `None`, bit-identical
    to parent.
- **Faithful DINO 3-step value and JAXPR probe**:
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round330_dino_probe.py`
  on the `faithful` DINO grid produces `faithful_values.json` with SHA-256
  `5f977ac145525b73d311bdaacd5c7620b4199567f62afac15d3d1c8d19b503fa` —
  byte-for-byte identical to the Round-333 baseline across all 66 leaf-step
  hashes (`0` differences; `faithful_program.json` SHA-256
  `5963788cd3b46a3d51440fc22f2ed58ac627c119d33e3d28d1094639bd1741df`).
- **30-day GPU DINO from-rest month gate**:
  `scripts/validate/ocean_fidelity/dino_1226/dino_fromrest_month_gate.py` passes
  its `--self-test` (`SELF-TEST OK`) and completes the 30-day (`kt = 960`) fp64
  integration on GPU 0 in `256.9 s` (`267.6 ms/step`). Day-30 wet 3-D `T` RMS vs
  NEMO `kt = 960` is `2.056821683e-03 K` against bar `2.244317642e-03 K`
  (`PASS`; snapshot SHA-256
  `40a63d3a43a80600c740067fd3446bf428cd20e362a60987908c1a079a6e6bb5`; score
  JSON SHA-256
  `04415f8cec382eb8df532a034647117ee9468187d9cef44663c16866e2db1629`).

### 4. Rung-10 `N = 1` gradient gate

Measured with `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round329_grad_probe.py`
(`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r336_resume18/grad/grad_rung10_n1.json`,
SHA-256 `eb0a11a12baf784b80e4c6413a527aeb2b25eaf83efe156808393b0c581b69bc`,
peak RSS `77.71 GiB`, `2,416.6 s`):

- **Forward and construction parity**: `forward_parity_non_bit_leaves = []`;
  rebuilt model vs traced scalar at `eps = 1e-4` has
  `rebuilt_vs_traced_non_bit_leaves = []`.
- **Reverse cotangent finiteness (`N = 1`)**: Both `T@1` (`loss = 32.91696958352461`,
  `dT0_l2 = 0.01931092431777194`, `dT0_nonzero_wet_frac = 1.0`) and `ssh@1`
  (`loss = 0.09080875357052787`, `dT0_l2 = 0.0006856828144876344`,
  `dT0_nonzero_wet_frac = 0.9999628384027945`) have `dT0_finite = True`,
  `dT0_nonfinite_count = 0`, `dT0_nonzero_dry = 0`, and
  `nonfinite_cotangent_leaves = {}`.
- **Reverse = forward (JVP) agreement**:
  - Directional `T_dir0` on `T` loss: `reverse = -0.04157151209367675`,
    `forward = -0.04157151209367753`, relative difference
    `1.869443948746345e-14`.
  - Directional `T_dir0` on `ssh` loss: `reverse = 0.0004299604909920475`,
    `forward = 0.0004299604909920481`, relative difference
    `1.3868976507379975e-15`.
  - Scalar `rn_ud` on `T` loss: `reverse = -4.51170448705634e-05`,
    `forward = -4.511704487056347e-05`, relative difference
    `1.501929835492298e-15` (central FD at `eps = 1e-4` is
    `-4.5116905766917625e-05`, relative error `3.0831728047889743e-06`).
  - Scalar `rn_ud` on `ssh` loss: `reverse = 0.0`, `forward = 0.0`, relative
    difference `0.0`.

### 5. Rung-10 `kt = 1..10` rescore (both labels, every stage)

From the wet-value audit embedded in
`rung10_given_nemo_entry_candidate.json` (SHA-256
`25f1bc86c762c9500b9881c72a16c274d7f875bd759d3a439c5bd95583cfc7fe`) and
`rung10_independent_candidate.json` (SHA-256
`e8ddde4e7e16e73fd84326dd468c22db4653d6144eb472ad966348e60f036dcf`):

- **Round 333's 18-word `kt = 2, stage3, T` departure (and 13-word `S` departure) is GONE**:
  At `kt = 2, stage3`, under both `given_nemo_entry` and `independent`, `T` has
  `unequal = 0`, `unequal_wet = 0`, `unequal_dry = 0`, `signed_zero_only_wet = 0`,
  `max_abs = 0.0`, `rms = 0.0` (`BIT_EXACT`, down from `18` unequal wet words,
  `max_abs = 7.678080393702658e-12 K`, `rms = 1.2687664596813964e-14 K`), and
  `S` has `unequal = 0`, `unequal_wet = 0`, `max_abs = 0.0`, `rms = 0.0`
  (`BIT_EXACT`, down from `13` unequal wet words,
  `max_abs = 2.3803181647963357e-12 psu`).
- **Summary across all 200 rows (`kt = 1..10`, `entry/stage1/stage2/stage3`, `ssh/T/S/u/v`) for both `given_nemo_entry` and `independent`**:
  - `BIT_EXACT`: **151 / 200** rows (up from 31 in parent `254fec16a822`). Every
    `stage3` row for `ssh`, `T`, `S`, and `u` across `kt = 1..10`, every
    `stage1` row for `ssh`, `T`, `S`, and `u` across `kt = 1..10`, every
    `stage2` row for `ssh`, `S`, and `u` across `kt = 1..10`, and `stage2 T` at
    `kt = 1..9` are `BIT_EXACT` (`0` unequal words).
  - `AT_BAR_NOT_EXACT`: **49 / 200** rows (`48` `u`/`v` rows with
    `unequal_wet = 0`, `max_abs = 0.0`, `rms = 0.0` consisting solely of
    signed-zero `+0.0` vs `-0.0` and dry-mask bit differences, plus `1` single
    1-ULP wet cell at `kt = 10, stage2, T`).
  - `OVER_BAR`: **0 / 200** rows (down from 158 in parent `254fec16a822`).
- **New first non-bit row (bit-level)**:
  - `kt = 1, stage1, v` (both `given_nemo_entry` and `independent`):
    `unequal = 7,534` (`signed_zero_only_wet = 5,550`, `unequal_dry = 1,984`,
    `unequal_wet = 0`), `max_abs = 0.0`, `rms = 0.0`, `AT_BAR_NOT_EXACT`.
- **New first wet-value row (`unequal_wet > 0`)**:
  - `kt = 10, stage2, T` (both `given_nemo_entry` and `independent`):
    `unequal = 1`, `unequal_wet = 1` out of `263,820` wet cells, at
    `(j=18, i=34, k=3, rank=0)` (`coast = True`, `under_ice = True`,
    `fold = False`, `seam = False`, `ice_fraction = 0.8657079645688692`),
    `max_abs = 2.220446049250313e-16 K` (1 ULP),
    `rms = 3.5125902029716364e-19 K`, `wet_rms = 4.3230112674073626e-19 K`,
    `wet_normalized_max_abs = 7.349646833843343e-18 <= 1.0e-15`
    (`AT_BAR_NOT_EXACT`). At the following checkpoint `kt = 10, stage3, T`, the
    field returns to `BIT_EXACT` (`0` unequal words).
- **New first `OVER_BAR` row (`kt = 1..10`)**: **`None`** (`0` `OVER_BAR` rows
  across `kt = 1..10` under both `given_nemo_entry` and `independent`).

### 6. Final gates and disposition

- **Per-receipt and suite citation gates**: Re-anchored all 24
  `ocean_model_latlon_cgrid.py` citations shifted by the 9-line addition in
  `_tke_bottom_dirichlet` across `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py`
  and the ORCA2 round 331, 332, 333, and 336 receipts. Both
  `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py --receipt docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_round336_trazdf_operand_receipt.md --from-heading "# ORCA2 round 336 — \`tra_zdf\` operand audit and acquisition handoff"`
  and `JAX_ENABLE_X64=1 pytest tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py`
  pass with zero failures, zero unmapped citations, zero map-audit failures, and
  all nine self-test controls firing.
- **Default ORCA2 landing gate and private-import ratchet**: All default ORCA2
  landing tests and `tests/test_no_private_cross_imports.py` pass on CPU under
  `JAX_ENABLE_X64=1`.
- **Review**: `independent review: operator`.
- **Disposition**: **LANDED**.
