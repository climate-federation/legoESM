# NEMO-testcases L2 GYRE round 69: JIT-native LDF routing receipt

Date: 2026-09-12. Final disposition: **HELD / RULE-12 REFUTED; no production
physics change remains**. Oracle producer
`3b3b045bd9e03b60330204e7590e4c4470b7a0ca`; 132 values admitted (43 of 63
inherited records exact and 20 changed).

## Verdict and first non-bit statement

The private JIT-native arm **CONFIRMED** the local source-order prediction, and
the temporary production transcription reproduced its complete frozen content
and kt=3 dictionaries exactly. It nevertheless **REFUTED** the landing claim:
the canonical Decision-36 comparison found 53 moved kt3--10 rows, and every
moved row contained at least one cell worsening by more than the permitted two
row-scale float64 ULPs. The first moved trajectory boundary is kt3-before T/S,
the carried result of the candidate's kt2 stage-3 source association. It owns
this walk. The production transcription and its no-op alias were withdrawn;
`ocean_model_latlon_cgrid.py` is byte-identical to round 68.

This is a cancelling-pair result, not evidence against NEMO's order. Compiled
GYRE clears and fills `Krhs` through advection and surface forcing, then at
stage 3 calls QSR, LDF, and ZDF in that order
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-950`,
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:917-965`). The
active LDF reads Kbb tracer gradients, evaluates Kmm-metric face fluxes, and
adds their divergence positively to `Krhs`
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:183-192`,
`:230-246`, `:287-305`). ZDF then forms the Kbb tracer plus Kmm-weighted
`Krhs` content before its solve
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565`). Thus the
source-derived route is still a faithful individual statement, but it cannot
ship separately from the larger FCT association error that presently cancels
part of it.

## Frozen causal measurement

The clean pre-edit report `round69_native_source_before.json` is stamped to
`076988602a83bc4eb0ee5bf67578c2dee5337e0a` and has SHA-256
`cc8f4bd8b3a0573695b174f2c66a8b13da10eb7bc176284cb4e80a239bead3de`.
The ordinary model and default-false arm match in all 23 state leaves / 198,956
cells and both 18,000-cell content arrays. The injected T/S source arrays are
also bit-identical to the same traced step's live GM/Redi arrays. The route
moves 17,990 T-content cells and 14,086 S-content cells; its returned kt3 state
moves 17,986 T and 14,043 S cells.

| frozen row | temperature | salinity |
|---|---:|---:|
| routed content maximum vs NEMO | `5.954039670541533e-5 K m` | `7.651457053725608e-6` |
| routed content RMS vs NEMO | `4.361510961132932e-6` | `5.281106958882405e-7` |
| routed kt3 maximum vs NEMO | `8.916073106490785e-7 K` | `7.235656340753849e-8` |
| routed kt3 RMS vs NEMO | `8.624264889977338e-8` | `6.389237340825272e-9` |
| native versus host association unequal / maximum | `2 / 2.842170943040401e-14` | `0 / 0` |

Temperature content improves 28.2x from the frozen un-routed
`1.679392691670500e-3 K m` maximum. This supersedes neither round 68's
three-cell host-reconstruction retraction nor its cause: the native candidate
itself differs from a host reconstruction in two T cells by one ULP.

Both preregistered plants fired and exited 1. The null route made both T and S
movement predicates false (`plant_native_null.json`, SHA-256
`cd4ed91ed95d35d421e748fd0c78bf0b4deab67fad0a7e658dd6cb2364efde66`);
the content plant changed exactly one T cell by
`2.842170943040401e-14 K m` (`plant_native_content_ulp.json`, SHA-256
`e9c2189dd6c2a785cfcf82fe70bffd93a490e4abc5ced13d2d32937346b0d723`).
Inherited dry-cell division warnings are diagnostic only; all acceptance rows
are wet-cell masked.

## Temporary production replay and retraction

The first post-edit report, `round69_native_source_after.json`, was refused by
two post-edit bookkeeping comparisons even though the four frozen physics
dictionaries matched. The preregistration's appended correction records that
retraction before the successful rerun: improvement must use round 68's
un-routed baseline, and the current association must match the native arm's
two-cell census rather than round 68's historical three-cell census.

The corrected clean report `round69_native_source_after_corrected.json` is
stamped to `a9de3cba7212d0f38efff3d325c8ce77c65176d1`, status **CONFIRMED**, SHA-256
`e6e6c96104a31e13dda69761adc73aed8da95b8daa56762f49b6f7a830a3c98f`.
It reproduces both native content dictionaries, both kt3 dictionaries, and the
two/zero T/S association census exactly, while proving the private selector
absent. That confirms the temporary edit transcribed the measured statement;
it does not override the subsequent trajectory failure.

Retraction lives in the instrument, not only this receipt. On the clean final
tree at `66329cef763952059cb91b5020f55389cbc7172d`, the same round-69 mode
exits 1 and prints **ROUND69 NATIVE SOURCE REFUTED**, with the original
un-routed T content/kt3 maxima
`1.679392691671e-3/1.627511417652e-4`. Report
`round69_final_retraction.json` is status REFUTED, SHA-256
`9472ef8d37e68c0e2da9375725d4a373e572920b0ad3c958d942249bee89d549`.

## Rule 12

The complete GYRE report `gyre_after_kt1_10.json` is stamped to the same clean
candidate commit, status DEBT, SHA-256
`2e8eb8429ac7a78e7a608a07291f249034c048c3f244c70d1e9002c2c1cec272`.
The ULP comparison against Decision 36's recorded after arm
`f78547b752f733c4d86f024df7effc6f5b2e376a` is **FAIL**
(`gyre_rule12_compare.json`, SHA-256
`9136e7110b2c888aded2e608d0ad1598a033fb6eb9d59d59fcadd1da67fddd9e`):
954 rows were compared, 53 moved, and all 53 violate the two-ULP no-worsening
bar in at least one cell. The largest oracle-residual worsening is
`104357871984.71875` row-scale ULPs. No AT-BAR/DEBT status changes occur and
first-over-bar remains kt2, but those two invariants cannot waive cellwise
worsening. At kt3-before, T improves in 9,632 cells and worsens in 8,352; S
improves in 7,789 and worsens in 6,254. This is the measured cancellation that
requires a pair analysis.

| card | changed statement | Rule-12 disposition |
|---|---|---|
| GYRE | temporary stage-3 LDF source route | **REFUTED** at kt1--10; candidate withdrawn |
| LOCK_EXCHANGE | none in final tree | exact before/after **UNREACHED** after the GYRE stop; final source equals its before arm |
| OVERFLOW | none in final tree | exact before/after **UNREACHED** after the GYRE stop; final source equals its before arm |
| DINO | none in final tree | execution gate **UNREACHED**; the candidate used the WS-only call site while DINO remains on the separate modified-leapfrog program. Its documented per-row cancellation risk remains open and no band statistic is used here. |
| ORCA2 | none in final tree; native record absent | **UNMEASURED WITH SPEC**: resolve its native card; record post-SBC, post-QSR, and post-LDF stage-3 `Krhs` plus pre/post-ZDF T/S for kt1--10; replay the content statement and an independent trajectory gate; require exact statement replay, every moved row registered, no AT-BAR loss, and no earlier first-over-bar boundary. |

The preregistered stopping rule was enforced. Days 1--30, tanks, DINO, and
ORCA2 were not run after the GYRE comparison rejected the only production
candidate; none is reported as passed. No configuration/default, carried
state, stabilizer, NEMO source/build/run, year harness, reconciliation gate,
freshwater pair, #1484 guard, or held manifest changed.

## Review and focused checks

The required separate review was invoked on clean commit
`01b17a379bf9b1ab75729dfce7c524480ee248cd` with
`codex exec --sandbox read-only`. It exited 1 before reviewing. Its terminal
result, verbatim, was **“Error: failed to initialize in-process app-server
client: Read-only file system (os error 30)”**. There is no `VERDICT:` line to
quote. The review requirement is **UNMET/BLOCKED**, and absence of a verdict is
not approval. The full log is `codex_round69_review.log`, SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The final focused admission/content/native-source/citation batch passes 32/32
(`focused_tests.log`, SHA-256
`a8d2261d677aeeb3f4dba9e47e480d49f3405e14f2bf19db36213fa56be93dd4`).
It includes the exact pytree census and the assertion that the refuted
production source text is absent. Python compilation, `git diff --check`, and
an empty production-model diff against `c70771c9d025` pass.

The citation gate checks all six compiled-source citations with no unmapped or
failed row (`round69_citation_gate.json`, PASS, SHA-256
`83f7ff9446eea51bec018e5839e2ee152851f7259a7ccdc147f8015644a75bec`).
Shifting the LDF face-flux citation by two lines produces
SYMBOL-NOT-AT-LINE and exits 1 (`round69_citation_plant.json`, SHA-256
`5b9bda869b1f7caad37fec65389a2895ca95ded5982eaee97f7e3102cd5bb0ab`).

## ASKED / UNASKED and OPEN

| state | item | disposition |
|---|---|---|
| ASKED | configuration choice | none encountered |
| UNASKED | configuration, carried state, stabilizer, NEMO, or harness change | none performed |

OPEN for round 70: do not retry LDF routing alone. Preregister the cancelling
pair between the source-confirmed LDF route and the larger complete-FCT
advection/association residual. Using the admitted round-64 split, make a
JIT-native two-variable matrix with untouched, LDF-only, FCT-only, and paired
arms at the same kt2 seed. Freeze exact helper content, kt3, and kt1--10
predictions before measuring. The pair is eligible only if its complete native
trajectory removes the 53-row regression while retaining the local content
gain; otherwise the first kt3-before T/S worsening remains the owner. Complete
FCT advection is still the largest measured residual after the LDF route.
