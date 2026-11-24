# SI3 round 24 — O15/O16 model and score preregistration

- Date: 2026-10-28. Lane `si3`, round `si3_r24`, authored by Codex.
- Parent: `87a1c8843803f0cb6263a2a84f44f8d83183ee96`.
- Scope: production model preparation and master-record scoring for O15
  (`ice_cor(kt,2)`) and O16 (`ice_update_flx` plus the recorded
  `ice_update_tau` boundary).
- Status: committed before reading any O15/O16 binary master-record frame.
- Independent review: operator (Claude), as required by the round note; the
  author does not review its own code.

## Pinned evidence and resolved deck

Master root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master`.
Producer commit: `cd34f1be648a05abbb23e4cb6bb2a9ed66e60fc5`. Amendment (2026-10-29 provenance audit): this was a pre-rebase commit and is no longer reachable.

- `manifest/run_manifest.json`: `bfb0b71d1500b7bc91c68a9e39c8af7a654b139258c0635ad5b1f25284b31fd0`
- `manifest/resource_usage.json`: `b30a5febe9248e5008999cb419953dad40c78658542a3c7221d301825270967c`
- `manifest/records.sha256`: `fbd3cb4d3450f428d8444dcb145bec3c1a832554e3e7e05d7f5b6a159daadd85`
- O15 projection `(13,14)`: `e1cd64c354294a071aca8496b014b59eb0b8f7be93d0304219e5c94b192ea292`
- O16 projection `(18,15,16,19,20)`: `6a3f9ed6f9330af1c65b603726c2ba407a28683f2de1b2a9dcaf9fd7dd3dc5e0`

The admitted resolved namelists select `ln_icethd=T`, `ln_cndflx=F`,
`ln_pnd_alb=T`, `nn_snwfra=2`, `nn_fsbc=1`, `ln_cpl=F`, and the production
deck selects `ln_drgice_imp=T`. No deck or configuration value will change.

## Frozen predictions

1. Admission and both projection digests above will pass unchanged.
2. O15 kind 13 fed through the production correction entry point will match
   every kind 14 field for all `kt=1..100` at `0` ULP, including signed zero.
3. O16 kind 15 fed through the production flux entry point will match every
   kind 16 field for all `kt=1..100` at `0` ULP. Kind 19 fed through the
   production implicit-drag stress entry point will match every kind 20 field
   for all `kt=1..100` at `0` ULP. Kind 18 will also exercise the production
   pre-aggregation call site that constructs kind 15 inputs.
4. One-ULP plants in O15 corrected state, O16 flux output, and O16 stress
   output will each fail. A model-arithmetic mutation at one executing
   statement per rung will produce a nonzero score and fail.
5. The D5 C1D one-year hash remains
   `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`;
   all callers outside O15/O16 remain bit-identical to the parent.

Any nonzero O15/O16 difference refutes the corresponding prediction. The
first differing `(kt, field, cell)` and producing NEMO/legoESM statement will
be localized, and that rung will stop without a post-score model change.

## Choices

- ASKED: O15/O16 scope, admitted record, production deck, CPU fp64/libm,
  0-ULP bar, all exit fields, and required plants are fixed by the round prompt
  and Decisions 158, 159, 162, 173, 179, 180, 184, and 186.
- UNASKED: none.

## Resume preregistration (2026-10-28)

The user confirmed the localized O15 mechanism and resumed this same round
after commit `2ca7bf533a`.  Before the resumed model edit, the predictions are:

1. Materializing each completed donor and receiver statement through the
   shared `nemo_source_round` primitive will preserve NEMO's nested transfer
   order under CPU JIT.  O15 will score every recorded field and category at
   0 ULP for all `kt=1..100` through `si3_o15_step`.
2. Removing those inter-statement materializations while retaining the same
   arithmetic will reproduce a nonzero O15 score; the control must fail.
3. Kind 20 will score only the owned interior written by NEMO's loops,
   `ji=ntsi..ntei` and `jj=ntsj..ntej`, excluding both two-cell halos.
4. Kind 15 remains unscored and untranslated while the user decides between
   a versioned translation and re-acquisition.  This round will record only
   the writer-fixed field order and a permutation check using independently
   known values.
5. The D5 full-year output hash remains
   `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`,
   and the unchanged O8--O12 production score remains 0 ULP.

Any nonzero baseline O15 field/category result, a barrier-stripped control
that remains 0 ULP, a changed D5 hash, or a changed O8--O12 score refutes the
corresponding prediction.  The resumed choices are all explicit in the user
instruction; no O16 kind-15 implementation choice is made.

## Resume 2 preregistration — Decision 187 (2026-10-28)

The user selected the versioned kind-15 reader translation, with re-acquisition
as the mandatory fallback if the proof or O16 score is not clean.  Before the
reader and score changes, the frozen predictions are:

1. For master format version 1 and kind 15 only, translating the three reader
   labels at the writer positions from `cloud_fra,qsr,sfx` to
   `qsr,sfx,cloud_fra` will make the immutable bytes agree with the committed
   writer's actual calls at `o8_o16_master_writer.F90:454-457`.  Other versions
   and kinds remain unchanged or fail closed.
2. At `kt=1,2,100`, all 12 state fields common to kinds 15 and 18 remain
   bit-identical; translated `qsr` remains bit-identical to the independently
   recorded `qsr_oce`; translated `cloud_fra` remains uniformly equal to
   `pp_cldf=0.81` from `src/OCE/SBC/sbc_phy.F90:69`.
3. A wrong permutation plant will fail those independent checks before any
   production score is accepted.
4. Kind 18 followed through the production `si3_o16_step` call site, including
   `si3_ice_update_flx`, will match every kind-15 aggregation field and every
   kind-16 output field at 0 ULP for all `kt=1..100`.  The already certified
   O15 and kind-20 owned-interior scores remain 0 ULP.
5. A one-ULP kind-16 output plant and an executing O16 heat-flux arithmetic
   mutation will each fail.  Any nonzero baseline result, failed proof check,
   or plant that stays green stops this round and requires re-acquisition with
   corrected writer order.

ASKED choices: version-1/kind-15-only translation; writer-source and independent
field proof; wrong-permutation plant; all-step/all-field production scoring;
0-ULP bar; re-acquisition fallback.  UNASKED choices: none.
