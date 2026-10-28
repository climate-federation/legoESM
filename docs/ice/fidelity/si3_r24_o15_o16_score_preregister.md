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
Producer commit: `cd34f1be648a05abbb23e4cb6bb2a9ed66e60fc5`.

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
