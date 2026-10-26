# GYRE/ORCA2 merge second-pass preregistration — 2026-09-20

Status: **PREREGISTERED BEFORE THE SECOND-PASS MEASUREMENTS.**

This pass corrects the ORCA2 Langmuir evaluation selector exposed by the
independent merge review and records Decision 46: the shared ``eice`` selector
uses NEMO's compiled numbering.  No NEMO executable will be built or run.

## Registered measurements and verdicts

1. Dump the complete resolved GYRE and ORCA2 model configs, using
   ``ocean_config_to_dict`` with sorted JSON, before and after the fix.
   **CONFIRM:** the two GYRE files are byte-identical; the ORCA2 recursive diff
   contains exactly
   ``physics.vertical_mixing.tke.tke_langmuir_evaluation``, moving from
   ``nemo_literal`` to ``vectorized``.  **REFUTE:** any other moved leaf or any
   GYRE byte.
2. Re-run the existing Phase-2v ordered TKE walk gate from the certified
   records.  **CONFIRM:** ``ordered_rows`` is byte-identical to
   ``/data/abyssal/dbalwada/nemo-testcases-l4/phase2w/tke_walk.json`` and the
   resolved Langmuir selector is ``vectorized``.  **REFUTE:** any row byte or
   verdict moves.
3. Pin compiled ``zdftke.f90:260-263``: mode 1 equals scalar-libm
   ``tanh(10*fr_i)``, mode 2 equals raw ``fr_i``, and every integer outside
   ``{0,1,2,3}`` raises.  **Non-vacuity:** temporarily swap modes 1 and 2; both
   formula tests must fail, then restore and verify a clean tree apart from the
   intended diff.
4. Give ``KPPConfig.eice`` the same NEMO numbering by reusing the shared
   effective-ice-fraction dispatcher: 0 none, 1 tanh, 2 raw, 3 clipped fourfold.
   **CONFIRM:** direct KPP tests distinguish modes 1 and 2 and reject invalid
   values.  Defaults remain zero.
5. Audit every in-tree production/deck/document selection.  **CONFIRM:** zero
   selections change meaning: GYRE/default zero remains none, MPAS/tripole mode
   3 remains clipped fourfold, and ORCA2 mode 1 was already written for tanh.
   Test-only examples are reported separately from selections.
6. On the final commit, rerun the GYRE ten-step trajectory ladder, offline
   comparison, and 30-day control member against the committed ``before/``
   evidence from lane tip ``4cac617cd928``.  **CONFIRM:** all trajectory rows
   are identical, every residual array passes ``np.array_equal``, and all 30
   daily snapshots are byte-identical.  **REFUTE:** any moved row, array, or
   snapshot.
7. Run the four-file push gate and citation gate after all changes.
   **CONFIRM:** both finish with their own success verdicts and every planted
   citation violation fires.  Citation repair, if needed, is rigid re-anchoring
   only; source text at each cited extent must remain identical.

## Choices

- F1 is a correction selected by the user: GYRE keeps ``nemo_literal``, ORCA2
  returns to its parent ``vectorized`` value, and other cards retain their
  parent value.
- Decision 46 is selected by the user: TKE and KPP expose the compiled NEMO
  ``eice`` numbering, with no legacy linear meaning under mode 1.
- No threshold, cadence, data source, state field, or scientific reduction is
  selected here.
