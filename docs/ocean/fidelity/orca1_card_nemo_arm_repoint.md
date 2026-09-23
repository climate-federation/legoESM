# ORCA1 card — resolved-config diff for the NEMO branch-isomorphism collapses

One section per collapsed row of `nemo_branch_isomorphism_map.md`. Each section
lists **one row per selector whose RESOLVED value on the ORCA1/OMIP standard
card changes**, with the NEMO namelist arm it now matches (`file:line`). This
is the record to read **before** the next ORCA1 baseline run: every row here is
a one-variable change to that card, so the run it precedes is not comparable to
the previous ORCA1 baseline unless the rows are accounted for.

The ORCA1 tripole config cannot be *run* on the audit machine (no mesh file, no
GPU), but its config builder **can** be instantiated there, so every "before"
value below is measured, never read off a comment:

```
PYTHONPATH=packages/... JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -c \
  "from legoesm.ocean.fidelity.nemo_match_recipe import \
   nemo_match_tripole_model_config as f; print(f().barotropic)"
```

The card itself is `scripts/cluster/omip_nemo/run_standard_faithful_1deg.sbatch`
→ `scripts/run/run_omip_core2.py --grid tripole`.

---

## S-16 — `dyn_spg_ts` continuity / transport accumulation / surface PGF

**NO ROWS. The ORCA1 card is not on this code path, and nothing about it
changes.**

| selector | before | after | NEMO arm |
|---|---|---|---|
| *(none)* | — | — | — |

Why the empty diff, measured on branch `fidelity/nemo-branch-isomorphism-audit`
at `646415f02`, fp64:

* The ORCA1 standard card resolves `barotropic.barotropic_solver =
  "implicit_cn"`. `run_standard_faithful_1deg.sbatch` passes no
  `--barotropic-solver`, so it takes the builder's value; the five other
  committed ORCA/eORCA cards pass `--barotropic-solver implicit_cn`
  explicitly.
* `implicit_cn` dispatches to
  `packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py`,
  which has its **own** private `_depth_average_to_faces` (`:96`, called at
  `:1370`) and reads **none** of S-16's five selectors. The split-explicit
  `_run_substep_loop` those selectors live in never executes on ORCA1.
* So the five fields resolving to `"generic"` on the ORCA1 card is an **inert
  value, not a running arm**. The earlier map claim that "ORCA1 runs the
  generic arm" is retracted (see the S-16 retraction in
  `nemo_branch_isomorphism_map.md`).

S-16 therefore stays **BLOCKED**: the generic arm it would delete is executed
by 19 ocean test-matrix experiments (`default_wright_v1` /
`legoesm_linear_v1`) plus the `legoesm_nemo_like_v1` catalog dycore (renamed
from `nemo_v1` 2026-09-02), and the committed test
`test_barotropic_continuity_and_drag.py::
test_association_selector_holds_face_depth_and_drag_fixed` proves the two arms
give bit-distinct `eta`. Unblocking is a separate decision about those 19
cases, not an ORCA1 re-point.

**2026-09-02 UPDATE — the "separate decision" above is now made.** USER
DECISION: keep the generic arm as a legitimate non-NEMO fork (it is the real
dycore of the 19 cases named above) and give it a real reference instead of
leaving it collapse-BLOCKED indefinitely. The reference: legoESM's own
in-house forward-backward split-explicit design (commit `adbb49f83`,
2026-04-08, #87), whose time-averaging and BEBT closure explicitly followed
the MOM6/ROMS family (Hallberg 1997; Shchepetkin & McWilliams 2005) per their
own introducing commits — full evidence and the mechanical registry fix are
in `nemo_branch_isomorphism_map.md`'s dated addendum under the S-16 section.
No re-baseline of the 19 idealized cases. Companion decision, same PR: the
recipe formerly named `nemo_v1` is RENAMED to `legoesm_nemo_like_v1` (it
resolves this generic arm, not NEMO's `dyn_spg_ts`, so the old name
overclaimed fidelity precisely on the routine this doc is about — see
`recipes.py`). This document's own references above are updated to the new
name.
