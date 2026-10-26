#!/usr/bin/env python
"""Mutation-sensitivity census for the FV3 duo-grid JAX lane — a gate ON the gates.

WHY THIS EXISTS
---------------
`docs/atmosphere/reviews/glm_review_strategy_2026-08-14.md` finding 4 (BLOCKER):

    "Zero demonstrated detection power; the track record is exactly what a
    regression harness produces.  Across all runs, not one of ~500 unit gates
    went red on a code defect, while codex found a BLOCKER in `_rezone` by
    *reading* -- detection has only ever been demonstrated where a reviewer
    looked.  And 'pinned to measurement' makes those gates structurally
    regression tests on the port itself: they cannot fail on the code they
    certified, only on later change.  Green is being narrated as correctness."

That is correct.  Seven runs of ~500 gates have never once gone red on a code
defect, so "the suite is green" currently carries no information about whether
the suite CAN go red.  This script measures that directly: it injects a
catalogue of SEMANTIC mutations (each one violates a specific, grep-verified
line of the pinned oracle), runs a targeted slice of the suite against each,
and records whether the slice went red and WHICH gate went red first.

The deliverable is the matrix, and the exit code: **non-zero if any mutation
that is expected to be observable is MISSED by every gate.**

HOW MUTATIONS ARE APPLIED -- the design decision that matters
-------------------------------------------------------------
**Nothing in the worktree is ever modified.**  The tree is not edited and then
restored; restore is a no-op because there was never anything to restore.  This
is not fastidiousness: the campaign has already been bitten by a stale file (a
`cp` aliased to prompt silently did nothing and a job was submitted against a
stale tree -- STATE lesson 15), and an edit-in-place mutation runner that dies
mid-run -- OOM, walltime, a preempted node -- leaves a MUTATED module on disk,
after which the next job certifies mutated code and reports it green.

Instead, per mutation:

1.  A scratch package directory is built at
    ``<work>/mutants/<ID>/legoesm/<pkg>/``.  Every entry of the real package is
    **symlinked** into it, except the one module under test.
2.  That one module is written as a **real file** containing the mutated text.
3.  pytest runs with the scratch dir **prepended to PYTHONPATH**, so the
    mutant shadows the real module.  `legoesm` is a PEP-420 namespace package
    (`packages/core/legoesm` has no `__init__.py`) while `legoesm.core` and
    `legoesm.grids` are REGULAR packages (they do) -- which is exactly why the
    symlink farm is required: a scratch dir holding only the single mutated
    file would shadow the whole `legoesm.core` package and every sibling module
    would vanish.  Verified 2026-08-14 by inspecting the `__init__.py` set.
4.  The scratch tree is deleted at the end.  A crash leaves scratch garbage in
    the work dir and a pristine worktree.

Shadowing is verified, not assumed (a mutation that silently failed to apply
reads as "no gate caught it", or worse as "suite green"):

*   the applier asserts the anchor text occurs **exactly once**, and re-reads
    the written file to assert the new text is present and the old text absent;
*   a subprocess with the run's exact environment imports the module and
    asserts ``os.path.realpath(mod.__file__)`` is the scratch file.  The
    unmutated siblings are symlinks, so their realpath resolves back into the
    repo -- only the genuinely-mutated module can satisfy this.  Together these
    two checks prove the loaded module is the mutated one, with **zero** change
    to the module's content (no injected sentinel that a source-inspecting test
    could trip on and manufacture a false "caught").

Instrument controls (this is a diagnostic, so it is untrusted until it passes
its own controls -- CLAUDE.md, "validate the instrument before quoting it"):

*   **NULL controls.**  One per mutated module: the identical staging path,
    with the module written back byte-identical.  If a NULL control goes red,
    the shadowing mechanism itself perturbs the suite and every result in the
    run is confounded -- the script exits 2 and reports no verdicts.
*   **Baselines, not green-suite assumptions.**  Every selection is first run
    UNMUTATED and its per-test outcomes recorded.  A mutation counts as CAUGHT
    only if a test that **passed in the baseline** fails under the mutant.  The
    suite currently carries known non-code failures (TOL-PENDING bounds), and
    without this a pre-existing red would be read as detection.
*   **Empty selections are a hard error**, never a silent no-op pass.

SCOPE CONTROL
-------------
A full suite run per mutation is unaffordable.  Each catalogue entry therefore
declares the SMALLEST selection that should plausibly catch it; that runs
first.  Only if it stays green does the run escalate to the module's whole test
file, and only if THAT stays green does it escalate to every JAX-lane gate
file.  "Caught only on escalation" is itself a finding about gate targeting and
is reported as its own verdict, not folded into CAUGHT.

USAGE
-----
    python scripts/validate/fv3_native/mutation_census.py \
        --repo /path/to/pinned/worktree \
        --work /burg-archive/glab/users/pg2328/fv3_duo_gaps/mutcensus_$JOBID \
        --out  docs/atmosphere/fv3_mutation_census.md

    --dry-run       stage every mutant and verify the anchors apply; run no
                    tests.  Seconds, one core, no JAX -- safe on a login node
                    and the cheap way to check the catalogue after a refactor.
    --only M01,M08  run a subset (ids as in the catalogue).
    --max-stage {fast,file,all}   cap the escalation ladder.

Exit codes: 0 all expected-observable mutations caught; **1 at least one
mutation MISSED by every gate** (the proven coverage hole -- this is the exit
code the review's finding 4 is asking for); 2 the run cannot answer -- the
instrument is broken (NULL control red, staging or shadow verification failed)
or the escalation budget ran out, in which case no detection verdict from it is
trustworthy.  1 and 2 are deliberately different: "no gate sees this defect" and
"we did not finish looking" must never be reported as the same thing.

ORACLE
------
Every citation below was verified with grep/sed against the pinned tree
    /burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean
on 2026-08-14.  No other copy of the source is cited (STATE lesson 3: three
copies exist and their line numbers differ).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

ORACLE_TREE = ("/burg-archive/glab/users/pg2328/fv3_oracle_pinned/"
               "atmos_cubed_sphere-symmetryclean")

#: Every JAX-lane gate file, in the order stage 3 runs them.
ALL_LANE_TESTS = [
    "tests/grids/test_fv3_duo_halos.py",
    "tests/grids/test_fv3_tp_core.py",
    "tests/grids/test_fv3_pgrad.py",
    "tests/grids/test_fv3_mapz.py",
    "tests/grids/test_fv3_nh_core.py",
    "tests/grids/test_fv3_duo_sw_core.py",
    "tests/grids/test_fv3_duo_stepper.py",
]


@dataclass(frozen=True)
class Mutation:
    """One semantic mutation of one JAX-lane module.

    ``old``/``new`` are exact source text.  ``old`` MUST occur exactly once in
    the target module or the mutation is refused -- an anchor that matches
    twice would mutate a site the catalogue does not describe, and an anchor
    that matches zero times means the module drifted and the entry is stale.
    Both are reported as APPLY-FAILED, never as a detection result.
    """

    mid: str
    title: str
    module: str                 # repo-relative path to the JAX-lane module
    old: str
    new: str
    oracle: str                 # the pinned-oracle file:line the mutant violates
    why: str                    # what the oracle says, in one line
    mode: str                   # the failure-mode class (strategy doc section 8)
    fast: list[str]             # smallest test selection: [file, ...]
    fast_k: str = ""            # optional -k expression for that selection
    expect: str = "caught"      # "caught" | "unknown" | "null"
    note: str = ""

    @property
    def pkg_root(self) -> str:
        """``packages/core`` for ``packages/core/legoesm/core/x.py``."""
        head, _, _ = self.module.partition("/legoesm/")
        return head

    @property
    def pkg_rel(self) -> str:
        """``legoesm/core`` for ``packages/core/legoesm/core/x.py``."""
        _, _, tail = self.module.partition("/legoesm/")
        return "legoesm/" + tail.rsplit("/", 1)[0]

    @property
    def modfile(self) -> str:
        return self.module.rsplit("/", 1)[1]

    @property
    def dotted(self) -> str:
        return (self.pkg_rel.replace("/", ".") + "."
                + self.modfile[:-len(".py")])

    @property
    def file_stage(self) -> list[str]:
        """Stage-2 selection: the module's own whole gate file."""
        return [_MODULE_TEST_FILE[self.module]]


#: module -> its dedicated gate file (stage 2 of the escalation ladder).
_MODULE_TEST_FILE = {
    "packages/core/legoesm/core/fv3_mapz.py":
        "tests/grids/test_fv3_mapz.py",
    "packages/core/legoesm/core/fv3_tp_core.py":
        "tests/grids/test_fv3_tp_core.py",
    "packages/core/legoesm/core/fv3_duo_sw_core.py":
        "tests/grids/test_fv3_duo_sw_core.py",
    "packages/core/legoesm/core/fv3_duo_stepper.py":
        "tests/grids/test_fv3_duo_stepper.py",
    "packages/core/legoesm/grids/fv3_duo_halos.py":
        "tests/grids/test_fv3_duo_halos.py",
}

_MAPZ = "packages/core/legoesm/core/fv3_mapz.py"
_SW = "packages/core/legoesm/core/fv3_duo_sw_core.py"
_HALOS = "packages/core/legoesm/grids/fv3_duo_halos.py"
_TP = "packages/core/legoesm/core/fv3_tp_core.py"
_STEP = "packages/core/legoesm/core/fv3_duo_stepper.py"

T_MAPZ = "tests/grids/test_fv3_mapz.py"
T_SW = "tests/grids/test_fv3_duo_sw_core.py"
T_HALOS = "tests/grids/test_fv3_duo_halos.py"
T_TP = "tests/grids/test_fv3_tp_core.py"
T_STEP = "tests/grids/test_fv3_duo_stepper.py"


# =====================================================================
# THE CATALOGUE
#
# Each entry is a defect a reviewer could plausibly ship, it changes
# PHYSICS or INDEXING (never a comment, never a rename), and it names the
# pinned-oracle line it violates -- so the catalogue doubles as a
# fidelity checklist for the ported sites.
# =====================================================================

MUTATIONS: list[Mutation] = [

    # ---- 1. the two review-found real defects, replayed ------------
    Mutation(
        mid="M01",
        title="_rezone: q_span = qsum / denom formed unconditionally",
        module=_MAPZ,
        old=("        den_safe = jnp.where(inside | (~found), "
             "jnp.ones_like(denom),\n"
             "                             denom)\n"
             "        q_span = qsum / den_safe                         "
             "# :1449\n"),
        new=("        q_span = qsum / denom       "
             "# MUTANT M01: mask removed\n"),
        oracle="fv_mapz.F90:1422-1425 (in map_scalar, :1361) vs :1449",
        why=("the contained-cell branch assigns q2 at :1422-1423 and "
             "`goto 555` at :1425 -- it NEVER reaches the :1449 divide, so a "
             "zero-thickness target inside one source layer is legal there; "
             "forming the quotient anyway is 0/0 in the primal select and in "
             "the adjoint of the branch that WAS taken"),
        mode="R1b unevaluated-branch divide (codex BLOCKER, fv3_mapz.py:998)",
        fast=[T_MAPZ], fast_k="zero_thickness",
    ),
    Mutation(
        mid="M02",
        title="_rezone: esl = dp / dp1_m on every scan iteration",
        module=_MAPZ,
        old=("            dp1_safe = jnp.where(active & (~whole), dp1_m,\n"
             "                                 jnp.ones_like(dp1_m))\n"
             "            esl = dp / dp1_safe\n"),
        new=("            esl = dp / dp1_m       "
             "# MUTANT M02: mask removed\n"),
        oracle="fv_mapz.F90:1431-1444 (the m = l+1..km walk)",
        why=("the Fortran walk starts at l+1 and `goto 123` at :1442 STOPS at "
             "its first partial layer, so it never divides by dp1 outside "
             "[l+1, terminating m]; the static-length scan visits every m"),
        mode="R1b unevaluated-branch divide (codex MAJOR, fv3_mapz.py:983)",
        fast=[T_MAPZ], fast_k="zero_thickness",
    ),
    Mutation(
        mid="M03",
        title="d_sw1_duo west panel edge: raw divide in both arms",
        module=_SW,
        old=("        ut = set_ut(1, 1, jsd, jed,\n"
             "                   _sel_div(ucc * dt > 0.0, ucc,\n"
             "                            rd(sg, 0, 0, jsd, jed, 2),\n"
             "                            rd(sg, 1, 1, jsd, jed, 0)))\n"),
        new=("        ut = set_ut(1, 1, jsd, jed,       # MUTANT M03\n"
             "                   jnp.where(ucc * dt > 0.0,\n"
             "                             ucc / rd(sg, 0, 0, jsd, jed, 2),\n"
             "                             ucc / rd(sg, 1, 1, jsd, jed, 0)))\n"),
        oracle="sw_core.F90:660-663 (in d_sw1, :500)",
        why=("the oracle's `if (uc(1,j)*dt > 0.)` is a Fortran branch, so only "
             "the SELECTED sin_sg entry is ever divided by; a bare jnp.where "
             "over two divisions evaluates BOTH arms, so a zero in the "
             "unselected entry gives NumPy a finite answer and JAX an Inf/NaN, "
             "and reverse mode carries NaN*0 into the selected branch"),
        mode="R1b unevaluated-branch divide (codex MAJOR, job 9404230)",
        note=("SCOPE, stated so the result is not over-read: the four "
              "panel-edge sites sit under `plain_edges = not "
              "flags.bounded_domain` (the oracle's `.not. bounded_domain "
              ".or. .not. duogrid` at sw_core.F90:656), and the production "
              "duo decks resolve bounded_domain=True -- so these lines are "
              "SKIPPED in a real duo run.  The `geo_dsw` fixture builds a "
              "bounded_domain=False gridstruct "
              "(tests/grids/test_fv3_duo_sw_core.py:115), which is why the "
              "selection can exercise them at all.  A CAUGHT here proves the "
              "gate works on the plain-conventions lane, not that the duo "
              "lane would notice."),
        fast=[T_SW], fast_k="sel_div",
    ),

    # ---- 2/3. the duo barriers -------------------------------------
    Mutation(
        mid="M04",
        title="barrier 1 allflux blend constant 0.5 -> 0.25",
        module=_HALOS,
        old="        0.5 * (flat[:, bl.dst] + bl.sign * flat[:, bl.src]))\n",
        new=("        0.25 * (flat[:, bl.dst] + bl.sign * flat[:, bl.src]))"
             "  # MUTANT M04\n"),
        oracle="dyn_core.F90:879-884 (barrier 1, mpp_get_boundary at :874)",
        why=("all four blend statements are `0.5*(mine + neighbour)`; any "
             "other weight makes the shared-edge flux non-conservative AND "
             "still single-valued, which is the exact class of defect the "
             "antisymmetry row cannot see (review finding 2)"),
        mode="latent mis-ported constant (strategy section 8, row 1)",
        fast=[T_HALOS], fast_k="allflux or barrier1",
    ),
    Mutation(
        mid="M05",
        title="barrier 1 also averages slot 2 (w)",
        module=_HALOS,
        old=("    tab.allflux_slots = np.asarray(\n"
             "        [iq - 1 for iq in range(1, 4 + int(nq) + 1)\n"
             "         if iq == 1 or iq >= 4], dtype=np.int32)\n"),
        new=("    tab.allflux_slots = np.asarray(       # MUTANT M05\n"
             "        [iq - 1 for iq in range(1, 4 + int(nq) + 1)\n"
             "         if iq == 1 or iq == 2 or iq >= 4], dtype=np.int32)\n"),
        oracle="dyn_core.F90:856",
        why=("`if (iq==1 .or. iq==4 .or. iq>4 ) then` -- slot 2 (w) and slot 3 "
             "(q_con) are deliberately EXCLUDED from the average and must come "
             "back byte-identical"),
        mode="namelist/scope exclusion dropped (strategy section 8, row 6)",
        fast=[T_HALOS], fast_k="allflux or barrier1",
    ),

    # ---- 4. halo bound off-by-one ----------------------------------
    Mutation(
        mid="M06",
        title="A-grid strip exchange write window shifted by one cell",
        module=_HALOS,
        old=("    eff = sc.base * np.where(sc.sgn_pow == 1, float(sign), 1.0)\n"
             "    return flat.at[sc.dst].set(flat[sc.src] * eff)\n"),
        new=("    eff = sc.base * np.where(sc.sgn_pow == 1, float(sign), 1.0)\n"
             "    dst = ((sc.dst + 1)       # MUTANT M06: +1 write window\n"
             "           if sc.name == \"exchange_agrid_scalar_halos.strips\"\n"
             "           else sc.dst)\n"
             "    return flat.at[dst].set(flat[sc.src] * eff)\n"),
        oracle="fv_duogrid.F90:456-502 (ext_scalar_2d) / :505-569 (3d)",
        why=("the A-scalar halo strips are written at the field's own "
             "staggering with the oracle's exact index ranges; a one-cell "
             "shift leaves ring-1 stale and feeds the upwind selection"),
        mode="halo loop-bound bug (strategy section 8, row 2)",
        fast=[T_HALOS], fast_k="exchange_agrid",
    ),

    # ---- 5. a dropped exchange -------------------------------------
    Mutation(
        mid="M07",
        title="ext_scalar_sixface is a silent no-op at the B stagger",
        module=_HALOS,
        old=("    if stag == \"B\":\n"
             "        f6 = exchange_bgrid_scalar_halos(f6, tab)\n"
             "        f6 = k2e_remap_halo_rings(f6, tab, \"B\", "
             "ring=\"stepper\")\n"
             "        return corner_lagrange_fill(f6, tab, \"b3\")\n"),
        new=("    if stag == \"B\":\n"
             "        return f6       # MUTANT M07: exchange dropped\n"),
        oracle="dyn_core.F90:652 (ext_scalar(divgd, ..., 1, 1))",
        why=("the (1,1)-staggered ext_scalar on divgd runs every substep under "
             "`duogrid .and. nord > 0`; the JAX stepper calls it at "
             "fv3_duo_stepper.py:696, so dropping it is a live silent no-op, "
             "which is the campaign's own most expensive defect class "
             "(STATE lesson 1: fill_corners, 9.79e-6 -> 1.19e-9)"),
        mode="silent no-op call (strategy section 8, row 7; repo rule R5)",
        fast=[T_HALOS], fast_k="ext_scalar_b",
    ),

    # ---- 6. a sign flip in a flux term -----------------------------
    Mutation(
        mid="M08",
        title="d_sw2_duo delp update: y-flux divergence sign flipped",
        module=_SW,
        old=("    dpn = dp0 + (cw(fx, is_, ie, js, je) "
             "- cw(fx, is_ + 1, ie + 1, js, je)\n"
             "                 + cw(fy, is_, ie, js, je)\n"
             "                 - cw(fy, is_, ie, js + 1, je + 1)) * ra\n"),
        new=("    dpn = dp0 + (cw(fx, is_, ie, js, je) "
             "- cw(fx, is_ + 1, ie + 1, js, je)\n"
             "                 - cw(fy, is_, ie, js, je)       # MUTANT M08\n"
             "                 - cw(fy, is_, ie, js + 1, je + 1)) * ra\n"),
        oracle="sw_core.F90:1189-1190 (in d_sw2, :1000)",
        why=("`delp(i,j) = delp(i,j) + (fx(i,j)-fx(i+1,j)+fy(i,j)-fy(i,j+1))"
             "*rarea(i,j)` -- the south face enters POSITIVE; flipping it "
             "destroys the flux-form divergence and global dry mass"),
        mode="sign convention (CLAUDE.md mandatory sign check)",
        fast=[T_SW], fast_k="d_sw2",
    ),

    # ---- 7. the stagger swap I FALSELY reported --------------------
    Mutation(
        mid="M09",
        title="k2e record (i, j) pair transposed at the consumer",
        module=_HALOS,
        old="    for (fi, fj), lv, cw in zip(ij, loc, coef):\n",
        new=("    for (fj, fi), lv, cw in zip(ij, loc, coef):"
             "       # MUTANT M09\n"),
        oracle="fv_duogrid.F90:977-1137 (cube_rmp)",
        why=("the k2e ring remap classifies each record as a W/E or S/N ring "
             "target from (fi, fj) and derives the destination column/row from "
             "them; transposing the pair reassigns every record"),
        mode="axis-order / stagger confusion (assumption gate: layout is an API)",
        expect="unknown",
        note=("THIS IS THE DEFECT I FALSELY REPORTED AS CONFIRMED (STATE "
              "'failure triage' 1, retracted).  The retraction's measurement "
              "says the c12_*_ij tables differ from compute_fv3_native_k2e "
              "only in ROW ORDER, and that the A and B record->value maps are "
              "INVARIANT under a column swap -- which predicts this mutation "
              "is INERT for the two staggers this lane uses.  A transposed "
              "PAIR is not the same edit as a reordered table, so the "
              "prediction may not hold.  The census settles it empirically; "
              "the outcome does NOT gate the exit code, precisely because "
              "'no gate caught it' and 'nothing to catch' are different "
              "findings and only a measurement separates them.  See "
              "tests/grids/test_fv3_duo_halos.py::"
              "test_k2e_column_convention_is_inert_for_the_staggers_this_"
              "lane_uses."),
        fast=[T_HALOS], fast_k="k2e",
    ),

    # ---- 8. a dropped limiter branch -------------------------------
    Mutation(
        mid="M10",
        title="xppm iord=10: monotonicity limiter forced to the raw arm",
        module=_TP,
        old=("        bl = bl.at[w_b, :].set(\n"
             "            jnp.where(flat, 0.0, jnp.where(use_c, bl_c, bl_w)))\n"
             "        br = br.at[w_b, :].set(\n"
             "            jnp.where(flat, 0.0, jnp.where(use_c, br_c, br_w)))\n"),
        new=("        bl = bl.at[w_b, :].set(bl_w)       # MUTANT M10\n"
             "        br = br.at[w_b, :].set(br_w)       # MUTANT M10\n"),
        oracle="tp_core.F90:561-571 (in xppm, :308-681)",
        why=("the `if (near_zero) ... elseif (abs(3*(bl+br)) > abs(bl-br))` "
             "chain flattens at a near-uniform stencil and applies the "
             "pmp/lac monotonicity bounds on a steep one; forcing the raw "
             "`al - q1` arm removes both and silently changes the transport "
             "scheme -- a branch a smooth-state gradient gate never enters"),
        mode="limiter branch dropped (strategy section 7, the switching surfaces)",
        fast=[T_TP], fast_k="xppm",
    ),

    # ---- 9. the STALE-BY-ONE halo, inverted ------------------------
    Mutation(
        mid="M11",
        title="stepper adds the post-step D-wind refresh the oracle omits",
        module=_STEP,
        old="    u6, v6 = stage[\"u\"], stage[\"v\"]\n",
        new=("    u6, v6 = ext_vector_dgrid_sixface(stage[\"u\"], "
             "stage[\"v\"], tab)       # MUTANT M11\n"),
        oracle="dyn_core.F90:1335-1338 vs :471",
        why=("the post-step duo block refreshes ONLY delp (:1336) and pt "
             "(:1337) -- there is no ext_vector there; the D-vector exchange "
             "runs at the NEXT step's entry (:471).  A twin that refreshes the "
             "winds here LOOKS more correct and is a divergence; the STATE doc "
             "calls this the single easiest place in the port to be wrong"),
        mode="joint bug -- kernels right, the join wrong (section 8, row 8)",
        fast=[T_STEP], fast_k="stale_by_one",
    ),

    # ---- 10. an entry_ascalar cadence break ------------------------
    Mutation(
        mid="M12",
        title="entry A-scalar exchange fires on every acoustic substep",
        module=_STEP,
        old="                      entry_ascalar=(it == 0))\n",
        new="                      entry_ascalar=True)       # MUTANT M12\n",
        oracle="dyn_core.F90:432 (gating :437-438)",
        why=("`if ( it==1 ) then` gates ext_scalar(delp) and ext_scalar(pt) to "
             "the FIRST inner step only; firing them every substep is a "
             "different exchange schedule and changes every step after the "
             "first"),
        mode="joint bug -- wrong cadence at the join (section 8, row 8)",
        fast=[T_STEP], fast_k="entry_ascalar or fires_entry or cadence",
    ),

    # ---- extra, from the strategy's own section-8 catalogue --------
    Mutation(
        mid="M13",
        title="corner_lagrange_fill is a silent no-op",
        module=_HALOS,
        old=("    st_dir, st_dia = tab.corner[key]\n"
             "    f6 = jnp.asarray(f6)\n"
             "    shape = f6.shape\n"
             "    flat = f6.reshape(-1)\n"
             "    out = _apply_stencil(flat, flat, st_dir)\n"
             "    out = _apply_stencil_pair(out, flat, st_dia)\n"
             "    return out.reshape(shape)\n"),
        new=("    st_dir, st_dia = tab.corner[key]\n"
             "    return jnp.asarray(f6)       # MUTANT M13: corner fill "
             "dropped\n"),
        oracle="fv_duogrid.F90:1719-1903 (fill_corner_region_2d)",
        why=("the corner regions are filled by 24 directional Lagrange "
             "targets plus 12 averaged diagonals; this is the DIRECT replay of "
             "the campaign's largest measured defect, where an inert corner "
             "call was the whole 9.7882e-6 -> 1.1866e-9 hydrostatic boundary "
             "floor (STATE lesson 1) -- so the census answers whether the same "
             "class would be caught next time by a gate rather than by a diff"),
        mode="silent no-op call (strategy section 8, row 7; repo rule R5)",
        fast=[T_HALOS], fast_k="corner_lagrange",
    ),
]


def null_controls() -> list[Mutation]:
    """One identity-staging control per mutated module.

    Same staging path, same PYTHONPATH prepend, same symlink farm -- but the
    module is written back byte-identical.  If any of these goes red the
    shadowing MECHANISM is perturbing the suite (an import-order effect, a
    stale .pyc, a path-dependent fixture) and no detection verdict from the
    run means anything.
    """
    out = []
    seen = []
    for m in MUTATIONS:
        if m.module in seen:
            continue
        seen.append(m.module)
        out.append(Mutation(
            mid="NULL-" + Path(m.module).stem,
            title="null control: identical module, staged and shadowed",
            module=m.module,
            old="", new="",
            oracle="n/a",
            why="proves the staging mechanism does not itself change outcomes",
            mode="instrument control",
            fast=m.file_stage,
            expect="null",
        ))
    return out


# =====================================================================
# staging
# =====================================================================

@dataclass
class Staged:
    ok: bool
    reason: str = ""
    root: Path | None = None
    modpath: Path | None = None


def stage_mutant(repo: Path, work: Path, m: Mutation) -> Staged:
    """Build the shadow package for ``m``.  Never touches ``repo``."""
    src_pkg = repo / m.pkg_root / m.pkg_rel
    if not src_pkg.is_dir():
        return Staged(False, f"package dir missing: {src_pkg}")
    src_mod = repo / m.module
    if not src_mod.is_file():
        return Staged(False, f"module missing: {src_mod}")

    root = work / "mutants" / m.mid
    if root.exists():
        shutil.rmtree(root)
    dst_pkg = root / m.pkg_rel
    dst_pkg.mkdir(parents=True)

    for entry in sorted(src_pkg.iterdir()):
        if entry.name in ("__pycache__", m.modfile):
            continue
        (dst_pkg / entry.name).symlink_to(entry.resolve())

    text = src_mod.read_text()
    if m.expect == "null":
        new_text = text
    else:
        n = text.count(m.old)
        if n != 1:
            return Staged(False,
                          f"anchor occurs {n} times (need exactly 1) in "
                          f"{m.module}")
        new_text = text.replace(m.old, m.new)
        if new_text == text:
            return Staged(False, "replacement is a no-op (old == new)")

    dst_mod = dst_pkg / m.modfile
    dst_mod.write_text(new_text)

    # re-read what was actually written: the mutation must be present and the
    # original text gone.  A write that silently did nothing is exactly the
    # failure this whole design exists to make impossible.
    back = dst_mod.read_text()
    if m.expect != "null":
        if m.new not in back:
            return Staged(False, "written file does not contain the mutation")
        if m.old in back:
            return Staged(False, "written file still contains the original")
    elif back != text:
        return Staged(False, "null control staged a non-identical file")

    # A BROKEN edit is not a semantic defect.  A mutant that does not parse,
    # or that leaves a name undefined, turns every test in the selection red
    # for a reason the catalogue does not describe -- i.e. it reports
    # detection power the suite never demonstrated.  Both are APPLY-FAILED.
    try:
        compile(back, str(dst_mod), "exec")
    except SyntaxError as exc:
        return Staged(False, f"mutant does not parse: line {exc.lineno}: "
                             f"{exc.msg}")
    bad = _new_undefined_names(text, back, dst_mod)
    if bad:
        return Staged(False, f"mutant introduces undefined name(s): {bad}")
    return Staged(True, root=root, modpath=dst_mod)


def _new_undefined_names(original: str, mutated: str, path: Path) -> list[str]:
    """Undefined names present in the mutant and NOT in the original.

    Fails OPEN (returns nothing) when ruff is unavailable: this is a guard
    against a false CAUGHT, not a certification, and a missing linter must not
    stop the census.
    """
    def f821(text: str) -> set[str]:
        try:
            p = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "--select", "F821",
                 "--output-format", "concise", "--no-cache",
                 "--stdin-filename", str(path), "-"],
                input=text, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.SubprocessError):
            return set()
        if p.returncode not in (0, 1):
            return set()
        return {ln.split("F821", 1)[1].strip()
                for ln in p.stdout.splitlines() if "F821" in ln}
    return sorted(f821(mutated) - f821(original))


def verify_shadow(m: Mutation, env: dict, py: str, st: Staged,
                  cwd: Path) -> tuple[bool, str]:
    """Import the module in a subprocess and prove the SCRATCH file loaded.

    The unmutated siblings are symlinks, so ``realpath`` resolves them back
    into the repo; only a real file inside the scratch tree can match.  No
    sentinel is injected into the source, so a source-inspecting gate cannot
    be tripped by the verification itself.
    """
    code = (
        "import importlib, os, sys\n"
        f"m = importlib.import_module({m.dotted!r})\n"
        "got = os.path.realpath(m.__file__)\n"
        f"want = os.path.realpath({str(st.modpath)!r})\n"
        "print('LOADED', got)\n"
        "sys.exit(0 if got == want else 9)\n"
    )
    p = subprocess.run([py, "-c", code], env=env, cwd=str(cwd),
                       capture_output=True, text=True, timeout=900)
    if p.returncode == 0:
        return True, p.stdout.strip()
    return False, (p.stdout + p.stderr).strip()[-1500:]


# =====================================================================
# pytest execution + junit parsing
# =====================================================================

@dataclass
class RunResult:
    ok: bool                     # the run itself completed and produced XML
    rc: int = -1
    seconds: float = 0.0
    outcomes: dict = field(default_factory=dict)   # test id -> pass/fail/skip
    order: list = field(default_factory=list)      # execution order
    detail: str = ""


def _short(classname: str, name: str) -> str:
    mod = classname.rsplit(".", 1)[-1] if classname else "?"
    return f"{mod}::{name}"


def parse_junit(path: Path) -> tuple[dict, list]:
    outcomes, order = {}, []
    tree = ET.parse(path)
    for tc in tree.iter("testcase"):
        tid = _short(tc.get("classname", ""), tc.get("name", ""))
        if tc.find("failure") is not None or tc.find("error") is not None:
            res = "fail"
        elif tc.find("skipped") is not None:
            res = "skip"
        else:
            res = "pass"
        # a duplicate id (setup+call+teardown reports) resolves to the worst
        if tid in outcomes:
            if res == "fail":
                outcomes[tid] = "fail"
            continue
        outcomes[tid] = res
        order.append(tid)
    return outcomes, order


def run_pytest(repo: Path, py: str, env: dict, files: list[str], kexpr: str,
               xml: Path, basetemp: Path, timeout: int) -> RunResult:
    cmd = [py, "-m", "pytest", *files, "-q", "--tb=line",
           "-p", "no:randomly",
           "--basetemp", str(basetemp), "--junit-xml", str(xml)]
    if kexpr:
        cmd += ["-k", kexpr]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, cwd=str(repo), capture_output=True,
                           text=True, timeout=timeout)
        rc, tail = p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return RunResult(False, -9, time.time() - t0,
                         detail=f"TIMEOUT after {timeout}s")
    dt = time.time() - t0
    if not xml.exists():
        return RunResult(False, rc, dt, detail="no junit xml written\n" + tail)
    outcomes, order = parse_junit(xml)
    if not outcomes:
        return RunResult(False, rc, dt,
                         detail="SELECTION EMPTY -- collected 0 tests\n" + tail)
    return RunResult(True, rc, dt, outcomes, order, tail[-800:])


# =====================================================================
# the census
# =====================================================================

@dataclass
class Verdict:
    mid: str
    status: str                 # CAUGHT | CAUGHT-ON-<stage> | MISSED | ...
    stage: str = "-"
    first_gate: str = "-"
    n_new_fail: int = 0
    seconds: float = 0.0
    detail: str = ""


def newly_failing(base: RunResult, mut: RunResult) -> list[str]:
    """Tests that PASSED unmutated and FAIL under the mutant, in run order.

    Requiring a baseline pass is what keeps the suite's known non-code
    failures (TOL-PENDING provisional bounds) from being read as detection.
    """
    return [t for t in mut.order
            if mut.outcomes.get(t) == "fail" and base.outcomes.get(t) == "pass"]


def census(args) -> int:
    repo = Path(args.repo).resolve()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "junit").mkdir(exist_ok=True)
    basetemp = work / "pytest_basetemp"
    basetemp.mkdir(exist_ok=True)
    py = args.python

    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = os.pathsep.join([
        str(repo / "packages" / "core"),
        str(repo / "packages" / "atmosphere"),
        str(repo / "src"),
    ])
    base_env["JAX_PLATFORMS"] = "cpu"
    base_env["JAX_ENABLE_X64"] = "1"
    base_env["PYTHONDONTWRITEBYTECODE"] = "1"

    wanted = [m for m in null_controls() + MUTATIONS
              if not args.only or m.mid in args.only]
    if args.only:
        missing = set(args.only) - {m.mid for m in wanted}
        if missing:
            print(f"ERROR: --only names no such mutation: {sorted(missing)}")
            return 2

    print(f"repo   {repo}")
    print(f"work   {work}")
    print(f"oracle {ORACLE_TREE}")
    print(f"count  {len(wanted)} entries "
          f"({sum(m.expect == 'null' for m in wanted)} null controls)")
    print()

    # ---- stage every mutant first: a stale catalogue is a hard stop -----
    staged: dict[str, Staged] = {}
    for m in wanted:
        st = stage_mutant(repo, work, m)
        staged[m.mid] = st
        print(f"  stage {m.mid:<18} {'OK' if st.ok else 'FAILED: ' + st.reason}")
    bad = [m.mid for m in wanted if not staged[m.mid].ok]
    if bad:
        print(f"\nAPPLY-FAILED: {bad}")
        print("A mutation that cannot be applied is worse than none: the "
              "catalogue is stale against the module it names.  Fix the "
              "anchor before trusting any verdict in this run.")
        if not args.keep_going:
            return 2
    if args.dry_run:
        print("\n--dry-run: anchors verified, no tests run.")
        _write_report(args, [], {}, {}, dry=True)
        return 0 if not bad else 2

    baselines: dict[tuple, RunResult] = {}

    def baseline_for(files, kexpr) -> RunResult:
        key = (tuple(files), kexpr)
        if key not in baselines:
            tag = "base_" + "_".join(Path(f).stem for f in files)
            tag += "_" + (kexpr.replace(" ", "") or "all")
            xml = work / "junit" / (tag[:120] + ".xml")
            print(f"    baseline: {files} -k {kexpr or '(all)'}")
            baselines[key] = run_pytest(repo, py, base_env, files, kexpr,
                                        xml, basetemp, args.timeout)
            r = baselines[key]
            print(f"      -> {'ok' if r.ok else 'RUN-ERROR'} "
                  f"{len(r.outcomes)} tests, "
                  f"{sum(v == 'pass' for v in r.outcomes.values())} pass, "
                  f"{r.seconds:.0f}s")
        return baselines[key]

    verdicts: list[Verdict] = []
    per_gate_caught: dict[str, set] = {}
    per_gate_ran: dict[str, set] = {}

    ladder_all = ["fast", "file", "all"]
    max_stage = ladder_all.index(args.max_stage)
    #: how many mutations have already been escalated to the whole-lane suite.
    #: That stage is ~25 min per mutant, so it is budgeted, not unlimited; a
    #: run that exhausts the budget reports MISSED-AT-FILE, which says "the
    #: ladder was capped here", not "nothing anywhere can see it".
    n_full = [0]

    for m in wanted:
        st = staged[m.mid]
        if not st.ok:
            verdicts.append(Verdict(m.mid, "APPLY-FAILED",
                                    detail=st.reason))
            continue
        env = dict(base_env)
        env["PYTHONPATH"] = str(st.root) + os.pathsep + base_env["PYTHONPATH"]

        okshadow, info = verify_shadow(m, env, py, st, repo)
        if not okshadow:
            verdicts.append(Verdict(m.mid, "SHADOW-FAILED", detail=info))
            print(f"\n{m.mid}: SHADOW-FAILED\n{info}")
            continue

        print(f"\n{m.mid}  {m.title}")
        print(f"    oracle {m.oracle}")

        # A null control is EXPECTED to stay green, so escalating it would
        # buy nothing and cost a whole-lane suite run; it is capped at its
        # own gate file.  Real mutations climb the ladder.
        stages = [("fast", m.fast, m.fast_k)]
        if m.expect != "null":
            stages += [("file", m.file_stage, "")]
            if n_full[0] < args.max_full:
                stages += [("all", ALL_LANE_TESTS, "")]
        v = None
        total_s = 0.0
        last_stage = "fast"
        for si, (sname, files, kexpr) in enumerate(stages):
            if si > max_stage:
                break
            if si > 0 and (tuple(files), kexpr) == (tuple(stages[si - 1][1]),
                                                    stages[si - 1][2]):
                continue                     # same selection, nothing to add
            base = baseline_for(files, kexpr)
            if not base.ok:
                v = Verdict(m.mid, "CONTROL-BAD", sname,
                            detail="baseline run failed: " + base.detail[-600:])
                break
            xml = work / "junit" / f"{m.mid}_{sname}.xml"
            if sname == "all":
                n_full[0] += 1
            res = run_pytest(repo, py, env, files, kexpr, xml, basetemp,
                             args.timeout)
            total_s += res.seconds
            if not res.ok:
                v = Verdict(m.mid, "RUN-ERROR", sname, seconds=total_s,
                            detail=res.detail[-600:])
                break
            nf = newly_failing(base, res)
            for t in res.outcomes:
                if base.outcomes.get(t) == "pass":
                    per_gate_ran.setdefault(t, set()).add(m.mid)
            print(f"    {sname:<5} {len(res.outcomes)} tests, "
                  f"{len(nf)} newly failing, {res.seconds:.0f}s")
            last_stage = sname
            if nf:
                for t in nf:
                    per_gate_caught.setdefault(t, set()).add(m.mid)
                status = "CAUGHT" if sname == "fast" else \
                    f"CAUGHT-ON-{sname.upper()}"
                v = Verdict(m.mid, status, sname, nf[0], len(nf), total_s)
                break
        if v is None:
            # Name the DEEPEST stage actually run, not the nominal "all": a
            # capped ladder that found nothing has not searched everywhere,
            # and reporting otherwise would overstate the miss.
            status = "MISSED" if last_stage == "all" else \
                f"MISSED-AT-{last_stage.upper()}"
            if m.expect == "null":
                status = "MISSED"          # a green null control is the goal
            v = Verdict(m.mid, status, last_stage, seconds=total_s)
        verdicts.append(v)
        print(f"    => {v.status}  first gate: {v.first_gate}")

        # ABORT EARLY on a red null control.  The null controls run first
        # (they are prepended to the work list) precisely so a broken
        # shadowing mechanism costs one selection instead of ten hours: if
        # staging alone turns a gate red, every verdict after it is
        # confounded and there is nothing to learn by continuing.
        if m.expect == "null" and v.status.startswith("CAUGHT"):
            print(f"\nABORT: null control {m.mid} went red on "
                  f"{v.first_gate} -- staging perturbs the suite.")
            break

    _write_report(args, verdicts, per_gate_caught, per_gate_ran)

    # ---- exit code ---------------------------------------------------
    by_id = {v.mid: v for v in verdicts}
    broken = [v.mid for v in verdicts
              if v.status in ("APPLY-FAILED", "SHADOW-FAILED", "RUN-ERROR",
                              "CONTROL-BAD")]
    null_red = [m.mid for m in wanted if m.expect == "null"
                and by_id.get(m.mid) and by_id[m.mid].status.startswith("CAUGHT")]
    missed = [m.mid for m in wanted if m.expect == "caught"
              and by_id.get(m.mid) and by_id[m.mid].status == "MISSED"]
    capped = [m.mid for m in wanted if m.expect == "caught"
              and by_id.get(m.mid)
              and by_id[m.mid].status.startswith("MISSED-AT-")]

    print("\n" + "=" * 62)
    if null_red:
        print(f"INSTRUMENT BROKEN: null control(s) went red: {null_red}")
        print("The staging mechanism itself perturbs the suite -- no "
              "detection verdict in this run is trustworthy.")
        return 2
    if broken:
        print(f"INSTRUMENT INCOMPLETE: {broken}")
        return 2
    if capped:
        print(f"INCONCLUSIVE (escalation budget exhausted): {capped}")
        print("These went green through the module's own gate file but the "
              "whole-lane stage was capped by --max-full, so this run cannot "
              "say whether ANY gate sees them.  Re-run those ids with a "
              "larger --max-full before quoting a coverage verdict.")
        return 2
    if missed:
        print(f"MISSED by every gate: {missed}")
        print("Each is a semantic defect the suite cannot see.  Per the "
              "review's finding 4, the gates that ran under these and caught "
              "nothing are candidates for reclassification as MONITORS.")
        return 1
    print("every expected-observable mutation was caught")
    return 0


def _write_report(args, verdicts, per_gate_caught, per_gate_ran, dry=False):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    by_id = {v.mid: v for v in verdicts}
    lines = []
    lines.append("# FV3 duo-grid JAX lane — mutation-sensitivity census")
    lines.append("")
    lines.append("Answers `glm_review_strategy_2026-08-14.md` finding 4 (BLOCKER): "
             "the suite had **zero demonstrated detection power**. This is a "
             "gate ON the gates — it fails when a semantic defect goes "
             "unnoticed.")
    lines.append("")
    lines.append(f"* repo (pinned): `{args.repo}`")
    lines.append(f"* oracle (pinned): `{ORACLE_TREE}`")
    lines.append(f"* work dir: `{args.work}`")
    lines.append("* mutants shadow the real modules via a symlink-farm scratch "
             "package prepended to `PYTHONPATH`; **the worktree is never "
             "modified**, so restore is a no-op and a crashed run cannot "
             "leave mutated code behind.")
    lines.append("* a mutation counts as CAUGHT only when a test that **passed "
             "unmutated** fails under the mutant (the suite carries known "
             "non-code failures; without the baseline they would read as "
             "detection).")
    if dry:
        lines.append("")
        lines.append("**--dry-run: anchors verified, no tests were run.**")
    lines.append("")
    lines.append("Verdict vocabulary — these are not synonyms:")
    lines.append("")
    lines.append("| verdict | meaning |")
    lines.append("|---|---|")
    lines.append("| `CAUGHT` | the smallest declared selection went red |")
    lines.append("| `CAUGHT-ON-FILE` / `CAUGHT-ON-ALL` | only a wider "
                 "selection saw it — a finding about **gate targeting**, not "
                 "a pass |")
    lines.append("| `MISSED` | the whole ladder stayed green: a semantic "
                 "defect **no gate sees** |")
    lines.append("| `MISSED-AT-FILE` | green so far, but the whole-lane stage "
                 "was budget-capped — **inconclusive**, re-run with a larger "
                 "`--max-full` |")
    lines.append("| `APPLY-FAILED` / `SHADOW-FAILED` / `CONTROL-BAD` | the "
                 "instrument, not the suite — no verdict |")
    lines.append("")
    lines.append("## Catalogue and result")
    lines.append("")
    lines.append("| id | mutation | oracle line violated | failure-mode class | "
             "verdict | stage | first catching gate |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in MUTATIONS:
        v = by_id.get(m.mid)
        lines.append(f"| {m.mid} | {m.title} | `{m.oracle}` | {m.mode} | "
                 f"{v.status if v else '(not run)'} | "
                 f"{v.stage if v else '-'} | "
                 f"`{v.first_gate if v else '-'}` |")
    lines.append("")
    lines.append("### Null controls (the instrument's own control)")
    lines.append("")
    lines.append("| id | verdict | meaning |")
    lines.append("|---|---|---|")
    for m in null_controls():
        v = by_id.get(m.mid)
        s = v.status if v else "(not run)"
        if s == "MISSED":
            mean = "staging is inert — results trustworthy"
        elif s.startswith("CAUGHT"):
            mean = "STAGING PERTURBS THE SUITE — every verdict confounded"
        else:
            mean = "no result — the control itself did not complete"
        lines.append(f"| {m.mid} | {s} | {mean} |")
    lines.append("")
    lines.append("### Why each mutation is a real defect (oracle text)")
    lines.append("")
    for m in MUTATIONS:
        lines.append(f"* **{m.mid}** — `{m.oracle}`: {m.why}.")
        if m.note:
            lines.append(f"  > {m.note}")
    lines.append("")
    lines.append("## Gates that caught nothing — reclassification candidates")
    lines.append("")
    lines.append("Only gates that actually RAN under at least one mutation are "
             "listed: a gate outside every selection had no opportunity, and "
             "listing it would be a measurement of the selection, not of the "
             "gate.")
    lines.append("")
    lines.append("| gate | mutations it ran under | mutations it caught |")
    lines.append("|---|---:|---:|")
    zero = [(g, len(ms)) for g, ms in sorted(per_gate_ran.items())
            if not per_gate_caught.get(g)]
    for g, n in zero:
        lines.append(f"| `{g}` | {n} | 0 |")
    if not zero:
        lines.append("| _(none — every gate that ran caught at least one)_ | | |")
    lines.append("")
    lines.append("## Gates with demonstrated detection power")
    lines.append("")
    lines.append("| gate | mutations caught |")
    lines.append("|---|---|")
    for g, ms in sorted(per_gate_caught.items(),
                        key=lambda kv: (-len(kv[1]), kv[0])):
        lines.append(f"| `{g}` | {', '.join(sorted(ms))} |")
    if not per_gate_caught:
        lines.append("| _(none)_ | |")
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    lines.append("| id | seconds | detail |")
    lines.append("|---|---:|---|")
    for v in verdicts:
        if v.detail or v.status not in ("CAUGHT", "MISSED"):
            d = v.detail.replace("\n", " ⏎ ")[:300]
            lines.append(f"| {v.mid} | {v.seconds:.0f} | {d} |")
    lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nreport written: {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mutation-sensitivity census for the FV3 JAX lane.")
    p.add_argument("--repo", required=True,
                   help="the PINNED worktree to census (never modified)")
    p.add_argument("--work", required=True,
                   help="scratch dir for mutants, junit xml and basetemp "
                        "(must be on a burg path: SLURM's TMPDIR=/local is "
                        "reaped mid-job)")
    p.add_argument("--out", required=True, help="markdown matrix output path")
    p.add_argument("--python", default=sys.executable,
                   help="interpreter used for the pytest subprocesses")
    p.add_argument("--only", default="",
                   type=lambda s: [x for x in s.split(",") if x],
                   help="comma-separated mutation ids to run")
    p.add_argument("--max-stage", default="all",
                   choices=["fast", "file", "all"],
                   help="cap the escalation ladder")
    p.add_argument("--max-full", type=int, default=4,
                   help="how many mutations may escalate to the whole-lane "
                        "suite (that stage costs ~25 min per mutant; beyond "
                        "the budget a green mutation is reported "
                        "MISSED-AT-FILE = inconclusive, never as a proven "
                        "coverage hole)")
    p.add_argument("--timeout", type=int, default=5400,
                   help="per-pytest-run timeout, seconds")
    p.add_argument("--dry-run", action="store_true",
                   help="stage and verify anchors only; run no tests")
    p.add_argument("--keep-going", action="store_true",
                   help="continue after an APPLY-FAILED entry")
    return p


def main(argv=None) -> int:
    return census(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
