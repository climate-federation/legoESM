# GRIDLADDER-VMIX lane — round 19 (vmix_r19): L2 A33 and face operands

Status: **DIRECT OPERAND DEBT CLOSED; CERTIFICATION PREDICTION REFUTED; L2
REMAINS DEBT / NOT CERTIFIED** (2026-10-19).

Round 19 made the two instructed operands exact and L2-only.  It did not make
the two labels pass the D127 trajectory bar.  The preregistered P19-4 claim is
therefore retracted here as a measured refutation, not softened into a
certification.

## Source transcription

The compiled oracle adds the U and V diffusivities as two bracketed pairs,
materializes each normalized sum, and then evaluates the two slope-square
terms (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:782-794`).  L2's selected
path now preserves those assignments with `nemo_source_round`; the shared
default remains the old graph
(`packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:2360-2385`,
`packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:2463-2487`).
The synthetic fp64 test traces both pair sums and the complete A33 expression
under JIT against an independent NumPy transcription; its left-fold plant is
non-identical
(`tests/ocean/unit/test_gm_redi_latlon_cgrid.py:100-152`).

The compiled stage stores the N+1/2 U/V ratios independently
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:205-217`) and calls that level
Kmm in stage 3 (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:229`).  L2
passes those stored native ratios to the live face builder
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:10006-10013`).
The builder uses them off-fold and retains its existing fold-aware last native
U/V row (`packages/ocean/legoesm/ocean/vertical.py:864-907`).

Both switches default false
(`packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:480-493`) and
the exact L2 selector requires them true
(`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:4525-4564`).
The shared SMT-3 constructor therefore remains unchanged unless its caller
explicitly selects either switch
(`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2428-2482`).

## Direct value gates

The jitted production trace against the admitted round-18 operator record
measures:

- `ah_wslp2`: **0 / 799,200 unequal**; removing source ordering restores
  **60,720 unequal**;
- `e3u_live`: **0 / 799,200 unequal**, versus **36** when the stored ratio is
  rederived from mean SSH;
- `e3v_live`: **0 / 799,200 unequal**, versus **20** under that plant;
- the retained pivot row is exact, including the independently counted 1,319
  wet V faces from the round-18 review evidence.

The committed gate and its fixed plants are
`tests/ocean/fidelity/test_nemo_testcase_l2_orca2iso.py:414-518`.  The copied
review evidence and checksums are in
`gridladder_vmix_rounds/round19/review_round18/`.

## Certification refutation

The independent kt 1..10 rescore has **33 / 68** surface rows and **19 / 36**
full-depth summaries AT-BAR.  The given-NEMO-entry label has **64 / 68**
surface rows and **19 / 20** full-depth summaries AT-BAR.  Its remaining
surface DEBT rows are the second-step kt-2 SSH/V and kt-4/kt-8 SSH rows.  Thus
P19-4 is refuted and L2 is not certified.

The first step still has exactly two T and two S surface cells unequal, each
with maximum absolute difference `7.105427357601002e-15`.  The post-refutation
discriminator was committed before its runs.  Disabling only source-ordered
A33 leaves that four-cell tuple unchanged; disabling only stored face ratios
also leaves it unchanged.  The exact direct operands are therefore real
fidelity fixes but are not the owner of the surviving final-update residual.
The admitted record is sufficient to prove their exactness and this
non-ownership; it does not identify the next downstream owner.

Evidence:
`gridladder_vmix_rounds/round19/score/l2_independent.json`,
`l2_given_entry.json`, `without_a33.json`, and `without_faces.json`.

## Unchanged-card controls

Normalized jaxpr hashes are exactly unchanged before/after for G5, V1-V5,
L1, full ORCA2, DINO, GYRE, VORTEX, SMT-1..4, rung 0, and OMT-0..4.  The
previously ungated cards now have explicit before/after evidence under
`gridladder_vmix_rounds/round19/purity_before/` and `purity_after/`.

GYRE's ten-step comparison passes all **70** compared rows with zero ULP
worsening.  Its residual archive is byte-identical to round 18 at SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
Evidence: `gridladder_vmix_rounds/round19/gyre_regression/`.

No NEMO executable was built or run.  No new record is needed to support the
claims made in this receipt.

## Landing gates

- explicit round-19 receipt citation gate: **PASS**, 12 citations, zero
  unmapped, zero failures, zero map-audit failures;
- default ORCA2 landing list: **145 passed in 1109.39 s**;
- private-import ratchet within that list: **6 passed**;
- focused selector/synthetic tests: **3 passed in 9.32 s**;
- jitted production operand gate: **1 passed in 154.51 s**.

independent review: operator (Claude)

codex review: operator

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/vmix_r19-1789813485
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: DIRECT L2 A33 AND FACE OPERANDS EXACT; P19-4 REFUTED; L2 SCORED, DEBT, NOT CERTIFIED — independent 33/68 surface + 19/36 depth rows AT-BAR; given-entry 64/68 + 19/20
