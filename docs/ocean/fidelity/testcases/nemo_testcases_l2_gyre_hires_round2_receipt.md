# NEMO testcase L2 GYRE high-resolution round 2 receipt

**Date:** 2026-10-13
**Incoming tip:** `7f7d2a207`
**Implementation commit:** `bc9ea7d53`

## Verdict and preregistration

Round 2 implements the user-selected quarter-degree `r4_smag` deck: the r4
grid, 3600 s clock, 25 km isoneutral Laplacian tracer mixing, and NEMO's
mode-32 level Smagorinsky bilaplacian momentum operator with `rn_csmc=3.5` and
`rn_minfac=rn_maxfac=1`.  Unsupported momentum operators and hybrids raise.
The shipped one-degree executable card is unchanged.

The `r4_smag` ten-step ladder is preregistered before acquisition.  Its bar is
zero unequal active float64 values for T, S, U, V and SSH at every before-state
entry kt=1..10 under CPU/fp64/libm.  Each field has an exactly one-ULP plant at
kt=1 and every plant must fire.  Missing records are `ACQUISITION_NEEDED`; no
other oracle may be substituted.

The long GPU trajectory is not preregistered to match NEMO.  The first
transcendental in a step is the analytic forcing's seasonal cosine; the
certified CPU ladder calls scalar libm, while the GPU driver deliberately uses
native XLA transcendentals.  Those last-bit differences compound in the
eddy-active `r4_smag` regime.  The future comparison is therefore pattern and
climate statistics over identical, separately preregistered windows: mean
fields, 31-day-high-pass surface EKE and EKE/KE, jet position and spectra.

This supersedes and retracts round 1's statement that the scaled Laplacian r4
deck was eddying.  The operator measured that deck as laminar.  The selected
Smagorinsky reference is the five-year NEMO run at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_hires/gyre_r4_5yr_smag/`;
the long-run statistical windows and bars remain OPEN below.

## NEMO program transcribed

The compiled mode-32 initializer allocates the deformation arrays and defines
the T/F grid scales as `(2 e1 e2/(e1+e2))^2` in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90:368-380`.  Its executed update
forms the T- and F-point deformation invariants, applies the velocity lower
bound and timestep upper bound, including the bilaplacian `16/9` factor, and
converts the limited Laplacian coefficient to `sqrt(esq*ahm/8)` in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90:462-528`.

The WS-RK3 step computes the coefficient from `Nbb` immediately after vertical
and lateral tracer physics and before the external mode in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90:159-183`.  legoESM already freezes
the momentum-LDF operand at the step entry for every RK stage, so this round
adds no new time-level selector.

NEMO identifies the level bilaplacian as two successive Laplacians in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynldf_lev.f90:218-229`.  The first pass uses
the e3-weighted div/curl form and stores face Laplacians in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynldf_lev.f90:271-290`; those first-pass
values are computed over the extended halo range and consumed directly by the
second pass, which repeats that form and enters the RHS with the bilaplacian
minus sign in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynldf_lev.f90:311-330`.  The port reuses the
existing source-literal e3-weighted NEMO Laplacian twice and applies that minus
sign; it does not route through legoESM's distinct MOM stress-tensor
Smagorinsky implementation.

## Card, acquisition, and shipped regression

`GyreResolvedNamelist` now records `ln_dynldf_blp`, mode 32, `rn_csmc`, and the
two limiting factors.  `GYRE_R4_SMAG_NAMELIST` requires `rn_Uv/rn_Lv` to be
absent, sets `A_h=B_h=0`, selects `C_smag=3.5`, NEMO e3 div/curl, and the new
`nemo_smag_blp` coefficient source.  The shipped and scaled-Laplacian decks
keep their old configuration and digest.

The acquisition selector now accepts `r4_smag`, applies the exact `MODE=mom`
operator-launcher replacement, derives target `GYRE_OMIP_L2_P3_R4_SMAG` and
round-2 evidence paths from the selector, verifies every mode-32 value and the
absence of `rn_Uv/rn_Lv`, and retains the pinned P3-writer hash.  Clean-tree
preflight passes for `shipped`, `r4`, and `r4_smag` at `bc9ea7d53`.
The operator acquisition command is:

```bash
EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round2/oracle_gyre_r4_smag_stage RANKS=8 /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-NmcnEbjP/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires/run.sh --deck r4_smag --run
```

The shipped ten-step oracle-relative regression against round 1's original
prechange artifact passes: 70 certified rows, zero moved cells, zero worsening
ULPs, and the same first-over-bar step 3.  The certified-card fingerprint gate
also passes.  Thus this round adds a new deck without moving the shipped card.

## Resolution rung and scan diagnosis

The operator-provided scaled-Laplacian r4 records close the first resolution
rung as **DEBT**, not AT-BAR.  kt=1 is bit-exact in all five fields.  kt=2 is
the first failure (T max `1.0658141036401503e-14`, S max
`2.1316282072803006e-14`); by kt=10 the maxima are T
`2.2064320148729166e-08`, S `1.3992078606861469e-09`, U
`7.613925046001668e-10`, V `1.990153063180511e-09`, and SSH
`1.9252387011003957e-10`.  All five one-ULP controls fire.

The two-day scan/per-step failure is a **COMPOSITION** difference.  The traced
step clock and both seasonal phase cosines are bit-identical to the per-step
calculation.  XLA composition first changes the separately returned positive
`tau_y` by `6.421189959886524e-18`; after one scanned step the six reported
prognostic fields remain exact, and after the second scanned step (model step
3) `ssh_m` is the first field in the driver's canonical ordering to differ.
The diagnostic records both prefixes independently rather than inferring the
first result from the second.
At step 3 all differences are still below the unchanged `1e-12` control bar;
after two days surface T reaches the measured `1.8106049992638873e-10` maximum.
No transcription correction is applicable and the long climate run proceeds
under its non-trajectory claim.

## Long-run output convention and controls

The GPU-1 wrapper now selects `r4_smag` for 1095 days.  The compiled-day scan
still evaluates seasonal forcing from the live step index inside the scan.
The CPU certified mode remains a per-step libm loop.  Daily files carry
end-of-day SSH, surface U/V/T and prognostic barotropic U/V plus their daily
means.  Surface-temperature means are now `mean(T*e3t)/mean(e3t)`, matching
the NEMO IOIPSL convention; other fields use arithmetic means.  The manifest
states the weighting, policy, backend, exact resolved namelist, commit, and
per-file hashes.

Direct tiny-grid controls prove that the new coefficient has exact zero flow
values with finite reverse-mode gradients, accepts a traced dynamic timestep,
and that its tendency is bitwise the negative of two calls to the existing
e3-weighted NEMO Laplacian.  An independent nonzero NumPy transcription checks
the deformation, lower-limit, and upper-limit regimes; an omitted lower-limit
factor is a planted failure.  Exact-isotropy rejection is also planted, since
the selected GYRE geometry is isotropic and the implemented F-point program is
not a general anisotropic-grid operator.  A one-step full GYRE dispatch on CPU
is finite.

## Tests and evidence

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round2/`.

- `r4_ladder.json`: DEBT; kt=1 exact, first debt kt=2, all plants fired.
- `shipped_trajectory_comparison.json`: PASS; 70 rows and zero field movement.
- `scan_parity.json`: the final-SHA, fp64/libm CPU report with independently
  scored model-step-2 and model-step-3 prefixes.
- acquisition preflight: PASS for all three selectors at the implementation
  commit and pinned writer SHA-256.
- the two-day CPU parity control remains refuted at the unchanged 1e-12 bar;
  its committed diagnosis is COMPOSITION, first reported field `ssh_m`, step 3.
- the final focused non-slow suite passes 71 tests (2 slow tests deselected), including
  unit, acquisition, driver, ladder, and certified-card fingerprint coverage.
- both slow controls pass: the complete fp64/libm GYRE step is finite and the
  two-day parity diagnostic identifies the measured composition debt.  The
  citation gate passes with six compiled-source citations mapped and all nine
  of its self-test controls firing.

Two independent adversarial reviews were dispositioned before landing.  The
physics review found the missing anisotropic-grid refusal; the tooling review
found reciprocal-operand/source-grouping debt, the zero-only coefficient
control, the inferred prefix result, and incorrect halo-exchange wording.  All
five findings are fixed above and covered by controls; both reviewers found no
selected-deck, acquisition, weighting, forcing-placement, or shipped-isolation
defect.

## Inherited pins / OPEN

Inherited unchanged pins are the 31-record MI96 ladder (30 wet levels plus
dummy bottom record), closed walls/free slip, TEOS-10, FCT2, along-isopycnal
Laplacian tracers (`rn_Ud=0.02`, `rn_Ld=25 km`, mode 0), vector/C2/ENE
momentum, SCO pressure gradient, split-explicit AB3/AM4 with automatic 50
substeps, nonlinear implicit drag, TKE plus EVD, two-band shortwave, seasonal
analytic forcing, qco/vco-1d/RK3, and every existing no-fixer selector.

OPEN:

1. operator acquisition of the `r4_smag` kt=1..10 P3 records and execution of
   the preregistered exact ladder;
2. the 1095-day legoESM GPU-1 run; the five-year NEMO reference already exists;
3. preregistration of a common spin-up exclusion, identical windows, and exact
   bars for mean fields, 31-day-high-pass EKE/EKE fraction, jet position and
   spectra before climate scoring;
4. the later 1/12-degree deck and runs;
5. NEMO daily T/S stitching must place all eight tiles from
   `DOMAIN_position_first`/`DOMAIN_size_global` and divide thickness-weighted
   T/S by same-file `vovvle3t`; snapshot comparisons use restarts/stage entries;
6. the r4 ten-step resolution rung is measured DEBT from kt=2, and the long
   scan composition debt is measured from step 3; neither may be called a
   trajectory match.
