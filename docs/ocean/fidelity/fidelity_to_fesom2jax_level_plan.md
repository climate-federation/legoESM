# Plan: DINO/NEMO fidelity to FESOM2-JAX level

**Goal.** Bring legoESM-vs-NEMO agreement on DINO to the standard demonstrated by
FESOM2-JAX-vs-Fortran (arXiv:2608.01546): climate-statistics differences ~two orders below
the model-vs-truth bias, drift residuals fluctuating about zero, and any one-signed residual
small, named, and tracked. Their SST number: JAX−Fortran RMS 0.004 °C against a 0.61 °C
model−obs bias. Our current state: ACC ratio 0.78 at matched years, deep density 66–70%,
deep-box heat excess +1.72 W/m² — with every operator verified near-clean at identical state
(gate: 18 AT BAR | 5 CEILING | 25 DEBT | 4 UNMEASURED | 1 WAIVED).

**The diagnosis this plan rests on** (campaign + paper, converging): per-operator exactness
is nearly exhausted as a lever. FESOM2-JAX's method statement fits our situation verbatim:
*"a drift that appears only over climate timescales cannot originate inside a kernel and must
instead arise where two kernels are joined. Checking which of the conservation budgets fails
to close then locates the defect."* Their one climate-scale bug (freshwater leak in the z*
flux normalization, visible only as a decadal salinity drift) had exactly the fingerprint of
our deep warm bias: a monotone, one-signed tracer-integral drift.

Standing rules unchanged: THE RULE (transcribe with citations; exact-match failure = escalate),
fp64 explicit everywhere, time-level registry, Rule 1e reconciliation, bar constants never
relaxed, adversarial review on production edits, one canonical committed probe per row.

---

## Phase 0 — Standards & instruments (adopt theirs; ~3 lanes)

**0.1 Conservation-closure audit — FIRST DEPLOYMENT, decisive either way.**
For Q ∈ {volume, salt, heat}: global closed-domain residual per step,
`d(∫Q dV)/dt − Σ(applied boundary fluxes)`, fp64, on the y20 twin trajectory (short rerun
with the accumulator extended to global integrals + applied-flux bookkeeping). Correct model
⇒ residual ~1e-15 relative. **Salt first** — DINO has virtual salt flux + z*, the exact
ingredients of the FESOM2-JAX bug. Then per-join sub-budgets for any Q that fails: Asselin/ATF
filter alone, z* layer-thickness commit (e3-update vs tracer-update consistency), barotropic
transport correction, FCT+bolus fold-in, implicit vertical solve, EVD.
*Either outcome wins:* residual found → the defect's address, sharper than any NEMO
comparison; all close to roundoff → the hidden-source hypothesis dies formally, the bias is
proven pure redistribution, Phase 3 becomes the only game.

**0.2 Tolerance-by-arithmetic-class (their a-priori standard, replacing post-hoc ceilings).**
Classify every gate row by what its arithmetic can achieve:
- POINTWISE / NEIGHBOUR-STENCIL → 1e-15 relative (rounding alone)
- ACCUMULATING (flux assembly, sums, depth integrals) → 1e-12 (reassociation)
- CONDITIONED (ratios with →0 denominators, branch flips) → mechanism-proven CEILING
This is the same physics as our CEILING category, assigned by design instead of by autopsy.
**Needs Dhruv's sign-off**: it amends the single literal bar (1e-9 per-element) to per-class
bars. Proposal: keep 1e-9 as the *default*, allow a row's class to set 1e-15 (stricter!) or
1e-12, with the class recorded in the gate and auditable. Many "DEBT" rows at 1e-12-class
residuals would reclassify AT BAR(class); several 1e-15-class rows currently "AT BAR" at 1e-9
would tighten.

**0.3 Canonical-probe convention (fixes the sh2 mess).** One committed probe per row, locked
population/state/input conventions, registered in PROVENANCE_SCRIPT (already machine-checked);
a row without a canonical probe is UNMEASURED regardless of history. First application: the
sh2 probe (three probes currently disagree 0.33/0.93/0.995 on corr).

## Phase 1 — Close the joins (~3–5 lanes)

**1.1 Multi-step replay tests** — the joins single-step gates structurally cannot see: MLF
leapfrog history threading, TKE/e carry, e3/ssh commit ordering, CG/solver warm starts.
N-step twin replay vs NEMO per-step restarts (we have `nn_stock=1` machinery) with the state
re-injected at step k — a divergence that appears only at k>1 is a threading defect. Ship as
CI tests (their pattern), not one-off probes.

**1.2 Fix whatever 0.1/1.1 finds** at the join, re-run the twin, re-measure the deep-box
budget and the y20→y21 climate deltas. THE RULE applies: transcribed fixes only.

**1.3 Instrument lock-in**: budget-closure residuals become standing outputs of every twin
run (like STABLE/finite flags), so a future leak cannot hide.

## Phase 2 — Whole-trajectory acceptance harness (their §4, adapted; ~3 lanes)

**2.1 NEMO internal-variability floor (the missing control).** Micro-perturbed ensemble
(1-ulp restart perturbations, 3–5 members, 10 years each, ~1.5 h/member CPU): NEMO-vs-NEMO
member spread defines the noise floor for every climate statistic. Without it, no claim that
a residual "matters" is meaningful. (This is what our two null acceptance runs lacked.)

**2.2 The acceptance suite** (computed identically on both models, matched years, fp64):
time-mean SST/SSS bias maps, zonal-mean T/S sections, global integrals drift curves
(volume-mean T, S, heat content), ACC transport, overturning, deep density-class census
(our instrument — their suite lacks one; ours is better targeted at the known bias), EKE.
**Acceptance bar per statistic: |lego−NEMO| ≤ NEMO ensemble spread** (their criterion made
precise). Staged targets: within 5× spread (now?), 2×, then 1×.

**2.3 Standing twin protocol**: the 1-yr budget-attributed twin from the y20 restart becomes
the regression harness — any model change re-runs it; deep-box delta and closure residuals
are its scoreboard.

## Phase 3 — Structural closure of the remaining gap (open-ended; ranked)

With joins clean and the noise floor known, the residual is structural. Ranked by leverage:
1. **The Rule-8 ladder paradox** (true 3-D e3t tracks NEMO *worse*, 7.4 Sv) — now a
   noise-floor question first: is 7.4 Sv above the ensemble spread? If yes, find the
   compensating error (Pierre flagged this as highest-value; agreed).
2. **sh2 canonical probe → wire-or-not** the face-avm option on the card (measured 2×
   magnitude improvement pending the probe reconciliation).
3. **traadv_fct local family** (~1e-4, limiter branch flips) and **dyn_vor diffuse ~1e-4** —
   re-judged under 0.2's class bars; chase only if above class.
4. **Architectural-gap ports** (tra_zdf z*-volume-form solve, dyn_zdf drag fold, wzv qco
   closure): implement-or-accept decisions once 2.2 says whether they matter above the floor.
5. **Deep-water pathway metrics** (density-class census, deep contrast at matched years) as
   the bias-specific acceptance criteria.

## Phase 4 — Lock-in (their reproducibility apparatus; ~2 lanes)

- **Bit-for-bit CI trajectory pin** for the kamm card, single device (their gate); multi-GPU
  to reassociation tolerance only.
- **fp64-by-design** on all oracle cards (policy set in the card, not the harness).
- **Oracle provenance**: version-control `cfgs/DINO/MY_SRC`, archive exe+source beside every
  reference run, keep annual restarts (the practices that saved us / bit us).
- **Gate end-state**: every row AT BAR(class) | CEILING(proven) | WAIVED(human); DEBT → 0.

---

## Success criteria ("their level", translated to DINO)

1. All three global budgets close to roundoff, standing check on every run.
2. y1–y20 matched-year statistics: |lego−NEMO| ≤ NEMO ensemble spread for SST/SSS maps,
   T/S sections, drift integrals.
3. ACC transport and deep density census within the ensemble spread at matched years
   (this is the 21% → noise-level goal; staged 5×→2×→1×).
4. Any remaining one-signed residual: small, named, tracked (their salinity-drift precedent).

## Deployment shape

One lane at a time, existing token-economy rules (sonnet default, scope caps, tool caps,
completion-only reporting). Order: 0.1 → 0.3(sh2) → 1.1 → [0.2 decision w/ Dhruv] → 2.1 →
2.2 → then Phase 3 ranked. Every lane updates the gate/digest; tally + closure residuals are
the public progress metrics on #1455.
