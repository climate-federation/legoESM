# DINO NEMO-faithful port — program summary & achievements

*Checkpoint written at the user's request for a pause + fresh-start review. Branch
`feat/nemo-dino-topo-bridge` (PR #1137). Honest status — claims are what was measured/
committed, open items are flagged as open.*

## Goal
Build a *verifiable, differentiable* NEMO oracle in legoESM by matching NEMO 5.0.2's DINO
experiment (Kamm et al. 2025, 1° R1) **term by term, exactly** — not an approximate
reconstruction. The oracle (NEMO) is exposed as source precisely so we trace and match the
complete per-timestep call chain, then measure whether the two cores' solutions agree.

## Methodology (the discipline that made it work)
- **Complete wiring diagram** of NEMO's `stp_MLF` timestep (`docs/ocean/fidelity/dino_wiring_diagram.md`),
  node by node, each mapped to its legoESM counterpart with an honest VERIFIED/MISMATCH status.
- **Build every piece exactly as NEMO — whether or not it "seems to matter."** Repeatedly, a
  piece dismissed as minor turned out to be the real lever (see Lessons).
- **Controlled comparisons**: change ONE variable, hold the eval protocol byte-identical to
  NEMO's `RUN_TRAJ` 180-day mean (÷`vovvle3t` for the no-XIOS output artifact).
- **No fabrication, no non-NEMO stabilizers**: subagents refused ~4× to report a 180-day
  number when a run would NaN, and refused to ship non-NEMO damping knobs.
- **Mandatory adversarial review** (physics-validator + code-reviewer) on every numerics
  change; ~5 confidently-reported "mismatches" were refuted by direct computation.

## Headline achievements

### 1. The BSF over-strength — SOLVED
The central diff chased all program: legoESM's barotropic streamfunction was **2.6× too
strong** (±106 Sv vs NEMO ±40). Cause, found by building the vertical coordinate exactly:
legoESM ran effectively **pure z-star** (all levels stretched, no dry bottom cells); NEMO is
**`ln_zco` full-step** (fixed levels, staircase dry bottom cells). The staircase provides
**bottom form stress** that steers the barotropic f/H flow. Wiring NEMO's full-step-z into the
bridge (`b51bf1b65`, reviewed, byte-identical default) drops **BSF 2.62× → 0.81×** — into
NEMO's band. SST corr 0.995, T@300m 0.989, SSH 0.993. *Every prior matched piece had moved
BSF ≤1.5%; the vertical coordinate was the lever.*

### 2. The deep-equatorial jet — ROOT CAUSE FOUND (fix in flight)
A spurious deep zonal jet at the equatorial western boundary (deep KE ~600× NEMO's) that
survived every dynamics match. Diagnosed (measured, controlled): **legoESM's DINO setup
imposes a land wall at column j=0 (outside the ACC channel, including the equator); NEMO is
zonally re-entrant** (`ln_Iperio=.true.`, 48/48 wet everywhere, no walls). The free-slip
equatorial wall traps a cold/dense grid-scale front (4.5°C vs NEMO's uniform 7.44°C) → a zonal
PGF unbalanced by Coriolis at f→0 → the jet. Controlled proof: opening the wall cuts the jet
3.4× and warms the western cells to NEMO's 7.4°C; GM-off ≡ base and masked-cell-fill ≡ base
(both ruled out). **My earlier "irreducible f→0 core amplification" conclusion was wrong** —
it is a concrete, fixable geometry mismatch (= the audit-#10 seam). Faithful re-entrant fix
building now.

## What was built & committed (this program)
- `dff50624f` — **nodes 13+19**: NEMO-faithful **leapfrog + Robert-Asselin integrator** +
  **EEN Coriolis-in-RHS** (`(f+ζ)/e3f` 12-pt triad). 3 time levels, 2Δt update, Euler
  first-step, `Nbb`-level `dyn_ldf(Kbb)` lateral diffusion. Stable dt=1350, reviewed SHIP.
- `ebb46c259` — **node 16**: **live in-substep EEN barotropic Coriolis** (`dyn_cor_2D`).
- `624798552` — **residual #1**: **leapfrog-consistent barotropic solver** (Nbb seed +
  `nn_bt_flt=2` AM4 dissipation). Full-step-independent leapfrog stable dt=2700, 180 days.
- `629346b1d` — **residual #2**: **thickness-weighted tracer Asselin filter** (`tra_atf_qco_lf`),
  content-conserving to 1e-13.
- `b51bf1b65` — **full-step-z vertical coordinate** (the BSF lever, above).
- `c1e3543a9` — **node 14**: faithful `dyn_ldf_lev_lap` div-curl viscosity (embedded
  `ahmt`/`ahmf`); **PGF full-step audit** confirms adcroft ≡ `hpg_zco` for full-step.
- Earlier in-session: `nemo_dino_kamm` complete card, EEN GM operator, MLD N²-integral, TKE
  Prandtl, etc. (see wiring diagram + git log). Doc/scoreboard commits `df2fcaae1`, `0b2606df7`.

## What was matched or ruled out (honestly, by controlled test)
VERIFIED-MATCH: forcing (byte), Hollingsworth KE-grad, EEN vorticity thickness, EVD-on-
momentum, bottom drag coeffs, S-EOS coeffs, `bn2`, `tra_zdf`+MSC K33, `dyn_zdf` avm, isoneutral
slopes/clip/ramp/Shapiro, barotropic split-explicit coupling (no Coriolis double-count),
PGF-at-rest, vertical mom advection. **Ruled out as the BSF/jet driver** (each measured):
the time integrator (leapfrog ≡ forward-Euler), the barotropic Coriolis (EEN ≡ avg), the
tracer filter form, GM/eiv, node-14 viscosity (climate-inert — grid is true Mercator so
`max(e1,e2)`≈`A_h·cosφ`), the PGF, and the exact bathymetry depth field (BSF 2.65→2.61×).

## Key methodological lessons
1. **A systematic diff between two cores proves un-matched numerics** — never "irreducible"
   until everything is bit-matched. Twice I concluded "irreducible/core" and was wrong (the
   integrator, then the jet); both had concrete causes.
2. **Build it exactly, whether or not it seems to matter.** The vertical coordinate (BSF) and
   the equatorial wall (jet) were both the *real* levers; pieces I'd have deferred as "minor."
3. **Each faithful piece exposes the next** — the neutral leapfrog kept unmasking modes that
   forward-Euler's numerical damping had hidden (barotropic null mode → live-EEN Coriolis →
   leapfrog barotropic coupling → high-lat staircase damping).
4. **Verify agent claims by direct computation.** ~5 confident "mismatches" were refuted;
   ~2 "library-default vs recipe" traps caught.

## What remains open
- **Deep-eq jet re-entrant fix** — in flight (the fix for root-cause #2 above).
- **Full-step + leapfrog stability** — the neutral leapfrog blows up at the high-lat staircase
  edges (a genuine dycore-hardening problem, NOT closable by faithful viscosity/PGF — those were
  disproven). Orthogonal to BSF/jet (integrator is ruled out, so full-step+FE is a valid
  comparison). Open.
- **Node 22** — GM/eiv bolus as advective-through-FCT (NEMO form) vs legoESM skew-flux (GM-off
  ≡ base, so not a jet driver, but still an un-matched form per the mandate).
- **Initial condition** bit-identity (+0.08°C), the `e1/e2` metric in the AL81 conservation.

## Scoreboard (180-day controlled means vs NEMO RUN_TRAJ)
| config | BSF ratio | SST corr | T@300m corr | SSH corr | deep-eq KE ratio |
|---|---|---|---|---|---|
| z-star + forward-Euler (baseline) | 2.62× | 0.995 | 0.987 | 0.991 | ~0.5 |
| z-star + faithful leapfrog | 2.65× | 0.994 | 0.990 | 0.992 | ~0.68 |
| **full-step-z + forward-Euler (best)** | **0.81×** | **0.995** | **0.989** | **0.993** | 0.68* |
| NEMO | 1.0 (±40 Sv) | — | — | — | 0.0011 |

*deep-eq jet = the equatorial wall (root cause found; re-entrant fix in flight).
