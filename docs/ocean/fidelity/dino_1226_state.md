## RESUMED 2026-08-02 — bn2 live-e3w divisor coverage sweep, CLOSED (one latent gate fixed, inert on DINO)

Parked item "live-e3w divisor in bn2" resolved by coverage, not by a new physics
change. NEMO's divisor is `eosbn2.F90:1467`: `pn2(...) = grav*(zaw*dT-zbw*dS) /
e3w(ji,jj,jk,Kmm) * wmask(ji,jj,jk)` — LIVE `e3w(Kmm)`, i.e. `e3w_0*(1+r3t)`
under `key_qco`. Searched (Rule 0): every call site of
`compute_buoyancy_frequency_nemo_bn2` (`eos.py`, `gm_redi_latlon_cgrid.py` x3,
`k_profiles.py` x2, `enhanced_diffusion.py`, `_shared.py` TKE passthrough).
ALL already thread live `gdept`/`gdepw_int` (via `nemo_bn2_live_ladders` or an
equivalent Jacobian stretch), except one inconsistency:
`_nemo_mld_from_n2_integral` (`gm_redi_latlon_cgrid.py:505`, the
`gm_redi_mld_criterion="n2_integral"` MLD path the DINO kamm card selects)
applied the `jacobian` stretch whenever non-`None`, with no
`isinstance(z_coord, OceanPartialCellCoordinate)` gate — unlike its sibling
`_nemo_wpoint_e3w_wmask_n2` (:787-788), which already has that gate because a
plain `OceanZStarCoordinate`'s `jacobian` is `(eta+H_bathy)/H_max` (global
normalisation), NOT `(1+r3t)`. **Currently INERT on DINO** — the kamm card
always builds `OceanPartialCellCoordinate` (real bathymetry/partial cells), so
the two forms coincide there — but a landmine for any future z*-only caller.
Fixed to match the sibling gate exactly (no new config surface); direct
synthetic-violation test added (`test_n2_integral_live_e3w_gated_on_partial_
cell_coordinate`, `tests/ocean/unit/test_nemo_mld_criterion.py`) proving BOTH
directions (z*-only: bit-identical to `jacobian=None`; partial-cell: the
stretch measurably changes the result).

Re-measured post-fix (fp64 explicit, `LEGOESM_NEMO_E3T=both`, BEFORE-level
`tke_dump_rn2b` per the time-level registry) via
`eos_rab_bn2_per_element.py` — reproduces the CLOSED note exactly (fix is a
no-op on the active card, as predicted):
- GLOBAL (n_wet=332,134 T / 332,214 W-interfaces): corr 1.00000000, mean-ratio
  1.000000, err_norm median **1.413e-17**, p99 2.940e-15, max 9.046e-15.
- DEEP SOUTHERN BOX (T-rows 14-22, k>=27, n_wet=2,941): corr **1.0**, ratio
  **0.9999999999999998**, err_norm median **4.444e-16**, p99 3.555e-15, max
  7.111e-15.

Both AT BAR (`corr>=1-1e-9`, `|ratio-1|<=1e-6`, per-element `<=1e-9`). Gate
row `bn2 (rn2b)` note updated with this re-verification; `AT BAR 15 | DEBT 32
| UNMEASURED 5 | WAIVED 1 | total 53` unchanged (no row moved category — this
was a coverage-completeness fix, not a new measurement).

Tests: `tests/ocean/unit/test_nemo_mld_criterion.py` (6 passed, incl. the new
one), `pytest -k "fidelity_bar or time_level"` (28 passed, 14 skipped).
REVIEW PENDING (agent-review) — codex CLI unavailable this session.

**Next**: item 5, `ldf_slp` (already CONDITIONING-LIMITED / RESOLVED-WITH-NOTE
per the 2026-07-30 human decisions above) — sweep otherwise ready to move to
whatever comes after `ldf_slp` in the NEMO execution order, or to the Redi/
GM lane the 2026-08-02 twin-budget work re-ranked as leading (see below).

---

# DINO/NEMO fidelity campaign (#1226) — current state digest

**Purpose.** One short file agent briefs can point at instead of re-typing context.
Authoritative detail lives in the gate script's row provenance and in the campaign
memory addenda; this is the map, not the territory. **Update it each iteration.**

Gate (run it, do not quote from here — this line goes stale):
```
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py
```
Last observed 2026-07-30: `AT BAR 15 | DEBT 32 | UNMEASURED 5 | WAIVED 1 | total 53`.
Bar: `corr >= 1-1e-9`, `|ratio-1| <= 1e-6`, per-element `<= 1e-9`.

---

## RETRACTION 2026-07-31 — the NEMO ACC reference is NOT converged (read before any ACC claim)

**The NEMO DINO reference run is 20 years. There is no NEMO data after year 20, and it was
still climbing when the record ended.** Source: `RUN_20Y` / `RUN_20Y_REBUILD`
(`nn_it000=57601`, `nn_itend=230400` = 20 yr x 11520 steps), file
`oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_20Y_REBUILD/DINO_1y_00060101_00201230_grid_U.nc`.
The only 50-year config on the machine (`EXP00`) was **never executed** — namelists and a
symlinked binary, no restart, no output, no `ocean.output`.

NEMO's own tail: y18 140.41 (+0.92), y19 140.74 (+0.33), y20 **142.81 (+2.07)**. A run that
stopped, not a plateau.

**RETRACTED:** the claim "NEMO settles by ~y18 and stays flat at 142.8, legoESM is still
climbing" — asserted verbally during the 40-year ladder A/B, **never present in the data**.
No repo doc ever claimed NEMO convergence; the error was mine.

**What it invalidates:** every `ratio_conv` for legoESM years 21-40 divided a moving legoESM
by a frozen snapshot of a still-moving NEMO — Rule 7 (different windows on the two sides).
"0.957 at y39" does NOT mean 95.7% of NEMO; it means 95.7% of where NEMO was at y20, a value
NEMO was leaving at 2 Sv/yr. Also RETRACTED: "the deficit is really ~4%", which was
manufactured entirely by running 20 years past the end of the reference.

**Valid comparison window is matched-year, y1-y20 ONLY:**

| year | lego (e3t=off) | NEMO | ratio |
|---|---|---|---|
| 5 | 68.8 | 91.1 | 0.755 |
| 10 | 85.4 | 121.07 | 0.705 |
| 20 | 113.4 | 142.81 | **0.794** |

⇒ The deficit is ~21-30% across the whole legitimate window. It did not close.

**Surviving positive signal (matched y20) — STATISTIC-DEPENDENT, quote both:**
- single year y20: NEMO **+2.07**/yr vs lego off **+1.525**/yr → NEMO faster
- decade y10→y20: NEMO **+2.174**/yr vs lego off **+2.800**/yr → lego faster

NEMO's annual increments are noisy (y18 +0.92, y19 +0.33, y20 +2.07), so the **decadal rate is
the trustworthy statistic**. Do NOT quote the single-year comparison alone in either direction.

**SUPERSEDED 2026-08-01 by the NEMO y21-40 run — do not reuse the "closing faster" framing.**
Over y20-40 the differential REVERSES (NEMO +1.666/yr vs lego `off` +1.216/yr) and the ACC
ratio is FLAT at 0.794→0.782. The y10-20 decade was not representative. Neither "phase lag" nor
"catching up" survives; see the ANSWERED 2026-08-01 section below.

## 40-year ladder A/B result (2026-07-31) — RAW, no ratios past y20

Both arms completed cleanly, 40/40 yr, `STABLE=True`, `finite=True` throughout. Worktree
`legoESM-worktrees/dino-e3t-ab` @ `be04588b8`; outputs in gitignored
`results/dino_1226_e3t_ab_40y/`.

| | y20 | y40 | rate y21-40 |
|---|---|---|---|
| `off` (1-D ladder) | 113.396 | **137.724** (+1.030) | ~+1.1/yr |
| `both` (true 3-D)  | 109.015 | **130.298** (+0.994) | ~+1.0/yr |

**RETRACTED:** the `ratio_conv` column the watcher printed on every row y21-y40. It was
configured `NEMO_CONVERGED = 142.8` on the false premise that NEMO's increments had decayed to
~+0.6/yr. Raw ACC values and increments in those rows are UNAFFECTED — only the ratio is bad.

**THE KEY OBSERVATION — increments did not decay at all.** They sat in a 1.0-1.4 band for
**twenty consecutive years**; the +0.5/yr equilibration flag never fired. An equilibrating
system's increments decay. A rate that halves once (~2.8 → ~1.1 near y20) then holds dead flat
for two decades looks like a **constant-rate drift**, not a settling transient. y40 is a LOWER
BOUND on each arm's asymptote, not the asymptote.

Density corroborates: channel band (T-rows 14-48, -64.4..-45.4 lat, 1400 m split, 342,134 wet
cells, identical mask both arms):

| | dρ upper | dρ deep | z(maxN2) |
|---|---|---|---|
| `off` y20 | -0.4767 | -0.1038 | — |
| `off` y40 | -0.5312 | -0.1505 | 202.5 m |
| `both` y40 | -0.5216 | -0.1409 | 202.5 m |

Deep contrast +45% over y21-40, still monotonic. z(maxN2) agrees between arms but is quantized
to the vertical ladder (~20-25 m there) — it can only say "same to within one level".

**Rule 8, unchanged and unexplained:** `both` stayed BELOW `off` for all 40 years (gap 7.4 Sv
at y40). The more faithful geometry gives the WEAKER ACC ⇒ something compensates for the wrong
1-D ladder.

**⇒ NEXT MEASUREMENT: extend the NEMO reference to 40-50 yr** (DINO's README works in 50-year
batches; RUN_20Y ran ~8.7 min/simulated-year on 16 ranks ⇒ ~3 h for 20 more years; the y20 end
restart `DINO_00230400_restart_*.nc` exists). Extending *legoESM* further cannot answer
anything — it adds uncomparable years. Until NEMO is longer, do not quote a legoESM ACC ratio
past year 20.

**The discriminating question this answers:** does NEMO also ramp linearly past y20?
- NEMO flattens, we don't ⇒ we have a genuine **drift** (likely deep). A real defect, and a far
  more tractable target than a 25% transport deficit — a drift has a source findable in a budget.
- Both ramp ⇒ neither model is near equilibrium at y40, the "deficit" framing was premature, and
  the meaningful comparison is the ramp RATE, not the level.

## ANSWERED 2026-08-01 — NEMO y21-40 run. **BRANCH 2: both ramp.** CONFIRMED

NEMO extended to y40 (`RUN_40Y/`, `RUN_40Y_REBUILD/`). Binary mtime Jul 30 22:25 = the one the
bit-identical repro control validated. Namelist diff vs `RUN_20Y`: ONLY `nn_it000`,
`nn_itend`, both `cn_ocerst_in` lines. 2h33m for 20 yr, clean `STOP 0`. Harness re-validated
against the known y20 = 142.8098 Sv exactly before any new number was trusted.

| | y20 | y40 | rate (endpoint) |
|---|---|---|---|
| **NEMO** | 142.810 | **176.124** | **+1.666/yr** (LSQ slope +1.255) |
| lego `off` | 113.396 | 137.724 | +1.216/yr |
| lego `both` | 109.015 | 130.298 | +1.064/yr |

**NEMO does NOT flatten.** Final increment +1.941 at y40 ≈ its +2.072 at y20 — twenty years, no
decay. It fakes a plateau at y27-31 (increments +0.98, +0.42, **−0.41**, +0.74, +0.69) then
re-accelerates monotonically to y40. ⇒ **A multi-year lull in DINO spin-up means nothing; only
multi-year rates are interpretable.** Single-year increments carry ~±1 Sv interannual noise.

**RETRACTED (mine, same night): the "constant-rate drift" hypothesis.** I flagged legoESM's flat
non-decaying 1.0-1.4 Sv/yr increments over y21-40 as drift-like rather than transient-like.
**NEMO does exactly the same thing.** Flat non-decaying increments are a property of DINO
spin-up from rest, not a legoESM defect. This was the next thing I would have chased.

### The real, well-posed result — a STABLE ~21% deficit, located in the DEEP

ACC ratio is FLAT across the window: **0.794 (y20) → 0.782 (y40)**. Not catching up, not
diverging ⇒ rules out BOTH "phase lag" and "runaway". Ramp-rate ratio 1.216/1.666 = **0.73**,
close to the transport ratio 0.78 — the whole thermal-wind structure is scaled down by a
similar factor.

Density contrast, identical protocol (T-rows 14-48, −64.4..−45.4 lat, 1400 m split, 342,134 wet
cells; all four reproduce the doc geometry exactly):

| | dρ upper | dρ deep |
|---|---|---|
| NEMO y20 | −0.5255 | −0.1739 |
| NEMO y30 | −0.5279 | −0.1998 |
| **NEMO y40** | **−0.5790** | **−0.2144** |
| lego `off` y40 | −0.5312 (**92%**) | −0.1505 (**70%**) |
| lego `both` y40 | −0.5216 (**90%**) | −0.1409 (**66%**) |

⇒ **Upper contrast ~90-92% right; deep contrast only 66-70%.** Same signature at y20 ⇒ stable in
time, not transient. The ACC ratio 0.78 sits BETWEEN the upper and deep ratios exactly as
depth-integrated thermal wind requires — three independently computed quantities that must be
consistent, and are. Corroborates the earlier "80-98% of missing thermal wind is sourced below
1000 m" attribution from a second independent direction. **The problem is BUOYANCY (deep
stratification), not momentum.**

### Next test — Rule 4, both-sided GM ablation (PLAUSIBLE lead, NOT confirmed)
An over-strong GM bolus flattens isopycnals hardest where slopes are steepest = the deep
southern channel ⇒ would give exactly this signature (upper nearly right, deep a third short,
ACC short in proportion, all stable in time). The campaign's open **37%-larger bolus in the
advecting flux** row now has a candidate consequence with the right depth structure AND sign.
This is a mechanism matching a signature — the exact pattern this campaign has repeatedly
refuted. So TEST it, do not adopt it.

**Ablate GM in BOTH models** from the y20 restarts (~10 yr; NEMO ~1.3 h). Gap in deep contrast
collapses ⇒ GM owns it. Gap survives ⇒ GM exonerated, look at diapycnal mixing / southern
convection. Verify the two ablations are the SAME ablation by reading both configs first
(`use_gm_redi=False` may kill only the bolus while leaving isoneutral diffusion at strength;
NEMO's flag may do something different).

y40 restarts `DINO_00460800_restart_*.nc` exist if a push to y60 is ever wanted.

## y40 DEEP CENSUS (2026-08-01) — fixed-depth. Numbers CONFIRMED, one inference REJECTED

Harness self-validated first (fresh code path importing `acc_thermal_wind.py` as `tw`; wet-cell
count 342,134 asserted; fp64 policy set explicitly; reproduced all recorded contrasts to
<5e-4). NEMO T is fp32 on disk (native netCDF write) — fine for a 0.5 degC signal.

Below 1400 m (k>=27, 13,502 cells, identical mask all three):

| | T (degC) | S (psu) | sigma |
|---|---|---|---|
| NEMO | 3.1158 | 35.1034 | +1.4191 |
| `off` | 3.6056 | 35.1156 | +1.3477 |
| `both` | 3.6184 | 35.1164 | +1.3462 |

- Deep water **lighter by 0.071-0.073**, driven by **T at 8:1** over S (+0.49 degC warm,
  +0.012 psu salty and partially compensating).
- sigma(z) divergence **GROWS monotonically with depth**: ~0 at 200 m, -0.036 at 1400 m,
  **-0.12 at 3757 m**. Sign FLIPS at k=11 (156 m) — lego is slightly DENSER above that.
- Deep vertical gradient **85%** of NEMO's (1.97e-4 vs 2.29e-4 kg/m3/m).
- Isopycnal meridional excursions **65-90%** of NEMO's (NEMO 974-1190 m; `off` 766-912 m).
  Deepest target sigma=+1.56 brackets only 19/35 band rows in lego vs 33/35 in NEMO.
- **Dense water is NOT missing at the tail**: at p5/p10 lego is *denser* than NEMO (+0.02 to
  +0.04). Only the min (lightest extreme) is lighter.

### REJECTED INFERENCE — do not repeat it
The census concluded the +0.49 degC warm bias argued for a formation component, reasoning that
"redistribution conserves water-mass properties". **That is wrong here.** The below-1400 m box
is a **FIXED GEOMETRIC volume, not a material one**. Purely adiabatic isopycnal flattening moves
isopycnal surfaces across the fixed 1400 m horizon and changes the box's contents with ZERO
diapycnal flux. A fixed-depth volume-mean warm bias therefore **cannot discriminate
redistribution from water-mass change** — and it was the number the verdict leaned on hardest.

**GENERAL RULE for this campaign: never diagnose water-mass change from a fixed-depth average.
Volume per density class is the invariant under adiabatic redistribution — census in density
coordinates.**

### Third hypothesis this raises
Dense tail present (p5/p10 denser than NEMO) + deep volume-mean warm ⇒ **dense water forms but
fails to fill the abyss** — a ventilation/spreading deficit, distinct from BOTH formation
failure and isopycnal flattening.

### Discriminator now running (free, existing y40 output)
Volumetric census by sigma class over the band, all three models:
- volume-per-class MATCHES NEMO while mean depth per class differs ⇒ purely **adiabatic
  redistribution**, GM/bolus owns it.
- lego holds **less volume in the densest classes** ⇒ dense water DESTROYED or never formed:
  diapycnal mixing (spurious numerical mixing from tracer advection, Redi slope leakage,
  too-large K_v) or a formation shortfall.

Do the GM ablation only AFTER this, and only if it points that way — it would otherwise be the
wrong experiment.

## DENSITY-CLASS CENSUS (2026-08-01) — **THE CAMPAIGN'S TARGET HAS MOVED.** CONFIRMED

Volume-per-sigma-class over the channel band, y40, existing output only. Instrument validated
first: reproduced all recorded contrasts and deep means; **total band volume identical across
all three models to machine precision** (2.694775e+16 m3, rel diff 0.000000%) — without that
control the histograms would not be comparable.

**Cumulative volume denser than sigma_0 (lego / NEMO):**

| sigma_0 | 1.20 | 1.40 | 1.50 | 1.60 | **1.7133 (NEMO p99)** |
|---|---|---|---|---|---|
| `off`/NEMO | 0.930 | 0.707 | 0.543 | 0.264 | **0.000** |
| `both`/NEMO | 0.931 | 0.719 | 0.549 | 0.243 | **0.000** |

**legoESM produces ZERO water denser than NEMO's 99th percentile — in either arm.** NEMO holds
1.03e15 m3 there (~3.8% of band volume). The deficit is MONOTONIC and worsens toward the dense
end (7% shortfall at sigma>1.20 → total absence at p99).

⇒ **NOT adiabatic redistribution.** Heave preserves volume-per-density-class and only moves its
depth. Volume is being destroyed or never created, progressively more so the denser the class.

**Independent confirmation from the decomposition:** within MATCHED sigma bins, lego's water is
both **warmer AND saltier** than NEMO's, compensating onto the same isopycnal (e.g. bin
[1.38,1.40): NEMO 3.1535 degC / 35.1035 vs `off` 3.7270 / 35.1170). Same density, different
point on the T-S curve ⇒ genuinely different water masses, NOT the same parcels sitting deeper.
NEMO's missing densest classes are **cold and fresh** (T 2.3-2.8 degC, S 35.09-35.10) —
AABW-like, surface-formed by southern cooling. legoESM never makes it anywhere in the band.
(Mid-range classes DO sit deeper in lego, so some heave coexists — but it rides on top of the
water-mass difference, it does not replace it.)

⇒ **GM ABLATION IS THE WRONG EXPERIMENT. Do not run it as the next step.** The free diagnostic
redirected ~3 h of compute; that is the argument for always doing the census before the run.

### Weighting subtlety worth keeping
The recorded deep-mean table used **thickness-only** weighting (e3t0); a true volumetric census
needs real cell volume (e1t*e2t*e3t0), and **e1t shrinks poleward** across a -64.4..-45.4 band.
The reweighting moves deep-mean T by +0.09 to +0.15 degC in all three models. Not a bug — but
state which weighting any future number uses. Also note: 342,134 is the WHOLE-DOMAIN wet count;
the band-restricted all-depth count is **62,642** and the band deep-only count is **13,502**.
Do not conflate the three.

### THE TARGET, restated
No longer "a 21% ACC deficit". It is: **we do not form, or we destroy, the densest Southern
Ocean water mass.** Deep stratification (66-70%), thermal wind, and ACC (78%) all follow
downstream of that.

### NEXT — free diagnostic, splits the last ambiguity
**Never formed, or formed then eroded?** Compare winter surface / mixed-layer density in the
southern convective region, both models.
- source MATCHES ⇒ **erosion in transit**: spurious diapycnal mixing from tracer advection,
  Redi slope leakage, or too-large K_v.
- source DIFFERS ⇒ **formation**: surface buoyancy forcing, convection, mixed-layer depth.

The campaign already has open rows in the second area (zdftke composite residual corr 0.966;
the `zdf_mxl` work; the pending MXL re-walk with the true `rn2` dump).

## SOURCE-vs-PATHWAY (2026-08-01) — **the defect is between surface and abyss.** CONFIRMED

Existing output only. Harness re-validated (reproduced the contrast table AND the cumulative
volume ladder) before new numbers.

**DATA LIMIT, stated up front:** every `RUN_*` dir under `cfgs/DINO/` writes **annual means
only** — no `_1m_`/`_5d_` file exists anywhere in the tree. legoESM's y40 npz has
`T,S,eta,u,v,land_mask` and **no MLD, no sub-annual field**. ⇒ **the winter formation event is
INVISIBLE to all existing output.** Everything below is an annual-mean proxy. NEMO does carry
`somxl010`/`somixhgt` (annual-mean MLD: 233 m mean / 414 m max at j=14, decaying to ~110 m
mid-band); legoESM has no MLD diagnostic at all — reported UNAVAILABLE, not substituted.

**Where NEMO's densest water is:** sigma>p99 is **100% at k=34, the single deepest wet level
(3757 m)**, across all 24 wet rows (j=14..37) and 43/52 columns — an interior **bottom-trapped
abyssal blanket**, NOT a boundary/shelf/corner feature. Mean T=2.34 degC, S=35.09. In those
exact 627 cells legoESM is **+0.72 to +0.77 degC warmer**, sigma 1.618-1.625 vs NEMO 1.745,
**0% reach p99** in either arm.

**Vertical structure of the deficit, southern rows j=14-22:**

| depth | 5 m | 514 m | 1420 m | 2192 m | 3757 m |
|---|---|---|---|---|---|
| `off` − NEMO | −0.0038 | −0.0286 | −0.0584 | −0.0791 | **−0.1151** |
| `both` − NEMO | −0.0139 | −0.0325 | −0.0620 | −0.0859 | **−0.1245** |

Surface nearly matched; bottom **10-30x worse**. The growth is **entirely thermal** — the
T-driven part amplifies ~5x surface→bottom (dT +0.21→+0.70 `off`, +0.36→+0.76 `both`) while the
salt bias stays roughly flat. ⇒ **The defect acts BETWEEN the surface and the abyss, not at the
source.** CONFIRMED and robust; this is the actionable conclusion.

### UNRESOLVED — "erosion" is NOT established over "incomplete ventilation"
The agent's verdict was FORMED-THEN-ERODED. **Not accepted.** Both models are still ramping at
y40 and the abyss fills last; a model with **weaker overturning** shows exactly this signature
(surface matched, deep lagging, deficit growing with depth) **with no excess mixing at all**.

The trend argues AGAINST steady-state erosion — the deep contrast ratio is **CATCHING UP**:

| | y20 | y40 |
|---|---|---|
| deep dρ lego/NEMO | 0.1038/0.1739 = **0.597** | 0.1505/0.2144 = **0.702** |

In absolute terms lego gained +0.047 over y21-40 vs NEMO's +0.041. Steady erosion at
equilibrium would not do that; a ventilation lag would. **Mechanism genuinely open between
"we mix it away" and "we haven't delivered it yet."**

### Calibration — temper the headline
"**ZERO volume above p99**" is a THRESHOLD statement, not a missing water mass. The underlying
deficit is continuous and graded (0.93 → 0.71 → 0.54 → 0.26 → 0.00); it is the tail of a
monotonic shortfall crossing a sharp cut, not a categorically absent process. Dramatic phrasing,
softer physics — do not quote the zero without the ladder.

### NEXT — separates the two branches, uses existing dump infrastructure
Compare NEMO's **`avt`** against legoESM's effective vertical diffusivity in the deep southern
water column at y40. Larger deep K_v in lego ⇒ erosion measured directly rather than inferred.
**TRAP (already bit us once):** post-EVD legoESM vs pre-EVD NEMO gave a ratio of 987 — that was
the instrument, not the physics. Pin the EVD stage on both sides before believing any number.

Other candidates on the pathway: spurious numerical diapycnal mixing from tracer advection,
Redi slope leakage (the 4 CONDITIONING-LIMITED `ldf_slp` rows), overturning strength.

## TWIN DEEP-BOX HEAT BUDGET (2026-08-01) — **ADVECTION delivers the spurious heat; VERTMIX EXONERATED**

Twin protocol: BOTH models from NEMO's y20 restart (16-tile restart stitched with NEMO's own
`rebuild_nemo.exe` → `RUN_TWIN_Y20_BUDGET_STITCHED/`), identical 90-day window (2880 steps,
rn_Dt=2700 s → 32/day), `e3t=both`, fp64 policy explicit (the run caught the fp32 trap AGAIN —
`JAX_ENABLE_X64=1` alone left the harness fp32). Day-0 gate: bit-identical T/eta/u/v/before/TKE
vs the restart. Reused `kamm_twin_90d._build_twin_state` + `BoxHeatBudgetAccumulator` verbatim;
new driver `scripts/validate/ocean_fidelity/dino_1226/twin_y20_box_budget.py`. NEMO arm
`RUN_TWIN_Y20_BUDGET/` (STOP 0). **Budget closure proven FIRST**: <2% residual both sides in
the deep bands; 0-200 m closes poorly on BOTH sides (12-76%) — surface band NOT usable for
attribution.

Deep southern box (rows 14-22, >1400 m), lego − NEMO, W/m2 on the box:

| term | NEMO | lego | Δ | share of gap |
|---|---|---|---|---|
| adv (h+v) | −9.75 | −8.81 | **+0.94** | **73%** |
| iso_redi (+k33) | −3.72 | −3.35 | +0.37 | 29% |
| vertmix | +0.05 | +0.03 | **+0.02** | ~2% |
| **total** | −13.42 | −12.13 | **+1.29** | |

- **VERTMIX EXONERATED at the climate-budget level.** The most-upgraded lane (avt per-step
  0.4%) delivers deep-box heat matching NEMO to 0.02 W/m2. Stop grinding vertical rows for
  THIS defect.
- **Localized**: full-band deep box (rows 14-48) gap ≈ +0.04 W/m2 — the excess is confined to
  the southernmost rows with compensation northward ⇒ a **circulation-pattern difference**
  (overturning/downwelling placement), not a uniform diffusive leak.
- **Magnitude**: +1.29 W/m2 → +0.0041 degC/yr ≈ **21% of the observed +0.02 degC/yr**. Right
  sign/order; single 90-day sample, NEMO's own in-window rate drifts 17% ⇒ PLAUSIBLE mechanism,
  NOT confirmed magnitude. Needs the full-year window.
- **Parked gap**: lego `k33` bucket ≡ 0.0 at fp64 while NEMO's analog (zdf−zdfp) ~5.6% of its
  vertmix — real hole in the iso/K33 split, small vs the 1.29 signal. Follow up separately.

**NEXT (one lane): extend twin to a FULL YEAR with adv split h vs v.** Kills the seasonal
caveat (NEMO ~8 min; lego a few GPU-h) and separates "spurious diapycnal mixing in the
advection scheme" from "overturning delivers heat elsewhere" — different fixes.

## YEAR-LONG TWIN (2026-08-01/02) — Redi share DOUBLES; h/v split VOID (convention mismatch)

Same twin protocol, 360 days (NEMO `RUN_TWIN_Y20_BUDGET_1Y/`; lego daily accumulator, e3t=both,
fp64). Outputs `twin_y20_1y_{nemo,lego}.npz` + `analyze_twin_1y.py` in the session scratchpad.
Lego closure residual −0.03 W/m2 deep-band.

**h/v ADVECTION SPLIT IS NOT COMPARABLE ACROSS THE MODELS — do not use it.** Deltas: adv_h
**+23.7**, adv_v **−22.7** W/m2, cancelling to +1.0. Equal-and-opposite ~20 W/m2 per-component
"errors" from an identical initial state = the two models bucket the same transport differently
(z-star grid-motion/dilation terms land in different components between lego's accumulator and
NEMO's ttrd_xad/yad/zad). Trap-catalogue class: "two A-grid winds are different quantities."
The SUM is robust; the components are bookkeeping.

Deep southern box (rows 14-22, >1400 m), FULL-YEAR mean, lego − NEMO in W/m2:

| term | Δ | share | quarters (Q1→Q4) |
|---|---|---|---|
| adv (h+v SUM) | **+0.98** | 57% | +0.99, +0.84, +0.91, +1.35 |
| iso_redi | **+0.77** | 44% | +0.37, +0.50, +0.88, **+1.21** |
| vertmix | −0.03 | ~0 | flat |
| **total** | **+1.72** | | +1.34, +1.31, +1.76, **+2.52** |

1. **Redi share DOUBLED vs the 90-day window** (29% → 44%, nearly co-equal with advection) —
   its delta GROWS monotonically as the states diverge; the 90-day snapshot undersold it. The
   4 CONDITIONING-LIMITED `ldf_slp` rows and the parked k33≡0 gap are PROMOTED in relevance.
2. **Total gap grows with divergence**: +1.29 (90 d) → +1.72 (yr mean) → +2.52 (Q4)
   ⇒ ~+0.0055 degC/yr (~27% of the observed +0.02) and trending up — plausibly sufficient in a
   free run over decades. Mechanism magnitude upgraded from "21% at face value".
3. **Localization re-confirmed**: full-band deep box (14-48) total Δ = +0.20 W/m2.
4. **Vertmix exoneration now solid across two windows** (−0.03 W/m2).

## RETRACTION 2026-08-02 — the "+0.37 operator seed" was ~86% BOOKKEEPING. Bolus REFUTED.

The face/bolus dump run (`RUN_TWIN_FACE10_DUMPS/`, WRITE-only edits proven bit-identical before
use; `ww` dump = Eulerian-only per `stpmlf.F90:244/315` + `traadv.F90:330/344`; ψ→increment
reconstruction after catching that the raw `eiv_dump_u/v` wrote streamfunction slot-1, registry
corrected) delivered the corrected decomposition, validated three ways (lego BOL faces sum
matches its own bucket to −0.004; BOL−EUL w1400 −3.86 independently matches NEMO's dumped
increment −3.78; NEMO cell integration reproduces lego's BOL−EUL to ~1% on all faces/days):

| component (days 1-10 mean, lego−NEMO, + into box) | W/m2 |
|---|---|
| bolus delivery (south +9.33/+9.23, north −9.53/−9.49, w1400 −3.86/−3.78) | **−0.02** |
| Eulerian faces (daily values noisy ±0.6; day-1 alone −0.32) | **+0.086** |
| NEMO advective-form-vs-flux-form residual (NEMO faces −8.396 vs bucket −8.800) | **+0.404** |
| bucket-based "seed" (identity check) | +0.472 |

1. **BOLUS REFUTED as the seed** — the two models' bolus heat deliveries agree to 1-2% on all
   three faces, all 10 days. **TENSION with the old "bolus 37% larger" row**: it cannot both be
   37% off and agree here — that row measured a different quantity (flux vs delivery) or a
   different state. Re-examine it before ever citing it again.
2. **RETRACTED (mine): "+0.374 operator-level seed at day 1".** It was lego's FLUX-form bucket
   minus NEMO's ADVECTIVE-form bucket (`trdtra.F90:257-259` subtracts T·div U) — non-commensurable.
   The convention-proof cross-model advective difference over days 1-10 is **~+0.07-0.09 W/m2**.
   CAVEAT kept: the +0.404 residual is confounded with snapshot-vs-accumulated sampling; a
   step-accumulated face-flux dump (new Fortran) would separate them. PLAUSIBLE that it is pure
   bookkeeping; NOT confirmed.
3. **What survives**: total +1.72 W/m2 excess (realized dH/dt, bucket-independent) — SOLID.
   Vertmix exoneration — SOLID (flux-form both sides). w1400 Eulerian flux now exact on NEMO
   (real `ww`), agrees with lego to ~2% (−21.3 vs −21.8).
4. **Unit-convention audit (post-review 2026-08-02):** the twin driver saved lego in Gill
   (1025, 3994) and NEMO in NEMO (1026, 3991.868) convention; the analysis diffed them raw.
   Measured conv_ratio = **1.00044** ⇒ contamination +0.005 W/m2 on the total, +0.003 on adv —
   three orders below the signals. **All recorded deltas stand.** Driver now passes NEMO's
   (rho0, cp) to both arms so future npz pairs share one convention.
5. **RE-RANKING ⇒ the Redi term (+0.77 yr-mean, +1.21 by Q4, flux-form BOTH sides = commensurable)
   is now the leading identified physical contributor.** The true advective excess is uncertain
   until form-matched — possibly small. **The hunt pivots to the Redi/isoneutral lane**: the 4
   CONDITIONING-LIMITED `ldf_slp` rows, slope structure in the deep southern box at the y20
   state, and the k33 wiring.

## #1455 A5 — TRUE same-state Redi comparison (2026-08-02): NO positive operator seed; delta OPPOSITE-signed & 10× smaller

The completing measurement for the Redi lane. Prior "localization" (sibling probe
`redi_localize_1455_a5.py`) compared lego's Redi field at the y20 origin against NEMO's
`ttrd_ldf` at kt=230720 — NEMO's OWN trajectory **10 days later** — a confound. Here NEMO's
`ttrd_ldf` is the trend accumulated during **STEP 1** out of the y20 restart (kt=230401,
`RUN_TWIN_STEP1/`, nn_stock=1), evaluated reading Kbb = the IDENTICAL y20 state lego is
evaluated on. Both sides in NEMO's (rho0=1026, cp=3991.868) convention. fp64 explicit
(dtypes printed). Time level registry-checked: `time_level_for_dump("ttrd_ldf")=="before"`
(traldf_iso_scheme.h90:26-30 Kbb read; new registry entry in `time_levels.py`).
Probe: `scripts/tmp/redi_samestate_1455_a5.py`.

**BIT-IDENTITY CONTROL (mandatory) — PASSED.** `RUN_TWIN_STEP1` (nn_stock=1) vs
`RUN_TWIN_FACE10` (nn_stock=32) day-1 (step-32, kt=230432) restarts are **bit-for-bit
identical** across all 16 rank pieces for every prognostic field (tn/sn/un/vn/tb/sb/ub/vb/
sshn/sshb). ⇒ the trend-dump cadence does not perturb the trajectory; the step-1 `ttrd_ldf`
is on the genuine y20 trajectory. Day-0 gate also 0.0 (max|dT|=max|d_eta|=max|du|=0), before-
bridge tb/sb/ub/vb = 0.0.

**Deep southern box (rows 14-22, k>=27 i.e. >1400 m), STEP 1, lego − NEMO, W/m2 on the box:**

| | box W/m2 |
|---|---|
| lego iso_redi+k33 | **−3.028** |
| NEMO ttrd_ldf (step 1) | **−2.951** |
| **DELTA (lego − NEMO), same-state** | **−0.077** |
| [context] lego(t=0) − NEMO(step 32) | −0.068 |

**RECONCILIATION (Rule 1e) vs the recorded year-mean +0.77 (same lego−NEMO convention):**
- **SIGN IS OPPOSITE.** Recorded Q1→Q4 iso_redi delta = +0.37, +0.50, +0.88, +1.21 (lego
  LESS negative ⇒ warming excess into box). At the truly identical state lego's Redi is
  **MORE** negative than NEMO (−3.028 vs −2.951) ⇒ delta **−0.077**. Magnitude ~10× smaller
  than the +0.77 year mean and ~5× smaller than the +0.37 Q1 mean.
- ⇒ **CONFIRMED: there is NO positive operator-level Redi seed on the identical state.** The
  +0.77 accumulated excess is NOT present as a same-state operator mismatch at t=0 — it must
  grow from state divergence (the two trajectories' Redi fields diverging as the states drift),
  supporting the attractor / divergence picture rather than a fixed operator bias. This mirrors
  the "+0.37 operator seed" retraction (that was ~86% adv-form-vs-flux-form bookkeeping): the
  Redi lane, like the advection lane, shows **no commensurable same-state seed**.

**CAVEAT — the two sides are NOT perfectly the same quantity (K33 hole; verified this run).**
lego's `k33` bucket is **identically 0** (nonzero frac 0.0, max 0.0 at fp64 — re-confirmed),
so lego's `iso_redi+k33` = horizontal isoneutral Redi ONLY, while NEMO's `ttrd_ldf` = horizontal
Redi **+ the K33 vertical diagonal** (traldf_iso_scheme.h90:128/136). The delta therefore
carries lego's-missing-K33 wherever NEMO's K33 is nonzero. This is visible as the cancelling
K33/diapycnal triple at k=24/25/26 (just above the horizon; net +0.007, so it barely touches
the k>=27 box integral) and taints the k>=27 integral by whatever NEMO K33 lives below 1400 m
(recorded NEMO K33 analog ~5.6% of its vertmix ⇒ small, same order as the −0.077). NET: the
**"no positive seed / opposite sign" conclusion is robust** (the horizontal-Redi comparison is
like-for-like and the K33 hole can only make lego MORE negative, reinforcing the negative
delta), but the exact −0.077 magnitude is confounded by the K33 split at the ~5% level — do not
over-precision it. The parked k33-split hole is the clean follow-up.

**Structure (CONFIRMED for the y20 snapshot; localization target, not the ~0 magnitude):**
- **Depth**: below the 1400 m horizon the per-level deltas are small (each k row ~−0.01 to
  −0.09 W/m2). The dominant signal sits in a **cancelling K33/diapycnal triple just ABOVE the
  horizon** (k=24/25/26 = 915/1059/1220 m: −0.353, +0.483, −0.138 W/m2, net +0.007) — a
  vertical-diagonal (K33) placement difference, not a horizontal Redi leak. Consistent with
  the parked "lego k33 bucket ≡ 0 vs NEMO zdf−zdfp ~5.6%" hole.
- **Latitude**: deltas peak at rows 11-19 (−65 to −62°S, each ~−0.09 to −0.10 W/m2) — the
  deep-convection southern rows — decaying northward. Same rows the surface warm bias lives.
- **Concentration**: top 10% of deep cells carry 58% of |W| / 75% of signed W (moderate, not
  a hotspot). Top-10 cells: rows 18-21, depth 2115-2429 m, **steep slopes |slope|~8-11e-3**
  (box median 2.6e-3), N²~4-7e-7 — high-slope deep cells where lego's Redi is more negative.

**VERDICT DATA (for the caller, no verdict baked in the probe):** the same-state Redi delta is
~0 (−0.077 W/m2) and opposite-signed to the accumulated +0.77 ⇒ the twin's Redi excess grows
from divergence, not from a fixed operator seed. The residual structure is a K33/diapycnal
placement difference in a thin band just above the 1400 m horizon (steep-slope deep cells),
not a horizontal-Redi bias — a candidate for the parked k33-split hole, but small.

## STEP-1 PROBE (2026-08-02) — an OPERATOR-LEVEL SEED EXISTS: +0.37 W/m2 at day 1
**[SUPERSEDED by the RETRACTION block above — the seed was ~86% form-mismatch bookkeeping]**

Twin from y20 restart, bit-identical day-0 gate, step-1 semantics MATCHED (NEMO `l_1st_euler=F`
⇒ leapfrog rDt=2dt from step 1; lego `--bridge-before` seeds tb/sb/ub/vb, rdt=2dt). No MY_SRC
edits — per-step trends via `nn_stock=1` namelist-only (`RUN_TWIN_STEP1/`, same nemo.exe md5).

Deep southern box, adv h+v SUM, day-1 mean (steps 1-32): lego −8.178 vs NEMO −8.552 ⇒
**delta +0.374 W/m2 on an IDENTICAL state**. CAVEAT: lego day-1 closure residual −0.107 (~30%
of signal; 1-day leapfrog-init noise) — delta is nonzero beyond closure, but carries ±~0.1.
Step-1-alone +1.86 (single-step, noisy, not load-bearing).

⇒ **Pure trajectory divergence RULED OUT.** The seed is ~1/3 of the Q1 excess (+1.0); the
remainder compounds with state drift. Year-run day-1 apparent mismatch (−8.75 vs −8.18)
reconciled: the year accumulator samples the endpoint step (−8.756 = step 32), a
sampling-convention difference, not a bug.

**NEXT probe (convention-proof): direct heat flux through the box FACES** — the 1400 m horizon
and the northern wall at rows 22/23 — computed IDENTICALLY on both sides from state x velocity,
not from either model's trend bookkeeping. Splits "diapycnal leak through the horizon" from
"lateral delivery pattern". KNOWN WEAKNESS, state it with any result: saved cadence is 10-day
(NEMO) / daily (lego) means, coarse for w'T' covariance — a face-flux from mean fields misses
the eddy component; treat as mean-flow attribution only.

### Output gap worth closing
No sub-annual output on either side at y40 ⇒ the actual formation event is unobservable, and
legoESM has no MLD diagnostic in its npz at all. If the small surface residual ever needs
resolving (max surface sigma: NEMO 0.9965 vs lego 0.9689-0.9739; lego already +0.21 to +0.36
degC warm at the surface in the deepest-convection rows), the needed output is a **monthly
grid_T from both models at y40 plus a legoESM MLD diagnostic**.

## ORACLE BINARY PROVENANCE GAP (2026-07-31) — CONFIRMED, fix regardless of outcome

**The binary that produced the reference trajectory no longer exists.** `RUN_20Y` finished
2026-07-24 08:52. `BLD/bin/nemo.exe` was rebuilt **2026-07-30 22:25** from `cfgs/DINO/MY_SRC/`
files edited 22:24. `cfgs/DINO/` is **untracked by git** (`??` in `git status` of the
`oracle-builds/nemo5/nemo_5.0.2` repo), so the Jul-24 source state is unrecoverable and no
archived `nemo.exe` exists.

The edited files are `stpmlf`, `zdftke`, `dynspg_ts`, `dynzdf`, `sshwzv`, `dynadv`,
`dynatf_qco`, `traatf_qco`, `usrdef_sbc` — all of which DO contain campaign `dump` writes.

### RESOLVED — reproduction control PASSED, **CONFIRMED** (2026-07-31)
Re-ran year 20 alone (steps 218881-230400) with the Jul-30 binary from the y19 restart
`DINO_00218880_restart_*`, in `RUN_REPRO_Y20/` (reference dirs never written). Namelist diff
clean: `namelist_ref` byte-identical; `namelist_cfg` differs ONLY on `nn_it000` and
`cn_ocerst_in` (`nn_itend` was already 230400 — RUN_20Y's namelist is the final-phase one).

| field | max abs diff | `array_equal` |
|---|---|---|
| U, V, T, S, ssh | **0.000000e+00** | **True** |

**fp64 restart state at step 230400: all 1520 variables across all 16 tiles bit-identical.**
ACC: reference 142.8098167694 vs repro 142.8098167694, diff exactly 0 — which also
**self-validates the harness** against the campaign's stated 142.81.

Instrument validated before the zero was believed (6 negative controls): ref y19-vs-y20 annual
means differ (max 112.24 T); y19-vs-y20 restarts differ in 76/95 vars (max 1.15e4); ACC y19
140.74 vs y20 142.81; distinct inodes/sizes/mtimes; 342,134/372,528 nonzero points; SHA256 of
decoded arrays matches ref-y20 and differs from ref-y19.

⇒ The Jul-30 edits are **instrumentation-only, established empirically not by reading source**.
Bit-identity of the full fp64 state after 11,520 steps in an eddying regime is far stronger
than roundoff agreement — a single-ULP perturbation reaches O(1) within a year. **Post-Jul-30
dump-based comparisons were against the SAME NEMO as the reference trajectory.** Extension to
y40 is SAFE.

**HYGIENE FIX REQUIRED EITHER WAY: the oracle's patched source is part of the experiment's
definition and must be version-controlled.** Track `cfgs/DINO/MY_SRC/` (or archive a tarball +
`nemo.exe` beside every reference run) so a reference trajectory can always be tied to the
source that produced it. Annual restarts (`nn_stock=11520`) are what made this testable at all
— keep them on every oracle run.

## THE RULE (human, 2026-07-30)
Never guess. Every fix is a **transcription from NEMO source with `file:line` cited**.
If an exact NEMO match does not clear the row or the behaviour, **PAUSE and ESCALATE** —
no invented stabilizers, no plausible corrections. An escalation after an exact match
that failed is a SUCCESS of the process.

## Standing mechanical preconditions (each earned by a real failure)
1. **fp64** — `precision_gate.require_fp64`. `JAX_ENABLE_X64=1` does NOT change the policy;
   `get_policy().storage` defaults to float32 and **constructors read it at build time**,
   so an fp64 state must be *built* under an fp64 policy.
2. **Time level** — `time_levels.time_level_for_dump` (fails closed). NEMO routines mix
   Kbb/Kmm/Kaa within one call; establish each input's level from source.
3. **`LEGOESM_NEMO_E3T` explicit** — the default `"off"` is a known-wrong 1-D ladder
   (12.9% off below k=25) and has contaminated four measurements.
4. **Metric identity** — before comparing two numbers from different scripts/ledgers,
   verify **same metric, same aggregation, same POPULATION, same reducer**.
   Four false conclusions today came from violating this.
5. **Dump provenance** — state which side of any later operation a dump sits on, with the
   F90 write line. Six confirmed cases where the dumped quantity was not the applied one.

## Recurring trap catalogue
- `avt_k`/`avm_k` are **closure-only**; EVD is applied to copies and never written back
  (`zdfphy.F90:313-323`). NEMO emits no post-EVD field.
- An MLF dump of `ts(Naa)`/`uu(Naa)` **after** a stage carries the whole step's accumulated RHS.
- A dump may predate a later overwrite (`ATF`: `mlf_baro_corr` rewrites `puu(Kaa)` *and*,
  under `ln_bt_fw=.false.`, `puu(Kmm)`, before `dyn_atf_qco` runs).
- A dump may omit an operation NEMO applies one line later (`eiv`: the minus at
  `ldftra.F90:833` vs the dump at `:864`; signature = corr exactly **-1.000000**).
- Filename tokens lie: `atf_dump_uu_before.bin` is **Kmm** (pre-filter *stage*, not Nbb).
- Probes go **config-blind**: `coverage_rows_measure.py` twice measured a static path
  regardless of the card. Fixing one blind dispatch does not fix its siblings.
- **Unit harness**: smooth **synthetic inputs can hide mask/boundary errors** (a mandatory
  `mask=` omission moved a synthetic case 0.9649→0.9673 but real data 0.706→0.9999999).
  Dynamic range and boundary coverage are different properties — cross-check geometry-
  sensitive routines on **real restart data**.
- Fortran **explicit-shape dummies** with decomposition macros (`A2D(0)`, extent 52 not 56)
  silently scramble (i,j,k) via sequence association; compiles and links clean.
- Halo/domain: some dumps are full `56x203` (halo=2), others interior `52x199` — use `_load_full`.
- legoESM raw `u`/`h_u` carry a **+1 column offset** vs NEMO's index (`_u_to_nemo` strips it).
- `w`/`G` live on the **interface** grid (37) vs `u`/`h_u` on cell centres (36).

## Harness artifacts found (rows that were never model defects)
`eiv transport u/v` (missing minus) · `dyn_ldf u/v` (probe fed NOW, NEMO reads Kbb) ·
`ATF filter u/v` (stale pre-`mlf_baro_corr` Kaa → **bit-exact**, now AT BAR) ·
`zdftke composite` (post-EVD reference) · `lbc_lnk sign` (no per-element entry) ·
`dyn_adv ZAD` **42.5% of the union metric** (dump carries below-seafloor Krhs that
`dynzdf.F90:121` discards). **The DEBT list overstates the model's real infidelity.**

## Live rows
- **`dyn_adv ZAD`** — highest leverage (dominates the v-unification, corr 0.9845).
  **ROOT CAUSE CONFIRMED**: `dynzad.F90:86` has no per-face `umask` guard, so NEMO averages
  `ww` from both T-neighbours at interface k+1 including one that is genuinely deeper/wet;
  legoESM's `face_active` min-rule mask (`vertical.py:1249-1252`) zeroes it. Explains
  **100.00%** of the active-only row (ratio 1.000, zero free parameters). **Real fidelity
  defect** — those u-faces are wet, so `dynzdf.F90:121` does not discard NEMO's result.
  active-only **3.0332e-02**, union 3.9993e-02. Fix in flight (gated option, default
  bit-identical, sign/conservation walk required).
- **`zu_frc` u (8.03e-3)** — drives the largest barotropic rows. **Nine candidates refuted.**
  Signature: broad envelope + damped ~34-row ripple, enhanced at the **periodic seam**
  (coincident with the sill). `zu_frc_write_ledger.py` (2026-07-30) enumerated EVERY
  `dynspg_ts.F90` write to `zu_frc`/`zv_frc` between :287-497 (self-check: table's write-line
  set == live-source regrep, exact match) and measured the two ACTIVE lines the six-piece
  budget omitted: `dyn_drg_init` (:381-382, dumped `drg_dump_zu_frc_inc.bin`) and the wind
  CENTRED term (:432). **Drag REFUTED as owner**: own-share 0.0001 (RMS 1.6e-10 vs zu_frc's
  3.0e-6), lego-vs-NEMO error RMS 1.56e-15 (machine-precision match, consistent with the
  already-recorded 0.999937/1.000278 gate row), corr with zu_frc's error -0.18/-0.15 (u/v).
  **Wind term UNMEASURED** — no NEMO-side increment-only dump exists (only raw `utau`); would
  need a new dump bracketing :420-433 the way :373-399 already brackets the drag call. On
  legoESM's side wind is structurally folded into the 3-D `du_dt` depth-mean (not a separate
  additive line, `surface_stress_implicit=False` on this card), so it isn't cleanly isolable
  without the NEMO-side dump either. Recommend **pause**: ledger now provably complete, drag
  ruled out, wind is the one remaining untested line but requires new instrumentation: out of
  scope for the unit harness (emergent solver behaviour).
- **`zdftke sh2`** — ESCALATION 1: exact transcription in, restricted-to-signal ratio 0.904.
  Family measured **climate-inert**, so parking is defensible.
- **`ldf_slp` ×4** — CONDITIONING-LIMITED, all three stopping-rule conditions verified.

## Measured strategic result — read before prioritising
The 5-year ACC acceptance run (protocol byte-identical, harness self-validated first) found
the campaign's **1e-6-class fixes are CLIMATE-INERT**: year-1 upper contrast 0.9082 → 0.9082,
ΔACC noise-level with two sign flips. The fixes change the state in the **wrong place**
(upper ocean, north of the channel band) while 80-98% of the missing thermal wind is sourced
**below 1000 m in the southern channel**. **⇒ Priority follows residual magnitude.** This does
not refute exactness-first — the 1e-2 rows remain untested — but it bounds the optimism.
Caveat: that run used `e3t=off` (the wrong 1-D ladder), matched to baseline for protocol
identity; a true-ladder acceptance is blocked on the restart-start instability.

## Human decisions — SETTLED 2026-07-30
1. `sh2` — **PARKED** (exact transcription in; family climate-inert).
2. Provenance — **re-measure only high-magnitude / fix-touched rows**; the machine check keeps
   the remaining 16 visible.
3. CONDITIONING-LIMITED — **counts as RESOLVED-WITH-NOTE** (the 4 `ldf_slp` rows).
4. `zu_frc` u — **PAUSED**, with one named resumption condition (below).
5. Push/PR — **done**: PR #1395, 91 commits.

---

# SESSION CLOSE 2026-07-30 — resume here

## Model health (verified, not assumed)
- **5-year from-rest DINO run completes clean** at the pushed HEAD with all six fixes active
  (ACC 55.578 → 65.540 Sv, census gates exact, ~11 min/yr). Card values were verified
  programmatically before launch, not assumed.
- **205 unit tests pass** across every touched module (`dino.py`, `tke.py`, `vertical.py`,
  `eos.py`, shortwave, box budget).
- **Non-NEMO recipes are byte-identical.** Every fix is a gated option, default = prior
  behaviour, each with a bit-identity test asserting `max|diff| == 0.0` (`==`, not `allclose`).
- **Known broken, and PRE-EXISTING**: restart-start on the true 3-D ladder
  (`LEGOESM_NEMO_E3T=both`) — max|u| 0.69 → ~3 m/s over 20 d. Predates this work (addendum 33).

## The strategic result — read this before spending anything
**TWO acceptance runs, both NULL.** Run 1: three 1e-6-class fixes. Run 2: the ZAD fix
(4.9e-3 → 3e-5, >100×, with a deep shelf-edge signature across 7255 columns). Year-5 ΔACC
**+0.0001 Sv**; upper contrast 0.8956 → 0.8954; the 5-year deep decay 0.4373 → 0.4370 untouched.
**Per-term transcription fidelity is not what costs the 25 Sv.**
**⚠ BUT BOTH RUNS USED THE WRONG LADDER** (`LEGOESM_NEMO_E3T` unset — 12.9% off below k=25),
kept for protocol identity with the baseline. **The ACC deficit is sourced below 1000 m, which
is exactly where that ladder is wrong.** So both nulls carry a background geometric error ~100×
the size of the fixes under test. **The nulls may be measuring the ladder, not the fixes** —
which is why the true-ladder instability is the campaign's critical path, not more gate rows.

## Live threads, with resume conditions
- **`zu_frc` u (8.03e-3)** — PAUSED. Nine candidates refuted. The **write ledger is provably
  complete** (`zu_frc_write_ledger.py`, self-check regreps live source): active writes are
  `:336` depth-mean, `:367` zu_trd subtraction, `:381-382` drag (measured, refuted),
  `:432` CENTRED wind. **RESUMPTION CONDITION NOW MET** — `wnd_dump_z{u,v}_frc_inc.bin` exists
  (batched rebuild `819ca1b59`). **One bounded comparison decides it**: lego's wind contribution
  to `F_slow_u` vs the dumped increment — RMS share, error corr vs the `zu_frc` error field,
  ripple/seam/asymmetry. Owns it → the largest row is solved. Doesn't → ledger exhausted, every
  line measured, close it as a bounded negative.
- **True-ladder instability** — the critical path. Eliminated by measurement: CFL, `ln_zad_Aimp`,
  `kappa_GM` magnitude, thin cells, slope cap, derived `gdept`, **GM-bolus discrete divergence**
  (2026-07-30: identical on both ladders, and structurally impossible — `nemo_eiv_bolus_transport`
  uses only `e2u`/`e1v`, no `e3` term), **abyssal slope-cap population** (flat, within ~10%).
  `use_gm_redi=False` (zeros κ_GM only, Redi untouched) still restores stability.
  **THE OPEN CONTRADICTION and the next cheap check**: κ_GM, slopes and bolus divergence are all
  ladder-insensitive, yet addendum 36 recorded the bolus entering the advecting flux **37% larger**
  on the true grid. Since `psi = κ × slope`, those cannot all hold. **Reconcile the 37%: is it
  real, and is it the same quantity?** (Five metric-identity incidents occurred on 2026-07-30 —
  treat any un-reconciled cross-script figure as suspect.)
- **zdftke composite** — real residual (corr 0.966; ~5% signal-weighted ratio), **INDEPENDENT of
  sh2** (substituting NEMO's own sh2 moved corr 0.9633 → 0.9656). Candidates, all now unblocked by
  the rebuild: buoyancy sink `p_avt*rn2` (`zdftke.F90:495`), `zmxlm` (`:814-819`), tridiagonal
  `zzd_up`/`zzd_lw` (`:485-492`).
- **MXL residual** — escalated; `tke_dump_rn2.bin` now exists. Re-walk with true `rn2`: expect
  0.0099 → ~1e-9 (confirming the rn2b-proxy explanation) or a real escalation.
- Also unblocked by `819ca1b59`: `wzv` row (`wzv_dump_ww_call1/2.bin` — **two calls per step**,
  use call2 for `tra_adv` consumers), `traldf_iso_lap` bracket, `dynzdf` stress bracket.

## Known harness flake — do not chase
Intermittent `buf_write` SIGSEGV in ~50% of bare NEMO runs, ASLR-sensitive, identical backtrace
in logs dated 2026-07-27. **Pre-existing, not ours.** Retry; registered dumps came from a clean run.

## Token economy (measured, in force)
Subagent lanes were **~95% of spend** (~30 lanes × 100–400k on 2026-07-30). In force: 3-hourly
cron (a FLOOR — fire manually for bursts), **ONE lane per iteration**, **explicit scope caps**
("do steps 1–2, report, STOP"), briefs point here instead of re-typing context.
**Protected: the adversarial reviews** (they caught a stale test encoding a bug as correct, a
wrong `Kmm` instruction, a missing mandatory `mask=`, a vacuous fp32 control) **and numeric
precision in any compression** — reconciliation needs old numbers exact and findable.
