# PRE-REGISTRATION — the implicit vertical solve's staircase face divisor

Written **before** either arm was launched.  #1455.  Branch
`fidelity/dino-basin-budget-1yr`.

## The defect, already measured (CONFIRMED, not a hypothesis)

`scripts/validate/ocean_fidelity/dino_1226/staircase_control_volume.py`, run at
HEAD `4a1704b43` on the shipped card `nemo_dino_kamm_mlf` with the twin's own
ladder (`e3t_mode="both"`), fp64:

* legoESM's cell-centre thickness is already zero below the sea floor
  (`ocean/vertical.py`, `h_partial = where(is_active, h_full, 0)`).
* `_apply_implicit_vertical_mixing` built its u/v-face thickness as an
  **arithmetic** cell→face mean of that already-masked field, so every face
  whose two columns have different bottom levels received **half a reference
  cell** on a level that exists on neither column.
* Against NEMO's own `hu_0 = SUM_k e3u_0 * umask` (mesh_mask.nc):
  **1396 of 9758** wet u-columns too deep, by up to **365.9 m**;
  **+0.27 %** at wall row 1, **+0.39 / +0.62 / +0.64 %** at rows 2/3/4,
  **+0.92 %** at row 20, **+0.84 %** at the equator.
* CONTROL: on the 8331 u-faces whose two columns have the SAME bottom level the
  two rules agree to **0.000e+00 m** — the bias is the staircase rule, not a
  convention offset.
* The min rule (`min_cell_to_uface` / `min_cell_to_vface`, the owner every
  other face thickness in this model already uses) reproduces `hu_0` to
  **0.000e+00 m on all 9758** columns.

**A handed premise that the measurement does NOT support, recorded here so it
cannot be quietly reused.**  The bias was expected to be wall-enriched.  It is
the opposite: the four wall rows average **+0.48 %** against **+0.84 %** over
the sampled interior rows, i.e. a wall/interior enrichment of **0.57×**.  The
meridional gradient runs the right way to overlap the deficit band but the
wall rows are the LEAST biased rows, not the most.

## The change under test

One owner for the pair.  `dz_u = min_cell_to_uface(dz_cell)` and
`dz_v = min_cell_to_vface(dz_cell, grid)` in
`_apply_implicit_vertical_mixing`, replacing the centred interpolations.  This
is the same rule the barotropic split, the PE tendency, the slow-forcing depth
average and the after-level reconcile already use, so the two sides of one step
stop disagreeing.  NEMO-faithful, not merely self-consistent: `e3u_0 = MIN(...)`
on the partial-step builder (`DOMAINcfg domzgr.F90:1166`), and the full-step
builder's arithmetic mean (`zgr_lib.F90:231`) is identical to the min on the
horizontally uniform reference ladder DINO's `ln_zco_nam=.true.` card has.

Side effect, named: `A_v`'s v-interpolation no longer fuses with `dz_v`'s halo
exchange, so this stage costs one extra send/receive pair per partition cut per
step.

## Where the divisor can reach the climate

`zdf_baroclinic_only=True` and `zdf_drag_in_matrix=True` on the shipped card.
The stage subtracts the column mean before the solve and adds **the same** mean
back after, so the split cancels exactly *except* through the two drag terms,
which make a constant no longer a null vector of the solve: the bottom-cell
diagonal, and NEMO's barotropic-drag right-hand-side correction
(`dynzdf.F90:156-159`), both of which multiply that column mean.

## Registered prediction (PLAUSIBLE — arithmetic below, done before the run)

The divisor is too deep by δ ≈ 0.0027 at wall row 1 (0.0048 mean over rows 1–4),
so the removed column mean is too small by the same fraction and the fix
increases it by δ.  NEMO quadratic drag with `cd0=1e-3` and
`sqrt(u²+v²+ke0) ≈ 0.05 m/s` gives a bottom rate `r_eff ≈ 5e-5 m/s`; over a
wall-row column depth of 2231 m the barotropic drag rate is `2.2e-8 s⁻¹`, i.e.
a dimensionless **0.17** over the 90-day window.  A δ error in the mean the
drag acts on therefore changes the realised barotropic velocity by
`δ · 0.17 · U_bt ≈ 0.0048 · 0.17 · 5e-3 m/s ≈ 4e-6 m/s`, against a measured
depth-uniform deficit of **4.6e-4 m/s** — about **0.9 %** of it.

* **Direction.** The band's depth-uniform velocity is westward (bottom cell
  −6.5e-3 m/s in legoESM against −5.0e-3 m/s in NEMO).  More damping shrinks
  its magnitude, i.e. moves it eastward, i.e. moves `G4` in the **improving**
  (less negative) direction.
* **Size.** `G4` = −0.288 Sv at day 90 in the current baseline.  Scaling by the
  velocity ratio above: **ΔG4 ≈ +0.003 Sv**, registered with an
  order-of-magnitude band **0.0003 – 0.03 Sv**.

## Decision rule, fixed now

Metrics, read **jointly in one table** (the viscosity-ablation lesson: transport
alone is tunable by a knob that breaks the density structure):

1. `acceptance_gate_90d.py --level 5` — ACC [Sv], upper contrast, deep contrast,
   surface sigma max, surface sigma mean, against floors
   `{acc 0.091, up 1.1e-4, deep 4.5e-5, smax 9.5e-5, smean 9.5e-5}`.
2. `wall_visc_ablation_gap.py` — `G4` (wall rows 1–4), the basin rows 0–13 sum,
   and the wall sea-surface excess `A_wall` [mm].

* **CONFIRM the divisor as the deficit's owner** requires `|G4|` to shrink by
  **more than 40 %**, i.e. `|G4| < 0.173 Sv`.
* **REFUTE as owner** if `|ΔG4| < 0.05 Sv`, which is what the arithmetic above
  predicts by two orders.
* **My registered prediction is FALSIFIED** if `ΔG4` is negative (the gap grows)
  or if `|ΔG4| > 0.03 Sv` in either direction.
* **Ship gate, independent of the above:** the four acceptance-gate metrics must
  not cross their floors in the degrading direction.  The channel is the crown
  jewel.  The fix ships on correctness (a control volume that carries water on a
  level neither column has, and a divisor that disagreed with the barotropic
  split inside one step) whatever the climate number does — the A/B decides
  attribution, not whether the bug is a bug.

## Harness (one variable, everything else byte-identical)

```
# arm A — pre-fix, clean tree at HEAD 4a1704b43
LEGOESM_NEMO_E3T=both JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455_staircase_divisor/twin90_armA_pre.npz \
  --days 90 --bridge-before --save-3d

# arm B — post-fix, clean tree at the fix commit; identical in every other flag
... results/dino_1455_staircase_divisor/twin90_armB_post.npz ...

# scoring
acceptance_gate_90d.py <arm>.npz --level 5
python -m scripts.validate.ocean_fidelity.dino_1226.wall_visc_ablation_gap \
  twin90_armA_pre.npz twin90_armB_post.npz     # columns in argument order
```

Both arms run from a **clean tracked tree** so `provenance_gate()` passes
without `LEGOESM_ALLOW_DIRTY`; arm A is run before the fix is committed and arm
B after, and each log stamps its own HEAD.
