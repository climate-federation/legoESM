# MG2 / Morrison: gradient-only gate on absent hydrometeors

Evidence for the fix in `_warm_rain.py` (`div_gated_tangent`,
`mg2_absent_gradient_gate`), `morrison.py` (snow-number aggregation limiter)
and `integration.py` (gate at the Morrison entry of the hydrostatic
microphysics wrapper).  The forward model is bitwise unchanged; only the
reverse pass changes.

Setup of every run below: production AMIP deck (`config/amip/amip_production.yaml`:
MPAS hydrostatic L32, CAM6 suite, Morrison/MG2 with two-moment snow, CLUBB,
`cld_macmic_num_steps: 3`, dt 112.5 s, physics every 16 steps, radiation every
32) at `--resolution 3` (642 columns), ERA5 1979-01-01 IC, fp64 runtime,
RRTMGP in fp64 (diagnostic), A100 80 GB unless marked CPU.  Gradients are of a
fixed random projection of the end state (or of the physics-call output)
w.r.t. T and q_v (plus q_i / N_s where stated); FD is central along a fixed
random direction.  Instruments: `scripts/validate/production_ad_map.py`
(`--stage step|physics|components|eqns` with `--stash-grads`,
`--stash-modules`, `--load-state`, `--h-sweep`) on branch
`feat/pll-mpas-checkpointing`; they land with that branch's PR.

## 1. The defect

32-step window (one hour, two physics calls), GPU: AD -9.9e84 vs FD -8.0e6.
The per-step adjoint record (identity custom_vjp tap writing max|cotangent|
per state leaf into a `jax.ref`, `runs/mpas_seg_stash-0-28780080.txt`)
puts the whole amplification inside ONE call: the step-16 physics step
(x1.5e86, landing on cloud ice q_i).  Dynamics steps change the adjoint by
<= 10 % per step.

Same physics call, bit-identical inputs (state saved to .npz, sha256 digest
printed and matched), both backends (`runs/mpas_x_*`):

| state \ backend | GPU AD | CPU AD | FD (both) |
|---|---|---|---|
| advanced on GPU | 1.3e75 | 1.2e58 | 1.95e3 |
| advanced on CPU | 2.0e77 | 2.5e58 | 1.95e3 |

Model defect on both backends; the backend changes only its size.

Module taps inside the call (input and output of every module call;
CLUBB and Morrison alternate three times per call): the cotangent jumps
inside the 2nd and 3rd Morrison calls (x2e87, x2e31), on the snow number
N_s at cells where N_s == 0.  Per-equation reverse pass of each Morrison
call (`eqn_trace`: forward and reverse one jaxpr equation at a time; summed
per-equation gradient == `jax.grad` to roundoff on every leaf;
`runs/mpas_eqns_micro-0-28877034.txt`): the largest single-equation gains are
the MG2 conservation limiters, ratios (available/dt + sources) / sink, at
stocks and sinks of 1e-223 .. 1e-291 -- far below MG2_QSMALL = 1e-18:

| equation | gain | primal |
|---|---|---|
| snow-number aggregation limiter (`morrison.py`) | x1.2e28 | 2.6e-223 |
| q_i conservation ratio (`mg2_qi_conservation_ratio`) | x2.5e13 | 2.1e-291 |

MG2 switches its processes off below qsmall but keeps computing with the
sub-threshold amounts in these places; the derivative of an O(1) ratio of two
1e-250 amounts is ~1/amount.  Correct for the code, meaningless physically,
and it does not cancel in floating point.

## 2. The fix (gradient only)

1. `div_gated_tangent(x, y, live)`: `x / y` forward bitwise; tangent
   `(dx - (x/y) dy) / y` where `live`, exactly 0 elsewhere.  Used in the two
   limiters, `live` = stock (q_avail; snow mass q_s) >= MG2_QSMALL.
2. `mg2_absent_gradient_gate(x, mass)`: value unchanged, held constant for
   the gradient where `mass < MG2_QSMALL`.  Applied at the Morrison entry of
   `_make_hydrostatic_microphysics` to every hydrometeor mass and, keyed on
   its mass, to every number concentration.

| check (GPU, step-16 state / 32-step window) | before | limiters only | limiters + entry gate | FD |
|---|---|---|---|---|
| one physics call (full physics), AD | -1.9e70 | -1404.32 | -1404.32 | -1778.28 |
| one physics call (no radiation), AD | -2.2e70 | 2155.34 | 2155.34 | 1952.02 |
| 32-step window, AD | -9.9e84 | -4.7e72 | **-7.98028e6** | -7.98921e6 |
| 32-step adjoint jump at step 16 | 1.5e86 | 1.8e80 | 1.8e12 | |
| d/d N_s at Morrison call 2 / 3 input | 1.2e122 / 1.9e99 | same | 2.1e3 / 3.0e3 | |

The 32-step GPU gradient is FD-consistent to 1.1e-3 after the fix, and its
step-16 jump equals the CPU's pre-fix profile (1.820e12) to 4 digits.

The remaining 10-27 % one-call mismatch is not this defect: the entry gate
leaves that AD identical to 7 digits.  CLUBB alone showed AD/FD = 6.1 at the
default FD step; an FD step sweep (`runs/mpas_kink_*`) converges to AD to
5 digits (1.0000 at 1e-4 and 1e-5 of the default step), i.e. a kink in the
temperature direction, not an adjoint error.  The one-call mismatch was not
swept (PLAUSIBLE: the same FD artefact).

## 3. Known differences and limits

- Gradient only, forward bitwise (tested).  In gated cells the gradient is
  that of a function in which sub-qsmall hydrometeor amounts cannot influence
  anything; it is discontinuous at MG2_QSMALL.
- Holding an input constant moves rather than removes sensitivity: where the
  scheme deletes a sub-qsmall amount, d(q_new)/d(q_old) reads 1, not 0.
- Scope: the ENTRY gate acts on Morrison in the hydrostatic/MPAS wrapper only.
  The two LIMITER gates live inside `morrison_microphysics`, so every Morrison
  caller in MG2 mode (SCM, nonhydrostatic, training harnesses) gets the
  limiter change.  The gated q_i also changes the gradient of the wrapper's
  ice cloud fraction in absent-ice cells.
- `mg2_qi_conservation_ratio` keys its gate on its stock against the mass
  threshold.  At the ice-NUMBER limiter the stock is N_i (a number), so the
  gate practically never closes there -- deliberately: keying it on q_i was
  MEASURED to restore the 1e75 adjoint (section 5).  The deposition limiter's
  stock is a vapour excess, where the gate only bites at the max(., 0) kink.
- Gating a limiter's FACTOR (not the limited product) removes its half of the
  `sink * (stock / sink)` cancellation, so the limited sink keeps
  `factor * d(sink)`.  That is benign at the two gated limiters (measured) but
  is the mechanism of the regression above; a product-level rule would be the
  robust form (not done).
- Not covered: tiny intermediates created inside one Morrison call (only the
  two named limiters gate those), and N_s where q_s >= qsmall but N_s ~ 0:
  d/d N_s at Morrison call 1 is still 2.5e92 at such a cell
  (`runs/mpas_gate_phys16-0-28897363.txt`).  It did not affect the 32-step
  window (which differentiates T and q_v); a loss on N_s would see it.
- The q_i cotangent entering step 16 is still 9.7e15 (CPU had the same
  before any fix).

## 4. Tests

`tests/atmosphere/microphysics/unit/test_mg2_limiter_gated_tangent.py`,
`tests/atmosphere/microphysics/unit/test_mg2_absent_gradient_gate.py`:
forward bitwise vs the plain division / ungated wrapper; planted sub-qsmall
stock: pre-fix derivative > 1e25, post-fix 0; present cells: gradient equal
to the plain one (rtol 1e-12); non-finite cotangent in a gated cell does not
leak; `check_grads` on the gated division; non-vacuity: each gate removed ->
the absent-cell test fails.  Microphysics suite + hydrostatic microphysics
tests: 984 passed; the 6 failures fail identically on unmodified main
(5 fixture-input regenerations differing in the last bit on this node type,
1 segfault in `test_morrison_nr_consistency_limiter`).

## 5. Review (fresh-context Claude agent; codex and GLM unavailable here -> UNREVIEWED by the mandated pair)

Findings, all CONFIRMED by the reviewer with snippets, and what was done:

| finding | action |
|---|---|
| second-order (Hessian) NaN in gated cells: the tangent divided by the gated tiny y | tangent divides by `where(live, y, 1)`; Hessian test fp64 + fp32 |
| per-volume N_c/N_r gated before the density conversion: absent number reached T, p, q_v through rho | gate applied after the conversion as well |
| CAM6 prognostic-Nc activation read an ungated N_c | reads the gated per-mass N_c |
| wrapper test used one presence mask for all species: a mis-keyed number gate passed | own mask per species, N_s included; mis-keying N_c->q_r etc. now fails the test (checked) |
| ice-number limiter keyed its gate on N_i (a number) against a kg/kg threshold | tried keying on q_i: MEASURED regression, one physics call back to AD 1e70 vs FD 1.8e3 and the 32-step window to 4.4e87 (bisection: reverting only this change restores AD -1404.32 vs FD -1778.28). Reverted; recorded as a known difference |
| deposition limiter gate undocumented; limiter scope wider than the README said; gradient-shift effect undocumented | documented (section 3) |

Confirmed fine by the reviewer: forward bitwise in fp32/fp64, eager and jit;
first order under vmap/jit/scan/remat/jacfwd with a boolean `live`; no
non-finite leak from a gated cell; the species mapping; the limiter tests'
non-vacuity; no new constants or private imports.
