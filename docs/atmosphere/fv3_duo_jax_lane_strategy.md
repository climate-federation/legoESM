# FV3 duo-grid JAX lane — conversion strategy

Status: **strategy, pre-implementation.** Written 2026-08-13. This document is
the contract the implementation is held to; every gate named here must exist in
code before the corresponding tier is claimed.

## 0. What is being finalized

The faithful FV3 duo-grid port in this repository is **certified but not
runnable as a model**. Its shallow-water and 3-D chains (`d_sw1…d_sw6`, `c_sw`,
`d2a2c_vect`, `divergence_corner`, `ext_scalar`/`ext_vector`, the NH Riemann
solvers, `fv_mapz`, `fv_tracer2d`) are loop-faithful **NumPy fp64**
transcriptions of the pinned Zenodo 8327578 `symmetryclean` Fortran, verified
against Fortran dumps down to 1.19e-9 (hydrostatic, one step) — but nothing
outside `tests/` and `scripts/validate/` imports them. They are a test fixture.

The deliverable of this campaign is the **JAX lane**: the same chain,
jit-compilable, differentiable, GPU-capable, wired into the dycore registry, and
scored on our own shallow-water and 3-D/non-hydrostatic test cases against the
oracle's own reference runs.

Two things are therefore in scope that were not before:

1. every hot-path kernel gets a JAX twin;
2. the twins are assembled into an end-to-end `fv_dynamics` step and run.

## 1. The two source papers, and what we take from each

**Koldunov et al., *An Ocean Model Ported by a Large Language Model: Experience
and Lessons from FESOM2 (Fortran to C to C++/Kokkos)*, arXiv:2606.11356.**
73,991 lines of Fortran → 20,010 lines of C (10 days with commits) → 30,663
lines of C++/Kokkos (7 days). What we take:

- **The literal-translation rule.** §4.2: *"port the Fortran line by line and
  never to simplify, approximate or 'improve' it. The only changes allowed are
  mechanical."* The justification is not style — it is that the Fortran is the
  verified specification, so **any divergence of the port is a port bug by
  definition**, which collapses the debugging search space. Their first attempt
  at this port failed precisely because small improvements accumulated
  (linearized expansion coefficients, a closed-form replacement for a KPP lookup
  table, a volume ratio where an area ratio belonged) — each producing plausible
  output and a long hunt.
- **The constants rule.** §4.2: *"any statement about what the Fortran does with
  a constant must quote the exact file:line and the literal value read there,
  never one recalled, rounded, or copied from a comment."* Adopted after a
  comment stating the wrong value hid a numerical instability for several
  sessions.
- **The live twin.** §4.4: when a kernel is rewritten, the previous
  implementation stays in the tree as its *twin*; an env-var-gated check runs the
  twin on the same live state each call, records the max abs difference, and
  **restores the new implementation's output** so the run is undisturbed. On the
  exact back-end the acceptance threshold is a difference of **zero**. It catches
  more than loop structure: a mistyped coefficient copied into the rewrite makes
  the two disagree.
- **The failure-mode catalog** (§7.1), reproduced in §8 below.

**Koldunov et al., *FESOM2-JAX v1.0: a differentiable shadow of the
ocean–sea-ice model FESOM2, cast onto GPUs*, arXiv:2608.01546.** The Python/JAX
shadow, ported "kernel by kernel" with Claude Code driving Opus 4.8 and Fable.
What we take:

- **The shadow doctrine** (§1): a shadow is *"neither a fork nor a successor; it
  is verified by running alongside the original on identical inputs and comparing
  outputs."* The Fortran remains the single source of truth. Our JAX lane is a
  shadow of a shadow, and §2 below says which authority each hop answers to.
- **The proximate reference is the previous port, not the Fortran** (§2.4):
  *"The proximate reference for FESOM2-JAX was not the Fortran source directly,
  but the instrumented serial C translation … every JAX kernel was gated against
  the C reference's dumps."* This is exactly our NumPy lane's role, and it is why
  **one authority per hop** (§2) is not a local invention.
- **Tolerance is set by what the kernel does** (§2.4): pointwise kernels and
  kernels that only read neighbouring points are held to *"about 1e-15 relative,
  which is the level of rounding alone"*; kernels that **accumulate** — flux
  assembly onto control volumes, global sums — are reassociated by JAX and are
  held to *"about 1e-12"*.
- **Multi-step replay** (§2.4): single-step tests cannot see state threading.
  Replay tests are required for the history/carry variables specifically.
- **Gradients checked kernel by kernel against finite differences, at states away
  from the switching points at which the model is not differentiable**, plus
  through short full-model integrations. Their suite: 78 test modules, ~640
  tests, 41 modules exercising gradients.
- **The localization argument** (§2.4): *"Because every kernel had already been
  verified on its own, a drift that appears only over climate timescales cannot
  originate inside a kernel and must instead arise where two kernels are joined.
  Checking which of the conservation budgets fails to close then locates the
  defect."* This is the diagnostic doctrine for tier 4/5 failures below.
- **The chain closes back on the original**: their hindcast compares the
  assembled JAX model against **Fortran FESOM2**, not against the C reference it
  was built from, so an infidelity introduced at *either* translation step
  appears. Our tier 5 does the same against the oracle's reference runs.

## 2. The authority chain — one authority per hop

```
  Fortran oracle            NumPy fp64 lane              JAX lane
  (Zenodo 8327578    --A-->  fv3_native_*.py    --B-->  fv3_jax_*.py
   symmetryclean)            loop-faithful               functional, jit,
                             fp64, certified             differentiable
        |                                                     |
        +---------------------- C ----------------------------+
                     end-to-end solution comparison
```

- **Hop A — Fortran → NumPy.** Already done for the hot path. Certified against
  instrumented Fortran dumps; hydrostatic one-step floor 1.19e-9, NH one-step
  delz 6e-8 / pt 1e-8 / delp 5e-8 / w 3.4e-7 m/s. **This campaign does not
  re-open hop A.** Where a NumPy twin does not yet exist for a routine the JAX
  lane needs, hop A is done first, against Fortran, before hop B.
- **Hop B — NumPy → JAX.** Every JAX kernel is gated against the **NumPy lane**,
  never against Fortran directly. Tolerances per §4.
- **Hop C — assembled model → oracle reference runs.** The end-to-end JAX
  `fv_dynamics` is scored against the oracle's own `atmos_daily.nc` output on
  matched cases, grids and days. This is the only tier that can catch an
  infidelity introduced at *either* hop, and it is the tier the campaign is
  finished at.

**What hop B cannot see, stated up front.** Certifying JAX against NumPy is
blind by construction to any defect the NumPy lane already has: the JAX twin
reproduces it faithfully and both agree to 1e-15. Only hop C sees those, and
only if hop C is sensitive to them. Two consequences that are accepted, not
solved:

- The **1.19e-9 hydrostatic one-step residual** is exactly such a defect. Both
  instrumented mechanisms for it are refuted and it remains open on the NumPy
  lane. The JAX lane inherits it. Any JAX number at or below that floor is
  reported as *inheriting an unexplained residual*, never as agreement (§10).
- A defect small enough to hide under hop C's solution-tier noise but large
  enough to matter over a long integration is the **latent-bug class** of §8,
  and the only guard against it is the one FESOM2-C names: run tier 5 at the
  **production time step and configuration**, not at a short smoke test.

The compensating asset is FESOM2-JAX §2.4's localization argument: *because
every kernel is verified on its own, a drift that appears only over many steps
cannot originate inside a kernel and must arise where two kernels are joined.*
That is why tier 3 (multi-step replay) and the conservation-budget checks are
not optional extras — they are the only tiers that interrogate the joins.

## 3. Rules

**R1 — Literal translation.** The NumPy lane is the specification for the JAX
lane. The only permitted transformations are mechanical:

- an explicit `for i` / `for j` loop over grid indices becomes a vectorized slice
  expression **over the identical index window** — **but only after a dependence
  and alias audit** (R1a below);
- an explicit *recurrence* becomes `lax.scan`, **on whatever axis it runs** —
  never `associative_scan` and never `cumsum`, because both reassociate the
  floating-point sum and break the parity contract (already the established
  doctrine in `fv3_nh_core.py`);
- in-place mutation of an argument becomes a functional return (§ R4);
- a static Python `if` on a config value stays a Python `if`; an `if` on array
  data is handled per R1b below — **not** by a blanket `jnp.where`.

Anything else — algebraic simplification, fusing two loops that the Fortran
keeps separate, "obviously equivalent" reorderings — is forbidden. If a
transformation looks like an improvement, it is out of scope by definition.

**R1a — Vectorization requires a proven-independent loop.** An `i`/`j` loop may
be sliced only after showing that **no iteration reads a location an earlier
iteration writes**, including through aliased views (the NumPy lane's `fort`
1-based adapters are views onto the same buffer) and through halo/corner
storage. *"Identical index window" is necessary and not sufficient.* Every
vectorized loop carries a one-line dependence argument in a comment at the site.
Where the dependence is real, it stays an ordered `lax.scan` — the axis is
irrelevant. Three sites in this core that will bite:

- the NH `w_limiter` (`fv3_native_mapz.py:1014-1044`) clips downward then
  upward and **reads the just-updated adjacent level**, so the spill one level
  sees depends on the previous level's clip. A bulk `.at[..., k+1].add(...)` is
  wrong; it is two ordered scans.
- tracer subcycling (`fv3_native_tracer2d.py:432-460`) overwrites `dp1` with
  `dp2`, refreshes the tracer halos, and the next iteration reads those new
  values.
- `_rezone` (`fv3_native_mapz.py:599-650`) carries `k0` across target layers.

The narrower claim that `c_sw` and `d_sw1…6_duo` are already functional (they
copy every input at entry and return dicts) **is** sound and was independently
confirmed in review — but it covers only those routines, not `mapz` or tracer
transport.

**R1b — `jnp.where` is a select, not lazy control flow.** Both branches are
traced and evaluated. A division, `log`, `sqrt`, or out-of-range index
arithmetic in the **unselected** branch produces NaN/Inf, and under reverse-mode
AD a NaN from the inactive branch **contaminates the gradient of the branch that
was selected**. The rule is therefore:

- `jnp.where` only when **both branches are total and finite over every admitted
  input**;
- otherwise mask or sanitize the dangerous operand *before* the select (the
  double-`where` idiom), or use `lax.cond` where the predicate is scalar and the
  branch must genuinely be lazy;
- state per site which was used and why.

The live hazard is `_rezone`: its contained-cell and spanning-cell formulas
perform *different divisions* and are valid only after a successful interval
match (`fv3_native_mapz.py:619-669`), so flattening them into a `where` over all
candidate layers evaluates formulas outside their intervals. The limiter spill
formulas dividing by the next layer's `d2` (`:1021-1040`) have the same shape.

Note also what `where` does **not** give: it returns the derivative of the
selected branch and no derivative of the predicate. That is a legitimate
piecewise derivative, but it is not a derivative through the discrete upwind or
interval decision, and §7 must not pretend otherwise.

**R2 — Constants and claims cite file:line.** Every constant in a JAX kernel is
either imported from the NumPy twin's module or carries a comment with the
oracle `file.F90:line` and the literal value read there. No value recalled from
memory, rounded, or taken from a comment. (Repo-local reinforcement: a code
comment or doc is a pointer, never a citable fact.)

**R3 — Structural preservation.** The oracle's `d_sw` is **six public routines**
— `public :: c_sw, d_sw1, d_sw2, d_sw3, d_sw4, d_sw5, d_sw6, fill_4corners,
del6_vt_flux, divergence_corner, divergence_corner_nest` (`sw_core.F90:74`) —
and its k-loop is **four separate `do k=1,npz` nests** (`dyn_core.F90:744, 914,
1066, 1218`) precisely so the two duo flux-averaging barriers have somewhere to
live:

- `dyn_core.F90:872` — `mpp_get_boundary(fxx_delp, fyy_delp, …,
  gridtype=CGRID_NE)` (`:874`), after `d_sw1` (`:831`), followed by the
  `0.5*(mine + neighbour)` blend;
- `dyn_core.F90:984` — `mpp_get_boundary(tempfx1, tempfy1, …,
  gridtype=BGRID_NE)` (`:986`), on the B-grid corner velocities loaded from
  `ubbtemp`/`vbbtemp`.

Three further `mpp_get_boundary` calls are **commented out** (`:1037`, `:1141`,
`:1189`), so exactly two barriers are live. Stage call sites, pinned tree:
`c_sw:489`, `p_grad_c:629`, `d_sw1:831`, `d_sw2:950`, `d_sw3:961`, `d_sw4:1102`,
`d_sw5:1107`, `d_sw6:1256`, `one_grad_p:1531/1540`, `nh_p_grad:1542`.

A monolithic `d_sw` **cannot express the barriers**. The JAX lane keeps the
six-stage decomposition. The existing monolithic JAX `_d_sw_native`
(`fv3_sw_core.py:3359`) is a different, research solver and is not the target of
this port.

Also live under duo, verified on the pinned tree: `duogrid = .true.` forces
`bounded_domain = .true.` (`fv_arrays.F90:1512`,
`Atm%gridstruct%bounded_domain = Atm%flagstruct%regional .or.
Atm%neststruct%nested .or. Atm%flagstruct%duogrid`), yet **three call sites
hardcode `.false.` and override it** — `d2a2c_vect` (`sw_core.F90:151`),
`ytp_v` (`:1316`), `xtp_u` (`:1374`), each with the original `bounded_domain`
call commented out directly above. And all four `fill_corner_region` call sites
are commented out (`sw_core.F90:1747, 1755, 1763, 1764`): adding duo corner
fills to `d_sw5` moves *away* from the oracle.

**R7 — Cite the PINNED oracle tree, and only it.** Three copies of the
`symmetryclean` source exist on this machine, and they are **not** all the same
file:

| tree | `dyn_core.F90` | status |
|---|---|---|
| `/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean` | md5 `e5a5fab9…`, 3128 lines | **THE oracle** — what the verbatim extracts cite |
| `/burg-archive/glab/users/pg2328/Code/FV3/duogrid_symmetryclean/atmos_cubed_sphere-symmetryclean` | md5 `e5a5fab9…`, 3128 lines | identical to the pinned tree |
| `/burg-archive/glab/users/pg2328/fv3_recon/duo_model/atmos_cubed_sphere-symmetryclean` | md5 `0a5df09a…`, **3230 lines** | **INSTRUMENTED working copy, +102 lines** |

`sw_core.F90` is byte-identical across all three (md5 `9e30c61d…`), which is why
the register's `sw_core` citations survive; `dyn_core.F90` is not.
`fv3_duo_gap_register_2026-08-06.md` instructs the reader to "use the **full**
tree" (the instrumented one) and its two barrier citations — `:932` and `:1049`
— are that copy's line numbers, while its k-loop citations (`744, 914, 1066,
1218`) are the pinned tree's. The register mixes two files under one name. The
barrier numbers above are re-derived here against the pinned tree; the register
is corrected in the same commit.

**R4 — Functional, no aliasing.** The NumPy lane mutates `pe`/`w2`/`dz2` and
friends in place; every JAX twin takes the same operands and **returns** them.
No output parameters, no `donate_argnums` anywhere in this lane (it conflicts
with reverse-mode AD, which is the point of the lane).

**R5 — Port the EXECUTION, not the call text.** A Fortran call whose body is
guarded on `present(...)` arguments the caller does not pass is a **silent
no-op**, and transcribing it faithfully-looking is a defect. This rule is not
imported from the papers; it was earned here, and it was worth four orders of
magnitude: the hydrostatic floor sat at 9.7882e-6 because the port executed
`fill_corners_agrid_y(f0)` where the oracle's `fill_corners(f0, npx, npy, YDir)`
(`test_cases.F90:800`) does nothing at all — `fv_mp_mod.F90:1032-1105` guards the
whole body on `present(BGRID)`/`present(AGRID)`, neither of which is passed.
Deleting one call gave 1.1866e-9.

**R6 — Read what the deck RESOLVES.** The pinned deck runs `adiabatic = .true.`
and `nwat = 0` (`run_hydro_1step_gfs` logfile `:110`, `:62`), and
`init_hydro.F90:138-141` makes `zvir` identically zero under that flag. The
moist / `q_con` / `moist_cv` path is therefore **structurally dead in this
configuration** — not untested, never executed. Certifying it needs a new
Fortran deck, not more porting. Same class as R5: read the resolved namelist,
not what the source *can* do.

## 4. Tolerance policy

Tolerances follow FESOM2-JAX §2.4 — **set by what the kernel does**, and in
every case **pinned to a measured value** with the measurement recorded in a
comment next to the bound. A round number inherited from another gate is a
guess, not a tolerance.

| kernel class | examples here | bound |
|---|---|---|
| pointwise / neighbour-reading | `edge_profile`, EOS-like algebra, the projections in `c2l_ord2` | ~1e-15 relative (rounding alone) |
| accumulating / reassociated | flux divergence sums, `pe` rebuild, area-weighted global sums | ~1e-12 relative |
| sequential recurrence under `lax.scan` | `sim1_solver`, `riem_solver3` Thomas sweeps | ~1e-15 own-scale; cancellation-amplified outputs (`pe`, `w2`) sit at ~3e-15 of their own small scale — record which |
| **branch-switching** | `xppm`/`yppm`/`xtp_u`/`ytp_v` limiters, `_rezone`, the NH `w_limiter` | **no rounding-scale bound is meaningful** — see below |

**The fourth class is why the first three are not enough** (codex finding 2,
BLOCKER). Several of this core's most important routines are simultaneously
pointwise, accumulating *and* branch-switching, and near a switching surface a
rounding-level difference flips the branch and produces a discrepancy far above
1e-15:

- `_rezone` (`fv3_native_mapz.py:594-669`) — one ULP at an interface selects a
  different layer *and* a different formula.
- the NH `w_limiter` (`:1014-1044`) — a conditional sequential redistribution;
  clipping one level changes what the next reads.
- `deln_flux` / `del6_vt_flux` (`fv3_native_d_sw.py:1091`/`:1150`, `:1707`/`:1753`)
  — repeated stencil passes over `nord`, so error is locally accumulated and
  window-sensitive, not one flux-divergence sum.
- `xppm`/`yppm` (`fv3_native_d_sw.py:260`/`:656`) — branch-heavy limiter
  pipelines, not innocuous neighbour-reading kernels.

For these, a gate must record **which regime its fixture sits in**, and a
branch-equivalence check (do both lanes take the same branch?) carries the claim
rather than the numeric bound.

Standing riders:

- **A bound is an error MODEL, not a multiplier on one sample.** Per output:
  absolute *and* relative tolerance, plus a **scale floor** so a near-zero
  reference does not make the relative bound meaningless. `measured × (3…10)`
  from a single fixture is not defensible on its own — and if the measured
  maximum is exactly zero, multiplying pins the bound to zero and makes harmless
  backend rounding fail.
- **Fixtures are a matrix, not a sample**: grids, faces, halo rings, every
  supported `hord`/`nord`/`kord`, vertical depths, remap ties, and both sides of
  each limiter regime. The bound comes from the worst observed case with a
  margin justified by operation count and conditioning.
- **Assert both hops**: JAX-vs-NumPy *and* jit-vs-eager. Evidence in a comment
  does not fail when the code regresses.
- **A residual at the tolerance floor is UNEXPLAINED, not agreement.** Say so.

Where a check's power does not depend on tolerance tightness — conservation and
budget closure, analytic invariants, equivariance, branch equivalence, N-step
growth judged against the 1-step floor, pattern correlation — prefer it.

## 5. The validation ladder

**Every tier is an executable specification.** A tier that says only "bounded
growth" or "solution-tier" can be claimed after observing almost any finite
number, which is not a gate (codex finding 8, BLOCKER). Each row below names the
fixtures, the fields, the window, the norm and a numeric criterion; where a
number is still to be measured it is marked `TOL-PENDING` in the code and the
measurement job censuses them before anything ships.

| tier | what is compared | against | acceptance |
|---|---|---|---|
| **0** | shapes, dtypes, **declared read/write window and minimum halo width** | `jax.eval_shape` + static preconditions | exact shapes; every operand f64 or `TypeError` at entry; a sentinel written outside the declared write window fails; the minimum `ng` for the requested `hord`/`nord`/stagger is asserted, not assumed |
| **1** | one JAX kernel, one call | its NumPy twin, same live inputs | §4 error model per output (abs + rel + scale floor), over the §4 fixture matrix; for branch-switching kernels, **branch equivalence** as well as the numeric bound |
| **1g** | one JAX kernel's gradient | finite differences | the three-part gate of §7 — smooth-region order-2, one-sided at each switching surface, and adjoint consistency |
| **2** | jit vs eager, same kernel | itself | §4 tolerance, plus a compiled-path/cache-count check. Divergence means a tracer bug **or** compiler reassociation, a static-argument mistake, or NaN-sensitive arithmetic — diagnose, do not assume |
| **2c** | **conservation and invariants** | analytic | see the conservation gate below — tolerance-independent, so it does not weaken as bounds are tuned |
| **3** | N-step replay of the assembled chain | NumPy lane, same IC | per-field growth vs the 1-step value, with a declared allowed amplification per N; the existing N-sweep gives the shape (hydro 9.79e-6 / 1.42e-5 / 9.56e-5 at N = 1/3/10) |
| **4** | one full `fv_dynamics` step, all fields | NumPy lane end-to-end | per field, per face, owned window and halo scored separately, pinned |
| **5** | 5–9 day integrations | **oracle `atmos_daily.nc`** on matched case/grid/day | field-by-field, see below |
| **5a** | one step, stage boundaries | **Fortran directly** | a periodic audit of hop A — see below |

**Tier 5 needs field-by-field criteria, not two headline metrics** (codex finding
1, BLOCKER). W2 `max|v|` and the J&W surface-pressure growth are *fingerprints*;
they are not sensitive to everything that can be wrong. A realistic survivor:
a NumPy defect confined to tracer remapping — wrong tracer ordering, a one-cell
seam error, a non-conservative `map1_q2` branch — which the JAX lane reproduces
exactly, and which leaves both headline numbers unchanged because passive
tracers do not feed them. Vertically compensating layer tendencies can likewise
be invisible in `p_s` while the temperature profile is wrong. **Five to nine days
does not make an insensitive observable sensitive.** Tier 5 therefore scores
every carried prognostic — `u`, `v`, `delp`, `pt`, `w`, `delz`, each tracer —
with vertical/profile and seam-localized norms alongside the global ones.

**Tier 5a exists because the localization argument is weaker than §2 claimed.**
"Every kernel agrees with NumPy" proves a drift does not originate in hop B; it
does **not** prove the NumPy kernel is right. So the campaign keeps at least one
direct Fortran-vs-JAX one-step comparison at stage boundaries as a standing
audit of hop A, even though NumPy remains the ordinary per-kernel authority.

Tier 5's cases:

- **Shallow water (2-D):** Williamson-2 at α=0 and α=45, `hord` 5/6/8/10, C48
  through C192 as budget permits; case-6 Rossby–Haurwitz C48; case-8 colliding
  modons C48 at α=0 and α=45. Of the 42 Zenodo run directories, every case ships
  a matched duo **and** plain arm (21 pairs), so the duo/plain ratio is a
  one-variable internal control that no absolute number can fake. Measured in
  `fv3_duo_gap_register_2026-08-06.md`: the duo grid buys **8.2× at hord8 and
  59× at hord6** on `max|v|` for W2 α=0 day 5, where the exact solution has
  `v ≡ 0` so `max|v|` *is* the cube imprint.
- **3-D, hydrostatic and non-hydrostatic:** DCMIP-2016 Jablonowski–Williamson
  baroclinic wave (`nh.case-13`, `test_cases.F90:77`), C48–C192, days 1–9,
  scored with the alignment-free area-weighted rms′ of `p_s` by
  `jw_duo_oracle_compare.py`. Measured in the same register: the oracle grows
  **80× over eight days** (0.0295 → 2.380 hPa) — that growth is the baroclinic
  instability. A lane that starts at 10 hPa, grows 1.6× and flattens from day 4
  is showing a static grid-locked pressure pattern, whatever its norms say.

Both sets of numbers are the register's measurements, not the published paper's;
they are quoted here so tier 5 has a target, and they are re-measured rather than
assumed when the JAX arm is first scored.
- Our own SW suite (W2/W5/W6, cosine bell) and NH cases (DCMIP TC2/TC3) run on
  the same lane, as the "tested against our test cases" half of the ask.

**Tier-5 protocol is fixed to the oracle's resolved namelist, not to ours.**
Every duo deck resolves `D_EXT = 0.0` (`logfile.000000.out:406`); a run with
`d_ext=0.02` is not comparable to any oracle number, and one already produced a
retracted "23–30×" claim in this campaign. Tier 5 also runs at the **production
time step and configuration** — FESOM2-C's most expensive bug was invisible at a
short validation step and only appeared after ~110 model days at the production
step (§8).

### Tier 2c — the conservation gate

Named as load-bearing in §2 and §4 but never specified, which made it
unenforceable (codex finding 10). It is executable and tolerance-independent:

| invariant | why it is the right quantity |
|---|---|
| global dry mass | the scheme's primary conserved quantity; the one thing a flux-form core must not leak |
| per-tracer mass | `tracer_2d_1L` + `map1_q2` are *designed* conservative; a subcycling or remap defect shows here and nowhere else |
| pressure-thickness closure (`Σ delp` vs `ps − ptop`) | catches a remap that loses column mass without moving the surface pressure |
| constant-field preservation | a uniform tracer must remap to itself exactly; fails on almost every interval-search defect |
| flux antisymmetry across all six shared edges | the duo barriers exist to make the seam flux single-valued; this is the direct test that they did |
| the `w_limiter`'s weighted momentum invariant | it **redistributes** between layers and must not create momentum |

Two protocol requirements, because both change the answer: state whether the
diagnostic includes halo cells, and use **cubed-sphere area weights** — an
unweighted mean on a non-uniform grid is not a global mean.

## 6. The live twin switch

Following FESOM2-C §4.4. `LEGOESM_FV3_JAX_VERIFY=1` makes each JAX kernel, on
each call, run its NumPy twin on the same live inputs, record `max|Δ|` per
output field to a manifest, and **return the JAX result unchanged** so the
trajectory is untouched. Default off; zero cost when off (a Python `if` on a
static flag, resolved at call/trace time, not a `jnp.where`).

**The constraint that makes this non-trivial in JAX, and how it is resolved.**
FESOM2-C's twin works because C is eager: the twin can read the live state at
any time. A JAX kernel under `jit` sees *tracers*, not values, so the NumPy twin
**cannot be called from inside a jitted region** — `np.asarray(tracer)` raises,
and the two escape hatches are both bad here (`io_callback` serialises the step
and interferes with the AD path; a `pure_callback` would hide the very
divergence it is meant to expose). Therefore:

> **Verify mode runs the lane EAGERLY.** `LEGOESM_FV3_JAX_VERIFY=1` bypasses the
> `make_*_jit()` factories and calls the unjitted kernels, so every operand is a
> concrete array and the NumPy twin can be applied directly. It is a diagnostic
> mode, not a production mode: expect it to be one to two orders slower, and
> never quote a timing from it.

That restriction is acceptable because of what the mode is *for* — turning
"which kernel first diverges over a 500-step run" from a bisection campaign into
a single instrumented run, and catching the kernel that is individually correct
but wired up wrong. Neither needs speed.

**What eager verification therefore does NOT cover, stated so it is not
over-claimed:** it never exercises XLA lowering, fusion, compiled control flow,
static-argument caching, or the actual jitted composition. It can localise a
wiring defect only if the eager and compiled dispatch paths are *structurally
identical* — so the eager and jitted wrappers must share **one pure kernel
body**, with the environment flag selecting the production callable *before*
tracing. A jaxpr test asserts no twin operation appears when the flag is off.
Compiler-induced discrepancies belong to **tier 2**, not to the live twin.

A second, cheaper instrument covers the jitted path: the **tier-3 replay**
(§5) runs N steps on both backends from the same initial state and compares the
end state. It sees the same class of defect, works under `jit`, and costs one
extra run — but it localises to "somewhere in the step", where the live twin
localises to a kernel.

## 7. Differentiability

The lane exists to be differentiated; a lane that runs but whose gradient is
wrong is worse than no lane.

**Order-2 `check_grads` at a smooth state is not, by itself, a meaningful gate
for this core** (codex finding 7). It verifies that AD agrees with finite
differences *on one fixed branch*. But PPM transport **is** extrema detection,
monotonicity constraints, upwind selection, clipping and CFL-dependent
subcycling — so a gate that systematically avoids every switch excludes exactly
the behaviour most likely to produce zero, unstable, or discontinuous
sensitivities. Demanding finite-difference agreement *at* a switch is equally
wrong: there is no single classical derivative there.

The gate is therefore three parts, all required:

1. **Smooth region** — branch-fixed JVP and VJP against finite differences,
   `check_grads(..., order=2)`. This is the old gate, kept, and demoted to one
   third of the answer.
2. **At every switching surface** — one-sided directional derivatives
   approaching from **both** sides, each asserted against the *documented branch
   derivative* for that side. The surfaces are enumerated per module, not left
   as "the limiters": PPM extrema/monotonicity clips, the upwind `sign`
   selections, `_rezone`'s interval boundaries, the `w_limiter` clip, and the
   integer crossings of the tracer subcycle schedule.
3. **Short rollout** — gradients of a physically meaningful scalar objective
   through a few acoustic substeps, checking finiteness, **adjoint consistency**
   (`⟨Jv, w⟩ = ⟨v, Jᵀw⟩`), convergence while the perturbation stays inside one
   branch, and bounded behaviour across a branch change.

If a downstream use ever needs gradients *through* a switch, that requires a
chosen and validated surrogate or `custom_jvp` policy, declared here. The exact
discrete FV3 algorithm does not supply one, and pretending otherwise is how a
silently-zero sensitivity ships.

Mechanics, unchanged: no `donate_argnums`; no Python control flow on traced
values; stable shapes; a retrace test on two different `dt` values.

## 7a. The state contract and the precision boundary

Two contracts the first revision left implicit. Both decide compilation-cache
stability and seam correctness, so both are fixed here rather than emerging from
whichever module is written first (codex findings 11 and 12).

**State layout.** The NumPy lane carries a **list of six per-face states** with
nested tracer lists (`fv3_native_dynamics.py:237-267`), while some helpers stack
explicitly (`fv3_native_dsw_phase_3d.py:230`). The JAX lane must choose, and the
choice is not free: staggered fields have *different shapes per stagger* (A, B,
C, D), so a leading face axis is not always even constructible. The contract
must state — and a test must pin — the canonical PyTree, face ordering, field
staggering, halo widths, array origin (the NumPy lane puts Fortran index `1-ng`
at numpy index 0 on both axes), and **which leaves are static** (integer index
tables, topology, `k2e` tables) versus traced (field data). "`ctx` is converted
to JAX tables once" is not a PyTree definition. Required tests: tree-structure
and shape stability, a compilation-cache-count check, and a round-trip
NumPy↔JAX state test.

**Precision.** Tier 0's per-kernel f64 gate is necessary and late: with
`jax_enable_x64` disabled, `jnp.asarray(x, dtype=float64)` silently truncates at
*array creation*, and no downstream check can recover the lost bits. So
`jax_enable_x64` is an **import/startup precondition** that fails before any
model array is constructed, and the audit covers constants, context tables,
initial conditions and boundary adapters — not just kernel operands. The stakes
are recorded in the NumPy authority itself: float32 costs `0.208` in `gz` and
`2.2e-3` in `pef` (`fv3_native_nh_core.py:68-71`). Mixed precision is outside
certification; the lane is f64 end to end.

## 8. Failure modes, and the tier that catches each

From FESOM2-C §7.1, mapped onto this port. Every row is a bug class that
"would break an unguarded port" and that is cheap to catch at the right tier.

| failure mode | symptom here | caught at |
|---|---|---|
| **latent bug** — a constant mis-ported, negligible at a short test step, significant at the production step | their instance: an Adams–Bashforth offset hardcoded 1e-9 where Fortran reads 0.1 → central-Arctic instability after ~110 model days, **only at the production time step** | tier 5 at the production `dt`, never a smoke test; prevented by R2 |
| **halo loop-bound bug** — a write loop run over owned points where the Fortran runs owned+halo | stale ring-1 values feeding the upwind selection | tier 1 with the halo included in the compared window; tier 3 |
| **allocated-but-never-computed field** | field stays 0; comparison cannot see it because the reference is 0 at the same phase | always-on physical-range check; first read in a later phase |
| **wrong vertical stride** (loop bound used as allocated depth) | gradient inflated ~1000×, solver fails in ~2 steps | tier 1, far above the noise floor |
| **per-vertex vs per-cell geometry confusion** | corrupted layer thicknesses / control volumes | tier 1 per-substep dump |
| **namelist value vs source default** — the assistant copies the source default | systematic regional bias | tier 5; prevented by R6 |
| **silent no-op call** (repo-earned, R5) | a boundary floor four orders above the interior | tier 4/5 boundary-cell localization |
| **joint bug** — every kernel individually correct, the join wrong | drift that appears only over many steps | tier 3/5 + the conservation-budget localization argument (§1) |

## 9. Work breakdown

Ordered so that every unit is certifiable the moment it lands, and so the
end-to-end assembly is reached as early as the dependencies allow. Derived from
a full call-graph trace of one `fv_dynamics` step in the NumPy lane.

**Already in the JAX lane** (`fv3_nh_core.py`): `sim1_solver`, `riem_solver_c`,
`riem_solver3`, `edge_profile`, `update_dz_c`.

**Not the target of this port, despite looking like it:** the JAX `fv_tp_2d`
(`fv_tp_2d.py:1122`), `ppm_transport_1d` (`fv3_sw_core.py:2927`),
`interp_center_to_corner_a2b_ord4` (`operators_cdgrid.py:1145`), and
`_d_sw_native` (`fv3_sw_core.py:3359`, monolithic). These are cdgrid-signature
research reimplementations, not loop-faithful mirrors of the NumPy duo lane.
They stay where they are; the mirrors are separate modules.

### Phase 1 — 2-D (shallow water), the runnable-and-scoreable milestone

| unit | mirrors | new module |
|---|---|---|
| duo halos + **both barriers** | `fv3_native_gridstruct` exchanges, `k2e_remap_halo_rings`, `_fill_corners_*`, `average_*_shared_edge*`; `fv3_native_ext_vector` `ext_scalar`/`ext_vector`, `c2l_ord2*`, corner-region Lagrange | `grids/fv3_duo_halos.py` |
| tp_core | `xppm`, `yppm`, `pert_ppm`, `fv_tp_2d`, `copy_corners`, `deln_flux`, `xtp_u`, `ytp_v` | `core/fv3_tp_core.py` |
| SW core | `c_sw`, `d2a2c_vect_duo`, `divergence_corner_duo`, `d_sw1_duo`…`d_sw6_duo`, `del6_vt_flux`, `a2b_ord4` | `core/fv3_duo_sw_core.py` |
| pressure gradient | `geopk`, `p_grad_c`, `one_grad_p`, `nh_p_grad`, `pk3_halo`, `pln_halo`, `pe_halo` | `core/fv3_pgrad.py` |
| km=1 acoustic step + outer step | `full_acoustic_step_sixface`, `advance_duo_outer_step`, `run_duo_sw` | `core/fv3_duo_stepper.py` |

Exit: `run_duo_stepper_w2.py` / `run_duo_stepper_case6.py` gain a
`--backend {numpy,jax}` flag, and the **existing** `w2_duo_oracle_gate.py` /
`case6_duo_oracle_gate.py` score the JAX arm unchanged. Swapping the engine
under calibrated gates is worth more than a new gate.

**The swap is one symbol.** `run_duo_stepper_w2.py:178-215` imports exactly
three things and calls one of them in the time loop:

```python
ctx    = build_six_face_duo_context(...)   # SETUP  — stays NumPy
states = w2_six_face_state(ctx)            # IC     — stays NumPy
for _ in range(steps_per_day):             # HOT    — the swap point
    states = full_acoustic_step_sixface(ctx, states, args.dt, d_ext=args.d_ext)
```

So Phase 1 needs a JAX `full_acoustic_step_sixface` with that exact signature
and nothing else changes: the context builder, the initial condition, the
`geographic_va` lens, the nearest-neighbour lat-lon map and the npz provenance
record all stay on the NumPy side, where they are already certified. `ctx` is
converted to JAX tables once, before the loop.

This also makes the tier-3 replay gate free: run N steps on both backends from
the same `states` and compare, which is precisely the multi-step state-threading
test FESOM2-JAX §2.4 requires and single-step kernel gates cannot provide.

### Phase 2 — 3-D hydrostatic

`fv3_native_state_3d`/`eta` conversion, the three phase drivers
(`cgrid_phase_3d`, `dsw_phase_3d`, `dsw_tail_3d`), `acoustic_substep_3d` +
`acoustic_loop_3d` (`lax.scan` over `n_split`), `fv_mapz` remap,
`tracer_2d_1L`, and the `fv_dynamics_step` shell (`lax.scan` over `k_split`).
Exit: `full_step_oracle_parity.py --backend jax` at the same `--max-rel`.

### Phase 3 — non-hydrostatic

`update_dz_d` (the last NH kernel with no JAX twin), `cgrid_nh_pressure_phase`,
`dgrid_nh_pressure_phase`, the NH arms of `c_sw`/`d_sw1`/`d_sw2`/`d_sw5`, the
`gz`↔`zh` acoustic carry, and the NH `map1_ppm` arms (`w` at `iv=-2`, `delz`
specific volume, the `w_limiter` passes). Exit:
`full_step_oracle_parity.py --nh --backend jax`, then the N-step gate.

### The two nodes that will not port mechanically

Both are flagged here so they are designed, not discovered:

1. **`_rezone`** (`fv3_native_mapz.py:594-669`) — the remap interval search:
   an interval search, *different formulas* for a contained versus a spanning
   target cell, a conditional accumulation of source layers, a `k0` carried
   from one target layer to the next (`:599-604`, `:614-650`), and an
   exceptional no-bracket path.

   **Prescribed design** (saying only "keep the traversal, avoid
   `searchsorted`" left the central question unanswered — codex finding 6):
   a **static-length nested `lax.scan`** carrying explicit
   `(found, k0, qsum, out)` state, fixed ascending traversal, sequential
   `qsum`, explicit first-match tie rule. Not a dynamic `while_loop` (no
   reverse-mode rule); not a vectorized interval mask (changes evaluation and
   reduction order). The no-bracket path must be an explicit returned error
   flag, a precondition gate, or `checkify` — never a plausible number.
   Fixtures: exact-interface, one-ULP-below, one-ULP-above. A one-ULP
   perturbation at an interface selects a different layer **and** a different
   formula, so its discrepancy is not rounding-scale and must not be certified
   at a rounding-scale tolerance.
2. **`tracer_2d_1l_sixface`** (`fv3_native_tracer2d.py:193`) — `nsplt` is a
   **data-derived loop trip count**: `nsplt_k[k] = int(1.0 + cmax[k])`
   (`:318`), where `cmax[k]` is the six-face max of a Courant-like number
   built from the flux capacitors (`:299-315`). jit cannot take that as a
   Python loop bound.

   The three candidates:

   | option | faithful? | reverse-differentiable? | cost |
   |---|---|---|---|
   | (a) static, resolved eagerly | yes, exactly | in isolation | **device→host sync every step; retrace when `nsplt_k` changes; and it CANNOT RUN inside `jit`/`grad`/`scan` at all** |
   | (b) `lax.fori_loop` with a traced bound | yes | **NO** — a traced bound lowers to `while_loop`, which has no reverse-mode rule | none |
   | (c) fixed `NSPLT_MAX`, mask the inactive iterations | yes, if the mask is a proven no-op | **yes** | `NSPLT_MAX ×` the transport work |

   > **RETRACTION (this document, commit `023b996b4`).** An earlier revision
   > chose **(a)** and argued it costs nothing in the gradient because
   > `int(1.0 + cmax)` is a floor and its derivative is zero almost everywhere.
   > The derivative argument is correct **and irrelevant**, because (a) does not
   > compose: the moment the step is placed inside `jit`, `jax.grad`, or a
   > `lax.scan` over time — which is the entire point of the lane — `cmax` is a
   > *tracer*, and `int(1.0 + cmax)` raises. (a) only works if the whole step
   > runs eagerly, which forfeits the lane. Caught by the codex adversarial
   > review of this document (finding 5, BLOCKER).

   **Resolved: (c).** A fixed `NSPLT_MAX` derived from the admitted CFL
   envelope; `lax.scan` (or a fixed-bound `fori_loop`) over `NSPLT_MAX`
   iterations; inactive iterations masked while the active ones keep the exact
   state and halo cadence; **fail loudly** if the resolved `nsplt` would exceed
   `NSPLT_MAX` rather than silently truncating the subcycling. The integer
   schedule carries `stop_gradient`, and the docstring states that the
   derivative is piecewise with no term through the trip-count decision — which
   is the true derivative a.e., the one correct part of the retracted argument.
   Tests must sit on **both sides of every schedule transition**, not only at
   `nsplt = 1`.

Good news from the same trace: `c_sw` and all six `d_sw*_duo` are **already
functional** in the NumPy lane (they copy every input at entry and return
dicts), so R4 costs nothing there.

## 10. Explicitly out of scope

- **Re-opening hop A.** The NumPy lane's Fortran certificates stand; this
  campaign does not re-derive them.
- **The moist lane** (`zvir ≠ 0`, `nwat ≥ 3`, `q_con`, `moist_cv`). Structurally
  dead under the pinned deck (R6); certifying it needs a new Fortran run and a
  new baseline.
- **The residual 1.19e-9 hydrostatic floor.** Both instrumented mechanisms are
  refuted; it is an open item on the NumPy lane, and the JAX lane inherits it
  rather than being blamed for it. Any JAX-lane number at or below that floor is
  reported as inheriting an unexplained residual, not as agreement.
- **Distributed / sharded execution.** Single-device first. The sharded path's
  acceptance (dense-path parity on one device, reassociation tolerance on many)
  is written down here so it is not re-invented later, but it is not built now.
- **Retiring the NumPy lane.** It is the hop-B authority; deleting it forfeits
  the ability to re-verify anything already certified. It stops growing, it does
  not go away.
