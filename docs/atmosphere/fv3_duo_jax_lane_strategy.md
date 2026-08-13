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

## 3. Rules

**R1 — Literal translation.** The NumPy lane is the specification for the JAX
lane. The only permitted transformations are mechanical:

- an explicit `for i` / `for j` loop over grid indices becomes a vectorized slice
  expression **over the identical index window**;
- an explicit `for k` *recurrence* becomes `lax.scan` over k — never
  `associative_scan` and never `cumsum`, because both reassociate the
  floating-point sum and break the parity contract (this is already the
  established doctrine in `fv3_nh_core.py`);
- in-place mutation of an argument becomes a functional return (§ R4);
- a static Python `if` on a config value stays a Python `if`; an `if` on array
  data becomes `jnp.where`.

Anything else — algebraic simplification, fusing two loops that the Fortran
keeps separate, "obviously equivalent" reorderings — is forbidden. If a
transformation looks like an improvement, it is out of scope by definition.

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
| pointwise / neighbour-reading | `edge_profile`, `xppm`/`yppm` limiters, EOS-like algebra | ~1e-15 relative (rounding alone) |
| accumulating / reassociated | flux divergence sums, `pe` rebuild, area-weighted global sums | ~1e-12 relative |
| sequential recurrence under `lax.scan` | `sim1_solver`, `riem_solver3` Thomas sweeps | ~1e-15 own-scale; cancellation-amplified outputs (`pe`, `w2`) sit at ~3e-15 of their own small scale — record which |

Three standing riders:

- **Bounds are `measured × (3…10)`**, never loosened without recording the new
  measurement.
- **Assert both hops**: JAX-vs-NumPy *and* jit-vs-eager. Evidence in a comment
  does not fail when the code regresses.
- **A residual at the tolerance floor is UNEXPLAINED, not agreement.** Say so.

Where a check's power does not depend on tolerance tightness — conservation and
budget closure, analytic invariants, equivariance, N-step growth judged against
the 1-step floor, pattern correlation — prefer it.

## 5. The validation ladder

| tier | what is compared | against | acceptance |
|---|---|---|---|
| **0** | shapes, dtypes | `jax.eval_shape` | exact; every operand f64 or `TypeError` at entry |
| **1** | one JAX kernel, one call | its NumPy twin, same live inputs | §4 tolerance, per-field, pinned to measurement |
| **1g** | one JAX kernel's gradient | finite differences (`check_grads`, order 2) | away from switching points; the point named where it is not |
| **2** | jit vs eager, same kernel | itself | §4 tolerance; divergence means a tracer bug |
| **3** | multi-step replay of the assembled chain | NumPy lane, same IC | bounded growth judged against the 1-step floor |
| **4** | one full `fv_dynamics` step, all fields | NumPy lane end-to-end | per-field, pinned |
| **5** | 5–9 day integrations | **oracle `atmos_daily.nc`** on matched case/grid/day | solution-tier; the campaign's exit criterion |

Tier 5 is the only tier that closes the chain back on the original. Its cases:

- **Shallow water (2-D):** Williamson-2 at α=0 and α=45, `hord` 5/6/8/10, C48
  through C192 as budget permits; case-6 Rossby–Haurwitz C48; case-8 colliding
  modons C48 at α=0 and α=45. The oracle ships duo **and** plain arms for each,
  so the duo/plain ratio is a one-variable internal control that no absolute
  number can fake: the published duo gain is **8.2× at hord8 and 59× at hord6**
  on `max|v|` for W2 α=0 day 5, where the exact solution has `v ≡ 0`.
- **3-D, hydrostatic and non-hydrostatic:** DCMIP-2016 Jablonowski–Williamson
  baroclinic wave (`nh.case-13`, `test_cases.F90:77`), C48–C192, days 1–9,
  scored with the alignment-free area-weighted rms′ of `p_s`. The oracle grows
  **80× over eight days** (0.0295 → 2.380 hPa); a lane that grows 1.6× and flats
  out is grid-locked, whatever its norms say.
- Our own SW suite (W2/W5/W6, cosine bell) and NH cases (DCMIP TC2/TC3) run on
  the same lane, as the "tested against our test cases" half of the ask.

**Tier-5 protocol is fixed to the oracle's resolved namelist, not to ours.**
Every duo deck resolves `D_EXT = 0.0` (`logfile.000000.out:406`); a run with
`d_ext=0.02` is not comparable to any oracle number, and one already produced a
retracted "23–30×" claim in this campaign.

## 6. The live twin switch

Following FESOM2-C §4.4. `LEGOESM_FV3_JAX_VERIFY=1` makes each JAX kernel, on
each call, run its NumPy twin on the same live inputs, record `max|Δ|` per
output field to a manifest, and **return the JAX result unchanged** so the
trajectory is untouched. Default off; zero cost when off (a Python `if` on a
static flag, resolved at trace time, not a `jnp.where`).

This is worth building early: it turns "which kernel first diverges in a 500-step
run" from a bisection campaign into one instrumented run, and it is the only
instrument that sees a kernel that is individually correct but wired up wrong.

## 7. Differentiability

The lane exists to be differentiated; a lane that runs but whose gradient is
wrong is worse than no lane.

- `check_grads(f, args, order=2)` on every kernel, at a state **away from**
  `jnp.maximum`/`jnp.minimum`/limiter switching points, with the non-smooth point
  named in the test rather than silently avoided.
- Gradients through a short full-model integration (a few acoustic substeps),
  not only per kernel.
- No `donate_argnums`. No Python control flow on traced values. Stable shapes,
  no retrace: a retrace test on two different `dt` values.
- PPM limiters and the upwind `sign` selections are the expected non-smooth
  sites. Each gets an explicit note stating whether the gradient is
  sub-differential there and which branch it takes.

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
end-to-end assembly is reached as early as the dependencies allow.

*(Filled in from the call-graph inventory; see the campaign STATE file for live
status. Kernels already in the JAX lane: `sim1_solver`, `riem_solver_c`,
`riem_solver3`, `edge_profile`, `update_dz_c` — `fv3_nh_core.py`.)*

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
