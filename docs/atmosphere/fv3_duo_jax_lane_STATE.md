# FV3 duo-grid JAX lane — campaign STATE / handoff

Companion to `fv3_duo_jax_lane_strategy.md`. That document is the contract and
does not change often; **this file is the live state and is rewritten at the end
of every session.** The pattern is taken from the FESOM2 port (arXiv:2606.11356
§4.3), which found that a port of this length cannot be carried in a model's
own memory: it needs a plan file, a lessons file, and a handoff file kept beside
the code, so a fresh short session can rebuild context from a few reads.

---

## Where the work lives

| | |
|---|---|
| worktree | `/burg-archive/glab/users/pg2328/legoESM_fv3jax` |
| branch | `feat/fv3-duo-jax-lane` (pushed to `cf`) |
| **PR** | **#1610, DRAFT** — not mergeable while `TOL-PENDING` markers remain |
| run/log dir | `/burg-archive/glab/users/pg2328/fv3_duo_gaps/` |
| pinned run worktrees | `_fv3jax_run` (SHA `dbfdccb67`, run 1), `_fv3jax_run2` (SHA `9b39dc254`, run 2) |
| **pinned oracle tree** | `/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean` |
| oracle reference runs (42) | `/burg-archive/glab/users/pg2328/Code/FV3/duogrid_zenodo/extracted/Code and simulations files` |
| oracle 1-step/N-step runs | `/burg-archive/glab/users/pg2328/fv3_oracle_pinned/run_{hydro,nh}_{zerostep,1step,96min,320min}*` |

**Do not cite `fv3_recon/duo_model/`.** Its `dyn_core.F90` is an instrumented
copy (3230 lines, md5 `0a5df09a…`) against the pinned tree's 3128 / `e5a5fab9…`.
`sw_core.F90` IS identical across copies. See the correction banner in
`fv3_duo_gap_register_2026-08-06.md`.

---

## The one-line summary of the gap

The faithful duo port is **certified but not runnable as a model**: ~15k lines
of loop-faithful NumPy fp64, verified against Fortran dumps down to 1.19e-9
(hydrostatic, one step), imported by nothing outside `tests/` and
`scripts/validate/`. This campaign gives it a JAX twin that is jittable,
differentiable, and wired in.

Exception worth knowing: the NumPy lane is **not** entirely leaf-only. Five
init-time imports are load-bearing — `cubed_sphere.py:476`,
`cubed_sphere_cdgrid.py:768/1081`, and `duogrid_bgrid_ring.py:95/149/250` pull
the halo tables, metrics, ext machinery and `build_six_face_duo_context` from
it. Those are setup, not hot path, and they stay NumPy.

---

## Status

### Done this session

- `fv3_duo_jax_lane_strategy.md` — the contract: three-hop authority chain,
  tolerance policy keyed to what a kernel does, six-tier validation ladder,
  differentiability gates, failure-mode catalogue, work breakdown, scope
  boundary.
- **Register correction**: two duo-barrier line numbers were from the
  instrumented tree. Pinned-tree values are `dyn_core.F90:872` (CGRID_NE,
  barrier 1) and `:984` (BGRID_NE, barrier 2). The CODE was already right.
- **Two self-caught strategy defects fixed**: the live-twin switch cannot call
  NumPy from inside `jit` (verify mode is therefore eager); and §2 did not state
  what hop B is blind to.
- **`nsplt` design resolved** — static, with the argument for why that costs
  nothing in the gradient (floor ⇒ derivative zero a.e.).
- Measurement job `scripts/cluster/fv3_native/jax_lane_measure.sbatch`.
- **Codex adversarial review of the strategy: 13 findings, all closed**
  (job 9398328, log in this directory). 4 BLOCKER, 7 MAJOR, 1 MINOR, and two
  concerns explicitly returned as unfounded. The four blockers:
  1. tier 5 was the only independent authority check and had no acceptance
     criterion → tiers are now executable specs, tier 5 scores every carried
     prognostic, and tier 5a adds a standing direct Fortran-vs-JAX audit of
     hop A;
  2. the tolerance taxonomy had no class for branch-switching kernels, where
     one ULP flips a branch and the discrepancy is not rounding-scale;
  3. R1's "grid loop → vectorized slice" was not semantics-preserving without
     a dependence/alias audit (three sites in this core break it);
  4. **my `nsplt` resolution was refuted** — see the retraction below.
- **RETRACTION, same session.** Commit `023b996b4` resolved the tracer
  subcycle count as "static, resolved outside the trace", arguing it costs
  nothing in the gradient because `int(1.0 + cmax)` is a floor. The derivative
  argument is correct and irrelevant: static resolution *does not compose* —
  under `jit`/`grad`/`scan`, `cmax` is a tracer and `int()` on it raises, so
  the option only works if the whole step runs eagerly, which forfeits the
  lane. Replaced by fixed `NSPLT_MAX` + masking + `stop_gradient` on the
  schedule (commit `d8df2bde0`).
- **GLM-5.2 is unreachable**: the `Z_AI_API_KEY` in the environment returns
  401 on `api.z.ai`, on `open.bigmodel.cn` with a bearer token, and with a
  signed JWT. The key is expired, not the endpoint. Dual adversarial review is
  standing policy (main `a6060d41d`), so **this campaign is one reviewer short
  until the key is refreshed** — everything below has had codex only.

### The JAX lane exists — first execution scorecard

~15k lines landed at `dbfdccb67` and RAN for the first time (job 9400424, x64
CPU, single node). **All five modules import cleanly.**

| module | LOC | gates | first run |
|---|---:|---:|---|
| `grids/fv3_duo_halos.py` | 2033 | 52 | 40 pass, 12 fail |
| `core/fv3_tp_core.py` | 2620 | 16* | 15 pass, 1 fail |
| `core/fv3_pgrad.py` | 1701 | 90 | **88 pass**, 2 fail |
| `core/fv3_mapz.py` | 1666 | 212 | **203 pass**, 9 fail |
| `core/fv3_nh_core.py` | 2493 | 127 | **126 pass**, 1 fail |
| **total** | **10513** | **497** | **472 pass (95.0 %), 25 fail** |

**Where the 25 failures actually are** — established by reading where each
traceback tops out, not by the test's name:

| class | count | what it means |
|---|---:|---|
| bitwise `jit`-vs-eager on a floating-point sum | 12 | expectation: XLA contracts mul+add into FMA when jitted, so a few-ULP gap is correct |
| **off-switch FIXTURE PRECONDITION refused** | **9** | `check_grads` **never ran** — the helper that must certify the state as away from every limiter switch could not, and raised first |
| genuine `check_grads(order=2)` FD failure | 3 | finite differencing cannot resolve the array's dynamic range |
| guard-test expectation | 1 | Python's own `TypeError` for a missing keyword-only arg fires before any body guard |

⛔ **CORRECTION to an earlier revision of this file (commit `0b9780620`).** It
said `check_grads(order=2)` failed in four of five modules and accounted for 12
of 25 failures. That is wrong on both counts, and it was wrong because I
attributed failures by the *test's name* instead of by where the traceback
stopped. `check_grads` itself failed **3** times, in **2** modules. The larger
group is the 9 above, where the gradient was never taken at all.

**And that is the sharper finding, not a weaker one.** Nine gates failed because
*constructing a provably off-switch state for a PPM limiter chain is hard* —
the margins were hand-derived and over-tight (one measured 0.70 against a switch
at 1.0, guarded at an arbitrary 0.5; another used an absolute 0.25 K margin on a
quantity that varies 30× across columns). So the strategy's original gradient
gate was not merely weak at detecting defects, it was **difficult to even set
up**, which is independent evidence for codex's finding 7 and for the three-part
gate that replaced it. The tolerance-free adjoint identity needs no such
fixture: it runs **on** the switching surface, where `check_grads` is
meaningless.

**Zero confirmed real code defects came out of the run.** The one real code
defect in the campaign so far came out of the *review*, not the run: codex's
`_rezone` BLOCKER. That asymmetry is worth remembering when budgeting: the run
found expectation defects, the review found the code defect.

Replacement gate, now required everywhere (strategy §7 part 3): the
tolerance-free adjoint identity `⟨Jv, w⟩ = ⟨v, Jᵀw⟩`, with `Jv` from `jax.jvp`
and `Jᵀw` from `jax.vjp`. It uses no finite differences, so neither the FD step
nor the array's dynamic range can make it fail spuriously — and unlike
`check_grads` it stays meaningful on a linear operator.

*the tp_core gate count is a stale snapshot: its test file grew 511 → 1772
lines (63 tests) after the pin, closing a coverage hole where `xtp_u` and
`ytp_v` had **zero** tests despite running every acoustic substep.

**Failure triage — the discipline is to say WHICH is wrong, expectation or
code:**

1. **⛔ RETRACTED — there was no k2e "column swap". The TEST was wrong.**
   I reported a transposed halo-source table as a confirmed real defect. It is
   not one. Measured (`fv3_duogrid_oracle_n2.npz`): every `c12_*_ij` family is
   sorted by **column 1** (`[[-1,-2],[0,-2],[1,-2],…]`), while
   `compute_fv3_native_k2e` emits records sorted by column 0
   (`[[-2,-1],[-2,0],…]`). They are the **same record set in a different row
   order**, and an element-wise `np.array_equal` on two differently-sorted
   tables of records looks exactly like a pairwise column swap. The assertion
   pinned a convention nobody had spot-checked.

   It cannot reach the lane, for two independent reasons, both measured rather
   than argued: the consumer unpacks `for (fi, fj), lv, cw in zip(ij, loc,
   coef)` **character-identically in both lanes**, so any convention cancels in
   a JAX-vs-NumPy comparison; and the A and B record→value maps are *invariant*
   under the swap, while CX/DY are not — and A and B are the only families this
   lane consumes.

   **LESSON:** a printed element-wise diff of two TABLES OF RECORDS cannot
   distinguish a column swap from a row reordering. Compare record **sets**.
   The discriminating check was one line and took thirty seconds; I pattern-
   matched to a known trap in this repo (a probe that indexed `(nCells,6)` on a
   `(6,nCells)` array) and reported CONFIRMED without running it.

   **So the halo module's first run contained ZERO real code defects** — all 12
   failures were wrong test expectations.
2. **EXPECTATION WRONG — bitwise `jit`-vs-eager on floating-point sums.** XLA
   contracts mul+add into FMA in the jitted lowering and not the eager one, so a
   few-ULP gap is correct behaviour, not a defect. This is the cause of **most**
   of the halo failures, including the two `k2e_remap` ones and the c2l one.
   The split is now structural, not empirical: bitwise is kept only where there
   is no `x*y + z` for XLA to contract — the four exchanges, the six
   `fill_corners`, `pack_p1`, `write_*_strips`, **and both barriers**, whose
   blend is `0.5 * (a + s·b)` with `s` a ±1 constant folded at trace time, i.e.
   a multiply *of* a sum with no contraction site. Everything containing
   `Σ w·v` now asserts a measured bound.
3. **⛔ MY DIAGNOSIS WAS WRONG — the c2l failure was not a NaN-equality
   problem.** I claimed `NaN == NaN` was making the gate unpassable. In fact
   `_bitwise_equal` already passes `equal_nan=True`, and the mask-equality gate
   passed, proving both lanes' NaN patterns are identical. The finite part
   diverged, from `a11·u1 + a12·v1` — FMA again, i.e. cause 2. "Both sides
   all-NaN" was pytest's truncated array repr sampling the scratch corners.
   The NaN region is legitimate untouched scratch: `do_halo=.true.` defines a
   14×14 window of an 18×18 face and the NumPy lane NaN-fills the rest by
   construction.

   The instruction still produced a better test, and it exposed a genuine
   defect in the comparison helper: `_cmp` returned `0.0` when *both* sides were
   entirely non-finite — **a comparison that could not fail**. It now raises on
   that case. The gate is restructured to assert the window entirely finite on
   both lanes (fires if the NaN is real), the scratch entirely non-finite on
   both (fires if the window moves), and values compared on the window only.
4. **EXPECTATION WRONG — pgrad's two `check_grads(order=2)`.** One element out
   of 312, 0.39 % against a 0.31 % tolerance, on a value of 3.5e5 in an array
   whose entries are ~1e2. Order-2 finite differencing cannot resolve that
   spread. Being replaced by the tolerance-free adjoint identity
   `⟨Jv,w⟩ = ⟨v,Jᵀw⟩` — which is what codex predicted when it called order-2 FD
   a weak gate for this core.

### Run 2 (job 9401521, SHA `9b39dc254`) — after the first triage round

| module | run 1 | run 2 |
|---|---|---|
| `fv3_duo_halos.py` | 40/52 | **52/53** |
| `fv3_pgrad.py` | 88/90 | **96/100** |
| `fv3_mapz.py` | 203/212 | **217/224** |
| `fv3_tp_core.py` | 15/16 | 227/**249** (suite grew 16 → 249) |

Codex's `_rezone` BLOCKER fix is confirmed working: the zero-thickness
regression and both map gradient gates now pass.

**Every remaining failure was again an expectation or fixture defect**, and the
two most instructive are worth keeping:

- **`deln_flux` at `nord=2` is not inert.** It multiplies by `rarea ≈ 1e-12`
  once per pass, so a hand-picked raw `damp = 1e-3` put the del-6 increment near
  1e-23 against an `fx0` of order 1…156 — *seven orders below one ULP* of the
  value it is added to, hence `fx + fx2 == fx` bitwise. `nord = 0, 1` passed
  because they carry one and two fewer factors of `rarea`. The oracle supplies
  exactly the compensating scaling at **`tp_core.F90:199`**,
  `damp = (damp_c · gridstruct%da_min)**(nord+1)`, under the `damp_c > 1.e-4`
  guard at `:198`; the test bypassed it. Now calibrated from a probe — exact,
  because `deln_flux` is *linear* in `damp` in all four branches.
- **A "rough" fixture that never left the limiter.** `smt5 = bl·br < 0` is false
  only at a local extremum, and for a linear field `bl·br = -(s/2)² < 0` at every
  cell — so the fixture was one-sided by construction and its own self-check
  caught it (on = 234, off = 0).

### Four times a measurement refuted MY reasoning

Recorded as a pattern, because it is one. In each case I inferred a mechanism
from a symptom instead of measuring, and an agent corrected it:

1. **k2e "column swap"** → differently-sorted record tables. Retracted.
2. **"`check_grads(order=2)` failed in 4 of 5 modules"** → attributed by the
   test's *name*; 9 of those never ran `check_grads` at all.
3. **"forward passed, reverse failed, so the VJP is suspect"** → `check_jvp` and
   `check_vjp` consume the **same** FD data; `_assert_numpy_close` scales the
   bound by leaf size (`atol*a.size`, `public_test_util.py:164`), so comparing a
   312-element array gets `rtol = 3.12e-3` while comparing a scalar inner
   product gets `1e-5`. **312× tighter on identical numbers.** The split carries
   no information about reverse mode.
4. **"normalise the loss to O(1)"** (carried over from what fixed `pgrad`) →
   that fixes an **atol** failure; `mapz`'s was **rtol**, and scaling the loss
   scales the AD tangent and the finite difference identically, so the ratio is
   invariant. The levers were the step size and the function's nonlinearity.

The common failure is reasoning from a plausible mechanism to a verdict without
running the one cheap check that separates it from the alternatives. Every one
of these was a sub-minute measurement.


### ★ THE JIT-VS-EAGER GAP IS EXPLAINED (run 9408346) — cancellation, not a defect

The campaign's one open defect. Chain, all measured:

- `del6_vt_flux` (`sw_core.F90:1948-1951`) computes `ut`/`vt` as a del-6 stencil
  — a chain of differences of large nearly-equal numbers. Measured condition
  number at the worst cell **(9, 2): 7.781e+12**.
- Under `jit`, XLA contracts mul+add into FMA; eager does not. That shifts an
  input by ~1 ULP.
- `cond x eps = 1.728e-03` is the ceiling such a perturbation can reach. The
  **measured gap is 1.257e-06** — three decades *under* the ceiling. The
  amplification capacity is present and sufficient.
- `ut` feeds **`v`** (`v -= ut`), and the stepper's full-step gate measures
  `v` at **7.456e-07**, with `acoustic_step` carrying **5.646e-07**. The chain
  closes.

**This is not a port defect.** It is an ill-conditioned difference operator
amplifying a legitimate compiler freedom. Treatment, already in place: gate on
the **cell count** (2 of 342; a branch to systematic goes red), report the
magnitude, label it UNEXPLAINED-BY-TOLERANCE rather than bounding it.

Two of my hypotheses were refuted on the way: a PPM limiter **branch flip**
(`del6_vt_flux` contains no data-dependent branch at all — verified by scanning
the symbol that runs, with `d_sw1_duo`'s `jnp.where` as the non-vacuity control)
and, earlier, FMA-scale rounding (7.5e-07 is 3.4e9 ULP).

### Codex adversarial review of the CODE — 1 BLOCKER, 2 MAJOR

All three in `_rezone` and the gradient gates, i.e. exactly the node the
strategy flagged as hardest.

- **BLOCKER** `fv3_mapz.py:998` — `q_span = qsum / denom` is formed
  unconditionally and then discarded by the `inside` select. The NumPy authority
  **returns** on the contained-cell path (`fv3_native_mapz.py:624`) without ever
  dividing, so a zero-thickness target inside one source layer gives NumPy a
  finite answer and JAX a 0/0 that reverse-mode AD propagates into the selected
  branch. The fix is not to invent a divisor — it is to not *form* a division
  whose result is discarded.
- **MAJOR** `fv3_mapz.py:983` — same defect in the source-layer loop: `esl = dp
  / dp1_m` runs on every static scan iteration including ones NumPy never
  visits.
- **MAJOR** — the gradient gates avoid every switching surface (`_rezone`
  interface ties; `update_dz_d` pinned to the linear `hord=2` arm), so they
  prove smooth arithmetic rather than the branch-heavy operators.

**Cleared by name** (so the review is known to have been real): both duo
barriers — slot selection `[0,3,4…]` preserving `w` and `q_con`, blend, and
compute-ring layout; duplicate-index scatter handling; the Lagrange diagonal
reading the incoming field; the `n ≥ 4` non-collision argument; no
`donate_argnums` anywhere.

### Also fixed

Two oracle citations in the NumPy lane were wrong by two lines — `pe_halo` was
cited as starting at `:1933`, which is `end subroutine pln_halo`. Found by the
JAX author re-deriving the spans instead of copying them.

### Not started

SW core (`c_sw`, `d_sw1..6_duo`) — blocked on `fv3_tp_core.py` landing so it can
be written against the real API. Then the km=1 stepper, then Phases 2 and 3.

---

## Next task, precisely

1. **Re-run the measurement job** against the post-triage code (re-pin a
   worktree at the new SHA, edit `REPO=` in
   `scripts/cluster/fv3_native/jax_lane_measure.sbatch`, `sbatch`). Expect the
   remaining `TOL-PENDING` bounds to trip on purpose — several were left at a
   provisional 1e-12 with the assertion message printing the measured value, so
   one run yields the number.
2. **Replace every `TOL-PENDING` marker** with `measured X, bound = measured ×
   N`. Nothing ships with a marker left; the job censuses them in step 1 for
   exactly this reason. Current count is ~120 across five test files.
3. **Close codex's `_rezone` BLOCKER and the two MAJORs**, then re-review.
4. **SW core** (`fv3_duo_sw_core.py`: `c_sw`, `d_sw1..6_duo`,
   `d2a2c_vect_duo`, `divergence_corner_duo`, `del6_vt_flux`) — in flight.
5. **The km=1 stepper**: a JAX `full_acoustic_step_sixface(ctx, states, dt,
   d_ext=…)`. This is the whole of Phase 1's remaining risk, and it is one
   function.
6. `--backend {numpy,jax}` on `run_duo_stepper_w2.py` /
   `run_duo_stepper_case6.py`, then score with the **existing** calibrated
   gates — do not write new ones.

### Step 5 in detail — the stepper is a thin composition, not a new algorithm

Read from `fv3_native_duo_stepper.py::full_acoustic_step_sixface`. Its whole
body is four things, and every JAX piece either exists or is in flight:

1. `acoustic_step_sixface(ctx, states, dt, sw_cfg, entry_ascalar)` — the
   `c_sw → p_grad_c → d_sw1…d_sw6` chain with both barriers.
   *Needs:* `fv3_duo_sw_core.py` (in flight) + `fv3_duo_halos.py` ✅ +
   `fv3_pgrad.py` ✅.
2. A post-step `delp`/`pt` halo refresh via `ext_scalar_sixface(…, "A", …)`
   (`dyn_core.F90:1336-1337`). *Needs:* `fv3_duo_halos.ext_scalar_sixface` ✅.
3. Per face: `geopk_sw_1lev_d` → the external-mode filter
   `divg2 = d_ext · da_min_c · saved divergence` (km=1, where the mass weight
   cancels at one level) → `one_grad_p_1lev`. *Needs:* km=1 wrappers over the
   already-ported `fv3_pgrad.geopk` / `one_grad_p` ✅.
4. **No post-step vector refresh on the ext path.** The returned D-wind halos
   are deliberately STALE-BY-ONE, faithful to `dyn_core.F90:1332-1338` — that
   routine refreshes only `delp`/`pt`, and `ext_vector` runs at the *next*
   step's entry (`:468-472`). A JAX twin that "helpfully" refreshes the winds
   here is a divergence, not a fix. This is the single easiest place in the
   whole port to be wrong while looking more correct.

Note `d_ext` **must default to 0** for any oracle comparison: every duo deck
resolves `D_EXT = 0.0`. The NumPy signature's `d_ext=0.02` default is a
research value and has already produced one retracted claim.

### Dependency note for step 4

`a2b_ord4` was promoted from `_a2b_ord4` in `fv3_pgrad.py` so the SW core's
`d_sw5_duo` can import it (a private cross-module import is banned by a CI
ratchet, and the alternative was a third copy of a 300-line routine).
`_a2b_ord4_k`, its vmap-over-levels wrapper, stays private. `del6_vt_flux` is a
**sw_core** routine and its home is the SW module; `fv3_nh_core._del6_vt_flux`
is a temporary private copy made before the SW module existed and should be
deleted once the SW one lands.

---

## The canonical reference runs, and their calibrated bounds

Recorded here so a later session does not re-derive them.

| gate | reference | calibrated bound |
|---|---|---|
| `w2_duo_oracle_gate.py` | `C48.sw.case2.alpha0.duo.hord6`, day 5.0 | envelope max\|v\| 0.0236, rms(v) 0.0096 m/s, scaled `(48/N)²`, `--tol-factor` 1.5 |
| `case6_duo_oracle_gate.py --case 6` | `C48.sw.case6.alpha0.duo.hord8` | `--max-gh 5e-3 --max-wind 6e-2` (worst measured 2.304e-3 / 2.850e-2) |
| `case6_duo_oracle_gate.py --case 2` | `C48.sw.case2.alpha0.duo.hord8` | `--max-gh 1.1e-2 --max-wind 2e-2` |
| `case6_duo_oracle_gate.py --case 2 --ref-alpha 45` | `C48.sw.case2.alpha45.duo.hord8` | `--max-gh 1.1e-2 --max-wind 2.5e-1` (wind bound is pole-artifact-limited) |
| `full_step_oracle_parity.py` | `run_hydro_1step_gfs` | IC control 1e-12 hard; step floor measured 1.1866e-9 |
| `full_step_oracle_parity.py --nh` | `run_nh_1step_gfs` | delz 6e-8, pt 1e-8, delp 5e-8, u/v 3-10e-6, w 3.4e-7 m/s |
| `jw_duo_oracle_compare.py` | `C48.nh.case-13.alpha0.duo.hord6` | `--max-rmse-hpa`; oracle grows 0.0295 → 2.380 hPa over days 1-9 |

`d_ext` **must** be 0: every duo deck resolves `D_EXT = 0.0`. A run with
`d_ext=0.02` is not comparable to any oracle number and has already produced one
retracted claim in this campaign.

---

## Lessons file (append, never prune)

Each entry cost something. New sessions read this before touching the port.

1. **Port the EXECUTION, not the call text.** A Fortran routine whose body is
   guarded on `present(...)` arguments the caller never passes is a silent
   no-op. `fill_corners_agrid_y(f0)` in the port vs the oracle's inert
   `fill_corners(f0, npx, npy, YDir)` was the entire 9.7882e-6 → 1.1866e-9
   hydrostatic boundary floor. One deletion, four orders of magnitude.
2. **Read what the deck RESOLVES.** The pinned deck runs `adiabatic = .true.`,
   `nwat = 0`, so `zvir` is identically zero and the whole moist / `q_con` /
   `moist_cv` path is structurally dead — not untested, never executed.
   Certifying it needs a new Fortran deck.
3. **Three copies of the oracle source exist and are not the same file.** Cite
   the pinned tree. This one bit the gap register itself.
4. **A tolerance inherited from another gate is a guess.** `1e-15` carried over
   to the kernel-4/5 gates when the measurements were 1.03e-15 and 1.02e-15 —
   a bound with no margin, which codex caught.
5. **Evidence in a comment does not fail when the code regresses.** Both hops
   (JAX-vs-authority and jit-vs-eager) must be ASSERTIONS. Also caught by codex,
   on this same file.
6. **`fv3_recon/duo_model` line numbers are off by roughly +60 in dyn_core.**
   If a citation looks close but not right, this is why.
7. **SLURM sets `TMPDIR=/local` and it gets reaped mid-job.** Force
   `pytest --basetemp` onto a burg path.
8. **Codex jobs serialise.** Two concurrent runs refreshing `CODEX_HOME` killed
   the token once. Use a node-local `CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID`.
9. **Login-node policy is hard.** Nothing over a few seconds or one core,
   including codex. Everything goes through `sbatch`.
10. **A printed element-wise diff of two TABLES OF RECORDS cannot distinguish a
    column swap from a row reordering.** Compare record *sets*. This produced a
    confidently-reported, wholly imaginary "transposed halo table" defect that
    took a thirty-second check to refute. Never report a verdict off a printed
    diff — the rule already existed as "diff ARRAYS, never printed summaries",
    and this is its table-shaped cousin.
11. **Attribute a failure by where the traceback STOPS, not by the test's
    name.** Nine gates named `..._check_grads_order2_off_switch` failed without
    `check_grads` ever running — the fixture precondition raised first. Reading
    the name instead of the traceback produced a wrong scorecard that went into
    a commit message. Two of the three retractions in this campaign are the same
    error: a verdict from a printed artifact rather than from the thing itself.
12. **`check_grads(order=2)` is a weak gate for this core, and hard to even set
    up.** Three genuine FD failures (dynamic range), plus nine failures just
    trying to construct a certifiably off-switch state for a PPM limiter chain —
    with hand-derived margins that were over-tight (0.5 guarding a switch at
    1.0, where the state measured 0.70; an absolute 0.25 K margin on a quantity
    varying 30× across columns). Use the adjoint identity `⟨Jv,w⟩ = ⟨v,Jᵀw⟩` as
    the primary gate: it needs no off-switch fixture and runs **on** the
    switching surface. Keep `check_grads` as a scoped smooth-region supplement,
    and run `order=1` before `order=2` so an FD-resolution failure can never be
    confused with a wrong Jacobian. Their blind spots are complementary — jvp
    and vjp of the same *wrong* Jacobian agree, so neither gate alone is enough.
12. **The run finds expectation defects; the review finds code defects.** On
    this campaign the 497-gate first execution surfaced zero confirmed code
    defects, while the adversarial review surfaced a BLOCKER. Budget for both;
    neither substitutes for the other.
13. **`max`-based statistics on the committed C12 fixtures are SENTINEL-
    CONTAMINATED.** `area`, `del6_u`, `del6_v` and `delp` in `dswcore_input.npz`
    all report a max of exactly **1.0000e+08** — the `BIG_NUMBER` fill in unset
    ghost/corner cells — and `max|rarea| = 1e-8` is `1/` that, about four
    decades from the physical ~2e-12. A gate anchored on any of those maxima is
    reading the sentinel, not physics. Worse, at a sentinel cell
    `del6 x rarea = 1e8 x 1e-8 = 1` exactly, so even a per-pass RATIO taken with
    `max` need not show the shrinkage a scaling law predicts. Swept the other
    six test modules: one hit, and it is a `max` over a windowed OUTPUT
    difference, which is safe. Anchor on a median over the compute window, or on
    an observed run fact, never on a `max` over a fixture field.
14. **A test seed derived from the parameter under test is a defect.** Eight
    adjoint gates died with `ValueError: expected non-negative integer` because
    the seed was `iord + 1`, and `np.random.default_rng` rejects negatives —
    so every negative-order scheme failed before reaching JAX. The failing set
    was reproducible standalone from the arithmetic alone. Seeds now come from
    `crc32(test name)`. Note what it cost even though it was "only" the harness:
    the negative-`iord` adjoints are UNTESTED, not broken, and no parity gate
    would have noticed because none of them differentiates.
15. **`cp` is ALIASED to prompt on this machine, so a plain `cp` AND `cp -f`
    both silently do nothing.** Hit twice in five minutes on 2026-08-14: a job
    was submitted against a stale file and would have been reported as a
    result. `/bin/cp` bypasses the alias — but the real fix is structural and is
    now enforced: **never hand-copy into a pinned worktree.** Commit first, then
    `scripts/cluster/fv3_native/pin_worktree.sh <name>`, which refuses on a
    dirty source, verifies the pinned SHA and asserts the new tree is clean.
    `jax_lane_measure.sbatch` additionally aborts with exit 3 if its worktree is
    dirty, so a run that cannot be attributed to a commit cannot silently
    produce numbers. This is the repo's existing "preflight before every sbatch"
    rule, made mechanical.
16. **A non-streaming request to GLM dies with `IncompleteRead`.** It is a
    reasoning model that can think for minutes with nothing crossing the wire,
    and the connection gets dropped. `glm_review.py` streams by default and
    keeps a partial answer on a mid-flight drop. Separately: a 1500-token cap
    produced 1497 reasoning tokens and EMPTY content, which reads as a refusal
    and is not one — the client now reports the reasoning/content split.

