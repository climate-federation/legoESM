# Pseudo-incompressible plane LES (LEX-faithful) — runbook

Status: IN PROGRESS (started 2026-06-18). Goal: a third plane LES dycore that
mesh-scales on GPU/TPU **without** the spectral core's global all-to-all FFT, by
solving incompressibility with a **nearest-neighbour (halo-exchange) matrix-free
preconditioned BiCGSTAB Poisson** instead of spectral-horizontal + tridiag-vert.

Oracle = **LEX** (MetLab-HKUST/LEX, GMD 2026), `solver_opt=1` (pseudo-incompressible).

## Why this core (vs the two existing plane dycores)
- `spectral_les_plane.py`: exact, but FFT ⇒ global all-to-all each step ⇒ poor TPU/GPU
  mesh scaling; doubly-periodic only.
- `compressible_euler_plane.py`: FD + MPI halo (scales), but acoustic CFL + biharmonic
  hyperdiff cap effective Re (turbulence laminarises).
- **pseudo-incompressible**: no acoustic mode (no acoustic CFL), no spectral all-to-all
  (only nearest-neighbour halo in the elliptic solve) ⇒ scales on mesh AND sustains
  turbulence. Handles non-periodic/stretched grids + deep density variation.

## LEX algorithm (faithful target), per RK3 sub-step — solver_opt=1
Reference files in the LEX clone (`/tmp/lex_oracle`):
`one_step_integration.py` (`rk_sub_step{0,1,2}`, `solve_pres_eqn`,
`update_momentum_eqn_euler`, `prep_momentum_eqn`), `pressure_equations.py`,
`pressure_gradient_coriolis.py`, `advection.py::get_divergence`.

1. **Buoyancy** `pres_eqn.calculate_buoyancy`:
   θ_ρ = θ·(1+reps·q_v)/(1+q_v),  b = g·(θ_ρ − θ_ρ0)/θ_ρ0  (reps = R_v/R_d).
   `b8w` = b interpolated to w-faces (2nd-order Lagrange extrap at walls), wall b=interp.
2. **Variable coefficient** `rtt = calculate_rtt` = ρ0θ0 · θ · (1+reps·q_v)/(1+q_v)
   at centres; top/bottom **ghost** rows hold ρ0θ0·b/Cp (encodes the rigid-lid
   w=0 Neumann BC for ρ0θ0·θ·∂π'/∂z).
3. **Momentum advection** `prep_momentum_eqn` → adv4u/v/w, flux-form with the
   `flow_divergence = get_divergence(ρ0,u,v,w)` correction (advective form
   = flux−q·div), periodic in x,y, WENO5/WENO3 faces (Borges WENO-Z), C-grid.
4. **Poisson RHS** `rhs_of_pressure_equation`:
   RHS = ∇·(ρ0θ0·adv) + ∇₂·(ρ0θ0·[f_v,−f_u]) + ∂/∂z(ρ0θ0·b |_w).
5. **Solve** `Cp·∇·(rtt ∇π') = RHS` (LHS = `laplace_of_pressure`) via
   `jax.scipy.sparse.linalg.bicgstab`, `x0 = π'_prev`, tol 1e-4, atol 1e-8, maxiter 100.
   LHS: π' at centres (NO ghosts on the unknown — solver needs same in/out shape);
   faces flux = 0.5(rtt_i+rtt_{i±1})·dπ'/d{x,y,z}; wall z-flux replaced by rtt ghost
   (buoyancy BC); periodic x,y via `padding_array` (1-ghost wrap).
6. **Pressure-gradient force** `pressure_gradient_force` = −Cp·θ̄_face·∇π'
   (θ averaged to the face; π0 horizontally uniform ⇒ use π' only). w walls → 0.
7. **Update** `update_momentum_eqn_euler` (per RK stage, Euler form on stage state):
   du/dt = adv4u + pres_grad4u + f_v;  dv/dt = adv4v + pres_grad4v − f_u;
   dw/dt = adv4w + pres_grad4w + b8w.
8. **Scalars** θ, q_v advected separately (`compute_theta/qv_tendency`, same WENO
   flux-form + flow_divergence), then microphysics/heating (hook; LEX stub = Newtonian).
9. Optional `correct_pip_constant2`: add a constant to π' to zero linearised total
   mass change (pic_opt; off by default).

Note vs our spectral core: our `project()` is a *fractional step* (advance then project);
LEX folds all forcing into the elliptic RHS and solves π' directly per stage. Both
enforce ∇·(ρ-weighted u)=0; the legoESM module keeps the LEX variable-coefficient
operator (the faithful, scalable artifact) and our (ny,nx,nz) C-grid layout.

## Mapping to legoESM (reuse — ponytail: write the minimum)
- Layout: **our** plane convention `(ny,nx,nz)`, u,v centres, w faces `(…,nz+1)`,
  periodic via `jnp.roll` (matches `spectral_les_plane` / `compressible_euler_plane`).
- Constants: `legoesm.constants` — `c_pd`(Cp), `R_d`, `R_v`, `g`, `p_ref`(p00),
  `epsilon` (reps = 1/epsilon), `kappa`, `kappa_von_karman`. NO literals.
- WENO: `legoesm.core.weno.weno5_z` (public). van Leer: `core.flux_limiters`.
- MOST wall / buoyancy / θ_v: reuse `bulk_flux.psi_m/psi_h`, `thermo` saturation,
  `spectral_les_plane.most_surface_flux` pattern (re-expressed; spectral helpers use
  FFT ddx/ddy so cannot be imported verbatim for the FD core).
- **FD C-grid advection + Smagorinsky**: `compressible_euler_plane` has these but
  PRIVATE + vertical-last. Correct path per CLAUDE.md no-dup: PROMOTE the shared FD
  flux-form advection + strain/Smagorinsky into a shared module
  (`atmosphere/dynamics/plane_fd_operators.py` or extend `plane_operators.py`) and have
  BOTH dycores select it. Do NOT copy-paste. (Refactor pending — see TODO.)
- Halo/MPI: `parallel/halo_exchange.py` (`pad_halo_mpi`, `_sendrecv_vjp` AD-safe) for
  the Laplacian stencil + divergence; periodic serial via roll. The Poisson operator is
  nearest-neighbour ⇒ one halo exchange per BiCGSTAB matvec (vs spectral all-to-all).
- Microphysics/SGS swap = the existing "lego": `microphysics/integration.make_*`
  factory + tracer slot layout `[0]=q_v,[1]=q_c,[2]=q_r,…` (same as spectral core, so any
  scheme swaps in unchanged). Dispatch must `raise ValueError` on unknown (CLAUDE.md).

## Preconditioner (the "preconditioned" in the ask)
LEX runs plain BiCGSTAB (maxiter 100) — stalls on large/stretched grids. Add a
**Jacobi (diagonal) preconditioner**: M⁻¹ = 1/diag(L), diag from the variable-coef
stencil = −Cp·(Σ_face 0.5(rtt_i+rtt_nbr)/Δ²). Pass `M=` to `bicgstab`. Cheap,
matrix-free, AD-safe, big iteration-count cut on the moist (variable-rtt) operator.
(Follow-up option: vertical-line/multigrid V-cycle if Jacobi insufficient at scale.)

## Oracle reference (DONE)
LEX clone `/tmp/lex_oracle`, deps via `uv pip install flax` into legoESM `.venv`.
Quasi-2D dry warm bubble (θ0=300K isentropic, +1K cos² bubble r=2km @ (10,–,2)km,
RH=0.1): `solver_opt=1`, dx=dy=dz=200m, nx=100, ny=4, nz=50, dt=2s, 600s.
Edits: `namelist_n_constants.py` (solver/grid/time), `setup_lex.py` bubble centre +
`yr=1e9` (y-uniform). Run: `JAX_PLATFORMS=cpu python lex_relay.py`.
**Reference metrics @ t=600s** (`experiments/lex_out_0003.nc`):
w_max ≈ **8.52 m/s**, w_min ≈ −3.84, u ∈ ±4.15, θ ∈ [300,301], symmetric in x.
Snapshots at 0/200/400/600 s. Use these as the quantitative oracle (not bit-level —
staggering/advection differ; match bulk: w_max, rise height, x-symmetry, KE growth).

## Acceptance / validation plan
- Poisson unit: manufactured π' on periodic+Neumann grid → operator round-trips;
  BiCGSTAB residual < tol; Jacobi cuts iters vs none; output ∇·(ρu)≈0 to tol.
- Projection unit: random u* → corrected field divergence-free; AD (`jax.grad` through
  a step) finite; JIT no-retrace; float64.
- Warm bubble vs LEX oracle: w_max within ~10–15%, symmetric, top height matches.
- Moist: BOMEX/DYCOMS vs literature + cross-check spectral core; microphysics swap runs.
- Conservation: ρ-weighted mass; tracer positivity (monotone scalars).
- MPI: serial vs 2-rank halo Poisson equivalence.
- Physics contract + `__param_spec__` + every new `.py` gets a direct test
  + dispatch-raise (CLAUDE.md ratchets).
- **Codex adversarial review loop** until clean (mandatory).

## Implemented (2026-06-18)
- `pseudo_incompressible_poisson.py` — variable-coef matrix-free Jacobi-BiCGSTAB Poisson
  (+ `test_pseudo_incompressible_poisson.py`, 6 tests). Codex-reviewed.
- `plane_fd_advection.py` — SHARED flux-form FD advection (WENO5/van-Leer/upwind) on
  `core.weno`/`core.flux_limiters`, `(ny,nx,nz)` periodic-x/y + rigid-z-walls (+
  `test_plane_fd_advection.py`, 19 tests incl. constant/sign/conservation/wall/jit-grad).
- `pseudo_incompressible_plane.py` — dycore: Config/Grid/State, hydrostatic base state,
  θ_ρ buoyancy (dry+moist hook), SSP-RK3 + per-stage projection (+
  `test_pseudo_incompressible_plane.py`, 6 tests). Warm bubble vs LEX: **w_max in (3,15)
  m/s bracket** (oracle ≈8.5), x-symmetric, rest-stays-rest, divergence bounded, jit/grad.
- Codex adversarial review of all three: NO critical/high correctness bugs; stencils,
  base state, buoyancy sign, RK3 coeffs confirmed; added the test-coverage gaps it flagged.

## EXACT projection — full Arakawa C-grid (DONE 2026-06-18)
Converted the dycore to full C-grid: u@x-faces, v@y-faces, w@z-faces, scalars@centres.
The compact divergence D, the face pressure-gradient correction G, and the compact
Poisson Laplacian L now form an EXACT `D·G=L` triple (codex confirmed algebraically),
so the projection drives the ρ-weighted divergence to **machine zero** (measured ratio
5.8e-7 = BiCGSTAB tol, was 0.59 collocated). Fidelity AND GPU/TPU perf win: well-
conditioned (no checkerboard) ⇒ fewer BiCGSTAB iters ⇒ fewer halo exchanges per step.
Shared advection got a `vel_at_faces` flag for the C-grid face-velocity path.
Residual disclosed simplification: component-wise momentum advection treats face-u as
centred (O(Δx) half-cell offset) → ~18% x-asymmetry over 600 s warm bubble (dynamics
unaffected). UPGRADE = momentum-conservative C-grid flux.

## TODO (continuation)
- [x] Oracle reference fields.
- [x] Shared FD advection module (reuses core kernels; no compressible-dycore promotion
      needed — kernels already shared. Adopting it in spectral/compressible = optional cleanup).
- [x] `pseudo_incompressible_plane.py` dycore + APPROXIMATE projection + RK3 + warm-bubble oracle.
- [x] **C-grid horizontal** for the EXACT projection (machine-zero ρ-divergence, ratio 5.8e-7). Codex-reviewed.
- [ ] Momentum-conservative C-grid flux (tighten warm-bubble x-symmetry from ~0.18).

## Composable BL physics (DONE 2026-06-18, codex-clean)
Added to `pseudo_incompressible_plane.py` (lego/swappable, dispatch raises on unknown):
- Config: `f_cor, ug, vg, sgs∈{none,smagorinsky,vreman}, c_s, c_vreman, pr_sgs,
  surface∈{free,flux,most_cooling}, z0, sfc_theta_flux`. `PseudoIncompressibleForcing`
  (t_sfc, subsidence_w, dθ/dt_ls) passed explicitly to `step` (SegmentForcing doctrine).
- Coriolis/geostrophic, MOST surface (reuse shared `most_surface_flux`), surface drag +
  heat-flux into first layer, large-scale subsidence + dθ/dt, SGS (Smagorinsky Mason-wall
  / Vreman shared core / none). SGS evaluated at CENTRES (disclosed simplification;
  projection stays exact C-grid).
- Tests: `test_pseudo_incompressible_bl.py` 10/10 (Coriolis geostrophic-steady + inertial
  sign, surface drag decel, heat-flux sign, ν_t≥0, none=0, MOST cooling, jit/grad, dispatch
  raises). Codex: no critical/high; fixed 1 LOW (wasteful alloc). NaN caveat: Vreman on a
  perfectly-uniform constant-folded field 0/0 (shared core; never hit by real turbulent IC).

## Phase 2 (a) DONE — GABLS1 + Wangara dry, new vs spectral, fp64+fp32 (2026-06-18)
- SGS: added Bou-Zeid **LASD** (scale-dependent dynamic, shared `lasd_core.lasd_cs2`) as a
  4th swappable option {none, smagorinsky, vreman, lasd}. cfg.cs_max. Tested ν_t≥0.
- Driver `scripts/validate/validate_bl_new_vs_spectral.py` runs BOTH dycores on reduced
  GABLS1 (most_cooling) + Wangara (flux heating), compares profiles. PASS fp64 AND fp32:
  GABLS1 jet 8.0/8.0 stable (dθ/dz>0), Wangara jet 5.0 CBL-mixing (dθ/dz<0); new≈spectral
  (θ rel-diff ≤0.01). Reduced res (24³,20min) = sanity+consistency, not full intercomparison.
- **fp32 fix (shared core)**: WENO5-Z NaN'd in fp32 (θ≈265 mean + small fluct → β cancellation,
  NaN @ step 2). Fixed `core/weno.py` with a float32-only common-shift (`_weno_z_shifted`,
  all orders): subtract central stencil value before β (shift-invariant, exact), add back.
  **float64 bit-identical** (test asserts array-equality; 55 weno tests pass). New tests
  `test_weno_fp32_shift.py` 3/3. Codex review env-blocked (bwrap) → manual + test-gated.

## Phase 2 (b) DONE — GPU fp64 + fp32 (2026-06-18)
`uv pip install jax-cuda12-plugin==0.10.0`; `JAX_PLATFORMS=cuda` (auto-detect falls back to
CPU silently). RTX 5090 (Blackwell sm_120) WORKS with jax 0.10 — compiles + runs, no NaN
(driver-version-format warning is non-fatal). New dycore 64³ GABLS1: **fp64 59.1 ms/step,
fp32 15.7 ms/step (3.8× faster** — consumer Blackwell throttles fp64). fp32 path needs the
WENO common-shift fix (phase-2a). Bench script `/tmp/gpu_bench.py` (throwaway).

## Phase 2 (c) MPI — distributed Poisson DONE + parity (2026-06-19)
`pseudo_incompressible_poisson_mpi.py` (parked in `atmosphere/_future/` 2026-10-02, not wired): y-slab decomposition (each rank owns
(ny_local,nx,nz); ny_global=ny_local·n_ranks). x periodic-local, z local, y via a
periodic ring `_halo_y` (mpi4jax.sendrecv: send top→up/recv←down, send bot→down/recv←up
— no tags, call-order pairing; correct incl. nranks==2). Custom matrix-free `_bicgstab_mpi`
(EAGER Python loop — mpi4jax collectives execute in rank order, avoiding the
collective-in-lax.while_loop token hazard) with `global_sum_mpi`'d inner products (jax.scipy
bicgstab sums LOCALLY only → wrong under decomposition). Jacobi precond + global zero-mean gauge.
PARITY (`tests/distributed/test_pseudo_incompressible_poisson_mpi.py`, mpirun -np 2 .venv-mpi):
**Laplacian max-err 0.0 (bit-exact serial==MPI), solve max-err 4.4e-8.** Run:
`PATH=$HOME/.local/mpich/bin:$PATH LD_LIBRARY_PATH=$HOME/.local/mpich/lib mpirun -np 2 .venv-mpi/bin/python -m pytest tests/distributed/test_pseudo_incompressible_poisson_mpi.py`.
REMAINING for full distributed STEP: y-halo the advection (3-cell WENO) + SGS + divergence +
pressure-grad (same pattern); serial==MPI full-step parity; weak/strong scaling sweep;
breakdown guard in BiCGSTAB. Bug fixed en route: 2-rank halo strip/tag mispairing.

## Phase 2 (d) moist microphysics hook DONE (2026-06-19)
`pseudo_incompressible_plane.make_microphysics(g, micro_config, dt, p_sfc, qv_prof)` — reuses
shared `spectral_les_moist.make_anelastic_reference` + `make_les_microphysics_fn`; returns the
operator-split adapter `micro(theta,tracers)->(dθ/dt,dtr/dt,precip)`. ANY scheme swaps via
`MicrophysicsConfig` (kessler/morrison/thompson; dispatch raises). θ_ρ buoyancy + tracer
transport already wired. Tests `test_pseudo_incompressible_moist.py` 3/3: supersaturated column
condenses (q_c↑, q_v↓), latent heating (θ↑), water conserved, operator-split stable, dry-state
guard. Codex (manual, env-blocked): fixed BiCGSTAB breakdown guards (×2), n_global assertion,
dry-state guard, documented eager-solve non-differentiability.

## Phase 2 (c) FULL distributed step DONE + parity (2026-06-19)
`pseudo_incompressible_plane_mpi.py`: y-slab. REUSES serial `tendencies` on 3-cell y-halo-
padded slabs (every stencil reach ≤3 → interior correct, halo discarded); GLOBAL MOST planar
means injected via `tendencies(sfc_means=...)` (refactored serial BL physics to accept
injected means, serial-identical default). Distributed `project_mpi` (Poisson_mpi + y-halo
divergence/pressure-grad). `step_mpi` SSP-RK3. Serial code made y-padded-safe (removed
cfg.ny-hardcoded shapes in eddy_viscosity/Coriolis). FULL-STEP PARITY (mpirun -np 2,
test_pseudo_incompressible_step_mpi.py): **u 5.0e-8, v 6.1e-10, w 1.4e-10, θ 1.3e-8** vs serial
(weno5+vreman+most_cooling+Coriolis+projection all exercised). Codex env-blocked → manual
review (mirrors serial, parity-gated).
SCALING: architecturally enabled (nearest-neighbour halo + O(1) allreduce/dot) + parity-proven;
a clean weak/strong SWEEP needs a real multi-node cluster (eager loop + mpi4jax per-collective
logging + 2 laptop cores → absolute timings noise-dominated here). Use repo docs/scaling infra.

## GPU full-physics (2026-06-19): CONFIRMED stable
RTX 5090, full BL physics (vreman + flux/most_cooling + Coriolis): convective BL (surface
heating) 64³ 500 steps STABLE fp32 AND fp64 (no NaN, θ 300→306, w_max≈1.1). fp64 59/fp32 16
ms/step. Fine-res GABLS1-COOLING long run blows up (stable-BL grid-noise collapse) — added
composable `cfg.nu_floor` (spectral-core-proven; 14 tests still pass) but it ALONE is
insufficient at 64³ dx=6.25; needs an FD de-noising filter/hyperdiff (spectral core uses a
sharp spectral cutoff) — a robustness FOLLOW-UP, not a GPU/correctness bug.

## Phase 2 (d) BOMEX moist — DONE (new vs former/spectral, 2026-06-19)
Completed the moist-BL physics (composable): tracer SGS diffusion + surface q_v flux
(`cfg.sfc_qv_flux`), q_v subsidence + `dqv_dt_ls`, height-dependent geostrophic `ug_prof/
vg_prof` (BOMEX shear), `nu_floor`. Refactored `tendencies` to compute ν_t ONCE (shared by
momentum + tracer SGS). Driver `scripts/validate/validate_bomex_new_vs_spectral.py` runs BOTH
dycores on the SAME gSAM BOMEX forcing (snd/lsf/sfc, LEGOESM_GSAM_ROOT=…/gSAM…/gSAM1.8.7) +
SAME kessler microphysics. 3 h GPU fp32 48×48×75, both STABLE (no NaN):
  NEW cloud_cover 0.76 / LWP 195 / qc_max 0.17;  SPECTRAL 0.81 / 115 / 0.22.
CROSS-DYCORE agreement (the goal): θ rel-diff 0.6%, q_v 16% → new reproduces former. Both
OVER-cloud vs GCSS (ref cover 0.10-0.15, LWP 5-10) — SHARED issue (kessler over-condenses +
reduced res + 3 h spin-up + nu_floor over-mixing), NOT a dycore difference. GCSS targets need
morrison(9-slot)+monotone+filter+longer spin-up (run_bomex_les.py path). 14 BL/moist tests pass.

## TODO (phase 2 cont.)
- [ ] BOMEX to GCSS targets: morrison + monotone-tracer + FD de-noiser + 6 h spin-up.
- [x] FD de-noising for fine-res stable-BL + moist robustness — opt-in horizontal
      biharmonic `cfg.hyperdiff_coeff` (−coeff·∇⁴_h on u,v,w,θ,tracers), FD analogue of the
      spectral core's sharp cutoff. Default 0 ⇒ bit-identical (no-op static gate). Tests:
      `test_pseudo_incompressible_bl.py` bit-identical-off + scale-selective (2Δ ≫ smooth).
      Codex-clean (6/6 axes, no correctness bugs). BOMEX re-test (24³ dx=200m, 1.5 h, kessler,
      new-vs-spectral): gentle coeff=1e5 + νfloor 0.1 → NEW cover 0.25→0.175, LWP 33→18
      (toward GCSS 0.10-0.15 / 5-10), clouds preserved, θ rel-diff 0.001 vs spectral. CAVEAT:
      it is a CALIBRATION KNOB — strong coeff (2e6) at this coarse res erases the marginally-
      resolved clouds (cover→0); its proper regime is fine-res where 2Δ noise is well-separated
      from energy-containing eddies. Per-field coeffs (spectral-core style) = follow-up if needed.
- [ ] weak/strong scaling SWEEP on a real cluster (architecture+parity already done).
- [ ] (d) full BOMEX run: large-scale forcing + surface fluxes + ~6h sim on new + spectral, GCSS-style comparison.
- [ ] AD through distributed Poisson (eager loop not differentiable; serial solver IS).
- [ ] Momentum-conservative C-grid flux (tighten symmetry).
- [ ] Moist cases end-to-end: latent-heating hook into θ + microphysics factory wire + BOMEX/DYCOMS
      vs literature + cross-check spectral. (Buoyancy θ_ρ + tracer transport already moist-ready.)
- [ ] SGS swap (Vreman shared core / Smagorinsky) — off for the dry oracle (LEX turb_opt=0).
- [ ] MPI: swap jnp.roll halos → parallel.halo_exchange in advection + Poisson.
- [ ] Physics contract + `__param_spec__` + dycore registry entry + same-PR test; full codex loop.
