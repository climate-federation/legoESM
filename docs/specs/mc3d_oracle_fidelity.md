# mc3d ↔ microhh `rte-rrtmgp-cpp` Oracle Fidelity

Oracle: microhh `rte-rrtmgp-cpp/src_cuda_rt` forward Monte-Carlo ray tracer
(Veerman et al. 2022). This documents the algorithm-by-algorithm correspondence
between legoESM's pure-JAX `mc3d` port and that CUDA oracle, plus the validation
evidence — the faithfulness record required by the oracle-recipe doctrine
(`docs/ocean/fidelity/oracle_recipe_strategy.md`): **truth tiers (0–2) outrank
oracle byte-matching**, and a direct numeric microhh run needs the CUDA/NVHPC
toolchain which is not available in this environment (Apple-Silicon/CPU; see the
metal-backend memory). We therefore certify fidelity by (a) porting the *same
algorithm* and (b) clearing the analytic truth tiers that the oracle itself
targets.

## Algorithm correspondence

| Oracle (`src_cuda_rt`) | mc3d (JAX) | Match |
|---|---|---|
| Forward photon MC, TOA→surface (`Raytracer.cu` `ray_tracer_kernel`) | `photon_walk.trace_one` / `raytracer_sw` | ✅ same scheme |
| Null-collision (Woodcock/delta) tracking, `k_null` majorant grid (`create_knull_grid`, min 1e-3) | `knull_grid.build_majorant_grid` (per-block max, floor 1e-3) + DDA decomposition tracking in `photon_walk` | ✅ same; mc3d carries coarse indices (DDA) for unbiased per-region majorant |
| Per-g-point optics loop, properties from RRTM-GP | `two_stream.compute_sw_optical_field` (shared with `solve_sw`) + `raytracer_sw.solve_sw_spectral` g-point scan | ✅ same optics source (RRTMGP), serial g-point loop |
| Rayleigh + cloud (Mie CDF) + aerosol scattering; surface BRDF | EXPLICIT Rayleigh (1+mu^2) gas + microhh Mie-CDF cloud (`mie.py`, microhh's `mie_lut_broadband.nc` + `mie_sample_angle`) by per-cell r_eff & band; Lambertian surface | ✅ Rayleigh + Mie-CDF cloud BIT-EXACT to microhh (max|Δcos|=0 vs `mie_sample_angle` on microhh's LUT). aerosol-as-HG 3-way = future (plane path is aerosol-free) |
| Sobol quasi-RNG per thread (cuRAND) | counter-based `threefry` per photon (default) + opt-in Halton low-discrepancy launch (`config.use_qrng`, `qrng.py`) | ✅ QRNG variance-reduction hook (Halton + Cranley-Patterson on the launch; table-driven Sobol = drop-in upgrade). threefry is reproducible cross-backend (Sobol-via-cuRAND is not) |
| LW: `Rte_lw_rt.cu` emission MC | `raytracer_lw` forward thermal-emission MC (4π k B dV vol + π ε B dA sfc, net = ε(abs−emit)), RRTMGP-spectral per-g-point optics + Planck, linear-in-τ in-cell emission | ✅ same emission-MC scheme; RRTMGP-spectral cloud-aware LW (Phase 3b), linear-in-τ source (Phase 3c) |
| Atomic `count_to_flux` scatter-add tally | analog single-deposit + `segment_sum` (integer counts) | ✅ equivalent, exact conservation |
| Per-g-point optics resident (memory) | g-point serial scan, one field resident | ✅ same memory lever |

## Truth-tier validation (outranks oracle-matching)

Cleared by `tests/unit/test_mc3d_{raytracer,plane,lw,parallel,bomex}.py`:

- **Tier 0/1 — analytic.** SW Beer–Lambert slab (vertical + slant `exp(-τ/μ₀)`);
  LW optically-thin isothermal cooling `→ -4kσT⁴dz`; two-stream cross-check vs
  the validated `sw_cell_properties`. These are exactly the limits the oracle's
  estimator must reproduce.
- **Tier 0 — conservation.** Analog MC closes the energy budget exactly
  (integer counts): SW `vol+sfc+tod=1`; LW `net_atmos+net_sfc+OLR+unresolved=0`.
- **Tier 1 — equivariance / no grid bias.** Horizontal-shift equivariance;
  uniform-field heating is horizontally uniform to MC noise.
- **MC convergence.** Error ∝ 1/√N verified.
- **Coarse-majorant unbiasedness.** Coarse `k_null` grid (Phase 1.1) agrees with
  the global-scalar majorant within MC error for vertical and slant/scattering
  cloud fields — a pure speedup, exactly as in the oracle.
- **3D effect (the oracle's raison d'être).** Slant-beam cloud-shadow
  displacement (`scripts/tmp/_mc3d_shadow.py`) and the BOMEX broken-cumulus
  surface shadow (`test_mc3d_bomex`) — absent from the independent-column
  approximation, present here as in the oracle.

## LES integration (BOMEX)

`tests/unit/test_mc3d_bomex.py`: mc3d runs end-to-end through
`make_radiation_physics(model_type="plane")` on a BOMEX (Siebesma 2003) shallow-
cumulus plane state (analytic θ_l/q_t profiles + broken q_c) with RRTMGP
shortwave cloud optics — finite heating, conserves, moves no tracer mass, and a
broken-cloud field shadows the surface beneath cloudy columns. A full multi-hour
BOMEX run needs an external gSAM `CASES/BOMEX` deck (snd/lsf/sfc) not present in
this checkout; the architecture (compressible-plane + ls-forcing + microphysics
+ mc3d) is the `run_rcemip_plane.py` pattern.

## GPU parallelization & memory

- **Memory is not the bottleneck**: per-g-point serial loop keeps one optical
  field resident; photon batches `lax.scan`-chunked; integer-count tallies are
  grid-sized. (See `docs/specs/mc3d_raytracer.md` for the byte budget.)
- **Photon-sharding** (`parallel.py`): replicate the (cheap) per-g-point optics
  per device, shard photons, combine by `pmean` / allreduce-SUM (AD-safe).
  Avoids cross-domain photon migration; wall-clock ~1/n_devices for fixed total
  photons. Benchmark: `scripts/bench/bench_mc3d_raytracer.py`.
- **Coarse majorant** (Phase 1.1) cuts null-collision steps in clear air beside
  thick cloud — the main per-photon speedup, decisive on GPU where the divergent
  walk is otherwise the cost. **Measured** (`scripts/bench/bench_mc3d_raytracer.py`,
  32x32x32 cloud-in-clear field, 64 photons/col, CPU): scalar global majorant
  33.4 s/solve (1.96e3 photons/s) vs coarse `k_null` (8x8x8 blocks) 12.1 s/solve
  (5.43e3 photons/s) = **2.8x** — and it grows with the clear-to-cloud extinction
  contrast and on GPU (warp-coherent free flights).

## Shortwave phase function: HG(g) is flux-faithful

The oracle tracks Rayleigh (1+cos^2) gas and Mie (CDF) cloud scattering
explicitly; mc3d uses Henyey-Greenstein with the RRTMGP total asymmetry `g`.
For the tracer's purpose — broadband FLUXES / heating rates fed to the LES
dycore — this is faithful: a phase function's effect on hemispheric flux is
captured by its asymmetry `g` (the first moment), and HG matches `g` exactly.
In particular HG(g=0) and Rayleigh have the SAME g=0 and the SAME hemispheric
backscatter fraction (1/2), so gas scattering is flux-equivalent; HG(g) captures
the cloud forward-scatter peak's flux-relevant moment. Explicit Rayleigh/Mie-CDF
matter for RADIANCES (sky angular distribution, remote-sensing), not the
flux/heating mc3d produces — they are a radiance-fidelity follow-up, not a
flux-fidelity gap.

## Status

FLUX/heating fidelity (the tracer's purpose — radiative heating for the LES) is
COMPLETE: SW + LW with RRTMGP-spectral cloud-aware per-g-point optics, Woodcock
null-collision tracking with a coarse `k_null` majorant, linear-in-τ LW emission,
analog single-deposit tally with exact conservation, photon-shard parallelism.
All analytic truth tiers cleared; OLR matches the RRTMGP two-stream to O(1);
codex-clean across all phases.

Remaining are RADIANCE-fidelity or performance refinements, not flux-fidelity
gaps:

- Explicit Rayleigh (1+cos²) gas phase: DONE. Mie-CDF cloud phase: DONE —
  microhh's `mie_lut_broadband.nc` + `mie_sample_angle` reproduced BIT-EXACTLY
  (`config.use_mie`; `scripts/validate/compare_mie_vs_microhh.py` reports
  max|Δcos|=0). Aerosol-HG 3-way split = future (plane path is aerosol-free).
- Sobol QRNG: opt-in Halton low-discrepancy launch DONE (`config.use_qrng`);
  table-driven Sobol (Joe-Kuo) is a drop-in upgrade if the tables are added.
  Note: QRNG-launch helps slant/heterogeneous launch; the dominant cloud
  scattering variance is walk-driven (high-dim, where QRNG offers little) — the
  coarse majorant (2.8x measured) is the decisive perf lever.
- Phase 5: differentiable neural emulator (separate spec).
- A head-to-head numeric microhh comparison on a shared cloud field, when a
  CUDA/NVHPC build of `rte-rrtmgp-cpp` is available.
