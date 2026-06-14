# Spectral SH transform → batched GEMM (review lever #2) — design

**Status**: designed, not yet implemented. Future-hardware (Ampere+ fp64
tensor cores); correctness-equivalent on Ginsburg (Turing sm_75 = no fp64
tensor cores; production spectral runs on CPU here). Implement OPT-IN
(`LEGOESM_SH_GEMM=1`), default unchanged, round-trip + parity validated.

## The lever (verified)
`grids/gaussian.py` does the Legendre step as broadcast-multiply + reduce:
- analysis (`sh_analysis*_3d`): `f_m_gathered = f_m[:, ms, :]` (n_lat, n_sh,
  nlev) then `2π·jnp.sum(W[:,:,None]*f_m_gathered, axis=0)` — `einsum('lk,
  lkv->kv')`, a batched-over-k matvec (k in BOTH operands via the gather) →
  memory-bound, ZERO `dot_general`, never hits tensor-core GEMM.
- synthesis (`sh_synthesis*_3d`): `Pnm[...,None]*coeffs` then
  `segment_sum` by `ms` → same memory-bound shape.

The GEMM win needs the contraction over LATITUDE with a free (n, nlev)
output, per zonal wavenumber m → a batched GEMM over m.

## Layout
Flat spectral index `k = _idx(n,m) = n(n+1)/2 + m`, m in 0..n, n in 0..n_max.
`grid.ms[k]` = m. n_sh = (n_max+1)(n_max+2)/2.
Build, at grid construction (one-time, host/numpy):
- `Pnm_bym`, `wPnm_bym`, `wPnm_oc2_bym`, `wDnm_bym`, `Hnm_bym`:
  dense (n_max+1, n_max+1, n_lat), `[m, n, lat] = <matrix>[lat, _idx(n,m)]`
  for m<=n<=n_max else 0 (lower-triangle zero-padded; ~½ waste, but the
  batched GEMM is far faster than the memory-bound reduce on Ampere).
- index maps flat-k ↔ (m,n) for the final scatter/gather:
  `k_of_mn[m,n] = _idx(n,m)` (and the inverse via `ms`, `ns`).

## Analysis (grid→spectral), 3D
```
f_hat = rfft(field, axis=1)/n_lon              # (n_lat, n_lon//2+1, nlev)
f_m   = f_hat[:, :n_max+1, :]                  # (n_lat, m, nlev)  — m-indexed already, NO gather
# batched GEMM over m: contract latitude l, free (n, v)
coeffs_bym = 2π * einsum('mnl,lmv->mnv', wPnm_bym, f_m)   # (m, n, nlev)
# dot_general: batch=[m], contract=[l] → tensor-core batched GEMM
coeffs = coeffs_bym[ms_grid, ns_grid, :]       # scatter (m,n)->flat k  (n_sh, nlev)
```
(`ms_grid`, `ns_grid` are the per-k (m,n) — gather coeffs_bym at [m_k, n_k].)
oc2 / dmu variants: swap `wPnm_bym` → `wPnm_oc2_bym` / `wDnm_bym` (the
combined oc2+dmu path applies both to the same f_m, one FFT).

## Synthesis (spectral→grid), 3D
```
coeffs_bym = zeros(m, n, nlev).at[ms_grid, ns_grid, :].set(coeffs)  # flat->(m,n)
f_m = einsum('mnl,mnv->lmv', Pnm_bym, coeffs_bym)   # contract n, free (l, v) per m
f_hat_full[:, :n_max+1, :] = f_m
field = irfft(f_hat_full * n_lon, n=n_lon, axis=1).real
```
H-variant (θ-derivative): `Pnm_bym` → `Hnm_bym`.

## Validation (Ginsburg, x64, correctness only)
1. round-trip: `sh_analysis_3d(sh_synthesis_3d(coeffs)) ≈ coeffs` (the
   truncation projector idempotence) — BOTH old and new path.
2. parity: new `dot_general` path == old broadcast-reduce path to ~1e-12
   (x64) for analysis, synthesis, oc2, dmu, H — random fields + levels.
3. AD: `jax.grad` through the new path finite (dot_general is AD-safe).
4. uv_from_vordiv round-trip unchanged.

## Wiring
- New `*_bym` fields on `GaussianGrid` (built in the grid factory; gated so
  they cost nothing unless SH_GEMM on — or always built, small).
- `LEGOESM_SH_GEMM` env (read at trace, like LEGOESM_VMIX_BATCHED): the 3D
  kernels branch on it (static Python bool → no double-trace). Default OFF
  (Ginsburg/CPU keep the memory-bound path; it's competitive there).
- Increment plan (gate+codex each): (1) grid `*_bym` tensors + index maps +
  a strided-equivalence test (Pnm_bym[m,n,:]==Pnm[:,_idx(n,m)]); (2)
  analysis GEMM kernel + parity; (3) synthesis GEMM kernel + parity +
  round-trip; (4) oc2/dmu/H variants; (5) wire the env flag + uv_from_vordiv.

## Why opt-in / not a Ginsburg win
RTX 8000 (Turing sm_75): fp64 = 1/32 rate, NO fp64 tensor cores → the GEMM
is not faster than the memory-bound reduce there; production atm spectral
runs on CPU on Ginsburg anyway. The win (cuHPX/dinosaur: 3-10×) is on
A100/H100 fp64 tensor cores. So this is FUTURE-PROOFING + latest-methods
(user ask), validated for correctness on Ginsburg, speedup realized on the
next-gen deployment. Bench on an Ampere+ node when available.
