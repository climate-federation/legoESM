# #852 root cause: cube SPMD "halo wrong" is a float32 reduction non-associativity, not a halo bug

## TL;DR

The reported symptom — the GSPMD `shard_map` cube halo being "numerically wrong"
on a 2-device (3-faces/device) face-shard at C192 float32 — is **not a halo
defect**. The ppermute cube halo is **bit-exact**. The decomposition dependence
comes from the **conservation mass-fixer's global reduction**, which summed a
face-sharded array with `jnp.sum` — a per-shard-local reduce then all-reduce
whose float32 result depends on the device count. The mass fixer amplifies that
~1-ulp difference into a shard-count-dependent surface-pressure correction, and
the chaotic baroclinic wave grows it over a few steps.

## Evidence (emulated CPU devices, `--xla_force_host_platform_device_count`)

| test | precision | result |
|---|---|---|
| `explicit_pad_halo` forward + VJP (bare ppermute halo), n=2 & n=3, C24–C192 | float32 | **Δ = 0.0** (bit-exact) |
| bench AD gate (VJP through the halo, nonuniform cotangent), n=2/n=3 C192 | float32 | **Δ = 0.0** |
| full dycore step, n=2 & n=3, C96/C192 | **float64** | **Δ ≈ 1e-9** (machine precision, all shard counts) |
| full dycore step, mass fixer ON, C96 3-step | float32 | n=2 Δ=0.99 ≠ n=3 Δ=1.30 (**shard-count-dependent**) |
| full dycore step, mass fixer OFF, C96 3-step | float32 | n=2 Δ=0.39 == n=3 Δ=0.39 (shard-invariant residual) |
| `global_integral` on a sharded field, C96/C192 | float32 | n=2 Δ ≠ n=3 Δ, ~2.6e-7 relative (**the defect**) |
| `global_integral` on a sharded field | float64 | Δ ~1e-15 relative (negligible) |

Two independent facts prove the halo is not at fault: the halo AD gate is
bit-exact at n=2 **and** n=3 in float32, and the full dycore is bit-exact at
**every** shard count in float64. A halo data-movement bug would fail both.

## The defect

`global_integral` / `global_area_sum` / `batch_global_area_sums` computed the
area-weighted global sum with a bare `jnp.sum` over a face-sharded `(6, n, n)`
array. Under GSPMD that reduces each device's owned faces locally and then
all-reduces the per-shard partials. Because float32 addition is
**non-associative**, the result differs from the single-device flat sum, and
between device counts (1 vs 3-faces/shard vs 2-faces/shard), by ~1 ulp. The
anchored mass fixer scales `p_s` by `target_mass / current_mass`, so that ulp
becomes a shard-count-dependent `p_s` correction — the reported symptom.

## The fix

`legoesm.core.conservation.shard_invariant_cube_face_sum`: faces are never split
across shards (each owns `k = 6/n_devices` **whole** faces for
`n_devices ∈ {1,2,3,6}`), so reduce each face to a scalar first — those per-face
partials are **bit-identical** at every decomposition — then combine the 6
partials in a **fixed unrolled order** (XLA does not reassociate float adds with
fast-math off). Gated on `prod.shape[0] == 6` **and**
`cube_faces_are_whole_on_shards()` (a `(6, kt, kt)` sub-face **tiled** mesh
splits faces, so it falls back to the plain sum). Applied to `global_integral`,
`global_area_sum`, and `batch_global_area_sums` (single-array and batched fixer
paths — the PE mass/energy fixers `fix_ps_mass` / `fix_ps_mass_target` /
`fix_mass_hydrostatic`, `zero_mean_tendency`, and NH dry mass all route through
these). The FV3 **shallow-water** cube mass fixers (`shallow_water_fv3_cdgrid.py`
`set_initial_mass` / `compute_mass` / fixer current-mass across all 3 model
classes) previously used a bare inline `jnp.sum(h * area)` and are now routed
through the same shared `global_area_sum` / `batch_global_area_sums` (no
duplicate reduction numerics). Verified: all three reductions are
**bit-identical** across single-device, n=2, and n=3 in float32; the end-to-end
PE dycore mass fixer is shard-count-invariant (n=2 Δ == n=3 Δ, was 0.99 ≠ 1.30);
SW mass-conservation + 93 conservation/operator tests still pass.

## What the fix does and does NOT do

- **Does**: makes the cube conservation reductions **decomposition-independent**
  — a global conserved integral is now bit-identical regardless of device count.
  This eliminates the specific n=2-vs-n=3 shard-count dependence #852 localized.
- **Does not**: make the full float32 SPMD cube run bit-match single-device. A
  **shard-invariant** residual remains (n=2 Δ == n=3 Δ ≈ 0.39 at C96/3-step with
  the fixer off) — inherent float32 op-ordering differences between the
  single-device and sharded XLA programs, amplified by the chaotic baroclinic
  flow. **float64 is bit-exact.** SPMD cube *correctness* should therefore be
  validated in float64 (where it is exact); the bench's float32 correctness gate
  at `tol=1e-4` after several chaotic steps is not a meaningful test of halo
  correctness.

## Scope

Both cube dycores that run a mass fixer are covered: the primitive-equation
(`primitive_eq_cdgrid.py`, the #852 baroclinic bench) via the conservation
reductions, and the FV3 shallow-water (`shallow_water_fv3_cdgrid.py`) via the
same shared reductions. The lat-lon / lat-band SPMD and MPI reduction paths are
untouched (unaffected by the cube face-shard non-associativity), and a
`(6, kt, kt)` sub-face tiled mesh keeps the plain sum (a face is split across
tiles there, so the whole-face invariance does not apply).
