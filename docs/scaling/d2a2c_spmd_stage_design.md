# Tiled FV3 d2a2c shard_map stage — wiring design (P4 phase-1b)

Status: the d2a2c NUMERICAL CORE is complete + codex-clean + zero-regression
(commits 6f11ce54..8634b336). This doc pins the remaining SPMD *wiring* so it
can be implemented cleanly. The hard part (the device-uniform kernel) is done;
this is parallel-module plumbing that reuses existing tiled-halo infra.

## What exists (reuse, do not rebuild)

- **Device-uniform body** `d2a2c_tile_unified(...)` in `core/fv3_sw_core.py`:
  one flag/mask-driven function producing a tile's (ua,va,uc,vc,ut,vt) from a
  WIDE low-padded covariant block `[a-1:a+nl+4]` + the symmetric metric/edge
  blocks + traced position `a,b` and flags `is_lo_i,is_hi_i,is_lo_j,is_hi_j`.
  Validated: `_assemble_tiled_d2a2c(..., unified=True) == d2a2c_vect`
  bit-for-bit (kt=2/3/4). The flags map to `lax.axis_index('tile_i')==0/==kt-1`
  (same for `tile_j`); `a = axis_index('tile_i')*nl`, `b = axis_index('tile_j')*nl`.
- **Single-device reference** `_assemble_tiled_d2a2c` (in the parity test): the
  bit-exact oracle the shard_map output must match.
- **Global fields** `d2a2c_global_fields` + post-gather strips
  `d2a2c_adjacent_strips` (both public in `core/fv3_sw_core.py`).
- **Tiled halo infra** in `parallel/cubesphere_exchange.py`:
  `_build_tile_connectivity(kt)`, `_build_tiled_tables(kt)`,
  `make_tiled_pad_body(mesh, ndim, halo, with_offsets)` /
  `make_tiled_pad_vector_body(...)`, `_make_exchange_ppermute_tiled(...)`,
  `activate_spmd_halo_backend` (accepts (6,kt,kt) meshes). The Kuhn
  perfect-matching peel gives the 4-round ppermute schedule.
- **Hook** `make_sharded_step` (`parallel/sharded_dynamics.py`) `_tiled_ok`
  gate on env `LEGOESM_TILED_SPMD=1` (EXPERIMENTAL).

## The stage (what to build)

`tiled_d2a2c_spmd(u_d, v_d, cdgrid, mesh)` — returns global uc/vc/ut/vt ==
`d2a2c_vect`. Approach C: the D->A is global, the A->C is tiled.

1. **Global D->A + fields (replicated).** Run `d2a2c_global_fields` in the
   global view (cheap: D->A averages + one h2 vector halo). utmp_pad/vtmp_pad
   and the static metric/edge fields are small (C96 utmp_pad ~480 KB) — keep
   them REPLICATED across devices (PartitionSpec(None)). This is what makes
   approach C cheap: NO dynamic staggered-D-wind halo, and no per-step wind
   halo exchange — each device slices its own tile block from the replicated
   utmp_pad.

2. **Per-tile blocks (in shard_map, via dynamic_slice on the replicated
   field).** Inside the (6,kt,kt) shard_map each device computes
   `a = axis_index('tile_i')*nl`, `b = axis_index('tile_j')*nl` (traced), then
   `lax.dynamic_slice` its WIDE block. WRINKLE: the unified kernel needs
   `[a-1:a+nl+4]` (nl+5) on the staggered axis. dynamic_slice with start `a-1`
   clamps at a=0 (the garbage low cell the kernel overwrites) — but verify
   dynamic_slice clamps START to [0, dim-size] (it does), so build the wide
   field once as `jnp.pad(utmp_pad, low=1)` (n+5) and dynamic_slice `[a, ...]`
   of length nl+5 (== `[a-1:a+nl+4]` of the unwide field). Symmetric metric /
   edge blocks (`ua_pad`,`dx`,`se`,`sw`,`va_pad`,`dy`,`sn`,`ss`) are
   `dynamic_slice` of length nl+4 / nl+2 at `[a, b]`. Cell/staggered metrics
   (cos_sg5,rsin2,cosa_u,...) via `dynamic_slice` of the replicated arrays.
   (Alternative if memory ever matters: shard utmp_pad by tile and use
   `make_tiled_pad_body(halo=3)` for the wide ring — but replication is far
   simpler and the field is tiny; prefer replication.)

3. **Per-tile A->C.** Call `d2a2c_tile_unified(...)` with the blocks + traced
   a/b/flags. Output: each device holds its tile's (ua,va,uc,vc,ut,vt) with the
   two adjacent strips left as base (DEFERRED).

4. **Strip ppermute (the only dynamic exchange).** The vt i-strip (i=0/n-1)
   needs the neighbour tile's ut at the tile's high-j face (1-cell j-halo from
   `tile_j+1`); the ut j-strip (j=0/n-1) needs the neighbour's vt at the high-i
   face (1-cell i-halo from `tile_i+1`). Exchange those 1-cell edges via
   `lax.ppermute` on the `tile_i`/`tile_j` mesh axes (a 1-wide analogue of the
   existing tiled-pad schedule — reuse `_build_tiled_tables(kt)` connectivity),
   then apply `d2a2c_adjacent_strips` LOCALLY per tile on the tile's
   uc/vc/ut/vt + the 1-cell neighbour halo. NB only the EDGE/CORNER tiles touch
   a global face edge (interior tiles have no strip) — but the kernel is
   uniform, so compute the strip for all and let the flags zero it where not on
   the edge, OR (cheaper) the strip math only writes the i=0/n-1 / j=0/n-1 rows
   which exist only on edge tiles' owned faces.

5. **Gather for I/O** (`gather_to_global`); keep sharded for the hot loop.

## Validation (no-regression discipline)

- Unit: a shard_map over 24 host devices (`XLA_FLAGS=
  --xla_force_host_platform_device_count=24`, the existing `/tmp/p4_uava.sbatch`
  pattern) at kt=2 (6*4=24) -> `tiled_d2a2c_spmd == _assemble_tiled_d2a2c`
  (== d2a2c_vect) bit-for-bit. Then kt=3 needs 54 devices (host-sim ok).
- Differentiability: gradient parity vs the global d2a2c_vect through one step
  (the ppermute is `lax.ppermute`, AD-safe).
- **Cube-imprint gate** (MANDATORY before claiming done):
  `scripts/validate/visual_regression.py --check` on a cube-SW run with the
  tiled stage active — passing norms != no edge artifact.
- Bench np24: target beat GSPMD-auto 2253 ms C96 (->~15-30 ms); regen + send
  the scaling PNG (laws_v6).

## Open risks

- dynamic_slice clamp semantics for the wide low cell at a=0 — verify the
  garbage cell is provably overwritten (codex confirmed it is in the kernel;
  re-confirm under dynamic_slice).
- The strip ppermute schedule at corners (a corner defers BOTH strips, and the
  ut j-strip reads vt which the vt i-strip just wrote — preserve that ordering
  per-tile, as `d2a2c_adjacent_strips` already does).
- Output sharding spec for the staggered uc/vc/ut/vt (shared boundary faces
  duplicated across neighbouring tiles — the gather takes the lower tile's copy,
  per `staggered_blocks_to_face`).
