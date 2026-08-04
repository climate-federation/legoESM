Reading additional input from stdin...
OpenAI Codex v0.145.0
--------
workdir: /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
model: gpt-5.6-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019fc696-92f6-72a2-aac7-dd28d2bec64c
--------
user
Round-22: review the atm-lane port (git diff HEAD -- packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py). Both geometry-stack sites (1-D + 2-D) swap broadcast_checked + device_put for the shared checked_shard_put (per-leading-slab gate + make_array_from_callback; single-process plain device_put), covering sharded AND replicated (P()) spec_of modes; n_bands=int(raw[name].shape[0]) per field (schema gate precedes both sites). jnp.asarray wrap dropped (helper returns jax Arrays). Receipts: 54 atm parity tests (4 virtual devices) + 2-proc multicontroller repros both modes (step 6.75 ms, segment 8.57 ms incl. the sharded-geometry default). Attack: (1) replicated-mode semantics change — devices now hold each process's OWN bytes (±gate) instead of process-0's; any consumer requiring bit-identical replicated geometry across processes? (2) n_bands from shape[0] on the polar-mask entries with [:, None] shapes at the 2-D site — leading axis meaning consistent? (3) broadcast_checked now unused in this file — import removed? any other caller left? (4) skipped-by-default tests now needing the virtual-device env documented? End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll inspect the targeted diff and its nearby geometry/schema paths, then trace the helper and tests for the four attack points.
exec
/usr/bin/bash -lc "git diff HEAD -- packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py && rg -n \"broadcast_checked|checked_shard_put|geometry|polar|n_bands|virtual|XLA_FLAGS\" packages/atmosphere/legoesm/atmosphere/dynamics/gcm -g '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 456ms:
diff --git a/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py b/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
index 4371fe976..f7da20d04 100644
--- a/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
+++ b/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
@@ -33,7 +33,7 @@ import jax.numpy as jnp
 from jax.sharding import NamedSharding, PartitionSpec as P
 
 from legoesm.parallel.geometry_consistency import (
-    FLAG_ABSENT, assert_flags_agree, assert_schema_agrees, broadcast_checked,
+    FLAG_ABSENT, assert_flags_agree, assert_schema_agrees, checked_shard_put,
     coerce_bool, coerce_count, config_digest48, name_digest48,
     tree_schema_digest48)
 
@@ -488,15 +488,20 @@ def _build_geometry_stacks(model, mesh, n_dev: int, shard_geometry: bool):
     assert_schema_agrees(ordered_names, n_dev,
                          context="make_sharded_atm_latlon_step",
                          arrays=[raw[n] for n in ordered_names])
-    raw = {
-        name: jnp.asarray(broadcast_checked(
-            raw[name], name, context="make_sharded_atm_latlon_step"))
-        for name in ordered_names
-    }
+    # checked_shard_put replaces broadcast_checked + device_put (the ocean
+    # walls, PR #1457): the broadcast's psum program is [n_processes,
+    # stack] and a numpy device_put onto an all-process sharding pays
+    # jax's whole-array assert_equal. Both are SOFT walls here (2-D
+    # geometry, ~MB-scale fields) but the shared helper removes them
+    # uniformly; the per-leading-slab gate keeps the divergence contract
+    # (schema gate above agreed shapes, so widths match).
     spec_of = lat_spec if shard_geometry else (lambda _arr: P())
     stacks = {
-        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
-        for name, arr in raw.items()
+        name: checked_shard_put(
+            raw[name], name, NamedSharding(mesh, spec_of(raw[name])),
+            context="make_sharded_atm_latlon_step",
+            n_bands=int(raw[name].shape[0]))
+        for name in ordered_names
     }
     stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
     return template, array_field_names, stacks, stacks_spec
@@ -1582,17 +1587,16 @@ def _build_geometry_stacks_2d(model, mesh, p_lat: int, p_lon: int,
     assert_schema_agrees(ordered_names, p_lat * p_lon,
                          context="make_sharded_atm_latlon_step_2d",
                          arrays=[raw[n] for n in ordered_names])
-    raw = {
-        name: jnp.asarray(broadcast_checked(
-            raw[name], name, context="make_sharded_atm_latlon_step_2d"))
-        for name in ordered_names
-    }
+    # 2-D twin of the checked_shard_put swap above (see that note).
     spec_of = tile_spec if shard_geometry else (lambda _arr: P())
     stacks = {
-        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
-        for name, arr in raw.items()
+        name: checked_shard_put(
+            raw[name], name, NamedSharding(mesh, spec_of(raw[name])),
+            context="make_sharded_atm_latlon_step_2d",
+            n_bands=int(raw[name].shape[0]))
+        for name in ordered_names
     }
-    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
+    stacks_spec = {name: spec_of(raw[name]) for name in ordered_names}
     return template, array_field_names, stacks, stacks_spec
 
 
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:9:``_spmd_lat_psum_or_none`` for global reductions, ``latlon_mpi`` band-geometry
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:10:slicers) and writes NEW only the atm-specific state layout + geometry-band glue.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:35:from legoesm.parallel.geometry_consistency import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:36:    FLAG_ABSENT, assert_flags_agree, assert_schema_agrees, checked_shard_put,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:407:def atm_latlon_geometry_bytes(grid, n_devices: int) -> dict:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:408:    """Per-device geometry residency of the lat-band SPMD step, computed from
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:415:    ``sharded_per_device_bytes``: the ``shard_geometry=True`` layout — each
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:418:    uniform).  Excludes the optional polar-filter mask stacks (two
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:429:        "n_geometry_fields": len(names),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:435:def _build_geometry_stacks(model, mesh, n_dev: int, shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:437:    polar-filter masks) over a leading band axis and lay them out on ``mesh``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:439:    ``shard_geometry=False`` (the historical layout): every stack is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:441:    geometry and the body indexes its own band at ``axis_index``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:443:    ``shard_geometry=True`` (M2b): every stack is sharded ``P("lat", ...)``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:448:    (per-device geometry bytes drop by ``n_dev`` —
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:449:    :func:`atm_latlon_geometry_bytes`).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:462:    # Per-band polar-filter masks (only when the filter is on): slice the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:465:    # self._polar_mask (also None), no filter.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:467:    if model._polar_mask is not None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:468:        raw["__polar_mask"] = jnp.stack(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:469:            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(n_dev)],
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:471:        raw["__polar_mask_v"] = jnp.stack(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:472:            [model._polar_mask_v[r * nl:r * nl + nl + 1] for r in range(n_dev)],
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:474:    # #1362: the band geometry above is RECOMPUTED per process from the same
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:480:    # blind: a REAL divergence (different polar masks = different filtering =
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:483:    # The schema gate matters more here than in the ocean lane: the polar-mask
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:484:    # entries are CONDITIONAL on ``model._polar_mask``, so a per-process
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:491:    # checked_shard_put replaces broadcast_checked + device_put (the ocean
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:495:    # geometry, ~MB-scale fields) but the shared helper removes them
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:498:    spec_of = lat_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:500:        name: checked_shard_put(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:503:            n_bands=int(raw[name].shape[0]))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:511:                         perm_north, physics_fn, shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:519:    layout.  ``shard_geometry`` selects the geometry index (STATIC Python
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:528:        gi = 0 if shard_geometry else jax.lax.axis_index(axis)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:531:        pmask = (stacks_local["__polar_mask"][gi]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:532:                 if "__polar_mask" in stacks_local else None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:533:        pmaskv = (stacks_local["__polar_mask_v"][gi]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:534:                  if "__polar_mask_v" in stacks_local else None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:543:            polar_mask=pmask, polar_mask_v=pmaskv,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:613:    "has_polar_mask", "has_polar_mask_v",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:615:    "has_physics_fn", "has_on_segment", "has_phys_state", "shard_geometry",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:622:                      shard_geometry=None, where: str) -> None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:647:      rejected the swapped peer while the valid one walked into the geometry
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:664:    * ``shard_geometry`` -- changes the geometry stacks' ``PartitionSpec``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:711:        # and `_build_geometry_stacks` broadcasts; agreeing their dtypes and
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:722:        # Rank-local CACHE presence. `_build_geometry_stacks` conditions on
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:723:        # `_polar_mask` and then slices `_polar_mask_v` unconditionally, so a
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:727:        float(getattr(model, "_polar_mask", None) is not None),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:728:        float(getattr(model, "_polar_mask_v", None) is not None),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:735:        _flag(shard_geometry, "shard_geometry"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:745:    # A polar mask present without its v-face twin would blow up mid-build on
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:748:    if (getattr(model, "_polar_mask", None) is not None
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:749:            and getattr(model, "_polar_mask_v", None) is None):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:751:            f"{where}: model._polar_mask is set but model._polar_mask_v is "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:752:            f"None; the band/tile geometry build slices BOTH, so this would "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:753:            f"fail mid-build (after the entry gate, before the geometry "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:860:                                 shard_geometry: bool = False):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:867:    UN-jitted ``model._step_cgrid_impl`` on the band geometry + band polar masks
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:871:    atm-NEW is only the 6-field state/geometry walk. ``check_vma=False`` (the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:877:    RK stage on the BAND geometry (``_step_cgrid_impl`` routes the band grid into
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:896:    ``shard_geometry`` (M2b, default ``False`` = historical layout,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:897:    byte-identical): ``True`` lays the band-geometry stacks out SHARDED
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:899:    geometry slice instead of a replicated all-band copy (1/n_dev the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:900:    bytes, :func:`atm_latlon_geometry_bytes`).  The body consumes the SAME
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:908:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:944:    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:945:        model, mesh, n_dev, shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:950:        shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1027:                                    shard_geometry: bool = True):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1047:    ``shard_geometry=True`` (default — a NEW API, no historical layout to
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1048:    preserve): each device holds ONLY its own band's geometry slice
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1050:    bit-identical numerics, 1/n_dev the geometry bytes
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1051:    (:func:`atm_latlon_geometry_bytes`).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1060:    C-grid step with the model's own geometry, same ``(state, all_finite)``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1081:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1128:    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1129:        model, mesh, n_dev, shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1133:        shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1190:    the band geometry; decomposition-invariant with no collectives.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1291:                                      band-SHARDED geometry) — one host
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1537:def _build_geometry_stacks_2d(model, mesh, p_lat: int, p_lon: int,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1538:                              shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1540:    polar-filter masks at ``p_lon == 1``) over LEADING ``(p_lat, p_lon)``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1542:    :func:`_build_geometry_stacks`.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1544:    ``shard_geometry=True``: stacks are sharded ``P("lat", "lon", ...)`` on
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1547:    ``shard_geometry=False``: replicated (``P()``) stacks, indexed at
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1564:    # Per-tile polar-filter masks: p_lon > 1 is refused by the factories
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1568:    if model._polar_mask is not None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1571:                "atm 2-D SPMD tiling: use_polar_filter=True with p_lon > 1 "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1572:                "is not wired — the polar filter FFTs the full longitude "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1575:        raw["__polar_mask"] = jnp.stack(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1576:            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(p_lat)],
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1578:        raw["__polar_mask_v"] = jnp.stack(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1579:            [model._polar_mask_v[r * nl:r * nl + nl + 1]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1582:    # #1362, 2-D twin of the guard in _build_geometry_stacks -- same
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1590:    # 2-D twin of the checked_shard_put swap above (see that note).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1591:    spec_of = tile_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1593:        name: checked_shard_put(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1596:            n_bands=int(raw[name].shape[0]))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1605:                            shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1614:    band/tile step on the tile geometry, convert both staggers back.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1621:        if shard_geometry:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1629:        pmask = (stacks_local["__polar_mask"][gi, gj]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1630:                 if "__polar_mask" in stacks_local else None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1631:        pmaskv = (stacks_local["__polar_mask_v"][gi, gj]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1632:                  if "__polar_mask_v" in stacks_local else None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1643:            polar_mask=pmask, polar_mask_v=pmaskv,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1668:    :func:`_agree_spmd_entry` having already agreed ``use_polar_filter`` (and
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1671:    `use_polar_filter` ALSO controls whether the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1672:    `__polar_mask`/`__polar_mask_v` entries exist in the geometry field list,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1677:    polar = bool(getattr(model.config, "use_polar_filter", False))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1678:    if p_lon > 1 and polar:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1680:            "atm 2-D SPMD tiling: use_polar_filter=True with p_lon > 1 is "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1681:            "not wired — the polar filter FFTs the full longitude circle "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1689:                                    shard_geometry: bool = True):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1698:    ``model._step_cgrid_impl`` on the tile geometry + per-tile pole masks,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1713:    evaluated per RK stage on the TILE geometry — decomposition-invariant
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1719:    ``shard_geometry=True`` (default — new API, no historical layout):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1720:    per-device tile geometry slices (``P("lat", "lon")`` stacks);
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1728:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1743:        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1747:        shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1784:                                       shard_geometry: bool = True):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1797:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1825:        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1829:        shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:49:from legoesm.grids.polar_filter import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:50:    compute_polar_filter_mask,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:89:    use_polar_filter: bool = False
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:90:    polar_filter_cutoff_deg: float = 60.0
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:91:    polar_filter_max_wave_speed: float = 300.0
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:403:    # inert) form, so routing it would change the north-fold value on a tripolar grid.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:498:        # Precompute polar filter masks (cached, not traced): one for
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:506:        if self.config.use_polar_filter:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:507:            self._polar_mask = compute_polar_filter_mask(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:509:                max_wave_speed=self.config.polar_filter_max_wave_speed,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:510:                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:512:            self._polar_mask_v = compute_polar_filter_mask(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:514:                max_wave_speed=self.config.polar_filter_max_wave_speed,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:515:                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:519:            self._polar_mask = None
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:520:            self._polar_mask_v = None
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:603:            if self._polar_mask is not None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:604:                dh = fourier_filter(dh, self.grid, self._polar_mask)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:605:                du_interior = fourier_filter(du[:, :-1], self.grid, self._polar_mask)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:608:                # mirrors the PE dycore's polar_mask_v treatment.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:609:                dv = fourier_filter(dv, self.grid, self._polar_mask_v)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_spectral.py:35:rotation about the polar axis (``u = Omega a cos(lat)``, ``v = 0``) each
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:21:  equations in spherical geometry. J. Comput. Phys., 102, 211-224.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:12:(``jnp.roll``), polar walls (``v = 0`` at the pole rows) and a domain-total mass
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:99:    The face-source weights live in ``nest`` (built once at construction, geometry
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:166:    them from the relaxation weights so the band geometry stays single-sourced.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:324:    it with the FULL standalone :class:`CGridLatLonShallowWaterModel` (polar
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:358:    (geometry / static config), so callers that JIT should close over them via
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:392:    # --- Parent: standalone global SW update (periodic lon, polar walls). ---
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:393:    # NOTE: this in-built parent step uses the BARE tendencies (no polar filter);
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:883:            # compute policy with x64 enabled, the f64 mesh-geometry
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:581:    # grid amplifies polar noise by 1/cos².  Instead we keep KE·cos²φ and
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:65:from legoesm.grids.polar_filter import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:66:    compute_polar_filter_mask,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:154:    use_polar_filter: bool = False
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:155:    polar_filter_cutoff_deg: float = 60.0
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:156:    polar_filter_max_wave_speed: float = 300.0
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:843:        # Precompute polar filter masks: one for cell-centered fields
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:851:        if self.config.use_polar_filter:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:852:            self._polar_mask = compute_polar_filter_mask(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:854:                max_wave_speed=self.config.polar_filter_max_wave_speed,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:855:                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:857:            self._polar_mask_v = compute_polar_filter_mask(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:859:                max_wave_speed=self.config.polar_filter_max_wave_speed,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:860:                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:864:            self._polar_mask = None
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:865:            self._polar_mask_v = None
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:930:        SPMD body passes the BAND geometry so a lat-dependent physics (e.g.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:972:        sigma_coord / polar masks. The lat-band SPMD body calls
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:973:        ``_step_cgrid_impl`` directly with the BAND geometry (un-jitted, so the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:990:        polar_mask=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:991:        polar_mask_v=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1002:        ``grid`` / ``sigma_coord`` / ``polar_mask`` / ``polar_mask_v`` default
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1004:        the BAND geometry + band polar masks so the whole step runs on the band;
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1012:        if polar_mask is None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1013:            polar_mask = self._polar_mask
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1014:        if polar_mask_v is None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1015:            polar_mask_v = self._polar_mask_v
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1069:            if polar_mask is not None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1070:                dT = fourier_filter_3d(dT, grid, polar_mask)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1071:                dps = fourier_filter(dps, grid, polar_mask)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1083:                du_int = fourier_filter_3d(du[:, :-1, :], grid, polar_mask)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1096:                dv = fourier_filter_3d(dv, grid, polar_mask_v)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1102:                # polar tracer advection past its CFL.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1105:                        name: fourier_filter_3d(dq_field, grid, polar_mask)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_latlon_cgrid.py:152:    handful of additional lat-lon-specific knobs (polar-filter
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_latlon_cgrid.py:167:    # has the full polar-filter scaffolding -- the NH port reuses it
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_latlon_cgrid.py:168:    # via ``apply_polar_filter`` (TODO: wire in v1).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_latlon_cgrid.py:169:    use_polar_filter: bool = False
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_latlon_cgrid.py:170:    polar_filter_cutoff_deg: float = 70.0
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_latlon_cgrid.py:242:    # intermediate stage can deliver a state whose polar v-faces
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_latlon_cgrid.py:322:    # telescopes for ANY J (periodic lon; v = 0 at the polar faces via

exec
/usr/bin/bash -lc "sed -n '1,280p' packages/atmosphere/legoesm/legoesm/parallel/geometry_consistency.py && sed -n '435,510p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py && sed -n '1535,1605p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py && rg -n \"def (lat_spec|tile_spec)|lat_spec\\s*=|tile_spec\\s*=|broadcast_checked\\(\" packages/atmosphere/legoesm -g '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 2 in 216ms:
sed: can't read packages/atmosphere/legoesm/legoesm/parallel/geometry_consistency.py: No such file or directory

exec
/usr/bin/bash -lc "rg --files | rg 'geometry_consistency\\.py"'$|sharded_atm_latlon_step'"\\.py"'$'"' && rg -n \"def (lat_spec|tile_spec)|lat_spec\\s*=|tile_spec\\s*=|broadcast_checked\\(\" packages -g '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1207ms:
tests/unit/test_geometry_consistency.py
packages/core/legoesm/parallel/geometry_consistency.py
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
packages/coupler/legoesm/driver/sharded_operator_split_step.py:483:            jnp.asarray(broadcast_checked(
packages/core/legoesm/parallel/geometry_consistency.py:528:def broadcast_checked(arr, name: str, *, context: str) -> np.ndarray:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:112:def lat_spec(arr) -> P:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1417:def tile_spec(arr) -> P:

exec
/usr/bin/bash -lc "sed -n '430,660p' packages/core/legoesm/parallel/geometry_consistency.py; sed -n '88,155p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; sed -n '1390,1445p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; sed -n '435,510p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; sed -n '1535,1605p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 196ms:
    follow-up rather than written blind — see #1405.

    ``jax.core.trace_state_clean`` was removed from the public ``jax.core`` in
    jax 0.7 and survives only as ``jax._src.core``, so this reads the private
    module.  The ``except`` returns False — i.e. the gate RUNS and the traced
    lane crashes loudly again — deliberately: for a correctness gate a loud
    crash beats a silent skip.  ``tests/unit/test_geometry_consistency_trace_
    gate.py`` asserts this returns True inside ``jax.jit`` AND inside
    ``lax.scan``, so a JAX version that moves the symbol turns CI red first.
    """
    try:
        from jax._src import core as _jax_core
        return not _jax_core.trace_state_clean()
    except Exception:  # pragma: no cover - JAX internal moved; test goes red
        return False


def assert_flags_agree(names, values, *, context: str) -> None:
    """Raise unless every process agrees on a tuple of rank-local CONFIG flags.

    Call this BEFORE any rank-local ``raise`` that inspects per-process
    config.  Otherwise one process can reject its config and exit while its
    peers proceed into a collective and block forever — a collective-ORDER
    violation whose symptom (hang vs backend error) is backend-dependent
    (codex 2026-07-29, blocker 1).

    ``names`` and ``values`` must be STATIC tuples written at the call site,
    so the payload length is fixed by the code path rather than by data.

    No-op under a JAX trace — see :func:`in_jax_trace` (#1405).
    """
    if jax.process_count() <= 1 or in_jax_trace():
        return
    from jax.experimental import multihost_utils

    payload = np.array(
        [float(len(values)), name_digest48(names),
         *(float(v) for v in values)], dtype=np.float64)
    gathered = multihost_utils.process_allgather(payload)
    if not bool(np.all(gathered == gathered[0])):
        raise RuntimeError(
            f"{context}: per-process CONFIG differs across processes "
            f"(flags {list(names)} -> gathered {gathered.tolist()}). Every "
            f"process must be built from the same config; refusing before "
            f"any rank-local rejection so the failure is symmetric rather "
            f"than a hang.")


def assert_schema_agrees(names, n_dev, *, context: str, arrays=None) -> None:
    """Raise unless every process agrees on the geometry field SCHEMA.

    ``names`` must be an ORDERED sequence — the per-field
    :func:`broadcast_checked` calls that follow are matched positionally
    across processes, so a reordering is itself a divergence worth catching.

    Pass ``arrays`` (the per-name arrays, same order) so the gate also covers
    each field's DTYPE CLASS and NDIM.  Those decide the per-field payload
    SHAPE in :func:`broadcast_checked`, so leaving them out lets a
    bool-vs-float disagreement slip past this gate and deadlock in the
    per-field gather instead of raising here.

    No-op when ``jax.process_count() == 1``.
    """
    if jax.process_count() <= 1:
        return
    from jax.experimental import multihost_utils

    names = list(names)
    if arrays is None:
        kinds, ndims = (), ()
    else:
        # Read dtype/ndim from array METADATA, never via np.asarray: a jax
        # array exposes both without materialising, so forcing a host copy
        # here would add a device sync per field AND could itself fail
        # (transfer error / OOM) BEFORE the collective below — reintroducing
        # the very "one rank exits while a peer blocks" hazard this gate
        # exists to remove (codex round-2 minor). `broadcast_checked` does
        # the single real materialisation later.
        #
        # `_dtype_kind_and_ndim` also survives a plain Python scalar, which a
        # bare `a.dtype` read did not (codex round-3, minor 3): no production
        # caller passes one today, but an AttributeError HERE would be a
        # rank-local raise BEFORE the collective, i.e. a hang rather than a
        # clear failure.
        described = [_dtype_kind_and_ndim(a) for a in arrays]
        kinds = [d[0] for d in described]
        ndims = [d[1] for d in described]
    gathered = multihost_utils.process_allgather(
        schema_fingerprint(names, n_dev, kinds, ndims))
    if not bool(np.all(gathered == gathered[0])):
        raise RuntimeError(
            f"{context}: the band-geometry SCHEMA differs across processes "
            f"(field list / x64 setting / device count / per-field dtype "
            f"class / ndim — gathered {gathered.tolist()}). Fix the "
            f"per-process config before sharding; the per-field checks "
            f"assume one schema.")


def broadcast_checked(arr, name: str, *, context: str) -> np.ndarray:
    """Verify ``arr`` agrees across processes, then broadcast process 0's bytes.

    Multi-process: returns a host ``np.ndarray`` that is bit-identical on
    every process, safe to hand to a replicated ``device_put``.
    Single-process: returns ``arr`` ITSELF, untouched — no collectives, no
    host round trip, no dtype/weak-type change.

    The fingerprint compares structural entries exactly; value entries
    EXACTLY for integer/bool arrays and to ``rtol=1e-5`` for float arrays.

    Integer/bool arrays (masks, index tables) are exact data, not autotuned
    arithmetic: their BYTES are fingerprinted so a positional difference is
    caught.  Moment-only compares are blind to a permutation — a bool mask's
    ``(sum, sumsq, absmax)`` is identical for every arrangement with the same
    true-count (codex round-5).  A mask that genuinely differs across
    processes means different wet domains = different physics: refusing is
    the correct outcome, not a false alarm.

    Residual, documented: a float divergence preserving sum, sum-of-squares
    AND absmax to ``rtol`` is not detected.  Band grids are analytic in
    lat/lon, so any real inconsistency moves those moments.
    """
    # EARLY return, BEFORE np.asarray: single process has nothing to compare,
    # and converting here would force a device->host->device round trip and
    # strip weak-type metadata on a 1-process mesh. The ocean lane already
    # held host arrays so it was unaffected, but the atmosphere lane passes
    # `jnp.stack` results straight in and WAS regressed by an unconditional
    # conversion (codex 2026-07-29, major 3). Return the caller's object
    # untouched.
    if jax.process_count() <= 1:
        return arr
    from jax.experimental import multihost_utils

    host = np.asarray(arr)
    flat = host.ravel()
    is_exact = host.dtype.kind in "biu"
    # FIXED-WIDTH payloads (see _STRUCT_WIDTH/_VALS_WIDTH): the gathered shape
    # must never depend on rank-local data, or two processes can enter this
    # collective with different shapes and hang. Shape is folded in as a
    # digest rather than splatted, so an ndim difference cannot change the
    # length either.
    struct = np.zeros(_STRUCT_WIDTH, dtype=np.float64)
    struct[0] = float(host.ndim)
    struct[1] = float(np.dtype(host.dtype).num)
    struct[2] = float(1.0 if is_exact else 0.0)
    struct[3] = float(host.size)
    struct[4] = name_digest48([str(d) for d in host.shape])
    vals = np.zeros(_VALS_WIDTH, dtype=np.float64)
    if is_exact:
        vals[0] = content_hash48(host)
    else:
        finite = flat[np.isfinite(flat)]
        f64 = finite.astype(np.float64)
        # Non-finite COUNT is structural: a NaN appearing on one process only
        # must not be averaged away by the moment compare below.
        struct[5] = float(flat.size - finite.size)
        vals[0] = float(f64.sum()) if f64.size else 0.0
        vals[1] = float((f64 * f64).sum()) if f64.size else 0.0
        vals[2] = float(np.abs(f64).max()) if f64.size else 0.0

    g_struct = multihost_utils.process_allgather(struct)
    g_vals = multihost_utils.process_allgather(vals)
    struct_ok = bool(np.all(g_struct == g_struct[0]))
    if is_exact:
        vals_ok = bool(np.all(g_vals == g_vals[0]))
    else:
        vals_ok = bool(np.allclose(g_vals, g_vals[0],
                                   rtol=_FLOAT_RTOL, atol=0.0))
    if not (struct_ok and vals_ok):
        raise RuntimeError(
            f"{context}: geometry field {name!r} DIVERGES across processes "
            f"(struct_ok={struct_ok}, vals_ok={vals_ok}, "
            f"exact_dtype={is_exact}, gathered={g_vals.tolist()}) — a real "
            f"config/grid inconsistency, not autotune noise; refusing to "
            f"broadcast process 0 over it.")
    return np.asarray(multihost_utils.broadcast_one_to_all(host))


# --- assert-free sharded puts + per-band gates (2026-08-03, ocean walls) ----
# Three stacked multicontroller walls were found on the ocean lane (codex
# r14-r19; PR #1457): (1) broadcast_one_to_all of a band stack lowers to an
# [n_processes, stack] psum program (nd x 849 MB at LL2304 L20 — 81.5 GB at
# 96 procs); (2) jax.device_put of a NUMPY array onto an all-process
# sharding internally runs multihost_utils.assert_equal on the FULL array
# ([n_proc, field] landing on ONE device: fits under an 80 GB A100 up to
# ~64 procs, dies at 96 — jax _src/dispatch.py::_device_put_sharding_impl);
# (3) a concrete sharded-global array captured by an OUTER trace (jit-of-
# jit) becomes an MLIR constant whose value cannot be fetched for
# non-addressable arrays. The helpers below remove (1) and (2) — (3) is the
# callers' aux-threading contract, see make_sharded_ocean_step.

def band_fingerprint(host, n_bands):
    """Per-band fingerprint of a band-STACKED field (leading axis n_bands).

    PREREQUISITE: ``n_bands`` (and each field's dtype class / shape) must
    already be schema-gated across processes (:func:`assert_schema_agrees`)
    — the payload widths depend on it, and mismatched widths would hang the
    allgather rather than raise.

    Exact dtypes (int/bool/uint): one positional 48-bit byte digest per
    band. Floats: per-band ``[sum, sum_of_squares, absmax]`` of finite
    entries plus per-band non-finite counts folded into ``struct``.
    Per-band (not whole-array) because each process's OWN bytes become the
    live inputs for the bands it owns under the assert-free put: a
    band-local drift must not hide in a whole-array sum (codex r14).
    DOCUMENTED RESIDUALS: a within-band float change preserving all three
    moments to rtol, and non-finite entries changing position/kind at a
    fixed per-band count, pass the float gate.
    """
    host = np.asarray(host)
    if host.ndim == 0 or host.shape[0] != n_bands:
        raise ValueError(
            f"band_fingerprint: leading axis "
            f"{host.shape[0] if host.ndim else '<0-d>'} != n_bands "
            f"{n_bands}")
    is_exact = host.dtype.kind in "biu"
    struct = [float(host.ndim), *map(float, host.shape),
              float(np.dtype(host.dtype).num)]
    if is_exact:
        vals = np.array([content_hash48(host[b]) for b in range(n_bands)],
                        dtype=np.float64)
    else:
        per_band = []
        for b in range(n_bands):
            flat = host[b].ravel()
            finite = flat[np.isfinite(flat)]
            f64 = finite.astype(np.float64)
            struct.append(float(flat.size - finite.size))
            per_band.extend([
                float(f64.sum()) if f64.size else 0.0,
                float((f64 * f64).sum()) if f64.size else 0.0,
                float(np.abs(f64).max()) if f64.size else 0.0,

    * ``mesh is None`` SKIPS the collective entirely, so a process that
      received ``None`` while its peers received a mesh leaves them blocked.
    * the STATE SCHEMA — one replication per leaf, so a tracer dict differing
      in key set, order or shape across processes produces mismatched
      schedules (codex round-4, blocker 5).  ``tree`` folds that whole schema
      into ONE fixed-width digest via :func:`tree_schema_digest48`.

    Cost: a 9-float allgather per SEGMENT BOUNDARY (never inside the step hot
    loop), alongside a full-state gather that is orders of magnitude larger —
    and an exact no-op under a single process.
    """
    names, sizes = _mesh_axis_terms(mesh)
    assert_flags_agree(_MESH_ENTRY_FLAGS, (
        float(mesh is not None),
        float(mesh.devices.size if mesh is not None else 0),
        float(len(names)),
        name_digest48(names),
        name_digest48(sizes),
        float(tree is not None),
        tree_schema_digest48(tree) if tree is not None else FLAG_ABSENT,
    ), context=where)


def lat_spec(arr) -> P:
    """``P("lat", None, ...)`` for an array sharded on its leading (lat) axis."""
    return P("lat", *((None,) * (arr.ndim - 1)))


def shard_state_atm_latlon(
    state: CGridLatLonHydrostaticState, mesh,
) -> CGridLatLonHydrostaticState:
    """Lay out a C-grid hydrostatic atm state for the lat-band shard_map.

    Cell/u leaves (``u, T, p_s, phis`` + every tracer; all leading-dim
    ``n_lat``) shard ``P("lat", None, ...)``. The staggered ``v`` (leading dim
    ``n_lat+1``) drops its top pole-wall face -> ``v_lower = v[:n_lat]``
    (``n_lat`` rows, divisible by the device count) sharded the same way; the
    dropped face is the north pole wall (``v[n_lat] == 0`` after any step) and
    is reconstructed inside the body. Mirrors ``ocean.shard_state_latlon`` but
    walks the 6-field atm pytree (bare arrays + a tracers dict, no masks).
    """
    # FIRST statement: the SCATTER runs one placement per leaf, so its
    # SCHEDULE is rank-local data (a tracer dict differing in key set / order
    # / shape across processes). #1362 round 4, blockers 5-6.
    _agree_mesh_entry(mesh, state, where="shard_state_atm_latlon")
    from legoesm.parallel.latlon_spmd import shard_leaf

    _mp = jax.process_count() > 1

    def _put(arr):
        # shard_leaf: single-process -> device_put (byte-unchanged); multi-
        # controller -> per-process local band via make_array_from_process_local_data
        # (no all-gather, no transient global replica — issue #1100).
        return shard_leaf(arr, NamedSharding(mesh, lat_spec(arr)), multiprocess=_mp)

    n_lat = state.T.shape[0]
    v_lower = state.v[:n_lat]
    return state._replace(
        u=_put(state.u),
        v=_put(v_lower),
        T=_put(state.T),
        p_s=_put(state.p_s),
        phis=_put(state.phis),
        tracers={k: _put(val) for k, val in state.tracers.items()},
    )


# becomes a periodic ring (cyclic ppermute — the wrap IS the roll
# permutation).  Staggered ownership mirrors the 1-D v convention:
#
#   * v (n_lat+1 rows)   -> v_lower = v[:n_lat]; each tile's north boundary
#     face is the lat-neighbour's v_lower[0] (reconstruct_vface_lower).
#   * u (n_lon+1 columns) -> u_left = u[:, :n_lon]; each tile's east seam
#     face is the lon-neighbour's u_left[:, 0] (reconstruct_uface_left) —
#     the +1 seam column is OWNED by the tile whose slice starts there and
#     reconstructed on the periodic wrap (u[:, n_lon] == u[:, 0] identity).
#
# The step body is the SAME un-jitted ``model._step_cgrid_impl`` the band
# path runs: all lon-direction neighbour access inside the operators already
# routes through the backend-dispatched ``pad_lon_cgrid`` / ``pad_halo_latlon*``
# (which this lane arms with the 2-D mesh), so no operator numerics are
# duplicated here.  Corner (diagonal) dependencies compose through the
# sequential lat-then-lon exchanges inside ``make_latlon_2d_pad_body``; the
# C-grid chain has no explicit-diagonal stencil (vertex circulations combine
# lat-padded u with lon-padded v).
#
# The 1-D band lane above is UNTOUCHED and remains the default production
# path (choose_latlon_2d_topology returns (N, 1) whenever the band is
# FEASIBLE — the 2-D pad's pole-fold lon-all_gathers outweigh its perimeter
# advantage until the partner-ppermute fold lands; the 2-D lane is for the
# beyond-band regime n_devices > n_lat/min_tile or indivisible n_lat.  The
# (N, 1) 2-D mesh degenerates bit-identically anyway).


def tile_spec(arr) -> P:
    """``P("lat", "lon", None, ...)`` for an array tiled on its two leading
    (lat, lon) axes — the 2-D twin of :func:`lat_spec`."""
    return P("lat", "lon", *((None,) * (arr.ndim - 2)))


def shard_state_atm_latlon_2d(
    state: CGridLatLonHydrostaticState, mesh,
) -> CGridLatLonHydrostaticState:
    """Lay out a C-grid hydrostatic atm state for the 2-D ("lat", "lon")
    tile shard_map.

    Cell leaves (``T, p_s, phis`` + every tracer) tile ``P("lat", "lon",
    ...)`` directly.  The staggered ``v`` (leading dim ``n_lat+1``) drops its
    north pole-wall face -> ``v_lower = v[:n_lat]`` exactly as the 1-D layout
    (:func:`shard_state_atm_latlon`).  The staggered ``u`` (lon dim
    ``n_lon+1``) drops its east periodic-seam column -> ``u_left =
    u[:, :n_lon]``; the dropped column is the periodic closure
    (``u[:, n_lon] == u[:, 0]`` — the identity ``interp_cell_to_uface``
    constructs and every C-grid tendency preserves) and is reconstructed
    inside the body from the east neighbour via the lon ring.
    """
    # FIRST statement: this is a DIRECT ``device_put`` of full global arrays
    # onto a cross-process ``NamedSharding``, which XLA services with an
    # ALL-GATHER (documented on ``latlon_spmd.shard_leaf``) -- so the round-3
    # allow-list reason "SCATTER, no collective" was FALSE for this entry
    # (codex round-4, blocker 6). Gate the mesh AND the leaf schedule.
    _agree_mesh_entry(mesh, state, where="shard_state_atm_latlon_2d")

def _build_geometry_stacks(model, mesh, n_dev: int, shard_geometry: bool):
    """Stack every band's ``LatLonGrid`` array fields (+ the optional
    polar-filter masks) over a leading band axis and lay them out on ``mesh``.

    ``shard_geometry=False`` (the historical layout): every stack is
    ``device_put`` REPLICATED (``P()``) — each device holds ALL bands'
    geometry and the body indexes its own band at ``axis_index``.

    ``shard_geometry=True`` (M2b): every stack is sharded ``P("lat", ...)``
    on the leading band axis — each device holds ONLY its own band's slice
    (leading extent 1 inside the shard_map body, static index ``[0]``).  The
    VALUES the body consumes are identical either way (the same band slice),
    so the step numerics are bit-unchanged; only the residency changes
    (per-device geometry bytes drop by ``n_dev`` —
    :func:`atm_latlon_geometry_bytes`).

    Returns ``(template, array_field_names, stacks, stacks_spec)``.
    """
    grid = model.grid
    band_grids = build_band_grids_atm(grid, n_dev)
    template = band_grids[0]
    array_field_names = atm_grid_array_field_names(template)
    raw = {
        name: jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
                        axis=0)
        for name in array_field_names
    }
    # Per-band polar-filter masks (only when the filter is on): slice the
    # global masks [s:e] (cell) / [s:e+1] (v-face stagger) and stack.  Absent
    # otherwise -> the body passes None -> _step_cgrid_impl resolves to
    # self._polar_mask (also None), no filter.
    nl = int(grid.n_lat) // n_dev
    if model._polar_mask is not None:
        raw["__polar_mask"] = jnp.stack(
            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(n_dev)],
            axis=0)
        raw["__polar_mask_v"] = jnp.stack(
            [model._polar_mask_v[r * nl:r * nl + nl + 1] for r in range(n_dev)],
            axis=0)
    # #1362: the band geometry above is RECOMPUTED per process from the same
    # config, and per-process XLA autotuning on device-derived grid fields
    # makes the last ULPs differ at larger sizes -- which trips the
    # bit-identical assert inside a replicated device_put (the ocean lane hit
    # exactly this at LL576 np=4; this lane hit it at LL768).  Verify
    # cross-process agreement, then broadcast process 0's bytes.  Guarded, not
    # blind: a REAL divergence (different polar masks = different filtering =
    # different physics) RAISES instead of being masked by process 0.
    #
    # The schema gate matters more here than in the ocean lane: the polar-mask
    # entries are CONDITIONAL on ``model._polar_mask``, so a per-process
    # difference in that one setting changes the field LIST itself, which
    # would desynchronize the per-field gathers rather than fail cleanly.
    ordered_names = list(raw)
    assert_schema_agrees(ordered_names, n_dev,
                         context="make_sharded_atm_latlon_step",
                         arrays=[raw[n] for n in ordered_names])
    # checked_shard_put replaces broadcast_checked + device_put (the ocean
    # walls, PR #1457): the broadcast's psum program is [n_processes,
    # stack] and a numpy device_put onto an all-process sharding pays
    # jax's whole-array assert_equal. Both are SOFT walls here (2-D
    # geometry, ~MB-scale fields) but the shared helper removes them
    # uniformly; the per-leading-slab gate keeps the divergence contract
    # (schema gate above agreed shapes, so widths match).
    spec_of = lat_spec if shard_geometry else (lambda _arr: P())
    stacks = {
        name: checked_shard_put(
            raw[name], name, NamedSharding(mesh, spec_of(raw[name])),
            context="make_sharded_atm_latlon_step",
            n_bands=int(raw[name].shape[0]))
        for name in ordered_names
    }
    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
    return template, array_field_names, stacks, stacks_spec


def _make_band_step_body(model, template, array_field_names, axis,


def _build_geometry_stacks_2d(model, mesh, p_lat: int, p_lon: int,
                              shard_geometry: bool):
    """Stack every tile's ``LatLonGrid`` array fields (+ the optional
    polar-filter masks at ``p_lon == 1``) over LEADING ``(p_lat, p_lon)``
    tile axes and lay them out on ``mesh`` — the 2-D twin of
    :func:`_build_geometry_stacks`.

    ``shard_geometry=True``: stacks are sharded ``P("lat", "lon", ...)`` on
    the tile axes — each device holds ONLY its own tile's slice (leading
    extents ``(1, 1)`` inside the body, static index ``[0, 0]``).
    ``shard_geometry=False``: replicated (``P()``) stacks, indexed at
    ``(axis_index("lat"), axis_index("lon"))``.  Same tile VALUES either way.

    Returns ``(template, array_field_names, stacks, stacks_spec)``.
    """
    grid = model.grid
    tile_grids = build_tile_grids_atm_2d(grid, p_lat, p_lon)
    template = tile_grids[0][0]
    array_field_names = atm_grid_array_field_names(template)
    raw = {
        name: jnp.stack([
            jnp.stack([jnp.asarray(getattr(tile_grids[r][c], name))
                       for c in range(p_lon)], axis=0)
            for r in range(p_lat)
        ], axis=0)
        for name in array_field_names
    }
    # Per-tile polar-filter masks: p_lon > 1 is refused by the factories
    # (the filter rfft's the full lon circle); at p_lon == 1 the stacks
    # mirror the band layout with a singleton lon-tile axis.
    nl = int(grid.n_lat) // p_lat
    if model._polar_mask is not None:
        if p_lon > 1:
            raise NotImplementedError(
                "atm 2-D SPMD tiling: use_polar_filter=True with p_lon > 1 "
                "is not wired — the polar filter FFTs the full longitude "
                "circle (needs a lon-gather FFT).  Use p_lon == 1 or "
                "disable the filter.")
        raw["__polar_mask"] = jnp.stack(
            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(p_lat)],
            axis=0)[:, None]
        raw["__polar_mask_v"] = jnp.stack(
            [model._polar_mask_v[r * nl:r * nl + nl + 1]
             for r in range(p_lat)],
            axis=0)[:, None]
    # #1362, 2-D twin of the guard in _build_geometry_stacks -- same
    # per-process recompute, same replicated-device_put bit-identity assert.
    # n_dev is the FULL device count here (p_lat * p_lon): the schema
    # fingerprint must describe this process's whole mesh, not one axis.
    ordered_names = list(raw)
    assert_schema_agrees(ordered_names, p_lat * p_lon,
                         context="make_sharded_atm_latlon_step_2d",
                         arrays=[raw[n] for n in ordered_names])
    # 2-D twin of the checked_shard_put swap above (see that note).
    spec_of = tile_spec if shard_geometry else (lambda _arr: P())
    stacks = {
        name: checked_shard_put(
            raw[name], name, NamedSharding(mesh, spec_of(raw[name])),
            context="make_sharded_atm_latlon_step_2d",
            n_bands=int(raw[name].shape[0]))
        for name in ordered_names
    }
    stacks_spec = {name: spec_of(raw[name]) for name in ordered_names}
    return template, array_field_names, stacks, stacks_spec


def _make_tile_step_body_2d(model, template, array_field_names,
                            perm_north, p_lon: int, physics_fn,
                            shard_geometry: bool):

exec
/usr/bin/bash -lc "sed -n '650,980p' packages/core/legoesm/parallel/geometry_consistency.py; git status --short; git diff --check HEAD; git log -1 --oneline --decorate; rg -n \"geometry_consistency|checked_shard_put|_build_geometry_stacks|multicontroller|multicontroller|virtual devices|XLA_FLAGS|shard_geometry=False\" tests packages -g '*.py' -g '*.md' -g '*.rst'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 7456ms:
    else:
        per_band = []
        for b in range(n_bands):
            flat = host[b].ravel()
            finite = flat[np.isfinite(flat)]
            f64 = finite.astype(np.float64)
            struct.append(float(flat.size - finite.size))
            per_band.extend([
                float(f64.sum()) if f64.size else 0.0,
                float((f64 * f64).sum()) if f64.size else 0.0,
                float(np.abs(f64).max()) if f64.size else 0.0,
            ])
        vals = np.array(per_band, dtype=np.float64)
    return np.array(struct, dtype=np.float64), vals, is_exact


def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=None):
    """True iff every process's :func:`band_fingerprint` matches process 0's."""
    if rtol is None:
        rtol = _FLOAT_RTOL
    struct_ok = bool(np.all(g_struct == g_struct[0]))
    if is_exact:
        vals_ok = bool(np.all(g_vals == g_vals[0]))
    else:
        vals_ok = bool(np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
    return struct_ok and vals_ok


def checked_shard_put(arr, name, sharding, *, context, n_bands):
    """Gate a band-stacked field per band, then put WITHOUT broadcast or
    jax's whole-array device_put assert (walls 1+2 above).

    Single-process: plain ``jax.device_put`` — byte-unchanged, no host
    round trip. Multi-process: per-band fingerprint gate (symmetric raise
    on real divergence), then ``jax.make_array_from_callback`` hands each
    process exactly its addressable slabs. Cross-process byte-identity of
    NON-owned bands is not required — owned bands are the only bytes that
    reach any device, and their drift is bounded by the gate.
    """
    if jax.process_count() <= 1:
        return jax.device_put(arr, sharding)
    from jax.experimental import multihost_utils

    host = np.asarray(arr)
    struct, vals, is_exact = band_fingerprint(host, n_bands)
    g_struct = multihost_utils.process_allgather(struct)
    g_vals = multihost_utils.process_allgather(vals)
    if not band_fingerprints_agree(g_struct, g_vals, is_exact):
        raise RuntimeError(
            f"{context}: band-stacked field {name!r} DIVERGES across "
            f"processes (exact_dtype={is_exact}, "
            f"gathered={g_vals.tolist()}) — a real config/grid "
            f"inconsistency, not autotune noise; refusing to shard it.")
    return jax.make_array_from_callback(
        host.shape, sharding, lambda idx: host[idx])


def leaf_digest48(x) -> float:
    # Dtype-AWARE 48-bit digest of one numeric leaf. content_hash48 alone
    # hashes raw bytes, so python 0 and 0.0 (both eight zero bytes on a
    # 64-bit host) collide although jnp.asarray builds distinct
    # integer/float inputs from them (codex r21). The dtype code and shape
    # are folded in so a cross-rank TYPE divergence fails the gate too.
    import hashlib

    a = np.asarray(x)
    meta = f"{np.dtype(a.dtype).num}:{a.shape}".encode()
    h = hashlib.blake2b(meta + a.tobytes(), digest_size=6)
    return float(int.from_bytes(h.digest(), "big"))


def assert_pytree_bytes_equal(tree, what):
    """Cheap multi-process replacement for the per-leaf assert_equal that
    :func:`checked_shard_put`-style puts bypass on NON-band inputs (state /
    forcing pytrees): one 48-bit digest per array leaf, one tiny allgather,
    symmetric raise on mismatch. No-op single-process.
    """
    if jax.process_count() <= 1:
        return
    from jax.experimental import multihost_utils

    # Numeric python scalars included (codex r20 item 1): the scatter
    # paths jnp.asarray + put them, so a rank-divergent scalar must not
    # bypass the gate. Non-numeric leaves (None, strings) stay excluded.
    leaves = [x for x in jax.tree_util.tree_leaves(tree)
              if hasattr(x, "ndim") or isinstance(x, (int, float, complex))]
    vals = np.array([leaf_digest48(x) for x in leaves], dtype=np.float64)
    g = multihost_utils.process_allgather(vals)
    if not bool(np.all(g == g[0])):
        bad = [i for i in range(len(leaves))
               if not bool(np.all(g[:, i] == g[0, i]))]
        raise RuntimeError(
            f"{what}: array leaves {bad} differ across processes (48-bit "
            f"byte digests disagree) — the per-process inputs are NOT "
            f"identical, which jax's device_put assert would have refused. "
            f"Fix the per-process build before sharding.")


def addressable_shard_put(arr, sharding):
    """Ungated assert-free put (walls 1+2) for inputs whose cross-process
    consistency the CALLER has already gated (state/forcing pytrees via
    :func:`assert_pytree_bytes_equal`). Single-process: plain device_put."""
    if jax.process_count() <= 1:
        return jax.device_put(arr, sharding)
    host = np.asarray(arr)
    return jax.make_array_from_callback(
        host.shape, sharding, lambda idx: host[idx])
 M packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
?? .physics-validator/scaling_campaign/codex_atm_port_r22_2026-08-03.md
714508e6b (HEAD -> worktree-scaling-campaign, origin/worktree-scaling-campaign) fix(parallel): dtype-aware leaf_digest48 — int-0 vs float-0.0 no longer collide in the scalar gate (codex r21)
tests/bench/test_scaling_metadata.py:264:        "XLA_FLAGS", "--xla_force_host_platform_device_count=8")
tests/bench/test_scaling_metadata.py:270:        "XLA_FLAGS", "--xla_force_host_platform_device_count=1")
tests/bench/test_scaling_metadata.py:272:    monkeypatch.delenv("XLA_FLAGS")
tests/bench/test_scaling_metadata.py:416:        # config-header echo of XLA_FLAGS -> must NOT match (no op-call paren)
packages/core/legoesm/parallel/early_init.py:140:    """Per-process ``local_device_ids`` for a multicontroller launch.
packages/core/legoesm/parallel/early_init.py:330:def init_multicontroller_distributed(coordinator: str | None = None) -> None:
packages/core/legoesm/parallel/early_init.py:331:    """Initialize ``jax.distributed`` for a route-B multicontroller launch.
packages/core/legoesm/parallel/early_init.py:333:    Shared by every ``--multicontroller`` entry point (the ocean/atm SPMD
packages/core/legoesm/parallel/geometry_consistency.py:52:    "checked_shard_put",
packages/core/legoesm/parallel/geometry_consistency.py:436:    crash beats a silent skip.  ``tests/unit/test_geometry_consistency_trace_
packages/core/legoesm/parallel/geometry_consistency.py:608:# Three stacked multicontroller walls were found on the ocean lane (codex
packages/core/legoesm/parallel/geometry_consistency.py:678:def checked_shard_put(arr, name, sharding, *, context, n_bands):
packages/core/legoesm/parallel/geometry_consistency.py:723:    :func:`checked_shard_put`-style puts bypass on NON-band inputs (state /
packages/core/legoesm/parallel/device_config.py:167:    and prime ``XLA_FLAGS`` here, before the device query.  The
packages/core/legoesm/parallel/device_config.py:178:    # mutating XLA_FLAGS is silently ineffective on most JAX versions.
packages/core/legoesm/parallel/device_config.py:182:            NVIDIA_GPU_XLA_FLAGS,
packages/core/legoesm/parallel/device_config.py:183:            AMD_GPU_XLA_FLAGS,
packages/core/legoesm/parallel/device_config.py:184:            TPU_XLA_FLAGS,
packages/core/legoesm/parallel/device_config.py:190:            set_xla_flags(TPU_XLA_FLAGS)
packages/core/legoesm/parallel/device_config.py:194:                set_xla_flags(NVIDIA_GPU_XLA_FLAGS)
packages/core/legoesm/parallel/device_config.py:196:                set_xla_flags(AMD_GPU_XLA_FLAGS)
packages/core/legoesm/parallel/device_config.py:271:    from legoesm.runtime.backend import set_xla_flags, TPU_XLA_FLAGS
packages/core/legoesm/parallel/device_config.py:273:    set_xla_flags(TPU_XLA_FLAGS)
packages/core/legoesm/parallel/device_config.py:295:    runs, mutating ``XLA_FLAGS`` is silently ineffective on most JAX
packages/core/legoesm/parallel/device_config.py:301:        set_xla_flags, NVIDIA_GPU_XLA_FLAGS, AMD_GPU_XLA_FLAGS,
packages/core/legoesm/parallel/device_config.py:305:    # XLA_FLAGS on time.
packages/core/legoesm/parallel/device_config.py:308:        set_xla_flags(NVIDIA_GPU_XLA_FLAGS)
packages/core/legoesm/parallel/device_config.py:310:        set_xla_flags(AMD_GPU_XLA_FLAGS)
packages/core/legoesm/parallel/device_config.py:325:            set_xla_flags(NVIDIA_GPU_XLA_FLAGS)
packages/core/legoesm/parallel/device_config.py:327:            set_xla_flags(AMD_GPU_XLA_FLAGS)
packages/core/legoesm/parallel/device_config.py:368:    if "XLA_FLAGS" not in os.environ:
packages/core/legoesm/parallel/device_config.py:373:            # use with a fatal `Unknown flag in XLA_FLAGS` error from
packages/core/legoesm/parallel/scaling_diagnostics.py:322:    info.xla_flags = os.environ.get("XLA_FLAGS", "")
packages/coupler/legoesm/driver/model_driver.py:8999:        Under route-B multicontroller (``jax.process_count() > 1``, after
packages/coupler/legoesm/driver/model_driver.py:9000:        ``init_multicontroller_distributed``) ``jax.devices()`` is the GLOBAL
packages/coupler/legoesm/driver/model_driver.py:9011:                f"--multicontroller (route-B) uses ALL {len(devs)} global "
packages/coupler/legoesm/driver/model_driver.py:9283:                "operator-split tiled cube: multicontroller (route-B "
packages/coupler/legoesm/driver/model_driver.py:9794:        # Route-B multicontroller (jax.distributed cross-process NCCL): the mesh
packages/coupler/legoesm/driver/model_driver.py:9929:            _lane = "route-B multicontroller" if _mp else "single-controller"
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:48:from legoesm.parallel.geometry_consistency import (
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:50:    assert_pytree_bytes_equal, assert_schema_agrees, checked_shard_put,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:732:        # checked_shard_put replaces the broadcast_checked+device_put pair:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:738:        # implementation: legoesm.parallel.geometry_consistency.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:739:        return checked_shard_put(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:1005:        # multicontroller, the nd-linear wall this module removes).
packages/coupler/legoesm/driver/sharded_operator_split_step.py:48:from legoesm.parallel.geometry_consistency import (
tests/bench/test_ppermute_microbench.py:3:Runs on CPU virtual devices — the FIT logic and the refusals are what these
packages/core/legoesm/runtime/backend.py:11:  ``XLA_FLAGS``, etc.) can be set *before* JAX initialises.
packages/core/legoesm/runtime/backend.py:100:TPU_XLA_FLAGS = {
packages/core/legoesm/runtime/backend.py:106:NVIDIA_GPU_XLA_FLAGS = {
packages/core/legoesm/runtime/backend.py:113:    # ``LOG(FATAL) Unknown flags in XLA_FLAGS`` the instant *any* GPU
packages/core/legoesm/runtime/backend.py:131:AMD_GPU_XLA_FLAGS: dict[str, str] = {
packages/core/legoesm/runtime/backend.py:137:    """Append XLA flags to ``XLA_FLAGS``, without duplicating existing keys."""
packages/core/legoesm/runtime/backend.py:138:    existing = os.environ.get("XLA_FLAGS", "")
packages/core/legoesm/runtime/backend.py:146:        os.environ["XLA_FLAGS"] = combined
packages/core/legoesm/runtime/backend.py:153:    further mutations of ``os.environ['XLA_FLAGS']`` no longer take
packages/core/legoesm/runtime/backend.py:155:    (``NVIDIA_GPU_XLA_FLAGS``, ``AMD_GPU_XLA_FLAGS``) are critical for
packages/core/legoesm/runtime/backend.py:161:    will at least pick the right matmul precision even if ``XLA_FLAGS``
packages/core/legoesm/runtime/backend.py:291:    set, otherwise ``XLA_FLAGS`` mutations no-op.  Returns one of
packages/core/legoesm/runtime/backend.py:338:        set_xla_flags(TPU_XLA_FLAGS)
packages/core/legoesm/runtime/backend.py:351:        # ``os.environ['XLA_FLAGS']`` is silently ineffective on most
packages/core/legoesm/runtime/backend.py:375:            set_xla_flags(NVIDIA_GPU_XLA_FLAGS)
packages/core/legoesm/runtime/backend.py:377:            set_xla_flags(AMD_GPU_XLA_FLAGS)
packages/core/legoesm/runtime/backend.py:382:            # Vendor was only known after JAX init; XLA_FLAGS already
packages/core/legoesm/runtime/backend.py:387:                set_xla_flags(NVIDIA_GPU_XLA_FLAGS)
packages/core/legoesm/runtime/backend.py:395:                set_xla_flags(AMD_GPU_XLA_FLAGS)
packages/core/legoesm/runtime/backend.py:419:        if "XLA_FLAGS" not in os.environ:
packages/core/legoesm/runtime/backend.py:425:                #     Unknown flag in XLA_FLAGS: --intra_op_parallelism_threads=N
tests/ml/test_data_parallel.py:5:        XLA_FLAGS=--xla_force_host_platform_device_count=2 <py> -m pytest \
tests/bench/test_bench_cube_shardmap_halo.py:111:    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
tests/bench/test_bench_cube_shardmap_halo.py:155:    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
tests/test_atmosphere_cross_grid_plots.py:43:                "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE"):
tests/unit/test_run_omip_latlon_spmd.py:49:        f"worker expected 2 virtual devices, got {jax.device_count()}")
tests/unit/test_run_omip_latlon_spmd.py:119:    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
tests/unit/test_run_omip_latlon_spmd.py:171:        f"worker expected 2 virtual devices, got {jax.device_count()}")
tests/unit/test_run_omip_latlon_spmd.py:245:    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
tests/bench/test_bench_mpas_spmd_gates.py:10:  new leaf symbol) over 2 virtual devices;
tests/bench/test_bench_mpas_spmd_gates.py:17:tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py.
tests/bench/test_bench_mpas_spmd_gates.py:58:                 "--multicontroller", "--coordinator", "--partition-method",
tests/bench/test_bench_mpas_spmd_gates.py:92:                    "(set XLA_FLAGS=--xla_force_host_platform_device_count=2)")
tests/bench/test_bench_mpas_spmd_gates.py:134:    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
tests/bench/test_bench_mpas_spmd_gates.py:169:    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
tests/bench/test_bench_mpas_spmd_gates.py:193:    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
tests/bench/test_bench_ocean_latlon_spmd_gates.py:15:tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py (and the
tests/bench/test_bench_ocean_latlon_spmd_gates.py:16:launcher-gated test_latlon_ocean_spmd_multicontroller.py).
tests/bench/test_bench_ocean_latlon_spmd_gates.py:54:                 "--multicontroller", "--coordinator"):
tests/bench/test_bench_ocean_latlon_spmd_gates.py:67:    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
tests/bench/test_bench_ocean_latlon_spmd_gates.py:137:    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:35:argument (SegmentForcing doctrine) before it can be scanned; multicontroller
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:35:from legoesm.parallel.geometry_consistency import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:36:    FLAG_ABSENT, assert_flags_agree, assert_schema_agrees, checked_shard_put,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:435:def _build_geometry_stacks(model, mesh, n_dev: int, shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:439:    ``shard_geometry=False`` (the historical layout): every stack is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:491:    # checked_shard_put replaces broadcast_checked + device_put (the ocean
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:500:        name: checked_shard_put(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:711:        # and `_build_geometry_stacks` broadcasts; agreeing their dtypes and
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:722:        # Rank-local CACHE presence. `_build_geometry_stacks` conditions on
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:944:    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1128:    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1537:def _build_geometry_stacks_2d(model, mesh, p_lat: int, p_lon: int,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1542:    :func:`_build_geometry_stacks`.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1547:    ``shard_geometry=False``: replicated (``P()``) stacks, indexed at
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1582:    # #1362, 2-D twin of the guard in _build_geometry_stacks -- same
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1590:    # 2-D twin of the checked_shard_put swap above (see that note).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1593:        name: checked_shard_put(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1743:        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1825:        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
tests/test_no_private_cross_imports.py:12:``get_sendrecv_vjp``, ``_TPU_XLA_FLAGS`` → ``TPU_XLA_FLAGS``, Wright-EOS
tests/distributed/test_geometry_consistency_mp.py:3:Run:  srun -n 2 python -m pytest tests/distributed/test_geometry_consistency_mp.py
tests/distributed/test_geometry_consistency_mp.py:7:``legoesm.parallel.geometry_consistency`` early-returns at
tests/distributed/test_geometry_consistency_mp.py:15:The governing invariant (see ``geometry_consistency`` module docstring):
tests/distributed/test_geometry_consistency_mp.py:67:    from legoesm.parallel.geometry_consistency import (
tests/atmosphere/hydrostatic/unit/test_radiation.py:1668:    ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` to
tests/atmosphere/hydrostatic/unit/test_radiation.py:1734:                "single-device host — set XLA_FLAGS to emulate 4 devices"
tests/atmosphere/hydrostatic/unit/test_radiation.py:1812:        ``XLA_FLAGS=--xla_force_host_platform_device_count=4``)."""
tests/atmosphere/hydrostatic/unit/test_radiation.py:1815:                "single-device host — set XLA_FLAGS to emulate 4 devices"
tests/unit/test_physics_grid_adapters.py:376:    visible to JAX (use ``XLA_FLAGS=--xla_force_host_platform_device_count=4``
tests/unit/test_physics_grid_adapters.py:416:                "single-device host — set XLA_FLAGS to emulate"
tests/unit/test_physics_grid_adapters.py:501:                "single-device host — set XLA_FLAGS to emulate"
tests/parallel/test_atm_latlon_segment.py:31:Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count>=2``).
tests/parallel/test_tiled_fix_ps_mass.py:12:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py:14:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.  sigma-only (Kessler).
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:19:(``XLA_FLAGS=--xla_force_host_platform_device_count=12`` -> 24 global).  It
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:27:NCCL-specific concurrent comm-init race the ``XLA_FLAGS`` serialize-flags fix
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:74:    repo's existing ``*_multicontroller_selfspawn.py`` gates hit this too).
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:112:    # 12 local virtual devices per process x 2 processes = 24 global.
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:116:              f"XLA_FLAGS=--xla_force_host_platform_device_count", flush=True)
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:201:    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={DEVICES_PER_PROC}"
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:211:            "-x", "XLA_FLAGS",
tests/parallel/test_latlon_state_sharding.py:7:Needs >1 host device: XLA_FLAGS=--xla_force_host_platform_device_count=4.
tests/parallel/test_tiled_operator_split_step.py:17:Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
tests/parallel/test_tiled_operator_split_step.py:24:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")
tests/parallel/test_tiled_operator_split_step.py:68:                    f"(XLA_FLAGS=--xla_force_host_platform_device_count=24)")
tests/parallel/test_latlon_ocean_spmd_step.py:39:Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=4 \
tests/parallel/test_latlon_ocean_spmd_step.py:46:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
tests/parallel/test_latlon_ocean_spmd_step.py:94:                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py:204:                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py:272:                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py:316:                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py:379:                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py:408:                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_duogrid_spmd_parity.py:20:    XLA_FLAGS=--xla_force_host_platform_device_count=6 JAX_ENABLE_X64=1 \
tests/parallel/test_duogrid_spmd_parity.py:34:                    "(set XLA_FLAGS=--xla_force_host_platform_device_count=6)")
tests/parallel/test_tiled_fv3_hydrostatic_step.py:19:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
tests/parallel/test_latlon_spmd_fused_halo.py:24:Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=4 \
tests/parallel/test_latlon_spmd_fused_halo.py:31:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
tests/parallel/test_latlon_spmd_fused_halo.py:48:    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_tiled_cgrid_gradient.py:10:``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
tests/parallel/test_tiled_3d_velocity.py:10:54 host CPU devices: ``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
tests/parallel/test_ppermute_halo_exchange.py:12:    XLA_FLAGS="--xla_force_host_platform_device_count=6" \\
tests/parallel/test_ppermute_halo_exchange.py:20:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=6")
tests/parallel/test_ppermute_halo_exchange.py:31:        "needs 6 devices (XLA_FLAGS=--xla_force_host_platform_device_count=6)",
tests/parallel/test_tiled_cube_spmd_driver.py:12:Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
tests/parallel/test_tiled_cube_spmd_driver.py:18:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")
tests/parallel/test_tiled_cube_spmd_driver.py:91:                    f"(XLA_FLAGS=--xla_force_host_platform_device_count=24)")
tests/parallel/test_persistent_sharded_ocean_loop.py:40:Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1
tests/parallel/test_persistent_sharded_ocean_loop.py:50:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
tests/parallel/test_persistent_sharded_ocean_loop.py:86:                    reason="needs >=2 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_spmd_northfold.py:15:(``XLA_FLAGS=--xla_force_host_platform_device_count=4``); the production target
tests/parallel/test_tiled_fv3_sw_full.py:14:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
tests/unit/test_column_shard.py:236:            XLA_FLAGS="--xla_force_host_platform_device_count=4"
tests/unit/test_column_shard.py:241:                "single-device host — set XLA_FLAGS to emulate 4 devices"
tests/unit/test_run_omip_cli.py:73:def test_multicontroller_flags_round_trip():
tests/unit/test_run_omip_cli.py:74:    """--multicontroller / --coordinator parse and reach OMIPRunConfig
tests/unit/test_run_omip_cli.py:77:    assert args.multicontroller is False
tests/unit/test_run_omip_cli.py:80:    assert cfg.multicontroller is False
tests/unit/test_run_omip_cli.py:84:        "--grid", "latlon", "--enable-latlon-spmd", "--multicontroller",
tests/unit/test_run_omip_cli.py:87:    assert cfg.multicontroller is True
tests/unit/test_run_omip_cli.py:91:def test_multicontroller_without_spmd_refused():
tests/unit/test_run_omip_cli.py:92:    """--multicontroller alone (no --enable-latlon-spmd) must hard-fail BEFORE
tests/unit/test_run_omip_cli.py:98:    args = parse_args(["--grid", "latlon", "--multicontroller"])
tests/parallel/test_atm_latlon_2d_tiling.py:50:Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
tests/parallel/test_atm_latlon_2d_tiling.py:337:        _build_geometry_stacks_2d, atm_grid_array_field_names)
tests/parallel/test_atm_latlon_2d_tiling.py:353:    template, afn, stacks, stacks_spec = _build_geometry_stacks_2d(
tests/parallel/test_mpas_partitionlocal_build.py:25:(``XLA_FLAGS=--xla_force_host_platform_device_count=4``, set by the runner —
tests/parallel/test_mpas_partitionlocal_build.py:45:        pytest.skip(f"needs {N_DEV} virtual devices "
tests/parallel/test_mpas_partitionlocal_build.py:46:                    f"(XLA_FLAGS did not take effect)")
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:16:Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=4 \
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:23:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:39:    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_atm_latlon_spmd_step.py:32:Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``);
tests/parallel/test_early_init_hardening.py:5:covered by the multicontroller self-spawn suites.
tests/parallel/test_operator_split_real_physics_column_local.py:17:Host CPU (``XLA_FLAGS=--xla_force_host_platform_device_count>=1``); needs no
tests/parallel/test_tiled_pad_body.py:15:``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
tests/parallel/test_tiled_pad_body.py:50:                    "(XLA_FLAGS=--xla_force_host_platform_device_count=24)")
tests/parallel/test_tiled_mass_divergence.py:21:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
tests/parallel/test_sharded_step_sharding_tripwire.py:29:    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1 \\
tests/parallel/test_sharded_step_sharding_tripwire.py:43:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
tests/parallel/test_sharded_step_sharding_tripwire.py:61:            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
tests/parallel/test_cubesphere_exchange.py:19:                     "(set XLA_FLAGS=--xla_force_host_platform_device_count=6)")
tests/parallel/test_cubesphere_exchange.py:514:    ``XLA_FLAGS=--xla_force_host_platform_device_count=6``) for the
tests/parallel/test_cubesphere_exchange.py:576:            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})")
tests/parallel/test_ppermute_multiface.py:26:    XLA_FLAGS="--xla_force_host_platform_device_count=6" \\
tests/parallel/test_ppermute_multiface.py:34:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=6")
tests/parallel/test_ppermute_multiface.py:45:        "needs >=2 devices (XLA_FLAGS=--xla_force_host_platform_device_count=6)",
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:1:"""SELF-SPAWNING route-B multicontroller gate for the ``run_omip`` DRIVER.
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:4:(``test_latlon_ocean_spmd_multicontroller_selfspawn.py``, which drives
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:7:(``scripts/run/run_omip.py --enable-latlon-spmd --multicontroller``)
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:11:* the early ``init_multicontroller_distributed`` bootstrap firing BEFORE any
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:52:def test_run_omip_two_process_selfspawn_multicontroller(tmp_path):
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:68:    base_env.pop("XLA_FLAGS", None)  # 1 real CPU device per process
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:88:            "--enable-latlon-spmd", "--multicontroller",
tests/parallel/test_latlon_spmd_shard_leaf.py:9:    XLA_FLAGS=--xla_force_host_platform_device_count=4
tests/parallel/test_latlon_spmd_shard_leaf.py:11:``*_multicontroller_selfspawn`` tests (which spawn real processes); here
tests/parallel/test_latlon_spmd_shard_leaf.py:26:        pytest.skip("needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
tests/parallel/test_latlon_spmd_halo.py:7:(``XLA_FLAGS=--xla_force_host_platform_device_count=4``); the production target
tests/parallel/test_latlon_vface_reconstruct.py:9:CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
tests/parallel/test_tiled_fv3_hydrostatic_momentum.py:32:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.  kt=3 (nl=8 != 12) pins
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:3:The ocean twin of ``test_atm_latlon_spmd_multicontroller.py``: N processes
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:20:  XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1 \
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:22:      tests/parallel/test_latlon_ocean_spmd_multicontroller.py -x -q
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:97:def test_multicontroller_ocean_step_matches_serial():
tests/parallel/test_tiled_center_to_dgrid_vector.py:16:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.  kt=3 (nl=8 != 12) pins
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py:17:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
tests/parallel/test_shard_forcing.py:16:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=6")
tests/parallel/test_shard_forcing.py:85:    assert dc.mesh is not None, "needs 6 virtual devices (XLA_FLAGS)"
tests/parallel/test_mpas_atm_native_step.py:17:  non-default integrator), 2 virtual devices vs single device;
tests/parallel/test_mpas_atm_native_step.py:26:Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=2`` (or
tests/parallel/test_mpas_atm_native_step.py:28:``test_mpas_spmd_multicontroller_selfspawn.py``; the route-A exchange
tests/parallel/test_mpas_atm_native_step.py:63:            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
tests/ocean/unit/test_sharded_geom_fingerprint.py:7:multicontroller allgather wiring is exercised by the distributed suite).
tests/ocean/unit/test_sharded_geom_fingerprint.py:12:from legoesm.parallel.geometry_consistency import (
tests/ocean/unit/test_sharded_geom_fingerprint.py:110:    from legoesm.parallel.geometry_consistency import leaf_digest48
tests/parallel/test_atm_latlon_spmd_driver.py:7:``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:3:Companion to tests/parallel/test_latlon_ocean_spmd_multicontroller.py (the
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:7:production bench (``bench_ocean_latlon_spmd_scaling.py --multicontroller``)
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:43:    base_env.pop("XLA_FLAGS", None)  # 1 real CPU device per process
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:48:            "--multicontroller", "--coordinator", f"localhost:{port}",
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:74:    # Rank 0 owns the JSONL; the record carries the multicontroller fields.
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:80:    assert rec["multicontroller"] is True
tests/parallel/test_tiled_fv3_hydrostatic_thermo.py:32:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.  kt=3 (nl=8 != 12) pins
tests/parallel/test_tiled_fv3_sw_momentum.py:22:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.  The kt=3 lane pins
tests/parallel/test_tiled_vector_pad.py:11:``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
tests/parallel/test_atm_latlon_state_layout.py:8:(``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
tests/parallel/test_voronoi_sharded_equivalence.py:13:``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
tests/parallel/test_voronoi_sharded_equivalence.py:27:            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
tests/parallel/test_tiled_vadv_omega.py:11:24 host CPU devices: ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
tests/parallel/test_tiled_vertical_pe_stage.py:10:24 host CPU devices: ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
tests/parallel/test_cubed_sphere_spmd_step.py:17:``XLA_FLAGS=--xla_force_host_platform_device_count=6``.
tests/parallel/test_cubed_sphere_spmd_step.py:31:            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
tests/parallel/test_cube_tile_native_segment.py:32:Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
tests/parallel/test_cube_tile_native_segment.py:38:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")
tests/parallel/test_segment_sharding_device_config.py:51:    XLA_FLAGS=--xla_force_host_platform_device_count=2 \\
tests/parallel/test_segment_sharding_device_config.py:67:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
tests/parallel/test_segment_sharding_device_config.py:81:            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
tests/parallel/test_tiled_geopotential.py:8:54 host CPU devices: ``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
tests/parallel/test_atm_latlon_bandlocal_build.py:9:Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``);
tests/parallel/test_tiled_cc_step_adapter.py:7:must match it to RK3-reorder fp tolerance on 24 virtual devices.
tests/parallel/test_tiled_cc_step_adapter.py:16:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")
tests/parallel/test_tiled_cc_step_adapter.py:47:        pytest.skip(f"needs {6*KT*KT} devices (XLA_FLAGS host device count)")
tests/parallel/test_warmup_tiled_cube_comms.py:17:Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
tests/parallel/test_warmup_tiled_cube_comms.py:23:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")
tests/parallel/test_warmup_tiled_cube_comms.py:45:                    f"(XLA_FLAGS=--xla_force_host_platform_device_count).")
tests/unit/test_device_config.py:30:    TPU_XLA_FLAGS,
tests/unit/test_device_config.py:31:    NVIDIA_GPU_XLA_FLAGS,
tests/unit/test_device_config.py:32:    AMD_GPU_XLA_FLAGS,
tests/unit/test_device_config.py:258:        """set_xla_flags appends flags to XLA_FLAGS."""
tests/unit/test_device_config.py:259:        old = os.environ.get("XLA_FLAGS", "")
tests/unit/test_device_config.py:261:            os.environ["XLA_FLAGS"] = ""
tests/unit/test_device_config.py:263:            assert "test_flag_abc=true" in os.environ["XLA_FLAGS"]
tests/unit/test_device_config.py:266:                os.environ["XLA_FLAGS"] = old
tests/unit/test_device_config.py:268:                os.environ.pop("XLA_FLAGS", None)
tests/unit/test_device_config.py:272:        old = os.environ.get("XLA_FLAGS", "")
tests/unit/test_device_config.py:274:            os.environ["XLA_FLAGS"] = "--test_flag_xyz=false"
tests/unit/test_device_config.py:277:            assert os.environ["XLA_FLAGS"].count("test_flag_xyz") == 1
tests/unit/test_device_config.py:280:                os.environ["XLA_FLAGS"] = old
tests/unit/test_device_config.py:282:                os.environ.pop("XLA_FLAGS", None)
tests/unit/test_device_config.py:286:        old = os.environ.get("XLA_FLAGS", "")
tests/unit/test_device_config.py:288:            os.environ["XLA_FLAGS"] = "--existing_flag=1"
tests/unit/test_device_config.py:290:            flags = os.environ["XLA_FLAGS"]
tests/unit/test_device_config.py:295:                os.environ["XLA_FLAGS"] = old
tests/unit/test_device_config.py:297:                os.environ.pop("XLA_FLAGS", None)
tests/unit/test_device_config.py:306:        `intra_op_parallelism_threads=N` XLA_FLAGS entry is not
tests/unit/test_device_config.py:310:              Unknown flag in XLA_FLAGS:
tests/unit/test_device_config.py:314:        this flag unconditionally (when XLA_FLAGS was unset), making
tests/unit/test_device_config.py:695:        assert "xla_gpu_cudnn_gemm_fusion_level" in NVIDIA_GPU_XLA_FLAGS
tests/unit/test_device_config.py:696:        assert "xla_gpu_cudnn_gemm_fusion_level" not in AMD_GPU_XLA_FLAGS
tests/parallel/test_latlon_ocean_spmd_tripole.py:19:Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=4 \
tests/parallel/test_latlon_ocean_spmd_tripole.py:26:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
tests/parallel/test_latlon_ocean_spmd_tripole.py:97:                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_operator_split_spmd_driver_parity.py:18:4 host CPU devices via ``XLA_FLAGS=--xla_force_host_platform_device_count=4``;
tests/parallel/test_latlon_ocean_spmd_multiprocess.py:79:            "XLA_FLAGS": "--xla_force_host_platform_device_count=4"}
tests/parallel/test_latlon_ocean_spmd_multiprocess.py:89:            "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
tests/parallel/test_atm_latlon_spmd_multicontroller.py:23:  XLA_FLAGS=--xla_force_host_platform_device_count=2 \
tests/parallel/test_atm_latlon_spmd_multicontroller.py:25:      tests/parallel/test_atm_latlon_spmd_multicontroller.py -x -q
tests/parallel/test_atm_latlon_spmd_multicontroller.py:97:def test_multicontroller_step_matches_serial():
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:1:"""SELF-SPAWNING route-B multicontroller gate for the ``run_amip`` DRIVER.
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:4:(``test_run_omip_latlon_spmd_multicontroller_selfspawn.py``): THIS variant
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:6:(``scripts/run/run_amip.py --enable-latlon-spmd --multicontroller``) end-to-end
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:10:* the ``main()`` ``init_multicontroller_distributed`` bootstrap firing AFTER
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:12:  correctly SKIPPED for ``--multicontroller`` (it would try to load libmpi
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:46:def test_run_amip_two_process_selfspawn_multicontroller(tmp_path):
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:53:    base_env.pop("XLA_FLAGS", None)          # 1 real CPU device per process
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:78:            "--enable-latlon-spmd", "--multicontroller",
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:97:    assert "route-B multicontroller" in root, (
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:98:        f"no route-B banner on rank 0 — the multicontroller lane did not "
tests/parallel/test_tiled_mass_flux.py:10:24 host CPU devices: ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
tests/parallel/test_atm_latlon_operator_split_spmd.py:19:(``XLA_FLAGS=--xla_force_host_platform_device_count=4``), x64.
tests/parallel/test_tiled_zero_mean_tendency.py:13:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:4:tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py: spawns
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:6:(``bench_mpas_spmd_scaling.py --multicontroller``) end-to-end with the
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:45:    base_env.pop("XLA_FLAGS", None)  # 1 real CPU device per process
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:50:            "--multicontroller", "--coordinator", f"localhost:{port}",
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:72:    # Rank 0 owns the JSONL; the record carries the multicontroller fields.
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:78:    assert rec["multicontroller"] is True
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:100:    base_env.pop("XLA_FLAGS", None)  # 1 real CPU device per process
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:105:            "--multicontroller", "--coordinator", f"localhost:{port}",
tests/unit/test_multi_gpu.py:1:"""Multi-GPU scaling tests using JAX virtual devices.
tests/unit/test_multi_gpu.py:30:    Sets ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` so that
tests/unit/test_multi_gpu.py:39:    existing_flags = env.get("XLA_FLAGS", "")
tests/unit/test_multi_gpu.py:40:    env["XLA_FLAGS"] = (
tests/unit/test_multi_gpu.py:81:# Test 1: Device mesh creation with 6 virtual devices
tests/unit/test_sharded_dynamics.py:538:           "(XLA_FLAGS=--xla_force_host_platform_device_count=6)",
tests/unit/test_early_init.py:258:# init_multicontroller_distributed: the shared --multicontroller entry point
tests/unit/test_early_init.py:262:def test_multicontroller_coordinator_uses_launcher_env(monkeypatch):
tests/unit/test_early_init.py:270:    early_init.init_multicontroller_distributed("host:5000")
tests/unit/test_early_init.py:281:def test_multicontroller_coordinator_missing_env_raises(monkeypatch):
tests/unit/test_early_init.py:293:        early_init.init_multicontroller_distributed("host:5000")
tests/unit/test_early_init.py:297:def test_multicontroller_no_coordinator_delegates_to_fallback(monkeypatch):
tests/unit/test_early_init.py:303:    early_init.init_multicontroller_distributed(None)
tests/unit/test_driver_spmd_halo_lifecycle.py:42:    existing_flags = env.get("XLA_FLAGS", "")
tests/unit/test_driver_spmd_halo_lifecycle.py:43:    env["XLA_FLAGS"] = (
tests/unit/test_driver_spmd_halo_lifecycle.py:212:        # in subprocess environments with only 6 virtual devices.
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:3:Companion to tests/parallel/test_atm_latlon_spmd_multicontroller.py (the
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:134:    env.pop("XLA_FLAGS", None)  # 1 real CPU device per process, no virtuals
tests/parallel/test_tiled_fv3_hydrostatic_tendencies.py:29:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
tests/parallel/test_spectral_level_shard.py:13:``XLA_FLAGS=--xla_force_host_platform_device_count=2`` to exercise
tests/parallel/test_spectral_level_shard.py:28:            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
tests/parallel/test_tiled_blocked_loop.py:16:Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
tests/parallel/test_tiled_blocked_loop.py:22:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")
tests/parallel/test_tiled_blocked_loop.py:62:                    f"(XLA_FLAGS=--xla_force_host_platform_device_count=24)")
tests/parallel/test_tiled_blocked_loop.py:165:    # virtual devices).
tests/parallel/test_tiled_dp_s_dt.py:17:``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
tests/parallel/test_operator_split_tiled_cube_driver_parity.py:18:``XLA_FLAGS=--xla_force_host_platform_device_count=24``; x64.
tests/parallel/test_operator_split_tiled_cube_driver_parity.py:24:os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")
tests/parallel/test_conservation_spmd_lat_reduction.py:10:(``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
tests/unit/test_geometry_consistency.py:1:"""Direct tests for ``legoesm.parallel.geometry_consistency`` (#1362).
tests/unit/test_geometry_consistency.py:17:from legoesm.parallel.geometry_consistency import (
tests/unit/test_run_amip_cli.py:2224:def test_multicontroller_coordinator_flags_parse():
tests/unit/test_run_amip_cli.py:2228:    a = parser.parse_args(["--enable-latlon-spmd", "--multicontroller",
tests/unit/test_run_amip_cli.py:2230:    assert a.multicontroller is True
tests/unit/test_run_amip_cli.py:2234:    assert d.multicontroller is False
tests/unit/test_run_amip_cli.py:2238:def test_multicontroller_requires_enable_latlon_spmd(capsys):
tests/unit/test_run_amip_cli.py:2239:    """--multicontroller without --enable-latlon-spmd is refused in main()
tests/unit/test_run_amip_cli.py:2244:              "--multicontroller"])
tests/unit/test_runtime_bootstrap.py:578:            XLA_FLAGS="--xla_force_host_platform_device_count=4" \\
tests/unit/test_geometry_consistency_trace_gate.py:20:from legoesm.parallel import geometry_consistency as gc
tests/unit/test_bench_ocean_latlon_spmd_scaling.py:6:tests/parallel/test_latlon_ocean_spmd_step.py; the --multicontroller path is
tests/unit/test_bench_ocean_latlon_spmd_scaling.py:56:    assert rec["multicontroller"] is False

exec
/usr/bin/bash -lc "sed -n '1,260p' tests/unit/test_geometry_consistency.py; sed -n '1,180p' tests/distributed/test_geometry_consistency_mp.py; sed -n '1,240p' tests/parallel/test_atm_latlon_spmd_multicontroller.py; sed -n '1,240p' tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py; rg -n \"shard_geometry=False|replicated|_build_geometry_stacks_2d|_build_geometry_stacks\\(\" tests/parallel/test_atm* tests/atmosphere -g '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 241ms:
"""Direct tests for ``legoesm.parallel.geometry_consistency`` (#1362).

These are the SHARED cross-process geometry-agreement primitives used by both
the ocean and atmosphere lat-lon SPMD lanes.  The collective paths
(``process_allgather`` / ``broadcast_one_to_all``) need a real multi-process
launch and are exercised by the distributed suite; what is tested here is the
part that decides WHETHER those collectives raise — the fingerprints.

Every test below is written so it FAILS if the property it names is lost:
the permutation tests in particular are the reason the byte digest exists at
all (a moment fingerprint cannot see a permutation).
"""

import numpy as np
import pytest

from legoesm.parallel.geometry_consistency import (
    FLAG_ABSENT,
    FLAG_NEGATIVE,
    FLAG_OUT_OF_RANGE,
    FLAG_UNCOERCIBLE,
    assert_flags_agree,
    assert_schema_agrees,
    broadcast_checked,
    coerce_bool,
    coerce_count,
    config_digest48,
    content_hash48,
    name_digest48,
    safe_repr,
    schema_fingerprint,
    tree_schema_digest48,
)


class TestContentHash48:
    def test_deterministic(self):
        a = np.arange(24, dtype=np.int32).reshape(4, 6)
        assert content_hash48(a) == content_hash48(a.copy())

    def test_exactly_representable_in_float64(self):
        """48 bits keeps the digest under 2**53, so the float64
        ``process_allgather`` payload carries it EXACTLY.  A wider digest
        would round in transit and two agreeing processes could compare
        unequal (or worse, two differing ones could compare equal)."""
        for seed in range(32):
            rng = np.random.default_rng(seed)
            h = content_hash48(rng.integers(0, 255, size=64).astype(np.uint8))
            assert 0 <= h < 2.0 ** 48
            assert h == float(int(h))            # integral
            assert int(h) == int(np.float64(h))  # survives a f64 round-trip

    def test_detects_permutation_that_moments_cannot(self):
        """THE reason this is a byte digest and not a moment fingerprint.

        A boolean mask and any permutation of it share sum, sum-of-squares
        and absmax exactly — so a moment-only compare is blind to two
        processes disagreeing about WHICH cells are wet, which is a physics
        difference, not autotune noise.
        """
        a = np.zeros(64, dtype=bool)
        a[:8] = True
        b = np.zeros(64, dtype=bool)
        b[-8:] = True  # same true-count, different positions

        f = lambda x: (x.sum(), (x * x).sum(), np.abs(x).max())
        assert f(a) == f(b), "fixture broken: moments must be identical here"
        assert content_hash48(a) != content_hash48(b)

    def test_detects_two_cell_flip_that_cancels_in_the_sum(self):
        """A +1/-1 pair leaves the sum unchanged; the digest still moves."""
        a = np.arange(32, dtype=np.int64)
        b = a.copy()
        b[3] += 1
        b[9] -= 1
        assert a.sum() == b.sum(), "fixture broken: sums must match"
        assert content_hash48(a) != content_hash48(b)

    def test_shape_change_is_visible_at_equal_bytes(self):
        a = np.arange(12, dtype=np.int32)
        assert content_hash48(a) == content_hash48(np.ascontiguousarray(a))
        # same bytes, different logical shape -> callers also fingerprint
        # ndim/shape structurally, so this documents that the DIGEST alone
        # is shape-blind and must not be used without the struct entry.
        assert content_hash48(a) == content_hash48(a.reshape(3, 4))


class TestSchemaFingerprint:
    def test_shape_and_dtype_are_fixed(self):
        """Fixed shape is what lets every process reach the SAME collective
        even when their field lists differ — a variable-length payload would
        deadlock instead of reporting.

        Asserts the payloads AGREE with each other rather than pinning a
        literal width, so extending the fingerprint (as the dtype-kind/ndim
        terms did) does not silently turn this into a stale-constant test.
        The one thing that must never vary is that the width is independent
        of the INPUTS.
        """
        a = schema_fingerprint(["x", "y"], 4)
        b = schema_fingerprint(["completely", "different", "names"], 9)
        c = schema_fingerprint(["x"], 1, ["exact", "inexact"], [1, 2, 3])
        assert a.shape == b.shape == c.shape
        assert a.ndim == 1 and a.shape[0] >= 4
        assert a.dtype == b.dtype == c.dtype == np.float64

    def test_sensitive_to_field_list(self):
        assert not np.array_equal(schema_fingerprint(["a", "b"], 4),
                                  schema_fingerprint(["a", "c"], 4))

    def test_sensitive_to_field_order(self):
        """Per-field checks are matched positionally across processes, so a
        reordering IS a divergence."""
        assert not np.array_equal(schema_fingerprint(["a", "b"], 4),
                                  schema_fingerprint(["b", "a"], 4))

    def test_sensitive_to_field_count_and_device_count(self):
        assert not np.array_equal(schema_fingerprint(["a"], 4),
                                  schema_fingerprint(["a", "b"], 4))
        assert not np.array_equal(schema_fingerprint(["a"], 4),
                                  schema_fingerprint(["a"], 8))

    def test_name_join_is_not_ambiguous(self):
        """SAME-LENGTH lists whose comma-join is identical must still differ.

        codex 2026-07-29 (minor 5): the earlier fixture used ``["a,b"]`` vs
        ``["a", "b"]``, which differ in COUNT — so it passed even with a
        colliding join and proved nothing. These two have equal length and an
        identical ``",".join``, so only a separator that cannot occur in a
        name (NUL) distinguishes them.
        """
        left, right = ["a,b", "c"], ["a", "b,c"]
        assert len(left) == len(right)
        assert ",".join(left) == ",".join(right), "fixture must collide"
        assert name_digest48(left) != name_digest48(right)
        assert not np.array_equal(schema_fingerprint(left, 1),
                                  schema_fingerprint(right, 1))


class TestSingleProcessIsAPassthrough:
    """With one process there is nothing to compare: no collective may run
    (they would hang), and the value must come back untouched."""

    def test_broadcast_checked_returns_the_SAME_OBJECT(self):
        a = np.linspace(0.0, 1.0, 40).reshape(5, 8)
        assert broadcast_checked(a, "area", context="test") is a

    def test_single_process_does_NOT_convert_a_device_array_to_host(self):
        """codex 2026-07-29 (major 3): converting unconditionally forced a
        device->host->device round trip and stripped weak-type metadata on a
        1-process mesh. The atmosphere lane passes `jnp.stack` results
        straight in, so it must come back as the SAME jax array."""
        jnp = pytest.importorskip("jax.numpy")
        a = jnp.arange(6, dtype=jnp.int32)
        out = broadcast_checked(a, "idx", context="test")
        assert out is a
        assert not isinstance(out, np.ndarray)

    def test_assert_schema_agrees_is_a_noop(self):
        assert_schema_agrees(["a", "b"], 1, context="test") is None


class _FakeMultihost:
    """Records every collective payload and echoes N identical copies.

    Lets the MULTI-process branch of `broadcast_checked` / `assert_*` run
    under pytest, which is the only way to test the properties that matter:
    payload SHAPE (a shape that varies per rank deadlocks in the real thing)
    and which fingerprint the routing actually computed.
    """

    def __init__(self, n=2, override=None):
        self.n, self.override, self.seen = n, override, []

    def process_allgather(self, payload):
        self.seen.append(np.array(payload, dtype=np.float64))
        rows = [np.asarray(payload, dtype=np.float64) for _ in range(self.n)]
        if self.override is not None:
            # `override` may return None to leave a payload alone. Needed
            # because a single call site emits SEVERAL payloads of different
            # widths (struct=8 then vals=3); blanket-overriding them all
            # would (a) make np.stack fail on ragged rows and (b) let a test
            # pass for the wrong reason (struct mismatch masquerading as a
            # value mismatch).
            replaced = self.override(rows[-1])
            if replaced is not None:
                rows[-1] = np.asarray(replaced, dtype=np.float64)
                assert rows[-1].shape == rows[0].shape, (
                    "test override changed the payload WIDTH; in the real "
                    "collective that is a deadlock, not a divergence")
        return np.stack(rows)

    def broadcast_one_to_all(self, host):
        return host


@pytest.fixture
def multiproc(monkeypatch):
    """Force `jax.process_count() > 1` and install the fake collectives."""
    import jax
    from jax.experimental import multihost_utils

    def _install(n=2, override=None):
        fake = _FakeMultihost(n, override)
        monkeypatch.setattr(jax, "process_count", lambda: n)
        monkeypatch.setattr(multihost_utils, "process_allgather",
                            fake.process_allgather)
        monkeypatch.setattr(multihost_utils, "broadcast_one_to_all",
                            fake.broadcast_one_to_all)
        return fake
    return _install


class TestBroadcastCheckedRouting:
    """Exercise `broadcast_checked` ITSELF, not just its helpers.

    codex 2026-07-29 (major 4) correctly observed that testing
    `content_hash48` directly proves nothing about whether the helper still
    CALLS it for exact arrays — swapping that branch to moments would leave a
    hash-only test green. These tests read the payload the helper actually
    emitted.
    """

    def test_exact_array_payload_is_the_byte_digest(self, multiproc):
        fake = multiproc()
        a = np.arange(24, dtype=np.int32).reshape(4, 6)
        broadcast_checked(a, "idx", context="t")
        vals = fake.seen[-1]
        assert vals[0] == content_hash48(a), (
            "exact arrays must be fingerprinted by the POSITIONAL byte "
            "digest; a moment fingerprint here loses permutation detection")

    def test_float_array_payload_is_the_moment_triple(self, multiproc):
        fake = multiproc()
        a = np.linspace(-2.0, 3.0, 30).reshape(5, 6)
        broadcast_checked(a, "area", context="t")
        vals = fake.seen[-1]
        np.testing.assert_allclose(
            vals[:3], [a.sum(), (a * a).sum(), np.abs(a).max()], rtol=1e-12)

    def test_permutation_of_a_mask_is_REJECTED_end_to_end(self, multiproc):
        """The property the byte digest exists for, through the real helper.

        Peer rank reports the digest of a PERMUTATION: identical shape,
        identical dtype, identical sum/sumsq/absmax — so `struct` matches and
        ONLY the value payload can catch it. That is what makes this a test
        of the digest rather than of the struct compare.
        """
        a = np.zeros(64, dtype=bool)
        a[:8] = True
        perm = np.zeros(64, dtype=bool)
        perm[-8:] = True
        assert a.sum() == perm.sum() and a.shape == perm.shape

        # touch ONLY the 3-wide value payload; leave the 8-wide struct alone
        multiproc(override=lambda row: (
            _make_vals_for(perm) if row.shape == (3,) else None))
        with pytest.raises(RuntimeError, match="DIVERGES"):
            broadcast_checked(a, "wet_mask", context="t")

"""Multi-process (srun/mpirun -n 2) tests for the cross-process geometry guard.

Run:  srun -n 2 python -m pytest tests/distributed/test_geometry_consistency_mp.py
      (or mpirun -np 2 ...)

Covers what the serial suite CANNOT. Every guard in
``legoesm.parallel.geometry_consistency`` early-returns at
``jax.process_count() <= 1``, so the ~150 single-process tests exercise the
fingerprints that DECIDE whether a collective raises -- never the collectives
themselves. That gap matters because the failure mode is not a wrong answer but
a HANG: a rank-local ``raise`` ordered before a peer enters the matching
collective deadlocks, which is strictly worse than the inconsistency being
guarded against, and no single-process test can tell the two apart.

The governing invariant (see ``geometry_consistency`` module docstring):

    Every process must reach the same collectives in the same order.

So each test below asserts not merely that a divergence is detected, but that it
is detected SYMMETRICALLY -- every rank raises, none is left blocked.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _launcher_size() -> int:
    """World size of THIS invocation, without importing mpi4py.

    The guard uses ``jax.distributed`` / ``multihost_utils``, not mpi4jax, so
    this tier only needs the process count -- read it from whichever launcher is
    in play rather than taking on an extra dependency.

    ``SLURM_NTASKS`` is deliberately NOT used: it reports the JOB ALLOCATION, so
    a plain ``python -m pytest`` inside a 2-task allocation reads 2, tries to
    ``jax.distributed.initialize()`` alone, and dies with a coordinator
    ``ValueError`` instead of skipping. ``SLURM_STEP_NUM_TASKS`` is set by srun
    per STEP and is absent for a non-srun invocation, which is exactly the
    distinction needed.
    """
    for key in ("SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
                "MV2_COMM_WORLD_SIZE"):
        value = os.environ.get(key)
        if value and value.isdigit():
            return int(value)
    return 1


SIZE = _launcher_size()

pytestmark = pytest.mark.skipif(
    SIZE < 2, reason="needs srun -n 2 / mpirun -np 2 (guards no-op at process_count == 1)",
)

if SIZE >= 2:  # pragma: no cover - only under a multi-process launcher
    import jax

    # MUST precede any other JAX work in the process.
    jax.distributed.initialize()


def _guards():
    from legoesm.parallel.geometry_consistency import (
        assert_flags_agree,
        assert_schema_agrees,
        broadcast_checked,
    )

    return assert_schema_agrees, assert_flags_agree, broadcast_checked


def _rank() -> int:
    import jax

    return int(jax.process_index())


def test_process_count_is_really_multi():
    """Anti-vacuity for the whole module: if this collapses to 1, every test
    below passes trivially because the guards early-return."""
    import jax

    assert jax.process_count() >= 2, (
        f"launcher reported {SIZE} but jax.process_count() == {jax.process_count()}; "
        "the guards would early-return and these tests would be vacuous"
    )


def test_agreeing_geometry_passes_on_every_rank():
    """Identical inputs: no rank raises, and no rank blocks."""
    assert_schema_agrees, assert_flags_agree, broadcast_checked = _guards()

    assert_schema_agrees(("lat", "lon"), 2, context="mp-test")
    assert_flags_agree(("alpha", "beta"), (1.0, 0.0), context="mp-test")
    out = broadcast_checked(np.arange(4, dtype=np.float64), "field", context="mp-test")
    np.testing.assert_allclose(out, np.arange(4, dtype=np.float64))


def test_axis_order_divergence_raises_on_every_rank():
    """The sharp case: ``(lat,lon)`` vs ``(lon,lat)`` is invisible to a
    count-and-size payload, so this is what the name digest exists for.

    Both ranks must raise. If only one did, the other would still be inside
    ``process_allgather`` and the job would hang rather than fail.
    """
    assert_schema_agrees, _, _ = _guards()
    names = ("lat", "lon") if _rank() == 0 else ("lon", "lat")
    with pytest.raises(RuntimeError, match="SCHEMA differs"):
        assert_schema_agrees(names, 2, context="mp-test")


def test_flag_value_divergence_raises_on_every_rank():
    _, assert_flags_agree, _ = _guards()
    values = (1.0, 0.0) if _rank() == 0 else (1.0, 1.0)
    with pytest.raises(RuntimeError, match="CONFIG differs"):
        assert_flags_agree(("alpha", "beta"), values, context="mp-test")


def test_broadcast_rejects_divergent_content_instead_of_overwriting():
    """``broadcast_checked`` must REJECT a divergence, never silently replace
    every rank's array with rank 0's -- that would convert a real inconsistency
    (different wet domain, different polar mask) into silently wrong physics,
    which is the entire reason the broadcast is guarded rather than blind.
    """
    _, _, broadcast_checked = _guards()
    arr = np.arange(4, dtype=np.float64) + (0.0 if _rank() == 0 else 1.0)
    with pytest.raises(RuntimeError, match="DIVERGES"):
        broadcast_checked(arr, "field", context="mp-test")


def test_guards_still_agree_after_a_divergence_was_raised():
    """Ordering regression guard: a raised divergence must leave the collective
    stream aligned, so a SUBSEQUENT agreeing call still works on every rank.

    If a guard consumed a different number of collectives on the raising path
    than on the passing path, this is where the ranks would desynchronise.
    """
    assert_schema_agrees, _, broadcast_checked = _guards()
    out = broadcast_checked(np.full(3, 2.5, dtype=np.float64), "after", context="mp-test")
    np.testing.assert_allclose(out, 2.5)
    assert_schema_agrees(("lat", "lon"), 2, context="mp-test")
"""Route-B multi-controller gate for the lat-band SPMD atm step.

True multi-process ``jax.distributed`` parity: N processes (one per GPU on a
cluster; CPU host devices for the fabric-free check) federate into ONE
multi-controller program, the ("lat",) mesh spans the GLOBAL device set, and
``make_sharded_atm_latlon_step`` + the band ppermute/psum halo run UNCHANGED —
the collectives cross processes via the distributed runtime (NCCL/gloo). NO
mpi4jax anywhere in this file (mixing the mpi4jax halo machinery with
jax.distributed collectives in one program is the documented mixed-stack
deadlock hazard) — which is also why this file lives in ``tests/parallel/``,
NOT ``tests/distributed/`` (whose session conftest auto-arms the mpi4jax
layout).

GATED: skips unless ``LEGOESM_JAX_DISTRIBUTED_TEST=1`` — ``jax.distributed
.initialize`` must run BEFORE the first backend touch, so this file must be
launched ALONE (its own pytest process per rank), never inside a shared pytest
session. It is also env-fragile by nature (gloo needs a resolvable hostname on
CPU — broken on stock macOS; NCCL needs a working local topology) — the gate
keeps default CI green.

Run (2 processes x 2 host CPU devices = 4 bands):
  LEGOESM_JAX_DISTRIBUTED_TEST=1 JAX_PLATFORMS=cpu \
  XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  mpiexec -n 2 python -m pytest \
      tests/parallel/test_atm_latlon_spmd_multicontroller.py -x -q
(SLURM: srun -n 2 ... — coordinator auto-detected, OMPI env vars unneeded.)
"""
from __future__ import annotations

import os

import pytest

if os.environ.get("LEGOESM_JAX_DISTRIBUTED_TEST") != "1":
    pytest.skip(
        "multi-controller jax.distributed test: set "
        "LEGOESM_JAX_DISTRIBUTED_TEST=1 and launch this file alone under "
        "mpiexec/srun (see module docstring)",
        allow_module_level=True,
    )

import jax  # noqa: E402  (import gated so the skip never touches the backend)

# initialize() BEFORE any backend touch — the process count/id decision comes
# from the LAUNCHER env ONLY (querying jax.process_count() here would
# instantiate the local backend client pre-federation, the exact order bug
# this file warns about; codex round-1 HIGH). SLURM auto-detects; OpenMPI
# needs the explicit coordinator (fixed localhost port, single-node check).
_n = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                        os.environ.get("SLURM_NTASKS", "1")))
_r = int(os.environ.get("OMPI_COMM_WORLD_RANK",
                        os.environ.get("SLURM_PROCID", "0")))
if _n > 1:
    if os.environ.get("SLURM_STEP_NODELIST"):
        jax.distributed.initialize()
    else:
        _coord = os.environ.get("LEGOESM_JAX_COORDINATOR", "127.0.0.1:29777")
        jax.distributed.initialize(
            coordinator_address=_coord, num_processes=_n, process_id=_r)
    # Verify federation AFTER init (safe to touch the backend now).
    if jax.process_count() != _n:
        pytest.skip(
            f"jax.distributed federated {jax.process_count()} processes, "
            f"launcher started {_n} — environment did not federate",
            allow_module_level=True,
        )

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (  # noqa: E402
    gather_state_atm_latlon,
    make_sharded_atm_latlon_step,
    shard_state_atm_latlon,
)

# Reuse the Stage-5 gate's fixtures — same model/state/mesh conventions.
from tests.parallel.test_atm_latlon_spmd_step import (  # noqa: E402
    _model_and_state,
)


@pytest.fixture(autouse=True)
def _restore_halo_backend():
    yield
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("local")


def _global_mesh():
    gdev = jax.devices()
    if jax.process_count() < 2:
        pytest.skip("needs >= 2 processes (mpiexec/srun -n 2)")
    return jax.sharding.Mesh(np.array(gdev), axis_names=("lat",))


def test_multicontroller_step_matches_serial():
    """The integrated Stage-5 gate, multi-process: N global bands across >= 2
    processes must reproduce the per-process serial reference (identical host
    ICs on every process) at the FV-PPM cut-truncation bound, and the gather
    (the multiprocess ``legoesm.parallel.latlon_spmd.replicate_leaf`` branch,
    exercised FOR REAL here) must return the full global state on every
    process."""
    mesh = _global_mesh()
    n_dev = mesh.devices.size
    model, state = _model_and_state(use_polar_filter=False)
    n_lat = int(model.grid.n_lat)
    if n_lat % n_dev != 0:
        pytest.skip(f"n_lat {n_lat} % global devices {n_dev} != 0")
    dt, n_steps = 100.0, 3

    # Serial reference: every process computes it from the identical host IC.
    s = state
    for _ in range(n_steps):
        s, _ = model._step_cgrid(s, dt, target_mass=None, physics_fn=None)

    sharded_step = make_sharded_atm_latlon_step(model, mesh)
    sc = shard_state_atm_latlon(state, mesh)
    for _ in range(n_steps):
        sc = sharded_step(sc, dt)
    out = gather_state_atm_latlon(sc, mesh)

    for field in ("u", "v", "T", "p_s"):
        a = np.asarray(getattr(out, field))
        b = np.asarray(getattr(s, field))
        assert a.shape == b.shape, f"{field} shape {a.shape} vs {b.shape}"
        np.testing.assert_allclose(
            a, b, rtol=1e-6, atol=1e-9,
            err_msg=(
                f"multi-controller lat-band SPMD ({n_dev} bands, "
                f"{jax.process_count()} processes) diverged from serial in "
                f"'{field}' — cross-process ppermute/psum or the "
                f"multi-controller gather is wrong."))

    # Non-vacuity: the FV-PPM cut truncation must be present (bands really ran).
    u_diff = float(np.max(np.abs(np.asarray(out.u) - np.asarray(s.u))))
    assert u_diff > 1e-13, (
        "multi-controller u is bit-identical to serial — the band "
        "decomposition did not actually run; this gate would be vacuous.")
"""SELF-SPAWNING multi-controller equivalence gate for the atm lat-band SPMD step.

Companion to tests/parallel/test_atm_latlon_spmd_multicontroller.py (the
launcher-gated variant that needs `srun -n 2` + LEGOESM_JAX_DISTRIBUTED_TEST=1
and therefore never runs in plain pytest CI): THIS variant spawns its own two
worker processes, so the federation path is exercised on every CI run, with
port-race retries via multihost_harness.

Federates TWO OS processes into one JAX program (jax.distributed, CPU/Gloo —
the same federation layer NCCL uses on GPU nodes) and asserts the 2-band
shard_map trajectory matches the single-device serial reference to the SAME
tolerances as the single-process gate (test_atm_latlon_spmd_step.py:
rtol=1e-6 / atol=1e-9 over 3 RK3 steps — the FV-PPM cut-truncation bound; a
real decomposition bug is O(1e-3)).

What this pins beyond the single-process gate: the multi-controller
construction path — ``shard_state_atm_latlon``'s ``device_put`` onto a mesh
spanning NON-addressable devices, cross-process ppermute/psum inside the
jitted shard_map, and ``gather_state_atm_latlon``'s replication gather — i.e.
exactly the pieces a Derecho/Levante multi-node NCCL run exercises.

DUAL-USE FILE: run under pytest it spawns two subprocesses of ITSELF
(``python this_file.py <rank> <port>``), each executing ``_worker``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

N_PROC = 2
N_LAT = 16      # divisible by N_PROC
N_LON = 16
NLEV = 4
N_STEPS = 3
DT = 100.0
RTOL, ATOL = 1e-6, 1e-9   # single-process gate tolerances


def _worker(rank: int, port: int) -> int:
    import jax

    jax.distributed.initialize(
        coordinator_address=f"localhost:{port}",
        num_processes=N_PROC,
        process_id=rank,
    )
    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import numpy as np

    from legoesm import constants
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonHydrostaticState,
        CGridLatLonPrimitiveEquationConfig,
        CGridLatLonPrimitiveEquationModel,
    )
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        gather_state_atm_latlon,
        make_sharded_atm_latlon_step,
        shard_state_atm_latlon,
    )
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    assert jax.process_count() == N_PROC, jax.process_count()
    assert len(jax.devices()) == N_PROC, jax.devices()

    # Deterministic identical build on every process (the multi-controller
    # contract: device_put slices the same host-global array per process).
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True,                # exercises the cross-process psum
        use_polar_filter=True,
        use_ppm_transport=True,
        time_integrator="ssp_rk3",
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    rng = np.random.default_rng(31337)
    eps = 1.0e-3
    state = CGridLatLonHydrostaticState(
        u=jnp.asarray(eps * rng.standard_normal((N_LAT, N_LON + 1, NLEV))),
        v=jnp.asarray(eps * rng.standard_normal((N_LAT + 1, N_LON, NLEV))),
        T=jnp.asarray(300.0 + eps * rng.standard_normal((N_LAT, N_LON, NLEV))),
        p_s=jnp.asarray(1.0e5 + 10.0 * rng.standard_normal((N_LAT, N_LON))),
        phis=jnp.zeros((N_LAT, N_LON)),
    )
    v0 = np.array(state.v)
    v0[0] = 0.0
    v0[-1] = 0.0
    state = state._replace(v=jnp.asarray(v0))

    # Serial reference: identical on every process (no collectives).
    s = state
    for _ in range(N_STEPS):
        s, _ = model._step_cgrid(s, DT, target_mass=None, physics_fn=None)
    serial_out = s

    # Multi-controller SPMD: 2 bands, one device per PROCESS.
    mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=("lat",))
    sharded_step = make_sharded_atm_latlon_step(model, mesh)
    sc = shard_state_atm_latlon(state, mesh)
    for _ in range(N_STEPS):
        sc = sharded_step(sc, DT)
    spmd_out = gather_state_atm_latlon(sc, mesh)  # fully replicated

    failures = []
    for field in ("u", "v", "T", "p_s"):
        a = np.asarray(getattr(spmd_out, field))   # replicated -> host OK
        b = np.asarray(getattr(serial_out, field))
        if a.shape != b.shape:
            failures.append(f"{field}: shape {a.shape} vs {b.shape}")
            continue
        if not np.allclose(a, b, rtol=RTOL, atol=ATOL):
            failures.append(
                f"{field}: max|diff|={float(np.max(np.abs(a - b))):.3e}")
    if failures:
        print(f"rank{rank} MULTIHOST PARITY FAIL: " + "; ".join(failures),
              flush=True)
        return 1
    print(f"rank{rank} multihost parity OK", flush=True)
    return 0


def test_atm_latlon_spmd_two_process_selfspawn_matches_serial(tmp_path):
    from multihost_harness import run_federated

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env.pop("XLA_FLAGS", None)  # 1 real CPU device per process, no virtuals

    def build_cmd(rank, port):
        return [sys.executable, str(Path(__file__).resolve()), str(rank),
                str(port)]

    rcs, outs = run_federated(build_cmd, N_PROC, env, timeout_s=420)
    for rank, (rc, out) in enumerate(zip(rcs, outs)):
        assert rc == 0, (
            f"rank {rank} exited {rc}\n--- output ---\n{out[-4000:]}"
        )
        assert "multihost parity OK" in out


if __name__ == "__main__":
    raise SystemExit(_worker(int(sys.argv[1]), int(sys.argv[2])))
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:109:    spmd_out = gather_state_atm_latlon(sc, mesh)  # fully replicated
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:113:        a = np.asarray(getattr(spmd_out, field))   # replicated -> host OK
tests/parallel/test_atm_latlon_segment.py:22:      slice, ``P("lat")`` stacks) BIT-matches the replicated-geometry step
tests/parallel/test_atm_latlon_segment.py:135:    # replicated geometry).
tests/parallel/test_atm_latlon_segment.py:261:# (iii) geometry-sharded step bit-matches the replicated-geometry step
tests/parallel/test_atm_latlon_segment.py:265:def test_geometry_sharded_step_bitmatches_replicated(use_polar_filter):
tests/parallel/test_atm_latlon_segment.py:270:    step_rep = make_sharded_atm_latlon_step(model_r, mesh)  # default: replicated
tests/parallel/test_atm_latlon_segment.py:274:    assert step_rep._geom_stacks["area"].sharding.is_fully_replicated
tests/parallel/test_atm_latlon_segment.py:275:    assert not step_shd._geom_stacks["area"].sharding.is_fully_replicated
tests/parallel/test_atm_latlon_segment.py:279:                    .sharding.is_fully_replicated)
tests/parallel/test_atm_latlon_segment.py:292:                     f"replicated-geometry step in '{f}' "
tests/parallel/test_atm_latlon_segment.py:298:    replicated layout (uniform bands), and the 2-D fields dominate."""
tests/parallel/test_atm_latlon_segment.py:304:        geo["replicated_per_device_bytes"]
tests/parallel/test_atm_latlon_segment.py:305:    assert geo["sharded_per_device_bytes"] < geo["replicated_per_device_bytes"]
tests/parallel/test_atm_latlon_segment.py:310:    assert geo["replicated_per_device_bytes"] >= N_LAT * N_LON * itemsize * 5
tests/parallel/test_atm_latlon_operator_split_spmd.py:183:    # Non-vacuity: the carry is ACTUALLY lat-PARTITIONED (not merely replicated
tests/parallel/test_atm_latlon_operator_split_spmd.py:184:    # across the mesh — num_devices==N_DEV holds for a replicated P() too). The
tests/parallel/test_atm_latlon_spmd_step.py:277:    replicated out_shardings) must produce the SAME replicated array as the
tests/parallel/test_atm_latlon_spmd_step.py:279:    lat-sharded and an already-replicated input. This exercises the
tests/parallel/test_atm_latlon_spmd_step.py:296:        assert b.sharding.is_fully_replicated, (
tests/parallel/test_atm_latlon_spmd_step.py:298:            "replicated sharding")
tests/parallel/test_atm_latlon_2d_tiling.py:337:        _build_geometry_stacks_2d, atm_grid_array_field_names)
tests/parallel/test_atm_latlon_2d_tiling.py:353:    template, afn, stacks, stacks_spec = _build_geometry_stacks_2d(
tests/atmosphere/nonhydrostatic/integration/test_bench_mpi_scaling_smoke.py:7:The replicated-dycore design (rank-0 step + bcast) is the
tests/atmosphere/nonhydrostatic/integration/test_bench_mpi_scaling_smoke.py:12:rank single-process smoke covering argparse + replicated step
tests/atmosphere/nonhydrostatic/integration/test_bench_mpi_scaling_smoke.py:40:    """Single-rank smoke covering argparse, replicated-dycore step
tests/atmosphere/nonhydrostatic/integration/test_bench_mpi_scaling_smoke.py:86:    # Real replicated step on 12x12x8 takes ~2 ms per step; an
tests/atmosphere/nonhydrostatic/integration/test_bench_mpi_scaling_smoke.py:89:        f"dycore_ms={dycore_ms} below 1us floor; replicated step "
tests/atmosphere/dycore/regression/test_dgrid_halo_iter1076.py:163:        # (replicated outward), which is face 1's own value (=2).
tests/atmosphere/hydrostatic/unit/test_e3sm_cam_gwd_faithful.py:61:    in test_gwd_e3sm_cam.py; replicated to keep this canary file standalone).
tests/atmosphere/dycore/unit/test_cdgrid_flux_form_moisture.py:31:    """The fail-closed predicate: MPI (single-rank / replicated / face-scatter) is
tests/atmosphere/hydrostatic/unit/test_topography.py:453:    ±90.  The source latitude axis is now edge-replicated to exactly ±90."""
tests/atmosphere/hydrostatic/unit/test_topography.py:455:    def test_regrid_pole_rows_edge_replicated(self):

exec
/usr/bin/bash -lc "sed -n '1,360p' tests/parallel/test_atm_latlon_segment.py; sed -n '300,420p' tests/parallel/test_atm_latlon_2d_tiling.py; sed -n '1,130p' tests/parallel/test_atm_latlon_spmd_step.py; sed -n '650,760p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; sed -n '900,980p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; sed -n '1685,1765p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 238ms:
"""M2b parity gate — compiled multi-step lat-band SPMD segments + band-SHARDED
geometry for the lat-lon C-grid hydrostatic atm.

Three merge-bar gates (2 virtual CPU devices, x64):

(i)   ``make_sharded_atm_latlon_segment(n_steps=10)`` — ONE compiled
      ``lax.scan`` — matches 10 sequential ``make_sharded_atm_latlon_step``
      calls to 1e-12, and 10 SERIAL single-device steps: to 1e-12 with
      centered transport (the band decomposition is exact there), and to the
      documented Stage-5 limited-FV-PPM cut-truncation bound with PPM on
      (``tests/parallel/test_atm_latlon_spmd_step.py`` module docstring — the
      2nd-order halo edge at cut rows is a boundary-order property of
      band-decomposed limited PPM, not an SPMD bug).

(ii)  the segment's IN-GRAPH finite scalar (``state_finite_scalar``: psum of
      per-band non-finite presence over ALL state leaves) agrees with a
      host-side isfinite of the gathered state — on a healthy run AND under
      synthetic NaN injections in T (band 0) and u-only (band 1), so the gate
      is provably non-vacuous and the cross-band psum is exercised.

(iii) the geometry-SHARDED step (``shard_geometry=True``: per-device band
      slice, ``P("lat")`` stacks) BIT-matches the replicated-geometry step
      (the historical default) — with and without the polar-filter mask
      stacks — and the layouts are REALLY different on device (sharding
      introspection), so the bit-match cannot pass vacuously.

Plus the production-lane wiring gate: ``run_atm_latlon_spmd(...,
compiled_segments=True)`` matches the per-step path (final state + status,
including the remainder segment and the BLOWUP status contract).

Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count>=2``).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
    cgrid_to_hydrostatic,
)
from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
    atm_latlon_geometry_bytes,
    gather_state_atm_latlon,
    make_sharded_atm_latlon_segment,
    make_sharded_atm_latlon_step,
    run_atm_latlon_spmd,
    shard_state_atm_latlon,
    state_finite_scalar,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

N_DEV = 2
N_LAT = 16        # divisible by N_DEV
N_LON = 16
NLEV = 4
DT = 100.0


@pytest.fixture(autouse=True)
def _restore_halo_backend():
    yield
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("local")


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]),
                             axis_names=("lat",))


def _model_and_state(use_ppm_transport=True, use_polar_filter=False):
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True,                 # exercises the psum'd mass fixer
        use_polar_filter=use_polar_filter,
        use_ppm_transport=use_ppm_transport,
        time_integrator="ssp_rk3",
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    rng = np.random.default_rng(31337)
    eps = 1.0e-3
    state = CGridLatLonHydrostaticState(
        u=jnp.asarray(eps * rng.standard_normal((N_LAT, N_LON + 1, NLEV))),
        v=jnp.asarray(eps * rng.standard_normal((N_LAT + 1, N_LON, NLEV))),
        T=jnp.asarray(300.0 + eps * rng.standard_normal((N_LAT, N_LON, NLEV))),
        p_s=jnp.asarray(1.0e5 + 10.0 * rng.standard_normal((N_LAT, N_LON))),
        phis=jnp.zeros((N_LAT, N_LON)),
    )
    # Pole-wall v (the state the model maintains: v == 0 at both poles), so
    # the v_lower drop + gather re-append round-trips.
    v0 = np.array(state.v)
    v0[0] = 0.0
    v0[-1] = 0.0
    return model, state._replace(v=jnp.asarray(v0))


def _fields(state):
    return {f: np.asarray(getattr(state, f)) for f in ("u", "v", "T", "p_s")}


# ==============================================================================
# (i) segment == sequential sharded steps == serial
# ==============================================================================

@pytest.mark.parametrize("use_ppm", [False, True])
def test_segment_matches_sequential_and_serial(use_ppm):
    mesh = _mesh()
    n_steps = 10
    seg_model, c0 = _model_and_state(use_ppm_transport=use_ppm)
    seq_model, _ = _model_and_state(use_ppm_transport=use_ppm)
    ser_model, _ = _model_and_state(use_ppm_transport=use_ppm)

    # (a) 10 SERIAL single-device steps (the truth trajectory).
    s = c0
    for _ in range(n_steps):
        s, _ = ser_model._step_cgrid(s, DT)
    serial_out = _fields(s)

    # (b) 10 sequential per-step sharded calls (the historical SPMD path,
    # replicated geometry).
    step = make_sharded_atm_latlon_step(seq_model, mesh)
    sc = shard_state_atm_latlon(c0, mesh)
    for _ in range(n_steps):
        sc = step(sc, DT)
    seq_out = _fields(gather_state_atm_latlon(sc, mesh))

    # (c) ONE compiled segment of 10 steps (scan inside one jitted shard_map,
    # band-SHARDED geometry).
    seg = make_sharded_atm_latlon_segment(seg_model, mesh, n_steps)
    c_seg, ok = seg(shard_state_atm_latlon(c0, mesh), DT)
    assert bool(ok), "segment finite scalar is False on a healthy run"
    # Non-vacuity: the segment output is genuinely lat-sharded across the mesh.
    assert c_seg.T.sharding.num_devices == N_DEV
    seg_out = _fields(gather_state_atm_latlon(c_seg, mesh))

    # segment vs sequential sharded: same band numerics + decomposition; only
    # scan-vs-unrolled compilation differs -> 1e-12.
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            seg_out[f], seq_out[f], rtol=1e-12, atol=1e-12,
            err_msg=(f"compiled segment diverged from sequential sharded "
                     f"steps in '{f}' — the scanned band body is not the "
                     f"per-step body"))

    # segment vs SERIAL: exact decomposition without PPM; the documented
    # Stage-5 limited-FV-PPM cut-row truncation bound with PPM on.
    rtol, atol = ((1e-6, 1e-9) if use_ppm else (1e-12, 1e-12))
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            seg_out[f], serial_out[f], rtol=rtol, atol=atol,
            err_msg=(f"compiled segment (ppm={use_ppm}) diverged from the "
                     f"serial single-device trajectory in '{f}' beyond the "
                     f"documented bound — a real decomposition/scan bug"))


def test_segment_single_device_matches_serial_loop():
    """mesh=None twin: jit(scan) over the serial C-grid step == the per-step
    _step_cgrid loop."""
    model, c0 = _model_and_state()
    ser_model, _ = _model_and_state()
    seg = make_sharded_atm_latlon_segment(model, None, 5)
    out, ok = seg(c0, DT)
    assert bool(ok)
    s = c0
    for _ in range(5):
        s, _ = ser_model._step_cgrid(s, DT)
    a, b = _fields(out), _fields(s)
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            a[f], b[f], rtol=1e-12, atol=1e-13,
            err_msg=f"mesh=None segment diverged from the serial loop in '{f}'")


def test_segment_with_held_suarez_matches_sequential():
    """The production physics envelope (stateless Held-Suarez) threads through
    the compiled scan identically to the per-step path."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
    mesh = _mesh()
    seg_model, c0 = _model_and_state()
    seq_model, _ = _model_and_state()
    n_steps = 5

    seg = make_sharded_atm_latlon_segment(
        seg_model, mesh, n_steps, physics_fn=held_suarez_forcing_latlon)
    c_seg, ok = seg(shard_state_atm_latlon(c0, mesh), DT)
    assert bool(ok)

    step = make_sharded_atm_latlon_step(
        seq_model, mesh, physics_fn=held_suarez_forcing_latlon)
    sc = shard_state_atm_latlon(c0, mesh)
    for _ in range(n_steps):
        sc = step(sc, DT)

    a = _fields(gather_state_atm_latlon(c_seg, mesh))
    b = _fields(gather_state_atm_latlon(sc, mesh))
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            a[f], b[f], rtol=1e-12, atol=1e-12,
            err_msg=f"HS segment diverged from HS per-step path in '{f}'")


# ==============================================================================
# (ii) in-graph finite scalar == host-side isfinite of the gathered state
# ==============================================================================

def _host_all_finite(c_gathered) -> bool:
    return all(bool(np.isfinite(np.asarray(leaf)).all())
               for leaf in jax.tree.leaves(c_gathered))


def test_segment_finite_scalar_matches_host_isfinite():
    mesh = _mesh()
    model, c0 = _model_and_state()
    seg = make_sharded_atm_latlon_segment(model, mesh, 3)

    # Healthy run: scalar True, host agrees.
    c_out, ok = seg(shard_state_atm_latlon(c0, mesh), DT)
    assert bool(ok) is True
    assert _host_all_finite(gather_state_atm_latlon(c_out, mesh)) is True

    # Synthetic violations (non-vacuous gate): NaN seeded in a band-0 T cell
    # and in a band-1 u cell (u-only exercises the all-leaves superset AND the
    # cross-band psum — the bad band is NOT the one whose scalar the host
    # would trivially see).
    for field, idx in (("T", (3, 5, 0)), ("u", (9, 2, 1))):
        arr = np.array(np.asarray(getattr(c0, field)))
        arr[idx] = np.nan
        c_bad = c0._replace(**{field: jnp.asarray(arr)})
        c_out_b, ok_b = seg(shard_state_atm_latlon(c_bad, mesh), DT)
        host_ok = _host_all_finite(gather_state_atm_latlon(c_out_b, mesh))
        assert bool(ok_b) is False, (
            f"in-graph finite scalar missed a NaN seeded in {field}[{idx}]")
        assert host_ok is False
        assert bool(ok_b) == host_ok


def test_state_finite_scalar_serial_and_inf():
    """The scalar reducer itself (no mesh): catches Inf as well as NaN."""
    _, c0 = _model_and_state()
    assert bool(state_finite_scalar(c0)) is True
    bad = c0._replace(p_s=c0.p_s.at[0, 0].set(jnp.inf))
    assert bool(state_finite_scalar(bad)) is False


# ==============================================================================
# (iii) geometry-sharded step bit-matches the replicated-geometry step
# ==============================================================================

@pytest.mark.parametrize("use_polar_filter", [False, True])
def test_geometry_sharded_step_bitmatches_replicated(use_polar_filter):
    mesh = _mesh()
    model_r, c0 = _model_and_state(use_polar_filter=use_polar_filter)
    model_s, _ = _model_and_state(use_polar_filter=use_polar_filter)

    step_rep = make_sharded_atm_latlon_step(model_r, mesh)  # default: replicated
    step_shd = make_sharded_atm_latlon_step(model_s, mesh, shard_geometry=True)

    # Non-vacuity: the two layouts are REALLY different on device.
    assert step_rep._geom_stacks["area"].sharding.is_fully_replicated
    assert not step_shd._geom_stacks["area"].sharding.is_fully_replicated
    if use_polar_filter:
        assert "__polar_mask" in step_shd._geom_stacks
        assert not (step_shd._geom_stacks["__polar_mask"]
                    .sharding.is_fully_replicated)

    sc_r = shard_state_atm_latlon(c0, mesh)
    sc_s = shard_state_atm_latlon(c0, mesh)
    for _ in range(3):
        sc_r = step_rep(sc_r, DT)
        sc_s = step_shd(sc_s, DT)
    a = _fields(gather_state_atm_latlon(sc_r, mesh))
    b = _fields(gather_state_atm_latlon(sc_s, mesh))
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_array_equal(
            a[f], b[f],
            err_msg=(f"geometry-sharded step is not BIT-identical to the "
                     f"replicated-geometry step in '{f}' "
                     f"(polar_filter={use_polar_filter})"))


def test_atm_latlon_geometry_bytes_shrinks_by_ndev():
    """The honest residency numbers: band-sharding is exactly 1/n_dev of the
    replicated layout (uniform bands), and the 2-D fields dominate."""
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    geo = atm_latlon_geometry_bytes(grid, N_DEV)
    assert geo["sharded_per_device_bytes"] * N_DEV == \
        geo["replicated_per_device_bytes"]
    assert geo["sharded_per_device_bytes"] < geo["replicated_per_device_bytes"]
    # At least the five (n_lat, n_lon) 2-D fields (lat2d, lon2d, f, dx, area)
    # at the grid's ACTUAL dtype — LatLonGrid arrays follow the finite-volume
    # precision policy (f32 today), so never hardcode an 8-byte itemsize.
    itemsize = np.asarray(grid.area).dtype.itemsize
    assert geo["replicated_per_device_bytes"] >= N_LAT * N_LON * itemsize * 5


# ==============================================================================
# production-lane wiring: run_atm_latlon_spmd(compiled_segments=True)
# ==============================================================================

def test_run_atm_latlon_spmd_compiled_segments_matches_per_step():
    """7 steps in 3-step segments (3+3+1: exercises the remainder-length
    compile) — final state matches the per-step path to 1e-12, same status."""
    mesh = _mesh()
    model_a, c0 = _model_and_state()
    model_b, _ = _model_and_state()
    hs0 = cgrid_to_hydrostatic(c0, model_a.grid)

    hs_ref, st_ref = run_atm_latlon_spmd(
        model_a, mesh, hs0, DT, 7, segment_steps=3)
    hs_new, st_new = run_atm_latlon_spmd(
        model_b, mesh, hs0, DT, 7, segment_steps=3, compiled_segments=True)

    assert st_ref == "COMPLETED"
    assert st_new == "COMPLETED"
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            np.asarray(getattr(hs_new, f).data),
            np.asarray(getattr(hs_ref, f).data),
            rtol=1e-12, atol=1e-12,
            err_msg=f"compiled_segments run diverged from per-step in '{f}'")


def test_run_atm_latlon_spmd_compiled_segments_blowup_status():
    """A NaN'd IC must report the SAME 'BLOWUP at step N' boundary on both
    paths (the in-graph scalar is a superset of the gathered p_s/T check)."""
    mesh = _mesh()
    model_a, c0 = _model_and_state()
    model_b, _ = _model_and_state()
    arr = np.array(np.asarray(c0.T))
    arr[0, 0, 0] = np.nan
    hs_bad = cgrid_to_hydrostatic(c0._replace(T=jnp.asarray(arr)),
                                  model_a.grid)

    _, st_old = run_atm_latlon_spmd(model_a, mesh, hs_bad, DT, 6,
                                    segment_steps=3)
    _, st_new = run_atm_latlon_spmd(model_b, mesh, hs_bad, DT, 6,
                                    segment_steps=3, compiled_segments=True)
    assert st_old == "BLOWUP at step 3"
    assert st_new == "BLOWUP at step 3"


def test_run_atm_latlon_spmd_compiled_segments_on_segment_cadence():
    """The on_segment callback still fires at every boundary with gathered
    suppresses any CUT-FACE METRIC defect to ~4e-12 in dT because the
    advective form's corrupted-face terms cancel to
    ``v*hx*(q_face - q_c)/area`` — both factors 1e-3-scale there.  At
    realistic amplitude the pre-fix fabricated-pole metric (codex M3a
    findings 2/6) measures 8.5e-5 K/s = 7 K/day at cut rows (job 8970803),
    20x the genuine halo-2 limiter truncation (4.3e-6)."""
    grid = model.grid
    rng = np.random.default_rng(4242)
    lat = np.asarray(grid.lat)                      # (n_lat,)
    u0 = 10.0 * rng.standard_normal((N_LAT, N_LON + 1, NLEV))
    v0 = 10.0 * rng.standard_normal((N_LAT + 1, N_LON, NLEV))
    v0[0] = 0.0
    v0[-1] = 0.0
    u0[:, -1] = u0[:, 0]
    # sin(lat): MAXIMUM meridional T gradient and ZERO curvature at the
    # equator — i.e. at the (2,2) lat-cut face — so the fabricated-pole
    # metric defect signal (~ v*hx*(q_face - q_c)) is maximal there while
    # the genuine PPM edge-order truncation (~ curvature) is minimal:
    # the sharpest discrimination between the two.
    T0 = (250.0 + 40.0 * np.sin(lat)[:, None, None]
          + 0.5 * rng.standard_normal((N_LAT, N_LON, NLEV)))
    ps0 = 1.0e5 + 1.0e3 * rng.standard_normal((N_LAT, N_LON))
    return CGridLatLonHydrostaticState(
        u=jnp.asarray(u0),
        v=jnp.asarray(v0),
        T=jnp.asarray(T0),
        p_s=jnp.asarray(ps0),
        phis=jnp.zeros((N_LAT, N_LON)),
    )


def _serial_and_tile_tendency(state=None):
    """Serial tendencies and the (2,2)-tile shard_map body's tendencies,
    gathered to global (u as u_left, v as v_lower)."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        cgrid_latlon_hydrostatic_tendencies)
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        _build_geometry_stacks_2d, atm_grid_array_field_names)
    from legoesm.parallel.latlon_spmd import (
        activate_latlon_spmd_halo, deactivate_latlon_spmd_halo,
        latlon_band_perms, reconstruct_uface_left, reconstruct_vface_lower,
        to_uface_left, to_vface_lower)

    p_lat, p_lon = 2, 2
    mesh = _mesh2d(p_lat, p_lon)
    model, default_state = _model_and_state()
    state = default_state if state is None else state
    grid, sigma, cfg = model.grid, model.sigma_coord, model.config

    du_s, dv_s, dT_s, dps_s, _ = cgrid_latlon_hydrostatic_tendencies(
        state, grid, sigma, cfg)
    serial = tuple(np.asarray(x) for x in (du_s, dv_s, dT_s, dps_s))

    template, afn, stacks, stacks_spec = _build_geometry_stacks_2d(
        model, mesh, p_lat, p_lon, shard_geometry=True)
    perm_north, _ = latlon_band_perms(p_lat)

    def _body(sl, st):
        tg = template._replace(**{n: st[n][0, 0] for n in afn})
        vf = reconstruct_vface_lower(sl.v, "lat", perm_north)
        uf = reconstruct_uface_left(sl.u, "lon", p_lon)
        du, dv, dT, dps, _ = cgrid_latlon_hydrostatic_tendencies(
            sl._replace(u=uf, v=vf), tg, sigma, cfg)
        return to_uface_left(du), to_vface_lower(dv), dT, dps

    sc = shard_state_atm_latlon_2d(state, mesh)
    in_spec = jax.tree.map(tile_spec, sc)
    sp3 = P("lat", "lon", None)
    sp2 = P("lat", "lon")
    fn = shard_map(
        _body, mesh=mesh, in_specs=(in_spec, stacks_spec),
        out_specs=(sp3, sp3, sp3, sp2), check_vma=False)
    activate_latlon_spmd_halo(mesh)
    try:
        out = fn(sc, stacks)
    finally:
        deactivate_latlon_spmd_halo()
    g = lambda x: np.asarray(jax.device_put(x, NamedSharding(mesh, P())))
    dul_b, dvl_b, dT_b, dps_b = (g(x) for x in out)
    return serial, (dul_b, dvl_b, dT_b, dps_b), (p_lat, p_lon)


def test_2d_tendency_matches_serial():
    """Momentum + continuity BIT-tight on the (2,2) tiling; dT carries only
    the documented limited-FV-PPM halo-2 boundary-order truncation at
    cut-adjacent lat rows AND lon columns."""
    (du_s, dv_s, dT_s, dps_s), (dul_b, dvl_b, dT_b, dps_b), (p_lat, p_lon) = (
        _serial_and_tile_tendency())

    # Layout invariants that justify the dropped stagger slots: the serial
    # tendency preserves the u periodic seam and the v pole walls.
    np.testing.assert_array_equal(
        du_s[:, -1], du_s[:, 0],
        err_msg="serial du seam identity broke — u_left carry is unsound.")
    np.testing.assert_array_equal(dv_s[0], np.zeros_like(dv_s[0]))
    np.testing.assert_array_equal(dv_s[-1], np.zeros_like(dv_s[-1]))

    for name, a, b in (("du_dt", dul_b, du_s[:, :N_LON]),
                       ("dv_dt", dvl_b, dv_s[:N_LAT]),
                       ("dp_s_dt", dps_b, dps_s)):
        np.testing.assert_allclose(
            a, b, rtol=1e-10, atol=1e-12,
            err_msg=(
                f"atm 2-D tile tendency '{name}' diverged from serial beyond "
                f"fp64 precision — a real tile-decomposition bug (lon ring, "
                f"u seam, Coriolis/vertex at cuts, geometry, or the "
                f"('lat','lon') mass psum)."))

    # dT: bit-tight away from cuts; bounded truncation at cut-adjacent lat
    # rows and lon columns (the wrap cut cols {n_lon-1, 0} included — the
    # tile reconstruction is halo-2 there while serial sees the full circle).
    #
    # WHY a residual is genuinely allowed at cuts (and only there): on the
    # halo-2 padded tile the outermost ``ppm_edge_values`` edges are
    # 2ND-order (a 4th-order edge needs 2 cells each side) and feed the
    # GHOST-cell parabolas, whose CW84-limited a_R/a_L ARE the owned
    # cut-face upwind states — a real one-sided-stencil truncation needing
    # halo 3 to vanish, not a decomposition bug.  The cut-face flux METRIC
    # carries NO residual: ``lat_v_interfaces`` returns the tile's owned
    # ``grid.lat_v`` (codex M3a findings 2/6 — the pre-fix operator
    # fabricated ±π/2 pole faces at every cut; gated bitwise below by
"""Stage 5 — the serial-vs-SPMD EQUIVALENCE GATE for the atm latlon step.

THE make-or-break validation of make_sharded_atm_latlon_step: N-band lat-band
SPMD integration must reproduce the single-device C-grid hydrostatic step. A
subtly-wrong band decomposition (wrong Coriolis / v-face interp at interior
cuts, band-local mass denominator, mis-reconstructed v-face row, wrong band
geometry) gives SILENT wrong answers and is caught here.

Two complementary gates:

1. ``test_atm_latlon_spmd_tendency_matches_serial`` — the RIGOROUS,
   RK-stage-independent decomposition check at the TENDENCY level.  The
   vector-invariant momentum (vorticity + Bernoulli gradient, incl. the
   absolute-vorticity Coriolis) and the centered flux-form continuity
   reconstruct the band cut EXACTLY from halo'd neighbour rows, so du/dv/dp_s
   and cor_u/cor_v are BIT-EXACT (fp64) on every band.  The ONLY non-bit-exact
   term is the LIMITED FV PPM scalar (T/tracer) advection: ``ppm_edge_values``
   uses a 2nd-order edge at the outermost ``halo=2`` padded row, where serial
   computes a 4th-order edge, and the CW84 limiter leaks a ~5e-12 difference
   into the two cut-ADJACENT T rows (verified: scalar-advection diff is
   confined to those rows; the scalar halo ghost rows are byte-identical).
   This is a PRE-EXISTING boundary-order property of band-decomposed limited
   FV PPM — the production MPI lat-band path uses the SAME backend-dispatched
   operator — NOT an SPMD bug, so it is asserted as a BOUNDED residual.

2. ``test_atm_latlon_spmd_step_matches_serial`` — the integrated full-step
   gate through the PUBLIC make_sharded interface.  Over a few RK3 steps the
   tiny cut-row T truncation advects into the small-magnitude u field; bound it
   FAR below any real decomposition error (~1e-3) while above the PPM
   truncation floor.

Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``);
the production target is multi-GPU/TPU but the shard_map/ppermute/psum logic is
device-agnostic.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
    make_sharded_atm_latlon_step,
    shard_state_atm_latlon,
    gather_state_atm_latlon,
    run_atm_latlon_spmd_segment,
    build_band_grids_atm,
    atm_grid_array_field_names,
    lat_spec,
)
from legoesm.parallel.latlon_spmd import replicate_leaf
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    cgrid_to_hydrostatic,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

N_DEV = 4
N_LAT = 16        # divisible by N_DEV
N_LON = 16
NLEV = 4


@pytest.fixture(autouse=True)
def _restore_halo_backend():
    yield
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("local")


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


def _model_and_state(use_polar_filter: bool):
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True,                 # exercises batch_global_area_sums psum
        use_polar_filter=use_polar_filter,
        use_ppm_transport=True,
        time_integrator="ssp_rk3",
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    rng = np.random.default_rng(31337)
    eps = 1.0e-3
    state = CGridLatLonHydrostaticState(
        u=jnp.asarray(eps * rng.standard_normal((N_LAT, N_LON + 1, NLEV))),
        v=jnp.asarray(eps * rng.standard_normal((N_LAT + 1, N_LON, NLEV))),
        T=jnp.asarray(300.0 + eps * rng.standard_normal((N_LAT, N_LON, NLEV))),
        p_s=jnp.asarray(1.0e5 + 10.0 * rng.standard_normal((N_LAT, N_LON))),
        phis=jnp.zeros((N_LAT, N_LON)),
    )
    # Pole-wall v (the state the model maintains: v == 0 at both poles), so the
    # v_lower drop + gather re-append round-trips. np.array makes a WRITABLE copy
    # (np.asarray of a jax array is read-only).
    v0 = np.array(state.v)
    v0[0] = 0.0
    v0[-1] = 0.0
    state = state._replace(v=jnp.asarray(v0))
    return model, state


def _cut_adjacent_lat_rows():
    """The two cell rows flanking every interior band cut (the only rows the
    limited FV PPM scalar advection truncates under band decomposition)."""
    nl = N_LAT // N_DEV
    rows = set()
    for r in range(1, N_DEV):
        rows.add(r * nl - 1)   # last cell of the south band
        rows.add(r * nl)       # first cell of the north band
    return rows


def _serial_and_band_tendency(use_polar_filter):
    """Serial ``cgrid_latlon_hydrostatic_tendencies`` and the N-band shard_map
      folding it in costs no payload width.
    * ``n_steps`` / ``segment_steps`` -- unequal positive values capture
      different STATIC scan lengths and drive different numbers of host-side
      SPMD dispatches (codex round-2 blocker 3, round-3 blocker 4). Both are
      routed through :func:`coerce_count`, which NEVER raises: an uncoercible
      or aliasing value becomes a sentinel that travels through the collective
      and is refused symmetrically BELOW.
    * ``compiled_segments`` -- selects the compiled-scan lane vs the per-step
      lane. Two processes on different lanes run different programs.
    * ``has_physics_fn`` / ``has_phys_state`` -- change the traced program and
      the carry contract.
    * ``has_on_segment`` -- the callback triggers a per-segment
      ``gather_atm_latlon_to_hydrostatic``, itself a cross-process
      replication. A callback on some ranks only = unmatched gathers.
    * ``shard_geometry`` -- changes the geometry stacks' ``PartitionSpec``.

    ``None`` for any of these encodes "not applicable at this call site" and
    maps to a fixed sentinel, so the payload WIDTH is set by
    ``_SPMD_ENTRY_FLAGS`` alone and never by rank-local data.
    """
    # Attribute reads are ALL defensive. The rule this enforces: any value
    # that can legitimately differ between processes (n_steps, grid dims, the
    # config flags) must not be able to raise while the payload is being
    # assembled, because that raise lands BEFORE the collective and hangs the
    # peers. Structural type errors (a caller passing the wrong object) are
    # identical on every rank in an SPMD launch, but reading them through
    # getattr costs nothing and removes the last pre-collective throw sites.
    grid = getattr(model, "grid", None)
    cfg = getattr(model, "config", None)
    fold = getattr(grid, "fold", None)
    shape = dict(mesh.shape) if mesh is not None else {}
    names, sizes = _mesh_axis_terms(mesh)
    # Every user- or object-supplied value goes through a STRICT, NON-THROWING
    # encoder so building this payload cannot raise before the collective
    # (codex round-3 blocker 3, round-4 blockers 1-2). The mesh-derived sizes
    # are plain ints by construction inside ``jax.sharding.Mesh``.
    problems = []

    def _count(value, label, absent=FLAG_ABSENT):
        payload, problem = coerce_count(value, absent=absent)
        if problem is not None:
            problems.append((label, problem))
        return payload

    def _flag(value, label):
        payload, problem = coerce_bool(value, absent=FLAG_ABSENT)
        if problem is not None:
            problems.append((label, problem))
        return payload

    flags = (
        float(mesh is not None),
        float(mesh.devices.size if mesh is not None else 0),
        float(len(names)),
        name_digest48(names),
        name_digest48(sizes),
        float(shape.get("lat", 0)),
        float(shape.get("lon", 0)),
        _count(getattr(grid, "n_lat", None), "grid.n_lat", absent=0.0),
        _count(getattr(grid, "n_lon", None), "grid.n_lon", absent=0.0),
        # The grid's own array fields are what `build_band_grids_atm` slices
        # and `_build_geometry_stacks` broadcasts; agreeing their dtypes and
        # shapes here turns a structural mismatch into a clean raise instead
        # of a mismatched per-field gather.
        tree_schema_digest48(grid),
        float(bool(fold is not None and getattr(fold, "is_active", False))),
        # ONE digest over EVERY static scalar of the model config, not a
        # hand-picked trio. `fix_mass` gates a global-area psum,
        # `zero_mean_ps_tendency` and the integrator selection select different
        # programs -- all were missing from the hand-written list (codex
        # round-4, blocker 3). A digest closes the class, not the instances.
        config_digest48(cfg),
        # Rank-local CACHE presence. `_build_geometry_stacks` conditions on
        # `_polar_mask` and then slices `_polar_mask_v` unconditionally, so a
        # rank holding one but not the other dies before the schema gather
        # (codex round-4, blocker 4). Both are agreed; the refusal that
        # follows is symmetric.
        float(getattr(model, "_polar_mask", None) is not None),
        float(getattr(model, "_polar_mask_v", None) is not None),
        _count(n_steps, "n_steps"),
        _count(segment_steps, "segment_steps"),
        _flag(compiled_segments, "compiled_segments"),
        _flag(has_physics_fn, "has_physics_fn"),
        _flag(has_on_segment, "has_on_segment"),
        _flag(has_phys_state, "has_phys_state"),
        _flag(shard_geometry, "shard_geometry"),
    )
    assert_flags_agree(_SPMD_ENTRY_FLAGS, flags, context=where)
    # ONLY NOW may a bad value raise. Every process has entered AND LEFT the
    # same collective above, and every process sees the same sentinel in its
    # own payload, so this refusal is symmetric by construction -- unlike the
    # `int(n_steps)` that used to sit inside the payload build and could kill
    # one rank while its peers blocked (codex round-3, blocker 3).
    for label, problem in problems:
        raise ValueError(f"{where}: {label} {problem}")
    # A polar mask present without its v-face twin would blow up mid-build on
    # every rank; refuse here, after the gate, so the message is the same
    # everywhere.
    if (getattr(model, "_polar_mask", None) is not None
            and getattr(model, "_polar_mask_v", None) is None):
        raise ValueError(
            f"{where}: model._polar_mask is set but model._polar_mask_v is "
            f"None; the band/tile geometry build slices BOTH, so this would "
            f"fail mid-build (after the entry gate, before the geometry "
            f"collective) instead of here.")


def _refuse_unsupported_spmd_config(model) -> None:
    """Dispatch-hardening shared by the step + segment factories: only a
    non-fold lat-lon grid with an SPMD-safe mass path is supported.  Fail
    LOUD rather than silently mis-fold / band-local-sum.
    bytes, :func:`atm_latlon_geometry_bytes`).  The body consumes the SAME
    band values either way, so the step is bit-identical (gated by
    ``tests/parallel/test_atm_latlon_segment.py``).
    """
    # FIRST statement: agree every rank-local input before ANY
    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
    _agree_spmd_entry(model, mesh, n_steps=None,
                      has_physics_fn=physics_fn is not None,
                      shard_geometry=shard_geometry,
                      where="make_sharded_atm_latlon_step")
    from legoesm.parallel.latlon_spmd import latlon_band_perms
    from legoesm.parallel.shard_map_compat import shard_map

    # Stochastic physics is SPMD-safe since increment 2: the Bechtold AR1
    # innovation folds the per-step sub-key with each column's GLOBAL id
    # (``PhysicsState.col_index`` — band-split with the carry, so every
    # shard holds its own global ids) and the replicated master key splits
    # identically on every band — the draw is decomposition-INVARIANT.
    # Deterministic prognostic carries (tke/qke, conv profiles, GWD
    # spectrum) thread exactly: the ColumnAdapter flatten is a C-order
    # (lat-major) reshape, so a contiguous dim-0 shard of every
    # ``(ncol, ...)`` PhysicsState leaf IS the band's own columns.

    # A stateful (tagged) physics_fn with NO carry would silently reseed
    # its PhysicsState every step (issue #405/#413) — model.step()'s
    # guard is bypassed here, so the returned step re-checks per call.
    from legoesm.timestepping.integration import (
        refuse_unthreaded_stateful_physics)

    if mesh is None:                       # single-device: plain C-grid step
        def _serial_step(c_state, dt, phys_state=None):
            refuse_unthreaded_stateful_physics(
                physics_fn, phys_state, where="atm lat-band SPMD step")
            out, ps_out = model._step_cgrid(
                c_state, dt, physics_fn=physics_fn, phys_state=phys_state)
            return out if phys_state is None else (out, ps_out)
        return _serial_step

    n_dev = mesh.devices.size
    axis = mesh.axis_names[0]
    grid = model.grid

    _refuse_unsupported_spmd_config(model)

    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
        model, mesh, n_dev, shard_geometry)

    perm_north, _perm_south = latlon_band_perms(n_dev)
    band_step = _make_band_step_body(
        model, template, array_field_names, axis, perm_north, physics_fn,
        shard_geometry)

    def _body(state_local, stacks_local, dt):
        out, _ = band_step(state_local, stacks_local, dt, None)
        return out

    def _body_with_carry(state_local, stacks_local, dt, ps_local):
        # Stateful variant: the band's PhysicsState chunk (ncol_band =
        # nl*n_lon leading dim — the C-order lat-major flatten makes a
        # contiguous dim-0 shard exactly the band's own columns) is
        # threaded into every RK stage and the carry-out returned.
        return band_step(state_local, stacks_local, dt, ps_local)

    _ncol_global = int(grid.n_lat) * int(grid.n_lon)

    def _ps_spec_leaf(leaf):
        # (ncol, ...) leaves band-split on dim 0 (lat-major flatten);
        # everything else (prng_key (2,), scalars) replicated.
        if (hasattr(leaf, "ndim") and leaf.ndim >= 1
                and leaf.shape[0] == _ncol_global):
            return P("lat")
        return P()

    # Build the JITTED shard_map ONCE and cache it. ``jax.jit`` is LOAD-BEARING:
    # a bare shard_map is NOT compilation-cached, so calling it re-traces +
    # recompiles the (large, un-jitted) band step EVERY call — a 32x64x10 nd=2
    # step took ~142 s/step (bench 8560671), and the whole equivalence gate ran
    # ~65 min. Wrapping in jit caches the compile: probe 8561202 measured
    # [3079, 1.4, 1.2, 1.1, 1.1] ms — first call compiles, the rest hit the
    # cache. ``dt`` is a TRACED operand (not a closure constant) so a changing dt
    # does not retrigger compilation. The grid-tracer concern that kept
            "follow-up.)  Use p_lon == 1 or disable the filter.")


def make_sharded_atm_latlon_step_2d(model, mesh, physics_fn=None, *,
                                    shard_geometry: bool = True):
    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic
    atm step 2-D-tile-SPMD over a ``("lat", "lon")`` mesh — the M3a native
    2-D tiling twin of :func:`make_sharded_atm_latlon_step`.

    ``c_state`` is a ``CGridLatLonHydrostaticState`` laid out with
    :func:`shard_state_atm_latlon_2d` (``v`` as ``v_lower``, ``u`` as
    ``u_left``).  The body reconstructs each tile's staggered faces (v via
    the lat ppermute, u via the periodic lon ring), runs the UN-jitted
    ``model._step_cgrid_impl`` on the tile geometry + per-tile pole masks,
    and converts both staggers back.  Halos: the armed 2-D SPMD backend
    routes ``pad_halo_latlon*`` through ``make_latlon_2d_pad_body`` (lat
    ppermute + lon ring + the EXACT serial 180-deg pole fold via a lon-ring
    all_gather at the pole tiles), ``pad_with_pole_bc_lat`` through the
    lat-only wall body, and ``pad_lon_cgrid`` through the lon ring — all
    shared machinery, no operator numerics duplicated.  Global reductions
    (the mass fixer's ``batch_global_area_sums``) psum over BOTH mesh axes.

    A degenerate ``(N, 1)`` mesh is bit-identical to the 1-D band step
    (every lon-ring op takes its static local branch; gated by
    ``tests/parallel/test_atm_latlon_2d_tiling.py``).  The 1-D band factory
    remains the default production lane.

    ``physics_fn``: STATELESS column-local closures only (Held-Suarez etc.),
    evaluated per RK stage on the TILE geometry — decomposition-invariant
    with no collectives.  A stateful ``PhysicsState`` carry is REFUSED: its
    ``(ncol, ...)`` leaves flatten lat-major over the GLOBAL grid, so a
    contiguous dim-0 shard is a lat BAND's columns, not a 2-D tile's —
    thread carries through the 1-D :func:`make_sharded_atm_latlon_step`.

    ``shard_geometry=True`` (default — new API, no historical layout):
    per-device tile geometry slices (``P("lat", "lon")`` stacks);
    ``False`` replicates the all-tile stacks (indexed at the axis indices).
    Same tile values either way (bit-identical numerics).
    """
    # FIRST statement: agree every rank-local input before ANY
    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
    _agree_spmd_entry(model, mesh, n_steps=None,
                      has_physics_fn=physics_fn is not None,
                      shard_geometry=shard_geometry,
                      where="make_sharded_atm_latlon_step_2d")
    from legoesm.parallel.latlon_spmd import latlon_band_perms
    from legoesm.parallel.shard_map_compat import shard_map
    from legoesm.timestepping.integration import (
        refuse_unthreaded_stateful_physics)

    if mesh is None:                       # single-device: plain C-grid step
        return make_sharded_atm_latlon_step(model, None,
                                            physics_fn=physics_fn)

    p_lat, p_lon = _check_2d_mesh(mesh)
    _refuse_unsupported_spmd_config_2d(model, p_lon)

    template, array_field_names, stacks, stacks_spec = (
        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
    perm_north, _perm_south = latlon_band_perms(p_lat)
    tile_step = _make_tile_step_body_2d(
        model, template, array_field_names, perm_north, p_lon, physics_fn,
        shard_geometry)

    def _body(state_local, stacks_local, dt):
        out, _ = tile_step(state_local, stacks_local, dt, None)
        return out

    _cache = {}

    def sharded_step(c_state, dt, phys_state=None):
        _agree_spmd_call(mesh, c_state, phys_state,
                         where="make_sharded_atm_latlon_step_2d.step")
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="atm lat-lon 2-D SPMD step")
        if phys_state is not None:
            raise NotImplementedError(
                "make_sharded_atm_latlon_step_2d: a stateful PhysicsState "
                "carry is not 2-D-tile-routed (its (ncol, ...) leaves "
                "flatten lat-major over the GLOBAL grid — a contiguous "
                "dim-0 shard is a lat band, not a 2-D tile).  Thread the "

exec
/usr/bin/bash -lc "sed -n '1,260p' .physics-validator/scaling_campaign/codex_atm_port_r22_2026-08-03.md; rg -n \"def test.*2d|shard_geometry\" tests/parallel/test_atm_latlon_2d_tiling.py; sed -n '1550,1670p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; sed -n '990,1105p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 233ms:
Reading additional input from stdin...
OpenAI Codex v0.145.0
--------
workdir: /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
model: gpt-5.6-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019fc696-92f6-72a2-aac7-dd28d2bec64c
--------
user
Round-22: review the atm-lane port (git diff HEAD -- packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py). Both geometry-stack sites (1-D + 2-D) swap broadcast_checked + device_put for the shared checked_shard_put (per-leading-slab gate + make_array_from_callback; single-process plain device_put), covering sharded AND replicated (P()) spec_of modes; n_bands=int(raw[name].shape[0]) per field (schema gate precedes both sites). jnp.asarray wrap dropped (helper returns jax Arrays). Receipts: 54 atm parity tests (4 virtual devices) + 2-proc multicontroller repros both modes (step 6.75 ms, segment 8.57 ms incl. the sharded-geometry default). Attack: (1) replicated-mode semantics change — devices now hold each process's OWN bytes (±gate) instead of process-0's; any consumer requiring bit-identical replicated geometry across processes? (2) n_bands from shape[0] on the polar-mask entries with [:, None] shapes at the 2-D site — leading axis meaning consistent? (3) broadcast_checked now unused in this file — import removed? any other caller left? (4) skipped-by-default tests now needing the virtual-device env documented? End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll inspect the targeted diff and its nearby geometry/schema paths, then trace the helper and tests for the four attack points.
exec
/usr/bin/bash -lc "git diff HEAD -- packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py && rg -n \"broadcast_checked|checked_shard_put|geometry|polar|n_bands|virtual|XLA_FLAGS\" packages/atmosphere/legoesm/atmosphere/dynamics/gcm -g '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 456ms:
diff --git a/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py b/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
index 4371fe976..f7da20d04 100644
--- a/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
+++ b/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
@@ -33,7 +33,7 @@ import jax.numpy as jnp
 from jax.sharding import NamedSharding, PartitionSpec as P
 
 from legoesm.parallel.geometry_consistency import (
-    FLAG_ABSENT, assert_flags_agree, assert_schema_agrees, broadcast_checked,
+    FLAG_ABSENT, assert_flags_agree, assert_schema_agrees, checked_shard_put,
     coerce_bool, coerce_count, config_digest48, name_digest48,
     tree_schema_digest48)
 
@@ -488,15 +488,20 @@ def _build_geometry_stacks(model, mesh, n_dev: int, shard_geometry: bool):
     assert_schema_agrees(ordered_names, n_dev,
                          context="make_sharded_atm_latlon_step",
                          arrays=[raw[n] for n in ordered_names])
-    raw = {
-        name: jnp.asarray(broadcast_checked(
-            raw[name], name, context="make_sharded_atm_latlon_step"))
-        for name in ordered_names
-    }
+    # checked_shard_put replaces broadcast_checked + device_put (the ocean
+    # walls, PR #1457): the broadcast's psum program is [n_processes,
+    # stack] and a numpy device_put onto an all-process sharding pays
+    # jax's whole-array assert_equal. Both are SOFT walls here (2-D
+    # geometry, ~MB-scale fields) but the shared helper removes them
+    # uniformly; the per-leading-slab gate keeps the divergence contract
+    # (schema gate above agreed shapes, so widths match).
     spec_of = lat_spec if shard_geometry else (lambda _arr: P())
     stacks = {
-        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
-        for name, arr in raw.items()
+        name: checked_shard_put(
+            raw[name], name, NamedSharding(mesh, spec_of(raw[name])),
+            context="make_sharded_atm_latlon_step",
+            n_bands=int(raw[name].shape[0]))
+        for name in ordered_names
     }
     stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
     return template, array_field_names, stacks, stacks_spec
@@ -1582,17 +1587,16 @@ def _build_geometry_stacks_2d(model, mesh, p_lat: int, p_lon: int,
     assert_schema_agrees(ordered_names, p_lat * p_lon,
                          context="make_sharded_atm_latlon_step_2d",
                          arrays=[raw[n] for n in ordered_names])
-    raw = {
-        name: jnp.asarray(broadcast_checked(
-            raw[name], name, context="make_sharded_atm_latlon_step_2d"))
-        for name in ordered_names
-    }
+    # 2-D twin of the checked_shard_put swap above (see that note).
     spec_of = tile_spec if shard_geometry else (lambda _arr: P())
     stacks = {
-        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
-        for name, arr in raw.items()
+        name: checked_shard_put(
+            raw[name], name, NamedSharding(mesh, spec_of(raw[name])),
+            context="make_sharded_atm_latlon_step_2d",
+            n_bands=int(raw[name].shape[0]))
+        for name in ordered_names
     }
-    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
+    stacks_spec = {name: spec_of(raw[name]) for name in ordered_names}
     return template, array_field_names, stacks, stacks_spec
 
 
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:9:``_spmd_lat_psum_or_none`` for global reductions, ``latlon_mpi`` band-geometry
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:10:slicers) and writes NEW only the atm-specific state layout + geometry-band glue.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:35:from legoesm.parallel.geometry_consistency import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:36:    FLAG_ABSENT, assert_flags_agree, assert_schema_agrees, checked_shard_put,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:407:def atm_latlon_geometry_bytes(grid, n_devices: int) -> dict:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:408:    """Per-device geometry residency of the lat-band SPMD step, computed from
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:415:    ``sharded_per_device_bytes``: the ``shard_geometry=True`` layout — each
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:418:    uniform).  Excludes the optional polar-filter mask stacks (two
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:429:        "n_geometry_fields": len(names),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:435:def _build_geometry_stacks(model, mesh, n_dev: int, shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:437:    polar-filter masks) over a leading band axis and lay them out on ``mesh``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:439:    ``shard_geometry=False`` (the historical layout): every stack is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:441:    geometry and the body indexes its own band at ``axis_index``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:443:    ``shard_geometry=True`` (M2b): every stack is sharded ``P("lat", ...)``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:448:    (per-device geometry bytes drop by ``n_dev`` —
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:449:    :func:`atm_latlon_geometry_bytes`).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:462:    # Per-band polar-filter masks (only when the filter is on): slice the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:465:    # self._polar_mask (also None), no filter.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:467:    if model._polar_mask is not None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:468:        raw["__polar_mask"] = jnp.stack(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:469:            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(n_dev)],
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:471:        raw["__polar_mask_v"] = jnp.stack(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:472:            [model._polar_mask_v[r * nl:r * nl + nl + 1] for r in range(n_dev)],
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:474:    # #1362: the band geometry above is RECOMPUTED per process from the same
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:480:    # blind: a REAL divergence (different polar masks = different filtering =
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:483:    # The schema gate matters more here than in the ocean lane: the polar-mask
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:484:    # entries are CONDITIONAL on ``model._polar_mask``, so a per-process
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:491:    # checked_shard_put replaces broadcast_checked + device_put (the ocean
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:495:    # geometry, ~MB-scale fields) but the shared helper removes them
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:498:    spec_of = lat_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:500:        name: checked_shard_put(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:503:            n_bands=int(raw[name].shape[0]))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:511:                         perm_north, physics_fn, shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:519:    layout.  ``shard_geometry`` selects the geometry index (STATIC Python
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:528:        gi = 0 if shard_geometry else jax.lax.axis_index(axis)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:531:        pmask = (stacks_local["__polar_mask"][gi]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:532:                 if "__polar_mask" in stacks_local else None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:533:        pmaskv = (stacks_local["__polar_mask_v"][gi]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:534:                  if "__polar_mask_v" in stacks_local else None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:543:            polar_mask=pmask, polar_mask_v=pmaskv,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:613:    "has_polar_mask", "has_polar_mask_v",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:615:    "has_physics_fn", "has_on_segment", "has_phys_state", "shard_geometry",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:622:                      shard_geometry=None, where: str) -> None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:647:      rejected the swapped peer while the valid one walked into the geometry
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:664:    * ``shard_geometry`` -- changes the geometry stacks' ``PartitionSpec``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:711:        # and `_build_geometry_stacks` broadcasts; agreeing their dtypes and
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:722:        # Rank-local CACHE presence. `_build_geometry_stacks` conditions on
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:723:        # `_polar_mask` and then slices `_polar_mask_v` unconditionally, so a
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:727:        float(getattr(model, "_polar_mask", None) is not None),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:728:        float(getattr(model, "_polar_mask_v", None) is not None),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:735:        _flag(shard_geometry, "shard_geometry"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:745:    # A polar mask present without its v-face twin would blow up mid-build on
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:748:    if (getattr(model, "_polar_mask", None) is not None
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:749:            and getattr(model, "_polar_mask_v", None) is None):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:751:            f"{where}: model._polar_mask is set but model._polar_mask_v is "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:752:            f"None; the band/tile geometry build slices BOTH, so this would "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:753:            f"fail mid-build (after the entry gate, before the geometry "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:860:                                 shard_geometry: bool = False):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:867:    UN-jitted ``model._step_cgrid_impl`` on the band geometry + band polar masks
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:871:    atm-NEW is only the 6-field state/geometry walk. ``check_vma=False`` (the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:877:    RK stage on the BAND geometry (``_step_cgrid_impl`` routes the band grid into
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:896:    ``shard_geometry`` (M2b, default ``False`` = historical layout,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:897:    byte-identical): ``True`` lays the band-geometry stacks out SHARDED
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:899:    geometry slice instead of a replicated all-band copy (1/n_dev the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:900:    bytes, :func:`atm_latlon_geometry_bytes`).  The body consumes the SAME
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:908:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:944:    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:945:        model, mesh, n_dev, shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:950:        shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1027:                                    shard_geometry: bool = True):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1047:    ``shard_geometry=True`` (default — a NEW API, no historical layout to
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1048:    preserve): each device holds ONLY its own band's geometry slice
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1050:    bit-identical numerics, 1/n_dev the geometry bytes
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1051:    (:func:`atm_latlon_geometry_bytes`).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1060:    C-grid step with the model's own geometry, same ``(state, all_finite)``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1081:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1128:    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1129:        model, mesh, n_dev, shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1133:        shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1190:    the band geometry; decomposition-invariant with no collectives.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1291:                                      band-SHARDED geometry) — one host
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1537:def _build_geometry_stacks_2d(model, mesh, p_lat: int, p_lon: int,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1538:                              shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1540:    polar-filter masks at ``p_lon == 1``) over LEADING ``(p_lat, p_lon)``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1542:    :func:`_build_geometry_stacks`.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1544:    ``shard_geometry=True``: stacks are sharded ``P("lat", "lon", ...)`` on
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1547:    ``shard_geometry=False``: replicated (``P()``) stacks, indexed at
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1564:    # Per-tile polar-filter masks: p_lon > 1 is refused by the factories
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1568:    if model._polar_mask is not None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1571:                "atm 2-D SPMD tiling: use_polar_filter=True with p_lon > 1 "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1572:                "is not wired — the polar filter FFTs the full longitude "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1575:        raw["__polar_mask"] = jnp.stack(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1576:            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(p_lat)],
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1578:        raw["__polar_mask_v"] = jnp.stack(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1579:            [model._polar_mask_v[r * nl:r * nl + nl + 1]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1582:    # #1362, 2-D twin of the guard in _build_geometry_stacks -- same
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1590:    # 2-D twin of the checked_shard_put swap above (see that note).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1591:    spec_of = tile_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1593:        name: checked_shard_put(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1596:            n_bands=int(raw[name].shape[0]))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1605:                            shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1614:    band/tile step on the tile geometry, convert both staggers back.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1621:        if shard_geometry:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1629:        pmask = (stacks_local["__polar_mask"][gi, gj]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1630:                 if "__polar_mask" in stacks_local else None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1631:        pmaskv = (stacks_local["__polar_mask_v"][gi, gj]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1632:                  if "__polar_mask_v" in stacks_local else None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1643:            polar_mask=pmask, polar_mask_v=pmaskv,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1668:    :func:`_agree_spmd_entry` having already agreed ``use_polar_filter`` (and
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1671:    `use_polar_filter` ALSO controls whether the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1672:    `__polar_mask`/`__polar_mask_v` entries exist in the geometry field list,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1677:    polar = bool(getattr(model.config, "use_polar_filter", False))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1678:    if p_lon > 1 and polar:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1680:            "atm 2-D SPMD tiling: use_polar_filter=True with p_lon > 1 is "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1681:            "not wired — the polar filter FFTs the full longitude circle "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1689:                                    shard_geometry: bool = True):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1698:    ``model._step_cgrid_impl`` on the tile geometry + per-tile pole masks,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1713:    evaluated per RK stage on the TILE geometry — decomposition-invariant
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1719:    ``shard_geometry=True`` (default — new API, no historical layout):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1720:    per-device tile geometry slices (``P("lat", "lon")`` stacks);
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1728:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1743:        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1747:        shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1784:                                       shard_geometry: bool = True):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1797:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1825:        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1829:        shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:49:from legoesm.grids.polar_filter import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:50:    compute_polar_filter_mask,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:89:    use_polar_filter: bool = False
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:90:    polar_filter_cutoff_deg: float = 60.0
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:91:    polar_filter_max_wave_speed: float = 300.0
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:403:    # inert) form, so routing it would change the north-fold value on a tripolar grid.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:498:        # Precompute polar filter masks (cached, not traced): one for
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:506:        if self.config.use_polar_filter:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:507:            self._polar_mask = compute_polar_filter_mask(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:509:                max_wave_speed=self.config.polar_filter_max_wave_speed,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:510:                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:512:            self._polar_mask_v = compute_polar_filter_mask(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:514:                max_wave_speed=self.config.polar_filter_max_wave_speed,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:515:                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:519:            self._polar_mask = None
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:520:            self._polar_mask_v = None
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:603:            if self._polar_mask is not None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:604:                dh = fourier_filter(dh, self.grid, self._polar_mask)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:605:                du_interior = fourier_filter(du[:, :-1], self.grid, self._polar_mask)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:608:                # mirrors the PE dycore's polar_mask_v treatment.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:609:                dv = fourier_filter(dv, self.grid, self._polar_mask_v)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_spectral.py:35:rotation about the polar axis (``u = Omega a cos(lat)``, ``v = 0``) each
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:21:  equations in spherical geometry. J. Comput. Phys., 102, 211-224.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:12:(``jnp.roll``), polar walls (``v = 0`` at the pole rows) and a domain-total mass
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:99:    The face-source weights live in ``nest`` (built once at construction, geometry
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:166:    them from the relaxation weights so the band geometry stays single-sourced.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:324:    it with the FULL standalone :class:`CGridLatLonShallowWaterModel` (polar
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:358:    (geometry / static config), so callers that JIT should close over them via
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:392:    # --- Parent: standalone global SW update (periodic lon, polar walls). ---
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_nesting.py:393:    # NOTE: this in-built parent step uses the BARE tendencies (no polar filter);
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:883:            # compute policy with x64 enabled, the f64 mesh-geometry
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:581:    # grid amplifies polar noise by 1/cos².  Instead we keep KE·cos²φ and
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:65:from legoesm.grids.polar_filter import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:66:    compute_polar_filter_mask,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:154:    use_polar_filter: bool = False
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:155:    polar_filter_cutoff_deg: float = 60.0
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:156:    polar_filter_max_wave_speed: float = 300.0
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:843:        # Precompute polar filter masks: one for cell-centered fields
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:851:        if self.config.use_polar_filter:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:852:            self._polar_mask = compute_polar_filter_mask(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:854:                max_wave_speed=self.config.polar_filter_max_wave_speed,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:855:                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:857:            self._polar_mask_v = compute_polar_filter_mask(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:859:                max_wave_speed=self.config.polar_filter_max_wave_speed,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:860:                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:864:            self._polar_mask = None
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:865:            self._polar_mask_v = None
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:930:        SPMD body passes the BAND geometry so a lat-dependent physics (e.g.
166:def test_2d_pad_body_matches_serial_window(p_lat, p_lon, halo, negate):
354:        model, mesh, p_lat, p_lon, shard_geometry=True)
382:def test_2d_tendency_matches_serial():
456:def test_2d_tendency_realistic_amplitude_cut_gate():
567:def test_2d_step_matches_serial_with_seam_checks():
620:def test_2d_state_layout_roundtrip():
633:def test_2d_step_with_held_suarez_matches_serial():
667:def test_2d_degenerate_41_bitmatches_1d_band():
671:    shard_geometry=True (the geometry VALUES are identical either way, gated
678:    step1d = make_sharded_atm_latlon_step(model, mesh1, shard_geometry=True)
685:                                             shard_geometry=True)
764:def test_2d_segment_matches_sequential_and_serial():
795:def test_2d_segment_finite_scalar_detects_nan():
815:def test_2d_segment_f32_ic_dtype_fixed_point():
920:def test_chooser_2d_when_band_infeasible():
954:def test_2d_factory_refusals():
    Returns ``(template, array_field_names, stacks, stacks_spec)``.
    """
    grid = model.grid
    tile_grids = build_tile_grids_atm_2d(grid, p_lat, p_lon)
    template = tile_grids[0][0]
    array_field_names = atm_grid_array_field_names(template)
    raw = {
        name: jnp.stack([
            jnp.stack([jnp.asarray(getattr(tile_grids[r][c], name))
                       for c in range(p_lon)], axis=0)
            for r in range(p_lat)
        ], axis=0)
        for name in array_field_names
    }
    # Per-tile polar-filter masks: p_lon > 1 is refused by the factories
    # (the filter rfft's the full lon circle); at p_lon == 1 the stacks
    # mirror the band layout with a singleton lon-tile axis.
    nl = int(grid.n_lat) // p_lat
    if model._polar_mask is not None:
        if p_lon > 1:
            raise NotImplementedError(
                "atm 2-D SPMD tiling: use_polar_filter=True with p_lon > 1 "
                "is not wired — the polar filter FFTs the full longitude "
                "circle (needs a lon-gather FFT).  Use p_lon == 1 or "
                "disable the filter.")
        raw["__polar_mask"] = jnp.stack(
            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(p_lat)],
            axis=0)[:, None]
        raw["__polar_mask_v"] = jnp.stack(
            [model._polar_mask_v[r * nl:r * nl + nl + 1]
             for r in range(p_lat)],
            axis=0)[:, None]
    # #1362, 2-D twin of the guard in _build_geometry_stacks -- same
    # per-process recompute, same replicated-device_put bit-identity assert.
    # n_dev is the FULL device count here (p_lat * p_lon): the schema
    # fingerprint must describe this process's whole mesh, not one axis.
    ordered_names = list(raw)
    assert_schema_agrees(ordered_names, p_lat * p_lon,
                         context="make_sharded_atm_latlon_step_2d",
                         arrays=[raw[n] for n in ordered_names])
    # 2-D twin of the checked_shard_put swap above (see that note).
    spec_of = tile_spec if shard_geometry else (lambda _arr: P())
    stacks = {
        name: checked_shard_put(
            raw[name], name, NamedSharding(mesh, spec_of(raw[name])),
            context="make_sharded_atm_latlon_step_2d",
            n_bands=int(raw[name].shape[0]))
        for name in ordered_names
    }
    stacks_spec = {name: spec_of(raw[name]) for name in ordered_names}
    return template, array_field_names, stacks, stacks_spec


def _make_tile_step_body_2d(model, template, array_field_names,
                            perm_north, p_lon: int, physics_fn,
                            shard_geometry: bool):
    """One tile's un-jitted C-grid step body — the 2-D twin of
    :func:`_make_band_step_body`, shared by the per-step and segment 2-D
    factories so the tile numerics are written ONCE.

    Returns ``tile_step(state_local, stacks_local, dt, ps_local) ->
    (state_out_local, ps_out)`` operating on the tile-local
    ``(u_left, v_lower)`` layout: reconstruct the tile's ``nl+1`` v-faces
    (lat ppermute) AND ``w+1`` u-faces (lon ring ppermute), run the un-jitted
    band/tile step on the tile geometry, convert both staggers back.
    """
    from legoesm.parallel.latlon_spmd import (
        reconstruct_uface_left, reconstruct_vface_lower,
        spmd_pole_end_masks, to_uface_left, to_vface_lower)

    def tile_step(state_local, stacks_local, dt, ps_local):
        if shard_geometry:
            gi, gj = 0, 0
        else:
            gi = jax.lax.axis_index("lat")
            gj = jax.lax.axis_index("lon")
        tile_geom = template._replace(
            **{name: stacks_local[name][gi, gj]
               for name in array_field_names})
        pmask = (stacks_local["__polar_mask"][gi, gj]
                 if "__polar_mask" in stacks_local else None)
        pmaskv = (stacks_local["__polar_mask_v"][gi, gj]
                  if "__polar_mask_v" in stacks_local else None)
        # Reconstruct the tile's nl+1 v-faces (shared interface row via the
        # lat ppermute) and w+1 u-faces (periodic seam column via the lon
        # ring), run the un-jitted step, convert both staggers back.
        v_full = reconstruct_vface_lower(state_local.v, "lat", perm_north)
        u_full = reconstruct_uface_left(state_local.u, "lon", p_lon)
        state_tile = state_local._replace(u=u_full, v=v_full)
        out, ps_out = model._step_cgrid_impl(
            state_tile, dt,
            physics_fn=physics_fn, phys_state=ps_local,
            grid=tile_geom, sigma_coord=model.sigma_coord,
            polar_mask=pmask, polar_mask_v=pmaskv,
            pole_v_bc_masks=spmd_pole_end_masks(),
        )
        return (out._replace(u=to_uface_left(out.u),
                             v=to_vface_lower(out.v)), ps_out)

    return tile_step


def _check_2d_mesh(mesh) -> tuple[int, int]:
    """Validate the 2-D tile mesh axes and return ``(p_lat, p_lon)``."""
    names = tuple(mesh.axis_names)
    if names != ("lat", "lon"):
        raise ValueError(
            f"atm 2-D SPMD tiling: mesh axes must be ('lat', 'lon'); got "
            f"{names}.  Build it as Mesh(devices.reshape(p_lat, p_lon), "
            f"axis_names=('lat', 'lon')) — choose_latlon_2d_topology picks "
            f"(p_lat, p_lon).")
    return int(mesh.shape["lat"]), int(mesh.shape["lon"])


def _refuse_unsupported_spmd_config_2d(model, p_lon: int) -> None:
    """2-D-specific dispatch-hardening on top of the shared band refusals.

    Like the 1-D twin this performs NO agreement itself; it relies on
    :func:`_agree_spmd_entry` having already agreed ``use_polar_filter`` (and
    the mesh's axis ORDER, which decides ``p_lon``) at the entry point.  That
    dependency is the sharpest instance of codex-2026-07-29 blocker 1:
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="atm lat-band SPMD step")
        # Cache key = (state pytree STRUCTURE, phys_state pytree structure):
        # in_specs/out_specs derive from both, so a structure change (optional
        # field None <-> Field, or the phys carry appearing/disappearing) must
        # rebuild the shard_map rather than reuse stale specs (codex finding,
        # ocean-twin parity). ``phys_state`` threading is the AIMIP-branch
        # feature main lacks (main rejects a non-None carry here).
        key = (jax.tree.structure(c_state),
               None if phys_state is None
               else jax.tree_util.tree_structure(phys_state))
        fn = _cache.get(key)
        if fn is None:
            in_spec = jax.tree.map(lat_spec, c_state)
            if phys_state is None:
                fn = jax.jit(shard_map(
                    _body, mesh=mesh, in_specs=(in_spec, stacks_spec, P()),
                    out_specs=in_spec, check_vma=False))
            else:
                ps_spec = jax.tree.map(_ps_spec_leaf, phys_state)
                fn = jax.jit(shard_map(
                    _body_with_carry, mesh=mesh,
                    in_specs=(in_spec, stacks_spec, P(), ps_spec),
                    out_specs=(in_spec, ps_spec), check_vma=False))
            _cache[key] = fn
        # Arm the SPMD band halo around the call ONLY (see _latlon_spmd_armed).
        with _latlon_spmd_armed(mesh):
            if phys_state is None:
                return fn(c_state, stacks, jnp.asarray(dt))
            return fn(c_state, stacks, jnp.asarray(dt), phys_state)

    sharded_step._geom_stacks = stacks   # test/introspection only
    return sharded_step


def make_sharded_atm_latlon_segment(model, mesh, n_steps: int,
                                    physics_fn=None, *,
                                    shard_geometry: bool = True):
    """Return ``segment(c_state, dt) -> (c_state, all_finite)`` advancing
    ``n_steps`` C-grid steps in ONE compiled program — a ``lax.scan`` of the
    band step inside a single jitted ``shard_map``, built once and reused
    (the M2b lever: "compile atmosphere lat-lon segments instead of
    launching one step at a time").

    Contrast with driving :func:`make_sharded_atm_latlon_step` in a Python
    loop: ONE host dispatch (+ halo-backend arm/restore + cache-key hash) per
    SEGMENT instead of per STEP, and no per-step host round-trip between
    device launches.  The scanned band body is the SAME
    ``_make_band_step_body`` the per-step path runs, so the trajectory
    matches the sequential sharded steps to compilation-order roundoff
    (gated at 1e-12 by ``tests/parallel/test_atm_latlon_segment.py``).

    ``all_finite`` is a REPLICATED traced scalar bool from
    :func:`state_finite_scalar` — the in-graph blowup guard (``psum`` of
    per-band non-finite presence over ALL state leaves).  The host reads
    this ONE scalar per segment instead of gathering the full state.

    ``shard_geometry=True`` (default — a NEW API, no historical layout to
    preserve): each device holds ONLY its own band's geometry slice
    (``P("lat")`` stacks) instead of a replicated all-band copy —
    bit-identical numerics, 1/n_dev the geometry bytes
    (:func:`atm_latlon_geometry_bytes`).

    STATELESS physics only (``None`` / Held-Suarez / column-local closures,
    the production ``run_atm_latlon_spmd`` envelope): a stateful
    ``PhysicsState`` carry is refused loudly — thread it through the
    per-step :func:`make_sharded_atm_latlon_step` until the segment lane
    routes the carry through the scan.

    ``mesh=None``: the single-device twin — ``jit(lax.scan)`` over the serial
    C-grid step with the model's own geometry, same ``(state, all_finite)``
    contract.

    ``n_steps`` is STATIC (the compiled scan length): one compiled program
    per distinct segment length (``run_atm_latlon_spmd`` caches per length —
    at most two: the regular segment and the final remainder).

    Carry dtype: leading steps are UNROLLED outside the ``lax.scan`` until
    the state's dtype signature is a fixed point of the step
    (:func:`unroll_to_dtype_fixed_point` — ``jax.eval_shape`` probe, zero
    FLOPs, trace-time constant).  A mixed-precision IC promotes over the
    first stepS (``p_s`` first, ``u/v/T`` next via the promoted ``p_s`` —
    observed jobs 8916406/8916740) exactly as the per-step Python loop
    absorbs silently; an already-stable state unrolls NOTHING and scans all
    ``n_steps``.  Zero extra casts, zero numerical difference vs the
    per-step lane — never a silent precision change.
    """
    # FIRST statement: agree every rank-local input before ANY
    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
    _agree_spmd_entry(model, mesh, n_steps=n_steps,
                      has_physics_fn=physics_fn is not None,
                      shard_geometry=shard_geometry,
                      where="make_sharded_atm_latlon_segment")
    from legoesm.parallel.latlon_spmd import latlon_band_perms
    from legoesm.parallel.shard_map_compat import shard_map
    from legoesm.timestepping.integration import (
        refuse_unthreaded_stateful_physics)

    if int(n_steps) < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    n_steps = int(n_steps)

    def _refuse_carry(phys_state):
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="atm lat-lon compiled segment")
        if phys_state is not None:
            raise NotImplementedError(
                "make_sharded_atm_latlon_segment: a stateful PhysicsState "
                "carry is not yet segment-routed — use the per-step "
                "make_sharded_atm_latlon_step(phys_state=...) path.")

    if mesh is None:                       # single-device compiled segment
        def _serial_seg(c_state, dt):
            def _step1(s):
                out, _ps = model._step_cgrid_impl(
                    s, dt, physics_fn=physics_fn, phys_state=None)

exec
/usr/bin/bash -lc "python - <<'PY'
import inspect
import jax
print(jax.__version__)
print(inspect.getsource(jax.make_array_from_callback))
PY
rg -n \"def make_array_from_callback|make_array_from_callback\" "'$(python - <<'"'PY'
import os, jax
print(os.path.dirname(jax.__file__))
PY
) -g '*.py' | head -40" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1041ms:
/usr/bin/bash: cannot create temp file for here-document: Read-only file system
/usr/bin/bash: line 6: cannot create temp file for here-document: Read-only file system
scripts/bench/bench_mpas_spmd_scaling.py:149:    # jax.make_array_from_callback (only THIS process's shard rows are
scripts/bench/bench_atm_latlon_spmd_scaling.py:278:        # created via make_array_from_callback for the rows this process's
tests/test_cases/baroclinic_wave.py:781:    ``jax.make_array_from_callback``, whose callback runs only for the
tests/test_cases/baroclinic_wave.py:873:        return jax.make_array_from_callback(gshape, shard, cb)
tests/unit/test_parallel.py:595:    process; routes through make_array_from_callback under >1 process so the
packages/core/legoesm/parallel/mesh.py:759:    process's LOCAL copy via ``jax.make_array_from_callback`` instead —
packages/core/legoesm/parallel/mesh.py:774:        return jax.make_array_from_callback(
packages/core/legoesm/parallel/geometry_consistency.py:684:    on real divergence), then ``jax.make_array_from_callback`` hands each
packages/core/legoesm/parallel/geometry_consistency.py:703:    return jax.make_array_from_callback(
packages/core/legoesm/parallel/geometry_consistency.py:755:    return jax.make_array_from_callback(
packages/core/legoesm/parallel/latlon_spmd.py:309:    each, rc=137 OOM). Instead ``jax.make_array_from_callback`` invokes the
packages/core/legoesm/parallel/latlon_spmd.py:316:    NOT differentiable: ``make_array_from_callback`` is a host construction API,
packages/core/legoesm/parallel/latlon_spmd.py:332:    return jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx])
tests/unit/test_geometry_consistency.py:818:        "band-LOCAL construction via jax.make_array_from_callback (#1100): "
tests/parallel/test_atm_latlon_bandlocal_build.py:11:(``make_array_from_callback`` per-process rows) leg of #1100 needs a real
scripts/validate/validate_tiled_fv3_sw_multinode.py:114:        return jax.make_array_from_callback(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:167:    ``jax.make_array_from_callback``, whose callback is invoked only for the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:219:        return jax.make_array_from_callback(gshape, sharding, cb)

exec
/usr/bin/bash -lc "python -c 'import inspect,jax; print(jax.__version__); print(inspect.getsource(jax.make_array_from_callback))' && sed -n '130,235p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py && sed -n '735,790p' packages/core/legoesm/parallel/mesh.py && sed -n '285,340p' packages/core/legoesm/parallel/latlon_spmd.py && sed -n '790,900p' tests/unit/test_geometry_consistency.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 127 in 218ms:
/usr/bin/bash: python: command not found

exec
/usr/bin/bash -lc "python3 -c 'import inspect,jax; print(jax.__version__); print(inspect.getsource(jax.make_array_from_callback))' && sed -n '130,235p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py && sed -n '735,790p' packages/core/legoesm/parallel/mesh.py && sed -n '285,340p' packages/core/legoesm/parallel/latlon_spmd.py && sed -n '790,900p' tests/unit/test_geometry_consistency.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 226ms:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'jax'

exec
/usr/bin/bash -lc "sed -n '130,235p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; sed -n '735,790p' packages/core/legoesm/parallel/mesh.py; sed -n '285,340p' packages/core/legoesm/parallel/latlon_spmd.py; sed -n '790,900p' tests/unit/test_geometry_consistency.py; rg -n \"checked_shard_put\" tests packages -g '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 384ms:
    # FIRST statement: the SCATTER runs one placement per leaf, so its
    # SCHEDULE is rank-local data (a tracer dict differing in key set / order
    # / shape across processes). #1362 round 4, blockers 5-6.
    _agree_mesh_entry(mesh, state, where="shard_state_atm_latlon")
    from legoesm.parallel.latlon_spmd import shard_leaf

    _mp = jax.process_count() > 1

    def _put(arr):
        # shard_leaf: single-process -> device_put (byte-unchanged); multi-
        # controller -> per-process local band via make_array_from_process_local_data
        # (no all-gather, no transient global replica — issue #1100).
        return shard_leaf(arr, NamedSharding(mesh, lat_spec(arr)), multiprocess=_mp)

    n_lat = state.T.shape[0]
    v_lower = state.v[:n_lat]
    return state._replace(
        u=_put(state.u),
        v=_put(v_lower),
        T=_put(state.T),
        p_s=_put(state.p_s),
        phis=_put(state.phis),
        tracers={k: _put(val) for k, val in state.tracers.items()},
    )


def build_sharded_held_suarez_state_atm_latlon(
    grid, sigma_coord, mesh, *,
    T_init: float = 300.0,
    p_s_init: float | None = None,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
) -> CGridLatLonHydrostaticState:
    """Band-local Held-Suarez C-grid state, directly in the sharded layout.

    The #1100 invariant for multi-process runs: **neither global builds nor
    ``device_put`` replication** — every global-shaped leaf is created with
    ``jax.make_array_from_callback``, whose callback is invoked only for the
    row slices owned by THIS process's addressable devices (documented JAX
    semantics: per-addressable-shard callbacks with GLOBAL index slices).  No
    full global array is ever handed to ``device_put`` — the path measured to
    detonate under many-process packing in #1100 (its cross-process
    consistency check amplified even small replicated objects; an
    implementation behaviour we cite as measured, not as API contract).  This
    removes the per-process global-state BUILD that made the route-B lat-lon
    bench OOM under full-node CPU packing (``held_suarez_init_latlon`` +
    ``hydrostatic_to_cgrid`` materialised the full ``(n_lat, n_lon, nlev)``
    state on every process before sharding).

    Bit-identical to
    ``shard_state_atm_latlon(hydrostatic_to_cgrid(held_suarez_init_latlon(
    grid, sigma), grid), mesh)`` for the flat-terrain case — gated by
    ``tests/parallel/test_atm_latlon_bandlocal_build.py``.  The staggered
    ``v`` is created directly as its sharded ``v_lower`` layout (the dropped
    north pole-wall face is identically zero in this at-rest IC, exactly what
    ``gather_state_atm_latlon`` re-appends).

    One deliberate exception: the 2-D lowest-level temperature perturbation
    (``jax.random.normal`` over ``(n_lat, n_lon)``) is evaluated in full on
    every process — identical threefry streams cannot be row-sliced without
    evaluating the whole field, and at bench scale it is O(10 MB) vs the
    O(GB) 3-D leaves the callback path avoids.

    Flat terrain only (the bench IC): there is deliberately no ``phis``
    parameter — the topography variant of ``held_suarez_init_latlon`` would
    need its own band-local surface-pressure callback; extend explicitly
    rather than reuse this builder.
    """
    from legoesm import constants
    from legoesm.core.precision import get_policy

    _dtype = get_policy().storage
    if p_s_init is None:
        p_s_init = constants.p_ref

    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = sigma_coord.n_levels

    # 2-D seed field, exact expression of held_suarez_init_latlon
    key = jax.random.PRNGKey(seed)
    pert2d = jax.random.normal(key, (n_lat, n_lon), dtype=_dtype) * jnp.asarray(
        perturbation_amplitude, dtype=_dtype
    )

    def _slice_shape(gshape, idx):
        return tuple(len(range(*sl.indices(n))) for sl, n in zip(idx, gshape))

    def _make(gshape, cb):
        sharding = NamedSharding(mesh, P("lat", *((None,) * (len(gshape) - 1))))
        return jax.make_array_from_callback(gshape, sharding, cb)

    def _zeros_cb(gshape):
        return lambda idx: jnp.zeros(_slice_shape(gshape, idx), dtype=_dtype)

    def _T_cb(idx):
        shape = _slice_shape((n_lat, n_lon, nlev), idx)
        # EXACT expression of held_suarez_init_latlon (``ones * T_init`` then
        # ``.at[:, :, -1].add(pert)``) so promotion semantics match for
        # strongly-typed ``T_init`` too, not just Python floats.
        block = jnp.ones(shape, dtype=_dtype) * T_init
        return block.at[:, :, -1].add(pert2d[idx[0], idx[1]])

    def _ps_cb(idx):
        shape = _slice_shape((n_lat, n_lon), idx)
        # held_suarez_init_latlon with phis=None: p_s_init * exp(-0/(R_d T))
        phis_block = jnp.zeros(shape, dtype=_dtype)
    )
    _active_config = config
    logger.info(
        "legoESM: %d-device voronoi mesh on %s "
        "(nCells=%d, nEdges=%d, nVertices=%d)",
        n_dev, backend_name, nCells, nEdges, nVertices,
    )
    return config


# ==============================================================================
# Pytree sharding utilities
# ==============================================================================

def multiprocess_safe_device_put(leaf, sharding):
    """``jax.device_put`` that is safe under multi-controller SPMD.

    ``jax.device_put(x, sharding)`` with a sharding that spans processes
    ASSERTS the value is bit-identical on every process.  Per-process XLA
    autotuning can legitimately pick different kernels on different nodes,
    producing last-bit differences in host-precomputed inputs (first hit:
    the external-forcing leaves on the 2-node Levante cs_spmd receipt run,
    #693 job 26030677 — values identical to 8 significant digits, assert
    still trips).  Under >1 process, build the global array from each
    process's LOCAL copy via ``jax.make_array_from_callback`` instead —
    each process materializes only its addressable shards, no cross-process
    equality requirement, no communication.

    Single-process (and non-array leaves) delegate to plain
    ``jax.device_put`` — byte-identical behavior to before.
    Already-global (non-fully-addressable) leaves pass through unchanged.
    """
    if not isinstance(leaf, (jax.Array, jnp.ndarray)):
        return jax.device_put(leaf, sharding)
    if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
        return leaf  # already a global sharded array; nothing to place
    if jax.process_count() > 1:
        import numpy as np
        host = np.asarray(leaf)
        return jax.make_array_from_callback(
            host.shape, sharding, lambda idx: host[idx])
    return jax.device_put(leaf, sharding)


def shard_pytree(pytree, config: DeviceConfig):
    """Shard a pytree across devices according to the grid type.

    - **cubed_sphere**: face-first arrays ``(6, n, n, ...)`` sharded on face.
      With sub-face tiling, a 3D mesh ``(face, tile_i, tile_j)`` is used so
      both horizontal dimensions are sharded directly.
    - **latlon**: arrays ``(n_lat, n_lon, ...)`` sharded on lat dimension.
    - **spectral**: arrays ``(nlev, ...)`` sharded on level dimension.

    Parameters
    ----------
    pytree
    arr : jax.Array (any sharding on ``rep``'s mesh)
    rep : NamedSharding — the replicated ``P()`` sharding of the target mesh.
    multiprocess : pass ``jax.process_count() > 1`` (keyword-only so the
        branch is explicit at every call site).
    """
    if multiprocess:
        return jax.jit(lambda a: a, out_shardings=rep)(arr)
    return jax.device_put(arr, rep)


def shard_leaf(arr, sharding, *, multiprocess: bool):
    """Scatter one full-global leaf onto ``sharding``'s mesh — the SCATTER
    primitive symmetric to :func:`replicate_leaf`, shared by the atm and ocean
    lat-band SPMD steps (``shard_state_atm_latlon`` / ``shard_state_latlon``).

    Single-process: plain ``jax.device_put`` (the historical path, unchanged and
    byte-identical).

    Multi-controller (``jax.process_count() > 1``, route-B ``jax.distributed``,
    a mesh spanning processes): a top-level ``jax.device_put`` of the FULL global
    array to a cross-process ``NamedSharding`` cannot place shards on peer
    processes' devices, so XLA falls back to an all-gather that (a) transiently
    materialises a second global copy per process and (b) is the collective seen
    to crash under full-node CPU packing (issue #1100: 128 procs × global-state
    each, rc=137 OOM). Instead ``jax.make_array_from_callback`` invokes the
    callback ONCE PER ADDRESSABLE SHARD with that shard's global index, and each
    process reads only its own shards out of the global array it already holds —
    no all-gather, no transient global replica. Using the per-shard index
    callback (not an enclosing [min,max) span) makes it correct for ANY
    device→process placement, including a non-contiguous/interleaved mesh order.

    NOT differentiable: ``make_array_from_callback`` is a host construction API,
    so (unlike the historical ``device_put``) a ``jax.grad``/``vjp`` cannot be
    taken THROUGH the multiprocess scatter. This is fine — the scatter is an
    init-time boundary (``scatter_to_local`` before the step loop, per the MPI
    pattern), never inside a differentiated loss; gradients w.r.t. params flow
    through the already-sharded state, not the scatter itself.

    Parameters
    ----------
    arr : the FULL global array, present on every process (host or device).
    sharding : NamedSharding — the lat-band ``P("lat", None, ...)`` target.
    multiprocess : pass ``jax.process_count() > 1`` (keyword-only so the branch
        is explicit at every call site, mirroring :func:`replicate_leaf`).
    """
    if not multiprocess:
        return jax.device_put(arr, sharding)
    return jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx])


def _pole_fold(rows, negate: bool):
    """Serial pole fold of ``rows`` (lat-mirror + 180 deg lon roll [+ sign]).

    ``rows`` is ``(halo, n_lon_padded[, nlev])`` already lon-wrapped; matches
    :func:`legoesm.grids.halo_latlon.fold_pole_rows` (``half = n_lon_pad // 2``,
    ``roll(rows[::-1], half, axis=lon)``)."""
# round 4 proved that a plausible-sounding reason hides a real defect, so the
# reasons below are now also checked MECHANICALLY (see
# TestExemptionsAreVerifiedNotAsserted).
_ATM_UNGATED = {
    "lat_spec":
        "pure: returns P() from arr.ndim. It CAN raise (AttributeError) on a "
        "non-array, but it enters no collective and is called only with real "
        "leaves, so a raise here cannot strand a peer in a gather",
    "tile_spec": "pure: same contract as lat_spec",
    "atm_grid_array_field_names": "pure host introspection of grid._fields",
    "atm_latlon_geometry_bytes":
        "pure host byte accounting; it calls build_band_grids_atm, which can "
        "raise on indivisibility, but neither enters a collective",
    "unroll_to_dtype_fixed_point":
        "trace-time only (jax.eval_shape probe); no host collective",
    "state_finite_scalar":
        "emits an in-graph psum when called with an axis inside a shard_map; "
        "it is a TRACED helper, never a host entry point that runs a "
        "collective by itself",
    "build_band_grids_atm":
        "pure host geometry. It is public and CAN be called directly (the "
        "byte-accounting helper does), so its divisibility ValueError is only "
        "symmetric because the inputs it raises on (grid.n_lat, n_devices) "
        "are themselves agreed at whichever gated entry the caller used; it "
        "enters no collective of its own",
    "build_tile_grids_atm_2d":
        "pure host geometry; same contract as build_band_grids_atm",
    "build_sharded_held_suarez_state_atm_latlon":
        "band-LOCAL construction via jax.make_array_from_callback (#1100): "
        "each addressable shard is built from local data, no gather, and no "
        "replicated put",
}

_OCEAN_GATED = {
    "make_sharded_ocean_step": "_agree_ocean_spmd_entry",
    "make_sharded_ocean_step_global": "_agree_ocean_spmd_entry",
    "gather_state_latlon": "_agree_ocean_mesh_entry",
    "shard_state_latlon": "_agree_ocean_mesh_entry",
    "shard_forcing_latlon": "_agree_ocean_mesh_entry",
    "shard_forcing_stack_latlon": "_agree_ocean_mesh_entry",
}

_OCEAN_UNGATED = {
    "build_band_grids":
        "pure host geometry; enters no collective of its own (same contract "
        "as build_band_grids_atm)",
    "append_vface_wall_row":
        "pure array op (concatenate a zero row); no collective, no branch",
}

_OPSPLIT_GATED = {
    "make_sharded_operator_split_step": "_agree_opsplit_spmd_entry",
    "shard_operator_split_carry": "_agree_opsplit_mesh_entry",
    "shard_operator_split_forcing": "_agree_opsplit_mesh_entry",
}

_OPSPLIT_UNGATED = {
    # CORRECTED (codex round 4). The previous reason said "no raise", which is
    # FALSE: `_need_rad_and_time` opens with `if rad_update_steps <= 1:`, a
    # Python comparison that raises TracerBoolConversionError on a traced value
    # and TypeError on a non-comparable one. CLAUDE.md treats every allow-list
    # reason as a CLAIM to be verified in code — a plausible-sounding but wrong
    # reason permanently hides a real defect — so the string is corrected here
    # rather than the entry being quietly kept.
    #
    # It stays exempt because the raise is SYMMETRIC, which is the property
    # that actually matters. `rad_update_steps` is agreed across processes by
    # the entry gate (it is a member of _OPSPLIT_SPMD_ENTRY_FLAGS), so every
    # rank evaluates this comparison on the same value: all raise or none do.
    # A symmetric raise is a clean error, not a one-sided hang.
    "need_rad_and_time":
        "module ALIAS of the pure _need_rad_and_time cadence helper "
        "(jnp arithmetic on step_index); NO COLLECTIVE. It CAN raise — "
        "`if rad_update_steps <= 1` is a Python comparison — but "
        "rad_update_steps is agreed by the entry gate "
        "(_OPSPLIT_SPMD_ENTRY_FLAGS), so the raise is symmetric across "
        "processes: all ranks raise or none, never a one-sided hang",
}


def _atm_module():
    import legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step as m
    return m


def _ocean_module():
    import legoesm.ocean.dynamics.sharded_ocean_step as m
    return m


def _opsplit_module():
    import legoesm.driver.sharded_operator_split_step as m
    return m


_LANES = [
    ("atm", _atm_module, _ATM_GATED, _ATM_UNGATED),
    ("ocean", _ocean_module, _OCEAN_GATED, _OCEAN_UNGATED),
    ("opsplit", _opsplit_module, _OPSPLIT_GATED, _OPSPLIT_UNGATED),
]


class TestEveryPublicEntryPointIsClassified:
    """Fix the CLASS, not the instance.

    Round 1 patched individual refusal functions; round 2 found five more
    unguarded paths; round 3 found two more (the run wrappers).  The only way
    off that treadmill is a check that FAILS when a NEW public symbol appears
    ungated, instead of a list of the sites someone happened to look at.

    A new public function in any of these three lat-band SPMD lanes must
packages/core/legoesm/parallel/geometry_consistency.py:52:    "checked_shard_put",
packages/core/legoesm/parallel/geometry_consistency.py:678:def checked_shard_put(arr, name, sharding, *, context, n_bands):
packages/core/legoesm/parallel/geometry_consistency.py:723:    :func:`checked_shard_put`-style puts bypass on NON-band inputs (state /
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:50:    assert_pytree_bytes_equal, assert_schema_agrees, checked_shard_put,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:732:        # checked_shard_put replaces the broadcast_checked+device_put pair:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:739:        return checked_shard_put(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:36:    FLAG_ABSENT, assert_flags_agree, assert_schema_agrees, checked_shard_put,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:491:    # checked_shard_put replaces broadcast_checked + device_put (the ocean
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:500:        name: checked_shard_put(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1590:    # 2-D twin of the checked_shard_put swap above (see that note).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1593:        name: checked_shard_put(

exec
/usr/bin/bash -lc "sed -n '680,760p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py; sed -n '700,750p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py; sed -n '1,85p' packages/core/legoesm/parallel/geometry_consistency.py; rg -n \"checked_shard_put|band_fingerprint|make_array_from_callback\" tests/unit/test_geometry_consistency.py tests/ocean/unit/test_sharded_geom_fingerprint.py tests/distributed -g '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 190ms:
    data (replication-unaware).

    The state must be laid out with :func:`shard_state_latlon` (``v`` /
    ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
    each band's ``nl+1`` v-faces, runs the step on the band geometry, and
    converts the result back to the ``v_lower`` representation.
    """
    # FIRST statement: agree every rank-local input before ANY
    # rank-local check can raise or return (codex round-2).
    _agree_ocean_spmd_entry(model, mesh, where="make_sharded_ocean_step")
    if mesh is None:                   # single-device: plain step
        return lambda state, dt, **forcing_kwargs: model.step(
            state, dt, **forcing_kwargs)

    n_dev = mesh.devices.size
    axis = mesh.axis_names[0]

    # Tripole north-fold (scaling-audit item 4): SUPPORTED under the same
    # v-carrier contract as the regular grid.  Every fold-touching operator
    # is already uniform-program fold-capable (the data-dependent
    # ``north_fold_mask``/``apply_north_fold`` selection on
    # ``axis_index == N-1`` — gated by test_latlon_spmd_northfold), and
    # ``build_band_grids``' slicer keeps ``is_active`` rank-consistent with
    # the ``fold_j=-1`` sentinel off the north band.  The one structural
    # assumption is the v-carrier's: the TOP v-face row (the seam/cap row,
    # ``v[n_lat]``) must be WALL-MASKED so the in-body reconstruction's
    # zero row is exact — true for the cap-row convention of
    # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
    # the serial step keeps ``v[-1] == 0`` identically).  That contract is
    # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
    # (unmasked) seam v-row refuses loudly there instead of silently
    # reconstructing zeros here.

    # --- host-side band geometries + vertex masks (replicated, indexed in-body) ---
    band_grids = build_band_grids(model.grid, n_dev)
    band_vmasks = _build_band_vertex_masks(model, n_dev)
    template = band_grids[0]           # static-scalar source (uniform bands)
    array_field_names = _geom_array_field_names(template)

    # Stack each geometry ARRAY field over the band axis (rank 0..N-1) and
    # SHARD along that axis (#1370 stage (iii), codex round-18): each device
    # holds ONLY its own band's slab instead of the whole global stack —
    # this was one of the residual ~1.4-1.7 global-field-equivalents of
    # per-device residency left after the host-side-build fix (probe
    # 26524423). The leading axis has length n_dev, so P("lat") divides it
    # exactly; the body indexes its local slab at [0]. Values are unchanged
    # — same stack, different placement; the process-0 broadcast +
    # divergence guard below runs on HOST values and is placement-blind.
    rep = NamedSharding(mesh, P("lat"))

    def _replicated_put(arr, name):
        # (Name kept for history; this is a SHARDED P("lat") stack put.)
        # checked_shard_put replaces the broadcast_checked+device_put pair:
        # the broadcast's psum program is [n_processes, stack] (nd x 849 MB
        # at LL2304 — the @96/@128 wall), and a numpy device_put onto an
        # all-process sharding pays jax's whole-array assert_equal on top.
        # The per-band gate keeps the divergence contract (n_bands is
        # schema-gated just below, so payload widths agree). ONE shared
        # implementation: legoesm.parallel.geometry_consistency.
        return checked_shard_put(
            arr, name, rep, context="make_sharded_ocean_step",
            n_bands=n_dev)

    # Schema gate FIRST (one fixed-shape collective every process reaches):
    # a process-dependent field list or a mixed jax_enable_x64 setting would
    # otherwise desynchronize the per-field gathers below instead of failing
    # with a clear message.
    # Build the raw stacks FIRST so the schema gate can also cover each
    # field's dtype class and ndim -- those decide the per-field payload
    # shape below, so a bool-vs-float disagreement must fail HERE rather than
    # deadlock in the per-field gather.
    _raw_geom = {
        name: jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
                        axis=0)
        for name in array_field_names
    }
    _raw_vmask = jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0)
    _gate_names = [*array_field_names, "vertex_mask"]
    assert_schema_agrees(
        _gate_names, n_dev, context="make_sharded_ocean_step",
        arrays=[*(_raw_geom[n] for n in array_field_names), _raw_vmask])
    # ``north_fold_mask``/``apply_north_fold`` selection on
    # ``axis_index == N-1`` — gated by test_latlon_spmd_northfold), and
    # ``build_band_grids``' slicer keeps ``is_active`` rank-consistent with
    # the ``fold_j=-1`` sentinel off the north band.  The one structural
    # assumption is the v-carrier's: the TOP v-face row (the seam/cap row,
    # ``v[n_lat]``) must be WALL-MASKED so the in-body reconstruction's
    # zero row is exact — true for the cap-row convention of
    # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
    # the serial step keeps ``v[-1] == 0`` identically).  That contract is
    # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
    # (unmasked) seam v-row refuses loudly there instead of silently
    # reconstructing zeros here.

    # --- host-side band geometries + vertex masks (replicated, indexed in-body) ---
    band_grids = build_band_grids(model.grid, n_dev)
    band_vmasks = _build_band_vertex_masks(model, n_dev)
    template = band_grids[0]           # static-scalar source (uniform bands)
    array_field_names = _geom_array_field_names(template)

    # Stack each geometry ARRAY field over the band axis (rank 0..N-1) and
    # SHARD along that axis (#1370 stage (iii), codex round-18): each device
    # holds ONLY its own band's slab instead of the whole global stack —
    # this was one of the residual ~1.4-1.7 global-field-equivalents of
    # per-device residency left after the host-side-build fix (probe
    # 26524423). The leading axis has length n_dev, so P("lat") divides it
    # exactly; the body indexes its local slab at [0]. Values are unchanged
    # — same stack, different placement; the process-0 broadcast +
    # divergence guard below runs on HOST values and is placement-blind.
    rep = NamedSharding(mesh, P("lat"))

    def _replicated_put(arr, name):
        # (Name kept for history; this is a SHARDED P("lat") stack put.)
        # checked_shard_put replaces the broadcast_checked+device_put pair:
        # the broadcast's psum program is [n_processes, stack] (nd x 849 MB
        # at LL2304 — the @96/@128 wall), and a numpy device_put onto an
        # all-process sharding pays jax's whole-array assert_equal on top.
        # The per-band gate keeps the divergence contract (n_bands is
        # schema-gated just below, so payload widths agree). ONE shared
        # implementation: legoesm.parallel.geometry_consistency.
        return checked_shard_put(
            arr, name, rep, context="make_sharded_ocean_step",
            n_bands=n_dev)

    # Schema gate FIRST (one fixed-shape collective every process reaches):
    # a process-dependent field list or a mixed jax_enable_x64 setting would
    # otherwise desynchronize the per-field gathers below instead of failing
    # with a clear message.
    # Build the raw stacks FIRST so the schema gate can also cover each
    # field's dtype class and ndim -- those decide the per-field payload
    # shape below, so a bool-vs-float disagreement must fail HERE rather than
    # deadlock in the per-field gather.
"""Cross-process agreement checks for per-process-recomputed SPMD geometry.

Every multi-controller SPMD lane faces the same hazard: each process rebuilds
the band/tile geometry from the same config, then hands it to a REPLICATED
``device_put``.  A ``P()`` (fully-replicated) put ASSERTS the value is
bit-identical on every process, and per-process XLA autotuning on
device-derived grid fields makes the last ULPs differ at larger sizes (job
26450848: LL576 np=4, area-scale fields differing at 1e-7 relative), which
trips that assert.

The remedy is to broadcast process 0's bytes — but broadcasting BLINDLY would
silently paper over a REAL cross-process inconsistency (a different wet
domain, a different field list, a mixed ``jax_enable_x64``), turning a loud
crash into wrong physics.  So every broadcast here is GUARDED: an allgathered
fingerprint must agree first, and a disagreement RAISES.

This module is the ONE implementation of that protocol.  It was extracted
from ``ocean.dynamics.sharded_ocean_step`` (where it was developed and
hardened over five rounds of adversarial review) so the atmosphere lat-lon
lane — which had the identical defect (#1362) — reuses it instead of growing
a second, drifting copy.  Per legoESM's no-duplicated-numerics rule, new SPMD
lanes MUST call these helpers rather than re-derive the fingerprints.

Sequencing contract, in this order:

1. :func:`assert_schema_agrees` ONCE, before any per-field work — a single
   fixed-shape collective that every process reaches.  A process-dependent
   field selection (e.g. an optional mask present on some ranks only) would
   otherwise DESYNCHRONIZE the per-field gathers below instead of failing
   with a clear message.
2. :func:`broadcast_checked` per field, in an order identical on every
   process.

NO DEADLOCK RISK: every process fingerprints the same fields in the same
order and derives its verdict from the SAME gathered array, so the refusal is
symmetric — all raise or none.
"""

from __future__ import annotations

import hashlib

import jax
import numpy as np

__all__ = [
    "addressable_shard_put",
    "leaf_digest48",
    "assert_pytree_bytes_equal",
    "band_fingerprint",
    "band_fingerprints_agree",
    "checked_shard_put",
    "content_hash48",
    "name_digest48",
    "schema_fingerprint",
    "assert_schema_agrees",
    "assert_flags_agree",
    "broadcast_checked",
    "coerce_count",
    "coerce_bool",
    "config_digest48",
    "tree_schema_digest48",
    "safe_repr",
    "FLAG_ABSENT",
    "FLAG_UNCOERCIBLE",
    "FLAG_OUT_OF_RANGE",
    "FLAG_NEGATIVE",
    "FLAG_MAX_EXACT",
    "FLAG_DIGEST_FAILED",
]

# --- entry-gate payload sentinels -------------------------------------------
# An entry gate turns rank-local scalars (n_steps, segment_steps, grid dims)
# into a fixed-width float payload.  Building that payload must NEVER raise:
# a rank that dies in `int(n_steps)` while its peers block in
# `process_allgather` is a HANG, which is strictly worse than the bug the gate
# exists to fix (codex 2026-07-29 round-3, blocker 3).  So an unusable value is
# mapped to a SENTINEL that travels through the collective; every rank then
# sees it in the gathered payload and the raise that follows is symmetric.
#
# The sentinels are large-magnitude NEGATIVE values that NO legitimate count
# can take.  They must also not collide with each other: ``FLAG_ABSENT`` used
# to be ``-1.0``, so a rank passing ``segment_steps=None`` and a peer passing
# ``-1`` produced the SAME payload entry, agreed, and then diverged downstream
# (codex round-4, blocker 1).  Counts are validated non-negative, so every
tests/unit/test_geometry_consistency.py:818:        "band-LOCAL construction via jax.make_array_from_callback (#1100): "
tests/ocean/unit/test_sharded_geom_fingerprint.py:13:    band_fingerprint as geom_band_fingerprint,
tests/ocean/unit/test_sharded_geom_fingerprint.py:14:    band_fingerprints_agree,
tests/ocean/unit/test_sharded_geom_fingerprint.py:23:    fps = [geom_band_fingerprint(h, N_BANDS) for h in hosts]
tests/ocean/unit/test_sharded_geom_fingerprint.py:34:    assert band_fingerprints_agree(*_gather(a, a.copy()))
tests/ocean/unit/test_sharded_geom_fingerprint.py:41:    assert band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:63:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:73:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:82:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:89:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:95:    fa = geom_band_fingerprint(a, N_BANDS)
tests/ocean/unit/test_sharded_geom_fingerprint.py:96:    fb = geom_band_fingerprint(b, N_BANDS)
tests/ocean/unit/test_sharded_geom_fingerprint.py:104:        geom_band_fingerprint(np.ones((3, 2)), N_BANDS)

exec
/usr/bin/bash -lc 'for f in tests/parallel/test_atm_latlon_spmd_step.py tests/parallel/test_atm_latlon_segment.py tests/parallel/test_atm_latlon_2d_tiling.py tests/parallel/test_atm_latlon_spmd_driver.py tests/parallel/test_atm_latlon_operator_split_spmd.py tests/parallel/test_atm_latlon_state_layout.py tests/parallel/test_atm_latlon_bandlocal_build.py; do echo "$f"; sed -n '"'1,55p' \""'$f"; done; rg -n "broadcast|process 0|bit-identical|per-process|P'"\\(\\)\" packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py packages/core/legoesm/parallel/geometry_consistency.py | head -160" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 257ms:
tests/parallel/test_atm_latlon_spmd_step.py
"""Stage 5 — the serial-vs-SPMD EQUIVALENCE GATE for the atm latlon step.

THE make-or-break validation of make_sharded_atm_latlon_step: N-band lat-band
SPMD integration must reproduce the single-device C-grid hydrostatic step. A
subtly-wrong band decomposition (wrong Coriolis / v-face interp at interior
cuts, band-local mass denominator, mis-reconstructed v-face row, wrong band
geometry) gives SILENT wrong answers and is caught here.

Two complementary gates:

1. ``test_atm_latlon_spmd_tendency_matches_serial`` — the RIGOROUS,
   RK-stage-independent decomposition check at the TENDENCY level.  The
   vector-invariant momentum (vorticity + Bernoulli gradient, incl. the
   absolute-vorticity Coriolis) and the centered flux-form continuity
   reconstruct the band cut EXACTLY from halo'd neighbour rows, so du/dv/dp_s
   and cor_u/cor_v are BIT-EXACT (fp64) on every band.  The ONLY non-bit-exact
   term is the LIMITED FV PPM scalar (T/tracer) advection: ``ppm_edge_values``
   uses a 2nd-order edge at the outermost ``halo=2`` padded row, where serial
   computes a 4th-order edge, and the CW84 limiter leaks a ~5e-12 difference
   into the two cut-ADJACENT T rows (verified: scalar-advection diff is
   confined to those rows; the scalar halo ghost rows are byte-identical).
   This is a PRE-EXISTING boundary-order property of band-decomposed limited
   FV PPM — the production MPI lat-band path uses the SAME backend-dispatched
   operator — NOT an SPMD bug, so it is asserted as a BOUNDED residual.

2. ``test_atm_latlon_spmd_step_matches_serial`` — the integrated full-step
   gate through the PUBLIC make_sharded interface.  Over a few RK3 steps the
   tiny cut-row T truncation advects into the small-magnitude u field; bound it
   FAR below any real decomposition error (~1e-3) while above the PPM
   truncation floor.

Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``);
the production target is multi-GPU/TPU but the shard_map/ppermute/psum logic is
device-agnostic.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
    make_sharded_atm_latlon_step,
    shard_state_atm_latlon,
    gather_state_atm_latlon,
tests/parallel/test_atm_latlon_segment.py
"""M2b parity gate — compiled multi-step lat-band SPMD segments + band-SHARDED
geometry for the lat-lon C-grid hydrostatic atm.

Three merge-bar gates (2 virtual CPU devices, x64):

(i)   ``make_sharded_atm_latlon_segment(n_steps=10)`` — ONE compiled
      ``lax.scan`` — matches 10 sequential ``make_sharded_atm_latlon_step``
      calls to 1e-12, and 10 SERIAL single-device steps: to 1e-12 with
      centered transport (the band decomposition is exact there), and to the
      documented Stage-5 limited-FV-PPM cut-truncation bound with PPM on
      (``tests/parallel/test_atm_latlon_spmd_step.py`` module docstring — the
      2nd-order halo edge at cut rows is a boundary-order property of
      band-decomposed limited PPM, not an SPMD bug).

(ii)  the segment's IN-GRAPH finite scalar (``state_finite_scalar``: psum of
      per-band non-finite presence over ALL state leaves) agrees with a
      host-side isfinite of the gathered state — on a healthy run AND under
      synthetic NaN injections in T (band 0) and u-only (band 1), so the gate
      is provably non-vacuous and the cross-band psum is exercised.

(iii) the geometry-SHARDED step (``shard_geometry=True``: per-device band
      slice, ``P("lat")`` stacks) BIT-matches the replicated-geometry step
      (the historical default) — with and without the polar-filter mask
      stacks — and the layouts are REALLY different on device (sharding
      introspection), so the bit-match cannot pass vacuously.

Plus the production-lane wiring gate: ``run_atm_latlon_spmd(...,
compiled_segments=True)`` matches the per-step path (final state + status,
including the remainder segment and the BLOWUP status contract).

Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count>=2``).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
    cgrid_to_hydrostatic,
)
from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
    atm_latlon_geometry_bytes,
    gather_state_atm_latlon,
    make_sharded_atm_latlon_segment,
    make_sharded_atm_latlon_step,
    run_atm_latlon_spmd,
tests/parallel/test_atm_latlon_2d_tiling.py
"""M3a parity gates — native 2-D ("lat", "lon") tiling for the atmosphere
lat-lon SPMD step.

The 1-D lat-band decomposition's halo perimeter is the CONSTANT n_lon per cut
(independent of the device count) — the term that caps band scaling.  M3a
adds a native 2-D tiling: periodic longitude as a cyclic ring ppermute,
staggered ownership for BOTH staggers (v_lower rows + u_left columns), the
EXACT serial 180-deg pole fold under a lon split (antipodal partner ppermute
for even p_lon, lon-ring all_gather for odd), and topology-aware
(p_lat, p_lon) selection.  The 1-D band path stays the default and
byte-identical.

Merge-bar gates (4 virtual CPU devices, x64):

(1)  PAD PARITY — ``make_latlon_2d_pad_body`` reproduces, per tile, the
     SERIAL ``pad_halo_latlon_local`` window BIT-exactly ((2,2) and (4,1);
     halo 1 and 2; scalar and vector fold).  This is the direct proof of the
     lon-ring exchange, the two-pass corner composition, and the all_gather
     pole fold.

(2)  TENDENCY — the (2,2)-tile tendency is BIT-tight vs serial for momentum
     + continuity; dT carries only the genuine limited-FV-PPM halo-2
     truncation at cut-adjacent lat ROWS and lon COLUMNS: the outermost
     ``ppm_edge_values`` edges on a halo-2 padded tile are 2nd-order and
     couple into the ghost-cell parabolas' LIMITED face states (CW84), so
     bit parity at a cut is impossible at halo 2.  The flux METRIC at cuts
     is exact (``lat_v_interfaces == grid.lat_v``, gated by the geometry
     test + the realistic-amplitude tendency gate — codex M3a findings 2/6:
     the pre-fix operator fabricated ±π/2 pole faces at every tile cut).

(3)  STEP — the integrated (2,2) 2-D step matches serial at the Stage-5
     bound, with SEAM-SPECIFIC assertions: the u periodic-seam column, the
     u/v values along every tile cut, and the v pole rows.

(4)  DEGENERACY — the (4,1) 2-D step is BIT-IDENTICAL to the existing 1-D
     band step (every lon-ring op takes its static local branch).

(5)  SEGMENT — the compiled 10-step 2-D ``lax.scan`` matches 10 sequential
     2-D steps at measured per-field ABSOLUTE caps (rtol=0 — honest gates,
     codex M3a finding 7) and serial at the Stage-5 bound; the in-graph
     finite scalar (psum over BOTH axes) is exercised healthy + NaN-injected;
     a mixed-precision (f32) IC exercises the dtype-fixed-point unroll, also
     gated at measured absolute caps with an injected-defect tripwire.

(6)  CHOOSER — modeled pad communication volume (ppermute perimeter PLUS the
     two pole-fold lon-all_gathers every lon split pays on every tile —
     codex M3a finding 4), band selection whenever feasible,
     divisibility/min-tile feasibility, and the loud no-factorization error.

Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
"""
from __future__ import annotations

import jax

tests/parallel/test_atm_latlon_spmd_driver.py
"""End-to-end of the ModelDriver lat-band SPMD run method
(``_run_compiled_latlon_spmd``), exercised through a lightweight driver stub so
no data-loading ModelDriver.setup() is needed: the method reads only
config / model / grid / state / _segment_callback. Validates the production glue
(mesh build + run_atm_latlon_spmd + on_segment + self.state update + status)
against a direct run_atm_latlon_spmd reference. 4 host CPU devices via
``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from legoesm import constants
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.model_driver import ModelDriver
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
    run_atm_latlon_spmd,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import (
    held_suarez_forcing_latlon, held_suarez_init_latlon)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

N_DEV, N_LAT, N_LON, NLEV = 4, 16, 16, 4
DT = 100.0


class _DriverStub:
    """Minimal stand-in exposing exactly what _run_compiled_latlon_spmd reads,
    with the three real ModelDriver methods bound onto it (so ``self.*`` resolves
    without a full data-loading ModelDriver construction)."""
    _latlon_spmd_mesh = ModelDriver._latlon_spmd_mesh
    _latlon_spmd_physics_fn = ModelDriver._latlon_spmd_physics_fn
    _operator_split_spmd_active = ModelDriver._operator_split_spmd_active
    _run_compiled_latlon_spmd = ModelDriver._run_compiled_latlon_spmd

    def __init__(self, model, state, cfg):
        self.config = cfg
        self.model = model
        self.grid = model.grid
        self.state = state
        self._segment_callback = None
        self._current_day = 0.0


def _model_and_hs():
tests/parallel/test_atm_latlon_operator_split_spmd.py
"""Parity gate for the lat-band-SPMD operator-split atmosphere step.

``make_sharded_operator_split_step`` runs the SAME operator-split integration as
the serial ``_single_step`` (dynamics -> dry-mass fixer -> column-local physics
-> Euler write-back -> saturation/moisture-fix/smoothing/Rayleigh -> carry pack)
on a lat-band-sharded ``SegmentCarry``. Because the physics is PURELY
column-local it is decomposition-INVARIANT: the sharded (mesh=4dev) trajectory
must match the serial (mesh=None) one BITWISE in every physics-touched /
column-local field, and only to the FV-PPM cut bound in the dynamical fields
(u,v,p_s) — the same limited-FV-PPM boundary-order residual the dynamics-only
SPMD step already carries.

The physics is a column-local MOCK ``step_unified`` (small constant warming +
an AR1 tke evolution ``0.9*tke + 1e-3``): it exercises the full sharded
MECHANICS — band cell<->C-grid dynamics, the dry-mass fixer psum, the flattened
tke carry shard, the hyperdiffusion halo, the Euler write-back + accumulators —
without depending on a full physics harness (the real-scheme end-to-end parity
is the driver-level gate). Host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``), x64.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig)
from legoesm.core.conservation import global_area_sum
from legoesm.core.operators_latlon_3d import hyperdiffusion_3d
from legoesm.driver.compiled_segments import (
    _SplitStepStatics, SegmentCarry, pack_carry, pack_forcing)
from legoesm.driver.physics_pipeline import PhysicsOutput
from legoesm.driver.sharded_operator_split_step import (
    make_sharded_operator_split_step, shard_operator_split_carry,
    shard_operator_split_forcing)

N_DEV = 4
N_LAT = 16
N_LON = 16
NLEV = 8
DT = 100.0


def _mesh():
    if len(jax.devices()) < N_DEV:
tests/parallel/test_atm_latlon_state_layout.py
"""Stage 2 of the atm latlon SPMD step: the 6-field C-grid state shard/gather.

``shard_state_atm_latlon`` lays the C-grid hydrostatic state onto the lat-band
mesh (cell/u leaves ``P("lat")``; the staggered ``v`` carried as
``v_lower=v[:n_lat]``); ``gather_state_atm_latlon`` is its inverse (re-append the
zero pole-wall v-face). Round-trip identity holds for a pole-walled v
(``v[-1]==0``), the state the model produces. Host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
)
from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
    shard_state_atm_latlon, gather_state_atm_latlon)

N_DEV = 4
N_LAT = 16
N_LON = 8
NLEV = 4


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


def _rand_state(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N_LAT + 1, N_LON, NLEV))
    v[0] = 0.0    # south pole wall
    v[-1] = 0.0   # north pole wall (the face dropped + re-appended)
    return CGridLatLonHydrostaticState(
        u=jnp.asarray(rng.standard_normal((N_LAT, N_LON + 1, NLEV))),
        v=jnp.asarray(v),
        T=jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)) + 250.0),
        p_s=jnp.asarray(rng.standard_normal((N_LAT, N_LON)) + 1.0e5),
        phis=jnp.asarray(rng.standard_normal((N_LAT, N_LON))),
        tracers={"q": jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))},
    )


def test_shard_gather_round_trip_identity():
    mesh = _mesh()
    s = _rand_state()
tests/parallel/test_atm_latlon_bandlocal_build.py
"""#1100 — band-local Held-Suarez IC construction parity gate.

``build_sharded_held_suarez_state_atm_latlon`` must be BIT-IDENTICAL, shard
by shard, to the reference path (global ``held_suarez_init_latlon`` →
``hydrostatic_to_cgrid`` → ``shard_state_atm_latlon``) on a virtual-device
lat mesh — this is the correctness contract that lets the route-B bench skip
the per-process global build (the OOM mechanism under full-node CPU packing).

Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``);
skips when fewer devices are available.  The multi-process
(``make_array_from_callback`` per-process rows) leg of #1100 needs a real
multi-controller launch and is validated by the bench itself on cluster —
this gate pins the single-process semantics both legs share.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

N_DEV = 4
N_LAT = 16   # divisible by N_DEV
N_LON = 12
NLEV = 5


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return jax.sharding.Mesh(
        np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


def _grid_sigma():
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    return grid, sigma


def _reference_sharded(grid, sigma, mesh):
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_latlon)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        shard_state_atm_latlon)
    hs0 = held_suarez_init_latlon(grid, sigma)
    c0 = hydrostatic_to_cgrid(hs0, grid)
packages/core/legoesm/parallel/geometry_consistency.py:1:"""Cross-process agreement checks for per-process-recomputed SPMD geometry.
packages/core/legoesm/parallel/geometry_consistency.py:5:``device_put``.  A ``P()`` (fully-replicated) put ASSERTS the value is
packages/core/legoesm/parallel/geometry_consistency.py:6:bit-identical on every process, and per-process XLA autotuning on
packages/core/legoesm/parallel/geometry_consistency.py:11:The remedy is to broadcast process 0's bytes — but broadcasting BLINDLY would
packages/core/legoesm/parallel/geometry_consistency.py:14:crash into wrong physics.  So every broadcast here is GUARDED: an allgathered
packages/core/legoesm/parallel/geometry_consistency.py:31:2. :func:`broadcast_checked` per field, in an order identical on every
packages/core/legoesm/parallel/geometry_consistency.py:58:    "broadcast_checked",
packages/core/legoesm/parallel/geometry_consistency.py:205:    :func:`broadcast_checked`, not of a cheap fixed-width entry gate.
packages/core/legoesm/parallel/geometry_consistency.py:309:    unsupported field fails later in :func:`broadcast_checked` with its own
packages/core/legoesm/parallel/geometry_consistency.py:379:    The dtype/ndim terms are not cosmetic.  :func:`broadcast_checked` routes
packages/core/legoesm/parallel/geometry_consistency.py:450:    Call this BEFORE any rank-local ``raise`` that inspects per-process
packages/core/legoesm/parallel/geometry_consistency.py:471:            f"{context}: per-process CONFIG differs across processes "
packages/core/legoesm/parallel/geometry_consistency.py:482:    :func:`broadcast_checked` calls that follow are matched positionally
packages/core/legoesm/parallel/geometry_consistency.py:487:    SHAPE in :func:`broadcast_checked`, so leaving them out lets a
packages/core/legoesm/parallel/geometry_consistency.py:506:        # exists to remove (codex round-2 minor). `broadcast_checked` does
packages/core/legoesm/parallel/geometry_consistency.py:524:            f"per-process config before sharding; the per-field checks "
packages/core/legoesm/parallel/geometry_consistency.py:528:def broadcast_checked(arr, name: str, *, context: str) -> np.ndarray:
packages/core/legoesm/parallel/geometry_consistency.py:529:    """Verify ``arr`` agrees across processes, then broadcast process 0's bytes.
packages/core/legoesm/parallel/geometry_consistency.py:531:    Multi-process: returns a host ``np.ndarray`` that is bit-identical on
packages/core/legoesm/parallel/geometry_consistency.py:603:            f"broadcast process 0 over it.")
packages/core/legoesm/parallel/geometry_consistency.py:604:    return np.asarray(multihost_utils.broadcast_one_to_all(host))
packages/core/legoesm/parallel/geometry_consistency.py:609:# r14-r19; PR #1457): (1) broadcast_one_to_all of a band stack lowers to an
packages/core/legoesm/parallel/geometry_consistency.py:667:    """True iff every process's :func:`band_fingerprint` matches process 0's."""
packages/core/legoesm/parallel/geometry_consistency.py:679:    """Gate a band-stacked field per band, then put WITHOUT broadcast or
packages/core/legoesm/parallel/geometry_consistency.py:743:            f"byte digests disagree) — the per-process inputs are NOT "
packages/core/legoesm/parallel/geometry_consistency.py:745:            f"Fix the per-process build before sharding.")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:140:        # controller -> per-process local band via make_array_from_process_local_data
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:174:    removes the per-process global-state BUILD that made the route-B lat-lon
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:271:    rep = NamedSharding(mesh, P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:392:    ``out_specs=P()``), and the host reads ONE scalar per segment instead of
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:440:    ``device_put`` REPLICATED (``P()``) — each device holds ALL bands'
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:475:    # config, and per-process XLA autotuning on device-derived grid fields
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:477:    # bit-identical assert inside a replicated device_put (the ocean lane hit
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:479:    # cross-process agreement, then broadcast process 0's bytes.  Guarded, not
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:481:    # different physics) RAISES instead of being masked by process 0.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:484:    # entries are CONDITIONAL on ``model._polar_mask``, so a per-process
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:491:    # checked_shard_put replaces broadcast_checked + device_put (the ocean
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:492:    # walls, PR #1457): the broadcast's psum program is [n_processes,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:498:    spec_of = lat_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:711:        # and `_build_geometry_stacks` broadcasts; agreeing their dtypes and
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:901:    band values either way, so the step is bit-identical (gated by
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:971:        return P()
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1006:                    _body, mesh=mesh, in_specs=(in_spec, stacks_spec, P()),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1012:                    in_specs=(in_spec, stacks_spec, P(), ps_spec),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1050:    bit-identical numerics, 1/n_dev the geometry bytes
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1161:                in_specs=(in_spec, stacks_spec, P()),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1162:                out_specs=(in_spec, P()), check_vma=False))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1414:# (N, 1) 2-D mesh degenerates bit-identically anyway).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1473:    rep = NamedSharding(mesh, P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1547:    ``shard_geometry=False``: replicated (``P()``) stacks, indexed at
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1583:    # per-process recompute, same replicated-device_put bit-identity assert.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1591:    spec_of = tile_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1673:    so a per-process difference would BOTH skew the field schema AND make one
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1707:    A degenerate ``(N, 1)`` mesh is bit-identical to the 1-D band step
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1722:    Same tile values either way (bit-identical numerics).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1772:                _body, mesh=mesh, in_specs=(in_spec, stacks_spec, P()),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1858:                in_specs=(in_spec, stacks_spec, P()),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1859:                out_specs=(in_spec, P()), check_vma=False))

exec
/usr/bin/bash -lc "sed -n '330,430p' packages/core/legoesm/parallel/geometry_consistency.py; sed -n '450,535p' packages/core/legoesm/parallel/geometry_consistency.py; sed -n '230,295p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; sed -n '1450,1495p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; rg -n \"def schema_fingerprint|shape\" packages/core/legoesm/parallel/geometry_consistency.py | head -70" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 209ms:
# non-finite count) would let two processes enter `process_allgather` with
# different shapes and DEADLOCK -- the exact failure this module exists to
# turn into a clean symmetric raise (codex 2026-07-29, blocker 2; the flaw was
# inherited from the pre-extraction ocean implementation, so fixing it here
# fixes BOTH lanes).
_STRUCT_WIDTH = 8
_VALS_WIDTH = 3

# Relative tolerance for FLOAT geometry fields. Only ULP-scale autotune drift
# is expected there; quantize-then-assert-equal false-positived on a rounding
# boundary (job 26453240), so compare with a tolerance instead.
_FLOAT_RTOL = 1e-5


def content_hash48(arr) -> float:
    """48-bit content digest of ``arr``'s bytes, exactly representable in f64.

    Used to compare EXACT-dtype arrays (masks, index tables) across
    processes: unlike moment fingerprints, a byte digest is positional, so a
    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
    under 2**53 so it survives the float64 ``process_allgather`` payload
    exactly. Not cryptographic — collision-resistance at 2**-48 is far
    beyond the ~10 setup-time comparisons this guard makes.
    """
    a = np.ascontiguousarray(arr)
    h = hashlib.blake2b(a.tobytes(), digest_size=6)
    return float(int.from_bytes(h.digest(), "big"))


def name_digest48(names) -> float:
    """Order-sensitive, UNAMBIGUOUS digest of a sequence of names.

    Uses a NUL separator, which cannot occur in a Python identifier or any
    legoESM field name, so ``["a,b", "c"]`` and ``["a", "b,c"]`` cannot
    collide.  A plain ``",".join`` COULD (codex 2026-07-29, minor 5): those
    two lists have the same length, so a count check does not separate them
    either.
    """
    joined = "\x00".join(names).encode()
    return float(int.from_bytes(
        hashlib.blake2b(joined, digest_size=6).digest(), "big"))


def schema_fingerprint(names, n_dev, dtype_kinds=(), ndims=()) -> np.ndarray:
    """Fixed-shape schema digest gathered ONCE before the per-field loop.

    Covers the field-name list (order-sensitive), the count, the x64 flag,
    ``n_dev``, and -- critically -- the per-field DTYPE CLASS and NDIM.

    The dtype/ndim terms are not cosmetic.  :func:`broadcast_checked` routes
    exact dtypes to a 1-value digest and float dtypes to a 3-moment
    fingerprint, and its struct entry depends on ndim.  If the schema gate
    did not cover those, a field that is bool on one process and float on
    another would PASS the gate and then deadlock inside the per-field
    gather with mismatched payloads.  Catching it here converts that hang
    into a clean symmetric RuntimeError (codex 2026-07-29, blocker 2).

    ``dtype_kinds``/``ndims`` default to empty for callers that have not yet
    resolved the arrays; passing them is strongly preferred.
    """
    return np.array(
        [float(len(names)),
         name_digest48(names),
         float(bool(jax.config.jax_enable_x64)),
         float(n_dev),
         name_digest48([str(k) for k in dtype_kinds]),
         name_digest48([str(int(n)) for n in ndims])],
        dtype=np.float64)


def in_jax_trace() -> bool:
    """True when the caller runs inside a JAX trace (``jit``/``scan``/``vmap``).

    The host-side gates below call ``multihost_utils.process_allgather``, which
    is an EAGER utility: it ``device_put``s its payload per addressable device.
    Under an active trace those puts are staged into the jaxpr and come back as
    tracers, so ``make_array_from_single_device_arrays`` is handed tracers and
    raises — every multi-process lat-lon SPMD run died this way once the step
    was wrapped in ``lax.scan``/``jax.jit`` (#1405, follow-up to #1362).

    TWO LIMITATIONS, stated because a reader will otherwise assume they are
    covered (both raised by codex adversarial review of this change, both
    accepted deliberately — the alternative is a lane that cannot run at all):

    1. The skip is symmetric only as long as every process reaches this call
       in the SAME transform state, which is the SPMD lockstep property the
       gate itself exists to enforce.  If one rank called the step eagerly
       while another traced it, the eager rank would now BLOCK in
       ``process_allgather`` instead of its peer crashing.  That divergence is
       already fatal today (the traced rank dies here), so this trades a
       guaranteed crash on every multi-process traced run for a hang in an
       already-divergent one.  It is NOT a proof of symmetry.
    2. Coverage IS lost on a lane that is only ever traced.  The build-time
       gates (``_agree_spmd_entry``) agree the model/mesh/config; the per-CALL
       payload — state pytree schema, ``phys_state``/forcing presence and its
       schema — is agreed ONLY here, and under a trace it now goes unchecked.

    The trace-safe design that would fix both (stage the digest comparison as
    a mesh collective inside the traced program instead of a host allgather)
    needs a real multi-process rig to validate and is deliberately left as
    follow-up rather than written blind — see #1405.
    Call this BEFORE any rank-local ``raise`` that inspects per-process
    config.  Otherwise one process can reject its config and exit while its
    peers proceed into a collective and block forever — a collective-ORDER
    violation whose symptom (hang vs backend error) is backend-dependent
    (codex 2026-07-29, blocker 1).

    ``names`` and ``values`` must be STATIC tuples written at the call site,
    so the payload length is fixed by the code path rather than by data.

    No-op under a JAX trace — see :func:`in_jax_trace` (#1405).
    """
    if jax.process_count() <= 1 or in_jax_trace():
        return
    from jax.experimental import multihost_utils

    payload = np.array(
        [float(len(values)), name_digest48(names),
         *(float(v) for v in values)], dtype=np.float64)
    gathered = multihost_utils.process_allgather(payload)
    if not bool(np.all(gathered == gathered[0])):
        raise RuntimeError(
            f"{context}: per-process CONFIG differs across processes "
            f"(flags {list(names)} -> gathered {gathered.tolist()}). Every "
            f"process must be built from the same config; refusing before "
            f"any rank-local rejection so the failure is symmetric rather "
            f"than a hang.")


def assert_schema_agrees(names, n_dev, *, context: str, arrays=None) -> None:
    """Raise unless every process agrees on the geometry field SCHEMA.

    ``names`` must be an ORDERED sequence — the per-field
    :func:`broadcast_checked` calls that follow are matched positionally
    across processes, so a reordering is itself a divergence worth catching.

    Pass ``arrays`` (the per-name arrays, same order) so the gate also covers
    each field's DTYPE CLASS and NDIM.  Those decide the per-field payload
    SHAPE in :func:`broadcast_checked`, so leaving them out lets a
    bool-vs-float disagreement slip past this gate and deadlock in the
    per-field gather instead of raising here.

    No-op when ``jax.process_count() == 1``.
    """
    if jax.process_count() <= 1:
        return
    from jax.experimental import multihost_utils

    names = list(names)
    if arrays is None:
        kinds, ndims = (), ()
    else:
        # Read dtype/ndim from array METADATA, never via np.asarray: a jax
        # array exposes both without materialising, so forcing a host copy
        # here would add a device sync per field AND could itself fail
        # (transfer error / OOM) BEFORE the collective below — reintroducing
        # the very "one rank exits while a peer blocks" hazard this gate
        # exists to remove (codex round-2 minor). `broadcast_checked` does
        # the single real materialisation later.
        #
        # `_dtype_kind_and_ndim` also survives a plain Python scalar, which a
        # bare `a.dtype` read did not (codex round-3, minor 3): no production
        # caller passes one today, but an AttributeError HERE would be a
        # rank-local raise BEFORE the collective, i.e. a hang rather than a
        # clear failure.
        described = [_dtype_kind_and_ndim(a) for a in arrays]
        kinds = [d[0] for d in described]
        ndims = [d[1] for d in described]
    gathered = multihost_utils.process_allgather(
        schema_fingerprint(names, n_dev, kinds, ndims))
    if not bool(np.all(gathered == gathered[0])):
        raise RuntimeError(
            f"{context}: the band-geometry SCHEMA differs across processes "
            f"(field list / x64 setting / device count / per-field dtype "
            f"class / ndim — gathered {gathered.tolist()}). Fix the "
            f"per-process config before sharding; the per-field checks "
            f"assume one schema.")


def broadcast_checked(arr, name: str, *, context: str) -> np.ndarray:
    """Verify ``arr`` agrees across processes, then broadcast process 0's bytes.

    Multi-process: returns a host ``np.ndarray`` that is bit-identical on
    every process, safe to hand to a replicated ``device_put``.
    Single-process: returns ``arr`` ITSELF, untouched — no collectives, no
    host round trip, no dtype/weak-type change.

        return block.at[:, :, -1].add(pert2d[idx[0], idx[1]])

    def _ps_cb(idx):
        shape = _slice_shape((n_lat, n_lon), idx)
        # held_suarez_init_latlon with phis=None: p_s_init * exp(-0/(R_d T))
        phis_block = jnp.zeros(shape, dtype=_dtype)
        return (p_s_init * jnp.exp(
            -phis_block / (constants.R_d * T_init))).astype(_dtype)

    sh_u = (n_lat, n_lon + 1, nlev)
    sh_vlow = (n_lat, n_lon, nlev)   # sharded layout: pole-wall face dropped
    sh_T = (n_lat, n_lon, nlev)
    sh_2d = (n_lat, n_lon)
    return CGridLatLonHydrostaticState(
        u=_make(sh_u, _zeros_cb(sh_u)),
        v=_make(sh_vlow, _zeros_cb(sh_vlow)),
        T=_make(sh_T, _T_cb),
        p_s=_make(sh_2d, _ps_cb),
        phis=_make(sh_2d, _zeros_cb(sh_2d)),
        tracers={},
    )


def gather_state_atm_latlon(
    state: CGridLatLonHydrostaticState, mesh,
) -> CGridLatLonHydrostaticState:
    """Inverse of :func:`shard_state_atm_latlon`: replicate every leaf and
    rebuild the full ``(n_lat+1, ...)`` ``v`` by re-appending the zero north
    pole-wall face. Bit-comparable to the single-device state (whose top v-face
    is the pole wall == 0).

    Multi-controller (route-B ``jax.distributed``, mesh spanning processes):
    replication routes through a jit-compiled identity instead of
    ``device_put`` (see :func:`legoesm.parallel.latlon_spmd.replicate_leaf`,
    the primitive shared with the ocean gather); the single-process path is
    byte-unchanged."""
    # FIRST statement: this entry point runs a cross-process replication
    # collective (see _agree_mesh_entry).
    _agree_mesh_entry(mesh, state, where="gather_state_atm_latlon")
    from legoesm.parallel.latlon_spmd import replicate_leaf

    rep = NamedSharding(mesh, P())
    _mp = jax.process_count() > 1

    def _get(arr):
        return replicate_leaf(arr, rep, multiprocess=_mp)

    v_lower = _get(state.v)
    v_full = jnp.concatenate([v_lower, jnp.zeros_like(v_lower[:1])], axis=0)
    return state._replace(
        u=_get(state.u),
        v=v_full,
        T=_get(state.T),
        p_s=_get(state.p_s),
        phis=_get(state.phis),
        tracers={k: _get(val) for k, val in state.tracers.items()},
    )


# ==============================================================================
# Stage 7 — cell-centered HydrostaticState <-> sharded C-grid state BRIDGE
# ==============================================================================
# The SPMD step (``make_sharded_atm_latlon_step``) consumes/produces a
# ``CGridLatLonHydrostaticState`` (raw face-staggered arrays, laid out as
# ``v_lower``).  The rest of the system — IC builders, the driver, output,
# restart I/O — speaks the cell-centered, Field-wrapped ``HydrostaticState``.
    return state._replace(
        u=_put(state.u[:, :n_lon]),
        v=_put(state.v[:n_lat]),
        T=_put(state.T),
        p_s=_put(state.p_s),
        phis=_put(state.phis),
        tracers={k: _put(val) for k, val in state.tracers.items()},
    )


def gather_state_atm_latlon_2d(
    state: CGridLatLonHydrostaticState, mesh,
) -> CGridLatLonHydrostaticState:
    """Inverse of :func:`shard_state_atm_latlon_2d`: replicate every leaf,
    re-append the zero north pole-wall v face and the periodic u seam column
    (``u[:, n_lon] = u[:, 0]``).  Bit-comparable to the single-device state,
    whose top v-face is the pole wall (== 0) and whose last u column is the
    periodic closure (== column 0)."""
    # FIRST statement: this entry point runs a cross-process replication
    # collective (see _agree_mesh_entry).
    _agree_mesh_entry(mesh, state, where="gather_state_atm_latlon_2d")
    from legoesm.parallel.latlon_spmd import replicate_leaf

    rep = NamedSharding(mesh, P())
    _mp = jax.process_count() > 1

    def _get(arr):
        return replicate_leaf(arr, rep, multiprocess=_mp)

    v_lower = _get(state.v)
    v_full = jnp.concatenate([v_lower, jnp.zeros_like(v_lower[:1])], axis=0)
    u_left = _get(state.u)
    u_full = jnp.concatenate([u_left, u_left[:, 0:1]], axis=1)
    return state._replace(
        u=u_full,
        v=v_full,
        T=_get(state.T),
        p_s=_get(state.p_s),
        phis=_get(state.phis),
        tracers={k: _get(val) for k, val in state.tracers.items()},
    )


def build_tile_grids_atm_2d(grid, p_lat: int, p_lon: int):
    """Build the ``p_lat x p_lon`` UNIFORM tile ``LatLonGrid`` geometries via
    the tested 2-D MPI slicer (no bespoke metric re-derivation) — the 2-D
27:   fixed-shape collective that every process reaches.  A process-dependent
140:      on rank-local shape.
204:    ``dtype`` + ``shape`` — comparing their VALUES is the job of
233:        elif hasattr(val, "dtype") and hasattr(val, "shape"):
235:                       f"{safe_repr(tuple(val.shape), 64)}")
266:    """Digest of a pytree's LEAF SCHEMA: ordered path, dtype and full shape.
271:    species, different insertion order, different shape) produces mismatched
286:            shape = getattr(leaf, "shape", None)
289:                f"{safe_repr(tuple(shape) if shape is not None else None, 64)}")
331:# different shapes and DEADLOCK -- the exact failure this module exists to
373:def schema_fingerprint(names, n_dev, dtype_kinds=(), ndims=()) -> np.ndarray:
374:    """Fixed-shape schema digest gathered ONCE before the per-field loop.
565:    # FIXED-WIDTH payloads (see _STRUCT_WIDTH/_VALS_WIDTH): the gathered shape
567:    # collective with different shapes and hang. Shape is folded in as a
575:    struct[4] = name_digest48([str(d) for d in host.shape])
623:    PREREQUISITE: ``n_bands`` (and each field's dtype class / shape) must
639:    if host.ndim == 0 or host.shape[0] != n_bands:
642:            f"{host.shape[0] if host.ndim else '<0-d>'} != n_bands "
645:    struct = [float(host.ndim), *map(float, host.shape),
704:        host.shape, sharding, lambda idx: host[idx])
711:    # integer/float inputs from them (codex r21). The dtype code and shape
716:    meta = f"{np.dtype(a.dtype).num}:{a.shape}".encode()
756:        host.shape, sharding, lambda idx: host[idx])

codex
The 2-D mask shape is consistent (`[p_lat, 1, …]` because `p_lon > 1` is refused), and the obsolete import is gone. I found one semantic blocker in the replicated `P()` path: the new callback placement deliberately permits non-identical replicas, while the prior path canonicalized process 0.
codex
Blocking issue:

- In replicated `P()` mode, each process now builds a different physical replica when float geometry differs within the allowed `rtol=1e-5` gate. Those stacks are inputs to `shard_map` as replicated values; no current body explicitly gathers geometry, but this violates the prior canonical replicated-array contract and can yield process-dependent state evolution. The multicontroller receipts use identical builds, so they do not exercise this allowed-divergence case. Preserve `broadcast_checked` for `shard_geometry=False` (or explicitly redesign/drop the replicated contract).

Other findings:

- The 2-D polar masks are fine: `[:, None]` yields `[p_lat, 1, …]`, and polar filtering rejects `p_lon > 1`. For non-polar 2-D fields, the gate is per latitude slab, not per tile—consistent with `shape[0]`, but weaker than a per-tile gate.
- `broadcast_checked` is removed from this file. Its remaining production caller is the operator-split lane.
- Virtual-device requirements are already documented in the relevant ATM parity modules.
- Update stale comments that still claim process-0 broadcast/canonical bytes, notably [sharded_atm_latlon_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:475).

VERDICT: FIX-FIRST
tokens used
127,253
Blocking issue:

- In replicated `P()` mode, each process now builds a different physical replica when float geometry differs within the allowed `rtol=1e-5` gate. Those stacks are inputs to `shard_map` as replicated values; no current body explicitly gathers geometry, but this violates the prior canonical replicated-array contract and can yield process-dependent state evolution. The multicontroller receipts use identical builds, so they do not exercise this allowed-divergence case. Preserve `broadcast_checked` for `shard_geometry=False` (or explicitly redesign/drop the replicated contract).

Other findings:

- The 2-D polar masks are fine: `[:, None]` yields `[p_lat, 1, …]`, and polar filtering rejects `p_lon > 1`. For non-polar 2-D fields, the gate is per latitude slab, not per tile—consistent with `shape[0]`, but weaker than a per-tile gate.
- `broadcast_checked` is removed from this file. Its remaining production caller is the operator-split lane.
- Virtual-device requirements are already documented in the relevant ATM parity modules.
- Update stale comments that still claim process-0 broadcast/canonical bytes, notably [sharded_atm_latlon_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:475).

VERDICT: FIX-FIRST
