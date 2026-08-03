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
session id: 019fc651-3060-7693-bdbd-1a80e753eaa4
--------
user
Round-19 confirm: your r18 P1 applied — both JRA55 block builders in scripts/run/run_omip.py now define a 2-arg-normalized _dyn_step(st, d, aux=None, **kw) (spmd path forwards aux, serial path ignores it), both block_fn signatures gained aux=None, both _dyn_step call sites pass aux=aux, the host loop defines _spmd_aux = getattr(spmd_step, 'aux', None) and all FOUR invocation sites pass aux=_spmd_aux. 44 run_omip tests pass. Read git diff HEAD -- scripts/run/run_omip.py. Any remaining capture path or signature mismatch? End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll inspect the exact `run_omip.py` diff and trace every JRA55 block-builder call path for argument compatibility.
exec
/usr/bin/bash -lc 'git diff HEAD -- scripts/run/run_omip.py && rg -n -C 4 "_dyn_step|block_fn|_spmd_aux|spmd_step" scripts/run/run_omip.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 378ms:
diff --git a/scripts/run/run_omip.py b/scripts/run/run_omip.py
index c9913ad76..06ad14db9 100644
--- a/scripts/run/run_omip.py
+++ b/scripts/run/run_omip.py
@@ -2367,12 +2367,21 @@ def _build_jra55_block_fn(model, jra55_state, dt, spmd_step=None):
             "spmd_step + prognostic sea ice is unsupported "
             "(run_omip_single refuses --jra55-sea-ice with "
             "--enable-latlon-spmd).")
-    _dyn_step = (spmd_step if spmd_step is not None
-                 else lambda st, d, **kw: model._step_impl(st, d, **kw))
+    # aux threading (codex r18 P1): the SPMD step's sharded geometry
+    # stacks must cross THIS jit boundary as an ARGUMENT — captured in the
+    # closure they become outer-trace constants whose value jax cannot
+    # fetch for non-addressable arrays on multicontroller (see
+    # make_sharded_ocean_step's aux note).
+    if spmd_step is not None:
+        def _dyn_step(st, d, aux=None, **kw):
+            return spmd_step(st, d, aux=aux, **kw)
+    else:
+        def _dyn_step(st, d, aux=None, **kw):
+            return model._step_impl(st, d, **kw)
 
     @jax.jit
     def block_fn(state, atm_stack, runoff_stack, block_start_step,
-                 ice_state=None):
+                 ice_state=None, aux=None):
         def step_body(carry, idx):
             if enable_sea_ice:
                 state_in, ice_in = carry
@@ -2449,7 +2458,7 @@ def _build_jra55_block_fn(model, jra55_state, dt, spmd_step=None):
                 sponge_step = sponge
 
             new_state = _dyn_step(
-                state_in, dt,
+                state_in, dt, aux=aux,
                 freshwater=fw, surface_forcing=sf, sponge=sponge_step,
             )
 
@@ -2567,8 +2576,12 @@ def _build_jra55_block_fn_interp(model, jra55_state, dt, spmd_step=None):
             "spmd_step + prognostic sea ice is unsupported "
             "(run_omip_single refuses --jra55-sea-ice with "
             "--enable-latlon-spmd).")
-    _dyn_step = (spmd_step if spmd_step is not None
-                 else lambda st, d, **kw: model._step_impl(st, d, **kw))
+    if spmd_step is not None:
+        def _dyn_step(st, d, aux=None, **kw):
+            return spmd_step(st, d, aux=aux, **kw)
+    else:
+        def _dyn_step(st, d, aux=None, **kw):
+            return model._step_impl(st, d, **kw)
 
     lat_2d = jra55_state["lat_2d"]
     lon_2d = jra55_state["lon_2d"]
@@ -2582,7 +2595,7 @@ def _build_jra55_block_fn_interp(model, jra55_state, dt, spmd_step=None):
         @jax.jit
         def block_fn(state, raw_stack, runoff_records, record_days,
                      block_start_day, block_start_day_forcing,
-                     ice_state=None):
+                     ice_state=None, aux=None):
             dt_days = dt / 86400.0
 
             def step_body(carry, idx):
@@ -2719,7 +2732,7 @@ def _build_jra55_block_fn_interp(model, jra55_state, dt, spmd_step=None):
                 sponge_k = (sponge._replace(gamma=sponge.gamma * ramp)
                             if enable_sponge else None)
                 new_state = _dyn_step(
-                    state_in, dt, freshwater=fw,
+                    state_in, dt, aux=aux, freshwater=fw,
                     surface_forcing=sf, sponge=sponge_k,
                 )
 
@@ -3384,6 +3397,9 @@ def _run_omip_loop(model, state, grid_type, grid, z_coord, dt, n_steps,
         # (ocean_state, ice_state); thread the ice state across blocks.
         _ice_on = bool(jra55_state.get("enable_sea_ice", False))
         ice_state = jra55_state.get("ice_state_init") if _ice_on else None
+        # SPMD aux (codex r18 P1): pass the step's sharded geometry stacks
+        # into every block_fn call as an ARGUMENT (see the builders' note).
+        _spmd_aux = getattr(spmd_step, "aux", None)
         if use_gpu_interp:
             _get_block_fn_interp = _build_jra55_block_fn_interp(
                 model, jra55_state, dt, spmd_step=spmd_step)
@@ -3445,7 +3461,7 @@ def _run_omip_loop(model, state, grid_type, grid, z_coord, dt, n_steps,
                         record_meta["record_days"],
                         jnp.float64(record_meta["block_start_day"]),
                         jnp.float64(record_meta["block_start_day_forcing"]),
-                        ice_state,
+                        ice_state, aux=_spmd_aux,
                     )
                 else:
                     state = bfn(
@@ -3453,17 +3469,18 @@ def _run_omip_loop(model, state, grid_type, grid, z_coord, dt, n_steps,
                         record_meta["record_days"],
                         jnp.float64(record_meta["block_start_day"]),
                         jnp.float64(record_meta["block_start_day_forcing"]),
+                        aux=_spmd_aux,
                     )
             else:
                 if _ice_on:
                     state, ice_state = block_fn(
                         state, atm_stack, runoff_stack,
-                        jnp.int32(block_start), ice_state,
+                        jnp.int32(block_start), ice_state, aux=_spmd_aux,
                     )
                 else:
                     state = block_fn(
                         state, atm_stack, runoff_stack,
-                        jnp.int32(block_start),
+                        jnp.int32(block_start), aux=_spmd_aux,
                     )
             jax.block_until_ready(state.T.data)
             compute_dt = time.time() - t_compute_start
2000-    ``(n_steps, n_lat, n_lon)`` and ``runoff_stack`` is the per-step
2001-    friver field with the same shape.
2002-
2003-    Pure host-side I/O — runs once per block, then the JIT-compiled
2004:    block_fn consumes the result.
2005-    """
2006-    from legoesm.forcing.jra55_do import (
2007-        jra55_to_atm_surface,
2008-        load_jra55_block,
--
2299-
2300-    return raw_stack, runoff_stack, record_meta
2301-
2302-
2303:def _build_jra55_block_fn(model, jra55_state, dt, spmd_step=None):
2304-    """Return a JIT-compiled block function that runs N steps via lax.scan.
2305-
2306-    Captures everything that's static across the block (sponge, SSS
2307-    target, freeze-cap mask, coupler config, dt) in the closure so
--
2361-    # Lat-band SPMD (--enable-latlon-spmd): the scan body's dynamics step
2362-    # runs through the sharded wrapper (same forcing kwargs as _step_impl;
2363-    # the wrapper's cache/arm-restore Python runs ONCE at block trace).
2364-    # Sea ice is refused upstream (the ice tile is not SPMD-audited yet).
2365:    if spmd_step is not None and enable_sea_ice:
2366-        raise ValueError(
2367:            "spmd_step + prognostic sea ice is unsupported "
2368-            "(run_omip_single refuses --jra55-sea-ice with "
2369-            "--enable-latlon-spmd).")
2370-    # aux threading (codex r18 P1): the SPMD step's sharded geometry
2371-    # stacks must cross THIS jit boundary as an ARGUMENT — captured in the
2372-    # closure they become outer-trace constants whose value jax cannot
2373-    # fetch for non-addressable arrays on multicontroller (see
2374-    # make_sharded_ocean_step's aux note).
2375:    if spmd_step is not None:
2376:        def _dyn_step(st, d, aux=None, **kw):
2377:            return spmd_step(st, d, aux=aux, **kw)
2378-    else:
2379:        def _dyn_step(st, d, aux=None, **kw):
2380-            return model._step_impl(st, d, **kw)
2381-
2382-    @jax.jit
2383:    def block_fn(state, atm_stack, runoff_stack, block_start_step,
2384-                 ice_state=None, aux=None):
2385-        def step_body(carry, idx):
2386-            if enable_sea_ice:
2387-                state_in, ice_in = carry
--
2456-                sponge_step = sponge._replace(gamma=sponge.gamma * ramp)
2457-            else:
2458-                sponge_step = sponge
2459-
2460:            new_state = _dyn_step(
2461-                state_in, dt, aux=aux,
2462-                freshwater=fw, surface_forcing=sf, sponge=sponge_step,
2463-            )
2464-
--
2508-            step_body, init, jnp.arange(n, dtype=jnp.int32),
2509-        )
2510-        return final  # (state, ice_state) when sea-ice on, else state
2511-
2512:    return block_fn
2513-
2514-
2515:def _build_jra55_block_fn_interp(model, jra55_state, dt, spmd_step=None):
2516-    """JIT-compiled block function with GPU-side forcing interpolation.
2517-
2518:    Like ``_build_jra55_block_fn``, but instead of receiving pre-
2519-    interpolated per-step forcing, receives the native 3-hourly records
2520-    and computes the linear interpolation + solar zenith inside the
2521-    ``lax.scan`` body on GPU.  This reduces host-side I/O from N calls
2522-    to ``jra55_to_atm_surface`` (N=288 for 1 day) down to ~9 Zarr reads
--
2562-    else:
2563-        freeze_mask_static = None
2564-        T_freeze_C_static = -1.8
2565-
2566:    # Prognostic slab sea ice (opt-in) — see _build_jra55_block_fn.
2567-    enable_sea_ice = bool(jra55_state.get("enable_sea_ice", False))
2568-    ice_cfg = jra55_state.get("ice_config")
2569-
2570-    _maxvel_3d = model.config.barotropic.maxvel_barotropic
2571-    enable_maxvel = _maxvel_3d > 0.0
2572-
2573:    # Lat-band SPMD: see _build_jra55_block_fn.
2574:    if spmd_step is not None and enable_sea_ice:
2575-        raise ValueError(
2576:            "spmd_step + prognostic sea ice is unsupported "
2577-            "(run_omip_single refuses --jra55-sea-ice with "
2578-            "--enable-latlon-spmd).")
2579:    if spmd_step is not None:
2580:        def _dyn_step(st, d, aux=None, **kw):
2581:            return spmd_step(st, d, aux=aux, **kw)
2582-    else:
2583:        def _dyn_step(st, d, aux=None, **kw):
2584-            return model._step_impl(st, d, **kw)
2585-
2586-    lat_2d = jra55_state["lat_2d"]
2587-    lon_2d = jra55_state["lon_2d"]
--
2589-    # Static (compile-time) repeat-year-forcing flag.  Selects which clock
2590-    # drives the solar-zenith insolation geometry (see the scan body).
2591-    cycle = bool(jra55_state.get("cycle", False))
2592-
2593:    def _make_block_fn(n_steps_block):
2594-        """Create a JIT-compiled block function for a fixed block size."""
2595-        @jax.jit
2596:        def block_fn(state, raw_stack, runoff_records, record_days,
2597-                     block_start_day, block_start_day_forcing,
2598-                     ice_state=None, aux=None):
2599-            dt_days = dt / 86400.0
2600-
--
2730-                    )
2731-
2732-                sponge_k = (sponge._replace(gamma=sponge.gamma * ramp)
2733-                            if enable_sponge else None)
2734:                new_state = _dyn_step(
2735-                    state_in, dt, aux=aux, freshwater=fw,
2736-                    surface_forcing=sf, sponge=sponge_k,
2737-                )
2738-
--
2773-                jnp.arange(n_steps_block, dtype=jnp.int32),
2774-            )
2775-            return final  # (state, ice_state) when sea-ice on, else state
2776-
2777:        return block_fn
2778-
2779-    # Cache block functions by size to avoid recompilation.
2780:    _block_fn_cache = {}
2781-
2782:    def _get_block_fn(n):
2783:        if n not in _block_fn_cache:
2784:            _block_fn_cache[n] = _make_block_fn(n)
2785:        return _block_fn_cache[n]
2786-
2787:    return _get_block_fn
2788-
2789-
2790-# ===========================================================================
2791-# Diagnostics
--
3171-                   max_wallclock_seconds: float = 0.0,
3172-                   restart_buffer_seconds: float = 600.0,
3173-                   start_step=0,
3174-                   nudge_woa_tau=0.0, T_woa_3d=None, S_woa_3d=None,
3175:                   snapshot_fn=None, spmd_step=None, spmd_gather=None,
3176-                   spmd_shard_stack=None):
3177-    """Run time loop with diagnostics.
3178-
3179-    Two forcing paths, mutually exclusive:
3180-
3181-    * **restoring** (default): plain ``model.step(state, dt)`` followed
3182-      by Haney SST/SSS restoring when ``restoring_targets`` is set.
3183:      With ``spmd_step`` set (``--enable-latlon-spmd``), the dynamics
3184-      step runs through that lat-band-SPMD callable instead — the state
3185-      arrives sharded and every downstream op (restoring, finite checks,
3186-      diagnostics, restart saves) works on the sharded global arrays
3187-      transparently under the single-controller GSPMD runtime.
--
3213-        raise ValueError(
3214-            "_run_omip_loop: jra55_state and restoring_targets are mutually "
3215-            "exclusive — choose one forcing path."
3216-        )
3217:    if spmd_step is not None and jra55_state is not None:
3218:        # The block-scan lanes now thread spmd_step; the two unsupported
3219-        # JRA sub-modes still refuse loudly.
3220-        if jra55_state.get("_use_single_step", False):
3221-            raise ValueError(
3222:                "spmd_step + the JRA55 single-step fallback is unsupported "
3223-                "(_jra55_step calls model.step directly); use the "
3224-                "block-scan path (default).")
3225-        if jra55_state.get("enable_sea_ice", False):
3226-            raise ValueError(
3227:                "spmd_step + prognostic sea ice is unsupported "
3228-                "(--jra55-sea-ice; the ice tile is not SPMD-audited).")
3229-    if checkpoint_days is not None and checkpoint_dir is None:
3230-        raise ValueError(
3231-            "_run_omip_loop: checkpoint_days requires checkpoint_dir."
--
3389-        # a constant (codex review MAJOR; census 8474554).
3390-        # SPMD: the caller (run_omip_single) already primed the caches from
3391-        # the UNSHARDED state before sharding; re-priming here would run
3392-        # np.asarray on the sharded ``state`` — a hard non-addressable error
3393:        # under route-B (shards span processes).  Skip it when spmd_step is set.
3394:        if spmd_step is None:
3395-            model.prime_step_caches(state)
3396-        # Prognostic slab sea ice (--jra55-sea-ice): the block scan carries
3397-        # (ocean_state, ice_state); thread the ice state across blocks.
3398-        _ice_on = bool(jra55_state.get("enable_sea_ice", False))
3399-        ice_state = jra55_state.get("ice_state_init") if _ice_on else None
3400-        # SPMD aux (codex r18 P1): pass the step's sharded geometry stacks
3401:        # into every block_fn call as an ARGUMENT (see the builders' note).
3402:        _spmd_aux = getattr(spmd_step, "aux", None)
3403-        if use_gpu_interp:
3404:            _get_block_fn_interp = _build_jra55_block_fn_interp(
3405:                model, jra55_state, dt, spmd_step=spmd_step)
3406-            print("  GPU-interp mode: forcing interpolation on GPU")
3407-            # Pre-load the full JRA55 cache for repeat-year runs to
3408-            # eliminate per-block Zarr I/O (~0.3s/block → ~0s/block).
3409-            _full_cache = None
--
3411-                _fc_all, _fc_days, _fc_len = _preload_jra55_full_cache(
3412-                    jra55_state)
3413-                _full_cache = (_fc_all, _fc_days, _fc_len)
3414-        else:
3415:            block_fn = _build_jra55_block_fn(model, jra55_state, dt,
3416:                                             spmd_step=spmd_step)
3417-            _full_cache = None
3418-        block_size = max(1, diag_every)
3419-        if checkpoint_days is not None:
3420-            steps_per_ckpt = max(1, int(round(checkpoint_days * 86400.0 / dt)))
--
3453-            io_dt = time.time() - t_io_start
3454-
3455-            t_compute_start = time.time()
3456-            if use_gpu_interp:
3457:                bfn = _get_block_fn_interp(actual)
3458-                if _ice_on:
3459-                    state, ice_state = bfn(
3460-                        state, raw_stack, runoff_records,
3461-                        record_meta["record_days"],
3462-                        jnp.float64(record_meta["block_start_day"]),
3463-                        jnp.float64(record_meta["block_start_day_forcing"]),
3464:                        ice_state, aux=_spmd_aux,
3465-                    )
3466-                else:
3467-                    state = bfn(
3468-                        state, raw_stack, runoff_records,
3469-                        record_meta["record_days"],
3470-                        jnp.float64(record_meta["block_start_day"]),
3471-                        jnp.float64(record_meta["block_start_day_forcing"]),
3472:                        aux=_spmd_aux,
3473-                    )
3474-            else:
3475-                if _ice_on:
3476:                    state, ice_state = block_fn(
3477-                        state, atm_stack, runoff_stack,
3478:                        jnp.int32(block_start), ice_state, aux=_spmd_aux,
3479-                    )
3480-                else:
3481:                    state = block_fn(
3482-                        state, atm_stack, runoff_stack,
3483:                        jnp.int32(block_start), aux=_spmd_aux,
3484-                    )
3485-            jax.block_until_ready(state.T.data)
3486-            compute_dt = time.time() - t_compute_start
3487-
--
3675-    # to 0.
3676-    ramp_days = float(restoring_ramp_days)
3677-    restoring_ramp_steps = max(1, int(ramp_days * 86400.0 / dt)) if ramp_days > 0 else 1
3678-
3679:    _dyn_step = spmd_step if spmd_step is not None else model.step
3680-
3681-    for i in range(start_step, n_steps):
3682:        state = _dyn_step(state, dt)
3683-
3684-        # Apply SST/SSS restoring (grid-agnostic, after dynamics step)
3685-        if restoring_targets is not None:
3686-            T_tgt, S_tgt = restoring_targets
--
4604-        jra55_state["_ocean_mask_2d"] = state.land_mask.data > 0.5
4605-        # GPU-interp path (default): interpolation inside the lax.scan
4606-        # block for all grids (MPAS regridding is handled in
4607-        # _preload_jra55_raw_records).  --no-gpu-interp routes to the
4608:        # CPU-interp block path (_build_jra55_block_fn).
4609-        jra55_state["_gpu_interp"] = bool(args.gpu_interp)
4610-        # Restart: restore the prognostic sea-ice state so a checkpointed
4611-        # --jra55-sea-ice run does NOT resume on the zero cold-start ice.  An
4612-        # old ocean-only restart (no ice_* keys) returns None -> cold start
--
4784-    # the validated multi-GPU sharded ocean step and shard the state.
4785-    # Single-controller only; restoring lane only (the JRA55 block
4786-    # functions call model._step_impl directly — follow-up).  Runs AFTER
4787-    # the restart load so a resumed state is sharded too.
4788:    spmd_step = None
4789-    spmd_gather = None
4790-    spmd_shard_stack = None
4791-    if run_config.enable_latlon_spmd:
4792-        if grid_type != "latlon":
--
4836-            # Prime the build-once vertex-mask cache from the CONCRETE
4837-            # state so the wrapper can build per-band masks host-side.
4838-            model.prime_step_caches(state)
4839-            _dev = create_latlon_mesh(n_devices=_nd)
4840:            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
4841-            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
4842-            state = shard_state_latlon(state, _dev.mesh)
4843-            # Lay per-block forcing stacks out lat-band-sharded so the
4844-            # in-scan interpolation / bulk fluxes stay shard-local (shared
--
4872-            state.T.data.dtype) if args.nudge_woa_tau > 0 and T_woa is not None else None,
4873-        S_woa_3d=(S_woa * state.land_mask.data[..., jnp.newaxis]).astype(
4874-            state.S.data.dtype) if args.nudge_woa_tau > 0 and S_woa is not None else None,
4875-        snapshot_fn=_snapshot_fn,
4876:        spmd_step=spmd_step,
4877-        spmd_gather=spmd_gather,
4878-        spmd_shard_stack=spmd_shard_stack,
4879-    )
4880-    if spmd_gather is not None:

exec
/usr/bin/bash -lc "rg -n -C 8 \"def make_sharded_ocean_step|\\.aux|make_sharded_ocean_step|spmd_shard_stack\" scripts legoesm tests | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 432ms:
rg: legoesm: No such file or directory (os error 2)
tests/unit/test_run_omip_latlon_spmd.py-48-    assert jax.device_count() >= 2, (
tests/unit/test_run_omip_latlon_spmd.py-49-        f"worker expected 2 virtual devices, got {jax.device_count()}")
tests/unit/test_run_omip_latlon_spmd.py-50-
tests/unit/test_run_omip_latlon_spmd.py-51-    _day23 = _load_harness()
tests/unit/test_run_omip_latlon_spmd.py-52-    run_omip = _day23.run_omip
tests/unit/test_run_omip_latlon_spmd.py-53-
tests/unit/test_run_omip_latlon_spmd.py-54-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/unit/test_run_omip_latlon_spmd.py-55-        gather_state_latlon,
tests/unit/test_run_omip_latlon_spmd.py:56:        make_sharded_ocean_step,
tests/unit/test_run_omip_latlon_spmd.py-57-        shard_state_latlon,
tests/unit/test_run_omip_latlon_spmd.py-58-    )
tests/unit/test_run_omip_latlon_spmd.py-59-    from legoesm.parallel.mesh import create_latlon_mesh
tests/unit/test_run_omip_latlon_spmd.py-60-
tests/unit/test_run_omip_latlon_spmd.py-61-    tmp_path = Path(tmp_dir)
tests/unit/test_run_omip_latlon_spmd.py-62-    n_steps, dt = 6, 600.0
tests/unit/test_run_omip_latlon_spmd.py-63-    grid, z_coord, _config, model, _gt = _day23._make_tiny_latlon_setup(
tests/unit/test_run_omip_latlon_spmd.py-64-        n_lat=8, n_lon=16)
--
tests/unit/test_run_omip_latlon_spmd.py-74-    )
tests/unit/test_run_omip_latlon_spmd.py-75-
tests/unit/test_run_omip_latlon_spmd.py-76-    serial_final, _d, _w, ok_serial, _b = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-77-        model, state0, checkpoint_dir=tmp_path / "serial", **common)
tests/unit/test_run_omip_latlon_spmd.py-78-    assert ok_serial
tests/unit/test_run_omip_latlon_spmd.py-79-
tests/unit/test_run_omip_latlon_spmd.py-80-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-81-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py:82:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-83-    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-84-    ckpt_dir = tmp_path / "spmd"
tests/unit/test_run_omip_latlon_spmd.py-85-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-86-        model, ss0, checkpoint_dir=ckpt_dir,
tests/unit/test_run_omip_latlon_spmd.py-87-        spmd_step=spmd_step,
tests/unit/test_run_omip_latlon_spmd.py-88-        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
tests/unit/test_run_omip_latlon_spmd.py-89-        **common)
tests/unit/test_run_omip_latlon_spmd.py-90-    assert ok_spmd
--
tests/unit/test_run_omip_latlon_spmd.py-172-
tests/unit/test_run_omip_latlon_spmd.py-173-    _day23 = _load_harness()
tests/unit/test_run_omip_latlon_spmd.py-174-    run_omip = _day23.run_omip
tests/unit/test_run_omip_latlon_spmd.py-175-
tests/unit/test_run_omip_latlon_spmd.py-176-    from functools import partial
tests/unit/test_run_omip_latlon_spmd.py-177-
tests/unit/test_run_omip_latlon_spmd.py-178-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/unit/test_run_omip_latlon_spmd.py-179-        gather_state_latlon,
tests/unit/test_run_omip_latlon_spmd.py:180:        make_sharded_ocean_step,
tests/unit/test_run_omip_latlon_spmd.py-181-        shard_forcing_stack_latlon,
tests/unit/test_run_omip_latlon_spmd.py-182-        shard_state_latlon,
tests/unit/test_run_omip_latlon_spmd.py-183-    )
tests/unit/test_run_omip_latlon_spmd.py-184-    from legoesm.parallel.mesh import create_latlon_mesh
tests/unit/test_run_omip_latlon_spmd.py-185-
tests/unit/test_run_omip_latlon_spmd.py-186-    tmp_path = Path(tmp_dir)
tests/unit/test_run_omip_latlon_spmd.py-187-    n_lat, n_lon = 8, 16
tests/unit/test_run_omip_latlon_spmd.py-188-    dt, n_steps = 600.0, 6
--
tests/unit/test_run_omip_latlon_spmd.py-211-    )
tests/unit/test_run_omip_latlon_spmd.py-212-
tests/unit/test_run_omip_latlon_spmd.py-213-    state0 = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
tests/unit/test_run_omip_latlon_spmd.py-214-    serial_final, _d, _w, ok_serial, _b = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-215-        model, state0, jra55_state=_fresh_js(), **common)
tests/unit/test_run_omip_latlon_spmd.py-216-    assert ok_serial, "serial JRA55 loop reported not-ok"
tests/unit/test_run_omip_latlon_spmd.py-217-
tests/unit/test_run_omip_latlon_spmd.py-218-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py:219:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-220-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-221-    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-222-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-223-        model, ss0, jra55_state=_fresh_js(),
tests/unit/test_run_omip_latlon_spmd.py-224-        spmd_step=spmd_step,
tests/unit/test_run_omip_latlon_spmd.py-225-        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
tests/unit/test_run_omip_latlon_spmd.py:226:        spmd_shard_stack=partial(shard_forcing_stack_latlon, mesh=dev.mesh),
tests/unit/test_run_omip_latlon_spmd.py-227-        **common)
tests/unit/test_run_omip_latlon_spmd.py-228-    assert ok_spmd, "SPMD JRA55 loop reported not-ok"
tests/unit/test_run_omip_latlon_spmd.py-229-    spmd_final = gather_state_latlon(spmd_final, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-230-
tests/unit/test_run_omip_latlon_spmd.py-231-    # Same re-association floor as the restoring-lane and step-level gates.
tests/unit/test_run_omip_latlon_spmd.py-232-    for nm in ("u", "v", "eta", "T", "S"):
tests/unit/test_run_omip_latlon_spmd.py-233-        np.testing.assert_allclose(
tests/unit/test_run_omip_latlon_spmd.py-234-            np.asarray(getattr(spmd_final, nm).data),
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-1-"""Strong/weak scaling bench for the lat-band SPMD lat-lon C-grid OCEAN step.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-2-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-3-The ocean FULL-STEP twin of ``bench_atm_latlon_spmd_scaling.py`` (which this
scripts/bench/bench_ocean_latlon_spmd_scaling.py-4-mirrors flag-for-flag), closing the "no automated ocean full-step strong/weak
scripts/bench/bench_ocean_latlon_spmd_scaling.py-5-harness" gap: ``bench_ocean_mpi_scaling.py`` is the route-A (mpi4jax) phase-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-6-split bench and ``bench_ocean_latlon_spmd_pcg.py`` times the barotropic PCG
scripts/bench/bench_ocean_latlon_spmd_scaling.py-7-KERNEL only — neither times the composed production step
scripts/bench/bench_ocean_latlon_spmd_scaling.py:8:(``make_sharded_ocean_step``: baroclinic + barotropic [implicit-CN
scripts/bench/bench_ocean_latlon_spmd_scaling.py-9-free-surface by default, split-explicit via ``--baro-solver``] + implicit
scripts/bench/bench_ocean_latlon_spmd_scaling.py-10-vmix + tracers) under the lat-band SPMD backend.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-11-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-12-  strong: fixed (n_lat, n_lon, nlev), vary n_devices -> speedup = t(1)/t(n).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-13-  weak:   n_lat = nlat_per_dev * n_devices (fixed per-device rows) -> ideal flat.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-14-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-15-Device count is fixed at process start, so each n_devices runs as a SEPARATE
scripts/bench/bench_ocean_latlon_spmd_scaling.py-16-process (one sbatch step per count); this script benches ONE n_devices and
scripts/bench/bench_ocean_latlon_spmd_scaling.py-17-appends a JSON line. JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count
scripts/bench/bench_ocean_latlon_spmd_scaling.py-18-gives virtual CPU devices (communication-overhead characterization, NOT a real
scripts/bench/bench_ocean_latlon_spmd_scaling.py-19-speedup); a real number needs one GPU per band. Run with JAX_ENABLE_X64=1 (the
scripts/bench/bench_ocean_latlon_spmd_scaling.py-20-ocean step's validated precision lane).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-21-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-22-Multi-controller (route-B, ``--multicontroller``): identical contract to the
scripts/bench/bench_ocean_latlon_spmd_scaling.py-23-atm bench — every process calls ``jax.distributed.initialize`` BEFORE any
scripts/bench/bench_ocean_latlon_spmd_scaling.py-24-other JAX use, the ("lat",) mesh is built over the GLOBAL ``jax.devices()``,
scripts/bench/bench_ocean_latlon_spmd_scaling.py:25:and the existing ``make_sharded_ocean_step`` band-ppermute halo + psum
scripts/bench/bench_ocean_latlon_spmd_scaling.py-26-reductions (incl. the barotropic ``_global_sum_pair``) run unchanged across
scripts/bench/bench_ocean_latlon_spmd_scaling.py-27-processes (NCCL on GPU / gloo on CPU). NO mpi4jax is armed in this mode (the
scripts/bench/bench_ocean_latlon_spmd_scaling.py-28-documented mixed-stack deadlock hazard).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-29-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-30-Launch (cluster, one process per GPU):
scripts/bench/bench_ocean_latlon_spmd_scaling.py-31-  srun -n 8 python bench_ocean_latlon_spmd_scaling.py --multicontroller \
scripts/bench/bench_ocean_latlon_spmd_scaling.py-32-      --n-devices 8 ...            # SLURM: coordinator auto-detected
scripts/bench/bench_ocean_latlon_spmd_scaling.py-33-  mpiexec -n 8 python ... --multicontroller --coordinator host0:9876
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-338-    if args.multicontroller:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-339-        # MUST run before any other JAX use (backend init). Shared helper:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-340-        # explicit --coordinator -> OMPI/PALS launcher-env init; else
scripts/bench/bench_ocean_latlon_spmd_scaling.py-341-        # SLURM/OMPI auto-detect or PALS mpi4py bootstrap.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-342-        from legoesm.parallel.early_init import init_multicontroller_distributed
scripts/bench/bench_ocean_latlon_spmd_scaling.py-343-        init_multicontroller_distributed(args.coordinator)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-344-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-345-    from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/bench/bench_ocean_latlon_spmd_scaling.py:346:        make_sharded_ocean_step,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-347-        shard_state_latlon,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-348-    )
scripts/bench/bench_ocean_latlon_spmd_scaling.py-349-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-350-    nd = args.n_devices
scripts/bench/bench_ocean_latlon_spmd_scaling.py-351-    avail = len(jax.devices())
scripts/bench/bench_ocean_latlon_spmd_scaling.py-352-    if avail < nd:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-353-        raise SystemExit(f"need {nd} devices, have {avail} "
scripts/bench/bench_ocean_latlon_spmd_scaling.py-354-                         f"(set --xla_force_host_platform_device_count)")
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-442-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-443-    inv_before = None
scripts/bench/bench_ocean_latlon_spmd_scaling.py-444-    if args.check_conservation:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-445-        from bench_ocean_mpi_scaling import ocean_invariants
scripts/bench/bench_ocean_latlon_spmd_scaling.py-446-        inv_before = ocean_invariants(model, s0, n_ranks=1)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-447-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-448-    if nd == 1:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-449-        mesh = None
scripts/bench/bench_ocean_latlon_spmd_scaling.py:450:        step = make_sharded_ocean_step(model, None)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-451-        s = s0
scripts/bench/bench_ocean_latlon_spmd_scaling.py-452-    else:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-453-        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-454-                                 axis_names=("lat",))
scripts/bench/bench_ocean_latlon_spmd_scaling.py:455:        step = make_sharded_ocean_step(model, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-456-        s = shard_state_latlon(s0, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-457-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-458-    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
scripts/bench/bench_ocean_latlon_spmd_scaling.py-459-    # blocks with sync only AROUND the block — the previous per-step
scripts/bench/bench_ocean_latlon_spmd_scaling.py-460-    # host-synced loop measured dispatch+sync latency, not fused device
scripts/bench/bench_ocean_latlon_spmd_scaling.py-461-    # throughput.  Per-step dispatch latency is still measured, SEPARATELY,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-462-    # by an individually-synced probe (``step_latency_ms``); multi-controller
scripts/bench/bench_ocean_latlon_spmd_scaling.py-463-    # runs record the SLOWEST-process block time + imbalance ratio (a
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-468-    # one (args.steps - 1)-long block, no probe.  Timing from a parity smoke
scripts/bench/bench_ocean_latlon_spmd_scaling.py-469-    # run is not reported as a scaling number anyway (steps are capped).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-470-    if args.parity_gate:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-471-        _blk, _nblk, _probe = max(0, args.steps - 1), 1, 0
scripts/bench/bench_ocean_latlon_spmd_scaling.py-472-    else:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-473-        _blk, _nblk, _probe = args.steps, args.blocks, args.probe_steps
scripts/bench/bench_ocean_latlon_spmd_scaling.py-474-    # aux threads the sharded geometry stacks through the jit boundary as
scripts/bench/bench_ocean_latlon_spmd_scaling.py-475-    # an ARGUMENT (outer-trace constants of non-addressable arrays are
scripts/bench/bench_ocean_latlon_spmd_scaling.py:476:    # unfetchable — see make_sharded_ocean_step's aux note).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-477-    _aux = getattr(step, "aux", None)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-478-    s, timing = timed_scan_blocks(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-479-        (lambda st, aux: step(st, args.dt, aux=aux)) if _aux is not None
scripts/bench/bench_ocean_latlon_spmd_scaling.py-480-        else (lambda st: step(st, args.dt)),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-481-        s,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-482-        block_steps=_blk, n_blocks=_nblk, probe_steps=_probe,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-483-        sync_label="ocean_latlon_spmd_bench", aux=_aux)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-484-
--
tests/parallel/test_latlon_ocean_spmd_step.py-1-"""SPMD equivalence gate for the lat-lon C-grid ocean step (multi-node OMIP).
tests/parallel/test_latlon_ocean_spmd_step.py-2-
tests/parallel/test_latlon_ocean_spmd_step.py:3:This is the CORRECTNESS GATE for ``make_sharded_ocean_step`` (lat-band SPMD over
tests/parallel/test_latlon_ocean_spmd_step.py-4-the ``"lat"`` axis): N steps under ``shard_map`` on 4 (CPU) devices must match
tests/parallel/test_latlon_ocean_spmd_step.py-5-the single-device step to tolerance. It is the eORCA025 ¼° enabler (the ¼° grid
tests/parallel/test_latlon_ocean_spmd_step.py-6-OOMs on one 32 GiB GPU; lat-band sharding fits it at N>=5).
tests/parallel/test_latlon_ocean_spmd_step.py-7-
tests/parallel/test_latlon_ocean_spmd_step.py-8-STATUS (2026-06-16): the WRAPPER + halo layer are DONE and validated — the
tests/parallel/test_latlon_ocean_spmd_step.py-9-staggered-v band decomposition (``shard_state_latlon`` ↔ ``v_lower`` ↔ in-body
tests/parallel/test_latlon_ocean_spmd_step.py-10-ppermute reconstruction), the replicated-stacked grid indexed by
tests/parallel/test_latlon_ocean_spmd_step.py-11-``axis_index``, and the SPMD branches of ``pad_with_pole_bc_lat`` /
--
tests/parallel/test_latlon_ocean_spmd_step.py-77-        v=state.v.replace(data=jnp.asarray(v)),
tests/parallel/test_latlon_ocean_spmd_step.py-78-        eta=state.eta.replace(data=jnp.asarray(eta)),
tests/parallel/test_latlon_ocean_spmd_step.py-79-        T=state.T.replace(data=jnp.asarray(T)))
tests/parallel/test_latlon_ocean_spmd_step.py-80-
tests/parallel/test_latlon_ocean_spmd_step.py-81-
tests/parallel/test_latlon_ocean_spmd_step.py-82-def _have_sharded_step():
tests/parallel/test_latlon_ocean_spmd_step.py-83-    try:
tests/parallel/test_latlon_ocean_spmd_step.py-84-        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
tests/parallel/test_latlon_ocean_spmd_step.py:85:            make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py-86-        )
tests/parallel/test_latlon_ocean_spmd_step.py-87-        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
tests/parallel/test_latlon_ocean_spmd_step.py-88-        return True
tests/parallel/test_latlon_ocean_spmd_step.py-89-    except Exception:
tests/parallel/test_latlon_ocean_spmd_step.py-90-        return False
tests/parallel/test_latlon_ocean_spmd_step.py-91-
tests/parallel/test_latlon_ocean_spmd_step.py-92-
tests/parallel/test_latlon_ocean_spmd_step.py-93-@pytest.mark.skipif(jax.device_count() < 4,
tests/parallel/test_latlon_ocean_spmd_step.py-94-                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py-95-@pytest.mark.skipif(not _have_sharded_step(),
tests/parallel/test_latlon_ocean_spmd_step.py-96-                    reason="sharded_ocean_step module not present")
tests/parallel/test_latlon_ocean_spmd_step.py-97-def test_latlon_ocean_spmd_matches_single_device():
tests/parallel/test_latlon_ocean_spmd_step.py-98-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-99-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py:100:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py-101-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-102-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-103-    )
tests/parallel/test_latlon_ocean_spmd_step.py-104-
tests/parallel/test_latlon_ocean_spmd_step.py-105-    n_lat, n_lon, nlev = 48, 96, 10
tests/parallel/test_latlon_ocean_spmd_step.py-106-    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
tests/parallel/test_latlon_ocean_spmd_step.py-107-    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
tests/parallel/test_latlon_ocean_spmd_step.py-108-    cfg = LatLonCGridOceanConfig.from_flat()
--
tests/parallel/test_latlon_ocean_spmd_step.py-122-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-123-
tests/parallel/test_latlon_ocean_spmd_step.py-124-    # lat-band SPMD on 4 devices.  The state is laid out with shard_state_latlon
tests/parallel/test_latlon_ocean_spmd_step.py-125-    # (cell fields P("lat"); the staggered v / v_mask carried as v_lower, the
tests/parallel/test_latlon_ocean_spmd_step.py-126-    # n_lat-row block that DOES divide N — a uniform tree.map(P("lat")) would
tests/parallel/test_latlon_ocean_spmd_step.py-127-    # fail on the n_lat+1 v rows).  The result is gathered (and the dropped pole
tests/parallel/test_latlon_ocean_spmd_step.py-128-    # row reappended) for the bit-comparison vs the single-device reference.
tests/parallel/test_latlon_ocean_spmd_step.py-129-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:130:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-131-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-132-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-133-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_step.py-134-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-135-
tests/parallel/test_latlon_ocean_spmd_step.py-136-    # Tolerance = the FLOATING-POINT RE-ASSOCIATION floor of the sharded
tests/parallel/test_latlon_ocean_spmd_step.py-137-    # split-explicit barotropic, NOT a bug margin. A clean-input bisection proves
tests/parallel/test_latlon_ocean_spmd_step.py-138-    # every dynamical UNIT is SPMD-exact to round-off (baroclinic tendency ~5e-13,
--
tests/parallel/test_latlon_ocean_spmd_step.py-216-    relaxation, and the replicated ``t_seconds`` scalar.  Also flips the
tests/parallel/test_latlon_ocean_spmd_step.py-217-    same step callable between dynamics-only and forcing calls to pin the
tests/parallel/test_latlon_ocean_spmd_step.py-218-    forcing-aware compile-cache keying (stale specs would crash or
tests/parallel/test_latlon_ocean_spmd_step.py-219-    corrupt).
tests/parallel/test_latlon_ocean_spmd_step.py-220-    """
tests/parallel/test_latlon_ocean_spmd_step.py-221-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-222-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py-223-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py:224:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py-225-        shard_forcing_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-226-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-227-    )
tests/parallel/test_latlon_ocean_spmd_step.py-228-
tests/parallel/test_latlon_ocean_spmd_step.py-229-    n_lat, n_lon, nlev = 48, 96, 10
tests/parallel/test_latlon_ocean_spmd_step.py-230-    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
tests/parallel/test_latlon_ocean_spmd_step.py-231-    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
tests/parallel/test_latlon_ocean_spmd_step.py-232-    cfg = LatLonCGridOceanConfig.from_flat()._replace(
--
tests/parallel/test_latlon_ocean_spmd_step.py-239-    # single-device reference
tests/parallel/test_latlon_ocean_spmd_step.py-240-    s = state0
tests/parallel/test_latlon_ocean_spmd_step.py-241-    for i in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-242-        s = model.step(s, dt, freshwater=fw, surface_forcing=sf,
tests/parallel/test_latlon_ocean_spmd_step.py-243-                       sponge=sponge, t_seconds=jnp.asarray(i * dt))
tests/parallel/test_latlon_ocean_spmd_step.py-244-
tests/parallel/test_latlon_ocean_spmd_step.py-245-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-246-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:247:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-248-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-249-    fws = shard_forcing_latlon(fw, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-250-    sfs = shard_forcing_latlon(sf, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-251-    sponges = shard_forcing_latlon(sponge, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-252-
tests/parallel/test_latlon_ocean_spmd_step.py-253-    # Cache-key flip smoke: dynamics-only compile first, then the forcing
tests/parallel/test_latlon_ocean_spmd_step.py-254-    # program — the second call MUST rebuild (different forcing structure),
tests/parallel/test_latlon_ocean_spmd_step.py-255-    # not reuse the no-forcing specs.
--
tests/parallel/test_latlon_ocean_spmd_step.py-379-                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py-380-@pytest.mark.skipif(not _have_sharded_step(),
tests/parallel/test_latlon_ocean_spmd_step.py-381-                    reason="sharded_ocean_step module not present")
tests/parallel/test_latlon_ocean_spmd_step.py-382-def test_sharded_step_refuses_staggered_forcing():
tests/parallel/test_latlon_ocean_spmd_step.py-383-    """A (n_lat+1, n_lon) forcing leaf must fail LOUDLY at the wrapper, not
tests/parallel/test_latlon_ocean_spmd_step.py-384-    with an opaque shard_map divisibility error."""
tests/parallel/test_latlon_ocean_spmd_step.py-385-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-386-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py:387:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py-388-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-389-    )
tests/parallel/test_latlon_ocean_spmd_step.py-390-    from legoesm.ocean.state import OceanSurfaceForcing
tests/parallel/test_latlon_ocean_spmd_step.py-391-
tests/parallel/test_latlon_ocean_spmd_step.py-392-    grid = create_latlon_grid(n_lat=16, n_lon=32)
tests/parallel/test_latlon_ocean_spmd_step.py-393-    z_coord = create_ocean_z_star(n_levels=3, H_max=4000.0)
tests/parallel/test_latlon_ocean_spmd_step.py-394-    model = LatLonCGridOceanModel(grid, z_coord,
tests/parallel/test_latlon_ocean_spmd_step.py-395-                                  LatLonCGridOceanConfig.from_flat())
tests/parallel/test_latlon_ocean_spmd_step.py-396-    state0 = _perturbed_state(grid, z_coord)
tests/parallel/test_latlon_ocean_spmd_step.py-397-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-398-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:399:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-400-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-401-    bad = OceanSurfaceForcing(
tests/parallel/test_latlon_ocean_spmd_step.py-402-        tau_y=jnp.zeros((grid.n_lat + 1, grid.n_lon)))
tests/parallel/test_latlon_ocean_spmd_step.py-403-    with pytest.raises(ValueError, match="leading dim"):
tests/parallel/test_latlon_ocean_spmd_step.py-404-        step(ss, 600.0, surface_forcing=bad)
tests/parallel/test_latlon_ocean_spmd_step.py-405-
tests/parallel/test_latlon_ocean_spmd_step.py-406-
tests/parallel/test_latlon_ocean_spmd_step.py-407-@pytest.mark.skipif(jax.device_count() < 4,
tests/parallel/test_latlon_ocean_spmd_step.py-408-                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py-409-@pytest.mark.skipif(not _have_sharded_step(),
tests/parallel/test_latlon_ocean_spmd_step.py-410-                    reason="sharded_ocean_step module not present")
tests/parallel/test_latlon_ocean_spmd_step.py-411-def test_sharded_ocean_step_global_matches_explicit_scatter_gather():
tests/parallel/test_latlon_ocean_spmd_step.py:412:    """``make_sharded_ocean_step_global`` (global-in/global-out, the minimal
tests/parallel/test_latlon_ocean_spmd_step.py-413-    driver entry) must equal the explicit ``shard_state_latlon`` -> inner step ->
tests/parallel/test_latlon_ocean_spmd_step.py-414-    ``gather_state_latlon`` path BIT-FOR-BIT (it is literally that composition),
tests/parallel/test_latlon_ocean_spmd_step.py-415-    AND match the single-device reference to the same re-association floor.
tests/parallel/test_latlon_ocean_spmd_step.py-416-
tests/parallel/test_latlon_ocean_spmd_step.py-417-    Also exercises the WITH-forcing path through the global wrapper (a smooth
tests/parallel/test_latlon_ocean_spmd_step.py-418-    cell-shaped wind-stress + heat ``OceanSurfaceForcing``)."""
tests/parallel/test_latlon_ocean_spmd_step.py-419-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-420-    from legoesm.grids.latlon import ensure_geometry
tests/parallel/test_latlon_ocean_spmd_step.py-421-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py:422:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py:423:        make_sharded_ocean_step_global,
tests/parallel/test_latlon_ocean_spmd_step.py-424-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-425-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-426-    )
tests/parallel/test_latlon_ocean_spmd_step.py-427-    from legoesm.ocean.state import OceanSurfaceForcing
tests/parallel/test_latlon_ocean_spmd_step.py-428-
tests/parallel/test_latlon_ocean_spmd_step.py-429-    n_lat, n_lon, nlev = 48, 96, 10
tests/parallel/test_latlon_ocean_spmd_step.py-430-    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
tests/parallel/test_latlon_ocean_spmd_step.py-431-    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
--
tests/parallel/test_latlon_ocean_spmd_step.py-446-    s = state0
tests/parallel/test_latlon_ocean_spmd_step.py-447-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-448-        s = model.step(s, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_step.py-449-
tests/parallel/test_latlon_ocean_spmd_step.py-450-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-451-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py-452-
tests/parallel/test_latlon_ocean_spmd_step.py-453-    # explicit scatter -> inner sharded step -> gather
tests/parallel/test_latlon_ocean_spmd_step.py:454:    inner = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-455-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-456-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-457-        ss = inner(ss, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_step.py-458-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-459-
tests/parallel/test_latlon_ocean_spmd_step.py-460-    # global-in/global-out wrapper (scatter + gather PER STEP)
tests/parallel/test_latlon_ocean_spmd_step.py:461:    glob = make_sharded_ocean_step_global(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-462-    sg = state0
tests/parallel/test_latlon_ocean_spmd_step.py-463-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-464-        sg = glob(sg, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_step.py-465-
tests/parallel/test_latlon_ocean_spmd_step.py-466-    for nm in ("u", "v", "eta", "T", "S"):
tests/parallel/test_latlon_ocean_spmd_step.py-467-        ref = np.asarray(getattr(s, nm).data)
tests/parallel/test_latlon_ocean_spmd_step.py-468-        man = np.asarray(getattr(ss, nm).data)
tests/parallel/test_latlon_ocean_spmd_step.py-469-        wrp = np.asarray(getattr(sg, nm).data)
--
tests/parallel/test_latlon_spmd_fused_halo.py-316-    collective-permutes with fusion on (the aggregation actually fires on
tests/parallel/test_latlon_spmd_fused_halo.py-317-    the production pad_multi sites)."""
tests/parallel/test_latlon_spmd_fused_halo.py-318-    from legoesm.grids.latlon import create_latlon_grid
tests/parallel/test_latlon_spmd_fused_halo.py-319-    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
tests/parallel/test_latlon_spmd_fused_halo.py-320-        LatLonCGridOceanModel,
tests/parallel/test_latlon_spmd_fused_halo.py-321-    )
tests/parallel/test_latlon_spmd_fused_halo.py-322-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_spmd_fused_halo.py-323-        gather_state_latlon,
tests/parallel/test_latlon_spmd_fused_halo.py:324:        make_sharded_ocean_step,
tests/parallel/test_latlon_spmd_fused_halo.py-325-        shard_state_latlon,
tests/parallel/test_latlon_spmd_fused_halo.py-326-    )
tests/parallel/test_latlon_spmd_fused_halo.py-327-    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
tests/parallel/test_latlon_spmd_fused_halo.py-328-    from legoesm.ocean.state import LatLonCGridOceanConfig
tests/parallel/test_latlon_spmd_fused_halo.py-329-    from legoesm.ocean.vertical import create_ocean_z_star
tests/parallel/test_latlon_spmd_fused_halo.py-330-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_spmd_fused_halo.py-331-
tests/parallel/test_latlon_spmd_fused_halo.py-332-    grid = create_latlon_grid(n_lat=48, n_lon=96)
--
tests/parallel/test_latlon_spmd_fused_halo.py-341-    state0 = state0._replace(eta=state0.eta.replace(data=jnp.asarray(eta)))
tests/parallel/test_latlon_spmd_fused_halo.py-342-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_spmd_fused_halo.py-343-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_spmd_fused_halo.py-344-
tests/parallel/test_latlon_spmd_fused_halo.py-345-    outs = {}
tests/parallel/test_latlon_spmd_fused_halo.py-346-    hlo_counts = {}
tests/parallel/test_latlon_spmd_fused_halo.py-347-    for flag in ("0", "1"):
tests/parallel/test_latlon_spmd_fused_halo.py-348-        _env_flag(monkeypatch, flag)
tests/parallel/test_latlon_spmd_fused_halo.py:349:        step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-350-        ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-351-        hlo_counts[flag] = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-352-            jax.jit(step).lower(ss, 600.0).compile().as_text())
tests/parallel/test_latlon_spmd_fused_halo.py-353-        for _ in range(n_steps):
tests/parallel/test_latlon_spmd_fused_halo.py-354-            ss = step(ss, 600.0)
tests/parallel/test_latlon_spmd_fused_halo.py-355-        outs[flag] = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-356-
tests/parallel/test_latlon_spmd_fused_halo.py-357-    for nm in ("eta", "u", "v", "T", "S"):
--
tests/parallel/test_latlon_spmd_fused_halo.py-364-    # Flag flip on a REUSED step object must rebuild the shard_map (the
tests/parallel/test_latlon_spmd_fused_halo.py-365-    # switch is trace-time; the step wrapper's per-CALL cache key carries
tests/parallel/test_latlon_spmd_fused_halo.py-366-    # it — codex).  Fresh outer lambdas per lowering: jax.jit caches on
tests/parallel/test_latlon_spmd_fused_halo.py-367-    # the callable identity, so re-jitting the SAME step object would
tests/parallel/test_latlon_spmd_fused_halo.py-368-    # freeze the wrapper's python body and never re-evaluate the key
tests/parallel/test_latlon_spmd_fused_halo.py-369-    # (an outer-jit artifact, not the production call pattern — the
tests/parallel/test_latlon_spmd_fused_halo.py-370-    # driver calls step() directly each step).
tests/parallel/test_latlon_spmd_fused_halo.py-371-    _env_flag(monkeypatch, "0")
tests/parallel/test_latlon_spmd_fused_halo.py:372:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-373-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-374-    n_off = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-375-        jax.jit(lambda s, d: step(s, d)).lower(ss, 600.0)
tests/parallel/test_latlon_spmd_fused_halo.py-376-        .compile().as_text())
tests/parallel/test_latlon_spmd_fused_halo.py-377-    _env_flag(monkeypatch, "1")
tests/parallel/test_latlon_spmd_fused_halo.py-378-    n_on = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-379-        jax.jit(lambda s, d: step(s, d)).lower(ss, 600.0)
tests/parallel/test_latlon_spmd_fused_halo.py-380-        .compile().as_text())
--
tests/parallel/test_persistent_sharded_ocean_loop.py-1-"""Parity gate for the PERSISTENT lat-band-sharded OMIP ocean loop
tests/parallel/test_persistent_sharded_ocean_loop.py-2-(scaling-M2 increment 1, ``run_omip_core2 --spmd-persistent-state``).
tests/parallel/test_persistent_sharded_ocean_loop.py-3-
tests/parallel/test_persistent_sharded_ocean_loop.py-4-The production multi-GPU OMIP host loop used to call
tests/parallel/test_persistent_sharded_ocean_loop.py:5:``make_sharded_ocean_step_global`` EVERY step — a full-state scatter
tests/parallel/test_persistent_sharded_ocean_loop.py-6-(``shard_state_latlon``) + gather (``gather_state_latlon``) per step, i.e.
tests/parallel/test_persistent_sharded_ocean_loop.py-7-2 full-state transfers/step.  The persistent lane instead keeps the state
tests/parallel/test_persistent_sharded_ocean_loop.py:8:lat-band SHARDED across steps via ``make_sharded_ocean_step`` and gathers
tests/parallel/test_persistent_sharded_ocean_loop.py-9-ONLY at real output boundaries.  This gate asserts, over a 10-step loop with
tests/parallel/test_persistent_sharded_ocean_loop.py-10-production-shaped surface + freshwater forcing (``normalize_freshwater=True``
tests/parallel/test_persistent_sharded_ocean_loop.py-11-so the in-step psum reduction is load-bearing):
tests/parallel/test_persistent_sharded_ocean_loop.py-12-
tests/parallel/test_persistent_sharded_ocean_loop.py-13-* EQUIVALENCE — the persistent loop's final state equals the per-step
tests/parallel/test_persistent_sharded_ocean_loop.py-14-  global-wrapper loop's BIT-FOR-BIT (atol=rtol=0; the wrapper is literally
tests/parallel/test_persistent_sharded_ocean_loop.py-15-  scatter->inner->gather and every band stays on its device, so the per-step
tests/parallel/test_persistent_sharded_ocean_loop.py-16-  round trip is value-preserving).  Far inside the 1e-12 f64 merge bar.
--
tests/parallel/test_persistent_sharded_ocean_loop.py-69-    sys.modules["_latlon_spmd_gate_sibling"] = mod
tests/parallel/test_persistent_sharded_ocean_loop.py-70-    spec.loader.exec_module(mod)
tests/parallel/test_persistent_sharded_ocean_loop.py-71-    return mod
tests/parallel/test_persistent_sharded_ocean_loop.py-72-
tests/parallel/test_persistent_sharded_ocean_loop.py-73-
tests/parallel/test_persistent_sharded_ocean_loop.py-74-def _have_sharded_step():
tests/parallel/test_persistent_sharded_ocean_loop.py-75-    try:
tests/parallel/test_persistent_sharded_ocean_loop.py-76-        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
tests/parallel/test_persistent_sharded_ocean_loop.py:77:            make_sharded_ocean_step,
tests/parallel/test_persistent_sharded_ocean_loop.py-78-        )
tests/parallel/test_persistent_sharded_ocean_loop.py-79-        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
tests/parallel/test_persistent_sharded_ocean_loop.py-80-        return True
tests/parallel/test_persistent_sharded_ocean_loop.py-81-    except Exception:
tests/parallel/test_persistent_sharded_ocean_loop.py-82-        return False
tests/parallel/test_persistent_sharded_ocean_loop.py-83-
tests/parallel/test_persistent_sharded_ocean_loop.py-84-
tests/parallel/test_persistent_sharded_ocean_loop.py-85-@pytest.mark.skipif(jax.device_count() < 2,
--
tests/parallel/test_persistent_sharded_ocean_loop.py-165-        u_sfc, v_sfc = _surface_currents(st, grid, "latlon")
tests/parallel/test_persistent_sharded_ocean_loop.py-166-        assert u_sfc.shape == (n_lat, n_lon)
tests/parallel/test_persistent_sharded_ocean_loop.py-167-        assert v_sfc.shape == (n_lat, n_lon)
tests/parallel/test_persistent_sharded_ocean_loop.py-168-        return (sf._replace(tau_x=sf.tau_x + 5.0e-3 * u_sfc,
tests/parallel/test_persistent_sharded_ocean_loop.py-169-                            tau_y=sf.tau_y + 5.0e-3 * v_sfc),
tests/parallel/test_persistent_sharded_ocean_loop.py-170-                u_sfc, v_sfc)
tests/parallel/test_persistent_sharded_ocean_loop.py-171-
tests/parallel/test_persistent_sharded_ocean_loop.py-172-    # ---------------- OLD lane: per-step global-in/global-out wrapper --------
tests/parallel/test_persistent_sharded_ocean_loop.py:173:    glob = sos.make_sharded_ocean_step_global(model, mesh)
tests/parallel/test_persistent_sharded_ocean_loop.py-174-    sg = state0
tests/parallel/test_persistent_sharded_ocean_loop.py-175-    cur_probe = {}
tests/parallel/test_persistent_sharded_ocean_loop.py-176-    for k in range(1, n_steps + 1):
tests/parallel/test_persistent_sharded_ocean_loop.py-177-        # surface-current consumer on the GLOBAL state (full staggered v)
tests/parallel/test_persistent_sharded_ocean_loop.py-178-        assert sg.v.data.shape[0] == n_lat + 1
tests/parallel/test_persistent_sharded_ocean_loop.py-179-        sf_k, u_sfc, v_sfc = _sf_with_current_feedback(sg)
tests/parallel/test_persistent_sharded_ocean_loop.py-180-        if k == cur_probe_step:
tests/parallel/test_persistent_sharded_ocean_loop.py-181-            cur_probe["u"] = np.asarray(u_sfc).copy()
--
tests/parallel/test_persistent_sharded_ocean_loop.py-193-            sg = _ice_thermo_touch_T(sg)
tests/parallel/test_persistent_sharded_ocean_loop.py-194-    sg = jax.block_until_ready(sg)
tests/parallel/test_persistent_sharded_ocean_loop.py-195-    old_calls = dict(calls)
tests/parallel/test_persistent_sharded_ocean_loop.py-196-    # the wrapper lane pays 2 full-state transfers PER STEP
tests/parallel/test_persistent_sharded_ocean_loop.py-197-    assert old_calls == {"shard": n_steps, "gather": n_steps}, old_calls
tests/parallel/test_persistent_sharded_ocean_loop.py-198-
tests/parallel/test_persistent_sharded_ocean_loop.py-199-    # ---------------- NEW lane: persistent sharded loop ----------------------
tests/parallel/test_persistent_sharded_ocean_loop.py-200-    calls["shard"] = calls["gather"] = 0
tests/parallel/test_persistent_sharded_ocean_loop.py:201:    inner = sos.make_sharded_ocean_step(model, mesh)
tests/parallel/test_persistent_sharded_ocean_loop.py-202-    ss = sos.shard_state_latlon(state0, mesh)          # ONE initial shard
tests/parallel/test_persistent_sharded_ocean_loop.py-203-    for k in range(1, n_steps + 1):
tests/parallel/test_persistent_sharded_ocean_loop.py-204-        # surface-current consumer on the SHARDED state: v is the n_lat-row
tests/parallel/test_persistent_sharded_ocean_loop.py-205-        # v_lower carrier at every loop top (the snapshot boundary re-shards
tests/parallel/test_persistent_sharded_ocean_loop.py-206-        # back to it); _surface_currents reconstructs the staggered top row
tests/parallel/test_persistent_sharded_ocean_loop.py-207-        # as the wall zero DEVICE-SIDE — the gather counters below prove no
tests/parallel/test_persistent_sharded_ocean_loop.py-208-        # full-state layout flip happens here.
tests/parallel/test_persistent_sharded_ocean_loop.py-209-        assert ss.v.data.shape[0] == n_lat        # carrier layout, not n_lat+1
--
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-1-"""Route-B multi-controller gate for the lat-band SPMD ocean step.
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-2-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-3-The ocean twin of ``test_atm_latlon_spmd_multicontroller.py``: N processes
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-4-federate via ``jax.distributed`` into ONE multi-controller program, the
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:5:("lat",) mesh spans the GLOBAL device set, and ``make_sharded_ocean_step`` +
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-6-the band ppermute/psum halo (incl. the barotropic ``_global_sum_pair`` psum)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-7-run UNCHANGED — collectives cross processes via the distributed runtime
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-8-(NCCL/gloo). NO mpi4jax anywhere in this file (mixed-stack deadlock hazard),
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-9-which is also why it lives in ``tests/parallel/``, NOT ``tests/distributed/``
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-10-(whose session conftest auto-arms the mpi4jax layout).
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-11-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-12-GATED: skips unless ``LEGOESM_JAX_DISTRIBUTED_TEST=1`` — ``jax.distributed
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-13-.initialize`` must run BEFORE the first backend touch, so this file must be
--
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-64-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-65-import numpy as np  # noqa: E402
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-66-from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-67-from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-68-    LatLonCGridOceanModel,
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-69-)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-70-from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: E402
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-71-    gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:72:    make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-73-    shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-74-)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-75-from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-76-from legoesm.ocean.vertical import create_ocean_z_star  # noqa: E402
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-77-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-78-# Reuse the single-process gate's perturbed-IC fixture — same conventions.
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-79-from tests.parallel.test_latlon_ocean_spmd_step import (  # noqa: E402
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-80-    _perturbed_state,
--
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-117-    # Serial reference: every process computes it from the identical host IC.
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-118-    s = state0
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-119-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-120-        s = model.step(s, dt)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-121-    # Prime the build-once vertex-mask cache from the CONCRETE state (the
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-122-    # serial run above already did; belt-and-braces for wrapper band masks).
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-124-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:125:    step = make_sharded_ocean_step(model, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-126-    ss = shard_state_latlon(state0, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-127-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-128-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-129-    out = gather_state_latlon(ss, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-130-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-131-    # Tolerance = the split-explicit FP re-association floor of the sharded
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-132-    # barotropic (see test_latlon_ocean_spmd_step's tolerance note verbatim).
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-133-    atol, rtol = 2.0e-4, 1.0e-3
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-1-"""SPMD equivalence + collective-count gate for the WIDE-HALO barotropic.
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-2-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-3-Twin of ``test_latlon_ocean_spmd_step.py`` with
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:4:``barotropic_wide_halo=True``: N steps of ``make_sharded_ocean_step`` across
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-5-4 (CPU) devices must match the single-device wide-halo step at the sharded
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-6-split-explicit re-association floor.  A staggered off-by-one in the wide
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-7-v-exchange, a reach under-budget at a band cut, or a pole-flag error in the
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-8-extended geometry shows up here as an O(1e-3+) band-cut mismatch.
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-9-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-10-Also gates the POINT of the wide path mechanically: the compiled sharded
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-11-step's WHILE-BODY ``collective-permute`` count must DROP vs the standard
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-12-config — per-iteration subcycle communication is exactly what the wide
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-68-    # (the wide subcycle forces local per-substep clamping by contract).
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-69-    flat.setdefault("barotropic_local_subcycle_clamp", True)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-70-    return LatLonCGridOceanConfig.from_flat(**flat)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-71-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-72-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-73-def _run_pair(cfg, n_lat=48, n_lon=96, nlev=10, dt=600.0, n_steps=3):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-74-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-75-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:76:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-77-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-78-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-79-    )
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-80-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-81-    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-82-    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-83-    model = LatLonCGridOceanModel(grid, z_coord, cfg)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-84-    state0 = _perturbed_state(grid, z_coord)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-85-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-86-    s = state0
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-87-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-88-        s = model.step(s, dt)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-89-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-90-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-91-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:92:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-93-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-94-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-95-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-96-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-97-    return s, ss
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-98-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-99-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-100-@pytest.mark.parametrize("chunk", [0, 2])
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-107-        np.testing.assert_allclose(
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-108-            b, a, atol=_ATOL, rtol=_RTOL,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-109-            err_msg=f"wide-halo SPMD {nm} mismatch (chunk={chunk})")
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-110-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-111-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-112-def _sharded_step_hlo(cfg, n_lat=48, n_lon=96, nlev=10):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-113-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-114-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:115:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-116-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-117-    )
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-118-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-119-    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-120-    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-121-    model = LatLonCGridOceanModel(grid, z_coord, cfg)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-122-    state0 = _perturbed_state(grid, z_coord)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-124-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:125:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-126-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-127-    lowered = jax.jit(step).lower(ss, 600.0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-128-    return lowered.compile().as_text()
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-129-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-130-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-131-def _while_body_collective_count(hlo_text: str) -> int:
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-132-    """Collective-permute ops inside while-LOOP BODY computations.
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-133-
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-1-"""SPMD equivalence gate for the lat-lon C-grid ocean step on a TRIPOLE grid.
tests/parallel/test_latlon_ocean_spmd_tripole.py-2-
tests/parallel/test_latlon_ocean_spmd_tripole.py-3-The tripole twin of ``test_latlon_ocean_spmd_step.py`` (regular grid): N steps of
tests/parallel/test_latlon_ocean_spmd_tripole.py:4:``make_sharded_ocean_step`` on a synthetic tripole (active bipolar north fold)
tests/parallel/test_latlon_ocean_spmd_tripole.py-5-across 4 (CPU) devices must match the single-device step to the same
tests/parallel/test_latlon_ocean_spmd_tripole.py-6-floating-point re-association tolerance.  This is the CORRECTNESS GATE for the
tests/parallel/test_latlon_ocean_spmd_tripole.py-7-tripole north fold under SPMD (the data-dependent ``north_fold_mask`` /
tests/parallel/test_latlon_ocean_spmd_tripole.py-8-``apply_north_fold`` conversion of the ~15 inline operator fold overwrites + the
tests/parallel/test_latlon_ocean_spmd_tripole.py-9-staggered-v north boundary): the single shard_map trace runs on every band, so
tests/parallel/test_latlon_ocean_spmd_tripole.py-10-the serial ``if fold_is_local`` overwrite is uniformly False — the fold is
tests/parallel/test_latlon_ocean_spmd_tripole.py-11-selected on the north band (``axis_index("lat")==N-1``).  A missed fold path
tests/parallel/test_latlon_ocean_spmd_tripole.py-12-(an operator still serial-only, or the v top-row reconstructed as a pole wall
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-80-        tau_y=jnp.zeros_like(jnp.asarray(tau_x)),
tests/parallel/test_latlon_ocean_spmd_tripole.py-81-        q_net=jnp.asarray(q_net),
tests/parallel/test_latlon_ocean_spmd_tripole.py-82-    )
tests/parallel/test_latlon_ocean_spmd_tripole.py-83-
tests/parallel/test_latlon_ocean_spmd_tripole.py-84-
tests/parallel/test_latlon_ocean_spmd_tripole.py-85-def _have_sharded_step():
tests/parallel/test_latlon_ocean_spmd_tripole.py-86-    try:
tests/parallel/test_latlon_ocean_spmd_tripole.py-87-        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
tests/parallel/test_latlon_ocean_spmd_tripole.py:88:            make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_tripole.py-89-        )
tests/parallel/test_latlon_ocean_spmd_tripole.py-90-        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
tests/parallel/test_latlon_ocean_spmd_tripole.py-91-        return True
tests/parallel/test_latlon_ocean_spmd_tripole.py-92-    except Exception:
tests/parallel/test_latlon_ocean_spmd_tripole.py-93-        return False
tests/parallel/test_latlon_ocean_spmd_tripole.py-94-
tests/parallel/test_latlon_ocean_spmd_tripole.py-95-
tests/parallel/test_latlon_ocean_spmd_tripole.py-96-@pytest.mark.skipif(jax.device_count() < 4,
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-107-    # implicit_cn is the OMIP production solver (cold-start-stable).  Under SPMD
tests/parallel/test_latlon_ocean_spmd_tripole.py-108-    # implicit_cn routes to the FIXED-iteration distributed PCG (static scan
tests/parallel/test_latlon_ocean_spmd_tripole.py-109-    # schedule) — barotropic_implicit_latlon_cgrid keys _use_pcg on the armed
tests/parallel/test_latlon_ocean_spmd_tripole.py-110-    # spmd halo backend — instead of the jax.scipy.cg while_loop (whose
tests/parallel/test_latlon_ocean_spmd_tripole.py-111-    # collectives SIGABRT under shard_map).  Both must match the single-device
tests/parallel/test_latlon_ocean_spmd_tripole.py-112-    # step on a tripole (north fold + barotropic solve).
tests/parallel/test_latlon_ocean_spmd_tripole.py-113-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_tripole.py-114-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_tripole.py:115:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_tripole.py-116-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_tripole.py-117-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_tripole.py-118-    )
tests/parallel/test_latlon_ocean_spmd_tripole.py-119-
tests/parallel/test_latlon_ocean_spmd_tripole.py-120-    n_lat, n_lon, nlev = 48, 96, 10          # n_lat % 4 == 0 (band-divisible)
tests/parallel/test_latlon_ocean_spmd_tripole.py-121-    grid = create_synthetic_tripole(n_lat, n_lon)
tests/parallel/test_latlon_ocean_spmd_tripole.py-122-    assert grid.fold.is_active, "synthetic tripole must carry an active fold"
tests/parallel/test_latlon_ocean_spmd_tripole.py-123-    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-144-    # single-device reference
tests/parallel/test_latlon_ocean_spmd_tripole.py-145-    s = state0
tests/parallel/test_latlon_ocean_spmd_tripole.py-146-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_tripole.py-147-        s = model.step(s, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_tripole.py-148-
tests/parallel/test_latlon_ocean_spmd_tripole.py-149-    model._ensure_vertex_mask(state0)        # prime the build-once vmask cache
tests/parallel/test_latlon_ocean_spmd_tripole.py-150-
tests/parallel/test_latlon_ocean_spmd_tripole.py-151-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_tripole.py:152:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-153-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-154-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_tripole.py-155-        ss = step(ss, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_tripole.py-156-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-157-
tests/parallel/test_latlon_ocean_spmd_tripole.py-158-    # FP re-association floor (NOT a bug margin): the sharded reductions are
tests/parallel/test_latlon_ocean_spmd_tripole.py-159-    # batch_psum_spmd across bands (provably global — a missing/local reduction
tests/parallel/test_latlon_ocean_spmd_tripole.py-160-    # gives O(1) error, a missing band-cut halo O(1e-2)), so the only
--
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-8-#PBS -j oe
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-9-#PBS -k eod
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-10-# ===========================================================================
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-11-# OCEAN weak+strong GPU scaling on a Derecho GPU node (4x A100-40GB, NVLink).
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-12-#
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-13-# Runs the SINGLE-PROCESS multi-device SPMD lane of
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-14-# scripts/bench/bench_ocean_latlon_spmd_scaling.py: the FULL
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-15-# lat-lon C-grid ocean step sharded over 1/2/4 A100s via
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs:16:# make_sharded_ocean_step (shard_map ppermute band halos + psum reductions
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-17-# over NVLink/NCCL; NO mpi4jax — uses the plain `legoesm-gpu` env from
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-18-# README Step 1, no route-A overlay needed).
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-19-#
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-20-# Correctness gates: tests/parallel/test_latlon_ocean_spmd_step.py +
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-21-# tests/bench/test_bench_ocean_latlon_spmd_gates.py; the job
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-22-# additionally runs one smoke case with --parity-gate + --check-conservation
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-23-# armed before the timed ladder (fail fast on a broken build).
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-24-#
--
scripts/run/run_omip_core2.py-3321-                        "--mpas-level).")
scripts/run/run_omip_core2.py-3322-    p.add_argument("--latlon-res", type=str, default="180x360",
scripts/run/run_omip_core2.py-3323-                   help="lat-lon resolution NxM for --grid latlon_bathy.")
scripts/run/run_omip_core2.py-3324-    p.add_argument("--cube-n", type=int, default=48,
scripts/run/run_omip_core2.py-3325-                   help="cubed-sphere face resolution n (C-n) for --grid cubed_sphere.")
scripts/run/run_omip_core2.py-3326-    p.add_argument("--n-gpus", type=int, default=1,
scripts/run/run_omip_core2.py-3327-                   help="Multi-GPU lat-band SPMD ocean step (latlon_bathy / tripole "
scripts/run/run_omip_core2.py-3328-                        "only): partition the ocean state by latitude band across N "
scripts/run/run_omip_core2.py:3329:                        "local devices via make_sharded_ocean_step_global. n_lat is "
scripts/run/run_omip_core2.py-3330-                        "padded with LAND rows at the SOUTH to a multiple of N (the "
scripts/run/run_omip_core2.py-3331-                        "tripole north fold stays at the north). The host post-step "
scripts/run/run_omip_core2.py-3332-                        "BCs (SSS restore / prognostic ice / geothermal / BBL / nudge) "
scripts/run/run_omip_core2.py-3333-                        "run on the gathered GLOBAL state, unchanged. Default 1 = the "
scripts/run/run_omip_core2.py-3334-                        "single-device path (byte-identical). Single-process: N must be "
scripts/run/run_omip_core2.py-3335-                        "<= jax.local_device_count(). MULTI-NODE: add --distributed and "
scripts/run/run_omip_core2.py-3336-                        "launch one process per GPU (mpirun/srun); then N must be <= "
scripts/run/run_omip_core2.py-3337-                        "jax.device_count() (the GLOBAL device set across processes).")
--
scripts/run/run_omip_core2.py-3343-                        "is then built over all global devices and the --n-gpus guard "
scripts/run/run_omip_core2.py-3344-                        "is relaxed to jax.device_count() (global). The OMIP host loop "
scripts/run/run_omip_core2.py-3345-                        "runs identically on every process over the all-gathered "
scripts/run/run_omip_core2.py-3346-                        "(replicated) state; I/O (manifest, CSV, snapshots, transports) "
scripts/run/run_omip_core2.py-3347-                        "is written ONLY by process 0. Single-process (default, no "
scripts/run/run_omip_core2.py-3348-                        "--distributed) is byte-unchanged.")
scripts/run/run_omip_core2.py-3349-    p.add_argument("--spmd-persistent-state", action="store_true",
scripts/run/run_omip_core2.py-3350-                   help="With --n-gpus > 1: keep the ocean state lat-band "
scripts/run/run_omip_core2.py:3351:                        "SHARDED across steps (make_sharded_ocean_step) instead "
scripts/run/run_omip_core2.py-3352-                        "of the global-in/global-out wrapper's full-state "
scripts/run/run_omip_core2.py-3353-                        "scatter+gather EVERY step (scaling-M2). Host post-step "
scripts/run/run_omip_core2.py-3354-                        "BCs run UNCHANGED: the leaf-wise host updates (SSS "
scripts/run/run_omip_core2.py-3355-                        "restore / ice-thermo / nudge / drag) read+write "
scripts/run/run_omip_core2.py-3356-                        "per-leaf on the addressable sharded arrays, and the "
scripts/run/run_omip_core2.py-3357-                        "jnp per-column BCs (geothermal / ISF / BBL) are "
scripts/run/run_omip_core2.py-3358-                        "sharding-transparent. The FULL state is gathered only "
scripts/run/run_omip_core2.py-3359-                        "at snapshot/abort/final boundaries — prognostic-ice / "
--
scripts/run/run_omip_core2.py-5536-                    f"ntasks=N).")
scripts/run/run_omip_core2.py-5537-        n_lat_final = int(grid.n_lat)
scripts/run/run_omip_core2.py-5538-        if n_lat_final % args.n_gpus != 0:
scripts/run/run_omip_core2.py-5539-            raise SystemExit(
scripts/run/run_omip_core2.py-5540-                f"internal: padded n_lat ({n_lat_final}) not divisible by "
scripts/run/run_omip_core2.py-5541-                f"n_gpus ({args.n_gpus}) — the south-pad failed.")
scripts/run/run_omip_core2.py-5542-        from legoesm.parallel.mesh import create_latlon_mesh
scripts/run/run_omip_core2.py-5543-        from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip_core2.py:5544:            make_sharded_ocean_step_global,
scripts/run/run_omip_core2.py-5545-        )
scripts/run/run_omip_core2.py-5546-        # Prime the build-once vertex-mask cache from the concrete state BEFORE
scripts/run/run_omip_core2.py-5547-        # building the sharded step (the wrapper slices the primed global vmask
scripts/run/run_omip_core2.py-5548-        # per band; an unprimed cache raises in _build_band_vertex_masks).
scripts/run/run_omip_core2.py-5549-        model.prime_step_caches(state)
scripts/run/run_omip_core2.py-5550-        _spmd_mesh = create_latlon_mesh(n_devices=args.n_gpus).mesh
scripts/run/run_omip_core2.py-5551-        _tf_spmd = getattr(model.config, "tidal_forcing", None)
scripts/run/run_omip_core2.py-5552-        if _tf_spmd is not None and _tf_spmd.enabled:
--
scripts/run/run_omip_core2.py-5560-            # lat-band sharded ACROSS steps via the pure-dynamics inner step
scripts/run/run_omip_core2.py-5561-            # (the wrapper docstring's own guidance); shard_state_latlon /
scripts/run/run_omip_core2.py-5562-            # gather_state_latlon run only at the residency boundaries the
scripts/run/run_omip_core2.py-5563-            # helpers below manage (initial shard, snapshot/abort/final
scripts/run/run_omip_core2.py-5564-            # gathers, and the counted per-step gathers forced by host-global
scripts/run/run_omip_core2.py-5565-            # consumers).  t_sec is always None here (tide fail-fasts above).
scripts/run/run_omip_core2.py-5566-            from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip_core2.py-5567-                gather_state_latlon,
scripts/run/run_omip_core2.py:5568:                make_sharded_ocean_step,
scripts/run/run_omip_core2.py-5569-                shard_state_latlon,
scripts/run/run_omip_core2.py-5570-            )
scripts/run/run_omip_core2.py:5571:            _spmd_inner = make_sharded_ocean_step(model, _spmd_mesh)
scripts/run/run_omip_core2.py-5572-            _ocean_step = (lambda st, sf, fw, t_sec=None:
scripts/run/run_omip_core2.py-5573-                           _spmd_inner(st, dt, surface_forcing=sf,
scripts/run/run_omip_core2.py-5574-                                       freshwater=fw))
scripts/run/run_omip_core2.py-5575-
scripts/run/run_omip_core2.py-5576-            def _pers_shard_fn(st, _mesh=_spmd_mesh):
scripts/run/run_omip_core2.py-5577-                return shard_state_latlon(st, _mesh)
scripts/run/run_omip_core2.py-5578-
scripts/run/run_omip_core2.py-5579-            def _pers_gather_fn(st, _mesh=_spmd_mesh):
--
scripts/run/run_omip_core2.py-5581-
scripts/run/run_omip_core2.py-5582-            _spmd_persistent = True
scripts/run/run_omip_core2.py-5583-            print(f"[setup] multi-GPU lat-band SPMD: {args.n_gpus} devices, "
scripts/run/run_omip_core2.py-5584-                  f"n_lat={n_lat_final} ({n_lat_final // args.n_gpus} "
scripts/run/run_omip_core2.py-5585-                  f"rows/band); PERSISTENT sharded state "
scripts/run/run_omip_core2.py-5586-                  f"(--spmd-persistent-state): full-state gathers only at "
scripts/run/run_omip_core2.py-5587-                  f"snapshot/abort/final + counted per-step forcings.")
scripts/run/run_omip_core2.py-5588-        else:
scripts/run/run_omip_core2.py:5589:            _spmd_step = make_sharded_ocean_step_global(model, _spmd_mesh)
scripts/run/run_omip_core2.py-5590-            # t_sec is always None here (tide-enabled fail-fasts above).
scripts/run/run_omip_core2.py-5591-            _ocean_step = (lambda st, sf, fw, t_sec=None:
scripts/run/run_omip_core2.py-5592-                           _spmd_step(st, dt, surface_forcing=sf,
scripts/run/run_omip_core2.py-5593-                                      freshwater=fw))
scripts/run/run_omip_core2.py-5594-            print(f"[setup] multi-GPU lat-band SPMD: {args.n_gpus} devices, "
scripts/run/run_omip_core2.py-5595-                  f"n_lat={n_lat_final} ({n_lat_final // args.n_gpus} "
scripts/run/run_omip_core2.py-5596-                  f"rows/band); global-in/global-out wrapper (host BCs on "
scripts/run/run_omip_core2.py-5597-                  f"gathered state).")
--
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-1-"""Direct test for scripts/bench/bench_ocean_latlon_spmd_scaling.py.
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-2-
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-3-Exercises the builder + the nd=1 single-device main() end-to-end on a tiny
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-4-grid (JSONL record shape) and the argument guards. The multi-device timing
tests/unit/test_bench_ocean_latlon_spmd_scaling.py:5:path reuses make_sharded_ocean_step, whose correctness gate is
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-6-tests/parallel/test_latlon_ocean_spmd_step.py; the --multicontroller path is
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-7-cluster-gated (see the script docstring).
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-8-"""
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-9-from __future__ import annotations
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-10-
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-11-import importlib.util
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-12-import json
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-13-import sys
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-1-"""Unit tests for the band-geometry cross-process fingerprint gate.
tests/ocean/unit/test_sharded_geom_fingerprint.py-2-
tests/ocean/unit/test_sharded_geom_fingerprint.py:3:The gate decides whether ``make_sharded_ocean_step`` accepts per-process
tests/ocean/unit/test_sharded_geom_fingerprint.py-4-band-geometry stacks without the (removed, nd-linear-cost) process-0
tests/ocean/unit/test_sharded_geom_fingerprint.py-5-broadcast — see the 2026-08-03 fix note at the sharded put. These tests
tests/ocean/unit/test_sharded_geom_fingerprint.py-6-pin the gate's discrimination properties single-process (the
tests/ocean/unit/test_sharded_geom_fingerprint.py-7-multicontroller allgather wiring is exercised by the distributed suite).
tests/ocean/unit/test_sharded_geom_fingerprint.py-8-"""
tests/ocean/unit/test_sharded_geom_fingerprint.py-9-import numpy as np
tests/ocean/unit/test_sharded_geom_fingerprint.py-10-import pytest
tests/ocean/unit/test_sharded_geom_fingerprint.py-11-
--
scripts/run/run_omip.py-118-    vertical_mixing: VerticalMixingConfig
scripts/run/run_omip.py-119-    precision: str = "fp64"
scripts/run/run_omip.py-120-    # GM/Redi bundle threaded into _create_setup's realistic-bathymetry
scripts/run/run_omip.py-121-    # lat-lon path (inert on flat-bottom / other-grid runs; --no-gm-redi
scripts/run/run_omip.py-122-    # still disables it entirely).  Carried here so --params can reach
scripts/run/run_omip.py-123-    # GMRediConfig / VisbeckConfig / TreguierConfig (#691/#724).
scripts/run/run_omip.py-124-    gm_redi: GMRediConfig = _DEFAULT_BATHY_GM_REDI
scripts/run/run_omip.py-125-    # Lat-band SPMD (single-controller multi-GPU) for the lat-lon restoring
scripts/run/run_omip.py:126:    # lane: wraps the loop's dynamics step in ``make_sharded_ocean_step``
scripts/run/run_omip.py-127-    # (the #751/#758-validated lane-D step).  0 devices = all local.
scripts/run/run_omip.py-128-    enable_latlon_spmd: bool = False
scripts/run/run_omip.py-129-    spmd_n_devices: int = 0
scripts/run/run_omip.py-130-    # Route-B multicontroller (jax.distributed cross-process NCCL): promote the
scripts/run/run_omip.py-131-    # lat-band lane to span ALL global devices across processes (multi-node).
scripts/run/run_omip.py-132-    multicontroller: bool = False
scripts/run/run_omip.py-133-    coordinator: str | None = None
scripts/run/run_omip.py-134-
--
scripts/run/run_omip.py-702-                       "Haney SST/SSS restoring toward WOA. 'jra55_do_tropical' "
scripts/run/run_omip.py-703-                       "uses LY09 bulk fluxes from a pre-built JRA55-do cache "
scripts/run/run_omip.py-704-                       "(see scripts/data/prepare_omip_forcing.py). Currently "
scripts/run/run_omip.py-705-                       "supports only --grid latlon."
scripts/run/run_omip.py-706-                   ))
scripts/run/run_omip.py-707-    p.add_argument("--enable-latlon-spmd", action="store_true", default=False,
scripts/run/run_omip.py-708-                   help=(
scripts/run/run_omip.py-709-                       "Run the lat-lon lane's dynamics step lat-band-SPMD "
scripts/run/run_omip.py:710:                       "across the local devices (make_sharded_ocean_step — "
scripts/run/run_omip.py-711-                       "the validated multi-GPU ocean lane). Supports the "
scripts/run/run_omip.py-712-                       "restoring lane AND the JRA55 block-scan lanes "
scripts/run/run_omip.py-713-                       "(forcing stacks are lat-band-sharded; the in-scan "
scripts/run/run_omip.py-714-                       "bulk fluxes stay shard-local). Single-controller "
scripts/run/run_omip.py-715-                       "only (one process; multi-node scaling lives in "
scripts/run/run_omip.py-716-                       "bench_ocean_latlon_spmd_scaling --multicontroller); "
scripts/run/run_omip.py-717-                       "requires --grid latlon and n_lat divisible by the "
scripts/run/run_omip.py-718-                       "device count. Unsupported: --jra55-sea-ice, the "
--
scripts/run/run_omip.py-2366-        raise ValueError(
scripts/run/run_omip.py-2367-            "spmd_step + prognostic sea ice is unsupported "
scripts/run/run_omip.py-2368-            "(run_omip_single refuses --jra55-sea-ice with "
scripts/run/run_omip.py-2369-            "--enable-latlon-spmd).")
scripts/run/run_omip.py-2370-    # aux threading (codex r18 P1): the SPMD step's sharded geometry
scripts/run/run_omip.py-2371-    # stacks must cross THIS jit boundary as an ARGUMENT — captured in the
scripts/run/run_omip.py-2372-    # closure they become outer-trace constants whose value jax cannot
scripts/run/run_omip.py-2373-    # fetch for non-addressable arrays on multicontroller (see
scripts/run/run_omip.py:2374:    # make_sharded_ocean_step's aux note).
scripts/run/run_omip.py-2375-    if spmd_step is not None:
scripts/run/run_omip.py-2376-        def _dyn_step(st, d, aux=None, **kw):
scripts/run/run_omip.py-2377-            return spmd_step(st, d, aux=aux, **kw)
scripts/run/run_omip.py-2378-    else:
scripts/run/run_omip.py-2379-        def _dyn_step(st, d, aux=None, **kw):
scripts/run/run_omip.py-2380-            return model._step_impl(st, d, **kw)
scripts/run/run_omip.py-2381-
scripts/run/run_omip.py-2382-    @jax.jit
--
scripts/run/run_omip.py-3168-                   restoring_ramp_days: float = 0.0,
scripts/run/run_omip.py-3169-                   jra55_state=None,
scripts/run/run_omip.py-3170-                   checkpoint_days=None, checkpoint_dir=None,
scripts/run/run_omip.py-3171-                   max_wallclock_seconds: float = 0.0,
scripts/run/run_omip.py-3172-                   restart_buffer_seconds: float = 600.0,
scripts/run/run_omip.py-3173-                   start_step=0,
scripts/run/run_omip.py-3174-                   nudge_woa_tau=0.0, T_woa_3d=None, S_woa_3d=None,
scripts/run/run_omip.py-3175-                   snapshot_fn=None, spmd_step=None, spmd_gather=None,
scripts/run/run_omip.py:3176:                   spmd_shard_stack=None):
scripts/run/run_omip.py-3177-    """Run time loop with diagnostics.
scripts/run/run_omip.py-3178-
scripts/run/run_omip.py-3179-    Two forcing paths, mutually exclusive:
scripts/run/run_omip.py-3180-
scripts/run/run_omip.py-3181-    * **restoring** (default): plain ``model.step(state, dt)`` followed
scripts/run/run_omip.py-3182-      by Haney SST/SSS restoring when ``restoring_targets`` is set.
scripts/run/run_omip.py-3183-      With ``spmd_step`` set (``--enable-latlon-spmd``), the dynamics
scripts/run/run_omip.py-3184-      step runs through that lat-band-SPMD callable instead — the state
--
scripts/run/run_omip.py-3434-            elif use_gpu_interp:
scripts/run/run_omip.py-3435-                raw_stack, runoff_records, record_meta = (
scripts/run/run_omip.py-3436-                    _preload_jra55_raw_records(
scripts/run/run_omip.py-3437-                        block_start, actual, dt, jra55_state))
scripts/run/run_omip.py-3438-            else:
scripts/run/run_omip.py-3439-                atm_stack, runoff_stack = _preload_jra55_forcing_block(
scripts/run/run_omip.py-3440-                    block_start, actual, dt, jra55_state,
scripts/run/run_omip.py-3441-                )
scripts/run/run_omip.py:3442:            if spmd_shard_stack is not None:
scripts/run/run_omip.py-3443-                # Lay the per-block forcing stacks out lat-band-sharded so
scripts/run/run_omip.py-3444-                # the in-scan interpolation / bulk fluxes stay shard-local
scripts/run/run_omip.py-3445-                # (an unsharded stack commits to device 0 and serializes
scripts/run/run_omip.py-3446-                # every forcing op there).
scripts/run/run_omip.py-3447-                if use_gpu_interp:
scripts/run/run_omip.py:3448:                    raw_stack = spmd_shard_stack(raw_stack)
scripts/run/run_omip.py:3449:                    runoff_records = spmd_shard_stack(runoff_records)
scripts/run/run_omip.py-3450-                else:
scripts/run/run_omip.py:3451:                    atm_stack = spmd_shard_stack(atm_stack)
scripts/run/run_omip.py:3452:                    runoff_stack = spmd_shard_stack(runoff_stack)
scripts/run/run_omip.py-3453-            io_dt = time.time() - t_io_start
scripts/run/run_omip.py-3454-
scripts/run/run_omip.py-3455-            t_compute_start = time.time()
scripts/run/run_omip.py-3456-            if use_gpu_interp:
scripts/run/run_omip.py-3457-                bfn = _get_block_fn_interp(actual)
scripts/run/run_omip.py-3458-                if _ice_on:
scripts/run/run_omip.py-3459-                    state, ice_state = bfn(
scripts/run/run_omip.py-3460-                        state, raw_stack, runoff_records,
--
scripts/run/run_omip.py-4782-
scripts/run/run_omip.py-4783-    # --- Lat-band SPMD (--enable-latlon-spmd): wrap the dynamics step in
scripts/run/run_omip.py-4784-    # the validated multi-GPU sharded ocean step and shard the state.
scripts/run/run_omip.py-4785-    # Single-controller only; restoring lane only (the JRA55 block
scripts/run/run_omip.py-4786-    # functions call model._step_impl directly — follow-up).  Runs AFTER
scripts/run/run_omip.py-4787-    # the restart load so a resumed state is sharded too.
scripts/run/run_omip.py-4788-    spmd_step = None
scripts/run/run_omip.py-4789-    spmd_gather = None
scripts/run/run_omip.py:4790:    spmd_shard_stack = None
scripts/run/run_omip.py-4791-    if run_config.enable_latlon_spmd:
scripts/run/run_omip.py-4792-        if grid_type != "latlon":
scripts/run/run_omip.py-4793-            raise SystemExit(
scripts/run/run_omip.py-4794-                f"--enable-latlon-spmd requires --grid latlon "
scripts/run/run_omip.py-4795-                f"(got {grid_type}).")
scripts/run/run_omip.py-4796-        if (jra55_state is not None
scripts/run/run_omip.py-4797-                and jra55_state.get("_use_single_step", False)):
scripts/run/run_omip.py-4798-            raise SystemExit(
--
scripts/run/run_omip.py-4823-                raise SystemExit(
scripts/run/run_omip.py-4824-                    f"--enable-latlon-spmd: n_lat ({grid.n_lat}) not "
scripts/run/run_omip.py-4825-                    f"divisible by the device count ({_nd}); pick "
scripts/run/run_omip.py-4826-                    f"--spmd-n-devices dividing n_lat.")
scripts/run/run_omip.py-4827-            from functools import partial
scripts/run/run_omip.py-4828-
scripts/run/run_omip.py-4829-            from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip.py-4830-                gather_state_latlon,
scripts/run/run_omip.py:4831:                make_sharded_ocean_step,
scripts/run/run_omip.py-4832-                shard_forcing_stack_latlon,
scripts/run/run_omip.py-4833-                shard_state_latlon,
scripts/run/run_omip.py-4834-            )
scripts/run/run_omip.py-4835-            from legoesm.parallel.mesh import create_latlon_mesh
scripts/run/run_omip.py-4836-            # Prime the build-once vertex-mask cache from the CONCRETE
scripts/run/run_omip.py-4837-            # state so the wrapper can build per-band masks host-side.
scripts/run/run_omip.py-4838-            model.prime_step_caches(state)
scripts/run/run_omip.py-4839-            _dev = create_latlon_mesh(n_devices=_nd)
scripts/run/run_omip.py:4840:            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
scripts/run/run_omip.py-4841-            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
scripts/run/run_omip.py-4842-            state = shard_state_latlon(state, _dev.mesh)
scripts/run/run_omip.py-4843-            # Lay per-block forcing stacks out lat-band-sharded so the
scripts/run/run_omip.py-4844-            # in-scan interpolation / bulk fluxes stay shard-local (shared
scripts/run/run_omip.py-4845-            # layout helper — see shard_forcing_stack_latlon).
scripts/run/run_omip.py:4846:            spmd_shard_stack = partial(
scripts/run/run_omip.py-4847-                shard_forcing_stack_latlon, mesh=_dev.mesh)
scripts/run/run_omip.py-4848-            if jax.process_index() == 0:
scripts/run/run_omip.py-4849-                _lane = "route-B multicontroller" if _multi else "single-controller"
scripts/run/run_omip.py-4850-                print(f"  SPMD ({_lane}): lat-band sharded dynamics step over "
scripts/run/run_omip.py-4851-                      f"{_nd} devices across {jax.process_count()} process(es) "
scripts/run/run_omip.py-4852-                      f"({jax.default_backend()}).")
scripts/run/run_omip.py-4853-        elif jax.process_index() == 0:
scripts/run/run_omip.py-4854-            print("  SPMD: single device visible — flag is a no-op.")
--
scripts/run/run_omip.py-4870-        nudge_woa_tau=args.nudge_woa_tau,
scripts/run/run_omip.py-4871-        T_woa_3d=(T_woa * state.land_mask.data[..., jnp.newaxis]).astype(
scripts/run/run_omip.py-4872-            state.T.data.dtype) if args.nudge_woa_tau > 0 and T_woa is not None else None,
scripts/run/run_omip.py-4873-        S_woa_3d=(S_woa * state.land_mask.data[..., jnp.newaxis]).astype(
scripts/run/run_omip.py-4874-            state.S.data.dtype) if args.nudge_woa_tau > 0 and S_woa is not None else None,
scripts/run/run_omip.py-4875-        snapshot_fn=_snapshot_fn,
scripts/run/run_omip.py-4876-        spmd_step=spmd_step,
scripts/run/run_omip.py-4877-        spmd_gather=spmd_gather,
scripts/run/run_omip.py:4878:        spmd_shard_stack=spmd_shard_stack,
scripts/run/run_omip.py-4879-    )
scripts/run/run_omip.py-4880-    if spmd_gather is not None:
scripts/run/run_omip.py-4881-        # Downstream report/plot/save paths expect the full (n_lat+1)
scripts/run/run_omip.py-4882-        # staggered v layout, not the sharded v_lower carry.  Every rank
scripts/run/run_omip.py-4883-        # dispatches this gather (it is a collective); only rank 0 writes.
scripts/run/run_omip.py-4884-        state = spmd_gather(state)
scripts/run/run_omip.py-4885-
scripts/run/run_omip.py-4886-    # Surface a failed FINAL async restart write while the run can still
--
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-1-"""Multi-PROCESS jax.distributed equivalence VALIDATION for the lat-band SPMD ocean
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py:2:step — the multi-node correctness gate for ``make_sharded_ocean_step_global``.
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-3-
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-4-Runs STANDALONE (not pytest): the root ``tests/conftest.py`` initializes the XLA
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-5-backend (``ensure_metal_or_fallback`` -> ``jax.default_backend()``) at collection,
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-6-before a test module could call ``jax.distributed.initialize()`` (which must
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-7-precede backend init).  A standalone script bootstraps jax.distributed FIRST.
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-8-
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-9-TWO-PHASE design (the SERIAL reference cannot run under jax.distributed):
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-10-``model.step`` (the single-device reference) runs OUTSIDE any ``shard_map``, so its
--
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-112-    model = LatLonCGridOceanModel(grid, z_coord, cfg)
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-113-    return model, grid, z_coord
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-114-
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-115-
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-116-def _spmd_step_result(model, state0, n_dev):
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-117-    """Run N SPMD steps over an ``n_dev``-band lat mesh (global devices), gather."""
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-118-    from legoesm.parallel.mesh import create_latlon_mesh
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-119-    from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py:120:        make_sharded_ocean_step_global,
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-121-    )
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-122-    dev = create_latlon_mesh(n_devices=n_dev)
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-123-    assert dev.mesh.devices.size == n_dev, dev.mesh.devices.size
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-124-    model._ensure_vertex_mask(state0)
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py:125:    step = make_sharded_ocean_step_global(model, dev.mesh)
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-126-    ss = state0
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-127-    for _ in range(N_STEPS):

exec
/usr/bin/bash -lc "rg -n -C 12 \"def make_sharded_ocean_step\\(\" src legoesm_pkg legoesm 2>/dev/null || true
rg --files | rg 'sharded_ocean_step\\.py"'$|test_run_omip.*'"\\.py"'$'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 522ms:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
tests/unit/test_run_omip_latlon_spmd.py
tests/unit/test_run_omip_core2_kpp_override.py
tests/unit/test_run_omip_core2_gm_treguier.py
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py
tests/unit/test_run_omip_jra55_restart.py
tests/unit/test_run_omip_core2_ice_categories.py
tests/unit/test_run_omip_jra55_smoke.py
tests/unit/test_run_omip_core2_spmd_persistent_cli.py
tests/unit/test_run_omip_latlon_polar_filter.py
tests/unit/test_run_omip_sea_ice_scan.py
tests/unit/test_run_omip_jra55_dispatch.py
tests/unit/test_run_omip_core2_pcg_variant.py
tests/unit/test_run_omip_cli.py

exec
/usr/bin/bash -lc "rg -n -C 12 \"def make_sharded_ocean_step\\(\" packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
rg -n -C 5 \"\\.aux|aux=\" packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | head -n 500
sed -n '120,270p' tests/unit/test_run_omip_latlon_spmd.py
sed -n '1,220p' tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 219ms:
600-        if name in _V_STAGGERED_STATE_FIELDS:
601-            continue
602-        val = getattr(state, name)
603-        if val is None:
604-            updates[name] = None
605-        elif hasattr(val, "data"):
606-            updates[name] = val.replace(data=_gather_arr(val.data))
607-        else:
608-            updates[name] = _gather_arr(jnp.asarray(val))
609-    return state._replace(**updates)
610-
611-
612:def make_sharded_ocean_step(model, mesh):
613-    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
614-    sponge=None, t_seconds=None) -> state`` running ``model.step``
615-    lat-band-SPMD.
616-
617-    The forcing channels mirror ``model.step``'s keyword surface: pass
618-    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
619-    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
620-    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
621-    dynamics-only program; each distinct None<->populated combination
622-    compiles (and caches) its own executable.
623-
624-    Parameters
890-                    f"has leading dim {leaf.shape[0]} != n_lat "
891-                    f"({n_lat_global}); forcing must be cell-centered "
892-                    f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")
893-
894-    def sharded_step(state, dt, freshwater=None, surface_forcing=None,
895:                     sponge=None, t_seconds=None, aux=None):
896-        # ``aux`` (codex/2-proc repro 2026-08-03): the geometry + vmask
897-        # stacks are SHARDED global arrays; when this wrapper runs INSIDE
898-        # an outer trace (a bench/driver ``jit``/``scan`` over the step —
899-        # jit-of-jit inlines the inner call), concrete closure arrays
900-        # become OUTER-TRACE CONSTANTS and jax's MLIR constant handler
901-        # tries to fetch their value — impossible for non-addressable
902-        # arrays (RuntimeError: 'Fetching value ... non-addressable'; the
903-        # multicontroller lane has been broken this way since the
904-        # #1370-iii stack sharding). Callers that wrap the step in their
905:        # own jit MUST thread ``step.aux`` through their jit boundary as
906-        # an ARGUMENT and pass it back here.
907-        # ONE forcing operand: None fields drop out of the pytree structure,
908-        # so specs derived by tree.map skip them automatically and the
909-        # structure key below distinguishes every None<->array combination.
910-        forcing = (freshwater, surface_forcing, sponge, t_seconds)
--
980-            set_spmd_mesh(_prev_mesh)
981-            set_halo_backend(_prev_backend, _prev_topo)
982-
983-    # Expose the stacks so outer-jit callers can pass them as arguments
984-    # (see the ``aux`` note in the signature).
985:    sharded_step.aux = (geom_stacks, vmask_stack)
986-    return sharded_step
987-
988-
989-def make_sharded_ocean_step_global(model, mesh):
990-    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
    # A bare `python tests/unit/<file>.py` puts tests/unit on sys.path, not
    # the repo root pytest provides — prepend it so the harness's
    # repo-relative imports resolve without an editable install (codex r2).
    _root = str(_THIS.parents[2])
    env["PYTHONPATH"] = _root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(_THIS), "--parity-worker", str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=870,
    )
    assert proc.returncode == 0 and "PARITY-OK" in proc.stdout, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
        f"stderr:\n{proc.stderr[-3000:]}")


def test_loop_refuses_spmd_with_jra55_single_step():
    """The single-step JRA55 fallback (``model.step``) cannot run sharded —
    only the block-scan lanes thread ``spmd_step`` (part 2b)."""
    _day23 = _load_harness()
    run_omip = _day23.run_omip
    grid, z_coord, _config, model, _gt = _day23._make_tiny_latlon_setup(
        n_lat=8, n_lon=16)
    state0 = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    with pytest.raises(ValueError, match="single-step fallback is unsupported"):
        run_omip._run_omip_loop(
            model, state0, grid_type="latlon", grid=grid, z_coord=z_coord,
            dt=600.0, n_steps=1, diag_every=1,
            jra55_state={"_use_single_step": True},
            spmd_step=lambda s, d: s)


# ---------------------------------------------------------------------------
# Part 2b: the JRA55 block-scan lanes thread spmd_step through the scan body
# and shard the per-block forcing stacks.  Both the CPU-interp
# (_build_jra55_block_fn) and the GPU-interp (_build_jra55_block_fn_interp,
# the run_omip default) block builders must match the serial block loop to
# the same re-association floor as the restoring lane.  Subprocess again so
# the 2-virtual-device XLA flag always takes effect.
# ---------------------------------------------------------------------------

def _jra55_parity_worker(tmp_dir: str, interp_mode: str) -> None:
    """Runs in the SUBPROCESS: JRA55 block-scan SPMD-vs-serial parity.

    ``interp_mode`` is ``"cpu"`` (host-interp block) or ``"gpu"`` (in-scan
    GPU-interp block).  Uses the SAME shard_forcing_stack_latlon layout the
    driver uses (imported, not re-derived) so the test can't drift from it.
    """
    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    assert jax.device_count() >= 2, (
        f"worker expected 2 virtual devices, got {jax.device_count()}")

    _day23 = _load_harness()
    run_omip = _day23.run_omip

    from functools import partial

    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon,
        make_sharded_ocean_step,
        shard_forcing_stack_latlon,
        shard_state_latlon,
    )
    from legoesm.parallel.mesh import create_latlon_mesh

    tmp_path = Path(tmp_dir)
    n_lat, n_lon = 8, 16
    dt, n_steps = 600.0, 6

    cache = _day23._make_synthetic_cache(
        tmp_path, n_lat=n_lat, n_lon=n_lon, n_records=64)
    grid, z_coord, _config, model, _gt = _day23._make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon)
    T_woa, S_woa = _day23._make_woa_like_targets(grid, nlev=4)
    args = _day23._argparse_namespace(jra55_cache=str(cache))

    def _fresh_js():
        # _setup_jra55_forcing_state does NOT set _gpu_interp (the driver
        # does, from --gpu-interp); set it here to pick the lane.  A fresh
        # dict per loop avoids sharing the sponge/target arrays across the
        # serial and sharded runs.
        js = run_omip._setup_jra55_forcing_state(
            args, grid, "latlon", z_coord=z_coord, T_woa=T_woa, S_woa=S_woa)
        js["_gpu_interp"] = (interp_mode == "gpu")
        return js

    common = dict(
        grid_type="latlon", grid=grid, z_coord=z_coord,
        dt=dt, n_steps=n_steps, diag_every=3,
        checkpoint_days=None, checkpoint_dir=None,
    )

    state0 = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    serial_final, _d, _w, ok_serial, _b = run_omip._run_omip_loop(
        model, state0, jra55_state=_fresh_js(), **common)
    assert ok_serial, "serial JRA55 loop reported not-ok"

    dev = create_latlon_mesh(n_devices=2)
    spmd_step = make_sharded_ocean_step(model, dev.mesh)
    model.prime_step_caches(state0)
    ss0 = shard_state_latlon(state0, dev.mesh)
    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
        model, ss0, jra55_state=_fresh_js(),
        spmd_step=spmd_step,
        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
        spmd_shard_stack=partial(shard_forcing_stack_latlon, mesh=dev.mesh),
        **common)
    assert ok_spmd, "SPMD JRA55 loop reported not-ok"
    spmd_final = gather_state_latlon(spmd_final, dev.mesh)

    # Same re-association floor as the restoring-lane and step-level gates.
    for nm in ("u", "v", "eta", "T", "S"):
        np.testing.assert_allclose(
            np.asarray(getattr(spmd_final, nm).data),
            np.asarray(getattr(serial_final, nm).data),
            atol=2e-4, rtol=1e-3,
            err_msg=f"JRA55 {interp_mode}-interp SPMD {nm} mismatch")
    print("JRA55-PARITY-OK")


def _run_jra55_parity_subprocess(tmp_path, interp_mode: str):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    _root = str(_THIS.parents[2])
    env["PYTHONPATH"] = _root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(_THIS), "--jra55-parity-worker",
         str(tmp_path), interp_mode],
        env=env, capture_output=True, text=True, timeout=870,
    )
    assert proc.returncode == 0 and "JRA55-PARITY-OK" in proc.stdout, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
        f"stderr:\n{proc.stderr[-3000:]}")


@pytest.mark.timeout(900)
def test_jra55_block_loop_spmd_matches_serial_cpu_interp(tmp_path):
    _run_jra55_parity_subprocess(tmp_path, "cpu")


@pytest.mark.timeout(900)
def test_jra55_block_loop_spmd_matches_serial_gpu_interp(tmp_path):
    _run_jra55_parity_subprocess(tmp_path, "gpu")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--parity-worker":
        _parity_worker(sys.argv[2])
"""SELF-SPAWNING route-B multicontroller gate for the ``run_omip`` DRIVER.

Companion to the ocean *bench* selfspawn gate
(``test_latlon_ocean_spmd_multicontroller_selfspawn.py``, which drives
``bench_ocean_latlon_spmd_scaling.py``): THIS variant spawns two worker
processes and drives the FULL PRODUCTION DRIVER
(``scripts/run/run_omip.py --enable-latlon-spmd --multicontroller``)
end-to-end — pinning the driver pieces a Derecho/Levante multi-node NCCL
OMIP run exercises that the bench does NOT:

* the early ``init_multicontroller_distributed`` bootstrap firing BEFORE any
  device work (a module-import device query would make
  ``jax.distributed.initialize`` raise "must be called before backend init");
* the lifted ``jax.process_count() > 1`` refusal + the ALL-global-device mesh;
* the rank-0-gated file I/O (``_save_output(write=...)``, the gather-aware
  ``save_restart`` splitting the collective gather from the rank-0 write, the
  MLD snapshot) while every rank still dispatches the collective gather;
* a consistent per-process exit code (all ranks build ``results``).

Numerics of the sharded step itself live in
``tests/parallel/test_latlon_ocean_spmd_step.py``; SPMD-vs-serial loop parity
in ``tests/unit/test_run_omip_latlon_spmd.py``.  This gate asserts the DRIVER
federates + writes exactly one clean output set without a hang or crash.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_RUN_OMIP = _ROOT / "scripts" / "run" / "run_omip.py"
N_PROC = 2


def _load_day23():
    """Load the JRA55 dispatch harness (synthetic-cache builder) by path."""
    day23 = _ROOT / "tests" / "unit" / "test_run_omip_jra55_dispatch.py"
    spec = importlib.util.spec_from_file_location("_day23_mc", day23)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_day23_mc"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.timeout(600)
def test_run_omip_two_process_selfspawn_multicontroller(tmp_path):
    from multihost_harness import run_federated

    # Build a tiny JRA55-do synthetic cache once in the PARENT; both workers
    # read it (read-only) — this drives the richest route-B path: the
    # gpu-interp block-scan with forcing stacks sharded across processes, the
    # cross-process gather, and the rank-0 restart write.
    _day23 = _load_day23()
    n_lat, n_lon = 16, 32
    cache = _day23._make_synthetic_cache(
        tmp_path, n_lat=n_lat, n_lon=n_lon, n_records=16)

    out_dir = tmp_path / "omip_mc"
    base_env = dict(os.environ)
    base_env["JAX_PLATFORMS"] = "cpu"
    base_env["JAX_ENABLE_X64"] = "1"
    base_env.pop("XLA_FLAGS", None)  # 1 real CPU device per process
    # The worker loads run_omip.py by path; give it repo root + every package
    # on PYTHONPATH so its imports resolve without an editable install.
    _pkgs = os.pathsep.join(
        str(p) for p in sorted((_ROOT / "packages").glob("*")) if p.is_dir())
    base_env["PYTHONPATH"] = os.pathsep.join(
        [_pkgs, str(_ROOT), base_env.get("PYTHONPATH", "")])

    def build_cmd(rank, port):
        cmd = [
            sys.executable, str(_RUN_OMIP),
            "--grid", "latlon", "--resolution", f"{n_lat}x{n_lon}",
            "--nlev", "4",
            "--forcing-mode", "jra55_do_tropical", "--jra55-cache", str(cache),
            "--days", "0.05", "--dt", "600",      # ~7 steps
            "--checkpoint-days", "0.03",          # force a rank-0 restart write
            # Arm the wallclock path (never exhausts at this size) so the
            # cross-process exit-consensus all-gather runs each block without
            # triggering exit — exercises the deadlock-free wallclock check.
            "--max-wallclock-seconds", "3600",
            "--enable-latlon-spmd", "--multicontroller",
            "--coordinator", f"localhost:{port}",
            "--output", str(out_dir),
        ]
        # run_omip reads rank/size from OMPI env when --coordinator is given;
        # run_federated only varies argv, so inject the rank via `env`.
        return [
            "env",
            f"OMPI_COMM_WORLD_SIZE={N_PROC}",
            f"OMPI_COMM_WORLD_RANK={rank}",
        ] + cmd

    rcs, outs = run_federated(build_cmd, N_PROC, base_env, timeout_s=540)
    for rank, (rc, txt) in enumerate(zip(rcs, outs)):
        assert rc == 0, (
            f"rank {rank} exited {rc}\n--- output ---\n{txt[-4000:]}")

    # Rank 0 owns the output; exactly one clean results.json under the
    # grid/resolution tree (non-root ranks return early with write=False).
    res_dir = out_dir / "latlon" / f"{n_lat}x{n_lon}"
    res_path = res_dir / "results.json"
    assert res_path.exists(), (
        f"no results.json at {res_path}\nrank0 tail:\n{outs[0][-3000:]}")
    rec = json.loads(res_path.read_text())
    assert rec["status"] in ("PASS", "FAIL"), rec
    # The federation advanced without blowing up (finite final SST).
    assert rec["final_SST"] is not None and abs(rec["final_SST"]) < 1e4, rec

    # The gather-aware restart write fired on rank 0: its collective gather ran
    # on BOTH ranks (a rank-0-only gather would have hung the federation), then
    # only rank 0 wrote the file.  The restart must carry the full (n_lat+1, ..)
    # staggered v the single-controller gate also checks.
    restarts = sorted((res_dir / "restarts").glob("restart_day*.npz"))
    assert restarts, (
        f"no restart under {res_dir/'restarts'}\nrank0 tail:\n{outs[0][-3000:]}")
    import numpy as np
    with np.load(restarts[-1]) as data:
        v_key = [k for k in data.files if k in ("v", "v_data")]
        assert v_key, f"no v field in restart: {data.files}"
        assert data[v_key[0]].shape[0] == n_lat + 1, (
            f"restart v leading dim {data[v_key[0]].shape[0]} != n_lat+1 "
            f"({n_lat + 1}) — route-B save missed the gather")

exec
/usr/bin/bash -lc "sed -n '612,990p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
rg -n -C 5 \"JRA55|jra55\" tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py tests -g '*.py' | head -n 450" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 684ms:
def make_sharded_ocean_step(model, mesh):
    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
    sponge=None, t_seconds=None) -> state`` running ``model.step``
    lat-band-SPMD.

    The forcing channels mirror ``model.step``'s keyword surface: pass
    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
    dynamics-only program; each distinct None<->populated combination
    compiles (and caches) its own executable.

    Parameters
    ----------
    model
        ``LatLonCGridOceanModel`` (latlon geometry; tripole north-fold is a
        follow-up — raises ``NotImplementedError`` if ``model.grid.fold`` is
        active).
    mesh
        A 1-D ``jax.sharding.Mesh`` with axis name ``"lat"`` (from
        ``legoesm.parallel.mesh.create_latlon_mesh(...).mesh``).

    Notes
    -----
    Build the ``N`` band geometries + band vertex masks host-side, stack their
    ARRAY fields over a leading band axis SHARDED ``P("lat")`` (each device
    holds only its own slab; the ``shard_map`` body reads it at ``[0]``; the
    geometry SCALAR fields stay static — see the module docstring).  ``dt`` is a TRACED,
    replicated operand and the jitted ``shard_map`` is built once and cached
    (a per-call rebuild re-traced the whole ocean step every call — see the
    ``_cache`` note below).  ``check_vma=False`` (the JAX >= 0.8
    replication check) because the band halo intentionally reads neighbour-rank
    data (replication-unaware).

    The state must be laid out with :func:`shard_state_latlon` (``v`` /
    ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
    each band's ``nl+1`` v-faces, runs the step on the band geometry, and
    converts the result back to the ``v_lower`` representation.
    """
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

    # --- host-side band geometries + vertex masks (band-stacked, P("lat")-sharded) ---
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
    # — same stack, different placement; the cross-process divergence
    # guard below runs on HOST values and is placement-blind.
    rep = NamedSharding(mesh, P("lat"))

    def _replicated_put(arr, name):
        # (Name kept for history; since 2026-08-03 this is a SHARDED stack
        # put.) The band-geometry arrays are (re)computed per process and
        # can differ in their last ULPs (per-process XLA autotuning on
        # device-derived grid fields) — the fully-replicated-put era
        # broadcast process 0's bytes to sidestep the P() bit-identity
        # assert (job 26450848). With the #1370-iii P("lat") sharding each
        # process's devices consume ONLY its own band rows, so the
        # broadcast became both unnecessary and, at nd>=96, fatal (its
        # psum program is nd x the stack — see the note at the put below).
        # GUARD (codex round-3): process 0 must not silently mask REAL
        # cross-process divergence. Compare an allgathered fingerprint:
        # structural entries exactly; value entries EXACTLY for integer/bool
        # arrays (masks are comparison results — bit-reproducible, and an
        # exact compare is the only way to catch a two-cell flip that cancels
        # in the sum, codex round-4) and to rtol 1e-5 for float arrays (only
        # ULP autotune drift is expected there; quantize-then-assert-equal
        # false-positived on a rounding boundary, job 26453240).
        # NO DEADLOCK RISK: every process fingerprints the same fields in the
        # same order and derives the verdict from the SAME gathered array, so
        # the refusal is symmetric — all raise or none.
        # Residual (documented): a float-geometry divergence preserving sum,
        # sum-of-squares AND absmax to 1e-5 is not detected; band grids are
        # analytic in lat/lon, so any real inconsistency moves those moments.
        host = np.asarray(arr)
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils

            # PER-BAND fingerprints (module-level, unit-tested): exact
            # dtypes hash positionally per band; floats compare per-band
            # moments to rtol 1e-5 — bounds each band's drift instead of
            # letting it hide in a whole-array sum, since each process's
            # own bytes are now the live inputs for the bands it owns
            # (codex r14). A mask that genuinely differs across processes
            # means different wet domains = different physics: refusing is
            # correct, not a false alarm.
            struct, vals, is_exact = geom_band_fingerprint(
                host, host.shape[0])
            g_struct = multihost_utils.process_allgather(struct)
            g_vals = multihost_utils.process_allgather(vals)
            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
                raise RuntimeError(
                    f"make_sharded_ocean_step: band-geometry field {name!r} "
                    f"DIVERGES across processes (exact_dtype={is_exact}, "
                    f"gathered={g_vals.tolist()}) — a real config/grid "
                    f"inconsistency, not autotune noise; refusing to "
                    f"shard it.")
            # NO broadcast_one_to_all here (removed 2026-08-03): its psum
            # program is [n_processes, stack] in / P() fully-replicated out,
            # so its logical arg bytes are nd x the global stack — 82.4 GB
            # at nd=96 and 109.6 GB at nd=128 for one 3-D field, the
            # near-linear-in-nd wall that killed oc @96/@128 (jobs
            # 26642771/26636762) while @64 sat just under XLA's 63.8 GB
            # limit.  The target sharding is P("lat"): each process's
            # devices consume ONLY its own band rows, so cross-process
            # byte-identity of non-owned rows is irrelevant, and REAL
            # divergence is already refused by the fingerprint gate above.
            # make_array_from_callback hands each process exactly its
            # addressable slabs — the same #1100 pattern as the state
            # build — with no global-sized collective program at all.
        host_np = np.asarray(host)
        return jax.make_array_from_callback(
            host_np.shape, rep, lambda idx: host_np[idx])

    if jax.process_count() > 1:
        # Schema gate FIRST (one fixed-shape collective every process
        # reaches): a process-dependent field list or a mixed
        # jax_enable_x64 setting would otherwise desynchronize the
        # per-field gathers below instead of failing with a clear message.
        from jax.experimental import multihost_utils as _mhu

        _g = _mhu.process_allgather(
            _schema_fingerprint(list(array_field_names), n_dev))
        if not bool(np.all(_g == _g[0])):
            raise RuntimeError(
                "make_sharded_ocean_step: the band-geometry SCHEMA differs "
                "across processes (field list / x64 setting / device count "
                f"— gathered {_g.tolist()}). Fix the per-process config "
                "before sharding; the per-field checks below assume one "
                "schema.")

    geom_stacks = {
        name: _replicated_put(
            jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
                      axis=0), name)
        for name in array_field_names
    }
    vmask_stack = _replicated_put(
        jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0),
        "vertex_mask")

    # Static perms for the v north-boundary-row ppermute (band r receives band
    # r+1's v_lower[0] = global v[e]; north band non-target receives 0).
    perm_north, _perm_south = latlon_band_perms(n_dev)

    def _body(state_local, forcing_local, geom_stacks_local,
              vmask_stack_local, dt):
        fw_local, sf_local, sponge_local, t_s_local = forcing_local
        r = jax.lax.axis_index(axis)

        # Rebuild this band's geometry. The stacks are SHARDED P("lat") on
        # their leading band axis, so inside shard_map each device's local
        # view is its own (1, ...) slab — index [0], NOT [r] (stage (iii);
        # [r] was the replicated-stack indexing). Static scalars
        # (n_lat_local, n_lon, radius, dlon, dlat, fold) come from the band
        # template. The fold is inactive + identical on every band here.
        geom_arrays = {name: geom_stacks_local[name][0]
                       for name in array_field_names}
        band_geom = template._replace(**geom_arrays)
        band_vmask = vmask_stack_local[0]

        # Reconstruct the band's nl+1 v-faces from the nl-row v_lower.  band r's
        # north boundary row is band r+1's v_lower[0] (= global v[e]); the north
        # band (no r+1) receives the pole-wall 0 via the ppermute non-target.
        def _reconstruct_v(field):
            if field is None:
                return None
            v_lower = field.data                      # (nl, n_lon[, nlev])
            boundary = jax.lax.ppermute(
                v_lower[0:1], axis, perm_north)        # band r+1's first row -> 0 at top
            v_band = jnp.concatenate([v_lower, boundary], axis=0)  # nl+1 rows
            return field.replace(data=v_band)

        # Convert a returned nl+1 v back to the nl-row v_lower (drop the boundary
        # row, owned by band r+1).
        def _to_v_lower(field):
            if field is None:
                return None
            return field.replace(data=field.data[:-1])

        # Fused v-carrier reconstruction (scaling-M4): under the SAME
        # trace-time switch as the pad aggregation, pack the boundary-row
        # ppermutes of ALL staggered carriers (v + v_mask) into one
        # collective per dtype group — value-identical (a bit-copy
        # exchange; the flag flip re-keys the sharded_step cache below so
        # a reused step object rebuilds).  Default OFF = the historical
        # per-field ppermutes, byte-identical.
        import os as _os
        _fused_v = _os.environ.get(
            "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
        present = [name for name in _V_STAGGERED_STATE_FIELDS
                   if getattr(state_local, name) is not None]
        if _fused_v and len(present) > 1:
            fulls = reconstruct_vface_lower_multi(
                tuple(getattr(state_local, n).data for n in present),
                axis, perm_north)
            v_updates = {n: getattr(state_local, n).replace(data=f)
                         for n, f in zip(present, fulls)}
            for name in _V_STAGGERED_STATE_FIELDS:
                v_updates.setdefault(name, None)
        else:
            v_updates = {name: _reconstruct_v(getattr(state_local, name))
                         for name in _V_STAGGERED_STATE_FIELDS}
        state_band = state_local._replace(**v_updates)

        result = _step_body(model, state_band, dt,
                            grid=band_geom, vertex_mask=band_vmask,
                            freshwater=fw_local, surface_forcing=sf_local,
                            sponge=sponge_local, t_seconds=t_s_local)

        out_updates = {name: _to_v_lower(getattr(result, name))
                       for name in _V_STAGGERED_STATE_FIELDS}
        return result._replace(**out_updates)

    # Build the JITTED shard_map ONCE and cache it — the atm sibling's fix
    # (make_sharded_atm_latlon_step, probe 8561202): a bare shard_map is NOT
    # compilation-cached, so rebuilding it per call re-traced + recompiled the
    # (large, un-jitted) ocean band step EVERY call (~80 s/step at 16x32x3 nd=4
    # on CPU — host tracing, not compute; the bench's scaling number was
    # meaningless). ``dt`` is a TRACED operand (was a mutated closure cell,
    # which also forced the per-call rebuild) so a changing dt does not
    # retrigger compilation.
    _cache = {}
    n_lat_global = int(model.grid.n_lat)

    def _validate_forcing_layout(forcing):
        """Loudly refuse forcing leaves the lat-band shard cannot split.

        Every OMIP forcing field is CELL-CENTERED ``(n_lat, n_lon[, nlev])``
        — there are no v-face ``(n_lat+1, …)`` forcing arrays, so no
        ``v_lower`` handling.  A wrong-leading-dim leaf would otherwise die
        inside shard_map with an opaque divisibility error.
        """
        leaves, _ = jax.tree_util.tree_flatten_with_path(forcing)
        for path, leaf in leaves:
            nd = int(getattr(leaf, "ndim", np.ndim(leaf)))
            if nd == 1:
                # No OMIP forcing field is 1-D; _lat_spec would shard a
                # profile/staggered vector over "lat" silently-wrong
                # (codex r1 #2).
                raise ValueError(
                    f"sharded ocean step: forcing leaf "
                    f"{jax.tree_util.keystr(path)} is 1-D "
                    f"(shape {tuple(leaf.shape)}); forcing must be "
                    f"cell-centered (n_lat, n_lon[, nlev]) arrays or "
                    f"scalars.")
            if nd >= 2 and int(leaf.shape[0]) != n_lat_global:
                raise ValueError(
                    f"sharded ocean step: forcing leaf {jax.tree_util.keystr(path)} "
                    f"has leading dim {leaf.shape[0]} != n_lat "
                    f"({n_lat_global}); forcing must be cell-centered "
                    f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")

    def sharded_step(state, dt, freshwater=None, surface_forcing=None,
                     sponge=None, t_seconds=None, aux=None):
        # ``aux`` (codex/2-proc repro 2026-08-03): the geometry + vmask
        # stacks are SHARDED global arrays; when this wrapper runs INSIDE
        # an outer trace (a bench/driver ``jit``/``scan`` over the step —
        # jit-of-jit inlines the inner call), concrete closure arrays
        # become OUTER-TRACE CONSTANTS and jax's MLIR constant handler
        # tries to fetch their value — impossible for non-addressable
        # arrays (RuntimeError: 'Fetching value ... non-addressable'; the
        # multicontroller lane has been broken this way since the
        # #1370-iii stack sharding). Callers that wrap the step in their
        # own jit MUST thread ``step.aux`` through their jit boundary as
        # an ARGUMENT and pass it back here.
        # ONE forcing operand: None fields drop out of the pytree structure,
        # so specs derived by tree.map skip them automatically and the
        # structure key below distinguishes every None<->array combination.
        forcing = (freshwater, surface_forcing, sponge, t_seconds)
        _validate_forcing_layout((freshwater, surface_forcing, sponge))
        # Cache key = the state's AND forcing's pytree STRUCTURE, plus the
        # forcing leaves' RANKS: in_specs/out_specs are derived from them,
        # so a later call with a different structure (an optional field
        # flipping None <-> Field, a sea-ice lane populating sf.salt_flux,
        # the restoring lane passing no forcing at all) OR a same-field
        # rank change (SpongeForcing.gamma is legitimately 2-D horizontal
        # OR 3-D full-rank — same structure, different _lat_spec; codex r1
        # #1) must rebuild the shard_map rather than reuse stale specs.
        forcing_ndims = tuple(
            int(getattr(leaf, "ndim", np.ndim(leaf)))
            for leaf in jax.tree.leaves(forcing))
        # The SPMD fused-halo switch is read at TRACE time inside the pad
        # dispatch — flipping LEGOESM_LATLON_SPMD_FUSED_HALO on a reused
        # step object must rebuild the shard_map, not reuse a stale jaxpr
        # (codex, audit item 7).
        import os as _os

        _fused_halo = _os.environ.get(
            "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
        key = (jax.tree.structure(state), jax.tree.structure(forcing),
               forcing_ndims, _fused_halo)
        fn = _cache.get(key)
        if fn is None:
            in_spec = jax.tree.map(_lat_spec, state)
            # Forcing leaves are cell-centered -> plain lat-band specs;
            # scalars (t_seconds) replicate, exactly like dt.
            forcing_spec = jax.tree.map(_lat_spec, forcing)
            # Stage (iii): the stacks are banded on their leading axis, so
            # the shard_map spec matches their P("lat") placement (the body
            # indexes its local (1, ...) slab at [0]).
            geom_spec = jax.tree.map(lambda _x: P("lat"), geom_stacks)
            vmask_spec = P("lat")
            # JAX >= 0.8 top-level shard_map takes ``check_vma`` (the
            # replication check); the band halo reads neighbour-rank data so
            # disable it (same as the validated PCG / halo-parity shard_maps).
            fn = jax.jit(shard_map(
                _body,
                mesh=mesh,
                in_specs=(in_spec, forcing_spec, geom_spec, vmask_spec, P()),
                out_specs=in_spec,
                check_vma=False,
            ))
            _cache[key] = fn
        # Arm the SPMD halo backend ONLY around the call, then RESTORE the
        # previous backend (codex finding): leaving it globally armed makes a
        # later serial/full-domain ocean call take SPMD-only branches
        # (axis_index / ppermute / psum in pad_with_pole_bc_lat, conservation,
        # eta_floor) OUTSIDE a shard_map -> crash. The FIRST call traces with
        # the backend armed (baking the SPMD halo/reduction ops into the
        # compiled program); later calls reuse the cached compile, and the
        # arm/restore keeps any interleaved serial path untouched.
        # Save+restore the FULL backend state (backend + MPI topology + SPMD
        # mesh) so a prior "mpi"/"spmd" backend is restored intact: activate_*
        # clears the MPI topology, and set_halo_backend("mpi") REQUIRES a
        # topology (codex).
        from legoesm.grids.halo import (
            get_halo_backend, get_mpi_topology, get_spmd_mesh,
            set_halo_backend, set_spmd_mesh,
        )
        _prev_backend = get_halo_backend()
        _prev_topo = get_mpi_topology()
        _prev_mesh = get_spmd_mesh()
        activate_latlon_spmd_halo(mesh)
        try:
            _geom, _vmask = aux if aux is not None else (geom_stacks,
                                                        vmask_stack)
            return fn(state, forcing, _geom, _vmask, jnp.asarray(dt))
        finally:
            set_spmd_mesh(_prev_mesh)
            set_halo_backend(_prev_backend, _prev_topo)

    # Expose the stacks so outer-jit callers can pass them as arguments
    # (see the ``aux`` note in the signature).
    sharded_step.aux = (geom_stacks, vmask_stack)
    return sharded_step


def make_sharded_ocean_step_global(model, mesh):
    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-37-_RUN_OMIP = _ROOT / "scripts" / "run" / "run_omip.py"
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-38-N_PROC = 2
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-39-
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-40-
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-41-def _load_day23():
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:42:    """Load the JRA55 dispatch harness (synthetic-cache builder) by path."""
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:43:    day23 = _ROOT / "tests" / "unit" / "test_run_omip_jra55_dispatch.py"
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-44-    spec = importlib.util.spec_from_file_location("_day23_mc", day23)
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-45-    mod = importlib.util.module_from_spec(spec)
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-46-    sys.modules["_day23_mc"] = mod
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-47-    spec.loader.exec_module(mod)
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-48-    return mod
--
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-50-
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-51-@pytest.mark.timeout(600)
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-52-def test_run_omip_two_process_selfspawn_multicontroller(tmp_path):
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-53-    from multihost_harness import run_federated
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-54-
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:55:    # Build a tiny JRA55-do synthetic cache once in the PARENT; both workers
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-56-    # read it (read-only) — this drives the richest route-B path: the
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-57-    # gpu-interp block-scan with forcing stacks sharded across processes, the
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-58-    # cross-process gather, and the rank-0 restart write.
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-59-    _day23 = _load_day23()
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-60-    n_lat, n_lon = 16, 32
--
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-76-    def build_cmd(rank, port):
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-77-        cmd = [
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-78-            sys.executable, str(_RUN_OMIP),
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-79-            "--grid", "latlon", "--resolution", f"{n_lat}x{n_lon}",
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-80-            "--nlev", "4",
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:81:            "--forcing-mode", "jra55_do_tropical", "--jra55-cache", str(cache),
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-82-            "--days", "0.05", "--dt", "600",      # ~7 steps
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-83-            "--checkpoint-days", "0.03",          # force a rank-0 restart write
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-84-            # Arm the wallclock path (never exhausts at this size) so the
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-85-            # cross-process exit-consensus all-gather runs each block without
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-86-            # triggering exit — exercises the deadlock-free wallclock check.
--
tests/parallel/test_latlon_ocean_spmd_step.py-160-    """Production-shaped forcing pytrees: the OMIP-populated fields of
tests/parallel/test_latlon_ocean_spmd_step.py-161-    ``OceanSurfaceForcing`` (sw_down/q_net/tau_x/tau_y), a full
tests/parallel/test_latlon_ocean_spmd_step.py-162-    ``FreshwaterForcing`` with a deliberately UNBALANCED net (so the
tests/parallel/test_latlon_ocean_spmd_step.py-163-    ``normalize_freshwater`` global-mean removal is load-bearing), and a
tests/parallel/test_latlon_ocean_spmd_step.py-164-    tracer-only ``SpongeForcing`` (gamma/T_ref/S_ref, u_ref=v_ref=None —
tests/parallel/test_latlon_ocean_spmd_step.py:165:    the run_omip shape).  All cell-centered, matching the JRA55 lanes."""
tests/parallel/test_latlon_ocean_spmd_step.py-166-    from legoesm.ocean.freshwater import FreshwaterForcing
tests/parallel/test_latlon_ocean_spmd_step.py-167-    from legoesm.ocean.sponge import SpongeForcing
tests/parallel/test_latlon_ocean_spmd_step.py-168-    from legoesm.ocean.state import OceanSurfaceForcing
tests/parallel/test_latlon_ocean_spmd_step.py-169-
tests/parallel/test_latlon_ocean_spmd_step.py-170-    rng = np.random.default_rng(7)
--
tests/parallel/test_latlon_ocean_spmd_step.py-205-@pytest.mark.skipif(not _have_sharded_step(),
tests/parallel/test_latlon_ocean_spmd_step.py-206-                    reason="sharded_ocean_step module not present")
tests/parallel/test_latlon_ocean_spmd_step.py-207-def test_latlon_ocean_spmd_forcing_matches_single_device():
tests/parallel/test_latlon_ocean_spmd_step.py-208-    """Forcing-channel parity: the run_omip promotion gate.
tests/parallel/test_latlon_ocean_spmd_step.py-209-
tests/parallel/test_latlon_ocean_spmd_step.py:210:    Exercises through the sharded step every forcing path the JRA55 lanes
tests/parallel/test_latlon_ocean_spmd_step.py-211-    hit — wind stress (the one neighbor-row stencil, cell->v-face via the
tests/parallel/test_latlon_ocean_spmd_step.py-212-    SPMD-aware pads), q_net + shortwave column deposition, virtual salt
tests/parallel/test_latlon_ocean_spmd_step.py-213-    with ``normalize_freshwater=True`` (the band-local-mean hazard: the
tests/parallel/test_latlon_ocean_spmd_step.py-214-    global-mean removal must psum over the "lat" axis inside the body or
tests/parallel/test_latlon_ocean_spmd_step.py-215-    every band subtracts a different correction), sponge tracer
--
tests/parallel/test_latlon_ocean_spmd_step.py-269-
tests/parallel/test_latlon_ocean_spmd_step.py-270-
tests/parallel/test_latlon_ocean_spmd_step.py-271-@pytest.mark.skipif(jax.device_count() < 4,
tests/parallel/test_latlon_ocean_spmd_step.py-272-                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py-273-def test_shard_forcing_stack_latlon_layout():
tests/parallel/test_latlon_ocean_spmd_step.py:274:    """The block-scan stack sharder (run_omip JRA55 lanes) puts the lat axis
tests/parallel/test_latlon_ocean_spmd_step.py-275-    on the ``"lat"`` mesh axis whether the leaf is a stacked
tests/parallel/test_latlon_ocean_spmd_step.py-276-    ``(n_rec, n_lat, n_lon[, nlev])`` record (lat at axis 1) or a bare
tests/parallel/test_latlon_ocean_spmd_step.py-277-    ``(n_lat, n_lon)`` field (lat at axis 0); 1-D metadata and scalars
tests/parallel/test_latlon_ocean_spmd_step.py-278-    replicate; ``None`` / non-array leaves pass through untouched.  A drift
tests/parallel/test_latlon_ocean_spmd_step.py-279-    here silently commits the whole forcing stack to device 0 and serializes
--
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-37-_RUN_OMIP = _ROOT / "scripts" / "run" / "run_omip.py"
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-38-N_PROC = 2
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-39-
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-40-
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-41-def _load_day23():
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:42:    """Load the JRA55 dispatch harness (synthetic-cache builder) by path."""
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:43:    day23 = _ROOT / "tests" / "unit" / "test_run_omip_jra55_dispatch.py"
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-44-    spec = importlib.util.spec_from_file_location("_day23_mc", day23)
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-45-    mod = importlib.util.module_from_spec(spec)
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-46-    sys.modules["_day23_mc"] = mod
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-47-    spec.loader.exec_module(mod)
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-48-    return mod
--
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-50-
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-51-@pytest.mark.timeout(600)
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-52-def test_run_omip_two_process_selfspawn_multicontroller(tmp_path):
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-53-    from multihost_harness import run_federated
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-54-
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:55:    # Build a tiny JRA55-do synthetic cache once in the PARENT; both workers
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-56-    # read it (read-only) — this drives the richest route-B path: the
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-57-    # gpu-interp block-scan with forcing stacks sharded across processes, the
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-58-    # cross-process gather, and the rank-0 restart write.
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-59-    _day23 = _load_day23()
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-60-    n_lat, n_lon = 16, 32
--
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-76-    def build_cmd(rank, port):
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-77-        cmd = [
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-78-            sys.executable, str(_RUN_OMIP),
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-79-            "--grid", "latlon", "--resolution", f"{n_lat}x{n_lon}",
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-80-            "--nlev", "4",
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:81:            "--forcing-mode", "jra55_do_tropical", "--jra55-cache", str(cache),
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-82-            "--days", "0.05", "--dt", "600",      # ~7 steps
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-83-            "--checkpoint-days", "0.03",          # force a rank-0 restart write
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-84-            # Arm the wallclock path (never exhausts at this size) so the
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-85-            # cross-process exit-consensus all-gather runs each block without
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py-86-            # triggering exit — exercises the deadlock-free wallclock check.
--
tests/unit/test_run_omip_latlon_spmd.py-29-
tests/unit/test_run_omip_latlon_spmd.py-30-_THIS = Path(__file__).resolve()
tests/unit/test_run_omip_latlon_spmd.py-31-
tests/unit/test_run_omip_latlon_spmd.py-32-
tests/unit/test_run_omip_latlon_spmd.py-33-def _load_harness():
tests/unit/test_run_omip_latlon_spmd.py:34:    day23 = _THIS.parent / "test_run_omip_jra55_dispatch.py"
tests/unit/test_run_omip_latlon_spmd.py-35-    spec = importlib.util.spec_from_file_location("_day23_spmd", day23)
tests/unit/test_run_omip_latlon_spmd.py-36-    mod = importlib.util.module_from_spec(spec)
tests/unit/test_run_omip_latlon_spmd.py-37-    sys.modules["_day23_spmd"] = mod
tests/unit/test_run_omip_latlon_spmd.py-38-    spec.loader.exec_module(mod)
tests/unit/test_run_omip_latlon_spmd.py-39-    return mod
--
tests/unit/test_run_omip_latlon_spmd.py-129-    assert proc.returncode == 0 and "PARITY-OK" in proc.stdout, (
tests/unit/test_run_omip_latlon_spmd.py-130-        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
tests/unit/test_run_omip_latlon_spmd.py-131-        f"stderr:\n{proc.stderr[-3000:]}")
tests/unit/test_run_omip_latlon_spmd.py-132-
tests/unit/test_run_omip_latlon_spmd.py-133-
tests/unit/test_run_omip_latlon_spmd.py:134:def test_loop_refuses_spmd_with_jra55_single_step():
tests/unit/test_run_omip_latlon_spmd.py:135:    """The single-step JRA55 fallback (``model.step``) cannot run sharded —
tests/unit/test_run_omip_latlon_spmd.py-136-    only the block-scan lanes thread ``spmd_step`` (part 2b)."""
tests/unit/test_run_omip_latlon_spmd.py-137-    _day23 = _load_harness()
tests/unit/test_run_omip_latlon_spmd.py-138-    run_omip = _day23.run_omip
tests/unit/test_run_omip_latlon_spmd.py-139-    grid, z_coord, _config, model, _gt = _day23._make_tiny_latlon_setup(
tests/unit/test_run_omip_latlon_spmd.py-140-        n_lat=8, n_lon=16)
tests/unit/test_run_omip_latlon_spmd.py-141-    state0 = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
tests/unit/test_run_omip_latlon_spmd.py-142-    with pytest.raises(ValueError, match="single-step fallback is unsupported"):
tests/unit/test_run_omip_latlon_spmd.py-143-        run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-144-            model, state0, grid_type="latlon", grid=grid, z_coord=z_coord,
tests/unit/test_run_omip_latlon_spmd.py-145-            dt=600.0, n_steps=1, diag_every=1,
tests/unit/test_run_omip_latlon_spmd.py:146:            jra55_state={"_use_single_step": True},
tests/unit/test_run_omip_latlon_spmd.py-147-            spmd_step=lambda s, d: s)
tests/unit/test_run_omip_latlon_spmd.py-148-
tests/unit/test_run_omip_latlon_spmd.py-149-
tests/unit/test_run_omip_latlon_spmd.py-150-# ---------------------------------------------------------------------------
tests/unit/test_run_omip_latlon_spmd.py:151:# Part 2b: the JRA55 block-scan lanes thread spmd_step through the scan body
tests/unit/test_run_omip_latlon_spmd.py-152-# and shard the per-block forcing stacks.  Both the CPU-interp
tests/unit/test_run_omip_latlon_spmd.py:153:# (_build_jra55_block_fn) and the GPU-interp (_build_jra55_block_fn_interp,
tests/unit/test_run_omip_latlon_spmd.py-154-# the run_omip default) block builders must match the serial block loop to
tests/unit/test_run_omip_latlon_spmd.py-155-# the same re-association floor as the restoring lane.  Subprocess again so
tests/unit/test_run_omip_latlon_spmd.py-156-# the 2-virtual-device XLA flag always takes effect.
tests/unit/test_run_omip_latlon_spmd.py-157-# ---------------------------------------------------------------------------
tests/unit/test_run_omip_latlon_spmd.py-158-
tests/unit/test_run_omip_latlon_spmd.py:159:def _jra55_parity_worker(tmp_dir: str, interp_mode: str) -> None:
tests/unit/test_run_omip_latlon_spmd.py:160:    """Runs in the SUBPROCESS: JRA55 block-scan SPMD-vs-serial parity.
tests/unit/test_run_omip_latlon_spmd.py-161-
tests/unit/test_run_omip_latlon_spmd.py-162-    ``interp_mode`` is ``"cpu"`` (host-interp block) or ``"gpu"`` (in-scan
tests/unit/test_run_omip_latlon_spmd.py-163-    GPU-interp block).  Uses the SAME shard_forcing_stack_latlon layout the
tests/unit/test_run_omip_latlon_spmd.py-164-    driver uses (imported, not re-derived) so the test can't drift from it.
tests/unit/test_run_omip_latlon_spmd.py-165-    """
--
tests/unit/test_run_omip_latlon_spmd.py-190-    cache = _day23._make_synthetic_cache(
tests/unit/test_run_omip_latlon_spmd.py-191-        tmp_path, n_lat=n_lat, n_lon=n_lon, n_records=64)
tests/unit/test_run_omip_latlon_spmd.py-192-    grid, z_coord, _config, model, _gt = _day23._make_tiny_latlon_setup(
tests/unit/test_run_omip_latlon_spmd.py-193-        n_lat=n_lat, n_lon=n_lon)
tests/unit/test_run_omip_latlon_spmd.py-194-    T_woa, S_woa = _day23._make_woa_like_targets(grid, nlev=4)
tests/unit/test_run_omip_latlon_spmd.py:195:    args = _day23._argparse_namespace(jra55_cache=str(cache))
tests/unit/test_run_omip_latlon_spmd.py-196-
tests/unit/test_run_omip_latlon_spmd.py-197-    def _fresh_js():
tests/unit/test_run_omip_latlon_spmd.py:198:        # _setup_jra55_forcing_state does NOT set _gpu_interp (the driver
tests/unit/test_run_omip_latlon_spmd.py-199-        # does, from --gpu-interp); set it here to pick the lane.  A fresh
tests/unit/test_run_omip_latlon_spmd.py-200-        # dict per loop avoids sharing the sponge/target arrays across the
tests/unit/test_run_omip_latlon_spmd.py-201-        # serial and sharded runs.
tests/unit/test_run_omip_latlon_spmd.py:202:        js = run_omip._setup_jra55_forcing_state(
tests/unit/test_run_omip_latlon_spmd.py-203-            args, grid, "latlon", z_coord=z_coord, T_woa=T_woa, S_woa=S_woa)
tests/unit/test_run_omip_latlon_spmd.py-204-        js["_gpu_interp"] = (interp_mode == "gpu")
tests/unit/test_run_omip_latlon_spmd.py-205-        return js
tests/unit/test_run_omip_latlon_spmd.py-206-
tests/unit/test_run_omip_latlon_spmd.py-207-    common = dict(
--
tests/unit/test_run_omip_latlon_spmd.py-210-        checkpoint_days=None, checkpoint_dir=None,
tests/unit/test_run_omip_latlon_spmd.py-211-    )
tests/unit/test_run_omip_latlon_spmd.py-212-
tests/unit/test_run_omip_latlon_spmd.py-213-    state0 = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
tests/unit/test_run_omip_latlon_spmd.py-214-    serial_final, _d, _w, ok_serial, _b = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py:215:        model, state0, jra55_state=_fresh_js(), **common)
tests/unit/test_run_omip_latlon_spmd.py:216:    assert ok_serial, "serial JRA55 loop reported not-ok"
tests/unit/test_run_omip_latlon_spmd.py-217-
tests/unit/test_run_omip_latlon_spmd.py-218-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py-219-    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-220-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-221-    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-222-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py:223:        model, ss0, jra55_state=_fresh_js(),
tests/unit/test_run_omip_latlon_spmd.py-224-        spmd_step=spmd_step,
tests/unit/test_run_omip_latlon_spmd.py-225-        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
tests/unit/test_run_omip_latlon_spmd.py-226-        spmd_shard_stack=partial(shard_forcing_stack_latlon, mesh=dev.mesh),
tests/unit/test_run_omip_latlon_spmd.py-227-        **common)
tests/unit/test_run_omip_latlon_spmd.py:228:    assert ok_spmd, "SPMD JRA55 loop reported not-ok"
tests/unit/test_run_omip_latlon_spmd.py-229-    spmd_final = gather_state_latlon(spmd_final, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-230-
tests/unit/test_run_omip_latlon_spmd.py-231-    # Same re-association floor as the restoring-lane and step-level gates.
tests/unit/test_run_omip_latlon_spmd.py-232-    for nm in ("u", "v", "eta", "T", "S"):
tests/unit/test_run_omip_latlon_spmd.py-233-        np.testing.assert_allclose(
tests/unit/test_run_omip_latlon_spmd.py-234-            np.asarray(getattr(spmd_final, nm).data),
tests/unit/test_run_omip_latlon_spmd.py-235-            np.asarray(getattr(serial_final, nm).data),
tests/unit/test_run_omip_latlon_spmd.py-236-            atol=2e-4, rtol=1e-3,
tests/unit/test_run_omip_latlon_spmd.py:237:            err_msg=f"JRA55 {interp_mode}-interp SPMD {nm} mismatch")
tests/unit/test_run_omip_latlon_spmd.py:238:    print("JRA55-PARITY-OK")
tests/unit/test_run_omip_latlon_spmd.py-239-
tests/unit/test_run_omip_latlon_spmd.py-240-
tests/unit/test_run_omip_latlon_spmd.py:241:def _run_jra55_parity_subprocess(tmp_path, interp_mode: str):
tests/unit/test_run_omip_latlon_spmd.py-242-    env = dict(os.environ)
tests/unit/test_run_omip_latlon_spmd.py-243-    env["JAX_PLATFORMS"] = "cpu"
tests/unit/test_run_omip_latlon_spmd.py-244-    env["JAX_ENABLE_X64"] = "1"
tests/unit/test_run_omip_latlon_spmd.py-245-    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
tests/unit/test_run_omip_latlon_spmd.py-246-    _root = str(_THIS.parents[2])
tests/unit/test_run_omip_latlon_spmd.py-247-    env["PYTHONPATH"] = _root + os.pathsep + env.get("PYTHONPATH", "")
tests/unit/test_run_omip_latlon_spmd.py-248-    proc = subprocess.run(
tests/unit/test_run_omip_latlon_spmd.py:249:        [sys.executable, str(_THIS), "--jra55-parity-worker",
tests/unit/test_run_omip_latlon_spmd.py-250-         str(tmp_path), interp_mode],
tests/unit/test_run_omip_latlon_spmd.py-251-        env=env, capture_output=True, text=True, timeout=870,
tests/unit/test_run_omip_latlon_spmd.py-252-    )
tests/unit/test_run_omip_latlon_spmd.py:253:    assert proc.returncode == 0 and "JRA55-PARITY-OK" in proc.stdout, (
tests/unit/test_run_omip_latlon_spmd.py-254-        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
tests/unit/test_run_omip_latlon_spmd.py-255-        f"stderr:\n{proc.stderr[-3000:]}")
tests/unit/test_run_omip_latlon_spmd.py-256-
tests/unit/test_run_omip_latlon_spmd.py-257-
tests/unit/test_run_omip_latlon_spmd.py-258-@pytest.mark.timeout(900)
tests/unit/test_run_omip_latlon_spmd.py:259:def test_jra55_block_loop_spmd_matches_serial_cpu_interp(tmp_path):
tests/unit/test_run_omip_latlon_spmd.py:260:    _run_jra55_parity_subprocess(tmp_path, "cpu")
tests/unit/test_run_omip_latlon_spmd.py-261-
tests/unit/test_run_omip_latlon_spmd.py-262-
tests/unit/test_run_omip_latlon_spmd.py-263-@pytest.mark.timeout(900)
tests/unit/test_run_omip_latlon_spmd.py:264:def test_jra55_block_loop_spmd_matches_serial_gpu_interp(tmp_path):
tests/unit/test_run_omip_latlon_spmd.py:265:    _run_jra55_parity_subprocess(tmp_path, "gpu")
tests/unit/test_run_omip_latlon_spmd.py-266-
tests/unit/test_run_omip_latlon_spmd.py-267-
tests/unit/test_run_omip_latlon_spmd.py-268-if __name__ == "__main__":
tests/unit/test_run_omip_latlon_spmd.py-269-    if len(sys.argv) >= 3 and sys.argv[1] == "--parity-worker":
tests/unit/test_run_omip_latlon_spmd.py-270-        _parity_worker(sys.argv[2])
tests/unit/test_run_omip_latlon_spmd.py:271:    elif len(sys.argv) >= 4 and sys.argv[1] == "--jra55-parity-worker":
tests/unit/test_run_omip_latlon_spmd.py:272:        _jra55_parity_worker(sys.argv[2], sys.argv[3])
tests/unit/test_run_omip_latlon_spmd.py-273-    else:
tests/unit/test_run_omip_latlon_spmd.py-274-        raise SystemExit("usage: test_run_omip_latlon_spmd.py "
tests/unit/test_run_omip_latlon_spmd.py-275-                         "--parity-worker <tmp_dir> | "
tests/unit/test_run_omip_latlon_spmd.py:276:                         "--jra55-parity-worker <tmp_dir> <cpu|gpu>")
--
tests/unit/test_run_manifest.py-66-    m = build_run_manifest(
tests/unit/test_run_manifest.py-67-        _sample_config(),
tests/unit/test_run_manifest.py-68-        runner_tag="ocean-runners@v1",
tests/unit/test_run_manifest.py-69-        patches=["jerlov_sw.patch"],
tests/unit/test_run_manifest.py-70-        rng_seeds={"master": 42},
tests/unit/test_run_manifest.py:71:        dataset_provenance=[{"name": "JRA55do", "version": "sample"}],
tests/unit/test_run_manifest.py-72-        model_weights_provenance={"checkpoint": "physics.eqx"},
tests/unit/test_run_manifest.py-73-    )
tests/unit/test_run_manifest.py-74-    assert m["reproducibility"]["runner_tag"] == "ocean-runners@v1"
tests/unit/test_run_manifest.py-75-    assert m["reproducibility"]["patches"] == ["jerlov_sw.patch"]
tests/unit/test_run_manifest.py-76-    assert m["result"]["rng_seeds"] == {"master": 42}
tests/unit/test_run_manifest.py:77:    assert m["result"]["dataset_provenance"][0]["name"] == "JRA55do"
tests/unit/test_run_manifest.py-78-    assert m["result"]["model_weights_provenance"] == {"checkpoint": "physics.eqx"}
tests/unit/test_run_manifest.py-79-
tests/unit/test_run_manifest.py-80-
tests/unit/test_run_manifest.py-81-def test_dataset_provenance_entry_shape_and_checksum(tmp_path: Path) -> None:
tests/unit/test_run_manifest.py-82-    """Entry records identity + cheap integrity facts; sha256 is opt-in."""
--
tests/unit/test_run_manifest.py-175-    import numpy as np
tests/unit/test_run_manifest.py-176-
tests/unit/test_run_manifest.py-177-    m = build_run_manifest(
tests/unit/test_run_manifest.py-178-        _sample_config(),
tests/unit/test_run_manifest.py-179-        rng_seeds={"master": np.int64(7)},
tests/unit/test_run_manifest.py:180:        dataset_provenance=[Path("/data/jra55.zarr")],
tests/unit/test_run_manifest.py-181-    )
tests/unit/test_run_manifest.py-182-    seed = m["result"]["rng_seeds"]["master"]
tests/unit/test_run_manifest.py-183-    assert seed == 7 and isinstance(seed, int) and not isinstance(seed, np.integer)
tests/unit/test_run_manifest.py:184:    assert m["result"]["dataset_provenance"] == ["/data/jra55.zarr"]
tests/unit/test_run_manifest.py-185-
tests/unit/test_run_manifest.py-186-
tests/unit/test_run_manifest.py-187-def test_in_memory_manifest_matches_written_file(tmp_path: Path) -> None:
tests/unit/test_run_manifest.py-188-    """build_run_manifest() must equal what write_run_manifest() persists.
tests/unit/test_run_manifest.py-189-
--
tests/unit/test_omip_forcing_fallback.py-1-"""OMIP forcing loaders must not SILENTLY substitute synthetic analytic data.
tests/unit/test_omip_forcing_fallback.py-2-
tests/unit/test_omip_forcing_fallback.py:3:The CORE-II (OMIP-1) and JRA55-do (OMIP-2) loaders fall back to a deterministic
tests/unit/test_omip_forcing_fallback.py-4-synthetic climatology when the on-disk cache is missing.  That fallback is a
tests/unit/test_omip_forcing_fallback.py-5-footgun: a run can look like an OMIP integration while using non-protocol
tests/unit/test_omip_forcing_fallback.py-6-forcing.  These tests pin the two safe behaviours:
tests/unit/test_omip_forcing_fallback.py-7-
tests/unit/test_omip_forcing_fallback.py-8-* ``allow_synthetic=False`` raises ``FileNotFoundError`` (fail-loud — the
--
tests/unit/test_omip_forcing_fallback.py-16-import logging
tests/unit/test_omip_forcing_fallback.py-17-
tests/unit/test_omip_forcing_fallback.py-18-import pytest
tests/unit/test_omip_forcing_fallback.py-19-
tests/unit/test_omip_forcing_fallback.py-20-from legoesm.ocean.forcing.core2 import load_core2_nyf
tests/unit/test_omip_forcing_fallback.py:21:from legoesm.ocean.forcing.jra55_do import load_jra55_do
tests/unit/test_omip_forcing_fallback.py-22-
tests/unit/test_omip_forcing_fallback.py-23-
tests/unit/test_omip_forcing_fallback.py-24-def test_core2_missing_cache_fails_loud_when_disallowed(tmp_path):
tests/unit/test_omip_forcing_fallback.py-25-    with pytest.raises(FileNotFoundError, match="CORE-II NYF cache missing"):
tests/unit/test_omip_forcing_fallback.py-26-        load_core2_nyf(cache_dir=tmp_path, allow_synthetic=False)
--
tests/unit/test_omip_forcing_fallback.py-33-    assert f.u10.shape[0] == 12  # synthetic returned
tests/unit/test_omip_forcing_fallback.py-34-    assert any("SYNTHETIC" in r.message and "OMIP-1" in r.message
tests/unit/test_omip_forcing_fallback.py-35-               for r in caplog.records), "missing loud synthetic-fallback warning"
tests/unit/test_omip_forcing_fallback.py-36-
tests/unit/test_omip_forcing_fallback.py-37-
tests/unit/test_omip_forcing_fallback.py:38:def test_jra55_missing_cache_fails_loud_when_disallowed(tmp_path):
tests/unit/test_omip_forcing_fallback.py:39:    with pytest.raises(FileNotFoundError, match="JRA55-do cache missing"):
tests/unit/test_omip_forcing_fallback.py:40:        load_jra55_do(1990, cache_dir=tmp_path, allow_synthetic=False)
tests/unit/test_omip_forcing_fallback.py-41-
tests/unit/test_omip_forcing_fallback.py-42-
tests/unit/test_omip_forcing_fallback.py:43:def test_jra55_synthetic_fallback_warns(tmp_path, caplog):
tests/unit/test_omip_forcing_fallback.py-44-    with caplog.at_level(logging.WARNING,
tests/unit/test_omip_forcing_fallback.py:45:                         logger="legoesm.ocean.forcing.jra55_do"):
tests/unit/test_omip_forcing_fallback.py:46:        load_jra55_do(1990, cache_dir=tmp_path, allow_synthetic=True)
tests/unit/test_omip_forcing_fallback.py-47-    assert any("SYNTHETIC" in r.message and "OMIP-2" in r.message
tests/unit/test_omip_forcing_fallback.py-48-               for r in caplog.records), "missing loud synthetic-fallback warning"
--
tests/unit/test_run_omip_cli.py-99-    assert args.enable_latlon_spmd is False
tests/unit/test_run_omip_cli.py-100-    with pytest.raises(SystemExit, match="requires --enable-latlon-spmd"):
tests/unit/test_run_omip_cli.py-101-        run_omip_single("latlon", args)
tests/unit/test_run_omip_cli.py-102-
tests/unit/test_run_omip_cli.py-103-
tests/unit/test_run_omip_cli.py:104:def test_jra55_sea_ice_flag_parses():
tests/unit/test_run_omip_cli.py:105:    """--jra55-sea-ice opt-in (default off) drives the prognostic slab ice
tests/unit/test_run_omip_cli.py:106:    wired into the JRA55 scan block loop."""
tests/unit/test_run_omip_cli.py:107:    assert parse_args(["--grid", "latlon"]).jra55_sea_ice is False
tests/unit/test_run_omip_cli.py:108:    assert parse_args(["--grid", "latlon", "--jra55-sea-ice"]).jra55_sea_ice is True
tests/unit/test_run_omip_cli.py-109-
tests/unit/test_run_omip_cli.py-110-
tests/unit/test_run_omip_cli.py-111-def test_surface_stability_scheme_flag_round_trip():
tests/unit/test_run_omip_cli.py-112-    """--surface-stability-scheme parses, defaults byte-identically to
tests/unit/test_run_omip_cli.py-113-    dyer1974, and rejects unknown names (dispatch hardening at argparse)."""
--
tests/unit/test_run_omip_jra55_dispatch.py-1-"""Tests for the tropical-OMIP dispatch in ``scripts/run/run_omip.py``
tests/unit/test_run_omip_jra55_dispatch.py-2-(Item 4 Day 2).
tests/unit/test_run_omip_jra55_dispatch.py-3-
tests/unit/test_run_omip_jra55_dispatch.py:4:Exercises the JRA55-do forcing path:
tests/unit/test_run_omip_jra55_dispatch.py-5-
tests/unit/test_run_omip_jra55_dispatch.py:6:- ``_setup_jra55_forcing_state``: cache validation, grid-shape match,
tests/unit/test_run_omip_jra55_dispatch.py-7-  required-flag enforcement.
tests/unit/test_run_omip_jra55_dispatch.py:8:- ``_jra55_step``: one bulk-flux-driven ocean step on a tiny lat-lon
tests/unit/test_run_omip_jra55_dispatch.py:9:  C-grid against a synthetic JRA55-do cache.
tests/unit/test_run_omip_jra55_dispatch.py-10-
tests/unit/test_run_omip_jra55_dispatch.py-11-The full ``run_omip.py`` driver run is too slow for unit-test latency;
tests/unit/test_run_omip_jra55_dispatch.py-12-we bypass ``run_omip_single`` and exercise the helpers directly with a
tests/unit/test_run_omip_jra55_dispatch.py-13-real lat-lon C-grid model so the wiring (AtmToSurface →
tests/unit/test_run_omip_jra55_dispatch.py-14-ocean_tile_response → FreshwaterForcing → model.step) is genuinely
--
tests/unit/test_run_omip_jra55_dispatch.py-51-    out_dir: Path,
tests/unit/test_run_omip_jra55_dispatch.py-52-    n_lat: int,
tests/unit/test_run_omip_jra55_dispatch.py-53-    n_lon: int,
tests/unit/test_run_omip_jra55_dispatch.py-54-    n_records: int = 8,
tests/unit/test_run_omip_jra55_dispatch.py-55-) -> Path:
tests/unit/test_run_omip_jra55_dispatch.py:56:    """Write a tiny pre-built JRA55-do cache directly to Zarr.
tests/unit/test_run_omip_jra55_dispatch.py-57-
tests/unit/test_run_omip_jra55_dispatch.py-58-    Skips the full build pipeline because we just need the shape +
tests/unit/test_run_omip_jra55_dispatch.py-59-    schema the runtime loader expects.
tests/unit/test_run_omip_jra55_dispatch.py-60-    """
tests/unit/test_run_omip_jra55_dispatch.py-61-    rng = np.random.default_rng(0)
--
tests/unit/test_run_omip_jra55_dispatch.py-122-    ``AttributeError`` — the setup helpers read ``args.<new_flag>``
tests/unit/test_run_omip_jra55_dispatch.py-123-    directly).  Seeding from ``parse_args`` keeps every current AND
tests/unit/test_run_omip_jra55_dispatch.py-124-    future flag present at its production default; ``kwargs`` override.
tests/unit/test_run_omip_jra55_dispatch.py-125-    """
tests/unit/test_run_omip_jra55_dispatch.py-126-    args = run_omip.parse_args(
tests/unit/test_run_omip_jra55_dispatch.py:127:        ["--grid", "latlon", "--forcing-mode", "jra55_do_tropical"])
tests/unit/test_run_omip_jra55_dispatch.py-128-    for k, v in kwargs.items():
tests/unit/test_run_omip_jra55_dispatch.py-129-        setattr(args, k, v)
tests/unit/test_run_omip_jra55_dispatch.py-130-    return args
tests/unit/test_run_omip_jra55_dispatch.py-131-
tests/unit/test_run_omip_jra55_dispatch.py-132-
tests/unit/test_run_omip_jra55_dispatch.py-133-# ============================================================================
tests/unit/test_run_omip_jra55_dispatch.py:134:# _setup_jra55_forcing_state
tests/unit/test_run_omip_jra55_dispatch.py-135-# ============================================================================
tests/unit/test_run_omip_jra55_dispatch.py-136-
tests/unit/test_run_omip_jra55_dispatch.py-137-def test_setup_rejects_non_latlon_grid(tmp_path):
tests/unit/test_run_omip_jra55_dispatch.py-138-    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py:139:    args = _argparse_namespace(jra55_cache=str(cache))
tests/unit/test_run_omip_jra55_dispatch.py-140-    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py-141-    with pytest.raises(ValueError, match=r"supports only --grid"):
tests/unit/test_run_omip_jra55_dispatch.py:142:        run_omip._setup_jra55_forcing_state(args, grid, "cubed_sphere")
tests/unit/test_run_omip_jra55_dispatch.py-143-
tests/unit/test_run_omip_jra55_dispatch.py-144-
tests/unit/test_run_omip_jra55_dispatch.py-145-def test_setup_requires_cache_path(tmp_path):
tests/unit/test_run_omip_jra55_dispatch.py-146-    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py:147:    args = _argparse_namespace(jra55_cache=None)
tests/unit/test_run_omip_jra55_dispatch.py:148:    with pytest.raises(ValueError, match="requires --jra55-cache"):
tests/unit/test_run_omip_jra55_dispatch.py:149:        run_omip._setup_jra55_forcing_state(args, grid, "latlon")
tests/unit/test_run_omip_jra55_dispatch.py-150-
tests/unit/test_run_omip_jra55_dispatch.py-151-
tests/unit/test_run_omip_jra55_dispatch.py-152-def test_setup_rejects_missing_cache(tmp_path):
tests/unit/test_run_omip_jra55_dispatch.py-153-    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py:154:    args = _argparse_namespace(jra55_cache=str(tmp_path / "no_such_cache.zarr"))
tests/unit/test_run_omip_jra55_dispatch.py-155-    with pytest.raises(FileNotFoundError, match="cache not found"):
tests/unit/test_run_omip_jra55_dispatch.py:156:        run_omip._setup_jra55_forcing_state(args, grid, "latlon")
tests/unit/test_run_omip_jra55_dispatch.py-157-
tests/unit/test_run_omip_jra55_dispatch.py-158-
tests/unit/test_run_omip_jra55_dispatch.py-159-def test_setup_rejects_grid_shape_mismatch(tmp_path):
tests/unit/test_run_omip_jra55_dispatch.py-160-    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py-161-    grid, *_ = _make_tiny_latlon_setup(n_lat=8, n_lon=16)  # different
tests/unit/test_run_omip_jra55_dispatch.py:162:    args = _argparse_namespace(jra55_cache=str(cache))
tests/unit/test_run_omip_jra55_dispatch.py-163-    with pytest.raises(ValueError, match="does not match model grid"):
tests/unit/test_run_omip_jra55_dispatch.py:164:        run_omip._setup_jra55_forcing_state(args, grid, "latlon")
tests/unit/test_run_omip_jra55_dispatch.py-165-
tests/unit/test_run_omip_jra55_dispatch.py-166-
tests/unit/test_run_omip_jra55_dispatch.py-167-def test_setup_returns_expected_keys(tmp_path):
tests/unit/test_run_omip_jra55_dispatch.py-168-    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py-169-    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py:170:    args = _argparse_namespace(jra55_cache=str(cache))
tests/unit/test_run_omip_jra55_dispatch.py:171:    state = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
tests/unit/test_run_omip_jra55_dispatch.py-172-    for key in ("cache_path", "ref_year", "lat_2d", "lon_2d", "coupler_cfg",
tests/unit/test_run_omip_jra55_dispatch.py-173-                "co2_ppmv"):
tests/unit/test_run_omip_jra55_dispatch.py-174-        assert key in state
tests/unit/test_run_omip_jra55_dispatch.py-175-    # Geometry shapes
tests/unit/test_run_omip_jra55_dispatch.py-176-    assert state["lat_2d"].shape == (4, 1)
--
tests/unit/test_run_omip_jra55_dispatch.py-184-
tests/unit/test_run_omip_jra55_dispatch.py-185-
tests/unit/test_run_omip_jra55_dispatch.py-186-def test_setup_propagates_co2_override(tmp_path):
tests/unit/test_run_omip_jra55_dispatch.py-187-    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py-188-    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py:189:    args = _argparse_namespace(jra55_cache=str(cache), jra55_co2_ppmv=420.0)
tests/unit/test_run_omip_jra55_dispatch.py:190:    state = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
tests/unit/test_run_omip_jra55_dispatch.py-191-    assert state["co2_ppmv"] == 420.0
tests/unit/test_run_omip_jra55_dispatch.py-192-
tests/unit/test_run_omip_jra55_dispatch.py-193-
tests/unit/test_run_omip_jra55_dispatch.py-194-def test_setup_default_cycle_is_off(tmp_path):
tests/unit/test_run_omip_jra55_dispatch.py-195-    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py-196-    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py:197:    args = _argparse_namespace(jra55_cache=str(cache))
tests/unit/test_run_omip_jra55_dispatch.py:198:    state = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
tests/unit/test_run_omip_jra55_dispatch.py-199-    assert state["cycle"] is False
tests/unit/test_run_omip_jra55_dispatch.py-200-
tests/unit/test_run_omip_jra55_dispatch.py-201-
tests/unit/test_run_omip_jra55_dispatch.py:202:def test_setup_jra55_cycle_flag_propagates(tmp_path):
tests/unit/test_run_omip_jra55_dispatch.py:203:    """--jra55-cycle should set the cycle flag in the forcing state."""
tests/unit/test_run_omip_jra55_dispatch.py-204-    cache = _make_synthetic_cache(tmp_path, n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py-205-    grid, *_ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
tests/unit/test_run_omip_jra55_dispatch.py:206:    args = _argparse_namespace(jra55_cache=str(cache), jra55_cycle=True)
tests/unit/test_run_omip_jra55_dispatch.py:207:    state = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
tests/unit/test_run_omip_jra55_dispatch.py-208-    assert state["cycle"] is True
tests/unit/test_run_omip_jra55_dispatch.py-209-
tests/unit/test_run_omip_jra55_dispatch.py-210-
tests/unit/test_run_omip_jra55_dispatch.py-211-def test_block_path_matches_per_step_path(tmp_path):
tests/unit/test_run_omip_jra55_dispatch.py-212-    """Running 24 steps via the block-scan path produces a state
--
tests/unit/test_run_omip_jra55_dispatch.py-226-    )
tests/unit/test_run_omip_jra55_dispatch.py-227-    T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)

exec
/usr/bin/bash -lc "rg -n -C 3 \"_build_jra55_block_fn(?:_interp)?\\(|block_fn\\(|bfn\\(\" scripts/run/run_omip.py tests -g '*.py'
rg -n -C 3 \"spmd_step=.*lambda|spmd_step=|_build_jra55_block_fn\" tests/unit/test_run_omip*.py tests/parallel/test_run_omip*.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 337ms:
scripts/run/run_omip.py-2300-    return raw_stack, runoff_stack, record_meta
scripts/run/run_omip.py-2301-
scripts/run/run_omip.py-2302-
scripts/run/run_omip.py:2303:def _build_jra55_block_fn(model, jra55_state, dt, spmd_step=None):
scripts/run/run_omip.py-2304-    """Return a JIT-compiled block function that runs N steps via lax.scan.
scripts/run/run_omip.py-2305-
scripts/run/run_omip.py-2306-    Captures everything that's static across the block (sponge, SSS
--
scripts/run/run_omip.py-2380-            return model._step_impl(st, d, **kw)
scripts/run/run_omip.py-2381-
scripts/run/run_omip.py-2382-    @jax.jit
scripts/run/run_omip.py:2383:    def block_fn(state, atm_stack, runoff_stack, block_start_step,
scripts/run/run_omip.py-2384-                 ice_state=None, aux=None):
scripts/run/run_omip.py-2385-        def step_body(carry, idx):
scripts/run/run_omip.py-2386-            if enable_sea_ice:
--
scripts/run/run_omip.py-2512-    return block_fn
scripts/run/run_omip.py-2513-
scripts/run/run_omip.py-2514-
scripts/run/run_omip.py:2515:def _build_jra55_block_fn_interp(model, jra55_state, dt, spmd_step=None):
scripts/run/run_omip.py-2516-    """JIT-compiled block function with GPU-side forcing interpolation.
scripts/run/run_omip.py-2517-
scripts/run/run_omip.py-2518-    Like ``_build_jra55_block_fn``, but instead of receiving pre-
--
scripts/run/run_omip.py-2590-    # drives the solar-zenith insolation geometry (see the scan body).
scripts/run/run_omip.py-2591-    cycle = bool(jra55_state.get("cycle", False))
scripts/run/run_omip.py-2592-
scripts/run/run_omip.py:2593:    def _make_block_fn(n_steps_block):
scripts/run/run_omip.py-2594-        """Create a JIT-compiled block function for a fixed block size."""
scripts/run/run_omip.py-2595-        @jax.jit
scripts/run/run_omip.py:2596:        def block_fn(state, raw_stack, runoff_records, record_days,
scripts/run/run_omip.py-2597-                     block_start_day, block_start_day_forcing,
scripts/run/run_omip.py-2598-                     ice_state=None, aux=None):
scripts/run/run_omip.py-2599-            dt_days = dt / 86400.0
--
scripts/run/run_omip.py-2779-    # Cache block functions by size to avoid recompilation.
scripts/run/run_omip.py-2780-    _block_fn_cache = {}
scripts/run/run_omip.py-2781-
scripts/run/run_omip.py:2782:    def _get_block_fn(n):
scripts/run/run_omip.py-2783-        if n not in _block_fn_cache:
scripts/run/run_omip.py:2784:            _block_fn_cache[n] = _make_block_fn(n)
scripts/run/run_omip.py-2785-        return _block_fn_cache[n]
scripts/run/run_omip.py-2786-
scripts/run/run_omip.py-2787-    return _get_block_fn
--
scripts/run/run_omip.py-3401-        # into every block_fn call as an ARGUMENT (see the builders' note).
scripts/run/run_omip.py-3402-        _spmd_aux = getattr(spmd_step, "aux", None)
scripts/run/run_omip.py-3403-        if use_gpu_interp:
scripts/run/run_omip.py:3404:            _get_block_fn_interp = _build_jra55_block_fn_interp(
scripts/run/run_omip.py-3405-                model, jra55_state, dt, spmd_step=spmd_step)
scripts/run/run_omip.py-3406-            print("  GPU-interp mode: forcing interpolation on GPU")
scripts/run/run_omip.py-3407-            # Pre-load the full JRA55 cache for repeat-year runs to
--
scripts/run/run_omip.py-3412-                    jra55_state)
scripts/run/run_omip.py-3413-                _full_cache = (_fc_all, _fc_days, _fc_len)
scripts/run/run_omip.py-3414-        else:
scripts/run/run_omip.py:3415:            block_fn = _build_jra55_block_fn(model, jra55_state, dt,
scripts/run/run_omip.py-3416-                                             spmd_step=spmd_step)
scripts/run/run_omip.py-3417-            _full_cache = None
scripts/run/run_omip.py-3418-        block_size = max(1, diag_every)
--
scripts/run/run_omip.py-3456-            if use_gpu_interp:
scripts/run/run_omip.py-3457-                bfn = _get_block_fn_interp(actual)
scripts/run/run_omip.py-3458-                if _ice_on:
scripts/run/run_omip.py:3459:                    state, ice_state = bfn(
scripts/run/run_omip.py-3460-                        state, raw_stack, runoff_records,
scripts/run/run_omip.py-3461-                        record_meta["record_days"],
scripts/run/run_omip.py-3462-                        jnp.float64(record_meta["block_start_day"]),
--
scripts/run/run_omip.py-3464-                        ice_state, aux=_spmd_aux,
scripts/run/run_omip.py-3465-                    )
scripts/run/run_omip.py-3466-                else:
scripts/run/run_omip.py:3467:                    state = bfn(
scripts/run/run_omip.py-3468-                        state, raw_stack, runoff_records,
scripts/run/run_omip.py-3469-                        record_meta["record_days"],
scripts/run/run_omip.py-3470-                        jnp.float64(record_meta["block_start_day"]),
--
scripts/run/run_omip.py-3473-                    )
scripts/run/run_omip.py-3474-            else:
scripts/run/run_omip.py-3475-                if _ice_on:
scripts/run/run_omip.py:3476:                    state, ice_state = block_fn(
scripts/run/run_omip.py-3477-                        state, atm_stack, runoff_stack,
scripts/run/run_omip.py-3478-                        jnp.int32(block_start), ice_state, aux=_spmd_aux,
scripts/run/run_omip.py-3479-                    )
scripts/run/run_omip.py-3480-                else:
scripts/run/run_omip.py:3481:                    state = block_fn(
scripts/run/run_omip.py-3482-                        state, atm_stack, runoff_stack,
scripts/run/run_omip.py-3483-                        jnp.int32(block_start), aux=_spmd_aux,
scripts/run/run_omip.py-3484-                    )
--
tests/unit/test_run_omip_jra55_dispatch.py-837-
tests/unit/test_run_omip_jra55_dispatch.py-838-    dt = 5400.0  # half a record interval
tests/unit/test_run_omip_jra55_dispatch.py-839-    probe_model = _make_forcing_probe_model()
tests/unit/test_run_omip_jra55_dispatch.py:840:    get_bfn = run_omip._build_jra55_block_fn_interp(probe_model, js, dt)
tests/unit/test_run_omip_jra55_dispatch.py-841-    state0 = _ForcingProbeState(
tests/unit/test_run_omip_jra55_dispatch.py-842-        T=_ForcingProbeField(data=jnp.zeros((n_lat, n_lon, 1))))
tests/unit/test_run_omip_jra55_dispatch.py-843-    ds = xr.open_zarr(str(cache), decode_times=False)
--
tests/unit/test_run_omip_jra55_dispatch.py-846-    # between records 4 (day 0.5) and 5 (day 0.625) -> alpha = 0.5.
tests/unit/test_run_omip_jra55_dispatch.py-847-    raw_stack, runoff, meta = run_omip._preload_jra55_raw_records(
tests/unit/test_run_omip_jra55_dispatch.py-848-        41, 1, dt, js)
tests/unit/test_run_omip_jra55_dispatch.py:849:    final = get_bfn(1)(
tests/unit/test_run_omip_jra55_dispatch.py-850-        state0, raw_stack, runoff, meta["record_days"],
tests/unit/test_run_omip_jra55_dispatch.py-851-        jnp.float64(meta["block_start_day"]),
tests/unit/test_run_omip_jra55_dispatch.py-852-        jnp.float64(meta["block_start_day_forcing"]),
--
tests/unit/test_run_omip_jra55_dispatch.py-860-    # the final step lands exactly ON the wrap record -> rsds[0].
tests/unit/test_run_omip_jra55_dispatch.py-861-    raw_stack, runoff, meta = run_omip._preload_jra55_raw_records(
tests/unit/test_run_omip_jra55_dispatch.py-862-        63, 2, dt, js)
tests/unit/test_run_omip_jra55_dispatch.py:863:    final = get_bfn(2)(
tests/unit/test_run_omip_jra55_dispatch.py-864-        state0, raw_stack, runoff, meta["record_days"],
tests/unit/test_run_omip_jra55_dispatch.py-865-        jnp.float64(meta["block_start_day"]),
tests/unit/test_run_omip_jra55_dispatch.py-866-        jnp.float64(meta["block_start_day_forcing"]),
--
tests/unit/test_run_omip_jra55_dispatch.py-1180-    """Drive one production block step and return its selected cos_zenith."""
tests/unit/test_run_omip_jra55_dispatch.py-1181-    _install_cos_zenith_tile_probe(monkeypatch)
tests/unit/test_run_omip_jra55_dispatch.py-1182-    probe_model = _make_zenith_probe_model()
tests/unit/test_run_omip_jra55_dispatch.py:1183:    get_bfn = run_omip._build_jra55_block_fn_interp(probe_model, js, dt=5400.0)
tests/unit/test_run_omip_jra55_dispatch.py-1184-    raw_stack, runoff, record_days = _mini_raw_stack(
tests/unit/test_run_omip_jra55_dispatch.py-1185-        block_start_day_forcing, n_lat, n_lon)
tests/unit/test_run_omip_jra55_dispatch.py-1186-    state0 = _ZenithProbeState(
tests/unit/test_run_omip_jra55_dispatch.py-1187-        T=_ZenithProbeField(data=jnp.zeros((n_lat, n_lon, 1))))
tests/unit/test_run_omip_jra55_dispatch.py:1188:    final = get_bfn(1)(
tests/unit/test_run_omip_jra55_dispatch.py-1189-        state0, raw_stack, runoff, record_days,
tests/unit/test_run_omip_jra55_dispatch.py-1190-        jnp.float64(block_start_day),
tests/unit/test_run_omip_jra55_dispatch.py-1191-        jnp.float64(block_start_day_forcing))
--
tests/ocean/unit/test_pgf_tiers.py-236-        return model.step(state, dt, surface_forcing=surface_forcing), None
tests/ocean/unit/test_pgf_tiers.py-237-
tests/ocean/unit/test_pgf_tiers.py-238-    @partial(jax.jit, static_argnames=("n_inner",))
tests/ocean/unit/test_pgf_tiers.py:239:    def block_fn(state, n_inner: int):
tests/ocean/unit/test_pgf_tiers.py-240-        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
tests/ocean/unit/test_pgf_tiers.py-241-        return state
tests/ocean/unit/test_pgf_tiers.py-242-
tests/ocean/unit/test_pgf_tiers.py-243-    speed_max_list = []
tests/ocean/unit/test_pgf_tiers.py-244-
tests/ocean/unit/test_pgf_tiers.py-245-    for day in range(n_days):
tests/ocean/unit/test_pgf_tiers.py:246:        state = block_fn(state, n_inner=steps_per_day)
tests/ocean/unit/test_pgf_tiers.py-247-        u = np.asarray(state.u.data)
tests/ocean/unit/test_pgf_tiers.py-248-        if not np.all(np.isfinite(u)):
tests/ocean/unit/test_pgf_tiers.py-249-            speed_max_list.append(float("nan"))
--
tests/unit/test_run_omip_sea_ice_scan.py-91-    finite and the ice concentration stays a valid [0,1] fraction."""
tests/unit/test_run_omip_sea_ice_scan.py-92-    model, state, js = _setup(tmp_path, sea_ice=True)
tests/unit/test_run_omip_sea_ice_scan.py-93-    dt = 600.0
tests/unit/test_run_omip_sea_ice_scan.py:94:    block_fn = run_omip._build_jra55_block_fn(model, js, dt)
tests/unit/test_run_omip_sea_ice_scan.py-95-    atm_stack, runoff_stack = run_omip._preload_jra55_forcing_block(
tests/unit/test_run_omip_sea_ice_scan.py-96-        0, 4, dt, js,
tests/unit/test_run_omip_sea_ice_scan.py-97-    )
tests/unit/test_run_omip_sea_ice_scan.py:98:    out = block_fn(state, atm_stack, runoff_stack, jnp.int32(0),
tests/unit/test_run_omip_sea_ice_scan.py-99-                   js["ice_state_init"])
tests/unit/test_run_omip_sea_ice_scan.py-100-    assert isinstance(out, tuple) and len(out) == 2, "scan must carry ice"
tests/unit/test_run_omip_sea_ice_scan.py-101-    new_state, new_ice = out
--
tests/unit/test_run_omip_sea_ice_scan.py-113-    ice reproduces the uninterrupted run (no silent ice-pack reset)."""
tests/unit/test_run_omip_sea_ice_scan.py-114-    model, state, js = _setup(tmp_path, sea_ice=True)
tests/unit/test_run_omip_sea_ice_scan.py-115-    dt = 600.0
tests/unit/test_run_omip_sea_ice_scan.py:116:    block_fn = run_omip._build_jra55_block_fn(model, js, dt)
tests/unit/test_run_omip_sea_ice_scan.py-117-    atmA, runA = run_omip._preload_jra55_forcing_block(0, 3, dt, js)
tests/unit/test_run_omip_sea_ice_scan.py-118-    atmB, runB = run_omip._preload_jra55_forcing_block(3, 3, dt, js)
tests/unit/test_run_omip_sea_ice_scan.py-119-    ice0 = js["ice_state_init"]
tests/unit/test_run_omip_sea_ice_scan.py-120-
tests/unit/test_run_omip_sea_ice_scan.py-121-    # Uninterrupted: block A then block B.
tests/unit/test_run_omip_sea_ice_scan.py:122:    s1, ice1 = block_fn(state, atmA, runA, jnp.int32(0), ice0)
tests/unit/test_run_omip_sea_ice_scan.py:123:    s2u, ice2u = block_fn(s1, atmB, runB, jnp.int32(3), ice1)
tests/unit/test_run_omip_sea_ice_scan.py-124-
tests/unit/test_run_omip_sea_ice_scan.py-125-    # Checkpoint after block A, reload ocean + ice, resume block B.
tests/unit/test_run_omip_sea_ice_scan.py-126-    ckpt = tmp_path / "ckpt"
--
tests/unit/test_run_omip_sea_ice_scan.py-133-        assert np.allclose(np.asarray(getattr(ice1r, f).data),
tests/unit/test_run_omip_sea_ice_scan.py-134-                           np.asarray(getattr(ice1, f).data), atol=0, rtol=0)
tests/unit/test_run_omip_sea_ice_scan.py-135-
tests/unit/test_run_omip_sea_ice_scan.py:136:    s2b, ice2b = block_fn(s1r, atmB, runB, jnp.int32(3), ice1r)
tests/unit/test_run_omip_sea_ice_scan.py-137-    # Resume reproduces the uninterrupted run (ocean + ice).
tests/unit/test_run_omip_sea_ice_scan.py-138-    assert np.allclose(np.asarray(ice2b.concentration.data),
tests/unit/test_run_omip_sea_ice_scan.py-139-                       np.asarray(ice2u.concentration.data), atol=1e-10)
--
tests/unit/test_run_omip_sea_ice_scan.py-171-    """Ice OFF: the block returns the ocean state alone (carry unchanged)."""
tests/unit/test_run_omip_sea_ice_scan.py-172-    model, state, js = _setup(tmp_path, sea_ice=False)
tests/unit/test_run_omip_sea_ice_scan.py-173-    dt = 600.0
tests/unit/test_run_omip_sea_ice_scan.py:174:    block_fn = run_omip._build_jra55_block_fn(model, js, dt)
tests/unit/test_run_omip_sea_ice_scan.py-175-    atm_stack, runoff_stack = run_omip._preload_jra55_forcing_block(
tests/unit/test_run_omip_sea_ice_scan.py-176-        0, 4, dt, js,
tests/unit/test_run_omip_sea_ice_scan.py-177-    )
tests/unit/test_run_omip_sea_ice_scan.py:178:    out = block_fn(state, atm_stack, runoff_stack, jnp.int32(0))
tests/unit/test_run_omip_sea_ice_scan.py-179-    # Not a (state, ice) tuple — the ocean LatLonCGridOceanState itself.
tests/unit/test_run_omip_sea_ice_scan.py-180-    assert hasattr(out, "T") and hasattr(out, "S")
tests/unit/test_run_omip_sea_ice_scan.py-181-    assert bool(jnp.all(jnp.isfinite(out.T.data)))
--
tests/ocean/unit/test_omip2_forcing_scan.py-160-    dt = 300.0
tests/ocean/unit/test_omip2_forcing_scan.py-161-    idx = [1, 3]
tests/ocean/unit/test_omip2_forcing_scan.py-162-
tests/ocean/unit/test_omip2_forcing_scan.py:163:    block_fn = build_omip2_scan_block_fn(model, dt, gshape)
tests/ocean/unit/test_omip2_forcing_scan.py:164:    s_scan = block_fn(
tests/ocean/unit/test_omip2_forcing_scan.py-165-        state, stack, nn_i, nn_j, jnp.asarray(idx, dtype=jnp.int32),
tests/ocean/unit/test_omip2_forcing_scan.py-166-        jnp.int32(1))
tests/ocean/unit/test_omip2_forcing_scan.py-167-
tests/unit/test_run_omip_sea_ice_scan.py-4-replaces the freeze-cap SST stand-in with a slab sea-ice tile (open-ocean bulk
tests/unit/test_run_omip_sea_ice_scan.py-5-fluxes scale by f_ocean=1-A; the ice tile feeds basal heat / melt-freeze
tests/unit/test_run_omip_sea_ice_scan.py-6-freshwater / brine salt / stress to the ocean).  These tests exercise the wired
tests/unit/test_run_omip_sea_ice_scan.py:7:``_build_jra55_block_fn`` scan path:
tests/unit/test_run_omip_sea_ice_scan.py-8-
tests/unit/test_run_omip_sea_ice_scan.py-9-* ice ON: the block returns ``(ocean_state, ice_state)``, the ocean stays
tests/unit/test_run_omip_sea_ice_scan.py-10-  finite, and the ice concentration stays a valid fraction;
--
tests/unit/test_run_omip_sea_ice_scan.py-91-    finite and the ice concentration stays a valid [0,1] fraction."""
tests/unit/test_run_omip_sea_ice_scan.py-92-    model, state, js = _setup(tmp_path, sea_ice=True)
tests/unit/test_run_omip_sea_ice_scan.py-93-    dt = 600.0
tests/unit/test_run_omip_sea_ice_scan.py:94:    block_fn = run_omip._build_jra55_block_fn(model, js, dt)
tests/unit/test_run_omip_sea_ice_scan.py-95-    atm_stack, runoff_stack = run_omip._preload_jra55_forcing_block(
tests/unit/test_run_omip_sea_ice_scan.py-96-        0, 4, dt, js,
tests/unit/test_run_omip_sea_ice_scan.py-97-    )
--
tests/unit/test_run_omip_sea_ice_scan.py-113-    ice reproduces the uninterrupted run (no silent ice-pack reset)."""
tests/unit/test_run_omip_sea_ice_scan.py-114-    model, state, js = _setup(tmp_path, sea_ice=True)
tests/unit/test_run_omip_sea_ice_scan.py-115-    dt = 600.0
tests/unit/test_run_omip_sea_ice_scan.py:116:    block_fn = run_omip._build_jra55_block_fn(model, js, dt)
tests/unit/test_run_omip_sea_ice_scan.py-117-    atmA, runA = run_omip._preload_jra55_forcing_block(0, 3, dt, js)
tests/unit/test_run_omip_sea_ice_scan.py-118-    atmB, runB = run_omip._preload_jra55_forcing_block(3, 3, dt, js)
tests/unit/test_run_omip_sea_ice_scan.py-119-    ice0 = js["ice_state_init"]
--
tests/unit/test_run_omip_sea_ice_scan.py-171-    """Ice OFF: the block returns the ocean state alone (carry unchanged)."""
tests/unit/test_run_omip_sea_ice_scan.py-172-    model, state, js = _setup(tmp_path, sea_ice=False)
tests/unit/test_run_omip_sea_ice_scan.py-173-    dt = 600.0
tests/unit/test_run_omip_sea_ice_scan.py:174:    block_fn = run_omip._build_jra55_block_fn(model, js, dt)
tests/unit/test_run_omip_sea_ice_scan.py-175-    atm_stack, runoff_stack = run_omip._preload_jra55_forcing_block(
tests/unit/test_run_omip_sea_ice_scan.py-176-        0, 4, dt, js,
tests/unit/test_run_omip_sea_ice_scan.py-177-    )
--
tests/unit/test_run_omip_latlon_spmd.py-84-    ckpt_dir = tmp_path / "spmd"
tests/unit/test_run_omip_latlon_spmd.py-85-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-86-        model, ss0, checkpoint_dir=ckpt_dir,
tests/unit/test_run_omip_latlon_spmd.py:87:        spmd_step=spmd_step,
tests/unit/test_run_omip_latlon_spmd.py-88-        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
tests/unit/test_run_omip_latlon_spmd.py-89-        **common)
tests/unit/test_run_omip_latlon_spmd.py-90-    assert ok_spmd
--
tests/unit/test_run_omip_latlon_spmd.py-144-            model, state0, grid_type="latlon", grid=grid, z_coord=z_coord,
tests/unit/test_run_omip_latlon_spmd.py-145-            dt=600.0, n_steps=1, diag_every=1,
tests/unit/test_run_omip_latlon_spmd.py-146-            jra55_state={"_use_single_step": True},
tests/unit/test_run_omip_latlon_spmd.py:147:            spmd_step=lambda s, d: s)
tests/unit/test_run_omip_latlon_spmd.py-148-
tests/unit/test_run_omip_latlon_spmd.py-149-
tests/unit/test_run_omip_latlon_spmd.py-150-# ---------------------------------------------------------------------------
tests/unit/test_run_omip_latlon_spmd.py-151-# Part 2b: the JRA55 block-scan lanes thread spmd_step through the scan body
tests/unit/test_run_omip_latlon_spmd.py-152-# and shard the per-block forcing stacks.  Both the CPU-interp
tests/unit/test_run_omip_latlon_spmd.py:153:# (_build_jra55_block_fn) and the GPU-interp (_build_jra55_block_fn_interp,
tests/unit/test_run_omip_latlon_spmd.py-154-# the run_omip default) block builders must match the serial block loop to
tests/unit/test_run_omip_latlon_spmd.py-155-# the same re-association floor as the restoring lane.  Subprocess again so
tests/unit/test_run_omip_latlon_spmd.py-156-# the 2-virtual-device XLA flag always takes effect.
--
tests/unit/test_run_omip_latlon_spmd.py-221-    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-222-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-223-        model, ss0, jra55_state=_fresh_js(),
tests/unit/test_run_omip_latlon_spmd.py:224:        spmd_step=spmd_step,
tests/unit/test_run_omip_latlon_spmd.py-225-        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
tests/unit/test_run_omip_latlon_spmd.py-226-        spmd_shard_stack=partial(shard_forcing_stack_latlon, mesh=dev.mesh),
tests/unit/test_run_omip_latlon_spmd.py-227-        **common)
--
tests/unit/test_run_omip_jra55_dispatch.py-837-
tests/unit/test_run_omip_jra55_dispatch.py-838-    dt = 5400.0  # half a record interval
tests/unit/test_run_omip_jra55_dispatch.py-839-    probe_model = _make_forcing_probe_model()
tests/unit/test_run_omip_jra55_dispatch.py:840:    get_bfn = run_omip._build_jra55_block_fn_interp(probe_model, js, dt)
tests/unit/test_run_omip_jra55_dispatch.py-841-    state0 = _ForcingProbeState(
tests/unit/test_run_omip_jra55_dispatch.py-842-        T=_ForcingProbeField(data=jnp.zeros((n_lat, n_lon, 1))))
tests/unit/test_run_omip_jra55_dispatch.py-843-    ds = xr.open_zarr(str(cache), decode_times=False)
--
tests/unit/test_run_omip_jra55_dispatch.py-995-def test_loop_dispatch_honors_gpu_interp_flag(tmp_path, monkeypatch,
tests/unit/test_run_omip_jra55_dispatch.py-996-                                              gpu_interp, expected_calls):
tests/unit/test_run_omip_jra55_dispatch.py-997-    """F3: _gpu_interp=False must route to the CPU-interp block path
tests/unit/test_run_omip_jra55_dispatch.py:998:    (_build_jra55_block_fn) and the run must complete finite — the CPU
tests/unit/test_run_omip_jra55_dispatch.py-999-    path stays exercised, not just reachable."""
tests/unit/test_run_omip_jra55_dispatch.py-1000-    n_lat, n_lon = 4, 8
tests/unit/test_run_omip_jra55_dispatch.py-1001-    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon,
--
tests/unit/test_run_omip_jra55_dispatch.py-1010-    js["_ocean_mask_2d"] = state.land_mask.data > 0.5
tests/unit/test_run_omip_jra55_dispatch.py-1011-
tests/unit/test_run_omip_jra55_dispatch.py-1012-    calls = {"cpu": 0, "gpu": 0}
tests/unit/test_run_omip_jra55_dispatch.py:1013:    real_cpu = run_omip._build_jra55_block_fn
tests/unit/test_run_omip_jra55_dispatch.py:1014:    real_gpu = run_omip._build_jra55_block_fn_interp
tests/unit/test_run_omip_jra55_dispatch.py-1015-
tests/unit/test_run_omip_jra55_dispatch.py-1016-    def _spy_cpu(*a, **k):
tests/unit/test_run_omip_jra55_dispatch.py-1017-        calls["cpu"] += 1
--
tests/unit/test_run_omip_jra55_dispatch.py-1021-        calls["gpu"] += 1
tests/unit/test_run_omip_jra55_dispatch.py-1022-        return real_gpu(*a, **k)
tests/unit/test_run_omip_jra55_dispatch.py-1023-
tests/unit/test_run_omip_jra55_dispatch.py:1024:    monkeypatch.setattr(run_omip, "_build_jra55_block_fn", _spy_cpu)
tests/unit/test_run_omip_jra55_dispatch.py:1025:    monkeypatch.setattr(run_omip, "_build_jra55_block_fn_interp", _spy_gpu)
tests/unit/test_run_omip_jra55_dispatch.py-1026-
tests/unit/test_run_omip_jra55_dispatch.py-1027-    state_out, _, _, ok, _ = run_omip._run_omip_loop(
tests/unit/test_run_omip_jra55_dispatch.py-1028-        model, state, "latlon", grid, z_coord,
--
tests/unit/test_run_omip_jra55_dispatch.py-1074-# FIX 1 (zenithfix): cycled JRA55 insolation clock must follow the FORCING
tests/unit/test_run_omip_jra55_dispatch.py-1075-# clock, not the raw sim day.
tests/unit/test_run_omip_jra55_dispatch.py-1076-#
tests/unit/test_run_omip_jra55_dispatch.py:1077:# In ``_build_jra55_block_fn_interp`` the recent forcing-clock fix split
tests/unit/test_run_omip_jra55_dispatch.py-1078-# ``day`` (RAW sim day) from ``day_f`` (CYCLED forcing clock).  The solar
tests/unit/test_run_omip_jra55_dispatch.py-1079-# zenith's ``doy``/``hour`` must be locked to the repeated forcing (``day_f``)
tests/unit/test_run_omip_jra55_dispatch.py-1080-# when cycling, otherwise the seasonal (and, for a non-integer cache length,
--
tests/unit/test_run_omip_jra55_dispatch.py-1126-def _install_cos_zenith_tile_probe(monkeypatch):
tests/unit/test_run_omip_jra55_dispatch.py-1127-    """Patch ``ocean_tile_response`` to expose the block's ``cos_zenith``.
tests/unit/test_run_omip_jra55_dispatch.py-1128-
tests/unit/test_run_omip_jra55_dispatch.py:1129:    Must run BEFORE ``_build_jra55_block_fn_interp`` is called — the builder
tests/unit/test_run_omip_jra55_dispatch.py-1130-    does a function-scope ``from legoesm.coupler.coupler import
tests/unit/test_run_omip_jra55_dispatch.py-1131-    ocean_tile_response`` that binds this module attribute at call time.
tests/unit/test_run_omip_jra55_dispatch.py-1132-    """
--
tests/unit/test_run_omip_jra55_dispatch.py-1180-    """Drive one production block step and return its selected cos_zenith."""
tests/unit/test_run_omip_jra55_dispatch.py-1181-    _install_cos_zenith_tile_probe(monkeypatch)
tests/unit/test_run_omip_jra55_dispatch.py-1182-    probe_model = _make_zenith_probe_model()
tests/unit/test_run_omip_jra55_dispatch.py:1183:    get_bfn = run_omip._build_jra55_block_fn_interp(probe_model, js, dt=5400.0)
tests/unit/test_run_omip_jra55_dispatch.py-1184-    raw_stack, runoff, record_days = _mini_raw_stack(
tests/unit/test_run_omip_jra55_dispatch.py-1185-        block_start_day_forcing, n_lat, n_lon)
tests/unit/test_run_omip_jra55_dispatch.py-1186-    state0 = _ZenithProbeState(

exec
/usr/bin/bash -lc "sed -n '925,990p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
rg -n \"spmd_step\\(\" scripts/run/run_omip.py
rg -n \"_dyn_step\\(\" scripts/run/run_omip.py
rg -n \"def block_fn\\(\" scripts/run/run_omip.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 223ms:
        # step object must rebuild the shard_map, not reuse a stale jaxpr
        # (codex, audit item 7).
        import os as _os

        _fused_halo = _os.environ.get(
            "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
        key = (jax.tree.structure(state), jax.tree.structure(forcing),
               forcing_ndims, _fused_halo)
        fn = _cache.get(key)
        if fn is None:
            in_spec = jax.tree.map(_lat_spec, state)
            # Forcing leaves are cell-centered -> plain lat-band specs;
            # scalars (t_seconds) replicate, exactly like dt.
            forcing_spec = jax.tree.map(_lat_spec, forcing)
            # Stage (iii): the stacks are banded on their leading axis, so
            # the shard_map spec matches their P("lat") placement (the body
            # indexes its local (1, ...) slab at [0]).
            geom_spec = jax.tree.map(lambda _x: P("lat"), geom_stacks)
            vmask_spec = P("lat")
            # JAX >= 0.8 top-level shard_map takes ``check_vma`` (the
            # replication check); the band halo reads neighbour-rank data so
            # disable it (same as the validated PCG / halo-parity shard_maps).
            fn = jax.jit(shard_map(
                _body,
                mesh=mesh,
                in_specs=(in_spec, forcing_spec, geom_spec, vmask_spec, P()),
                out_specs=in_spec,
                check_vma=False,
            ))
            _cache[key] = fn
        # Arm the SPMD halo backend ONLY around the call, then RESTORE the
        # previous backend (codex finding): leaving it globally armed makes a
        # later serial/full-domain ocean call take SPMD-only branches
        # (axis_index / ppermute / psum in pad_with_pole_bc_lat, conservation,
        # eta_floor) OUTSIDE a shard_map -> crash. The FIRST call traces with
        # the backend armed (baking the SPMD halo/reduction ops into the
        # compiled program); later calls reuse the cached compile, and the
        # arm/restore keeps any interleaved serial path untouched.
        # Save+restore the FULL backend state (backend + MPI topology + SPMD
        # mesh) so a prior "mpi"/"spmd" backend is restored intact: activate_*
        # clears the MPI topology, and set_halo_backend("mpi") REQUIRES a
        # topology (codex).
        from legoesm.grids.halo import (
            get_halo_backend, get_mpi_topology, get_spmd_mesh,
            set_halo_backend, set_spmd_mesh,
        )
        _prev_backend = get_halo_backend()
        _prev_topo = get_mpi_topology()
        _prev_mesh = get_spmd_mesh()
        activate_latlon_spmd_halo(mesh)
        try:
            _geom, _vmask = aux if aux is not None else (geom_stacks,
                                                        vmask_stack)
            return fn(state, forcing, _geom, _vmask, jnp.asarray(dt))
        finally:
            set_spmd_mesh(_prev_mesh)
            set_halo_backend(_prev_backend, _prev_topo)

    # Expose the stacks so outer-jit callers can pass them as arguments
    # (see the ``aux`` note in the signature).
    sharded_step.aux = (geom_stacks, vmask_stack)
    return sharded_step


def make_sharded_ocean_step_global(model, mesh):
    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
2377:            return spmd_step(st, d, aux=aux, **kw)
2581:            return spmd_step(st, d, aux=aux, **kw)
2376:        def _dyn_step(st, d, aux=None, **kw):
2379:        def _dyn_step(st, d, aux=None, **kw):
2460:            new_state = _dyn_step(
2580:        def _dyn_step(st, d, aux=None, **kw):
2583:        def _dyn_step(st, d, aux=None, **kw):
2734:                new_state = _dyn_step(
3682:        state = _dyn_step(state, dt)
2383:    def block_fn(state, atm_stack, runoff_stack, block_start_step,
2596:        def block_fn(state, raw_stack, runoff_records, record_days,

codex
Reviewed the diff and all JRA55 block-scan paths. The sharded geometry auxiliary is now dynamic through both outer JIT boundaries; all four host invocations match the new signatures. No remaining capture path or functional signature mismatch found.

VERDICT: SHIP
tokens used
52,789
Reviewed the diff and all JRA55 block-scan paths. The sharded geometry auxiliary is now dynamic through both outer JIT boundaries; all four host invocations match the new signatures. No remaining capture path or functional signature mismatch found.

VERDICT: SHIP
