Reading prompt from stdin...
OpenAI Codex v0.145.0
--------
workdir: /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
model: gpt-5.6-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019f95c0-3300-7833-8565-493cb2caf79c
--------
user
# Adversarial review ROUND 3 (final) — `gwd_config_for` resolver

Round 2 you returned 3 CONFIRMED bugs + 5 FALSE-POSITIVES. Two are fixed, one is rebutted
with grep evidence. Verify the fixes, judge the rebuttal, and state whether any substantive
issue remains. Repo root = cwd; run `git diff` for the real change.

## R2-1 CONFIRMED (`mcfarlane_k_wave` not settable) — FIXED, and you were right.

Verified your mechanism: `load_yaml_config` (`run_config_yaml.py:81-86`) raises SystemExit
on any `--config` key that is not an argparse dest, so with no flag the knob was
unreachable by BOTH routes. Added `--mcfarlane-k-wave` (`scripts/run/run_amip.py:1274-1282`)
+ `build_config_from_args` wiring, so `--config mcfarlane_k_wave:` is now legal too. The map
NOTE was corrected to say the flag is the gate for both routes (it previously claimed
`--config` worked without one — that was my error, not a comment nit).

New guard test `test_every_overlaid_scalar_has_a_run_amip_route` asserts EVERY scalar
`gwd_config_for` overlays is settable via a run_amip flag OR a scalar-map entry, so the next
overlaid scalar cannot ship unreachable.

## R2-3 CONFIRMED (goldens stale because of MY fields) — FIXED, and you were right.

I wrongly lumped the 2 new `hines_*` keys into the pre-existing golden drift. Measured with
the golden test's own `_canonical_from_config`:

```
BEFORE fix: williamson_test{2,5} produced-only keys = 26, including
            ['hines_Fmax', 'hines_total_rms_wind']
AFTER  fix: produced-only keys = 24, gwd-related = []
```

Both goldens gained `"hines_Fmax": 0.1` and `"hines_total_rms_wind": 2.0` (plus the earlier
`mcfarlane_k_wave` precision edit). This change's contribution to the golden diff is now
EXACTLY ZERO; the residual 24 keys + `bechtold_downdraft_entrain_rate` 0.0005/0.0003 +
`dycore`/`output` sub-dict drift are the documented upstream drift that fails at clean HEAD.

## R2-2 (legacy AMIP round-trip drops GWD scalars) — REBUTTED with evidence.

Claim: this is not a production drop-path, and it is not made worse by this change.

Evidence:
1. `to_amip_config()` has ZERO production callers.
   `grep -rn "to_amip_config()" --include=*.py packages src scripts | grep -v tests` = empty.
   Only `tests/unit/test_config_roundtrip.py` and `test_run_amip_cli.py` call it.
2. Checkpoint SAVE does not use it: `checkpoint.py:173-174` writes
   `json.dumps(config_to_dict(config))`, and `config_to_dict` (`config.py:2789-2805`) is a
   plain `_asdict()` over the NATIVE ExperimentConfig — it emits `hines_total_rms_wind`,
   `hines_Fmax` and every `mcfarlane_*` scalar. Restore goes through
   `experiment_config_from_dict`, which restores them by name.
3. The legacy flat schema is lossy for ~24 other ExperimentConfig knobs by EXPLICIT design
   (`config.py:2705-2712`: "ExperimentConfig has accreted many newer knobs the flat
   AMIPExperimentConfig never mirrored; drop those instead of raising (drift-proof
   round-trip)"). Extending it for GWD alone would be inconsistent, and would not close the
   general case.
4. `from_amip_config` is only reached for a LEGACY-FORMAT checkpoint
   (`checkpoint.py:42-63` dispatches on the absence of a `"grid"` dict). Such a checkpoint
   predates both fields, so there is no non-default value to lose — and the k_wave fallback
   it uses is now the live field default (R1-6 fix), which the new
   `test_legacy_amip_upconvert_keeps_the_exact_k_wave_default` pins.

If you still call this a bug, name the concrete production call sequence that loses a value.

## Related pre-existing issue I found while checking R2-2 (NOT fixed, flagging)

`config_to_dict` drops `turbulence_override` before serializing but NOT
`gravity_wave_drag_override` (`config.py:2799-2801`). A run with an override therefore
serializes it, and `experiment_config_from_dict` restores it as a plain DICT (only
grid/dycore/output are reconstructed as NamedTuples, `_SUB_CONFIGS`). The failure is LOUD —
`validate_strict` raises "gravity_wave_drag_override must be a GravityWaveDragConfig" — not
silent, and it predates this change. Do you agree it is out of scope here, or does the
resolver's `return override` verbatim make it newly dangerous?

## Test evidence (all on a compute node, JAX_ENABLE_X64=1)

```
test_gwd_config_for.py test_gwd_override.py test_run_amip_cli.py
  test_yaml_to_experiment_config_golden.py test_params_reachability_audit.py
  test_params_config_loader.py test_config_roundtrip.py     253 passed, 4 failed
test_params_reachability_audit test_params_config_loader
  test_tuning_trainable_bounds_sync test_param_specs
  test_no_hardcoded_constants                               3506 passed, 2 skipped
test_mpas_qv_smoothing.py                                   17 passed
```
The 4 failures are the documented pre-existing ones (2x `'mcfarlane+hines' == 'mcfarlane'`
default pin, 2x golden upstream drift with my contribution now zero).

## Final question

Is there any remaining substantive finding — a silently-inert parameter path, a wrong
value reaching a kernel, a broken conservation/JIT property, or a test that would not catch
a regression of what we fixed? If not, say "no substantive findings" and explain why each
residual concern (R1-4 k_wave param-spec deferral, R2-2, the override-serialization note)
is not a blocker.

codex
I’ll inspect the actual diff and the relevant resolver, config loader, CLI, checkpoint, and test coverage to independently verify each claim.
exec
/usr/bin/bash -lc 'git diff --stat && git diff -- scripts/run/run_amip.py src tests packages 2>/dev/null || git diff' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 382ms:
 packages/coupler/legoesm/driver/config.py          | 52 ++++++++++++++++--
 packages/coupler/legoesm/driver/model_driver.py    | 26 +++++----
 .../coupler/legoesm/driver/physics_pipeline.py     | 61 +++++++++++++++++++---
 packages/coupler/legoesm/driver/run_config_yaml.py | 23 +++++++-
 packages/ml/legoesm/tuning.py                      | 18 +++----
 scripts/run/run_amip.py                            | 35 +++++++++++++
 tests/unit/_params_reachability_baseline.py        | 11 ++--
 .../yaml_to_experiment/williamson_test2.json       |  4 +-
 .../yaml_to_experiment/williamson_test5.json       |  4 +-
 tests/unit/test_params_config_loader.py            | 12 +++++
 10 files changed, 202 insertions(+), 44 deletions(-)
diff --git a/packages/coupler/legoesm/driver/config.py b/packages/coupler/legoesm/driver/config.py
index e123b4c33..e3b40740d 100644
--- a/packages/coupler/legoesm/driver/config.py
+++ b/packages/coupler/legoesm/driver/config.py
@@ -983,8 +983,16 @@ class ExperimentConfig(NamedTuple):
     louis_z0: float = 1.0e-4                    # SurfaceLayerConfig.z0 [m]
     louis_Ch_neutral: float = 1.5e-3            # SurfaceLayerConfig.Ch_neutral
     louis_Cd_neutral: float = 1.5e-3            # SurfaceLayerConfig.Cd_neutral
-    mcfarlane_k_wave: float = 6.283185307e-5    # McFarlaneConfig.k_wave [1/m]
-    mcfarlane_N_ref: float = 0.01               # McFarlaneConfig.N_ref [1/s]
+    # Exact 2*pi/100 km — MUST equal McFarlaneConfig.k_wave's own
+    # expression: gwd_config_for overlays this onto the leaf, so a
+    # truncated literal would silently perturb the default kernel.
+    mcfarlane_k_wave: float = 2.0 * math.pi / 100e3  # McFarlaneConfig.k_wave [1/m]
+    # INERT: no McFarlaneConfig field of this name exists — the scheme derives
+    # N from the column state (mcfarlane.py).  Kept only for the positional ABI
+    # + serialized-config compatibility; gwd_config_for deliberately does not
+    # wire it, and it was dropped from the ml/tuning.py catalog so it can no
+    # longer be advertised as a live knob (codex round 1, finding 5).
+    mcfarlane_N_ref: float = 0.01               # INERT (no leaf field)
     mcfarlane_directional_spread: float = 1.0   # McFarlaneConfig.directional_spread
     mcfarlane_tau_max: float = 10.0             # McFarlaneConfig.tau_max [Pa]
     # Morrison ice-microphysics tunables (active when microphysics='morrison'
@@ -1264,6 +1272,14 @@ class ExperimentConfig(NamedTuple):
     # surface temperature).
     mpas_ice_skin_prognostic: bool = False
     mpas_ice_thickness_m: float = 2.0      # climatological ice slab thickness [m]
+    # Hines (1997) non-orographic GWD launch amplitude + saturation flux cap.
+    # Reached through gwd_config_for on EVERY lane (like the mcfarlane_*
+    # scalars above, which were silently inert on every production path until
+    # that resolver existed).  The low-level extratropical westerlies are the
+    # observable lever: hines deposits momentum that decelerates them.
+    # Appended at the tuple END to preserve the positional ABI.
+    hines_total_rms_wind: float = 2.0           # HinesConfig.total_rms_wind [m/s]
+    hines_Fmax: float = 0.1                     # HinesConfig.Fmax [Pa]
 
     def validate_strict(self) -> None:
         """Raise ValueError for invalid parameter values.
@@ -2060,6 +2076,25 @@ class ExperimentConfig(NamedTuple):
                     "refines the same scheme's sub-config, it does not switch "
                     "schemes)"
                 )
+        # GWD scalars that ``gwd_config_for`` overlays onto the kernel leaves.
+        # Every one is a strictly-positive physical quantity (a wavenumber, a
+        # spreading factor, a stress/flux cap, an rms launch wind), and none is
+        # bounds-checked anywhere else on the CLI/--config route (the
+        # ``__param_spec__`` bounds only gate the ``--params`` loader).  Silent
+        # failure modes without this guard: ``hines_Fmax < 0`` makes
+        # ``jnp.clip(drag, 0.0, Fmax)`` return the NEGATIVE cap at every level
+        # (constant spurious drag, no error), and ``hines_total_rms_wind <= 0``
+        # zeroes the amplitude growth so the scheme silently does nothing.
+        # Positivity + finiteness only — the calibratable RANGE stays in
+        # ``__param_spec__`` so it is not maintained twice.
+        for _f in ("mcfarlane_k_wave", "mcfarlane_directional_spread",
+                   "mcfarlane_tau_max", "hines_total_rms_wind", "hines_Fmax"):
+            _v = getattr(self, _f)
+            if not math.isfinite(_v) or _v <= 0.0:
+                errors.append(
+                    f"{_f} must be a positive, finite gravity-wave-drag "
+                    f"parameter, got {_v!r}"
+                )
         _valid_gwd = VALID_GWD
         # A ``+``-joined string composes multiple GWD sources whose tendencies
         # are summed — orographic (mcfarlane/lindzen) and non-orographic
@@ -2443,7 +2478,14 @@ class ExperimentConfig(NamedTuple):
             louis_z0=getattr(amip_cfg, 'louis_z0', 1.0e-4),
             louis_Ch_neutral=getattr(amip_cfg, 'louis_Ch_neutral', 1.5e-3),
             louis_Cd_neutral=getattr(amip_cfg, 'louis_Cd_neutral', 1.5e-3),
-            mcfarlane_k_wave=getattr(amip_cfg, 'mcfarlane_k_wave', 6.283185307e-5),
+            # Default from the live field default (NOT a re-typed literal): the
+            # truncated 6.283185307e-5 that used to sit here is ~3e-11 off the
+            # exact 2*pi/100 km, which gwd_config_for now overlays onto the
+            # kernel — a legacy-checkpoint upconvert would silently run a
+            # different default k_wave (codex round 1, finding 6).
+            mcfarlane_k_wave=getattr(
+                amip_cfg, 'mcfarlane_k_wave',
+                ExperimentConfig._field_defaults['mcfarlane_k_wave']),
             mcfarlane_N_ref=getattr(amip_cfg, 'mcfarlane_N_ref', 0.01),
             mcfarlane_directional_spread=getattr(amip_cfg, 'mcfarlane_directional_spread', 1.0),
             mcfarlane_tau_max=getattr(amip_cfg, 'mcfarlane_tau_max', 10.0),
@@ -2619,7 +2661,9 @@ class ExperimentConfig(NamedTuple):
             louis_z0=getattr(self, 'louis_z0', 1.0e-4),
             louis_Ch_neutral=getattr(self, 'louis_Ch_neutral', 1.5e-3),
             louis_Cd_neutral=getattr(self, 'louis_Cd_neutral', 1.5e-3),
-            mcfarlane_k_wave=getattr(self, 'mcfarlane_k_wave', 6.283185307e-5),
+            mcfarlane_k_wave=getattr(
+                self, 'mcfarlane_k_wave',
+                ExperimentConfig._field_defaults['mcfarlane_k_wave']),
             mcfarlane_N_ref=getattr(self, 'mcfarlane_N_ref', 0.01),
             mcfarlane_directional_spread=getattr(self, 'mcfarlane_directional_spread', 1.0),
             mcfarlane_tau_max=getattr(self, 'mcfarlane_tau_max', 10.0),
diff --git a/packages/coupler/legoesm/driver/model_driver.py b/packages/coupler/legoesm/driver/model_driver.py
index 3844c4880..282531218 100644
--- a/packages/coupler/legoesm/driver/model_driver.py
+++ b/packages/coupler/legoesm/driver/model_driver.py
@@ -35,6 +35,7 @@ from legoesm.driver.config import ExperimentConfig
 from legoesm.driver.physics_pipeline import (
     convection_config_for,
     build_physics_pipeline,
+    gwd_config_for,
     required_microphysics_tracer_slots,
     turbulence_config_for,
     validate_microphysics_tracer_slots,
@@ -5878,7 +5879,6 @@ class ModelDriver:
         from legoesm.atmosphere.physics.microphysics.config import (
             MicrophysicsConfig, apply_microphysics_experiment_flags,
         )
-        from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
         from legoesm.atmosphere.physics.radiation.config import (
             RRTMGPConfig, OzoneProfileConfig,
         )
@@ -6063,7 +6063,7 @@ class ModelDriver:
                 grid_dx_m=float(np.sqrt(np.mean(np.asarray(self.grid.areaCell))))),
             turbulence=turbulence_config_for(cfg),
             microphysics=_micro_cfg,
-            gravity_wave_drag=GravityWaveDragConfig(scheme=cfg.gravity_wave_drag),
+            gravity_wave_drag=gwd_config_for(cfg),
         )
         # Phase D perf: shard the per-column RRTMGP workload across all local
         # devices (issue #273 ``column_mesh``).  rrtmgp is the dominant MPAS
@@ -7524,10 +7524,6 @@ class ModelDriver:
             from legoesm.atmosphere.physics.microphysics.config import (
                 MicrophysicsConfig,
             )
-            from legoesm.atmosphere.physics.gravity_wave_drag.config import (
-                GravityWaveDragConfig,
-            )
-
             # Normalize the CLI radiation alias ("rrtmg") to the
             # physics-layer scheme name — see the _run_mpas rationale.
             _rad_scheme = ("rrtmgp" if cfg.radiation in ("rrtmg", "rrtmgp")
@@ -7574,8 +7570,7 @@ class ModelDriver:
                 convection=convection_config_for(cfg),
                 turbulence=turbulence_config_for(cfg),
                 microphysics=MicrophysicsConfig(scheme=cfg.microphysics),
-                gravity_wave_drag=GravityWaveDragConfig(
-                    scheme=cfg.gravity_wave_drag),
+                gravity_wave_drag=gwd_config_for(cfg),
             )
             _combined_fn = make_physics(
                 phys_cfg, model_type="spectral_pe", dt=DT,
@@ -9397,9 +9392,6 @@ class ModelDriver:
             from legoesm.atmosphere.physics.turbulence import (
                 TurbulenceConfig,
             )
-            from legoesm.atmosphere.physics.gravity_wave_drag.config import (
-                GravityWaveDragConfig,
-            )
             from legoesm.atmosphere.physics.physics_state import (
                 init_physics_state,
             )
@@ -9415,9 +9407,15 @@ class ModelDriver:
                 conv_ncol, _nlev,
                 PhysicsConfig(
                     turbulence=TurbulenceConfig(scheme=cfg.turbulence),
-                    gravity_wave_drag=GravityWaveDragConfig(
-                        scheme=cfg.gravity_wave_drag,
-                    ),
+                    # MUST be the SAME resolved config the pipeline kernel gets
+                    # (_resolve_gwd -> gwd_config_for): init_physics_state sizes
+                    # and fills the gwd_spectrum carry from
+                    # ``prognostic_spectral.n_azimuths/.n_wavenumbers/
+                    # .launch_flux``, and a gravity_wave_drag_override may set
+                    # all three.  A bare config here seeded (ncol,4,20) at the
+                    # default launch flux while the kernel expected the
+                    # override's shape/amplitude (codex round 1, finding 2).
+                    gravity_wave_drag=gwd_config_for(cfg),
                 ),
                 dtype=_seed_dtype,
             )
diff --git a/packages/coupler/legoesm/driver/physics_pipeline.py b/packages/coupler/legoesm/driver/physics_pipeline.py
index 78cb86d52..79f1f44f6 100644
--- a/packages/coupler/legoesm/driver/physics_pipeline.py
+++ b/packages/coupler/legoesm/driver/physics_pipeline.py
@@ -3382,6 +3382,60 @@ def _resolve_turbulence(config):
 # Gravity wave drag resolver
 # ---------------------------------------------------------------------------
 
+def gwd_config_for(config):
+    """The ``GravityWaveDragConfig`` (scheme + tuned per-scheme leaves) to
+    build a gravity-wave-drag kernel from.
+
+    Third member of the resolver family (:func:`turbulence_config_for`,
+    :func:`convection_config_for`): every lane — FV pipeline, MPAS,
+    spectral — previously built ``GravityWaveDragConfig(scheme=...)`` from
+    the scheme STRING alone, so the tuned ExperimentConfig scalars
+    (``mcfarlane_k_wave`` / ``mcfarlane_directional_spread`` /
+    ``mcfarlane_tau_max``, and the Hines launch amplitude) silently never
+    reached the kernel on ANY production AMIP path; only the AIMIP training
+    path consumed them.  Same gap class as the 2026-07-23 convection and
+    hard-sat overrides.
+
+    ``config.gravity_wave_drag_override`` (a full config whose ``scheme``
+    must equal ``config.gravity_wave_drag`` — enforced by
+    ``validate_strict``) still wins verbatim: an explicitly injected config
+    is never second-guessed by the scalar overlay.
+
+    COMPOSITE schemes ("mcfarlane+hines") carry BOTH leaves, so the overlay
+    is applied per-leaf independently of which names appear in the string.
+    Static Python floats — trace-time constants, no retrace.
+    """
+    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
+        GravityWaveDragConfig,
+    )
+
+    scheme = getattr(config, "gravity_wave_drag", "none")
+    override = getattr(config, "gravity_wave_drag_override", None)
+    if override is not None:
+        return override
+    gc = GravityWaveDragConfig(scheme=scheme)
+    if scheme == "none":
+        return gc
+    # McFarlane (orographic) tunables. ``mcfarlane_N_ref`` is deliberately
+    # NOT wired: no McFarlaneConfig field of that name exists (dangling
+    # ExperimentConfig scalar, tracked separately).
+    mc = gc.mcfarlane._replace(
+        k_wave=float(getattr(config, "mcfarlane_k_wave", gc.mcfarlane.k_wave)),
+        directional_spread=float(getattr(
+            config, "mcfarlane_directional_spread",
+            gc.mcfarlane.directional_spread)),
+        tau_max=float(getattr(config, "mcfarlane_tau_max",
+                              gc.mcfarlane.tau_max)),
+    )
+    # Hines (non-orographic) launch amplitude + saturation flux cap.
+    hn = gc.hines._replace(
+        total_rms_wind=float(getattr(config, "hines_total_rms_wind",
+                                     gc.hines.total_rms_wind)),
+        Fmax=float(getattr(config, "hines_Fmax", gc.hines.Fmax)),
+    )
+    return gc._replace(mcfarlane=mc, hines=hn)
+
+
 def _resolve_gwd(config):
     """Resolve gravity wave drag kernel and config from ExperimentConfig.
 
@@ -3399,16 +3453,11 @@ def _resolve_gwd(config):
     if scheme == "none":
         return None, None
 
-    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
-        GravityWaveDragConfig,
-    )
     from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
         get_gwd_fn,
     )
 
-    override = getattr(config, 'gravity_wave_drag_override', None)
-    gc = override if override is not None else GravityWaveDragConfig(scheme=scheme)
-    _name, gwd_fn, gwd_config = get_gwd_fn(gc)
+    _name, gwd_fn, gwd_config = get_gwd_fn(gwd_config_for(config))
     return gwd_fn, gwd_config
 
 
diff --git a/packages/coupler/legoesm/driver/run_config_yaml.py b/packages/coupler/legoesm/driver/run_config_yaml.py
index 6cc1753f5..39794f935 100644
--- a/packages/coupler/legoesm/driver/run_config_yaml.py
+++ b/packages/coupler/legoesm/driver/run_config_yaml.py
@@ -167,8 +167,9 @@ def load_params_config(path) -> dict:
 # physics pipeline ACTUALLY threads from that scalar into the resolved scheme
 # config (``build_cloud_config`` for clouds; ``_resolve_convection`` for
 # sbm/bechtold).  Many ``<prefix>_<field>`` scalars EXIST on ExperimentConfig
-# yet are never read (``_resolve_turbulence``/``_resolve_microphysics``/
-# ``_resolve_gwd`` build default configs and patch only a few fields), so a
+# yet are never read (``_resolve_turbulence``/``_resolve_microphysics`` build
+# default configs and patch only a few fields; ``_resolve_gwd`` did too until
+# ``gwd_config_for`` landed — see its GWD block below), so a
 # name-convention map would silently claim dead overrides.  The membership here
 # is machine-verified end-to-end by ``test_atm_scalar_map_is_pipeline_threaded``
 # (builds a pipeline per entry, asserts the resolved scheme config carries the
@@ -230,6 +231,24 @@ _ATM_SCALAR_PARAM_MAP: dict[str, str] = {
     # remains settable via --config / CLI.  FOLLOW-UP: if --params reachability
     # is wanted back, re-spec the param with an activation-aware transform
     # instead of re-adding a dangling map key.
+    # gravity wave drag -> gwd_config_for (physics_pipeline), which every lane
+    # (FV pipeline / MPAS / spectral) now routes through.  Before it, these
+    # scalars existed on ExperimentConfig but NO production path read them.
+    # ``tau_max`` is tier 3 (a numerics clip), so the tier-1/2 reachability
+    # audit does not require it — it is mapped anyway because the same resolver
+    # threads it and the map's contract is "what the pipeline actually threads".
+    "atm.gwd.HinesConfig.total_rms_wind": "hines_total_rms_wind",
+    "atm.gwd.HinesConfig.Fmax": "hines_Fmax",
+    "atm.gwd.McFarlaneConfig.directional_spread": "mcfarlane_directional_spread",
+    "atm.gwd.McFarlaneConfig.tau_max": "mcfarlane_tau_max",
+    # NOTE: ``mcfarlane_k_wave`` is threaded too but has NO ``__param_spec__``
+    # entry (its computed 2*pi/100e3 default is not a float literal, so the
+    # AST-based spec gate never required one) — there is no qualified name to
+    # map.  Speccing it needs a bounds decision (ml/tuning.py says 1e-5..2e-4,
+    # aimip_params says 1e-5..5e-4).  Until then its ONLY route is the
+    # ``--mcfarlane-k-wave`` flag (which is also what makes the key legal in a
+    # ``--config`` YAML — load_yaml_config rejects any key that is not a parser
+    # dest), and validate_strict guards it positive+finite.
     # NOTE: the idealized GRAY radiation scheme threads a few of its params
     # (tau_equator, tau_pole via same-named scalars; sfc_albedo via the shared
     # `albedo_ocean` scalar) — deliberately NOT in this map.  Gray is not the
diff --git a/packages/ml/legoesm/tuning.py b/packages/ml/legoesm/tuning.py
index d858b602e..f54043c85 100644
--- a/packages/ml/legoesm/tuning.py
+++ b/packages/ml/legoesm/tuning.py
@@ -502,17 +502,13 @@ TUNING_PARAMETERS: dict[str, TuningParameter] = {
             "waves; scales the launch stress tau_0 ~ G_0*rho*N*k*h^2*U."
         ),
     ),
-    "mcfarlane_N_ref": TuningParameter(
-        name="mcfarlane_N_ref",
-        default=0.01,
-        min_val=0.005,
-        max_val=0.025,
-        units="1/s",
-        description="McFarlane GWD reference Brunt-Vaisala frequency",
-        category="gwd",
-        sensitivity="medium",
-        notes="Reference stratification used in the orographic launch-stress closure.",
-    ),
+    # NOTE: ``mcfarlane_N_ref`` was REMOVED from this catalog 2026-07-24.  It
+    # advertised a "reference stratification used in the orographic
+    # launch-stress closure" that does not exist: McFarlaneConfig has no
+    # ``N_ref`` field and mcfarlane_gwd derives N from the column state, so
+    # tuning it changed nothing.  The ExperimentConfig scalar stays (positional
+    # ABI + serialized configs) but is marked INERT there.  Re-add only if a
+    # real reference-stratification closure parameter is introduced.
     "mcfarlane_directional_spread": TuningParameter(
         name="mcfarlane_directional_spread",
         default=1.0,
diff --git a/scripts/run/run_amip.py b/scripts/run/run_amip.py
index dd7d021dc..ed01d26c0 100644
--- a/scripts/run/run_amip.py
+++ b/scripts/run/run_amip.py
@@ -1256,6 +1256,29 @@ def build_arg_parser() -> argparse.ArgumentParser:
                              "filter). "
                              "Setup refuses coefficients above the explicit "
                              "monotonicity bound for the mesh+dt.")
+    parser.add_argument("--hines-total-rms-wind", type=float, default=None,
+                        dest="hines_total_rms_wind",
+                        help="Hines (1997) non-orographic GWD launch RMS wind "
+                             "[m/s] (default 2.0). Larger = stronger "
+                             "non-orographic drag; the low-level extratropical "
+                             "westerly bias is the observable lever.")
+    parser.add_argument("--hines-fmax", type=float, default=None,
+                        dest="hines_Fmax",
+                        help="Hines saturation momentum-flux cap [Pa] "
+                             "(default 0.1).")
+    parser.add_argument("--mcfarlane-tau-max", type=float, default=None,
+                        dest="mcfarlane_tau_max",
+                        help="McFarlane orographic GWD surface stress cap [Pa] "
+                             "(default 10.0). Reaches the kernel on every lane "
+                             "via gwd_config_for.")
+    parser.add_argument("--mcfarlane-k-wave", type=float, default=None,
+                        dest="mcfarlane_k_wave",
+                        help="McFarlane orographic GWD horizontal wavenumber "
+                             "[1/m] (default 2*pi/100 km). Scales the launch "
+                             "stress tau_0 ~ G_0*rho*N*k*h^2*U. No "
+                             "__param_spec__ entry yet (bounds undecided), so "
+                             "this flag is its ONLY route -- --params cannot "
+                             "reach it.")
     parser.add_argument("--hard-sat-ice-curve",
                         action=argparse.BooleanOptionalAction, default=False,
                         dest="hard_sat_ice_curve",
@@ -1759,6 +1782,18 @@ def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
             if args.mpas_qv_smooth_del2_m2s is not None
             else _EXPERIMENT_DEFAULTS.mpas_qv_smooth_del2_m2s),
         hard_sat_ice_curve=args.hard_sat_ice_curve,
+        hines_total_rms_wind=(
+            args.hines_total_rms_wind
+            if args.hines_total_rms_wind is not None
+            else _EXPERIMENT_DEFAULTS.hines_total_rms_wind),
+        hines_Fmax=(args.hines_Fmax if args.hines_Fmax is not None
+                    else _EXPERIMENT_DEFAULTS.hines_Fmax),
+        mcfarlane_tau_max=(
+            args.mcfarlane_tau_max if args.mcfarlane_tau_max is not None
+            else _EXPERIMENT_DEFAULTS.mcfarlane_tau_max),
+        mcfarlane_k_wave=(
+            args.mcfarlane_k_wave if args.mcfarlane_k_wave is not None
+            else _EXPERIMENT_DEFAULTS.mcfarlane_k_wave),
         mpas_ice_skin_prognostic=args.mpas_ice_skin_prognostic,
         mpas_ice_thickness_m=(
             args.mpas_ice_thickness_m
diff --git a/tests/unit/_params_reachability_baseline.py b/tests/unit/_params_reachability_baseline.py
index d83b00a92..f8886c1b1 100644
--- a/tests/unit/_params_reachability_baseline.py
+++ b/tests/unit/_params_reachability_baseline.py
@@ -181,9 +181,9 @@ UNREACHABLE_PARAMS = frozenset({
     'atm.rad.GrayRadiationConfig.tau_equator',
     'atm.rad.GrayRadiationConfig.tau_moist_coeff',
     'atm.rad.GrayRadiationConfig.tau_pole',
-    # atm: HinesConfig (2)
-    'atm.gwd.HinesConfig.Fmax',
-    'atm.gwd.HinesConfig.total_rms_wind',
+    # atm: HinesConfig (0) — Fmax + total_rms_wind became reachable when
+    # gwd_config_for started threading the hines_* ExperimentConfig scalars
+    # into the kernel leaf on every lane (2026-07-24).
     # atm: HoltslagBovilleConfig (13)
     'atm.turb.HoltslagBovilleConfig.Ri_crit',
     'atm.turb.HoltslagBovilleConfig.betah',
@@ -253,9 +253,10 @@ UNREACHABLE_PARAMS = frozenset({
     'atm.conv.MassFluxConfig.cape_threshold',
     'atm.conv.MassFluxConfig.delta_0',
     'atm.conv.MassFluxConfig.tau_adj',
-    # atm: McFarlaneConfig (7)
+    # atm: McFarlaneConfig (6) — directional_spread became reachable via the
+    # mcfarlane_directional_spread scalar + gwd_config_for (2026-07-24); the
+    # rest still have no ExperimentConfig scalar to route through.
     'atm.gwd.McFarlaneConfig.G_0',
-    'atm.gwd.McFarlaneConfig.directional_spread',
     'atm.gwd.McFarlaneConfig.efficiency',
     'atm.gwd.McFarlaneConfig.envelope_scale',
     'atm.gwd.McFarlaneConfig.fcrit2',
diff --git a/tests/unit/golden/yaml_to_experiment/williamson_test2.json b/tests/unit/golden/yaml_to_experiment/williamson_test2.json
index 5f63fdac4..7873bb840 100644
--- a/tests/unit/golden/yaml_to_experiment/williamson_test2.json
+++ b/tests/unit/golden/yaml_to_experiment/williamson_test2.json
@@ -110,6 +110,8 @@
     "vertical_coord": "none"
   },
   "held_suarez_forcing": false,
+  "hines_Fmax": 0.1,
+  "hines_total_rms_wind": 2.0,
   "ic": "default",
   "ic_path": "",
   "insolation_start_doy": null,
@@ -146,7 +148,7 @@
   "lw_diff_factor": 1.66,
   "mcfarlane_N_ref": 0.01,
   "mcfarlane_directional_spread": 1.0,
-  "mcfarlane_k_wave": 6.283185307e-05,
+  "mcfarlane_k_wave": 6.283185307179586e-05,
   "mcfarlane_tau_max": 10.0,
   "micro_substeps": 1,
   "microphysics": "none",
diff --git a/tests/unit/golden/yaml_to_experiment/williamson_test5.json b/tests/unit/golden/yaml_to_experiment/williamson_test5.json
index 1d7fcab02..f60f049f9 100644
--- a/tests/unit/golden/yaml_to_experiment/williamson_test5.json
+++ b/tests/unit/golden/yaml_to_experiment/williamson_test5.json
@@ -110,6 +110,8 @@
     "vertical_coord": "none"
   },
   "held_suarez_forcing": false,
+  "hines_Fmax": 0.1,
+  "hines_total_rms_wind": 2.0,
   "ic": "default",
   "ic_path": "",
   "insolation_start_doy": null,
@@ -146,7 +148,7 @@
   "lw_diff_factor": 1.66,
   "mcfarlane_N_ref": 0.01,
   "mcfarlane_directional_spread": 1.0,
-  "mcfarlane_k_wave": 6.283185307e-05,
+  "mcfarlane_k_wave": 6.283185307179586e-05,
   "mcfarlane_tau_max": 10.0,
   "micro_substeps": 1,
   "microphysics": "none",
diff --git a/tests/unit/test_params_config_loader.py b/tests/unit/test_params_config_loader.py
index 8b45a53c1..55720a2a0 100644
--- a/tests/unit/test_params_config_loader.py
+++ b/tests/unit/test_params_config_loader.py
@@ -171,8 +171,14 @@ def test_atm_scalar_map_is_pipeline_threaded():
         "P3Config": {"microphysics": "p3"},
         "SeifertBehengConfig": {"microphysics": "seifert_beheng"},
         "ThompsonConfig": {"microphysics": "thompson"},
+        # GWD: gwd_config_for overlays the mcfarlane_*/hines_* scalars onto the
+        # scheme leaf, and get_gwd_fn hands the pipeline that LEAF as gwd_config.
+        "HinesConfig": {"gravity_wave_drag": "hines"},
+        "McFarlaneConfig": {"gravity_wave_drag": "mcfarlane"},
     }
     resolved_attr = {
+        "HinesConfig": "gwd_config",
+        "McFarlaneConfig": "gwd_config",
         "SBMConfig": "convection_config",
         "BechtoldConfig": "convection_config",
         "TiedtkeConfig": "convection_config",
@@ -278,6 +284,12 @@ def test_atm_scalar_map_has_no_under_claim():
          lambda pipe, field: getattr(pipe.micro_config, field, None)),
         ("ThompsonConfig", "", {"microphysics": "thompson"},
          lambda pipe, field: getattr(pipe.micro_config, field, None)),
+        # GWD families: _resolve_gwd -> gwd_config_for overlays the flat
+        # scalars onto the scheme leaf, which get_gwd_fn returns as gwd_config.
+        ("HinesConfig", "hines_", {"gravity_wave_drag": "hines"},
+         lambda pipe, field: getattr(pipe.gwd_config, field, None)),
+        ("McFarlaneConfig", "mcfarlane_", {"gravity_wave_drag": "mcfarlane"},
+         lambda pipe, field: getattr(pipe.gwd_config, field, None)),
     ]
     # Companion drift-guard: the family list scanned below must exactly match
     # the config classes present in the verified allowlist map.  The selector /

exec
/usr/bin/bash -lc 'git status --short && rg -n -C 5 "every_overlaid_scalar|gwd_config_for|gravity_wave_drag_override|def to_amip_config|def config_to_dict|def experiment_config_from_dict|_SUB_CONFIGS|mcfarlane_k_wave" packages scripts/run tests/unit | head -n 1000' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 692ms:
 M packages/coupler/legoesm/driver/config.py
 M packages/coupler/legoesm/driver/model_driver.py
 M packages/coupler/legoesm/driver/physics_pipeline.py
 M packages/coupler/legoesm/driver/run_config_yaml.py
 M packages/ml/legoesm/tuning.py
 M scripts/run/run_amip.py
 M tests/unit/_params_reachability_baseline.py
 M tests/unit/golden/yaml_to_experiment/williamson_test2.json
 M tests/unit/golden/yaml_to_experiment/williamson_test5.json
 M tests/unit/test_params_config_loader.py
?? .physics-validator/gwd_config_for/
?? tests/unit/test_gwd_config_for.py
scripts/run/run_amip.py-1268-                             "(default 0.1).")
scripts/run/run_amip.py-1269-    parser.add_argument("--mcfarlane-tau-max", type=float, default=None,
scripts/run/run_amip.py-1270-                        dest="mcfarlane_tau_max",
scripts/run/run_amip.py-1271-                        help="McFarlane orographic GWD surface stress cap [Pa] "
scripts/run/run_amip.py-1272-                             "(default 10.0). Reaches the kernel on every lane "
scripts/run/run_amip.py:1273:                             "via gwd_config_for.")
scripts/run/run_amip.py-1274-    parser.add_argument("--mcfarlane-k-wave", type=float, default=None,
scripts/run/run_amip.py:1275:                        dest="mcfarlane_k_wave",
scripts/run/run_amip.py-1276-                        help="McFarlane orographic GWD horizontal wavenumber "
scripts/run/run_amip.py-1277-                             "[1/m] (default 2*pi/100 km). Scales the launch "
scripts/run/run_amip.py-1278-                             "stress tau_0 ~ G_0*rho*N*k*h^2*U. No "
scripts/run/run_amip.py-1279-                             "__param_spec__ entry yet (bounds undecided), so "
scripts/run/run_amip.py-1280-                             "this flag is its ONLY route -- --params cannot "
--
scripts/run/run_amip.py-1789-        hines_Fmax=(args.hines_Fmax if args.hines_Fmax is not None
scripts/run/run_amip.py-1790-                    else _EXPERIMENT_DEFAULTS.hines_Fmax),
scripts/run/run_amip.py-1791-        mcfarlane_tau_max=(
scripts/run/run_amip.py-1792-            args.mcfarlane_tau_max if args.mcfarlane_tau_max is not None
scripts/run/run_amip.py-1793-            else _EXPERIMENT_DEFAULTS.mcfarlane_tau_max),
scripts/run/run_amip.py:1794:        mcfarlane_k_wave=(
scripts/run/run_amip.py:1795:            args.mcfarlane_k_wave if args.mcfarlane_k_wave is not None
scripts/run/run_amip.py:1796:            else _EXPERIMENT_DEFAULTS.mcfarlane_k_wave),
scripts/run/run_amip.py-1797-        mpas_ice_skin_prognostic=args.mpas_ice_skin_prognostic,
scripts/run/run_amip.py-1798-        mpas_ice_thickness_m=(
scripts/run/run_amip.py-1799-            args.mpas_ice_thickness_m
scripts/run/run_amip.py-1800-            if args.mpas_ice_thickness_m is not None
scripts/run/run_amip.py-1801-            else _EXPERIMENT_DEFAULTS.mpas_ice_thickness_m),
--
tests/unit/test_gwd_config_for.py:1:"""``gwd_config_for`` — the third resolver (turbulence / convection / GWD).
tests/unit/test_gwd_config_for.py-2-
tests/unit/test_gwd_config_for.py-3-Every lane (FV pipeline, MPAS, spectral) previously built
tests/unit/test_gwd_config_for.py-4-``GravityWaveDragConfig(scheme=...)`` from the scheme STRING alone, so the
tests/unit/test_gwd_config_for.py-5-tuned ExperimentConfig scalars never reached the kernel on ANY production
tests/unit/test_gwd_config_for.py-6-AMIP path (only the AIMIP training path consumed them).  This pins the
--
tests/unit/test_gwd_config_for.py-16-    DycoreConfig,
tests/unit/test_gwd_config_for.py-17-    ExperimentConfig,
tests/unit/test_gwd_config_for.py-18-    GridConfig,
tests/unit/test_gwd_config_for.py-19-    OutputConfig,
tests/unit/test_gwd_config_for.py-20-)
tests/unit/test_gwd_config_for.py:21:from legoesm.driver.physics_pipeline import gwd_config_for
tests/unit/test_gwd_config_for.py-22-
tests/unit/test_gwd_config_for.py-23-
tests/unit/test_gwd_config_for.py-24-def _cfg(**kw):
tests/unit/test_gwd_config_for.py-25-    return ExperimentConfig(
tests/unit/test_gwd_config_for.py-26-        grid=GridConfig(grid_type="mpas", resolution=1, nlev=8,
--
tests/unit/test_gwd_config_for.py-38-])
tests/unit/test_gwd_config_for.py-39-def test_defaults_match_bare_config(scheme):
tests/unit/test_gwd_config_for.py-40-    """Untouched scalars reproduce the bare config EXACTLY — whole tuple, every
tests/unit/test_gwd_config_for.py-41-    scheme.  Byte-identical defaults are the precondition for wiring the overlay
tests/unit/test_gwd_config_for.py-42-    into lanes that previously built the bare config."""
tests/unit/test_gwd_config_for.py:43:    assert (gwd_config_for(_cfg(gravity_wave_drag=scheme))
tests/unit/test_gwd_config_for.py-44-            == GravityWaveDragConfig(scheme=scheme))
tests/unit/test_gwd_config_for.py-45-
tests/unit/test_gwd_config_for.py-46-
tests/unit/test_gwd_config_for.py-47-def test_none_scheme_short_circuits():
tests/unit/test_gwd_config_for.py:48:    got = gwd_config_for(_cfg(gravity_wave_drag="none"))
tests/unit/test_gwd_config_for.py-49-    assert got.scheme == "none"
tests/unit/test_gwd_config_for.py-50-    assert got == GravityWaveDragConfig(scheme="none")
tests/unit/test_gwd_config_for.py-51-
tests/unit/test_gwd_config_for.py-52-
tests/unit/test_gwd_config_for.py-53-def test_mcfarlane_scalars_reach_the_leaf():
tests/unit/test_gwd_config_for.py:54:    got = gwd_config_for(_cfg(
tests/unit/test_gwd_config_for.py-55-        gravity_wave_drag="mcfarlane",
tests/unit/test_gwd_config_for.py-56-        mcfarlane_tau_max=4.0,
tests/unit/test_gwd_config_for.py:57:        mcfarlane_k_wave=1.0e-4,
tests/unit/test_gwd_config_for.py-58-        mcfarlane_directional_spread=0.5,
tests/unit/test_gwd_config_for.py-59-    ))
tests/unit/test_gwd_config_for.py-60-    assert got.mcfarlane.tau_max == 4.0
tests/unit/test_gwd_config_for.py-61-    assert got.mcfarlane.k_wave == 1.0e-4
tests/unit/test_gwd_config_for.py-62-    assert got.mcfarlane.directional_spread == 0.5
tests/unit/test_gwd_config_for.py-63-
tests/unit/test_gwd_config_for.py-64-
tests/unit/test_gwd_config_for.py-65-def test_hines_scalars_reach_the_leaf():
tests/unit/test_gwd_config_for.py:66:    got = gwd_config_for(_cfg(
tests/unit/test_gwd_config_for.py-67-        gravity_wave_drag="hines",
tests/unit/test_gwd_config_for.py-68-        hines_total_rms_wind=1.2,
tests/unit/test_gwd_config_for.py-69-        hines_Fmax=0.05,
tests/unit/test_gwd_config_for.py-70-    ))
tests/unit/test_gwd_config_for.py-71-    assert got.hines.total_rms_wind == 1.2
tests/unit/test_gwd_config_for.py-72-    assert got.hines.Fmax == 0.05
tests/unit/test_gwd_config_for.py-73-
tests/unit/test_gwd_config_for.py-74-
tests/unit/test_gwd_config_for.py-75-def test_composite_scheme_carries_both_tuned_leaves():
tests/unit/test_gwd_config_for.py-76-    """'mcfarlane+hines' runs BOTH kernels — both overlays must apply."""
tests/unit/test_gwd_config_for.py:77:    got = gwd_config_for(_cfg(
tests/unit/test_gwd_config_for.py-78-        gravity_wave_drag="mcfarlane+hines",
tests/unit/test_gwd_config_for.py-79-        mcfarlane_tau_max=6.0,
tests/unit/test_gwd_config_for.py-80-        hines_total_rms_wind=1.5,
tests/unit/test_gwd_config_for.py-81-    ))
tests/unit/test_gwd_config_for.py-82-    assert got.scheme == "mcfarlane+hines"
--
tests/unit/test_gwd_config_for.py-86-
tests/unit/test_gwd_config_for.py-87-def test_explicit_override_wins_verbatim():
tests/unit/test_gwd_config_for.py-88-    """An injected full config is never second-guessed by the overlay."""
tests/unit/test_gwd_config_for.py-89-    inj = GravityWaveDragConfig(scheme="mcfarlane")
tests/unit/test_gwd_config_for.py-90-    inj = inj._replace(mcfarlane=inj.mcfarlane._replace(tau_max=99.0))
tests/unit/test_gwd_config_for.py:91:    got = gwd_config_for(_cfg(
tests/unit/test_gwd_config_for.py-92-        gravity_wave_drag="mcfarlane",
tests/unit/test_gwd_config_for.py:93:        gravity_wave_drag_override=inj,
tests/unit/test_gwd_config_for.py-94-        mcfarlane_tau_max=4.0,   # must NOT win over the override
tests/unit/test_gwd_config_for.py-95-    ))
tests/unit/test_gwd_config_for.py-96-    assert got is inj
tests/unit/test_gwd_config_for.py-97-    assert got.mcfarlane.tau_max == 99.0
tests/unit/test_gwd_config_for.py-98-
tests/unit/test_gwd_config_for.py-99-
tests/unit/test_gwd_config_for.py-100-def test_other_leaves_untouched():
tests/unit/test_gwd_config_for.py-101-    """Overlaying mcfarlane/hines leaves the other scheme leaves at defaults."""
tests/unit/test_gwd_config_for.py:102:    got = gwd_config_for(_cfg(gravity_wave_drag="mcfarlane",
tests/unit/test_gwd_config_for.py-103-                              mcfarlane_tau_max=4.0))
tests/unit/test_gwd_config_for.py-104-    bare = GravityWaveDragConfig(scheme="mcfarlane")
tests/unit/test_gwd_config_for.py-105-    assert got.rayleigh == bare.rayleigh
tests/unit/test_gwd_config_for.py-106-    assert got.lindzen == bare.lindzen
tests/unit/test_gwd_config_for.py-107-    assert got.e3sm_cam == bare.e3sm_cam
--
tests/unit/test_gwd_config_for.py-135-        "--mcfarlane-tau-max", "4.0", "--mcfarlane-k-wave", "1.0e-4",
tests/unit/test_gwd_config_for.py-136-    ]), parser))
tests/unit/test_gwd_config_for.py-137-    assert cfg.hines_total_rms_wind == 1.2
tests/unit/test_gwd_config_for.py-138-    assert cfg.hines_Fmax == 0.05
tests/unit/test_gwd_config_for.py-139-    assert cfg.mcfarlane_tau_max == 4.0
tests/unit/test_gwd_config_for.py:140:    assert cfg.mcfarlane_k_wave == 1.0e-4
tests/unit/test_gwd_config_for.py:141:    assert gwd_config_for(cfg).hines.total_rms_wind == 1.2
tests/unit/test_gwd_config_for.py-142-
tests/unit/test_gwd_config_for.py-143-
tests/unit/test_gwd_config_for.py:144:def test_every_overlaid_scalar_has_a_run_amip_route():
tests/unit/test_gwd_config_for.py:145:    """Each scalar gwd_config_for threads must be settable by an operator.
tests/unit/test_gwd_config_for.py-146-    ``--config`` YAML keys must be argparse dests (load_yaml_config rejects
tests/unit/test_gwd_config_for.py-147-    anything else), so a flag is the gate for BOTH routes; the --params route
tests/unit/test_gwd_config_for.py:148:    additionally needs a _ATM_SCALAR_PARAM_MAP entry, which mcfarlane_k_wave
tests/unit/test_gwd_config_for.py-149-    lacks by design (no __param_spec__ bounds yet)."""
tests/unit/test_gwd_config_for.py-150-    from legoesm.driver.run_config_yaml import build_atm_scalar_param_map
tests/unit/test_gwd_config_for.py-151-    from scripts.run.run_amip import build_arg_parser
tests/unit/test_gwd_config_for.py-152-
tests/unit/test_gwd_config_for.py-153-    dests = {a.dest for a in build_arg_parser()._actions}
tests/unit/test_gwd_config_for.py:154:    overlaid = ("mcfarlane_k_wave", "mcfarlane_directional_spread",
tests/unit/test_gwd_config_for.py-155-                "mcfarlane_tau_max", "hines_total_rms_wind", "hines_Fmax")
tests/unit/test_gwd_config_for.py-156-    # mcfarlane_directional_spread is --params-only (mapped, no flag); every
tests/unit/test_gwd_config_for.py-157-    # other overlaid scalar must have a flag.
tests/unit/test_gwd_config_for.py-158-    amap = set(build_atm_scalar_param_map().values())
tests/unit/test_gwd_config_for.py-159-    for f in overlaid:
tests/unit/test_gwd_config_for.py-160-        assert f in dests or f in amap, (
tests/unit/test_gwd_config_for.py:161:            f"{f} is overlaid onto the GWD kernel by gwd_config_for but an "
tests/unit/test_gwd_config_for.py-162-            "operator cannot set it: no run_amip flag AND no scalar-map entry."
tests/unit/test_gwd_config_for.py-163-        )
tests/unit/test_gwd_config_for.py-164-
tests/unit/test_gwd_config_for.py-165-
tests/unit/test_gwd_config_for.py-166-def test_physics_state_seed_honours_a_nested_override():
--
tests/unit/test_gwd_config_for.py-180-        scheme="prognostic_spectral",
tests/unit/test_gwd_config_for.py-181-        prognostic_spectral=PrognosticSpectralConfig(
tests/unit/test_gwd_config_for.py-182-            n_azimuths=8, n_wavenumbers=6, launch_flux=2.5e-3),
tests/unit/test_gwd_config_for.py-183-    )
tests/unit/test_gwd_config_for.py-184-    cfg = _cfg(gravity_wave_drag="prognostic_spectral",
tests/unit/test_gwd_config_for.py:185:               gravity_wave_drag_override=inj)
tests/unit/test_gwd_config_for.py-186-    ps = init_physics_state(
tests/unit/test_gwd_config_for.py-187-        3, 8,
tests/unit/test_gwd_config_for.py-188-        PhysicsConfig(turbulence=TurbulenceConfig(scheme="none"),
tests/unit/test_gwd_config_for.py:189:                      gravity_wave_drag=gwd_config_for(cfg)),
tests/unit/test_gwd_config_for.py-190-    )
tests/unit/test_gwd_config_for.py-191-    assert ps.gwd_spectrum.shape == (3, 8, 6)
tests/unit/test_gwd_config_for.py-192-    assert float(ps.gwd_spectrum[0, 0, 0]) == 2.5e-3
tests/unit/test_gwd_config_for.py-193-
tests/unit/test_gwd_config_for.py-194-
tests/unit/test_gwd_config_for.py-195-@pytest.mark.parametrize("field,bad", [
tests/unit/test_gwd_config_for.py-196-    ("hines_total_rms_wind", -1.0),
tests/unit/test_gwd_config_for.py-197-    ("hines_total_rms_wind", 0.0),
tests/unit/test_gwd_config_for.py-198-    ("hines_Fmax", -0.1),
tests/unit/test_gwd_config_for.py-199-    ("mcfarlane_tau_max", float("nan")),
tests/unit/test_gwd_config_for.py:200:    ("mcfarlane_k_wave", float("inf")),
tests/unit/test_gwd_config_for.py-201-    ("mcfarlane_directional_spread", -1.0),
tests/unit/test_gwd_config_for.py-202-])
tests/unit/test_gwd_config_for.py-203-def test_validate_strict_rejects_nonpositive_or_nonfinite(field, bad):
tests/unit/test_gwd_config_for.py-204-    """A negative Fmax makes clip(drag, 0, Fmax) return the NEGATIVE cap at
tests/unit/test_gwd_config_for.py-205-    every level (constant spurious drag, no error); a non-positive launch rms
--
tests/unit/test_gwd_config_for.py-218-    from legoesm.forcing.amip_config import AMIPExperimentConfig
tests/unit/test_gwd_config_for.py-219-
tests/unit/test_gwd_config_for.py-220-    # The legacy flat schema has NO mcfarlane_* fields, so the converter's
tests/unit/test_gwd_config_for.py-221-    # getattr fallback is what supplies k_wave.
tests/unit/test_gwd_config_for.py-222-    exp = ExperimentConfig.from_amip_config(AMIPExperimentConfig())
tests/unit/test_gwd_config_for.py:223:    assert exp.mcfarlane_k_wave == McFarlaneConfig().k_wave
tests/unit/test_gwd_config_for.py:224:    assert gwd_config_for(
tests/unit/test_gwd_config_for.py-225-        exp._replace(gravity_wave_drag="mcfarlane"),
tests/unit/test_gwd_config_for.py-226-    ).mcfarlane == McFarlaneConfig()
tests/unit/test_gwd_config_for.py-227-
tests/unit/test_gwd_config_for.py-228-
tests/unit/test_gwd_config_for.py-229-@pytest.mark.parametrize("scheme", ["mcfarlane", "hines", "mcfarlane+hines"])
tests/unit/test_gwd_config_for.py-230-def test_kernel_buildable_for_every_wired_scheme(scheme):
tests/unit/test_gwd_config_for.py-231-    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
tests/unit/test_gwd_config_for.py-232-        get_gwd_fn,
tests/unit/test_gwd_config_for.py-233-    )
tests/unit/test_gwd_config_for.py:234:    name, fn, _kcfg = get_gwd_fn(gwd_config_for(_cfg(
tests/unit/test_gwd_config_for.py-235-        gravity_wave_drag=scheme, hines_total_rms_wind=1.5,
tests/unit/test_gwd_config_for.py-236-        mcfarlane_tau_max=6.0)))
tests/unit/test_gwd_config_for.py-237-    assert fn is not None
tests/unit/test_gwd_config_for.py-238-    assert name
--
tests/unit/test_gwd_override.py:1:"""``gravity_wave_drag_override``: the coupled-path route to nested GWD
tests/unit/test_gwd_override.py-2-scheme options, mirroring ``turbulence_override``.
tests/unit/test_gwd_override.py-3-
tests/unit/test_gwd_override.py-4-Codex wave-4 P2: ``_resolve_gwd`` rebuilt ``GravityWaveDragConfig`` from the
tests/unit/test_gwd_override.py-5-scheme STRING alone, silently discarding every nested option
tests/unit/test_gwd_override.py-6-(``mcfarlane.use_e3sm_hdsp``, ``e3sm_cam.use_discrete_ke_heating``, tuned
--
tests/unit/test_gwd_override.py-34-        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
tests/unit/test_gwd_override.py-35-        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
tests/unit/test_gwd_override.py-36-                            discretization="finite_volume"),
tests/unit/test_gwd_override.py-37-        radiation="gray",
tests/unit/test_gwd_override.py-38-        gravity_wave_drag=gwd,
tests/unit/test_gwd_override.py:39:        gravity_wave_drag_override=override,
tests/unit/test_gwd_override.py-40-    )
tests/unit/test_gwd_override.py-41-
tests/unit/test_gwd_override.py-42-
tests/unit/test_gwd_override.py-43-def _pipe(gwd, override=None):
tests/unit/test_gwd_override.py-44-    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
--
tests/unit/test_gwd_override.py-73-
tests/unit/test_gwd_override.py-74-
tests/unit/test_gwd_override.py-75-def test_validate_strict_rejects_scheme_mismatch():
tests/unit/test_gwd_override.py-76-    over = GravityWaveDragConfig(scheme="lindzen")
tests/unit/test_gwd_override.py-77-    cfg = _config("mcfarlane", over)
tests/unit/test_gwd_override.py:78:    with pytest.raises(ValueError, match="gravity_wave_drag_override.scheme"):
tests/unit/test_gwd_override.py-79-        cfg.validate_strict()
tests/unit/test_gwd_override.py-80-
tests/unit/test_gwd_override.py-81-
tests/unit/test_gwd_override.py-82-def test_validate_strict_rejects_wrong_type():
tests/unit/test_gwd_override.py-83-    cfg = _config("mcfarlane", McFarlaneConfig())
--
tests/unit/_params_reachability_baseline.py-180-    'atm.rad.GrayRadiationConfig.sw_tau_0',
tests/unit/_params_reachability_baseline.py-181-    'atm.rad.GrayRadiationConfig.tau_equator',
tests/unit/_params_reachability_baseline.py-182-    'atm.rad.GrayRadiationConfig.tau_moist_coeff',
tests/unit/_params_reachability_baseline.py-183-    'atm.rad.GrayRadiationConfig.tau_pole',
tests/unit/_params_reachability_baseline.py-184-    # atm: HinesConfig (0) — Fmax + total_rms_wind became reachable when
tests/unit/_params_reachability_baseline.py:185:    # gwd_config_for started threading the hines_* ExperimentConfig scalars
tests/unit/_params_reachability_baseline.py-186-    # into the kernel leaf on every lane (2026-07-24).
tests/unit/_params_reachability_baseline.py-187-    # atm: HoltslagBovilleConfig (13)
tests/unit/_params_reachability_baseline.py-188-    'atm.turb.HoltslagBovilleConfig.Ri_crit',
tests/unit/_params_reachability_baseline.py-189-    'atm.turb.HoltslagBovilleConfig.betah',
tests/unit/_params_reachability_baseline.py-190-    'atm.turb.HoltslagBovilleConfig.betam',
--
tests/unit/_params_reachability_baseline.py-252-    'atm.conv.MassFluxConfig.cape_activation_scale',
tests/unit/_params_reachability_baseline.py-253-    'atm.conv.MassFluxConfig.cape_threshold',
tests/unit/_params_reachability_baseline.py-254-    'atm.conv.MassFluxConfig.delta_0',
tests/unit/_params_reachability_baseline.py-255-    'atm.conv.MassFluxConfig.tau_adj',
tests/unit/_params_reachability_baseline.py-256-    # atm: McFarlaneConfig (6) — directional_spread became reachable via the
tests/unit/_params_reachability_baseline.py:257:    # mcfarlane_directional_spread scalar + gwd_config_for (2026-07-24); the
tests/unit/_params_reachability_baseline.py-258-    # rest still have no ExperimentConfig scalar to route through.
tests/unit/_params_reachability_baseline.py-259-    'atm.gwd.McFarlaneConfig.G_0',
tests/unit/_params_reachability_baseline.py-260-    'atm.gwd.McFarlaneConfig.efficiency',
tests/unit/_params_reachability_baseline.py-261-    'atm.gwd.McFarlaneConfig.envelope_scale',
tests/unit/_params_reachability_baseline.py-262-    'atm.gwd.McFarlaneConfig.fcrit2',
--
tests/unit/test_gwd_landfrac_wiring.py-151-        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
tests/unit/test_gwd_landfrac_wiring.py-152-                            discretization="finite_volume"),
tests/unit/test_gwd_landfrac_wiring.py-153-        radiation="gray",
tests/unit/test_gwd_landfrac_wiring.py-154-        convection=convection,
tests/unit/test_gwd_landfrac_wiring.py-155-        gravity_wave_drag="e3sm_cam",
tests/unit/test_gwd_landfrac_wiring.py:156:        gravity_wave_drag_override=gwd_over,
tests/unit/test_gwd_landfrac_wiring.py-157-    )
tests/unit/test_gwd_landfrac_wiring.py-158-    cfg.validate_strict()
tests/unit/test_gwd_landfrac_wiring.py-159-    return grid, build_physics_pipeline(grid, sigma, cfg)
tests/unit/test_gwd_landfrac_wiring.py-160-
tests/unit/test_gwd_landfrac_wiring.py-161-
--
tests/unit/golden/yaml_to_experiment/williamson_test5.json-146-  "louis_l_mix_max": 100.0,
tests/unit/golden/yaml_to_experiment/williamson_test5.json-147-  "louis_z0": 0.0001,
tests/unit/golden/yaml_to_experiment/williamson_test5.json-148-  "lw_diff_factor": 1.66,
tests/unit/golden/yaml_to_experiment/williamson_test5.json-149-  "mcfarlane_N_ref": 0.01,
tests/unit/golden/yaml_to_experiment/williamson_test5.json-150-  "mcfarlane_directional_spread": 1.0,
tests/unit/golden/yaml_to_experiment/williamson_test5.json:151:  "mcfarlane_k_wave": 6.283185307179586e-05,
tests/unit/golden/yaml_to_experiment/williamson_test5.json-152-  "mcfarlane_tau_max": 10.0,
tests/unit/golden/yaml_to_experiment/williamson_test5.json-153-  "micro_substeps": 1,
tests/unit/golden/yaml_to_experiment/williamson_test5.json-154-  "microphysics": "none",
tests/unit/golden/yaml_to_experiment/williamson_test5.json-155-  "moisture_advection": false,
tests/unit/golden/yaml_to_experiment/williamson_test5.json-156-  "morrison_agg_coeff": 0.001,
--
tests/unit/golden/yaml_to_experiment/williamson_test2.json-146-  "louis_l_mix_max": 100.0,
tests/unit/golden/yaml_to_experiment/williamson_test2.json-147-  "louis_z0": 0.0001,
tests/unit/golden/yaml_to_experiment/williamson_test2.json-148-  "lw_diff_factor": 1.66,
tests/unit/golden/yaml_to_experiment/williamson_test2.json-149-  "mcfarlane_N_ref": 0.01,
tests/unit/golden/yaml_to_experiment/williamson_test2.json-150-  "mcfarlane_directional_spread": 1.0,
tests/unit/golden/yaml_to_experiment/williamson_test2.json:151:  "mcfarlane_k_wave": 6.283185307179586e-05,
tests/unit/golden/yaml_to_experiment/williamson_test2.json-152-  "mcfarlane_tau_max": 10.0,
tests/unit/golden/yaml_to_experiment/williamson_test2.json-153-  "micro_substeps": 1,
tests/unit/golden/yaml_to_experiment/williamson_test2.json-154-  "microphysics": "none",
tests/unit/golden/yaml_to_experiment/williamson_test2.json-155-  "moisture_advection": false,
tests/unit/golden/yaml_to_experiment/williamson_test2.json-156-  "morrison_agg_coeff": 0.001,
--
tests/unit/test_params_config_loader.py-169-        "KesslerConfig": {"microphysics": "kessler"},
tests/unit/test_params_config_loader.py-170-        "MorrisonConfig": {"microphysics": "morrison"},
tests/unit/test_params_config_loader.py-171-        "P3Config": {"microphysics": "p3"},
tests/unit/test_params_config_loader.py-172-        "SeifertBehengConfig": {"microphysics": "seifert_beheng"},
tests/unit/test_params_config_loader.py-173-        "ThompsonConfig": {"microphysics": "thompson"},
tests/unit/test_params_config_loader.py:174:        # GWD: gwd_config_for overlays the mcfarlane_*/hines_* scalars onto the
tests/unit/test_params_config_loader.py-175-        # scheme leaf, and get_gwd_fn hands the pipeline that LEAF as gwd_config.
tests/unit/test_params_config_loader.py-176-        "HinesConfig": {"gravity_wave_drag": "hines"},
tests/unit/test_params_config_loader.py-177-        "McFarlaneConfig": {"gravity_wave_drag": "mcfarlane"},
tests/unit/test_params_config_loader.py-178-    }
tests/unit/test_params_config_loader.py-179-    resolved_attr = {
--
tests/unit/test_params_config_loader.py-282-         lambda pipe, field: getattr(pipe.micro_config, field, None)),
tests/unit/test_params_config_loader.py-283-        ("SeifertBehengConfig", "", {"microphysics": "seifert_beheng"},
tests/unit/test_params_config_loader.py-284-         lambda pipe, field: getattr(pipe.micro_config, field, None)),
tests/unit/test_params_config_loader.py-285-        ("ThompsonConfig", "", {"microphysics": "thompson"},
tests/unit/test_params_config_loader.py-286-         lambda pipe, field: getattr(pipe.micro_config, field, None)),
tests/unit/test_params_config_loader.py:287:        # GWD families: _resolve_gwd -> gwd_config_for overlays the flat
tests/unit/test_params_config_loader.py-288-        # scalars onto the scheme leaf, which get_gwd_fn returns as gwd_config.
tests/unit/test_params_config_loader.py-289-        ("HinesConfig", "hines_", {"gravity_wave_drag": "hines"},
tests/unit/test_params_config_loader.py-290-         lambda pipe, field: getattr(pipe.gwd_config, field, None)),
tests/unit/test_params_config_loader.py-291-        ("McFarlaneConfig", "mcfarlane_", {"gravity_wave_drag": "mcfarlane"},
tests/unit/test_params_config_loader.py-292-         lambda pipe, field: getattr(pipe.gwd_config, field, None)),
--
packages/tools/legoesm/forcing/amip_config.py-196-    physics_parameterization_hidden_dim: int = 128
packages/tools/legoesm/forcing/amip_config.py-197-    physics_parameterization_layers: int = 3
packages/tools/legoesm/forcing/amip_config.py-198-    physics_parameterization_seed: int = 0
packages/tools/legoesm/forcing/amip_config.py-199-
packages/tools/legoesm/forcing/amip_config.py-200-
packages/tools/legoesm/forcing/amip_config.py:201:def config_to_dict(config) -> dict:
packages/tools/legoesm/forcing/amip_config.py-202-    """Generic config -> JSON-safe dict codec.
packages/tools/legoesm/forcing/amip_config.py-203-
packages/tools/legoesm/forcing/amip_config.py-204-    Thin delegator to the canonical home ``legoesm.driver.config.config_to_dict``
packages/tools/legoesm/forcing/amip_config.py-205-    (federation carve, Step 3): a single implementation, kept here as a back-compat
packages/tools/legoesm/forcing/amip_config.py-206-    entry point for callers that import it from ``forcing.amip_config``. Deferred
--
packages/coupler/legoesm/driver/run_config_yaml.py-167-# physics pipeline ACTUALLY threads from that scalar into the resolved scheme
packages/coupler/legoesm/driver/run_config_yaml.py-168-# config (``build_cloud_config`` for clouds; ``_resolve_convection`` for
packages/coupler/legoesm/driver/run_config_yaml.py-169-# sbm/bechtold).  Many ``<prefix>_<field>`` scalars EXIST on ExperimentConfig
packages/coupler/legoesm/driver/run_config_yaml.py-170-# yet are never read (``_resolve_turbulence``/``_resolve_microphysics`` build
packages/coupler/legoesm/driver/run_config_yaml.py-171-# default configs and patch only a few fields; ``_resolve_gwd`` did too until
packages/coupler/legoesm/driver/run_config_yaml.py:172:# ``gwd_config_for`` landed — see its GWD block below), so a
packages/coupler/legoesm/driver/run_config_yaml.py-173-# name-convention map would silently claim dead overrides.  The membership here
packages/coupler/legoesm/driver/run_config_yaml.py-174-# is machine-verified end-to-end by ``test_atm_scalar_map_is_pipeline_threaded``
packages/coupler/legoesm/driver/run_config_yaml.py-175-# (builds a pipeline per entry, asserts the resolved scheme config carries the
packages/coupler/legoesm/driver/run_config_yaml.py-176-# value) — extend this dict only when the pipeline threads a new scalar
packages/coupler/legoesm/driver/run_config_yaml.py-177-# (issue #691, codex audit).
--
packages/coupler/legoesm/driver/run_config_yaml.py-229-    # sits ON its bound), so the qualified name no longer exists for the
packages/coupler/legoesm/driver/run_config_yaml.py-230-    # --params loader.  The flat scalar louis_cloudtop_entrainment_efficiency
packages/coupler/legoesm/driver/run_config_yaml.py-231-    # remains settable via --config / CLI.  FOLLOW-UP: if --params reachability
packages/coupler/legoesm/driver/run_config_yaml.py-232-    # is wanted back, re-spec the param with an activation-aware transform
packages/coupler/legoesm/driver/run_config_yaml.py-233-    # instead of re-adding a dangling map key.
packages/coupler/legoesm/driver/run_config_yaml.py:234:    # gravity wave drag -> gwd_config_for (physics_pipeline), which every lane
packages/coupler/legoesm/driver/run_config_yaml.py-235-    # (FV pipeline / MPAS / spectral) now routes through.  Before it, these
packages/coupler/legoesm/driver/run_config_yaml.py-236-    # scalars existed on ExperimentConfig but NO production path read them.
packages/coupler/legoesm/driver/run_config_yaml.py-237-    # ``tau_max`` is tier 3 (a numerics clip), so the tier-1/2 reachability
packages/coupler/legoesm/driver/run_config_yaml.py-238-    # audit does not require it — it is mapped anyway because the same resolver
packages/coupler/legoesm/driver/run_config_yaml.py-239-    # threads it and the map's contract is "what the pipeline actually threads".
packages/coupler/legoesm/driver/run_config_yaml.py-240-    "atm.gwd.HinesConfig.total_rms_wind": "hines_total_rms_wind",
packages/coupler/legoesm/driver/run_config_yaml.py-241-    "atm.gwd.HinesConfig.Fmax": "hines_Fmax",
packages/coupler/legoesm/driver/run_config_yaml.py-242-    "atm.gwd.McFarlaneConfig.directional_spread": "mcfarlane_directional_spread",
packages/coupler/legoesm/driver/run_config_yaml.py-243-    "atm.gwd.McFarlaneConfig.tau_max": "mcfarlane_tau_max",
packages/coupler/legoesm/driver/run_config_yaml.py:244:    # NOTE: ``mcfarlane_k_wave`` is threaded too but has NO ``__param_spec__``
packages/coupler/legoesm/driver/run_config_yaml.py-245-    # entry (its computed 2*pi/100e3 default is not a float literal, so the
packages/coupler/legoesm/driver/run_config_yaml.py-246-    # AST-based spec gate never required one) — there is no qualified name to
packages/coupler/legoesm/driver/run_config_yaml.py-247-    # map.  Speccing it needs a bounds decision (ml/tuning.py says 1e-5..2e-4,
packages/coupler/legoesm/driver/run_config_yaml.py-248-    # aimip_params says 1e-5..5e-4).  Until then its ONLY route is the
packages/coupler/legoesm/driver/run_config_yaml.py-249-    # ``--mcfarlane-k-wave`` flag (which is also what makes the key legal in a
--
packages/coupler/legoesm/driver/model_driver.py-33-)
packages/coupler/legoesm/driver/model_driver.py-34-from legoesm.driver.config import ExperimentConfig
packages/coupler/legoesm/driver/model_driver.py-35-from legoesm.driver.physics_pipeline import (
packages/coupler/legoesm/driver/model_driver.py-36-    convection_config_for,
packages/coupler/legoesm/driver/model_driver.py-37-    build_physics_pipeline,
packages/coupler/legoesm/driver/model_driver.py:38:    gwd_config_for,
packages/coupler/legoesm/driver/model_driver.py-39-    required_microphysics_tracer_slots,
packages/coupler/legoesm/driver/model_driver.py-40-    turbulence_config_for,
packages/coupler/legoesm/driver/model_driver.py-41-    validate_microphysics_tracer_slots,
packages/coupler/legoesm/driver/model_driver.py-42-)
packages/coupler/legoesm/driver/model_driver.py-43-from legoesm.driver.diagnostics import DiagnosticCollector
--
packages/coupler/legoesm/driver/model_driver.py-6061-            convection=convection_config_for(
packages/coupler/legoesm/driver/model_driver.py-6062-                cfg,
packages/coupler/legoesm/driver/model_driver.py-6063-                grid_dx_m=float(np.sqrt(np.mean(np.asarray(self.grid.areaCell))))),
packages/coupler/legoesm/driver/model_driver.py-6064-            turbulence=turbulence_config_for(cfg),
packages/coupler/legoesm/driver/model_driver.py-6065-            microphysics=_micro_cfg,
packages/coupler/legoesm/driver/model_driver.py:6066:            gravity_wave_drag=gwd_config_for(cfg),
packages/coupler/legoesm/driver/model_driver.py-6067-        )
packages/coupler/legoesm/driver/model_driver.py-6068-        # Phase D perf: shard the per-column RRTMGP workload across all local
packages/coupler/legoesm/driver/model_driver.py-6069-        # devices (issue #273 ``column_mesh``).  rrtmgp is the dominant MPAS
packages/coupler/legoesm/driver/model_driver.py-6070-        # cost — the per-column k-distribution × RTE solve already saturates a
packages/coupler/legoesm/driver/model_driver.py-6071-        # single GPU, so sharding the ``nCells`` columns across N devices is
--
packages/coupler/legoesm/driver/model_driver.py-7568-                    ozone=OzoneProfileConfig(source=cfg.ozone_source),
packages/coupler/legoesm/driver/model_driver.py-7569-                ),
packages/coupler/legoesm/driver/model_driver.py-7570-                convection=convection_config_for(cfg),
packages/coupler/legoesm/driver/model_driver.py-7571-                turbulence=turbulence_config_for(cfg),
packages/coupler/legoesm/driver/model_driver.py-7572-                microphysics=MicrophysicsConfig(scheme=cfg.microphysics),
packages/coupler/legoesm/driver/model_driver.py:7573:                gravity_wave_drag=gwd_config_for(cfg),
packages/coupler/legoesm/driver/model_driver.py-7574-            )
packages/coupler/legoesm/driver/model_driver.py-7575-            _combined_fn = make_physics(
packages/coupler/legoesm/driver/model_driver.py-7576-                phys_cfg, model_type="spectral_pe", dt=DT,
packages/coupler/legoesm/driver/model_driver.py-7577-            )
packages/coupler/legoesm/driver/model_driver.py-7578-
--
packages/coupler/legoesm/driver/model_driver.py-9406-            _seed_ps = init_physics_state(
packages/coupler/legoesm/driver/model_driver.py-9407-                conv_ncol, _nlev,
packages/coupler/legoesm/driver/model_driver.py-9408-                PhysicsConfig(
packages/coupler/legoesm/driver/model_driver.py-9409-                    turbulence=TurbulenceConfig(scheme=cfg.turbulence),
packages/coupler/legoesm/driver/model_driver.py-9410-                    # MUST be the SAME resolved config the pipeline kernel gets
packages/coupler/legoesm/driver/model_driver.py:9411:                    # (_resolve_gwd -> gwd_config_for): init_physics_state sizes
packages/coupler/legoesm/driver/model_driver.py-9412-                    # and fills the gwd_spectrum carry from
packages/coupler/legoesm/driver/model_driver.py-9413-                    # ``prognostic_spectral.n_azimuths/.n_wavenumbers/
packages/coupler/legoesm/driver/model_driver.py:9414:                    # .launch_flux``, and a gravity_wave_drag_override may set
packages/coupler/legoesm/driver/model_driver.py-9415-                    # all three.  A bare config here seeded (ncol,4,20) at the
packages/coupler/legoesm/driver/model_driver.py-9416-                    # default launch flux while the kernel expected the
packages/coupler/legoesm/driver/model_driver.py-9417-                    # override's shape/amplitude (codex round 1, finding 2).
packages/coupler/legoesm/driver/model_driver.py:9418:                    gravity_wave_drag=gwd_config_for(cfg),
packages/coupler/legoesm/driver/model_driver.py-9419-                ),
packages/coupler/legoesm/driver/model_driver.py-9420-                dtype=_seed_dtype,
packages/coupler/legoesm/driver/model_driver.py-9421-            )
packages/coupler/legoesm/driver/model_driver.py-9422-
packages/coupler/legoesm/driver/model_driver.py-9423-            def _seed_carry(name, default):
--
packages/coupler/legoesm/driver/config.py-982-    louis_d_louis: float = 5.0                  # LouisConfig.d_louis
packages/coupler/legoesm/driver/config.py-983-    louis_z0: float = 1.0e-4                    # SurfaceLayerConfig.z0 [m]
packages/coupler/legoesm/driver/config.py-984-    louis_Ch_neutral: float = 1.5e-3            # SurfaceLayerConfig.Ch_neutral
packages/coupler/legoesm/driver/config.py-985-    louis_Cd_neutral: float = 1.5e-3            # SurfaceLayerConfig.Cd_neutral
packages/coupler/legoesm/driver/config.py-986-    # Exact 2*pi/100 km — MUST equal McFarlaneConfig.k_wave's own
packages/coupler/legoesm/driver/config.py:987:    # expression: gwd_config_for overlays this onto the leaf, so a
packages/coupler/legoesm/driver/config.py-988-    # truncated literal would silently perturb the default kernel.
packages/coupler/legoesm/driver/config.py:989:    mcfarlane_k_wave: float = 2.0 * math.pi / 100e3  # McFarlaneConfig.k_wave [1/m]
packages/coupler/legoesm/driver/config.py-990-    # INERT: no McFarlaneConfig field of this name exists — the scheme derives
packages/coupler/legoesm/driver/config.py-991-    # N from the column state (mcfarlane.py).  Kept only for the positional ABI
packages/coupler/legoesm/driver/config.py:992:    # + serialized-config compatibility; gwd_config_for deliberately does not
packages/coupler/legoesm/driver/config.py-993-    # wire it, and it was dropped from the ml/tuning.py catalog so it can no
packages/coupler/legoesm/driver/config.py-994-    # longer be advertised as a live knob (codex round 1, finding 5).
packages/coupler/legoesm/driver/config.py-995-    mcfarlane_N_ref: float = 0.01               # INERT (no leaf field)
packages/coupler/legoesm/driver/config.py-996-    mcfarlane_directional_spread: float = 1.0   # McFarlaneConfig.directional_spread
packages/coupler/legoesm/driver/config.py-997-    mcfarlane_tau_max: float = 10.0             # McFarlaneConfig.tau_max [Pa]
--
packages/coupler/legoesm/driver/config.py-1235-    # validated in ``validate_strict``).  ``None`` (default) ⇒ the driver builds
packages/coupler/legoesm/driver/config.py-1236-    # ``GravityWaveDragConfig(scheme=...)``, byte-identical to before.  This is
packages/coupler/legoesm/driver/config.py-1237-    # the ONLY coupled-path route to nested GWD scheme options
packages/coupler/legoesm/driver/config.py-1238-    # (``mcfarlane.use_e3sm_hdsp``, ``e3sm_cam.use_discrete_ke_heating``, tuned
packages/coupler/legoesm/driver/config.py-1239-    # ``fcrit2``, ...); without it they were silently discarded.
packages/coupler/legoesm/driver/config.py:1240:    gravity_wave_drag_override: Any = None
packages/coupler/legoesm/driver/config.py-1241-
packages/coupler/legoesm/driver/config.py-1242-    # Horizontal q_v smoothing on the MPAS lane [m^2/s]; 0 = off (byte-
packages/coupler/legoesm/driver/config.py-1243-    # identical; appended at the tuple END to preserve the positional ABI).
packages/coupler/legoesm/driver/config.py-1244-    # The MPAS lane historically had NO horizontal moisture smoothing (the FV
packages/coupler/legoesm/driver/config.py-1245-    # lanes smooth q_v every step in their step factories) — the confirmed
--
packages/coupler/legoesm/driver/config.py-1271-    # fluxes); refused on non-MPAS lanes (their land/ice tiles own the
packages/coupler/legoesm/driver/config.py-1272-    # surface temperature).
packages/coupler/legoesm/driver/config.py-1273-    mpas_ice_skin_prognostic: bool = False
packages/coupler/legoesm/driver/config.py-1274-    mpas_ice_thickness_m: float = 2.0      # climatological ice slab thickness [m]
packages/coupler/legoesm/driver/config.py-1275-    # Hines (1997) non-orographic GWD launch amplitude + saturation flux cap.
packages/coupler/legoesm/driver/config.py:1276:    # Reached through gwd_config_for on EVERY lane (like the mcfarlane_*
packages/coupler/legoesm/driver/config.py-1277-    # scalars above, which were silently inert on every production path until
packages/coupler/legoesm/driver/config.py-1278-    # that resolver existed).  The low-level extratropical westerlies are the
packages/coupler/legoesm/driver/config.py-1279-    # observable lever: hines deposits momentum that decelerates them.
packages/coupler/legoesm/driver/config.py-1280-    # Appended at the tuple END to preserve the positional ABI.
packages/coupler/legoesm/driver/config.py-1281-    hines_total_rms_wind: float = 2.0           # HinesConfig.total_rms_wind [m/s]
--
packages/coupler/legoesm/driver/config.py-2054-                errors.append(
packages/coupler/legoesm/driver/config.py-2055-                    f"turbulence_override.scheme={self.turbulence_override.scheme!r} "
packages/coupler/legoesm/driver/config.py-2056-                    f"must equal turbulence={self.turbulence!r} (an override refines "
packages/coupler/legoesm/driver/config.py-2057-                    f"the same scheme's sub-config, it does not switch schemes)"
packages/coupler/legoesm/driver/config.py-2058-                )
packages/coupler/legoesm/driver/config.py:2059:        if self.gravity_wave_drag_override is not None:
packages/coupler/legoesm/driver/config.py-2060-            from legoesm.atmosphere.physics.gravity_wave_drag.config import (
packages/coupler/legoesm/driver/config.py-2061-                GravityWaveDragConfig,
packages/coupler/legoesm/driver/config.py-2062-            )
packages/coupler/legoesm/driver/config.py-2063-            if not isinstance(
packages/coupler/legoesm/driver/config.py:2064:                self.gravity_wave_drag_override, GravityWaveDragConfig
packages/coupler/legoesm/driver/config.py-2065-            ):
packages/coupler/legoesm/driver/config.py-2066-                errors.append(
packages/coupler/legoesm/driver/config.py:2067:                    "gravity_wave_drag_override must be a GravityWaveDragConfig, "
packages/coupler/legoesm/driver/config.py:2068:                    f"got {type(self.gravity_wave_drag_override).__name__}"
packages/coupler/legoesm/driver/config.py-2069-                )
packages/coupler/legoesm/driver/config.py:2070:            elif (self.gravity_wave_drag_override.scheme
packages/coupler/legoesm/driver/config.py-2071-                  != self.gravity_wave_drag):
packages/coupler/legoesm/driver/config.py-2072-                errors.append(
packages/coupler/legoesm/driver/config.py:2073:                    "gravity_wave_drag_override.scheme="
packages/coupler/legoesm/driver/config.py:2074:                    f"{self.gravity_wave_drag_override.scheme!r} must equal "
packages/coupler/legoesm/driver/config.py-2075-                    f"gravity_wave_drag={self.gravity_wave_drag!r} (an override "
packages/coupler/legoesm/driver/config.py-2076-                    "refines the same scheme's sub-config, it does not switch "
packages/coupler/legoesm/driver/config.py-2077-                    "schemes)"
packages/coupler/legoesm/driver/config.py-2078-                )
packages/coupler/legoesm/driver/config.py:2079:        # GWD scalars that ``gwd_config_for`` overlays onto the kernel leaves.
packages/coupler/legoesm/driver/config.py-2080-        # Every one is a strictly-positive physical quantity (a wavenumber, a
packages/coupler/legoesm/driver/config.py-2081-        # spreading factor, a stress/flux cap, an rms launch wind), and none is
packages/coupler/legoesm/driver/config.py-2082-        # bounds-checked anywhere else on the CLI/--config route (the
packages/coupler/legoesm/driver/config.py-2083-        # ``__param_spec__`` bounds only gate the ``--params`` loader).  Silent
packages/coupler/legoesm/driver/config.py-2084-        # failure modes without this guard: ``hines_Fmax < 0`` makes
packages/coupler/legoesm/driver/config.py-2085-        # ``jnp.clip(drag, 0.0, Fmax)`` return the NEGATIVE cap at every level
packages/coupler/legoesm/driver/config.py-2086-        # (constant spurious drag, no error), and ``hines_total_rms_wind <= 0``
packages/coupler/legoesm/driver/config.py-2087-        # zeroes the amplitude growth so the scheme silently does nothing.
packages/coupler/legoesm/driver/config.py-2088-        # Positivity + finiteness only — the calibratable RANGE stays in
packages/coupler/legoesm/driver/config.py-2089-        # ``__param_spec__`` so it is not maintained twice.
packages/coupler/legoesm/driver/config.py:2090:        for _f in ("mcfarlane_k_wave", "mcfarlane_directional_spread",
packages/coupler/legoesm/driver/config.py-2091-                   "mcfarlane_tau_max", "hines_total_rms_wind", "hines_Fmax"):
packages/coupler/legoesm/driver/config.py-2092-            _v = getattr(self, _f)
packages/coupler/legoesm/driver/config.py-2093-            if not math.isfinite(_v) or _v <= 0.0:
packages/coupler/legoesm/driver/config.py-2094-                errors.append(
packages/coupler/legoesm/driver/config.py-2095-                    f"{_f} must be a positive, finite gravity-wave-drag "
--
packages/coupler/legoesm/driver/config.py-2478-            louis_z0=getattr(amip_cfg, 'louis_z0', 1.0e-4),
packages/coupler/legoesm/driver/config.py-2479-            louis_Ch_neutral=getattr(amip_cfg, 'louis_Ch_neutral', 1.5e-3),
packages/coupler/legoesm/driver/config.py-2480-            louis_Cd_neutral=getattr(amip_cfg, 'louis_Cd_neutral', 1.5e-3),
packages/coupler/legoesm/driver/config.py-2481-            # Default from the live field default (NOT a re-typed literal): the
packages/coupler/legoesm/driver/config.py-2482-            # truncated 6.283185307e-5 that used to sit here is ~3e-11 off the
packages/coupler/legoesm/driver/config.py:2483:            # exact 2*pi/100 km, which gwd_config_for now overlays onto the
packages/coupler/legoesm/driver/config.py-2484-            # kernel — a legacy-checkpoint upconvert would silently run a
packages/coupler/legoesm/driver/config.py-2485-            # different default k_wave (codex round 1, finding 6).
packages/coupler/legoesm/driver/config.py:2486:            mcfarlane_k_wave=getattr(
packages/coupler/legoesm/driver/config.py:2487:                amip_cfg, 'mcfarlane_k_wave',
packages/coupler/legoesm/driver/config.py:2488:                ExperimentConfig._field_defaults['mcfarlane_k_wave']),
packages/coupler/legoesm/driver/config.py-2489-            mcfarlane_N_ref=getattr(amip_cfg, 'mcfarlane_N_ref', 0.01),
packages/coupler/legoesm/driver/config.py-2490-            mcfarlane_directional_spread=getattr(amip_cfg, 'mcfarlane_directional_spread', 1.0),
packages/coupler/legoesm/driver/config.py-2491-            mcfarlane_tau_max=getattr(amip_cfg, 'mcfarlane_tau_max', 10.0),
packages/coupler/legoesm/driver/config.py-2492-            sbm_tau_c=amip_cfg.sbm_tau_c,
packages/coupler/legoesm/driver/config.py-2493-            sbm_RH_ref=amip_cfg.sbm_RH_ref,
--
packages/coupler/legoesm/driver/config.py-2564-            ),
packages/coupler/legoesm/driver/config.py-2565-            distributed=amip_cfg.distributed,
packages/coupler/legoesm/driver/config.py-2566-            ensemble_size=amip_cfg.ensemble_size,
packages/coupler/legoesm/driver/config.py-2567-        )
packages/coupler/legoesm/driver/config.py-2568-
packages/coupler/legoesm/driver/config.py:2569:    def to_amip_config(self):
packages/coupler/legoesm/driver/config.py-2570-        """Convert to AMIPExperimentConfig for backward-compatible serialization.
packages/coupler/legoesm/driver/config.py-2571-
packages/coupler/legoesm/driver/config.py-2572-        Used at serialization boundaries (checkpoint save, legacy config
packages/coupler/legoesm/driver/config.py-2573-        export) — not in core runtime paths.
packages/coupler/legoesm/driver/config.py-2574-        """
--
packages/coupler/legoesm/driver/config.py-2659-            louis_c_louis=getattr(self, 'louis_c_louis', 16.6),
packages/coupler/legoesm/driver/config.py-2660-            louis_d_louis=getattr(self, 'louis_d_louis', 5.0),
packages/coupler/legoesm/driver/config.py-2661-            louis_z0=getattr(self, 'louis_z0', 1.0e-4),
packages/coupler/legoesm/driver/config.py-2662-            louis_Ch_neutral=getattr(self, 'louis_Ch_neutral', 1.5e-3),
packages/coupler/legoesm/driver/config.py-2663-            louis_Cd_neutral=getattr(self, 'louis_Cd_neutral', 1.5e-3),
packages/coupler/legoesm/driver/config.py:2664:            mcfarlane_k_wave=getattr(
packages/coupler/legoesm/driver/config.py:2665:                self, 'mcfarlane_k_wave',
packages/coupler/legoesm/driver/config.py:2666:                ExperimentConfig._field_defaults['mcfarlane_k_wave']),
packages/coupler/legoesm/driver/config.py-2667-            mcfarlane_N_ref=getattr(self, 'mcfarlane_N_ref', 0.01),
packages/coupler/legoesm/driver/config.py-2668-            mcfarlane_directional_spread=getattr(self, 'mcfarlane_directional_spread', 1.0),
packages/coupler/legoesm/driver/config.py-2669-            mcfarlane_tau_max=getattr(self, 'mcfarlane_tau_max', 10.0),
packages/coupler/legoesm/driver/config.py-2670-            sbm_tau_c=self.sbm_tau_c,
packages/coupler/legoesm/driver/config.py-2671-            sbm_RH_ref=self.sbm_RH_ref,
--
packages/coupler/legoesm/driver/config.py-2718-
packages/coupler/legoesm/driver/config.py-2719-# ======================================================================
packages/coupler/legoesm/driver/config.py-2720-# Native JSON serialization for ExperimentConfig
packages/coupler/legoesm/driver/config.py-2721-# ======================================================================
packages/coupler/legoesm/driver/config.py-2722-
packages/coupler/legoesm/driver/config.py:2723:_SUB_CONFIGS = {
packages/coupler/legoesm/driver/config.py-2724-    "grid": GridConfig,
packages/coupler/legoesm/driver/config.py-2725-    "dycore": DycoreConfig,
packages/coupler/legoesm/driver/config.py-2726-    "output": OutputConfig,
packages/coupler/legoesm/driver/config.py-2727-}
packages/coupler/legoesm/driver/config.py-2728-
--
packages/coupler/legoesm/driver/config.py-2741-    # per-column JAX array C_K) — it is NOT persisted: the serialized config is
packages/coupler/legoesm/driver/config.py-2742-    # the base config, and the override is re-applied in memory after load (the
packages/coupler/legoesm/driver/config.py-2743-    # correction loop's build_driver). Drop it so it cannot be stringified +
packages/coupler/legoesm/driver/config.py-2744-    # silently corrupted on round-trip.
packages/coupler/legoesm/driver/config.py-2745-    d["turbulence_override"] = None
packages/coupler/legoesm/driver/config.py:2746:    for key in _SUB_CONFIGS:
packages/coupler/legoesm/driver/config.py-2747-        sub = d[key]
packages/coupler/legoesm/driver/config.py-2748-        if hasattr(sub, '_asdict'):
packages/coupler/legoesm/driver/config.py-2749-            sub_d = sub._asdict()
packages/coupler/legoesm/driver/config.py-2750-            if hasattr(sub_d.get('evaluation'), '_asdict'):
packages/coupler/legoesm/driver/config.py-2751-                sub_d['evaluation'] = sub_d['evaluation']._asdict()
packages/coupler/legoesm/driver/config.py-2752-            d[key] = sub_d
packages/coupler/legoesm/driver/config.py-2753-    return d
packages/coupler/legoesm/driver/config.py-2754-
packages/coupler/legoesm/driver/config.py-2755-
packages/coupler/legoesm/driver/config.py:2756:def experiment_config_from_dict(d: dict) -> ExperimentConfig:
packages/coupler/legoesm/driver/config.py-2757-    """Reconstruct ExperimentConfig from a dict (e.g. loaded from JSON).
packages/coupler/legoesm/driver/config.py-2758-
packages/coupler/legoesm/driver/config.py-2759-    Unknown fields are silently dropped for forward-compatibility
packages/coupler/legoesm/driver/config.py-2760-    (so older checkpoints with removed fields still load).
packages/coupler/legoesm/driver/config.py-2761-    """
packages/coupler/legoesm/driver/config.py-2762-    # Reconstruct sub-configs
packages/coupler/legoesm/driver/config.py-2763-    sub_values = {}
packages/coupler/legoesm/driver/config.py:2764:    for key, cls in _SUB_CONFIGS.items():
packages/coupler/legoesm/driver/config.py-2765-        if key in d and isinstance(d[key], dict):
packages/coupler/legoesm/driver/config.py-2766-            sub_d = dict(d[key])
packages/coupler/legoesm/driver/config.py-2767-            if isinstance(sub_d.get('evaluation'), dict):
packages/coupler/legoesm/driver/config.py-2768-                known_eval = set(EvaluationConfig._fields)
packages/coupler/legoesm/driver/config.py-2769-                filtered_eval = {
--
packages/coupler/legoesm/driver/config.py-2784-    filtered.update(sub_values)
packages/coupler/legoesm/driver/config.py-2785-
packages/coupler/legoesm/driver/config.py-2786-    return ExperimentConfig(**filtered)
packages/coupler/legoesm/driver/config.py-2787-
packages/coupler/legoesm/driver/config.py-2788-
packages/coupler/legoesm/driver/config.py:2789:def config_to_dict(config) -> dict:
packages/coupler/legoesm/driver/config.py-2790-    """Generic config -> JSON-safe dict codec (canonical home).
packages/coupler/legoesm/driver/config.py-2791-
packages/coupler/legoesm/driver/config.py-2792-    Accepts an ``ExperimentConfig`` or the legacy flat ``AMIPExperimentConfig``;
packages/coupler/legoesm/driver/config.py-2793-    nested sub-config NamedTuples (grid/dycore/output) are inlined as dicts.
packages/coupler/legoesm/driver/config.py-2794-    This is the codec the experiment-level checkpoint/restart I/O in
--
packages/coupler/legoesm/driver/physics_pipeline.py-3380-
packages/coupler/legoesm/driver/physics_pipeline.py-3381-# ---------------------------------------------------------------------------
packages/coupler/legoesm/driver/physics_pipeline.py-3382-# Gravity wave drag resolver
packages/coupler/legoesm/driver/physics_pipeline.py-3383-# ---------------------------------------------------------------------------
packages/coupler/legoesm/driver/physics_pipeline.py-3384-
packages/coupler/legoesm/driver/physics_pipeline.py:3385:def gwd_config_for(config):
packages/coupler/legoesm/driver/physics_pipeline.py-3386-    """The ``GravityWaveDragConfig`` (scheme + tuned per-scheme leaves) to
packages/coupler/legoesm/driver/physics_pipeline.py-3387-    build a gravity-wave-drag kernel from.
packages/coupler/legoesm/driver/physics_pipeline.py-3388-
packages/coupler/legoesm/driver/physics_pipeline.py-3389-    Third member of the resolver family (:func:`turbulence_config_for`,
packages/coupler/legoesm/driver/physics_pipeline.py-3390-    :func:`convection_config_for`): every lane — FV pipeline, MPAS,
packages/coupler/legoesm/driver/physics_pipeline.py-3391-    spectral — previously built ``GravityWaveDragConfig(scheme=...)`` from
packages/coupler/legoesm/driver/physics_pipeline.py-3392-    the scheme STRING alone, so the tuned ExperimentConfig scalars
packages/coupler/legoesm/driver/physics_pipeline.py:3393:    (``mcfarlane_k_wave`` / ``mcfarlane_directional_spread`` /
packages/coupler/legoesm/driver/physics_pipeline.py-3394-    ``mcfarlane_tau_max``, and the Hines launch amplitude) silently never
packages/coupler/legoesm/driver/physics_pipeline.py-3395-    reached the kernel on ANY production AMIP path; only the AIMIP training
packages/coupler/legoesm/driver/physics_pipeline.py-3396-    path consumed them.  Same gap class as the 2026-07-23 convection and
packages/coupler/legoesm/driver/physics_pipeline.py-3397-    hard-sat overrides.
packages/coupler/legoesm/driver/physics_pipeline.py-3398-
packages/coupler/legoesm/driver/physics_pipeline.py:3399:    ``config.gravity_wave_drag_override`` (a full config whose ``scheme``
packages/coupler/legoesm/driver/physics_pipeline.py-3400-    must equal ``config.gravity_wave_drag`` — enforced by
packages/coupler/legoesm/driver/physics_pipeline.py-3401-    ``validate_strict``) still wins verbatim: an explicitly injected config
packages/coupler/legoesm/driver/physics_pipeline.py-3402-    is never second-guessed by the scalar overlay.
packages/coupler/legoesm/driver/physics_pipeline.py-3403-
packages/coupler/legoesm/driver/physics_pipeline.py-3404-    COMPOSITE schemes ("mcfarlane+hines") carry BOTH leaves, so the overlay
--
packages/coupler/legoesm/driver/physics_pipeline.py-3408-    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
packages/coupler/legoesm/driver/physics_pipeline.py-3409-        GravityWaveDragConfig,
packages/coupler/legoesm/driver/physics_pipeline.py-3410-    )
packages/coupler/legoesm/driver/physics_pipeline.py-3411-
packages/coupler/legoesm/driver/physics_pipeline.py-3412-    scheme = getattr(config, "gravity_wave_drag", "none")
packages/coupler/legoesm/driver/physics_pipeline.py:3413:    override = getattr(config, "gravity_wave_drag_override", None)
packages/coupler/legoesm/driver/physics_pipeline.py-3414-    if override is not None:
packages/coupler/legoesm/driver/physics_pipeline.py-3415-        return override
packages/coupler/legoesm/driver/physics_pipeline.py-3416-    gc = GravityWaveDragConfig(scheme=scheme)
packages/coupler/legoesm/driver/physics_pipeline.py-3417-    if scheme == "none":
packages/coupler/legoesm/driver/physics_pipeline.py-3418-        return gc
packages/coupler/legoesm/driver/physics_pipeline.py-3419-    # McFarlane (orographic) tunables. ``mcfarlane_N_ref`` is deliberately
packages/coupler/legoesm/driver/physics_pipeline.py-3420-    # NOT wired: no McFarlaneConfig field of that name exists (dangling
packages/coupler/legoesm/driver/physics_pipeline.py-3421-    # ExperimentConfig scalar, tracked separately).
packages/coupler/legoesm/driver/physics_pipeline.py-3422-    mc = gc.mcfarlane._replace(
packages/coupler/legoesm/driver/physics_pipeline.py:3423:        k_wave=float(getattr(config, "mcfarlane_k_wave", gc.mcfarlane.k_wave)),
packages/coupler/legoesm/driver/physics_pipeline.py-3424-        directional_spread=float(getattr(
packages/coupler/legoesm/driver/physics_pipeline.py-3425-            config, "mcfarlane_directional_spread",
packages/coupler/legoesm/driver/physics_pipeline.py-3426-            gc.mcfarlane.directional_spread)),
packages/coupler/legoesm/driver/physics_pipeline.py-3427-        tau_max=float(getattr(config, "mcfarlane_tau_max",
packages/coupler/legoesm/driver/physics_pipeline.py-3428-                              gc.mcfarlane.tau_max)),
--
packages/coupler/legoesm/driver/physics_pipeline.py-3439-def _resolve_gwd(config):
packages/coupler/legoesm/driver/physics_pipeline.py-3440-    """Resolve gravity wave drag kernel and config from ExperimentConfig.
packages/coupler/legoesm/driver/physics_pipeline.py-3441-
packages/coupler/legoesm/driver/physics_pipeline.py-3442-    Returns (kernel_fn, kernel_config) or (None, None) if disabled.
packages/coupler/legoesm/driver/physics_pipeline.py-3443-
packages/coupler/legoesm/driver/physics_pipeline.py:3444:    ``config.gravity_wave_drag_override`` (a full ``GravityWaveDragConfig``
packages/coupler/legoesm/driver/physics_pipeline.py-3445-    whose ``scheme`` must equal ``config.gravity_wave_drag`` — enforced by
packages/coupler/legoesm/driver/physics_pipeline.py-3446-    ``ExperimentConfig.validate_strict``) is honoured verbatim, mirroring
packages/coupler/legoesm/driver/physics_pipeline.py-3447-    ``turbulence_override``: without it the coupled path rebuilt the config
packages/coupler/legoesm/driver/physics_pipeline.py-3448-    from the scheme STRING alone, silently discarding every nested scheme
packages/coupler/legoesm/driver/physics_pipeline.py-3449-    option (``mcfarlane.use_e3sm_hdsp``, ``e3sm_cam.use_discrete_ke_heating``,
--
packages/coupler/legoesm/driver/physics_pipeline.py-3455-
packages/coupler/legoesm/driver/physics_pipeline.py-3456-    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
packages/coupler/legoesm/driver/physics_pipeline.py-3457-        get_gwd_fn,
packages/coupler/legoesm/driver/physics_pipeline.py-3458-    )
packages/coupler/legoesm/driver/physics_pipeline.py-3459-
packages/coupler/legoesm/driver/physics_pipeline.py:3460:    _name, gwd_fn, gwd_config = get_gwd_fn(gwd_config_for(config))
packages/coupler/legoesm/driver/physics_pipeline.py-3461-    return gwd_fn, gwd_config
packages/coupler/legoesm/driver/physics_pipeline.py-3462-
packages/coupler/legoesm/driver/physics_pipeline.py-3463-
packages/coupler/legoesm/driver/physics_pipeline.py-3464-def _resolve_physics_parameterization(config, nlev: int):
packages/coupler/legoesm/driver/physics_pipeline.py-3465-    """Resolve the optional joint ML physics parameterization."""
--
packages/ml/legoesm/tuning.py-486-        description="Louis surface-layer neutral momentum drag coefficient",
packages/ml/legoesm/tuning.py-487-        category="surface",
packages/ml/legoesm/tuning.py-488-        sensitivity="medium",
packages/ml/legoesm/tuning.py-489-        notes="Constant-scheme Cd: sets surface wind stress and friction velocity.",
packages/ml/legoesm/tuning.py-490-    ),
packages/ml/legoesm/tuning.py:491:    "mcfarlane_k_wave": TuningParameter(
packages/ml/legoesm/tuning.py:492:        name="mcfarlane_k_wave",
packages/ml/legoesm/tuning.py-493-        default=6.283185307e-5,
packages/ml/legoesm/tuning.py-494-        min_val=1.0e-5,
packages/ml/legoesm/tuning.py-495-        max_val=2.0e-4,
packages/ml/legoesm/tuning.py-496-        units="1/m",
packages/ml/legoesm/tuning.py-497-        description="McFarlane orographic GWD horizontal wavenumber",
--
packages/ml/legoesm/training/aimip_params.py-108-    ParamConstraint("mcfarlane_G_0", 0.1, 1.0, "sigmoid"),
packages/ml/legoesm/training/aimip_params.py-109-    ParamConstraint("mcfarlane_efficiency", 0.1, 1.0, "sigmoid"),
packages/ml/legoesm/training/aimip_params.py-110-    ParamConstraint("mcfarlane_min_wind", 0.5, 5.0, "sigmoid"),
packages/ml/legoesm/training/aimip_params.py-111-    ParamConstraint("mcfarlane_envelope_scale", 0.5, 2.0, "sigmoid"),
packages/ml/legoesm/training/aimip_params.py-112-    # Extended: orographic wavenumber + spread + tau cap.
packages/ml/legoesm/training/aimip_params.py:113:    ParamConstraint("mcfarlane_k_wave", 1.0e-5, 5.0e-4, "sigmoid"),
packages/ml/legoesm/training/aimip_params.py-114-    ParamConstraint("mcfarlane_directional_spread", 0.5, 2.0, "sigmoid"),
packages/ml/legoesm/training/aimip_params.py-115-    ParamConstraint("mcfarlane_tau_max", 1.0, 50.0, "sigmoid"),
packages/ml/legoesm/training/aimip_params.py-116-]
packages/ml/legoesm/training/aimip_params.py-117-
packages/ml/legoesm/training/aimip_params.py-118-# Xu-Randall cloud fraction.  ``T_freeze`` is NOT trainable per
--
packages/ml/legoesm/training/aimip_params.py-407-            h_topo=d["mcfarlane_h_topo"],
packages/ml/legoesm/training/aimip_params.py-408-            G_0=d["mcfarlane_G_0"],
packages/ml/legoesm/training/aimip_params.py-409-            efficiency=d["mcfarlane_efficiency"],
packages/ml/legoesm/training/aimip_params.py-410-            min_wind=d["mcfarlane_min_wind"],
packages/ml/legoesm/training/aimip_params.py-411-            envelope_scale=d["mcfarlane_envelope_scale"],
packages/ml/legoesm/training/aimip_params.py:412:            k_wave=d["mcfarlane_k_wave"],
packages/ml/legoesm/training/aimip_params.py-413-            directional_spread=d["mcfarlane_directional_spread"],
packages/ml/legoesm/training/aimip_params.py-414-            tau_max=d["mcfarlane_tau_max"],
packages/ml/legoesm/training/aimip_params.py-415-        )
packages/ml/legoesm/training/aimip_params.py-416-
packages/ml/legoesm/training/aimip_params.py-417-    def to_cloud_config(self) -> CloudConfig:
--
packages/ml/legoesm/training/aimip_params.py-553-        "mcfarlane_h_topo": float(mc.h_topo),
packages/ml/legoesm/training/aimip_params.py-554-        "mcfarlane_G_0": float(mc.G_0),
packages/ml/legoesm/training/aimip_params.py-555-        "mcfarlane_efficiency": float(mc.efficiency),
packages/ml/legoesm/training/aimip_params.py-556-        "mcfarlane_min_wind": float(mc.min_wind),
packages/ml/legoesm/training/aimip_params.py-557-        "mcfarlane_envelope_scale": float(mc.envelope_scale),
packages/ml/legoesm/training/aimip_params.py:558:        "mcfarlane_k_wave": float(mc.k_wave),
packages/ml/legoesm/training/aimip_params.py-559-        "mcfarlane_directional_spread": float(mc.directional_spread),
packages/ml/legoesm/training/aimip_params.py-560-        "mcfarlane_tau_max": float(mc.tau_max),
packages/ml/legoesm/training/aimip_params.py-561-        # Cloud (Xu-Randall)
packages/ml/legoesm/training/aimip_params.py-562-        "cloud_rh_crit": float(cl.rh_crit),
packages/ml/legoesm/training/aimip_params.py-563-        "cloud_alpha_xr": float(cl.alpha_xr),

exec
/usr/bin/bash -lc "rg -n -C 3 \"GravityWaveDragConfig\\(\\s*(scheme\\s*=\\s*)?(cfg|config|.*gravity_wave_drag)|gravity_wave_drag\\s*=\\s*GravityWaveDragConfig|_resolve_gwd\\(|gwd_config_for\\(\" packages scripts --glob '*.py' && sed -n '2550,2805p' packages/coupler/legoesm/driver/config.py && sed -n '1,220p' tests/unit/test_gwd_config_for.py && sed -n '1,120p' packages/coupler/legoesm/driver/checkpoint.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 365ms:
packages/coupler/legoesm/driver/model_driver.py-6063-                grid_dx_m=float(np.sqrt(np.mean(np.asarray(self.grid.areaCell))))),
packages/coupler/legoesm/driver/model_driver.py-6064-            turbulence=turbulence_config_for(cfg),
packages/coupler/legoesm/driver/model_driver.py-6065-            microphysics=_micro_cfg,
packages/coupler/legoesm/driver/model_driver.py:6066:            gravity_wave_drag=gwd_config_for(cfg),
packages/coupler/legoesm/driver/model_driver.py-6067-        )
packages/coupler/legoesm/driver/model_driver.py-6068-        # Phase D perf: shard the per-column RRTMGP workload across all local
packages/coupler/legoesm/driver/model_driver.py-6069-        # devices (issue #273 ``column_mesh``).  rrtmgp is the dominant MPAS
--
packages/coupler/legoesm/driver/model_driver.py-7570-                convection=convection_config_for(cfg),
packages/coupler/legoesm/driver/model_driver.py-7571-                turbulence=turbulence_config_for(cfg),
packages/coupler/legoesm/driver/model_driver.py-7572-                microphysics=MicrophysicsConfig(scheme=cfg.microphysics),
packages/coupler/legoesm/driver/model_driver.py:7573:                gravity_wave_drag=gwd_config_for(cfg),
packages/coupler/legoesm/driver/model_driver.py-7574-            )
packages/coupler/legoesm/driver/model_driver.py-7575-            _combined_fn = make_physics(
packages/coupler/legoesm/driver/model_driver.py-7576-                phys_cfg, model_type="spectral_pe", dt=DT,
--
packages/coupler/legoesm/driver/model_driver.py-9415-                    # all three.  A bare config here seeded (ncol,4,20) at the
packages/coupler/legoesm/driver/model_driver.py-9416-                    # default launch flux while the kernel expected the
packages/coupler/legoesm/driver/model_driver.py-9417-                    # override's shape/amplitude (codex round 1, finding 2).
packages/coupler/legoesm/driver/model_driver.py:9418:                    gravity_wave_drag=gwd_config_for(cfg),
packages/coupler/legoesm/driver/model_driver.py-9419-                ),
packages/coupler/legoesm/driver/model_driver.py-9420-                dtype=_seed_dtype,
packages/coupler/legoesm/driver/model_driver.py-9421-            )
--
packages/coupler/legoesm/driver/physics_pipeline.py-3382-# Gravity wave drag resolver
packages/coupler/legoesm/driver/physics_pipeline.py-3383-# ---------------------------------------------------------------------------
packages/coupler/legoesm/driver/physics_pipeline.py-3384-
packages/coupler/legoesm/driver/physics_pipeline.py:3385:def gwd_config_for(config):
packages/coupler/legoesm/driver/physics_pipeline.py-3386-    """The ``GravityWaveDragConfig`` (scheme + tuned per-scheme leaves) to
packages/coupler/legoesm/driver/physics_pipeline.py-3387-    build a gravity-wave-drag kernel from.
packages/coupler/legoesm/driver/physics_pipeline.py-3388-
--
packages/coupler/legoesm/driver/physics_pipeline.py-3436-    return gc._replace(mcfarlane=mc, hines=hn)
packages/coupler/legoesm/driver/physics_pipeline.py-3437-
packages/coupler/legoesm/driver/physics_pipeline.py-3438-
packages/coupler/legoesm/driver/physics_pipeline.py:3439:def _resolve_gwd(config):
packages/coupler/legoesm/driver/physics_pipeline.py-3440-    """Resolve gravity wave drag kernel and config from ExperimentConfig.
packages/coupler/legoesm/driver/physics_pipeline.py-3441-
packages/coupler/legoesm/driver/physics_pipeline.py-3442-    Returns (kernel_fn, kernel_config) or (None, None) if disabled.
--
packages/coupler/legoesm/driver/physics_pipeline.py-3457-        get_gwd_fn,
packages/coupler/legoesm/driver/physics_pipeline.py-3458-    )
packages/coupler/legoesm/driver/physics_pipeline.py-3459-
packages/coupler/legoesm/driver/physics_pipeline.py:3460:    _name, gwd_fn, gwd_config = get_gwd_fn(gwd_config_for(config))
packages/coupler/legoesm/driver/physics_pipeline.py-3461-    return gwd_fn, gwd_config
packages/coupler/legoesm/driver/physics_pipeline.py-3462-
packages/coupler/legoesm/driver/physics_pipeline.py-3463-
--
packages/coupler/legoesm/driver/physics_pipeline.py-3585-    turb_fn, turb_config = _resolve_turbulence(config)
packages/coupler/legoesm/driver/physics_pipeline.py-3586-
packages/coupler/legoesm/driver/physics_pipeline.py-3587-    # Resolve gravity wave drag
packages/coupler/legoesm/driver/physics_pipeline.py:3588:    gwd_fn, gwd_config = _resolve_gwd(config)
packages/coupler/legoesm/driver/physics_pipeline.py-3589-
packages/coupler/legoesm/driver/physics_pipeline.py-3590-    # Resolve optional joint ML physics parameterization
packages/coupler/legoesm/driver/physics_pipeline.py-3591-    physics_parameterization = _resolve_physics_parameterization(
--
scripts/matrix/scm/gabls1.py-121-        convection=ConvectionConfig(scheme="none"),
scripts/matrix/scm/gabls1.py-122-        turbulence=_turbulence_config(turbulence),
scripts/matrix/scm/gabls1.py-123-        microphysics=MicrophysicsConfig(scheme="none"),
scripts/matrix/scm/gabls1.py:124:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
scripts/matrix/scm/gabls1.py-125-    )
scripts/matrix/scm/gabls1.py-126-    forcing = SCMForcing(
scripts/matrix/scm/gabls1.py-127-        f_c=_F_C,
--
scripts/matrix/scm/rce.py-63-        convection=ConvectionConfig(scheme="mass_flux"),
scripts/matrix/scm/rce.py-64-        turbulence=TurbulenceConfig(scheme="louis"),
scripts/matrix/scm/rce.py-65-        microphysics=MicrophysicsConfig(scheme="kessler"),
scripts/matrix/scm/rce.py:66:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
scripts/matrix/scm/rce.py-67-    )
scripts/matrix/scm/rce.py-68-
scripts/matrix/scm/rce.py-69-    nsteps = int((args.days * 86400.0) / args.dt)
--
scripts/matrix/scm/ekman.py-74-            ),
scripts/matrix/scm/ekman.py-75-        ),
scripts/matrix/scm/ekman.py-76-        microphysics=MicrophysicsConfig(scheme="none"),
scripts/matrix/scm/ekman.py:77:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
scripts/matrix/scm/ekman.py-78-    )
scripts/matrix/scm/ekman.py-79-    forcing = SCMForcing(
scripts/matrix/scm/ekman.py-80-        f_c=_F_C,
--
scripts/matrix/scm/wangara.py-94-            ),
scripts/matrix/scm/wangara.py-95-        ),
scripts/matrix/scm/wangara.py-96-        microphysics=MicrophysicsConfig(scheme="none"),
scripts/matrix/scm/wangara.py:97:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
scripts/matrix/scm/wangara.py-98-    )
scripts/matrix/scm/wangara.py-99-    u_geo = _u_geo_profile(nlev, z_full)
scripts/matrix/scm/wangara.py-100-    forcing = SCMForcing(
--
scripts/matrix/run_atmosphere_test_matrix.py-1441-        convection=ConvectionConfig(scheme="none"),
scripts/matrix/run_atmosphere_test_matrix.py-1442-        turbulence=TurbulenceConfig(scheme="none"),
scripts/matrix/run_atmosphere_test_matrix.py-1443-        microphysics=MicrophysicsConfig(scheme="none"),
scripts/matrix/run_atmosphere_test_matrix.py:1444:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
scripts/matrix/run_atmosphere_test_matrix.py-1445-    )
scripts/matrix/run_atmosphere_test_matrix.py-1446-    rrtmgp_fn = make_physics(phys_cfg, model_type=model_type, dt=dt)
scripts/matrix/run_atmosphere_test_matrix.py-1447-
--
scripts/validate/rce_scm_scheme_sweep.py-313-        convection=ConvectionConfig(scheme=conv),
scripts/validate/rce_scm_scheme_sweep.py-314-        turbulence=TurbulenceConfig(scheme=turb),
scripts/validate/rce_scm_scheme_sweep.py-315-        microphysics=MicrophysicsConfig(scheme="kessler"),
scripts/validate/rce_scm_scheme_sweep.py:316:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
scripts/validate/rce_scm_scheme_sweep.py-317-    )
scripts/validate/rce_scm_scheme_sweep.py-318-    nsteps = int((args.days * 86400.0) / args.dt)
scripts/validate/rce_scm_scheme_sweep.py-319-    try:
--
scripts/run/run_aimip_amip_finetune.py-249-                convection=ConvectionConfig(scheme=str(cfg["aimip_convection"])),
scripts/run/run_aimip_amip_finetune.py-250-                turbulence=TurbulenceConfig(scheme=str(cfg["aimip_turbulence"])),
scripts/run/run_aimip_amip_finetune.py-251-                microphysics=MicrophysicsConfig(scheme=str(cfg["aimip_microphysics"])),
scripts/run/run_aimip_amip_finetune.py:252:                gravity_wave_drag=GravityWaveDragConfig(scheme=str(cfg["aimip_gwd"])),
scripts/run/run_aimip_amip_finetune.py-253-            ),
scripts/run/run_aimip_amip_finetune.py-254-        )
scripts/run/run_aimip_amip_finetune.py-255-
--
scripts/run/run_scm_rce_campaign.py-669-        convection=cfg.convection,
scripts/run/run_scm_rce_campaign.py-670-        turbulence=TurbulenceConfig(scheme="none"),
scripts/run/run_scm_rce_campaign.py-671-        microphysics=MicrophysicsConfig(scheme="none"),
scripts/run/run_scm_rce_campaign.py:672:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
scripts/run/run_scm_rce_campaign.py-673-    )
scripts/run/run_scm_rce_campaign.py-674-
scripts/run/run_scm_rce_campaign.py-675-
--
scripts/run/run_scm_rce_campaign.py-853-            convection=cfg.convection,
scripts/run/run_scm_rce_campaign.py-854-            turbulence=TurbulenceConfig(scheme="none"),
scripts/run/run_scm_rce_campaign.py-855-            microphysics=MicrophysicsConfig(scheme="none"),
scripts/run/run_scm_rce_campaign.py:856:            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
scripts/run/run_scm_rce_campaign.py-857-        ),
scripts/run/run_scm_rce_campaign.py-858-        model_type="hydrostatic",
scripts/run/run_scm_rce_campaign.py-859-        dt=dt,
--
scripts/run/run_scm_rce_campaign.py-945-        convection=ConvectionConfig(scheme="none"),
scripts/run/run_scm_rce_campaign.py-946-        turbulence=TurbulenceConfig(scheme="none"),
scripts/run/run_scm_rce_campaign.py-947-        microphysics=MicrophysicsConfig(scheme="none"),
scripts/run/run_scm_rce_campaign.py:948:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
scripts/run/run_scm_rce_campaign.py-949-    )
scripts/run/run_scm_rce_campaign.py-950-
scripts/run/run_scm_rce_campaign.py-951-
--
packages/atmosphere/legoesm/atmosphere/physics/combined.py-29-...     convection=ConvectionConfig(scheme="sbm"),
packages/atmosphere/legoesm/atmosphere/physics/combined.py-30-...     turbulence=TurbulenceConfig(scheme="louis"),
packages/atmosphere/legoesm/atmosphere/physics/combined.py-31-...     microphysics=MicrophysicsConfig(scheme="kessler"),
packages/atmosphere/legoesm/atmosphere/physics/combined.py:32:...     gravity_wave_drag=GravityWaveDragConfig(scheme="lindzen"),
packages/atmosphere/legoesm/atmosphere/physics/combined.py-33-... )
packages/atmosphere/legoesm/atmosphere/physics/combined.py-34->>> physics_fn = make_physics(config, model_type="hydrostatic", dt=300.0)
packages/atmosphere/legoesm/atmosphere/physics/combined.py-35->>> state = model.step_with_physics(state, dt, physics_fn)
--
scripts/run/run_scm_rce_ice_tuning.py-175-        convection=ConvectionConfig(scheme=convection),
scripts/run/run_scm_rce_ice_tuning.py-176-        turbulence=TurbulenceConfig(scheme="louis"),
scripts/run/run_scm_rce_ice_tuning.py-177-        microphysics=MicrophysicsConfig(scheme="morrison", morrison=morrison),
scripts/run/run_scm_rce_ice_tuning.py:178:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
scripts/run/run_scm_rce_ice_tuning.py-179-    )
scripts/run/run_scm_rce_ice_tuning.py-180-
scripts/run/run_scm_rce_ice_tuning.py-181-
--
scripts/run/run_held_suarez_rrtmgp_4grids.py-64-        convection=ConvectionConfig(scheme="none"),
scripts/run/run_held_suarez_rrtmgp_4grids.py-65-        turbulence=TurbulenceConfig(scheme="none"),
scripts/run/run_held_suarez_rrtmgp_4grids.py-66-        microphysics=MicrophysicsConfig(scheme="none"),
scripts/run/run_held_suarez_rrtmgp_4grids.py:67:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
scripts/run/run_held_suarez_rrtmgp_4grids.py-68-    )
scripts/run/run_held_suarez_rrtmgp_4grids.py-69-    rrtmgp_fn = make_physics(phys_cfg, model_type=model_type, dt=dt)
scripts/run/run_held_suarez_rrtmgp_4grids.py-70-
--
scripts/run/run_aimip_amip_inference.py-279-                convection=ConvectionConfig(scheme=str(cfg["aimip_convection"])),
scripts/run/run_aimip_amip_inference.py-280-                turbulence=TurbulenceConfig(scheme=str(cfg["aimip_turbulence"])),
scripts/run/run_aimip_amip_inference.py-281-                microphysics=MicrophysicsConfig(scheme=str(cfg["aimip_microphysics"])),
scripts/run/run_aimip_amip_inference.py:282:                gravity_wave_drag=GravityWaveDragConfig(scheme=str(cfg["aimip_gwd"])),
scripts/run/run_aimip_amip_inference.py-283-            ),
scripts/run/run_aimip_amip_inference.py-284-        )
scripts/run/run_aimip_amip_inference.py-285-
            physics_parameterization_checkpoint=getattr(
                amip_cfg, 'physics_parameterization_checkpoint', '',
            ),
            physics_parameterization_stats=getattr(
                amip_cfg, 'physics_parameterization_stats', '',
            ),
            physics_parameterization_hidden_dim=getattr(
                amip_cfg, 'physics_parameterization_hidden_dim', 128,
            ),
            physics_parameterization_layers=getattr(
                amip_cfg, 'physics_parameterization_layers', 3,
            ),
            physics_parameterization_seed=getattr(
                amip_cfg, 'physics_parameterization_seed', 0,
            ),
            distributed=amip_cfg.distributed,
            ensemble_size=amip_cfg.ensemble_size,
        )

    def to_amip_config(self):
        """Convert to AMIPExperimentConfig for backward-compatible serialization.

        Used at serialization boundaries (checkpoint save, legacy config
        export) — not in core runtime paths.
        """
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        _amip_kwargs = dict(
            resolution=self.grid.resolution,
            nlev=self.grid.nlev,
            dt=self.dycore.dt,
            vertical_coord=self.grid.vertical_coord,
            p_top_Pa=self.grid.p_top_Pa,
            stretching=self.grid.stretching,
            start_day=self.start_day,
            days=self.days,
            diag_days=self.output.diag_days,
            checkpoint_days=self.output.checkpoint_days,
            dataset=self.dataset,
            forcing_path=self.forcing_path,
            sic_path=self.sic_path,
            sst_var=self.sst_var,
            sic_var=self.sic_var,
            time_var=self.time_var,
            lat_var=self.lat_var,
            lon_var=self.lon_var,
            sst_offset=self.sst_offset,
            sic_scale=self.sic_scale,
            radiation=self.radiation,
            rad_update_steps=self.rad_update_steps,
            diurnal_cycle=self.diurnal_cycle,
            co2_ppmv=self.co2_ppmv,
            ch4_ppbv=self.ch4_ppbv,
            n2o_ppbv=self.n2o_ppbv,
            S_0=self.S_0,
            ozone_source=self.ozone_source,
            ozone_forcing=self.ozone_forcing,
            ozone_file=self.ozone_file,
            ghg_forcing=self.ghg_forcing,
            ghg_file=self.ghg_file,
            solar_source=self.solar_source,
            solar_file=self.solar_file,
            solar_tsi_var=self.solar_tsi_var,
            solar_spectral_var=self.solar_spectral_var,
            solar_spectral_band_order=self.solar_spectral_band_order,
            aerosol_forcing=self.aerosol_forcing,
            aerosol_file=self.aerosol_file,
            aerosol_reference_aod=self.aerosol_reference_aod,
            volcanic_aerosol_file=self.volcanic_aerosol_file,
            volcanic_aerosol_scale=self.volcanic_aerosol_scale,
            cloud_scheme=self.cloud_scheme,
            microphysics=self.microphysics,
            convection=self.convection,
            turbulence=self.turbulence,
            gravity_wave_drag=self.gravity_wave_drag,
            fix_moisture=self.fix_moisture,
            energy_consistent_moisture_clip=self.energy_consistent_moisture_clip,
            topography=self.topography,
            topo_smoothing=self.topo_smoothing,
            topo_edge_blend=self.topo_edge_blend,
            T_init=self.T_init,
            rh_init=self.rh_init,
            dynamic_albedo=self.dynamic_albedo,
            carbon_cycle=self.carbon_cycle,
            experiment=self.experiment,
            start_year=self.start_year,
            C_H=self.C_H,
            C_E=self.C_E,
            T_ice=self.T_ice,
            albedo_ice=self.albedo_ice,
            albedo_ocean=self.albedo_ocean,
            sfc_emissivity=self.sfc_emissivity,
            emissivity_ice=self.emissivity_ice,
            tau_equator=self.tau_equator,
            tau_pole=self.tau_pole,
            tau_moist_coeff=getattr(self, 'tau_moist_coeff', 0.0115),
            linear_frac=getattr(self, 'linear_frac', 0.2),
            lw_diff_factor=getattr(self, 'lw_diff_factor', 1.66),
            sw_tau_0=getattr(self, 'sw_tau_0', 0.22),
            sw_exponent=getattr(self, 'sw_exponent', 2.0),
            sundqvist_auto_rate=getattr(self, 'sundqvist_auto_rate', 1e-3),
            sundqvist_evap_coeff=getattr(self, 'sundqvist_evap_coeff', 5e-4),
            cloud_rh_crit=getattr(self, 'cloud_rh_crit', 0.7),
            cloud_r_eff_liq=getattr(self, 'cloud_r_eff_liq', 10.0e-6),
            sundqvist_sigmoid_sharpness=getattr(self, 'sundqvist_sigmoid_sharpness', 20.0),
            sbm_T_min_convect=getattr(self, 'sbm_T_min_convect', 200.0),
            louis_l_mix_max=getattr(self, 'louis_l_mix_max', 100.0),
            louis_Ck=getattr(self, 'louis_Ck', 0.4),
            louis_Ri_crit=getattr(self, 'louis_Ri_crit', 0.25),
            louis_b_louis=getattr(self, 'louis_b_louis', 5.0),
            louis_c_louis=getattr(self, 'louis_c_louis', 16.6),
            louis_d_louis=getattr(self, 'louis_d_louis', 5.0),
            louis_z0=getattr(self, 'louis_z0', 1.0e-4),
            louis_Ch_neutral=getattr(self, 'louis_Ch_neutral', 1.5e-3),
            louis_Cd_neutral=getattr(self, 'louis_Cd_neutral', 1.5e-3),
            mcfarlane_k_wave=getattr(
                self, 'mcfarlane_k_wave',
                ExperimentConfig._field_defaults['mcfarlane_k_wave']),
            mcfarlane_N_ref=getattr(self, 'mcfarlane_N_ref', 0.01),
            mcfarlane_directional_spread=getattr(self, 'mcfarlane_directional_spread', 1.0),
            mcfarlane_tau_max=getattr(self, 'mcfarlane_tau_max', 10.0),
            sbm_tau_c=self.sbm_tau_c,
            sbm_RH_ref=self.sbm_RH_ref,
            sbm_cape_threshold=self.sbm_cape_threshold,
            bechtold_cape_threshold=self.bechtold_cape_threshold,
            convective_precip_efficiency=self.convective_precip_efficiency,
            convective_precip_split=self.convective_precip_split,
            autoconv_q_c_crit=self.autoconv_q_c_crit,
            autoconv_pe_max=self.autoconv_pe_max,
            bechtold_conv_top_pa=self.bechtold_conv_top_pa,
            bechtold_downdraft_evap=self.bechtold_downdraft_evap,
            bechtold_downdraft_alpha=self.bechtold_downdraft_alpha,
            bechtold_downdraft_rh_min=self.bechtold_downdraft_rh_min,
            bechtold_downdraft_transport=self.bechtold_downdraft_transport,
            bechtold_downdraft_entrain_rate=self.bechtold_downdraft_entrain_rate,
            bechtold_downdraft_detrain_scale_m=self.bechtold_downdraft_detrain_scale_m,
            bechtold_use_ifs_cape_closure=self.bechtold_use_ifs_cape_closure,
            bechtold_use_ifs_subcloud_evap=self.bechtold_use_ifs_subcloud_evap,
            bechtold_use_ifs_inplume_precip=self.bechtold_use_ifs_inplume_precip,
            bechtold_dx_m=self.bechtold_dx_m,
            bechtold_use_ifs_downdraft=self.bechtold_use_ifs_downdraft,
            bechtold_use_ifs_shallow_closure=self.bechtold_use_ifs_shallow_closure,
            bechtold_use_ifs_capdcycl=self.bechtold_use_ifs_capdcycl,
            bechtold_use_ifs_land_rhebc=self.bechtold_use_ifs_land_rhebc,
            bechtold_use_ifs_snow_melt=self.bechtold_use_ifs_snow_melt,
            sigma_b=self.sigma_b,
            k_BL_max_per_day=self.k_BL_max_per_day,
            k_free_per_day=self.k_free_per_day,
            held_suarez_forcing=self.held_suarez_forcing,
            physics_parameterization=self.physics_parameterization,
            physics_parameterization_checkpoint=self.physics_parameterization_checkpoint,
            physics_parameterization_stats=self.physics_parameterization_stats,
            physics_parameterization_hidden_dim=self.physics_parameterization_hidden_dim,
            physics_parameterization_layers=self.physics_parameterization_layers,
            physics_parameterization_seed=self.physics_parameterization_seed,
            monthly_means=self.output.monthly_means,
            cmip_output=self.output.cmip_output,
            clear_sky_diag=self.output.clear_sky_diag,
            distributed=self.distributed,
            ensemble_size=self.ensemble_size,
            output_dir=self.output.output_dir,
        )
        # Filter to the legacy AMIP schema's fields — ExperimentConfig has
        # accreted many newer knobs the flat AMIPExperimentConfig never
        # mirrored; drop those instead of raising (drift-proof round-trip).
        return AMIPExperimentConfig(**{
            k: v for k, v in _amip_kwargs.items()
            if k in AMIPExperimentConfig._fields})


# ======================================================================
# Native JSON serialization for ExperimentConfig
# ======================================================================

_SUB_CONFIGS = {
    "grid": GridConfig,
    "dycore": DycoreConfig,
    "output": OutputConfig,
}


def experiment_config_to_dict(config: ExperimentConfig) -> dict:
    """Serialize ExperimentConfig to a JSON-safe dict.

    Sub-configs (grid, dycore, output) are inlined as nested dicts. This
    is the canonical serialization format. ``output.evaluation`` is
    itself a nested ``EvaluationConfig`` NamedTuple one level deeper —
    without this it round-trips through ``json.dumps`` as a bare
    positional list (NamedTuple is a tuple), losing field names.
    """
    d = config._asdict()
    # ``turbulence_override`` is a RUNTIME-ONLY injection (it may carry a
    # per-column JAX array C_K) — it is NOT persisted: the serialized config is
    # the base config, and the override is re-applied in memory after load (the
    # correction loop's build_driver). Drop it so it cannot be stringified +
    # silently corrupted on round-trip.
    d["turbulence_override"] = None
    for key in _SUB_CONFIGS:
        sub = d[key]
        if hasattr(sub, '_asdict'):
            sub_d = sub._asdict()
            if hasattr(sub_d.get('evaluation'), '_asdict'):
                sub_d['evaluation'] = sub_d['evaluation']._asdict()
            d[key] = sub_d
    return d


def experiment_config_from_dict(d: dict) -> ExperimentConfig:
    """Reconstruct ExperimentConfig from a dict (e.g. loaded from JSON).

    Unknown fields are silently dropped for forward-compatibility
    (so older checkpoints with removed fields still load).
    """
    # Reconstruct sub-configs
    sub_values = {}
    for key, cls in _SUB_CONFIGS.items():
        if key in d and isinstance(d[key], dict):
            sub_d = dict(d[key])
            if isinstance(sub_d.get('evaluation'), dict):
                known_eval = set(EvaluationConfig._fields)
                filtered_eval = {
                    k: v for k, v in sub_d['evaluation'].items() if k in known_eval
                }
                # ``suites`` is a tuple field; JSON round-trips it as a list, so
                # coerce back so the reconstructed config == the original.
                if isinstance(filtered_eval.get('suites'), list):
                    filtered_eval['suites'] = tuple(filtered_eval['suites'])
                sub_d['evaluation'] = EvaluationConfig(**filtered_eval)
            known_sub = set(cls._fields)
            filtered = {k: v for k, v in sub_d.items() if k in known_sub}
            sub_values[key] = cls(**filtered)

    # Filter top-level fields
    known = set(ExperimentConfig._fields)
    filtered = {k: v for k, v in d.items() if k in known}
    filtered.update(sub_values)

    return ExperimentConfig(**filtered)


def config_to_dict(config) -> dict:
    """Generic config -> JSON-safe dict codec (canonical home).

    Accepts an ``ExperimentConfig`` or the legacy flat ``AMIPExperimentConfig``;
    nested sub-config NamedTuples (grid/dycore/output) are inlined as dicts.
    This is the codec the experiment-level checkpoint/restart I/O in
    ``driver`` uses, so that layer no longer reaches up into
    ``forcing.amip_config`` for serialization (federation carve, Step 3).
    """
    d = config._asdict()
    # Runtime-only injection — never serialized (see experiment_config_to_dict).
    if "turbulence_override" in d:
        d["turbulence_override"] = None
    for key, val in d.items():
        if hasattr(val, "_asdict"):
            d[key] = val._asdict()
    return d
"""``gwd_config_for`` — the third resolver (turbulence / convection / GWD).

Every lane (FV pipeline, MPAS, spectral) previously built
``GravityWaveDragConfig(scheme=...)`` from the scheme STRING alone, so the
tuned ExperimentConfig scalars never reached the kernel on ANY production
AMIP path (only the AIMIP training path consumed them).  This pins the
overlay, the override precedence, and the composite-scheme behaviour.
"""

import pytest

from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    GravityWaveDragConfig,
)
from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.physics_pipeline import gwd_config_for


def _cfg(**kw):
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=1, nlev=8,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(dt=600.0, discretization="mpas"),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        **kw,
    )


@pytest.mark.parametrize("scheme", [
    "none", "rayleigh", "lindzen", "mcfarlane", "hines",
    "prognostic_spectral", "e3sm_cam", "ml_emulator", "mcfarlane+hines",
])
def test_defaults_match_bare_config(scheme):
    """Untouched scalars reproduce the bare config EXACTLY — whole tuple, every
    scheme.  Byte-identical defaults are the precondition for wiring the overlay
    into lanes that previously built the bare config."""
    assert (gwd_config_for(_cfg(gravity_wave_drag=scheme))
            == GravityWaveDragConfig(scheme=scheme))


def test_none_scheme_short_circuits():
    got = gwd_config_for(_cfg(gravity_wave_drag="none"))
    assert got.scheme == "none"
    assert got == GravityWaveDragConfig(scheme="none")


def test_mcfarlane_scalars_reach_the_leaf():
    got = gwd_config_for(_cfg(
        gravity_wave_drag="mcfarlane",
        mcfarlane_tau_max=4.0,
        mcfarlane_k_wave=1.0e-4,
        mcfarlane_directional_spread=0.5,
    ))
    assert got.mcfarlane.tau_max == 4.0
    assert got.mcfarlane.k_wave == 1.0e-4
    assert got.mcfarlane.directional_spread == 0.5


def test_hines_scalars_reach_the_leaf():
    got = gwd_config_for(_cfg(
        gravity_wave_drag="hines",
        hines_total_rms_wind=1.2,
        hines_Fmax=0.05,
    ))
    assert got.hines.total_rms_wind == 1.2
    assert got.hines.Fmax == 0.05


def test_composite_scheme_carries_both_tuned_leaves():
    """'mcfarlane+hines' runs BOTH kernels — both overlays must apply."""
    got = gwd_config_for(_cfg(
        gravity_wave_drag="mcfarlane+hines",
        mcfarlane_tau_max=6.0,
        hines_total_rms_wind=1.5,
    ))
    assert got.scheme == "mcfarlane+hines"
    assert got.mcfarlane.tau_max == 6.0
    assert got.hines.total_rms_wind == 1.5


def test_explicit_override_wins_verbatim():
    """An injected full config is never second-guessed by the overlay."""
    inj = GravityWaveDragConfig(scheme="mcfarlane")
    inj = inj._replace(mcfarlane=inj.mcfarlane._replace(tau_max=99.0))
    got = gwd_config_for(_cfg(
        gravity_wave_drag="mcfarlane",
        gravity_wave_drag_override=inj,
        mcfarlane_tau_max=4.0,   # must NOT win over the override
    ))
    assert got is inj
    assert got.mcfarlane.tau_max == 99.0


def test_other_leaves_untouched():
    """Overlaying mcfarlane/hines leaves the other scheme leaves at defaults."""
    got = gwd_config_for(_cfg(gravity_wave_drag="mcfarlane",
                              mcfarlane_tau_max=4.0))
    bare = GravityWaveDragConfig(scheme="mcfarlane")
    assert got.rayleigh == bare.rayleigh
    assert got.lindzen == bare.lindzen
    assert got.e3sm_cam == bare.e3sm_cam
    assert got.prognostic_spectral == bare.prognostic_spectral


def test_resolve_gwd_uses_the_resolver():
    """The FV pipeline path honours the same tuned leaf (no divergence)."""
    from legoesm.driver.physics_pipeline import _resolve_gwd

    fn, kernel_cfg = _resolve_gwd(_cfg(gravity_wave_drag="mcfarlane",
                                       mcfarlane_tau_max=4.0))
    assert fn is not None
    assert kernel_cfg.tau_max == 4.0


def test_cli_round_trip():
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args,
    )
    parser = build_arg_parser()
    d = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert d.hines_total_rms_wind == 2.0
    assert d.hines_Fmax == 0.1
    assert d.mcfarlane_tau_max == 10.0

    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--hines-total-rms-wind", "1.2", "--hines-fmax", "0.05",
        "--mcfarlane-tau-max", "4.0", "--mcfarlane-k-wave", "1.0e-4",
    ]), parser))
    assert cfg.hines_total_rms_wind == 1.2
    assert cfg.hines_Fmax == 0.05
    assert cfg.mcfarlane_tau_max == 4.0
    assert cfg.mcfarlane_k_wave == 1.0e-4
    assert gwd_config_for(cfg).hines.total_rms_wind == 1.2


def test_every_overlaid_scalar_has_a_run_amip_route():
    """Each scalar gwd_config_for threads must be settable by an operator.
    ``--config`` YAML keys must be argparse dests (load_yaml_config rejects
    anything else), so a flag is the gate for BOTH routes; the --params route
    additionally needs a _ATM_SCALAR_PARAM_MAP entry, which mcfarlane_k_wave
    lacks by design (no __param_spec__ bounds yet)."""
    from legoesm.driver.run_config_yaml import build_atm_scalar_param_map
    from scripts.run.run_amip import build_arg_parser

    dests = {a.dest for a in build_arg_parser()._actions}
    overlaid = ("mcfarlane_k_wave", "mcfarlane_directional_spread",
                "mcfarlane_tau_max", "hines_total_rms_wind", "hines_Fmax")
    # mcfarlane_directional_spread is --params-only (mapped, no flag); every
    # other overlaid scalar must have a flag.
    amap = set(build_atm_scalar_param_map().values())
    for f in overlaid:
        assert f in dests or f in amap, (
            f"{f} is overlaid onto the GWD kernel by gwd_config_for but an "
            "operator cannot set it: no run_amip flag AND no scalar-map entry."
        )


def test_physics_state_seed_honours_a_nested_override():
    """The stateful-carry seed must resolve through the SAME resolver as the
    kernel: init_physics_state sizes gwd_spectrum from
    prognostic_spectral.n_azimuths/.n_wavenumbers/.launch_flux, and an override
    may set all three (codex round 1, finding 2 — the seed at
    model_driver.py:9418 used to build a bare config)."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        PrognosticSpectralConfig,
    )
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig

    inj = GravityWaveDragConfig(
        scheme="prognostic_spectral",
        prognostic_spectral=PrognosticSpectralConfig(
            n_azimuths=8, n_wavenumbers=6, launch_flux=2.5e-3),
    )
    cfg = _cfg(gravity_wave_drag="prognostic_spectral",
               gravity_wave_drag_override=inj)
    ps = init_physics_state(
        3, 8,
        PhysicsConfig(turbulence=TurbulenceConfig(scheme="none"),
                      gravity_wave_drag=gwd_config_for(cfg)),
    )
    assert ps.gwd_spectrum.shape == (3, 8, 6)
    assert float(ps.gwd_spectrum[0, 0, 0]) == 2.5e-3


@pytest.mark.parametrize("field,bad", [
    ("hines_total_rms_wind", -1.0),
    ("hines_total_rms_wind", 0.0),
    ("hines_Fmax", -0.1),
    ("mcfarlane_tau_max", float("nan")),
    ("mcfarlane_k_wave", float("inf")),
    ("mcfarlane_directional_spread", -1.0),
])
def test_validate_strict_rejects_nonpositive_or_nonfinite(field, bad):
    """A negative Fmax makes clip(drag, 0, Fmax) return the NEGATIVE cap at
    every level (constant spurious drag, no error); a non-positive launch rms
    wind silently disables the scheme.  Neither may reach the kernel."""
    with pytest.raises(ValueError, match=field):
        _cfg(gravity_wave_drag="hines", **{field: bad}).validate_strict()


def test_legacy_amip_upconvert_keeps_the_exact_k_wave_default():
    """A legacy checkpoint upconverted through from_amip_config must land on
    the SAME k_wave the kernel default uses (the truncated 6.283185307e-5
    fallback was ~3e-11 off once the overlay went live)."""
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        McFarlaneConfig,
    )
    from legoesm.forcing.amip_config import AMIPExperimentConfig

    # The legacy flat schema has NO mcfarlane_* fields, so the converter's
"""Zarr-based checkpoint IO for legoESM.

Provides save/load of atmospheric state to/from Zarr stores with Zstd
compression. Backward-compatible with existing .npz checkpoints
via auto-detection in ``load_checkpoint_auto``.

Compatible with zarr v2 (>= 2.18) and v3 (>= 3.0).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np


def wallclock_exhausted(elapsed_s: float, max_s: float, buffer_s: float) -> bool:
    """True if the run should checkpoint and exit to fit the wallclock budget.

    ``max_s <= 0`` disables the check.  Otherwise fire once the elapsed time is
    within ``buffer_s`` of the budget, leaving time to write the checkpoint
    before the scheduler kills the job (so a dependency chain can resume).
    Shared by the OMIP/LMIP run drivers (previously copy-pasted).
    """
    return max_s > 0.0 and elapsed_s >= (max_s - buffer_s)


# Lazy imports to avoid circular dependency:
#   io.checkpoint → driver.config → driver.__init__ → model_driver → io.checkpoint
_experiment_config_from_dict = None


def _get_experiment_config_from_dict():
    global _experiment_config_from_dict
    if _experiment_config_from_dict is None:
        from legoesm.driver.config import experiment_config_from_dict
        _experiment_config_from_dict = experiment_config_from_dict
    return _experiment_config_from_dict


def _config_from_dict_auto(d: dict):
    """Detect config format and deserialize.

    ExperimentConfig JSON has a ``"grid"`` key with nested sub-configs;
    legacy AMIPExperimentConfig JSON has flat fields (``"resolution"``
    at the top level).

    Returns an ``ExperimentConfig`` in both cases (upconverting legacy
    AMIP format via ``ExperimentConfig.from_amip_config``).
    """
    if "grid" in d and isinstance(d["grid"], dict):
        # New ExperimentConfig format
        ecfd = _get_experiment_config_from_dict()
        return ecfd(d)
    else:
        # Legacy AMIPExperimentConfig format — load then upconvert.
        # Deferred driver.config import: a top-level one would reintroduce the
        # io.checkpoint -> driver.config -> driver.__init__ -> model_driver cycle.
        from legoesm.driver.config import config_from_dict
        amip_cfg = config_from_dict(d)
        from legoesm.driver.config import ExperimentConfig
        return ExperimentConfig.from_amip_config(amip_cfg)


def _auto_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Choose chunk shape: per-face for 4D (6,n,n,nlev), whole-array otherwise."""
    if len(shape) == 4 and shape[0] == 6:
        # One chunk per face: (1, n, n, nlev)
        return (1,) + shape[1:]
    return shape


def save_checkpoint_zarr(
    path,
    state,
    q_v,
    step: int,
    day: float,
    config,
    *,
    q_c=None,
    q_r=None,
    diag_accumulators: dict | None = None,
    compressor=None,
) -> None:
    """Save checkpoint to Zarr store.

    Layout::

        checkpoint.zarr/
            T/          -- temperature array
            u/          -- zonal wind
            v/          -- meridional wind
            p_s/        -- surface pressure
            phis/       -- surface geopotential
            q_v/        -- specific humidity
            q_c/        -- cloud water (optional)
            q_r/        -- rain water (optional)
            diag_*/     -- diagnostic accumulators
            .zattrs     -- metadata (step, day, config JSON)

    Parameters
    ----------
    path : Path-like
        Output Zarr store path (directory).
    state : HydrostaticState
    q_v : array-like
    step : int
    day : float
    config : ExperimentConfig or AMIPExperimentConfig
    q_c, q_r : array-like, optional
    diag_accumulators : dict, optional
    compressor : optional
        Zarr v3 codec. Default: Zstd(level=3).
    """
    import zarr

    path = Path(path)


codex
The diff contains the claimed fixes, including the untracked resolver test module. I’m now checking route semantics and checkpoint behavior against the implementation rather than relying on the reported grep/test output.
exec
/usr/bin/bash -lc "sed -n '1970,2120p' packages/coupler/legoesm/driver/config.py && sed -n '1,260p' packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/config.py && sed -n '1,135p' packages/coupler/legoesm/driver/run_config_yaml.py && sed -n '135,310p' tests/unit/test_gwd_config_for.py && rg -n -C 3 \"to_amip_config\\(\" --glob '*.py' . -g '"'!tests/**'"' -g '"'!**/tests/**'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 475ms:
        # microphysics="none" early-returns (None, None) in _resolve_microphysics
        # / the MPAS post-step drain before the flag is ever read, so the gate
        # would do nothing.  Reject it at config time (codex F3) rather than let
        # it silently no-op.  Same scheme set as the fail-loud runtime raise in
        # microphysics/config.apply_microphysics_experiment_flags.
        _warm_rain_micro = (
            "kessler", "seifert_beheng", "morrison", "thompson", "p3")
        if (self.hard_saturation_adjustment
                and self.microphysics not in _warm_rain_micro):
            errors.append(
                "hard_saturation_adjustment requires a warm-rain microphysics "
                f"scheme {_warm_rain_micro}; got microphysics="
                f"{self.microphysics!r} (the guard would be silently inert)."
            )
        # gs_max is a physical conductance [mol/m2/s]: must be finite and
        # strictly positive (nan/<=0 would zero or NaN the whole land latent
        # flux).  Upper sanity bound 2.0 is well above the StomataConfig
        # __param_spec__ tunable range (0.099, 0.9).
        if not (0.0 < self.land_gs_max <= 2.0):
            errors.append(
                f"land_gs_max (max stomatal conductance [mol/m2/s]) must be "
                f"finite and in (0, 2]; got {self.land_gs_max!r}."
            )
        # Marine-Sc cloud-top entrainment efficiency: finite, in [0, 1] (matches
        # LouisConfig.__param_spec__; the not(lo<=x<=hi) form also rejects NaN/Inf).
        if not (0.0 <= self.louis_cloudtop_entrainment_efficiency <= 1.0):
            errors.append(
                f"louis_cloudtop_entrainment_efficiency (marine-Sc cloud-top "
                f"entrainment A) must be finite and in [0, 1]; got "
                f"{self.louis_cloudtop_entrainment_efficiency!r}."
            )
        # Soil-moisture init fraction of saturation: finite, in (0, 1].
        if not (0.0 < self.land_soil_moisture_init_frac <= 1.0):
            errors.append(
                f"land_soil_moisture_init_frac (theta_init/theta_sat) must be "
                f"finite and in (0, 1]; got {self.land_soil_moisture_init_frac!r}."
            )
        # Land surface-scheme membership (mirror the model_driver dispatch so a
        # typo fails here, not at run time).
        _valid_land_surface = ("simple_seb", "two_leaf", "clm_ml")
        if self.land_surface_scheme not in _valid_land_surface:
            errors.append(
                f"land_surface_scheme must be one of {_valid_land_surface}, "
                f"got {self.land_surface_scheme!r}"
            )
        # Stomatal soil-water limitation needs the bucket to supply beta_soil.
        if (self.land_stomatal_beta and not self.land_soil_bucket
                and not self.use_multilayer_land):
            errors.append(
                "land_stomatal_beta=True requires land_soil_bucket=True "
                "(the bucket supplies the soil availability beta_soil that the "
                "Jarvis stomatal model down-regulates) — UNLESS use_multilayer_land "
                "is set, in which case the Richards multilayer soil supplies the "
                "root-zone moisture availability instead (the stomata are threaded "
                "into MultiLayerLandConfig.stomata, PR #715)."
            )
        # Optional cloud-tuning override bounds (mirror CloudConfig.__param_spec__
        # so an out-of-range knob fails early, not deep in the cloud diagnosis).
        for _f, _lo, _hi in (
            ("cloud_rh_crit", 0.5, 0.99),
            ("cloud_q_c_diagnostic", 5.0e-5, 1.0e-3),
            ("cloud_conv_cloud_max", 0.1, 1.0),
            ("cloud_conv_cloud_condensate", 1.0e-5, 1.0e-3),
            ("cloud_inhomogeneity_factor", 0.3, 1.0),
            ("cloud_fsd", 0.0, 1.0),
            ("cloud_p_xr", 0.05, 1.0),
            ("cloud_alpha_xr", 10.0, 1000.0),
            ("cloud_adiabatic_lwc_rate", 5.0e-7, 3.0e-6),
        ):
            _v = getattr(self, _f)
            if _v is not None and not (_lo <= _v <= _hi):
                errors.append(
                    f"{_f}={_v!r} out of range [{_lo}, {_hi}]"
                )
        if self.turbulence_override is not None:
            from legoesm.atmosphere.physics.turbulence.config import (
                TurbulenceConfig,
            )
            if not isinstance(self.turbulence_override, TurbulenceConfig):
                errors.append(
                    "turbulence_override must be a TurbulenceConfig, got "
                    f"{type(self.turbulence_override).__name__}"
                )
            elif self.turbulence_override.scheme != self.turbulence:
                errors.append(
                    f"turbulence_override.scheme={self.turbulence_override.scheme!r} "
                    f"must equal turbulence={self.turbulence!r} (an override refines "
                    f"the same scheme's sub-config, it does not switch schemes)"
                )
        if self.gravity_wave_drag_override is not None:
            from legoesm.atmosphere.physics.gravity_wave_drag.config import (
                GravityWaveDragConfig,
            )
            if not isinstance(
                self.gravity_wave_drag_override, GravityWaveDragConfig
            ):
                errors.append(
                    "gravity_wave_drag_override must be a GravityWaveDragConfig, "
                    f"got {type(self.gravity_wave_drag_override).__name__}"
                )
            elif (self.gravity_wave_drag_override.scheme
                  != self.gravity_wave_drag):
                errors.append(
                    "gravity_wave_drag_override.scheme="
                    f"{self.gravity_wave_drag_override.scheme!r} must equal "
                    f"gravity_wave_drag={self.gravity_wave_drag!r} (an override "
                    "refines the same scheme's sub-config, it does not switch "
                    "schemes)"
                )
        # GWD scalars that ``gwd_config_for`` overlays onto the kernel leaves.
        # Every one is a strictly-positive physical quantity (a wavenumber, a
        # spreading factor, a stress/flux cap, an rms launch wind), and none is
        # bounds-checked anywhere else on the CLI/--config route (the
        # ``__param_spec__`` bounds only gate the ``--params`` loader).  Silent
        # failure modes without this guard: ``hines_Fmax < 0`` makes
        # ``jnp.clip(drag, 0.0, Fmax)`` return the NEGATIVE cap at every level
        # (constant spurious drag, no error), and ``hines_total_rms_wind <= 0``
        # zeroes the amplitude growth so the scheme silently does nothing.
        # Positivity + finiteness only — the calibratable RANGE stays in
        # ``__param_spec__`` so it is not maintained twice.
        for _f in ("mcfarlane_k_wave", "mcfarlane_directional_spread",
                   "mcfarlane_tau_max", "hines_total_rms_wind", "hines_Fmax"):
            _v = getattr(self, _f)
            if not math.isfinite(_v) or _v <= 0.0:
                errors.append(
                    f"{_f} must be a positive, finite gravity-wave-drag "
                    f"parameter, got {_v!r}"
                )
        _valid_gwd = VALID_GWD
        # A ``+``-joined string composes multiple GWD sources whose tendencies
        # are summed — orographic (mcfarlane/lindzen) and non-orographic
        # (hines/rayleigh/prognostic_spectral) parameterize distinct wave
        # populations and are run together in CMIP-class GCMs
        # (e.g. ``hines+mcfarlane`` or ``mcfarlane+prognostic_spectral``,
        # issue #834).  ``prognostic_spectral`` is the one STATEFUL composable
        # source — its wave-action spectrum threads through the physics carry,
        # so at most one stateful source may appear.  ``e3sm_cam`` /
        # ``ml_emulator`` need extra per-column source fields / a network
        # module the composite path does not carry and are NOT composable.
        _composable_stateless = ("rayleigh", "lindzen", "mcfarlane", "hines")
        _composable_stateful = ("prognostic_spectral",)
        _composable = _composable_stateless + _composable_stateful
        _gwd_parts = self.gravity_wave_drag.split("+")
        if len(_gwd_parts) > 1:
            bad = [p for p in _gwd_parts if p not in _composable]
            if bad:
                errors.append(
                    f"composite gravity_wave_drag parts must each be one of "
                    f"{_composable}, got invalid {bad} in "
                    f"{self.gravity_wave_drag!r}"
                )
"""Configuration for gravity wave drag schemes.

Provides configuration NamedTuples for:
1. Rayleigh: simple Rayleigh friction drag
2. Lindzen: smoothed Lindzen (1981) orographic GWD
3. McFarlane: smoothed McFarlane (1987) orographic GWD
4. Hines: Hines (1997) Doppler-spread parameterization
5. PrognosticSpectral: prognostic spectral GWD
6. MLEmulator: ML-based GWD emulator (Equinox MLP)
7. GravityWaveDragConfig: top-level selector

References
----------
- Lindzen, R. S. (1981). Turbulence and stress owing to gravity wave and
  tidal breakdown. J. Geophys. Res., 86, 9707-9714.
- McFarlane, N. A. (1987). The effect of orographically excited gravity
  wave drag on the general circulation of the lower stratosphere and
  troposphere. J. Atmos. Sci., 44, 1775-1800.
- Hines, C. O. (1997). Doppler-spread parameterization of gravity-wave
  momentum deposition in the middle atmosphere. 1. Basic formulation.
  J. Atmos. Solar-Terr. Phys., 59, 371-386.
"""

from __future__ import annotations

import math
from typing import NamedTuple


__param_spec__ = {
    "E3SMBeresConfig": {
        "scheme_key": "atm.gwd.E3SMBeresConfig",
        "excluded": {
            "mfcc_uh_slope": "stand-in-table-only surrogate slope (default 0 = uh-independent); the real source spectrum is the offline mfcc table, not a trained scalar",
            "storm_speed_min": "ground-relative storm-speed floor used only to truncate the integer Doppler cell-speed CS; a discretisation threshold, not a continuum closure",
        },
        "params": {
            # --- convective source amplitude (Beres et al. 2004) ---
            "cf": {"units": "1", "bounds": (5.0, 60.0), "tunable_tier": 1, "transform": "sigmoid", "category": "source_spectrum", "reference": "Beres et al. (2004); E3SM gw_convect_hcf", "shape": None},
            "al": {"units": "m", "bounds": (1.0e4, 1.0e6), "tunable_tier": 1, "transform": "sigmoid", "category": "source_spectrum", "reference": "Beres et al. (2004); E3SM AL", "shape": None},
            "hdepth_scaling_factor": {"units": "1", "bounds": (0.25, 4.0), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "Beres et al. (2004); E3SM hdepth_scaling_factor", "shape": None},
            # --- convective source triggering / search window ---
            "hdepth_min_km": {"units": "km", "bounds": (0.5, 10.0), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "Beres et al. (2004); E3SM hdepth_min", "shape": None},
            "z_heat_max": {"units": "m", "bounds": (5.0e3, 4.0e4), "tunable_tier": 3, "transform": "sigmoid", "category": "source_spectrum", "reference": "Beres et al. (2004) scheme default", "shape": None},
            "source_wind_p": {"units": "Pa", "bounds": (4.0e4, 9.0e4), "tunable_tier": 3, "transform": "sigmoid", "category": "launch_level", "reference": "E3SM gw_beres_src k700 source-wind level", "shape": None},
            # --- analytic stand-in spectrum surrogate (not Beres-faithful) ---
            "mfcc_peak": {"units": "Pa", "bounds": (1.0e-4, 1.0e-1), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "stand-in mfcc surrogate (E3SM mfcc table replacement)", "shape": None},
            "mfcc_c0": {"units": "m/s", "bounds": (5.0, 90.0), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "stand-in mfcc surrogate phase-speed width", "shape": None},
            "mfcc_hdepth_growth": {"units": "1/km", "bounds": (0.0, 0.5), "tunable_tier": 3, "transform": "sigmoid", "category": "source_spectrum", "reference": "stand-in mfcc surrogate depth-growth slope", "shape": None},
        },
    },
    "E3SMCAMConfig": {
        "scheme_key": "atm.gwd.E3SMCAMConfig",
        "excluded": {
            "alpha_newtonian": "uniform Newtonian-cooling coefficient (default 0 = off); when used it is replaced by the E3SM height profile, not a single trained scalar",
            # dc is the phase-speed bin width but the frontal source quadrature
            # turns it into a Python integer sub-interval COUNT
            # (e3sm_cam._front_fav: n_sub = int(math.floor(dc/dca + 0.5)) - 1,
            # then jnp.arange(1, n_sub+1)). A traced trainable would hit int() on
            # a tracer / fix a static, non-differentiable quadrature shape. Not
            # trainable until the quadrature uses a fixed max grid with masking.
            "dc": "phase-speed bin width -> Python int quadrature count (jnp.arange shape) in _front_fav; not traceable",
        },
        "params": {
            # --- efficiency (primary amplitude knob) ---
            "effgw": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "efficiency", "reference": "CAM gw_drag effgw_oro/effgw_cm", "shape": None},
            # --- saturation / wave breaking ---
            "fcrit2": {"units": "1", "bounds": (0.5, 2.0), "tunable_tier": 1, "transform": "sigmoid", "category": "saturation", "reference": "Lindzen (1981)/McFarlane (1987); CAM fcrit2", "shape": None},
            "umcfac": {"units": "1", "bounds": (0.1, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "wave_breaking", "reference": "Scinocca (2003); E3SM gw_common umcfac", "shape": None},
            # --- saturation floors ---
            "n2min": {"units": "1/s^2", "bounds": (1.0e-9, 1.0e-7), "tunable_tier": 3, "transform": "sigmoid", "category": "numerics", "reference": "E3SM gw_prof N^2 floor", "shape": None},
            "taumin": {"units": "Pa", "bounds": (1.0e-12, 1.0e-8), "tunable_tier": 3, "transform": "sigmoid", "category": "numerics", "reference": "E3SM gw_common minimum-stress floor", "shape": None},
            "ubmc2mn": {"units": "m^2/s^2", "bounds": (1.0e-3, 1.0e-1), "tunable_tier": 3, "transform": "sigmoid", "category": "numerics", "reference": "E3SM gw_common (u-c)^2 floor", "shape": None},
            "tndmax_per_day": {"units": "m/s/day", "bounds": (100.0, 1000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common tendency ceiling", "shape": None},
            # --- GW-induced eddy diffusion ---
            "dback": {"units": "m^2/s", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "saturation", "reference": "E3SM gw_common background diffusivity dback", "shape": None},
            "prndl": {"units": "1", "bounds": (0.05, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "saturation", "reference": "E3SM gw_diffusion inverse Prandtl number", "shape": None},
            "egwd_max": {"units": "m^2/s", "bounds": (10.0, 500.0), "tunable_tier": 3, "transform": "sigmoid", "category": "saturation", "reference": "E3SM gw_diffusion eddy-diffusivity cap", "shape": None},
            "ediff_kbot_p": {"units": "Pa", "bounds": (3.0e4, 9.0e4), "tunable_tier": 3, "transform": "sigmoid", "category": "launch_level", "reference": "E3SM gw_diffusion kbotbg eddy-diffusion bottom level", "shape": None},
        },
    },
    "E3SMFrontalConfig": {
        "scheme_key": "atm.gwd.E3SMFrontalConfig",
        "excluded": {
            "front_spectrum_dc_resolution": "sub-bin c-grid quadrature spacing dca for the Gaussian integration; a numerical resolution of the spectrum, not a closure",
        },
        "params": {
            # --- frontal source amplitude / triggering ---
            "taubgnd": {"units": "Pa", "bounds": (1.0e-4, 1.0e-2), "tunable_tier": 1, "transform": "sigmoid", "category": "momentum_flux", "reference": "Charron & Manzini (2002); CAM taubgnd", "shape": None},
            "frontgfc": {"units": "K^2/(m^2 s)", "bounds": (1.0e-16, 1.0e-13), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "E3SM namelist_defaults_eam.xml frontgfc 1.25e-15 (7.5e-16 at 4x5; 2e-14 at ne120np4 on E3SM master); Charron & Manzini (2002)", "shape": None},
            "c0": {"units": "m/s", "bounds": (10.0, 90.0), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "CAM gw_front Gaussian phase-speed width c0", "shape": None},
            # --- launch / trigger levels ---
            "launch_p": {"units": "Pa", "bounds": (3.0e4, 9.0e4), "tunable_tier": 3, "transform": "sigmoid", "category": "launch_level", "reference": "E3SM gw_front kbotbg launch interface", "shape": None},
            "front_p": {"units": "Pa", "bounds": (4.0e4, 9.0e4), "tunable_tier": 3, "transform": "sigmoid", "category": "launch_level", "reference": "E3SM gw_front kfront trigger level", "shape": None},
        },
    },
    "E3SMOrographicConfig": {
        "scheme_key": "atm.gwd.E3SMOrographicConfig",
        "excluded": {
            "sgh_default": "default subgrid-orography stddev (default 0 = no oro waves); a per-column boundary input supplied by the dataset, not a trained scalar",
        },
        "params": {
            # --- orographic source activation thresholds ---
            "oro_min_h": {"units": "m", "bounds": (1.0, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "orographic", "reference": "McFarlane (1987); E3SM orohmin", "shape": None},
            "oro_min_wind": {"units": "m/s", "bounds": (0.5, 6.0), "tunable_tier": 2, "transform": "sigmoid", "category": "orographic", "reference": "McFarlane (1987); E3SM orovmin", "shape": None},
        },
    },
    "GWDMLEmulatorConfig": {
        "scheme_key": "atm.gwd.GWDMLEmulatorConfig",
        "excluded": {
            "norm_u": "input feature-normalisation wind scale; a preprocessing constant for the MLP, not a physical closure",
            "norm_T": "input feature-normalisation temperature scale; a preprocessing constant for the MLP, not a physical closure",
            "norm_z": "input feature-normalisation height scale; a preprocessing constant for the MLP, not a physical closure",
        },
        "params": {
        },
    },
    "HinesConfig": {
        "scheme_key": "atm.gwd.HinesConfig",
        "excluded": {
            "U_mag_floor": "wind-magnitude floor for the direction projection; a divide-by-zero safety floor, not a closure",
            "doppler_sharpness": "sigmoid sharpness of the smooth saturation gate; a differentiability/smoothing width, not a closure",
        },
        "params": {
            # --- launch source spectrum (Hines 1997) ---
            "total_rms_wind": {"units": "m/s", "bounds": (0.5, 10.0), "tunable_tier": 1, "transform": "sigmoid", "category": "source_spectrum", "reference": "Hines (1997) launch rms wind", "shape": None},
            # --- saturation / momentum-flux cap ---
            "Fmax": {"units": "Pa", "bounds": (0.01, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "saturation", "reference": "Hines (1997) saturation momentum-flux cap", "shape": None},
            # --- tendency limiters ---
            "tndmax_per_day": {"units": "m/s/day", "bounds": (100.0, 1000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common tendency ceiling", "shape": None},
            "umcfac": {"units": "1", "bounds": (0.1, 0.9), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common umcfac no-reversal limiter", "shape": None},
        },
    },
    "LindzenConfig": {
        "scheme_key": "atm.gwd.LindzenConfig",
        "excluded": {
            "Fr_sharpness": "sigmoid sharpness of the saturation stress-ratio breaking transition (NOT a Froude-number transition); a differentiability/smoothing width, not a closure",
            "crit_level_sharpness": "sigmoid sharpness of the smooth critical-level filter; a differentiability/smoothing width, not a closure",
            "crit_level_floor": "signed source-projected wind U_proj (NOT a wind magnitude) at which the smooth critical-level filter is half-on; a smoothing/regulariser offset, not a closure",
        },
        "params": {
            # --- orographic launch amplitude ---
            "h_topo": {"units": "m", "bounds": (50.0, 2000.0), "tunable_tier": 1, "transform": "sigmoid", "category": "orographic", "reference": "Lindzen (1981) subgrid topographic height", "shape": None},
            # --- saturation / wave breaking ---
            "fcrit2": {"units": "1", "bounds": (0.5, 2.0), "tunable_tier": 1, "transform": "sigmoid", "category": "saturation", "reference": "Lindzen (1981) / E3SM fcrit2: critical Froude number squared scaling the saturation CAP VALUE (effkwv semantics, gw_common.F90:153)", "shape": None, "legacy_name": "critical_Fr"},
            # --- tendency limiters ---
            "tndmax_per_day": {"units": "m/s/day", "bounds": (100.0, 1000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common tendency ceiling (orographic)", "shape": None},
            "umcfac": {"units": "1", "bounds": (0.1, 0.9), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common umcfac no-reversal limiter", "shape": None},
        },
    },
    "McFarlaneConfig": {
        "scheme_key": "atm.gwd.McFarlaneConfig",
        "excluded": {
            "crit_level_floor": "signed source-projected wind U_proj (NOT a wind magnitude) at which the smooth critical-level filter is half-on; a smoothing/regulariser offset, not a closure",
            "crit_level_sharpness": "sigmoid sharpness of the smooth critical-level filter; a differentiability/smoothing width, not a closure",
            "min_wind_sharpness": "sigmoid sharpness of the smooth min-wind activation; a differentiability/smoothing width, not a closure",
            "softmin_sharpness": "sigmoid sharpness of the saturation-cap blend; a differentiability/smoothing width, not a closure",
        },
        "params": {
            # --- orographic launch amplitude / efficiency ---
            "G_0": {"units": "1", "bounds": (0.1, 2.0), "tunable_tier": 1, "transform": "sigmoid", "category": "orographic", "reference": "McFarlane (1987)/Palmer et al. (1986) launch-flux factor E", "shape": None},
            "efficiency": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "efficiency", "reference": "McFarlane (1987) breaking efficiency", "shape": None},
            "h_topo": {"units": "m", "bounds": (50.0, 2000.0), "tunable_tier": 1, "transform": "sigmoid", "category": "orographic", "reference": "McFarlane (1987) subgrid topographic height", "shape": None},
            # --- saturation / wave breaking ---
            "fcrit2": {"units": "1", "bounds": (0.5, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "saturation", "reference": "McFarlane (1987); E3SM fcrit2 Froude cap", "shape": None},
            "envelope_scale": {"units": "1", "bounds": (0.25, 4.0), "tunable_tier": 2, "transform": "sigmoid", "category": "saturation", "reference": "McFarlane (1987) vertical envelope scale", "shape": None},
            "directional_spread": {"units": "1", "bounds": (0.25, 4.0), "tunable_tier": 2, "transform": "sigmoid", "category": "source_spectrum", "reference": "McFarlane (1987) multi-directional spreading factor", "shape": None},
            # --- source activation / clips / tendency limiters ---
            "min_wind": {"units": "m/s", "bounds": (0.5, 6.0), "tunable_tier": 2, "transform": "sigmoid", "category": "orographic", "reference": "McFarlane (1987) minimum source-level wind", "shape": None},
            "tau_max": {"units": "Pa", "bounds": (1.0, 50.0), "tunable_tier": 3, "transform": "sigmoid", "category": "numerics", "reference": "McFarlane scheme launch-stress clip", "shape": None},
            "tndmax_per_day": {"units": "m/s/day", "bounds": (100.0, 1000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common tendency ceiling (orographic)", "shape": None},
            "umcfac": {"units": "1", "bounds": (0.1, 0.9), "tunable_tier": 3, "transform": "sigmoid", "category": "damping", "reference": "E3SM gw_common umcfac no-reversal limiter", "shape": None},
        },
    },
    "PrognosticSpectralConfig": {
        "scheme_key": "atm.gwd.PrognosticSpectralConfig",
        "excluded": {
            "breaking_sharpness": "sigmoid sharpness of the breaking transition; a differentiability/smoothing width, not a closure",
            "direction_sign_width": "tanh width of the smooth sign(c - U_launch) launch-fixed directional deposition factor; a differentiability/smoothing width, not a closure",
        },
        "params": {
            # --- launch source spectrum ---
            "launch_flux": {"units": "Pa", "bounds": (1.0e-4, 1.0e-2), "tunable_tier": 1, "transform": "sigmoid", "category": "momentum_flux", "reference": "prognostic-spectral scheme launch momentum flux", "shape": None},
            # --- saturation / wave breaking ---
            "breaking_threshold": {"units": "1", "bounds": (0.5, 2.0), "tunable_tier": 1, "transform": "sigmoid", "category": "saturation", "reference": "Lindzen (1981) Froude breaking threshold", "shape": None},
            # --- prognostic relaxation timescale ---
            "tau_decay": {"units": "s", "bounds": (3.6e3, 2.592e5), "tunable_tier": 2, "transform": "sigmoid", "category": "damping", "reference": "prognostic-spectral relaxation timescale", "shape": None},
        },
    },
    "RayleighConfig": {
        "scheme_key": "atm.gwd.RayleighConfig",
        "excluded": {
        },
        "params": {
            # --- boundary-layer Rayleigh drag ---
            "k_max": {"units": "1/s", "bounds": (1.0e-6, 1.0e-4), "tunable_tier": 1, "transform": "sigmoid", "category": "damping", "reference": "Held & Suarez (1994) boundary-layer drag rate", "shape": None},
            "sigma_b": {"units": "1", "bounds": (0.5, 0.95), "tunable_tier": 2, "transform": "sigmoid", "category": "damping", "reference": "Held & Suarez (1994) boundary-layer top sigma", "shape": None},
            # --- upper sponge ---
            "sponge_k": {"units": "1/s", "bounds": (1.0e-6, 1.0e-4), "tunable_tier": 1, "transform": "sigmoid", "category": "damping", "reference": "Rayleigh sponge drag rate", "shape": None},
            "sponge_top": {"units": "1", "bounds": (0.001, 0.1), "tunable_tier": 2, "transform": "sigmoid", "category": "damping", "reference": "Rayleigh sponge top sigma level", "shape": None},
        },
    },
}


class RayleighConfig(NamedTuple):
    """Configuration for Rayleigh friction drag.

    Fields
    ------
    k_max : float
        Maximum drag coefficient [1/s] (default 1/(1*86400)).
    sigma_b : float
        Boundary layer top sigma level (default 0.7).
    sponge_top : float
        Upper sponge sigma level (default 0.02).
    sponge_k : float
        Upper sponge drag coefficient [1/s] (default 1/(0.5*86400)).
    """
    k_max: float = 1.0 / 86400.0
    sigma_b: float = 0.7
    sponge_top: float = 0.02
    sponge_k: float = 1.0 / (0.5 * 86400.0)


class LindzenConfig(NamedTuple):
    """Configuration for smoothed Lindzen (1981) orographic GWD.

    Fields
    ------
    h_topo : float
        Sub-grid topographic height [m] (default 500).
    k_wave : float
        Horizontal wavenumber [1/m] (default 2*pi/100e3).
    fcrit2 : float
        Critical Froude number squared scaling the saturation CAP VALUE
        (``tau_sat_eff = fcrit2*tau_sat`` — the oracle ``effkwv = kwv*fcrit2``
        semantics, E3SM gw_common.F90:153; Lindzen's limit scales with
        ``Fr_c²``). The breaking sigmoid activates at the fixed threshold
        ``tau_carry > tau_sat_eff`` and relaxes toward ``tau_sat_eff``; the
        saturation is SOFT (finite-sharpness sigmoid), NOT an exact hard cap.
        Default 1.0 (``Fr_c = 1``) caps at the exact Lindzen ``tau_sat``.
        Formerly named ``critical_Fr`` and wired only as the sigmoid
        ACTIVATION center with an unscaled relaxation target, which left the
        knob inert wherever ``tau_carry <= tau_sat``.
    Fr_sharpness : float
        Sigmoid sharpness for the saturation stress-ratio breaking transition
        (default 20.0). NOT a Froude-number transition (see ``fcrit2``).
    crit_level_sharpness : float
        Sigmoid sharpness [s/m] for the smooth critical-level filter
        (default 10.0).  The orographic wave (c = 0) is absorbed where the
        source-projected wind ``U_proj`` reverses sign (E3SM
        gw_common.F90:492 ``where ubmc*(ubi_above - c) > 0``); the earlier
        ``|U_proj|^3`` saturation alone gave no explicit critical-level
        absorption.  This smooth gate handles the differentiable absorption of
        the carried stress; the deposited drag additionally carries a HARD
        ``U_proj > 0`` positivity mask in the scheme body so the VECTOR sink
        ``u*du_dt + v*dv_dt <= 0`` is enforced STRICTLY (only the vector
        projection is guaranteed — componentwise ``du_dt*u`` can be >0 for an
        oblique wind; the smooth sigmoid alone is never identically zero).
"""Shared YAML run-config loader for the production run drivers.

A ``--config FILE`` supplies argument DEFAULTS (applied via
``parser.set_defaults``), so any explicit CLI flag still overrides the file
(precedence: CLI > config file > parser default).  An optional ``include:`` base
is merged FIRST so tuned physics can live in one shared file
(``config/cmip/cmip_tuned_physics.yaml``) and be reused across run configs.

The loader is keyed purely on ``parser._actions`` dests, so it is
driver-agnostic: ``run_coupled.py`` and ``run_amip.py`` each pass their OWN
parser and the file is validated against that driver's CLI surface.  Every
merged key MUST be a known argument dest of the calling parser — an unknown key
is a hard error (dispatch-hardening: no silent typo'd / dropped override).

This is the single source of truth for the ``--config`` mechanism; the drivers
must not re-implement it (CLAUDE.md: no duplicate utilities).
"""

from __future__ import annotations

from pathlib import Path


def read_yaml_with_includes(path, _seen=None) -> dict:
    """Read a run YAML config, recursively merging an optional ``include:`` base
    FIRST so the tuned physics can live in one shared file and be reused across
    run configs.

    Precedence: the including file's keys override the base it includes
    (base < file).  An ``include:`` path is resolved relative to the including
    file.  Cycles and missing/non-mapping files raise.  Returns the merged raw
    dict; ``load_yaml_config`` then validates + type-coerces it.
    """
    import yaml
    p = Path(path).resolve()
    _seen = set() if _seen is None else _seen
    if p in _seen:
        raise SystemExit(f"--config: 'include' cycle detected at {p}.")
    _seen.add(p)
    if not p.exists():
        raise SystemExit(f"--config: file not found: {p}.")
    doc = yaml.safe_load(p.read_text())
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise SystemExit(
            f"--config {path}: expected a YAML mapping of argument=value, "
            f"got {type(doc).__name__}.")
    base_ref = doc.pop("include", None)
    merged: dict = {}
    if base_ref is not None:
        if not isinstance(base_ref, str):
            raise SystemExit(
                f"--config {path}: 'include' must be a single path string, "
                f"got {type(base_ref).__name__}.")
        merged.update(read_yaml_with_includes(p.parent / base_ref, _seen))
    merged.update(doc)  # the including file overrides its base
    return merged


def load_yaml_config(path, parser, *, example_keys: str | None = None) -> dict:
    """Load a run YAML config file into a dict of argument defaults.

    Used by ``--config`` to make a canonical run (e.g. the tuned
    ``config/cmip/cmip_ocean_{slab,3D}.yaml`` or the AMIP production config)
    reproducible from one file.  Supports an optional ``include:`` base merged
    first (see ``read_yaml_with_includes``).  Every (merged) key MUST be a known
    argument dest of ``parser``; an unknown key raises (no silent typo'd /
    dropped override — dispatch-hardening).

    Each scalar is coerced through that argument's ``type=`` callable, because
    ``parser.set_defaults`` (how the caller applies this) BYPASSES argparse's own
    type conversion: a value written as a quoted string (e.g. ``dt: "300"``)
    would otherwise reach the run as a str.  Returns the mapping so the caller
    can feed it to ``parser.set_defaults`` (an explicit CLI flag still wins).

    ``example_keys`` is an optional human hint (driver-specific example dests)
    appended to the unknown-key error to help operators fix typos.
    """
    doc = read_yaml_with_includes(path)
    actions = {a.dest: a for a in parser._actions}
    unknown = sorted(set(doc) - set(actions))
    if unknown:
        hint = f" (e.g. {example_keys})" if example_keys else ""
        raise SystemExit(
            f"--config {path}: unknown key(s) {unknown}. Keys must be valid "
            f"argument dests of this run script{hint}.")
    out: dict = {}
    for key, value in doc.items():
        argtype = getattr(actions[key], "type", None)
        # Coerce only string scalars through the arg's type (a YAML native
        # float/int/bool is already the right Python type; type=None args are
        # str/bool flags that need no conversion).
        if argtype is not None and isinstance(value, str):
            try:
                value = argtype(value)
            except (ValueError, TypeError) as exc:
                raise SystemExit(
                    f"--config {path}: key '{key}' value {value!r} is not a "
                    f"valid {getattr(argtype, '__name__', argtype)}: {exc}")
        # ``parser.set_defaults`` BYPASSES argparse's own ``choices`` check, so
        # a scheme literal (e.g. ``vertical_mixing_scheme: garbage``) supplied
        # via --config would otherwise reach the factory unvalidated.  Enforce
        # it here so a typo'd scheme fails loudly at load (dispatch-hardening),
        # exactly as an explicit CLI flag would.
        choices = getattr(actions[key], "choices", None)
        if choices is not None and value not in choices:
            try:
                allowed = sorted(choices)
            except TypeError:
                allowed = list(choices)
            raise SystemExit(
                f"--config {path}: key '{key}' value {value!r} is not one of "
                f"the allowed choices {allowed}.")
        out[key] = value
    return out


def require_config(config_value, *, driver: str = "run") -> None:
    """Enforce ``--require-config``: a run must be driven by a committed config.

    Raises ``SystemExit`` when ``--require-config`` is set but no ``--config``
    file was supplied, so a production/test run cannot silently fall back to
    parser defaults (issue #691).  A no-op when strict mode is off.
    """
    if config_value is None:
        raise SystemExit(
            f"{driver}: --require-config was set but no --config file was "
            "given.  Pass --config <yaml> so the run is fully specified by a "
            "committed configuration (no hidden parser defaults)."
        )


# ---------------------------------------------------------------------------
# --params : calibration (tuned parameter) loader (issue #691, format #690)
        "--mcfarlane-tau-max", "4.0", "--mcfarlane-k-wave", "1.0e-4",
    ]), parser))
    assert cfg.hines_total_rms_wind == 1.2
    assert cfg.hines_Fmax == 0.05
    assert cfg.mcfarlane_tau_max == 4.0
    assert cfg.mcfarlane_k_wave == 1.0e-4
    assert gwd_config_for(cfg).hines.total_rms_wind == 1.2


def test_every_overlaid_scalar_has_a_run_amip_route():
    """Each scalar gwd_config_for threads must be settable by an operator.
    ``--config`` YAML keys must be argparse dests (load_yaml_config rejects
    anything else), so a flag is the gate for BOTH routes; the --params route
    additionally needs a _ATM_SCALAR_PARAM_MAP entry, which mcfarlane_k_wave
    lacks by design (no __param_spec__ bounds yet)."""
    from legoesm.driver.run_config_yaml import build_atm_scalar_param_map
    from scripts.run.run_amip import build_arg_parser

    dests = {a.dest for a in build_arg_parser()._actions}
    overlaid = ("mcfarlane_k_wave", "mcfarlane_directional_spread",
                "mcfarlane_tau_max", "hines_total_rms_wind", "hines_Fmax")
    # mcfarlane_directional_spread is --params-only (mapped, no flag); every
    # other overlaid scalar must have a flag.
    amap = set(build_atm_scalar_param_map().values())
    for f in overlaid:
        assert f in dests or f in amap, (
            f"{f} is overlaid onto the GWD kernel by gwd_config_for but an "
            "operator cannot set it: no run_amip flag AND no scalar-map entry."
        )


def test_physics_state_seed_honours_a_nested_override():
    """The stateful-carry seed must resolve through the SAME resolver as the
    kernel: init_physics_state sizes gwd_spectrum from
    prognostic_spectral.n_azimuths/.n_wavenumbers/.launch_flux, and an override
    may set all three (codex round 1, finding 2 — the seed at
    model_driver.py:9418 used to build a bare config)."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        PrognosticSpectralConfig,
    )
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig

    inj = GravityWaveDragConfig(
        scheme="prognostic_spectral",
        prognostic_spectral=PrognosticSpectralConfig(
            n_azimuths=8, n_wavenumbers=6, launch_flux=2.5e-3),
    )
    cfg = _cfg(gravity_wave_drag="prognostic_spectral",
               gravity_wave_drag_override=inj)
    ps = init_physics_state(
        3, 8,
        PhysicsConfig(turbulence=TurbulenceConfig(scheme="none"),
                      gravity_wave_drag=gwd_config_for(cfg)),
    )
    assert ps.gwd_spectrum.shape == (3, 8, 6)
    assert float(ps.gwd_spectrum[0, 0, 0]) == 2.5e-3


@pytest.mark.parametrize("field,bad", [
    ("hines_total_rms_wind", -1.0),
    ("hines_total_rms_wind", 0.0),
    ("hines_Fmax", -0.1),
    ("mcfarlane_tau_max", float("nan")),
    ("mcfarlane_k_wave", float("inf")),
    ("mcfarlane_directional_spread", -1.0),
])
def test_validate_strict_rejects_nonpositive_or_nonfinite(field, bad):
    """A negative Fmax makes clip(drag, 0, Fmax) return the NEGATIVE cap at
    every level (constant spurious drag, no error); a non-positive launch rms
    wind silently disables the scheme.  Neither may reach the kernel."""
    with pytest.raises(ValueError, match=field):
        _cfg(gravity_wave_drag="hines", **{field: bad}).validate_strict()


def test_legacy_amip_upconvert_keeps_the_exact_k_wave_default():
    """A legacy checkpoint upconverted through from_amip_config must land on
    the SAME k_wave the kernel default uses (the truncated 6.283185307e-5
    fallback was ~3e-11 off once the overlay went live)."""
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        McFarlaneConfig,
    )
    from legoesm.forcing.amip_config import AMIPExperimentConfig

    # The legacy flat schema has NO mcfarlane_* fields, so the converter's
    # getattr fallback is what supplies k_wave.
    exp = ExperimentConfig.from_amip_config(AMIPExperimentConfig())
    assert exp.mcfarlane_k_wave == McFarlaneConfig().k_wave
    assert gwd_config_for(
        exp._replace(gravity_wave_drag="mcfarlane"),
    ).mcfarlane == McFarlaneConfig()


@pytest.mark.parametrize("scheme", ["mcfarlane", "hines", "mcfarlane+hines"])
def test_kernel_buildable_for_every_wired_scheme(scheme):
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        get_gwd_fn,
    )
    name, fn, _kcfg = get_gwd_fn(gwd_config_for(_cfg(
        gravity_wave_drag=scheme, hines_total_rms_wind=1.5,
        mcfarlane_tau_max=6.0)))
    assert fn is not None
    assert name
./packages/coupler/legoesm/driver/config.py-2566-            ensemble_size=amip_cfg.ensemble_size,
./packages/coupler/legoesm/driver/config.py-2567-        )
./packages/coupler/legoesm/driver/config.py-2568-
./packages/coupler/legoesm/driver/config.py:2569:    def to_amip_config(self):
./packages/coupler/legoesm/driver/config.py-2570-        """Convert to AMIPExperimentConfig for backward-compatible serialization.
./packages/coupler/legoesm/driver/config.py-2571-
./packages/coupler/legoesm/driver/config.py-2572-        Used at serialization boundaries (checkpoint save, legacy config

exec
/usr/bin/bash -lc "sed -n '135,360p' tests/unit/test_params_config_loader.py && sed -n '215,330p' packages/coupler/legoesm/driver/run_config_yaml.py && rg -n -C 2 \"load_params_config|build_atm_scalar_param_map|_ATM_SCALAR_PARAM_MAP|--params|params_config\" scripts/run/run_amip.py packages/coupler/legoesm/driver/run_config_yaml.py tests/unit/test_params_config_loader.py tests/unit/test_run_amip_cli.py && rg -n -C 3 \"validate_strict\\(\\)\" scripts/run/run_amip.py packages/coupler/legoesm/driver/model_driver.py packages/coupler/legoesm/driver/physics_pipeline.py | head -n 400" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 1084ms:
    # spot-check the documented cloud + convection mappings
    assert amap.get("atm.clouds.CloudConfig.q_c_diagnostic") == "cloud_q_c_diagnostic"
    assert amap.get("atm.conv.SBMConfig.rh_ref") == "sbm_RH_ref"


def test_atm_scalar_map_is_pipeline_threaded():
    """Every atm scalar-map entry is threaded END-TO-END: setting its
    ExperimentConfig scalar changes the resolved scheme config the pipeline
    builds.  This is the guard against the name-convention hazard — many
    ``<prefix>_<field>`` scalars EXIST but are never read (codex #691), so the
    map is a verified allowlist, not a convention.  Also fails if a NEW scalar
    becomes threaded but is missing from the map (extend it)."""
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
    )
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import make_hybrid_levels
    from legoesm.training.param_collector import build_registry

    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(10)
    reg = {m.qualified_name: m for m in build_registry()}
    # scheme selector that ACTIVATES each config class in the resolver.
    sel = {
        "CloudConfig": {"cloud_scheme": "sundqvist"},
        "SBMConfig": {"convection": "sbm", "microphysics": "kessler"},
        "BechtoldConfig": {"convection": "bechtold", "microphysics": "kessler"},
        "TiedtkeConfig": {"convection": "tiedtke", "microphysics": "kessler"},
        # warm-rain hard-saturation-adjustment overrides: activate each scheme
        # so _resolve_microphysics threads the flat scalar into its sub-config.
        "KesslerConfig": {"microphysics": "kessler"},
        "MorrisonConfig": {"microphysics": "morrison"},
        "P3Config": {"microphysics": "p3"},
        "SeifertBehengConfig": {"microphysics": "seifert_beheng"},
        "ThompsonConfig": {"microphysics": "thompson"},
        # GWD: gwd_config_for overlays the mcfarlane_*/hines_* scalars onto the
        # scheme leaf, and get_gwd_fn hands the pipeline that LEAF as gwd_config.
        "HinesConfig": {"gravity_wave_drag": "hines"},
        "McFarlaneConfig": {"gravity_wave_drag": "mcfarlane"},
    }
    resolved_attr = {
        "HinesConfig": "gwd_config",
        "McFarlaneConfig": "gwd_config",
        "SBMConfig": "convection_config",
        "BechtoldConfig": "convection_config",
        "TiedtkeConfig": "convection_config",
        "KesslerConfig": "micro_config",
        "MorrisonConfig": "micro_config",
        "P3Config": "micro_config",
        "SeifertBehengConfig": "micro_config",
        "ThompsonConfig": "micro_config",
    }
    sentinel = 0.123456789
    for qname, ec_field in build_atm_scalar_param_map().items():
        m = reg[qname]
        base = {
            "grid": GridConfig(grid_type="cubed_sphere", resolution=4, nlev=10),
            "dycore": DycoreConfig(model_type="hydrostatic",
                                   discretization="cdgrid"),
            ec_field: sentinel,
            **sel.get(m.config_class, {}),
        }
        pipe = build_physics_pipeline(grid, sigma, ExperimentConfig(**base))
        if m.config_class == "CloudConfig":
            # Reconstruct the SAME build_cloud_config call the pipeline makes
            # (all threaded self._cloud_* attrs), else a genuinely-threaded
            # cloud param (p_xr/alpha_xr) would be falsely dropped.
            cc = build_cloud_config(
                "sundqvist",
                rh_crit=getattr(pipe, "_cloud_rh_crit", None),
                q_c_diagnostic=getattr(pipe, "_cloud_q_c_diagnostic", None),
                conv_cloud_max=getattr(pipe, "_cloud_conv_cloud_max", None),
                conv_cloud_condensate=getattr(
                    pipe, "_cloud_conv_cloud_condensate", None),
                cloud_inhomogeneity_factor=getattr(
                    pipe, "_cloud_inhomogeneity_factor", None),
                cloud_optics_inhomogeneity=getattr(
                    pipe, "_cloud_optics_inhomogeneity", None),
                cloud_fsd=getattr(pipe, "_cloud_fsd", None),
                p_xr=getattr(pipe, "_cloud_p_xr", None),
                alpha_xr=getattr(pipe, "_cloud_alpha_xr", None),
                diagnostic_condensate_scheme=getattr(
                    pipe, "_cloud_diagnostic_condensate_scheme", None),
                adiabatic_lwc_rate=getattr(
                    pipe, "_cloud_adiabatic_lwc_rate", None))
            got = getattr(cc, m.field)
        else:
            got = getattr(getattr(pipe, resolved_attr[m.config_class]), m.field)
        assert got == sentinel, (
            f"{qname} -> {ec_field} is in the atm scalar map but the pipeline "
            f"does NOT thread it into {m.config_class} (got {got!r}, not the "
            "sentinel).  Remove it from _ATM_SCALAR_PARAM_MAP or wire the "
            "pipeline to read it."
        )


def test_atm_scalar_map_has_no_under_claim():
    """Reverse of the over-claim test: every production cloud/convection param
    whose convention-named ExperimentConfig scalar the pipeline DOES thread must
    be IN the map — so a newly-threaded scalar can't be silently omitted (codex
    #691).  (The gray-radiation scheme's non-convention scalars are the
    documented conscious exclusion — see _ATM_SCALAR_PARAM_MAP.)"""
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
    )
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import make_hybrid_levels
    from legoesm.training.param_collector import build_registry

    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(10)
    ec_fields = set(ExperimentConfig._fields)
    amap = build_atm_scalar_param_map()
    sentinel = 0.123456789
    # (config_class, ExperimentConfig scalar prefix, scheme selector, how to
    # read the threaded value from the built pipeline).
    schemes = [
        ("CloudConfig", "cloud_", {"cloud_scheme": "sundqvist"},
         lambda pipe, field: getattr(pipe, f"_cloud_{field}", None)),
        ("SBMConfig", "sbm_", {"convection": "sbm", "microphysics": "kessler"},
         lambda pipe, field: getattr(pipe.convection_config, field, None)),
        ("BechtoldConfig", "bechtold_", {"convection": "bechtold",
                                         "microphysics": "kessler"},
         lambda pipe, field: getattr(pipe.convection_config, field, None)),
        # Tiedtke shares the UNPREFIXED autoconv_* flat scalars with Bechtold
        # (one ExperimentConfig scalar serves whichever mass-flux scheme is
        # active), so its convention prefix is empty.
        ("TiedtkeConfig", "", {"convection": "tiedtke",
                               "microphysics": "kessler"},
         lambda pipe, field: getattr(pipe.convection_config, field, None)),
        # Warm-rain micro families: their threaded flat scalars carry the SAME
        # name as the scheme field (hard_sat_adjust_threshold /
        # hard_sat_max_heating_K), so the convention prefix is empty — the
        # ec_lower lookup below simply skips micro fields with no same-named
        # ExperimentConfig scalar.
        ("KesslerConfig", "", {"microphysics": "kessler"},
         lambda pipe, field: getattr(pipe.micro_config, field, None)),
        ("MorrisonConfig", "", {"microphysics": "morrison"},
         lambda pipe, field: getattr(pipe.micro_config, field, None)),
        ("P3Config", "", {"microphysics": "p3"},
         lambda pipe, field: getattr(pipe.micro_config, field, None)),
        ("SeifertBehengConfig", "", {"microphysics": "seifert_beheng"},
         lambda pipe, field: getattr(pipe.micro_config, field, None)),
        ("ThompsonConfig", "", {"microphysics": "thompson"},
         lambda pipe, field: getattr(pipe.micro_config, field, None)),
        # GWD families: _resolve_gwd -> gwd_config_for overlays the flat
        # scalars onto the scheme leaf, which get_gwd_fn returns as gwd_config.
        ("HinesConfig", "hines_", {"gravity_wave_drag": "hines"},
         lambda pipe, field: getattr(pipe.gwd_config, field, None)),
        ("McFarlaneConfig", "mcfarlane_", {"gravity_wave_drag": "mcfarlane"},
         lambda pipe, field: getattr(pipe.gwd_config, field, None)),
    ]
    # Companion drift-guard: the family list scanned below must exactly match
    # the config classes present in the verified allowlist map.  The selector /
    # prefix / reader triple is per-family knowledge that CANNOT be derived
    # from the resolver (threading is scattered hand-written getattr code in
    # physics_pipeline), so when a future resolver threads a NEW scheme
    # family's scalars, extend BOTH _ATM_SCALAR_PARAM_MAP and this `schemes`
    # list — this assertion goes red until both agree.
    reg_by_qname = {m.qualified_name: m for m in build_registry()}
    map_classes = {reg_by_qname[q].config_class for q in amap}
    scanned_classes = {cls for cls, _, _, _ in schemes}
    assert map_classes == scanned_classes, (
        f"_ATM_SCALAR_PARAM_MAP covers config classes {sorted(map_classes)} but "
        f"this under-claim scan covers {sorted(scanned_classes)} — extend the "
        "schemes list (selector + prefix + reader) so newly-threaded families "
        "are scanned too."
    )
    ec_lower = {f.lower(): f for f in ec_fields}
    for cls, prefix, sel, reader in schemes:
        params = [m for m in build_registry()
                  if m.config_class == cls and 1 <= m.tunable_tier <= 2]
        for m in params:
            ec_field = ec_lower.get(f"{prefix}{m.field}".lower())
            if ec_field is None:
                continue  # no convention scalar -> genuinely unreachable
            base = {
                "grid": GridConfig(grid_type="cubed_sphere", resolution=4,
                                   nlev=10),
                "dycore": DycoreConfig(model_type="hydrostatic",
                                       discretization="cdgrid"),
                ec_field: sentinel,
                **sel,
            }
            pipe = build_physics_pipeline(grid, sigma, ExperimentConfig(**base))
            threaded = reader(pipe, m.field) == sentinel
            if threaded:
                assert m.qualified_name in amap, (
                    f"{m.qualified_name} is threaded from ExperimentConfig scalar "
                    f"{ec_field!r} but is MISSING from _ATM_SCALAR_PARAM_MAP — add "
                    "it (a newly-threaded scalar must be settable via --params)."
                )


def test_scalar_map_applies_atm_param_to_flat_experimentconfig():
    """An atm param routes to the FLAT ExperimentConfig scalar (not a nested
    *Config, which ExperimentConfig doesn't have) via scalar_param_map."""
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
    )
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
        cloud_scheme="sundqvist",
    )
    out = apply_params_to_config(
        cfg, {"atm.clouds.CloudConfig.q_c_diagnostic": 3.0e-4},
        driver="test", scalar_param_map=build_atm_scalar_param_map())
    assert out.cloud_q_c_diagnostic == 3.0e-4
    # scheme's sub-config is built).
    "atm.micro.KesslerConfig.hard_sat_adjust_threshold": "hard_sat_adjust_threshold",
    "atm.micro.KesslerConfig.hard_sat_max_heating_K": "hard_sat_max_heating_K",
    "atm.micro.MorrisonConfig.hard_sat_adjust_threshold": "hard_sat_adjust_threshold",
    "atm.micro.MorrisonConfig.hard_sat_max_heating_K": "hard_sat_max_heating_K",
    "atm.micro.P3Config.hard_sat_adjust_threshold": "hard_sat_adjust_threshold",
    "atm.micro.P3Config.hard_sat_max_heating_K": "hard_sat_max_heating_K",
    "atm.micro.SeifertBehengConfig.hard_sat_adjust_threshold": "hard_sat_adjust_threshold",
    "atm.micro.SeifertBehengConfig.hard_sat_max_heating_K": "hard_sat_max_heating_K",
    "atm.micro.ThompsonConfig.hard_sat_adjust_threshold": "hard_sat_adjust_threshold",
    "atm.micro.ThompsonConfig.hard_sat_max_heating_K": "hard_sat_max_heating_K",
    # NOTE: LouisConfig.cloudtop_entrainment_efficiency was REMOVED 2026-07-23
    # for the same #1280 semantic conflict as cloud_inhomogeneity_factor above:
    # upstream excluded it from the __param_spec__ registry (default 0.0 = off
    # sits ON its bound), so the qualified name no longer exists for the
    # --params loader.  The flat scalar louis_cloudtop_entrainment_efficiency
    # remains settable via --config / CLI.  FOLLOW-UP: if --params reachability
    # is wanted back, re-spec the param with an activation-aware transform
    # instead of re-adding a dangling map key.
    # gravity wave drag -> gwd_config_for (physics_pipeline), which every lane
    # (FV pipeline / MPAS / spectral) now routes through.  Before it, these
    # scalars existed on ExperimentConfig but NO production path read them.
    # ``tau_max`` is tier 3 (a numerics clip), so the tier-1/2 reachability
    # audit does not require it — it is mapped anyway because the same resolver
    # threads it and the map's contract is "what the pipeline actually threads".
    "atm.gwd.HinesConfig.total_rms_wind": "hines_total_rms_wind",
    "atm.gwd.HinesConfig.Fmax": "hines_Fmax",
    "atm.gwd.McFarlaneConfig.directional_spread": "mcfarlane_directional_spread",
    "atm.gwd.McFarlaneConfig.tau_max": "mcfarlane_tau_max",
    # NOTE: ``mcfarlane_k_wave`` is threaded too but has NO ``__param_spec__``
    # entry (its computed 2*pi/100e3 default is not a float literal, so the
    # AST-based spec gate never required one) — there is no qualified name to
    # map.  Speccing it needs a bounds decision (ml/tuning.py says 1e-5..2e-4,
    # aimip_params says 1e-5..5e-4).  Until then its ONLY route is the
    # ``--mcfarlane-k-wave`` flag (which is also what makes the key legal in a
    # ``--config`` YAML — load_yaml_config rejects any key that is not a parser
    # dest), and validate_strict guards it positive+finite.
    # NOTE: the idealized GRAY radiation scheme threads a few of its params
    # (tau_equator, tau_pole via same-named scalars; sfc_albedo via the shared
    # `albedo_ocean` scalar) — deliberately NOT in this map.  Gray is not the
    # production radiation (rrtmgp is), and its scalars are set via `--config`
    # directly; the reachability audit baselines them under a documented
    # "idealized / --config-only" reason rather than the qualified-name loader.
}


def build_atm_scalar_param_map() -> dict[str, str]:
    """Return ``{registry qualified_name: ExperimentConfig scalar field}`` for
    the atmosphere tunable parameters that ExperimentConfig exposes as a flat
    scalar AND the physics pipeline actually threads into the scheme config
    (issue #691).  See :data:`_ATM_SCALAR_PARAM_MAP` for why this is a verified
    allowlist rather than a name-convention derivation."""
    return dict(_ATM_SCALAR_PARAM_MAP)


def _route_overrides_by_class(node, by_key: dict, *, applied: set):
    """Recursively splice ``{(module, class_name): {field: value}}`` into a
    config NamedTuple tree (depth-first; child configs updated before parent).

    Keyed on the FULL ``(defining module, class name)`` — not the bare class
    name — so same-named configs in different components (e.g. the ocean
    vertical-mixing ``TKEConfig`` vs the atmosphere turbulence ``TKEConfig``)
    never cross-route.  Records each applied key in ``applied`` so the caller can
    detect a target config that is ABSENT (never applied) or AMBIGUOUS (two
    instances of the same class in the tree)."""
    fields = getattr(node, "_fields", None)
    if fields is None or not isinstance(node, tuple):
        return node  # not a NamedTuple leaf
    replacements = {}
    for f in fields:
        child = getattr(node, f)
        new_child = _route_overrides_by_class(child, by_key, applied=applied)
        if new_child is not child:
            replacements[f] = new_child
    if replacements:
        node = node._replace(**replacements)
    key = (type(node).__module__, type(node).__name__)
    if key in by_key:
        if key in applied:
            raise SystemExit(
                f"--params: config {key[1]!r} ({key[0]}) appears more than once "
                "in the run config tree — cannot route the override unambiguously."
            )
        applied.add(key)
        # apply_param_overrides validates every field is on the NamedTuple.
        from legoesm.training.param_collector import apply_param_overrides
        node = apply_param_overrides(node, by_key[key])
    return node


def apply_params_to_config(config, params: dict, *, driver: str = "run",
                           scalar_param_map: dict | None = None):
    """Return ``config`` with calibration ``params`` (qualified_name: value)
    spliced into the matching nested ``*Config`` NamedTuples.

    ``scalar_param_map`` (``{qualified_name: config_field}``) handles configs
    that FLATTEN their tunable scheme params to top-level scalar fields instead
    of nesting the ``*Config`` — the atmosphere ExperimentConfig (see
    :func:`build_atm_scalar_param_map`).  A mapped parameter is applied to the
    top ``config`` via ``_replace`` (validated + bounds-checked identically);
    unmapped parameters use the nested class-router below.

    Each key is validated against ``param_collector.build_registry`` (must be a
    known parameter and, for scalar params, within its ``__param_spec__``
    bounds) and routed to the config of its declared ``(module, config_class)``.
    Raises ``SystemExit`` on an unknown parameter, a non-numeric or out-of-bounds
    value, or a target config that is absent from / ambiguous in this run's
    config tree — a ``--params`` file can never silently mis-set physics.  A
    no-op for empty ``params``.

    Note (soft limitation): a union config that holds ALL of a family's scheme
    sub-configs simultaneously (``VerticalMixingConfig`` carries kpp/tke/catke;
    ``MultiLayerLandConfig`` carries carbon + stomata) is always "present", so an
    override for a scheme that is not the *selected* one is applied to that
    (inert) sub-config rather than raising — it simply has no effect on the run.
    The strict absent-raise still catches wrong-component params (e.g. an
packages/coupler/legoesm/driver/run_config_yaml.py-133-
packages/coupler/legoesm/driver/run_config_yaml.py-134-# ---------------------------------------------------------------------------
packages/coupler/legoesm/driver/run_config_yaml.py:135:# --params : calibration (tuned parameter) loader (issue #691, format #690)
packages/coupler/legoesm/driver/run_config_yaml.py-136-# ---------------------------------------------------------------------------
packages/coupler/legoesm/driver/run_config_yaml.py:137:# A ``--params FILE`` supplies the CALIBRATION layer: converged scheme-parameter
packages/coupler/legoesm/driver/run_config_yaml.py-138-# values, keyed by the ``param_collector`` qualified name ``scheme_key.field``
packages/coupler/legoesm/driver/run_config_yaml.py-139-# (the SAME keying the registry and the SCM-RCE / AIMIP training output use, e.g.
--
packages/coupler/legoesm/driver/run_config_yaml.py-146-
packages/coupler/legoesm/driver/run_config_yaml.py-147-
packages/coupler/legoesm/driver/run_config_yaml.py:148:def load_params_config(path) -> dict:
packages/coupler/legoesm/driver/run_config_yaml.py-149-    """Load a calibration params YAML (``{qualified_name: value}``) → dict.
packages/coupler/legoesm/driver/run_config_yaml.py-150-
--
packages/coupler/legoesm/driver/run_config_yaml.py-155-    doc = read_yaml_with_includes(path)
packages/coupler/legoesm/driver/run_config_yaml.py-156-    if not isinstance(doc, dict):
packages/coupler/legoesm/driver/run_config_yaml.py:157:        raise SystemExit(f"--params {path}: top level must be a mapping of "
packages/coupler/legoesm/driver/run_config_yaml.py-158-                         "'scheme_key.field: value' entries.")
packages/coupler/legoesm/driver/run_config_yaml.py-159-    return dict(doc)
--
packages/coupler/legoesm/driver/run_config_yaml.py-176-# value) — extend this dict only when the pipeline threads a new scalar
packages/coupler/legoesm/driver/run_config_yaml.py-177-# (issue #691, codex audit).
packages/coupler/legoesm/driver/run_config_yaml.py:178:_ATM_SCALAR_PARAM_MAP: dict[str, str] = {
packages/coupler/legoesm/driver/run_config_yaml.py-179-    # clouds -> build_cloud_config (physics_pipeline)
packages/coupler/legoesm/driver/run_config_yaml.py-180-    "atm.clouds.CloudConfig.rh_crit": "cloud_rh_crit",
--
packages/coupler/legoesm/driver/run_config_yaml.py-196-    # partition (default 1.0 sits ON its physical bound — not a well-posed
packages/coupler/legoesm/driver/run_config_yaml.py-197-    # sigmoid tunable), which drops it from the param registry, and a map key
packages/coupler/legoesm/driver/run_config_yaml.py:198:    # absent from the registry breaks the --params loader contract (the
packages/coupler/legoesm/driver/run_config_yaml.py-199-    # semantic conflict this branch inherited on merge).  The flat
packages/coupler/legoesm/driver/run_config_yaml.py-200-    # ExperimentConfig scalar remains settable via --config / its CLI flag.
--
packages/coupler/legoesm/driver/run_config_yaml.py-204-    # bechtold penetrative-downdraft closure knobs -> the dedicated
packages/coupler/legoesm/driver/run_config_yaml.py-205-    # _bechtold_kwargs threading in _resolve_convection (unconditional), so
packages/coupler/legoesm/driver/run_config_yaml.py:206:    # they are --params-reachable (2026-07-23; previously baselined CLI-only).
packages/coupler/legoesm/driver/run_config_yaml.py-207-    "atm.conv.BechtoldConfig.downdraft_alpha": "bechtold_downdraft_alpha",
packages/coupler/legoesm/driver/run_config_yaml.py-208-    "atm.conv.BechtoldConfig.downdraft_entrain_rate": "bechtold_downdraft_entrain_rate",
--
packages/coupler/legoesm/driver/run_config_yaml.py-228-    # upstream excluded it from the __param_spec__ registry (default 0.0 = off
packages/coupler/legoesm/driver/run_config_yaml.py-229-    # sits ON its bound), so the qualified name no longer exists for the
packages/coupler/legoesm/driver/run_config_yaml.py:230:    # --params loader.  The flat scalar louis_cloudtop_entrainment_efficiency
packages/coupler/legoesm/driver/run_config_yaml.py:231:    # remains settable via --config / CLI.  FOLLOW-UP: if --params reachability
packages/coupler/legoesm/driver/run_config_yaml.py-232-    # is wanted back, re-spec the param with an activation-aware transform
packages/coupler/legoesm/driver/run_config_yaml.py-233-    # instead of re-adding a dangling map key.
--
packages/coupler/legoesm/driver/run_config_yaml.py-259-
packages/coupler/legoesm/driver/run_config_yaml.py-260-
packages/coupler/legoesm/driver/run_config_yaml.py:261:def build_atm_scalar_param_map() -> dict[str, str]:
packages/coupler/legoesm/driver/run_config_yaml.py-262-    """Return ``{registry qualified_name: ExperimentConfig scalar field}`` for
packages/coupler/legoesm/driver/run_config_yaml.py-263-    the atmosphere tunable parameters that ExperimentConfig exposes as a flat
packages/coupler/legoesm/driver/run_config_yaml.py-264-    scalar AND the physics pipeline actually threads into the scheme config
packages/coupler/legoesm/driver/run_config_yaml.py:265:    (issue #691).  See :data:`_ATM_SCALAR_PARAM_MAP` for why this is a verified
packages/coupler/legoesm/driver/run_config_yaml.py-266-    allowlist rather than a name-convention derivation."""
packages/coupler/legoesm/driver/run_config_yaml.py:267:    return dict(_ATM_SCALAR_PARAM_MAP)
packages/coupler/legoesm/driver/run_config_yaml.py-268-
packages/coupler/legoesm/driver/run_config_yaml.py-269-
--
packages/coupler/legoesm/driver/run_config_yaml.py-293-        if key in applied:
packages/coupler/legoesm/driver/run_config_yaml.py-294-            raise SystemExit(
packages/coupler/legoesm/driver/run_config_yaml.py:295:                f"--params: config {key[1]!r} ({key[0]}) appears more than once "
packages/coupler/legoesm/driver/run_config_yaml.py-296-                "in the run config tree — cannot route the override unambiguously."
packages/coupler/legoesm/driver/run_config_yaml.py-297-            )
--
packages/coupler/legoesm/driver/run_config_yaml.py-311-    that FLATTEN their tunable scheme params to top-level scalar fields instead
packages/coupler/legoesm/driver/run_config_yaml.py-312-    of nesting the ``*Config`` — the atmosphere ExperimentConfig (see
packages/coupler/legoesm/driver/run_config_yaml.py:313:    :func:`build_atm_scalar_param_map`).  A mapped parameter is applied to the
packages/coupler/legoesm/driver/run_config_yaml.py-314-    top ``config`` via ``_replace`` (validated + bounds-checked identically);
packages/coupler/legoesm/driver/run_config_yaml.py-315-    unmapped parameters use the nested class-router below.
--
packages/coupler/legoesm/driver/run_config_yaml.py-320-    Raises ``SystemExit`` on an unknown parameter, a non-numeric or out-of-bounds
packages/coupler/legoesm/driver/run_config_yaml.py-321-    value, or a target config that is absent from / ambiguous in this run's
packages/coupler/legoesm/driver/run_config_yaml.py:322:    config tree — a ``--params`` file can never silently mis-set physics.  A
packages/coupler/legoesm/driver/run_config_yaml.py-323-    no-op for empty ``params``.
packages/coupler/legoesm/driver/run_config_yaml.py-324-
--
packages/coupler/legoesm/driver/run_config_yaml.py-343-        if meta is None:
packages/coupler/legoesm/driver/run_config_yaml.py-344-            raise SystemExit(
packages/coupler/legoesm/driver/run_config_yaml.py:345:                f"{driver} --params: unknown parameter {qname!r} (not in the "
packages/coupler/legoesm/driver/run_config_yaml.py-346-                "parameter registry).  Keys must be a param_collector qualified "
packages/coupler/legoesm/driver/run_config_yaml.py-347-                "name 'scheme_key.field' (see config/cmip/params_tuned.yaml)."
--
packages/coupler/legoesm/driver/run_config_yaml.py-355-            except (TypeError, ValueError):
packages/coupler/legoesm/driver/run_config_yaml.py-356-                raise SystemExit(
packages/coupler/legoesm/driver/run_config_yaml.py:357:                    f"{driver} --params: {qname} value {value!r} is not numeric."
packages/coupler/legoesm/driver/run_config_yaml.py-358-                )
packages/coupler/legoesm/driver/run_config_yaml.py-359-            lo, hi = meta.bounds
--
packages/coupler/legoesm/driver/run_config_yaml.py-361-                if not (lo <= value <= hi):
packages/coupler/legoesm/driver/run_config_yaml.py-362-                    raise SystemExit(
packages/coupler/legoesm/driver/run_config_yaml.py:363:                        f"{driver} --params: {qname}={value} is outside its "
packages/coupler/legoesm/driver/run_config_yaml.py-364-                        f"__param_spec__ bounds [{lo}, {hi}]."
packages/coupler/legoesm/driver/run_config_yaml.py-365-                    )
--
packages/coupler/legoesm/driver/run_config_yaml.py-376-        if bad:
packages/coupler/legoesm/driver/run_config_yaml.py-377-            raise SystemExit(
packages/coupler/legoesm/driver/run_config_yaml.py:378:                f"{driver} --params: mapped scalar field(s) {bad} are not on "
packages/coupler/legoesm/driver/run_config_yaml.py-379-                f"{type(config).__name__} — the atmosphere scalar-param map does "
packages/coupler/legoesm/driver/run_config_yaml.py-380-                "not match this driver's config."
--
packages/coupler/legoesm/driver/run_config_yaml.py-387-        examples = ", ".join(sorted(key_to_qname[k] for k in missing))
packages/coupler/legoesm/driver/run_config_yaml.py-388-        raise SystemExit(
packages/coupler/legoesm/driver/run_config_yaml.py:389:            f"{driver} --params: parameter(s) {examples} target config "
packages/coupler/legoesm/driver/run_config_yaml.py-390-            f"class(es) {sorted(k[1] for k in missing)} that are not present in "
packages/coupler/legoesm/driver/run_config_yaml.py-391-            "this run's config (the scheme is not built into this driver's "
--
tests/unit/test_run_amip_cli.py-1792-def test_params_flag_parses():
tests/unit/test_run_amip_cli.py-1793-    parser = build_arg_parser()
tests/unit/test_run_amip_cli.py:1794:    args = parser.parse_args(_AMIP_DUMMY_PATHS + ["--params", "x.yaml"])
tests/unit/test_run_amip_cli.py-1795-    assert args.params == "x.yaml"
tests/unit/test_run_amip_cli.py-1796-
tests/unit/test_run_amip_cli.py-1797-
tests/unit/test_run_amip_cli.py-1798-def test_params_calibration_applies_to_atm_experimentconfig(tmp_path):
tests/unit/test_run_amip_cli.py:1799:    """A --params calibration entry (registry qualified name) applies to the
tests/unit/test_run_amip_cli.py-1800-    flattened ExperimentConfig scalar via the atm scalar-param map — the same
tests/unit/test_run_amip_cli.py-1801-    path run_amip.main() takes (issue #691)."""
tests/unit/test_run_amip_cli.py-1802-    from legoesm.driver.run_config_yaml import (
tests/unit/test_run_amip_cli.py-1803-        apply_params_to_config,
tests/unit/test_run_amip_cli.py:1804:        build_atm_scalar_param_map,
tests/unit/test_run_amip_cli.py:1805:        load_params_config,
tests/unit/test_run_amip_cli.py-1806-    )
tests/unit/test_run_amip_cli.py-1807-    parser = build_arg_parser()
--
tests/unit/test_run_amip_cli.py-1811-    p.write_text("atm.clouds.CloudConfig.q_c_diagnostic: 3.0e-4\n")
tests/unit/test_run_amip_cli.py-1812-    out = apply_params_to_config(
tests/unit/test_run_amip_cli.py:1813:        cfg, load_params_config(str(p)), driver="run_amip",
tests/unit/test_run_amip_cli.py:1814:        scalar_param_map=build_atm_scalar_param_map())
tests/unit/test_run_amip_cli.py-1815-    assert out.cloud_q_c_diagnostic == 3.0e-4
tests/unit/test_run_amip_cli.py-1816-
--
tests/unit/test_params_config_loader.py:1:"""Unit tests for the --params calibration loader (issue #691).
tests/unit/test_params_config_loader.py-2-
tests/unit/test_params_config_loader.py-3-The loader takes a ``{qualified_name: value}`` calibration mapping (the same
--
tests/unit/test_params_config_loader.py-15-from legoesm.driver.run_config_yaml import (
tests/unit/test_params_config_loader.py-16-    apply_params_to_config,
tests/unit/test_params_config_loader.py:17:    build_atm_scalar_param_map,
tests/unit/test_params_config_loader.py:18:    load_params_config,
tests/unit/test_params_config_loader.py-19-)
tests/unit/test_params_config_loader.py-20-
--
tests/unit/test_params_config_loader.py-68-
tests/unit/test_params_config_loader.py-69-
tests/unit/test_params_config_loader.py:70:def test_load_params_config_reads_yaml(tmp_path):
tests/unit/test_params_config_loader.py-71-    p = tmp_path / "params.yaml"
tests/unit/test_params_config_loader.py-72-    p.write_text(f"{_QNAME}: 3.0e-4\n")
tests/unit/test_params_config_loader.py:73:    params = load_params_config(str(p))
tests/unit/test_params_config_loader.py-74-    assert params[_QNAME] == 3.0e-4
tests/unit/test_params_config_loader.py-75-
tests/unit/test_params_config_loader.py-76-
tests/unit/test_params_config_loader.py:77:def test_load_params_config_rejects_non_mapping(tmp_path):
tests/unit/test_params_config_loader.py-78-    p = tmp_path / "bad.yaml"
tests/unit/test_params_config_loader.py-79-    p.write_text("- just\n- a\n- list\n")
tests/unit/test_params_config_loader.py-80-    with pytest.raises(SystemExit):
tests/unit/test_params_config_loader.py:81:        load_params_config(str(p))
tests/unit/test_params_config_loader.py-82-
tests/unit/test_params_config_loader.py-83-
--
tests/unit/test_params_config_loader.py-121-# --- atmosphere flat-scalar map (run_amip / run_coupled), issue #691 --------
tests/unit/test_params_config_loader.py-122-
tests/unit/test_params_config_loader.py:123:def test_build_atm_scalar_param_map_is_valid_and_nonempty():
tests/unit/test_params_config_loader.py-124-    """Every derived (qualified_name -> ExperimentConfig field) pair maps a real
tests/unit/test_params_config_loader.py-125-    registry parameter to a real ExperimentConfig field (drift guard)."""
--
tests/unit/test_params_config_loader.py-128-    reg = {m.qualified_name for m in build_registry()}
tests/unit/test_params_config_loader.py-129-    ec = set(ExperimentConfig._fields)
tests/unit/test_params_config_loader.py:130:    amap = build_atm_scalar_param_map()
tests/unit/test_params_config_loader.py-131-    assert amap, "atm scalar-param map should not be empty"
tests/unit/test_params_config_loader.py-132-    for qname, ec_field in amap.items():
--
tests/unit/test_params_config_loader.py-190-    }
tests/unit/test_params_config_loader.py-191-    sentinel = 0.123456789
tests/unit/test_params_config_loader.py:192:    for qname, ec_field in build_atm_scalar_param_map().items():
tests/unit/test_params_config_loader.py-193-        m = reg[qname]
tests/unit/test_params_config_loader.py-194-        base = {
--
tests/unit/test_params_config_loader.py-228-            f"{qname} -> {ec_field} is in the atm scalar map but the pipeline "
tests/unit/test_params_config_loader.py-229-            f"does NOT thread it into {m.config_class} (got {got!r}, not the "
tests/unit/test_params_config_loader.py:230:            "sentinel).  Remove it from _ATM_SCALAR_PARAM_MAP or wire the "
tests/unit/test_params_config_loader.py-231-            "pipeline to read it."
tests/unit/test_params_config_loader.py-232-        )
--
tests/unit/test_params_config_loader.py-238-    be IN the map — so a newly-threaded scalar can't be silently omitted (codex
tests/unit/test_params_config_loader.py-239-    #691).  (The gray-radiation scheme's non-convention scalars are the
tests/unit/test_params_config_loader.py:240:    documented conscious exclusion — see _ATM_SCALAR_PARAM_MAP.)"""
tests/unit/test_params_config_loader.py-241-    from legoesm.driver.config import (
tests/unit/test_params_config_loader.py-242-        DycoreConfig,
--
tests/unit/test_params_config_loader.py-252-    sigma = make_hybrid_levels(10)
tests/unit/test_params_config_loader.py-253-    ec_fields = set(ExperimentConfig._fields)
tests/unit/test_params_config_loader.py:254:    amap = build_atm_scalar_param_map()
tests/unit/test_params_config_loader.py-255-    sentinel = 0.123456789
tests/unit/test_params_config_loader.py-256-    # (config_class, ExperimentConfig scalar prefix, scheme selector, how to
--
tests/unit/test_params_config_loader.py-297-    # from the resolver (threading is scattered hand-written getattr code in
tests/unit/test_params_config_loader.py-298-    # physics_pipeline), so when a future resolver threads a NEW scheme
tests/unit/test_params_config_loader.py:299:    # family's scalars, extend BOTH _ATM_SCALAR_PARAM_MAP and this `schemes`
tests/unit/test_params_config_loader.py-300-    # list — this assertion goes red until both agree.
tests/unit/test_params_config_loader.py-301-    reg_by_qname = {m.qualified_name: m for m in build_registry()}
--
tests/unit/test_params_config_loader.py-303-    scanned_classes = {cls for cls, _, _, _ in schemes}
tests/unit/test_params_config_loader.py-304-    assert map_classes == scanned_classes, (
tests/unit/test_params_config_loader.py:305:        f"_ATM_SCALAR_PARAM_MAP covers config classes {sorted(map_classes)} but "
tests/unit/test_params_config_loader.py-306-        f"this under-claim scan covers {sorted(scanned_classes)} — extend the "
tests/unit/test_params_config_loader.py-307-        "schemes list (selector + prefix + reader) so newly-threaded families "
--
tests/unit/test_params_config_loader.py-329-                assert m.qualified_name in amap, (
tests/unit/test_params_config_loader.py-330-                    f"{m.qualified_name} is threaded from ExperimentConfig scalar "
tests/unit/test_params_config_loader.py:331:                    f"{ec_field!r} but is MISSING from _ATM_SCALAR_PARAM_MAP — add "
tests/unit/test_params_config_loader.py:332:                    "it (a newly-threaded scalar must be settable via --params)."
tests/unit/test_params_config_loader.py-333-                )
tests/unit/test_params_config_loader.py-334-
--
tests/unit/test_params_config_loader.py-349-    out = apply_params_to_config(
tests/unit/test_params_config_loader.py-350-        cfg, {"atm.clouds.CloudConfig.q_c_diagnostic": 3.0e-4},
tests/unit/test_params_config_loader.py:351:        driver="test", scalar_param_map=build_atm_scalar_param_map())
tests/unit/test_params_config_loader.py-352-    assert out.cloud_q_c_diagnostic == 3.0e-4
--
scripts/run/run_amip.py-129-                             "unknown key is a hard error (no silent typo'd "
scripts/run/run_amip.py-130-                             "override).")
scripts/run/run_amip.py:131:    parser.add_argument("--params", default=None,
scripts/run/run_amip.py-132-                        help="YAML calibration file of tuned parameters keyed by "
scripts/run/run_amip.py-133-                             "param_collector qualified name 'scheme_key.field' "
--
scripts/run/run_amip.py-962-                             "validated configuration); override via "
scripts/run/run_amip.py-963-                             "--hard-sat-adjust-threshold / "
scripts/run/run_amip.py:964:                             "--hard-sat-max-heating-k or --params.")
scripts/run/run_amip.py-965-    parser.add_argument("--hard-sat-adjust-threshold", type=float, default=None,
scripts/run/run_amip.py-966-                        dest="hard_sat_adjust_threshold",
--
scripts/run/run_amip.py-1278-                             "stress tau_0 ~ G_0*rho*N*k*h^2*U. No "
scripts/run/run_amip.py-1279-                             "__param_spec__ entry yet (bounds undecided), so "
scripts/run/run_amip.py:1280:                             "this flag is its ONLY route -- --params cannot "
scripts/run/run_amip.py-1281-                             "reach it.")
scripts/run/run_amip.py-1282-    parser.add_argument("--hard-sat-ice-curve",
--
scripts/run/run_amip.py-2590-
scripts/run/run_amip.py-2591-    config = build_config_from_args(args)
scripts/run/run_amip.py:2592:    # Apply the --params calibration layer to the flattened atmosphere
scripts/run/run_amip.py-2593-    # ExperimentConfig scalar fields (issue #691).
scripts/run/run_amip.py-2594-    if getattr(args, "params", None):
scripts/run/run_amip.py-2595-        from legoesm.driver.run_config_yaml import (
scripts/run/run_amip.py-2596-            apply_params_to_config,
scripts/run/run_amip.py:2597:            build_atm_scalar_param_map,
scripts/run/run_amip.py:2598:            load_params_config,
scripts/run/run_amip.py-2599-        )
scripts/run/run_amip.py-2600-        config = apply_params_to_config(
scripts/run/run_amip.py:2601:            config, load_params_config(args.params), driver="run_amip",
scripts/run/run_amip.py:2602:            scalar_param_map=build_atm_scalar_param_map())
scripts/run/run_amip.py-2603-
scripts/run/run_amip.py-2604-    from legoesm.driver.model_driver import ModelDriver
packages/coupler/legoesm/driver/model_driver.py-821-        # scale-guard or shape crash (codex M2 review).
packages/coupler/legoesm/driver/model_driver.py-822-        self._reject_shallow_water_unrunnable()
packages/coupler/legoesm/driver/model_driver.py-823-        # Strict validation — abort early on invalid parameters
packages/coupler/legoesm/driver/model_driver.py:824:        self.config.validate_strict()
packages/coupler/legoesm/driver/model_driver.py-825-
packages/coupler/legoesm/driver/model_driver.py-826-        # Bootstrap runtime: precision, backend, devices, and (optionally) MPI.
packages/coupler/legoesm/driver/model_driver.py-827-        # This is the canonical single entry point — handles everything before
--
packages/coupler/legoesm/driver/model_driver.py-874-        probe MUST be built from the SAME config that drives the run.
packages/coupler/legoesm/driver/model_driver.py-875-        """
packages/coupler/legoesm/driver/model_driver.py-876-        if self._phis_data is None:
packages/coupler/legoesm/driver/model_driver.py:877:            self.config.validate_strict()
packages/coupler/legoesm/driver/model_driver.py-878-            self._bootstrap_runtime()
packages/coupler/legoesm/driver/model_driver.py-879-            self._create_grid()
packages/coupler/legoesm/driver/model_driver.py-880-            self._create_topography()

exec
/usr/bin/bash -lc 'PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 pytest -q -p no:cacheprovider tests/unit/test_gwd_config_for.py tests/unit/test_gwd_override.py tests/unit/test_params_config_loader.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 127 in 305ms:
/usr/bin/bash: pytest: command not found

exec
/usr/bin/bash -lc "rg -n -C 3 \"class McFarlaneConfig|k_wave:\" packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/config.py && sed -n '400,445p' packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/config.py && sed -n '785,840p' packages/coupler/legoesm/driver/model_driver.py && sed -n '2575,2620p' scripts/run/run_amip.py && rg -n -C 3 \"experiment_config_to_dict\\(|config_to_dict\\(\" packages/coupler/legoesm/driver/checkpoint.py packages scripts --glob '*.py' -g '"'!tests/**'"' -g '"'!**/tests/**'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 668ms:
275-        step cannot reverse the wind past the phase speed.
276-    """
277-    h_topo: float = 500.0
278:    k_wave: float = 2.0 * math.pi / 100e3
279-    fcrit2: float = 1.0
280-    Fr_sharpness: float = 20.0
281-    crit_level_sharpness: float = 10.0
--
284-    umcfac: float = 0.5
285-
286-
287:class McFarlaneConfig(NamedTuple):
288-    """Configuration for smoothed McFarlane (1987) orographic GWD.
289-
290-    Extends the Lindzen approach with explicit launch flux control
--
397-        single-step tendency from reversing the wind past the phase speed.
398-    """
399-    h_topo: float = 500.0
400:    k_wave: float = 2.0 * math.pi / 100e3
401-    G_0: float = 0.5
402-    efficiency: float = 0.5
403-    min_wind: float = 2.0
    k_wave: float = 2.0 * math.pi / 100e3
    G_0: float = 0.5
    efficiency: float = 0.5
    min_wind: float = 2.0
    envelope_scale: float = 1.0
    directional_spread: float = 1.0
    min_wind_sharpness: float = 20.0
    softmin_sharpness: float = 50.0
    tau_max: float = 10.0
    fcrit2: float = 1.0
    use_e3sm_hdsp: bool | str = "auto"
    use_depth_averaged_source: bool = True
    crit_level_sharpness: float = 10.0
    crit_level_floor: float = 0.5
    tndmax_per_day: float = 500.0
    umcfac: float = 0.5


class HinesConfig(NamedTuple):
    """Configuration for Hines (1997) Doppler-spread parameterization.

    Fields
    ------
    m_star : float
        Characteristic vertical wavenumber [1/m] (default 2*pi/2e3).
    total_rms_wind : float
        Total RMS gravity wave wind [m/s] (default 2.0).
    Fmax : float
        Saturation momentum flux cap [Pa] (default 0.1).
    doppler_sharpness : float
        Sigmoid sharpness for Doppler saturation (default 50.0).
    tndmax_per_day : float
        Absolute ceiling on ``|du/dt|`` [m/s/day] (default 400.0, CAM
        ``tndmax`` for spectral/non-orographic sources; gw_common.F90:158).
        Caps the ``Fmax/(rho*dz)`` accelerations that blow up in thin,
        low-density upper layers where a fixed momentum-flux cap is divided
        by a tiny ``rho*dz``.
    umcfac : float
        Maximum fraction of the local wind magnitude the deposition may
        remove per step (default 0.5, CAM ``umcfac``; gw_common.F90:642), so
        the single-step drag cannot reverse the wind.
    """
    m_star: float = 2.0 * math.pi / 2e3
    total_rms_wind: float = 2.0
    Fmax: float = 0.1
    doppler_sharpness: float = 50.0
        tracers = getattr(self, "tracers", None)
        have_slots = 0
        if isinstance(tracers, dict):
            for name in self.tracer_registry.names:
                if tracers.get(name) is None:
                    break
                have_slots += 1
        return validate_microphysics_tracer_slots(
            self.config.microphysics,
            have_slots,
            context=context,
        )

    def _reject_shallow_water_unrunnable(self) -> None:
        """Shallow-water is not a runnable ModelDriver equation set.

        ``_init_state`` builds a hydrostatic primitive-equation state
        (``held_suarez_init`` / ``isothermal_rest_state_spectral``), never a
        shallow-water state, so a SW dycore would be handed a PE state and
        crash cryptically at the first step.  Reject LOUDLY at the public
        entry points (setup/run) and as an _init_state backstop (codex M2
        review).  The component factory still builds the correct
        ``FV3EdgeShallowWaterModel`` for component-registry / build-time use.
        """
        if self.config.dycore.model_type == "shallow_water":
            raise NotImplementedError(
                "shallow-water is not runnable via ModelDriver: it builds a "
                "hydrostatic primitive-equation state, not a shallow-water "
                "state.  Use `legoesm test williamson` or "
                "`scripts/matrix/run_atmosphere_test_matrix.py --only sw` "
                "(both construct the SW model + initial state directly).")

    def setup(self) -> None:
        """Initialize grid, dycore, physics, forcing, and state."""
        # SW is not a runnable ModelDriver equation set — reject before any
        # dycore/state construction so the failure is clear, not a downstream
        # scale-guard or shape crash (codex M2 review).
        self._reject_shallow_water_unrunnable()
        # Strict validation — abort early on invalid parameters
        self.config.validate_strict()

        # Bootstrap runtime: precision, backend, devices, and (optionally) MPI.
        # This is the canonical single entry point — handles everything before
        # any JAX array creation.
        self._bootstrap_runtime()

        # Config cross-validation
        config_warnings = self.config.validate()
        for w in config_warnings:
            logger.warning(f"  Config: {w}")

        # Only rank 0 creates output directory (or single-rank)
        if self._mpi_rank is None or self._mpi_rank == 0:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        self._create_grid()
        self._create_topography()
        except ValueError:
            # N > 96 — auto_dt_rce refuses; no comparison possible.
            _auto = None
        if _auto is not None and args.dt > 2.0 * _auto:
            print(
                f"WARNING: --dt {args.dt:.0f} s exceeds the iter-13/26 "
                f"AMIP/RCE cross-grid ladder ({_auto:.0f} s for "
                f"{args.grid_type}/N={args.resolution}) by "
                f"{args.dt / _auto:.1f}×. Long runs at this dt may "
                "BLOWUP (see CRM_implementation.md iter-12/20). "
                "Pass --dt explicitly to suppress this warning.",
                file=sys.stderr,
            )
    except ImportError:
        pass

    config = build_config_from_args(args)
    # Apply the --params calibration layer to the flattened atmosphere
    # ExperimentConfig scalar fields (issue #691).
    if getattr(args, "params", None):
        from legoesm.driver.run_config_yaml import (
            apply_params_to_config,
            build_atm_scalar_param_map,
            load_params_config,
        )
        config = apply_params_to_config(
            config, load_params_config(args.params), driver="run_amip",
            scalar_param_map=build_atm_scalar_param_map())

    from legoesm.driver.model_driver import ModelDriver

    driver = ModelDriver(config)
    print("Setup...")
    driver.setup()

    _is_root = (driver._mpi_rank is None or driver._mpi_rank == 0)

    # AIMIP-classical trained params: override the built pipeline's settable
    # scheme configs with the trained tiedtke / louis / mcfarlane values (the
    # same post-setup, pre-run() mutation the driver does for f_land / albedo;
    # captured at compile).  Cloud (xu_randall) was seeded via the flat fields
    # above (it is built inline in the pipeline).
    if getattr(args, "_aimip_params", None) is not None:
        _p = args._aimip_params
        driver.physics.convection_config = _p.to_tiedtke_config()
        # to_louis_config() rebuilds SurfaceLayerConfig at the DEFAULTS
packages/coupler/legoesm/driver/checkpoint.py-171-    root.attrs["step"] = int(step)
packages/coupler/legoesm/driver/checkpoint.py-172-    root.attrs["day"] = float(day)
packages/coupler/legoesm/driver/checkpoint.py-173-    from legoesm.driver.config import config_to_dict
packages/coupler/legoesm/driver/checkpoint.py:174:    root.attrs["config_json"] = json.dumps(config_to_dict(config))
packages/coupler/legoesm/driver/checkpoint.py-175-
packages/coupler/legoesm/driver/checkpoint.py-176-    zarr.consolidate_metadata(root.store)
packages/coupler/legoesm/driver/checkpoint.py-177-
--
scripts/run/run_correction_campaign.py-1287-    path = _effective_config_path(out_path)
scripts/run/run_correction_campaign.py-1288-    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
scripts/run/run_correction_campaign.py-1289-    with open(path, "w") as f:
scripts/run/run_correction_campaign.py:1290:        json.dump(config_to_dict(base_cfg), f, indent=2)
scripts/run/run_correction_campaign.py-1291-    return path
scripts/run/run_correction_campaign.py-1292-
scripts/run/run_correction_campaign.py-1293-
--
packages/ocean/legoesm/ocean/config.py-698-    )
packages/ocean/legoesm/ocean/config.py-699-
packages/ocean/legoesm/ocean/config.py-700-
packages/ocean/legoesm/ocean/config.py:701:def ocean_config_to_dict(cfg) -> dict:
packages/ocean/legoesm/ocean/config.py-702-    """Recursively serialize an ocean runtime config NamedTuple to a tagged dict.
packages/ocean/legoesm/ocean/config.py-703-
packages/ocean/legoesm/ocean/config.py-704-    Every NamedTuple level is tagged with ``__type__`` = ``"module:QualName"``;
--
packages/tools/legoesm/forcing/amip_config.py-198-    physics_parameterization_seed: int = 0
packages/tools/legoesm/forcing/amip_config.py-199-
packages/tools/legoesm/forcing/amip_config.py-200-
packages/tools/legoesm/forcing/amip_config.py:201:def config_to_dict(config) -> dict:
packages/tools/legoesm/forcing/amip_config.py-202-    """Generic config -> JSON-safe dict codec.
packages/tools/legoesm/forcing/amip_config.py-203-
packages/tools/legoesm/forcing/amip_config.py-204-    Thin delegator to the canonical home ``legoesm.driver.config.config_to_dict``
--
packages/tools/legoesm/forcing/amip_config.py-224-def save_config(config: AMIPExperimentConfig, path: Path) -> None:
packages/tools/legoesm/forcing/amip_config.py-225-    """Save experiment config to JSON."""
packages/tools/legoesm/forcing/amip_config.py-226-    with open(path, "w") as f:
packages/tools/legoesm/forcing/amip_config.py:227:        json.dump(config_to_dict(config), f, indent=2)
packages/tools/legoesm/forcing/amip_config.py-228-
packages/tools/legoesm/forcing/amip_config.py-229-
packages/tools/legoesm/forcing/amip_config.py-230-def load_config(path: Path) -> AMIPExperimentConfig:
--
packages/tools/legoesm/forcing/amip_config.py-291-        arrays["q_r"] = np.asarray(q_r)
packages/tools/legoesm/forcing/amip_config.py-292-
packages/tools/legoesm/forcing/amip_config.py-293-    # Save config as JSON string inside npz
packages/tools/legoesm/forcing/amip_config.py:294:    arrays["config_json"] = np.array(json.dumps(config_to_dict(config)))
packages/tools/legoesm/forcing/amip_config.py-295-
packages/tools/legoesm/forcing/amip_config.py-296-    # Save diagnostic accumulators if provided
packages/tools/legoesm/forcing/amip_config.py-297-    if diag_accumulators:
--
packages/coupler/legoesm/driver/checkpoint.py-171-    root.attrs["step"] = int(step)
packages/coupler/legoesm/driver/checkpoint.py-172-    root.attrs["day"] = float(day)
packages/coupler/legoesm/driver/checkpoint.py-173-    from legoesm.driver.config import config_to_dict
packages/coupler/legoesm/driver/checkpoint.py:174:    root.attrs["config_json"] = json.dumps(config_to_dict(config))
packages/coupler/legoesm/driver/checkpoint.py-175-
packages/coupler/legoesm/driver/checkpoint.py-176-    zarr.consolidate_metadata(root.store)
packages/coupler/legoesm/driver/checkpoint.py-177-
--
packages/coupler/legoesm/driver/distributed_checkpoint.py-140-            "n_ranks": n_ranks,
packages/coupler/legoesm/driver/distributed_checkpoint.py-141-        }
packages/coupler/legoesm/driver/distributed_checkpoint.py-142-        if config is not None:
packages/coupler/legoesm/driver/distributed_checkpoint.py:143:            meta["config"] = config_to_dict(config)
packages/coupler/legoesm/driver/distributed_checkpoint.py-144-
packages/coupler/legoesm/driver/distributed_checkpoint.py-145-        with open(path / "metadata.json", "w") as f:
packages/coupler/legoesm/driver/distributed_checkpoint.py-146-            json.dump(meta, f, indent=2)
--
packages/coupler/legoesm/driver/distributed_checkpoint.py-302-            "sharded": True,
packages/coupler/legoesm/driver/distributed_checkpoint.py-303-        }
packages/coupler/legoesm/driver/distributed_checkpoint.py-304-        if config is not None:
packages/coupler/legoesm/driver/distributed_checkpoint.py:305:            meta["config"] = config_to_dict(config)
packages/coupler/legoesm/driver/distributed_checkpoint.py-306-
packages/coupler/legoesm/driver/distributed_checkpoint.py-307-        with open(path / "metadata.json", "w") as f:
packages/coupler/legoesm/driver/distributed_checkpoint.py-308-            json.dump(meta, f, indent=2)
--
packages/coupler/legoesm/driver/distributed_checkpoint.py-382-        root.attrs["day"] = float(day)
packages/coupler/legoesm/driver/distributed_checkpoint.py-383-        root.attrs["n_ranks"] = n_ranks
packages/coupler/legoesm/driver/distributed_checkpoint.py-384-        if config is not None:
packages/coupler/legoesm/driver/distributed_checkpoint.py:385:            root.attrs["config_json"] = json.dumps(config_to_dict(config))
packages/coupler/legoesm/driver/distributed_checkpoint.py-386-
packages/coupler/legoesm/driver/distributed_checkpoint.py-387-        zarr.consolidate_metadata(root.store)
packages/coupler/legoesm/driver/distributed_checkpoint.py-388-
--
packages/coupler/legoesm/driver/restart.py-105-    # import (canonical codec) — restart routes through the cycle-prone driver
packages/coupler/legoesm/driver/restart.py-106-    # package, so a top-level import is unsafe.
packages/coupler/legoesm/driver/restart.py-107-    from legoesm.driver.config import config_to_dict
packages/coupler/legoesm/driver/restart.py:108:    return config_to_dict(config)
packages/coupler/legoesm/driver/restart.py-109-
packages/coupler/legoesm/driver/restart.py-110-
packages/coupler/legoesm/driver/restart.py-111-# ---------------------------------------------------------------------------
--
packages/coupler/legoesm/driver/restart.py-135-def _serialize_config(config, kind: str) -> dict:
packages/coupler/legoesm/driver/restart.py-136-    if kind == "ocean":
packages/coupler/legoesm/driver/restart.py-137-        from legoesm.ocean.config import ocean_config_to_dict
packages/coupler/legoesm/driver/restart.py:138:        return ocean_config_to_dict(config)
packages/coupler/legoesm/driver/restart.py-139-    return _config_to_dict_any(config)
packages/coupler/legoesm/driver/restart.py-140-
packages/coupler/legoesm/driver/restart.py-141-
--
packages/coupler/legoesm/driver/config.py-2727-}
packages/coupler/legoesm/driver/config.py-2728-
packages/coupler/legoesm/driver/config.py-2729-
packages/coupler/legoesm/driver/config.py:2730:def experiment_config_to_dict(config: ExperimentConfig) -> dict:
packages/coupler/legoesm/driver/config.py-2731-    """Serialize ExperimentConfig to a JSON-safe dict.
packages/coupler/legoesm/driver/config.py-2732-
packages/coupler/legoesm/driver/config.py-2733-    Sub-configs (grid, dycore, output) are inlined as nested dicts. This
--
packages/coupler/legoesm/driver/config.py-2786-    return ExperimentConfig(**filtered)
packages/coupler/legoesm/driver/config.py-2787-
packages/coupler/legoesm/driver/config.py-2788-
packages/coupler/legoesm/driver/config.py:2789:def config_to_dict(config) -> dict:
packages/coupler/legoesm/driver/config.py-2790-    """Generic config -> JSON-safe dict codec (canonical home).
packages/coupler/legoesm/driver/config.py-2791-
packages/coupler/legoesm/driver/config.py-2792-    Accepts an ``ExperimentConfig`` or the legacy flat ``AMIPExperimentConfig``;
--
packages/coupler/legoesm/driver/config.py-2825-def save_experiment_config(config: ExperimentConfig, path: Path | str) -> None:
packages/coupler/legoesm/driver/config.py-2826-    """Save ExperimentConfig to JSON file."""
packages/coupler/legoesm/driver/config.py-2827-    with open(path, "w") as f:
packages/coupler/legoesm/driver/config.py:2828:        json.dump(experiment_config_to_dict(config), f, indent=2, default=str)
packages/coupler/legoesm/driver/config.py-2829-
packages/coupler/legoesm/driver/config.py-2830-
packages/coupler/legoesm/driver/config.py-2831-def load_experiment_config(path: Path | str) -> ExperimentConfig:
--
scripts/experiment/write_amip_clubb_lite_config.py-140-        days=args.days, land_mask_path=args.land_mask_path, grid_type=args.grid_type,
scripts/experiment/write_amip_clubb_lite_config.py-141-        diag_days=args.diag_days)
scripts/experiment/write_amip_clubb_lite_config.py-142-    with open(args.out, "w") as f:
scripts/experiment/write_amip_clubb_lite_config.py:143:        json.dump(experiment_config_to_dict(cfg), f, indent=2)
scripts/experiment/write_amip_clubb_lite_config.py-144-    print(f"[config] wrote AMIP clubb_lite base config (turbulence=clubb_lite, "
scripts/experiment/write_amip_clubb_lite_config.py-145-          f"{args.grid_type} {args.resolution} L{args.nlev}, {args.days}-day climatology) "
scripts/experiment/write_amip_clubb_lite_config.py-146-          f"to {args.out}")

exec
/usr/bin/bash -lc "rg -n -C 3 \"gwd_config|gravity_wave_drag_override|gravity_wave_drag=.*(mcfarlane|hines)|mcfarlane_tau_max|hines_Fmax|hines_total_rms_wind\" tests --glob '*.py' | head -n 1600 && rg -n -C 3 \"def _run_mpas|def _run_spectral|PhysicsConfig\\(\" packages/coupler/legoesm/driver/model_driver.py | sed -n '1,1000p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 607ms:
tests/integration/test_aimip_smoke.py-191-        aimip_variant="classical",
tests/integration/test_aimip_smoke.py-192-        convection="tiedtke",
tests/integration/test_aimip_smoke.py-193-        turbulence="louis",
tests/integration/test_aimip_smoke.py:194:        gravity_wave_drag="mcfarlane",
tests/integration/test_aimip_smoke.py-195-        cloud_scheme="xu_randall",
tests/integration/test_aimip_smoke.py-196-    )
tests/integration/test_aimip_smoke.py-197-    cfg.validate_strict()  # should not raise
--
tests/integration/test_aimip_smoke.py-210-        aimip_variant="classical",
tests/integration/test_aimip_smoke.py-211-        convection="sbm",          # wrong: must be tiedtke
tests/integration/test_aimip_smoke.py-212-        turbulence="louis",
tests/integration/test_aimip_smoke.py:213:        gravity_wave_drag="mcfarlane",
tests/integration/test_aimip_smoke.py-214-        cloud_scheme="xu_randall",
tests/integration/test_aimip_smoke.py-215-    )
tests/integration/test_aimip_smoke.py-216-    with pytest.raises(ValueError, match="tiedtke"):
--
tests/unit/test_config_validation.py-70-    def test_combined_mcfarlane_spectral_is_valid(self):
tests/unit/test_config_validation.py-71-        # The #834 target composite must pass strict validation.
tests/unit/test_config_validation.py-72-        ExperimentConfig(
tests/unit/test_config_validation.py:73:            gravity_wave_drag="mcfarlane+prognostic_spectral"
tests/unit/test_config_validation.py-74-        ).validate_strict()
tests/unit/test_config_validation.py-75-
tests/unit/test_config_validation.py-76-    def test_stateless_composite_is_valid(self):
tests/unit/test_config_validation.py:77:        ExperimentConfig(gravity_wave_drag="hines+mcfarlane").validate_strict()
tests/unit/test_config_validation.py-78-
tests/unit/test_config_validation.py-79-    def test_single_prognostic_spectral_is_valid(self):
tests/unit/test_config_validation.py-80-        ExperimentConfig(gravity_wave_drag="prognostic_spectral").validate_strict()
--
tests/unit/test_config_validation.py-83-        import pytest
tests/unit/test_config_validation.py-84-        with pytest.raises(ValueError, match="composite gravity_wave_drag"):
tests/unit/test_config_validation.py-85-            ExperimentConfig(
tests/unit/test_config_validation.py:86:                gravity_wave_drag="mcfarlane+nonsense"
tests/unit/test_config_validation.py-87-            ).validate_strict()
tests/unit/test_config_validation.py-88-
tests/unit/test_config_validation.py-89-    def test_two_stateful_parts_raises(self):
--
tests/unit/test_config_validation.py-98-        import pytest
tests/unit/test_config_validation.py-99-        with pytest.raises(ValueError, match="composite gravity_wave_drag"):
tests/unit/test_config_validation.py-100-            ExperimentConfig(
tests/unit/test_config_validation.py:101:                gravity_wave_drag="mcfarlane+e3sm_cam"
tests/unit/test_config_validation.py-102-            ).validate_strict()
tests/unit/test_config_validation.py-103-
tests/unit/test_config_validation.py-104-
--
tests/atmosphere/hydrostatic/unit/test_e3sm_cam_gwd_faithful.py-543-    the frontal source could NEVER fire — a dead-on-arrival trigger the
tests/atmosphere/hydrostatic/unit/test_e3sm_cam_gwd_faithful.py-544-    2026-07-20 frontal A/B exposed.  The __param_spec__ tuning window must
tests/atmosphere/hydrostatic/unit/test_e3sm_cam_gwd_faithful.py-545-    bracket BOTH namelist values so --params can reach them."""
tests/atmosphere/hydrostatic/unit/test_e3sm_cam_gwd_faithful.py:546:    from legoesm.atmosphere.physics.gravity_wave_drag import config as gwd_config
tests/atmosphere/hydrostatic/unit/test_e3sm_cam_gwd_faithful.py-547-
tests/atmosphere/hydrostatic/unit/test_e3sm_cam_gwd_faithful.py-548-    cfg = E3SMFrontalConfig()
tests/atmosphere/hydrostatic/unit/test_e3sm_cam_gwd_faithful.py-549-    assert cfg.frontgfc == 1.25e-15
tests/atmosphere/hydrostatic/unit/test_e3sm_cam_gwd_faithful.py:550:    spec = gwd_config.__param_spec__["E3SMFrontalConfig"]["params"]["frontgfc"]
tests/atmosphere/hydrostatic/unit/test_e3sm_cam_gwd_faithful.py-551-    lo, hi = spec["bounds"]
tests/atmosphere/hydrostatic/unit/test_e3sm_cam_gwd_faithful.py-552-    assert lo < 7.5e-16 < hi, "bounds must cover the 4x5 coarse-grid value"
tests/atmosphere/hydrostatic/unit/test_e3sm_cam_gwd_faithful.py-553-    assert lo < 1.25e-15 < hi, "bounds must cover the operational default"
--
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py-308-
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py-309-    def test_mcfarlane(self):
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py-310-        state, grid, sigma = _make_hydrostatic_setup()
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:311:        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="mcfarlane"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py-312-        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py-313-        _check_tendencies(tend, "gwd/mcfarlane")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py-314-
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py-315-    def test_hines(self):
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py-316-        state, grid, sigma = _make_hydrostatic_setup()
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:317:        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="hines"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py-318-        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py-319-        _check_tendencies(tend, "gwd/hines")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py-320-
--
tests/atmosphere/hydrostatic/unit/test_gwd_composite.py-92-    from legoesm.driver.config import ExperimentConfig
tests/atmosphere/hydrostatic/unit/test_gwd_composite.py-93-
tests/atmosphere/hydrostatic/unit/test_gwd_composite.py-94-    with pytest.raises(ValueError, match="duplicate parts"):
tests/atmosphere/hydrostatic/unit/test_gwd_composite.py:95:        ExperimentConfig(gravity_wave_drag="mcfarlane+mcfarlane").validate_strict()
tests/atmosphere/hydrostatic/unit/test_gwd_composite.py-96-    # a valid composite still passes strict validation
tests/atmosphere/hydrostatic/unit/test_gwd_composite.py:97:    ExperimentConfig(gravity_wave_drag="hines+mcfarlane").validate_strict()
tests/atmosphere/hydrostatic/unit/test_gwd_composite.py-98-
tests/atmosphere/hydrostatic/unit/test_gwd_composite.py-99-
tests/atmosphere/hydrostatic/unit/test_gwd_composite.py-100-def test_orographic_helper_truth_table():
--
tests/unit/test_physics_gwd.py-434-    return state, grid, sigma
tests/unit/test_physics_gwd.py-435-
tests/unit/test_physics_gwd.py-436-
tests/unit/test_physics_gwd.py:437:def _gwd_config_with_active_drag(scheme):
tests/unit/test_physics_gwd.py-438-    """Per-scheme config tuned so the held_suarez-init test column
tests/unit/test_physics_gwd.py-439-    actually generates non-trivial drag.
tests/unit/test_physics_gwd.py-440-
--
tests/unit/test_physics_gwd.py-505-    def loss(T_perturb):
tests/unit/test_physics_gwd.py-506-        T_perturbed = T_field.replace(data=T_field.data + T_perturb)
tests/unit/test_physics_gwd.py-507-        state_p = state._replace(T=T_perturbed)
tests/unit/test_physics_gwd.py:508:        config = _gwd_config_with_active_drag(scheme)
tests/unit/test_physics_gwd.py-509-        gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
tests/unit/test_physics_gwd.py-510-        tend, _ = gwd_fn(state_p, grid, sigma)
tests/unit/test_physics_gwd.py-511-        return jnp.sum(tend.dT_dt.data ** 2)
--
tests/unit/test_physics_gwd.py-576-    def loss(u_perturb):
tests/unit/test_physics_gwd.py-577-        u_perturbed = u_field.replace(data=u_field.data + u_perturb)
tests/unit/test_physics_gwd.py-578-        state_p = state._replace(u=u_perturbed)
tests/unit/test_physics_gwd.py:579:        config = _gwd_config_with_active_drag(scheme)
tests/unit/test_physics_gwd.py-580-        gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
tests/unit/test_physics_gwd.py-581-        tend, _ = gwd_fn(state_p, grid, sigma)
tests/unit/test_physics_gwd.py-582-        return jnp.sum(tend.du_dt.data ** 2)
--
tests/unit/test_gwd_override.py:1:"""``gravity_wave_drag_override``: the coupled-path route to nested GWD
tests/unit/test_gwd_override.py-2-scheme options, mirroring ``turbulence_override``.
tests/unit/test_gwd_override.py-3-
tests/unit/test_gwd_override.py-4-Codex wave-4 P2: ``_resolve_gwd`` rebuilt ``GravityWaveDragConfig`` from the
--
tests/unit/test_gwd_override.py-36-                            discretization="finite_volume"),
tests/unit/test_gwd_override.py-37-        radiation="gray",
tests/unit/test_gwd_override.py-38-        gravity_wave_drag=gwd,
tests/unit/test_gwd_override.py:39:        gravity_wave_drag_override=override,
tests/unit/test_gwd_override.py-40-    )
tests/unit/test_gwd_override.py-41-
tests/unit/test_gwd_override.py-42-
--
tests/unit/test_gwd_override.py-53-        mcfarlane=McFarlaneConfig(use_e3sm_hdsp=True),
tests/unit/test_gwd_override.py-54-    )
tests/unit/test_gwd_override.py-55-    pipe = _pipe("mcfarlane", over)
tests/unit/test_gwd_override.py:56:    assert pipe.gwd_config.use_e3sm_hdsp is True
tests/unit/test_gwd_override.py-57-
tests/unit/test_gwd_override.py-58-
tests/unit/test_gwd_override.py-59-def test_override_reaches_built_pipeline_e3sm_flag():
--
tests/unit/test_gwd_override.py-63-        e3sm_cam=E3SMCAMConfig(use_discrete_ke_heating=True),
tests/unit/test_gwd_override.py-64-    )
tests/unit/test_gwd_override.py-65-    pipe = _pipe("e3sm_cam", over)
tests/unit/test_gwd_override.py:66:    assert pipe.gwd_config.use_discrete_ke_heating is True
tests/unit/test_gwd_override.py-67-
tests/unit/test_gwd_override.py-68-
tests/unit/test_gwd_override.py-69-def test_override_none_is_default_byte_identical():
tests/unit/test_gwd_override.py-70-    """No override ⇒ exactly the scheme-string default config (legacy)."""
tests/unit/test_gwd_override.py-71-    pipe = _pipe("mcfarlane", None)
tests/unit/test_gwd_override.py:72:    assert pipe.gwd_config == GravityWaveDragConfig(scheme="mcfarlane").mcfarlane
tests/unit/test_gwd_override.py-73-
tests/unit/test_gwd_override.py-74-
tests/unit/test_gwd_override.py-75-def test_validate_strict_rejects_scheme_mismatch():
tests/unit/test_gwd_override.py-76-    over = GravityWaveDragConfig(scheme="lindzen")
tests/unit/test_gwd_override.py-77-    cfg = _config("mcfarlane", over)
tests/unit/test_gwd_override.py:78:    with pytest.raises(ValueError, match="gravity_wave_drag_override.scheme"):
tests/unit/test_gwd_override.py-79-        cfg.validate_strict()
tests/unit/test_gwd_override.py-80-
tests/unit/test_gwd_override.py-81-
--
tests/unit/test_gwd_config_for.py:1:"""``gwd_config_for`` — the third resolver (turbulence / convection / GWD).
tests/unit/test_gwd_config_for.py-2-
tests/unit/test_gwd_config_for.py-3-Every lane (FV pipeline, MPAS, spectral) previously built
tests/unit/test_gwd_config_for.py-4-``GravityWaveDragConfig(scheme=...)`` from the scheme STRING alone, so the
--
tests/unit/test_gwd_config_for.py-18-    GridConfig,
tests/unit/test_gwd_config_for.py-19-    OutputConfig,
tests/unit/test_gwd_config_for.py-20-)
tests/unit/test_gwd_config_for.py:21:from legoesm.driver.physics_pipeline import gwd_config_for
tests/unit/test_gwd_config_for.py-22-
tests/unit/test_gwd_config_for.py-23-
tests/unit/test_gwd_config_for.py-24-def _cfg(**kw):
--
tests/unit/test_gwd_config_for.py-40-    """Untouched scalars reproduce the bare config EXACTLY — whole tuple, every
tests/unit/test_gwd_config_for.py-41-    scheme.  Byte-identical defaults are the precondition for wiring the overlay
tests/unit/test_gwd_config_for.py-42-    into lanes that previously built the bare config."""
tests/unit/test_gwd_config_for.py:43:    assert (gwd_config_for(_cfg(gravity_wave_drag=scheme))
tests/unit/test_gwd_config_for.py-44-            == GravityWaveDragConfig(scheme=scheme))
tests/unit/test_gwd_config_for.py-45-
tests/unit/test_gwd_config_for.py-46-
tests/unit/test_gwd_config_for.py-47-def test_none_scheme_short_circuits():
tests/unit/test_gwd_config_for.py:48:    got = gwd_config_for(_cfg(gravity_wave_drag="none"))
tests/unit/test_gwd_config_for.py-49-    assert got.scheme == "none"
tests/unit/test_gwd_config_for.py-50-    assert got == GravityWaveDragConfig(scheme="none")
tests/unit/test_gwd_config_for.py-51-
tests/unit/test_gwd_config_for.py-52-
tests/unit/test_gwd_config_for.py-53-def test_mcfarlane_scalars_reach_the_leaf():
tests/unit/test_gwd_config_for.py:54:    got = gwd_config_for(_cfg(
tests/unit/test_gwd_config_for.py:55:        gravity_wave_drag="mcfarlane",
tests/unit/test_gwd_config_for.py:56:        mcfarlane_tau_max=4.0,
tests/unit/test_gwd_config_for.py-57-        mcfarlane_k_wave=1.0e-4,
tests/unit/test_gwd_config_for.py-58-        mcfarlane_directional_spread=0.5,
tests/unit/test_gwd_config_for.py-59-    ))
--
tests/unit/test_gwd_config_for.py-63-
tests/unit/test_gwd_config_for.py-64-
tests/unit/test_gwd_config_for.py-65-def test_hines_scalars_reach_the_leaf():
tests/unit/test_gwd_config_for.py:66:    got = gwd_config_for(_cfg(
tests/unit/test_gwd_config_for.py:67:        gravity_wave_drag="hines",
tests/unit/test_gwd_config_for.py:68:        hines_total_rms_wind=1.2,
tests/unit/test_gwd_config_for.py:69:        hines_Fmax=0.05,
tests/unit/test_gwd_config_for.py-70-    ))
tests/unit/test_gwd_config_for.py-71-    assert got.hines.total_rms_wind == 1.2
tests/unit/test_gwd_config_for.py-72-    assert got.hines.Fmax == 0.05
--
tests/unit/test_gwd_config_for.py-74-
tests/unit/test_gwd_config_for.py-75-def test_composite_scheme_carries_both_tuned_leaves():
tests/unit/test_gwd_config_for.py-76-    """'mcfarlane+hines' runs BOTH kernels — both overlays must apply."""
tests/unit/test_gwd_config_for.py:77:    got = gwd_config_for(_cfg(
tests/unit/test_gwd_config_for.py:78:        gravity_wave_drag="mcfarlane+hines",
tests/unit/test_gwd_config_for.py:79:        mcfarlane_tau_max=6.0,
tests/unit/test_gwd_config_for.py:80:        hines_total_rms_wind=1.5,
tests/unit/test_gwd_config_for.py-81-    ))
tests/unit/test_gwd_config_for.py-82-    assert got.scheme == "mcfarlane+hines"
tests/unit/test_gwd_config_for.py-83-    assert got.mcfarlane.tau_max == 6.0
--
tests/unit/test_gwd_config_for.py-88-    """An injected full config is never second-guessed by the overlay."""
tests/unit/test_gwd_config_for.py-89-    inj = GravityWaveDragConfig(scheme="mcfarlane")
tests/unit/test_gwd_config_for.py-90-    inj = inj._replace(mcfarlane=inj.mcfarlane._replace(tau_max=99.0))
tests/unit/test_gwd_config_for.py:91:    got = gwd_config_for(_cfg(
tests/unit/test_gwd_config_for.py:92:        gravity_wave_drag="mcfarlane",
tests/unit/test_gwd_config_for.py:93:        gravity_wave_drag_override=inj,
tests/unit/test_gwd_config_for.py:94:        mcfarlane_tau_max=4.0,   # must NOT win over the override
tests/unit/test_gwd_config_for.py-95-    ))
tests/unit/test_gwd_config_for.py-96-    assert got is inj
tests/unit/test_gwd_config_for.py-97-    assert got.mcfarlane.tau_max == 99.0
--
tests/unit/test_gwd_config_for.py-99-
tests/unit/test_gwd_config_for.py-100-def test_other_leaves_untouched():
tests/unit/test_gwd_config_for.py-101-    """Overlaying mcfarlane/hines leaves the other scheme leaves at defaults."""
tests/unit/test_gwd_config_for.py:102:    got = gwd_config_for(_cfg(gravity_wave_drag="mcfarlane",
tests/unit/test_gwd_config_for.py:103:                              mcfarlane_tau_max=4.0))
tests/unit/test_gwd_config_for.py-104-    bare = GravityWaveDragConfig(scheme="mcfarlane")
tests/unit/test_gwd_config_for.py-105-    assert got.rayleigh == bare.rayleigh
tests/unit/test_gwd_config_for.py-106-    assert got.lindzen == bare.lindzen
--
tests/unit/test_gwd_config_for.py-112-    """The FV pipeline path honours the same tuned leaf (no divergence)."""
tests/unit/test_gwd_config_for.py-113-    from legoesm.driver.physics_pipeline import _resolve_gwd
tests/unit/test_gwd_config_for.py-114-
tests/unit/test_gwd_config_for.py:115:    fn, kernel_cfg = _resolve_gwd(_cfg(gravity_wave_drag="mcfarlane",
tests/unit/test_gwd_config_for.py:116:                                       mcfarlane_tau_max=4.0))
tests/unit/test_gwd_config_for.py-117-    assert fn is not None
tests/unit/test_gwd_config_for.py-118-    assert kernel_cfg.tau_max == 4.0
tests/unit/test_gwd_config_for.py-119-
--
tests/unit/test_gwd_config_for.py-125-    parser = build_arg_parser()
tests/unit/test_gwd_config_for.py-126-    d = build_config_from_args(_postprocess_args(
tests/unit/test_gwd_config_for.py-127-        parser.parse_args(["--dataset", "analytical"]), parser))
tests/unit/test_gwd_config_for.py:128:    assert d.hines_total_rms_wind == 2.0
tests/unit/test_gwd_config_for.py:129:    assert d.hines_Fmax == 0.1
tests/unit/test_gwd_config_for.py:130:    assert d.mcfarlane_tau_max == 10.0
tests/unit/test_gwd_config_for.py-131-
tests/unit/test_gwd_config_for.py-132-    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
tests/unit/test_gwd_config_for.py-133-        "--dataset", "analytical",
tests/unit/test_gwd_config_for.py-134-        "--hines-total-rms-wind", "1.2", "--hines-fmax", "0.05",
tests/unit/test_gwd_config_for.py-135-        "--mcfarlane-tau-max", "4.0", "--mcfarlane-k-wave", "1.0e-4",
tests/unit/test_gwd_config_for.py-136-    ]), parser))
tests/unit/test_gwd_config_for.py:137:    assert cfg.hines_total_rms_wind == 1.2
tests/unit/test_gwd_config_for.py:138:    assert cfg.hines_Fmax == 0.05
tests/unit/test_gwd_config_for.py:139:    assert cfg.mcfarlane_tau_max == 4.0
tests/unit/test_gwd_config_for.py-140-    assert cfg.mcfarlane_k_wave == 1.0e-4
tests/unit/test_gwd_config_for.py:141:    assert gwd_config_for(cfg).hines.total_rms_wind == 1.2
tests/unit/test_gwd_config_for.py-142-
tests/unit/test_gwd_config_for.py-143-
tests/unit/test_gwd_config_for.py-144-def test_every_overlaid_scalar_has_a_run_amip_route():
tests/unit/test_gwd_config_for.py:145:    """Each scalar gwd_config_for threads must be settable by an operator.
tests/unit/test_gwd_config_for.py-146-    ``--config`` YAML keys must be argparse dests (load_yaml_config rejects
tests/unit/test_gwd_config_for.py-147-    anything else), so a flag is the gate for BOTH routes; the --params route
tests/unit/test_gwd_config_for.py-148-    additionally needs a _ATM_SCALAR_PARAM_MAP entry, which mcfarlane_k_wave
--
tests/unit/test_gwd_config_for.py-152-
tests/unit/test_gwd_config_for.py-153-    dests = {a.dest for a in build_arg_parser()._actions}
tests/unit/test_gwd_config_for.py-154-    overlaid = ("mcfarlane_k_wave", "mcfarlane_directional_spread",
tests/unit/test_gwd_config_for.py:155:                "mcfarlane_tau_max", "hines_total_rms_wind", "hines_Fmax")
tests/unit/test_gwd_config_for.py-156-    # mcfarlane_directional_spread is --params-only (mapped, no flag); every
tests/unit/test_gwd_config_for.py-157-    # other overlaid scalar must have a flag.
tests/unit/test_gwd_config_for.py-158-    amap = set(build_atm_scalar_param_map().values())
tests/unit/test_gwd_config_for.py-159-    for f in overlaid:
tests/unit/test_gwd_config_for.py-160-        assert f in dests or f in amap, (
tests/unit/test_gwd_config_for.py:161:            f"{f} is overlaid onto the GWD kernel by gwd_config_for but an "
tests/unit/test_gwd_config_for.py-162-            "operator cannot set it: no run_amip flag AND no scalar-map entry."
tests/unit/test_gwd_config_for.py-163-        )
tests/unit/test_gwd_config_for.py-164-
--
tests/unit/test_gwd_config_for.py-182-            n_azimuths=8, n_wavenumbers=6, launch_flux=2.5e-3),
tests/unit/test_gwd_config_for.py-183-    )
tests/unit/test_gwd_config_for.py-184-    cfg = _cfg(gravity_wave_drag="prognostic_spectral",
tests/unit/test_gwd_config_for.py:185:               gravity_wave_drag_override=inj)
tests/unit/test_gwd_config_for.py-186-    ps = init_physics_state(
tests/unit/test_gwd_config_for.py-187-        3, 8,
tests/unit/test_gwd_config_for.py-188-        PhysicsConfig(turbulence=TurbulenceConfig(scheme="none"),
tests/unit/test_gwd_config_for.py:189:                      gravity_wave_drag=gwd_config_for(cfg)),
tests/unit/test_gwd_config_for.py-190-    )
tests/unit/test_gwd_config_for.py-191-    assert ps.gwd_spectrum.shape == (3, 8, 6)
tests/unit/test_gwd_config_for.py-192-    assert float(ps.gwd_spectrum[0, 0, 0]) == 2.5e-3
tests/unit/test_gwd_config_for.py-193-
tests/unit/test_gwd_config_for.py-194-
tests/unit/test_gwd_config_for.py-195-@pytest.mark.parametrize("field,bad", [
tests/unit/test_gwd_config_for.py:196:    ("hines_total_rms_wind", -1.0),
tests/unit/test_gwd_config_for.py:197:    ("hines_total_rms_wind", 0.0),
tests/unit/test_gwd_config_for.py:198:    ("hines_Fmax", -0.1),
tests/unit/test_gwd_config_for.py:199:    ("mcfarlane_tau_max", float("nan")),
tests/unit/test_gwd_config_for.py-200-    ("mcfarlane_k_wave", float("inf")),
tests/unit/test_gwd_config_for.py-201-    ("mcfarlane_directional_spread", -1.0),
tests/unit/test_gwd_config_for.py-202-])
--
tests/unit/test_gwd_config_for.py-205-    every level (constant spurious drag, no error); a non-positive launch rms
tests/unit/test_gwd_config_for.py-206-    wind silently disables the scheme.  Neither may reach the kernel."""
tests/unit/test_gwd_config_for.py-207-    with pytest.raises(ValueError, match=field):
tests/unit/test_gwd_config_for.py:208:        _cfg(gravity_wave_drag="hines", **{field: bad}).validate_strict()
tests/unit/test_gwd_config_for.py-209-
tests/unit/test_gwd_config_for.py-210-
tests/unit/test_gwd_config_for.py-211-def test_legacy_amip_upconvert_keeps_the_exact_k_wave_default():
--
tests/unit/test_gwd_config_for.py-221-    # getattr fallback is what supplies k_wave.
tests/unit/test_gwd_config_for.py-222-    exp = ExperimentConfig.from_amip_config(AMIPExperimentConfig())
tests/unit/test_gwd_config_for.py-223-    assert exp.mcfarlane_k_wave == McFarlaneConfig().k_wave
tests/unit/test_gwd_config_for.py:224:    assert gwd_config_for(
tests/unit/test_gwd_config_for.py:225:        exp._replace(gravity_wave_drag="mcfarlane"),
tests/unit/test_gwd_config_for.py-226-    ).mcfarlane == McFarlaneConfig()
tests/unit/test_gwd_config_for.py-227-
tests/unit/test_gwd_config_for.py-228-
--
tests/unit/test_gwd_config_for.py-231-    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
tests/unit/test_gwd_config_for.py-232-        get_gwd_fn,
tests/unit/test_gwd_config_for.py-233-    )
tests/unit/test_gwd_config_for.py:234:    name, fn, _kcfg = get_gwd_fn(gwd_config_for(_cfg(
tests/unit/test_gwd_config_for.py:235:        gravity_wave_drag=scheme, hines_total_rms_wind=1.5,
tests/unit/test_gwd_config_for.py:236:        mcfarlane_tau_max=6.0)))
tests/unit/test_gwd_config_for.py-237-    assert fn is not None
tests/unit/test_gwd_config_for.py-238-    assert name
--
tests/unit/_params_reachability_baseline.py-182-    'atm.rad.GrayRadiationConfig.tau_moist_coeff',
tests/unit/_params_reachability_baseline.py-183-    'atm.rad.GrayRadiationConfig.tau_pole',
tests/unit/_params_reachability_baseline.py-184-    # atm: HinesConfig (0) — Fmax + total_rms_wind became reachable when
tests/unit/_params_reachability_baseline.py:185:    # gwd_config_for started threading the hines_* ExperimentConfig scalars
tests/unit/_params_reachability_baseline.py-186-    # into the kernel leaf on every lane (2026-07-24).
tests/unit/_params_reachability_baseline.py-187-    # atm: HoltslagBovilleConfig (13)
tests/unit/_params_reachability_baseline.py-188-    'atm.turb.HoltslagBovilleConfig.Ri_crit',
--
tests/unit/_params_reachability_baseline.py-254-    'atm.conv.MassFluxConfig.delta_0',
tests/unit/_params_reachability_baseline.py-255-    'atm.conv.MassFluxConfig.tau_adj',
tests/unit/_params_reachability_baseline.py-256-    # atm: McFarlaneConfig (6) — directional_spread became reachable via the
tests/unit/_params_reachability_baseline.py:257:    # mcfarlane_directional_spread scalar + gwd_config_for (2026-07-24); the
tests/unit/_params_reachability_baseline.py-258-    # rest still have no ExperimentConfig scalar to route through.
tests/unit/_params_reachability_baseline.py-259-    'atm.gwd.McFarlaneConfig.G_0',
tests/unit/_params_reachability_baseline.py-260-    'atm.gwd.McFarlaneConfig.efficiency',
--
tests/unit/test_physics_params_audit.py-448-        assert_grad_ok(lambda x: _gwd_call(mcfarlane_gwd,
tests/unit/test_physics_params_audit.py-449-            McFarlaneConfig()._replace(G_0=x)), 0.5, "mcfarlane.G_0")
tests/unit/test_physics_params_audit.py-450-
tests/unit/test_physics_params_audit.py:451:    def test_hines_total_rms_wind(self):
tests/unit/test_physics_params_audit.py-452-        from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
tests/unit/test_physics_params_audit.py-453-        from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
tests/unit/test_physics_params_audit.py-454-        assert_grad_ok(lambda x: _gwd_call(hines_gwd,
--
tests/unit/test_gwd_landfrac_wiring.py-153-        radiation="gray",
tests/unit/test_gwd_landfrac_wiring.py-154-        convection=convection,
tests/unit/test_gwd_landfrac_wiring.py-155-        gravity_wave_drag="e3sm_cam",
tests/unit/test_gwd_landfrac_wiring.py:156:        gravity_wave_drag_override=gwd_over,
tests/unit/test_gwd_landfrac_wiring.py-157-    )
tests/unit/test_gwd_landfrac_wiring.py-158-    cfg.validate_strict()
tests/unit/test_gwd_landfrac_wiring.py-159-    return grid, build_physics_pipeline(grid, sigma, cfg)
--
tests/unit/test_params_config_loader.py-171-        "P3Config": {"microphysics": "p3"},
tests/unit/test_params_config_loader.py-172-        "SeifertBehengConfig": {"microphysics": "seifert_beheng"},
tests/unit/test_params_config_loader.py-173-        "ThompsonConfig": {"microphysics": "thompson"},
tests/unit/test_params_config_loader.py:174:        # GWD: gwd_config_for overlays the mcfarlane_*/hines_* scalars onto the
tests/unit/test_params_config_loader.py:175:        # scheme leaf, and get_gwd_fn hands the pipeline that LEAF as gwd_config.
tests/unit/test_params_config_loader.py-176-        "HinesConfig": {"gravity_wave_drag": "hines"},
tests/unit/test_params_config_loader.py-177-        "McFarlaneConfig": {"gravity_wave_drag": "mcfarlane"},
tests/unit/test_params_config_loader.py-178-    }
tests/unit/test_params_config_loader.py-179-    resolved_attr = {
tests/unit/test_params_config_loader.py:180:        "HinesConfig": "gwd_config",
tests/unit/test_params_config_loader.py:181:        "McFarlaneConfig": "gwd_config",
tests/unit/test_params_config_loader.py-182-        "SBMConfig": "convection_config",
tests/unit/test_params_config_loader.py-183-        "BechtoldConfig": "convection_config",
tests/unit/test_params_config_loader.py-184-        "TiedtkeConfig": "convection_config",
--
tests/unit/test_params_config_loader.py-284-         lambda pipe, field: getattr(pipe.micro_config, field, None)),
tests/unit/test_params_config_loader.py-285-        ("ThompsonConfig", "", {"microphysics": "thompson"},
tests/unit/test_params_config_loader.py-286-         lambda pipe, field: getattr(pipe.micro_config, field, None)),
tests/unit/test_params_config_loader.py:287:        # GWD families: _resolve_gwd -> gwd_config_for overlays the flat
tests/unit/test_params_config_loader.py:288:        # scalars onto the scheme leaf, which get_gwd_fn returns as gwd_config.
tests/unit/test_params_config_loader.py-289-        ("HinesConfig", "hines_", {"gravity_wave_drag": "hines"},
tests/unit/test_params_config_loader.py:290:         lambda pipe, field: getattr(pipe.gwd_config, field, None)),
tests/unit/test_params_config_loader.py-291-        ("McFarlaneConfig", "mcfarlane_", {"gravity_wave_drag": "mcfarlane"},
tests/unit/test_params_config_loader.py:292:         lambda pipe, field: getattr(pipe.gwd_config, field, None)),
tests/unit/test_params_config_loader.py-293-    ]
tests/unit/test_params_config_loader.py-294-    # Companion drift-guard: the family list scanned below must exactly match
tests/unit/test_params_config_loader.py-295-    # the config classes present in the verified allowlist map.  The selector /
5808-        """
5809-        return day_to_calendar(self._insolation_day(day))
5810-
5811:    def _run_mpas(self, start_step: int = 0, start_day: float | None = None) -> str:
5812-        """Run MPAS model with the unified physics pipeline.
5813-
5814-        Uses the same physics pipeline as cubed-sphere/lat-lon, built
--
6021-                    * float(_r_nuc) ** 3
6022-        from legoesm.atmosphere.physics.radiation.solar import earth_orbit
6023-        _orbit_params = earth_orbit() if cfg.orbital_insolation else None
6024:        phys_cfg = PhysicsConfig(
6025-            radiation=RadiationConfig(
6026-                scheme=_rad_scheme if _rad_scheme != "none" else "none",
6027-                rrtmgp=RRTMGPConfig(
--
7311-                "compiled, and per-step loops)."
7312-            )
7313-
7314:    def _run_spectral(self, start_step: int = 0, start_day: float | None = None) -> str:
7315-        """Run spectral PE model with physics coupling.
7316-
7317-        Physics tendencies are computed on the Gaussian grid and converted
--
7544-                self._refuse_stateful_physics_unthreaded(cfg)
7545-            from legoesm.atmosphere.physics.radiation.solar import earth_orbit
7546-            _orbit_params = earth_orbit() if cfg.orbital_insolation else None
7547:            phys_cfg = PhysicsConfig(
7548-                radiation=RadiationConfig(
7549-                    scheme=_rad_scheme,
7550-                    rrtmgp=RRTMGPConfig(
--
9405-            _seed_dtype = None if _gwd_prognostic else _sd
9406-            _seed_ps = init_physics_state(
9407-                conv_ncol, _nlev,
9408:                PhysicsConfig(
9409-                    turbulence=TurbulenceConfig(scheme=cfg.turbulence),
9410-                    # MUST be the SAME resolved config the pipeline kernel gets
9411-                    # (_resolve_gwd -> gwd_config_for): init_physics_state sizes

exec
/usr/bin/bash -lc 'rg -n "mcfarlane_(k_wave|N_ref|directional_spread|tau_max)|hines_(Fmax|total_rms_wind)" packages/tools/legoesm/forcing/amip_config.py packages/coupler/legoesm/driver/config.py && rg -n -C 2 "def get_gwd_fn|return.*gwd_config|gwd_config" packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/integration.py && git diff --check && git diff -- tests/unit/golden/yaml_to_experiment/williamson_test2.json tests/unit/golden/yaml_to_experiment/williamson_test5.json' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 588ms:
packages/coupler/legoesm/driver/config.py:989:    mcfarlane_k_wave: float = 2.0 * math.pi / 100e3  # McFarlaneConfig.k_wave [1/m]
packages/coupler/legoesm/driver/config.py:995:    mcfarlane_N_ref: float = 0.01               # INERT (no leaf field)
packages/coupler/legoesm/driver/config.py:996:    mcfarlane_directional_spread: float = 1.0   # McFarlaneConfig.directional_spread
packages/coupler/legoesm/driver/config.py:997:    mcfarlane_tau_max: float = 10.0             # McFarlaneConfig.tau_max [Pa]
packages/coupler/legoesm/driver/config.py:1281:    hines_total_rms_wind: float = 2.0           # HinesConfig.total_rms_wind [m/s]
packages/coupler/legoesm/driver/config.py:1282:    hines_Fmax: float = 0.1                     # HinesConfig.Fmax [Pa]
packages/coupler/legoesm/driver/config.py:2084:        # failure modes without this guard: ``hines_Fmax < 0`` makes
packages/coupler/legoesm/driver/config.py:2086:        # (constant spurious drag, no error), and ``hines_total_rms_wind <= 0``
packages/coupler/legoesm/driver/config.py:2090:        for _f in ("mcfarlane_k_wave", "mcfarlane_directional_spread",
packages/coupler/legoesm/driver/config.py:2091:                   "mcfarlane_tau_max", "hines_total_rms_wind", "hines_Fmax"):
packages/coupler/legoesm/driver/config.py:2486:            mcfarlane_k_wave=getattr(
packages/coupler/legoesm/driver/config.py:2487:                amip_cfg, 'mcfarlane_k_wave',
packages/coupler/legoesm/driver/config.py:2488:                ExperimentConfig._field_defaults['mcfarlane_k_wave']),
packages/coupler/legoesm/driver/config.py:2489:            mcfarlane_N_ref=getattr(amip_cfg, 'mcfarlane_N_ref', 0.01),
packages/coupler/legoesm/driver/config.py:2490:            mcfarlane_directional_spread=getattr(amip_cfg, 'mcfarlane_directional_spread', 1.0),
packages/coupler/legoesm/driver/config.py:2491:            mcfarlane_tau_max=getattr(amip_cfg, 'mcfarlane_tau_max', 10.0),
packages/coupler/legoesm/driver/config.py:2664:            mcfarlane_k_wave=getattr(
packages/coupler/legoesm/driver/config.py:2665:                self, 'mcfarlane_k_wave',
packages/coupler/legoesm/driver/config.py:2666:                ExperimentConfig._field_defaults['mcfarlane_k_wave']),
packages/coupler/legoesm/driver/config.py:2667:            mcfarlane_N_ref=getattr(self, 'mcfarlane_N_ref', 0.01),
packages/coupler/legoesm/driver/config.py:2668:            mcfarlane_directional_spread=getattr(self, 'mcfarlane_directional_spread', 1.0),
packages/coupler/legoesm/driver/config.py:2669:            mcfarlane_tau_max=getattr(self, 'mcfarlane_tau_max', 10.0),
67-
68-
69:def get_gwd_fn(config: GravityWaveDragConfig):
70-    """Select the GWD backend based on config.scheme."""
71-    if config.scheme == "rayleigh":
--
251-
252-def make_gwd_physics(
253:    gwd_config: GravityWaveDragConfig,
254-    model_type: str = "hydrostatic",
255-    dt: float = 300.0,  # coeff-ok: default physics timestep [s]
--
259-    Parameters
260-    ----------
261:    gwd_config : GravityWaveDragConfig
262-        GWD configuration (selects scheme).
263-    model_type : str
--
272-    """
273-    if model_type == "hydrostatic":
274:        return _make_hydrostatic_gwd(gwd_config, dt)
275-    elif model_type == "nonhydrostatic":
276:        return _make_nonhydrostatic_gwd(gwd_config, dt)
277-    elif model_type == "spectral_pe":
278:        return _make_spectral_pe_gwd(gwd_config, dt)
279-    elif model_type == "mpas":
280:        return _make_mpas_gwd(gwd_config, dt)
281-    else:
282-        raise ValueError(
--
291-
292-def _make_hydrostatic_gwd(
293:    gwd_config: GravityWaveDragConfig,
294-    dt: float,
295-) -> Callable:
--
303-    returned as the second element of the result tuple.
304-    """
305:    scheme_name, gwd_fn, scheme_config = get_gwd_fn(gwd_config)
306-    is_prognostic = scheme_name == "prognostic_spectral"
307-    is_ml = scheme_name == "ml_emulator"
--
523-
524-def _make_mpas_gwd(
525:    gwd_config: GravityWaveDragConfig,
526-    dt: float,
527-) -> Callable:
--
539-    at runtime if such a scheme is selected.
540-    """
541:    scheme_name, gwd_fn, scheme_config = get_gwd_fn(gwd_config)
542-    is_prognostic = scheme_name == "prognostic_spectral"
543-    is_ml = scheme_name == "ml_emulator"
--
674-
675-def _make_nonhydrostatic_gwd(
676:    gwd_config: GravityWaveDragConfig,
677-    dt: float,
678-) -> Callable:
--
682-               -> NonHydrostaticTendencies
683-    """
684:    scheme_name, gwd_fn, scheme_config = get_gwd_fn(gwd_config)
685-    is_prognostic = scheme_name == "prognostic_spectral"
686-    is_ml = scheme_name == "ml_emulator"
--
841-
842-def _make_spectral_pe_gwd(
843:    gwd_config: GravityWaveDragConfig,
844-    dt: float,
845-) -> Callable:
--
849-               -> SpectralHydrostaticState
850-    """
851:    scheme_name, gwd_fn, scheme_config = get_gwd_fn(gwd_config)
852-    is_prognostic = scheme_name == "prognostic_spectral"
853-    is_ml = scheme_name == "ml_emulator"
diff --git a/tests/unit/golden/yaml_to_experiment/williamson_test2.json b/tests/unit/golden/yaml_to_experiment/williamson_test2.json
index 5f63fdac4..7873bb840 100644
--- a/tests/unit/golden/yaml_to_experiment/williamson_test2.json
+++ b/tests/unit/golden/yaml_to_experiment/williamson_test2.json
@@ -110,6 +110,8 @@
     "vertical_coord": "none"
   },
   "held_suarez_forcing": false,
+  "hines_Fmax": 0.1,
+  "hines_total_rms_wind": 2.0,
   "ic": "default",
   "ic_path": "",
   "insolation_start_doy": null,
@@ -146,7 +148,7 @@
   "lw_diff_factor": 1.66,
   "mcfarlane_N_ref": 0.01,
   "mcfarlane_directional_spread": 1.0,
-  "mcfarlane_k_wave": 6.283185307e-05,
+  "mcfarlane_k_wave": 6.283185307179586e-05,
   "mcfarlane_tau_max": 10.0,
   "micro_substeps": 1,
   "microphysics": "none",
diff --git a/tests/unit/golden/yaml_to_experiment/williamson_test5.json b/tests/unit/golden/yaml_to_experiment/williamson_test5.json
index 1d7fcab02..f60f049f9 100644
--- a/tests/unit/golden/yaml_to_experiment/williamson_test5.json
+++ b/tests/unit/golden/yaml_to_experiment/williamson_test5.json
@@ -110,6 +110,8 @@
     "vertical_coord": "none"
   },
   "held_suarez_forcing": false,
+  "hines_Fmax": 0.1,
+  "hines_total_rms_wind": 2.0,
   "ic": "default",
   "ic_path": "",
   "insolation_start_doy": null,
@@ -146,7 +148,7 @@
   "lw_diff_factor": 1.66,
   "mcfarlane_N_ref": 0.01,
   "mcfarlane_directional_spread": 1.0,
-  "mcfarlane_k_wave": 6.283185307e-05,
+  "mcfarlane_k_wave": 6.283185307179586e-05,
   "mcfarlane_tau_max": 10.0,
   "micro_substeps": 1,
   "microphysics": "none",

codex
Static verification is complete: all three runtime construction sites now use the resolver; the resolver preserves defaults exactly and delivers the active leaf chosen by `get_gwd_fn`. The local image lacks `pytest`, so I can’t independently rerun the reported suite here.
codex
No substantive findings.

- R2-1 is fixed: all five live overlays now have an operator route; `k_wave` is reachable by CLI/YAML, while `directional_spread` is reachable through `--params`.
- R2-3 is fixed: the only golden changes are the two new Hines defaults and exact `k_wave` precision.
- R2-2 is not a production drop path. `to_amip_config()` has no non-test caller; checkpoints serialize native `ExperimentConfig` scalars and restore them by name.
- R1-4 is not blocking: missing `k_wave` param-spec bounds only excludes `--params`; the flag/YAML path is live, validated positive/finite, and reaches the kernel.
- The override serialization issue is pre-existing and loud, not newly made silent by this resolver. A restored dict fails `validate_strict` before `ModelDriver` runs; `_resolve_gwd` already honored overrides verbatim before this diff. Track its serialization fix separately.

The resolver uses static Python floats and preserves bare-config defaults exactly, so I found no JIT or kernel-value regression.

Minor test-scope note: `test_every_overlaid_scalar_has_a_run_amip_route` hard-codes the current five fields, so it guards regressions of those fields but cannot automatically discover a future sixth overlay unless its list is updated. That is not a blocker for this change. I could not rerun pytest locally because it is unavailable in this environment.
tokens used
84,923
No substantive findings.

- R2-1 is fixed: all five live overlays now have an operator route; `k_wave` is reachable by CLI/YAML, while `directional_spread` is reachable through `--params`.
- R2-3 is fixed: the only golden changes are the two new Hines defaults and exact `k_wave` precision.
- R2-2 is not a production drop path. `to_amip_config()` has no non-test caller; checkpoints serialize native `ExperimentConfig` scalars and restore them by name.
- R1-4 is not blocking: missing `k_wave` param-spec bounds only excludes `--params`; the flag/YAML path is live, validated positive/finite, and reaches the kernel.
- The override serialization issue is pre-existing and loud, not newly made silent by this resolver. A restored dict fails `validate_strict` before `ModelDriver` runs; `_resolve_gwd` already honored overrides verbatim before this diff. Track its serialization fix separately.

The resolver uses static Python floats and preserves bare-config defaults exactly, so I found no JIT or kernel-value regression.

Minor test-scope note: `test_every_overlaid_scalar_has_a_run_amip_route` hard-codes the current five fields, so it guards regressions of those fields but cannot automatically discover a future sixth overlay unless its list is updated. That is not a blocker for this change. I could not rerun pytest locally because it is unavailable in this environment.
EXIT=0
