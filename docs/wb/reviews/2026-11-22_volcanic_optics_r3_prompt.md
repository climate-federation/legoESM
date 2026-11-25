# r3 (GLM only): your r2 HOLD was finding 1 (a simple_lw pipeline builder could drop aerosol_sw_volc_col).
Evidence (read from code, worktree /glade/derecho/scratch/pg2328/wt_volc):
- packages/coupler/legoesm/driver/physics_pipeline.py _RADIATION_BUILDERS = {"ecrad_ifs", "none", "gray",
  "rrtmgp", "rrtmg"}; there is NO simple_lw pipeline builder. none/gray radiation_fn raise ValueError on a
  non-None aerosol_sw_volc_col (tested); ecrad_ifs radiation_fn(*args, **kwargs) raises ValueError on ANY call
  ("not wired into the compiled physics pipeline"); rrtmgp/rrtmg consume it.
- simple_lw exists only in integration._call_radiation_backend, where the new guard raises before the
  simple_lw/gray returns (tested for both). The driver maps cfg.radiation "rrtmg" -> RadiationConfig.scheme
  "rrtmgp" (model_driver.py:13610, :18141), so production is not caught by that guard.
- finding 4 applied (loaded[:5]). finding 3 (negative g_sun raises): kept -- vote author raise / codex no
  defect (approved) / you clip; production file min g_sun = 0.0149 (measured).
Is finding 1 resolved? Any remaining BLOCKER/MAJOR? Reply one paragraph + VERDICT (APPROVE / HOLD).
