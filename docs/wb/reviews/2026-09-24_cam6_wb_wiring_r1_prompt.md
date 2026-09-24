Adversarial review of a DIFF (uncommitted) in worktree
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6, branch wb/cam6-baseline,
on top of 938855bde (feat/cam6-suite 6323e04ef + the CAM6 WB deck).

WHAT IT DOES. The AMIP production physics (CAM6 suite) selects
clouds=cam6_clubb, which takes its liquid cloud fraction from the CLUBB PDF
carry and its deep-convective part from Zhang-McFarlane's published interface
mass flux + in-cloud water, all read from the lagged PhysicsState. On the
hydrostatic/MPAS lanes this READ side existed; on the spectral lane (the
WeatherBench training core, model_type="spectral_pe") only the WRITE side did,
so cam6_clubb raised at the first radiation trace. Your own earlier review
(docs/wb/reviews/2026-09-24_cam6_wb_codex.md) named this the blocker and
named the three missing pieces. This diff adds them, mirroring the hydrostatic
code path piece for piece:

 1. radiation/integration.py: _make_spectral_pe_radiation gains
    use_clubb_cloud_fraction; same static cam6 gate as the hydrostatic builder
    (cam6_clubb without the flag -> ValueError at build); _physics_fn_core and
    the jax.checkpoint wrapper gain phys_state=None; reads
    phys_state.cloud_fraction / conv_mass_flux_up / conv_icwmr (lagged carry)
    and passes them to _call_radiation_backend; sets _wants_phys_state_ro when
    the flag is on. make_radiation_physics forwards the flag for spectral_pe
    and the lane refusal now admits spectral_pe.
 2. combined.py _make_spectral_pe_combined: the producer gate
    (use_clubb_cloud_fraction requires turbulence=clubb, copied from the
    hydrostatic combined) and forwarding of phys_state to any module that
    advertises _wants_phys_state_ro, in both dispatcher branches.
 3. convection/integration.py spectral bridge: after the scheme dispatch,
    publish conv_out.mass_flux_up (ncol, nlev+1) and conv_out.icwmr into the
    carry dict, with the both-or-neither ValueError, exactly as the
    hydrostatic bridge does at its lines ~892-908.
 4. aimip_params.py: RadiationConfig(use_clubb_cloud_fraction=(cloud_scheme ==
    "cam6_clubb")) in both the rrtmgp and gray branches -- DERIVED, not a knob,
    because the radiation builder refuses cam6_clubb without it; the split-rad
    rad_only_raw gets the flag; rad_fn gains phys_state=None and forwards it.
 5. neural_gcm_spectral.py spectral_rollout: _call_rad/_rad_refresh accept
    phys_state and pass it only when the rad fn's signature has it; the
    stateful gated step hands radiation the carry ENTERING the step (lagged);
    the initial radiation call is DEFERRED into a closure so the stateful
    branch seeds phys0 first and hands it in (the old eager call at function
    top ran before any carry existed and would now raise for cam6_clubb).

TESTS: tests/unit/test_spectral_cam6_cloud_routing.py, 7 cases. All 7 FAIL
on the pre-wiring commit 938855bde (run in a separate worktree); on the wired
tree they pass (result pasted at launch). The spy tests bypass jax.checkpoint
via monkeypatch so they can assert VALUES, not tracer shapes.

ATTACK THESE, each separately, MEASURED vs READ-OFF-THE-CODE:
 A. Lag convention. Radiation reads the carry entering the step; CLUBB and ZM
    write the carry during the same step's non-rad physics; the rollout's
    stateful step calls _ps_entry once per step on the pre-step state. Is the
    lag exactly one step as on the hydrostatic lane, or zero/two? Does the
    3-hourly radiation cadence (rad_update_interval=6) change that reasoning?
 B. Is there any path where use_clubb_cloud_fraction=True but the carry's
    cloud_fraction is still its ZERO initialisation when radiation reads it
    (first step, seed path, spectral_amip_rollout)? Zero would silently clear
    every cloud -- the hazard the producer gate exists for. Note
    spectral_amip_rollout (inference/finetune scripts, not WB training) does
    NOT thread phys_state to rad_fn, so cam6_clubb there now RAISES (loud).
    I left it. Right call?
 C. jax.checkpoint(_physics_fn_core, static_argnums=(1,2)) now receives
    phys_state as a 7th positional arg, a NamedTuple pytree of arrays with
    None fields. Any tracing/None-leaf/static-arg hazard? Under
    eqx.filter_value_and_grad, is the gradient path through
    cloud_fraction_override into RRTMGP intact (the CLUBB carry must stay
    differentiable for training)?
 D. combined.py: the ro-forwarding rebuilds _fwd0/_fwd dicts; any case where a
    fn advertises _wants_phys_state_ro but its signature lacks phys_state?
 E. The convection publication sits after the whole dispatch chain guarded by
    `conv_fn is not None`; conv_out is defined in every branch? Any branch
    where conv_out lacks mass_flux_up/icwmr attributes (older ConvectionOutput
    producers)? Shapes: ZM publishes mass_flux_up (ncol, nlev+1) -- is the
    reshape correct for the spectral column ordering?
 F. Anything in the diff that is a scientific CHOICE rather than a mechanical
    mirror of the hydrostatic path? Name it; it must be asked, not shipped.
 G. Dispatch-hardening / ratchets: the lane refusal message changed and the
    spectral_pe membership widened. Which tests/baselines pin that?

Read the actual files (git diff, and the hydrostatic reference sites). Rank
by decision impact. Say SHIP or HOLD.

TEST RESULT AT LAUNCH: wired tree 7 passed; touched regression suites (cam6 cloud fraction, clubb cf routing, spectral clubb carry, prescribed-surface radiation, spectral surface/moisture, radiation number coupling, orbital rrtmgp, ZM, dispatch-hardening, private-import ratchet) 125 passed / 1 deselected slow.
