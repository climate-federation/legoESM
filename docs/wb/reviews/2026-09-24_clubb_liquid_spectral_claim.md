# CLAIM REVIEW (before code): CLUBB's liquid partition on the spectral training lane

Worktree: this directory (branch wb/cam6-baseline, just merged origin/main). No code written yet.
Be adversarial: attack the premise, the mechanism, the conservation argument, the scope.

## Situation
Owner directive 2026-09-24: the WeatherBench spectral training deck
(config/wb/campaign/spectral_t63_cam6.yaml) must be CONSISTENT WITH THE MPAS PRODUCTION DECK
(config/amip/amip_production.yaml) on three settings, which production now carries:
  1. clubb_liquid_partition: true   (CLUBB's diagnosed liquid handed to the host cloud water)
  2. hard_saturation_adjustment: false  (no condensation step after the physics; CAM6 clubb_do_liqsupersat=.false.)
  3. hard_sat_ice_curve: false
(2) and (3) are deck rows and are already flipped on the WB deck (3 was never set; the spectral
lane also has no post-step ice curve). (1) is BLOCKED BY CODE on this lane, twice:
 - turbulence/integration.py:~545 raises NotImplementedError for any non-MPAS model_type:
   "liquid_partition returns a cloud-liquid tendency (TurbulenceOutput.dq_c_dt) that the
   'spectral_pe' turbulence lane does not route to a condensate tracer, so the liquid the closure
   removed from vapour would be destroyed."
 - combined.py:251 refuses cld_macmic_num_steps>1 outside ("hydrostatic","mpas"); combined.py:281
   requires cld_macmic_num_steps>=2 whenever the partition is on, because the closure REPLACES the
   host cloud water and the microphysics must run on the replaced value; the spectral combined
   path (_make_spectral_pe_combined, ~L1385) evaluates modules in PARALLEL on one state and sums
   tendencies, while the macmic sub-cycle lives only in _make_hydrostatic_combined (~L606).

## THE CLAIM
C1. Piece A (small): the spectral turbulence bridge routes TurbulenceOutput.dq_c_dt into the q_c
    tracer tendency exactly as the MPAS bridge does (turbulence/integration.py ~L847/~L1022),
    refusing a state with no q_c tracer. Vapour tendency already has that liquid removed, so water
    is conserved; no new sign, no clip.
C2. Piece B (larger): the ordering requirement is REAL (autoconversion ~ q_c^2.5 on pre-exchange
    liquid), so the partition cannot ship on the spectral lane by merely lifting the gate. Either
    the macmic sequential sub-cycle is ported to the spectral combined path (apply each module's
    increment to the spectral state between modules -- extra spectral<->grid transforms per
    sub-step, and the WB gradient's memory arena is ALREADY 643 GiB on one GPU before this), or
    the partition stays refused on this lane and the WB deck documents it as an open departure.
C3. Scope: nothing about the MPAS/hydrostatic lanes changes; the gate at combined.py:281 keeps its
    meaning; (2) and (3) are already consistent.

## Questions
1. Is C2's ordering requirement really binding on the spectral lane, or is there a cheaper
   faithful ordering (e.g. evaluating turbulence first and feeding its updated q_c only to the
   microphysics within the existing parallel accumulate)?
2. What is the memory and compile cost of a sequential sub-cycle inside a rollout step that is
   already remat'd with nothing_saveable, at T63/L32?
3. Does piece A alone (with the partition still refused) have any value, or is it dead code?
4. What must be MEASURED before any of this lands?
Verdict SHIP/HOLD first, then numbered findings with file:line.
