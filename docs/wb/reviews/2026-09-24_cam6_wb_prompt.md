Adversarial review of a CLAIM and a new DECK, before any run is launched.
Repo legoESM. Worktree /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6, branch
wb/cam6-baseline, based on feat/cam6-suite @ 6323e04ef (PR #1787).

TASK THE OWNER GAVE: "import our AMIP CAM6 configuration, use this as our
baseline for WeatherBench training, train the parameters of the model within
our WeatherBench pipeline."

WHAT I DID: wrote config/wb/campaign/spectral_t63_cam6.yaml from the existing
config/wb/campaign/spectral_t63_amip.yaml, moving the rows that can cross from
config/amip/amip_production.yaml (which IS the CAM6 suite since 2026-09-23).
Differences from the previous WB AMIP deck: convection bechtold ->
zhang_mcfarlane; cloud sundqvist -> cam6_clubb; gwd mcfarlane+e3sm_cam ->
mcfarlane (CAM6 run 1 is orographic only); clubb_top_press_hpa 150.0 added;
CloudConfig.saturation_scheme mixed_phase; MorrisonConfig sed_cfl_substeps
True/max 256/strict False; the Bechtold and E3SM gravity-wave pinned blocks
dropped because their schemes are gone.

CLAIMS TO ATTACK, each separately, label MEASURED vs READ-OFF-THE-CODE:

(C1) BLOCKER 1, memory. The PREVIOUS and LIGHTER WB AMIP arm already dies at
the first gradient: "RESOURCE_EXHAUSTED: Out of memory while trying to allocate
375.97GiB" on an 80 GiB A100 (job 27624529). Compile-only measurement gave
490.7 GiB temp. Radiation column blocking was tried and REVERTED (PR #1780):
largest single tensor 17.9 -> 0.86 GiB, total 490.7 -> 489.1 GiB, 0.3%. The
consumer is UNIDENTIFIED. CAM6 adds ZM convection and CAM6 macrophysics, so
I claim it is strictly heavier and will not fit either. Is that sound? Is there
anything about ZM or cam6_clubb that would make it LIGHTER?

(C2) BLOCKER 2, stability, and this is the one I most want attacked.
CAM6 in the AMIP campaign COLD-COLLAPSED to 51 K with hydrometeor pile-up when
Morrison ran one 1800 s call; the adopted fix was CAM-faithful sub-cycling of
CLUBB+Morrison at 600 s (cld_macmic_num_steps=3) plus sedimentation CFL
sub-stepping. The WB training lane runs dt=1800 s. I read
packages/atmosphere/legoesm/atmosphere/physics/combined.py:218 as REFUSING
cld_macmic_num_steps>1 unless model_type is "hydrostatic" or "mpas", and
aimip_params.py:1562/1631 builds the WB lane with model_type="spectral_pe".
So the macmic sub-cycle CANNOT cross, and the deck above therefore runs CAM6
physics at 1800 s WITHOUT the lever that stopped it collapsing. Verify or
refute. If true, is sed_cfl_substeps alone enough, and what is the cheapest
measurement that would settle it?

(C3) The deck itself. Check every row I moved and every row I did NOT move:
  - is `cloud: cam6_clubb` reachable at all on the spectral lane? The WB
    builder's else-branch builds CloudConfig(scheme=cloud_scheme); the AMIP
    driver additionally sets use_clubb_cloud_fraction, which has NO route in
    the WB deck schema. Does cam6_clubb silently degrade without it?
  - `convective_precip_efficiency: 0.8` is in the CAM6 AMIP deck but
    ZhangMcFarlaneConfig has NO precip_efficiency field (its fields are
    c0_lnd c0_ocn ke momcu momcd num_cin tau capelmt tiedke_add dmpdz alfa
    limcnv_p_pa pbl_top_pa parcel_tpert enable_cmt). Where does the AMIP driver
    apply it for ZM, and is dropping it from the WB deck correct or a silent
    departure?
  - the existing WB AMIP deck pins
    MorrisonConfig.homogeneous_ice_nucleation, but the field list I read is
    homogeneous_ice_supersaturation. Is that pin INERT, i.e. has that deck been
    carrying a row that reaches nothing?
  - CAM6 uses radiation rrtmg and surface_bulk large_yeager_cesm; the WB lane
    keeps rrtmgp and `constant`. Departure or blocker?
  - what else in amip_production.yaml is a SCIENTIFIC choice that I silently
    dropped rather than deliberately excluded? The structurally impossible ones
    I already know: vertical_coord cam_l32 (WB is sigma T63/L32), every mpas_*
    row, dt 112.5 / physics_update_steps 16 / rad_update_steps 32, all
    land/ocean/ice rows (the WB surface is prescribed).

(C4) Training. The owner wants the parameters TRAINED in the WB pipeline. The
previous classical arm had a nearly FLAT objective (1.1% over three epochs) and
110 of 157 parameters had zero gradient. Under ZM + cam6_clubb the trainable
set changes entirely. Which of the new schemes' parameters are even reachable
by the loss, and is there anything structurally non-differentiable in the ZM
port (it has integer sub-step counts, a qmin floor added 2026-09-22, and a
limiter) that would zero the gradients?

Read the actual files. Rank findings by whether they change the decision.
Say explicitly if you think this task cannot proceed as asked.
