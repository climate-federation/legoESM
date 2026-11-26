"""NEMO ORCA1 ``&namzdf_tke`` card as a :class:`TKEConfig` (shared by the
OMIP drivers and the coupled MPAS ocean)."""
from __future__ import annotations

import numpy as np

# Single source of truth for --tke-mxl-choice, consumed by BOTH argparse
# (choices=) and orca1_zdftke_config's guard.  Codex 2026-08-07 #4: a finite
# -2..9 sweep in the test is a nearby-mutation check, NOT set equality -- it
# still passes if the builder accepts 10 while argparse does not.  Sharing one
# constant makes divergence impossible by construction instead of policed.
TKE_MXL_CHOICES = (2, 3, 4)   # 2=Veros BL, 3=NEMO nn_mxl=3, 4=NEMO nn_mxl=2


def orca1_zdftke_config(iwm_enabled: bool = False, surface_bc: str | None = None,
                        surface_bc_level: str | None = None,
                        mxl_choice: int | None = None,
                        n2_mode: str | None = None,
                        n2_eos_form: str | None = None,
                        prognostic: bool | None = None,
                        kappa_convention: str | None = None,
                        shear_production: str | None = None,
                        lc: bool | None = None,
                        etau_mode: str | None = None,
                        preclosure_coeff_source: str | None = None,
                        buoyancy_sink: str | None = None,
                        step_evaluation: str | None = None,
                        langmuir_rhs_coupling: str | None = None,
                        shear_avm_weighting: str | None = None,
                        prandtl_mode: str | None = None):
    """NEMO ORCA1 ``&namzdf_tke`` mapped onto :class:`TKEConfig`, value by value.

    Source of truth: ``cfgs/ORCA1/EXP00/RUN_REF/namelist_cfg`` overrides on top
    of ``namelist_ref`` defaults (NEMO 5.0.1).  Structural conventions
    started from the validated DINO NEMO-faithful recipe
    (``ocean/experiments/dino.py::_dino_vertical_mixing_config``); ONLY the
    namelist values change for ORCA1.

    Mapping (namelist -> field):
      rn_ediff = 0.1   -> c_k          (avm = c·mxl·sqrt(e); see amplitude note)
      rn_ediss = 0.7   -> c_eps        (Kolmogorov dissipation coefficient)
      rn_emin  = 1e-6  -> tke_background   [m²/s²]
      rn_emin0 = 1e-4  -> tke_surface_min  [m²/s²]
      nn_pdl   = 1     -> prandtl_mode="richardson" with prandtl_ri_coeff =
                          1/ri_cri, ri_cri = 2/(2 + rn_ediss/rn_ediff) = 2/9
                          (zdftke.F90:772,399: Pr = clamp(Ri/ri_cri, 1, 10) ==
                          clamp(4.5·Ri, 1, 10) — exactly _prandtl_number's form)
      nn_mxl   = 2     -> tke_mxl_choice=4 (NOT 2: the choice numbering is
                          Veros-derived, so choice 2 is Veros Bougeault-
                          Lacarrere and choice 4 is NEMO nn_mxl=2)
      ln_lc    = T     -> lc=True
      rn_lc    = 0.25  -> lc_coeff       (namelist_cfg override of 0.15)
      nn_etau  = 1     -> etau_mode="below_ml"
      rn_efr   = 0.08  -> etau_frac      (namelist_cfg override of 0.05)
      nn_htau  = 1     -> etau_htau_mode="latitude" (0.5–30 m, 45·|sinφ|)
      rn_ebb   = 67.83 -> tke.py module constant _NEMO_TKE_EBB (etau e_sfc;
                          identical value, no config field)
      &namzdf rn_avm0/rn_avt0 (1.2e-4/1.2e-5 backgrounds) -> kappaM_min /
                          kappaH_min.  NEMO composes avm = max(closure, avmb)
                          and avt = max(pdl·avt, avtb) (zdftke.F90:715,723);
                          compute_K_from_tke applies EXACTLY that max via the
                          kappaM_min/kappaH_min floors, so the namzdf
                          backgrounds live INSIDE the closure and build_tripole
                          forces the model-level additive A_v/K_v to molecular
                          (residual +1.4e-6/+1e-10 add — negligible, flagged).
                          With --iwm, NEMO's zdfiwm_init forces avmb/avtb to
                          molecular (rnu=1.4e-6, 1e-10): ``iwm_enabled=True``
                          switches the floors to those values so the wave
                          field is the interior background, as in NEMO.
      nn_avb   = 0     -> bg_diff_scale=0.0 (no Bryan-Lewis depth profile;
                          abyssal mixing comes from zdfiwm as in NEMO)

    nn_eice  = 3     -> eice=3 (under-ice attenuation of lc/etau: the
                          kernels' (1-ice_frac) factor fed max(0,1-4*fi) via
                          surface_forcing.ice_concentration — CLOSED
                          2026-07-18; was a flagged no-ice_frac gap).

    NO TKEConfig counterpart (fidelity gaps, flagged not stubbed):
      ln_mxl0=T / rn_mxl0=0.04  surface mixing length = F(wind stress);
      nn_mxlice=2               under-ice mixing-length scaling;
      rn_bshear=1e-20           background-shear floor (legoESM uses 1e-12).
    The card body now defaults to the NEMO Dirichlet surface TKE
    (``surface_bc="nemo_dirichlet"``) and the prognostic ``en`` carry
    (``prognostic=True``); both were listed here as gaps before those arms
    existed.
    """
    from legoesm import constants as _const
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

    rn_ediff = 0.1        # namelist_ref &namzdf_tke default (ORCA1 keeps it)
    rn_ediss = 0.7        # namelist_ref &namzdf_tke default (ORCA1 keeps it)
    # zdftke.F90:772 — ri_cri deduced from rn_ediff/rn_ediss; Pr slope = 1/ri_cri.
    pr_ri_slope = (2.0 + rn_ediss / rn_ediff) / 2.0     # = 4.5
    if iwm_enabled:
        # zdfiwm_init: avmb = rnu = 1.4e-6 m²/s, avtb = 1e-10 m²/s — the
        # wave field IS the interior background (matches the model-level
        # A_v/K_v override in the build_tripole iwm block).
        avmb, avtb = _const.nu_ocean_molecular, 1.0e-10
        # zdftke.F90:840-843 — ln_zdfiwm ALSO forces the TKE and mixing-length
        # FLOORS, overriding &namzdf_tke: rn_emin 1e-6 -> 1e-10 and rmxl_min ->
        # 1e-3 (associated avt minimum = molecular salt diffusivity 1e-9). ORCA1
        # runs ln_zdfiwm=T, so these are the run's real floors. Without them our
        # equatorial en pins at rn_emin=1e-6, ~100x NEMO's ~1e-8, over-mixing the
        # 65-105 m entrainment zone (avm 15x) and damping the EUC -> warm cold
        # tongue. Localised by the frozen-column TKE twin + state-swap
        # decomposition (2026-08-27); NEMO's avm_k uses en floored at rn_emin
        # (order zdftke.F90:182-184,469).
        rn_emin, rmxl_min = 1.0e-10, 1.0e-3
    else:
        avmb, avtb = 1.2e-4, 1.2e-5     # &namzdf rn_avm0 / rn_avt0
        rn_emin, rmxl_min = 1.0e-6, 1.0e-8   # rn_emin (standard); model mxl_min default
    _cfg = TKEConfig(
        c_k=rn_ediff,
        c_eps=rn_ediss,
        tke_background=rn_emin,         # rn_emin (1e-10 under ln_zdfiwm, else 1e-6)
        mxl_min=rmxl_min,               # rmxl_min (1e-3 under ln_zdfiwm, else model default)
        # Both arms above already carry NEMO's OWN rmxl_min, so the card must
        # NOT re-derive it: zdf_tke_init evaluates 1e-6/(rn_ediff*SQRT(rn_emin))
        # only on the ln_zdfiwm=.FALSE. arm (zdftke.F90:846), and ORCA1 runs
        # ln_zdfiwm=.TRUE., where :841-843 FORCES rmxl_min=1e-3.
        nemo_derived_mxl_min=False,
        # DECISION 72 (user, 2026-09-28): this ORCA1 card keeps MAIN's
        # calm-column ln_mxl0 surface floor, the namelist rn_mxl0 = 0.04 m,
        # stated here by value rather than left to a default.  NEMO itself
        # OVERWRITES rn_mxl0 with the active mixing-length floor whenever
        # ln_mxl0 is true -- 1.0e-3 m on this ORCA1 arm, because ln_zdfiwm
        # forces rmxl_min = 1.0e-3 (shipped zdftke.F90:859-862 and :841-843;
        # GYRE ppsrc:828-831 and :810-812) -- and that transcription stays
        # available, but only to the NEMO-literal cards, which select it with
        # nemo_mxl0_rmxl_min_overwrite=True.  The two arms differ ONLY where
        # the wind anchor does not already exceed the floor, i.e. on calm and
        # land columns; every windy column is identical.  Same principle as
        # decisions 66 and 68: this card is not ours to switch, and the PR body
        # states NEMO's own value so Pierre can decide.  False is also the
        # library default, so both lines are a record of the choice, not a
        # behaviour change.  Pinned by tests/ocean/unit/
        # test_nemo_card_opt_in_defaults.py::
        # test_orca1_card_keeps_mains_rn_mxl0_surface_floor.
        mxl0_min_m=0.04,
        nemo_mxl0_rmxl_min_overwrite=False,
        # DECISION 66 (user, 2026-09-28): this ORCA1 card keeps its PREVIOUS,
        # UNMASKED ln_mxl0 surface anchor.  NEMO's compiled statement does
        # multiply the stress by tmask(:,:,1) (zdftke.F90:602), and that
        # transcription stays available -- but only to the NEMO-literal cards,
        # which select it with nemo_mxl0_surface_tmask=True.  tmask(:,:,1) is 1
        # on every WET column, so the two arms differ on LAND columns only, and
        # the user's instruction is to leave this card alone: "only switch it in
        # the stuff that impacts our work, no need to touch his."  False is also
        # the library default, so this line is a record of the choice, not a
        # behaviour change.  Pinned by
        # tests/ocean/unit/test_nemo_card_opt_in_defaults.py::
        # test_orca1_card_keeps_the_unmasked_ln_mxl0_anchor.
        nemo_mxl0_surface_tmask=False,
        tke_surface_min=1.0e-4,         # rn_emin0
        # nn_mxl: choice=3 IS the NEMO nn_mxl construction (lup/ldown |dl/dz|<=e3t
        # sweeps) WITH the ln_mxl0 wind-stress surface anchor that NEMO ORCA1 runs
        # (ln_mxl0=T).  choice=2 (Veros Bougeault-Lacarrere, no ln_mxl0) was the
        # flagged fidelity gap; =3 closes it.  See the A/B campaign (2026-07-24).
        # ORCA1's namelist_cfg sets `nn_mxl = 2` (EXP00/namelist_cfg:444), which
        # is choice 4, NOT 3.  The card carried 3 while its own comment above
        # said 4 was the ORCA1 value -- flagged by codex 9405117 as a direct
        # card inconsistency that invalidates one-step parity.  Choices 3 and 4
        # share l_k = min(lup,ldn) and differ only in the dissipation length
        # (3: sqrt(lup*ldn), 4: min(lup,ldn)), so 4 dissipates more.
        tke_mxl_choice=4,
        # ---- what survived codex 9405307, which REFUTED three of five ------
        # The cumulative offline test (job 9405195) put all five at 0.854
        # Antarctic / 0.930 Arctic and I wired them as a new baseline. Codex
        # refuted three of them and the headline. Reverted, with the reasons
        # kept here so nobody re-wires them:
        #
        # veros_dz_slots=True -- REVERTED. The interior solver does map
        #   dz_cell<->e3t and dz_half<->e3w, but on --partial-cell the tripole
        #   caller supplies dz_ref*J while the real bottom thickness is
        #   h_partial*J, so the partial metric is silently discarded. It is
        #   not an e3t/e3w mapping at partial bottoms. It ALSO used to collide
        #   with nemo_z0 (both overloaded dz_surface, one as a Veros surface
        #   half-volume and one as a top-cell face-gradient distance, a factor
        #   2 apart on the centred grid) -- #1704 removed that collision by
        #   giving nemo_z0 its own face distance derived from
        #   dz_ref[0]*jacobian, so only the partial-bottom objection above
        #   still stands.
        #
        # n2_mode="nemo_bn2" -- REVERTED, and this one was simply wrong.
        #   ORCA1 selects TEOS-10 (namelist_cfg:307). legoESM's `nemo_bn2`
        #   calls nemo_seos_alpha_beta and its own docstring says "exact bn2
        #   (S-EOS)". That is a DIFFERENT N2, not ORCA1's rn2. Closing this
        #   properly needs a TEOS-10 rab/bn2, not a config flip.
        #
        # tke_surface_bc_level="nemo_z0" -- was reverted here for a REAL
        #   metric defect (dz_surface = -z_full_ref[0]*J is the top cell's
        #   MIDPOINT, half dz_ref[0], while NEMO's jk=2 lower coefficient
        #   needs the full top-cell e3t(1), which doubled the virtual-surface
        #   coupling).  #1704 FIXED that metric -- the face distance is now
        #   derived as dz_ref[0]*jacobian inside the kernel -- and the option
        #   is ON below (#1690).  Kept here only so the old reason is not
        #   mistaken for a live objection.
        #
        # ALSO REVERTED BY THE SAME REVIEW: --grid mpas shares this card and
        # mpas_integration.py:686 fail-loud rejects BOTH n2_mode != "insitu"
        # and veros_dz_slots=True, so the five-field card could not run on
        # MPAS at all. The user's standing ask is three-grid agreement; a card
        # that only one grid can execute is not a baseline.
        # ---- the two the review confirmed --------------------------------
        # Job 9405195, our K_M vs NEMO's own avm after one matched 3600 s step
        # from NEMO's state and en, Antarctic 491743 interfaces: the card as it
        # stood was ~2.6x NEMO, the amplitude fix took it to 1.846, alpha_tke=1
        # OVERSHOT to 0.634, and these five together land it at 0.854
        # (Arctic 0.930) -- inside the pre-registered 0.85-1.20 band.  They
        # converge rather than trade: alpha alone undershoots and these lift it
        # back.  This block is therefore a deliberate NEW BASELINE, not five
        # independent one-variable edits, and must not be compared term-by-term
        # against arms that predate it.
        #
        # veros_dz_slots: NEMO's zzd = -0.5*rn_Dt*mean(avm)/(e3t*e3w) is
        #   flux-form -- gradient across the intervening T cell (/e3t),
        #   divergence into the W control volume (/e3w).  legoESM maps
        #   dz_cell<->e3t and dz_half<->e3w ONLY in this branch; the legacy
        #   default uses dz_half for BOTH, so the coefficient was NEMO's while
        #   the stencil was not (codex 9405117 #2).  Alone: 0.627.
        # n2_mode: the clipped in-situ N2 carries a compressibility bias
        #   (+4.3e-5 measured in a prior session -- enough to stop the EVD
        #   trigger firing in the Arctic at all).  It sets BOTH the buoyancy
        #   length and the buoyancy sink.  k_profiles threads the gdept /
        #   interior-gdepw ladders when this mode is selected.  Alone: 0.808.
        # dissipation_discretization: NEMO linearises the Kolmogoroff sink as a
        #   NEWTON split -- zfact2 = 1.5*rn_Dt*rn_ediss on the diagonal
        #   (zdftke.F90:241,414) plus zfact3 = 0.5*rn_ediss added back
        #   explicitly (:242,:419) -- the correct Jacobian for eps ~ e^{3/2}.
        #   Plain backward Euler puts 1.0 on the diagonal with no add-back.
        #   Codex 9405117 predicted the direction before it was run ("retains
        #   more TKE than plain BE"); alone it lifts 0.634 -> 0.739.
        dissipation_discretization="nemo_1p5_split",
        # tke_surface_bc_level: NEMO holds en(1) at the z=0 W-point and SOLVES
        #   the tridiagonal from jk=2 (zdftke.F90:264,403-410).
        #   "interior_pinned" pins the Dirichlet value AT the first interior
        #   interface instead -- one w-level too deep.  Worth only ~4% on its
        #   own (measured, and it REFUTED a root-cause hypothesis of mine), but
        #   the z=0 placement is what NEMO does, and this card now selects it.
        #   It requires the surface Dirichlet value, which this card sets; the
        #   face distance it needs is NEMO's e3t(1), derived inside the closure
        #   from the (dz_ref, jacobian) pair both the C-grid and the MPAS
        #   bridge already thread (#1704) -- it is NOT dz_surface, which is the
        #   top cell's midpoint depth and doubled the coupling before #1704.
        # STILL NOT NEMO: tke_shear_production stays "squared_centered".
        # NEMO's zdf_sh2 is face-native with a now x before velocity product
        # and DOUBLES production adjacent to coasts via (2 - umask*umask)
        # (zdfsh2.F90).  "nemo_face_native"/"_now2" implement exactly that and
        # need the raw C-grid face state plus per-level wumask/wvmask/coast
        # masks, which no offline column probe can supply.
        # NO LONGER UNTESTED -- this comment used to say it was, and the
        # tripole run it asked for has since happened: the one-variable arm
        # `_trp_shear_d30.sbatch` (nemo_face_native_now2 off trp_base2_d30,
        # user-approved 2026-09-08) ran to day 30 as job 9689614 on 2026-09-10.
        # It MISSED its pre-registered primary: the 20-60 m tracer diffusivity
        # rose 1.22x where 10x was needed, the 220E undercurrent core stayed at
        # 0.199 m/s against the oracle's 0.546, and nino3 bias came in at +1.15
        # against a <= +1.10 guardrail.  The shear DISCRETISATION is therefore
        # eliminated as a bias lever (one of eleven; the cold-tongue cause is
        # still open and upstream of the closure -- weak resolved jet, hence
        # weak shear, hence weak production).
        # WHAT REMAINS TRUE is the FIDELITY statement, and it is now the only
        # open item on this card: ORCA1 builds with key_RK3, so the oracle runs
        # the "_now2" face-native form while this card runs a centred-average
        # one.  Averaging before differencing smooths, so we are systematically
        # weaker than NEMO near coasts.  Selecting it is a live user choice --
        # faithful but measurably slightly worse on nino3 -- so the default is
        # left alone rather than moved silently; --tke-shear-production selects
        # it per-run.
        # NEMO nn_bc_surf=1: en(1)=max(rn_emin0, rn_ebb·|τ|/ρ0) Dirichlet surface
        # TKE.  The Veros flux (|τ|/ρ0)^{3/2} default was a flagged gap; the
        # Dirichlet form matches NEMO and cuts the summer-hemisphere warm SST.
        surface_bc="nemo_dirichlet",
        # WHERE that Dirichlet value is held.  NEMO holds en(1) at the z=0
        # W-point and solves the tridiagonal from jk=2, leaving the first
        # interior interface (~10 m) on the interior floor rn_emin
        # (zdftke.F90:361,564-565).  "interior_pinned" -- the library default
        # this card used to inherit -- pins the FULL surface value (~6.6e-3
        # m2/s2 under typical wind) one W-level too deep, i.e. ~66x the
        # displacement of the DINO floor bug fixed in #1689.  The metric
        # defect that forced the earlier revert (dz_surface = the top cell's
        # MIDPOINT, half e3t(1), which doubled the virtual-surface coupling)
        # was fixed in #1704: the face distance is now derived as
        # dz_ref[0]*jacobian inside the kernel, with ONE owner, and the MPAS
        # bridge threads the same (dz_ref, jacobian) pair, so this card still
        # runs on --grid mpas.  Controlled d30 A/B on the tripole (#1690):
        # NOT WORSE on the pre-registered falsifier -- global MLD rmse, median
        # bias, correlation and the subpolar band unchanged, Southern Ocean a
        # hair better, largest move anywhere +0.3 m in the tropics against a
        # 42 m rmse; final-state gate metrics move in the 4th digit.  Set
        # EXPLICITLY (never inherited) because it is a scientific choice.
        # Revert with --tke-surface-bc-level interior_pinned (added with this
        # change, so BOTH axes of the surface-TKE choice appear in the run's
        # own command line rather than one of them moving implicitly).
        tke_surface_bc_level="nemo_z0",
        # NEMO integrates `en` PROGNOSTICALLY (one backward-Euler step/model-step
        # carrying OceanState.tke).  The quasi-steady diagnostic Mode-B was the
        # flagged gap; prognostic accumulates the tropical mixing energy and
        # RECOVERS the tropical SST (+0.55→+0.12; SST rmse 1.44→0.90, MLD 74→38 in
        # the controlled tripole d30 A/B) — the faithful tropical fix.  Revert any
        # lever via --tke-surface-bc/--tke-mxl-choice/--tke-prognostic.
        prognostic=True,
        prandtl_mode="richardson",      # nn_pdl=1
        prandtl_ri_coeff=pr_ri_slope,   # 1/ri_cri = 4.5 (NOT the Veros 6.6)
        # TKE VERTICAL-DIFFUSION COEFFICIENT.  NEMO diffuses `en` with the
        # PLAIN viscosity: zzd_up = -0.5*rn_Dt*(avm(k+1)+avm(k))/(e3t*e3w)
        # (zdftke.F90:407-409) is a face coefficient of mean(avm), i.e. x1.
        # TKEConfig's default is the Veros/Gaspar 30.0 -- as its own
        # __param_spec__ reference says outright, "NEMO avm x1 (zdftke);
        # Veros/Gaspar 30" -- so the card was diffusing TKE THIRTY TIMES too
        # fast against a NEMO oracle.  MEASURED on NEMO's own state and `en`,
        # one 3600 s step, Antarctic, 491743 interfaces, scored against NEMO's
        # avm (EVD-free: nn_evdm=0), job 9405026:
        #     zero step                          K_M/avm 0.861
        #     alpha_tke=30 (was the card)  e 3.20x, K_M/avm 1.846
        #     alpha_tke=1.0 (NEMO)         e 0.95x, K_M/avm 0.634
        # Arctic 1.666 -> 0.673.  Nothing else in the closure came within an
        # order of it: Langmuir, nn_etau, the mixing-length choice, the surface
        # BC placement and the dissipation split are each worth 2-9%.
        # HONEST CAVEAT: 1.0 overshoots the other way -- over that hour NEMO's
        # avm rises 5% and ours then FALLS 22%.  1.0 is still the faithful
        # value (it is what NEMO's discretisation computes); a residual of the
        # same order as the 0.861 zero-step offset remains, and the dissipation
        # split moves the wrong way for it.
        # No CLI flag: alpha_tke carries a __param_spec__ (tier 2, bounds
        # 1-90), so `--params vertical_mixing.tke.alpha_tke=30` reverts it.
        alpha_tke=1.0,                  # NEMO zdftke: TKE diffused by avm x1
        # STRATIFICATION. Precise statement, after GLM-5.2 pushed back and
        # zdftke.F90:650 settled it: BOTH forms floor N2 inside the length --
        # NEMO evaluates zrn2 = MAX(rn2, rsmall) and legoESM's choice-3/4
        # branch takes sqrt(max(N2, 1e-12)). So this is NOT "signed vs
        # clipped"; the floor is on both sides. What changes is the N2 VALUE.
        # The in-situ density gradient carries a POSITIVE compressibility bias
        # (+4.3e-5 measured in a prior session) that keeps N2 comfortably
        # above the floor even in neutral water, so the buoyancy length stays
        # SHORT. The adiabatic form reaches the floor where the water really
        # is neutral, the length runs to the sweep bound, and that is NEMO's
        # long zmxlm. Same mechanism as described below, correctly named.
        #
        # Carried from the same review, not acted on: the stacked MIN sweeps
        # make the diffusivity non-differentiable at the mixed-layer base, and
        # in neutral layers the limiter IS effectively the convection
        # parameterisation (length set by ramp geometry, not closure physics)
        # -- the documented resolution-sensitivity of convective mixing in
        # ORCA-type runs. ORCA1 sets no Galperin cap (no rn_clim_galp in the
        # namelist), so NEMO lives with this too; matching it is the goal here.
        #
        # ORIGINAL NOTE: the card computed N2 from the in-situ
        # density gradient CLIPPED at zero; NEMO's rn2 is SIGNED. In
        # convectively neutral water the signed form gives N -> 0, the
        # buoyancy length sqrt(2e)/N blows up, and the lup/ldown sweeps set a
        # LONG length -- which is what NEMO's zmxlm does there. Clipping
        # suppresses exactly those, and job 9407791 showed that is the whole
        # remaining mixing-length deficit: seeding the length with a signed N2
        # moved our zero-step length ratio against NEMO's own from 0.897 to
        # 0.997 (Southern Ocean), 0.914 to 0.998 (tropics) and 0.792 to 1.004
        # (Arctic). The largest correction is the Arctic, the most convective
        # band -- the mechanism's own prediction.
        #
        # This is also the counterpart of alpha_tke: at day 30 the sqrt(2) and
        # alpha fixes drove the Southern Ocean mixed layer from +1.5 m to
        # -19.7 m, because removing two large over-mixing errors left the
        # short-length under-mixing error uncompensated. This is that error.
        #
        # n2_eos_form: ORCA1 runs ln_teos10=.true. (namelist_cfg:308), so the
        # alpha/beta come from the Roquet polynomial with the TEOS-10
        # coefficient set. A previous revision set n2_mode alone and codex
        # 9405307 refuted it -- the bn2 helper defaulted to S-EOS, so the
        # TEOS-10 path was unreachable. Both fields are needed.
        n2_mode="nemo_bn2",
        n2_eos_form="teos10",
        # K-from-TKE AMPLITUDE.  NEMO zdftke tke_avn computes
        #     zsqen = SQRT(en) ; zav = rn_ediff*zmxlm*zsqen
        #     p_avm = MAX(zav, avmb)*wmask
        # -- the sqrt carries `en`, NOT `2*en`; the factor 2 lives in the
        # LENGTH, zmxlm = SQRT(2*en/rn2) (zdftke.F90:651).  (NEMO's own header
        # comment at :150/:553 writes the momentum floor as `avtb`; the CODE
        # uses `avmb` and `avtb` is the TRACER floor -- codex 9400815 #1.
        # nn_pdl=1 changes only avt, and NOT as pdlr*avm: the tracer floor
        # avtb is applied AFTER the Prandtl reduction, which
        # compute_K_from_tke mirrors as K_H = max(kappaH_min, K_M_raw/Pr).)
        # legoESM's `gaspar_sqrt2e` DEFAULT applies sqrt(2*e) in the amplitude
        # as well, and the nn_mxl=2/3 branch already builds the length from
        # sqrt(2)*sqrt(e)/N (tke.py:698-702), so the default DOUBLE-COUNTS the
        # sqrt(2) on exactly the path this card selects.  MEASURED on NEMO's
        # own state (Stage A, commit 39ce0701c, Arctic calm columns):
        #     gaspar_sqrt2e  K_H 4.048e-2 = 4.81x NEMO avt 8.410e-3
        #     veros_sqrte    K_H 2.730e-2 = 3.25x
        # SCOPE OF "exact", stated narrowly on purpose (codex 9400815 #2):
        # `veros_sqrte` is c_k*l_k*sqrt(max(0,e)) and is EXACT against NEMO on
        # wet rows of this card's normal trajectory, because positivity="floor"
        # ends every solve at e >= tke_background = rn_emin (1e-6, or 1e-10 under
        # --iwm per zdftke.F90:840), so
        # sqrt(max(0,e)) == sqrt(e) there.  It is NOT globally identical: NEMO
        # zeroes dry rows via *wmask while this card leaves tke_dry_wmask=False
        # and keeps the background there, and the e<=0 branch differs (ours
        # returns 0, floored back up by kappaM_min/kappaH_min; NEMO floors en
        # first).  Neither is reachable on the floored path this card runs.
        # SIGN, also narrowly: at fixed e/l_k/N2/shear the pin divides K by
        # sqrt(2).  It does NOT follow that the mixed layer shallows by any
        # particular amount -- the prognostic solve is coupled (lower K_M cuts
        # shear production and reinforces; lower K_H cuts buoyancy destruction
        # and offsets), so the integrated response is what the day-90 A/B
        # measures, not something the algebra gives.
        # Revert with --tke-kappa-convention, which reaches the tripole and
        # (since 2026-08-22) the MPAS grid as well, so the same A/B can be run
        # on either.
        kappa_convention="veros_sqrte",
        lc=True,                        # ln_lc
        lc_coeff=0.25,                  # rn_lc (namelist_cfg override)
        etau_mode="below_ml",           # nn_etau=1
        etau_frac=0.08,                 # rn_efr (namelist_cfg override)
        etau_htau_mode="latitude",      # nn_htau=1 (namelist_ref default)
        eice=3,                         # nn_eice=3 — under-ice lc/etau attenuation
        kappaM_min=avmb,                # NEMO avm = max(closure, avmb)
        kappaH_min=avtb,                # NEMO avt = max(pdl·avt, avtb)
        bg_diff_scale=0.0,              # nn_avb=0 — no depth-profile background
    )
    # Surface TKE boundary condition (``--tke-surface-bc``).  DEFAULT keeps the
    # TKEConfig default (``veros_flux``, the flagged fidelity gap listed above).
    # ``nemo_dirichlet`` selects the NEMO ``en(1)=max(rn_emin0, rn_ebb·|τ|/ρ0)``
    # Dirichlet condition (rn_ebb=67.83; tke.py ``_surface_tke_dirichlet``) —
    # closing the surface-BC gap so wind energy enters the near-surface TKE at
    # the NEMO rate rather than the weaker Veros flux ``(|τ|/ρ0)^{3/2}`` form.
    if surface_bc is not None:
        if surface_bc not in ("veros_flux", "nemo_dirichlet"):
            raise ValueError(
                f"orca1_zdftke_config surface_bc {surface_bc!r} invalid; "
                "expected 'veros_flux' or 'nemo_dirichlet' (NEMO nn_bc_surf).")
        _cfg = _cfg._replace(surface_bc=surface_bc)
    # WHERE the surface TKE is held (``--tke-surface-bc-level``).  The card
    # selects NEMO's z=0 W-point (#1690); this knob is how an A/B arm asks for
    # the old interior placement back.
    if surface_bc_level is not None:
        if surface_bc_level not in ("interior_pinned", "nemo_z0"):
            raise ValueError(
                f"orca1_zdftke_config surface_bc_level {surface_bc_level!r} "
                "invalid; expected 'interior_pinned' or 'nemo_z0'.")
        _cfg = _cfg._replace(tke_surface_bc_level=surface_bc_level)
    # The z=0 placement places a HELD Dirichlet surface value.  The Veros flux
    # form supplies none, and the closure raises on that pair -- so refuse it
    # HERE, at config build, naming the flag that resolves it.  Quietly
    # swapping the placement instead would move TWO axes on an arm whose whole
    # purpose is to isolate ONE (the BC form), which is how an A/B stops
    # measuring what it claims (GLM review; the crash itself was codex's).
    if (_cfg.surface_bc != "nemo_dirichlet"
            and _cfg.tke_surface_bc_level == "nemo_z0"):
        raise ValueError(
            "tke_surface_bc_level='nemo_z0' holds a Dirichlet surface TKE "
            f"value at the z=0 W-point, but surface_bc={_cfg.surface_bc!r} "
            "supplies no held value (the closure raises on this pair). "
            "Run the legacy flux arm with "
            "--tke-surface-bc veros_flux --tke-surface-bc-level interior_pinned "
            "so BOTH axes are recorded in the run's own configuration.")
    # Langmuir + surface-TKE penetration overrides (``--tke-lc``/``--tke-etau``).
    # DEFAULT keeps the ORCA1 card (ln_lc=T, nn_etau=1).  The OFF settings
    # exist to build a "fesom-mimic" card: fesom-jax's CVMix TKE has no
    # Langmuir and no etau penetration, and quantifying the FESOM2 skill gap
    # requires running OUR physics with THOSE branches off (2026-08-18).
    if lc is not None:
        _cfg = _cfg._replace(lc=bool(lc))
    if etau_mode is not None:
        if etau_mode not in ("below_ml", "none"):
            raise ValueError(f"orca1_zdftke_config etau_mode {etau_mode!r} "
                             "invalid; expected 'below_ml' or 'none'.")
        _cfg = _cfg._replace(etau_mode=etau_mode)
    # K-from-TKE amplitude (``--tke-kappa-convention``).  DEFAULT keeps the
    # card value (``veros_sqrte`` = NEMO's ``rn_ediff*zmxlm*sqrt(en)``);
    # ``gaspar_sqrt2e`` restores the legacy sqrt(2)-double-counting amplitude
    # so the fix can be A/B'd against every arm that predates it.
    if kappa_convention is not None:
        if kappa_convention not in ("veros_sqrte", "gaspar_sqrt2e"):
            raise ValueError(
                f"orca1_zdftke_config kappa_convention {kappa_convention!r} "
                "invalid; expected 'veros_sqrte' (NEMO avm = rn_ediff*zmxlm*"
                "sqrt(en)) or 'gaspar_sqrt2e' (the legacy double-count).")
        _cfg = _cfg._replace(kappa_convention=kappa_convention)
    # Shear-production discretisation (``--tke-shear-production``).  DEFAULT
    # keeps the card value (``squared_centered``).  ``nemo_face_native`` is
    # NEMO's zdf_sh2: face-native differences, a now x before velocity
    # product, and production DOUBLED adjacent to coasts via
    # (2 - umask*umask) (zdfsh2.F90:78-94).  It is the LAST unclosed gap on
    # this card and the only one the offline column probe cannot evaluate --
    # it needs the raw C-grid face state, which k_profiles supplies on the
    # tripole under --partial-cell.
    if shear_production is not None:
        if shear_production not in ("squared_centered", "nemo_face_native",
                                    "nemo_face_native_now2",
                                    "nemo_face_native_nbb2", "nemo_burchard"):
            raise ValueError(
                f"orca1_zdftke_config shear_production {shear_production!r} "
                "invalid; expected 'squared_centered', 'nemo_face_native' "
                "(NEMO zdf_sh2, leap-frog family), 'nemo_face_native_nbb2' "
                "(same face geometry at Nbb^2 -- the key_RK3 step-entry arm) "
                "or its legacy alias 'nemo_face_native_now2', or "
                "'nemo_burchard'.")
        _cfg = _cfg._replace(tke_shear_production=shear_production)
    # avm inside the zdf_sh2 face sum (``--tke-shear-avm-weighting``).  DEFAULT
    # keeps the card value ("tpoint": one T-point K_M times the T-collapsed
    # shear).  "nemo_face" is zdfsh2.F90:80-94: avm(i+1)+avm(i) summed on each
    # u/v face, times that face's shear, then the 0.25 coast-doubled T combine.
    # Only meaningful on the face-native geometry, so it is refused otherwise.
    if shear_avm_weighting is not None:
        if shear_avm_weighting not in ("tpoint", "nemo_face"):
            raise ValueError(
                f"orca1_zdftke_config shear_avm_weighting {shear_avm_weighting!r} "
                "invalid; expected 'tpoint' or 'nemo_face'.")
        _cfg = _cfg._replace(tke_shear_avm_weighting=shear_avm_weighting)
    # Prandtl number (``--tke-prandtl-mode``).  DEFAULT keeps the card value
    # ("richardson": Pr = clamp(Ri/ri_cri, 1, 10) on the gradient Ri =
    # N2/shear^2).  "nemo_ri" is NEMO nn_pdl=1 as zdftke.F90 writes it:
    # zri = rn2b*avm/(p_sh2 + rn_bshear) on the avm-weighted shear production
    # (the face-summed p_sh2 when --tke-shear-avm-weighting nemo_face).
    if prandtl_mode is not None:
        if prandtl_mode not in ("richardson", "nemo_ri"):
            raise ValueError(
                f"orca1_zdftke_config prandtl_mode {prandtl_mode!r} invalid; "
                "expected 'richardson' or 'nemo_ri'.")
        _cfg = _cfg._replace(prandtl_mode=prandtl_mode)
    # Coefficient lifetime (``--tke-preclosure-coeff-source``).  DEFAULT keeps
    # the card value ("current_subiteration").  NEMO's zdf_tke consumes the
    # SAVE'd avm_k/avt_k pair in zdf_sh2, the Prandtl ratio, the TKE matrix and
    # the RHS, and overwrites that pair only AFTER the solve (zdftke.F90); it
    # never evaluates production against a coefficient recomputed in the same
    # call.  "carried_previous_step" is that lifetime, and it is what the DINO
    # NEMO-oracle preset selects (experiments/dino.py).  The memory it needs
    # (tke_avm/tke_avt/tke_avm_surface, plus tke_dissl under the literal
    # matrix) lives on LatLonCGridOceanState only, so the caller restricts this
    # to the tripole; the model seeds it at cold start from zdf_phy_init's
    # background-times-wmask construction and bridges it across restarts.
    if preclosure_coeff_source is not None:
        if preclosure_coeff_source not in ("current_subiteration",
                                           "carried_previous_step"):
            raise ValueError(
                "orca1_zdftke_config preclosure_coeff_source "
                f"{preclosure_coeff_source!r} invalid; expected "
                "'current_subiteration' (recompute within the call) or "
                "'carried_previous_step' (NEMO's avm_k/avt_k lifetime).")
        _cfg = _cfg._replace(
            tke_preclosure_coeff_source=preclosure_coeff_source)
    # Buoyancy sink discretisation (``--tke-buoyancy-sink``).  NEMO puts the
    # whole -avt*rn2 term on the RHS explicitly (zdftke.F90:417-420, no
    # Patankar split); the card's default 'implicit_linearized' charges the
    # stable part against the NEW e on the diagonal.  None keeps the card.
    if buoyancy_sink is not None:
        if buoyancy_sink not in ("implicit_linearized", "nemo_explicit"):
            raise ValueError(
                f"orca1_zdftke_config buoyancy_sink {buoyancy_sink!r} invalid; "
                "expected 'implicit_linearized' or 'nemo_explicit'.")
        _cfg = _cfg._replace(tke_buoyancy_sink=buoyancy_sink)
    # TKE step evaluation (``--tke-step-evaluation``).  'nemo_literal' selects
    # the zdftke.F90:403-455 matrix assembly AND its tridiagonal recurrences
    # (tke.py tke_matrix_evaluation / tke_solver_evaluation, the DINO-certified
    # pair); it needs the nemo_z0 surface row and the carried avm_k/dissl
    # (--tke-surface-bc-level nemo_z0 --tke-preclosure-coeff-source
    # carried_previous_step), which the closure enforces.  None keeps the card
    # ('factored' + 'shared_thomas').
    # Langmuir vs explicit dissipation add-back (``--tke-langmuir-rhs-coupling``,
    # zdftke.F90:367,419).  None keeps the card ('separate').
    if langmuir_rhs_coupling is not None:
        if langmuir_rhs_coupling not in ("separate", "nemo_pre_solve"):
            raise ValueError(
                f"orca1_zdftke_config langmuir_rhs_coupling {langmuir_rhs_coupling!r} "
                "invalid; expected 'separate' or 'nemo_pre_solve'.")
        _cfg = _cfg._replace(tke_langmuir_rhs_coupling=langmuir_rhs_coupling)
    if step_evaluation is not None:
        if step_evaluation not in ("factored", "nemo_literal"):
            raise ValueError(
                f"orca1_zdftke_config step_evaluation {step_evaluation!r} invalid; "
                "expected 'factored' or 'nemo_literal'.")
        if step_evaluation == "nemo_literal":
            # The literal step is a BUNDLE (tke.py:1220, :2786-2800): live
            # e3t/e3w from the step-entry N2 stage and the wmask-zeroed dry rows
            # (w_active) are operands of the transcribed matrix, so they switch
            # with it; nemo_z0 and carried_previous_step come from their own
            # flags and are checked at parse time in _validate_tke_card_grid.
            _cfg = _cfg._replace(
                tke_matrix_evaluation="nemo_literal",
                tke_solver_evaluation="nemo_literal",
                tke_n2_evaluation_stage="step_entry",
                tke_dry_wmask=True)
        else:
            _cfg = _cfg._replace(tke_matrix_evaluation="factored",
                                 tke_solver_evaluation="shared_thomas")
    # Mixing-length formulation (``--tke-mxl-choice``).  DEFAULT keeps the card
    # value (2 = Veros Bougeault-Lacarrere, the current production).  3 selects
    # NEMO nn_mxl=3: the lup/ldown |dl/dz|<=e3t sweeps WITH the ln_mxl0 wind-
    # stress surface anchor (l_sfc=max(rn_mxl0, vkarmn*2e5/(rho0*g)*|tau|)) that
    # choice 2 omits — a larger upper-ocean mixing length -> more mixed-layer
    # mixing -> cooler SST (the tropical-warm fix candidate).  The tripole
    # k_profiles path already threads dz_ref/jacobian/taum so choice 3 is live.
    # 4 selects NEMO nn_mxl=2 — the value ORCA1's namelist_cfg ACTUALLY sets
    # (`nn_mxl = 2`, verified in RUN_GATEWAY/namelist_cfg).  Choices 3 and 4
    # share the lup/ldown sweeps and the same eddy-coefficient LENGTH
    # l_k = min(lup,ldn); they differ only in the dissipation length
    # (tke.py:764-785): choice 3 uses l_eps = sqrt(lup*ldn), choice 4 uses
    # l_eps = l_k = min(lup,ldn).  Since min <= sqrt(product), choice 4
    # dissipates MORE (eps = c_eps*e^{3/2}/l_eps).
    # NOTE (codex 2026-08-07 r1+r2, correcting an earlier claim of mine):
    # do NOT say the diffusivity is identical.  Only the LENGTH l_k is
    # shared.  A shorter l_eps raises the dissipation coefficient at fixed
    # positive TKE (tke.py:1007), and K is recomputed from the updated TKE
    # (tke.py:2153), so choice 4 CAN reduce TKE and alter K.  It need not:
    # the two lengths can be equal, later trajectories differ in l_k too,
    # and K can coincide at the floors/ceilings (tke.py:1530).  'Can alter',
    # never 'must'.  Until 2026-08-06 argparse accepted 4 while this
    # builder raised on it, so `--tke-mxl-choice 4` crashed the run.
    if mxl_choice is not None:
        # int(4.9) would silently truncate to 4; argparse blocks that via
        # type=int but a PROGRAMMATIC caller does not (codex 2026-08-07 #5).
        # Reject non-integral numerics (int(4.9)->4 would run choice-4 physics
        # under a nonsense value) and anything outside the SHARED set.  bool is
        # excluded explicitly: True == 1 would otherwise sneak through int().
        if (isinstance(mxl_choice, bool)
                or not isinstance(mxl_choice, (int, np.integer))
                or int(mxl_choice) not in TKE_MXL_CHOICES):
            raise ValueError(
                f"orca1_zdftke_config mxl_choice {mxl_choice!r} invalid; expected "
                f"an int in {TKE_MXL_CHOICES} (2=Veros Bougeault-Lacarrere, "
                "3=NEMO nn_mxl=3, 4=NEMO nn_mxl=2 -- the value ORCA1's "
                "namelist_cfg actually runs).")
        _cfg = _cfg._replace(tke_mxl_choice=int(mxl_choice))
    if n2_mode is not None:
        # Scheme Literal -> validate at config-build time on the STATIC value,
        # never a silent fallback (dispatch hardening).
        _N2_MODES = ("insitu", "insitu_signed", "adiabatic", "nemo_bn2")
        if n2_mode not in _N2_MODES:
            raise ValueError(
                f"orca1_zdftke_config n2_mode {n2_mode!r} invalid; expected "
                f"one of {_N2_MODES}")
        _cfg = _cfg._replace(n2_mode=n2_mode)
    if n2_eos_form is not None:
        _EOS_FORMS = ("seos", "teos10")
        if n2_eos_form not in _EOS_FORMS:
            raise ValueError(
                f"orca1_zdftke_config n2_eos_form {n2_eos_form!r} invalid; "
                f"expected one of {_EOS_FORMS}")
        _cfg = _cfg._replace(n2_eos_form=n2_eos_form)
    # Prognostic vs diagnostic TKE (``--tke-prognostic``).  DEFAULT keeps the
    # card value (False = the DINO-validated quasi-steady Mode-B diagnostic, 3
    # backward-Euler iters).  True selects NEMO's PROGNOSTIC en integration
    # (Mode-A, one backward-Euler step/model-step carrying OceanState.tke) — the
    # closure ACCUMULATES the diurnal-SW + wind TKE that mixes the tropical ML,
    # the candidate for the tropical warm that the mixing-length levers can't
    # touch.  The tripole implicit path carries state.tke
    # (_tke_prognostic_active gates it) so this is live.
    if prognostic is not None:
        _cfg = _cfg._replace(prognostic=bool(prognostic))
    # ORDERING: this block runs LAST because it VALIDATES AGAINST the
    # resolved card -- surface_bc and tke_mxl_choice may themselves have
    # been overridden above, and checking them before they are applied
    # read the card defaults instead of the requested values (its own
    # precondition test caught that).
    # Surface TKE boundary PLACEMENT (``--tke-surface-bc-level``).  DEFAULT
    # keeps the card value ("interior_pinned").  NEMO holds en(1) at the z=0
    # W-point and SOLVES the tridiagonal from jk=2 (zdftke.F90:264,403-410),
    # so its first interior interface is a genuinely solved row that receives
    # downward TKE transport; "interior_pinned" pins that row itself, one
    # w-level too deep, and a pinned row can receive no transport at all.
    #
    # The 2026-09-08 revert of this option is SUPERSEDED: it was reverted
    # because the virtual-surface face distance was taken from dz_surface
    # (the top cell's MIDPOINT depth, half e3t(1)), which doubled the surface
    # coupling.  #1690 gave that face its own metric -- tke.py derives
    # dz_face_surface = dz_ref[0]*jacobian internally, and k_profiles already
    # threads dz_ref and the Jacobian on this path -- so the metric objection
    # no longer applies.  The measured "~4% on its own" caveat DOES still
    # stand; it came from an offline column probe, not a run.
    if surface_bc_level is not None:
        if surface_bc_level not in ("interior_pinned", "nemo_z0"):
            raise ValueError(
                f"orca1_zdftke_config surface_bc_level {surface_bc_level!r} "
                "invalid; expected 'interior_pinned' (pin the first interior "
                "interface) or 'nemo_z0' (NEMO's held z=0 row, solved from "
                "jk=2).")
        if (surface_bc_level == "nemo_z0"
                and _cfg.surface_bc != "nemo_dirichlet"):
            raise ValueError(
                "tke_surface_bc_level='nemo_z0' requires "
                "surface_bc='nemo_dirichlet': the virtual surface row IS the "
                f"held Dirichlet value, and this card has {_cfg.surface_bc!r}.")
        if surface_bc_level == "nemo_z0" and _cfg.tke_mxl_choice not in (3, 4):
            raise ValueError(
                "tke_surface_bc_level='nemo_z0' requires tke_mxl_choice 3 or "
                "4: the surface viscosity comes from the ln_mxl0 anchor, and "
                f"this card has tke_mxl_choice={_cfg.tke_mxl_choice}.")
        _cfg = _cfg._replace(tke_surface_bc_level=surface_bc_level)

    return _cfg
