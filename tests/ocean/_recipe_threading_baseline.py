"""Seed baselines for ``tests/ocean/test_recipe_option_threading.py``.

Format follows the existing ratchet convention (see ``tests/_inline_coeff_baseline.py``,
``tests/test_dispatch_hardening.py``): each dict below pins the EXACT set of
findings the four invariants produced when this gate was introduced (2026-08-08),
against every card in ``legoesm.ocean.experiments.dino.DINO_RECIPES``. All four
are SHRINK-ONLY -- the consuming test asserts the currently-discovered set is a
SUBSET of the baseline (a new/different finding is red) AND that no baseline
entry has gone stale (a fixed finding must be deleted from here, not left as
dead slack) -- so a regression that reproduces the class this gate exists for
(CLAUDE.md: "a card sets a faithful option and it never reaches the code that
consumes it") cannot silently widen the allowance.

Do not hand-edit a count up to make a new violation pass -- fix the threading
bug, or, if the finding is a genuine non-bug (a name legitimately reused by
unrelated schemes, or a documented harness-only / different-call-site option),
add a reasoned entry and say so in the PR.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Invariant A -- two-surface disagreement.
#
# Leaves of the assembled (LatLonCGridOceanConfig, OceanPhysicsConfig) pair are
# grouped by TRAILING field name; >=2 paths carrying >=2 distinct values under
# the same trailing name is a candidate silent-shadow (the ``g`` / GM-Redi-shadow
# class). Most entries here are legitimate: the ocean physics tree re-uses short
# field names (``enabled``, ``scheme``, ``alpha``, ``K_max``, ``n2_mode``, ...)
# across UNRELATED scheme blocks by design (KPP epsilon vs GM/Redi alpha are
# different physical knobs sharing a name) -- this invariant is a blunt
# tripwire, not a semantic check, so the baseline absorbs the real reuse and
# only a NEW disagreement (a name/pair not seen before) goes red.
#
# Entries: (card, path_a, path_b, reason). path_a/path_b are the two
# alphabetically-first representative paths for two DIFFERING values under the
# same trailing name (one pair per distinct-value boundary, not the full
# cross-product -- see generator in the PR description).
#
# OBSERVATION (recorded, deliberately NOT acted on): ~190 of these entries are
# legitimate short-name reuse rather than distinct bug classes, and it is
# tempting to "sharpen" the grouping by also requiring path-prefix similarity.
# DO NOT. The shrink-only rule already makes a large baseline work as a
# tripwire -- a NEW disagreement goes red regardless of how many known ones sit
# beside it -- and prefix-similarity would silently drop exactly the
# cross-subsystem case this gate exists for: ``model.constants.rho_0`` vs
# ``model.physics.surface_forcing.flux_feedback.rho_0`` share no prefix by
# construction, and that pair IS the #1226 bug class.
INVARIANT_A_BASELINE: tuple[tuple[str, str, str, str], ...] = (
    ('legoesm_default', 'model.lateral_viscosity.A_h', 'model.physics.lateral_mixing.harmonic.A_h', 'seeded'),
    ('legoesm_default', 'model.K_h', 'model.physics.lateral_mixing.harmonic.K_h', 'seeded'),
    ('legoesm_default', 'model.A_v', 'model.physics.vertical_mixing.constant.A_v', 'seeded'),
    ('legoesm_default', 'model.K_v', 'model.physics.vertical_mixing.constant.K_v', 'seeded'),
    ('legoesm_default', 'model.gm_redi.kappa_GM', 'model.physics.lateral_mixing.gm_redi.kappa_GM', 'seeded'),
    ('legoesm_default', 'model.gm_redi.kappa_Redi', 'model.physics.lateral_mixing.gm_redi.kappa_Redi', 'seeded'),
    ('legoesm_default', 'model.gm_redi.S_max', 'model.physics.lateral_mixing.gm_redi.S_max', 'seeded'),
    ('legoesm_default', 'model.gm_redi.treguier.enabled', 'model.gm_redi.visbeck.enabled', 'seeded'),
    ('legoesm_default', 'model.gm_redi.visbeck.alpha', 'model.physics.vertical_mixing.richardson.alpha', 'seeded'),
    ('legoesm_default', 'model.gm_redi.treguier.kappa_min', 'model.gm_redi.visbeck.kappa_min', 'seeded'),
    ('legoesm_default', 'model.gm_redi.visbeck.kappa_min', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_min', 'seeded'),
    ('legoesm_default', 'model.gm_redi.visbeck.kappa_max', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_max', 'seeded'),
    ('legoesm_default', 'model.gm_redi.treguier.aei0', 'model.physics.lateral_mixing.gm_redi.treguier.aei0', 'seeded'),
    ('legoesm_default', 'model.freezing.scheme', 'model.physics.bottom_drag.scheme', 'seeded'),
    ('legoesm_default', 'model.physics.bottom_drag.scheme', 'model.physics.convection.scheme', 'seeded'),
    ('legoesm_default', 'model.physics.convection.scheme', 'model.physics.shortwave_penetration.scheme', 'seeded'),
    ('legoesm_default', 'model.physics.shortwave_penetration.scheme', 'model.physics.vertical_mixing.scheme', 'seeded'),
    ('legoesm_default', 'model.physics.convection.enhanced_diffusion.K_bg', 'model.physics.vertical_mixing.kpp.K_bg', 'seeded'),
    ('legoesm_default', 'model.physics.vertical_mixing.kpp.A_bg', 'model.physics.vertical_mixing.richardson.A_bg', 'seeded'),
    ('legoesm_default', 'model.physics.vertical_mixing.kpp.K_max', 'model.physics.vertical_mixing.tidal.K_max', 'seeded'),
    ('legoesm_default', 'model.physics.convection.enhanced_diffusion.K_conv', 'model.physics.vertical_mixing.kpp.K_conv', 'seeded'),
    ('legoesm_default', 'model.constants.rho_0', 'model.physics.surface_forcing.flux_feedback.rho_0', "inert-but-ARMED: physics.surface_forcing.flux_feedback.rho_0=1025.0 and physics.vertical_mixing.tidal.rho_0=1025.0 both differ from constants.rho_0=1026.0. VERIFIED inert on this card: surface_forcing.scheme='none' and vertical_mixing.tidal.enabled=False, so neither surface is read. BECOMES A LIVE TWO-SURFACE DIVERGENCE the moment tidal mixing is enabled or a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate rho_0/c_sw deliberate (the Veros surface-forcing block owns cp_0) -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 1025.0 is the right value there."),
    ('legoesm_default', 'model.physics.convection.enhanced_diffusion.cfl_safety', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'seeded'),
    ('legoesm_default', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'model.physics.lateral_mixing.harmonic.cfl_safety', 'seeded'),
    ('legoesm_default', 'model.constants.c_sw', 'model.physics.surface_forcing.flux_feedback.c_sw', "inert-but-ARMED: physics.surface_forcing.flux_feedback.c_sw=3994.0 differs from constants.c_sw=3991.86795711963 (the card's own c_p pin). VERIFIED inert on this card: surface_forcing.scheme='none', so the flux_feedback surface is never read. BECOMES A LIVE TWO-SURFACE DIVERGENCE as soon as a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate c_sw deliberate -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 3994.0 is the right value there."),
    ('mitgcm', 'model.lateral_viscosity.A_h', 'model.physics.lateral_mixing.harmonic.A_h', 'seeded'),
    ('mitgcm', 'model.K_h', 'model.physics.lateral_mixing.harmonic.K_h', 'seeded'),
    ('mitgcm', 'model.A_v', 'model.physics.vertical_mixing.constant.A_v', 'seeded'),
    ('mitgcm', 'model.K_v', 'model.physics.vertical_mixing.constant.K_v', 'seeded'),
    ('mitgcm', 'model.gm_redi.kappa_GM', 'model.physics.lateral_mixing.gm_redi.kappa_GM', 'seeded'),
    ('mitgcm', 'model.gm_redi.kappa_Redi', 'model.physics.lateral_mixing.gm_redi.kappa_Redi', 'seeded'),
    ('mitgcm', 'model.gm_redi.S_max', 'model.physics.lateral_mixing.gm_redi.S_max', 'seeded'),
    ('mitgcm', 'model.gm_redi.treguier.enabled', 'model.gm_redi.visbeck.enabled', 'seeded'),
    ('mitgcm', 'model.gm_redi.visbeck.alpha', 'model.physics.vertical_mixing.richardson.alpha', 'seeded'),
    ('mitgcm', 'model.gm_redi.treguier.kappa_min', 'model.gm_redi.visbeck.kappa_min', 'seeded'),
    ('mitgcm', 'model.gm_redi.visbeck.kappa_min', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_min', 'seeded'),
    ('mitgcm', 'model.gm_redi.visbeck.kappa_max', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_max', 'seeded'),
    ('mitgcm', 'model.gm_redi.treguier.aei0', 'model.physics.lateral_mixing.gm_redi.treguier.aei0', 'seeded'),
    ('mitgcm', 'model.freezing.scheme', 'model.physics.bottom_drag.scheme', 'seeded'),
    ('mitgcm', 'model.physics.bottom_drag.scheme', 'model.physics.convection.scheme', 'seeded'),
    ('mitgcm', 'model.physics.convection.scheme', 'model.physics.shortwave_penetration.scheme', 'seeded'),
    ('mitgcm', 'model.physics.shortwave_penetration.scheme', 'model.physics.vertical_mixing.scheme', 'seeded'),
    ('mitgcm', 'model.physics.convection.enhanced_diffusion.K_bg', 'model.physics.vertical_mixing.kpp.K_bg', 'seeded'),
    ('mitgcm', 'model.physics.vertical_mixing.kpp.A_bg', 'model.physics.vertical_mixing.richardson.A_bg', 'seeded'),
    ('mitgcm', 'model.physics.vertical_mixing.kpp.K_max', 'model.physics.vertical_mixing.tidal.K_max', 'seeded'),
    ('mitgcm', 'model.physics.convection.enhanced_diffusion.K_conv', 'model.physics.vertical_mixing.kpp.K_conv', 'seeded'),
    ('mitgcm', 'model.constants.rho_0', 'model.physics.surface_forcing.flux_feedback.rho_0', "inert-but-ARMED: physics.surface_forcing.flux_feedback.rho_0=1025.0 and physics.vertical_mixing.tidal.rho_0=1025.0 both differ from constants.rho_0=1026.0. VERIFIED inert on this card: surface_forcing.scheme='none' and vertical_mixing.tidal.enabled=False, so neither surface is read. BECOMES A LIVE TWO-SURFACE DIVERGENCE the moment tidal mixing is enabled or a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate rho_0/c_sw deliberate (the Veros surface-forcing block owns cp_0) -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 1025.0 is the right value there."),
    ('mitgcm', 'model.physics.convection.enhanced_diffusion.cfl_safety', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'seeded'),
    ('mitgcm', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'model.physics.lateral_mixing.harmonic.cfl_safety', 'seeded'),
    ('mitgcm', 'model.constants.c_sw', 'model.physics.surface_forcing.flux_feedback.c_sw', "inert-but-ARMED: physics.surface_forcing.flux_feedback.c_sw=3994.0 differs from constants.c_sw=3991.86795711963 (the card's own c_p pin). VERIFIED inert on this card: surface_forcing.scheme='none', so the flux_feedback surface is never read. BECOMES A LIVE TWO-SURFACE DIVERGENCE as soon as a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate c_sw deliberate -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 3994.0 is the right value there."),
    ('nemo_dino_kamm', 'model.lateral_viscosity.A_h', 'model.physics.lateral_mixing.harmonic.A_h', 'seeded'),
    ('nemo_dino_kamm', 'model.K_h', 'model.physics.lateral_mixing.harmonic.K_h', 'seeded'),
    ('nemo_dino_kamm', 'model.A_v', 'model.physics.vertical_mixing.constant.A_v', 'seeded'),
    ('nemo_dino_kamm', 'model.K_v', 'model.physics.vertical_mixing.constant.K_v', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.kappa_GM', 'model.physics.lateral_mixing.gm_redi.kappa_GM', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.kappa_Redi', 'model.physics.lateral_mixing.gm_redi.kappa_Redi', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.treguier.enabled', 'model.gm_redi.visbeck.enabled', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.visbeck.alpha', 'model.physics.vertical_mixing.richardson.alpha', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.treguier.kappa_min', 'model.gm_redi.visbeck.kappa_min', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.visbeck.kappa_min', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_min', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.visbeck.kappa_max', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_max', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.visbeck.n2_mode', 'model.physics.convection.enhanced_diffusion.n2_mode', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.treguier.aei0', 'model.physics.lateral_mixing.gm_redi.treguier.aei0', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.slope_scheme', 'model.physics.lateral_mixing.gm_redi.slope_scheme', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.gm_bolus_advection', 'model.physics.lateral_mixing.gm_redi.gm_bolus_advection', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.gm_bolus_kappa_face_average', 'model.physics.lateral_mixing.gm_redi.gm_bolus_kappa_face_average', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.slope_density', 'model.physics.lateral_mixing.gm_redi.slope_density', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.msc_stabilize', 'model.physics.lateral_mixing.gm_redi.msc_stabilize', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.slope_limit', 'model.physics.lateral_mixing.gm_redi.slope_limit', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.slope_positions', 'model.physics.lateral_mixing.gm_redi.slope_positions', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.mld_criterion', 'model.physics.lateral_mixing.gm_redi.mld_criterion', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.slope_n2', 'model.physics.lateral_mixing.gm_redi.slope_n2', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.kappa_redi_lat_scaling', 'model.physics.lateral_mixing.gm_redi.kappa_redi_lat_scaling', 'seeded'),
    ('nemo_dino_kamm', 'model.gm_redi.implicit_K33', 'model.physics.lateral_mixing.gm_redi.implicit_K33', 'seeded'),
    ('nemo_dino_kamm', 'model.freezing.scheme', 'model.physics.bottom_drag.scheme', 'seeded'),
    ('nemo_dino_kamm', 'model.physics.bottom_drag.scheme', 'model.physics.convection.scheme', 'seeded'),
    ('nemo_dino_kamm', 'model.physics.convection.scheme', 'model.physics.shortwave_penetration.scheme', 'seeded'),
    ('nemo_dino_kamm', 'model.physics.shortwave_penetration.scheme', 'model.physics.vertical_mixing.scheme', 'seeded'),
    ('nemo_dino_kamm', 'model.physics.vertical_mixing.kpp.K_max', 'model.physics.vertical_mixing.tidal.K_max', 'seeded'),
    ('nemo_dino_kamm', 'model.physics.convection.enhanced_diffusion.K_conv', 'model.physics.vertical_mixing.kpp.K_conv', 'seeded'),
    ('nemo_dino_kamm', 'model.constants.rho_0', 'model.physics.surface_forcing.flux_feedback.rho_0', "inert-but-ARMED: physics.surface_forcing.flux_feedback.rho_0=1025.0 and physics.vertical_mixing.tidal.rho_0=1025.0 both differ from constants.rho_0=1026.0. VERIFIED inert on this card: surface_forcing.scheme='none' and vertical_mixing.tidal.enabled=False, so neither surface is read. BECOMES A LIVE TWO-SURFACE DIVERGENCE the moment tidal mixing is enabled or a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate rho_0/c_sw deliberate (the Veros surface-forcing block owns cp_0) -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 1025.0 is the right value there."),
    ('nemo_dino_kamm', 'model.physics.convection.enhanced_diffusion.cfl_safety', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'seeded'),
    ('nemo_dino_kamm', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'model.physics.lateral_mixing.harmonic.cfl_safety', 'seeded'),
    ('nemo_dino_kamm', 'model.constants.c_sw', 'model.physics.surface_forcing.flux_feedback.c_sw', "inert-but-ARMED: physics.surface_forcing.flux_feedback.c_sw=3994.0 differs from constants.c_sw=3991.86795711963 (the card's own c_p pin). VERIFIED inert on this card: surface_forcing.scheme='none', so the flux_feedback surface is never read. BECOMES A LIVE TWO-SURFACE DIVERGENCE as soon as a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate c_sw deliberate -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 3994.0 is the right value there."),
    ('nemo_dino_kamm_mlf', 'model.lateral_viscosity.A_h', 'model.physics.lateral_mixing.harmonic.A_h', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.K_h', 'model.physics.lateral_mixing.harmonic.K_h', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.A_v', 'model.physics.vertical_mixing.constant.A_v', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.K_v', 'model.physics.vertical_mixing.constant.K_v', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.kappa_GM', 'model.physics.lateral_mixing.gm_redi.kappa_GM', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.kappa_Redi', 'model.physics.lateral_mixing.gm_redi.kappa_Redi', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.treguier.enabled', 'model.gm_redi.visbeck.enabled', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.visbeck.alpha', 'model.physics.vertical_mixing.richardson.alpha', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.treguier.kappa_min', 'model.gm_redi.visbeck.kappa_min', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.visbeck.kappa_min', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_min', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.visbeck.kappa_max', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_max', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.visbeck.n2_mode', 'model.physics.convection.enhanced_diffusion.n2_mode', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.treguier.aei0', 'model.physics.lateral_mixing.gm_redi.treguier.aei0', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.slope_scheme', 'model.physics.lateral_mixing.gm_redi.slope_scheme', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.gm_bolus_advection', 'model.physics.lateral_mixing.gm_redi.gm_bolus_advection', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.gm_bolus_kappa_face_average', 'model.physics.lateral_mixing.gm_redi.gm_bolus_kappa_face_average', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.slope_density', 'model.physics.lateral_mixing.gm_redi.slope_density', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.msc_stabilize', 'model.physics.lateral_mixing.gm_redi.msc_stabilize', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.slope_limit', 'model.physics.lateral_mixing.gm_redi.slope_limit', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.slope_positions', 'model.physics.lateral_mixing.gm_redi.slope_positions', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.mld_criterion', 'model.physics.lateral_mixing.gm_redi.mld_criterion', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.slope_n2', 'model.physics.lateral_mixing.gm_redi.slope_n2', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.kappa_redi_lat_scaling', 'model.physics.lateral_mixing.gm_redi.kappa_redi_lat_scaling', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.gm_redi.implicit_K33', 'model.physics.lateral_mixing.gm_redi.implicit_K33', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.freezing.scheme', 'model.physics.bottom_drag.scheme', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.physics.bottom_drag.scheme', 'model.physics.convection.scheme', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.physics.convection.scheme', 'model.physics.shortwave_penetration.scheme', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.physics.shortwave_penetration.scheme', 'model.physics.vertical_mixing.scheme', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.physics.vertical_mixing.kpp.K_max', 'model.physics.vertical_mixing.tidal.K_max', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.physics.convection.enhanced_diffusion.K_conv', 'model.physics.vertical_mixing.kpp.K_conv', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.constants.rho_0', 'model.physics.surface_forcing.flux_feedback.rho_0', "inert-but-ARMED: physics.surface_forcing.flux_feedback.rho_0=1025.0 and physics.vertical_mixing.tidal.rho_0=1025.0 both differ from constants.rho_0=1026.0. VERIFIED inert on this card: surface_forcing.scheme='none' and vertical_mixing.tidal.enabled=False, so neither surface is read. BECOMES A LIVE TWO-SURFACE DIVERGENCE the moment tidal mixing is enabled or a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate rho_0/c_sw deliberate (the Veros surface-forcing block owns cp_0) -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 1025.0 is the right value there."),
    ('nemo_dino_kamm_mlf', 'model.physics.convection.enhanced_diffusion.cfl_safety', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'model.physics.lateral_mixing.harmonic.cfl_safety', 'seeded'),
    ('nemo_dino_kamm_mlf', 'model.constants.c_sw', 'model.physics.surface_forcing.flux_feedback.c_sw', "inert-but-ARMED: physics.surface_forcing.flux_feedback.c_sw=3994.0 differs from constants.c_sw=3991.86795711963 (the card's own c_p pin). VERIFIED inert on this card: surface_forcing.scheme='none', so the flux_feedback surface is never read. BECOMES A LIVE TWO-SURFACE DIVERGENCE as soon as a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate c_sw deliberate -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 3994.0 is the right value there."),
    ('nemo_paper', 'model.lateral_viscosity.A_h', 'model.physics.lateral_mixing.harmonic.A_h', 'seeded'),
    ('nemo_paper', 'model.K_h', 'model.physics.lateral_mixing.harmonic.K_h', 'seeded'),
    ('nemo_paper', 'model.A_v', 'model.physics.vertical_mixing.constant.A_v', 'seeded'),
    ('nemo_paper', 'model.K_v', 'model.physics.vertical_mixing.constant.K_v', 'seeded'),
    ('nemo_paper', 'model.gm_redi.kappa_GM', 'model.physics.lateral_mixing.gm_redi.kappa_GM', 'seeded'),
    ('nemo_paper', 'model.gm_redi.kappa_Redi', 'model.physics.lateral_mixing.gm_redi.kappa_Redi', 'seeded'),
    ('nemo_paper', 'model.gm_redi.S_max', 'model.physics.lateral_mixing.gm_redi.S_max', 'seeded'),
    ('nemo_paper', 'model.gm_redi.treguier.enabled', 'model.gm_redi.visbeck.enabled', 'seeded'),
    ('nemo_paper', 'model.gm_redi.visbeck.alpha', 'model.physics.vertical_mixing.richardson.alpha', 'seeded'),
    ('nemo_paper', 'model.gm_redi.treguier.kappa_min', 'model.gm_redi.visbeck.kappa_min', 'seeded'),
    ('nemo_paper', 'model.gm_redi.visbeck.kappa_min', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_min', 'seeded'),
    ('nemo_paper', 'model.gm_redi.visbeck.kappa_max', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_max', 'seeded'),
    ('nemo_paper', 'model.gm_redi.visbeck.n2_mode', 'model.physics.convection.enhanced_diffusion.n2_mode', 'seeded'),
    ('nemo_paper', 'model.gm_redi.treguier.aei0', 'model.physics.lateral_mixing.gm_redi.treguier.aei0', 'seeded'),
    ('nemo_paper', 'model.freezing.scheme', 'model.physics.bottom_drag.scheme', 'seeded'),
    ('nemo_paper', 'model.physics.bottom_drag.scheme', 'model.physics.convection.scheme', 'seeded'),
    ('nemo_paper', 'model.physics.convection.scheme', 'model.physics.shortwave_penetration.scheme', 'seeded'),
    ('nemo_paper', 'model.physics.shortwave_penetration.scheme', 'model.physics.vertical_mixing.scheme', 'seeded'),
    ('nemo_paper', 'model.physics.vertical_mixing.kpp.K_max', 'model.physics.vertical_mixing.tidal.K_max', 'seeded'),
    ('nemo_paper', 'model.physics.convection.enhanced_diffusion.K_conv', 'model.physics.vertical_mixing.kpp.K_conv', 'seeded'),
    ('nemo_paper', 'model.constants.rho_0', 'model.physics.surface_forcing.flux_feedback.rho_0', "inert-but-ARMED: physics.surface_forcing.flux_feedback.rho_0=1025.0 and physics.vertical_mixing.tidal.rho_0=1025.0 both differ from constants.rho_0=1026.0. VERIFIED inert on this card: surface_forcing.scheme='none' and vertical_mixing.tidal.enabled=False, so neither surface is read. BECOMES A LIVE TWO-SURFACE DIVERGENCE the moment tidal mixing is enabled or a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate rho_0/c_sw deliberate (the Veros surface-forcing block owns cp_0) -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 1025.0 is the right value there."),
    ('nemo_paper', 'model.physics.convection.enhanced_diffusion.cfl_safety', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'seeded'),
    ('nemo_paper', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'model.physics.lateral_mixing.harmonic.cfl_safety', 'seeded'),
    ('nemo_paper', 'model.constants.c_sw', 'model.physics.surface_forcing.flux_feedback.c_sw', "inert-but-ARMED: physics.surface_forcing.flux_feedback.c_sw=3994.0 differs from constants.c_sw=3991.86795711963 (the card's own c_p pin). VERIFIED inert on this card: surface_forcing.scheme='none', so the flux_feedback surface is never read. BECOMES A LIVE TWO-SURFACE DIVERGENCE as soon as a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate c_sw deliberate -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 3994.0 is the right value there."),
    ('oceananigans', 'model.lateral_viscosity.A_h', 'model.physics.lateral_mixing.harmonic.A_h', 'seeded'),
    ('oceananigans', 'model.K_h', 'model.physics.lateral_mixing.harmonic.K_h', 'seeded'),
    ('oceananigans', 'model.A_v', 'model.physics.vertical_mixing.constant.A_v', 'seeded'),
    ('oceananigans', 'model.K_v', 'model.physics.vertical_mixing.constant.K_v', 'seeded'),
    ('oceananigans', 'model.gm_redi.kappa_GM', 'model.physics.lateral_mixing.gm_redi.kappa_GM', 'seeded'),
    ('oceananigans', 'model.gm_redi.kappa_Redi', 'model.physics.lateral_mixing.gm_redi.kappa_Redi', 'seeded'),
    ('oceananigans', 'model.gm_redi.S_max', 'model.physics.lateral_mixing.gm_redi.S_max', 'seeded'),
    ('oceananigans', 'model.gm_redi.treguier.enabled', 'model.gm_redi.visbeck.enabled', 'seeded'),
    ('oceananigans', 'model.gm_redi.visbeck.alpha', 'model.physics.vertical_mixing.richardson.alpha', 'seeded'),
    ('oceananigans', 'model.gm_redi.treguier.kappa_min', 'model.gm_redi.visbeck.kappa_min', 'seeded'),
    ('oceananigans', 'model.gm_redi.visbeck.kappa_min', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_min', 'seeded'),
    ('oceananigans', 'model.gm_redi.visbeck.kappa_max', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_max', 'seeded'),
    ('oceananigans', 'model.gm_redi.treguier.aei0', 'model.physics.lateral_mixing.gm_redi.treguier.aei0', 'seeded'),
    ('oceananigans', 'model.freezing.scheme', 'model.physics.bottom_drag.scheme', 'seeded'),
    ('oceananigans', 'model.physics.bottom_drag.scheme', 'model.physics.convection.scheme', 'seeded'),
    ('oceananigans', 'model.physics.convection.scheme', 'model.physics.shortwave_penetration.scheme', 'seeded'),
    ('oceananigans', 'model.physics.shortwave_penetration.scheme', 'model.physics.vertical_mixing.scheme', 'seeded'),
    ('oceananigans', 'model.physics.vertical_mixing.kpp.K_max', 'model.physics.vertical_mixing.tidal.K_max', 'seeded'),
    ('oceananigans', 'model.physics.convection.enhanced_diffusion.K_conv', 'model.physics.vertical_mixing.kpp.K_conv', 'seeded'),
    ('oceananigans', 'model.constants.rho_0', 'model.physics.surface_forcing.flux_feedback.rho_0', "inert-but-ARMED: physics.surface_forcing.flux_feedback.rho_0=1025.0 and physics.vertical_mixing.tidal.rho_0=1025.0 both differ from constants.rho_0=1026.0. VERIFIED inert on this card: surface_forcing.scheme='none' and vertical_mixing.tidal.enabled=False, so neither surface is read. BECOMES A LIVE TWO-SURFACE DIVERGENCE the moment tidal mixing is enabled or a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate rho_0/c_sw deliberate (the Veros surface-forcing block owns cp_0) -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 1025.0 is the right value there."),
    ('oceananigans', 'model.physics.convection.enhanced_diffusion.cfl_safety', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'seeded'),
    ('oceananigans', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'model.physics.lateral_mixing.harmonic.cfl_safety', 'seeded'),
    ('oceananigans', 'model.constants.c_sw', 'model.physics.surface_forcing.flux_feedback.c_sw', "inert-but-ARMED: physics.surface_forcing.flux_feedback.c_sw=3994.0 differs from constants.c_sw=3991.86795711963 (the card's own c_p pin). VERIFIED inert on this card: surface_forcing.scheme='none', so the flux_feedback surface is never read. BECOMES A LIVE TWO-SURFACE DIVERGENCE as soon as a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate c_sw deliberate -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 3994.0 is the right value there."),
    ('veros', 'model.lateral_viscosity.A_h', 'model.physics.lateral_mixing.harmonic.A_h', 'seeded'),
    ('veros', 'model.K_h', 'model.physics.lateral_mixing.harmonic.K_h', 'seeded'),
    ('veros', 'model.A_v', 'model.physics.vertical_mixing.constant.A_v', 'seeded'),
    ('veros', 'model.K_v', 'model.physics.vertical_mixing.constant.K_v', 'seeded'),
    ('veros', 'model.gm_redi.kappa_GM', 'model.physics.lateral_mixing.gm_redi.kappa_GM', 'seeded'),
    ('veros', 'model.gm_redi.kappa_Redi', 'model.physics.lateral_mixing.gm_redi.kappa_Redi', 'seeded'),
    ('veros', 'model.gm_redi.S_max', 'model.physics.lateral_mixing.gm_redi.S_max', 'seeded'),
    ('veros', 'model.gm_redi.treguier.enabled', 'model.gm_redi.visbeck.enabled', 'seeded'),
    ('veros', 'model.gm_redi.visbeck.alpha', 'model.physics.vertical_mixing.richardson.alpha', 'seeded'),
    ('veros', 'model.gm_redi.treguier.kappa_min', 'model.gm_redi.visbeck.kappa_min', 'seeded'),
    ('veros', 'model.gm_redi.visbeck.kappa_min', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_min', 'seeded'),
    ('veros', 'model.gm_redi.visbeck.kappa_max', 'model.physics.lateral_mixing.gm_redi.visbeck.kappa_max', 'seeded'),
    ('veros', 'model.gm_redi.treguier.aei0', 'model.physics.lateral_mixing.gm_redi.treguier.aei0', 'seeded'),
    ('veros', 'model.freezing.scheme', 'model.physics.bottom_drag.scheme', 'seeded'),
    ('veros', 'model.physics.bottom_drag.scheme', 'model.physics.convection.scheme', 'seeded'),
    ('veros', 'model.physics.convection.scheme', 'model.physics.shortwave_penetration.scheme', 'seeded'),
    ('veros', 'model.physics.shortwave_penetration.scheme', 'model.physics.vertical_mixing.scheme', 'seeded'),
    ('veros', 'model.physics.vertical_mixing.kpp.K_max', 'model.physics.vertical_mixing.tidal.K_max', 'seeded'),
    ('veros', 'model.physics.convection.enhanced_diffusion.K_conv', 'model.physics.vertical_mixing.kpp.K_conv', 'seeded'),
    ('veros', 'model.constants.rho_0', 'model.physics.surface_forcing.flux_feedback.rho_0', "inert-but-ARMED: physics.surface_forcing.flux_feedback.rho_0=1025.0 and physics.vertical_mixing.tidal.rho_0=1025.0 both differ from constants.rho_0=1026.0. VERIFIED inert on this card: surface_forcing.scheme='none' and vertical_mixing.tidal.enabled=False, so neither surface is read. BECOMES A LIVE TWO-SURFACE DIVERGENCE the moment tidal mixing is enabled or a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate rho_0/c_sw deliberate (the Veros surface-forcing block owns cp_0) -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 1025.0 is the right value there."),
    ('veros', 'model.physics.convection.enhanced_diffusion.cfl_safety', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'seeded'),
    ('veros', 'model.physics.lateral_mixing.biharmonic.cfl_safety', 'model.physics.lateral_mixing.harmonic.cfl_safety', 'seeded'),
    ('veros', 'model.constants.c_sw', 'model.physics.surface_forcing.flux_feedback.c_sw', "inert-but-ARMED: physics.surface_forcing.flux_feedback.c_sw=3994.0 differs from constants.c_sw=3991.86795711963 (the card's own c_p pin). VERIFIED inert on this card: surface_forcing.scheme='none', so the flux_feedback surface is never read. BECOMES A LIVE TWO-SURFACE DIVERGENCE as soon as a surface_forcing scheme is selected. FluxFeedbackConfig's docstring calls its separate c_sw deliberate -- treat that as a POINTER, NOT a verified fact: for a NEMO oracle card it is UNVERIFIED against NEMO whether 3994.0 is the right value there."),
)

# ---------------------------------------------------------------------------
# Invariant B -- every DINO_RECIPES card key lands somewhere in the assembled
# (model_config, physics_config) tree, keyed on CONSTANTS_FLAT_ALIASES for the
# constants family (not on name-matching -- ``c_p`` -> ``c_sw`` is exactly the
# case a name-match check sails past).  A card key that is a BRANCH SELECTOR
# (chooses which sub-config gets built, so its own string value is never
# itself stored as a leaf) or that is read at a DIFFERENT call site than
# ``dino_lat_lon_model_config`` (the per-step forcing helpers) is allow-listed
# here with a verified reason -- each reason was checked against the actual
# consuming code before being added (see PR description for the exact
# call-site line numbers).
HARNESS_ONLY: dict[str, str] = {
    "vertical_coordinate": (
        "branch selector consumed inside dino_lat_lon_model_config / "
        "dino_mpas_model_config construction (zstar vs masked_zco chooses "
        "which vertical-coordinate builder runs); its EFFECT lands in the "
        "assembled tree via different z_coord/dz fields, not the string "
        "itself (dino.py:2212-2234, 3192-3196)"
    ),
    "lateral_tracer_mixing": (
        "branch selector consumed inside dino_lat_lon_model_config (chooses "
        "the geopotential vs isoneutral GM/Redi sub-block); its effect lands "
        "as K_h / gm_redi.enabled leaves, not the string itself "
        "(dino.py:2798-2803, 3047, 3198-3201)"
    ),
    "gm_kappa_scheme": (
        "branch selector consumed inside dino_lat_lon_model_config (chooses "
        "visbeck vs treguier sub-config; enabled=True on the selected "
        "branch); the string itself is not stored as a leaf "
        "(dino.py:2772-2775, 2861-2900, 3266-3268)"
    ),
    "surface_flux_divisor": (
        "consumed by dino_step_surface_forcing (the per-step forcing "
        "helper), a DIFFERENT call site than dino_lat_lon_model_config -- "
        "the assembled (model_cfg, physics_cfg) this test walks never sees "
        "it (dino.py:3620-3629)"
    ),
    "shortwave_penetration_ladder": (
        "consumed by dino_step_surface_forcing (the per-step forcing "
        "helper), a DIFFERENT call site than dino_lat_lon_model_config -- "
        "the assembled (model_cfg, physics_cfg) this test walks never sees "
        "it (dino.py:3667-3675)"
    ),
    "surface_tendency_placement": (
        "VERIFIED 2026-08-13 on the current tree, three facts: (1) CONSUMER "
        "is packages/ocean/legoesm/ocean/experiments/dino.py:3560, "
        "`placement = getattr(cfg, \"surface_tendency_placement\", "
        "\"applied_now\")`, whose ENCLOSING function is "
        "`_check_surface_tendency_placement` (def at dino.py:3547) -- it "
        "reads the value off the DINOConfig object, NOT off the assembled "
        "config; (2) that checker is INVOKED at dino.py:3608, inside "
        "`apply_dino_lat_lon_surface_forcing` (def at dino.py:3576), the "
        "per-step surface-forcing helper -- a DIFFERENT call site than "
        "`dino_lat_lon_model_config`, so the (model_cfg, physics_cfg) pair "
        "this test walks never sees it; (3) it is not a field of the "
        "assembled config at all -- `grep -c surface_tendency_placement` is "
        "0 in BOTH ocean/state.py and ocean/physics/combined.py (the two "
        "modules defining LatLonCGridOceanConfig / OceanPhysicsConfig). "
        "It is a ROUTE ASSERTION, not a routed value: its only effect is to "
        "raise when the declared placement disagrees with the driver's "
        "`return_rate` argument (dino.py:3565-3573); the physics difference "
        "it names is carried by the driver's "
        "`model.step(external_tracer_rate=...)` route "
        "(ocean_model_latlon_cgrid.py:3399, 7899), not by any config leaf. "
        "Same class as the surface_flux_divisor / "
        "shortwave_penetration_ladder entries above. NB those two entries' "
        "reason strings name `dino_step_surface_forcing` as the consumer; "
        "on the current tree surface_flux_divisor is actually read at "
        "dino.py:3658, also inside `apply_dino_lat_lon_surface_forcing` -- "
        "their function name and line ranges have drifted (not corrected "
        "here; the classification itself still holds)."
    ),
}

# ---------------------------------------------------------------------------
# Invariant C -- AST sibling-signature parity + call-site completeness.
#
# (i) Sibling-declaration parity: for the two registered SIBLING_PAIRS, if one
# module declares a card-option-name parameter the other must too. Real,
# currently-true asymmetries (NOT yet fixed) are seeded here as
# (declaring_module, missing_sibling_module, option_name):
# NB the EEN options the original bug was about -- een_q_boundary /
# een_e3f_scheme -- are NOT in this baseline: BOTH barotropic_latlon_cgrid and
# ocean_pe_latlon_cgrid declare both names, i.e. the fix holds and parity is
# clean on the pair that actually matters. Only the entries below survive, each
# verified individually as a legitimate asymmetry rather than a threading gap.
SIBLING_PARITY_BASELINE: tuple[tuple[str, str, str], ...] = (
    # LEGITIMATE: the barotropic solver takes the free-surface gravity-wave
    # speed `g` as an explicit parameter (the barotropic mode IS the fast
    # gravity-wave system); the 3-D PE path receives gravity through the
    # config/constants read path rather than as a bare kwarg on the same
    # function. Different plumbing for the same constant -- and whether the
    # VALUE agrees across surfaces is invariant A's job, not C's.
    ("barotropic_latlon_cgrid", "ocean_pe_latlon_cgrid", "g"),
    # LEGITIMATE: the two paths have SEPARATE, deliberately independent card
    # selectors. The 3-D path dispatches on `vorticity_scheme`
    # ("al81"/"ene"/"ene_total"/"een_total", ocean_pe_latlon_cgrid.py:1974,
    # 1992-1994); the barotropic path dispatches on its own
    # `barotropic_coriolis` ("avg"/"een"/"een_metric") -- itself a
    # DINO_RECIPES card key, itself hardened with an unknown-value raise
    # (barotropic_latlon_cgrid.py:1302-1305). VERIFIED: `vorticity_scheme`
    # appears NOWHERE in barotropic_latlon_cgrid, by design, not by omission.
    ("ocean_pe_latlon_cgrid", "barotropic_latlon_cgrid", "vorticity_scheme"),
    # LEGITIMATE: gm_redi_latlon_cgrid is the full grid-specific tendency;
    # _gm_redi_common is the small shared-math helper module (taper functions,
    # resolution scaling, Visbeck/Treguier kappa formulas) that operates on
    # ALREADY-COMPUTED slope/density inputs, so it genuinely does not need the
    # EOS/gravity/bolus-advection plumbing its caller carries.
    ("gm_redi_latlon_cgrid", "_gm_redi_common", "eos"),
    ("gm_redi_latlon_cgrid", "_gm_redi_common", "eos_depth"),
    ("gm_redi_latlon_cgrid", "_gm_redi_common", "g"),
    ("gm_redi_latlon_cgrid", "_gm_redi_common", "gm_bolus_advection"),
    ("gm_redi_latlon_cgrid", "_gm_redi_common", "gm_bolus_kappa_face_average"),
)

# (ii) Call-site completeness: (rel_path, function_name, option_name, ordinal)
# for a call site the static positional/keyword resolver could NOT prove
# passes the option explicitly. Scoped to UNIQUELY-named declaring functions
# only (a name declared by >1 function in the package is skipped -- resolving
# which overload a call targets needs import-graph resolution this AST walk
# does not do, and guessing produces cross-signature false positives). Most
# entries here are resolver limitations (multi-line calls, defaulted
# keyword-only params the resolver conservatively still flags, wrapped
# positional chains) rather than confirmed threading bugs -- this is a
# TRIPWIRE (a NEW entry appearing is the interesting event), not a proof any
# specific listed site is broken. See PR description for spot-checks.
#
# Keyed on (file, function, param, ordinal), NOT (file, LINE, function,
# param): a bare line number rots on every unrelated edit that shifts lines
# above the call site, and this gate has shipped red from pure line drift
# TWICE (2026-08, both re-pinned below with no code change). ``ordinal`` is
# the 0-based rank, by ascending line number, of a call site among every call
# site in the SAME FILE sharing the same (function, param) pair -- exactly
# the ``seen[call.name]``-style name-and-ordinal join
# ``test_nemo_shared_state_coverage.py`` already uses to survive its own
# oracle's line churn. A call site that moves a few lines without changing
# order relative to its same-triple siblings keeps its ordinal, so it keeps
# matching this baseline without a re-pin.
CALL_SITE_BASELINE: tuple[tuple[str, str, str, int], ...] = (
    ("packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py", "_make_multigrid_preconditioner_banded", "omega", 0),
    ("packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py", "_make_multigrid_preconditioner", "omega", 0),
    ("packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py", "compute_ocean_rho", "eos_depth", 0),
    ("packages/ocean/legoesm/ocean/dynamics/ocean_pe_cdgrid.py", "iterate_eos_and_pressure_anomaly", "eos_depth", 0),
    ("packages/ocean/legoesm/ocean/dynamics/ocean_pe_mpas.py", "iterate_eos_and_pressure_anomaly", "eos_depth", 0),
    # These three are NOT resolver false positives: the enclosing helpers have
    # no `g` / `eos_depth` parameter at all, so the card value is
    # unforwardable and the callee default is taken unconditionally.
    #   ordinal 0 of compute_hydrostatic_pressure/g is inside
    #        `compute_ocean_rho`'s IN-SITU branch (its "geometric" branch
    #        returns earlier, reading `constants.g` directly).
    #   compute_ocean_rho/eos_depth (ordinal 0) and ordinal 1 of
    #        compute_hydrostatic_pressure/g are inside
    #        `compute_ocean_rho_and_pressure`, which takes
    #        (state, z_coord, jacobian, eos_fn) only.
    # SUSPECTED LIVE GAP, deliberately NOT fixed here: the DINO cards
    # `nemo_dino_kamm`/`nemo_dino_kamm_mlf` pin a non-default gravity value
    # (differs from constants.g) and eos_depth="geometric" (default  # const-ok: prose, no literal
    # "insitu"), and reach the convection bridge that calls these two sites
    # via convection_n2_mode="adiabatic". Whether that bridge is actually
    # appended for those cards is further gated by `_fallback_owns_evd`
    # (combined.py:260) which was NOT resolved, so this is PLAUSIBLE, not
    # confirmed. Same class as the open k_profiles.py
    # compute_hydrostatic_pressure/g ordinals 2/3 entries (#1603).
    ("packages/ocean/legoesm/ocean/eos.py", "compute_hydrostatic_pressure", "g", 0),
    ("packages/ocean/legoesm/ocean/eos.py", "compute_ocean_rho", "eos_depth", 0),
    ("packages/ocean/legoesm/ocean/eos.py", "compute_hydrostatic_pressure", "g", 1),
    ("packages/ocean/legoesm/ocean/fidelity/box_heat_budget.py", "gm_redi_tracer_tendency_latlon", "omega", 0),
    ("packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py", "mitgcm_canonical_ocean_config", "barotropic_solver", 0),
    ("packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py", "mitgcm_canonical_ocean_config", "tracer_advection", 0),
    ("packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py", "mitgcm_canonical_ocean_config", "barotropic_solver", 0),
    ("packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py", "mitgcm_canonical_ocean_config", "tracer_advection", 0),
    ("packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py", "build_eady_uniform_setup", "barotropic_solver", 0),
    ("packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py", "build_eady_uniform_setup", "ke_gradient_scheme", 0),
    ("packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py", "build_eady_uniform_setup", "momentum_advection", 0),
    ("packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py", "build_eady_uniform_setup", "tracer_advection", 0),
    ("packages/ocean/legoesm/ocean/init_latlon_cgrid.py", "iterate_eos_and_pressure_anomaly", "eos_depth", 0),
    ("packages/ocean/legoesm/ocean/physics/combined.py", "make_eos_fn", "eos", 0),
    ("packages/ocean/legoesm/ocean/physics/combined.py", "compute_buoyancy_frequency_adiabatic", "g", 0),
    ("packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py", "compute_buoyancy_frequency_adiabatic", "g", 0),
    ("packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py", "compute_hydrostatic_pressure", "g", 0),
    ("packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py", "iterate_eos_and_pressure_anomaly", "eos_depth", 0),
    ("packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py", "iterate_eos_and_pressure_anomaly", "eos_depth", 1),
    ("packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py", "compute_isoneutral_K33_latlon", "eos_depth", 0),
    ("packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py", "iterate_eos_and_pressure_anomaly", "eos_depth", 2),
    ("packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py", "compute_hydrostatic_pressure", "g", 1),
    ("packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_mpas.py", "iterate_eos_and_pressure_anomaly", "eos_depth", 0),
    # MPAS lane: no card that pins `g` (nemo_dino_kamm, nemo_dino_kamm_mlf)
    # selects the MPAS dycore, so no card value is dropped here. Same reason
    # as the already-baselined MPAS siblings ocean_model_mpas.py and
    # mpas_physics.py enhanced_diffusion_convection/g.
    ("packages/ocean/legoesm/ocean/physics/lateral_mixing/mle_mpas.py", "compute_buoyancy_frequency_adiabatic", "g", 0),
    ("packages/ocean/legoesm/ocean/physics/mpas_physics.py", "restoring_surface_forcing", "c_p", 0),
    ("packages/ocean/legoesm/ocean/physics/mpas_physics.py", "compute_ocean_rho", "eos_depth", 0),
    ("packages/ocean/legoesm/ocean/physics/mpas_physics.py", "enhanced_diffusion_convection", "g", 0),
    ("packages/ocean/legoesm/ocean/physics/mpas_physics.py", "enhanced_diffusion_convection", "g", 1),
    ("packages/ocean/legoesm/ocean/physics/surface_forcing/integration.py", "restoring_surface_forcing", "c_p", 0),
    ("packages/ocean/legoesm/ocean/physics/surface_forcing/integration.py", "restoring_surface_forcing", "c_p", 1),
    ("packages/ocean/legoesm/ocean/physics/vertical_mixing/integration.py", "kpp_vertical_mixing", "g", 0),
    ("packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py", "compute_hydrostatic_pressure", "g", 0),
    ("packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py", "compute_hydrostatic_pressure", "g", 1),
    ("packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py", "kpp_vertical_mixing", "g", 0),
    ("packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py", "compute_hydrostatic_pressure", "g", 2),
    ("packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py", "compute_hydrostatic_pressure", "g", 3),
    ("packages/ocean/legoesm/ocean/physics/vertical_mixing/kpp.py", "compute_buoyancy_frequency", "g", 0),
    ("packages/ocean/legoesm/ocean/physics/vertical_mixing/kpp.py", "compute_buoyancy_frequency", "g", 1),
    # MPAS lane, as above: the cards pinning `eos_depth` (nemo_paper,
    # nemo_dino_kamm, nemo_dino_kamm_mlf) and `g` (the two kamm cards) all run
    # the lat-lon C-grid, never MPAS.
    ("packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py", "compute_ocean_rho", "eos_depth", 0),
    ("packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py", "kpp_vertical_mixing", "g", 0),
    ("packages/ocean/legoesm/ocean/physics/vertical_mixing/richardson.py", "compute_buoyancy_frequency_adiabatic", "g", 0),
    ("packages/ocean/legoesm/ocean/physics/vertical_mixing/richardson.py", "compute_buoyancy_frequency", "g", 0),
)

# ---------------------------------------------------------------------------
# Invariant D -- no consumerless field. Every leaf a card explicitly sets must
# have >=1 AST-visible read in packages/ocean OUTSIDE dino.py (its own config
# module). Currently EMPTY -- every one of the 81 DINO_RECIPES override keys
# has a verified outside-dino.py read (keyword-arg name, getattr string
# literal, dispatch string comparison, or bare Name/Attribute). A newly-added
# card option with no consumer anywhere else in the package would appear here.
CONSUMERLESS_BASELINE: tuple[tuple[str, str], ...] = ()
