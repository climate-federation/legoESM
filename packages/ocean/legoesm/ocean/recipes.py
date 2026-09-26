"""Named, reusable ocean model recipes — the catalog (#490).

A **recipe** is the model-agnostic SCHEME identity of an ocean model: which EOS,
advection, pressure-gradient, KE-gradient, barotropic solver, Coriolis scheme and
time integrators it uses. It is decoupled from the **setup** (grid, initial
condition, forcing, and resolution-dependent parameters like ``A_h``).

A user picks a recipe BY NAME from this catalog and pairs it with any setup via an
experiment factory::

    from legoesm.ocean.recipes import get_recipe, list_recipes
    list_recipes()                                  # the menu
    global_overturning_model_config(cfg, recipe="veros_faithful_v1")

Each entry is the dict of scheme-selection fields to splat into the matching
``*OceanConfig`` constructor; the experiment factory supplies the setup-dependent
parameters (``physics``, ``A_h``, ``eos_linear``, ``gm_redi``, ...) and any
per-run overrides on top.

Provenance: each recipe is the verified dycore identity of an existing,
validated factory — ``legoesm_linear_v1`` = the global-overturning default,
``eady_weno5_v1`` = the eady_uniform corrected eddy-resolving stack,
``veros_faithful_v1`` = the Veros ACC oracle dycore (``build_acc_recipe``),
``nemo_dino_v1`` = the DINO (Kamm et al. 2025) NEMO approximation,
``omip_nemo_match_mpas_v1`` = the proven OMIP MPAS ico6 NEMO-climate-match dycore
(``nemo_match_mpas_model_config``; SST RMSE 0.84 vs NEMO ORCA1),
``omip_nemo_match_tripole_v1`` = the proven OMIP tripole eORCA025 NEMO-climate-match
dycore (``nemo_match_tripole_model_config``; SST RMSE 1.15). These are
locked against drift by ``tests/ocean/unit/test_recipes.py`` +
``test_recipe_snapshots.py``.

NOTE (the verified-registry caveat, #388): the catalog does NOT validate that an
arbitrary recipe x setup pairing is physically consistent or stable. It only
provides named scheme bundles and a mechanical assembler; compatibility
validation + provenance lineage is the follow-on #388 layer.
"""

from __future__ import annotations

# --- Lat-lon C-grid recipes (LatLonCGridOceanConfig scheme bundles) ----------

LATLON_RECIPES = {
    # The DE-FACTO default dycore most test-matrix experiments run today: the
    # legoESM model defaults with eos="wright" (the matrix scrape path leaves eos
    # at its default). Named here so the ~16 experiments that silently share it
    # are explicit — most did NOT deliberately choose this dycore (#488).
    # Identical to legoesm_linear_v1 except eos.
    "default_wright_v1": {
        "eos": "wright",
        "momentum_advection": "vector_invariant",
        "tracer_advection": "tvd",
        "pgf_scheme": "adcroft",
        "ke_gradient_scheme": "centered",
        "barotropic_solver": "explicit_substep",
        "coriolis_scheme": "matsuno_split",
        "outer_integrator": "forward_euler",
        "tracer_time_integrator": "euler",
        "implicit_vertical_mixing": True,
    },
    # legoESM default linear-EOS dycore — the global-overturning recipe.
    "legoesm_linear_v1": {
        "eos": "linear",
        "momentum_advection": "vector_invariant",
        "tracer_advection": "tvd",
        "pgf_scheme": "adcroft",
        "ke_gradient_scheme": "centered",
        "barotropic_solver": "explicit_substep",
        "coriolis_scheme": "matsuno_split",
        "outer_integrator": "forward_euler",
        "tracer_time_integrator": "euler",
        "implicit_vertical_mixing": True,
        "n_barotropic_substeps": 30,
    },
    # Corrected eddy-resolving WENO5 stack — the eady_uniform recipe.
    "eady_weno5_v1": {
        "eos": "linear",
        "momentum_advection": "weno5",
        "tracer_advection": "weno5",
        "pgf_scheme": "smc03",
        "ke_gradient_scheme": "centered",
        "barotropic_solver": "implicit_cn",
        "outer_integrator": "ab2",
        "tracer_time_integrator": "rk3",
    },
    # Veros ACC oracle dycore (build_acc_recipe, with_surface_forcing=True).
    "veros_faithful_v1": {
        "eos": "veros_nonlin2",
        "momentum_advection": "flux_form",
        "momentum_flux_scheme": "centered",
        "tracer_advection": "centered",
        "pgf_scheme": "adcroft",
        "ke_gradient_scheme": "centered",
        "barotropic_solver": "rigid_lid",
        "coriolis_scheme": "explicit_ab2",
        "outer_integrator": "ab2",
        "tracer_time_integrator": "euler",
        "ab2_scope": "advective",
        "momentum_friction_additive": True,
        "implicit_vmix_dzw_slot": True,
        "implicit_vertical_mixing": True,
        "lateral_viscosity_operator": "flux_divergence",
        "vertical_momentum_scheme": "centered_full",
    },
    # NEMO-STYLE dycore (#487/#496; RENAMED from "nemo_v1" 2026-09-02 — that
    # name overclaimed fidelity: this entry's barotropic solver is legoESM's
    # own generic split-explicit arm, NOT NEMO's dyn_spg_ts, since it never
    # sets any of the 5 barotropic_*_evaluation selectors — see
    # docs/ocean/fidelity/nemo_branch_isomorphism_map.md S-16) — the scheme
    # identity of fidelity/nemo_recipe.py's nemo_lat_lon_model_config:
    # EEN-style vector-invariant momentum + Hollingsworth KE-gradient,
    # PPM/FCT tracers, TEOS-10-like veros_gsw EOS, smc03 PGF, RK3 momentum,
    # adaptive-implicit vertical advection, cosine free-surface filter (all
    # genuinely NEMO-referenced) riding on the generic (non-NEMO) barotropic
    # substep loop. NB this catalog entry is the DYCORE SCHEMES only — the
    # full NEMO card ALSO supplies NEMO constants + the prognostic TKE
    # physics + GM/Redi/MLE; use nemo_recipe.build_nemo_recipe for full
    # fidelity. Distinct from nemo_dino_v1 (the cruder Wright/KPP approx).
    # Drift-guarded against the card by tests/ocean/unit/test_recipes.py.
    "legoesm_nemo_like_v1": {
        "eos": "veros_gsw",
        "momentum_advection": "vector_invariant",
        "ke_gradient_scheme": "hollingsworth",
        "tracer_advection": "ppm_fct",
        "pgf_scheme": "smc03",
        "barotropic_solver": "explicit_substep",
        "barotropic_time_filter": "cosine",
        "momentum_time_integrator": "rk3",
        "adaptive_implicit_vertadv": True,
        "implicit_vertical_mixing": True,
        "n_barotropic_substeps": 30,
    },
    # DINO (Kamm et al. 2025) NEMO double-gyre approximation — see dino.py for
    # the NEMO-namelist cross-checks (ke_gradient/barotropic ARE NEMO matches;
    # the forward_euler/euler integrators are a known NEMO-unfaithfulness, #487).
    "nemo_dino_v1": {
        "eos": "wright",
        "momentum_advection": "vector_invariant",
        "tracer_advection": "tvd",
        "pgf_scheme": "adcroft",
        "ke_gradient_scheme": "hollingsworth",
        "barotropic_solver": "implicit_cn",
        "coriolis_scheme": "matsuno_split",
        "outer_integrator": "forward_euler",
        "tracer_time_integrator": "euler",
        "implicit_vertical_mixing": True,
        "A_h_lat_scaling": True,
    },
    # Proven OMIP tripole eORCA025 (¼°) NEMO-CLIMATE-match dycore — the tripole
    # branch of run_omip.py::_create_setup (nemo_match_tripole_model_config).
    # Matches NEMO ORCA1 CLIMATE: SST RMSE 1.15, corr 0.99
    # (docs/dev-notes/ocean_faithfulness_nemo.md) — a climate-match stack
    # (KPP vertical mixing + Laplacian Smagorinsky), NOT a numerics-faithful card
    # like legoesm_nemo_like_v1. ke_gradient/outer/tracer integrators are config
    # defaults the tripole branch intentionally leaves unset (the Hollingsworth
    # KE stencil is not yet north-fold-aware — see the _create_setup tripole
    # note).
    "omip_nemo_match_tripole_v1": {
        "eos": "wright",
        "momentum_advection": "vector_invariant",
        "tracer_advection": "tvd",
        "pgf_scheme": "adcroft",
        "ke_gradient_scheme": "centered",
        "barotropic_solver": "implicit_cn",
        "coriolis_scheme": "matsuno_split",
        "outer_integrator": "forward_euler",
        "tracer_time_integrator": "euler",
        "implicit_vertical_mixing": True,
        "n_barotropic_substeps": 30,
        # OMIP global freshwater correction — part of the proven configuration
        # (run_omip _create_setup + nemo_match_tripole_model_config both set it);
        # carried here so catalog-based assembly does not silently fall back to
        # the config default False (codex).
        "normalize_freshwater": True,
    },
    # Oracle dycores — the canonical numerics of the runnable oracle models. SINGLE
    # SOURCE for the *_canonical_ocean_config factory defaults (the factories splat
    # get_recipe(...) then add per-deck dimensional args; their parameterized scheme
    # args override these per deck). See ocean/fidelity/{oceananigans,mitgcm}_recipe.py.
    "oceananigans_v1": {
        "eos": "linear",
        "momentum_advection": "vector_invariant",   # VectorInvariant() [APPROX]
        "tracer_advection": "weno7",                 # WENO(order=7)
        "coriolis_scheme": "explicit_ab2",           # spherical Coriolis [APPROX]
        "barotropic_solver": "implicit_cn",          # ImplicitFreeSurface
        "outer_integrator": "ab2",
        # STABILITY HAZARD (diagnosed via the DINO 'oceananigans'-card barotropic
        # blowup): explicit_ab2 × implicit_cn WITHOUT barotropic_slow_forcing_ab2
        # integrates the barotropic-mode Coriolis FORWARD EULER (the CN predictor
        # gates its FB Coriolis off and the outer AB2 excludes the barotropic
        # increment) — unconditionally unstable, |G|=sqrt(1+(f·dt)²)/step.  Long
        # basin runs MUST add barotropic_slow_forcing_ab2=True (Oceananigans
        # AB2-extrapolates Gᵁ INCLUDING Coriolis) — the DINO 'oceananigans' card
        # and the Silvestri §5 jet do.  NOT pinned here because the fidelity
        # compare decks (oceananigans_canonical_ocean_config consumers) override
        # coriolis_scheme per-deck (matsuno decks would trip the flag's
        # explicit_ab2 validation) and one passes the flag via **overrides
        # (duplicate-kwarg).  Short compare decks tolerate the weak growth.
        "lateral_viscosity_operator": "flux_divergence",
        "A_h_lat_scaling": False,
        "C_smag": 0.0,
        "differentiable_barotropic": True,
        "use_conservation_fixer": False,
        "enable_runtime_checks": False,
        "weno_vertadv_full_velocity": True,
    },
    "mitgcm_v1": {
        "eos": "linear",
        "momentum_advection": "flux_form",
        "momentum_flux_scheme": "centered",          # MITgcm centered momentum
        "coriolis_scheme": "explicit_ab2",           # face-f
        "coriolis_energy_conserving": False,
        "outer_integrator": "ab2",
        "ab2_scope": "total",
        "barotropic_implicit_theta_eta": 1.0,        # fully backward-Euler free surface
        "barotropic_implicit_theta_pgf": 1.0,
        "lateral_viscosity_operator": "flux_divergence",
        "A_h_lat_scaling": False,
        "C_smag": 0.0,
        "bottom_drag_r": 0.0,
        "differentiable_barotropic": True,
        "use_conservation_fixer": False,
        "enable_runtime_checks": False,
    },
}

# --- MPAS (Voronoi C-grid) recipes (MPASOceanConfig scheme bundles) ----------

MPAS_RECIPES = {
    # The de-facto default MPAS dycore (legoESM MPAS defaults + eos="wright") —
    # what an MPAS scrape-path experiment runs. Identical to legoesm_linear_mpas_v1
    # except eos. The MPAS sibling of default_wright_v1.
    "default_wright_mpas_v1": {
        "eos": "wright",
        "tracer_advection": "upwind",
        "pgf_scheme": "centered",
        "barotropic_solver": "explicit_substep",
        "pv_scheme": "enstrophy",
        "implicit_vertical_mixing": True,
    },
    # legoESM default linear-EOS MPAS dycore — the global-overturning MPAS recipe.
    "legoesm_linear_mpas_v1": {
        "eos": "linear",
        "tracer_advection": "upwind",
        "pgf_scheme": "centered",
        "barotropic_solver": "explicit_substep",
        "pv_scheme": "enstrophy",
        "implicit_vertical_mixing": True,
    },
    # Proven OMIP MPAS ico6 (~115 km ≈ ORCA1) NEMO-CLIMATE-match dycore — the
    # mpas branch of run_omip.py::_create_setup (nemo_match_mpas_model_config).
    # Matches NEMO ORCA1 CLIMATE: SST RMSE 0.84 (the best grid)
    # (docs/dev-notes/ocean_faithfulness_nemo.md) — a climate-match stack
    # (KPP vertical mixing + Laplacian Smagorinsky), NOT a numerics-faithful card
    # like legoesm_nemo_like_v1. eos/pv_scheme are the config defaults the mpas
    # branch leaves unset; tracer_advection/pgf/barotropic/implicit are set
    # explicitly.
    "omip_nemo_match_mpas_v1": {
        "eos": "wright",
        "tracer_advection": "tvd",
        "pgf_scheme": "adcroft",
        "barotropic_solver": "implicit_cn",
        "pv_scheme": "enstrophy",
        "implicit_vertical_mixing": True,
        # Part of the proven configuration (see the tripole entry note).
        "normalize_freshwater": True,
    },
}

_TABLES = {"latlon": LATLON_RECIPES, "mpas": MPAS_RECIPES}


def list_recipes(kind: str | None = None) -> list[str]:
    """Return the sorted available recipe names.

    With ``kind=None`` (the public menu), returns the union of lat-lon and MPAS
    recipe names. Pass ``kind="latlon"`` or ``kind="mpas"`` when a selector must
    restrict lookup to a specific config family.
    """
    if kind is None:
        return sorted({name for table in _TABLES.values() for name in table})
    if kind not in _TABLES:
        raise ValueError(
            f"unknown recipe kind {kind!r}; choose from {sorted(_TABLES)}")
    return sorted(_TABLES[kind])


def get_recipe(name: str, kind: str | None = None) -> dict:
    """Return a COPY of the named recipe's scheme bundle.

    Splat the result into the matching ``*OceanConfig`` constructor; the caller
    supplies setup-dependent params (physics, A_h, eos_linear, gm_redi, ...).

    Raises ``ValueError`` on an unknown ``kind`` or ``name`` (no silent default —
    a typo must fail loudly). With ``kind=None``, searches both catalogs.
    """
    if kind is None:
        matches = [table[name] for table in _TABLES.values() if name in table]
        if len(matches) == 1:
            return dict(matches[0])
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous recipe {name!r}; pass kind= to choose from "
                f"{sorted(_TABLES)}")
        raise ValueError(
            f"unknown recipe {name!r}; choose from {list_recipes()}")
    if kind not in _TABLES:
        raise ValueError(
            f"unknown recipe kind {kind!r}; choose from {sorted(_TABLES)}")
    table = _TABLES[kind]
    if name not in table:
        raise ValueError(
            f"unknown {kind} recipe {name!r}; choose from {sorted(table)}")
    return dict(table[name])


def assemble_ocean_config(recipe_bundle, config_cls, *,
                          overrides=None, validate=True, **setup_params):
    """Assemble a ``*OceanConfig`` from a recipe bundle + setup parameters.

    This is the pure ``assemble(recipe_bundle, setup)`` entry point (#490): the
    **recipe bundle** supplies the scheme identity; ``setup_params`` supply the
    resolution/experiment-dependent values (``physics``, ``A_h``, ``gm_redi``,
    ``eos_linear``, ...); ``overrides`` win last.

    Parameters
    ----------
    recipe_bundle : mapping
        Scheme-identity bundle, usually from :func:`get_recipe`.
    config_cls : type
        The ``*OceanConfig`` class to construct (``LatLonCGridOceanConfig`` /
        ``MPASOceanConfig``); passed in so this module stays config-class-agnostic.
    overrides : dict or None
        Per-run field overrides, applied last (win over recipe + setup).
    validate : bool
        Run the verified-registry compatibility check on the assembled config
        (default True). Pass ``validate=False`` to skip it.
    **setup_params
        Setup-dependent ``config_cls`` fields (``physics``, ``A_h``, ``A_v``,
        ``K_v``, ``bottom_drag_r``, ``eos_linear``, ``gm_redi``, ...).

    Returns
    -------
    config_cls instance

    Notes
    -----
    **EOS-config matching** (the rung that lets ANY recipe pair with a setup):
    the setup's ``eos_linear`` is attached ONLY when the recipe's ``eos`` is
    ``"linear"``. A non-linear EOS scheme (``veros_nonlin2`` / ``wright`` / ...)
    carries its own fixed coefficients, so a dangling ``eos_linear`` is dropped —
    i.e. ``global_overturning`` (a linear setup) x ``veros_faithful_v1`` assembles
    the *Veros* EOS rather than a linear-config-attached-to-a-non-linear-EOS mix.
    The assembled config is then run through
    :func:`assert_recipe_setup_compatible` (#490 verified-registry rung 1).
    """
    params = dict(recipe_bundle)
    params.update(setup_params)
    if overrides:
        params.update(overrides)
    # EOS-config matching: a non-linear recipe ignores eos_linear, so drop it
    # (a dangling linear config under a non-linear EOS signals a recipe x setup
    # mismatch the validator below would otherwise reject).
    if params.get("eos") != "linear":
        params.pop("eos_linear", None)
    # #501: recipe bundles carry FLAT field names (e.g. ``barotropic_solver``)
    # that are now nested sub-configs on grouped configs.  Route through
    # ``from_flat`` when the target config exposes it (LatLonCGridOceanConfig)
    # so flat bundle keys distribute into their sub-configs; configs without
    # grouping (cube/MPAS/spectral) fall back to the plain constructor.  Matches
    # the config.py to_ocean_config / _decode_config from_flat pattern; the PR2
    # one-time codemod missed this call because it is on a *variable*, not the
    # literal ``LatLonCGridOceanConfig(``.
    _ctor = getattr(config_cls, "from_flat", config_cls)
    cfg = _ctor(**params)
    if validate:
        assert_recipe_setup_compatible(cfg)
    return cfg


def assert_recipe_setup_compatible(config) -> None:
    """Fail loudly when an assembled recipe x setup is internally inconsistent
    (#490 verified-registry, rung 1).

    Enforces EOS consistency — the only scheme-level config the ``*OceanConfig``
    objects carry today (``eos`` + ``eos_linear``):

    * ``eos="linear"`` requires an ``eos_linear`` config — else the model falls
      back to DEFAULT ``LinearEOSConfig`` coefficients (wrong for any setup with
      non-default ``alpha_T`` etc.);
    * a non-linear ``eos`` must NOT be paired with an ``eos_linear`` config (it is
      ignored — its presence means a non-linear recipe was forced onto a
      linear-EOS setup).

    Extend with further rules (GM/Redi, grid-feature support, ...) as recipes
    carry more scheme-level configs. Raises ``ValueError`` on a mismatch.
    """
    eos = getattr(config, "eos", None)
    eos_linear = getattr(config, "eos_linear", None)
    if eos == "linear" and eos_linear is None:
        raise ValueError(
            "recipe x setup mismatch: eos='linear' but no eos_linear config was "
            "supplied — the model would run with DEFAULT LinearEOSConfig "
            "coefficients. Supply the setup's eos_linear (e.g. via eos_config=).")
    if eos is not None and eos != "linear" and eos_linear is not None:
        raise ValueError(
            f"recipe x setup mismatch: eos={eos!r} (a non-linear EOS) was paired "
            "with a linear eos_linear config, which it IGNORES. Use a linear-EOS "
            f"recipe, or a setup whose EOS matches eos={eos!r}.")
