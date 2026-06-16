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
``nemo_dino_v1`` = the DINO (Kamm et al. 2025) NEMO approximation. These are
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
}

# --- MPAS (Voronoi C-grid) recipes (MPASOceanConfig scheme bundles) ----------

MPAS_RECIPES = {
    # legoESM default linear-EOS MPAS dycore — the global-overturning MPAS recipe.
    "legoesm_linear_mpas_v1": {
        "eos": "linear",
        "tracer_advection": "upwind",
        "pgf_scheme": "centered",
        "barotropic_solver": "explicit_substep",
        "pv_scheme": "enstrophy",
        "implicit_vertical_mixing": True,
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
    cfg = config_cls(**params)
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
