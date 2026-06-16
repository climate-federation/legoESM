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

NOTE (the verified-registry caveat, #388): the catalog does NOT yet validate that
an arbitrary recipe x setup pairing is physically consistent — e.g. selecting a
non-linear-EOS recipe under a setup that supplies a linear ``eos_linear`` is a
mismatch the assembler will not catch. Compatibility validation + provenance
lineage is the follow-on #388 layer.
"""

from __future__ import annotations

# --- Lat-lon C-grid recipes (LatLonCGridOceanConfig scheme bundles) ----------

LATLON_RECIPES = {
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


def list_recipes(kind: str = "latlon") -> list[str]:
    """Return the sorted names of available recipes for a grid ``kind``
    (``"latlon"`` or ``"mpas"``) — the menu a user picks from."""
    if kind not in _TABLES:
        raise ValueError(
            f"unknown recipe kind {kind!r}; choose from {sorted(_TABLES)}")
    return sorted(_TABLES[kind])


def get_recipe(name: str, kind: str = "latlon") -> dict:
    """Return a COPY of the named recipe's scheme bundle.

    Splat the result into the matching ``*OceanConfig`` constructor; the caller
    supplies setup-dependent params (physics, A_h, eos_linear, gm_redi, ...).

    Raises ``ValueError`` on an unknown ``kind`` or ``name`` (no silent default —
    a typo must fail loudly).
    """
    if kind not in _TABLES:
        raise ValueError(
            f"unknown recipe kind {kind!r}; choose from {sorted(_TABLES)}")
    table = _TABLES[kind]
    if name not in table:
        raise ValueError(
            f"unknown {kind} recipe {name!r}; choose from {sorted(table)}")
    return dict(table[name])


def assemble_ocean_config(recipe, kind, config_cls, *,
                          eos_linear=None, overrides=None, **setup_params):
    """Assemble a ``*OceanConfig`` from a named recipe + setup parameters.

    This is the ``assemble(recipe, setup)`` entry point (#490): the **recipe**
    (a catalog name) supplies the SCHEME identity; ``setup_params`` supply the
    resolution/experiment-dependent values (``physics``, ``A_h``, ``gm_redi``,
    ...); ``overrides`` win last; the result is compatibility-validated.

    **EOS-config matching** is the rung-2 piece that lets ANY recipe pair with a
    setup: the setup's ``eos_linear`` is attached ONLY when the recipe's ``eos``
    is ``"linear"``. A non-linear EOS scheme (``veros_nonlin2`` / ``wright`` /
    ...) carries its own fixed coefficients, so ``eos_linear`` is irrelevant and
    is dropped — i.e. ``global_overturning`` (a linear setup) x
    ``veros_faithful_v1`` now assembles the *Veros* EOS, instead of a
    half-configured linear-config-attached-to-a-nonlinear-EOS mix.

    Parameters
    ----------
    recipe : str
        Catalog recipe name (see :func:`list_recipes`).
    kind : str
        ``"latlon"`` or ``"mpas"`` (selects the catalog table).
    config_cls : type
        The ``*OceanConfig`` class to construct (``LatLonCGridOceanConfig`` /
        ``MPASOceanConfig``); passed in so this module stays config-class-agnostic.
    eos_linear : LinearEOSConfig or None
        The setup's linear-EOS coefficients; attached only for a linear recipe.
    overrides : dict or None
        Per-run field overrides, applied last (win over recipe + setup).
    **setup_params
        Setup-dependent ``config_cls`` fields (``physics``, ``A_h``, ``A_v``,
        ``K_v``, ``bottom_drag_r``, ``gm_redi``, ...).

    Returns
    -------
    config_cls instance

    Raises
    ------
    ValueError
        On an unknown recipe, or a config the compatibility check rejects.
    """
    params = get_recipe(recipe, kind)
    if params.get("eos") == "linear" and eos_linear is not None:
        params["eos_linear"] = eos_linear
    params.update(setup_params)
    if overrides:
        params.update(overrides)
    cfg = config_cls(**params)
    assert_recipe_setup_compatible(cfg)
    return cfg


def assert_recipe_setup_compatible(config) -> None:
    """Fail loudly when an assembled recipe x setup is internally inconsistent.

    This is the first rung of the verified-registry layer (#388/#490): a recipe
    selects schemes, a setup supplies the matching scheme-level configs, and a
    mismatched pairing must NOT silently run the wrong physics.

    Currently enforces EOS consistency — the only scheme-level config the
    ``*OceanConfig`` objects carry today (``eos`` + ``eos_linear``):

    * ``eos="linear"`` requires an ``eos_linear`` config — otherwise the model
      falls back to the DEFAULT ``LinearEOSConfig`` coefficients (wrong for any
      setup with non-default ``alpha_T`` etc.).
    * a non-linear ``eos`` (``"wright"`` / ``"veros_nonlin2"`` / ...) must NOT be
      paired with an ``eos_linear`` config: the non-linear EOS ignores it, so its
      presence means a non-linear *recipe* was paired with a linear-EOS *setup* —
      almost always a mistake (e.g. ``global_overturning`` setup x
      ``veros_faithful_v1`` recipe).

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
            "with a linear eos_linear config, which it IGNORES. This usually means "
            "a non-linear recipe was run on a linear-EOS setup (e.g. "
            "global_overturning x veros_faithful_v1). Use a linear-EOS recipe for "
            f"this setup, or supply a setup whose EOS matches eos={eos!r}.")
