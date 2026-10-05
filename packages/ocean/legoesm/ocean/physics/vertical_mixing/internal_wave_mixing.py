"""Internal wave-driven vertical mixing (NEMO zdfiwm; de Lavergne et al. 2020).

Faithful port of ``nemo_5.0.1/src/OCE/ZDF/zdfiwm.F90`` (the stock module
the ORCA1 oracle build compiles; ``ln_zdfiwm = .true.`` in the ORCA1
namelist with ``ln_mevar = ln_tsdiff = .false.``).

The scheme distributes four 2-D maps of column-integrated internal-wave
power [W/m²] over the vertical with fixed structure functions, converts
the resulting local dissipation ``ε`` [W/kg] into a diffusivity through
the turbulence-intensity parameter ``Reb = ε/(ν·N²)``, and ADDS the
result to the closure's ``avt``/``avs``/``avm``:

1. ``ecri`` — bottom-intensified dissipation at topographic slopes,
   exponential decay above the seafloor with e-folding scale
   ``1/hcri_inv``::

       ε_cri(w_k) = ecri/ρ0 · [e^{(z_t(k+1)−H)·hcri_inv} − e^{(z_t(k)−H)·hcri_inv}]
                    / (1 − e^{−H·hcri_inv}) / e3w_k

2. ``ebot`` — bottom-intensified dissipation above abyssal hills,
   algebraic decay::

       ε_bot(w_k) = ebot/ρ0 · (1 + hbot/H)
                    · [ (1 + (H−z_t(k+1))/hbot)⁻¹ − (1 + (H−z_t(k))/hbot)⁻¹ ] / e3w_k

3. ``ensq`` — dissipation scaling with N²::

       ε_nsq(w_k) = ensq/ρ0 · max(0,N²_k) / Σ_k e3w_k·max(0,N²_k)

4. ``esho`` — shoaling internal tides, scaling with N::

       ε_sho(w_k) = esho/ρ0 · √max(0,N²_k) / Σ_k e3w_k·√max(0,N²_k)

Then ``Reb = ε / max(1e-20, ν·N²)`` and, with the constant mixing
efficiency Γ = 1/6 (``ln_mevar = .false.``), ``K = Reb·ν/6``; the
variable-efficiency option modifies the energetic (``Reb > 480``) and
buoyancy-controlled (``Reb < 10.224``) regimes.  ``K`` is bounded to
``[1.4e-7, 1e-2] m²/s`` and added to ``avt``/``avs``/``avm`` alike;
``ln_tsdiff`` additionally scales the SALINITY diffusivity by a
Reb-dependent ratio (Jackson & Rehmann-type differential mixing).

References
----------
de Lavergne, C., et al. (2020). A parameterization of local and remote
tidal mixing. *JAMES*, 12, https://doi.org/10.1029/2020MS002065

de Lavergne, C., et al. (2016). *JPO*, 46,
https://doi.org/10.1175/JPO-D-14-0259.1
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants

__physics_contract__ = {
    "summary": (
        "Internal wave-driven vertical mixing (NEMO zdfiwm, de Lavergne "
        "2020): distributes 2-D internal-wave power maps [W/m²] over the "
        "column with cri/bot/nsq/sho structure functions, converts the "
        "dissipation to a diffusivity via Reb = eps/(nu*N^2), and returns "
        "an ADDITIVE (K, ratio) contribution for avt/avs/avm."
    ),
    "inputs": {
        "forcing.ebot/ecri/ensq/esho": "W/m^2 (column-integrated power)",
        "forcing.hbot": "m (abyssal-hill decay scale)",
        "forcing.hcri_inv": "1/m (INVERSE topographic-slope decay scale)",
        "depth_cell": "m (cell-centre depth, positive down; NEMO gdept)",
        "dz_w": "m (interface-to-interface spacing; NEMO e3w, interior)",
        "H": "m (column depth; NEMO ht)",
        "N2": "1/s^2 (Brunt-Väisälä frequency² at interior interfaces)",
        "rho_0": "kg/m^3 (reference density; recipe ConstantsConfig.rho_0)",
    },
    "outputs": {
        "K_wave": "m^2/s (additive diffusivity/viscosity at interior interfaces)",
        "av_ratio": "1 (S/T diffusivity ratio; 1 when tsdiff=False)",
    },
    "sign_convention": (
        "K_wave >= k_min > 0 at every interior interface (a diffusivity, "
        "added to avt/avs/avm — never subtracted); depth positive down."
    ),
    # A diffusivity provider — no conserved quantity of its own.  (The
    # column integral of rho0·eps·e3w recovering the input power maps is
    # the scheme's internal-consistency property, verified in tests.)
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "NEMO 5.0.1 zdfiwm.F90; de Lavergne et al. 2020 JAMES "
        "doi:10.1029/2020MS002065"
    ),
    "idealized_test": (
        "Uniform-N² column: nsq/sho parts integrate exactly to their map "
        "power; cri/bot telescope to power·(1 − surface-residual); "
        "Reb regimes reproduce the F90 mevar branches; ratio → 1 at high "
        "Reb, → 0.505 − 0.495 at low Reb."
    ),
}

# --- Reb-regime mixing-efficiency fit (de Lavergne et al. 2020; zdfiwm.F90) ---
_REB_ENERGETIC = 480.0        # Reb above which efficiency drops (√Reb regime)
_REB_ENERGETIC_COEF = 3.6515  # K = 3.6515·ν·√Reb  (zdfiwm.F90:222)
_REB_BUOYANCY = 10.224        # Reb below which flux is buoyancy-controlled
_REB_BUOYANCY_COEF = 0.052125  # K = 0.052125·ν·Reb^1.5 (zdfiwm.F90:224)
# --- S/T differential-diffusion ratio fit (Jackson & Rehmann; zdfiwm.F90:239) ---
_TSDIFF_BASE = 0.505
_TSDIFF_AMP = 0.495
_TSDIFF_TANH_SLOPE = 0.92
_TSDIFF_LOG10_OFFSET = 0.60
_TSDIFF_REB_PREFAC = 5.0 / 6.0   # zdfiwm.F90: zReb * 5 * r1_6
# NEMO zdfiwm denominators' regularisation floor
_EPS_DENOM = 1.0e-20

__param_spec__ = {
    "IWMConfig": {
        "scheme_key": "ocean.vm.iwm",
        "excluded": {
            "nu_molecular": "physical constant (molecular viscosity reference)",
            "k_min": "numerics: bound (molecular diffusivity floor)",
            "k_max": "numerics: bound (NEMO 100 cm²/s cap)",
            "power_bot_wm2": "boundary-default: default 1e-10 = ~zero uniform-fallback "
            "placeholder (production reads a spatial abyssal-hill power map); at the "
            "lower bound of (0, 5e-3), so it has no interior sigmoid seed",
            "power_cri_wm2": "boundary-default: default 1e-10 = ~zero uniform-fallback "
            "placeholder (production reads a spatial critical-slope power map); at the "
            "lower bound of (0, 5e-3), so it has no interior sigmoid seed",
            "power_sho_wm2": "boundary-default: default 1e-10 = ~zero uniform-fallback "
            "placeholder (production reads a spatial shoaling power map); at the lower "
            "bound of (0, 5e-3), so it has no interior sigmoid seed",
        },
        "params": {
            "power_nsq_wm2": {
                "units": "W/m^2", "bounds": (0.0, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "vertical_mixing",
                "reference": "de Lavergne 2020 N²-scaled power (uniform fallback)",
                "shape": None,
            },
            "scale_bot_m": {
                "units": "m", "bounds": (10.0, 1000.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "vertical_mixing",
                "reference": "de Lavergne 2020 abyssal-hill decay scale (uniform fallback)",
                "shape": None,
            },
            "scale_cri_m": {
                "units": "m", "bounds": (10.0, 1000.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "vertical_mixing",
                "reference": "de Lavergne 2020 critical-slope decay scale (uniform fallback)",
                "shape": None,
            },
        },
    },
}


class IWMConfig(NamedTuple):
    """Internal wave-driven mixing configuration (NEMO namzdf_iwm).

    ADDITIVE to the primary vertical-mixing scheme — applied inside
    :func:`..k_profiles.compute_vertical_K_profiles` AFTER the closure
    (exactly NEMO's zdfphy ordering: zdf_tke then zdf_iwm adding onto
    avt/avs/avm).  Default off preserves bit-exact legacy behaviour.

    The six ``power_*``/``scale_*`` fields are the UNIFORM constant-power
    fallback used when no :class:`IWMForcing` maps are supplied (defaults
    = the zdfiwm_init hard-coded pre-read values); production ORCA1-style
    runs load the de Lavergne maps (``zdfiwm_forcing_TRA.nc``) instead.

    NEMO note: with ``ln_zdfiwm`` NEMO forces the TKE background to
    molecular values (``avmb = 1.4e-6``, ``avtb = 1e-10``) since the
    wave field now provides the interior background.  legoESM keeps the
    configured backgrounds — set the model/TKE backgrounds down
    explicitly for a strictly faithful configuration.
    """
    enabled: bool = False
    mevar: bool = False       # ln_mevar (ORCA1: .false. — constant Γ=1/6)
    tsdiff: bool = False      # ln_tsdiff (ORCA1: .false. — avs == avt)
    nu_molecular: float = constants.nu_ocean_molecular  # [m²/s] (NEMO rnu)
    k_min: float = constants.kappa_T_ocean_molecular    # [m²/s] K bound (1.4e-7)
    k_max: float = 1.0e-2                               # [m²/s] K bound (100 cm²/s)
    power_bot_wm2: float = 1.0e-10   # uniform fallback [W/m²]
    power_cri_wm2: float = 1.0e-10   # uniform fallback [W/m²]
    power_nsq_wm2: float = 1.0e-5    # uniform fallback [W/m²]
    power_sho_wm2: float = 1.0e-10   # uniform fallback [W/m²]
    scale_bot_m: float = 100.0       # uniform fallback [m]
    scale_cri_m: float = 100.0       # uniform fallback [m]
    # N² fed to the wave formula: "insitu" (legacy; in-situ density contrast,
    # carries compressibility so a neutral layer never reads N²<=0) or
    # "nemo_bn2" (NEMO zdfiwm reads rn2 = eosbn2 bn2, zdfiwm.F90:185-211).
    n2_mode: str = "insitu"
    n2_eos_form: str = "seos"        # alpha/beta for nemo_bn2: seos | teos10
    # A card that reads real de Lavergne maps sets this, so a host that
    # forgets to thread them gets a refusal instead of the uniform
    # constant-power fallback, which is different physics.
    require_forcing_maps: bool = False


class IWMForcing(NamedTuple):
    """Static 2-D internal-wave power / decay-scale maps (one per column).

    Same fields as NEMO's ``zdfiwm_forcing`` file after ``zdfiwm_init``:
    note ``hcri_inv`` stores the INVERSE of the file's ``scale_cri``
    (zdfiwm.F90:421 ``hcri_iwm = 1 / scale_cri``).
    """
    ebot: jnp.ndarray      # abyssal-hill power [W/m²]
    ecri: jnp.ndarray      # critical-slope power [W/m²]
    ensq: jnp.ndarray      # N²-scaled power [W/m²]
    esho: jnp.ndarray      # shoaling power [W/m²]
    hbot: jnp.ndarray      # abyssal-hill decay scale [m]
    hcri_inv: jnp.ndarray  # INVERSE critical-slope decay scale [1/m]


def uniform_iwm_forcing(
    cfg: IWMConfig, shape: tuple, dtype=jnp.float64,
) -> IWMForcing:
    """Constant-power fallback maps from the config scalars."""
    full = lambda v: jnp.full(shape, v, dtype=dtype)  # noqa: E731
    return IWMForcing(
        ebot=full(cfg.power_bot_wm2),
        ecri=full(cfg.power_cri_wm2),
        ensq=full(cfg.power_nsq_wm2),
        esho=full(cfg.power_sho_wm2),
        hbot=full(cfg.scale_bot_m),
        hcri_inv=full(1.0 / cfg.scale_cri_m),
    )


def compute_iwm_diffusivity(
    forcing: IWMForcing,
    depth_cell: jnp.ndarray,
    dz_w: jnp.ndarray,
    H: jnp.ndarray,
    N2: jnp.ndarray,
    *,
    cfg: IWMConfig,
    rho_0: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Wave-driven diffusivity + S/T ratio at interior interfaces.

    Transliterates ``zdf_iwm`` (zdfiwm.F90:164-253).  The Fortran w-level
    ``jk`` (2..jpkm1) maps to interior interface ``k = jk-2``; ``gdept(jk)``
    → ``depth_cell[..., k+1]``, ``gdept(jk-1)`` → ``depth_cell[..., k]``,
    ``e3w(jk)`` → ``dz_w[..., k]``, ``rn2(jk)`` → ``N2[..., k]``.

    Parameters
    ----------
    forcing : IWMForcing
        2-D maps, shape ``H.shape``.
    depth_cell : array ``(..., nlev)``
        Cell-centre depth [m], positive DOWN (NEMO gdept).
    dz_w : array ``(..., nlev-1)``
        Interface control-volume thickness [m] (NEMO e3w at interior
        w-points = centre-to-centre spacing).
    H : array ``(...)``
        Column depth [m] (NEMO ht).  Columns with ``H <= 0`` return the
        clipped floor (mask at the host).
    N2 : array ``(..., nlev-1)``
        Interface N² [1/s²].  Negative/zero values are handled exactly
        as NEMO: the structure functions use ``max(0, N²)``; the Reb
        denominator floors at ``1e-20`` so unstable interfaces saturate
        at ``k_max``.
    cfg : IWMConfig
    rho_0 : float
        Reference density [kg/m³] — thread the model's ConstantsConfig
        value so the W/m² → W/kg conversion matches the oracle.

    Returns
    -------
    (K_wave, av_ratio) : arrays ``(..., nlev-1)``
        ``K_wave`` in ``[cfg.k_min, cfg.k_max]`` [m²/s] (NOT wet-masked —
        the host applies its interface wet mask); ``av_ratio`` in (0, 1],
        identically 1 when ``cfg.tsdiff`` is False.
    """
    r1_rho0 = 1.0 / rho_0
    nu = cfg.nu_molecular

    H_col = H[..., jnp.newaxis]                       # (..., 1)
    H_pos = H_col > 0.0
    H_safe = jnp.where(H_pos, H_col, 1.0)

    # --- 'cri' component: exponential decay above the seafloor -------------
    # zfact1 = ecri/rho0 / (1 - exp(-ht * hcri_inv))     (zdfiwm.F90:170-171)
    hcri_inv = forcing.hcri_inv[..., jnp.newaxis]
    denom_cri = 1.0 - jnp.exp(-H_safe * hcri_inv)
    zfact1 = jnp.where(
        H_pos & (denom_cri > 0.0),
        forcing.ecri[..., jnp.newaxis] * r1_rho0
        / jnp.where(denom_cri > 0.0, denom_cri, 1.0),
        0.0,
    )
    # CDF difference between the bracketing cell centres (F90:199-200).
    # The exponent (gdept − ht)·hcri_inv is <= 0 for every wet cell
    # (gdept <= ht); clamp it there so reference depths BELOW the local
    # seafloor (partial-cell / land columns, later wet-masked by the
    # host) cannot overflow exp() into 0·inf = NaN — which would also
    # poison reverse-mode AD through the masked branch.
    z_up = depth_cell[..., :-1]    # gdept(jk-1)
    z_dn = depth_cell[..., 1:]     # gdept(jk)
    cri_part = zfact1 * (
        jnp.exp(jnp.minimum((z_dn - H_safe) * hcri_inv, 0.0))
        - jnp.exp(jnp.minimum((z_up - H_safe) * hcri_inv, 0.0))
    )

    # --- 'bot' component: algebraic decay above the seafloor ---------------
    # zfact2 = ebot/rho0 * (1 + hbot/ht)                 (zdfiwm.F90:176-177)
    hbot = forcing.hbot[..., jnp.newaxis]
    zfact2 = jnp.where(
        H_pos,
        forcing.ebot[..., jnp.newaxis] * (1.0 + hbot / H_safe) * r1_rho0,
        0.0,
    )
    # Height above the seafloor is >= 0 for every wet cell; clamp so
    # below-seafloor reference depths cannot cross the 1/(1+x) pole at
    # x = −1 (same NaN/AD hazard as the cri exponent above).
    bot_part = zfact2 * (
        1.0 / (1.0 + jnp.maximum(H_safe - z_dn, 0.0) / hbot)
        - 1.0 / (1.0 + jnp.maximum(H_safe - z_up, 0.0) / hbot)
    )

    # --- 'nsq' / 'sho' components: N² and N weighting ----------------------
    N2_pos = jnp.maximum(N2, 0.0)
    N_pos = jnp.sqrt(N2_pos)
    sum_n2 = jnp.sum(dz_w * N2_pos, axis=-1, keepdims=True)   # (F90:184-186)
    sum_n = jnp.sum(dz_w * N_pos, axis=-1, keepdims=True)
    zfact3 = jnp.where(
        sum_n2 > 0.0,
        forcing.ensq[..., jnp.newaxis] * r1_rho0
        / jnp.where(sum_n2 > 0.0, sum_n2, 1.0),
        0.0,
    )
    zfact4 = jnp.where(
        sum_n > 0.0,
        forcing.esho[..., jnp.newaxis] * r1_rho0
        / jnp.where(sum_n > 0.0, sum_n, 1.0),
        0.0,
    )

    # --- total local energy density available for mixing [W/kg] ------------
    # (cri + bot are CDF differences / e3w; nsq + sho are densities already)
    dz_w_safe = jnp.maximum(dz_w, _EPS_DENOM)
    zemx = (cri_part + bot_part) / dz_w_safe + zfact3 * N2_pos + zfact4 * N_pos

    # --- turbulence intensity parameter and diffusivity ---------------------
    # Reb = zemx / max(1e-20, nu * N²)  — NEMO uses the RAW rn2 here; our
    # insitu N2 is clipped >= 0 upstream, which lands in the same floor
    # branch for statically unstable interfaces (K saturates at k_max).
    reb = zemx / jnp.maximum(_EPS_DENOM, nu * N2)
    # Constant mixing efficiency Γ = 1/6 (F90:216).
    k_wave = reb * (1.0 / 6.0) * nu
    if cfg.mevar:
        # Variable-efficiency regimes (F90:219-227).
        # AD-safe sqrt: ``jnp.sqrt(jnp.maximum(reb, 0.0))`` has a NaN
        # reverse-mode gradient at reb=0 (land/zero-forcing cells) because the
        # derivative 1/(2·sqrt(reb)) blows up while ``maximum`` still routes the
        # cotangent through the sqrt. The double-``where`` keeps the primal
        # BIT-IDENTICAL for reb>0 and yields a finite 0 (0 gradient) at reb<=0.
        sqrt_reb = jnp.where(
            reb > 0.0, jnp.sqrt(jnp.where(reb > 0.0, reb, 1.0)), 0.0)
        k_wave = jnp.where(
            reb > _REB_ENERGETIC, _REB_ENERGETIC_COEF * nu * sqrt_reb, k_wave)
        k_wave = jnp.where(
            reb < _REB_BUOYANCY, _REB_BUOYANCY_COEF * nu * reb * sqrt_reb,
            k_wave)
    # Bound by molecular value and 100 cm²/s (F90:229-231).
    k_wave = jnp.minimum(jnp.maximum(cfg.k_min, k_wave), cfg.k_max)

    if cfg.tsdiff:
        # S/T diffusivity ratio as a function of Reb (F90:237-242).
        av_ratio = _TSDIFF_BASE + _TSDIFF_AMP * jnp.tanh(
            _TSDIFF_TANH_SLOPE
            * (jnp.log10(jnp.maximum(_EPS_DENOM, reb * _TSDIFF_REB_PREFAC))
               - _TSDIFF_LOG10_OFFSET)
        )
    else:
        av_ratio = jnp.ones_like(k_wave)

    return k_wave, av_ratio
