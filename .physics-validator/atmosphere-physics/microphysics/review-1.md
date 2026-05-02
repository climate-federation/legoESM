Reading prompt from stdin...
OpenAI Codex v0.125.0 (research preview)
--------
workdir: /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
model: gpt-5.5
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, /Users/pierregentine/.codex/memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019de361-3ff6-7a50-af6e-20c82f3c6832
--------
user
# Adversarial review: legoESM microphysics package

You are an independent adversarial physics reviewer for an Earth-system model.

Your task: read the source files below and find every concrete bug, sign error, unit inconsistency, broken-gradient pattern, conservation violation, and AD-safety issue. Cite line numbers. Severity: critical | major | minor.

The codebase under review is a JAX implementation of atmospheric microphysics that must be:
- Physically correct (units, signs, mass+latent-heat conservation, monotonicity).
- Differentiable end-to-end (no hard `if` on traced values, AD-safe limiters).

Static analysis already produced these candidate concerns for you to verify or rebut:

1. **Morrison (`morrison.py:149-154`) — Bergeron + riming missing latent heat contributions.** The current `dT_dt` only includes condensation, evaporation, deposition, and melting. **Bergeron (cloud water → ice) and riming (cloud water → ice/snow) should each release `L_f` (latent heat of fusion).** For default `bergeron_rate = 1e-3 /s` and `q_c = 1e-4 kg/kg`, the missing heating is `L_f * 1e-7 / c_pd ~ 3.3e-5 K/s ~ 2.8 K/day` in mixed-phase clouds. Correct?

2. **Thompson (`thompson.py:174-179`) — same Bergeron + riming + graupel formation latent heat omitted.**

3. **Thompson (`thompson.py:143-144, 151`) — `melt_X = melt_rate * q_X * melt_frac` not clamped to `q_X / dt`.** Morrison (`morrison.py:121-128`) has this clamp. For melt_rate=5e-3/s and dt=1200s, `melt_rate * dt = 6` ⇒ explicit step yields negative q_i_new for small q_i.

4. **Morrison/Thompson — N_i not decremented during melting** (`morrison.py:165`, `thompson.py:191`). When q_i melts, N_i stays the same; mean particle mass `q_i / N_i` becomes unphysically small.

5. **Sundqvist (`sundqvist.py:68`) — `P_auto = config.auto_rate * jnp.maximum(q_c + condensation * dt, 0.0)` uses post-condensation cloud water.** This is forward-Euler coupled to dt linearly. Standard Sundqvist uses `q_c` only. May or may not be a bug.

6. **`_warm_rain.autoconversion_sb:93` — `x_c = q_c_pos * rho / max(N_c_eff, 1.0)`.** When q_c=0, x_c=0; `dq_c_au = k_au * q_c**2 * onset(x_c-x_star)`. For q_c approaching zero, onset(0-x_star) ≈ 0, dq_c_au ≈ 0. AD: gradient w.r.t. q_c is well-defined.

7. **`_warm_rain.rain_evaporation` and `kessler` use `q_r ** 0.525`.** AD-unstable at q_r = 0 (derivative diverges). Practically, q_r > 0 once precipitation has occurred. Probably fine in practice.

8. **Thompson `_gamma_ratio(mu) = (mu+3)(mu+2)(mu+1)`** — Γ(μ+4)/Γ(μ+1). Comment says "for integer-like μ" but the recurrence Γ(μ+1) = μΓ(μ) makes this exact for ANY real μ ≥ 0. Non-bug.

9. **All schemes lack a saturation adjustment iteration**: a single sigmoid-based saturation step does not enforce thermodynamic equilibrium. The `cond_frac = sigmoid(s * excess) * excess / dt` produces a smooth condensation but doesn't iterate to (T_new, q_new) on the saturation curve. Acceptable for AD-friendly column physics but not as accurate as Newton-iterated saturation.

10. **Conservation guard tests**: I'm told some conservation tests are `xfail` in the suite. Verify by checking conservation column-by-column in the source.

Files to read:
- `src/legoesm/atmosphere/physics/microphysics/_warm_rain.py` (176 LOC).
- `src/legoesm/atmosphere/physics/microphysics/kessler.py` (136 LOC).
- `src/legoesm/atmosphere/physics/microphysics/sundqvist.py` (170 LOC).
- `src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py` (130 LOC).
- `src/legoesm/atmosphere/physics/microphysics/morrison.py` (189 LOC).
- `src/legoesm/atmosphere/physics/microphysics/thompson.py` (213 LOC).
- `src/legoesm/atmosphere/physics/microphysics/config.py`.
- `src/legoesm/atmosphere/physics/microphysics/output.py`.

Constants from `legoesm.constants`: L_v = 2.501e6 J/kg, L_s = 2.838e6 J/kg, L_f = L_s - L_v = 0.337e6 J/kg, c_pd = 1004.64 J/kg/K, T_freeze = 273.15 K, R_v = 461.5 J/kg/K.

For each finding, output:
- File path : line range.
- One-sentence summary.
- Brief technical explanation (≤3 sentences).
- Severity.
- If addressing one of the 10 hypotheses above, explicitly say so.

If no findings beyond those flagged, end with `NO ADDITIONAL SUBSTANTIVE FINDINGS`.

codex
I’ll read the microphysics sources with line numbers first, then cross-check the candidate concerns against the actual update equations and any conservation tests. After that I’ll report only concrete findings, with explicit “verified/rebutted” notes for the numbered hypotheses.
exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/microphysics/sundqvist.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/microphysics/kessler.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/microphysics/_warm_rain.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Seifert-Beheng two-moment warm-rain microphysics.
     2	
     3	A two-moment scheme tracking mass and number concentration of cloud
     4	droplets and rain drops. Processes: saturation adjustment, autoconversion
     5	(mass-dependent), accretion, self-collection, breakup, rain evaporation,
     6	and sedimentation.
     7	
     8	All operations use smooth (differentiable) approximations.
     9	
    10	References
    11	----------
    12	- Seifert, A., & Beheng, K. D. (2001). A two-moment cloud microphysics
    13	  parameterization for mixed-phase clouds. Meteorol. Atmos. Phys., 77, 127-151.
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	import jax
    19	import jax.numpy as jnp
    20	
    21	from legoesm import constants
    22	from legoesm.atmosphere.physics.microphysics.config import SeifertBehengConfig
    23	from legoesm.atmosphere.physics.microphysics.output import (
    24	    HydrometeorState,
    25	    MicrophysicsOutput,
    26	    sedimentation_tendency,
    27	)
    28	from legoesm.atmosphere.physics.microphysics._warm_rain import (
    29	    saturation_adjustment,
    30	    effective_Nc,
    31	    autoconversion_sb,
    32	    accretion,
    33	    self_collection_breakup,
    34	    rain_evaporation,
    35	)
    36	
    37	
    38	def seifert_beheng_microphysics(
    39	    T: jax.Array,
    40	    q_v: jax.Array,
    41	    hydrometeors: HydrometeorState,
    42	    p_full: jax.Array,
    43	    p_half: jax.Array,
    44	    rho: jax.Array,
    45	    dz: jax.Array,
    46	    dt: float,
    47	    config: SeifertBehengConfig = SeifertBehengConfig(),
    48	) -> MicrophysicsOutput:
    49	    """Compute Seifert-Beheng two-moment warm-rain tendencies.
    50	
    51	    Parameters
    52	    ----------
    53	    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
    54	        Same interface as all microphysics backends.
    55	
    56	    Returns
    57	    -------
    58	    MicrophysicsOutput
    59	    """
    60	    ncol, nlev = T.shape
    61	    q_c = hydrometeors.q_c
    62	    q_r = hydrometeors.q_r
    63	    N_c = hydrometeors.N_c
    64	    N_r = hydrometeors.N_r
    65	    sharpness = config.saturation_sharpness
    66	
    67	    N_c_eff = effective_Nc(N_c, config.Nc_0)
    68	
    69	    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
    70	
    71	    # 1. Autoconversion (mass-dependent)
    72	    dq_c_au, dN_r_au, x_c = autoconversion_sb(
    73	        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness,
    74	    )
    75	
    76	    # 2. Accretion
    77	    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac)
    78	
    79	    # 3-4. Self-collection and breakup
    80	    dN_r_sc, dN_r_br = self_collection_breakup(
    81	        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    82	    )
    83	
    84	    # 5. Rain evaporation
    85	    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)
    86	
    87	    # 6. Sedimentation
    88	    rho_sfc = rho[:, -1:]
    89	    V_t_r = config.a_v_r * (jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_r
    90	    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
    91	    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
    92	
    93	    # 7. Latent heating
    94	    dT_dt = constants.L_v * (condensation - evaporation) / constants.c_pd
    95	
    96	    # Combine tendencies
    97	    dq_v_dt = -condensation + evaporation
    98	    dq_c_dt = condensation - dq_c_au - dq_c_ac
    99	    dq_r_dt = dq_c_au + dq_c_ac - evaporation + sed_r
   100	    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
   101	    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
   102	
   103	    # Precipitation
   104	    q_r_bot = jnp.clip(q_r[:, -1], 0.0)
   105	    precipitation = q_r_bot * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
   106	
   107	    # Pin dtype to the input precision so we never silently promote
   108	    # the unused-species placeholders to f64 under x64 mode.
   109	    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
   110	    return MicrophysicsOutput(
   111	        dT_dt=dT_dt,
   112	        dq_v_dt=dq_v_dt,
   113	        dq_c_dt=dq_c_dt,
   114	        dq_r_dt=dq_r_dt,
   115	        dq_i_dt=z,
   116	        dq_s_dt=z,
   117	        dq_g_dt=z,
   118	        dN_c_dt=dN_c_dt,
   119	        dN_r_dt=dN_r_dt,
   120	        dN_i_dt=z,
   121	        precipitation=precipitation,
   122	    )

 succeeded in 0ms:
     1	"""Shared warm-rain microphysics helpers.
     2	
     3	Functions here are used by multiple microphysics backends (Seifert-Beheng,
     4	Morrison, Thompson, Kessler) to avoid duplicating identical physics code.
     5	"""
     6	
     7	from __future__ import annotations
     8	
     9	import jax
    10	import jax.numpy as jnp
    11	
    12	from legoesm import constants
    13	from legoesm.thermo import saturation_mixing_ratio
    14	
    15	
    16	def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0):
    17	    """Compute smooth saturation adjustment (condensation tendency).
    18	
    19	    Parameters
    20	    ----------
    21	    T : array (ncol, nlev)
    22	        Temperature [K].
    23	    q_v : array (ncol, nlev)
    24	        Water vapor mixing ratio [kg/kg].
    25	    p_full : array (ncol, nlev)
    26	        Pressure [Pa].
    27	    dt : float
    28	        Time step [s].
    29	    sharpness : float
    30	        Sigmoid sharpness for smooth condensation switch.
    31	
    32	    Returns
    33	    -------
    34	    condensation : array (ncol, nlev)
    35	        Condensation tendency [kg/kg/s].
    36	    q_sat : array (ncol, nlev)
    37	        Saturation mixing ratio [kg/kg].
    38	    """
    39	    q_sat = saturation_mixing_ratio(T, p_full)
    40	    excess = q_v - q_sat
    41	    cond_frac = jax.nn.sigmoid(sharpness * excess)
    42	    condensation = cond_frac * excess / dt
    43	    return condensation, q_sat
    44	
    45	
    46	def effective_Nc(N_c, Nc_0):
    47	    """Use config default cloud droplet number where N_c is zero.
    48	
    49	    Parameters
    50	    ----------
    51	    N_c : array
    52	        Cloud droplet number concentration [1/kg].
    53	    Nc_0 : float
    54	        Default cloud droplet number.
    55	
    56	    Returns
    57	    -------
    58	    array : Effective N_c.
    59	    """
    60	    return jnp.where(N_c > 1.0, N_c, Nc_0 * jnp.ones_like(N_c))
    61	
    62	
    63	def autoconversion_sb(q_c, N_c_eff, rho, k_au, x_star, sharpness=50.0, gamma_norm=1.0):
    64	    """Seifert-Beheng mass-dependent autoconversion.
    65	
    66	    Parameters
    67	    ----------
    68	    q_c : array
    69	        Cloud water mixing ratio [kg/kg].
    70	    N_c_eff : array
    71	        Effective cloud droplet number [1/kg].
    72	    rho : array
    73	        Air density [kg/m3].
    74	    k_au : float
    75	        Autoconversion rate constant.
    76	    x_star : float
    77	        Mean droplet mass threshold [kg].
    78	    sharpness : float
    79	        Sigmoid sharpness.
    80	    gamma_norm : float
    81	        Gamma distribution correction (1.0 for SB/Morrison, != 1.0 for Thompson).
    82	
    83	    Returns
    84	    -------
    85	    dq_c_au : array
    86	        Cloud water autoconversion rate [kg/kg/s].
    87	    dN_r_au : array
    88	        Rain number formation rate [1/kg/s].
    89	    x_c : array
    90	        Mean cloud droplet mass [kg].
    91	    """
    92	    q_c_pos = jnp.clip(q_c, 0.0)
    93	    x_c = q_c_pos * rho / jnp.clip(N_c_eff, 1.0)
    94	    onset = jax.nn.sigmoid(sharpness * (x_c - x_star))
    95	    dq_c_au = k_au * q_c_pos ** 2 * onset * gamma_norm * rho
    96	    dN_r_au = dq_c_au * rho / (x_star * 20.0)
    97	    return dq_c_au, dN_r_au, x_c
    98	
    99	
   100	def accretion(q_c, q_r, rho, k_ac, gamma_norm=1.0):
   101	    """Rain collecting cloud water (accretion).
   102	
   103	    Parameters
   104	    ----------
   105	    q_c, q_r : array
   106	        Cloud water and rain mixing ratios [kg/kg].
   107	    rho : array
   108	        Air density [kg/m3].
   109	    k_ac : float
   110	        Accretion rate constant.
   111	    gamma_norm : float
   112	        Gamma distribution correction.
   113	
   114	    Returns
   115	    -------
   116	    array : Accretion rate [kg/kg/s].
   117	    """
   118	    return k_ac * jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0) * rho * gamma_norm
   119	
   120	
   121	def self_collection_breakup(N_r, q_r, rho, k_sc, breakup_sharpness, D_eq):
   122	    """Self-collection and breakup of rain drops.
   123	
   124	    Parameters
   125	    ----------
   126	    N_r : array
   127	        Rain drop number concentration [1/kg].
   128	    q_r : array
   129	        Rain mixing ratio [kg/kg].
   130	    rho : array
   131	        Air density [kg/m3].
   132	    k_sc : float
   133	        Self-collection rate constant.
   134	    breakup_sharpness : float
   135	        Sigmoid sharpness for breakup onset.
   136	    D_eq : float
   137	        Equilibrium drop diameter [m].
   138	
   139	    Returns
   140	    -------
   141	    dN_r_sc : array
   142	        Self-collection tendency [1/kg/s].
   143	    dN_r_br : array
   144	        Breakup tendency [1/kg/s].
   145	    """
   146	    dN_r_sc = -k_sc * jnp.clip(N_r, 0.0) * jnp.clip(q_r, 0.0) * rho
   147	    D_r = jnp.clip(
   148	        (jnp.clip(q_r, 0.0) * rho / jnp.clip(N_r, 1.0) / (jnp.pi / 6.0 * constants.rho_water)),
   149	        0.0,
   150	    ) ** (1.0 / 3.0)
   151	    breakup_frac = jax.nn.sigmoid(breakup_sharpness * (D_r - D_eq))
   152	    dN_r_br = -dN_r_sc * breakup_frac
   153	    return dN_r_sc, dN_r_br
   154	
   155	
   156	def rain_evaporation(q_v, q_r, q_sat, evap_coeff):
   157	    """Compute rain evaporation in subsaturated air.
   158	
   159	    Parameters
   160	    ----------
   161	    q_v : array
   162	        Water vapor mixing ratio [kg/kg].
   163	    q_r : array
   164	        Rain mixing ratio [kg/kg].
   165	    q_sat : array
   166	        Saturation mixing ratio [kg/kg].
   167	    evap_coeff : float
   168	        Evaporation rate coefficient.
   169	
   170	    Returns
   171	    -------
   172	    array : Evaporation rate [kg/kg/s].
   173	    """
   174	    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
   175	    return evap_coeff * subsaturation * jnp.clip(q_r, 0.0) ** 0.525

 succeeded in 0ms:
     1	"""Kessler warm-rain microphysics column backend.
     2	
     3	A simple one-moment warm-rain scheme tracking cloud water and rain.
     4	Processes: saturation adjustment, autoconversion, accretion, evaporation,
     5	rain sedimentation, and latent heating.
     6	
     7	All operations use smooth (differentiable) approximations for
     8	compatibility with jax.grad.
     9	
    10	References
    11	----------
    12	- Kessler (1969): On the Distribution and Continuity of Water
    13	  Substance in Atmospheric Circulations.
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	import jax
    19	import jax.numpy as jnp
    20	
    21	from legoesm import constants
    22	from legoesm.thermo import saturation_mixing_ratio
    23	from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
    24	from legoesm.atmosphere.physics.microphysics.output import (
    25	    HydrometeorState,
    26	    MicrophysicsOutput,
    27	    sedimentation_tendency,
    28	)
    29	
    30	
    31	def kessler_microphysics(
    32	    T: jax.Array,
    33	    q_v: jax.Array,
    34	    hydrometeors: HydrometeorState,
    35	    p_full: jax.Array,
    36	    p_half: jax.Array,
    37	    rho: jax.Array,
    38	    dz: jax.Array,
    39	    dt: float,
    40	    config: KesslerConfig = KesslerConfig(),
    41	) -> MicrophysicsOutput:
    42	    """Compute Kessler microphysics tendencies.
    43	
    44	    Parameters
    45	    ----------
    46	    T : jax.Array
    47	        Temperature [K], shape (ncol, nlev).
    48	    q_v : jax.Array
    49	        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    50	    hydrometeors : HydrometeorState
    51	        Hydrometeor state (only q_c, q_r used).
    52	    p_full : jax.Array
    53	        Pressure at full levels [Pa], shape (ncol, nlev).
    54	    p_half : jax.Array
    55	        Pressure at half levels [Pa], shape (ncol, nlev+1).
    56	    rho : jax.Array
    57	        Air density [kg/m^3], shape (ncol, nlev).
    58	    dz : jax.Array
    59	        Layer thickness [m], shape (ncol, nlev).
    60	    dt : float
    61	        Time step [s].
    62	    config : KesslerConfig
    63	
    64	    Returns
    65	    -------
    66	    MicrophysicsOutput
    67	    """
    68	    ncol, nlev = T.shape
    69	    q_c = hydrometeors.q_c
    70	    q_r = hydrometeors.q_r
    71	    sharpness = config.saturation_sharpness
    72	
    73	    # Saturation mixing ratio
    74	    q_sat = saturation_mixing_ratio(T, p_full)
    75	
    76	    # 1. Saturation adjustment — convert from increment [kg/kg] to tendency [kg/kg/s]
    77	    excess = q_v - q_sat
    78	    cond_frac = jax.nn.sigmoid(sharpness * excess)
    79	    condensation = cond_frac * excess / dt  # [kg/kg/s]
    80	
    81	    dq_v_sat = -condensation
    82	    dq_c_sat = condensation
    83	
    84	    # 2. Autoconversion: cloud -> rain (threshold excess)
    85	    # dq_c_sat is a tendency [kg/kg/s]; multiply by dt to get increment [kg/kg]
    86	    q_c_updated = q_c + dq_c_sat * dt
    87	    autoconv = config.autoconversion_rate * jnp.maximum(
    88	        q_c_updated - config.autoconversion_threshold, 0.0
    89	    )
    90	
    91	    # 3. Accretion: cloud collected by rain
    92	    accretion = config.accretion_coeff * q_c * jnp.clip(q_r, 0.0) ** 0.875
    93	
    94	    # 4. Evaporation of rain
    95	    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
    96	    evaporation = config.evaporation_coeff * subsaturation * jnp.clip(q_r, 0.0) ** 0.525
    97	
    98	    # 5. Rain sedimentation
    99	    rho_sfc = rho[:, -1:]
   100	    V_t = config.rain_fall_speed * jnp.sqrt(
   101	        rho_sfc / jnp.clip(rho, 0.1)
   102	    )
   103	    sed_tend = sedimentation_tendency(q_r, rho, V_t, dz)
   104	
   105	    # 6. Latent heating
   106	    dT_dt = constants.L_v * (condensation - evaporation) / constants.c_pd
   107	
   108	    # Combine tracer tendencies
   109	    dq_v_dt = dq_v_sat + evaporation
   110	    dq_c_dt = dq_c_sat - autoconv - accretion
   111	    dq_r_dt = autoconv + accretion - evaporation + sed_tend
   112	
   113	    # Precipitation: surface flux
   114	    q_r_bot = jnp.clip(q_r[:, -1], 0.0)
   115	    V_t_bot = V_t[:, -1]
   116	    precipitation = q_r_bot * rho[:, -1] * V_t_bot
   117	
   118	    # Pin dtype to the input precision so we never silently promote
   119	    # the unused-tendency placeholders to f64 under x64 mode.
   120	    _dtype = T.dtype
   121	    z = jnp.zeros((ncol, nlev), dtype=_dtype)
   122	    z1 = jnp.zeros((ncol,), dtype=_dtype)
   123	    return MicrophysicsOutput(
   124	        dT_dt=dT_dt,
   125	        dq_v_dt=dq_v_dt,
   126	        dq_c_dt=dq_c_dt,
   127	        dq_r_dt=dq_r_dt,
   128	        dq_i_dt=z,
   129	        dq_s_dt=z,
   130	        dq_g_dt=z,
   131	        dN_c_dt=z,
   132	        dN_r_dt=z,
   133	        dN_i_dt=z,
   134	        precipitation=precipitation,
   135	    )

 succeeded in 0ms:
     1	"""Sundqvist large-scale diagnostic condensation scheme.
     2	
     3	A diagnostic scheme that activates condensation when relative humidity
     4	exceeds a critical threshold. Produces large-scale (non-convective)
     5	precipitation through autoconversion and sub-cloud evaporation.
     6	
     7	All operations use smooth (differentiable) approximations.
     8	
     9	References
    10	----------
    11	- Sundqvist et al. (1989): Condensation and cloud parameterization
    12	  studies with a mesoscale numerical weather prediction model.
    13	"""
    14	
    15	from __future__ import annotations
    16	
    17	from typing import NamedTuple
    18	
    19	import jax
    20	import jax.numpy as jnp
    21	
    22	from legoesm import constants
    23	from legoesm.thermo import saturation_mixing_ratio
    24	from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
    25	from legoesm.atmosphere.physics.microphysics.output import (
    26	    HydrometeorState,
    27	    MicrophysicsOutput,
    28	)
    29	
    30	
    31	class SundqvistProcessRates(NamedTuple):
    32	    """Intermediate Sundqvist process rates used to assemble tendencies."""
    33	
    34	    condensation: jax.Array
    35	    autoconversion: jax.Array
    36	    evaporation: jax.Array
    37	    precipitation: jax.Array
    38	
    39	
    40	def diagnose_sundqvist_process_rates(
    41	    T: jax.Array,
    42	    q_v: jax.Array,
    43	    hydrometeors: HydrometeorState,
    44	    p_full: jax.Array,
    45	    p_half: jax.Array,
    46	    rho: jax.Array,
    47	    dz: jax.Array,
    48	    dt: float,
    49	    config: SundqvistConfig = SundqvistConfig(),
    50	) -> SundqvistProcessRates:
    51	    """Diagnose the Sundqvist condensation, rain conversion, and evaporation terms."""
    52	    del p_half  # Included for signature parity with ``sundqvist_microphysics``.
    53	    q_c = hydrometeors.q_c
    54	    sharpness = config.sigmoid_sharpness
    55	
    56	    # Saturation
    57	    q_sat = saturation_mixing_ratio(T, p_full)
    58	    RH = q_v / jnp.clip(q_sat, 1e-10)
    59	
    60	    # 1. Smooth condensation activation — convert increment [kg/kg] to tendency [kg/kg/s]
    61	    f = jax.nn.sigmoid(sharpness * (RH - config.RH_crit))
    62	    condensation = (
    63	        f * jnp.maximum(q_v - config.RH_crit * q_sat, 0.0) / dt
    64	    )  # [kg/kg/s]
    65	
    66	    # 2. Autoconversion
    67	    # condensation is a tendency [kg/kg/s]; multiply by dt to get increment [kg/kg]
    68	    P_auto = config.auto_rate * jnp.maximum(q_c + condensation * dt, 0.0)
    69	
    70	    # 3. Sub-cloud evaporation
    71	    evap_mask = jax.nn.sigmoid(sharpness * (config.RH_crit - RH))
    72	    P_flux_layer = P_auto * rho * dz
    73	
    74	    def scan_fn(carry, x):
    75	        P_above = carry
    76	        P_local, evap_m, rho_k, dz_k = x
    77	        P_total = P_above + P_local
    78	        evap = config.evap_coeff * evap_m * P_total / jnp.clip(rho_k * dz_k, 1.0)
    79	        evap = jnp.minimum(evap, P_total / jnp.clip(rho_k * dz_k, 1.0))
    80	        P_out = jnp.clip(P_total - evap * rho_k * dz_k, 0.0)
    81	        return P_out, evap
    82	
    83	    # Pick a working dtype that ``scan`` can carry without promotion.
    84	    # Under ``JAX_ENABLE_X64=1`` ``jnp.zeros``/``jnp.ones`` default to
    85	    # f64, so a state assembled from a mix of (f32) ``T`` and (f64)
    86	    # tracers ends up with f64 ``q_v``/``q_c``.  ``P_flux_layer``
    87	    # then inherits the f64 promotion from ``q_c + condensation * dt``,
    88	    # while a carry pinned to ``T.dtype`` (f32) would mismatch the
    89	    # f64 scan output.  Promoting to the wider of carry/input dtype
    90	    # keeps ``scan`` happy without silently downcasting precipitation
    91	    # mass.
    92	    _scan_dtype = jnp.promote_types(T.dtype, P_flux_layer.dtype)
    93	    inputs = (
    94	        jnp.moveaxis(P_flux_layer.astype(_scan_dtype), 1, 0),
    95	        jnp.moveaxis(evap_mask.astype(_scan_dtype), 1, 0),
    96	        jnp.moveaxis(rho.astype(_scan_dtype), 1, 0),
    97	        jnp.moveaxis(dz.astype(_scan_dtype), 1, 0),
    98	    )
    99	    P_init = jnp.zeros(T.shape[0], dtype=_scan_dtype)
   100	    P_final, evap_col = jax.lax.scan(scan_fn, P_init, inputs)
   101	    evaporation = jnp.moveaxis(evap_col, 0, 1)
   102	    return SundqvistProcessRates(
   103	        condensation=condensation,
   104	        autoconversion=P_auto,
   105	        evaporation=evaporation,
   106	        precipitation=P_final,
   107	    )
   108	
   109	
   110	def sundqvist_microphysics(
   111	    T: jax.Array,
   112	    q_v: jax.Array,
   113	    hydrometeors: HydrometeorState,
   114	    p_full: jax.Array,
   115	    p_half: jax.Array,
   116	    rho: jax.Array,
   117	    dz: jax.Array,
   118	    dt: float,
   119	    config: SundqvistConfig = SundqvistConfig(),
   120	) -> MicrophysicsOutput:
   121	    """Compute Sundqvist diagnostic condensation tendencies.
   122	
   123	    Parameters
   124	    ----------
   125	    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
   126	        Same interface as all microphysics backends.
   127	
   128	    Returns
   129	    -------
   130	    MicrophysicsOutput
   131	    """
   132	    ncol, nlev = T.shape
   133	    rates = diagnose_sundqvist_process_rates(
   134	        T=T,
   135	        q_v=q_v,
   136	        hydrometeors=hydrometeors,
   137	        p_full=p_full,
   138	        p_half=p_half,
   139	        rho=rho,
   140	        dz=dz,
   141	        dt=dt,
   142	        config=config,
   143	    )
   144	
   145	    # 4. Latent heating
   146	    net_cond = rates.condensation - rates.evaporation
   147	    dT_dt = constants.L_v * net_cond / constants.c_pd
   148	
   149	    # Tendencies
   150	    dq_v_dt = -rates.condensation + rates.evaporation
   151	    dq_c_dt = rates.condensation - rates.autoconversion
   152	    dq_r_dt = rates.autoconversion - rates.evaporation
   153	
   154	    # Pin dtype to the input precision so we never silently promote
   155	    # the unused-tendency placeholders to f64 under x64 mode.
   156	    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
   157	    return MicrophysicsOutput(
   158	        dT_dt=dT_dt,
   159	        dq_v_dt=dq_v_dt,
   160	        dq_c_dt=dq_c_dt,
   161	        dq_r_dt=dq_r_dt,
   162	        dq_i_dt=z,
   163	        dq_s_dt=z,
   164	        dq_g_dt=z,
   165	        dN_c_dt=z,
   166	        dN_r_dt=z,
   167	        dN_i_dt=z,
   168	        precipitation=rates.precipitation,
   169	    )

codex
The warm-rain files show a few patterns to scrutinize: several explicit process rates are unconstrained by available mass over `dt`, and sedimentation returns a column divergence while the precipitation diagnostic uses a separate bottom-flux expression. I’m reading the mixed-phase schemes and shared output/config next because that’s where the latent-heat and number-tendency claims live.
exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/microphysics/morrison.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/microphysics/thompson.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Morrison double-moment ice+liquid microphysics.
     2	
     3	Extends Seifert-Beheng warm-rain with ice-phase processes: nucleation
     4	(Cooper 1986), depositional growth, Bergeron process, riming, snow
     5	aggregation, and melting. Tracks cloud water, rain, ice, and snow.
     6	
     7	All operations use smooth (differentiable) approximations.
     8	
     9	References
    10	----------
    11	- Morrison, H., Curry, J. A., & Khvorostyanov, V. I. (2005). A new
    12	  double-moment microphysics parameterization. Part I: Description.
    13	  J. Atmos. Sci., 62, 1665-1677.
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	import jax
    19	import jax.numpy as jnp
    20	
    21	from legoesm import constants
    22	from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
    23	from legoesm.atmosphere.physics.microphysics._warm_rain import (
    24	    saturation_adjustment,
    25	    effective_Nc,
    26	    autoconversion_sb,
    27	    accretion,
    28	    self_collection_breakup,
    29	    rain_evaporation,
    30	)
    31	from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    32	from legoesm.atmosphere.physics.microphysics.output import (
    33	    HydrometeorState,
    34	    MicrophysicsOutput,
    35	    sedimentation_tendency,
    36	)
    37	
    38	
    39	def morrison_microphysics(
    40	    T: jax.Array,
    41	    q_v: jax.Array,
    42	    hydrometeors: HydrometeorState,
    43	    p_full: jax.Array,
    44	    p_half: jax.Array,
    45	    rho: jax.Array,
    46	    dz: jax.Array,
    47	    dt: float,
    48	    config: MorrisonConfig = MorrisonConfig(),
    49	) -> MicrophysicsOutput:
    50	    """Compute Morrison double-moment microphysics tendencies.
    51	
    52	    Parameters
    53	    ----------
    54	    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
    55	        Same interface as all microphysics backends.
    56	
    57	    Returns
    58	    -------
    59	    MicrophysicsOutput
    60	    """
    61	    ncol, nlev = T.shape
    62	    q_c = hydrometeors.q_c
    63	    q_r = hydrometeors.q_r
    64	    q_i = hydrometeors.q_i
    65	    q_s = hydrometeors.q_s
    66	    N_c = hydrometeors.N_c
    67	    N_r = hydrometeors.N_r
    68	    N_i = hydrometeors.N_i
    69	    sharpness = config.saturation_sharpness
    70	
    71	    N_c_eff = effective_Nc(N_c, config.Nc_0)
    72	
    73	    # === WARM RAIN (shared Seifert-Beheng helpers) ===
    74	    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
    75	    dq_c_au, dN_r_au, x_c = autoconversion_sb(
    76	        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness,
    77	    )
    78	    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac)
    79	    dN_r_sc, dN_r_br = self_collection_breakup(
    80	        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    81	    )
    82	    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)
    83	
    84	    # === ICE PHASE ===
    85	    T_freeze = constants.T_freeze
    86	    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))
    87	
    88	    # 1. Ice nucleation (Cooper 1986, smoothed)
    89	    N_i_target = config.N_i0 * jnp.exp(
    90	        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
    91	    ) / jnp.clip(rho, 0.1)
    92	    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
    93	
    94	    # 2. Depositional growth
    95	    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
    96	    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
    97	    dq_i_dep = (
    98	        config.dep_coeff
    99	        * jnp.maximum(S_i, 0.0)
   100	        * jnp.clip(q_i, 0.0)
   101	        * jnp.clip(N_i, 0.0) ** (1.0 / 3.0)
   102	        * f_ice
   103	    )
   104	
   105	    # 3. Bergeron process: cloud water -> ice in mixed-phase zone
   106	    berg_window = (
   107	        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
   108	        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
   109	    )
   110	    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window
   111	
   112	    # 4. Riming: ice/snow collect cloud water
   113	    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   114	    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   115	
   116	    # 5. Snow aggregation: ice -> snow
   117	    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice
   118	
   119	    # 6. Melting near T_freeze: ice/snow -> rain (clipped to available mass)
   120	    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
   121	    melt_ice = jnp.minimum(
   122	        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
   123	        jnp.clip(q_i, 0.0) / jnp.maximum(dt, 1e-10),
   124	    )
   125	    melt_snow = jnp.minimum(
   126	        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
   127	        jnp.clip(q_s, 0.0) / jnp.maximum(dt, 1e-10),
   128	    )
   129	
   130	    # === SEDIMENTATION ===
   131	    rho_sfc = rho[:, -1:]
   132	    V_t_r = config.a_v_r * (jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_r
   133	    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
   134	    V_t_i = config.a_v_i * (jnp.clip(q_i, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_i
   135	    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
   136	    V_t_s = config.a_v_s * (jnp.clip(q_s, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_s
   137	    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)
   138	
   139	    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
   140	    sed_i = sedimentation_tendency(q_i, rho, V_t_i, dz)
   141	    sed_s = sedimentation_tendency(q_s, rho, V_t_s, dz)
   142	
   143	    # === LATENT HEATING ===
   144	    L_v = constants.L_v
   145	    L_s = constants.L_s
   146	    L_f = constants.L_f
   147	    c_pd = constants.c_pd
   148	
   149	    dT_dt = (
   150	        L_v * condensation / c_pd
   151	        - L_v * evaporation / c_pd
   152	        + L_s * dq_i_dep / c_pd
   153	        - L_f * (melt_ice + melt_snow) / c_pd
   154	    )
   155	
   156	    # === COMBINE TENDENCIES ===
   157	    dq_v_dt = -condensation + evaporation - dq_i_dep
   158	    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
   159	    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + sed_r
   160	    dq_i_dt = dq_i_dep + bergeron + riming_i - aggregation - melt_ice + sed_i
   161	    dq_s_dt = aggregation + riming_s - melt_snow + sed_s
   162	
   163	    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
   164	    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
   165	    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
   166	
   167	    # Precipitation (rain + ice + snow at surface)
   168	    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
   169	    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)
   170	    precip_s = jnp.clip(q_s[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_s[:, -1], 0.0)
   171	    precipitation = precip_r + precip_i + precip_s
   172	
   173	    # Pin dtype to the input precision so we never silently promote
   174	    # the unused-species placeholders to f64 under x64 mode.
   175	    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
   176	    return MicrophysicsOutput(
   177	        dT_dt=dT_dt,
   178	        dq_v_dt=dq_v_dt,
   179	        dq_c_dt=dq_c_dt,
   180	        dq_r_dt=dq_r_dt,
   181	        dq_i_dt=dq_i_dt,
   182	        dq_s_dt=dq_s_dt,
   183	        dq_g_dt=z,
   184	        dN_c_dt=dN_c_dt,
   185	        dN_r_dt=dN_r_dt,
   186	        dN_i_dt=dN_i_dt,
   187	        precipitation=precipitation,
   188	    )

 succeeded in 0ms:
     1	"""Thompson hybrid-moment microphysics.
     2	
     3	Extends Morrison with graupel formation from intense riming and
     4	gamma distribution shape corrections for autoconversion/accretion.
     5	
     6	All operations use smooth (differentiable) approximations.
     7	
     8	References
     9	----------
    10	- Thompson, G., Field, P. R., Rasmussen, R. M., & Hall, W. D. (2008).
    11	  Explicit forecasts of winter precipitation using an improved bulk
    12	  microphysics scheme. Part II: Implementation of a new snow
    13	  parameterization. Mon. Wea. Rev., 136, 5095-5115.
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	import jax
    19	import jax.numpy as jnp
    20	
    21	from legoesm import constants
    22	from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
    23	from legoesm.atmosphere.physics.microphysics._warm_rain import (
    24	    saturation_adjustment,
    25	    effective_Nc,
    26	    autoconversion_sb,
    27	    accretion,
    28	    self_collection_breakup,
    29	    rain_evaporation,
    30	)
    31	from legoesm.atmosphere.physics.microphysics.config import ThompsonConfig
    32	from legoesm.atmosphere.physics.microphysics.output import (
    33	    HydrometeorState,
    34	    MicrophysicsOutput,
    35	    sedimentation_tendency,
    36	)
    37	
    38	
    39	def _gamma_ratio(mu):
    40	    """Gamma(mu+4)/Gamma(mu+1) = (mu+3)(mu+2)(mu+1) for integer-like mu."""
    41	    return (mu + 3.0) * (mu + 2.0) * (mu + 1.0)
    42	
    43	
    44	def thompson_microphysics(
    45	    T: jax.Array,
    46	    q_v: jax.Array,
    47	    hydrometeors: HydrometeorState,
    48	    p_full: jax.Array,
    49	    p_half: jax.Array,
    50	    rho: jax.Array,
    51	    dz: jax.Array,
    52	    dt: float,
    53	    config: ThompsonConfig = ThompsonConfig(),
    54	) -> MicrophysicsOutput:
    55	    """Compute Thompson hybrid-moment microphysics tendencies.
    56	
    57	    Parameters
    58	    ----------
    59	    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
    60	        Same interface as all microphysics backends.
    61	
    62	    Returns
    63	    -------
    64	    MicrophysicsOutput
    65	    """
    66	    ncol, nlev = T.shape
    67	    q_c = hydrometeors.q_c
    68	    q_r = hydrometeors.q_r
    69	    q_i = hydrometeors.q_i
    70	    q_s = hydrometeors.q_s
    71	    q_g = hydrometeors.q_g
    72	    N_c = hydrometeors.N_c
    73	    N_r = hydrometeors.N_r
    74	    N_i = hydrometeors.N_i
    75	    sharpness = config.saturation_sharpness
    76	
    77	    N_c_eff = effective_Nc(N_c, config.Nc_0)
    78	
    79	    # === WARM RAIN ===
    80	    # Saturation adjustment — convert increment [kg/kg] to tendency [kg/kg/s]
    81	    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
    82	
    83	    # Gamma distribution corrections
    84	    gamma_c = _gamma_ratio(config.mu_c)
    85	    gamma_r = _gamma_ratio(config.mu_r)
    86	    gamma_c_norm = gamma_c / _gamma_ratio(0.0)  # normalize to mu=0 baseline (=24)
    87	    gamma_r_norm = gamma_r / _gamma_ratio(0.0)
    88	
    89	    # Autoconversion (gamma-corrected)
    90	    dq_c_au, dN_r_au, x_c = autoconversion_sb(
    91	        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness, gamma_norm=gamma_c_norm,
    92	    )
    93	
    94	    # Accretion (gamma-corrected)
    95	    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac, gamma_norm=gamma_r_norm)
    96	
    97	    # Self-collection / breakup
    98	    dN_r_sc, dN_r_br = self_collection_breakup(
    99	        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
   100	    )
   101	
   102	    # Rain evaporation
   103	    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)
   104	
   105	    # === ICE PHASE (Morrison processes) ===
   106	    T_freeze = constants.T_freeze
   107	    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))
   108	
   109	    # Ice nucleation
   110	    N_i_target = config.N_i0 * jnp.exp(
   111	        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
   112	    ) / jnp.clip(rho, 0.1)
   113	    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
   114	
   115	    # Depositional growth
   116	    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
   117	    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
   118	    dq_i_dep = (
   119	        config.dep_coeff
   120	        * jnp.maximum(S_i, 0.0)
   121	        * jnp.clip(q_i, 0.0)
   122	        * jnp.clip(N_i, 0.0) ** (1.0 / 3.0)
   123	        * f_ice
   124	    )
   125	
   126	    # Bergeron
   127	    berg_window = (
   128	        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
   129	        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
   130	    )
   131	    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window
   132	
   133	    # Riming
   134	    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   135	    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   136	    total_riming = riming_i + riming_s
   137	
   138	    # Aggregation
   139	    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice
   140	
   141	    # Melting
   142	    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
   143	    melt_ice = config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac
   144	    melt_snow = config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac
   145	
   146	    # === GRAUPEL (Thompson extension) ===
   147	    graupel_frac = jax.nn.sigmoid(
   148	        config.graupel_sharpness * (total_riming - config.rime_to_graupel_threshold)
   149	    )
   150	    rime_to_graupel = config.rime_to_graupel_rate * total_riming * graupel_frac
   151	    melt_graupel = config.melt_rate * jnp.clip(q_g, 0.0) * melt_frac
   152	
   153	    # === SEDIMENTATION ===
   154	    rho_sfc = rho[:, -1:]
   155	    V_t_r = config.a_v_r * (jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_r
   156	    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
   157	    V_t_i = config.a_v_i * (jnp.clip(q_i, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_i
   158	    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
   159	    V_t_s = config.a_v_s * (jnp.clip(q_s, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_s
   160	    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)
   161	    V_t_g = config.a_v_g * (jnp.clip(q_g, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_g
   162	    V_t_g = jnp.clip(V_t_g, 0.0, 30.0)
   163	
   164	    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
   165	    sed_i = sedimentation_tendency(q_i, rho, V_t_i, dz)
   166	    sed_s = sedimentation_tendency(q_s, rho, V_t_s, dz)
   167	    sed_g = sedimentation_tendency(q_g, rho, V_t_g, dz)
   168	
   169	    # === LATENT HEATING ===
   170	    L_v = constants.L_v
   171	    L_s = constants.L_s
   172	    L_f = constants.L_f
   173	    c_pd = constants.c_pd
   174	    dT_dt = (
   175	        L_v * condensation / c_pd
   176	        - L_v * evaporation / c_pd
   177	        + L_s * dq_i_dep / c_pd
   178	        - L_f * (melt_ice + melt_snow + melt_graupel) / c_pd
   179	    )
   180	
   181	    # === COMBINE TENDENCIES ===
   182	    dq_v_dt = -condensation + evaporation - dq_i_dep
   183	    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
   184	    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + melt_graupel + sed_r
   185	    dq_i_dt = dq_i_dep + bergeron + riming_i - aggregation - melt_ice - rime_to_graupel + sed_i
   186	    dq_s_dt = aggregation + riming_s - melt_snow - rime_to_graupel * 0.5 + sed_s
   187	    dq_g_dt = rime_to_graupel * 1.5 - melt_graupel + sed_g
   188	
   189	    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
   190	    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
   191	    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
   192	
   193	    # Precipitation
   194	    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
   195	    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)
   196	    precip_s = jnp.clip(q_s[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_s[:, -1], 0.0)
   197	    precip_g = jnp.clip(q_g[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_g[:, -1], 0.0)
   198	    precipitation = precip_r + precip_i + precip_s + precip_g
   199	
   200	    return MicrophysicsOutput(
   201	        dT_dt=dT_dt,
   202	        dq_v_dt=dq_v_dt,
   203	        dq_c_dt=dq_c_dt,
   204	        dq_r_dt=dq_r_dt,
   205	        dq_i_dt=dq_i_dt,
   206	        dq_s_dt=dq_s_dt,
   207	        dq_g_dt=dq_g_dt,
   208	        dN_c_dt=dN_c_dt,
   209	        dN_r_dt=dN_r_dt,
   210	        dN_i_dt=dN_i_dt,
   211	        precipitation=precipitation,
   212	    )

exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/microphysics/config.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/microphysics/output.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Configuration for atmospheric microphysics schemes.
     2	
     3	Provides configuration NamedTuples for:
     4	1. Kessler — warm-rain one-moment (refactored from physics/kessler.py)
     5	2. Sundqvist — large-scale diagnostic condensation
     6	3. Seifert-Beheng — two-moment warm rain
     7	4. Morrison — double-moment ice+liquid
     8	5. Thompson — hybrid moment with graupel
     9	6. ML Emulator — Equinox MLP surrogate
    10	7. Top-level MicrophysicsConfig that selects the active scheme.
    11	
    12	References
    13	----------
    14	- Kessler (1969): On the Distribution and Continuity of Water Substance.
    15	- Sundqvist et al. (1989): Condensation and cloud parameterization studies.
    16	- Seifert & Beheng (2001): A two-moment cloud microphysics scheme.
    17	- Morrison et al. (2005): A new double-moment microphysics scheme.
    18	- Thompson et al. (2008): Explicit forecasts of winter precipitation.
    19	"""
    20	
    21	from __future__ import annotations
    22	
    23	from typing import NamedTuple
    24	
    25	
    26	class KesslerConfig(NamedTuple):
    27	    """Configuration for Kessler warm-rain microphysics."""
    28	    autoconversion_threshold: float = 1.0e-3   # q_c threshold [kg/kg]
    29	    autoconversion_rate: float = 1.0e-3         # Rate [1/s]
    30	    accretion_coeff: float = 2.2                # Collection coefficient
    31	    evaporation_coeff: float = 1.0              # Evaporation coefficient
    32	    rain_fall_speed: float = 5.0                # Terminal velocity [m/s]
    33	    saturation_sharpness: float = 100.0         # Smooth switch sharpness
    34	
    35	
    36	class SundqvistConfig(NamedTuple):
    37	    """Configuration for Sundqvist large-scale condensation."""
    38	    RH_crit: float = 0.8              # Critical relative humidity
    39	    sigmoid_sharpness: float = 20.0   # Sharpness for smooth activation
    40	    auto_rate: float = 1e-3           # Autoconversion rate [1/s]
    41	    evap_coeff: float = 5e-4          # Sub-cloud evaporation coefficient
    42	
    43	
    44	class SeifertBehengConfig(NamedTuple):
    45	    """Configuration for Seifert-Beheng two-moment warm rain."""
    46	    k_au: float = 6e2                # Autoconversion rate [1/(kg*s)]
    47	    x_star: float = 2.6e-10          # Separation mass [kg]
    48	    Nc_0: float = 1e8                # Initial cloud droplet number [1/kg]
    49	    k_ac: float = 5.25               # Accretion rate [m^3/(kg*s)]
    50	    k_sc: float = 1e-3               # Self-collection rate [m^3/(kg*s)]
    51	    D_eq: float = 1.1e-3             # Equilibrium breakup diameter [m]
    52	    breakup_sharpness: float = 1e4   # Sigmoid sharpness for breakup
    53	    a_v_r: float = 130.0             # Rain fall speed coefficient a [m^(1-b)/s]
    54	    b_v_r: float = 0.5               # Rain fall speed exponent b
    55	    evap_coeff: float = 1.0          # Evaporation coefficient
    56	    saturation_sharpness: float = 100.0  # Sigmoid sharpness for saturation
    57	
    58	
    59	class MorrisonConfig(NamedTuple):
    60	    """Configuration for Morrison double-moment (ice+liquid)."""
    61	    # Warm rain (same as SB)
    62	    k_au: float = 6e2
    63	    x_star: float = 2.6e-10
    64	    Nc_0: float = 1e8
    65	    k_ac: float = 5.25
    66	    k_sc: float = 1e-3
    67	    D_eq: float = 1.1e-3
    68	    breakup_sharpness: float = 1e4
    69	    a_v_r: float = 130.0
    70	    b_v_r: float = 0.5
    71	    evap_coeff: float = 1.0
    72	    saturation_sharpness: float = 100.0
    73	    # Ice nucleation (Cooper 1986)
    74	    N_i0: float = 5e3               # Base ice crystal number [1/m^3]
    75	    cooper_a: float = 0.304          # Cooper exponent
    76	    cooper_T_act: float = 265.0      # Activation temperature [K]
    77	    ice_sigmoid_sharpness: float = 5.0  # Sharpness for ice-liquid partition
    78	    # Depositional growth
    79	    dep_coeff: float = 1e-3          # Deposition growth coefficient
    80	    # Bergeron
    81	    bergeron_rate: float = 1e-3      # Bergeron conversion rate [1/s]
    82	    T_center: float = 258.0          # Bergeron T window center [K]
    83	    T_width: float = 10.0            # Bergeron T window width [K]
    84	    # Riming
    85	    rime_coeff: float = 1.0          # Riming collection efficiency
    86	    # Aggregation
    87	    agg_coeff: float = 1e-3          # Ice-to-snow aggregation rate [1/s]
    88	    # Melting
    89	    melt_rate: float = 5e-3          # Melting rate [1/s]
    90	    melt_sharpness: float = 2.0      # Sigmoid sharpness near T_freeze
    91	    # Ice sedimentation
    92	    a_v_i: float = 50.0              # Ice fall speed coefficient [m^(1-b)/s]
    93	    b_v_i: float = 0.25              # Ice fall speed exponent
    94	    # Snow sedimentation
    95	    a_v_s: float = 30.0              # Snow fall speed coefficient
    96	    b_v_s: float = 0.3               # Snow fall speed exponent
    97	
    98	
    99	class ThompsonConfig(NamedTuple):
   100	    """Configuration for Thompson hybrid-moment microphysics."""
   101	    # All Morrison params
   102	    k_au: float = 6e2
   103	    x_star: float = 2.6e-10
   104	    Nc_0: float = 1e8
   105	    k_ac: float = 5.25
   106	    k_sc: float = 1e-3
   107	    D_eq: float = 1.1e-3
   108	    breakup_sharpness: float = 1e4
   109	    a_v_r: float = 130.0
   110	    b_v_r: float = 0.5
   111	    evap_coeff: float = 1.0
   112	    saturation_sharpness: float = 100.0
   113	    N_i0: float = 5e3
   114	    cooper_a: float = 0.304
   115	    cooper_T_act: float = 265.0
   116	    ice_sigmoid_sharpness: float = 5.0
   117	    dep_coeff: float = 1e-3
   118	    bergeron_rate: float = 1e-3
   119	    T_center: float = 258.0
   120	    T_width: float = 10.0
   121	    rime_coeff: float = 1.0
   122	    agg_coeff: float = 1e-3
   123	    melt_rate: float = 5e-3
   124	    melt_sharpness: float = 2.0
   125	    a_v_i: float = 50.0
   126	    b_v_i: float = 0.25
   127	    a_v_s: float = 30.0
   128	    b_v_s: float = 0.3
   129	    # Graupel
   130	    rime_to_graupel_threshold: float = 1e-4  # Riming threshold for graupel [kg/kg/s]
   131	    rime_to_graupel_rate: float = 0.5        # Fraction converted to graupel
   132	    graupel_sharpness: float = 1e4           # Sigmoid sharpness
   133	    a_v_g: float = 80.0                      # Graupel fall speed coefficient
   134	    b_v_g: float = 0.4                       # Graupel fall speed exponent
   135	    # Gamma distribution shape
   136	    mu_c: float = 3.0                        # Cloud droplet shape parameter
   137	    mu_r: float = 1.0                        # Rain drop shape parameter
   138	
   139	
   140	class MLEmulatorConfig(NamedTuple):
   141	    """Configuration for ML microphysics emulator (Equinox MLP)."""
   142	    n_input: int = 9
   143	    n_hidden: int = 128
   144	    n_layers: int = 3
   145	    n_output: int = 7
   146	    seed: int = 0
   147	    use_residual: bool = True
   148	    norm_T: float = 300.0       # Temperature scale [K] for input normalization
   149	    norm_q_factor: float = 1e3  # q_v / q_c / q_r / q_i scale
   150	    norm_rho: float = 1.2       # Air density scale [kg/m^3]
   151	    norm_dz: float = 1000.0     # Layer thickness scale [m]
   152	    norm_dt: float = 3600.0     # Time-step scale [s]
   153	
   154	
   155	class MicrophysicsConfig(NamedTuple):
   156	    """Top-level microphysics configuration.
   157	
   158	    Selects the active scheme and holds sub-configurations.
   159	
   160	    Fields
   161	    ------
   162	    scheme : str
   163	        Active scheme: "kessler", "sundqvist", "seifert_beheng",
   164	        "morrison", "thompson", "ml_emulator", or "none".
   165	    kessler : KesslerConfig
   166	    sundqvist : SundqvistConfig
   167	    seifert_beheng : SeifertBehengConfig
   168	    morrison : MorrisonConfig
   169	    thompson : ThompsonConfig
   170	    ml_emulator : MLEmulatorConfig
   171	    """
   172	    scheme: str = "none"
   173	    kessler: KesslerConfig = KesslerConfig()
   174	    sundqvist: SundqvistConfig = SundqvistConfig()
   175	    seifert_beheng: SeifertBehengConfig = SeifertBehengConfig()
   176	    morrison: MorrisonConfig = MorrisonConfig()
   177	    thompson: ThompsonConfig = ThompsonConfig()
   178	    ml_emulator: MLEmulatorConfig = MLEmulatorConfig()

 succeeded in 0ms:
     1	"""Microphysics output containers.
     2	
     3	HydrometeorState holds the prognostic hydrometeor fields passed to backends.
     4	MicrophysicsOutput is the common interface returned by all backends.
     5	
     6	All backends accept and return the same containers so that integration
     7	code can be backend-agnostic.
     8	"""
     9	
    10	from __future__ import annotations
    11	
    12	from typing import NamedTuple
    13	
    14	import jax
    15	import jax.numpy as jnp
    16	
    17	
    18	class HydrometeorState(NamedTuple):
    19	    """Hydrometeor state for backends. All fields shape (ncol, nlev)."""
    20	    q_c: jax.Array    # cloud water [kg/kg]
    21	    q_r: jax.Array    # rain water [kg/kg]
    22	    q_i: jax.Array    # cloud ice [kg/kg]
    23	    q_s: jax.Array    # snow [kg/kg]
    24	    q_g: jax.Array    # graupel [kg/kg]
    25	    N_c: jax.Array    # cloud droplet number [1/kg]
    26	    N_r: jax.Array    # rain drop number [1/kg]
    27	    N_i: jax.Array    # ice crystal number [1/kg]
    28	
    29	
    30	class MicrophysicsOutput(NamedTuple):
    31	    """Backend-agnostic output. All (ncol, nlev) except precipitation (ncol,)."""
    32	    dT_dt: jax.Array          # latent heating [K/s]
    33	    dq_v_dt: jax.Array        # vapor tendency [kg/kg/s]
    34	    dq_c_dt: jax.Array        # cloud water tendency
    35	    dq_r_dt: jax.Array        # rain tendency
    36	    dq_i_dt: jax.Array        # ice tendency
    37	    dq_s_dt: jax.Array        # snow tendency
    38	    dq_g_dt: jax.Array        # graupel tendency
    39	    dN_c_dt: jax.Array        # cloud number tendency [1/kg/s]
    40	    dN_r_dt: jax.Array        # rain number tendency
    41	    dN_i_dt: jax.Array        # ice number tendency
    42	    precipitation: jax.Array  # surface precip [kg/m^2/s]
    43	
    44	
    45	def make_zero_hydrometeors(
    46	    ncol: int, nlev: int, dtype=None,
    47	) -> HydrometeorState:
    48	    """Create a zero-initialized HydrometeorState.
    49	
    50	    ``dtype`` defaults to the JAX default float (``float64`` under x64,
    51	    ``float32`` otherwise).  Callers integrating with the column physics
    52	    pipeline should pass the upstream state dtype explicitly so this
    53	    fallback never silently promotes a float32 column path to float64.
    54	    """
    55	    z = jnp.zeros((ncol, nlev), dtype=dtype)
    56	    return HydrometeorState(
    57	        q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
    58	        N_c=z, N_r=z, N_i=z,
    59	    )
    60	
    61	
    62	def make_zero_output(
    63	    ncol: int, nlev: int, dtype=None,
    64	) -> MicrophysicsOutput:
    65	    """Create a zero-initialized MicrophysicsOutput.
    66	
    67	    ``dtype`` is forwarded to ``jnp.zeros`` for the same reason as
    68	    ``make_zero_hydrometeors``: defaulting allows x64 mode to silently
    69	    promote the precip path.
    70	    """
    71	    z2 = jnp.zeros((ncol, nlev), dtype=dtype)
    72	    z1 = jnp.zeros((ncol,), dtype=dtype)
    73	    return MicrophysicsOutput(
    74	        dT_dt=z2, dq_v_dt=z2, dq_c_dt=z2, dq_r_dt=z2,
    75	        dq_i_dt=z2, dq_s_dt=z2, dq_g_dt=z2,
    76	        dN_c_dt=z2, dN_r_dt=z2, dN_i_dt=z2,
    77	        precipitation=z1,
    78	    )
    79	
    80	
    81	def sedimentation_tendency(
    82	    q: jax.Array,
    83	    rho: jax.Array,
    84	    V_t: jax.Array,
    85	    dz: jax.Array,
    86	) -> jax.Array:
    87	    """Compute sedimentation tendency from vertical flux divergence.
    88	
    89	    Parameters
    90	    ----------
    91	    q : jax.Array
    92	        Hydrometeor mixing ratio [kg/kg], shape (ncol, nlev).
    93	    rho : jax.Array
    94	        Air density [kg/m^3], shape (ncol, nlev).
    95	    V_t : jax.Array
    96	        Terminal velocity [m/s], shape (ncol, nlev).
    97	    dz : jax.Array
    98	        Layer thickness [m], shape (ncol, nlev).
    99	
   100	    Returns
   101	    -------
   102	    jax.Array
   103	        Sedimentation tendency [kg/kg/s], shape (ncol, nlev).
   104	    """
   105	    q_pos = jnp.clip(q, 0.0, None)
   106	    flux = V_t * q_pos * rho  # (ncol, nlev)
   107	
   108	    # Flux from above: zero at top, flux[k-1] enters level k.  Use
   109	    # ``jnp.pad`` (single Pad HLO) instead of allocating a fresh
   110	    # zero buffer + concatenate.
   111	    flux_in = jnp.pad(flux[:, :-1], ((0, 0), (1, 0)))
   112	    dz_safe = jnp.clip(dz, 1.0, None)
   113	    return (flux_in - flux) / (rho * dz_safe)

exec
/bin/zsh -lc 'rg --files tests src/legoesm/atmosphere/physics/microphysics src/legoesm/atmosphere/physics | rg "microphysics|conserv|test"' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'rg -n "microphysics|conservation|xfail|dT_dt|dq_.*_dt|precipitation|sedimentation_tendency" tests src/legoesm | head -n 300' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/tuning.py:254:    radiation, cloud_scheme, microphysics, C_H, C_E, albedo_ice,
src/legoesm/tuning.py:275:        ns.microphysics = config.microphysics
src/legoesm/tuning.py:368:    if cfg.microphysics != "none" and cfg.cloud_scheme == "none":
src/legoesm/diagnostics/precision_drift.py:448:    # tracers under full microphysics).
tests/core/test_vertical_remap.py:78:    def test_conservation(self):
src/legoesm/cli.py:137:    from legoesm.core.conservation import compute_conservation_diagnostics
src/legoesm/cli.py:166:        use_conservation_fixer=True,
src/legoesm/cli.py:175:    diagnostics = [compute_conservation_diagnostics(state, grid)]
src/legoesm/cli.py:182:            diag = compute_conservation_diagnostics(state, grid)
src/legoesm/cli.py:217:                plot_global_field, plot_conservation_timeseries,
src/legoesm/cli.py:229:            plot_conservation_timeseries(
src/legoesm/cli.py:232:                save_path=os.path.join(args.output, f"williamson{args.case}_conservation.png"),
src/legoesm/diagnostics/energy_budget.py:5:for validating conservation in climate simulations.
src/legoesm/diagnostics/energy_budget.py:479:    precip_rate: float       # precipitation rate [mm/day]
src/legoesm/diagnostics/energy_budget.py:486:    Tracks column-integrated water vapor and its tendency, precipitation,
src/legoesm/diagnostics/energy_budget.py:536:        # Fuse the column-water-vapor + precipitation means into one
src/legoesm/atmosphere/physics/ml_parameterization.py:16:from legoesm.atmosphere.physics.microphysics.output import MicrophysicsOutput
src/legoesm/atmosphere/physics/ml_parameterization.py:17:from legoesm.atmosphere.physics.microphysics.sundqvist import (
src/legoesm/atmosphere/physics/ml_parameterization.py:73:            dT_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:74:            dq_v_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:118:        dT_dt=(T_new - T) / dt,
src/legoesm/atmosphere/physics/ml_parameterization.py:119:        dq_v_dt=(q_new - q_v) / dt,
src/legoesm/atmosphere/physics/ml_parameterization.py:137:    microphysics_scheme: str = "none",
src/legoesm/atmosphere/physics/ml_parameterization.py:148:        microphysics_scheme=microphysics_scheme,
src/legoesm/atmosphere/physics/ml_parameterization.py:177:    microphysics_scheme = assets.model.microphysics_scheme
src/legoesm/atmosphere/physics/ml_parameterization.py:196:                microphysics_scheme=microphysics_scheme,
src/legoesm/atmosphere/physics/ml_parameterization.py:222:        microphysics_scheme=microphysics_scheme,
src/legoesm/atmosphere/physics/ml_parameterization.py:229:    if microphysics_scheme == "kessler":
src/legoesm/atmosphere/physics/ml_parameterization.py:232:                "dq_v_dt_micro": unpacked["dq_v_dt_micro"],
src/legoesm/atmosphere/physics/ml_parameterization.py:233:                "dq_c_dt_micro": unpacked["dq_c_dt_micro"],
src/legoesm/atmosphere/physics/ml_parameterization.py:234:                "dq_r_dt_micro": unpacked["dq_r_dt_micro"],
src/legoesm/atmosphere/physics/ml_parameterization.py:238:    elif microphysics_scheme == "sundqvist":
src/legoesm/atmosphere/physics/ml_parameterization.py:247:def _apply_predicted_kessler_microphysics(
src/legoesm/atmosphere/physics/ml_parameterization.py:251:    """Convert direct ML microphysics outputs into the common backend interface."""
src/legoesm/atmosphere/physics/ml_parameterization.py:252:    dq_v_dt = predicted["dq_v_dt_micro"]
src/legoesm/atmosphere/physics/ml_parameterization.py:253:    zeros = jnp.zeros_like(dq_v_dt, dtype=dtype)
src/legoesm/atmosphere/physics/ml_parameterization.py:255:        dT_dt=-constants.L_v * dq_v_dt / constants.c_pd,
src/legoesm/atmosphere/physics/ml_parameterization.py:256:        dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/ml_parameterization.py:257:        dq_c_dt=predicted["dq_c_dt_micro"],
src/legoesm/atmosphere/physics/ml_parameterization.py:258:        dq_r_dt=predicted["dq_r_dt_micro"],
src/legoesm/atmosphere/physics/ml_parameterization.py:259:        dq_i_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:260:        dq_s_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:261:        dq_g_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:265:        precipitation=predicted["precip_micro"],
src/legoesm/atmosphere/physics/ml_parameterization.py:269:def _limit_predicted_kessler_microphysics_tendencies(
src/legoesm/atmosphere/physics/ml_parameterization.py:280:    limited["dq_v_dt_micro"] = jnp.maximum(
src/legoesm/atmosphere/physics/ml_parameterization.py:281:        predicted["dq_v_dt_micro"],
src/legoesm/atmosphere/physics/ml_parameterization.py:284:    limited["dq_c_dt_micro"] = jnp.maximum(
src/legoesm/atmosphere/physics/ml_parameterization.py:285:        predicted["dq_c_dt_micro"],
src/legoesm/atmosphere/physics/ml_parameterization.py:288:    limited["dq_r_dt_micro"] = jnp.maximum(
src/legoesm/atmosphere/physics/ml_parameterization.py:289:        predicted["dq_r_dt_micro"],
src/legoesm/atmosphere/physics/ml_parameterization.py:309:    """Rebuild Sundqvist microphysics from a learned column rain-survival fraction."""
src/legoesm/atmosphere/physics/ml_parameterization.py:331:    precipitation = jnp.clip(
src/legoesm/atmosphere/physics/ml_parameterization.py:338:        dT_dt=constants.L_v * (rates.condensation - evaporation) / constants.c_pd,
src/legoesm/atmosphere/physics/ml_parameterization.py:339:        dq_v_dt=-rates.condensation + evaporation,
src/legoesm/atmosphere/physics/ml_parameterization.py:340:        dq_c_dt=rates.condensation - rates.autoconversion,
src/legoesm/atmosphere/physics/ml_parameterization.py:341:        dq_r_dt=rates.autoconversion - evaporation,
src/legoesm/atmosphere/physics/ml_parameterization.py:342:        dq_i_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:343:        dq_s_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:344:        dq_g_dt=zeros,
src/legoesm/atmosphere/physics/ml_parameterization.py:348:        precipitation=precipitation,
src/legoesm/atmosphere/physics/ml_parameterization.py:395:    if assets.model.microphysics_scheme == "kessler":
src/legoesm/atmosphere/physics/ml_parameterization.py:396:        predicted = _limit_predicted_kessler_microphysics_tendencies(
src/legoesm/atmosphere/physics/ml_parameterization.py:433:    if assets.model.microphysics_scheme == "kessler":
src/legoesm/atmosphere/physics/ml_parameterization.py:434:        micro_out = _apply_predicted_kessler_microphysics(predicted, T.dtype)
tests/sea_ice/validation/test_sea_ice_validation.py:4:  - Transport: conservation, non-negativity, temperature bounds
tests/sea_ice/validation/test_sea_ice_validation.py:5:  - ITD remap: category-bound preservation, volume conservation,
tests/sea_ice/validation/test_sea_ice_validation.py:61:        has_precipitation=jnp.ones(shape),
tests/sea_ice/validation/test_sea_ice_validation.py:173:        # Approximate conservation (clamping can break exact conservation)
src/legoesm/timestepping/ssp_rk3.py:5:forward Euler method. It is the workhorse for hyperbolic conservation laws.
tests/parallel/test_scaling_operators.py:420:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/parallel/test_scaling_operators.py:423:        assert tend.dT_dt.data.shape == (mesh.nCells, nlev)
src/legoesm/ml/conservation.py:1:"""Post-hoc conservation correctors for SFNO predictions.
src/legoesm/ml/conservation.py:3:Neural operators do not enforce conservation laws by construction.
src/legoesm/ml/conservation.py:129:# Ocean conservation correctors
src/legoesm/training/training_driver.py:48:        microphysics="none",
src/legoesm/atmosphere/physics/combined.py:5:microphysics, and **gravity wave drag** tendencies. Each sub-module
src/legoesm/atmosphere/physics/combined.py:31:...     microphysics=MicrophysicsConfig(scheme="kessler"),
src/legoesm/atmosphere/physics/combined.py:53:from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
src/legoesm/atmosphere/physics/combined.py:65:from legoesm.atmosphere.physics.microphysics.integration import (
src/legoesm/atmosphere/physics/combined.py:66:    make_microphysics_physics,
src/legoesm/atmosphere/physics/combined.py:90:    microphysics : MicrophysicsConfig
src/legoesm/atmosphere/physics/combined.py:100:    microphysics: MicrophysicsConfig = MicrophysicsConfig()
src/legoesm/atmosphere/physics/combined.py:162:    if config.microphysics.scheme != "none":
src/legoesm/atmosphere/physics/combined.py:163:        tagged_fns.append((make_microphysics_physics(config.microphysics, model_type, dt), False, None))
src/legoesm/atmosphere/physics/combined.py:183:                dT_dt=Field(data=jnp.zeros_like(state.T.data), name="dT_dt_phys",
src/legoesm/atmosphere/physics/combined.py:210:        dT_dt = first.dT_dt.data
src/legoesm/atmosphere/physics/combined.py:238:            dT_dt = dT_dt + t.dT_dt.data
src/legoesm/atmosphere/physics/combined.py:254:                         dims=first.dT_dt.dims, units="kg/kg/s")
src/legoesm/atmosphere/physics/combined.py:264:            dT_dt=first.dT_dt.replace(data=dT_dt),
src/legoesm/atmosphere/physics/combined.py:303:    if config.microphysics.scheme != "none":
src/legoesm/atmosphere/physics/combined.py:304:        tagged_fns.append((make_microphysics_physics(config.microphysics, "nonhydrostatic", dt), False, None))
src/legoesm/atmosphere/physics/combined.py:416:    if config.microphysics.scheme != "none":
src/legoesm/atmosphere/physics/combined.py:417:        tagged_fns.append((make_microphysics_physics(config.microphysics, "spectral_pe", dt), False, None))
tests/test_mpas_conservation.py:3:Tests mass/energy/enstrophy conservation for shallow water
tests/test_mpas_conservation.py:4:and volume/heat/salt conservation for the ocean model.
tests/test_mpas_conservation.py:15:from legoesm.core.conservation import global_integral_voronoi
tests/test_mpas_conservation.py:46:    """Compute shallow water conservation diagnostics."""
tests/test_mpas_conservation.py:77:    """Compute ocean conservation diagnostics."""
tests/test_mpas_conservation.py:132:def check_sw_conservation():
tests/test_mpas_conservation.py:133:    """Check conservation for shallow water on icosahedral grid."""
tests/test_mpas_conservation.py:198:def check_ocean_conservation():
tests/test_mpas_conservation.py:199:    """Check conservation for ocean PE on icosahedral grid."""
tests/test_mpas_conservation.py:226:            use_conservation_fixer=(fix_vol or fix_heat or fix_salt),
tests/test_mpas_conservation.py:309:    check_sw_conservation()
tests/test_mpas_conservation.py:310:    check_ocean_conservation()
src/legoesm/atmosphere/physics/learned_column.py:149:        dT_dt = y[:, :nlev].reshape(n_lat, n_lon, nlev)
src/legoesm/atmosphere/physics/learned_column.py:152:        dT_hat = sh_analysis_3d(grid_, dT_dt.astype(jnp.float64))
src/legoesm/ml/physics/data.py:15:from legoesm.atmosphere.physics.microphysics.config import (
src/legoesm/ml/physics/data.py:19:from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
src/legoesm/ml/physics/data.py:20:from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
src/legoesm/ml/physics/data.py:21:from legoesm.atmosphere.physics.microphysics.sundqvist import (
src/legoesm/ml/physics/data.py:65:    dq_v_dt_micro: jax.Array
src/legoesm/ml/physics/data.py:66:    dq_c_dt_micro: jax.Array
src/legoesm/ml/physics/data.py:67:    dq_r_dt_micro: jax.Array
src/legoesm/ml/physics/data.py:71:    microphysics_scheme: str = "none"
src/legoesm/ml/physics/data.py:92:        dq_v_dt_micro=dataset.dq_v_dt_micro[idx],
src/legoesm/ml/physics/data.py:93:        dq_c_dt_micro=dataset.dq_c_dt_micro[idx],
src/legoesm/ml/physics/data.py:94:        dq_r_dt_micro=dataset.dq_r_dt_micro[idx],
src/legoesm/ml/physics/data.py:98:        microphysics_scheme=dataset.microphysics_scheme,
src/legoesm/ml/physics/data.py:179:    micro_scheme = getattr(driver.config, "microphysics", "none")
src/legoesm/ml/physics/data.py:180:    dq_v_dt_micro = zeros_3d
src/legoesm/ml/physics/data.py:181:    dq_c_dt_micro = zeros_3d
src/legoesm/ml/physics/data.py:182:    dq_r_dt_micro = zeros_3d
src/legoesm/ml/physics/data.py:201:            micro_out = kessler_microphysics(
src/legoesm/ml/physics/data.py:212:            dq_v_dt_micro = _copy_array(micro_out.dq_v_dt)
src/legoesm/ml/physics/data.py:213:            dq_c_dt_micro = _copy_array(micro_out.dq_c_dt)
src/legoesm/ml/physics/data.py:214:            dq_r_dt_micro = _copy_array(micro_out.dq_r_dt)
src/legoesm/ml/physics/data.py:215:            precip_micro = _copy_array(micro_out.precipitation)
src/legoesm/ml/physics/data.py:234:                    jnp.clip(rates.precipitation / generated_rain_flux, 0.0, 1.0),
src/legoesm/ml/physics/data.py:238:            dq_v_dt_micro = _copy_array(-rates.condensation + rates.evaporation)
src/legoesm/ml/physics/data.py:239:            dq_c_dt_micro = _copy_array(rates.condensation - rates.autoconversion)
src/legoesm/ml/physics/data.py:240:            dq_r_dt_micro = _copy_array(rates.autoconversion - rates.evaporation)
src/legoesm/ml/physics/data.py:241:            precip_micro = _copy_array(rates.precipitation)
src/legoesm/ml/physics/data.py:245:            "microphysics='none', 'kessler', or 'sundqvist'",
src/legoesm/ml/physics/data.py:273:        dq_v_dt_micro=dq_v_dt_micro,
src/legoesm/ml/physics/data.py:274:        dq_c_dt_micro=dq_c_dt_micro,
src/legoesm/ml/physics/data.py:275:        dq_r_dt_micro=dq_r_dt_micro,
src/legoesm/ml/physics/data.py:279:        microphysics_scheme=micro_scheme,
src/legoesm/ml/physics/data.py:297:    microphysics_scheme = datasets[0].microphysics_scheme
src/legoesm/ml/physics/data.py:299:        if dataset.microphysics_scheme != microphysics_scheme:
src/legoesm/ml/physics/data.py:300:            raise ValueError("Cannot concatenate teacher datasets with mixed microphysics schemes")
src/legoesm/ml/physics/data.py:307:        dq_v_dt_micro=jnp.concatenate(
src/legoesm/ml/physics/data.py:308:            [dataset.dq_v_dt_micro for dataset in datasets],
src/legoesm/ml/physics/data.py:311:        dq_c_dt_micro=jnp.concatenate(
src/legoesm/ml/physics/data.py:312:            [dataset.dq_c_dt_micro for dataset in datasets],
src/legoesm/ml/physics/data.py:315:        dq_r_dt_micro=jnp.concatenate(
src/legoesm/ml/physics/data.py:316:            [dataset.dq_r_dt_micro for dataset in datasets],
src/legoesm/ml/physics/data.py:328:        microphysics_scheme=microphysics_scheme,
src/legoesm/ml/physics/data.py:350:    if experiment_config.microphysics not in ("none", "kessler", "sundqvist"):
src/legoesm/ml/physics/data.py:353:            "microphysics='none', 'kessler', or 'sundqvist'",
src/legoesm/training/sfno_dycore_coupling.py:31:    "dq_i_dt",
src/legoesm/training/sfno_dycore_coupling.py:32:    "dq_s_dt",
src/legoesm/training/sfno_dycore_coupling.py:33:    "dq_g_dt",
src/legoesm/training/sfno_dycore_coupling.py:55:    dT_dt,
src/legoesm/training/sfno_dycore_coupling.py:56:    dq_v_dt,
src/legoesm/training/sfno_dycore_coupling.py:57:    dq_c_dt,
src/legoesm/training/sfno_dycore_coupling.py:58:    dq_r_dt,
src/legoesm/training/sfno_dycore_coupling.py:70:        dT_dt=dT_dt,
src/legoesm/training/sfno_dycore_coupling.py:71:        dq_v_dt=dq_v_dt,
src/legoesm/training/sfno_dycore_coupling.py:72:        dq_c_dt=dq_c_dt,
src/legoesm/training/sfno_dycore_coupling.py:73:        dq_r_dt=dq_r_dt,
src/legoesm/training/sfno_dycore_coupling.py:138:        dT_dt = out[..., spec.T_slice]
src/legoesm/training/sfno_dycore_coupling.py:139:        dq_v_dt = out[..., spec.q_slice]
src/legoesm/training/sfno_dycore_coupling.py:146:                dT_dt=dT_dt,
src/legoesm/training/sfno_dycore_coupling.py:147:                dq_v_dt=dq_v_dt,
src/legoesm/training/sfno_dycore_coupling.py:148:                dq_c_dt=zeros_3d,
src/legoesm/training/sfno_dycore_coupling.py:149:                dq_r_dt=zeros_3d,
src/legoesm/training/sfno_dycore_coupling.py:260:                dT_dt=trad_out.dT_dt + sfno_out.dT_dt,
src/legoesm/training/sfno_dycore_coupling.py:261:                dq_v_dt=trad_out.dq_v_dt + sfno_out.dq_v_dt,
src/legoesm/training/sfno_dycore_coupling.py:262:                dq_c_dt=trad_out.dq_c_dt + sfno_out.dq_c_dt,
src/legoesm/training/sfno_dycore_coupling.py:263:                dq_r_dt=trad_out.dq_r_dt + sfno_out.dq_r_dt,
src/legoesm/training/sfno_dycore_coupling.py:270:                reference_3d=trad_out.dT_dt,
src/legoesm/ml/physics/evaluate.py:25:    rmse_dq_v_dt_micro: float = 0.0
src/legoesm/ml/physics/evaluate.py:26:    rmse_dq_c_dt_micro: float = 0.0
src/legoesm/ml/physics/evaluate.py:27:    rmse_dq_r_dt_micro: float = 0.0
src/legoesm/ml/physics/evaluate.py:37:    microphysics_scheme = getattr(model, "microphysics_scheme", "none")
src/legoesm/ml/physics/evaluate.py:56:                microphysics_scheme=microphysics_scheme,
src/legoesm/ml/physics/evaluate.py:82:        microphysics_scheme=microphysics_scheme,
src/legoesm/ml/physics/evaluate.py:111:    if dataset.microphysics_scheme == "kessler":
src/legoesm/ml/physics/evaluate.py:113:            rmse_dq_v_dt_micro=float(
src/legoesm/ml/physics/evaluate.py:114:                jnp.sqrt(jnp.mean((predicted["dq_v_dt_micro"] - dataset.dq_v_dt_micro) ** 2))
src/legoesm/ml/physics/evaluate.py:116:            rmse_dq_c_dt_micro=float(
src/legoesm/ml/physics/evaluate.py:117:                jnp.sqrt(jnp.mean((predicted["dq_c_dt_micro"] - dataset.dq_c_dt_micro) ** 2))
src/legoesm/ml/physics/evaluate.py:119:            rmse_dq_r_dt_micro=float(
src/legoesm/ml/physics/evaluate.py:120:                jnp.sqrt(jnp.mean((predicted["dq_r_dt_micro"] - dataset.dq_r_dt_micro) ** 2))
src/legoesm/ml/physics/evaluate.py:126:    elif dataset.microphysics_scheme == "sundqvist":
tests/stress/test_phase1_sea_ice.py:44:        has_precipitation=jnp.zeros(shape),
tests/stress/test_phase1_sea_ice.py:149:    # 1A.3  Energy conservation (approximate check)
tests/stress/test_phase1_sea_ice.py:151:    def test_energy_conservation(self):
tests/stress/test_phase1_sea_ice.py:254:    def test_itd_volume_conservation(self):
tests/stress/test_phase1_sea_ice.py:291:    def test_transport_volume_conservation(self):
src/legoesm/ml/physics/train.py:81:    microphysics_scheme: str = "none",
src/legoesm/ml/physics/train.py:102:                microphysics_scheme=microphysics_scheme,
src/legoesm/ml/physics/train.py:131:        dq_v_dt_micro=dataset.dq_v_dt_micro,
src/legoesm/ml/physics/train.py:132:        dq_c_dt_micro=dataset.dq_c_dt_micro,
src/legoesm/ml/physics/train.py:133:        dq_r_dt_micro=dataset.dq_r_dt_micro,
src/legoesm/ml/physics/train.py:135:        microphysics_scheme=dataset.microphysics_scheme,
src/legoesm/ml/physics/train.py:167:        microphysics_scheme=dataset.microphysics_scheme,
src/legoesm/ml/physics/train.py:193:        microphysics_scheme=dataset.microphysics_scheme,
src/legoesm/ml/physics/train.py:248:        dq_v_dt_micro=dataset.dq_v_dt_micro[test_idx],
src/legoesm/ml/physics/train.py:249:        dq_c_dt_micro=dataset.dq_c_dt_micro[test_idx],
src/legoesm/ml/physics/train.py:250:        dq_r_dt_micro=dataset.dq_r_dt_micro[test_idx],
src/legoesm/atmosphere/physics/neural_physics.py:18:    dT_dt, dq_v_dt, dq_c_dt, dq_r_dt at each level
src/legoesm/atmosphere/physics/neural_physics.py:44:    "dq_i_dt",
src/legoesm/atmosphere/physics/neural_physics.py:45:    "dq_s_dt",
src/legoesm/atmosphere/physics/neural_physics.py:46:    "dq_g_dt",
src/legoesm/atmosphere/physics/neural_physics.py:68:    dT_dt,
src/legoesm/atmosphere/physics/neural_physics.py:69:    dq_v_dt,
src/legoesm/atmosphere/physics/neural_physics.py:70:    dq_c_dt,
src/legoesm/atmosphere/physics/neural_physics.py:71:    dq_r_dt,
src/legoesm/atmosphere/physics/neural_physics.py:83:        dT_dt=dT_dt,
src/legoesm/atmosphere/physics/neural_physics.py:84:        dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/neural_physics.py:85:        dq_c_dt=dq_c_dt,
src/legoesm/atmosphere/physics/neural_physics.py:86:        dq_r_dt=dq_r_dt,
src/legoesm/atmosphere/physics/neural_physics.py:231:    dT_dt = y[:nlev]
src/legoesm/atmosphere/physics/neural_physics.py:232:    dq_v_dt = y[nlev:2 * nlev]
src/legoesm/atmosphere/physics/neural_physics.py:233:    dq_c_dt = y[2 * nlev:3 * nlev]
src/legoesm/atmosphere/physics/neural_physics.py:234:    dq_r_dt = y[3 * nlev:4 * nlev]
src/legoesm/atmosphere/physics/neural_physics.py:238:            dT_dt=dT_dt,
src/legoesm/atmosphere/physics/neural_physics.py:239:            dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/neural_physics.py:240:            dq_c_dt=dq_c_dt,
src/legoesm/atmosphere/physics/neural_physics.py:241:            dq_r_dt=dq_r_dt,
src/legoesm/atmosphere/physics/neural_physics.py:248:            reference_3d=dT_dt,
src/legoesm/atmosphere/physics/neural_physics.py:331:                dT_dt=adapter.unflatten_3d(col_out.dT_dt),
src/legoesm/atmosphere/physics/neural_physics.py:332:                dq_v_dt=adapter.unflatten_3d(col_out.dq_v_dt),
src/legoesm/atmosphere/physics/neural_physics.py:333:                dq_c_dt=adapter.unflatten_3d(col_out.dq_c_dt),
src/legoesm/atmosphere/physics/neural_physics.py:334:                dq_r_dt=adapter.unflatten_3d(col_out.dq_r_dt),
src/legoesm/atmosphere/physics/neural_physics.py:438:                dT_dt=trad_out.dT_dt + _alpha * neural_out.dT_dt,
src/legoesm/atmosphere/physics/neural_physics.py:439:                dq_v_dt=trad_out.dq_v_dt + _alpha * neural_out.dq_v_dt,
src/legoesm/atmosphere/physics/neural_physics.py:440:                dq_c_dt=trad_out.dq_c_dt + _alpha * neural_out.dq_c_dt,
src/legoesm/atmosphere/physics/neural_physics.py:441:                dq_r_dt=trad_out.dq_r_dt + _alpha * neural_out.dq_r_dt,
src/legoesm/atmosphere/physics/neural_physics.py:448:                reference_3d=trad_out.dT_dt,
src/legoesm/ml/physics/model.py:20:    microphysics_scheme: str = eqx.field(static=True)
src/legoesm/ml/physics/model.py:27:        microphysics_scheme: str = "none",
src/legoesm/ml/physics/model.py:34:        self.microphysics_scheme = _validate_microphysics_scheme(microphysics_scheme)
src/legoesm/ml/physics/model.py:37:            microphysics_scheme=self.microphysics_scheme,
src/legoesm/ml/physics/model.py:41:            microphysics_scheme=self.microphysics_scheme,
src/legoesm/ml/physics/model.py:57:def _validate_microphysics_scheme(microphysics_scheme: str) -> str:
src/legoesm/ml/physics/model.py:58:    """Validate and normalize the supported microphysics scheme label."""
src/legoesm/ml/physics/model.py:59:    if microphysics_scheme not in ("none", "kessler", "sundqvist"):
src/legoesm/ml/physics/model.py:61:            f"Unsupported microphysics_scheme={microphysics_scheme!r}; "
src/legoesm/ml/physics/model.py:64:    return microphysics_scheme
src/legoesm/ml/physics/model.py:67:def _microphysics_feature_levels(microphysics_scheme: str) -> int:
src/legoesm/ml/physics/model.py:69:    scheme = _validate_microphysics_scheme(microphysics_scheme)
src/legoesm/ml/physics/model.py:80:    microphysics_scheme: str = "none",
src/legoesm/ml/physics/model.py:84:    size += _microphysics_feature_levels(microphysics_scheme) * nlev
src/legoesm/ml/physics/model.py:91:    microphysics_scheme: str = "none",
src/legoesm/ml/physics/model.py:94:    scheme = _validate_microphysics_scheme(microphysics_scheme)
src/legoesm/ml/physics/model.py:120:    microphysics_scheme: str = "none",
src/legoesm/ml/physics/model.py:123:    scheme = _validate_microphysics_scheme(microphysics_scheme)
src/legoesm/ml/physics/model.py:162:    dq_v_dt_micro: jax.Array | None = None,
src/legoesm/ml/physics/model.py:163:    dq_c_dt_micro: jax.Array | None = None,
src/legoesm/ml/physics/model.py:164:    dq_r_dt_micro: jax.Array | None = None,
src/legoesm/ml/physics/model.py:166:    microphysics_scheme: str = "none",
src/legoesm/ml/physics/model.py:169:    scheme = _validate_microphysics_scheme(microphysics_scheme)
src/legoesm/ml/physics/model.py:173:            dq_v_dt_micro is None
src/legoesm/ml/physics/model.py:174:            or dq_c_dt_micro is None
src/legoesm/ml/physics/model.py:175:            or dq_r_dt_micro is None
src/legoesm/ml/physics/model.py:179:                "Direct microphysics targets require "
src/legoesm/ml/physics/model.py:180:                "dq_v_dt_micro, dq_c_dt_micro, dq_r_dt_micro, and precip_micro",
src/legoesm/ml/physics/model.py:184:                dq_v_dt_micro,
src/legoesm/ml/physics/model.py:185:                dq_c_dt_micro,
src/legoesm/ml/physics/model.py:186:                dq_r_dt_micro,
src/legoesm/ml/physics/model.py:193:                "Sundqvist microphysics targets require rain_survival_fraction labels",
src/legoesm/ml/physics/model.py:203:    microphysics_scheme: str = "none",
src/legoesm/ml/physics/model.py:206:    scheme = _validate_microphysics_scheme(microphysics_scheme)
src/legoesm/ml/physics/model.py:209:        microphysics_scheme=scheme,
src/legoesm/ml/physics/model.py:222:                "dq_v_dt_micro": targets[..., offset:offset + nlev],
src/legoesm/ml/physics/model.py:223:                "dq_c_dt_micro": targets[..., offset + nlev:offset + 2 * nlev],
src/legoesm/ml/physics/model.py:224:                "dq_r_dt_micro": targets[..., offset + 2 * nlev:offset + 3 * nlev],
tests/ocean/validation/test_differentiability_ocean.py:39:        use_conservation_fixer=False,
tests/ocean/validation/test_differentiability_ocean.py:72:        use_conservation_fixer=False,
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:28:    dT_dt : jax.Array
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:36:    dT_dt: jax.Array
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:49:        dT_dt=z,
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:75:    # decreasing density (energy conservation: F ~ rho * sigma^2 = const).
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:119:    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:124:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
tests/stress/test_phase2_coupled.py:5:boundedness, conservation, and diagnostic population.
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:83:    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:89:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
tests/ocean/distributed/test_ocean_mpi_conservation.py:1:"""Long-run MPI ocean conservation regression.
tests/ocean/distributed/test_ocean_mpi_conservation.py:5:    mpirun -np 6 python -m pytest tests/ocean/distributed/test_ocean_mpi_conservation.py -v
tests/ocean/distributed/test_ocean_mpi_conservation.py:50:from legoesm.ocean.conservation import ocean_conservation_fixer
tests/ocean/distributed/test_ocean_mpi_conservation.py:102:    """Nightly long-run MPI conservation regression."""
tests/ocean/distributed/test_ocean_mpi_conservation.py:104:    def test_longrun_mpi_ocean_conservation(self, topology):
tests/ocean/distributed/test_ocean_mpi_conservation.py:136:            use_conservation_fixer=True,
tests/ocean/distributed/test_ocean_mpi_conservation.py:179:            state = ocean_conservation_fixer(
tests/stress/test_phase1_ocean_slab.py:40:        has_precipitation=jnp.ones(shape),
tests/stress/test_phase1_ocean_slab.py:90:    # 1D.2 -- Two-layer conservation and relaxation
tests/stress/test_phase1_ocean_slab.py:92:    def test_two_layer_conservation(self):
tests/atmosphere/shallow_water/integration/test_shallow_water.py:53:            use_conservation_fixer=True,
tests/atmosphere/shallow_water/integration/test_shallow_water.py:89:    def test_mass_conservation(self, model, grid, cdgrid):
tests/atmosphere/shallow_water/integration/test_shallow_water.py:148:            use_conservation_fixer=False
tests/atmosphere/shallow_water/integration/test_shallow_water.py:165:            use_conservation_fixer=False
tests/atmosphere/shallow_water/integration/test_shallow_water.py:181:        config = CDGridShallowWaterConfig(use_conservation_fixer=False)
src/legoesm/ice/itd.py:12:ensures category means stay within bounds, but conservation is only
src/legoesm/driver/model_driver.py:22:from legoesm.core.conservation import compute_global_moisture, fix_moisture_hydrostatic
src/legoesm/driver/model_driver.py:57:        # Use full moisture registry for mixed-phase/two-moment microphysics

 succeeded in 0ms:
src/legoesm/atmosphere/physics/microphysics/output.py
src/legoesm/atmosphere/physics/microphysics/sundqvist.py
src/legoesm/atmosphere/physics/microphysics/thompson.py
src/legoesm/atmosphere/physics/microphysics/integration.py
src/legoesm/atmosphere/physics/microphysics/ml_emulator.py
src/legoesm/atmosphere/physics/microphysics/__init__.py
src/legoesm/atmosphere/physics/microphysics/kessler.py
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py
src/legoesm/atmosphere/physics/microphysics/config.py
src/legoesm/atmosphere/physics/microphysics/morrison.py
src/legoesm/atmosphere/physics/microphysics/output.py
src/legoesm/atmosphere/physics/microphysics/sundqvist.py
src/legoesm/atmosphere/physics/microphysics/thompson.py
src/legoesm/atmosphere/physics/microphysics/integration.py
src/legoesm/atmosphere/physics/microphysics/ml_emulator.py
src/legoesm/atmosphere/physics/microphysics/__init__.py
src/legoesm/atmosphere/physics/microphysics/kessler.py
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py
src/legoesm/atmosphere/physics/microphysics/config.py
src/legoesm/atmosphere/physics/microphysics/morrison.py
tests/debug/spectral_hs100_l10.png
tests/debug/spectral_hs100.png
tests/debug/spectral_hs100_freedrag.png
tests/debug/spectral_hs100_correct_nu.png
tests/debug/spectral_hs100_dealias.png
tests/debug/spectral_hs100_eqinit.png
tests/debug/spectral_hs100_implicit.png
tests/debug/spectral_hs100_final.png
tests/debug/spectral_hs100_strong.png
tests/distributed/test_mpi_driver.py
tests/distributed/test_scale_mpi_halo.py
tests/distributed/test_voronoi_halo.py
tests/distributed/test_mpi_differentiability.py
tests/distributed/__init__.py
tests/distributed/test_mpi_bootstrap.py
tests/distributed/test_coupler_mpi.py
tests/distributed/test_halo_mpi.py
tests/distributed/conftest.py
tests/distributed/test_voronoi_mpi.py
tests/williamson_diagnostic.py
tests/da/test_control_vector.py
tests/da/test_minimizer.py
tests/da/test_background_error.py
tests/da/test_gen_be.py
tests/da/test_incremental.py
tests/da/test_preconditioning.py
tests/da/__init__.py
tests/da/test_cycling.py
tests/da/test_observation.py
tests/stress/test_phase5_cmip_io.py
tests/stress/test_phase1_sea_ice.py
tests/stress/test_phase0_infrastructure.py
tests/stress/test_cmip_operationalization.py
tests/stress/test_phase2_coupled.py
tests/stress/test_phase1_ocean_slab.py
tests/stress/test_phase4_coupled_mpi.py
tests/stress/test_phase1_radiation_ghg.py
tests/stress/test_phase3_restart.py
tests/stress/__init__.py
tests/stress/test_phase4_sea_ice_mpi.py
tests/stress/test_phase7_multiyear.py
tests/stress/test_phase6_cmip_e2e.py
tests/stress/test_phase1_land_carbon.py
tests/da/integration/test_ocean_4dvar.py
tests/da/integration/test_end_to_end.py
tests/da/integration/test_pe_4dvar.py
tests/da/integration/__init__.py
tests/da/test_cost_function.py
tests/validation/test_ec_eigenvalues2.py
tests/validation/test_ensemble_correctness.py
tests/validation/README_DYCORE_PROGRESSION.md
tests/validation/bench_spectral_nh.py
tests/validation/test_continuous_stability.py
tests/validation/test_corrected_eigenvalues.py
tests/validation/test_differentiability_regression.py
tests/validation/bench_spectral_sw.py
tests/validation/test_isolate_instability.py
tests/validation/test_restart_reproducibility.py
tests/validation/test_conservation_baseline.py
tests/validation/test_scaling_readiness.py
tests/validation/bench_spectral_pe.py
tests/validation/validation_differentiability_all.py
tests/validation/test_precision_amip.py
tests/validation/test_evar_instability.py
tests/validation/run_dycore_progression_suite.py
tests/validation/__init__.py
tests/validation/test_amip_validation.py
tests/validation/test_ec_eigenvalues.py
tests/test_mpas_conservation.py
tests/test_smagorinsky_biharmonic_comprehensive.py
tests/ocean/run_ocean_all_grids_matrix.py
tests/sea_ice/validation/test_sea_ice_validation.py
tests/sea_ice/validation/__init__.py
tests/sea_ice/__init__.py
tests/core/test_vertical_remap.py
tests/core/test_weno.py
tests/ocean/validation/test_differentiability_ocean.py
tests/ocean/validation/__init__.py
tests/ocean/__init__.py
tests/parallel/test_scaling_operators.py
tests/parallel/test_cubesphere_exchange.py
tests/__init__.py
tests/sea_ice/unit/__init__.py
tests/sea_ice/unit/test_surface_albedo.py
tests/atmosphere/shallow_water/validation/__init__.py
tests/atmosphere/shallow_water/__init__.py
tests/integration/__init__.py
tests/ocean/distributed/test_ocean_mpi_conservation.py
tests/ocean/distributed/__init__.py
tests/atmosphere/shallow_water/integration/test_shallow_water.py
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py
tests/atmosphere/shallow_water/integration/__init__.py
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py
tests/atmosphere/shallow_water/integration/test_boundary_fix.py
tests/atmosphere/shallow_water/integration/test_fv_convergence.py
tests/atmosphere/shallow_water/unit/test_spectral.py
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py
tests/atmosphere/shallow_water/unit/test_sfno_sw.py
tests/atmosphere/shallow_water/unit/__init__.py
tests/test_cases/cosine_bell.py
tests/test_cases/dcmip_transport.py
tests/test_cases/williamson.py
tests/test_cases/baroclinic_wave.py
tests/atmosphere/shallow_water/test_cases/williamson.py
tests/atmosphere/hydrostatic/validation/test_stability_fix.py
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py
tests/atmosphere/hydrostatic/validation/__init__.py
tests/atmosphere/shallow_water/test_cases/__init__.py
tests/atmosphere/hydrostatic/validation/test_held_suarez_fix.py
tests/atmosphere/shallow_water/test_cases/williamson_latlon.py
tests/atmosphere/shallow_water/test_cases/williamson_mpas.py
tests/atmosphere/hydrostatic/__init__.py
tests/ocean/unit/test_advection_fct_zalesak.py
tests/ocean/unit/test_barotropic_cgrid.py
tests/ocean/unit/test_ocean.py
tests/conftest.py
tests/ocean/unit/test_bathymetry.py
tests/ocean/unit/test_biharmonic_vorticity.py
tests/ocean/unit/test_smagorinsky.py
tests/ocean/unit/test_momentum_diagnostics_closure.py
tests/ocean/unit/test_advection_dst3.py
tests/ocean/unit/test_inertia_gravity_wave.py
tests/ocean/unit/test_implicit_solver.py
tests/ocean/unit/test_implicit_vertical_solver.py
tests/test_cases/williamson_latlon.py
tests/test_cases/__init__.py
tests/atmosphere/hydrostatic/integration/test_amip_smoke.py
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py
tests/atmosphere/hydrostatic/integration/__init__.py
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py
tests/atmosphere/hydrostatic/integration/test_amip_stability.py
tests/land/unit/test_multilayer_land.py
tests/land/unit/test_land_water_budget.py
tests/land/unit/test_stomata.py
tests/land/unit/test_land_audit_fixes.py
tests/land/unit/__init__.py
tests/land/unit/test_carbon_cycle.py
tests/land/__init__.py
tests/land/test_land_stability.py
tests/unit/test_diff_coupler.py
tests/unit/test_cdgrid.py
tests/unit/test_grid_dycore_fixes.py
tests/unit/test_precision_modes.py
tests/unit/test_latlon_grid.py
tests/unit/test_grid.py
tests/unit/test_cdgrid_fv3_regression.py
tests/unit/test_voronoi_trisk_weights.py
tests/unit/test_scale_mpi_layout.py
tests/unit/test_diagnostic_collector.py
tests/unit/test_learned_column.py
tests/unit/test_zhang_mcfarlane.py
tests/unit/test_scale_portability.py
tests/unit/test_driver_forcing_dispatch.py
tests/unit/test_cross_discretization.py
tests/unit/test_sea_ice_dynamics.py
tests/unit/test_convergence_rates.py
tests/unit/test_land_ice_carbon.py
tests/unit/test_cfl.py
tests/unit/test_diff_coupled_system.py
tests/unit/test_scale_metal.py
tests/unit/test_operators_latlon.py
tests/unit/test_land_ice_multilayer.py
tests/unit/test_emanuel.py
tests/unit/test_smooth.py
tests/unit/test_precision.py
tests/unit/test_async_halo.py
tests/unit/test_neuralgcm_s2s.py
tests/unit/test_moisture_budget.py
tests/atmosphere/__init__.py
tests/ocean/unit/test_leith.py
tests/ocean/unit/test_bottom_drag_sponge.py
tests/ocean/unit/test_shortwave_penetration.py
tests/ocean/unit/test_mpas_ocean.py
tests/ocean/unit/test_freshwater.py
tests/ocean/unit/test_mpas_physics.py
tests/ocean/unit/test_no_scheme_duplication.py
tests/ocean/unit/test_backscatter.py
tests/ocean/unit/test_eta_floor.py
tests/unit/test_vector_calculus_identities.py
tests/ocean/unit/test_visbeck_gm.py
tests/ocean/unit/test_advection_som.py
tests/ocean/unit/test_cross_grid_parity.py
tests/ocean/unit/test_ocean_biogeochemistry.py
tests/ocean/unit/test_ocean_fv.py
tests/ocean/unit/test_barotropic_implicit_mpas.py
tests/ocean/unit/__init__.py
tests/ocean/unit/test_gm_redi_mpas.py
tests/unit/test_physics_combined.py
tests/ocean/unit/test_eady_overrides.py
tests/ocean/unit/test_gm_redi_eady_physics.py
tests/test_cases/dcmip2025/test_case_3.py
tests/test_cases/dcmip2025/test_case_2.py
tests/test_cases/dcmip2025/common.py
tests/test_cases/dcmip2025/__init__.py
tests/test_cases/dcmip2025/test_case_1.py
tests/ocean/unit/test_surface_forcing_dispatch.py
tests/ocean/unit/test_ocean_fc.py
tests/ocean/unit/test_barotropic_noise_invariant.py
tests/ocean/unit/test_gm_redi_latlon_cgrid.py
tests/ocean/unit/test_ocean_differentiability.py
tests/ocean/unit/test_ocean_diagnostics.py
tests/ocean/unit/test_sfno_ocean.py
tests/ocean/unit/test_ocean_compatibility.py
tests/ocean/unit/test_latlon_cgrid_ocean.py
tests/ocean/unit/test_weno_momentum.py
tests/ocean/unit/test_advection_weno.py
tests/atmosphere/nonhydrostatic/__init__.py
tests/atmosphere/hydrostatic/test_cases/dcmip_transport.py
tests/atmosphere/hydrostatic/test_cases/__init__.py
tests/atmosphere/nonhydrostatic/test_cases/__init__.py
tests/unit/test_run_amip_cli.py
tests/unit/test_physics_radiation.py
tests/unit/test_voronoi_precision.py
tests/unit/test_duogrid.py
tests/unit/test_operators_fc_3d.py
tests/unit/test_parallel_runtime.py
tests/unit/test_physics_microphysics.py
tests/unit/test_config_roundtrip.py
tests/unit/test_deprecation_warnings.py
tests/unit/test_scale_latlon_spectral.py
tests/unit/test_kain_fritsch.py
tests/unit/test_diff_taylor_tests.py
tests/unit/test_scale_ensemble.py
tests/unit/test_zarr_checkpoint.py
tests/unit/test_land_params.py
tests/unit/test_neural_gcm_spectral.py
tests/unit/test_thermodynamics.py
tests/unit/test_conservation.py
tests/unit/test_scale_halo.py
tests/unit/test_backend_guard.py
tests/unit/test_gradient_checkpointing.py
tests/atmosphere/hydrostatic/unit/test_radiation.py
tests/atmosphere/hydrostatic/unit/test_convection.py
tests/atmosphere/nonhydrostatic/validation/__init__.py
tests/unit/test_scale_sharded_dynamics.py
tests/unit/test_operators.py
tests/unit/test_external_forcing.py
tests/unit/test_land_ice_lake.py
tests/unit/test_scale_jit_health.py
tests/unit/test_distributed_layout.py
tests/unit/test_conservation_laws.py
tests/unit/test_training_modules.py
tests/unit/test_physics_units.py
tests/unit/test_spectral_dycores_comprehensive.py
tests/unit/test_device_config.py
tests/unit/__init__.py
tests/unit/test_sharded_dynamics.py
tests/unit/test_diff_ocean.py
tests/unit/test_multi_gpu.py
tests/unit/test_operators_fc.py
tests/unit/test_sfno_s2s.py
tests/atmosphere/hydrostatic/unit/__init__.py
tests/atmosphere/hydrostatic/unit/test_amip_forcing.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py
tests/atmosphere/hydrostatic/unit/test_energy_budget.py
tests/atmosphere/hydrostatic/unit/test_monthly_means.py
tests/atmosphere/hydrostatic/unit/test_microphysics.py
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py
tests/atmosphere/hydrostatic/unit/test_amip_config.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py
tests/unit/test_runtime_bootstrap.py
tests/atmosphere/hydrostatic/unit/test_combined_physics.py
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py
tests/unit/test_physics_turbulence.py
tests/atmosphere/hydrostatic/unit/test_turbulence.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py
tests/atmosphere/hydrostatic/unit/test_topography.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_1.py
tests/unit/test_dcmip_transport.py
tests/unit/test_tracer_transport.py
tests/unit/test_diff_land.py
tests/unit/test_symmetry_invariance.py
tests/unit/test_ml_physics_parameterization.py
tests/unit/test_regridding_cubedsphere.py
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/__init__.py
tests/land/validation/test_bulk_flux_all_tiles.py
tests/land/validation/__init__.py
tests/unit/test_scale_runtime.py
tests/unit/test_physics_state_migration.py
tests/unit/test_field.py
tests/unit/test_land_ice_sea_ice_transport.py
tests/unit/test_step_cache.py
tests/unit/test_convection_triggers.py
tests/unit/test_issue_fixes.py
tests/unit/test_coupler.py
tests/atmosphere/nonhydrostatic/unit/__init__.py
tests/land/validation/test_bulk_flux_differentiability.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_2_mpas.py
tests/atmosphere/nonhydrostatic/integration/__init__.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/common.py
tests/unit/test_config_validation.py
tests/unit/test_timestepping.py
tests/unit/test_scale_tpu_compat.py
tests/unit/test_production_blockers.py
tests/unit/test_halo.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_2.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_3_mpas.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_1_mpas.py
tests/unit/test_diff_data_assimilation.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_3.py
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py
tests/unit/test_parallel.py
tests/unit/test_coupled_esm.py
tests/unit/test_batch_allreduce.py
tests/unit/test_fv3_audit_harness.py
tests/unit/test_cmor_output_dtype.py
tests/unit/test_ml_physics_workflow.py
tests/unit/test_regional_voronoi.py
tests/unit/test_physics_surface_models.py
tests/unit/test_surface_exchange.py
tests/unit/test_weatherbench.py
tests/unit/test_land_ice_sea_ice_thermo.py
tests/unit/test_spectral_pe_training_tracers.py
tests/unit/test_equation_fixes.py
tests/unit/test_grid_protocol.py
tests/unit/test_state_checkpoint.py
tests/unit/test_installed_package_imports.py
tests/unit/test_physics_smoke.py
tests/unit/test_cmor_experiments_restart.py
tests/unit/test_convection_plume.py
tests/unit/test_physics_convection.py
tests/unit/test_williamson2_cdgrid.py
tests/unit/test_precision_dtype_contracts.py
tests/unit/test_land_ice_stomata.py
tests/unit/test_bechtold.py
tests/unit/test_sfno.py
tests/unit/test_land_ice_slab_land.py
tests/unit/test_hybrid_vertical.py
tests/unit/test_ensemble_diagnostics.py
tests/unit/test_land_ice_units.py
tests/unit/test_earth_system_driver.py
tests/unit/test_physics_grid_adapters.py
tests/unit/test_land_ice_sea_ice_dynamics.py
tests/unit/test_land_ice_soil_hydraulics.py
tests/unit/test_physical_balances.py
tests/unit/test_component_factory.py
tests/unit/test_rce_script.py
tests/unit/test_ensemble.py
tests/unit/test_backend_precision.py
tests/unit/test_land_ice_soil_grid.py
tests/unit/test_corrections.py
tests/unit/test_duogrid_metrics.py
tests/unit/test_scale_device_mesh.py
tests/unit/test_version_guards.py
tests/unit/test_plot_amip.py
tests/unit/test_physics_ocean.py
tests/unit/test_diff_sea_ice.py
tests/unit/test_greens_function.py
tests/unit/test_operators_fv.py
tests/unit/test_diff_atmosphere_physics.py
tests/unit/test_fc_gram.py
tests/unit/test_physics_gwd.py
tests/unit/test_land_ice_integrated.py
tests/unit/test_land_ice_snow.py
tests/unit/test_distributed_checkpoint.py
tests/unit/test_warm_rain.py
tests/unit/test_diff_atmosphere_dynamics.py
tests/unit/test_mpas_atmosphere.py
tests/unit/test_land_ice_albedo.py
tests/unit/test_hardware_runtime_config.py
tests/unit/test_tiedtke.py
tests/unit/test_compiled_segments.py
tests/unit/test_scale_global_reductions.py

exec
/bin/zsh -lc "rg -n \"class .*Config|L_v|L_s|L_f|c_pd|T_freeze|R_v|rho_water\" src/legoesm/constants.py src/legoesm -g'*.py' | head -n 200" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/constants.py:19:c_pd = 1004.64                  # Specific heat at constant pressure [J/(kg*K)]
src/legoesm/constants.py:21:kappa = R_d / c_pd              # Poisson constant R_d/c_pd (~0.2857)
src/legoesm/constants.py:27:R_v = 461.51                    # Gas constant for water vapor [J/(kg*K)]
src/legoesm/constants.py:31:L_v = 2.501e6                   # Latent heat of vaporization at 0C [J/kg]
src/legoesm/constants.py:32:L_s = 2.834e6                   # Latent heat of sublimation at 0C [J/kg]
src/legoesm/constants.py:33:L_f = 3.337e5                   # Latent heat of fusion at 0C [J/kg]
src/legoesm/constants.py:34:rho_water = 1000.0              # Density of liquid water [kg/m^3]
src/legoesm/constants.py:36:T_freeze = 273.15               # Freezing point of water [K]
src/legoesm/constants.py:37:T_freeze_ocean = 271.35         # Freezing point of seawater [K] (~-1.8 C)
src/legoesm/constants.py:42:epsilon = R_d / R_v              # Molecular weight ratio (~0.622)
src/legoesm/surface_albedo.py:37:class LandAlbedoConfig(NamedTuple):
src/legoesm/surface_albedo.py:66:class IceAlbedoConfig(NamedTuple):
src/legoesm/surface_albedo.py:79:        Temperature range [K] over which transition occurs below T_freeze.
src/legoesm/surface_albedo.py:80:    T_freeze : float
src/legoesm/surface_albedo.py:86:    T_freeze: float = 271.35
src/legoesm/surface_albedo.py:89:class OceanAlbedoConfig(NamedTuple):
src/legoesm/surface_albedo.py:236:    When T_ice < T_freeze - T_transition_width: alpha = alpha_ice_cold.
src/legoesm/surface_albedo.py:237:    When T_ice >= T_freeze: alpha = alpha_ice_warm.
src/legoesm/surface_albedo.py:250:    T_cold = config.T_freeze - config.T_transition_width
src/legoesm/diagnostics/precision_drift.py:108:    c_p = constants.c_pd
src/legoesm/diagnostics/precision_drift.py:392:        c_p = constants.c_pd
src/legoesm/training/dycore_rollout.py:28:class RolloutConfig(NamedTuple):
src/legoesm/thermo.py:35:    T_c = T - constants.T_freeze
src/legoesm/thermo.py:78:    e_sat_i = 611.2 * exp(L_s/R_v * (1/T_freeze - 1/T))
src/legoesm/thermo.py:94:        constants.L_s / constants.R_v * (1.0 / constants.T_freeze - 1.0 / T)
src/legoesm/training/losses.py:18:class LossConfig(NamedTuple):
src/legoesm/diagnostics/energy_budget.py:46:    E = ∫ (c_p·T + L_v·q + g·z + ½(u²+v²)) dp/g
src/legoesm/diagnostics/energy_budget.py:49:    E = Σ_k (c_p·T_k + L_v·q_k + Φ_k + ½(u_k²+v_k²)) · p_s · dσ_k / g
src/legoesm/diagnostics/energy_budget.py:76:    c_p = constants.c_pd
src/legoesm/diagnostics/energy_budget.py:77:    L_v = constants.L_v
src/legoesm/diagnostics/energy_budget.py:130:    # Energy integrand per level: (c_p * T + L_v * q + Phi + KE) * dp / g
src/legoesm/diagnostics/energy_budget.py:132:    integrand = (c_p * T + L_v * q_v + Phi + KE) * dp / g
src/legoesm/diagnostics/energy_budget.py:151:    c_p = constants.c_pd
src/legoesm/atmosphere/physics/ml_parameterization.py:106:    sflx_T = shflx / constants.c_pd
src/legoesm/atmosphere/physics/ml_parameterization.py:107:    sflx_q = lhflx / constants.L_v
src/legoesm/atmosphere/physics/ml_parameterization.py:255:        dT_dt=-constants.L_v * dq_v_dt / constants.c_pd,
src/legoesm/atmosphere/physics/ml_parameterization.py:338:        dT_dt=constants.L_v * (rates.condensation - evaporation) / constants.c_pd,
src/legoesm/runtime/config.py:26:class RuntimeConfig(NamedTuple):
src/legoesm/ice/itd.py:192:    T_freeze_ocean: float = 271.35,
src/legoesm/ice/itd.py:219:    T_freeze_ocean : float
src/legoesm/ice/itd.py:230:        clamped to [T_ice_min, T_freeze_ocean]).
src/legoesm/ice/itd.py:322:    T_remap = jnp.clip(T_remap, T_ice_min, T_freeze_ocean)
src/legoesm/da/incremental.py:28:class IncrementalConfig(NamedTuple):
src/legoesm/ml/training.py:28:class TrainingConfig(NamedTuple):
src/legoesm/atmosphere/physics/combined.py:74:class PhysicsConfig(NamedTuple):
src/legoesm/ice/sea_ice.py:18:    Conductive flux through ice: F_cond = k_ice * (T_freeze - T_ice) / (h + h_min)
src/legoesm/ice/sea_ice.py:19:    Growth/melt: dh/dt = (F_cond - F_ocean) / (rho_ice * L_f)
src/legoesm/ice/sea_ice.py:143:            L_latent=constants.L_s,  # sublimation over ice, not evaporation
src/legoesm/ice/sea_ice.py:151:            L_latent=constants.L_s,  # sublimation over ice
src/legoesm/ice/sea_ice.py:400:            L_latent=constants.L_s,  # sublimation over ice
src/legoesm/ice/sea_ice.py:420:        config.k_ice * (config.T_freeze_ocean - T_ice) / h_eff,
src/legoesm/ice/sea_ice.py:431:        jnp.clip(T_trial, config.T_ice_min, config.T_freeze_ocean),
src/legoesm/ice/sea_ice.py:432:        jnp.full(T_ice.shape, config.T_freeze_ocean, dtype=T_ice.dtype),
src/legoesm/ice/sea_ice.py:438:        T_trial - config.T_freeze_ocean, 0.0
src/legoesm/ice/sea_ice.py:440:    dh_dt_surface_melt = -excess_energy / (config.rho_ice * config.L_f)
src/legoesm/ice/sea_ice.py:444:        ocean_sst - config.T_freeze_ocean, 0.0,
src/legoesm/ice/sea_ice.py:446:    dh_dt_basal = (F_cond - F_ocean) / (config.rho_ice * config.L_f)
src/legoesm/ice/sea_ice.py:452:    dh_dt_open = freeze_flux_open / (config.rho_ice * config.L_f)
src/legoesm/ice/sea_ice.py:512:            L_latent=constants.L_s,  # sublimation over ice
src/legoesm/ice/sea_ice.py:520:            L_latent=constants.L_s,  # sublimation over ice
src/legoesm/da/cycling.py:22:class CyclingConfig(NamedTuple):
src/legoesm/ml/physics/train.py:26:class PhysicsModelConfig(NamedTuple):
src/legoesm/ml/physics/train.py:34:class PhysicsTrainingConfig(NamedTuple):
src/legoesm/timestepping/split_explicit.py:37:class SplitExplicitConfig(NamedTuple):
src/legoesm/ice/transport.py:36:    T_freeze_ocean: float = 271.35,
src/legoesm/ice/transport.py:59:    T_ice_min, T_freeze_ocean : float
src/legoesm/ice/transport.py:120:        T_freeze_ocean,
src/legoesm/ice/transport.py:124:    T_new = jnp.clip(T_new, T_ice_min, T_freeze_ocean)
src/legoesm/ml/s2s/sfno_slab/preparation.py:21:class ArcoSurfaceForcingConfig:
src/legoesm/ml/s2s/sfno_slab/preparation.py:36:class ArcoSSTCacheConfig:
src/legoesm/ml/s2s/sfno_slab/preparation.py:202:        return values - np.float32(constants.T_freeze)
src/legoesm/training/era5_to_state.py:91:class TrainingERA5Config(NamedTuple):
src/legoesm/ice/config.py:10:class SeaIceConfig(NamedTuple):
src/legoesm/ice/config.py:25:    L_f: float = 3.337e5            # Latent heat of fusion [J/kg] (= constants.L_f)
src/legoesm/ice/config.py:38:    T_freeze_ocean: float = 271.35  # Ocean freezing point [K] (= constants.T_freeze_ocean)
src/legoesm/training/neural_gcm_spectral.py:77:class NeuralGCMSpectralConfig(NamedTuple):
src/legoesm/ml/s2s/sfno_slab/training.py:23:class S2SStochasticConfig(NamedTuple):
src/legoesm/ml/s2s/sfno_slab/training.py:34:class S2STrainingConfig(NamedTuple):
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:119:    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
src/legoesm/da/generate_nmc.py:81:class NMCConfig(NamedTuple):
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:82:    # Frictional heating: dT/dt = -(u*du/dt + v*dv/dt) / c_pd
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:83:    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
src/legoesm/grids/voronoi.py:1333:    L_for_shift = None
src/legoesm/grids/voronoi.py:1337:            L_for_shift = _L
src/legoesm/grids/voronoi.py:1343:        if L_for_shift is None:
src/legoesm/grids/voronoi.py:1347:            p1 = _shift_near(p0, cell_xyz[i1], L_for_shift)
src/legoesm/grids/voronoi.py:1348:            p2 = _shift_near(p0, cell_xyz[i2], L_for_shift)
src/legoesm/coupler/lake/two_layer_lake.py:77:    F_mix = config.rho_water * config.c_water * k_eff * (T_epi - T_hypo) / d_mid
src/legoesm/coupler/lake/two_layer_lake.py:80:    cap_epi = config.rho_water * config.c_water * config.h_epi
src/legoesm/coupler/lake/two_layer_lake.py:88:    T_freeze = config.T_freeze
src/legoesm/coupler/lake/two_layer_lake.py:89:    T_epi_new = jnp.maximum(T_trial_epi, T_freeze)
src/legoesm/coupler/lake/two_layer_lake.py:90:    Q_freeze_epi = cap_epi * jnp.maximum(T_freeze - T_trial_epi, 0.0) / dt
src/legoesm/coupler/lake/two_layer_lake.py:93:    cap_hypo = config.rho_water * config.c_water * config.h_hypo
src/legoesm/coupler/lake/two_layer_lake.py:96:    T_hypo_new = jnp.maximum(T_trial_hypo, T_freeze)
src/legoesm/coupler/lake/two_layer_lake.py:97:    Q_freeze_hypo = cap_hypo * jnp.maximum(T_freeze - T_trial_hypo, 0.0) / dt
src/legoesm/constants.py:19:c_pd = 1004.64                  # Specific heat at constant pressure [J/(kg*K)]
src/legoesm/constants.py:21:kappa = R_d / c_pd              # Poisson constant R_d/c_pd (~0.2857)
src/legoesm/constants.py:27:R_v = 461.51                    # Gas constant for water vapor [J/(kg*K)]
src/legoesm/constants.py:31:L_v = 2.501e6                   # Latent heat of vaporization at 0C [J/kg]
src/legoesm/constants.py:32:L_s = 2.834e6                   # Latent heat of sublimation at 0C [J/kg]
src/legoesm/constants.py:33:L_f = 3.337e5                   # Latent heat of fusion at 0C [J/kg]
src/legoesm/constants.py:34:rho_water = 1000.0              # Density of liquid water [kg/m^3]
src/legoesm/constants.py:36:T_freeze = 273.15               # Freezing point of water [K]
src/legoesm/constants.py:37:T_freeze_ocean = 271.35         # Freezing point of seawater [K] (~-1.8 C)
src/legoesm/constants.py:42:epsilon = R_d / R_v              # Molecular weight ratio (~0.622)
src/legoesm/ml/sfno.py:32:class SFNOConfig(NamedTuple):
src/legoesm/grids/topography.py:195:class TopographyConfig(NamedTuple):
src/legoesm/driver/model_driver.py:2033:            new_T = new_T + constants.L_v * excess / constants.c_pd
src/legoesm/driver/model_driver.py:2136:                new_T = new_T + constants.L_v * excess / constants.c_pd
src/legoesm/coupler/config.py:12:class TileConfig(NamedTuple):
src/legoesm/coupler/config.py:23:class CouplerConfig(NamedTuple):
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:120:    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
src/legoesm/coupler/lake/config.py:8:class LakeConfig(NamedTuple):
src/legoesm/coupler/lake/config.py:12:    rho_water: float = 1000.0       # Water density [kg/m3] (= constants.rho_water)
src/legoesm/coupler/lake/config.py:21:    T_freeze: float = 273.15        # Freezing point [K] (= constants.T_freeze)
src/legoesm/ml/s2s/sfno_slab/coupling.py:32:class S2SSlabCouplingConfig:
src/legoesm/ml/s2s/sfno_slab/coupling.py:104:    return values + np.float32(constants.T_freeze) if uses_celsius else values
src/legoesm/ml/s2s/sfno_slab/coupling.py:109:    return values - np.float32(constants.T_freeze) if uses_celsius else values
src/legoesm/ml/s2s/sfno_slab/coupling.py:261:        freeze_temperature=config.ocean.T_freeze,
src/legoesm/ml/s2s/sfno_slab/coupling.py:270:        freeze_temperature=config.ocean.T_freeze,
src/legoesm/ml/s2s/sfno_slab/coupling.py:333:                np.float32(config.ocean.T_freeze),
src/legoesm/ml/s2s/sfno_slab/coupling.py:339:            freeze_temperature=config.ocean.T_freeze,
src/legoesm/atmosphere/physics/clouds/config.py:12:class CloudConfig(NamedTuple):
src/legoesm/atmosphere/physics/clouds/config.py:37:    T_freeze : float
src/legoesm/atmosphere/physics/clouds/config.py:50:    T_freeze: float = 273.15  # = constants.T_freeze
src/legoesm/grids/vertical.py:1206:    c_p = constants.c_pd
src/legoesm/driver/physics_pipeline.py:387:        shflx = rho_low * constants.c_pd * _C_H * wind_speed * (T_sfc - T[..., -1])
src/legoesm/driver/physics_pipeline.py:389:        lhflx = rho_low * constants.L_v * _C_E * wind_speed * (q_sat_sfc - q_v[..., -1])
src/legoesm/driver/physics_pipeline.py:390:        evap_rate = lhflx / constants.L_v
src/legoesm/driver/physics_pipeline.py:392:        dT_BL = constants.g * shflx / (constants.c_pd * dp_low)
src/legoesm/atmosphere/physics/radiation/gray.py:301:    return (constants.g / constants.c_pd) * dF / dp
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:164:    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
src/legoesm/atmosphere/physics/clouds/cloud_fraction.py:64:    Linear ramp from 0 (all liquid) at T_freeze to 1 (all ice) at T_ice_only.
src/legoesm/atmosphere/physics/clouds/cloud_fraction.py:66:    frac = (config.T_freeze - T) / jnp.maximum(
src/legoesm/atmosphere/physics/clouds/cloud_fraction.py:67:        config.T_freeze - config.T_ice_only, 1.0
src/legoesm/ml/s2s/sfno_slab/config.py:29:class ChaosBenchS2SConfig(NamedTuple):
src/legoesm/driver/earth_system_driver.py:150:        snow_frac = jnp.where(T_low < constants.T_freeze, 1.0, 0.0)
src/legoesm/coupler/bulk_flux.py:282:    _L = constants.L_v if L_latent is None else L_latent
src/legoesm/coupler/bulk_flux.py:285:    shflx = rho * constants.c_pd * u_star * theta_star
src/legoesm/coupler/bulk_flux.py:336:    _L = constants.L_v if L_latent is None else L_latent
src/legoesm/coupler/bulk_flux.py:339:    shflx = rho * constants.c_pd * Ch * wind_speed * (T_sfc - T_lowest)
src/legoesm/atmosphere/physics/radiation/config.py:23:class GrayRadiationConfig(NamedTuple):
src/legoesm/atmosphere/physics/radiation/config.py:77:class RRTMGPConfig(NamedTuple):
src/legoesm/atmosphere/physics/radiation/config.py:137:class OzoneProfileConfig(NamedTuple):
src/legoesm/atmosphere/physics/radiation/config.py:172:class RadiationConfig(NamedTuple):
src/legoesm/driver/compiled_segments.py:626:                T_upd = T_upd + constants.L_v * excess / constants.c_pd
src/legoesm/driver/compiled_segments.py:652:            # (e.g., constants.L_v, constants.c_pd) breaking jax.lax.scan.
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:117:    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
src/legoesm/driver/coupled_config.py:17:class CoupledConfig(NamedTuple):
src/legoesm/atmosphere/physics/convection/tiedtke.py:283:        dT_dt_dd = -(constants.L_v / constants.c_pd) * (
src/legoesm/atmosphere/physics/turbulence/tke.py:176:    sflx_T = shflx / constants.c_pd
src/legoesm/atmosphere/physics/turbulence/tke.py:177:    sflx_q = lhflx / constants.L_v
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:30:class RayleighConfig(NamedTuple):
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:50:class LindzenConfig(NamedTuple):
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:73:class McFarlaneConfig(NamedTuple):
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:108:class HinesConfig(NamedTuple):
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:135:class PrognosticSpectralConfig(NamedTuple):
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:167:class MLEmulatorConfig(NamedTuple):
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:196:class GravityWaveDragConfig(NamedTuple):
src/legoesm/atmosphere/physics/convection/sbm.py:103:            cloud_mask * (constants.c_pd * (T_trial - T)
src/legoesm/atmosphere/physics/convection/sbm.py:104:                          + constants.L_v * (q_trial - q_v)) * dp,
src/legoesm/atmosphere/physics/convection/sbm.py:108:        dqsat_dT = constants.L_v * q_sat_trial / (constants.R_v * T_trial ** 2)
src/legoesm/atmosphere/physics/convection/sbm.py:110:            cloud_mask * (constants.c_pd
src/legoesm/atmosphere/physics/convection/sbm.py:111:                          + constants.L_v * RH_ref[:, None] * dqsat_dT) * dp,
src/legoesm/driver/config.py:20:class GridConfig(NamedTuple):
src/legoesm/driver/config.py:30:class DycoreConfig(NamedTuple):
src/legoesm/driver/config.py:41:class OutputConfig(NamedTuple):
src/legoesm/driver/config.py:54:class ExperimentConfig(NamedTuple):
src/legoesm/coupler/mpas_adapter.py:141:    L_v: float = constants.L_v,
src/legoesm/coupler/mpas_adapter.py:161:    L_v : float
src/legoesm/coupler/mpas_adapter.py:187:        L_v=L_v,
src/legoesm/atmosphere/physics/turbulence/surface_layer.py:10:    SH = rho * c_pd * Ch * |V| * (T_sfc - T)
src/legoesm/atmosphere/physics/turbulence/surface_layer.py:11:    LH = rho * L_v * Ch * |V| * (q_sfc - q_v)
src/legoesm/atmosphere/physics/turbulence/surface_layer.py:97:    shflx = rho * constants.c_pd * Ch * wind_speed * (T_sfc - T)
src/legoesm/atmosphere/physics/turbulence/surface_layer.py:100:    lhflx = rho * constants.L_v * Ch * wind_speed * (q_sfc - q_v)
src/legoesm/driver/coupled_esm_driver.py:359:        snow_frac = jnp.where(T_low < constants.T_freeze, 1.0, 0.0)
src/legoesm/ml/s2s/neuralgcm_slab/preparation.py:27:class PreparationConfig:
src/legoesm/ml/s2s/neuralgcm_slab/ensemble.py:48:class PerturbationConfig:
src/legoesm/atmosphere/physics/turbulence/ysu.py:154:    wtheta_sfc = shflx / (rho[:, -1] * constants.c_pd)  # kinematic (ncol,)
src/legoesm/atmosphere/physics/turbulence/ysu.py:193:    sflx_T = shflx / constants.c_pd
src/legoesm/atmosphere/physics/turbulence/ysu.py:194:    sflx_q = lhflx / constants.L_v
src/legoesm/atmosphere/physics/microphysics/sundqvist.py:147:    dT_dt = constants.L_v * net_cond / constants.c_pd
src/legoesm/atmosphere/physics/convection/bechtold.py:291:        dT_dt_dd = -(constants.L_v / constants.c_pd) * (
src/legoesm/atmosphere/physics/convection/kuo.py:150:    # water by latent heat balance: c_pd * dT_dt = L_v * (-dq_v) for
src/legoesm/atmosphere/physics/convection/kuo.py:156:    implied_condensation = dT_dt * constants.c_pd / constants.L_v  # (ncol, nlev)
src/legoesm/atmosphere/physics/turbulence/holtslag_boville.py:176:    sflx_T = shflx / constants.c_pd
src/legoesm/atmosphere/physics/turbulence/holtslag_boville.py:177:    sflx_q = lhflx / constants.L_v
src/legoesm/atmosphere/physics/turbulence/holtslag_boville.py:183:    wtheta_sfc = shflx / (rho[:, -1] * constants.c_pd)  # kinematic heat flux (ncol,)
src/legoesm/forcing/amip.py:30:class AMIPForcingConfig(NamedTuple):
src/legoesm/forcing/amip.py:125:            sst_offset=constants.T_freeze,    # Celsius -> Kelvin
src/legoesm/atmosphere/physics/turbulence/clubb_lite.py:63:    e_sat = 611.2 * jnp.exp(17.67 * (T - constants.T_freeze) /
src/legoesm/atmosphere/physics/turbulence/clubb_lite.py:64:                             (T - constants.T_freeze + 243.5))
src/legoesm/atmosphere/physics/turbulence/clubb_lite.py:65:    de_dT = e_sat * 17.67 * 243.5 / (T - constants.T_freeze + 243.5) ** 2
src/legoesm/atmosphere/physics/turbulence/clubb_lite.py:175:    # Liquid water potential temperature (simplified: theta_l ~ theta - L_v*q_c/c_pd*Pi)
src/legoesm/atmosphere/physics/turbulence/clubb_lite.py:270:    sflx_T = shflx / constants.c_pd
src/legoesm/atmosphere/physics/turbulence/clubb_lite.py:271:    sflx_q = lhflx / constants.L_v
src/legoesm/atmosphere/physics/microphysics/thompson.py:106:    T_freeze = constants.T_freeze
src/legoesm/atmosphere/physics/microphysics/thompson.py:111:        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
src/legoesm/atmosphere/physics/microphysics/thompson.py:128:        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
src/legoesm/atmosphere/physics/microphysics/thompson.py:142:    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))

exec
/bin/zsh -lc 'nl -ba tests/unit/test_physics_microphysics.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'nl -ba tests/unit/test_warm_rain.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Smoke tests for warm-rain microphysics helpers (`_warm_rain.py`).
     2	
     3	Run with:
     4	
     5	    JAX_ENABLE_X64=1 .venv/bin/python3.14 -m pytest \
     6	        tests/unit/test_warm_rain.py -v
     7	"""
     8	
     9	from __future__ import annotations
    10	
    11	import jax
    12	import jax.numpy as jnp
    13	import pytest
    14	
    15	from legoesm import constants
    16	from legoesm.atmosphere.physics.microphysics._warm_rain import (
    17	    saturation_adjustment,
    18	    effective_Nc,
    19	    autoconversion_sb,
    20	    accretion,
    21	    self_collection_breakup,
    22	    rain_evaporation,
    23	)
    24	
    25	
    26	@pytest.fixture
    27	def column_state():
    28	    """Build a small (ncol, nlev) column with realistic warm-cloud values."""
    29	    ncol, nlev = 4, 6
    30	    T = jnp.full((ncol, nlev), 285.0)            # mid-troposphere temperature [K]
    31	    p = jnp.full((ncol, nlev), 8.0e4)            # ~800 hPa
    32	    rho = p / (constants.R_d * T)
    33	    q_v = jnp.full((ncol, nlev), 1.5e-2)         # high vapor (likely supersat)
    34	    q_c = jnp.full((ncol, nlev), 5.0e-4)         # cloud water
    35	    q_r = jnp.full((ncol, nlev), 2.0e-4)         # rain
    36	    N_c = 1.0e8 * jnp.ones((ncol, nlev))         # cloud droplet number [1/kg]
    37	    N_r = 1.0e3 * jnp.ones((ncol, nlev))         # rain number [1/kg]
    38	    return T, p, rho, q_v, q_c, q_r, N_c, N_r
    39	
    40	
    41	def test_saturation_adjustment(column_state):
    42	    T, p, _, q_v, _, _, _, _ = column_state
    43	    cond, q_sat = saturation_adjustment(T, q_v, p, dt=300.0)
    44	    assert cond.shape == T.shape
    45	    assert q_sat.shape == T.shape
    46	    assert jnp.all(jnp.isfinite(cond))
    47	    assert jnp.all(jnp.isfinite(q_sat))
    48	    assert jnp.all(q_sat > 0.0)
    49	    # Supersaturated air → positive condensation
    50	    assert float(jnp.mean(cond)) > 0.0
    51	
    52	
    53	def test_effective_Nc(column_state):
    54	    *_, N_c, _ = column_state
    55	    Nc_eff = effective_Nc(N_c, Nc_0=5.0e7)
    56	    assert Nc_eff.shape == N_c.shape
    57	    assert jnp.all(jnp.isfinite(Nc_eff))
    58	    assert jnp.all(Nc_eff > 1.0)
    59	    # Where N_c is set (>1) the effective value passes through.
    60	    assert jnp.allclose(Nc_eff, N_c)
    61	    # Where N_c is zero, fallback kicks in.
    62	    fallback = effective_Nc(jnp.zeros_like(N_c), Nc_0=5.0e7)
    63	    assert jnp.allclose(fallback, 5.0e7)
    64	
    65	
    66	def test_autoconversion_sb(column_state):
    67	    _, _, rho, _, q_c, _, N_c, _ = column_state
    68	    dq_au, dN_au, x_c = autoconversion_sb(
    69	        q_c, N_c, rho, k_au=9.44e9, x_star=2.6e-10,
    70	    )
    71	    assert dq_au.shape == q_c.shape
    72	    assert dN_au.shape == q_c.shape
    73	    assert x_c.shape == q_c.shape
    74	    assert jnp.all(jnp.isfinite(dq_au))
    75	    assert jnp.all(jnp.isfinite(dN_au))
    76	    assert jnp.all(jnp.isfinite(x_c))
    77	    assert jnp.all(dq_au >= 0.0)
    78	    assert jnp.all(x_c > 0.0)
    79	
    80	
    81	def test_accretion(column_state):
    82	    _, _, rho, _, q_c, q_r, _, _ = column_state
    83	    rate = accretion(q_c, q_r, rho, k_ac=5.25)
    84	    assert rate.shape == q_c.shape
    85	    assert jnp.all(jnp.isfinite(rate))
    86	    assert jnp.all(rate >= 0.0)
    87	
    88	
    89	def test_self_collection_breakup(column_state):
    90	    _, _, rho, _, _, q_r, _, N_r = column_state
    91	    dN_sc, dN_br = self_collection_breakup(
    92	        N_r, q_r, rho, k_sc=7.12, breakup_sharpness=10.0, D_eq=9.0e-4,
    93	    )
    94	    assert dN_sc.shape == N_r.shape
    95	    assert dN_br.shape == N_r.shape
    96	    assert jnp.all(jnp.isfinite(dN_sc))
    97	    assert jnp.all(jnp.isfinite(dN_br))
    98	    # Self-collection always reduces number; breakup either zero or positive.
    99	    assert jnp.all(dN_sc <= 0.0)
   100	    assert jnp.all(dN_br >= 0.0)
   101	
   102	
   103	def test_rain_evaporation(column_state):
   104	    T, p, _, _, _, q_r, _, _ = column_state
   105	    # Subsaturated vapor profile to drive evaporation.
   106	    q_v_low = 1.0e-3 * jnp.ones_like(q_r)
   107	    _, q_sat = saturation_adjustment(T, q_v_low, p, dt=300.0)
   108	    evap = rain_evaporation(q_v_low, q_r, q_sat, evap_coeff=1.0e-3)
   109	    assert evap.shape == q_r.shape
   110	    assert jnp.all(jnp.isfinite(evap))
   111	    assert jnp.all(evap >= 0.0)
   112	    # With substantial subsaturation, average evaporation is strictly positive.
   113	    assert float(jnp.mean(evap)) > 0.0
   114	
   115	
   116	def test_jit_compiles(column_state):
   117	    T, p, rho, q_v, q_c, q_r, N_c, N_r = column_state
   118	
   119	    @jax.jit
   120	    def chain(T_, q_v_, q_c_, q_r_, N_c_, N_r_, rho_, p_):
   121	        cond, q_sat = saturation_adjustment(T_, q_v_, p_, dt=300.0)
   122	        Nc_eff = effective_Nc(N_c_, Nc_0=5.0e7)
   123	        dq_au, dN_au, _ = autoconversion_sb(
   124	            q_c_, Nc_eff, rho_, k_au=9.44e9, x_star=2.6e-10,
   125	        )
   126	        acc = accretion(q_c_, q_r_, rho_, k_ac=5.25)
   127	        evap = rain_evaporation(q_v_, q_r_, q_sat, evap_coeff=1.0e-3)
   128	        return cond + dq_au + dN_au + acc + evap
   129	
   130	    out = chain(T, q_v, q_c, q_r, N_c, N_r, rho, p)
   131	    assert out.shape == T.shape
   132	    assert jnp.all(jnp.isfinite(out))

 succeeded in 0ms:
     1	"""Category 4: Microphysics -- Physical Consistency.
     2	
     3	Tests total water conservation, temperature-moisture coupling (Clausius-
     4	Clapeyron), saturation adjustment, precipitation positivity, ice-phase
     5	bounds, and autoconversion threshold for all microphysics schemes.
     6	"""
     7	
     8	from __future__ import annotations
     9	
    10	import jax
    11	import jax.numpy as jnp
    12	import pytest
    13	
    14	from legoesm import constants
    15	from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
    16	from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
    17	from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
    18	from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
    19	from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
    20	from legoesm.atmosphere.physics.microphysics.config import (
    21	    KesslerConfig, SundqvistConfig, SeifertBehengConfig, MorrisonConfig,
    22	    ThompsonConfig,
    23	)
    24	from legoesm.atmosphere.physics.microphysics.output import (
    25	    HydrometeorState, make_zero_hydrometeors,
    26	)
    27	from legoesm.thermo import saturation_mixing_ratio
    28	
    29	
    30	# ---------------------------------------------------------------------------
    31	# Helpers
    32	# ---------------------------------------------------------------------------
    33	
    34	def _make_column(nlev=20, ncol=4, T_sfc=280.0, q_c_val=1e-4, supersaturated=False):
    35	    """Build a microphysics column with realistic conditions."""
    36	    p_s = 1.0e5
    37	    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    38	    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    39	    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    40	    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))
    41	
    42	    T = T_sfc * jnp.clip(sigma_full, 0.01, None) ** 0.19
    43	    T = jnp.maximum(T, 200.0)
    44	    T = jnp.broadcast_to(T[None, :], (ncol, nlev))
    45	
    46	    rho = p_full / (constants.R_d * jnp.clip(T, 1.0, None))
    47	
    48	    dp = p_half[:, 1:] - p_half[:, :-1]
    49	    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    50	    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    51	    dz = jnp.abs(dz)
    52	
    53	    q_sat = saturation_mixing_ratio(T, p_full)
    54	    if supersaturated:
    55	        q_v = 1.2 * q_sat
    56	    else:
    57	        q_v = 0.8 * q_sat
    58	
    59	    q_c = jnp.zeros((ncol, nlev))
    60	    q_c = q_c.at[..., -5:].set(q_c_val)
    61	
    62	    hydro = HydrometeorState(
    63	        q_c=q_c,
    64	        q_r=jnp.zeros((ncol, nlev)),
    65	        q_i=jnp.zeros((ncol, nlev)),
    66	        q_s=jnp.zeros((ncol, nlev)),
    67	        q_g=jnp.zeros((ncol, nlev)),
    68	        N_c=1e8 * jnp.ones((ncol, nlev)),
    69	        N_r=jnp.zeros((ncol, nlev)),
    70	        N_i=jnp.zeros((ncol, nlev)),
    71	    )
    72	    return T, q_v, hydro, p_full, p_half, rho, dz
    73	
    74	
    75	def _call_scheme(name, T, q_v, hydro, p_full, p_half, rho, dz, dt=300.0):
    76	    """Call a microphysics scheme backend and return MicrophysicsOutput."""
    77	    if name == "kessler":
    78	        return kessler_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
    79	                                     config=KesslerConfig())
    80	    elif name == "sundqvist":
    81	        return sundqvist_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
    82	                                       config=SundqvistConfig())
    83	    elif name == "seifert_beheng":
    84	        return seifert_beheng_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
    85	                                            config=SeifertBehengConfig())
    86	    elif name == "morrison":
    87	        return morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
    88	                                      config=MorrisonConfig())
    89	    elif name == "thompson":
    90	        return thompson_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
    91	                                      config=ThompsonConfig())
    92	    else:
    93	        raise ValueError(f"Unknown scheme: {name}")
    94	
    95	
    96	ALL_SCHEMES = ["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"]
    97	
    98	
    99	# ============================================================================
   100	# 4a  All outputs finite
   101	# ============================================================================
   102	
   103	@pytest.mark.parametrize("scheme", ALL_SCHEMES)
   104	def test_all_outputs_finite(scheme):
   105	    """All MicrophysicsOutput fields should be finite."""
   106	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
   107	    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
   108	    for fname in out._fields:
   109	        val = getattr(out, fname)
   110	        assert jnp.all(jnp.isfinite(val)), f"{scheme}: {fname} has NaN/Inf"
   111	
   112	
   113	# ============================================================================
   114	# 4b  Temperature-moisture coupling (Clausius-Clapeyron)
   115	# ============================================================================
   116	
   117	@pytest.mark.parametrize("scheme", ALL_SCHEMES)
   118	def test_temperature_moisture_coupling(scheme):
   119	    """cp * dT_dt approx Lv * condensation_rate (first order).
   120	
   121	    Checks that where heating is significant, it correlates with moisture
   122	    removal (condensation heats, evaporation cools).
   123	    """
   124	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(supersaturated=True)
   125	    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
   126	
   127	    lhs = constants.c_pd * out.dT_dt
   128	    rhs = -constants.L_v * out.dq_v_dt
   129	    scale = jnp.maximum(jnp.abs(lhs), 1e-10)
   130	    rel_err = jnp.abs(lhs - rhs) / scale
   131	    active = jnp.abs(lhs) > 1e-8
   132	    if jnp.any(active):
   133	        median_err = float(jnp.median(rel_err[active]))
   134	        assert median_err < 0.5, (
   135	            f"{scheme}: median Clausius-Clapeyron rel_err = {median_err:.3f}"
   136	        )
   137	
   138	
   139	# ============================================================================
   140	# 4c  Saturation adjustment: supersaturated -> vapor decreases
   141	# ============================================================================
   142	
   143	@pytest.mark.parametrize("scheme", ["kessler", "sundqvist"])
   144	def test_saturation_adjustment_vapor_decreases(scheme):
   145	    """Starting from supersaturated, vapor should decrease (dq_v_dt < 0)."""
   146	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(
   147	        supersaturated=True, q_c_val=0.0
   148	    )
   149	    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
   150	
   151	    # Some levels should show condensation (dq_v_dt < 0)
   152	    min_dqv = float(jnp.min(out.dq_v_dt))
   153	    assert min_dqv < 0.0, (
   154	        f"{scheme}: no condensation in supersaturated column, min dq_v_dt = {min_dqv:.2e}"
   155	    )
   156	
   157	
   158	# ============================================================================
   159	# 4d  Precipitation non-negative
   160	# ============================================================================
   161	
   162	@pytest.mark.parametrize("scheme", ALL_SCHEMES)
   163	def test_precipitation_non_negative(scheme):
   164	    """Precipitation must be >= 0."""
   165	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
   166	    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
   167	    min_precip = float(jnp.min(out.precipitation))
   168	    assert min_precip >= -1e-15, (
   169	        f"{scheme}: negative precipitation = {min_precip:.2e}"
   170	    )
   171	
   172	
   173	# ============================================================================
   174	# 4e  Ice-phase temperature bounds (Morrison, Thompson)
   175	# ============================================================================
   176	
   177	@pytest.mark.parametrize("scheme", ["morrison", "thompson"])
   178	def test_ice_no_formation_above_freezing(scheme):
   179	    """Above freezing (T > 273.15 K), ice formation should be negligible."""
   180	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(T_sfc=290.0)
   181	    T_warm = jnp.maximum(T, 280.0)
   182	    out = _call_scheme(scheme, T_warm, q_v, hydro, p_full, p_half, rho, dz)
   183	
   184	    warm_mask = T_warm > constants.T_freeze
   185	    if jnp.any(warm_mask):
   186	        ice_formation = out.dq_i_dt[warm_mask]
   187	        max_ice_form = float(jnp.max(ice_formation))
   188	        # Allow small numerical noise from sigmoid tails
   189	        assert max_ice_form <= 1e-10, (
   190	            f"{scheme}: ice forms above freezing, max dq_i_dt = {max_ice_form:.2e}"
   191	        )
   192	
   193	
   194	# ============================================================================
   195	# 4g  Autoconversion threshold (Kessler)
   196	# ============================================================================
   197	
   198	def test_kessler_autoconversion_sensitivity():
   199	    """Rain production should increase when q_c exceeds autoconversion threshold."""
   200	    config = KesslerConfig()
   201	    # Below threshold: subsaturated, small q_c
   202	    T_lo, q_v_lo, hydro_lo, p_full, p_half, rho, dz = _make_column(
   203	        q_c_val=0.1 * config.autoconversion_threshold, supersaturated=False
   204	    )
   205	    out_lo = kessler_microphysics(T_lo, q_v_lo, hydro_lo, p_full, p_half, rho, dz,
   206	                                   300.0, config=config)
   207	    # Above threshold: subsaturated, large q_c
   208	    _, _, hydro_hi, _, _, _, _ = _make_column(
   209	        q_c_val=5.0 * config.autoconversion_threshold, supersaturated=False
   210	    )
   211	    out_hi = kessler_microphysics(T_lo, q_v_lo, hydro_hi, p_full, p_half, rho, dz,
   212	                                   300.0, config=config)
   213	
   214	    max_dqr_lo = float(jnp.max(out_lo.dq_r_dt))
   215	    max_dqr_hi = float(jnp.max(out_hi.dq_r_dt))
   216	    assert max_dqr_hi > max_dqr_lo, (
   217	        f"Kessler: rain production not higher above threshold: "
   218	        f"lo={max_dqr_lo:.2e}, hi={max_dqr_hi:.2e}"
   219	    )
   220	
   221	
   222	# ============================================================================
   223	# Heating rate magnitude bounds
   224	# ============================================================================
   225	
   226	@pytest.mark.parametrize("scheme", ALL_SCHEMES)
   227	def test_heating_rate_bounded(scheme):
   228	    """|dT_dt| should be bounded (< 10 K/s for strongly supersaturated)."""
   229	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(supersaturated=True)
   230	    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
   231	    max_hr = float(jnp.max(jnp.abs(out.dT_dt)))
   232	    assert max_hr < 10.0, (
   233	        f"{scheme}: |dT_dt| = {max_hr:.2e} K/s exceeds 10 K/s"
   234	    )

exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/microphysics/integration.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'nl -ba tests/atmosphere/hydrostatic/unit/test_microphysics.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Model integration bridge for microphysics.
     2	
     3	Provides `make_microphysics_physics()`, a factory that returns a physics
     4	function matching each dynamical core's `step_with_physics` signature.
     5	
     6	Supported model types:
     7	- "hydrostatic"  : PrimitiveEquationModel (sigma coordinates)
     8	- "nonhydrostatic": CompressibleEulerModel (z* coordinates)
     9	- "spectral_pe"  : SpectralPEModel (Gaussian grid + sigma coordinates)
    10	"""
    11	
    12	from __future__ import annotations
    13	
    14	from typing import Callable
    15	
    16	import jax
    17	import jax.numpy as jnp
    18	
    19	from legoesm.core.field import Field
    20	from legoesm.core.state import (
    21	    HydrostaticState,
    22	    HydrostaticTendencies,
    23	    NonHydrostaticState,
    24	    NonHydrostaticTendencies,
    25	)
    26	from legoesm.grids.cubed_sphere import CubedSphereGrid
    27	from legoesm.grids.vertical import (
    28	    HeightCoordinate,
    29	    SigmaCoordinate,
    30	    TerrainMetric,
    31	    pressure_from_sigma,
    32	)
    33	from legoesm import constants
    34	
    35	from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    36	from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    37	from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
    38	from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
    39	from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
    40	from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
    41	from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
    42	from legoesm.atmosphere.physics.microphysics.ml_emulator import (
    43	    ml_microphysics,
    44	    MicrophysicsEmulator,
    45	)
    46	from legoesm.atmosphere.physics.thermodynamics import (
    47	    pressure_from_eos,
    48	    reconstruct_half_level_pressure_hydrostatic,
    49	    sanitize_theta_rho,
    50	)
    51	
    52	
    53	def _get_microphysics_fn(config: MicrophysicsConfig):
    54	    """Select the microphysics backend based on config.scheme.
    55	
    56	    Returns
    57	    -------
    58	    scheme_name : str
    59	    micro_fn : callable or None
    60	    scheme_config : NamedTuple or None
    61	    """
    62	    if config.scheme == "kessler":
    63	        return "kessler", kessler_microphysics, config.kessler
    64	    elif config.scheme == "sundqvist":
    65	        return "sundqvist", sundqvist_microphysics, config.sundqvist
    66	    elif config.scheme == "seifert_beheng":
    67	        return "seifert_beheng", seifert_beheng_microphysics, config.seifert_beheng
    68	    elif config.scheme == "morrison":
    69	        return "morrison", morrison_microphysics, config.morrison
    70	    elif config.scheme == "thompson":
    71	        return "thompson", thompson_microphysics, config.thompson
    72	    elif config.scheme == "ml_emulator":
    73	        return "ml_emulator", ml_microphysics, config.ml_emulator
    74	    elif config.scheme == "none":
    75	        return "none", None, None
    76	    else:
    77	        raise ValueError(f"Unknown microphysics scheme: {config.scheme!r}")
    78	
    79	
    80	from legoesm.atmosphere.physics._shared import (
    81	    compute_layer_dz as _compute_heights_from_sigma,
    82	    compute_rho as _compute_rho,
    83	)
    84	
    85	
    86	def make_microphysics_physics(
    87	    microphysics_config: MicrophysicsConfig,
    88	    model_type: str = "hydrostatic",
    89	    dt: float = 300.0,
    90	) -> Callable:
    91	    """Create a physics function for microphysics matching a model's signature.
    92	
    93	    Parameters
    94	    ----------
    95	    microphysics_config : MicrophysicsConfig
    96	        Microphysics configuration (selects scheme).
    97	    model_type : str
    98	        One of "hydrostatic", "nonhydrostatic", "spectral_pe".
    99	    dt : float
   100	        Model time step [s].
   101	
   102	    Returns
   103	    -------
   104	    Callable
   105	        Physics function with the correct signature for the model.
   106	    """
   107	    # ``model_type="mpas"`` reuses the hydrostatic factory: the
   108	    # ``_make_hydrostatic_microphysics`` bridge reshapes
   109	    # ``(*shape_2d, nlev)`` to ``(ncol, nlev)`` and never references
   110	    # grid lat/lon — works identically for cubed-sphere ``(face, n, n)``,
   111	    # lat-lon ``(n_lat, n_lon)``, and MPAS Voronoi ``(nCells,)``.
   112	    if model_type in ("hydrostatic", "mpas"):
   113	        return _make_hydrostatic_microphysics(microphysics_config, dt)
   114	    elif model_type == "nonhydrostatic":
   115	        return _make_nonhydrostatic_microphysics(microphysics_config, dt)
   116	    elif model_type == "spectral_pe":
   117	        return _make_spectral_pe_microphysics(microphysics_config, dt)
   118	    else:
   119	        raise ValueError(
   120	            f"Unknown model_type: {model_type!r}. "
   121	            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe', 'mpas'."
   122	        )
   123	
   124	
   125	# ===========================================================================
   126	# Hydrostatic PE
   127	# ===========================================================================
   128	
   129	def _make_hydrostatic_microphysics(
   130	    microphysics_config: MicrophysicsConfig,
   131	    dt: float,
   132	) -> Callable:
   133	    """Create microphysics physics_fn for PrimitiveEquationModel.
   134	
   135	    Signature: (state, grid, sigma_coord) -> HydrostaticTendencies
   136	    """
   137	    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(microphysics_config)
   138	    is_ml = scheme_name == "ml_emulator"
   139	    _ml_model_cache = [None]
   140	
   141	    def physics_fn(
   142	        state: HydrostaticState,
   143	        grid,
   144	        sigma_coord: SigmaCoordinate,
   145	    ) -> HydrostaticTendencies:
   146	        T = state.T.data
   147	        p_s = state.p_s.data
   148	
   149	        nlev = sigma_coord.n_levels
   150	        shape_3d = T.shape
   151	        shape_2d = p_s.shape
   152	
   153	        # Derive Field metadata from the input state so the returned
   154	        # tendencies match the underlying grid: cubed-sphere uses
   155	        # ("face","x","y",...), lat-lon uses ("lat","lon",...), and
   156	        # MPAS uses ("nCells",...).
   157	        dims_3d = state.T.dims
   158	        dims_2d = state.p_s.dims
   159	        u_shape = state.u.data.shape
   160	        u_dims = state.u.dims
   161	        v_dims = state.v.dims if state.v is not None else None
   162	        v_shape = state.v.data.shape if state.v is not None else None
   163	
   164	        # Pin defaulted allocations to the state precision so we never
   165	        # silently flow x64 zeros into the column physics path.
   166	        _state_dtype = T.dtype
   167	
   168	        def _zero_dv_dt():
   169	            """``None`` for MPAS (no v), Field of zeros otherwise."""
   170	            if state.v is None:
   171	                return None
   172	            return Field(
   173	                data=jnp.zeros(v_shape, dtype=_state_dtype),
   174	                name="dv_dt_micro", dims=v_dims, units="m/s^2",
   175	            )
   176	
   177	        if micro_fn is None:
   178	            return HydrostaticTendencies(
   179	                du_dt=Field(
   180	                    data=jnp.zeros(u_shape, dtype=_state_dtype),
   181	                    name="du_dt_micro", dims=u_dims, units="m/s^2",
   182	                ),
   183	                dv_dt=_zero_dv_dt(),
   184	                dT_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dT_dt_micro", dims=dims_3d, units="K/s"),
   185	                dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dp_s_dt_micro", dims=dims_2d, units="Pa/s"),
   186	                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
   187	            )
   188	
   189	        # Pressure at full and half levels
   190	        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
   191	        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)
   192	
   193	        # Reshape to columns generically across cubed-sphere
   194	        # ``shape_2d=(6,n,n)``, lat-lon ``(n_lat,n_lon)``, and MPAS
   195	        # ``(nCells,)``.
   196	        ncol = 1
   197	        for s in shape_2d:
   198	            ncol *= int(s)
   199	        T_col = T.reshape(ncol, nlev)
   200	        p_full_col = p_full.reshape(ncol, nlev)
   201	        p_half_col = p_half.reshape(ncol, nlev + 1)
   202	        # Helper: extract a tracer from the tracer dict, returning a
   203	        # column-reshaped (ncol, nlev) array clipped to non-negative.
   204	        def _get_tracer(name):
   205	            if state.tracers is not None and name in state.tracers:
   206	                raw = state.tracers[name]
   207	                data = raw.data if hasattr(raw, "data") else raw
   208	                return jnp.maximum(data.reshape(ncol, nlev), 0.0)
   209	            return jnp.zeros((ncol, nlev), dtype=_state_dtype)
   210	
   211	        # Extract water vapor from tracers if available; else assume dry.
   212	        q_v_col = _get_tracer("q_v")
   213	
   214	        rho = _compute_rho(T_col, p_full_col)
   215	        dz = _compute_heights_from_sigma(T_col, p_half_col)
   216	
   217	        # Extract actual hydrometeor state from tracers (fall back to zero
   218	        # for any species not present in the tracer registry).
   219	        hydrometeors = HydrometeorState(
   220	            q_c=_get_tracer("q_c"),
   221	            q_r=_get_tracer("q_r"),
   222	            q_i=_get_tracer("q_i"),
   223	            q_s=_get_tracer("q_s"),
   224	            q_g=_get_tracer("q_g"),
   225	            N_c=_get_tracer("N_c"),
   226	            N_r=_get_tracer("N_r"),
   227	            N_i=_get_tracer("N_i"),
   228	        )
   229	
   230	        if is_ml:
   231	            if _ml_model_cache[0] is None:
   232	                key = jax.random.PRNGKey(scheme_config.seed)
   233	                _ml_model_cache[0] = MicrophysicsEmulator(
   234	                    scheme_config.n_input, scheme_config.n_hidden,
   235	                    scheme_config.n_layers, scheme_config.n_output, key=key,
   236	                )
   237	            micro_out = micro_fn(
   238	                T_col, q_v_col, hydrometeors,
   239	                p_full_col, p_half_col, rho, dz, dt,
   240	                scheme_config, _ml_model_cache[0],
   241	            )
   242	        else:
   243	            micro_out = micro_fn(
   244	                T_col, q_v_col, hydrometeors,
   245	                p_full_col, p_half_col, rho, dz, dt, scheme_config,
   246	            )
   247	
   248	        dT_dt = micro_out.dT_dt.reshape(shape_3d)
   249	
   250	        # Propagate tracer tendencies from microphysics backend
   251	        tracer_tends = {
   252	            "q_v": Field(data=micro_out.dq_v_dt.reshape(shape_3d),
   253	                         name="dq_v_dt_micro", dims=dims_3d, units="kg/kg/s"),
   254	            "q_c": Field(data=micro_out.dq_c_dt.reshape(shape_3d),
   255	                         name="dq_c_dt_micro", dims=dims_3d, units="kg/kg/s"),
   256	            "q_r": Field(data=micro_out.dq_r_dt.reshape(shape_3d),
   257	                         name="dq_r_dt_micro", dims=dims_3d, units="kg/kg/s"),
   258	            "q_i": Field(data=micro_out.dq_i_dt.reshape(shape_3d),
   259	                         name="dq_i_dt_micro", dims=dims_3d, units="kg/kg/s"),
   260	            "q_s": Field(data=micro_out.dq_s_dt.reshape(shape_3d),
   261	                         name="dq_s_dt_micro", dims=dims_3d, units="kg/kg/s"),
   262	            "q_g": Field(data=micro_out.dq_g_dt.reshape(shape_3d),
   263	                         name="dq_g_dt_micro", dims=dims_3d, units="kg/kg/s"),
   264	        }
   265	
   266	        return HydrostaticTendencies(
   267	            du_dt=Field(
   268	                data=jnp.zeros(u_shape, dtype=_state_dtype),
   269	                name="du_dt_micro", dims=u_dims, units="m/s^2",
   270	            ),
   271	            dv_dt=_zero_dv_dt(),
   272	            dT_dt=Field(data=dT_dt, name="dT_dt_micro", dims=dims_3d, units="K/s"),
   273	            dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dp_s_dt_micro", dims=dims_2d, units="Pa/s"),
   274	            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
   275	            tracer_tendencies=tracer_tends,
   276	        )
   277	
   278	    def reset_state():
   279	        _ml_model_cache[0] = None
   280	
   281	    physics_fn.reset_state = reset_state
   282	    return physics_fn
   283	
   284	
   285	# ===========================================================================
   286	# Non-hydrostatic Compressible Euler
   287	# ===========================================================================
   288	
   289	def _make_nonhydrostatic_microphysics(
   290	    microphysics_config: MicrophysicsConfig,
   291	    dt: float,
   292	) -> Callable:
   293	    """Create microphysics physics_fn for CompressibleEulerModel.
   294	
   295	    Signature: (state, grid, height_coord, terrain_metric) -> NonHydrostaticTendencies
   296	    """
   297	    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(microphysics_config)
   298	    is_ml = scheme_name == "ml_emulator"
   299	    _ml_model_cache = [None]
   300	
   301	    def physics_fn(
   302	        state: NonHydrostaticState,
   303	        grid: CubedSphereGrid,
   304	        height_coord: HeightCoordinate,
   305	        terrain_metric: TerrainMetric,
   306	    ) -> NonHydrostaticTendencies:
   307	        theta_p = state.theta_prime.data
   308	        rho_p = state.rho_prime.data
   309	        tracers = state.tracers.data
   310	
   311	        theta_0 = height_coord.theta_ref
   312	        rho_0 = height_coord.rho_ref
   313	
   314	        theta_total, rho_total = sanitize_theta_rho(
   315	            theta_0 + theta_p,
   316	            rho_0 + rho_p,
   317	        )
   318	
   319	        p = pressure_from_eos(rho_total, theta_total)
   320	        exner = (p / constants.p_ref) ** constants.kappa
   321	        T = theta_total * exner
   322	
   323	        nlev = height_coord.n_levels
   324	        shape_3d = theta_p.shape
   325	        shape_w = state.w.data.shape
   326	        shape_2d = state.phis.data.shape
   327	        n_tracers = tracers.shape[-1] if tracers.ndim >= 5 else 0
   328	
   329	        dims_3d = ("face", "x", "y", "level")
   330	        dims_w = ("face", "x", "y", "level_half")
   331	        dims_2d = ("face", "x", "y")
   332	        dims_tr = ("face", "x", "y", "level", "tracer")
   333	
   334	        # Pin defaulted allocations to the state precision so x64 zeros
   335	        # do not silently flow into the column physics path.
   336	        _state_dtype = T.dtype
   337	        _phis_dtype = state.phis.data.dtype
   338	        if micro_fn is None:
   339	            return NonHydrostaticTendencies(
   340	                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_micro", dims=dims_3d, units="m/s^2"),
   341	                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_micro", dims=dims_3d, units="m/s^2"),
   342	                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_micro", dims=dims_w, units="m/s^2"),
   343	                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dtheta_prime_dt_micro", dims=dims_3d, units="K/s"),
   344	                drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_micro", dims=dims_3d, units="kg/m^3/s"),
   345	                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
   346	                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_micro", dims=dims_tr, units="1/s"),
   347	            )
   348	
   349	        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
   350	
   351	        # Terrain-aware layer thickness and interface pressure.
   352	        z_half_3d = terrain_metric.z_half_3d
   353	        dz = jnp.abs(z_half_3d[..., :-1] - z_half_3d[..., 1:]).reshape(ncol, nlev)
   354	        p_half = reconstruct_half_level_pressure_hydrostatic(
   355	            p_full=p,
   356	            rho_full=rho_total,
   357	            z_half=z_half_3d,
   358	        ).reshape(ncol, nlev + 1)
   359	
   360	        # Reshape to columns
   361	        T_col = T.reshape(ncol, nlev)
   362	        p_full_col = p.reshape(ncol, nlev)
   363	        rho_col = rho_total.reshape(ncol, nlev)
   364	
   365	        # Map tracers -> HydrometeorState
   366	        # [0]=q_v, [1]=q_c, [2]=q_r, [3]=q_i, [4]=q_s, [5]=q_g, [6]=N_c, [7]=N_r, [8]=N_i
   367	        def _get_tracer(idx):
   368	            if n_tracers > idx:
   369	                return tracers[..., idx].reshape(ncol, nlev)
   370	            return jnp.zeros((ncol, nlev), dtype=_state_dtype)
   371	
   372	        q_v_col = _get_tracer(0)
   373	        hydrometeors = HydrometeorState(
   374	            q_c=_get_tracer(1),
   375	            q_r=_get_tracer(2),
   376	            q_i=_get_tracer(3),
   377	            q_s=_get_tracer(4),
   378	            q_g=_get_tracer(5),
   379	            N_c=_get_tracer(6),
   380	            N_r=_get_tracer(7),
   381	            N_i=_get_tracer(8),
   382	        )
   383	
   384	        if is_ml:
   385	            if _ml_model_cache[0] is None:
   386	                key = jax.random.PRNGKey(scheme_config.seed)
   387	                _ml_model_cache[0] = MicrophysicsEmulator(
   388	                    scheme_config.n_input, scheme_config.n_hidden,
   389	                    scheme_config.n_layers, scheme_config.n_output, key=key,
   390	                )
   391	            micro_out = micro_fn(
   392	                T_col, q_v_col, hydrometeors,
   393	                p_full_col, p_half, rho_col, dz, dt,
   394	                scheme_config, _ml_model_cache[0],
   395	            )
   396	        else:
   397	            micro_out = micro_fn(
   398	                T_col, q_v_col, hydrometeors,
   399	                p_full_col, p_half, rho_col, dz, dt, scheme_config,
   400	            )
   401	
   402	        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner).
   403	        dT_dt = micro_out.dT_dt.reshape(shape_3d)
   404	        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)
   405	
   406	        # Map output fields -> dtracers_dt
   407	        dtracers = jnp.zeros_like(tracers)
   408	        # Tracer mapping: 0=q_v, 1=q_c, 2=q_r, 3=q_i, 4=q_s, 5=q_g, 6=N_c, 7=N_r, 8=N_i
   409	        tend_fields = [
   410	            micro_out.dq_v_dt, micro_out.dq_c_dt, micro_out.dq_r_dt,
   411	            micro_out.dq_i_dt, micro_out.dq_s_dt, micro_out.dq_g_dt,
   412	            micro_out.dN_c_dt, micro_out.dN_r_dt, micro_out.dN_i_dt,
   413	        ]
   414	        for idx, field in enumerate(tend_fields):
   415	            if n_tracers > idx:
   416	                dtracers = dtracers.at[..., idx].set(field.reshape(shape_3d))
   417	
   418	        return NonHydrostaticTendencies(
   419	            du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_micro", dims=dims_3d, units="m/s^2"),
   420	            dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_micro", dims=dims_3d, units="m/s^2"),
   421	            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_micro", dims=dims_w, units="m/s^2"),
   422	            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_micro", dims=dims_3d, units="K/s"),
   423	            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_micro", dims=dims_3d, units="kg/m^3/s"),
   424	            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
   425	            dtracers_dt=Field(data=dtracers, name="dtracers_dt_micro", dims=dims_tr, units="1/s"),
   426	        )
   427	
   428	    def reset_state():
   429	        _ml_model_cache[0] = None
   430	
   431	    physics_fn.reset_state = reset_state
   432	    return physics_fn
   433	
   434	
   435	# ===========================================================================
   436	# Spectral PE
   437	# ===========================================================================
   438	
   439	def _make_spectral_pe_microphysics(
   440	    microphysics_config: MicrophysicsConfig,
   441	    dt: float,
   442	) -> Callable:
   443	    """Create microphysics physics_fn for SpectralPEModel.
   444	
   445	    Signature: (state, grid, sigma_coord, grid_fields=None) -> SpectralHydrostaticState
   446	
   447	    The bridge pulls ``q_v`` and the full hydrometeor state out of
   448	    ``state.tracers`` (when present), runs the column microphysics
   449	    backend, and returns a ``SpectralHydrostaticState`` whose ``T_hat``
   450	    carries the spectral latent-heating tendency *and* whose ``tracers``
   451	    dict carries grid-space ``dq_v_dt`` / ``dq_c_dt`` / ``dq_r_dt`` /
   452	    etc.  The dycore RHS (``spectral_pe_tendencies``) adds these tracer
   453	    tendencies to its own advective tendencies during the SSP-RK stages.
   454	    """
   455	    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(microphysics_config)
   456	    is_ml = scheme_name == "ml_emulator"
   457	    _ml_model_cache = [None]
   458	
   459	    # Tracer key → MicrophysicsOutput attribute name.  Mirrors the
   460	    # ``HydrometeorState`` field layout in ``microphysics/output.py``
   461	    # plus ``q_v``.  The dycore RHS only flows tendencies for keys that
   462	    # exist on the input ``state.tracers``; missing keys are silently
   463	    # dropped (no carry to write into).
   464	    _TRACER_TEND_MAP = {
   465	        "q_v": "dq_v_dt",
   466	        "q_c": "dq_c_dt",
   467	        "q_r": "dq_r_dt",
   468	        "q_i": "dq_i_dt",
   469	        "q_s": "dq_s_dt",
   470	        "q_g": "dq_g_dt",
   471	        "N_c": "dN_c_dt",
   472	        "N_r": "dN_r_dt",
   473	        "N_i": "dN_i_dt",
   474	    }
   475	
   476	    def physics_fn(state, grid, sigma_coord, grid_fields=None):
   477	        from legoesm.atmosphere.dynamics.spectral_pe import (
   478	            SpectralHydrostaticState,
   479	            spectral_pe_to_grid,
   480	        )
   481	        from legoesm.atmosphere.physics._shared import zero_like_tracers
   482	        from legoesm.grids.gaussian import sh_analysis_3d
   483	
   484	        # Transform spectral state to grid space
   485	        fields = grid_fields
   486	        if fields is None:
   487	            fields = spectral_pe_to_grid(state, grid, sigma_coord)
   488	        T = fields['T']
   489	        p_s = fields['p_s']
   490	
   491	        nlev = sigma_coord.n_levels
   492	        n_lat, n_lon = p_s.shape
   493	
   494	        zero_3d = jnp.zeros_like(state.vor_hat.data)
   495	        zero_2d = jnp.zeros_like(state.lnps_hat.data)
   496	        # Pin the column-physics dtype to the gridded state precision so
   497	        # we do not silently flow x64 zeros into the column path.
   498	        _state_dtype = T.dtype
   499	
   500	        if micro_fn is None:
   501	            # Mirror the input tracer pytree shape with zeros so the
   502	            # orchestrator's accumulator and the dycore RHS see a
   503	            # consistent tendency structure even when microphysics is
   504	            # disabled.
   505	            return SpectralHydrostaticState(
   506	                vor_hat=state.vor_hat.replace(data=zero_3d),
   507	                div_hat=state.div_hat.replace(data=zero_3d),
   508	                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
   509	                lnps_hat=state.lnps_hat.replace(data=zero_2d),
   510	                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
   511	                tracers=zero_like_tracers(state.tracers),
   512	            )
   513	
   514	        # Pressure at full and half levels
   515	        sigma_full = sigma_coord.sigma_full
   516	        sigma_half = sigma_coord.sigma_half
   517	        p_full = p_s[..., None] * sigma_full
   518	        p_half = p_s[..., None] * sigma_half
   519	
   520	        # Reshape to columns
   521	        ncol = n_lat * n_lon
   522	        T_col = T.reshape(ncol, nlev)
   523	        p_full_col = p_full.reshape(ncol, nlev)
   524	        p_half_col = p_half.reshape(ncol, nlev + 1)
   525	
   526	        # Pull tracer fields out of ``state.tracers`` and reshape to the
   527	        # column-physics ``(ncol, nlev)`` layout.  Backend microphysics
   528	        # schemes assume non-negative mixing ratios, so clip on the way
   529	        # in (matches the hydrostatic bridge's ``_get_tracer``).
   530	        def _get_tracer(name):
   531	            if state.tracers is not None and name in state.tracers:
   532	                raw = state.tracers[name]
   533	                data = raw.data if hasattr(raw, "data") else raw
   534	                return jnp.maximum(data.reshape(ncol, nlev), 0.0)
   535	            return jnp.zeros((ncol, nlev), dtype=_state_dtype)
   536	
   537	        q_v_col = _get_tracer("q_v")
   538	
   539	        rho = _compute_rho(T_col, p_full_col)
   540	        dz = _compute_heights_from_sigma(T_col, p_half_col)
   541	
   542	        hydrometeors = HydrometeorState(
   543	            q_c=_get_tracer("q_c"),
   544	            q_r=_get_tracer("q_r"),
   545	            q_i=_get_tracer("q_i"),
   546	            q_s=_get_tracer("q_s"),
   547	            q_g=_get_tracer("q_g"),
   548	            N_c=_get_tracer("N_c"),
   549	            N_r=_get_tracer("N_r"),
   550	            N_i=_get_tracer("N_i"),
   551	        )
   552	
   553	        if is_ml:
   554	            if _ml_model_cache[0] is None:
   555	                key = jax.random.PRNGKey(scheme_config.seed)
   556	                _ml_model_cache[0] = MicrophysicsEmulator(
   557	                    scheme_config.n_input, scheme_config.n_hidden,
   558	                    scheme_config.n_layers, scheme_config.n_output, key=key,
   559	                )
   560	            micro_out = micro_fn(
   561	                T_col, q_v_col, hydrometeors,
   562	                p_full_col, p_half_col, rho, dz, dt,
   563	                scheme_config, _ml_model_cache[0],
   564	            )
   565	        else:
   566	            micro_out = micro_fn(
   567	                T_col, q_v_col, hydrometeors,
   568	                p_full_col, p_half_col, rho, dz, dt, scheme_config,
   569	            )
   570	
   571	        dT_dt = micro_out.dT_dt.reshape(n_lat, n_lon, nlev)
   572	
   573	        # Transform T tendency to spectral space
   574	        dT_hat = sh_analysis_3d(grid, dT_dt)
   575	
   576	        # Build the tracer tendency dict in grid-space ``(n_lat, n_lon,
   577	        # nlev)`` layout, matching ``SpectralHydrostaticState.tracers``.
   578	        # Wrap each tendency back into the same container type as the
   579	        # input state's tracer (``Field`` vs raw ``jax.Array``) so the
   580	        # SSP-RK ``tree.map`` pytree leaves line up.  Untouched tracer
   581	        # keys are mirrored as zeros via ``zero_like_tracers``.
   582	        tracers_tend = None
   583	        if state.tracers is not None:
   584	            tt = {}
   585	            for name, attr in _TRACER_TEND_MAP.items():
   586	                if name not in state.tracers:
   587	                    continue
   588	                template = state.tracers[name]
   589	                tend_grid = getattr(micro_out, attr).reshape(
   590	                    n_lat, n_lon, nlev,
   591	                )
   592	                if hasattr(template, "data") and hasattr(template, "replace"):
   593	                    tt[name] = template.replace(
   594	                        data=tend_grid.astype(template.data.dtype),
   595	                    )
   596	                else:
   597	                    tt[name] = tend_grid.astype(template.dtype)
   598	            # Mirror any untouched tracer keys (e.g. a passive scalar
   599	            # the user attached) as zeros so the orchestrator's
   600	            # accumulator and the dycore RHS see a complete pytree.
   601	            zeros = zero_like_tracers(state.tracers)
   602	            if zeros is not None:
   603	                for k, zv in zeros.items():
   604	                    tt.setdefault(k, zv)
   605	            tracers_tend = tt
   606	
   607	        return SpectralHydrostaticState(
   608	            vor_hat=state.vor_hat.replace(data=zero_3d),
   609	            div_hat=state.div_hat.replace(data=zero_3d),
   610	            T_hat=state.T_hat.replace(data=dT_hat),
   611	            lnps_hat=state.lnps_hat.replace(data=zero_2d),
   612	            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
   613	            tracers=tracers_tend,
   614	        )
   615	
   616	    def reset_state():
   617	        _ml_model_cache[0] = None
   618	
   619	    physics_fn.reset_state = reset_state
   620	    return physics_fn

 succeeded in 0ms:
     1	"""Unit tests for the microphysics module.
     2	
     3	Tests all 6 backends (Kessler, Sundqvist, Seifert-Beheng, Morrison,
     4	Thompson, ML emulator) and the integration bridge for hydrostatic
     5	and non-hydrostatic dycores.
     6	"""
     7	
     8	from __future__ import annotations
     9	
    10	import pytest
    11	import jax
    12	import jax.numpy as jnp
    13	
    14	from legoesm.atmosphere.physics.microphysics.config import (
    15	    MicrophysicsConfig,
    16	    KesslerConfig,
    17	    SundqvistConfig,
    18	    SeifertBehengConfig,
    19	    MorrisonConfig,
    20	    ThompsonConfig,
    21	    MLEmulatorConfig,
    22	)
    23	from legoesm.atmosphere.physics.microphysics.output import (
    24	    HydrometeorState,
    25	    MicrophysicsOutput,
    26	    make_zero_hydrometeors,
    27	    make_zero_output,
    28	    sedimentation_tendency,
    29	)
    30	from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
    31	from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
    32	from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
    33	from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
    34	from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
    35	from legoesm.atmosphere.physics.microphysics.ml_emulator import (
    36	    ml_microphysics,
    37	    MicrophysicsEmulator,
    38	)
    39	from legoesm.atmosphere.physics.microphysics.integration import (
    40	    make_microphysics_physics,
    41	)
    42	from legoesm.thermo import saturation_mixing_ratio
    43	from legoesm import constants
    44	
    45	
    46	# ======================================================================
    47	# Test helpers
    48	# ======================================================================
    49	
    50	def _make_warm_columns(ncol=4, nlev=10):
    51	    """Create warm, near-saturated columns for testing warm-rain schemes."""
    52	    # Temperature profile: 290K at surface, 220K at top
    53	    T = jnp.linspace(220.0, 290.0, nlev)[None, :].repeat(ncol, axis=0)
    54	    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    55	    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    56	
    57	    # Near-saturated vapor
    58	    q_sat = saturation_mixing_ratio(T, p_full)
    59	    q_v = 0.95 * q_sat
    60	
    61	    # Cloud and rain water at mid levels
    62	    q_c = jnp.zeros((ncol, nlev))
    63	    q_c = q_c.at[:, 3:7].set(1e-3)
    64	    q_r = jnp.zeros((ncol, nlev))
    65	    q_r = q_r.at[:, 5:9].set(1e-4)
    66	
    67	    rho = p_full / (constants.R_d * T)
    68	    dp = p_half[:, 1:] - p_half[:, :-1]
    69	    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_full, 1.0))
    70	    dz = jnp.abs(dz)
    71	
    72	    hydrometeors = HydrometeorState(
    73	        q_c=q_c, q_r=q_r,
    74	        q_i=jnp.zeros((ncol, nlev)),
    75	        q_s=jnp.zeros((ncol, nlev)),
    76	        q_g=jnp.zeros((ncol, nlev)),
    77	        N_c=jnp.full((ncol, nlev), 1e8),
    78	        N_r=jnp.full((ncol, nlev), 1e4),
    79	        N_i=jnp.zeros((ncol, nlev)),
    80	    )
    81	
    82	    return T, q_v, hydrometeors, p_full, p_half, rho, dz
    83	
    84	
    85	def _make_cold_columns(ncol=4, nlev=10):
    86	    """Create cold columns with ice and snow for testing mixed-phase schemes."""
    87	    T = jnp.linspace(200.0, 265.0, nlev)[None, :].repeat(ncol, axis=0)
    88	    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    89	    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    90	
    91	    q_sat = saturation_mixing_ratio(T, p_full)
    92	    q_v = 0.9 * q_sat
    93	
    94	    q_c = jnp.zeros((ncol, nlev)).at[:, 6:9].set(5e-4)
    95	    q_r = jnp.zeros((ncol, nlev)).at[:, 7:9].set(5e-5)
    96	    q_i = jnp.zeros((ncol, nlev)).at[:, 2:6].set(1e-4)
    97	    q_s = jnp.zeros((ncol, nlev)).at[:, 3:7].set(5e-5)
    98	    q_g = jnp.zeros((ncol, nlev)).at[:, 4:7].set(2e-5)
    99	
   100	    rho = p_full / (constants.R_d * T)
   101	    dp = p_half[:, 1:] - p_half[:, :-1]
   102	    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_full, 1.0)))
   103	
   104	    hydrometeors = HydrometeorState(
   105	        q_c=q_c, q_r=q_r, q_i=q_i, q_s=q_s, q_g=q_g,
   106	        N_c=jnp.full((ncol, nlev), 1e8),
   107	        N_r=jnp.full((ncol, nlev), 1e4),
   108	        N_i=jnp.full((ncol, nlev), 1e3),
   109	    )
   110	
   111	    return T, q_v, hydrometeors, p_full, p_half, rho, dz
   112	
   113	
   114	def _make_evaporation_columns(ncol=4, nlev=10):
   115	    """Create warm, subsaturated columns with rain to isolate evaporation."""
   116	    T = jnp.full((ncol, nlev), 290.0)
   117	    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
   118	    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
   119	
   120	    q_sat = saturation_mixing_ratio(T, p_full)
   121	    q_v = 0.2 * q_sat
   122	
   123	    q_r = jnp.full((ncol, nlev), 1e-3)
   124	    z = jnp.zeros((ncol, nlev))
   125	    hydrometeors = HydrometeorState(
   126	        q_c=z,
   127	        q_r=q_r,
   128	        q_i=z,
   129	        q_s=z,
   130	        q_g=z,
   131	        N_c=jnp.full((ncol, nlev), 1e8),
   132	        N_r=jnp.full((ncol, nlev), 1e4),
   133	        N_i=jnp.full((ncol, nlev), 1e3),
   134	    )
   135	
   136	    rho = p_full / (constants.R_d * T)
   137	    dz = jnp.full((ncol, nlev), 500.0)
   138	    return T, q_v, hydrometeors, p_full, p_half, rho, dz
   139	
   140	
   141	# ======================================================================
   142	# Output helpers
   143	# ======================================================================
   144	
   145	class TestOutputHelpers:
   146	
   147	    def test_make_zero_hydrometeors_shape(self):
   148	        h = make_zero_hydrometeors(4, 10)
   149	        assert h.q_c.shape == (4, 10)
   150	        assert jnp.all(h.q_c == 0)
   151	
   152	    def test_make_zero_output_shape(self):
   153	        out = make_zero_output(4, 10)
   154	        assert out.dT_dt.shape == (4, 10)
   155	        assert out.precipitation.shape == (4,)
   156	
   157	    def test_sedimentation_tendency_shape(self):
   158	        q = jnp.ones((4, 10)) * 1e-4
   159	        rho = jnp.ones((4, 10)) * 1.2
   160	        V_t = jnp.ones((4, 10)) * 5.0
   161	        dz = jnp.ones((4, 10)) * 500.0
   162	        tend = sedimentation_tendency(q, rho, V_t, dz)
   163	        assert tend.shape == (4, 10)
   164	        assert jnp.all(jnp.isfinite(tend))
   165	
   166	
   167	# ======================================================================
   168	# Kessler tests
   169	# ======================================================================
   170	
   171	class TestKessler:
   172	
   173	    def test_output_shapes(self):
   174	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   175	        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   176	        assert out.dT_dt.shape == T.shape
   177	        assert out.precipitation.shape == (T.shape[0],)
   178	
   179	    def test_precipitation_non_negative(self):
   180	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   181	        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   182	        assert jnp.all(out.precipitation >= 0)
   183	
   184	    def test_nonzero_tendencies(self):
   185	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   186	        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   187	        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
   188	
   189	    def test_finite_outputs(self):
   190	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   191	        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   192	        for field in out:
   193	            assert jnp.all(jnp.isfinite(field)), f"Non-finite in {field}"
   194	
   195	    def test_dry_air_zero_tendency(self):
   196	        ncol, nlev = 4, 10
   197	        T = jnp.full((ncol, nlev), 280.0)
   198	        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
   199	        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
   200	        q_v = jnp.zeros((ncol, nlev))
   201	        rho = p_full / (constants.R_d * T)
   202	        dz = jnp.full((ncol, nlev), 500.0)
   203	        h = make_zero_hydrometeors(ncol, nlev)
   204	        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   205	        # With zero moisture, all ice/snow/graupel/number tendencies are exact zero
   206	        assert float(jnp.max(jnp.abs(out.dq_i_dt))) == 0.0
   207	        assert float(jnp.max(jnp.abs(out.dq_s_dt))) == 0.0
   208	        assert float(jnp.max(jnp.abs(out.dq_g_dt))) == 0.0
   209	
   210	    def test_differentiable(self):
   211	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   212	
   213	        def loss(T_in):
   214	            out = kessler_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   215	            return jnp.sum(out.dT_dt ** 2)
   216	
   217	        grad = jax.grad(loss)(T)
   218	        assert jnp.all(jnp.isfinite(grad))
   219	
   220	    def test_evaporation_enthalpy_balance(self):
   221	        """Evaporation cooling should balance vapor tendency latent energy."""
   222	        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
   223	        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   224	        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
   225	        assert float(jnp.max(jnp.abs(residual))) < 1e-6
   226	
   227	
   228	# ======================================================================
   229	# Sundqvist tests
   230	# ======================================================================
   231	
   232	class TestSundqvist:
   233	
   234	    def test_output_shapes(self):
   235	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   236	        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   237	        assert out.dT_dt.shape == T.shape
   238	        assert out.precipitation.shape == (T.shape[0],)
   239	
   240	    def test_precipitation_non_negative(self):
   241	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   242	        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   243	        assert jnp.all(out.precipitation >= 0)
   244	
   245	    def test_nonzero_tendencies(self):
   246	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   247	        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   248	        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
   249	
   250	    def test_finite_outputs(self):
   251	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   252	        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   253	        for field in out:
   254	            assert jnp.all(jnp.isfinite(field))
   255	
   256	    def test_below_RH_crit_no_condensation(self):
   257	        ncol, nlev = 4, 10
   258	        T = jnp.full((ncol, nlev), 280.0)
   259	        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
   260	        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
   261	        q_sat = saturation_mixing_ratio(T, p_full)
   262	        q_v = 0.5 * q_sat  # well below RH_crit=0.8
   263	        rho = p_full / (constants.R_d * T)
   264	        dz = jnp.full((ncol, nlev), 500.0)
   265	        h = make_zero_hydrometeors(ncol, nlev)
   266	        # Use very high sharpness to make the sigmoid effectively a step
   267	        config = SundqvistConfig(RH_crit=0.8, sigmoid_sharpness=200.0)
   268	        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
   269	        # Condensation should be much smaller than saturated case
   270	        out_sat = sundqvist_microphysics(T, q_sat, h, p_full, p_half, rho, dz, dt=10.0, config=config)
   271	        ratio = float(jnp.max(jnp.abs(out.dq_c_dt))) / float(jnp.max(jnp.abs(out_sat.dq_c_dt)) + 1e-20)
   272	        assert ratio < 0.1
   273	
   274	    def test_differentiable(self):
   275	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   276	
   277	        def loss(T_in):
   278	            out = sundqvist_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   279	            return jnp.sum(out.dT_dt ** 2)
   280	
   281	        grad = jax.grad(loss)(T)
   282	        assert jnp.all(jnp.isfinite(grad))
   283	
   284	
   285	# ======================================================================
   286	# Seifert-Beheng tests
   287	# ======================================================================
   288	
   289	class TestSeifertBeheng:
   290	
   291	    def test_output_shapes(self):
   292	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   293	        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   294	        assert out.dT_dt.shape == T.shape
   295	        assert out.dN_c_dt.shape == T.shape
   296	        assert out.dN_r_dt.shape == T.shape
   297	
   298	    def test_precipitation_non_negative(self):
   299	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   300	        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   301	        assert jnp.all(out.precipitation >= 0)
   302	
   303	    def test_nonzero_tendencies(self):
   304	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   305	        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   306	        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
   307	
   308	    def test_finite_outputs(self):
   309	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   310	        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   311	        for field in out:
   312	            assert jnp.all(jnp.isfinite(field))
   313	
   314	    def test_differentiable(self):
   315	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   316	
   317	        def loss(T_in):
   318	            out = seifert_beheng_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   319	            return jnp.sum(out.dT_dt ** 2)
   320	
   321	        grad = jax.grad(loss)(T)
   322	        assert jnp.all(jnp.isfinite(grad))
   323	
   324	    def test_evaporation_enthalpy_balance(self):
   325	        """Evaporation cooling should balance vapor tendency latent energy."""
   326	        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
   327	        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   328	        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
   329	        assert float(jnp.max(jnp.abs(residual))) < 1e-6
   330	
   331	
   332	# ======================================================================
   333	# Morrison tests
   334	# ======================================================================
   335	
   336	class TestMorrison:
   337	
   338	    def test_output_shapes(self):
   339	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   340	        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   341	        assert out.dT_dt.shape == T.shape
   342	        assert out.dq_i_dt.shape == T.shape
   343	        assert out.dN_i_dt.shape == T.shape
   344	
   345	    def test_precipitation_non_negative(self):
   346	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   347	        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   348	        assert jnp.all(out.precipitation >= 0)
   349	
   350	    def test_nonzero_tendencies(self):
   351	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   352	        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   353	        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
   354	
   355	    def test_finite_outputs(self):
   356	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   357	        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   358	        for field in out:
   359	            assert jnp.all(jnp.isfinite(field))
   360	
   361	    def test_ice_only_below_freezing(self):
   362	        """Ice tendencies should be near-zero when T > T_freeze everywhere."""
   363	        ncol, nlev = 4, 10
   364	        T = jnp.full((ncol, nlev), 290.0)
   365	        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
   366	        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
   367	        q_v = 0.5 * saturation_mixing_ratio(T, p_full)
   368	        rho = p_full / (constants.R_d * T)
   369	        dz = jnp.full((ncol, nlev), 500.0)
   370	        h = make_zero_hydrometeors(ncol, nlev)
   371	        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   372	        # Ice tendencies should be very small above freezing
   373	        assert float(jnp.max(jnp.abs(out.dq_i_dt))) < 1e-6
   374	
   375	    def test_differentiable(self):
   376	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   377	
   378	        def loss(T_in):
   379	            out = morrison_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   380	            return jnp.sum(out.dT_dt ** 2)
   381	
   382	        grad = jax.grad(loss)(T)
   383	        assert jnp.all(jnp.isfinite(grad))
   384	
   385	    def test_evaporation_enthalpy_balance(self):
   386	        """Warm-rain evaporation cooling should close latent energy tendency."""
   387	        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
   388	        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   389	        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
   390	        assert float(jnp.max(jnp.abs(residual))) < 5e-3
   391	
   392	
   393	# ======================================================================
   394	# Thompson tests
   395	# ======================================================================
   396	
   397	class TestThompson:
   398	
   399	    def test_output_shapes(self):
   400	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   401	        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   402	        assert out.dT_dt.shape == T.shape
   403	        assert out.dq_g_dt.shape == T.shape
   404	
   405	    def test_precipitation_non_negative(self):
   406	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   407	        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   408	        assert jnp.all(out.precipitation >= 0)
   409	
   410	    def test_nonzero_tendencies(self):
   411	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   412	        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   413	        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
   414	
   415	    def test_finite_outputs(self):
   416	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   417	        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   418	        for field in out:
   419	            assert jnp.all(jnp.isfinite(field))
   420	
   421	    def test_graupel_from_riming(self):
   422	        """With strong riming, graupel tendencies should be nonzero."""
   423	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   424	        # Boost cloud water and ice to promote riming
   425	        h = h._replace(
   426	            q_c=jnp.full_like(h.q_c, 5e-3),
   427	            q_i=jnp.full_like(h.q_i, 5e-3),
   428	            q_s=jnp.full_like(h.q_s, 5e-3),
   429	        )
   430	        config = ThompsonConfig(
   431	            rime_coeff=10.0,
   432	            rime_to_graupel_threshold=1e-6,
   433	        )
   434	        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
   435	        assert float(jnp.max(jnp.abs(out.dq_g_dt))) > 0
   436	
   437	    def test_differentiable(self):
   438	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   439	
   440	        def loss(T_in):
   441	            out = thompson_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   442	            return jnp.sum(out.dT_dt ** 2)
   443	
   444	        grad = jax.grad(loss)(T)
   445	        assert jnp.all(jnp.isfinite(grad))
   446	
   447	    def test_evaporation_enthalpy_balance(self):
   448	        """Warm-rain evaporation cooling should close latent energy tendency."""
   449	        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
   450	        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   451	        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
   452	        assert float(jnp.max(jnp.abs(residual))) < 5e-3
   453	
   454	
   455	# ======================================================================
   456	# ML Emulator tests
   457	# ======================================================================
   458	
   459	class TestMLEmulator:
   460	
   461	    def _make_model(self, config=None):
   462	        if config is None:
   463	            config = MLEmulatorConfig()
   464	        key = jax.random.PRNGKey(config.seed)
   465	        return MicrophysicsEmulator(
   466	            config.n_input, config.n_hidden, config.n_layers,
   467	            config.n_output, key=key,
   468	        )
   469	
   470	    def test_output_shapes(self):
   471	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   472	        config = MLEmulatorConfig()
   473	        model = self._make_model(config)
   474	        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
   475	                              config=config, model=model)
   476	        assert out.dT_dt.shape == T.shape
   477	        assert out.precipitation.shape == (T.shape[0],)
   478	
   479	    def test_precipitation_non_negative(self):
   480	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   481	        config = MLEmulatorConfig()
   482	        model = self._make_model(config)
   483	        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
   484	                              config=config, model=model)
   485	        assert jnp.all(out.precipitation >= 0)
   486	
   487	    def test_nonzero_tendencies(self):
   488	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   489	        config = MLEmulatorConfig()
   490	        model = self._make_model(config)
   491	        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
   492	                              config=config, model=model)
   493	        # Untrained model will still produce nonzero output
   494	        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
   495	
   496	    def test_finite_outputs(self):
   497	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   498	        config = MLEmulatorConfig()
   499	        model = self._make_model(config)
   500	        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
   501	                              config=config, model=model)
   502	        for field in out:
   503	            assert jnp.all(jnp.isfinite(field))
   504	
   505	    def test_differentiable(self):
   506	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   507	        config = MLEmulatorConfig()
   508	        model = self._make_model(config)
   509	
   510	        def loss(T_in):
   511	            out = ml_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0,
   512	                                  config=config, model=model)
   513	            return jnp.sum(out.dT_dt ** 2)
   514	
   515	        grad = jax.grad(loss)(T)
   516	        assert jnp.all(jnp.isfinite(grad))
   517	
   518	
   519	# ======================================================================
   520	# Integration bridge tests
   521	# ======================================================================
   522	
   523	class TestIntegrationHydrostatic:
   524	
   525	    @pytest.fixture
   526	    def setup(self):
   527	        from legoesm.grids.cubed_sphere import create_cubed_sphere
   528	        from legoesm.grids.vertical import create_sigma_coordinate
   529	        from legoesm.atmosphere.held_suarez import held_suarez_init
   530	
   531	        grid = create_cubed_sphere(8)
   532	        sigma = create_sigma_coordinate(10)
   533	        state = held_suarez_init(grid, sigma)
   534	        return state, grid, sigma
   535	
   536	    def test_hydrostatic_shapes(self, setup):
   537	        state, grid, sigma = setup
   538	        config = MicrophysicsConfig(scheme="kessler")
   539	        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
   540	        tend = physics_fn(state, grid, sigma)
   541	        assert tend.dT_dt.data.shape == state.T.data.shape
   542	
   543	    def test_none_scheme_zeros(self, setup):
   544	        state, grid, sigma = setup
   545	        config = MicrophysicsConfig(scheme="none")
   546	        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
   547	        tend = physics_fn(state, grid, sigma)
   548	        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) == 0.0
   549	
   550	    def test_grad_through_hydrostatic(self, setup):
   551	        state, grid, sigma = setup
   552	        config = MicrophysicsConfig(scheme="kessler")
   553	        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
   554	
   555	        def loss(T_data):
   556	            s = state._replace(T=state.T.replace(data=T_data))
   557	            tend = physics_fn(s, grid, sigma)
   558	            return jnp.sum(tend.dT_dt.data ** 2)
   559	
   560	        grad = jax.grad(loss)(state.T.data)
   561	        assert jnp.all(jnp.isfinite(grad))
   562	
   563	
   564	class TestIntegrationNonhydrostatic:
   565	
   566	    @pytest.fixture
   567	    def setup(self):
   568	        from legoesm.grids.cubed_sphere import create_cubed_sphere
   569	        from legoesm.grids.vertical import (
   570	            create_height_coordinate,
   571	            compute_terrain_metric,
   572	        )
   573	        from legoesm.core.field import Field
   574	        from legoesm.core.state import NonHydrostaticState
   575	
   576	        grid = create_cubed_sphere(8)
   577	        height_coord = create_height_coordinate(10, 10000.0)
   578	        z_s = jnp.zeros((6, grid.n, grid.n))
   579	        terrain_metric = compute_terrain_metric(z_s, height_coord)
   580	
   581	        nlev = height_coord.n_levels
   582	        n = grid.n
   583	        shape_3d = (6, n, n, nlev)
   584	        shape_w = (6, n, n, nlev + 1)
   585	        shape_2d = (6, n, n)
   586	        n_tracers = 3
   587	
   588	        state = NonHydrostaticState(
   589	            u=Field(data=jnp.zeros(shape_3d), name="u",
   590	                    dims=("face", "x", "y", "level"), units="m/s"),
   591	            v=Field(data=jnp.zeros(shape_3d), name="v",
   592	                    dims=("face", "x", "y", "level"), units="m/s"),
   593	            w=Field(data=jnp.zeros(shape_w), name="w",
   594	                    dims=("face", "x", "y", "level_half"), units="m/s"),
   595	            theta_prime=Field(data=jnp.zeros(shape_3d), name="theta_prime",
   596	                              dims=("face", "x", "y", "level"), units="K"),
   597	            rho_prime=Field(data=jnp.zeros(shape_3d), name="rho_prime",
   598	                            dims=("face", "x", "y", "level"), units="kg/m^3"),
   599	            phis=Field(data=jnp.zeros(shape_2d), name="phis",
   600	                       dims=("face", "x", "y"), units="m^2/s^2"),
   601	            tracers=Field(
   602	                data=jnp.zeros((*shape_3d, n_tracers)),
   603	                name="tracers",
   604	                dims=("face", "x", "y", "level", "tracer"),
   605	                units="kg/kg",
   606	            ),
   607	        )
   608	        return state, grid, height_coord, terrain_metric
   609	
   610	    def test_nonhydrostatic_shapes(self, setup):
   611	        state, grid, hc, tm = setup
   612	        config = MicrophysicsConfig(scheme="kessler")
   613	        physics_fn = make_microphysics_physics(config, "nonhydrostatic", dt=1.0)
   614	        tend = physics_fn(state, grid, hc, tm)
   615	        assert tend.dtheta_prime_dt.data.shape == state.theta_prime.data.shape
   616	        assert tend.dtracers_dt.data.shape == state.tracers.data.shape
   617	
   618	    def test_nonhydrostatic_nonzero_heating(self, setup):
   619	        state, grid, hc, tm = setup
   620	        # Add moisture to trigger microphysics
   621	        tracers = state.tracers.data.at[..., 0].set(0.01)
   622	        state = state._replace(tracers=state.tracers.replace(data=tracers))
   623	        config = MicrophysicsConfig(scheme="kessler")
   624	        physics_fn = make_microphysics_physics(config, "nonhydrostatic", dt=1.0)
   625	        tend = physics_fn(state, grid, hc, tm)
   626	        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
   627	        assert jnp.all(jnp.isfinite(tend.dtracers_dt.data))
   628	
   629	
   630	class TestSchemeSelection:
   631	
   632	    def test_scheme_selection(self):
   633	        """All 6 scheme strings are accepted by the factory."""
   634	        for scheme in ["kessler", "sundqvist", "seifert_beheng",
   635	                       "morrison", "thompson", "ml_emulator", "none"]:
   636	            config = MicrophysicsConfig(scheme=scheme)
   637	            physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
   638	            assert callable(physics_fn)
   639	
   640	    def test_invalid_scheme_raises(self):
   641	        config = MicrophysicsConfig(scheme="invalid")
   642	        with pytest.raises(ValueError, match="Unknown microphysics scheme"):
   643	            make_microphysics_physics(config, "hydrostatic", dt=300.0)
   644	
   645	    def test_invalid_model_type_raises(self):
   646	        config = MicrophysicsConfig(scheme="kessler")
   647	        with pytest.raises(ValueError, match="Unknown model_type"):
   648	            make_microphysics_physics(config, "invalid", dt=300.0)
   649	
   650	
   651	class TestKesslerNonhydrostatic:
   652	
   653	    def test_kessler_nonhydrostatic_tendencies(self):
   654	        """Kessler microphysics via integration bridge on nonhydrostatic state."""
   655	        from legoesm.atmosphere.physics.microphysics.config import KesslerConfig, MicrophysicsConfig
   656	        from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
   657	        from legoesm.grids.cubed_sphere import create_cubed_sphere
   658	        from legoesm.grids.vertical import (
   659	            create_height_coordinate,
   660	            compute_terrain_metric,
   661	        )
   662	        from legoesm.core.field import Field
   663	        from legoesm.core.state import NonHydrostaticState
   664	
   665	        grid = create_cubed_sphere(8)
   666	        hc = create_height_coordinate(10, 10000.0)
   667	        z_s = jnp.zeros((6, grid.n, grid.n))
   668	        tm = compute_terrain_metric(z_s, hc)
   669	
   670	        nlev = hc.n_levels
   671	        n = grid.n
   672	        shape_3d = (6, n, n, nlev)
   673	        shape_w = (6, n, n, nlev + 1)
   674	        shape_2d = (6, n, n)
   675	
   676	        state = NonHydrostaticState(
   677	            u=Field(data=jnp.zeros(shape_3d), name="u",
   678	                    dims=("face", "x", "y", "level"), units="m/s"),
   679	            v=Field(data=jnp.zeros(shape_3d), name="v",
   680	                    dims=("face", "x", "y", "level"), units="m/s"),
   681	            w=Field(data=jnp.zeros(shape_w), name="w",
   682	                    dims=("face", "x", "y", "level_half"), units="m/s"),
   683	            theta_prime=Field(data=jnp.zeros(shape_3d), name="theta_prime",
   684	                              dims=("face", "x", "y", "level"), units="K"),
   685	            rho_prime=Field(data=jnp.zeros(shape_3d), name="rho_prime",
   686	                            dims=("face", "x", "y", "level"), units="kg/m^3"),
   687	            phis=Field(data=jnp.zeros(shape_2d), name="phis",
   688	                       dims=("face", "x", "y"), units="m^2/s^2"),
   689	            tracers=Field(
   690	                data=jnp.zeros((*shape_3d, 3)),
   691	                name="tracers",
   692	                dims=("face", "x", "y", "level", "tracer"),
   693	                units="kg/kg",
   694	            ),
   695	        )
   696	
   697	        config = KesslerConfig()
   698	        micro_config = MicrophysicsConfig(scheme="kessler", kessler=config)
   699	        physics_fn = make_microphysics_physics(micro_config, model_type="nonhydrostatic", dt=1.0)
   700	        tend = physics_fn(state, grid, hc, tm)
   701	        assert tend.dtheta_prime_dt.data.shape == shape_3d
   702	        assert tend.dtracers_dt.data.shape == (*shape_3d, 3)
   703	        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
   704	
   705	
   706	# ======================================================================
   707	# AMIP integration tests (checkpoint save/load with q_c/q_r)
   708	# ======================================================================
   709	
   710	class TestCheckpointWithHydrometeors:
   711	    """Test checkpoint save/load roundtrip with q_c and q_r fields."""
   712	
   713	    @pytest.fixture
   714	    def setup(self, tmp_path):
   715	        from legoesm.grids.cubed_sphere import create_cubed_sphere
   716	        from legoesm.grids.vertical import create_sigma_coordinate
   717	        from legoesm.atmosphere.held_suarez import held_suarez_init
   718	        from legoesm.forcing.amip_config import (
   719	            AMIPExperimentConfig, save_checkpoint, load_checkpoint,
   720	        )
   721	
   722	        grid = create_cubed_sphere(8)
   723	        sigma = create_sigma_coordinate(10)
   724	        state = held_suarez_init(grid, sigma)
   725	        n = grid.n
   726	        nlev = 10
   727	        q_v = jnp.full((6, n, n, nlev), 0.005)
   728	        q_c = jnp.full((6, n, n, nlev), 1e-4)
   729	        q_r = jnp.full((6, n, n, nlev), 5e-5)
   730	        config = AMIPExperimentConfig(
   731	            resolution=n, nlev=nlev, microphysics="kessler",
   732	        )
   733	        return state, q_v, q_c, q_r, config, grid, sigma, tmp_path
   734	
   735	    def test_roundtrip_with_hydrometeors(self, setup):
   736	        state, q_v, q_c, q_r, config, grid, sigma, tmp_path = setup
   737	        from legoesm.forcing.amip_config import save_checkpoint, load_checkpoint
   738	        path = tmp_path / "ckpt.npz"
   739	        save_checkpoint(path, state, q_v, step=100, day=10.0, config=config,
   740	                        q_c=q_c, q_r=q_r)
   741	
   742	        loaded = load_checkpoint(path, grid, sigma)
   743	        assert len(loaded) == 8  # state, q_v, step, day, config, diag, q_c, q_r
   744	        _, _, step, day, _, _, q_c_loaded, q_r_loaded = loaded
   745	        assert step == 100
   746	        assert day == 10.0
   747	        assert q_c_loaded is not None
   748	        assert q_r_loaded is not None
   749	        assert float(jnp.max(jnp.abs(q_c_loaded - q_c))) < 1e-10
   750	        assert float(jnp.max(jnp.abs(q_r_loaded - q_r))) < 1e-10
   751	
   752	    def test_roundtrip_without_hydrometeors(self, setup):
   753	        """Old checkpoints without q_c/q_r should load with None."""
   754	        state, q_v, q_c, q_r, config, grid, sigma, tmp_path = setup
   755	        from legoesm.forcing.amip_config import save_checkpoint, load_checkpoint
   756	        path = tmp_path / "ckpt_old.npz"
   757	        # Save without q_c/q_r (old-style)
   758	        save_checkpoint(path, state, q_v, step=50, day=5.0, config=config)
   759	
   760	        loaded = load_checkpoint(path, grid, sigma)
   761	        _, _, _, _, _, _, q_c_loaded, q_r_loaded = loaded
   762	        assert q_c_loaded is None
   763	        assert q_r_loaded is None
   764	
   765	    def test_config_microphysics_field_roundtrip(self, setup):
   766	        state, q_v, q_c, q_r, config, grid, sigma, tmp_path = setup
   767	        from legoesm.forcing.amip_config import save_checkpoint, load_checkpoint
   768	        path = tmp_path / "ckpt_cfg.npz"
   769	        save_checkpoint(path, state, q_v, step=10, day=1.0, config=config,
   770	                        q_c=q_c, q_r=q_r)
   771	        _, _, _, _, restored_config, _, _, _ = load_checkpoint(path, grid, sigma)
   772	        assert restored_config.microphysics == "kessler"
   773	
   774	
   775	class TestAMIPMicrophysicsConfig:
   776	    """Test AMIPExperimentConfig microphysics field."""
   777	
   778	    def test_default_microphysics_none(self):
   779	        from legoesm.forcing.amip_config import AMIPExperimentConfig
   780	        config = AMIPExperimentConfig()
   781	        assert config.microphysics == "none"
   782	
   783	    def test_kessler_microphysics(self):
   784	        from legoesm.forcing.amip_config import AMIPExperimentConfig
   785	        config = AMIPExperimentConfig(microphysics="kessler")
   786	        assert config.microphysics == "kessler"
   787	
   788	    def test_config_json_roundtrip(self):
   789	        from legoesm.forcing.amip_config import (
   790	            AMIPExperimentConfig, config_to_dict, config_from_dict,
   791	        )
   792	        config = AMIPExperimentConfig(microphysics="sundqvist")
   793	        d = config_to_dict(config)
   794	        assert d["microphysics"] == "sundqvist"
   795	        restored = config_from_dict(d)
   796	        assert restored.microphysics == "sundqvist"
   797	
   798	
   799	class TestMicrophysicsInPhysicsStep:
   800	    """Test microphysics backend dispatch for AMIP-like operator-split setup."""
   801	
   802	    def test_kessler_in_column_physics(self):
   803	        """Kessler produces non-trivial tendencies on moist columns."""
   804	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   805	        config = KesslerConfig()
   806	        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz,
   807	                                   dt=600.0, config=config)
   808	        # Should have nonzero cloud water and rain tendencies
   809	        assert float(jnp.max(jnp.abs(out.dq_c_dt))) > 0
   810	        assert float(jnp.max(jnp.abs(out.dq_r_dt))) > 0
   811	        # Latent heating should be nonzero
   812	        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
   813	
   814	    def test_multi_step_stability(self):
   815	        """Multiple Kessler steps shouldn't produce NaN."""
   816	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   817	        config = KesslerConfig()
   818	        dt = 600.0
   819	        q_c = h.q_c
   820	        q_r = h.q_r
   821	
   822	        for _ in range(10):
   823	            h_step = h._replace(q_c=q_c, q_r=q_r)
   824	            out = kessler_microphysics(T, q_v, h_step, p_full, p_half, rho, dz,
   825	                                       dt=dt, config=config)
   826	            q_v = jnp.maximum(q_v + dt * out.dq_v_dt, 0.0)
   827	            q_c = jnp.maximum(q_c + dt * out.dq_c_dt, 0.0)
   828	            q_r = jnp.maximum(q_r + dt * out.dq_r_dt, 0.0)
   829	            T = T + dt * out.dT_dt
   830	
   831	        assert jnp.all(jnp.isfinite(T))
   832	        assert jnp.all(jnp.isfinite(q_v))
   833	        assert jnp.all(jnp.isfinite(q_c))
   834	        assert jnp.all(jnp.isfinite(q_r))

exec
/bin/zsh -lc "sed -n '420,540p' tests/atmosphere/hydrostatic/unit/test_microphysics.py | nl -ba -v 420" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   420	
   421	    def test_graupel_from_riming(self):
   422	        """With strong riming, graupel tendencies should be nonzero."""
   423	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   424	        # Boost cloud water and ice to promote riming
   425	        h = h._replace(
   426	            q_c=jnp.full_like(h.q_c, 5e-3),
   427	            q_i=jnp.full_like(h.q_i, 5e-3),
   428	            q_s=jnp.full_like(h.q_s, 5e-3),
   429	        )
   430	        config = ThompsonConfig(
   431	            rime_coeff=10.0,
   432	            rime_to_graupel_threshold=1e-6,
   433	        )
   434	        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
   435	        assert float(jnp.max(jnp.abs(out.dq_g_dt))) > 0
   436	
   437	    def test_differentiable(self):
   438	        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
   439	
   440	        def loss(T_in):
   441	            out = thompson_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   442	            return jnp.sum(out.dT_dt ** 2)
   443	
   444	        grad = jax.grad(loss)(T)
   445	        assert jnp.all(jnp.isfinite(grad))
   446	
   447	    def test_evaporation_enthalpy_balance(self):
   448	        """Warm-rain evaporation cooling should close latent energy tendency."""
   449	        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
   450	        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
   451	        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
   452	        assert float(jnp.max(jnp.abs(residual))) < 5e-3
   453	
   454	
   455	# ======================================================================
   456	# ML Emulator tests
   457	# ======================================================================
   458	
   459	class TestMLEmulator:
   460	
   461	    def _make_model(self, config=None):
   462	        if config is None:
   463	            config = MLEmulatorConfig()
   464	        key = jax.random.PRNGKey(config.seed)
   465	        return MicrophysicsEmulator(
   466	            config.n_input, config.n_hidden, config.n_layers,
   467	            config.n_output, key=key,
   468	        )
   469	
   470	    def test_output_shapes(self):
   471	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   472	        config = MLEmulatorConfig()
   473	        model = self._make_model(config)
   474	        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
   475	                              config=config, model=model)
   476	        assert out.dT_dt.shape == T.shape
   477	        assert out.precipitation.shape == (T.shape[0],)
   478	
   479	    def test_precipitation_non_negative(self):
   480	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   481	        config = MLEmulatorConfig()
   482	        model = self._make_model(config)
   483	        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
   484	                              config=config, model=model)
   485	        assert jnp.all(out.precipitation >= 0)
   486	
   487	    def test_nonzero_tendencies(self):
   488	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   489	        config = MLEmulatorConfig()
   490	        model = self._make_model(config)
   491	        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
   492	                              config=config, model=model)
   493	        # Untrained model will still produce nonzero output
   494	        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
   495	
   496	    def test_finite_outputs(self):
   497	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   498	        config = MLEmulatorConfig()
   499	        model = self._make_model(config)
   500	        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
   501	                              config=config, model=model)
   502	        for field in out:
   503	            assert jnp.all(jnp.isfinite(field))
   504	
   505	    def test_differentiable(self):
   506	        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
   507	        config = MLEmulatorConfig()
   508	        model = self._make_model(config)
   509	
   510	        def loss(T_in):
   511	            out = ml_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0,
   512	                                  config=config, model=model)
   513	            return jnp.sum(out.dT_dt ** 2)
   514	
   515	        grad = jax.grad(loss)(T)
   516	        assert jnp.all(jnp.isfinite(grad))
   517	
   518	
   519	# ======================================================================
   520	# Integration bridge tests
   521	# ======================================================================
   522	
   523	class TestIntegrationHydrostatic:
   524	
   525	    @pytest.fixture
   526	    def setup(self):
   527	        from legoesm.grids.cubed_sphere import create_cubed_sphere
   528	        from legoesm.grids.vertical import create_sigma_coordinate
   529	        from legoesm.atmosphere.held_suarez import held_suarez_init
   530	
   531	        grid = create_cubed_sphere(8)
   532	        sigma = create_sigma_coordinate(10)
   533	        state = held_suarez_init(grid, sigma)
   534	        return state, grid, sigma
   535	
   536	    def test_hydrostatic_shapes(self, setup):
   537	        state, grid, sigma = setup
   538	        config = MicrophysicsConfig(scheme="kessler")
   539	        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
   540	        tend = physics_fn(state, grid, sigma)

exec
/bin/zsh -lc 'rg -n "xfail|conserv|water|enthalpy|latent|mass|negative|nonnegative|dt" tests/unit/test_physics_microphysics.py tests/atmosphere/hydrostatic/unit/test_microphysics.py tests/unit/test_warm_rain.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
tests/unit/test_physics_microphysics.py:3:Tests total water conservation, temperature-moisture coupling (Clausius-
tests/unit/test_physics_microphysics.py:75:def _call_scheme(name, T, q_v, hydro, p_full, p_half, rho, dz, dt=300.0):
tests/unit/test_physics_microphysics.py:78:        return kessler_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
tests/unit/test_physics_microphysics.py:81:        return sundqvist_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
tests/unit/test_physics_microphysics.py:84:        return seifert_beheng_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
tests/unit/test_physics_microphysics.py:87:        return morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
tests/unit/test_physics_microphysics.py:90:        return thompson_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
tests/unit/test_physics_microphysics.py:119:    """cp * dT_dt approx Lv * condensation_rate (first order).
tests/unit/test_physics_microphysics.py:127:    lhs = constants.c_pd * out.dT_dt
tests/unit/test_physics_microphysics.py:128:    rhs = -constants.L_v * out.dq_v_dt
tests/unit/test_physics_microphysics.py:145:    """Starting from supersaturated, vapor should decrease (dq_v_dt < 0)."""
tests/unit/test_physics_microphysics.py:151:    # Some levels should show condensation (dq_v_dt < 0)
tests/unit/test_physics_microphysics.py:152:    min_dqv = float(jnp.min(out.dq_v_dt))
tests/unit/test_physics_microphysics.py:154:        f"{scheme}: no condensation in supersaturated column, min dq_v_dt = {min_dqv:.2e}"
tests/unit/test_physics_microphysics.py:159:# 4d  Precipitation non-negative
tests/unit/test_physics_microphysics.py:163:def test_precipitation_non_negative(scheme):
tests/unit/test_physics_microphysics.py:169:        f"{scheme}: negative precipitation = {min_precip:.2e}"
tests/unit/test_physics_microphysics.py:186:        ice_formation = out.dq_i_dt[warm_mask]
tests/unit/test_physics_microphysics.py:190:            f"{scheme}: ice forms above freezing, max dq_i_dt = {max_ice_form:.2e}"
tests/unit/test_physics_microphysics.py:214:    max_dqr_lo = float(jnp.max(out_lo.dq_r_dt))
tests/unit/test_physics_microphysics.py:215:    max_dqr_hi = float(jnp.max(out_hi.dq_r_dt))
tests/unit/test_physics_microphysics.py:228:    """|dT_dt| should be bounded (< 10 K/s for strongly supersaturated)."""
tests/unit/test_physics_microphysics.py:231:    max_hr = float(jnp.max(jnp.abs(out.dT_dt)))
tests/unit/test_physics_microphysics.py:233:        f"{scheme}: |dT_dt| = {max_hr:.2e} K/s exceeds 10 K/s"
tests/unit/test_warm_rain.py:34:    q_c = jnp.full((ncol, nlev), 5.0e-4)         # cloud water
tests/unit/test_warm_rain.py:43:    cond, q_sat = saturation_adjustment(T, q_v, p, dt=300.0)
tests/unit/test_warm_rain.py:107:    _, q_sat = saturation_adjustment(T, q_v_low, p, dt=300.0)
tests/unit/test_warm_rain.py:121:        cond, q_sat = saturation_adjustment(T_, q_v_, p_, dt=300.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:61:    # Cloud and rain water at mid levels
tests/atmosphere/hydrostatic/unit/test_microphysics.py:154:        assert out.dT_dt.shape == (4, 10)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:175:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:176:        assert out.dT_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:179:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:181:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:186:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:187:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:191:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:204:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:206:        assert float(jnp.max(jnp.abs(out.dq_i_dt))) == 0.0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:207:        assert float(jnp.max(jnp.abs(out.dq_s_dt))) == 0.0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:208:        assert float(jnp.max(jnp.abs(out.dq_g_dt))) == 0.0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:214:            out = kessler_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:215:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:220:    def test_evaporation_enthalpy_balance(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:221:        """Evaporation cooling should balance vapor tendency latent energy."""
tests/atmosphere/hydrostatic/unit/test_microphysics.py:223:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:224:        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
tests/atmosphere/hydrostatic/unit/test_microphysics.py:236:        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:237:        assert out.dT_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:240:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:242:        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:247:        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:248:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:252:        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:268:        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:270:        out_sat = sundqvist_microphysics(T, q_sat, h, p_full, p_half, rho, dz, dt=10.0, config=config)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:271:        ratio = float(jnp.max(jnp.abs(out.dq_c_dt))) / float(jnp.max(jnp.abs(out_sat.dq_c_dt)) + 1e-20)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:278:            out = sundqvist_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:279:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:293:        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:294:        assert out.dT_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:295:        assert out.dN_c_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:296:        assert out.dN_r_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:298:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:300:        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:305:        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:306:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:310:        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:318:            out = seifert_beheng_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:319:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:324:    def test_evaporation_enthalpy_balance(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:325:        """Evaporation cooling should balance vapor tendency latent energy."""
tests/atmosphere/hydrostatic/unit/test_microphysics.py:327:        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:328:        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
tests/atmosphere/hydrostatic/unit/test_microphysics.py:340:        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:341:        assert out.dT_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:342:        assert out.dq_i_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:343:        assert out.dN_i_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:345:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:347:        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:352:        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:353:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:357:        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:371:        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:373:        assert float(jnp.max(jnp.abs(out.dq_i_dt))) < 1e-6
tests/atmosphere/hydrostatic/unit/test_microphysics.py:379:            out = morrison_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:380:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:385:    def test_evaporation_enthalpy_balance(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:386:        """Warm-rain evaporation cooling should close latent energy tendency."""
tests/atmosphere/hydrostatic/unit/test_microphysics.py:388:        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:389:        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
tests/atmosphere/hydrostatic/unit/test_microphysics.py:401:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:402:        assert out.dT_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:403:        assert out.dq_g_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:405:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:407:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:412:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:413:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:417:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:424:        # Boost cloud water and ice to promote riming
tests/atmosphere/hydrostatic/unit/test_microphysics.py:434:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:435:        assert float(jnp.max(jnp.abs(out.dq_g_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:441:            out = thompson_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:442:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:447:    def test_evaporation_enthalpy_balance(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:448:        """Warm-rain evaporation cooling should close latent energy tendency."""
tests/atmosphere/hydrostatic/unit/test_microphysics.py:450:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:451:        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
tests/atmosphere/hydrostatic/unit/test_microphysics.py:474:        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
tests/atmosphere/hydrostatic/unit/test_microphysics.py:476:        assert out.dT_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:479:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:483:        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
tests/atmosphere/hydrostatic/unit/test_microphysics.py:491:        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
tests/atmosphere/hydrostatic/unit/test_microphysics.py:494:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:500:        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
tests/atmosphere/hydrostatic/unit/test_microphysics.py:511:            out = ml_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0,
tests/atmosphere/hydrostatic/unit/test_microphysics.py:513:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:539:        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:541:        assert tend.dT_dt.data.shape == state.T.data.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:546:        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:548:        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) == 0.0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:553:        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:558:            return jnp.sum(tend.dT_dt.data ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:613:        physics_fn = make_microphysics_physics(config, "nonhydrostatic", dt=1.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:615:        assert tend.dtheta_prime_dt.data.shape == state.theta_prime.data.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:616:        assert tend.dtracers_dt.data.shape == state.tracers.data.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:624:        physics_fn = make_microphysics_physics(config, "nonhydrostatic", dt=1.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:626:        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:627:        assert jnp.all(jnp.isfinite(tend.dtracers_dt.data))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:637:            physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:643:            make_microphysics_physics(config, "hydrostatic", dt=300.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:648:            make_microphysics_physics(config, "invalid", dt=300.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:699:        physics_fn = make_microphysics_physics(micro_config, model_type="nonhydrostatic", dt=1.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:701:        assert tend.dtheta_prime_dt.data.shape == shape_3d
tests/atmosphere/hydrostatic/unit/test_microphysics.py:702:        assert tend.dtracers_dt.data.shape == (*shape_3d, 3)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:703:        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:711:    """Test checkpoint save/load roundtrip with q_c and q_r fields."""
tests/atmosphere/hydrostatic/unit/test_microphysics.py:735:    def test_roundtrip_with_hydrometeors(self, setup):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:752:    def test_roundtrip_without_hydrometeors(self, setup):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:765:    def test_config_microphysics_field_roundtrip(self, setup):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:788:    def test_config_json_roundtrip(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:807:                                   dt=600.0, config=config)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:808:        # Should have nonzero cloud water and rain tendencies
tests/atmosphere/hydrostatic/unit/test_microphysics.py:809:        assert float(jnp.max(jnp.abs(out.dq_c_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:810:        assert float(jnp.max(jnp.abs(out.dq_r_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:812:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:818:        dt = 600.0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:825:                                       dt=dt, config=config)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:826:            q_v = jnp.maximum(q_v + dt * out.dq_v_dt, 0.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:827:            q_c = jnp.maximum(q_c + dt * out.dq_c_dt, 0.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:828:            q_r = jnp.maximum(q_r + dt * out.dq_r_dt, 0.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:829:            T = T + dt * out.dT_dt

exec
/bin/zsh -lc "rg -n \"xfail|microphysics.*conservation|total water|enthalpy|morrison|thompson|kessler|sundqvist|seifert\" tests -g'*.py'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
tests/stress/test_phase1_sea_ice.py:176:        # Ice enthalpy: E = rho_ice * conc * h * (c_ice * T + L_f)
tests/unit/test_bechtold.py:520:@pytest.mark.xfail(
tests/unit/test_bechtold.py:525:        "Currently ~92% non-conservation residual; flagged as xfail so "
tests/unit/test_emanuel.py:257:@pytest.mark.xfail(
tests/unit/test_emanuel.py:261:        "residual; flagged xfail so any future kernel improvement that "
tests/unit/test_sea_ice_dynamics.py:780:    discarding excess enthalpy at the freezing clamp."""
tests/unit/test_physics_grid_adapters.py:237:        expected = {"kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"}
tests/unit/test_physics_grid_adapters.py:240:    def test_resolve_kessler(self):
tests/unit/test_physics_grid_adapters.py:241:        fn = resolve_kernel(MICROPHYSICS_REGISTRY, "kessler")
tests/unit/test_physics_grid_adapters.py:243:        assert fn.__name__ == "kessler_microphysics"
tests/unit/test_physics_grid_adapters.py:318:    def test_kessler_microphysics(self, cs_grid):
tests/unit/test_physics_grid_adapters.py:320:        config = _make_config(microphysics="kessler")
tests/validation/validation_differentiability_all.py:146:    @pytest.mark.xfail(reason="Pre-existing Field subscript issue in SW cdgrid")
tests/validation/validation_differentiability_all.py:177:    @pytest.mark.xfail(reason="Zero initial perturbation produces zero gradients")
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:402:        micro_config = MicrophysicsConfig(scheme="kessler", kessler=config)
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:419:    def test_kessler_tendencies_finite(self, grid, height_coord, terrain_metric):
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:428:        micro_config = MicrophysicsConfig(scheme="kessler", kessler=config)
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:147:                scheme="kessler", kessler=KesslerConfig(),
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:151:        kessler_fn = make_physics(cfg_phys, model_type="spectral_pe", dt=600.0)
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:164:            kess, _phys_out = kessler_fn(s, g, sc)
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:243:                scheme="kessler", kessler=KesslerConfig(),
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:247:        kessler_fn = make_physics(cfg_phys, model_type="spectral_pe", dt=600.0)
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:251:            kess, _ = kessler_fn(s, g, sc)
tests/unit/test_kain_fritsch.py:293:@pytest.mark.xfail(
tests/unit/test_kain_fritsch.py:297:        "residual; flagged xfail so any future kernel improvement that "
tests/unit/test_scale_mpi_layout.py:276:    @pytest.mark.xfail(
tests/unit/test_zhang_mcfarlane.py:401:@pytest.mark.xfail(
tests/unit/test_zhang_mcfarlane.py:406:        "residual; flagged xfail so any future kernel improvement that "
tests/unit/test_physics_microphysics.py:3:Tests total water conservation, temperature-moisture coupling (Clausius-
tests/unit/test_physics_microphysics.py:15:from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
tests/unit/test_physics_microphysics.py:16:from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
tests/unit/test_physics_microphysics.py:17:from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
tests/unit/test_physics_microphysics.py:18:from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
tests/unit/test_physics_microphysics.py:19:from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
tests/unit/test_physics_microphysics.py:77:    if name == "kessler":
tests/unit/test_physics_microphysics.py:78:        return kessler_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
tests/unit/test_physics_microphysics.py:80:    elif name == "sundqvist":
tests/unit/test_physics_microphysics.py:81:        return sundqvist_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
tests/unit/test_physics_microphysics.py:83:    elif name == "seifert_beheng":
tests/unit/test_physics_microphysics.py:84:        return seifert_beheng_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
tests/unit/test_physics_microphysics.py:86:    elif name == "morrison":
tests/unit/test_physics_microphysics.py:87:        return morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
tests/unit/test_physics_microphysics.py:89:    elif name == "thompson":
tests/unit/test_physics_microphysics.py:90:        return thompson_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
tests/unit/test_physics_microphysics.py:96:ALL_SCHEMES = ["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"]
tests/unit/test_physics_microphysics.py:143:@pytest.mark.parametrize("scheme", ["kessler", "sundqvist"])
tests/unit/test_physics_microphysics.py:177:@pytest.mark.parametrize("scheme", ["morrison", "thompson"])
tests/unit/test_physics_microphysics.py:198:def test_kessler_autoconversion_sensitivity():
tests/unit/test_physics_microphysics.py:205:    out_lo = kessler_microphysics(T_lo, q_v_lo, hydro_lo, p_full, p_half, rho, dz,
tests/unit/test_physics_microphysics.py:211:    out_hi = kessler_microphysics(T_lo, q_v_lo, hydro_hi, p_full, p_half, rho, dz,
tests/unit/test_physics_combined.py:55:    """All 5 physics modules active: gray + sbm + smagorinsky + kessler + rayleigh."""
tests/unit/test_physics_combined.py:61:        microphysics=MicrophysicsConfig(scheme="kessler"),
tests/unit/test_ml_physics_parameterization.py:12:    apply_predicted_sundqvist_rain_survival_fraction,
tests/unit/test_ml_physics_parameterization.py:16:from legoesm.atmosphere.physics.microphysics.sundqvist import (
tests/unit/test_ml_physics_parameterization.py:17:    diagnose_sundqvist_process_rates,
tests/unit/test_ml_physics_parameterization.py:18:    sundqvist_microphysics,
tests/unit/test_ml_physics_parameterization.py:61:def test_kessler_feature_vector_shape():
tests/unit/test_ml_physics_parameterization.py:79:        microphysics_scheme="kessler",
tests/unit/test_ml_physics_parameterization.py:84:def test_sundqvist_feature_vector_shape():
tests/unit/test_ml_physics_parameterization.py:102:        microphysics_scheme="sundqvist",
tests/unit/test_ml_physics_parameterization.py:121:def test_kessler_target_vector_shape_and_unpack():
tests/unit/test_ml_physics_parameterization.py:131:        microphysics_scheme="kessler",
tests/unit/test_ml_physics_parameterization.py:137:        microphysics_scheme="kessler",
tests/unit/test_ml_physics_parameterization.py:145:def test_sundqvist_target_vector_shape_and_unpack():
tests/unit/test_ml_physics_parameterization.py:152:        microphysics_scheme="sundqvist",
tests/unit/test_ml_physics_parameterization.py:158:        microphysics_scheme="sundqvist",
tests/unit/test_ml_physics_parameterization.py:195:def test_kessler_checkpoint_and_stats_roundtrip():
tests/unit/test_ml_physics_parameterization.py:201:        microphysics_scheme="kessler",
tests/unit/test_ml_physics_parameterization.py:228:def test_sundqvist_checkpoint_and_stats_roundtrip():
tests/unit/test_ml_physics_parameterization.py:234:        microphysics_scheme="sundqvist",
tests/unit/test_ml_physics_parameterization.py:261:def test_sundqvist_training_pipeline_evaluates_microphysics_targets():
tests/unit/test_ml_physics_parameterization.py:299:        microphysics_scheme="sundqvist",
tests/unit/test_ml_physics_parameterization.py:322:def test_sundqvist_rain_survival_rebuild_matches_physical_scheme():
tests/unit/test_ml_physics_parameterization.py:342:    physical = sundqvist_microphysics(
tests/unit/test_ml_physics_parameterization.py:353:    rates = diagnose_sundqvist_process_rates(
tests/unit/test_ml_physics_parameterization.py:370:    rebuilt = apply_predicted_sundqvist_rain_survival_fraction(
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:292:# Energy conservation (KE + enthalpy)
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:298:        """Compute area-weighted column-integrated KE + enthalpy.
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:305:        enthalpy = constants.c_pd * state.T
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:308:        integrand = (ke + enthalpy) * state.p_s[:, :, None] * dsigma / constants.g
tests/unit/test_config_validation.py:17:        cfg = ExperimentConfig(radiation="gray", microphysics="kessler")
tests/unit/test_physics_smoke.py:119:@pytest.mark.parametrize("scheme", ["kessler", "sundqvist", "seifert_beheng",
tests/unit/test_physics_smoke.py:120:                                     "morrison", "thompson"])
tests/unit/test_physics_smoke.py:214:        microphysics=MicrophysicsConfig(scheme="kessler"),
tests/unit/test_diff_atmosphere_physics.py:11:  2e) Microphysics (kessler, sundqvist)
tests/unit/test_diff_atmosphere_physics.py:210:    @pytest.mark.parametrize("scheme", ["kessler", "sundqvist"])
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:125:        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:143:        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:202:        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:268:        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:282:        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:295:        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:360:        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:408:                scheme="kessler", kessler=KesslerConfig(),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:442:                scheme="kessler", kessler=KesslerConfig(),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:520:                scheme="kessler", kessler=KesslerConfig(),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:530:                scheme="kessler", kessler=KesslerConfig(),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:576:            scheme="sundqvist", sundqvist=SundqvistConfig(),
tests/unit/test_ml_physics_workflow.py:29:def test_kessler_workflow_uses_rrtmgp_base_config():
tests/unit/test_ml_physics_workflow.py:31:    args = parser.parse_args(["--microphysics", "kessler"])
tests/unit/test_ml_physics_workflow.py:33:    assert cfg.microphysics == "kessler"
tests/unit/test_ml_physics_workflow.py:39:def test_sundqvist_workflow_uses_rrtmgp_base_config():
tests/unit/test_ml_physics_workflow.py:41:    args = parser.parse_args(["--microphysics", "sundqvist"])
tests/unit/test_ml_physics_workflow.py:43:    assert cfg.microphysics == "sundqvist"
tests/unit/test_ml_physics_workflow.py:45:    assert cfg.cloud_scheme == "sundqvist"
tests/unit/test_tiedtke.py:254:@pytest.mark.xfail(
tests/unit/test_tiedtke.py:259:        "residual on this CAPE-positive fixture; flagged xfail so any "
tests/unit/test_equation_fixes.py:166:        config = MicrophysicsConfig(scheme="kessler")
tests/unit/test_equation_fixes.py:185:    def test_kessler_condensation_depends_on_dt(self):
tests/unit/test_equation_fixes.py:187:        from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
tests/unit/test_equation_fixes.py:190:        out1 = kessler_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt=60.0)
tests/unit/test_equation_fixes.py:191:        out2 = kessler_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt=600.0)
tests/unit/test_equation_fixes.py:198:    def test_kessler_condensation_heating_scales_correctly(self):
tests/unit/test_equation_fixes.py:200:        from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
tests/unit/test_equation_fixes.py:220:        out = kessler_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt)
tests/unit/test_equation_fixes.py:235:        ("sundqvist", "sundqvist_microphysics"),
tests/unit/test_equation_fixes.py:236:        ("seifert_beheng", "seifert_beheng_microphysics"),
tests/unit/test_equation_fixes.py:237:        ("morrison", "morrison_microphysics"),
tests/unit/test_equation_fixes.py:238:        ("thompson", "thompson_microphysics"),
tests/unit/test_equation_fixes.py:254:        from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
tests/unit/test_equation_fixes.py:257:        out = kessler_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt)
tests/unit/test_physics_convection.py:438:#   * SBM enforces a Newton enthalpy correction so c_pd*int(dT) + Lv*int(dq_v)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:30:from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
tests/atmosphere/hydrostatic/unit/test_microphysics.py:31:from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
tests/atmosphere/hydrostatic/unit/test_microphysics.py:32:from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
tests/atmosphere/hydrostatic/unit/test_microphysics.py:33:from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
tests/atmosphere/hydrostatic/unit/test_microphysics.py:34:from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
tests/atmosphere/hydrostatic/unit/test_microphysics.py:175:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:181:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:186:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:191:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:204:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:214:            out = kessler_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:220:    def test_evaporation_enthalpy_balance(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:223:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:236:        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:242:        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:247:        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:252:        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:268:        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:270:        out_sat = sundqvist_microphysics(T, q_sat, h, p_full, p_half, rho, dz, dt=10.0, config=config)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:278:            out = sundqvist_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:293:        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:300:        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:305:        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:310:        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:318:            out = seifert_beheng_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:324:    def test_evaporation_enthalpy_balance(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:327:        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:340:        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:347:        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:352:        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:357:        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:371:        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:379:            out = morrison_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:385:    def test_evaporation_enthalpy_balance(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:388:        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:401:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:407:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:412:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:417:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:434:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:441:            out = thompson_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:447:    def test_evaporation_enthalpy_balance(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:450:        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:538:        config = MicrophysicsConfig(scheme="kessler")
tests/atmosphere/hydrostatic/unit/test_microphysics.py:552:        config = MicrophysicsConfig(scheme="kessler")
tests/atmosphere/hydrostatic/unit/test_microphysics.py:612:        config = MicrophysicsConfig(scheme="kessler")
tests/atmosphere/hydrostatic/unit/test_microphysics.py:623:        config = MicrophysicsConfig(scheme="kessler")
tests/atmosphere/hydrostatic/unit/test_microphysics.py:634:        for scheme in ["kessler", "sundqvist", "seifert_beheng",
tests/atmosphere/hydrostatic/unit/test_microphysics.py:635:                       "morrison", "thompson", "ml_emulator", "none"]:
tests/atmosphere/hydrostatic/unit/test_microphysics.py:646:        config = MicrophysicsConfig(scheme="kessler")
tests/atmosphere/hydrostatic/unit/test_microphysics.py:653:    def test_kessler_nonhydrostatic_tendencies(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:698:        micro_config = MicrophysicsConfig(scheme="kessler", kessler=config)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:731:            resolution=n, nlev=nlev, microphysics="kessler",
tests/atmosphere/hydrostatic/unit/test_microphysics.py:772:        assert restored_config.microphysics == "kessler"
tests/atmosphere/hydrostatic/unit/test_microphysics.py:783:    def test_kessler_microphysics(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:785:        config = AMIPExperimentConfig(microphysics="kessler")
tests/atmosphere/hydrostatic/unit/test_microphysics.py:786:        assert config.microphysics == "kessler"
tests/atmosphere/hydrostatic/unit/test_microphysics.py:792:        config = AMIPExperimentConfig(microphysics="sundqvist")
tests/atmosphere/hydrostatic/unit/test_microphysics.py:794:        assert d["microphysics"] == "sundqvist"
tests/atmosphere/hydrostatic/unit/test_microphysics.py:796:        assert restored.microphysics == "sundqvist"
tests/atmosphere/hydrostatic/unit/test_microphysics.py:802:    def test_kessler_in_column_physics(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:806:        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz,
tests/atmosphere/hydrostatic/unit/test_microphysics.py:824:            out = kessler_microphysics(T, q_v, h_step, p_full, p_half, rho, dz,
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:410:        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:423:        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:445:        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:461:        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:215:    def test_kessler(self):
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:217:        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="kessler"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:219:        _check_tendencies(tend, "microphysics/kessler")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:221:    def test_sundqvist(self):
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:223:        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="sundqvist"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:225:        _check_tendencies(tend, "microphysics/sundqvist")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:227:    def test_seifert_beheng(self):
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:229:        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="seifert_beheng"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:231:        _check_tendencies(tend, "microphysics/seifert_beheng")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:233:    def test_morrison(self):
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:235:        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="morrison"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:237:        _check_tendencies(tend, "microphysics/morrison")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:239:    def test_thompson(self):
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:241:        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="thompson"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:243:        _check_tendencies(tend, "microphysics/thompson")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:318:            microphysics=MicrophysicsConfig(scheme="kessler"),
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:333:            microphysics=MicrophysicsConfig(scheme="sundqvist"),
tests/atmosphere/hydrostatic/unit/test_radiation.py:34:    sundqvist_cloud_fraction,
tests/atmosphere/hydrostatic/unit/test_radiation.py:914:    def test_sundqvist_zero_below_rh_crit(self):
tests/atmosphere/hydrostatic/unit/test_radiation.py:916:        config = CloudConfig(scheme="sundqvist", rh_crit=0.7)
tests/atmosphere/hydrostatic/unit/test_radiation.py:918:        cf = sundqvist_cloud_fraction(RH, config)
tests/atmosphere/hydrostatic/unit/test_radiation.py:921:    def test_sundqvist_one_at_saturation(self):
tests/atmosphere/hydrostatic/unit/test_radiation.py:923:        config = CloudConfig(scheme="sundqvist", rh_crit=0.7)
tests/atmosphere/hydrostatic/unit/test_radiation.py:925:        cf = sundqvist_cloud_fraction(RH, config)
tests/atmosphere/hydrostatic/unit/test_radiation.py:928:    def test_sundqvist_linear_ramp(self):
tests/atmosphere/hydrostatic/unit/test_radiation.py:930:        config = CloudConfig(scheme="sundqvist", rh_crit=0.7)
tests/atmosphere/hydrostatic/unit/test_radiation.py:932:        cf = sundqvist_cloud_fraction(RH_mid, config)
tests/atmosphere/hydrostatic/unit/test_radiation.py:960:        config = CloudConfig(scheme="sundqvist")
tests/atmosphere/hydrostatic/unit/test_radiation.py:978:        config = CloudConfig(scheme="sundqvist", rh_crit=0.7)
tests/atmosphere/hydrostatic/unit/test_radiation.py:995:        config = CloudConfig(scheme="sundqvist", rh_crit=0.5)
tests/atmosphere/hydrostatic/unit/test_radiation.py:1083:    def test_integration_with_sundqvist_clouds(self):
tests/atmosphere/hydrostatic/unit/test_radiation.py:1095:            cloud_scheme="sundqvist",
tests/atmosphere/hydrostatic/unit/test_convection.py:5:- SBM: shapes, enthalpy conservation, precipitation, trigger, differentiability
tests/atmosphere/hydrostatic/unit/test_convection.py:218:    def test_enthalpy_conservation(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:219:        """Column enthalpy tendency should be approximately conserved.
tests/atmosphere/hydrostatic/unit/test_convection.py:221:        The linearized enthalpy correction ensures approximate conservation.
tests/atmosphere/hydrostatic/unit/test_convection.py:223:        relative to the total enthalpy change from temperature alone.
tests/atmosphere/hydrostatic/unit/test_convection.py:231:        enthalpy_tend = jnp.sum(
tests/atmosphere/hydrostatic/unit/test_convection.py:240:        # Relative conservation: enthalpy residual should be small fraction of total
tests/atmosphere/hydrostatic/unit/test_convection.py:241:        relative_error = jnp.abs(enthalpy_tend) / jnp.clip(T_scale, 1.0, None)

codex
The tests do not appear to mark microphysics conservation checks `xfail`; the main microphysics test file advertises total-water conservation in its docstring, but the actual checks are mostly finiteness, sign, shapes, and a few enthalpy balances. I’m now probing AD at zero hydrometeor mass because several rates use fractional powers after clipping, which is a common NaN-gradient source in JAX.
exec
/bin/zsh -lc "python - <<'PY'
import jax, jax.numpy as jnp

for name, f in [
    ('pow0525', lambda x: jnp.clip(x,0.0)**0.525),
    ('pow0875', lambda x: jnp.clip(x,0.0)**0.875),
    ('flux_q_qb', lambda x: (jnp.clip(x,0.0)**0.5)*jnp.clip(x,0.0)),
    ('fall_qb_only', lambda x: jnp.clip(x,0.0)**0.5),
    ('sed_1lev', lambda x: ((0.0) - (jnp.clip(x,0.0)**0.5)*jnp.clip(x,0.0)*1.0)/(1.0*500.0)),
]
    y = f(jnp.array(0.0))
    g = jax.grad(f)(jnp.array(0.0))
    print(name, 'y=', y, 'grad=', g)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 127 in 0ms:
zsh:1: command not found: python

exec
/bin/zsh -lc 'which python3 || true' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
/opt/homebrew/bin/python3

exec
/bin/zsh -lc 'ls -la' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
total 4824
drwxr-xr-x@ 31 pierregentine  staff      992 May  1 13:25 .
drwxr-xr-x@ 12 pierregentine  staff      384 Apr 15 11:09 ..
-rw-r--r--@  1 pierregentine  staff    22532 May  1 12:51 .DS_Store
drwxr-xr-x@  7 pierregentine  staff      224 Apr 29 09:41 .claude
drwxr-xr-x@ 17 pierregentine  staff      544 May  1 12:04 .git
drwxr-xr-x@  3 pierregentine  staff       96 Mar  2 12:12 .github
-rw-r--r--@  1 pierregentine  staff      557 Apr 30 09:06 .gitignore
drwxr-xr-x@  3 pierregentine  staff       96 Mar 17 09:57 .mplconfig
drwxr-xr-x@  3 pierregentine  staff       96 May  1 13:25 .physics-validator
drwxr-xr-x@  7 pierregentine  staff      224 Mar  3 07:17 .pytest_cache
drwxr-xr-x@  5 pierregentine  staff      160 Apr 16 14:13 .ralph
-rw-r--r--@  1 pierregentine  staff     1874 Apr  2 08:36 .ralphrc
drwxr-xr-x@  5 pierregentine  staff      160 Mar  8 21:55 .ruff_cache
-rw-r--r--@  1 pierregentine  staff      485 Mar 15 18:50 .slopbuster.yaml
drwxr-xr-x@  9 pierregentine  staff      288 Mar  2 22:21 .venv
-rw-r--r--@  1 pierregentine  staff    24903 May  1 09:14 CLAUDE.md
-rw-r--r--@  1 pierregentine  staff    24031 Mar 21 20:14 FIXME_test_audit_prompt.md
-rw-r--r--@  1 pierregentine  staff    38314 Mar  9 16:47 NEXT_STEPS.md
-rw-r--r--@  1 pierregentine  staff     8871 Apr 28 09:12 README.md
-rw-r--r--@  1 pierregentine  staff   148531 Mar 30 18:51 SPECIFICATION.md
drwxr-xr-x@  4 pierregentine  staff      128 Mar  8 21:55 config
drwxr-xr-x@  3 pierregentine  staff       96 Apr 28 09:12 diagnostics
drwxr-xr-x@ 38 pierregentine  staff     1216 May  1 08:53 docs
drwxr-xr-x@  9 pierregentine  staff      288 Apr  5 09:45 evaluations
-rw-------@  1 pierregentine  staff  2166787 Mar  2 11:02 legoESM.png
drwxr-xr-x@  4 pierregentine  staff      128 Apr  1 18:36 output
-rw-r--r--@  1 pierregentine  staff     2181 Apr  6 18:38 pyproject.toml
drwxr-xr-x@ 13 pierregentine  staff      416 May  1 10:05 results
drwxr-xr-x@ 80 pierregentine  staff     2560 May  1 10:19 scripts
drwxr-xr-x@  4 pierregentine  staff      128 Feb 28 18:30 src
drwxr-xr-x@ 23 pierregentine  staff      736 Apr 28 09:24 tests

exec
/bin/zsh -lc 'ls -la .venv/bin 2>/dev/null | head -n 50' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
total 45704
drwxr-xr-x@ 47 pierregentine  staff      1504 Apr 29 22:33 .
drwxr-xr-x@  9 pierregentine  staff       288 Mar  2 22:21 ..
-rw-r--r--@  1 pierregentine  staff      9031 Feb  3 16:32 Activate.ps1
-rw-r--r--@  1 pierregentine  staff      2318 Mar 30 16:32 activate
-rw-r--r--@  1 pierregentine  staff       994 Mar 30 16:32 activate.csh
-rw-r--r--@  1 pierregentine  staff      2265 Mar 30 16:32 activate.fish
-rwxr-xr-x@  1 pierregentine  staff       215 Feb 26 15:45 cartopy_feature_download
-rwxr-xr-x@  1 pierregentine  staff       195 Feb 26 15:45 differesm
-rwxr-xr-x@  1 pierregentine  staff       217 Feb 26 15:45 dmypy
-rwxr-xr-x@  1 pierregentine  staff       199 Feb 26 15:45 f2py
-rwxr-xr-x@  1 pierregentine  staff       200 Feb 26 15:45 fonttools
-rwxr-xr-x@  1 pierregentine  staff       312 Mar 28 19:03 google-oauthlib-tool
-rwxr-xr-x@  1 pierregentine  staff       194 Feb 26 15:45 identify-cli
-rwxr-xr-x@  1 pierregentine  staff       187 Feb 28 21:56 legoesm
-rwxr-xr-x@  1 pierregentine  staff       297 Mar 23 06:40 markdown_py
-rwxr-xr-x@  1 pierregentine  staff       213 Feb 26 15:45 mypy
-rwxr-xr-x@  1 pierregentine  staff       196 Feb 26 15:45 mypyc
-rwxr-xr-x@  1 pierregentine  staff       197 Mar  2 14:06 nc3tonc4
-rwxr-xr-x@  1 pierregentine  staff       197 Mar  2 14:06 nc4tonc3
-rwxr-xr-x@  1 pierregentine  staff       193 Mar  2 14:06 ncinfo
-rwxr-xr-x@  1 pierregentine  staff       189 Feb 26 15:45 nodeenv
-rwxr-xr-x@  1 pierregentine  staff       312 Mar 28 19:03 normalizer
-rwxr-xr-x@  1 pierregentine  staff       199 Feb 26 15:45 numpy-config
-rwxr-xr-x@  1 pierregentine  staff       204 Feb 26 15:44 pip
-rwxr-xr-x@  1 pierregentine  staff       204 Feb 26 15:44 pip3
-rwxr-xr-x@  1 pierregentine  staff       204 Feb 26 15:44 pip3.14
-rwxr-xr-x@  1 pierregentine  staff       197 Feb 26 15:45 pre-commit
-rwxr-xr-x@  1 pierregentine  staff       204 Feb 26 15:45 py.test
-rwxr-xr-x@  1 pierregentine  staff       294 Apr 29 22:33 pyflakes
-rwxr-xr-x@  1 pierregentine  staff       197 Feb 26 15:45 pyftmerge
-rwxr-xr-x@  1 pierregentine  staff       198 Feb 26 15:45 pyftsubset
-rwxr-xr-x@  1 pierregentine  staff       198 Feb 26 15:45 pygmentize
-rwxr-xr-x@  1 pierregentine  staff       197 Feb 26 15:45 pyproj
-rwxr-xr-x@  1 pierregentine  staff       204 Feb 26 15:45 pytest
lrwxr-xr-x@  1 pierregentine  staff        10 Mar  8 22:24 python -> python3.14
lrwxr-xr-x@  1 pierregentine  staff        10 Mar  8 22:24 python3 -> python3.14
lrwxr-xr-x@  1 pierregentine  staff        44 Mar  8 22:24 python3.14 -> /opt/homebrew/opt/python@3.14/bin/python3.14
-rwxr-xr-x@  1 pierregentine  staff  23224384 Feb 26 15:45 ruff
-rwxr-xr-x@  1 pierregentine  staff       194 Feb 26 15:45 stubgen
-rwxr-xr-x@  1 pierregentine  staff       195 Feb 26 15:45 stubtest
-rwxr-xr-x@  1 pierregentine  staff       195 Feb 26 15:45 ttx
-rwxr-xr-x@  1 pierregentine  staff       221 Feb 26 15:45 virtualenv
-rwxr-xr-x@  1 pierregentine  staff       188 Mar  5 18:31 vulture
-rwxr-xr-x@  1 pierregentine  staff       301 Mar 23 06:40 weasyprint
-rwxr-xr-x@  1 pierregentine  staff       293 Mar 30 16:32 wheel
-rwxr-xr-x@  1 pierregentine  staff       193 Feb 26 15:45 zarr
lrwxr-xr-x@  1 pierregentine  staff        10 Mar  8 22:24 𝜋thon -> python3.14

exec
/bin/zsh -lc ".venv/bin/python - <<'PY'
import jax, jax.numpy as jnp

for name, f in [
    ('pow0525', lambda x: jnp.clip(x,0.0)**0.525),
    ('pow0875', lambda x: jnp.clip(x,0.0)**0.875),
    ('flux_q_qb', lambda x: (jnp.clip(x,0.0)**0.5)*jnp.clip(x,0.0)),
    ('fall_qb_only', lambda x: jnp.clip(x,0.0)**0.5),
    ('sed_1lev', lambda x: ((0.0) - (jnp.clip(x,0.0)**0.5)*jnp.clip(x,0.0)*1.0)/(1.0*500.0)),
]
    y = f(jnp.array(0.0))
    g = jax.grad(f)(jnp.array(0.0))
    print(name, 'y=', y, 'grad=', g)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 0ms:
  File "<stdin>", line 9
    ]
     ^
SyntaxError: expected ':'

exec
/bin/zsh -lc ".venv/bin/python - <<'PY'
import jax, jax.numpy as jnp

cases = [
    ('pow0525', lambda x: jnp.clip(x,0.0)**0.525),
    ('pow0875', lambda x: jnp.clip(x,0.0)**0.875),
    ('flux_q_qb', lambda x: (jnp.clip(x,0.0)**0.5)*jnp.clip(x,0.0)),
    ('fall_qb_only', lambda x: jnp.clip(x,0.0)**0.5),
    ('sed_1lev', lambda x: ((0.0) - (jnp.clip(x,0.0)**0.5)*jnp.clip(x,0.0)*1.0)/(1.0*500.0)),
]
for name, f in cases:
    y = f(jnp.array(0.0))
    g = jax.grad(f)(jnp.array(0.0))
    print(name, 'y=', y, 'grad=', g)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited -1 in 643ms:
WARNING:2026-05-01 13:53:46,165:jax._src.xla_bridge:905: Platform 'METAL' is experimental and not all JAX functionality may be correctly supported!
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1777636426.166568 26349370 mps_client.cc:510] WARNING: JAX Apple GPU support is experimental and not all JAX functionality is correctly supported!
F0000 00:00:1777636426.194225 26349370 mps_client.cc:494] Check failed: platform->VisibleDeviceCount() == 1 (0 vs. 1) 
No supported GPU was found.
*** Check failure stack trace: ***
    @        0x13207f0c4  absl::lts_20230802::log_internal::LogMessage::SendToLog()
    @        0x13207eb34  absl::lts_20230802::log_internal::LogMessage::Flush()
    @        0x13207f3e4  absl::lts_20230802::log_internal::LogMessageFatal::~LogMessageFatal()
    @        0x13207f3fc  absl::lts_20230802::log_internal::LogMessageFatal::~LogMessageFatal()
    @        0x12d601a10  xla::InitMetalPlugin()
    @        0x12d601dc4  xla::GetMetalClient()
    @        0x12d5fc288  pjrt::metal_plugin::PJRT_Client_Create()
    @        0x119179050  xla::WrapClientAroundCApi()
    @        0x119178dcc  xla::GetCApiClient()
    @        0x1190a28f8  nanobind::detail::func_create<>()::{lambda()#1}::__invoke()
    @        0x115ce0a48  nanobind::detail::nb_func_vectorcall_complex()
    @        0x10325a690  PyObject_Vectorcall
    @        0x10338f538  _PyEval_EvalFrameDefault
    @        0x10338985c  _PyEval_Vector
    @        0x10325a554  _PyVectorcall_Call
    @        0x1034a4684  partial_call
    @        0x103259b78  _PyObject_MakeTpCall
    @        0x10325a690  PyObject_Vectorcall
    @        0x10338cdd4  _PyEval_EvalFrameDefault
    @        0x10338985c  _PyEval_Vector
    @        0x1034a7590  infinite_lru_cache_wrapper
    @        0x103259b78  _PyObject_MakeTpCall
    @        0x10338f538  _PyEval_EvalFrameDefault
    @        0x10338985c  _PyEval_Vector
    @        0x1034a7590  infinite_lru_cache_wrapper
    @        0x103259b78  _PyObject_MakeTpCall
    @        0x10338f538  _PyEval_EvalFrameDefault
    @        0x10338985c  _PyEval_Vector
    @        0x10325a690  PyObject_Vectorcall
    @        0x115cd8dfc  nanobind::detail::obj_vectorcall()
    @        0x11b974178  nanobind::detail::api<>::operator()<>()
    @        0x11b972f80  jax::WeakrefLRUCache::Call()
    @        0x11b976250  nanobind::detail::func_create<>()::{lambda()#1}::__invoke()
    @        0x115ce0a48  nanobind::detail::nb_func_vectorcall_complex()
    @        0x103259a1c  _PyObject_VectorcallDictTstate
    @        0x1032fbf30  call_method
    @        0x103259b78  _PyObject_MakeTpCall
    @        0x1033934fc  _PyEval_EvalFrameDefault
    @        0x10338985c  _PyEval_Vector
    @        0x10325a690  PyObject_Vectorcall
    @        0x1190f8124  jax::(anonymous namespace)::PjitFunction::Call()
    @        0x1190f6a28  PjitFunction_tp_vectorcall
    @        0x10338fc24  _PyEval_EvalFrameDefault
    @        0x10338985c  _PyEval_Vector
    @        0x1034a583c  partial_vectorcall
    @        0x10325a554  _PyVectorcall_Call
    @        0x10338fc24  _PyEval_EvalFrameDefault
    @        0x10338985c  _PyEval_Vector
    @        0x10325d56c  method_vectorcall
    @        0x10325a554  _PyVectorcall_Call
    @        0x10338fc24  _PyEval_EvalFrameDefault
    @        0x103389664  PyEval_EvalCode
    @        0x103401324  run_mod
    @        0x1033ffc90  pyrun_file
    @        0x1033ff214  _PyRun_SimpleFileObject
    @        0x1033fee78  _PyRun_AnyFileObject
    @        0x1033ff430  PyRun_AnyFileExFlags
    @        0x10342d3dc  pymain_run_stdin
    @        0x10342cb60  Py_RunMain
    @        0x10342cdcc  pymain_main
    @        0x10342ce68  Py_BytesMain

exec
/bin/zsh -lc "JAX_PLATFORMS=cpu .venv/bin/python - <<'PY'
import jax, jax.numpy as jnp

cases = [
    ('pow0525', lambda x: jnp.clip(x,0.0)**0.525),
    ('pow0875', lambda x: jnp.clip(x,0.0)**0.875),
    ('flux_q_qb', lambda x: (jnp.clip(x,0.0)**0.5)*jnp.clip(x,0.0)),
    ('fall_qb_only', lambda x: jnp.clip(x,0.0)**0.5),
    ('sed_1lev', lambda x: ((0.0) - (jnp.clip(x,0.0)**0.5)*jnp.clip(x,0.0)*1.0)/(1.0*500.0)),
]
for name, f in cases:
    y = f(jnp.array(0.0))
    g = jax.grad(f)(jnp.array(0.0))
    print(name, 'y=', y, 'grad=', g)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 769ms:
pow0525 y= 0.0 grad= inf
pow0875 y= 0.0 grad= inf
flux_q_qb y= 0.0 grad= nan
fall_qb_only y= 0.0 grad= inf
sed_1lev y= 0.0 grad= nan

codex
A concrete conservation issue is emerging in `sundqvist.py`: the scan computes a surface precipitation flux, but the returned `dq_r_dt` never removes that fallout from the column, so column total water tendency sums to zero while `precipitation` is positive. I’m doing one more pass for unit consistency in number moments and for donor-limited process rates.
exec
/bin/zsh -lc "JAX_PLATFORMS=cpu .venv/bin/python - <<'PY'
import jax.numpy as jnp
from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics

ncol,nlev=1,5
T=jnp.full((ncol,nlev),280.0)
p_half=jnp.linspace(2e4,1e5,nlev+1)[None,:]
p_full=0.5*(p_half[:,:-1]+p_half[:,1:])
rho=p_full/(constants.R_d*T)
dz=jnp.full((ncol,nlev),500.0)
q_sat=saturation_mixing_ratio(T,p_full)
q_v=1.2*q_sat
h=HydrometeorState(q_c=jnp.full((ncol,nlev),1e-3),q_r=jnp.full((ncol,nlev),1e-4),q_i=jnp.zeros((ncol,nlev)),q_s=jnp.zeros((ncol,nlev)),q_g=jnp.zeros((ncol,nlev)),N_c=jnp.full((ncol,nlev),1e8),N_r=jnp.full((ncol,nlev),1e4),N_i=jnp.full((ncol,nlev),1e3))
for name,fn in [('kessler',kessler_microphysics),('sundqvist',sundqvist_microphysics),('seifert',seifert_beheng_microphysics),('morrison',morrison_microphysics),('thompson',thompson_microphysics)]:
    out=fn(T,q_v,h,p_full,p_half,rho,dz,300.0)
    total=out.dq_v_dt+out.dq_c_dt+out.dq_r_dt+out.dq_i_dt+out.dq_s_dt+out.dq_g_dt
    col=jnp.sum(total*rho*dz,axis=1)
    print(name, 'col_dW=', float(col[0]), 'P=', float(out.precipitation[0]), 'resid col+P=', float((col+out.precipitation)[0]))
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 2780ms:
kessler col_dW= -0.0005723244976252317 P= 0.0005723243812099099 resid col+P= -1.1641532182693481e-10
sundqvist col_dW= 5.035523109597762e-10 P= 0.009688454680144787 resid col+P= 0.009688455611467361
seifert col_dW= -0.00014880743401590735 P= 0.0001488043344579637 resid col+P= -3.0995579436421394e-09
morrison col_dW= -0.00014880533854011446 P= 0.0001488043344579637 resid col+P= -1.0040821507573128e-09
thompson col_dW= -0.00014868692960590124 P= 0.0001488043344579637 resid col+P= 1.1740485206246376e-07

exec
/bin/zsh -lc "sed -n '150,230p' tests/unit/test_equation_fixes.py | nl -ba -v 150" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "sed -n '230,280p' tests/unit/test_equation_fixes.py | nl -ba -v 230" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   230	        # Should not be dt-independent (i.e. max_expected * dt >> max_actual is wrong)
   231	        assert max_actual > max_expected * 0.1, \
   232	            f"Heating {max_actual} << expected {max_expected}"
   233	
   234	    @pytest.mark.parametrize("scheme_name,scheme_fn", [
   235	        ("sundqvist", "sundqvist_microphysics"),
   236	        ("seifert_beheng", "seifert_beheng_microphysics"),
   237	        ("morrison", "morrison_microphysics"),
   238	        ("thompson", "thompson_microphysics"),
   239	    ])
   240	    def test_all_schemes_condensation_depends_on_dt(self, scheme_name, scheme_fn):
   241	        """All backends' condensation component should depend on dt."""
   242	        import importlib
   243	        mod = importlib.import_module(f"legoesm.atmosphere.physics.microphysics.{scheme_name}")
   244	        fn = getattr(mod, scheme_fn)
   245	        T, q_v, hydrometeors, p_full, p_half, rho, dz = _make_warm_micro_columns()
   246	        out1 = fn(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt=60.0)
   247	        out2 = fn(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt=600.0)
   248	
   249	        diff = jnp.abs(out1.dT_dt - out2.dT_dt).sum()
   250	        assert diff > 0, f"{scheme_name}: dT_dt should depend on dt"
   251	
   252	    def test_precipitation_units_kg_m2_s(self):
   253	        """Precipitation should be in kg/m^2/s (O(1e-5) to O(1e-2))."""
   254	        from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
   255	        dt = 300.0
   256	        T, q_v, hydrometeors, p_full, p_half, rho, dz = _make_warm_micro_columns(dt=dt)
   257	        out = kessler_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt)
   258	        max_precip = jnp.max(out.precipitation)
   259	        # Reasonable precipitation is 0 to ~100 mm/hr = ~0.028 kg/m²/s
   260	        assert max_precip < 1.0, f"Precipitation {max_precip} kg/m²/s unreasonably large"
   261	
   262	
   263	# ======================================================================
   264	# Issues 4-5: Mass-flux and EDMF precipitation dimensions
   265	# ======================================================================
   266	
   267	class TestIssue4_MassFluxPrecipitation:
   268	
   269	    def test_dq_c_conv_dimensional_consistency(self):
   270	        """Convective cloud-water source from mass-flux should be kg/kg/s.
   271	
   272	        Post-Option-C: mass_flux now emits a 3D ``dq_c_conv_dt``
   273	        (kg/kg/s) instead of a scalar surface ``precipitation``
   274	        (kg/m²/s); microphysics owns the surface-flux diagnostic. A
   275	        sane parameterization keeps per-level condensation rates well
   276	        under 1e-3 kg/kg/s — this bound catches gross unit errors.
   277	        """
   278	        from legoesm.atmosphere.physics.convection.mass_flux import mass_flux_convection
   279	        from legoesm.atmosphere.physics.convection.config import MassFluxConfig
   280	

 succeeded in 0ms:
   150	            sigma.sigma_full, shape_3d,
   151	        )
   152	        q_sat = saturation_mixing_ratio(T_data, p_full)
   153	        q_v_data = 0.95 * q_sat
   154	
   155	        state = HydrostaticState(
   156	            u=Field(data=jnp.zeros(shape_3d), name="u", dims=("face", "x", "y", "level"), units="m/s"),
   157	            v=Field(data=jnp.zeros(shape_3d), name="v", dims=("face", "x", "y", "level"), units="m/s"),
   158	            T=Field(data=T_data, name="T", dims=("face", "x", "y", "level"), units="K"),
   159	            p_s=Field(data=p_s_data, name="p_s", dims=("face", "x", "y"), units="Pa"),
   160	            phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=("face", "x", "y"), units="m^2/s^2"),
   161	            tracers={
   162	                "q_v": Field(data=q_v_data, name="q_v", dims=("face", "x", "y", "level"), units="kg/kg"),
   163	            },
   164	        )
   165	
   166	        config = MicrophysicsConfig(scheme="kessler")
   167	        physics_fn = make_microphysics_physics(config, model_type="hydrostatic", dt=300.0)
   168	        tendencies = physics_fn(state, grid, sigma)
   169	
   170	        assert tendencies.tracer_tendencies is not None, \
   171	            "Hydrostatic microphysics must return tracer tendencies"
   172	        for key in ("q_v", "q_c", "q_r"):
   173	            assert key in tendencies.tracer_tendencies, f"Must include {key} tendency"
   174	        # At least vapor tendency should be nonzero for near-saturated air
   175	        dq_v = tendencies.tracer_tendencies["q_v"].data
   176	        assert jnp.any(dq_v != 0), "q_v tendency should be nonzero"
   177	
   178	
   179	# ======================================================================
   180	# Issue 3: Microphysics rate-vs-increment
   181	# ======================================================================
   182	
   183	class TestIssue3_MicrophysicsRateSemantics:
   184	
   185	    def test_kessler_condensation_depends_on_dt(self):
   186	        """Condensation tendency should differ when dt changes (not dt-independent)."""
   187	        from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
   188	        T, q_v, hydrometeors, p_full, p_half, rho, dz = _make_warm_micro_columns()
   189	
   190	        out1 = kessler_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt=60.0)
   191	        out2 = kessler_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt=600.0)
   192	
   193	        # The saturation adjustment component scales with 1/dt
   194	        # Even though other processes may dominate, dT_dt should differ
   195	        diff = jnp.abs(out1.dT_dt - out2.dT_dt).sum()
   196	        assert diff > 0, "dT_dt should depend on dt (condensation is divided by dt)"
   197	
   198	    def test_kessler_condensation_heating_scales_correctly(self):
   199	        """Condensation-only heating should be L_v/c_p * excess/dt, not L_v/c_p * excess."""
   200	        from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
   201	        from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
   202	
   203	        # Isolate condensation: use supersaturated air, no cloud/rain (no evaporation)
   204	        ncol, nlev = 2, 5
   205	        dt = 300.0
   206	        T = jnp.full((ncol, nlev), 280.0)
   207	        p_half = jnp.linspace(5e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
   208	        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
   209	        q_sat = saturation_mixing_ratio(T, p_full)
   210	        q_v = 1.05 * q_sat  # 5% supersaturated
   211	        rho = p_full / (constants.R_d * T)
   212	        dp = p_half[:, 1:] - p_half[:, :-1]
   213	        dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_full, 1.0)))
   214	        zero = jnp.zeros((ncol, nlev))
   215	        hydrometeors = HydrometeorState(
   216	            q_c=zero, q_r=zero, q_i=zero, q_s=zero, q_g=zero,
   217	            N_c=zero, N_r=zero, N_i=zero,
   218	        )
   219	
   220	        out = kessler_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt)
   221	
   222	        # Condensation heating: L_v/c_p * (excess/dt), should scale with 1/dt
   223	        excess = 0.05 * q_sat  # ~5% of q_sat
   224	        max_expected = float(constants.L_v / constants.c_pd * jnp.max(excess) / dt)
   225	        max_actual = float(jnp.max(jnp.abs(out.dT_dt)))
   226	
   227	        # Should be within 2x of expected (sigmoid smoothing may reduce it)
   228	        assert max_actual < 2 * max_expected, \
   229	            f"Heating {max_actual} >> expected {max_expected} (increment not divided by dt?)"
   230	        # Should not be dt-independent (i.e. max_expected * dt >> max_actual is wrong)

codex
**Findings**
- `src/legoesm/atmosphere/physics/microphysics/sundqvist.py:72-80,150-168` - Surface precipitation is diagnosed but not removed from the column. Severity: **critical**.  
  `dq_v_dt + dq_c_dt + dq_r_dt` sums to zero, while `rates.precipitation` can be positive, so column water plus precipitation is not conserved. Addresses hypothesis 10: source-level conservation fails here; I did not find a microphysics `xfail`.

- `src/legoesm/atmosphere/physics/microphysics/morrison.py:149-154` - Bergeron and riming phase changes omit fusion heating. Severity: **major**.  
  `dq_c -> dq_i/dq_s` freezes liquid water, so `+ L_f * (bergeron + riming_i + riming_s) / c_pd` is missing. Addresses hypothesis 1: correct; the quoted `~3.3e-5 K/s`, `~2.8 K/day` estimate is right.

- `src/legoesm/atmosphere/physics/microphysics/thompson.py:174-179` - Bergeron and riming fusion heating is also omitted in Thompson. Severity: **major**.  
  Same missing `L_f` term as Morrison for liquid-to-ice/snow riming and Bergeron conversion. Addresses hypothesis 2: Bergeron/riming verified; I would not add separate latent heat for `rime_to_graupel` as currently coded, because that term is frozen-species-to-frozen-species mass transfer.

- `src/legoesm/atmosphere/physics/microphysics/thompson.py:141-151,184-187` - Ice, snow, and graupel melting are not limited by donor mass over `dt`. Severity: **critical**.  
  With default `melt_rate=5e-3 s^-1` and `dt=1200 s`, `melt_rate * dt = 6`, so explicit updates can drive `q_i`, `q_s`, and `q_g` negative. Addresses hypothesis 3: verified, including graupel.

- `src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:39-42` - Saturation adjustment returns negative “condensation” in subsaturated air without checking cloud-water availability. Severity: **major**.  
  Callers add this directly to cloud water, so clear subsaturated air can evaporate nonexistent `q_c` and produce negative cloud water. Addresses hypothesis 9: the issue is stronger than “no Newton iteration”; the smooth formula is not donor-limited.

- `src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:93-96` - Seifert-Beheng autoconversion has number-unit and threshold-scale errors. Severity: **major**.  
  `N_c` is documented as `[1/kg]`, so mean droplet mass should be `q_c / N_c`, not `q_c * rho / N_c`; likewise `dN_r_au` should not multiply by `rho` for a `[1/kg/s]` tendency. Addresses hypothesis 6: the AD-at-zero concern is mostly fine, but `onset(0 - x_star)` is not near zero with `sharpness=50-100`; it is essentially `0.5`, so the mass threshold is almost disabled.

- `src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:174-175`, `src/legoesm/atmosphere/physics/microphysics/kessler.py:92,96`, `src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:89-91`, `src/legoesm/atmosphere/physics/microphysics/morrison.py:132-141`, `src/legoesm/atmosphere/physics/microphysics/thompson.py:155-167` - Fractional powers of clipped hydrometeor mass break gradients at zero. Severity: **major**.  
  `clip(q, 0) ** a` with `0<a<1` gives infinite gradients at `q=0`, and sedimentation products can yield NaN adjoints. Addresses hypothesis 7: verified as an AD bug, not just a practical edge case.

- `src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:97-101`, `src/legoesm/atmosphere/physics/microphysics/morrison.py:157-165`, `src/legoesm/atmosphere/physics/microphysics/thompson.py:182-191` - Number moments are not updated for several mass-removal processes. Severity: **major**.  
  `N_i` is not reduced by melting, and more broadly `N_c` ignores accretion/riming/Bergeron losses while `N_r` ignores evaporation/sedimentation losses. Addresses hypothesis 4: verified and broader than ice melting alone.

- `src/legoesm/atmosphere/physics/microphysics/morrison.py:105-117,157-161`, `src/legoesm/atmosphere/physics/microphysics/thompson.py:126-136,182-187` - Mixed-phase donor conversions are not limited by available donor mass. Severity: **critical**.  
  Default Bergeron alone can remove `1.2 * q_c` over `dt=1200 s`; riming and Thompson `rime_to_graupel` can similarly overdraw donors. This is an additional substantive finding beyond the listed hypotheses.

- `src/legoesm/atmosphere/physics/microphysics/morrison.py:88-103`, `src/legoesm/atmosphere/physics/microphysics/thompson.py:109-124` - Ice nucleation creates number but no ice mass seed. Severity: **major**.  
  If `q_i=0`, `dN_i_nuc` can be positive but `dq_i_dep` remains zero because it is proportional to `q_i`; cold supersaturated vapor can create massless ice crystals that never grow. Additional substantive finding.

- `src/legoesm/atmosphere/physics/microphysics/sundqvist.py:68,150-152` - Sundqvist autoconversion is not clamped to available cloud water over `dt`. Severity: **major**.  
  Using `q_c + condensation * dt` is not necessarily a bug by itself, but `P_auto * dt` can exceed that donor amount when `auto_rate * dt > 1`. Addresses hypothesis 5: coupling choice is debatable; missing donor limiter is concrete.

**Rebutted / Non-Findings**
- Hypothesis 8 is a non-bug: `_gamma_ratio(mu) = (mu+3)(mu+2)(mu+1)` is exact for real `mu` where the gamma functions are defined.
- Hypothesis 9, narrowly stated as “no saturation iteration,” is an accuracy/design limitation rather than a concrete bug; the concrete saturation bug is the negative, unbounded cloud-water tendency above.
- Hypothesis 10: I found no microphysics conservation tests marked `xfail`; the larger issue is that current tests do not actually enforce column water plus precipitation conservation for Sundqvist.
2026-05-01T11:58:09.559907Z ERROR codex_core::session: failed to record rollout items: thread 019de361-3ff6-7a50-af6e-20c82f3c6832 not found
tokens used
129,284
**Findings**
- `src/legoesm/atmosphere/physics/microphysics/sundqvist.py:72-80,150-168` - Surface precipitation is diagnosed but not removed from the column. Severity: **critical**.  
  `dq_v_dt + dq_c_dt + dq_r_dt` sums to zero, while `rates.precipitation` can be positive, so column water plus precipitation is not conserved. Addresses hypothesis 10: source-level conservation fails here; I did not find a microphysics `xfail`.

- `src/legoesm/atmosphere/physics/microphysics/morrison.py:149-154` - Bergeron and riming phase changes omit fusion heating. Severity: **major**.  
  `dq_c -> dq_i/dq_s` freezes liquid water, so `+ L_f * (bergeron + riming_i + riming_s) / c_pd` is missing. Addresses hypothesis 1: correct; the quoted `~3.3e-5 K/s`, `~2.8 K/day` estimate is right.

- `src/legoesm/atmosphere/physics/microphysics/thompson.py:174-179` - Bergeron and riming fusion heating is also omitted in Thompson. Severity: **major**.  
  Same missing `L_f` term as Morrison for liquid-to-ice/snow riming and Bergeron conversion. Addresses hypothesis 2: Bergeron/riming verified; I would not add separate latent heat for `rime_to_graupel` as currently coded, because that term is frozen-species-to-frozen-species mass transfer.

- `src/legoesm/atmosphere/physics/microphysics/thompson.py:141-151,184-187` - Ice, snow, and graupel melting are not limited by donor mass over `dt`. Severity: **critical**.  
  With default `melt_rate=5e-3 s^-1` and `dt=1200 s`, `melt_rate * dt = 6`, so explicit updates can drive `q_i`, `q_s`, and `q_g` negative. Addresses hypothesis 3: verified, including graupel.

- `src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:39-42` - Saturation adjustment returns negative “condensation” in subsaturated air without checking cloud-water availability. Severity: **major**.  
  Callers add this directly to cloud water, so clear subsaturated air can evaporate nonexistent `q_c` and produce negative cloud water. Addresses hypothesis 9: the issue is stronger than “no Newton iteration”; the smooth formula is not donor-limited.

- `src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:93-96` - Seifert-Beheng autoconversion has number-unit and threshold-scale errors. Severity: **major**.  
  `N_c` is documented as `[1/kg]`, so mean droplet mass should be `q_c / N_c`, not `q_c * rho / N_c`; likewise `dN_r_au` should not multiply by `rho` for a `[1/kg/s]` tendency. Addresses hypothesis 6: the AD-at-zero concern is mostly fine, but `onset(0 - x_star)` is not near zero with `sharpness=50-100`; it is essentially `0.5`, so the mass threshold is almost disabled.

- `src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:174-175`, `src/legoesm/atmosphere/physics/microphysics/kessler.py:92,96`, `src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:89-91`, `src/legoesm/atmosphere/physics/microphysics/morrison.py:132-141`, `src/legoesm/atmosphere/physics/microphysics/thompson.py:155-167` - Fractional powers of clipped hydrometeor mass break gradients at zero. Severity: **major**.  
  `clip(q, 0) ** a` with `0<a<1` gives infinite gradients at `q=0`, and sedimentation products can yield NaN adjoints. Addresses hypothesis 7: verified as an AD bug, not just a practical edge case.

- `src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:97-101`, `src/legoesm/atmosphere/physics/microphysics/morrison.py:157-165`, `src/legoesm/atmosphere/physics/microphysics/thompson.py:182-191` - Number moments are not updated for several mass-removal processes. Severity: **major**.  
  `N_i` is not reduced by melting, and more broadly `N_c` ignores accretion/riming/Bergeron losses while `N_r` ignores evaporation/sedimentation losses. Addresses hypothesis 4: verified and broader than ice melting alone.

- `src/legoesm/atmosphere/physics/microphysics/morrison.py:105-117,157-161`, `src/legoesm/atmosphere/physics/microphysics/thompson.py:126-136,182-187` - Mixed-phase donor conversions are not limited by available donor mass. Severity: **critical**.  
  Default Bergeron alone can remove `1.2 * q_c` over `dt=1200 s`; riming and Thompson `rime_to_graupel` can similarly overdraw donors. This is an additional substantive finding beyond the listed hypotheses.

- `src/legoesm/atmosphere/physics/microphysics/morrison.py:88-103`, `src/legoesm/atmosphere/physics/microphysics/thompson.py:109-124` - Ice nucleation creates number but no ice mass seed. Severity: **major**.  
  If `q_i=0`, `dN_i_nuc` can be positive but `dq_i_dep` remains zero because it is proportional to `q_i`; cold supersaturated vapor can create massless ice crystals that never grow. Additional substantive finding.

- `src/legoesm/atmosphere/physics/microphysics/sundqvist.py:68,150-152` - Sundqvist autoconversion is not clamped to available cloud water over `dt`. Severity: **major**.  
  Using `q_c + condensation * dt` is not necessarily a bug by itself, but `P_auto * dt` can exceed that donor amount when `auto_rate * dt > 1`. Addresses hypothesis 5: coupling choice is debatable; missing donor limiter is concrete.

**Rebutted / Non-Findings**
- Hypothesis 8 is a non-bug: `_gamma_ratio(mu) = (mu+3)(mu+2)(mu+1)` is exact for real `mu` where the gamma functions are defined.
- Hypothesis 9, narrowly stated as “no saturation iteration,” is an accuracy/design limitation rather than a concrete bug; the concrete saturation bug is the negative, unbounded cloud-water tendency above.
- Hypothesis 10: I found no microphysics conservation tests marked `xfail`; the larger issue is that current tests do not actually enforce column water plus precipitation conservation for Sundqvist.
