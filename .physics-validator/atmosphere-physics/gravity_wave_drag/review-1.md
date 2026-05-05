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
session id: 019de368-fa2b-71b1-8920-cf6bcf7b7202
--------
user
# Adversarial review: legoESM gravity wave drag

You are an independent adversarial physics reviewer. Read the GWD source files below.

Convention: column shape `(ncol, nlev)`, surface at `[:, -1]`, top at `[:, 0]`.

Hypotheses to verify or rebut:

1. **`hines.py:90` — `sigma_grown = sigma_gw * rho_ratio[:, k]` may be wrong**.
   - `rho_ratio[:, k] = sqrt(rho_sfc / rho[:, k])` — cumulative surface-to-k ratio.
   - The carry holds `sigma_new` from previous level's iteration.
   - Per WKB, amplitude grows as `sqrt(rho_(k+1)/rho_k)` between adjacent levels.
   - The code multiplies by `sqrt(rho_sfc/rho_k)` at every step, which compounds across levels.
   - In absence of dissipation: at k=nlev-2, sigma = sigma_init * sqrt(rho_sfc/rho[nlev-2]). At k=nlev-3, sigma = previous * sqrt(rho_sfc/rho[nlev-3]) = sigma_init * sqrt(rho_sfc/rho[nlev-2]) * sqrt(rho_sfc/rho[nlev-3]). **The expected value should be sigma_init * sqrt(rho_sfc/rho[nlev-3]).**
   - Effect: amplitude growth wildly overshoots WKB; saturation cap at sigma_sat fires aggressively to mask this.

2. **`hines.py:81` — `sigma_sat = N / (m_star * rho_ratio)`**.
   - `rho_ratio` decreases with altitude (rho_sfc/rho > 1 aloft).
   - This makes `sigma_sat` smaller aloft — opposite of expected (gravity wave saturation amplitude should be larger aloft).
   - Suspect copy-paste typo; should likely be `* rho_ratio` (multiplied) or equivalent formulation.

3. **All GWD schemes — no gradient tests in `tests/unit/test_physics_gwd.py`** (verified by grep).

4. **`mcfarlane.py:75, 100` — hardcoded sharpness 20 and softmin 50** in body, not config.

5. **`mcfarlane.py:86` — `tau_0 = jnp.clip(tau_0, 0.0, 10.0)`** with hardcoded upper bound 10 (units Pa or kg/m²/s).

6. **`lindzen.py:91-100`, `mcfarlane.py:96-106`, `hines.py:85-102` — same scan-then-T-then-reverse ordering pattern**. Verify correctness in each.

7. **Frictional heating formula `dT_dt = -(u du_dt + v dv_dt) / c_pd`** in all schemes. Verify sign.

Files:
- `src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py` (90 LOC).
- `src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py` (123 LOC).
- `src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py` (126 LOC).
- `src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py` (125 LOC).
- `src/legoesm/atmosphere/physics/gravity_wave_drag/config.py`.

For each finding: file path : line range, summary, technical explanation (≤3 sentences), severity (critical|major|minor).

If no findings beyond those flagged, end with `NO ADDITIONAL SUBSTANTIVE FINDINGS`.

codex
I’ll read the GWD implementations and the current unit tests first, then map each hypothesis to exact line ranges and separate confirmed issues from rebutted ones.
exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Smoothed McFarlane (1987) orographic gravity wave drag.
     2	
     3	Extends the Lindzen approach with explicit launch flux control,
     4	minimum wind activation, and directional spreading. Uses smooth
     5	(sigmoid / softmin) approximations for full differentiability.
     6	
     7	References
     8	----------
     9	- McFarlane, N. A. (1987). The effect of orographically excited gravity
    10	  wave drag on the general circulation of the lower stratosphere and
    11	  troposphere. J. Atmos. Sci., 44, 1775-1800.
    12	"""
    13	
    14	from __future__ import annotations
    15	
    16	import jax
    17	import jax.numpy as jnp
    18	
    19	from legoesm import constants
    20	from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
    21	from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
    22	
    23	
    24	def mcfarlane_gwd(
    25	    u: jax.Array,
    26	    v: jax.Array,
    27	    T: jax.Array,
    28	    p_full: jax.Array,
    29	    p_half: jax.Array,
    30	    z_full: jax.Array,
    31	    z_half: jax.Array,
    32	    rho: jax.Array,
    33	    lat: jax.Array,
    34	    dt: float,
    35	    config: McFarlaneConfig,
    36	) -> GWDOutput:
    37	    """Compute McFarlane orographic GWD tendencies.
    38	
    39	    Parameters
    40	    ----------
    41	    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
    42	        Standard GWD backend signature. All column arrays (ncol, nlev).
    43	
    44	    Returns
    45	    -------
    46	    GWDOutput
    47	    """
    48	    ncol, nlev = u.shape
    49	
    50	    # Brunt-Väisälä frequency
    51	    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    52	    dz_full = jnp.abs(z_full[:, :-1] - z_full[:, 1:])
    53	    dz_full = jnp.clip(dz_full, 1.0, None)
    54	    dtheta_dz = (theta[:, :-1] - theta[:, 1:]) / dz_full
    55	    theta_bar = 0.5 * (theta[:, :-1] + theta[:, 1:])
    56	    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
    57	    N2_half = jnp.clip(N2_half, 1e-8, None)
    58	    N_half = jnp.sqrt(N2_half)
    59	
    60	    # Extrapolate N to full levels
    61	    N_full = jnp.concatenate([
    62	        N_half[:, :1],
    63	        0.5 * (N_half[:, :-1] + N_half[:, 1:]),
    64	        N_half[:, -1:],
    65	    ], axis=1)
    66	
    67	    # Low-level wind
    68	    u_sfc = u[:, -1]
    69	    v_sfc = v[:, -1]
    70	    U_ll = jnp.sqrt(u_sfc ** 2 + v_sfc ** 2 + 1e-10)
    71	    cos_a = u_sfc / U_ll
    72	    sin_a = v_sfc / U_ll
    73	
    74	    # Smooth minimum wind activation
    75	    U_activated = jax.nn.sigmoid(20.0 * (U_ll - config.min_wind)) * U_ll
    76	
    77	    # Wind projection along wave direction
    78	    U_proj = u * cos_a[:, None] + v * sin_a[:, None]
    79	    U_proj_abs = jnp.clip(jnp.abs(U_proj), 1e-2, None)
    80	
    81	    # Launch flux
    82	    rho_sfc = rho[:, -1]
    83	    N_sfc = N_full[:, -1]
    84	    tau_0 = config.G_0 * U_activated * config.h_topo ** 2 * N_sfc * rho_sfc
    85	    tau_0 = tau_0 * config.directional_spread
    86	    tau_0 = jnp.clip(tau_0, 0.0, 10.0)
    87	
    88	    # Saturation stress per level
    89	    envelope = config.envelope_scale
    90	    tau_sat = config.efficiency * rho * U_proj_abs ** 3 / (
    91	        jnp.clip(N_full, 1e-6, None) * envelope
    92	    )
    93	    tau_sat = jnp.clip(tau_sat, 1e-10, None)
    94	
    95	    # Top-down scan with smooth min (softmin via logsumexp)
    96	    def scan_fn(carry, k_rev):
    97	        tau_carry = carry
    98	        k = nlev - 1 - k_rev
    99	        # Smooth min: softmin(a, b) = -logsumexp(-alpha*[a,b])/alpha
   100	        alpha = 50.0
   101	        tau_k = -jax.nn.logsumexp(
   102	            jnp.stack([-alpha * tau_carry, -alpha * tau_sat[:, k]], axis=0),
   103	            axis=0,
   104	        ) / alpha
   105	        drag = tau_carry - tau_k
   106	        return tau_k, drag
   107	
   108	    _, drag_stack = jax.lax.scan(scan_fn, tau_0, jnp.arange(nlev))
   109	    drag_all = drag_stack.T[:, ::-1]  # (ncol, nlev), top-first
   110	
   111	    # Convert to tendency
   112	    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
   113	    dz = jnp.clip(dz, 1.0, None)
   114	    accel = -drag_all / jnp.clip(rho * dz, 1e-10, None)
   115	
   116	    du_dt = accel * cos_a[:, None]
   117	    dv_dt = accel * sin_a[:, None]
   118	
   119	    # Frictional heating
   120	    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
   121	
   122	    # Column dissipation (positive-definite: KE lost by the mean flow)
   123	    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
   124	
   125	    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)

 succeeded in 0ms:
     1	"""Hines (1997) Doppler-spread gravity wave drag parameterization.
     2	
     3	Non-orographic GWD scheme based on Doppler shifting and spectral
     4	saturation of gravity waves. Uses bottom-up propagation with smooth
     5	sigmoid activation for full differentiability.
     6	
     7	References
     8	----------
     9	- Hines, C. O. (1997). Doppler-spread parameterization of gravity-wave
    10	  momentum deposition in the middle atmosphere. 1. Basic formulation.
    11	  J. Atmos. Solar-Terr. Phys., 59, 371-386.
    12	"""
    13	
    14	from __future__ import annotations
    15	
    16	import jax
    17	import jax.numpy as jnp
    18	
    19	from legoesm import constants
    20	from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
    21	from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
    22	
    23	
    24	def hines_gwd(
    25	    u: jax.Array,
    26	    v: jax.Array,
    27	    T: jax.Array,
    28	    p_full: jax.Array,
    29	    p_half: jax.Array,
    30	    z_full: jax.Array,
    31	    z_half: jax.Array,
    32	    rho: jax.Array,
    33	    lat: jax.Array,
    34	    dt: float,
    35	    config: HinesConfig,
    36	) -> GWDOutput:
    37	    """Compute Hines Doppler-spread GWD tendencies.
    38	
    39	    Parameters
    40	    ----------
    41	    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
    42	        Standard GWD backend signature. All column arrays (ncol, nlev).
    43	
    44	    Returns
    45	    -------
    46	    GWDOutput
    47	    """
    48	    ncol, nlev = u.shape
    49	
    50	    # Brunt-Väisälä frequency
    51	    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    52	    dz_full = jnp.abs(z_full[:, :-1] - z_full[:, 1:])
    53	    dz_full = jnp.clip(dz_full, 1.0, None)
    54	    dtheta_dz = (theta[:, :-1] - theta[:, 1:]) / dz_full
    55	    theta_bar = 0.5 * (theta[:, :-1] + theta[:, 1:])
    56	    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
    57	    N2_half = jnp.clip(N2_half, 1e-8, None)
    58	    N_half = jnp.sqrt(N2_half)
    59	
    60	    N_full = jnp.concatenate([
    61	        N_half[:, :1],
    62	        0.5 * (N_half[:, :-1] + N_half[:, 1:]),
    63	        N_half[:, -1:],
    64	    ], axis=1)
    65	
    66	    # Wind magnitude at each level
    67	    U_mag = jnp.sqrt(u ** 2 + v ** 2 + 1e-10)
    68	
    69	    # Layer thickness
    70	    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    71	    dz = jnp.clip(dz, 1.0, None)
    72	
    73	    # Saturation amplitude per level:
    74	    # As gravity waves propagate upward, their amplitude grows with
    75	    # decreasing density (energy conservation: F ~ rho * sigma^2 = const).
    76	    # Saturation occurs when the wave-induced velocity perturbation
    77	    # reaches a fraction of the local wind: sigma_sat ~ U / m_star_norm
    78	    rho_sfc = rho[:, -1:]  # (ncol, 1)
    79	    rho_ratio = jnp.sqrt(jnp.clip(rho_sfc / jnp.clip(rho, 0.01, None), 1.0, None))
    80	    # sigma_sat = wind_fraction * N / (m_star) at each level
    81	    sigma_sat = N_full / jnp.clip(config.m_star * rho_ratio, 1e-6, None)
    82	
    83	    # Bottom-up scan: propagate sigma_gw upward from surface
    84	    # As wave propagates up, amplitude grows with sqrt(rho_sfc/rho)
    85	    def scan_fn(carry, k_rev):
    86	        sigma_gw = carry
    87	        k = nlev - 1 - k_rev
    88	
    89	        # Amplitude growth from density decrease
    90	        sigma_grown = sigma_gw * rho_ratio[:, k]
    91	
    92	        # Dissipation where grown amplitude exceeds saturation
    93	        f_diss = jax.nn.sigmoid(
    94	            config.doppler_sharpness * (sigma_grown - sigma_sat[:, k])
    95	        )
    96	        sigma_new = sigma_grown * (1.0 - f_diss) + sigma_sat[:, k] * f_diss
    97	
    98	        # Momentum deposited: rho * (sigma_grown - sigma_new) ~ stress gradient
    99	        drag = (sigma_grown - sigma_new) * rho[:, k]
   100	        drag = jnp.clip(drag, -config.Fmax, config.Fmax)
   101	
   102	        return sigma_new, drag
   103	
   104	    # Pin the carry dtype so the scan body stays at the input precision
   105	    # (defaulting allows x64 to silently promote the launch wind to f64).
   106	    sigma_gw_init = jnp.full((ncol,), config.total_rms_wind, dtype=u.dtype)
   107	    _, drag_stack = jax.lax.scan(scan_fn, sigma_gw_init, jnp.arange(nlev))
   108	    drag_all = drag_stack.T[:, ::-1]  # (ncol, nlev), top-first
   109	
   110	    # Convert to acceleration
   111	    accel = -drag_all / jnp.clip(rho * dz, 1e-10, None)
   112	
   113	    cos_a = u / jnp.clip(U_mag, config.U_mag_floor, None)
   114	    sin_a = v / jnp.clip(U_mag, config.U_mag_floor, None)
   115	    du_dt = accel * cos_a
   116	    dv_dt = accel * sin_a
   117	
   118	    # Frictional heating
   119	    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
   120	
   121	    # Column dissipation (positive-definite: KE lost by the mean flow)
   122	    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
   123	
   124	    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)

exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Smoothed Lindzen (1981) orographic gravity wave drag.
     2	
     3	Orographic GWD with smooth sigmoid activation for wave breaking,
     4	fully differentiable via jax.lax.scan for the vertical stress profile.
     5	
     6	References
     7	----------
     8	- Lindzen, R. S. (1981). Turbulence and stress owing to gravity wave and
     9	  tidal breakdown. J. Geophys. Res., 86, 9707-9714.
    10	"""
    11	
    12	from __future__ import annotations
    13	
    14	import jax
    15	import jax.numpy as jnp
    16	
    17	from legoesm import constants
    18	from legoesm.atmosphere.physics.gravity_wave_drag.config import LindzenConfig
    19	from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
    20	
    21	
    22	def lindzen_gwd(
    23	    u: jax.Array,
    24	    v: jax.Array,
    25	    T: jax.Array,
    26	    p_full: jax.Array,
    27	    p_half: jax.Array,
    28	    z_full: jax.Array,
    29	    z_half: jax.Array,
    30	    rho: jax.Array,
    31	    lat: jax.Array,
    32	    dt: float,
    33	    config: LindzenConfig,
    34	) -> GWDOutput:
    35	    """Compute Lindzen orographic GWD tendencies.
    36	
    37	    Parameters
    38	    ----------
    39	    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
    40	        Standard GWD backend signature. All column arrays (ncol, nlev).
    41	
    42	    Returns
    43	    -------
    44	    GWDOutput
    45	    """
    46	    ncol, nlev = u.shape
    47	
    48	    # Brunt-Väisälä frequency at full levels
    49	    # theta_v = T * (p_ref/p)^kappa
    50	    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    51	    dz_full = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    52	    dz_full = jnp.clip(dz_full, 1.0, None)
    53	    dtheta_dz = (theta[:, :-1] - theta[:, 1:]) / dz_full
    54	    theta_bar = 0.5 * (theta[:, :-1] + theta[:, 1:])
    55	    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
    56	    N2_half = jnp.clip(N2_half, 1e-8, None)
    57	    N_half = jnp.sqrt(N2_half)  # (ncol, nlev-1)
    58	
    59	    # Extrapolate N to full levels by padding
    60	    N_full = jnp.concatenate([
    61	        N_half[:, :1],
    62	        0.5 * (N_half[:, :-1] + N_half[:, 1:]),
    63	        N_half[:, -1:],
    64	    ], axis=1)  # (ncol, nlev)
    65	
    66	    # Low-level wind at surface level
    67	    u_sfc = u[:, -1]
    68	    v_sfc = v[:, -1]
    69	    U_ll = jnp.sqrt(u_sfc ** 2 + v_sfc ** 2 + 1e-10)
    70	    cos_a = u_sfc / U_ll
    71	    sin_a = v_sfc / U_ll
    72	
    73	    # Wind projection along wave direction at each level
    74	    U_proj = u * cos_a[:, None] + v * sin_a[:, None]  # (ncol, nlev)
    75	
    76	    # Source stress at surface: tau_0 = rho * N * k * h^2 * U
    77	    rho_sfc = rho[:, -1]
    78	    N_sfc = N_full[:, -1]
    79	    tau_0 = rho_sfc * N_sfc * config.k_wave * config.h_topo ** 2 * U_ll
    80	    tau_0 = jnp.clip(tau_0, 0.0, None)
    81	
    82	    # Saturation stress per level: tau_sat = rho * U^3 * k / N
    83	    # Wave breaks where carried stress exceeds local saturation
    84	    U_proj_abs = jnp.clip(jnp.abs(U_proj), 0.1, None)
    85	    tau_sat = rho * U_proj_abs ** 3 * config.k_wave / jnp.clip(N_full, 1e-6, None)
    86	    tau_sat = jnp.clip(tau_sat, 1e-10, None)
    87	
    88	    # Top-down scan: propagate stress from surface upward
    89	    # Levels: 0=top, -1=surface. Scan from surface to top (reversed).
    90	    # Breaking occurs smoothly where tau_carry exceeds tau_sat
    91	    def scan_fn(carry, k_rev):
    92	        tau_carry = carry
    93	        k = nlev - 1 - k_rev
    94	        # Smooth breaking: sigmoid activation where stress exceeds saturation
    95	        excess = tau_carry / jnp.clip(tau_sat[:, k], 1e-10, None) - config.critical_Fr
    96	        f_break = jax.nn.sigmoid(config.Fr_sharpness * excess)
    97	        tau_new = tau_carry * (1.0 - f_break) + tau_sat[:, k] * f_break
    98	        tau_new = jnp.minimum(tau_new, tau_carry)
    99	        drag = tau_carry - tau_new
   100	        return tau_new, drag
   101	
   102	    _, drag_stack = jax.lax.scan(scan_fn, tau_0, jnp.arange(nlev))
   103	    # drag_stack: (nlev, ncol) — reverse to get (ncol, nlev) top-first
   104	    drag_all = drag_stack.T  # (ncol, nlev)
   105	    drag_all = drag_all[:, ::-1]  # back to top-first ordering
   106	
   107	    # Convert stress deposit to tendency: drag / (rho * dz)
   108	    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
   109	    dz = jnp.clip(dz, 1.0, None)
   110	    accel = -drag_all / (jnp.clip(rho * dz, 1e-10, None))
   111	
   112	    # Project back to (du_dt, dv_dt)
   113	    du_dt = accel * cos_a[:, None]
   114	    dv_dt = accel * sin_a[:, None]
   115	
   116	    # Frictional heating
   117	    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
   118	
   119	    # Column dissipation (positive-definite: KE lost by the mean flow)
   120	    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
   121	
   122	    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)

exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Rayleigh friction gravity wave drag.
     2	
     3	Simplest GWD parameterization: applies linear drag proportional to wind
     4	speed in the boundary layer and an upper-atmosphere sponge layer.
     5	
     6	Follows the same pattern as held_suarez.py for sigma-based drag profiles.
     7	"""
     8	
     9	from __future__ import annotations
    10	
    11	import jax
    12	import jax.numpy as jnp
    13	
    14	from legoesm import constants
    15	from legoesm.atmosphere.physics.gravity_wave_drag.config import RayleighConfig
    16	from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
    17	
    18	
    19	def rayleigh_gwd(
    20	    u: jax.Array,
    21	    v: jax.Array,
    22	    T: jax.Array,
    23	    p_full: jax.Array,
    24	    p_half: jax.Array,
    25	    z_full: jax.Array,
    26	    z_half: jax.Array,
    27	    rho: jax.Array,
    28	    lat: jax.Array,
    29	    dt: float,
    30	    config: RayleighConfig,
    31	) -> GWDOutput:
    32	    """Compute Rayleigh friction GWD tendencies.
    33	
    34	    Parameters
    35	    ----------
    36	    u, v : jax.Array
    37	        Wind components [m/s], shape (ncol, nlev).
    38	    T : jax.Array
    39	        Temperature [K], shape (ncol, nlev).
    40	    p_full : jax.Array
    41	        Pressure at full levels [Pa], shape (ncol, nlev).
    42	    p_half : jax.Array
    43	        Pressure at half levels [Pa], shape (ncol, nlev+1).
    44	    z_full, z_half : jax.Array
    45	        Heights [m], shapes (ncol, nlev) and (ncol, nlev+1).
    46	    rho : jax.Array
    47	        Air density [kg/m^3], shape (ncol, nlev).
    48	    lat : jax.Array
    49	        Latitude [rad], shape (ncol,).
    50	    dt : float
    51	        Time step [s].
    52	    config : RayleighConfig
    53	
    54	    Returns
    55	    -------
    56	    GWDOutput
    57	    """
    58	    ncol, nlev = u.shape
    59	
    60	    # Sigma coordinate
    61	    p_sfc = p_half[:, -1:]  # (ncol, 1)
    62	    sigma = p_full / jnp.clip(p_sfc, 1.0, None)
    63	
    64	    # Boundary layer drag: ramps from 0 at sigma_b to k_max at surface
    65	    k_bl = config.k_max * jnp.clip(
    66	        (sigma - config.sigma_b) / (1.0 - config.sigma_b), 0.0, 1.0
    67	    )
    68	
    69	    # Upper sponge: sin^2 taper near model top
    70	    sponge_arg = jnp.clip(
    71	        (config.sponge_top - sigma) / config.sponge_top, 0.0, 1.0
    72	    )
    73	    k_sponge = config.sponge_k * jnp.sin(0.5 * jnp.pi * sponge_arg) ** 2
    74	
    75	    # Combined drag coefficient
    76	    k_drag = k_bl + k_sponge
    77	
    78	    # Tendencies
    79	    du_dt = -k_drag * u
    80	    dv_dt = -k_drag * v
    81	
    82	    # Frictional heating: dT/dt = -(u*du/dt + v*dv/dt) / c_pd
    83	    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
    84	
    85	    # Column dissipation (positive-definite: KE lost by the mean flow)
    86	    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    87	    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
    88	
    89	    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)

exec
/bin/zsh -lc 'nl -ba src/legoesm/atmosphere/physics/gravity_wave_drag/config.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Configuration for gravity wave drag schemes.
     2	
     3	Provides configuration NamedTuples for:
     4	1. Rayleigh: simple Rayleigh friction drag
     5	2. Lindzen: smoothed Lindzen (1981) orographic GWD
     6	3. McFarlane: smoothed McFarlane (1987) orographic GWD
     7	4. Hines: Hines (1997) Doppler-spread parameterization
     8	5. PrognosticSpectral: prognostic spectral GWD
     9	6. MLEmulator: ML-based GWD emulator (Equinox MLP)
    10	7. GravityWaveDragConfig: top-level selector
    11	
    12	References
    13	----------
    14	- Lindzen, R. S. (1981). Turbulence and stress owing to gravity wave and
    15	  tidal breakdown. J. Geophys. Res., 86, 9707-9714.
    16	- McFarlane, N. A. (1987). The effect of orographically excited gravity
    17	  wave drag on the general circulation of the lower stratosphere and
    18	  troposphere. J. Atmos. Sci., 44, 1775-1800.
    19	- Hines, C. O. (1997). Doppler-spread parameterization of gravity-wave
    20	  momentum deposition in the middle atmosphere. 1. Basic formulation.
    21	  J. Atmos. Solar-Terr. Phys., 59, 371-386.
    22	"""
    23	
    24	from __future__ import annotations
    25	
    26	import math
    27	from typing import NamedTuple
    28	
    29	
    30	class RayleighConfig(NamedTuple):
    31	    """Configuration for Rayleigh friction drag.
    32	
    33	    Fields
    34	    ------
    35	    k_max : float
    36	        Maximum drag coefficient [1/s] (default 1/(1*86400)).
    37	    sigma_b : float
    38	        Boundary layer top sigma level (default 0.7).
    39	    sponge_top : float
    40	        Upper sponge sigma level (default 0.02).
    41	    sponge_k : float
    42	        Upper sponge drag coefficient [1/s] (default 1/(0.5*86400)).
    43	    """
    44	    k_max: float = 1.0 / 86400.0
    45	    sigma_b: float = 0.7
    46	    sponge_top: float = 0.02
    47	    sponge_k: float = 1.0 / (0.5 * 86400.0)
    48	
    49	
    50	class LindzenConfig(NamedTuple):
    51	    """Configuration for smoothed Lindzen (1981) orographic GWD.
    52	
    53	    Fields
    54	    ------
    55	    h_topo : float
    56	        Sub-grid topographic height [m] (default 500).
    57	    k_wave : float
    58	        Horizontal wavenumber [1/m] (default 2*pi/100e3).
    59	    N_ref : float
    60	        Reference Brunt-Väisälä frequency [1/s] (default 0.01).
    61	    critical_Fr : float
    62	        Critical Froude number threshold (default 1.0).
    63	    Fr_sharpness : float
    64	        Sigmoid sharpness for Froude number transition (default 20.0).
    65	    """
    66	    h_topo: float = 500.0
    67	    k_wave: float = 2.0 * math.pi / 100e3
    68	    N_ref: float = 0.01
    69	    critical_Fr: float = 1.0
    70	    Fr_sharpness: float = 20.0
    71	
    72	
    73	class McFarlaneConfig(NamedTuple):
    74	    """Configuration for smoothed McFarlane (1987) orographic GWD.
    75	
    76	    Extends the Lindzen approach with explicit launch flux control
    77	    and directional spreading.
    78	
    79	    Fields
    80	    ------
    81	    h_topo : float
    82	        Sub-grid topographic height [m] (default 500).
    83	    k_wave : float
    84	        Horizontal wavenumber [1/m] (default 2*pi/100e3).
    85	    N_ref : float
    86	        Reference Brunt-Väisälä frequency [1/s] (default 0.01).
    87	    G_0 : float
    88	        Launch momentum flux scale [Pa] (default 0.5).
    89	    efficiency : float
    90	        Breaking efficiency (default 0.5).
    91	    min_wind : float
    92	        Minimum wind for wave activity [m/s] (default 2.0).
    93	    envelope_scale : float
    94	        Vertical envelope scale (default 1.0).
    95	    directional_spread : float
    96	        Multi-directional spreading factor (default 1.0).
    97	    """
    98	    h_topo: float = 500.0
    99	    k_wave: float = 2.0 * math.pi / 100e3
   100	    N_ref: float = 0.01
   101	    G_0: float = 0.5
   102	    efficiency: float = 0.5
   103	    min_wind: float = 2.0
   104	    envelope_scale: float = 1.0
   105	    directional_spread: float = 1.0
   106	
   107	
   108	class HinesConfig(NamedTuple):
   109	    """Configuration for Hines (1997) Doppler-spread parameterization.
   110	
   111	    Fields
   112	    ------
   113	    rms_gw_speed : float
   114	        RMS gravity wave speed [m/s] (default 1.0).
   115	    m_star : float
   116	        Characteristic vertical wavenumber [1/m] (default 2*pi/2e3).
   117	    total_rms_wind : float
   118	        Total RMS gravity wave wind [m/s] (default 2.0).
   119	    cutoff_wn : float
   120	        Maximum vertical wavenumber [1/m] (default 2*pi/500).
   121	    Fmax : float
   122	        Saturation momentum flux cap [Pa] (default 0.1).
   123	    doppler_sharpness : float
   124	        Sigmoid sharpness for Doppler saturation (default 50.0).
   125	    """
   126	    rms_gw_speed: float = 1.0
   127	    m_star: float = 2.0 * math.pi / 2e3
   128	    total_rms_wind: float = 2.0
   129	    cutoff_wn: float = 2.0 * math.pi / 500.0
   130	    Fmax: float = 0.1
   131	    doppler_sharpness: float = 50.0
   132	    U_mag_floor: float = 0.1  # Wind-magnitude floor for projection [m/s]
   133	
   134	
   135	class PrognosticSpectralConfig(NamedTuple):
   136	    """Configuration for prognostic spectral GWD.
   137	
   138	    Fields
   139	    ------
   140	    n_azimuths : int
   141	        Number of azimuthal directions (default 4).
   142	    n_wavenumbers : int
   143	        Number of spectral bins (default 20).
   144	    k_min : float
   145	        Minimum horizontal wavenumber [1/m] (default 2*pi/100e3).
   146	    k_max : float
   147	        Maximum horizontal wavenumber [1/m] (default 2*pi/1e3).
   148	    launch_flux : float
   149	        Source momentum flux [Pa] (default 1e-3).
   150	    breaking_threshold : float
   151	        Froude threshold for wave breaking (default 1.0).
   152	    breaking_sharpness : float
   153	        Sigmoid sharpness for breaking transition (default 10.0).
   154	    tau_decay : float
   155	        Relaxation timescale for prognostic spectrum [s] (default 86400).
   156	    """
   157	    n_azimuths: int = 4
   158	    n_wavenumbers: int = 20
   159	    k_min: float = 2.0 * math.pi / 100e3
   160	    k_max: float = 2.0 * math.pi / 1e3
   161	    launch_flux: float = 1e-3
   162	    breaking_threshold: float = 1.0
   163	    breaking_sharpness: float = 10.0
   164	    tau_decay: float = 86400.0
   165	
   166	
   167	class MLEmulatorConfig(NamedTuple):
   168	    """Configuration for ML-based GWD emulator.
   169	
   170	    Fields
   171	    ------
   172	    n_input : int
   173	        Number of input features per level (default 7).
   174	    n_hidden : int
   175	        Hidden layer width (default 128).
   176	    n_layers : int
   177	        Number of MLP layers (default 3).
   178	    n_output : int
   179	        Number of output tendencies per level (default 3).
   180	    seed : int
   181	        Random seed for initialization (default 0).
   182	    use_residual : bool
   183	        Scale outputs for residual learning (default True).
   184	    """
   185	    n_input: int = 7
   186	    n_hidden: int = 128
   187	    n_layers: int = 3
   188	    n_output: int = 3
   189	    seed: int = 0
   190	    use_residual: bool = True
   191	    norm_u: float = 30.0     # Wind scale [m/s] for u, v normalization
   192	    norm_T: float = 300.0    # Temperature scale [K]
   193	    norm_z: float = 30000.0  # Height scale [m]
   194	
   195	
   196	class GravityWaveDragConfig(NamedTuple):
   197	    """Top-level gravity wave drag configuration.
   198	
   199	    Selects the active scheme and holds sub-configurations.
   200	
   201	    Fields
   202	    ------
   203	    scheme : str
   204	        Active GWD scheme: "rayleigh", "lindzen", "mcfarlane",
   205	        "hines", "prognostic_spectral", "ml_emulator", or "none".
   206	    rayleigh : RayleighConfig
   207	        Configuration for Rayleigh friction scheme.
   208	    lindzen : LindzenConfig
   209	        Configuration for Lindzen orographic scheme.
   210	    mcfarlane : McFarlaneConfig
   211	        Configuration for McFarlane orographic scheme.
   212	    hines : HinesConfig
   213	        Configuration for Hines Doppler-spread scheme.
   214	    prognostic_spectral : PrognosticSpectralConfig
   215	        Configuration for prognostic spectral scheme.
   216	    ml_emulator : MLEmulatorConfig
   217	        Configuration for ML emulator scheme.
   218	    """
   219	    scheme: str = "none"
   220	    rayleigh: RayleighConfig = RayleighConfig()
   221	    lindzen: LindzenConfig = LindzenConfig()
   222	    mcfarlane: McFarlaneConfig = McFarlaneConfig()
   223	    hines: HinesConfig = HinesConfig()
   224	    prognostic_spectral: PrognosticSpectralConfig = PrognosticSpectralConfig()
   225	    ml_emulator: MLEmulatorConfig = MLEmulatorConfig()

exec
/bin/zsh -lc 'rg -n "grad|gradient|value_and_grad|jac|jvp|vjp|finite|diff|gwd|gravity" tests/unit/test_physics_gwd.py tests src/legoesm/atmosphere/physics/gravity_wave_drag' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
tests/unit/test_physics_gwd.py:13:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/unit/test_physics_gwd.py:14:from legoesm.atmosphere.physics.gravity_wave_drag.integration import make_gwd_physics
tests/unit/test_physics_gwd.py:39:def _run_gwd(scheme, **kwargs):
tests/unit/test_physics_gwd.py:42:    gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
tests/unit/test_physics_gwd.py:43:    tend, prog = gwd_fn(state, grid, sigma)
tests/unit/test_physics_gwd.py:59:    tend, state = _run_gwd(scheme, wind_speed=15.0)
tests/unit/test_physics_gwd.py:86:    gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
tests/unit/test_physics_gwd.py:87:    tend, _ = gwd_fn(state, grid, sigma)
tests/unit/test_physics_gwd.py:101:    tend, _ = _run_gwd(scheme)
tests/unit/test_physics_gwd.py:109:# All outputs finite
tests/unit/test_physics_gwd.py:113:def test_all_finite(scheme):
tests/unit/test_physics_gwd.py:114:    """All GWD output fields should be finite."""
tests/unit/test_physics_gwd.py:115:    tend, _ = _run_gwd(scheme)
tests/unit/test_physics_gwd.py:116:    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{scheme}: du_dt NaN/Inf"
tests/unit/test_physics_gwd.py:117:    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{scheme}: dv_dt NaN/Inf"
tests/unit/test_physics_gwd.py:118:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt NaN/Inf"
tests/unit/test_physics_gwd.py:127:    tend, _ = _run_gwd("rayleigh", nlev=20, wind_speed=20.0)
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:17:    """Output from a gravity wave drag scheme (backend-agnostic).
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:30:    eps_gwd : jax.Array
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:32:        Positive-definite: eps_gwd = -sum(rho * (u*du_dt + v*dv_dt) * dz).
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:37:    eps_gwd: jax.Array
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:50:        eps_gwd=jnp.zeros((ncol,), dtype=dtype),
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py:1:"""ML gravity wave drag emulator using Equinox MLP.
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py:4:differentiable with jax.grad. The MLP maps per-level inputs
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py:17:from legoesm.atmosphere.physics.gravity_wave_drag.config import MLEmulatorConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py:18:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py:39:def ml_gwd(
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py:96:    eps_gwd = jnp.sum(rho * jnp.abs(u * du_dt + v * dv_dt) * dz, axis=1)
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py:98:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:1:"""Hines (1997) Doppler-spread gravity wave drag parameterization.
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:4:saturation of gravity waves. Uses bottom-up propagation with smooth
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:5:sigmoid activation for full differentiability.
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:9:- Hines, C. O. (1997). Doppler-spread parameterization of gravity-wave
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:20:from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:21:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:24:def hines_gwd(
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:74:    # As gravity waves propagate upward, their amplitude grows with
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:98:        # Momentum deposited: rho * (sigma_grown - sigma_new) ~ stress gradient
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:121:    # Column dissipation (positive-definite: KE lost by the mean flow)
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:122:    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:124:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:1:"""Rayleigh friction gravity wave drag.
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:15:from legoesm.atmosphere.physics.gravity_wave_drag.config import RayleighConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:16:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:19:def rayleigh_gwd(
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:85:    # Column dissipation (positive-definite: KE lost by the mean flow)
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:87:    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:89:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:1:"""Prognostic spectral gravity wave drag parameterization.
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:7:Uses jax.lax.scan for the vertical propagation, fully differentiable.
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:18:from legoesm.atmosphere.physics.gravity_wave_drag.config import (
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:21:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:26:def prognostic_spectral_gwd(
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:155:    # drag is stress gradient -> acceleration = -drag_deposit (already divided by dp)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:166:    # Column dissipation (positive-definite: KE lost by the mean flow)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:167:    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:179:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd), spectrum_new
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:1:"""Smoothed McFarlane (1987) orographic gravity wave drag.
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:5:(sigmoid / softmin) approximations for full differentiability.
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:9:- McFarlane, N. A. (1987). The effect of orographically excited gravity
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:20:from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:21:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:24:def mcfarlane_gwd(
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:122:    # Column dissipation (positive-definite: KE lost by the mean flow)
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:123:    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:125:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:13:Use `make_gwd_physics()` to create a physics function matching
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:18:>>> from legoesm.atmosphere.physics.gravity_wave_drag import (
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:19:...     GravityWaveDragConfig, make_gwd_physics,
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:22:>>> physics_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:26:from legoesm.atmosphere.physics.gravity_wave_drag.config import (
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:35:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:36:from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:37:from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:38:from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:39:from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:40:from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:41:    prognostic_spectral_gwd,
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:43:from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import (
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:44:    ml_gwd,
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:47:from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py:48:    make_gwd_physics,
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:1:"""Configuration for gravity wave drag schemes.
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:14:- Lindzen, R. S. (1981). Turbulence and stress owing to gravity wave and
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:16:- McFarlane, N. A. (1987). The effect of orographically excited gravity
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:19:- Hines, C. O. (1997). Doppler-spread parameterization of gravity-wave
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:114:        RMS gravity wave speed [m/s] (default 1.0).
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:118:        Total RMS gravity wave wind [m/s] (default 2.0).
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:197:    """Top-level gravity wave drag configuration.
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:1:"""Model integration bridge for gravity wave drag.
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:3:Provides `make_gwd_physics()`, a factory that returns a physics
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:39:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:40:from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:41:from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:42:from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:43:from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:44:from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:45:    prognostic_spectral_gwd,
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:47:from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import (
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:48:    ml_gwd,
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:58:def _get_gwd_fn(config: GravityWaveDragConfig):
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:61:        return "rayleigh", rayleigh_gwd, config.rayleigh
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:63:        return "lindzen", lindzen_gwd, config.lindzen
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:65:        return "mcfarlane", mcfarlane_gwd, config.mcfarlane
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:67:        return "hines", hines_gwd, config.hines
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:69:        return "prognostic_spectral", prognostic_spectral_gwd, config.prognostic_spectral
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:71:        return "ml_emulator", ml_gwd, config.ml_emulator
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:84:def make_gwd_physics(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:85:    gwd_config: GravityWaveDragConfig,
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:93:    gwd_config : GravityWaveDragConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:106:        return _make_hydrostatic_gwd(gwd_config, dt)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:108:        return _make_nonhydrostatic_gwd(gwd_config, dt)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:110:        return _make_spectral_pe_gwd(gwd_config, dt)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:123:            "MPAS with gravity_wave_drag='none'."
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:136:def _make_hydrostatic_gwd(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:137:    gwd_config: GravityWaveDragConfig,
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:146:    from ``phys_state.gwd_spectrum`` and the updated spectrum is
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:149:    scheme_name, gwd_fn, scheme_config = _get_gwd_fn(gwd_config)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:160:        gwd_spectrum_out = None
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:187:        if gwd_fn is None:
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:189:                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_gwd", dims=dims_3d, units="m/s^2"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:190:                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_gwd", dims=dims_3d, units="m/s^2"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:191:                dT_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dT_dt_gwd", dims=dims_3d, units="K/s"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:192:                dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dp_s_dt_gwd", dims=dims_2d, units="Pa/s"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:193:                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dphis_dt_gwd", dims=dims_2d, units="m^2/s^3"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:195:            return tendencies, gwd_spectrum_out
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:208:            # propagation in ``prognostic_spectral_gwd`` builds
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:215:                spec_in = phys_state.gwd_spectrum
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:224:            gwd_out, spec_new = gwd_fn(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:228:            gwd_spectrum_out = spec_new
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:237:            gwd_out = gwd_fn(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:243:            gwd_out = gwd_fn(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:248:        du_dt = gwd_out.du_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:249:        dv_dt = gwd_out.dv_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:250:        dT_dt = gwd_out.dT_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:253:            du_dt=Field(data=du_dt, name="du_dt_gwd", dims=dims_3d, units="m/s^2"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:254:            dv_dt=Field(data=dv_dt, name="dv_dt_gwd", dims=dims_3d, units="m/s^2"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:255:            dT_dt=Field(data=dT_dt, name="dT_dt_gwd", dims=dims_3d, units="K/s"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:256:            dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dp_s_dt_gwd", dims=dims_2d, units="Pa/s"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:257:            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dphis_dt_gwd", dims=dims_2d, units="m^2/s^3"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:259:        return tendencies, gwd_spectrum_out
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:277:def _make_nonhydrostatic_gwd(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:278:    gwd_config: GravityWaveDragConfig,
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:286:    scheme_name, gwd_fn, scheme_config = _get_gwd_fn(gwd_config)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:298:        gwd_spectrum_out = None
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:333:        if gwd_fn is None:
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:335:                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_gwd", dims=dims_3d, units="m/s^2"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:336:                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_gwd", dims=dims_3d, units="m/s^2"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:337:                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_gwd", dims=dims_w, units="m/s^2"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:338:                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dtheta_prime_dt_gwd", dims=dims_3d, units="K/s"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:339:                drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_gwd", dims=dims_3d, units="kg/m^3/s"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:340:                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_gwd", dims=dims_2d, units="m^2/s^3"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:341:                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_gwd", dims=dims_tr, units="1/s"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:343:            return tendencies, gwd_spectrum_out
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:365:                spec_in = phys_state.gwd_spectrum
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:374:            gwd_out, spec_new = gwd_fn(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:378:            gwd_spectrum_out = spec_new
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:387:            gwd_out = gwd_fn(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:393:            gwd_out = gwd_fn(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:398:        du_dt = gwd_out.du_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:399:        dv_dt = gwd_out.dv_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:400:        dT_dt = gwd_out.dT_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:404:            du_dt=Field(data=du_dt, name="du_dt_gwd", dims=dims_3d, units="m/s^2"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:405:            dv_dt=Field(data=dv_dt, name="dv_dt_gwd", dims=dims_3d, units="m/s^2"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:406:            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_gwd", dims=dims_w, units="m/s^2"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:407:            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_gwd", dims=dims_3d, units="K/s"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:408:            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_gwd", dims=dims_3d, units="kg/m^3/s"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:409:            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_gwd", dims=dims_2d, units="m^2/s^3"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:410:            dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_gwd", dims=dims_tr, units="1/s"),
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:412:        return tendencies, gwd_spectrum_out
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:425:def _make_spectral_pe_gwd(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:426:    gwd_config: GravityWaveDragConfig,
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:434:    scheme_name, gwd_fn, scheme_config = _get_gwd_fn(gwd_config)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:440:        gwd_spectrum_out = None
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:477:        if gwd_fn is None:
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:485:            return tendencies, gwd_spectrum_out
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:496:                spec_in = phys_state.gwd_spectrum
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:505:            gwd_out, spec_new = gwd_fn(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:509:            gwd_spectrum_out = spec_new
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:518:            gwd_out = gwd_fn(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:524:            gwd_out = gwd_fn(
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:529:        du_dt = gwd_out.du_dt.reshape(n_lat, n_lon, nlev)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:530:        dv_dt = gwd_out.dv_dt.reshape(n_lat, n_lon, nlev)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:531:        dT_dt = gwd_out.dT_dt.reshape(n_lat, n_lon, nlev)
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py:560:        return tendencies, gwd_spectrum_out
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:1:"""Smoothed Lindzen (1981) orographic gravity wave drag.
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:4:fully differentiable via jax.lax.scan for the vertical stress profile.
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:8:- Lindzen, R. S. (1981). Turbulence and stress owing to gravity wave and
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:18:from legoesm.atmosphere.physics.gravity_wave_drag.config import LindzenConfig
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:19:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:22:def lindzen_gwd(
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:119:    # Column dissipation (positive-definite: KE lost by the mean flow)
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:120:    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:122:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
tests/test_mpas_conservation.py:276:        # Check all fields finite
tests/test_mpas_conservation.py:277:        all_finite = (
tests/test_mpas_conservation.py:278:            jnp.all(jnp.isfinite(state.u.data))
tests/test_mpas_conservation.py:279:            and jnp.all(jnp.isfinite(state.T.data))
tests/test_mpas_conservation.py:280:            and jnp.all(jnp.isfinite(state.S.data))
tests/test_mpas_conservation.py:281:            and jnp.all(jnp.isfinite(state.eta.data))
tests/test_mpas_conservation.py:283:        print(f"  All fields finite: {bool(all_finite)}")
tests/da/test_control_vector.py:167:    def test_grad_through_round_trip(self):
tests/da/test_control_vector.py:168:        """jax.grad through state_to_control(control_to_state(x)) should work."""
tests/da/test_control_vector.py:177:        grad = jax.grad(f)(x0)
tests/da/test_control_vector.py:178:        assert grad.shape == x0.shape
tests/da/test_control_vector.py:179:        assert jnp.all(jnp.isfinite(grad))
tests/da/test_control_vector.py:181:    def test_identity_jacobian(self):
tests/da/test_control_vector.py:190:        J = jax.jacobian(round_trip)(x0)
tests/core/test_vertical_remap.py:33:        """Top and bottom edges should equal the adjacent cell value."""
tests/core/test_vertical_remap.py:44:        assert jnp.all(jnp.isfinite(q_hat))
tests/core/test_vertical_remap.py:96:    def test_finite_output(self):
tests/core/test_vertical_remap.py:101:        assert jnp.all(jnp.isfinite(q_new))
tests/distributed/test_mpi_driver.py:65:            diff = jnp.max(jnp.abs(result - ref))
tests/distributed/test_mpi_driver.py:66:            assert diff == 0.0, f"4D MPI halo mismatch: max diff = {diff}"
tests/distributed/test_mpi_driver.py:96:            hyperdiff_coeff=0.0,
tests/distributed/test_mpi_driver.py:97:            hyperdiff_ps_coeff=0.0,
tests/distributed/test_mpi_driver.py:130:                        f"MPI {world_size}-rank result differs from "
tests/core/test_weno.py:6:  3. AD correctness via Taylor test (jax.grad through WENO kernels)
tests/core/test_weno.py:159:    """Verify jax.grad through WENO kernels gives correct gradients."""
tests/core/test_weno.py:173:        grad_f = jax.grad(scalar_fn)(x0)
tests/core/test_weno.py:174:        directional_deriv = jnp.dot(grad_f, d)
tests/core/test_weno.py:214:        grad_phi, grad_psi = jax.grad(scalar_fn, argnums=(0, 1))(phi0, psi0)
tests/core/test_weno.py:215:        dd = jnp.dot(grad_phi, d_phi) + jnp.dot(grad_psi, d_psi)
tests/core/test_weno.py:270:    """WENO kernels should produce finite results in float32."""
tests/core/test_weno.py:278:        assert jnp.isfinite(fp), "Left-biased contains NaN/Inf"
tests/core/test_weno.py:279:        assert jnp.isfinite(fm), "Right-biased contains NaN/Inf"
tests/core/test_weno.py:289:        assert jnp.isfinite(fp)
tests/core/test_weno.py:290:        assert jnp.isfinite(fm)
tests/core/test_weno.py:294:    def test_float32_gradient_finite(self, weno_fn, width):
tests/core/test_weno.py:295:        """Gradients through WENO should be finite in float32."""
tests/core/test_weno.py:303:        g = jax.grad(fn)(x)
tests/core/test_weno.py:304:        assert jnp.all(jnp.isfinite(g)), f"Non-finite gradients: {g}"
tests/parallel/test_scaling_operators.py:100:    def test_gradient_x_3d(self, grid, rng):
tests/parallel/test_scaling_operators.py:101:        from legoesm.core.operators_latlon_3d import gradient_x_3d
tests/parallel/test_scaling_operators.py:102:        from legoesm.core.operators_latlon import gradient_x
tests/parallel/test_scaling_operators.py:107:        result = gradient_x_3d(data, grid)
tests/parallel/test_scaling_operators.py:114:            ref_levels.append(gradient_x(f, grid).data)
tests/parallel/test_scaling_operators.py:118:            f"max diff = {jnp.max(jnp.abs(result - ref))}")
tests/parallel/test_scaling_operators.py:120:    def test_gradient_y_3d(self, grid, rng):
tests/parallel/test_scaling_operators.py:121:        from legoesm.core.operators_latlon_3d import gradient_y_3d
tests/parallel/test_scaling_operators.py:122:        from legoesm.core.operators_latlon import gradient_y
tests/parallel/test_scaling_operators.py:127:        result = gradient_y_3d(data, grid)
tests/parallel/test_scaling_operators.py:133:            ref_levels.append(gradient_y(f, grid).data)
tests/parallel/test_scaling_operators.py:222:            f"max diff = {jnp.max(jnp.abs(result - ref))}")
tests/parallel/test_scaling_operators.py:279:    def test_gradient_edge_3d(self, mesh_and_state):
tests/parallel/test_scaling_operators.py:281:            gradient_edge, gradient_edge_3d)
tests/parallel/test_scaling_operators.py:285:        result = gradient_edge_3d(T_3d, mesh)
tests/parallel/test_scaling_operators.py:288:            [gradient_edge(T_3d[:, k], mesh) for k in range(nlev)],
tests/parallel/test_scaling_operators.py:386:        """Run a single tendency eval and check it produces finite output."""
tests/parallel/test_scaling_operators.py:419:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/parallel/test_scaling_operators.py:420:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/parallel/test_scaling_operators.py:421:        assert jnp.all(jnp.isfinite(tend.dp_s_dt.data))
tests/da/test_minimizer.py:17:        def cost_and_grad(x):
tests/da/test_minimizer.py:23:        result = minimize_lbfgs(cost_and_grad, x0, max_iter=50, gtol=1e-8)
tests/da/test_minimizer.py:25:        assert result.grad_norm < 1e-4
tests/da/test_minimizer.py:29:        def cost_and_grad(xy):
tests/da/test_minimizer.py:32:            return f, jax.grad(lambda z: (1-z[0])**2 + 100*(z[1]-z[0]**2)**2)(xy)
tests/da/test_minimizer.py:35:        result = minimize_lbfgs(cost_and_grad, x0, max_iter=200, gtol=1e-5)
tests/da/test_minimizer.py:43:        def cost_and_grad(x):
tests/da/test_minimizer.py:48:        result = jax.jit(lambda x0: minimize_lbfgs(cost_and_grad, x0, max_iter=20))(
tests/da/test_minimizer.py:55:        def cost_and_grad(x):
tests/da/test_minimizer.py:60:        result = minimize_lbfgs(cost_and_grad, x0, max_iter=20)
tests/da/test_minimizer.py:71:        def cost_and_grad(x):
tests/da/test_minimizer.py:77:        result = minimize_cg(cost_and_grad, x0, max_iter=50, gtol=1e-8)
tests/da/test_minimizer.py:86:        def cost_and_grad(x):
tests/da/test_minimizer.py:98:        result = minimize_cg(cost_and_grad, x0, max_iter=20, gtol=1e-6,
tests/da/test_minimizer.py:103:        def cost_and_grad(x):
tests/da/test_minimizer.py:106:        result = jax.jit(lambda x0: minimize_cg(cost_and_grad, x0, max_iter=20))(
tests/sea_ice/validation/test_sea_ice_validation.py:105:        # Create a sharp temperature gradient
tests/sea_ice/validation/test_sea_ice_validation.py:327:        assert jnp.all(jnp.isfinite(new_state.h_ice.data))
tests/sea_ice/validation/test_sea_ice_validation.py:389:            assert jnp.all(jnp.isfinite(s)), f"Slab {name} not finite"
tests/sea_ice/validation/test_sea_ice_validation.py:390:            assert jnp.all(jnp.isfinite(d)), f"Dynamic {name} not finite"
tests/sea_ice/validation/test_sea_ice_validation.py:395:            # Within factor of 2 (generous due to different T_ice evaluation points)
tests/test_smagorinsky_biharmonic_comprehensive.py:151:        grad_u_jax, grad_v_jax = jax.grad(energy_fn, argnums=(0, 1))(u, v)
tests/test_smagorinsky_biharmonic_comprehensive.py:161:        coded_grad_u = -tend_u * area_u_dual[:, jnp.newaxis]
tests/test_smagorinsky_biharmonic_comprehensive.py:162:        coded_grad_v = -tend_v * area_v_dual[:, jnp.newaxis]
tests/test_smagorinsky_biharmonic_comprehensive.py:165:        diff_u_int = jnp.abs(coded_grad_u[:, 1:n_lon] - grad_u_jax[:, 1:n_lon])
tests/test_smagorinsky_biharmonic_comprehensive.py:166:        scale_u = jnp.maximum(jnp.max(jnp.abs(grad_u_jax[:, 1:n_lon])), 1e-30)
tests/test_smagorinsky_biharmonic_comprehensive.py:167:        rel_u = float(jnp.max(diff_u_int) / scale_u)
tests/test_smagorinsky_biharmonic_comprehensive.py:170:        diff_v = jnp.abs(coded_grad_v - grad_v_jax)
tests/test_smagorinsky_biharmonic_comprehensive.py:171:        scale_v = jnp.maximum(jnp.max(jnp.abs(grad_v_jax)), 1e-30)
tests/test_smagorinsky_biharmonic_comprehensive.py:172:        rel_v = float(jnp.max(diff_v) / scale_v)
tests/test_smagorinsky_biharmonic_comprehensive.py:235:        grad_u_jax, grad_v_jax = jax.grad(energy_nonperiodic, argnums=(0, 1))(u, v)
tests/test_smagorinsky_biharmonic_comprehensive.py:242:        coded_grad_u = -tend_u * area_u_dual[:, jnp.newaxis]
tests/test_smagorinsky_biharmonic_comprehensive.py:243:        coded_grad_v = -tend_v * area_v_dual[:, jnp.newaxis]
tests/test_smagorinsky_biharmonic_comprehensive.py:245:        rel_u = float(jnp.max(jnp.abs(coded_grad_u - grad_u_jax))
tests/test_smagorinsky_biharmonic_comprehensive.py:246:                       / jnp.maximum(jnp.max(jnp.abs(grad_u_jax)), 1e-30))
tests/test_smagorinsky_biharmonic_comprehensive.py:247:        rel_v = float(jnp.max(jnp.abs(coded_grad_v - grad_v_jax))
tests/test_smagorinsky_biharmonic_comprehensive.py:248:                       / jnp.maximum(jnp.max(jnp.abs(grad_v_jax)), 1e-30))
tests/test_smagorinsky_biharmonic_comprehensive.py:304:    max_rel_diff_u = 0.0
tests/test_smagorinsky_biharmonic_comprehensive.py:305:    max_rel_diff_v = 0.0
tests/test_smagorinsky_biharmonic_comprehensive.py:322:        diff_u = jnp.abs(tend_u_st[:, 1:n_lon] - vlap_u[:, 1:n_lon])
tests/test_smagorinsky_biharmonic_comprehensive.py:323:        diff_v = jnp.abs(tend_v_st - vlap_v)
tests/test_smagorinsky_biharmonic_comprehensive.py:327:        rel_u = float(jnp.max(diff_u) / scale_u)
tests/test_smagorinsky_biharmonic_comprehensive.py:328:        rel_v = float(jnp.max(diff_v) / scale_v)
tests/test_smagorinsky_biharmonic_comprehensive.py:329:        max_rel_diff_u = max(max_rel_diff_u, rel_u)
tests/test_smagorinsky_biharmonic_comprehensive.py:330:        max_rel_diff_v = max(max_rel_diff_v, rel_v)
tests/test_smagorinsky_biharmonic_comprehensive.py:334:                f"  seed={seed}: max rel diff u={rel_u:.3e}, v={rel_v:.3e}")
tests/test_smagorinsky_biharmonic_comprehensive.py:336:    # Expected O(1e-3) difference from spherical metric terms
tests/test_smagorinsky_biharmonic_comprehensive.py:337:    result.check(max_rel_diff_u < 0.01,
tests/test_smagorinsky_biharmonic_comprehensive.py:338:        f"u-component: max relative diff = {max_rel_diff_u:.3e} (threshold 0.01)")
tests/test_smagorinsky_biharmonic_comprehensive.py:339:    result.check(max_rel_diff_v < 0.01,
tests/test_smagorinsky_biharmonic_comprehensive.py:340:        f"v-component: max relative diff = {max_rel_diff_v:.3e} (threshold 0.01)")
tests/test_smagorinsky_biharmonic_comprehensive.py:342:    if max_rel_diff_u > 1e-10 or max_rel_diff_v > 1e-10:
tests/test_smagorinsky_biharmonic_comprehensive.py:344:            "NOTE: Stress-tensor and vector Laplacian differ by O(1e-3). "
tests/test_smagorinsky_biharmonic_comprehensive.py:346:            "of the strain-rate, while the vector Laplacian uses grad-div "
tests/test_smagorinsky_biharmonic_comprehensive.py:347:            "minus curl-curl which differs by spherical metric terms.")
tests/test_smagorinsky_biharmonic_comprehensive.py:528:    max_diff_u = 0.0
tests/test_smagorinsky_biharmonic_comprehensive.py:529:    max_diff_v = 0.0
tests/test_smagorinsky_biharmonic_comprehensive.py:539:        diff_u = float(jnp.max(jnp.abs(tend_u_3d[:, :, k] - tend_u_2d)))
tests/test_smagorinsky_biharmonic_comprehensive.py:540:        diff_v = float(jnp.max(jnp.abs(tend_v_3d[:, :, k] - tend_v_2d)))
tests/test_smagorinsky_biharmonic_comprehensive.py:541:        max_diff_u = max(max_diff_u, diff_u)
tests/test_smagorinsky_biharmonic_comprehensive.py:542:        max_diff_v = max(max_diff_v, diff_v)
tests/test_smagorinsky_biharmonic_comprehensive.py:545:            result.info(f"  level {k}: max diff u={diff_u:.3e}, v={diff_v:.3e}")
tests/test_smagorinsky_biharmonic_comprehensive.py:547:    result.check(max_diff_u < 1e-14,
tests/test_smagorinsky_biharmonic_comprehensive.py:548:        f"u 3D-2D max diff = {max_diff_u:.3e} (threshold 1e-14)")
tests/test_smagorinsky_biharmonic_comprehensive.py:549:    result.check(max_diff_v < 1e-14,
tests/test_smagorinsky_biharmonic_comprehensive.py:550:        f"v 3D-2D max diff = {max_diff_v:.3e} (threshold 1e-14)")
tests/unit/test_land_ice_stomata.py:44:        assert jnp.all(jnp.isfinite(vals))
tests/unit/test_land_ice_stomata.py:167:        assert jnp.all(jnp.isfinite(gs))
tests/unit/test_land_ice_stomata.py:168:        assert jnp.all(jnp.isfinite(gpp))
tests/parallel/test_cubesphere_exchange.py:92:                                       err_msg="ppermute 3D differs from local")
tests/parallel/test_cubesphere_exchange.py:112:                                       err_msg="ppermute 4D differs from local")
tests/da/test_cycling.py:71:        assert jnp.all(jnp.isfinite(final.h.data))
tests/da/test_background_error.py:20:    def test_positive_definite(self):
tests/da/test_background_error.py:29:    def test_sqrt_multiply_differentiable(self):
tests/da/test_background_error.py:33:        grad = jax.grad(lambda v: jnp.sum(B.sqrt_multiply(v) ** 2))(x)
tests/da/test_background_error.py:34:        assert jnp.all(jnp.isfinite(grad))
tests/da/test_background_error.py:36:    def test_inv_multiply_differentiable(self):
tests/da/test_background_error.py:40:        grad = jax.grad(lambda v: jnp.sum(B.inv_multiply(v) ** 2))(x)
tests/da/test_background_error.py:41:        assert jnp.all(jnp.isfinite(grad))
tests/da/test_background_error.py:63:    def test_sqrt_multiply_differentiable(self, grid):
tests/da/test_background_error.py:66:        B = DiffusionB(grid, sigma, horizontal_length_scale=1000e3, n_diffusion_iter=4)
tests/da/test_background_error.py:68:        grad = jax.grad(lambda v: jnp.sum(B.sqrt_multiply(v)))(x)
tests/da/test_background_error.py:69:        assert jnp.all(jnp.isfinite(grad))
tests/da/test_background_error.py:75:        B = DiffusionB(grid, sigma, horizontal_length_scale=5000e3, n_diffusion_iter=10)
tests/da/test_background_error.py:96:        assert jnp.all(jnp.diff(vals) <= 1e-6)
tests/stress/test_phase1_sea_ice.py:86:        assert jnp.all(jnp.isfinite(h)), "h_ice contains non-finite values"
tests/stress/test_phase1_sea_ice.py:87:        assert jnp.all(jnp.isfinite(conc)), "concentration contains non-finite values"
tests/stress/test_phase1_sea_ice.py:88:        assert jnp.all(jnp.isfinite(T)), "T_ice contains non-finite values"
tests/stress/test_phase1_sea_ice.py:135:        assert jnp.all(jnp.isfinite(h)), "h_ice contains non-finite values"
tests/stress/test_phase1_sea_ice.py:152:        """Single step: energy quantities should be finite and non-zero,
tests/stress/test_phase1_sea_ice.py:192:        assert jnp.all(jnp.isfinite(E_before)), "E_before not finite"
tests/stress/test_phase1_sea_ice.py:193:        assert jnp.all(jnp.isfinite(E_after)), "E_after not finite"
tests/stress/test_phase1_sea_ice.py:194:        assert jnp.all(jnp.isfinite(dE)), "dE not finite"
tests/stress/test_phase1_sea_ice.py:205:    # 1A.4  EVP dynamics produce bounded, finite velocities
tests/stress/test_phase1_sea_ice.py:208:        """EVP solver should produce finite, bounded ice velocities and stresses."""
tests/stress/test_phase1_sea_ice.py:238:        # All outputs should be finite
tests/stress/test_phase1_sea_ice.py:239:        assert jnp.all(jnp.isfinite(u_out)), "u_ice has non-finite values"
tests/stress/test_phase1_sea_ice.py:240:        assert jnp.all(jnp.isfinite(v_out)), "v_ice has non-finite values"
tests/stress/test_phase1_sea_ice.py:241:        assert jnp.all(jnp.isfinite(s11_out)), "sigma_11 has non-finite values"
tests/stress/test_phase1_sea_ice.py:242:        assert jnp.all(jnp.isfinite(s22_out)), "sigma_22 has non-finite values"
tests/stress/test_phase1_sea_ice.py:243:        assert jnp.all(jnp.isfinite(s12_out)), "sigma_12 has non-finite values"
tests/da/test_gen_be.py:5:2. sqrt_multiply returns a finite array with increased spatial correlation.
tests/da/test_gen_be.py:7:4. sqrt_multiply is differentiable (jax.grad passes).
tests/da/test_gen_be.py:8:5. inv_multiply is differentiable.
tests/da/test_gen_be.py:98:    def test_params_finite(self):
tests/da/test_gen_be.py:105:        assert jnp.all(jnp.isfinite(params.vert_eig_vec)), "eig_vec has non-finite values"
tests/da/test_gen_be.py:106:        assert jnp.all(jnp.isfinite(params.vert_eig_val)), "eig_val has non-finite values"
tests/da/test_gen_be.py:107:        assert jnp.all(jnp.isfinite(params.reg_coeff)), "reg_coeff has non-finite values"
tests/da/test_gen_be.py:108:        assert jnp.all(jnp.isfinite(params.len_scale)), "len_scale has non-finite values"
tests/da/test_gen_be.py:109:        assert jnp.isfinite(params.std_ps), "std_ps is not finite"
tests/da/test_gen_be.py:183:        transform = GenBETransform(params, spec, grid, n_diffusion_iter=5)
tests/da/test_gen_be.py:186:    def test_sqrt_multiply_finite(self):
tests/da/test_gen_be.py:187:        """sqrt_multiply returns a finite array."""
tests/da/test_gen_be.py:195:        assert jnp.all(jnp.isfinite(out)), "sqrt_multiply output has non-finite values"
tests/da/test_gen_be.py:207:    def test_inv_multiply_finite(self):
tests/da/test_gen_be.py:208:        """inv_multiply returns a finite array."""
tests/da/test_gen_be.py:214:        assert jnp.all(jnp.isfinite(out)), "inv_multiply output has non-finite values"
tests/da/test_gen_be.py:216:    def test_sqrt_multiply_differentiable(self):
tests/da/test_gen_be.py:217:        """jax.grad passes through sqrt_multiply."""
tests/da/test_gen_be.py:224:        grad = jax.grad(loss)(v0)
tests/da/test_gen_be.py:225:        assert jnp.all(jnp.isfinite(grad)), "grad through sqrt_multiply has non-finite values"
tests/da/test_gen_be.py:227:    def test_inv_multiply_differentiable(self):
tests/da/test_gen_be.py:228:        """jax.grad passes through inv_multiply."""
tests/da/test_gen_be.py:235:        grad = jax.grad(loss)(x0)
tests/da/test_gen_be.py:236:        assert jnp.all(jnp.isfinite(grad)), "grad through inv_multiply has non-finite values"
tests/da/test_gen_be.py:239:        """B^{-1} (B^{1/2} v) is finite and non-trivially non-zero.
tests/da/test_gen_be.py:243:        We only verify finiteness and non-degeneracy.
tests/da/test_gen_be.py:250:        assert jnp.all(jnp.isfinite(BinvBv)), "B^{-1} B^{1/2} v has non-finite values"
tests/da/test_gen_be.py:256:    def test_b_positive_definite(self):
tests/da/test_gen_be.py:257:        """<v, B^{-1} B^{1/2} v> > 0 for non-zero v (positive definiteness check)."""
tests/da/test_gen_be.py:334:        transform = GenBETransform(params, spec, grid, n_diffusion_iter=3)
tests/da/test_gen_be.py:338:        assert jnp.all(jnp.isfinite(out)), "sqrt_multiply failed with v=None state"
tests/da/test_observation.py:44:    def test_differentiable(self):
tests/da/test_observation.py:53:        grad = jax.grad(f)(state.h.data)
tests/da/test_observation.py:54:        assert jnp.all(jnp.isfinite(grad))
tests/da/test_observation.py:67:    def test_differentiable(self, latlon_grid):
tests/da/test_observation.py:77:        grad = jax.grad(f)(state.h.data)
tests/da/test_observation.py:78:        assert jnp.all(jnp.isfinite(grad))
tests/da/test_observation.py:103:    def test_differentiable(self):
tests/da/test_observation.py:115:        grad = jax.grad(f)(state.h.data)
tests/da/test_observation.py:116:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_bechtold.py:5:* tendency shape / dtype / finiteness;
tests/unit/test_bechtold.py:16:* finite gradients through ``epsilon_deep``, ``cape_pbl_depth``,
tests/unit/test_bechtold.py:44:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/unit/test_bechtold.py:73:# Shape / finiteness
tests/unit/test_bechtold.py:76:def test_bechtold_shape_finiteness():
tests/unit/test_bechtold.py:89:        assert jnp.all(jnp.isfinite(arr))
tests/unit/test_bechtold.py:111:    # The two diagnoses differ by some non-trivial amount.
tests/unit/test_bechtold.py:161:    """With ``enable_stochastic=True``, two different PRNG keys produce
tests/unit/test_bechtold.py:162:    different AR1 noise states and different diagnosed mass fluxes.
tests/unit/test_bechtold.py:214:def test_bechtold_grad_through_epsilon_deep():
tests/unit/test_bechtold.py:227:    g = float(jax.grad(f)(jnp.asarray(1.75e-3)))
tests/unit/test_bechtold.py:228:    assert bool(jnp.isfinite(g))
tests/unit/test_bechtold.py:231:def test_bechtold_grad_through_cape_pbl_depth():
tests/unit/test_bechtold.py:244:    g = float(jax.grad(f)(jnp.asarray(500.0)))
tests/unit/test_bechtold.py:245:    assert bool(jnp.isfinite(g))
tests/unit/test_bechtold.py:248:def test_bechtold_grad_through_stochastic_amplitude_when_off():
tests/unit/test_bechtold.py:249:    """Even when ``enable_stochastic=False``, gradient through
tests/unit/test_bechtold.py:250:    ``stochastic_amplitude`` is finite (it's a static config field
tests/unit/test_bechtold.py:264:    g = float(jax.grad(f)(jnp.asarray(0.5)))
tests/unit/test_bechtold.py:265:    assert bool(jnp.isfinite(g))
tests/unit/test_bechtold.py:266:    # In the off branch the gradient is 0 (parameter unused) — that's
tests/unit/test_bechtold.py:267:    # fine, just must be finite.
tests/unit/test_bechtold.py:282:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_bechtold.py:295:def test_bechtold_orchestrator_one_step_finite():
tests/unit/test_bechtold.py:311:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_bechtold.py:320:        assert jnp.all(jnp.isfinite(f.data))
tests/unit/test_bechtold.py:328:    """When ``enable_stochastic=True``, two PhysicsStates with different
tests/unit/test_bechtold.py:329:    master PRNG keys produce different conv_stoch_state outputs after
tests/unit/test_bechtold.py:352:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_bechtold.py:365:    # Different seeds → different stochastic state.
tests/unit/test_bechtold.py:367:        "Two different PRNG seeds should produce different AR1 noise"
tests/unit/test_bechtold.py:381:def test_bechtold_orchestrator_grad_through_phys_state():
tests/unit/test_bechtold.py:382:    """jax.grad through the orchestrator with stochastic Bechtold
tests/unit/test_bechtold.py:383:    succeeds — the AR1 perturbation does not break differentiability of
tests/unit/test_bechtold.py:385:    fixed factor at the time of differentiation)."""
tests/unit/test_bechtold.py:406:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_bechtold.py:418:    g = jax.grad(loss)(jnp.array(1.0))
tests/unit/test_bechtold.py:419:    assert bool(jnp.isfinite(g))
tests/unit/test_bechtold.py:444:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_bechtold.py:490:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_bechtold.py:511:    # Tendencies are finite.
tests/unit/test_bechtold.py:513:        assert jnp.all(jnp.isfinite(f.data))
tests/unit/test_physics_combined.py:19:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/unit/test_physics_combined.py:62:        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
tests/unit/test_physics_combined.py:68:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), "dT_dt NaN"
tests/unit/test_physics_combined.py:69:    assert jnp.all(jnp.isfinite(tend.du_dt.data)), "du_dt NaN"
tests/unit/test_physics_combined.py:70:    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), "dv_dt NaN"
tests/unit/test_physics_combined.py:88:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_physics_combined.py:99:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_physics_combined.py:104:    cfg_gwd = PhysicsConfig(
tests/unit/test_physics_combined.py:109:        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
tests/unit/test_physics_combined.py:111:    gwd_fn = make_physics(cfg_gwd, model_type="hydrostatic", dt=dt)
tests/unit/test_physics_combined.py:112:    tend_gwd, _ = gwd_fn(state, grid, sigma)
tests/unit/test_physics_combined.py:120:        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
tests/unit/test_physics_combined.py:127:    sum_dT = tend_rad.dT_dt.data + tend_turb.dT_dt.data + tend_gwd.dT_dt.data
tests/unit/test_physics_combined.py:128:    sum_du = tend_rad.du_dt.data + tend_turb.du_dt.data + tend_gwd.du_dt.data
tests/unit/test_physics_combined.py:143:    """Gray radiation + each convection scheme should produce finite output."""
tests/unit/test_physics_combined.py:150:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_physics_combined.py:155:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"NaN with gray+{conv}"
tests/unit/test_physics_combined.py:156:    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"NaN with gray+{conv}"
tests/unit/test_physics_combined.py:193:        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
tests/unit/test_physics_combined.py:220:    assert jnp.all(jnp.isfinite(T_data)), "T has NaN/Inf after 20 steps"
tests/unit/test_physics_combined.py:221:    assert jnp.all(jnp.isfinite(state.u.data)), "u has NaN/Inf after 20 steps"
tests/stress/test_phase0_infrastructure.py:53:            assert math.isfinite(co2), f"CO2 not finite at year {y}"
tests/stress/test_phase0_infrastructure.py:54:            assert math.isfinite(ch4), f"CH4 not finite at year {y}"
tests/stress/test_phase0_infrastructure.py:55:            assert math.isfinite(n2o), f"N2O not finite at year {y}"
tests/stress/test_phase0_infrastructure.py:67:            assert math.isfinite(co2), f"CO2 not finite at year {y}"
tests/stress/test_phase0_infrastructure.py:68:            assert math.isfinite(ch4), f"CH4 not finite at year {y}"
tests/stress/test_phase0_infrastructure.py:69:            assert math.isfinite(n2o), f"N2O not finite at year {y}"
tests/stress/test_phase0_infrastructure.py:81:            assert math.isfinite(co2), f"CO2 not finite at year {y}"
tests/stress/test_phase0_infrastructure.py:82:            assert math.isfinite(ch4), f"CH4 not finite at year {y}"
tests/stress/test_phase0_infrastructure.py:83:            assert math.isfinite(n2o), f"N2O not finite at year {y}"
tests/stress/test_phase0_infrastructure.py:126:        assert cfg.carbon_land == "differland"
tests/distributed/test_voronoi_halo.py:22:    gradient_edge,
tests/distributed/test_voronoi_halo.py:213:    def test_gradient(self, mesh, n_ranks):
tests/distributed/test_voronoi_halo.py:216:        grad_global = gradient_edge(phi_global, mesh)
tests/distributed/test_voronoi_halo.py:222:            grad_local = gradient_edge(phi_local, local_mesh)
tests/distributed/test_voronoi_halo.py:227:                    float(grad_local[i]), float(grad_global[g]),
tests/unit/test_sfno.py:120:        """Output should differ from input (non-trivial transform) but
tests/unit/test_sfno.py:180:        diff = jnp.mean(jnp.abs(out - x))
tests/unit/test_sfno.py:182:        assert diff < 10.0
tests/unit/test_sfno.py:185:        """Without residual prediction, output can differ significantly."""
tests/unit/test_sfno.py:199:    def test_gradient_flow(self, grid_t10, key):
tests/unit/test_sfno.py:215:        grads = eqx.filter_grad(loss_fn)(model)
tests/unit/test_sfno.py:217:        # Check encoder gradient exists and has no NaNs
tests/unit/test_sfno.py:218:        assert grads.encoder.weight is not None
tests/unit/test_sfno.py:219:        assert not jnp.any(jnp.isnan(grads.encoder.weight))
tests/unit/test_sfno.py:221:        # Check decoder gradient
tests/unit/test_sfno.py:222:        assert grads.decoder.weight is not None
tests/unit/test_sfno.py:223:        assert not jnp.any(jnp.isnan(grads.decoder.weight))
tests/unit/test_sfno.py:225:    def test_different_in_out_channels(self, grid_t10, key):
tests/unit/test_sfno.py:226:        """SFNO should work with different input/output channel counts."""
tests/unit/test_sfno.py:254:        assert jnp.all(jnp.isfinite(out))
tests/sea_ice/unit/test_surface_albedo.py:13:- JAX differentiability
tests/sea_ice/unit/test_surface_albedo.py:104:        assert jnp.all(jnp.diff(alpha) <= 1e-10)
tests/sea_ice/unit/test_surface_albedo.py:159:    def test_differentiable(self):
tests/sea_ice/unit/test_surface_albedo.py:160:        """jax.grad should work through land_albedo."""
tests/sea_ice/unit/test_surface_albedo.py:167:        grad = jax.grad(loss)(jnp.array([20.0, 40.0]))
tests/sea_ice/unit/test_surface_albedo.py:168:        assert jnp.all(jnp.isfinite(grad))
tests/sea_ice/unit/test_surface_albedo.py:202:        assert jnp.all(jnp.diff(alpha) <= 1e-10)
tests/sea_ice/unit/test_surface_albedo.py:204:    def test_differentiable(self):
tests/sea_ice/unit/test_surface_albedo.py:205:        """jax.grad should work through ice_albedo."""
tests/sea_ice/unit/test_surface_albedo.py:211:        grad = jax.grad(loss)(jnp.array([265.0, 270.0]))
tests/sea_ice/unit/test_surface_albedo.py:212:        assert jnp.all(jnp.isfinite(grad))
tests/sea_ice/unit/test_surface_albedo.py:244:    def test_differentiable(self):
tests/sea_ice/unit/test_surface_albedo.py:245:        """jax.grad should work through zenith ocean albedo."""
tests/sea_ice/unit/test_surface_albedo.py:251:        grad = jax.grad(loss)(jnp.array([0.3, 0.7]))
tests/sea_ice/unit/test_surface_albedo.py:252:        assert jnp.all(jnp.isfinite(grad))
tests/stress/test_cmip_operationalization.py:87:        # Target: different resolution (36x72 = 5-degree)
tests/stress/test_cmip_operationalization.py:96:        assert np.all(np.isfinite(result))
tests/stress/test_cmip_operationalization.py:114:        assert np.all(np.isfinite(result))
tests/atmosphere/shallow_water/integration/test_shallow_water.py:52:            hyperdiff_coeff=1e15,
tests/atmosphere/shallow_water/integration/test_shallow_water.py:66:    def test_tendencies_finite(self, grid, cdgrid):
tests/atmosphere/shallow_water/integration/test_shallow_water.py:67:        """All tendencies should be finite."""
tests/atmosphere/shallow_water/integration/test_shallow_water.py:71:        assert jnp.all(jnp.isfinite(dh))
tests/atmosphere/shallow_water/integration/test_shallow_water.py:72:        assert jnp.all(jnp.isfinite(du))
tests/atmosphere/shallow_water/integration/test_shallow_water.py:74:    def test_single_step_finite(self, model, grid, cdgrid):
tests/atmosphere/shallow_water/integration/test_shallow_water.py:75:        """A single time step should produce finite values."""
tests/atmosphere/shallow_water/integration/test_shallow_water.py:79:        assert jnp.all(jnp.isfinite(state_new.h))
tests/atmosphere/shallow_water/integration/test_shallow_water.py:87:        assert jnp.all(jnp.isfinite(state.h))
tests/atmosphere/shallow_water/integration/test_shallow_water.py:135:    """Tests for end-to-end differentiability."""
tests/atmosphere/shallow_water/integration/test_shallow_water.py:145:    def test_grad_through_single_step(self, grid, cdgrid):
tests/atmosphere/shallow_water/integration/test_shallow_water.py:146:        """jax.grad should work through a single model step."""
tests/atmosphere/shallow_water/integration/test_shallow_water.py:158:        grads = jax.grad(loss)(state.h)
tests/atmosphere/shallow_water/integration/test_shallow_water.py:159:        assert jnp.all(jnp.isfinite(grads))
tests/atmosphere/shallow_water/integration/test_shallow_water.py:160:        assert not jnp.allclose(grads, 0.0)
tests/atmosphere/shallow_water/integration/test_shallow_water.py:162:    def test_grad_through_multi_step(self, grid, cdgrid):
tests/atmosphere/shallow_water/integration/test_shallow_water.py:163:        """jax.grad should work through multiple time steps."""
tests/atmosphere/shallow_water/integration/test_shallow_water.py:176:        grads = jax.grad(loss)(state.h)
tests/atmosphere/shallow_water/integration/test_shallow_water.py:177:        assert jnp.all(jnp.isfinite(grads))
tests/atmosphere/shallow_water/integration/test_shallow_water.py:179:    def test_grad_through_scan(self, grid, cdgrid):
tests/atmosphere/shallow_water/integration/test_shallow_water.py:180:        """jax.grad should work through lax.scan integration."""
tests/atmosphere/shallow_water/integration/test_shallow_water.py:191:        grads = jax.grad(loss)(state.h)
tests/atmosphere/shallow_water/integration/test_shallow_water.py:192:        assert jnp.all(jnp.isfinite(grads))
tests/ocean/run_ocean_all_grids_matrix.py:5:horizontal resolution relative to baseline defaults, and runs finite-volume and
tests/ocean/run_ocean_all_grids_matrix.py:10:    inertia_gravity_wave, lock_exchange, overflow, stommel_gyre_tracer
tests/ocean/run_ocean_all_grids_matrix.py:36:    "inertia_gravity_wave",
tests/ocean/run_ocean_all_grids_matrix.py:45:    "gravity_wave",
tests/ocean/run_ocean_all_grids_matrix.py:75:    nonfinite = [name for name, m in cases.items() if not bool(m.get("all_finite", False))]
tests/ocean/run_ocean_all_grids_matrix.py:76:    if nonfinite:
tests/ocean/run_ocean_all_grids_matrix.py:77:        return False, f"nonfinite_cases={nonfinite}"
tests/ocean/run_ocean_all_grids_matrix.py:88:    nonfinite = [name for name, m in cases.items() if not bool(m.get("all_finite", False))]
tests/ocean/run_ocean_all_grids_matrix.py:89:    if nonfinite:
tests/ocean/run_ocean_all_grids_matrix.py:90:        return False, f"nonfinite_cases={nonfinite}"
tests/ocean/run_ocean_all_grids_matrix.py:105:        finite = bool(run.get("all_finite", False))
tests/ocean/run_ocean_all_grids_matrix.py:106:        if (not stable) or (not finite):
tests/ocean/run_ocean_all_grids_matrix.py:240:        default="finite_volume,fc_gram",
tests/ocean/run_ocean_all_grids_matrix.py:382:    # Native lat-lon finite-volume branch (dedicated FV test suite).
tests/ocean/run_ocean_all_grids_matrix.py:385:        out_dir = out_root / "lat_lon" / "native_finite_volume" / "pytest"
tests/ocean/run_ocean_all_grids_matrix.py:399:                label="lat_lon/native_finite_volume",
tests/ocean/run_ocean_all_grids_matrix.py:455:        out_dir = out_root / "icosahedral" / "finite_volume"
tests/ocean/run_ocean_all_grids_matrix.py:471:                label="icosahedral/finite_volume",
tests/ocean/run_ocean_all_grids_matrix.py:494:                "native finite-volume lat-lon branch runs tests/ocean/test_latlon_ocean.py "
tests/da/test_preconditioning.py:41:        # Find minimizers (using JAX grad)
tests/da/test_preconditioning.py:45:        J_vg = jax.value_and_grad(J)
tests/da/test_preconditioning.py:49:        Jt_vg = jax.value_and_grad(J_tilde)
tests/da/test_preconditioning.py:55:    def test_differentiable(self):
tests/da/test_preconditioning.py:56:        """Preconditioned cost should be differentiable."""
tests/da/test_preconditioning.py:66:        grad = jax.grad(J_tilde)(v)
tests/da/test_preconditioning.py:67:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_diff_coupler.py:18:def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
tests/unit/test_diff_coupler.py:19:    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
tests/unit/test_diff_coupler.py:20:    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
tests/unit/test_diff_coupler.py:34:    def test_grad_wrt_T_sfc(self, scheme, n_iter):
tests/unit/test_diff_coupler.py:53:        grad = jax.grad(loss)(T_sfc)
tests/unit/test_diff_coupler.py:54:        assert_gradient_ok(grad, f"MOST({scheme}, n_iter={n_iter}) w.r.t. T_sfc")
tests/unit/test_diff_coupler.py:63:    def test_grad_wrt_ice_concentration(self):
tests/unit/test_diff_coupler.py:104:        grad = jax.grad(loss)(ice_conc)
tests/unit/test_diff_coupler.py:105:        assert_gradient_ok(grad, "Tile blending w.r.t. ice_concentration")
tests/unit/test_diff_coupler.py:182:    def test_grad_wrt_sst(self):
tests/unit/test_diff_coupler.py:196:        grad = jax.grad(loss)(self.ocean_sst)
tests/unit/test_diff_coupler.py:197:        assert_gradient_ok(grad, "Full coupler w.r.t. SST")
tests/unit/test_diff_coupler.py:201:# 6d  Flux accumulator differentiability
tests/unit/test_diff_coupler.py:206:    def test_accumulate_and_mean_grad(self):
tests/unit/test_diff_coupler.py:243:        grad = jax.grad(loss)(T_sfc)
tests/unit/test_diff_coupler.py:244:        assert_gradient_ok(grad, "Flux accumulator w.r.t. T_surface")
tests/unit/test_diff_coupler.py:253:    def test_grad_T_atm_to_shflx(self):
tests/unit/test_diff_coupler.py:273:        grad = jax.grad(loss)(T_atm)
tests/unit/test_diff_coupler.py:274:        assert_gradient_ok(grad, "Cross-component T_atm -> shflx")
tests/unit/test_diff_coupler.py:278:# 6f  Lake model differentiability
tests/unit/test_diff_coupler.py:283:    def test_grad_wrt_T_epi(self):
tests/unit/test_diff_coupler.py:301:        grad = jax.grad(loss)(state.T_epi.data)
tests/unit/test_diff_coupler.py:302:        assert_gradient_ok(grad, "Lake model w.r.t. T_epi")
tests/unit/test_diff_coupler.py:304:    def test_grad_wrt_sw_down(self):
tests/unit/test_diff_coupler.py:322:        grad = jax.grad(loss)(forcing.sw_down)
tests/unit/test_diff_coupler.py:323:        assert_gradient_ok(grad, "Lake model w.r.t. sw_down")
tests/distributed/test_mpi_differentiability.py:1:"""MPI reverse-mode AD (gradient) tests.
tests/distributed/test_mpi_differentiability.py:4:    mpirun -np 2 python -m pytest tests/distributed/test_mpi_differentiability.py -v
tests/distributed/test_mpi_differentiability.py:5:    mpirun -np 3 python -m pytest tests/distributed/test_mpi_differentiability.py -v
tests/distributed/test_mpi_differentiability.py:6:    mpirun -np 6 python -m pytest tests/distributed/test_mpi_differentiability.py -v
tests/distributed/test_mpi_differentiability.py:8:Verifies that ``jax.grad`` flows correctly through:
tests/distributed/test_mpi_differentiability.py:10:- ``pad_halo`` with MPI backend (sendrecv with custom_vjp)
tests/distributed/test_mpi_differentiability.py:15:- tests/unit/test_halo.py (test_grad_compatible)
tests/distributed/test_mpi_differentiability.py:16:- tests/unit/test_scale_global_reductions.py (test_global_integral_differentiable)
tests/distributed/test_mpi_differentiability.py:17:- tests/validation/test_differentiability_regression.py
tests/distributed/test_mpi_differentiability.py:55:    def test_grad_scalar(self):
tests/distributed/test_mpi_differentiability.py:61:        g = jax.grad(f)(x)
tests/distributed/test_mpi_differentiability.py:64:    def test_grad_nonzero(self):
tests/distributed/test_mpi_differentiability.py:71:        g = jax.grad(f)(x)
tests/distributed/test_mpi_differentiability.py:72:        assert jnp.all(jnp.isfinite(g))
tests/distributed/test_mpi_differentiability.py:80:    """Gradient through MPI halo exchange (sendrecv with custom_vjp)."""
tests/distributed/test_mpi_differentiability.py:82:    def test_pad_halo_mpi_grad_finite(self, topology):
tests/distributed/test_mpi_differentiability.py:83:        """jax.grad through pad_halo with MPI backend produces finite results."""
tests/distributed/test_mpi_differentiability.py:92:        g = jax.grad(loss)(data)
tests/distributed/test_mpi_differentiability.py:94:        assert jnp.all(jnp.isfinite(g))
tests/distributed/test_mpi_differentiability.py:97:    def test_pad_halo_mpi_grad_matches_local(self, topology):
tests/distributed/test_mpi_differentiability.py:98:        """MPI gradient matches local-backend gradient on rank 0."""
tests/distributed/test_mpi_differentiability.py:109:        # Local gradient (reference)
tests/distributed/test_mpi_differentiability.py:111:        g_local = jax.grad(loss)(data)
tests/distributed/test_mpi_differentiability.py:113:        # MPI gradient
tests/distributed/test_mpi_differentiability.py:115:        g_mpi = jax.grad(loss)(data)
tests/distributed/test_mpi_differentiability.py:121:                err_msg="MPI halo gradient does not match local gradient",
tests/distributed/test_mpi_differentiability.py:124:    def test_pad_halo_4d_mpi_grad_finite(self, topology):
tests/distributed/test_mpi_differentiability.py:125:        """jax.grad through pad_halo_4d with MPI backend produces finite results."""
tests/distributed/test_mpi_differentiability.py:134:        g = jax.grad(loss)(data)
tests/distributed/test_mpi_differentiability.py:136:        assert jnp.all(jnp.isfinite(g))
tests/distributed/test_mpi_differentiability.py:139:    def test_pad_halo_4d_mpi_grad_matches_local(self, topology):
tests/distributed/test_mpi_differentiability.py:140:        """4D MPI gradient matches local-backend gradient on rank 0."""
tests/distributed/test_mpi_differentiability.py:152:        g_local = jax.grad(loss)(data)
tests/distributed/test_mpi_differentiability.py:155:        g_mpi = jax.grad(loss)(data)
tests/distributed/test_mpi_differentiability.py:161:                err_msg="4D MPI halo gradient does not match local gradient",
tests/distributed/test_mpi_differentiability.py:168:    def test_fix_mass_mpi_grad(self, topology):
tests/distributed/test_mpi_differentiability.py:169:        """jax.grad flows through fix_mass_hydrostatic with MPI backend."""
tests/distributed/test_mpi_differentiability.py:182:        g = jax.grad(loss)(p_s)
tests/distributed/test_mpi_differentiability.py:184:        assert jnp.all(jnp.isfinite(g))
tests/unit/test_land_ice_slab_land.py:4:  - finite outputs and physical bounds
tests/unit/test_land_ice_slab_land.py:101:# 1a  Smoke test -- step_land runs and returns finite outputs
tests/unit/test_land_ice_slab_land.py:106:    def test_returns_finite(self):
tests/unit/test_land_ice_slab_land.py:114:            assert jnp.all(jnp.isfinite(arr)), f"LandState.{name} has non-finite values"
tests/unit/test_land_ice_slab_land.py:119:            assert jnp.all(jnp.isfinite(arr)), f"TileResponse.{name} has non-finite values"
tests/unit/test_land_ice_slab_land.py:339:        assert jnp.all(jnp.isfinite(T_arr)), "NaN in T history"
tests/da/integration/test_end_to_end.py:5:- Verify: fully differentiable (jax.grad through cost)
tests/da/integration/test_end_to_end.py:94:        assert jnp.all(jnp.isfinite(final.h.data))
tests/da/integration/test_end_to_end.py:101:    def test_grad_through_cost(self):
tests/da/integration/test_end_to_end.py:102:        """jax.grad through a single-window cost should work."""
tests/da/integration/test_end_to_end.py:124:        # jax.grad should work
tests/da/integration/test_end_to_end.py:125:        g = jax.grad(cost_fn)(x_b)
tests/da/integration/test_end_to_end.py:126:        assert jnp.all(jnp.isfinite(g))
tests/unit/test_vector_calculus_identities.py:4:div, grad, curl must satisfy:
tests/unit/test_vector_calculus_identities.py:5:  - curl(grad(phi)) = 0
tests/unit/test_vector_calculus_identities.py:7:  - Laplacian = div(grad)
tests/unit/test_vector_calculus_identities.py:9:  - Null-space: grad(const) = 0, Laplacian(const) = 0, div(solid-body) = 0
tests/unit/test_vector_calculus_identities.py:99:    """grad(const) = 0, Laplacian(const) = 0, div(solid-body) = 0."""
tests/unit/test_vector_calculus_identities.py:101:    # --- gradient of constant = 0 ---
tests/unit/test_vector_calculus_identities.py:103:    def test_grad_constant_cubesphere(self, cs_grid):
tests/unit/test_vector_calculus_identities.py:105:        from legoesm.core.operators import gradient_x, gradient_y
tests/unit/test_vector_calculus_identities.py:107:        gx = gradient_x(phi, cs_grid)
tests/unit/test_vector_calculus_identities.py:108:        gy = gradient_y(phi, cs_grid)
tests/unit/test_vector_calculus_identities.py:112:    def test_grad_constant_latlon(self, ll_grid):
tests/unit/test_vector_calculus_identities.py:114:        from legoesm.core.operators_latlon import gradient_x, gradient_y
tests/unit/test_vector_calculus_identities.py:116:        gx = gradient_x(phi, ll_grid)
tests/unit/test_vector_calculus_identities.py:117:        gy = gradient_y(phi, ll_grid)
tests/unit/test_vector_calculus_identities.py:121:    def test_grad_constant_voronoi(self, voronoi_mesh):
tests/unit/test_vector_calculus_identities.py:123:        from legoesm.core.operators_voronoi import gradient_edge
tests/unit/test_vector_calculus_identities.py:125:        grad = gradient_edge(phi, voronoi_mesh)
tests/unit/test_vector_calculus_identities.py:126:        assert float(jnp.max(jnp.abs(grad))) < 1e-10
tests/unit/test_vector_calculus_identities.py:192:    # --- gradient of constant = 0 on Gaussian grid (spectral) ---
tests/unit/test_vector_calculus_identities.py:194:    def test_grad_constant_gaussian(self, gauss_grid):
tests/unit/test_vector_calculus_identities.py:198:        spherical harmonic mode. The spectral gradient of n=0 is zero, so
tests/unit/test_vector_calculus_identities.py:210:        grad_coeffs = jnp.where(mask_n0, 0.0, coeffs)
tests/unit/test_vector_calculus_identities.py:212:        max_grad_coeff = float(jnp.max(jnp.abs(grad_coeffs)))
tests/unit/test_vector_calculus_identities.py:213:        assert max_grad_coeff < 1e-10, (
tests/unit/test_vector_calculus_identities.py:214:            f"Non-zero gradient coefficients for constant field: {max_grad_coeff:.3e}"
tests/unit/test_vector_calculus_identities.py:236:    # --- hyperdiffusion of constant = 0 on cubed-sphere ---
tests/unit/test_vector_calculus_identities.py:238:    def test_hyperdiff_constant_cubesphere(self, cs_grid):
tests/unit/test_vector_calculus_identities.py:239:        """Hyperdiffusion (nabla^4) of a constant field must be zero on cubed-sphere."""
tests/unit/test_vector_calculus_identities.py:240:        from legoesm.core.operators import hyperdiffusion
tests/unit/test_vector_calculus_identities.py:243:        result = hyperdiffusion(phi, cs_grid, coeff)
tests/unit/test_vector_calculus_identities.py:249:    # --- hyperdiffusion of constant = 0 on lat-lon ---
tests/unit/test_vector_calculus_identities.py:251:    def test_hyperdiff_constant_latlon(self, ll_grid):
tests/unit/test_vector_calculus_identities.py:252:        """Hyperdiffusion (nabla^4) of a constant field must be zero on lat-lon."""
tests/unit/test_vector_calculus_identities.py:253:        from legoesm.core.operators_latlon import hyperdiffusion
tests/unit/test_vector_calculus_identities.py:256:        result = hyperdiffusion(phi, ll_grid, coeff)
tests/unit/test_vector_calculus_identities.py:264:# 1b) curl(grad(phi)) = 0
tests/unit/test_vector_calculus_identities.py:268:    """curl(grad(phi)) must vanish for any smooth scalar phi."""
tests/unit/test_vector_calculus_identities.py:270:    def test_curl_grad_cubesphere(self, cs_grid):
tests/unit/test_vector_calculus_identities.py:271:        """curl(grad(phi)) = 0 on cubed-sphere A-grid.
tests/unit/test_vector_calculus_identities.py:277:        from legoesm.core.operators import gradient_x, gradient_y, curl_z
tests/unit/test_vector_calculus_identities.py:279:        gx = gradient_x(phi, cs_grid)
tests/unit/test_vector_calculus_identities.py:280:        gy = gradient_y(phi, cs_grid)
tests/unit/test_vector_calculus_identities.py:283:        # curl(grad) should be near machine precision for smooth fields
tests/unit/test_vector_calculus_identities.py:284:        # The absolute value matters more than scaling by grad/dx
tests/unit/test_vector_calculus_identities.py:285:        assert max_vort < 1e-10, f"|curl(grad)| = {max_vort:.3e} on cubed-sphere"
tests/unit/test_vector_calculus_identities.py:287:    def test_curl_grad_latlon(self, ll_grid):
tests/unit/test_vector_calculus_identities.py:288:        """curl(grad(phi)) = 0 on lat-lon grid."""
tests/unit/test_vector_calculus_identities.py:289:        from legoesm.core.operators_latlon import gradient_x, gradient_y, curl_z
tests/unit/test_vector_calculus_identities.py:291:        gx = gradient_x(phi, ll_grid)
tests/unit/test_vector_calculus_identities.py:292:        gy = gradient_y(phi, ll_grid)
tests/unit/test_vector_calculus_identities.py:295:        grad_scale = float(jnp.max(jnp.abs(gx.data)))
tests/unit/test_vector_calculus_identities.py:297:        assert max_vort < grad_scale / dx_min * 0.5, (
tests/unit/test_vector_calculus_identities.py:298:            f"|curl(grad)| = {max_vort:.3e}, grad_scale/dx = {grad_scale/dx_min:.3e}"
tests/unit/test_vector_calculus_identities.py:301:    def test_curl_grad_voronoi(self, voronoi_mesh):
tests/unit/test_vector_calculus_identities.py:302:        """curl(grad(phi)) = 0 on MPAS Voronoi mesh.
tests/unit/test_vector_calculus_identities.py:304:        grad maps cells -> edges, curl maps edges -> vertices.
tests/unit/test_vector_calculus_identities.py:305:        curl(grad) should vanish exactly (Stokes' theorem on discrete mesh).
tests/unit/test_vector_calculus_identities.py:307:        from legoesm.core.operators_voronoi import gradient_edge, curl_vertex
tests/unit/test_vector_calculus_identities.py:310:        grad = gradient_edge(phi, mesh)
tests/unit/test_vector_calculus_identities.py:311:        curl = curl_vertex(grad, mesh)
tests/unit/test_vector_calculus_identities.py:313:        assert max_curl < 1e-10, f"|curl(grad)| = {max_curl:.3e} on Voronoi mesh"
tests/unit/test_vector_calculus_identities.py:317:# 1c) Laplacian consistency: nabla^2(phi) = div(grad(phi))
tests/unit/test_vector_calculus_identities.py:321:    """The Laplacian computed directly must agree with div(grad)."""
tests/unit/test_vector_calculus_identities.py:323:    def test_lap_eq_divgrad_cubesphere(self, cs_grid):
tests/unit/test_vector_calculus_identities.py:324:        """On cubed-sphere, laplacian() IS div(grad()), so this verifies
tests/unit/test_vector_calculus_identities.py:326:        from legoesm.core.operators import laplacian, laplacian_compact, gradient_x, gradient_y, divergence
tests/unit/test_vector_calculus_identities.py:328:        gx = gradient_x(phi, cs_grid)
tests/unit/test_vector_calculus_identities.py:329:        gy = gradient_y(phi, cs_grid)
tests/unit/test_vector_calculus_identities.py:330:        divgrad = divergence(gx, gy, cs_grid)
tests/unit/test_vector_calculus_identities.py:332:        # These should be identical since laplacian() = div(grad())
tests/unit/test_vector_calculus_identities.py:333:        diff = float(jnp.max(jnp.abs(lap.data - divgrad.data)))
tests/unit/test_vector_calculus_identities.py:334:        assert diff < 1e-12, f"laplacian != div(grad): max diff = {diff:.3e}"
tests/unit/test_vector_calculus_identities.py:336:        # Compact Laplacian uses a different stencil but same sign/magnitude
tests/unit/test_vector_calculus_identities.py:343:    def test_lap_eq_divgrad_latlon(self, ll_grid):
tests/unit/test_vector_calculus_identities.py:344:        """On lat-lon, the laplacian uses compact stencil; check vs div(grad)."""
tests/unit/test_vector_calculus_identities.py:346:            laplacian, gradient_x, gradient_y, divergence,
tests/unit/test_vector_calculus_identities.py:349:        gx = gradient_x(phi, ll_grid)
tests/unit/test_vector_calculus_identities.py:350:        gy = gradient_y(phi, ll_grid)
tests/unit/test_vector_calculus_identities.py:351:        divgrad = divergence(gx, gy, ll_grid)
tests/unit/test_vector_calculus_identities.py:354:        l2_diff = float(jnp.sqrt(jnp.mean((lap.data - divgrad.data) ** 2)))
tests/unit/test_vector_calculus_identities.py:356:        rel = l2_diff / (l2_lap + 1e-30)
tests/unit/test_vector_calculus_identities.py:357:        assert rel < 0.5, f"Laplacian vs div(grad) relative L2 = {rel:.3e}"
tests/unit/test_vector_calculus_identities.py:360:        """On Voronoi mesh, scalar Laplacian = div(grad(phi)).
tests/unit/test_vector_calculus_identities.py:361:        Check it is finite and has zero global integral."""
tests/unit/test_vector_calculus_identities.py:362:        from legoesm.core.operators_voronoi import gradient_edge, divergence_cell
tests/unit/test_vector_calculus_identities.py:365:        grad_phi = gradient_edge(phi, mesh)
tests/unit/test_vector_calculus_identities.py:366:        lap_scalar = divergence_cell(grad_phi, mesh)
tests/unit/test_vector_calculus_identities.py:367:        assert jnp.all(jnp.isfinite(lap_scalar))
tests/unit/test_vector_calculus_identities.py:380:    r"""On a closed sphere, grad and -div are adjoints:
tests/unit/test_vector_calculus_identities.py:381:    \int grad(phi) . F dA = -\int phi * div(F) dA
tests/unit/test_vector_calculus_identities.py:388:    def test_adjoint_grad_div_cubesphere(self, cs_grid):
tests/unit/test_vector_calculus_identities.py:389:        """<grad(phi), F> ~ -<phi, div(F)> on cubed-sphere."""
tests/unit/test_vector_calculus_identities.py:391:            gradient_x, gradient_y, divergence, global_integral,
tests/unit/test_vector_calculus_identities.py:401:        gx = gradient_x(phi_f, cs_grid)
tests/unit/test_vector_calculus_identities.py:402:        gy = gradient_y(phi_f, cs_grid)
tests/unit/test_vector_calculus_identities.py:424:    def test_adjoint_grad_div_latlon(self, ll_grid):
tests/unit/test_vector_calculus_identities.py:425:        """<grad(phi), F> ~ -<phi, div(F)> on lat-lon."""
tests/unit/test_vector_calculus_identities.py:427:            gradient_x, gradient_y, divergence, global_integral,
tests/unit/test_vector_calculus_identities.py:437:        gx = gradient_x(phi_f, ll_grid)
tests/unit/test_vector_calculus_identities.py:438:        gy = gradient_y(phi_f, ll_grid)
tests/unit/test_vector_calculus_identities.py:519:        from legoesm.core.operators import gradient_x, gradient_y, divergence
tests/unit/test_vector_calculus_identities.py:522:        dpsi_dx = gradient_x(psi, cs_grid)
tests/unit/test_vector_calculus_identities.py:523:        dpsi_dy = gradient_y(psi, cs_grid)
tests/unit/test_vector_calculus_identities.py:528:        grad_scale = float(jnp.max(jnp.sqrt(dpsi_dx.data**2 + dpsi_dy.data**2)))
tests/unit/test_vector_calculus_identities.py:530:        rel = max_div / (grad_scale / dx_min + 1e-30)
tests/unit/test_vector_calculus_identities.py:531:        assert rel < 0.5, f"|div(curl-field)| / (grad_scale/dx) = {rel:.3e}"
tests/unit/test_vector_calculus_identities.py:535:        from legoesm.core.operators_latlon import gradient_x, gradient_y, divergence
tests/unit/test_vector_calculus_identities.py:538:        dpsi_dx = gradient_x(psi, ll_grid)
tests/unit/test_vector_calculus_identities.py:539:        dpsi_dy = gradient_y(psi, ll_grid)
tests/unit/test_vector_calculus_identities.py:544:        grad_scale = float(jnp.max(jnp.sqrt(dpsi_dx.data**2 + dpsi_dy.data**2)))
tests/unit/test_vector_calculus_identities.py:546:        rel = max_div / (grad_scale / dx_min + 1e-30)
tests/unit/test_vector_calculus_identities.py:547:        assert rel < 0.5, f"|div(curl-field)| / (grad_scale/dx) = {rel:.3e}"
tests/unit/test_vector_calculus_identities.py:578:        from legoesm.core.operators_voronoi import gradient_edge, divergence_cell
tests/unit/test_vector_calculus_identities.py:581:        grad = gradient_edge(phi, mesh)
tests/unit/test_vector_calculus_identities.py:582:        lap = divergence_cell(grad, mesh)
tests/unit/test_vector_calculus_identities.py:617:        """D-grid vorticity of a gradient field should be ~0.
tests/unit/test_vector_calculus_identities.py:619:        If (u_d, v_d) = grad(phi) at corners, vorticity = curl(grad(phi)) ~ 0.
tests/unit/test_vector_calculus_identities.py:620:        We build phi on corners and take finite differences for the "gradient".
tests/unit/test_vector_calculus_identities.py:626:        # Simple gradient approximation on the D-grid: centered differences
tests/unit/test_vector_calculus_identities.py:654:        # Vorticity should be finite and have a well-defined pattern
tests/unit/test_vector_calculus_identities.py:655:        assert jnp.all(jnp.isfinite(vort)), "Vorticity contains NaN/Inf"
tests/unit/test_vector_calculus_identities.py:664:# Spectral div(curl)=0 and curl(grad)=0
tests/unit/test_vector_calculus_identities.py:673:    def test_spectral_curl_grad_zero(self, gauss_grid):
tests/unit/test_vector_calculus_identities.py:674:        """curl(grad(phi)) = 0 in spectral space.
tests/unit/test_vector_calculus_identities.py:677:        Applying curl after grad should yield zero coefficients.
tests/unit/test_vector_calculus_identities.py:684:        # Spectral gradient: vorticity of a gradient is zero
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:54:            hyperdiff_coeff=1e15,
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:66:    def test_tendencies_finite(self, grid_sw):
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:74:        assert jnp.all(jnp.isfinite(dh))
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:75:        assert jnp.all(jnp.isfinite(du))
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:76:        assert jnp.all(jnp.isfinite(dv))
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:81:        assert jnp.all(jnp.isfinite(s1.h))
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:82:        assert jnp.all(jnp.isfinite(s1.u_d))
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:83:        assert jnp.all(jnp.isfinite(s1.v_d))
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:91:        assert jnp.all(jnp.isfinite(s.h))
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:92:        assert jnp.all(jnp.isfinite(s.u_d))
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:93:        assert jnp.all(jnp.isfinite(s.v_d))
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:120:            hyperdiff_coeff=1e15,
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:130:        assert jnp.all(jnp.isfinite(s.h))
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:132:    def test_differentiable_10_steps(self, grid_sw):
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:133:        """FV SW should be differentiable through 10 steps via scan."""
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:139:            hyperdiff_coeff=1e15,
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:153:        grad_fn = jax.grad(loss_fn)
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:154:        g = grad_fn(state.h)
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py:155:        assert jnp.all(jnp.isfinite(g))
tests/stress/test_phase2_coupled.py:57:    def test_atm_fields_finite(self, driver):
tests/stress/test_phase2_coupled.py:58:        assert jnp.all(jnp.isfinite(driver.state.T.data))
tests/stress/test_phase2_coupled.py:59:        assert jnp.all(jnp.isfinite(driver.state.u.data))
tests/stress/test_phase2_coupled.py:60:        assert jnp.all(jnp.isfinite(driver.state.v.data))
tests/stress/test_phase2_coupled.py:61:        assert jnp.all(jnp.isfinite(driver.state.p_s.data))
tests/stress/test_phase2_coupled.py:112:    def test_surface_fields_finite(self, driver):
tests/stress/test_phase2_coupled.py:119:                assert jnp.all(jnp.isfinite(leaf)), "Non-finite value in land state"
tests/stress/test_phase2_coupled.py:138:    def test_land_state_finite(self, driver):
tests/stress/test_phase2_coupled.py:143:                assert jnp.all(jnp.isfinite(leaf)), "Non-finite in multilayer land"
tests/stress/test_phase2_coupled.py:171:        assert jnp.all(jnp.isfinite(driver._co2_field)), "CO2 field has NaN/inf"
tests/stress/test_phase2_coupled.py:209:    def test_all_fields_finite(self, driver):
tests/stress/test_phase2_coupled.py:210:        assert jnp.all(jnp.isfinite(driver.state.T.data))
tests/stress/test_phase2_coupled.py:211:        assert jnp.all(jnp.isfinite(driver.ocean_state.T_sfc.data))
tests/stress/test_phase2_coupled.py:216:        assert jnp.all(jnp.isfinite(driver._co2_field)), "CO2 field has NaN/inf"
tests/da/integration/test_pe_4dvar.py:25:    """A very simple hydrostatic model: temperature diffuses toward mean."""
tests/distributed/test_halo_mpi.py:93:                f"MPI halo mismatch on owned faces (max diff: "
tests/distributed/test_halo_mpi.py:166:                f"MPI halo=2 mismatch on owned faces (max diff: "
tests/distributed/test_halo_mpi.py:387:        # MPI allreduce uses different FP summation order than serial,
tests/distributed/test_halo_mpi.py:388:        # producing O(1e-5) differences in float32.  Use atol=1e-4
tests/ocean/validation/test_differentiability_ocean.py:2:"""Test JAX differentiability of maintained ocean discretizations over 10 steps.
tests/ocean/validation/test_differentiability_ocean.py:4:For each supported ocean discretization (cdgrid), compute jax.grad through
tests/ocean/validation/test_differentiability_ocean.py:5:10 time steps and verify finite, non-zero gradients.
tests/ocean/validation/test_differentiability_ocean.py:47:def test_differentiability(disc_name, ocean_setup):
tests/ocean/validation/test_differentiability_ocean.py:48:    """Test jax.grad through N_STEPS of model.step()."""
tests/ocean/validation/test_differentiability_ocean.py:58:    grads = jax.grad(loss)(state.T.data)
tests/ocean/validation/test_differentiability_ocean.py:59:    assert jnp.all(jnp.isfinite(grads)), f"{disc_name}: non-finite gradients"
tests/ocean/validation/test_differentiability_ocean.py:60:    assert not jnp.allclose(grads, 0.0), f"{disc_name}: gradients are all zero"
tests/ocean/validation/test_differentiability_ocean.py:91:            grads = jax.grad(loss)(state.T.data)
tests/ocean/validation/test_differentiability_ocean.py:92:            is_finite = bool(jnp.all(jnp.isfinite(grads)))
tests/ocean/validation/test_differentiability_ocean.py:93:            is_nonzero = bool(not jnp.allclose(grads, 0.0))
tests/ocean/validation/test_differentiability_ocean.py:94:            grad_norm = float(jnp.max(jnp.abs(grads)))
tests/ocean/validation/test_differentiability_ocean.py:96:            if is_finite and is_nonzero:
tests/ocean/validation/test_differentiability_ocean.py:97:                print(f"  PASS  {disc}  |grad|_max={grad_norm:.4e}")
tests/ocean/validation/test_differentiability_ocean.py:98:            elif is_finite:
tests/ocean/validation/test_differentiability_ocean.py:99:                print(f"  WARN  {disc}  gradients are all zero")
tests/ocean/validation/test_differentiability_ocean.py:101:                print(f"  FAIL  {disc}  non-finite gradients, |grad|_max={grad_norm}")
tests/distributed/test_mpi_bootstrap.py:197:    def test_halo_exchange_produces_finite_result(self, topology):
tests/distributed/test_mpi_bootstrap.py:198:        """A scalar halo exchange should produce finite results on local faces."""
tests/distributed/test_mpi_bootstrap.py:214:        # Local faces should have all-finite values.
tests/distributed/test_mpi_bootstrap.py:216:            assert jnp.all(jnp.isfinite(result[f]))
tests/unit/test_hybrid_vertical.py:77:        dp_ref = jnp.diff(p_half_ref)
tests/unit/test_hybrid_vertical.py:115:        np.testing.assert_allclose(coord.dsigma_full, np.diff(coord.sigma_full), atol=1e-7)
tests/unit/test_hybrid_vertical.py:367:    """Test JAX differentiability of hybrid coordinate functions."""
tests/unit/test_hybrid_vertical.py:369:    def test_geopotential_differentiable(self):
tests/unit/test_hybrid_vertical.py:370:        """compute_geopotential_hybrid should be differentiable w.r.t. T."""
tests/unit/test_hybrid_vertical.py:379:        grad = jax.grad(f)(T)
tests/unit/test_hybrid_vertical.py:380:        assert grad.shape == T.shape
tests/unit/test_hybrid_vertical.py:381:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_hybrid_vertical.py:383:    def test_mass_flux_differentiable(self):
tests/unit/test_hybrid_vertical.py:384:        """compute_mass_flux_hybrid should be differentiable."""
tests/unit/test_hybrid_vertical.py:392:        grad = jax.grad(f)(div)
tests/unit/test_hybrid_vertical.py:393:        assert grad.shape == div.shape
tests/unit/test_hybrid_vertical.py:394:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_hybrid_vertical.py:396:    def test_pressure_differentiable_wrt_ps(self):
tests/unit/test_hybrid_vertical.py:397:        """Pressure should be differentiable w.r.t. p_s."""
tests/unit/test_hybrid_vertical.py:404:        grad = jax.grad(f)(p_s)
tests/unit/test_hybrid_vertical.py:405:        assert grad.shape == p_s.shape
tests/unit/test_hybrid_vertical.py:406:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_hybrid_vertical.py:442:        dp_ref = jnp.diff(p_half_ref)
tests/unit/test_hybrid_vertical.py:458:        dp = jnp.diff(p_half, axis=-1)
tests/unit/test_hybrid_vertical.py:487:        dp = jnp.diff(p_half_ref)
tests/unit/test_hybrid_vertical.py:494:        dp_ref = jnp.diff(p_half_ref)
tests/unit/test_hybrid_vertical.py:517:    def test_l40_geopotential_finite(self):
tests/unit/test_hybrid_vertical.py:518:        """L40 geopotential should be finite and well-behaved."""
tests/unit/test_hybrid_vertical.py:524:        assert jnp.all(jnp.isfinite(Phi))
tests/stress/test_phase1_ocean_slab.py:52:        """Single slab step: temperature change should be finite, nonzero,
tests/stress/test_phase1_ocean_slab.py:66:        assert jnp.all(jnp.isfinite(dT)), "dT is not finite"
tests/stress/test_phase1_ocean_slab.py:72:        # We just check energy change is finite and consistent:
tests/stress/test_phase1_ocean_slab.py:76:        assert jnp.all(jnp.isfinite(energy_change)), "Energy change not finite"
tests/stress/test_phase1_ocean_slab.py:175:        # All state variables should be finite.
tests/stress/test_phase1_ocean_slab.py:176:        assert jnp.all(jnp.isfinite(state.T_sfc.data)), "T_sfc not finite"
tests/stress/test_phase1_ocean_slab.py:177:        assert jnp.all(jnp.isfinite(state.T_deep.data)), "T_deep not finite"
tests/unit/test_cdgrid.py:5:2. Operators: vorticity, divergence, mass flux, gradients
tests/unit/test_cdgrid.py:82:        self.assertEqual(cdgrid.grad_c00.dtype, jnp.float64)
tests/unit/test_cdgrid.py:91:        self.assertEqual(self.cdgrid.grad_c00.dtype, jnp.float32)
tests/unit/test_cdgrid.py:198:        self.assertTrue(jnp.all(jnp.isfinite(div)))
tests/unit/test_cdgrid.py:211:        self.assertTrue(jnp.all(jnp.isfinite(dh_dt)))
tests/unit/test_cdgrid.py:213:    def test_gradient_constant_field(self):
tests/unit/test_cdgrid.py:215:        from legoesm.core.operators_cdgrid import _arakawa_lamb_gradient
tests/unit/test_cdgrid.py:218:        dB_dx, dB_dy = _arakawa_lamb_gradient(B, self.cdgrid)
tests/unit/test_cdgrid.py:234:        self.assertTrue(jnp.all(jnp.isfinite(du)))
tests/unit/test_cdgrid.py:235:        self.assertTrue(jnp.all(jnp.isfinite(dv)))
tests/unit/test_cdgrid.py:271:    def test_one_step_finite(self):
tests/unit/test_cdgrid.py:272:        """One time step should produce finite values."""
tests/unit/test_cdgrid.py:278:        self.assertTrue(jnp.all(jnp.isfinite(state_new.h)))
tests/unit/test_cdgrid.py:279:        self.assertTrue(jnp.all(jnp.isfinite(state_new.u_d)))
tests/unit/test_cdgrid.py:280:        self.assertTrue(jnp.all(jnp.isfinite(state_new.v_d)))
tests/unit/test_cdgrid.py:309:        self.assertTrue(jnp.all(jnp.isfinite(state.h)))
tests/unit/test_cdgrid.py:310:        self.assertTrue(jnp.all(jnp.isfinite(state.u_d)))
tests/unit/test_cdgrid.py:311:        self.assertTrue(jnp.all(jnp.isfinite(state.v_d)))
tests/unit/test_cdgrid.py:320:        self.assertTrue(jnp.all(jnp.isfinite(state_final.h)))
tests/unit/test_cdgrid.py:323:    def test_differentiable(self):
tests/unit/test_cdgrid.py:324:        """Tendency function should be differentiable."""
tests/unit/test_cdgrid.py:336:        grad_fn = jax.grad(loss)
tests/unit/test_cdgrid.py:337:        g = grad_fn(self.state0.h)
tests/unit/test_cdgrid.py:339:        self.assertTrue(jnp.all(jnp.isfinite(g)))
tests/unit/test_cdgrid.py:398:    def test_tendency_finite(self):
tests/unit/test_cdgrid.py:408:        self.assertTrue(jnp.all(jnp.isfinite(tend.du_dt.data)))
tests/unit/test_cdgrid.py:409:        self.assertTrue(jnp.all(jnp.isfinite(tend.dv_dt.data)))
tests/unit/test_cdgrid.py:410:        self.assertTrue(jnp.all(jnp.isfinite(tend.dT_dt.data)))
tests/unit/test_cdgrid.py:411:        self.assertTrue(jnp.all(jnp.isfinite(tend.dS_dt.data)))
tests/unit/test_cdgrid.py:412:        self.assertTrue(jnp.all(jnp.isfinite(tend.deta_dt.data)))
tests/unit/test_cdgrid.py:428:        # Tracer tendencies should be zero (uniform, no gradients)
tests/unit/test_cdgrid.py:498:    def test_gradient_3d_constant(self):
tests/unit/test_cdgrid.py:499:        from legoesm.core.operators_cdgrid import _arakawa_lamb_gradient
tests/unit/test_cdgrid.py:502:        dB_dx, dB_dy = _arakawa_lamb_gradient(B, self.cdgrid)
tests/unit/test_cdgrid.py:515:        self.assertTrue(jnp.all(jnp.isfinite(dh)))
tests/unit/test_ensemble_diagnostics.py:146:        assert info['spread_T'] > 0.0  # members differ in T
tests/unit/test_ensemble_diagnostics.py:147:        assert info['spread_u'] > 0.0  # members differ in u
tests/da/test_cost_function.py:12:from legoesm.da.cost_function import build_cost_fn, build_cost_and_grad_fn
tests/da/test_cost_function.py:75:    def test_gradient_finite(self, setup):
tests/da/test_cost_function.py:76:        """Gradient should be finite."""
tests/da/test_cost_function.py:85:        cost_and_grad = build_cost_and_grad_fn(
tests/da/test_cost_function.py:90:        J, g = cost_and_grad(x_b)
tests/da/test_cost_function.py:91:        assert jnp.all(jnp.isfinite(g))
tests/da/test_cost_function.py:94:    def test_gradient_correctness_finite_diff(self, setup):
tests/da/test_cost_function.py:95:        """AD gradient should match finite differences."""
tests/da/test_cost_function.py:119:        grad_ad = jax.grad(cost_fn)(x_b)
tests/da/test_cost_function.py:121:        # Finite difference
tests/da/test_cost_function.py:123:        grad_fd = jnp.zeros_like(x_b)
tests/da/test_cost_function.py:129:            grad_fd = grad_fd.at[i].set((fp - fm) / (2 * h))
tests/da/test_cost_function.py:133:        rel_err = jnp.abs(grad_ad[:n_check] - grad_fd[:n_check]) / (
tests/da/test_cost_function.py:134:            jnp.maximum(jnp.abs(grad_ad[:n_check]), 1e-10)
tests/da/test_cost_function.py:148:        assert jnp.isfinite(J)
tests/stress/test_phase4_coupled_mpi.py:55:    """Coupled aquaplanet under MPI should produce bounded, finite results."""
tests/stress/test_phase4_coupled_mpi.py:65:        # All ranks should have finite state
tests/stress/test_phase4_coupled_mpi.py:66:        assert jnp.all(jnp.isfinite(driver.state.T.data))
tests/stress/test_phase4_coupled_mpi.py:67:        assert jnp.all(jnp.isfinite(driver.state.p_s.data))
tests/stress/test_phase4_coupled_mpi.py:68:        assert jnp.all(jnp.isfinite(driver.ocean_state.T_sfc.data))
tests/stress/test_phase4_coupled_mpi.py:87:        assert jnp.all(jnp.isfinite(driver.state.T.data))
tests/stress/test_phase4_coupled_mpi.py:88:        assert jnp.all(jnp.isfinite(driver.ocean_state.T_sfc.data))
tests/stress/test_phase4_coupled_mpi.py:113:            # (some difference expected due to partitioning)
tests/unit/test_neuralgcm_s2s.py:462:    assert np.all(np.isfinite(crps["t850"]))
tests/unit/test_neuralgcm_s2s.py:463:    assert np.all(np.isfinite(crps["sst"]))
tests/distributed/test_coupler_mpi.py:153:        assert jnp.all(jnp.isfinite(g)), f"{name}: non-finite in distributed output"
tests/distributed/test_coupler_mpi.py:154:        assert jnp.all(jnp.isfinite(r)), f"{name}: non-finite in reference output"
tests/distributed/test_voronoi_mpi.py:197:            # differences per reduction.  The mass fixer amplifies this
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:71:        assert jnp.all(jnp.isfinite(s.h.data)), "h contains NaN/Inf"
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:72:        assert jnp.all(jnp.isfinite(s.u.data)), "u contains NaN/Inf"
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:74:        # icosahedral mesh without diffusion. Higher resolution converges.
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:101:        assert jnp.all(jnp.isfinite(s.h.data)), "h contains NaN/Inf after 15 days"
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:102:        assert jnp.all(jnp.isfinite(s.u.data)), "u contains NaN/Inf after 15 days"
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:152:        assert jnp.all(jnp.isfinite(s.h.data)), "h contains NaN/Inf after 5 days"
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:153:        assert jnp.all(jnp.isfinite(s.u.data)), "u contains NaN/Inf after 5 days"
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:228:    """JAX differentiability through the MPAS solver."""
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:230:    def test_differentiable_3_steps(self, mesh):
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:231:        """Model should be differentiable through 3 steps."""
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:247:        grad_fn = jax.grad(loss_fn)
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:248:        g = grad_fn(state.h.data)
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py:249:        assert jnp.all(jnp.isfinite(g)), "Gradient contains NaN/Inf"
tests/stress/test_phase1_radiation_ghg.py:68:        assert jnp.all(jnp.isfinite(olr_standard)), "OLR(standard) not finite"
tests/stress/test_phase1_radiation_ghg.py:69:        assert jnp.all(jnp.isfinite(olr_doubled)), "OLR(doubled) not finite"
tests/stress/test_phase1_radiation_ghg.py:77:        diff = float(jnp.mean(olr_standard - olr_doubled))
tests/stress/test_phase1_radiation_ghg.py:78:        assert diff > 5.0, (
tests/stress/test_phase1_radiation_ghg.py:79:            f"Mean OLR reduction = {diff:.2f} W/m2, expected > 5 W/m2"
tests/stress/test_phase1_radiation_ghg.py:95:        assert np.isfinite(result["co2_ppmv"]), "co2_ppmv not finite"
tests/stress/test_phase1_radiation_ghg.py:96:        assert np.isfinite(result["ch4_ppbv"]), "ch4_ppbv not finite"
tests/stress/test_phase1_radiation_ghg.py:97:        assert np.isfinite(result["n2o_ppbv"]), "n2o_ppbv not finite"
tests/unit/test_grid_dycore_fixes.py:289:        structural test: the function must produce finite output and
tests/unit/test_grid_dycore_fixes.py:298:        # All values must be finite
tests/unit/test_grid_dycore_fixes.py:299:        assert np.all(np.isfinite(vort_np)), "Non-finite vorticity values"
tests/atmosphere/shallow_water/integration/test_boundary_fix.py:17:        div_damp=..., hyperdiff_coeff=..., boundary_fix=True)
tests/atmosphere/shallow_water/integration/test_boundary_fix.py:93:        """boundary_fix produces finite output after 100 steps on TC2."""
tests/atmosphere/shallow_water/integration/test_boundary_fix.py:99:            hyperdiff_coeff=dx ** 4 / (86400.0 * 10),
tests/atmosphere/shallow_water/integration/test_boundary_fix.py:107:        assert jnp.all(jnp.isfinite(state.h))
tests/atmosphere/shallow_water/integration/test_boundary_fix.py:108:        assert jnp.all(jnp.isfinite(state.u_d))
tests/atmosphere/shallow_water/integration/test_boundary_fix.py:111:        """boundary_fix produces finite output after 100 steps on TC5."""
tests/atmosphere/shallow_water/integration/test_boundary_fix.py:117:            hyperdiff_coeff=dx ** 4 / (86400.0 * 10),
tests/atmosphere/shallow_water/integration/test_boundary_fix.py:125:        assert jnp.all(jnp.isfinite(state.h))
tests/atmosphere/shallow_water/integration/test_boundary_fix.py:161:                hyperdiff_coeff=dx ** 4 / (86400.0 * 10),
tests/stress/test_phase3_restart.py:50:# This causes O(0.05 K) differences over 5+ days of post-restart integration.
tests/williamson_diagnostic.py:60:            is_finite = bool(jnp.all(jnp.isfinite(state.h)))
tests/williamson_diagnostic.py:64:                  f"finite={is_finite}")
tests/williamson_diagnostic.py:66:            if not is_finite:
tests/williamson_diagnostic.py:164:    # Model config -- mild diffusion since edge-midpoint path eliminates HK
tests/williamson_diagnostic.py:168:        hyperdiff_coeff=dx_min ** 4 / (86400.0 * 30.0),
tests/williamson_diagnostic.py:205:    all_finite = bool(jnp.all(jnp.isfinite(state_final.h)) and
tests/williamson_diagnostic.py:206:                       jnp.all(jnp.isfinite(state_final.u_d)) and
tests/williamson_diagnostic.py:207:                       jnp.all(jnp.isfinite(state_final.v_d)))
tests/williamson_diagnostic.py:208:    print(f"  Stable (all finite): {all_finite}")
tests/williamson_diagnostic.py:210:    if not all_finite:
tests/williamson_diagnostic.py:256:        "stable": stable and all_finite,
tests/williamson_diagnostic.py:269:    # Strict targets: mild hyperdiffusion + non-orthogonality corrections
tests/williamson_diagnostic.py:348:        hyperdiff_coeff=dx_min ** 4 / (86400.0 * 30.0),
tests/williamson_diagnostic.py:376:    all_finite = bool(jnp.all(jnp.isfinite(state_final.h)) and
tests/williamson_diagnostic.py:377:                       jnp.all(jnp.isfinite(state_final.u_d)) and
tests/williamson_diagnostic.py:378:                       jnp.all(jnp.isfinite(state_final.v_d)))
tests/williamson_diagnostic.py:379:    print(f"  Stable (all finite): {all_finite}")
tests/williamson_diagnostic.py:381:    if not all_finite:
tests/williamson_diagnostic.py:405:        "stable": stable and all_finite,
tests/williamson_diagnostic.py:417:    c1 = stable and all_finite
tests/williamson_diagnostic.py:518:        hyperdiff_coeff=dx_min ** 4 / (86400.0 * 30.0),
tests/williamson_diagnostic.py:550:    all_finite = bool(jnp.all(jnp.isfinite(state_final.h)) and
tests/williamson_diagnostic.py:551:                       jnp.all(jnp.isfinite(state_final.u_d)) and
tests/williamson_diagnostic.py:552:                       jnp.all(jnp.isfinite(state_final.v_d)))
tests/williamson_diagnostic.py:553:    print(f"  Stable (all finite): {all_finite}")
tests/williamson_diagnostic.py:555:    if not all_finite:
tests/williamson_diagnostic.py:585:        "stable": stable and all_finite,
tests/williamson_diagnostic.py:598:    c1 = stable and all_finite
tests/unit/test_precision.py:150:        set_module_override("pressure_gradient", compute="fp64")
tests/unit/test_precision.py:152:        assert "pressure_gradient" in overrides
tests/unit/test_precision.py:153:        assert overrides["pressure_gradient"]["compute"] == jnp.float64
tests/unit/test_precision.py:325:        assert "pressure_gradient" in overrides
tests/unit/test_precision.py:388:        # Use relaxed energy threshold since 0.01 K offset → 3.5e-5 rel energy diff
tests/unit/test_precision.py:585:    def test_grad_through_cast(self):
tests/unit/test_precision.py:586:        """Verify gradients flow through precision casts."""
tests/unit/test_precision.py:588:        @jax.grad
tests/unit/test_precision.py:596:        grad = f(x)
tests/unit/test_precision.py:597:        assert jnp.allclose(grad, 2.0 * x, atol=1e-5)
tests/ocean/distributed/test_ocean_mpi_conservation.py:208:            assert jnp.all(jnp.isfinite(state_global.u.data))
tests/ocean/distributed/test_ocean_mpi_conservation.py:209:            assert jnp.all(jnp.isfinite(state_global.T.data))
tests/unit/test_precision_modes.py:114:        assert cdg.grad_c00.dtype == jnp.float64
tests/unit/test_precision_modes.py:124:        assert cdg.grad_c00.dtype == jnp.float64
tests/unit/test_precision_modes.py:174:        assert jnp.all(jnp.isfinite(s.h))
tests/unit/test_precision_modes.py:179:        assert jnp.all(jnp.isfinite(s.h))
tests/unit/test_precision_modes.py:185:        assert jnp.all(jnp.isfinite(s.h))
tests/unit/test_precision_modes.py:211:        assert jnp.all(jnp.isfinite(s.h.data))
tests/unit/test_precision_modes.py:216:        assert jnp.all(jnp.isfinite(s.h.data))
tests/unit/test_precision_modes.py:221:        assert jnp.all(jnp.isfinite(s.h.data))
tests/atmosphere/shallow_water/integration/test_fv_convergence.py:51:        hyperdiff_coeff=1e15,
tests/atmosphere/shallow_water/integration/test_fv_convergence.py:79:        The C-D grid scheme's discrete steady state differs from the
tests/atmosphere/shallow_water/integration/test_fv_convergence.py:119:            hyperdiff_coeff=1e15,
tests/unit/test_latlon_grid.py:47:        assert jnp.all(jnp.diff(grid.lat) > 0)
tests/unit/test_latlon_grid.py:99:    """Test that different resolutions work correctly."""
tests/unit/test_latlon_grid.py:107:        assert jnp.all(jnp.isfinite(grid.area))
tests/unit/test_physics_grid_adapters.py:78:            self.dsigma = jnp.diff(self.sigma_half)
tests/unit/test_physics_grid_adapters.py:405:    def test_finite_values(self, cs_grid):
tests/unit/test_physics_grid_adapters.py:407:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_physics_grid_adapters.py:408:        assert jnp.all(jnp.isfinite(out.dq_v_dt))
tests/unit/test_physics_grid_adapters.py:409:        assert jnp.all(jnp.isfinite(out.precip))
tests/unit/test_physics_grid_adapters.py:423:    def test_finite_values(self, ll_grid):
tests/unit/test_physics_grid_adapters.py:425:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_physics_grid_adapters.py:426:        assert jnp.all(jnp.isfinite(out.precip))
tests/unit/test_physics_grid_adapters.py:438:    def test_finite_values(self, sc_grid):
tests/unit/test_physics_grid_adapters.py:440:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_physics_grid_adapters.py:441:        assert jnp.all(jnp.isfinite(out.precip))
tests/unit/test_physics_grid_adapters.py:582:        assert jnp.all(jnp.isfinite(phys_out.dT_dt))
tests/unit/test_physics_grid_adapters.py:624:        assert jnp.all(jnp.isfinite(phys_out.dT_dt))
tests/unit/test_physics_grid_adapters.py:666:        assert jnp.all(jnp.isfinite(phys_out.conv_prog))
tests/unit/test_grid.py:65:        # due to centered difference. Allow wide range for low-res grid.
tests/unit/test_grid.py:69:    def test_different_resolutions(self):
tests/stress/test_phase4_sea_ice_mpi.py:32:    """Sea ice operations under MPI should produce finite, bounded results."""
tests/stress/test_phase4_sea_ice_mpi.py:59:        assert jnp.all(jnp.isfinite(u_new)), "NaN in u_ice after EVP"
tests/stress/test_phase4_sea_ice_mpi.py:60:        assert jnp.all(jnp.isfinite(v_new)), "NaN in v_ice after EVP"
tests/stress/test_phase4_sea_ice_mpi.py:61:        assert jnp.all(jnp.isfinite(s12)), "NaN in sigma_12 after EVP"
tests/stress/test_phase4_sea_ice_mpi.py:83:        assert jnp.all(jnp.isfinite(h_new)), "NaN in h_ice after transport"
tests/stress/test_phase4_sea_ice_mpi.py:84:        assert jnp.all(jnp.isfinite(conc_new)), "NaN in concentration"
tests/stress/test_phase4_sea_ice_mpi.py:85:        assert jnp.all(jnp.isfinite(T_new)), "NaN in T_ice after transport"
tests/test_cases/baroclinic_wave.py:9:   state (gradient-wind + hydrostatic balance). Verify the model maintains
tests/test_cases/baroclinic_wave.py:80:_constB = (_T0 - T0P) / (_T0 * T0P)  # Meridional T gradient parameter
tests/test_cases/baroclinic_wave.py:100:    hydrostatic balance and gradient-wind balance.
tests/test_cases/baroclinic_wave.py:190:    """Compute zonal wind from gradient-wind balance.
tests/test_cases/baroclinic_wave.py:192:    Solves the quadratic gradient-wind equation for the steady-state
tests/test_cases/baroclinic_wave.py:197:    where bigU encodes the pressure gradient.
tests/test_cases/baroclinic_wave.py:206:        Temperature [K] (needed for the pressure gradient term).
tests/test_cases/baroclinic_wave.py:224:    # Latitude-dependent term for wind (different from temperature term!)
tests/test_cases/baroclinic_wave.py:348:        # Compute zonal wind from gradient-wind balance
tests/test_cases/baroclinic_wave.py:588:        # Compute zonal wind from gradient-wind balance
tests/unit/test_land_ice_soil_hydraulics.py:59:        dtheta = jnp.diff(theta)
tests/unit/test_land_ice_soil_hydraulics.py:65:        dK = jnp.diff(K)
tests/unit/test_land_ice_soil_hydraulics.py:133:        dK = jnp.diff(K)
tests/unit/test_land_ice_soil_hydraulics.py:160:        dtheta = jnp.diff(theta)
tests/unit/test_land_ice_soil_hydraulics.py:210:        dtheta = jnp.diff(theta)
tests/unit/test_land_ice_soil_hydraulics.py:264:        assert jnp.all(jnp.isfinite(theta))
tests/unit/test_land_ice_soil_hydraulics.py:273:        assert jnp.all(jnp.isfinite(K))
tests/atmosphere/shallow_water/test_cases/williamson.py:128:    u_0 = 20.0  # m/s (note: different from Test 2!)
tests/atmosphere/hydrostatic/validation/test_stability_fix.py:20:# Small hyperdiffusion for high wavenumbers
tests/atmosphere/hydrostatic/validation/test_stability_fix.py:25:    hyperdiff_coeff=HYPERDIFF,
tests/atmosphere/hydrostatic/validation/test_stability_fix.py:26:    hyperdiff_order=2,
tests/atmosphere/hydrostatic/validation/test_stability_fix.py:63:        if T_max > 100 or u_max > 100 or not jnp.isfinite(T_mean):
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:12:- JAX differentiability
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:134:        """Full levels are between adjacent half levels."""
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:161:        dpi_dz = jnp.diff(hc.exner_ref) / jnp.diff(hc.z_full)
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:177:    def test_flat_terrain_jacobian(self, grid, height_coord):
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:181:        assert jnp.allclose(tm.jacobian, 1.0)
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:183:    def test_mountain_terrain_jacobian(self, grid, height_coord):
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:188:        assert jnp.allclose(tm.jacobian, expected_J, rtol=1e-5)
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:242:        config = CDGridCompressibleEulerConfig(hyperdiff_coeff=0.0, sponge_coeff=0.0)
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:250:    def test_tendencies_finite(self, grid, height_coord, terrain_metric, cdgrid):
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:251:        """All tendencies should be finite."""
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:261:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:262:        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:263:        assert jnp.all(jnp.isfinite(tend.dw_dt.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:264:        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:265:        assert jnp.all(jnp.isfinite(tend.drho_prime_dt.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:306:    def test_large_negative_perturbations_remain_finite(self, height_coord, grid):
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:307:        """Guarded Exner computation should stay finite for stressed states."""
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:313:        assert jnp.all(jnp.isfinite(pi_p))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:326:            hyperdiff_coeff=0.0,
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:332:        assert jnp.all(jnp.isfinite(new_state.u.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:333:        assert jnp.all(jnp.isfinite(new_state.w.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:338:            hyperdiff_coeff=0.0,
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:346:        assert jnp.all(jnp.isfinite(state.u.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:347:        assert jnp.all(jnp.isfinite(state.theta_prime.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:361:        assert jnp.all(jnp.isfinite(state.u.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:362:        assert jnp.all(jnp.isfinite(state.theta_prime.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:376:        assert jnp.all(jnp.isfinite(state.u.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:384:        assert jnp.all(jnp.isfinite(state.u.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:419:    def test_kessler_tendencies_finite(self, grid, height_coord, terrain_metric):
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:420:        """Kessler tendencies are finite with moisture."""
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:431:        assert jnp.all(jnp.isfinite(tend.dtracers_dt.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:432:        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:440:    """Tests that the NH model is differentiable through JAX."""
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:442:    def test_grad_through_tendencies(self, grid, height_coord, terrain_metric, cdgrid):
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:443:        """jax.grad works through C-D grid tendency computation."""
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:444:        config = CDGridCompressibleEulerConfig(hyperdiff_coeff=0.0, sponge_coeff=0.0)
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:456:        grad = jax.grad(loss_fn)(state.theta_prime.data)
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:457:        assert jnp.all(jnp.isfinite(grad))
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:488:        """NH dry mass diagnostic should be positive and finite."""
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:494:        assert jnp.isfinite(mass)
tests/unit/test_operators_latlon.py:9:    gradient_x,
tests/unit/test_operators_latlon.py:10:    gradient_y,
tests/unit/test_operators_latlon.py:14:    hyperdiffusion,
tests/unit/test_operators_latlon.py:20:    gradient_x_3d,
tests/unit/test_operators_latlon.py:21:    gradient_y_3d,
tests/unit/test_operators_latlon.py:23:    hyperdiffusion_3d,
tests/unit/test_operators_latlon.py:74:    def test_zero_gradient_latitude(self, grid):
tests/unit/test_operators_latlon.py:75:        """Latitude halo should be zero-gradient (copy edge rows)."""
tests/unit/test_operators_latlon.py:88:    """Tests for gradient operators."""
tests/unit/test_operators_latlon.py:90:    def test_gradient_uniform_zero(self, grid):
tests/unit/test_operators_latlon.py:93:        gx = gradient_x(f, grid)
tests/unit/test_operators_latlon.py:94:        gy = gradient_y(f, grid)
tests/unit/test_operators_latlon.py:98:    def test_gradient_shape(self, grid):
tests/unit/test_operators_latlon.py:101:        gx = gradient_x(f, grid)
tests/unit/test_operators_latlon.py:102:        gy = gradient_y(f, grid)
tests/unit/test_operators_latlon.py:106:    def test_gradient_finite(self, grid):
tests/unit/test_operators_latlon.py:107:        """Gradient should produce finite values for random input."""
tests/unit/test_operators_latlon.py:111:        gx = gradient_x(f, grid)
tests/unit/test_operators_latlon.py:112:        gy = gradient_y(f, grid)
tests/unit/test_operators_latlon.py:113:        assert jnp.all(jnp.isfinite(gx.data))
tests/unit/test_operators_latlon.py:114:        assert jnp.all(jnp.isfinite(gy.data))
tests/unit/test_operators_latlon.py:127:        assert jnp.all(jnp.isfinite(div.data))
tests/unit/test_operators_latlon.py:136:    def test_divergence_finite(self, grid):
tests/unit/test_operators_latlon.py:137:        """Divergence should be finite for random input."""
tests/unit/test_operators_latlon.py:145:        assert jnp.all(jnp.isfinite(div.data))
tests/unit/test_operators_latlon.py:158:    def test_curl_finite(self, grid):
tests/unit/test_operators_latlon.py:159:        """Vorticity should be finite."""
tests/unit/test_operators_latlon.py:167:        assert jnp.all(jnp.isfinite(vort.data))
tests/unit/test_operators_latlon.py:185:    def test_laplacian_finite(self, grid):
tests/unit/test_operators_latlon.py:186:        """Laplacian should be finite for random input."""
tests/unit/test_operators_latlon.py:191:        assert jnp.all(jnp.isfinite(lap.data))
tests/unit/test_operators_latlon.py:194:class TestHyperdiffusion:
tests/unit/test_operators_latlon.py:195:    """Tests for hyperdiffusion operator."""
tests/unit/test_operators_latlon.py:197:    def test_hyperdiffusion_uniform_zero(self, grid):
tests/unit/test_operators_latlon.py:198:        """Hyperdiffusion of a uniform field should be zero."""
tests/unit/test_operators_latlon.py:200:        hd = hyperdiffusion(f, grid, coeff=1e15)
tests/unit/test_operators_latlon.py:203:    def test_hyperdiffusion_finite(self, grid):
tests/unit/test_operators_latlon.py:204:        """Hyperdiffusion should be finite."""
tests/unit/test_operators_latlon.py:208:        hd = hyperdiffusion(f, grid, coeff=1e15)
tests/unit/test_operators_latlon.py:209:        assert jnp.all(jnp.isfinite(hd.data))
tests/unit/test_operators_latlon.py:257:    def test_gradient_3d_shape(self, grid):
tests/unit/test_operators_latlon.py:258:        """3D gradient should have correct shape."""
tests/unit/test_operators_latlon.py:262:        gx = gradient_x_3d(f, grid)
tests/unit/test_operators_latlon.py:263:        gy = gradient_y_3d(f, grid)
tests/unit/test_operators_latlon.py:267:    def test_hyperdiffusion_3d_shape(self, grid):
tests/unit/test_operators_latlon.py:268:        """3D hyperdiffusion should have correct shape."""
tests/unit/test_operators_latlon.py:272:        hd = hyperdiffusion_3d(f, grid, 1e15)
tests/unit/test_operators_latlon.py:275:    def test_3d_operators_finite(self, grid):
tests/unit/test_operators_latlon.py:276:        """All 3D operators should produce finite values."""
tests/unit/test_operators_latlon.py:284:        assert jnp.all(jnp.isfinite(vorticity_3d(u, v, grid)))
tests/unit/test_operators_latlon.py:285:        assert jnp.all(jnp.isfinite(divergence_3d(u, v, grid)))
tests/unit/test_operators_latlon.py:286:        assert jnp.all(jnp.isfinite(gradient_x_3d(u, grid)))
tests/unit/test_operators_latlon.py:287:        assert jnp.all(jnp.isfinite(gradient_y_3d(u, grid)))
tests/unit/test_operators_latlon.py:367:    """Tests that operators are differentiable."""
tests/unit/test_operators_latlon.py:369:    def test_grad_through_gradient(self, grid):
tests/unit/test_operators_latlon.py:370:        """jax.grad should work through gradient_x."""
tests/unit/test_operators_latlon.py:373:            gx = gradient_x(f, grid)
tests/unit/test_operators_latlon.py:378:        grads = jax.grad(loss)(data)
tests/unit/test_operators_latlon.py:379:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_operators_latlon.py:381:    def test_grad_through_divergence(self, grid):
tests/unit/test_operators_latlon.py:382:        """jax.grad should work through divergence."""
tests/unit/test_operators_latlon.py:392:        grads = jax.grad(loss)(data)
tests/unit/test_operators_latlon.py:393:        assert jnp.all(jnp.isfinite(grads))
tests/stress/test_phase7_multiyear.py:61:        assert jnp.all(jnp.isfinite(driver.state.T.data))
tests/stress/test_phase7_multiyear.py:102:        assert jnp.all(jnp.isfinite(driver.state.T.data))
tests/stress/test_phase7_multiyear.py:104:    def test_co2_finite_and_positive(self, driver):
tests/stress/test_phase7_multiyear.py:105:        """CO2 field stays finite and positive over 1 year."""
tests/stress/test_phase7_multiyear.py:108:        assert jnp.all(jnp.isfinite(driver._co2_field)), "CO2 has NaN/inf"
tests/unit/test_cdgrid_fv3_regression.py:175:    def test_residual_finite(self):
tests/unit/test_cdgrid_fv3_regression.py:176:        """Tendencies should be finite for balanced flow."""
tests/unit/test_cdgrid_fv3_regression.py:191:        self.assertTrue(jnp.all(jnp.isfinite(dh_dt)), "dh_dt has non-finite values")
tests/unit/test_cdgrid_fv3_regression.py:192:        self.assertTrue(jnp.all(jnp.isfinite(du_dt)), "du_dt has non-finite values")
tests/unit/test_cdgrid_fv3_regression.py:193:        self.assertTrue(jnp.all(jnp.isfinite(dv_dt)), "dv_dt has non-finite values")
tests/unit/test_cdgrid_fv3_regression.py:210:        max_diff = float(jnp.max(jnp.abs(cdgrid.rsin_u - expected)))
tests/unit/test_cdgrid_fv3_regression.py:211:        self.assertLess(max_diff, 1e-6,
tests/unit/test_cdgrid_fv3_regression.py:212:                        f"rsin_u not uniform 1/sin²: max diff = {max_diff:.2e}")
tests/unit/test_cdgrid_fv3_regression.py:225:        max_diff = float(jnp.max(jnp.abs(cdgrid.rsin_v - expected)))
tests/unit/test_cdgrid_fv3_regression.py:226:        self.assertLess(max_diff, 1e-6,
tests/unit/test_cdgrid_fv3_regression.py:227:                        f"rsin_v not uniform 1/sin²: max diff = {max_diff:.2e}")
tests/unit/test_cdgrid_fv3_regression.py:248:    def test_one_step_finite(self):
tests/unit/test_cdgrid_fv3_regression.py:249:        """One FB step should produce finite values (even if inaccurate)."""
tests/unit/test_cdgrid_fv3_regression.py:264:        self.assertTrue(jnp.all(jnp.isfinite(h_new)),
tests/unit/test_cdgrid_fv3_regression.py:265:                        "h_new has non-finite values after 1 FB step")
tests/unit/test_cdgrid_fv3_regression.py:266:        self.assertTrue(jnp.all(jnp.isfinite(u_new)),
tests/unit/test_cdgrid_fv3_regression.py:267:                        "u_new has non-finite values after 1 FB step")
tests/unit/test_cdgrid_fv3_regression.py:268:        self.assertTrue(jnp.all(jnp.isfinite(v_new)),
tests/unit/test_cdgrid_fv3_regression.py:269:                        "v_new has non-finite values after 1 FB step")
tests/unit/test_cdgrid_fv3_regression.py:304:    on this path were shape/finite checks; the adversarial review noted
tests/unit/test_cdgrid_fv3_regression.py:305:    that wrong-but-finite boundary winds could ship silently.
tests/unit/test_cdgrid_fv3_regression.py:372:        # Boundary values should not differ wildly from interior (<3x)
tests/unit/test_cdgrid_fv3_regression.py:386:        # Result must be finite everywhere.
tests/unit/test_cdgrid_fv3_regression.py:389:            self.assertTrue(bool(jnp.all(jnp.isfinite(arr))),
tests/unit/test_cdgrid_fv3_regression.py:390:                            f"{name} has non-finite values")
tests/unit/test_cdgrid_fv3_regression.py:397:        use stale state (e.g., when the JAX trace captures different
tests/unit/test_cdgrid_fv3_regression.py:426:        # Eager and JIT may differ at floating-point precision; tolerance
tests/unit/test_cdgrid_fv3_regression.py:433:                            f"{name} eager vs jit differ by {d:.3e} "
tests/unit/test_cdgrid_fv3_regression.py:570:        For ω > 0 (prograde rotation), the contravariant transport
tests/unit/test_physical_balances.py:65:            hyperdiff_coeff=0.0,
tests/unit/test_physical_balances.py:141:            hyperdiff_coeff=0.0,
tests/unit/test_physical_balances.py:155:        h_diff = state.h - self.h_init
tests/unit/test_physical_balances.py:158:        rel_drift = float(jnp.max(jnp.abs(h_diff))) / h_range
tests/unit/test_physical_balances.py:161:        assert jnp.all(jnp.isfinite(state.h)), "h non-finite"
tests/unit/test_physical_balances.py:213:        h_diff = state.h.data - self.h_init
tests/unit/test_physical_balances.py:216:        rel_drift = float(jnp.max(jnp.abs(h_diff))) / h_range
tests/unit/test_physical_balances.py:219:        assert jnp.all(jnp.isfinite(state.h.data)), "h non-finite"
tests/unit/test_physical_balances.py:274:        assert jnp.all(jnp.isfinite(state.u.data)), "u non-finite"
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:35:            hyperdiff_coeff=1e14,
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:79:    def test_tendencies_finite(self, model_state_dt):
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:82:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:83:        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:84:        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:85:        assert jnp.all(jnp.isfinite(tend.drho_prime_dt.data))
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:90:        assert jnp.all(jnp.isfinite(s1.theta_prime.data))
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:91:        assert jnp.all(jnp.isfinite(s1.rho_prime.data))
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:92:        assert jnp.all(jnp.isfinite(s1.u.data))
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:100:        assert jnp.all(jnp.isfinite(s.theta_prime.data))
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:101:        assert jnp.all(jnp.isfinite(s.rho_prime.data))
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:102:        assert jnp.all(jnp.isfinite(s.u.data))
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:106:    def test_differentiable_10_steps(self, model_state_dt):
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:107:        """FV CE should be differentiable through 10 steps."""
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:115:        grad_fn = jax.grad(loss_fn)
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:116:        g = grad_fn(state.theta_prime.data)
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py:117:        assert jnp.all(jnp.isfinite(g))
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:5:- Slow tendency computation (rest state, shapes, finite values)
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:9:- JAX differentiability
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:46:def _proper_hyperdiff(grid):
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:47:    """Resolution-appropriate hyperdiffusion (4-hour damping at truncation)."""
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:84:    """Config with weak hyperdiffusion for stability."""
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:86:        hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:87:        hyperdiff_order=2,
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:174:    def test_tendencies_finite(
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:177:        """All tendency values should be finite."""
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:187:            assert jnp.all(jnp.isfinite(data)), \
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:188:                f"Tendency {field_name} has non-finite values"
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:245:        """Spectral->grid->acoustic->grid->spectral roundtrip should be finite."""
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:273:        assert jnp.all(jnp.isfinite(w_hat_new)), "w_hat has non-finite after roundtrip"
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:274:        assert jnp.all(jnp.isfinite(theta_hat_new)), "theta_hat non-finite after roundtrip"
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:275:        assert jnp.all(jnp.isfinite(rho_hat_new)), "rho_hat non-finite after roundtrip"
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:318:        assert jnp.all(jnp.isfinite(pi_p)), "Exner perturbation has non-finite values"
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:336:    def test_single_step_finite(self, grid, height_coord, terrain_metric, config):
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:337:        """Single time step should produce finite state."""
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:351:            assert jnp.all(jnp.isfinite(data)), \
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:352:                f"After 1 step, {field_name} has non-finite values"
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:371:            assert jnp.all(jnp.isfinite(data)), \
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:372:                f"After 20 steps, {field_name} has non-finite values"
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:391:        assert jnp.all(jnp.isfinite(final.vor_hat.data))
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:399:    """Tests for JAX differentiability of spectral NH."""
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:401:    def test_tendency_differentiable(
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:404:        """jax.grad should work through spectral_nh_slow_tendencies."""
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:416:        grad = jax.grad(loss)(state.theta_prime_hat.data)
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:417:        assert grad.shape == state.theta_prime_hat.data.shape
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py:418:        assert jnp.all(jnp.isfinite(grad)), "Gradient has non-finite values"
tests/ocean/unit/test_ocean_fc.py:35:    """FC ocean tracer tendencies have correct shapes and are finite."""
tests/ocean/unit/test_ocean_fc.py:37:    config = OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0)
tests/ocean/unit/test_ocean_fc.py:43:    assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/ocean/unit/test_ocean_fc.py:44:    assert jnp.all(jnp.isfinite(tend.dS_dt.data))
tests/ocean/unit/test_ocean_fc.py:47:def test_ocean_fc_pressure_gradient(ocean_grid, ocean_z_coord, ocean_state):
tests/ocean/unit/test_ocean_fc.py:48:    """FC ocean momentum tendencies have correct shapes and are finite."""
tests/ocean/unit/test_ocean_fc.py:50:    config = OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0)
tests/ocean/unit/test_ocean_fc.py:56:    assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/ocean/unit/test_ocean_fc.py:57:    assert jnp.all(jnp.isfinite(tend.dv_dt.data))
tests/ocean/unit/test_ocean_fc.py:63:    config = OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0)
tests/ocean/unit/test_ocean_fc.py:77:    config = OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0)
tests/ocean/unit/test_ocean_fc.py:82:    assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/ocean/unit/test_ocean_fc.py:84:    assert jnp.all(jnp.isfinite(tend.deta_dt.data))
tests/ocean/unit/test_ocean_fc.py:99:    assert jnp.all(jnp.isfinite(rho))
tests/ocean/unit/test_ocean_fc.py:112:        A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0,
tests/ocean/unit/test_ocean_fc.py:119:    assert jnp.all(jnp.isfinite(state_new.eta.data))
tests/ocean/unit/test_ocean_fc.py:120:    assert jnp.all(jnp.isfinite(state_new.T.data))
tests/ocean/unit/test_ocean_fc.py:130:        A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0,
tests/ocean/unit/test_ocean_fc.py:137:    assert jnp.all(jnp.isfinite(state_new.eta.data))
tests/ocean/unit/test_ocean_fc.py:138:    assert jnp.all(jnp.isfinite(state_new.T.data))
tests/ocean/unit/test_ocean_fc.py:147:    assert jnp.all(jnp.isfinite(state_new.eta.data))
tests/ocean/unit/test_ocean_fc.py:148:    assert jnp.all(jnp.isfinite(state_new.T.data))
tests/unit/test_land_ice_multilayer.py:3:Tests the multi-layer soil model including thermal diffusion,
tests/unit/test_land_ice_multilayer.py:71:    def test_returns_finite(self):
tests/unit/test_land_ice_multilayer.py:79:            assert jnp.all(jnp.isfinite(arr)), f"MultiLayerLandState.{name} has non-finite values"
tests/unit/test_land_ice_multilayer.py:82:            assert jnp.all(jnp.isfinite(arr)), f"TileResponse.{name} has non-finite values"
tests/unit/test_land_ice_multilayer.py:100:# 2b  Soil thermal diffusion -- uniform T steady state
tests/stress/test_phase6_cmip_e2e.py:60:        assert jnp.all(jnp.isfinite(driver.state.T.data))
tests/stress/test_phase6_cmip_e2e.py:61:        assert jnp.all(jnp.isfinite(driver.state.p_s.data))
tests/stress/test_phase6_cmip_e2e.py:78:        assert jnp.all(jnp.isfinite(driver.state.T.data))
tests/stress/test_phase6_cmip_e2e.py:161:        assert jnp.all(jnp.isfinite(driver.state.T.data))
tests/stress/test_phase6_cmip_e2e.py:177:        assert jnp.all(jnp.isfinite(driver.state.T.data))
tests/stress/test_phase6_cmip_e2e.py:247:        """GHG VMR changes between different simulation days for transient experiments."""
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:6:(advection + microphysics + spectral/hyperdiff filter) end-to-end.
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:39:from legoesm.atmosphere.physics.gravity_wave_drag.config import (
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:68:def _proper_hyperdiff(grid):
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:131:    def test_state_remains_finite_200_steps(self, grid, sigma_coord):
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:134:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:149:            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:186:        # All prognostic fields must be finite.
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:187:        assert bool(jnp.all(jnp.isfinite(s.vor_hat.data))), "vor diverged"
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:188:        assert bool(jnp.all(jnp.isfinite(s.div_hat.data))), "div diverged"
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:189:        assert bool(jnp.all(jnp.isfinite(s.T_hat.data))), "T diverged"
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:190:        assert bool(jnp.all(jnp.isfinite(s.lnps_hat.data))), "lnps diverged"
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:193:        assert bool(jnp.all(jnp.isfinite(qv))), "q_v diverged"
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:222:        With Held-Suarez relaxing T to a strong equator-pole gradient
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:232:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:245:            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:275:        relative_jumps = jnp.abs(jnp.diff(means_arr) / means_arr[:-1])
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:298:    """The spectral filter and implicit hyperdiffusion filter operate
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:324:            hyperdiff_coeff=_proper_hyperdiff(grid) * 50.0,  # aggressive
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:372:        # Aggressive hyperdiff to make the n=20 mode decay quickly.
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:374:            hyperdiff_coeff=_proper_hyperdiff(grid) * 100.0,
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:421:        wave-1 zonal jet stays bounded and finite."""
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:446:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py:460:        assert bool(jnp.all(jnp.isfinite(new_qv))), (
tests/unit/test_voronoi_trisk_weights.py:182:    def test_stencil_sizes_match_cell_adjacency(self, mesh, arrays):
tests/unit/test_voronoi_trisk_weights.py:210:# This field represents a discrete streamfunction-gradient flow.
tests/unit/test_voronoi_trisk_weights.py:226:    def test_curl_of_trisk_of_vertex_gradient_is_zero(self, mesh):
tests/unit/test_voronoi_trisk_weights.py:227:        """curl_vertex(TRiSK(grad_v ψ)) ≈ 0 for arbitrary vertex ψ."""
tests/unit/test_voronoi_trisk_weights.py:234:        # u_edge = ψ(v1) - ψ(v0): a discrete tangential gradient of ψ.
tests/unit/test_voronoi_trisk_weights.py:244:        # small SCVT orthogonality error (from finite Lloyd iterations)
tests/unit/test_voronoi_trisk_weights.py:247:            f"curl(v_t) of vertex-gradient flow: max |curl|/|v_t| = {rel:.3e}"
tests/unit/test_voronoi_trisk_weights.py:340:        """curl(TRiSK(grad_v ψ)) ≈ 0 on the channel mesh."""
tests/atmosphere/shallow_water/unit/test_sfno_sw.py:100:        """Packed state should be finite."""
tests/atmosphere/shallow_water/unit/test_sfno_sw.py:102:        assert jnp.all(jnp.isfinite(packed))
tests/atmosphere/shallow_water/unit/test_spectral.py:16:    spectral_hyperdiffusion,
tests/atmosphere/shallow_water/unit/test_spectral.py:34:def _proper_hyperdiff(grid):
tests/atmosphere/shallow_water/unit/test_spectral.py:35:    """Compute resolution-appropriate hyperdiffusion (4-hour damping at truncation)."""
tests/atmosphere/shallow_water/unit/test_spectral.py:215:    def test_hyperdiffusion_is_dissipative_for_order1(self, grid_t21):
tests/atmosphere/shallow_water/unit/test_spectral.py:219:        diff = spectral_hyperdiffusion(g, coeffs, nu=1.0, order=1)
tests/atmosphere/shallow_water/unit/test_spectral.py:221:        assert float(jnp.real(diff[1])) < 0.0
tests/atmosphere/shallow_water/unit/test_spectral.py:223:    def test_hyperdiffusion_invalid_order_raises(self, grid_t21):
tests/atmosphere/shallow_water/unit/test_spectral.py:224:        """Non-positive hyperdiffusion order should raise."""
tests/atmosphere/shallow_water/unit/test_spectral.py:227:            spectral_hyperdiffusion(grid_t21, coeffs, nu=1.0, order=0)
tests/atmosphere/shallow_water/unit/test_spectral.py:229:    def test_hyperdiffusion_negative_nu_raises(self, grid_t21):
tests/atmosphere/shallow_water/unit/test_spectral.py:230:        """Negative hyperdiffusion coefficient should be rejected."""
tests/atmosphere/shallow_water/unit/test_spectral.py:233:            spectral_hyperdiffusion(grid_t21, coeffs, nu=-1.0, order=2)
tests/atmosphere/shallow_water/unit/test_spectral.py:235:    def test_hyperdiffusion_zero_nu_returns_zero(self, grid_t21):
tests/atmosphere/shallow_water/unit/test_spectral.py:236:        """Zero hyperdiffusion coefficient should return identically zero tendency."""
tests/atmosphere/shallow_water/unit/test_spectral.py:238:        diff = spectral_hyperdiffusion(grid_t21, coeffs, nu=0.0, order=2)
tests/atmosphere/shallow_water/unit/test_spectral.py:239:        assert jnp.all(diff == 0.0)
tests/atmosphere/shallow_water/unit/test_spectral.py:289:        """Test Case 2 initialization produces finite spectral coefficients."""
tests/atmosphere/shallow_water/unit/test_spectral.py:291:        assert jnp.all(jnp.isfinite(state.vor_hat.data))
tests/atmosphere/shallow_water/unit/test_spectral.py:292:        assert jnp.all(jnp.isfinite(state.div_hat.data))
tests/atmosphere/shallow_water/unit/test_spectral.py:293:        assert jnp.all(jnp.isfinite(state.phi_hat.data))
tests/atmosphere/shallow_water/unit/test_spectral.py:297:        """Test Case 5 initialization produces finite spectral coefficients."""
tests/atmosphere/shallow_water/unit/test_spectral.py:299:        assert jnp.all(jnp.isfinite(state.vor_hat.data))
tests/atmosphere/shallow_water/unit/test_spectral.py:300:        assert jnp.all(jnp.isfinite(state.phi_hat.data))
tests/atmosphere/shallow_water/unit/test_spectral.py:307:        config = SpectralSWConfig(hyperdiff_coeff=0.0, mean_depth=2.94e4 / constants.g)
tests/atmosphere/shallow_water/unit/test_spectral.py:323:    def test_single_step_finite(self, grid_t21):
tests/atmosphere/shallow_water/unit/test_spectral.py:324:        """One time step should produce all-finite state."""
tests/atmosphere/shallow_water/unit/test_spectral.py:326:        nu = _proper_hyperdiff(grid_t21)
tests/atmosphere/shallow_water/unit/test_spectral.py:327:        config = SpectralSWConfig(hyperdiff_coeff=nu)
tests/atmosphere/shallow_water/unit/test_spectral.py:330:        assert jnp.all(jnp.isfinite(state_new.vor_hat.data))
tests/atmosphere/shallow_water/unit/test_spectral.py:331:        assert jnp.all(jnp.isfinite(state_new.div_hat.data))
tests/atmosphere/shallow_water/unit/test_spectral.py:332:        assert jnp.all(jnp.isfinite(state_new.phi_hat.data))
tests/atmosphere/shallow_water/unit/test_spectral.py:337:        nu = _proper_hyperdiff(grid_t21)
tests/atmosphere/shallow_water/unit/test_spectral.py:338:        config = SpectralSWConfig(hyperdiff_coeff=nu)
tests/atmosphere/shallow_water/unit/test_spectral.py:342:        assert jnp.all(jnp.isfinite(state.vor_hat.data))
tests/atmosphere/shallow_water/unit/test_spectral.py:343:        assert jnp.all(jnp.isfinite(state.phi_hat.data))
tests/atmosphere/shallow_water/unit/test_spectral.py:356:        """Conservation diagnostics return finite values."""
tests/atmosphere/shallow_water/unit/test_spectral.py:362:        assert np.isfinite(diag['mass'])
tests/atmosphere/shallow_water/unit/test_spectral.py:363:        assert np.isfinite(diag['energy'])
tests/atmosphere/shallow_water/unit/test_spectral.py:365:    def test_differentiability(self, grid_t21):
tests/atmosphere/shallow_water/unit/test_spectral.py:366:        """jax.grad through a single step should produce finite gradients."""
tests/atmosphere/shallow_water/unit/test_spectral.py:368:        nu = _proper_hyperdiff(grid_t21)
tests/atmosphere/shallow_water/unit/test_spectral.py:369:        config = SpectralSWConfig(hyperdiff_coeff=nu,
tests/atmosphere/shallow_water/unit/test_spectral.py:379:        grad_fn = jax.grad(loss_fn)
tests/atmosphere/shallow_water/unit/test_spectral.py:380:        g = grad_fn(state.vor_hat.data)
tests/atmosphere/shallow_water/unit/test_spectral.py:381:        assert jnp.all(jnp.isfinite(g))
tests/unit/test_component_factory.py:22:    compute_diffusion,
tests/unit/test_component_factory.py:120:    """compute_diffusion returns physically sensible values."""
tests/unit/test_component_factory.py:125:        diff = compute_diffusion(grid, dc)
tests/unit/test_component_factory.py:126:        assert isinstance(diff, DiffusionCoeffs)
tests/unit/test_component_factory.py:131:        diff = compute_diffusion(grid, dc)
tests/unit/test_component_factory.py:132:        assert diff.A_h > 0.0
tests/unit/test_component_factory.py:134:    def test_hyperdiff_positive(self):
tests/unit/test_component_factory.py:136:        dc = DycoreConfig(dt=600.0, hyperdiff_scale=1.0)
tests/unit/test_component_factory.py:137:        diff = compute_diffusion(grid, dc)
tests/unit/test_component_factory.py:138:        assert diff.hyperdiff > 0.0
tests/unit/test_component_factory.py:143:        diff = compute_diffusion(grid, dc)
tests/unit/test_component_factory.py:144:        assert diff.div_damp > 0.0
tests/unit/test_component_factory.py:148:        dc = DycoreConfig(dt=600.0, hyperdiff_scale=0.0, div_damp_scale=0.0)
tests/unit/test_component_factory.py:149:        diff = compute_diffusion(grid, dc)
tests/unit/test_component_factory.py:150:        assert diff.hyperdiff == 0.0
tests/unit/test_component_factory.py:151:        assert diff.div_damp == 0.0
tests/unit/test_component_factory.py:228:        assert driver._hyperdiff > 0.0
tests/unit/test_component_factory.py:325:                discretization="finite_volume",
tests/unit/test_component_factory.py:344:                discretization="finite_volume",
tests/unit/test_component_factory.py:363:                discretization="finite_volume",
tests/unit/test_component_factory.py:386:    """Factory must reject unsupported hyperdiff/div_damp on latlon_cgrid."""
tests/unit/test_component_factory.py:388:    def test_hyperdiff_scale_accepted(self):
tests/unit/test_component_factory.py:389:        """hyperdiff_scale is used by the driver for moisture smoothing,
tests/unit/test_component_factory.py:398:                discretization="finite_volume",
tests/unit/test_component_factory.py:399:                hyperdiff_scale=2.0,
tests/unit/test_component_factory.py:413:                discretization="finite_volume",
tests/unit/test_component_factory.py:421:        """Default hyperdiff_scale=1.0, div_damp_scale=1.0 must not raise."""
tests/unit/test_component_factory.py:429:                discretization="finite_volume",
tests/unit/test_component_factory.py:436:        """hyperdiff_scale=0.0, div_damp_scale=0.0 (disabled) must not raise."""
tests/unit/test_component_factory.py:444:                discretization="finite_volume",
tests/unit/test_component_factory.py:445:                hyperdiff_scale=0.0,
tests/unit/test_component_factory.py:465:                discretization="finite_volume",
tests/unit/test_component_factory.py:476:        """A_h must not exceed the pole-cell diffusive CFL limit."""
tests/unit/test_component_factory.py:485:                discretization="finite_volume",
tests/unit/test_component_factory.py:506:                discretization="finite_volume",
tests/unit/test_component_factory.py:515:        assert jnp.all(jnp.isfinite(state.T.data))
tests/unit/test_component_factory.py:519:# 9. create_model() grid-aware routing for lat-lon finite_volume
tests/unit/test_component_factory.py:526:        """Axis-based resolution (name=None, config with finite_volume)
tests/unit/test_component_factory.py:539:                "atmosphere.discretization": "finite_volume",
tests/ocean/unit/test_barotropic_noise_invariant.py:182:    assert bool(jnp.all(jnp.isfinite(final_state.eta.data))), \
tests/ocean/unit/test_barotropic_noise_invariant.py:184:    assert bool(jnp.all(jnp.isfinite(final_state.u.data))), \
tests/ocean/unit/test_barotropic_noise_invariant.py:289:def test_implicit_solver_grad_smoke():
tests/ocean/unit/test_barotropic_noise_invariant.py:290:    """jax.grad through the implicit CN solver returns finite values."""
tests/ocean/unit/test_barotropic_noise_invariant.py:314:    g = jax.grad(loss)(eta0)
tests/ocean/unit/test_barotropic_noise_invariant.py:315:    assert bool(jnp.all(jnp.isfinite(g))), "jax.grad produced NaN/Inf"
tests/ocean/unit/test_barotropic_noise_invariant.py:317:        "jax.grad returned exactly zero gradient"
tests/validation/test_ec_eigenvalues2.py:1:"""Test eigenvalues with different PGF formulations and adiabatic discretizations.
tests/validation/test_ec_eigenvalues2.py:189:    # Check if E_mat is positive definite
tests/unit/test_emanuel.py:5:* tendency shape / dtype / finiteness;
tests/unit/test_emanuel.py:11:* finite gradients through ``cape_threshold`` and
tests/unit/test_emanuel.py:37:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/unit/test_emanuel.py:63:# Shape / finiteness / no CMT
tests/unit/test_emanuel.py:78:def test_emanuel_outputs_finite():
tests/unit/test_emanuel.py:85:        assert jnp.all(jnp.isfinite(arr))
tests/unit/test_emanuel.py:112:def test_emanuel_n_fractions_finite_for_all_choices():
tests/unit/test_emanuel.py:113:    """Tendencies are finite for ``n_mixing_fractions`` ∈ {1, 4, 8, 16}.
tests/unit/test_emanuel.py:122:        assert jnp.all(jnp.isfinite(out.dT_dt)), (
tests/unit/test_emanuel.py:154:    sub-cloud cooling term — the difference in ``dT_dt`` between the
tests/unit/test_emanuel.py:155:    on/off configurations is non-zero in the surface-adjacent
tests/unit/test_emanuel.py:168:    # Sub-cloud (last 4 levels) tendencies differ between the two
tests/unit/test_emanuel.py:170:    diff = jnp.abs(out_on.dT_dt[:, -4:] - out_off.dT_dt[:, -4:])
tests/unit/test_emanuel.py:171:    assert float(jnp.max(diff)) > 1e-8
tests/unit/test_emanuel.py:178:def test_emanuel_grad_through_cape_threshold():
tests/unit/test_emanuel.py:188:    g = float(jax.grad(f)(jnp.asarray(70.0)))
tests/unit/test_emanuel.py:189:    assert np.isfinite(g)
tests/unit/test_emanuel.py:192:def test_emanuel_grad_through_smooth_trigger_sharpness():
tests/unit/test_emanuel.py:202:    g = float(jax.grad(f)(jnp.asarray(0.5)))
tests/unit/test_emanuel.py:203:    assert np.isfinite(g)
tests/unit/test_emanuel.py:223:def test_emanuel_orchestrator_one_step_finite():
tests/unit/test_emanuel.py:239:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_emanuel.py:247:        assert jnp.all(jnp.isfinite(f.data))
tests/stress/test_phase1_land_carbon.py:12:    step_carbon_differland,
tests/stress/test_phase1_land_carbon.py:26:        config = CarbonConfig(scheme="differland")
tests/stress/test_phase1_land_carbon.py:36:        assert jnp.all(jnp.isfinite(gpp_280)), "GPP(280) not finite"
tests/stress/test_phase1_land_carbon.py:37:        assert jnp.all(jnp.isfinite(gpp_560)), "GPP(560) not finite"
tests/stress/test_phase1_land_carbon.py:49:        config = CarbonConfig(scheme="differland")
tests/stress/test_phase1_land_carbon.py:74:            state, co2_flux = step_carbon_differland(
tests/stress/test_phase1_land_carbon.py:95:        """NEE should differ meaningfully between cold and warm conditions."""
tests/stress/test_phase1_land_carbon.py:96:        config = CarbonConfig(scheme="differland")
tests/stress/test_phase1_land_carbon.py:111:        _, nee_cold = step_carbon_differland(
tests/stress/test_phase1_land_carbon.py:114:        _, nee_warm = step_carbon_differland(
tests/stress/test_phase1_land_carbon.py:118:        assert jnp.all(jnp.isfinite(nee_cold)), "NEE(cold) not finite"
tests/stress/test_phase1_land_carbon.py:119:        assert jnp.all(jnp.isfinite(nee_warm)), "NEE(warm) not finite"
tests/stress/test_phase1_land_carbon.py:122:        # and respiration (via Q10).  The key assertion is that they differ.
tests/stress/test_phase1_land_carbon.py:123:        diff = jnp.max(jnp.abs(nee_warm - nee_cold))
tests/stress/test_phase1_land_carbon.py:124:        assert diff > 1e-12, (
tests/stress/test_phase1_land_carbon.py:125:            f"NEE should differ between 280K and 300K, max|diff|={float(diff):.3e}"
tests/stress/test_phase1_land_carbon.py:160:        assert jnp.all(jnp.isfinite(gs_night)), "gs(night) not finite"
tests/stress/test_phase1_land_carbon.py:161:        assert jnp.all(jnp.isfinite(gs_day)), "gs(day) not finite"
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:118:    def test_tendencies_finite(self, grid):
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:121:        assert jnp.all(jnp.isfinite(dh))
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:122:        assert jnp.all(jnp.isfinite(du))
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:123:        assert jnp.all(jnp.isfinite(dv))
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:134:        pressure-gradient balance.  At C32 (~600 km) this is ~O(1e-4).
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:155:    def test_gradient_y_zero_at_poles(self, grid):
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:156:        from legoesm.ocean.dynamics.latlon_cgrid_operators import gradient_y_cgrid
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:158:        grad_y = gradient_y_cgrid(h, grid)
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:159:        assert jnp.allclose(grad_y[0, :], 0.0)
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:160:        assert jnp.allclose(grad_y[-1, :], 0.0)
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:203:        assert jnp.all(jnp.isfinite(state.h))
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:204:        assert jnp.all(jnp.isfinite(state.u))
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:205:        assert jnp.all(jnp.isfinite(state.v))
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:233:        assert jnp.all(jnp.isfinite(state.h))
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:339:        assert jnp.all(jnp.isfinite(state.h))
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:340:        assert jnp.all(jnp.isfinite(state.u))
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:341:        assert jnp.all(jnp.isfinite(state.v))
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:363:        assert jnp.all(jnp.isfinite(state.h))
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:378:        assert jnp.all(jnp.isfinite(state.h))
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py:386:        assert jnp.all(jnp.isfinite(state.h))
tests/unit/test_diagnostic_collector.py:14:    dsigma = np.diff(np.linspace(0, 1, nlev + 1))
tests/unit/test_rce_script.py:47:        assert jnp.all(jnp.isfinite(state.T.data))
tests/unit/test_rce_script.py:48:        assert jnp.all(jnp.isfinite(state.u.data))
tests/unit/test_rce_script.py:69:        assert jnp.all(jnp.isfinite(rad.heating_rate))
tests/unit/test_rce_script.py:79:        assert jnp.all(jnp.isfinite(conv.dT_dt))
tests/atmosphere/hydrostatic/validation/test_held_suarez_fix.py:4:to verify stability after the pressure gradient force fix.
tests/atmosphere/hydrostatic/validation/test_held_suarez_fix.py:33:        hyperdiff_coeff=HYPERDIFF,
tests/atmosphere/hydrostatic/validation/test_held_suarez_fix.py:34:        hyperdiff_order=2,
tests/atmosphere/hydrostatic/validation/test_held_suarez_fix.py:58:    assert jnp.isfinite(jnp.array(T_mean)), "Temperature became non-finite"
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_1_mpas.py:3:Mountain-triggered breaking gravity waves on an icosahedral grid.
tests/unit/test_smooth.py:14:    """Tests for differentiable smooth approximations."""
tests/unit/test_smooth.py:63:    def test_all_differentiable(self):
tests/unit/test_smooth.py:64:        """All smooth functions should have finite gradients."""
tests/unit/test_smooth.py:75:            grad = jax.grad(fn)(x)
tests/unit/test_smooth.py:76:            assert jnp.isfinite(grad), f"{fn_name} has non-finite gradient at x=0"
tests/validation/test_differentiability_regression.py:1:"""Regression tests for differentiability fixes.
tests/validation/test_differentiability_regression.py:4:- jax.grad and jit(grad) work through combined atmospheric physics with
tests/validation/test_differentiability_regression.py:6:- Repeated differentiated calls do not leave Tracer objects in Python attributes
tests/validation/test_differentiability_regression.py:7:- grad through full coupler step (ocean SST, land T, sea-ice T, concentration)
tests/validation/test_differentiability_regression.py:58:    def test_grad_through_combined_physics(self, grid, sigma, hydrostatic_state):
tests/validation/test_differentiability_regression.py:59:        """jax.grad works through combined physics with PhysicsState."""
tests/validation/test_differentiability_regression.py:79:        grads = jax.grad(loss)(state.T.data)
tests/validation/test_differentiability_regression.py:80:        assert jnp.all(jnp.isfinite(grads))
tests/validation/test_differentiability_regression.py:82:    def test_jit_grad_through_combined_physics(self, grid, sigma, hydrostatic_state):
tests/validation/test_differentiability_regression.py:83:        """jit(grad) works through combined physics with PhysicsState."""
tests/validation/test_differentiability_regression.py:99:        def grad_fn(T_data):
tests/validation/test_differentiability_regression.py:104:            return jax.grad(loss)(T_data)
tests/validation/test_differentiability_regression.py:106:        grads = grad_fn(state.T.data)
tests/validation/test_differentiability_regression.py:107:        assert jnp.all(jnp.isfinite(grads))
tests/validation/test_differentiability_regression.py:110:        """Repeated differentiated calls don't leave Tracer objects in attributes."""
tests/validation/test_differentiability_regression.py:128:        # Run grad twice — no Tracer leakage
tests/validation/test_differentiability_regression.py:129:        g1 = jax.grad(loss)(state.T.data)
tests/validation/test_differentiability_regression.py:130:        g2 = jax.grad(loss)(state.T.data)
tests/validation/test_differentiability_regression.py:131:        assert jnp.all(jnp.isfinite(g1))
tests/validation/test_differentiability_regression.py:132:        assert jnp.all(jnp.isfinite(g2))
tests/validation/test_differentiability_regression.py:166:    def test_pe_grad_with_mass_fixer(self, grid, sigma):
tests/validation/test_differentiability_regression.py:167:        """PE dycore with mass fixer is differentiable."""
tests/validation/test_differentiability_regression.py:185:            hyperdiff_coeff=0.0,
tests/validation/test_differentiability_regression.py:195:        grads = jax.grad(loss)(state.u.data)
tests/validation/test_differentiability_regression.py:196:        assert jnp.all(jnp.isfinite(grads))
tests/validation/test_differentiability_regression.py:197:        assert not jnp.allclose(grads, 0.0)
tests/validation/test_differentiability_regression.py:201:# Task 4 & 5: Coupler differentiability
tests/validation/test_differentiability_regression.py:206:    def test_grad_through_coupler_sst(self):
tests/validation/test_differentiability_regression.py:207:        """grad through coupler step wrt ocean SST."""
tests/validation/test_differentiability_regression.py:266:        grads = jax.grad(loss)(ocean_sst)
tests/validation/test_differentiability_regression.py:267:        assert jnp.all(jnp.isfinite(grads))
tests/validation/test_differentiability_regression.py:268:        assert not jnp.allclose(grads, 0.0)
tests/validation/test_differentiability_regression.py:277:    def test_grad_through_ocean_step(self):
tests/validation/test_differentiability_regression.py:278:        """grad through OceanModel.step() (AD-safe kernel)."""
tests/validation/test_differentiability_regression.py:299:        grads = jax.grad(loss)(state.T.data)
tests/validation/test_differentiability_regression.py:300:        assert jnp.all(jnp.isfinite(grads))
tests/validation/test_differentiability_regression.py:302:    def test_jit_grad_through_ocean(self):
tests/validation/test_differentiability_regression.py:303:        """jit(grad) through OceanModel.step()."""
tests/validation/test_differentiability_regression.py:320:        def grad_fn(T_data):
tests/validation/test_differentiability_regression.py:325:            return jax.grad(loss)(T_data)
tests/validation/test_differentiability_regression.py:327:        grads = grad_fn(state.T.data)
tests/validation/test_differentiability_regression.py:328:        assert jnp.all(jnp.isfinite(grads))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:7:Covers: shape/finiteness, conservation, variance reduction, land masks,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:8:differentiability, Visbeck coefficient, and structural enforcement against
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:21:from legoesm.ocean.vertical import create_ocean_z_star, compute_ocean_jacobian
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:55:    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:56:    return grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:64:    - Meridional gradient proportional to requested slope
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:66:    grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian = \
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:71:    # Add meridional gradient.
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:87:            jacobian, rho, T, S, cfg)
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:91:# 1. Shape and finiteness
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:98:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:102:            rho, mask, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:110:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:113:            rho, mask, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:117:            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:120:        assert jnp.all(jnp.isfinite(dT))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:124:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:136:        assert jnp.all(jnp.isfinite(dT))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:137:        assert jnp.all(jnp.isfinite(dS))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:149:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:152:            rho, mask, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:157:            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:163:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian = _make_setup()
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:174:            rho, mask, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:189:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:192:            rho, mask, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:196:            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:198:        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:220:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:223:            rho, mask, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:227:            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:229:        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:247:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:261:            rho, mask_land, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:265:            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:274:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:286:            rho, mask_land, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:290:            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:292:        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:311:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:314:            rho, mask, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:318:            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:336:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:339:            rho, mask, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:343:            z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:361:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:364:            rho, mask, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:371:            rho, S_x, S_y, z_coord, jacobian, f_coriolis, vcfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:376:        assert jnp.all(jnp.isfinite(kappa))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:380:# 9. JAX differentiability
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:385:    def test_grad_through_full_tendency(self):
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:386:        """jax.grad must produce finite gradients through GM/Redi."""
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:388:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:391:            rho, mask, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:397:                z_coord, jacobian, grid, cfg.kappa_GM, cfg.kappa_Redi,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:401:        grad = jax.grad(loss)(T)
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:402:        assert jnp.all(jnp.isfinite(grad))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:404:    def test_grad_through_orchestrator(self):
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:405:        """jax.grad must flow through the full orchestrator (EOS + slopes + tendency)."""
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:407:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:408:        # Pin centered: triads-orchestrator grad coverage lives in
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:409:        # TestTriadDifferentiability.test_grad_through_triad.
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:419:        grad = jax.grad(loss)(T)
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:420:        assert jnp.all(jnp.isfinite(grad))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:432:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:435:            rho, mask, z_coord, jacobian, grid, cfg,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:439:        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:459:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:467:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:468:        assert jnp.all(jnp.isfinite(out.dS_dt))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:516:    """The triad function must produce finite, correctly-shaped output."""
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:520:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:523:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:528:        assert jnp.all(jnp.isfinite(dT))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:535:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:538:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:542:        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:556:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:560:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:571:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:584:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:594:    def test_grad_through_triad(self):
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:596:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:601:                z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:607:        grad = jax.grad(loss)(T)
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:608:        assert jnp.all(jnp.isfinite(grad))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:618:    def test_triad_with_2d_kappa_GM_runs_and_finite(self):
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:620:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:628:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:633:        assert jnp.all(jnp.isfinite(dT))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:640:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:648:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:654:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:664:        """slope_scheme='triads' produces finite, correctly-shaped output."""
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:666:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:675:        assert jnp.all(jnp.isfinite(dT))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:676:        assert jnp.all(jnp.isfinite(dS))
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:680:        grid, z_coord, mask, u_mask, v_mask, eta, H_bathy, jacobian, rho, T, S, cfg = setup
tests/validation/test_ensemble_correctness.py:6:invariant for differentiable ensemble data assimilation and ensemble
tests/validation/test_ensemble_correctness.py:12:- 2 ensemble members with different initial temperature (280 K, 281 K).
tests/validation/test_ensemble_correctness.py:244:    from legoesm.core.operators_3d import hyperdiffusion_3d
tests/validation/test_ensemble_correctness.py:326:                q_v_upd + _dt * hyperdiffusion_3d(q_v_upd, grid, qv_smooth_coeff),
tests/validation/test_ensemble_correctness.py:386:    This is the key invariant for differentiable ensemble workflows:
tests/validation/test_ensemble_correctness.py:387:    batched execution must not introduce numerical differences compared
tests/validation/test_ensemble_correctness.py:397:        # Two ensemble members: different initial T
tests/validation/test_ensemble_correctness.py:421:        non-zero tolerance since operation ordering may differ.
tests/validation/test_ensemble_correctness.py:423:        # Diagnostic accumulators where ULP-level differences from
tests/validation/test_ensemble_correctness.py:455:                        f"vmap result differs from independent run"
tests/validation/test_ensemble_correctness.py:463:                        f"vmap result differs from independent run"
tests/validation/test_ensemble_correctness.py:477:                        f"vmap result differs from independent run"
tests/validation/test_ensemble_correctness.py:485:                        f"vmap result differs from independent run"
tests/validation/test_ensemble_correctness.py:489:    def test_ensemble_members_differ(self):
tests/validation/test_ensemble_correctness.py:490:        """Sanity check: the two ensemble members produce different results.
tests/validation/test_ensemble_correctness.py:493:        pass vacuously.  This confirms the 1 K initial T difference
tests/validation/test_ensemble_correctness.py:506:        # The 1 K initial difference should persist (mock dynamics adds
tests/validation/test_ensemble_correctness.py:510:            "Ensemble members should differ in temperature"
tests/validation/test_ensemble_correctness.py:512:        # The difference should be O(1 K), not catastrophically large.
tests/validation/test_ensemble_correctness.py:513:        max_diff = np.max(np.abs(T_member_0 - T_member_1))
tests/validation/test_ensemble_correctness.py:514:        assert 0.5 < max_diff < 5.0, (
tests/validation/test_ensemble_correctness.py:515:            f"Temperature difference between members is {max_diff:.4f} K, "
tests/validation/test_ensemble_correctness.py:535:    def test_vmap_all_fields_finite(self):
tests/validation/test_ensemble_correctness.py:536:        """All carry fields remain finite after vmapped ensemble integration."""
tests/validation/test_ensemble_correctness.py:548:                assert np.all(np.isfinite(arr)), (
tests/validation/test_ensemble_correctness.py:549:                    f"Member {m}, field '{field_name}' has non-finite values"
tests/validation/test_ensemble_correctness.py:559:        # Diagnostic accumulators where ULP-level differences from
tests/validation/test_ensemble_correctness.py:583:                        f"differs from compiled segment fn"
tests/validation/test_ensemble_correctness.py:590:                        f"Field '{field_name}': raw segment fn differs "
tests/validation/bench_spectral_pe.py:63:def proper_hyperdiff(grid, tau_hours=4.0):
tests/validation/bench_spectral_pe.py:64:    """∇⁴ hyperdiffusion with given e-folding time at truncation wavenumber."""
tests/validation/bench_spectral_pe.py:149:    nu21 = proper_hyperdiff(grid21)
tests/validation/bench_spectral_pe.py:150:    config21 = SpectralPEConfig(hyperdiff_coeff=nu21, hyperdiff_order=2)
tests/validation/bench_spectral_pe.py:191:    nu = proper_hyperdiff(grid)
tests/validation/bench_spectral_pe.py:193:    # - Leapfrog is neutral for oscillatory gravity-wave modes
tests/validation/bench_spectral_pe.py:194:    # - SI treats gravity waves implicitly (Hoskins & Simmons 1975)
tests/validation/bench_spectral_pe.py:197:    # - Level-dependent diffusion (pscale) strengthens damping at low-p
tests/validation/bench_spectral_pe.py:199:        hyperdiff_coeff=nu,
tests/validation/bench_spectral_pe.py:200:        hyperdiff_order=2,
tests/validation/bench_spectral_pe.py:208:        hyperdiff_pscale=0.5,           # level-dependent diffusion
tests/validation/bench_spectral_pe.py:301:    jw_pass = (np.all(np.isfinite(max_wind_ts)) and
tests/validation/bench_spectral_pe.py:355:    # 3. Zonal-mean T profiles at different times
tests/validation/bench_spectral_pe.py:483:    summary.append(f"  All finite:      {bool(np.all(np.isfinite(max_wind_ts)))}")
tests/unit/test_ensemble.py:7:- Ensemble scan integration (correctness, differentiability)
tests/unit/test_ensemble.py:151:    def test_members_differ(self):
tests/unit/test_ensemble.py:152:        """Ensemble members have different values."""
tests/unit/test_ensemble.py:158:        # Members 0 and 1 should differ
tests/unit/test_ensemble.py:179:        # T should differ across members
tests/unit/test_ensemble.py:183:        # u should NOT differ (not in fields list)
tests/unit/test_ensemble.py:206:    def test_different_keys_different_results(self):
tests/unit/test_ensemble.py:207:        """Different PRNG keys produce different ensembles."""
tests/unit/test_ensemble.py:275:        """Perturbed members should produce different results after stepping."""
tests/unit/test_ensemble.py:350:    def test_differentiable(self):
tests/unit/test_ensemble.py:365:        grad = jax.grad(loss_fn)(s.T.data)
tests/unit/test_ensemble.py:366:        assert grad.shape == s.T.data.shape
tests/unit/test_ensemble.py:367:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_ensemble.py:564:        # Members should differ
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_2.py:112:def _gradient_wind_balanced_rho_prime(
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_2.py:117:    """Compute density perturbation for gradient-wind balance with u = u0·cos(lat).
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_2.py:120:    gradient-wind balance equation (momentum equation in steady state with v=0)
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_2.py:158:    # Exner perturbation from gradient-wind balance
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_2.py:234:    # Note: _gradient_wind_balanced_rho_prime() is available for
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_2.py:235:    # gradient-wind balanced initialization, but the explicit split-explicit
tests/atmosphere/hydrostatic/integration/test_amip_smoke.py:75:        hyperdiff_coeff=1e15,
tests/atmosphere/hydrostatic/integration/test_amip_smoke.py:76:        hyperdiff_ps_coeff=1e15,
tests/atmosphere/hydrostatic/integration/test_amip_smoke.py:156:    # Verify everything is finite
tests/atmosphere/hydrostatic/integration/test_amip_smoke.py:157:    assert jnp.all(jnp.isfinite(state.T.data)), "Temperature not finite"
tests/atmosphere/hydrostatic/integration/test_amip_smoke.py:158:    assert jnp.all(jnp.isfinite(state.u.data)), "Winds not finite"
tests/atmosphere/hydrostatic/integration/test_amip_smoke.py:159:    assert jnp.all(jnp.isfinite(state.p_s.data)), "Surface pressure not finite"
tests/land/validation/test_bulk_flux_differentiability.py:2:"""Test JAX differentiability of MOST bulk flux algorithms.
tests/land/validation/test_bulk_flux_differentiability.py:4:Verifies that jax.grad works through the Obukhov length iteration
tests/land/validation/test_bulk_flux_differentiability.py:26:    print("\n--- Stability function gradients ---")
tests/land/validation/test_bulk_flux_differentiability.py:27:    dpsi_m = jax.grad(lambda z: psi_m(z).sum())
tests/land/validation/test_bulk_flux_differentiability.py:28:    dpsi_h = jax.grad(lambda z: psi_h(z).sum())
tests/land/validation/test_bulk_flux_differentiability.py:63:            grads = jax.grad(loss)(T_sfc)
tests/land/validation/test_bulk_flux_differentiability.py:64:            is_finite = bool(jnp.all(jnp.isfinite(grads)))
tests/land/validation/test_bulk_flux_differentiability.py:65:            is_nonzero = bool(not jnp.allclose(grads, 0.0))
tests/land/validation/test_bulk_flux_differentiability.py:66:            grad_norm = float(jnp.max(jnp.abs(grads)))
tests/land/validation/test_bulk_flux_differentiability.py:68:            if is_finite and is_nonzero:
tests/land/validation/test_bulk_flux_differentiability.py:69:                print(f"  PASS  n_iter={n_iter:2d}  |grad|_max={grad_norm:.4e}")
tests/land/validation/test_bulk_flux_differentiability.py:70:            elif is_finite:
tests/land/validation/test_bulk_flux_differentiability.py:71:                print(f"  WARN  n_iter={n_iter:2d}  gradients are all zero")
tests/land/validation/test_bulk_flux_differentiability.py:74:                print(f"  FAIL  n_iter={n_iter:2d}  non-finite gradients, |grad|_max={grad_norm}")
tests/land/validation/test_bulk_flux_differentiability.py:77:        # Test gradient w.r.t. wind speed
tests/land/validation/test_bulk_flux_differentiability.py:87:        grads_u = jax.grad(loss_wind)(u_rel)
tests/land/validation/test_bulk_flux_differentiability.py:88:        is_finite_u = bool(jnp.all(jnp.isfinite(grads_u)))
tests/land/validation/test_bulk_flux_differentiability.py:89:        is_nonzero_u = bool(not jnp.allclose(grads_u, 0.0))
tests/land/validation/test_bulk_flux_differentiability.py:90:        grad_norm_u = float(jnp.max(jnp.abs(grads_u)))
tests/land/validation/test_bulk_flux_differentiability.py:92:        if is_finite_u and is_nonzero_u:
tests/land/validation/test_bulk_flux_differentiability.py:93:            print(f"  PASS  grad w.r.t. wind  |grad|_max={grad_norm_u:.4e}")
tests/land/validation/test_bulk_flux_differentiability.py:95:            print(f"  FAIL  grad w.r.t. wind  finite={is_finite_u} nonzero={is_nonzero_u}")
tests/land/validation/test_bulk_flux_differentiability.py:174:            # Test differentiability through coupler
tests/land/validation/test_bulk_flux_differentiability.py:179:            grads = jax.grad(coupler_loss)(sst)
tests/land/validation/test_bulk_flux_differentiability.py:180:            is_finite = bool(jnp.all(jnp.isfinite(grads)))
tests/land/validation/test_bulk_flux_differentiability.py:181:            is_nonzero = bool(not jnp.allclose(grads, 0.0))
tests/land/validation/test_bulk_flux_differentiability.py:182:            grad_norm = float(jnp.max(jnp.abs(grads)))
tests/land/validation/test_bulk_flux_differentiability.py:184:            if is_finite and is_nonzero:
tests/land/validation/test_bulk_flux_differentiability.py:185:                print(f"  PASS  {scheme:12s}  |grad|_max={grad_norm:.4e}"
tests/land/validation/test_bulk_flux_differentiability.py:188:                print(f"  FAIL  {scheme:12s}  finite={is_finite} nonzero={is_nonzero}")
tests/unit/test_sea_ice_dynamics.py:16:- JAX differentiability
tests/unit/test_sea_ice_dynamics.py:159:    def test_finite(self):
tests/unit/test_sea_ice_dynamics.py:166:        assert jnp.all(jnp.isfinite(eps_11))
tests/unit/test_sea_ice_dynamics.py:167:        assert jnp.all(jnp.isfinite(eps_22))
tests/unit/test_sea_ice_dynamics.py:168:        assert jnp.all(jnp.isfinite(eps_12))
tests/unit/test_sea_ice_dynamics.py:210:        assert jnp.isfinite(s12)
tests/unit/test_sea_ice_dynamics.py:304:            differentiable=False,
tests/unit/test_sea_ice_dynamics.py:308:        assert jnp.all(jnp.isfinite(u_new))
tests/unit/test_sea_ice_dynamics.py:309:        assert jnp.all(jnp.isfinite(v_new))
tests/unit/test_sea_ice_dynamics.py:331:            differentiable=False,
tests/unit/test_sea_ice_dynamics.py:344:            differentiable=False,
tests/unit/test_sea_ice_dynamics.py:348:        assert jnp.all(jnp.isfinite(u_strong))
tests/unit/test_sea_ice_dynamics.py:351:    def test_finite_output(self):
tests/unit/test_sea_ice_dynamics.py:364:            assert jnp.all(jnp.isfinite(arr)), f"Non-finite values found"
tests/unit/test_sea_ice_dynamics.py:526:    def test_finite(self):
tests/unit/test_sea_ice_dynamics.py:536:        assert jnp.all(jnp.isfinite(h_new))
tests/unit/test_sea_ice_dynamics.py:537:        assert jnp.all(jnp.isfinite(a_new))
tests/unit/test_sea_ice_dynamics.py:538:        assert jnp.all(jnp.isfinite(T_new))
tests/unit/test_sea_ice_dynamics.py:571:        assert jnp.all(jnp.isfinite(new_state.h_ice.data))
tests/unit/test_sea_ice_dynamics.py:572:        assert jnp.all(jnp.isfinite(response.T_surface))
tests/unit/test_sea_ice_dynamics.py:606:        assert jnp.all(jnp.isfinite(new_state.h_ice.data))
tests/unit/test_sea_ice_dynamics.py:626:        assert jnp.all(jnp.isfinite(new_state.h_ice.data))
tests/unit/test_sea_ice_dynamics.py:627:        assert jnp.all(jnp.isfinite(new_state.u_ice.data))
tests/unit/test_sea_ice_dynamics.py:628:        assert jnp.all(jnp.isfinite(new_state.sigma_11.data))
tests/unit/test_sea_ice_dynamics.py:646:        assert jnp.all(jnp.isfinite(new_state.h_ice.data))
tests/unit/test_sea_ice_dynamics.py:665:        assert jnp.all(jnp.isfinite(state.h_ice.data))
tests/unit/test_sea_ice_dynamics.py:666:        assert jnp.all(jnp.isfinite(state.u_ice.data))
tests/unit/test_sea_ice_dynamics.py:700:        assert jnp.all(jnp.isfinite(new_state.h_ice.data))
tests/unit/test_sea_ice_dynamics.py:710:    def test_grad_through_rheology(self):
tests/unit/test_sea_ice_dynamics.py:711:        """ice_strength is differentiable."""
tests/unit/test_sea_ice_dynamics.py:714:        g = jax.grad(f)(jnp.array(1.0))
tests/unit/test_sea_ice_dynamics.py:715:        assert jnp.isfinite(g)
tests/unit/test_sea_ice_dynamics.py:718:    def test_grad_through_strain_rates(self):
tests/unit/test_sea_ice_dynamics.py:726:        g = jax.grad(f)(jnp.ones((6, n, n)))
tests/unit/test_sea_ice_dynamics.py:727:        assert jnp.all(jnp.isfinite(g))
tests/unit/test_sea_ice_dynamics.py:729:    def test_grad_through_evp(self):
tests/unit/test_sea_ice_dynamics.py:730:        """Full EVP solver should be differentiable in scan mode."""
tests/unit/test_sea_ice_dynamics.py:743:                differentiable=True,
tests/unit/test_sea_ice_dynamics.py:747:        g = jax.grad(f)(jnp.full(shape, 5.0))
tests/unit/test_sea_ice_dynamics.py:748:        assert jnp.all(jnp.isfinite(g))
tests/unit/test_sea_ice_dynamics.py:750:    def test_grad_through_slab_step(self):
tests/unit/test_sea_ice_dynamics.py:751:        """Slab step_sea_ice is differentiable."""
tests/unit/test_sea_ice_dynamics.py:770:        g = jax.grad(f)(jnp.full(shape, 260.0))
tests/unit/test_sea_ice_dynamics.py:771:        assert jnp.all(jnp.isfinite(g))
tests/validation/bench_spectral_sw.py:56:def proper_hyperdiff(grid):
tests/validation/bench_spectral_sw.py:220:    nu = proper_hyperdiff(grid)
tests/validation/bench_spectral_sw.py:222:    print(f"  Hyperdiffusion: nu={nu:.4e} (4-hour e-folding)")
tests/validation/bench_spectral_sw.py:229:        hyperdiff_coeff=nu,
tests/validation/bench_spectral_sw.py:259:    # TC5 needs stronger hyperdiffusion (1-hour e-folding) to handle
tests/validation/bench_spectral_sw.py:262:    config5 = SpectralSWConfig(hyperdiff_coeff=nu5)
tests/validation/bench_spectral_sw.py:283:    print(f"    All finite      = {bool(np.all(np.isfinite(h_final5)))}")
tests/validation/bench_spectral_sw.py:285:    tc5_pass = np.all(np.isfinite(h_final5)) and mass_drift5 < 1e-6
tests/validation/bench_spectral_sw.py:326:    summary.append(f"  All finite:        {bool(np.all(np.isfinite(h_final5)))}")
tests/validation/bench_spectral_sw.py:331:    summary.append("       (with hyperdiffusion: < 1 m at T42)")
tests/validation/validation_differentiability_all.py:1:"""Test JAX differentiability of maintained atmosphere discretizations.
tests/validation/validation_differentiability_all.py:4:hydrostatic, nonhydrostatic), compute jax.grad through 10 time steps
tests/validation/validation_differentiability_all.py:5:and verify the gradients are finite and non-zero.
tests/validation/validation_differentiability_all.py:107:def _run_grad_test(model, state, dt, grad_field="u"):
tests/validation/validation_differentiability_all.py:108:    """Compute jax.grad through N_STEPS and return (is_finite, is_nonzero)."""
tests/validation/validation_differentiability_all.py:110:        if grad_field == "h":
tests/validation/validation_differentiability_all.py:112:        elif grad_field == "u":
tests/validation/validation_differentiability_all.py:114:        elif grad_field == "theta_prime":
tests/validation/validation_differentiability_all.py:117:            raise ValueError(f"Unknown grad_field={grad_field}")
tests/validation/validation_differentiability_all.py:129:    if grad_field == "h":
tests/validation/validation_differentiability_all.py:131:    elif grad_field == "u":
tests/validation/validation_differentiability_all.py:133:    elif grad_field == "theta_prime":
tests/validation/validation_differentiability_all.py:136:        raise ValueError(f"Unknown grad_field={grad_field}")
tests/validation/validation_differentiability_all.py:138:    grads = jax.grad(loss)(data)
tests/validation/validation_differentiability_all.py:139:    return grads
tests/validation/validation_differentiability_all.py:155:        grads = _run_grad_test(model, sw_state, DT_SW, grad_field="h")
tests/validation/validation_differentiability_all.py:156:        assert jnp.all(jnp.isfinite(grads)), "non-finite gradients"
tests/validation/validation_differentiability_all.py:157:        assert not jnp.allclose(grads, 0.0), "gradients are all zero"
tests/validation/validation_differentiability_all.py:168:            hyperdiff_coeff=0.0, use_conservation_fixer=False,
tests/validation/validation_differentiability_all.py:170:        grads = _run_grad_test(model, pe_state, DT_PE)
tests/validation/validation_differentiability_all.py:171:        assert jnp.all(jnp.isfinite(grads)), "non-finite gradients"
tests/validation/validation_differentiability_all.py:172:        assert not jnp.allclose(grads, 0.0), "gradients are all zero"
tests/validation/validation_differentiability_all.py:177:    @pytest.mark.xfail(reason="Zero initial perturbation produces zero gradients")
tests/validation/validation_differentiability_all.py:186:                hyperdiff_coeff=0.0, sponge_coeff=0.0,
tests/validation/validation_differentiability_all.py:189:        grads = _run_grad_test(model, ce_state, DT_CE, grad_field="theta_prime")
tests/validation/validation_differentiability_all.py:190:        assert jnp.all(jnp.isfinite(grads)), "non-finite gradients"
tests/validation/validation_differentiability_all.py:191:        assert not jnp.allclose(grads, 0.0), "gradients are all zero"
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:34:            hyperdiff_coeff=1e14,
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:59:    def test_tendencies_finite(self, model_state_dt):
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:62:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:63:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:64:        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:65:        assert jnp.all(jnp.isfinite(tend.dp_s_dt.data))
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:70:        assert jnp.all(jnp.isfinite(s1.T.data))
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:71:        assert jnp.all(jnp.isfinite(s1.u.data))
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:72:        assert jnp.all(jnp.isfinite(s1.p_s.data))
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:80:        assert jnp.all(jnp.isfinite(s.T.data))
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:81:        assert jnp.all(jnp.isfinite(s.u.data))
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:82:        assert jnp.all(jnp.isfinite(s.p_s.data))
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:86:    def test_differentiable_10_steps(self, grid_pe, model_state_dt):
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:87:        """FV PE should be differentiable through 10 steps."""
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:95:        grad_fn = jax.grad(loss_fn)
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:96:        g = grad_fn(state.T.data)
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:97:        assert jnp.all(jnp.isfinite(g))
tests/ocean/unit/test_ocean_differentiability.py:3:Tests jax.grad through:
tests/ocean/unit/test_ocean_differentiability.py:25:    """Test jax.grad through smagorinsky_biharmonic_3d on MPAS mesh."""
tests/ocean/unit/test_ocean_differentiability.py:38:    def test_grad_finite_nonzero(self, mpas_setup):
tests/ocean/unit/test_ocean_differentiability.py:39:        """jax.grad of sum(tendency) w.r.t. velocity is finite and nonzero.
tests/ocean/unit/test_ocean_differentiability.py:44:        underflows to ~1e-44 and gradients vanish numerically.
tests/ocean/unit/test_ocean_differentiability.py:53:        grad = jax.grad(loss)(u_edge)
tests/ocean/unit/test_ocean_differentiability.py:54:        assert jnp.all(jnp.isfinite(grad)), (
tests/ocean/unit/test_ocean_differentiability.py:55:            f"Non-finite gradients in Smagorinsky biharmonic MPAS. "
tests/ocean/unit/test_ocean_differentiability.py:56:            f"NaN count: {int(jnp.sum(jnp.isnan(grad)))}, "
tests/ocean/unit/test_ocean_differentiability.py:57:            f"Inf count: {int(jnp.sum(jnp.isinf(grad)))}"
tests/ocean/unit/test_ocean_differentiability.py:59:        nonzero_frac = float(jnp.mean(jnp.abs(grad) > 1e-30))
tests/ocean/unit/test_ocean_differentiability.py:61:            f"Too few nonzero gradient entries: {nonzero_frac:.1%}"
tests/ocean/unit/test_ocean_differentiability.py:64:    def test_grad_squared_loss_large_velocity(self, mpas_setup):
tests/ocean/unit/test_ocean_differentiability.py:65:        """jax.grad of sum(tendency**2) works with larger velocities.
tests/ocean/unit/test_ocean_differentiability.py:68:        scale as u / dx^4 ~ 1e-24, so tendency**2 ~ 1e-48 and gradients
tests/ocean/unit/test_ocean_differentiability.py:70:        but below any naive 1e-30 cutoff.  We verify finiteness and that
tests/ocean/unit/test_ocean_differentiability.py:71:        all entries are strictly nonzero, confirming the gradient flows.
tests/ocean/unit/test_ocean_differentiability.py:84:        grad = jax.grad(loss)(u_large)
tests/ocean/unit/test_ocean_differentiability.py:85:        assert jnp.all(jnp.isfinite(grad)), "Non-finite grad with large velocity"
tests/ocean/unit/test_ocean_differentiability.py:87:        nonzero_frac = float(jnp.mean(grad != 0.0))
tests/ocean/unit/test_ocean_differentiability.py:89:            f"Too few nonzero gradient entries with large velocity: {nonzero_frac:.1%}"
tests/ocean/unit/test_ocean_differentiability.py:92:    def test_grad_jit_consistent(self, mpas_setup):
tests/ocean/unit/test_ocean_differentiability.py:93:        """jax.jit(jax.grad(loss)) matches jax.grad(loss)."""
tests/ocean/unit/test_ocean_differentiability.py:101:        grad_eager = jax.grad(loss)(u_edge)
tests/ocean/unit/test_ocean_differentiability.py:102:        grad_jit = jax.jit(jax.grad(loss))(u_edge)
tests/ocean/unit/test_ocean_differentiability.py:103:        assert jnp.allclose(grad_eager, grad_jit, atol=1e-10), (
tests/ocean/unit/test_ocean_differentiability.py:104:            f"JIT gradient differs from eager. Max diff: "
tests/ocean/unit/test_ocean_differentiability.py:105:            f"{float(jnp.max(jnp.abs(grad_eager - grad_jit))):.2e}"
tests/ocean/unit/test_ocean_differentiability.py:114:    """Test jax.grad through baroclinic tendencies with bottom_drag_r > 0."""
tests/ocean/unit/test_ocean_differentiability.py:156:    def test_grad_wrt_temperature(self, latlon_cgrid_setup):
tests/ocean/unit/test_ocean_differentiability.py:157:        """Gradient w.r.t. T through tendencies with bottom drag is finite."""
tests/ocean/unit/test_ocean_differentiability.py:165:        grad = jax.grad(loss)(state.T.data)
tests/ocean/unit/test_ocean_differentiability.py:166:        assert jnp.all(jnp.isfinite(grad)), "Non-finite grad w.r.t. T with bottom drag"
tests/ocean/unit/test_ocean_differentiability.py:167:        # T gradient may be mostly zero if drag doesn't affect tracers much,
tests/ocean/unit/test_ocean_differentiability.py:168:        # but should still be finite.
tests/ocean/unit/test_ocean_differentiability.py:170:    def test_grad_wrt_velocity(self, latlon_cgrid_setup):
tests/ocean/unit/test_ocean_differentiability.py:171:        """Gradient w.r.t. u through tendencies with bottom drag is finite and nonzero."""
tests/ocean/unit/test_ocean_differentiability.py:179:        grad = jax.grad(loss)(state.u.data)
tests/ocean/unit/test_ocean_differentiability.py:180:        assert jnp.all(jnp.isfinite(grad)), (
tests/ocean/unit/test_ocean_differentiability.py:181:            f"Non-finite grad w.r.t. u with bottom drag. "
tests/ocean/unit/test_ocean_differentiability.py:182:            f"NaN: {int(jnp.sum(jnp.isnan(grad)))}"
tests/ocean/unit/test_ocean_differentiability.py:184:        nonzero_frac = float(jnp.mean(jnp.abs(grad) > 1e-30))
tests/ocean/unit/test_ocean_differentiability.py:186:            f"Too few nonzero gradient entries for u: {nonzero_frac:.1%}"
tests/ocean/unit/test_ocean_differentiability.py:189:    def test_bottom_drag_changes_gradient(self, latlon_cgrid_setup):
tests/ocean/unit/test_ocean_differentiability.py:190:        """Bottom drag actually contributes to the gradient (not a dead code path)."""
tests/ocean/unit/test_ocean_differentiability.py:201:        grad_drag = jax.grad(loss)(state.u.data, config)
tests/ocean/unit/test_ocean_differentiability.py:202:        grad_no_drag = jax.grad(loss)(state.u.data, config_no_drag)
tests/ocean/unit/test_ocean_differentiability.py:203:        diff = float(jnp.max(jnp.abs(grad_drag - grad_no_drag)))
tests/ocean/unit/test_ocean_differentiability.py:204:        assert diff > 1e-15, (
tests/ocean/unit/test_ocean_differentiability.py:205:            f"Bottom drag has no effect on gradient. Max diff = {diff:.2e}"
tests/ocean/unit/test_ocean_differentiability.py:214:    """Test jax.grad through sponge tendency gamma * (T_ref - T)."""
tests/ocean/unit/test_ocean_differentiability.py:257:    def test_grad_wrt_temperature(self, sponge_setup):
tests/ocean/unit/test_ocean_differentiability.py:266:        grad = jax.grad(loss)(state.T.data)
tests/ocean/unit/test_ocean_differentiability.py:267:        assert jnp.all(jnp.isfinite(grad)), "Non-finite grad through sponge relaxation"
tests/ocean/unit/test_ocean_differentiability.py:268:        nonzero_frac = float(jnp.mean(jnp.abs(grad) > 1e-30))
tests/ocean/unit/test_ocean_differentiability.py:270:            f"Too few nonzero gradient entries through sponge: {nonzero_frac:.1%}"
tests/ocean/unit/test_ocean_differentiability.py:273:    def test_sponge_contributes_to_gradient(self, sponge_setup):
tests/ocean/unit/test_ocean_differentiability.py:274:        """Sponge relaxation is not a dead code path in gradient computation."""
tests/ocean/unit/test_ocean_differentiability.py:287:        grad_with = jax.grad(loss_with_sponge)(state.T.data)
tests/ocean/unit/test_ocean_differentiability.py:288:        grad_without = jax.grad(loss_without_sponge)(state.T.data)
tests/ocean/unit/test_ocean_differentiability.py:289:        diff = float(jnp.max(jnp.abs(grad_with - grad_without)))
tests/ocean/unit/test_ocean_differentiability.py:290:        assert diff > 1e-15, (
tests/ocean/unit/test_ocean_differentiability.py:291:            f"Sponge has no effect on gradient. Max diff = {diff:.2e}"
tests/ocean/unit/test_ocean_differentiability.py:327:        grad = jax.grad(loss)(state_pert.u.data)
tests/ocean/unit/test_ocean_differentiability.py:328:        assert jnp.all(jnp.isfinite(grad)), (
tests/ocean/unit/test_ocean_differentiability.py:329:            "Non-finite grad through sponge velocity relaxation"
tests/ocean/unit/test_ocean_differentiability.py:338:    """Test jax.grad through barotropic substeps with bottom_drag_r > 0."""
tests/ocean/unit/test_ocean_differentiability.py:374:        # Must use differentiable_barotropic=True to use lax.scan
tests/ocean/unit/test_ocean_differentiability.py:382:            differentiable_barotropic=True,
tests/ocean/unit/test_ocean_differentiability.py:383:            barotropic_diffusion_alpha=0.01,
tests/ocean/unit/test_ocean_differentiability.py:389:    def test_grad_eta_wrt_eta(self, baro_setup):
tests/ocean/unit/test_ocean_differentiability.py:402:        grad = jax.grad(loss)(state.eta.data)
tests/ocean/unit/test_ocean_differentiability.py:403:        assert jnp.all(jnp.isfinite(grad)), (
tests/ocean/unit/test_ocean_differentiability.py:404:            f"Non-finite grad of eta through barotropic loop with bottom drag. "
tests/ocean/unit/test_ocean_differentiability.py:405:            f"NaN: {int(jnp.sum(jnp.isnan(grad)))}"
tests/ocean/unit/test_ocean_differentiability.py:407:        nonzero_frac = float(jnp.mean(jnp.abs(grad) > 1e-30))
tests/ocean/unit/test_ocean_differentiability.py:409:            f"Too few nonzero eta gradient entries: {nonzero_frac:.1%}"
tests/ocean/unit/test_ocean_differentiability.py:412:    def test_grad_u_wrt_u(self, baro_setup):
tests/ocean/unit/test_ocean_differentiability.py:425:        grad = jax.grad(loss)(state.u.data)
tests/ocean/unit/test_ocean_differentiability.py:426:        assert jnp.all(jnp.isfinite(grad)), (
tests/ocean/unit/test_ocean_differentiability.py:427:            f"Non-finite grad of u through barotropic loop with bottom drag. "
tests/ocean/unit/test_ocean_differentiability.py:428:            f"NaN: {int(jnp.sum(jnp.isnan(grad)))}"
tests/ocean/unit/test_ocean_differentiability.py:430:        nonzero_frac = float(jnp.mean(jnp.abs(grad) > 1e-30))
tests/ocean/unit/test_ocean_differentiability.py:432:            f"Too few nonzero u gradient entries: {nonzero_frac:.1%}"
tests/ocean/unit/test_ocean_differentiability.py:435:    def test_bottom_drag_changes_barotropic_gradient(self, baro_setup):
tests/ocean/unit/test_ocean_differentiability.py:436:        """Bottom drag in barotropic solver contributes to the gradient."""
tests/ocean/unit/test_ocean_differentiability.py:450:        grad_drag = jax.grad(loss)(state.u.data, config)
tests/ocean/unit/test_ocean_differentiability.py:451:        grad_no_drag = jax.grad(loss)(state.u.data, config_no_drag)
tests/ocean/unit/test_ocean_differentiability.py:452:        diff = float(jnp.max(jnp.abs(grad_drag - grad_no_drag)))
tests/ocean/unit/test_ocean_differentiability.py:453:        assert diff > 1e-15, (
tests/ocean/unit/test_ocean_differentiability.py:454:            f"Bottom drag in barotropic solver has no effect on gradient. "
tests/ocean/unit/test_ocean_differentiability.py:455:            f"Max diff = {diff:.2e}"
tests/ocean/unit/test_ocean_differentiability.py:459:        """Verify fori_loop path (non-differentiable) runs without error.
tests/ocean/unit/test_ocean_differentiability.py:461:        This is not a differentiability test — it just verifies that the
tests/ocean/unit/test_ocean_differentiability.py:462:        fori_loop path with bottom drag doesn't crash. jax.grad through
tests/ocean/unit/test_ocean_differentiability.py:467:        config_fori = config._replace(differentiable_barotropic=False)
tests/ocean/unit/test_ocean_differentiability.py:473:        assert jnp.all(jnp.isfinite(s_out.eta.data)), "Non-finite eta from fori_loop"
tests/ocean/unit/test_ocean_differentiability.py:507:        grad = jax.grad(loss_smag)(u_edge)
tests/ocean/unit/test_ocean_differentiability.py:508:        ok = bool(jnp.all(jnp.isfinite(grad)))
tests/ocean/unit/test_ocean_differentiability.py:509:        nz = float(jnp.mean(jnp.abs(grad) > 1e-30))
tests/ocean/unit/test_ocean_differentiability.py:511:            print(f"  PASS  |grad|_max={float(jnp.max(jnp.abs(grad))):.4e}, "
tests/ocean/unit/test_ocean_differentiability.py:514:            print(f"  FAIL  finite={ok}, nonzero_frac={nz:.1%}")
tests/ocean/unit/test_ocean_differentiability.py:550:        grad = jax.grad(loss_bd)(state.u.data)
tests/ocean/unit/test_ocean_differentiability.py:551:        ok = bool(jnp.all(jnp.isfinite(grad)))
tests/ocean/unit/test_ocean_differentiability.py:552:        nz = float(jnp.mean(jnp.abs(grad) > 1e-30))
tests/ocean/unit/test_ocean_differentiability.py:554:            print(f"  PASS  |grad|_max={float(jnp.max(jnp.abs(grad))):.4e}, "
tests/ocean/unit/test_ocean_differentiability.py:557:            print(f"  FAIL  finite={ok}, nonzero_frac={nz:.1%}")
tests/ocean/unit/test_ocean_differentiability.py:590:        grad = jax.grad(loss_sp)(state.T.data)
tests/ocean/unit/test_ocean_differentiability.py:591:        ok = bool(jnp.all(jnp.isfinite(grad)))
tests/ocean/unit/test_ocean_differentiability.py:592:        nz = float(jnp.mean(jnp.abs(grad) > 1e-30))
tests/ocean/unit/test_ocean_differentiability.py:594:            print(f"  PASS  |grad|_max={float(jnp.max(jnp.abs(grad))):.4e}, "
tests/ocean/unit/test_ocean_differentiability.py:597:            print(f"  FAIL  finite={ok}, nonzero_frac={nz:.1%}")
tests/ocean/unit/test_ocean_differentiability.py:631:            differentiable_barotropic=True, barotropic_diffusion_alpha=0.01,
tests/ocean/unit/test_ocean_differentiability.py:643:        grad = jax.grad(loss_bt)(state.eta.data)
tests/ocean/unit/test_ocean_differentiability.py:644:        ok = bool(jnp.all(jnp.isfinite(grad)))
tests/ocean/unit/test_ocean_differentiability.py:645:        nz = float(jnp.mean(jnp.abs(grad) > 1e-30))
tests/ocean/unit/test_ocean_differentiability.py:647:            print(f"  PASS  |grad|_max={float(jnp.max(jnp.abs(grad))):.4e}, "
tests/ocean/unit/test_ocean_differentiability.py:650:            print(f"  FAIL  finite={ok}, nonzero_frac={nz:.1%}")
tests/unit/test_zhang_mcfarlane.py:9:* finite, non-zero gradient through the relaxation timescale and
tests/unit/test_zhang_mcfarlane.py:39:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/unit/test_zhang_mcfarlane.py:111:def test_zm_outputs_finite():
tests/unit/test_zhang_mcfarlane.py:119:        assert jnp.all(jnp.isfinite(arr)), f"NaN/Inf in {arr.shape}"
tests/unit/test_zhang_mcfarlane.py:157:    # Successive differences shrink (relaxation is convergent).
tests/unit/test_zhang_mcfarlane.py:158:    diffs = np.diff(M_b_history)
tests/unit/test_zhang_mcfarlane.py:159:    # Absolute differences must monotonically decay (implicit Euler
tests/unit/test_zhang_mcfarlane.py:161:    assert np.all(np.abs(diffs[1:]) <= np.abs(diffs[:-1]) + 1e-15), (
tests/unit/test_zhang_mcfarlane.py:162:        f"Relaxation should be monotone-contracting, diffs={diffs}"
tests/unit/test_zhang_mcfarlane.py:172:    (the parcel-environment temperature difference, integrated over
tests/unit/test_zhang_mcfarlane.py:216:def test_zm_grad_through_tau_cape_finite():
tests/unit/test_zhang_mcfarlane.py:217:    """``d (sum dT_dt) / d tau_cape`` is finite — relaxation timescale
tests/unit/test_zhang_mcfarlane.py:230:    g = float(jax.grad(f)(jnp.asarray(3600.0)))
tests/unit/test_zhang_mcfarlane.py:231:    assert np.isfinite(g)
tests/unit/test_zhang_mcfarlane.py:234:def test_zm_grad_through_cape_threshold_finite_at_threshold():
tests/unit/test_zhang_mcfarlane.py:235:    """``d (sum dT_dt) / d cape_threshold`` is finite even when the
tests/unit/test_zhang_mcfarlane.py:251:    g = float(jax.grad(f)(jnp.asarray(70.0)))
tests/unit/test_zhang_mcfarlane.py:252:    assert np.isfinite(g)
tests/unit/test_zhang_mcfarlane.py:255:def test_zm_grad_through_cmt_coefficient():
tests/unit/test_zhang_mcfarlane.py:256:    """``d (sum du_dt_conv) / d cmt_c_u`` is finite — Gregory et al.
tests/unit/test_zhang_mcfarlane.py:269:    g = float(jax.grad(f)(jnp.asarray(0.55)))
tests/unit/test_zhang_mcfarlane.py:270:    assert np.isfinite(g)
tests/unit/test_zhang_mcfarlane.py:349:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_zhang_mcfarlane.py:353:def test_zm_orchestrator_one_step_finite():
tests/unit/test_zhang_mcfarlane.py:355:    produces finite tendencies and a valid carry."""
tests/unit/test_zhang_mcfarlane.py:368:    # All dycore tendencies finite.
tests/unit/test_zhang_mcfarlane.py:370:        assert jnp.all(jnp.isfinite(f.data))
tests/unit/test_zhang_mcfarlane.py:386:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/unit/test_zhang_mcfarlane.py:387:        assert jnp.all(jnp.isfinite(ps.conv_prog_profile))
tests/validation/test_isolate_instability.py:39:# Build different M matrices
tests/land/validation/test_bulk_flux_all_tiles.py:2:"""Test MOST bulk fluxes across all surface tiles and verify differentiability.
tests/land/validation/test_bulk_flux_all_tiles.py:4:Tests the Obukhov length iteration (jax.lax.fori_loop) is differentiable
tests/land/validation/test_bulk_flux_all_tiles.py:68:        grads = jax.grad(ocean_loss)(sst)
tests/land/validation/test_bulk_flux_all_tiles.py:69:        ok = bool(jnp.all(jnp.isfinite(grads)) and not jnp.allclose(grads, 0.0))
tests/land/validation/test_bulk_flux_all_tiles.py:70:        grad_norm = float(jnp.max(jnp.abs(grads)))
tests/land/validation/test_bulk_flux_all_tiles.py:72:        print(f"  {status}  {scheme:12s}  |grad|_max={grad_norm:.4e}")
tests/land/validation/test_bulk_flux_all_tiles.py:100:        grads = jax.grad(land_loss)(T_init)
tests/land/validation/test_bulk_flux_all_tiles.py:101:        ok = bool(jnp.all(jnp.isfinite(grads)) and not jnp.allclose(grads, 0.0))
tests/land/validation/test_bulk_flux_all_tiles.py:102:        grad_norm = float(jnp.max(jnp.abs(grads)))
tests/land/validation/test_bulk_flux_all_tiles.py:104:        print(f"  {status}  {scheme:12s}  |grad|_max={grad_norm:.4e}")
tests/land/validation/test_bulk_flux_all_tiles.py:134:        grads = jax.grad(ice_loss)(T_ice_init)
tests/land/validation/test_bulk_flux_all_tiles.py:135:        ok = bool(jnp.all(jnp.isfinite(grads)) and not jnp.allclose(grads, 0.0))
tests/land/validation/test_bulk_flux_all_tiles.py:136:        grad_norm = float(jnp.max(jnp.abs(grads)))
tests/land/validation/test_bulk_flux_all_tiles.py:138:        print(f"  {status}  {scheme:12s}  |grad|_max={grad_norm:.4e}")
tests/land/validation/test_bulk_flux_all_tiles.py:162:        grads = jax.grad(lake_loss)(T_epi_init)
tests/land/validation/test_bulk_flux_all_tiles.py:163:        ok = bool(jnp.all(jnp.isfinite(grads)) and not jnp.allclose(grads, 0.0))
tests/land/validation/test_bulk_flux_all_tiles.py:164:        grad_norm = float(jnp.max(jnp.abs(grads)))
tests/land/validation/test_bulk_flux_all_tiles.py:166:        print(f"  {status}  {scheme:12s}  |grad|_max={grad_norm:.4e}")
tests/land/validation/test_bulk_flux_all_tiles.py:171:    # 5. Multi-step land differentiability (the real test)
tests/land/validation/test_bulk_flux_all_tiles.py:194:    grads = jax.grad(land_multistep_loss)(T_init)
tests/land/validation/test_bulk_flux_all_tiles.py:195:    ok = bool(jnp.all(jnp.isfinite(grads)) and not jnp.allclose(grads, 0.0))
tests/land/validation/test_bulk_flux_all_tiles.py:196:    grad_norm = float(jnp.max(jnp.abs(grads)))
tests/land/validation/test_bulk_flux_all_tiles.py:198:    print(f"  {status}  5-step grad  |grad|_max={grad_norm:.4e}")
tests/unit/test_cfl.py:12:    adaptive_hyperdiff_coeff,
tests/unit/test_cfl.py:109:# Adaptive hyperdiffusion coefficient
tests/unit/test_cfl.py:112:class TestAdaptiveHyperdiff:
tests/unit/test_cfl.py:114:        nu = adaptive_hyperdiff_coeff(100e3, 600.0, order=4)
tests/unit/test_cfl.py:119:        nu_fine = adaptive_hyperdiff_coeff(50e3, 600.0, order=4, safety=1.0)
tests/unit/test_cfl.py:120:        nu_coarse = adaptive_hyperdiff_coeff(100e3, 600.0, order=4, safety=1.0)
tests/ocean/unit/test_ocean_diagnostics.py:72:        jacobian = jnp.ones((n_lat, n_lon))
tests/ocean/unit/test_ocean_diagnostics.py:76:            rho_3d, z_coord, jacobian, f)
tests/ocean/unit/test_ocean_diagnostics.py:83:        jacobian = jnp.ones((1, 1))
tests/ocean/unit/test_ocean_diagnostics.py:87:            rho_3d, z_coord, jacobian, f)
tests/ocean/unit/test_ocean_diagnostics.py:105:        jacobian = jnp.ones((1,))
tests/ocean/unit/test_ocean_diagnostics.py:109:            rho_3d, z50, jacobian, f)
tests/ocean/unit/test_ocean_diagnostics.py:121:        jacobian = jnp.ones((1,))
tests/ocean/unit/test_ocean_diagnostics.py:127:            rho_3d, z_coord, jacobian, f1)
tests/ocean/unit/test_ocean_diagnostics.py:129:            rho_3d, z_coord, jacobian, f2)
tests/ocean/unit/test_ocean_diagnostics.py:138:        jacobian = jnp.ones((1,))
tests/ocean/unit/test_ocean_diagnostics.py:142:            rho_3d, z_coord, jacobian, f)
tests/ocean/unit/test_ocean_diagnostics.py:150:        jacobian = jnp.ones((1,))
tests/ocean/unit/test_ocean_diagnostics.py:155:            rho_3d, z_coord, jacobian, f, f_min=f_min)
tests/ocean/unit/test_ocean_diagnostics.py:156:        assert jnp.isfinite(L_d).all()
tests/ocean/unit/test_ocean_diagnostics.py:157:        # Should use f_min, giving a very large but finite radius
tests/ocean/unit/test_ocean_diagnostics.py:161:    def test_grad_finite(self, z_coord):
tests/ocean/unit/test_ocean_diagnostics.py:162:        """jax.grad through deformation radius is finite."""
tests/ocean/unit/test_ocean_diagnostics.py:166:        jacobian = jnp.ones((1,))
tests/ocean/unit/test_ocean_diagnostics.py:171:                rho_in, z_coord, jacobian, f) ** 2)
tests/ocean/unit/test_ocean_diagnostics.py:173:        g = jax.grad(loss)(rho_3d)
tests/ocean/unit/test_ocean_diagnostics.py:174:        assert jnp.all(jnp.isfinite(g))
tests/validation/bench_spectral_nh.py:6:  2. DCMIP-2025 TC1 gravity waves (T31/L40, 1 hour, dt=5s).
tests/validation/bench_spectral_nh.py:7:     Expected: acoustic/gravity wave propagation from mountain forcing,
tests/validation/bench_spectral_nh.py:15:  - snapshots_w.png        — w at mid-level, lat-lon at different times
tests/validation/bench_spectral_nh.py:21:  DCMIP-2025 Test Case 1: mountain-forced gravity waves.
tests/validation/bench_spectral_nh.py:64:def proper_hyperdiff(grid):
tests/validation/bench_spectral_nh.py:103:    nu21 = proper_hyperdiff(grid21)
tests/validation/bench_spectral_nh.py:106:        hyperdiff_coeff=nu21,
tests/validation/bench_spectral_nh.py:107:        hyperdiff_order=2,
tests/validation/bench_spectral_nh.py:157:    nu_tc1 = proper_hyperdiff(grid_tc1)
tests/validation/bench_spectral_nh.py:190:        hyperdiff_coeff=nu_tc1,
tests/validation/bench_spectral_nh.py:191:        hyperdiff_order=2,
tests/validation/bench_spectral_nh.py:257:    tc1_pass = (np.all(np.isfinite(tc1_max_w)) and
tests/validation/bench_spectral_nh.py:258:                np.all(np.isfinite(tc1_max_theta)))
tests/validation/bench_spectral_nh.py:263:    print(f"    All finite:   {tc1_pass}")
tests/validation/bench_spectral_nh.py:403:    summary.append(f"  All finite:       {tc1_pass}")
tests/validation/bench_spectral_nh.py:406:    summary.append("Reference: DCMIP-2025 TC1 (mountain gravity waves)")
tests/unit/test_backend_precision.py:32:    fv_gradient_x,
tests/unit/test_backend_precision.py:33:    fv_gradient_y,
tests/unit/test_backend_precision.py:100:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_backend_precision.py:106:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_backend_precision.py:108:    def test_gradient_x(self, grid):
tests/unit/test_backend_precision.py:110:        dq = fv_gradient_x(q, grid)
tests/unit/test_backend_precision.py:112:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_backend_precision.py:114:    def test_gradient_y(self, grid):
tests/unit/test_backend_precision.py:116:        dq = fv_gradient_y(q, grid)
tests/unit/test_backend_precision.py:118:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_backend_precision.py:153:            hyperdiff_coeff=0.0, use_conservation_fixer=False,
tests/unit/test_backend_precision.py:157:        assert jnp.all(jnp.isfinite(state_new.h))
tests/unit/test_backend_precision.py:158:        assert jnp.all(jnp.isfinite(state_new.u_d))
tests/unit/test_backend_precision.py:160:    def test_differentiable_float32(self, grid):
tests/unit/test_backend_precision.py:161:        """jax.grad works through FV operators in float32."""
tests/unit/test_backend_precision.py:170:        grads = jax.grad(loss)(q)
tests/unit/test_backend_precision.py:171:        assert grads.dtype == jnp.float32
tests/unit/test_backend_precision.py:172:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_backend_precision.py:192:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_backend_precision.py:199:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_backend_precision.py:201:    def test_gradient_xy(self, grid):
tests/unit/test_backend_precision.py:204:        dx = fv_gradient_x(q, grid)
tests/unit/test_backend_precision.py:205:        dy = fv_gradient_y(q, grid)
tests/unit/test_backend_precision.py:244:            hyperdiff_coeff=0.0, use_conservation_fixer=False,
tests/unit/test_backend_precision.py:248:        assert jnp.all(jnp.isfinite(state_new.h))
tests/unit/test_backend_precision.py:263:        """FV flux divergence should produce finite results in float16."""
tests/unit/test_backend_precision.py:272:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_backend_precision.py:281:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_backend_precision.py:287:        assert jnp.all(jnp.isfinite(edges))
tests/unit/test_backend_precision.py:290:    def test_gradient_float16(self, grid):
tests/unit/test_backend_precision.py:293:        dx = fv_gradient_x(q, grid)
tests/unit/test_backend_precision.py:294:        dy = fv_gradient_y(q, grid)
tests/unit/test_backend_precision.py:295:        # Constant field → gradient should be ~0
tests/unit/test_backend_precision.py:296:        assert jnp.all(jnp.isfinite(dx))
tests/unit/test_backend_precision.py:297:        assert jnp.all(jnp.isfinite(dy))
tests/unit/test_backend_precision.py:338:        assert jnp.all(jnp.isfinite(phi_data))
tests/unit/test_backend_precision.py:340:    def test_spectral_sw_differentiable(self, grid):
tests/unit/test_backend_precision.py:341:        """jax.grad through spectral SW step."""
tests/unit/test_backend_precision.py:365:        grads = jax.grad(loss)(phi_data)
tests/unit/test_backend_precision.py:366:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_backend_precision.py:412:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_backend_precision.py:459:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_backend_precision.py:507:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_backend_precision.py:544:    def test_float32_vs_float64_gradient(self, grid):
tests/unit/test_backend_precision.py:545:        """float32 and float64 FV gradients should agree approximately."""
tests/unit/test_backend_precision.py:550:        dx32 = fv_gradient_x(q32, grid)
tests/unit/test_backend_precision.py:551:        dx64 = fv_gradient_x(q64, grid)
tests/unit/test_backend_precision.py:557:        assert rel_err < 1e-4, f"f32/f64 gradient relative error: {rel_err:.2e}"
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/common.py:89:        # Above tropopause: isothermal-like with different lapse rate
tests/validation/test_precision_amip.py:9:4. Mixed vs fp64 RMS temperature difference < 1.0 K after 5 days.
tests/validation/test_precision_amip.py:167:        assert np.all(np.isfinite(T)), f"{mode}: temperature contains NaN/Inf"
tests/validation/test_precision_amip.py:179:        assert np.all(np.isfinite(ps)), f"{mode}: p_s contains NaN/Inf"
tests/validation/test_precision_amip.py:195:        """Mixed vs fp64 RMS temperature difference should be < 1.0 K.
tests/validation/test_precision_amip.py:205:            f"mixed vs fp64 RMS T difference = {rms:.4f} K >= 1.0 K"
tests/validation/test_precision_amip.py:208:    def test_fp32_vs_fp64_finite(self, amip_runs):
tests/validation/test_precision_amip.py:209:        """fp32 vs fp64 may diverge but must remain finite.
tests/validation/test_precision_amip.py:216:        assert np.all(np.isfinite(T_f32)), "fp32 temperature has NaN/Inf"
tests/validation/test_precision_amip.py:217:        assert np.all(np.isfinite(T_ref)), "fp64 temperature has NaN/Inf"
tests/validation/test_precision_amip.py:220:        print(f"  fp32 vs fp64 RMS T difference: {rms:.4f} K")
tests/validation/test_precision_amip.py:223:        """Mixed vs fp64 RMS surface pressure difference should be < 100 Pa.
tests/validation/test_precision_amip.py:232:            f"mixed vs fp64 RMS p_s difference = {rms:.2f} Pa >= 100 Pa"
tests/validation/test_precision_amip.py:264:        assert "atm_pressure_gradient" in overrides, (
tests/validation/test_precision_amip.py:265:            "atm_pressure_gradient missing from mixed-mode overrides"
tests/validation/test_precision_amip.py:272:        assert "pressure_gradient" in overrides, (
tests/validation/test_precision_amip.py:273:            "pressure_gradient missing from mixed-mode overrides"
tests/validation/test_precision_amip.py:319:            "pressure_gradient",
tests/validation/test_restart_reproducibility.py:57:            hyperdiff_scale=1.0,
tests/validation/test_restart_reproducibility.py:75:        gravity_wave_drag="none",
tests/validation/test_continuous_stability.py:91:                assert np.isfinite(max_real), f"nlev={nlev}: non-finite eigenvalue"
tests/validation/test_continuous_stability.py:98:                assert np.isfinite(max_real), f"nlev={nlev}: non-finite eigenvalue"
tests/validation/test_continuous_stability.py:105:                assert np.isfinite(max_real), f"sigma_top={st}: non-finite eigenvalue"
tests/validation/test_continuous_stability.py:162:            assert np.isfinite(max_real), f"nlev={nlev}: non-finite eigenvalue"
tests/validation/test_continuous_stability.py:213:            assert np.isfinite(max_real), f"n={n_wave}: non-finite eigenvalue"
tests/unit/test_diff_coupled_system.py:3:Tests that gradients flow correctly through multi-component chains —
tests/unit/test_diff_coupled_system.py:21:def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
tests/unit/test_diff_coupled_system.py:22:    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
tests/unit/test_diff_coupled_system.py:23:    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
tests/unit/test_diff_coupled_system.py:62:    def test_grad_dynamics_plus_hs(self):
tests/unit/test_diff_coupled_system.py:87:        grad = jax.grad(loss)(state.T.data)
tests/unit/test_diff_coupled_system.py:88:        assert_gradient_ok(grad, "Atm dynamics + Held-Suarez")
tests/unit/test_diff_coupled_system.py:90:        assert not jnp.all(grad == grad.ravel()[0]), "Gradient has no spatial structure"
tests/unit/test_diff_coupled_system.py:99:    def test_grad_T_lowest_to_T_soil(self):
tests/unit/test_diff_coupled_system.py:130:        grad = jax.grad(loss)(T_lowest)
tests/unit/test_diff_coupled_system.py:131:        assert_gradient_ok(grad, "T_lowest → T_soil")
tests/unit/test_diff_coupled_system.py:132:        # Warmer atmosphere → warmer soil (positive gradient expected)
tests/unit/test_diff_coupled_system.py:133:        assert jnp.mean(grad) > 0, "Expected positive gradient: warmer air → warmer soil"
tests/unit/test_diff_coupled_system.py:142:    def test_grad_sst_to_shflx(self):
tests/unit/test_diff_coupled_system.py:161:        grad = jax.grad(loss)(sst)
tests/unit/test_diff_coupled_system.py:162:        assert_gradient_ok(grad, "SST → shflx")
tests/unit/test_diff_coupled_system.py:164:        # Sign depends on convention, but gradient should be non-zero
tests/unit/test_diff_coupled_system.py:220:        grad = jax.grad(loss)(sw)
tests/unit/test_diff_coupled_system.py:221:        assert jnp.all(jnp.isfinite(grad)), "Ice albedo chain gradient not finite"
tests/unit/test_diff_coupled_system.py:222:        assert jnp.any(grad != 0), "Ice albedo chain gradient all zero"
tests/unit/test_diff_coupled_system.py:234:        Tests gradient through the entire atmosphere + surface chain.
tests/unit/test_diff_coupled_system.py:304:        grad = jax.grad(loss)(state.T.data)
tests/unit/test_diff_coupled_system.py:305:        assert_gradient_ok(grad, "Full AMIP chain: dynamics + physics + land")
tests/unit/test_diff_coupled_system.py:307:        assert not jnp.all(grad == grad.ravel()[0]), "Gradient has no spatial structure"
tests/ocean/unit/test_sfno_ocean.py:113:        """Packed ocean state should be finite."""
tests/ocean/unit/test_sfno_ocean.py:115:        assert jnp.all(jnp.isfinite(packed))
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:82:        hyperdiff_coeff=1e15,
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:83:        hyperdiff_ps_coeff=1e15,
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:105:    def test_rrtmg_produces_finite_output(self, rrtmg_setup):
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:106:        """RRTMG should produce finite fluxes and heating rates."""
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:137:        assert jnp.all(jnp.isfinite(rad_out.heating_rate)), "RRTMG heating rate not finite"
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:138:        assert jnp.all(jnp.isfinite(rad_out.lw_flux_up)), "RRTMG LW flux up not finite"
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:139:        assert jnp.all(jnp.isfinite(rad_out.sw_flux_down)), "RRTMG SW flux down not finite"
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:273:        diff = float(jnp.mean(jnp.abs(out_default.sw_flux_down - out_custom.sw_flux_down)))
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:274:        assert diff > 1.0e-8
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:353:        # Verify everything is finite
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:354:        assert jnp.all(jnp.isfinite(state.T.data)), "Temperature not finite after RRTMG"
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:355:        assert jnp.all(jnp.isfinite(state.u.data)), "Winds not finite after RRTMG"
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:367:    def test_gray_and_rrtmg_produce_different_output(self, rrtmg_setup):
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:368:        """Gray and RRTMG should produce meaningfully different heating rates."""
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:405:        # Both should be finite
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:406:        assert jnp.all(jnp.isfinite(gray_out.heating_rate))
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:407:        assert jnp.all(jnp.isfinite(rrtmg_out.heating_rate))
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:409:        # They should produce different results (not identical)
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:410:        diff = float(jnp.max(jnp.abs(gray_out.heating_rate - rrtmg_out.heating_rate)))
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:411:        assert diff > 1e-8, f"Gray and RRTMG should differ, max diff = {diff}"
tests/validation/test_conservation_baseline.py:11:3. Moisture budget: tracker produces finite, non-degenerate values
tests/validation/test_conservation_baseline.py:67:    def test_energy_budget_finite(self, run_result):
tests/validation/test_conservation_baseline.py:68:        """Energy budget residual is finite and bounded."""
tests/validation/test_conservation_baseline.py:73:        assert np.all(np.isfinite(res)), "Energy residual has non-finite values"
tests/validation/test_conservation_baseline.py:77:        # The test checks for finite values and decreasing trend (equilibrating).
tests/validation/test_conservation_baseline.py:88:        assert all(np.isfinite(v) for v in tracker.column_water)
tests/validation/test_conservation_baseline.py:89:        assert all(np.isfinite(v) for v in tracker.residual)
tests/validation/test_evar_instability.py:10:If N*M + R_d*T_ref > 0: stable (imaginary eigenvalues = gravity waves)
tests/unit/test_convergence_rates.py:14:from legoesm.core.operators import gradient_x, gradient_y
tests/unit/test_convergence_rates.py:28:    def _compute_gradient_error(n):
tests/unit/test_convergence_rates.py:38:        # Analytic gradient in geographic coordinates:
tests/unit/test_convergence_rates.py:44:        # But gradient_x and gradient_y return derivatives in grid-aligned coords.
tests/unit/test_convergence_rates.py:58:        # Numerical gradient
tests/unit/test_convergence_rates.py:59:        dphi_dx_num = gradient_x(phi, grid).data
tests/unit/test_convergence_rates.py:62:        diff = dphi_dx_num - dphi_dx_exact
tests/unit/test_convergence_rates.py:64:        l2_err = float(jnp.sqrt(jnp.sum(diff ** 2 * area) / jnp.sum(area)))
tests/unit/test_convergence_rates.py:68:        err_c4 = self._compute_gradient_error(4)
tests/unit/test_convergence_rates.py:69:        err_c8 = self._compute_gradient_error(8)
tests/unit/test_convergence_rates.py:70:        err_c16 = self._compute_gradient_error(16)
tests/unit/test_convergence_rates.py:160:            hyperdiff_coeff=0.0,
tests/unit/test_convergence_rates.py:194:    def _compute_gradient_error(n_lat):
tests/unit/test_convergence_rates.py:196:        from legoesm.core.operators_latlon import gradient_x as gradient_x_ll
tests/unit/test_convergence_rates.py:209:        # Analytic eastward gradient:
tests/unit/test_convergence_rates.py:215:        # Numerical gradient
tests/unit/test_convergence_rates.py:216:        dphi_dx_num = gradient_x_ll(phi, grid).data
tests/unit/test_convergence_rates.py:219:        diff = dphi_dx_num - dphi_dx_exact
tests/unit/test_convergence_rates.py:221:        l2_err = float(jnp.sqrt(jnp.sum(diff ** 2 * area) / jnp.sum(area)))
tests/unit/test_convergence_rates.py:225:        err_8 = self._compute_gradient_error(8)
tests/unit/test_convergence_rates.py:226:        err_16 = self._compute_gradient_error(16)
tests/unit/test_convergence_rates.py:227:        err_32 = self._compute_gradient_error(32)
tests/unit/test_convergence_rates.py:243:# 6e  Hyperdiffusion convergence on cubed-sphere
tests/unit/test_convergence_rates.py:246:class TestHyperdiffusionConvergence:
tests/unit/test_convergence_rates.py:247:    """Hyperdiffusion convergence on the cubed-sphere.
tests/unit/test_convergence_rates.py:252:    hyperdiffusion result scales regularly with resolution (h^4
tests/unit/test_convergence_rates.py:254:    when resolution doubles). The L2 difference of successive
tests/unit/test_convergence_rates.py:257:    Additionally, we verify that the hyperdiffusion operator is
tests/unit/test_convergence_rates.py:259:    a dt-stepped diffusion reduces variance at all resolutions.
tests/unit/test_convergence_rates.py:263:    def _compute_hyperdiff_rms(n):
tests/unit/test_convergence_rates.py:264:        from legoesm.core.operators import hyperdiffusion as hyperdiff_cs
tests/unit/test_convergence_rates.py:277:        result = hyperdiff_cs(phi, grid, coeff=coeff).data
tests/unit/test_convergence_rates.py:279:        # After one Euler step: phi_new = phi + dt * hyperdiff
tests/unit/test_convergence_rates.py:283:        # Variance reduction: hyperdiffusion should reduce variance
tests/unit/test_convergence_rates.py:294:        """Hyperdiffusion should reduce variance at all resolutions."""
tests/unit/test_convergence_rates.py:296:            var_old, var_new = self._compute_hyperdiff_rms(n)
tests/unit/test_convergence_rates.py:298:                f"C{n}: hyperdiffusion did not reduce variance: "
tests/unit/test_convergence_rates.py:305:        Because the hyperdiffusion coefficient is scaled as dx^4/tau, finer
tests/unit/test_convergence_rates.py:312:            var_old, var_new = self._compute_hyperdiff_rms(n)
tests/unit/test_convergence_rates.py:363:# 6g  Voronoi gradient operator convergence
tests/unit/test_convergence_rates.py:370:    def _compute_gradient_error(level):
tests/unit/test_convergence_rates.py:372:        from legoesm.core.operators_voronoi import gradient_edge
tests/unit/test_convergence_rates.py:378:        grad = gradient_edge(phi, mesh)
tests/unit/test_convergence_rates.py:380:        # Analytic gradient projected onto edge normals:
tests/unit/test_convergence_rates.py:386:        grad_exact = dphi_dx * jnp.cos(mesh.angleEdge) + dphi_dy * jnp.sin(mesh.angleEdge)
tests/unit/test_convergence_rates.py:388:        err = float(jnp.sqrt(jnp.mean((grad - grad_exact)**2)))
tests/unit/test_convergence_rates.py:393:        err_2 = self._compute_gradient_error(2)
tests/unit/test_convergence_rates.py:394:        err_3 = self._compute_gradient_error(3)
tests/unit/test_convergence_rates.py:504:            hyperdiff_coeff=0.0,
tests/unit/test_land_ice_carbon.py:9:    compute_gpp, compute_phenology, step_carbon_differland, step_carbon,
tests/unit/test_land_ice_carbon.py:16:CONFIG = CarbonConfig(scheme="differland")
tests/unit/test_land_ice_carbon.py:124:        new_state, co2_flux = step_carbon_differland(
tests/unit/test_land_ice_carbon.py:179:            state, _ = step_carbon_differland(
tests/unit/test_land_ice_carbon.py:186:            assert jnp.all(jnp.isfinite(arr)), f"{name} has NaN"
tests/unit/test_land_ice_carbon.py:205:        _, flux_cold = step_carbon_differland(
tests/unit/test_land_ice_carbon.py:208:        _, flux_warm = step_carbon_differland(
tests/unit/test_land_ice_carbon.py:245:        _, flux = step_carbon_differland(
tests/unit/test_land_ice_carbon.py:256:        _, flux = step_carbon_differland(
tests/land/test_land_stability.py:141:        cc = CarbonConfig(scheme="differland") if carbon else CarbonConfig()
tests/land/test_land_stability.py:205:        cc = CarbonConfig(scheme="differland") if carbon else CarbonConfig()
tests/land/test_land_stability.py:339:    def test_matric_potential_finite(self, run15):
tests/land/test_land_stability.py:340:        """Matric potential is finite everywhere."""
tests/land/test_land_stability.py:342:        assert np.all(np.isfinite(psi))
tests/land/test_land_stability.py:559:    def test_pools_finite(self, run30_carbon):
tests/land/test_land_stability.py:560:        """All carbon pools are finite."""
tests/land/test_land_stability.py:564:            assert np.all(np.isfinite(val)), f"{pool} has NaN/Inf"
tests/land/test_land_stability.py:599:            # Check no large jumps between adjacent layers
tests/land/test_land_stability.py:600:            dT = np.abs(np.diff(T_col))
tests/land/test_land_stability.py:653:    def test_coupled_state_finite(self, run_ml_carbon):
tests/land/test_land_stability.py:654:        """All multi-layer state variables are finite with carbon."""
tests/land/test_land_stability.py:658:            assert np.all(np.isfinite(arr)), f"{name} has NaN/Inf"
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:89:        over 100 steps due to gravity wave oscillations without semi-implicit
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:90:        treatment. The key check is that all fields remain finite and
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:103:            hyperdiff_coeff=1e14,
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:112:        # All fields finite (primary stability check)
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:113:        assert jnp.all(jnp.isfinite(s.T.data)), "T contains NaN/Inf"
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:114:        assert jnp.all(jnp.isfinite(s.u.data)), "u contains NaN/Inf"
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:115:        assert jnp.all(jnp.isfinite(s.p_s.data)), "p_s contains NaN/Inf"
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:132:        that grows rapidly without strong diffusion. This test verifies
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:146:            hyperdiff_coeff=1e14,
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:156:        # All fields finite
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:157:        assert jnp.all(jnp.isfinite(s.T.data)), "T contains NaN/Inf"
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:158:        assert jnp.all(jnp.isfinite(s.u.data)), "u contains NaN/Inf"
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:159:        assert jnp.all(jnp.isfinite(s.v.data)), "v contains NaN/Inf"
tests/atmosphere/hydrostatic/integration/test_amip_stability.py:160:        assert jnp.all(jnp.isfinite(s.p_s.data)), "p_s contains NaN/Inf"
tests/ocean/unit/test_ocean_compatibility.py:7:- Full differentiability through tendencies, model.step, and integrate_scan
tests/ocean/unit/test_ocean_compatibility.py:22:    compute_ocean_jacobian,
tests/ocean/unit/test_ocean_compatibility.py:66:        hyperdiff_coeff=0.0,
tests/ocean/unit/test_ocean_compatibility.py:94:        assert jnp.isfinite(rho)
tests/ocean/unit/test_ocean_compatibility.py:97:    def test_tendencies_finite(self, state, grid, cdgrid, z_coord, config):
tests/ocean/unit/test_ocean_compatibility.py:98:        """Baroclinic tendencies should be finite (works in any precision)."""
tests/ocean/unit/test_ocean_compatibility.py:102:                assert jnp.all(jnp.isfinite(leaf))
tests/ocean/unit/test_ocean_compatibility.py:104:    def test_model_step_finite(self, grid, z_coord, config, state):
tests/ocean/unit/test_ocean_compatibility.py:105:        """Model step should produce finite results (works in any precision)."""
tests/ocean/unit/test_ocean_compatibility.py:110:                assert jnp.all(jnp.isfinite(leaf))
tests/ocean/unit/test_ocean_compatibility.py:153:        assert "vertical_diffusion(" in source
tests/ocean/unit/test_ocean_compatibility.py:155:    def test_spectral_ocean_rejects_invalid_hyperdiff_config(self, z_coord):
tests/ocean/unit/test_ocean_compatibility.py:156:        """Spectral ocean model should fail fast on invalid hyperdiff settings."""
tests/ocean/unit/test_ocean_compatibility.py:160:        with pytest.raises(ValueError, match="hyperdiff_coeff"):
tests/ocean/unit/test_ocean_compatibility.py:161:            SpectralOceanModel(None, z_coord, SpectralOceanConfig(hyperdiff_coeff=-1.0))
tests/ocean/unit/test_ocean_compatibility.py:162:        with pytest.raises(ValueError, match="hyperdiff_order"):
tests/ocean/unit/test_ocean_compatibility.py:163:            SpectralOceanModel(None, z_coord, SpectralOceanConfig(hyperdiff_order=0))
tests/ocean/unit/test_ocean_compatibility.py:243:        # C-D grid ocean uses unified C-D grid operators (not A-grid gradient_x_3d)
tests/ocean/unit/test_ocean_compatibility.py:253:            _gradient_x_raw,
tests/ocean/unit/test_ocean_compatibility.py:260:        grad_source = inspect.getsource(_gradient_x_raw)
tests/ocean/unit/test_ocean_compatibility.py:261:        assert "gradient_x(" in grad_source
tests/ocean/unit/test_ocean_compatibility.py:264:        """Conservation fixer should produce an OceanState with finite data."""
tests/ocean/unit/test_ocean_compatibility.py:271:        assert jnp.all(jnp.isfinite(fixed.eta.data))
tests/ocean/unit/test_ocean_compatibility.py:274:        """Heat fixer should produce finite output."""
tests/ocean/unit/test_ocean_compatibility.py:281:        assert jnp.all(jnp.isfinite(fixed.T.data))
tests/ocean/unit/test_ocean_compatibility.py:304:    """Full differentiability verification through the ocean module."""
tests/ocean/unit/test_ocean_compatibility.py:306:    def test_grad_through_eos(self):
tests/ocean/unit/test_ocean_compatibility.py:307:        """jax.grad should work through the Wright EOS."""
tests/ocean/unit/test_ocean_compatibility.py:312:        grad = jax.grad(loss)(jnp.array(10.0))
tests/ocean/unit/test_ocean_compatibility.py:313:        assert jnp.isfinite(grad)
tests/ocean/unit/test_ocean_compatibility.py:315:        assert float(grad) < 0
tests/ocean/unit/test_ocean_compatibility.py:317:    def test_grad_through_hydrostatic_pressure(self):
tests/ocean/unit/test_ocean_compatibility.py:318:        """jax.grad through hydrostatic pressure computation."""
tests/ocean/unit/test_ocean_compatibility.py:329:        grad = jax.grad(loss)(jnp.array(1025.0))
tests/ocean/unit/test_ocean_compatibility.py:330:        assert jnp.isfinite(grad)
tests/ocean/unit/test_ocean_compatibility.py:332:    def test_grad_through_tendencies(self, state, grid, cdgrid, z_coord, config):
tests/ocean/unit/test_ocean_compatibility.py:333:        """jax.grad should work through baroclinic tendency computation."""
tests/ocean/unit/test_ocean_compatibility.py:341:        grad = jax.grad(loss_fn)(state.eta.data)
tests/ocean/unit/test_ocean_compatibility.py:342:        assert jnp.all(jnp.isfinite(grad))
tests/ocean/unit/test_ocean_compatibility.py:344:    def test_grad_through_temperature(self, state, grid, cdgrid, z_coord, config):
tests/ocean/unit/test_ocean_compatibility.py:345:        """jax.grad should work through temperature tendencies."""
tests/ocean/unit/test_ocean_compatibility.py:353:        grad = jax.grad(loss_fn)(state.T.data)
tests/ocean/unit/test_ocean_compatibility.py:354:        assert jnp.all(jnp.isfinite(grad))
tests/ocean/unit/test_ocean_compatibility.py:356:    def test_grad_through_model_step(self, grid, z_coord, state):
tests/ocean/unit/test_ocean_compatibility.py:357:        """jax.grad through a full model step (differentiable barotropic)."""
tests/ocean/unit/test_ocean_compatibility.py:361:            hyperdiff_coeff=0.0,
tests/ocean/unit/test_ocean_compatibility.py:363:            differentiable_barotropic=True,
tests/ocean/unit/test_ocean_compatibility.py:374:        grad = jax.grad(loss_fn)(state.eta.data)
tests/ocean/unit/test_ocean_compatibility.py:375:        assert jnp.all(jnp.isfinite(grad))
tests/ocean/unit/test_ocean_compatibility.py:377:    def test_grad_through_scan(self, grid, z_coord, state):
tests/ocean/unit/test_ocean_compatibility.py:378:        """jax.grad through lax.scan integration (multi-step)."""
tests/ocean/unit/test_ocean_compatibility.py:389:            hyperdiff_coeff=0.0,
tests/ocean/unit/test_ocean_compatibility.py:391:            differentiable_barotropic=True,
tests/ocean/unit/test_ocean_compatibility.py:402:        grad = jax.grad(loss_fn)(state.eta.data)
tests/ocean/unit/test_ocean_compatibility.py:403:        assert jnp.all(jnp.isfinite(grad))
tests/ocean/unit/test_ocean_compatibility.py:414:        assert jnp.all(jnp.isfinite(rho_batch))
tests/ocean/unit/test_ocean_compatibility.py:423:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/ocean/unit/test_ocean_compatibility.py:444:            hyperdiff_coeff=0.0,
tests/ocean/unit/test_ocean_compatibility.py:445:            differentiable_barotropic=False,
tests/ocean/unit/test_ocean_compatibility.py:447:        config_scan = config_fori._replace(differentiable_barotropic=True)
tests/ocean/unit/test_ocean_compatibility.py:519:    def test_land_mask_preserved_through_grad(self, state, grid, cdgrid, z_coord, config):
tests/ocean/unit/test_ocean_compatibility.py:528:        grad = jax.grad(loss_fn)(state.eta.data)
tests/ocean/unit/test_ocean_compatibility.py:532:            assert float(jnp.max(jnp.abs(grad[land]))) == 0.0
tests/unit/test_conservation.py:51:        # in infinite precision, but float32 accumulation rounds off
tests/unit/test_conservation.py:54:    def test_mass_fixer_preserves_gradients(self, grid, state):
tests/unit/test_conservation.py:55:        """Mass fixer should be differentiable."""
tests/unit/test_conservation.py:61:        grads = jax.grad(loss)(state.h.data)
tests/unit/test_conservation.py:62:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_conservation.py:121:        assert jnp.all(jnp.isfinite(state_fixed.u.data)), "u contains NaN/Inf"
tests/unit/test_conservation.py:122:        assert jnp.all(jnp.isfinite(state_fixed.v.data)), "v contains NaN/Inf"
tests/unit/test_conservation.py:124:    def test_energy_fixer_preserves_gradients(self, grid, state):
tests/unit/test_conservation.py:125:        """Energy fixer should be differentiable."""
tests/unit/test_conservation.py:133:        grads = jax.grad(loss)(state.u.data)
tests/unit/test_conservation.py:134:        assert jnp.all(jnp.isfinite(grads)), "Gradients through energy fixer are not finite"
tests/unit/test_corrections.py:27:    """eps_gwd must be positive-definite (KE sink)."""
tests/unit/test_corrections.py:29:    def _make_gwd_inputs(self, ncol=4, nlev=20):
tests/unit/test_corrections.py:31:        # so that eps_gwd is positive-definite for all schemes.
tests/unit/test_corrections.py:54:        from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
tests/unit/test_corrections.py:55:        from legoesm.atmosphere.physics.gravity_wave_drag.config import RayleighConfig
tests/unit/test_corrections.py:56:        args = self._make_gwd_inputs()
tests/unit/test_corrections.py:57:        out = rayleigh_gwd(*args, dt=300.0, config=RayleighConfig())
tests/unit/test_corrections.py:58:        assert float(jnp.min(out.eps_gwd)) >= -1e-10, "eps_gwd should be >= 0"
tests/unit/test_corrections.py:61:        from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
tests/unit/test_corrections.py:62:        from legoesm.atmosphere.physics.gravity_wave_drag.config import LindzenConfig
tests/unit/test_corrections.py:63:        args = self._make_gwd_inputs()
tests/unit/test_corrections.py:64:        out = lindzen_gwd(*args, dt=300.0, config=LindzenConfig())
tests/unit/test_corrections.py:65:        assert float(jnp.min(out.eps_gwd)) >= -1e-10, "eps_gwd should be >= 0"
tests/unit/test_corrections.py:68:        from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
tests/unit/test_corrections.py:69:        from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
tests/unit/test_corrections.py:70:        args = self._make_gwd_inputs()
tests/unit/test_corrections.py:71:        out = mcfarlane_gwd(*args, dt=300.0, config=McFarlaneConfig())
tests/unit/test_corrections.py:72:        assert float(jnp.min(out.eps_gwd)) >= -1e-10, "eps_gwd should be >= 0"
tests/unit/test_corrections.py:75:        from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
tests/unit/test_corrections.py:76:        from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
tests/unit/test_corrections.py:77:        args = self._make_gwd_inputs()
tests/unit/test_corrections.py:78:        out = hines_gwd(*args, dt=300.0, config=HinesConfig())
tests/unit/test_corrections.py:79:        assert float(jnp.min(out.eps_gwd)) >= -1e-10, "eps_gwd should be >= 0"
tests/unit/test_corrections.py:82:        """eps_gwd should equal integrated dT_dt * c_p * rho * dz."""
tests/unit/test_corrections.py:83:        from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
tests/unit/test_corrections.py:84:        from legoesm.atmosphere.physics.gravity_wave_drag.config import RayleighConfig
tests/unit/test_corrections.py:86:        u, v, T, p_full, p_half, z_full, z_half, rho, lat = self._make_gwd_inputs()
tests/unit/test_corrections.py:87:        out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat,
tests/unit/test_corrections.py:91:        # eps_gwd should match heating power
tests/unit/test_corrections.py:92:        assert jnp.allclose(out.eps_gwd, heating_power, rtol=1e-5), \
tests/unit/test_corrections.py:93:            f"Energy mismatch: eps_gwd={out.eps_gwd}, heating={heating_power}"
tests/unit/test_corrections.py:185:                f"SW up at level {k} differs from surface"
tests/unit/test_corrections.py:222:        # but mid-level profiles should differ (exponent changes how tau
tests/unit/test_corrections.py:226:        # SW heating rates should also differ
tests/unit/test_corrections.py:244:        jacobian = jnp.ones((6, n, n))
tests/unit/test_corrections.py:245:        return grid, z_coord, jacobian, shape
tests/unit/test_corrections.py:248:        """Flat isopycnals => no GM transport, Redi = horizontal diffusion."""
tests/unit/test_corrections.py:251:        grid, z_coord, jacobian, shape = self._make_ocean_setup()
tests/unit/test_corrections.py:267:        out = gm_redi_lateral_mixing(u, v, T, S, rho, z_coord, jacobian, grid, cfg)
tests/unit/test_corrections.py:272:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_corrections.py:278:        grid, z_coord, jacobian, shape = self._make_ocean_setup()
tests/unit/test_corrections.py:287:        out = gm_redi_lateral_mixing(u, v, T, S, rho, z_coord, jacobian, grid, cfg)
tests/unit/test_corrections.py:301:        grid, z_coord, jacobian, shape = self._make_ocean_setup(n, nlev)
tests/unit/test_corrections.py:303:        # Density increases with depth and has a meridional gradient
tests/unit/test_corrections.py:310:        # Tracer with horizontal gradient (so slopes matter)
tests/unit/test_corrections.py:317:        return grid, z_coord, jacobian, shape, rho, T, S, u, v
tests/unit/test_corrections.py:319:    def test_unequal_kappas_finite(self):
tests/unit/test_corrections.py:320:        """kappa_GM != kappa_Redi should produce finite tendencies."""
tests/unit/test_corrections.py:323:        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
tests/unit/test_corrections.py:328:        out = gm_redi_lateral_mixing(u, v, T, S, rho, z_coord, jacobian, grid, cfg)
tests/unit/test_corrections.py:329:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_corrections.py:330:        assert jnp.all(jnp.isfinite(out.dS_dt))
tests/unit/test_corrections.py:332:    def test_unequal_kappas_differ_from_equal(self):
tests/unit/test_corrections.py:333:        """kappa_GM != kappa_Redi should give different tendencies than equal."""
tests/unit/test_corrections.py:336:        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
tests/unit/test_corrections.py:344:            u, v, T, S, rho, z_coord, jacobian, grid, cfg_equal
tests/unit/test_corrections.py:347:            u, v, T, S, rho, z_coord, jacobian, grid, cfg_unequal
tests/unit/test_corrections.py:351:        # so unequal kappas must give different tendencies
tests/unit/test_corrections.py:360:        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
tests/unit/test_corrections.py:365:        S_x, S_y, _ = _compute_tapered_slopes(rho, z_coord, jacobian, grid, cfg)
tests/unit/test_corrections.py:370:        # when slopes are zero. With non-zero slopes they must differ,
tests/unit/test_corrections.py:373:            T, S_x, S_y, z_coord, jacobian, grid, 1e3, 1e3,
tests/unit/test_corrections.py:377:            T, S_x, S_y, z_coord, jacobian, grid, 1e3 + 1.0, 1e3,
tests/unit/test_corrections.py:379:        # The 1 m²/s change in kG should produce a small but non-zero difference
tests/unit/test_corrections.py:380:        diff = jnp.max(jnp.abs(dT_equal - dT_perturbed))
tests/unit/test_corrections.py:381:        assert diff > 0, "Small kG perturbation should produce a difference"
tests/unit/test_corrections.py:387:        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
tests/unit/test_corrections.py:392:        out = gm_redi_lateral_mixing(u, v, T, S, rho, z_coord, jacobian, grid, cfg)
tests/unit/test_corrections.py:393:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_corrections.py:394:        assert jnp.all(jnp.isfinite(out.dS_dt))
tests/unit/test_corrections.py:397:        """Pure Redi (kappa_GM=0): full isopycnal diffusion tensor."""
tests/unit/test_corrections.py:400:        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
tests/unit/test_corrections.py:405:        out = gm_redi_lateral_mixing(u, v, T, S, rho, z_coord, jacobian, grid, cfg)
tests/unit/test_corrections.py:406:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_corrections.py:410:        (relative to their difference)."""
tests/unit/test_corrections.py:415:        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
tests/unit/test_corrections.py:420:        S_x, S_y, _ = _compute_tapered_slopes(rho, z_coord, jacobian, grid, cfg)
tests/unit/test_corrections.py:424:            T, S_x, S_y, z_coord, jacobian, grid,
tests/unit/test_corrections.py:429:            T, S_x, S_y, z_coord, jacobian, grid,
tests/unit/test_corrections.py:432:        # The two should differ (the off-diagonal sign flips)
tests/unit/test_corrections.py:442:        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
tests/unit/test_corrections.py:447:        S_x, S_y, _ = _compute_tapered_slopes(rho, z_coord, jacobian, grid, cfg)
tests/unit/test_corrections.py:451:            T, S_x, S_y, z_coord, jacobian, grid,
tests/unit/test_corrections.py:456:            T, S_x, S_y, z_coord, jacobian, grid,
tests/unit/test_corrections.py:459:        # Diagonal laplacian differs (kR=800 vs kR=1200), so totals differ.
tests/unit/test_corrections.py:460:        # But since kR+kG is the same, difference must come only from
tests/unit/test_corrections.py:462:        # Just verify both are finite and different.
tests/unit/test_corrections.py:463:        assert jnp.all(jnp.isfinite(dT_a))
tests/unit/test_corrections.py:464:        assert jnp.all(jnp.isfinite(dT_b))
tests/unit/test_corrections.py:479:        jacobian = jnp.ones((6, n, n))
tests/unit/test_corrections.py:489:        return u, v, T, S, rho, eta, z_coord, jacobian
tests/unit/test_corrections.py:500:        # dT_dt should only come from diffusion, not nonlocal
tests/unit/test_corrections.py:502:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_corrections.py:515:        # Tendencies should differ when nonlocal is active
tests/unit/test_corrections.py:516:        diff = jnp.max(jnp.abs(out_unstable.dT_dt - out_stable.dT_dt))
tests/unit/test_corrections.py:517:        assert float(diff) > 0, "Nonlocal should cause different tendencies"
tests/unit/test_corrections.py:576:        # Both must be finite
tests/unit/test_corrections.py:577:        assert jnp.all(jnp.isfinite(out_neg.dT_dt))
tests/unit/test_corrections.py:578:        assert jnp.all(jnp.isfinite(out_pos.dT_dt))
tests/unit/test_corrections.py:580:        # The tendencies should differ (nonlocal active for unstable, not stable)
tests/unit/test_corrections.py:581:        diff = float(jnp.max(jnp.abs(out_pos.dT_dt - out_neg.dT_dt)))
tests/unit/test_corrections.py:582:        assert diff > 0, "B_f sign flip: stable and unstable produced identical output"
tests/unit/test_corrections.py:601:        # Tendencies should differ when imposed flux is used
tests/unit/test_corrections.py:602:        diff = float(jnp.max(jnp.abs(out_imposed.dT_dt - out_diag.dT_dt)))
tests/unit/test_corrections.py:603:        assert diff > 1e-15, "Q_sfc_T had no effect on KPP output"
tests/unit/test_corrections.py:625:        # BL depth estimate should differ
tests/unit/test_corrections.py:627:        diff = float(jnp.max(jnp.abs(out_shallow.K_v - out_default.K_v)))
tests/unit/test_corrections.py:628:        assert diff > 1e-15, "h_bl_prev had no effect on K_v profile"
tests/unit/test_corrections.py:671:        """Results should differ for z_ref=10 vs z_ref=20."""
tests/unit/test_corrections.py:679:            "u_star should differ with different z_ref"
tests/unit/test_corrections.py:682:        """Stable and unstable cases should give different heat fluxes."""
tests/unit/test_corrections.py:765:        # Should be between adjacent values
tests/ocean/unit/test_latlon_cgrid_ocean.py:21:    gradient_x_cgrid,
tests/ocean/unit/test_latlon_cgrid_ocean.py:22:    gradient_y_cgrid,
tests/ocean/unit/test_latlon_cgrid_ocean.py:59:    def test_gradient_x_shape(self, grid):
tests/ocean/unit/test_latlon_cgrid_ocean.py:61:        gx = gradient_x_cgrid(f, grid)
tests/ocean/unit/test_latlon_cgrid_ocean.py:64:    def test_gradient_y_shape(self, grid):
tests/ocean/unit/test_latlon_cgrid_ocean.py:66:        gy = gradient_y_cgrid(f, grid)
tests/ocean/unit/test_latlon_cgrid_ocean.py:69:    def test_gradient_constant_field_zero(self, grid):
tests/ocean/unit/test_latlon_cgrid_ocean.py:72:        gx = gradient_x_cgrid(f, grid)
tests/ocean/unit/test_latlon_cgrid_ocean.py:73:        gy = gradient_y_cgrid(f, grid)
tests/ocean/unit/test_latlon_cgrid_ocean.py:94:        assert jnp.all(jnp.isfinite(div))
tests/ocean/unit/test_latlon_cgrid_ocean.py:127:    def test_tendencies_finite(self, state, grid, z_coord, config):
tests/ocean/unit/test_latlon_cgrid_ocean.py:131:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:132:        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:133:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:134:        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:135:        assert jnp.all(jnp.isfinite(tend.deta_dt.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:174:        Two states differ only by a spatially constant δU added to u.
tests/ocean/unit/test_latlon_cgrid_ocean.py:175:        The total-velocity KE gradient produces an extra −δU·∂u/∂x
tests/ocean/unit/test_latlon_cgrid_ocean.py:207:        diff_du = float(
tests/ocean/unit/test_latlon_cgrid_ocean.py:210:        # Total-velocity KE gradient produces O(δU · ∂u/∂x) difference.
tests/ocean/unit/test_latlon_cgrid_ocean.py:211:        assert diff_du > 1e-12, (
tests/ocean/unit/test_latlon_cgrid_ocean.py:213:            f"via the KE gradient; got diff={diff_du:.3e} (invariant → bug)."
tests/ocean/unit/test_latlon_cgrid_ocean.py:227:        assert jnp.all(jnp.isfinite(s1.u.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:228:        assert jnp.all(jnp.isfinite(s1.v.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:229:        assert jnp.all(jnp.isfinite(s1.T.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:230:        assert jnp.all(jnp.isfinite(s1.eta.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:261:        assert jnp.all(jnp.isfinite(s.u.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:262:        assert jnp.all(jnp.isfinite(s.T.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:263:        assert jnp.all(jnp.isfinite(s.eta.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:304:        assert jnp.all(jnp.isfinite(s.eta.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:305:        assert jnp.all(jnp.isfinite(s.u.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:306:        assert jnp.all(jnp.isfinite(s.v.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:312:    def test_differentiable(self, grid, z_coord, config, state):
tests/ocean/unit/test_latlon_cgrid_ocean.py:313:        """C-grid model should be differentiable through tendencies."""
tests/ocean/unit/test_latlon_cgrid_ocean.py:321:        grad_fn = jax.grad(loss_fn)
tests/ocean/unit/test_latlon_cgrid_ocean.py:322:        g = grad_fn(state.eta.data)
tests/ocean/unit/test_latlon_cgrid_ocean.py:323:        assert jnp.all(jnp.isfinite(g))
tests/unit/test_duogrid_metrics.py:55:        max_diff = float(jnp.max(jnp.abs(sin_E - sin_W)))
tests/unit/test_duogrid_metrics.py:56:        self.assertLess(max_diff, 1e-6,
tests/unit/test_duogrid_metrics.py:57:                        f"Interior E/W edge mismatch: {max_diff}")
tests/unit/test_duogrid_metrics.py:61:        max_diff_c = float(jnp.max(jnp.abs(cos_E - cos_W)))
tests/unit/test_duogrid_metrics.py:62:        self.assertLess(max_diff_c, 1e-6,
tests/unit/test_duogrid_metrics.py:63:                        f"Interior E/W cos_sg mismatch: {max_diff_c}")
tests/unit/test_duogrid_metrics.py:86:        the cosine values differ.
tests/unit/test_duogrid_metrics.py:91:        max_diff = float(jnp.max(jnp.abs(sin_N_face0 - sin_S_face4)))
tests/unit/test_duogrid_metrics.py:92:        self.assertLess(max_diff, 1e-6,
tests/unit/test_duogrid_metrics.py:94:                        f"max diff = {max_diff}")
tests/land/unit/test_multilayer_land.py:7:- 8D: Soil thermal diffusion (energy conservation, convergence)
tests/land/unit/test_multilayer_land.py:119:        diffs = jnp.diff(K)
tests/land/unit/test_multilayer_land.py:120:        self.assertTrue(jnp.all(diffs >= 0))
tests/land/unit/test_multilayer_land.py:210:        diffs = jnp.diff(theta)
tests/land/unit/test_multilayer_land.py:211:        self.assertTrue(jnp.all(diffs >= -1e-10))
tests/land/unit/test_multilayer_land.py:260:        self.assertTrue(jnp.all(jnp.isfinite(C)))
tests/land/unit/test_multilayer_land.py:280:        # PDI should differ from standard VG near saturation due to constraint
tests/land/unit/test_multilayer_land.py:281:        # (PDI interpolates to K_sat, VG drops from K_sat differently)
tests/land/unit/test_multilayer_land.py:282:        self.assertTrue(jnp.all(jnp.isfinite(K_pdi)))
tests/land/unit/test_multilayer_land.py:297:        self.assertTrue(jnp.all(jnp.isfinite(out.theta_new)))
tests/land/unit/test_multilayer_land.py:325:        diffs = jnp.diff(theta)
tests/land/unit/test_multilayer_land.py:326:        self.assertTrue(jnp.all(diffs >= -1e-10))
tests/land/unit/test_multilayer_land.py:374:        self.assertTrue(jnp.all(jnp.isfinite(C)))
tests/land/unit/test_multilayer_land.py:405:        self.assertTrue(jnp.all(jnp.isfinite(out.theta_new)))
tests/land/unit/test_multilayer_land.py:435:        # theta should change very little (gravity still causes small redistribution)
tests/land/unit/test_multilayer_land.py:504:    """Test soil thermal diffusion."""
tests/land/unit/test_multilayer_land.py:676:        # Temperature should be finite and reasonable
tests/land/unit/test_multilayer_land.py:677:        self.assertTrue(jnp.all(jnp.isfinite(state.T_soil)))
tests/land/unit/test_multilayer_land.py:696:    def test_response_fields_finite(self):
tests/land/unit/test_multilayer_land.py:697:        """All TileResponse fields should be finite."""
tests/land/unit/test_multilayer_land.py:714:                jnp.all(jnp.isfinite(arr)),
tests/land/unit/test_multilayer_land.py:715:                f"TileResponse.{name} has non-finite values",
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:124:> **What's tested:** Pressure-gradient accuracy, mountain wave generation/propagation. First actual dynamical integration.
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:137:- Any non-zero `u, v, w` is a *direct measure of pressure-gradient error*
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:162:K_diffusion = K_ref / X**(2*k-1) # scaled diffusion coefficient (order k)
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:166:- Vertically propagating gravity waves above the mountain
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:188:> **What's tested:** Acoustic-gravity wave propagation without orography or Coriolis. Clean test of the NH wave solver.
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:194:| **Fortran init** | `dcmip_initial_conditions_test_1_2_3_v5.f90` → `subroutine test3_gravity_wave(lon, lat, p, z, zcoords, u, v, w, t, phis, ps, rho, q)` |
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:276:        # If using explicit diffusion K * del^(2k):
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:338:# 3. Boundary-layer mixing (vertical diffusion)
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:498:- **Diffusion is critical.** DCMIP2016 models used varying diffusion strategies. Results are very sensitive to explicit/implicit numerical diffusion at these scales.
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:538:g       = 9.80616      # gravity (m/s²)
tests/unit/test_land_params.py:6:3. PFTParamProvider (convexity, single-PFT, differentiability)
tests/unit/test_land_params.py:7:4. NeuralParamProvider (bounded outputs, differentiability, JIT)
tests/unit/test_land_params.py:215:    def test_differentiable(self):
tests/unit/test_land_params.py:224:        loss, grads = eqx.filter_value_and_grad(loss_fn)(provider)
tests/unit/test_land_params.py:225:        assert jnp.isfinite(loss)
tests/unit/test_land_params.py:226:        # raw_table should have non-zero gradient
tests/unit/test_land_params.py:227:        assert grads.raw_table is not None
tests/unit/test_land_params.py:228:        assert jnp.any(grads.raw_table != 0.0)
tests/unit/test_land_params.py:230:    def test_stop_gradient_on_fractions(self):
tests/unit/test_land_params.py:239:        grads = eqx.filter_grad(loss_fn)(provider)
tests/unit/test_land_params.py:240:        # pft_fractions should be zero (stop_gradient)
tests/unit/test_land_params.py:241:        if grads.pft_fractions is not None:
tests/unit/test_land_params.py:242:            assert jnp.allclose(grads.pft_fractions, 0.0)
tests/unit/test_land_params.py:275:    def test_differentiable(self):
tests/unit/test_land_params.py:284:        loss, grads = eqx.filter_value_and_grad(loss_fn)(provider)
tests/unit/test_land_params.py:285:        assert jnp.isfinite(loss)
tests/unit/test_land_params.py:286:        # Check at least one layer has non-zero gradient
tests/unit/test_land_params.py:287:        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
tests/unit/test_land_params.py:288:        assert any(jnp.any(g != 0.0) for g in grad_leaves)
tests/unit/test_land_params.py:302:        assert jnp.all(jnp.isfinite(result))
tests/unit/test_land_params.py:326:        assert jnp.all(jnp.isfinite(features))
tests/unit/test_kain_fritsch.py:5:* tendency shape / dtype / finiteness;
tests/unit/test_kain_fritsch.py:7:* the trigger function's smoothness — gradient through ``w_grid`` is
tests/unit/test_kain_fritsch.py:8:  finite and non-zero across the threshold (this is the central
tests/unit/test_kain_fritsch.py:11:* differentiability through ``parcel_perturb_T`` and ``trigger_sharpness``;
tests/unit/test_kain_fritsch.py:37:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/unit/test_kain_fritsch.py:73:# Shape / dtype / finiteness
tests/unit/test_kain_fritsch.py:88:def test_kf_outputs_finite():
tests/unit/test_kain_fritsch.py:95:        assert jnp.all(jnp.isfinite(arr))
tests/unit/test_kain_fritsch.py:128:def test_kf_trigger_smoothness_grad_finite_at_crossing():
tests/unit/test_kain_fritsch.py:130:    ``w_grid``: a finite, non-zero gradient w.r.t. the column-mean
tests/unit/test_kain_fritsch.py:142:    grad_at_threshold = float(jax.grad(f)(jnp.asarray(2.0)))
tests/unit/test_kain_fritsch.py:143:    grad_below = float(jax.grad(f)(jnp.asarray(-2.0)))
tests/unit/test_kain_fritsch.py:144:    grad_above = float(jax.grad(f)(jnp.asarray(8.0)))
tests/unit/test_kain_fritsch.py:145:    for label, g in [("below", grad_below), ("at", grad_at_threshold), ("above", grad_above)]:
tests/unit/test_kain_fritsch.py:146:        assert np.isfinite(g), f"grad {label} not finite"
tests/unit/test_kain_fritsch.py:147:    # At threshold the gradient should be larger (sigmoid steepest in
tests/unit/test_kain_fritsch.py:150:    assert abs(grad_at_threshold) > abs(grad_below) - 1e-12
tests/unit/test_kain_fritsch.py:151:    assert abs(grad_at_threshold) > abs(grad_above) - 1e-12
tests/unit/test_kain_fritsch.py:154:def test_kf_trigger_grad_through_parcel_perturb_T():
tests/unit/test_kain_fritsch.py:168:    g = float(jax.grad(f)(jnp.asarray(0.5)))
tests/unit/test_kain_fritsch.py:169:    assert np.isfinite(g)
tests/unit/test_kain_fritsch.py:172:def test_kf_trigger_grad_through_w_thresh_offset():
tests/unit/test_kain_fritsch.py:174:    is finite — supports training-time tuning of the trigger."""
tests/unit/test_kain_fritsch.py:186:    g = float(jax.grad(f)(jnp.asarray(2.0)))
tests/unit/test_kain_fritsch.py:187:    assert np.isfinite(g)
tests/unit/test_kain_fritsch.py:261:def test_kf_orchestrator_one_step_finite():
tests/unit/test_kain_fritsch.py:270:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_kain_fritsch.py:280:        assert jnp.all(jnp.isfinite(f.data))
tests/validation/test_amip_validation.py:5:- All diagnostic checks produce finite results
tests/validation/test_amip_validation.py:49:        assert all(np.isfinite(r.value) for r in report.results)
tests/unit/test_duogrid.py:104:                    diffs = np.diff(strip)
tests/unit/test_duogrid.py:105:                    assert np.all(diffs > 0) or np.all(diffs < 0), (
tests/unit/test_duogrid.py:107:                        f"depth={d}, diffs=[{diffs.min():.4e}, {diffs.max():.4e}]"
tests/unit/test_duogrid.py:170:        assert jnp.all(jnp.isfinite(south_halo))
tests/unit/test_duogrid.py:203:# T4: JIT + grad smoke tests
tests/unit/test_duogrid.py:207:    """Tests T4.1 through T4.4: JIT and gradient compatibility."""
tests/unit/test_duogrid.py:240:    def test_t4_3_grad_through_cube_rmp(self, duogrid_data):
tests/unit/test_duogrid.py:241:        """Gradient through cube_rmp should produce finite values."""
tests/unit/test_duogrid.py:252:        grad = jax.grad(loss)(field)
tests/unit/test_duogrid.py:253:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_duogrid.py:255:    def test_t4_3_grad_through_full_pipeline(self, duogrid_data):
tests/unit/test_duogrid.py:256:        """Gradient through cube_rmp + fill_corner should be finite."""
tests/unit/test_duogrid.py:268:        grad = jax.grad(loss)(field)
tests/unit/test_duogrid.py:269:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_duogrid.py:294:    def test_t2_1_all_edges_produce_finite_values(self, duogrid_data):
tests/unit/test_duogrid.py:295:        """After cube_rmp, all halo cells should be finite."""
tests/unit/test_duogrid.py:302:        assert jnp.all(jnp.isfinite(result))
tests/unit/test_duogrid.py:432:        assert jnp.all(jnp.isfinite(ua))
tests/unit/test_duogrid.py:433:        assert jnp.all(jnp.isfinite(uc))
tests/unit/test_duogrid.py:434:        assert jnp.all(jnp.isfinite(vc))
tests/unit/test_duogrid.py:461:        assert jnp.all(jnp.isfinite(ua))
tests/unit/test_duogrid.py:464:        """fv3_csw_tendencies should produce finite tendencies with duogrid."""
tests/unit/test_duogrid.py:476:        assert jnp.all(jnp.isfinite(dh))
tests/unit/test_duogrid.py:477:        assert jnp.all(jnp.isfinite(du))
tests/unit/test_duogrid.py:478:        assert jnp.all(jnp.isfinite(dv))
tests/unit/test_duogrid.py:520:        assert jnp.all(jnp.isfinite(ud))
tests/unit/test_duogrid.py:521:        assert jnp.all(jnp.isfinite(vd))
tests/unit/test_duogrid.py:554:        assert jnp.all(jnp.isfinite(ud))
tests/unit/test_duogrid.py:555:        assert jnp.all(jnp.isfinite(vd))
tests/ocean/unit/test_weno_momentum.py:8:4. Full tendency: finite output with WENO momentum advection
tests/ocean/unit/test_weno_momentum.py:9:5. AD: reverse-mode gradient finiteness through WENO momentum path
tests/ocean/unit/test_weno_momentum.py:78:        # Skip faces near boundaries where ghosts degrade accuracy
tests/ocean/unit/test_weno_momentum.py:85:    def test_finite_values(self):
tests/ocean/unit/test_weno_momentum.py:99:        assert jnp.all(jnp.isfinite(result))
tests/ocean/unit/test_weno_momentum.py:174:        assert jnp.all(jnp.isfinite(result))
tests/ocean/unit/test_weno_momentum.py:176:    def test_finite_values(self):
tests/ocean/unit/test_weno_momentum.py:190:        assert jnp.all(jnp.isfinite(result))
tests/ocean/unit/test_weno_momentum.py:278:        # a nonzero tendency only from w gradients but the overall effect
tests/ocean/unit/test_weno_momentum.py:300:    def test_finite_values(self):
tests/ocean/unit/test_weno_momentum.py:312:        assert jnp.all(jnp.isfinite(result))
tests/ocean/unit/test_weno_momentum.py:330:# AD (reverse-mode gradient) correctness
tests/ocean/unit/test_weno_momentum.py:336:    def test_ad_zeta_at_u_finite(self):
tests/ocean/unit/test_weno_momentum.py:337:        """Gradient through _weno_zeta_at_u is finite and nonzero."""
tests/ocean/unit/test_weno_momentum.py:355:        g = jax.grad(loss)(zeta)
tests/ocean/unit/test_weno_momentum.py:356:        assert jnp.all(jnp.isfinite(g)), "Gradient has non-finite values"
tests/ocean/unit/test_weno_momentum.py:359:    def test_ad_zeta_at_v_finite(self):
tests/ocean/unit/test_weno_momentum.py:360:        """Gradient through _weno_zeta_at_v is finite and nonzero."""
tests/ocean/unit/test_weno_momentum.py:378:        g = jax.grad(loss)(zeta)
tests/ocean/unit/test_weno_momentum.py:379:        assert jnp.all(jnp.isfinite(g)), "Gradient has non-finite values"
tests/ocean/unit/test_weno_momentum.py:383:        """Gradient through vertical momentum WENO is finite."""
tests/ocean/unit/test_weno_momentum.py:402:        g = jax.grad(loss)(u)
tests/ocean/unit/test_weno_momentum.py:403:        assert jnp.all(jnp.isfinite(g)), "Gradient has non-finite values"
tests/ocean/unit/test_weno_momentum.py:430:        g = jax.grad(loss)(z0)
tests/ocean/unit/test_weno_momentum.py:515:    def test_finite_values(self):
tests/ocean/unit/test_weno_momentum.py:527:        assert jnp.all(jnp.isfinite(result))
tests/ocean/unit/test_weno_momentum.py:577:        # Skip faces near boundaries where ghosts degrade accuracy
tests/ocean/unit/test_weno_momentum.py:584:    def test_finite_values(self):
tests/ocean/unit/test_weno_momentum.py:596:        assert jnp.all(jnp.isfinite(result))
tests/ocean/unit/test_weno_momentum.py:607:    def test_ad_div_at_u_finite(self):
tests/ocean/unit/test_weno_momentum.py:608:        """Gradient through _weno_cell_to_uface is finite and nonzero."""
tests/ocean/unit/test_weno_momentum.py:622:        g = jax.grad(loss)(D)
tests/ocean/unit/test_weno_momentum.py:623:        assert jnp.all(jnp.isfinite(g)), "Gradient has non-finite values"
tests/ocean/unit/test_weno_momentum.py:626:    def test_ad_div_at_v_finite(self):
tests/ocean/unit/test_weno_momentum.py:627:        """Gradient through _weno_cell_to_vface is finite and nonzero."""
tests/ocean/unit/test_weno_momentum.py:641:        g = jax.grad(loss)(D)
tests/ocean/unit/test_weno_momentum.py:642:        assert jnp.all(jnp.isfinite(g)), "Gradient has non-finite values"
tests/ocean/unit/test_weno_momentum.py:666:        g = jax.grad(loss)(d0)
tests/ocean/unit/test_weno_momentum.py:690:    """Integration test: full tendency with WENO Z+D+K+C gives finite output."""
tests/ocean/unit/test_weno_momentum.py:719:    def test_weno5_tendency_finite(self):
tests/ocean/unit/test_weno_momentum.py:720:        """Full tendency with weno5 (Z+D+K+C) produces finite output."""
tests/ocean/unit/test_weno_momentum.py:732:        assert jnp.all(jnp.isfinite(du)), "du tendency has non-finite"
tests/ocean/unit/test_weno_momentum.py:733:        assert jnp.all(jnp.isfinite(dv)), "dv tendency has non-finite"
tests/ocean/unit/test_weno_momentum.py:735:    def test_weno_differs_from_centered(self):
tests/ocean/unit/test_weno_momentum.py:736:        """WENO tendency should differ from centered (D+K add dissipation)."""
tests/ocean/unit/test_weno_momentum.py:752:        # They should differ — WENO adds D term and modifies K term
tests/ocean/unit/test_weno_momentum.py:753:        du_diff = float(jnp.max(jnp.abs(
tests/ocean/unit/test_weno_momentum.py:755:        dv_diff = float(jnp.max(jnp.abs(
tests/ocean/unit/test_weno_momentum.py:757:        assert du_diff > 1e-15 or dv_diff > 1e-15, (
tests/ocean/unit/test_weno_momentum.py:759:            f"du_diff={du_diff}, dv_diff={dv_diff}")
tests/atmosphere/hydrostatic/test_cases/dcmip_transport.py:87:    dsigma_full = jnp.diff(sigma_full)
tests/atmosphere/hydrostatic/test_cases/dcmip_transport.py:455:    _, dzs_dlon, dzs_dlat = _mountain_height_with_gradients(lon, lat)
tests/atmosphere/hydrostatic/test_cases/dcmip_transport.py:995:    zs, _, _ = _mountain_height_with_gradients(lon, lat)
tests/atmosphere/hydrostatic/test_cases/dcmip_transport.py:999:def _mountain_height_with_gradients(lon, lat):
tests/atmosphere/hydrostatic/test_cases/dcmip_transport.py:1101:    _, dzs_dlon, dzs_dlat = _mountain_height_with_gradients(lon, lat)
tests/unit/test_diff_taylor_tests.py:1:"""Taylor test gradient verification for each component.
tests/unit/test_diff_taylor_tests.py:3:The Taylor test verifies gradient correctness by checking:
tests/unit/test_diff_taylor_tests.py:4:  |J(x + h*dx) - J(x) - h * <grad, dx>| / h^2 → C  as h → 0
tests/unit/test_diff_taylor_tests.py:7:gradient is correct (2nd-order convergence). If it blows up, the
tests/unit/test_diff_taylor_tests.py:8:gradient is WRONG.
tests/unit/test_diff_taylor_tests.py:31:    For correct gradients, these should be approximately constant.
tests/unit/test_diff_taylor_tests.py:34:    grad = jax.grad(loss_fn)(x0)
tests/unit/test_diff_taylor_tests.py:35:    dx = grad / jnp.linalg.norm(grad)
tests/unit/test_diff_taylor_tests.py:36:    gdx = jnp.sum(grad * dx)
tests/unit/test_diff_taylor_tests.py:50:    and approximately constant for correct gradients.
tests/unit/test_diff_taylor_tests.py:54:    assert all(jnp.isfinite(r) for r in ratios), f"{name}: Taylor ratios have NaN/Inf"
tests/validation/test_ec_eigenvalues.py:175:print("4. Comparison of M matrices (max abs difference)")
tests/unit/test_operators_fc_3d.py:14:from legoesm.core.operators_fc import build_fc_config, fc_gradient_x, fc_divergence
tests/unit/test_operators_fc_3d.py:16:    fc_gradient_x_3d,
tests/unit/test_operators_fc_3d.py:17:    fc_gradient_y_3d,
tests/unit/test_operators_fc_3d.py:21:    fc_hyperdiffusion_3d,
tests/unit/test_operators_fc_3d.py:36:def test_gradient_x_3d_shape(grid_and_config):
tests/unit/test_operators_fc_3d.py:40:    gx = fc_gradient_x_3d(field, grid, fc_config)
tests/unit/test_operators_fc_3d.py:46:def test_gradient_y_3d_shape(grid_and_config):
tests/unit/test_operators_fc_3d.py:50:    gy = fc_gradient_y_3d(field, grid, fc_config)
tests/unit/test_operators_fc_3d.py:62:    assert jnp.all(jnp.isfinite(div))
tests/unit/test_operators_fc_3d.py:72:    assert jnp.all(jnp.isfinite(curl))
tests/unit/test_operators_fc_3d.py:84:def test_hyperdiffusion_3d_shape(grid_and_config):
tests/unit/test_operators_fc_3d.py:88:    hd = fc_hyperdiffusion_3d(field, grid, fc_config, coeff=1e15)
tests/unit/test_operators_fc_3d.py:100:    assert jnp.all(jnp.isfinite(fd))
tests/unit/test_operators_fc_3d.py:135:    gx_3d = fc_gradient_x_3d(field_3d, grid, fc_config)
tests/unit/test_operators_fc_3d.py:137:        gx_2d_k = fc_gradient_x(field_3d[..., k], grid, fc_config)
tests/unit/test_run_amip_cli.py:17:        "--gravity-wave-drag", "rayleigh",
tests/unit/test_run_amip_cli.py:35:    assert cfg.gravity_wave_drag == "rayleigh"
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_1.py:1:"""DCMIP-2025 Test Case 1: Mountain-triggered breaking gravity waves.
tests/unit/test_operators.py:9:    gradient_x, gradient_y, gradient,
tests/unit/test_operators.py:17:    """Tests for discrete differential operators."""
tests/unit/test_operators.py:19:    def test_gradient_constant_field(self, small_grid):
tests/unit/test_operators.py:23:        gx, gy = gradient(f, small_grid)
tests/unit/test_operators.py:56:    def test_operators_differentiable(self, small_grid):
tests/unit/test_operators.py:57:        """All operators should be differentiable with jax.grad."""
tests/unit/test_operators.py:60:            gx = gradient_x(f, small_grid)
tests/unit/test_operators.py:64:        grad_fn = jax.grad(loss)
tests/unit/test_operators.py:65:        grads = grad_fn(data)
tests/unit/test_operators.py:66:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_operators.py:68:    def test_divergence_differentiable(self, small_grid):
tests/unit/test_operators.py:69:        """Divergence should be differentiable."""
tests/unit/test_operators.py:79:        grads = jax.grad(loss, argnums=(0, 1))(u_data, v_data)
tests/unit/test_operators.py:80:        assert all(jnp.all(jnp.isfinite(g)) for g in grads)
tests/unit/test_operators.py:116:        diff = corrected - data
tests/unit/test_operators.py:117:        assert jnp.std(diff) < 1e-6
tests/unit/test_operators.py:157:    """Tests that conservation code is differentiable with jax.grad."""
tests/unit/test_operators.py:159:    def test_zero_mean_tendency_differentiable(self, small_grid):
tests/unit/test_operators.py:160:        """zero_mean_tendency should be differentiable."""
tests/unit/test_operators.py:168:        grads = jax.grad(loss)(data)
tests/unit/test_operators.py:169:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_operators.py:171:    def test_zero_mean_tendency_3d_differentiable(self, small_grid):
tests/unit/test_operators.py:172:        """3D zero_mean_tendency should be differentiable."""
tests/unit/test_operators.py:180:        grads = jax.grad(loss)(data)
tests/unit/test_operators.py:181:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_operators.py:183:    def test_fix_mass_hydrostatic_target_differentiable(self, small_grid):
tests/unit/test_operators.py:184:        """fix_mass_hydrostatic_target should be differentiable w.r.t. p_s."""
tests/unit/test_operators.py:209:        grads = jax.grad(loss)(p_s_data)
tests/unit/test_operators.py:210:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_version_guards.py:92:        """strict=True upgrades the warning to RuntimeError."""
tests/unit/test_scale_ensemble.py:70:    def test_perturb_members_differ(self):
tests/unit/test_scale_ensemble.py:74:        # Different members should have different values
tests/unit/test_scale_ensemble.py:75:        diffs = batched["T"][0] - batched["T"][1]
tests/unit/test_scale_ensemble.py:76:        assert jnp.any(diffs != 0.0)
tests/unit/test_scale_ensemble.py:237:    def test_ensemble_differentiable(self):
tests/unit/test_scale_ensemble.py:250:        grad = jax.grad(loss)(init)
tests/unit/test_scale_ensemble.py:252:        np.testing.assert_allclose(grad, 2.0)
tests/ocean/unit/test_advection_weno.py:4:1. Output shapes and finiteness
tests/ocean/unit/test_advection_weno.py:10:7. AD (reverse-mode gradient) correctness via Taylor test
tests/ocean/unit/test_advection_weno.py:91:    def test_finite_values(self):
tests/ocean/unit/test_advection_weno.py:92:        """Output is finite for random input."""
tests/ocean/unit/test_advection_weno.py:99:        assert jnp.all(jnp.isfinite(result))
tests/ocean/unit/test_advection_weno.py:140:    def test_finite_values(self):
tests/ocean/unit/test_advection_weno.py:147:        assert jnp.all(jnp.isfinite(result))
tests/ocean/unit/test_advection_weno.py:203:    def test_less_diffusive_than_upwind(self):
tests/ocean/unit/test_advection_weno.py:479:        g = jax.grad(loss)(f0)
tests/ocean/unit/test_advection_weno.py:503:        """Gradient through weno5_to_u_points is finite and nonzero."""
tests/ocean/unit/test_advection_weno.py:515:        g = jax.grad(loss)(f)
tests/ocean/unit/test_advection_weno.py:516:        assert jnp.all(jnp.isfinite(g))
tests/ocean/unit/test_advection_weno.py:520:        """Gradient through weno7 vertical is finite and nonzero."""
tests/ocean/unit/test_advection_weno.py:537:        g = jax.grad(loss)(field)
tests/ocean/unit/test_advection_weno.py:538:        assert jnp.all(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:10:* Differentiability: ``jax.grad`` through the tracer advection flows
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:11:  finitely w.r.t. an initial-q amplitude.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:128:        has a 1/cos φ pole that's not represented at finite truncation.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:131:        # helper (jnp.diff on a length-1 axis collapses to length 0).
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:177:    def test_grad_through_tracer_amplitude(self, grid, sigma_coord):
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:196:        g = jax.grad(loss)(jnp.array(0.01))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:197:        assert bool(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:205:    def _proper_hyperdiff(self, grid):
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:227:            hyperdiff_coeff=self._proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:238:        assert bool(jnp.all(jnp.isfinite(new_qv)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:270:            hyperdiff_coeff=self._proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:285:    def test_grad_through_step_with_tracers(
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:288:        """``jax.grad`` flows through one dycore step with tracers."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:294:            hyperdiff_coeff=self._proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:310:        g = jax.grad(loss)(jnp.array(1.0))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:311:        assert bool(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:319:    def _proper_hyperdiff(self, grid):
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:345:        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:381:            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:388:            hyperdiff_coeff=self._proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:399:        assert bool(jnp.all(jnp.isfinite(new_qv_w_conv)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:400:        assert bool(jnp.all(jnp.isfinite(new_qv_no_conv)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:406:        diff = float(jnp.max(jnp.abs(new_qv_w_conv - new_qv_no_conv)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:407:        assert diff > 0.0, (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:413:        assert diff < 0.1 * float(jnp.max(jnp.abs(qv))), (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:414:            f"Convection-induced change {diff} exceeded 10% of q_v"
tests/unit/test_physics_microphysics.py:100:# 4a  All outputs finite
tests/unit/test_physics_microphysics.py:104:def test_all_outputs_finite(scheme):
tests/unit/test_physics_microphysics.py:105:    """All MicrophysicsOutput fields should be finite."""
tests/unit/test_physics_microphysics.py:110:        assert jnp.all(jnp.isfinite(val)), f"{scheme}: {fname} has NaN/Inf"
tests/land/unit/test_stomata.py:22:    step_carbon_differland,
tests/land/unit/test_stomata.py:389:        cfg = CarbonConfig(scheme="differland")
tests/land/unit/test_stomata.py:394:        state1, flux1 = step_carbon_differland(
tests/land/unit/test_stomata.py:399:        state2, flux2 = step_carbon_differland(
tests/land/unit/test_stomata.py:407:        cfg = CarbonConfig(scheme="differland")
tests/land/unit/test_stomata.py:415:        self.assertTrue(jnp.all(jnp.isfinite(flux_out)))
tests/land/unit/test_stomata.py:467:        self.assertTrue(jnp.all(jnp.isfinite(resp.lhflx)))
tests/land/unit/test_stomata.py:468:        self.assertTrue(jnp.all(jnp.isfinite(resp.shflx)))
tests/land/unit/test_stomata.py:485:        # Jarvis should give different (typically lower) LH
tests/land/unit/test_stomata.py:499:            carbon=CarbonConfig(scheme="differland"),
tests/land/unit/test_stomata.py:511:        self.assertTrue(jnp.all(jnp.isfinite(resp.lhflx)))
tests/land/unit/test_stomata.py:512:        self.assertTrue(jnp.all(jnp.isfinite(resp.co2_flux)))
tests/land/unit/test_stomata.py:516:        """Farquhar+Medlyn stomata give different results from Ball-Berry."""
tests/land/unit/test_stomata.py:525:            carbon=CarbonConfig(scheme="differland"),
tests/land/unit/test_stomata.py:529:            carbon=CarbonConfig(scheme="differland"),
tests/land/unit/test_stomata.py:544:        # They should produce different fluxes
tests/land/unit/test_stomata.py:550:    """JAX differentiability of stomatal models."""
tests/land/unit/test_stomata.py:552:    def test_farquhar_differentiable(self):
tests/land/unit/test_stomata.py:553:        """Farquhar photosynthesis is differentiable w.r.t. Ci."""
tests/land/unit/test_stomata.py:561:        grad = jax.grad(f)(jnp.array(280.0))
tests/land/unit/test_stomata.py:562:        self.assertTrue(jnp.isfinite(grad))
tests/land/unit/test_stomata.py:563:        self.assertGreater(float(grad), 0.0)  # dA/dCi > 0
tests/land/unit/test_stomata.py:565:    def test_ball_berry_differentiable(self):
tests/land/unit/test_stomata.py:566:        """Ball-Berry is differentiable w.r.t. A."""
tests/land/unit/test_stomata.py:573:        grad = jax.grad(f)(jnp.array(10.0))
tests/land/unit/test_stomata.py:574:        self.assertTrue(jnp.isfinite(grad))
tests/land/unit/test_stomata.py:575:        self.assertGreater(float(grad), 0.0)
tests/land/unit/test_stomata.py:577:    def test_coupled_solver_differentiable(self):
tests/land/unit/test_stomata.py:578:        """Coupled Farquhar-stomata solver is differentiable w.r.t. T."""
tests/land/unit/test_stomata.py:588:        grad = jax.grad(f)(jnp.array([298.15]))
tests/land/unit/test_stomata.py:589:        self.assertTrue(jnp.all(jnp.isfinite(grad)))
tests/land/unit/test_stomata.py:591:    def test_jarvis_differentiable(self):
tests/land/unit/test_stomata.py:592:        """Jarvis model is differentiable w.r.t. T."""
tests/land/unit/test_stomata.py:600:        grad = jax.grad(f)(jnp.array(298.15))
tests/land/unit/test_stomata.py:601:        self.assertTrue(jnp.isfinite(grad))
tests/unit/test_gradient_checkpointing.py:1:"""Tests for gradient checkpointing through the compiled dycore.
tests/unit/test_gradient_checkpointing.py:111:def _build(gradient_checkpoint, tau_equator=7.2):
tests/unit/test_gradient_checkpointing.py:119:        gradient_checkpoint=gradient_checkpoint,
tests/unit/test_gradient_checkpointing.py:128:        run_ckpt = _build(gradient_checkpoint=True)
tests/unit/test_gradient_checkpointing.py:129:        run_nockpt = _build(gradient_checkpoint=False)
tests/unit/test_gradient_checkpointing.py:140:        run_ckpt = _build(gradient_checkpoint=True)
tests/unit/test_gradient_checkpointing.py:141:        run_nockpt = _build(gradient_checkpoint=False)
tests/unit/test_gradient_checkpointing.py:155:    def test_grad_5_steps(self):
tests/unit/test_gradient_checkpointing.py:156:        """5-step gradient is finite and nonzero."""
tests/unit/test_gradient_checkpointing.py:157:        run_seg = _build(gradient_checkpoint=True, tau_equator=7.2)
tests/unit/test_gradient_checkpointing.py:160:            seg = _build(gradient_checkpoint=True, tau_equator=tau)
tests/unit/test_gradient_checkpointing.py:166:        grad = jax.grad(loss)(jnp.float32(7.2))
tests/unit/test_gradient_checkpointing.py:167:        assert jnp.isfinite(grad)
tests/unit/test_gradient_checkpointing.py:168:        assert float(jnp.abs(grad)) > 0
tests/unit/test_gradient_checkpointing.py:170:    def test_grad_100_steps(self):
tests/unit/test_gradient_checkpointing.py:171:        """100-step gradient through checkpointed scan is finite."""
tests/unit/test_gradient_checkpointing.py:173:            seg = _build(gradient_checkpoint=True, tau_equator=tau)
tests/unit/test_gradient_checkpointing.py:179:        grad = jax.grad(loss)(jnp.float32(7.2))
tests/unit/test_gradient_checkpointing.py:180:        assert jnp.isfinite(grad), f"Gradient is {grad}"
tests/unit/test_gradient_checkpointing.py:181:        assert float(jnp.abs(grad)) > 0
tests/unit/test_gradient_checkpointing.py:183:    def test_grad_checkpointed_matches_uncheckpointed(self):
tests/unit/test_gradient_checkpointing.py:184:        """Checkpointed and uncheckpointed gradients agree."""
tests/unit/test_gradient_checkpointing.py:186:            seg = _build(gradient_checkpoint=ckpt, tau_equator=tau)
tests/unit/test_gradient_checkpointing.py:193:        grad_ckpt = jax.grad(loss)(tau, True)
tests/unit/test_gradient_checkpointing.py:194:        grad_nockpt = jax.grad(loss)(tau, False)
tests/unit/test_gradient_checkpointing.py:197:            float(grad_ckpt), float(grad_nockpt), rtol=1e-4,
tests/unit/test_gradient_checkpointing.py:204:    def test_multi_segment_gradient(self):
tests/unit/test_gradient_checkpointing.py:207:            seg = _build(gradient_checkpoint=True, tau_equator=tau)
tests/unit/test_gradient_checkpointing.py:220:        grad = jax.grad(loss)(jnp.float32(7.2))
tests/unit/test_gradient_checkpointing.py:221:        assert jnp.isfinite(grad)
tests/unit/test_gradient_checkpointing.py:222:        assert float(jnp.abs(grad)) > 0
tests/unit/test_gradient_checkpointing.py:223:        print(f"Hierarchical checkpoint gradient: {float(grad):.6e}")
tests/unit/test_external_forcing.py:293:        assert np.all(np.isfinite(np.asarray(out)))
tests/unit/test_scale_latlon_spectral.py:218:        diffs = jnp.diff(grid.lat)
tests/unit/test_scale_latlon_spectral.py:219:        assert jnp.all(diffs > 0)
tests/land/unit/test_carbon_cycle.py:30:    step_carbon_differland,
tests/land/unit/test_carbon_cycle.py:69:    def test_differland_construction(self):
tests/land/unit/test_carbon_cycle.py:70:        cfg = CarbonConfig(scheme="differland", epsilon=1.5)
tests/land/unit/test_carbon_cycle.py:71:        self.assertEqual(cfg.scheme, "differland")
tests/land/unit/test_carbon_cycle.py:101:        self.assertTrue(jnp.all(jnp.isfinite(gpp)))
tests/land/unit/test_carbon_cycle.py:205:    def test_phenology_finite(self):
tests/land/unit/test_carbon_cycle.py:209:        self.assertTrue(jnp.all(jnp.isfinite(lrf)))
tests/land/unit/test_carbon_cycle.py:210:        self.assertTrue(jnp.all(jnp.isfinite(lff)))
tests/land/unit/test_carbon_cycle.py:250:        cfg = config or _default_config(scheme="differland")
tests/land/unit/test_carbon_cycle.py:260:        return step_carbon_differland(
tests/land/unit/test_carbon_cycle.py:264:    def test_step_finite(self):
tests/land/unit/test_carbon_cycle.py:268:            self.assertTrue(jnp.all(jnp.isfinite(arr)), f"{name} not finite")
tests/land/unit/test_carbon_cycle.py:269:        self.assertTrue(jnp.all(jnp.isfinite(co2_flux)))
tests/land/unit/test_carbon_cycle.py:286:        cfg = _default_config(scheme="differland", epsilon=2.0)
tests/land/unit/test_carbon_cycle.py:294:        _, co2_flux = step_carbon_differland(
tests/land/unit/test_carbon_cycle.py:304:        cfg = _default_config(scheme="differland")
tests/land/unit/test_carbon_cycle.py:312:        _, co2_flux = step_carbon_differland(
tests/land/unit/test_carbon_cycle.py:321:        cfg = _default_config(scheme="differland")
tests/land/unit/test_carbon_cycle.py:332:            state, flux = step_carbon_differland(
tests/land/unit/test_carbon_cycle.py:337:            self.assertTrue(jnp.all(jnp.isfinite(arr)),
tests/land/unit/test_carbon_cycle.py:338:                            f"{name} not finite after 100 steps")
tests/land/unit/test_carbon_cycle.py:342:        cfg = _default_config(scheme="differland")
tests/land/unit/test_carbon_cycle.py:353:        new_state, co2_flux = step_carbon_differland(
tests/land/unit/test_carbon_cycle.py:451:        self.assertTrue(jnp.all(jnp.isfinite(flux)))
tests/land/unit/test_carbon_cycle.py:453:    def test_differland_requires_state(self):
tests/land/unit/test_carbon_cycle.py:454:        cfg = _default_config(scheme="differland")
tests/land/unit/test_carbon_cycle.py:458:    def test_differland_dispatch(self):
tests/land/unit/test_carbon_cycle.py:459:        cfg = _default_config(scheme="differland")
tests/land/unit/test_carbon_cycle.py:463:        self.assertTrue(jnp.all(jnp.isfinite(flux)))
tests/land/unit/test_carbon_cycle.py:491:    def test_slab_land_with_differland(self):
tests/land/unit/test_carbon_cycle.py:492:        """step_land with differland carbon returns non-zero co2_flux."""
tests/land/unit/test_carbon_cycle.py:501:        carbon_cfg = CarbonConfig(scheme="differland", epsilon=2.0)
tests/land/unit/test_carbon_cycle.py:537:                         "co2_flux should be non-zero with differland")
tests/land/unit/test_carbon_cycle.py:538:        self.assertTrue(jnp.all(jnp.isfinite(resp.co2_flux)))
tests/land/unit/test_carbon_cycle.py:589:    def test_gpp_grad(self):
tests/land/unit/test_carbon_cycle.py:590:        """GPP should be differentiable w.r.t. T."""
tests/land/unit/test_carbon_cycle.py:599:        grad = jax.grad(loss)(jnp.array([290.0]))
tests/land/unit/test_carbon_cycle.py:600:        self.assertTrue(jnp.all(jnp.isfinite(grad)))
tests/land/unit/test_carbon_cycle.py:601:        self.assertFalse(jnp.allclose(grad, 0.0))
tests/land/unit/test_carbon_cycle.py:603:    def test_step_grad(self):
tests/land/unit/test_carbon_cycle.py:604:        """Full carbon step should be differentiable w.r.t. radiation."""
tests/land/unit/test_carbon_cycle.py:605:        cfg = _default_config(scheme="differland")
tests/land/unit/test_carbon_cycle.py:610:            new_state, flux = step_carbon_differland(
tests/land/unit/test_carbon_cycle.py:618:        grad = jax.grad(loss)(jnp.full(ncol, 300.0))
tests/land/unit/test_carbon_cycle.py:619:        self.assertTrue(jnp.all(jnp.isfinite(grad)))
tests/land/unit/test_carbon_cycle.py:620:        # More SW -> more GPP -> more negative NEE -> negative gradient
tests/land/unit/test_carbon_cycle.py:621:        self.assertTrue(jnp.all(grad < 0),
tests/land/unit/test_carbon_cycle.py:622:                        f"Expected negative grad w.r.t. sw, got {grad}")
tests/land/unit/test_carbon_cycle.py:624:    def test_seasonal_grad(self):
tests/land/unit/test_carbon_cycle.py:625:        """Seasonal cycle should be differentiable w.r.t. latitude."""
tests/land/unit/test_carbon_cycle.py:631:        grad = jax.grad(loss)(jnp.array([0.7]))
tests/land/unit/test_carbon_cycle.py:632:        self.assertTrue(jnp.all(jnp.isfinite(grad)))
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:13:* the result is finite at numerical singularities (T → 0);
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:14:* differentiability through ``ω`` and ``T``.
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:84:def test_finite_at_low_temperature():
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:85:    """The 1 K floor on ``T_v`` should keep ``w`` finite when the
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:91:    assert bool(jnp.all(jnp.isfinite(w)))
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:94:def test_grad_through_omega():
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:95:    """``jax.grad`` through ω flows; sensitivity is ``-1/(ρ g)`` per
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:104:    g = jax.grad(loss)(jnp.array(0.1))
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:105:    assert bool(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:110:def test_grad_through_temperature():
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:111:    """jax.grad through T flows (no nondiff branches)."""
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:119:    g = jax.grad(loss)(jnp.array(1.0))
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:120:    assert bool(jnp.isfinite(g))
tests/unit/test_physics_ocean.py:3:Tests vertical mixing, bottom drag, and ocean convection for finite
tests/unit/test_physics_ocean.py:46:    """Each vertical mixing scheme produces finite outputs."""
tests/unit/test_physics_ocean.py:55:    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{scheme}: du_dt has NaN/Inf"
tests/unit/test_physics_ocean.py:56:    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{scheme}: dv_dt has NaN/Inf"
tests/unit/test_physics_ocean.py:57:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt has NaN/Inf"
tests/unit/test_physics_ocean.py:58:    assert jnp.all(jnp.isfinite(tend.dS_dt.data)), f"{scheme}: dS_dt has NaN/Inf"
tests/unit/test_physics_ocean.py:67:    """Each bottom drag scheme produces finite outputs."""
tests/unit/test_physics_ocean.py:76:    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{scheme}: du_dt NaN/Inf"
tests/unit/test_physics_ocean.py:77:    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{scheme}: dv_dt NaN/Inf"
tests/unit/test_physics_ocean.py:128:@pytest.mark.parametrize("scheme", ["enhanced_diffusion", "plume"])
tests/unit/test_physics_ocean.py:130:    """Each ocean convection scheme produces finite outputs."""
tests/unit/test_physics_ocean.py:141:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt NaN/Inf"
tests/unit/test_physics_ocean.py:142:    assert jnp.all(jnp.isfinite(tend.dS_dt.data)), f"{scheme}: dS_dt NaN/Inf"
tests/unit/test_backend_guard.py:87:            with pytest.raises(ValueError, match="finite-volume"):
tests/land/unit/test_land_audit_fixes.py:275:    def test_dry_top_wet_deep_differs_from_all_dry(self):
tests/unit/test_neural_gcm_spectral.py:7:- Loss function returns a finite scalar
tests/unit/test_neural_gcm_spectral.py:64:def _assert_state_finite(pytree, label="state"):
tests/unit/test_neural_gcm_spectral.py:66:    leaves are finite.  Handles both ``Field`` slots and the
tests/unit/test_neural_gcm_spectral.py:74:                assert bool(jnp.all(jnp.isfinite(data))), (
tests/unit/test_neural_gcm_spectral.py:78:            assert bool(jnp.all(jnp.isfinite(field.data))), (
tests/unit/test_neural_gcm_spectral.py:119:    def test_finite_values(self):
tests/unit/test_neural_gcm_spectral.py:123:        _assert_state_finite(state, label="carry-to-spectral state")
tests/unit/test_neural_gcm_spectral.py:149:    def test_tendencies_finite(self):
tests/unit/test_neural_gcm_spectral.py:161:        _assert_state_finite(tendencies, label="tendency")
tests/unit/test_neural_gcm_spectral.py:184:            hyperdiff_coeff=1e14,
tests/unit/test_neural_gcm_spectral.py:193:        _assert_state_finite(result, label="rollout result")
tests/unit/test_neural_gcm_spectral.py:225:    def test_finite_scalar(self):
tests/unit/test_neural_gcm_spectral.py:240:        assert jnp.isfinite(loss)
tests/unit/test_neural_gcm_spectral.py:265:    def test_grad_through_rollout(self):
tests/unit/test_neural_gcm_spectral.py:266:        """Verify gradients flow from loss through rollout to SFNO weights."""
tests/unit/test_neural_gcm_spectral.py:282:            hyperdiff_coeff=1e14,
tests/unit/test_neural_gcm_spectral.py:296:        loss, grads = eqx.filter_value_and_grad(loss_fn)(sfno)
tests/unit/test_neural_gcm_spectral.py:298:        # Loss should be finite
tests/unit/test_neural_gcm_spectral.py:299:        assert jnp.isfinite(loss)
tests/unit/test_neural_gcm_spectral.py:301:        # Grads should be finite and non-zero for at least some params
tests/unit/test_neural_gcm_spectral.py:302:        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
tests/unit/test_neural_gcm_spectral.py:303:        assert len(grad_leaves) > 0
tests/unit/test_neural_gcm_spectral.py:304:        all_finite = all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
tests/unit/test_neural_gcm_spectral.py:305:        assert all_finite, "Some gradients are NaN/Inf"
tests/unit/test_neural_gcm_spectral.py:307:        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves)
tests/unit/test_neural_gcm_spectral.py:308:        assert has_nonzero, "All gradients are zero — no signal flows"
tests/unit/test_neural_gcm_spectral.py:366:    def test_tendencies_finite(self):
tests/unit/test_neural_gcm_spectral.py:378:        _assert_state_finite(tend, label="column-MLP tendency")
tests/unit/test_neural_gcm_spectral.py:394:            hyperdiff_coeff=1e14, time_integrator="ssp_rk3",
tests/unit/test_neural_gcm_spectral.py:400:        _assert_state_finite(result, label="column-MLP rollout")
tests/unit/test_neural_gcm_spectral.py:402:    def test_grad_through_column_mlp(self):
tests/unit/test_neural_gcm_spectral.py:419:            hyperdiff_coeff=1e14, time_integrator="ssp_rk3",
tests/unit/test_neural_gcm_spectral.py:432:        loss, grads = eqx.filter_value_and_grad(loss_fn)(nn)
tests/unit/test_neural_gcm_spectral.py:434:        assert jnp.isfinite(loss)
tests/unit/test_neural_gcm_spectral.py:435:        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
tests/unit/test_neural_gcm_spectral.py:436:        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
tests/unit/test_neural_gcm_spectral.py:437:        assert any(jnp.any(g != 0) for g in grad_leaves)
tests/unit/test_neural_gcm_spectral.py:460:        _assert_state_finite(tend, label="physics-params tendency")
tests/unit/test_neural_gcm_spectral.py:462:    def test_grad_through_physics_params(self):
tests/unit/test_neural_gcm_spectral.py:480:            hyperdiff_coeff=1e14, time_integrator="ssp_rk3",
tests/unit/test_neural_gcm_spectral.py:493:        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
tests/unit/test_neural_gcm_spectral.py:495:        assert jnp.isfinite(loss)
tests/unit/test_neural_gcm_spectral.py:496:        # Grads should be finite for all raw_values
tests/unit/test_neural_gcm_spectral.py:497:        for name, g in grads.raw_values.items():
tests/unit/test_neural_gcm_spectral.py:498:            assert jnp.isfinite(g), f"Non-finite grad for {name}"
tests/unit/test_neural_gcm_spectral.py:499:        # At least some grads should be non-zero
tests/unit/test_neural_gcm_spectral.py:500:        has_nonzero = any(g != 0 for g in grads.raw_values.values())
tests/unit/test_neural_gcm_spectral.py:501:        assert has_nonzero, "All physics param gradients are zero"
tests/unit/test_deprecation_warnings.py:81:        """'centered' and 'finite_volume' are ambiguous (cdgrid on cubed-sphere,
tests/unit/test_deprecation_warnings.py:119:    def test_finite_volume_resolves_to_cdgrid(self):
tests/unit/test_deprecation_warnings.py:120:        """'finite_volume' is ambiguous — defaults to cdgrid without warning."""
tests/unit/test_deprecation_warnings.py:125:                dynamics="hydrostatic", discretization="finite_volume",
tests/unit/test_deprecation_warnings.py:162:    @pytest.mark.parametrize("alias", ["centered", "finite_volume", "fv"])
tests/unit/test_thermodynamics.py:48:        assert jnp.all(jnp.isfinite(p_half))
tests/unit/test_thermodynamics.py:54:    def test_handles_unphysical_inputs_with_positive_finite_output(self):
tests/unit/test_thermodynamics.py:58:        assert jnp.all(jnp.isfinite(p))
tests/ocean/unit/test_gm_redi_eady_physics.py:12:  the isopycnal gradient of T is zero everywhere.
tests/ocean/unit/test_gm_redi_eady_physics.py:31:from legoesm.ocean.vertical import create_ocean_z_star, compute_ocean_jacobian
tests/ocean/unit/test_gm_redi_eady_physics.py:83:    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
tests/ocean/unit/test_gm_redi_eady_physics.py:85:    return grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config
tests/ocean/unit/test_gm_redi_eady_physics.py:90:    Redi (isopycnal diffusion) should produce zero T tendency."""
tests/ocean/unit/test_gm_redi_eady_physics.py:93:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
tests/ocean/unit/test_gm_redi_eady_physics.py:102:            rho, mask, z_coord, jacobian, grid, cfg_redi,
tests/ocean/unit/test_gm_redi_eady_physics.py:116:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_eady_physics.py:139:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
tests/ocean/unit/test_gm_redi_eady_physics.py:146:            rho, mask, z_coord, jacobian, grid, cfg_gm,
tests/ocean/unit/test_gm_redi_eady_physics.py:151:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_eady_physics.py:158:        assert jnp.all(jnp.isfinite(dT_gm)), "GM tendency has NaN/Inf"
tests/ocean/unit/test_gm_redi_eady_physics.py:162:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
tests/ocean/unit/test_gm_redi_eady_physics.py:168:            rho, mask, z_coord, jacobian, grid, cfg_gm,
tests/ocean/unit/test_gm_redi_eady_physics.py:173:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_eady_physics.py:177:        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
tests/ocean/unit/test_gm_redi_eady_physics.py:190:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
tests/ocean/unit/test_gm_redi_eady_physics.py:196:            rho, mask, z_coord, jacobian, grid, cfg_gm,
tests/ocean/unit/test_gm_redi_eady_physics.py:201:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_eady_physics.py:205:        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
tests/ocean/unit/test_gm_redi_eady_physics.py:227:# kappa_Redi or how steep the slopes are.  Centred-difference
tests/ocean/unit/test_gm_redi_eady_physics.py:234:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
tests/ocean/unit/test_gm_redi_eady_physics.py:236:        return grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho
tests/ocean/unit/test_gm_redi_eady_physics.py:245:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho = self._eady_rho()
tests/ocean/unit/test_gm_redi_eady_physics.py:251:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_eady_physics.py:276:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho = self._eady_rho()
tests/ocean/unit/test_gm_redi_eady_physics.py:282:                z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_eady_physics.py:308:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho = self._eady_rho()
tests/ocean/unit/test_gm_redi_eady_physics.py:313:            rho, mask, z_coord, jacobian, grid, cfg_c,
tests/ocean/unit/test_gm_redi_eady_physics.py:317:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_eady_physics.py:322:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_eady_physics.py:328:        # factor that absorbs the (slightly different) floating-point
tests/ocean/unit/test_gm_redi_eady_physics.py:337:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, _ = self._eady_rho()
tests/ocean/unit/test_gm_redi_eady_physics.py:377:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
tests/ocean/unit/test_gm_redi_eady_physics.py:379:        return grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho
tests/ocean/unit/test_gm_redi_eady_physics.py:382:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho = self._eady_rho()
tests/ocean/unit/test_gm_redi_eady_physics.py:385:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_eady_physics.py:390:        assert jnp.all(jnp.isfinite(dT_gm))
tests/ocean/unit/test_gm_redi_eady_physics.py:392:        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
tests/ocean/unit/test_gm_redi_eady_physics.py:399:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, rho = self._eady_rho()
tests/ocean/unit/test_gm_redi_eady_physics.py:402:            z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_eady_physics.py:405:        dz = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
tests/ocean/unit/test_gm_redi_eady_physics.py:416:    """jax.grad must flow cleanly through the triad path."""
tests/ocean/unit/test_gm_redi_eady_physics.py:418:    def test_grad_through_triad_tendency(self):
tests/ocean/unit/test_gm_redi_eady_physics.py:419:        grid, z_coord, mask, u_mask, v_mask, jacobian, T, S, config = _make_eady_setup()
tests/ocean/unit/test_gm_redi_eady_physics.py:425:                z_coord, jacobian, grid,
tests/ocean/unit/test_gm_redi_eady_physics.py:430:        grad = jax.grad(loss)(T)
tests/ocean/unit/test_gm_redi_eady_physics.py:431:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_land_ice_lake.py:60:    def test_returns_finite(self):
tests/unit/test_land_ice_lake.py:64:        assert jnp.all(jnp.isfinite(new_state.T_epi.data))
tests/unit/test_land_ice_lake.py:65:        assert jnp.all(jnp.isfinite(new_state.T_hypo.data))
tests/unit/test_land_ice_lake.py:67:            assert jnp.all(jnp.isfinite(getattr(resp, name)))
tests/unit/test_land_ice_lake.py:117:        assert jnp.all(jnp.isfinite(state.T_epi.data))
tests/unit/test_scale_halo.py:62:            # Depth-0 halo (adjacent to interior)
tests/unit/test_scale_halo.py:237:    def test_pad_halo_differentiable(self):
tests/unit/test_scale_halo.py:238:        """pad_halo should be differentiable for AD."""
tests/unit/test_scale_halo.py:243:        grad = jax.grad(loss)(data)
tests/unit/test_scale_halo.py:244:        assert grad.shape == data.shape
tests/unit/test_scale_halo.py:245:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_conservation_laws.py:62:            hyperdiff_coeff=0.0,
tests/unit/test_conservation_laws.py:99:    def test_energy_finite(self):
tests/unit/test_conservation_laws.py:100:        """Energy should stay finite after integration."""
tests/unit/test_conservation_laws.py:105:        assert jnp.all(jnp.isfinite(state.h)), "h has non-finite values"
tests/unit/test_conservation_laws.py:106:        assert jnp.all(jnp.isfinite(state.u_d)), "u_d has non-finite values"
tests/unit/test_conservation_laws.py:107:        assert jnp.all(jnp.isfinite(state.v_d)), "v_d has non-finite values"
tests/unit/test_conservation_laws.py:258:        assert jnp.all(jnp.isfinite(state.h.data)), "h non-finite"
tests/unit/test_diff_sea_ice.py:18:def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
tests/unit/test_diff_sea_ice.py:19:    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
tests/unit/test_diff_sea_ice.py:20:    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
tests/unit/test_diff_sea_ice.py:75:    def test_grad_wrt_T_ice(self):
tests/unit/test_diff_sea_ice.py:86:        grad = jax.grad(loss)(state.T_ice.data)
tests/unit/test_diff_sea_ice.py:87:        assert_gradient_ok(grad, "Slab ice w.r.t. T_ice")
tests/unit/test_diff_sea_ice.py:89:    def test_grad_wrt_ocean_sst(self):
tests/unit/test_diff_sea_ice.py:100:        grad = jax.grad(loss)(self.ocean_sst)
tests/unit/test_diff_sea_ice.py:101:        assert_gradient_ok(grad, "Slab ice h_ice w.r.t. ocean_sst")
tests/unit/test_diff_sea_ice.py:123:            differentiable_dynamics=True,
tests/unit/test_diff_sea_ice.py:142:    def test_grad_wrt_h_ice(self):
tests/unit/test_diff_sea_ice.py:153:        grad = jax.grad(loss)(state.h_ice.data)
tests/unit/test_diff_sea_ice.py:154:        assert_gradient_ok(grad, "Dynamic ice w.r.t. h_ice")
tests/unit/test_diff_sea_ice.py:163:    def test_ice_strength_grad(self):
tests/unit/test_diff_sea_ice.py:169:        dP_dh = jax.grad(lambda h: jnp.sum(ice_strength(h, A)))(h)
tests/unit/test_diff_sea_ice.py:170:        assert jnp.all(jnp.isfinite(dP_dh)), "dP/dh not finite"
tests/unit/test_diff_sea_ice.py:173:    def test_evp_stress_update_grad(self):
tests/unit/test_diff_sea_ice.py:193:        grad = jax.grad(loss)(eps_11)
tests/unit/test_diff_sea_ice.py:194:        assert_gradient_ok(grad, "EVP stress update w.r.t. eps_11")
tests/unit/test_diff_sea_ice.py:198:# 5d  Multi-category ITD — linear_remap differentiability
tests/unit/test_diff_sea_ice.py:203:    def test_linear_remap_grad_h(self):
tests/unit/test_diff_sea_ice.py:229:        grad = jax.grad(loss)(h_new)
tests/unit/test_diff_sea_ice.py:230:        assert_gradient_ok(grad, "ITD linear_remap w.r.t. h_new", min_nonzero_frac=0.05)
tests/unit/test_diff_sea_ice.py:232:    def test_aggregate_state_grad(self):
tests/unit/test_diff_sea_ice.py:246:        grad = jax.grad(loss)(h_ice)
tests/unit/test_diff_sea_ice.py:247:        assert_gradient_ok(grad, "ITD aggregate w.r.t. h_ice", min_nonzero_frac=0.05)
tests/unit/test_diff_sea_ice.py:272:        grad = jax.grad(absorbed_sw)(T_ice)
tests/unit/test_diff_sea_ice.py:273:        assert jnp.all(jnp.isfinite(grad)), "Albedo feedback gradient not finite"
tests/unit/test_diff_sea_ice.py:275:        assert jnp.mean(grad) > 0, (
tests/unit/test_diff_sea_ice.py:276:            f"Expected positive d(absorbed_SW)/d(T_ice), got mean={jnp.mean(grad):.6e}"
tests/unit/test_diff_sea_ice.py:279:    def test_slab_ice_with_temp_albedo_grad(self):
tests/unit/test_diff_sea_ice.py:304:        grad = jax.grad(loss)(state.T_ice.data)
tests/unit/test_diff_sea_ice.py:305:        assert_gradient_ok(grad, "Slab ice (temp-dependent albedo) w.r.t. T_ice")
tests/atmosphere/hydrostatic/unit/test_topography.py:88:    def test_latitude_gradient(self):
tests/atmosphere/hydrostatic/unit/test_topography.py:89:        """Linear latitude gradient should interpolate correctly."""
tests/atmosphere/hydrostatic/unit/test_topography.py:394:        # Compute edge differences for face 0-1 boundary
tests/atmosphere/hydrostatic/unit/test_topography.py:396:        edge_diff_no_blend = float(jnp.mean(jnp.abs(
tests/atmosphere/hydrostatic/unit/test_topography.py:399:        edge_diff_blend = float(jnp.mean(jnp.abs(
tests/atmosphere/hydrostatic/unit/test_topography.py:404:        self.assertLessEqual(edge_diff_blend, edge_diff_no_blend + 1e-10)
tests/unit/test_field.py:51:    def test_grad_compatible(self):
tests/unit/test_field.py:52:        """jax.grad should work through Field operations."""
tests/unit/test_field.py:57:        grad_fn = jax.grad(loss)
tests/unit/test_field.py:59:        grads = grad_fn(x)
tests/unit/test_field.py:60:        assert jnp.allclose(grads, 2 * x)
tests/unit/test_scale_jit_health.py:48:    def test_different_metadata_retrace(self):
tests/unit/test_scale_jit_health.py:65:        assert c2 > c1, "Did not recompile with different metadata"
tests/unit/test_scale_jit_health.py:112:        _ = step(jnp.ones(10))  # different shape
tests/unit/test_training_modules.py:4:and gradient flow is verified for each training mode.
tests/unit/test_training_modules.py:129:    def test_differentiable(self):
tests/unit/test_training_modules.py:134:        grad = jax.grad(lambda ps: jnp.mean(
tests/unit/test_training_modules.py:137:        assert jnp.isfinite(grad)
tests/unit/test_training_modules.py:188:    def test_gradient_flow(self):
tests/unit/test_training_modules.py:193:        _, grads = eqx.filter_value_and_grad(loss_fn)(p)
tests/unit/test_training_modules.py:194:        assert all(jnp.isfinite(v) for v in grads.raw_values.values())
tests/unit/test_training_modules.py:426:    def test_sbm_params_gradient_flow(self):
tests/unit/test_training_modules.py:436:            _, grads = eqx.filter_value_and_grad(loss_fn)(p)
tests/unit/test_training_modules.py:437:            assert all(jnp.isfinite(v) for v in grads.raw_values.values()), (
tests/unit/test_training_modules.py:438:                f"Non-finite grad for scheme={scheme}"
tests/unit/test_physics_convection.py:182:    known consequence of the differentiable formulation, not a bug.
tests/unit/test_physics_convection.py:298:# All outputs finite
tests/unit/test_physics_convection.py:339:    assert jnp.all(jnp.diff(gate) >= 0), "gate must be monotonic in pressure"
tests/unit/test_physics_convection.py:342:    assert jnp.all(jnp.diff(transition) > 0), (
tests/unit/test_physics_convection.py:388:    du_layer = jnp.diff(u_env, axis=-1, prepend=u_env[:, :1])
tests/unit/test_physics_convection.py:390:    dflux_ref = jnp.diff(flux_ref, axis=-1, append=flux_ref[:, -1:])
tests/unit/test_physics_convection.py:407:    dflux_ungated = jnp.diff(
tests/unit/test_physics_convection.py:420:def test_all_outputs_finite(scheme):
tests/unit/test_physics_convection.py:421:    """All ConvectionOutput fields should be finite."""
tests/unit/test_physics_convection.py:424:    assert jnp.all(jnp.isfinite(out.dT_dt)), f"{scheme}: dT_dt has NaN/Inf"
tests/unit/test_physics_convection.py:425:    assert jnp.all(jnp.isfinite(out.dq_v_dt)), f"{scheme}: dq_v_dt has NaN/Inf"
tests/unit/test_physics_convection.py:426:    assert jnp.all(jnp.isfinite(out.dq_c_conv_dt)), (
tests/unit/test_physics_convection.py:429:    assert jnp.all(jnp.isfinite(out.cape)), f"{scheme}: cape has NaN/Inf"
tests/unit/test_physics_convection.py:436:# Each scheme's "natural" energy invariant differs by design under Option C:
tests/unit/test_physics_convection.py:454:#     budgets do NOT close to <10 W/m^2 at finite resolution because the
tests/unit/test_physics_convection.py:604:    without locking to a specific level (different schemes peak their
tests/unit/test_physics_convection.py:605:    drying at different heights: SBM/DCA in the BL, mass-flux/EDMF
tests/unit/test_fv3_audit_harness.py:269:    def test_all_tendencies_finite(self):
tests/unit/test_fv3_audit_harness.py:275:        self.assertTrue(jnp.all(jnp.isfinite(dh_dt)))
tests/unit/test_fv3_audit_harness.py:276:        self.assertTrue(jnp.all(jnp.isfinite(du_dt)))
tests/unit/test_fv3_audit_harness.py:277:        self.assertTrue(jnp.all(jnp.isfinite(dv_dt)))
tests/unit/test_fv3_audit_harness.py:301:            div_damp=0.0, hyperdiff_coeff=0.0,
tests/unit/test_fv3_audit_harness.py:333:            div_damp=0.0, hyperdiff_coeff=0.0,
tests/unit/test_fv3_audit_harness.py:395:            # Check that edge positions are finite
tests/unit/test_fv3_audit_harness.py:397:                jnp.all(jnp.isfinite(cdgrid.lon_edge_x[face])),
tests/unit/test_fv3_audit_harness.py:398:                f"Face {face} lon_edge_x has non-finite values")
tests/unit/test_fv3_audit_harness.py:400:                jnp.all(jnp.isfinite(cdgrid.lat_edge_x[face])),
tests/unit/test_fv3_audit_harness.py:401:                f"Face {face} lat_edge_x has non-finite values")
tests/unit/test_fv3_audit_harness.py:403:    def test_corner_grad_c_matrix_finite(self):
tests/unit/test_fv3_audit_harness.py:404:        """Arakawa-Lamb gradient matrix should be finite at all corners."""
tests/unit/test_fv3_audit_harness.py:406:        self.assertTrue(jnp.all(jnp.isfinite(cdgrid.grad_c00)))
tests/unit/test_fv3_audit_harness.py:407:        self.assertTrue(jnp.all(jnp.isfinite(cdgrid.grad_c01)))
tests/unit/test_fv3_audit_harness.py:408:        self.assertTrue(jnp.all(jnp.isfinite(cdgrid.grad_c10)))
tests/unit/test_fv3_audit_harness.py:409:        self.assertTrue(jnp.all(jnp.isfinite(cdgrid.grad_c11)))
tests/unit/test_fv3_audit_harness.py:411:    def test_corner_grad_c_nonzero_interior(self):
tests/unit/test_fv3_audit_harness.py:416:        interior_00 = cdgrid.grad_c00[:, 1:n, 1:n]
tests/unit/test_fv3_audit_harness.py:417:        interior_11 = cdgrid.grad_c11[:, 1:n, 1:n]
tests/unit/test_fv3_audit_harness.py:419:                        "grad_c00 is zero in interior")
tests/unit/test_fv3_audit_harness.py:421:                        "grad_c11 is zero in interior")
tests/unit/test_fv3_audit_harness.py:432:        """100 steps of cosine bell should remain stable and finite."""
tests/unit/test_fv3_audit_harness.py:442:            div_damp=0.0, hyperdiff_coeff=0.0,
tests/unit/test_fv3_audit_harness.py:455:        self.assertTrue(jnp.all(jnp.isfinite(state.h)),
tests/unit/test_fv3_audit_harness.py:456:                        "h has non-finite values after 100 steps")
tests/unit/test_fv3_audit_harness.py:457:        self.assertTrue(jnp.all(jnp.isfinite(state.u_d)),
tests/unit/test_fv3_audit_harness.py:458:                        "u_d has non-finite values after 100 steps")
tests/unit/test_fv3_audit_harness.py:459:        self.assertTrue(jnp.all(jnp.isfinite(state.v_d)),
tests/unit/test_fv3_audit_harness.py:460:                        "v_d has non-finite values after 100 steps")
tests/unit/test_fv3_audit_harness.py:473:            div_damp=0.0, hyperdiff_coeff=0.0,
tests/unit/test_fv3_audit_harness.py:510:        # Light hyperdiffusion for multi-day stability (standard practice)
tests/unit/test_fv3_audit_harness.py:514:            hyperdiff_coeff=dx_min**4 / (86400.0 * 10),
tests/unit/test_fv3_audit_harness.py:547:        The edge-midpoint FV3 model at C16 with light hyperdiffusion
tests/unit/test_fv3_audit_harness.py:618:            hyperdiff_coeff=dx_min**4 / (86400.0 * 10),
tests/unit/test_fv3_audit_harness.py:633:        self.assertTrue(jnp.all(jnp.isfinite(state.h)),
tests/unit/test_fv3_audit_harness.py:634:                        "h has non-finite values after 5-day TC5")
tests/unit/test_fv3_audit_harness.py:635:        self.assertTrue(jnp.all(jnp.isfinite(state.u_d)),
tests/unit/test_fv3_audit_harness.py:636:                        "u_d has non-finite values after 5-day TC5")
tests/unit/test_fv3_audit_harness.py:637:        self.assertTrue(jnp.all(jnp.isfinite(state.v_d)),
tests/unit/test_fv3_audit_harness.py:638:                        "v_d has non-finite values after 5-day TC5")
tests/unit/test_fv3_audit_harness.py:661:            hyperdiff_coeff=dx_min**4 / (86400.0 * 10),
tests/unit/test_williamson2_cdgrid.py:84:            hyperdiff_coeff=dx_min ** 4 / (86400.0 * 10),
tests/unit/test_williamson2_cdgrid.py:148:        self.assertTrue(jnp.all(jnp.isfinite(state.h)),
tests/unit/test_williamson2_cdgrid.py:149:                        "Height field has non-finite values")
tests/unit/test_williamson2_cdgrid.py:150:        self.assertTrue(jnp.all(jnp.isfinite(state.u_d)),
tests/unit/test_williamson2_cdgrid.py:151:                        "u_d has non-finite values")
tests/unit/test_williamson2_cdgrid.py:197:        self.assertTrue(jnp.all(jnp.isfinite(state.h)),
tests/unit/test_williamson2_cdgrid.py:199:        self.assertTrue(jnp.all(jnp.isfinite(state.u_d)),
tests/unit/test_williamson2_cdgrid.py:201:        self.assertTrue(jnp.all(jnp.isfinite(state.v_d)),
tests/unit/test_coupled_esm.py:81:        # All fields finite
tests/unit/test_coupled_esm.py:82:        self.assertTrue(jnp.all(jnp.isfinite(driver.state.T.data)))
tests/unit/test_coupled_esm.py:83:        self.assertTrue(jnp.all(jnp.isfinite(driver.ocean_state.T_sfc.data)))
tests/unit/test_coupled_esm.py:163:        # Should be finite and non-zero (net flux is non-zero)
tests/unit/test_coupled_esm.py:164:        self.assertTrue(jnp.all(jnp.isfinite(dE)))
tests/unit/test_coupled_esm.py:179:    def test_carbon_presets_have_differland(self):
tests/unit/test_coupled_esm.py:185:            self.assertEqual(cfg.carbon_land, "differland")
tests/ocean/unit/test_barotropic_implicit_mpas.py:155:    assert bool(jnp.all(jnp.isfinite(state_new.eta.data)))
tests/ocean/unit/test_barotropic_implicit_mpas.py:156:    assert bool(jnp.all(jnp.isfinite(state_new.u.data)))
tests/ocean/unit/test_barotropic_implicit_mpas.py:157:    # Rest state has zero forcing and zero gradient → stays put modulo
tests/ocean/unit/test_barotropic_implicit_mpas.py:163:def test_implicit_solver_finite_under_wind(state, mesh, z_coord):
tests/ocean/unit/test_barotropic_implicit_mpas.py:164:    """Implicit solver produces finite state after a 5-day cosine-
tests/ocean/unit/test_barotropic_implicit_mpas.py:178:    assert bool(jnp.all(jnp.isfinite(s.eta.data)))
tests/ocean/unit/test_barotropic_implicit_mpas.py:179:    assert bool(jnp.all(jnp.isfinite(s.u.data)))
tests/ocean/unit/test_barotropic_implicit_mpas.py:219:def test_implicit_solver_grad_smoke(state, mesh, z_coord):
tests/ocean/unit/test_barotropic_implicit_mpas.py:220:    """``jax.grad`` through the implicit MPAS solver returns finite,
tests/ocean/unit/test_barotropic_implicit_mpas.py:240:    g = jax.grad(loss)(eta0)
tests/ocean/unit/test_barotropic_implicit_mpas.py:241:    assert bool(jnp.all(jnp.isfinite(g))), "jax.grad produced NaN/Inf"
tests/ocean/unit/test_barotropic_implicit_mpas.py:243:        "jax.grad returned exactly zero gradient"
tests/ocean/unit/test_barotropic_implicit_mpas.py:301:        f"magnitude difference)"
tests/ocean/unit/test_barotropic_implicit_mpas.py:307:    weak F_slow_eta returns finite (eta, u_bar, Hu_avg)."""
tests/ocean/unit/test_barotropic_implicit_mpas.py:321:    assert bool(jnp.all(jnp.isfinite(eta_new)))
tests/ocean/unit/test_barotropic_implicit_mpas.py:322:    assert bool(jnp.all(jnp.isfinite(u_bar_new)))
tests/ocean/unit/test_barotropic_implicit_mpas.py:323:    assert bool(jnp.all(jnp.isfinite(Hu_avg)))
tests/unit/test_regional_voronoi.py:131:    def test_finite_geometry(self, regional_mesh):
tests/unit/test_regional_voronoi.py:132:        """All geometry arrays should be finite (no NaN or Inf)."""
tests/unit/test_regional_voronoi.py:136:            assert jnp.all(jnp.isfinite(arr)), f"{name} has non-finite values"
tests/unit/test_regional_voronoi.py:170:    def test_gradient_of_constant(self, regional_mesh):
tests/unit/test_regional_voronoi.py:172:        from legoesm.core.operators_voronoi import gradient_edge
tests/unit/test_regional_voronoi.py:175:        grad = gradient_edge(phi, regional_mesh)
tests/unit/test_regional_voronoi.py:176:        max_grad = float(jnp.max(jnp.abs(grad)))
tests/unit/test_regional_voronoi.py:177:        assert max_grad < 1e-10, (
tests/unit/test_regional_voronoi.py:178:            f"Gradient of constant field should be zero, got max={max_grad:.2e}"
tests/unit/test_regional_voronoi.py:234:            barotropic_diffusion_alpha=0.0,
tests/unit/test_regional_voronoi.py:266:        assert jnp.all(jnp.isfinite(eta)), "eta contains NaN/Inf"
tests/unit/test_regional_voronoi.py:267:        assert jnp.all(jnp.isfinite(u_bar)), "u_bar contains NaN/Inf"
tests/unit/test_regional_voronoi.py:365:        # should be finite and non-NaN everywhere and the SIGN should be
tests/unit/test_regional_voronoi.py:367:        assert np.all(np.isfinite(div)), "divergence contains NaN/Inf"
tests/unit/test_regional_voronoi.py:410:        assert jnp.all(jnp.isfinite(s1.u.data)), "u blew up in 1 step"
tests/unit/test_regional_voronoi.py:411:        assert jnp.all(jnp.isfinite(s1.T.data)), "T blew up in 1 step"
tests/unit/test_regional_voronoi.py:412:        assert jnp.all(jnp.isfinite(s1.eta.data)), "eta blew up in 1 step"
tests/unit/test_greens_function.py:4:  - Hyperdiffusion impulse response is smooth and monotonically decaying
tests/unit/test_greens_function.py:44:# 2b) Hyperdiffusion impulse response
tests/unit/test_greens_function.py:47:class TestHyperdiffusionImpulse:
tests/unit/test_greens_function.py:48:    """Apply a point perturbation, step with pure hyperdiffusion,
tests/unit/test_greens_function.py:51:    def test_hyperdiff_impulse_cubesphere(self, cs_grid):
tests/unit/test_greens_function.py:53:        decrease monotonically under hyperdiffusion.
tests/unit/test_greens_function.py:59:        from legoesm.core.operators import hyperdiffusion
tests/unit/test_greens_function.py:73:            tendency = hyperdiffusion(field, cs_grid, nu)
tests/unit/test_greens_function.py:81:        # All values should be finite
tests/unit/test_greens_function.py:82:        assert jnp.all(jnp.isfinite(field.data))
tests/unit/test_greens_function.py:84:    def test_hyperdiff_impulse_latlon(self, ll_grid):
tests/unit/test_greens_function.py:86:        from legoesm.core.operators_latlon import hyperdiffusion
tests/unit/test_greens_function.py:98:            tendency = hyperdiffusion(field, ll_grid, nu)
tests/unit/test_greens_function.py:105:        assert jnp.all(jnp.isfinite(field.data))
tests/unit/test_greens_function.py:116:    def test_gravity_wave_speed_mpas(self, voronoi_mesh):
tests/unit/test_greens_function.py:158:        assert jnp.all(jnp.isfinite(state.h.data))
tests/unit/test_greens_function.py:169:    def test_hyperdiff_face_independence(self, cs_grid):
tests/unit/test_greens_function.py:171:        from legoesm.core.operators import hyperdiffusion
tests/unit/test_greens_function.py:184:                tendency = hyperdiffusion(field, cs_grid, nu)
tests/unit/test_greens_function.py:194:                f"Face {i} L2 = {l2:.6e} differs from mean {mean_l2:.6e} by {rel:.3e}"
tests/unit/test_greens_function.py:207:        """Iteratively smooth a delta function with Laplacian diffusion.
tests/unit/test_greens_function.py:209:        and is finite everywhere."""
tests/unit/test_greens_function.py:218:        dt_diff = 0.1 * dx_min**2 / nu
tests/unit/test_greens_function.py:222:            field = field + dt_diff * nu * lap
tests/unit/test_greens_function.py:224:        # Response should be finite and smooth
tests/unit/test_greens_function.py:225:        assert jnp.all(jnp.isfinite(field)), "Non-finite values after diffusion"
tests/unit/test_greens_function.py:228:        # The field should be mostly non-negative (diffusion of positive source)
tests/unit/test_greens_function.py:234:# 2f) High-wavenumber damping verification (hyperdiffusion spectral decay)
tests/unit/test_greens_function.py:237:class TestHyperdiffusionSpectralDecay:
tests/unit/test_greens_function.py:238:    """Verify that spectral hyperdiffusion damps high-k modes more
tests/unit/test_greens_function.py:243:        apply spectral hyperdiffusion, verify high-k is damped more."""
tests/unit/test_greens_function.py:246:            spectral_hyperdiffusion,
tests/unit/test_greens_function.py:274:        # Apply hyperdiffusion stepping: d(coeffs)/dt = hyperdiff_tendency
tests/unit/test_greens_function.py:280:            tend = spectral_hyperdiffusion(grid, coeffs, nu, order=2)
tests/unit/test_greens_function.py:308:    def test_gravity_wave_cdgrid(self, cs_grid):
tests/unit/test_greens_function.py:338:            hyperdiff_coeff=0.0,
tests/unit/test_greens_function.py:355:        assert jnp.all(jnp.isfinite(state.h)), "Non-finite h"
tests/unit/test_greens_function.py:366:    def test_gravity_wave_spectral(self):
tests/unit/test_greens_function.py:408:            hyperdiff_coeff=0.0,
tests/unit/test_greens_function.py:429:        assert jnp.all(jnp.isfinite(phi_grid)), "Non-finite phi"
tests/unit/test_greens_function.py:433:# 2i) Voronoi hyperdiffusion impulse response
tests/unit/test_greens_function.py:437:    """Laplacian diffusion impulse test on MPAS Voronoi mesh.
tests/unit/test_greens_function.py:438:    Verify energy decreases and field stays finite."""
tests/unit/test_greens_function.py:441:        """A point impulse on the MPAS mesh should be damped by del2 diffusion."""
tests/unit/test_greens_function.py:469:        assert jnp.all(jnp.isfinite(field))
tests/unit/test_dcmip_transport.py:94:    def test_wind_finite(self, grid, sigma_coord):
tests/unit/test_dcmip_transport.py:95:        """Wind field is finite at all times."""
tests/unit/test_dcmip_transport.py:99:            assert jnp.all(jnp.isfinite(u)), f"u not finite at t={t_days} days"
tests/unit/test_dcmip_transport.py:100:            assert jnp.all(jnp.isfinite(v)), f"v not finite at t={t_days} days"
tests/unit/test_dcmip_transport.py:101:            assert jnp.all(jnp.isfinite(sigma_dot)), f"sigma_dot not finite at t={t_days} days"
tests/unit/test_dcmip_transport.py:116:        assert jnp.all(jnp.isfinite(state.tracers.data))
tests/unit/test_dcmip_transport.py:146:    def test_wind_finite(self, grid, sigma_coord):
tests/unit/test_dcmip_transport.py:147:        """Wind field is finite at all times."""
tests/unit/test_dcmip_transport.py:151:            assert jnp.all(jnp.isfinite(u))
tests/unit/test_dcmip_transport.py:152:            assert jnp.all(jnp.isfinite(v))
tests/unit/test_dcmip_transport.py:153:            assert jnp.all(jnp.isfinite(sigma_dot))
tests/unit/test_dcmip_transport.py:169:        assert jnp.all(jnp.isfinite(state.tracers.data))
tests/unit/test_dcmip_transport.py:239:        assert jnp.all(jnp.isfinite(state.tracers.data))
tests/unit/test_dcmip_transport.py:265:    def test_norms_positive_for_different_states(self, grid, sigma_coord):
tests/unit/test_dcmip_transport.py:266:        """Error norms are positive for different states."""
tests/unit/test_runtime_bootstrap.py:234:        # Simulate multi-node: ranks on different hosts
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:90:    def test_tendencies_finite(self, grid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:93:        assert jnp.all(jnp.isfinite(du))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:94:        assert jnp.all(jnp.isfinite(dv))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:95:        assert jnp.all(jnp.isfinite(dT))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:96:        assert jnp.all(jnp.isfinite(dps))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:133:        # Sloping surface pressure: east-west gradient
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:148:        (grad(ln p_s) = 0), so dT/dt comes only from advection+omega terms.
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:184:    def test_step_finite(self, model, grid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:187:        assert jnp.all(jnp.isfinite(state_new.u))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:188:        assert jnp.all(jnp.isfinite(state_new.v))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:189:        assert jnp.all(jnp.isfinite(state_new.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:190:        assert jnp.all(jnp.isfinite(state_new.p_s))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:196:        assert jnp.all(jnp.isfinite(state_new.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:197:        assert jnp.all(jnp.isfinite(state_new.p_s))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:216:        assert jnp.all(jnp.isfinite(state.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:231:        assert jnp.all(jnp.isfinite(state.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:346:        assert jnp.all(jnp.isfinite(state_new.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:347:        assert jnp.all(jnp.isfinite(state_new.p_s))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:359:        assert jnp.all(jnp.isfinite(state_new.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:360:        assert jnp.all(jnp.isfinite(state_new.p_s))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:386:        assert jnp.all(jnp.isfinite(state_new.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:410:        assert jnp.all(jnp.isfinite(hs_new.T.data))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:411:        assert jnp.all(jnp.isfinite(hs_new.p_s.data))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:426:        assert jnp.all(jnp.isfinite(hs_new.T.data))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:438:        assert jnp.all(jnp.isfinite(state.T.data))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:554:            f"T diverged: max diff = "
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:558:            f"p_s diverged: max diff = "
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:572:        # Create a completely different state (restart)
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:618:        assert jnp.all(jnp.isfinite(du))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:628:        assert jnp.all(jnp.isfinite(state_new.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:629:        assert jnp.all(jnp.isfinite(state_new.p_s))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:675:        assert jnp.all(jnp.isfinite(state.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:676:        assert jnp.all(jnp.isfinite(state.p_s))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:690:        assert jnp.all(jnp.isfinite(state_new.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:697:        assert jnp.all(jnp.isfinite(state_new.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:724:        """Cell-centered gradient fallback should be stable for 50 steps."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:736:        assert jnp.all(jnp.isfinite(state.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:737:        assert jnp.all(jnp.isfinite(state.p_s))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:754:        assert jnp.all(jnp.isfinite(state.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:784:        assert jnp.all(jnp.isfinite(state.tracers["q_v"]))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:935:        assert jnp.all(jnp.isfinite(state.T))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:936:        assert jnp.all(jnp.isfinite(state.p_s))
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:949:        """With uniform wind + sloping p_s, flux-form dp_s/dt must differ
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:961:        # Sloping p_s with large gradient to amplify the difference
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:976:        assert jnp.all(jnp.isfinite(dps))
tests/unit/test_physics_surface_models.py:84:    def test_all_outputs_finite(self):
tests/unit/test_physics_surface_models.py:85:        """All output fields should be finite."""
tests/unit/test_physics_surface_models.py:87:        assert jnp.all(jnp.isfinite(state.T_soil.data)), "T_soil NaN"
tests/unit/test_physics_surface_models.py:88:        assert jnp.all(jnp.isfinite(state.W_bucket.data)), "W_bucket NaN"
tests/unit/test_physics_surface_models.py:89:        assert jnp.all(jnp.isfinite(resp.shflx)), "SH NaN"
tests/unit/test_physics_surface_models.py:90:        assert jnp.all(jnp.isfinite(resp.lhflx)), "LH NaN"
tests/unit/test_physics_surface_models.py:168:    def test_all_outputs_finite(self):
tests/unit/test_physics_surface_models.py:173:        assert jnp.all(jnp.isfinite(state.h_ice.data)), "h_ice NaN"
tests/unit/test_physics_surface_models.py:174:        assert jnp.all(jnp.isfinite(state.T_ice.data)), "T_ice NaN"
tests/unit/test_physics_surface_models.py:175:        assert jnp.all(jnp.isfinite(resp.shflx)), "SH NaN"
tests/unit/test_physics_surface_models.py:217:    def test_all_outputs_finite(self):
tests/unit/test_physics_surface_models.py:219:        assert jnp.all(jnp.isfinite(state.T_epi.data)), "T_epi NaN"
tests/unit/test_physics_surface_models.py:220:        assert jnp.all(jnp.isfinite(state.T_hypo.data)), "T_hypo NaN"
tests/unit/test_physics_surface_models.py:221:        assert jnp.all(jnp.isfinite(resp.shflx)), "SH NaN"
tests/unit/test_physics_smoke.py:4:without errors, and produces finite outputs with correct shapes.
tests/unit/test_physics_smoke.py:53:def check_tendencies_finite(tend, name):
tests/unit/test_physics_smoke.py:54:    """Assert all tendency fields are finite."""
tests/unit/test_physics_smoke.py:62:                assert jnp.all(jnp.isfinite(data)), f"{name}.{fname}[{k}] has NaN/Inf"
tests/unit/test_physics_smoke.py:64:            assert jnp.all(jnp.isfinite(val.data)), f"{name}.{fname} has NaN/Inf"
tests/unit/test_physics_smoke.py:66:            assert jnp.all(jnp.isfinite(val)), f"{name}.{fname} has NaN/Inf"
tests/unit/test_physics_smoke.py:81:    """Each radiation scheme produces finite tendencies with correct shape."""
tests/unit/test_physics_smoke.py:90:    check_tendencies_finite(tend, f"radiation({scheme})")
tests/unit/test_physics_smoke.py:103:    """Each convection scheme produces finite tendencies."""
tests/unit/test_physics_smoke.py:111:    check_tendencies_finite(tend, f"convection({scheme})")
tests/unit/test_physics_smoke.py:122:    """Each microphysics scheme produces finite tendencies."""
tests/unit/test_physics_smoke.py:140:    check_tendencies_finite(tend, f"microphysics({scheme})")
tests/unit/test_physics_smoke.py:151:    """Each turbulence scheme produces finite tendencies."""
tests/unit/test_physics_smoke.py:159:    check_tendencies_finite(tend, f"turbulence({scheme})")
tests/unit/test_physics_smoke.py:171:def test_gwd_smoke(scheme):
tests/unit/test_physics_smoke.py:172:    """Each GWD scheme produces finite tendencies."""
tests/unit/test_physics_smoke.py:173:    from legoesm.atmosphere.physics.gravity_wave_drag.integration import make_gwd_physics
tests/unit/test_physics_smoke.py:174:    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/unit/test_physics_smoke.py:178:    gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
tests/unit/test_physics_smoke.py:179:    tend, prog = gwd_fn(state, grid, sigma)
tests/unit/test_physics_smoke.py:180:    check_tendencies_finite(tend, f"gwd({scheme})")
tests/unit/test_physics_smoke.py:181:    check_shape(tend.dT_dt, (6, 8, 8, 10), f"gwd({scheme}).dT_dt")
tests/unit/test_physics_smoke.py:189:    """Default PhysicsConfig produces finite tendencies."""
tests/unit/test_physics_smoke.py:197:    check_tendencies_finite(tend, "combined_defaults")
tests/unit/test_physics_smoke.py:201:    """All five physics modules active simultaneously produce finite output."""
tests/unit/test_physics_smoke.py:207:    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/unit/test_physics_smoke.py:215:        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
tests/unit/test_physics_smoke.py:220:    check_tendencies_finite(tend, "combined_full_stack")
tests/unit/test_spectral_pe_training_tracers.py:17:* ``jax.grad`` flows through a tracer-carrying rollout into the
tests/unit/test_spectral_pe_training_tracers.py:141:        """Loss differs when q_v values differ between predicted and target."""
tests/unit/test_spectral_pe_training_tracers.py:159:        loss_diff = spectral_state_vs_carry_loss(
tests/unit/test_spectral_pe_training_tracers.py:162:        # The q_v difference is 0.005 → mean squared error ~ 2.5e-5.
tests/unit/test_spectral_pe_training_tracers.py:163:        assert float(loss_diff) > float(loss_self) + 1e-7, (
tests/unit/test_spectral_pe_training_tracers.py:165:            f"diff={float(loss_diff)}"
tests/unit/test_spectral_pe_training_tracers.py:188:        assert bool(jnp.isfinite(loss))
tests/unit/test_spectral_pe_training_tracers.py:211:            hyperdiff_coeff=1e14,
tests/unit/test_spectral_pe_training_tracers.py:220:        assert bool(jnp.all(jnp.isfinite(result.tracers["q_v"].data)))
tests/unit/test_spectral_pe_training_tracers.py:260:            hyperdiff_coeff=1e14,
tests/unit/test_spectral_pe_training_tracers.py:298:    def test_grad_through_rollout_with_tracers(self):
tests/unit/test_spectral_pe_training_tracers.py:316:            hyperdiff_coeff=1e14,
tests/unit/test_spectral_pe_training_tracers.py:331:        loss, grads = eqx.filter_value_and_grad(loss_fn)(sfno)
tests/unit/test_spectral_pe_training_tracers.py:332:        assert bool(jnp.isfinite(loss))
tests/unit/test_spectral_pe_training_tracers.py:333:        # All gradients finite.
tests/unit/test_spectral_pe_training_tracers.py:334:        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
tests/unit/test_spectral_pe_training_tracers.py:335:        all_finite = all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves)
tests/unit/test_spectral_pe_training_tracers.py:336:        assert all_finite, "Some gradients are NaN/Inf"
tests/unit/test_spectral_pe_training_tracers.py:337:        # At least one gradient is non-zero (signal flows).
tests/unit/test_spectral_pe_training_tracers.py:338:        has_nonzero = any(bool(jnp.any(g != 0)) for g in grad_leaves)
tests/unit/test_spectral_pe_training_tracers.py:339:        assert has_nonzero, "All gradients are zero — no signal"
tests/unit/test_spectral_pe_training_tracers.py:419:        assert bool(jnp.all(jnp.isfinite(qv)))
tests/unit/test_spectral_pe_training_tracers.py:463:# NMC end-to-end: build two fake forecasts, diff, check moisture flows
tests/unit/test_spectral_pe_training_tracers.py:597:        assert bool(jnp.all(jnp.isfinite(dq_v)))
tests/unit/test_spectral_pe_training_tracers.py:608:        """Different q_v in state → different MLP output (proves the
tests/unit/test_spectral_pe_training_tracers.py:628:        diff = float(jnp.max(jnp.abs(out_a.T_hat.data - out_b.T_hat.data)))
tests/unit/test_spectral_pe_training_tracers.py:629:        assert diff > 1e-12, (
tests/unit/test_spectral_pe_training_tracers.py:631:            f"dT_hat for q_v=0.001 vs q_v=0.020 (diff {diff})"
tests/unit/test_spectral_pe_training_tracers.py:660:    def test_hydrostatic_diff_carries_tracer_differences(self):
tests/unit/test_spectral_pe_training_tracers.py:661:        """``_hydrostatic_diff`` returns per-key tracer differences when
tests/unit/test_spectral_pe_training_tracers.py:666:            _hydrostatic_diff,
tests/unit/test_spectral_pe_training_tracers.py:679:        diff = _hydrostatic_diff(hs_a, hs_b)
tests/unit/test_spectral_pe_training_tracers.py:680:        assert diff.tracers is not None
tests/unit/test_spectral_pe_training_tracers.py:681:        assert "q_v" in diff.tracers
tests/unit/test_spectral_pe_training_tracers.py:684:        dqv = diff.tracers["q_v"].data
tests/unit/test_spectral_pe_training_tracers.py:691:    def test_hydrostatic_diff_drops_tracers_when_either_side_missing(self):
tests/unit/test_spectral_pe_training_tracers.py:692:        """When either operand has tracers=None, diff also has None
tests/unit/test_spectral_pe_training_tracers.py:696:            _hydrostatic_diff,
tests/unit/test_spectral_pe_training_tracers.py:710:        diff = _hydrostatic_diff(hs_a, hs_b)
tests/unit/test_spectral_pe_training_tracers.py:711:        assert diff.tracers is None
tests/unit/test_spectral_pe_training_tracers.py:713:    def test_hydrostatic_diff_intersects_keys(self):
tests/unit/test_spectral_pe_training_tracers.py:715:        tracer key sets, the diff returns only the intersection."""
tests/unit/test_spectral_pe_training_tracers.py:719:            _hydrostatic_diff,
tests/unit/test_spectral_pe_training_tracers.py:733:        diff = _hydrostatic_diff(hs_a, hs_b)
tests/unit/test_spectral_pe_training_tracers.py:734:        assert diff.tracers is not None
tests/unit/test_spectral_pe_training_tracers.py:735:        assert set(diff.tracers.keys()) == {"q_v"}
tests/unit/test_spectral_pe_training_tracers.py:739:        dycore step → back to grid → diff vs initial state.
tests/unit/test_spectral_pe_training_tracers.py:743:        zero tracer-difference field in the NMC error proxy.
tests/unit/test_spectral_pe_training_tracers.py:748:            _hydrostatic_diff,
tests/unit/test_spectral_pe_training_tracers.py:765:            hyperdiff_coeff=1.0 / (4.0 * 3600.0 * eig_max ** 2),
tests/unit/test_spectral_pe_training_tracers.py:785:        err = _hydrostatic_diff(f_phys, ic_phys)
tests/unit/test_spectral_pe_training_tracers.py:787:        # q_v difference field is finite (the NMC error logger reads
tests/unit/test_spectral_pe_training_tracers.py:790:        assert bool(jnp.all(jnp.isfinite(dqv)))
tests/unit/test_land_ice_sea_ice_thermo.py:68:    def test_returns_finite(self):
tests/unit/test_land_ice_sea_ice_thermo.py:73:            assert jnp.all(jnp.isfinite(getattr(new_state, name).data)), f"{name} non-finite"
tests/unit/test_land_ice_sea_ice_thermo.py:75:            assert jnp.all(jnp.isfinite(getattr(resp, name))), f"TileResponse.{name} non-finite"
tests/unit/test_land_ice_sea_ice_thermo.py:158:        assert jnp.all(jnp.isfinite(state.T_ice.data))
tests/unit/test_land_ice_sea_ice_thermo.py:159:        assert jnp.all(jnp.isfinite(state.h_ice.data))
tests/unit/test_operators_fv.py:1:"""Unit tests for FV3-style finite-volume transport operators."""
tests/unit/test_operators_fv.py:74:        assert jnp.all(jnp.isfinite(q_L_lim))
tests/unit/test_operators_fv.py:75:        assert jnp.all(jnp.isfinite(q_R_lim))
tests/unit/test_operators_fv.py:113:    def test_all_finite(self, grid):
tests/unit/test_operators_fv.py:114:        """All outputs should be finite."""
tests/unit/test_operators_fv.py:121:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_operators_fv.py:123:    def test_differentiable(self, grid):
tests/unit/test_operators_fv.py:124:        """Should be differentiable with jax.grad."""
tests/unit/test_operators_fv.py:134:        grads = jax.grad(loss)(q)
tests/unit/test_operators_fv.py:135:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_operators_fv.py:164:    def test_output_shape_and_finite(self, grid):
tests/unit/test_operators_fv.py:165:        """Output should be correct shape and finite."""
tests/unit/test_operators_fv.py:174:        assert jnp.all(jnp.isfinite(dq))
tests/unit/test_operators_fv.py:176:    def test_differentiable(self, grid):
tests/unit/test_operators_fv.py:177:        """Should be differentiable with jax.grad."""
tests/unit/test_operators_fv.py:187:        grads = jax.grad(loss)(q)
tests/unit/test_operators_fv.py:188:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_operators_fv.py:220:        assert jnp.all(jnp.isfinite(q))
tests/ocean/unit/test_gm_redi_mpas.py:68:def _unit_jacobian(mesh) -> jnp.ndarray:
tests/ocean/unit/test_gm_redi_mpas.py:92:    jac = _unit_jacobian(mesh)
tests/ocean/unit/test_gm_redi_mpas.py:94:    S_n, taper = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
tests/ocean/unit/test_gm_redi_mpas.py:104:    """ρ = ρ₀ + α·(-z) (depth-only) ⇒ no horizontal gradient ⇒ S_n = 0."""
tests/ocean/unit/test_gm_redi_mpas.py:111:    jac = _unit_jacobian(mesh)
tests/ocean/unit/test_gm_redi_mpas.py:113:    S_n, taper = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
tests/ocean/unit/test_gm_redi_mpas.py:118:def test_slopes_meridional_gradient_matches_analytic(mesh, z_coord, cfg):
tests/ocean/unit/test_gm_redi_mpas.py:143:    jac = _unit_jacobian(mesh)
tests/ocean/unit/test_gm_redi_mpas.py:145:    S_n, taper = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
tests/ocean/unit/test_gm_redi_mpas.py:150:    # Same value at every interface (vertical gradient is constant).
tests/ocean/unit/test_gm_redi_mpas.py:156:    # gradient samples ρ at neighbouring cells whose (lat, sin(lat))
tests/ocean/unit/test_gm_redi_mpas.py:157:    # values differ by O(dcEdge / R_earth).  The level-2 mesh has
tests/ocean/unit/test_gm_redi_mpas.py:174:    """Edges on the land/ocean coastline see a finite slope only because
tests/ocean/unit/test_gm_redi_mpas.py:184:    # Land cell with sentinel value that would blow up the gradient.
tests/ocean/unit/test_gm_redi_mpas.py:187:    jac = _unit_jacobian(mesh)
tests/ocean/unit/test_gm_redi_mpas.py:189:    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
tests/ocean/unit/test_gm_redi_mpas.py:192:    # everywhere ⇒ zero slopes everywhere, including land-adjacent edges.
tests/ocean/unit/test_gm_redi_mpas.py:227:    jac = _unit_jacobian(mesh)
tests/ocean/unit/test_gm_redi_mpas.py:231:    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
tests/ocean/unit/test_gm_redi_mpas.py:235:        q, S_n, mask, em, z_coord, jac, mesh,
tests/ocean/unit/test_gm_redi_mpas.py:265:    jac = _unit_jacobian(mesh)
tests/ocean/unit/test_gm_redi_mpas.py:268:    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
tests/ocean/unit/test_gm_redi_mpas.py:273:        q, S_n, mask, em, z_coord, jac, mesh,
tests/ocean/unit/test_gm_redi_mpas.py:291:def test_centered_pure_horizontal_q_diffuses(mesh, z_coord, cfg):
tests/ocean/unit/test_gm_redi_mpas.py:292:    """Diagonal Redi flux is alive: a horizontal q-gradient with no
tests/ocean/unit/test_gm_redi_mpas.py:294:    has the sign of horizontal Laplacian diffusion."""
tests/ocean/unit/test_gm_redi_mpas.py:297:    jac = _unit_jacobian(mesh)
tests/ocean/unit/test_gm_redi_mpas.py:302:    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
tests/ocean/unit/test_gm_redi_mpas.py:305:    # Tracer with sin(lat) gradient.
tests/ocean/unit/test_gm_redi_mpas.py:310:        q, S_n, mask, em, z_coord, jac, mesh,
tests/ocean/unit/test_gm_redi_mpas.py:314:    # The diffusive tendency on a sin(lat) field has the opposite sign
tests/ocean/unit/test_gm_redi_mpas.py:323:        f"diffusive tendency should oppose the field, got corr={correlation:.3g}"
tests/ocean/unit/test_gm_redi_mpas.py:342:    jac = _unit_jacobian(mesh)
tests/ocean/unit/test_gm_redi_mpas.py:345:    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
tests/ocean/unit/test_gm_redi_mpas.py:351:        q, S_n, mask, em, z_coord, jac, mesh,
tests/ocean/unit/test_gm_redi_mpas.py:355:    dz_actual = z_coord.dz_ref * jac[:, None]                  # (nCells, nlev)
tests/ocean/unit/test_gm_redi_mpas.py:375:    jac = _unit_jacobian(mesh)
tests/ocean/unit/test_gm_redi_mpas.py:380:    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
tests/ocean/unit/test_gm_redi_mpas.py:387:        q, S_n, mask, em, z_coord, jac, mesh,
tests/ocean/unit/test_gm_redi_mpas.py:401:    (which doubles the meridional density gradient and hence |S|)
tests/ocean/unit/test_gm_redi_mpas.py:407:    jac = _unit_jacobian(mesh)
tests/ocean/unit/test_gm_redi_mpas.py:423:        S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
tests/ocean/unit/test_gm_redi_mpas.py:425:            rho, S_n, mesh, z_coord, jac, f_cor, cfg_v,
tests/ocean/unit/test_gm_redi_mpas.py:448:    (dT_dt, dS_dt) with the expected shapes; output is finite."""
tests/ocean/unit/test_gm_redi_mpas.py:452:    # Stable stratification + mild horizontal T gradient.
tests/ocean/unit/test_gm_redi_mpas.py:474:    assert np.all(np.isfinite(np.asarray(dT_dt)))
tests/ocean/unit/test_gm_redi_mpas.py:475:    assert np.all(np.isfinite(np.asarray(dS_dt)))
tests/ocean/unit/test_gm_redi_mpas.py:476:    # S is uniform ⇒ no salt tendency from gradients.
tests/ocean/unit/test_gm_redi_mpas.py:514:    from legoesm.ocean.vertical import compute_ocean_jacobian
tests/ocean/unit/test_gm_redi_mpas.py:518:    jac = compute_ocean_jacobian(eta, H_bathy, z_coord)
tests/ocean/unit/test_gm_redi_mpas.py:525:    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
tests/ocean/unit/test_gm_redi_mpas.py:527:    kappa_GM = _visbeck_kappa_gm_mpas(rho, S_n, mesh, z_coord, jac, f_cor, visbeck)
tests/ocean/unit/test_gm_redi_mpas.py:529:        T, S_n, mask, em, z_coord, jac, mesh, kappa_GM, cfg.kappa_Redi,
tests/ocean/unit/test_gm_redi_mpas.py:565:    and complete one step with finite output that differs from the
tests/ocean/unit/test_gm_redi_mpas.py:582:    # Strong horizontal T gradient so GM/Redi has a clearly resolvable
tests/ocean/unit/test_gm_redi_mpas.py:611:    assert np.all(np.isfinite(np.asarray(state_off.T.data)))
tests/ocean/unit/test_gm_redi_mpas.py:612:    assert np.all(np.isfinite(np.asarray(state_on.T.data)))
tests/ocean/unit/test_gm_redi_mpas.py:614:    # GM/Redi must produce a non-trivial difference: the hook is alive.
tests/ocean/unit/test_gm_redi_mpas.py:647:    assert np.all(np.isfinite(np.asarray(new_state.T.data)))
tests/unit/test_batch_allreduce.py:131:    def test_different_ops(self):
tests/unit/test_batch_allreduce.py:151:        """Values with different dtypes are promoted to common dtype."""
tests/unit/test_state_checkpoint.py:191:        # Create a modified template with different bathymetry
tests/unit/test_state_checkpoint.py:276:        different_config = _MiniConfig(resolution=8, dt=900.0, scheme="rk4")
tests/unit/test_state_checkpoint.py:278:            load_state_checkpoint(path, atm_state, config=different_config)
tests/unit/test_state_checkpoint.py:291:        # Create template with different shape
tests/unit/test_tracer_transport.py:90:    def test_tendency_finite_with_wind(self, grid, sigma_coord):
tests/unit/test_tracer_transport.py:91:        """Tendencies are finite with non-zero wind."""
tests/unit/test_tracer_transport.py:107:        assert jnp.all(jnp.isfinite(tend.tracers.data))
tests/unit/test_tracer_transport.py:113:    def test_single_step_finite(self, grid, sigma_coord):
tests/unit/test_tracer_transport.py:114:        """Single step produces finite results."""
tests/unit/test_tracer_transport.py:118:        assert jnp.all(jnp.isfinite(new_state.tracers.data))
tests/unit/test_tracer_transport.py:119:        assert jnp.all(jnp.isfinite(new_state.time.data))
tests/unit/test_tracer_transport.py:138:        """Works with different numbers of tracers (1, 4, 10)."""
tests/unit/test_tracer_transport.py:144:            assert jnp.all(jnp.isfinite(new_state.tracers.data))
tests/unit/test_tracer_transport.py:146:    def test_hyperdiffusion(self, grid, sigma_coord):
tests/unit/test_tracer_transport.py:147:        """Hyperdiffusion produces different tendency than without."""
tests/unit/test_tracer_transport.py:150:        # Use grid-scale noise to get measurable hyperdiffusion
tests/unit/test_tracer_transport.py:162:            TracerTransportConfig(hyperdiff_coeff=0.0)
tests/unit/test_tracer_transport.py:166:            TracerTransportConfig(hyperdiff_coeff=1e15)
tests/unit/test_tracer_transport.py:168:        # Tendencies should differ for the non-uniform tracer
tests/unit/test_tracer_transport.py:169:        diff = jnp.max(jnp.abs(tend_no.tracers.data - tend_yes.tracers.data))
tests/unit/test_tracer_transport.py:170:        assert diff > 0.0, f"Hyperdiffusion had no effect, diff={float(diff)}"
tests/unit/test_tracer_transport.py:182:    """Test that tracer transport is differentiable."""
tests/unit/test_tracer_transport.py:184:    def test_grad_through_tendencies(self, grid, sigma_coord):
tests/unit/test_tracer_transport.py:185:        """jax.grad works through tendency computation."""
tests/unit/test_tracer_transport.py:198:        grad = jax.grad(loss)(q0)
tests/unit/test_tracer_transport.py:199:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_tracer_transport.py:201:    def test_grad_through_step(self, grid, sigma_coord):
tests/unit/test_tracer_transport.py:202:        """jax.grad works through a full step."""
tests/unit/test_tracer_transport.py:214:        grad = jax.grad(loss)(state.tracers.data)
tests/unit/test_tracer_transport.py:215:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_tracer_transport.py:426:    def test_face_eval_differs_from_cell_avg(self):
tests/unit/test_tracer_transport.py:427:        """A wind that varies sharply in longitude should produce different
tests/unit/test_tracer_transport.py:447:        lon_diff = float(jnp.abs(lon_u[0, 1] - grid.lon[0]))
tests/unit/test_tracer_transport.py:448:        assert lon_diff > 0.01, f"u-face lon not offset: diff={lon_diff}"
tests/unit/test_tracer_transport.py:451:        """When wind varies sharply, face-evaluated transport must differ
tests/unit/test_tracer_transport.py:482:        assert jnp.all(jnp.isfinite(state_new.tracers.data))
tests/unit/test_tracer_transport.py:484:        diff = float(jnp.max(jnp.abs(state_new.tracers.data - q_data)))
tests/unit/test_tracer_transport.py:485:        assert diff > 1e-6, f"Tracer unchanged after transport: max diff = {diff}"
tests/unit/test_surface_exchange.py:46:    assert jnp.all(jnp.isfinite(out.rho_lowest))
tests/unit/test_sfno_s2s.py:263:    assert np.isfinite(out).all()
tests/unit/test_sfno_s2s.py:468:    interior_gaps = np.abs(np.diff(lon_values))
tests/unit/test_sfno_s2s.py:610:    assert np.isfinite(filled).all()
tests/unit/test_sfno_s2s.py:668:    assert np.isfinite(ds["surface_pressure"].values).all()
tests/unit/test_sfno_s2s.py:704:def test_uncoupled_rollout_keeps_fixed_sst_finite_in_forcing_state(tmp_path: Path):
tests/unit/test_sfno_s2s.py:759:    assert np.isfinite(ds["forcing"].values).all()
tests/unit/test_sfno_s2s.py:760:    assert np.isfinite(ds["prediction"].values).all()
tests/unit/test_sfno_s2s.py:762:    assert np.isfinite(ds["sea_surface_temperature"].isel(latitude=1).values).all()
tests/ocean/unit/test_visbeck_gm.py:42:    jacobian = jnp.ones((6, n, n))
tests/ocean/unit/test_visbeck_gm.py:43:    return grid, z_coord, jacobian, shape
tests/ocean/unit/test_visbeck_gm.py:49:    Builds a density field whose mean vertical gradient is set so that
tests/ocean/unit/test_visbeck_gm.py:53:    grid, z_coord, jacobian, shape = _make_setup(n=n, nlev=nlev)
tests/ocean/unit/test_visbeck_gm.py:57:    # Add a meridional gradient proportional to latitude index.  At slope
tests/ocean/unit/test_visbeck_gm.py:58:    # = 1e-4 and dρ/dz ≈ 2/H, horizontal gradient ≈ slope × dρ/dz.
tests/ocean/unit/test_visbeck_gm.py:64:    # Build a density field with a zonal gradient along the first axis of
tests/ocean/unit/test_visbeck_gm.py:67:    grad_h = drho_dh * jnp.arange(n_ax, dtype=jnp.float64)
tests/ocean/unit/test_visbeck_gm.py:69:           + grad_h[None, :, None, None]
tests/ocean/unit/test_visbeck_gm.py:78:    return grid, z_coord, jacobian, shape, u, v, T, S, rho
tests/ocean/unit/test_visbeck_gm.py:125:        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields()
tests/ocean/unit/test_visbeck_gm.py:127:            rho, z_coord, jacobian, grid, GMRediConfig())
tests/ocean/unit/test_visbeck_gm.py:129:            rho, S_x, S_y, z_coord, jacobian,
tests/ocean/unit/test_visbeck_gm.py:131:        assert kappa.shape == jacobian.shape
tests/ocean/unit/test_visbeck_gm.py:134:        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields()
tests/ocean/unit/test_visbeck_gm.py:136:            rho, z_coord, jacobian, grid, GMRediConfig())
tests/ocean/unit/test_visbeck_gm.py:139:            rho, S_x, S_y, z_coord, jacobian,
tests/ocean/unit/test_visbeck_gm.py:146:        grid, z_coord, jacobian, shape = _make_setup()
tests/ocean/unit/test_visbeck_gm.py:156:            rho, S_x, S_y, z_coord, jacobian,
tests/ocean/unit/test_visbeck_gm.py:163:        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields(
tests/ocean/unit/test_visbeck_gm.py:166:            rho, z_coord, jacobian, grid, GMRediConfig())
tests/ocean/unit/test_visbeck_gm.py:172:            rho, S_x, S_y, z_coord, jacobian,
tests/ocean/unit/test_visbeck_gm.py:175:            rho, S_x, S_y, z_coord, jacobian,
tests/ocean/unit/test_visbeck_gm.py:184:        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields(
tests/ocean/unit/test_visbeck_gm.py:187:            rho, z_coord, jacobian, grid, GMRediConfig())
tests/ocean/unit/test_visbeck_gm.py:193:            rho, S_x, S_y, z_coord, jacobian,
tests/ocean/unit/test_visbeck_gm.py:196:            rho, S_x, S_y, z_coord, jacobian,
tests/ocean/unit/test_visbeck_gm.py:211:        grid, z_coord, jacobian, shape = _make_setup(n=4, nlev=10)
tests/ocean/unit/test_visbeck_gm.py:224:            rho, S_x, S_y, z_coord, jacobian,
tests/ocean/unit/test_visbeck_gm.py:244:        grid, z_coord, jacobian, shape = _make_setup(n=4, nlev=10)
tests/ocean/unit/test_visbeck_gm.py:260:            rho, S_x, S_y, z_coord, jacobian,
tests/ocean/unit/test_visbeck_gm.py:263:        # Sanity: value is finite and far below what the RMS-bias form
tests/ocean/unit/test_visbeck_gm.py:269:            rho, z_coord.dz_ref, jacobian, rho_ref=1025.0, g=constants.g)
tests/ocean/unit/test_visbeck_gm.py:271:        dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
tests/ocean/unit/test_visbeck_gm.py:297:    def test_finite_and_differentiable(self):
tests/ocean/unit/test_visbeck_gm.py:298:        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields()
tests/ocean/unit/test_visbeck_gm.py:304:                rho_in, z_coord, jacobian, grid, GMRediConfig())
tests/ocean/unit/test_visbeck_gm.py:306:                rho_in, S_x, S_y, z_coord, jacobian, f_cor, cfg)
tests/ocean/unit/test_visbeck_gm.py:309:        grad = jax.grad(loss)(rho)
tests/ocean/unit/test_visbeck_gm.py:310:        assert jnp.all(jnp.isfinite(grad))
tests/ocean/unit/test_visbeck_gm.py:319:    def test_tendency_shapes_and_finite(self):
tests/ocean/unit/test_visbeck_gm.py:320:        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields()
tests/ocean/unit/test_visbeck_gm.py:323:            u, v, T, S, rho, z_coord, jacobian, grid, cfg)
tests/ocean/unit/test_visbeck_gm.py:326:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/ocean/unit/test_visbeck_gm.py:327:        assert jnp.all(jnp.isfinite(out.dS_dt))
tests/ocean/unit/test_visbeck_gm.py:332:        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields()
tests/ocean/unit/test_visbeck_gm.py:339:            u, v, T, S, rho, z_coord, jacobian, grid, cfg_off)
tests/ocean/unit/test_visbeck_gm.py:341:            u, v, T, S, rho, z_coord, jacobian, grid, cfg_on)
tests/ocean/unit/test_visbeck_gm.py:348:        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields()
tests/ocean/unit/test_visbeck_gm.py:351:            rho, z_coord, jacobian, grid, cfg)
tests/ocean/unit/test_visbeck_gm.py:354:        arr_k = jnp.full(jacobian.shape, scalar_k)
tests/ocean/unit/test_visbeck_gm.py:357:            T, S_x, S_y, z_coord, jacobian, grid, scalar_k, cfg.kappa_Redi)
tests/ocean/unit/test_visbeck_gm.py:359:            T, S_x, S_y, z_coord, jacobian, grid, arr_k, cfg.kappa_Redi)
tests/ocean/unit/test_visbeck_gm.py:363:        grid, z_coord, jacobian, shape, u, v, T, S, rho = _stratified_fields(
tests/ocean/unit/test_visbeck_gm.py:372:            u, v, T, S, rho, z_coord, jacobian, grid, cfg_scalar)
tests/ocean/unit/test_visbeck_gm.py:374:            u, v, T, S, rho, z_coord, jacobian, grid, cfg_visbeck)
tests/ocean/unit/test_visbeck_gm.py:375:        max_diff = float(jnp.max(jnp.abs(o1.dT_dt - o2.dT_dt)))
tests/ocean/unit/test_visbeck_gm.py:376:        assert max_diff > 0.0
tests/atmosphere/hydrostatic/unit/test_turbulence.py:4:- Vertical diffusion: shape, conservation, smoothing, differentiability
tests/atmosphere/hydrostatic/unit/test_turbulence.py:6:- Smagorinsky: output shapes, mixing, differentiability
tests/atmosphere/hydrostatic/unit/test_turbulence.py:7:- Louis: stable vs unstable Ri, shapes, differentiability
tests/atmosphere/hydrostatic/unit/test_turbulence.py:8:- TKE: shear response, minimum TKE, shapes, differentiability
tests/atmosphere/hydrostatic/unit/test_turbulence.py:9:- Integration: hydrostatic/NH shapes, nonzero tendencies, jax.grad, scheme selection
tests/atmosphere/hydrostatic/unit/test_turbulence.py:37:from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
tests/atmosphere/hydrostatic/unit/test_turbulence.py:38:    implicit_vertical_diffusion,
tests/atmosphere/hydrostatic/unit/test_turbulence.py:128:        result = implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, 60.0, sflx)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:142:        result = implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, 300.0, sflx)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:150:    def test_smooths_sharp_gradient(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:162:        result = implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, 60.0, sflx)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:164:        # The max gradient should be reduced
tests/atmosphere/hydrostatic/unit/test_turbulence.py:165:        max_grad_before = float(jnp.max(jnp.abs(jnp.diff(phi, axis=1))))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:166:        max_grad_after = float(jnp.max(jnp.abs(jnp.diff(result, axis=1))))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:167:        assert max_grad_after < max_grad_before
tests/atmosphere/hydrostatic/unit/test_turbulence.py:169:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:170:        """jax.grad should work through the solver."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:179:            result = implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, 60.0, sflx)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:183:        g = jax.grad(loss)(phi)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:184:        assert jnp.all(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:310:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:311:        """jax.grad should work through Smagorinsky turbulence."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:325:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:326:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:327:        assert grad_T.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_turbulence.py:385:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:386:        """jax.grad should work through Louis turbulence."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:400:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:401:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:477:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:478:        """jax.grad should work through TKE turbulence."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:493:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:494:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:610:    def test_grad_through_hydrostatic_turbulence(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:611:        """jax.grad should work through hydrostatic turbulence physics."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:628:        grad_T = jax.grad(loss)(state.T.data)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:629:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:632:        """Different schemes should produce different tendencies."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:660:        # Both nonzero but different
tests/atmosphere/hydrostatic/unit/test_turbulence.py:740:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:741:        """jax.grad should work through HB turbulence."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:755:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:756:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:757:        assert grad_T.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_turbulence.py:759:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:760:        """All outputs should be finite."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:772:        assert jnp.all(jnp.isfinite(out.du_dt))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:773:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:774:        assert jnp.all(jnp.isfinite(out.Km))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:775:        assert jnp.all(jnp.isfinite(out.Kh))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:777:    def test_counter_gradient_effect(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:778:        """Warm surface should activate counter-gradient, changing T tendency."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:784:        # With counter-gradient
tests/atmosphere/hydrostatic/unit/test_turbulence.py:791:        # Without counter-gradient
tests/atmosphere/hydrostatic/unit/test_turbulence.py:798:        # T tendencies should differ when counter-gradient is active
tests/atmosphere/hydrostatic/unit/test_turbulence.py:846:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:847:        """jax.grad should work through YSU turbulence."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:861:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:862:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:863:        assert grad_T.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_turbulence.py:865:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:866:        """All outputs should be finite."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:878:        assert jnp.all(jnp.isfinite(out.du_dt))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:879:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:880:        assert jnp.all(jnp.isfinite(out.Km))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:883:        """YSU with entrainment should differ from zero-entrainment."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:903:        # Tendencies should differ when entrainment is active
tests/atmosphere/hydrostatic/unit/test_turbulence.py:912:    """Tests for EDMF eddy-diffusivity mass-flux turbulence."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:953:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:954:        """jax.grad should work through EDMF turbulence."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:969:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:970:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:971:        assert grad_T.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_turbulence.py:973:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:974:        """All outputs should be finite."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:987:        assert jnp.all(jnp.isfinite(out.du_dt))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:988:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:989:        assert jnp.all(jnp.isfinite(out.Km))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:990:        assert jnp.all(jnp.isfinite(tke_new))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1032:        # T tendencies should differ when MF is active
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1055:        """Ri at the surface level should be near zero (no buoyancy difference)."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1063:    def test_bulk_richardson_finite(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1064:        """Ri should be finite everywhere."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1069:        assert jnp.all(jnp.isfinite(Ri_bulk))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1070:        assert jnp.all(jnp.isfinite(theta_v))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1090:    def test_diagnose_pbl_height_finite(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1091:        """h_pbl should be finite."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1096:        assert jnp.all(jnp.isfinite(h_pbl))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1108:        assert jnp.all(jnp.isfinite(h_pbl))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1149:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1150:        """jax.grad should work through PBL height diagnosis."""
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1158:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1159:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1160:        assert grad_T.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1176:        assert jnp.all(jnp.isfinite(out.h_pbl))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1184:        assert jnp.all(jnp.isfinite(out.h_pbl))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1192:        assert jnp.all(jnp.isfinite(out.h_pbl))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1200:        assert jnp.all(jnp.isfinite(out.h_pbl))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1208:        assert jnp.all(jnp.isfinite(out.h_pbl))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1216:        assert jnp.all(jnp.isfinite(out.h_pbl))
tests/unit/test_diff_data_assimilation.py:7:  7d) Cost function gradient accuracy (Taylor test)
tests/unit/test_diff_data_assimilation.py:21:def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
tests/unit/test_diff_data_assimilation.py:22:    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
tests/unit/test_diff_data_assimilation.py:23:    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
tests/unit/test_diff_data_assimilation.py:53:    def test_round_trip_grad(self):
tests/unit/test_diff_data_assimilation.py:60:        grad = jax.grad(loss)(self.x0)
tests/unit/test_diff_data_assimilation.py:61:        assert_gradient_ok(grad, "Control vector round-trip")
tests/unit/test_diff_data_assimilation.py:70:    def test_direct_obs_grad(self):
tests/unit/test_diff_data_assimilation.py:90:        grad = jax.grad(loss)(state.h.data)
tests/unit/test_diff_data_assimilation.py:91:        assert jnp.all(jnp.isfinite(grad)), "DirectObs gradient not finite"
tests/unit/test_diff_data_assimilation.py:92:        # Only observed points should have non-zero gradient
tests/unit/test_diff_data_assimilation.py:93:        assert jnp.sum(jnp.abs(grad) > 0) > 0, "DirectObs gradient all zero"
tests/unit/test_diff_data_assimilation.py:102:    def test_diagonal_B_sqrt_grad(self):
tests/unit/test_diff_data_assimilation.py:113:        grad = jax.grad(loss)(x)
tests/unit/test_diff_data_assimilation.py:114:        assert_gradient_ok(grad, "DiagonalB sqrt_multiply")
tests/unit/test_diff_data_assimilation.py:116:    def test_diagonal_B_inv_grad(self):
tests/unit/test_diff_data_assimilation.py:127:        grad = jax.grad(loss)(x)
tests/unit/test_diff_data_assimilation.py:128:        assert_gradient_ok(grad, "DiagonalB inv_multiply")
tests/unit/test_diff_data_assimilation.py:145:        def cost_and_grad(x):
tests/unit/test_diff_data_assimilation.py:151:        result = minimize_lbfgs(cost_and_grad, x0, max_iter=100, gtol=1e-4)
tests/unit/test_diff_data_assimilation.py:165:        def cost_and_grad(x):
tests/unit/test_diff_data_assimilation.py:171:        result = minimize_cg(cost_and_grad, x0, max_iter=100, gtol=1e-4)
tests/unit/test_diff_data_assimilation.py:185:    def test_preconditioned_cost_grad(self):
tests/unit/test_diff_data_assimilation.py:201:        grad = jax.grad(precond_cost)(v0)
tests/unit/test_diff_data_assimilation.py:202:        assert_gradient_ok(grad, "Preconditioned cost gradient")
tests/unit/test_diff_land.py:20:def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
tests/unit/test_diff_land.py:21:    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
tests/unit/test_diff_land.py:22:    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
tests/unit/test_diff_land.py:75:    def test_grad_wrt_T_soil(self):
tests/unit/test_diff_land.py:85:        grad = jax.grad(loss)(state.T_soil.data)
tests/unit/test_diff_land.py:86:        assert_gradient_ok(grad, "Slab land w.r.t. T_soil")
tests/unit/test_diff_land.py:88:    def test_grad_wrt_sw_down(self):
tests/unit/test_diff_land.py:98:        grad = jax.grad(loss)(forcing.sw_down)
tests/unit/test_diff_land.py:99:        assert_gradient_ok(grad, "Slab land w.r.t. sw_down")
tests/unit/test_diff_land.py:120:    def test_grad_wrt_T_soil(self):
tests/unit/test_diff_land.py:130:        grad = jax.grad(loss)(state.T_soil)
tests/unit/test_diff_land.py:131:        assert_gradient_ok(grad, "Multi-layer land w.r.t. T_soil")
tests/unit/test_diff_land.py:140:    def test_grad_wrt_T_surface(self):
tests/unit/test_diff_land.py:155:        grad = jax.grad(loss)(T_sfc)
tests/unit/test_diff_land.py:156:        assert_gradient_ok(grad, "Snow budget w.r.t. T_surface")
tests/unit/test_diff_land.py:165:    def test_grad_wrt_T(self):
tests/unit/test_diff_land.py:169:        config = CarbonConfig(scheme="differland")
tests/unit/test_diff_land.py:184:        grad = jax.grad(loss)(T)
tests/unit/test_diff_land.py:185:        assert_gradient_ok(grad, "GPP w.r.t. T")
tests/unit/test_diff_land.py:194:    def test_jarvis_grad_wrt_T(self):
tests/unit/test_diff_land.py:212:        grad = jax.grad(loss)(T)
tests/unit/test_diff_land.py:213:        assert_gradient_ok(grad, "Jarvis gs w.r.t. T")
tests/unit/test_diff_atmosphere_physics.py:3:Tests that jax.grad produces finite, non-zero gradients through each
tests/unit/test_diff_atmosphere_physics.py:29:def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
tests/unit/test_diff_atmosphere_physics.py:30:    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
tests/unit/test_diff_atmosphere_physics.py:31:    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
tests/unit/test_diff_atmosphere_physics.py:77:    def test_grad_wrt_T(self):
tests/unit/test_diff_atmosphere_physics.py:86:        grad = jax.grad(loss)(state.T.data)
tests/unit/test_diff_atmosphere_physics.py:87:        assert_gradient_ok(grad, "Held-Suarez dT_dt w.r.t. T")
tests/unit/test_diff_atmosphere_physics.py:115:    def test_grad_wrt_sfc_temp(self):
tests/unit/test_diff_atmosphere_physics.py:125:        grad = jax.grad(loss)(self.sfc_temp)
tests/unit/test_diff_atmosphere_physics.py:126:        assert_gradient_ok(grad, "Gray radiation w.r.t. sfc_temp")
tests/unit/test_diff_atmosphere_physics.py:145:    def test_grad_wrt_T(self, scheme):
tests/unit/test_diff_atmosphere_physics.py:158:        grad = jax.grad(loss)(state.T.data)
tests/unit/test_diff_atmosphere_physics.py:159:        assert_gradient_ok(grad, f"Convection({scheme}) w.r.t. T", min_nonzero_frac=0.01)
tests/unit/test_diff_atmosphere_physics.py:178:    def test_grad_wrt_T(self, scheme):
tests/unit/test_diff_atmosphere_physics.py:191:        grad = jax.grad(loss)(state.T.data)
tests/unit/test_diff_atmosphere_physics.py:192:        assert_gradient_ok(grad, f"Turbulence({scheme}) w.r.t. T", min_nonzero_frac=0.01)
tests/unit/test_diff_atmosphere_physics.py:211:    def test_grad_wrt_qv(self, scheme):
tests/unit/test_diff_atmosphere_physics.py:225:        grad = jax.grad(loss)(state.tracers["q_v"].data)
tests/unit/test_diff_atmosphere_physics.py:226:        assert_gradient_ok(grad, f"Microphysics({scheme}) w.r.t. q_v", min_nonzero_frac=0.01)
tests/unit/test_diff_atmosphere_physics.py:259:    def test_grad_wrt_T(self):
tests/unit/test_diff_atmosphere_physics.py:267:        grad = jax.grad(loss)(state.T.data)
tests/unit/test_diff_atmosphere_physics.py:268:        assert_gradient_ok(grad, "Combined physics w.r.t. T")
tests/unit/test_fc_gram.py:35:    """FC continuation of a padded polynomial is finite."""
tests/unit/test_fc_gram.py:44:    assert jnp.all(jnp.isfinite(f_ext))
tests/unit/test_fc_gram.py:80:    assert jnp.all(jnp.isfinite(f_filt))
tests/unit/test_fc_gram.py:109:    # in practice, Laplacian is computed as div(grad), two 1st-derivative steps
tests/unit/test_physics_gwd.py:13:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/unit/test_physics_gwd.py:14:from legoesm.atmosphere.physics.gravity_wave_drag.integration import make_gwd_physics
tests/unit/test_physics_gwd.py:39:def _run_gwd(scheme, **kwargs):
tests/unit/test_physics_gwd.py:42:    gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
tests/unit/test_physics_gwd.py:43:    tend, prog = gwd_fn(state, grid, sigma)
tests/unit/test_physics_gwd.py:59:    tend, state = _run_gwd(scheme, wind_speed=15.0)
tests/unit/test_physics_gwd.py:86:    gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
tests/unit/test_physics_gwd.py:87:    tend, _ = gwd_fn(state, grid, sigma)
tests/unit/test_physics_gwd.py:101:    tend, _ = _run_gwd(scheme)
tests/unit/test_physics_gwd.py:109:# All outputs finite
tests/unit/test_physics_gwd.py:113:def test_all_finite(scheme):
tests/unit/test_physics_gwd.py:114:    """All GWD output fields should be finite."""
tests/unit/test_physics_gwd.py:115:    tend, _ = _run_gwd(scheme)
tests/unit/test_physics_gwd.py:116:    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{scheme}: du_dt NaN/Inf"
tests/unit/test_physics_gwd.py:117:    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{scheme}: dv_dt NaN/Inf"
tests/unit/test_physics_gwd.py:118:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt NaN/Inf"
tests/unit/test_physics_gwd.py:127:    tend, _ = _run_gwd("rayleigh", nlev=20, wind_speed=20.0)
tests/unit/test_cmor_experiments_restart.py:536:    def test_different_values(self):
tests/unit/test_cmor_experiments_restart.py:537:        """Different array values give different digest."""
tests/unit/test_cmor_experiments_restart.py:566:    def test_different_config(self):
tests/unit/test_cmor_experiments_restart.py:567:        """Different config gives different hash."""
tests/unit/test_cmor_experiments_restart.py:688:            self.assertEqual(len(report.differences), 0)
tests/unit/test_cmor_experiments_restart.py:704:        expected = {"dynamics", "radiation", "convection", "diffusion", "surface"}
tests/unit/test_cmor_experiments_restart.py:765:        self.assertGreater(r["hyperdiff_scale"], 1e15)
tests/ocean/unit/test_advection_som.py:8:5. Full 3D: conservation, differentiability
tests/ocean/unit/test_advection_som.py:9:6. Comparison with TVD: SOM should be less diffusive on smooth profiles
tests/ocean/unit/test_advection_som.py:69:        # With sx != 0 and different sign_edge, fp_o should differ
tests/ocean/unit/test_advection_som.py:282:        assert jnp.all(jnp.isfinite(T_new))
tests/ocean/unit/test_advection_som.py:351:    def test_differentiable(self):
tests/ocean/unit/test_advection_som.py:352:        """jax.grad should work through som_advect_tracers."""
tests/ocean/unit/test_advection_som.py:366:        grad = jax.grad(loss)(tracer)
tests/ocean/unit/test_advection_som.py:367:        # Gradient should exist and be finite
tests/ocean/unit/test_advection_som.py:368:        assert jnp.all(jnp.isfinite(grad))
tests/ocean/unit/test_advection_som.py:369:        # Non-zero gradient (the loss depends on tracer)
tests/ocean/unit/test_advection_som.py:370:        assert jnp.any(grad != 0.0)
tests/unit/test_land_ice_integrated.py:62:        carbon_cfg = CarbonConfig(scheme="differland")
tests/unit/test_land_ice_integrated.py:81:        # All carbon pools should be finite
tests/unit/test_land_ice_integrated.py:83:            assert jnp.all(jnp.isfinite(getattr(new_carbon, name)))
tests/unit/test_land_ice_integrated.py:115:        # All responses should be finite
tests/unit/test_land_ice_integrated.py:117:            assert jnp.all(jnp.isfinite(getattr(resp_land, name))), f"Land.{name} not finite"
tests/unit/test_land_ice_integrated.py:118:            assert jnp.all(jnp.isfinite(getattr(resp_ice, name))), f"Ice.{name} not finite"
tests/unit/test_land_ice_integrated.py:119:            assert jnp.all(jnp.isfinite(getattr(resp_lake, name))), f"Lake.{name} not finite"
tests/unit/test_land_ice_integrated.py:121:        # T_surface should differ between tiles
tests/unit/test_land_ice_integrated.py:125:        # Albedo should differ
tests/unit/test_operators_fc.py:9:    fc_gradient_x,
tests/unit/test_operators_fc.py:10:    fc_gradient_y,
tests/unit/test_operators_fc.py:28:def test_gradient_constant_is_zero(grid_and_config):
tests/unit/test_operators_fc.py:34:    gx = fc_gradient_x(q, grid, fc_config)
tests/unit/test_operators_fc.py:35:    gy = fc_gradient_y(q, grid, fc_config)
tests/unit/test_operators_fc.py:50:    assert jnp.all(jnp.isfinite(div))
tests/unit/test_operators_fc.py:62:    assert jnp.all(jnp.isfinite(vort))
tests/unit/test_operators_fc.py:92:    """Divergence damping produces finite results."""
tests/unit/test_operators_fc.py:103:    assert jnp.all(jnp.isfinite(du_damp))
tests/unit/test_operators_fc.py:104:    assert jnp.all(jnp.isfinite(dv_damp))
tests/unit/test_operators_fc.py:127:    assert jnp.all(jnp.isfinite(lap))
tests/unit/test_operators_fc.py:130:def test_operators_differentiable(grid_and_config):
tests/unit/test_operators_fc.py:131:    """FC operators are differentiable with jax.grad."""
tests/unit/test_operators_fc.py:136:        gx = fc_gradient_x(q, grid, fc_config)
tests/unit/test_operators_fc.py:140:    grad_q = jax.grad(loss)(q)
tests/unit/test_operators_fc.py:141:    assert grad_q.shape == (6, n, n)
tests/unit/test_operators_fc.py:142:    assert jnp.all(jnp.isfinite(grad_q))
tests/unit/test_convection_plume.py:15:* finite gradients through every helper at non-trivial input values;
tests/unit/test_convection_plume.py:134:def test_compute_lcl_grad_through_temperature():
tests/unit/test_convection_plume.py:135:    """``d T_LCL / d T_parcel`` is finite — preserves training signal
tests/unit/test_convection_plume.py:144:    g = float(jax.grad(f)(jnp.asarray(290.0)))
tests/unit/test_convection_plume.py:145:    assert np.isfinite(g)
tests/unit/test_convection_plume.py:150:# compute_lfc_lnb — order and smooth-grad
tests/unit/test_convection_plume.py:172:def test_compute_lfc_lnb_grad_through_environment():
tests/unit/test_convection_plume.py:173:    """``d k_lfc / d T_sfc`` is finite — training-signal preservation."""
tests/unit/test_convection_plume.py:183:    g = float(jax.grad(f)(jnp.asarray(0.0)))
tests/unit/test_convection_plume.py:184:    assert np.isfinite(g)
tests/unit/test_convection_plume.py:191:def test_compute_cin_non_negative_and_finite():
tests/unit/test_convection_plume.py:193:    buoyancy) and finite for every column."""
tests/unit/test_convection_plume.py:201:    assert jnp.all(jnp.isfinite(cin))
tests/unit/test_convection_plume.py:221:def test_plume_outputs_finite_and_correct_shape():
tests/unit/test_convection_plume.py:222:    """Smoke: every output array is finite with the expected
tests/unit/test_convection_plume.py:239:        assert jnp.all(jnp.isfinite(arr))
tests/unit/test_convection_plume.py:273:    diff = jnp.abs(plume_T - ma_T)
tests/unit/test_convection_plume.py:274:    assert jnp.all(diff < 5.0), (
tests/unit/test_convection_plume.py:276:        f"got differences {np.asarray(diff)} K"
tests/unit/test_convection_plume.py:310:def test_plume_grad_through_epsilon():
tests/unit/test_convection_plume.py:311:    """``d (sum M_u) / d epsilon_0`` is finite — the plume integrator
tests/unit/test_convection_plume.py:312:    is differentiable through its tunables."""
tests/unit/test_convection_plume.py:329:    g = float(jax.grad(f)(jnp.asarray(5.0e-4)))
tests/unit/test_convection_plume.py:330:    assert np.isfinite(g)
tests/unit/test_convection_plume.py:356:def test_cmt_finite_in_sheared_environment():
tests/unit/test_convection_plume.py:358:    finite, non-trivial momentum tendencies."""
tests/unit/test_convection_plume.py:368:    assert jnp.all(jnp.isfinite(du))
tests/unit/test_convection_plume.py:369:    assert jnp.all(jnp.isfinite(dv))
tests/unit/test_convection_plume.py:375:def test_cmt_grad_through_c_u():
tests/unit/test_convection_plume.py:376:    """``d (sum du_dt) / d c_u`` is finite — the closure coefficient
tests/unit/test_convection_plume.py:377:    is a tunable parameter and gradients must flow.  ``c_u`` is
tests/unit/test_convection_plume.py:394:    g = float(jax.grad(f)(jnp.asarray(0.55)))
tests/unit/test_convection_plume.py:395:    assert np.isfinite(g)
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:9:- Differentiability: jax.grad through the bridge succeeds.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:214:    def test_grad_flows(self, grid):
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:226:        g = jax.grad(loss)(jnp.array(2.0))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:227:        assert bool(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:350:        spectral states with the same temperature/q_v but different
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:351:        wind divergences produce different ``q_v`` flux divergences →
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:352:        different Tiedtke tendencies.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:356:        spectral-MC plumbing the two columns now differ.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:411:        # The cloud-base mass-flux carry should differ — Tiedtke's
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:416:        # b_carry should differ from a_carry — proves the MC plumbing
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:418:        max_diff = float(jnp.max(jnp.abs(prog_a - prog_b)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:419:        assert max_diff > 0.0, (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:429:        should produce different KF tendencies when CAPE > 0 — the
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:431:        differ at the diagnostic-mass-flux carry.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:487:        # differ between the two states because divergence shifts the
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:491:        max_diff = float(jnp.max(jnp.abs(a_carry - b_carry)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:492:        assert max_diff > 0.0, (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:548:            hyperdiff_coeff=1.0 / (4.0 * 3600.0 * (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:561:        # vapor) — but stay finite and bounded by the initial q_v.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:565:        assert bool(jnp.all(jnp.isfinite(new_qv)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:575:        # All spectral fields are finite (no NaN from tracer plumbing).
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:576:        assert bool(jnp.all(jnp.isfinite(new_state.vor_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:577:        assert bool(jnp.all(jnp.isfinite(new_state.div_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:578:        assert bool(jnp.all(jnp.isfinite(new_state.T_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:579:        assert bool(jnp.all(jnp.isfinite(new_state.lnps_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:581:    def test_grad_through_bridge(self, grid, sigma_coord, rest_state):
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:582:        """jax.grad through the spectral PE convection bridge succeeds.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:584:        Uses a real-valued amplitude parameter as the differentiation
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:607:        g = jax.grad(loss)(jnp.array(1.0))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:608:        assert bool(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:640:    Gaussian spectral-PE state without raising and produce finite
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:690:    def test_finite_output(
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:693:        """Tendencies are finite for each scheme on spectral PE."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:700:        assert bool(jnp.all(jnp.isfinite(tend.T_hat.data))), (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:703:        assert bool(jnp.all(jnp.isfinite(tend.vor_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:704:        assert bool(jnp.all(jnp.isfinite(tend.div_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:705:        assert bool(jnp.all(jnp.isfinite(tend.lnps_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:709:            # must be finite — the dict carries
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:714:                        assert bool(jnp.all(jnp.isfinite(v))), (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:718:                assert bool(jnp.all(jnp.isfinite(prog)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:739:        # And the resulting T tendency must be finite (no NaN).
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:740:        assert bool(jnp.all(jnp.isfinite(tend.T_hat.data)))
tests/unit/test_halo.py:145:        assert jnp.isfinite(result)
tests/unit/test_halo.py:147:    def test_grad_compatible(self):
tests/unit/test_halo.py:148:        """jax.grad should work through pad_halo."""
tests/unit/test_halo.py:154:        grads = jax.grad(loss)(data)
tests/unit/test_halo.py:155:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_halo.py:156:        assert not jnp.allclose(grads, 0.0)
tests/unit/test_halo.py:180:            # Depth 0 (adjacent to interior) and depth 1 (outer)
tests/unit/test_halo.py:254:        assert jnp.isfinite(result)
tests/unit/test_land_ice_albedo.py:36:        dalpha = jnp.abs(jnp.diff(alpha))
tests/unit/test_land_ice_albedo.py:51:        assert jnp.all(jnp.diff(alpha) <= 0)
tests/ocean/unit/test_cross_grid_parity.py:6:Parity criteria (not exact equality --- different numerics):
tests/ocean/unit/test_cross_grid_parity.py:7:- All grids produce finite output
tests/ocean/unit/test_cross_grid_parity.py:55:        hyperdiff_coeff=0.0,
tests/ocean/unit/test_cross_grid_parity.py:139:        assert jnp.all(jnp.isfinite(s.T.data))
tests/ocean/unit/test_cross_grid_parity.py:175:        assert jnp.all(jnp.isfinite(s.T.data))
tests/ocean/unit/test_cross_grid_parity.py:216:        assert jnp.all(jnp.isfinite(T_grid))
tests/ocean/unit/test_cross_grid_parity.py:325:            "inertia_gravity_wave",
tests/ocean/unit/test_cross_grid_parity.py:360:            "baroclinic", "phillips_two_layer", "inertia_gravity_wave",
tests/unit/test_land_ice_snow.py:137:        dalpha = jnp.diff(alpha)
tests/unit/test_symmetry_invariance.py:60:            hyperdiff_coeff=0.0,
tests/unit/test_symmetry_invariance.py:71:        A zonal jet naturally has different mean h on polar vs equatorial faces.
tests/unit/test_symmetry_invariance.py:89:            f"Equatorial face means differ by {max_eq_dev:.4e}, expected < 0.01"
tests/unit/test_symmetry_invariance.py:96:            f"Polar face means differ by {pol_dev:.4e}, expected < 0.01"
tests/unit/test_symmetry_invariance.py:105:    """Run the same physical problem on different faces of the cubed-sphere.
tests/unit/test_symmetry_invariance.py:112:        from legoesm.core.operators import hyperdiffusion
tests/unit/test_symmetry_invariance.py:115:        return grid, hyperdiffusion
tests/unit/test_symmetry_invariance.py:118:        """Place an impulse on each face and compare L2 norms after diffusion."""
tests/unit/test_symmetry_invariance.py:119:        grid, hyperdiffusion = setup
tests/unit/test_symmetry_invariance.py:132:                tendency = hyperdiffusion(field, grid, nu)
tests/unit/test_symmetry_invariance.py:141:                f"Face {i} L2 = {l2:.6e} differs from mean {mean_l2:.6e} by {rel:.3e}"
tests/unit/test_symmetry_invariance.py:198:        rel_diff = abs(mean_NH - mean_SH) / abs(mean_NH)
tests/unit/test_symmetry_invariance.py:199:        assert rel_diff < 0.01, (
tests/unit/test_symmetry_invariance.py:200:            f"NH/SH mean h differs: NH={mean_NH:.2f}, SH={mean_SH:.2f}, "
tests/unit/test_symmetry_invariance.py:201:            f"rel_diff={rel_diff:.4e}"
tests/unit/test_equation_fixes.py:186:        """Condensation tendency should differ when dt changes (not dt-independent)."""
tests/unit/test_equation_fixes.py:194:        # Even though other processes may dominate, dT_dt should differ
tests/unit/test_equation_fixes.py:195:        diff = jnp.abs(out1.dT_dt - out2.dT_dt).sum()
tests/unit/test_equation_fixes.py:196:        assert diff > 0, "dT_dt should depend on dt (condensation is divided by dt)"
tests/unit/test_equation_fixes.py:249:        diff = jnp.abs(out1.dT_dt - out2.dT_dt).sum()
tests/unit/test_equation_fixes.py:250:        assert diff > 0, f"{scheme_name}: dT_dt should depend on dt"
tests/unit/test_equation_fixes.py:421:        jacobian = jnp.ones(shape_2d)
tests/unit/test_equation_fixes.py:435:            eta, z_coord.dz_ref, jacobian,
tests/unit/test_equation_fixes.py:443:        out = kpp_vertical_mixing(u, v, T, S, rho, eta, z_coord, jacobian, cfg, B_f=B_f)
tests/unit/test_equation_fixes.py:467:        jacobian = jnp.ones(shape_2d)
tests/unit/test_equation_fixes.py:480:        out = bulk_formula_surface_forcing(T, S, z_coord, jacobian, cfg)
tests/unit/test_equation_fixes.py:546:        # The difference should scale with q_v * dp / g
tests/unit/test_diff_ocean.py:19:def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
tests/unit/test_diff_ocean.py:20:    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
tests/unit/test_diff_ocean.py:21:    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
tests/unit/test_diff_ocean.py:53:    def test_grad_wrt_T(self):
tests/unit/test_diff_ocean.py:61:        grad = jax.grad(loss)(state.T.data)
tests/unit/test_diff_ocean.py:62:        assert_gradient_ok(grad, "CS Ocean single step w.r.t. T")
tests/unit/test_diff_ocean.py:85:    def test_grad_wrt_T(self):
tests/unit/test_diff_ocean.py:93:        grad = jax.grad(loss)(state.T.data)
tests/unit/test_diff_ocean.py:94:        assert_gradient_ok(grad, "MPAS Ocean single step w.r.t. T")
tests/unit/test_diff_ocean.py:103:    def test_wright_eos_grad_T(self):
tests/unit/test_diff_ocean.py:110:        drho_dT = jax.grad(lambda T: wright_eos(T, S, p))(T)
tests/unit/test_diff_ocean.py:111:        assert jnp.isfinite(drho_dT), "drho/dT is not finite"
tests/unit/test_diff_ocean.py:115:    def test_wright_eos_grad_S(self):
tests/unit/test_diff_ocean.py:122:        drho_dS = jax.grad(lambda S: wright_eos(T, S, p))(S)
tests/unit/test_diff_ocean.py:123:        assert jnp.isfinite(drho_dS), "drho/dS is not finite"
tests/unit/test_diff_ocean.py:129:# 3e  Barotropic solver differentiability
tests/unit/test_diff_ocean.py:149:            differentiable_barotropic=True,  # use lax.scan for AD
tests/unit/test_diff_ocean.py:155:    def test_grad_wrt_eta(self):
tests/unit/test_diff_ocean.py:164:        grad = jax.grad(loss)(state.eta.data)
tests/unit/test_diff_ocean.py:165:        assert_gradient_ok(grad, "Barotropic solver w.r.t. eta")
tests/unit/test_diff_ocean.py:167:    def test_grad_3_steps(self):
tests/unit/test_diff_ocean.py:168:        """Multi-step gradient through ocean model with differentiable barotropic."""
tests/unit/test_diff_ocean.py:178:        grad = jax.grad(loss)(state.T.data)
tests/unit/test_diff_ocean.py:179:        assert_gradient_ok(grad, "Ocean 3 steps (diff barotropic) w.r.t. T")
tests/unit/test_diff_ocean.py:188:    def test_grad_with_vertical_mixing(self):
tests/unit/test_diff_ocean.py:203:            K_v=1e-4,  # Vertical diffusivity enabled
tests/unit/test_diff_ocean.py:214:        grad = jax.grad(loss)(state.T.data)
tests/unit/test_diff_ocean.py:215:        assert_gradient_ok(grad, "Ocean + vertical mixing w.r.t. T")
tests/unit/test_compiled_segments.py:420:        assert np.all(np.isfinite(T_final)), "Temperature should be finite"
tests/unit/test_compiled_segments.py:422:    def test_output_all_finite(self):
tests/unit/test_compiled_segments.py:423:        """All carry fields remain finite after a segment run."""
tests/unit/test_compiled_segments.py:448:            assert np.all(np.isfinite(arr)), f"{field_name} has non-finite values"
tests/unit/test_compiled_segments.py:491:    from legoesm.core.operators_3d import hyperdiffusion_3d
tests/unit/test_compiled_segments.py:554:            q_v_upd + dt * hyperdiffusion_3d(q_v_upd, grid, qv_smooth_coeff),
tests/unit/test_compiled_segments.py:745:        assert np.all(np.isfinite(np.asarray(result.T)))
tests/unit/test_production_blockers.py:152:    def test_gravity_wave_drag_field_exists(self):
tests/unit/test_production_blockers.py:154:        assert hasattr(ec, "gravity_wave_drag")
tests/unit/test_production_blockers.py:155:        assert ec.gravity_wave_drag == "none"
tests/unit/test_production_blockers.py:493:            gravity_wave_drag="rayleigh",
tests/unit/test_production_blockers.py:499:        assert ec2.gravity_wave_drag == "rayleigh"
tests/unit/test_production_blockers.py:511:        assert ec.gravity_wave_drag == "none"
tests/unit/test_scale_global_reductions.py:96:# JIT and differentiability
tests/unit/test_scale_global_reductions.py:104:        assert jnp.isfinite(result)
tests/unit/test_scale_global_reductions.py:106:    def test_global_integral_differentiable(self, grid):
tests/unit/test_scale_global_reductions.py:111:        grad = jax.grad(loss)(data)
tests/unit/test_scale_global_reductions.py:113:        np.testing.assert_allclose(grad, grid.area, rtol=1e-12)
tests/unit/test_tiedtke.py:5:* tendency shape / dtype / finiteness;
tests/unit/test_tiedtke.py:13:* finite gradients through ``epsilon_deep``, ``cape_threshold``,
tests/unit/test_tiedtke.py:40:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/unit/test_tiedtke.py:70:# Shape / finiteness
tests/unit/test_tiedtke.py:73:def test_tiedtke_shape_finiteness():
tests/unit/test_tiedtke.py:82:        assert jnp.all(jnp.isfinite(arr))
tests/unit/test_tiedtke.py:108:    profile toward equilibrium with monotone-contracting differences."""
tests/unit/test_tiedtke.py:117:    # Successive layer-summed differences shrink (relaxation).
tests/unit/test_tiedtke.py:118:    diffs = [
tests/unit/test_tiedtke.py:123:    # up; check that the final differences are smaller than the
tests/unit/test_tiedtke.py:125:    assert diffs[-1] < diffs[0] + 1e-12
tests/unit/test_tiedtke.py:169:    diff = jnp.abs(out_on.dT_dt[:, -4:] - out_off.dT_dt[:, -4:])
tests/unit/test_tiedtke.py:170:    assert float(jnp.max(diff)) > 1e-8
tests/unit/test_tiedtke.py:177:def test_tiedtke_grad_through_epsilon_deep():
tests/unit/test_tiedtke.py:187:    g = float(jax.grad(f)(jnp.asarray(1.0e-4)))
tests/unit/test_tiedtke.py:188:    assert np.isfinite(g)
tests/unit/test_tiedtke.py:191:def test_tiedtke_grad_through_cape_threshold():
tests/unit/test_tiedtke.py:201:    g = float(jax.grad(f)(jnp.asarray(70.0)))
tests/unit/test_tiedtke.py:202:    assert np.isfinite(g)
tests/unit/test_tiedtke.py:205:def test_tiedtke_grad_through_cmt_c_u():
tests/unit/test_tiedtke.py:215:    g = float(jax.grad(f)(jnp.asarray(0.7)))
tests/unit/test_tiedtke.py:216:    assert np.isfinite(g)
tests/unit/test_tiedtke.py:223:def test_tiedtke_orchestrator_one_step_finite():
tests/unit/test_tiedtke.py:239:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_tiedtke.py:247:        assert jnp.all(jnp.isfinite(f.data))
tests/unit/test_diff_atmosphere_dynamics.py:3:Tests that jax.grad produces finite, non-zero, spatially structured gradients
tests/unit/test_diff_atmosphere_dynamics.py:25:def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
tests/unit/test_diff_atmosphere_dynamics.py:26:    """Check that a gradient array is finite, non-trivially non-zero, and has
tests/unit/test_diff_atmosphere_dynamics.py:28:    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
tests/unit/test_diff_atmosphere_dynamics.py:29:    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
tests/unit/test_diff_atmosphere_dynamics.py:33:    assert not jnp.all(grad_array == grad_array.ravel()[0]), (
tests/unit/test_diff_atmosphere_dynamics.py:34:        f"{name}: gradient is spatially uniform (no structure)"
tests/unit/test_diff_atmosphere_dynamics.py:66:    def test_grad_single_step(self):
tests/unit/test_diff_atmosphere_dynamics.py:74:        grad = jax.grad(loss)(state.h)
tests/unit/test_diff_atmosphere_dynamics.py:75:        assert_gradient_ok(grad, "CDGrid SW single step")
tests/unit/test_diff_atmosphere_dynamics.py:77:    def test_grad_5_steps(self):
tests/unit/test_diff_atmosphere_dynamics.py:87:        grad = jax.grad(loss)(state.h)
tests/unit/test_diff_atmosphere_dynamics.py:88:        assert_gradient_ok(grad, "CDGrid SW 5 steps")
tests/unit/test_diff_atmosphere_dynamics.py:125:    def test_grad_single_step(self):
tests/unit/test_diff_atmosphere_dynamics.py:134:        grad = jax.grad(loss)(state.phi_hat.data.real)
tests/unit/test_diff_atmosphere_dynamics.py:135:        assert_gradient_ok(grad, "Spectral SW single step")
tests/unit/test_diff_atmosphere_dynamics.py:137:    def test_grad_5_steps(self):
tests/unit/test_diff_atmosphere_dynamics.py:148:        grad = jax.grad(loss)(state.phi_hat.data.real)
tests/unit/test_diff_atmosphere_dynamics.py:149:        assert_gradient_ok(grad, "Spectral SW 5 steps")
tests/unit/test_diff_atmosphere_dynamics.py:179:    def test_grad_single_step(self):
tests/unit/test_diff_atmosphere_dynamics.py:187:        grad = jax.grad(loss)(state.h.data)
tests/unit/test_diff_atmosphere_dynamics.py:188:        assert_gradient_ok(grad, "MPAS SW single step")
tests/unit/test_diff_atmosphere_dynamics.py:190:    def test_grad_5_steps(self):
tests/unit/test_diff_atmosphere_dynamics.py:200:        grad = jax.grad(loss)(state.h.data)
tests/unit/test_diff_atmosphere_dynamics.py:201:        assert_gradient_ok(grad, "MPAS SW 5 steps")
tests/unit/test_diff_atmosphere_dynamics.py:240:    def test_grad_single_step(self):
tests/unit/test_diff_atmosphere_dynamics.py:248:        grad = jax.grad(loss)(state.T.data)
tests/unit/test_diff_atmosphere_dynamics.py:249:        assert_gradient_ok(grad, "CDGrid PE single step")
tests/unit/test_diff_atmosphere_dynamics.py:251:    def test_grad_3_steps(self):
tests/unit/test_diff_atmosphere_dynamics.py:261:        grad = jax.grad(loss)(state.T.data)
tests/unit/test_diff_atmosphere_dynamics.py:262:        assert_gradient_ok(grad, "CDGrid PE 3 steps")
tests/unit/test_diff_atmosphere_dynamics.py:303:    def test_grad_single_step(self):
tests/unit/test_diff_atmosphere_dynamics.py:311:        grad = jax.grad(loss)(state.theta_prime.data)
tests/unit/test_diff_atmosphere_dynamics.py:312:        assert_gradient_ok(grad, "CompEuler single step")
tests/unit/test_diff_atmosphere_dynamics.py:358:    def test_grad_single_step(self):
tests/unit/test_diff_atmosphere_dynamics.py:367:        grad = jax.grad(loss)(state.T_hat.data.real)
tests/unit/test_diff_atmosphere_dynamics.py:368:        assert_gradient_ok(grad, "Spectral PE single step")
tests/unit/test_multi_gpu.py:169:            from legoesm.core.operators import gradient_x, divergence
tests/unit/test_multi_gpu.py:179:            # Single-device gradient
tests/unit/test_multi_gpu.py:180:            gx_single = gradient_x(field, grid)
tests/unit/test_multi_gpu.py:182:            # Sharded gradient
tests/unit/test_multi_gpu.py:189:            def compute_grad(f):
tests/unit/test_multi_gpu.py:190:                return gradient_x(f, grid)
tests/unit/test_multi_gpu.py:192:            gx_sharded = compute_grad(field_sharded)
tests/unit/test_multi_gpu.py:198:                "grad_err": err,
tests/unit/test_multi_gpu.py:200:                "finite": bool(jnp.all(jnp.isfinite(gx_sharded.data))),
tests/unit/test_multi_gpu.py:204:        assert out["finite"] is True
tests/unit/test_multi_gpu.py:206:        assert out["grad_err"] < 1e-10, f"gradient mismatch: {out['grad_err']}"
tests/unit/test_multi_gpu.py:229:            # Smooth test field (face index + gradient)
tests/unit/test_multi_gpu.py:337:            # Simple diffusion step using pad_halo (tests cross-face comm)
tests/unit/test_multi_gpu.py:338:            def diffusion_step(state, _):
tests/unit/test_multi_gpu.py:351:            result_single, _ = jax.lax.scan(diffusion_step, data, None, length=10)
tests/unit/test_multi_gpu.py:359:                return jax.lax.scan(diffusion_step, d, None, length=10)[0]
tests/unit/test_multi_gpu.py:365:            finite = bool(
tests/unit/test_multi_gpu.py:366:                jnp.all(jnp.isfinite(result_single))
tests/unit/test_multi_gpu.py:367:                and jnp.all(jnp.isfinite(result_sharded))
tests/unit/test_multi_gpu.py:373:                "finite": finite,
tests/unit/test_multi_gpu.py:377:        assert out["finite"] is True
tests/unit/test_multi_gpu.py:380:        # can differ, causing ~1e-7 differences from accumulation reordering.
tests/unit/test_distributed_checkpoint.py:318:        # Rank 1 should have different T values
tests/unit/test_precision_dtype_contracts.py:193:    def test_gwd_zero_output_dtype(self, mode_name, policy):
tests/unit/test_precision_dtype_contracts.py:197:        from legoesm.atmosphere.physics.gravity_wave_drag.output import (
tests/unit/test_precision_dtype_contracts.py:203:        assert out.eps_gwd.dtype == policy.storage
tests/ocean/unit/test_ocean_biogeochemistry.py:279:        assert jnp.all(jnp.isfinite(pCO2))
tests/ocean/unit/test_ocean_biogeochemistry.py:280:        assert jnp.all(jnp.isfinite(pH))
tests/ocean/unit/test_ocean_biogeochemistry.py:357:    def test_diagnostics_finite(self):
tests/ocean/unit/test_ocean_biogeochemistry.py:365:        assert jnp.isfinite(diag.pCO2_ocean)
tests/ocean/unit/test_ocean_biogeochemistry.py:366:        assert jnp.isfinite(diag.pH)
tests/ocean/unit/test_ocean_biogeochemistry.py:367:        assert jnp.isfinite(diag.flux_co2)
tests/ocean/unit/test_ocean_biogeochemistry.py:368:        assert jnp.isfinite(diag.k_w)
tests/ocean/unit/test_ocean_biogeochemistry.py:407:    def test_source_sink_finite(self, z_ref):
tests/ocean/unit/test_ocean_biogeochemistry.py:422:            assert jnp.all(jnp.isfinite(r))
tests/ocean/unit/test_ocean_biogeochemistry.py:487:        diff = float(jnp.max(jnp.abs(new_state.DIC - state.DIC)))
tests/ocean/unit/test_ocean_biogeochemistry.py:488:        assert diff > 0
tests/ocean/unit/test_ocean_biogeochemistry.py:539:            assert jnp.all(jnp.isfinite(val))
tests/ocean/unit/test_ocean_biogeochemistry.py:557:        """10 steps should remain finite and non-negative."""
tests/ocean/unit/test_ocean_biogeochemistry.py:572:            assert jnp.all(jnp.isfinite(val)), f"{field} not finite after 10 steps"
tests/ocean/unit/test_ocean_biogeochemistry.py:597:    def test_grad_through_carbonate_solver(self):
tests/ocean/unit/test_ocean_biogeochemistry.py:598:        """Carbonate solver should be differentiable."""
tests/ocean/unit/test_ocean_biogeochemistry.py:604:        grad = jax.grad(loss)(jnp.array(2.1))
tests/ocean/unit/test_ocean_biogeochemistry.py:605:        assert jnp.isfinite(grad)
tests/ocean/unit/test_ocean_biogeochemistry.py:607:    def test_grad_through_gas_exchange(self):
tests/ocean/unit/test_ocean_biogeochemistry.py:608:        """Air-sea flux should be differentiable w.r.t. DIC."""
tests/ocean/unit/test_ocean_biogeochemistry.py:615:        grad = jax.grad(loss)(jnp.array(2.1))
tests/ocean/unit/test_ocean_biogeochemistry.py:616:        assert jnp.isfinite(grad)
tests/ocean/unit/test_ocean_biogeochemistry.py:618:    def test_grad_through_step(self, z_ref):
tests/ocean/unit/test_ocean_biogeochemistry.py:619:        """Full step should be differentiable."""
tests/ocean/unit/test_ocean_biogeochemistry.py:634:        grad = jax.grad(loss)(state.DIC)
tests/ocean/unit/test_ocean_biogeochemistry.py:635:        assert jnp.all(jnp.isfinite(grad))
tests/ocean/unit/test_ocean_biogeochemistry.py:636:        assert grad.shape == state.DIC.shape
tests/ocean/unit/test_ocean_biogeochemistry.py:638:    def test_grad_through_npzd(self, z_ref):
tests/ocean/unit/test_ocean_biogeochemistry.py:639:        """NPZD step should be differentiable."""
tests/ocean/unit/test_ocean_biogeochemistry.py:654:        grad = jax.grad(loss)(state.DIC)
tests/ocean/unit/test_ocean_biogeochemistry.py:655:        assert jnp.all(jnp.isfinite(grad))
tests/unit/test_mpas_atmosphere.py:9:6. JAX differentiability
tests/unit/test_mpas_atmosphere.py:163:    def test_tendencies_finite(self):
tests/unit/test_mpas_atmosphere.py:168:        self.assertTrue(jnp.all(jnp.isfinite(tend.du_dt.data)))
tests/unit/test_mpas_atmosphere.py:169:        self.assertTrue(jnp.all(jnp.isfinite(tend.dT_dt.data)))
tests/unit/test_mpas_atmosphere.py:170:        self.assertTrue(jnp.all(jnp.isfinite(tend.dp_s_dt.data)))
tests/unit/test_mpas_atmosphere.py:182:        """Model.step() produces finite state."""
tests/unit/test_mpas_atmosphere.py:187:        self.assertTrue(jnp.all(jnp.isfinite(state_new.u.data)))
tests/unit/test_mpas_atmosphere.py:188:        self.assertTrue(jnp.all(jnp.isfinite(state_new.T.data)))
tests/unit/test_mpas_atmosphere.py:189:        self.assertTrue(jnp.all(jnp.isfinite(state_new.p_s.data)))
tests/unit/test_mpas_atmosphere.py:211:        self.assertTrue(jnp.all(jnp.isfinite(state.u.data)))
tests/unit/test_mpas_atmosphere.py:212:        self.assertTrue(jnp.all(jnp.isfinite(state.T.data)))
tests/unit/test_mpas_atmosphere.py:213:        self.assertTrue(jnp.all(jnp.isfinite(state.p_s.data)))
tests/unit/test_mpas_atmosphere.py:215:    def test_differentiability(self):
tests/unit/test_mpas_atmosphere.py:216:        """Tendencies are differentiable w.r.t. velocity."""
tests/unit/test_mpas_atmosphere.py:224:        grad_fn = jax.grad(loss)
tests/unit/test_mpas_atmosphere.py:225:        g = grad_fn(self.state_pert.u.data)
tests/unit/test_mpas_atmosphere.py:227:        self.assertTrue(jnp.all(jnp.isfinite(g)))
tests/unit/test_mpas_atmosphere.py:271:    def test_slow_tendencies_finite(self):
tests/unit/test_mpas_atmosphere.py:277:        self.assertTrue(jnp.all(jnp.isfinite(tend.du_dt.data)))
tests/unit/test_mpas_atmosphere.py:278:        self.assertTrue(jnp.all(jnp.isfinite(tend.dw_dt.data)))
tests/unit/test_mpas_atmosphere.py:279:        self.assertTrue(jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data)))
tests/unit/test_mpas_atmosphere.py:280:        self.assertTrue(jnp.all(jnp.isfinite(tend.drho_prime_dt.data)))
tests/unit/test_mpas_atmosphere.py:283:        """Model.step() produces finite state."""
tests/unit/test_mpas_atmosphere.py:288:        self.assertTrue(jnp.all(jnp.isfinite(state_new.u.data)))
tests/unit/test_mpas_atmosphere.py:289:        self.assertTrue(jnp.all(jnp.isfinite(state_new.w.data)))
tests/unit/test_mpas_atmosphere.py:290:        self.assertTrue(jnp.all(jnp.isfinite(state_new.theta_prime.data)))
tests/unit/test_mpas_atmosphere.py:291:        self.assertTrue(jnp.all(jnp.isfinite(state_new.rho_prime.data)))
tests/unit/test_mpas_atmosphere.py:310:        self.assertTrue(jnp.all(jnp.isfinite(state.u.data)))
tests/unit/test_mpas_atmosphere.py:311:        self.assertTrue(jnp.all(jnp.isfinite(state.w.data)))
tests/unit/test_mpas_atmosphere.py:312:        self.assertTrue(jnp.all(jnp.isfinite(state.theta_prime.data)))
tests/unit/test_mpas_atmosphere.py:313:        self.assertTrue(jnp.all(jnp.isfinite(state.rho_prime.data)))
tests/unit/test_mpas_atmosphere.py:330:        self.assertTrue(jnp.all(jnp.isfinite(tend.dtracers_dt.data)))
tests/unit/test_mpas_atmosphere.py:332:    def test_differentiability(self):
tests/unit/test_mpas_atmosphere.py:333:        """Slow tendencies are differentiable w.r.t. theta_prime."""
tests/unit/test_mpas_atmosphere.py:342:        grad_fn = jax.grad(loss)
tests/unit/test_mpas_atmosphere.py:343:        g = grad_fn(self.state_pert.theta_prime.data)
tests/unit/test_mpas_atmosphere.py:345:        self.assertTrue(jnp.all(jnp.isfinite(g)))
tests/ocean/unit/test_implicit_vertical_solver.py:1:"""Tests for the implicit backward-Euler vertical diffusion solver (#204).
tests/ocean/unit/test_implicit_vertical_solver.py:4:physical vertical viscosity / diffusivity from the numerical diffusion
tests/ocean/unit/test_implicit_vertical_solver.py:22:    implicit_vertical_diffusion_ocean,
tests/ocean/unit/test_implicit_vertical_solver.py:46:        out = implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=60.0)
tests/ocean/unit/test_implicit_vertical_solver.py:55:        out = implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=60.0)
tests/ocean/unit/test_implicit_vertical_solver.py:64:        out = implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=100.0)
tests/ocean/unit/test_implicit_vertical_solver.py:73:        out = implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=300.0)
tests/ocean/unit/test_implicit_vertical_solver.py:81:            implicit_vertical_diffusion_ocean(phi, 1.0, dz, dzh, dt=0.0)
tests/ocean/unit/test_implicit_vertical_solver.py:83:            implicit_vertical_diffusion_ocean(phi, 1.0, dz, dzh, dt=-1.0)
tests/ocean/unit/test_implicit_vertical_solver.py:91:            implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=1.0)
tests/ocean/unit/test_implicit_vertical_solver.py:98:        out = implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=100.0)
tests/ocean/unit/test_implicit_vertical_solver.py:114:        out = implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=60.0)
tests/ocean/unit/test_implicit_vertical_solver.py:126:        out = implicit_vertical_diffusion_ocean(
tests/ocean/unit/test_implicit_vertical_solver.py:140:            phi = implicit_vertical_diffusion_ocean(
tests/ocean/unit/test_implicit_vertical_solver.py:163:            out = implicit_vertical_diffusion_ocean(
tests/ocean/unit/test_implicit_vertical_solver.py:187:            out = implicit_vertical_diffusion_ocean(out, K, dz, dzh, dt)
tests/ocean/unit/test_implicit_vertical_solver.py:189:        # Higher modes leak into the solution via the centred-difference
tests/ocean/unit/test_implicit_vertical_solver.py:207:            out = implicit_vertical_diffusion_ocean(
tests/ocean/unit/test_implicit_vertical_solver.py:209:        assert jnp.all(jnp.isfinite(out))
tests/ocean/unit/test_implicit_vertical_solver.py:220:        out = implicit_vertical_diffusion_ocean(phi, 0.0, dz, dzh, dt=3600.0)
tests/ocean/unit/test_implicit_vertical_solver.py:238:        out = implicit_vertical_diffusion_ocean(
tests/ocean/unit/test_implicit_vertical_solver.py:244:        # Surface value must have *decreased* (diffused down).
tests/ocean/unit/test_implicit_vertical_solver.py:256:        out = implicit_vertical_diffusion_ocean(
tests/ocean/unit/test_implicit_vertical_solver.py:271:    def test_grad_finite_wrt_field(self):
tests/ocean/unit/test_implicit_vertical_solver.py:278:            out = implicit_vertical_diffusion_ocean(p, K, dz, dzh, dt=60.0)
tests/ocean/unit/test_implicit_vertical_solver.py:281:        g = jax.grad(loss)(phi)
tests/ocean/unit/test_implicit_vertical_solver.py:282:        assert jnp.all(jnp.isfinite(g))
tests/ocean/unit/test_implicit_vertical_solver.py:284:    def test_grad_finite_wrt_K(self):
tests/ocean/unit/test_implicit_vertical_solver.py:291:            out = implicit_vertical_diffusion_ocean(phi, kk, dz, dzh, dt=60.0)
tests/ocean/unit/test_implicit_vertical_solver.py:294:        g = jax.grad(loss)(K)
tests/ocean/unit/test_implicit_vertical_solver.py:295:        assert jnp.all(jnp.isfinite(g))
tests/ocean/unit/test_implicit_vertical_solver.py:305:            return implicit_vertical_diffusion_ocean(p, K, dz, dzh, dt=60.0)
tests/ocean/unit/test_implicit_vertical_solver.py:321:            return implicit_vertical_diffusion_ocean(p, K, dz, dzh, dt=dt)
tests/ocean/unit/test_implicit_vertical_solver.py:325:        assert jnp.all(jnp.isfinite(out))
tests/ocean/unit/test_implicit_vertical_solver.py:336:            implicit_vertical_diffusion_ocean as f1,
tests/unit/test_physics_units.py:31:    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/unit/test_physics_units.py:53:        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
tests/unit/test_physics_units.py:186:    def test_gravity(self):
tests/unit/test_scale_tpu_compat.py:199:        assert jnp.isfinite(result)
tests/unit/test_warm_rain.py:46:    assert jnp.all(jnp.isfinite(cond))
tests/unit/test_warm_rain.py:47:    assert jnp.all(jnp.isfinite(q_sat))
tests/unit/test_warm_rain.py:57:    assert jnp.all(jnp.isfinite(Nc_eff))
tests/unit/test_warm_rain.py:74:    assert jnp.all(jnp.isfinite(dq_au))
tests/unit/test_warm_rain.py:75:    assert jnp.all(jnp.isfinite(dN_au))
tests/unit/test_warm_rain.py:76:    assert jnp.all(jnp.isfinite(x_c))
tests/unit/test_warm_rain.py:85:    assert jnp.all(jnp.isfinite(rate))
tests/unit/test_warm_rain.py:96:    assert jnp.all(jnp.isfinite(dN_sc))
tests/unit/test_warm_rain.py:97:    assert jnp.all(jnp.isfinite(dN_br))
tests/unit/test_warm_rain.py:110:    assert jnp.all(jnp.isfinite(evap))
tests/unit/test_warm_rain.py:132:    assert jnp.all(jnp.isfinite(out))
tests/unit/test_physics_turbulence.py:3:Tests diffusivity positivity, surface flux signs, momentum drag,
tests/unit/test_physics_turbulence.py:72:#     via the finite-output and sign checks. Skip explicit Km/Kh check.
tests/unit/test_physics_turbulence.py:115:# 5d  All tendency fields finite
tests/unit/test_physics_turbulence.py:119:def test_all_tendencies_finite(scheme):
tests/unit/test_physics_turbulence.py:120:    """All tendency fields should be finite."""
tests/unit/test_physics_turbulence.py:123:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt has NaN/Inf"
tests/unit/test_physics_turbulence.py:124:    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{scheme}: du_dt has NaN/Inf"
tests/unit/test_physics_turbulence.py:125:    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{scheme}: dv_dt has NaN/Inf"
tests/ocean/unit/test_implicit_solver.py:10:    implicit_vertical_diffusion_ocean,
tests/ocean/unit/test_implicit_solver.py:22:def test_implicit_diffusion_no_op_with_zero_K():
tests/ocean/unit/test_implicit_solver.py:29:    out = implicit_vertical_diffusion_ocean(field, K, dz, dz_half, dt=600.0)
tests/ocean/unit/test_implicit_solver.py:35:def test_implicit_diffusion_relaxes_jump_to_smooth():
tests/ocean/unit/test_implicit_solver.py:43:    out = implicit_vertical_diffusion_ocean(field.astype(jnp.float64), K, dz, dz_half, dt=86400.0)
tests/ocean/unit/test_implicit_solver.py:46:    assert jnp.all(jnp.isfinite(out))
tests/ocean/unit/test_implicit_solver.py:55:def test_implicit_diffusion_multidim_columns():
tests/ocean/unit/test_implicit_solver.py:62:    out = implicit_vertical_diffusion_ocean(field, K_scalar, dz, dz_half, dt=3600.0)
tests/ocean/unit/test_implicit_solver.py:65:    assert jnp.all(jnp.isfinite(out))
tests/ocean/unit/test_implicit_solver.py:68:def test_implicit_diffusion_rejects_nonpositive_dt():
tests/ocean/unit/test_implicit_solver.py:73:        implicit_vertical_diffusion_ocean(field, 0.0, dz, dz_half, dt=0.0)
tests/ocean/unit/test_implicit_solver.py:76:def test_implicit_diffusion_one_level_noop():
tests/ocean/unit/test_implicit_solver.py:79:    out = implicit_vertical_diffusion_ocean(field, K=jnp.zeros((0,)), dz=dz, dz_half=jnp.zeros((0,)), dt=600.0)
tests/ocean/unit/test_inertia_gravity_wave.py:5:  2. Ocean: inertia-gravity wave (THIS TEST)
tests/ocean/unit/test_inertia_gravity_wave.py:15:Run with: JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/ocean/unit/test_inertia_gravity_wave.py -v
tests/ocean/unit/test_inertia_gravity_wave.py:30:from legoesm.ocean.experiments.inertia_gravity_wave import (
tests/ocean/unit/test_inertia_gravity_wave.py:67:        hyperdiff_coeff=0.0,
tests/ocean/unit/test_inertia_gravity_wave.py:74:    def test_eta_finite(self, igw_state):
tests/ocean/unit/test_inertia_gravity_wave.py:75:        """SSH perturbation should be finite."""
tests/ocean/unit/test_inertia_gravity_wave.py:76:        assert jnp.all(jnp.isfinite(igw_state.eta.data))
tests/ocean/unit/test_inertia_gravity_wave.py:83:    def test_velocities_finite(self, igw_state):
tests/ocean/unit/test_inertia_gravity_wave.py:84:        """Velocities should be finite."""
tests/ocean/unit/test_inertia_gravity_wave.py:85:        assert jnp.all(jnp.isfinite(igw_state.u.data))
tests/ocean/unit/test_inertia_gravity_wave.py:86:        assert jnp.all(jnp.isfinite(igw_state.v.data))
tests/ocean/unit/test_inertia_gravity_wave.py:98:    def test_one_step_finite(self, igw_grid, igw_z_coord, igw_state, ocean_config):
tests/ocean/unit/test_inertia_gravity_wave.py:99:        """One model step should produce finite output."""
tests/ocean/unit/test_inertia_gravity_wave.py:106:        assert jnp.all(jnp.isfinite(state_new.eta.data)), "eta NaN after 1 step"
tests/ocean/unit/test_inertia_gravity_wave.py:107:        assert jnp.all(jnp.isfinite(state_new.u.data)), "u NaN after 1 step"
tests/ocean/unit/test_inertia_gravity_wave.py:108:        assert jnp.all(jnp.isfinite(state_new.v.data)), "v NaN after 1 step"
tests/ocean/unit/test_inertia_gravity_wave.py:119:        assert jnp.all(jnp.isfinite(state.eta.data)), "eta NaN after 10 steps"
tests/ocean/unit/test_inertia_gravity_wave.py:120:        assert jnp.all(jnp.isfinite(state.u.data)), "u NaN after 10 steps"
tests/ocean/unit/test_inertia_gravity_wave.py:121:        assert jnp.all(jnp.isfinite(state.v.data)), "v NaN after 10 steps"
tests/ocean/unit/test_inertia_gravity_wave.py:155:        assert np.all(np.isfinite(eta_exact))
tests/ocean/unit/test_ocean_fv.py:4:and differentiable results.
tests/ocean/unit/test_ocean_fv.py:54:    def test_tendencies_finite(self, state, grid, z_coord, config):
tests/ocean/unit/test_ocean_fv.py:57:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/ocean/unit/test_ocean_fv.py:58:        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
tests/ocean/unit/test_ocean_fv.py:59:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/ocean/unit/test_ocean_fv.py:60:        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
tests/ocean/unit/test_ocean_fv.py:61:        assert jnp.all(jnp.isfinite(tend.deta_dt.data))
tests/ocean/unit/test_ocean_fv.py:83:        assert jnp.all(jnp.isfinite(s1.u.data))
tests/ocean/unit/test_ocean_fv.py:84:        assert jnp.all(jnp.isfinite(s1.T.data))
tests/ocean/unit/test_ocean_fv.py:85:        assert jnp.all(jnp.isfinite(s1.eta.data))
tests/ocean/unit/test_ocean_fv.py:112:        assert jnp.all(jnp.isfinite(s.u.data))
tests/ocean/unit/test_ocean_fv.py:113:        assert jnp.all(jnp.isfinite(s.T.data))
tests/ocean/unit/test_ocean_fv.py:114:        assert jnp.all(jnp.isfinite(s.eta.data))
tests/ocean/unit/test_ocean_fv.py:116:    def test_differentiable(self, grid, z_coord, config, state):
tests/ocean/unit/test_ocean_fv.py:117:        """C-D grid ocean should be differentiable through tendencies."""
tests/ocean/unit/test_ocean_fv.py:125:        grad_fn = jax.grad(loss_fn)
tests/ocean/unit/test_ocean_fv.py:126:        g = grad_fn(state.eta.data)
tests/ocean/unit/test_ocean_fv.py:127:        assert jnp.all(jnp.isfinite(g))
tests/ocean/unit/test_freshwater.py:444:        assert jnp.all(jnp.isfinite(state1.eta.data))
tests/ocean/unit/test_freshwater.py:445:        assert jnp.all(jnp.isfinite(state1.S.data))
tests/ocean/unit/test_freshwater.py:557:        assert jnp.all(jnp.isfinite(state.eta.data))
tests/ocean/unit/test_freshwater.py:558:        assert jnp.all(jnp.isfinite(state.T.data))
tests/ocean/unit/test_freshwater.py:559:        assert jnp.all(jnp.isfinite(state.S.data))
tests/ocean/unit/test_freshwater.py:560:        assert jnp.all(jnp.isfinite(state.u.data))
tests/ocean/unit/test_freshwater.py:624:        assert jnp.all(jnp.isfinite(state1.eta.data))
tests/ocean/unit/test_freshwater.py:625:        assert jnp.all(jnp.isfinite(state1.S.data))
tests/ocean/unit/test_freshwater.py:742:        assert jnp.all(jnp.isfinite(state.eta.data))
tests/ocean/unit/test_freshwater.py:743:        assert jnp.all(jnp.isfinite(state.T.data))
tests/ocean/unit/test_freshwater.py:744:        assert jnp.all(jnp.isfinite(state.S.data))
tests/ocean/unit/test_freshwater.py:745:        assert jnp.all(jnp.isfinite(state.u.data))
tests/ocean/unit/test_freshwater.py:780:    def test_vsf_differentiable(self):
tests/ocean/unit/test_freshwater.py:781:        """Virtual salt flux should be differentiable w.r.t. precip."""
tests/ocean/unit/test_freshwater.py:795:        grad_fn = jax.grad(loss)
tests/ocean/unit/test_freshwater.py:796:        g = grad_fn(1e-4)
tests/ocean/unit/test_freshwater.py:797:        assert jnp.isfinite(g)
tests/ocean/unit/test_freshwater.py:800:    def test_eta_tendency_differentiable(self):
tests/ocean/unit/test_freshwater.py:801:        """Eta tendency should be differentiable."""
tests/ocean/unit/test_freshwater.py:814:        grad_fn = jax.grad(loss)
tests/ocean/unit/test_freshwater.py:815:        g = grad_fn(1e-4)
tests/ocean/unit/test_freshwater.py:816:        assert jnp.isfinite(g)
tests/ocean/unit/test_smagorinsky.py:193:        # Create 3D fields with different values per level
tests/ocean/unit/test_smagorinsky.py:211:            max_diff_u = float(jnp.max(jnp.abs(tu_3d[:, :, k] - tu_2d)))
tests/ocean/unit/test_smagorinsky.py:212:            max_diff_v = float(jnp.max(jnp.abs(tv_3d[:, :, k] - tv_2d)))
tests/ocean/unit/test_smagorinsky.py:213:            assert max_diff_u < 1e-14, f"3D/2D u mismatch at level {k}: {max_diff_u}"
tests/ocean/unit/test_smagorinsky.py:214:            assert max_diff_v < 1e-14, f"3D/2D v mismatch at level {k}: {max_diff_v}"
tests/ocean/unit/test_smagorinsky.py:259:    def test_jax_autodiff(self, latlon_grid):
tests/ocean/unit/test_smagorinsky.py:260:        """jax.grad flows through the Smagorinsky operator."""
tests/ocean/unit/test_smagorinsky.py:271:        grad_u, grad_v = jax.grad(loss_fn, argnums=(0, 1))(u, v)
tests/ocean/unit/test_smagorinsky.py:273:        # Gradients should be finite
tests/ocean/unit/test_smagorinsky.py:274:        assert jnp.all(jnp.isfinite(grad_u)), "grad_u contains NaN/Inf"
tests/ocean/unit/test_smagorinsky.py:275:        assert jnp.all(jnp.isfinite(grad_v)), "grad_v contains NaN/Inf"
tests/ocean/unit/test_smagorinsky.py:278:        assert jnp.max(jnp.abs(grad_u)) > 0, "grad_u is all zeros"
tests/ocean/unit/test_smagorinsky.py:279:        assert jnp.max(jnp.abs(grad_v)) > 0, "grad_v is all zeros"
tests/ocean/unit/test_smagorinsky.py:446:        assert jnp.all(jnp.isfinite(tend)), "Tendency contains NaN/Inf"
tests/ocean/unit/test_smagorinsky.py:448:    def test_jax_autodiff(self, mpas_mesh):
tests/ocean/unit/test_smagorinsky.py:449:        """jax.grad flows through the MPAS Smagorinsky operator."""
tests/ocean/unit/test_smagorinsky.py:457:        grad_u = jax.grad(loss_fn)(u)
tests/ocean/unit/test_smagorinsky.py:459:        assert jnp.all(jnp.isfinite(grad_u)), "grad_u contains NaN/Inf"
tests/ocean/unit/test_smagorinsky.py:460:        assert jnp.max(jnp.abs(grad_u)) > 0, "grad_u is all zeros"
tests/ocean/unit/test_smagorinsky.py:486:            max_diff = float(jnp.max(jnp.abs(tend_3d[:, k:k+1] - tend_1d)))
tests/ocean/unit/test_smagorinsky.py:487:            assert max_diff < 1e-14, (
tests/ocean/unit/test_smagorinsky.py:488:                f"Multi-level/single-level mismatch at level {k}: {max_diff}"
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:15:  with microphysics produces a finite ``q_v`` decrease relative to the
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:19:  differently than with either scheme alone).
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:20:* ``jax.grad`` through one bridge call w.r.t. an initial-q amplitude
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:21:  remains finite — full AD compatibility.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:49:from legoesm.atmosphere.physics.gravity_wave_drag.config import (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:82:def _proper_hyperdiff(grid):
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:132:        assert bool(jnp.all(jnp.isfinite(out.T_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:138:        differs from the dT_hat output when state.tracers has q_v set
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:154:        assert bool(jnp.all(jnp.isfinite(dq_v_wet)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:181:        max_diff = float(jnp.max(jnp.abs(dq_v_wet - dq_v_dry)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:182:        assert max_diff > 1e-12, (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:185:            f"q_v=0 inputs (max diff {max_diff})."
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:187:        # Same plumbing test on the latent-heating signal — different
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:188:        # q_v should drive different condensation, hence different
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:190:        dT_diff = float(
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:193:        assert dT_diff > 1e-12, (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:195:            f"got identical dT_hat for wet vs q_v=0 (diff {dT_diff})."
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:290:        assert bool(jnp.all(jnp.isfinite(out.T_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:372:        assert bool(jnp.all(jnp.isfinite(dq_v)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:398:    def test_full_step_preserves_finite_state(
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:402:        leave all prognostic fields finite."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:410:            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:414:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:421:        assert bool(jnp.all(jnp.isfinite(new_state.vor_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:422:        assert bool(jnp.all(jnp.isfinite(new_state.div_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:423:        assert bool(jnp.all(jnp.isfinite(new_state.T_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:424:        assert bool(jnp.all(jnp.isfinite(new_state.lnps_hat.data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:426:        assert bool(jnp.all(jnp.isfinite(new_state.tracers["q_v"].data)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:431:        """q_v with microphysics differs from q_v without microphysics
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:444:            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:448:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:463:        assert bool(jnp.all(jnp.isfinite(qv_w_micro)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:464:        assert bool(jnp.all(jnp.isfinite(qv_no_micro)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:468:        diff = float(jnp.max(jnp.abs(qv_w_micro - qv_no_micro)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:469:        assert diff > 0.0, (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:471:            "dycore-only baseline; got zero diff — bridge probably "
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:497:        """Convection + microphysics together produce a different
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:501:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:512:            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:522:            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:532:            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:546:        # All finite.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:547:        assert bool(jnp.all(jnp.isfinite(qv_conv)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:548:        assert bool(jnp.all(jnp.isfinite(qv_micro)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:549:        assert bool(jnp.all(jnp.isfinite(qv_both)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:550:        # The combined run differs from BOTH single-scheme runs (i.e.,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:553:        diff_to_conv = float(jnp.max(jnp.abs(qv_both - qv_conv)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:554:        diff_to_micro = float(jnp.max(jnp.abs(qv_both - qv_micro)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:555:        assert diff_to_conv > 0.0, (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:556:            "Combined run should differ from convection-only — "
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:560:        assert diff_to_micro > 0.0, (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:561:            "Combined run should differ from microphysics-only — "
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:572:    def test_grad_through_bridge_call(self, grid, sigma_coord, rest_state):
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:573:        """``jax.grad`` flows through one microphysics-bridge call w.r.t.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:574:        the q_v amplitude (no NotImplementedError, no NaN gradient)."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:609:        g = jax.grad(loss)(jnp.array(0.95))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:610:        assert bool(jnp.isfinite(g))
tests/unit/test_timestepping.py:62:        assert jnp.isfinite(result.x).all()
tests/unit/test_timestepping.py:63:        assert jnp.isfinite(result.v).all()
tests/unit/test_timestepping.py:65:    def test_differentiable(self):
tests/unit/test_timestepping.py:66:        """jax.grad should work through SSP-RK3."""
tests/unit/test_timestepping.py:73:        grads = jax.grad(loss)(y0)
tests/unit/test_timestepping.py:74:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_timestepping.py:249:        assert jnp.isfinite(result.x).all()
tests/unit/test_timestepping.py:250:        assert jnp.isfinite(result.v).all()
tests/unit/test_timestepping.py:252:    def test_differentiable(self):
tests/unit/test_timestepping.py:253:        """jax.grad should work through SSP-RK54."""
tests/unit/test_timestepping.py:260:        grads = jax.grad(loss)(y0)
tests/unit/test_timestepping.py:261:        assert jnp.all(jnp.isfinite(grads))
tests/unit/test_step_cache.py:99:    """StepCacheKey correctly discriminates different call shapes."""
tests/unit/test_step_cache.py:108:    def test_different_physics_flag_different_key(self):
tests/unit/test_step_cache.py:116:    def test_different_state_type_different_key(self):
tests/unit/test_step_cache.py:124:    def test_different_shape_different_key(self):
tests/unit/test_step_cache.py:125:        """Different resolution produces different leaf_meta."""
tests/unit/test_step_cache.py:191:    def test_different_dt_no_recompile(self):
tests/unit/test_step_cache.py:310:    def test_cached_step_different_dt_values(self):
tests/unit/test_step_cache.py:350:        # Result should be finite and larger than initial T
tests/unit/test_step_cache.py:353:        assert np.all(np.isfinite(T_final))
tests/ocean/unit/test_backscatter.py:146:        coefficient driving each is different (|D| vs √E) we only check
tests/ocean/unit/test_backscatter.py:163:        # differs.  Here we just verify both are non-trivial.
tests/ocean/unit/test_backscatter.py:192:    def test_grad_finite(self, latlon_grid):
tests/ocean/unit/test_backscatter.py:204:        du, dv = jax.grad(loss)((u, v))
tests/ocean/unit/test_backscatter.py:205:        assert jnp.all(jnp.isfinite(du))
tests/ocean/unit/test_backscatter.py:206:        assert jnp.all(jnp.isfinite(dv))
tests/ocean/unit/test_backscatter.py:284:    def test_shape_and_finite(self, mpas_mesh):
tests/ocean/unit/test_backscatter.py:293:        assert jnp.all(jnp.isfinite(tend))
tests/ocean/unit/test_backscatter.py:321:    def test_grad_finite(self, mpas_mesh):
tests/ocean/unit/test_backscatter.py:332:        du = jax.grad(loss)(u)
tests/ocean/unit/test_backscatter.py:333:        assert jnp.all(jnp.isfinite(du))
tests/ocean/unit/test_leith.py:195:        assert jnp.all(jnp.isfinite(tu))
tests/ocean/unit/test_leith.py:196:        assert jnp.all(jnp.isfinite(tv))
tests/ocean/unit/test_leith.py:269:    def test_tendency_grad_finite(self, latlon_grid):
tests/ocean/unit/test_leith.py:270:        """JAX grad through the biharmonic tendency must be finite."""
tests/ocean/unit/test_leith.py:280:        du, dv = jax.grad(loss)((u, v))
tests/ocean/unit/test_leith.py:281:        assert jnp.all(jnp.isfinite(du))
tests/ocean/unit/test_leith.py:282:        assert jnp.all(jnp.isfinite(dv))
tests/ocean/unit/test_leith.py:322:    def test_tendency_shapes_and_finite(self, mpas_mesh):
tests/ocean/unit/test_leith.py:326:        assert jnp.all(jnp.isfinite(tend))
tests/ocean/unit/test_leith.py:340:            max_diff = float(jnp.max(jnp.abs(tend_3d[:, k:k + 1] - tend_1d)))
tests/ocean/unit/test_leith.py:341:            assert max_diff < 1e-14, (
tests/ocean/unit/test_leith.py:342:                f"Multi-level/single-level mismatch at level {k}: {max_diff}"
tests/ocean/unit/test_leith.py:345:    def test_tendency_grad_finite(self, mpas_mesh):
tests/ocean/unit/test_leith.py:351:        du = jax.grad(loss)(u)
tests/ocean/unit/test_leith.py:352:        assert jnp.all(jnp.isfinite(du))
tests/ocean/unit/test_leith.py:371:    def test_both_respond_to_large_velocity_gradients(
tests/unit/test_spectral_dycores_comprehensive.py:8:  6. Convergence rates (Williamson TC2, operator, hyperdiffusion)
tests/unit/test_spectral_dycores_comprehensive.py:32:    spectral_hyperdiffusion,
tests/unit/test_spectral_dycores_comprehensive.py:33:    spectral_hyperdiffusion_3d,
tests/unit/test_spectral_dycores_comprehensive.py:63:def _proper_hyperdiff(grid):
tests/unit/test_spectral_dycores_comprehensive.py:64:    """Resolution-appropriate hyperdiffusion coefficient."""
tests/unit/test_spectral_dycores_comprehensive.py:126:    def test_1a_curl_grad_phi_zero(self, grid_t21):
tests/unit/test_spectral_dycores_comprehensive.py:127:        """curl(grad(phi)) = 0 for any scalar phi."""
tests/unit/test_spectral_dycores_comprehensive.py:135:        # grad(phi) in spectral space -> vorticity and divergence
tests/unit/test_spectral_dycores_comprehensive.py:136:        # grad(phi) has zero curl component. We compute the gradient
tests/unit/test_spectral_dycores_comprehensive.py:138:        # which means the gradient is purely divergent (curl-free).
tests/unit/test_spectral_dycores_comprehensive.py:143:        # Recover u_cos, v_cos (the gradient components times cos(lat))
tests/unit/test_spectral_dycores_comprehensive.py:146:        # Now compute curl of (u,v). For a gradient field, curl should be 0.
tests/unit/test_spectral_dycores_comprehensive.py:158:        assert max_curl < 1e-10, f"curl(grad(phi)) = {max_curl:.3e}, expected < 1e-10"
tests/unit/test_spectral_dycores_comprehensive.py:190:    def test_1c_laplacian_equals_div_grad(self, grid_t21):
tests/unit/test_spectral_dycores_comprehensive.py:191:        """Laplacian = div(grad) for a scalar field."""
tests/unit/test_spectral_dycores_comprehensive.py:203:        # Method 2: div(grad) via transform chain
tests/unit/test_spectral_dycores_comprehensive.py:204:        # grad(phi): use velocity potential chi = ilap * div_hat,
tests/unit/test_spectral_dycores_comprehensive.py:206:        # Actually, grad components via uv_from_vordiv with vor=0, div=lap*phi
tests/unit/test_spectral_dycores_comprehensive.py:207:        div_hat_grad = spectral_laplacian(grid, phi_hat)  # = lap * phi_hat
tests/unit/test_spectral_dycores_comprehensive.py:209:        u_cos, v_cos = uv_from_vordiv(grid, vor_hat_zero, div_hat_grad)
tests/unit/test_spectral_dycores_comprehensive.py:215:        div_grad_hat = (im_over_a * sh_analysis_oc2(grid, u_cos)
tests/unit/test_spectral_dycores_comprehensive.py:217:        div_grad_grid = sh_synthesis(grid, div_grad_hat)
tests/unit/test_spectral_dycores_comprehensive.py:219:        rel_err = _relative_l2(div_grad_grid, lap_grid_spectral, grid)
tests/unit/test_spectral_dycores_comprehensive.py:220:        assert rel_err < 1e-10, f"Laplacian vs div(grad): relative L2 = {rel_err:.3e}"
tests/unit/test_spectral_dycores_comprehensive.py:252:        """grad(constant) = 0 and Laplacian(constant) = 0."""
tests/unit/test_spectral_dycores_comprehensive.py:266:        # grad uses div_hat = lap * const_hat = 0 and vor_hat = 0
tests/unit/test_spectral_dycores_comprehensive.py:268:        assert _linf_norm(u_cos) < 1e-10, f"grad_u(constant) = {_linf_norm(u_cos):.3e}"
tests/unit/test_spectral_dycores_comprehensive.py:269:        assert _linf_norm(v_cos) < 1e-10, f"grad_v(constant) = {_linf_norm(v_cos):.3e}"
tests/unit/test_spectral_dycores_comprehensive.py:323:            hyperdiff_coeff=0.0,  # no diffusion for balance test
tests/unit/test_spectral_dycores_comprehensive.py:356:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:402:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:440:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:462:        """SW total energy conservation in TC2 (no diffusion) for 500 steps."""
tests/unit/test_spectral_dycores_comprehensive.py:466:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:488:        """SW potential enstrophy conservation in TC2 (no diffusion) for 500 steps."""
tests/unit/test_spectral_dycores_comprehensive.py:492:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:519:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:548:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:596:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:624:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:642:        rel_diff = float(jnp.max(jnp.abs(h_north - h_south)) / jnp.max(jnp.abs(h)))
tests/unit/test_spectral_dycores_comprehensive.py:643:        assert rel_diff < 1e-12, (
tests/unit/test_spectral_dycores_comprehensive.py:644:            f"Hemispheric symmetry broken: relative diff = {rel_diff:.3e}"
tests/unit/test_spectral_dycores_comprehensive.py:654:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:717:                hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:806:    def test_6c_hyperdiffusion_convergence(self, grid_t21):
tests/unit/test_spectral_dycores_comprehensive.py:820:        result = spectral_hyperdiffusion(grid, coeffs, nu, order)
tests/unit/test_spectral_dycores_comprehensive.py:829:            f"Hyperdiffusion eigenvalue error: {rel_err:.3e}"
tests/unit/test_spectral_dycores_comprehensive.py:876:        # SW uses: uv_from_vordiv, spectral_hyperdiffusion
tests/unit/test_spectral_dycores_comprehensive.py:878:        assert spectral_sw.spectral_hyperdiffusion is gaussian.spectral_hyperdiffusion
tests/unit/test_spectral_dycores_comprehensive.py:880:        # PE uses: uv_from_vordiv_3d, spectral_hyperdiffusion_3d
tests/unit/test_spectral_dycores_comprehensive.py:882:        assert spectral_pe.spectral_hyperdiffusion_3d is gaussian.spectral_hyperdiffusion_3d
tests/unit/test_spectral_dycores_comprehensive.py:884:        # NH uses: uv_from_vordiv_3d, spectral_hyperdiffusion_3d
tests/unit/test_spectral_dycores_comprehensive.py:886:        assert spectral_nh.spectral_hyperdiffusion_3d is gaussian.spectral_hyperdiffusion_3d
tests/unit/test_spectral_dycores_comprehensive.py:900:        sw_config = SpectralSWConfig(hyperdiff_coeff=0.0)
tests/unit/test_spectral_dycores_comprehensive.py:919:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:970:    def test_3d_hyperdiffusion_shape(self, grid_t10):
tests/unit/test_spectral_dycores_comprehensive.py:971:        """3D hyperdiffusion should return correct shape."""
tests/unit/test_spectral_dycores_comprehensive.py:978:        result = spectral_hyperdiffusion_3d(grid, coeffs, nu, order=2)
tests/unit/test_spectral_dycores_comprehensive.py:987:            "n=0 mode should not be damped by hyperdiffusion"
tests/unit/test_spectral_dycores_comprehensive.py:1021:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:1067:            hyperdiff_coeff=0.0,
tests/unit/test_spectral_dycores_comprehensive.py:1094:        Energy is expected to drift ~0.1% due to hyperdiffusion.
tests/unit/test_spectral_dycores_comprehensive.py:1098:        nu = _proper_hyperdiff(grid)
tests/unit/test_spectral_dycores_comprehensive.py:1100:            hyperdiff_coeff=nu,
tests/unit/test_spectral_dycores_comprehensive.py:1126:        # Energy drift with diffusion: < 1% over 15 days
tests/unit/test_spectral_dycores_comprehensive.py:1132:        assert bool(jnp.all(jnp.isfinite(h))), "TC5 15-day: non-finite height"
tests/unit/test_spectral_dycores_comprehensive.py:1148:        nu = _proper_hyperdiff(grid)
tests/unit/test_spectral_dycores_comprehensive.py:1150:            hyperdiff_coeff=nu,
tests/unit/test_spectral_dycores_comprehensive.py:1151:            hyperdiff_order=2,
tests/unit/test_spectral_dycores_comprehensive.py:1188:        nu = _proper_hyperdiff(grid)
tests/unit/test_spectral_dycores_comprehensive.py:1190:            hyperdiff_coeff=nu,
tests/unit/test_spectral_dycores_comprehensive.py:1227:        nu = _proper_hyperdiff(grid)
tests/unit/test_spectral_dycores_comprehensive.py:1233:            hyperdiff_coeff=nu,
tests/unit/test_spectral_dycores_comprehensive.py:1234:            hyperdiff_order=2,
tests/unit/test_spectral_dycores_comprehensive.py:1269:    def test_nh_gravity_wave_speed(self):
tests/unit/test_spectral_dycores_comprehensive.py:1270:        """NH gravity wave from Gaussian perturbation should propagate at c = sqrt(g*H).
tests/unit/test_spectral_dycores_comprehensive.py:1272:        This tests that the acoustic/gravity wave coupling in the split-explicit
tests/unit/test_spectral_dycores_comprehensive.py:1287:        nu = _proper_hyperdiff(grid)
tests/unit/test_spectral_dycores_comprehensive.py:1289:            hyperdiff_coeff=nu,
tests/unit/test_spectral_dycores_comprehensive.py:1290:            hyperdiff_order=2,
tests/unit/test_spectral_dycores_comprehensive.py:1318:        # Step forward - the perturbation should excite acoustic/gravity waves
tests/unit/test_spectral_dycores_comprehensive.py:1324:        # Check 1: all fields still finite (no blowup from split-explicit)
tests/unit/test_spectral_dycores_comprehensive.py:1329:        assert jnp.all(jnp.isfinite(theta_p_final)), (
tests/unit/test_spectral_dycores_comprehensive.py:1330:            "NH gravity wave: theta' has non-finite values after 100 steps"
tests/unit/test_spectral_dycores_comprehensive.py:1332:        assert jnp.all(jnp.isfinite(rho_p_final)), (
tests/unit/test_spectral_dycores_comprehensive.py:1333:            "NH gravity wave: rho' has non-finite values after 100 steps"
tests/unit/test_spectral_dycores_comprehensive.py:1335:        assert jnp.all(jnp.isfinite(w_final)), (
tests/unit/test_spectral_dycores_comprehensive.py:1336:            "NH gravity wave: w has non-finite values after 100 steps"
tests/unit/test_spectral_dycores_comprehensive.py:1343:            f"NH gravity wave: no w response excited. max|w| = {w_max:.3e}. "
tests/unit/test_spectral_dycores_comprehensive.py:1351:            f"NH gravity wave: perturbation grew. "
tests/ocean/unit/test_mpas_physics.py:151:            enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0, K_bg=1e-5),
tests/ocean/unit/test_mpas_physics.py:161:    the enhanced_diffusion branch must detect.
tests/ocean/unit/test_mpas_physics.py:169:    """Convective adjustment routing for ``scheme='enhanced_diffusion'``."""
tests/ocean/unit/test_mpas_physics.py:177:    def test_enhanced_diffusion_fires_on_unstable_column(
tests/ocean/unit/test_mpas_physics.py:183:            _convection_physics_config("enhanced_diffusion"))
tests/ocean/unit/test_mpas_physics.py:190:            "enhanced_diffusion convection must produce non-zero dT/dt "
tests/ocean/unit/test_mpas_physics.py:202:                scheme="enhanced_diffusion",
tests/ocean/unit/test_mpas_physics.py:203:                enhanced_diffusion=EnhancedDiffusionConfig(
tests/ocean/unit/test_mpas_physics.py:223:            _convection_physics_config("enhanced_diffusion"))
tests/ocean/unit/test_bathymetry.py:333:        assert np.all(np.isfinite(result))
tests/ocean/unit/test_bathymetry.py:486:    def test_cubed_sphere_state_finite(self, small_grid, z_coord):
tests/ocean/unit/test_bathymetry.py:487:        """All state fields should be finite."""
tests/ocean/unit/test_bathymetry.py:492:            assert jnp.all(jnp.isfinite(data)), f"{field_name} has non-finite values"
tests/ocean/unit/test_bathymetry.py:509:    def test_differentiable(self, small_grid, z_coord):
tests/ocean/unit/test_bathymetry.py:510:        """Resulting state should be usable in JAX grad computations."""
tests/ocean/unit/test_bathymetry.py:517:        grad_T = jax.grad(loss)(state.T.data)
tests/ocean/unit/test_bathymetry.py:518:        assert jnp.all(jnp.isfinite(grad_T))
tests/ocean/unit/test_bathymetry.py:519:        assert grad_T.shape == state.T.data.shape
tests/ocean/unit/test_advection_dst3.py:8:5. Comparison with TVD Van Leer: DST-3 should be less diffusive
tests/ocean/unit/test_advection_dst3.py:162:    def test_less_diffusive_than_upwind(self):
tests/ocean/unit/test_advection_dst3.py:419:    """Compare DST-3 against TVD Van Leer: DST-3 should be less diffusive."""
tests/ocean/unit/test_advection_dst3.py:449:        # DST-3 should be at least somewhat less diffusive
tests/ocean/unit/test_eta_floor.py:47:    assert jnp.all(jnp.isfinite(out))
tests/ocean/unit/test_eta_floor.py:90:    assert jnp.all(jnp.isfinite(out))
tests/ocean/unit/test_no_scheme_duplication.py:11:because the grid-specific stencils legitimately differ).
tests/unit/test_convection_triggers.py:7:discontinuities with differentiable approximations.  These tests pin
tests/unit/test_convection_triggers.py:16:* finite, non-zero gradients at the threshold (the AD-safety
tests/unit/test_convection_triggers.py:38:    assert jnp.all(jnp.diff(out) > 0)
tests/unit/test_convection_triggers.py:52:def test_smooth_step_grad_at_zero_is_sharpness_over_four():
tests/unit/test_convection_triggers.py:55:        g = float(jax.grad(lambda x: T.smooth_step(x, s))(jnp.asarray(0.0)))
tests/unit/test_convection_triggers.py:72:    """smooth_max(a, b, s) >= max(a, b) for all finite s > 0."""
tests/unit/test_convection_triggers.py:134:def test_smooth_positive_part_grad_finite_everywhere():
tests/unit/test_convection_triggers.py:135:    grad_fn = jax.grad(lambda x: T.smooth_positive_part(x, 5.0))
tests/unit/test_convection_triggers.py:137:        g = float(grad_fn(jnp.asarray(x)))
tests/unit/test_convection_triggers.py:138:        assert np.isfinite(g)
tests/unit/test_convection_triggers.py:174:    """Surface-last profile with a single linear gradient.
tests/unit/test_convection_triggers.py:220:def test_smooth_lowest_crossing_grad_finite_at_threshold():
tests/unit/test_convection_triggers.py:221:    """Gradient of the diagnosed index w.r.t. the threshold is finite
tests/unit/test_convection_triggers.py:229:    g = float(jax.grad(f)(jnp.asarray(0.5)))
tests/unit/test_convection_triggers.py:230:    assert np.isfinite(g)
tests/unit/test_convection_triggers.py:243:def test_cape_trigger_equals_smooth_step_of_difference():
tests/unit/test_convection_triggers.py:252:def test_cape_trigger_grad_at_threshold_finite_nonzero():
tests/unit/test_convection_triggers.py:253:    """``jax.grad`` w.r.t. each of the three arguments is finite at
tests/unit/test_convection_triggers.py:255:    naïve hard trigger (``cape > threshold``) zeros the gradient."""
tests/unit/test_convection_triggers.py:260:    g_cape = float(jax.grad(lambda c: T.cape_trigger(c, threshold, sharpness))(cape))
tests/unit/test_convection_triggers.py:261:    g_thr = float(jax.grad(lambda t: T.cape_trigger(cape, t, sharpness))(jnp.asarray(threshold)))
tests/unit/test_convection_triggers.py:262:    g_s = float(jax.grad(lambda s: T.cape_trigger(cape, threshold, s))(jnp.asarray(sharpness)))
tests/unit/test_convection_triggers.py:265:        assert np.isfinite(g), f"grad w.r.t. {label} not finite"
tests/ocean/unit/test_advection_fct_zalesak.py:5:limiter applied to anti-diffusive face fluxes.  These tests cover:
tests/ocean/unit/test_advection_fct_zalesak.py:13:* AD compatibility: ``jax.grad`` of a scalar built from the limited
tests/ocean/unit/test_advection_fct_zalesak.py:14:  divergence is finite — needed for tracer-tuning workflows.
tests/ocean/unit/test_advection_fct_zalesak.py:163:        """When q_min, q_max are infinite (Q+, Q- → ∞) every face passes
tests/ocean/unit/test_advection_fct_zalesak.py:164:        the full anti-diffusive flux — α = 1 everywhere.  This is the
tests/ocean/unit/test_advection_fct_zalesak.py:184:        For an anti-diffusive u-face flux ``F_u > 0`` whose RECEIVER is
tests/ocean/unit/test_advection_fct_zalesak.py:201:        # All anti-diffusive u-fluxes positive (eastward).
tests/ocean/unit/test_advection_fct_zalesak.py:225:        anti-diffusive flux that would change them.  Interior faces
tests/ocean/unit/test_advection_fct_zalesak.py:257:    """``jax.grad`` through the limited divergence must give finite
tests/ocean/unit/test_advection_fct_zalesak.py:260:    def test_grad_through_fct_advection_finite(self, grid_small, smooth_state):
tests/ocean/unit/test_advection_fct_zalesak.py:269:        g = jax.grad(loss)(tracer)
tests/ocean/unit/test_advection_fct_zalesak.py:271:        assert bool(jnp.all(jnp.isfinite(g)))
tests/unit/test_physics_state_migration.py:35:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/unit/test_physics_state_migration.py:48:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/unit/test_physics_state_migration.py:81:    """``mass_flux`` packs ``M_c_init`` at the surface-adjacent slot,
tests/unit/test_physics_state_migration.py:88:    # Surface-adjacent slot has the initial value.
tests/unit/test_physics_state_migration.py:98:        "Slots above the surface-adjacent index must be zero in the "
tests/unit/test_physics_state_migration.py:148:    assert jnp.allclose(ps_out.gwd_spectrum, ps.gwd_spectrum)
tests/ocean/unit/test_shortwave_penetration.py:4:a finite, sane temperature tendency that integrates to ~Q_sw / (rho_0 c_sw)
tests/ocean/unit/test_shortwave_penetration.py:42:    jacobian = jnp.ones((nx, ny))                    # eta=0, J=1
tests/ocean/unit/test_shortwave_penetration.py:43:    return sw_down, dz_ref, z_half_ref, jacobian
tests/ocean/unit/test_shortwave_penetration.py:68:def test_shortwave_penetration_tendency_shape_and_finite():
tests/ocean/unit/test_shortwave_penetration.py:75:    assert jnp.all(jnp.isfinite(dT_dt))
tests/ocean/unit/test_shortwave_penetration.py:117:    assert jnp.all(jnp.isfinite(out))
tests/ocean/unit/test_biharmonic_vorticity.py:9:2. ``vertex_laplacian_3d`` matches sign convention of ordinary diffusion:
tests/ocean/unit/test_biharmonic_vorticity.py:68:    assert jnp.all(jnp.isfinite(lap))
tests/ocean/unit/test_biharmonic_vorticity.py:74:    negative at the bump centre (standard diffusion sign)."""
tests/ocean/unit/test_biharmonic_vorticity.py:100:    assert jnp.all(jnp.isfinite(out))
tests/ocean/unit/test_biharmonic_vorticity.py:159:    assert jnp.all(jnp.isfinite(out))
tests/ocean/unit/test_biharmonic_vorticity.py:180:    from legoesm.core.operators_voronoi import gradient_edge_3d
tests/ocean/unit/test_biharmonic_vorticity.py:181:    u_coarse_raw = gradient_edge_3d(jnp.asarray(phi_cell_coarse), mesh)
tests/ocean/unit/test_biharmonic_vorticity.py:230:def test_biharmonic_vorticity_grad_smoke(mesh):
tests/ocean/unit/test_biharmonic_vorticity.py:231:    """jax.grad through the operator works (AD-compatibility)."""
tests/ocean/unit/test_biharmonic_vorticity.py:236:    g = jax.grad(loss)(u)
tests/ocean/unit/test_biharmonic_vorticity.py:238:    assert jnp.all(jnp.isfinite(g))
tests/ocean/unit/test_biharmonic_vorticity.py:239:    # Trivially, gradient should be non-zero for a non-trivial loss.
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:30:    gradient_x_3d as _gradient_x_3d,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:228:        diff = Phi_mountain - Phi_flat
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:229:        assert jnp.allclose(diff, 1000.0, rtol=1e-5)
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:259:        instead of 1. This is standard for GCMs with finite model tops.
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:300:    def test_vertical_advection_finite(self, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:301:        """Vertical advection should produce finite values."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:310:        assert jnp.all(jnp.isfinite(tendency))
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:357:    def test_gradient_3d_shape(self, grid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:358:        """3D gradient should have correct shape."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:361:        gx = _gradient_x_3d(f, grid)
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:364:    def test_vorticity_3d_finite(self, grid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:365:        """3D vorticity should be finite."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:372:        assert jnp.all(jnp.isfinite(vort))
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:386:        - No pressure gradients
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:393:        config = PrimitiveEquationConfig(hyperdiff_coeff=0.0)
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:408:    def test_tendencies_finite(self, grid, cdgrid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:409:        """Tendencies should be finite for any reasonable state."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:411:        config = PrimitiveEquationConfig(hyperdiff_coeff=0.0)
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:415:        assert jnp.all(jnp.isfinite(tend.du_dt.data)), "du_dt not finite"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:416:        assert jnp.all(jnp.isfinite(tend.dv_dt.data)), "dv_dt not finite"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:417:        assert jnp.all(jnp.isfinite(tend.dT_dt.data)), "dT_dt not finite"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:418:        assert jnp.all(jnp.isfinite(tend.dp_s_dt.data)), "dp_s_dt not finite"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:420:    def test_tendencies_with_hyperdiffusion(self, grid, cdgrid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:421:        """Tendencies with hyperdiffusion should be finite."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:423:        config = PrimitiveEquationConfig(hyperdiff_coeff=1e15)
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:427:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:428:        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:429:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:434:        config = PrimitiveEquationConfig(hyperdiff_coeff=0.0)
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:453:    def test_single_step_finite(self, grid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:454:        """A single time step should produce finite results."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:457:            hyperdiff_coeff=0.0,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:464:        assert jnp.all(jnp.isfinite(state_new.u.data)), "u not finite after step"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:465:        assert jnp.all(jnp.isfinite(state_new.v.data)), "v not finite after step"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:466:        assert jnp.all(jnp.isfinite(state_new.T.data)), "T not finite after step"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:467:        assert jnp.all(jnp.isfinite(state_new.p_s.data)), "p_s not finite after step"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:473:            hyperdiff_coeff=0.0,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:490:            hyperdiff_coeff=0.0,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:501:        # in infinite precision, but float32 accumulation rounds off
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:532:    def test_mass_fixer_preserves_gradients(self, grid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:533:        """Mass fixer should be differentiable."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:543:        grads = jax.grad(loss)(state.p_s.data)
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:544:        assert jnp.all(jnp.isfinite(grads))
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:552:    """Tests that the PE model is differentiable."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:554:    def test_grad_through_tendencies(self, grid, cdgrid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:555:        """jax.grad should work through tendency computation."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:557:        config = PrimitiveEquationConfig(hyperdiff_coeff=0.0)
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:564:        grads = jax.grad(loss)(state.u.data)
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:565:        assert jnp.all(jnp.isfinite(grads)), "Gradients through tendencies are not finite"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:567:    def test_grad_through_single_step(self, grid, sigma):
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:568:        """jax.grad should work through a single model step."""
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:571:            hyperdiff_coeff=0.0,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:581:        grads = jax.grad(loss)(state.u.data)
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:582:        assert jnp.all(jnp.isfinite(grads)), "Gradients through model step are not finite"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:714:                hyperdiff_coeff=1e18,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:735:        assert jnp.all(jnp.isfinite(T)), "Non-finite temperatures detected"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:745:                hyperdiff_coeff=1e18,
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:766:        assert jnp.all(jnp.isfinite(T)), "Non-finite temperatures detected"
tests/unit/test_coupler.py:116:def test_land_step_finite():
tests/unit/test_coupler.py:117:    """Land model produces finite outputs."""
tests/unit/test_coupler.py:124:    assert jnp.all(jnp.isfinite(new_state.T_soil.data))
tests/unit/test_coupler.py:125:    assert jnp.all(jnp.isfinite(new_state.W_bucket.data))
tests/unit/test_coupler.py:126:    assert jnp.all(jnp.isfinite(resp.shflx))
tests/unit/test_coupler.py:127:    assert jnp.all(jnp.isfinite(resp.lhflx))
tests/unit/test_coupler.py:128:    assert jnp.all(jnp.isfinite(resp.lw_up))
tests/unit/test_coupler.py:161:def test_sea_ice_step_finite():
tests/unit/test_coupler.py:162:    """Sea ice model produces finite outputs."""
tests/unit/test_coupler.py:173:    assert jnp.all(jnp.isfinite(new_state.h_ice.data))
tests/unit/test_coupler.py:174:    assert jnp.all(jnp.isfinite(new_state.T_ice.data))
tests/unit/test_coupler.py:175:    assert jnp.all(jnp.isfinite(new_state.concentration.data))
tests/unit/test_coupler.py:176:    assert jnp.all(jnp.isfinite(resp.shflx))
tests/unit/test_coupler.py:227:def test_lake_step_finite():
tests/unit/test_coupler.py:228:    """Lake model produces finite outputs."""
tests/unit/test_coupler.py:235:    assert jnp.all(jnp.isfinite(new_state.T_epi.data))
tests/unit/test_coupler.py:236:    assert jnp.all(jnp.isfinite(new_state.T_hypo.data))
tests/unit/test_coupler.py:237:    assert jnp.all(jnp.isfinite(resp.shflx))
tests/unit/test_coupler.py:381:    # All outputs finite
tests/unit/test_coupler.py:382:    assert jnp.all(jnp.isfinite(blended.T_surface))
tests/unit/test_coupler.py:383:    assert jnp.all(jnp.isfinite(blended.shflx))
tests/unit/test_coupler.py:384:    assert jnp.all(jnp.isfinite(blended.lhflx))
tests/unit/test_coupler.py:385:    assert jnp.all(jnp.isfinite(blended.tau_x))
tests/unit/test_coupler.py:386:    assert jnp.all(jnp.isfinite(blended.lw_up))
tests/unit/test_coupler.py:393:    """Multiple coupler steps stay finite and accumulator grows."""
tests/unit/test_coupler.py:410:    assert jnp.all(jnp.isfinite(blended.T_surface))
tests/unit/test_coupler.py:508:    """Wind floor sqrt(u^2 + v^2 + U_min^2) is smooth (differentiable at zero)."""
tests/unit/test_coupler.py:514:    # At u=v=0 the gradient should be finite (not NaN)
tests/unit/test_coupler.py:515:    grad_u = jax.grad(lambda u: wind_speed(u, 0.0).sum())(jnp.array(0.0))
tests/unit/test_coupler.py:516:    grad_v = jax.grad(lambda v: wind_speed(0.0, v).sum())(jnp.array(0.0))
tests/unit/test_coupler.py:518:    assert jnp.isfinite(grad_u)
tests/unit/test_coupler.py:519:    assert jnp.isfinite(grad_v)
tests/unit/test_coupler.py:521:    assert jnp.abs(grad_u) < 1e-6
tests/unit/test_coupler.py:522:    assert jnp.abs(grad_v) < 1e-6
tests/unit/test_coupler.py:582:def test_slab_ocean_finite():
tests/unit/test_coupler.py:583:    """Slab ocean step produces all-finite outputs."""
tests/unit/test_coupler.py:592:    assert jnp.all(jnp.isfinite(new_state.T_sfc.data))
tests/unit/test_coupler.py:593:    assert jnp.all(jnp.isfinite(sst))
tests/unit/test_coupler.py:594:    assert jnp.all(jnp.isfinite(u_sfc))
tests/unit/test_coupler.py:595:    assert jnp.all(jnp.isfinite(v_sfc))
tests/unit/test_coupler.py:629:def test_two_layer_ocean_finite():
tests/unit/test_coupler.py:630:    """Two-layer ocean step produces all-finite outputs."""
tests/unit/test_coupler.py:639:    assert jnp.all(jnp.isfinite(new_state.T_sfc.data))
tests/unit/test_coupler.py:640:    assert jnp.all(jnp.isfinite(new_state.T_deep.data))
tests/unit/test_coupler.py:641:    assert jnp.all(jnp.isfinite(sst))
tests/unit/test_coupler.py:697:# Test differentiability
tests/unit/test_coupler.py:700:def test_coupler_differentiable():
tests/unit/test_coupler.py:701:    """Full coupler step is differentiable w.r.t. ocean SST."""
tests/unit/test_coupler.py:719:    grad_sst = jax.grad(loss)(ocean_sst)
tests/unit/test_coupler.py:720:    assert grad_sst.shape == SHAPE
tests/unit/test_coupler.py:721:    assert jnp.all(jnp.isfinite(grad_sst))
tests/unit/test_coupler.py:722:    # Non-zero gradient (shflx depends on SST via bulk formula)
tests/unit/test_coupler.py:723:    assert float(jnp.max(jnp.abs(grad_sst))) > 0.0
tests/unit/test_coupler.py:726:def test_coupler_differentiable_through_surface_state():
tests/unit/test_coupler.py:727:    """Coupler step is differentiable w.r.t. land soil temperature."""
tests/unit/test_coupler.py:760:    grad_T = jax.grad(loss)(T_soil)
tests/unit/test_coupler.py:761:    assert grad_T.shape == SHAPE
tests/unit/test_coupler.py:762:    assert jnp.all(jnp.isfinite(grad_T))
tests/unit/test_coupler.py:901:        A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0,
tests/unit/test_coupler.py:910:    assert jnp.all(jnp.isfinite(sst))
tests/unit/test_coupler.py:931:    assert jnp.all(jnp.isfinite(blended.T_surface))
tests/unit/test_coupler.py:932:    assert jnp.all(jnp.isfinite(blended.shflx))
tests/unit/test_coupler.py:959:    assert jnp.all(jnp.isfinite(sst))
tests/unit/test_coupler.py:978:    assert jnp.all(jnp.isfinite(blended.T_surface))
tests/unit/test_coupler.py:979:    assert jnp.all(jnp.isfinite(blended.shflx))
tests/ocean/unit/test_barotropic_cgrid.py:7:- Reduced diffusion (alpha=0.01 vs 0.05)
tests/ocean/unit/test_barotropic_cgrid.py:24:from legoesm.ocean.experiments.inertia_gravity_wave import (
tests/ocean/unit/test_barotropic_cgrid.py:55:        barotropic_diffusion_alpha=alpha,
tests/ocean/unit/test_barotropic_cgrid.py:60:    """C-grid solver should produce finite output."""
tests/ocean/unit/test_barotropic_cgrid.py:62:    def test_one_step_finite(self, grid, z_coord, igw_state):
tests/ocean/unit/test_barotropic_cgrid.py:67:        assert jnp.all(jnp.isfinite(state_new.eta.data)), "eta NaN"
tests/ocean/unit/test_barotropic_cgrid.py:68:        assert jnp.all(jnp.isfinite(state_new.u.data)), "u NaN"
tests/ocean/unit/test_barotropic_cgrid.py:69:        assert jnp.all(jnp.isfinite(state_new.v.data)), "v NaN"
tests/ocean/unit/test_barotropic_cgrid.py:78:        assert jnp.all(jnp.isfinite(state.eta.data)), "eta NaN after 10 steps"
tests/ocean/unit/test_barotropic_cgrid.py:79:        assert jnp.all(jnp.isfinite(state.u.data)), "u NaN after 10 steps"
tests/ocean/unit/test_barotropic_cgrid.py:106:    def test_low_diffusion_stable(self, grid, z_coord, igw_state):
tests/ocean/unit/test_barotropic_cgrid.py:114:        assert jnp.all(jnp.isfinite(state.eta.data))
tests/ocean/unit/test_barotropic_cgrid.py:115:        # Wave amplitude should be better preserved with lower diffusion
tests/ocean/unit/test_barotropic_cgrid.py:127:        # A-grid with its required higher diffusion
tests/ocean/unit/test_barotropic_cgrid.py:134:        # C-grid with lower diffusion
tests/ocean/unit/test_barotropic_cgrid.py:144:        # C-grid with lower diffusion should preserve more amplitude
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:25:from legoesm.atmosphere.physics.turbulence.vertical_diffusion import implicit_vertical_diffusion
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:80:            hyperdiff_coeff=0.0,
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:81:            hyperdiff_ps_coeff=0.0,
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:126:                hyperdiff_coeff=1e15,
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:152:        assert jnp.isfinite(jnp.array(ef))
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:227:        tv_prev = float(jnp.sum(jnp.abs(jnp.diff(phi_prev, axis=1))))
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:230:            phi_new = implicit_vertical_diffusion(
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:236:            tv_new = float(jnp.sum(jnp.abs(jnp.diff(phi_new, axis=1))))
tests/unit/test_issue_fixes.py:35:from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
tests/unit/test_issue_fixes.py:36:    implicit_vertical_diffusion,
tests/unit/test_issue_fixes.py:72:    near-zero tendencies from the dynamical core (no hyperdiffusion)."""
tests/unit/test_issue_fixes.py:80:        config = PrimitiveEquationConfig(hyperdiff_coeff=0.0)
tests/unit/test_issue_fixes.py:106:        config = PrimitiveEquationConfig(hyperdiff_coeff=0.0)
tests/unit/test_issue_fixes.py:113:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/unit/test_issue_fixes.py:122:    vertical diffusion, validating the Thomas algorithm sign convention."""
tests/unit/test_issue_fixes.py:132:        K_half = jnp.ones((ncol, nlev - 1)) * 10.0   # strong diffusivity
tests/unit/test_issue_fixes.py:139:        phi_new = implicit_vertical_diffusion(
tests/unit/test_issue_fixes.py:144:        assert jnp.all(jnp.isfinite(phi_new)), "Thomas solver produced NaN/Inf"
tests/unit/test_issue_fixes.py:146:        # 2. Peak should decrease (diffusion smooths)
tests/unit/test_issue_fixes.py:148:            "Spike was not smoothed by diffusion")
tests/unit/test_issue_fixes.py:162:        """A uniform field should remain unchanged after diffusion."""
tests/unit/test_issue_fixes.py:172:        phi_new = implicit_vertical_diffusion(
tests/unit/test_issue_fixes.py:176:            "Uniform field changed after diffusion")
tests/unit/test_issue_fixes.py:189:        def diffuse(phi):
tests/unit/test_issue_fixes.py:190:            return implicit_vertical_diffusion(
tests/unit/test_issue_fixes.py:194:        result = diffuse(phi)
tests/unit/test_issue_fixes.py:195:        assert jnp.all(jnp.isfinite(result))
tests/unit/test_issue_fixes.py:364:        """The correction should be a uniform additive shift (preserves gradients)."""
tests/unit/test_issue_fixes.py:382:    def test_mass_fixer_differentiable(self):
tests/unit/test_issue_fixes.py:383:        """The mass fixer should be differentiable via jax.grad."""
tests/unit/test_issue_fixes.py:395:        grads = jax.grad(loss)(state_old.p_s.data)
tests/unit/test_issue_fixes.py:396:        assert jnp.all(jnp.isfinite(grads)), "Gradients through mass fixer not finite"
tests/unit/test_issue_fixes.py:406:    def test_cubed_and_latlon_hybrid_forcing_finite(self):
tests/unit/test_issue_fixes.py:422:        assert jnp.all(jnp.isfinite(tend_cube.du_dt.data))
tests/unit/test_issue_fixes.py:423:        assert jnp.all(jnp.isfinite(tend_cube.dT_dt.data))
tests/unit/test_issue_fixes.py:428:        assert jnp.all(jnp.isfinite(tend_ll.du_dt.data))
tests/unit/test_issue_fixes.py:429:        assert jnp.all(jnp.isfinite(tend_ll.dT_dt.data))
tests/unit/test_issue_fixes.py:431:    def test_spectral_hybrid_forcing_finite(self):
tests/unit/test_issue_fixes.py:447:        assert jnp.all(jnp.isfinite(tend.vor_hat.data.real))
tests/unit/test_issue_fixes.py:448:        assert jnp.all(jnp.isfinite(tend.div_hat.data.real))
tests/unit/test_issue_fixes.py:449:        assert jnp.all(jnp.isfinite(tend.T_hat.data.real))
tests/ocean/unit/test_mpas_ocean.py:215:    def test_reconstruction_finite(self, mesh):
tests/ocean/unit/test_mpas_ocean.py:216:        """Reconstructed velocity is finite."""
tests/ocean/unit/test_mpas_ocean.py:219:        assert jnp.all(jnp.isfinite(u_east))
tests/ocean/unit/test_mpas_ocean.py:220:        assert jnp.all(jnp.isfinite(v_north))
tests/ocean/unit/test_mpas_ocean.py:238:    def test_tendency_finite(self, state, mesh, z_coord, config):
tests/ocean/unit/test_mpas_ocean.py:239:        """All tendencies are finite."""
tests/ocean/unit/test_mpas_ocean.py:241:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/ocean/unit/test_mpas_ocean.py:242:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/ocean/unit/test_mpas_ocean.py:243:        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
tests/ocean/unit/test_mpas_ocean.py:244:        assert jnp.all(jnp.isfinite(tend.deta_dt.data))
tests/ocean/unit/test_mpas_ocean.py:249:        # Velocity tendency should be small (only from pressure gradient)
tests/ocean/unit/test_mpas_ocean.py:269:        """div(grad(const)) == 0 for cell-centered constant field."""
tests/ocean/unit/test_mpas_ocean.py:297:        """Cell mask zeroes the output on land and gradients at coastlines."""
tests/ocean/unit/test_mpas_ocean.py:315:            laplacian_cell_3d, divergence_cell_3d, gradient_edge_3d,
tests/ocean/unit/test_mpas_ocean.py:324:        grad = gradient_edge_3d(f, mesh) * edge_mask[:, None]
tests/ocean/unit/test_mpas_ocean.py:325:        lap_manual = divergence_cell_3d(grad, mesh) * mask[:, None]
tests/ocean/unit/test_mpas_ocean.py:331:    """Issue #206: scalar biharmonic tracer diffusion (K_bih) on MPAS."""
tests/ocean/unit/test_mpas_ocean.py:350:            dT/dt_horiz = K_h * div(grad(T) * edge_mask) / h_safe * h_k
tests/ocean/unit/test_mpas_ocean.py:358:            divergence_cell_3d, gradient_edge_3d,
tests/ocean/unit/test_mpas_ocean.py:380:            grad = gradient_edge_3d(f, mesh) * edge_mask[:, jnp.newaxis]
tests/ocean/unit/test_mpas_ocean.py:381:            return (K_h * divergence_cell_3d(grad, mesh) / h_safe * h_k
tests/ocean/unit/test_mpas_ocean.py:398:        diff = jnp.max(jnp.abs(t_on.dT_dt.data - t_off.dT_dt.data))
tests/ocean/unit/test_mpas_ocean.py:399:        assert float(diff) > 0.0
tests/ocean/unit/test_mpas_ocean.py:400:        assert jnp.all(jnp.isfinite(t_on.dT_dt.data))
tests/ocean/unit/test_mpas_ocean.py:401:        assert jnp.all(jnp.isfinite(t_on.dS_dt.data))
tests/ocean/unit/test_mpas_ocean.py:404:        """Bilaplacian diffusion must not inject tendency on land cells."""
tests/ocean/unit/test_mpas_ocean.py:440:        Two passes of ``div(grad(·))`` telescope on a closed manifold:
tests/ocean/unit/test_mpas_ocean.py:442:        is zero by discrete Stokes. This is the finite-volume
tests/ocean/unit/test_mpas_ocean.py:462:        Laplacian built from ``div(grad())``).  Therefore
tests/ocean/unit/test_mpas_ocean.py:499:    def test_barotropic_finite(self, state, mesh, z_coord, config):
tests/ocean/unit/test_mpas_ocean.py:500:        """Barotropic output is finite."""
tests/ocean/unit/test_mpas_ocean.py:505:        assert jnp.all(jnp.isfinite(eta_new))
tests/ocean/unit/test_mpas_ocean.py:506:        assert jnp.all(jnp.isfinite(u_bar_new))
tests/ocean/unit/test_mpas_ocean.py:507:        assert jnp.all(jnp.isfinite(Hu_avg))
tests/ocean/unit/test_mpas_ocean.py:538:        that has both ∇·u_bar ≈ 0 and is not damped by eta diffusion
tests/ocean/unit/test_mpas_ocean.py:567:            barotropic_diffusion_alpha=0.0,
tests/ocean/unit/test_mpas_ocean.py:577:        # diffusion timescale dx²/A ≈ 700 s, so ~30 substeps × 30 s
tests/ocean/unit/test_mpas_ocean.py:581:            barotropic_diffusion_alpha=0.0,
tests/ocean/unit/test_mpas_ocean.py:620:        assert jnp.all(jnp.isfinite(state_new.u.data))
tests/ocean/unit/test_mpas_ocean.py:621:        assert jnp.all(jnp.isfinite(state_new.T.data))
tests/ocean/unit/test_mpas_ocean.py:622:        assert jnp.all(jnp.isfinite(state_new.eta.data))
tests/ocean/unit/test_mpas_ocean.py:638:        assert jnp.all(jnp.isfinite(s.u.data))
tests/ocean/unit/test_mpas_ocean.py:639:        assert jnp.all(jnp.isfinite(s.T.data))
tests/ocean/unit/test_mpas_ocean.py:640:        assert jnp.all(jnp.isfinite(s.eta.data))
tests/ocean/unit/test_mpas_ocean.py:642:    def test_step_checked_finite_state(self, mesh, z_coord, config, state):
tests/ocean/unit/test_mpas_ocean.py:643:        """step_checked must validate finite state without crashing.
tests/ocean/unit/test_mpas_ocean.py:648:        cleanly and return a finite stepped state.
tests/ocean/unit/test_mpas_ocean.py:653:        assert jnp.all(jnp.isfinite(state_new.T.data))
tests/ocean/unit/test_mpas_ocean.py:654:        assert jnp.all(jnp.isfinite(state_new.S.data))
tests/ocean/unit/test_mpas_ocean.py:655:        assert jnp.all(jnp.isfinite(state_new.eta.data))
tests/ocean/unit/test_mpas_ocean.py:721:        """With n_iter=1, only land cells adjacent to ocean get filled.
tests/ocean/unit/test_mpas_ocean.py:889:        assert jnp.all(jnp.isfinite(state_fixed.eta.data))
tests/ocean/unit/test_mpas_ocean.py:890:        assert jnp.all(jnp.isfinite(state_fixed.T.data))
tests/ocean/unit/test_mpas_ocean.py:902:        x64-capable host — the fixer must run and return finite, fp32
tests/ocean/unit/test_mpas_ocean.py:943:            assert jnp.all(jnp.isfinite(state_fixed.eta.data))
tests/ocean/unit/test_mpas_ocean.py:944:            assert jnp.all(jnp.isfinite(state_fixed.T.data))
tests/ocean/unit/test_mpas_ocean.py:945:            assert jnp.all(jnp.isfinite(state_fixed.S.data))
tests/ocean/unit/test_mpas_ocean.py:1028:        assert jnp.all(jnp.isfinite(u_east))
tests/ocean/unit/test_mpas_ocean.py:1122:        assert jnp.all(jnp.isfinite(state_new.u.data))
tests/ocean/unit/test_mpas_ocean.py:1123:        assert jnp.all(jnp.isfinite(state_new.T.data))
tests/ocean/unit/test_mpas_ocean.py:1124:        assert jnp.all(jnp.isfinite(state_new.eta.data))
tests/ocean/unit/test_mpas_ocean.py:1154:    def test_upup_cell_differs_from_neighbor(self, mesh):
tests/ocean/unit/test_mpas_ocean.py:1155:        """Upup cell is generally different from the direct neighbor.
tests/ocean/unit/test_mpas_ocean.py:1170:            f"Only {frac_distinct_pos:.0%} of upup_pos differ from c2 — "
tests/ocean/unit/test_mpas_ocean.py:1212:        assert jnp.all(jnp.isfinite(state_new.u.data))
tests/ocean/unit/test_mpas_ocean.py:1213:        assert jnp.all(jnp.isfinite(state_new.T.data))
tests/ocean/unit/test_mpas_ocean.py:1214:        assert jnp.all(jnp.isfinite(state_new.S.data))
tests/ocean/unit/test_mpas_ocean.py:1215:        assert jnp.all(jnp.isfinite(state_new.eta.data))
tests/ocean/unit/test_mpas_ocean.py:1228:        assert jnp.all(jnp.isfinite(s.T.data))
tests/ocean/unit/test_mpas_ocean.py:1229:        assert jnp.all(jnp.isfinite(s.S.data))
tests/ocean/unit/test_mpas_ocean.py:1231:    def test_tvd_less_diffusive_than_upwind(self, mesh, z_coord):
tests/ocean/unit/test_mpas_ocean.py:1232:        """TVD produces less numerical diffusion than upwind.
tests/ocean/unit/test_mpas_ocean.py:1234:        Creates a state with a sharp temperature gradient at the equator,
tests/ocean/unit/test_mpas_ocean.py:1236:        gradient better than upwind after several steps.
tests/ocean/unit/test_mpas_ocean.py:1272:        # TVD should preserve more variance (less diffusive)
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:6:- Spectral PE tendency computation (rest state, shapes, finite values)
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:9:- JAX differentiability
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:27:    spectral_hyperdiffusion_3d,
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:47:def _proper_hyperdiff(grid):
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:48:    """Resolution-appropriate hyperdiffusion (4-hour damping at truncation)."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:80:    """Config with weak hyperdiffusion for stability."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:82:        hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:83:        hyperdiff_order=2,
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:97:        # Create smooth field: constant per level with different values
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:158:    def test_hyperdiffusion_3d_shape(self, grid):
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:159:        """spectral_hyperdiffusion_3d should broadcast correctly."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:162:        result = spectral_hyperdiffusion_3d(grid, coeffs, 1e10, order=2)
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:252:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:288:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:329:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:364:        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:385:            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:410:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:420:        # in at the surface).  Pin: finite, positive, and bounded
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:426:        assert bool(jnp.all(jnp.isfinite(new_qv)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:472:    def test_geopotential_finite(self, sigma_coord):
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:473:        """Geopotential should be finite everywhere."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:480:        assert jnp.all(jnp.isfinite(Phi)), "Geopotential has non-finite values"
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:515:    def test_tendencies_finite(self, rest_state, grid, sigma_coord, config):
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:516:        """All tendency values should be finite."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:521:            assert jnp.all(jnp.isfinite(data)), \
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:522:                f"Tendency {field_name} has non-finite values"
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:545:    def test_single_step_finite(self, grid, sigma_coord, config):
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:546:        """Single time step should produce finite state."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:559:            assert jnp.all(jnp.isfinite(data)), \
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:560:                f"After 1 step, {field_name} has non-finite values"
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:583:        # All fields should be finite
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:586:            assert jnp.all(jnp.isfinite(data)), \
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:587:                f"After 50 steps, {field_name} has non-finite values"
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:601:        assert jnp.all(jnp.isfinite(final.T_hat.data))
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:613:        """SI substeps should keep large-dt updates finite."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:617:            si_hyperdiff_boost=8.0,
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:631:            assert jnp.all(jnp.isfinite(data)), (
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:632:                f"With SI substeps, {field_name} has non-finite values"
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:666:    """Tests for JAX differentiability of spectral PE."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:668:    def test_tendency_differentiable(self, grid, sigma_coord, config):
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:669:        """jax.grad should work through spectral_pe_tendencies."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:679:        grad = jax.grad(loss)(state.T_hat.data)
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:680:        assert grad.shape == state.T_hat.data.shape
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:681:        assert jnp.all(jnp.isfinite(grad)), "Gradient has non-finite values"
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:9:- Is differentiable with jax.grad
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:125:        """Radiation + turbulence T tendency should differ from either alone."""
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:171:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:177:        """All three modules active should produce finite tendencies."""
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:187:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:188:        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:189:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:190:        assert jnp.all(jnp.isfinite(tend.dp_s_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:212:        # Both should be finite
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:213:        assert jnp.all(jnp.isfinite(tend_sbm.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:214:        assert jnp.all(jnp.isfinite(tend_dca.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:215:        # Should differ (different algorithms)
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:238:        # Should be different
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:261:        # But different
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:277:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:278:        """jax.grad should work through the combined physics."""
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:292:        grad_T = jax.grad(loss)(state.T.data)
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:293:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:378:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:379:        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:385:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:386:        """jax.grad should work through NH combined physics."""
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:400:        grad_u = jax.grad(loss)(state.u.data)
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:401:        assert jnp.all(jnp.isfinite(grad_u))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:466:        assert jnp.all(jnp.isfinite(tend.T_hat.data))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:1:"""Tests for spectral PE tracer hyperdiffusion + spectral filter (PR3).
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:6:  implicit hyperdiffusion factor into a single ``(n_sh,)`` multiplier
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:9:  ``hyperdiff_coeff==0``) the tracer filter is a no-op (no transform
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:13:* When only the hyperdiffusion is active, tracers are damped via the
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:19:* Long-roll stability: with hyperdiffusion ON, a high-wavenumber tracer
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:24:* AD: ``jax.grad`` flows through one filtered step.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:77:def _proper_hyperdiff(grid):
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:78:    """Hyperdiff coeff that gives ~ 4-hour e-folding at the cutoff wavenumber."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:223:            hyperdiff_coeff=0.0,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:233:            hyperdiff_coeff=0.0,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:243:    def test_only_hyperdiff_filter(self, grid, sigma_coord):
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:244:        """hyperdiff_coeff on → tracer_filter equals exp(-nu*eig*dt)."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:245:        nu = _proper_hyperdiff(grid)
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:247:            hyperdiff_coeff=nu,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:248:            hyperdiff_order=2,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:256:        eig = (grid.ls * (grid.ls + 1) / grid.radius ** 2) ** config.hyperdiff_order
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:262:        nu = _proper_hyperdiff(grid)
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:264:            hyperdiff_coeff=nu,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:265:            hyperdiff_order=2,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:283:        """Leapfrog → dt_eff = 2*dt in the hyperdiff factor."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:284:        nu = _proper_hyperdiff(grid)
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:286:            hyperdiff_coeff=nu,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:287:            hyperdiff_order=2,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:315:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:328:        mean_diff = float(jnp.abs(jnp.mean(new_qv) - jnp.mean(qv)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:329:        assert mean_diff < 1e-9, (
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:330:            f"Uniform tracer global mean drift {mean_diff} exceeded 1e-9"
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:351:            hyperdiff_coeff=0.0,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:372:            hyperdiff_coeff=0.0,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:397:        """50 steps with hyperdiffusion must keep a high-wavenumber
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:408:            hyperdiff_coeff=_proper_hyperdiff(grid) * 100.0,  # aggressive
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:421:        assert bool(jnp.all(jnp.isfinite(new_qv)))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:431:            f"50 steps of hyperdiff should reduce wave-18; "
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:441:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:450:    def test_grad_through_step_with_filter(
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:453:        """``jax.grad`` flows through one filtered step with tracers."""
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:459:            hyperdiff_coeff=_proper_hyperdiff(grid),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:473:        g = jax.grad(loss)(jnp.array(1.0))
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py:474:        assert bool(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:5:2. All outputs are finite (no NaN/Inf)
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:20:from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:57:        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:64:    """Check tendencies are finite and physically reasonable.
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:76:    # Check finite
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:77:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{label}: dT_dt has NaN/Inf"
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:78:    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{label}: du_dt has NaN/Inf"
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:79:    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{label}: dv_dt has NaN/Inf"
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:257:    """Test each gravity wave drag scheme individually."""
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:261:        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:263:        _check_tendencies(tend, "gwd/rayleigh")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:268:        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="lindzen"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:270:        _check_tendencies(tend, "gwd/lindzen")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:274:        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="mcfarlane"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:276:        _check_tendencies(tend, "gwd/mcfarlane")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:280:        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="hines"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:282:        _check_tendencies(tend, "gwd/hines")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:286:        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="prognostic_spectral"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:288:        _check_tendencies(tend, "gwd/prognostic_spectral")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:292:        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="ml_emulator"))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:294:        _check_tendencies(tend, "gwd/ml_emulator")
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:319:            gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:334:            gravity_wave_drag=GravityWaveDragConfig(scheme="lindzen"),
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:1:"""Unit tests for gravity wave drag module.
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:4:- All 6 backends: output shapes, drag-opposes-wind, finite outputs, differentiability
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:6:- Integration bridge: hydrostatic/NH shapes, nonzero tendencies, jax.grad, scheme selection
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:17:from legoesm.atmosphere.physics.gravity_wave_drag.config import (
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:26:from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput, make_zero_output
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:27:from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:28:from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:29:from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:30:from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:31:from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:32:    prognostic_spectral_gwd,
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:34:from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import (
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:35:    ml_gwd,
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:38:from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:39:    make_gwd_physics,
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:127:        out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:131:        assert out.eps_gwd.shape == (ncol,)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:137:        out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:142:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:146:        out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:147:        assert jnp.all(jnp.isfinite(out.du_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:148:        assert jnp.all(jnp.isfinite(out.dv_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:149:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:150:        assert jnp.all(jnp.isfinite(out.eps_gwd))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:152:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:158:            out = rayleigh_gwd(u_in, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:161:        g = jax.grad(loss)(u)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:162:        assert jnp.all(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:170:        out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:178:        out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:194:        out = lindzen_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:196:        assert out.eps_gwd.shape == (ncol,)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:202:        out = lindzen_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:207:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:211:        out = lindzen_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:212:        assert jnp.all(jnp.isfinite(out.du_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:213:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:215:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:221:            out = lindzen_gwd(u_in, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:224:        g = jax.grad(loss)(u)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:225:        assert jnp.all(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:233:        out_low = lindzen_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config_low)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:234:        out_high = lindzen_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config_high)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:252:        out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:254:        assert out.eps_gwd.shape == (ncol,)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:260:        out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:264:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:268:        out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:269:        assert jnp.all(jnp.isfinite(out.du_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:270:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:272:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:278:            out = mcfarlane_gwd(u_in, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:281:        g = jax.grad(loss)(u)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:282:        assert jnp.all(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:291:        out = mcfarlane_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:308:        out = hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:310:        assert out.eps_gwd.shape == (ncol,)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:316:        out = hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:320:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:324:        out = hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:325:        assert jnp.all(jnp.isfinite(out.du_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:326:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:328:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:334:            out = hines_gwd(u_in, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:337:        g = jax.grad(loss)(u)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:338:        assert jnp.all(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:345:        out = hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:362:        out, spec_new = prognostic_spectral_gwd(
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:366:        assert out.eps_gwd.shape == (ncol,)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:369:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:374:        out, spec_new = prognostic_spectral_gwd(
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:377:        assert jnp.all(jnp.isfinite(out.du_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:378:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:379:        assert jnp.all(jnp.isfinite(spec_new))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:381:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:388:            out, _ = prognostic_spectral_gwd(
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:393:        g = jax.grad(loss)(u)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:394:        assert jnp.all(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:402:        _, spec_new = prognostic_spectral_gwd(
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:422:        out = ml_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config, model)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:424:        assert out.eps_gwd.shape == (ncol,)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:426:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:432:        out = ml_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config, model)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:433:        assert jnp.all(jnp.isfinite(out.du_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:434:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:436:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:444:            out = ml_gwd(u_in, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config, model)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:447:        g = jax.grad(loss)(u)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:448:        assert jnp.all(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:457:        out = ml_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, config, model)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:467:    """Tests for make_gwd_physics integration bridge."""
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:487:        physics_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:526:        physics_fn = make_gwd_physics(config, model_type="nonhydrostatic", dt=300.0)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:550:        physics_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:556:    def test_grad_through_hydrostatic(self):
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:573:        physics_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:582:        g = jax.grad(loss)(state.u.data)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:583:        assert jnp.all(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:590:            physics_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:603:        physics_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:637:            gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:655:        assert out.eps_gwd.shape == (4,)
tests/atmosphere/hydrostatic/unit/test_radiation.py:6:- Integration: correct tendency shapes, differentiability
tests/atmosphere/hydrostatic/unit/test_radiation.py:202:        """Heating rate should be finite and bounded (< 10 K/day)."""
tests/atmosphere/hydrostatic/unit/test_radiation.py:207:        assert jnp.all(jnp.isfinite(out.heating_rate))
tests/atmosphere/hydrostatic/unit/test_radiation.py:301:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_radiation.py:302:        """jax.grad should work through gray_radiation."""
tests/atmosphere/hydrostatic/unit/test_radiation.py:311:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_radiation.py:312:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_radiation.py:313:        assert grad_T.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_radiation.py:379:    def test_rrtmgp_backend_differs_from_gray(self):
tests/atmosphere/hydrostatic/unit/test_radiation.py:394:        diff = float(jnp.max(jnp.abs(gray_tend - rrtmgp_tend)))
tests/atmosphere/hydrostatic/unit/test_radiation.py:395:        assert diff > 1e-8
tests/atmosphere/hydrostatic/unit/test_radiation.py:477:    def test_grad_through_hydrostatic_radiation(self):
tests/atmosphere/hydrostatic/unit/test_radiation.py:478:        """jax.grad should work through hydrostatic radiation physics."""
tests/atmosphere/hydrostatic/unit/test_radiation.py:495:        grad_T = jax.grad(loss)(state.T.data)
tests/atmosphere/hydrostatic/unit/test_radiation.py:496:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_radiation.py:511:        """RRTMGP clear-sky heating rates should be finite."""
tests/atmosphere/hydrostatic/unit/test_radiation.py:525:        assert jnp.all(jnp.isfinite(out.heating_rate))
tests/atmosphere/hydrostatic/unit/test_radiation.py:528:    def test_rrtmgp_differentiable_temperature(self):
tests/atmosphere/hydrostatic/unit/test_radiation.py:529:        """jax.grad w.r.t. temperature should work through rrtmgp_radiation."""
tests/atmosphere/hydrostatic/unit/test_radiation.py:545:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_radiation.py:546:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_radiation.py:547:        assert grad_T.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_radiation.py:549:    def test_rrtmgp_differentiable_cos_zenith(self):
tests/atmosphere/hydrostatic/unit/test_radiation.py:550:        """jax.grad w.r.t. cos_zenith should work through rrtmgp_radiation."""
tests/atmosphere/hydrostatic/unit/test_radiation.py:566:        grad_cz = jax.grad(loss)(cos_zen)
tests/atmosphere/hydrostatic/unit/test_radiation.py:567:        assert jnp.all(jnp.isfinite(grad_cz))
tests/atmosphere/hydrostatic/unit/test_radiation.py:568:        assert grad_cz.shape == cos_zen.shape
tests/atmosphere/hydrostatic/unit/test_radiation.py:604:            assert jnp.all(jnp.isfinite(a))
tests/atmosphere/hydrostatic/unit/test_radiation.py:605:            assert jnp.all(jnp.isfinite(b))
tests/atmosphere/hydrostatic/unit/test_radiation.py:644:        # At least one of these should be significantly different
tests/atmosphere/hydrostatic/unit/test_radiation.py:667:        assert jnp.all(jnp.isfinite(tendencies.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_radiation.py:698:        # Two calls with different times should give the same result
tests/atmosphere/hydrostatic/unit/test_radiation.py:731:        assert jnp.all(jnp.isfinite(o3))
tests/atmosphere/hydrostatic/unit/test_radiation.py:845:        assert jnp.all(jnp.isfinite(out.heating_rate))
tests/atmosphere/hydrostatic/unit/test_radiation.py:849:        """External ozone should produce different heating than no ozone."""
tests/atmosphere/hydrostatic/unit/test_radiation.py:878:        # Heating rates should differ
tests/atmosphere/hydrostatic/unit/test_radiation.py:879:        diff = float(jnp.max(jnp.abs(
tests/atmosphere/hydrostatic/unit/test_radiation.py:882:        assert diff > 1e-8
tests/atmosphere/hydrostatic/unit/test_radiation.py:902:        assert jnp.all(jnp.isfinite(tendencies.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_radiation.py:972:        assert jnp.all(jnp.isfinite(props.cloud_fraction))
tests/atmosphere/hydrostatic/unit/test_radiation.py:1039:        assert jnp.all(jnp.isfinite(out.heating_rate))
tests/atmosphere/hydrostatic/unit/test_radiation.py:1072:        # Heating rates should differ
tests/atmosphere/hydrostatic/unit/test_radiation.py:1073:        diff = float(jnp.max(jnp.abs(
tests/atmosphere/hydrostatic/unit/test_radiation.py:1076:        assert diff > 1e-8
tests/atmosphere/hydrostatic/unit/test_radiation.py:1101:        assert jnp.all(jnp.isfinite(tendencies.dT_dt.data))
tests/ocean/unit/test_ocean.py:12:- JAX differentiability
tests/ocean/unit/test_ocean.py:38:    compute_ocean_jacobian,
tests/ocean/unit/test_ocean.py:47:from legoesm.ocean.physics.mixing import vertical_diffusion
tests/ocean/unit/test_ocean.py:96:        hyperdiff_coeff=0.0,
tests/ocean/unit/test_ocean.py:143:        assert jnp.all(jnp.diff(rho) < 0)
tests/ocean/unit/test_ocean.py:157:        assert jnp.isfinite(rho)
tests/ocean/unit/test_ocean.py:159:    def test_grad_interior_nonzero(self):
tests/ocean/unit/test_ocean.py:163:        g = jax.grad(rho_of_T)(jnp.array(10.0))
tests/ocean/unit/test_ocean.py:164:        assert jnp.isfinite(g)
tests/ocean/unit/test_ocean.py:167:    def test_grad_finite_nonzero_outside_valid_range(self):
tests/ocean/unit/test_ocean.py:168:        """Gradient must be finite and nonzero outside [-2, 40] degC.
tests/ocean/unit/test_ocean.py:171:        box. The EOS must not clip inputs — silent clipping zeros grads at
tests/ocean/unit/test_ocean.py:177:        g_hot = jax.grad(rho_of_T)(jnp.array(45.0))
tests/ocean/unit/test_ocean.py:178:        assert jnp.isfinite(g_hot)
tests/ocean/unit/test_ocean.py:179:        assert float(g_hot) != 0.0, f"Expected nonzero grad at T=45C, got {float(g_hot)}"
tests/ocean/unit/test_ocean.py:181:        g_cold = jax.grad(rho_of_T)(jnp.array(-5.0))
tests/ocean/unit/test_ocean.py:182:        assert jnp.isfinite(g_cold)
tests/ocean/unit/test_ocean.py:183:        assert float(g_cold) != 0.0, f"Expected nonzero grad at T=-5C, got {float(g_cold)}"
tests/ocean/unit/test_ocean.py:188:        g_fresh = jax.grad(rho_of_S)(jnp.array(-1.0))
tests/ocean/unit/test_ocean.py:189:        assert jnp.isfinite(g_fresh) and float(g_fresh) != 0.0
tests/ocean/unit/test_ocean.py:190:        g_brine = jax.grad(rho_of_S)(jnp.array(45.0))
tests/ocean/unit/test_ocean.py:191:        assert jnp.isfinite(g_brine) and float(g_brine) != 0.0
tests/ocean/unit/test_ocean.py:193:    def test_density_finite_outside_valid_range(self):
tests/ocean/unit/test_ocean.py:194:        """Density itself must also be finite outside the nominal box."""
tests/ocean/unit/test_ocean.py:195:        # Mild overshoot (advection / diffusion style)
tests/ocean/unit/test_ocean.py:197:        assert jnp.isfinite(rho_mild)
tests/ocean/unit/test_ocean.py:199:        # Larger excursion — should still be finite, may be unphysical.
tests/ocean/unit/test_ocean.py:201:        assert jnp.isfinite(rho_wild)
tests/ocean/unit/test_ocean.py:251:        assert jnp.all(jnp.diff(rho) < 0)
tests/ocean/unit/test_ocean.py:253:    def test_jit_and_grad(self):
tests/ocean/unit/test_ocean.py:254:        """Linear EOS should be JIT-able and differentiable."""
tests/ocean/unit/test_ocean.py:257:        assert jnp.isfinite(rho)
tests/ocean/unit/test_ocean.py:259:        g = jax.grad(lambda T: linear_eos(T, jnp.array(35.0), jnp.array(0.0)))
tests/ocean/unit/test_ocean.py:277:        assert jnp.isfinite(rho)
tests/ocean/unit/test_ocean.py:345:    def test_jacobian(self, ocean_z_coord):
tests/ocean/unit/test_ocean.py:350:        J = compute_ocean_jacobian(eta, H_bathy, ocean_z_coord)
tests/ocean/unit/test_ocean.py:353:    def test_jacobian_respects_min_water_column(self, ocean_z_coord):
tests/ocean/unit/test_ocean.py:357:        J = compute_ocean_jacobian(
tests/ocean/unit/test_ocean.py:414:        assert jnp.all(jnp.isfinite(state_new.u.data))
tests/ocean/unit/test_ocean.py:415:        assert jnp.all(jnp.isfinite(state_new.T.data))
tests/ocean/unit/test_ocean.py:416:        assert jnp.all(jnp.isfinite(state_new.eta.data))
tests/ocean/unit/test_ocean.py:477:    def test_tendencies_finite(self, ocean_state, ocean_grid, ocean_cdgrid, ocean_z_coord, ocean_config):
tests/ocean/unit/test_ocean.py:478:        """All tendencies should be finite."""
tests/ocean/unit/test_ocean.py:482:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/ocean/unit/test_ocean.py:483:        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
tests/ocean/unit/test_ocean.py:484:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/ocean/unit/test_ocean.py:485:        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
tests/ocean/unit/test_ocean.py:486:        assert jnp.all(jnp.isfinite(tend.deta_dt.data))
tests/ocean/unit/test_ocean.py:488:    def test_tendencies_finite_for_thin_columns(self, ocean_state, ocean_grid, ocean_cdgrid, ocean_z_coord):
tests/ocean/unit/test_ocean.py:489:        """Tendency path should remain finite when eta approaches dry columns."""
tests/ocean/unit/test_ocean.py:501:            hyperdiff_coeff=0.0,
tests/ocean/unit/test_ocean.py:507:        assert jnp.all(jnp.isfinite(tend.du_dt.data))
tests/ocean/unit/test_ocean.py:508:        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
tests/ocean/unit/test_ocean.py:509:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/ocean/unit/test_ocean.py:510:        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
tests/ocean/unit/test_ocean.py:511:        assert jnp.all(jnp.isfinite(tend.deta_dt.data))
tests/ocean/unit/test_ocean.py:553:            hyperdiff_coeff=0.0,
tests/ocean/unit/test_ocean.py:567:        """Upward flow should use deeper donor and zero-gradient at bottom."""
tests/ocean/unit/test_ocean.py:569:        jac = jnp.ones((1, 1, 1))
tests/ocean/unit/test_ocean.py:572:        adv = _vertical_advection_ocean(field, w_half, ocean_z_coord, jac)
tests/ocean/unit/test_ocean.py:578:        """Downward flow should use shallower donor and zero-gradient at surface."""
tests/ocean/unit/test_ocean.py:580:        jac = jnp.ones((1, 1, 1))
tests/ocean/unit/test_ocean.py:583:        adv = _vertical_advection_ocean(field, w_half, ocean_z_coord, jac)
tests/ocean/unit/test_ocean.py:588:    def test_vertical_advection_jacobian_scaling(self, ocean_z_coord):
tests/ocean/unit/test_ocean.py:591:        jac = jnp.full((1, 1, 1), J)
tests/ocean/unit/test_ocean.py:596:        adv = _vertical_advection_ocean(field, w_half, ocean_z_coord, jac)
tests/ocean/unit/test_ocean.py:721:        """Top and bottom levels receive nonzero tendency from adjacent
tests/ocean/unit/test_ocean.py:722:        interface flux, unlike the cell-upwind gradient form which
tests/ocean/unit/test_ocean.py:723:        forced ``grad = 0`` at k=0 and k=nlev-1.
tests/ocean/unit/test_ocean.py:744:    """Tests for vertical diffusion operator."""
tests/ocean/unit/test_ocean.py:746:    def test_vertical_diffusion_no_scatter_dtype_warning(self, ocean_z_coord):
tests/ocean/unit/test_ocean.py:747:        """vertical_diffusion should avoid mixed-dtype scatter updates."""
tests/ocean/unit/test_ocean.py:752:        jac_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
tests/ocean/unit/test_ocean.py:753:        jac = jnp.ones((1, 1, 1), dtype=jac_dtype)
tests/ocean/unit/test_ocean.py:761:            tendency = vertical_diffusion(field, ocean_z_coord, jac, coeff=1.0e-4)
tests/ocean/unit/test_ocean.py:764:        assert jnp.all(jnp.isfinite(tendency))
tests/ocean/unit/test_ocean.py:778:        assert jnp.all(jnp.isfinite(state_new.u.data))
tests/ocean/unit/test_ocean.py:779:        assert jnp.all(jnp.isfinite(state_new.eta.data))
tests/ocean/unit/test_ocean.py:848:            ({"hyperdiff_coeff": -1.0}, "hyperdiff_coeff"),
tests/ocean/unit/test_ocean.py:849:            ({"barotropic_diffusion_alpha": -1.0}, "barotropic_diffusion_alpha"),
tests/ocean/unit/test_ocean.py:851:            ({"barotropic_diffusion_dt_ref": 0.0}, "barotropic_diffusion_dt_ref"),
tests/ocean/unit/test_ocean.py:870:        """Runtime-checked step should run and return finite state."""
tests/ocean/unit/test_ocean.py:882:        assert jnp.all(jnp.isfinite(state_new.u.data))
tests/ocean/unit/test_ocean.py:883:        assert jnp.all(jnp.isfinite(state_new.eta.data))
tests/ocean/unit/test_ocean.py:979:    def test_heat_salt_fixers_finite_for_thin_columns(
tests/ocean/unit/test_ocean.py:982:        """Heat/salt fixers should remain finite with thin-column clipping."""
tests/ocean/unit/test_ocean.py:1002:        assert jnp.all(jnp.isfinite(state_heat.T.data))
tests/ocean/unit/test_ocean.py:1003:        assert jnp.all(jnp.isfinite(state_salt.S.data))
tests/ocean/unit/test_ocean.py:1011:    """Tests for JAX differentiability through ocean model."""
tests/ocean/unit/test_ocean.py:1013:    def test_grad_through_tendencies(self, ocean_state, ocean_grid, ocean_cdgrid, ocean_z_coord, ocean_config):
tests/ocean/unit/test_ocean.py:1014:        """jax.grad should work through tendency computation."""
tests/ocean/unit/test_ocean.py:1024:        grad_fn = jax.grad(loss_fn)
tests/ocean/unit/test_ocean.py:1025:        grad = grad_fn(ocean_state.eta.data)
tests/ocean/unit/test_ocean.py:1026:        assert jnp.all(jnp.isfinite(grad))
tests/ocean/unit/test_ocean.py:1263:    def test_tendencies_finite_for_all_ocean_mask(self):
tests/ocean/unit/test_ocean.py:1284:            hyperdiff_coeff=0.0,
tests/ocean/unit/test_ocean.py:1291:                assert jnp.all(jnp.isfinite(leaf))
tests/ocean/unit/test_ocean.py:1354:        # Check fields remain finite and bounded.
tests/ocean/unit/test_ocean.py:1356:        assert jnp.all(jnp.isfinite(state.T.data))
tests/ocean/unit/test_ocean.py:1357:        assert jnp.all(jnp.isfinite(state.S.data))
tests/ocean/unit/test_ocean.py:1358:        assert jnp.all(jnp.isfinite(state.u.data))
tests/ocean/unit/test_ocean.py:1359:        assert jnp.all(jnp.isfinite(state.v.data))
tests/ocean/unit/test_ocean.py:1360:        assert jnp.all(jnp.isfinite(state.eta.data))
tests/ocean/unit/test_ocean.py:1366:        """50-step integration WITHOUT fixer: state should remain finite and bounded."""
tests/ocean/unit/test_ocean.py:1383:        # Without fixer, check stability (finite and bounded).
tests/ocean/unit/test_ocean.py:1384:        assert jnp.all(jnp.isfinite(state.T.data))
tests/ocean/unit/test_ocean.py:1385:        assert jnp.all(jnp.isfinite(state.S.data))
tests/ocean/unit/test_ocean.py:1386:        assert jnp.all(jnp.isfinite(state.eta.data))
tests/ocean/unit/test_ocean.py:1431:        """Z-star velocity should differ from Eulerian when deta/dt != 0."""
tests/ocean/unit/test_ocean.py:1449:        # Interior values should differ.
tests/ocean/unit/test_ocean.py:1450:        diff = float(jnp.max(jnp.abs(w_zstar - w_euler)))
tests/ocean/unit/test_ocean.py:1451:        assert diff > 0.0
tests/atmosphere/hydrostatic/unit/test_convection.py:5:- SBM: shapes, enthalpy conservation, precipitation, trigger, differentiability
tests/atmosphere/hydrostatic/unit/test_convection.py:6:- DCA: shapes, stable unchanged, instability reduction, differentiability
tests/atmosphere/hydrostatic/unit/test_convection.py:7:- Integration: hydrostatic/NH shapes, nonzero heating, jax.grad, scheme selection
tests/atmosphere/hydrostatic/unit/test_convection.py:111:        assert jnp.all(jnp.diff(q_sat) > 0)
tests/atmosphere/hydrostatic/unit/test_convection.py:189:    def test_qsat_grad_works(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:190:        """jax.grad should work through saturation_mixing_ratio."""
tests/atmosphere/hydrostatic/unit/test_convection.py:194:        g = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_convection.py:195:        assert jnp.all(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_convection.py:272:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:273:        """jax.grad should work through SBM convection."""
tests/atmosphere/hydrostatic/unit/test_convection.py:282:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_convection.py:283:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_convection.py:284:        assert grad_T.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_convection.py:298:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_convection.py:299:        assert jnp.all(jnp.isfinite(out.dq_v_dt))
tests/atmosphere/hydrostatic/unit/test_convection.py:366:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:367:        """jax.grad should work through DCA convection."""
tests/atmosphere/hydrostatic/unit/test_convection.py:376:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_convection.py:377:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_convection.py:378:        assert grad_T.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_convection.py:523:    def test_grad_through_hydrostatic_convection(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:524:        """jax.grad should work through hydrostatic convection physics."""
tests/atmosphere/hydrostatic/unit/test_convection.py:541:        grad_T = jax.grad(loss)(state.T.data)
tests/atmosphere/hydrostatic/unit/test_convection.py:542:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_convection.py:566:        # Both should produce nonzero but different tendencies
tests/atmosphere/hydrostatic/unit/test_convection.py:586:        """scheme='kuo' should give different results from 'sbm'."""
tests/atmosphere/hydrostatic/unit/test_convection.py:640:    def test_grad_through_mass_flux(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:641:        """jax.grad should work through mass_flux integration."""
tests/atmosphere/hydrostatic/unit/test_convection.py:658:        grad_T = jax.grad(loss)(state.T.data)
tests/atmosphere/hydrostatic/unit/test_convection.py:659:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_convection.py:661:    def test_grad_through_edmf(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:662:        """jax.grad should work through edmf integration."""
tests/atmosphere/hydrostatic/unit/test_convection.py:679:        grad_T = jax.grad(loss)(state.T.data)
tests/atmosphere/hydrostatic/unit/test_convection.py:680:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_convection.py:718:        off (which would be a different test).
tests/atmosphere/hydrostatic/unit/test_convection.py:730:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:731:        """jax.grad should work through Kuo convection."""
tests/atmosphere/hydrostatic/unit/test_convection.py:740:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_convection.py:741:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_convection.py:742:        assert grad_T.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_convection.py:781:        """Calling twice with updated M_c should give different results."""
tests/atmosphere/hydrostatic/unit/test_convection.py:840:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:841:        """jax.grad should work through Mass-Flux convection."""
tests/atmosphere/hydrostatic/unit/test_convection.py:853:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_convection.py:854:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_convection.py:855:        assert grad_T.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_convection.py:923:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:924:        """jax.grad should work through EDMF convection."""
tests/atmosphere/hydrostatic/unit/test_convection.py:936:        grad_T = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_convection.py:937:        assert jnp.all(jnp.isfinite(grad_T))
tests/atmosphere/hydrostatic/unit/test_convection.py:938:        assert grad_T.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_convection.py:962:        # Mid-troposphere levels (away from boundaries where gradient is zero)
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:11:* the result is finite for typical synthetic fields and matches the
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:13:* differentiability through ``q_v`` (the diagnostic is part of the
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:38:# Shape / finiteness
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:41:def test_moisture_convergence_shape_and_finite():
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:48:    assert jnp.all(jnp.isfinite(mc))
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:95:# Differentiability — gradient through q_v
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:98:def test_grad_through_q_v_finite():
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:99:    """``d (sum MC) / d q_v`` is finite (the limiter-off variant is
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:109:    g = jax.grad(f)(q_v)
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:111:    assert jnp.all(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:138:    lifting.  These tests pin shape, sign, and differentiability.
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:147:    def test_gaussian_shape_and_finite(self):
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:154:        assert jnp.all(jnp.isfinite(mc))
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:179:    def test_gaussian_grad_through_q_v_finite(self):
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:187:        g = jax.grad(f)(q_v)
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:189:        assert jnp.all(jnp.isfinite(g))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:164:        assert jnp.all(jnp.isfinite(tend))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:189:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:193:            assert jnp.all(jnp.isfinite(field)), f"Non-finite in {field}"
tests/atmosphere/hydrostatic/unit/test_microphysics.py:210:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:217:        grad = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:218:        assert jnp.all(jnp.isfinite(grad))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:250:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:254:            assert jnp.all(jnp.isfinite(field))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:274:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:281:        grad = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:282:        assert jnp.all(jnp.isfinite(grad))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:308:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:312:            assert jnp.all(jnp.isfinite(field))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:314:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:321:        grad = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:322:        assert jnp.all(jnp.isfinite(grad))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:355:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:359:            assert jnp.all(jnp.isfinite(field))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:375:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:382:        grad = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:383:        assert jnp.all(jnp.isfinite(grad))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:415:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:419:            assert jnp.all(jnp.isfinite(field))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:437:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:444:        grad = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:445:        assert jnp.all(jnp.isfinite(grad))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:496:    def test_finite_outputs(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:503:            assert jnp.all(jnp.isfinite(field))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:505:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:515:        grad = jax.grad(loss)(T)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:516:        assert jnp.all(jnp.isfinite(grad))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:550:    def test_grad_through_hydrostatic(self, setup):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:560:        grad = jax.grad(loss)(state.T.data)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:561:        assert jnp.all(jnp.isfinite(grad))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:626:        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:627:        assert jnp.all(jnp.isfinite(tend.dtracers_dt.data))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:703:        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:831:        assert jnp.all(jnp.isfinite(T))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:832:        assert jnp.all(jnp.isfinite(q_v))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:833:        assert jnp.all(jnp.isfinite(q_c))
tests/atmosphere/hydrostatic/unit/test_microphysics.py:834:        assert jnp.all(jnp.isfinite(q_r))
tests/atmosphere/hydrostatic/unit/test_energy_budget.py:9:- JAX differentiability
tests/atmosphere/hydrostatic/unit/test_energy_budget.py:34:    dsigma = jnp.diff(sigma_half)
tests/atmosphere/hydrostatic/unit/test_energy_budget.py:164:    def test_finite(self):
tests/atmosphere/hydrostatic/unit/test_energy_budget.py:165:        """Output should be finite."""
tests/atmosphere/hydrostatic/unit/test_energy_budget.py:176:        assert jnp.isfinite(E)
tests/atmosphere/hydrostatic/unit/test_energy_budget.py:178:    def test_differentiable(self):
tests/atmosphere/hydrostatic/unit/test_energy_budget.py:179:        """jax.grad should work through column_moist_static_energy."""
tests/atmosphere/hydrostatic/unit/test_energy_budget.py:191:        grad = jax.grad(loss)(jnp.full((nlev,), 280.0))
tests/atmosphere/hydrostatic/unit/test_energy_budget.py:192:        assert jnp.all(jnp.isfinite(grad))
tests/atmosphere/hydrostatic/unit/test_energy_budget.py:194:        assert jnp.all(grad > 0.0)
tests/atmosphere/hydrostatic/unit/test_energy_budget.py:329:        """After two updates with different T, dE/dt should be nonzero."""
tests/atmosphere/hydrostatic/unit/test_monthly_means.py:100:    def test_different_months(self):
tests/atmosphere/hydrostatic/unit/test_monthly_means.py:101:        """Adds in different months should produce separate entries."""
tests/atmosphere/hydrostatic/unit/test_amip_forcing.py:40:    # Add latitude gradient to SST
tests/atmosphere/hydrostatic/unit/test_amip_forcing.py:203:        assert jnp.all(jnp.isfinite(forcing.sst))
tests/atmosphere/hydrostatic/unit/test_amip_forcing.py:204:        assert jnp.all(jnp.isfinite(forcing.sic))
tests/atmosphere/hydrostatic/unit/test_amip_forcing.py:236:        assert jnp.all(jnp.isfinite(forcing.sst))
tests/atmosphere/hydrostatic/unit/test_amip_forcing.py:237:        assert jnp.all(jnp.isfinite(forcing.sic))
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:30:- All output is finite.
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:165:    def test_finite_output(
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:168:        """All tendency fields are finite."""
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:174:        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data))), (
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:177:        assert bool(jnp.all(jnp.isfinite(tend.du_dt.data)))
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:178:        assert bool(jnp.all(jnp.isfinite(tend.dv_dt.data)))
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:179:        assert bool(jnp.all(jnp.isfinite(tend.dp_s_dt.data)))
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:239:    def test_finite_output(
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:242:        """All tendency fields are finite on MPAS."""
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:248:        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data))), (
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:251:        assert bool(jnp.all(jnp.isfinite(tend.du_dt.data)))
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:252:        assert bool(jnp.all(jnp.isfinite(tend.dp_s_dt.data)))
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:271:        proxy call and *differs* from the zeros call.  This pins the
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:272:        actual semantic difference the fix is meant to deliver.
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:289:        # are zeros (this is the bridge's MPAS CMT graceful-degrade
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:347:        # And the proxy and zero-MC paths must differ meaningfully.
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:352:        zeros_diff = float(jnp.max(jnp.abs(proxy_dT - zeros_dT)))
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:353:        assert zeros_diff / proxy_scale > 1e-2, (
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:355:            "Tiedtke tendencies — the differential test cannot "
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:358:            f"{zeros_diff / proxy_scale:.3e}.  Tighten the test "
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:414:    def test_mpas_bridge_returns_finite_tendencies(
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:432:        # All output finite.
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:433:        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data)))
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:434:        assert bool(jnp.all(jnp.isfinite(tend.du_dt.data)))
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:452:    def test_lat_lon_bridge_returns_finite_tendencies(
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:467:        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data)))
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:487:    def test_gwd_mpas_raises_not_implemented(self):
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:488:        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:491:        from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:492:            make_gwd_physics,
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:496:            make_gwd_physics(cfg, model_type="mpas", dt=300.0)

codex
The core files confirm the exact scan structure and constants the hypotheses point at. I’m narrowing the test check now, because the first grep also matched unrelated differentiability tests elsewhere in the repository.
exec
/bin/zsh -lc 'nl -ba tests/unit/test_physics_gwd.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Category 6: Gravity Wave Drag -- Physical Consistency.
     2	
     3	Tests drag opposing wind, energy dissipation, Rayleigh sponge structure,
     4	zero-wind behavior, and magnitude bounds for all GWD schemes.
     5	"""
     6	
     7	from __future__ import annotations
     8	
     9	import jax.numpy as jnp
    10	import pytest
    11	
    12	from legoesm.core.field import Field
    13	from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    14	from legoesm.atmosphere.physics.gravity_wave_drag.integration import make_gwd_physics
    15	
    16	
    17	def _make_state(n=8, nlev=10, wind_speed=10.0):
    18	    from legoesm.grids.cubed_sphere import create_cubed_sphere
    19	    from legoesm.grids.vertical import create_sigma_coordinate
    20	    from legoesm.atmosphere.held_suarez import held_suarez_init
    21	
    22	    grid = create_cubed_sphere(n)
    23	    sigma = create_sigma_coordinate(nlev)
    24	    state = held_suarez_init(grid, sigma)
    25	    state = state._replace(
    26	        u=Field(data=jnp.ones((6, n, n, nlev)) * wind_speed,
    27	                name="u", dims=("face", "x", "y", "level"), units="m/s"),
    28	        v=Field(data=jnp.ones((6, n, n, nlev)) * 3.0,
    29	                name="v", dims=("face", "x", "y", "level"), units="m/s"),
    30	    )
    31	    tracers = {
    32	        "q_v": Field(1e-3 * jnp.ones((6, n, n, nlev)), name="q_v",
    33	                     dims=("face", "x", "y", "level"), units="kg/kg"),
    34	    }
    35	    state = state._replace(tracers=tracers)
    36	    return state, grid, sigma
    37	
    38	
    39	def _run_gwd(scheme, **kwargs):
    40	    state, grid, sigma = _make_state(**kwargs)
    41	    config = GravityWaveDragConfig(scheme=scheme)
    42	    gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
    43	    tend, prog = gwd_fn(state, grid, sigma)
    44	    return tend, state
    45	
    46	
    47	ALL_SCHEMES = ["rayleigh", "lindzen", "mcfarlane", "hines", "prognostic_spectral"]
    48	# Diagnostic schemes (not multi-directional) where drag should oppose wind
    49	DIAGNOSTIC_SCHEMES = ["rayleigh", "lindzen", "mcfarlane", "hines"]
    50	
    51	
    52	# ============================================================================
    53	# 6a  Drag opposes wind (diagnostic schemes only)
    54	# ============================================================================
    55	
    56	@pytest.mark.parametrize("scheme", DIAGNOSTIC_SCHEMES)
    57	def test_drag_opposes_wind(scheme):
    58	    """du_dt * u <= 0 wherever drag is active (drag decelerates)."""
    59	    tend, state = _run_gwd(scheme, wind_speed=15.0)
    60	    u = state.u.data
    61	    du_dt = tend.du_dt.data
    62	    product = u * du_dt
    63	    active = jnp.abs(du_dt) > 1e-12
    64	    if jnp.any(active):
    65	        n_opposing = int(jnp.sum((product[active] <= 1e-10)))
    66	        n_active = int(jnp.sum(active))
    67	        frac = n_opposing / max(n_active, 1)
    68	        assert frac > 0.9, (
    69	            f"{scheme}: only {frac:.0%} of active points have drag opposing wind"
    70	        )
    71	
    72	
    73	# ============================================================================
    74	# 6d  Zero wind -> zero drag
    75	# ============================================================================
    76	
    77	@pytest.mark.parametrize("scheme", ALL_SCHEMES)
    78	def test_zero_wind_zero_drag(scheme):
    79	    """With u = v = 0, GWD tendencies should be zero."""
    80	    state, grid, sigma = _make_state(wind_speed=0.0)
    81	    state = state._replace(
    82	        v=Field(data=jnp.zeros_like(state.v.data),
    83	                name="v", dims=state.v.dims, units="m/s"),
    84	    )
    85	    config = GravityWaveDragConfig(scheme=scheme)
    86	    gwd_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
    87	    tend, _ = gwd_fn(state, grid, sigma)
    88	    max_du = float(jnp.max(jnp.abs(tend.du_dt.data)))
    89	    max_dv = float(jnp.max(jnp.abs(tend.dv_dt.data)))
    90	    assert max_du < 1e-10, f"{scheme}: du_dt = {max_du:.2e} with zero wind"
    91	    assert max_dv < 1e-10, f"{scheme}: dv_dt = {max_dv:.2e} with zero wind"
    92	
    93	
    94	# ============================================================================
    95	# 6f  Magnitude bounds
    96	# ============================================================================
    97	
    98	@pytest.mark.parametrize("scheme", ALL_SCHEMES)
    99	def test_magnitude_bounds(scheme):
   100	    """GWD tendencies should be within physical bounds."""
   101	    tend, _ = _run_gwd(scheme)
   102	    max_du = float(jnp.max(jnp.abs(tend.du_dt.data)))
   103	    max_dT = float(jnp.max(jnp.abs(tend.dT_dt.data)))
   104	    assert max_du < 0.1, f"{scheme}: |du_dt| = {max_du:.2e} exceeds 0.1 m/s^2"
   105	    assert max_dT < 1e-3, f"{scheme}: |dT_dt| = {max_dT:.2e} exceeds 1e-3 K/s"
   106	
   107	
   108	# ============================================================================
   109	# All outputs finite
   110	# ============================================================================
   111	
   112	@pytest.mark.parametrize("scheme", ALL_SCHEMES)
   113	def test_all_finite(scheme):
   114	    """All GWD output fields should be finite."""
   115	    tend, _ = _run_gwd(scheme)
   116	    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{scheme}: du_dt NaN/Inf"
   117	    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{scheme}: dv_dt NaN/Inf"
   118	    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt NaN/Inf"
   119	
   120	
   121	# ============================================================================
   122	# Rayleigh: structure check
   123	# ============================================================================
   124	
   125	def test_rayleigh_sponge_structure():
   126	    """Rayleigh drag should be active at model top and/or BL."""
   127	    tend, _ = _run_gwd("rayleigh", nlev=20, wind_speed=20.0)
   128	    du_dt = tend.du_dt.data
   129	    top_drag = float(jnp.max(jnp.abs(du_dt[..., :3])))
   130	    bot_drag = float(jnp.max(jnp.abs(du_dt[..., -5:])))
   131	    assert top_drag > 1e-8 or bot_drag > 1e-8, "Rayleigh: no drag anywhere"

exec
/bin/zsh -lc 'rg -n "grad|gradient|value_and_grad|jac|jvp|vjp|finite diff|differenti" tests/unit/test_physics_gwd.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 0ms:
exec
/bin/zsh -lc "rg -n \"dT_dt = -\\(|frictional heating|eps_gwd|scan_fn|drag_all|jnp\\.clip\\(tau_0|sigmoid\\(20\\.0|alpha = 50\\.0|rho_ratio|sigma_sat|sigma_grown\" src/legoesm/atmosphere/physics/gravity_wave_drag" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:30:    eps_gwd : jax.Array
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:32:        Positive-definite: eps_gwd = -sum(rho * (u*du_dt + v*dv_dt) * dz).
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:37:    eps_gwd: jax.Array
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py:50:        eps_gwd=jnp.zeros((ncol,), dtype=dtype),
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py:96:    eps_gwd = jnp.sum(rho * jnp.abs(u * du_dt + v * dv_dt) * dz, axis=1)
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py:98:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:83:    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:87:    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py:89:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:77:    # reaches a fraction of the local wind: sigma_sat ~ U / m_star_norm
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:79:    rho_ratio = jnp.sqrt(jnp.clip(rho_sfc / jnp.clip(rho, 0.01, None), 1.0, None))
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:80:    # sigma_sat = wind_fraction * N / (m_star) at each level
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:81:    sigma_sat = N_full / jnp.clip(config.m_star * rho_ratio, 1e-6, None)
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:85:    def scan_fn(carry, k_rev):
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:90:        sigma_grown = sigma_gw * rho_ratio[:, k]
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:94:            config.doppler_sharpness * (sigma_grown - sigma_sat[:, k])
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:96:        sigma_new = sigma_grown * (1.0 - f_diss) + sigma_sat[:, k] * f_diss
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:98:        # Momentum deposited: rho * (sigma_grown - sigma_new) ~ stress gradient
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:99:        drag = (sigma_grown - sigma_new) * rho[:, k]
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:107:    _, drag_stack = jax.lax.scan(scan_fn, sigma_gw_init, jnp.arange(nlev))
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:108:    drag_all = drag_stack.T[:, ::-1]  # (ncol, nlev), top-first
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:111:    accel = -drag_all / jnp.clip(rho * dz, 1e-10, None)
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:119:    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:122:    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:124:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:80:    tau_0 = jnp.clip(tau_0, 0.0, None)
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:91:    def scan_fn(carry, k_rev):
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:102:    _, drag_stack = jax.lax.scan(scan_fn, tau_0, jnp.arange(nlev))
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:104:    drag_all = drag_stack.T  # (ncol, nlev)
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:105:    drag_all = drag_all[:, ::-1]  # back to top-first ordering
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:110:    accel = -drag_all / (jnp.clip(rho * dz, 1e-10, None))
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:117:    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:120:    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:122:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:75:    U_activated = jax.nn.sigmoid(20.0 * (U_ll - config.min_wind)) * U_ll
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:86:    tau_0 = jnp.clip(tau_0, 0.0, 10.0)
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:96:    def scan_fn(carry, k_rev):
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:100:        alpha = 50.0
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:108:    _, drag_stack = jax.lax.scan(scan_fn, tau_0, jnp.arange(nlev))
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:109:    drag_all = drag_stack.T[:, ::-1]  # (ncol, nlev), top-first
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:114:    accel = -drag_all / jnp.clip(rho * dz, 1e-10, None)
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:120:    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:123:    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:125:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:138:    def scan_fn(carry, k_rev):
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:148:    _, drag_stack = jax.lax.scan(scan_fn, F_init, jnp.arange(nlev))
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:164:    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:167:    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py:179:    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd), spectrum_new

codex
The scan order itself looks consistent with the stated top/surface convention: each scan walks from surface toward model top, then reverses the stacked deposits back to top-first column order. The more serious problems are in what is carried through the scan, not in the final transpose/reverse.
exec
/bin/zsh -lc "python - <<'PY'
import jax.numpy as jnp
from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.config import LindzenConfig, McFarlaneConfig, HinesConfig

nlev=6
p_full=jnp.array([[15000.,25000.,40000.,60000.,80000.,95000.]])
p_half=jnp.array([[10000.,20000.,32500.,50000.,70000.,87500.,100000.]])
z_full=jnp.array([[15000.,12000.,9000.,6000.,3000.,500.]])
z_half=jnp.array([[17000.,13500.,10500.,7500.,4500.,1500.,0.]])
rho=jnp.array([[0.2,0.3,0.45,0.65,0.9,1.2]])
T=jnp.array([[230.,240.,250.,260.,275.,290.]])
v=jnp.zeros((1,nlev))
lat=jnp.zeros((1,))
# sign reversal above low levels
u=jnp.array([[-20.,-20.,-20.,10.,10.,10.]])
for name, fn, cfg in [
    ('lindzen', lindzen_gwd, LindzenConfig(h_topo=2000.0)),
    ('mcfarlane', mcfarlane_gwd, McFarlaneConfig(h_topo=2000.0)),
    ('hines', hines_gwd, HinesConfig()),
]:
    out=fn(u,v,T,p_full,p_half,z_full,z_half,rho,lat,300.0,cfg)
    print(name)
    print('du', out.du_dt)
    print('u du', u*out.du_dt)
    print('dT', out.dT_dt)
    print('eps', out.eps_gwd)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 127 in 0ms:
zsh:1: command not found: python

codex
I tried to exercise a reversed-wind column to check the heating sign dynamically, but this environment does not expose `python` on PATH. I’ll keep the sign assessment grounded in the algebra from the source unless `python3` is available.
exec
/bin/zsh -lc "python3 - <<'PY'
import jax.numpy as jnp
from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.config import LindzenConfig, McFarlaneConfig, HinesConfig

nlev=6
p_full=jnp.array([[15000.,25000.,40000.,60000.,80000.,95000.]])
p_half=jnp.array([[10000.,20000.,32500.,50000.,70000.,87500.,100000.]])
z_full=jnp.array([[15000.,12000.,9000.,6000.,3000.,500.]])
z_half=jnp.array([[17000.,13500.,10500.,7500.,4500.,1500.,0.]])
rho=jnp.array([[0.2,0.3,0.45,0.65,0.9,1.2]])
T=jnp.array([[230.,240.,250.,260.,275.,290.]])
v=jnp.zeros((1,nlev))
lat=jnp.zeros((1,))
u=jnp.array([[-20.,-20.,-20.,10.,10.,10.]])
for name, fn, cfg in [
    ('lindzen', lindzen_gwd, LindzenConfig(h_topo=2000.0)),
    ('mcfarlane', mcfarlane_gwd, McFarlaneConfig(h_topo=2000.0)),
    ('hines', hines_gwd, HinesConfig()),
]:
    out=fn(u,v,T,p_full,p_half,z_full,z_half,rho,lat,300.0,cfg)
    print(name)
    print('du', out.du_dt)
    print('u du', u*out.du_dt)
    print('dT', out.dT_dt)
    print('eps', out.eps_gwd)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 895ms:
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
ModuleNotFoundError: No module named 'legoesm'

exec
/bin/zsh -lc "PYTHONPATH=src python3 - <<'PY'
import jax.numpy as jnp
from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.config import LindzenConfig, McFarlaneConfig, HinesConfig

nlev=6
p_full=jnp.array([[15000.,25000.,40000.,60000.,80000.,95000.]])
p_half=jnp.array([[10000.,20000.,32500.,50000.,70000.,87500.,100000.]])
z_full=jnp.array([[15000.,12000.,9000.,6000.,3000.,500.]])
z_half=jnp.array([[17000.,13500.,10500.,7500.,4500.,1500.,0.]])
rho=jnp.array([[0.2,0.3,0.45,0.65,0.9,1.2]])
T=jnp.array([[230.,240.,250.,260.,275.,290.]])
v=jnp.zeros((1,nlev))
lat=jnp.zeros((1,))
u=jnp.array([[-20.,-20.,-20.,10.,10.,10.]])
for name, fn, cfg in [
    ('lindzen', lindzen_gwd, LindzenConfig(h_topo=2000.0)),
    ('mcfarlane', mcfarlane_gwd, McFarlaneConfig(h_topo=2000.0)),
    ('hines', hines_gwd, HinesConfig()),
]:
    out=fn(u,v,T,p_full,p_half,z_full,z_half,rho,lat,300.0,cfg)
    print(name)
    print('du', out.du_dt)
    print('u du', u*out.du_dt)
    print('dT', out.dT_dt)
    print('eps', out.eps_gwd)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 771ms:
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/__init__.py", line 3, in <module>
    from legoesm.atmosphere.physics.radiation import (
    ...<2 lines>...
    )
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/__init__.py", line 40, in <module>
    from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import rrtmgp_radiation
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/rrtmgp_radiation.py", line 14, in <module>
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py", line 26, in <module>
    from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp_common
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp_common.py", line 17, in <module>
    from legoesm.atmosphere.physics.radiation.rrtmgp.config import radiative_transfer
  File "/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/radiation/rrtmgp/config/radiative_transfer.py", line 18, in <module>
    import dataclasses_json  # Used for JSON serialization.
    ^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'dataclasses_json'

codex
**Findings**

1. `src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:83-108` — Hines WKB growth is compounded incorrectly.  
Technical explanation: `sigma_new` is carried from the previous level, then multiplied by the cumulative surface-to-level `rho_ratio[:, k]` at every step. Without dissipation this gives a product of cumulative ratios, not the expected single `sqrt(rho_sfc / rho_k)` profile; the step growth should be adjacent-level `sqrt(rho[:, k + 1] / rho[:, k])` or the carry must be kept in a consistent source-normalized space.  
Severity: critical

2. `src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:78-96` — Hines saturation threshold is density-scaled in the wrong space.  
Technical explanation: `rho_ratio` increases aloft, so `N_full / (m_star * rho_ratio)` lowers the local velocity cap with altitude. If `sigma_sat` is a local wave-velocity saturation amplitude, it should not be divided by cumulative density growth; if it is source-normalized, the comparison and returned carry are inconsistent with `sigma_grown`.  
Severity: major

3. `src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:93-100` — Hines smooth limiter can create negative dissipation.  
Technical explanation: For `sigma_grown < sigma_sat`, the sigmoid is still positive and the blend can make `sigma_new > sigma_grown`. Then `drag = sigma_grown - sigma_new` is negative and the clip explicitly permits negative drag, allowing local wind acceleration and cooling.  
Severity: major

4. `src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:98-111`, `src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:121-130` — Hines “drag” has inconsistent stress units.  
Technical explanation: `(sigma_grown - sigma_new) * rho` has units of momentum density flux-like mass rate, not Pa stress, yet it is clipped by `Fmax` documented as Pa and divided by `rho * dz` as if it were a stress divergence. The resulting tendency is dimensionally missing a velocity/time-scale factor.  
Severity: major

5. `tests/unit/test_physics_gwd.py:1-131` — No GWD gradient tests are present.  
Technical explanation: The file checks sign, zero wind, magnitude, finite outputs, and Rayleigh structure, but contains no `jax.grad`, `value_and_grad`, `jac*`, `jvp`, `vjp`, or finite-difference differentiability checks. That leaves the advertised differentiable scan/sigmoid/softmin behavior unpinned.  
Severity: minor

6. `src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:75,96-104`, `src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:73-105` — McFarlane smoothing constants are hardcoded.  
Technical explanation: The min-wind sigmoid sharpness `20.0` and softmin `alpha = 50.0` control transition width, artificial softmin leakage, and gradients, but are not configurable. These are numerical/physical tuning parameters embedded in the implementation body.  
Severity: minor

7. `src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:84-86`, `src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:73-105` — McFarlane launch stress is hard-capped at 10.  
Technical explanation: `tau_0 = jnp.clip(tau_0, 0.0, 10.0)` silently imposes a fixed upper stress bound with no config field or unit documentation. High topography or launch-flux configurations will hit a nonphysical ceiling and lose gradients above it.  
Severity: major

8. `src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:84-92`, `src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:87-105` — McFarlane stress formulas are dimensionally suspect.  
Technical explanation: `G_0` is documented as a launch momentum flux scale `[Pa]`, but it is multiplied by `U * h_topo**2 * N * rho`; likewise `tau_sat` omits the horizontal wavenumber factor present in Lindzen and is not Pa unless `envelope_scale` secretly carries length units. This makes the hard cap likely mask unit/scaling errors.  
Severity: major

9. `src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:73-85,112-120`, `src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:77-93,116-123` — Orographic schemes can violate dissipative heating in reversed-flow layers.  
Technical explanation: Both schemes compute saturation from `abs(U_proj)` but apply acceleration along the low-level launch direction, so where local projected wind reverses, positive stress deposition can increase local kinetic energy. The formula `dT_dt = -(u du_dt + v dv_dt) / c_pd` has the right sign only if the tendency removes local kinetic energy.  
Severity: major

**Rebutted Checks**

`lindzen.py:91-105`, `mcfarlane.py:96-109`, and `hines.py:85-108`: the scan/transpose/reverse ordering is correct for the stated convention. The scan walks surface-to-top with `k = nlev - 1 - k_rev`, then reverses the stacked output back to top-first order; Hines’ problem is the carried amplitude physics, not the final ordering.

`rayleigh.py:82-87`, `lindzen.py:116-120`, `mcfarlane.py:119-123`, `hines.py:118-122`: the frictional-heating sign formula itself is correct for true drag. The substantive issue is that Hines and reversed-flow Lindzen/McFarlane paths do not always guarantee true local kinetic-energy loss.
2026-05-01T12:04:11.755532Z ERROR codex_core::session: failed to record rollout items: thread 019de368-fa2b-71b1-8920-cf6bcf7b7202 not found
tokens used
49,608
**Findings**

1. `src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:83-108` — Hines WKB growth is compounded incorrectly.  
Technical explanation: `sigma_new` is carried from the previous level, then multiplied by the cumulative surface-to-level `rho_ratio[:, k]` at every step. Without dissipation this gives a product of cumulative ratios, not the expected single `sqrt(rho_sfc / rho_k)` profile; the step growth should be adjacent-level `sqrt(rho[:, k + 1] / rho[:, k])` or the carry must be kept in a consistent source-normalized space.  
Severity: critical

2. `src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:78-96` — Hines saturation threshold is density-scaled in the wrong space.  
Technical explanation: `rho_ratio` increases aloft, so `N_full / (m_star * rho_ratio)` lowers the local velocity cap with altitude. If `sigma_sat` is a local wave-velocity saturation amplitude, it should not be divided by cumulative density growth; if it is source-normalized, the comparison and returned carry are inconsistent with `sigma_grown`.  
Severity: major

3. `src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:93-100` — Hines smooth limiter can create negative dissipation.  
Technical explanation: For `sigma_grown < sigma_sat`, the sigmoid is still positive and the blend can make `sigma_new > sigma_grown`. Then `drag = sigma_grown - sigma_new` is negative and the clip explicitly permits negative drag, allowing local wind acceleration and cooling.  
Severity: major

4. `src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:98-111`, `src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:121-130` — Hines “drag” has inconsistent stress units.  
Technical explanation: `(sigma_grown - sigma_new) * rho` has units of momentum density flux-like mass rate, not Pa stress, yet it is clipped by `Fmax` documented as Pa and divided by `rho * dz` as if it were a stress divergence. The resulting tendency is dimensionally missing a velocity/time-scale factor.  
Severity: major

5. `tests/unit/test_physics_gwd.py:1-131` — No GWD gradient tests are present.  
Technical explanation: The file checks sign, zero wind, magnitude, finite outputs, and Rayleigh structure, but contains no `jax.grad`, `value_and_grad`, `jac*`, `jvp`, `vjp`, or finite-difference differentiability checks. That leaves the advertised differentiable scan/sigmoid/softmin behavior unpinned.  
Severity: minor

6. `src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:75,96-104`, `src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:73-105` — McFarlane smoothing constants are hardcoded.  
Technical explanation: The min-wind sigmoid sharpness `20.0` and softmin `alpha = 50.0` control transition width, artificial softmin leakage, and gradients, but are not configurable. These are numerical/physical tuning parameters embedded in the implementation body.  
Severity: minor

7. `src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:84-86`, `src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:73-105` — McFarlane launch stress is hard-capped at 10.  
Technical explanation: `tau_0 = jnp.clip(tau_0, 0.0, 10.0)` silently imposes a fixed upper stress bound with no config field or unit documentation. High topography or launch-flux configurations will hit a nonphysical ceiling and lose gradients above it.  
Severity: major

8. `src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:84-92`, `src/legoesm/atmosphere/physics/gravity_wave_drag/config.py:87-105` — McFarlane stress formulas are dimensionally suspect.  
Technical explanation: `G_0` is documented as a launch momentum flux scale `[Pa]`, but it is multiplied by `U * h_topo**2 * N * rho`; likewise `tau_sat` omits the horizontal wavenumber factor present in Lindzen and is not Pa unless `envelope_scale` secretly carries length units. This makes the hard cap likely mask unit/scaling errors.  
Severity: major

9. `src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py:73-85,112-120`, `src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py:77-93,116-123` — Orographic schemes can violate dissipative heating in reversed-flow layers.  
Technical explanation: Both schemes compute saturation from `abs(U_proj)` but apply acceleration along the low-level launch direction, so where local projected wind reverses, positive stress deposition can increase local kinetic energy. The formula `dT_dt = -(u du_dt + v dv_dt) / c_pd` has the right sign only if the tendency removes local kinetic energy.  
Severity: major

**Rebutted Checks**

`lindzen.py:91-105`, `mcfarlane.py:96-109`, and `hines.py:85-108`: the scan/transpose/reverse ordering is correct for the stated convention. The scan walks surface-to-top with `k = nlev - 1 - k_rev`, then reverses the stacked output back to top-first order; Hines’ problem is the carried amplitude physics, not the final ordering.

`rayleigh.py:82-87`, `lindzen.py:116-120`, `mcfarlane.py:119-123`, `hines.py:118-122`: the frictional-heating sign formula itself is correct for true drag. The substantive issue is that Hines and reversed-flow Lindzen/McFarlane paths do not always guarantee true local kinetic-energy loss.
