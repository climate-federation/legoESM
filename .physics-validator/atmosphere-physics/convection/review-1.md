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
session id: 019de356-ae20-7f21-8138-42b7a610f27e
--------
user
# Adversarial review: legoESM convection package

You are an independent adversarial physics reviewer for an Earth-system model. The codebase under review is a JAX implementation of atmospheric convection schemes that must be:

- Physically correct (units, signs, conservation, monotonicity).
- Differentiable end-to-end (no hard `if` on traced values, no NaN gradients, no unintended dead-gradient regions).
- Numerically stable (avoid divisions by zero, AD-safe limiters).

Your job: read the source files listed below and find every concrete bug, sign error, unit inconsistency, broken-gradient pattern, conservation violation, and AD-safety issue. Cite line numbers from the file paths I will give. If you believe a candidate concern is NOT a bug, explain why concretely. Be specific.

Static analysis already produced (please critique these as hypotheses, not as established facts):

1. **Bechtold/Tiedtke `0.05` evap proxy hardcoded in body** (`bechtold.py:283-294`, `tiedtke.py:283-287`). Style violation; not a numerical bug per-se but should be in config.

2. **Bechtold MC enhancement uses hardcoded `0.05 kg/m²/s` normalizer** (`bechtold.py:201`). Same.

3. **DCA mass conservation via column-rescale** (`dca.py:257-265`). Used to rescale per-level condensation candidate so column integral matches column-net drying. Allegedly preserves column water but distributes per-level non-physically.

4. **Plume integrator `entraining_detraining_plume` carry vs reported `M_u` decoupling** (`_plume.py:484-542`): claims `above_base_weight` and `plume_alive` are reporting filters only, NOT folded into the carry. Question: does the carry's `M_u_raw` correctly reflect entrainment/detrainment dynamics absent these filters?

5. **`compute_lcl` Bolton 1980 formula** (`_plume.py:122-190`). Floor `T - 55 ≥ 1` may produce a non-physical LCL for very cold parcels. Check.

6. **`smooth_lowest_crossing_index` no-crossing fallback** (`_triggers.py:201-327`): blends to the surface index when no crossing found. Question: is the gate sharpness adequate?

7. **Mass-flux kernel uses smoothed pressure gate `stratosphere_mass_flux_gate`** (`mass_flux.py:132-159`): defaults to 100 hPa with 15 hPa width. Sigmoid never returns hard zero. At the model top, a small fraction (factor ~0.013) of the mass flux still lands. Acceptable?

8. **Implicit-Euler relaxation of `M_u_new`** (`tiedtke.py:220`, `bechtold.py:247`, `zhang_mcfarlane.py:142`): `M_u_new = (M_u_old + (dt/τ) M_eq) / (1 + dt/τ)`. Always stable for any dt/τ. Then capped at `M_b_max`. Verify the cap timing — does capping after relaxation mask any other issue?

9. **Per-class `delta_0_eff` rescale in tiedtke/bechtold** (`tiedtke.py:234-260`, `bechtold.py:254-272`): kernel called with nominal `config.delta_deep`, then `dT_dt`/`dq_v_dt` rescaled by `delta_0_eff / config.delta_deep`. Does this preserve sign and magnitude correctly?

10. **Emanuel buoyancy-sort `sort_multiplier ∈ [1, 1+cu]`** (`emanuel.py:160-170`): claim is that `4 * var(asc_weight) ≤ 1`. Verify.

Files to read (full source; please cite line numbers):

- `src/legoesm/atmosphere/physics/convection/_triggers.py` (363 LOC).
- `src/legoesm/atmosphere/physics/convection/_plume.py` (673 LOC).
- `src/legoesm/atmosphere/physics/convection/mass_flux.py` (495 LOC).
- `src/legoesm/atmosphere/physics/convection/sbm.py` (165 LOC).
- `src/legoesm/atmosphere/physics/convection/dca.py` (277 LOC).
- `src/legoesm/atmosphere/physics/convection/kuo.py` (217 LOC).
- `src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py` (220 LOC).
- `src/legoesm/atmosphere/physics/convection/kain_fritsch.py` (241 LOC).
- `src/legoesm/atmosphere/physics/convection/emanuel.py` (258 LOC).
- `src/legoesm/atmosphere/physics/convection/tiedtke.py` (316 LOC).
- `src/legoesm/atmosphere/physics/convection/bechtold.py` (322 LOC).

Convention: column shape `(ncol, nlev)`, surface at `[:, -1]`, top at `[:, 0]`. Pressure increases with index.

Physical constants (`from legoesm import constants`): `R_d=287.04`, `c_pd=1004.64`, `g=9.80616`, `L_v=2.501e6`, `R_v=461.5`, `T_freeze=273.15`, `kappa=R_d/c_pd`, `p_ref=1e5`. `constants.epsilon = R_d/R_v ≈ 0.622`.

For each finding:
- File path : line range.
- One-sentence summary of the concern.
- Brief technical explanation (≤3 sentences).
- Severity: `critical` | `major` | `minor`.
- If the concern is **already-flagged-as-not-a-bug** (the 10 hypotheses above), explicitly say so.

If you find no substantive issues beyond the 10 already flagged, end with: `NO ADDITIONAL SUBSTANTIVE FINDINGS`.

codex
I’ll read the listed convection files directly and treat the static-analysis items as hypotheses. I’m going to focus on line-cited behavioral issues: conservation, signs/units, differentiability, and numerics.
  → Inspect convection source files with line numbers
  • Evaluate the 10 supplied hypotheses against code
  • Identify additional physics, AD, and numerical issues
  • Prepare line-cited adversarial findings
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/_triggers.py | sed -n '1,380p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/mass_flux.py | sed -n '1,540p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/_plume.py | sed -n '1,720p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Smooth (differentiable) trigger and level-membership primitives.
     2	
     3	Shared helpers used by every convection scheme that needs a smooth
     4	replacement for a hard ``if``/``where`` branch in the reference paper.
     5	The legoESM model is trained via :func:`jax.grad` and
     6	``eqx.filter_value_and_grad`` — every trigger threshold therefore must
     7	have a non-zero gradient on both sides of the threshold so that
     8	:class:`legoesm.training.physics_params.TrainablePhysicsParams` and the
     9	neural-physics blend can flow gradients through the convection step.
    10	
    11	The conceptual split with :mod:`._plume`:
    12	
    13	* This module is **sigmoid math**: scalar-to-scalar smooth
    14	  approximations of step / max / positive-part / lowest-crossing-index
    15	  primitives, plus a thin convenience wrapper for the CAPE > threshold
    16	  trigger reused in four (now nine) convection backends.
    17	* :mod:`._plume` is **column physics math**: LCL / LFC / LNB / CIN
    18	  diagnostics, the entraining-detraining plume integrator,
    19	  unsaturated-downdraft thermodynamics, the Emanuel buoyancy-sorting
    20	  step, and the Gregory et al. 1997 convective momentum-transport
    21	  closure.
    22	
    23	The level-membership and lowest-crossing helpers consume profiles in
    24	the canonical legoESM convention: shape ``(ncol, nlev)`` with the
    25	**surface at the last index** ``[:, -1]`` and the model top at
    26	``[:, 0]``.  Indices returned by :func:`smooth_lowest_crossing_index`
    27	are in the same surface-last convention (``nlev - 1`` is the surface,
    28	``0`` is the top).
    29	
    30	References
    31	----------
    32	- Pattern for the soft-crossing × log-sum-exp softmin transcribed from
    33	  ``legoesm.atmosphere.physics.turbulence.pbl_height.diagnose_pbl_height_interp``.
    34	"""
    35	
    36	from __future__ import annotations
    37	
    38	import jax
    39	import jax.numpy as jnp
    40	
    41	
    42	__all__ = (
    43	    "smooth_step",
    44	    "smooth_heaviside",
    45	    "smooth_max",
    46	    "smooth_min",
    47	    "smooth_positive_part",
    48	    "smooth_level_indicator",
    49	    "smooth_lowest_crossing_index",
    50	    "cape_trigger",
    51	)
    52	
    53	
    54	# ---------------------------------------------------------------------------
    55	# Scalar / elementwise smooth primitives
    56	# ---------------------------------------------------------------------------
    57	
    58	def smooth_step(x: jax.Array, sharpness: float = 1.0) -> jax.Array:
    59	    """Differentiable approximation of the unit step function.
    60	
    61	    ``smooth_step(x, s) = sigmoid(s * x) ∈ (0, 1)``.
    62	
    63	    The result tends to ``Heaviside(x)`` as ``sharpness → ∞`` while
    64	    remaining strictly differentiable for any finite ``sharpness``.
    65	    Gradient at ``x = 0`` is ``sharpness / 4``.
    66	
    67	    Parameters
    68	    ----------
    69	    x : jax.Array
    70	        Argument (typically a difference ``value - threshold``).
    71	    sharpness : float
    72	        Inverse-width of the transition.  Larger values approach a
    73	        sharp step; smaller values broaden the transition.
    74	
    75	    Returns
    76	    -------
    77	    jax.Array
    78	        Same shape and dtype family as ``x``, values in ``(0, 1)``.
    79	    """
    80	    return jax.nn.sigmoid(sharpness * x)
    81	
    82	
    83	def smooth_heaviside(x: jax.Array, sharpness: float = 1.0) -> jax.Array:
    84	    """Alias for :func:`smooth_step` for callers that prefer the
    85	    Heaviside name at trigger sites."""
    86	    return smooth_step(x, sharpness)
    87	
    88	
    89	def smooth_max(
    90	    a: jax.Array,
    91	    b: jax.Array,
    92	    sharpness: float = 1.0,
    93	) -> jax.Array:
    94	    """Differentiable upper bound on ``max(a, b)``.
    95	
    96	    Implements the log-sum-exp soft-max ::
    97	
    98	        smooth_max(a, b, s) = (1/s) * log(exp(s*a) + exp(s*b)).
    99	
   100	    Properties:
   101	
   102	    * ``smooth_max(a, b, s) >= max(a, b)`` for all finite ``s > 0``;
   103	      the inequality tightens to equality as ``s → ∞``.
   104	    * Symmetric in ``a`` and ``b``.
   105	    * Gradients are well-defined everywhere, including at ``a == b``.
   106	
   107	    Implemented with :func:`jax.numpy.logaddexp` for numerical
   108	    stability across magnitudes (avoids ``exp`` overflow).
   109	    """
   110	    return jnp.logaddexp(sharpness * a, sharpness * b) / sharpness
   111	
   112	
   113	def smooth_min(
   114	    a: jax.Array,
   115	    b: jax.Array,
   116	    sharpness: float = 1.0,
   117	) -> jax.Array:
   118	    """Differentiable lower bound on ``min(a, b)``.
   119	
   120	    ``smooth_min(a, b, s) = -smooth_max(-a, -b, s)``.
   121	    """
   122	    return -smooth_max(-a, -b, sharpness)
   123	
   124	
   125	def smooth_positive_part(
   126	    x: jax.Array,
   127	    sharpness: float = 1.0,
   128	) -> jax.Array:
   129	    """Differentiable approximation of ``max(x, 0)``.
   130	
   131	    Returns ``softplus(sharpness * x) / sharpness``, equivalent to
   132	    :func:`smooth_max` against zero.  The gradient at ``x = 0`` is
   133	    ``0.5``, transitioning smoothly to ``1`` for ``x ≫ 0`` and ``0``
   134	    for ``x ≪ 0``.
   135	
   136	    Used wherever the reference paper writes ``(... )_+`` — most
   137	    notably the CAPE-relaxation cloud-base mass-flux closure
   138	    ``M_b ∝ (CAPE - CAPE_threshold)_+ / tau``.
   139	
   140	    Parameters
   141	    ----------
   142	    x : jax.Array
   143	        Argument.  Any shape.
   144	    sharpness : float
   145	        Larger values approach the kink at ``x = 0`` more sharply.
   146	
   147	    Returns
   148	    -------
   149	    jax.Array
   150	        Strictly positive everywhere; tends to ``max(x, 0)`` as
   151	        ``sharpness → ∞``.
   152	    """
   153	    return jax.nn.softplus(sharpness * x) / sharpness
   154	
   155	
   156	# ---------------------------------------------------------------------------
   157	# Per-level membership and crossing diagnostics
   158	# ---------------------------------------------------------------------------
   159	
   160	def smooth_level_indicator(
   161	    profile: jax.Array,
   162	    threshold: jax.Array | float,
   163	    sharpness: float,
   164	    direction: str = "above",
   165	) -> jax.Array:
   166	    """Per-level smooth membership in ``[0, 1]``.
   167	
   168	    Returns a per-level weight that smoothly transitions from 0 to 1
   169	    where the profile crosses the threshold.  Useful for
   170	    mass-weighted averages over a column subset (e.g. PBL-mean
   171	    parcel for the Bechtold/IFS departure-CAPE closure).
   172	
   173	    Parameters
   174	    ----------
   175	    profile : jax.Array, shape ``(..., nlev)``
   176	        Profile to test against the threshold.  Surface at last index.
   177	    threshold : jax.Array or float
   178	        Threshold value.  Broadcastable against ``profile``.
   179	    sharpness : float
   180	        Sigmoid sharpness on the threshold transition.
   181	    direction : str
   182	        ``"above"`` returns ``sigmoid(sharpness * (profile - threshold))``
   183	        (1 where profile is above threshold, 0 below).
   184	        ``"below"`` is the complement: 1 where profile is below
   185	        threshold (used to weight a PBL-depth average).
   186	
   187	    Returns
   188	    -------
   189	    jax.Array
   190	        Same shape as ``profile``; values in ``(0, 1)``.
   191	    """
   192	    if direction == "above":
   193	        return jax.nn.sigmoid(sharpness * (profile - threshold))
   194	    if direction == "below":
   195	        return jax.nn.sigmoid(sharpness * (threshold - profile))
   196	    raise ValueError(
   197	        f"direction must be 'above' or 'below', got {direction!r}"
   198	    )
   199	
   200	
   201	def smooth_lowest_crossing_index(
   202	    profile: jax.Array,
   203	    threshold: jax.Array | float,
   204	    sharpness: float,
   205	) -> jax.Array:
   206	    """Differentiable fractional index of the lowest upward crossing.
   207	
   208	    Scans each column from the surface upward (last index → first
   209	    index) and returns the fractional level index of the lowest
   210	    altitude where ``profile`` crosses ``threshold`` from below to
   211	    above.  The fractional component is a linear interpolation
   212	    between adjacent levels; the choice of which crossing to return
   213	    when several exist is made by a log-sum-exp softmin so that
   214	    columns with multiple candidate crossings still produce a smooth
   215	    (differentiable) answer.
   216	
   217	    Used for LFC / LCL / LNB localization where the reference paper
   218	    would normally take the first satisfying integer index.
   219	
   220	    Parameters
   221	    ----------
   222	    profile : jax.Array, shape ``(ncol, nlev)``
   223	        Per-level profile.  Surface at last index ``[:, -1]``.
   224	    threshold : jax.Array or float
   225	        Threshold value.  Scalar, ``(ncol,)``, or
   226	        ``(ncol, nlev-1)``-broadcastable.
   227	    sharpness : float
   228	        Sigmoid sharpness on the soft crossing indicator AND the
   229	        scale parameter for the softmin (larger = closer to a hard
   230	        argmin).
   231	
   232	    Returns
   233	    -------
   234	    jax.Array, shape ``(ncol,)``
   235	        Fractional index in surface-last coordinates: a return value
   236	        of ``ncol_index = nlev - 1`` means the surface, and ``0``
   237	        means the model top.  When no crossing is detected the
   238	        returned value falls back to ``nlev - 1`` (surface) — callers
   239	        that need a "no-crossing" sentinel should use
   240	        :func:`smooth_level_indicator` to compute a column-wide
   241	        confidence first.
   242	    """
   243	    *_, nlev = profile.shape
   244	    if nlev < 2:
   245	        raise ValueError(
   246	            "smooth_lowest_crossing_index requires nlev >= 2; got "
   247	            f"profile.shape={profile.shape}"
   248	        )
   249	
   250	    # Reverse so that surface is first; matches the convention used in
   251	    # turbulence/pbl_height.py.
   252	    profile_rev = profile[..., ::-1]      # (ncol, nlev), surface-first
   253	
   254	    profile_below = profile_rev[..., :-1]  # (ncol, nlev-1)
   255	    profile_above = profile_rev[..., 1:]
   256	
   257	    # Soft upward-crossing indicator: simultaneously below threshold
   258	    # at the lower level and above at the upper level.
   259	    cross_weight = (
   260	        jax.nn.sigmoid(sharpness * (threshold - profile_below))
   261	        * jax.nn.sigmoid(sharpness * (profile_above - threshold))
   262	    )
   263	
   264	    # Linear interpolation fraction within each pair (in [0, 1]).
   265	    dprofile = profile_above - profile_below
   266	    # Numerical floor avoids 0/0 when adjacent levels are equal.
   267	    safe_d = jnp.where(jnp.abs(dprofile) > 1e-30, dprofile, 1e-30)
   268	    frac = jnp.clip((threshold - profile_below) / safe_d, 0.0, 1.0)
   269	
   270	    # Surface-first fractional indices of each candidate crossing.
   271	    # Pair k connects level k (below) and level k+1 (above), so the
   272	    # interpolated index is k + frac.
   273	    k_pairs = jnp.arange(nlev - 1, dtype=profile.dtype)
   274	    idx_surface_first = k_pairs + frac    # (ncol, nlev-1)
   275	
   276	    # First-crossing probability at each pair, computed as the
   277	    # sequential survival product
   278	    #
   279	    #     first_cross[k] = cross_weight[k] * Π_{j<k}(1 - cross_weight[j])
   280	    #
   281	    # This is the probability (under the soft-crossing model) that
   282	    # pair ``k`` is the *first* upward crossing — it naturally
   283	    # suppresses higher-altitude crossings even when they are
   284	    # individually strong.  Unlike a global log-sum-exp softmin,
   285	    # this formulation is robust to multiple actual crossings and to
   286	    # vanishingly-small but nonzero cross_weight at non-crossing
   287	    # levels (those contribute ``cross_weight ≈ 0`` and drop out
   288	    # multiplicatively).
   289	    not_yet_crossed = jnp.concatenate(
   290	        [
   291	            jnp.ones(profile.shape[:-1] + (1,), dtype=profile.dtype),
   292	            jnp.cumprod(1.0 - cross_weight, axis=-1)[..., :-1],
   293	        ],
   294	        axis=-1,
   295	    )
   296	    first_cross_weight = cross_weight * not_yet_crossed
   297	
   298	    # Weighted average of fractional indices.  ``maximum`` with a
   299	    # tiny floor protects against ``0/0`` when ``total_first_cross``
   300	    # is exactly zero; the no-crossing branch below replaces this
   301	    # value with the surface fallback when the gate fires.
   302	    total_first_cross = jnp.sum(first_cross_weight, axis=-1)
   303	    safe_total = jnp.maximum(total_first_cross, 1e-12)
   304	    idx_min_naive = (
   305	        jnp.sum(first_cross_weight * idx_surface_first, axis=-1)
   306	        / safe_total
   307	    )
   308	
   309	    # No-crossing fallback: blend toward the surface index (which is
   310	    # ``0`` in surface-first coordinates) when ``total_first_cross``
   311	    # is small.  Uses a sharpness independent of the user-supplied
   312	    # ``sharpness`` so the gate is consistent across calls — gate
   313	    # sharpness ``20`` around midpoint ``0.5`` gives ``≈ 0%`` trust at
   314	    # ``total = 0`` and ``≈ 100%`` trust at ``total = 1``.  At total
   315	    # values typical of a single firm crossing (~0.8–1.0) the gate
   316	    # fully selects the weighted-average answer.
   317	    no_cross_blend = jax.nn.sigmoid(20.0 * (total_first_cross - 0.5))
   318	    fallback = jnp.full_like(idx_min_naive, 0.0)  # surface-first 0 = surface-last nlev-1
   319	    idx_min_surface_first = (
   320	        no_cross_blend * idx_min_naive
   321	        + (1.0 - no_cross_blend) * fallback
   322	    )
   323	
   324	    # Convert to surface-last (canonical legoESM) coordinates:
   325	    #   surface-first index 0       -> surface-last index nlev - 1
   326	    #   surface-first index nlev-1  -> surface-last index 0
   327	    return float(nlev - 1) - idx_min_surface_first
   328	
   329	
   330	# ---------------------------------------------------------------------------
   331	# Convenience wrapper used at convection-trigger call sites
   332	# ---------------------------------------------------------------------------
   333	
   334	def cape_trigger(
   335	    cape: jax.Array,
   336	    threshold: jax.Array | float,
   337	    sharpness: float,
   338	) -> jax.Array:
   339	    """Smooth ``CAPE > threshold`` indicator for convection triggers.
   340	
   341	    Equivalent to ``smooth_step(cape - threshold, sharpness)``.  The
   342	    same soft-trigger formula is duplicated inline in four legacy
   343	    schemes (``sbm``, ``dca``, ``kuo``, ``mass_flux``) and in every
   344	    new scheme added by PRs 1 – 5; this convenience wrapper centralizes
   345	    the semantics.
   346	
   347	    Parameters
   348	    ----------
   349	    cape : jax.Array
   350	        CAPE diagnostic [J/kg], any shape.
   351	    threshold : jax.Array or float
   352	        CAPE threshold below which convection is suppressed.
   353	    sharpness : float
   354	        Sigmoid sharpness in units of [1/(J/kg)].  Smaller values
   355	        produce a wider transition; larger values approach a hard
   356	        ``CAPE > threshold`` step.
   357	
   358	    Returns
   359	    -------
   360	    jax.Array
   361	        Same shape as ``cape``; values in ``(0, 1)``.
   362	    """
   363	    return smooth_step(cape - threshold, sharpness)

 succeeded in 0ms:
     1	"""Mass-flux convection schemes (Arakawa-Wu and simplified EDMF).
     2	
     3	This module provides two prognostic mass-flux convection
     4	parameterizations that share most of their numerics:
     5	
     6	1. **Prognostic Mass-Flux** (``mass_flux_convection``): Arakawa-Wu type
     7	   scheme with a single column-mean prognostic variable ``M_c``. The
     8	   vertical profile of the mass flux is a fixed sinusoidal shape;
     9	   ``M_c`` relaxes toward an equilibrium value diagnosed from CAPE.
    10	
    11	2. **Simplified EDMF** (``edmf_convection``): single-updraft mass-flux
    12	   scheme inspired by Siebesma et al. (2007). The prognostic variable
    13	   is the updraft area fraction ``a_u``; the mass-flux profile
    14	   ``M_u(z) = ρ · a_u · w_u(z)`` is built from a buoyancy-derived
    15	   updraft velocity. This is a reduced-complexity surrogate, not a
    16	   full EDMF (single plume, no downdrafts, no stochastic triggering,
    17	   no PDF closure).
    18	
    19	Both schemes share the same downstream kernel: given a vertical
    20	mass-flux profile ``M(z)``, an entraining/diluting updraft ``(T_u,
    21	q_u)``, and grid geometry, they apply the same compensating
    22	subsidence + detrainment tendencies and diagnose surface
    23	precipitation from the column integral of detrained condensate.
    24	The kernel is factored into ``_apply_mass_flux_kernel`` so the two
    25	schemes differ only in (a) the prognostic update rule and (b) the
    26	M(z) profile.
    27	
    28	All operations use smooth (differentiable) approximations for
    29	compatibility with ``jax.grad``.
    30	
    31	References
    32	----------
    33	- Arakawa, A., & Wu, C.-M. (2013). A unified representation of deep
    34	  moist convection in numerical modeling of the atmosphere. Part I.
    35	  J. Atmos. Sci., 70, 1977-1992.
    36	- Siebesma, A. P., et al. (2007). A combined eddy-diffusivity
    37	  mass-flux approach for the convective boundary layer.
    38	  J. Atmos. Sci., 64, 1230-1248.
    39	- Tiedtke, M. (1989). A comprehensive mass flux scheme for cumulus
    40	  parameterization in large-scale models. Mon. Wea. Rev., 117,
    41	  1779-1800.
    42	"""
    43	
    44	from __future__ import annotations
    45	
    46	from typing import NamedTuple, Tuple
    47	
    48	import jax
    49	import jax.numpy as jnp
    50	
    51	from legoesm import constants
    52	from legoesm.thermo import saturation_mixing_ratio
    53	from legoesm.atmosphere.physics.thermodynamics import (
    54	    compute_moist_adiabat,
    55	    compute_cape,
    56	)
    57	from legoesm.atmosphere.physics.convection.config import (
    58	    EDMFConfig,
    59	    MassFluxConfig,
    60	)
    61	from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    62	
    63	
    64	# =============================================================================
    65	# Shared helpers — used by both the Arakawa-Wu and simplified EDMF paths.
    66	# =============================================================================
    67	
    68	
    69	def _compute_column_geometry(
    70	    T: jax.Array,
    71	    p_full: jax.Array,
    72	    p_half: jax.Array,
    73	) -> Tuple[jax.Array, jax.Array, jax.Array]:
    74	    """Compute layer thickness, density, and surface-relative height.
    75	
    76	    Returns ``(dz, rho, z)``, all of shape ``(ncol, nlev)``. ``dz`` is
    77	    the layer thickness from hydrostatic balance using the mid-layer
    78	    pressure; ``rho`` is the dry-air density at full levels; ``z`` is
    79	    the cumulative height above the surface (note: levels are ordered
    80	    top-down, so ``z[:, -1]`` is the surface).
    81	    """
    82	    dp = p_half[:, 1:] - p_half[:, :-1]
    83	    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    84	    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    85	    dz = jnp.abs(dz)
    86	    rho = p_full / (constants.R_d * jnp.clip(T, 1.0, None))
    87	    # Cumulative height from the surface (level nlev-1) upward.
    88	    z = jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1]
    89	    return dz, rho, z
    90	
    91	
    92	def _compute_cape_diagnostics(
    93	    T: jax.Array,
    94	    p_full: jax.Array,
    95	    p_half: jax.Array,
    96	    cape_threshold: float,
    97	    cape_activation_scale: float,
    98	) -> Tuple[jax.Array, jax.Array, jax.Array]:
    99	    """Compute moist-adiabat profile, CAPE, and a smooth convective mask.
   100	
   101	    Returns ``(T_moist, cape, convective_mask)``. ``T_moist`` is shape
   102	    ``(ncol, nlev)``; ``cape`` and ``convective_mask`` are shape
   103	    ``(ncol,)``. The mask is a sigmoid of ``(cape -
   104	    cape_threshold) / cape_activation_scale`` and is reused as the
   105	    smooth activation factor for both schemes.
   106	    """
   107	    T_base = T[:, -1]
   108	    T_moist = compute_moist_adiabat(T_base, p_full)
   109	    cape = compute_cape(T, T_moist, p_full, p_half)
   110	    convective_mask = jax.nn.sigmoid(
   111	        (cape - cape_threshold) / cape_activation_scale
   112	    )
   113	    return T_moist, cape, convective_mask
   114	
   115	
   116	def _compute_centered_gradients(
   117	    T: jax.Array,
   118	    q_v: jax.Array,
   119	    z: jax.Array,
   120	) -> Tuple[jax.Array, jax.Array]:
   121	    """Centered vertical gradients with zero edges via ``jnp.pad``.
   122	
   123	    Single Pad HLO op vs. allocate-zeros + scatter. Returns
   124	    ``(dT_dz, dq_dz)``, both shape ``(ncol, nlev)``.
   125	    """
   126	    dz_centered = jnp.clip(z[:, :-2] - z[:, 2:], 1.0, None)
   127	    dT_dz = jnp.pad((T[:, :-2] - T[:, 2:]) / dz_centered, ((0, 0), (1, 1)))
   128	    dq_dz = jnp.pad((q_v[:, :-2] - q_v[:, 2:]) / dz_centered, ((0, 0), (1, 1)))
   129	    return dT_dz, dq_dz
   130	
   131	
   132	def stratosphere_mass_flux_gate(
   133	    p_full: jax.Array,
   134	    p_min_convection: float = 10_000.0,
   135	    p_gate_sharpness: float = 1_500.0,
   136	) -> jax.Array:
   137	    """Smooth sigmoid factor in [0, 1] that vanishes above the
   138	    tropopause (low ``p``) and equals one in the troposphere.
   139	
   140	    Multiplying any mass-flux profile by the returned factor prevents
   141	    convective tendencies from accumulating in the model top layer,
   142	    where the small mass per unit area (Δp/g) would amplify modest
   143	    heating into unphysical spikes (>400 K observed in 1-year RCE).
   144	
   145	    Defaults: cutoff at 100 hPa (canonical tropical tropopause) with
   146	    a 15-hPa transition width.  This gives factor ≈ 0.013 at the
   147	    model top (35 hPa), 0.034 at 50 hPa, 0.5 at 100 hPa, 0.91 at
   148	    130 hPa, and ≈ 1.0 below 200 hPa — i.e. the gate is *actually
   149	    closed* (not merely attenuated) in the deep stratosphere while
   150	    leaving the upper troposphere unaffected.  ``p_gate_sharpness``
   151	    must be << ``p_min_convection`` for the sigmoid to saturate
   152	    within the integration range; sharpness ≥ p_min only attenuates.
   153	
   154	    Differentiable everywhere; ``p_gate_sharpness`` sets the width of
   155	    the transition (Pa).
   156	    """
   157	    return jax.nn.sigmoid(
   158	        (p_full - p_min_convection) / jnp.maximum(p_gate_sharpness, 1.0)
   159	    )
   160	
   161	
   162	def _apply_mass_flux_kernel(
   163	    T: jax.Array,
   164	    q_v: jax.Array,
   165	    p_full: jax.Array,
   166	    T_u: jax.Array,
   167	    q_v_u: jax.Array,
   168	    q_c_u: jax.Array,
   169	    M_profile: jax.Array,
   170	    z: jax.Array,
   171	    rho: jax.Array,
   172	    delta_0: float,
   173	    M_u_max: float = 0.05,
   174	    p_min_convection: float = 10_000.0,
   175	    p_gate_sharpness: float = 1_500.0,
   176	) -> Tuple[jax.Array, jax.Array, jax.Array]:
   177	    """Mass-flux core kernel: compensating subsidence + detrainment.
   178	
   179	    Given the vertical mass-flux profile ``M_profile`` and the
   180	    entraining updraft thermodynamics ``(T_u, q_v_u, q_c_u)``,
   181	    returns ``(dT_dt, dq_v_dt, dq_c_conv_dt)`` — all shape
   182	    ``(ncol, nlev)``.
   183	
   184	    The decomposition follows Tiedtke (1989) / Siebesma et al. (2007):
   185	      (a) Compensating subsidence: ``(M/rho) * (dT/dz + g/c_p)`` for T
   186	          and ``(M/rho) * dq/dz`` for moisture (q is conserved so no
   187	          adiabatic correction).
   188	      (b) Detrainment mixing: ``+delta_0 * M * (X_u - X) / rho``.
   189	
   190	    The mass flux ``M_profile`` is gated by a smooth sigmoid in
   191	    pressure so that levels above ``p_min_convection`` (default 100
   192	    hPa, the canonical tropical tropopause) receive no convective
   193	    tendency.  See ``stratosphere_mass_flux_gate`` for details.
   194	
   195	    The convective source for cloud water is the per-level detrainment
   196	    of the plume's cloud water:
   197	    ``dq_c_conv_dt = delta_0 * M * q_c_u / rho`` [kg/kg/s], non-negative
   198	    by construction.  Splitting the plume into separate vapor (``q_v_u``)
   199	    and cloud (``q_c_u``) pieces — instead of a single ``q_u`` that
   200	    conflates total water with vapor — is what makes the column MSE
   201	    budget close.  The earlier formulation passed ``q_u = q_v_u + q_c_u``
   202	    as if it were vapor and computed condensate as ``max(q_u - q_sat,
   203	    0)``, which is essentially zero for an entraining-diluted plume —
   204	    the leaf then leaked latent energy.  Microphysics processes
   205	    ``dq_c_conv_dt`` through its full chain (autoconversion,
   206	    sedimentation, evaporation) and produces the resulting surface
   207	    precipitation; convection no longer assumes the condensate falls
   208	    instantly.
   209	    Unit check: (1/m) * (kg/m²/s) * (kg/kg) / (kg/m³) = 1/s × kg/kg.
   210	    """
   211	    dT_dz, dq_dz = _compute_centered_gradients(T, q_v, z)
   212	    rho_safe = jnp.clip(rho, 0.01, None)
   213	
   214	    # Per-level mass-flux cap.  The plume integrator can yield ``M_u``
   215	    # that grows with height when ``epsilon > delta`` (entraining
   216	    # plumes) or that responds non-linearly to a high-CAPE column.
   217	    # Per-layer convective heating ``≈ delta_0 · M_u · (T_u−T)/ρ``
   218	    # scales linearly with ``M_u``, so an uncapped ``M_u`` produces
   219	    # column heating well in excess of what surface fluxes can supply
   220	    # and destabilises the integration.  Clipping at the cap (default
   221	    # ``0.05 kg/m²/s``, the literature peak tropical updraft mass flux)
   222	    # bounds per-layer tendencies without distorting the moist adiabat
   223	    # or the q_v / q_c split.
   224	    M_profile = jnp.clip(M_profile, 0.0, M_u_max)
   225	
   226	    # Stratospheric pressure gate — see ``stratosphere_mass_flux_gate``.
   227	    M_profile = M_profile * stratosphere_mass_flux_gate(
   228	        p_full, p_min_convection, p_gate_sharpness,
   229	    )
   230	
   231	    dT_subsidence = (M_profile / rho_safe) * (dT_dz + constants.g / constants.c_pd)
   232	    dq_subsidence = (M_profile / rho_safe) * dq_dz
   233	
   234	    dT_detrain = delta_0 * M_profile * (T_u - T) / rho_safe
   235	    dq_detrain = delta_0 * M_profile * (q_v_u - q_v) / rho_safe
   236	
   237	    dT_dt = dT_subsidence + dT_detrain
   238	    dq_v_dt = dq_subsidence + dq_detrain
   239	
   240	    dq_c_conv_dt = delta_0 * M_profile * jnp.clip(q_c_u, 0.0, None) / rho_safe
   241	    return dT_dt, dq_v_dt, dq_c_conv_dt
   242	
   243	
   244	# =============================================================================
   245	# Arakawa-Wu prognostic mass-flux (M_c × fixed sinusoidal profile)
   246	# =============================================================================
   247	
   248	
   249	class MassFluxClosureDiagnostics(NamedTuple):
   250	    """Intermediate closure state reused by physical and ML mass-flux paths."""
   251	
   252	    dz: jax.Array
   253	    rho: jax.Array
   254	    z: jax.Array
   255	    T_moist: jax.Array
   256	    cape: jax.Array
   257	    M_eq: jax.Array
   258	    M_c_new: jax.Array
   259	    convective_mask: jax.Array
   260	
   261	
   262	def diagnose_mass_flux_closure(
   263	    T: jax.Array,
   264	    q_v: jax.Array,
   265	    p_full: jax.Array,
   266	    p_half: jax.Array,
   267	    M_c: jax.Array,
   268	    dt: float,
   269	    config: MassFluxConfig = MassFluxConfig(),
   270	) -> MassFluxClosureDiagnostics:
   271	    """Diagnose closure terms before computing mass-flux tendencies."""
   272	    del q_v  # retained for interface symmetry with full convection call
   273	
   274	    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
   275	    T_moist, cape, convective_mask = _compute_cape_diagnostics(
   276	        T, p_full, p_half, config.cape_threshold, config.cape_activation_scale,
   277	    )
   278	
   279	    M_eq = convective_mask * config.M_scale
   280	    M_c_new = jnp.maximum(M_c + dt * (M_eq - M_c) / config.tau_adj, 0.0)
   281	
   282	    return MassFluxClosureDiagnostics(
   283	        dz=dz,
   284	        rho=rho,
   285	        z=z,
   286	        T_moist=T_moist,
   287	        cape=cape,
   288	        M_eq=M_eq,
   289	        M_c_new=M_c_new,
   290	        convective_mask=convective_mask,
   291	    )
   292	
   293	
   294	def mass_flux_convection_from_closure(
   295	    T: jax.Array,
   296	    q_v: jax.Array,
   297	    p_full: jax.Array,
   298	    p_half: jax.Array,
   299	    closure: MassFluxClosureDiagnostics,
   300	    config: MassFluxConfig = MassFluxConfig(),
   301	) -> ConvectionOutput:
   302	    """Compute mass-flux tendencies from a supplied closure state."""
   303	    del p_half
   304	    dz = closure.dz
   305	    rho = closure.rho
   306	    z = closure.z
   307	    T_moist = closure.T_moist
   308	    cape = closure.cape
   309	    M_c_new = closure.M_c_new
   310	    convective_mask = closure.convective_mask
   311	
   312	    p_base = p_full[:, -1:]
   313	    p_top = p_full[:, :1]
   314	    p_range = jnp.clip(p_base - p_top, 1.0, None)
   315	    # Sinusoidal vertical profile: 0 at base/top, peak at mid-troposphere.
   316	    m_profile = jnp.sin(jnp.pi * (p_base - p_full) / p_range)
   317	    M_profile = M_c_new[:, None] * m_profile
   318	
   319	    # Entraining updraft.  The undiluted plume rises along the moist
   320	    # adiabat from the cloud-base parcel (T_base, q_sat_base): its
   321	    # vapor at level z is q_sat(T_moist(z), p), and its cloud water
   322	    # is the moisture lost during ascent ``q_sat_base − q_sat(T_moist)``.
   323	    # Entrainment dilutes both T and q_v with environmental values;
   324	    # the entrained env air carries no q_c, so q_c_u just scales by
   325	    # ``dilution``.
   326	    dilution = jnp.exp(-config.epsilon_0 * z)
   327	    T_u = dilution * T_moist + (1.0 - dilution) * T
   328	    q_sat_base = saturation_mixing_ratio(T[:, -1:], p_full[:, -1:])
   329	    q_sat_moist = saturation_mixing_ratio(T_moist, p_full)
   330	    q_v_u = dilution * q_sat_moist + (1.0 - dilution) * q_v
   331	    q_c_u = dilution * jnp.clip(q_sat_base - q_sat_moist, 0.0, None)
   332	
   333	    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
   334	        T=T,
   335	        q_v=q_v,
   336	        p_full=p_full,
   337	        T_u=T_u,
   338	        q_v_u=q_v_u,
   339	        q_c_u=q_c_u,
   340	        M_profile=M_profile,
   341	        z=z,
   342	        rho=rho,
   343	        delta_0=config.delta_0,
   344	        M_u_max=config.M_b_max,
   345	    )
   346	
   347	    return ConvectionOutput(
   348	        dT_dt=dT_dt,
   349	        dq_v_dt=dq_v_dt,
   350	        dq_c_conv_dt=dq_c_conv_dt,
   351	        cape=cape,
   352	        convective_mask=convective_mask,
   353	    )
   354	
   355	
   356	def mass_flux_convection(
   357	    T: jax.Array,
   358	    q_v: jax.Array,
   359	    p_full: jax.Array,
   360	    p_half: jax.Array,
   361	    M_c: jax.Array,
   362	    dt: float,
   363	    config: MassFluxConfig = MassFluxConfig(),
   364	) -> Tuple[ConvectionOutput, jax.Array]:
   365	    """Compute Prognostic Mass-Flux convection tendencies."""
   366	    closure = diagnose_mass_flux_closure(
   367	        T=T,
   368	        q_v=q_v,
   369	        p_full=p_full,
   370	        p_half=p_half,
   371	        M_c=M_c,
   372	        dt=dt,
   373	        config=config,
   374	    )
   375	    conv_out = mass_flux_convection_from_closure(
   376	        T=T,
   377	        q_v=q_v,
   378	        p_full=p_full,
   379	        p_half=p_half,
   380	        closure=closure,
   381	        config=config,
   382	    )
   383	    return conv_out, closure.M_c_new
   384	
   385	
   386	# =============================================================================
   387	# Simplified EDMF (M_u(z) = ρ · a_u · w_u(z) from a buoyancy integral)
   388	# =============================================================================
   389	
   390	
   391	def edmf_convection(
   392	    T: jax.Array,
   393	    q_v: jax.Array,
   394	    p_full: jax.Array,
   395	    p_half: jax.Array,
   396	    a_u: jax.Array,
   397	    dt: float,
   398	    config: EDMFConfig = EDMFConfig(),
   399	) -> Tuple[ConvectionOutput, jax.Array]:
   400	    """Compute simplified EDMF convection tendencies.
   401	
   402	    Parameters
   403	    ----------
   404	    T : jax.Array
   405	        Temperature at full levels [K], shape ``(ncol, nlev)``.
   406	    q_v : jax.Array
   407	        Water vapor specific humidity [kg/kg], shape ``(ncol, nlev)``.
   408	    p_full : jax.Array
   409	        Pressure at full levels [Pa], shape ``(ncol, nlev)``.
   410	    p_half : jax.Array
   411	        Pressure at half levels [Pa], shape ``(ncol, nlev+1)``.
   412	    a_u : jax.Array
   413	        Updraft area fraction, shape ``(ncol,)``.
   414	    dt : float
   415	        Model time step [s].
   416	    config : EDMFConfig
   417	        Convection configuration.
   418	
   419	    Returns
   420	    -------
   421	    ConvectionOutput
   422	        Convective tendencies and diagnostics.
   423	    a_u_new : jax.Array
   424	        Updated updraft area fraction, shape ``(ncol,)``.
   425	    """
   426	    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
   427	    T_moist, cape, convective_mask = _compute_cape_diagnostics(
   428	        T, p_full, p_half, config.cape_threshold, config.cape_activation_scale,
   429	    )
   430	
   431	    # Diagnosed equilibrium updraft area fraction; prognostic relaxation
   432	    # then clipping to a physically plausible range.
   433	    a_u_eq = convective_mask * config.a_u_init
   434	    a_u_new = a_u + dt * (a_u_eq - a_u) / config.tau_a
   435	    a_u_new = jnp.clip(a_u_new, 0.0, 0.5)
   436	
   437	    # Entraining updraft: thermal dilutes from the moist adiabat toward
   438	    # environment; moisture follows the moist-adiabat saturation profile
   439	    # (this differs from mass_flux, which dilutes from a single base
   440	    # parcel — a small but deliberate scientific distinction between
   441	    # the two schemes).
   442	    dilution = jnp.exp(-config.epsilon_0 * z)
   443	    T_u = dilution * T_moist + (1.0 - dilution) * T
   444	    q_sat_base = saturation_mixing_ratio(T[:, -1:], p_full[:, -1:])
   445	    q_sat_moist = saturation_mixing_ratio(T_moist, p_full)
   446	    # Diluted plume: vapor follows the saturated moist adiabat at each
   447	    # level (mixed with env vapor by entrainment), and cloud water is
   448	    # the integrated condensation lost during ascent (zero contribution
   449	    # from entrained env which carries no cloud water).
   450	    q_v_u = dilution * q_sat_moist + (1.0 - dilution) * q_v
   451	    q_c_u = dilution * jnp.clip(q_sat_base - q_sat_moist, 0.0, None)
   452	
   453	    # Buoyancy: B = g * (T_v_u - T_v_env) / T_v_env.  Vapor contributes
   454	    # ``+0.61 q_v`` (water vapour is lighter than dry air) and cloud
   455	    # water contributes ``-q_c`` (the loaded condensate is mass drag,
   456	    # not buoyancy).  An earlier formulation used
   457	    # ``T_v = T*(1 + 0.61*(q_v + q_c))`` which treated q_c with the
   458	    # *wrong sign* — making cloudy plumes spuriously buoyant — so we
   459	    # use the standard form here.
   460	    T_v_env = T * (1.0 + 0.61 * q_v)
   461	    T_v_u = T_u * (1.0 + 0.61 * q_v_u - q_c_u)
   462	    B = constants.g * (T_v_u - T_v_env) / jnp.clip(T_v_env, 1.0, None)
   463	
   464	    # Updraft velocity from buoyancy integral (surface upward), with a
   465	    # small floor for numerical stability.
   466	    B_dz_rev = jnp.clip(B * dz, 0.0, None)[:, ::-1]
   467	    B_integral = jnp.cumsum(B_dz_rev, axis=1)[:, ::-1]
   468	    w_u = jnp.sqrt(2.0 * B_integral + config.w_u_min ** 2)
   469	
   470	    # Mass flux profile: M_u(z) = rho * a_u * w_u(z).
   471	    M_profile = rho * a_u_new[:, None] * w_u
   472	
   473	    del dz  # (kept for interface symmetry — kernel no longer needs it)
   474	    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
   475	        T=T,
   476	        q_v=q_v,
   477	        p_full=p_full,
   478	        T_u=T_u,
   479	        q_v_u=q_v_u,
   480	        q_c_u=q_c_u,
   481	        M_profile=M_profile,
   482	        z=z,
   483	        rho=rho,
   484	        delta_0=config.delta_0,
   485	        M_u_max=config.M_b_max,
   486	    )
   487	
   488	    conv_out = ConvectionOutput(
   489	        dT_dt=dT_dt,
   490	        dq_v_dt=dq_v_dt,
   491	        dq_c_conv_dt=dq_c_conv_dt,
   492	        cape=cape,
   493	        convective_mask=convective_mask,
   494	    )
   495	    return conv_out, a_u_new

 succeeded in 0ms:
     1	"""Column physics primitives shared across convection schemes.
     2	
     3	Smooth, fully-differentiable helpers consumed by Zhang-McFarlane (PR 1),
     4	Kain-Fritsch (PR 2), Emanuel (PR 3), Tiedtke (PR 4), and Bechtold/IFS
     5	(PR 5).  The goal of this module is to provide a small set of
     6	well-tested column workhorses so that no convection scheme has to
     7	re-implement an LCL, LFC, plume integrator, or CMT closure.
     8	
     9	This module pairs with :mod:`._triggers` (sigmoid math) — together they
    10	are the only place these primitives live.
    11	
    12	What's in this PR (PR 0):
    13	
    14	* :func:`compute_lcl`           — Bolton (1980) lifting condensation level.
    15	* :func:`compute_lfc_lnb`       — smooth fractional levels of free
    16	                                  convection and neutral buoyancy.
    17	* :func:`compute_cin`           — CIN [J/kg] from the buoyancy profile.
    18	* :func:`entraining_detraining_plume` — vmappable updraft integrator
    19	                                  (``jax.lax.scan`` over levels).
    20	* :func:`cmt_gregory_1997`      — Gregory et al. 1997 convective
    21	                                  momentum transport closure.
    22	
    23	Deferred to later scheme PRs (each lands alongside its first consumer):
    24	
    25	* ``diagnose_grid_w_from_omega``     — Kain-Fritsch trigger (PR 2).
    26	* ``buoyancy_sort_emanuel``           — Emanuel mixing ensemble (PR 3).
    27	* ``downdraft_thermo``                — Tiedtke / Bechtold downdrafts
    28	                                         (PRs 4–5).
    29	
    30	This deferral was a pragmatic scope reduction within PR 0; each helper
    31	is small (~50–200 LOC) and is documented in
    32	``add-more-complex-convection-fluttering-quasar.md`` under its scheme.
    33	
    34	Conventions
    35	-----------
    36	* Column shape ``(ncol, nlev)``.  **Surface at the last index**
    37	  ``[:, -1]``; the model top is at ``[:, 0]``.  Pressure ``p_full``
    38	  *increases* with the array index (TOA ≪ surface).
    39	* Half levels ``p_half`` have shape ``(ncol, nlev+1)`` with ``p_half[:, 0]``
    40	  at TOA and ``p_half[:, -1]`` at the surface.
    41	* All temperatures in Kelvin, pressures in Pa, mass flux in kg/m²/s.
    42	* Reused upstream — never re-implement: ``legoesm.constants`` for
    43	  physical constants; ``legoesm.thermo.saturation_mixing_ratio`` and
    44	  ``saturation_vapor_pressure`` for thermodynamics; the ``moist_adiabat``
    45	  / ``moist_adiabat_lapse_rate`` / ``compute_cape`` family from
    46	  ``legoesm.atmosphere.physics.thermodynamics``; column geometry
    47	  helpers (``compute_heights_from_sigma``, ``compute_layer_dz``,
    48	  ``compute_rho``) from ``legoesm.atmosphere.physics._shared``.
    49	
    50	References
    51	----------
    52	* Bolton, D. (1980). The computation of equivalent potential
    53	  temperature.  *Mon. Wea. Rev.*, 108(7), 1046–1053.
    54	* Gregory, D., Kershaw, R., & Inness, P. M. (1997). Parametrization of
    55	  momentum transport by convection. II: Tests in single-column and
    56	  general circulation models.  *Quart. J. Roy. Meteor. Soc.*, 123,
    57	  1153–1183.
    58	* Zhang, G. J., & McFarlane, N. A. (1995). Sensitivity of climate
    59	  simulations to the parameterization of cumulus convection in the
    60	  Canadian Climate Centre general circulation model.  *Atmos.-Ocean*,
    61	  33(3), 407–446.
    62	"""
    63	
    64	from __future__ import annotations
    65	
    66	from typing import NamedTuple
    67	
    68	import jax
    69	import jax.numpy as jnp
    70	
    71	from legoesm import constants
    72	from legoesm.thermo import (
    73	    saturation_mixing_ratio,
    74	    saturation_vapor_pressure,
    75	)
    76	from legoesm.atmosphere.physics.thermodynamics import (
    77	    compute_cape,
    78	    compute_moist_adiabat,
    79	    moist_adiabat_lapse_rate,
    80	)
    81	
    82	from legoesm.atmosphere.physics.convection._triggers import (
    83	    smooth_level_indicator,
    84	    smooth_lowest_crossing_index,
    85	    smooth_step,
    86	)
    87	
    88	__all__ = (
    89	    "LCL",
    90	    "Plume",
    91	    "compute_lcl",
    92	    "compute_lfc_lnb",
    93	    "compute_cin",
    94	    "entraining_detraining_plume",
    95	    "cmt_gregory_1997",
    96	)
    97	
    98	
    99	# ---------------------------------------------------------------------------
   100	# Lifting condensation level (Bolton 1980)
   101	# ---------------------------------------------------------------------------
   102	
   103	class LCL(NamedTuple):
   104	    """LCL diagnostics for one column.
   105	
   106	    Fields
   107	    ------
   108	    p_lcl : jax.Array, shape (ncol,)
   109	        LCL pressure [Pa].
   110	    T_lcl : jax.Array, shape (ncol,)
   111	        LCL temperature [K].
   112	    k_lcl_smooth : jax.Array, shape (ncol,)
   113	        Smooth fractional level index (surface-last convention) at
   114	        which ``p_full`` crosses ``p_lcl``.  Used by downstream
   115	        helpers that need to localize the LCL on the model grid.
   116	    """
   117	    p_lcl: jax.Array
   118	    T_lcl: jax.Array
   119	    k_lcl_smooth: jax.Array
   120	
   121	
   122	def compute_lcl(
   123	    T_parcel: jax.Array,
   124	    q_parcel: jax.Array,
   125	    p_parcel: jax.Array,
   126	    p_full: jax.Array,
   127	    *,
   128	    crossing_sharpness: float = 0.001,
   129	) -> LCL:
   130	    """Lifting condensation level via Bolton (1980) Eq. 22.
   131	
   132	    Bolton's empirical formula::
   133	
   134	        T_LCL = 1 / [ 1/(T - 55) - ln(RH)/2840 ] + 55
   135	
   136	    where ``T`` is parcel temperature [K], ``RH`` is relative humidity
   137	    in ``(0, 1]``, and the result is the LCL temperature [K].  The
   138	    LCL pressure follows from Poisson's equation along a dry adiabat
   139	    from parcel level::
   140	
   141	        p_LCL = p * (T_LCL / T)^(c_pd / R_d)
   142	
   143	    Parameters
   144	    ----------
   145	    T_parcel : jax.Array, shape (ncol,)
   146	        Parcel temperature at the launch level [K].
   147	    q_parcel : jax.Array, shape (ncol,)
   148	        Parcel water-vapor specific humidity [kg/kg].
   149	    p_parcel : jax.Array, shape (ncol,)
   150	        Parcel launch pressure [Pa] (typically the lowest model
   151	        full-level pressure).
   152	    p_full : jax.Array, shape (ncol, nlev)
   153	        Full-level pressure [Pa] for the column, surface-last.  Used
   154	        only to compute ``k_lcl_smooth``.
   155	    crossing_sharpness : float
   156	        Sharpness for the soft fractional level diagnosis.  Units of
   157	        [1/Pa]; ``0.001`` sharpens to ~95%/5% transition over a
   158	        ~1000 Pa pressure range, which is finer than typical model
   159	        layer thickness in the boundary layer.
   160	
   161	    Returns
   162	    -------
   163	    LCL
   164	        ``p_lcl, T_lcl, k_lcl_smooth``.
   165	    """
   166	    # Vapor pressure and relative humidity at parcel level.
   167	    e_sat = saturation_vapor_pressure(T_parcel)
   168	    q_sat = saturation_mixing_ratio(T_parcel, p_parcel)
   169	    # RH = q / q_sat, clipped to (0, 1] so log is well-defined and
   170	    # supersaturated parcels produce LCL at parcel level.
   171	    RH = jnp.clip(q_parcel / jnp.maximum(q_sat, 1e-12), 1e-4, 1.0)
   172	
   173	    # Bolton (1980) Eq. 22.  ``T - 55`` floored to avoid singularity
   174	    # at very cold parcels (defensively — convective parcels are rarely
   175	    # below 200 K, but the formula is sensitive in pathological cases).
   176	    T_minus_55 = jnp.maximum(T_parcel - 55.0, 1.0)
   177	    T_lcl = 1.0 / (1.0 / T_minus_55 - jnp.log(RH) / 2840.0) + 55.0
   178	
   179	    # Poisson: dry-adiabatic descent from parcel to LCL.
   180	    p_lcl = p_parcel * (T_lcl / T_parcel) ** (constants.c_pd / constants.R_d)
   181	
   182	    # Soft fractional level index where p_full = p_lcl.  Pressure
   183	    # *decreases* with altitude (surface-first), so we apply
   184	    # smooth_lowest_crossing_index to negated pressure to convert the
   185	    # downward-with-altitude crossing into an upward one.
   186	    k_lcl_smooth = smooth_lowest_crossing_index(
   187	        -p_full, -p_lcl[:, None], crossing_sharpness,
   188	    )
   189	
   190	    return LCL(p_lcl=p_lcl, T_lcl=T_lcl, k_lcl_smooth=k_lcl_smooth)
   191	
   192	
   193	# ---------------------------------------------------------------------------
   194	# LFC and LNB
   195	# ---------------------------------------------------------------------------
   196	
   197	def compute_lfc_lnb(
   198	    T_env: jax.Array,
   199	    T_parcel_ma: jax.Array,
   200	    *,
   201	    sharpness: float = 1.0,
   202	) -> tuple[jax.Array, jax.Array]:
   203	    """Smooth fractional levels of free convection and neutral buoyancy.
   204	
   205	    The buoyancy proxy is ``T_parcel_ma - T_env`` (positive where the
   206	    parcel is warmer than the environment); the LFC is the lowest
   207	    level where this turns from negative to positive going upward,
   208	    and the LNB is the lowest level above the LFC where the proxy
   209	    turns back from positive to negative.
   210	
   211	    Both are returned as smooth fractional indices in surface-last
   212	    convention (``ncol, nlev`` indexing).  For columns with no clear
   213	    crossing the no-crossing fallback in
   214	    :func:`._triggers.smooth_lowest_crossing_index` returns a value
   215	    near the surface (LFC) or near the top (LNB).
   216	
   217	    Parameters
   218	    ----------
   219	    T_env : jax.Array, shape (ncol, nlev)
   220	        Environmental temperature [K].
   221	    T_parcel_ma : jax.Array, shape (ncol, nlev)
   222	        Moist-adiabatic parcel temperature [K] from
   223	        :func:`legoesm.atmosphere.physics.thermodynamics.compute_moist_adiabat`.
   224	    sharpness : float
   225	        Sigmoid sharpness in [1/K] on the buoyancy threshold.
   226	
   227	    Returns
   228	    -------
   229	    k_lfc_smooth, k_lnb_smooth : jax.Array, shape (ncol,)
   230	        Smooth fractional level indices.
   231	    """
   232	    buoyancy = T_parcel_ma - T_env
   233	    nlev = buoyancy.shape[-1]
   234	
   235	    # LFC: lowest UPWARD crossing of buoyancy = 0.
   236	    k_lfc = smooth_lowest_crossing_index(buoyancy, 0.0, sharpness)
   237	
   238	    # LNB: lowest UPWARD crossing of negative-buoyancy.  Equivalently,
   239	    # the lowest level (above the LFC) where the parcel turns from
   240	    # positive to negative buoyancy.  Implementation: apply the same
   241	    # primitive to ``-buoyancy`` AFTER the LFC is reached; the soft
   242	    # gating-by-LFC weight ensures we don't pick up sub-cloud layers
   243	    # where the parcel was negatively buoyant.
   244	    above_lfc_weight = smooth_level_indicator(
   245	        jnp.broadcast_to(jnp.arange(nlev, dtype=buoyancy.dtype), buoyancy.shape),
   246	        threshold=(nlev - 1.0) - k_lfc[:, None],  # surface-first index of LFC
   247	        sharpness=sharpness,
   248	        direction="below",  # surface-last: indices below LFC index are above LFC altitude
   249	    )
   250	    masked_buoyancy = buoyancy * above_lfc_weight
   251	    k_lnb = smooth_lowest_crossing_index(-masked_buoyancy, 0.0, sharpness)
   252	    return k_lfc, k_lnb
   253	
   254	
   255	# ---------------------------------------------------------------------------
   256	# CIN
   257	# ---------------------------------------------------------------------------
   258	
   259	def compute_cin(
   260	    T_env: jax.Array,
   261	    T_parcel_ma: jax.Array,
   262	    p_full: jax.Array,
   263	    p_half: jax.Array,
   264	    k_lcl_smooth: jax.Array,
   265	    k_lfc_smooth: jax.Array,
   266	    *,
   267	    indicator_sharpness: float = 1.0,
   268	) -> jax.Array:
   269	    """Convective Inhibition (CIN) [J/kg].
   270	
   271	    Integrates the negative buoyancy between the LCL and the LFC::
   272	
   273	        CIN = R_d * ∫_{LCL}^{LFC} max(0, T_env - T_parcel) * dp/p
   274	
   275	    The bounds of integration are encoded as a smooth window in
   276	    surface-last index space: ``window[k] = above_LCL(k) * below_LFC(k)``,
   277	    each factor a sigmoid on the level index relative to the
   278	    fractional indices.  Smooth bounds preserve gradients.
   279	
   280	    Convention check: for a positively-CAPE column where the parcel
   281	    is warmer than the environment between the LCL and the LFC,
   282	    ``T_env - T_parcel`` is negative and the ``max(., 0)`` clamp gives
   283	    zero.  CIN therefore counts only the genuinely-inhibiting layers.
   284	
   285	    Parameters
   286	    ----------
   287	    T_env, T_parcel_ma : jax.Array, shape (ncol, nlev)
   288	        Environmental and moist-adiabatic parcel temperatures [K].
   289	    p_full, p_half : jax.Array
   290	        Full-level (``ncol, nlev``) and half-level (``ncol, nlev+1``)
   291	        pressures [Pa].
   292	    k_lcl_smooth, k_lfc_smooth : jax.Array, shape (ncol,)
   293	        Smooth fractional level indices from :func:`compute_lcl` and
   294	        :func:`compute_lfc_lnb` (surface-last convention).
   295	    indicator_sharpness : float
   296	        Sigmoid sharpness on the level-window bounds, in [1/level].
   297	
   298	    Returns
   299	    -------
   300	    jax.Array, shape (ncol,)
   301	        CIN [J/kg].  Non-negative.
   302	    """
   303	    nlev = T_env.shape[-1]
   304	    levels = jnp.arange(nlev, dtype=T_env.dtype)
   305	    levels = jnp.broadcast_to(levels, T_env.shape)
   306	
   307	    # The integration window in surface-last index space is
   308	    # ``LFC_idx <= k <= LCL_idx`` because LFC is *above* LCL and
   309	    # surface-last indices DECREASE with altitude.  The smooth window
   310	    # is the product of two sigmoids.
   311	    window_above_lfc = jax.nn.sigmoid(
   312	        indicator_sharpness * (k_lfc_smooth[:, None] + 0.5 - levels)
   313	    )  # 1 at indices above LFC altitude (smaller index), 0 below.
   314	    window_below_lcl = jax.nn.sigmoid(
   315	        indicator_sharpness * (levels - (k_lcl_smooth[:, None] - 0.5))
   316	    )  # 1 at indices at LCL or below (larger index, lower altitude), 0 above.
   317	    window = window_above_lfc * window_below_lcl
   318	
   319	    dp = p_half[:, 1:] - p_half[:, :-1]
   320	    inhibiting_buoyancy = jnp.maximum(0.0, T_env - T_parcel_ma)
   321	
   322	    return constants.R_d * jnp.sum(window * inhibiting_buoyancy * dp / p_full, axis=-1)
   323	
   324	
   325	# ---------------------------------------------------------------------------
   326	# Entraining-detraining updraft plume
   327	# ---------------------------------------------------------------------------
   328	
   329	class Plume(NamedTuple):
   330	    """Output of :func:`entraining_detraining_plume`.
   331	
   332	    All arrays are surface-last with shape ``(ncol, nlev)``.
   333	
   334	    Fields
   335	    ------
   336	    M_u : jax.Array
   337	        Updraft mass flux [kg/m²/s].  Non-negative; zero below the
   338	        cloud base and at/above the level of neutral buoyancy.
   339	    T_u : jax.Array
   340	        Updraft temperature [K].
   341	    q_u : jax.Array
   342	        Updraft water-vapor specific humidity [kg/kg].
   343	    q_c_u : jax.Array
   344	        Updraft cloud-water mixing ratio [kg/kg].  Non-negative.
   345	    B_u : jax.Array
   346	        Updraft buoyancy ``T_u - T_env`` [K].  Negative aloft is the
   347	        signal for the plume to terminate.
   348	    """
   349	    M_u: jax.Array
   350	    T_u: jax.Array
   351	    q_u: jax.Array
   352	    q_c_u: jax.Array
   353	    B_u: jax.Array
   354	
   355	
   356	def entraining_detraining_plume(
   357	    T_env: jax.Array,
   358	    q_v_env: jax.Array,
   359	    p_full: jax.Array,
   360	    p_half: jax.Array,
   361	    z_full: jax.Array,
   362	    T_parcel_base: jax.Array,
   363	    q_parcel_base: jax.Array,
   364	    k_base_smooth: jax.Array,
   365	    epsilon_profile: jax.Array,
   366	    delta_profile: jax.Array,
   367	    M_b: jax.Array,
   368	    *,
   369	    buoyancy_sharpness: float = 0.5,
   370	) -> Plume:
   371	    """Bulk entraining-detraining updraft from cloud base to LNB.
   372	
   373	    Integrates the standard plume budget upward from the cloud base
   374	    using :func:`jax.lax.scan` over levels (surface-first ordering
   375	    inside the scan).  At each layer the plume entrains environmental
   376	    air at fractional rate ``epsilon`` and detrains plume air at
   377	    rate ``delta`` (both with units [1/m]):
   378	
   379	    .. math::
   380	
   381	        \\frac{1}{M_u} \\frac{\\partial M_u}{\\partial z} = \\epsilon - \\delta
   382	
   383	        \\frac{\\partial T_u}{\\partial z} = -\\frac{g}{c_{pd}} - \\epsilon (T_u - T_{env})
   384	
   385	        \\frac{\\partial q_u}{\\partial z} = -\\epsilon (q_u - q_{v,env}) - C
   386	
   387	    where ``C`` is the condensation rate (the surplus of the parcel's
   388	    water vapor over its saturation value at the current level).
   389	    Cloud water accumulates at rate ``C - precip_rate`` (precipitation
   390	    is delegated to the calling scheme).
   391	
   392	    The plume's effective extent is encoded smoothly: the mass-flux
   393	    profile is multiplied by a sigmoid on the buoyancy ``T_u - T_env``
   394	    so that the plume tapers off (rather than being abruptly
   395	    truncated) once the parcel becomes negatively buoyant aloft.
   396	
   397	    Parameters
   398	    ----------
   399	    T_env, q_v_env : jax.Array, shape (ncol, nlev)
   400	        Environmental temperature [K] and vapor specific humidity
   401	        [kg/kg].
   402	    p_full, p_half : jax.Array
   403	        Pressures [Pa] at full and half levels.
   404	    z_full : jax.Array, shape (ncol, nlev)
   405	        Geopotential heights [m] at full levels.
   406	    T_parcel_base, q_parcel_base : jax.Array, shape (ncol,)
   407	        Parcel temperature [K] and specific humidity [kg/kg] at the
   408	        cloud-base launch level.
   409	    k_base_smooth : jax.Array, shape (ncol,)
   410	        Smooth fractional level index of the cloud base
   411	        (surface-last).  Levels below the cloud base contribute
   412	        zero plume mass flux.
   413	    epsilon_profile, delta_profile : jax.Array, shape (ncol, nlev)
   414	        Per-level fractional entrainment / detrainment rates [1/m].
   415	    M_b : jax.Array, shape (ncol,)
   416	        Cloud-base mass flux [kg/m²/s].
   417	    buoyancy_sharpness : float
   418	        Sharpness on the buoyancy-based plume-tapering sigmoid in
   419	        [1/K].  Default ``0.5`` per Kelvin of buoyancy means the
   420	        plume is at half mass flux when ``T_u - T_env`` reaches the
   421	        modest negative value of about ``-1.4 K``.
   422	
   423	    Returns
   424	    -------
   425	    Plume
   426	        Per-level mass flux, plume temperature, plume vapor, plume
   427	        cloud water, and plume buoyancy.
   428	    """
   429	    ncol, nlev = T_env.shape
   430	
   431	    # Pin everything to the input precision so saturation_mixing_ratio
   432	    # (which promotes f32 → f64 via its Clausius-Clapeyron literal
   433	    # constants) doesn't break ``jax.lax.scan``'s "carry-in dtype must
   434	    # equal carry-out dtype" invariant.
   435	    _dtype = T_env.dtype
   436	
   437	    # Reverse to surface-first for the scan.
   438	    T_env_rev = T_env[:, ::-1].astype(_dtype)
   439	    q_v_env_rev = q_v_env[:, ::-1].astype(_dtype)
   440	    p_full_rev = p_full[:, ::-1].astype(_dtype)
   441	    z_full_rev = z_full[:, ::-1].astype(_dtype)
   442	    eps_rev = epsilon_profile[:, ::-1].astype(_dtype)
   443	    del_rev = delta_profile[:, ::-1].astype(_dtype)
   444	
   445	    # Per-level above-base weight.  In surface-last indexing the cloud
   446	    # base is at ``k_base_smooth``; levels with surface-last index
   447	    # smaller than ``k_base_smooth`` are above the base.  In
   448	    # surface-first reversed indexing the relationship inverts:
   449	    # surface-first index ``k_rev`` corresponds to surface-last index
   450	    # ``nlev - 1 - k_rev``, and "above cloud base" means
   451	    # ``k_rev > (nlev - 1 - k_base_smooth)``.
   452	    k_rev = jnp.arange(nlev, dtype=T_env.dtype)
   453	    k_rev = jnp.broadcast_to(k_rev, T_env.shape)
   454	    k_base_rev = (nlev - 1.0) - k_base_smooth
   455	    above_base_weight = jax.nn.sigmoid(buoyancy_sharpness * (k_rev - k_base_rev[:, None]))
   456	
   457	    # Initial plume state at the surface-first index 0 (which is the
   458	    # actual surface).  We launch with the parcel values; the
   459	    # ``above_base_weight`` mask will suppress mass flux below cloud
   460	    # base.  All carry components are pinned to ``_dtype``.
   461	    init_carry = (
   462	        T_parcel_base.astype(_dtype),                       # T_u_prev
   463	        q_parcel_base.astype(_dtype),                       # q_u_prev
   464	        jnp.zeros_like(T_parcel_base, dtype=_dtype),        # q_c_u_prev
   465	        M_b.astype(_dtype),                                 # M_u_prev
   466	        z_full_rev[:, 0],                                   # z_prev (already cast)
   467	    )
   468	
   469	    # Per-level inputs to the scan.  Transpose to (nlev, ncol).
   470	    inputs = (
   471	        jnp.moveaxis(T_env_rev, 1, 0),
   472	        jnp.moveaxis(q_v_env_rev, 1, 0),
   473	        jnp.moveaxis(p_full_rev, 1, 0),
   474	        jnp.moveaxis(z_full_rev, 1, 0),
   475	        jnp.moveaxis(eps_rev, 1, 0),
   476	        jnp.moveaxis(del_rev, 1, 0),
   477	        jnp.moveaxis(above_base_weight, 1, 0),
   478	    )
   479	
   480	    g = constants.g
   481	    c_pd = constants.c_pd
   482	    L_v = constants.L_v
   483	
   484	    def step(carry, layer_inputs):
   485	        T_u_prev, q_u_prev, q_c_u_prev, M_u_raw_prev, z_prev = carry
   486	        T_e, q_e, p_e, z_e, eps, dlt, abv = layer_inputs
   487	
   488	        dz = jnp.maximum(z_e - z_prev, 1.0)  # ascending; floor to avoid div-by-zero
   489	
   490	        # Raw plume mass flux: dM/dz = (epsilon - delta) * M.  We
   491	        # intentionally do NOT bake the buoyancy / sub-cloud masks
   492	        # into the carry — those are reporting filters, not dynamics.
   493	        # Folding them into the carry would compound across levels and
   494	        # destroy the cloud-base-to-LNB profile that consumers expect.
   495	        M_u_raw = M_u_raw_prev * (1.0 + (eps - dlt) * dz)
   496	        M_u_raw = jnp.maximum(M_u_raw, 0.0)
   497	
   498	        # Entrainment of environmental T, q.
   499	        T_u_ent = T_u_prev + eps * dz * (T_e - T_u_prev)
   500	        q_u_ent = q_u_prev + eps * dz * (q_e - q_u_prev)
   501	
   502	        # Use the analytic moist-adiabatic lapse rate from
   503	        # ``moist_adiabat_lapse_rate`` (Iribarne–Godson) at the
   504	        # entrained parcel state.  The function evaluates dT/dp on
   505	        # the moist adiabat assuming the parcel is saturated; for
   506	        # *unsaturated* parcels this is approximate but matches the
   507	        # behavior of the existing ``compute_moist_adiabat`` helper
   508	        # that we're benchmarked against.  Latent heating is already
   509	        # baked into the lapse rate, so the post-hoc condensation
   510	        # step below does NOT add an additional ``L_v/c_pd *
   511	        # condensate`` correction — that would double-count.
   512	        rho_u_ent = p_e / (constants.R_d * jnp.maximum(T_u_ent, 100.0))
   513	        # ``moist_adiabat_lapse_rate`` returns dT/dp [K/Pa]; convert
   514	        # to dT/dz [K/m] via dp/dz = -rho*g.
   515	        Gamma_moist_per_pa = moist_adiabat_lapse_rate(T_u_ent, p_e)
   516	        dT_dz = Gamma_moist_per_pa * (-rho_u_ent * g)
   517	
   518	        T_u = T_u_ent + dT_dz * dz
   519	
   520	        # Condense any super-saturation into cloud water.  This is the
   521	        # diagnostic that resolves the q-budget; the temperature
   522	        # already incorporates the latent heat from condensation via
   523	        # the moist lapse rate.
   524	        q_sat_new = saturation_mixing_ratio(T_u, p_e).astype(_dtype)
   525	        condensate = jnp.maximum(q_u_ent - q_sat_new, 0.0).astype(_dtype)
   526	        q_u = (q_u_ent - condensate).astype(_dtype)
   527	        q_c_u = (q_c_u_prev + condensate).astype(_dtype)
   528	
   529	        T_u = T_u.astype(_dtype)
   530	
   531	        # Buoyancy at this level.
   532	        B_u = T_u - T_e
   533	
   534	        # Reporting filters: smoothly suppress the mass flux below
   535	        # cloud base (``abv``) and where the plume has lost buoyancy
   536	        # (``plume_alive``).  These do NOT enter the carry.
   537	        plume_alive = jax.nn.sigmoid(buoyancy_sharpness * B_u)
   538	        M_u_reported = M_u_raw * plume_alive * abv
   539	
   540	        new_carry = (T_u, q_u, q_c_u, M_u_raw, z_e)
   541	        outputs = (T_u, q_u, q_c_u, M_u_reported, B_u)
   542	        return new_carry, outputs
   543	
   544	    _, scan_out = jax.lax.scan(step, init_carry, inputs)
   545	
   546	    T_u_rev, q_u_rev, q_c_u_rev, M_u_rev, B_u_rev = scan_out  # (nlev, ncol)
   547	
   548	    # Move axis back and reverse to surface-last.
   549	    T_u = jnp.moveaxis(T_u_rev, 0, 1)[:, ::-1]
   550	    q_u = jnp.moveaxis(q_u_rev, 0, 1)[:, ::-1]
   551	    q_c_u = jnp.moveaxis(q_c_u_rev, 0, 1)[:, ::-1]
   552	    M_u = jnp.moveaxis(M_u_rev, 0, 1)[:, ::-1]
   553	    B_u = jnp.moveaxis(B_u_rev, 0, 1)[:, ::-1]
   554	
   555	    return Plume(M_u=M_u, T_u=T_u, q_u=q_u, q_c_u=q_c_u, B_u=B_u)
   556	
   557	
   558	# ---------------------------------------------------------------------------
   559	# Convective momentum transport (Gregory et al. 1997)
   560	# ---------------------------------------------------------------------------
   561	
   562	def cmt_gregory_1997(
   563	    u_env: jax.Array,
   564	    v_env: jax.Array,
   565	    M_u: jax.Array,
   566	    M_d: jax.Array | None,
   567	    p_full: jax.Array,
   568	    p_half: jax.Array,
   569	    rho: jax.Array,
   570	    *,
   571	    c_u: float = 0.55,
   572	    c_d: float = 0.55,
   573	) -> tuple[jax.Array, jax.Array]:
   574	    """Gregory et al. 1997 convective momentum transport closure.
   575	
   576	    The eddy-flux closure carries plume momentum aloft minus a
   577	    pressure-gradient correction that limits the upward transport in
   578	    sheared environments.  In bulk-plume form ::
   579	
   580	        F_u = M_u * (u_u - u_env) - c_u * M_u * (du/dz)_layer * dz_layer
   581	
   582	    and the resulting tendency is ``du/dt = -(1/rho) * dF/dz``.  The
   583	    same form applies to the downdraft with sign convention
   584	    ``M_d < 0`` and parameter ``c_d``.
   585	
   586	    The implementation here is the **simplified bulk closure**: we
   587	    approximate ``u_u`` by the environmental wind at the cloud base
   588	    plus a fraction of the layer-by-layer environmental shear,
   589	    yielding a numerically stable form that does not require a
   590	    separate plume-momentum integrator.  This is the formulation
   591	    used in CESM/CAM with the ZM scheme and follows Gregory et al.
   592	    1997 Eq. 14.
   593	
   594	    Parameters
   595	    ----------
   596	    u_env, v_env : jax.Array, shape (ncol, nlev)
   597	        Environmental zonal / meridional wind [m/s].
   598	    M_u : jax.Array, shape (ncol, nlev)
   599	        Updraft mass flux [kg/m²/s].  Non-negative.
   600	    M_d : jax.Array or None, shape (ncol, nlev)
   601	        Downdraft mass flux [kg/m²/s] (negative by convention).
   602	        ``None`` skips the downdraft contribution.
   603	    p_full, p_half : jax.Array
   604	        Pressures [Pa] at full / half levels.  Used to compute layer
   605	        thickness ``dp = p_half[1:] - p_half[:-1]``.
   606	    rho : jax.Array, shape (ncol, nlev)
   607	        Air density [kg/m³].
   608	    c_u, c_d : float
   609	        Pressure-gradient correction coefficients in Gregory et al.
   610	        1997 Eq. 12.  ``0.55`` is the canonical value (range
   611	        ``0.3–0.7`` in the literature).
   612	
   613	    Returns
   614	    -------
   615	    du_dt, dv_dt : jax.Array, shape (ncol, nlev)
   616	        Convective momentum tendencies [m/s²].
   617	    """
   618	    nlev = u_env.shape[-1]
   619	
   620	    # Layer pressure thickness; with surface-last convention dp > 0.
   621	    dp = p_half[:, 1:] - p_half[:, :-1]
   622	
   623	    # Stratospheric mass-flux gate — same factor the kernel applies to
   624	    # T/q_v tendencies.  Without this, the CMT path detrains
   625	    # convective momentum into the model top (where the plume should
   626	    # already be dead), producing wind-driven dycore blowups (e.g.
   627	    # KF at day 10 in 1-year lat-lon FV RCE).  Imported lazily to
   628	    # avoid a circular import (`mass_flux` imports from `_plume`).
   629	    from legoesm.atmosphere.physics.convection.mass_flux import (
   630	        stratosphere_mass_flux_gate,
   631	    )
   632	    p_gate = stratosphere_mass_flux_gate(p_full)
   633	    M_u = M_u * p_gate
   634	    if M_d is not None:
   635	        M_d = M_d * p_gate
   636	
   637	    # Environmental shear (forward difference per layer).  Edge layers
   638	    # use one-sided differences via padded edges to keep shape
   639	    # ``(ncol, nlev)``.
   640	    du_layer = jnp.diff(u_env, axis=-1, prepend=u_env[:, :1])
   641	    dv_layer = jnp.diff(v_env, axis=-1, prepend=v_env[:, :1])
   642	
   643	    # Eddy momentum flux from Gregory et al. 1997 §3.  In the bulk
   644	    # plume approximation:
   645	    #     M_u (u_u - u_env) ≈ -c_u * M_u * du/dz_layer
   646	    # i.e. only the pressure-gradient correction term contributes —
   647	    # the leading ``M_u * u_env`` mass-transport term cancels when
   648	    # the mass-flux divergence is also accounted for in the
   649	    # large-scale momentum equation.  As a consequence a uniform
   650	    # wind column produces zero CMT regardless of mass flux.
   651	    flux_u_up = -c_u * M_u * du_layer
   652	    flux_v_up = -c_u * M_u * dv_layer
   653	
   654	    if M_d is not None:
   655	        flux_u_down = -c_d * M_d * du_layer
   656	        flux_v_down = -c_d * M_d * dv_layer
   657	    else:
   658	        flux_u_down = jnp.zeros_like(flux_u_up)
   659	        flux_v_down = jnp.zeros_like(flux_v_up)
   660	
   661	    flux_u = flux_u_up + flux_u_down
   662	    flux_v = flux_v_up + flux_v_down
   663	
   664	    # Vertical divergence of the flux: ``du/dt = -(1/rho) * dF/dz``.
   665	    # Using hydrostatic ``dz = -dp/(rho*g)`` gives
   666	    # ``du/dt = -(g) * dF/dp`` after the rho cancels.
   667	    dflux_u = jnp.diff(flux_u, axis=-1, append=flux_u[:, -1:])
   668	    dflux_v = jnp.diff(flux_v, axis=-1, append=flux_v[:, -1:])
   669	
   670	    du_dt = -constants.g * dflux_u / dp
   671	    dv_dt = -constants.g * dflux_v / dp
   672	
   673	    return du_dt, dv_dt

codex
I’ve got the shared trigger, plume, and mass-flux kernel in view. A few places already deserve scrutiny: soft crossing fallbacks, plume condensation/thermal budgets, and the kernel’s flux-form consistency; I’m reading the scheme wrappers next before deciding which are true defects.
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/sbm.py | sed -n '1,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/kuo.py | sed -n '1,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/dca.py | sed -n '1,320p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Simplified Betts-Miller (SBM) convection scheme.
     2	
     3	A relaxation-based convection parameterization for idealized aquaplanet
     4	experiments. Convective columns are relaxed toward a moist adiabatic
     5	temperature profile with an enthalpy-conserving correction.
     6	
     7	Algorithm:
     8	1. Compute moist adiabat from surface temperature upward
     9	2. Construct reference moisture profile: q_ref = RH_ref * q_sat(T_moist, p)
    10	3. Apply enthalpy-conserving correction (energy budget closure)
    11	4. Compute CAPE and smooth trigger
    12	5. Relax T and q_v toward reference profiles over timescale tau_c
    13	6. Diagnose precipitation from column moisture convergence
    14	
    15	All operations use smooth (differentiable) approximations for
    16	compatibility with jax.grad.
    17	
    18	References
    19	----------
    20	- Frierson, D. M. W. (2007). The Dynamics of Idealized Convection
    21	  Schemes and Their Effect on the Zonally Averaged Tropical Circulation.
    22	  J. Atmos. Sci., 64, 1959-1976.
    23	- Betts, A. K., & Miller, M. J. (1986). A new convective adjustment
    24	  scheme. Part II: Single column tests using GATE wave, BOMEX, ATEX
    25	  and arctic air-mass data sets. Q. J. R. Meteorol. Soc., 112, 693-709.
    26	"""
    27	
    28	from __future__ import annotations
    29	
    30	import jax
    31	import jax.numpy as jnp
    32	
    33	from legoesm import constants
    34	from legoesm.thermo import saturation_mixing_ratio
    35	from legoesm.atmosphere.physics.thermodynamics import (
    36	    compute_moist_adiabat,
    37	    compute_cape,
    38	)
    39	from legoesm.atmosphere.physics.convection.config import SBMConfig
    40	from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    41	
    42	
    43	def sbm_convection(
    44	    T: jax.Array,
    45	    q_v: jax.Array,
    46	    p_full: jax.Array,
    47	    p_half: jax.Array,
    48	    dt: float,
    49	    config: SBMConfig = SBMConfig(),
    50	) -> ConvectionOutput:
    51	    """Compute Simplified Betts-Miller convection tendencies.
    52	
    53	    Parameters
    54	    ----------
    55	    T : jax.Array
    56	        Temperature at full levels [K], shape (ncol, nlev).
    57	    q_v : jax.Array
    58	        Water vapor specific humidity [kg/kg], shape (ncol, nlev).
    59	    p_full : jax.Array
    60	        Pressure at full levels [Pa], shape (ncol, nlev).
    61	    p_half : jax.Array
    62	        Pressure at half levels [Pa], shape (ncol, nlev+1).
    63	    dt : float
    64	        Model time step [s].
    65	    config : SBMConfig
    66	        Convection configuration.
    67	
    68	    Returns
    69	    -------
    70	    ConvectionOutput
    71	        Convective tendencies and diagnostics.
    72	    """
    73	    ncol, nlev = T.shape
    74	    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev) layer thickness
    75	    # ``jnp.full`` lowers to a single ``Broadcast`` HLO op; the previous
    76	    # ``broadcast_to(jnp.asarray(scalar, dtype), shape)`` form additionally
    77	    # forced a ``ConvertElementType`` for the implicit promotion of the
    78	    # Python float, which is unnecessary work per convection step.
    79	    tau_c = jnp.full((ncol,), config.tau_c, dtype=T.dtype)
    80	    RH_ref = jnp.full((ncol,), config.RH_ref, dtype=T.dtype)
    81	    CAPE_threshold = jnp.full((ncol,), config.CAPE_threshold, dtype=T.dtype)
    82	
    83	    # 1. Surface temperature as parcel starting point
    84	    T_base = T[:, -1]  # (ncol,)
    85	
    86	    # 2. Compute moist adiabatic temperature profile
    87	    T_moist = compute_moist_adiabat(T_base, p_full)  # (ncol, nlev)
    88	
    89	    # 3. Identify the convective layer: only levels where the moist adiabat
    90	    #    is warmer than the environment (conditional instability).
    91	    #    This prevents adjusting the stable stratosphere (Frierson 2007).
    92	    cloud_mask = (T_moist >= T).astype(T.dtype)  # (ncol, nlev)
    93	
    94	    # 4. Compute CAPE from the RAW moist adiabat (before enthalpy correction)
    95	    #    to avoid artificial CAPE from the Newton correction.
    96	    cape = compute_cape(T, T_moist, p_full, p_half)  # (ncol,)
    97	
    98	    # 5. Enthalpy-conserving correction (Newton iteration)
    99	    #    Only over the cloud layer (masked levels).
   100	    def _newton_step(T_trial):
   101	        q_trial = RH_ref[:, None] * saturation_mixing_ratio(T_trial, p_full)
   102	        residual = jnp.sum(
   103	            cloud_mask * (constants.c_pd * (T_trial - T)
   104	                          + constants.L_v * (q_trial - q_v)) * dp,
   105	            axis=1,
   106	        )  # (ncol,)
   107	        q_sat_trial = saturation_mixing_ratio(T_trial, p_full)
   108	        dqsat_dT = constants.L_v * q_sat_trial / (constants.R_v * T_trial ** 2)
   109	        jacobian = jnp.sum(
   110	            cloud_mask * (constants.c_pd
   111	                          + constants.L_v * RH_ref[:, None] * dqsat_dT) * dp,
   112	            axis=1,
   113	        )  # (ncol,)
   114	        dT = -residual / jnp.clip(jacobian, 1.0, None)
   115	        return T_trial + dT[:, None]
   116	
   117	    T_ref = _newton_step(T_moist)  # first iteration
   118	    T_ref = _newton_step(T_ref)    # second iteration
   119	
   120	    # Reference moisture at converged temperature
   121	    q_ref = RH_ref[:, None] * saturation_mixing_ratio(T_ref, p_full)
   122	
   123	    # 6. Smooth trigger: sigmoid(sharpness * (CAPE - threshold))
   124	    trigger = jax.nn.sigmoid(
   125	        config.smooth_trigger_sharpness * (cape - CAPE_threshold)
   126	    )  # (ncol,)
   127	
   128	    # 7. Relaxation tendencies — only within the convective (cloud) layer
   129	    dT_dt = trigger[:, None] * cloud_mask * (T_ref - T) / tau_c[:, None]
   130	    dq_v_dt = trigger[:, None] * cloud_mask * (q_ref - q_v) / tau_c[:, None]
   131	
   132	    # 7. Convective source for cloud water: vapor that condenses at each
   133	    # level becomes cloud water rather than precipitating instantly.
   134	    # Microphysics processes this through autoconversion, sedimentation,
   135	    # and evaporation, and produces the surface precipitation diagnostic.
   136	    #
   137	    # Naive ``max(-dq_v_dt, 0)`` per level would *create* water
   138	    # column-wide whenever the relaxation has both drying and
   139	    # moistening layers (column-integrated dq_v + column-integrated
   140	    # max(-dq_v, 0) = moistening_part > 0). To preserve column water
   141	    # conservation we rescale the per-level condensation candidate so
   142	    # its column integral equals the column-net drying — this matches
   143	    # the legacy ``precipitation`` formula exactly. Per-level the
   144	    # field is still non-negative (no negative q_c production); when
   145	    # the column is net moistening (col_dq_v > 0) the scale is 0 and
   146	    # dq_c_conv_dt = 0 everywhere, mirroring the legacy
   147	    # ``clip(-col_dq_v, 0)`` behavior.
   148	    local_cond = jnp.maximum(-dq_v_dt, 0.0)
   149	    col_local_cond = jnp.sum(local_cond * dp / constants.g, axis=-1, keepdims=True)
   150	    col_net_drying = jnp.clip(
   151	        -jnp.sum(dq_v_dt * dp / constants.g, axis=-1, keepdims=True),
   152	        0.0, None,
   153	    )
   154	    dq_c_conv_dt = local_cond * (
   155	        col_net_drying / jnp.clip(col_local_cond, 1e-30, None)
   156	    )  # (ncol, nlev) [kg/kg/s]
   157	
   158	    return ConvectionOutput(
   159	        dT_dt=dT_dt,
   160	        dq_v_dt=dq_v_dt,
   161	        dq_c_conv_dt=dq_c_conv_dt,
   162	        cape=cape,
   163	        convective_mask=trigger,
   164	    )

 succeeded in 0ms:
     1	"""Kuo column moisture-excess convection scheme.
     2	
     3	A column moisture-excess convection scheme (Kuo 1965/1974). Convective
     4	heating and moistening are proportional to the column-integrated moisture
     5	excess above saturation, partitioned by alpha_heat.
     6	
     7	Algorithm (Kuo 1965/1974 — column moisture-excess formulation):
     8	1. Compute column moisture excess above saturation [kg/m^2]
     9	2. Smooth sigmoid trigger based on moisture excess
    10	3. Compute moist adiabat reference profile
    11	4. Relax temperature and moisture toward reference profiles
    12	5. Emit per-level cloud-water source (``dq_c_conv_dt``) from the
    13	   implied condensation rate; microphysics processes it through
    14	   autoconversion / sedimentation / evaporation and produces the
    15	   surface precipitation diagnostic.
    16	
    17	All operations use smooth (differentiable) approximations for
    18	compatibility with jax.grad.
    19	
    20	Water budget
    21	------------
    22	Kuo is a *non-conservative* scheme by design: a fraction
    23	``(1 - alpha_heat)`` of the column moisture excess appears as
    24	moistening that has no in-scheme sink — physically interpreted as
    25	surface evaporation or large-scale moisture convergence implicit in
    26	the parameterization. The column water residual is therefore
    27	``(1 - alpha_heat) * MC / tau_relax`` per timestep (≈0.25 ⋅ MC/τ for
    28	the default ``alpha_heat = 0.75``). This was already true under the
    29	legacy ``ConvectionOutput.precipitation`` formulation; Option C
    30	preserves it and exposes the full ``alpha_heat * MC / tau_relax``
    31	condensation rate to microphysics rather than the column-net drying.
    32	
    33	References
    34	----------
    35	- Kuo, H. L. (1965). On formation and intensification of tropical
    36	  cyclones through latent heat release by cumulus convection.
    37	  J. Atmos. Sci., 22, 40-63.
    38	- Kuo, H. L. (1974). Further studies of the parameterization of the
    39	  influence of cumulus convection on large-scale flow.
    40	  J. Atmos. Sci., 31, 1232-1240.
    41	"""
    42	
    43	from __future__ import annotations
    44	
    45	import jax
    46	import jax.numpy as jnp
    47	
    48	from legoesm import constants
    49	from legoesm.thermo import saturation_mixing_ratio
    50	from legoesm.atmosphere.physics.thermodynamics import (
    51	    compute_moist_adiabat,
    52	    compute_cape,
    53	)
    54	from legoesm.atmosphere.physics.convection.config import KuoConfig
    55	from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    56	
    57	
    58	def kuo_convection(
    59	    T: jax.Array,
    60	    q_v: jax.Array,
    61	    p_full: jax.Array,
    62	    p_half: jax.Array,
    63	    dt: float,
    64	    config: KuoConfig = KuoConfig(),
    65	) -> ConvectionOutput:
    66	    """Compute Kuo column moisture-excess convection tendencies.
    67	
    68	    Parameters
    69	    ----------
    70	    T : jax.Array
    71	        Temperature at full levels [K], shape (ncol, nlev).
    72	    q_v : jax.Array
    73	        Water vapor specific humidity [kg/kg], shape (ncol, nlev).
    74	    p_full : jax.Array
    75	        Pressure at full levels [Pa], shape (ncol, nlev).
    76	    p_half : jax.Array
    77	        Pressure at half levels [Pa], shape (ncol, nlev+1).
    78	    dt : float
    79	        Model time step [s].
    80	    config : KuoConfig
    81	        Convection configuration.
    82	
    83	    Returns
    84	    -------
    85	    ConvectionOutput
    86	        Convective tendencies and diagnostics.
    87	    """
    88	    ncol, nlev = T.shape
    89	    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)
    90	
    91	    # 1. Saturation mixing ratio
    92	    q_sat = saturation_mixing_ratio(T, p_full)  # (ncol, nlev)
    93	
    94	    # 2. Column moisture excess: positive part only (zero when subsaturated)
    95	    excess = jnp.maximum(q_v - q_sat, 0.0)  # (ncol, nlev)
    96	    MC = jnp.sum(excess * dp, axis=1) / constants.g  # (ncol,) [kg/m^2]
    97	
    98	    # 3. Smooth trigger based on column moisture excess
    99	    trigger = jax.nn.sigmoid(
   100	        config.smooth_trigger_sharpness * (MC - config.me_threshold)
   101	    )  # (ncol,)
   102	
   103	    # 4. Moist adiabatic reference profile from surface temperature
   104	    T_base = T[:, -1]  # (ncol,)
   105	    T_moist = compute_moist_adiabat(T_base, p_full)  # (ncol, nlev)
   106	
   107	    # 5. Heating tendency: relax toward moist adiabat, gated by MC.
   108	    #
   109	    # The smooth sigmoid trigger alone is not enough: at MC = 0 the
   110	    # default ``(smooth_trigger_sharpness, me_threshold)`` give
   111	    # ``trigger ≈ 0.475`` rather than zero, so a relaxation
   112	    # ``(T_moist - T) / tau_relax`` not gated by MC would still
   113	    # heat (and condense, and remove vapor) in undersaturated
   114	    # columns — destroying water with no source.
   115	    #
   116	    # Kuo (1965/1974) actually prescribes column heating proportional
   117	    # to ``alpha_heat * MC / tau_relax``; the legacy implementation
   118	    # drifted from that design by using a pure relaxation rate. The
   119	    # ``tanh(MC / me_threshold)`` factor restores the MC-proportional
   120	    # scaling smoothly: zero at MC = 0, ≈1 once ``MC >> me_threshold``,
   121	    # and differentiable everywhere. Combined with the existing
   122	    # sigmoid trigger this guarantees ``dT_dt = 0``,
   123	    # ``implied_condensation = 0``, ``dq_v_dt = 0``, and
   124	    # ``dq_c_conv_dt = 0`` whenever MC = 0 — no spurious heating,
   125	    # no destroyed vapor, no created cloud water.
   126	    mc_gate = jnp.tanh(MC / jnp.clip(config.me_threshold, 1e-30, None))
   127	    dT_dt = (
   128	        trigger[:, None] * mc_gate[:, None]
   129	        * config.alpha_heat
   130	        * (T_moist - T)
   131	        / config.tau_relax
   132	    )  # (ncol, nlev)
   133	
   134	    # 6. Moistening tendency (budget-consistent with heating)
   135	    #
   136	    # The column moisture excess MC is the scheme's internal source.
   137	    # Fraction ``alpha_heat`` becomes condensational heating — this is
   138	    # the cloud-water source rate handed to microphysics through
   139	    # ``dq_c_conv_dt`` (see step 7). Fraction ``(1 - alpha_heat)``
   140	    # appears as a column-distributed vapor source representing
   141	    # external moistening implicit in Kuo's design (surface evaporation
   142	    # / large-scale moisture convergence).
   143	    #
   144	    # We distribute the moistening budget proportional to the local
   145	    # subsaturation deficit, then normalize so the column integral
   146	    # exactly equals (1 - alpha_heat) * MC / tau_relax.
   147	
   148	    # Implied condensation rate from heating (moisture sink, kg/kg/s).
   149	    # This is the per-level rate at which Kuo converts vapor to cloud
   150	    # water by latent heat balance: c_pd * dT_dt = L_v * (-dq_v) for
   151	    # the condensation contribution. Microphysics receives this
   152	    # directly via ``dq_c_conv_dt`` so it can process the convective
   153	    # condensate through its full chain (autoconversion, sedimentation,
   154	    # evaporation) rather than the legacy assumption that all of it
   155	    # falls instantly to the surface.
   156	    implied_condensation = dT_dt * constants.c_pd / constants.L_v  # (ncol, nlev)
   157	
   158	    # Subsaturation deficit profile for distributing moistening
   159	    deficit = jnp.maximum(q_sat - q_v, 0.0)  # (ncol, nlev)
   160	    deficit_integral = jnp.sum(deficit * dp, axis=1, keepdims=True) / constants.g  # (ncol, 1)
   161	    deficit_integral_safe = jnp.maximum(deficit_integral, 1e-20)
   162	
   163	    # Moistening budget: (1 - alpha_heat) * MC / tau_relax [kg/m^2/s]
   164	    moistening_budget = (
   165	        trigger * (1.0 - config.alpha_heat) * MC / config.tau_relax
   166	    )  # (ncol,)
   167	
   168	    # Distribute moistening proportional to deficit, normalized to budget
   169	    dq_v_dt = (
   170	        moistening_budget[:, None]
   171	        * (deficit / deficit_integral_safe)
   172	        * constants.g / dp
   173	    )  # (ncol, nlev) [kg/kg/s]
   174	
   175	    # Subtract condensation implied by heating
   176	    dq_v_dt = dq_v_dt - implied_condensation
   177	
   178	    # 7. Convective source for cloud water — column integral equals
   179	    # Kuo's design-intent condensation rate ``trigger * alpha_heat *
   180	    # MC / tau_relax`` (the gross condensation that microphysics
   181	    # processes), distributed per-level by the implied-condensation
   182	    # profile from latent heating.
   183	    #
   184	    # Critically the target rate is *MC-gated*: it is zero whenever
   185	    # MC = 0, so the scheme does **not** create cloud water in
   186	    # undersaturated columns. (A naive ``max(implied_condensation, 0)``
   187	    # at every level would create cloud water in an undersaturated
   188	    # column whenever the smooth sigmoid trigger had any nonzero
   189	    # value — at the default ``smooth_trigger_sharpness`` and
   190	    # ``me_threshold`` the trigger is ≈0.475 at MC=0, large enough
   191	    # to yield a spurious ``dT_dt`` and therefore a spurious
   192	    # condensation rate from a column with no moisture excess.)
   193	    #
   194	    # When MC > 0 the rescaling produces the same column total as the
   195	    # design-intent formula and the same per-level shape as the
   196	    # implied-condensation profile (no underreporting of the
   197	    # moistening fraction).
   198	    local_cond = jnp.maximum(implied_condensation, 0.0)
   199	    col_local_cond = jnp.sum(local_cond * dp / constants.g, axis=-1, keepdims=True)
   200	    target_col_cond = (
   201	        trigger * config.alpha_heat * MC / config.tau_relax
   202	    )[:, None]
   203	    dq_c_conv_dt = local_cond * (
   204	        target_col_cond / jnp.clip(col_local_cond, 1e-30, None)
   205	    )  # (ncol, nlev) [kg/kg/s]
   206	
   207	    # 8. CAPE diagnostic
   208	    cape = compute_cape(T, T_moist, p_full, p_half)  # (ncol,)
   209	
   210	    return ConvectionOutput(
   211	        dT_dt=dT_dt,
   212	        dq_v_dt=dq_v_dt,
   213	        dq_c_conv_dt=dq_c_conv_dt,
   214	        cape=cape,
   215	        convective_mask=trigger,
   216	    )

 succeeded in 0ms:
     1	"""Deep Convective Adjustment (DCA) scheme.
     2	
     3	The simplest convection parameterization: scans from bottom to top,
     4	adjusting adjacent layer pairs toward moist-adiabatic neutrality.
     5	Excess moisture is removed as precipitation.
     6	
     7	Uses jax.lax.scan for JIT-friendliness and differentiability.
     8	Smooth sigmoid triggers ensure continuous gradients.
     9	
    10	References
    11	----------
    12	- Manabe, S., Smagorinsky, J., & Strickler, R. F. (1965).
    13	  Simulated climatology of a general circulation model with a
    14	  hydrological cycle. Mon. Wea. Rev., 93, 769-798.
    15	"""
    16	
    17	from __future__ import annotations
    18	
    19	import jax
    20	import jax.numpy as jnp
    21	
    22	from legoesm import constants
    23	from legoesm.thermo import saturation_mixing_ratio
    24	from legoesm.atmosphere.physics.thermodynamics import (
    25	    moist_adiabat_lapse_rate,
    26	    compute_cape,
    27	)
    28	from legoesm.atmosphere.physics.convection.config import DCAConfig
    29	from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    30	
    31	
    32	def _adjust_one_iteration(
    33	    T: jax.Array,
    34	    q_v: jax.Array,
    35	    p_full: jax.Array,
    36	    dp: jax.Array,
    37	    mixing_fraction: float,
    38	) -> tuple[jax.Array, jax.Array, jax.Array]:
    39	    """One bottom-to-top sweep adjusting unstable layer pairs.
    40	
    41	    Scans from the bottom-most pair upward using jax.lax.scan.
    42	    For each adjacent pair (k, k-1) with k being lower:
    43	    - Compare actual lapse rate to moist adiabatic
    44	    - If unstable, adjust toward neutral with smooth blending
    45	    - Redistribute excess moisture as precipitation
    46	
    47	    Parameters
    48	    ----------
    49	    T : jax.Array
    50	        Temperature [K], shape (ncol, nlev).
    51	    q_v : jax.Array
    52	        Water vapor specific humidity [kg/kg], shape (ncol, nlev).
    53	    p_full : jax.Array
    54	        Pressure at full levels [Pa], shape (ncol, nlev).
    55	    dp : jax.Array
    56	        Layer thickness [Pa], shape (ncol, nlev).
    57	    mixing_fraction : float
    58	        Fraction of adjustment per iteration.
    59	
    60	    Returns
    61	    -------
    62	    T_new : jax.Array
    63	        Adjusted temperature, shape (ncol, nlev).
    64	    q_v_new : jax.Array
    65	        Adjusted moisture, shape (ncol, nlev).
    66	    precip_col : jax.Array
    67	        Precipitation from this sweep [kg/m^2/s equivalent: kg/kg * Pa/g],
    68	        shape (ncol,).
    69	    """
    70	    ncol, nlev = T.shape
    71	
    72	    # Reverse to scan from bottom to top
    73	    # Level indices: 0=top, nlev-1=bottom
    74	    # Reversed: 0=bottom, nlev-1=top
    75	    T_rev = T[:, ::-1]          # (ncol, nlev)
    76	    q_v_rev = q_v[:, ::-1]      # (ncol, nlev)
    77	    p_rev = p_full[:, ::-1]     # (ncol, nlev)
    78	    dp_rev = dp[:, ::-1]        # (ncol, nlev)
    79	
    80	    def scan_step(carry, k):
    81	        """Adjust adjacent pair (k-1, k) using progressively updated profiles."""
    82	        T_work, q_work, precip_accum = carry
    83	
    84	        T_below = T_work[:, k - 1]
    85	        q_below = q_work[:, k - 1]
    86	        p_below = p_rev[:, k - 1]
    87	        dp_below = dp_rev[:, k - 1]
    88	
    89	        T_upper = T_work[:, k]
    90	        q_upper = q_work[:, k]
    91	        p_upper = p_rev[:, k]
    92	        dp_upper = dp_rev[:, k]
    93	
    94	        # Midpoint for lapse rate
    95	        p_mid = 0.5 * (p_below + p_upper)
    96	        T_mid = 0.5 * (T_below + T_upper)
    97	        dp_pair = p_below - p_upper  # pressure difference (positive)
    98	        dp_pair = jnp.clip(dp_pair, 1.0, None)
    99	
   100	        # Actual lapse rate: dT/dp (temperature decrease per pressure decrease)
   101	        actual_dTdp = (T_below - T_upper) / dp_pair
   102	
   103	        # Moist adiabatic lapse rate
   104	        gamma_m = moist_adiabat_lapse_rate(T_mid, p_mid)
   105	
   106	        # Dry adiabatic lapse rate for normalization
   107	        gamma_dry = constants.R_d * T_mid / (constants.c_pd * p_mid)
   108	
   109	        # Dimensionless instability: positive means superadiabatic
   110	        instability = (actual_dTdp - gamma_m) / jnp.clip(gamma_dry, 1e-10, None)
   111	
   112	        # Smooth trigger: sigmoid with steep transition on dimensionless metric
   113	        blend = jax.nn.sigmoid(10.0 * instability) * mixing_fraction
   114	
   115	        # Target temperature for upper level: T_target = T_below - gamma_m * dp_pair
   116	        T_target_upper = T_below - gamma_m * dp_pair
   117	
   118	        # Adjusted temperatures: weighted average preserving layer enthalpy
   119	        # Weight by layer dp for energy conservation
   120	        total_dp = dp_below + dp_upper
   121	        T_mean_weighted = (T_below * dp_below + T_upper * dp_upper) / total_dp
   122	        T_target_mean = (T_below * dp_below + T_target_upper * dp_upper) / total_dp
   123	
   124	        # Shift both layers to preserve mean while achieving target lapse rate
   125	        delta_mean = T_mean_weighted - T_target_mean
   126	        T_new_upper = T_target_upper + delta_mean
   127	        T_new_below = T_below + (T_mean_weighted - (T_new_upper * dp_upper + T_below * dp_below) / total_dp) * total_dp / dp_below
   128	
   129	        # Actually, simpler: preserve total enthalpy exactly
   130	        # T_new_below = (total_dp * T_mean_weighted - dp_upper * T_new_upper) / dp_below
   131	        T_new_below = (T_mean_weighted * total_dp - dp_upper * T_target_upper) / dp_below
   132	
   133	        # Blend between original and adjusted
   134	        T_adj_upper = T_upper + blend * (T_target_upper - T_upper)
   135	        T_adj_below = T_below + blend * (T_new_below - T_below)
   136	
   137	        # Moisture adjustment: saturate at the new temperature
   138	        q_sat_upper = saturation_mixing_ratio(T_adj_upper, p_upper)
   139	        q_sat_below = saturation_mixing_ratio(T_adj_below, p_below)
   140	
   141	        # Remove excess moisture (precipitation)
   142	        q_new_upper = jnp.minimum(q_upper, q_sat_upper)
   143	        q_new_below = jnp.minimum(q_below, q_sat_below)
   144	
   145	        # Blend moisture adjustment
   146	        q_adj_upper = q_upper + blend * (q_new_upper - q_upper)
   147	        q_adj_below = q_below + blend * (q_new_below - q_below)
   148	
   149	        # Accumulate precipitation from moisture removal
   150	        dq_upper = (q_upper - q_adj_upper) * dp_upper
   151	        dq_below = (q_below - q_adj_below) * dp_below
   152	        precip_new = precip_accum + (dq_upper + dq_below) / constants.g
   153	
   154	        T_work = T_work.at[:, k - 1].set(T_adj_below)
   155	        T_work = T_work.at[:, k].set(T_adj_upper)
   156	        q_work = q_work.at[:, k - 1].set(q_adj_below)
   157	        q_work = q_work.at[:, k].set(q_adj_upper)
   158	
   159	        return (T_work, q_work, precip_new), None
   160	
   161	    # Pin the precip carry dtype to whatever ``q * dp`` actually
   162	    # produces inside the scan body — under standard promotion the
   163	    # compute precision wins when ``q_v`` is at storage precision but
   164	    # ``dp_rev`` comes from sigma-coord arrays at compute precision.
   165	    # ``jnp.result_type`` resolves this without materializing a scalar.
   166	    _precip_dtype = jnp.result_type(q_v_rev, dp_rev)
   167	    init_carry = (T_rev, q_v_rev, jnp.zeros(ncol, dtype=_precip_dtype))
   168	    level_indices = jnp.arange(1, nlev)
   169	    (T_adj_rev, q_adj_rev, precip_col), _ = jax.lax.scan(
   170	        scan_step, init_carry, level_indices,
   171	    )
   172	
   173	    # Un-reverse to original top-to-bottom ordering
   174	    T_new = T_adj_rev[:, ::-1]
   175	    q_new = q_adj_rev[:, ::-1]
   176	
   177	    return T_new, q_new, precip_col
   178	
   179	
   180	def dca_convection(
   181	    T: jax.Array,
   182	    q_v: jax.Array,
   183	    p_full: jax.Array,
   184	    p_half: jax.Array,
   185	    dt: float,
   186	    config: DCAConfig = DCAConfig(),
   187	) -> ConvectionOutput:
   188	    """Compute Deep Convective Adjustment tendencies.
   189	
   190	    Parameters
   191	    ----------
   192	    T : jax.Array
   193	        Temperature at full levels [K], shape (ncol, nlev).
   194	    q_v : jax.Array
   195	        Water vapor specific humidity [kg/kg], shape (ncol, nlev).
   196	    p_full : jax.Array
   197	        Pressure at full levels [Pa], shape (ncol, nlev).
   198	    p_half : jax.Array
   199	        Pressure at half levels [Pa], shape (ncol, nlev+1).
   200	    dt : float
   201	        Model time step [s].
   202	    config : DCAConfig
   203	        Convection configuration.
   204	
   205	    Returns
   206	    -------
   207	    ConvectionOutput
   208	        Convective tendencies and diagnostics.
   209	    """
   210	    ncol, nlev = T.shape
   211	    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)
   212	
   213	    # Apply adjustment iterations.  ``prec_iter`` returned by the inner
   214	    # scan inherits ``q * dp`` precision (compute precision wins when
   215	    # state is f32 but sigma-coord-derived dp is f64), so pin
   216	    # ``precip_total`` to the same result-type so the outer scan carry
   217	    # input matches its output.
   218	    T_adj = T
   219	    q_adj = q_v
   220	    precip_total = jnp.zeros(ncol, dtype=jnp.result_type(q_v, dp))
   221	
   222	    def body_fn(carry, _):
   223	        T_c, q_c, prec = carry
   224	        T_new, q_new, prec_iter = _adjust_one_iteration(
   225	            T_c, q_c, p_full, dp, config.mixing_fraction,
   226	        )
   227	        return (T_new, q_new, prec + prec_iter), None
   228	
   229	    (T_adj, q_adj, precip_total), _ = jax.lax.scan(
   230	        body_fn,
   231	        (T_adj, q_adj, precip_total),
   232	        jnp.arange(config.n_iterations),
   233	    )
   234	
   235	    # CAPE diagnostic BEFORE gating (using original profiles)
   236	    cape = compute_cape(T, T_adj, p_full, p_half)
   237	
   238	    # Gate tendencies by CAPE: only adjust where CAPE exceeds threshold.
   239	    # Smooth sigmoid gating preserves differentiability.
   240	    cape_gate = jax.nn.sigmoid(
   241	        config.cape_sharpness * (cape - config.cape_threshold)
   242	    )  # (ncol,)
   243	
   244	    # Convert to tendencies, gated by CAPE
   245	    dT_dt = cape_gate[:, None] * (T_adj - T) / dt
   246	    dq_v_dt = cape_gate[:, None] * (q_adj - q_v) / dt
   247	    # Convective source for cloud water — column-conservative
   248	    # rescaling so that ∫ dq_c_conv_dt dp/g equals the column-net
   249	    # drying (matches the legacy ``precipitation`` formula). Naive
   250	    # per-level ``max(-dq_v_dt, 0)`` would create water column-wide
   251	    # whenever the adjustment has mixed-sign vapor tendencies; this
   252	    # rescaling removes that bug while keeping the field non-negative
   253	    # at every level. ``precip_total`` (the scan-accumulated column
   254	    # total) is no longer surfaced — microphysics owns the surface
   255	    # precipitation diagnostic.
   256	    del precip_total
   257	    local_cond = jnp.maximum(-dq_v_dt, 0.0)
   258	    col_local_cond = jnp.sum(local_cond * dp / constants.g, axis=-1, keepdims=True)
   259	    col_net_drying = jnp.clip(
   260	        -jnp.sum(dq_v_dt * dp / constants.g, axis=-1, keepdims=True),
   261	        0.0, None,
   262	    )
   263	    dq_c_conv_dt = local_cond * (
   264	        col_net_drying / jnp.clip(col_local_cond, 1e-30, None)
   265	    )  # (ncol, nlev) [kg/kg/s]
   266	
   267	    # Convective mask: CAPE-gated
   268	    convective_mask = cape_gate
   269	
   270	    return ConvectionOutput(
   271	        dT_dt=dT_dt,
   272	        dq_v_dt=dq_v_dt,
   273	        dq_c_conv_dt=dq_c_conv_dt,
   274	        cape=cape,
   275	        convective_mask=convective_mask,
   276	    )

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py | sed -n '1,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/kain_fritsch.py | sed -n '1,300p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Zhang & McFarlane (1995) deep-convection scheme.
     2	
     3	A single-plume mass-flux scheme with a quasi-equilibrium CAPE-relaxation
     4	closure: the cloud-base mass flux ``M_b`` is diagnosed from the column
     5	CAPE excess over a threshold and relaxed (implicit Euler) toward the
     6	diagnosed equilibrium value over a tunable timescale.  The plume itself
     7	is integrated using the shared
     8	:func:`legoesm.atmosphere.physics.convection._plume.entraining_detraining_plume`
     9	helper; the environmental tendencies (compensating subsidence +
    10	detrainment) are computed by the existing
    11	:func:`legoesm.atmosphere.physics.convection.mass_flux._apply_mass_flux_kernel`
    12	so this scheme reuses every piece of column physics rather than
    13	re-implementing it.
    14	
    15	The scheme is **smooth-everywhere** in the AD sense:
    16	
    17	* ``CAPE > threshold`` is replaced by ``cape_trigger`` (sigmoid).
    18	* ``(CAPE - threshold)+`` uses ``smooth_positive_part``.
    19	* Cloud-base / LFC / LNB localization uses the smooth-fractional
    20	  level diagnostics from ``_plume``.
    21	* The plume integrator's mass-flux profile is gated by a sigmoid on
    22	  buoyancy, not a hard cut.
    23	
    24	Convective momentum transport (CMT) is enabled by default via the
    25	Gregory et al. 1997 closure
    26	(:func:`legoesm.atmosphere.physics.convection._plume.cmt_gregory_1997`).
    27	
    28	References
    29	----------
    30	- Zhang, G. J., & McFarlane, N. A. (1995). Sensitivity of climate
    31	  simulations to the parameterization of cumulus convection in the
    32	  Canadian Climate Centre general circulation model.  *Atmos.-Ocean*,
    33	  33(3), 407–446.
    34	- Gregory, D., Kershaw, R., & Inness, P. M. (1997). Parametrization of
    35	  momentum transport by convection. II.  *Quart. J. Roy. Meteor. Soc.*,
    36	  123, 1153–1183.
    37	"""
    38	
    39	from __future__ import annotations
    40	
    41	import jax
    42	import jax.numpy as jnp
    43	
    44	from legoesm import constants
    45	from legoesm.thermo import saturation_mixing_ratio
    46	from legoesm.atmosphere.physics.thermodynamics import (
    47	    compute_cape,
    48	    compute_moist_adiabat,
    49	)
    50	
    51	from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
    52	from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    53	from legoesm.atmosphere.physics.convection.mass_flux import (
    54	    _apply_mass_flux_kernel,
    55	    _compute_column_geometry,
    56	)
    57	from legoesm.atmosphere.physics.convection._triggers import (
    58	    cape_trigger,
    59	    smooth_positive_part,
    60	)
    61	from legoesm.atmosphere.physics.convection._plume import (
    62	    cmt_gregory_1997,
    63	    compute_lcl,
    64	    entraining_detraining_plume,
    65	)
    66	
    67	
    68	__all__ = ("zhang_mcfarlane_convection",)
    69	
    70	
    71	def zhang_mcfarlane_convection(
    72	    T: jax.Array,
    73	    q_v: jax.Array,
    74	    p_full: jax.Array,
    75	    p_half: jax.Array,
    76	    u: jax.Array,
    77	    v: jax.Array,
    78	    conv_prog_profile: jax.Array,
    79	    dt: float,
    80	    config: ZhangMcFarlaneConfig = ZhangMcFarlaneConfig(),
    81	) -> tuple[ConvectionOutput, jax.Array]:
    82	    """Zhang-McFarlane deep convection (smooth, differentiable).
    83	
    84	    Parameters
    85	    ----------
    86	    T : jax.Array, shape (ncol, nlev)
    87	        Environmental temperature [K].  Surface at ``[:, -1]``.
    88	    q_v : jax.Array, shape (ncol, nlev)
    89	        Water-vapor specific humidity [kg/kg].
    90	    p_full, p_half : jax.Array
    91	        Full / half-level pressures [Pa].  Shapes ``(ncol, nlev)`` and
    92	        ``(ncol, nlev+1)`` respectively.
    93	    u, v : jax.Array, shape (ncol, nlev)
    94	        Environmental wind components [m/s].  Used by the Gregory et
    95	        al. 1997 CMT closure when ``config.enable_cmt = True``.
    96	    conv_prog_profile : jax.Array, shape (ncol, nlev)
    97	        Convection prognostic carry (PR-0 schema).  ZM uses only the
    98	        surface-adjacent slot ``[:, -1]`` to remember the previous
    99	        cloud-base mass flux ``M_b`` for implicit-Euler relaxation
   100	        toward the diagnosed equilibrium value; aloft slots are
   101	        unused (zeros in / zeros out).
   102	    dt : float
   103	        Time step [s].
   104	    config : ZhangMcFarlaneConfig
   105	        Scheme tunables — see :class:`ZhangMcFarlaneConfig`.
   106	
   107	    Returns
   108	    -------
   109	    out : ConvectionOutput
   110	        Tendencies on environment T, q_v, q_c plus the CAPE diagnostic
   111	        and (when CMT is enabled) du/dt, dv/dt.
   112	    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
   113	        Updated carry with the relaxed ``M_b_new`` packed at
   114	        ``[:, -1]``; aloft entries are zero.
   115	    """
   116	    ncol, nlev = T.shape
   117	
   118	    # -- Column geometry, moist adiabat, CAPE --------------------------------
   119	    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
   120	    T_base = T[:, -1]
   121	    q_base = q_v[:, -1]
   122	    p_base = p_full[:, -1]
   123	
   124	    T_moist = compute_moist_adiabat(T_base, p_full)
   125	    cape = compute_cape(T, T_moist, p_full, p_half)
   126	
   127	    # -- Smooth CAPE trigger and cloud-base mass-flux closure ---------------
   128	    cape_weight = cape_trigger(
   129	        cape, config.cape_threshold, config.cape_sharpness,
   130	    )
   131	    M_b_eq = (
   132	        cape_weight
   133	        * smooth_positive_part(
   134	            cape - config.cape_threshold, config.cape_sharpness,
   135	        )
   136	        / config.tau_cape
   137	    )
   138	    # Implicit-Euler relaxation toward equilibrium — stable for any
   139	    # ``dt / tau_cape`` ratio:
   140	    #     M_b_new = (M_b_old + (dt/tau) * M_b_eq) / (1 + dt/tau).
   141	    M_b_old = conv_prog_profile[:, -1]
   142	    dt_over_tau = dt / jnp.maximum(config.tau_cape, dt)
   143	    M_b = (M_b_old + dt_over_tau * M_b_eq) / (1.0 + dt_over_tau)
   144	    # Bound M_b to a literature peak tropical value (config.M_b_max,
   145	    # default 0.1 kg/m²/s).  Without this cap a column with very large
   146	    # CAPE drives M_b unboundedly and emits column heating that breaks
   147	    # the next dynamics step on the lat-lon FV pole-cell CFL.
   148	    M_b = jnp.clip(M_b, 0.0, config.M_b_max)
   149	
   150	    # -- Plume launch / cloud-base index ------------------------------------
   151	    # Surface parcel perturbed slightly per Zhang & McFarlane 1995 §3a;
   152	    # this avoids zero-perturbation degeneracies and gives a smooth
   153	    # cloud-base diagnosis.
   154	    T_parcel = T_base + config.parcel_dT
   155	    q_parcel = q_base + config.parcel_dq
   156	    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
   157	    k_base_smooth = lcl.k_lcl_smooth
   158	
   159	    # Constant entrainment / detrainment profiles in the surface-last
   160	    # convention (level index nlev-1 = surface, 0 = top).  Tunable per
   161	    # scheme but identical across levels in the standard Zhang-McFarlane
   162	    # bulk plume.
   163	    eps_profile = jnp.full_like(T, config.epsilon_0)
   164	    dlt_profile = jnp.full_like(T, config.delta_0)
   165	
   166	    plume = entraining_detraining_plume(
   167	        T, q_v, p_full, p_half, z,
   168	        T_parcel, q_parcel, k_base_smooth,
   169	        eps_profile, dlt_profile, M_b,
   170	    )
   171	
   172	    # Cap plume.M_u once at the source so every downstream use (kernel
   173	    # tendencies, CMT, q_c sources) sees the same bounded value.  The
   174	    # kernel's internal cap is now redundant but kept for safety.
   175	    plume_M_u_capped = jnp.clip(plume.M_u, 0.0, config.M_b_max)
   176	    plume = plume._replace(M_u=plume_M_u_capped)
   177	
   178	    # -- Environmental tendencies via the shared mass-flux kernel ----------
   179	    # Plume splits vapor (``plume.q_u`` — saturation-clipped per level)
   180	    # and cloud water (``plume.q_c_u`` — accumulated condensation)
   181	    # explicitly, so the kernel's ``q_c_conv_dt`` is now the correct
   182	    # detrainment of plume cloud water and we use it directly.
   183	    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
   184	        T, q_v, p_full,
   185	        plume.T_u, plume.q_u, plume.q_c_u, plume.M_u,
   186	        z, rho, config.delta_0, M_u_max=config.M_b_max,
   187	    )
   188	
   189	    # -- Convective momentum transport --------------------------------------
   190	    if config.enable_cmt:
   191	        du_dt_conv, dv_dt_conv = cmt_gregory_1997(
   192	            u, v, plume.M_u, None,
   193	            p_full, p_half, rho,
   194	            c_u=config.cmt_c_u, c_d=config.cmt_c_d,
   195	        )
   196	    else:
   197	        du_dt_conv = None
   198	        dv_dt_conv = None
   199	
   200	    # -- Convective mask (column-mean diagnostic) ---------------------------
   201	    convective_mask = cape_weight  # already a smooth (ncol,) indicator
   202	
   203	    out = ConvectionOutput(
   204	        dT_dt=dT_dt,
   205	        dq_v_dt=dq_v_dt,
   206	        dq_c_conv_dt=dq_c_conv_dt,
   207	        cape=cape,
   208	        convective_mask=convective_mask,
   209	        du_dt_conv=du_dt_conv,
   210	        dv_dt_conv=dv_dt_conv,
   211	    )
   212	
   213	    # Pack the relaxed M_b back into the surface-adjacent carry slot
   214	    # for downstream visibility (training diagnostics, conservation
   215	    # checks).  Aloft slots are zero-filled — the orchestrator schema
   216	    # is uniform across all schemes.
   217	    conv_prog_profile_new = jnp.zeros_like(conv_prog_profile).at[:, -1].set(M_b)
   218	
   219	    return out, conv_prog_profile_new

 succeeded in 0ms:
     1	"""Kain & Fritsch (1990; 2004 update) deep & shallow convection.
     2	
     3	Bulk mass-flux scheme distinguished by its boundary-layer trigger
     4	function: convection fires when a smoothly-perturbed parcel
     5	temperature at the LCL exceeds the environmental temperature there
     6	by a sigmoid amount.  Cloud-depth-dependent blending of deep and
     7	shallow branches.  No convective momentum transport — KF emits
     8	``du_dt_conv = dv_dt_conv = None`` and the orchestrator zero-fills.
     9	
    10	The scheme is **smooth-everywhere**:
    11	
    12	* The trigger function uses
    13	  ``trigger_weight = sigmoid(s * (T_LCL_perturbed - T_env_at_LCL))``
    14	  in place of the original hard ``> 0`` switch.  This is the central
    15	  AD-safety property of the smooth-everywhere KF: gradients flow
    16	  through the trigger threshold so training-time perturbations to
    17	  ``parcel_perturb_T``, ``w_thresh_offset``, and ``trigger_sharpness``
    18	  all have non-zero gradient signal.
    19	* The deep/shallow blend is a sigmoid on cloud depth.
    20	* The CAPE gate is the same ``cape_trigger`` used by ZM.
    21	
    22	References
    23	----------
    24	* Kain, J. S. & Fritsch, J. M. (1990). A one-dimensional entraining /
    25	  detraining plume model and its application in convective
    26	  parameterization.  *J. Atmos. Sci.*, 47, 2784–2802.
    27	* Kain, J. S. (2004). The Kain–Fritsch convective parameterization:
    28	  An update.  *J. Appl. Meteor.*, 43, 170–181.
    29	"""
    30	
    31	from __future__ import annotations
    32	
    33	import jax
    34	import jax.numpy as jnp
    35	
    36	from legoesm.atmosphere.physics.thermodynamics import (
    37	    compute_cape,
    38	    compute_moist_adiabat,
    39	)
    40	
    41	from legoesm.atmosphere.physics.convection.config import KainFritschConfig
    42	from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    43	from legoesm.atmosphere.physics.convection.mass_flux import (
    44	    _apply_mass_flux_kernel,
    45	    _compute_column_geometry,
    46	)
    47	from legoesm.atmosphere.physics.convection._triggers import (
    48	    cape_trigger,
    49	    smooth_level_indicator,
    50	    smooth_step,
    51	)
    52	from legoesm.atmosphere.physics.convection._plume import (
    53	    compute_lcl,
    54	    compute_lfc_lnb,
    55	    entraining_detraining_plume,
    56	)
    57	
    58	
    59	__all__ = ("kain_fritsch_convection",)
    60	
    61	
    62	def _interpolate_at_smooth_level(
    63	    profile: jax.Array,
    64	    k_smooth: jax.Array,
    65	    sharpness: float = 2.0,
    66	) -> jax.Array:
    67	    """Smooth interpolation of ``profile`` at a fractional level index.
    68	
    69	    Uses a soft level-membership weighting so that the result is
    70	    differentiable in ``k_smooth``.  ``profile`` shape ``(ncol, nlev)``,
    71	    ``k_smooth`` shape ``(ncol,)``; returns shape ``(ncol,)``.
    72	    """
    73	    nlev = profile.shape[-1]
    74	    levels = jnp.arange(nlev, dtype=profile.dtype)
    75	    # Centered Gaussian-like weight peaked at k_smooth.
    76	    weight = jax.nn.softmax(
    77	        -sharpness * (levels[None, :] - k_smooth[:, None]) ** 2,
    78	        axis=-1,
    79	    )
    80	    return jnp.sum(weight * profile, axis=-1)
    81	
    82	
    83	def kain_fritsch_convection(
    84	    T: jax.Array,
    85	    q_v: jax.Array,
    86	    p_full: jax.Array,
    87	    p_half: jax.Array,
    88	    w_grid: jax.Array,
    89	    conv_prog_profile: jax.Array,
    90	    dt: float,
    91	    config: KainFritschConfig = KainFritschConfig(),
    92	) -> tuple[ConvectionOutput, jax.Array]:
    93	    """Kain-Fritsch convection (smooth, differentiable).
    94	
    95	    Parameters
    96	    ----------
    97	    T : jax.Array, shape (ncol, nlev)
    98	        Environmental temperature [K].  Surface at ``[:, -1]``.
    99	    q_v : jax.Array, shape (ncol, nlev)
   100	        Water-vapor specific humidity [kg/kg].
   101	    p_full, p_half : jax.Array
   102	        Full / half-level pressures [Pa].
   103	    w_grid : jax.Array, shape (ncol, nlev)
   104	        Grid-scale vertical-velocity proxy [m/s].  The non-hydrostatic
   105	        bridge passes ``state.w`` (interpolated to full levels).  The
   106	        hydrostatic and spectral-PE bridges derive ``w`` from
   107	        ``∇·v_h`` via the standard sigma-coord continuity (``σ̇`` →
   108	        ``ω``) and then ``w = -ω/(ρ g)`` (see
   109	        :func:`legoesm.atmosphere.physics._shared.diagnose_grid_w_from_omega`).
   110	        On grids that do not expose a divergence operator the bridge
   111	        falls back to zeros and the trigger is driven by
   112	        ``parcel_perturb_T`` alone.
   113	    conv_prog_profile : jax.Array, shape (ncol, nlev)
   114	        Convection prognostic carry.  KF is fully diagnostic at the
   115	        physics-state level — we pack the diagnosed cloud-base mass
   116	        flux ``M_b`` at ``[:, -1]`` for visibility but do not use it
   117	        for relaxation (unlike ZM).
   118	    dt : float
   119	        Time step [s].
   120	    config : KainFritschConfig
   121	        Scheme tunables.
   122	
   123	    Returns
   124	    -------
   125	    out : ConvectionOutput
   126	        Tendencies on environment T, q_v, q_c plus CAPE diagnostic.
   127	        ``du_dt_conv = dv_dt_conv = None`` (KF has no CMT).
   128	    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
   129	        Updated carry with diagnosed ``M_b`` packed at ``[:, -1]``.
   130	    """
   131	    ncol, nlev = T.shape
   132	    del conv_prog_profile  # KF is diagnostic; we only emit a fresh profile.
   133	
   134	    # -- Column geometry, moist adiabat, CAPE ------------------------------
   135	    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
   136	    T_base = T[:, -1]
   137	    q_base = q_v[:, -1]
   138	    p_base = p_full[:, -1]
   139	
   140	    T_moist = compute_moist_adiabat(T_base, p_full)
   141	    cape = compute_cape(T, T_moist, p_full, p_half)
   142	
   143	    # -- LCL, LFC, LNB diagnostics -----------------------------------------
   144	    T_parcel = T_base + config.parcel_perturb_T
   145	    q_parcel = q_base + config.parcel_perturb_q
   146	    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
   147	    k_lcl_smooth = lcl.k_lcl_smooth
   148	    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)
   149	
   150	    # -- The KF trigger function (the AD chokepoint) -----------------------
   151	    # Smooth-interpolate environment T and grid-scale w at the LCL.
   152	    T_env_at_lcl = _interpolate_at_smooth_level(T, k_lcl_smooth)
   153	    w_grid_at_lcl = _interpolate_at_smooth_level(w_grid, k_lcl_smooth)
   154	
   155	    T_lcl_perturbed = (
   156	        lcl.T_lcl + config.w_thresh_scale * w_grid_at_lcl - config.w_thresh_offset
   157	    )
   158	    trigger_weight = smooth_step(
   159	        T_lcl_perturbed - T_env_at_lcl, config.trigger_sharpness,
   160	    )
   161	
   162	    # -- CAPE gate (a secondary safety net) --------------------------------
   163	    cape_weight = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)
   164	    overall_weight = trigger_weight * cape_weight
   165	
   166	    # -- Cloud-base mass flux closure: CAPE / cape_consumption_time --------
   167	    # Following Kain (2004) §3 — the cloud-base mass flux is
   168	    # ``M_b = CAPE / (g * tau_consume)`` modulated by the trigger.
   169	    M_b = (
   170	        overall_weight
   171	        * cape
   172	        / jnp.maximum(config.cape_consumption_time, dt)
   173	    )
   174	    # Bound M_b to a literature peak tropical value
   175	    # (config.M_b_max, default 0.1 kg/m²/s) — see ZhangMcFarlaneConfig.
   176	    M_b = jnp.clip(M_b, 0.0, config.M_b_max)
   177	
   178	    # -- Plume integration -------------------------------------------------
   179	    # Same entraining-detraining plume as ZM, with KF default
   180	    # ``epsilon_0 = delta_0 = 2e-3`` (a slightly stronger entrainment).
   181	    eps_profile = jnp.full_like(T, config.epsilon_0)
   182	    dlt_profile = jnp.full_like(T, config.delta_0)
   183	    plume = entraining_detraining_plume(
   184	        T, q_v, p_full, p_half, z,
   185	        T_parcel, q_parcel, k_lcl_smooth,
   186	        eps_profile, dlt_profile, M_b,
   187	    )
   188	
   189	    # -- Cloud depth — z(LCL) → z(LNB) -------------------------------------
   190	    z_lcl = _interpolate_at_smooth_level(z, k_lcl_smooth)
   191	    z_lnb = _interpolate_at_smooth_level(z, k_lnb_smooth)
   192	    cloud_depth = jnp.maximum(z_lnb - z_lcl, 0.0)
   193	
   194	    # Deep vs shallow blend — applied as a per-column scalar weight.
   195	    deep_weight = smooth_step(
   196	        cloud_depth - config.cloud_depth_min, config.cloud_depth_sharpness,
   197	    )
   198	    if config.enable_shallow:
   199	        shallow_weight = 1.0 - deep_weight
   200	    else:
   201	        shallow_weight = jnp.zeros_like(deep_weight)
   202	    branch_weight = deep_weight + shallow_weight  # = 1 with shallow on; = deep_weight only
   203	
   204	    # Cap plume.M_u once at the source so every downstream use sees
   205	    # the bounded value (see ZM).
   206	    plume_M_u_capped = jnp.clip(plume.M_u, 0.0, config.M_b_max)
   207	    plume = plume._replace(M_u=plume_M_u_capped)
   208	
   209	    # -- Environmental tendencies via the shared mass-flux kernel ----------
   210	    # Plume splits vapor (``plume.q_u``) and cloud water
   211	    # (``plume.q_c_u``) explicitly so we use the kernel's correct
   212	    # cloud-water source directly (see ZM).
   213	    dT_dt_raw, dq_v_dt_raw, dq_c_conv_dt_raw = _apply_mass_flux_kernel(
   214	        T, q_v, p_full,
   215	        plume.T_u, plume.q_u, plume.q_c_u, plume.M_u,
   216	        z, rho, config.delta_0, M_u_max=config.M_b_max,
   217	    )
   218	
   219	    # Apply the deep+shallow weight as a per-column scalar.
   220	    dT_dt = dT_dt_raw * branch_weight[:, None]
   221	    dq_v_dt = dq_v_dt_raw * branch_weight[:, None]
   222	    dq_c_conv_dt = dq_c_conv_dt_raw * branch_weight[:, None]
   223	
   224	    out = ConvectionOutput(
   225	        dT_dt=dT_dt,
   226	        dq_v_dt=dq_v_dt,
   227	        dq_c_conv_dt=dq_c_conv_dt,
   228	        cape=cape,
   229	        convective_mask=overall_weight,
   230	        # KF has no convective momentum transport.
   231	        du_dt_conv=None,
   232	        dv_dt_conv=None,
   233	    )
   234	
   235	    # Diagnostic carry: pack final M_b at [:, -1] for visibility; the
   236	    # other slots stay zero — KF is fully diagnostic.
   237	    conv_prog_profile_new = (
   238	        jnp.zeros((ncol, nlev), dtype=T.dtype).at[:, -1].set(M_b)
   239	    )
   240	    return out, conv_prog_profile_new

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/emanuel.py | sed -n '1,310p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Emanuel (1991) buoyancy-sorting convection.
     2	
     3	The distinctive feature versus ZM (single bulk plume) and KF
     4	(single plume with deep/shallow blend) is the **buoyancy-sorted
     5	ensemble**: at each cloud level the parcel may mix with environmental
     6	air in a discrete spectrum of mixing fractions ``f_i ∈ [0, 1]``.  The
     7	mixed parcel's buoyancy at that level determines whether it
     8	contributes to the upward mass flux (positive buoyancy) or detrains
     9	into a downdraft (negative buoyancy).  This produces a per-level
    10	spread in detrainment height that is impossible with a single bulk
    11	plume.
    12	
    13	The smooth-everywhere replacement: instead of a hard ``if B_mix > 0:
    14	ascend`` switch we weight each mixing fraction's contribution by
    15	``sigmoid(s * B_mix_i)``.  This preserves training-time gradients
    16	through the buoyancy threshold while still producing the
    17	qualitatively-correct detrainment-height spread.
    18	
    19	Optional unsaturated-downdraft branch (rain evaporation cooling) is
    20	toggled by ``enable_unsaturated_downdraft``.  Implementation: a
    21	fraction ``downdraft_efficiency`` of the column-integrated detrained
    22	condensate is moved as a per-level cooling + moistening tendency in
    23	the cloud layer below LCL.
    24	
    25	No convective momentum transport (Emanuel CMT is a separate
    26	extension, deferred).
    27	
    28	References
    29	----------
    30	* Emanuel, K. A. (1991). A scheme for representing cumulus convection
    31	  in large-scale models.  *J. Atmos. Sci.*, 48, 2313–2335.
    32	"""
    33	
    34	from __future__ import annotations
    35	
    36	import jax
    37	import jax.numpy as jnp
    38	
    39	from legoesm import constants
    40	from legoesm.thermo import saturation_mixing_ratio
    41	from legoesm.atmosphere.physics.thermodynamics import (
    42	    compute_cape,
    43	    compute_moist_adiabat,
    44	)
    45	from legoesm.atmosphere.physics.convection.config import EmanuelConfig
    46	from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    47	from legoesm.atmosphere.physics.convection.mass_flux import (
    48	    _apply_mass_flux_kernel,
    49	    _compute_column_geometry,
    50	)
    51	from legoesm.atmosphere.physics.convection._triggers import (
    52	    cape_trigger,
    53	    smooth_positive_part,
    54	    smooth_step,
    55	)
    56	from legoesm.atmosphere.physics.convection._plume import (
    57	    compute_lcl,
    58	    entraining_detraining_plume,
    59	)
    60	
    61	
    62	__all__ = ("emanuel_convection",)
    63	
    64	
    65	def emanuel_convection(
    66	    T: jax.Array,
    67	    q_v: jax.Array,
    68	    p_full: jax.Array,
    69	    p_half: jax.Array,
    70	    conv_prog_profile: jax.Array,
    71	    dt: float,
    72	    config: EmanuelConfig = EmanuelConfig(),
    73	) -> tuple[ConvectionOutput, jax.Array]:
    74	    """Emanuel buoyancy-sorting convection (smooth, differentiable).
    75	
    76	    Parameters
    77	    ----------
    78	    T : jax.Array, shape (ncol, nlev)
    79	        Environmental temperature [K].
    80	    q_v : jax.Array, shape (ncol, nlev)
    81	        Water-vapor specific humidity [kg/kg].
    82	    p_full, p_half : jax.Array
    83	        Full / half-level pressures [Pa].
    84	    conv_prog_profile : jax.Array, shape (ncol, nlev)
    85	        Convection prognostic carry — Emanuel is diagnostic; we pack
    86	        the diagnosed cloud-base mass flux at ``[:, -1]`` for
    87	        visibility.
    88	    dt : float
    89	        Time step [s].
    90	    config : EmanuelConfig
    91	        Scheme tunables.
    92	
    93	    Returns
    94	    -------
    95	    out : ConvectionOutput
    96	        Tendencies on environment T, q_v, q_c plus CAPE diagnostic.
    97	        ``du_dt_conv = dv_dt_conv = None`` (Emanuel has no CMT in
    98	        this implementation).
    99	    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
   100	    """
   101	    ncol, nlev = T.shape
   102	    del conv_prog_profile  # diagnostic carry only — emit a fresh profile
   103	
   104	    # -- Column geometry, moist adiabat, CAPE ------------------------------
   105	    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
   106	    T_base = T[:, -1]
   107	    q_base = q_v[:, -1]
   108	    p_base = p_full[:, -1]
   109	
   110	    T_moist = compute_moist_adiabat(T_base, p_full)
   111	    cape = compute_cape(T, T_moist, p_full, p_half)
   112	
   113	    # -- Smooth CAPE trigger -----------------------------------------------
   114	    cape_weight = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)
   115	
   116	    # -- LCL and cloud base ------------------------------------------------
   117	    T_parcel = T_base + config.parcel_perturb_T
   118	    q_parcel = q_base + config.parcel_perturb_q
   119	    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
   120	    k_lcl_smooth = lcl.k_lcl_smooth
   121	
   122	    # -- Cloud-base mass flux closure (CAPE-relaxation, Emanuel style) ----
   123	    # Emanuel 1991 uses a sub-cloud-layer relaxation.  We approximate
   124	    # it as a CAPE-driven mass flux with the configured timescale.
   125	    M_b_eq = (
   126	        cape_weight
   127	        * smooth_positive_part(cape - config.cape_threshold, config.cape_sharpness)
   128	        / config.sub_cloud_relaxation
   129	    )
   130	    # See ZhangMcFarlaneConfig.M_b_max.
   131	    M_b = jnp.clip(M_b_eq, 0.0, config.M_b_max)
   132	
   133	    # -- Standard entraining plume from cloud base ------------------------
   134	    eps_profile = jnp.full_like(T, config.epsilon_0)
   135	    dlt_profile = jnp.full_like(T, config.delta_0)
   136	    plume = entraining_detraining_plume(
   137	        T, q_v, p_full, p_half, z,
   138	        T_parcel, q_parcel, k_lcl_smooth,
   139	        eps_profile, dlt_profile, M_b,
   140	    )
   141	
   142	    # -- Buoyancy-sorted ensemble enhancement to detrainment ---------------
   143	    # Build a discrete grid of mixing fractions f_i ∈ [0, 1] with N
   144	    # equal-weight bins.  At each level k the mixed parcel buoyancy is
   145	    #     B_mix_i(k) = f_i * (T_u(k) - T_env(k))
   146	    # and its smooth contribution to "ascending" mass is sigmoid(s *
   147	    # B_mix_i).  The buoyancy-sort multiplier on detrainment is the
   148	    # variance of the ascending-weight distribution: where all
   149	    # fractions agree (deep in the cloud or above LNB) it's small;
   150	    # where the ensemble is split (near LNB) it's large.  This acts as
   151	    # a per-level enhancement of the bulk detrainment, producing the
   152	    # height-spread that distinguishes Emanuel from a single plume.
   153	    n_frac = config.n_mixing_fractions
   154	    fractions = jnp.linspace(
   155	        1.0 / (2 * n_frac), 1.0 - 1.0 / (2 * n_frac), n_frac
   156	    )  # midpoint fractions
   157	    B_u = plume.B_u                                       # (ncol, nlev)
   158	    # Outer-product: (ncol, nlev, n_frac).
   159	    B_mix = B_u[:, :, None] * fractions[None, None, :]
   160	    ascending_weight_per_frac = jax.nn.sigmoid(
   161	        config.smooth_trigger_sharpness * B_mix
   162	    )                                                    # (ncol, nlev, n_frac)
   163	    # Mean ascending fraction at each level.
   164	    ascending_mean = jnp.mean(ascending_weight_per_frac, axis=-1)
   165	    # Variance — peaks where the ensemble is split (B_u ≈ 0).
   166	    ascending_var = jnp.mean(
   167	        (ascending_weight_per_frac - ascending_mean[..., None]) ** 2, axis=-1
   168	    )
   169	    # Buoyancy-sort detrainment multiplier in [1, 1 + cu].
   170	    sort_multiplier = 1.0 + 4.0 * config.cu_coefficient * ascending_var
   171	
   172	    # Cap plume.M_u once at the source so every downstream use sees
   173	    # the bounded value (see ZM).
   174	    plume_M_u_capped = jnp.clip(plume.M_u, 0.0, config.M_b_max)
   175	    plume = plume._replace(M_u=plume_M_u_capped)
   176	
   177	    # -- Environment tendencies via the shared mass-flux kernel ------------
   178	    # Plume splits vapor (``plume.q_u``) and cloud water (``plume.q_c_u``)
   179	    # explicitly so we use the kernel's correct cloud-water source.
   180	    dT_dt_raw, dq_v_dt_raw, dq_c_conv_dt_raw = _apply_mass_flux_kernel(
   181	        T, q_v, p_full,
   182	        plume.T_u, plume.q_u, plume.q_c_u, plume.M_u,
   183	        z, rho, config.delta_0, M_u_max=config.M_b_max,
   184	    )
   185	
   186	    # Emanuel's per-level detrainment enhancement:
   187	    dT_dt = dT_dt_raw * sort_multiplier
   188	    dq_v_dt = dq_v_dt_raw * sort_multiplier
   189	    dq_c_conv_dt = dq_c_conv_dt_raw * sort_multiplier
   190	
   191	    # -- Optional unsaturated-downdraft cooling ---------------------------
   192	    # Implemented as a static Python branch (closure-time decision) so
   193	    # it does NOT add a JAX trace overhead when disabled.  When enabled
   194	    # the column-integrated condensate evaporates a fraction
   195	    # ``downdraft_efficiency`` below LCL, cooling and moistening the
   196	    # sub-cloud layer.
   197	    if config.enable_unsaturated_downdraft:
   198	        # Smooth indicator of "below LCL" (surface-last: index larger
   199	        # than k_lcl_smooth ⇒ below).
   200	        nlev_idx = jnp.arange(nlev, dtype=T.dtype)
   201	        below_lcl = jax.nn.sigmoid(
   202	            2.0 * (nlev_idx[None, :] - k_lcl_smooth[:, None])
   203	        )                                                # (ncol, nlev)
   204	        # Column-integrated condensate source [kg/m^2/s].
   205	        dp = p_half[:, 1:] - p_half[:, :-1]
   206	        column_condensate = jnp.sum(dq_c_conv_dt_raw * dp, axis=-1) / constants.g
   207	        # Distribute evaporation cooling proportional to below_lcl
   208	        # mass.
   209	        below_mass = jnp.sum(below_lcl * dp, axis=-1) / constants.g
   210	        evap_rate = (
   211	            config.downdraft_efficiency
   212	            * column_condensate[:, None]
   213	            * below_lcl
   214	            / jnp.maximum(below_mass[:, None], 1e-6)
   215	        )
   216	        # Evaporation cools T and moistens q (BL).  Conserve column
   217	        # water by removing the same column-integrated mass from the
   218	        # cloud-water source — distributed proportional to where
   219	        # cloud water is *produced* (i.e. dq_c_conv_dt_raw), not where
   220	        # it evaporates (BL).  The earlier formulation subtracted
   221	        # ``evap_rate`` from ``dq_c_conv_dt`` *at the BL*, then clipped
   222	        # to zero — which lost the bookkeeping (the BL has little
   223	        # ``dq_c_conv_dt_raw``) and effectively created vapor from
   224	        # nothing, flipping the sign of column ``Q_v`` on CAPE-positive
   225	        # soundings.
   226	        dT_evap = -(constants.L_v / constants.c_pd) * evap_rate
   227	        dq_v_evap = evap_rate
   228	        dT_dt = dT_dt + dT_evap
   229	        dq_v_dt = dq_v_dt + dq_v_evap
   230	        # Per-column total evap [kg/m²/s] = downdraft_efficiency *
   231	        # column_condensate by construction.  Subtract from the source
   232	        # at the levels where condensate is produced to keep
   233	        # ``dq_c_conv_dt ≥ 0`` and column water conserved.
   234	        col_dq_c = jnp.sum(dq_c_conv_dt_raw * dp, axis=-1) / constants.g
   235	        weight = dq_c_conv_dt_raw / jnp.maximum(col_dq_c[:, None], 1e-12)
   236	        weight = jnp.where(
   237	            (col_dq_c > 1e-12)[:, None], weight, 0.0,
   238	        )
   239	        # ``∫ weight * dp/g = 1`` when ``col_dq_c > 0``, so
   240	        # ``∫ subtract * dp/g = downdraft_efficiency * col_dq_c``.
   241	        subtract = (
   242	            config.downdraft_efficiency * col_dq_c[:, None] * weight
   243	        )
   244	        dq_c_conv_dt = dq_c_conv_dt - subtract
   245	
   246	    out = ConvectionOutput(
   247	        dT_dt=dT_dt,
   248	        dq_v_dt=dq_v_dt,
   249	        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
   250	        cape=cape,
   251	        convective_mask=cape_weight,
   252	        du_dt_conv=None,
   253	        dv_dt_conv=None,
   254	    )
   255	    conv_prog_profile_new = (
   256	        jnp.zeros((ncol, nlev), dtype=T.dtype).at[:, -1].set(M_b)
   257	    )
   258	    return out, conv_prog_profile_new

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/bechtold.py | sed -n '1,380p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/config.py | sed -n '1,360p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/tiedtke.py | sed -n '1,360p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Bechtold / IFS convection (Bechtold et al. 2008, 2014).
     2	
     3	Builds on the Tiedtke 1989 skeleton (see :mod:`.tiedtke`) and adds:
     4	
     5	1. **PBL-CAPE / departure-CAPE closure** (Bechtold 2008).  ``M_b`` is
     6	   diagnosed from a mass-weighted parcel within the boundary layer
     7	   rather than the surface parcel.  This sharpens the diurnal cycle
     8	   of deep convection over land.
     9	2. **AR1 stochastic perturbation** (Bechtold 2014).  ``M_b *= (1 +
    10	   amplitude * ε)`` where ``ε`` is an AR1 process with prescribed
    11	   decorrelation timescale.  The AR1 noise state is carried in
    12	   :attr:`legoesm.atmosphere.physics.physics_state.PhysicsState.conv_stoch_state`.
    13	   Stochasticity is OFF by default; when enabled, the leaf takes a
    14	   ``prng_key`` argument.
    15	
    16	The smooth-everywhere / differentiability properties are inherited
    17	from Tiedtke; the AR1 stochastic factor is treated as a fixed
    18	multiplier per call so ``jax.grad`` flows through the deterministic
    19	``M_b``.
    20	
    21	References
    22	----------
    23	* Bechtold, P., Köhler, M., Jung, T., Doblas-Reyes, F., Leutbecher,
    24	  M., Rodwell, M. J., Vitart, F., & Balsamo, G. (2008). Advances in
    25	  simulating atmospheric variability with the ECMWF model.  *Quart.
    26	  J. Roy. Meteor. Soc.*, 134, 1337–1351.
    27	* Bechtold, P., Semane, N., Lopez, P., Chaboureau, J.-P., Beljaars,
    28	  A., & Bormann, N. (2014). Representing equilibrium and
    29	  nonequilibrium convection in large-scale models.  *J. Atmos. Sci.*,
    30	  71, 734–753.
    31	"""
    32	
    33	from __future__ import annotations
    34	
    35	import jax
    36	import jax.numpy as jnp
    37	
    38	from legoesm import constants
    39	from legoesm.thermo import saturation_mixing_ratio
    40	from legoesm.atmosphere.physics.thermodynamics import (
    41	    compute_cape,
    42	    compute_moist_adiabat,
    43	)
    44	from legoesm.atmosphere.physics.convection.config import BechtoldConfig
    45	from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    46	from legoesm.atmosphere.physics.convection.mass_flux import (
    47	    _apply_mass_flux_kernel,
    48	    stratosphere_mass_flux_gate,
    49	    _compute_column_geometry,
    50	)
    51	from legoesm.atmosphere.physics.convection._triggers import (
    52	    cape_trigger,
    53	    smooth_level_indicator,
    54	    smooth_positive_part,
    55	    smooth_step,
    56	)
    57	from legoesm.atmosphere.physics.convection._plume import (
    58	    cmt_gregory_1997,
    59	    compute_lcl,
    60	    compute_lfc_lnb,
    61	    entraining_detraining_plume,
    62	)
    63	
    64	
    65	__all__ = ("bechtold_convection",)
    66	
    67	
    68	def bechtold_convection(
    69	    T: jax.Array,
    70	    q_v: jax.Array,
    71	    p_full: jax.Array,
    72	    p_half: jax.Array,
    73	    u: jax.Array,
    74	    v: jax.Array,
    75	    conv_prog_profile: jax.Array,
    76	    conv_stoch_state: jax.Array,
    77	    prng_key: jax.Array | None,
    78	    dt: float,
    79	    config: BechtoldConfig = BechtoldConfig(),
    80	    moisture_convergence: jax.Array | None = None,
    81	) -> tuple[ConvectionOutput, jax.Array, jax.Array]:
    82	    """Bechtold/IFS convection (smooth, differentiable).
    83	
    84	    Parameters
    85	    ----------
    86	    T, q_v : jax.Array, shape (ncol, nlev)
    87	        Environmental temperature and water-vapor specific humidity.
    88	    p_full, p_half : jax.Array
    89	        Full / half-level pressures.
    90	    u, v : jax.Array, shape (ncol, nlev)
    91	        Environmental winds (for CMT).
    92	    conv_prog_profile : jax.Array, shape (ncol, nlev)
    93	        Updraft mass-flux profile from the previous step.
    94	    conv_stoch_state : jax.Array, shape (ncol,)
    95	        AR1 noise state from the previous step.
    96	    prng_key : jax.Array or None
    97	        PRNG key for the stochastic perturbation.  When
    98	        ``config.enable_stochastic`` is ``False`` this argument is
    99	        ignored.  When stochastic is on but ``prng_key`` is ``None``
   100	        the leaf falls back to a deterministic (zero-noise)
   101	        realization.  The convection bridge derives a per-step
   102	        sub-key from ``PhysicsState.prng_key`` (split + ``fold_in``
   103	        with module id ``0xBEC4``) when stochasticity is enabled.
   104	    dt : float
   105	        Time step [s].
   106	    config : BechtoldConfig
   107	
   108	    Returns
   109	    -------
   110	    out : ConvectionOutput
   111	    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
   112	        Updated M_u profile (implicit-Euler relaxed).
   113	    conv_stoch_state_new : jax.Array, shape (ncol,)
   114	        Updated AR1 noise state.
   115	    """
   116	    ncol, nlev = T.shape
   117	
   118	    # -- Column geometry, moist adiabat, CAPE ------------------------------
   119	    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
   120	    T_base = T[:, -1]
   121	    q_base = q_v[:, -1]
   122	    p_base = p_full[:, -1]
   123	
   124	    # -- PBL parcel: mass-weighted average over the boundary-layer
   125	    # depth.  Smooth weighting via ``smooth_level_indicator`` so the
   126	    # PBL-depth threshold is differentiable.
   127	    pbl_weight = smooth_level_indicator(
   128	        z, threshold=config.cape_pbl_depth, sharpness=2.0e-3,
   129	        direction="below",
   130	    )                                                       # (ncol, nlev)
   131	    pbl_norm = jnp.sum(pbl_weight, axis=-1, keepdims=True).clip(1e-6, None)
   132	    T_pbl = jnp.sum(pbl_weight * T, axis=-1) / pbl_norm.squeeze(-1)
   133	    q_pbl = jnp.sum(pbl_weight * q_v, axis=-1) / pbl_norm.squeeze(-1)
   134	    if config.use_pbl_cape:
   135	        T_parcel_source = T_pbl
   136	        q_parcel_source = q_pbl
   137	    else:
   138	        T_parcel_source = T_base
   139	        q_parcel_source = q_base
   140	
   141	    T_parcel = T_parcel_source + config.parcel_dT
   142	    q_parcel = q_parcel_source + config.parcel_dq
   143	
   144	    T_moist = compute_moist_adiabat(T_parcel, p_full)
   145	    cape_pbl = compute_cape(T, T_moist, p_full, p_half)
   146	
   147	    cape_weight = cape_trigger(
   148	        cape_pbl, config.cape_threshold, config.cape_sharpness,
   149	    )
   150	
   151	    # -- LCL, LFC/LNB ------------------------------------------------------
   152	    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
   153	    k_lcl_smooth = lcl.k_lcl_smooth
   154	    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)
   155	
   156	    # Cloud depth.
   157	    levels_arr = jnp.arange(nlev, dtype=T.dtype)
   158	    weight_lcl = jax.nn.softmax(
   159	        -2.0 * (levels_arr[None, :] - k_lcl_smooth[:, None]) ** 2, axis=-1,
   160	    )
   161	    weight_lnb = jax.nn.softmax(
   162	        -2.0 * (levels_arr[None, :] - k_lnb_smooth[:, None]) ** 2, axis=-1,
   163	    )
   164	    z_lcl = jnp.sum(weight_lcl * z, axis=-1)
   165	    z_lnb = jnp.sum(weight_lnb * z, axis=-1)
   166	    cloud_depth = jnp.maximum(z_lnb - z_lcl, 0.0)
   167	
   168	    # -- Three-class blend -------------------------------------------------
   169	    deep_weight = smooth_step(
   170	        cloud_depth - config.cloud_depth_deep, config.depth_split_sharpness,
   171	    )
   172	    shallow_weight = smooth_step(
   173	        config.cloud_depth_shallow_max - cloud_depth,
   174	        config.depth_split_sharpness,
   175	    )
   176	    midlevel_weight = jnp.clip(
   177	        1.0 - deep_weight - shallow_weight, 0.0, 1.0
   178	    )
   179	
   180	    # -- PBL-CAPE closure for cloud-base mass flux -------------------------
   181	    # Bechtold 2008 / 2014 use a hybrid closure: PBL-CAPE drives the
   182	    # baseline mass flux, optionally enhanced where the column is
   183	    # moisture-convergent.  We add the (column-integrated) MC term as
   184	    # a multiplicative enhancement (1 + MC_normalized) so the closure
   185	    # gracefully reduces to pure PBL-CAPE when MC is unavailable
   186	    # (zero-filled by the bridge for spectral PE and other dycores
   187	    # without an MC diagnostic).
   188	    M_b_pbl_cape = (
   189	        cape_weight
   190	        * smooth_positive_part(cape_pbl - config.cape_threshold, config.cape_sharpness)
   191	        / config.tau_bl
   192	    )
   193	    if moisture_convergence is not None:
   194	        dp_full = p_half[:, 1:] - p_half[:, :-1]
   195	        column_MC = jnp.sum(
   196	            jnp.maximum(moisture_convergence, 0.0) * dp_full, axis=-1,
   197	        ) / constants.g
   198	        # Normalize the MC term so it acts as an O(1) multiplier.
   199	        # 0.05 kg/m^2/s is a typical strong-convergence value over
   200	        # tropical convective regions (Bechtold 2008 Fig. 2).
   201	        mc_enhancement = column_MC / 0.05
   202	        M_b_deterministic = M_b_pbl_cape * (1.0 + mc_enhancement)
   203	    else:
   204	        M_b_deterministic = M_b_pbl_cape
   205	
   206	    # -- AR1 stochastic perturbation ---------------------------------------
   207	    if config.enable_stochastic and prng_key is not None:
   208	        alpha_AR1 = jnp.exp(-dt / config.stochastic_decorrelation)
   209	        innovation = jax.random.normal(prng_key, shape=(ncol,), dtype=T.dtype)
   210	        conv_stoch_state_new = (
   211	            alpha_AR1 * conv_stoch_state
   212	            + jnp.sqrt(jnp.maximum(1.0 - alpha_AR1 ** 2, 0.0)) * innovation
   213	        )
   214	        stoch_factor = 1.0 + config.stochastic_amplitude * conv_stoch_state_new
   215	    else:
   216	        # Either stochastic disabled or no PRNG provided — preserve
   217	        # input AR1 state and use deterministic factor 1.
   218	        conv_stoch_state_new = conv_stoch_state
   219	        stoch_factor = jnp.ones_like(M_b_deterministic)
   220	
   221	    M_b = M_b_deterministic * jnp.maximum(stoch_factor, 0.0)
   222	    # See ZhangMcFarlaneConfig.M_b_max.
   223	    M_b = jnp.clip(M_b, 0.0, config.M_b_max)
   224	
   225	    # -- Per-class entrainment / detrainment profiles ----------------------
   226	    eps_per_class = (
   227	        deep_weight[:, None] * config.epsilon_deep
   228	        + shallow_weight[:, None] * config.epsilon_shallow
   229	        + midlevel_weight[:, None] * config.epsilon_midlevel
   230	    )
   231	    dlt_per_class = (
   232	        deep_weight[:, None] * config.delta_deep
   233	        + shallow_weight[:, None] * config.delta_shallow
   234	        + midlevel_weight[:, None] * config.delta_midlevel
   235	    )
   236	    eps_profile = jnp.broadcast_to(eps_per_class, T.shape)
   237	    dlt_profile = jnp.broadcast_to(dlt_per_class, T.shape)
   238	
   239	    plume = entraining_detraining_plume(
   240	        T, q_v, p_full, p_half, z,
   241	        T_parcel, q_parcel, k_lcl_smooth,
   242	        eps_profile, dlt_profile, M_b,
   243	    )
   244	
   245	    # -- Implicit-Euler relaxation of the M_u profile carry ---------------
   246	    dt_over_tau = dt / jnp.maximum(config.tau_M_u_relax, dt)
   247	    M_u_new = (conv_prog_profile + dt_over_tau * plume.M_u) / (1.0 + dt_over_tau)
   248	    # Cap M_u_new at config.M_b_max so every downstream use (kernel
   249	    # tendencies, dq_c_conv_raw, downdraft trigger, CMT, carry update)
   250	    # sees the same bounded value.
   251	    M_u_new = jnp.clip(M_u_new, 0.0, config.M_b_max)
   252	
   253	    # -- Environmental tendencies (using relaxed M_u) ---------------------
   254	    delta_0_eff = (
   255	        deep_weight * config.delta_deep
   256	        + shallow_weight * config.delta_shallow
   257	        + midlevel_weight * config.delta_midlevel
   258	    )
   259	    dT_dt_raw, dq_v_dt_raw, _ = _apply_mass_flux_kernel(
   260	        T, q_v, p_full,
   261	        plume.T_u, plume.q_u, plume.q_c_u, M_u_new,
   262	        z, rho, float(config.delta_deep), M_u_max=config.M_b_max,
   263	    )
   264	    rho_safe = jnp.clip(rho, 0.01, None)
   265	    p_gate_qc = stratosphere_mass_flux_gate(p_full)
   266	    dq_c_conv_dt_raw = (
   267	        delta_0_eff[:, None] * M_u_new * p_gate_qc * plume.q_c_u / rho_safe
   268	    )
   269	    rescale = delta_0_eff[:, None] / config.delta_deep
   270	    dT_dt = dT_dt_raw * rescale
   271	    dq_v_dt = dq_v_dt_raw * rescale
   272	    dq_c_conv_dt = dq_c_conv_dt_raw
   273	
   274	    # -- Optional downdraft (RH-dependent) ---------------------------------
   275	    if config.enable_downdraft:
   276	        below_lcl = jax.nn.sigmoid(
   277	            2.0 * (levels_arr[None, :] - k_lcl_smooth[:, None])
   278	        )
   279	        q_sat_env = saturation_mixing_ratio(T, p_full)
   280	        rh_layer = q_v / jnp.maximum(q_sat_env, 1e-12)
   281	        dp = p_half[:, 1:] - p_half[:, :-1]
   282	        below_mass = jnp.sum(below_lcl * dp, axis=-1) + 1e-6
   283	        rh_below = (
   284	            jnp.sum(below_lcl * rh_layer * dp, axis=-1) / below_mass
   285	        )
   286	        downdraft_trigger = jax.nn.sigmoid(
   287	            10.0 * (config.downdraft_RH_min - rh_below)
   288	        )
   289	        M_d_base = -config.downdraft_alpha * M_b * downdraft_trigger
   290	        below_lcl_norm = below_lcl / jnp.sum(below_lcl, axis=-1, keepdims=True).clip(1e-6, None)
   291	        dT_dt_dd = -(constants.L_v / constants.c_pd) * (
   292	            jnp.abs(M_d_base[:, None]) * below_lcl_norm * 0.05
   293	        ) / rho_safe
   294	        dT_dt = dT_dt + dT_dt_dd
   295	
   296	    # -- CMT --------------------------------------------------------------
   297	    if config.enable_cmt:
   298	        if config.enable_downdraft:
   299	            M_d = -config.downdraft_alpha * M_u_new * 0.3
   300	        else:
   301	            M_d = None
   302	        du_dt_conv, dv_dt_conv = cmt_gregory_1997(
   303	            u, v, M_u_new, M_d,
   304	            p_full, p_half, rho,
   305	            c_u=config.cmt_c_u, c_d=config.cmt_c_d,
   306	        )
   307	    else:
   308	        du_dt_conv = None
   309	        dv_dt_conv = None
   310	
   311	    convective_mask = cape_weight * (deep_weight + shallow_weight + midlevel_weight)
   312	
   313	    out = ConvectionOutput(
   314	        dT_dt=dT_dt,
   315	        dq_v_dt=dq_v_dt,
   316	        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
   317	        cape=cape_pbl,
   318	        convective_mask=convective_mask,
   319	        du_dt_conv=du_dt_conv,
   320	        dv_dt_conv=dv_dt_conv,
   321	    )
   322	    return out, M_u_new, conv_stoch_state_new

 succeeded in 0ms:
     1	"""Tiedtke (1989) bulk mass-flux convection.
     2	
     3	The IFS heritage scheme — three-class soft assignment of cloud type
     4	(deep / mid-level / shallow) blended on cloud depth, downdraft with
     5	RH-dependent trigger, and convective momentum transport via Gregory
     6	et al. 1997.  This is the **first scheme that exercises the
     7	``conv_prog_profile = M_u(k)`` profile carry**: the diagnosed
     8	per-level updraft mass flux relaxes via implicit Euler toward the
     9	new diagnosis on every step, smoothing fast oscillations.
    10	
    11	Smooth-everywhere implementation:
    12	
    13	* Three-class blend on cloud depth — sigmoid weights, not hard
    14	  threshold.
    15	* Downdraft RH trigger — sigmoid on (downdraft_RH_min - column_RH).
    16	* CAPE gate — sigmoid via :func:`._triggers.cape_trigger`.
    17	* Moisture-convergence proxy — saturation-deficit
    18	  ``MC_proxy = max(q_sat - q_v, 0) / tau_MC_proxy`` (a placeholder
    19	  until the PR-0 ``compute_moisture_convergence`` diagnostic ships).
    20	* Plume integrator's mass-flux profile is gated by buoyancy sigmoid.
    21	
    22	References
    23	----------
    24	* Tiedtke, M. (1989). A comprehensive mass flux scheme for cumulus
    25	  parameterization in large-scale models.  *Mon. Wea. Rev.*, 117,
    26	  1779–1800.
    27	* Gregory, D., et al. (1997). Parametrization of momentum transport
    28	  by convection. II.  *Quart. J. Roy. Meteor. Soc.*, 123, 1153–1183.
    29	"""
    30	
    31	from __future__ import annotations
    32	
    33	import jax
    34	import jax.numpy as jnp
    35	
    36	from legoesm import constants
    37	from legoesm.thermo import saturation_mixing_ratio
    38	from legoesm.atmosphere.physics.thermodynamics import (
    39	    compute_cape,
    40	    compute_moist_adiabat,
    41	)
    42	from legoesm.atmosphere.physics.convection.config import TiedtkeConfig
    43	from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    44	from legoesm.atmosphere.physics.convection.mass_flux import (
    45	    _apply_mass_flux_kernel,
    46	    stratosphere_mass_flux_gate,
    47	    _compute_column_geometry,
    48	)
    49	from legoesm.atmosphere.physics.convection._triggers import (
    50	    cape_trigger,
    51	    smooth_positive_part,
    52	    smooth_step,
    53	)
    54	from legoesm.atmosphere.physics.convection._plume import (
    55	    cmt_gregory_1997,
    56	    compute_lcl,
    57	    compute_lfc_lnb,
    58	    entraining_detraining_plume,
    59	)
    60	
    61	
    62	__all__ = ("tiedtke_convection",)
    63	
    64	
    65	def tiedtke_convection(
    66	    T: jax.Array,
    67	    q_v: jax.Array,
    68	    p_full: jax.Array,
    69	    p_half: jax.Array,
    70	    u: jax.Array,
    71	    v: jax.Array,
    72	    conv_prog_profile: jax.Array,
    73	    dt: float,
    74	    config: TiedtkeConfig = TiedtkeConfig(),
    75	    moisture_convergence: jax.Array | None = None,
    76	) -> tuple[ConvectionOutput, jax.Array]:
    77	    """Tiedtke (1989) convection (smooth, differentiable).
    78	
    79	    Parameters
    80	    ----------
    81	    T : jax.Array, shape (ncol, nlev)
    82	        Environmental temperature [K].
    83	    q_v : jax.Array, shape (ncol, nlev)
    84	        Water-vapor specific humidity [kg/kg].
    85	    p_full, p_half : jax.Array
    86	        Full / half-level pressures [Pa].
    87	    u, v : jax.Array, shape (ncol, nlev)
    88	        Environmental wind components [m/s] for the CMT closure.
    89	    conv_prog_profile : jax.Array, shape (ncol, nlev)
    90	        Updraft mass-flux profile from the previous time step.
    91	        Tiedtke is the first scheme that uses this as a real per-level
    92	        carry — relaxation is implicit Euler toward
    93	        ``M_u_diagnosed`` over ``tau_M_u_relax`` seconds.
    94	    dt : float
    95	        Time step [s].
    96	    config : TiedtkeConfig
    97	        Scheme tunables.
    98	
    99	    Returns
   100	    -------
   101	    out : ConvectionOutput
   102	        Tendencies on T, q_v, q_c plus CAPE diagnostic.  CMT
   103	        ``du_dt_conv`` / ``dv_dt_conv`` populated when
   104	        ``config.enable_cmt`` is ``True``.
   105	    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
   106	        Implicit-Euler-relaxed ``M_u(k)`` profile.
   107	    """
   108	    ncol, nlev = T.shape
   109	
   110	    # -- Column geometry, moist adiabat, CAPE ------------------------------
   111	    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
   112	    T_base = T[:, -1]
   113	    q_base = q_v[:, -1]
   114	    p_base = p_full[:, -1]
   115	    T_moist = compute_moist_adiabat(T_base, p_full)
   116	    cape = compute_cape(T, T_moist, p_full, p_half)
   117	    cape_weight = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)
   118	
   119	    # -- LCL, LFC, LNB diagnostics -----------------------------------------
   120	    T_parcel = T_base + config.parcel_dT
   121	    q_parcel = q_base + config.parcel_dq
   122	    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
   123	    k_lcl_smooth = lcl.k_lcl_smooth
   124	    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)
   125	
   126	    # Cloud depth: smooth interpolation of z at fractional indices.
   127	    levels = jnp.arange(nlev, dtype=T.dtype)
   128	    weight_lcl = jax.nn.softmax(
   129	        -2.0 * (levels[None, :] - k_lcl_smooth[:, None]) ** 2, axis=-1,
   130	    )
   131	    weight_lnb = jax.nn.softmax(
   132	        -2.0 * (levels[None, :] - k_lnb_smooth[:, None]) ** 2, axis=-1,
   133	    )
   134	    z_lcl = jnp.sum(weight_lcl * z, axis=-1)
   135	    z_lnb = jnp.sum(weight_lnb * z, axis=-1)
   136	    cloud_depth = jnp.maximum(z_lnb - z_lcl, 0.0)
   137	
   138	    # -- Three-class soft assignment ---------------------------------------
   139	    # deep_weight rises with cloud depth; shallow_weight falls with it.
   140	    # mid-level fills the gap.
   141	    deep_weight = smooth_step(
   142	        cloud_depth - config.cloud_depth_deep, config.depth_split_sharpness,
   143	    )
   144	    shallow_weight = smooth_step(
   145	        config.cloud_depth_shallow_max - cloud_depth,
   146	        config.depth_split_sharpness,
   147	    )
   148	    midlevel_weight = jnp.clip(
   149	        1.0 - deep_weight - shallow_weight, 0.0, 1.0
   150	    )
   151	
   152	    # -- Per-class entrainment / detrainment profiles ----------------------
   153	    eps_per_class = (
   154	        deep_weight[:, None] * config.epsilon_deep
   155	        + shallow_weight[:, None] * config.epsilon_shallow
   156	        + midlevel_weight[:, None] * config.epsilon_midlevel
   157	    )
   158	    dlt_per_class = (
   159	        deep_weight[:, None] * config.delta_deep
   160	        + shallow_weight[:, None] * config.delta_shallow
   161	        + midlevel_weight[:, None] * config.delta_midlevel
   162	    )
   163	    # Broadcast to (ncol, nlev) — entrainment is constant over the
   164	    # column for a given class blend.
   165	    eps_profile = jnp.broadcast_to(eps_per_class, T.shape)
   166	    dlt_profile = jnp.broadcast_to(dlt_per_class, T.shape)
   167	
   168	    # -- Closure: deep uses moisture convergence; shallow and midlevel
   169	    # use a CAPE-relaxation closure.  Combined per-column closure is a
   170	    # class-weighted blend.  When the bridge supplies a real
   171	    # ``moisture_convergence`` array (from
   172	    # ``_shared.compute_moisture_convergence``, available for cubed-
   173	    # sphere and lat-lon dycores), use it directly; otherwise fall
   174	    # back to a saturation-deficit proxy that is qualitatively similar
   175	    # (positive in moist columns, vanishing in dry ones).
   176	    q_sat_env = saturation_mixing_ratio(T, p_full)
   177	    dp = p_half[:, 1:] - p_half[:, :-1]
   178	    if moisture_convergence is not None:
   179	        # Real MC — column-integrate per (kg/m^2/s).
   180	        column_MC = (
   181	            jnp.sum(moisture_convergence * dp, axis=-1) / constants.g
   182	        )
   183	        column_MC_proxy = jnp.maximum(column_MC, 0.0)
   184	    else:
   185	        sat_deficit = jnp.maximum(q_sat_env - q_v, 0.0)
   186	        column_MC_proxy = (
   187	            jnp.sum(sat_deficit * dp, axis=-1)
   188	            / (constants.g * config.tau_MC_proxy)
   189	        )
   190	    # Smooth gate on MC threshold for deep.
   191	    mc_gate = smooth_positive_part(
   192	        column_MC_proxy - config.moisture_convergence_threshold,
   193	        config.moisture_convergence_sharpness,
   194	    )
   195	
   196	    # Cloud-base mass flux (per class, then blended).
   197	    M_b_deep = mc_gate * cape_weight
   198	    M_b_shallow = (
   199	        cape_weight
   200	        * smooth_positive_part(cape - config.cape_threshold, config.cape_sharpness)
   201	        / config.tau_shallow_M_b
   202	    )
   203	    M_b_midlevel = M_b_shallow * config.midlevel_M_b_fraction
   204	    M_b = (
   205	        deep_weight * M_b_deep
   206	        + shallow_weight * M_b_shallow
   207	        + midlevel_weight * M_b_midlevel
   208	    )
   209	    # See ZhangMcFarlaneConfig.M_b_max.
   210	    M_b = jnp.clip(M_b, 0.0, config.M_b_max)
   211	
   212	    # -- Plume integration -------------------------------------------------
   213	    plume = entraining_detraining_plume(
   214	        T, q_v, p_full, p_half, z,
   215	        T_parcel, q_parcel, k_lcl_smooth,
   216	        eps_profile, dlt_profile, M_b,
   217	    )
   218	
   219	    # -- Implicit-Euler relaxation of the M_u profile carry ---------------
   220	    dt_over_tau = dt / jnp.maximum(config.tau_M_u_relax, dt)
   221	    M_u_new = (conv_prog_profile + dt_over_tau * plume.M_u) / (1.0 + dt_over_tau)
   222	    # Cap M_u_new at config.M_b_max so every downstream use (kernel
   223	    # tendencies, dq_c_conv_raw, downdraft trigger, CMT, carry update)
   224	    # sees the same bounded value.
   225	    M_u_new = jnp.clip(M_u_new, 0.0, config.M_b_max)
   226	
   227	    # Use the relaxed M_u for the actual environmental tendencies — this
   228	    # smooths the time evolution of the convective forcing.
   229	    M_u_for_kernel = M_u_new
   230	
   231	    # -- Environmental tendencies ----------------------------------------
   232	    # Build an effective per-class delta_0 for the kernel and the
   233	    # cloud-water source.
   234	    delta_0_eff = (
   235	        deep_weight * config.delta_deep
   236	        + shallow_weight * config.delta_shallow
   237	        + midlevel_weight * config.delta_midlevel
   238	    )
   239	    # Kernel signature now takes (q_v_u, q_c_u) explicitly.  We still
   240	    # call with a single nominal ``delta_deep`` and rescale below by
   241	    # the per-column class blend to keep the existing branch-blending
   242	    # behaviour bit-identical.
   243	    dT_dt_raw, dq_v_dt_raw, _ = _apply_mass_flux_kernel(
   244	        T, q_v, p_full,
   245	        plume.T_u, plume.q_u, plume.q_c_u, M_u_for_kernel,
   246	        z, rho, float(config.delta_deep), M_u_max=config.M_b_max,
   247	    )
   248	    rho_safe = jnp.clip(rho, 0.01, None)
   249	    # Reuse the same stratospheric gate the kernel applies so this
   250	    # custom q_c path does not detrain condensate above the tropopause.
   251	    p_gate_qc = stratosphere_mass_flux_gate(p_full)
   252	    dq_c_conv_dt_raw = (
   253	        delta_0_eff[:, None] * M_u_for_kernel * p_gate_qc * plume.q_c_u / rho_safe
   254	    )
   255	    # The kernel's dT/dq computations used ``config.delta_deep`` as the
   256	    # detrainment scale; rescale by the per-column class blend.
   257	    rescale = delta_0_eff[:, None] / config.delta_deep
   258	    dT_dt = dT_dt_raw * rescale
   259	    dq_v_dt = dq_v_dt_raw * rescale
   260	    dq_c_conv_dt = dq_c_conv_dt_raw
   261	
   262	    # -- Optional downdraft (RH-dependent trigger) -------------------------
   263	    if config.enable_downdraft:
   264	        # Column-mean RH below LCL.
   265	        levels = jnp.arange(nlev, dtype=T.dtype)
   266	        below_lcl = jax.nn.sigmoid(
   267	            2.0 * (levels[None, :] - k_lcl_smooth[:, None])
   268	        )
   269	        rh_layer = q_v / jnp.maximum(q_sat_env, 1e-12)
   270	        below_mass = jnp.sum(below_lcl * dp, axis=-1) + 1e-6
   271	        rh_below = (
   272	            jnp.sum(below_lcl * rh_layer * dp, axis=-1) / below_mass
   273	        )
   274	        downdraft_trigger = jax.nn.sigmoid(
   275	            10.0 * (config.downdraft_RH_min - rh_below)
   276	        )
   277	        # Downdraft mass flux = -alpha * M_b at cloud base.
   278	        M_d_base = -config.downdraft_alpha * M_b * downdraft_trigger
   279	        # Distribute uniformly below cloud base.
   280	        below_lcl_norm = below_lcl / jnp.sum(below_lcl, axis=-1, keepdims=True).clip(1e-6, None)
   281	        # Downdraft cools by entraining colder layers above and
   282	        # bringing them down — net cooling tendency on environment.
   283	        dT_dt_dd = -(constants.L_v / constants.c_pd) * (
   284	            jnp.abs(M_d_base[:, None]) * below_lcl_norm
   285	            * jnp.maximum(0.05, 0.0)  # rough evap rate proxy [kg/kg]
   286	        ) / rho_safe
   287	        dT_dt = dT_dt + dT_dt_dd
   288	
   289	    # -- CMT --------------------------------------------------------------
   290	    if config.enable_cmt:
   291	        if config.enable_downdraft:
   292	            M_d = -config.downdraft_alpha * M_u_for_kernel * 0.3
   293	        else:
   294	            M_d = None
   295	        du_dt_conv, dv_dt_conv = cmt_gregory_1997(
   296	            u, v, M_u_for_kernel, M_d,
   297	            p_full, p_half, rho,
   298	            c_u=config.cmt_c_u, c_d=config.cmt_c_d,
   299	        )
   300	    else:
   301	        du_dt_conv = None
   302	        dv_dt_conv = None
   303	
   304	    # -- Convective mask ---------------------------------------------------
   305	    convective_mask = cape_weight * (deep_weight + shallow_weight + midlevel_weight)
   306	
   307	    out = ConvectionOutput(
   308	        dT_dt=dT_dt,
   309	        dq_v_dt=dq_v_dt,
   310	        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
   311	        cape=cape,
   312	        convective_mask=convective_mask,
   313	        du_dt_conv=du_dt_conv,
   314	        dv_dt_conv=dv_dt_conv,
   315	    )
   316	    return out, M_u_new

 succeeded in 0ms:
     1	"""Configuration for atmospheric convection schemes.
     2	
     3	Provides configuration NamedTuples for:
     4	1. Simplified Betts-Miller (SBM) — relaxation-based convection (Frierson 2007)
     5	2. Deep Convective Adjustment (DCA) — simplest baseline adjustment
     6	3. Kuo — column moisture-excess (Kuo 1965/1974)
     7	4. Prognostic Mass-Flux — Arakawa-Wu type (1 prognostic var: M_c)
     8	5. Simplified EDMF — eddy-diffusivity mass-flux (1 prognostic var: a_u)
     9	6. Top-level ConvectionConfig that selects the active scheme.
    10	
    11	References
    12	----------
    13	- Frierson, D. M. W. (2007). The Dynamics of Idealized Convection
    14	  Schemes and Their Effect on the Zonally Averaged Tropical Circulation.
    15	  J. Atmos. Sci., 64, 1959-1976.
    16	- Kuo, H. L. (1974). Further studies of the parameterization of the
    17	  influence of cumulus convection on large-scale flow. J. Atmos. Sci.,
    18	  31, 1232-1240.
    19	- Arakawa, A., & Wu, C.-M. (2013). A unified representation of deep
    20	  moist convection in numerical modeling of the atmosphere. Part I.
    21	  J. Atmos. Sci., 70, 1977-1992.
    22	"""
    23	
    24	from __future__ import annotations
    25	
    26	from typing import NamedTuple
    27	
    28	
    29	class SBMConfig(NamedTuple):
    30	    """Configuration for Simplified Betts-Miller convection.
    31	
    32	    Fields
    33	    ------
    34	    tau_c : float
    35	        Relaxation timescale [s] (default 7200 = 2 hours).
    36	    RH_ref : float
    37	        Reference relative humidity for moisture profile (default 0.7).
    38	    CAPE_threshold : float
    39	        Minimum CAPE [J/kg] to trigger convection (default 70.0).
    40	    T_min_convect : float
    41	        Minimum temperature [K] for convection (default 200.0).
    42	    smooth_trigger_sharpness : float
    43	        Sigmoid sharpness for smooth trigger [1/(J/kg)] (default 0.01).
    44	    """
    45	    tau_c: float = 7200.0
    46	    RH_ref: float = 0.7
    47	    CAPE_threshold: float = 70.0
    48	    T_min_convect: float = 200.0
    49	    # Default 0.1 (was 0.01).  At CAPE=0 the looser 0.01 gives
    50	    # ``sigmoid(0.01·-70) ≈ 0.33`` (33 % activation when CAPE is zero
    51	    # — gating leaks).  0.1 gives ``sigmoid(-7) ≈ 9e-4`` (effectively 0)
    52	    # while preserving smoothness near the threshold.
    53	    smooth_trigger_sharpness: float = 0.1
    54	
    55	
    56	class DCAConfig(NamedTuple):
    57	    """Configuration for Deep Convective Adjustment.
    58	
    59	    Fields
    60	    ------
    61	    n_iterations : int
    62	        Number of adjustment iterations per call (default 1).
    63	    mixing_fraction : float
    64	        Fraction of adjustment applied per iteration (default 1.0).
    65	    cape_threshold : float
    66	        Minimum CAPE [J/kg] to trigger convection (default 100.0).
    67	        Columns with CAPE below this are not adjusted.
    68	    cape_sharpness : float
    69	        Sigmoid sharpness [1/(J/kg)] for smooth CAPE gating (default 0.02).
    70	    """
    71	    n_iterations: int = 1
    72	    mixing_fraction: float = 1.0
    73	    cape_threshold: float = 100.0
    74	    cape_sharpness: float = 0.1   # sigmoid(-10)≈5e-5 at CAPE=0; 0.5 at threshold
    75	
    76	
    77	class KuoConfig(NamedTuple):
    78	    """Configuration for Kuo column moisture-excess convection.
    79	
    80	    Fields
    81	    ------
    82	    alpha_heat : float
    83	        Fraction of column moisture excess going to heating vs moistening.
    84	    me_threshold : float
    85	        Minimum column moisture excess to trigger convection [kg/m^2].
    86	    smooth_trigger_sharpness : float
    87	        Sigmoid sharpness on column moisture excess trigger [1/(kg/m^2)].
    88	    tau_relax : float
    89	        Relaxation timescale [s].  Default 7200 (2 h).  The earlier
    90	        default 3600 (1 h) ate the entire column moisture excess every
    91	        hour, which combined with the surface-evap supply rate gave
    92	        ~10× too much precipitation in tropical RCE.  CCM2/CCM3 used
    93	        21600 (6 h); 7200 is a compromise that keeps the scheme
    94	        responsive to real precipitating columns without
    95	        over-precipitating.
    96	    """
    97	    alpha_heat: float = 0.75
    98	    me_threshold: float = 1e-5
    99	    smooth_trigger_sharpness: float = 1e4
   100	    tau_relax: float = 7200.0
   101	
   102	
   103	class MassFluxConfig(NamedTuple):
   104	    """Configuration for Prognostic Mass-Flux convection (Arakawa-Wu type).
   105	
   106	    Fields
   107	    ------
   108	    tau_adj : float
   109	        Mass flux relaxation timescale [s].
   110	    epsilon_0 : float
   111	        Entrainment rate [1/m].
   112	    delta_0 : float
   113	        Detrainment rate [1/m].
   114	    M_scale : float
   115	        Equilibrium mass flux scale [kg/m^2/s].
   116	    cape_activation_scale : float
   117	        Sigmoid scale for CAPE trigger [J/kg].
   118	    cape_threshold : float
   119	        CAPE threshold [J/kg].
   120	    M_c_init : float
   121	        Initial base mass flux [kg/m^2/s].
   122	    """
   123	    tau_adj: float = 3600.0
   124	    epsilon_0: float = 1e-3
   125	    delta_0: float = 1e-3
   126	    M_scale: float = 0.01
   127	    # Trigger: ``convective_mask = sigmoid((cape − cape_threshold)
   128	    # / cape_activation_scale)``.  Defaults below give ``sigmoid(-7)
   129	    # ≈ 9e-4`` at CAPE=0 (effectively no firing) and full activation
   130	    # 100 J/kg above the 70 J/kg threshold.  The earlier defaults
   131	    # (``cape_threshold=0``, ``cape_activation_scale=100``) gave 50 %
   132	    # activation at CAPE=0 — the gating-leak that the validation
   133	    # script flags.
   134	    cape_activation_scale: float = 10.0
   135	    cape_threshold: float = 70.0
   136	    M_c_init: float = 0.0
   137	    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
   138	
   139	
   140	class ZhangMcFarlaneConfig(NamedTuple):
   141	    """Configuration for the Zhang & McFarlane (1995) deep-convection scheme.
   142	
   143	    Single-plume mass-flux scheme with CAPE-relaxation closure and
   144	    optional Gregory et al. 1997 convective momentum transport.
   145	
   146	    Fields
   147	    ------
   148	    tau_cape : float
   149	        CAPE relaxation timescale [s] (default 3600 = 1 hour).
   150	    cape_threshold : float
   151	        CAPE threshold [J/kg] below which convection is suppressed
   152	        (default 70.0, the classic ZM 1995 value).
   153	    cape_sharpness : float
   154	        Sigmoid sharpness on the CAPE trigger [1/(J/kg)].  Default
   155	        ``0.02`` gives ~95% activation 100 J/kg above threshold.
   156	    parcel_dT : float
   157	        Sub-cloud parcel temperature perturbation [K] (default 0.5).
   158	    parcel_dq : float
   159	        Sub-cloud parcel humidity perturbation [kg/kg] (default 1e-3).
   160	    epsilon_0 : float
   161	        Bulk-plume entrainment rate [1/m] (default 1e-3).
   162	    delta_0 : float
   163	        Bulk-plume detrainment rate [1/m] (default 1e-3).
   164	    enable_cmt : bool
   165	        Whether to compute convective momentum transport tendencies
   166	        (default ``True``).  When ``False``, the bridge zero-fills
   167	        ``du_dt_conv`` / ``dv_dt_conv``.
   168	    cmt_c_u, cmt_c_d : float
   169	        Pressure-gradient correction coefficients in the Gregory et
   170	        al. 1997 closure (default 0.55 each — the canonical value).
   171	    M_b_max : float
   172	        Hard upper bound on the cloud-base mass flux ``M_b`` [kg/m²/s]
   173	        (default 0.005 — about 1/20 of the literature peak tropical value 0.1; tighter than peak because the unbounded CAPE/tau closure can spike to ~2 kg/m²/s in a high-CAPE column and the per-layer heating ~M·(T_u−T)·δ scales linearly).  The
   174	        CAPE/τ_cape closure is unbounded above; without this cap a
   175	        column with CAPE >> 5 kJ/kg yields M_b that drives
   176	        column-integrated heating > 10⁴ W/m² and blows up the
   177	        integration on the next dynamics step.
   178	    """
   179	    tau_cape: float = 3600.0
   180	    cape_threshold: float = 70.0
   181	    cape_sharpness: float = 0.1
   182	    parcel_dT: float = 0.5
   183	    parcel_dq: float = 1.0e-3
   184	    epsilon_0: float = 1.0e-3
   185	    delta_0: float = 1.0e-3
   186	    enable_cmt: bool = True
   187	    cmt_c_u: float = 0.55
   188	    cmt_c_d: float = 0.55
   189	    M_b_max: float = 0.05
   190	
   191	
   192	class KainFritschConfig(NamedTuple):
   193	    """Configuration for the Kain & Fritsch (1990, 2004 update) scheme.
   194	
   195	    Single-plume bulk mass-flux scheme distinguished by its
   196	    boundary-layer trigger function: convection fires when the
   197	    perturbed parcel temperature at the LCL exceeds the environmental
   198	    temperature at the LCL.  The trigger is smoothed via a sigmoid
   199	    (``trigger_sharpness``) to preserve gradients.  Deep-vs-shallow
   200	    cloud branches are blended on cloud depth.  No convective
   201	    momentum transport — KF emits ``du_dt_conv = dv_dt_conv = None``.
   202	
   203	    Fields
   204	    ------
   205	    w_thresh_offset : float
   206	        Trigger offset [K] (default 2.0; the canonical KF 1990 value).
   207	    w_thresh_scale : float
   208	        Conversion factor from ``w_grid`` [m/s] to a temperature
   209	        perturbation [K] in the trigger function.  Default 1.0 K per
   210	        m/s — the dimensionful scaling depends on resolution; users
   211	        with grid-scale ``w`` available should tune this.
   212	    trigger_sharpness : float
   213	        Sigmoid sharpness on the trigger threshold [1/K].  Larger
   214	        values approach a hard ``> 0`` step; smaller values broaden
   215	        the transition.  Default 5.0 → ~95% activation 0.6 K above
   216	        threshold.
   217	    cape_consumption_time : float
   218	        CAPE-removal timescale [s] (default 1800.0).
   219	    parcel_perturb_T : float
   220	        Sub-cloud parcel temperature perturbation [K] (default 0.5).
   221	    parcel_perturb_q : float
   222	        Sub-cloud parcel humidity perturbation [kg/kg] (default 1e-3).
   223	    epsilon_0 : float
   224	        Bulk-plume entrainment rate [1/m] (default 2e-3).
   225	    delta_0 : float
   226	        Bulk-plume detrainment rate [1/m] (default 2e-3).
   227	    cloud_depth_min : float
   228	        Cloud-depth threshold [m] separating shallow and deep
   229	        branches (default 4000.0).
   230	    cloud_depth_sharpness : float
   231	        Sigmoid sharpness on the deep/shallow blend [1/m] (default
   232	        1e-3 — ~95% activation 1500 m above threshold).
   233	    enable_shallow : bool
   234	        Whether to include the shallow-cloud branch.  ``False``
   235	        disables shallow tendencies regardless of cloud depth
   236	        (default ``True``).
   237	    cape_threshold : float
   238	        CAPE threshold [J/kg] below which convection is gated off.
   239	        Default 0.0 — KF gates primarily on the trigger function,
   240	        not on CAPE.
   241	    cape_sharpness : float
   242	        Sigmoid sharpness on the CAPE gate [1/(J/kg)] (default 0.02).
   243	    M_b_max : float
   244	        Hard upper bound on the cloud-base mass flux ``M_b`` [kg/m²/s]
   245	        (default 0.005 — about 1/20 of the literature peak tropical value 0.1; tighter than peak because the unbounded CAPE/tau closure can spike to ~2 kg/m²/s in a high-CAPE column and the per-layer heating ~M·(T_u−T)·δ scales linearly).
   246	    """
   247	    w_thresh_offset: float = 2.0
   248	    w_thresh_scale: float = 1.0
   249	    trigger_sharpness: float = 5.0
   250	    cape_consumption_time: float = 1800.0
   251	    parcel_perturb_T: float = 0.5
   252	    parcel_perturb_q: float = 1.0e-3
   253	    epsilon_0: float = 2.0e-3
   254	    delta_0: float = 2.0e-3
   255	    cloud_depth_min: float = 4000.0
   256	    cloud_depth_sharpness: float = 1.0e-3
   257	    enable_shallow: bool = True
   258	    cape_threshold: float = 0.0
   259	    cape_sharpness: float = 0.1
   260	    M_b_max: float = 0.05
   261	
   262	
   263	class EmanuelConfig(NamedTuple):
   264	    """Configuration for the Emanuel (1991) buoyancy-sorting scheme.
   265	
   266	    Single-plume mass-flux scheme with a buoyancy-sorted ensemble of
   267	    mixed parcels: at every cloud level the parcel may mix with
   268	    environmental air in a discrete spectrum of mixing fractions; the
   269	    fractions with positive buoyancy continue to ascend while the
   270	    fractions with negative buoyancy descend.  The smooth-everywhere
   271	    formulation replaces the hard ascend/descend switch with a
   272	    sigmoid weighting on buoyancy.
   273	
   274	    Distinct from Zhang-McFarlane (single bulk plume) and Kain-Fritsch
   275	    (single plume with deep/shallow blend) by the per-level
   276	    distribution of detrainment that the buoyancy-sorted ensemble
   277	    produces.
   278	
   279	    Fields
   280	    ------
   281	    n_mixing_fractions : int
   282	        Number of discrete mixing fractions in the buoyancy-sort
   283	        ensemble.  Default 8 — Emanuel 1991 uses 50; the smaller
   284	        value here is a cost / accuracy compromise.
   285	    cu_coefficient : float
   286	        Entrainment scale factor (Emanuel's α).  Default 0.7.
   287	    precip_efficiency_water : float
   288	        Precipitation efficiency above LCL (default 1.0).
   289	    precip_efficiency_lcl : float
   290	        Precipitation efficiency below LCL (default 0.0).
   291	    precip_threshold_qc : float
   292	        Cloud-water threshold above which precipitation falls
   293	        [kg/kg] (default 1e-3).
   294	    cape_threshold : float
   295	        CAPE gate [J/kg] (default 70.0).
   296	    cape_sharpness : float
   297	        Sigmoid sharpness on the CAPE gate [1/(J/kg)] (default 0.02).
   298	    parcel_perturb_T : float
   299	        Sub-cloud parcel temperature perturbation [K] (default 0.5).
   300	    parcel_perturb_q : float
   301	        Sub-cloud parcel humidity perturbation [kg/kg] (default 1e-3).
   302	    sub_cloud_relaxation : float
   303	        Sub-cloud layer mixing timescale [s] (default 100.0).
   304	    enable_unsaturated_downdraft : bool
   305	        Whether to include the unsaturated downdraft branch (rain
   306	        evaporation cooling) (default ``True``).
   307	    downdraft_efficiency : float
   308	        Fraction of precipitation that re-evaporates below cloud base
   309	        in the downdraft (default 0.2).
   310	    smooth_trigger_sharpness : float
   311	        Sigmoid sharpness on the buoyancy-sort weighting [1/K]
   312	        (default 0.5).
   313	    epsilon_0 : float
   314	        Bulk-plume entrainment rate [1/m] (default 1.5e-3).
   315	    delta_0 : float
   316	        Bulk-plume detrainment rate [1/m] (default 1.5e-3).
   317	    M_b_max : float
   318	        Hard upper bound on the cloud-base mass flux ``M_b`` [kg/m²/s]
   319	        (default 0.005 — about 1/20 of the literature peak tropical value 0.1; tighter than peak because the unbounded CAPE/tau closure can spike to ~2 kg/m²/s in a high-CAPE column and the per-layer heating ~M·(T_u−T)·δ scales linearly).
   320	    """
   321	    n_mixing_fractions: int = 8
   322	    cu_coefficient: float = 0.7
   323	    precip_efficiency_water: float = 1.0
   324	    precip_efficiency_lcl: float = 0.0
   325	    precip_threshold_qc: float = 1.0e-3
   326	    cape_threshold: float = 70.0
   327	    cape_sharpness: float = 0.1
   328	    parcel_perturb_T: float = 0.5
   329	    parcel_perturb_q: float = 1.0e-3
   330	    # Emanuel 1991 §3 uses a sub-cloud-layer mixing timescale of
   331	    # several thousand seconds.  The earlier default of 100 s gave
   332	    # M_b ~72× larger than published values and produced 28 MW/m²
   333	    # of column heating from a CAPE-positive sounding.
   334	    sub_cloud_relaxation: float = 7200.0
   335	    # Default-OFF.  Emanuel 1991's downdraft re-evaporates a fraction
   336	    # of *precipitation* (rain) back to vapor in the BL.  In a model
   337	    # without an explicit q_r tracer the implementation can only draw
   338	    # from ``dq_c_conv_dt`` (the cloud-water source), so enabling it
   339	    # produces a column-net moistening on CAPE-positive soundings —
   340	    # the wrong sign of ``Q_v`` that the validation script flags.
   341	    # Production runs with a full microphysics chain that owns q_r
   342	    # should override this to ``True``.
   343	    enable_unsaturated_downdraft: bool = False
   344	    downdraft_efficiency: float = 0.2
   345	    smooth_trigger_sharpness: float = 0.5
   346	    epsilon_0: float = 1.5e-3
   347	    delta_0: float = 1.5e-3
   348	    M_b_max: float = 0.05
   349	
   350	
   351	class TiedtkeConfig(NamedTuple):
   352	    """Configuration for the Tiedtke (1989) bulk mass-flux scheme.
   353	
   354	    Three-class scheme with deep, mid-level, and shallow branches
   355	    blended on cloud depth.  Downdraft is included with an RH-based
   356	    trigger; convective momentum transport via Gregory et al. 1997.
   357	    First scheme that actually exercises the new ``(ncol, nlev)``
   358	    profile carry for ``M_u(k)``.
   359	
   360	    The full Tiedtke 1989 closure uses column moisture convergence

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/config.py | sed -n '351,520p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   351	class TiedtkeConfig(NamedTuple):
   352	    """Configuration for the Tiedtke (1989) bulk mass-flux scheme.
   353	
   354	    Three-class scheme with deep, mid-level, and shallow branches
   355	    blended on cloud depth.  Downdraft is included with an RH-based
   356	    trigger; convective momentum transport via Gregory et al. 1997.
   357	    First scheme that actually exercises the new ``(ncol, nlev)``
   358	    profile carry for ``M_u(k)``.
   359	
   360	    The full Tiedtke 1989 closure uses column moisture convergence
   361	    for the deep branch.  Until the PR-0 ``compute_moisture_convergence``
   362	    diagnostic ships, we use a saturation-deficit proxy
   363	    ``MC_proxy = (q_sat - q_v) / tau_relax`` that has the same
   364	    qualitative behavior (positive in moist columns, zero in dry
   365	    columns).
   366	
   367	    Fields
   368	    ------
   369	    epsilon_deep, delta_deep : float
   370	        Entrainment / detrainment rates for deep branch [1/m].
   371	    epsilon_shallow, delta_shallow : float
   372	        Same for shallow branch.
   373	    epsilon_midlevel, delta_midlevel : float
   374	        Same for mid-level branch.
   375	    enable_downdraft : bool
   376	        Whether to include the downdraft branch (default ``True``).
   377	    downdraft_alpha : float
   378	        Downdraft / updraft mass flux ratio at LFS (default 0.3).
   379	    downdraft_RH_min : float
   380	        Below this column-mean RH the downdraft fires (default 0.2).
   381	    moisture_convergence_threshold : float
   382	        Saturation-deficit proxy threshold [kg/kg/s].  Below this the
   383	        deep branch is suppressed (default 1e-8).
   384	    moisture_convergence_sharpness : float
   385	        Sigmoid sharpness on the MC threshold [s/(kg/kg)] (default 1e8).
   386	    cape_threshold : float
   387	        Secondary CAPE gate [J/kg] (default 70.0).
   388	    cape_sharpness : float
   389	        Sigmoid sharpness on CAPE gate (default 0.02).
   390	    cloud_depth_deep : float
   391	        Depth threshold separating mid-level from deep branches [m]
   392	        (default 3000.0).
   393	    cloud_depth_shallow_max : float
   394	        Depth threshold separating shallow from mid-level branches
   395	        [m] (default 1500.0).
   396	    depth_split_sharpness : float
   397	        Sigmoid sharpness on the depth thresholds [1/m] (default 1e-3).
   398	    enable_cmt : bool
   399	        Whether to compute CMT (default ``True``).
   400	    cmt_c_u, cmt_c_d : float
   401	        Gregory et al. 1997 closure coefficients (default 0.7).
   402	    smooth_trigger_sharpness : float
   403	        Sigmoid sharpness on the buoyancy / RH soft triggers (default
   404	        0.02).
   405	    precip_efficiency : float
   406	        Fraction of detrained condensate that precipitates (default 0.5).
   407	    tau_M_u_relax : float
   408	        Implicit-Euler relaxation timescale [s] for the profile carry
   409	        ``M_u`` toward its diagnosed equilibrium (default 1800.0).
   410	    parcel_dT, parcel_dq : float
   411	        Sub-cloud parcel perturbations (defaults 0.5 K, 1e-3 kg/kg).
   412	    """
   413	    epsilon_deep: float = 1.0e-4
   414	    delta_deep: float = 1.0e-4
   415	    epsilon_shallow: float = 3.0e-4
   416	    delta_shallow: float = 3.0e-4
   417	    epsilon_midlevel: float = 1.0e-4
   418	    delta_midlevel: float = 2.0e-4
   419	    enable_downdraft: bool = True
   420	    downdraft_alpha: float = 0.3
   421	    downdraft_RH_min: float = 0.2
   422	    moisture_convergence_threshold: float = 1.0e-8
   423	    moisture_convergence_sharpness: float = 1.0e8
   424	    cape_threshold: float = 70.0
   425	    cape_sharpness: float = 0.1
   426	    cloud_depth_deep: float = 3000.0
   427	    cloud_depth_shallow_max: float = 1500.0
   428	    depth_split_sharpness: float = 1.0e-3
   429	    enable_cmt: bool = True
   430	    cmt_c_u: float = 0.7
   431	    cmt_c_d: float = 0.7
   432	    smooth_trigger_sharpness: float = 0.02
   433	    precip_efficiency: float = 0.5
   434	    tau_M_u_relax: float = 1800.0
   435	    parcel_dT: float = 0.5
   436	    parcel_dq: float = 1.0e-3
   437	    tau_MC_proxy: float = 3600.0   # for saturation-deficit MC proxy
   438	    tau_shallow_M_b: float = 3600.0  # Shallow-cloud-base mass-flux timescale [s]
   439	    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
   440	    midlevel_M_b_fraction: float = 0.5  # M_b_midlevel = M_b_shallow * this
   441	
   442	
   443	class BechtoldConfig(NamedTuple):
   444	    """Configuration for the Bechtold/IFS convection scheme.
   445	
   446	    Builds on :class:`TiedtkeConfig` (three-class blend, downdraft,
   447	    CMT) with two distinguishing features:
   448	
   449	    * **PBL-CAPE / departure-CAPE closure** (Bechtold 2008):
   450	      ``M_b ∝ (CAPE_pbl - CAPE_eq)+ / tau_bl`` where ``CAPE_pbl`` is
   451	      diagnosed from a mass-weighted parcel within the boundary layer
   452	      rather than the surface parcel.
   453	    * **AR1 stochastic perturbation** (Bechtold 2014): ``M_b *= (1 +
   454	      amplitude * ε)`` where ``ε`` is an AR1-process realization with
   455	      decorrelation timescale ``stochastic_decorrelation``.  When
   456	      ``enable_stochastic`` is ``False`` the multiplier is 1.
   457	
   458	    Stochasticity defaults to OFF for reproducibility.  When enabled,
   459	    the leaf consumes a ``prng_key`` argument; the convection bridge
   460	    splits ``PhysicsState.prng_key`` into a Bechtold sub-key (folded
   461	    with module id ``0xBEC4``) and an advanced master key, returning
   462	    the latter as part of the multi-field carry update so subsequent
   463	    steps see independent random streams.
   464	
   465	    Inherits sensible defaults from Tiedtke 1989 with the entrainment
   466	    revision from Bechtold et al. 2008.
   467	
   468	    Fields
   469	    ------
   470	    epsilon_deep, delta_deep : float
   471	        Deep-branch entrainment / detrainment.  Default 1.75e-3 /
   472	        5e-4 (Bechtold et al. 2008 calibration — deeper entrainment
   473	        than Tiedtke 1989).
   474	    epsilon_shallow, delta_shallow : float
   475	        Shallow-branch (default 3e-3, 3e-3).
   476	    epsilon_midlevel, delta_midlevel : float
   477	        Mid-level branch (default 1e-4, 2e-4).
   478	    cape_pbl_depth : float
   479	        PBL depth [m] for the parcel-source mass weighting (default
   480	        500.0).
   481	    tau_bl : float
   482	        PBL closure timescale [s] (default 3600.0).
   483	    enable_stochastic : bool
   484	        Whether to apply the AR1 stochastic perturbation (default
   485	        ``False`` — reproducibility).
   486	    stochastic_amplitude : float
   487	        Multiplicative perturbation amplitude (default 0.5).
   488	    stochastic_decorrelation : float
   489	        AR1 decorrelation timescale [s] (default 7200.0).
   490	    enable_downdraft, downdraft_alpha, downdraft_RH_min : as Tiedtke.
   491	    enable_cmt, cmt_c_u, cmt_c_d : as Tiedtke.
   492	    cape_threshold, cape_sharpness, smooth_trigger_sharpness,
   493	    precip_efficiency, parcel_dT, parcel_dq, tau_M_u_relax,
   494	    cloud_depth_deep, cloud_depth_shallow_max, depth_split_sharpness :
   495	        as Tiedtke.
   496	    """
   497	    # Tiedtke-inherited / revised.
   498	    # Bechtold 2008 §2 calibrates ``delta_deep ≈ epsilon_deep`` for a
   499	    # near-neutral plume; the earlier default ``delta_deep = 5e-4`` (with
   500	    # ``epsilon_deep = 1.75e-3``) gives ``dM/dz ≈ +1.25e-3 M`` so the
   501	    # mass flux *grows* exponentially with height and peaks at the
   502	    # model top, not the cloud base — physically wrong.  Setting
   503	    # ``delta_deep = epsilon_deep`` matches the published calibration.
   504	    epsilon_deep: float = 1.75e-3
   505	    delta_deep: float = 1.75e-3
   506	    epsilon_shallow: float = 3.0e-3
   507	    delta_shallow: float = 3.0e-3
   508	    epsilon_midlevel: float = 1.0e-4
   509	    delta_midlevel: float = 2.0e-4
   510	    enable_downdraft: bool = True
   511	    downdraft_alpha: float = 0.3
   512	    downdraft_RH_min: float = 0.2
   513	    enable_cmt: bool = True
   514	    cmt_c_u: float = 0.7
   515	    cmt_c_d: float = 0.7
   516	    # The earlier ``cape_threshold = 0.0`` with ``cape_sharpness = 0.005``
   517	    # left the closure essentially always-on (``softplus(0)/0.005 ≈ 138
   518	    # J/kg`` of phantom CAPE even when CAPE = 0).  Match ZM/Tiedtke at
   519	    # 70 J/kg, 0.02 1/(J/kg) so the trigger is meaningful.
   520	    cape_threshold: float = 70.0

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/config.py | sed -n '520,700p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   520	    cape_threshold: float = 70.0
   521	    cape_sharpness: float = 0.1
   522	    smooth_trigger_sharpness: float = 0.02
   523	    precip_efficiency: float = 0.55
   524	    parcel_dT: float = 0.5
   525	    parcel_dq: float = 1.0e-3
   526	    tau_M_u_relax: float = 1800.0
   527	    cloud_depth_deep: float = 3000.0
   528	    cloud_depth_shallow_max: float = 1500.0
   529	    depth_split_sharpness: float = 1.0e-3
   530	    # Bechtold-specific
   531	    use_pbl_cape: bool = True
   532	    cape_pbl_depth: float = 500.0
   533	    tau_bl: float = 3600.0
   534	    enable_stochastic: bool = False
   535	    stochastic_amplitude: float = 0.5
   536	    stochastic_decorrelation: float = 7200.0
   537	    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
   538	
   539	
   540	class EDMFConfig(NamedTuple):
   541	    """Configuration for simplified EDMF convection (mass-flux part only).
   542	
   543	    Fields
   544	    ------
   545	    epsilon_0 : float
   546	        Entrainment rate [1/m].
   547	    delta_0 : float
   548	        Detrainment rate [1/m].
   549	    a_u_init : float
   550	        Initial updraft area fraction.
   551	    tau_a : float
   552	        Relaxation timescale for a_u [s].
   553	    w_u_min : float
   554	        Minimum updraft velocity [m/s].
   555	    cape_activation_scale : float
   556	        Sigmoid scale for CAPE trigger [J/kg].
   557	    cape_threshold : float
   558	        CAPE threshold [J/kg].
   559	    """
   560	    epsilon_0: float = 2e-3
   561	    delta_0: float = 2e-3
   562	    a_u_init: float = 0.1
   563	    tau_a: float = 1800.0
   564	    w_u_min: float = 0.1
   565	    # Trigger gating — see MassFluxConfig for rationale (sharper
   566	    # ``cape_activation_scale`` and a 70 J/kg threshold close the
   567	    # CAPE=0 leak from the earlier 50 % activation).
   568	    cape_activation_scale: float = 10.0
   569	    cape_threshold: float = 70.0
   570	    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
   571	
   572	
   573	class ConvectionConfig(NamedTuple):
   574	    """Top-level convection configuration.
   575	
   576	    Selects the active scheme and holds sub-configurations.
   577	
   578	    Fields
   579	    ------
   580	    scheme : str
   581	        Active convection scheme: ``"sbm"``, ``"dca"``, ``"kuo"``,
   582	        ``"mass_flux"``, ``"edmf"``, ``"zhang_mcfarlane"``, or
   583	        ``"none"``.  Future PRs (KF, Emanuel, Tiedtke, Bechtold) add
   584	        their literal here.
   585	    sbm, dca, kuo, mass_flux, edmf, zhang_mcfarlane :
   586	        Per-scheme configuration NamedTuples.
   587	    update_interval_steps : int
   588	        Recompute convection every N time steps (1 = every step).
   589	    """
   590	    scheme: str = "sbm"
   591	    sbm: SBMConfig = SBMConfig()
   592	    dca: DCAConfig = DCAConfig()
   593	    kuo: KuoConfig = KuoConfig()
   594	    mass_flux: MassFluxConfig = MassFluxConfig()
   595	    edmf: EDMFConfig = EDMFConfig()
   596	    zhang_mcfarlane: ZhangMcFarlaneConfig = ZhangMcFarlaneConfig()
   597	    kain_fritsch: KainFritschConfig = KainFritschConfig()
   598	    emanuel: EmanuelConfig = EmanuelConfig()
   599	    tiedtke: TiedtkeConfig = TiedtkeConfig()
   600	    bechtold: BechtoldConfig = BechtoldConfig()
   601	    update_interval_steps: int = 1

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/convection/output.py | sed -n '1,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Convection output container.
     2	
     3	ConvectionOutput is the common interface produced by every convection
     4	backend (SBM, DCA, Kuo, mass-flux, EDMF). Convection no longer carries
     5	its own surface-precipitation diagnostic; instead it emits a 3D
     6	convective source for cloud water (``dq_c_conv_dt``). Downstream the
     7	orchestrator routes that source into the cloud-water budget so that
     8	microphysics processes the convective condensate through its
     9	autoconversion / sedimentation / evaporation chain. Total surface
    10	precipitation is the sole responsibility of microphysics
    11	(``MicrophysicsOutput.precipitation``).
    12	
    13	This routing was introduced because the previous scalar
    14	``precipitation`` field forced every convection scheme to assume that
    15	all detrained condensate falls instantly to the surface, bypassing
    16	melting, evaporation in dry layers, and proper terminal-velocity
    17	sedimentation — a restrictive simplification.
    18	"""
    19	
    20	from __future__ import annotations
    21	
    22	from typing import NamedTuple
    23	
    24	import jax
    25	
    26	
    27	class ConvectionOutput(NamedTuple):
    28	    """Output from a convection scheme (backend-agnostic).
    29	
    30	    All required fields are at full levels with shape ``(ncol, nlev)``
    31	    except column-mean diagnostics which have shape ``(ncol,)``.
    32	
    33	    Fields
    34	    ------
    35	    dT_dt : jax.Array
    36	        Temperature tendency [K/s], shape (ncol, nlev).
    37	    dq_v_dt : jax.Array
    38	        Water vapor specific humidity tendency [kg/kg/s], shape
    39	        (ncol, nlev).
    40	    dq_c_conv_dt : jax.Array
    41	        Convective source term for cloud water mixing ratio [kg/kg/s],
    42	        shape (ncol, nlev). Replaces the legacy scalar surface
    43	        ``precipitation``: convective condensate now joins the
    44	        cloud-water bucket and is processed by microphysics, which
    45	        owns the resulting surface precipitation diagnostic. By
    46	        construction this field is non-negative.
    47	    cape : jax.Array
    48	        CAPE diagnostic [J/kg], shape (ncol,).
    49	    convective_mask : jax.Array
    50	        Smooth 0-1 convective indicator, shape (ncol,).
    51	    du_dt_conv : jax.Array or None
    52	        Optional convective momentum tendency for zonal wind [m/s²],
    53	        shape (ncol, nlev).  ``None`` for schemes without convective
    54	        momentum transport (the existing five schemes plus the
    55	        Kain-Fritsch and Emanuel schemes added in PRs 2 and 3).
    56	        Populated by Zhang-McFarlane (PR 1), Tiedtke (PR 4), and
    57	        Bechtold/IFS (PR 5) via the Gregory et al. 1997 closure.  When
    58	        ``None``, the integration bridge zero-fills the dynamical-core
    59	        wind tendencies.
    60	    dv_dt_conv : jax.Array or None
    61	        Optional convective momentum tendency for meridional wind
    62	        [m/s²], shape (ncol, nlev).  Same conventions as ``du_dt_conv``.
    63	    """
    64	    dT_dt: jax.Array
    65	    dq_v_dt: jax.Array
    66	    dq_c_conv_dt: jax.Array
    67	    cape: jax.Array
    68	    convective_mask: jax.Array
    69	    du_dt_conv: jax.Array | None = None
    70	    dv_dt_conv: jax.Array | None = None

exec
/bin/zsh -lc 'rg -n "convective|dq_c_conv_dt|dT_dt|dq_v_dt|precip|M_u_new|stratosphere_mass_flux_gate|delta_0_eff" src/legoesm/atmosphere/physics/convection tests -S' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/atmosphere/physics/convection/output.py:5:its own surface-precipitation diagnostic; instead it emits a 3D
src/legoesm/atmosphere/physics/convection/output.py:6:convective source for cloud water (``dq_c_conv_dt``). Downstream the
src/legoesm/atmosphere/physics/convection/output.py:8:microphysics processes the convective condensate through its
src/legoesm/atmosphere/physics/convection/output.py:10:precipitation is the sole responsibility of microphysics
src/legoesm/atmosphere/physics/convection/output.py:11:(``MicrophysicsOutput.precipitation``).
src/legoesm/atmosphere/physics/convection/output.py:14:``precipitation`` field forced every convection scheme to assume that
src/legoesm/atmosphere/physics/convection/output.py:35:    dT_dt : jax.Array
src/legoesm/atmosphere/physics/convection/output.py:37:    dq_v_dt : jax.Array
src/legoesm/atmosphere/physics/convection/output.py:40:    dq_c_conv_dt : jax.Array
src/legoesm/atmosphere/physics/convection/output.py:43:        ``precipitation``: convective condensate now joins the
src/legoesm/atmosphere/physics/convection/output.py:45:        owns the resulting surface precipitation diagnostic. By
src/legoesm/atmosphere/physics/convection/output.py:49:    convective_mask : jax.Array
src/legoesm/atmosphere/physics/convection/output.py:50:        Smooth 0-1 convective indicator, shape (ncol,).
src/legoesm/atmosphere/physics/convection/output.py:52:        Optional convective momentum tendency for zonal wind [m/s²],
src/legoesm/atmosphere/physics/convection/output.py:53:        shape (ncol, nlev).  ``None`` for schemes without convective
src/legoesm/atmosphere/physics/convection/output.py:61:        Optional convective momentum tendency for meridional wind
src/legoesm/atmosphere/physics/convection/output.py:64:    dT_dt: jax.Array
src/legoesm/atmosphere/physics/convection/output.py:65:    dq_v_dt: jax.Array
src/legoesm/atmosphere/physics/convection/output.py:66:    dq_c_conv_dt: jax.Array
src/legoesm/atmosphere/physics/convection/output.py:68:    convective_mask: jax.Array
src/legoesm/atmosphere/physics/convection/tiedtke.py:5:RH-dependent trigger, and convective momentum transport via Gregory
src/legoesm/atmosphere/physics/convection/tiedtke.py:46:    stratosphere_mass_flux_gate,
src/legoesm/atmosphere/physics/convection/tiedtke.py:221:    M_u_new = (conv_prog_profile + dt_over_tau * plume.M_u) / (1.0 + dt_over_tau)
src/legoesm/atmosphere/physics/convection/tiedtke.py:222:    # Cap M_u_new at config.M_b_max so every downstream use (kernel
src/legoesm/atmosphere/physics/convection/tiedtke.py:225:    M_u_new = jnp.clip(M_u_new, 0.0, config.M_b_max)
src/legoesm/atmosphere/physics/convection/tiedtke.py:228:    # smooths the time evolution of the convective forcing.
src/legoesm/atmosphere/physics/convection/tiedtke.py:229:    M_u_for_kernel = M_u_new
src/legoesm/atmosphere/physics/convection/tiedtke.py:234:    delta_0_eff = (
src/legoesm/atmosphere/physics/convection/tiedtke.py:243:    dT_dt_raw, dq_v_dt_raw, _ = _apply_mass_flux_kernel(
src/legoesm/atmosphere/physics/convection/tiedtke.py:251:    p_gate_qc = stratosphere_mass_flux_gate(p_full)
src/legoesm/atmosphere/physics/convection/tiedtke.py:252:    dq_c_conv_dt_raw = (
src/legoesm/atmosphere/physics/convection/tiedtke.py:253:        delta_0_eff[:, None] * M_u_for_kernel * p_gate_qc * plume.q_c_u / rho_safe
src/legoesm/atmosphere/physics/convection/tiedtke.py:257:    rescale = delta_0_eff[:, None] / config.delta_deep
src/legoesm/atmosphere/physics/convection/tiedtke.py:258:    dT_dt = dT_dt_raw * rescale
src/legoesm/atmosphere/physics/convection/tiedtke.py:259:    dq_v_dt = dq_v_dt_raw * rescale
src/legoesm/atmosphere/physics/convection/tiedtke.py:260:    dq_c_conv_dt = dq_c_conv_dt_raw
src/legoesm/atmosphere/physics/convection/tiedtke.py:283:        dT_dt_dd = -(constants.L_v / constants.c_pd) * (
src/legoesm/atmosphere/physics/convection/tiedtke.py:287:        dT_dt = dT_dt + dT_dt_dd
src/legoesm/atmosphere/physics/convection/tiedtke.py:305:    convective_mask = cape_weight * (deep_weight + shallow_weight + midlevel_weight)
src/legoesm/atmosphere/physics/convection/tiedtke.py:308:        dT_dt=dT_dt,
src/legoesm/atmosphere/physics/convection/tiedtke.py:309:        dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/convection/tiedtke.py:310:        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
src/legoesm/atmosphere/physics/convection/tiedtke.py:312:        convective_mask=convective_mask,
src/legoesm/atmosphere/physics/convection/tiedtke.py:316:    return out, M_u_new
src/legoesm/atmosphere/physics/convection/sbm.py:13:6. Diagnose precipitation from column moisture convergence
src/legoesm/atmosphere/physics/convection/sbm.py:23:- Betts, A. K., & Miller, M. J. (1986). A new convective adjustment
src/legoesm/atmosphere/physics/convection/sbm.py:89:    # 3. Identify the convective layer: only levels where the moist adiabat
src/legoesm/atmosphere/physics/convection/sbm.py:128:    # 7. Relaxation tendencies — only within the convective (cloud) layer
src/legoesm/atmosphere/physics/convection/sbm.py:129:    dT_dt = trigger[:, None] * cloud_mask * (T_ref - T) / tau_c[:, None]
src/legoesm/atmosphere/physics/convection/sbm.py:130:    dq_v_dt = trigger[:, None] * cloud_mask * (q_ref - q_v) / tau_c[:, None]
src/legoesm/atmosphere/physics/convection/sbm.py:133:    # level becomes cloud water rather than precipitating instantly.
src/legoesm/atmosphere/physics/convection/sbm.py:135:    # and evaporation, and produces the surface precipitation diagnostic.
src/legoesm/atmosphere/physics/convection/sbm.py:137:    # Naive ``max(-dq_v_dt, 0)`` per level would *create* water
src/legoesm/atmosphere/physics/convection/sbm.py:143:    # the legacy ``precipitation`` formula exactly. Per-level the
src/legoesm/atmosphere/physics/convection/sbm.py:146:    # dq_c_conv_dt = 0 everywhere, mirroring the legacy
src/legoesm/atmosphere/physics/convection/sbm.py:148:    local_cond = jnp.maximum(-dq_v_dt, 0.0)
src/legoesm/atmosphere/physics/convection/sbm.py:151:        -jnp.sum(dq_v_dt * dp / constants.g, axis=-1, keepdims=True),
src/legoesm/atmosphere/physics/convection/sbm.py:154:    dq_c_conv_dt = local_cond * (
src/legoesm/atmosphere/physics/convection/sbm.py:159:        dT_dt=dT_dt,
src/legoesm/atmosphere/physics/convection/sbm.py:160:        dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/convection/sbm.py:161:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/sbm.py:163:        convective_mask=trigger,
src/legoesm/atmosphere/physics/convection/bechtold.py:48:    stratosphere_mass_flux_gate,
src/legoesm/atmosphere/physics/convection/bechtold.py:200:        # tropical convective regions (Bechtold 2008 Fig. 2).
src/legoesm/atmosphere/physics/convection/bechtold.py:247:    M_u_new = (conv_prog_profile + dt_over_tau * plume.M_u) / (1.0 + dt_over_tau)
src/legoesm/atmosphere/physics/convection/bechtold.py:248:    # Cap M_u_new at config.M_b_max so every downstream use (kernel
src/legoesm/atmosphere/physics/convection/bechtold.py:251:    M_u_new = jnp.clip(M_u_new, 0.0, config.M_b_max)
src/legoesm/atmosphere/physics/convection/bechtold.py:254:    delta_0_eff = (
src/legoesm/atmosphere/physics/convection/bechtold.py:259:    dT_dt_raw, dq_v_dt_raw, _ = _apply_mass_flux_kernel(
src/legoesm/atmosphere/physics/convection/bechtold.py:261:        plume.T_u, plume.q_u, plume.q_c_u, M_u_new,
src/legoesm/atmosphere/physics/convection/bechtold.py:265:    p_gate_qc = stratosphere_mass_flux_gate(p_full)
src/legoesm/atmosphere/physics/convection/bechtold.py:266:    dq_c_conv_dt_raw = (
src/legoesm/atmosphere/physics/convection/bechtold.py:267:        delta_0_eff[:, None] * M_u_new * p_gate_qc * plume.q_c_u / rho_safe
src/legoesm/atmosphere/physics/convection/bechtold.py:269:    rescale = delta_0_eff[:, None] / config.delta_deep
src/legoesm/atmosphere/physics/convection/bechtold.py:270:    dT_dt = dT_dt_raw * rescale
src/legoesm/atmosphere/physics/convection/bechtold.py:271:    dq_v_dt = dq_v_dt_raw * rescale
src/legoesm/atmosphere/physics/convection/bechtold.py:272:    dq_c_conv_dt = dq_c_conv_dt_raw
src/legoesm/atmosphere/physics/convection/bechtold.py:291:        dT_dt_dd = -(constants.L_v / constants.c_pd) * (
src/legoesm/atmosphere/physics/convection/bechtold.py:294:        dT_dt = dT_dt + dT_dt_dd
src/legoesm/atmosphere/physics/convection/bechtold.py:299:            M_d = -config.downdraft_alpha * M_u_new * 0.3
src/legoesm/atmosphere/physics/convection/bechtold.py:303:            u, v, M_u_new, M_d,
src/legoesm/atmosphere/physics/convection/bechtold.py:311:    convective_mask = cape_weight * (deep_weight + shallow_weight + midlevel_weight)
src/legoesm/atmosphere/physics/convection/bechtold.py:314:        dT_dt=dT_dt,
src/legoesm/atmosphere/physics/convection/bechtold.py:315:        dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/convection/bechtold.py:316:        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
src/legoesm/atmosphere/physics/convection/bechtold.py:318:        convective_mask=convective_mask,
src/legoesm/atmosphere/physics/convection/bechtold.py:322:    return out, M_u_new, conv_stoch_state_new
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:7:shallow branches.  No convective momentum transport — KF emits
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:25:  detraining plume model and its application in convective
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:27:* Kain, J. S. (2004). The Kain–Fritsch convective parameterization:
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:213:    dT_dt_raw, dq_v_dt_raw, dq_c_conv_dt_raw = _apply_mass_flux_kernel(
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:220:    dT_dt = dT_dt_raw * branch_weight[:, None]
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:221:    dq_v_dt = dq_v_dt_raw * branch_weight[:, None]
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:222:    dq_c_conv_dt = dq_c_conv_dt_raw * branch_weight[:, None]
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:225:        dT_dt=dT_dt,
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:226:        dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:227:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:229:        convective_mask=overall_weight,
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:230:        # KF has no convective momentum transport.
src/legoesm/atmosphere/physics/convection/integration.py:159:    When *phys_state* is passed, the convective prognostic profile is
src/legoesm/atmosphere/physics/convection/integration.py:349:            dT_dt = jnp.zeros(shape_3d, dtype=_state_dtype)
src/legoesm/atmosphere/physics/convection/integration.py:372:            dT_dt = conv_out.dT_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/convection/integration.py:473:            dT_dt = conv_out.dT_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/convection/integration.py:480:            dT_dt = conv_out.dT_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/convection/integration.py:491:        # Convective detrained condensate (``dq_c_conv_dt``) feeds the
src/legoesm/atmosphere/physics/convection/integration.py:498:            dq_v_dt = conv_out.dq_v_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/convection/integration.py:499:            dq_c_conv_dt = conv_out.dq_c_conv_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/convection/integration.py:502:                    data=dq_v_dt, name="dq_v_dt_conv",
src/legoesm/atmosphere/physics/convection/integration.py:506:                    data=dq_c_conv_dt, name="dq_c_conv_dt",
src/legoesm/atmosphere/physics/convection/integration.py:555:            dT_dt=Field(
src/legoesm/atmosphere/physics/convection/integration.py:556:                data=dT_dt, name="dT_dt_conv",
src/legoesm/atmosphere/physics/convection/integration.py:591:    When *phys_state* is passed, the convective prognostic profile is
src/legoesm/atmosphere/physics/convection/integration.py:830:        dT_dt = conv_out.dT_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/convection/integration.py:831:        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)
src/legoesm/atmosphere/physics/convection/integration.py:840:        # 1 (``q_c``) carries the convective detrained-condensate
src/legoesm/atmosphere/physics/convection/integration.py:848:            dq_v_dt = conv_out.dq_v_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/convection/integration.py:849:            dtracers = dtracers.at[..., 0].set(dq_v_dt)
src/legoesm/atmosphere/physics/convection/integration.py:851:            dq_c_conv_dt = conv_out.dq_c_conv_dt.reshape(shape_3d)
src/legoesm/atmosphere/physics/convection/integration.py:852:            dtracers = dtracers.at[..., 1].set(dq_c_conv_dt)
src/legoesm/atmosphere/physics/convection/integration.py:895:    When *phys_state* is passed, the convective prognostic profile is
src/legoesm/atmosphere/physics/convection/integration.py:1042:            dT_dt = jnp.zeros_like(T)
src/legoesm/atmosphere/physics/convection/integration.py:1060:            dT_dt = conv_out.dT_dt.reshape(n_lat, n_lon, nlev)
src/legoesm/atmosphere/physics/convection/integration.py:1155:            dT_dt = conv_out.dT_dt.reshape(n_lat, n_lon, nlev)
src/legoesm/atmosphere/physics/convection/integration.py:1162:            dT_dt = conv_out.dT_dt.reshape(n_lat, n_lon, nlev)
src/legoesm/atmosphere/physics/convection/integration.py:1165:        dT_hat = sh_analysis_3d(grid, dT_dt)
src/legoesm/atmosphere/physics/convection/integration.py:1181:                dq_v_dt_grid = conv_out.dq_v_dt.reshape(n_lat, n_lon, nlev)
src/legoesm/atmosphere/physics/convection/integration.py:1185:                        data=dq_v_dt_grid.astype(_qv_template.data.dtype)
src/legoesm/atmosphere/physics/convection/integration.py:1188:                    tt["q_v"] = dq_v_dt_grid.astype(_qv_template.dtype)
src/legoesm/atmosphere/physics/convection/integration.py:1190:                dq_c_dt_grid = conv_out.dq_c_conv_dt.reshape(n_lat, n_lon, nlev)
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:183:    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:201:    convective_mask = cape_weight  # already a smooth (ncol,) indicator
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:204:        dT_dt=dT_dt,
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:205:        dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:206:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:208:        convective_mask=convective_mask,
src/legoesm/atmosphere/physics/convection/_plume.py:20:* :func:`cmt_gregory_1997`      — Gregory et al. 1997 convective
src/legoesm/atmosphere/physics/convection/_plume.py:174:    # at very cold parcels (defensively — convective parcels are rarely
src/legoesm/atmosphere/physics/convection/_plume.py:389:    Cloud water accumulates at rate ``C - precip_rate`` (precipitation
src/legoesm/atmosphere/physics/convection/_plume.py:574:    """Gregory et al. 1997 convective momentum transport closure.
src/legoesm/atmosphere/physics/convection/_plume.py:625:    # convective momentum into the model top (where the plume should
src/legoesm/atmosphere/physics/convection/_plume.py:630:        stratosphere_mass_flux_gate,
src/legoesm/atmosphere/physics/convection/_plume.py:632:    p_gate = stratosphere_mass_flux_gate(p_full)
src/legoesm/atmosphere/physics/convection/mass_flux.py:23:precipitation from the column integral of detrained condensate.
src/legoesm/atmosphere/physics/convection/mass_flux.py:37:  mass-flux approach for the convective boundary layer.
src/legoesm/atmosphere/physics/convection/mass_flux.py:99:    """Compute moist-adiabat profile, CAPE, and a smooth convective mask.
src/legoesm/atmosphere/physics/convection/mass_flux.py:101:    Returns ``(T_moist, cape, convective_mask)``. ``T_moist`` is shape
src/legoesm/atmosphere/physics/convection/mass_flux.py:102:    ``(ncol, nlev)``; ``cape`` and ``convective_mask`` are shape
src/legoesm/atmosphere/physics/convection/mass_flux.py:110:    convective_mask = jax.nn.sigmoid(
src/legoesm/atmosphere/physics/convection/mass_flux.py:113:    return T_moist, cape, convective_mask
src/legoesm/atmosphere/physics/convection/mass_flux.py:132:def stratosphere_mass_flux_gate(
src/legoesm/atmosphere/physics/convection/mass_flux.py:141:    convective tendencies from accumulating in the model top layer,
src/legoesm/atmosphere/physics/convection/mass_flux.py:181:    returns ``(dT_dt, dq_v_dt, dq_c_conv_dt)`` — all shape
src/legoesm/atmosphere/physics/convection/mass_flux.py:192:    hPa, the canonical tropical tropopause) receive no convective
src/legoesm/atmosphere/physics/convection/mass_flux.py:193:    tendency.  See ``stratosphere_mass_flux_gate`` for details.
src/legoesm/atmosphere/physics/convection/mass_flux.py:195:    The convective source for cloud water is the per-level detrainment
src/legoesm/atmosphere/physics/convection/mass_flux.py:197:    ``dq_c_conv_dt = delta_0 * M * q_c_u / rho`` [kg/kg/s], non-negative
src/legoesm/atmosphere/physics/convection/mass_flux.py:205:    ``dq_c_conv_dt`` through its full chain (autoconversion,
src/legoesm/atmosphere/physics/convection/mass_flux.py:207:    precipitation; convection no longer assumes the condensate falls
src/legoesm/atmosphere/physics/convection/mass_flux.py:217:    # Per-layer convective heating ``≈ delta_0 · M_u · (T_u−T)/ρ``
src/legoesm/atmosphere/physics/convection/mass_flux.py:226:    # Stratospheric pressure gate — see ``stratosphere_mass_flux_gate``.
src/legoesm/atmosphere/physics/convection/mass_flux.py:227:    M_profile = M_profile * stratosphere_mass_flux_gate(
src/legoesm/atmosphere/physics/convection/mass_flux.py:237:    dT_dt = dT_subsidence + dT_detrain
src/legoesm/atmosphere/physics/convection/mass_flux.py:238:    dq_v_dt = dq_subsidence + dq_detrain
src/legoesm/atmosphere/physics/convection/mass_flux.py:240:    dq_c_conv_dt = delta_0 * M_profile * jnp.clip(q_c_u, 0.0, None) / rho_safe
src/legoesm/atmosphere/physics/convection/mass_flux.py:241:    return dT_dt, dq_v_dt, dq_c_conv_dt
src/legoesm/atmosphere/physics/convection/mass_flux.py:259:    convective_mask: jax.Array
src/legoesm/atmosphere/physics/convection/mass_flux.py:275:    T_moist, cape, convective_mask = _compute_cape_diagnostics(
src/legoesm/atmosphere/physics/convection/mass_flux.py:279:    M_eq = convective_mask * config.M_scale
src/legoesm/atmosphere/physics/convection/mass_flux.py:290:        convective_mask=convective_mask,
src/legoesm/atmosphere/physics/convection/mass_flux.py:310:    convective_mask = closure.convective_mask
src/legoesm/atmosphere/physics/convection/mass_flux.py:333:    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
src/legoesm/atmosphere/physics/convection/mass_flux.py:348:        dT_dt=dT_dt,
src/legoesm/atmosphere/physics/convection/mass_flux.py:349:        dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/convection/mass_flux.py:350:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/mass_flux.py:352:        convective_mask=convective_mask,
src/legoesm/atmosphere/physics/convection/mass_flux.py:427:    T_moist, cape, convective_mask = _compute_cape_diagnostics(
src/legoesm/atmosphere/physics/convection/mass_flux.py:433:    a_u_eq = convective_mask * config.a_u_init
src/legoesm/atmosphere/physics/convection/mass_flux.py:474:    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
src/legoesm/atmosphere/physics/convection/mass_flux.py:489:        dT_dt=dT_dt,
src/legoesm/atmosphere/physics/convection/mass_flux.py:490:        dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/convection/mass_flux.py:491:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/mass_flux.py:493:        convective_mask=convective_mask,
src/legoesm/atmosphere/physics/convection/dca.py:5:Excess moisture is removed as precipitation.
src/legoesm/atmosphere/physics/convection/dca.py:45:    - Redistribute excess moisture as precipitation
src/legoesm/atmosphere/physics/convection/dca.py:66:    precip_col : jax.Array
src/legoesm/atmosphere/physics/convection/dca.py:82:        T_work, q_work, precip_accum = carry
src/legoesm/atmosphere/physics/convection/dca.py:141:        # Remove excess moisture (precipitation)
src/legoesm/atmosphere/physics/convection/dca.py:149:        # Accumulate precipitation from moisture removal
src/legoesm/atmosphere/physics/convection/dca.py:152:        precip_new = precip_accum + (dq_upper + dq_below) / constants.g
src/legoesm/atmosphere/physics/convection/dca.py:159:        return (T_work, q_work, precip_new), None
src/legoesm/atmosphere/physics/convection/dca.py:161:    # Pin the precip carry dtype to whatever ``q * dp`` actually
src/legoesm/atmosphere/physics/convection/dca.py:166:    _precip_dtype = jnp.result_type(q_v_rev, dp_rev)
src/legoesm/atmosphere/physics/convection/dca.py:167:    init_carry = (T_rev, q_v_rev, jnp.zeros(ncol, dtype=_precip_dtype))
src/legoesm/atmosphere/physics/convection/dca.py:169:    (T_adj_rev, q_adj_rev, precip_col), _ = jax.lax.scan(
src/legoesm/atmosphere/physics/convection/dca.py:177:    return T_new, q_new, precip_col
src/legoesm/atmosphere/physics/convection/dca.py:216:    # ``precip_total`` to the same result-type so the outer scan carry
src/legoesm/atmosphere/physics/convection/dca.py:220:    precip_total = jnp.zeros(ncol, dtype=jnp.result_type(q_v, dp))
src/legoesm/atmosphere/physics/convection/dca.py:229:    (T_adj, q_adj, precip_total), _ = jax.lax.scan(
src/legoesm/atmosphere/physics/convection/dca.py:231:        (T_adj, q_adj, precip_total),
src/legoesm/atmosphere/physics/convection/dca.py:245:    dT_dt = cape_gate[:, None] * (T_adj - T) / dt
src/legoesm/atmosphere/physics/convection/dca.py:246:    dq_v_dt = cape_gate[:, None] * (q_adj - q_v) / dt
src/legoesm/atmosphere/physics/convection/dca.py:248:    # rescaling so that ∫ dq_c_conv_dt dp/g equals the column-net
src/legoesm/atmosphere/physics/convection/dca.py:249:    # drying (matches the legacy ``precipitation`` formula). Naive
src/legoesm/atmosphere/physics/convection/dca.py:250:    # per-level ``max(-dq_v_dt, 0)`` would create water column-wide
src/legoesm/atmosphere/physics/convection/dca.py:253:    # at every level. ``precip_total`` (the scan-accumulated column
src/legoesm/atmosphere/physics/convection/dca.py:255:    # precipitation diagnostic.
src/legoesm/atmosphere/physics/convection/dca.py:256:    del precip_total
src/legoesm/atmosphere/physics/convection/dca.py:257:    local_cond = jnp.maximum(-dq_v_dt, 0.0)
src/legoesm/atmosphere/physics/convection/dca.py:260:        -jnp.sum(dq_v_dt * dp / constants.g, axis=-1, keepdims=True),
src/legoesm/atmosphere/physics/convection/dca.py:263:    dq_c_conv_dt = local_cond * (
src/legoesm/atmosphere/physics/convection/dca.py:268:    convective_mask = cape_gate
src/legoesm/atmosphere/physics/convection/dca.py:271:        dT_dt=dT_dt,
src/legoesm/atmosphere/physics/convection/dca.py:272:        dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/convection/dca.py:273:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/dca.py:275:        convective_mask=convective_mask,
tests/stress/test_phase1_sea_ice.py:32:        precip_total=jnp.zeros(shape),
tests/stress/test_phase1_sea_ice.py:33:        precip_snow=jnp.zeros(shape),
tests/stress/test_phase1_sea_ice.py:44:        has_precipitation=jnp.zeros(shape),
tests/sea_ice/validation/test_sea_ice_validation.py:49:        precip_total=jnp.zeros(shape),
tests/sea_ice/validation/test_sea_ice_validation.py:50:        precip_snow=jnp.zeros(shape),
tests/sea_ice/validation/test_sea_ice_validation.py:61:        has_precipitation=jnp.ones(shape),
src/legoesm/atmosphere/physics/convection/kuo.py:12:5. Emit per-level cloud-water source (``dq_c_conv_dt``) from the
src/legoesm/atmosphere/physics/convection/kuo.py:15:   surface precipitation diagnostic.
src/legoesm/atmosphere/physics/convection/kuo.py:29:legacy ``ConvectionOutput.precipitation`` formulation; Option C
src/legoesm/atmosphere/physics/convection/kuo.py:122:    # sigmoid trigger this guarantees ``dT_dt = 0``,
src/legoesm/atmosphere/physics/convection/kuo.py:123:    # ``implied_condensation = 0``, ``dq_v_dt = 0``, and
src/legoesm/atmosphere/physics/convection/kuo.py:124:    # ``dq_c_conv_dt = 0`` whenever MC = 0 — no spurious heating,
src/legoesm/atmosphere/physics/convection/kuo.py:127:    dT_dt = (
src/legoesm/atmosphere/physics/convection/kuo.py:139:    # ``dq_c_conv_dt`` (see step 7). Fraction ``(1 - alpha_heat)``
src/legoesm/atmosphere/physics/convection/kuo.py:150:    # water by latent heat balance: c_pd * dT_dt = L_v * (-dq_v) for
src/legoesm/atmosphere/physics/convection/kuo.py:152:    # directly via ``dq_c_conv_dt`` so it can process the convective
src/legoesm/atmosphere/physics/convection/kuo.py:156:    implied_condensation = dT_dt * constants.c_pd / constants.L_v  # (ncol, nlev)
src/legoesm/atmosphere/physics/convection/kuo.py:169:    dq_v_dt = (
src/legoesm/atmosphere/physics/convection/kuo.py:176:    dq_v_dt = dq_v_dt - implied_condensation
src/legoesm/atmosphere/physics/convection/kuo.py:191:    # to yield a spurious ``dT_dt`` and therefore a spurious
src/legoesm/atmosphere/physics/convection/kuo.py:203:    dq_c_conv_dt = local_cond * (
src/legoesm/atmosphere/physics/convection/kuo.py:211:        dT_dt=dT_dt,
src/legoesm/atmosphere/physics/convection/kuo.py:212:        dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/convection/kuo.py:213:        dq_c_conv_dt=dq_c_conv_dt,
src/legoesm/atmosphere/physics/convection/kuo.py:215:        convective_mask=trigger,
src/legoesm/atmosphere/physics/convection/config.py:92:        ~10× too much precipitation in tropical RCE.  CCM2/CCM3 used
src/legoesm/atmosphere/physics/convection/config.py:94:        responsive to real precipitating columns without
src/legoesm/atmosphere/physics/convection/config.py:95:        over-precipitating.
src/legoesm/atmosphere/physics/convection/config.py:127:    # Trigger: ``convective_mask = sigmoid((cape − cape_threshold)
src/legoesm/atmosphere/physics/convection/config.py:144:    optional Gregory et al. 1997 convective momentum transport.
src/legoesm/atmosphere/physics/convection/config.py:165:        Whether to compute convective momentum transport tendencies
src/legoesm/atmosphere/physics/convection/config.py:200:    cloud branches are blended on cloud depth.  No convective
src/legoesm/atmosphere/physics/convection/config.py:287:    precip_efficiency_water : float
src/legoesm/atmosphere/physics/convection/config.py:289:    precip_efficiency_lcl : float
src/legoesm/atmosphere/physics/convection/config.py:291:    precip_threshold_qc : float
src/legoesm/atmosphere/physics/convection/config.py:292:        Cloud-water threshold above which precipitation falls
src/legoesm/atmosphere/physics/convection/config.py:308:        Fraction of precipitation that re-evaporates below cloud base
src/legoesm/atmosphere/physics/convection/config.py:323:    precip_efficiency_water: float = 1.0
src/legoesm/atmosphere/physics/convection/config.py:324:    precip_efficiency_lcl: float = 0.0
src/legoesm/atmosphere/physics/convection/config.py:325:    precip_threshold_qc: float = 1.0e-3
src/legoesm/atmosphere/physics/convection/config.py:336:    # of *precipitation* (rain) back to vapor in the BL.  In a model
src/legoesm/atmosphere/physics/convection/config.py:338:    # from ``dq_c_conv_dt`` (the cloud-water source), so enabling it
src/legoesm/atmosphere/physics/convection/config.py:356:    trigger; convective momentum transport via Gregory et al. 1997.
src/legoesm/atmosphere/physics/convection/config.py:405:    precip_efficiency : float
src/legoesm/atmosphere/physics/convection/config.py:406:        Fraction of detrained condensate that precipitates (default 0.5).
src/legoesm/atmosphere/physics/convection/config.py:433:    precip_efficiency: float = 0.5
src/legoesm/atmosphere/physics/convection/config.py:493:    precip_efficiency, parcel_dT, parcel_dq, tau_M_u_relax,
src/legoesm/atmosphere/physics/convection/config.py:523:    precip_efficiency: float = 0.55
tests/land/validation/test_bulk_flux_differentiability.py:154:                precip_total=jnp.zeros(shape),
tests/land/validation/test_bulk_flux_differentiability.py:155:                precip_snow=jnp.zeros(shape),
tests/land/validation/test_bulk_flux_differentiability.py:166:                has_precipitation=jnp.array(1.0),
src/legoesm/atmosphere/physics/convection/_triggers.py:20:  step, and the Gregory et al. 1997 convective momentum-transport
src/legoesm/atmosphere/physics/convection/emanuel.py:25:No convective momentum transport (Emanuel CMT is a separate
src/legoesm/atmosphere/physics/convection/emanuel.py:180:    dT_dt_raw, dq_v_dt_raw, dq_c_conv_dt_raw = _apply_mass_flux_kernel(
src/legoesm/atmosphere/physics/convection/emanuel.py:187:    dT_dt = dT_dt_raw * sort_multiplier
src/legoesm/atmosphere/physics/convection/emanuel.py:188:    dq_v_dt = dq_v_dt_raw * sort_multiplier
src/legoesm/atmosphere/physics/convection/emanuel.py:189:    dq_c_conv_dt = dq_c_conv_dt_raw * sort_multiplier
src/legoesm/atmosphere/physics/convection/emanuel.py:206:        column_condensate = jnp.sum(dq_c_conv_dt_raw * dp, axis=-1) / constants.g
src/legoesm/atmosphere/physics/convection/emanuel.py:219:        # cloud water is *produced* (i.e. dq_c_conv_dt_raw), not where
src/legoesm/atmosphere/physics/convection/emanuel.py:221:        # ``evap_rate`` from ``dq_c_conv_dt`` *at the BL*, then clipped
src/legoesm/atmosphere/physics/convection/emanuel.py:223:        # ``dq_c_conv_dt_raw``) and effectively created vapor from
src/legoesm/atmosphere/physics/convection/emanuel.py:228:        dT_dt = dT_dt + dT_evap
src/legoesm/atmosphere/physics/convection/emanuel.py:229:        dq_v_dt = dq_v_dt + dq_v_evap
src/legoesm/atmosphere/physics/convection/emanuel.py:233:        # ``dq_c_conv_dt ≥ 0`` and column water conserved.
src/legoesm/atmosphere/physics/convection/emanuel.py:234:        col_dq_c = jnp.sum(dq_c_conv_dt_raw * dp, axis=-1) / constants.g
src/legoesm/atmosphere/physics/convection/emanuel.py:235:        weight = dq_c_conv_dt_raw / jnp.maximum(col_dq_c[:, None], 1e-12)
src/legoesm/atmosphere/physics/convection/emanuel.py:244:        dq_c_conv_dt = dq_c_conv_dt - subtract
src/legoesm/atmosphere/physics/convection/emanuel.py:247:        dT_dt=dT_dt,
src/legoesm/atmosphere/physics/convection/emanuel.py:248:        dq_v_dt=dq_v_dt,
src/legoesm/atmosphere/physics/convection/emanuel.py:249:        dq_c_conv_dt=jnp.maximum(dq_c_conv_dt, 0.0),
src/legoesm/atmosphere/physics/convection/emanuel.py:251:        convective_mask=cape_weight,
tests/parallel/test_scaling_operators.py:420:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/parallel/test_scaling_operators.py:423:        assert tend.dT_dt.data.shape == (mesh.nCells, nlev)
tests/land/validation/test_bulk_flux_all_tiles.py:35:        precip_total=jnp.full(shape, 1e-5),
tests/land/validation/test_bulk_flux_all_tiles.py:36:        precip_snow=jnp.zeros(shape),
tests/land/validation/test_bulk_flux_all_tiles.py:47:        has_precipitation=jnp.array(1.0),
tests/stress/test_phase1_ocean_slab.py:28:        precip_total=z,
tests/stress/test_phase1_ocean_slab.py:29:        precip_snow=z,
tests/stress/test_phase1_ocean_slab.py:40:        has_precipitation=jnp.ones(shape),
tests/unit/test_bechtold.py:81:    out, M_u_new, stoch_new = bechtold_convection(
tests/unit/test_bechtold.py:84:    assert M_u_new.shape == (ncol, nlev)
tests/unit/test_bechtold.py:86:    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
tests/unit/test_bechtold.py:87:                out.convective_mask, out.du_dt_conv, out.dv_dt_conv,
tests/unit/test_bechtold.py:88:                M_u_new, stoch_new):
tests/unit/test_bechtold.py:155:    assert jnp.allclose(out1.dT_dt, out2.dT_dt)
tests/unit/test_bechtold.py:225:        return jnp.sum(out.dT_dt)
tests/unit/test_bechtold.py:242:        return jnp.sum(out.dT_dt)
tests/unit/test_bechtold.py:262:        return jnp.sum(out.dT_dt)
tests/unit/test_bechtold.py:319:    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
tests/unit/test_bechtold.py:416:        return jnp.sum(tend.dT_dt.data ** 2)
tests/unit/test_bechtold.py:512:    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
tests/unit/test_bechtold.py:543:    H = float(jnp.sum(out.dT_dt * dp / constants.g, axis=1).mean()) * constants.c_pd
tests/unit/test_bechtold.py:544:    Q = float(jnp.sum(out.dq_v_dt * dp / constants.g, axis=1).mean()) * constants.L_v
tests/unit/test_bechtold.py:545:    C = float(jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1).mean()) * constants.L_v
tests/land/test_land_stability.py:8:- Water budget closes (precip - evap - runoff = storage change)
tests/land/test_land_stability.py:48:    precip_rate: float = 2e-5,
tests/land/test_land_stability.py:58:    cold : if True, set sub-freezing air temperature and snow precip.
tests/land/test_land_stability.py:89:    precip_total = jnp.full(ncol, precip_rate, dtype=dtype)
tests/land/test_land_stability.py:91:        precip_snow = precip_total
tests/land/test_land_stability.py:93:        precip_snow = jnp.where(T_atm < 275.0, precip_total, jnp.zeros(ncol, dtype=dtype))
tests/land/test_land_stability.py:101:        precip_total=precip_total,
tests/land/test_land_stability.py:102:        precip_snow=precip_snow,
tests/land/test_land_stability.py:113:        has_precipitation=jnp.ones(ncol, dtype=dtype),
tests/land/test_land_stability.py:424:        """Net moisture change is consistent with precip and evaporation."""
tests/land/test_land_stability.py:432:        # Total precip = precip_rate * total_time
tests/land/test_land_stability.py:434:        precip_acc = 2e-5 * total_time  # kg/m2
tests/land/test_land_stability.py:436:        # dW should be approximately precip - evap, clipped to [0, W_max]
tests/land/test_land_stability.py:437:        expected_dW = precip_acc - evap_acc
tests/land/test_land_stability.py:456:        """Snow accumulates when forcing is cold with snow precip."""
tests/land/test_land_stability.py:466:                cold=True, precip_rate=3e-5,
tests/land/test_land_stability.py:491:        # 5 days of warm forcing, no precip
tests/land/test_land_stability.py:496:                NCOL, LATITUDES, day=170.0, hour=hour, precip_rate=0.0,
tests/stress/test_phase3_restart.py:49:# precipitation) which undergoes a one-step reset transient after restart.
tests/ocean/unit/test_ocean_fc.py:41:    assert tend.dT_dt.data.shape == (6, 8, 8, 5)
tests/ocean/unit/test_ocean_fc.py:43:    assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/ocean/unit/test_ocean_fc.py:71:    assert float(jnp.max(jnp.abs(tend.dT_dt.data * (1 - mask_3d)))) == 0.0
tests/unit/test_land_ice_slab_land.py:70:    precip_total=0.0,
tests/unit/test_land_ice_slab_land.py:71:    precip_snow=0.0,
tests/unit/test_land_ice_slab_land.py:73:    has_precipitation=1.0,
tests/unit/test_land_ice_slab_land.py:80:        precip_total=jnp.full(s, precip_total, f),
tests/unit/test_land_ice_slab_land.py:81:        precip_snow=jnp.full(s, precip_snow, f),
tests/unit/test_land_ice_slab_land.py:92:        has_precipitation=jnp.full(s, has_precipitation, f),
tests/unit/test_land_ice_slab_land.py:260:    def test_precip_accumulation(self):
tests/unit/test_land_ice_slab_land.py:262:        precip = 1e-4  # kg/m2/s
tests/unit/test_land_ice_slab_land.py:265:            precip_total=precip, precip_snow=0.0,
tests/unit/test_land_ice_slab_land.py:272:        dW_expected = (precip - float(evap_rate[0])) * DT
tests/unit/test_land_ice_slab_land.py:278:        """When W = W_max and precip > evap, W stays capped."""
tests/unit/test_land_ice_slab_land.py:281:            precip_total=1e-3, precip_snow=0.0,
tests/unit/test_land_ice_slab_land.py:352:        forcing = make_forcing(precip_total=1e-5)
tests/unit/test_physics_combined.py:68:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), "dT_dt NaN"
tests/unit/test_physics_combined.py:127:    sum_dT = tend_rad.dT_dt.data + tend_turb.dT_dt.data + tend_gwd.dT_dt.data
tests/unit/test_physics_combined.py:131:    max_dT_err = float(jnp.max(jnp.abs(tend_all.dT_dt.data - sum_dT)))
tests/unit/test_physics_combined.py:155:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"NaN with gray+{conv}"
tests/unit/test_physics_combined.py:175:    assert tend.dT_dt.data.shape == state.T.data.shape, "dT_dt shape mismatch"
tests/unit/test_physics_combined.py:202:        new_T = state.T.data + tend.dT_dt.data * dt
tests/ocean/unit/test_gm_redi_latlon_cgrid.py:467:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/land/unit/test_multilayer_land.py:603:            precip_total=jnp.full(ncol, 1e-4),  # 0.1 mm/s
tests/land/unit/test_multilayer_land.py:604:            precip_snow=jnp.zeros(ncol),
tests/land/unit/test_multilayer_land.py:615:            has_precipitation=jnp.ones(ncol),
tests/land/unit/test_multilayer_land.py:718:    def test_no_precip_no_runoff(self):
tests/land/unit/test_multilayer_land.py:719:        """Without precipitation, should have minimal surface runoff."""
tests/land/unit/test_multilayer_land.py:728:        # Zero precipitation, low q so minimal evap demand
tests/land/unit/test_multilayer_land.py:732:            precip_total=jnp.zeros(ncol),
tests/land/unit/test_multilayer_land.py:733:            precip_snow=jnp.zeros(ncol),
tests/land/unit/test_multilayer_land.py:744:            has_precipitation=jnp.ones(ncol),
tests/land/unit/test_multilayer_land.py:749:        # Surface runoff should be zero (no precip and evap may be negative flux_top,
tests/land/unit/test_multilayer_land.py:755:        """No-precip, zero-bottom-flux: total soil water loss matches
tests/land/unit/test_multilayer_land.py:772:        # No precip, warm sunny → drives evaporation from soil
tests/land/unit/test_multilayer_land.py:776:            precip_total=jnp.zeros(ncol),
tests/land/unit/test_multilayer_land.py:777:            precip_snow=jnp.zeros(ncol),
tests/land/unit/test_multilayer_land.py:788:            has_precipitation=jnp.ones(ncol),
tests/land/unit/test_multilayer_land.py:838:            precip_total=jnp.zeros(ncol),
tests/land/unit/test_multilayer_land.py:839:            precip_snow=jnp.zeros(ncol),
tests/land/unit/test_multilayer_land.py:850:            has_precipitation=jnp.ones(ncol),
tests/unit/test_ensemble_diagnostics.py:138:            precip_accum=jnp.zeros(s2),
tests/stress/test_phase1_land_carbon.py:59:        precip = jnp.full(shape, 5e-5)
tests/stress/test_phase1_land_carbon.py:75:                state, sw_down, T, co2, beta, lat, doy, precip, config, dt,
tests/stress/test_phase1_land_carbon.py:105:        precip = jnp.full(shape, 5e-5)
tests/stress/test_phase1_land_carbon.py:112:            state, sw_down, T_cold, co2, beta, lat, doy, precip, config, dt,
tests/stress/test_phase1_land_carbon.py:115:            state, sw_down, T_warm, co2, beta, lat, doy, precip, config, dt,
tests/ocean/unit/test_ocean_differentiability.py:163:            return jnp.sum(tendencies.dT_dt.data ** 2)
tests/ocean/unit/test_ocean_differentiability.py:264:            return jnp.sum(tendencies.dT_dt.data ** 2)
tests/ocean/unit/test_ocean_differentiability.py:280:            return jnp.sum(tendencies.dT_dt.data ** 2)
tests/ocean/unit/test_ocean_differentiability.py:285:            return jnp.sum(tendencies.dT_dt.data ** 2)
tests/ocean/unit/test_ocean_differentiability.py:588:            return jnp.sum(t.dT_dt.data ** 2)
tests/unit/test_moisture_budget.py:30:        precip_1 = jnp.zeros((6, 4, 4))              # no precip
tests/unit/test_moisture_budget.py:31:        tracker.update(q_v_1, p_s_1, dsigma, precip_1, elapsed_seconds=0.0)
tests/unit/test_moisture_budget.py:39:        precip_2 = jnp.full((6, 4, 4), 1.18e-3)     # matching precip
tests/unit/test_moisture_budget.py:40:        tracker.update(q_v_2, p_s_2, dsigma, precip_2, elapsed_seconds=86400.0)
tests/unit/test_moisture_budget.py:44:        precip_3 = jnp.zeros((6, 4, 4))
tests/unit/test_moisture_budget.py:45:        tracker.update(q_v_3, p_s_2, dsigma, precip_3, elapsed_seconds=172800.0)
tests/unit/test_moisture_budget.py:56:        precip = jnp.zeros((6, 4, 4))
tests/unit/test_moisture_budget.py:58:        budget = tracker.update(q_v, p_s, dsigma, precip, 0.0)
tests/unit/test_moisture_budget.py:64:    def test_precip_in_mm_per_day(self):
tests/unit/test_moisture_budget.py:73:        precip = jnp.full((2, 2), 1.0 / 86400.0)  # 1 mm/day
tests/unit/test_moisture_budget.py:75:        budget = tracker.update(q_v, p_s, dsigma, precip, 0.0)
tests/unit/test_moisture_budget.py:76:        np.testing.assert_allclose(budget.precip_rate, 1.0, rtol=1e-4)
tests/unit/test_moisture_budget.py:114:        # Constant moisture, zero precip → residual = 0
tests/unit/test_moisture_budget.py:116:        precip = jnp.zeros((4, 4))
tests/unit/test_moisture_budget.py:119:            tracker.update(q_v, p_s, dsigma, precip, float(i * 86400))
tests/land/unit/test_land_water_budget.py:30:        sw_down=200.0, lw_down=300.0, precip_total=1e-5, precip_snow=0.0,
tests/land/unit/test_land_water_budget.py:33:        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
tests/land/unit/test_land_water_budget.py:54:    """When W=0 and precip=0, evaporation must be water-limited to zero."""
tests/land/unit/test_land_water_budget.py:57:        """LH flux must be zero when the bucket is empty and there is no precip."""
tests/land/unit/test_land_water_budget.py:65:        forcing = _make_forcing(ncol, precip_total=0.0, precip_snow=0.0,
tests/land/unit/test_land_water_budget.py:79:        """Evaporation should not exceed available water + precip over dt."""
tests/land/unit/test_land_water_budget.py:88:        # Hot surface → strong evaporative demand, no precip
tests/land/unit/test_land_water_budget.py:89:        forcing = _make_forcing(ncol, precip_total=0.0, precip_snow=0.0,
tests/land/unit/test_land_water_budget.py:111:        forcing = _make_forcing(ncol, precip_total=0.0, precip_snow=0.0,
tests/land/unit/test_land_water_budget.py:142:        forcing = _make_forcing(ncol, precip_total=1e-2)
tests/land/unit/test_land_water_budget.py:161:        forcing = _make_forcing(ncol, precip_total=5e-3, precip_snow=0.0)
tests/land/unit/test_land_water_budget.py:165:        precip = float(forcing.precip_total[0])
tests/land/unit/test_land_water_budget.py:171:        budget_residual = abs(W_new - W_old - (precip - evap - runoff) * dt)
tests/land/unit/test_land_water_budget.py:183:        forcing = _make_forcing(ncol, precip_total=1e-5)
tests/unit/test_land_ice_units.py:36:        precip_total=jnp.zeros(s, f), precip_snow=jnp.zeros(s, f),
tests/unit/test_land_ice_units.py:42:        has_radiation=jnp.ones(s, f), has_precipitation=jnp.ones(s, f),
tests/validation/test_ensemble_correctness.py:76:        dT_dt=jnp.zeros(T.shape),
tests/validation/test_ensemble_correctness.py:77:        dq_v_dt=jnp.zeros(T.shape),
tests/validation/test_ensemble_correctness.py:80:        precip=jnp.zeros(p_s.shape),
tests/validation/test_ensemble_correctness.py:132:    kwargs["dT_dt"] = jnp.full(shape_3d, 1e-5)
tests/validation/test_ensemble_correctness.py:311:            T_upd = T_new + _dt * phys_out.dT_dt
tests/validation/test_ensemble_correctness.py:312:            q_v_upd = jnp.maximum(carry.q_v + _dt * phys_out.dq_v_dt, 0.0)
tests/validation/test_ensemble_correctness.py:338:            precip_step = (phys_out.precip
tests/validation/test_ensemble_correctness.py:339:                           if hasattr(phys_out, "precip")
tests/validation/test_ensemble_correctness.py:341:            precip_accum = carry.precip_accum + precip_step * _dt
tests/validation/test_ensemble_correctness.py:368:                precip_accum=_match_dtype(precip_accum, carry.precip_accum),
tests/land/unit/test_carbon_cycle.py:220:        """At reference T and precip, modifier should be near 1."""
tests/land/unit/test_carbon_cycle.py:223:        precip = jnp.array([cfg.precip_ref])
tests/land/unit/test_carbon_cycle.py:224:        mod = _temperate_modifier(T, precip, cfg)
tests/land/unit/test_carbon_cycle.py:230:        precip = jnp.array([cfg.precip_ref])
tests/land/unit/test_carbon_cycle.py:231:        mod_cold = _temperate_modifier(jnp.array([270.0]), precip, cfg)
tests/land/unit/test_carbon_cycle.py:232:        mod_warm = _temperate_modifier(jnp.array([300.0]), precip, cfg)
tests/land/unit/test_carbon_cycle.py:256:        precip = jnp.full(ncol, 3e-5)
tests/land/unit/test_carbon_cycle.py:261:            state, sw, T, co2, beta, lat, 150.0, precip, cfg, dt,
tests/land/unit/test_carbon_cycle.py:292:        precip = jnp.full(ncol, 3e-5)
tests/land/unit/test_carbon_cycle.py:296:            precip, cfg, 600.0,
tests/land/unit/test_carbon_cycle.py:310:        precip = jnp.full(ncol, 3e-5)
tests/land/unit/test_carbon_cycle.py:314:            precip, cfg, 600.0,
tests/land/unit/test_carbon_cycle.py:327:        precip = jnp.full(ncol, 3e-5)
tests/land/unit/test_carbon_cycle.py:333:                state, sw, T, co2, beta, lat, doy, precip, cfg, 600.0,
tests/land/unit/test_carbon_cycle.py:348:        precip = jnp.full(ncol, 3e-5)
tests/land/unit/test_carbon_cycle.py:354:            state, sw, T, co2, beta, lat, 150.0, precip, cfg, dt,
tests/land/unit/test_carbon_cycle.py:437:            precip=jnp.full(ncol, 3e-5),
tests/land/unit/test_carbon_cycle.py:513:            precip_total=jnp.full(shape, 3e-5),
tests/land/unit/test_carbon_cycle.py:514:            precip_snow=jnp.zeros(shape),
tests/land/unit/test_carbon_cycle.py:525:            has_precipitation=jnp.ones(shape),
tests/land/unit/test_carbon_cycle.py:561:            precip_total=jnp.full(shape, 3e-5),
tests/land/unit/test_carbon_cycle.py:562:            precip_snow=jnp.zeros(shape),
tests/land/unit/test_carbon_cycle.py:573:            has_precipitation=jnp.ones(shape),
tests/land/unit/test_stomata.py:383:            precip=jnp.full(shape, 3e-5),
tests/land/unit/test_stomata.py:426:            precip_total=jnp.full(shape, 3e-5),
tests/land/unit/test_stomata.py:427:            precip_snow=jnp.zeros(shape),
tests/land/unit/test_stomata.py:438:            has_precipitation=True,
tests/unit/test_earth_system_driver.py:87:            precip_total=zeros,
tests/unit/test_earth_system_driver.py:88:            precip_snow=zeros,
tests/unit/test_earth_system_driver.py:99:            has_precipitation=zeros,
tests/unit/test_earth_system_driver.py:102:        assert hasattr(forcing, 'has_precipitation')
tests/unit/test_earth_system_driver.py:105:    def test_missing_has_precipitation_raises(self):
tests/unit/test_earth_system_driver.py:106:        """Omitting has_precipitation must raise TypeError, not silently succeed."""
tests/unit/test_earth_system_driver.py:117:                precip_total=zeros,
tests/unit/test_earth_system_driver.py:118:                precip_snow=zeros,
tests/unit/test_earth_system_driver.py:129:                # has_precipitation intentionally missing
tests/ocean/unit/test_mpas_ocean.py:120:            dT_dt=Field(jnp.zeros((nCells, nlev)), "dT_dt", ("nCells", "nlev"), "degC/s"),
tests/ocean/unit/test_mpas_ocean.py:234:        assert tend.dT_dt.data.shape == state.T.data.shape
tests/ocean/unit/test_mpas_ocean.py:242:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/ocean/unit/test_mpas_ocean.py:260:            assert jnp.allclose(tend.dT_dt.data[land_cells], 0.0)
tests/ocean/unit/test_mpas_ocean.py:386:        assert jnp.allclose(tend.dT_dt.data, expected_dT, atol=0.0, rtol=0.0)
tests/ocean/unit/test_mpas_ocean.py:398:        diff = jnp.max(jnp.abs(t_on.dT_dt.data - t_off.dT_dt.data))
tests/ocean/unit/test_mpas_ocean.py:400:        assert jnp.all(jnp.isfinite(t_on.dT_dt.data))
tests/ocean/unit/test_mpas_ocean.py:411:            assert jnp.allclose(tend.dT_dt.data[land], 0.0)
tests/ocean/unit/test_mpas_ocean.py:432:        delta = t_on.dT_dt.data - t_off.dT_dt.data
tests/ocean/unit/test_mpas_ocean.py:1075:        assert tend.dT_dt.data.shape == state.T.data.shape
tests/land/unit/test_land_audit_fixes.py:32:        sw_down=200.0, lw_down=300.0, precip_total=1e-5, precip_snow=0.0,
tests/land/unit/test_land_audit_fixes.py:35:        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
tests/land/unit/test_land_audit_fixes.py:90:                                sw_down=100.0, precip_total=0.0, precip_snow=0.0)
tests/land/unit/test_land_audit_fixes.py:111:                                precip_total=0.0, precip_snow=0.0)
tests/land/unit/test_land_audit_fixes.py:134:                                precip_total=0.0, precip_snow=0.0)
tests/land/unit/test_land_audit_fixes.py:381:                                precip_total=0.0, precip_snow=0.0)
tests/unit/test_physics_grid_adapters.py:377:    dT_dt_rad = jnp.zeros(shape_3d, dtype=jnp.float32)
tests/unit/test_physics_grid_adapters.py:386:        dT_dt_rad, sw_net_sfc, lw_net_sfc,
tests/unit/test_physics_grid_adapters.py:399:        assert out.dT_dt.shape == shape_3d
tests/unit/test_physics_grid_adapters.py:400:        assert out.dq_v_dt.shape == shape_3d
tests/unit/test_physics_grid_adapters.py:403:        assert out.precip.shape == shape_2d
tests/unit/test_physics_grid_adapters.py:407:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_physics_grid_adapters.py:408:        assert jnp.all(jnp.isfinite(out.dq_v_dt))
tests/unit/test_physics_grid_adapters.py:409:        assert jnp.all(jnp.isfinite(out.precip))
tests/unit/test_physics_grid_adapters.py:419:        assert out.dT_dt.shape == shape_3d
tests/unit/test_physics_grid_adapters.py:420:        assert out.dq_v_dt.shape == shape_3d
tests/unit/test_physics_grid_adapters.py:421:        assert out.precip.shape == shape_2d
tests/unit/test_physics_grid_adapters.py:425:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_physics_grid_adapters.py:426:        assert jnp.all(jnp.isfinite(out.precip))
tests/unit/test_physics_grid_adapters.py:434:        assert out.dT_dt.shape == (1, NLEV)
tests/unit/test_physics_grid_adapters.py:435:        assert out.dq_v_dt.shape == (1, NLEV)
tests/unit/test_physics_grid_adapters.py:436:        assert out.precip.shape == (1,)
tests/unit/test_physics_grid_adapters.py:440:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_physics_grid_adapters.py:441:        assert jnp.all(jnp.isfinite(out.precip))
tests/unit/test_physics_grid_adapters.py:503:            dT_col = ad.flatten_3d(out.dT_dt)[0]
tests/unit/test_physics_grid_adapters.py:504:            dqv_col = ad.flatten_3d(out.dq_v_dt)[0]
tests/unit/test_physics_grid_adapters.py:505:            precip_col = ad.flatten_2d(out.precip)[0]
tests/unit/test_physics_grid_adapters.py:507:            results[grid_name] = (dT_col, dqv_col, precip_col)
tests/unit/test_physics_grid_adapters.py:512:            err_msg="dT_dt mismatch: cubed-sphere vs lat-lon",
tests/unit/test_physics_grid_adapters.py:516:            err_msg="dq_v_dt mismatch: cubed-sphere vs lat-lon",
tests/unit/test_physics_grid_adapters.py:520:            err_msg="precip mismatch: cubed-sphere vs lat-lon",
tests/unit/test_physics_grid_adapters.py:526:            err_msg="dT_dt mismatch: cubed-sphere vs single-column",
tests/unit/test_physics_grid_adapters.py:530:            err_msg="dq_v_dt mismatch: cubed-sphere vs single-column",
tests/unit/test_physics_grid_adapters.py:581:        assert phys_out.dT_dt.shape == shape_3d
tests/unit/test_physics_grid_adapters.py:582:        assert jnp.all(jnp.isfinite(phys_out.dT_dt))
tests/unit/test_physics_grid_adapters.py:623:        assert phys_out.dT_dt.shape == shape_3d
tests/unit/test_physics_grid_adapters.py:624:        assert jnp.all(jnp.isfinite(phys_out.dT_dt))
tests/ocean/unit/test_ocean_compatibility.py:351:            return jnp.sum(tend.dT_dt.data ** 2)
tests/distributed/test_coupler_mpi.py:56:    precip_total = _pattern(
tests/distributed/test_coupler_mpi.py:71:        precip_total=precip_total,
tests/distributed/test_coupler_mpi.py:72:        precip_snow=zero,
tests/distributed/test_coupler_mpi.py:83:        has_precipitation=jnp.array(1.0, dtype=jnp.float32),
tests/ocean/unit/test_latlon_cgrid_ocean.py:133:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/ocean/unit/test_latlon_cgrid_ocean.py:158:        assert tend.dT_dt.data.shape == (n_lat, n_lon, nlev)
tests/ocean/unit/test_freshwater.py:65:        assert fw.precip.shape == (100,)
tests/ocean/unit/test_freshwater.py:72:        assert jnp.all(fw.precip == 0.0)
tests/ocean/unit/test_freshwater.py:82:    def test_net_flux_precip_only(self):
tests/ocean/unit/test_freshwater.py:85:            precip=jnp.ones(n) * 1e-5,
tests/ocean/unit/test_freshwater.py:95:        P = jnp.ones(n) * 3e-5   # precip in
tests/ocean/unit/test_freshwater.py:99:        fw = FreshwaterForcing(precip=P, evap=E, runoff=R, ice_fw=M)
tests/ocean/unit/test_freshwater.py:107:            precip=jnp.zeros(n),
tests/ocean/unit/test_freshwater.py:132:    def test_eta_tendency_precip_positive(self):
tests/ocean/unit/test_freshwater.py:136:            precip=jnp.ones(n) * 1e-3,  # 1 mm/s
tests/ocean/unit/test_freshwater.py:151:            precip=jnp.zeros(n),
tests/ocean/unit/test_freshwater.py:178:    def test_vsf_precip_dilutes(self):
tests/ocean/unit/test_freshwater.py:182:            precip=jnp.ones(n) * 1e-4,
tests/ocean/unit/test_freshwater.py:195:            precip=jnp.zeros(n),
tests/ocean/unit/test_freshwater.py:212:            precip=jnp.ones(n) * P,
tests/ocean/unit/test_freshwater.py:225:            precip=jnp.ones(n) * 1e-4,
tests/ocean/unit/test_freshwater.py:243:        precip = jnp.ones(n) * 1e-4
tests/ocean/unit/test_freshwater.py:247:        fw = freshwater_from_coupler(precip, lhflx, L_v, ocean_mask=mask)
tests/ocean/unit/test_freshwater.py:248:        assert fw.precip.shape == (n,)
tests/ocean/unit/test_freshwater.py:249:        assert jnp.allclose(fw.precip, 1e-4)
tests/ocean/unit/test_freshwater.py:256:        precip = jnp.ones(n) * 1e-4
tests/ocean/unit/test_freshwater.py:259:        fw = freshwater_from_coupler(precip, lhflx, 2.5e6, ocean_mask=mask)
tests/ocean/unit/test_freshwater.py:260:        assert jnp.all(fw.precip[5:] == 0.0)
tests/ocean/unit/test_freshwater.py:261:        assert jnp.all(fw.precip[:5] > 0.0)
tests/ocean/unit/test_freshwater.py:265:        precip = jnp.zeros(n)
tests/ocean/unit/test_freshwater.py:270:            precip, lhflx, 2.5e6,
tests/ocean/unit/test_freshwater.py:461:    def test_step_with_precip_raises_eta(self, mesh, z_coord, state0):
tests/ocean/unit/test_freshwater.py:470:        # Uniform 1 mm/s precip over ocean
tests/ocean/unit/test_freshwater.py:472:            precip=jnp.ones(mesh.nCells) * 1e-3 * mask,
tests/ocean/unit/test_freshwater.py:481:        # Mean ocean eta should be higher with precip
tests/ocean/unit/test_freshwater.py:487:    def test_step_with_precip_decreases_S(self, mesh, z_coord, state0):
tests/ocean/unit/test_freshwater.py:497:            precip=jnp.ones(mesh.nCells) * 1e-3 * mask,
tests/ocean/unit/test_freshwater.py:506:        # Mean top-layer S should be lower with precip
tests/ocean/unit/test_freshwater.py:522:            precip=jnp.zeros(mesh.nCells),
tests/ocean/unit/test_freshwater.py:545:        # Typical precip rate: ~3 mm/day = 3.5e-8 kg/m2/s
tests/ocean/unit/test_freshwater.py:547:            precip=jnp.ones(mesh.nCells) * 3.5e-8 * mask,
tests/ocean/unit/test_freshwater.py:574:            precip=jnp.ones(mesh.nCells) * 1e-2 * mask,
tests/ocean/unit/test_freshwater.py:636:            precip=jnp.zeros((n_lat, n_lon)),
tests/ocean/unit/test_freshwater.py:646:    def test_step_with_precip_raises_eta(self, ll_grid, ll_z_coord, ll_state0):
tests/ocean/unit/test_freshwater.py:657:            precip=jnp.ones((n_lat, n_lon)) * 1e-3 * mask,
tests/ocean/unit/test_freshwater.py:671:    def test_step_with_precip_decreases_S(self, ll_grid, ll_z_coord, ll_state0):
tests/ocean/unit/test_freshwater.py:682:            precip=jnp.ones((n_lat, n_lon)) * 1e-3 * mask,
tests/ocean/unit/test_freshwater.py:707:            precip=jnp.zeros((n_lat, n_lon)),
tests/ocean/unit/test_freshwater.py:732:            precip=jnp.ones((n_lat, n_lon)) * 3.5e-8 * mask,
tests/ocean/unit/test_freshwater.py:761:            precip=jnp.ones((n_lat, n_lon)) * 1e-2 * mask,
tests/ocean/unit/test_freshwater.py:781:        """Virtual salt flux should be differentiable w.r.t. precip."""
tests/ocean/unit/test_freshwater.py:784:        def loss(precip_val):
tests/ocean/unit/test_freshwater.py:786:                precip=jnp.ones(n) * precip_val,
tests/ocean/unit/test_freshwater.py:804:        def loss(precip_val):
tests/ocean/unit/test_freshwater.py:806:                precip=jnp.ones(n) * precip_val,
tests/ocean/unit/test_freshwater.py:840:            precip_total=ones * 1e-4,
tests/ocean/unit/test_freshwater.py:841:            precip_snow=z,
tests/ocean/unit/test_freshwater.py:844:            cos_zenith=z, co2_ppmv=z, has_radiation=z, has_precipitation=ones,
tests/ocean/unit/test_freshwater.py:856:        assert fw.precip.shape == (n,)
tests/ocean/unit/test_freshwater.py:857:        assert jnp.allclose(fw.precip, 1e-4)
tests/validation/test_differentiability_regression.py:77:            return jnp.mean(tend.dT_dt.data ** 2)
tests/validation/test_differentiability_regression.py:103:                return jnp.mean(tend.dT_dt.data ** 2)
tests/validation/test_differentiability_regression.py:126:            return jnp.mean(tend.dT_dt.data ** 2)
tests/validation/test_differentiability_regression.py:241:            precip_total=z, precip_snow=z,
tests/validation/test_differentiability_regression.py:252:            has_precipitation=jnp.array(1.0),
tests/ocean/unit/test_mpas_physics.py:83:        float(jnp.max(jnp.abs(tend.dT_dt.data))),
tests/ocean/unit/test_mpas_physics.py:106:        assert dT_max > 0.0, "SST restoring branch must produce dT_dt"
tests/ocean/unit/test_mpas_physics.py:131:    def test_dT_dt_zero_on_pure_land_cells(self, mesh, z_coord, state):
tests/ocean/unit/test_mpas_physics.py:142:            land_dT = tend.dT_dt.data[land_idx, 0]
tests/ocean/unit/test_mpas_physics.py:175:        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) == 0.0
tests/ocean/unit/test_mpas_physics.py:187:        dT_top = tend.dT_dt.data[:, 0]
tests/ocean/unit/test_mpas_physics.py:210:        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) < 1e-12
tests/ocean/unit/test_mpas_physics.py:229:            assert float(jnp.max(jnp.abs(tend.dT_dt.data[land_idx]))) == 0.0
tests/unit/test_land_ice_multilayer.py:41:    co2_ppmv=415.0, precip_total=0.0,
tests/unit/test_land_ice_multilayer.py:42:    precip_snow=0.0,
tests/unit/test_land_ice_multilayer.py:49:        precip_total=jnp.full(s, precip_total, f),
tests/unit/test_land_ice_multilayer.py:50:        precip_snow=jnp.full(s, precip_snow, f),
tests/unit/test_land_ice_multilayer.py:61:        has_precipitation=jnp.full(s, 1.0, f),
tests/unit/test_emanuel.py:71:    assert out.dT_dt.shape == (ncol, nlev)
tests/unit/test_emanuel.py:72:    assert out.dq_v_dt.shape == (ncol, nlev)
tests/unit/test_emanuel.py:73:    assert out.dq_c_conv_dt.shape == (ncol, nlev)
tests/unit/test_emanuel.py:83:    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
tests/unit/test_emanuel.py:84:                out.convective_mask, cpp_new):
tests/unit/test_emanuel.py:98:    """The convective cloud-water source is non-negative even after
tests/unit/test_emanuel.py:105:    assert jnp.all(out.dq_c_conv_dt >= 0.0)
tests/unit/test_emanuel.py:122:        assert jnp.all(jnp.isfinite(out.dT_dt)), (
tests/unit/test_emanuel.py:143:    mag_no = float(jnp.sum(jnp.abs(out_no_sort.dT_dt)))
tests/unit/test_emanuel.py:144:    mag_yes = float(jnp.sum(jnp.abs(out_strong_sort.dT_dt)))
tests/unit/test_emanuel.py:154:    sub-cloud cooling term — the difference in ``dT_dt`` between the
tests/unit/test_emanuel.py:170:    diff = jnp.abs(out_on.dT_dt[:, -4:] - out_off.dT_dt[:, -4:])
tests/unit/test_emanuel.py:186:        return jnp.sum(out.dT_dt)
tests/unit/test_emanuel.py:200:        return jnp.sum(out.dT_dt)
tests/unit/test_emanuel.py:246:    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
tests/unit/test_emanuel.py:278:    H = float(jnp.sum(out.dT_dt * dp / constants.g, axis=1).mean()) * constants.c_pd
tests/unit/test_emanuel.py:279:    Q = float(jnp.sum(out.dq_v_dt * dp / constants.g, axis=1).mean()) * constants.L_v
tests/unit/test_emanuel.py:280:    C = float(jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1).mean()) * constants.L_v
tests/unit/test_rce_script.py:79:        assert jnp.all(jnp.isfinite(conv.dT_dt))
tests/unit/test_rce_script.py:80:        # Post-Option-C: convection emits a 3D ``dq_c_conv_dt``
tests/unit/test_rce_script.py:81:        # cloud-water source instead of a scalar surface precip;
tests/unit/test_rce_script.py:83:        assert conv.dq_c_conv_dt.shape == (ncol, NLEV)
tests/unit/test_rce_script.py:84:        assert jnp.all(conv.dq_c_conv_dt >= 0)
tests/ocean/unit/test_gm_redi_eady_physics.py:355:        dT_dt, dS_dt = gm_redi_tracer_tendency_latlon(
tests/ocean/unit/test_gm_redi_eady_physics.py:362:        max_dT = float(jnp.max(jnp.abs(dT_dt)))
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:308:        precip_rate = dq * dp / (g * rho_water * dt)
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:311:        precip_rate = 0.0
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:312:    return T_new, q_new, precip_rate
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:333:T, q, precip = large_scale_condensation(T, q, p, dt_phys)
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:397:> **What's tested:** Advanced NH dynamics with Kessler microphysics and physics-dynamics coupling at convective scales. All source code in the DCMIP2016 GitHub repo.
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:491:!   - Horizontally uniform environment with convective instability
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:581:| 4-2, 4-3 | Same as 4-1 + `q, precip` |
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:582:| 5-1, 5-2 | `ps, T, u, v, w, q, precip` every 6 h for 10 days |
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md:584:| D16-2 | `ps, T, u, v, w, q, qc, qr, precip` every 6 h |
tests/validation/test_conservation_baseline.py:76:        # a radiative-convective equilibrium, so R_TOA - dE/dt is large (~300 W/m²).
tests/validation/test_conservation_baseline.py:103:            "days", "T_atm", "max_wind", "precip", "CWV",
tests/validation/test_scaling_readiness.py:46:        dT_dt=jnp.zeros(T.shape),
tests/validation/test_scaling_readiness.py:47:        dq_v_dt=jnp.zeros(T.shape),
tests/validation/test_scaling_readiness.py:50:        precip=jnp.zeros(p_s.shape),
tests/validation/test_scaling_readiness.py:83:            precip_accum=jnp.zeros(s2),
tests/validation/test_scaling_readiness.py:149:            precip_accum=jnp.zeros(s2),
tests/ocean/unit/test_visbeck_gm.py:324:        assert out.dT_dt.shape == shape
tests/ocean/unit/test_visbeck_gm.py:326:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/ocean/unit/test_visbeck_gm.py:342:        assert jnp.allclose(o1.dT_dt, o2.dT_dt, rtol=0, atol=0)
tests/ocean/unit/test_visbeck_gm.py:375:        max_diff = float(jnp.max(jnp.abs(o1.dT_dt - o2.dT_dt)))
tests/ocean/unit/test_ocean.py:484:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/ocean/unit/test_ocean.py:509:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/unit/test_sea_ice_dynamics.py:78:        precip_total=jnp.zeros(shape),
tests/unit/test_sea_ice_dynamics.py:79:        precip_snow=jnp.zeros(shape),
tests/unit/test_sea_ice_dynamics.py:90:        has_precipitation=jnp.ones(shape),
tests/ocean/unit/test_gm_redi_mpas.py:448:    (dT_dt, dS_dt) with the expected shapes; output is finite."""
tests/ocean/unit/test_gm_redi_mpas.py:467:    dT_dt, dS_dt = gm_redi_tracer_tendency_mpas(
tests/ocean/unit/test_gm_redi_mpas.py:472:    assert dT_dt.shape == (mesh.nCells, nlev)
tests/ocean/unit/test_gm_redi_mpas.py:474:    assert np.all(np.isfinite(np.asarray(dT_dt)))
tests/ocean/unit/test_ocean_fv.py:59:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/unit/test_diagnostic_collector.py:55:        collector.precip.append(3.0)
tests/unit/test_diagnostic_collector.py:86:        collector.precip.append(3.0)
tests/unit/test_diagnostic_collector.py:110:        collector.precip.extend([3.0, 3.1])
tests/ocean/unit/test_shortwave_penetration.py:70:    dT_dt = shortwave_penetration_tendency(
tests/ocean/unit/test_shortwave_penetration.py:74:    assert dT_dt.shape == expected_shape
tests/ocean/unit/test_shortwave_penetration.py:75:    assert jnp.all(jnp.isfinite(dT_dt))
tests/ocean/unit/test_shortwave_penetration.py:77:    assert jnp.all(dT_dt >= 0.0)
tests/ocean/unit/test_shortwave_penetration.py:82:    dT_dt = shortwave_penetration_tendency(
tests/ocean/unit/test_shortwave_penetration.py:86:    top = float(jnp.mean(dT_dt[..., 0]))
tests/ocean/unit/test_shortwave_penetration.py:87:    bottom = float(jnp.mean(dT_dt[..., -1]))
tests/unit/test_corrections.py:82:        """eps_gwd should equal integrated dT_dt * c_p * rho * dz."""
tests/unit/test_corrections.py:90:        heating_power = jnp.sum(rho * out.dT_dt * constants.c_pd * dz, axis=1)
tests/unit/test_corrections.py:272:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_corrections.py:288:        assert out.dT_dt.shape == shape
tests/unit/test_corrections.py:329:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_corrections.py:352:        assert not jnp.allclose(out_eq.dT_dt, out_uneq.dT_dt, atol=1e-20)
tests/unit/test_corrections.py:393:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_corrections.py:406:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_corrections.py:500:        # dT_dt should only come from diffusion, not nonlocal
tests/unit/test_corrections.py:502:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/unit/test_corrections.py:516:        diff = jnp.max(jnp.abs(out_unstable.dT_dt - out_stable.dT_dt))
tests/unit/test_corrections.py:519:    def test_convective_instability_enhances_mixing(self):
tests/unit/test_corrections.py:577:        assert jnp.all(jnp.isfinite(out_neg.dT_dt))
tests/unit/test_corrections.py:578:        assert jnp.all(jnp.isfinite(out_pos.dT_dt))
tests/unit/test_corrections.py:581:        diff = float(jnp.max(jnp.abs(out_pos.dT_dt - out_neg.dT_dt)))
tests/unit/test_corrections.py:602:        diff = float(jnp.max(jnp.abs(out_imposed.dT_dt - out_diag.dT_dt)))
tests/unit/test_diff_coupler.py:166:            precip_total=1e-5 * ones,
tests/unit/test_diff_coupler.py:167:            precip_snow=0.0 * ones,
tests/unit/test_diff_coupler.py:178:            has_precipitation=1.0 * ones,
tests/unit/test_diff_coupler.py:332:            precip_total=1e-5 * ones,
tests/unit/test_diff_coupler.py:333:            precip_snow=0.0 * ones,
tests/unit/test_diff_coupler.py:344:            has_precipitation=1.0 * ones,
tests/unit/test_zhang_mcfarlane.py:7:* moisture-conservation budget (vapor + cloud-water source vs precip);
tests/unit/test_zhang_mcfarlane.py:97:    assert out.dT_dt.shape == (ncol, nlev)
tests/unit/test_zhang_mcfarlane.py:98:    assert out.dq_v_dt.shape == (ncol, nlev)
tests/unit/test_zhang_mcfarlane.py:99:    assert out.dq_c_conv_dt.shape == (ncol, nlev)
tests/unit/test_zhang_mcfarlane.py:101:    assert out.convective_mask.shape == (ncol,)
tests/unit/test_zhang_mcfarlane.py:116:    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
tests/unit/test_zhang_mcfarlane.py:117:                out.convective_mask, out.du_dt_conv, out.dv_dt_conv,
tests/unit/test_zhang_mcfarlane.py:123:    """Convective condensate source ``dq_c_conv_dt`` is non-negative
tests/unit/test_zhang_mcfarlane.py:129:    assert jnp.all(out.dq_c_conv_dt >= -1e-12)
tests/unit/test_zhang_mcfarlane.py:196:    upper_mean = float(jnp.mean(out.dT_dt[0, :nlev_half]))
tests/unit/test_zhang_mcfarlane.py:197:    lower_mean = float(jnp.mean(out.dT_dt[0, nlev_half:]))
tests/unit/test_zhang_mcfarlane.py:202:    column_integrated_dT_dt = float(jnp.sum(out.dT_dt[0]))
tests/unit/test_zhang_mcfarlane.py:206:    assert column_integrated_dT_dt >= -1e-3, (
tests/unit/test_zhang_mcfarlane.py:207:        f"Column-integrated dT/dt = {column_integrated_dT_dt:.2e}; "
tests/unit/test_zhang_mcfarlane.py:217:    """``d (sum dT_dt) / d tau_cape`` is finite — relaxation timescale
tests/unit/test_zhang_mcfarlane.py:228:        return jnp.sum(out.dT_dt)
tests/unit/test_zhang_mcfarlane.py:235:    """``d (sum dT_dt) / d cape_threshold`` is finite even when the
tests/unit/test_zhang_mcfarlane.py:249:        return jnp.sum(out.dT_dt)
tests/unit/test_zhang_mcfarlane.py:318:    assert float(jnp.max(jnp.abs(out.dT_dt))) < 1e-3
tests/unit/test_zhang_mcfarlane.py:369:    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
tests/unit/test_zhang_mcfarlane.py:386:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/unit/test_zhang_mcfarlane.py:426:    H = float(jnp.sum(out.dT_dt * dp / constants.g, axis=1).mean()) * constants.c_pd
tests/unit/test_zhang_mcfarlane.py:427:    Q = float(jnp.sum(out.dq_v_dt * dp / constants.g, axis=1).mean()) * constants.L_v
tests/unit/test_zhang_mcfarlane.py:428:    C = float(jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1).mean()) * constants.L_v
tests/unit/test_diff_coupled_system.py:84:            T_final = s.T.data + dt * tend.dT_dt.data
tests/unit/test_diff_coupled_system.py:118:                precip_total=1e-5 * ones, precip_snow=0.0 * ones,
tests/unit/test_diff_coupled_system.py:124:                has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
tests/unit/test_diff_coupled_system.py:197:                precip_total=0.0 * ones,
tests/unit/test_diff_coupled_system.py:198:                precip_snow=0.0 * ones,
tests/unit/test_diff_coupled_system.py:209:                has_precipitation=1.0 * ones,
tests/unit/test_diff_coupled_system.py:287:            T_after_phys = s.T.data + dt * tend.dT_dt.data
tests/unit/test_diff_coupled_system.py:292:                precip_total=1e-5 * ones, precip_snow=0.0 * ones,
tests/unit/test_diff_coupled_system.py:298:                has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
tests/unit/test_land_ice_carbon.py:122:        precip = jnp.full(SHAPE, 3e-5)
tests/unit/test_land_ice_carbon.py:125:            state, sw, T, co2, beta, lat, 200.0, precip, CONFIG, dt,
tests/unit/test_land_ice_carbon.py:175:        precip = jnp.full(SHAPE, 3e-5)
tests/unit/test_land_ice_carbon.py:180:                state, sw, T, co2, beta, lat, float(i % 365), precip, CONFIG, dt,
tests/unit/test_land_ice_carbon.py:202:        precip = jnp.full(SHAPE, 3e-5)
tests/unit/test_land_ice_carbon.py:206:            state, sw, T_cold, co2, beta, lat, 200.0, precip, CONFIG, dt,
tests/unit/test_land_ice_carbon.py:209:            state, sw, T_warm, co2, beta, lat, 200.0, precip, CONFIG, dt,
tests/unit/test_cdgrid.py:394:        self.assertEqual(tend.dT_dt.data.shape, (6, n, n, nlev))
tests/unit/test_cdgrid.py:410:        self.assertTrue(jnp.all(jnp.isfinite(tend.dT_dt.data)))
tests/unit/test_cdgrid.py:429:        self.assertLess(float(jnp.max(jnp.abs(tend.dT_dt.data))), 1e-8)
tests/unit/test_coupled_esm.py:140:            precip_total=jnp.zeros(shape),
tests/unit/test_coupled_esm.py:141:            precip_snow=jnp.zeros(shape),
tests/unit/test_coupled_esm.py:152:            has_precipitation=jnp.zeros(shape),
tests/unit/test_scale_tpu_compat.py:51:            precip_accum=jnp.zeros(shape2d),
tests/unit/test_compiled_segments.py:105:        dT_dt=jnp.full(shape_3d, 1e-5),    # small warming
tests/unit/test_compiled_segments.py:106:        dq_v_dt=jnp.zeros(shape_3d),
tests/unit/test_compiled_segments.py:109:        precip=jnp.zeros(shape_2d),
tests/unit/test_compiled_segments.py:229:        (new_state, qv_out, qc_out, qr_out, conv_prog_out, held_tuple, step_idx, precip,
tests/unit/test_compiled_segments.py:539:        T_upd = T_new + dt * phys_out.dT_dt
tests/unit/test_compiled_segments.py:540:        q_v_upd = jnp.maximum(carry.q_v + dt * phys_out.dq_v_dt, 0.0)
tests/unit/test_compiled_segments.py:565:        # Accumulate precipitation
tests/unit/test_compiled_segments.py:566:        precip_step = phys_out.precip if hasattr(phys_out, 'precip') else jnp.zeros_like(p_s_new)
tests/unit/test_compiled_segments.py:567:        precip_accum = carry.precip_accum + precip_step * dt
tests/unit/test_compiled_segments.py:584:            precip_accum=precip_accum,
tests/unit/test_issue_fixes.py:93:        assert jnp.allclose(tend.dT_dt.data, 0.0, atol=atol_wind), (
tests/unit/test_issue_fixes.py:94:            f"max |dT/dt| = {float(jnp.max(jnp.abs(tend.dT_dt.data))):.2e}")
tests/unit/test_issue_fixes.py:423:        assert jnp.all(jnp.isfinite(tend_cube.dT_dt.data))
tests/unit/test_issue_fixes.py:429:        assert jnp.all(jnp.isfinite(tend_ll.dT_dt.data))
tests/unit/test_kain_fritsch.py:81:    assert out.dT_dt.shape == (ncol, nlev)
tests/unit/test_kain_fritsch.py:82:    assert out.dq_v_dt.shape == (ncol, nlev)
tests/unit/test_kain_fritsch.py:83:    assert out.dq_c_conv_dt.shape == (ncol, nlev)
tests/unit/test_kain_fritsch.py:93:    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
tests/unit/test_kain_fritsch.py:94:                out.convective_mask):
tests/unit/test_kain_fritsch.py:103:    """KF does not produce convective momentum transport — by design."""
tests/unit/test_kain_fritsch.py:124:    assert float(jnp.max(jnp.abs(out.dT_dt))) < 1e-4
tests/unit/test_kain_fritsch.py:125:    assert float(out.convective_mask[0]) < 1e-4
tests/unit/test_kain_fritsch.py:209:    # With deep-only mode and a shallow cloud, dT_dt should be small.
tests/unit/test_kain_fritsch.py:210:    assert float(jnp.max(jnp.abs(out.dT_dt))) < 1e-3
tests/unit/test_kain_fritsch.py:225:    assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-6
tests/unit/test_kain_fritsch.py:279:    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
tests/unit/test_kain_fritsch.py:314:    H = float(jnp.sum(out.dT_dt * dp / constants.g, axis=1).mean()) * constants.c_pd
tests/unit/test_kain_fritsch.py:315:    Q = float(jnp.sum(out.dq_v_dt * dp / constants.g, axis=1).mean()) * constants.L_v
tests/unit/test_kain_fritsch.py:316:    C = float(jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1).mean()) * constants.L_v
tests/unit/test_plot_amip.py:26:        "precip": 3.0 + np.random.default_rng(4).standard_normal(n),
tests/unit/test_physics_surface_models.py:19:        sw_down=200.0, lw_down=300.0, precip_total=1e-5, precip_snow=0.0,
tests/unit/test_physics_surface_models.py:22:        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
tests/unit/test_physics_surface_models.py:68:        heavy_rain = make_forcing((16,), precip_total=1e-2)
tests/unit/test_physics_surface_models.py:79:        dry_forcing = make_forcing((16,), precip_total=0.0, T_lowest=300.0)
tests/unit/test_precision_dtype_contracts.py:184:        assert out.dT_dt.dtype == policy.storage
tests/unit/test_precision_dtype_contracts.py:185:        # Post-Option-C: ``ConvectionOutput.precipitation`` was replaced
tests/unit/test_precision_dtype_contracts.py:186:        # by the 3D ``dq_c_conv_dt`` field (cloud-water source rate);
tests/unit/test_precision_dtype_contracts.py:187:        # microphysics owns the surface precipitation diagnostic. The
tests/unit/test_precision_dtype_contracts.py:190:        assert out.dq_c_conv_dt.dtype == policy.storage
tests/unit/test_diff_taylor_tests.py:91:            return jnp.sum(tend.dT_dt.data ** 2)
tests/unit/test_diff_taylor_tests.py:120:            precip_total=1e-5 * ones, precip_snow=0.0 * ones,
tests/unit/test_diff_taylor_tests.py:126:            has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
tests/unit/test_diff_taylor_tests.py:160:            precip_total=0.0 * ones, precip_snow=0.0 * ones,
tests/unit/test_diff_taylor_tests.py:166:            has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
tests/unit/test_physics_ocean.py:57:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt has NaN/Inf"
tests/unit/test_physics_ocean.py:141:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt NaN/Inf"
tests/unit/test_coupler.py:48:                  precip=1e-5):
tests/unit/test_coupler.py:54:        precip_total=jnp.full(shape, precip),
tests/unit/test_coupler.py:55:        precip_snow=z,
tests/unit/test_coupler.py:66:        has_precipitation=jnp.array(1.0),
tests/unit/test_coupler.py:145:    # Saturated bucket with zero precip
tests/unit/test_coupler.py:147:    forcing = _make_forcing(precip=0.0)
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py:62:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/unit/test_physics_microphysics.py:4:Clapeyron), saturation adjustment, precipitation positivity, ice-phase
tests/unit/test_physics_microphysics.py:119:    """cp * dT_dt approx Lv * condensation_rate (first order).
tests/unit/test_physics_microphysics.py:127:    lhs = constants.c_pd * out.dT_dt
tests/unit/test_physics_microphysics.py:128:    rhs = -constants.L_v * out.dq_v_dt
tests/unit/test_physics_microphysics.py:145:    """Starting from supersaturated, vapor should decrease (dq_v_dt < 0)."""
tests/unit/test_physics_microphysics.py:151:    # Some levels should show condensation (dq_v_dt < 0)
tests/unit/test_physics_microphysics.py:152:    min_dqv = float(jnp.min(out.dq_v_dt))
tests/unit/test_physics_microphysics.py:154:        f"{scheme}: no condensation in supersaturated column, min dq_v_dt = {min_dqv:.2e}"
tests/unit/test_physics_microphysics.py:163:def test_precipitation_non_negative(scheme):
tests/unit/test_physics_microphysics.py:167:    min_precip = float(jnp.min(out.precipitation))
tests/unit/test_physics_microphysics.py:168:    assert min_precip >= -1e-15, (
tests/unit/test_physics_microphysics.py:169:        f"{scheme}: negative precipitation = {min_precip:.2e}"
tests/unit/test_physics_microphysics.py:228:    """|dT_dt| should be bounded (< 10 K/s for strongly supersaturated)."""
tests/unit/test_physics_microphysics.py:231:    max_hr = float(jnp.max(jnp.abs(out.dT_dt)))
tests/unit/test_physics_microphysics.py:233:        f"{scheme}: |dT_dt| = {max_hr:.2e} K/s exceeds 10 K/s"
tests/unit/test_diff_land.py:35:        precip_total=1e-5 * ones,
tests/unit/test_diff_land.py:36:        precip_snow=0.0 * ones,
tests/unit/test_diff_land.py:47:        has_precipitation=1.0 * ones,
tests/unit/test_diff_land.py:146:        precip_snow = 1e-5 * jnp.ones(ncol)
tests/unit/test_diff_land.py:150:                snow, snow_age, T_surface, precip_snow, dt=3600.0,
tests/unit/test_physics_convection.py:4:profile behavior, precipitation sign, and tendency signs for all convection
tests/unit/test_physics_convection.py:21:    stratosphere_mass_flux_gate,
tests/unit/test_physics_convection.py:35:    """Build a convectively unstable column: warm, moist BL under cool mid-trop."""
tests/unit/test_physics_convection.py:98:    Convection emits a 3D ``dq_c_conv_dt`` (rate of cloud-water creation
tests/unit/test_physics_convection.py:99:    at each level) instead of a scalar surface ``precipitation``;
tests/unit/test_physics_convection.py:104:        ∫ dq_v_dt dp/g  +  ∫ dq_c_conv_dt dp/g  ≈  0
tests/unit/test_physics_convection.py:108:    (= the legacy ``precipitation`` formula). This guarantees
tests/unit/test_physics_convection.py:110:    sign pattern of ``dq_v_dt`` — without per-level negative cloud
tests/unit/test_physics_convection.py:111:    water source. (A naive per-level ``max(-dq_v_dt, 0)`` would
tests/unit/test_physics_convection.py:125:    ``dq_v_dt`` is not the negative of the column condensation by
tests/unit/test_physics_convection.py:133:    col_dqv = jnp.sum(out.dq_v_dt * dp / constants.g, axis=1)
tests/unit/test_physics_convection.py:134:    col_dqc = jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1)
tests/unit/test_physics_convection.py:163:    mse_tend = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
tests/unit/test_physics_convection.py:187:    max_dT = float(jnp.max(jnp.abs(out.dT_dt)))
tests/unit/test_physics_convection.py:191:    # (broken) mass-flux kernel that returned ``dq_c_conv_dt ≈ 0`` for
tests/unit/test_physics_convection.py:195:    max_dq_c = float(jnp.max(out.dq_c_conv_dt))
tests/unit/test_physics_convection.py:197:    assert max_dT < 1e-2, f"{scheme}: dT_dt = {max_dT:.2e} in stable column"
tests/unit/test_physics_convection.py:199:        f"{scheme}: dq_c_conv_dt = {max_dq_c:.2e} kg/kg/s in stable column"
tests/unit/test_physics_convection.py:208:def test_precipitation_non_negative(scheme):
tests/unit/test_physics_convection.py:212:    min_dq_c = float(jnp.min(out.dq_c_conv_dt))
tests/unit/test_physics_convection.py:214:        f"{scheme}: negative dq_c_conv_dt = {min_dq_c:.2e}"
tests/unit/test_physics_convection.py:233:    T_after = T + out.dT_dt * dt
tests/unit/test_physics_convection.py:253:    """SBM: convection should dry the lower levels (dq_v_dt < 0)."""
tests/unit/test_physics_convection.py:256:    min_dqv = float(jnp.min(out.dq_v_dt))
tests/unit/test_physics_convection.py:258:        f"SBM: no drying found, min dq_v_dt = {min_dqv:.2e}"
tests/unit/test_physics_convection.py:301:def test_stratosphere_mass_flux_gate_actually_closes():
tests/unit/test_physics_convection.py:312:    gate = stratosphere_mass_flux_gate(p_full)
tests/unit/test_physics_convection.py:351:    cannot dump convective momentum into the model top.
tests/unit/test_physics_convection.py:386:    gate = stratosphere_mass_flux_gate(p_full)
tests/unit/test_physics_convection.py:400:        "stratosphere_mass_flux_gate to M_u (and M_d when present)"
tests/unit/test_physics_convection.py:424:    assert jnp.all(jnp.isfinite(out.dT_dt)), f"{scheme}: dT_dt has NaN/Inf"
tests/unit/test_physics_convection.py:425:    assert jnp.all(jnp.isfinite(out.dq_v_dt)), f"{scheme}: dq_v_dt has NaN/Inf"
tests/unit/test_physics_convection.py:426:    assert jnp.all(jnp.isfinite(out.dq_c_conv_dt)), (
tests/unit/test_physics_convection.py:427:        f"{scheme}: dq_c_conv_dt has NaN/Inf"
tests/unit/test_physics_convection.py:475:        constants.c_pd * out.dT_dt
tests/unit/test_physics_convection.py:476:        + constants.L_v * (out.dq_v_dt + out.dq_c_conv_dt)
tests/unit/test_physics_convection.py:489:    diluted condensate as dq_c_conv_dt at detrainment.  Column-integrated
tests/unit/test_physics_convection.py:500:    mse = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
tests/unit/test_physics_convection.py:528:    T_after = T + out.dT_dt * 300.0
tests/unit/test_physics_convection.py:553:    max_dT = float(jnp.max(jnp.abs(out.dT_dt)))
tests/unit/test_physics_convection.py:554:    max_dq_c = float(jnp.max(out.dq_c_conv_dt))
tests/unit/test_physics_convection.py:557:    assert max_dT < 1e-2, f"{scheme}: dT_dt = {max_dT:.2e} in stable column"
tests/unit/test_physics_convection.py:559:        f"{scheme}: dq_c_conv_dt = {max_dq_c:.2e} kg/kg/s in stable column"
tests/unit/test_physics_convection.py:564:# 3e extended -- precipitation non-negative across multiple random profiles
tests/unit/test_physics_convection.py:568:def test_precipitation_non_negative_random_profiles(scheme):
tests/unit/test_physics_convection.py:569:    """dq_c_conv_dt >= 0 at every level for a spread of random profiles.
tests/unit/test_physics_convection.py:588:    min_dq_c = float(jnp.min(out.dq_c_conv_dt))
tests/unit/test_physics_convection.py:591:        f"{scheme}: random-profile min dq_c_conv_dt = {min_dq_c:.2e}"
tests/unit/test_physics_convection.py:610:    min_dqv = float(jnp.min(out.dq_v_dt))
tests/unit/test_physics_convection.py:612:        f"{scheme}: no drying detected, min dq_v_dt = {min_dqv:.2e}"
tests/unit/test_physics_convection.py:621:    should produce dT_dt > 0 in at least the upper-troposphere section.
tests/unit/test_physics_convection.py:626:    upper = out.dT_dt[:, : nlev // 2]   # top half (low pressure)
tests/unit/test_physics_convection.py:629:        f"{scheme}: no upper-trop warming, max dT_dt[upper] = {max_dT_upper:.2e}"
tests/unit/test_physics_convection.py:677:    The diagnosed equilibrium a_u_eq = convective_mask * a_u_init.  With
tests/unit/test_diff_sea_ice.py:32:        precip_total=0.0 * ones,
tests/unit/test_diff_sea_ice.py:33:        precip_snow=0.0 * ones,
tests/unit/test_diff_sea_ice.py:44:        has_precipitation=1.0 * ones,
tests/unit/test_physics_smoke.py:92:    check_shape(tend.dT_dt, (6, 8, 8, 10), f"radiation({scheme}).dT_dt")
tests/unit/test_physics_smoke.py:94:    assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0, f"{scheme}: zero heating"
tests/unit/test_physics_smoke.py:112:    check_shape(tend.dT_dt, (6, 8, 8, 10), f"convection({scheme}).dT_dt")
tests/unit/test_physics_smoke.py:141:    check_shape(tend.dT_dt, (6, 8, 8, 10), f"microphysics({scheme}).dT_dt")
tests/unit/test_physics_smoke.py:160:    check_shape(tend.dT_dt, (6, 8, 8, 10), f"turbulence({scheme}).dT_dt")
tests/unit/test_physics_smoke.py:181:    check_shape(tend.dT_dt, (6, 8, 8, 10), f"gwd({scheme}).dT_dt")
tests/unit/test_physics_smoke.py:221:    assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:342:            new_T = new_T + DT * conv_out.dT_dt.reshape(state.T.data.shape)
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py:343:            q_v = q_v + DT * conv_out.dq_v_dt.reshape(q_v.shape)
tests/unit/test_land_ice_lake.py:39:        cos_zenith=0.5, co2_ppmv=415.0, precip_total=0.0, precip_snow=0.0,
tests/unit/test_land_ice_lake.py:44:        precip_total=jnp.full(s, d["precip_total"], f),
tests/unit/test_land_ice_lake.py:45:        precip_snow=jnp.full(s, d["precip_snow"], f),
tests/unit/test_land_ice_lake.py:55:        has_radiation=jnp.ones(s, f), has_precipitation=jnp.ones(s, f),
tests/unit/test_land_params.py:346:            mean_precip=jnp.full(NCOL, 3e-5),
tests/unit/test_ml_physics_parameterization.py:127:        dq_v_dt_micro=jnp.ones((2, nlev)) * -1.0e-6,
tests/unit/test_ml_physics_parameterization.py:130:        precip_micro=jnp.ones((2,)) * 1.0e-4,
tests/unit/test_ml_physics_parameterization.py:139:    assert unpacked["dq_v_dt_micro"].shape == (2, nlev)
tests/unit/test_ml_physics_parameterization.py:142:    assert unpacked["precip_micro"].shape == (2,)
tests/unit/test_ml_physics_parameterization.py:294:        dq_v_dt_micro=jnp.ones((n_samples, nlev)) * -2.0e-7,
tests/unit/test_ml_physics_parameterization.py:297:        precip_micro=jnp.ones((n_samples,)) * 8.0e-5,
tests/unit/test_ml_physics_parameterization.py:316:    assert result.metrics.rmse_dq_v_dt_micro == 0.0
tests/unit/test_ml_physics_parameterization.py:319:    assert result.metrics.rmse_precip_micro == 0.0
tests/unit/test_ml_physics_parameterization.py:367:        rates.precipitation / generated_flux,
tests/unit/test_ml_physics_parameterization.py:383:        rebuilt.dT_dt,
tests/unit/test_ml_physics_parameterization.py:384:        physical.dT_dt,
tests/unit/test_ml_physics_parameterization.py:389:        rebuilt.dq_v_dt,
tests/unit/test_ml_physics_parameterization.py:390:        physical.dq_v_dt,
tests/unit/test_ml_physics_parameterization.py:397:        rebuilt.precipitation,
tests/unit/test_ml_physics_parameterization.py:398:        physical.precipitation,
tests/unit/test_convection_plume.py:7:* :func:`compute_cin`                     — convective inhibition [J/kg].
tests/unit/test_gradient_checkpointing.py:44:        dT_dt=jnp.zeros(T.shape),
tests/unit/test_gradient_checkpointing.py:45:        dq_v_dt=jnp.zeros(T.shape),
tests/unit/test_gradient_checkpointing.py:48:        precip=jnp.zeros(p_s.shape),
tests/unit/test_gradient_checkpointing.py:78:    kwargs["dT_dt"] = jnp.full(T.shape, 1e-5) * tau / 7.2
tests/unit/test_equation_fixes.py:194:        # Even though other processes may dominate, dT_dt should differ
tests/unit/test_equation_fixes.py:195:        diff = jnp.abs(out1.dT_dt - out2.dT_dt).sum()
tests/unit/test_equation_fixes.py:196:        assert diff > 0, "dT_dt should depend on dt (condensation is divided by dt)"
tests/unit/test_equation_fixes.py:225:        max_actual = float(jnp.max(jnp.abs(out.dT_dt)))
tests/unit/test_equation_fixes.py:249:        diff = jnp.abs(out1.dT_dt - out2.dT_dt).sum()
tests/unit/test_equation_fixes.py:250:        assert diff > 0, f"{scheme_name}: dT_dt should depend on dt"
tests/unit/test_equation_fixes.py:252:    def test_precipitation_units_kg_m2_s(self):
tests/unit/test_equation_fixes.py:258:        max_precip = jnp.max(out.precipitation)
tests/unit/test_equation_fixes.py:259:        # Reasonable precipitation is 0 to ~100 mm/hr = ~0.028 kg/m²/s
tests/unit/test_equation_fixes.py:260:        assert max_precip < 1.0, f"Precipitation {max_precip} kg/m²/s unreasonably large"
tests/unit/test_equation_fixes.py:264:# Issues 4-5: Mass-flux and EDMF precipitation dimensions
tests/unit/test_equation_fixes.py:272:        Post-Option-C: mass_flux now emits a 3D ``dq_c_conv_dt``
tests/unit/test_equation_fixes.py:273:        (kg/kg/s) instead of a scalar surface ``precipitation``
tests/unit/test_equation_fixes.py:288:        max_rate = jnp.max(out.dq_c_conv_dt)
tests/unit/test_equation_fixes.py:289:        assert max_rate >= 0, "dq_c_conv_dt must be non-negative"
tests/unit/test_equation_fixes.py:291:            f"Mass-flux dq_c_conv_dt {max_rate:.4e} kg/kg/s unreasonably "
tests/unit/test_equation_fixes.py:295:    def test_analytic_column_precipitation(self):
tests/unit/test_equation_fixes.py:319:        # The precipitation integral uses dz, not dp/g
tests/unit/test_equation_fixes.py:330:        Replaces the legacy ``out.precipitation`` (kg/m²/s) check after
tests/unit/test_equation_fixes.py:332:        ``dq_c_conv_dt`` field; surface precipitation is owned by
tests/unit/test_equation_fixes.py:347:        max_rate = jnp.max(out.dq_c_conv_dt)
tests/unit/test_equation_fixes.py:348:        assert max_rate >= 0, "dq_c_conv_dt must be non-negative"
tests/unit/test_equation_fixes.py:350:            f"EDMF dq_c_conv_dt {max_rate:.4e} kg/kg/s unreasonably large "
tests/unit/test_equation_fixes.py:393:        assert jnp.any(out.convective_mask > 0.1), "Kuo should trigger for moist columns"
tests/unit/test_equation_fixes.py:394:        assert jnp.any(out.dT_dt != 0), "Kuo should produce nonzero heating"
tests/unit/test_equation_fixes.py:439:        # Positive B_f = unstable (convective)
tests/unit/test_equation_fixes.py:445:        # dT_dt should be nonzero (non-local transport active in unstable BL)
tests/unit/test_equation_fixes.py:446:        assert jnp.any(jnp.abs(out.dT_dt) > 0), "KPP non-local should produce nonzero dT_dt"
tests/unit/test_distributed_checkpoint.py:239:            "precip_total": np.array(42.0),
tests/unit/test_distributed_checkpoint.py:253:        assert "precip_total" in loaded_diag
tests/unit/test_distributed_checkpoint.py:256:            loaded_diag["precip_total"], 42.0, atol=1e-12
tests/unit/test_scale_jit_health.py:193:            precip_accum=jnp.zeros(shape2d, dtype=jnp.float32),
tests/unit/test_physics_turbulence.py:81:    """When T_sfc > T_air (warm surface), dT_dt > 0 at lowest level (heating)."""
tests/unit/test_physics_turbulence.py:123:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt has NaN/Inf"
tests/unit/test_physics_turbulence.py:138:    max_dT = float(jnp.max(jnp.abs(tend.dT_dt.data)))
tests/unit/test_physics_turbulence.py:141:    # dT_dt < 1 K/s (reasonable for boundary layer)
tests/unit/test_physics_turbulence.py:142:    assert max_dT < 1.0, f"{scheme}: |dT_dt| = {max_dT:.2e} exceeds 1 K/s"
tests/unit/test_diff_atmosphere_physics.py:84:            return jnp.sum(tend.dT_dt.data ** 2)
tests/unit/test_diff_atmosphere_physics.py:87:        assert_gradient_ok(grad, "Held-Suarez dT_dt w.r.t. T")
tests/unit/test_diff_atmosphere_physics.py:156:            return jnp.sum(tend.dT_dt.data ** 2)
tests/unit/test_diff_atmosphere_physics.py:189:            return jnp.sum(tend.dT_dt.data ** 2)
tests/unit/test_diff_atmosphere_physics.py:223:            return jnp.sum(tend.dT_dt.data ** 2)
tests/unit/test_diff_atmosphere_physics.py:265:            return jnp.sum(tend.dT_dt.data ** 2)
tests/unit/test_training_modules.py:55:        dT_dt=jnp.zeros(T.shape),
tests/unit/test_training_modules.py:56:        dq_v_dt=jnp.zeros(T.shape),
tests/unit/test_training_modules.py:59:        precip=jnp.zeros(p_s.shape),
tests/unit/test_physics_gwd.py:103:    max_dT = float(jnp.max(jnp.abs(tend.dT_dt.data)))
tests/unit/test_physics_gwd.py:105:    assert max_dT < 1e-3, f"{scheme}: |dT_dt| = {max_dT:.2e} exceeds 1e-3 K/s"
tests/unit/test_physics_gwd.py:118:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt NaN/Inf"
tests/unit/test_physics_units.py:66:    """dT_dt [K/s] * dt [s] -> DeltaT [K]. |DeltaT| < 10 K for dt=300s."""
tests/unit/test_physics_units.py:69:    delta_T = tend.dT_dt.data * dt
tests/unit/test_physics_units.py:153:    mse_tend = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
tests/unit/test_tiedtke.py:77:    out, M_u_new = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0)
tests/unit/test_tiedtke.py:78:    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
tests/unit/test_tiedtke.py:79:                out.convective_mask, out.du_dt_conv, out.dv_dt_conv,
tests/unit/test_tiedtke.py:80:                M_u_new):
tests/unit/test_tiedtke.py:83:    assert M_u_new.shape == (ncol, nlev)
tests/unit/test_tiedtke.py:91:    """The diagnosed ``M_u_new`` carries non-trivial values on at
tests/unit/test_tiedtke.py:97:    _, M_u_new = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0)
tests/unit/test_tiedtke.py:100:    aloft_max = float(jnp.max(M_u_new[:, :-1]))
tests/unit/test_tiedtke.py:169:    diff = jnp.abs(out_on.dT_dt[:, -4:] - out_off.dT_dt[:, -4:])
tests/unit/test_tiedtke.py:185:        return jnp.sum(out.dT_dt)
tests/unit/test_tiedtke.py:199:        return jnp.sum(out.dT_dt)
tests/unit/test_tiedtke.py:246:    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
tests/unit/test_tiedtke.py:278:    H = float(jnp.sum(out.dT_dt * dp / constants.g, axis=1).mean()) * constants.c_pd
tests/unit/test_tiedtke.py:279:    Q = float(jnp.sum(out.dq_v_dt * dp / constants.g, axis=1).mean()) * constants.L_v
tests/unit/test_tiedtke.py:280:    C = float(jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1).mean()) * constants.L_v
tests/unit/test_mpas_atmosphere.py:156:        self.assertEqual(tend.dT_dt.data.shape,
tests/unit/test_mpas_atmosphere.py:169:        self.assertTrue(jnp.all(jnp.isfinite(tend.dT_dt.data)))
tests/unit/test_land_ice_integrated.py:40:        precip_total=0.0, precip_snow=0.0,
tests/unit/test_land_ice_integrated.py:45:        precip_total=jnp.full(s, d["precip_total"], f),
tests/unit/test_land_ice_integrated.py:46:        precip_snow=jnp.full(s, d["precip_snow"], f),
tests/unit/test_land_ice_integrated.py:56:        has_radiation=jnp.ones(s, f), has_precipitation=jnp.ones(s, f),
tests/unit/test_surface_exchange.py:48:    assert float(out.has_precipitation) == 0.0
tests/unit/test_surface_exchange.py:60:    precip_total = jnp.full(shape, 1e-5)
tests/unit/test_surface_exchange.py:64:        sw_down=sw_down, lw_down=lw_down, precip_total=precip_total,
tests/unit/test_surface_exchange.py:69:    assert jnp.allclose(out.precip_total, 1e-5)
tests/unit/test_surface_exchange.py:71:    assert float(out.has_precipitation) == 1.0
tests/unit/test_land_ice_snow.py:26:        precip_snow = jnp.array([1e-4])  # kg/m2/s
tests/unit/test_land_ice_snow.py:30:            snow, snow_age, T_surface, precip_snow, dt, Q_net=jnp.array([0.0]),
tests/unit/test_land_ice_snow.py:32:        expected = precip_snow * dt  # 0.36 kg/m2
tests/unit/test_land_ice_snow.py:42:        precip_snow = jnp.array([0.0])
tests/unit/test_land_ice_snow.py:46:            snow, snow_age, T_surface, precip_snow, dt, Q_net=Q_net,
tests/unit/test_land_ice_snow.py:59:        precip_snow = jnp.array([0.0])
tests/unit/test_land_ice_snow.py:63:            snow, snow_age, T_surface, precip_snow, dt, Q_net=Q_net,
tests/unit/test_land_ice_snow.py:74:        precip_snow = jnp.array([0.0])
tests/unit/test_land_ice_snow.py:78:            snow, snow_age, T_surface, precip_snow, dt, Q_net=Q_net,
tests/unit/test_land_ice_snow.py:89:        precip_snow = jnp.array([0.0])
tests/unit/test_land_ice_snow.py:92:            snow, snow_age, T_surface, precip_snow, dt, Q_net=jnp.array([0.0]),
tests/unit/test_land_ice_snow.py:100:        precip_snow = jnp.array([1e-4])
tests/unit/test_land_ice_snow.py:103:            snow, snow_age, T_surface, precip_snow, dt, Q_net=jnp.array([0.0]),
tests/unit/test_land_ice_snow.py:111:        precip_snow = jnp.array([0.0])
tests/unit/test_land_ice_snow.py:114:            snow, snow_age, T_surface, precip_snow, dt, Q_net=jnp.array([0.0]),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:411:        # Bound: convective tendency over 300s with q ~ 1.5e-2 should
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:130:        assert out.dT_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:149:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:213:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:270:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:326:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:378:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:434:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:490:        assert tendencies.dT_dt.data.shape == (6, n, n, nlev)
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py:608:        assert jnp.allclose(tendencies.dT_dt.data, 0.0)
tests/unit/test_land_ice_sea_ice_thermo.py:40:        precip_total=0.0, precip_snow=0.0,
tests/unit/test_land_ice_sea_ice_thermo.py:46:        precip_total=jnp.full(s, d["precip_total"], f),
tests/unit/test_land_ice_sea_ice_thermo.py:47:        precip_snow=jnp.full(s, d["precip_snow"], f),
tests/unit/test_land_ice_sea_ice_thermo.py:57:        has_radiation=jnp.ones(s, f), has_precipitation=jnp.ones(s, f),
tests/atmosphere/hydrostatic/unit/test_amip_config.py:122:            "precip_total": np.ones((6, 4, 4)) * 3.14,
tests/atmosphere/hydrostatic/unit/test_amip_config.py:129:        assert "precip_total" in diag_acc
tests/atmosphere/hydrostatic/unit/test_amip_config.py:131:        np.testing.assert_allclose(diag_acc["precip_total"], 3.14, atol=1e-12)
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:402:        assert jnp.allclose(tend.dT_dt.data, 0.0, atol=1e-6), \
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:403:            f"dT_dt max: {float(jnp.max(jnp.abs(tend.dT_dt.data)))}"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:417:        assert jnp.all(jnp.isfinite(tend.dT_dt.data)), "dT_dt not finite"
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:429:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py:441:        assert tend.dT_dt.data.shape == state.T.data.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:154:        assert out.dT_dt.shape == (4, 10)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:155:        assert out.precipitation.shape == (4,)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:176:        assert out.dT_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:177:        assert out.precipitation.shape == (T.shape[0],)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:179:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:182:        assert jnp.all(out.precipitation >= 0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:187:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:215:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:224:        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
tests/atmosphere/hydrostatic/unit/test_microphysics.py:237:        assert out.dT_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:238:        assert out.precipitation.shape == (T.shape[0],)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:240:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:243:        assert jnp.all(out.precipitation >= 0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:248:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:279:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:294:        assert out.dT_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:298:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:301:        assert jnp.all(out.precipitation >= 0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:306:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:319:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:328:        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
tests/atmosphere/hydrostatic/unit/test_microphysics.py:341:        assert out.dT_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:345:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:348:        assert jnp.all(out.precipitation >= 0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:353:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:380:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:389:        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
tests/atmosphere/hydrostatic/unit/test_microphysics.py:402:        assert out.dT_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:405:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:408:        assert jnp.all(out.precipitation >= 0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:413:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:442:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:451:        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
tests/atmosphere/hydrostatic/unit/test_microphysics.py:476:        assert out.dT_dt.shape == T.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:477:        assert out.precipitation.shape == (T.shape[0],)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:479:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_microphysics.py:485:        assert jnp.all(out.precipitation >= 0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:494:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:513:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:541:        assert tend.dT_dt.data.shape == state.T.data.shape
tests/atmosphere/hydrostatic/unit/test_microphysics.py:548:        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) == 0.0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:558:            return jnp.sum(tend.dT_dt.data ** 2)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:812:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0
tests/atmosphere/hydrostatic/unit/test_microphysics.py:826:            q_v = jnp.maximum(q_v + dt * out.dq_v_dt, 0.0)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:829:            T = T + dt * out.dT_dt
tests/atmosphere/hydrostatic/unit/test_monthly_means.py:146:        accum.add_scalar(15.0, 0, {'T_atm': 260.0, 'precip': 3.0})
tests/atmosphere/hydrostatic/unit/test_monthly_means.py:150:        assert 'scalar_precip' in result
tests/atmosphere/hydrostatic/unit/test_monthly_means.py:190:        accum.add_scalar(15.0, 0, {'precip': 2.5})
tests/atmosphere/hydrostatic/unit/test_monthly_means.py:203:            assert 'scalar_precip' in data
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:77:    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{label}: dT_dt has NaN/Inf"
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:82:    max_dT_actual = float(jnp.max(jnp.abs(tend.dT_dt.data)))
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:109:        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0, "gray: zero heating"
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:116:        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0, "rrtmgp: zero heating"
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:323:        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py:421:        # change relative to a reasonable convective drying rate
tests/atmosphere/hydrostatic/unit/test_radiation.py:339:        assert tendencies.dT_dt.data.shape == (6, n, n, nlev)
tests/atmosphere/hydrostatic/unit/test_radiation.py:358:        max_hr = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
tests/atmosphere/hydrostatic/unit/test_radiation.py:392:        gray_tend = gray_fn(state, grid, sigma).dT_dt.data
tests/atmosphere/hydrostatic/unit/test_radiation.py:393:        rrtmgp_tend = rrtmgp_fn(state, grid, sigma).dT_dt.data
tests/atmosphere/hydrostatic/unit/test_radiation.py:493:            return jnp.sum(tendencies.dT_dt.data ** 2)
tests/atmosphere/hydrostatic/unit/test_radiation.py:664:        assert tendencies.dT_dt.data.shape == (6, 8, 8, 10)
tests/atmosphere/hydrostatic/unit/test_radiation.py:665:        max_hr = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
tests/atmosphere/hydrostatic/unit/test_radiation.py:667:        assert jnp.all(jnp.isfinite(tendencies.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_radiation.py:706:        assert jnp.allclose(tend1.dT_dt.data, tend2.dT_dt.data)
tests/atmosphere/hydrostatic/unit/test_radiation.py:901:        assert tendencies.dT_dt.data.shape == (6, 4, 4, 8)
tests/atmosphere/hydrostatic/unit/test_radiation.py:902:        assert jnp.all(jnp.isfinite(tendencies.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_radiation.py:903:        max_hr = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
tests/atmosphere/hydrostatic/unit/test_radiation.py:1100:        assert tendencies.dT_dt.data.shape == (6, 4, 4, 8)
tests/atmosphere/hydrostatic/unit/test_radiation.py:1101:        assert jnp.all(jnp.isfinite(tendencies.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_radiation.py:1123:        assert jnp.allclose(tend_clear.dT_dt.data, tend_none.dT_dt.data)
tests/atmosphere/hydrostatic/unit/test_convection.py:5:- SBM: shapes, enthalpy conservation, precipitation, trigger, differentiability
tests/atmosphere/hydrostatic/unit/test_convection.py:212:        assert out.dT_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:213:        assert out.dq_v_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:214:        assert out.dq_c_conv_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:216:        assert out.convective_mask.shape == (ncol,)
tests/atmosphere/hydrostatic/unit/test_convection.py:222:        sum(c_pd * dT_dt + L_v * dq_v_dt) * dp / g should be small
tests/atmosphere/hydrostatic/unit/test_convection.py:232:            (constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt) * dp / constants.g,
tests/atmosphere/hydrostatic/unit/test_convection.py:237:            constants.c_pd * jnp.abs(out.dT_dt) * dp / constants.g,
tests/atmosphere/hydrostatic/unit/test_convection.py:244:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:249:        assert jnp.all(out.dq_c_conv_dt >= 0)
tests/atmosphere/hydrostatic/unit/test_convection.py:261:        max_stable = float(jnp.max(jnp.abs(out_stable.dT_dt)))
tests/atmosphere/hydrostatic/unit/test_convection.py:262:        max_unstable = float(jnp.max(jnp.abs(out_unstable.dT_dt)))
tests/atmosphere/hydrostatic/unit/test_convection.py:270:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-6
tests/atmosphere/hydrostatic/unit/test_convection.py:280:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_convection.py:298:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_convection.py:299:        assert jnp.all(jnp.isfinite(out.dq_v_dt))
tests/atmosphere/hydrostatic/unit/test_convection.py:300:        assert out.dT_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:301:        assert float(jnp.max(jnp.abs(out.dT_dt[0]))) >= float(jnp.max(jnp.abs(out.dT_dt[-1])))
tests/atmosphere/hydrostatic/unit/test_convection.py:318:        assert out.dT_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:319:        assert out.dq_v_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:320:        assert out.dq_c_conv_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:322:        assert out.convective_mask.shape == (ncol,)
tests/atmosphere/hydrostatic/unit/test_convection.py:333:        max_stable = float(jnp.max(jnp.abs(out_stable.dT_dt)))
tests/atmosphere/hydrostatic/unit/test_convection.py:334:        max_unstable = float(jnp.max(jnp.abs(out_unstable.dT_dt)))
tests/atmosphere/hydrostatic/unit/test_convection.py:345:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-6
tests/atmosphere/hydrostatic/unit/test_convection.py:364:        assert float(jnp.abs(out.dT_dt[0, 0])) > 1e-4
tests/atmosphere/hydrostatic/unit/test_convection.py:374:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_convection.py:404:        assert tendencies.dT_dt.data.shape == (6, n, n, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:422:        max_hr = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
tests/atmosphere/hydrostatic/unit/test_convection.py:539:            return jnp.sum(tendencies.dT_dt.data ** 2)
tests/atmosphere/hydrostatic/unit/test_convection.py:567:        assert not jnp.allclose(tend_sbm.dT_dt.data, tend_dca.dT_dt.data, atol=1e-10)
tests/atmosphere/hydrostatic/unit/test_convection.py:583:        assert jnp.allclose(tendencies.dT_dt.data, 0.0)
tests/atmosphere/hydrostatic/unit/test_convection.py:604:        assert not jnp.allclose(tend_kuo.dT_dt.data, tend_sbm.dT_dt.data, atol=1e-10)
tests/atmosphere/hydrostatic/unit/test_convection.py:620:        max_val = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
tests/atmosphere/hydrostatic/unit/test_convection.py:637:        max_val = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
tests/atmosphere/hydrostatic/unit/test_convection.py:656:            return jnp.sum(tendencies.dT_dt.data ** 2)
tests/atmosphere/hydrostatic/unit/test_convection.py:677:            return jnp.sum(tendencies.dT_dt.data ** 2)
tests/atmosphere/hydrostatic/unit/test_convection.py:697:        assert out.dT_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:698:        assert out.dq_v_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:699:        assert out.dq_c_conv_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:701:        assert out.convective_mask.shape == (ncol,)
tests/atmosphere/hydrostatic/unit/test_convection.py:703:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:708:        assert jnp.all(out.dq_c_conv_dt >= 0)
tests/atmosphere/hydrostatic/unit/test_convection.py:728:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-6
tests/atmosphere/hydrostatic/unit/test_convection.py:738:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_convection.py:762:        assert out.dT_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:763:        assert out.dq_v_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:764:        assert out.dq_c_conv_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:766:        assert out.convective_mask.shape == (ncol,)
tests/atmosphere/hydrostatic/unit/test_convection.py:805:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10
tests/atmosphere/hydrostatic/unit/test_convection.py:807:    def test_precipitation_is_nonzero_for_moist_unstable_columns(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:808:        """Moist unstable columns should produce some convective condensate."""
tests/atmosphere/hydrostatic/unit/test_convection.py:817:        assert float(jnp.max(out.dq_c_conv_dt)) > 0.0
tests/atmosphere/hydrostatic/unit/test_convection.py:836:        assert jnp.allclose(out_split.dT_dt, out_full.dT_dt)
tests/atmosphere/hydrostatic/unit/test_convection.py:837:        assert jnp.allclose(out_split.dq_v_dt, out_full.dq_v_dt)
tests/atmosphere/hydrostatic/unit/test_convection.py:838:        assert jnp.allclose(out_split.dq_c_conv_dt, out_full.dq_c_conv_dt)
tests/atmosphere/hydrostatic/unit/test_convection.py:851:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_convection.py:875:        assert out.dT_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:876:        assert out.dq_v_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:877:        assert out.dq_c_conv_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection.py:879:        assert out.convective_mask.shape == (ncol,)
tests/atmosphere/hydrostatic/unit/test_convection.py:921:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10
tests/atmosphere/hydrostatic/unit/test_convection.py:934:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_convection.py:964:        dT_mid = out.dT_dt[:, mid]
tests/atmosphere/hydrostatic/unit/test_convection.py:985:        assert jnp.any(out.dT_dt > 0), "Detrainment should produce some warming"
tests/atmosphere/hydrostatic/unit/test_convection.py:1003:        mean_dq = jnp.mean(out.dq_v_dt[:, mid])
tests/atmosphere/hydrostatic/unit/test_convection.py:1008:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:1017:        assert jnp.all(out.dq_c_conv_dt >= 0)
tests/atmosphere/hydrostatic/unit/test_convection.py:1036:        mean_warming = jnp.mean(out.dT_dt[:, mid])
tests/atmosphere/hydrostatic/unit/test_convection.py:1053:        assert jnp.any(out.dT_dt > 0), "Detrainment should produce some warming"
tests/atmosphere/hydrostatic/unit/test_convection.py:1075:        rms_sub = jnp.sqrt(jnp.mean(out_sub.dT_dt ** 2))
tests/atmosphere/hydrostatic/unit/test_convection.py:1076:        rms_det = jnp.sqrt(jnp.mean(out_det.dT_dt ** 2))
tests/atmosphere/hydrostatic/unit/test_convection.py:1077:        rms_both = jnp.sqrt(jnp.mean(out_both.dT_dt ** 2))
tests/atmosphere/hydrostatic/unit/test_convection.py:1095:        mean_dq = jnp.mean(out.dq_v_dt[:, mid])
tests/atmosphere/hydrostatic/unit/test_convection.py:1100:    def test_precipitation_non_negative(self):
tests/atmosphere/hydrostatic/unit/test_convection.py:1109:        assert jnp.all(out.dq_c_conv_dt >= 0)
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py:91:        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) < 1e-7
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:93:        assert jnp.allclose(tend.dT_dt.data, 0.0)
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:106:        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:122:        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:152:        # Combined dT_dt should equal sum of individuals
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:153:        expected_dT = tend_rad.dT_dt.data + tend_turb.dT_dt.data
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:154:        assert jnp.allclose(tend_both.dT_dt.data, expected_dT, atol=1e-6)
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:171:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:189:        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:193:        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:213:        assert jnp.all(jnp.isfinite(tend_sbm.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:214:        assert jnp.all(jnp.isfinite(tend_dca.dT_dt.data))
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:216:        assert not jnp.allclose(tend_sbm.dT_dt.data, tend_dca.dT_dt.data, atol=1e-10)
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:239:        assert not jnp.allclose(tend_gray.dT_dt.data, tend_none.dT_dt.data)
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:274:        assert tend.dT_dt.data.shape == (6, n, n, nlev)
tests/atmosphere/hydrostatic/unit/test_combined_physics.py:290:            return jnp.sum(tend.dT_dt.data ** 2)
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:26:- ``dT_dt`` has the same shape as the input ``T``.
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:138:        assert tend.dT_dt.data.shape == latlon_state.T.data.shape, (
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:139:            f"{scheme_name}: dT_dt shape mismatch"
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:158:        assert tend.dT_dt.dims == latlon_state.T.dims, (
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:159:            f"{scheme_name}: dT_dt dims {tend.dT_dt.dims} should match "
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:174:        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data))), (
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:175:            f"{scheme_name}: dT_dt has NaN/Inf"
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:210:        assert tend.dT_dt.data.shape == mpas_state.T.data.shape, (
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:211:            f"{scheme_name}: dT_dt shape mismatch"
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:232:        assert tend.dT_dt.dims == mpas_state.T.dims, (
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:233:            f"{scheme_name}: dT_dt dims {tend.dT_dt.dims} should match "
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:248:        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data))), (
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:249:            f"{scheme_name}: dT_dt has NaN/Inf"
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:330:        proxy_dT = out_proxy.dT_dt.reshape(ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:331:        zeros_dT = out_zeros.dT_dt.reshape(ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:332:        bridge_dT = tend_bridge.dT_dt.data.reshape(ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:367:        """Documented PR6 limitation: convective momentum transport
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:427:        assert tend.dT_dt.data.shape == mpas_state.T.data.shape
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:433:        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data)))
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:448:        assert tend.dT_dt.dims == mpas_state.T.dims
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:464:        assert tend.dT_dt.data.shape == latlon_state.T.data.shape
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:466:        assert tend.dT_dt.dims == latlon_state.T.dims
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:467:        assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data)))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:286:        assert out.dT_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:287:        assert out.dq_v_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:308:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10
tests/atmosphere/hydrostatic/unit/test_turbulence.py:323:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:398:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:491:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:520:        assert tendencies.dT_dt.data.shape == (6, n, n, nlev)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:566:        max_dT = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:626:            return jnp.sum(tendencies.dT_dt.data ** 2)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:690:        assert jnp.allclose(tendencies.dT_dt.data, 0.0)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:716:        assert out.dT_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:717:        assert out.dq_v_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:738:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10
tests/atmosphere/hydrostatic/unit/test_turbulence.py:753:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:773:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:799:        assert not jnp.allclose(out_cg.dT_dt, out_no_cg.dT_dt, atol=1e-10)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:824:        assert out.dT_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:825:        assert out.dq_v_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:844:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10
tests/atmosphere/hydrostatic/unit/test_turbulence.py:859:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:879:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:886:        T_sfc = T[:, -1] + 15.0  # very warm surface for strong convective BL
tests/atmosphere/hydrostatic/unit/test_turbulence.py:904:        assert not jnp.allclose(out_ent.dT_dt, out_no_ent.dT_dt, atol=1e-12)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:930:        assert out.dT_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:931:        assert out.dq_v_dt.shape == (ncol, nlev)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:951:        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10
tests/atmosphere/hydrostatic/unit/test_turbulence.py:967:            return jnp.sum(out.dT_dt ** 2)
tests/atmosphere/hydrostatic/unit/test_turbulence.py:988:        assert jnp.all(jnp.isfinite(out.dT_dt))
tests/atmosphere/hydrostatic/unit/test_turbulence.py:1033:        assert not jnp.allclose(out_mf.dT_dt, out_no_mf.dT_dt, atol=1e-10)
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:379:                dT_dt=Field(data=zeros_3d, name="dT_dt", dims=dims_3d, units="K/s"),
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:485:                dT_dt=Field(data=zeros_3d, name="dT_dt", dims=dims_3d, units="K/s"),
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:823:                dT_dt=Field(data=zeros_3d, name="dT_dt", dims=dims_3d, units="K/s"),
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py:829:                                 name="dq_v_dt", dims=dims_3d, units="1/s"),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:9:  dict carries grid-space ``dq_v_dt`` / ``dq_c_dt`` / ``dq_r_dt`` /
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:158:            "Saturated column should produce non-zero dq_v_dt"
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:184:            f"microphysics backend; got identical dq_v_dt for wet vs "
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py:491:            f"Bridge may have wrong sign of dq_v_dt."
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:569:        # convective drying over 1 step at dt=300s with q_sat ~ 1.4e-2:

exec
/bin/zsh -lc 'rg -n "def compute_cape|def compute_moist_adiabat|def moist_adiabat_lapse_rate|saturation_mixing_ratio|def saturation" src/legoesm -S' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/thermo.py:3:This module provides `saturation_mixing_ratio` and
src/legoesm/thermo.py:4:`saturation_mixing_ratio_ice` with *no* dependency on
src/legoesm/thermo.py:20:def saturation_vapor_pressure(T: jax.Array) -> jax.Array:
src/legoesm/thermo.py:39:def saturation_mixing_ratio(
src/legoesm/thermo.py:72:def saturation_mixing_ratio_ice(
src/legoesm/thermo.py:101:def saturation_specific_humidity(
src/legoesm/thermo.py:125:    w_sat = saturation_mixing_ratio(T, p)
src/legoesm/driver/model_driver.py:19:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/driver/model_driver.py:414:            q_sat_init = saturation_mixing_ratio(self.state.T.data, p_full_init)
src/legoesm/driver/model_driver.py:2030:            q_sat = saturation_mixing_ratio(new_T, p_full)
src/legoesm/driver/model_driver.py:2133:                q_sat = saturation_mixing_ratio(new_T, p_full)
src/legoesm/coupler/coupler.py:18:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/coupler/coupler.py:183:    q_sfc = saturation_mixing_ratio(ocean_sst, forcing.p_surface)
src/legoesm/coupler/lake/two_layer_lake.py:19:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/coupler/lake/two_layer_lake.py:44:    q_sfc = saturation_mixing_ratio(T_epi, forcing.p_surface)
src/legoesm/coupler/lake/two_layer_lake.py:126:    q_sfc_new = saturation_mixing_ratio(T_epi_new, forcing.p_surface)
src/legoesm/driver/compiled_segments.py:45:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/driver/compiled_segments.py:623:                q_sat = saturation_mixing_ratio(T_upd, p_full)
src/legoesm/ice/sea_ice.py:28:from legoesm.thermo import saturation_mixing_ratio_ice
src/legoesm/ice/sea_ice.py:131:    q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)
src/legoesm/ice/sea_ice.py:191:    q_sfc_new = saturation_mixing_ratio_ice(T_ice_new, forcing.p_surface)
src/legoesm/ice/sea_ice.py:394:        q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)
src/legoesm/ice/sea_ice.py:479:    q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)
src/legoesm/atmosphere/physics/microphysics/sundqvist.py:23:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/microphysics/sundqvist.py:57:    q_sat = saturation_mixing_ratio(T, p_full)
src/legoesm/atmosphere/dynamics/spectral_nh.py:1217:    from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/dynamics/spectral_nh.py:1258:    q_v_profile = RH_profile * saturation_mixing_ratio(T_sounding, p_sounding)
src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py:8:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py:19:    return saturation_mixing_ratio(T_K, jnp.full_like(T_K, _P_ATM))
src/legoesm/atmosphere/physics/microphysics/thompson.py:22:from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
src/legoesm/atmosphere/physics/microphysics/thompson.py:116:    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
src/legoesm/atmosphere/physics/turbulence/smagorinsky.py:17:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/microphysics/kessler.py:22:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/microphysics/kessler.py:74:    q_sat = saturation_mixing_ratio(T, p_full)
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:13:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:16:def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0):
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:39:    q_sat = saturation_mixing_ratio(T, p_full)
src/legoesm/atmosphere/physics/turbulence/clubb_lite.py:49:from legoesm.thermo import saturation_mixing_ratio as _q_sat
src/legoesm/atmosphere/physics/turbulence/integration.py:48:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/turbulence/integration.py:213:        q_sfc = saturation_mixing_ratio(T_sfc, p_full_col[:, -1])
src/legoesm/atmosphere/physics/turbulence/integration.py:366:        q_sfc = saturation_mixing_ratio(T_sfc, p_full_col[:, -1])
src/legoesm/atmosphere/physics/turbulence/integration.py:496:        q_sfc = saturation_mixing_ratio(T_sfc, p_full_col[:, -1])
src/legoesm/atmosphere/physics/microphysics/morrison.py:22:from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
src/legoesm/atmosphere/physics/microphysics/morrison.py:95:    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
src/legoesm/atmosphere/physics/thermodynamics.py:36:from legoesm.thermo import saturation_mixing_ratio as saturation_mixing_ratio  # noqa: F401
src/legoesm/atmosphere/physics/thermodynamics.py:37:from legoesm.thermo import saturation_mixing_ratio_ice as saturation_mixing_ratio_ice  # noqa: F401
src/legoesm/atmosphere/physics/thermodynamics.py:160:def moist_adiabat_lapse_rate(
src/legoesm/atmosphere/physics/thermodynamics.py:187:    q_sat = saturation_mixing_ratio(T, p)
src/legoesm/atmosphere/physics/thermodynamics.py:195:def compute_moist_adiabat(
src/legoesm/atmosphere/physics/thermodynamics.py:268:def compute_cape(
src/legoesm/atmosphere/physics/convection/tiedtke.py:37:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/convection/tiedtke.py:176:    q_sat_env = saturation_mixing_ratio(T, p_full)
src/legoesm/atmosphere/physics/convection/sbm.py:34:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/convection/sbm.py:101:        q_trial = RH_ref[:, None] * saturation_mixing_ratio(T_trial, p_full)
src/legoesm/atmosphere/physics/convection/sbm.py:107:        q_sat_trial = saturation_mixing_ratio(T_trial, p_full)
src/legoesm/atmosphere/physics/convection/sbm.py:121:    q_ref = RH_ref[:, None] * saturation_mixing_ratio(T_ref, p_full)
src/legoesm/atmosphere/physics/convection/bechtold.py:39:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/convection/bechtold.py:279:        q_sat_env = saturation_mixing_ratio(T, p_full)
src/legoesm/atmosphere/physics/convection/kuo.py:49:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/convection/kuo.py:92:    q_sat = saturation_mixing_ratio(T, p_full)  # (ncol, nlev)
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:45:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/convection/emanuel.py:40:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/convection/_plume.py:43:  physical constants; ``legoesm.thermo.saturation_mixing_ratio`` and
src/legoesm/atmosphere/physics/convection/_plume.py:73:    saturation_mixing_ratio,
src/legoesm/atmosphere/physics/convection/_plume.py:168:    q_sat = saturation_mixing_ratio(T_parcel, p_parcel)
src/legoesm/atmosphere/physics/convection/_plume.py:431:    # Pin everything to the input precision so saturation_mixing_ratio
src/legoesm/atmosphere/physics/convection/_plume.py:524:        q_sat_new = saturation_mixing_ratio(T_u, p_e).astype(_dtype)
src/legoesm/atmosphere/physics/convection/mass_flux.py:52:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/convection/mass_flux.py:328:    q_sat_base = saturation_mixing_ratio(T[:, -1:], p_full[:, -1:])
src/legoesm/atmosphere/physics/convection/mass_flux.py:329:    q_sat_moist = saturation_mixing_ratio(T_moist, p_full)
src/legoesm/atmosphere/physics/convection/mass_flux.py:444:    q_sat_base = saturation_mixing_ratio(T[:, -1:], p_full[:, -1:])
src/legoesm/atmosphere/physics/convection/mass_flux.py:445:    q_sat_moist = saturation_mixing_ratio(T_moist, p_full)
src/legoesm/atmosphere/physics/convection/dca.py:23:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/convection/dca.py:138:        q_sat_upper = saturation_mixing_ratio(T_adj_upper, p_upper)
src/legoesm/atmosphere/physics/convection/dca.py:139:        q_sat_below = saturation_mixing_ratio(T_adj_below, p_below)
src/legoesm/ocean/simple_ocean_mpas.py:17:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/ocean/simple_ocean_mpas.py:77:    q_sfc = saturation_mixing_ratio(T_sfc, forcing.p_surface)
src/legoesm/ocean/simple_ocean_mpas.py:117:    q_sfc = saturation_mixing_ratio(T_sfc, forcing.p_surface)
src/legoesm/atmosphere/physics/clouds/cloud_fraction.py:32:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/atmosphere/physics/clouds/cloud_fraction.py:159:    q_sat = saturation_mixing_ratio(T, p_full)
src/legoesm/ocean/simple_ocean.py:20:from legoesm.thermo import saturation_mixing_ratio
src/legoesm/ocean/simple_ocean.py:107:    q_sfc = saturation_mixing_ratio(T_sfc, forcing.p_surface)
src/legoesm/ocean/simple_ocean.py:156:    q_sfc = saturation_mixing_ratio(T_sfc, forcing.p_surface)
src/legoesm/land/slab_land.py:23:from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice
src/legoesm/land/slab_land.py:118:    q_sat_liq = saturation_mixing_ratio(T_soil, forcing.p_surface)
src/legoesm/land/slab_land.py:119:    q_sat_ice = saturation_mixing_ratio_ice(T_soil, forcing.p_surface)
src/legoesm/land/slab_land.py:252:    q_sat_liq_new = saturation_mixing_ratio(T_soil_new, forcing.p_surface)
src/legoesm/land/slab_land.py:253:    q_sat_ice_new = saturation_mixing_ratio_ice(T_soil_new, forcing.p_surface)
src/legoesm/land/multilayer_land.py:35:from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice
src/legoesm/land/multilayer_land.py:170:    q_sat_liq = saturation_mixing_ratio(T_surface, forcing.p_surface)
src/legoesm/land/multilayer_land.py:171:    q_sat_ice = saturation_mixing_ratio_ice(T_surface, forcing.p_surface)
src/legoesm/land/multilayer_land.py:369:    q_sat_liq_new = saturation_mixing_ratio(T_surface_new, forcing.p_surface)
src/legoesm/land/multilayer_land.py:370:    q_sat_ice_new = saturation_mixing_ratio_ice(T_surface_new, forcing.p_surface)

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/thermodynamics.py | sed -n '130,340p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba tests/atmosphere/hydrostatic/unit/test_convection.py | sed -n '930,1120p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/thermo.py | sed -n '1,160p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   130	        Full-level density [kg/m^3], shape (..., nlev).
   131	    z_half : jax.Array
   132	        Interface height [m], shape (..., nlev+1), top-to-bottom ordering.
   133	
   134	    Returns
   135	    -------
   136	    jax.Array
   137	        Reconstructed half-level pressure [Pa], shape (..., nlev+1).
   138	    """
   139	    # Layer thicknesses are positive with top-to-bottom level indexing.
   140	    dz = jnp.abs(z_half[..., :-1] - z_half[..., 1:])
   141	    rho_pos = jnp.clip(rho_full, 1e-9, None)
   142	
   143	    # Hydrostatic increment across each full layer.
   144	    dp = constants.g * rho_pos * dz
   145	
   146	    # Top interface: centered estimate from top full level.
   147	    p_top = p_full[..., 0] - 0.5 * dp[..., 0]
   148	    p_top = jnp.clip(p_top, 1.0, None)
   149	
   150	    # Downward integration to all interfaces.
   151	    p_interfaces_inner = p_top[..., None] + jnp.cumsum(dp, axis=-1)
   152	    p_half = jnp.concatenate([p_top[..., None], p_interfaces_inner], axis=-1)
   153	    return jnp.clip(p_half, 1.0, None)
   154	
   155	
   156	# ==============================================================================
   157	# Moist thermodynamic functions (for convection)
   158	# ==============================================================================
   159	
   160	def moist_adiabat_lapse_rate(
   161	    T: jax.Array,
   162	    p: jax.Array,
   163	) -> jax.Array:
   164	    """Compute the moist adiabatic lapse rate dT/dp.
   165	
   166	    Gamma_m = (R_d * T / (c_pd * p)) *
   167	              (1 + L_v * q_sat / (R_d * T)) /
   168	              (1 + L_v^2 * q_sat / (c_pd * R_v * T^2))
   169	
   170	    Parameters
   171	    ----------
   172	    T : jax.Array
   173	        Temperature [K].
   174	    p : jax.Array
   175	        Pressure [Pa].
   176	
   177	    Returns
   178	    -------
   179	    jax.Array
   180	        Moist adiabatic lapse rate dT/dp [K/Pa].
   181	    """
   182	    R_d = constants.R_d
   183	    c_pd = constants.c_pd
   184	    L_v = constants.L_v
   185	    R_v = constants.R_v
   186	
   187	    q_sat = saturation_mixing_ratio(T, p)
   188	
   189	    numerator = 1.0 + L_v * q_sat / (R_d * T)
   190	    denominator = 1.0 + L_v ** 2 * q_sat / (c_pd * R_v * T ** 2)
   191	
   192	    return (R_d * T / (c_pd * p)) * numerator / denominator
   193	
   194	
   195	def compute_moist_adiabat(
   196	    T_base: jax.Array,
   197	    p_levels: jax.Array,
   198	) -> jax.Array:
   199	    """Compute moist adiabatic temperature profile from surface upward.
   200	
   201	    Integrates dT/dp = Gamma_m(T, p) upward from the lowest pressure
   202	    level using trapezoidal predictor-corrector via jax.lax.scan.
   203	
   204	    Parameters
   205	    ----------
   206	    T_base : jax.Array
   207	        Temperature at the lowest level (surface) [K], shape (ncol,).
   208	    p_levels : jax.Array
   209	        Pressure at full levels [Pa], shape (ncol, nlev).
   210	        Ordered top-to-bottom (p increasing with index).
   211	
   212	    Returns
   213	    -------
   214	    jax.Array
   215	        Moist adiabatic temperature profile [K], shape (ncol, nlev).
   216	    """
   217	    ncol, nlev = p_levels.shape
   218	
   219	    # Reverse to scan from surface (bottom) upward (top)
   220	    p_rev = p_levels[:, ::-1]  # (ncol, nlev), surface first
   221	
   222	    # Promote to common dtype so scan carry types are consistent.
   223	    # Physical constants in moist_adiabat_lapse_rate are Python float64;
   224	    # computation internally uses the promoted dtype, but the carry must
   225	    # have matching input/output dtypes for jax.lax.scan.
   226	    _dtype = jnp.result_type(T_base, p_rev)
   227	    T_base = T_base.astype(_dtype)
   228	    p_rev = p_rev.astype(_dtype)
   229	
   230	    def scan_step(T_prev, p_k):
   231	        """Trapezoidal predictor-corrector step."""
   232	        T_prev_val, p_prev_val = T_prev
   233	
   234	        dp = p_k - p_prev_val  # negative (going upward)
   235	
   236	        # Predictor: Euler step
   237	        gamma_1 = moist_adiabat_lapse_rate(T_prev_val, p_prev_val)
   238	        T_pred = T_prev_val + gamma_1 * dp
   239	
   240	        # Corrector: trapezoidal
   241	        gamma_2 = moist_adiabat_lapse_rate(T_pred, p_k)
   242	        T_new = T_prev_val + 0.5 * (gamma_1 + gamma_2) * dp
   243	
   244	        # Ensure temperature stays physical
   245	        T_new = jnp.clip(T_new, 100.0, 350.0).astype(_dtype)
   246	
   247	        return (T_new, p_k.astype(_dtype)), T_new
   248	
   249	    # Initial state: temperature at surface level
   250	    init = (T_base, p_rev[:, 0])
   251	
   252	    # Scan over levels 1..nlev-1 (moving upward from surface)
   253	    # Transpose to (nlev-1, ncol) for scan
   254	    p_scan = jnp.moveaxis(p_rev[:, 1:], 1, 0)  # (nlev-1, ncol)
   255	
   256	    _, T_scan = jax.lax.scan(scan_step, init, p_scan)
   257	    # T_scan: (nlev-1, ncol) — levels from surface+1 to top
   258	
   259	    T_scan = jnp.moveaxis(T_scan, 0, 1)  # (ncol, nlev-1)
   260	
   261	    # Prepend surface temperature
   262	    T_moist_rev = jnp.concatenate([T_base[:, None], T_scan], axis=1)  # (ncol, nlev)
   263	
   264	    # Reverse back to top-to-bottom ordering
   265	    return T_moist_rev[:, ::-1]
   266	
   267	
   268	def compute_cape(
   269	    T_env: jax.Array,
   270	    T_parcel: jax.Array,
   271	    p_full: jax.Array,
   272	    p_half: jax.Array,
   273	) -> jax.Array:
   274	    """Compute Convective Available Potential Energy (CAPE).
   275	
   276	    CAPE = R_d * sum(max(0, T_parcel - T_env) * dp / p)
   277	
   278	    where dp is the layer pressure thickness and the sum is over
   279	    all levels where the parcel is warmer than the environment.
   280	
   281	    Parameters
   282	    ----------
   283	    T_env : jax.Array
   284	        Environmental temperature [K], shape (ncol, nlev).
   285	    T_parcel : jax.Array
   286	        Parcel temperature [K], shape (ncol, nlev).
   287	    p_full : jax.Array
   288	        Pressure at full levels [Pa], shape (ncol, nlev).
   289	    p_half : jax.Array
   290	        Pressure at half levels [Pa], shape (ncol, nlev+1).
   291	
   292	    Returns
   293	    -------
   294	    jax.Array
   295	        CAPE [J/kg], shape (ncol,).
   296	    """
   297	    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)
   298	    buoyancy = jnp.maximum(0.0, T_parcel - T_env)
   299	
   300	    return constants.R_d * jnp.sum(buoyancy * dp / p_full, axis=1)

 succeeded in 0ms:
     1	"""Lightweight saturation thermodynamics for legoESM.
     2	
     3	This module provides `saturation_mixing_ratio` and
     4	`saturation_mixing_ratio_ice` with *no* dependency on
     5	``atmosphere.physics`` so that ``land/``, ``ice/``, ``ocean/``, and
     6	``coupler/`` modules can import them without pulling in the full
     7	atmosphere physics package.
     8	
     9	All operations are pure JAX and compatible with jit, grad, vmap, scan.
    10	"""
    11	
    12	from __future__ import annotations
    13	
    14	import jax
    15	import jax.numpy as jnp
    16	
    17	from legoesm import constants
    18	
    19	
    20	def saturation_vapor_pressure(T: jax.Array) -> jax.Array:
    21	    """Compute saturation vapor pressure using Tetens formula.
    22	
    23	    e_sat = 611.2 * exp(17.67 * T_c / (T_c + 243.5))
    24	
    25	    Parameters
    26	    ----------
    27	    T : jax.Array
    28	        Temperature [K].
    29	
    30	    Returns
    31	    -------
    32	    jax.Array
    33	        Saturation vapor pressure [Pa].
    34	    """
    35	    T_c = T - constants.T_freeze
    36	    return 611.2 * jnp.exp(17.67 * T_c / (T_c + 243.5))
    37	
    38	
    39	def saturation_mixing_ratio(
    40	    T: jax.Array,
    41	    p: jax.Array,
    42	) -> jax.Array:
    43	    """Compute saturation mixing ratio using Tetens formula.
    44	
    45	    e_sat = 611.2 * exp(17.67 * T_c / (T_c + 243.5))   where T_c = T - 273.15
    46	    q_sat = epsilon * e_sat / (p - e_sat)
    47	
    48	    Parameters
    49	    ----------
    50	    T : jax.Array
    51	        Temperature [K].
    52	    p : jax.Array
    53	        Pressure [Pa].
    54	
    55	    Returns
    56	    -------
    57	    jax.Array
    58	        Saturation mixing ratio [kg/kg].
    59	    """
    60	    e_sat = saturation_vapor_pressure(T)
    61	    # Smooth floor on denominator: preserves gradients near e_sat ≈ p
    62	    # instead of a hard clip that creates a zero-gradient plateau.
    63	    # softplus(x - 1) + 1 ≈ x for x >> 1, ≈ 1 for x << 1, smooth at x = 1.
    64	    denom = jax.nn.softplus(p - e_sat - 1.0) + 1.0
    65	    q_sat = constants.epsilon * e_sat / denom
    66	    # Smooth cap at 1.0 kg/kg: prevents singularity at low-pressure levels
    67	    # while allowing gradients to flow (unlike hard jnp.minimum).
    68	    # Uses LogSumExp smooth-min: 1 - softplus(β(1 - x))/β with β = 20.
    69	    return 1.0 - jax.nn.softplus(20.0 * (1.0 - q_sat)) / 20.0
    70	
    71	
    72	def saturation_mixing_ratio_ice(
    73	    T: jax.Array,
    74	    p: jax.Array,
    75	) -> jax.Array:
    76	    """Compute saturation mixing ratio over ice (Clausius-Clapeyron).
    77	
    78	    e_sat_i = 611.2 * exp(L_s/R_v * (1/T_freeze - 1/T))
    79	    q_sat_i = epsilon * e_sat_i / (p - e_sat_i)
    80	
    81	    Parameters
    82	    ----------
    83	    T : jax.Array
    84	        Temperature [K].
    85	    p : jax.Array
    86	        Pressure [Pa].
    87	
    88	    Returns
    89	    -------
    90	    jax.Array
    91	        Ice saturation mixing ratio [kg/kg].
    92	    """
    93	    e_sat_i = 611.2 * jnp.exp(
    94	        constants.L_s / constants.R_v * (1.0 / constants.T_freeze - 1.0 / T)
    95	    )
    96	    denom = jax.nn.softplus(p - e_sat_i - 1.0) + 1.0
    97	    q_sat_i = constants.epsilon * e_sat_i / denom
    98	    return 1.0 - jax.nn.softplus(20.0 * (1.0 - q_sat_i)) / 20.0
    99	
   100	
   101	def saturation_specific_humidity(
   102	    T: jax.Array,
   103	    p: jax.Array,
   104	) -> jax.Array:
   105	    """Compute saturation specific humidity from saturation mixing ratio.
   106	
   107	    q = w_sat / (1 + w_sat)
   108	
   109	    where w_sat = epsilon * e_sat / (p - e_sat) is the saturation mixing
   110	    ratio.  Use this function when working with specific humidity fields
   111	    (q = m_v / (m_v + m_d)) rather than mixing ratio (w = m_v / m_d).
   112	
   113	    Parameters
   114	    ----------
   115	    T : jax.Array
   116	        Temperature [K].
   117	    p : jax.Array
   118	        Pressure [Pa].
   119	
   120	    Returns
   121	    -------
   122	    jax.Array
   123	        Saturation specific humidity [kg/kg].
   124	    """
   125	    w_sat = saturation_mixing_ratio(T, p)
   126	    return w_sat / (1.0 + w_sat)

 succeeded in 0ms:
   930	        def loss(T_in):
   931	            out, _ = edmf_convection(
   932	                T_in, q_v, p_full, p_half, a_u, dt=300.0, config=config,
   933	            )
   934	            return jnp.sum(out.dT_dt ** 2)
   935	
   936	        grad_T = jax.grad(loss)(T)
   937	        assert jnp.all(jnp.isfinite(grad_T))
   938	        assert grad_T.shape == T.shape
   939	
   940	
   941	# ===========================================================================
   942	# Physics-correctness tests for plume schemes
   943	# ===========================================================================
   944	
   945	class TestMassFluxPhysics:
   946	    """Physics-correctness tests for the mass-flux scheme."""
   947	
   948	    def test_subsidence_warms_troposphere(self):
   949	        """Compensating subsidence should produce net warming in the mid-troposphere
   950	        where lapse rate is negative (T decreases with height)."""
   951	        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
   952	        config = MassFluxConfig(
   953	            M_scale=0.05, cape_threshold=0.0,
   954	            delta_0=0.0,  # disable detrainment to isolate subsidence
   955	        )
   956	        ncol = T.shape[0]
   957	        # Use a large M_c to see clear subsidence signal
   958	        M_c = jnp.full(ncol, 0.05)
   959	        out, _ = mass_flux_convection(
   960	            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
   961	        )
   962	        # Mid-troposphere levels (away from boundaries where gradient is zero)
   963	        mid = slice(5, 15)
   964	        dT_mid = out.dT_dt[:, mid]
   965	        # Subsidence should warm (positive dT/dt) in the troposphere
   966	        mean_warming = jnp.mean(dT_mid)
   967	        assert float(mean_warming) > 0, (
   968	            f"Subsidence should warm mid-troposphere, got mean dT/dt = {float(mean_warming):.2e}"
   969	        )
   970	
   971	    def test_detrainment_warms_where_updraft_warmer(self):
   972	        """Detrainment of warm updraft air should warm the environment."""
   973	        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
   974	        config = MassFluxConfig(
   975	            M_scale=0.05, cape_threshold=0.0,
   976	            epsilon_0=0.0,  # no entrainment: T_u stays on moist adiabat
   977	        )
   978	        ncol = T.shape[0]
   979	        M_c = jnp.full(ncol, 0.05)
   980	        out, _ = mass_flux_convection(
   981	            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
   982	        )
   983	        # With conditionally unstable profile, updraft (moist adiabat) is
   984	        # warmer than environment at upper levels where CAPE > 0
   985	        assert jnp.any(out.dT_dt > 0), "Detrainment should produce some warming"
   986	
   987	    def test_subsidence_dries_troposphere(self):
   988	        """Compensating subsidence should produce drying (dq/dt < 0) in
   989	        the mid-troposphere where moisture decreases with height."""
   990	        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
   991	        config = MassFluxConfig(
   992	            M_scale=0.05, cape_threshold=0.0,
   993	            delta_0=0.0,  # disable detrainment to isolate subsidence
   994	        )
   995	        ncol = T.shape[0]
   996	        M_c = jnp.full(ncol, 0.05)
   997	        out, _ = mass_flux_convection(
   998	            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
   999	        )
  1000	        # Moisture typically decreases with height, so subsidence brings
  1001	        # drier air down: dq/dt should be mostly negative
  1002	        mid = slice(5, 15)
  1003	        mean_dq = jnp.mean(out.dq_v_dt[:, mid])
  1004	        assert float(mean_dq) < 0, (
  1005	            f"Subsidence should dry mid-troposphere, got mean dq/dt = {float(mean_dq):.2e}"
  1006	        )
  1007	
  1008	    def test_precipitation_non_negative(self):
  1009	        """Precipitation should always be >= 0."""
  1010	        T, q_v, p_full, p_half = _make_unstable_columns()
  1011	        ncol = T.shape[0]
  1012	        config = MassFluxConfig()
  1013	        M_c = jnp.full(ncol, config.M_c_init)
  1014	        out, _ = mass_flux_convection(
  1015	            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
  1016	        )
  1017	        assert jnp.all(out.dq_c_conv_dt >= 0)
  1018	
  1019	
  1020	class TestEDMFPhysics:
  1021	    """Physics-correctness tests for the EDMF scheme."""
  1022	
  1023	    def test_subsidence_warms_troposphere(self):
  1024	        """Compensating subsidence should warm the mid-troposphere."""
  1025	        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
  1026	        config = EDMFConfig(
  1027	            a_u_init=0.1, cape_threshold=0.0,
  1028	            delta_0=0.0,  # disable detrainment to isolate subsidence
  1029	        )
  1030	        ncol = T.shape[0]
  1031	        a_u = jnp.full(ncol, 0.1)
  1032	        out, _ = edmf_convection(
  1033	            T, q_v, p_full, p_half, a_u, dt=300.0, config=config,
  1034	        )
  1035	        mid = slice(5, 15)
  1036	        mean_warming = jnp.mean(out.dT_dt[:, mid])
  1037	        assert float(mean_warming) > 0, (
  1038	            f"Subsidence should warm mid-troposphere, got mean dT/dt = {float(mean_warming):.2e}"
  1039	        )
  1040	
  1041	    def test_detrainment_warms_where_updraft_warmer(self):
  1042	        """Detrainment of warm updraft air should warm the environment."""
  1043	        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
  1044	        config = EDMFConfig(
  1045	            a_u_init=0.1, cape_threshold=0.0,
  1046	            epsilon_0=0.0,  # no entrainment: T_u stays on moist adiabat
  1047	        )
  1048	        ncol = T.shape[0]
  1049	        a_u = jnp.full(ncol, 0.1)
  1050	        out, _ = edmf_convection(
  1051	            T, q_v, p_full, p_half, a_u, dt=300.0, config=config,
  1052	        )
  1053	        assert jnp.any(out.dT_dt > 0), "Detrainment should produce some warming"
  1054	
  1055	    def test_both_terms_contribute(self):
  1056	        """With both subsidence and detrainment active, tendency should be
  1057	        larger than either alone."""
  1058	        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
  1059	        ncol = T.shape[0]
  1060	        a_u = jnp.full(ncol, 0.1)
  1061	
  1062	        # Subsidence only
  1063	        cfg_sub = EDMFConfig(a_u_init=0.1, cape_threshold=0.0, delta_0=0.0)
  1064	        out_sub, _ = edmf_convection(T, q_v, p_full, p_half, a_u, 300.0, cfg_sub)
  1065	
  1066	        # Detrainment only (epsilon_0=0 makes T_u=T_moist, strong detrainment)
  1067	        cfg_det = EDMFConfig(a_u_init=0.1, cape_threshold=0.0, epsilon_0=0.0)
  1068	        out_det, _ = edmf_convection(T, q_v, p_full, p_half, a_u, 300.0, cfg_det)
  1069	
  1070	        # Both active
  1071	        cfg_both = EDMFConfig(a_u_init=0.1, cape_threshold=0.0)
  1072	        out_both, _ = edmf_convection(T, q_v, p_full, p_half, a_u, 300.0, cfg_both)
  1073	
  1074	        # RMS of combined should generally be larger than either alone
  1075	        rms_sub = jnp.sqrt(jnp.mean(out_sub.dT_dt ** 2))
  1076	        rms_det = jnp.sqrt(jnp.mean(out_det.dT_dt ** 2))
  1077	        rms_both = jnp.sqrt(jnp.mean(out_both.dT_dt ** 2))
  1078	
  1079	        # At minimum, both-active should be non-zero
  1080	        assert float(rms_both) > 1e-10
  1081	
  1082	    def test_subsidence_dries_troposphere(self):
  1083	        """Compensating subsidence should dry the mid-troposphere."""
  1084	        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
  1085	        config = EDMFConfig(
  1086	            a_u_init=0.1, cape_threshold=0.0,
  1087	            delta_0=0.0,  # isolate subsidence
  1088	        )
  1089	        ncol = T.shape[0]
  1090	        a_u = jnp.full(ncol, 0.1)
  1091	        out, _ = edmf_convection(
  1092	            T, q_v, p_full, p_half, a_u, dt=300.0, config=config,
  1093	        )
  1094	        mid = slice(5, 15)
  1095	        mean_dq = jnp.mean(out.dq_v_dt[:, mid])
  1096	        assert float(mean_dq) < 0, (
  1097	            f"Subsidence should dry mid-troposphere, got mean dq/dt = {float(mean_dq):.2e}"
  1098	        )
  1099	
  1100	    def test_precipitation_non_negative(self):
  1101	        """Convective cloud-water source should always be >= 0."""
  1102	        T, q_v, p_full, p_half = _make_unstable_columns()
  1103	        ncol = T.shape[0]
  1104	        config = EDMFConfig()
  1105	        a_u = jnp.full(ncol, config.a_u_init)
  1106	        out, _ = edmf_convection(
  1107	            T, q_v, p_full, p_half, a_u, dt=300.0, config=config,
  1108	        )
  1109	        assert jnp.all(out.dq_c_conv_dt >= 0)

exec
/bin/zsh -lc "rg -n \"compute_cin\\(|compute_lfc_lnb\\(|entraining_detraining_plume\\(|cmt_gregory_1997\\(\" src tests -S" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/atmosphere/physics/convection/tiedtke.py:124:    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)
src/legoesm/atmosphere/physics/convection/tiedtke.py:213:    plume = entraining_detraining_plume(
src/legoesm/atmosphere/physics/convection/tiedtke.py:295:        du_dt_conv, dv_dt_conv = cmt_gregory_1997(
src/legoesm/atmosphere/physics/convection/bechtold.py:154:    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)
src/legoesm/atmosphere/physics/convection/bechtold.py:239:    plume = entraining_detraining_plume(
src/legoesm/atmosphere/physics/convection/bechtold.py:302:        du_dt_conv, dv_dt_conv = cmt_gregory_1997(
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:166:    plume = entraining_detraining_plume(
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:191:        du_dt_conv, dv_dt_conv = cmt_gregory_1997(
src/legoesm/atmosphere/physics/convection/_plume.py:197:def compute_lfc_lnb(
src/legoesm/atmosphere/physics/convection/_plume.py:259:def compute_cin(
src/legoesm/atmosphere/physics/convection/_plume.py:356:def entraining_detraining_plume(
src/legoesm/atmosphere/physics/convection/_plume.py:562:def cmt_gregory_1997(
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:148:    k_lfc_smooth, k_lnb_smooth = compute_lfc_lnb(T, T_moist, sharpness=1.0)
src/legoesm/atmosphere/physics/convection/kain_fritsch.py:183:    plume = entraining_detraining_plume(
src/legoesm/atmosphere/physics/convection/emanuel.py:136:    plume = entraining_detraining_plume(
tests/unit/test_physics_convection.py:380:    du_dt_func, _ = cmt_gregory_1997(
tests/unit/test_convection_plume.py:161:    k_lfc, k_lnb = P.compute_lfc_lnb(T_env, T_ma, sharpness=1.0)
tests/unit/test_convection_plume.py:180:        k_lfc, _ = P.compute_lfc_lnb(T_perturbed, T_ma, sharpness=1.0)
tests/unit/test_convection_plume.py:197:    k_lfc, _ = P.compute_lfc_lnb(T_env, T_ma, sharpness=1.0)
tests/unit/test_convection_plume.py:199:    cin = P.compute_cin(T_env, T_ma, p_full, p_half, lcl.k_lcl_smooth, k_lfc)
tests/unit/test_convection_plume.py:212:    k_lfc, _ = P.compute_lfc_lnb(T_env, T_ma, sharpness=1.0)
tests/unit/test_convection_plume.py:213:    cin = P.compute_cin(T_env, T_ma, p_full, p_half, lcl.k_lcl_smooth, k_lfc)
tests/unit/test_convection_plume.py:233:    plume = P.entraining_detraining_plume(
tests/unit/test_convection_plume.py:263:    plume = P.entraining_detraining_plume(
tests/unit/test_convection_plume.py:296:    plume = P.entraining_detraining_plume(
tests/unit/test_convection_plume.py:323:        plume = P.entraining_detraining_plume(
tests/unit/test_convection_plume.py:351:    du, dv = P.cmt_gregory_1997(u, v, M_u, None, p_full, p_half, rho)
tests/unit/test_convection_plume.py:367:    du, dv = P.cmt_gregory_1997(u, v, M_u, None, p_full, p_half, rho)
tests/unit/test_convection_plume.py:389:        du, _ = P.cmt_gregory_1997(

codex
The scheme wrappers introduce more concrete candidates: the LFC/LNB helper appears to mix surface-last and surface-first indices, and the Tiedtke/Bechtold detrainment rescale changes more than detrainment. I’m checking these against the local tests and thermodynamics helpers so the final review separates definite bugs from design choices.
exec
/bin/zsh -lc "nl -ba tests/unit/test_physics_convection.py | sed -n '340,410p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba tests/unit/test_convection_plume.py | sed -n '1,430p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   340	    # Strict monotonic in the transition region (below saturation).
   341	    transition = gate[:5]
   342	    assert jnp.all(jnp.diff(transition) > 0), (
   343	        "gate must be strictly monotonic across the transition region "
   344	        f"(35–134 hPa); got {[float(v) for v in transition]}"
   345	    )
   346	
   347	
   348	def test_cmt_gregory_1997_applies_stratospheric_gate():
   349	    """``cmt_gregory_1997`` must apply the same stratospheric gate the
   350	    kernel uses, so the default-enabled CMT path in ZM/Tiedtke/Bechtold
   351	    cannot dump convective momentum into the model top.
   352	
   353	    Codex caught that the kernel gate covered T/q_v but the CMT call
   354	    used the ungated ``plume.M_u``, leaving wind-driven dycore
   355	    instabilities (e.g. KF blowup at day 10) on the table.
   356	
   357	    Contract: with uniform ``M_u`` the function's output must equal
   358	    the same formula evaluated at ``M_u * gate(p_full)`` — that is the
   359	    operational definition of "the gate is applied inside the function".
   360	    """
   361	    from legoesm.atmosphere.physics.convection._plume import cmt_gregory_1997
   362	
   363	    ncol, nlev = 1, 8
   364	    p_full = jnp.array([[3_470.0, 5_000.0, 8_430.0, 10_000.0,
   365	                         13_370.0, 20_000.0, 50_000.0, 100_000.0]])
   366	    p_half = jnp.concatenate([
   367	        jnp.array([[0.0]]),
   368	        0.5 * (p_full[:, :-1] + p_full[:, 1:]),
   369	        jnp.array([[101_300.0]]),
   370	    ], axis=-1)
   371	    # Non-uniform shear so the ungated calculation has a non-zero
   372	    # divergence at every level (uniform du_layer + uniform M_u ⇒
   373	    # uniform flux ⇒ zero divergence in the interior).
   374	    u_env = jnp.array([[60.0, 50.0, 40.0, 25.0, 12.0, 5.0, 2.0, 0.0]])
   375	    v_env = jnp.zeros((ncol, nlev))
   376	    M_u = jnp.full((ncol, nlev), 0.05)
   377	    rho = p_full / (287.0 * 250.0)
   378	    c_u = 0.55
   379	
   380	    du_dt_func, _ = cmt_gregory_1997(
   381	        u_env, v_env, M_u, None, p_full, p_half, rho, c_u=c_u, c_d=c_u,
   382	    )
   383	
   384	    # Reference: replicate the formula exactly with the gate applied
   385	    # to M_u once.  If the function gates internally, this matches.
   386	    gate = stratosphere_mass_flux_gate(p_full)
   387	    dp = p_half[:, 1:] - p_half[:, :-1]
   388	    du_layer = jnp.diff(u_env, axis=-1, prepend=u_env[:, :1])
   389	    flux_ref = -c_u * (M_u * gate) * du_layer
   390	    dflux_ref = jnp.diff(flux_ref, axis=-1, append=flux_ref[:, -1:])
   391	    g = 9.80616
   392	    du_dt_ref = -g * dflux_ref / dp
   393	
   394	    # The function output must equal the gated reference exactly.
   395	    rel_err = float(jnp.max(jnp.abs(du_dt_func - du_dt_ref)
   396	                            / (jnp.abs(du_dt_ref) + 1e-30)))
   397	    assert rel_err < 1e-5, (
   398	        f"cmt_gregory_1997 output disagrees with gated reference "
   399	        f"by {rel_err:.3e}; the function must apply "
   400	        "stratosphere_mass_flux_gate to M_u (and M_d when present)"
   401	    )
   402	
   403	    # Sanity contrast with the *ungated* calculation — at the model
   404	    # top the function output must be much smaller than the ungated
   405	    # value (gate ≈ 0.013 at 35 hPa).
   406	    flux_ungated = -c_u * M_u * du_layer
   407	    dflux_ungated = jnp.diff(
   408	        flux_ungated, axis=-1, append=flux_ungated[:, -1:],
   409	    )
   410	    du_dt_ungated = -g * dflux_ungated / dp

 succeeded in 0ms:
     1	"""Unit tests for ``legoesm.atmosphere.physics.convection._plume``.
     2	
     3	The five column-physics workhorses delivered in PR 0:
     4	
     5	* :func:`compute_lcl`                     — Bolton (1980) LCL.
     6	* :func:`compute_lfc_lnb`                 — smooth fractional indices.
     7	* :func:`compute_cin`                     — convective inhibition [J/kg].
     8	* :func:`entraining_detraining_plume`     — vmappable updraft integrator.
     9	* :func:`cmt_gregory_1997`                — Gregory et al. 1997 CMT closure.
    10	
    11	These tests pin:
    12	
    13	* Bolton 1980 reference LCL (an unsaturated parcel test that exercises
    14	  the formula non-trivially);
    15	* finite gradients through every helper at non-trivial input values;
    16	* analytical limits of the plume integrator (no entrainment ⇒ moist
    17	  adiabat; strong entrainment ⇒ environmental relaxation; sub-cloud
    18	  layer is ``M_u = 0``);
    19	* CMT zero-mass-flux invariant and shear-sign convention.
    20	
    21	The tests run on CPU regardless of the host's JAX backend so that the
    22	suite is reproducible on the Apple-Silicon Metal hosts used during
    23	development.
    24	"""
    25	
    26	from __future__ import annotations
    27	
    28	import jax
    29	import jax.numpy as jnp
    30	import numpy as np
    31	import pytest
    32	
    33	from legoesm import constants
    34	from legoesm.thermo import saturation_mixing_ratio
    35	from legoesm.atmosphere.physics.thermodynamics import compute_moist_adiabat
    36	from legoesm.atmosphere.physics.convection import _plume as P
    37	
    38	
    39	# ---------------------------------------------------------------------------
    40	# Synthetic-column helpers reused across tests
    41	# ---------------------------------------------------------------------------
    42	
    43	def _synthetic_column(
    44	    ncol: int = 2,
    45	    nlev: int = 16,
    46	    *,
    47	    p_s: float = 1.0e5,
    48	    p_top: float = 5.0e3,
    49	    T_sfc: float = 295.0,
    50	    lapse_rate_K_per_km: float = 6.5,
    51	    q_sfc: float = 8.0e-3,
    52	    q_scale_height_m: float = 3000.0,
    53	):
    54	    """Surface-last column with a stable troposphere and an exponential
    55	    moisture profile.
    56	
    57	    Returns (T_env, q_v_env, p_full, p_half, z_full).  The default
    58	    ``q_sfc=8 g/kg`` gives an unsaturated parcel at the surface, which
    59	    exercises Bolton's LCL formula non-trivially (a saturated parcel
    60	    short-circuits to ``T_LCL = T_parcel``).
    61	    """
    62	    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)         # surface=1.0
    63	    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)   # (ncol, nlev)
    64	
    65	    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    66	    p_half = jnp.concatenate(
    67	        [
    68	            jnp.full((ncol, 1), p_top * 0.5),  # half-level above TOA
    69	            p_half_inner,
    70	            jnp.full((ncol, 1), p_s),
    71	        ],
    72	        axis=1,
    73	    )
    74	
    75	    H = 8500.0  # scale height [m]
    76	    z_full = -H * jnp.log(p_full / p_s)                  # surface=0
    77	    T_env = (
    78	        jnp.full((ncol,), T_sfc)[:, None]
    79	        - (lapse_rate_K_per_km * 1e-3) * z_full
    80	    )
    81	    q_v_env = q_sfc * jnp.exp(-z_full / q_scale_height_m)
    82	
    83	    return T_env, q_v_env, p_full, p_half, z_full
    84	
    85	
    86	# ---------------------------------------------------------------------------
    87	# compute_lcl — Bolton 1980 reference
    88	# ---------------------------------------------------------------------------
    89	
    90	def test_compute_lcl_unsaturated_parcel_bolton_reference():
    91	    """Bolton 1980 Eq. 22 against a reference parcel.
    92	
    93	    Parcel state: ``T = 290 K``, ``q = 8 g/kg``, ``p = 1000 hPa``.
    94	    For these inputs the Bolton formula produces ``T_LCL ≈ 282.6 K``
    95	    and Poisson's relation gives ``p_LCL ≈ 88,250 Pa`` — well above
    96	    the LCL coincidence (where the parcel is saturated and LCL =
    97	    parcel level).  Tolerances are generous because Bolton's formula
    98	    is itself an empirical fit accurate to ~0.5 K.
    99	    """
   100	    T_parcel = jnp.asarray([290.0])
   101	    q_parcel = jnp.asarray([8.0e-3])
   102	    p_parcel = jnp.asarray([1.0e5])
   103	
   104	    nlev = 16
   105	    p_full = jnp.linspace(5.0e3, 1.0e5, nlev)[None, :]
   106	
   107	    lcl = P.compute_lcl(T_parcel, q_parcel, p_parcel, p_full)
   108	
   109	    # T_LCL: literature value ~283 K (Bolton 1980 produces values in
   110	    # the 280–284 K window for this input depending on which form of
   111	    # the formula one cites).  Allow a 2 K tolerance.
   112	    assert pytest.approx(283.0, abs=2.0) == float(lcl.T_lcl[0])
   113	    # p_LCL: between 86 and 92 kPa for this parcel.
   114	    assert 8.6e4 < float(lcl.p_lcl[0]) < 9.2e4
   115	    # Smooth fractional level index lies in the valid range.
   116	    k_lcl = float(lcl.k_lcl_smooth[0])
   117	    assert 0.0 <= k_lcl <= float(nlev - 1)
   118	
   119	
   120	def test_compute_lcl_saturated_parcel_returns_parcel_level():
   121	    """If the parcel is already saturated (or supersaturated), the
   122	    Bolton formula gives ``T_LCL = T_parcel`` and ``p_LCL = p_parcel``."""
   123	    T = jnp.asarray([290.0])
   124	    p = jnp.asarray([1.0e5])
   125	    q_sat = saturation_mixing_ratio(T, p)
   126	    q_super = q_sat * 1.5  # supersaturated — should still produce LCL = parcel level
   127	
   128	    p_full = jnp.linspace(5.0e3, 1.0e5, 8)[None, :]
   129	    lcl = P.compute_lcl(T, q_super, p, p_full)
   130	    assert pytest.approx(float(T[0]), rel=1e-6) == float(lcl.T_lcl[0])
   131	    assert pytest.approx(float(p[0]), rel=1e-6) == float(lcl.p_lcl[0])
   132	
   133	
   134	def test_compute_lcl_grad_through_temperature():
   135	    """``d T_LCL / d T_parcel`` is finite — preserves training signal
   136	    flowing through the Bolton formula."""
   137	    p = jnp.asarray([1.0e5])
   138	    q = jnp.asarray([8.0e-3])
   139	    p_full = jnp.linspace(5.0e3, 1.0e5, 8)[None, :]
   140	
   141	    def f(T_p):
   142	        return P.compute_lcl(T_p, q, p, p_full).T_lcl[0]
   143	
   144	    g = float(jax.grad(f)(jnp.asarray(290.0)))
   145	    assert np.isfinite(g)
   146	    assert abs(g) > 1e-3
   147	
   148	
   149	# ---------------------------------------------------------------------------
   150	# compute_lfc_lnb — order and smooth-grad
   151	# ---------------------------------------------------------------------------
   152	
   153	def test_compute_lfc_lnb_capping_inversion_column():
   154	    """In a column with a capping inversion, the LFC sits above the
   155	    inversion and the LNB sits above the LFC."""
   156	    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
   157	        T_sfc=300.0, q_sfc=15.0e-3, lapse_rate_K_per_km=5.0,  # destabilized profile
   158	    )
   159	    T_sfc = T_env[:, -1]
   160	    T_ma = compute_moist_adiabat(T_sfc, p_full)
   161	    k_lfc, k_lnb = P.compute_lfc_lnb(T_env, T_ma, sharpness=1.0)
   162	
   163	    # In surface-last convention the LFC has a *larger* index than
   164	    # the LNB (closer to surface).  This ordering is the structural
   165	    # invariant.
   166	    assert float(k_lfc[0]) > float(k_lnb[0]), (
   167	        f"Surface-last ordering: LFC must have larger index than LNB; "
   168	        f"got LFC={float(k_lfc[0])}, LNB={float(k_lnb[0])}"
   169	    )
   170	
   171	
   172	def test_compute_lfc_lnb_grad_through_environment():
   173	    """``d k_lfc / d T_sfc`` is finite — training-signal preservation."""
   174	    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(T_sfc=298.0)
   175	    T_ma_base = compute_moist_adiabat(T_env[:, -1], p_full)
   176	
   177	    def f(dT):
   178	        T_perturbed = T_env + dT
   179	        T_ma = compute_moist_adiabat(T_env[:, -1] + dT, p_full)
   180	        k_lfc, _ = P.compute_lfc_lnb(T_perturbed, T_ma, sharpness=1.0)
   181	        return k_lfc[0]
   182	
   183	    g = float(jax.grad(f)(jnp.asarray(0.0)))
   184	    assert np.isfinite(g)
   185	
   186	
   187	# ---------------------------------------------------------------------------
   188	# compute_cin
   189	# ---------------------------------------------------------------------------
   190	
   191	def test_compute_cin_non_negative_and_finite():
   192	    """CIN is non-negative by construction (positive-part of negative
   193	    buoyancy) and finite for every column."""
   194	    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column()
   195	    T_ma = compute_moist_adiabat(T_env[:, -1], p_full)
   196	    lcl = P.compute_lcl(T_env[:, -1], q_v_env[:, -1], p_full[:, -1], p_full)
   197	    k_lfc, _ = P.compute_lfc_lnb(T_env, T_ma, sharpness=1.0)
   198	
   199	    cin = P.compute_cin(T_env, T_ma, p_full, p_half, lcl.k_lcl_smooth, k_lfc)
   200	    assert jnp.all(cin >= 0.0)
   201	    assert jnp.all(jnp.isfinite(cin))
   202	
   203	
   204	def test_compute_cin_zero_for_unstable_parcel_above_LCL():
   205	    """When the parcel is positively buoyant immediately above the
   206	    LCL (no cap), CIN ≈ 0."""
   207	    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
   208	        T_sfc=302.0, q_sfc=18.0e-3, lapse_rate_K_per_km=8.0,
   209	    )
   210	    T_ma = compute_moist_adiabat(T_env[:, -1], p_full)
   211	    lcl = P.compute_lcl(T_env[:, -1], q_v_env[:, -1], p_full[:, -1], p_full)
   212	    k_lfc, _ = P.compute_lfc_lnb(T_env, T_ma, sharpness=1.0)
   213	    cin = P.compute_cin(T_env, T_ma, p_full, p_half, lcl.k_lcl_smooth, k_lfc)
   214	    assert float(cin[0]) < 5.0   # J/kg
   215	
   216	
   217	# ---------------------------------------------------------------------------
   218	# entraining_detraining_plume
   219	# ---------------------------------------------------------------------------
   220	
   221	def test_plume_outputs_finite_and_correct_shape():
   222	    """Smoke: every output array is finite with the expected
   223	    ``(ncol, nlev)`` shape."""
   224	    ncol, nlev = 3, 14
   225	    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
   226	    T_base = T_env[:, -1]
   227	    q_base = q_v_env[:, -1]
   228	    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
   229	    eps = jnp.full((ncol, nlev), 5.0e-4)
   230	    dlt = jnp.full((ncol, nlev), 5.0e-4)
   231	    M_b = jnp.full((ncol,), 0.05)
   232	
   233	    plume = P.entraining_detraining_plume(
   234	        T_env, q_v_env, p_full, p_half, z_full,
   235	        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
   236	    )
   237	    for arr in (plume.M_u, plume.T_u, plume.q_u, plume.q_c_u, plume.B_u):
   238	        assert arr.shape == (ncol, nlev)
   239	        assert jnp.all(jnp.isfinite(arr))
   240	
   241	
   242	def test_plume_no_entrainment_limit_matches_moist_adiabat():
   243	    """With ``epsilon = delta = 0`` and a strongly buoyant parcel,
   244	    the in-cloud plume temperature follows the moist adiabat (within
   245	    a tolerance set by the discretization)."""
   246	    ncol, nlev = 1, 16
   247	    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(
   248	        ncol, nlev, T_sfc=300.0, q_sfc=18.0e-3, lapse_rate_K_per_km=8.0,
   249	    )
   250	    T_base = T_env[:, -1]
   251	    q_base = q_v_env[:, -1]
   252	
   253	    # Cloud base at surface.  With unsaturated air the plume condenses
   254	    # immediately above the surface; the moist adiabat starts from
   255	    # T_base.
   256	    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
   257	    T_ma = compute_moist_adiabat(T_base, p_full)
   258	
   259	    eps = jnp.zeros((ncol, nlev))
   260	    dlt = jnp.zeros((ncol, nlev))
   261	    M_b = jnp.full((ncol,), 0.05)
   262	
   263	    plume = P.entraining_detraining_plume(
   264	        T_env, q_v_env, p_full, p_half, z_full,
   265	        T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
   266	    )
   267	    # Sample three mid-troposphere levels and compare to the moist
   268	    # adiabat.  The first-order Euler ascent in the plume integrator
   269	    # accumulates discretization error, so the tolerance is generous.
   270	    sampled = jnp.array([nlev // 2, nlev // 2 + 1, nlev // 2 + 2])
   271	    plume_T = plume.T_u[0, sampled]
   272	    ma_T = T_ma[0, sampled]
   273	    diff = jnp.abs(plume_T - ma_T)
   274	    assert jnp.all(diff < 5.0), (
   275	        f"No-entrainment plume should track moist adiabat within ~5 K, "
   276	        f"got differences {np.asarray(diff)} K"
   277	    )
   278	
   279	
   280	def test_plume_sub_cloud_mass_flux_is_suppressed():
   281	    """Levels strictly below the cloud base have ``M_u ≈ 0`` because
   282	    the ``above_base_weight`` mask suppresses them."""
   283	    ncol, nlev = 1, 16
   284	    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
   285	    T_base = T_env[:, -1]
   286	    q_base = q_v_env[:, -1]
   287	    # Force a cloud-base index well above the surface by passing a
   288	    # synthetic ``k_base_smooth`` (mid-column) so we have several
   289	    # sub-cloud levels to inspect.
   290	    k_base_synthetic = jnp.full((ncol,), float(nlev) - 4.0)
   291	
   292	    eps = jnp.full((ncol, nlev), 5.0e-4)
   293	    dlt = jnp.full((ncol, nlev), 5.0e-4)
   294	    M_b = jnp.full((ncol,), 0.05)
   295	
   296	    plume = P.entraining_detraining_plume(
   297	        T_env, q_v_env, p_full, p_half, z_full,
   298	        T_base, q_base, k_base_synthetic, eps, dlt, M_b,
   299	        buoyancy_sharpness=2.0,
   300	    )
   301	    # Surface-last: levels with index > k_base are below the cloud
   302	    # base (lower altitude).
   303	    sub_cloud_mass = plume.M_u[:, -1]   # surface-most level
   304	    base_mass = plume.M_u[:, int(float(nlev) - 4.0)]
   305	    # Sub-cloud value should be much smaller than the cloud-base
   306	    # value.  Tolerance reflects the smoothness of the sigmoid mask.
   307	    assert float(sub_cloud_mass[0]) < 0.5 * float(base_mass[0])
   308	
   309	
   310	def test_plume_grad_through_epsilon():
   311	    """``d (sum M_u) / d epsilon_0`` is finite — the plume integrator
   312	    is differentiable through its tunables."""
   313	    ncol, nlev = 1, 12
   314	    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
   315	    T_base = T_env[:, -1]
   316	    q_base = q_v_env[:, -1]
   317	    lcl = P.compute_lcl(T_base, q_base, p_full[:, -1], p_full)
   318	    M_b = jnp.full((ncol,), 0.05)
   319	    dlt = jnp.full((ncol, nlev), 5.0e-4)
   320	
   321	    def f(eps_0):
   322	        eps = jnp.full((ncol, nlev), eps_0)
   323	        plume = P.entraining_detraining_plume(
   324	            T_env, q_v_env, p_full, p_half, z_full,
   325	            T_base, q_base, lcl.k_lcl_smooth, eps, dlt, M_b,
   326	        )
   327	        return jnp.sum(plume.M_u)
   328	
   329	    g = float(jax.grad(f)(jnp.asarray(5.0e-4)))
   330	    assert np.isfinite(g)
   331	
   332	
   333	# ---------------------------------------------------------------------------
   334	# cmt_gregory_1997
   335	# ---------------------------------------------------------------------------
   336	
   337	def test_cmt_zero_when_no_mass_flux():
   338	    """``M_u = M_d = 0`` ⇒ both wind tendencies are exactly zero."""
   339	    ncol, nlev = 2, 8
   340	    u = jnp.linspace(0.0, 30.0, nlev)[None, :].repeat(ncol, axis=0)
   341	    v = jnp.zeros((ncol, nlev))
   342	    M_u = jnp.zeros((ncol, nlev))
   343	    p_full = jnp.linspace(5.0e3, 1.0e5, nlev)[None, :].repeat(ncol, axis=0)
   344	    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
   345	    p_half = jnp.concatenate(
   346	        [jnp.full((ncol, 1), 2.5e3), p_half_inner, jnp.full((ncol, 1), 1.0e5)],
   347	        axis=1,
   348	    )
   349	    rho = p_full / (constants.R_d * jnp.full(p_full.shape, 280.0))
   350	
   351	    du, dv = P.cmt_gregory_1997(u, v, M_u, None, p_full, p_half, rho)
   352	    assert jnp.all(du == 0.0)
   353	    assert jnp.all(dv == 0.0)
   354	
   355	
   356	def test_cmt_finite_in_sheared_environment():
   357	    """Non-zero updraft mass flux in a sheared environment produces
   358	    finite, non-trivial momentum tendencies."""
   359	    ncol, nlev = 1, 12
   360	    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
   361	    rho = p_full / (constants.R_d * T_env)
   362	
   363	    u = jnp.linspace(0.0, 30.0, nlev)[None, :].repeat(ncol, axis=0)
   364	    v = jnp.zeros_like(u)
   365	    M_u = jnp.linspace(0.0, 0.05, nlev)[None, :].repeat(ncol, axis=0)
   366	
   367	    du, dv = P.cmt_gregory_1997(u, v, M_u, None, p_full, p_half, rho)
   368	    assert jnp.all(jnp.isfinite(du))
   369	    assert jnp.all(jnp.isfinite(dv))
   370	    # Some level has non-trivial du/dt — the tendency should not be
   371	    # uniformly zero in a sheared column with non-zero mass flux.
   372	    assert float(jnp.max(jnp.abs(du))) > 1e-8
   373	
   374	
   375	def test_cmt_grad_through_c_u():
   376	    """``d (sum du_dt) / d c_u`` is finite — the closure coefficient
   377	    is a tunable parameter and gradients must flow.  ``c_u`` is
   378	    declared as ``float`` in the signature, but the implementation
   379	    only multiplies it against array-valued shears so a traced array
   380	    works at call time."""
   381	    ncol, nlev = 1, 8
   382	    T_env, q_v_env, p_full, p_half, z_full = _synthetic_column(ncol, nlev)
   383	    rho = p_full / (constants.R_d * T_env)
   384	    u = jnp.linspace(0.0, 25.0, nlev)[None, :].repeat(ncol, axis=0)
   385	    v = jnp.zeros_like(u)
   386	    M_u = jnp.full((ncol, nlev), 0.03)
   387	
   388	    def f(c_u_val):
   389	        du, _ = P.cmt_gregory_1997(
   390	            u, v, M_u, None, p_full, p_half, rho, c_u=c_u_val, c_d=0.55,
   391	        )
   392	        return jnp.sum(du)
   393	
   394	    g = float(jax.grad(f)(jnp.asarray(0.55)))
   395	    assert np.isfinite(g)

exec
/bin/zsh -lc 'rg -n "Tiedtke|Bechtold|Emanuel|Kain|LNB|LFC|cloud_depth|delta_0_eff|downdraft" tests -S' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
tests/unit/test_bechtold.py:1:"""Unit tests for the Bechtold/IFS convection scheme.
tests/unit/test_bechtold.py:8:* downdraft toggle effect;
tests/unit/test_bechtold.py:40:    BechtoldConfig,
tests/unit/test_bechtold.py:105:        config=BechtoldConfig(use_pbl_cape=True),
tests/unit/test_bechtold.py:109:        config=BechtoldConfig(use_pbl_cape=False),
tests/unit/test_bechtold.py:136:        config=BechtoldConfig(enable_cmt=False),
tests/unit/test_bechtold.py:175:    config = BechtoldConfig(
tests/unit/test_bechtold.py:194:    config = BechtoldConfig(
tests/unit/test_bechtold.py:223:            config=BechtoldConfig(epsilon_deep=eps),
tests/unit/test_bechtold.py:240:            config=BechtoldConfig(cape_pbl_depth=depth),
tests/unit/test_bechtold.py:260:            config=BechtoldConfig(enable_stochastic=False, stochastic_amplitude=amp),
tests/unit/test_bechtold.py:345:            bechtold=BechtoldConfig(
tests/unit/test_bechtold.py:382:    """jax.grad through the orchestrator with stochastic Bechtold
tests/unit/test_bechtold.py:400:            bechtold=BechtoldConfig(
tests/unit/test_bechtold.py:440:            bechtold=BechtoldConfig(enable_stochastic=False),
tests/unit/test_bechtold.py:460:    With ``radiation=gray`` + ``convection=bechtold``, Bechtold is the
tests/unit/test_bechtold.py:463:    so Bechtold's multi-field dict (``conv_prog_profile`` /
tests/unit/test_bechtold.py:468:    step, even with a non-Bechtold module registered first.
tests/unit/test_bechtold.py:484:            bechtold=BechtoldConfig(
tests/unit/test_bechtold.py:509:        "when Bechtold is not the first tagged module."
tests/unit/test_bechtold.py:522:        "Bechtold inherits the standard Tiedtke kernel for env tendencies, "
tests/unit/test_bechtold.py:523:        "but its PBL-CAPE closure pushes M_b larger than Tiedtke's, so "
tests/unit/test_bechtold.py:539:        config=BechtoldConfig(enable_stochastic=False, enable_cmt=False),
tests/unit/test_bechtold.py:548:        f"Bechtold MSE residual {H+Q+C:.1f} W/m^2 ({rel*100:.1f}% of total)"
tests/unit/test_zhang_mcfarlane.py:192:    # Environment is heated above LFC and roughly conserved below;
tests/unit/test_emanuel.py:1:"""Unit tests for the Emanuel (1991) convection scheme.
tests/unit/test_emanuel.py:9:* the unsaturated-downdraft toggle (column conservation under both
tests/unit/test_emanuel.py:33:    EmanuelConfig,
tests/unit/test_emanuel.py:99:    the unsaturated-downdraft evaporation step subtracts column
tests/unit/test_emanuel.py:115:    approach Emanuel 1991's 50-bin spectrum more closely."""
tests/unit/test_emanuel.py:120:        config = EmanuelConfig(n_mixing_fractions=n)
tests/unit/test_emanuel.py:136:        config=EmanuelConfig(cu_coefficient=0.0),
tests/unit/test_emanuel.py:140:        config=EmanuelConfig(cu_coefficient=1.0),
tests/unit/test_emanuel.py:149:# Unsaturated downdraft toggle
tests/unit/test_emanuel.py:152:def test_emanuel_downdraft_toggle_changes_subcloud_dT():
tests/unit/test_emanuel.py:153:    """Enabling the unsaturated downdraft introduces an additional
tests/unit/test_emanuel.py:162:        config=EmanuelConfig(enable_unsaturated_downdraft=False),
tests/unit/test_emanuel.py:166:        config=EmanuelConfig(enable_unsaturated_downdraft=True),
tests/unit/test_emanuel.py:169:    # branches — the downdraft is doing something visible.
tests/unit/test_emanuel.py:184:        config = EmanuelConfig(cape_threshold=threshold)
tests/unit/test_emanuel.py:198:        config = EmanuelConfig(smooth_trigger_sharpness=s)
tests/unit/test_emanuel.py:248:    # Emanuel has no CMT — orchestrator zeros these.
tests/unit/test_emanuel.py:283:        f"Emanuel MSE residual {H+Q+C:.1f} W/m^2 ({rel*100:.1f}% of total)"
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:327:        """In a CAPE-positive moist column, Tiedtke's q_v tendency
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py:331:        Direction-of-change is not asserted because Tiedtke's
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py:6:to feed the Kain-Fritsch BL trigger.
tests/unit/test_kain_fritsch.py:1:"""Unit tests for the Kain & Fritsch (1990, 2004) convection scheme.
tests/unit/test_kain_fritsch.py:33:    KainFritschConfig,
tests/unit/test_kain_fritsch.py:162:        config = KainFritschConfig(parcel_perturb_T=perturb_T)
tests/unit/test_kain_fritsch.py:180:        config = KainFritschConfig(w_thresh_offset=offset)
tests/unit/test_kain_fritsch.py:207:    config = KainFritschConfig(enable_shallow=False)
tests/unit/test_kain_fritsch.py:223:    config = KainFritschConfig(enable_shallow=False)
tests/unit/test_convection_triggers.py:177:    ``val_top > val_surf`` (the canonical LFC orientation).
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:35:    KainFritschConfig,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:36:    EmanuelConfig,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:37:    TiedtkeConfig,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:38:    BechtoldConfig,
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:349:        """Tiedtke's deep closure consumes ``moisture_convergence``.  Two
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:352:        different Tiedtke tendencies.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:411:        # The cloud-base mass-flux carry should differ — Tiedtke's
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:420:            "Tiedtke carry should respond to spectral-derived MC "
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:558:        # With the new tracer-aware dycore, Tiedtke's q_v sink is
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:615:# to any non-stochastic / non-w-grid / non-MC scheme — Emanuel's leaf
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:624:        ("kain_fritsch", "kain_fritsch", KainFritschConfig),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:625:        ("emanuel", "emanuel", EmanuelConfig),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:626:        ("tiedtke", "tiedtke", TiedtkeConfig),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:627:        ("bechtold", "bechtold", BechtoldConfig),
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:641:    tendencies.  This catches the spectral-PE Emanuel dispatch bug
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:707:            # Profile-prognostic carry can be a dict (Bechtold) or a
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:730:        schemes — Emanuel's leaf has no wind kwargs.
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:734:            ConvectionConfig(scheme="emanuel", emanuel=EmanuelConfig()),
tests/unit/test_physics_convection.py:350:    kernel uses, so the default-enabled CMT path in ZM/Tiedtke/Bechtold
tests/unit/test_physics_state_migration.py:5:profile-carrying schemes (Tiedtke, Bechtold) and scalar-carrying schemes
tests/unit/test_convection_plume.py:154:    """In a column with a capping inversion, the LFC sits above the
tests/unit/test_convection_plume.py:155:    inversion and the LNB sits above the LFC."""
tests/unit/test_convection_plume.py:163:    # In surface-last convention the LFC has a *larger* index than
tests/unit/test_convection_plume.py:164:    # the LNB (closer to surface).  This ordering is the structural
tests/unit/test_convection_plume.py:167:        f"Surface-last ordering: LFC must have larger index than LNB; "
tests/unit/test_convection_plume.py:168:        f"got LFC={float(k_lfc[0])}, LNB={float(k_lnb[0])}"
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:46:    KainFritschConfig,
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:47:    EmanuelConfig,
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:48:    TiedtkeConfig,
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:49:    BechtoldConfig,
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:108:        ("kain_fritsch", "kain_fritsch", KainFritschConfig),
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:109:        ("emanuel", "emanuel", EmanuelConfig),
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:110:        ("tiedtke", "tiedtke", TiedtkeConfig),
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:111:        ("bechtold", "bechtold", BechtoldConfig),
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:260:        the leaf, which silently bypassed Tiedtke's saturation-deficit
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:278:        cfg = ConvectionConfig(scheme="tiedtke", tiedtke=TiedtkeConfig())
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:287:        # MPAS Tiedtke path.  ``state.tracers is None`` → q_v_col is
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:341:            "Tiedtke MPAS bridge dT/dt should match the proxy-path "
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py:355:            "Tiedtke tendencies — the differential test cannot "
tests/unit/test_tiedtke.py:1:"""Unit tests for the Tiedtke (1989) convection scheme.
tests/unit/test_tiedtke.py:6:* the **profile carry** — Tiedtke is the first scheme that exercises
tests/unit/test_tiedtke.py:12:* downdraft toggle changes the sub-cloud T tendency;
tests/unit/test_tiedtke.py:14:  ``cmt_c_u``, and ``downdraft_alpha``;
tests/unit/test_tiedtke.py:36:    TiedtkeConfig,
tests/unit/test_tiedtke.py:92:    least one level above the surface — distinguishing Tiedtke from
tests/unit/test_tiedtke.py:102:        f"Tiedtke should carry M_u aloft; got max above surface = {aloft_max}"
tests/unit/test_tiedtke.py:112:    config = TiedtkeConfig(tau_M_u_relax=1800.0)
tests/unit/test_tiedtke.py:145:    config = TiedtkeConfig(enable_cmt=False)
tests/unit/test_tiedtke.py:155:def test_tiedtke_downdraft_toggle_changes_subcloud_T():
tests/unit/test_tiedtke.py:156:    """Enabling the downdraft branch changes the sub-cloud T
tests/unit/test_tiedtke.py:163:        config=TiedtkeConfig(enable_downdraft=False),
tests/unit/test_tiedtke.py:167:        config=TiedtkeConfig(enable_downdraft=True),
tests/unit/test_tiedtke.py:183:        config = TiedtkeConfig(epsilon_deep=eps)
tests/unit/test_tiedtke.py:197:        config = TiedtkeConfig(cape_threshold=threshold)
tests/unit/test_tiedtke.py:211:        config = TiedtkeConfig(cmt_c_u=c_u)
tests/unit/test_tiedtke.py:256:        "Standard Tiedtke kernel (subsidence g/c_p + detrainment of "
tests/unit/test_tiedtke.py:274:        config=TiedtkeConfig(enable_cmt=False),
tests/unit/test_tiedtke.py:283:        f"Tiedtke MSE residual {H+Q+C:.1f} W/m^2 ({rel*100:.1f}% of total) "
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py:14:  Tiedtke / Bechtold AD path).

exec
/bin/zsh -lc "nl -ba tests/unit/test_tiedtke.py | sed -n '1,320p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Unit tests for the Tiedtke (1989) convection scheme.
     2	
     3	Tests pin:
     4	
     5	* tendency shape / dtype / finiteness;
     6	* the **profile carry** — Tiedtke is the first scheme that exercises
     7	  ``conv_prog_profile = M_u(k)`` across levels (not just at
     8	  ``[:, -1]``);
     9	* the implicit-Euler relaxation of ``M_u`` toward the diagnosed
    10	  profile (monotone-contracting under quasi-equilibrium);
    11	* CMT signs and `enable_cmt=False` opt-out;
    12	* downdraft toggle changes the sub-cloud T tendency;
    13	* finite gradients through ``epsilon_deep``, ``cape_threshold``,
    14	  ``cmt_c_u``, and ``downdraft_alpha``;
    15	* selection through ``make_physics(PhysicsConfig(...))``.
    16	"""
    17	
    18	from __future__ import annotations
    19	
    20	from legoesm import constants
    21	
    22	import jax
    23	import jax.numpy as jnp
    24	import numpy as np
    25	import pytest
    26	
    27	from legoesm.core.field import Field
    28	from legoesm.grids.cubed_sphere import create_cubed_sphere
    29	from legoesm.grids.vertical import create_sigma_coordinate
    30	from legoesm.atmosphere.held_suarez import held_suarez_init
    31	from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    32	from legoesm.atmosphere.physics.physics_state import init_physics_state
    33	from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    34	from legoesm.atmosphere.physics.convection.config import (
    35	    ConvectionConfig,
    36	    TiedtkeConfig,
    37	)
    38	from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    39	from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    40	from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    41	from legoesm.atmosphere.physics.convection.tiedtke import tiedtke_convection
    42	
    43	
    44	def _column(
    45	    ncol=2, nlev=16, T_sfc=302.0, q_sfc=16e-3, lapse_rate=7.5,
    46	    p_s=1.0e5, p_top=5.0e3, u_sfc=2.0, u_top=25.0,
    47	):
    48	    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    49	    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    50	    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    51	    p_half = jnp.concatenate(
    52	        [
    53	            jnp.full((ncol, 1), p_top * 0.5),
    54	            p_half_inner,
    55	            jnp.full((ncol, 1), p_s),
    56	        ],
    57	        axis=1,
    58	    )
    59	    z_full = -8500.0 * jnp.log(p_full / p_s)
    60	    T = jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z_full
    61	    q = q_sfc * jnp.exp(-z_full / 3000.0)
    62	    fraction = jnp.linspace(1.0, 0.0, nlev)[None, :]   # 1 at top, 0 at surface
    63	    u = u_sfc + (u_top - u_sfc) * fraction
    64	    u = jnp.broadcast_to(u, (ncol, nlev))
    65	    v = jnp.zeros_like(u)
    66	    return T, q, p_full, p_half, u, v
    67	
    68	
    69	# ---------------------------------------------------------------------------
    70	# Shape / finiteness
    71	# ---------------------------------------------------------------------------
    72	
    73	def test_tiedtke_shape_finiteness():
    74	    T, q, pf, ph, u, v = _column(ncol=3, nlev=12)
    75	    ncol, nlev = T.shape
    76	    cpp = jnp.zeros((ncol, nlev))
    77	    out, M_u_new = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0)
    78	    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
    79	                out.convective_mask, out.du_dt_conv, out.dv_dt_conv,
    80	                M_u_new):
    81	        assert arr.shape[0] == ncol
    82	        assert jnp.all(jnp.isfinite(arr))
    83	    assert M_u_new.shape == (ncol, nlev)
    84	
    85	
    86	# ---------------------------------------------------------------------------
    87	# Profile carry — the first scheme that uses M_u(k) across levels
    88	# ---------------------------------------------------------------------------
    89	
    90	def test_tiedtke_profile_carry_distributes_across_levels():
    91	    """The diagnosed ``M_u_new`` carries non-trivial values on at
    92	    least one level above the surface — distinguishing Tiedtke from
    93	    the scalar-carry schemes that only populate ``[:, -1]``."""
    94	    T, q, pf, ph, u, v = _column()
    95	    ncol, nlev = T.shape
    96	    cpp = jnp.zeros((ncol, nlev))
    97	    _, M_u_new = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0)
    98	    # At least one level *above* the surface (index < nlev - 1) is
    99	    # non-trivially populated.
   100	    aloft_max = float(jnp.max(M_u_new[:, :-1]))
   101	    assert aloft_max > 1e-6, (
   102	        f"Tiedtke should carry M_u aloft; got max above surface = {aloft_max}"
   103	    )
   104	
   105	
   106	def test_tiedtke_implicit_relaxation_monotone():
   107	    """Repeated application with the same input drives ``M_u``
   108	    profile toward equilibrium with monotone-contracting differences."""
   109	    T, q, pf, ph, u, v = _column()
   110	    ncol, nlev = T.shape
   111	    cpp = jnp.zeros((ncol, nlev))
   112	    config = TiedtkeConfig(tau_M_u_relax=1800.0)
   113	    M_u_traj = []
   114	    for _ in range(15):
   115	        _, cpp = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=config)
   116	        M_u_traj.append(np.asarray(cpp[0]))
   117	    # Successive layer-summed differences shrink (relaxation).
   118	    diffs = [
   119	        float(jnp.sum(jnp.abs(M_u_traj[i + 1] - M_u_traj[i])))
   120	        for i in range(len(M_u_traj) - 1)
   121	    ]
   122	    # Allow some non-monotonicity early on while the profile builds
   123	    # up; check that the final differences are smaller than the
   124	    # initial.
   125	    assert diffs[-1] < diffs[0] + 1e-12
   126	
   127	
   128	# ---------------------------------------------------------------------------
   129	# CMT
   130	# ---------------------------------------------------------------------------
   131	
   132	def test_tiedtke_cmt_present_with_default_config():
   133	    T, q, pf, ph, u, v = _column()
   134	    ncol, nlev = T.shape
   135	    cpp = jnp.zeros((ncol, nlev))
   136	    out, _ = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0)
   137	    assert out.du_dt_conv is not None
   138	    assert out.dv_dt_conv is not None
   139	
   140	
   141	def test_tiedtke_cmt_disabled():
   142	    T, q, pf, ph, u, v = _column()
   143	    ncol, nlev = T.shape
   144	    cpp = jnp.zeros((ncol, nlev))
   145	    config = TiedtkeConfig(enable_cmt=False)
   146	    out, _ = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=config)
   147	    assert out.du_dt_conv is None
   148	    assert out.dv_dt_conv is None
   149	
   150	
   151	# ---------------------------------------------------------------------------
   152	# Downdraft
   153	# ---------------------------------------------------------------------------
   154	
   155	def test_tiedtke_downdraft_toggle_changes_subcloud_T():
   156	    """Enabling the downdraft branch changes the sub-cloud T
   157	    tendency."""
   158	    T, q, pf, ph, u, v = _column()
   159	    ncol, nlev = T.shape
   160	    cpp = jnp.zeros((ncol, nlev))
   161	    out_off, _ = tiedtke_convection(
   162	        T, q, pf, ph, u, v, cpp, dt=300.0,
   163	        config=TiedtkeConfig(enable_downdraft=False),
   164	    )
   165	    out_on, _ = tiedtke_convection(
   166	        T, q, pf, ph, u, v, cpp, dt=300.0,
   167	        config=TiedtkeConfig(enable_downdraft=True),
   168	    )
   169	    diff = jnp.abs(out_on.dT_dt[:, -4:] - out_off.dT_dt[:, -4:])
   170	    assert float(jnp.max(diff)) > 1e-8
   171	
   172	
   173	# ---------------------------------------------------------------------------
   174	# Differentiability
   175	# ---------------------------------------------------------------------------
   176	
   177	def test_tiedtke_grad_through_epsilon_deep():
   178	    T, q, pf, ph, u, v = _column()
   179	    ncol, nlev = T.shape
   180	    cpp = jnp.zeros((ncol, nlev))
   181	
   182	    def f(eps):
   183	        config = TiedtkeConfig(epsilon_deep=eps)
   184	        out, _ = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=config)
   185	        return jnp.sum(out.dT_dt)
   186	
   187	    g = float(jax.grad(f)(jnp.asarray(1.0e-4)))
   188	    assert np.isfinite(g)
   189	
   190	
   191	def test_tiedtke_grad_through_cape_threshold():
   192	    T, q, pf, ph, u, v = _column()
   193	    ncol, nlev = T.shape
   194	    cpp = jnp.zeros((ncol, nlev))
   195	
   196	    def f(threshold):
   197	        config = TiedtkeConfig(cape_threshold=threshold)
   198	        out, _ = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=config)
   199	        return jnp.sum(out.dT_dt)
   200	
   201	    g = float(jax.grad(f)(jnp.asarray(70.0)))
   202	    assert np.isfinite(g)
   203	
   204	
   205	def test_tiedtke_grad_through_cmt_c_u():
   206	    T, q, pf, ph, u, v = _column()
   207	    ncol, nlev = T.shape
   208	    cpp = jnp.zeros((ncol, nlev))
   209	
   210	    def f(c_u):
   211	        config = TiedtkeConfig(cmt_c_u=c_u)
   212	        out, _ = tiedtke_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=config)
   213	        return jnp.sum(out.du_dt_conv)
   214	
   215	    g = float(jax.grad(f)(jnp.asarray(0.7)))
   216	    assert np.isfinite(g)
   217	
   218	
   219	# ---------------------------------------------------------------------------
   220	# Orchestrator
   221	# ---------------------------------------------------------------------------
   222	
   223	def test_tiedtke_orchestrator_one_step_finite():
   224	    grid = create_cubed_sphere(4)
   225	    sigma = create_sigma_coordinate(12)
   226	    state = held_suarez_init(grid, sigma)
   227	    tracers = {
   228	        "q_v": Field(0.014 * jnp.ones((6, 4, 4, 12)), name="q_v",
   229	                     dims=("face", "x", "y", "level"), units="kg/kg"),
   230	        "q_c": Field(jnp.zeros((6, 4, 4, 12)), name="q_c",
   231	                     dims=("face", "x", "y", "level"), units="kg/kg"),
   232	    }
   233	    state = state._replace(tracers=tracers)
   234	    cfg = PhysicsConfig(
   235	        radiation=RadiationConfig(scheme="none"),
   236	        convection=ConvectionConfig(scheme="tiedtke"),
   237	        turbulence=TurbulenceConfig(scheme="none"),
   238	        microphysics=MicrophysicsConfig(scheme="none"),
   239	        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
   240	    )
   241	    ncol = 6 * 4 * 4
   242	    ps = init_physics_state(ncol, 12, cfg)
   243	    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
   244	    tend, ps_out = physics_fn(state, grid, sigma, phys_state=ps)
   245	    assert ps_out.conv_prog_profile.shape == (ncol, 12)
   246	    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
   247	        assert jnp.all(jnp.isfinite(f.data))
   248	
   249	
   250	# ---------------------------------------------------------------------------
   251	# MSE conservation guard
   252	# ---------------------------------------------------------------------------
   253	
   254	@pytest.mark.xfail(
   255	    reason=(
   256	        "Standard Tiedtke kernel (subsidence g/c_p + detrainment of "
   257	        "moist-adiabat T_u) does not conserve column MSE on a closed "
   258	        "(no-surface-flux) probe.  Currently ~96% non-conservation "
   259	        "residual on this CAPE-positive fixture; flagged xfail so any "
   260	        "future kernel improvement that closes this is detected."
   261	    ),
   262	    strict=True,
   263	)
   264	def test_tiedtke_mse_conservation_within_tolerance():
   265	    """Column-integrated ``c_p ∫dT + L_v ∫(dq_v + dq_c_conv) dp/g`` should
   266	    be small relative to the heating magnitude on a CAPE-positive sounding.
   267	    """
   268	    T, q, pf, ph, u, v = _column()
   269	    ncol, nlev = T.shape
   270	    cpp = jnp.zeros((ncol, nlev))
   271	    out, _ = tiedtke_convection(
   272	        T=T, q_v=q, p_full=pf, p_half=ph, u=u, v=v,
   273	        conv_prog_profile=cpp, dt=1800.0,
   274	        config=TiedtkeConfig(enable_cmt=False),
   275	        moisture_convergence=jnp.zeros_like(T),
   276	    )
   277	    dp = ph[:, 1:] - ph[:, :-1]
   278	    H = float(jnp.sum(out.dT_dt * dp / constants.g, axis=1).mean()) * constants.c_pd
   279	    Q = float(jnp.sum(out.dq_v_dt * dp / constants.g, axis=1).mean()) * constants.L_v
   280	    C = float(jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1).mean()) * constants.L_v
   281	    rel = abs(H + Q + C) / (abs(H) + abs(Q) + abs(C) + 1e-10)
   282	    assert rel < 0.30, (
   283	        f"Tiedtke MSE residual {H+Q+C:.1f} W/m^2 ({rel*100:.1f}% of total) "
   284	        f"exceeds 30% — kernel formulation has regressed"
   285	    )

exec
/bin/zsh -lc "nl -ba tests/unit/test_bechtold.py | sed -n '1,290p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Unit tests for the Bechtold/IFS convection scheme.
     2	
     3	Tests pin:
     4	
     5	* tendency shape / dtype / finiteness;
     6	* the PBL-CAPE closure (use_pbl_cape toggle changes diagnosed CAPE);
     7	* CMT signs and ``enable_cmt`` opt-out;
     8	* downdraft toggle effect;
     9	* **stochastic perturbation** — when enabled with a given PRNG key
    10	  the AR1 noise state evolves; when disabled the result is
    11	  deterministic and identical across calls;
    12	* the AR1 decorrelation: variance of the AR1 process matches the
    13	  expected stationary variance ``1`` for a sufficiently long run;
    14	* PhysicsState ``conv_stoch_state`` field is round-tripped
    15	  correctly;
    16	* finite gradients through ``epsilon_deep``, ``cape_pbl_depth``,
    17	  ``stochastic_amplitude``, ``cmt_c_u``;
    18	* selection through ``make_physics(PhysicsConfig(...))``.
    19	"""
    20	
    21	from __future__ import annotations
    22	
    23	from legoesm import constants
    24	
    25	import jax
    26	import jax.numpy as jnp
    27	import pytest
    28	
    29	from legoesm.core.field import Field
    30	from legoesm.grids.cubed_sphere import create_cubed_sphere
    31	from legoesm.grids.vertical import create_sigma_coordinate
    32	from legoesm.atmosphere.held_suarez import held_suarez_init
    33	from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    34	from legoesm.atmosphere.physics.physics_state import (
    35	    PhysicsState, init_physics_state,
    36	)
    37	from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    38	from legoesm.atmosphere.physics.convection.config import (
    39	    ConvectionConfig,
    40	    BechtoldConfig,
    41	)
    42	from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    43	from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    44	from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    45	from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
    46	
    47	
    48	def _column(
    49	    ncol=2, nlev=16, T_sfc=302.0, q_sfc=16e-3, lapse_rate=7.5,
    50	    p_s=1.0e5, p_top=5.0e3,
    51	):
    52	    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    53	    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    54	    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    55	    p_half = jnp.concatenate(
    56	        [
    57	            jnp.full((ncol, 1), p_top * 0.5),
    58	            p_half_inner,
    59	            jnp.full((ncol, 1), p_s),
    60	        ],
    61	        axis=1,
    62	    )
    63	    z_full = -8500.0 * jnp.log(p_full / p_s)
    64	    T = jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z_full
    65	    q = q_sfc * jnp.exp(-z_full / 3000.0)
    66	    u = jnp.linspace(0, 25, nlev)[None, :]
    67	    u = jnp.broadcast_to(u, (ncol, nlev))
    68	    v = jnp.zeros_like(u)
    69	    return T, q, p_full, p_half, u, v
    70	
    71	
    72	# ---------------------------------------------------------------------------
    73	# Shape / finiteness
    74	# ---------------------------------------------------------------------------
    75	
    76	def test_bechtold_shape_finiteness():
    77	    T, q, pf, ph, u, v = _column(ncol=3, nlev=12)
    78	    ncol, nlev = T.shape
    79	    cpp = jnp.zeros((ncol, nlev))
    80	    stoch = jnp.zeros((ncol,))
    81	    out, M_u_new, stoch_new = bechtold_convection(
    82	        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
    83	    )
    84	    assert M_u_new.shape == (ncol, nlev)
    85	    assert stoch_new.shape == (ncol,)
    86	    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
    87	                out.convective_mask, out.du_dt_conv, out.dv_dt_conv,
    88	                M_u_new, stoch_new):
    89	        assert jnp.all(jnp.isfinite(arr))
    90	
    91	
    92	# ---------------------------------------------------------------------------
    93	# PBL-CAPE closure: switching to surface-parcel CAPE changes M_b
    94	# ---------------------------------------------------------------------------
    95	
    96	def test_bechtold_pbl_cape_changes_m_b():
    97	    """Toggling between ``use_pbl_cape=True`` and ``use_pbl_cape=False``
    98	    changes the diagnosed cloud-base mass flux."""
    99	    T, q, pf, ph, u, v = _column()
   100	    ncol, nlev = T.shape
   101	    cpp = jnp.zeros((ncol, nlev))
   102	    stoch = jnp.zeros((ncol,))
   103	    _, M_u_pbl, _ = bechtold_convection(
   104	        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
   105	        config=BechtoldConfig(use_pbl_cape=True),
   106	    )
   107	    _, M_u_sfc, _ = bechtold_convection(
   108	        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
   109	        config=BechtoldConfig(use_pbl_cape=False),
   110	    )
   111	    # The two diagnoses differ by some non-trivial amount.
   112	    assert float(jnp.sum(jnp.abs(M_u_pbl - M_u_sfc))) > 1e-12
   113	
   114	
   115	# ---------------------------------------------------------------------------
   116	# CMT
   117	# ---------------------------------------------------------------------------
   118	
   119	def test_bechtold_cmt_present():
   120	    T, q, pf, ph, u, v = _column()
   121	    ncol, nlev = T.shape
   122	    cpp = jnp.zeros((ncol, nlev))
   123	    stoch = jnp.zeros((ncol,))
   124	    out, _, _ = bechtold_convection(T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0)
   125	    assert out.du_dt_conv is not None
   126	    assert out.dv_dt_conv is not None
   127	
   128	
   129	def test_bechtold_cmt_disabled():
   130	    T, q, pf, ph, u, v = _column()
   131	    ncol, nlev = T.shape
   132	    cpp = jnp.zeros((ncol, nlev))
   133	    stoch = jnp.zeros((ncol,))
   134	    out, _, _ = bechtold_convection(
   135	        T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
   136	        config=BechtoldConfig(enable_cmt=False),
   137	    )
   138	    assert out.du_dt_conv is None
   139	    assert out.dv_dt_conv is None
   140	
   141	
   142	# ---------------------------------------------------------------------------
   143	# Stochastic perturbation
   144	# ---------------------------------------------------------------------------
   145	
   146	def test_bechtold_deterministic_when_stochastic_off():
   147	    """With ``enable_stochastic=False``, two calls with the same input
   148	    produce identical output."""
   149	    T, q, pf, ph, u, v = _column()
   150	    ncol, nlev = T.shape
   151	    cpp = jnp.zeros((ncol, nlev))
   152	    stoch = jnp.zeros((ncol,))
   153	    out1, M1, s1 = bechtold_convection(T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0)
   154	    out2, M2, s2 = bechtold_convection(T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0)
   155	    assert jnp.allclose(out1.dT_dt, out2.dT_dt)
   156	    assert jnp.allclose(M1, M2)
   157	    assert jnp.allclose(s1, s2)
   158	
   159	
   160	def test_bechtold_stochastic_changes_with_key():
   161	    """With ``enable_stochastic=True``, two different PRNG keys produce
   162	    different AR1 noise states and different diagnosed mass fluxes.
   163	
   164	    The fixture uses a high-CAPE sounding that drives diagnosed M_b
   165	    above the production ``M_b_max=0.05`` cap on both keys; we set
   166	    ``M_b_max=10.0`` here so the cap does not bind and mask the
   167	    stochastic variation.  In production the cap is intentional — it
   168	    bounds single-step shocks from outlier columns — and a no-cap
   169	    setup like this should never appear in a real run.
   170	    """
   171	    T, q, pf, ph, u, v = _column()
   172	    ncol, nlev = T.shape
   173	    cpp = jnp.zeros((ncol, nlev))
   174	    stoch = jnp.zeros((ncol,))
   175	    config = BechtoldConfig(
   176	        enable_stochastic=True, stochastic_amplitude=0.5, M_b_max=10.0,
   177	    )
   178	    key1 = jax.random.PRNGKey(0)
   179	    key2 = jax.random.PRNGKey(7)
   180	    _, M1, s1 = bechtold_convection(T, q, pf, ph, u, v, cpp, stoch, key1, dt=300.0, config=config)
   181	    _, M2, s2 = bechtold_convection(T, q, pf, ph, u, v, cpp, stoch, key2, dt=300.0, config=config)
   182	    assert not jnp.allclose(s1, s2)
   183	    assert not jnp.allclose(M1, M2)
   184	
   185	
   186	def test_bechtold_AR1_stationary_variance():
   187	    """Long-run AR1 noise has stationary variance ≈ 1 (per unit
   188	    amplitude).  We integrate 500 steps and check that the empirical
   189	    variance lands near 1."""
   190	    T, q, pf, ph, u, v = _column(ncol=200)
   191	    ncol, nlev = T.shape
   192	    cpp = jnp.zeros((ncol, nlev))
   193	    stoch = jnp.zeros((ncol,))
   194	    config = BechtoldConfig(
   195	        enable_stochastic=True, stochastic_amplitude=1.0,
   196	        stochastic_decorrelation=1800.0,
   197	    )
   198	    key = jax.random.PRNGKey(0)
   199	    # 500 steps; sample stoch_new at the end.
   200	    for i in range(500):
   201	        key, subkey = jax.random.split(key)
   202	        _, _, stoch = bechtold_convection(
   203	            T, q, pf, ph, u, v, cpp, stoch, subkey, dt=300.0, config=config,
   204	        )
   205	    var = float(jnp.var(stoch))
   206	    # Stationary variance is theoretically 1; allow generous tolerance.
   207	    assert 0.5 < var < 2.0, f"AR1 stationary variance off-target: {var}"
   208	
   209	
   210	# ---------------------------------------------------------------------------
   211	# Differentiability
   212	# ---------------------------------------------------------------------------
   213	
   214	def test_bechtold_grad_through_epsilon_deep():
   215	    T, q, pf, ph, u, v = _column()
   216	    ncol, nlev = T.shape
   217	    cpp = jnp.zeros((ncol, nlev))
   218	    stoch = jnp.zeros((ncol,))
   219	
   220	    def f(eps):
   221	        out, _, _ = bechtold_convection(
   222	            T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
   223	            config=BechtoldConfig(epsilon_deep=eps),
   224	        )
   225	        return jnp.sum(out.dT_dt)
   226	
   227	    g = float(jax.grad(f)(jnp.asarray(1.75e-3)))
   228	    assert bool(jnp.isfinite(g))
   229	
   230	
   231	def test_bechtold_grad_through_cape_pbl_depth():
   232	    T, q, pf, ph, u, v = _column()
   233	    ncol, nlev = T.shape
   234	    cpp = jnp.zeros((ncol, nlev))
   235	    stoch = jnp.zeros((ncol,))
   236	
   237	    def f(depth):
   238	        out, _, _ = bechtold_convection(
   239	            T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
   240	            config=BechtoldConfig(cape_pbl_depth=depth),
   241	        )
   242	        return jnp.sum(out.dT_dt)
   243	
   244	    g = float(jax.grad(f)(jnp.asarray(500.0)))
   245	    assert bool(jnp.isfinite(g))
   246	
   247	
   248	def test_bechtold_grad_through_stochastic_amplitude_when_off():
   249	    """Even when ``enable_stochastic=False``, gradient through
   250	    ``stochastic_amplitude`` is finite (it's a static config field
   251	    that doesn't enter the computation in the off branch)."""
   252	    T, q, pf, ph, u, v = _column()
   253	    ncol, nlev = T.shape
   254	    cpp = jnp.zeros((ncol, nlev))
   255	    stoch = jnp.zeros((ncol,))
   256	
   257	    def f(amp):
   258	        out, _, _ = bechtold_convection(
   259	            T, q, pf, ph, u, v, cpp, stoch, None, dt=300.0,
   260	            config=BechtoldConfig(enable_stochastic=False, stochastic_amplitude=amp),
   261	        )
   262	        return jnp.sum(out.dT_dt)
   263	
   264	    g = float(jax.grad(f)(jnp.asarray(0.5)))
   265	    assert bool(jnp.isfinite(g))
   266	    # In the off branch the gradient is 0 (parameter unused) — that's
   267	    # fine, just must be finite.
   268	
   269	
   270	# ---------------------------------------------------------------------------
   271	# PhysicsState round-trip
   272	# ---------------------------------------------------------------------------
   273	
   274	def test_bechtold_physics_state_has_conv_stoch_state():
   275	    """``init_physics_state`` produces a PhysicsState with
   276	    ``conv_stoch_state`` of shape ``(ncol,)``."""
   277	    cfg = PhysicsConfig(
   278	        radiation=RadiationConfig(scheme="none"),
   279	        convection=ConvectionConfig(scheme="bechtold"),
   280	        turbulence=TurbulenceConfig(scheme="none"),
   281	        microphysics=MicrophysicsConfig(scheme="none"),
   282	        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
   283	    )
   284	    ps = init_physics_state(64, 12, cfg)
   285	    assert hasattr(ps, "conv_stoch_state"), \
   286	        "PhysicsState should expose conv_stoch_state field"
   287	    assert ps.conv_stoch_state.shape == (64,)
   288	    assert jnp.all(ps.conv_stoch_state == 0.0)
   289	
   290	

exec
/bin/zsh -lc "nl -ba tests/unit/test_emanuel.py | sed -n '1,320p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Unit tests for the Emanuel (1991) convection scheme.
     2	
     3	Tests pin:
     4	
     5	* tendency shape / dtype / finiteness;
     6	* the no-CMT invariant;
     7	* the buoyancy-sort detrainment enhancement (the distinguishing
     8	  feature relative to ZM and KF);
     9	* the unsaturated-downdraft toggle (column conservation under both
    10	  modes);
    11	* finite gradients through ``cape_threshold`` and
    12	  ``smooth_trigger_sharpness`` — the AD-safety property.
    13	"""
    14	
    15	from __future__ import annotations
    16	
    17	from legoesm import constants
    18	
    19	import jax
    20	import jax.numpy as jnp
    21	import numpy as np
    22	import pytest
    23	
    24	from legoesm.core.field import Field
    25	from legoesm.grids.cubed_sphere import create_cubed_sphere
    26	from legoesm.grids.vertical import create_sigma_coordinate
    27	from legoesm.atmosphere.held_suarez import held_suarez_init
    28	from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    29	from legoesm.atmosphere.physics.physics_state import init_physics_state
    30	from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    31	from legoesm.atmosphere.physics.convection.config import (
    32	    ConvectionConfig,
    33	    EmanuelConfig,
    34	)
    35	from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    36	from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    37	from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    38	from legoesm.atmosphere.physics.convection.emanuel import emanuel_convection
    39	
    40	
    41	def _column(
    42	    ncol=2, nlev=16, T_sfc=302.0, q_sfc=16e-3, lapse_rate=7.5,
    43	    p_s=1.0e5, p_top=5.0e3,
    44	):
    45	    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    46	    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    47	    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    48	    p_half = jnp.concatenate(
    49	        [
    50	            jnp.full((ncol, 1), p_top * 0.5),
    51	            p_half_inner,
    52	            jnp.full((ncol, 1), p_s),
    53	        ],
    54	        axis=1,
    55	    )
    56	    z_full = -8500.0 * jnp.log(p_full / p_s)
    57	    T = jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z_full
    58	    q = q_sfc * jnp.exp(-z_full / 3000.0)
    59	    return T, q, p_full, p_half
    60	
    61	
    62	# ---------------------------------------------------------------------------
    63	# Shape / finiteness / no CMT
    64	# ---------------------------------------------------------------------------
    65	
    66	def test_emanuel_shape_dtype():
    67	    T, q, pf, ph = _column(ncol=3, nlev=12)
    68	    ncol, nlev = T.shape
    69	    cpp = jnp.zeros((ncol, nlev))
    70	    out, cpp_new = emanuel_convection(T, q, pf, ph, cpp, dt=300.0)
    71	    assert out.dT_dt.shape == (ncol, nlev)
    72	    assert out.dq_v_dt.shape == (ncol, nlev)
    73	    assert out.dq_c_conv_dt.shape == (ncol, nlev)
    74	    assert out.cape.shape == (ncol,)
    75	    assert cpp_new.shape == (ncol, nlev)
    76	
    77	
    78	def test_emanuel_outputs_finite():
    79	    T, q, pf, ph = _column()
    80	    ncol, nlev = T.shape
    81	    cpp = jnp.zeros((ncol, nlev))
    82	    out, cpp_new = emanuel_convection(T, q, pf, ph, cpp, dt=300.0)
    83	    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
    84	                out.convective_mask, cpp_new):
    85	        assert jnp.all(jnp.isfinite(arr))
    86	
    87	
    88	def test_emanuel_no_cmt():
    89	    T, q, pf, ph = _column()
    90	    ncol, nlev = T.shape
    91	    cpp = jnp.zeros((ncol, nlev))
    92	    out, _ = emanuel_convection(T, q, pf, ph, cpp, dt=300.0)
    93	    assert out.du_dt_conv is None
    94	    assert out.dv_dt_conv is None
    95	
    96	
    97	def test_emanuel_cloud_water_source_non_negative():
    98	    """The convective cloud-water source is non-negative even after
    99	    the unsaturated-downdraft evaporation step subtracts column
   100	    condensate (we ``maximum(., 0)`` the result)."""
   101	    T, q, pf, ph = _column()
   102	    ncol, nlev = T.shape
   103	    cpp = jnp.zeros((ncol, nlev))
   104	    out, _ = emanuel_convection(T, q, pf, ph, cpp, dt=300.0)
   105	    assert jnp.all(out.dq_c_conv_dt >= 0.0)
   106	
   107	
   108	# ---------------------------------------------------------------------------
   109	# Buoyancy-sort: n_mixing_fractions affects detrainment but not gross sign
   110	# ---------------------------------------------------------------------------
   111	
   112	def test_emanuel_n_fractions_finite_for_all_choices():
   113	    """Tendencies are finite for ``n_mixing_fractions`` ∈ {1, 4, 8, 16}.
   114	    n_fractions=1 is the limit of a single bulk plume; larger values
   115	    approach Emanuel 1991's 50-bin spectrum more closely."""
   116	    T, q, pf, ph = _column()
   117	    ncol, nlev = T.shape
   118	    cpp = jnp.zeros((ncol, nlev))
   119	    for n in (1, 4, 8, 16):
   120	        config = EmanuelConfig(n_mixing_fractions=n)
   121	        out, _ = emanuel_convection(T, q, pf, ph, cpp, dt=300.0, config=config)
   122	        assert jnp.all(jnp.isfinite(out.dT_dt)), (
   123	            f"NaN with n_mixing_fractions={n}"
   124	        )
   125	
   126	
   127	def test_emanuel_buoyancy_sort_detrainment_increases_tendency_magnitude():
   128	    """A larger ``cu_coefficient`` (buoyancy-sort detrainment
   129	    enhancement) increases the magnitude of the per-level tendencies
   130	    relative to ``cu = 0`` (single-plume limit)."""
   131	    T, q, pf, ph = _column()
   132	    ncol, nlev = T.shape
   133	    cpp = jnp.zeros((ncol, nlev))
   134	    out_no_sort, _ = emanuel_convection(
   135	        T, q, pf, ph, cpp, dt=300.0,
   136	        config=EmanuelConfig(cu_coefficient=0.0),
   137	    )
   138	    out_strong_sort, _ = emanuel_convection(
   139	        T, q, pf, ph, cpp, dt=300.0,
   140	        config=EmanuelConfig(cu_coefficient=1.0),
   141	    )
   142	    # Sort-enhanced should have at least the magnitude of no-sort.
   143	    mag_no = float(jnp.sum(jnp.abs(out_no_sort.dT_dt)))
   144	    mag_yes = float(jnp.sum(jnp.abs(out_strong_sort.dT_dt)))
   145	    assert mag_yes >= mag_no - 1e-12
   146	
   147	
   148	# ---------------------------------------------------------------------------
   149	# Unsaturated downdraft toggle
   150	# ---------------------------------------------------------------------------
   151	
   152	def test_emanuel_downdraft_toggle_changes_subcloud_dT():
   153	    """Enabling the unsaturated downdraft introduces an additional
   154	    sub-cloud cooling term — the difference in ``dT_dt`` between the
   155	    on/off configurations is non-zero in the surface-adjacent
   156	    layers."""
   157	    T, q, pf, ph = _column()
   158	    ncol, nlev = T.shape
   159	    cpp = jnp.zeros((ncol, nlev))
   160	    out_off, _ = emanuel_convection(
   161	        T, q, pf, ph, cpp, dt=300.0,
   162	        config=EmanuelConfig(enable_unsaturated_downdraft=False),
   163	    )
   164	    out_on, _ = emanuel_convection(
   165	        T, q, pf, ph, cpp, dt=300.0,
   166	        config=EmanuelConfig(enable_unsaturated_downdraft=True),
   167	    )
   168	    # Sub-cloud (last 4 levels) tendencies differ between the two
   169	    # branches — the downdraft is doing something visible.
   170	    diff = jnp.abs(out_on.dT_dt[:, -4:] - out_off.dT_dt[:, -4:])
   171	    assert float(jnp.max(diff)) > 1e-8
   172	
   173	
   174	# ---------------------------------------------------------------------------
   175	# Differentiability through tunable parameters
   176	# ---------------------------------------------------------------------------
   177	
   178	def test_emanuel_grad_through_cape_threshold():
   179	    T, q, pf, ph = _column()
   180	    ncol, nlev = T.shape
   181	    cpp = jnp.zeros((ncol, nlev))
   182	
   183	    def f(threshold):
   184	        config = EmanuelConfig(cape_threshold=threshold)
   185	        out, _ = emanuel_convection(T, q, pf, ph, cpp, dt=300.0, config=config)
   186	        return jnp.sum(out.dT_dt)
   187	
   188	    g = float(jax.grad(f)(jnp.asarray(70.0)))
   189	    assert np.isfinite(g)
   190	
   191	
   192	def test_emanuel_grad_through_smooth_trigger_sharpness():
   193	    T, q, pf, ph = _column()
   194	    ncol, nlev = T.shape
   195	    cpp = jnp.zeros((ncol, nlev))
   196	
   197	    def f(s):
   198	        config = EmanuelConfig(smooth_trigger_sharpness=s)
   199	        out, _ = emanuel_convection(T, q, pf, ph, cpp, dt=300.0, config=config)
   200	        return jnp.sum(out.dT_dt)
   201	
   202	    g = float(jax.grad(f)(jnp.asarray(0.5)))
   203	    assert np.isfinite(g)
   204	
   205	
   206	# ---------------------------------------------------------------------------
   207	# Carry layout
   208	# ---------------------------------------------------------------------------
   209	
   210	def test_emanuel_carry_layout():
   211	    T, q, pf, ph = _column()
   212	    ncol, nlev = T.shape
   213	    cpp = jnp.zeros((ncol, nlev))
   214	    _, cpp_new = emanuel_convection(T, q, pf, ph, cpp, dt=300.0)
   215	    assert jnp.all(cpp_new[:, :-1] == 0.0)
   216	    assert jnp.all(cpp_new[:, -1] >= 0.0)
   217	
   218	
   219	# ---------------------------------------------------------------------------
   220	# Orchestrator
   221	# ---------------------------------------------------------------------------
   222	
   223	def test_emanuel_orchestrator_one_step_finite():
   224	    grid = create_cubed_sphere(4)
   225	    sigma = create_sigma_coordinate(12)
   226	    state = held_suarez_init(grid, sigma)
   227	    tracers = {
   228	        "q_v": Field(0.014 * jnp.ones((6, 4, 4, 12)), name="q_v",
   229	                     dims=("face", "x", "y", "level"), units="kg/kg"),
   230	        "q_c": Field(jnp.zeros((6, 4, 4, 12)), name="q_c",
   231	                     dims=("face", "x", "y", "level"), units="kg/kg"),
   232	    }
   233	    state = state._replace(tracers=tracers)
   234	    cfg = PhysicsConfig(
   235	        radiation=RadiationConfig(scheme="none"),
   236	        convection=ConvectionConfig(scheme="emanuel"),
   237	        turbulence=TurbulenceConfig(scheme="none"),
   238	        microphysics=MicrophysicsConfig(scheme="none"),
   239	        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
   240	    )
   241	    ncol = 6 * 4 * 4
   242	    ps = init_physics_state(ncol, 12, cfg)
   243	    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
   244	    tend, ps_out = physics_fn(state, grid, sigma, phys_state=ps)
   245	    assert ps_out.conv_prog_profile.shape == (ncol, 12)
   246	    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
   247	        assert jnp.all(jnp.isfinite(f.data))
   248	    # Emanuel has no CMT — orchestrator zeros these.
   249	    assert float(jnp.max(jnp.abs(tend.du_dt.data))) == 0.0
   250	    assert float(jnp.max(jnp.abs(tend.dv_dt.data))) == 0.0
   251	
   252	
   253	# ---------------------------------------------------------------------------
   254	# MSE conservation regression guard (currently expected to fail)
   255	# ---------------------------------------------------------------------------
   256	
   257	@pytest.mark.xfail(
   258	    reason=(
   259	        "Standard mass-flux kernel does not conserve column MSE on a "
   260	        "closed (no-surface-flux) probe.  Currently ~99% non-conservation "
   261	        "residual; flagged xfail so any future kernel improvement that "
   262	        "closes this is detected."
   263	    ),
   264	    strict=True,
   265	)
   266	def test_emanuel_mse_conservation_within_tolerance():
   267	    """Column-integrated ``c_p ∫dT + L_v ∫(dq_v + dq_c_conv) dp/g`` should
   268	    be small relative to the heating magnitude on a CAPE-positive sounding.
   269	    """
   270	    T, q, pf, ph = _column()
   271	    ncol, nlev = T.shape
   272	    cpp = jnp.zeros((ncol, nlev))
   273	    out, _ = emanuel_convection(
   274	        T=T, q_v=q, p_full=pf, p_half=ph,
   275	        conv_prog_profile=cpp, dt=1800.0,
   276	    )
   277	    dp = ph[:, 1:] - ph[:, :-1]
   278	    H = float(jnp.sum(out.dT_dt * dp / constants.g, axis=1).mean()) * constants.c_pd
   279	    Q = float(jnp.sum(out.dq_v_dt * dp / constants.g, axis=1).mean()) * constants.L_v
   280	    C = float(jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1).mean()) * constants.L_v
   281	    rel = abs(H + Q + C) / (abs(H) + abs(Q) + abs(C) + 1e-10)
   282	    assert rel < 0.30, (
   283	        f"Emanuel MSE residual {H+Q+C:.1f} W/m^2 ({rel*100:.1f}% of total)"
   284	    )

exec
/bin/zsh -lc "nl -ba tests/unit/test_bechtold.py | sed -n '510,570p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   510	    )
   511	    # Tendencies are finite.
   512	    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
   513	        assert jnp.all(jnp.isfinite(f.data))
   514	
   515	
   516	# ---------------------------------------------------------------------------
   517	# MSE conservation regression guard (currently expected to fail)
   518	# ---------------------------------------------------------------------------
   519	
   520	@pytest.mark.xfail(
   521	    reason=(
   522	        "Bechtold inherits the standard Tiedtke kernel for env tendencies, "
   523	        "but its PBL-CAPE closure pushes M_b larger than Tiedtke's, so "
   524	        "subsidence ``g/c_p`` heating overwhelms the kernel's vapor sink. "
   525	        "Currently ~92% non-conservation residual; flagged as xfail so "
   526	        "any future kernel improvement that closes this is detected."
   527	    ),
   528	    strict=True,
   529	)
   530	def test_bechtold_mse_conservation_within_tolerance():
   531	    T, q, pf, ph, u, v = _column()
   532	    ncol, nlev = T.shape
   533	    cpp = jnp.zeros((ncol, nlev))
   534	    stoch = jnp.zeros((ncol,))
   535	    out, _, _ = bechtold_convection(
   536	        T=T, q_v=q, p_full=pf, p_half=ph, u=u, v=v,
   537	        conv_prog_profile=cpp, conv_stoch_state=stoch, prng_key=None,
   538	        dt=1800.0,
   539	        config=BechtoldConfig(enable_stochastic=False, enable_cmt=False),
   540	        moisture_convergence=jnp.zeros_like(T),
   541	    )
   542	    dp = ph[:, 1:] - ph[:, :-1]
   543	    H = float(jnp.sum(out.dT_dt * dp / constants.g, axis=1).mean()) * constants.c_pd
   544	    Q = float(jnp.sum(out.dq_v_dt * dp / constants.g, axis=1).mean()) * constants.L_v
   545	    C = float(jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1).mean()) * constants.L_v
   546	    rel = abs(H + Q + C) / (abs(H) + abs(Q) + abs(C) + 1e-10)
   547	    assert rel < 0.30, (
   548	        f"Bechtold MSE residual {H+Q+C:.1f} W/m^2 ({rel*100:.1f}% of total)"
   549	    )

exec
/bin/zsh -lc "python - <<'PY'
import jax.numpy as jnp
from legoesm.atmosphere.physics.convection._plume import compute_lfc_lnb, compute_cin
from legoesm.atmosphere.physics.convection._triggers import smooth_lowest_crossing_index

# Simple surface-last buoyancy: top index 0, surface index 5.
# Surface negative, mid positive, top negative => LFC near 4.5, LNB near 1.5 ideally.
b = jnp.array([[-1., -1., 1., 1., -1., -1.]])
T_env = jnp.zeros_like(b)
T_parcel = b
print('k_lfc, k_lnb', compute_lfc_lnb(T_env, T_parcel, sharpness=10.0))

# Weak but true crossing: surface first after internal reverse has pair -0.1 to +0.1.
profile = jnp.array([[1.0, 0.1, -0.1, -1.0]]) # surface-last; from surface upward: -1,-0.1,0.1,1
for s in [1.0, 5.0, 10.0]:
    print('weak crossing s', s, smooth_lowest_crossing_index(profile, 0.0, s))
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 127 in 0ms:
zsh:1: command not found: python

exec
/bin/zsh -lc "python3 - <<'PY'
import jax.numpy as jnp
from legoesm.atmosphere.physics.convection._plume import compute_lfc_lnb
from legoesm.atmosphere.physics.convection._triggers import smooth_lowest_crossing_index

b = jnp.array([[-1., -1., 1., 1., -1., -1.]])
T_env = jnp.zeros_like(b)
T_parcel = b
print('k_lfc, k_lnb', compute_lfc_lnb(T_env, T_parcel, sharpness=10.0))

profile = jnp.array([[1.0, 0.1, -0.1, -1.0]])
for s in [1.0, 5.0, 10.0]:
    print('weak crossing s', s, smooth_lowest_crossing_index(profile, 0.0, s))
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 551ms:
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
ModuleNotFoundError: No module named 'legoesm'

exec
/bin/zsh -lc "PYTHONPATH=src python3 - <<'PY'
import jax.numpy as jnp
from legoesm.atmosphere.physics.convection._plume import compute_lfc_lnb
from legoesm.atmosphere.physics.convection._triggers import smooth_lowest_crossing_index

b = jnp.array([[-1., -1., 1., 1., -1., -1.]])
T_env = jnp.zeros_like(b)
T_parcel = b
print('k_lfc, k_lnb', compute_lfc_lnb(T_env, T_parcel, sharpness=10.0))

profile = jnp.array([[1.0, 0.1, -0.1, -1.0]])
for s in [1.0, 5.0, 10.0]:
    print('weak crossing s', s, smooth_lowest_crossing_index(profile, 0.0, s))
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 692ms:
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

exec
/bin/zsh -lc "python3 - <<'PY'
import jax, jax.numpy as jnp

def smooth_lowest_crossing_index(profile, threshold, sharpness):
    *_, nlev = profile.shape
    profile_rev=profile[..., ::-1]
    pb=profile_rev[..., :-1]; pa=profile_rev[..., 1:]
    cw=jax.nn.sigmoid(sharpness*(threshold-pb))*jax.nn.sigmoid(sharpness*(pa-threshold))
    d=pa-pb; safe=jnp.where(jnp.abs(d)>1e-30,d,1e-30)
    frac=jnp.clip((threshold-pb)/safe,0,1)
    k=jnp.arange(nlev-1,dtype=profile.dtype)
    idx=k+frac
    notyet=jnp.concatenate([jnp.ones(profile.shape[:-1]+(1,),dtype=profile.dtype), jnp.cumprod(1-cw,axis=-1)[..., :-1]],axis=-1)
    fc=cw*notyet
    total=jnp.sum(fc,axis=-1)
    naive=jnp.sum(fc*idx,axis=-1)/jnp.maximum(total,1e-12)
    blend=jax.nn.sigmoid(20*(total-0.5))
    return float(nlev-1)-blend*naive, total, cw

def compute_lfc_lnb_buoy(buoy, sharpness):
    nlev=buoy.shape[-1]
    k_lfc=smooth_lowest_crossing_index(buoy,0.0,sharpness)[0]
    levels=jnp.broadcast_to(jnp.arange(nlev,dtype=buoy.dtype),buoy.shape)
    above=jax.nn.sigmoid(sharpness*(((nlev-1.0)-k_lfc)[:,None]-levels))
    masked=buoy*above
    k_lnb=smooth_lowest_crossing_index(-masked,0.0,sharpness)[0]
    return k_lfc,k_lnb,above,masked

b=jnp.array([[-1., -1., 1., 1., -1., -1.]])
print(compute_lfc_lnb_buoy(b,10.0)[:2])
print('above mask', compute_lfc_lnb_buoy(b,10.0)[2])
profile=jnp.array([[1.0,0.1,-0.1,-1.0]])
for s in [1.,5.,10.]:
    print(s, smooth_lowest_crossing_index(profile,0.,s))
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 1269ms:
(Array([3.500091], dtype=float32), Array([3.590458], dtype=float32))
above mask [[9.9999964e-01 9.9330115e-01 6.6867992e-03 3.0562376e-07 1.3875287e-11
  6.2993713e-16]]
1.0 (Array([1.6615322], dtype=float32), Array([0.69136417], dtype=float32), Array([[0.34726804, 0.27560315, 0.34726804]], dtype=float32))
5.0 (Array([1.6594079], dtype=float32), Array([0.7607355], dtype=float32), Array([[0.37501386, 0.38745564, 0.37501386]], dtype=float32))
10.0 (Array([1.6271151], dtype=float32), Array([0.75117826], dtype=float32), Array([[0.2689292 , 0.53444666, 0.2689292 ]], dtype=float32))

exec
/bin/zsh -lc "python3 - <<'PY'
import jax, jax.numpy as jnp

def f(profile,s):
 nlev=profile.shape[-1]; pr=profile[...,::-1]; pb=pr[...,:-1]; pa=pr[...,1:]; cw=jax.nn.sigmoid(s*(0-pb))*jax.nn.sigmoid(s*(pa-0)); d=pa-pb; frac=jnp.clip((0-pb)/jnp.where(jnp.abs(d)>1e-30,d,1e-30),0,1); idx=jnp.arange(nlev-1,dtype=profile.dtype)+frac; total=jnp.sum(cw,axis=-1); naive=jnp.sum(cw*idx,axis=-1)/jnp.maximum(total,1e-12); blend=jax.nn.sigmoid(20*(total-0.5)); return float(nlev-1)-blend*naive,total,cw,blend,naive
for s in [1.,5.,10.]: print(s, f(jnp.array([[0.1,-0.1]]),s))
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 673ms:
1.0 (Array([0.9944406], dtype=float32), Array([0.27560315], dtype=float32), Array([[0.27560315]], dtype=float32), Array([0.0111188], dtype=float32), Array([0.5], dtype=float32))
5.0 (Array([0.9523635], dtype=float32), Array([0.38745564], dtype=float32), Array([[0.38745564]], dtype=float32), Array([0.09527297], dtype=float32), Array([0.5], dtype=float32))
10.0 (Array([0.66713524], dtype=float32), Array([0.53444666], dtype=float32), Array([[0.53444666]], dtype=float32), Array([0.6657296], dtype=float32), Array([0.5], dtype=float32))

codex
**Findings**
- `src/legoesm/atmosphere/physics/convection/_plume.py:244-251` - LNB masking mixes surface-last and surface-first indices.  
  `levels` is surface-last, but the threshold is converted to surface-first, so LNB is often diagnosed near the LFC or surface instead of above the LFC. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/_plume.py:311-317` - CIN window bounds are inverted.  
  The code selects levels above the LFC and below the LCL, which is not the layer between LCL and LFC in surface-last indexing; capped-inversion CIN can be driven toward zero. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/_triggers.py:259-321` - Weak real crossings can be treated as “no crossing.”  
  The no-crossing gate is based on total soft-crossing weight with midpoint `0.5`; a genuine but gradual crossing can have total weight below that and blend back to the surface fallback. This confirms hypothesis 6 as a substantive bug/risk. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/_plume.py:495-496` - Plume mass flux uses explicit Euler plus a hard nonnegative clip.  
  For coarse layers or strong detrainment, `1 + (eps - dlt) * dz` can go negative, after which `jnp.maximum` kills both mass flux and gradients. Hypothesis 4 is not a bug as stated, but this raw-carry update is not robust. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/_plume.py:499-527` - Plume cloud water is accumulated but not diluted by entrainment.  
  Environmental entrainment dilutes `T_u` and `q_u`, but existing `q_c_u_prev` is carried forward unchanged; plume total water is therefore too large aloft. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/mass_flux.py:107-112`, `319-331`, `437-451` - Mass-flux and EDMF plumes create condensate from saturation humidity, not available vapor.  
  `q_c_u = q_sat_base - q_sat_moist` can produce cloud water even when the actual surface `q_v` is dry, and CAPE is diagnosed from a saturated surface parcel independent of `q_v`. Severity: `critical`.

- `src/legoesm/atmosphere/physics/convection/mass_flux.py:211-240` - Shared mass-flux kernel is not column MSE conservative.  
  Subsidence, detrainment heating, vapor tendency, and cloud-water source are added independently rather than in a telescoping moist-static-energy flux form. This matches the xfailed conservation guards in the tests and violates the stated conservation requirement. Severity: `critical`.

- `src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:131-148`, `kain_fritsch.py:166-176`, `emanuel.py:125-131`, `tiedtke.py:196-210`, `bechtold.py:188-223` - CAPE divided by a timescale is used as a mass flux.  
  `CAPE / tau` has units `m^2 s^-3`, not `kg m^-2 s^-1`; the subsequent `M_b_max` cap masks the unit error and creates a hard zero-gradient region. This is the real issue behind hypothesis 8’s cap-timing question. Severity: `critical`.

- `src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:142`, `tiedtke.py:220`, `bechtold.py:246` - The documented implicit-Euler factor is not what is implemented.  
  The code uses `dt / max(tau, dt)`, not `dt / tau`; for `dt > tau` it silently limits the adjustment ratio to 1. Severity: `minor`.

- `src/legoesm/atmosphere/physics/convection/mass_flux.py:279-280`, `433-435` - Prognostic mass-flux and EDMF state updates are explicit Euler.  
  These are not unconditionally stable for large `dt / tau`, unlike the implicit relaxation used elsewhere. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/tiedtke.py:243-260`, `bechtold.py:259-272` - Per-class `delta_0_eff` rescale incorrectly scales subsidence.  
  `_apply_mass_flux_kernel` includes both subsidence and detrainment, but the code multiplies the whole returned `dT_dt`/`dq_v_dt` by `delta_0_eff / delta_deep`; sign is preserved, but magnitude is wrong. This confirms hypothesis 9 as a substantive bug. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/tiedtke.py:185-193` - Saturation deficit is used as a moisture-convergence proxy with the wrong physical sign.  
  `max(q_sat - q_v, 0)` is largest in dry subsaturated columns, not moist/convergent columns, so the fallback deep closure preferentially fires in dry air. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/tiedtke.py:263-287`, `bechtold.py:274-294` - Downdraft evaporation cooling is dimensionally wrong and not water-conserving.  
  The hardcoded `0.05 kg/kg` is multiplied by `M/rho` without a depth or detrainment scale, so the tendency units are not `K/s`; it also cools without adding the evaporated water to `q_v`. Hypothesis 1 is more than style here. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/bechtold.py:127-133`, `152-153` - Bechtold’s PBL parcel is not actually mass weighted and uses surface pressure for a PBL-mean parcel.  
  The average omits `dp/g` weights, and `compute_lcl` is called with `p_base` even when `T_parcel/q_parcel` came from a PBL mean. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/mass_flux.py:82-89` - Column geometry does not make the surface height zero.  
  `z[:, -1]` equals the surface-layer thickness, not `0`, despite the docstring; this biases PBL-depth masks and adds a spurious first-layer plume step. Severity: `minor`.

- `src/legoesm/atmosphere/physics/convection/emanuel.py:180-189` - Emanuel buoyancy sorting scales the entire kernel, not just detrainment.  
  `sort_multiplier` is described as a detrainment enhancement, but it multiplies subsidence and cloud-water source as well. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/kuo.py:160-173` - Kuo moistening-budget distribution has an extra `g / dp`.  
  The column integral of the distributed moistening no longer equals `(1 - alpha_heat) * MC / tau_relax`. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/kuo.py:127-156`, `198-205` - Kuo heating-implied condensation and reported cloud-water source are inconsistent.  
  Vapor is reduced by unscaled `implied_condensation`, but `dq_c_conv_dt` is rescaled to a separate target column condensation budget, so latent energy and water bookkeeping diverge. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/sbm.py:89-93`, `dca.py:141-147`, `kuo.py:94-96`, `mass_flux.py:224-240` - Several “smooth” schemes still use hard masks/clips in active physics paths.  
  These are JAX-compatible but create kinks and dead-gradient regions at cloud membership, saturation, mass-flux caps, and condensation thresholds. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/_plume.py:147-171`, `sbm.py:101-121`, `kuo.py:91-96`, `mass_flux.py:328-331` - The code documents `q_v` as specific humidity but compares it to `saturation_mixing_ratio`.  
  If model `q_v` is specific humidity, saturation should use `saturation_specific_humidity`; otherwise the docs and variable names are wrong. Severity: `minor`.

**Hypotheses Not Substantive Bugs**
- Hypothesis 2, `bechtold.py:199-202`: the `0.05 kg/m²/s` MC normalizer is dimensionally coherent and only a configurability/style issue.
- Hypothesis 3, `dca.py:257-265`: the rescale does preserve column water algebraically; the defect is the nonlocal, nonphysical vertical distribution, not column conservation.
- Hypothesis 5, `_plume.py:176-177`: the `T - 55 >= 1` floor only activates for `T < 56 K`, so it is not a normal “very cold parcel” atmospheric bug.
- Hypothesis 7, `mass_flux.py:132-159`: the smooth stratospheric gate being nonzero is an AD tradeoff, but the docstring overstates “vanishes/actually closed.”
- Hypothesis 10, `emanuel.py:160-170`: the bound is correct because the variance of values in `[0, 1]` is at most `1/4`; the bug is applying the multiplier to the whole kernel.
2026-05-01T11:47:53.630611Z ERROR codex_core::session: failed to record rollout items: thread 019de356-ae20-7f21-8138-42b7a610f27e not found
tokens used
157,571
**Findings**
- `src/legoesm/atmosphere/physics/convection/_plume.py:244-251` - LNB masking mixes surface-last and surface-first indices.  
  `levels` is surface-last, but the threshold is converted to surface-first, so LNB is often diagnosed near the LFC or surface instead of above the LFC. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/_plume.py:311-317` - CIN window bounds are inverted.  
  The code selects levels above the LFC and below the LCL, which is not the layer between LCL and LFC in surface-last indexing; capped-inversion CIN can be driven toward zero. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/_triggers.py:259-321` - Weak real crossings can be treated as “no crossing.”  
  The no-crossing gate is based on total soft-crossing weight with midpoint `0.5`; a genuine but gradual crossing can have total weight below that and blend back to the surface fallback. This confirms hypothesis 6 as a substantive bug/risk. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/_plume.py:495-496` - Plume mass flux uses explicit Euler plus a hard nonnegative clip.  
  For coarse layers or strong detrainment, `1 + (eps - dlt) * dz` can go negative, after which `jnp.maximum` kills both mass flux and gradients. Hypothesis 4 is not a bug as stated, but this raw-carry update is not robust. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/_plume.py:499-527` - Plume cloud water is accumulated but not diluted by entrainment.  
  Environmental entrainment dilutes `T_u` and `q_u`, but existing `q_c_u_prev` is carried forward unchanged; plume total water is therefore too large aloft. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/mass_flux.py:107-112`, `319-331`, `437-451` - Mass-flux and EDMF plumes create condensate from saturation humidity, not available vapor.  
  `q_c_u = q_sat_base - q_sat_moist` can produce cloud water even when the actual surface `q_v` is dry, and CAPE is diagnosed from a saturated surface parcel independent of `q_v`. Severity: `critical`.

- `src/legoesm/atmosphere/physics/convection/mass_flux.py:211-240` - Shared mass-flux kernel is not column MSE conservative.  
  Subsidence, detrainment heating, vapor tendency, and cloud-water source are added independently rather than in a telescoping moist-static-energy flux form. This matches the xfailed conservation guards in the tests and violates the stated conservation requirement. Severity: `critical`.

- `src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:131-148`, `kain_fritsch.py:166-176`, `emanuel.py:125-131`, `tiedtke.py:196-210`, `bechtold.py:188-223` - CAPE divided by a timescale is used as a mass flux.  
  `CAPE / tau` has units `m^2 s^-3`, not `kg m^-2 s^-1`; the subsequent `M_b_max` cap masks the unit error and creates a hard zero-gradient region. This is the real issue behind hypothesis 8’s cap-timing question. Severity: `critical`.

- `src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py:142`, `tiedtke.py:220`, `bechtold.py:246` - The documented implicit-Euler factor is not what is implemented.  
  The code uses `dt / max(tau, dt)`, not `dt / tau`; for `dt > tau` it silently limits the adjustment ratio to 1. Severity: `minor`.

- `src/legoesm/atmosphere/physics/convection/mass_flux.py:279-280`, `433-435` - Prognostic mass-flux and EDMF state updates are explicit Euler.  
  These are not unconditionally stable for large `dt / tau`, unlike the implicit relaxation used elsewhere. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/tiedtke.py:243-260`, `bechtold.py:259-272` - Per-class `delta_0_eff` rescale incorrectly scales subsidence.  
  `_apply_mass_flux_kernel` includes both subsidence and detrainment, but the code multiplies the whole returned `dT_dt`/`dq_v_dt` by `delta_0_eff / delta_deep`; sign is preserved, but magnitude is wrong. This confirms hypothesis 9 as a substantive bug. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/tiedtke.py:185-193` - Saturation deficit is used as a moisture-convergence proxy with the wrong physical sign.  
  `max(q_sat - q_v, 0)` is largest in dry subsaturated columns, not moist/convergent columns, so the fallback deep closure preferentially fires in dry air. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/tiedtke.py:263-287`, `bechtold.py:274-294` - Downdraft evaporation cooling is dimensionally wrong and not water-conserving.  
  The hardcoded `0.05 kg/kg` is multiplied by `M/rho` without a depth or detrainment scale, so the tendency units are not `K/s`; it also cools without adding the evaporated water to `q_v`. Hypothesis 1 is more than style here. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/bechtold.py:127-133`, `152-153` - Bechtold’s PBL parcel is not actually mass weighted and uses surface pressure for a PBL-mean parcel.  
  The average omits `dp/g` weights, and `compute_lcl` is called with `p_base` even when `T_parcel/q_parcel` came from a PBL mean. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/mass_flux.py:82-89` - Column geometry does not make the surface height zero.  
  `z[:, -1]` equals the surface-layer thickness, not `0`, despite the docstring; this biases PBL-depth masks and adds a spurious first-layer plume step. Severity: `minor`.

- `src/legoesm/atmosphere/physics/convection/emanuel.py:180-189` - Emanuel buoyancy sorting scales the entire kernel, not just detrainment.  
  `sort_multiplier` is described as a detrainment enhancement, but it multiplies subsidence and cloud-water source as well. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/kuo.py:160-173` - Kuo moistening-budget distribution has an extra `g / dp`.  
  The column integral of the distributed moistening no longer equals `(1 - alpha_heat) * MC / tau_relax`. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/kuo.py:127-156`, `198-205` - Kuo heating-implied condensation and reported cloud-water source are inconsistent.  
  Vapor is reduced by unscaled `implied_condensation`, but `dq_c_conv_dt` is rescaled to a separate target column condensation budget, so latent energy and water bookkeeping diverge. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/sbm.py:89-93`, `dca.py:141-147`, `kuo.py:94-96`, `mass_flux.py:224-240` - Several “smooth” schemes still use hard masks/clips in active physics paths.  
  These are JAX-compatible but create kinks and dead-gradient regions at cloud membership, saturation, mass-flux caps, and condensation thresholds. Severity: `major`.

- `src/legoesm/atmosphere/physics/convection/_plume.py:147-171`, `sbm.py:101-121`, `kuo.py:91-96`, `mass_flux.py:328-331` - The code documents `q_v` as specific humidity but compares it to `saturation_mixing_ratio`.  
  If model `q_v` is specific humidity, saturation should use `saturation_specific_humidity`; otherwise the docs and variable names are wrong. Severity: `minor`.

**Hypotheses Not Substantive Bugs**
- Hypothesis 2, `bechtold.py:199-202`: the `0.05 kg/m²/s` MC normalizer is dimensionally coherent and only a configurability/style issue.
- Hypothesis 3, `dca.py:257-265`: the rescale does preserve column water algebraically; the defect is the nonlocal, nonphysical vertical distribution, not column conservation.
- Hypothesis 5, `_plume.py:176-177`: the `T - 55 >= 1` floor only activates for `T < 56 K`, so it is not a normal “very cold parcel” atmospheric bug.
- Hypothesis 7, `mass_flux.py:132-159`: the smooth stratospheric gate being nonzero is an AD tradeoff, but the docstring overstates “vanishes/actually closed.”
- Hypothesis 10, `emanuel.py:160-170`: the bound is correct because the variance of values in `[0, 1]` is at most `1/4`; the bug is applying the multiplier to the whole kernel.
