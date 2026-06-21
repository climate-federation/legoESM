# Why is the depth-mean flow in our Drake band westward at -3.7 cm/s?

A literature-anchored synthesis of the zonal-momentum balance for a coarse-resolution,
GM/Redi-parameterized, flat-bottom circumpolar channel and a diagnosis of the most likely
single cause of the observed sign flip.

Context recap. 5 deg x 5 deg, 20 z-star levels, H_max = 4000 m. Drake band is flat at
4000 m. Two-belt Gaussian westerlies (peak 0.1 Pa, band-mean tau_x ~ +0.024 Pa).
Linear bottom drag r = 0.0011 m/s applied implicitly to U_baro (not bottom cell).
GM/Redi triads with Visbeck adaptive kappa. After 50 yr the band is in barotropic
quasi-equilibrium with U_baro ~ -3.7 cm/s and Drake transport ~ -405 Sv;
the naive wind/drag prediction is U_baro = tau_x / (rho0 r) ~ +2.1 cm/s.

---

## 1. Where is the missing momentum sink/source?

Punchline. In a flat-bottom zonally periodic channel, the wind cannot reach the bottom
through topographic form drag (the dominant sink at 95% in the real ACC). The leftover
sinks are bottom friction and lateral momentum-flux convergence at the channel walls,
and in coarse z-models these are too weak and structurally asymmetric to set the right
sign.

The depth- and zonally-integrated zonal momentum balance for the band (closed by N-S
walls, zonally periodic) is, after vanishing of the zonal-mean Coriolis term in zonal
average,

  d/dt INT u dz dx = INT tau_x dx + [TFS] - INT tau_b dx
                     - d/dy INT INT (u_bar v_bar + u' v') dz dx + [lateral viscous].

[TFS] = topographic form stress, identically zero for flat bathymetry. In the real ACC
TFS balances ~95% of the wind, with bottom friction + meridional momentum-flux divergence
contributing ~5% (Masich et al. 2015, JGR; Stewart-Hogg 2017; Olbers-Eden review of
Johnson-Bryden 1989, DSR; Munk-Palmen 1951). With TFS = 0, the model is forced to close
the budget through (i) drag on U_baro and (ii) momentum flux divergence at the walls.

In a coarse model both are pathological:
- (i) Linear drag is parametrically weak: tau_x / (rho0 r) ~ 2 cm/s, but actually achieving
  this barotropic level requires that everything else in the budget integrates cleanly to
  zero, which in a stratified, eddying channel it does not.
- (ii) Eulerian-mean v_bar d_y u_bar (and w_bar d_z u_bar) is not negligible when the
  Deacon cell is fully present (it is, because in coarse z-models without GM-on-momentum
  the Deacon cell is not cancelled in the Eulerian-mean equations). LaCasce-Isachsen
  (2010, Prog. Oceanogr.) review the linear flat-bottom analytical solutions: transport
  scales like 1/r and is "too large for reasonable values of the bottom drag coefficient",
  but more importantly the *direction* in such solutions is set entirely by the f-plane
  geostrophic adjustment of the meridional transport against the walls. With closed
  geostrophic contours and continents to the north, that direction can flip sign relative
  to the wind in a way that has nothing to do with the wind itself.

So in our flat-bottom band the momentum balance is structurally underdetermined: with
TFS = 0 and only depth-averaged drag and weak lateral fluxes available, the model is
free to settle on a barotropic state that violates the naive wind/drag scaling.

## 2. Is GM-on-tracers without GM-on-momentum the dominant problem here?

Punchline. Yes - this is the single most-cited structural defect of coarse z-channel
ACC simulations. Standard GM (Gent-McWilliams 1990, JPO; Gent et al. 1995, JPO) acts on
tracers/thickness only, leaving the Eulerian-mean momentum equation uncoupled to eddy
form stress; in TEM language this means the resolved momentum equation sees no
interfacial form drag, so wind cannot be transmitted from surface to bottom by the
parameterized eddies.

Wardle-Marshall (2000, JPO 30) showed explicitly in a primitive-equation channel that
the equilibrium ACC momentum balance is wind = vertical eddy form stress = bottom drag
or topographic form stress. The vertical transmission is *eddy form stress on the
Eulerian momentum*; it does not appear in standard GM-on-tracers schemes. Treguier et
al. (1997, JPO 27) and Treguier-McWilliams (1990, JPO) made the same point in QG
channels. The recognized fixes:

- Greatbatch-Lamb (1990, JPO) parameterized the eddy interfacial form stress as a
  vertical viscosity on momentum (`GL90`). This is mathematically equivalent to GM on
  tracers under QG (see e.g., Loose et al. 2023, JAMES) but has to be added separately
  to the momentum equation in a z-coordinate Eulerian-mean code.
- Eden-Greatbatch (2008, Ocean Mod. 20) and Marshall-Adcroft (2010, Ocean Mod. 32)
  derive PV-mixing closures that fold both eddy buoyancy and eddy momentum into a single
  PV-flux term, generating a Reynolds-stress-equivalent forcing of the mean momentum.
- Marshall et al. (2012, JPO 42) and Ringler-Gent (2011, Ocean Mod. 39) follow up with
  full Ertel-PV closures.

In production GCMs that build on standard z-coordinate GM (NCAR POP/CESM, GFDL MOM, MPI),
the Greatbatch-Lamb-style vertical momentum mixing is *not* the default; only MOM6 and
some research configurations include it. Gent (2001, JGR) showed that ACC transport in
the NCAR ocean is set primarily by the GM thickness diffusivity, not by wind, precisely
because the wind reaches the bottom via the Deacon-cell-cancelling residual circulation
in the tracer equation rather than via the momentum equation. Without GM-on-momentum
(or at least Visbeck-style closure that is coupled to the momentum tendency), the
Eulerian-mean u in a coarse flat-bottom channel sees no eddy-mediated coupling between
the fast surface response and the slow deep response, and the resulting depth-mean state
is governed by the residual of two large numbers (mean advection vs. bottom drag) that
in our run lands on the wrong side of zero.

This is your most likely structural culprit. In the Treguier-McWilliams hierarchy our
configuration is the worst-case: flat bottom (no TFS) plus GM-on-tracers-only (no
parameterized vertical momentum transmission).

## 3. Drag-on-U_baro vs. drag-on-bottom-cell

Punchline. The depth-mean drag formulation `(1 - dt r/H)` does *not* directly
constrain u_bottom; it only constrains the column average. In a stratified channel that
sets up a strong thermal-wind shear, this leaves a free integration constant that the
slow modes select on the abyssal-flushing timescale; bottom-cell drag, by contrast,
imposes a kinematic constraint at the actual level where the deep flow lives.

For an unstratified column they are equivalent. With stratification, "drag on U_baro"
can be satisfied by *any* u(z) profile whose depth average sits at the wind/drag
equilibrium. Now the GM thickness flux feeds back into the density field, which through
thermal wind sets the shape of u(z) but does not pin the level of no motion. With a
weak constraint on the integration constant and an effectively rigid lid (z-star with
small free surface), the model picks whatever U_baro minimizes the residual of the much
larger thermal-wind and Coriolis terms - this can easily land at the "wrong" sign,
especially when the deep T field is still carrying the initial-condition memory
(see Section 4).

There is also a subtle interaction with GM. In z-coordinates GM bolus velocity has zero
depth integral (mass-conserving), so GM does not directly create a barotropic mode.
However, GM modifies the *meridional buoyancy gradient* by flattening isopycnals; the
resulting thermal wind shear is consistent with the observed -7 cm/s bottom vs +1 cm/s
top decomposition. The integration constant for that shear is set by the slow column
mass balance (continuity + Coriolis + drag) and is essentially unconstrained on 50-yr
timescales.

## 4. Is 50 years simply too short?

Punchline. Yes. In a coarse z-coordinate flat-bottom channel the abyssal mass field
equilibrates on the diapycnal mixing timescale H^2/kappa_v, which for kappa_v ~ 1e-5 m^2/s
and H = 4000 m is O(1500 yr); Wolfe-Cessi (2010, JPO; 2011, JPO) and Munday-Hogg-Marshall
(2013, JPO 43) integrate idealized channels for 200-2000 yr to claim equilibrium.

The deep T field in your run still carries the initial profile, so the meridional
density gradient at depth is not the equilibrium GM-balanced gradient - it is a transient
"thermal wind" whose integration constant has not yet relaxed against the slow
abyssal-flushing process. The bottom-mean -7 cm/s is consistent with: (a) initial dT/dy
at 4000 m being large (full surface gradient propagated by the IC), (b) thermal wind
giving large u(z) shear, (c) the column average closing onto a non-zero U_baro because
the slow Coriolis-meridional-mass balance has not finished.

Munday-Hogg-Marshall (2013) note this explicitly: equilibrated transport in coarse
GM-channel runs is approached only after ~1000 yr; the first few hundred years can
exhibit transient ACC transports of the wrong sign or magnitude, especially when the
deep stratification has not adjusted.

## 5. Is the pattern a recognized failure mode?

Punchline. Yes. "Flat-bottom coarse channel + GM + linear drag = pathological barotropic
mode" is the textbook diagnosis going back to Munk-Palmen (1951), Gill (1968), Hidaka
(see Hidaka-Tsuchiya 1953), Johnson-Bryden (1989), Olbers et al. (2004, Ocean Dynamics),
and the LaCasce-Isachsen (2010) review.

The classic statement: with no topography, the only available momentum sink is bottom
friction acting on the depth-mean flow; analytical linear-channel solutions (Gill 1968,
Ishida 1994) give transport ~ tau / r times the channel width and length, which for
realistic r yields O(1000 Sv) transports of either sign depending on the f-plane
geometry. Olbers-Borowski-Voelker-Wolff (2004), Olbers (2005), Hughes-De Cuevas (2001),
and the Olbers-Eden chapter in Olbers-Willebrand-Eden (2012, *Ocean Dynamics*) all
emphasize that no flat-bottom channel - QG or PE, coarse or eddy-resolving - reproduces
the observed ACC transport without form stress; the failure can be sign as well as
magnitude.

The specific signature you see (large westward U_baro with eastward forcing) is
consistent with the Ishida (1994) regime, where closed geostrophic contours bend the
Sverdrup balance into the channel and the resulting depth-mean flow follows f/H contours
rather than the wind.

## 6. The single cleanest diagnostic next step

Punchline. Compute the depth-and-zonally-integrated zonal momentum budget for the Drake
band, term by term, and identify the residual.

Define the band as Omega = {j in [j_S, j_N], all i, all k}. The exact discrete form
(Eulerian, on the C-grid) is:

  d/dt INT_Omega u dV
    =   F_wind     = INT (tau_x / rho0) dA_top
      - F_botdrag  = - INT (tau_b / rho0) dA_bot      where tau_b = rho0 r U_baro for your scheme
      - F_advmean  = INT_Omega [d_x(u u_bar) + d_y(v u_bar) + d_z(w u_bar)] dV
      - F_eddyflux = INT_Omega [d_x<u'u'> + d_y<v'u'> + d_z<w'u'>] dV   ~ 0 if no resolved eddies
      - F_GMadv    = INT_Omega [d_y(v* u_bar) + d_z(w* u_bar)] dV       (GM bolus advection of u, if used)
      - F_pgf      = - INT_Omega (1/rho0) d_x p dV     -> 0 by zonal periodicity (band-integrated)
      - F_lateral_visc + F_vertical_visc.

For zonally periodic, walls-N/S band the Coriolis and zonal-PGF terms drop in the
zonal-integral; the meaningful terms are F_wind, F_botdrag, F_advmean (Deacon-cell
plus mean meridional shear), F_lateral_visc.

What to record:
- F_wind (positive, eastward; ~ +0.024 Pa * area / rho0)
- F_botdrag (negative if U_baro positive; in your run *positive*, because U_baro is negative
  and tau_b = rho0 r U_baro is westward at the bottom, meaning the ocean does work on the
  bottom in the westward sense, removing westward momentum - i.e., drag is *adding*
  positive momentum to the column, so it is on the same sign side as the wind).
- F_advmean: split into v_bar d_y u_bar (Eulerian-mean Deacon-cell advection) and
  w_bar d_z u_bar (vertical advection). This is the term you most need to plot - in flat-
  bottom coarse channels it can dominate.
- F_lateral_visc: at the N/S walls (no-slip or free-slip).
- Residual = d/dt - sum, which should be ~ 0 in equilibrium; non-zero residual identifies
  the unphysical numerical sink/source.

Implementation. Compute all six volume integrals each step (or output them as an MFDataset
diagnostic), zonal-mean and time-average the last 5 yr. The expected outcome:

  F_wind  +  F_botdrag  ~  F_advmean   (all O(0.1) N/m * length, with F_advmean dominating)

If F_advmean closes the budget and is ~5-10x F_wind, you have confirmed that the Eulerian-
mean Deacon-cell advection of u is the missing sink/source. That points directly to the
GM-on-momentum gap (Section 2).

If instead F_lateral_visc or a numerical residual closes it, the diagnosis is different
(numerical viscosity at walls, or a discretization sink in the depth-mean drag scheme).

A faster smoke-test: re-run with (a) GM kappa = 0 (turn off GM on tracers entirely;
expect different but still pathological); (b) drag on bottom cell only instead of U_baro;
(c) double H to 8000 m to weaken bottom drag effect. Compare U_baro evolution. The
sensitivity pattern across (a, b, c) plus the budget terms above should uniquely identify
the dominant cause.

---

## Bottom line

The most likely single cause of your -3.7 cm/s westward U_baro is structural and
well-known: a flat-bottom coarse-resolution z-coordinate channel with GM acting on
tracers only (no Greatbatch-Lamb-style eddy form stress on momentum) has no physical
mechanism to transmit wind stress from the surface to a momentum sink at the bottom.
With topographic form stress identically zero and GM-on-momentum absent, the only
remaining sinks (depth-mean linear drag, weak lateral viscosity at the N/S walls) cannot
balance the Eulerian-mean Deacon-cell advection of zonal momentum, and the column-mean
flow drifts off the naive wind/drag scaling onto a state determined by f/H geostrophic
contours and the still-unequilibrated deep mass field. After only 50 yr the abyssal
T field is far from steady (equilibration timescale O(1000 yr); see Munday-Hogg-Marshall
2013, Wolfe-Cessi 2010), so the integration constant for thermal wind has not relaxed,
and the depth-mean transport is partly transient.

The cleanest test is the depth-zonal-integrated band momentum budget (Section 6): if
F_advmean dominates and F_botdrag has the wrong sign relative to the naive scaling, you
have confirmed the GM-on-momentum gap. The cure is either (i) add a GL90 vertical
momentum viscosity matched to the GM kappa, (ii) introduce even modest topography to
generate TFS (your 25% ridge experiment already gave +82 Sv recovery and is the right
direction), or (iii) integrate to 500-1000 yr so the deep mass field equilibrates and
the analysis becomes meaningful.

## References (selected)

- Eden, C. and R. J. Greatbatch, 2008: Towards a mesoscale eddy closure. *Ocean Modelling* 20, 223-239.
- Gent, P. R. and J. C. McWilliams, 1990: Isopycnal mixing in ocean circulation models. *J. Phys. Oceanogr.* 20, 150-155.
- Gent, P. R., 2001: What sets the mean transport through Drake Passage? *J. Geophys. Res.* 106, 2693-2712.
- Gill, A. E., 1968: A linear model of the Antarctic Circumpolar Current. *J. Fluid Mech.* 32, 465-488.
- Greatbatch, R. J. and K. G. Lamb, 1990: On parameterizing vertical mixing of momentum in non-eddy resolving ocean models. *J. Phys. Oceanogr.* 20, 1634-1637.
- Hallberg, R. and A. Gnanadesikan, 2001: An exploration of the role of transient eddies in determining the transport of a zonally reentrant current. *J. Phys. Oceanogr.* 31, 3312-3330.
- Johnson, G. C. and H. L. Bryden, 1989: On the size of the Antarctic Circumpolar Current. *Deep-Sea Res.* 36, 39-53.
- LaCasce, J. H. and P. E. Isachsen, 2010: The linear models of the ACC. *Prog. Oceanogr.* 84, 139-157.
- Loose, N., et al., 2023: Comparing two parameterizations for the restratification effect of mesoscale eddies. *J. Adv. Model. Earth Syst.* 15.
- Marshall, D. P. and A. J. Adcroft, 2010: Parameterization of ocean eddies: Potential vorticity mixing, energetics and Arnold's first stability theorem. *Ocean Modelling* 32, 188-204.
- Marshall, J. and T. Radko, 2003: Residual-mean solutions for the Antarctic Circumpolar Current and its associated overturning circulation. *J. Phys. Oceanogr.* 33, 2341-2354.
- Marshall, D. P., J. R. Maddison, and P. S. Berloff, 2012: A framework for parameterizing eddy potential vorticity fluxes. *J. Phys. Oceanogr.* 42, 539-557.
- Masich, J., T. K. Chereskin, and M. R. Mazloff, 2015: Topographic form stress in the Southern Ocean State Estimate. *J. Geophys. Res.* 120, 7919-7933.
- Munday, D. R., H. L. Johnson, and D. P. Marshall, 2013: Eddy saturation of equilibrated circumpolar currents. *J. Phys. Oceanogr.* 43, 507-532.
- Munk, W. H. and E. Palmen, 1951: Note on the dynamics of the Antarctic Circumpolar Current. *Tellus* 3, 53-55.
- Olbers, D., D. Borowski, C. Voelker, and J.-O. Wolff, 2004: The dynamical balance, transport and circulation of the Antarctic Circumpolar Current. *Antarctic Sci.* 16, 439-470.
- Olbers, D., J. Willebrand, and C. Eden, 2012: *Ocean Dynamics.* Springer.
- Ringler, T. and P. Gent, 2011: An eddy closure for potential vorticity. *Ocean Modelling* 39, 125-134.
- Stewart, A. L. and A. McC. Hogg, 2017 / Bai-Wang-Stewart, 2021: Does topographic form stress impede prograde ocean currents? *J. Phys. Oceanogr.*
- Treguier, A. M. and J. C. McWilliams, 1990: Topographic influences on wind-driven stratified flow in a beta-plane channel. *J. Phys. Oceanogr.* 20, 321-343.
- Treguier, A. M., 1997: The Southern Ocean momentum balance. *J. Phys. Oceanogr.* 27, 2219.
- Wardle, R. and J. Marshall, 2000: Representation of eddies in primitive equation models by a PV flux. *J. Phys. Oceanogr.* 30, 2481-2503.
- Wolfe, C. L. and P. Cessi, 2010 / 2011: The adiabatic pole-to-pole overturning circulation. *J. Phys. Oceanogr.* 41, 1795-1810.
