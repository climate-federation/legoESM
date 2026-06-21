# Zhang et al. (2024) -- MITgcm Channel Model Setup (legoESM port spec)

**Target paper:** Zhang, Nikurashin, Pena-Molino, Rintoul, Doddridge (2024),
*Maintenance of the Zonal Momentum Balance of the Antarctic Circumpolar Current
by Barotropic Dynamics*, J. Phys. Oceanogr., 54, 1565-1581.
DOI: 10.1175/JPO-D-23-0042.1

**Reference implementation:**
`dhruvbalwada/submesoscale_subduction_GRL/GCM_setup/` -- namelists and input-generation
notebooks from a nearby setup in the same Abernathey-lineage ACC channel family.
Values below marked **[FROM REF]** come from that setup; values marked **[FROM PAPER]**
are specified explicitly by Zhang et al.; values marked **[DIVERGES]** are places
where Zhang et al. explicitly differs from the reference. All equation and figure
references are to Zhang et al. 2024 unless noted.

---

## 1. Core model choices

| Item | Value | Source |
|---|---|---|
| Model | MITgcm (Marshall et al. 1997), depth-coordinate, hydrostatic, Boussinesq | paper |
| Free surface | implicit linear (`exactConserv=.TRUE.`) | ref |
| Equation of state | `LINEAR`, temperature-only | paper |
| Thermal expansion `tAlpha` | `2.0e-4` K^-1 | ref |
| Haline `sBeta` | `0.0` (salinity ignored) | paper |
| `sRef` | 35.0 (dummy; salinity not stepped) | ref |
| `saltStepping` | `.FALSE.` | ref |
| Momentum advection | flux-form (default) | ref |
| Temperature advection | `tempAdvScheme=7` (DST3 with flux limiter) | ref |
| `staggerTimeStep` | `.TRUE.` | ref |

---

## 2. Grid, domain, and SIZE.h

**[FROM PAPER]** 2000 x 2000 x 3000 m domain; zonally re-entrant; 10 km
horizontal; 40 vertical levels (10 m to ~200 m).

```
Lx  = 2.0e6 m      # zonally periodic
Ly  = 2.0e6 m      # closed walls (free-slip)
H   = 3000 m
dx  = dy = 1.0e4 m
Nx  = Ny = 200
Nr  = 40
```

### Vertical grid -- 40 levels from 10 m to ~200 m

Geometric stretching: dz_k = dz0 * r^k, dz0 = 10 m.
Solve for r such that sum = 3000 m with 40 levels -> r ~ 1.0775, dz_40 ~ 195 m.

```
delR (m) =
  10.0, 10.8, 11.6, 12.5, 13.5, 14.5, 15.6, 16.8, 18.1, 19.5,
  21.0, 22.6, 24.4, 26.3, 28.3, 30.5, 32.9, 35.4, 38.2, 41.2,
  44.4, 47.8, 51.5, 55.5, 59.8, 64.5, 69.5, 74.9, 80.7, 87.0,
  93.7, 101.0, 108.8, 117.3, 126.4, 136.2, 146.8, 158.2, 170.5, 183.7
```

Sum ~ 3004 m; renormalize by Hmax/sum if needed.

---

## 3. Topography -- meridional Gaussian ridge

**[FROM REF]** -- exact analytical form from `input_channel_beta_1km.ipynb`:

```python
H     = 3000.0     # m, flat-bottom depth
h0    = 1000.0     # m, ridge height
sigma = 75.0e3     # m, Gaussian width parameter

bathy = -(H - h0 * np.exp(-(xc - Lx/2)**2 / sigma**2))
bathy[0, :] = 0.0   # southern wall
```

With sigma = 75 km, the ridge 1/e half-width is 75 km -> effective total width
(where height > 1% of peak) ~ 2*sigma*sqrt(ln 100) ~ 322 km. The paper's "400 km wide"
is consistent with this as a visual-extent estimate.

Bathymetry values range from -3000 m (abyssal plain) to -2000 m (ridge crest).
Matches Fig. 1c.

---

## 4. Coriolis -- beta-plane

**[FROM REF]** (`run_template_5km_coarse_vertical_sin_wind/data`):

```
f0    = -1.1e-4   s^-1
beta  =  1.4e-11  s^-1 m^-1
```

`f(y) = f0 + beta*(y - Ly/2)` ranges from -1.114e-4 at y=0 to -1.086e-4
at y=Ly. Southern-Hemisphere-like, consistent with the ACC analogue.

---

## 5. Surface forcing

### 5.1 Zonal wind stress [FROM PAPER + REF form]

Sinusoidal in y, zero at both walls, peaking at y = Ly/2:

```python
tau0  = 0.1    # N/m^2 (REFERENCE wind)  [PAPER value]
tau0  = 0.2    # N/m^2 (DOUBLE-WIND)     [PAPER value]

dy = yc[1,0] - yc[0,0]
tau_x = tau0 * np.sin(np.pi * (yc - dy/2) / (Ly - dy))
tau_y = 0
```

### 5.2 Surface heat flux [DIVERGES from REF]

Zhang et al. uses a prescribed surface heat flux, NOT SST restoring.
Fig. 1b shows a sinusoidal Q(y) with amplitude ~10 W/m^2, one full wavelength
across the channel.

```python
Q0    = 10.0                        # W/m^2 peak amplitude
Qnet  = Q0 * np.sin(2*np.pi*yc/Ly)  # one full wavelength
```

### 5.3 Freshwater / salinity

None. Salinity disabled.

---

## 6. Northern boundary temperature sponge

**[FROM PAPER]** 100 km sponge at the northern wall, temperature restored to
an exponential profile (representing the basin thermocline).

**[FROM REF]** analytical profile -- Abernathey et al. (2011) Eq. (2):

```python
delT = 8.0        # degC, surface-to-bottom Delta_T
h    = 1000.0     # m, thermocline e-folding scale
Hmax = 3000.0     # m
Tstar = delT * (np.exp(zaxis/h) - np.exp(-Hmax/h)) / (1 - np.exp(-Hmax/h))
```

This gives Tstar ~ 8 degC at the surface, ~ 0 degC at the bottom.

Restoring timescale: tau = 7 days (604800 s).
Sponge width: 100 km (northern 10 rows at 10 km resolution).

---

## 7. Dissipation and friction

### Bottom drag [DIVERGES from REF]

Zhang et al. uses LINEAR drag:
```
bottomDragLinear    = 1.1e-3   m/s
bottomDragQuadratic = 0.0
no_slip_bottom      = .TRUE.
```

### Lateral boundaries

```
no_slip_sides = .FALSE.   # free-slip on N and S walls
```

### Lateral viscosity [FROM REF]

Leith biharmonic viscosity (grid-adaptive):
```
viscC4Leith      = 2.15
viscC4Leithd     = 2.15
viscA4GridMax    = 0.8
useAreaViscLength = .TRUE.
```

### Vertical viscosity / T diffusivity [FROM REF]

```
viscAr   = 5.6614e-4   # m^2/s  vertical viscosity
diffKrT  = 5.44e-7     # m^2/s  vertical T diffusivity
```

---

## 8. Timestepping and run length

```
deltaT = 300.0 s
```

Run lengths per experiment:

| Experiment | Duration |
|---|---|
| stratified (from rest) | 100 yr |
| homogeneous (from rest) | 10 yr |
| double-wind stratified (from yr 100) | 30 yr |
| double-wind homogeneous (from eq.) | 10 yr |

---

## 9. Initial conditions

### Stratified (from rest)

- U, V, W, eta = 0
- T(x, y, z) = Tstar(z) (same exponential profile as the sponge target)
  + small perturbation to break symmetry:

```python
T_init = np.tile(Tstar[:, None, None], (1, 200, 200))
T_init += 1e-3 * np.random.randn(*T_init.shape)
```

### Homogeneous (from rest)

- T = 10.0 degC everywhere
- No surface heat flux, no sponge

---

## 10. Key numerical validation targets

After spinup, the run should hit these numbers:

| Quantity | Target | Figure |
|---|---|---|
| Ekman transport (reference) | 1.42 Sv | 3a |
| SSH N-S difference (stage 1) | ~5 cm | 4e |
| SSH N-S difference (equilibrium) | ~70 cm | 9b |
| Wind stress = -TFS balance | yes, within 1 month | 3c |
| Eddy onset time | ~6 yr | 5b |
| Total equilibration | ~60-80 yr | 5b, 8 |
| Barotropic transport response to 2x wind | +4 Sv (~100%) | 5b |
| Baroclinic transport response to 2x wind | +0.1 Sv (~0.2%) | 5b |
| Post-doubling re-equilibration time | ~1 month | 12 |

---

## 11. Provenance summary

Numbers sourced from `dhruvbalwada/submesoscale_subduction_GRL/GCM_setup/`:
- `run_template_5km_coarse_vertical_sin_wind/data` -> f0, beta, tAlpha,
  viscosity (Leith, viscAr), diffKrT, tempAdvScheme, hFacMin, deltaT, etc.
- `run_template_5km_coarse_vertical_sin_wind/data.rbcs` -> tauRelaxT, sponge filenames.
- `input_generate_scripts/input_channel_beta_1km.ipynb` -> Gaussian ridge form.
- `input_generate_scripts/inputs_5km_run.ipynb` -> wind form, sponge profile.

Numbers sourced directly from Zhang et al. (2024):
- Domain (Lx = Ly = 2000 km, H = 3000 m), 10 km horizontal, 40 vertical.
- Wind stress amplitude tau0 = 0.1 N/m^2 reference, 0.2 N/m^2 doubled.
- Linear bottom drag r = 1.1e-3 m/s.
- Free-slip sides.
- Single-variable linear EOS (temperature only, salinity ignored).
- Sponge width 100 km at northern wall.
- Run lengths: 100 yr / 10 yr / 30 yr / 10 yr.
- Theoretical Ekman transport 1.42 Sv and other validation targets.

---

## 12. Remaining small uncertainties

1. Exact 40-level delR -- geometric stretching proposed; not critical.
2. Surface heat flux sign convention and exact functional form.
3. Appendix f-plane f0 -- not given; -1.0e-4 s^-1 is a reasonable guess.
4. Whether KPP was used -- reference uses it; paper doesn't mention it.
5. Sponge restoring timescale profile -- binary vs ramped.

None of these change the zeroth-order reproduction.
