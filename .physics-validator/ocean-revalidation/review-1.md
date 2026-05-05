**Findings**

I found additional remaining bugs.

1. KPP non-local tracer transport is not conservative. `F_T`/`F_S` are built as interface fluxes, then converted to a conservative divergence, but the code masks the resulting full-level tendencies afterward: [kpp.py:365](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:365>), [kpp.py:373](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:373>), [kpp.py:378](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:378>), [kpp.py:385](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:385>). If `h_bl` cuts through a grid cell, this drops the compensating tendency just below the BL. A 3-layer probe with `dz=10 m`, `h_bl=15 m` gives raw vertical integral `0.0`, masked integral `-0.074074... * gamma*Q`.

2. The plume convection tendency has wrong units and is non-conservative. `epsilon` is `[1/m]`, `T_plume - T` is `[K]`, so [plume.py:75](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/convection/plume.py:75>) produces `[K/m]`, not `[K/s]`; same for salinity at [plume.py:76](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/convection/plume.py:76>). `w_plume_min` exists in config but is unused: [config.py:20](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/convection/config.py:20>). There is also no compensating entrainment/source tendency, so heat and salt can be created or destroyed by the plume.

3. Visbeck GM is not AD-safe in unstable layers. [\_gm_redi_common.py:154](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py:154>) uses `sqrt(max(N2, 0))`. In JAX reverse mode, `grad(sqrt(max(x,0)))` is `nan` for `x < 0` and `inf` at `x == 0`. I confirmed `compute_visbeck_kappa_gm` on an unstable column returns finite primal `kappa` but `[[nan, nan, nan]]` gradients, even with the default `kappa_min` clip.

4. KPP Monin-Obukhov stability has a sign error relative to `B_f > 0 = unstable`. [kpp.py:234](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:234>) gives stable `B_f < 0` a negative `L_MO`, hence negative `zeta`; the stable branch then uses `max(zeta, 0)` at [kpp.py:263](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:263>) and applies no stable suppression. Probe: `B_f=-1e-7`, `u*=0.01`, `d=10 m` gives code `zeta=-0.4`, `w_s=0.004`; expected stable suppression gives `zeta=+0.4`, `w_s=0.00133`.

5. Your suspected KPP `B_f=None` proxy bug is real. For stable stratification, `rho[0] < rho[1]`, so [kpp.py:206](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:206>) is negative, and [kpp.py:207](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:207>) makes `B_f` positive, i.e. convective. The sign should be reversed, or the fallback should be removed/neutralized.

6. KPP interior momentum viscosity ignores `A_bg`. `K_interior` is built with `cfg.K_bg` at [kpp.py:282](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:282>), then reused for `A_v` outside the BL at [kpp.py:302](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:302>). With defaults, stable interior viscosity gets `1e-5`, not `A_bg=1e-4`.

7. The surface-forcing/KPP coupling is architecturally inconsistent. KPP reads only the external `OceanSurfaceForcing` object at [integration.py:109](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/integration.py:109>), while configured prescribed/bulk surface forcing is called as a separate tendency module and its diagnostics are discarded when wrapped: [surface_forcing/integration.py:58](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/surface_forcing/integration.py:58>), [surface_forcing/integration.py:62](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/surface_forcing/integration.py:62>). So configured `Q_net/tau` can force the top layer while KPP remains on proxies; external `q_net/freshwater` can drive KPP without applying the matching local heat/salt/free-surface tendency.

8. `implicit_bottom_drag_factor` is indeed explicit Euler. [ocean_tendency_common.py:279](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/dynamics/ocean_tendency_common.py:279>) returns `1 - dt*r/H`; if `dt*r/H > 1`, it flips velocity sign. I would not call this only cosmetic, though typical parameters make it small.

**Candidate Checks**

`B_salt = +g*beta*Q_sfc_S` is now correct at [integration.py:149](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/integration.py:149>).

`Ri_conv` is a naming/API issue, not a physics bug with the default `0.0`.

The convective `K_conv` add-then-cap behavior is redundant but functionally capped.

The Visbeck formula using `N`, not `N2`, is physically right; the bug is the JAX gradient through the clamp.

EOS sign tests look consistent in the shown code.

The cosine filter is a periodic Hann-like window; `n=1` is fixed. It is not the endpoint-including finite-sample Hann, but that is not necessarily wrong for substep cycling.
