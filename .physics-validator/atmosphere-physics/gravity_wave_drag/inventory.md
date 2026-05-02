# Gravity wave drag inventory

Scope: `src/legoesm/atmosphere/physics/gravity_wave_drag/`. Six schemes + integration.

## Schemes

| Scheme | File | LOC | Type |
|--------|------|-----|------|
| rayleigh | rayleigh.py | 90 | linear drag (sigma-based) |
| lindzen | lindzen.py | 123 | orographic, saturation breaking |
| mcfarlane | mcfarlane.py | 126 | orographic, smooth-min saturation |
| hines | hines.py | 125 | non-orographic, Doppler-spread |
| prognostic_spectral | prognostic_spectral.py | ~? | spectral wave action |
| ml_emulator | ml_emulator.py | ~? | learned |

## Output (`GWDOutput`)

* `du_dt`, `dv_dt` [m/s²], shape `(ncol, nlev)`.
* `dT_dt` [K/s] — frictional heating: `-(u du_dt + v dv_dt) / c_pd`.
* `eps_gwd` [W/m²] — column dissipation.

## Convention

* Surface at `[:, -1]`.
* Drag in opposite direction to low-level wind.
* All schemes apply gating to keep tendencies smooth/differentiable.
