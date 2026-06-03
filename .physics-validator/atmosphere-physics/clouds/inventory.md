# Clouds inventory

Scope: `src/legoesm/atmosphere/physics/clouds/`. One module — diagnostic cloud fraction.

## Files

* `cloud_fraction.py` (213 LOC): `sundqvist_cloud_fraction`, `xu_randall_cloud_fraction`, `compute_cloud_properties`.
* `config.py` (52 LOC): `CloudConfig` NamedTuple.

## Output (`CloudProperties`)

* `cloud_fraction` ∈ [0, 1], shape `(ncol, nlev)`.
* `lwp`, `iwp` [kg/m²] grid-mean liquid/ice water paths per layer.
* `r_eff_liq`, `r_eff_ice` [m] effective radii (constants in current implementation).

## Convention

* Surface at `[:, -1]`.
* `q_cloud`, `q_ice` from microphysics (or diagnosed from `q_c_diagnostic` if absent).
* Ice fraction: linear ramp T_freeze → T_ice_only.
