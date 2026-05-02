# Clouds — static analysis

## A. Confirmed correctness

1. **Sundqvist cloud fraction** `cf = clamp((RH - rh_crit)/(1 - rh_crit), 0, 1)`. Standard. ✓
2. **Xu-Randall**: `cf = RH^p * (1 - exp(-α q_c / ((1-RH) q_s)))`. ✓
3. **Ice fraction**: linear ramp 0→1 from `T_freeze` to `T_ice_only`. Standard. ✓
4. **Grid-mean LWP/IWP**: `lwp = q_c * dp / g`. ✓
5. **`saturation_mixing_ratio` from `legoesm.thermo`** — correctly imported.

## B. Issues (RANKED)

### B1. **`config.py:50` `T_freeze = 273.15` literal in default** [STYLE]
```python
T_freeze: float = 273.15  # = constants.T_freeze
```
Comment correctly notes the canonical name. Per CLAUDE.md "NamedTuple field defaults may use literal floats with a `# = constants.X` comment" — this IS allowed. ✓

### B2. **`config.py:51` `T_ice_only = 233.15`** [STYLE]
No corresponding `constants.T_ice_only`. Acceptable for scheme-specific tuning.

### B3. **`r_eff_liq=10e-6`, `r_eff_ice=30e-6` constants** [INFO]
Reasonable broadband averages. Production codes typically diagnose r_eff from microphysics two-moment N_c. Acceptable simplification noted in the docstring.

### B4. **`Sundqvist`: clamping when q_sat is tiny may produce 0 cloud where there should be cf>0** [INFO]
`compute_cloud_properties:160`: `RH = q_v / max(q_sat, 1e-10)`. Tiny q_sat yields RH ≫ 1, cloud fraction = 1. ✓ correct limit.

### B5. **`xu_randall_cloud_fraction:117` `denominator = max((1-RH) * q_sat, 1e-10)`** [INFO]
At RH = 1, denom = 1e-10, exponent → -∞, exp → 0, cf = RH^p. So at saturation cf approaches `RH^p`, capped at 1. Reasonable.

### B6. **No support for cloud overlap** [INFO]
The grid-mean LWP/IWP are computed assuming each layer is independent — equivalent to "random overlap" or "no overlap" assumption when fed to RRTMGP. Not a bug; documented in docstring: "These are grid-mean (not in-cloud) values, which is what RRTMGP expects when treating each layer independently (no overlap assumption)."

## C. Top issues

None requiring action. Module is small, tight, and physically reasonable.
