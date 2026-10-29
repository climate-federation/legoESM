# Optimality switches of the two-leaf canopy

Three arms are compared on the global LMIP driver and the EC-site harness:
prescribed capacities (baseline), the least-cost P model, and P-hydro.  Four
physics switches select them; `physics.optimality` (YAML) / `--optimality`
(site driver) is a wrapper that fills all four at once.

| `optimality` | `canopy_capacity_scheme` | `canopy_g1_source` | `transpiration_stress` | `stomatal_model` |
|---|---|---|---|---|
| *(absent / `none`)* | as given (`prescribed`) | as given (`table`) | as given (`beta_theta`) | as given (`ball_berry` / `medlyn` / `leuning`) |
| `least_cost` | `p_model` | `p_model` | `beta_theta` | `medlyn` |
| `phydro` | `phydro` | `table` | `phydro` | `phydro` |

Meaning of the values:

* `canopy_capacity_scheme`: where leaf-top Vcmax25 and Jmax25 come from.
  `prescribed` = site / PFT table; `p_model` = the least-cost acclimated
  optimum (`land/p_model.py`); `phydro` = the P-hydro acclimated optimum
  (`land/phydro.py`), two-leaf canopy only.
* `canopy_g1_source`: where the Medlyn slope comes from. `table`, or
  `p_model` = the least-cost chi mapped to a slope (requires `medlyn`).  The
  P-hydro arm reads no slope.
* `transpiration_stress`: `beta_theta` = the empirical root-zone
  soil-moisture factor on Vcmax and on the Medlyn slope (tabulated or
  least-cost);
  `phydro` = the hydraulic supply curve replaces it.
* `stomatal_model`: the half-hourly leaf conductance.  `phydro` = the
  instantaneous P-hydro optimum solved inside the canopy Newton closure
  (two extra unknowns, the sunlit and shaded draw-downs).

Rules, enforced in `lmip_config.validate_config` and at trace time in
`two_leaf_canopy`:

* A switch given next to `optimality` must agree with it.
* The three `phydro` values come together (`optimality: phydro` does this);
  any one alone is refused, and `canopy_g1_source` must then be `table`.
* `capacity_scheme='phydro'` is refused on the big-leaf and CLM-ML paths.

The Ginsburg launcher (`scripts/run/submit_lmip_ginsburg.sh --arm`) maps
`baseline` to the explicit switches with `optimality=none`, and the other
two arms to `optimality=least_cost` / `optimality=phydro`.
