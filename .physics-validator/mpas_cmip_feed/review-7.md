CLEAN — no substantive blocking bugs found.

[The feed](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1386) coerces and validates every listed numeric input before regridding or accumulator mutation; `T` rank and `self.nlev` are enforced first. This covers spatial-only, zonal-only, and combined feeds. The no-accumulator mode returns without mutation.

The only remaining non-atomic path is the inherited field-by-field CMOR writing shared with `save()`. I could not rerun pytest here because it is not installed in this shell.
