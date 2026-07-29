Not closed: item 1 still rejects a valid `(nCells, 1)` SST/SIC source when the prognostic skin is enabled.

The raw guard correctly accepts it after flattening, but `_blend_T_sfc` passes the original 2-D arrays to `blend_surface_temperature`. With skin `(nCells,)`, `(nCells,1) * (nCells,)` broadcasts to `(nCells,nCells)`; the retained `_ts0` guard then rejects it. See [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6261) and [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6299).

Item 1 is fully resolved for scalar/mis-counted inputs and normal `(nCells,)` inputs, but not `(nCells,1)` inputs. Normalize inside `_blend_T_sfc`:

```python
_sst = jnp.asarray(_sst).reshape(-1)
_sic = jnp.asarray(_sic).reshape(-1)
_ts = blend_surface_temperature(_sst, _sic, _ice_component)
```

Item 2 is correctly fixed: the restored skin is shape-checked and finiteness-checked before assignment and before the first anchor blend at [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6212). The NaN refusal test is present.

So the packet has one substantive remaining compatibility regression: feature-on `(nCells,1)` forcing.