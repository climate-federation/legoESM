Confirmed. The flattening at [model_driver.py:6264](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6264) occurs before every blend/re-anchor path, while the raw per-cell guard remains intact at [model_driver.py:6296](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6296).

The targeted regression test passes, and direct CPU JAX checks confirm:

- `(nCells, 1)` SST/SIC + `(nCells,)` skin yields elementwise `(nCells,)`.
- Unflattened inputs reproduce the original `(nCells, nCells)` failure mode.
- Feature-off scalar-ice results remain bit-identical for `(nCells,1)` sources.

No substantive findings remain. The packet is closed.