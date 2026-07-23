Verdict: **not clean**.

Yes: a 1-D or `(nCells, 0)` `T` now raises in phase 1, before any accumulator mutation. I verified both leave `_max_count_ever == 0` and `_data == {}`.

But one commit-time failure path remains:

- A 2-D `T` with the wrong nonzero level count passes validation because `_nlev` is derived from `T`, rather than checked against `self.nlev` ([diagnostics.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1408)). After a normal `nlev=2` zonal feed, a malformed `(nCells, 3)` `T` causes `add_2d` to commit, then `add_3d` raises `IndexError` while updating the pre-existing 2-level profile ([diagnostics.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1537), [monthly_means.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/diagnostics/monthly_means.py:193)). Direct reproduction changed `T_low` and profile sums and advanced the call count from 2 to 4 before raising.

- Separately, a shape-correct `object`-dtype `T` also passes phase 1 and raises in zonal `add_2d` during `np.mean`, after creating/incrementing the bucket. Numeric coercion/validation belongs in phase 1 if the guarantee covers malformed values as well as shapes.

Require `T_np.shape[1] == self.nlev` in the initial guard (and validate/coerce native input dtypes in phase 1) before this can be called transactional in all modes. The CMOR-writer non-atomicity is therefore not the only residual.
