1. Fail. “Exact” compares only three `float64` moments, not elements; bool masks with equal true-count can differ positionally and pass. Float-comparison masks are not bit-reproducible across ULP-different inputs.

2. Pass conditionally. The payload bypasses `process_allgather`; `np.asarray` and broadcast preserve `float64`, and `jnp.asarray` preserves it when x64 is enabled on every process. Mixed x64 settings are not checked.

3. Pass conditionally. `_fields` order is deterministic, the list preserves it, and Python dict insertion order is guaranteed. A process-dependent field selection/schema would still desynchronize collectives before validation.

4. Conditional only. Once both gathers complete, every process computes the same verdict and raises together. Different field lists or differing `ndim` (variable-length `struct`) can break/mismatch a gather instead of reaching that symmetric `RuntimeError`.