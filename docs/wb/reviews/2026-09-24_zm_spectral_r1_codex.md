
codex
HOLD

1. **[P1] Update the existing rejection test.** [test_spectral_pe_convection.py:263](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6_mem/tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:263) still requires ZM to raise with both `q_v` and `q_c` present. The new behavior correctly returns, breaking this test. Replace the obsolete expectation and update the exclusion rationale at line 617.

Verified against this checkout on CPU/fp64: **6 new tests passed; existing rejection test failed (`DID NOT RAISE`).** No additional correctness defect found in the guard, either rain-booking branch, or sign/units handling.

UNVERIFIED: compiled JIT/gradient execution and the reported reverted-change control were not independently rerun. Static inspection found no new data-dependent Python gate or nondifferentiable operation.
