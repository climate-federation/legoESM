
codex
SHIP — no new blocking findings.

1. Round-1 P1 resolved: [test_spectral_pe_convection.py:247](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6_mem/tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:247) passes, as do all six new routing tests.
2. Pre-existing failures remain at [test_spectral_pe_convection.py:699](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6_mem/tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py:699), :712 and :751: Emanuel/Bechtold reject the condensate-less fixture. All five reproduce with HEAD’s unchanged implementation.

Requested CPU/x64 run: **23 passed, 5 failed**.

UNVERIFIED: ZM-specific compiled JIT/gradient execution; static inspection found no new tracing or differentiation hazard.
