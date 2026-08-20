No substantive runtime physics, unit, sign, conservation, monotonicity, or AD defect remains.

The three round-2 items are materially resolved:

- B2 now has the log-pressure midpoint discriminator, matching the log-CDF implementation.
- The real-file loader → remap → RRTMGP OLR regression covers the requested compositional chain.
- B3 now clearly calls the gray collapse a heuristic, and the geopotential-pressure comment is corrected.

One test-adequacy gap remains: B3’s loader use of Planck weights is not regression-protected. The loader fixture uses band-uniform extinction, so Planck and flat band means produce the same result; the Planck test exercises only the helper. A regression that replaces the loader’s weighted collapse with a flat mean could therefore evade these tests. See [test_volcanic_lw_aerosol.py:89](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_volcanic_lw_aerosol.py:89) and [test_volcanic_lw_aerosol.py:193](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_volcanic_lw_aerosol.py:193). Add a two-band, band-varying synthetic profile with a known weighted expected value.

Minor wording only: “strictly better” remains stronger than warranted for a documented heuristic in [external.py:133](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/forcing/external.py:133).

I independently ran five non-solver targeted nodes, including real-file vertical placement and AD checks: **5 passed**. The solver regression could not be rerun here because the sandbox hit an XLA thread-creation limit, not an assertion failure.
