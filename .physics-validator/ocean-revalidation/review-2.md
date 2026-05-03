**Verdict:** I confirm fixes #1-#6 and O1 as physically/sign-consistent under the code’s stated KPP convention (`B_f > 0` = unstable). I did not find a new production physics bug introduced by these diffs.

I verified with:

`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/unit/test_corrections.py::TestKPP tests/ocean/unit/test_visbeck_gm.py::TestGMRediWithVisbeck::test_visbeck_kappa_grad_finite_in_unstable_column tests/ocean/unit/test_plume_convection.py -q`

Result: `20 passed in 30.12s`.

Standalone probe results also matched the packet:

- Freshening: `B_salt = -2.491e-09`, stabilizing.
- Brine/evap: `B_salt = +2.491e-09`, destabilizing.
- KPP proxy stable column: `B_f = -1.977e-09`; proxy agrees with explicit stable forcing.
- Visbeck unstable-column gradient: no NaN/Inf.
- KPP `V_t²` zero-`N²` gradient: no NaN.
- KPP non-local column drift: `1.409e-18`, relative `2.135e-14`.

Open items #7 and #8 remain accurately classified. The surface-forcing/KPP diagnostic plumbing is still unresolved, and bottom drag remains a latent explicit-Euler stability risk. Plume units are fixed, but the noted lack of compensating entrainment/source conservation is still a real model limitation, not solved by this round.

Minor non-blocking note: `kpp.py` still computes `in_bl_full = sigma < 1.0` after the mask removal, but it is now unused. That is cleanup/lint noise, not a physics regression.
