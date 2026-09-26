# Preregistration: ordered Redi tracer-tail entry, round 76

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 75 certifies tracer-entry rows 8.3--8.10.  The next live NEMO call is
`tra_ldf -> traldf_iso_lap` (`stpmlf.F90:547-551`).  Reuse the existing
full-halo stage-22/stage-23 T/S RHS dumps; their adjacent difference isolates
the Redi increment.  During the same production `_nemo_mlf_step`, capture the
two `nemo_iso_lap_tracer_tendency_latlon_cgrid` calls and require their inputs
to equal `T_before` and `S_before` on every registered wet element.

Score temperature then salinity at the accumulating `1e-12` column bar over
the exact 9,920 T-column census.  Identity must pass; sign reversal,
meridional roll, and one wet-point perturbation must fail.  Stop at the first
ordered failure.  Only if both pass may the walk bind the already frozen
literal tracer-ZDF solver receipt and proceed to tracer Asselin.

Frozen round-75 receipt SHA-256:
`750c40875300ddda48287d84089c8931eaaecc71e8aab6ce7f4e18a0c806edf4`.
