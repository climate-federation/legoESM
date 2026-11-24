# Current-main Wave / KPP comparison — 2026-11-24

This protocol is committed before running the refreshed comparison. It uses
`current_main.json` and the existing instrument, profile files and S-L2
reference conditions in `PREREG.md`. The original configuration and evidence
remain unchanged as a historical comparison.

## Sources and controls

- A: unmodified upstream main `5ffaa759bf543e7f598a7a6b1bfbace139aff509`,
  with KPP selected and Langmuir disabled. Use upstream's default
  `actual_buoyancy_flux=False`, adiabatic N², column velocity scale and
  disabled gradient surrogates. The pinned KPP blob is in the configuration.
- B: the current committed PR implementation, including that same main,
  with KPP selected and Langmuir disabled. Its default
  `actual_buoyancy_flux=True` is the intended no-Wave PR behavior.
- C: B with prescribed Wave-derived surface Stokes and VR12-AL enabled at
  amplitudes 0.40, 0.53 and 0.80 m. All other settings are identical.

A must import the exported upstream source, not simulate it with a switch
on PR code. Every imported model module is checked against its selected Git
tree. Save the complete active KPP configuration for each arm.

## Fixed quantities and interpretation

Keep the existing 48-hour duration, 60-s step, 256 uniform cells, forcing,
initial profiles, linear EOS, constants and diagnostic definitions. Keep all
original formula/Stokes tolerances and the 1e-10 heat/salt/momentum threshold.
No parameter, threshold, sampling window or output is selected after running.

The outputs are the same complete histories, final MLD / N²-peak depths,
diffusivity and temperature profiles and independently accumulated budgets.
Report signed A→B, B→C and A→C changes at all three amplitudes. Prescribed
waves are expected to increase diffusivity and tend to deepen the density
MLD relative to B; a decrease or non-monotonic response refutes that trend
for this setup. Report actual results in either event. MLD shifts below
0.64 m remain below one model layer, regardless of interpolation precision.

Budget PASS requires all three residuals to meet the unchanged 1e-10 gate;
any excess remains a failed verification with a nonzero suite exit status.
No numerical agreement with LES, global skill or full wave-atmosphere
feedback is claimed by this column experiment.

## Reporting and execution

Run serially on CPU in fp64 with two CPU cores, one BLAS thread and a
4-GiB child-memory / 2-GiB available-memory guard. Keep only one JAX process
active. Use a new `kpp_s_l2_20261124` output directory; never overwrite the
original artifacts. Source exports are local execution inputs only.

The refreshed figures label A as main / Wave off, B as PR / Wave off, and
C by prescribed amplitude. The formula panel explicitly marks the existing
main enhancement as an option that is disabled by default. Figure captions
and the PR identify the selected main version through the report, rather
than displaying a historical implementation as current upstream.
