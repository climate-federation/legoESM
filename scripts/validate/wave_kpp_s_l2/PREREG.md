# Preregistered Wave / KPP S-L2 comparison

## Claim and controls (before integration)

A runs the unmodified KPP in upstream revision named in config.json, with
Langmuir disabled. B runs the current KPP, with the weak-buoyancy numerical
correction and Langmuir disabled. C runs that same corrected KPP with Wave
surface-Stokes diagnostics and the published VR12-AL enhancement enabled.
The main comparison is A to C; A to B attributes the numerical correction;
B to C attributes the wave / Langmuir feedback. A is NOT a switch on the
modified implementation. All five columns use identical initial data,
forcing, state precision, timestep, vertical grid and baseline KPP settings.

Hypothesis: prescribed waves increase mixing and tend to deepen the mixed
layer relative to B under weak cooling. Report the complete 48-hour histories
and all three amplitudes, even if the hypothesis is refuted. Temperature
response has no preassigned sign. A to C need not have the same sign as B to C:
the original weak-buoyancy floor is itself a confound. No coefficients, floors,
windows or diagnostics will be tuned after seeing results.

Formula and monochromatic-Stokes comparisons must meet config.json tolerances.
Heat, salt and horizontal-momentum residuals must meet its budget tolerance;
heat is normalized by integrated applied cooling, salt by initial salt content,
and momentum by cumulative magnitudes of wind and discrete Coriolis impulses.
Separate nonuniform-salt unit tests make the zero salt-flux check non-vacuous.
For trend interpretation report signed changes and compare MLD changes to the
0.64-m vertical grid spacing; this is a resolution annotation, not a tuned
physics acceptance threshold. Failure/refutation is retained in the report.

## Reference conditions and deliberate limits

- Li & Fox-Kemper (2017), JPO 47, 2863-2886,
  doi:10.1175/JPO-D-17-0085.1, Table 1 S-L2, Eq. (4): aligned monochromatic
  waves, k=0.1 m^-1, A=0.40/0.53/0.80 m, weak-wind group, cooling 5 W/m2.
- Li et al. (2016), Ocean Modelling 103, 145-160,
  doi:10.1016/j.ocemod.2015.07.020, Table 2 / Eq. (12): VR12-AL constants
  c1=3.1 and c2=5.4. Wind stress 0.036 Pa is the author's LF17 weak-wind
  runner value; that runner also supplies 48 hours, 60-s steps, 15-min output,
  256 cells over 163.84 m. No LES turbulence is simulated here.
- Initial profile files are transcribed verbatim from qingli411/gotmwork,
  revision 81edc1b4b297acc5ce388205ad51fc33984ac907,
  data/Idealized_Tests_LF17/{tprof_file.dat,sprof_file.dat}. The upper 42 m
  are initially neutral, salinity 35. Linear interpolation to cell centres.
- Use the existing linear EOS with alpha=2e-4 K^-1 and the model's canonical
  density, heat capacity and gravity consistently in all arms. These constants
  and the source's rounded profile are recorded; no exact LES reproduction is
  claimed. No additional convection, restoring, solar, lateral or bottom-drag
  schemes; SCM extra background K/A are zero (KPP already includes backgrounds).
- Prescribed monochromatic spectra are homogeneous on an 8x4 global grid,
  with 4 directions and 4 logarithmic frequencies (ratio 1.1). The second
  frequency is exactly the finite-depth dispersion frequency of the specified
  k under the Stokes diagnostic's documented gravity. Place all variance
  A^2/2 in that bin and one wind-aligned direction. No learned weights or
  source-term calibration is involved. The scientific column holds this
  prescribed sea state fixed; separate real-driver tests advance Wave.
- Compare the production surface diagnostic against the independent
  finite-depth analytic Stokes expression, also recording the deep-water
  value and the paper's first-cell-centre value at z=-0.32 m. The latter is
  not silently substituted for the coupled surface input.
- MLD uses the existing density threshold (0.03 kg/m3 from 10 m). Also record
  the depth of maximum N2, labelled separately from model KPP depth and MLD.
  Reduced KPP shape, smoothed BL diagnosis, missing full Stokes-profile and
  Coriolis-Stokes treatment are retained limitations. Numerical LES output
  is not available in this lane; compare physical trends, not LES error scores.

## Execution and artifacts

Export upstream packages and src with git archive into a separate directory;
workers run in separate processes with import-origin and KPP-blob checks.
The current source must be committed and tracked-clean before full integration.
Use CPU with JAX_ENABLE_X64=1 AND PrecisionPolicy.fp64(). Reuse OceanColumnModel
and its backward-Euler diffusion / Crank-Nicolson Coriolis operators. JIT-wrap
the existing tendency, diffusion and rotation calls without changing ordering.

The existing validate_ocean_scm.py entrypoint dispatches the wave_langmuir case.
Outputs: per-case NPZ histories, JSON configuration/provenance/budget reports,
CSV time series, and PNG/PDF plots. Relative module paths, source revisions,
source/input checksums, canonical constants and active KPP fields are stamped.
Local absolute paths must not appear in committed evidence or PR text.

## Instrument handling amendment after the first A run

The first baseline run used instrument revision d02a38ff99. It completed but
exceeded the unchanged 1e-10 cooling-normalized heat gate, so the initial
runner stopped before collecting B/C. The instrument now collects all finite,
source-verified cases and plots them while preserving each strict budget
failure and an overall nonzero exit status. Fatal import/provenance/nonfinite
errors are separate and prevent a complete comparison plot. Existing evidence
is never overwritten. This changes reporting only: no model, input, statistic,
tolerance, time window or interpretation threshold is changed. Any diagnosis
of accumulated roundoff is post-hoc and does not convert a gate failure to PASS.
