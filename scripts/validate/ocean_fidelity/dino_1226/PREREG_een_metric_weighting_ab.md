# PRE-REGISTRATION — the EEN transport metric weighting: the fix, and its one 90-day A/B

Written 2026-08-23, BEFORE the option was implemented and BEFORE any twin was
run. Issue #1455. Parent: `PREREG_een_e3f_mechanism.md` and its result
(commit 0fc77f3aa).

## What earned this

The registered `e3f` mechanism was REFUTED as the owner (it closes 0%). The
term-by-term decomposition that ran in the same pass exonerated every INPUT to
the EEN triad — F-point thickness, vertex Coriolis, relative vorticity, face
mass fluxes, and all four together — and located the mismatch in the ASSEMBLY:

* NEMO's `vor_een` weights the meridional transport by `e1v`, the V-face zonal
  width (`dynvor.F90:791-792`), and divides the assembled u-tendency by `e1u`
  (`dynvor.F90:804`).
* legoESM's Arakawa-Lamb-81 triad (`pv_flux_al81_partial_cell`) uses the bare
  `h_v·v` and neither factor.
* Supplying the weighting closes **98.7%** of the wall-row mismatch
  (3.10e-10 → 4.1e-12 m/s²).
* NEMO's assembly transcribed verbatim reproduces NEMO's own dumped tendency to
  7.7e-15 of the field RMS, so the decomposition is exhaustive, not merely
  ranked.

**This registration is CONDITIONAL on both adversarial reviews not overturning
that measurement.** If either reviewer refutes the operator-level claim, this
document is void and no run happens.

## The change

A new selectable option on the vector-invariant vorticity flux, **default off**,
that applies NEMO's metric weighting to the transport inside the AL81/EEN triad.
Recipe doctrine applies: an unknown value raises; the option is threaded through
every path that builds the same triad (the 3-D tendency and the barotropic
solver's own EEN inputs, if the latter carries the same gap); and the DINO
oracle card selects it while every other recipe keeps the current behaviour
bit-identical.

A direct test must be shown RED before the fix and GREEN after — a
non-vacuous test, not one that merely runs.

## The A/B

ONE 90-day twin from NEMO's developed day-180 restart, the standard harness
(`kamm_twin_90d.py --save-3d`), scored by the two existing gates. The baseline
arm is RE-RUN at the same HEAD rather than reusing
`results/dino_1455_visc_ablation/twin90_uv027.npz` (built at an older commit),
so the two arms differ ONLY in the new option. Each arm costs about 214 s.

**The transport gate, registered:** `wall_visc_ablation_gap.py` reports `G4`,
the day-90 southern-basin zonal-transport gap summed over wall rows 1-4. The
baseline value is **−0.288 Sv** (67.7% of the −0.426 Sv basin gap). CONFIRM
requires **|G4| to shrink by more than 40%**, i.e. |G4| < 0.173 Sv.

**The joint gate, registered — a transport gain bought with density is a
compensating error, not a fix.** `acceptance_gate_90d.py --level 5` scores four
metrics against the #1492 noise floors (ACC 0.091 Sv, upper contrast 1.1e-4,
deep contrast 4.5e-5, surface sigma 9.5e-5 kg/m³). The channel is the campaign's
crown jewel. Registered:

* **No metric may move past its floor** in the degrading direction relative to
  the baseline arm. Any that does is a LOUD finding, reported as such, and the
  fix is NOT recommended for the card on the transport number alone.
* The transport and density numbers are reported **together, in one table**, and
  the verdict is read jointly. The lateral-viscosity ablation already produced
  the counter-example this clause exists for: the circumpolar transport error
  fell monotonically (0.382 → 0.242 → 0.069 Sv) while the density contrast
  behind it degraded monotonically (2.4e-4 → 1.4e-3 kg/m³), which is two errors
  cancelling and not a better model.

**Falsification.** If |G4| does not shrink by 40%, the operator-level mismatch
was real (it is measured, closed to 1.3%) but is NOT what carries the basin
deficit, and the campaign learns that the wall-row tendency error and the
wall-row transport error are different objects. That is a publishable negative
and it is registered as such here, in advance.

## What this does not test

Nothing about the year-long deficit; 90 days is the controlled window with an
established noise floor. Nothing about the `e3f` difference, which is real,
measured at 1.47e-4 relative at wall vertices, and inert at 0.010× the
tendency bar — it is logged as transcription debt, not fixed here.

Nothing about any outcome is known at the time of writing.

---

# AMENDMENT, written after the dual review and BEFORE either arm was launched

Both reviewers reported. Neither overturned the operator-level measurement —
both confirmed it, and one measured the energy-norm property behind it. But
two things in the parent registration are now void and are corrected here
rather than quietly reinterpreted.

## The reduction the bars were written on is not the one they name

Every wall-row number this campaign has published, including the −0.288 Sv
transport gap's companion tendency numbers, was called "thickness-weighted".
It is not: the reduction weights by the wet MASK, an unweighted mean over
levels, and DINO's layers span 10.14 m to 545.20 m. Rescored with real
thickness the wall-row tendency mismatch is 12.4× smaller and **inverts sign
on three of the four wall rows**.

Consequence for this registration: **the directional prediction is WITHDRAWN.**
The parent registered "|G4| must shrink by more than 40%". That direction was
derived from a reading of the wall rows that the corrected reduction does not
support. No directional prediction replaces it.

## What is registered instead, before the arms run

The A/B is now a **two-sided measurement with a one-sided ship gate**:

* **Measured, no prediction:** `G4`, the day-90 wall-row transport gap, and
  the whole-basin gap. Both directions are reported. A shrink is evidence the
  operator error carried the deficit; a growth is evidence it did not and that
  the campaign's wall-row narrative was an artifact of the reduction. Either
  is a result. Nothing is registered as CONFIRM on this metric alone.
* **The ship gate is UNCHANGED and remains one-sided:** the four
  acceptance-gate metrics (ACC 0.091 Sv, upper contrast 1.1e-4, deep contrast
  4.5e-5, surface sigma 9.5e-5 kg/m³) must not move past their floors in the
  degrading direction. The channel is the crown jewel. Transport and density
  are read jointly, in one table, exactly as before.
* **Scope correction:** the metric weighting closes 86% of the FAR-INTERIOR
  operator mismatch as well as 96-98% of the wall's. It is a global fidelity
  correction, so the basin-wide and circumpolar metrics are as relevant as the
  wall rows, not a side check.

## The arms

Both re-run at the same HEAD; they differ in ONE config field, set by
`DINO_EEN_METRIC` (`off` | `nemo`, unknown raises). The card itself keeps the
option OFF, so the baseline arm is the shipped card verbatim.

Nothing about either outcome is known at the time of writing.
