# Split-explicit momentum chain round 40: production ZAD QCO composition

Status: **FROZEN BEFORE IMPLEMENTATION MEASUREMENT**.  The official round-39
retained-input factorial localised row 4 to the inseparable composition of
NEMO call-1 ``ww`` and live ``e3u/e3v(Kmm)``: the joint arm scores
``2.33481e-16/5.98622e-16`` normalized RMS for U/V, while live thickness alone
worsens both components.  Source association is inert at this scale.

## Production change

A single static selector, ``zad_qco_evaluation``, has two legal values:

* ``generic`` retains the existing z-star continuity diagnosis and min-face
  thickness byte-for-byte;
* ``nemo_literal`` constructs the pair as one operation: source-ordered live
  Kmm face thickness from the carried raw NEMO QCO mesh, predicts Kaa SSH from
  the same Kmm transport continuity over the active leapfrog span, and applies
  ``sshwzv.F90:218-227`` bottom-up to obtain the QCO ``ww`` passed together
  with those live face thicknesses to the existing ``dynzad.F90:83-119``
  kernel.

There are deliberately no selectable W-only or H-only production states.  The
selector defaults to ``nemo_literal`` only on ``nemo_dino_kamm`` and
``nemo_dino_kamm_mlf``.  Every other recipe and a flat/default config must
remain ``generic`` and byte-identical.

## Frozen gates

The implementation is admitted only if all of the following hold:

1. config validation rejects unknown literals and rejects ``nemo_literal``
   unless ``vertical_momentum_scheme='nemo_advective'`` and the carried raw
   NEMO ``e3t_0``, ``hu_0/hv_0``, and ``e1e2t/u/v`` operands are present;
2. a deterministic unit mesh proves the live face thickness and Kaa-continuity
   recurrence against a source-loop oracle, including zero surface and bottom
   ``ww`` and a nonzero interior result;
3. a byte pin proves ``generic`` returns exactly the pre-change tendency;
4. a red-capable planted H-only violation substitutes the literal thickness
   while retaining generic ``w`` and must increase row-4 U and V normalized
   RMS relative to the generic control, reproducing round 39's direction;
5. the production day-180 row-4 replay uses the unchanged active-cell
   ``1e-12`` normalized-RMS bar and must be AT BAR for both U and V.  Identity,
   sign, longitude-roll, wet-NaN, and two-bar controls must all fire.

If either component remains above bar, row 4 stays OPEN and the production
change is not described as closing ZAD.  Rows 5-6 and later chains resume only
after row 4 is AT BAR.  Existing round-37/39 receipts and operand hashes are
fail-closed.  CPU/fp64 only; no GPU, MPI, or NEMO rebuild is authorised.
