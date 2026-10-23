# ORCA2 round 328 preregistration — kt 1 stage-3 `dyn_zdf` source-order record

Frozen 2026-10-23, before constructing or running the acquisition (Codex round).

## Finding that sets this round

- The given-NEMO-entry comparison is exact through kt 1 stage 2.  At stage 3,
  u/v/T/S first depart together; NEMO calls `dyn_zdf` before any tracer call at
  `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:492-555`.
- The admitted records contain completed stage states and the source-ordered
  TKE/vertical-mixing coefficient chain, but no frame at `dyn_zdf` entry, its
  explicit update, barotropic subtraction, top/bottom drag additions, matrix
  construction, or implicit solve.  A statement-level walk therefore cannot
  distinguish an input-coefficient departure from a `dyn_zdf` transcription.
- Per the round contract, this round writes a passive acquisition and stops.
  No NEMO run and no model/card/configuration change are permitted here.

## Acquisition predictions

The acquisition re-runs the admitted round-326 trajectory with additions-only,
write-only hooks.  It records kt 1 stage 3 on both ranks in source order.

- A1 PASSIVITY: NEMO reports STOP 0; all inherited ocean/ice restarts, all 20
  step-entry frames, all 20 surface frames, and all existing round-326 hm_i
  frames are byte-identical to the admitted round-326 record.  Falsifier: any
  inherited byte differs; admission refuses.
- A2 PROGRAM IDENTITY: every preprocessed source and code-bearing object equals
  the round-326 producer after normalising only the target configuration path,
  except for the additions-only `dynzdf` insertions and the new writer module.
  Falsifier: a removed/replaced source line, changed code byte, flag, compiler,
  architecture file, CPP key, or non-observer object; preflight/run refuses.
- A3 SOURCE ORDER: each rank emits exactly one kt 1 stage-3 record with
  Kbb/Kmm/Krhs/Kaa = 1/2/3/3 and complete owned rows for: `avm` input; explicit
  update; barotropic subtraction; bottom-drag RHS; top-drag RHS; viscosity
  matrix; bottom-drag diagonal; top-drag diagonal; U solve; V solve.  Falsifier:
  wrong header/order/count/shape or a missing rank; admission refuses.
- A4 ENDPOINT CONSISTENCY: the record's solve output, after NEMO's ordinary
  post-`dyn_zdf` barotropic correction, reproduces the already admitted kt 1
  stage-3 u/v frame bit for bit on both ranks.  Falsifier: any unequal wet
  point; admission refuses.
- A5 ACTIVE CONTROLS: `avm` is finite and nonzero on a wet interface; the
  recorded bottom and top drag coefficients each affect at least one wet
  matrix row; one-ULP plants in the `avm`, top-drag, bottom-drag, and solve
  payloads each make admission fail.  Falsifier: an inactive channel or a
  plant that stays green.

## Resume after acquisition

The next round admits the record, then replays these boundaries offline from
the passive completed stage-2 state.  It names the first non-bit statement,
its cell subset, NEMO and legoESM lines, and certified-rung coverage.  A model
change remains forbidden unless one cited statement passes Decision 122 on all
seven pairs, both labels, with Decision-150 drag enabled.
