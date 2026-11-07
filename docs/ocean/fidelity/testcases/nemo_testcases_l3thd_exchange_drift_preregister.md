# Lane 3b exchange-stream drift preregistration

Date: 2026-09-04  
Tracker: climate-federation/legoESM#1699  
Status: PREREGISTERED; no byte-difference values had been inspected when this
document was written.

## Question and source-defined record

Why do retained `oracle_si3_exchange_frames.bin` files have different SHA-256
digests despite all full-year files having size 18,571,200 bytes?  The
config-local WRITE-only writer is
`nemo502_si3thd_MY_SRC/icestp.F90:245-268`.  It writes, per ice step, a 16-byte
magic, six native 32-bit integers, and the arrays in lines 257-264.  The reader
derives the C1D payload as 260 fp64 values
(`nemo_si3thd_oracle_gate.py:214-246`), hence a 2,120-byte record and 8,760
records.

The retained full-year roots are `*_gate`, `*_scope_gate2`,
`*_phase2_inputs`, `*_phase2b_oracle`, `*_phase2b_replay`,
`*_phase4_operands`, `*_phase6_dh_operands`, and
`*_phase6_dh_operands_writeonly` under
`/data/abyssal/dbalwada/nemo-testcases-l3/`.  Their pre-analysis inventory
digests are recorded by the eventual ticket; the four receipt-pinned points
are `998f4790...`, `7f22ca91...`, `6eefecd0...`, and `f02d1edd...`.

## Registered measurement

The committed audit will:

1. parse every record using the writer's exact array shapes and Fortran order;
2. enumerate every differing byte and map it to step, header/payload, named
   field, category, and allocated `(i,j)` element;
3. compare the active C1D scalar/member separately from every allocated halo;
4. report finite-value, bit-pattern, signed-zero, and non-finite classes; and
5. correlate any varying field with source assignments and with the immutable
   thermodynamics frames, ZDF inputs, `ocean.output`, and restart hashes.

Classification is fixed in advance:

- **uninitialized/halo:** every changed byte lies in an allocated element that
  is outside the active C1D scalar/member, source inspection shows that element
  is not defined before the write, and no active element changes;
- **timestamp/metadata:** changes are confined to header or an explicitly
  registered metadata field;
- **real field:** any active scalar/member differs, regardless of magnitude;
- **mixed:** more than one preceding class occurs.

The audit is **CONFIRMED** only if it accounts for every differing byte in every
retained stream.  Otherwise the result is **PLAUSIBLE** and rung 3.5b remains
locked.

## Registered remedy and controls

If the cause is uninitialized or inactive allocated storage, the config-local
writer will serialize an explicitly zero-initialized payload and copy only the
registered active elements into it.  Model arrays will remain untouched.  Two
fresh, independently rebuilt, full-year CPU/no-MPI reruns must then have
byte-identical streams and equal SHA-256 digests.  A planted one-bit change in
an active scalar and a planted one-bit change in an inactive serialized slot
must each make the audit exit nonzero.

If a real active field changed, no writer normalization is allowed.  The ticket
will name the field, first step, and oracle phase transition, and will rerun
every downstream check that actually consumed that stream.  Rung 3.5b is
locked until that is complete.

## Decisions

| Choice | State | Disposition |
|---|---|---|
| Diagnose the exchange drift before bulk-flux certification | ASKED | Registered above. |
| Initialize only the serialized payload, never NEMO model state | ASKED | Required if inactive storage is the owner. |
| Rebuild and rerun twice after a writer fix | ASKED | Exact byte identity is the acceptance condition. |
| Treat equal file size as evidence of equal content | UNASKED | Rejected. |
| Modify shipped NEMO source/configuration | UNASKED | Forbidden; config-local copy only. |
| Certify rung 3.5b before this ticket closes | UNASKED | Forbidden by dispatch ordering. |

## Flagged for future deletion

Nothing is deleted.  All prior run roots and streams remain retained evidence.
