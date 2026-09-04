# SI3 lane 3b round 13 — WRITE-only header-validity ticket

Date: 2026-09-06  
Tracker: `climate-federation/legoESM#1699`  
State: **CONFIRMED VALID; BRITTLE LITERALS REMOVED**

The reported `l3rea_dump` defect does not reproduce in this lane.  Its write
list is three scalar fields (`t_su`, `qns_ice`, `dqns_ice`), three fields with
`nlay_i` values (`t_i`, `sz_i`, `e_i`), and two fields with `nlay_s` values
(`t_s`, `e_s`).  Its derived count is therefore
`(3 + 3*nlay_i + 2*nlay_s)*npti`, which is 18—not 43—for this case's
`npti=1,nlay_i=3,nlay_s=3`.  Every retained C1D reassociation stream is exactly
192 bytes: 16-byte magic, eight 4-byte header integers, and 18 binary64
values.  The gate's planted claim of 43 is rejected at that record.

The old header was true but fragile.  The config-local writer and committed
WRITE-only source now emit the formula above.  All seven ZDF operand counts
and the DH/DH-remap counts were likewise replaced by formulas in `nlay_i` and
`nlay_s`.  No model field is assigned by these changes.  Existing writers that
already use `SIZE(z)` (bulk and SSM) remain unchanged.  The thermodynamic and
exchange streams carry dimensions and payload selectors rather than a literal
count; the schema gate derives their record sizes from those dimensions.

## Complete writer audit

| stream | header contract | result on retained oracle |
|---|---|---|
| `oracle_si3_thd_frames.bin` | derived from `jpi,jpj,jpl,nlay_i,nlay_s,npti,payload` | VALID, 70,080 records |
| `oracle_si3_exchange_frames.bin` | derived from `jpi,jpj,jpl` and A2D/full-array write list | VALID, 8,760 records |
| `oracle_si3_zdf_inputs.bin` | `(13+nlay_s)*npti` | VALID, 8,760 records |
| `oracle_si3_reassoc_operands.bin` | `(3+3*nlay_i+2*nlay_s)*npti` | VALID, 1 record; 18 values |
| `oracle_si3_zdf_operands.bin` | frame-specific formulas in layer counts | VALID, 26 records |
| `oracle_si3_dh_operands.bin` | `16+2*nlay_i+2*nlay_s` | VALID, 18 records |
| `oracle_si3_dh_remap_operands.bin` | `9+7*nlay_s` | VALID, 2 records |
| `oracle_si3_bulk_operands.bin` | `SIZE(z)`, `SIZE(z1)`, or `SIZE(z2)` | VALID, 26,280 records |
| `oracle_rung36_ssm_frames.bin` | `SIZE(z)` | VALID on retained 36-step construction, 72 records |

The full scalar-math V2 validation JSON is outside git at
`c1d_omip_l3_sasice_scalarmath_v2_a/header_validity_round13.json`, SHA-256
`370d15b98afcf749cdc9ad5f5cb8324590154a83d4815c85c7eaa58966387ce3`.
The retained rung-3.6 construction audit is SHA-256
`df0bc827015cad73aa9641713e256b94c433a2d6c7ffa7bb319dd980e1f1ecbe`.
Because every extant header agrees with the bytes actually written, no oracle
stream needs rerunning or repinning for header correctness.  The forthcoming
10 m oracle will be built from the formula-based writers.

The schema gate is `nemo_si3_stream_header_gate.py`.  Its focused suite reports
`2 passed`; the second test plants a 43-value declaration above an 18-value
write and confirms a row-specific nonzero failure.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| audit every config-local WRITE-only stream header | ASKED | completed above |
| derive counts rather than retain layer-specific literals | ASKED | completed for reassociation, ZDF, DH, and remap writers |
| rerun an oracle whose retained header is malformed | ASKED, conditional | condition false; no retained stream is malformed |
| represent the reported 43-value diagnosis as measured C1D fact | UNASKED | not done; byte and source audits refute it here |
| modify a shipped NEMO source/configuration or delete a retained stream | UNASKED | not done |
