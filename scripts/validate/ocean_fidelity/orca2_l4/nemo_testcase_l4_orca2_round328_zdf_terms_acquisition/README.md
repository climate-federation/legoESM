# Round-328 ZDF term-boundary acquisition

Status: `ACQUISITION_NEEDED`. This is an operator-only, additions-only record
request. It must reuse the admitted round-328 two-rank kt-1 trajectory and must
not change any namelist, model option, timestep, decomposition, or input.

Instrument `zdf_phy` at these source-ordered boundaries on both ranks:

1. immediately after `zdf_tke` and the `avt_k`/`avm_k` copies into `avt`/`avm`;
2. immediately after `zdf_ddm`; and
3. immediately after `zdf_iwm`, before `zdf_mxl_turb` and the `lbc_lnk` call.

At every boundary write the native fp64 `avm`, `avt`, and `avs` arrays for
`jk=2:jpkm1`. At boundary 1 also write `en`, `avm_k`, `avt_k`, `rn2`, `r3t(Kmm)`,
`ht_0`, `mbkt`, and `wmask`; at boundary 2 write the DDM increments
`zavft+zavdt`, `zavfs+zavds`, and `MAX(heat,salt)`; at boundary 3 write
`zav_wave`, `zemx_iwm`, and `zReb`. The writer is write-only and may not feed
any payload back into NEMO state.

Admission must require both ranks, a producer commit, exact field order and
extents, finite fp64 payloads, and byte identity of all inherited round-328
oracle frames and terminal restarts. One-ULP plants in each of the three
`avm` boundaries, plus header, field-order, truncation, and producer-stamp
plants, must refuse. The clean record must reproduce the already admitted
final `avm`/`avt`/`avs` bits before it may be used.

The scorer must compare legoESM's carried TKE closure, post-DDM profile, and
post-IWM profile separately. It must report the first unequal boundary and
statement for the zero-based `k=28` family and `k=0` family independently.
Until this record exists, ownership below `zdf_phy` is `UNMEASURED`.

No NEMO executable was run while writing this request.
