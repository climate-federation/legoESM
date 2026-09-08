# Split-explicit / momentum chain: first Redi face production replay, round 80

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Receipt `/tmp/dino_split_explicit_momentum_chain_round80.json`, SHA-256
`e0803ac9f844ec77c843885247b6ad0809b33f02894b88376690ec870f9b8493`,
validly stops as `REDI_ZFU_T_POSTFIX_REGRESSION`. The first implementation
reduced temperature-zfu maximum normalized error from `3.302117e-3` to
`9.858694e-7` (99.97%) but did not reproduce NEMO's live face thickness: its
direct maximum normalized error was `1.2047005e-7`. The gate therefore
prevented a false promotion.

Existing streams localize the miss to time level, not reference geometry or
QCO arithmetic. In `mesh_mask.nc`, wet `e3t_0` and `e3u_0` are bit-exact.
Replaying the shared QCO builder from the bridged step-entry `state.eta`
reproduces all 336,338 wet values of retained
`fct_entry_dump_e3u.bin` bit-exactly (maximum absolute error zero); replaying
from `eta_before` does not. During production, however, traldf was called with
`state_new.eta` after the dynamics update. NEMO calls `tra_ldf` at
`stpmlf.F90:548` with `Kmm=Nnn`, and `traldf_iso_scheme.h90:73-74` explicitly
indexes `e3u/e3v(...,Kmm)`. The corrected production boundary therefore
carries step-entry Nnn eta across the dynamics update solely for these flux
faces. No new NEMO run or writer was needed.

