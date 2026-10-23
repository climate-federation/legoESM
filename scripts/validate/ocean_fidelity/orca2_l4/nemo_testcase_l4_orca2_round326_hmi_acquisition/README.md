# ORCA2 round-326 acquisition — SI3 mean ice thickness as TKE reads it

Records, for kt 1..10 on both ranks, the arrays tke_avn's nn_mxlice = 2
surface mixing length reads (`zdftke.f90:621-622`): hm_i, fr_i, taum and
tmask(:,:,1), inner domain, zero halo. The hook sits in zdf_tke just before
tke_avn; the writer only copies and writes.

Record `NEMO_L4_HMI_1`: 16-byte magic, 12 header integers (version, kt,
rank, nimpp, njmpp, jpi, jpj, real bits, array count, nn_mxlice, ln_mxl0,
nn_fsbc), then per array a 16-byte name, 7 integers (rank, three extents,
three origins) and fp64 payload in Fortran order. File
`oracle_tke_hmi_kt<kt:08>_r<rank:04>.bin`.

Admission: restarts, surface frames and step-entry frames byte-identical to
`phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`;
five gate plants fire; the gate reports fr_i/taum against the round-5 surface
frames and the even-step hm_i identity.

Files: `zdftke_round326_writer.F90`, `zdftke_round326.patch` (additions only,
against `src/OCE/ZDF/zdftke.F90`), `run.sh` (`--preflight-only`, `--run`,
`--admit-existing`), gate `../nemo_testcase_l4_orca2_round326_hmi_gate.py`,
test `tests/ocean/fidelity/test_orca2_round326_hmi_gate.py`.
