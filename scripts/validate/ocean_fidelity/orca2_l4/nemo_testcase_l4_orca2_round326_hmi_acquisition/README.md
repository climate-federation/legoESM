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

Same program: the build is fresh from today's NEMO tree, then refused unless
it is round 5's program plus the observer: build config (arch, cpp keys,
bld.cfg; config name substituted) identical, every preprocessed source
byte-identical to round 5's `BLD/`, and every object identical section by
section except `zdftke` (its only difference is the patch's inserted lines)
and the new writer. Objects are compared with the config root normalised:
gfortran stores each source's absolute path in runtime-error strings, so a
build under another config name differs only there (string sections re-pad,
string-label symbol offsets shift); code, data and relocations stay
byte-exact. Every differing file is listed. `object_path_probe.sh OUT`
recompiles all 425 round-5 objects at round 5's path (all byte-identical)
and under a renamed config root (311 differ in bytes, 0 after
normalisation; an -O2 build and a flipped code byte refused). Preflight shows
today's cpp and toolchain reproduce round 5's `zdftke.f90`/`zdftke.o` byte for
byte (a -O2 plant differs) and that the check refuses a planted unit.

Admission: restarts, surface frames and step-entry frames byte-identical to
`phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`;
nine gate plants fire; then the gate REFUSES on any preregistered validity
failure: hm_i negative, zero on every ice-covered wet point, or >= 20 m;
fr_i/taum unequal to the round-5 surface frames; even-step hm_i unequal to
the previous odd step's.

Outside the evidence and target directories, makenemo also rewrites the
shared `cfgs/work_cfgs.txt` (deletes, then appends, the target's own line);
the run refuses if any other line changed. A failed `--run` removes the
target config and run directory it created (the work_cfgs line stays;
makenemo replaces it on the next run), so a re-run starts clean. Scratch
lives beside the evidence (never /tmp) and is removed on exit.

Safety: refuses below 150 GB available memory (`free -g`) in preflight and
just before mpirun; mpirun is killed after 900 s (round 5 took 13 s).
The default mode is the read-only preflight; `--run` must be explicit.

Files: `zdftke_round326_writer.F90`, `zdftke_round326.patch` (additions only,
against `src/OCE/ZDF/zdftke.F90`), `round5_ocean.output.excerpt` (verbatim
lines of round 5's ocean.output with its SHA-256; the log checks run on it),
`run.sh` (`--preflight-only` default, `--run`, `--admit-existing`),
`object_path_probe.sh` (gfortran only; no makenemo, no NEMO), gate
`../nemo_testcase_l4_orca2_round326_hmi_gate.py`, test
`tests/ocean/fidelity/test_orca2_round326_hmi_gate.py`.
