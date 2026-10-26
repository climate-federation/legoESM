# NEMO testcase Lane 4 — ORCA2 Phase-2z handoff receipt

Date: 2026-09-05

Parent: `392583abe1f9` (phase-2y resolved-namelist gate)

Status: **CONFIRMED STAGED; MPI NOT EXECUTED.** Two new WRITE-only sea-ice
thermodynamics frames added, a scalar-math binary built in an **isolated**
config (the shared build tree was explicitly left untouched per the
coordinator's instruction — see phase-2y receipt §5), and two twin run
directories staged. No legoESM numerics changed. No MPI/NEMO run was
launched from the sandbox.

## 1. Isolation: a new config, not a reset of the shared tree

The ORCA1-ice binary is actually built under
`/tmp/nemo-orca2-phase2p/cfgs/ORCA2_ORCA1ICE_OMIP_L4/` (not the unrelated
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/` tree —
the two share a naming convention but are different trees). Two independent
peer reviews (`si3thd-r8-review`, read-only; `scalar-math-control`,
read-only) confirmed: (a) `icethd.F90`/`icedyn_rhg_evp.F90`/
`icedyn_adv_pra.F90` in that live `MY_SRC/` are byte-identical to the
phase-2x-receipt-cited applied hashes, and no file there has an mtime newer
than phase-2x's own build (11:37:58); (b) several unrelated dycore/EOS files
also sit there with recent mtimes and must not be touched or attributed to
this lane. Per the coordinator's explicit instruction, nothing in that
directory was reset, deleted, or overwritten.

Instead, a **new** config was created via `makenemo -r
ORCA2_ORCA1ICE_OMIP_L4 -n ORCA2_ORCA1ICE_OMIP_L4_P2Z -d "OCE TOP ICE NST
ABL" -m conda-scalarmath`, which copies (not moves) the reference's `MY_SRC/`
into a fresh `ORCA2_ORCA1ICE_OMIP_L4_P2Z/MY_SRC/`. Before editing, the three
ice-dynamics files copied into it were diffed against the phase-2x receipt's
cited applied hashes and found identical:

| file | phase-2x receipt applied SHA-256 | copied-into-P2Z SHA-256 (pre-edit) |
|---|---|---|
| `icethd.F90` | `b66823d02f35e60c9fa315c73ba9d51235b5bb88aa45247d0ed7142cb1d64bdf` | identical |
| `icedyn_rhg_evp.F90` | `bf61835a80faaa38a49c49644a11a16fb45be1049caed8aa55cb224f62f5d7b1` | identical |
| `icedyn_adv_pra.F90` | `0ff0fc32c0845f23e093a2ad7957da89a3eef13ab1dc741005edc30e0785aa24` | identical |

Only `icethd.F90` and `icesbc.F90` were then edited, for the two requested
frames below. Nothing else in the 18-file `MY_SRC/` was touched, so the
rebuild differs from the phase-2x binary by exactly these two frames (Rule
7, one variable at a time).

## 2. Frame (a): `POST_FRAZIL_PRE_ZDF_1D`

**CONFIRMED added.** New subroutine
`l4_dump_post_frazil_pre_zdf_1d(kt, kl)` in `icethd.F90`, called immediately
after the existing `CALL l3zin_dump(kt, jl)` (itself immediately after `CALL
ice_thd_1d2d(jl,1)` — shipped-numbering `icethd.F90:140`, this instrumented
copy's line 150 — and before the `dh_s_tot(1:npti)=0._wp` init block at
shipped-numbering `icethd.F90:143`). Writes, to a new stream
`oracle_l4_post_frazil_pre_zdf_1d.bin`:

- magic `NEMO_L4_PFZ1D_1`, header `(version, kt, kl, jpi, jpj, jpl, nlay_i,
  nlay_s, npti, storage_size)`;
- `nptidx(1:npti)` (global packed-cell index, `ice1D.F90:30`, `(jj-1)*jpi+ji`
  per column) — **blocking**, per the reviewer's own read of the existing
  format: without it the packed slots cannot be mapped back to columns;
- `a_i_1d, h_i_1d, h_s_1d, t_su_1d, s_i_1d, oa_i_1d` (`ice1D.F90:103-123`,
  each `(jpij)`);
- `e_i_1d(1:npti,1:nlay_i), sz_i_1d(1:npti,1:nlay_i)`
  (`ice1D.F90:133,135`, each `(jpij,nlay_i)`);
- `e_s_1d(1:npti,1:nlay_s), t_s_1d(1:npti,1:nlay_s)`
  (`ice1D.F90:131,136`, each `(jpij,nlay_s)`).

All ten requested fields plus `nptidx` are present. This is a **genuinely
new time point**, not a duplicate of the existing `l3thd_dump_1d` stages:
seven of the ten field names are shared with `l3thd_dump_1d`, but every
`l3thd_dump_1d` call fires only **after** `ice_thd_zdf` or later stages —
never at the pre-physics point this frame targets (verified by reading the
call sequence directly, not by trusting the field-name overlap; see the
oracle-fidelity Rule 1d time-level trap this avoids).

## 3. Frame (b): SI3 bulk boundary, full domain

**CONFIRMED added, as a companion, not a replacement.** The existing
`oracle_si3_bulk_operands.bin` stream (`l3bulk_dump_tau`/`l3bulk_dump_flux`
in `icesbc.F90`, part of the frozen 90-stream Phase-1 inventory) already
captures every requested input and output **at one column only**
(`ji=Nis0, jj=Njs0`, the certified C1D probe). Its byte layout is
load-bearing for every existing admission gate and was left untouched.

A new, separate stream `oracle_l4_bulk_boundary_2d.bin` (magic
`NEMO_L4BULK2D_1`) adds the **full masked-domain** version, at the same two
call sites, so the certified single-column identity can be re-scored under
real geometry as `si3thd-r8-review` suggested (measured, not assumed — see
§1's independent verification of that claim):

- **stage 0** (`l4_dump_bulk_tau_2d`, called right after the existing `CALL
  l3bulk_dump_tau`, `icesbc.F90:88-91` as executed): INPUTS `theta_air_zt`,
  `q_air_zt`, `sf(jp_slp)%fnow(:,:,1)` [p_surface], `sf(jp_wndi)%fnow(:,:,1)`
  / `sf(jp_wndj)%fnow(:,:,1)` [wind], `rhoa` [rho_air],
  `sf(jp_qlw)%fnow(:,:,1)` [lw_down], `qsr`, `precip`,
  `sf(jp_snow)%fnow(:,:,1)` [snow], `cloud_fra`, `Ch_ice`, `Ce_ice`; OUTPUTS
  `utau_ice`, `vtau_ice` (dummy args of `ice_sbc_tau`, in scope here),
  `wndm_ice`. 16 fields, header carries `(n2d=16, n3d=0)`.
- **stage 1** (`l4_dump_bulk_flux_2d`, called right after `CALL
  ice_flx_other`, `icesbc.F90:200-201` as executed): 5 per-category
  `(jpi,jpj,jpl)` OUTPUTS `qla_ice`, `dqla_ice`, `devap_ice`, `alb_ice`,
  `qevap_ice` (`sbc_ice.F90:44,45,48,63,68`) plus 11 aggregate `(jpi,jpj)`
  OUTPUTS `emp_ice`, `emp_oce`, `tprecip`, `sprecip`, `qemp_oce`,
  `qemp_ice`, `qsb_ice_bot`, `fhld`, `qlead`, `drag_io`, `frq_m`
  (`sbc_ice.F90:56,70,133-134,66-67`; `ice.F90:140-142,202`;
  `sbc_oce.F90:159`). Header carries `(n2d=11, n3d=5)`, so the payload byte
  count is derivable as `(n2d + n3d*jpl)*jpi*jpj*8`.

`utau_ice`/`vtau_ice` are `INTENT(out)` dummy arguments of `ice_sbc_tau`,
not module-level state, so they are not in scope inside `ice_sbc_flx` where
`ice_flx_other` runs — this is why the frame is split into the same two call
sites the existing single-column probe already uses, rather than one
consolidated write. All INPUTS are timestep-constant forcing fields (loaded
once per step, before either subroutine runs), so reading them at the
tau-stage call site gives the same value they would have at the flux-stage
call site.

Both new subroutines use `l4_canon_2d`/`l4_canon_3d` (the same WRITE-only
masking helper `icethd.F90` already includes), so `icesbc.F90` gained one
new `USE dom_oce, ONLY : nn_hls, tmask, umask, vmask, ssmask, ssumask,
ssvmask, ssfmask` (mirroring `icethd.F90`'s own import for the same reason)
and one new `#include "l4_oracle_canon_subroutines.h90"`.

## 4. Build

**CONFIRMED green, scalar math.** `makenemo -r ORCA2_ORCA1ICE_OMIP_L4 -n
ORCA2_ORCA1ICE_OMIP_L4_P2Z -d "OCE TOP ICE NST ABL" -m conda-scalarmath -j
4` (PATH prepended with the `nemo-build` conda env for its Perl/gfortran/
MPI toolchain) completed in 66 seconds after the isolation steps above.
Binary SHA-256 `93b780ab39e8e84b3fb61e43da4ce84b879b233c38a3e2dacde8f347e3eed654`
(55,870,504 bytes); `nm -D` finds **0 `_ZGV*` symbols**. Build log SHA-256
`1ffd2eb8256322147d2099caf6ae33f5c444b9165919b905a0394b830a2f108b`.

Two encoding notes for whoever reproduces this: (1) `makenemo`'s own
`work_cfgs.txt` bookkeeping fails with `/work_cfgs.txt: Permission denied`
in this environment (a pre-existing, harmless path-resolution quirk,
reproduced identically on a `-j 0` dry run before any edit was made) — pass
`-r`/`-d` explicitly on every invocation rather than relying on
config-name lookup. (2) the system `perl` lacks `Text::Balanced`, needed by
FCM's Fortran interface generator; prepend the `nemo-build` conda env to
`PATH` before invoking `makenemo`.

Patches (unified diff, shipped NEMO 5.0.2 vs the file actually compiled),
hashes, and untouched copies:

| file | shipped base SHA-256 | patch SHA-256 | applied SHA-256 |
|---|---|---|---|
| `icethd.F90` | `8c5242271435a79139a9a4976dde366da129b4213728cd47f4fef0c60431e851` | `aa09159fab1f1e76dc43bf6ad3195812c1ef9923694a2224f6c6f2aaeee8de31` | `3c9912d4930a59f481fd1c52cb1d2c47e55634ccdf66753e0dd66785173fb3db` |
| `icesbc.F90` | `395a2c39ed64d6ec550b69afb9f4d382f18d0899195b941ee60107c1c860ffa5` | `4ef677fea02aeeb93d2c56e6aa75d2c98f2c0e8f06fa72a19defcda447745b9b` | `9d2532b62731f66cc53b903300e2b52b4702cf4fa755b9bf41da2bedf7041896` |

Both patches are `PATCH_REPLAY_EXACT`: applying them to the hash-pinned
shipped source reproduces the file actually compiled, byte for byte.
`icesbc.F90` had **no prior committed patch at all** (the existing
single-column `l3bulk_dump_tau`/`l3bulk_dump_flux`/`oracle_si3_bulk_
operands.bin` instrumentation predates this session and was never captured
in git) — this is the first one, and it now also documents that pre-existing
instrumentation, not only the new frames. Untouched copies of all four files
plus the binary and build log are hashed under
`/data/abyssal/dbalwada/nemo-testcases-l4/phase2z/MANIFEST.sha256`.

## 5. Staged twins

Both directories reuse the phase-2x deck and inputs **unchanged** — the
freshly computed deck-manifest and input-manifest hashes
(`51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9` and
`3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5`) are
**identical** to the values the phase-2x receipt cites, confirming the deck
and forcing inputs are the single unchanged variable; only the binary
differs. Each directory has 19 copied deck files, 40 symlinked inputs (same
targets as phase-2x, under
`/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/`), an
absolute symlink `nemo` to
`/data/abyssal/dbalwada/nemo-testcases-l4/binaries/nemo_ORCA2_ORCA1ICE_OMIP_L4_phase2z.exe`,
and no pre-existing output, restart, or oracle record. `run.sh` is
byte-for-byte the phase-2x launcher with only the `EXPECTED_DIR` and
`EXPECTED_BINARY_SHA256` lines changed (`EXPECTED_DECK_MANIFEST_SHA256`/
`EXPECTED_INPUT_MANIFEST_SHA256` are unchanged in *value*, since deck and
inputs did not change). The launcher's own preflight hash checks (binary,
deck manifest, input manifest, and the full per-file `sha256sum -c`) were
run manually and pass on both arms — **without invoking `mpirun`/`./nemo`**.

Run these one at a time from the user shell, unchanged:

1. `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_orca1ice_phase2z_a_10step_np2/run.sh`
2. `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_orca1ice_phase2z_b_10step_np2/run.sh`

`run.sh` SHA-256 values are respectively
`82382c5a79af99b2a21906f2d09a8a01b098cf65da448f22ca976f082a1ab6e3` and
`8fd6b5cdeb77c3c7832b3deaef4bb9f5f5f319b1c7be22ca4c20e01d684d54bc`.
Do not pin either root until both runs finish, the new streams
(`oracle_l4_post_frazil_pre_zdf_1d.bin`, `oracle_l4_bulk_boundary_2d.bin`)
decode to exact EOF, A/B identity is complete, and the inherited 116-stream
family still matches `VARIANT_ORACLE_ORCA1ICE`.

## ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| add the two named frames | ASKED | CONFIRMED, cited to NEMO source lines |
| capture the bulk boundary at full domain, not just C1D | ASKED (task spec) | CONFIRMED, new stream, existing single-column stream untouched |
| coordinate before touching the shared build tree | self-initiated, then coordinator-confirmed | CONFIRMED: isolated new config used, nothing reset |
| rebuild scalar-math and verify zero `_ZGV` | ASKED | CONFIRMED |
| stage two new twin run directories, phase-2z naming | ASKED | CONFIRMED |
| user-shell MPI execution | ASKED | not performed here; two launchers handed off |
| capture `nptidx` even though not explicitly reachable in the ten-field list | ASKED (task spec calls it out as "blocking") | CONFIRMED included |
| commit a first-ever `icesbc.F90` patch | ASKED-enabling (task named icesbc as an extension target; none existed) | CONFIRMED |
| reset/overwrite the shared MY_SRC tree | explicitly forbidden | not done |
| edit shipped NEMO, delete artifacts, commit multi-MB data, push, run MPI | forbidden | not done |
