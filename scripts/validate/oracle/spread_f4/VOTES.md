# F4 detail votes (me / codex / GLM, majority) — 2026-11-03

| id | question | me | codex | GLM | result |
|---|---|---|---|---|---|
| V1 | perturb constants/params written to the oracle input (ZM 10-28 ruling) vs hold | a | a | a | a: perturb every passed float |
| V2 | several oracle calls per run: same seed each call vs per-call seed | a | b | b | b: seed + 20*call_index (call 0 = seed) |
| V3 | score only port-vs-oracle comparisons (no jit/gap/grad/budget tests) | a | a | a | a |
| V4 | seed runs: spy records true value, returns 0 so TOL asserts don't abort | a | a | a | a |
| V5 | harness in oracle_audit/f4 + committed copy with a direct rewriter test | a | a | a | a |
| V6 | oracle array changes shape/disappears under a seed -> UNMEASURED (not C) | a | a | a | a |
| V7 | discrete flips (zero pattern / ints) under 1 ulp: (a) family-wide no-certify, (b) per comparison | a | a | b | a: family-wide |
| V8 | MG2 prog-N_c C1 fails (fixture built Spack GCC 13.4 elsewhere): (a) UNMEASURED, (b) regenerate fixtures here | a | a | a | a |

Codex findings r1 (8) on the harness: all fixed (array-shaped neutral return, `_rel_flux` metric, argv[1]-only input file,
C3 requires non-empty equal sources, Louis `ri_pin` input pins excluded, C5 requires logged gfortran command flags via
`bash -x`, discrete-change detection for codex #6). #5 (readback is a separate parser, not the oracle's own read) kept as a
stated limitation: C1 bitwise reproduction + every float moved one step are the evidence the oracle consumed the text.
GLM review r1: proxy-vs-internal-round-off concern is the approved method (not changed); pairing divergence now reported.
- V2 implementation: a call whose float input values equal an earlier call's reuses that call's seed index (generator asserts tying two calls on the same values -- MYNN orig vs dump build, Kuo klev_max 20 vs 1 -- keep their tie).
GLM review r2: 9 points. Taken: unsafe-math denylist widened in C5 (#9). Not taken, checked: #1-3 the spy wraps the TEST's
metric, not the oracle/generator (generator self-asserts run unwrapped and stayed live); #4 every driver in these families
reads argv[1] and writes argv[2] (read in each make_fixture), and C1 + all-floats-moved confirm; #5 seed+20k cannot collide
for seeds 0..19 and default_rng(int) streams do not overlap; #7 rel <= 1e-12 = A is the user's rule (perturb.classify).
Codex r2 (2 findings: integer-valued float/bool discrete detection; C5 flags not from the executed binary) fixed; codex r3
(MG2 C5) fixed by tracing the generator's own build.sh call; codex r4: NO FINDINGS. GLM r3 on the last two changes: 6 points,
none changes a class -- every case it names errs toward UNMEASURED/C5-FAIL (conservative), the traced build REPLACES the
generator's call (not run twice) and each mode's log is a fresh directory.

## Round 2 (coordinator follow-up, 2026-11-03)
| id | question | me | codex | GLM | result |
|---|---|---|---|---|---|
| V9 | MG2 prog-N_c: (a) regenerate committed fixtures, (b) local spread-only reference, committed untouched | b | b | b | b |
| V10 | TKE: C-structural, closest (a) WRF MYJ / (b) WRF MYNN | a | a | a | a: MYJ module_bl_myjpbl |
| V11 | clubb_lite: (a) C-structural vs CLUBB release / (b) harness anyway | a | a | a | a |
| V12 | Smagorinsky 1-D shear proxy: (a) C-structural vs WRF km_opt=3 / (b) Lilly transcription | a | a | a | a |
| V13 | gray: (a) Isca two_stream_gray_rad.F90 @ Isca v1.0 5868cfd, FMS stubs, our constants / (b) HEAD | a | a | a | a |
| V14 | Hines: C-structural, closest (a) ICON mo_gw_hines / (b) CCCma/CMAM | a | b | a | a: ICON |
| V15 | Rayleigh: (a) HS94 part vs FMS hs_forcing, sponge C-structural / (b) all C-structural | a | a | a | a |
| V16 | Beres: (a) E3SM gw_convect with the REAL mfcc table, stop if absent / (b) stand-in table | a | a | a | a |
| V17 | mc3d: (a) Tripleclouds vs ecRad + MC C-structural (I3RC) / (b) both C-structural | a | b | a | a |
| V18 | Sundqvist condensation: (a) C-structural vs ccpp zhaocarr_gscond / (b) harness | a | a | a | a |
| V19 | CAM6 sequencing: (a) static call-order audit vs physpkg.F90 / (b) SCAM run | a | a | a | a |
CLUBB implemented-arm harness review: codex NO FINDINGS; GLM 7 points, all already covered -- perturbed calls raise on
err_code; C1 compares the perturbed run's own base trajectory with the committed fixture (catches Fortran state leaking
into the main trajectory) and the seed-0 rerun after seeds 0-19 must be bitwise (catches history pollution between
copies); spread_step runs only inside the recorded-step loop (bigup2 warm-up is outside it); seed outputs are stored as
fresh float64 copies.

## Round 3 (F4 part 3, 2026-11-03)
| id | question | me | codex | GLM | result |
|---|---|---|---|---|---|
| V20 | gray cases: (a) E0/E1/E2 matched subset (pstd_mks = column p_s), 8 tier-2 columns, dry, oracle insolation fed to ours; departure + RCE loop out / (b) include them | a | a | a | a |
| V21 | metric: (a) per-column max\|o-r\|/max\|r\| (Louis convention), tol 1e-10 / (b) pointwise rel | a | a | a | a |
| V22 | constants stefan/grav/cp_air (gray), KAPPA/CP_AIR/GRAV (HS): (a) written into the oracle input (perturbed, V1) / (b) hardcoded in stubs | a | a | a | a |
| V23 | HS: (a) public hs_forcing unmodified, kf = our k_max in 1/s, udt/vdt only, ours sponge_k = 0, 13 tier-2 GWD columns / (b) visibility edit on rayleigh_damping | a | a | a | a |
| V24 | Tripleclouds: (a) C-structural (closed-form grey factor, no region solver; tier-2 ecRad 1.7.2: fsd 0->1 TOA SW -8.2 vs -18.2 W/m2), no harness / (b) ecRad spread harness anyway | a | a | a | a |
| V25 | CLUBB helper tests: (a) no edit, runpy + mapped open() on synthetic anchors + byte-identical rebuild of committed gen_spread_impl.py; analyze: bitwise_diff, load / (b) refactor first | a | a | a | a |
| V26 | land: (a) new commit on feat/f4-oracle-spread (PR #2344) / (b) stacked branch | a | b | a | a (codex cited a rule against commits on an open PR; none found -- the brief allows PR updates, not merges) |
Reviews: codex (job 11028851) NO FINDINGS on the two harnesses + CLUBB helper tests. GLM: gray NONE; HS 3 points, none
taken -- (1)/(2) zero-wind/below-sigma_b columns: 12 of 13 columns have nonzero wind and the generator asserts the damping
branch is reached; the test is shown to fail with the sponge on and with linear_frac off by 1e-6; (3) v IS compared;
sigma > 1 and p_s < 1 Pa branches are unreachable for physical columns (stated scope limit).

## Round 4 (Beres on the real table, 2026-11-04)
| id | question | me | codex | GLM | result |
|---|---|---|---|---|---|
| V27 | source: (a) CESM svn inputdata (ETag rev 15) + sha256, E3SM inputdata copy fetched only to compare sha256 (identical) / (b) CESM only | a | a | a | a |
| V28 | storage: (a) .nc outside git + SHA256SUMS; fixture stores the sliced table; loader bitwise test when .nc present / (b) skip when absent | a | a | a | a |
| V29 | arms: (a) the two existing Beres arms, both column sets, NEW fixture, existing fixture untouched / (b) regenerate existing | a | a | a | a |
| V30 | oracle table read: (a) E3SM's own gw_init_beres (verbatim) + netcdf-fortran; table not perturbed by the spread / (b) Python read, text | a | a | a | a |
| V31 | branch: (a) stacked on feat/f4-oracle-spread / (b) off main | a | a | b | a |
| V32 | harness path: (a) F4_WT env override, default unchanged / (b) copy harness | a | a | a | a |
| V33 | spread selects the 4 real-table comparison tests only (a) / plus loader test (b) | a | a | a | a |
| V34 | codex P2 (netcdf build made the existing build need netcdf): (a) existing build.sh restored, netcdf steps in gw_beres_table/build.sh / (b) conditional | a | a | a | a |
| V35 | GLM "Beres may be inactive": (a) cite control (table x(1+1e-6) fails all 4 comparisons), no change / (b) store hdepth/launch spectrum | a | a | a | a |
GLM review points 3 (coverage: uh/hdepth clamps, zero heating) and 4 (metric denominators are max|oracle tendency| per
column / field, not |U|): 3 noted as scope (the arms are the existing harness's, V29), 4 is a misreading -- not taken.

RULE 3 follow-up (coordinator 2026-11-04: production must point at the real table):

| id | question | me | codex | GLM | result |
|---|---|---|---|---|---|
| V36 | no committed config selects convective Beres: (a) validate_strict refuses convective + empty table path / (b) default path = Ginsburg absolute path / (c) report only | a | a | a | a (superseded in part by V38: both reviewers showed the run-config guard alone is bypassable via gravity_wave_drag_override and the WB spectral param path) |
| V37 | validate-only tests: (a) point at the pinned file, skip when absent / (b) assert refusal only | a | a | a | a (superseded by V41) |
| V38 | (a) E3SMBeresConfig.use_stand_in_table default False so every lane raises without a table; stand-in tests opt in explicitly; guard reads the override too / (b) keep True, guard each factory | a | a | a | a |
| V39 | Hines oracle = LMDZ r5891 hines_gwd.f90 (recorded on the Hines branch) | a | a | a | a |
| V40 | codex re-review "table pin unenforced": (a) sha256 checked at load, explicit allow_unpinned=True for synthetic tables / (b) structure check + log hash | a | a | a | a |
| V41 | codex re-review "wiring tests skip off-Ginsburg": (a) validate-only tests use a tmp empty file (validate_strict checks existence only) / (b) keep skip | a | a | a | a |
| V42 | codex re-review "spectral/WB lane forwards no table or convective heating, so it now raises": (a) leave fail-closed, report, separate PR if asked / (b) wire it here | a | a | a | a |

GLM re-review of V38: restart serialisation (no checkpoint stores GWD config: grep finds the flag only in
driver/config.py and the harness), MPAS own default (refuted: the MPAS factory test failed before its explicit
opt-in, job 11034123), explicit opt-in still allowed (by design, tests only) -- not taken. Non-vacuity: reverting
the default and the guard makes the kernel-refusal and run-config-refusal tests FAIL (job 11034429, DID NOT RAISE x2).
V40/V41 delta re-review: codex CLEAN. GLM: (1) lru_cache key has no mtime, so a table replaced on disk after the
first load is not re-hashed -- not taken (pre-existing cache design; the pinned file is read-only, 0444); (2) "no
negative hash test" -- refuted: both hash-gate tests FAIL with the gate removed (job 11034524, DID NOT RAISE x2),
and a failing first load raises before anything is cached.
