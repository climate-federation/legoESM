# ORCA2 round 338 — standalone S3 ice-interface model preparation

Date: 2026-10-26.  Base `aae42be88d67`.  Preregistration `c89dba148`
(erratum `fd1189afd`), model commit `8052799b0`.  Final disposition:
**LANDED-pending; the literal `sbc_ice_if` transcription is exact against an
independent scalar transcription under JIT and truly eager; no NEMO record
exists yet, so nothing is NEMO-certified**.  No configuration changed and no
NEMO executable ran.  Evidence: `orca2_rounds/round338/`.

## Source basis

No K0 build exists, so every NEMO line below is the record build's ppsrc,
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/` (keys `key_si3 key_qco
key_vco_1d3d key_RK3`, flags `-fdefault-real-8 -O3 -funroll-all-loops ...
-fno-tree-vectorize`).  K0 keeps the same keys; its ppsrc must be re-compared
before the S3 acquisition (preregistration, prerequisite 3).

The update block is read at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcice_if.f90:158-159`:

    CALL eos_fzp( sss_m(:,:), fr_i(:,:) )       ! sea surface freezing temperature [Celsius]
    fr_i(:,:) = fr_i(:,:) * tmask(:,:,1)

then the halo clamp at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcice_if.f90:164-167`:

    DO jj = ntsj-( nn_hls), ntej+(  nn_hls ) ; DO ji = ntsi-( nn_hls), ntei+(  nn_hls)
    ts(ji,jj,1,jp_tem,Kmm) = MAX( ts(ji,jj,1,jp_tem,Kmm), zt_fzp )     ! avoid over-freezing point temperature

and the interior flux loop at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcice_if.f90:182-185`:

    zqri = ztrp * ( ts(ji,jj,1,jp_tem,Kbb) - ( zt_fzp - 1.) )
    zqrj = ztrp * MIN( 0., ts(ji,jj,1,jp_tem,Kbb) - zt_fzp )

and `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcice_if.f90:192-195`:

    zqi = -3. + SIGN( 1._wp, ff_f(ji,jj) )
    qns(ji,jj) = ( ( 1.- zfr_obs ) * qns(ji,jj)                             &

The deck runs EOS-80 (`ln_eos80 = .true.`), so the executing freezing point
is `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:1867-1868`:

    ptf(ji,jj) = ( - 0.0575_wp + 1.710523e-3_wp * SQRT( psal(ji,jj) )   &
    &                       - 2.154996e-4_wp *       psal(ji,jj)   ) * psal(ji,jj)

`SIGN` resolves to `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/lib_fortran.f90:1815-1816`:

    IF ( pb >= 0._wp ) THEN   ;   SIGN_SCALAR =  ABS(pa)
    ELSE                      ;   SIGN_SCALAR = -ABS(pa)

## Result

- **Premise correction (CONFIRMED, read off the source).**  The routine does
  not modify `emp` or `sfx`, although its header comment says so; `zsice` is
  computed and never read.  The Decision-161 shared function is the TEOS-10
  branch; ORCA2 runs EOS-80, so a literal EOS-80 sibling was added next to it
  instead of reusing it.
- **Kbb = Kmm (CONFIRMED, read off the source).**  The RK3 step calls the
  surface module with both levels equal,
  `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:155`:

      CALL sbc        ( kstp, Nbb, Nbb )                ! Sea Boundary Condition (including sea-ice)

  so the flux loop reads the clamped temperature.  Consequence: the
  under-ice damping `MIN(0., ts - zt_fzp)` is always `0.`, and `zqrj` is
  always `-0`.  The unit test measures `ts - zt_fzp >= +0` on every finite
  cell.
- **Object-code tie audit (CONFIRMED, `objdump -dr` of `sbcice_if.o` and
  `eosbn2.o`).**  No fused multiply-add in either.  The clamp is
  `movsd ts; maxsd fr_i; movsd -> ts`, so a tie or NaN keeps `zt_fzp`, the
  second operand.  `MIN(0., x)` is `minsd %xmm9(=+0), x`, so a tie keeps the
  literal `0.`, the FIRST operand — the opposite of the shared min helper's
  ordering, so the call passes the operands reversed.  That tie cannot occur
  here, because `x >= +0`, and a plant swapping it changes no bit.  The
  `fr_i` test is `comisd` + `jb`, so NaN gives `fr_i = 0`, as the source's
  `<=` does.  The `-3. + SIGN` term is a call to `sign_scalar`.  EOS-80 `SQRT`
  is inline `sqrtsd`, so no libm arm is involved.  The rodata constants are
  `1.0, -40.0, 3.0` and `1.710523e-3, 0.0575, 2.154996e-4`, all binary64
  (`-fdefault-real-8`).
- **Signed zero on land (CONFIRMED, unit test).**  EOS-80 at `S = +0` gives
  `-0.0575 * (+0) = -0`; masked, it stays `-0`; MAXSD keeps it, so land
  `ts = +0` leaves as `-0`.  `jnp.maximum` would keep `+0`: its plant breaks
  the pin.
- **Exactness (CONFIRMED).**  The CPU binary64 JIT routine matches an
  independent pure-Python scalar transcription, one rounding per source
  statement, at **0 unequal words** on the four outputs: `ts` and `fr_i` on
  the full halo-inclusive array, `qsr` and `qns` on the interior.  The
  synthetic grid includes ties, `±0`, NaN, land and both `fr_i` branches.
  Truly eager (whole call under `jax.disable_jit()`) equals JIT bitwise.
- **Barriers are load-bearing (CONFIRMED, M1).**  On a 150 x 182 synthetic
  grid, the barrier-free arm differs in 3,920 / 107,880 words; with barriers,
  0.  This is now a committed plant.
- **Gradients (CONFIRMED, FD).**  On an open, unclamped cell
  `dqns/dts = cover * ztrp` (-24 at cover 0.6), matching central FD to 1e-7.
  Below freezing the clamp sends no gradient to `ts`; at the tie it goes to
  the freezing point (second operand), matching the forward choice.  Once
  clamped, `zqri = ztrp*(zt - (zt - 1))` is the constant -40, so qns has
  exactly zero sensitivity to salinity.  Instead the salinity gradient
  reaches the clamped temperature through `dT_f/dS`, matching FD.  `fr_i`
  is a step, so no gradient flows through `sst_m`.  The sqrt gradient is cut
  at `S <= 0` with forward bits kept, so land cells give finite gradients.
- **Production path: NONE YET (stated, not hidden).**  As with S1's literal
  arm, no shipped card can call this routine: it runs only under
  `nn_ice = 1`, which only the future K0 standalone card selects (S4).
  Wiring it now would be a configuration change.  The M4 production-call
  plant is therefore deferred to S4.

## fld_read of `sn_ice` (S0-F's term; not transcribed here)

`cfgs/SHARED/namelist_ref:418` declares the record:

    sn_ice      ='ice_cover_clim.nc'      ,        -12.       ,'ice_cover',   .true.    , .true.  , 'yearly'  ,   ''            ,   ''     ,   ''

A `-12` record is one record per file:
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/fldread.f90:885`:

    IF    ( NINT(sdjf%freqh) == -12 ) THEN            ;   ireclast = 1    ! yearly mean: consider only 1 record

Time interpolation still runs between two copies of that field,
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/fldread.f90:244-246`:

    ztinta =  REAL( isecsbc - sd(jf)%nrec(2,ibb), wp ) / REAL( sd(jf)%nrec(2,iaa) - sd(jf)%nrec(2,ibb), wp )
    ztintb =  1. - ztinta
    sd(jf)%fnow(:,:,:) = ztintb * sd(jf)%fdta(:,:,:,ibb) + ztinta * sd(jf)%fdta(:,:,:,iaa)

`(1-w)*F + w*F` is not bitwise `F`.  On 200,003 synthetic uniform values,
12,000--37,500 cells (6--19%) moved one ULP at w = 0.1234567 and 0.4983, and
none at w = 0.25, 0.5 or 0.75 (PLAUSIBLE for the real file).  The routine
therefore takes `fnow` as its input, and the S3 record must carry `fnow`.

## Ice-cover production plan and S3 preregistration

The operator task is in the design (S3, "Operator task — ice-cover file
production").  No admitted record supplies a yearly mean, so it needs a new
two-rank run of the round-5 build with one passive observer after the SI3
call, `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcmod.f90:501`:

    CASE(  2 )   ;         CALL ice_stp  ( kt, Kbb, Kmm, nsbc )      ! SI3 ice model

which sets the fraction at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceupdate.f90:212`:

    fr_i(ji,jj) = at_i(ji,jj)             ! Sea-ice fraction

The plan averages 1,460 SBC samples in binary64 in kt order and writes
`ice_cover(time_counter=1, y=148, x=180)` as doubles with digests.  The run
takes about 55 minutes of stepping; reserve 90 minutes.  Its length,
averaging year and deck are the open decision below.  The S3 acquisition is
preregistered (arrays, five call points around
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcmod.f90:500`, kt
1--10, two ranks, A/B twins, six falsifiers).  It requires the K0 build and
its smoke run first.

## Non-NEMO compatibility control (M6)

Both touched shared files change by pure addition: `git diff -U0` over
`packages/` removes 0 lines.  The new import adds no caller.  Round 335's
complete caller census was rerun, parent = `aae42be88` export, current =
this tree.  Result: **0 / 10,231,076 unequal across 214 arrays**, inventory
equal; the one-ULP `A/app_jax/q_net` plant exits 1.  The current census file
is byte-identical to round 335's.  The freezing-point, frazil, ice-shelf,
EOS-80, NCAR-bulk and OMIP-applicator suites pass, together with the new
file: 185 passed.

## Gates

- New S3 tests: 12 / 12 PASS (`s3_tests.log`).
- Shared-caller suites: 185 PASS (`shared_tests.log`).
- Ratchet suite (round 335's three invocations): 3 FAIL, 12,306 PASS,
  4 SKIP.  The failing set equals round 335's lane-inherited set exactly
  (JRA55 test literal, land restart coefficient, shortwave-penetration
  spec); none names a changed file.
- Default `land.sh` inventory and per-receipt citation gate: see the
  verification section appended below.
- codex review: operator.

## CHOICES

- ASKED: S3 model preparation, ice-cover production plan, S3
  preregistration, no NEMO run (round prompt; Decisions 168, 171, 175).
- UNASKED, offered for revert: (1) placing the routine in
  `bulk_flux_omip.py` beside the S1 literal arm; (2) adding
  `nemo_eos_fzp_eos80` instead of reusing the TEOS-10 function the prompt
  named, because the deck runs EOS-80; (3) the caller applies the
  `MOD(kt-1, nn_fsbc)` gate; (4) one temperature input, because Kbb = Kmm
  in this build; (5) the gradient-only sqrt cut at `S <= 0`; (6) the
  levitating-ice initial-SSH recipe keeps its own NumPy EOS-80 copy, measured
  bit-identical to the new function and not rerouted.
- PENDING (asked below, not applied): the ice-cover producer's run length,
  averaging year and deck.

## Open

1. DECISION: ice-cover producer (one year of the round-5 deck is the pick).
2. S0-F: certify `fld_read` for `sn_ice`, including the one-ULP blend.
3. S4: build K0, boot/10-step smoke, wire S1 -> S3 -> S2 into the
   standalone card with a production-call plant, then acquire and score S3.

## Verification (final tree, foreground, CPU binary64)

- Per-receipt citation gate on this receipt: PASS, 13 citations, 0
  failures, 0 unmapped, 0 map-audit failures, 9 / 9 self-tests fired.  The
  two-line shift plant on the RK3 `sbc` call key exits 1
  (SYMBOL-NOT-AT-LINE).
- Default `land.sh` ORCA2 inventory, split to fit the foreground cap:
  99 + 41 + 11 = **151 / 152 PASS**.  The remaining test, the staircase
  stage-velocity test in the face-mask-rank file (574 s alone in round 335),
  was killed by the 595 s cap before it finished, with the machine's load
  average near 25.  It is **UNVERIFIED here**; the operator's push gate runs
  it.  Both package changes are pure additions (0 removed lines), so no
  existing function body it could call changed.
