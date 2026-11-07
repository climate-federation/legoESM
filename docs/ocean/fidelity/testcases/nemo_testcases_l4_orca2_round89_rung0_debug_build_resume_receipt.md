# ORCA2 round 89 — rung-0 debug-build resume

**Verdict: STOPPED_FOR_RECORD.** The operator-returned round-88 action never
built or ran NEMO: it stopped at launcher line 118 because the dry-run
`makenemo -j 0` setup intentionally left no `BLD/bld.cfg`, and the launcher
called `fcm build` without installing the template used by the earlier debug
pattern. A fail-closed resume launcher is committed and preflight-clean. No
NEMO source, legoESM package, card, hierarchy switch, carried state, threshold,
stabilizer, or sea-ice selector changed. Every label is **independent**.

Base: `5ca47b129`. Preregistration commit: `614c9b4e5`. Launcher commit:
`721039c32`. The preregistration preceded all round-89 target inspection beyond
the operator-returned failure log.

## Operator result and exact checkpoint

The returned log ends with:

```text
Unable to read config file ".../ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/bld.cfg"
REFUSE: round-88 rung-0 frame debug failed at line 118 (exit 2)
```

The partial target contains its generated debug architecture, CPP key files,
copied EXP00 and MY_SRC trees, and 445 WORK links. It contains no `bld.cfg`, no
binary, no parsed build configuration, no run directory, and no scientific
output. Its complete depth-two filesystem/content/link fingerprint is
`c0bfee549829a77a5443c0a4fba0caa4946dbf4de5054fdfc1f46d287b9bc341`.

The resume validates that fingerprint before its first target write. It also
pins both copied source units, the failed optimized binary and namelist, the
complete deck/input manifests, and the prior complete build template. A copied
checkpoint reproduces the fingerprint before each control: changing one source
byte and adding a mock built binary then both change it. Thus neither plant is
green before its perturbation.

The only target mutation is adding the missing build configuration after a
five-occurrence root rewrite from the pinned round-87 template. The resumed
build must retain the exact debug flags, reproduce the two inherited compiled
source digests, contain no vector-math symbol, and then produce a nonzero run
with both a tracer-transport routine and a concrete `traadv.f90:N` backtrace.
Debug values remain diagnostic-only.

## Citation-audited evidence

The copied step source is unchanged from the failed optimized record. Its
stage-0 frame call remains before the calendar/surface work in
`ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/stprk3.f90:90-108`; this round does not
promote either existing frame into a stage-1 or physics claim.

## Frozen prediction ledger

| Prediction | Verdict |
|---|---|
| exit 2 is build setup, not a NEMO result | **CONFIRMED** — no build recipe, binary, parsed config, run directory, or scientific output exists |
| the partial target can be pinned before mutation | **CONFIRMED** — exact fingerprint reproduces in two copied baselines and both post-baseline plants fire |
| adding only the re-anchored build template produces the exact debug executable | **UNMEASURED** — operator acquisition required |
| debug run names a concrete tracer-transport source line | **UNMEASURED** — operator acquisition required |

## Gates, tests, and review

- Clean-tree preflight printed
  `ORCA2_ROUND89_RUNG0_FRAME_DEBUG_RESUME_PREFLIGHT_READY` with the round-89
  target run path. Both checkpoint plants fired.
- Focused launcher controls: **4 passed**. `bash -n` and `git diff --check`
  pass.
- The one allowed `tests/ocean/fidelity -n 12` battery selected 2,219 tests,
  reached 99%, then repeated the established no-summary stall and was stopped
  after repeated silent waits. One failure was visible and reproduced alone:
  the listed pre-existing SI3 scalar-math provenance red, `A MY_SRC is not
  verbatim`. No round-89 test failed; there is no suite-PASS claim.
- The receipt citation gate passes this receipt and the default cumulative
  receipt with no unmapped citations; shifting the cited stage-program span
  makes the gate fail.
- Required review verdict: **independent review unavailable in-sandbox**. The
  separate read-only `codex exec` stopped before reading the diff with
  `failed to initialize in-process app-server client: Read-only file system`.

No `packages/` file changed, so no GYRE trajectory or card result is claimed or
required. No scientific/configuration choice was made.

## OPEN

1. The operator runs the committed round-89 launcher below. It resumes only the
   exact partial debug target and must return the source-resolved
   `traadv.f90:N` backtrace.
2. Repair only that proved write-only instrumentation statement under a fresh
   optimized target; admit 80 frames and prove 20/20 terminal restarts
   byte-identical to round 83.
3. Only after admission: build the explicit rung-0 card, compare it with the
   stage-0 entry frames, walk the first non-bit boundary, and score the given-
   entry ladder, independent ladder, and independent month.

Acquisition launcher:
`/tmp/autopilot-orca2-1128011938/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round89_rung0_frame_debug_resume/run.sh`.

## UNVERIFIED

- The debug executable and source-resolved fault line do not exist yet.
- Stage 1 through stage 3, restart identity, the rung-0 card ladders, and the
  month score remain unmeasured.
