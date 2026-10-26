# Preregistration — ORCA2 round 89 rung-0 debug-build resume

Date: 2026-10-01.  Base: `5ca47b129`.  Scope: repair the operator-facing
round-88 debug launcher only.  This round changes no NEMO source, legoESM
package, card, hierarchy switch, carried state, threshold, stabilizer, or
sea-ice selector.  All labels are **independent**.

## Frozen predictions

1. The operator-returned exit at round-88 launcher line 118 is a build-tool
   setup refusal, not a model result: the inherited partial target lacks
   `BLD/bld.cfg` because both preceding `makenemo` calls used `-j 0`.
2. The partial target is safe to resume only if a fail-closed census pins its
   configuration, copied sources, CPP keys, generated architecture, histories,
   and the absence of a binary, parsed build configuration, run directory, and
   scientific output.
3. Copying a pinned complete NEMO `bld.cfg`, changing only its configuration
   root from the template target to the round-88 target, and retaining the
   already generated debug architecture is sufficient for `fcm build` to
   produce the debug executable.
4. The debug executable will reproduce a nonzero failure whose backtrace names
   `tra_adv_trp[_t]` and one concrete `traadv.f90:N` line.  Debug values remain
   diagnostic-only.

## Falsifiers

- Any source/config/history hash or inventory differs from the committed
  expected manifest, or the target has advanced beyond the recorded pre-build
  state: refuse rather than resume.
- A build configuration needs a semantic edit beyond the target-root rewrite,
  the debug flags do not survive into `parsed_bld.cfg`, or vector-math symbols
  appear: prediction 3 is REFUTED.
- The debug run exits zero, or its nonzero log lacks both the tracer-transport
  routine and a source-resolved `traadv.f90:N`: prediction 4 is REFUTED and no
  instrumentation repair is authorized.

## Gates

The launcher must be clean-tree and committed, parse under `bash -n`, validate
the pinned partial target before any write, carry firing inventory/advanced-
target controls, and write a self-describing evidence manifest.  The focused
tests, citation gate plus a real plant, one ocean-fidelity battery, and a
separate read-only Codex review are required.  No physics or ladder gate is
applicable unless the source-resolved debug record returns in this round.
