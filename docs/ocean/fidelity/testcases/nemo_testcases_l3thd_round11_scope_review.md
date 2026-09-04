# Codex-internal Round-11 scope review

Reviewer identity: `/root/round11_scope_review`  
Commit reviewed: `b135e32ac73b5635f535faf3b74a2b7108db2bc4`  
Verdict: **SHIP**

## Findings

- No blocking findings.
- The rung-3.6 stop is scope-faithful: the reviewed design requires both real
  ORCA1 RGB inputs and says implementation must stop when they are unavailable.
- Exact-name search under `/data/abyssal/dbalwada` found neither required file;
  only the C1D `CHLA_BATS.nc` exists.  ORCA1 deck lines 160-170 select the exact
  missing pair.
- The receipt reports both names, the documented URLs, the absence of a
  published/found checksum, and the exact sandbox fetch failure.  It refuses
  every forbidden substitute.
- Archive MD5/SHA-256 and ERA5 SHA-256 were independently recomputed and agree
  with the receipt.
- Labels and ASKED/UNASKED dispositions are honest; rung 3.6 remains blocked
  and unmeasured.
- Delta `b834e73..b135e32` contains only 15 small source/test/docs files, no
  NEMO-tree path and no runtime artifact.
- Exact GYRE blobs, commit segmentation, and messages were verified.
- Reviewer tests: 72/72 core policy tests, 13/13 bulk gate/all plants, 35/35
  earlier SI3 regression tests, and 7/7 touched production-file constants
  ratchets.

## Packaging note

The bundle inspected during this review was the expected stale pre-Round-11
bundle.  Final packaging must regenerate it after the review records are
committed and verify the branch ref.  This was not a defect in the reviewed
commit.

## UNVERIFIED

- The reviewer could not load the external documentation pages; it verified
  the exact archive fetch failure and local selectors, but not the page text's
  expiry statement or the global absence of a published checksum.
- The reviewer validated the retained gate artifact and reran its full test
  harness, rather than separately rerunning the 8,760-step CLI command.
- The final refreshed bundle did not yet exist at review time.
