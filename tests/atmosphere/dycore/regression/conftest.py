"""Tier + slow tagging for the curated FV3/cubed-sphere dycore regression suite.

These ~50 tests are the curated survivors of the ralph-loop iteration corpus
(see ``MANIFEST.md``): one per distinct locked numerical invariant.  They run at
``tier1`` (research-tier regression).  Four are genuinely slow (full-toolkit AD
at rest, full-stack imprint, the Smagorinsky→W2 calibration sentinel) and are
additionally marked ``slow`` so the default ``-m 'not slow'`` run skips them
while CI / explicit ``-m tier1`` still exercises them.

Tagging here (collection hook) keeps the 56 test files unmodified — they were
moved verbatim from ``tests/`` root, so a future ``git log --follow`` / ``blame``
sees a pure rename.
"""
from __future__ import annotations

import pytest

# Curated regressions whose single-file runtime exceeds ~60s (measured CPU x64):
# mark slow so the fast default run skips them; they remain tier1.
_SLOW = {
    "test_iter962_smagorinsky_tweak.py",        # ~167s — Smag calibration -> W2
    "test_nh_full_fv3_stack_imprint_iter330.py",  # ~100s — full NH stack imprint
    "test_fv3_full_toolkit_ad_at_rest_iter184.py",  # ~95s — NH full-toolkit AD
    "test_pe_full_toolkit_ad_at_rest_iter185.py",   # ~73s — PE full-toolkit AD
}


def pytest_collection_modifyitems(config, items):
    here = __file__.rsplit("/", 1)[0]
    for item in items:
        if not str(item.fspath).startswith(here):
            continue
        item.add_marker(pytest.mark.tier1)
        if item.fspath.basename in _SLOW:
            item.add_marker(pytest.mark.slow)
