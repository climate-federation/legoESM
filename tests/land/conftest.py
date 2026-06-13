"""Land-test collection fixup.

``legoesm.land.__init__`` eagerly imports ``legoesm.land.config``, which pulls
``surface_scheme -> coupler -> ... -> driver.coupled_config`` and back to
``land.config`` — a partially-initialized circular import that aborts collection
when a land test is the *first* thing to import ``legoesm.land`` in a process
(e.g. running a single land test file).  In a full-suite run an earlier test
happens to import ``legoesm.driver`` first and breaks the cycle, which is why it
only bites isolated runs.

Importing ``legoesm.driver`` here, before any land test module is collected,
resolves the import order deterministically for every land test.
"""

import legoesm.driver  # noqa: F401  (import for side effect: break land<->driver cycle)
