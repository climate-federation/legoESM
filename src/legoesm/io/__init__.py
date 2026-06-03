"""Input/Output module for legoESM (legoesm-core substrate).

CF/CMOR-compliant NetCDF output, the generic (component-agnostic) state
checkpoint, and the pure state-digest helpers.

The experiment-aware checkpoint/restart I/O (run manifest, reproducibility
spine, AMIP/ExperimentConfig serialization) lives in the driver layer —
``legoesm.driver.{restart,checkpoint,distributed_checkpoint}`` — because it
depends on the experiment-config schema, which sits ABOVE this substrate
(federation carve, Step 3).
"""

from legoesm.io.cmor_output import (
    CFWriter,
    CMOR_TABLES,
    CMIP6_PLEV19,
    lookup_cmor_entry,
)
from legoesm.io.state_digest import (
    compute_state_digest,
    pytree_state_digest,
)
from legoesm.io.state_checkpoint import (
    save_state_checkpoint,
    load_state_checkpoint,
    validate_state_checkpoint,
)
