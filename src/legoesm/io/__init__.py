"""Input/Output module for legoESM.

Provides CF/CMOR-compliant NetCDF output and enhanced restart I/O.
"""

from legoesm.io.cmor_output import (
    CFWriter,
    CMOR_TABLES,
    CMIP6_PLEV19,
    lookup_cmor_entry,
)
from legoesm.io.restart import (
    RestartMetadata,
    ReproducibilityReport,
    compute_state_digest,
    compute_config_hash,
    save_restart,
    load_restart,
    verify_reproducibility,
)
from legoesm.io.distributed_checkpoint import (
    save_checkpoint_distributed,
    load_checkpoint_distributed,
    save_checkpoint_sharded,
)
