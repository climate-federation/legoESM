# ORCA2 round-62 vector-advection split acquisition

The operator runs:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round62_vector_advection_acquisition/run.sh --run
```

The script creates the fresh configuration
`ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP` and writes two rank-tagged stage-2
records under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round62/acquisition_vector_split/orca1ice_vector_advection_split_np2`.
It adds calls and one WRITE-only module; it removes no NEMO arithmetic line.

Each record is self-describing: magic, 20 header integers, and a sequence of
`(name, rank, n1, n2, n3, payload)` fields through physical EOF.  It separates
the cumulative accumulator immediately before KEG, after KEG, and after ZAD,
and carries the Kmm velocities, effective vertical velocity, live thicknesses,
metrics, masks, MPI rank, global origin, owned bounds, and executed flags.
Admission requires both rank streams, parses every field and payload length
from the stream, and proves the four final restarts remain byte-identical to
the admitted parent.
