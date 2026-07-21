"""--radiation-column-chunk round-trips through the run_coupled CLI.

The coupled radiation path (``physics_pipeline._build_rrtmgp_radiation_fn``)
honours ``RRTMGPConfig.column_chunk_size``, so the coupled runner must be able
to set it. ``main()`` wires ``rrtmgp_column_chunk_size=args.radiation_column_chunk``
into the driver ``ExperimentConfig``; here we lock the parser default (0 = off,
byte-identical) and that a set value flows to the arg.
"""

from scripts.run.run_coupled import build_parser


def test_radiation_column_chunk_default_off():
    args = build_parser().parse_args([])
    assert args.radiation_column_chunk == 0


def test_radiation_column_chunk_flag_parses():
    args = build_parser().parse_args(["--radiation-column-chunk", "512"])
    assert args.radiation_column_chunk == 512
