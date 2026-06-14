"""Single-column model (SCM) test cases.

Each submodule exposes ``add_args(parser)`` and ``run(args) -> int`` so that
the cases can be invoked either standalone (``python -m scripts.scm.rce``-
style is not supported because ``scripts/`` is not a package; use
``scripts/matrix/run_scm_test_matrix.py <case>``) or via the dispatcher.
"""
