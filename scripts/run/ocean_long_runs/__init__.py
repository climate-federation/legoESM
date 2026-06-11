"""Climate-scale ocean run drivers (OMIP-2, Bryan THC spinup).

These drivers are designed to run on a cluster (weeks to months of
wall-clock at 1 deg / 100 km production resolutions). Each driver
accepts a ``--smoke`` flag that integrates a single day so the
matrix CI can exercise the full code path without consuming compute
budget.
"""

__all__ = []
