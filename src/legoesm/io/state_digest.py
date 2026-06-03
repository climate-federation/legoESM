"""Backend-agnostic SHA-256 digests of model state.

Pure substrate (legoesm-core): no config / experiment dependency, so the
reproducibility spine (``driver.restart``) and the state checkpointer
(``io.state_checkpoint``) can both share these without either importing the
other or pulling in the experiment-config layer.
"""

from __future__ import annotations

import hashlib

import jax
import numpy as np


def compute_state_digest(state_arrays: dict[str, np.ndarray]) -> str:
    """SHA-256 of the concatenated raw bytes of *state_arrays*.

    Keys are sorted so that the digest is independent of insertion order.
    """
    h = hashlib.sha256()
    for key in sorted(state_arrays.keys()):
        arr = np.asarray(state_arrays[key])
        h.update(arr.tobytes())
    return h.hexdigest()


def pytree_state_digest(*trees) -> str:
    """Backend-agnostic SHA-256 digest of one or more state pytrees.

    Unlike :func:`compute_state_digest` (which assumes the grid-point checkpoint
    array layout ``state.T``/``state.u``/...), this flattens whatever pytrees it
    is given via ``jax.tree_util.tree_leaves`` and digests every array leaf, so it
    works for *any* backend's state — grid-point, spectral (``T_hat`` complex
    coefficients), or MPAS (edge-normal ``u``, ``v=None``).  Shape and dtype are
    folded in alongside the bytes so a structural change cannot collide.

    The pytree *structure* (dict keys, nesting, ``None`` placeholders) is folded
    into the hash alongside the leaf bytes, so a layout/schema change — e.g. a
    leaf moving keys, or a ``None`` slot appearing/disappearing — cannot collide
    with the original even when the surviving array leaves are identical.
    """
    import numpy as _np

    h = hashlib.sha256()
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        # treedef repr encodes keys / nesting / None structure deterministically.
        h.update(str(treedef).encode("utf-8"))
        host = jax.device_get(leaves)  # single batched device->host transfer
        for arr in host:
            a = _np.asarray(arr)
            h.update(str(a.shape).encode("utf-8"))
            h.update(str(a.dtype).encode("utf-8"))
            h.update(a.tobytes())
    return h.hexdigest()
