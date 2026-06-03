"""Central deterministic RNG for reproducible runs (Stage A1).

Every random key in a run should descend from one master seed, so the whole run
is reproducible from the single ``seed`` recorded in the run manifest.  A
consumer asks for a *named* subkey::

    from legoesm.runtime.rng import split_keys
    keys = split_keys(config.seed, ["ensemble_ic", "sppt", "ml_init"])
    perturb(..., keys["ensemble_ic"])

rather than calling ``jax.random.PRNGKey`` with its own ad-hoc seed (which is how
seeds silently diverge and runs stop reproducing).

Provenance model: a run is reproducible because the run manifest records *every*
seed — both ``config.seed`` (the master, here) and any dedicated seed *config
fields* (e.g. ``physics_parameterization_seed``), which live in
``resolved_config``.  ``config.seed`` + :func:`split_key` is the central entry
point for *new* randomness that would otherwise carry no provenance (the
ensemble-IC key was a hardcoded ``PRNGKey(42)`` before this).  Migrating the
remaining dedicated-seed consumers onto ``split_key`` is an incremental
follow-up: it is a *behaviour* change (the derived stream differs), not a
reproducibility fix, since those seeds are already captured in the manifest.

Subkeys are derived by *folding the name's full SHA-256 digest* into the master
key, so the mapping is:

* **deterministic across processes** — names fold through SHA-256, not Python's
  salted ``hash()``;
* **order-independent** — ``split_keys(s, ["a", "b"])["a"]`` equals
  ``split_keys(s, ["b", "a"])["a"]``; adding a new consumer does not perturb the
  keys handed to existing ones (so adding randomness in one place cannot silently
  change another place's stream);
* **collision-resistant** — all 256 bits of the digest are folded in (as eight
  32-bit words), not a 32-bit truncation, so distinct names cannot alias onto the
  same stream at any realistic number of consumers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import jax


def _derive_key(base, name: str):
    """Fold the full SHA-256 digest of *name* into *base*, one 32-bit word at a time."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()  # 32 bytes
    key = base
    for i in range(0, len(digest), 4):
        key = jax.random.fold_in(key, int.from_bytes(digest[i:i + 4], "big"))
    return key


def split_keys(master_seed: int, names: Sequence[str]) -> dict:
    """Return a ``{name: jax PRNG key}`` map derived from one *master_seed*.

    Each key folds the full SHA-256 digest of *name* into
    ``PRNGKey(master_seed)`` — distinct per name, identical for the same
    ``(master_seed, name)``, and independent of the order/membership of *names*.
    """
    base = jax.random.PRNGKey(master_seed)
    return {name: _derive_key(base, name) for name in names}


def split_key(master_seed: int, name: str):
    """Convenience: a single named subkey (see :func:`split_keys`)."""
    return split_keys(master_seed, [name])[name]
