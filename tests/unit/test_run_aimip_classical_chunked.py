"""Dispatch tests for the AIMIP *classical* chunked-streaming path.

Classical (full physics + RRTMGP) originally loaded the whole dataset
host-resident and trained one non-chunked epoch, so a T106 epoch longer
than the walltime cap wrote NO ``chunk_latest.eqx`` and the chained driver
refused to resume (rc=124, "NO new checkpoint" — the observed empty-result
failure).  ``_train_aimip_classical`` now mirrors the NN variants: when
``aimip_chunk_windows > 0`` it builds ``_make_chunk_loader`` and hands it to
the shared ``_train_spectral_loop``, which streams the data and writes a
per-chunk mid-epoch checkpoint (#942).

These tests pin the *dispatch* (which branch runs, and what the shared loop
is handed) by stubbing the three ``neural_gcm_spectral`` entry points the
classical trainer calls — the actual streaming/checkpoint round-trip is
covered by ``test_aimip_resume.py`` and ``test_neural_gcm_spectral.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax

# Building the real Gaussian/spectral grid in ``_train_aimip_classical``
# requires x64 (create_gaussian_grid raises otherwise).
jax.config.update("jax_enable_x64", True)

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "_aimip_classical_chunked_mod", REPO / "scripts" / "run" / "run_aimip.py"
)
run_aimip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_aimip)


def _base_cfg(tmp_path, **over):
    """Minimal cfg that ``_build_spectral_config`` + the classical trainer
    accept.  ``aimip_radiation='gray'`` avoids the RRTMGP GHG lookup;
    ``spatial_surface`` defaults off so no land mask / eager load is needed."""
    cfg = dict(
        n_max=21, nlev=8, dt=1800.0,
        aimip_n_epochs=1, aimip_lr=1e-3, aimip_weight_decay=0.0,
        aimip_grad_clip=1.0, n_train_days=1,
        aimip_radiation="gray", aimip_rad_update_interval=1,
        output_dir=str(tmp_path), aimip_variant="classical",
        train_windows=[[2015, 0, 1], [2015, 121, 1], [2015, 243, 1]],
    )
    cfg.update(over)
    return cfg


class _Rec:
    """Records the (args, kwargs) of the last call and returns ``ret``."""

    def __init__(self, ret):
        self.ret = ret
        self.calls = []

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.ret

    @property
    def called(self):
        return bool(self.calls)


def _patch_ng(monkeypatch, *, loader_ret, loop_ret, load_ret):
    """Stub the three neural_gcm_spectral entry points the classical trainer
    imports at call time; return the recorders."""
    import legoesm.training.neural_gcm_spectral as ng

    make_chunk = _Rec(loader_ret)
    train_loop = _Rec(loop_ret)
    load_data = _Rec(load_ret)
    monkeypatch.setattr(ng, "_make_chunk_loader", make_chunk)
    monkeypatch.setattr(ng, "_train_spectral_loop", train_loop)
    monkeypatch.setattr(ng, "load_training_data", load_data)
    return make_chunk, train_loop, load_data


def test_chunked_dispatch_streams_and_never_full_loads(tmp_path, monkeypatch):
    """chunk_windows>0 → build a chunk_loader, hand it to the loop with
    ic_states/target_carries=None + resume_from_dir, and NEVER host-load the
    full dataset (the OOM/timeout path we are replacing)."""
    cfg = _base_cfg(tmp_path, aimip_chunk_windows=2)
    spec_cfg = run_aimip._build_spectral_config(cfg)

    sentinel_loader = object()
    make_chunk, train_loop, load_data = _patch_ng(
        monkeypatch,
        loader_ret=(sentinel_loader, 42),
        loop_ret=("model", [0.1]),
        load_ret=(["ic"], ["tgt"], ["t"]),
    )

    out = run_aimip._train_aimip_classical(
        spec_cfg, "cache", cfg=cfg, resume_from_dir="/some/ckpt/dir",
    )
    assert out == ("model", [0.1])

    assert make_chunk.called, "chunked path must build a chunk loader"
    assert not load_data.called, "chunked path must NOT full-load the dataset"

    (loop_args, loop_kwargs) = train_loop.calls[-1]
    # Positional signature: (params, make_physics_fn, grid, sigma,
    #                        ic_states, target_carries, config, ...)
    ic_states, target_carries = loop_args[4], loop_args[5]
    assert ic_states is None and target_carries is None
    assert loop_kwargs["chunk_loader"] is sentinel_loader
    assert loop_kwargs["n_samples_total"] == 42
    assert loop_kwargs["resume_from_dir"] == "/some/ckpt/dir"
    # The streamed path must NOT also claim host_staged (that flag is for the
    # non-chunked host-resident list; the chunk loader signals staging itself).
    assert loop_kwargs.get("host_staged") is not True


def test_nonchunked_dispatch_host_loads_and_stages(tmp_path, monkeypatch):
    """chunk_windows=0 → preserve the legacy behaviour: host-resident full
    load, no chunk loader, host_staged=True."""
    cfg = _base_cfg(tmp_path, aimip_chunk_windows=0)
    spec_cfg = run_aimip._build_spectral_config(cfg)

    make_chunk, train_loop, load_data = _patch_ng(
        monkeypatch,
        loader_ret=(object(), 0),
        loop_ret=("model", []),
        load_ret=(["ic"], ["tgt"], ["t"]),
    )

    run_aimip._train_aimip_classical(spec_cfg, "cache", cfg=cfg)

    assert not make_chunk.called, "non-chunked path must not build a chunk loader"
    assert load_data.called, "non-chunked path must host-load the dataset"
    assert load_data.calls[-1][1]["host_resident"] is True

    (loop_args, loop_kwargs) = train_loop.calls[-1]
    assert loop_args[4] == ["ic"] and loop_args[5] == ["tgt"]
    assert loop_kwargs.get("host_staged") is True
    assert "chunk_loader" not in loop_kwargs or loop_kwargs["chunk_loader"] is None


def test_chunked_spatial_surface_loads_one_window_for_land_mask(
    tmp_path, monkeypatch
):
    """chunk_windows>0 AND spatial_surface → the static phis-derived land mask
    still needs one loaded carry; load EXACTLY the first window (cheap), then
    stream the rest via the chunk loader."""
    cfg = _base_cfg(
        tmp_path, aimip_chunk_windows=2, aimip_spatial_surface=True,
    )
    spec_cfg = run_aimip._build_spectral_config(cfg)

    class _FakeCarry:
        phis = 0.0  # land_mask_from_phis is stubbed, so any value is fine

    make_chunk, train_loop, load_data = _patch_ng(
        monkeypatch,
        loader_ret=(object(), 42),
        loop_ret=("model", []),
        load_ret=(["ic"], [_FakeCarry()], ["t"]),
    )
    # Stub the mask builder + stager so no grid-shaped array is required.
    import legoesm.training.aimip_spatial as sp
    import legoesm.training.neural_gcm_spectral as ng
    monkeypatch.setattr(sp, "land_mask_from_phis", lambda phis, smooth=True: phis)
    monkeypatch.setattr(ng, "stage_sample", lambda x, device=None: x)

    run_aimip._train_aimip_classical(spec_cfg, "cache", cfg=cfg)

    assert make_chunk.called, "spatial+chunked still streams the training data"
    # The land-mask load must be the FIRST window only, not the full set.
    assert load_data.called
    windows_used = load_data.calls[-1][1]["windows"]
    assert tuple(windows_used) == spec_cfg.windows[:1]
