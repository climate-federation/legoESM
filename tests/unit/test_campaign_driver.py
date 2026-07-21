"""Unit tests for the shared campaign driver (training/campaign_driver.py, D1)."""

import pytest

from legoesm.training.campaign_driver import (
    latest_checkpoint,
    nonempty,
    run_staged_campaign,
    validate_classical_radiation,
    validate_training_core,
)


# --- dispatch hardening -------------------------------------------------

def test_training_core_valid():
    assert validate_training_core("spectral") == "spectral"
    assert validate_training_core("latlon") == "latlon"


@pytest.mark.parametrize("core", ["cubed_sphere", "mpas"])
def test_training_core_reserved_raises_not_implemented(core):
    with pytest.raises(NotImplementedError, match="reserved"):
        validate_training_core(core)


def test_training_core_unknown_raises():
    with pytest.raises(ValueError, match="Unknown training_core"):
        validate_training_core("spectal")  # typo must never select anything


def test_classical_radiation_pin():
    assert validate_classical_radiation("rrtmgp") == "rrtmgp"
    with pytest.raises(ValueError, match="rrtmgp"):
        validate_classical_radiation("gray")
    # explicit escapes, never silent
    assert validate_classical_radiation("gray", smoke=True) == "gray"
    assert validate_classical_radiation("gray", allow_non_rrtmgp=True) == "gray"


# --- checkpoint resolution ----------------------------------------------

def _touch(p, content=b"x"):
    p.write_bytes(content)


def test_latest_checkpoint_orders_by_epoch(tmp_path):
    _touch(tmp_path / "epoch_0002.eqx")
    _touch(tmp_path / "epoch_0010.eqx")
    _touch(tmp_path / "epoch_0009.eqx")
    assert latest_checkpoint(tmp_path).endswith("epoch_0010.eqx")


def test_latest_checkpoint_prefers_ema_sibling(tmp_path):
    _touch(tmp_path / "epoch_0003.eqx")
    _touch(tmp_path / "epoch_0003_ema.eqx")
    # raw by default; EMA only when asked (evaluate-the-EMA doctrine, D3)
    assert latest_checkpoint(tmp_path).endswith("epoch_0003.eqx")
    assert latest_checkpoint(tmp_path, prefer_ema=True).endswith(
        "epoch_0003_ema.eqx"
    )


def test_latest_checkpoint_ema_files_never_win_epoch_ordering(tmp_path):
    # An EMA file of a NEWER epoch than any raw checkpoint must not be
    # selected as "the newest raw checkpoint".
    _touch(tmp_path / "epoch_0004.eqx")
    _touch(tmp_path / "epoch_0007_ema.eqx")
    assert latest_checkpoint(tmp_path).endswith("epoch_0004.eqx")
    assert latest_checkpoint(tmp_path, prefer_ema=True).endswith(
        "epoch_0004.eqx"
    )


def test_latest_checkpoint_none_cases(tmp_path):
    assert latest_checkpoint(tmp_path / "missing_dir") is None
    assert latest_checkpoint(tmp_path) is None


def test_nonempty(tmp_path):
    f = tmp_path / "a.json"
    assert not nonempty(f)
    f.write_bytes(b"")
    assert not nonempty(f)  # zero-byte = failed write, not an artifact
    f.write_bytes(b"{}")
    assert nonempty(f)


# --- staged executor -----------------------------------------------------

def _mk_dirs(tmp_path, modes):
    out_root = tmp_path / "camp"
    dirs = {m: out_root / m for m in modes}
    for d in dirs.values():
        d.mkdir(parents=True)
    return out_root, dirs


def test_staged_campaign_full_pipeline(tmp_path):
    modes = ("physics", "sfno")
    out_root, dirs = _mk_dirs(tmp_path, modes)
    calls = {"train": [], "eval": [], "plot": []}

    def train_fn(mode):
        calls["train"].append(mode)
        _touch(dirs[mode] / "epoch_0000.eqx")

    def eval_fn(mode, ckpt):
        calls["eval"].append((mode, ckpt))
        _touch(dirs[mode] / "scorecard.json", b"{}")

    def plot_fn(family_scorecards):
        calls["plot"].append(sorted(family_scorecards))
        png = out_root / "scorecard.png"
        _touch(png)
        return str(png)

    result = run_staged_campaign(
        modes=modes,
        stages=("train", "eval", "plot"),
        out_root=str(out_root),
        train_fn=train_fn,
        eval_fn=eval_fn,
        plot_fn=plot_fn,
        mode_out_dir_fn=lambda m: str(dirs[m]),
        scorecard_path_fn=lambda m: str(dirs[m] / "scorecard.json"),
    )
    assert calls["train"] == list(modes)
    assert [m for m, _ in calls["eval"]] == list(modes)
    assert calls["plot"] == [["physics", "sfno"]]
    assert set(result) == set(modes)


def test_staged_campaign_missing_artifact_is_failure(tmp_path):
    out_root, dirs = _mk_dirs(tmp_path, ("physics",))

    def train_fn(mode):
        pass  # writes NO checkpoint (diverged/OOM analogue)

    with pytest.raises(SystemExit, match="missing"):
        run_staged_campaign(
            modes=("physics",),
            stages=("train", "eval"),
            out_root=str(out_root),
            train_fn=train_fn,
            eval_fn=lambda m, c: None,
            plot_fn=lambda fs: None,
            mode_out_dir_fn=lambda m: str(dirs[m]),
            scorecard_path_fn=lambda m: str(dirs[m] / "scorecard.json"),
        )


def test_staged_campaign_allow_missing_downgrades(tmp_path):
    out_root, dirs = _mk_dirs(tmp_path, ("physics",))
    result = run_staged_campaign(
        modes=("physics",),
        stages=("eval",),
        out_root=str(out_root),
        train_fn=lambda m: None,
        eval_fn=lambda m, c: None,
        plot_fn=lambda fs: None,
        mode_out_dir_fn=lambda m: str(dirs[m]),
        scorecard_path_fn=lambda m: str(dirs[m] / "scorecard.json"),
        allow_missing_artifacts=True,
    )
    assert result == {}


def test_staged_campaign_eval_uses_ema_when_preferred(tmp_path):
    out_root, dirs = _mk_dirs(tmp_path, ("sfno",))
    _touch(dirs["sfno"] / "epoch_0001.eqx")
    _touch(dirs["sfno"] / "epoch_0001_ema.eqx")
    seen = []

    def eval_fn(mode, ckpt):
        seen.append(ckpt)
        _touch(dirs[mode] / "scorecard.json", b"{}")

    run_staged_campaign(
        modes=("sfno",),
        stages=("eval",),
        out_root=str(out_root),
        train_fn=lambda m: None,
        eval_fn=eval_fn,
        plot_fn=lambda fs: None,
        mode_out_dir_fn=lambda m: str(dirs[m]),
        scorecard_path_fn=lambda m: str(dirs[m] / "scorecard.json"),
        prefer_ema=True,
    )
    assert seen and seen[0].endswith("epoch_0001_ema.eqx")
