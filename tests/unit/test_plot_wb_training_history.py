"""The training-history plotter draws the per-term held-out loss figure."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "plot"))

import plot_wb_training_history as p  # noqa: E402


def _write(tmp_path, rows):
    h = tmp_path / "training_history.jsonl"
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return h


def test_term_series_keeps_epochs_aligned_with_missing_terms():
    rows = [{"epoch": 0, "train_loss": 1.0, "val_terms": {"mse_T": 0.5}},
            {"epoch": 1, "train_loss": 0.9,
             "val_terms": {"mse_T": 0.4, "crps_q": 0.1}}]
    assert p.term_series(rows) == {"crps_q": [None, 0.1], "mse_T": [0.5, 0.4]}


def test_terms_figure_written_only_when_terms_logged(tmp_path):
    rows = [{"epoch": e, "train_loss": 1.0 - 0.1 * e, "val_loss": 1.0,
             "val_terms": {"mse_T": 0.5 - 0.1 * e, "crps_ps": 0.2, "mse_q": 0.0},
             "params": {}}
            for e in range(3)]
    h = _write(tmp_path, rows)
    out = subprocess.run([sys.executable, str(ROOT / "scripts/plot/plot_wb_training_history.py"),
                          "--history", str(h)], check=True, capture_output=True, text=True)
    assert (tmp_path / "training_history_terms.png").stat().st_size > 0
    assert "always zero (not drawn): mse_q" in out.stdout

    old = tmp_path / "old"
    old.mkdir()
    (old / "training_history_terms.png").write_bytes(b"stale")
    h = _write(old, [{"epoch": 0, "train_loss": 1.0, "params": {}}])
    subprocess.run([sys.executable, str(ROOT / "scripts/plot/plot_wb_training_history.py"),
                    "--history", str(h)], check=True)
    assert not (old / "training_history_terms.png").exists()
