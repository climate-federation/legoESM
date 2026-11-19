"""Append-only local catalogue of every download attempt."""
import datetime as dt
import json


def append(root, record):
    record = dict(record, recorded_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
    with open(root / "catalogue.jsonl", "a") as handle:
        handle.write(json.dumps(record) + "\n")


def read(root):
    path = root / "catalogue.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
