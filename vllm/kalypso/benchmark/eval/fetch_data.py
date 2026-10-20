"""Download the full public datasets behind the Kalypso paper's workloads.

    python -m vllm.kalypso.benchmark.eval.fetch_data            # everything
    python -m vllm.kalypso.benchmark.eval.fetch_data contractnli

Writes under ``vllm/kalypso/benchmark/data/`` (git-ignored), in the same
format as the paper's 10-row samples in ``sample_data/``:

- ``contract-nli/<split>/contracts/<n>.txt``: ContractNLI documents in
  dataset order (the paper's sample is ``train`` documents 0-9);
  ``hypotheses/<n>.txt``: the 17 hypotheses in label order, formatted like
  the sample; ``labels.csv``: the gold choice for every (contract,
  hypothesis) pair.
- ``medec/medec_train.csv``: the MEDEC-MS training set with the ``data``
  column the paper's sample uses (Kalypso's Text+Sentences template).

``manifest.json`` records each source URL, its SHA-256 and the row counts.
Both datasets are CC BY 4.0; their licence files are copied alongside.
"""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

CONTRACTNLI_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
MEDEC_URL = ("https://raw.githubusercontent.com/abachaa/MEDEC/main/MEDEC-MS/"
             "MEDEC-Full-TrainingSet-with-ErrorType.csv")


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hypothesis_text(label_id: str, label: dict) -> str:
    """Same layout as sample_data/contract-nli/hypotheses/*.txt."""
    return (f"LABEL_ID: {label_id}\n"
            f"SHORT_DESCRIPTION: {label['short_description']}\n\n"
            f"HYPOTHESIS:\n{label['hypothesis']}\n")


def write_contractnli_split(split: dict, out: Path) -> dict:
    """Write one ContractNLI split; return its counts."""
    contracts, hypotheses = out / "contracts", out / "hypotheses"
    contracts.mkdir(parents=True, exist_ok=True)
    hypotheses.mkdir(parents=True, exist_ok=True)
    label_ids = list(split["labels"])
    for n, label_id in enumerate(label_ids, start=1):
        (hypotheses / f"{n}.txt").write_text(
            hypothesis_text(label_id, split["labels"][label_id]), encoding="utf-8")
    with (out / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["contract", "hypothesis", "label_id", "choice", "document_id"])
        for n, doc in enumerate(split["documents"], start=1):
            (contracts / f"{n}.txt").write_text(doc["text"], encoding="utf-8")
            annotations = doc["annotation_sets"][0]["annotations"]
            for h, label_id in enumerate(label_ids, start=1):
                writer.writerow([n, h, label_id, annotations[label_id]["choice"], doc["id"]])
    return {"contracts": len(split["documents"]), "hypotheses": len(label_ids)}


def fetch_contractnli(root: Path) -> dict:
    raw = _download(CONTRACTNLI_URL)
    out = root / "contract-nli"
    counts = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in ("LICENSE", "TERMS", "README.md"):
            out.mkdir(parents=True, exist_ok=True)
            (out / name).write_bytes(archive.read(f"contract-nli/{name}"))
        for split in ("train", "dev", "test"):
            data = json.loads(archive.read(f"contract-nli/{split}.json"))
            counts[split] = write_contractnli_split(data, out / split)
    return {"url": CONTRACTNLI_URL, "sha256": _sha256(raw), "license": "CC BY 4.0",
            "counts": counts}


def medec_data_column(row: dict) -> str:
    """Kalypso's CSV reader template for Text+Sentences rows (data/data.py)."""
    return (f"Text ID: {str(row.get('Text ID', '')).strip()}\n\n"
            f"Clinical note:\n{str(row['Text']).strip()}\n\n"
            f"Numbered sentences:\n{str(row['Sentences']).strip()}")


def write_medec(raw: bytes, out: Path) -> int:
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "medec_train.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["data", *rows[0].keys()])
        writer.writeheader()
        for row in rows:
            writer.writerow({"data": medec_data_column(row), **row})
    return len(rows)


def fetch_medec(root: Path) -> dict:
    raw = _download(MEDEC_URL)
    rows = write_medec(raw, root / "medec")
    return {"url": MEDEC_URL, "sha256": _sha256(raw), "license": "CC BY 4.0",
            "counts": {"train": rows}}


FETCHERS = {"contractnli": fetch_contractnli, "medec": fetch_medec}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", choices=list(FETCHERS), default=[],
                        help="default: all")
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    names = args.datasets or list(FETCHERS)
    manifest_path = args.root / "manifest.json"
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists() else {})
    for name in names:
        print(f"fetching {name} ...", flush=True)
        entry = FETCHERS[name](args.root)
        entry["fetched"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        manifest[name] = entry
        print(f"  {entry['counts']}", flush=True)
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
