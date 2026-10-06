"""Workloads for the evaluation harness.

A workload is the rows, instruction and (when known) labels every arm runs
over. Rows are read the way Kalypso reads them (``vllm/kalypso/data/data.py``)
so the direct-vLLM arm can rebuild byte-identical prompts:

- a directory of ``*.txt``: sorted by integer stem, text stripped, idx = stem;
- a CSV: the ``data`` column (else ``Text``), stripped, idx = 0-based row
  position.
"""

import csv
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vllm.kalypso.benchmark import scenarios

SAMPLE_DATA = Path(__file__).resolve().parents[1] / "sample_data"
BIODEX_ARTICLES = SAMPLE_DATA / "articles_500"
REVIEWS = SAMPLE_DATA / "block_filter" / "reviews_20.csv"
MEDEC = SAMPLE_DATA / "MEDEC-TrainingSet-1000.csv"
JOIN_DIR = SAMPLE_DATA / "block_join"

# Same instructions as the standalone block checks, so results line up.
REVIEWS_INSTRUCTION = "Does the review complain about the restaurant?"
JOIN_INSTRUCTION = "The complaint in the left item belongs to the category in the right item."

MAP, FILTER, JOIN = "map", "filter", "join"


@dataclass(frozen=True)
class Row:
    idx: int
    text: str


@dataclass(frozen=True)
class Workload:
    name: str
    kind: str                       # MAP, FILTER or JOIN
    data_path: Path                 # what Kalypso's /v1/semantic/query reads
    instruction: str
    rows: list[Row]
    right_path: Path | None = None  # JOIN only
    right_rows: list[Row] | None = None
    # FILTER: idxs that should pass; JOIN: (left idx, right idx) matches;
    # None when the workload has no labels.
    positives: frozenset | None = None


def _txt_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.txt"), key=lambda p: int(p.stem))


def _subset_dir(files: list[Path]) -> Path:
    """Copy ``files`` into a fresh directory, keeping names (and so idx).

    Kalypso reads every ``*.txt`` in ``data_path``, so a subset needs its own
    directory. Inline ``records`` are not an option: they go through the CSV
    reader, which builds a different prompt than the directory reader.
    """
    target = Path(tempfile.mkdtemp(prefix="kalypso_eval_"))
    for path in files:
        shutil.copyfile(path, target / path.name)
    return target


def _csv_text(record: dict) -> str:
    # Kalypso's CSV reader also handles Text+Sentences; none of our files
    # need it, so refuse rather than silently build a different prompt.
    if "data" in record:
        return str(record["data"]).strip()
    if "Text" in record and "Sentences" not in record:
        return str(record["Text"]).strip()
    raise ValueError(f"unsupported CSV columns {sorted(record)}")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _limit(n: int | None, total: int, name: str) -> int:
    if n is None:
        return total
    if not 1 <= n <= total:
        raise ValueError(f"--n must be between 1 and {total} for {name}")
    return n


def load_biodex(n: int | None = None) -> Workload:
    """BioDEX reaction extraction (map). No labels for this subset."""
    all_files = _txt_files(BIODEX_ARTICLES)
    if not all_files:
        raise FileNotFoundError(f"no *.txt articles under {BIODEX_ARTICLES}")
    files = all_files[:_limit(n, len(all_files), "biodex")]
    data_path = BIODEX_ARTICLES if len(files) == len(all_files) else _subset_dir(files)
    return Workload(
        name="biodex", kind=MAP, data_path=data_path.resolve(),
        instruction=scenarios.BIODEX_MAP_REACTIONS,
        rows=[Row(int(p.stem), p.read_text(encoding="utf-8").strip()) for p in files],
    )


def _csv_filter(name: str, path: Path, instruction: str, is_positive, n: int | None) -> Workload:
    records = _read_csv(path)
    if n is not None and _limit(n, len(records), name) < len(records):
        raise ValueError(f"{name} is too small to subset; run all {len(records)} rows")
    rows = [Row(i, _csv_text(r)) for i, r in enumerate(records)]
    return Workload(
        name=name, kind=FILTER, data_path=path.resolve(), instruction=instruction,
        rows=rows,
        positives=frozenset(i for i, r in enumerate(records) if is_positive(r)),
    )


def load_reviews(n: int | None = None) -> Workload:
    """20 hand-written restaurant reviews; label = complains about the restaurant."""
    return _csv_filter("reviews", REVIEWS, REVIEWS_INSTRUCTION,
                       lambda r: r["label"].strip().lower() == "true", n)


def load_medec(n: int | None = None) -> Workload:
    """MEDEC clinical notes; label = the note contains a medical error."""
    return _csv_filter("medec", MEDEC, scenarios.MEDEC_ERROR_FILTER,
                       lambda r: r["Error Flag"].strip() == "1", n)


def load_join5x5(n: int | None = None) -> Workload:
    """5 complaints x 5 categories; label = the expected (left, right) matches."""
    if n is not None:
        raise ValueError("join5x5 has a fixed size; drop --n")
    left = _read_csv(JOIN_DIR / "left_5.csv")
    right = _read_csv(JOIN_DIR / "right_5.csv")
    # matches_5x5.csv uses the files' id columns; Kalypso uses row position.
    left_pos = {r["id"]: i for i, r in enumerate(left)}
    right_pos = {r["id"]: i for i, r in enumerate(right)}
    matches = frozenset(
        (left_pos[m["left_id"]], right_pos[m["right_id"]])
        for m in _read_csv(JOIN_DIR / "matches_5x5.csv"))
    return Workload(
        name="join5x5", kind=JOIN,
        data_path=(JOIN_DIR / "left_5.csv").resolve(),
        right_path=(JOIN_DIR / "right_5.csv").resolve(),
        instruction=JOIN_INSTRUCTION,
        rows=[Row(i, _csv_text(r)) for i, r in enumerate(left)],
        right_rows=[Row(i, _csv_text(r)) for i, r in enumerate(right)],
        positives=matches,
    )


WORKLOADS = {
    "biodex": load_biodex,
    "reviews": load_reviews,
    "medec": load_medec,
    "join5x5": load_join5x5,
}
