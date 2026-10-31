"""Workloads for the evaluation harness.

A workload is the rows, instruction and (when known) labels every arm runs
over. Rows are read the way Kalypso reads them (``vllm/kalypso/data/data.py``)
so the direct-vLLM arm can rebuild byte-identical prompts:

- a directory of ``*.txt``: sorted by integer stem, text stripped, idx = stem;
- a CSV: the ``data`` column (else ``Text``), stripped, idx = 0-based row
  position.

SemBench's movie scenario (github.com/skddj/sembench) is downloaded on first
use, pinned to one commit and checked by SHA-256, into the git-ignored
``benchmark/data/sembench/movie/``.
"""

import csv
import hashlib
import html
import io
import json
import random
import shutil
import tempfile
import urllib.request
import zipfile
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

SEMBENCH_MOVIE = Path(__file__).resolve().parents[1] / "data" / "sembench" / "movie"
SEMBENCH_COMMIT = "6334161d3865c120674335fd8dc98abcbac173e3"
SEMBENCH_REVIEWS_URL = (f"https://raw.githubusercontent.com/skddj/sembench/{SEMBENCH_COMMIT}/"
                        "files/movie/data/sf_2000/Reviews.csv")
SEMBENCH_REVIEWS_SHA256 = "df231cfb189c9e169aa6afe415e4fa24533b04499179bbd61416c4c6af28ca21"
# SemBench's own prompts (files/movie/query/bigquery/Q1.sql, Q6.sql).
MOVIE_POSITIVE_INSTRUCTION = "Determine if the following movie review is clearly positive."
MOVIE_OPPOSITE_INSTRUCTION = ("These two movie reviews express opposite sentiments - "
                              "one is positive and the other is negative.")

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


def _sembench_reviews() -> list[dict]:
    """SemBench movie reviews (scale factor 2000), one row per reviewId.

    The file repeats some reviews verbatim (every Ant-Man review twice), so
    only the first copy of each reviewId is kept. Review text is HTML-escaped
    in the source (``&#44;``) and is unescaped here.
    """
    path = SEMBENCH_MOVIE / "Reviews.csv"
    if not path.exists():
        raw = urllib.request.urlopen(SEMBENCH_REVIEWS_URL, timeout=300).read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != SEMBENCH_REVIEWS_SHA256:
            raise ValueError(f"{SEMBENCH_REVIEWS_URL} has SHA-256 {digest}, "
                             f"expected {SEMBENCH_REVIEWS_SHA256}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    rows, seen = [], set()
    for record in _read_csv(path):
        if record["reviewId"] not in seen:
            seen.add(record["reviewId"])
            rows.append({**record, "reviewText": html.unescape(record["reviewText"]).strip()})
    return rows


def _movie_csv(name: str, reviews: list[dict]) -> Path:
    """Write ``reviews`` as a ``data``-column CSV that Kalypso reads."""
    path = SEMBENCH_MOVIE / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["data", "reviewId"])
        writer.writerows([r["reviewText"], r["reviewId"]] for r in reviews)
    return path.resolve()


def _movie_reviews(name: str, movie: str | None, n: int | None) -> list[dict]:
    reviews = [r for r in _sembench_reviews() if movie is None or r["id"] == movie]
    return reviews[:_limit(n, len(reviews), name)]


def _positive(review: dict) -> bool:
    return review["scoreSentiment"] == "POSITIVE"


def _movie_filter(name: str, movie: str | None, n: int | None) -> Workload:
    reviews = _movie_reviews(name, movie, n)
    return Workload(
        name=name, kind=FILTER,
        data_path=_movie_csv(f"{name}_{len(reviews)}", reviews),
        instruction=MOVIE_POSITIVE_INSTRUCTION,
        rows=[Row(i, r["reviewText"]) for i, r in enumerate(reviews)],
        positives=frozenset(i for i, r in enumerate(reviews) if _positive(r)),
    )


def load_movie_positive(n: int | None = None) -> Workload:
    """SemBench movie Q1: clearly positive reviews, any movie (1,865 reviews)."""
    return _movie_filter("movie_positive", None, n)


def load_movie_taken3(n: int | None = None) -> Workload:
    """SemBench movie Q2-Q4/Q8: positive reviews of Taken 3 (14 of 120)."""
    return _movie_filter("movie_taken3", "taken_3", n)


def load_movie_opposite(n: int | None = None) -> Workload:
    """SemBench movie Q6/Q7: Ant-Man review pairs of opposite sentiment.

    A self-join of the first ``n`` of the movie's 128 reviews (default all:
    128 x 128 pairs). SemBench excludes a review paired with itself; here
    those pairs stay in, labeled negative, since they never differ.
    """
    name = "movie_opposite"
    reviews = _movie_reviews(name, "ant_man_and_the_wasp_quantumania", n)
    path = _movie_csv(f"{name}_{len(reviews)}", reviews)
    rows = [Row(i, r["reviewText"]) for i, r in enumerate(reviews)]
    return Workload(
        name=name, kind=JOIN, data_path=path, right_path=path,
        instruction=MOVIE_OPPOSITE_INSTRUCTION,
        rows=rows, right_rows=rows,
        positives=frozenset((i, j) for i, a in enumerate(reviews)
                            for j, b in enumerate(reviews) if _positive(a) != _positive(b)),
    )


# Fixed-selectivity versions of movie_positive: SELECTIVITY_ROWS reviews, of
# which a share s is positive, drawn with a fixed seed and shuffled so
# positives and negatives mix inside every block. The 448 negative reviews
# bound s from below: 400 rows at s = 0.10 need 360 of them.
SELECTIVITY_ROWS = 400
SELECTIVITY_SEED = 0
SELECTIVITIES = {"movie_sel10": 0.10, "movie_sel25": 0.25, "movie_sel50": 0.50,
                 "movie_sel75": 0.75, "movie_sel90": 0.90}


def load_movie_selectivity(name: str, n: int | None = None) -> Workload:
    """SemBench movie Q1 with a set share of positive reviews (``SELECTIVITIES``).

    ``--n`` keeps the first n of the shuffled rows, so s holds only roughly.
    """
    reviews = _sembench_reviews()
    positives = [r for r in reviews if _positive(r)]
    negatives = [r for r in reviews if not _positive(r)]
    want = round(SELECTIVITIES[name] * SELECTIVITY_ROWS)
    if want > len(positives) or SELECTIVITY_ROWS - want > len(negatives):
        raise ValueError(f"{name}: not enough reviews for {SELECTIVITY_ROWS} rows")
    rng = random.Random(SELECTIVITY_SEED)
    chosen = rng.sample(positives, want) + rng.sample(negatives, SELECTIVITY_ROWS - want)
    rng.shuffle(chosen)
    chosen = chosen[:_limit(n, len(chosen), name)]
    return Workload(
        name=name, kind=FILTER,
        data_path=_movie_csv(f"{name}_{len(chosen)}", chosen),
        instruction=MOVIE_POSITIVE_INSTRUCTION,
        rows=[Row(i, r["reviewText"]) for i, r in enumerate(chosen)],
        positives=frozenset(i for i, r in enumerate(chosen) if _positive(r)),
    )


# ContractNLI (Stanford, CC BY 4.0): long rows (NDAs, ~1.9k tokens on
# average). Downloaded on first use into the git-ignored benchmark/data/.
CONTRACTNLI_DIR = Path(__file__).resolve().parents[1] / "data" / "contract-nli"
CONTRACTNLI_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
# Longer contracts are left out so a block of 8 rows still fits a 32k context
# (12,000 characters is about 3k tokens).
CNLI_MAX_CHARS = 12000
CONTRACTNLI_ENTAILMENT = ("Does the contract entail the hypothesis? Answer yes only if "
                          "the contract clearly supports the hypothesis.")


def _contractnli() -> tuple[list[dict], dict[str, str]]:
    """ContractNLI documents (train, dev, test; at most CNLI_MAX_CHARS) and
    the 17 hypotheses.

    Each document is ``{"id", "text", "choices": {label_id: choice}}`` with
    choice Entailment, Contradiction or NotMentioned.
    """
    path = CONTRACTNLI_DIR / "contract-nli.zip"
    if not path.exists():
        raw = urllib.request.urlopen(CONTRACTNLI_URL, timeout=300).read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    docs, hypotheses = [], {}
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        for split in ("train", "dev", "test"):
            data = json.loads(archive.read(f"contract-nli/{split}.json"))
            hypotheses.update({k: v["hypothesis"] for k, v in data["labels"].items()})
            for doc in data["documents"]:
                if len(doc["text"]) > CNLI_MAX_CHARS:
                    continue
                annotations = doc["annotation_sets"][0]["annotations"]
                docs.append({"id": f"{split}-{doc['id']}", "text": doc["text"].strip(),
                             "choices": {k: a["choice"] for k, a in annotations.items()}})
    return docs, hypotheses


def _cnli_csv(name: str, texts: list[str]) -> Path:
    path = CONTRACTNLI_DIR / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["data"])
        writer.writerows([t] for t in texts)
    return path.resolve()


# Fixed-selectivity ContractNLI filter, the long-row counterpart of
# movie_sel*: SELECTIVITY_ROWS rows, a share s of them entailed. A row is one
# distinct contract plus one of its hypotheses (an entailed one for a
# positive row, a contradicted or not-mentioned one for a negative row), so no
# two rows share a contract and the prefix cache cannot help across rows.
CNLI_SELECTIVITIES = {"cnli_sel10": 0.10, "cnli_sel25": 0.25, "cnli_sel50": 0.50,
                      "cnli_sel75": 0.75, "cnli_sel90": 0.90}


def _cnli_row(text: str, hypothesis: str) -> str:
    return f"Contract:\n{text}\n\nHypothesis: {hypothesis}"


def load_cnli_selectivity(name: str, n: int | None = None) -> Workload:
    """ContractNLI entailment filter with a set share of entailed rows.

    ``--n`` keeps the first n of the shuffled rows, so s holds only roughly.
    """
    docs, hypotheses = _contractnli()
    rng = random.Random(SELECTIVITY_SEED)
    docs = rng.sample(docs, len(docs))
    want = round(CNLI_SELECTIVITIES[name] * SELECTIVITY_ROWS)
    rows = []  # (text, is_positive)
    for doc in docs:
        positive = len([r for r in rows if r[1]]) < want
        negative = len(rows) - len([r for r in rows if r[1]]) < SELECTIVITY_ROWS - want
        if not (positive or negative):
            break
        entailed = sorted(k for k, c in doc["choices"].items() if c == "Entailment")
        other = sorted(k for k, c in doc["choices"].items() if c != "Entailment")
        if positive and entailed:
            rows.append((_cnli_row(doc["text"], hypotheses[rng.choice(entailed)]), True))
        elif negative and other:
            rows.append((_cnli_row(doc["text"], hypotheses[rng.choice(other)]), False))
    if len(rows) < SELECTIVITY_ROWS:
        raise ValueError(f"{name}: only {len(rows)} usable contracts for {SELECTIVITY_ROWS} rows")
    rng.shuffle(rows)
    rows = rows[:_limit(n, len(rows), name)]
    return Workload(
        name=name, kind=FILTER,
        data_path=_cnli_csv(f"{name}_{len(rows)}", [t for t, _ in rows]),
        instruction=CONTRACTNLI_ENTAILMENT,
        rows=[Row(i, t) for i, (t, _) in enumerate(rows)],
        positives=frozenset(i for i, (_, p) in enumerate(rows) if p),
    )


# ContractNLI join: contracts x hypotheses, matched when the contract entails
# the hypothesis. Selectivity is set by which hypotheses form the right table:
# the CNLI_JOIN_RIGHT hypotheses entailed least often (low) or most often
# (high) across the dataset. ``--n`` is the number of contracts (left rows).
CNLI_JOIN_LEFT = 24
CNLI_JOIN_RIGHT = 8
CNLI_JOINS = ("cnli_join_low", "cnli_join_high")


def load_cnli_join(name: str, n: int | None = None) -> Workload:
    docs, hypotheses = _contractnli()
    rate = {k: sum(d["choices"].get(k) == "Entailment" for d in docs) / len(docs)
            for k in hypotheses}
    ranked = sorted(hypotheses, key=lambda k: (rate[k], k))
    right = ranked[:CNLI_JOIN_RIGHT] if name == "cnli_join_low" else ranked[-CNLI_JOIN_RIGHT:]
    left = random.Random(SELECTIVITY_SEED).sample(docs, len(docs))
    left = left[:_limit(CNLI_JOIN_LEFT if n is None else n, len(left), name)]
    positives = frozenset((i, j) for i, d in enumerate(left) for j, k in enumerate(right)
                          if d["choices"].get(k) == "Entailment")
    return Workload(
        name=name, kind=JOIN,
        data_path=_cnli_csv(f"{name}_left_{len(left)}", [d["text"] for d in left]),
        right_path=_cnli_csv(f"{name}_right", [hypotheses[k] for k in right]),
        instruction=scenarios.CONTRACT_NLI_ENTAILMENT_JOIN,
        rows=[Row(i, d["text"]) for i, d in enumerate(left)],
        right_rows=[Row(j, hypotheses[k]) for j, k in enumerate(right)],
        positives=positives,
    )


WORKLOADS = {
    "biodex": load_biodex,
    "reviews": load_reviews,
    "medec": load_medec,
    "join5x5": load_join5x5,
    "movie_positive": load_movie_positive,
    "movie_taken3": load_movie_taken3,
    "movie_opposite": load_movie_opposite,
    **{name: (lambda n=None, name=name: load_movie_selectivity(name, n))
       for name in SELECTIVITIES},
    **{name: (lambda n=None, name=name: load_cnli_selectivity(name, n))
       for name in CNLI_SELECTIVITIES},
    **{name: (lambda n=None, name=name: load_cnli_join(name, n)) for name in CNLI_JOINS},
}
