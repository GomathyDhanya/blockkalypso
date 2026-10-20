"""Workloads for the evaluation harness.

A workload is the rows, instruction and (when known) labels every arm runs
over. Rows are read the way Kalypso reads them (``vllm/kalypso/data/data.py``)
so the direct-vLLM arm can rebuild byte-identical prompts:

- a directory of ``*.txt``: sorted by integer stem, text stripped, idx = stem;
- a CSV: the ``data`` column (else ``Text``), stripped, idx = 0-based row
  position.
"""

import csv
import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vllm.kalypso.benchmark import scenarios

SAMPLE_DATA = Path(__file__).resolve().parents[1] / "sample_data"
# Full public datasets, written by fetch_data.py (git-ignored).
FULL_DATA = Path(__file__).resolve().parents[1] / "data"
CONTRACTNLI = FULL_DATA / "contract-nli" / "train"
MEDEC_FULL = FULL_DATA / "medec" / "medec_train.csv"
ARXIV = FULL_DATA / "arxiv" / "papers.csv"
BIODEX_ARTICLES = SAMPLE_DATA / "articles_500"
REVIEWS = SAMPLE_DATA / "block_filter" / "reviews_20.csv"
MEDEC = SAMPLE_DATA / "MEDEC-TrainingSet-1000.csv"
JOIN_DIR = SAMPLE_DATA / "block_join"

# Same instructions as the standalone block checks, so results line up.
REVIEWS_INSTRUCTION = "Does the review complain about the restaurant?"
JOIN_INSTRUCTION = "The complaint in the left item belongs to the category in the right item."

MAP, FILTER, JOIN, PIPELINE = "map", "filter", "join", "pipeline"


@dataclass(frozen=True)
class Row:
    idx: int
    text: str


@dataclass(frozen=True)
class Workload:
    name: str
    kind: str                       # MAP, FILTER, JOIN or PIPELINE
    data_path: Path                 # what Kalypso's /v1/semantic/query reads
    instruction: str
    rows: list[Row]
    right_path: Path | None = None  # JOIN and PIPELINE (the join's right table)
    right_rows: list[Row] | None = None
    # FILTER: idxs that should pass; JOIN: (left idx, right idx) matches;
    # None when the workload has no labels.
    positives: frozenset | None = None
    # Which source rows the workload uses (e.g. MEDEC Text IDs), so a run
    # can be reproduced from the full dataset.
    source_ids: tuple | None = None
    # PIPELINE only: the query's stages in order, as dicts with "op"
    # (sem_filter, join or sem_map) and that op's prompt/instruction. A
    # pipeline is scored on the (left, right) pairs it outputs, like a join.
    stages: tuple | None = None
    # Arms to run when --arms is not given; None means arms.DEFAULT_ARMS.
    default_arms: tuple | None = None
    # PIPELINE only: how many rows the first filter should pass per the
    # labels, used to spot runs whose model answers changed the workload.
    gold_filter_pass: int | None = None


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


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run: python -m vllm.kalypso.benchmark.eval.fetch_data")
    return path


def load_contractnli(n: int | None = None) -> Workload:
    """ContractNLI join: the first ``n`` train contracts x all 17 hypotheses.

    Contracts are taken in dataset order, as the paper's 10-row sample is
    (train documents 0-9). Labels: a pair matches when its gold choice is
    Entailment, which is what the paper's join prompt asks.
    """
    contracts = _txt_files(_require(CONTRACTNLI) / "contracts")
    hypotheses = _txt_files(CONTRACTNLI / "hypotheses")
    n = 20 if n is None else n
    if not 1 <= n <= len(contracts):
        raise ValueError(f"--n must be between 1 and {len(contracts)} for contractnli")
    chosen = contracts[:n]
    keep = {int(p.stem) for p in chosen}
    with (CONTRACTNLI / "labels.csv").open(newline="", encoding="utf-8") as f:
        positives = frozenset(
            (int(r["contract"]), int(r["hypothesis"])) for r in csv.DictReader(f)
            if r["choice"] == "Entailment" and int(r["contract"]) in keep)
    return Workload(
        name="contractnli", kind=JOIN,
        data_path=_subset_dir(chosen).resolve(),
        right_path=(CONTRACTNLI / "hypotheses").resolve(),
        instruction=scenarios.CONTRACT_NLI_ENTAILMENT_JOIN,
        rows=[Row(int(p.stem), p.read_text(encoding="utf-8").strip()) for p in chosen],
        right_rows=[Row(int(p.stem), p.read_text(encoding="utf-8").strip()) for p in hypotheses],
        positives=positives,
        source_ids=tuple(int(p.stem) for p in chosen),
    )


def load_contractnli_filter(n: int | None = None) -> Workload:
    """The paper's ContractNLI filter (CONTRACT_NLI_VALID_CONTRACT) on its own.

    The first ``n`` train contracts (default 64), in dataset order. Every
    document is a real NDA, so the gold answer is yes for all of them. Used
    as the long-row case of the filter experiments (median ~1.9k tokens).
    """
    contracts = _txt_files(_require(CONTRACTNLI) / "contracts")
    n = 64 if n is None else n
    if not 1 <= n <= len(contracts):
        raise ValueError(f"--n must be between 1 and {len(contracts)} for contractnli_filter")
    chosen = contracts[:n]
    rows = [Row(int(p.stem), p.read_text(encoding="utf-8").strip()) for p in chosen]
    return Workload(
        name="contractnli_filter", kind=FILTER, data_path=_subset_dir(chosen).resolve(),
        instruction=scenarios.CONTRACT_NLI_VALID_CONTRACT, rows=rows,
        positives=frozenset(r.idx for r in rows),
        source_ids=tuple(int(p.stem) for p in chosen),
    )


def load_contractnli_pipeline(n: int | None = None, with_map: bool = True) -> Workload:
    """The paper's ContractNLI query: filter -> join -> map.

    Stages and prompts as in client_contract_nli_filter_join_map.py: keep
    valid contracts, join them with the 17 hypotheses they entail, explain
    each entailment. ``with_map=False`` stops after the join. Scored on the
    output pairs against the gold Entailment labels; every ContractNLI
    document is an NDA, so the filter's gold answer is yes for all of them.
    """
    join = load_contractnli(n)
    stages = (
        {"op": "sem_filter", "prompt": scenarios.CONTRACT_NLI_VALID_CONTRACT},
        {"op": "join", "instruction": scenarios.CONTRACT_NLI_ENTAILMENT_JOIN},
    )
    if with_map:
        stages += ({"op": "sem_map", "prompt": scenarios.CONTRACT_NLI_EXPLAIN_ENTAILMENT},)
    return Workload(
        name="contractnli_fjm" if with_map else "contractnli_fj", kind=PIPELINE,
        data_path=join.data_path, right_path=join.right_path,
        instruction=scenarios.CONTRACT_NLI_ENTAILMENT_JOIN,
        rows=join.rows, right_rows=join.right_rows, positives=join.positives,
        source_ids=join.source_ids, stages=stages, gold_filter_pass=len(join.rows),
        default_arms=("kalypso_per_tuple", "kalypso_block:4/1x1",
                      "kalypso_block:1/1x17", "kalypso_block:4/1x17"),
    )


def load_medec_full(n: int | None = None, seed: int = 0) -> Workload:
    """MEDEC-MS training notes: a seeded random sample of ``n`` (default 64).

    The paper's prompt (``MEDEC_ERROR_FILTER``) and row text (the ``data``
    column); label = the note contains an error. The sample keeps roughly the
    dataset's own share of erroneous notes (56%).
    """
    with _require(MEDEC_FULL).open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    n = 64 if n is None else n
    if not 1 <= n <= len(records):
        raise ValueError(f"--n must be between 1 and {len(records)} for medec_full")
    chosen = sorted(random.Random(seed).sample(range(len(records)), n))
    rows = [records[i] for i in chosen]
    target = Path(tempfile.mkdtemp(prefix="kalypso_eval_")) / "medec.csv"
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(rows)
    return Workload(
        name="medec_full", kind=FILTER, data_path=target.resolve(),
        instruction=scenarios.MEDEC_ERROR_FILTER,
        rows=[Row(i, _csv_text(r)) for i, r in enumerate(rows)],
        positives=frozenset(i for i, r in enumerate(rows) if r["Error Flag"].strip() == "1"),
        source_ids=tuple(r["Text ID"] for r in rows),
    )


def _arxiv_pool():
    """arXiv papers by group: "robotics/<topic>" and "other/<area>"."""
    with _require(ARXIV).open(newline="", encoding="utf-8") as f:
        papers = list(csv.DictReader(f))
    groups = {}
    for paper in papers:
        groups.setdefault(paper["group"], []).append(paper)
    return groups


def _arxiv_text(paper: dict) -> str:
    return f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"


class _Draw:
    """Seeded draws without replacement from each arXiv group."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.groups = {g: self.rng.sample(ps, len(ps)) for g, ps in sorted(_arxiv_pool().items())}

    @property
    def topics(self) -> list[str]:
        return [g for g in self.groups if g.startswith("robotics/")]

    def take(self, group: str) -> dict:
        if not self.groups[group]:
            raise ValueError(f"not enough arXiv papers in {group}; lower n, s or f")
        return self.groups[group].pop()

    def robotics(self, count: int) -> list[dict]:
        """``count`` robotics papers, spread evenly over the topics."""
        topics = self.topics
        return [self.take(topics[i % len(topics)]) for i in range(count)]

    def other(self, count: int) -> list[dict]:
        areas = [g for g in self.groups if g.startswith("other/")]
        return [self.take(areas[i % len(areas)]) for i in range(count)]


def _arxiv_dir(papers: list[dict]) -> Path:
    target = Path(tempfile.mkdtemp(prefix="kalypso_eval_"))
    for i, paper in enumerate(papers, start=1):
        (target / f"{i}.txt").write_text(_arxiv_text(paper), encoding="utf-8")
    return target


def _arxiv_rows(papers):
    return [Row(i, _arxiv_text(p)) for i, p in enumerate(papers, start=1)]


def _arxiv_left(draw: _Draw, n: int, s: float) -> list[dict]:
    """``n`` papers, a share ``s`` of them robotics, in a seeded order."""
    if not 0 <= s <= 1:
        raise ValueError("s must be between 0 and 1")
    papers = draw.robotics(round(s * n)) + draw.other(n - round(s * n))
    draw.rng.shuffle(papers)
    return papers


def _arxiv_right(draw: _Draw, size: int, f: float) -> list[dict]:
    """``size`` papers giving each robotics left paper ``f`` matches on average.

    f >= 1: round(f) papers from every robotics topic. f < 1: one paper from
    round(f * topics) topics, so only that share of left papers has a match.
    The rest are non-robotics papers, which match nothing.
    """
    topics = draw.topics
    if f >= 1:
        matching = [draw.take(t) for t in topics for _ in range(round(f))]
    else:
        matching = [draw.take(t) for t in topics[:round(f * len(topics))]]
    if len(matching) > size:
        raise ValueError(f"f={f} needs {len(matching)} right papers; raise right")
    papers = matching + draw.other(size - len(matching))
    draw.rng.shuffle(papers)
    return papers


_DATASET = re.compile(r"\bdatasets?\b", re.IGNORECASE)
_IMAGERY = re.compile(r"\bimages?\b|\bcamera|\bRGB\b|\bvisual\b|\bvideo", re.IGNORECASE)


def _mentions_image_dataset(paper: dict) -> bool:
    """Proxy for CASE_3_FILTER_3: the abstract names a dataset and imagery."""
    return bool(_DATASET.search(paper["abstract"]) and _IMAGERY.search(paper["abstract"]))


def _same_topic(left, right):
    return frozenset(
        (i, j) for i, l in enumerate(left, start=1) for j, r in enumerate(right, start=1)
        if l["group"].startswith("robotics/") and l["group"] == r["group"])


# Kalypso's join template asks whether two contexts "together satisfy the
# condition"; the paper's arXiv join condition is a question. ``join_prompt=
# "statement"`` restates it as a statement, the form the paper's ContractNLI
# join uses ("... Answer true only if ...").
ARXIV_JOIN_PROMPTS = {
    "paper": scenarios.CASE_2_JOIN_ARXIV,
    "statement": ("The two papers study the same research topic. "
                  "Answer true only if both papers clearly address the same topic."),
    # Qwen2.5-7B answered false to every pair under both wordings above, even
    # two drone papers: it reads "topic" much more narrowly than the labels.
    # These two state the relation at the level the labels are built on.
    "related": "The two papers are on related research topics.",
    "task": ("Both papers work on the same type of robot or robotic task, "
             "for example both on drones or both on grasping."),
}


def _join_prompt(name: str) -> str:
    if name not in ARXIV_JOIN_PROMPTS:
        raise ValueError(f"join_prompt must be one of {sorted(ARXIV_JOIN_PROMPTS)}")
    return ARXIV_JOIN_PROMPTS[name]


def load_arxiv_filter(n: int | None = None, s: float = 0.5, seed: int = 0) -> Workload:
    """The paper's arXiv filter (CASE_2_FILTER_ARXIV) over ``n`` abstracts.

    Proxy label: the paper's primary category is cs.RO.
    """
    papers = _arxiv_left(_Draw(seed), 64 if n is None else n, s)
    return Workload(
        name="arxiv_filter", kind=FILTER, data_path=_arxiv_dir(papers).resolve(),
        instruction=scenarios.CASE_2_FILTER_ARXIV, rows=_arxiv_rows(papers),
        positives=frozenset(i for i, p in enumerate(papers, start=1)
                            if p["group"].startswith("robotics/")),
        source_ids=tuple(p["id"] for p in papers),
    )


def load_arxiv_join(n: int | None = None, right: int = 8, f: float = 1.0,
                    seed: int = 0, join_prompt: str = "paper") -> Workload:
    """The paper's arXiv join (CASE_2_JOIN_ARXIV): ``n`` robotics papers x ``right``.

    Proxy label: both papers are robotics papers on the same task named in
    their titles (fetch_data.ARXIV_TOPICS, e.g. both about grasping).
    """
    draw = _Draw(seed)
    left = draw.robotics(16 if n is None else n)
    draw.rng.shuffle(left)
    right_papers = _arxiv_right(draw, right, f)
    return Workload(
        name="arxiv_join", kind=JOIN, data_path=_arxiv_dir(left).resolve(),
        right_path=_arxiv_dir(right_papers).resolve(),
        instruction=_join_prompt(join_prompt),
        rows=_arxiv_rows(left), right_rows=_arxiv_rows(right_papers),
        positives=_same_topic(left, right_papers),
        source_ids=tuple(p["id"] for p in left + right_papers),
    )


def load_arxiv_pipeline(n: int | None = None, s: float = 0.5, f: float = 1.0,
                        right: int = 32, seed: int = 0,
                        join_prompt: str = "paper") -> Workload:
    """The paper's arXiv FILTER -> JOIN -> FILTER query (handoff's F1 -> Join -> F2).

    F1: CASE_2_FILTER_ARXIV (AI robotics?), join: CASE_2_JOIN_ARXIV (similar
    topic?) against ``right`` papers, F2: CASE_3_FILTER_3 (uses an image
    dataset?). ``s`` sets F1's selectivity and ``f`` the join fanout. Proxy
    labels: F1 = cs.RO; join = same robotics task in the titles; F2 = either
    abstract names a dataset and imagery (images, camera, RGB, video).
    """
    draw = _Draw(seed)
    left = _arxiv_left(draw, 32 if n is None else n, s)
    right_papers = _arxiv_right(draw, right, f)
    joined = _same_topic(left, right_papers)
    return Workload(
        name="arxiv_fjf", kind=PIPELINE, data_path=_arxiv_dir(left).resolve(),
        right_path=_arxiv_dir(right_papers).resolve(),
        instruction=_join_prompt(join_prompt),
        rows=_arxiv_rows(left), right_rows=_arxiv_rows(right_papers),
        positives=frozenset((i, j) for i, j in joined
                            if _mentions_image_dataset(left[i - 1])
                            or _mentions_image_dataset(right_papers[j - 1])),
        source_ids=tuple(p["id"] for p in left + right_papers),
        stages=({"op": "sem_filter", "prompt": scenarios.CASE_2_FILTER_ARXIV},
                {"op": "join", "instruction": _join_prompt(join_prompt)},
                {"op": "sem_filter", "prompt": scenarios.CASE_3_FILTER_3}),
        # Handoff E4: one block size k for F1, the join (k x k) and F2.
        default_arms=("kalypso_per_tuple",) + tuple(
            f"kalypso_block:{k}/{k}x{k}/{k}" for k in (2, 4, 8, 16)),
        gold_filter_pass=sum(p["group"].startswith("robotics/") for p in left),
    )


WORKLOADS = {
    "biodex": load_biodex,
    "reviews": load_reviews,
    "medec": load_medec,
    "join5x5": load_join5x5,
    "contractnli": load_contractnli,
    "contractnli_filter": load_contractnli_filter,
    "medec_full": load_medec_full,
    "contractnli_fjm": load_contractnli_pipeline,
    "contractnli_fj": lambda n=None: load_contractnli_pipeline(n, with_map=False),
    "arxiv_filter": load_arxiv_filter,
    "arxiv_join": load_arxiv_join,
    "arxiv_fjf": load_arxiv_pipeline,
}
