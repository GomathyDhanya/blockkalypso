"""Workloads for the evaluation harness.

A workload is the rows and instruction every arm runs over. Rows are read the
way Kalypso reads a directory of ``*.txt`` files (``vllm/kalypso/data/data.py``:
sorted by integer stem, text stripped, ``idx`` = stem), so the direct-vLLM arm
can rebuild byte-identical prompts.
"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vllm.kalypso.benchmark import scenarios

SAMPLE_DATA = Path(__file__).resolve().parents[1] / "sample_data"
BIODEX_ARTICLES = SAMPLE_DATA / "articles_500"


@dataclass(frozen=True)
class Row:
    idx: int
    text: str


@dataclass(frozen=True)
class Workload:
    name: str
    data_path: Path      # directory Kalypso's /v1/semantic/query reads
    instruction: str
    op: str              # Kalypso op name, e.g. "sem_map"
    rows: list[Row]


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


def load_biodex(n: int | None = None) -> Workload:
    """The first ``n`` BioDEX articles (all of them when ``n`` is None)."""
    files = _txt_files(BIODEX_ARTICLES)
    if not files:
        raise FileNotFoundError(f"no *.txt articles under {BIODEX_ARTICLES}")
    if n is not None:
        if not 1 <= n <= len(files):
            raise ValueError(f"--n must be between 1 and {len(files)} for biodex")
        files = files[:n]
    data_path = BIODEX_ARTICLES if len(files) == len(_txt_files(BIODEX_ARTICLES)) \
        else _subset_dir(files)
    rows = [Row(int(p.stem), p.read_text(encoding="utf-8").strip()) for p in files]
    return Workload(
        name="biodex",
        data_path=data_path.resolve(),
        instruction=scenarios.BIODEX_MAP_REACTIONS,
        op="sem_map",
        rows=rows,
    )


WORKLOADS = {
    "biodex": load_biodex,
}
