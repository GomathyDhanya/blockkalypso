from enum import Enum, auto
from dataclasses import dataclass, field


class OpName:
    SEM_FILTER = "sem_filter"
    SEM_JOIN = "sem_join"
    SEM_CLASSIFY = "sem_classify"
    SEM_TOPK = "sem_topk"
    SEM_MAP = "sem_map"
    SEM_AGG = "sem_agg"
    JOIN = "join"
    CARTESIAN_PRODUCT = "cp"
    INDEXED_SEARCH = "indexed_search"


OPERATOR_LIST = [
    OpName.SEM_FILTER,
    OpName.SEM_JOIN,
    OpName.SEM_CLASSIFY,
    OpName.SEM_TOPK,
    OpName.SEM_MAP,
    OpName.SEM_AGG,
    OpName.JOIN,
    OpName.CARTESIAN_PRODUCT,
    OpName.INDEXED_SEARCH,
]


class OpBehavior(Enum):
    TUPLE_INDEPENDENT = auto()   # sem_filter, sem_map, sem_classify, sem_join
    BLOCKING = auto()     # sem_topk, sem_agg, 
    JOIN = auto()     # join, cartesian product


# NOTE ON TERMINOLOGY: BLOCKING above is Kalypso's own term from its API
# paper — an operator that cannot be pipelined and must materialize its
# entire input table before running (sem_topk, sem_agg). That is a
# different, unrelated concept from this project's "block operators"
# (BlockSemFilter, BlockSemJoin, below), which batch several rows into
# one LLM call but remain TUPLE_INDEPENDENT / pipelining operators in
# Kalypso's sense — they process bounded blocks as they arrive, not the
# whole table at once. Don't conflate the two when reading the Kalypso
# paper's "blocking vs. pipelined execution" experiments against this
# project's block operators; they are not the same axis.
#
# CONTRACT FOR BLOCK OPERATORS: a block operator (one that processes
# several rows in a single LLM call) declares this to Kalypso's planner/
# scheduler through two independent flags, read as plain attributes
# (not constructor args — see BlockSemFilter/BlockSemJoin for where they
# are set):
#   - supports_blocks: the operator has a run_block(rows) method and can
#     be handed a whole block of rows in one call instead of one row at
#     a time. Set on every block operator (BlockSemFilter, BlockSemJoin).
#   - requires_block_task: the stage this operator sits in must *always*
#     be scheduled as a block task (Stage.block_tasks), even if its own
#     configured block_size is 1 — e.g. BlockSemJoin needs its left rows
#     gathered into a left-block before it can run at all, regardless of
#     block_size. Most block operators don't need this: Stage.block_tasks
#     already becomes True whenever block_size > 1, so requires_block_task
#     only matters for an operator that must run as a block task even at
#     block_size == 1.


class BaseOp:
    def __init__(self, behavior, position, predicate=False):
        self.behavior = behavior
        self.position = position
        self.predicate = predicate

    async def __call__(self, ctx):
        raise NotImplementedError

    def predicate_passed(self, verdict):
        false_token = getattr(self, "FALSE", "false")
        passed = false_token not in str(verdict).strip().lower()
        if getattr(self, "negate", False):
            passed = not passed
        return passed

    async def handle_output(self, ctx, output):
        passed = output if isinstance(output, bool) else self.predicate_passed(output)

        if (getattr(self, "unpin", False) or not passed) and ctx.state.pin_req_id is not None:
            if getattr(self, "LOG", False):
                print(
                    "[sem-op] "
                    f"{self.__class__.__name__} unpin pin_req_id={ctx.state.pin_req_id} "
                    f"passed={passed} "
                    f"self_unpin={getattr(self, 'unpin', False)}"
                )
            await ctx.state.executor.unpin(ctx.state.raw_request, ctx.state.pin_req_id)
            ctx.state.pin_req_id = None

        return passed if self.predicate else ctx
     
