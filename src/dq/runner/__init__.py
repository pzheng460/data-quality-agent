"""Production data cleaning pipeline runner.

Multi-phase pipeline: parse → filter → dedup → contamination → package.
Each phase reads from previous stage and writes to kept/rejected shards.
"""

from dq.runner.engine import PhaseEngine

__all__ = ["PhaseEngine"]
