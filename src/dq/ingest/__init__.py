"""Data ingestion from multiple arxiv sources.

Supported sources:
- arxiv e-print: Download LaTeX source and convert to text
- arxiv OAI-PMH: Discover new papers by date range
- Local files: JSONL/Parquet with pre-parsed text
- HuggingFace datasets: RedPajama, Dolma, etc.
"""

from dq.ingest.arxiv_source import ArxivSource
from dq.ingest.base import IngestSource

__all__ = ["ArxivSource", "IngestSource"]
