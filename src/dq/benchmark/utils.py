"""Utility functions for benchmarking."""

# Import canonical SFT field set from sft_rules — single source of truth
from dq.stages.curation.filters.sft_rules import SFT_DETECT_FIELDS as SFT_FIELDS


def detect_data_type(docs: list[dict]) -> str:
    """Detect whether docs are SFT (instruction) or pre-training data.

    Returns 'sft' if docs have instruction/conversations fields, 'pretrain' otherwise.
    """
    if not docs:
        return "pretrain"

    # Check first few docs for SFT-specific fields
    sample = docs[:min(10, len(docs))]
    sft_count = 0
    for doc in sample:
        doc_fields = set(doc.keys())
        if doc_fields & SFT_FIELDS:
            sft_count += 1

    # If majority of sampled docs have SFT fields, classify as SFT
    if sft_count > len(sample) / 2:
        return "sft"
    return "pretrain"


def _extract_sft_fields(doc: dict) -> tuple[str, str]:
    """Extract instruction and output from a doc.

    Handles both structured SFT docs (with 'instruction'/'output' fields)
    and merged text docs (split on first newline).
    """
    instruction = doc.get("instruction", "") or ""
    output = doc.get("output", "") or ""

    if instruction and output:
        return instruction, output

    # For merged text, try splitting on newlines
    text = doc.get("text", "") or ""
    if text:
        parts = text.split("\n", 1)
        instruction = parts[0].strip()
        output = parts[1].strip() if len(parts) > 1 else ""

    return instruction, output