"""Shared test fixtures."""

import pytest
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def good_docs():
    """Load good sample documents."""
    import json
    docs = []
    with open(FIXTURES_DIR / "sample_good.jsonl") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


@pytest.fixture
def bad_docs():
    """Load bad sample documents."""
    import json
    docs = []
    with open(FIXTURES_DIR / "sample_bad.jsonl") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs
