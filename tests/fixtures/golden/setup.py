#!/usr/bin/env python3
"""Save golden test fixtures from current pipeline output.

Run after a successful pipeline run:
    python tests/fixtures/golden/setup.py /tmp/arxiv_test/real_output

Saves ingested docs and final output as JSONL for regression testing.
"""
import json
import sys
from pathlib import Path

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/arxiv_test/real_output"
    fixture_dir = Path(__file__).parent

    from dq.shared.shard import read_shards

    # Save ingested (raw LaTeX)
    ingested = list(read_shards(f"{output_dir}/stage1_ingested/kept"))
    with open(fixture_dir / "ingested.jsonl", "w") as f:
        for doc in ingested:
            f.write(json.dumps({"id": doc["id"], "metadata": doc.get("metadata", {}),
                                "text": doc["text"][:500]}, ensure_ascii=False) + "\n")
    print(f"Saved {len(ingested)} ingested docs (truncated)")

    # Save full extraction output for golden comparison
    final = list(read_shards(f"{output_dir}/stage4_final"))
    with open(fixture_dir / "final_output.jsonl", "w") as f:
        for doc in final:
            f.write(json.dumps({"id": doc["id"], "text": doc["text"],
                                "metadata": doc.get("metadata", {})},
                               ensure_ascii=False) + "\n")
    print(f"Saved {len(final)} final docs")

if __name__ == "__main__":
    main()
