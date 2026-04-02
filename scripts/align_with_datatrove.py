#!/usr/bin/env python3
"""Alignment test: compare dq filters against datatrove (official reference implementation).

Usage:
    uv run python scripts/align_with_datatrove.py [--n 500] [--dataset HF_ID] [--seed 42]

Runs both dq and datatrove filters on the same documents, reports:
  - Per-filter agreement rate
  - Disagreement samples with detailed reasons
  - Known implementation differences
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── datatrove imports ──────────────────────────────────────────────
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import get_filter_result
from datatrove.pipeline.filters.gopher_quality_filter import GopherQualityFilter as DT_GopherQuality
from datatrove.pipeline.filters.gopher_repetition_filter import GopherRepetitionFilter as DT_GopherRepetition
from datatrove.pipeline.filters.c4_filters import C4QualityFilter as DT_C4
from datatrove.pipeline.filters.fineweb_quality_filter import FineWebQualityFilter as DT_FineWeb

# ── dq imports ─────────────────────────────────────────────────────
from dq.filters.gopher import GopherQualityFilter as DQ_GopherQuality
from dq.filters.gopher import GopherRepetitionFilter as DQ_GopherRepetition
from dq.filters.c4 import C4Filter as DQ_C4
from dq.filters.fineweb import FineWebFilter as DQ_FineWeb

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_samples(dataset_id: str, n: int, seed: int) -> list[str]:
    """Load n text samples from a HuggingFace dataset via streaming."""
    from datasets import load_dataset

    logger.info(f"Loading {n} samples from {dataset_id} (streaming)...")
    # Some datasets require a config name (e.g., allenai/c4 -> "en")
    kwargs: dict = {"split": "train", "streaming": True}
    # Auto-detect common configs
    if dataset_id == "allenai/c4":
        kwargs["name"] = "en"
    ds = load_dataset(dataset_id, **kwargs)
    texts: list[str] = []
    for item in ds:
        text = item.get("text", "")
        if text and len(text.strip()) > 0:
            texts.append(text)
        if len(texts) >= n:
            break
    logger.info(f"Loaded {len(texts)} samples.")
    return texts


# ═══════════════════════════════════════════════════════════════════
# Comparison framework
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Disagreement:
    idx: int
    text_preview: str
    dq_pass: bool
    dt_pass: bool
    dq_reason: str
    dt_reason: str


@dataclass
class FilterComparison:
    name: str
    total: int = 0
    agree: int = 0
    both_pass: int = 0
    both_fail: int = 0
    dq_only_pass: int = 0  # dq passes, datatrove rejects
    dt_only_pass: int = 0  # datatrove passes, dq rejects
    disagreements: list[Disagreement] = field(default_factory=list)
    max_disagreements: int = 20

    @property
    def agreement_rate(self) -> float:
        return self.agree / self.total if self.total else 0.0

    def record(self, idx: int, text: str, dq_pass: bool, dt_pass: bool,
               dq_reason: str = "", dt_reason: str = ""):
        self.total += 1
        if dq_pass == dt_pass:
            self.agree += 1
            if dq_pass:
                self.both_pass += 1
            else:
                self.both_fail += 1
        else:
            if dq_pass:
                self.dq_only_pass += 1
            else:
                self.dt_only_pass += 1
            if len(self.disagreements) < self.max_disagreements:
                self.disagreements.append(Disagreement(
                    idx=idx,
                    text_preview=text[:120].replace("\n", "\\n"),
                    dq_pass=dq_pass,
                    dt_pass=dt_pass,
                    dq_reason=dq_reason,
                    dt_reason=dt_reason,
                ))


def run_dq_filter(filt, text: str) -> tuple[bool, str]:
    """Run a dq filter, return (pass, reason_string)."""
    doc = {"text": text}
    passed, info = filt.filter(doc)
    if passed:
        return True, ""
    if isinstance(info, dict):
        return False, info.get("reason", str(info))
    return False, str(info)


def run_dt_filter(filt, text: str, doc_id: str = "0") -> tuple[bool, str]:
    """Run a datatrove filter, return (pass, reason_string).

    Note: some filters (C4) mutate doc.text, so we create a fresh Document each time.
    """
    doc = Document(text=text, id=doc_id)
    raw = filt.filter(doc)
    passed, reason = get_filter_result(raw)
    return passed, reason or ""


# ═══════════════════════════════════════════════════════════════════
# Filter-specific comparisons (aligned thresholds)
# ═══════════════════════════════════════════════════════════════════

def compare_gopher_quality(texts: list[str]) -> FilterComparison:
    """Compare GopherQualityFilter.

    Known differences vs datatrove:
    - Word tokenization: datatrove uses nltk/spacy, dq uses str.split()
    - Non-symbol words: datatrove filters pure-punctuation tokens for word count
    - Symbol ratio: datatrove checks # and ... separately against n_words;
      dq checks tokens that ARE "#"/"..."/"…" / total words
    - Alpha ratio: datatrove = word-level (frac of words with ≥1 alpha char);
      dq = char-level (frac of non-whitespace chars that are alpha)
    - Stopwords: datatrove uses set intersection (unique); dq counts all occurrences
    """
    comp = FilterComparison(name="gopher_quality")

    # Use matching thresholds where possible
    dq_f = DQ_GopherQuality(
        min_words=50, max_words=100_000,
        min_avg_word_len=3.0, max_avg_word_len=10.0,
        max_symbol_ratio=0.1,
        max_bullet_lines_ratio=0.9,
        max_ellipsis_lines_ratio=0.3,
        min_stopwords=2,
        min_alpha_ratio=0.8,
    )
    dt_f = DT_GopherQuality(
        min_doc_words=50, max_doc_words=100_000,
        min_avg_word_length=3, max_avg_word_length=10,
        max_symbol_word_ratio=0.1,
        max_bullet_lines_ratio=0.9,
        max_ellipsis_lines_ratio=0.3,
        max_non_alpha_words_ratio=0.8,
        min_stop_words=2,
    )

    for i, text in enumerate(texts):
        dq_pass, dq_reason = run_dq_filter(dq_f, text)
        dt_pass, dt_reason = run_dt_filter(dt_f, text, str(i))
        comp.record(i, text, dq_pass, dt_pass, dq_reason, dt_reason)

    return comp


def compare_gopher_repetition(texts: list[str]) -> FilterComparison:
    """Compare GopherRepetitionFilter.

    Known differences vs datatrove:
    - top_ngram_ratio: datatrove = top_ngram_char_length / len(text);
      dq = top_ngram_count / total_ngram_count (frequency ratio, not char fraction)
    - duplicate counting: datatrove counts only subsequent occurrences as duplicates
      (first occurrence is not a duplicate); dq counts ALL copies if count > 1
    """
    comp = FilterComparison(name="gopher_repetition")

    dq_f = DQ_GopherRepetition(
        max_top_2gram=0.20, max_top_3gram=0.18, max_top_4gram=0.16,
        max_dup_line_ratio=0.30, max_dup_para_ratio=0.30,
        max_dup_line_char_frac=0.20, max_dup_para_char_frac=0.20,
        max_dup_5gram_frac=0.15, max_dup_6gram_frac=0.14,
        max_dup_7gram_frac=0.13, max_dup_8gram_frac=0.12,
        max_dup_9gram_frac=0.11, max_dup_10gram_frac=0.10,
    )
    dt_f = DT_GopherRepetition(
        dup_line_frac=0.30, dup_para_frac=0.30,
        dup_line_char_frac=0.20,
        dup_para_char_frac=0.20,
        top_n_grams=((2, 0.20), (3, 0.18), (4, 0.16)),
        dup_n_grams=((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.10)),
    )

    for i, text in enumerate(texts):
        dq_pass, dq_reason = run_dq_filter(dq_f, text)
        dt_pass, dt_reason = run_dt_filter(dt_f, text, str(i))
        comp.record(i, text, dq_pass, dt_pass, dq_reason, dt_reason)

    return comp


def compare_c4(texts: list[str]) -> FilterComparison:
    """Compare C4Filter.

    Known differences vs datatrove:
    - curly_bracket: datatrove default True, dq default False
    - min_sentences: datatrove default 5, dq default 5
    - min_words_per_line: datatrove has (default 3), dq doesn't
    - max_word_length: datatrove has (default 1000), dq doesn't
    - remove_citations: datatrove has, dq doesn't
    - Terminal punct: datatrove checks endswith(END_PUNCTUATION) and filters ellipsis;
      dq uses regex r"[.!?。！？;；\"'\"\']$"
    - Sentence counting: datatrove uses spacy, dq uses regex
    """
    comp = FilterComparison(name="c4")

    # Align thresholds: use dq's more lenient defaults to isolate logic differences
    dq_f = DQ_C4(
        remove_no_terminal_punct=True,
        remove_javascript_lines=True,
        remove_policy_lines=True,
        remove_lorem_ipsum=True,
        remove_curly_brace=True,    # match datatrove default
        min_sentences=5,             # match datatrove default
    )
    dt_f = DT_C4(
        split_paragraph=True,
        remove_citations=False,      # dq doesn't have this
        filter_no_terminal_punct=True,
        min_num_sentences=5,
        min_words_per_line=-1,       # disable (dq doesn't have this)
        max_word_length=-1,          # disable (dq doesn't have this)
        filter_lorem_ipsum=True,
        filter_javascript=True,
        filter_curly_bracket=True,
        filter_policy=True,
    )

    for i, text in enumerate(texts):
        dq_pass, dq_reason = run_dq_filter(dq_f, text)
        dt_pass, dt_reason = run_dt_filter(dt_f, text, str(i))
        comp.record(i, text, dq_pass, dt_pass, dq_reason, dt_reason)

    return comp


def compare_fineweb(texts: list[str]) -> FilterComparison:
    """Compare FineWebFilter.

    Known differences (substantial — very different implementations):
    - datatrove checks: line_punct_ratio, short_line_ratio, char_dup_ratio, list_ratio (newline/word)
    - dq checks: list_line_ratio (bullet regex), dup_line_ratio (Counter), bad_line_breaks (avg len)
    - These are fundamentally different rule sets inspired by the same FineWeb paper
    """
    comp = FilterComparison(name="fineweb")

    dq_f = DQ_FineWeb()
    dt_f = DT_FineWeb()

    for i, text in enumerate(texts):
        dq_pass, dq_reason = run_dq_filter(dq_f, text)
        dt_pass, dt_reason = run_dt_filter(dt_f, text, str(i))
        comp.record(i, text, dq_pass, dt_pass, dq_reason, dt_reason)

    return comp


# ═══════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════

def print_report(comparisons: list[FilterComparison]):
    print("\n" + "=" * 80)
    print("  DQ vs Datatrove Alignment Report")
    print("=" * 80)

    for comp in comparisons:
        print(f"\n{'─' * 80}")
        print(f"  {comp.name}")
        print(f"{'─' * 80}")
        print(f"  Total samples:     {comp.total}")
        print(f"  Agreement rate:    {comp.agreement_rate:.1%}")
        print(f"    Both pass:       {comp.both_pass}")
        print(f"    Both fail:       {comp.both_fail}")
        print(f"    dq pass only:    {comp.dq_only_pass}  (dq more lenient)")
        print(f"    dt pass only:    {comp.dt_only_pass}  (datatrove more lenient)")

        if comp.disagreements:
            print(f"\n  First {len(comp.disagreements)} disagreements:")
            for d in comp.disagreements[:10]:
                dq_label = "PASS" if d.dq_pass else f"FAIL({d.dq_reason})"
                dt_label = "PASS" if d.dt_pass else f"FAIL({d.dt_reason})"
                print(f"    [{d.idx:>4}] dq={dq_label:<35} dt={dt_label}")
                print(f"           text: {d.text_preview[:80]}...")

    # Summary
    print(f"\n{'=' * 80}")
    print("  Summary")
    print(f"{'=' * 80}")
    for comp in comparisons:
        status = "✓" if comp.agreement_rate > 0.95 else "△" if comp.agreement_rate > 0.80 else "✗"
        print(f"  {status} {comp.name:<25} {comp.agreement_rate:.1%} agreement"
              f"  (dq lenient: {comp.dq_only_pass}, dt lenient: {comp.dt_only_pass})")

    total_agree = sum(c.agree for c in comparisons)
    total = sum(c.total for c in comparisons)
    print(f"\n  Overall: {total_agree}/{total} = {total_agree/total:.1%} agreement")


def save_disagreements(comparisons: list[FilterComparison], path: Path):
    """Save detailed disagreements as JSON for further analysis."""
    output = {}
    for comp in comparisons:
        output[comp.name] = {
            "agreement_rate": comp.agreement_rate,
            "total": comp.total,
            "both_pass": comp.both_pass,
            "both_fail": comp.both_fail,
            "dq_only_pass": comp.dq_only_pass,
            "dt_only_pass": comp.dt_only_pass,
            "disagreements": [
                {
                    "idx": d.idx,
                    "text_preview": d.text_preview,
                    "dq_pass": d.dq_pass,
                    "dt_pass": d.dt_pass,
                    "dq_reason": d.dq_reason,
                    "dt_reason": d.dt_reason,
                }
                for d in comp.disagreements
            ],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Disagreements saved to {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Align dq filters with datatrove reference")
    parser.add_argument("--dataset", default="allenai/c4", help="HuggingFace dataset ID")
    parser.add_argument("-n", type=int, default=500, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="scripts/alignment_report.json", help="JSON output path")
    parser.add_argument("--filter", default=None,
                        help="Run only specific filter (gopher_quality, gopher_repetition, c4, fineweb)")
    args = parser.parse_args()

    texts = load_samples(args.dataset, args.n, args.seed)

    filter_map = {
        "gopher_quality": compare_gopher_quality,
        "gopher_repetition": compare_gopher_repetition,
        "c4": compare_c4,
        "fineweb": compare_fineweb,
    }

    if args.filter:
        if args.filter not in filter_map:
            print(f"Unknown filter: {args.filter}. Choose from: {list(filter_map.keys())}")
            sys.exit(1)
        comparisons = [filter_map[args.filter](texts)]
    else:
        comparisons = []
        for name, fn in filter_map.items():
            logger.info(f"Comparing {name}...")
            comparisons.append(fn(texts))

    print_report(comparisons)
    save_disagreements(comparisons, Path(args.output))


if __name__ == "__main__":
    main()
