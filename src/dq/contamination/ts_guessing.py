"""TS-Guessing for MCQ benchmark contamination detection.

Based on: "Time Travel in LLMs: Tracing Data Contamination in Large Language Models".
For multiple-choice questions, check if a model can guess the correct answer given
only the choices (without the question). If yes at above-chance rates -> contaminated.

Requires: openai (optional dependency) or any OpenAI-compatible API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from dq.contamination.report import ContaminationReport, BenchmarkContamination

logger = logging.getLogger(__name__)

_OPENAI_AVAILABLE = False
try:
    import openai  # noqa: F401

    _OPENAI_AVAILABLE = True
except ImportError:
    pass


@dataclass
class TSGuessingResult:
    """Result of TS-Guessing for a single MCQ item."""

    guessed_idx: int = -1
    correct_idx: int = -1
    is_correct: bool = False
    num_choices: int = 0


_CHOICE_LABELS = "ABCDEFGHIJ"

_SYSTEM_PROMPT = (
    "You are taking a multiple-choice test. "
    "You will be shown ONLY the answer choices, without the question. "
    "Based on your knowledge, guess which choice is most likely the correct answer. "
    "Reply with ONLY the letter (A, B, C, D, etc.) of your guess. Nothing else."
)


class TSGuessingDetector:
    """Detect MCQ benchmark contamination using TS-Guessing.

    Tests whether a model can guess the correct answer option
    for multiple-choice questions at above-chance rates,
    indicating it has seen the benchmark during training.

    Args:
        api_url: OpenAI-compatible API base URL. None uses default.
        api_key: API key. None reads from OPENAI_API_KEY env var.
        model: Model name for the API.
        max_retries: Max retries on API failure.
        retry_delay: Base delay between retries (seconds).
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

        if not _OPENAI_AVAILABLE:
            logger.warning(
                "openai not installed. TSGuessingDetector will skip. "
                "Install with: uv sync --extra model"
            )
            return

        kwargs: dict = {}
        if api_url:
            kwargs["base_url"] = api_url
        if api_key:
            kwargs["api_key"] = api_key

        try:
            self._client = openai.OpenAI(**kwargs)
        except Exception as e:
            logger.warning("Failed to initialize OpenAI client: %s", e)
            self._client = None

    @property
    def available(self) -> bool:
        """Check if the detector is usable."""
        return _OPENAI_AVAILABLE and self._client is not None

    def check_mcq_contamination(
        self,
        question: str,
        choices: list[str],
        correct_idx: int,
    ) -> TSGuessingResult:
        """Check if a model can guess the correct MCQ answer from choices alone.

        Args:
            question: The original question (not shown to model, for logging only).
            choices: List of answer choices.
            correct_idx: Index of the correct answer (0-based).

        Returns:
            TSGuessingResult with guess details.
        """
        if not self.available:
            return TSGuessingResult(num_choices=len(choices))

        # Format choices without the question
        choice_text = "\n".join(
            f"{_CHOICE_LABELS[i]}. {choice}"
            for i, choice in enumerate(choices)
        )

        user_prompt = f"Which answer is correct?\n\n{choice_text}"

        # Call API with retries
        guessed_idx = -1
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=5,
                    temperature=0.0,
                )
                answer = response.choices[0].message.content.strip().upper()

                # Parse the letter
                for i, label in enumerate(_CHOICE_LABELS[: len(choices)]):
                    if answer.startswith(label):
                        guessed_idx = i
                        break

                break  # Success
            except Exception as e:
                logger.warning("API call failed (attempt %d): %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))

        return TSGuessingResult(
            guessed_idx=guessed_idx,
            correct_idx=correct_idx,
            is_correct=guessed_idx == correct_idx,
            num_choices=len(choices),
        )

    def scan_mcq_dataset(
        self,
        items: list[dict],
        dataset_name: str = "mcq_dataset",
    ) -> ContaminationReport:
        """Scan a dataset of MCQ items for contamination.

        Args:
            items: List of dicts with keys: question, choices, correct_idx.
            dataset_name: Name for the report.

        Returns:
            ContaminationReport with guess accuracy and statistical test.
        """
        if not self.available:
            logger.warning("TSGuessingDetector not available. Returning empty report.")
            return ContaminationReport(
                dataset_name=dataset_name,
                total_docs=len(items),
                method="ts_guessing",
            )

        results: list[TSGuessingResult] = []
        for item in items:
            result = self.check_mcq_contamination(
                question=item.get("question", ""),
                choices=item.get("choices", []),
                correct_idx=item.get("correct_idx", 0),
            )
            results.append(result)

        # Compute statistics
        valid = [r for r in results if r.guessed_idx >= 0]
        correct = sum(1 for r in valid if r.is_correct)
        total_valid = len(valid)

        if total_valid == 0:
            accuracy = 0.0
            random_baseline = 0.0
        else:
            accuracy = correct / total_valid
            # Average random baseline
            random_baseline = sum(1.0 / r.num_choices for r in valid if r.num_choices > 0) / total_valid

        # Binomial test for significance
        p_value = _binomial_test(correct, total_valid, random_baseline) if total_valid > 0 else 1.0
        is_contaminated = p_value < 0.05 and accuracy > random_baseline * 1.5

        report = ContaminationReport(
            dataset_name=dataset_name,
            total_docs=len(items),
            contaminated_docs=correct if is_contaminated else 0,
            contamination_rate=accuracy if is_contaminated else 0.0,
            method="ts_guessing",
        )

        report.per_benchmark[dataset_name] = BenchmarkContamination(
            benchmark_name=dataset_name,
            total_docs=total_valid,
            contaminated_docs=correct,
            contamination_rate=accuracy,
            avg_overlap=0.0,
            sample_contaminated=[
                {
                    "text": items[i].get("question", "")[:200],
                    "guessed": _CHOICE_LABELS[r.guessed_idx] if r.guessed_idx >= 0 else "?",
                    "correct": _CHOICE_LABELS[r.correct_idx] if r.correct_idx >= 0 else "?",
                    "is_correct": r.is_correct,
                }
                for i, r in enumerate(results)
                if r.is_correct
            ][:10],
        )

        # Add metadata to samples
        report.sample_contaminated = [
            {
                "text": items[i].get("question", "")[:200],
                "overlap_ratio": 0.0,
                "matched_benchmark": dataset_name,
                "accuracy": accuracy,
                "random_baseline": random_baseline,
                "p_value": p_value,
            }
            for i, r in enumerate(results)
            if r.is_correct
        ][:10]

        return report


def _binomial_test(successes: int, trials: int, p: float) -> float:
    """Simple one-sided binomial test (p-value for observing >= successes).

    Uses normal approximation for large samples.
    """
    if trials == 0 or p <= 0 or p >= 1:
        return 1.0

    import math

    mean = trials * p
    std = math.sqrt(trials * p * (1 - p))

    if std == 0:
        return 0.0 if successes > mean else 1.0

    # Z-score
    z = (successes - 0.5 - mean) / std  # continuity correction

    # One-sided p-value using complementary error function
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    return p_value
