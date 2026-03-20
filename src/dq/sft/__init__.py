"""SFT data quality scoring and filtering.

Primary module: LLM Binary Judge (SFTQualityJudge).
Deprecated modules: ComplexityScorer, QualityScorer, EducationalValueScorer, WritingQualityScorer.
"""

from dq.sft.llm_judge import SFTQualityJudge
from dq.sft.diversity import DiversityFilter

# Deprecated — kept for backward compatibility
from dq.sft.complexity import ComplexityScorer
from dq.sft.educational import EducationalValueScorer
from dq.sft.quality import QualityScorer
from dq.sft.writing_quality import WritingQualityScorer

__all__ = [
    "SFTQualityJudge",
    "DiversityFilter",
    # Deprecated
    "ComplexityScorer",
    "EducationalValueScorer",
    "QualityScorer",
    "WritingQualityScorer",
]
