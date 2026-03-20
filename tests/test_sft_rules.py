"""Tests for SFTRulesFilter (Layer 1)."""

import dq.filters  # noqa: F401 — trigger filter registration
from dq.filters.sft_rules import SFTRulesFilter, _cjk_ratio, _simple_similarity


class TestSFTRulesFilter:
    """Tests for the SFT-specific rule-based filter."""

    def _make_filter(self, **kwargs):
        return SFTRulesFilter(**kwargs)

    # --- empty_output ---

    def test_empty_output_drops(self):
        f = self._make_filter()
        doc = {"instruction": "Write a poem", "output": ""}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "empty_output"

    def test_whitespace_output_drops(self):
        f = self._make_filter()
        doc = {"instruction": "Write a poem", "output": "   \n  "}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "empty_output"

    def test_empty_output_detailed(self):
        f = self._make_filter()
        doc = {"instruction": "Write a poem", "output": ""}
        passed, failures = f.filter_detailed(doc)
        assert not passed
        assert any(fail["rule"] == "empty_output" for fail in failures)

    # --- output_too_short ---

    def test_short_output_long_instruction_drops(self):
        f = self._make_filter(min_output_words=5, min_instruction_words_for_short_check=20)
        instruction = " ".join(["word"] * 25)  # 25 words
        doc = {"instruction": instruction, "output": "Yes"}  # 1 word
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "output_too_short"

    def test_short_output_short_instruction_passes(self):
        """Short output is OK if instruction is also short."""
        f = self._make_filter(min_output_words=5, min_instruction_words_for_short_check=20)
        doc = {"instruction": "Say hello", "output": "Hi"}
        keep, info = f.filter(doc)
        # Should pass because instruction < 20 words
        assert keep

    def test_adequate_output_passes(self):
        f = self._make_filter(min_output_words=5)
        instruction = " ".join(["word"] * 25)
        doc = {"instruction": instruction, "output": "Here is a nice detailed response for you."}
        keep, _ = f.filter(doc)
        assert keep

    # --- instruction_copy ---

    def test_instruction_copy_drops(self):
        f = self._make_filter(max_copy_similarity=0.80)
        doc = {"instruction": "Write a detailed explanation of photosynthesis.",
               "output": "Write a detailed explanation of photosynthesis."}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "instruction_copy"

    def test_different_output_passes(self):
        f = self._make_filter(max_copy_similarity=0.80)
        doc = {"instruction": "Write a poem about trees.",
               "output": "Tall oaks stand in morning light, their leaves a canopy of green."}
        keep, _ = f.filter(doc)
        assert keep

    # --- ai_refusal ---

    def test_ai_refusal_drops(self):
        f = self._make_filter()
        doc = {"instruction": "Do something harmful",
               "output": "As an AI language model, I cannot do that."}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "ai_refusal"

    def test_refusal_i_cannot(self):
        f = self._make_filter()
        doc = {"instruction": "Do X", "output": "I cannot help with that request."}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "ai_refusal"

    def test_refusal_i_apologize(self):
        f = self._make_filter()
        doc = {"instruction": "Do X", "output": "I apologize, but I can't help with that."}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "ai_refusal"

    def test_non_refusal_passes(self):
        f = self._make_filter()
        doc = {"instruction": "Write hello", "output": "Hello! How are you today?"}
        keep, _ = f.filter(doc)
        assert keep

    def test_custom_refusal_patterns(self):
        f = self._make_filter(refusal_patterns=["sorry dave"])
        doc = {"instruction": "Open pod bay doors",
               "output": "Sorry Dave, I'm afraid I can't do that."}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "ai_refusal"

    # --- language_mismatch ---

    def test_language_mismatch_drops(self):
        f = self._make_filter(lang_mismatch_threshold=0.3)
        doc = {"instruction": "请解释光合作用",  # Chinese
               "output": "Photosynthesis is a process by which plants convert light."}  # English
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "language_mismatch"

    def test_same_language_passes(self):
        f = self._make_filter()
        doc = {"instruction": "Explain photosynthesis",
               "output": "Photosynthesis is the process by which plants make food."}
        keep, _ = f.filter(doc)
        assert keep

    def test_chinese_both_passes(self):
        f = self._make_filter()
        doc = {"instruction": "请解释光合作用",
               "output": "光合作用是植物利用光能将二氧化碳和水转化为葡萄糖的过程。"}
        keep, _ = f.filter(doc)
        assert keep

    # --- filter_detailed ---

    def test_filter_detailed_returns_all_failures(self):
        """filter_detailed should return ALL failures, not just the first."""
        f = self._make_filter()
        doc = {"instruction": "Write something", "output": ""}
        passed, failures = f.filter_detailed(doc)
        assert not passed
        # Empty output should be flagged
        rules = {fail["rule"] for fail in failures}
        assert "empty_output" in rules

    def test_filter_detailed_passes_good_doc(self):
        f = self._make_filter()
        doc = {"instruction": "Write a poem about trees.",
               "output": "Tall oaks stand in morning light, their leaves dancing in the breeze."}
        passed, failures = f.filter_detailed(doc)
        assert passed
        assert failures == []

    def test_filter_detailed_multiple_failures(self):
        """A doc can fail multiple SFT rules."""
        f = self._make_filter(min_instruction_words_for_short_check=5)
        instruction = " ".join(["word"] * 10)
        doc = {"instruction": instruction,
               "output": "I cannot help with that."}  # short + ai_refusal
        passed, failures = f.filter_detailed(doc)
        assert not passed
        rules = {fail["rule"] for fail in failures}
        assert "ai_refusal" in rules

    # --- text field fallback ---

    def test_text_only_no_sft_fields(self):
        """Text-only doc without SFT fields should be rejected."""
        f = self._make_filter()
        doc = {"text": "What is 2+2?\nThe answer is 4. That is the result of adding two and two."}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "missing_sft_fields"

    def test_text_field_no_sft_fields_rejected(self):
        """Non-SFT data (no instruction/output fields) should be rejected."""
        f = self._make_filter()
        doc = {"text": "What is 2+2?"}  # No SFT fields
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "missing_sft_fields"

    # --- registration ---

    def test_registered(self):
        from dq.pipeline import get_filter_class
        cls = get_filter_class("sft_rules")
        assert cls is SFTRulesFilter

    # --- works on pretrain data (mostly passes) ---

    def test_pretrain_doc_with_newlines_rejected(self):
        """Pretrain data (text only) should be rejected even with newlines."""
        f = self._make_filter()
        doc = {"text": "This is a normal article about science.\nIt has multiple sentences and plenty of content to analyze."}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "missing_sft_fields"

    def test_pretrain_doc_rejected(self):
        """Pure pretrain data (text only, no SFT fields) should be rejected."""
        f = self._make_filter()
        doc = {"text": "This is a normal article about science."}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "missing_sft_fields"


class TestHelpers:
    def test_cjk_ratio_english(self):
        assert _cjk_ratio("Hello world") == 0.0

    def test_cjk_ratio_chinese(self):
        ratio = _cjk_ratio("你好世界")
        assert ratio > 0.9

    def test_cjk_ratio_mixed(self):
        ratio = _cjk_ratio("Hello 你好")
        assert 0.2 < ratio < 0.8

    def test_cjk_ratio_empty(self):
        assert _cjk_ratio("") == 0.0

    def test_simple_similarity_identical(self):
        assert _simple_similarity("hello world", "hello world") == 1.0

    def test_simple_similarity_different(self):
        sim = _simple_similarity("hello world", "completely different text entirely")
        assert sim < 0.3

    def test_simple_similarity_empty(self):
        assert _simple_similarity("", "hello") == 0.0
        assert _simple_similarity("hello", "") == 0.0
