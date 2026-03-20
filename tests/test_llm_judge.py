"""Tests for rule-based binary LLM judges.

Tests both SFTQualityJudge and PretrainingQualityJudge with mocked API calls.
"""

import json
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# SFTQualityJudge Tests
# ---------------------------------------------------------------------------

class TestSFTQualityJudge:
    """Tests for rule-based SFT quality judge."""

    def _make_judge(self, mock_client=None):
        """Create a judge with optional mock client."""
        from dq.sft.llm_judge import SFTQualityJudge
        judge = SFTQualityJudge(api_key="test-key", model="test-model")
        judge.max_retries = 1
        judge.retry_delay = 0.01
        if mock_client is not None:
            judge._get_client = lambda: mock_client
        return judge

    def _mock_response(self, content: str):
        """Create a mock API response."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp

    def _valid_json_response(self, rules_dict: dict):
        """Create a valid JSON response for the given rules."""
        return json.dumps(rules_dict)

    def test_skip_when_no_client(self):
        """Judge should return low quality when no API client available."""
        from dq.sft.llm_judge import SFTQualityJudge
        judge = SFTQualityJudge()
        judge._get_client = lambda: None

        result = judge.judge_one("Write hello world", "Hello, world!")
        assert result["quality"] == "low"
        assert "api_unavailable" in result["failed_rules"]
        assert "error" in result

    def test_all_rules_pass_high_quality(self):
        """When all rules pass, should return HIGH quality."""
        all_pass = {
            "instruction_following": {"pass": True, "reason": ""},
            "factuality": {"pass": True, "reason": ""},
            "completeness": {"pass": True, "reason": ""},
            "format_compliance": {"pass": True, "reason": ""},
            "harmlessness": {"pass": True, "reason": ""}
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            self._valid_json_response(all_pass)
        )
        judge = self._make_judge(mock_client)

        result = judge.judge_one("Explain photosynthesis", "Photosynthesis is the process...")
        assert result["quality"] == "high"
        assert len(result["failed_rules"]) == 0
        assert result["rules"] == all_pass

    def test_one_rule_fails_low_quality(self):
        """When any rule fails, should return LOW quality."""
        one_fail = {
            "instruction_following": {"pass": True, "reason": ""},
            "factuality": {"pass": False, "reason": "Claims Python was invented in 2005"},
            "completeness": {"pass": True, "reason": ""},
            "format_compliance": {"pass": True, "reason": ""},
            "harmlessness": {"pass": True, "reason": ""}
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            self._valid_json_response(one_fail)
        )
        judge = self._make_judge(mock_client)

        result = judge.judge_one("When was Python created?", "Python was created in 2005")
        assert result["quality"] == "low"
        assert "factuality" in result["failed_rules"]
        assert len(result["failed_rules"]) == 1

    def test_multiple_rules_fail_low_quality(self):
        """When multiple rules fail, should return LOW quality with all failed rules listed."""
        multi_fail = {
            "instruction_following": {"pass": False, "reason": "Off topic"},
            "factuality": {"pass": False, "reason": "Wrong facts"},
            "completeness": {"pass": True, "reason": ""},
            "format_compliance": {"pass": True, "reason": ""},
            "harmlessness": {"pass": True, "reason": ""}
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            self._valid_json_response(multi_fail)
        )
        judge = self._make_judge(mock_client)

        result = judge.judge_one("Explain gravity", "The sky is blue because of unicorns")
        assert result["quality"] == "low"
        assert "instruction_following" in result["failed_rules"]
        assert "factuality" in result["failed_rules"]
        assert len(result["failed_rules"]) == 2

    def test_malformed_json_uses_regex_fallback(self):
        """When LLM returns malformed JSON, should use regex fallback."""
        # Malformed JSON but with extractable patterns
        malformed_response = '''Here's my analysis:
        {
          "instruction_following": {"pass": true, "reason": "good"},
          "factuality": {"pass": false, "reason": "wrong"},
          "completeness": {"pass": true},
          "format_compliance": {"pass": true},
          "harmlessness": {"pass": true}
        } extra text'''

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(malformed_response)
        judge = self._make_judge(mock_client)

        result = judge.judge_one("Test instruction", "Test output")
        assert result["quality"] == "low"  # Because factuality fails
        assert "factuality" in result["failed_rules"]

    def test_completely_unparseable_response_defaults_fail(self):
        """When response is completely unparseable, should default all rules to fail."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            "Sorry, I can't help with that request."
        )
        judge = self._make_judge(mock_client)

        result = judge.judge_one("Test instruction", "Test output")
        assert result["quality"] == "low"
        assert len(result["failed_rules"]) == 5  # All rules should fail

    def test_api_error_returns_low_quality(self):
        """When API call fails, should return low quality with error info."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        judge = self._make_judge(mock_client)

        result = judge.judge_one("Test instruction", "Test output")
        assert result["quality"] == "low"
        assert "api_error" in result["failed_rules"]
        assert "error" in result
        assert "API error" in result["error"]

    def test_batch_processing(self):
        """Test judge_batch processes multiple documents correctly."""
        all_pass = {
            "instruction_following": {"pass": True, "reason": ""},
            "factuality": {"pass": True, "reason": ""},
            "completeness": {"pass": True, "reason": ""},
            "format_compliance": {"pass": True, "reason": ""},
            "harmlessness": {"pass": True, "reason": ""}
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            self._valid_json_response(all_pass)
        )
        judge = self._make_judge(mock_client)

        docs = [
            {"instruction": "Question 1", "output": "Answer 1"},
            {"instruction": "Question 2", "output": "Answer 2"},
        ]

        result = judge.judge_batch(docs)
        assert len(result) == 2
        assert all("sft_quality" in doc for doc in result)
        assert all(doc["sft_quality"] == "high" for doc in result)

    def test_batch_with_missing_fields(self):
        """Test batch processing handles missing instruction/output fields."""
        docs = [
            {"instruction": "Question", "output": "Answer"},  # Valid
            {"instruction": "Question"},  # Missing output
            {"output": "Answer"},  # Missing instruction
            {},  # Missing both
        ]

        judge = self._make_judge(MagicMock())  # Won't be called for invalid docs

        result = judge.judge_batch(docs)
        assert result[0].get("sft_quality") != "low" or "missing_fields" not in result[0]["sft_failed_rules"]
        assert result[1]["sft_quality"] == "low"
        assert "missing_fields" in result[1]["sft_failed_rules"]
        assert result[2]["sft_quality"] == "low"
        assert "missing_fields" in result[2]["sft_failed_rules"]
        assert result[3]["sft_quality"] == "low"
        assert "missing_fields" in result[3]["sft_failed_rules"]


# ---------------------------------------------------------------------------
# PretrainingQualityJudge Tests
# ---------------------------------------------------------------------------

class TestPretrainingQualityJudge:
    """Tests for rule-based pre-training quality judge."""

    def _make_judge(self, mock_client=None):
        """Create a judge with optional mock client."""
        from dq.model_filters.llm_quality_judge import PretrainingQualityJudge
        judge = PretrainingQualityJudge(api_key="test-key", model="test-model")
        judge.max_retries = 1
        judge.retry_delay = 0.01
        if mock_client is not None:
            judge._get_client = lambda: mock_client
        return judge

    def _mock_response(self, content: str):
        """Create a mock API response."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp

    def _valid_json_response(self, rules_dict: dict):
        """Create a valid JSON response for the given rules."""
        return json.dumps(rules_dict)

    def test_skip_when_no_client(self):
        """Judge should return low quality when no API client available."""
        from dq.model_filters.llm_quality_judge import PretrainingQualityJudge
        judge = PretrainingQualityJudge()
        judge._get_client = lambda: None

        result = judge.judge_one("Some text content here")
        assert result["quality"] == "low"
        assert "api_unavailable" in result["failed_rules"]
        assert "error" in result

    def test_all_rules_pass_high_quality(self):
        """When all rules pass, should return HIGH quality."""
        all_pass = {
            "information_density": {"pass": True, "reason": ""},
            "coherence": {"pass": True, "reason": ""},
            "originality": {"pass": True, "reason": ""}
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            self._valid_json_response(all_pass)
        )
        judge = self._make_judge(mock_client)

        result = judge.judge_one("This is a well-written article about machine learning...")
        assert result["quality"] == "high"
        assert len(result["failed_rules"]) == 0
        assert result["rules"] == all_pass

    def test_information_density_fails_low_quality(self):
        """When information_density fails (ads/boilerplate), should return LOW quality."""
        density_fail = {
            "information_density": {"pass": False, "reason": "Contains cookie notice and ads"},
            "coherence": {"pass": True, "reason": ""},
            "originality": {"pass": True, "reason": ""}
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            self._valid_json_response(density_fail)
        )
        judge = self._make_judge(mock_client)

        result = judge.judge_one("Cookie Notice: This website uses cookies... [Skip to content]")
        assert result["quality"] == "low"
        assert "information_density" in result["failed_rules"]
        assert len(result["failed_rules"]) == 1

    def test_coherence_fails_low_quality(self):
        """When coherence fails (fragmented text), should return LOW quality."""
        coherence_fail = {
            "information_density": {"pass": True, "reason": ""},
            "coherence": {"pass": False, "reason": "Text appears truncated and fragmented"},
            "originality": {"pass": True, "reason": ""}
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            self._valid_json_response(coherence_fail)
        )
        judge = self._make_judge(mock_client)

        result = judge.judge_one("The quick brown... fox jumps over... incomplete...")
        assert result["quality"] == "low"
        assert "coherence" in result["failed_rules"]

    def test_originality_fails_low_quality(self):
        """When originality fails (SEO spam), should return LOW quality."""
        originality_fail = {
            "information_density": {"pass": True, "reason": ""},
            "coherence": {"pass": True, "reason": ""},
            "originality": {"pass": False, "reason": "Appears to be SEO-generated content"}
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            self._valid_json_response(originality_fail)
        )
        judge = self._make_judge(mock_client)

        result = judge.judge_one("Best AI tools 2024! Top 10 AI tools! AI tools review!")
        assert result["quality"] == "low"
        assert "originality" in result["failed_rules"]

    def test_regex_fallback_on_malformed_json(self):
        """When LLM returns malformed JSON, should use regex fallback."""
        malformed_response = '''Analysis:
        {
          "information_density": {"pass": true},
          "coherence": {"pass": false, "reason": "fragmented"},
          "originality": {"pass": true}
        } etc'''

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(malformed_response)
        judge = self._make_judge(mock_client)

        result = judge.judge_one("Test text")
        assert result["quality"] == "low"  # Because coherence fails
        assert "coherence" in result["failed_rules"]

    def test_batch_processing(self):
        """Test judge_batch processes multiple documents correctly."""
        all_pass = {
            "information_density": {"pass": True, "reason": ""},
            "coherence": {"pass": True, "reason": ""},
            "originality": {"pass": True, "reason": ""}
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            self._valid_json_response(all_pass)
        )
        judge = self._make_judge(mock_client)

        docs = [
            {"text": "Article about science..."},
            {"text": "Tutorial on programming..."},
        ]

        result = judge.judge_batch(docs)
        assert len(result) == 2
        assert all("pretrain_quality" in doc for doc in result)
        assert all(doc["pretrain_quality"] == "high" for doc in result)

    def test_batch_with_empty_text(self):
        """Test batch processing handles empty text fields."""
        docs = [
            {"text": "Valid content"},
            {"text": ""},  # Empty text
            {"text": "   "},  # Whitespace only
            {},  # Missing text field
        ]

        judge = self._make_judge(MagicMock())  # Won't be called for empty docs

        result = judge.judge_batch(docs)
        # First doc should be processed normally, others should fail for empty text
        assert result[1]["pretrain_quality"] == "low"
        assert "empty_text" in result[1]["pretrain_failed_rules"]
        assert result[2]["pretrain_quality"] == "low"
        assert "empty_text" in result[2]["pretrain_failed_rules"]
        assert result[3]["pretrain_quality"] == "low"
        assert "empty_text" in result[3]["pretrain_failed_rules"]

    def test_api_error_handling(self):
        """When API call fails, should return low quality with error info."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Network error")
        judge = self._make_judge(mock_client)

        result = judge.judge_one("Test text")
        assert result["quality"] == "low"
        assert "api_error" in result["failed_rules"]
        assert "error" in result
        assert "Network error" in result["error"]