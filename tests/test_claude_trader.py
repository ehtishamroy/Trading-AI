"""Tests for ClaudeTrader verdict parsing."""

import pytest
import json


class TestParseVerdict:
    def _get_parser(self):
        """Get the static _try_parse_json method without initializing ClaudeTrader
        (which requires Ollama)."""
        from core.claude_trader import ClaudeTrader
        return ClaudeTrader._try_parse_json

    def test_valid_json(self):
        parse = self._get_parser()
        data = {
            "decision": "BUY",
            "confidence": 8,
            "entry_price": 1.1050,
            "stop_loss": 1.1020,
            "take_profit": 1.1110,
            "lot_size": 0.01,
            "reasoning": "Strong bullish signal",
        }
        result = parse(json.dumps(data))
        assert result is not None
        assert result["decision"] == "BUY"
        assert result["confidence"] == 8

    def test_json_with_markdown_fences(self):
        parse = self._get_parser()
        raw = '```json\n{"decision": "SELL", "confidence": 6}\n```'
        result = parse(raw)
        assert result is not None
        assert result["decision"] == "SELL"

    def test_json_with_plain_fences(self):
        parse = self._get_parser()
        raw = '```\n{"decision": "HOLD", "confidence": 3}\n```'
        result = parse(raw)
        assert result is not None
        assert result["decision"] == "HOLD"

    def test_json_with_extra_text(self):
        parse = self._get_parser()
        raw = 'Here is my analysis:\n{"decision": "BUY", "confidence": 7}\nThank you.'
        result = parse(raw)
        assert result is not None
        assert result["decision"] == "BUY"

    def test_invalid_json_returns_none(self):
        parse = self._get_parser()
        result = parse("This is not JSON at all")
        assert result is None

    def test_empty_string_returns_none(self):
        parse = self._get_parser()
        result = parse("")
        assert result is None

    def test_partial_json_returns_none(self):
        parse = self._get_parser()
        result = parse('{"decision": "BUY", "confidence":')
        assert result is None

    def test_nested_json_extracts_outer(self):
        parse = self._get_parser()
        raw = '{"decision": "BUY", "confidence": 7, "key_risks": ["risk1", "risk2"]}'
        result = parse(raw)
        assert result is not None
        assert result["decision"] == "BUY"
        assert len(result["key_risks"]) == 2
