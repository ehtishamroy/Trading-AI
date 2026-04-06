"""
LLM Trading Integration — Supports both FREE Ollama and paid Claude API.

Modes (set LLM_PROVIDER in .env):
  - "ollama"  → Free local LLM (default)
  - "claude"  → Anthropic Claude API (cost-optimized: single Haiku call)

Cost savings with Claude API:
  1. Single call (no Bull/Bear debate) — 1 API call instead of 3-4
  2. Uses claude-haiku-4-5-20251001 (cheapest model)
  3. Minimal prompt (~500 tokens input)
  4. Max 500 tokens output
  5. Pre-score skip in main.py avoids calling when unnecessary
  6. Strong Aegis skip avoids calling when ML is already confident
"""

import json
import os
import requests
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_MAX_TOKENS

# Claude API settings (from .env)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "claude"
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")  # Cheapest model
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "500"))  # Keep responses short


class ClaudeTrader:
    """
    LLM-powered trading verdict system.
    - Ollama mode: 3-agent debate (free but slow)
    - Claude mode: Single optimized call (fast, cheap)
    """

    def __init__(self):
        self.provider = LLM_PROVIDER

        if self.provider == "claude":
            if CLAUDE_API_KEY:
                logger.info(f"✅ Claude API mode — Model: {CLAUDE_MODEL} (cost-optimized)")
                self.available = True
            else:
                logger.warning("CLAUDE_API_KEY not set — falling back to Ollama")
                self.provider = "ollama"
                self._init_ollama()
        else:
            self._init_ollama()

    def _init_ollama(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self.max_tokens = OLLAMA_MAX_TOKENS
        if self._check_ollama():
            logger.info(f"✅ Ollama connected — Model: {self.model}")
            self.available = True
        else:
            logger.warning(
                f"⚠️  Ollama not reachable at {self.base_url}. "
                "Start Ollama: https://ollama.com | then run: ollama pull llama3.1:8b"
            )
            self.available = False

    def _check_ollama(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                base = self.model.split(":")[0]
                if any(base in m for m in models):
                    return True
                logger.warning(f"Model '{self.model}' not found locally. Available: {models}")
                return False
            return False
        except requests.exceptions.ConnectionError:
            return False
        except Exception as e:
            logger.warning(f"Ollama check error: {e}")
            return False

    # ─── COMPACT JUDGE PROMPT (saves tokens) ──────────────

    JUDGE_PROMPT_COMPACT = """You are a quant trader. Analyze the signals and give a trading verdict.
Rules: Max 2% risk, 2:1 reward/risk minimum, 0.01 lots.
This is DEMO mode — lean towards trading when ML models agree strongly.

Respond ONLY with JSON:
{"decision":"BUY/SELL/HOLD","confidence":1-10,"entry_price":0,"stop_loss":0,"take_profit":0,"lot_size":0.01,"reasoning":"brief explanation","key_risks":["risk1"]}"""

    # ─── Full prompts for Ollama (free, can afford verbosity) ──

    TRADER_BASE = """You are a professional quantitative trader with 20 years experience.
You think in terms of risk/reward, market regimes, and capital preservation.

Core Rules:
- Never risk more than 2% of portfolio per trade
- Minimum 2:1 reward-to-risk ratio
- Cut losers fast, let winners run

You are trading on an Exness MT5 micro-account (0.01 lots), 1:100 leverage.

IMPORTANT: This is a DEMO learning phase. The system NEEDS trades to build pattern memory.
- Be willing to take trades at 5+/10 confidence
- When ML models AGREE with high confidence, lean towards trading rather than holding
- Only say HOLD if you see a clear and specific risk (not general uncertainty)
- "Lack of historical patterns" is NOT a reason to HOLD"""

    BULL_PROMPT = TRADER_BASE + """

YOUR ROLE: Agent BULL — build the STRONGEST case for BUYING.
Be passionate but data-driven. Cite specific numbers."""

    BEAR_PROMPT = TRADER_BASE + """

YOUR ROLE: Agent BEAR — build the STRONGEST case AGAINST buying (or for SELLING).
Find risks, traps, fake breakouts. Protect capital."""

    JUDGE_PROMPT = TRADER_BASE + """

YOUR ROLE: Agent JUDGE — render final verdict as JSON.
You MUST respond in this exact JSON format and NOTHING else:
{"decision":"BUY/SELL/HOLD","confidence":1-10,"entry_price":0,"stop_loss":0,"take_profit":0,"lot_size":0.01,"reasoning":"explanation","bull_strength":"STRONG/MODERATE/WEAK","bear_strength":"STRONG/MODERATE/WEAK","pre_mortem":"what could go wrong","key_risks":["risk1","risk2"],"what_changes_mind":"exit condition"}"""

    # ─── MAIN ENTRY POINT ────────────────────────────────

    def run_debate(self, market_context: str) -> dict:
        """Run LLM analysis. Uses single call for Claude API, 3-call debate for Ollama."""
        if not self.available:
            return self._mock_response()

        if self.provider == "claude":
            return self._run_claude_single(market_context)
        else:
            return self._run_ollama_debate(market_context)

    # ─── CLAUDE API (COST-OPTIMIZED) ─────────────────────

    def _run_claude_single(self, market_context: str) -> dict:
        """Single Claude API call — ~500 input tokens, ~300 output tokens.
        Cost: ~$0.001 per call with Haiku."""
        logger.info(f"⚖️  Claude API single-call verdict ({CLAUDE_MODEL})...")

        response = self._call_claude(
            system=self.JUDGE_PROMPT_COMPACT,
            user=market_context
        )

        verdict = self._parse_verdict(response)

        result = {
            "verdict": verdict,
            "bull_case": "Single-call mode (no debate)",
            "bear_case": "Single-call mode (no debate)",
            "judge_raw": response,
        }

        logger.info(
            f"⚖️  Verdict: {verdict.get('decision', 'HOLD')} | "
            f"Confidence: {verdict.get('confidence', 0)}/10"
        )
        return result

    def _call_claude(self, system: str, user: str) -> str:
        """Make a single Anthropic Claude API call."""
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": CLAUDE_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": CLAUDE_MAX_TOKENS,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]
        except requests.exceptions.Timeout:
            logger.error("Claude API timed out")
            return "Error: Claude timeout"
        except requests.exceptions.HTTPError as e:
            logger.error(f"Claude API error: {e} — {resp.text[:200]}")
            return f"Error: {e}"
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Error: {e}"

    # ─── OLLAMA (FREE) ───────────────────────────────────

    def _run_ollama_debate(self, market_context: str) -> dict:
        """Full 3-agent debate via Ollama (free)."""
        logger.info("🟢 Agent BULL building case...")
        bull_case = self._call_ollama(
            system=self.BULL_PROMPT,
            user=f"Analyze this setup and make your strongest case for BUYING:\n\n{market_context}"
        )

        logger.info("🔴 Agent BEAR building case...")
        bear_case = self._call_ollama(
            system=self.BEAR_PROMPT,
            user=f"Analyze this setup and make your strongest case AGAINST buying (or for SELLING):\n\n{market_context}"
        )

        logger.info("⚖️  Agent JUDGE rendering verdict...")
        judge_context = (
            f"## Market Data & Signals\n{market_context}\n\n"
            f"---\n\n"
            f"## Agent BULL's Case\n{bull_case}\n\n"
            f"---\n\n"
            f"## Agent BEAR's Case\n{bear_case}\n\n"
            f"---\n\n"
            f"Now render your verdict. Respond ONLY with JSON. No preamble, no markdown fences."
        )

        judge_response = self._call_ollama(
            system=self.JUDGE_PROMPT,
            user=judge_context
        )

        verdict = self._parse_verdict(judge_response, retry_context=judge_context)

        result = {
            "verdict": verdict,
            "bull_case": bull_case,
            "bear_case": bear_case,
            "judge_raw": judge_response,
        }

        logger.info(
            f"⚖️  Verdict: {verdict.get('decision', 'HOLD')} | "
            f"Confidence: {verdict.get('confidence', 0)}/10"
        )
        return result

    def _call_ollama(self, system: str, user: str) -> str:
        """Make a single Ollama API call."""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {
                    "num_predict": self.max_tokens,
                    "temperature": 0.7,
                }
            }
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "Error: Ollama timeout"
        except requests.exceptions.ConnectionError:
            logger.error("Ollama connection refused — is Ollama running?")
            self.available = False
            return "Error: Ollama not running"
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return f"Error: {e}"

    # ─── SHARED METHODS ──────────────────────────────────

    def debrief_trade(self, trade_info: str) -> str:
        """Post-trade analysis."""
        if not self.available:
            return "LLM unavailable — debrief skipped"

        prompt = (
            f"Review this trade briefly (3 sentences max):\n{trade_info}\n"
            "What pattern should we remember? Rate quality 1-10."
        )

        if self.provider == "claude":
            return self._call_claude(
                system="You are a trade reviewer. Be concise.",
                user=prompt
            )
        return self._call_ollama(system=self.TRADER_BASE, user=prompt)

    def score_central_bank(self, transcript: str) -> dict:
        """Score a central bank speech as hawkish/dovish."""
        if not self.available:
            return {"score": 0, "label": "Neutral"}

        prompt = (
            f"Score this central bank statement. Respond ONLY with JSON:\n"
            f'{{"score": -5 to +5, "label": "Hawkish/Dovish/Neutral", "impact": "bullish/bearish/neutral"}}\n\n'
            f"{transcript[:1000]}"
        )

        if self.provider == "claude":
            response = self._call_claude(
                system="Central bank analyst. JSON only.",
                user=prompt
            )
        else:
            response = self._call_ollama(
                system="You are a central bank policy analyst. Respond only with JSON.",
                user=prompt
            )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"score": 0, "label": "Neutral", "impact": "neutral"}

    def _parse_verdict(self, response: str, retry_context: str = None) -> dict:
        """Parse JSON verdict. Retries once on failure."""
        result = self._try_parse_json(response)
        if result is not None:
            return result

        logger.warning("Could not parse verdict JSON — retrying with stricter prompt")

        if self.available and retry_context:
            retry_prompt = (
                f"Convert to JSON only: {{\"decision\":\"BUY/SELL/HOLD\",\"confidence\":1-10,"
                f"\"entry_price\":0,\"stop_loss\":0,\"take_profit\":0,\"lot_size\":0.01,"
                f"\"reasoning\":\"text\"}}\n\nText:\n{response[:1000]}"
            )

            if self.provider == "claude":
                retry_response = self._call_claude(
                    system="JSON formatter. Output ONLY valid JSON.",
                    user=retry_prompt
                )
            else:
                retry_response = self._call_ollama(
                    system="JSON formatter. Output ONLY valid JSON.",
                    user=retry_prompt
                )

            result = self._try_parse_json(retry_response)
            if result is not None:
                logger.info("Successfully parsed verdict on retry")
                return result

        logger.warning("Verdict parse failed — defaulting to HOLD")
        return {
            "decision": "HOLD",
            "confidence": 0,
            "reasoning": response[:200] if response else "No response",
            "error": "Failed to parse JSON"
        }

    @staticmethod
    def _try_parse_json(response: str) -> dict | None:
        """Extract and parse JSON from a response string."""
        try:
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            if "{" in text and "}" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                text = text[start:end]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError, ValueError):
            return None

    def _mock_response(self) -> dict:
        """Return mock response when no LLM is available."""
        return {
            "verdict": {
                "decision": "HOLD",
                "confidence": 0,
                "reasoning": "No LLM available. Set CLAUDE_API_KEY or start Ollama.",
            },
            "bull_case": "N/A — LLM not available",
            "bear_case": "N/A — LLM not available",
        }


def build_market_context(
    market: str,
    current_price: dict,
    ensemble_signal: str,
    regime_info: str,
    sentiment_info: str,
    account_info: dict = None,
    pattern_memory: str = "",
) -> str:
    """Build compact market context for LLM. Kept short to save tokens."""
    balance = account_info.get('balance', 500) if account_info else 500
    context = (
        f"# {market} | Bid:{current_price.get('bid','N/A')} Ask:{current_price.get('ask','N/A')} "
        f"Spread:{current_price.get('spread','N/A')}\n"
        f"Balance:${balance} | Risk:${balance*0.02:.0f} | Leverage:1:100\n\n"
        f"{ensemble_signal}\n\n{regime_info}\n\n{sentiment_info}"
    )

    if pattern_memory:
        context += f"\n\n{pattern_memory}"

    return context
