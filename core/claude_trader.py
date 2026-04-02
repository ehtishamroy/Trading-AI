"""
Ollama Local LLM Integration — Multi-Agent Debate System.

Three AI agents debate each trade using a FREE local LLM via Ollama:
  🟢 Agent BULL — Makes the strongest case for BUYING
  🔴 Agent BEAR — Makes the strongest case for SELLING
  ⚖️ Agent JUDGE — Reads both + data → renders final verdict

100% FREE — No API keys required.
Requires Ollama running locally: https://ollama.com
Model: llama3.1:8b (or any other Ollama model)
"""

import json
import requests
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_MAX_TOKENS


class ClaudeTrader:
    """
    Multi-Agent Debate system using Ollama local LLM.
    Drop-in replacement for the Claude API version.
    Each trade goes through a Bull/Bear/Judge debate
    before a final recommendation is made.
    """

    def __init__(self):
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
                # Accept any variant of the model name (e.g. llama3.1:8b, llama3.1:latest)
                base = self.model.split(":")[0]
                if any(base in m for m in models):
                    return True
                # Model list fetched but model not found — try to pull it automatically
                logger.warning(f"Model '{self.model}' not found locally. Available: {models}")
                logger.info(f"Run this command to pull it: ollama pull {self.model}")
                return False
            return False
        except requests.exceptions.ConnectionError:
            return False
        except Exception as e:
            logger.warning(f"Ollama check error: {e}")
            return False

    # ─── SYSTEM PROMPTS ──────────────────────────────────

    TRADER_BASE = """You are a professional quantitative trader with 20 years experience.
You think in terms of risk/reward, market regimes, and capital preservation.

Core Rules you ALWAYS follow:
- Never risk more than 2% of portfolio per trade
- In high volatility, reduce position size by 50%
- Require minimum 2:1 reward-to-risk ratio
- Never chase a trade — if entry is missed, move on
- Cut losers fast, let winners run
- Never average down on a losing position

You are trading on an Exness MT5 micro-account (0.01 lots).
Leverage is 1:100. Your current account balance and risk limits will be provided in the 'Account' section below. Every pip matters at your account size. This is a demo learning phase — be willing to take trades at 5+/10 confidence."""

    BULL_PROMPT = TRADER_BASE + """

YOUR ROLE: You are Agent BULL 🟢
Your job is to build the STRONGEST possible case for BUYING.
Find every reason the price should go UP.
Use the ML signals, technical indicators, news, and market regime.
Be passionate but data-driven. Cite specific numbers.
If you cannot make a compelling bull case, say so honestly."""

    BEAR_PROMPT = TRADER_BASE + """

YOUR ROLE: You are Agent BEAR 🔴
Your job is to build the STRONGEST possible case for SELLING / avoiding this trade.
Find every reason the price should go DOWN or the trade is risky.
Challenge the ML signals. Look for divergences, traps, and fake breakouts.
Consider news risks, upcoming events, and regime shifts.
Be the devil's advocate. Your job is to protect capital."""

    JUDGE_PROMPT = TRADER_BASE + """

YOUR ROLE: You are Agent JUDGE ⚖️
You have read both the Bull and Bear arguments.
You also have the raw data, ML signals, and market context.

Your job:
1. Weigh both arguments fairly
2. Decide: BUY, SELL, or HOLD (no trade)
3. If trading, provide exact entry, stop loss, and take profit prices
4. Score your confidence 1-10 (only trade if 7+)
5. Explain your reasoning in plain English

PRE-MORTEM: Before finalizing, imagine this trade just hit stop loss.
What went wrong? If you can easily explain the failure,
reduce your confidence score.

You MUST respond in this exact JSON format and NOTHING else — no markdown, no explanation, just the JSON:
{
    "decision": "BUY" or "SELL" or "HOLD",
    "confidence": 1-10,
    "entry_price": number,
    "stop_loss": number,
    "take_profit": number,
    "lot_size": 0.01,
    "reasoning": "Your plain-English explanation",
    "bull_strength": "STRONG/MODERATE/WEAK",
    "bear_strength": "STRONG/MODERATE/WEAK",
    "pre_mortem": "What could go wrong",
    "key_risks": ["risk 1", "risk 2"],
    "what_changes_mind": "What would cause you to exit early"
}"""

    # ─── DEBATE SYSTEM ───────────────────────────────────

    def run_debate(self, market_context: str) -> dict:
        """
        Run the full Bull vs Bear vs Judge debate.

        Args:
            market_context: Formatted string with all signals, indicators, news

        Returns:
            {
                verdict: Judge's decision dict,
                bull_case: Bull's argument,
                bear_case: Bear's argument,
                reasoning: Full reasoning
            }
        """
        if not self.available:
            return self._mock_response()

        # Step 1: Agent BULL makes the case for buying
        logger.info("🟢 Agent BULL building case...")
        bull_case = self._call_ollama(
            system=self.BULL_PROMPT,
            user=f"Analyze this setup and make your strongest case for BUYING:\n\n{market_context}"
        )

        # Step 2: Agent BEAR makes the case against
        logger.info("🔴 Agent BEAR building case...")
        bear_case = self._call_ollama(
            system=self.BEAR_PROMPT,
            user=f"Analyze this setup and make your strongest case AGAINST buying (or for SELLING):\n\n{market_context}"
        )

        # Step 3: Agent JUDGE reads both and decides
        logger.info("⚖️  Agent JUDGE rendering verdict...")
        judge_context = (
            f"## Market Data & Signals\n{market_context}\n\n"
            f"---\n\n"
            f"## Agent BULL's Case 🟢\n{bull_case}\n\n"
            f"---\n\n"
            f"## Agent BEAR's Case 🔴\n{bear_case}\n\n"
            f"---\n\n"
            f"Now render your verdict. Respond ONLY with the JSON format specified. No preamble, no markdown fences."
        )

        judge_response = self._call_ollama(
            system=self.JUDGE_PROMPT,
            user=judge_context
        )

        # Parse Judge's JSON response (with retry on failure)
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

    def debrief_trade(self, trade_info: str) -> str:
        """
        Post-trade analysis. LLM reviews what happened and why.
        Feeds into the self-learning system.
        """
        if not self.available:
            return "Ollama unavailable — debrief skipped"

        debrief_prompt = f"""You are reviewing a completed trade. Analyze what happened:

{trade_info}

Provide:
1. Was the original thesis correct?
2. What did the ML model get right/wrong?
3. What pattern should we remember for next time?
4. Rate the trade quality (1-10) regardless of outcome
5. One specific lesson learned

Be concise and actionable."""

        return self._call_ollama(
            system=self.TRADER_BASE,
            user=debrief_prompt
        )

    def score_central_bank(self, transcript: str) -> dict:
        """Score a central bank speech/statement as hawkish/dovish."""
        if not self.available:
            return {"score": 0, "label": "Neutral"}

        prompt = f"""Analyze this central bank statement and score it:

{transcript}

Respond ONLY with JSON (no markdown, no explanation):
{{"score": -5 to +5, "label": "Very Hawkish" to "Very Dovish", "key_phrases": ["phrase1", "phrase2"], "impact": "bullish/bearish/neutral for EUR/USD"}}"""

        response = self._call_ollama(
            system="You are a central bank policy analyst. Respond only with JSON.",
            user=prompt
        )
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"score": 0, "label": "Neutral", "key_phrases": [], "impact": "neutral"}

    # ─── INTERNAL METHODS ────────────────────────────────

    def _call_ollama(self, system: str, user: str) -> str:
        """Make a single Ollama API call using the /api/chat endpoint."""
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
                timeout=120  # Local LLM can be slow on first run
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out — model may still be loading, try again")
            return "Error: Ollama timeout"
        except requests.exceptions.ConnectionError:
            logger.error("Ollama connection refused — is Ollama running? Start it first.")
            self.available = False
            return "Error: Ollama not running"
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return f"Error: {e}"

    def _parse_verdict(self, response: str, retry_context: str = None) -> dict:
        """Parse Judge's JSON response, handling markdown code blocks.
        If parsing fails and retry_context is provided, re-prompts Ollama once."""
        result = self._try_parse_json(response)
        if result is not None:
            return result

        logger.warning("Could not parse Judge's response as JSON — retrying with stricter prompt")

        # Retry once with a stricter prompt if Ollama is available
        if self.available and retry_context:
            retry_response = self._call_ollama(
                system="You are a JSON formatter. Respond ONLY with valid JSON. No markdown, no explanation, no code fences.",
                user=(
                    f"Convert this trading verdict into the exact JSON format below. "
                    f"Output ONLY the JSON object, nothing else.\n\n"
                    f"Required format: {{\"decision\": \"BUY/SELL/HOLD\", \"confidence\": 1-10, "
                    f"\"entry_price\": number, \"stop_loss\": number, \"take_profit\": number, "
                    f"\"lot_size\": 0.01, \"reasoning\": \"text\"}}\n\n"
                    f"Text to convert:\n{response[:1500]}"
                )
            )
            result = self._try_parse_json(retry_response)
            if result is not None:
                logger.info("Successfully parsed verdict on retry")
                return result

        logger.warning("Verdict parse failed after retry — defaulting to HOLD")
        return {
            "decision": "HOLD",
            "confidence": 0,
            "reasoning": response,
            "error": "Failed to parse JSON response"
        }

    @staticmethod
    def _try_parse_json(response: str) -> dict | None:
        """Attempt to extract and parse JSON from a response string."""
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
        """Return mock response when Ollama is not available."""
        return {
            "verdict": {
                "decision": "HOLD",
                "confidence": 0,
                "reasoning": (
                    "Ollama LLM not available. "
                    "Install Ollama from https://ollama.com then run: ollama pull llama3.1:8b"
                ),
            },
            "bull_case": "N/A — Ollama not running",
            "bear_case": "N/A — Ollama not running",
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
    """
    Build the complete market context string that feeds into the debate.
    This is what all three LLM agents receive to analyze.
    """
    context = f"""# Trade Analysis — {market}

## Current State
- **Price**: Bid {current_price.get('bid', 'N/A')} / Ask {current_price.get('ask', 'N/A')}
- **Spread**: {current_price.get('spread', 'N/A')}
- **Time**: {current_price.get('time', 'N/A')}

## Account
- **Balance**: ${account_info.get('balance', 500) if account_info else 500}
- **Open Positions**: {account_info.get('positions', 0) if account_info else 0}
- **Leverage**: 1:100
- **Risk per trade**: 2% (${account_info.get('balance', 500) * 0.02:.2f} if account_info else '$10')

{ensemble_signal}

{regime_info}

{sentiment_info}
"""

    if pattern_memory:
        context += f"\n{pattern_memory}\n"

    return context
