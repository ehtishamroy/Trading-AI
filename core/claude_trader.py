"""
Claude AI Integration — Multi-Agent Debate System.

Three Claude agents debate each trade:
  🟢 Agent BULL — Makes the strongest case for BUYING
  🔴 Agent BEAR — Makes the strongest case for SELLING
  ⚖️ Agent JUDGE — Reads both + data → renders final verdict

This kills confirmation bias and forces adversarial thinking.
"""

import anthropic
import json
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import CLAUDE_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS


class ClaudeTrader:
    """
    Multi-Agent Debate system using Claude AI.
    Each trade goes through a Bull/Bear/Judge debate
    before a final recommendation is made.
    """

    def __init__(self):
        if not CLAUDE_API_KEY:
            logger.warning("No CLAUDE_API_KEY set — Claude features disabled")
            self.client = None
            return
        self.client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        logger.info("Claude AI connected")

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

You are trading with $200 capital using micro lots (0.01) on Exness MT5.
Leverage is 1:100. Every pip matters at this account size."""

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

You MUST respond in this exact JSON format:
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
        if not self.client:
            return self._mock_response()

        # Step 1: Agent BULL makes the case for buying
        logger.info("🟢 Agent BULL building case...")
        bull_case = self._call_claude(
            system=self.BULL_PROMPT,
            user=f"Analyze this setup and make your strongest case for BUYING:\n\n{market_context}"
        )

        # Step 2: Agent BEAR makes the case against
        logger.info("🔴 Agent BEAR building case...")
        bear_case = self._call_claude(
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
            f"Now render your verdict. Respond ONLY with the JSON format specified."
        )

        judge_response = self._call_claude(
            system=self.JUDGE_PROMPT,
            user=judge_context
        )

        # Parse Judge's JSON response
        verdict = self._parse_verdict(judge_response)

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
        Post-trade analysis. Claude reviews what happened and why.
        Feeds into the self-learning system.
        """
        if not self.client:
            return "Claude unavailable — debrief skipped"

        debrief_prompt = f"""You are reviewing a completed trade. Analyze what happened:

{trade_info}

Provide:
1. Was the original thesis correct?
2. What did the ML model get right/wrong?
3. What pattern should we remember for next time?
4. Rate the trade quality (1-10) regardless of outcome
5. One specific lesson learned

Be concise and actionable."""

        return self._call_claude(
            system=self.TRADER_BASE,
            user=debrief_prompt
        )

    def score_central_bank(self, transcript: str) -> dict:
        """Score a central bank speech/statement as hawkish/dovish."""
        if not self.client:
            return {"score": 0, "label": "Neutral"}

        prompt = f"""Analyze this central bank statement and score it:

{transcript}

Respond with JSON:
{{"score": -5 to +5, "label": "Very Hawkish" to "Very Dovish", "key_phrases": ["phrase1", "phrase2"], "impact": "bullish/bearish/neutral for EUR/USD"}}"""

        response = self._call_claude(system="You are a central bank policy analyst.", user=prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"score": 0, "label": "Neutral", "key_phrases": [], "impact": "neutral"}

    # ─── INTERNAL METHODS ────────────────────────────────

    def _call_claude(self, system: str, user: str) -> str:
        """Make a single Claude API call."""
        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Error: {e}"

    def _parse_verdict(self, response: str) -> dict:
        """Parse Judge's JSON response, handling markdown code blocks."""
        try:
            # Try to extract JSON from response
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            logger.warning("Could not parse Judge's response as JSON")
            return {
                "decision": "HOLD",
                "confidence": 0,
                "reasoning": response,
                "error": "Failed to parse JSON response"
            }

    def _mock_response(self) -> dict:
        """Return mock response when Claude API is not available."""
        return {
            "verdict": {
                "decision": "HOLD",
                "confidence": 0,
                "reasoning": "Claude API not configured. Set CLAUDE_API_KEY in .env",
            },
            "bull_case": "N/A — Claude not connected",
            "bear_case": "N/A — Claude not connected",
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
    This is what all three Claude agents receive to analyze.
    """
    context = f"""# Trade Analysis — {market}

## Current State
- **Price**: Bid {current_price.get('bid', 'N/A')} / Ask {current_price.get('ask', 'N/A')}
- **Spread**: {current_price.get('spread', 'N/A')}
- **Time**: {current_price.get('time', 'N/A')}

## Account
- **Balance**: ${account_info.get('balance', 200) if account_info else 200}
- **Open Positions**: {account_info.get('positions', 0) if account_info else 0}
- **Leverage**: 1:100
- **Risk per trade**: 2% (${account_info.get('balance', 200) * 0.02:.2f} if account_info else '$4')

{ensemble_signal}

{regime_info}

{sentiment_info}
"""

    if pattern_memory:
        context += f"\n{pattern_memory}\n"

    return context
