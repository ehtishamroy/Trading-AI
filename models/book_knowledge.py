"""
Book Knowledge Module (Phase 4b) — Encode trading book strategies as features.
Feed trading books into the system so it can learn from established patterns.

Supports:
- Candlestick patterns (Steve Nison methodology)
- Price action / ICT concepts (Order Blocks, Fair Value Gaps, BOS)
- Custom strategies loaded from JSON config
"""

import pandas as pd
import numpy as np
from loguru import logger


class BookKnowledge:
    """
    Encodes trading book patterns as ML-ready features.
    Also provides Claude-readable strategy descriptions.
    """

    def __init__(self):
        self.detected_patterns = []

    def compute_book_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all book-based features to the DataFrame.
        These features come from established trading literature.
        """
        df = df.copy()

        # ═══ CANDLESTICK PATTERNS (Steve Nison) ═════════════
        df = self._nison_patterns(df)

        # ═══ ICT CONCEPTS ═══════════════════════════════════
        df = self._ict_order_blocks(df)
        df = self._ict_fair_value_gaps(df)
        df = self._ict_break_of_structure(df)
        df = self._ict_liquidity_sweeps(df)

        # ═══ SUPPLY & DEMAND ZONES ══════════════════════════
        df = self._supply_demand_zones(df)

        # ═══ DIVERGENCES ════════════════════════════════════
        df = self._advanced_divergences(df)

        logger.info(f"Book features computed: {len([c for c in df.columns if c.startswith('book_')])} patterns")
        return df

    def _nison_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Steve Nison's Japanese Candlestick patterns."""
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body = abs(c - o)
        upper_shadow = h - c.combine(o, max)
        lower_shadow = c.combine(o, min) - l
        candle_range = h - l + 1e-9

        # Morning Star (3-candle bullish reversal)
        bearish_1 = (c.shift(2) < o.shift(2)) & (body.shift(2) > candle_range.shift(2) * 0.5)
        small_2 = body.shift(1) < candle_range.shift(1) * 0.3
        bullish_3 = (c > o) & (c > (o.shift(2) + c.shift(2)) / 2)
        df["book_morning_star"] = (bearish_1 & small_2 & bullish_3).astype(int)

        # Evening Star (3-candle bearish reversal)
        bullish_1 = (c.shift(2) > o.shift(2)) & (body.shift(2) > candle_range.shift(2) * 0.5)
        small_2 = body.shift(1) < candle_range.shift(1) * 0.3
        bearish_3 = (c < o) & (c < (o.shift(2) + c.shift(2)) / 2)
        df["book_evening_star"] = (bullish_1 & small_2 & bearish_3).astype(int)

        # Three White Soldiers (strong bullish continuation)
        bull_1 = c.shift(2) > o.shift(2)
        bull_2 = c.shift(1) > o.shift(1)
        bull_3 = c > o
        increasing = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        df["book_three_white_soldiers"] = (bull_1 & bull_2 & bull_3 & increasing).astype(int)

        # Three Black Crows (strong bearish continuation)
        bear_1 = c.shift(2) < o.shift(2)
        bear_2 = c.shift(1) < o.shift(1)
        bear_3 = c < o
        decreasing = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
        df["book_three_black_crows"] = (bear_1 & bear_2 & bear_3 & decreasing).astype(int)

        # Spinning Top (indecision)
        df["book_spinning_top"] = (
            (body / candle_range < 0.3) &
            (upper_shadow > body * 0.5) &
            (lower_shadow > body * 0.5)
        ).astype(int)

        # Marubozu (strong conviction)
        df["book_bullish_marubozu"] = (
            (c > o) & (lower_shadow < body * 0.05) & (upper_shadow < body * 0.05)
        ).astype(int)
        df["book_bearish_marubozu"] = (
            (c < o) & (lower_shadow < body * 0.05) & (upper_shadow < body * 0.05)
        ).astype(int)

        # Tweezer Bottoms/Tops
        df["book_tweezer_bottom"] = (
            (abs(l - l.shift(1)) < candle_range * 0.05) &
            (c.shift(1) < o.shift(1)) & (c > o)
        ).astype(int)
        df["book_tweezer_top"] = (
            (abs(h - h.shift(1)) < candle_range * 0.05) &
            (c.shift(1) > o.shift(1)) & (c < o)
        ).astype(int)

        return df

    def _ict_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """ICT Order Blocks — institutional footprints."""
        c, o = df["close"], df["open"]

        # Bullish OB: Last bearish candle before a strong bullish move
        strong_up = (c - c.shift(1)) > df["close"].rolling(20).std() * 1.5
        prev_bearish = c.shift(1) < o.shift(1)
        df["book_bullish_ob"] = (strong_up & prev_bearish).astype(int)

        # Bearish OB: Last bullish candle before a strong bearish move
        strong_down = (c.shift(1) - c) > df["close"].rolling(20).std() * 1.5
        prev_bullish = c.shift(1) > o.shift(1)
        df["book_bearish_ob"] = (strong_down & prev_bullish).astype(int)

        return df

    def _ict_fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """ICT Fair Value Gaps — price inefficiencies."""
        h, l = df["high"], df["low"]

        # Bullish FVG: gap between candle 1 high and candle 3 low
        df["book_bullish_fvg"] = (l > h.shift(2)).astype(int)
        df["book_fvg_size"] = np.where(
            df["book_bullish_fvg"] == 1,
            (l - h.shift(2)) / df["close"],
            0
        )

        # Bearish FVG
        df["book_bearish_fvg"] = (h < l.shift(2)).astype(int)

        return df

    def _ict_break_of_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """ICT Break of Structure — trend shift signals."""
        h, l = df["high"], df["low"]

        # Higher high break (bullish BOS)
        prev_high = h.rolling(20).max().shift(1)
        df["book_bullish_bos"] = (h > prev_high).astype(int)

        # Lower low break (bearish BOS)
        prev_low = l.rolling(20).min().shift(1)
        df["book_bearish_bos"] = (l < prev_low).astype(int)

        return df

    def _ict_liquidity_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        """ICT Liquidity Sweeps — stop hunts."""
        h, l, c = df["high"], df["low"], df["close"]

        # Sweep below previous low then close above (bull trap resolved)
        prev_low = l.rolling(10).min().shift(1)
        df["book_bull_sweep"] = ((l < prev_low) & (c > prev_low)).astype(int)

        # Sweep above previous high then close below (bear trap resolved)
        prev_high = h.rolling(10).max().shift(1)
        df["book_bear_sweep"] = ((h > prev_high) & (c < prev_high)).astype(int)

        return df

    def _supply_demand_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supply and demand zone proximity."""
        h, l, c = df["high"], df["low"], df["close"]

        # Demand zone: area where big bullish candle originated
        big_bull = (c - df["open"]) > df["close"].rolling(20).std() * 2
        df["book_at_demand"] = (
            big_bull.shift(5).rolling(5).max().fillna(0) &
            (c < c.shift(5) * 1.005)  # Price returned to the zone
        ).astype(int)

        # Supply zone: area where big bearish candle originated
        big_bear = (df["open"] - c) > df["close"].rolling(20).std() * 2
        df["book_at_supply"] = (
            big_bear.shift(5).rolling(5).max().fillna(0) &
            (c > c.shift(5) * 0.995)
        ).astype(int)

        return df

    def _advanced_divergences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hidden divergences — continuation signals."""
        c = df["close"]
        rsi = df.get("rsi_14")
        if rsi is None:
            return df

        # Hidden bullish: higher low in price + lower low in RSI
        df["book_hidden_bull_div"] = (
            (c > c.shift(10)) & (rsi < rsi.shift(10)) &
            (c.shift(5) < c) & (c.shift(5) < c.shift(10))
        ).astype(int)

        # Hidden bearish: lower high in price + higher high in RSI
        df["book_hidden_bear_div"] = (
            (c < c.shift(10)) & (rsi > rsi.shift(10)) &
            (c.shift(5) > c) & (c.shift(5) > c.shift(10))
        ).astype(int)

        return df

    def get_book_feature_columns(self) -> list:
        """Return column names of all book-based features."""
        return [
            # Nison candlestick patterns
            "book_morning_star", "book_evening_star",
            "book_three_white_soldiers", "book_three_black_crows",
            "book_spinning_top",
            "book_bullish_marubozu", "book_bearish_marubozu",
            "book_tweezer_bottom", "book_tweezer_top",
            # ICT concepts
            "book_bullish_ob", "book_bearish_ob",
            "book_bullish_fvg", "book_bearish_fvg", "book_fvg_size",
            "book_bullish_bos", "book_bearish_bos",
            "book_bull_sweep", "book_bear_sweep",
            # Supply/Demand
            "book_at_demand", "book_at_supply",
            # Divergences
            "book_hidden_bull_div", "book_hidden_bear_div",
        ]

    def get_claude_knowledge(self) -> str:
        """
        Book knowledge injected into Claude's system prompt.
        Claude references these strategies during analysis.
        Automatically loads text from uploaded books in data/books.
        """
        # Base built-in knowledge
        knowledge = """
## Trading Book Knowledge (Injected Strategies)

### Steve Nison — Japanese Candlestick Charting
- Morning Star at support = strong bullish reversal signal
- Evening Star at resistance = strong bearish reversal signal
- Three White Soldiers = strong bullish continuation, especially after a downtrend
- Three Black Crows = strong bearish continuation
- Doji after a strong trend = potential reversal, wait for confirmation
- Marubozu = very strong conviction candle, trade in its direction

### ICT (Inner Circle Trader) Concepts
- Order Blocks: last opposing candle before a big move = institutional entry zone
- Fair Value Gaps: price gaps that tend to get filled — trade the fill
- Break of Structure: when price breaks a key swing high/low = trend shift
- Liquidity Sweeps: price hunts stop losses then reverses — enter AFTER the sweep
- Look for "smart money" entering at order blocks after a liquidity sweep

### Supply & Demand
- Demand zone = area where price exploded upward — expect buyers there again
- Supply zone = area where price crashed downward — expect sellers there again
- Fresh zones (untested) are stronger than zones already visited

### Key Rules from Trading Psychology (Mark Douglas)
- Every trade is independent — past results don't predict next trade
- Think in probabilities, not certainties
- The market can stay irrational longer than you can stay solvent
"""
        
        # Load user-uploaded books from data/books directory
        try:
            from pathlib import Path
            import json
            
            books_dir = Path("data/books")
            if books_dir.exists():
                custom_books = list(books_dir.glob("*.json"))
                if custom_books:
                    knowledge += "\n\n### USER UPLOADED STRATEGY BOOKS MAPPED INTO KNOWLEDGE (Use these rules fiercely)\n\n"
                    
                    for book_file in custom_books:
                        try:
                            with open(book_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                title = data.get("title", book_file.stem)
                                extract = data.get("extract", "")
                                
                                # Take top 1500 chars to avoid prompt limits
                                if len(extract) > 1500:
                                    extract = extract[:1500] + "... (Truncated)"
                                
                                knowledge += f"\n📖 **From Book: {title}**\n{extract}\n"
                        except Exception as e:
                            logger.warning(f"Failed to load book {book_file.name}: {e}")
        except Exception as e:
            logger.warning(f"Error loading custom books: {e}")
            
        return knowledge
