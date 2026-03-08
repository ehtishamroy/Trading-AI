"""
MT5 Connector — Handles all communication with MetaTrader 5 (Exness).
Pulls data, places orders, checks positions.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MARKETS


# ─── MT5 Timeframe Mapping ───────────────────────────────
TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


def connect_mt5() -> bool:
    """
    Initialize and login to MT5.
    MT5 terminal must be installed and running on your PC.
    """
    if not mt5.initialize():
        logger.error(f"MT5 initialize failed: {mt5.last_error()}")
        return False

    if MT5_LOGIN and MT5_LOGIN != 0:
        authorized = mt5.login(
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            server=MT5_SERVER
        )
        if not authorized:
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            return False
        logger.success(f"Logged in to MT5 | Account: {MT5_LOGIN} | Server: {MT5_SERVER}")
    else:
        logger.info("MT5 connected without login (using default account)")

    account_info = mt5.account_info()
    if account_info:
        logger.info(f"Balance: ${account_info.balance} | Leverage: 1:{account_info.leverage}")
    return True


def disconnect_mt5():
    """Cleanly shut down MT5 connection."""
    mt5.shutdown()
    logger.info("MT5 connection closed")


def get_ohlcv(
    symbol: str,
    timeframe: str = "M15",
    num_bars: int = 5000,
    start_date: datetime = None
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from MT5.

    Args:
        symbol: MT5 symbol name (e.g., 'EURUSD')
        timeframe: 'M1','M5','M15','M30','H1','H4','D1'
        num_bars: Number of bars to fetch (max ~100,000)
        start_date: If set, fetch from this date instead of N bars back

    Returns:
        DataFrame with columns: time, open, high, low, close, volume
    """
    tf = TIMEFRAMES.get(timeframe)
    if tf is None:
        raise ValueError(f"Invalid timeframe: {timeframe}. Use: {list(TIMEFRAMES.keys())}")

    if start_date:
        rates = mt5.copy_rates_from(symbol, tf, start_date, num_bars)
    else:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, num_bars)

    if rates is None or len(rates) == 0:
        logger.error(f"No data for {symbol} {timeframe}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    # Keep only OHLCV columns
    df = df[["open", "high", "low", "close", "volume"]]

    logger.info(f"Fetched {len(df)} bars | {symbol} {timeframe} | {df.index[0]} → {df.index[-1]}")
    return df


def get_current_price(symbol: str) -> dict:
    """Get the latest bid/ask price for a symbol."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"Cannot get price for {symbol}")
        return {}
    return {
        "symbol": symbol,
        "bid": tick.bid,
        "ask": tick.ask,
        "spread": round(tick.ask - tick.bid, 5),
        "time": datetime.fromtimestamp(tick.time)
    }


def get_symbol_info(symbol: str) -> dict:
    """Get trading specs for a symbol (lot size, margins, etc.)."""
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(f"Symbol {symbol} not found in MT5")
        return {}
    return {
        "symbol": symbol,
        "point": info.point,
        "digits": info.digits,
        "trade_contract_size": info.trade_contract_size,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
        "spread": info.spread,
    }


def place_order(
    symbol: str,
    order_type: str,  # 'buy' or 'sell'
    lot_size: float = 0.01,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    comment: str = "ML_Trading_v2"
) -> dict:
    """
    Place a market order on MT5.

    Args:
        symbol: Trading symbol
        order_type: 'buy' or 'sell'
        lot_size: Position size (0.01 = micro lot)
        stop_loss: Stop loss price
        take_profit: Take profit price
        comment: Order comment for identification

    Returns:
        dict with order result info
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"success": False, "error": f"Cannot get price for {symbol}"}

    if order_type.lower() == "buy":
        trade_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    elif order_type.lower() == "sell":
        trade_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        return {"success": False, "error": f"Invalid order type: {order_type}"}

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": trade_type,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 20,  # Max slippage in points
        "magic": 234000,  # Unique ID for our bot's orders
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        return {"success": False, "error": str(mt5.last_error())}

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "error": f"Order failed: {result.retcode} - {result.comment}"}

    logger.success(
        f"{'BUY' if order_type == 'buy' else 'SELL'} {symbol} | "
        f"Lot: {lot_size} | Price: {price} | SL: {stop_loss} | TP: {take_profit}"
    )
    return {
        "success": True,
        "ticket": result.order,
        "price": price,
        "volume": lot_size,
        "symbol": symbol,
        "type": order_type,
    }


def close_position(ticket: int) -> dict:
    """Close an open position by ticket number."""
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return {"success": False, "error": f"Position {ticket} not found"}

    pos = positions[0]
    trade_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(pos.symbol)
    price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": trade_type,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "ML_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "error": f"Close failed: {result.comment}"}

    logger.success(f"Closed position {ticket} | PnL: {pos.profit}")
    return {"success": True, "pnl": pos.profit}


def get_open_positions() -> list:
    """Get all currently open positions."""
    positions = mt5.positions_get()
    if positions is None:
        return []
    return [
        {
            "ticket": p.ticket,
            "symbol": p.symbol,
            "type": "buy" if p.type == 0 else "sell",
            "volume": p.volume,
            "price_open": p.price_open,
            "sl": p.sl,
            "tp": p.tp,
            "profit": p.profit,
            "time": datetime.fromtimestamp(p.time),
        }
        for p in positions
    ]


def get_account_info() -> dict:
    """Get current account balance and info."""
    info = mt5.account_info()
    if info is None:
        return {}
    return {
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "free_margin": info.margin_free,
        "leverage": info.leverage,
        "profit": info.profit,
    }


if __name__ == "__main__":
    # Quick test: python data/mt5_connector.py
    if connect_mt5():
        print("\n--- Account Info ---")
        print(get_account_info())

        print("\n--- EUR/USD Latest Price ---")
        print(get_current_price("EURUSD"))

        print("\n--- Fetching 100 bars of EURUSD M15 ---")
        df = get_ohlcv("EURUSD", "M15", 100)
        print(df.tail())

        disconnect_mt5()
