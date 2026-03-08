"""Quick system test — checks all connections."""
import sys
sys.path.insert(0, '.')

print('=== Testing System Components ===')
print()

# Test 1: Config loads
try:
    from config.settings import (
        ACTIVE_MARKET, STARTING_CAPITAL, CLAUDE_API_KEY,
        NEWS_API_KEY, MT5_LOGIN, CLAUDE_MODEL
    )
    print('[OK] Config loaded')
    print(f'     Active market: {ACTIVE_MARKET}')
    print(f'     Starting capital: ${STARTING_CAPITAL}')
    if CLAUDE_API_KEY:
        print(f'     Claude API key: {CLAUDE_API_KEY[:12]}...')
    else:
        print('     [WARN] No Claude API key')
    if NEWS_API_KEY:
        print(f'     News API key: {NEWS_API_KEY[:8]}...')
    else:
        print('     [WARN] No News API key')
    print(f'     MT5 login: {MT5_LOGIN}')
except Exception as e:
    print(f'[FAIL] Config: {e}')

print()

# Test 2: MT5 connection
try:
    from data.mt5_connector import connect_mt5, disconnect_mt5, get_account_info, get_current_price
    if connect_mt5():
        info = get_account_info()
        print('[OK] MT5 Connected!')
        balance = info.get('balance', 'N/A')
        leverage = info.get('leverage', 'N/A')
        server = info.get('server', 'N/A')
        print(f'     Balance: ${balance}')
        print(f'     Leverage: 1:{leverage}')
        print(f'     Server: {server}')

        price = get_current_price('EURUSD')
        if price:
            print(f'     EUR/USD: {price.get("bid", "N/A")} / {price.get("ask", "N/A")}')

        price_gold = get_current_price('XAUUSD')
        if price_gold:
            print(f'     XAU/USD: {price_gold.get("bid", "N/A")} / {price_gold.get("ask", "N/A")}')

        disconnect_mt5()
    else:
        print('[FAIL] MT5 could not connect — is MT5 open and logged in?')
except Exception as e:
    print(f'[FAIL] MT5: {e}')

print()

# Test 3: Claude API
try:
    import anthropic
    if CLAUDE_API_KEY:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=50,
            messages=[{'role': 'user', 'content': 'Say exactly: Trading system online'}]
        )
        print(f'[OK] Claude API: {response.content[0].text}')
    else:
        print('[SKIP] Claude API — no key configured')
except Exception as e:
    print(f'[FAIL] Claude API: {e}')

print()

# Test 4: PyTorch
try:
    import torch
    cuda = 'CUDA available' if torch.cuda.is_available() else 'CPU only'
    print(f'[OK] PyTorch {torch.__version__} ({cuda})')
except Exception as e:
    print(f'[FAIL] PyTorch: {e}')

# Test 5: XGBoost
try:
    import xgboost
    print(f'[OK] XGBoost {xgboost.__version__}')
except Exception as e:
    print(f'[FAIL] XGBoost: {e}')

# Test 6: Streamlit
try:
    import streamlit
    print(f'[OK] Streamlit {streamlit.__version__}')
except Exception as e:
    print(f'[FAIL] Streamlit: {e}')

print()
print('=== Test Complete ===')
