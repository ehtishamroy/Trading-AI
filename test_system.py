"""Quick system test — checks all connections."""
import sys
sys.path.insert(0, '.')

print('=== Testing System Components ===')
print()

# Test 1: Config loads
try:
    from config.settings import (
        ACTIVE_MARKET, STARTING_CAPITAL,
        NEWS_API_KEY, MT5_LOGIN, OLLAMA_MODEL, OLLAMA_BASE_URL
    )
    print('[OK] Config loaded')
    print(f'     Active market: {ACTIVE_MARKET}')
    print(f'     Starting capital: ${STARTING_CAPITAL}')
    print(f'     Ollama URL: {OLLAMA_BASE_URL}')
    print(f'     Ollama Model: {OLLAMA_MODEL}')
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

# Test 3: Ollama Local LLM
try:
    import requests
    resp = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=3)
    if resp.status_code == 200:
        models = [m['name'] for m in resp.json().get('models', [])]
        base = OLLAMA_MODEL.split(':')[0]
        matched = [m for m in models if base in m]
        if matched:
            print(f'[OK] Ollama running — model "{matched[0]}" ready (FREE local LLM 🎉)')
        else:
            print(f'[WARN] Ollama running but model not pulled yet.')
            print(f'       Run: ollama pull {OLLAMA_MODEL}')
            print(f'       Available models: {models}')
    else:
        print(f'[FAIL] Ollama returned status {resp.status_code}')
except Exception as e:
    print(f'[FAIL] Ollama not running — install from https://ollama.com then run: ollama pull {OLLAMA_MODEL}')
    print(f'       Error: {e}')

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
