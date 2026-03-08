"""
Setup Guide — Step-by-step beginner instructions.
Run: python setup_guide.py

This interactive script walks you through:
1. Installing MT5
2. Creating an Exness demo account
3. Setting up API keys
4. Downloading historical data
5. Training your first model
6. Starting the trading system
"""

import os
import sys
from pathlib import Path

def print_banner():
    print()
    print("=" * 60)
    print("  🧠 ML TRADING SYSTEM v2 — SETUP GUIDE")
    print("=" * 60)
    print()

def step_1():
    print("─" * 50)
    print("STEP 1: Install MetaTrader 5")
    print("─" * 50)
    print()
    print("1. Go to: https://www.exness.com/")
    print("2. Sign up for a FREE account")
    print("3. In the dashboard, create a DEMO account:")
    print("   - Platform: MetaTrader 5")
    print("   - Account type: Standard")
    print("   - Currency: USD")
    print("   - Leverage: 1:100")
    print()
    print("4. Download MT5 from Exness and install it")
    print("5. Login to MT5 with your demo credentials")
    print()
    print("📝 NOTE DOWN:")
    print("   - MT5 Login Number (e.g., 12345678)")
    print("   - MT5 Password")
    print("   - Server Name (e.g., 'Exness-MT5Trial')")
    print()
    input("Press Enter when MT5 is installed and running... ")

def step_2():
    print()
    print("─" * 50)
    print("STEP 2: Get Claude AI API Key")
    print("─" * 50)
    print()
    print("1. Go to: https://console.anthropic.com/")
    print("2. Sign up / Log in")
    print("3. Go to API Keys")
    print("4. Create a new key")
    print("5. Copy the key (starts with 'sk-ant-...')")
    print()
    print("💡 Cost: ~$3-5/month for our usage")
    print()
    input("Press Enter when you have your Claude API key... ")

def step_3():
    print()
    print("─" * 50)
    print("STEP 3: Get News API Key (Free)")
    print("─" * 50)
    print()
    print("1. Go to: https://newsapi.org/")
    print("2. Sign up for FREE plan")
    print("3. Copy your API key")
    print()
    input("Press Enter when you have your News API key... ")

def step_4():
    print()
    print("─" * 50)
    print("STEP 4: Configure .env File")
    print("─" * 50)
    print()

    env_path = Path(__file__).parent / ".env"
    template_path = Path(__file__).parent / ".env.template"

    if not env_path.exists():
        if template_path.exists():
            import shutil
            shutil.copy(template_path, env_path)
            print("✅ Created .env from template")
        else:
            print("❌ No .env.template found!")
            return

    print("Now let's fill in your credentials:")
    print()

    mt5_login = input("MT5 Login Number: ").strip()
    mt5_pass = input("MT5 Password: ").strip()
    mt5_server = input("MT5 Server (press Enter for 'Exness-MT5Trial'): ").strip() or "Exness-MT5Trial"
    claude_key = input("Claude API Key: ").strip()
    news_key = input("News API Key (press Enter to skip): ").strip()

    # Write to .env
    with open(env_path, "w") as f:
        f.write(f"# ML Trading System v2 Configuration\n")
        f.write(f"TRADING_MODE=demo\n")
        f.write(f"ACTIVE_MARKET=EURUSD\n\n")
        f.write(f"# MT5 (Exness)\n")
        f.write(f"MT5_LOGIN={mt5_login}\n")
        f.write(f"MT5_PASSWORD={mt5_pass}\n")
        f.write(f"MT5_SERVER={mt5_server}\n\n")
        f.write(f"# Claude AI\n")
        f.write(f"CLAUDE_API_KEY={claude_key}\n\n")
        f.write(f"# News API\n")
        f.write(f"NEWS_API_KEY={news_key}\n\n")
        f.write(f"# Telegram (set up later)\n")
        f.write(f"TELEGRAM_BOT_TOKEN=\n")
        f.write(f"TELEGRAM_CHAT_ID=\n")

    print()
    print("✅ .env file saved!")

def step_5():
    print()
    print("─" * 50)
    print("STEP 5: Install Python Packages")
    print("─" * 50)
    print()
    print("Running: pip install -r requirements.txt")
    print("This may take 5-10 minutes on first install...")
    print()

    result = os.system(f"{sys.executable} -m pip install -r requirements.txt")
    if result == 0:
        print("\n✅ All packages installed!")
    else:
        print("\n⚠️ Some packages may have failed.")
        print("Try running manually: pip install -r requirements.txt")

def step_6():
    print()
    print("─" * 50)
    print("STEP 6: Test MT5 Connection")
    print("─" * 50)
    print()
    print("Testing connection to MetaTrader 5...")
    print("Make sure MT5 is OPEN and logged in!")
    print()

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from data.mt5_connector import connect_mt5, disconnect_mt5, get_account_info, get_current_price

        if connect_mt5():
            info = get_account_info()
            print(f"✅ Connected! Balance: ${info.get('balance', 'N/A')}")
            print(f"   Leverage: 1:{info.get('leverage', 'N/A')}")

            price = get_current_price("EURUSD")
            if price:
                print(f"   EUR/USD: {price.get('bid', 'N/A')} / {price.get('ask', 'N/A')}")

            disconnect_mt5()
        else:
            print("❌ Cannot connect to MT5!")
            print("   Make sure MT5 is open and logged in")
    except ImportError:
        print("❌ MetaTrader5 package not installed")
        print("   Run: pip install MetaTrader5")

def step_7():
    print()
    print("─" * 50)
    print("STEP 7: Download Historical Data")
    print("─" * 50)
    print()
    print("Downloading 50,000 bars of data for each market...")
    print()

    try:
        from data.fetcher import fetch_all_data
        fetch_all_data()
        print("\n✅ Historical data downloaded!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def step_8():
    print()
    print("─" * 50)
    print("STEP 8: Train Your First Model")
    print("─" * 50)
    print()
    print("Training ML models for EUR/USD...")
    print("This takes 5-15 minutes on a GTX 1070")
    print()

    try:
        os.system(f"{sys.executable} train.py --market EURUSD")
        print("\n✅ Models trained!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def step_done():
    print()
    print("=" * 60)
    print("  🎉 SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("You can now:")
    print()
    print("  📊 Start trading loop:")
    print("     python main.py")
    print()
    print("  🖥️  Open dashboard:")
    print("     streamlit run dashboard/app.py")
    print()
    print("  📱 Telegram bot (after setting up bot token):")
    print("     Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env")
    print()
    print("  🔄 Weekly retrain:")
    print("     python retrain.py")
    print()
    print("⚠️ IMPORTANT: The system starts in DEMO mode.")
    print("   Trade on demo for AT LEAST 30 days before going live!")
    print()


if __name__ == "__main__":
    print_banner()

    steps = [
        ("Install MetaTrader 5", step_1),
        ("Get Claude API Key", step_2),
        ("Get News API Key", step_3),
        ("Configure .env", step_4),
        ("Install Packages", step_5),
        ("Test MT5 Connection", step_6),
        ("Download Data", step_7),
        ("Train Models", step_8),
    ]

    for i, (name, func) in enumerate(steps):
        print(f"\n{'='*50}")
        print(f"  [{i+1}/{len(steps)}] {name}")
        print(f"{'='*50}")
        func()

    step_done()
