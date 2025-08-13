import time
import requests
from datetime import datetime
from trend_analyzer import TrendAnalyzer
from gmgn_trader import GMGNTrader
from config import *
from db import init_db, insert_token
import pandas as pd

def generate_analysis_urls(chain, contract_address):
    rugcheck_url = f"https://rugcheck.xyz/tokens/{chain}/{contract_address}"
    bubblemaps_url = f"https://app.bubblemaps.io/{chain}/token/{contract_address}"
    return rugcheck_url, bubblemaps_url

def fetch_dexscreener_pairs():
    try:
        url = "https://api.dexscreener.com/latest/dex/pairs"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("pairs", [])
    except Exception as e:
        print("Error fetching Dexscreener data:", e)
        return []

def get_recent_price_series(pair_address):
    prices = [1.00 + i * 0.01 for i in range(30)]
    times = [datetime.utcnow() - pd.Timedelta(minutes=i*5) for i in range(30)][::-1]
    return pd.Series(prices, index=times)

def process_pairs():
    pairs = fetch_dexscreener_pairs()
    trader = GMGNTrader(api_key=GMGN_API_KEY, wallet_id=GMGN_WALLET_ID)
    now = datetime.utcnow()
    saved = 0

    for pair in pairs:
        market_cap = pair.get("fdv")
        volume_1h = pair.get("volume", {}).get("h1", 0)
        holders = pair.get("holders", 0)
        created_at_ms = pair.get("pairCreatedAt")
        if not created_at_ms:
            continue

        created_at = datetime.fromtimestamp(int(created_at_ms) / 1000)
        age_hours = (now - created_at).total_seconds() / 3600

        if not (
            market_cap and market_cap >= MIN_MARKET_CAP and
            volume_1h and volume_1h >= MIN_VOLUME_1H and
            age_hours >= MIN_PAIR_AGE_HOURS and
            holders and holders >= MIN_HOLDERS
        ):
            continue

        price_series = get_recent_price_series(pair.get("pairAddress"))
        analyzer = TrendAnalyzer(price_series)
        trend = analyzer.detect_trend()

        if trend != "bullish":
            continue

        contract = pair.get("baseToken", {}).get("address")
        symbol = pair.get("baseToken", {}).get("symbol")
        quote = pair.get("quoteToken", {}).get("symbol")
        price = pair.get("priceUsd")
        chain = pair.get("chainId", "eth")

        rug_url, bubble_url = generate_analysis_urls(chain, contract)

        insert_token(
            pair_address=pair.get("pairAddress"),
            base_symbol=symbol,
            quote_symbol=quote,
            price_usd=price,
            market_cap=market_cap,
            volume_1h=volume_1h,
            tx_count_1h=pair.get("txCount", {}).get("h1", 0),
            holders=holders,
            pair_created_at=created_at,
            fetched_at=now,
            rugcheck_url=rug_url,
            bubblemaps_url=bubble_url,
            trend=trend
        )
        saved += 1

        print(f"BUYING {symbol} on {chain} - {price} USD (Trend: {trend})")
        trade = trader.buy_token(contract, chain, TRADE_AMOUNT_ETH)
        print("Trade response:", trade)

    print(f"[{now.isoformat()}] Processed and saved {saved} tokens.")

if __name__ == "__main__":
    print("Initializing GMGN Memecoin Bot...")
    init_db()
    while True:
        try:
            process_pairs()
        except Exception as e:
            print("Fatal error:", e)
        print(f"Sleeping for {FETCH_INTERVAL_SECONDS} seconds...\n")
        time.sleep(FETCH_INTERVAL_SECONDS)
