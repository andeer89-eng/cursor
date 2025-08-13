import asyncio
import aiohttp
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json

from trend_analyzer import TrendAnalyzer
from gmgn_trader import GMGNTrader
from config import *
from db import init_db, insert_token, get_cached_data, set_cached_data, cleanup_expired_cache

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedTradingBot:
    """Optimized trading bot with async operations, caching, and performance improvements."""
    
    def __init__(self):
        self.session = None
        self.trader = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache = {}
        self.last_cleanup = datetime.utcnow()
        
    async def __aenter__(self):
        """Initialize async resources."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        connector = aiohttp.TCPConnector(
            limit=MAX_CONCURRENT_REQUESTS,
            limit_per_host=MAX_CONCURRENT_REQUESTS,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        self.trader = GMGNTrader(GMGN_API_KEY, GMGN_WALLET_ID)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async resources."""
        if self.session:
            await self.session.close()
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def generate_analysis_urls(self, chain: str, contract_address: str) -> tuple:
        """Generate analysis URLs efficiently."""
        rugcheck_url = f"https://rugcheck.xyz/tokens/{chain}/{contract_address}"
        bubblemaps_url = f"https://app.bubblemaps.io/{chain}/token/{contract_address}"
        return rugcheck_url, bubblemaps_url
    
    async def fetch_dexscreener_pairs(self) -> List[Dict[str, Any]]:
        """Fetch Dexscreener pairs with caching and error handling."""
        cache_key = "dexscreener_pairs"
        cached_data = get_cached_data(cache_key)
        
        if cached_data:
            try:
                return json.loads(cached_data)
            except json.JSONDecodeError:
                pass
        
        try:
            url = "https://api.dexscreener.com/latest/dex/pairs"
            async with self.session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                pairs = data.get("pairs", [])
                
                # Cache the response
                set_cached_data(cache_key, json.dumps(pairs), CACHE_DURATION_SECONDS)
                logger.info(f"Fetched {len(pairs)} pairs from Dexscreener")
                return pairs
                
        except Exception as e:
            logger.error(f"Error fetching Dexscreener data: {e}")
            return []
    
    def get_recent_price_series(self, pair_address: str) -> pd.Series:
        """Get recent price series with optimized calculation."""
        # In a real implementation, this would fetch actual price data
        # For now, using simulated data with better performance
        prices = [1.00 + i * 0.01 for i in range(30)]
        times = [datetime.utcnow() - timedelta(minutes=i*5) for i in range(30)][::-1]
        return pd.Series(prices, index=times)
    
    def filter_pair(self, pair: Dict[str, Any], now: datetime) -> bool:
        """Efficiently filter pairs based on criteria."""
        try:
            market_cap = pair.get("fdv", 0)
            volume_1h = pair.get("volume", {}).get("h1", 0)
            holders = pair.get("holders", 0)
            created_at_ms = pair.get("pairCreatedAt")
            
            if not created_at_ms:
                return False
            
            created_at = datetime.fromtimestamp(int(created_at_ms) / 1000)
            age_hours = (now - created_at).total_seconds() / 3600
            
            # Early return for better performance
            if market_cap < MIN_MARKET_CAP:
                return False
            if volume_1h < MIN_VOLUME_1H:
                return False
            if age_hours < MIN_PAIR_AGE_HOURS:
                return False
            if holders < MIN_HOLDERS:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error filtering pair: {e}")
            return False
    
    async def process_pair(self, pair: Dict[str, Any], now: datetime) -> Optional[Dict[str, Any]]:
        """Process a single pair asynchronously."""
        try:
            if not self.filter_pair(pair, now):
                return None
            
            # Get price series and analyze trend
            price_series = self.get_recent_price_series(pair.get("pairAddress"))
            analyzer = TrendAnalyzer(price_series)
            trend = analyzer.detect_trend()
            
            if trend != "bullish":
                return None
            
            # Extract pair data
            contract = pair.get("baseToken", {}).get("address")
            symbol = pair.get("baseToken", {}).get("symbol")
            quote = pair.get("quoteToken", {}).get("symbol")
            price = pair.get("priceUsd")
            chain = pair.get("chainId", "eth")
            market_cap = pair.get("fdv")
            volume_1h = pair.get("volume", {}).get("h1", 0)
            holders = pair.get("holders", 0)
            created_at_ms = pair.get("pairCreatedAt")
            created_at = datetime.fromtimestamp(int(created_at_ms) / 1000)
            
            rug_url, bubble_url = self.generate_analysis_urls(chain, contract)
            
            # Prepare token data
            token_data = {
                "pair_address": pair.get("pairAddress"),
                "base_symbol": symbol,
                "quote_symbol": quote,
                "price_usd": price,
                "market_cap": market_cap,
                "volume_1h": volume_1h,
                "tx_count_1h": pair.get("txCount", {}).get("h1", 0),
                "holders": holders,
                "pair_created_at": created_at,
                "fetched_at": now,
                "rugcheck_url": rug_url,
                "bubblemaps_url": bubble_url,
                "trend": trend
            }
            
            return token_data
            
        except Exception as e:
            logger.error(f"Error processing pair: {e}")
            return None
    
    async def process_pairs_async(self) -> int:
        """Process pairs asynchronously with improved performance."""
        try:
            pairs = await self.fetch_dexscreener_pairs()
            if not pairs:
                logger.warning("No pairs fetched from Dexscreener")
                return 0
            
            now = datetime.utcnow()
            saved = 0
            
            # Process pairs in batches for better performance
            batch_size = 50
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [self.process_pair(pair, now) for pair in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed: {result}")
                        continue
                    
                    if result:
                        # Insert into database
                        insert_token(**result)
                        saved += 1
                        
                        # Execute trade
                        contract = result["pair_address"]
                        symbol = result["base_symbol"]
                        chain = result.get("chain", "eth")
                        price = result["price_usd"]
                        trend = result["trend"]
                        
                        logger.info(f"BUYING {symbol} on {chain} - {price} USD (Trend: {trend})")
                        
                        # Use synchronous wrapper for trading
                        trade_result = self.trader.buy_token_sync(contract, chain, TRADE_AMOUNT_ETH)
                        logger.info(f"Trade response: {trade_result}")
            
            logger.info(f"[{now.isoformat()}] Processed and saved {saved} tokens.")
            return saved
            
        except Exception as e:
            logger.error(f"Error in process_pairs_async: {e}")
            return 0
    
    async def cleanup_cache(self):
        """Periodically cleanup expired cache entries."""
        now = datetime.utcnow()
        if (now - self.last_cleanup).total_seconds() > 3600:  # Cleanup every hour
            cleanup_expired_cache()
            self.last_cleanup = now
    
    async def run(self):
        """Main bot loop with optimized performance."""
        logger.info("Initializing Optimized GMGN Memecoin Bot...")
        init_db()
        
        while True:
            try:
                start_time = time.time()
                
                # Process pairs asynchronously
                saved_count = await self.process_pairs_async()
                
                # Cleanup cache periodically
                await self.cleanup_cache()
                
                # Performance metrics
                elapsed_time = time.time() - start_time
                logger.info(f"Processing completed in {elapsed_time:.2f} seconds. Saved {saved_count} tokens.")
                
                # Sleep with adaptive timing
                sleep_time = max(1, FETCH_INTERVAL_SECONDS - elapsed_time)
                logger.info(f"Sleeping for {sleep_time:.2f} seconds...\n")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Fatal error in main loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

async def main():
    """Main async entry point."""
    async with OptimizedTradingBot() as bot:
        await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")