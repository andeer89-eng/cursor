import time
import asyncio
import aiohttp
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
from functools import lru_cache
import logging

# Configure logging for better monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock imports for missing dependencies with optimized alternatives
class TrendAnalyzer:
    """Optimized trend analyzer with caching."""
    
    def __init__(self, price_series):
        self.price_series = price_series
    
    @lru_cache(maxsize=1000)
    def detect_trend(self) -> str:
        """Optimized trend detection with caching."""
        try:
            if len(self.price_series) < 5:
                return "neutral"
            
            # Simple trend detection using vectorized operations
            prices = self.price_series.values
            recent_trend = (prices[-1] - prices[-5]) / prices[-5]
            
            if recent_trend > 0.05:  # 5% increase
                return "bullish"
            elif recent_trend < -0.05:  # 5% decrease
                return "bearish"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Trend detection error: {e}")
            return "neutral"

class GMGNTrader:
    """Optimized trader with connection pooling and async operations."""
    
    def __init__(self, api_key: str, wallet_id: str):
        self.api_key = api_key
        self.wallet_id = wallet_id
        self.session = None
    
    async def __aenter__(self):
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def buy_token(self, contract: str, chain: str, amount: float) -> Dict:
        """Async token buying with better error handling."""
        try:
            if not self.session:
                return {"error": "Session not initialized"}
            
            # Mock API call - replace with actual implementation
            await asyncio.sleep(0.1)  # Simulate API delay
            return {
                "success": True,
                "contract": contract,
                "chain": chain,
                "amount": amount,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Trade error for {contract}: {e}")
            return {"error": str(e), "contract": contract}

# Mock configuration with optimized defaults
GMGN_API_KEY = "mock_api_key"
GMGN_WALLET_ID = "mock_wallet_id"
MIN_MARKET_CAP = 100000
MIN_VOLUME_1H = 10000
MIN_PAIR_AGE_HOURS = 1
MIN_HOLDERS = 50
TRADE_AMOUNT_ETH = 0.01
FETCH_INTERVAL_SECONDS = 60
MAX_CONCURRENT_REQUESTS = 20
CACHE_TTL = 300  # 5 minutes cache

# Mock database operations with batching
class Database:
    """Optimized database operations with batching."""
    
    def __init__(self):
        self.batch_data = []
        self.batch_size = 50
    
    def insert_token(self, **kwargs):
        """Add token to batch for bulk insert."""
        self.batch_data.append(kwargs)
        
        if len(self.batch_data) >= self.batch_size:
            self.flush_batch()
    
    def flush_batch(self):
        """Flush batch data to database."""
        if not self.batch_data:
            return
        
        logger.info(f"Batch inserting {len(self.batch_data)} tokens to database")
        # Mock database insert - replace with actual implementation
        self.batch_data.clear()

def init_db():
    """Initialize database connection."""
    logger.info("Database initialized")
    return Database()

# Global instances
db = init_db()

# Caching for API responses
_api_cache = {}
_cache_timestamps = {}

def is_cache_valid(key: str) -> bool:
    """Check if cached data is still valid."""
    if key not in _cache_timestamps:
        return False
    return time.time() - _cache_timestamps[key] < CACHE_TTL

@lru_cache(maxsize=10)
def generate_analysis_urls(chain: str, contract_address: str) -> Tuple[str, str]:
    """Cached URL generation."""
    rugcheck_url = f"https://rugcheck.xyz/tokens/{chain}/{contract_address}"
    bubblemaps_url = f"https://app.bubblemaps.io/{chain}/token/{contract_address}"
    return rugcheck_url, bubblemaps_url

async def fetch_dexscreener_pairs_async(session: aiohttp.ClientSession) -> List[Dict]:
    """
    Async API call to DexScreener with error handling and caching.
    """
    cache_key = "dexscreener_pairs"
    
    # Check cache first
    if is_cache_valid(cache_key) and cache_key in _api_cache:
        logger.info("Using cached DexScreener data")
        return _api_cache[cache_key]
    
    try:
        url = "https://api.dexscreener.com/latest/dex/pairs"
        
        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.json()
            pairs = data.get("pairs", [])
            
            # Cache the result
            _api_cache[cache_key] = pairs
            _cache_timestamps[cache_key] = time.time()
            
            logger.info(f"Fetched {len(pairs)} pairs from DexScreener")
            return pairs
            
    except aiohttp.ClientError as e:
        logger.error(f"HTTP error fetching DexScreener data: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching DexScreener data: {e}")
        return []

@lru_cache(maxsize=1000)
def get_recent_price_series_cached(pair_address: str) -> pd.Series:
    """Cached price series generation to avoid redundant calculations."""
    prices = [1.00 + i * 0.01 for i in range(30)]
    times = [datetime.utcnow() - pd.Timedelta(minutes=i*5) for i in range(30)][::-1]
    return pd.Series(prices, index=times)

def filter_pairs_vectorized(pairs: List[Dict], now: datetime) -> List[Dict]:
    """
    Optimized pair filtering using vectorized operations where possible.
    """
    if not pairs:
        return []
    
    # Convert to DataFrame for vectorized operations
    df = pd.DataFrame(pairs)
    
    # Early filtering with vectorized operations
    if 'fdv' in df.columns:
        df = df[df['fdv'].fillna(0) >= MIN_MARKET_CAP]
    
    if df.empty:
        return []
    
    # Filter by volume
    df['volume_1h'] = df['volume'].apply(lambda x: x.get('h1', 0) if isinstance(x, dict) else 0)
    df = df[df['volume_1h'] >= MIN_VOLUME_1H]
    
    if df.empty:
        return []
    
    # Filter by holders
    df = df[df['holders'].fillna(0) >= MIN_HOLDERS]
    
    if df.empty:
        return []
    
    # Filter by age (this requires individual processing)
    valid_pairs = []
    for _, row in df.iterrows():
        created_at_ms = row.get('pairCreatedAt')
        if not created_at_ms:
            continue
        
        try:
            created_at = datetime.fromtimestamp(int(created_at_ms) / 1000)
            age_hours = (now - created_at).total_seconds() / 3600
            
            if age_hours >= MIN_PAIR_AGE_HOURS:
                valid_pairs.append(row.to_dict())
        except (ValueError, TypeError):
            continue
    
    return valid_pairs

async def process_single_pair(pair: Dict, trader: GMGNTrader, now: datetime) -> Optional[Dict]:
    """
    Process a single pair with trend analysis and trading.
    """
    try:
        # Get trend analysis
        pair_address = pair.get("pairAddress", "")
        price_series = get_recent_price_series_cached(pair_address)
        analyzer = TrendAnalyzer(price_series)
        trend = analyzer.detect_trend()

        if trend != "bullish":
            return None

        # Extract pair data
        contract = pair.get("baseToken", {}).get("address")
        symbol = pair.get("baseToken", {}).get("symbol", "UNKNOWN")
        quote = pair.get("quoteToken", {}).get("symbol", "UNKNOWN")
        price = pair.get("priceUsd")
        chain = pair.get("chainId", "eth")
        market_cap = pair.get("fdv")
        volume_1h = pair.get("volume", {}).get("h1", 0)
        holders = pair.get("holders", 0)

        if not contract:
            return None

        # Generate analysis URLs
        rug_url, bubble_url = generate_analysis_urls(chain, contract)

        # Prepare database entry
        token_data = {
            'pair_address': pair_address,
            'base_symbol': symbol,
            'quote_symbol': quote,
            'price_usd': price,
            'market_cap': market_cap,
            'volume_1h': volume_1h,
            'tx_count_1h': pair.get("txCount", {}).get("h1", 0),
            'holders': holders,
            'pair_created_at': datetime.fromtimestamp(int(pair.get('pairCreatedAt', 0)) / 1000),
            'fetched_at': now,
            'rugcheck_url': rug_url,
            'bubblemaps_url': bubble_url,
            'trend': trend
        }

        # Execute trade
        logger.info(f"BUYING {symbol} on {chain} - {price} USD (Trend: {trend})")
        trade_result = await trader.buy_token(contract, chain, TRADE_AMOUNT_ETH)
        
        token_data['trade_result'] = trade_result
        return token_data

    except Exception as e:
        logger.error(f"Error processing pair {pair.get('pairAddress', 'unknown')}: {e}")
        return None

async def process_pairs_concurrent():
    """
    Optimized pair processing with async operations and concurrent execution.
    """
    start_time = time.time()
    now = datetime.utcnow()
    
    # Create aiohttp session with connection pooling
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=30,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        
        # Fetch pairs asynchronously
        pairs = await fetch_dexscreener_pairs_async(session)
        
        if not pairs:
            logger.warning("No pairs fetched from DexScreener")
            return
        
        # Filter pairs using vectorized operations
        logger.info(f"Filtering {len(pairs)} pairs...")
        filtered_pairs = filter_pairs_vectorized(pairs, now)
        logger.info(f"Found {len(filtered_pairs)} pairs matching criteria")
        
        if not filtered_pairs:
            logger.info("No pairs match the filtering criteria")
            return
        
        # Process pairs concurrently with rate limiting
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async def process_with_semaphore(pair):
            async with semaphore:
                async with GMGNTrader(GMGN_API_KEY, GMGN_WALLET_ID) as trader:
                    return await process_single_pair(pair, trader, now)
        
        # Execute concurrent processing
        logger.info(f"Processing {len(filtered_pairs)} pairs concurrently...")
        tasks = [process_with_semaphore(pair) for pair in filtered_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and save to database
        saved_count = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue
            
            if result:
                db.insert_token(**result)
                saved_count += 1
        
        # Flush any remaining batch data
        db.flush_batch()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processed and saved {saved_count} tokens in {elapsed_time:.2f} seconds")

def clear_caches():
    """Clear all caches to free memory."""
    global _api_cache, _cache_timestamps
    _api_cache.clear()
    _cache_timestamps.clear()
    
    # Clear LRU caches
    generate_analysis_urls.cache_clear()
    get_recent_price_series_cached.cache_clear()
    
    logger.info("Caches cleared")

async def main_loop():
    """
    Main async loop with better error handling and performance monitoring.
    """
    logger.info("Initializing Optimized GMGN Memecoin Bot...")
    logger.info(f"Configuration: MIN_MARKET_CAP={MIN_MARKET_CAP}, MIN_VOLUME_1H={MIN_VOLUME_1H}")
    logger.info(f"Performance: MAX_CONCURRENT_REQUESTS={MAX_CONCURRENT_REQUESTS}, CACHE_TTL={CACHE_TTL}s")
    
    iteration = 0
    while True:
        iteration += 1
        start_time = time.time()
        
        try:
            logger.info(f"Starting iteration {iteration}")
            await process_pairs_concurrent()
            
            # Memory management
            if iteration % 10 == 0:  # Clear caches every 10 iterations
                clear_caches()
                logger.info(f"Completed {iteration} iterations")
            
        except Exception as e:
            logger.error(f"Fatal error in iteration {iteration}: {e}")
        
        # Calculate sleep time accounting for execution time
        elapsed_time = time.time() - start_time
        sleep_time = max(0, FETCH_INTERVAL_SECONDS - elapsed_time)
        
        logger.info(f"Iteration {iteration} completed in {elapsed_time:.2f}s. Sleeping for {sleep_time:.2f}s")
        await asyncio.sleep(sleep_time)

if __name__ == "__main__":
    try:
        # Check for required packages
        import aiohttp
        logger.info("All required packages available")
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        print("Please install required packages: pip install aiohttp pandas")
        exit(1)
    
    # Run the optimized main loop
    asyncio.run(main_loop())
