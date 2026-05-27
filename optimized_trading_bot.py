import asyncio
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import psutil
import gc

from trend_analyzer import TrendAnalyzer
from gmgn_trader import GMGNTrader
from config import *
from db import init_db, batch_insert_tokens, record_metric
import requests

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric."""
        self.metrics[name] = value
        record_metric(name, value)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        memory_usage = self.get_memory_usage()
        self.record_metric("memory_usage_mb", memory_usage)
        
        if memory_usage > MAX_MEMORY_USAGE_MB:
            logger.warning(f"Memory usage {memory_usage:.2f}MB exceeds limit {MAX_MEMORY_USAGE_MB}MB")
            return False
        return True
    
    def cleanup_memory(self):
        """Force garbage collection to free memory."""
        gc.collect()
        logger.info("Memory cleanup performed")

class OptimizedTradingBot:
    """Optimized trading bot with async processing and performance monitoring."""
    
    def __init__(self):
        self.trader = GMGNTrader(api_key=GMGN_API_KEY, wallet_id=GMGN_WALLET_ID)
        self.monitor = PerformanceMonitor()
        self.session = requests.Session()
        self.executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
        
        # Configure session for better performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=3
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Cache for API responses
        self._cache = {}
        self._cache_ttl = CACHE_TTL_SECONDS
        self._last_cleanup = time.time()
        
        # Performance tracking
        self.processed_pairs = 0
        self.successful_trades = 0
        self.start_time = time.time()
    
    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        if current_time - self._last_cleanup > 60:
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self._cache_ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            self._last_cleanup = current_time
    
    def _fetch_dexscreener_pairs_optimized(self) -> List[Dict[str, Any]]:
        """Fetch Dexscreener pairs with caching and error handling."""
        cache_key = "dexscreener_pairs"
        self._cleanup_cache()
        
        if cache_key in self._cache:
            return self._cache[cache_key][0]
        
        try:
            start_time = time.time()
            response = self.session.get(
                "https://api.dexscreener.com/latest/dex/pairs",
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            pairs = response.json().get("pairs", [])
            
            # Cache the result
            self._cache[cache_key] = (pairs, time.time())
            
            fetch_time = time.time() - start_time
            self.monitor.record_metric("dexscreener_fetch_time", fetch_time)
            
            logger.info(f"Fetched {len(pairs)} pairs in {fetch_time:.2f}s")
            return pairs
            
        except Exception as e:
            logger.error(f"Error fetching Dexscreener data: {e}")
            return []
    
    def _filter_pairs_efficiently(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter pairs efficiently using vectorized operations."""
        if not pairs:
            return []
        
        # Convert to DataFrame for vectorized operations
        df = pd.DataFrame(pairs)
        
        # Extract required fields with default values
        df['market_cap'] = df.get('fdv', pd.Series([0] * len(df)))
        df['volume_1h'] = df.get('volume', {}).apply(lambda x: x.get('h1', 0) if isinstance(x, dict) else 0)
        df['holders'] = df.get('holders', 0)
        df['pair_created_at'] = df.get('pairCreatedAt', 0)
        
        # Convert timestamps to datetime
        df['created_at'] = pd.to_datetime(df['pair_created_at'], unit='ms', errors='coerce')
        now = datetime.utcnow()
        df['age_hours'] = (now - df['created_at']).dt.total_seconds() / 3600
        
        # Apply filters using vectorized operations
        mask = (
            (df['market_cap'] >= MIN_MARKET_CAP) &
            (df['volume_1h'] >= MIN_VOLUME_1H) &
            (df['age_hours'] >= MIN_PAIR_AGE_HOURS) &
            (df['holders'] >= MIN_HOLDERS) &
            (df['created_at'].notna())  # Valid creation date
        )
        
        filtered_pairs = df[mask].to_dict('records')
        
        logger.info(f"Filtered {len(filtered_pairs)} pairs from {len(pairs)} total")
        return filtered_pairs
    
    def _analyze_pair_trend(self, pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze trend for a single pair."""
        try:
            # Generate mock price series (replace with real data)
            prices = [1.00 + i * 0.01 for i in range(30)]
            times = [datetime.utcnow() - timedelta(minutes=i*5) for i in range(30)][::-1]
            price_series = pd.Series(prices, index=times)
            
            analyzer = TrendAnalyzer(price_series)
            trend = analyzer.detect_trend()
            
            if trend == "bullish":
                return {
                    'pair': pair,
                    'trend': trend,
                    'analysis': analyzer.get_analysis_summary()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing pair trend: {e}")
            return None
    
    def _process_pairs_batch(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process pairs in batches for better performance."""
        if not pairs:
            return []
        
        # Split pairs into batches
        batches = [pairs[i:i + BATCH_SIZE] for i in range(0, len(pairs), BATCH_SIZE)]
        results = []
        
        for batch in batches:
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=min(len(batch), MAX_CONCURRENT_REQUESTS)) as executor:
                future_to_pair = {
                    executor.submit(self._analyze_pair_trend, pair): pair 
                    for pair in batch
                }
                
                for future in as_completed(future_to_pair):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing pair: {e}")
        
        return results
    
    def _prepare_tokens_for_db(self, analyzed_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare analyzed pairs for database insertion."""
        tokens = []
        now = datetime.utcnow()
        
        for analyzed in analyzed_pairs:
            pair = analyzed['pair']
            
            contract = pair.get("baseToken", {}).get("address")
            symbol = pair.get("baseToken", {}).get("symbol")
            quote = pair.get("quoteToken", {}).get("symbol")
            price = pair.get("priceUsd")
            chain = pair.get("chainId", "eth")
            
            rug_url, bubble_url = self._generate_analysis_urls(chain, contract)
            
            token_data = {
                'pair_address': pair.get("pairAddress"),
                'base_symbol': symbol,
                'quote_symbol': quote,
                'price_usd': price,
                'market_cap': pair.get("fdv"),
                'volume_1h': pair.get("volume", {}).get("h1", 0),
                'tx_count_1h': pair.get("txCount", {}).get("h1", 0),
                'holders': pair.get("holders", 0),
                'pair_created_at': datetime.fromtimestamp(int(pair.get("pairCreatedAt", 0)) / 1000),
                'fetched_at': now,
                'rugcheck_url': rug_url,
                'bubblemaps_url': bubble_url,
                'trend': analyzed['trend']
            }
            
            tokens.append(token_data)
        
        return tokens
    
    def _generate_analysis_urls(self, chain: str, contract_address: str) -> tuple:
        """Generate analysis URLs for a token."""
        rugcheck_url = f"https://rugcheck.xyz/tokens/{chain}/{contract_address}"
        bubblemaps_url = f"https://app.bubblemaps.io/{chain}/token/{contract_address}"
        return rugcheck_url, bubblemaps_url
    
    def _execute_trades(self, analyzed_pairs: List[Dict[str, Any]]) -> int:
        """Execute trades for analyzed pairs."""
        successful_trades = 0
        
        for analyzed in analyzed_pairs:
            pair = analyzed['pair']
            contract = pair.get("baseToken", {}).get("address")
            symbol = pair.get("baseToken", {}).get("symbol")
            chain = pair.get("chainId", "eth")
            
            try:
                logger.info(f"Executing trade for {symbol} on {chain}")
                trade_result = self.trader.buy_token(contract, chain, TRADE_AMOUNT_ETH)
                
                if trade_result.success:
                    successful_trades += 1
                    logger.info(f"Successfully traded {symbol}: {trade_result.transaction_hash}")
                else:
                    logger.error(f"Trade failed for {symbol}: {trade_result.error_message}")
                    
            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {e}")
        
        return successful_trades
    
    def run_cycle(self) -> Dict[str, Any]:
        """Run a single trading cycle with performance monitoring."""
        cycle_start = time.time()
        
        try:
            # Check memory usage
            if not self.monitor.check_memory_limit():
                self.monitor.cleanup_memory()
            
            # Fetch pairs
            pairs = self._fetch_dexscreener_pairs_optimized()
            if not pairs:
                return {"error": "No pairs fetched"}
            
            # Filter pairs efficiently
            filtered_pairs = self._filter_pairs_efficiently(pairs)
            if not filtered_pairs:
                return {"message": "No pairs meet criteria"}
            
            # Process pairs in batches
            analyzed_pairs = self._process_pairs_batch(filtered_pairs)
            if not analyzed_pairs:
                return {"message": "No bullish trends detected"}
            
            # Prepare tokens for database
            tokens = self._prepare_tokens_for_db(analyzed_pairs)
            
            # Batch insert into database
            inserted_count = batch_insert_tokens(tokens)
            
            # Execute trades
            successful_trades = self._execute_trades(analyzed_pairs)
            
            # Update metrics
            self.processed_pairs += len(pairs)
            self.successful_trades += successful_trades
            
            cycle_time = time.time() - cycle_start
            self.monitor.record_metric("cycle_time", cycle_time)
            self.monitor.record_metric("pairs_processed", len(pairs))
            self.monitor.record_metric("trades_executed", successful_trades)
            
            return {
                "processed_pairs": len(pairs),
                "filtered_pairs": len(filtered_pairs),
                "analyzed_pairs": len(analyzed_pairs),
                "inserted_tokens": inserted_count,
                "successful_trades": successful_trades,
                "cycle_time": cycle_time
            }
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return {"error": str(e)}
    
    def run(self):
        """Main bot loop with performance monitoring."""
        logger.info("Starting Optimized GMGN Trading Bot...")
        
        # Initialize database
        init_db()
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                logger.info(f"Starting cycle {cycle_count}")
                
                # Run trading cycle
                result = self.run_cycle()
                
                if "error" in result:
                    logger.error(f"Cycle {cycle_count} failed: {result['error']}")
                else:
                    logger.info(f"Cycle {cycle_count} completed: {result}")
                
                # Record performance metrics
                self.monitor.record_metric("total_cycles", cycle_count)
                self.monitor.record_metric("total_pairs_processed", self.processed_pairs)
                self.monitor.record_metric("total_trades", self.successful_trades)
                
                # Memory cleanup every N cycles
                if cycle_count % GARBAGE_COLLECTION_INTERVAL == 0:
                    self.monitor.cleanup_memory()
                
                # Sleep before next cycle
                logger.info(f"Sleeping for {FETCH_INTERVAL_SECONDS} seconds...")
                time.sleep(FETCH_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Fatal error in main loop: {e}")
                time.sleep(FETCH_INTERVAL_SECONDS)
        
        # Cleanup
        self.trader.close()
        self.session.close()
        self.executor.shutdown()
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    # Validate configuration
    issues = validate_config()
    if issues:
        logger.error("Configuration issues found:")
        for key, message in issues.items():
            logger.error(f"  {key}: {message}")
        exit(1)
    
    # Start the bot
    bot = OptimizedTradingBot()
    bot.run()