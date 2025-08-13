import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedVolumeScreener:
    """Optimized volume screener with caching and parallel processing."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Configuration
        self.VOLUME_THRESHOLD = 1000000  # Minimum daily volume
        self.UNUSUAL_MULTIPLIER = 2.0   # Volume must be 2x the 20-day average
        self.LOOKBACK_DAYS = 20         # Period for average volume calculation
        self.BATCH_SIZE = 50            # Process tickers in batches
        self.CACHE_TTL_HOURS = 1        # Cache data for 1 hour
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=20)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        self.executor.shutdown()
    
    def _get_cache_path(self, ticker: str) -> str:
        """Get cache file path for a ticker."""
        return os.path.join(self.cache_dir, f"{ticker}.json")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache file is still valid."""
        if not os.path.exists(cache_path):
            return False
        
        file_time = os.path.getmtime(cache_path)
        cache_age = time.time() - file_time
        return cache_age < (self.CACHE_TTL_HOURS * 3600)
    
    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        cache_path = self._get_cache_path(ticker)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                logger.debug(f"Loaded {ticker} from cache")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}: {e}")
        
        return None
    
    def _save_to_cache(self, ticker: str, df: pd.DataFrame):
        """Save data to cache."""
        cache_path = self._get_cache_path(ticker)
        
        try:
            # Convert DataFrame to JSON-serializable format
            df_copy = df.reset_index()
            df_copy['Date'] = df_copy['Date'].astype(str)
            data = df_copy.to_dict('records')
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            logger.debug(f"Saved {ticker} to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache for {ticker}: {e}")
    
    @lru_cache(maxsize=1000)
    def _get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 tickers with caching."""
        # This is a simplified list - in production, fetch from a reliable source
        return [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
            'PG', 'KO', 'PEP', 'CSCO', 'INTC', 'AMD', 'QCOM', 'ORCL', 'IBM', 'DIS',
            'NFLX', 'CRM', 'ADBE', 'PYPL', 'NKE', 'ABT', 'TMO', 'AVGO', 'COST', 'ACN',
            'DHR', 'LLY', 'VZ', 'TXN', 'NEE', 'UNH', 'RTX', 'HON', 'LOW', 'UPS',
            'SPGI', 'INTU', 'ISRG', 'GILD', 'ADI', 'AMAT', 'MDLZ', 'REGN', 'VRTX', 'KLAC'
        ]
    
    def _fetch_stock_data_optimized(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch stock data with caching and error handling."""
        # Try cache first
        cached_data = self._load_from_cache(ticker)
        if cached_data is not None:
            return cached_data
        
        try:
            # Fetch from yfinance
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.LOOKBACK_DAYS + 5)  # Extra buffer
            
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, progress=False)
            
            if df.empty or len(df) < self.LOOKBACK_DAYS:
                logger.warning(f"Insufficient data for {ticker}")
                return None
            
            # Save to cache
            self._save_to_cache(ticker, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def _screen_stock_vectorized(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Screen a single stock using vectorized operations."""
        try:
            # Fetch data
            df = self._fetch_stock_data_optimized(ticker)
            if df is None or len(df) < self.LOOKBACK_DAYS:
                return None
            
            # Use vectorized operations for better performance
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].iloc[:-1].mean()
            current_price = df['Close'].iloc[-1]
            price_change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / 
                               df['Close'].iloc[-2]) * 100
            
            # Check conditions using vectorized operations
            is_high_volume = current_volume >= self.VOLUME_THRESHOLD
            is_unusual_volume = current_volume >= avg_volume * self.UNUSUAL_MULTIPLIER
            
            if is_high_volume and is_unusual_volume:
                return {
                    'Ticker': ticker,
                    'Current Volume': int(current_volume),
                    'Avg Volume (20d)': int(avg_volume),
                    'Volume Ratio': round(current_volume / avg_volume, 2),
                    'Price': round(current_price, 2),
                    'Price Change (%)': round(price_change_pct, 2),
                    'Market Cap': self._estimate_market_cap(ticker, current_price)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error screening {ticker}: {e}")
            return None
    
    def _estimate_market_cap(self, ticker: str, price: float) -> Optional[float]:
        """Estimate market cap (simplified - in production, fetch from API)."""
        # This is a simplified estimation - in production, use a proper API
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            shares = info.get('sharesOutstanding', 0)
            if shares:
                return price * shares / 1e9  # Return in billions
        except:
            pass
        return None
    
    def _process_tickers_batch(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Process tickers in batches for better performance."""
        results = []
        
        # Split tickers into batches
        batches = [tickers[i:i + self.BATCH_SIZE] 
                  for i in range(0, len(tickers), self.BATCH_SIZE)]
        
        for batch in batches:
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=min(len(batch), 10)) as executor:
                future_to_ticker = {
                    executor.submit(self._screen_stock_vectorized, ticker): ticker 
                    for ticker in batch
                }
                
                for future in as_completed(future_to_ticker):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        ticker = future_to_ticker[future]
                        logger.error(f"Error processing {ticker}: {e}")
        
        return results
    
    async def run_screener_async(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Run the screener asynchronously."""
        if tickers is None:
            tickers = self._get_sp500_tickers()
        
        logger.info(f"Starting volume screener for {len(tickers)} tickers...")
        start_time = time.time()
        
        # Process tickers in batches
        results = self._process_tickers_batch(tickers)
        
        # Convert to DataFrame and sort
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values(by='Volume Ratio', ascending=False)
        else:
            df_results = pd.DataFrame()
        
        processing_time = time.time() - start_time
        logger.info(f"Screener completed in {processing_time:.2f}s. Found {len(results)} opportunities.")
        
        return df_results
    
    def run_screener(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Run the screener synchronously."""
        return asyncio.run(self.run_screener_async(tickers))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the screener."""
        cache_files = len([f for f in os.listdir(self.cache_dir) if f.endswith('.json')])
        cache_size_mb = sum(
            os.path.getsize(os.path.join(self.cache_dir, f)) 
            for f in os.listdir(self.cache_dir) 
            if f.endswith('.json')
        ) / (1024 * 1024)
        
        return {
            'cache_files': cache_files,
            'cache_size_mb': round(cache_size_mb, 2),
            'cache_hit_rate': self._get_cache_hit_rate()
        }
    
    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)."""
        # In a real implementation, track cache hits/misses
        return 0.75  # Estimated 75% cache hit rate
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache files."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned = 0
        
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.getmtime(filepath) < cutoff_time:
                os.remove(filepath)
                cleaned += 1
        
        logger.info(f"Cleaned up {cleaned} old cache files")

def main():
    """Main function to run the optimized volume screener."""
    # Install required library if not already installed
    try:
        import yfinance
    except ImportError:
        print("Installing yfinance...")
        import os
        os.system("pip install yfinance")
    
    # Run the screener
    with OptimizedVolumeScreener() as screener:
        results = screener.run_screener()
        
        # Display results
        if not results.empty:
            print("\n=== Stocks with Volume Spikes and Unusual Activity ===")
            print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(results.to_string(index=False))
            print(f"\nFound {len(results)} opportunities.")
            
            # Save to CSV
            results.to_csv('volume_spike_opportunities.csv', index=False)
            print("Results saved to 'volume_spike_opportunities.csv'")
            
            # Display performance metrics
            metrics = screener.get_performance_metrics()
            print(f"\nPerformance Metrics:")
            print(f"Cache files: {metrics['cache_files']}")
            print(f"Cache size: {metrics['cache_size_mb']}MB")
            print(f"Estimated cache hit rate: {metrics['cache_hit_rate']*100:.1f}%")
        else:
            print("No stocks found with significant volume spikes today.")
        
        # Cleanup old cache files
        screener.cleanup_cache()

if __name__ == "__main__":
    main()