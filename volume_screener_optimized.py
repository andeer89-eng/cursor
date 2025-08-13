import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import json
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedVolumeScreener:
    """Optimized volume screener with async operations and caching."""
    
    def __init__(self):
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.cache = {}
        
        # Parameters
        self.VOLUME_THRESHOLD = 1000000
        self.UNUSUAL_MULTIPLIER = 2.0
        self.LOOKBACK_DAYS = 20
        
        # S&P 500 tickers (expanded list)
        self.sp500_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
            'PG', 'KO', 'PEP', 'CSCO', 'INTC', 'AMD', 'QCOM', 'ORCL', 'IBM', 'DIS',
            'NFLX', 'CRM', 'ADBE', 'PYPL', 'ABT', 'TMO', 'ACN', 'LLY', 'DHR', 'NEE',
            'UNH', 'HD', 'MA', 'PFE', 'ABBV', 'BAC', 'TXN', 'COST', 'AVGO', 'WFC',
            'MRK', 'CVX', 'XOM', 'CMCSA', 'PEP', 'ADP', 'T', 'BMY', 'RTX', 'LOW',
            'UPS', 'SPGI', 'MS', 'SCHW', 'BA', 'CAT', 'GE', 'F', 'GM', 'DE'
        ]
    
    async def __aenter__(self):
        """Initialize async resources."""
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async resources."""
        if self.session:
            await self.session.close()
        if self.executor:
            self.executor.shutdown(wait=True)
    
    @lru_cache(maxsize=1000)
    def _get_cached_stock_data(self, ticker: str, days: int) -> Optional[pd.DataFrame]:
        """Get cached stock data with LRU cache."""
        cache_key = f"{ticker}_{days}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        return None
    
    def _set_cached_stock_data(self, ticker: str, days: int, data: pd.DataFrame):
        """Cache stock data."""
        cache_key = f"{ticker}_{days}"
        self.cache[cache_key] = data
    
    async def fetch_stock_data_async(self, ticker: str, days: int = None) -> Optional[pd.DataFrame]:
        """Fetch stock data asynchronously with caching."""
        if days is None:
            days = self.LOOKBACK_DAYS + 1
        
        # Check cache first
        cached_data = self._get_cached_stock_data(ticker, days)
        if cached_data is not None:
            return cached_data
        
        try:
            # Use ThreadPoolExecutor for yfinance calls (yfinance is not async)
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                self.executor,
                self._fetch_stock_data_sync,
                ticker,
                days
            )
            
            if df is not None and not df.empty:
                self._set_cached_stock_data(ticker, days, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def _fetch_stock_data_sync(self, ticker: str, days: int) -> Optional[pd.DataFrame]:
        """Synchronous stock data fetching."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            return df
        except Exception as e:
            logger.error(f"Error in sync fetch for {ticker}: {e}")
            return None
    
    def screen_stock_efficient(self, df: pd.DataFrame, ticker: str) -> Optional[Dict[str, Any]]:
        """Screen a single stock with optimized calculations."""
        try:
            if df.empty or len(df) < self.LOOKBACK_DAYS:
                return None
            
            # Use numpy for faster calculations
            volumes = df['Volume'].values
            prices = df['Close'].values
            
            # Current volume (latest day)
            current_volume = volumes[-1]
            
            # Average volume over lookback period (excluding today)
            avg_volume = np.mean(volumes[:-1])
            
            # Check conditions with early returns
            if current_volume < self.VOLUME_THRESHOLD:
                return None
            
            if current_volume < avg_volume * self.UNUSUAL_MULTIPLIER:
                return None
            
            # Calculate additional metrics
            volume_ratio = current_volume / avg_volume
            current_price = prices[-1]
            price_change_pct = ((prices[-1] - prices[-2]) / prices[-2]) * 100 if len(prices) > 1 else 0
            
            # Calculate volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized volatility
            
            return {
                'Ticker': ticker,
                'Current Volume': int(current_volume),
                'Avg Volume (20d)': int(avg_volume),
                'Volume Ratio': round(volume_ratio, 2),
                'Price': round(current_price, 2),
                'Price Change (%)': round(price_change_pct, 2),
                'Volatility (%)': round(volatility, 2),
                'Market Cap': self._estimate_market_cap(current_price, ticker)
            }
            
        except Exception as e:
            logger.error(f"Error screening {ticker}: {e}")
            return None
    
    def _estimate_market_cap(self, price: float, ticker: str) -> str:
        """Estimate market cap based on price (simplified)."""
        # This is a simplified estimation - in production, fetch actual market cap
        if price > 1000:
            return "Large Cap"
        elif price > 100:
            return "Mid Cap"
        else:
            return "Small Cap"
    
    async def screen_stock_async(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Screen a single stock asynchronously."""
        try:
            df = await self.fetch_stock_data_async(ticker)
            if df is None:
                return None
            
            return self.screen_stock_efficient(df, ticker)
            
        except Exception as e:
            logger.error(f"Error in async screening for {ticker}: {e}")
            return None
    
    async def run_screener_async(self, tickers: List[str] = None) -> pd.DataFrame:
        """Run the screener asynchronously across all tickers."""
        if tickers is None:
            tickers = self.sp500_tickers
        
        logger.info(f"Scanning {len(tickers)} stocks for volume spike opportunities...")
        
        # Process tickers in batches for better performance
        batch_size = 10
        all_results = []
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.screen_stock_async(ticker) for ticker in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed: {result}")
                    continue
                
                if result:
                    all_results.append(result)
        
        # Convert to DataFrame and sort by Volume Ratio
        if all_results:
            df_results = pd.DataFrame(all_results)
            df_results = df_results.sort_values(by='Volume Ratio', ascending=False)
            return df_results
        else:
            return pd.DataFrame()
    
    async def run_continuous_screener(self, interval_minutes: int = 15):
        """Run continuous screening with specified interval."""
        logger.info(f"Starting continuous volume screener (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                start_time = datetime.now()
                
                # Run screener
                results = await self.run_screener_async()
                
                # Display results
                if not results.empty:
                    print(f"\n=== Stocks with Volume Spikes and Unusual Activity ===")
                    print(f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(results.to_string(index=False))
                    print(f"\nFound {len(results)} opportunities.")
                    
                    # Save to CSV with timestamp
                    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
                    filename = f'volume_spike_opportunities_{timestamp}.csv'
                    results.to_csv(filename, index=False)
                    print(f"Results saved to '{filename}'")
                else:
                    print(f"No stocks found with significant volume spikes at {start_time.strftime('%H:%M:%S')}")
                
                # Calculate next run time
                elapsed_time = (datetime.now() - start_time).total_seconds()
                sleep_time = max(1, (interval_minutes * 60) - elapsed_time)
                
                logger.info(f"Next scan in {sleep_time/60:.1f} minutes...")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in continuous screener: {e}")
                await asyncio.sleep(60)  # Wait before retrying

async def main():
    """Main async entry point."""
    async with OptimizedVolumeScreener() as screener:
        # Run once
        results = await screener.run_screener_async()
        
        if not results.empty:
            print("\n=== Stocks with Volume Spikes and Unusual Activity ===")
            print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(results.to_string(index=False))
            print(f"\nFound {len(results)} opportunities.")
            
            # Save to CSV
            results.to_csv('volume_spike_opportunities.csv', index=False)
            print("Results saved to 'volume_spike_opportunities.csv'")
        else:
            print("No stocks found with significant volume spikes today.")

async def run_continuous():
    """Run continuous screening."""
    async with OptimizedVolumeScreener() as screener:
        await screener.run_continuous_screener(interval_minutes=15)

if __name__ == "__main__":
    try:
        # Install required library if not already installed
        try:
            import yfinance
        except ImportError:
            print("Installing yfinance...")
            import os
            os.system("pip install yfinance")
        
        # Run the screener
        asyncio.run(main())
        
        # Uncomment the line below to run continuous screening
        # asyncio.run(run_continuous())
        
    except KeyboardInterrupt:
        print("\nScreener stopped by user")
    except Exception as e:
        logger.error(f"Screener crashed: {e}")