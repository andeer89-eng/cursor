import asyncio
import time
import psutil
import statistics
import json
from datetime import datetime
from typing import Dict, List, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import pandas as pd

# Import our optimized modules
from hh_optimized import OptimizedTradingBot
from volume_screener_optimized import OptimizedVolumeScreener
from performance_monitor import PerformanceMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTester:
    """Comprehensive performance testing suite for the trading bot."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.initial_memory = None
        self.final_memory = None
        
    def start_test(self):
        """Start performance test and record initial state."""
        self.start_time = time.time()
        self.initial_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        logger.info(f"Performance test started at {datetime.now()}")
        logger.info(f"Initial memory usage: {self.initial_memory:.2f} MB")
    
    def end_test(self):
        """End performance test and record final state."""
        self.end_time = time.time()
        self.final_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        logger.info(f"Performance test ended at {datetime.now()}")
        logger.info(f"Final memory usage: {self.final_memory:.2f} MB")
    
    async def test_http_performance(self, url: str, num_requests: int = 100) -> Dict[str, Any]:
        """Test HTTP request performance."""
        logger.info(f"Testing HTTP performance with {num_requests} requests to {url}")
        
        async with aiohttp.ClientSession() as session:
            response_times = []
            errors = 0
            
            start_time = time.time()
            
            for i in range(num_requests):
                try:
                    request_start = time.time()
                    async with session.get(url) as response:
                        await response.text()
                        response_time = time.time() - request_start
                        response_times.append(response_time)
                        
                        if i % 10 == 0:
                            logger.info(f"Completed {i+1}/{num_requests} requests")
                            
                except Exception as e:
                    errors += 1
                    logger.error(f"Request {i+1} failed: {e}")
            
            total_time = time.time() - start_time
            
            return {
                'total_requests': num_requests,
                'successful_requests': num_requests - errors,
                'failed_requests': errors,
                'total_time': total_time,
                'avg_response_time': statistics.mean(response_times) if response_times else 0,
                'min_response_time': min(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0,
                'requests_per_second': (num_requests - errors) / total_time,
                'error_rate': errors / num_requests
            }
    
    async def test_database_performance(self, num_operations: int = 1000) -> Dict[str, Any]:
        """Test database operation performance."""
        logger.info(f"Testing database performance with {num_operations} operations")
        
        from db import init_db, insert_token, get_cached_data, set_cached_data
        
        # Initialize database
        init_db()
        
        operation_times = []
        errors = 0
        
        start_time = time.time()
        
        for i in range(num_operations):
            try:
                op_start = time.time()
                
                # Test insert operation
                insert_token(
                    pair_address=f"test_pair_{i}",
                    base_symbol=f"TEST{i}",
                    quote_symbol="ETH",
                    price_usd=1.0 + (i * 0.01),
                    market_cap=100000 + (i * 1000),
                    volume_1h=10000 + (i * 100),
                    tx_count_1h=i,
                    holders=100 + i,
                    pair_created_at=datetime.utcnow(),
                    fetched_at=datetime.utcnow(),
                    rugcheck_url=f"https://test{i}.com",
                    bubblemaps_url=f"https://bubble{i}.com",
                    trend="bullish"
                )
                
                # Test cache operations
                cache_key = f"test_cache_{i}"
                set_cached_data(cache_key, f"test_data_{i}", 300)
                get_cached_data(cache_key)
                
                operation_time = time.time() - op_start
                operation_times.append(operation_time)
                
                if i % 100 == 0:
                    logger.info(f"Completed {i+1}/{num_operations} database operations")
                    
            except Exception as e:
                errors += 1
                logger.error(f"Database operation {i+1} failed: {e}")
        
        total_time = time.time() - start_time
        
        return {
            'total_operations': num_operations,
            'successful_operations': num_operations - errors,
            'failed_operations': errors,
            'total_time': total_time,
            'avg_operation_time': statistics.mean(operation_times) if operation_times else 0,
            'min_operation_time': min(operation_times) if operation_times else 0,
            'max_operation_time': max(operation_times) if operation_times else 0,
            'operations_per_second': (num_operations - errors) / total_time,
            'error_rate': errors / num_operations
        }
    
    async def test_trend_analyzer_performance(self, num_analyses: int = 1000) -> Dict[str, Any]:
        """Test trend analyzer performance."""
        logger.info(f"Testing trend analyzer performance with {num_analyses} analyses")
        
        from trend_analyzer import TrendAnalyzer
        import pandas as pd
        import numpy as np
        
        analysis_times = []
        errors = 0
        
        start_time = time.time()
        
        for i in range(num_analyses):
            try:
                analysis_start = time.time()
                
                # Generate test price series
                prices = np.random.randn(50).cumsum() + 100
                times = pd.date_range(start='2024-01-01', periods=50, freq='5min')
                price_series = pd.Series(prices, index=times)
                
                # Test trend analysis
                analyzer = TrendAnalyzer(price_series)
                trend = analyzer.detect_trend()
                strength = analyzer.get_trend_strength()
                
                analysis_time = time.time() - analysis_start
                analysis_times.append(analysis_time)
                
                if i % 100 == 0:
                    logger.info(f"Completed {i+1}/{num_analyses} trend analyses")
                    
            except Exception as e:
                errors += 1
                logger.error(f"Trend analysis {i+1} failed: {e}")
        
        total_time = time.time() - start_time
        
        return {
            'total_analyses': num_analyses,
            'successful_analyses': num_analyses - errors,
            'failed_analyses': errors,
            'total_time': total_time,
            'avg_analysis_time': statistics.mean(analysis_times) if analysis_times else 0,
            'min_analysis_time': min(analysis_times) if analysis_times else 0,
            'max_analysis_time': max(analysis_times) if analysis_times else 0,
            'analyses_per_second': (num_analyses - errors) / total_time,
            'error_rate': errors / num_analyses
        }
    
    async def test_volume_screener_performance(self, num_tickers: int = 50) -> Dict[str, Any]:
        """Test volume screener performance."""
        logger.info(f"Testing volume screener performance with {num_tickers} tickers")
        
        # Use a subset of tickers for testing
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] * (num_tickers // 5)
        
        start_time = time.time()
        
        try:
            async with OptimizedVolumeScreener() as screener:
                results = await screener.run_screener_async(test_tickers)
                
                total_time = time.time() - start_time
                
                return {
                    'total_tickers': len(test_tickers),
                    'successful_screens': len(results) if not results.empty else 0,
                    'total_time': total_time,
                    'screens_per_second': len(test_tickers) / total_time,
                    'results_count': len(results) if not results.empty else 0
                }
                
        except Exception as e:
            logger.error(f"Volume screener test failed: {e}")
            return {
                'total_tickers': len(test_tickers),
                'successful_screens': 0,
                'total_time': time.time() - start_time,
                'screens_per_second': 0,
                'results_count': 0,
                'error': str(e)
            }
    
    async def test_memory_usage(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Test memory usage over time."""
        logger.info(f"Testing memory usage for {duration_seconds} seconds")
        
        memory_samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            memory = psutil.virtual_memory()
            memory_samples.append({
                'timestamp': time.time() - start_time,
                'memory_mb': memory.used / (1024 * 1024),
                'memory_percent': memory.percent
            })
            await asyncio.sleep(1)
        
        memory_values = [sample['memory_mb'] for sample in memory_samples]
        memory_percentages = [sample['memory_percent'] for sample in memory_samples]
        
        return {
            'duration_seconds': duration_seconds,
            'samples_count': len(memory_samples),
            'avg_memory_mb': statistics.mean(memory_values),
            'max_memory_mb': max(memory_values),
            'min_memory_mb': min(memory_values),
            'avg_memory_percent': statistics.mean(memory_percentages),
            'max_memory_percent': max(memory_percentages),
            'memory_samples': memory_samples
        }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite."""
        logger.info("Starting comprehensive performance test suite")
        
        self.start_test()
        
        # Run all tests
        tests = {
            'http_performance': await self.test_http_performance('https://httpbin.org/get', 50),
            'database_performance': await self.test_database_performance(500),
            'trend_analyzer_performance': await self.test_trend_analyzer_performance(500),
            'volume_screener_performance': await self.test_volume_screener_performance(25),
            'memory_usage': await self.test_memory_usage(30)
        }
        
        self.end_test()
        
        # Calculate overall metrics
        total_test_time = self.end_time - self.start_time
        memory_increase = self.final_memory - self.initial_memory
        
        overall_results = {
            'test_summary': {
                'total_test_time': total_test_time,
                'initial_memory_mb': self.initial_memory,
                'final_memory_mb': self.final_memory,
                'memory_increase_mb': memory_increase,
                'memory_increase_percent': (memory_increase / self.initial_memory) * 100 if self.initial_memory > 0 else 0
            },
            'test_results': tests,
            'performance_score': self._calculate_performance_score(tests)
        }
        
        self.results = overall_results
        return overall_results
    
    def _calculate_performance_score(self, tests: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)."""
        score = 100.0
        
        # Deduct points for poor performance
        if tests.get('http_performance', {}).get('error_rate', 0) > 0.05:
            score -= 20
        
        if tests.get('database_performance', {}).get('error_rate', 0) > 0.05:
            score -= 20
        
        if tests.get('trend_analyzer_performance', {}).get('error_rate', 0) > 0.05:
            score -= 20
        
        avg_response_time = tests.get('http_performance', {}).get('avg_response_time', 0)
        if avg_response_time > 2.0:
            score -= 15
        
        avg_db_time = tests.get('database_performance', {}).get('avg_operation_time', 0)
        if avg_db_time > 0.1:
            score -= 15
        
        memory_increase = self.final_memory - self.initial_memory
        if memory_increase > 100:  # More than 100MB increase
            score -= 10
        
        return max(0, score)
    
    def save_results(self, filename: str = None) -> str:
        """Save test results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'performance_test_results_{timestamp}.json'
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Test results saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    
    def print_summary(self):
        """Print performance test summary."""
        if not self.results:
            logger.warning("No test results available")
            return
        
        summary = self.results['test_summary']
        score = self.results['performance_score']
        
        print("\n" + "="*60)
        print("PERFORMANCE TEST SUMMARY")
        print("="*60)
        print(f"Overall Performance Score: {score:.1f}/100")
        print(f"Total Test Time: {summary['total_test_time']:.2f} seconds")
        print(f"Memory Usage: {summary['initial_memory_mb']:.1f} MB → {summary['final_memory_mb']:.1f} MB")
        print(f"Memory Increase: {summary['memory_increase_mb']:.1f} MB ({summary['memory_increase_percent']:.1f}%)")
        
        print("\n" + "-"*60)
        print("DETAILED RESULTS")
        print("-"*60)
        
        for test_name, test_results in self.results['test_results'].items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            for key, value in test_results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

async def main():
    """Main function for performance testing."""
    tester = PerformanceTester()
    
    try:
        # Run comprehensive test
        results = await tester.run_comprehensive_test()
        
        # Print summary
        tester.print_summary()
        
        # Save results
        filename = tester.save_results()
        if filename:
            print(f"\nDetailed results saved to: {filename}")
        
        # Performance recommendations
        score = results['performance_score']
        if score >= 90:
            print("\n🎉 Excellent performance! The system is well-optimized.")
        elif score >= 70:
            print("\n✅ Good performance with room for improvement.")
        elif score >= 50:
            print("\n⚠️  Moderate performance issues detected.")
        else:
            print("\n❌ Significant performance issues detected.")
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())