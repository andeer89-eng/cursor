import requests
import time
import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from config import (
    GMGN_API_KEY, GMGN_WALLET_ID, REQUEST_TIMEOUT, 
    MAX_RETRIES, RETRY_DELAY, RATE_LIMIT_REQUESTS, 
    RATE_LIMIT_WINDOW, LOG_LEVEL
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    """Data class for trade results."""
    success: bool
    transaction_hash: Optional[str] = None
    error_message: Optional[str] = None
    gas_used: Optional[int] = None
    gas_price: Optional[float] = None
    timestamp: Optional[datetime] = None

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self.lock = threading.Lock()
    
    def can_proceed(self) -> bool:
        """Check if a request can proceed."""
        now = time.time()
        
        with self.lock:
            # Remove old requests outside the window
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.window_seconds]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        while not self.can_proceed():
            time.sleep(0.1)  # Wait 100ms before checking again

class GMGNTrader:
    """Optimized GMGN trading client with connection pooling and rate limiting."""
    
    def __init__(self, api_key: str, wallet_id: str):
        self.api_key = api_key
        self.wallet_id = wallet_id
        self.base_url = "https://api.gmgn.com"  # Replace with actual API URL
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
        
        # Configure session for better performance
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "GMGN-Trading-Bot/1.0"
        })
        
        # Connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Cache for frequently accessed data
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        if current_time - self._last_cleanup > 60:  # Cleanup every minute
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self._cache_ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            self._last_cleanup = current_time
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make an API request with rate limiting and retry logic."""
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All request attempts failed for {endpoint}")
                    raise
    
    def get_wallet_balance(self, chain: str = "eth") -> Dict[str, Any]:
        """Get wallet balance with caching."""
        cache_key = f"balance_{chain}"
        self._cleanup_cache()
        
        if cache_key in self._cache:
            return self._cache[cache_key][0]
        
        try:
            result = self._make_request(
                "GET", 
                f"/wallet/{self.wallet_id}/balance",
                params={"chain": chain}
            )
            
            # Cache the result
            self._cache[cache_key] = (result, time.time())
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting wallet balance: {e}")
            return {"error": str(e)}
    
    def get_token_info(self, contract_address: str, chain: str = "eth") -> Dict[str, Any]:
        """Get token information with caching."""
        cache_key = f"token_info_{chain}_{contract_address}"
        self._cleanup_cache()
        
        if cache_key in self._cache:
            return self._cache[cache_key][0]
        
        try:
            result = self._make_request(
                "GET",
                f"/token/{chain}/{contract_address}"
            )
            
            # Cache the result
            self._cache[cache_key] = (result, time.time())
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting token info: {e}")
            return {"error": str(e)}
    
    def estimate_gas(self, contract_address: str, amount: float, 
                    chain: str = "eth") -> Dict[str, Any]:
        """Estimate gas for a transaction."""
        try:
            return self._make_request(
                "POST",
                "/estimate-gas",
                data={
                    "contract_address": contract_address,
                    "amount": amount,
                    "chain": chain,
                    "wallet_id": self.wallet_id
                }
            )
        except Exception as e:
            logger.error(f"Error estimating gas: {e}")
            return {"error": str(e)}
    
    def buy_token(self, contract_address: str, chain: str = "eth", 
                 amount: float = 0.01) -> TradeResult:
        """Buy a token with optimized error handling."""
        start_time = time.time()
        
        try:
            # Validate inputs
            if not contract_address or not chain or amount <= 0:
                return TradeResult(
                    success=False,
                    error_message="Invalid input parameters",
                    timestamp=datetime.utcnow()
                )
            
            # Check wallet balance first
            balance = self.get_wallet_balance(chain)
            if "error" in balance:
                return TradeResult(
                    success=False,
                    error_message=f"Failed to get balance: {balance['error']}",
                    timestamp=datetime.utcnow()
                )
            
            # Estimate gas
            gas_estimate = self.estimate_gas(contract_address, amount, chain)
            if "error" in gas_estimate:
                return TradeResult(
                    success=False,
                    error_message=f"Failed to estimate gas: {gas_estimate['error']}",
                    timestamp=datetime.utcnow()
                )
            
            # Execute trade
            trade_data = {
                "contract_address": contract_address,
                "amount": amount,
                "chain": chain,
                "wallet_id": self.wallet_id,
                "gas_limit": gas_estimate.get("gas_limit"),
                "gas_price": gas_estimate.get("gas_price")
            }
            
            result = self._make_request("POST", "/trade/buy", data=trade_data)
            
            if result.get("success"):
                return TradeResult(
                    success=True,
                    transaction_hash=result.get("tx_hash"),
                    gas_used=result.get("gas_used"),
                    gas_price=result.get("gas_price"),
                    timestamp=datetime.utcnow()
                )
            else:
                return TradeResult(
                    success=False,
                    error_message=result.get("error", "Unknown error"),
                    timestamp=datetime.utcnow()
                )
                
        except Exception as e:
            logger.error(f"Error buying token: {e}")
            return TradeResult(
                success=False,
                error_message=str(e),
                timestamp=datetime.utcnow()
            )
    
    def sell_token(self, contract_address: str, chain: str = "eth", 
                  amount: float = None) -> TradeResult:
        """Sell a token (amount=None for full balance)."""
        try:
            trade_data = {
                "contract_address": contract_address,
                "amount": amount,
                "chain": chain,
                "wallet_id": self.wallet_id
            }
            
            result = self._make_request("POST", "/trade/sell", data=trade_data)
            
            if result.get("success"):
                return TradeResult(
                    success=True,
                    transaction_hash=result.get("tx_hash"),
                    gas_used=result.get("gas_used"),
                    gas_price=result.get("gas_price"),
                    timestamp=datetime.utcnow()
                )
            else:
                return TradeResult(
                    success=False,
                    error_message=result.get("error", "Unknown error"),
                    timestamp=datetime.utcnow()
                )
                
        except Exception as e:
            logger.error(f"Error selling token: {e}")
            return TradeResult(
                success=False,
                error_message=str(e),
                timestamp=datetime.utcnow()
            )
    
    def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trade history."""
        try:
            result = self._make_request(
                "GET",
                f"/wallet/{self.wallet_id}/trades",
                params={"limit": limit}
            )
            
            return result.get("trades", [])
            
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
    
    def get_portfolio_value(self) -> Dict[str, Any]:
        """Get current portfolio value."""
        try:
            return self._make_request(
                "GET",
                f"/wallet/{self.wallet_id}/portfolio"
            )
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return {"error": str(e)}
    
    def batch_get_token_info(self, tokens: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Get information for multiple tokens in batch."""
        try:
            result = self._make_request(
                "POST",
                "/tokens/batch",
                data={"tokens": tokens}
            )
            
            return result.get("tokens", [])
            
        except Exception as e:
            logger.error(f"Error in batch token info: {e}")
            return []
    
    def close(self):
        """Close the session and cleanup resources."""
        if self.session:
            self.session.close()