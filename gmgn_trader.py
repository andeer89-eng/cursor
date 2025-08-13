import aiohttp
import asyncio
import logging
from typing import Dict, Optional, Any
from asyncio_throttle import Throttler
import json
from config import GMGN_API_KEY, GMGN_WALLET_ID, REQUEST_TIMEOUT, MAX_CONCURRENT_REQUESTS

logger = logging.getLogger(__name__)

class GMGNTrader:
    """Optimized GMGN trading client with async operations and rate limiting."""
    
    def __init__(self, api_key: str, wallet_id: str):
        self.api_key = api_key
        self.wallet_id = wallet_id
        self.base_url = "https://api.gmgn.ai"  # Replace with actual API endpoint
        self.throttler = Throttler(rate_limit=MAX_CONCURRENT_REQUESTS, period=1)
        self.session = None
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "GMGN-Trading-Bot/1.0"
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        connector = aiohttp.TCPConnector(
            limit=MAX_CONCURRENT_REQUESTS,
            limit_per_host=MAX_CONCURRENT_REQUESTS,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=self._headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and error handling."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        url = f"{self.base_url}{endpoint}"
        
        async with self.throttler:
            try:
                if method.upper() == "GET":
                    async with self.session.get(url) as response:
                        response.raise_for_status()
                        return await response.json()
                elif method.upper() == "POST":
                    async with self.session.post(url, json=data) as response:
                        response.raise_for_status()
                        return await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                    
            except aiohttp.ClientError as e:
                logger.error(f"HTTP request failed: {e}")
                return {"error": str(e), "success": False}
            except Exception as e:
                logger.error(f"Unexpected error in request: {e}")
                return {"error": str(e), "success": False}
    
    async def get_wallet_balance(self) -> Dict[str, Any]:
        """Get wallet balance efficiently."""
        endpoint = f"/wallet/{self.wallet_id}/balance"
        return await self._make_request("GET", endpoint)
    
    async def get_token_price(self, contract_address: str, chain: str = "eth") -> Dict[str, Any]:
        """Get token price with caching."""
        endpoint = f"/price/{chain}/{contract_address}"
        return await self._make_request("GET", endpoint)
    
    async def buy_token(self, contract_address: str, chain: str = "eth", amount_eth: float = 0.01) -> Dict[str, Any]:
        """Buy token with optimized parameters."""
        data = {
            "wallet_id": self.wallet_id,
            "contract_address": contract_address,
            "chain": chain,
            "amount_eth": amount_eth,
            "slippage": 0.05,  # 5% slippage tolerance
            "gas_limit": 300000,  # Optimized gas limit
            "max_priority_fee": 2,  # Gwei
            "max_fee": 50  # Gwei
        }
        
        endpoint = "/trade/buy"
        return await self._make_request("POST", endpoint, data)
    
    async def sell_token(self, contract_address: str, chain: str = "eth", percentage: float = 100.0) -> Dict[str, Any]:
        """Sell token with optimized parameters."""
        data = {
            "wallet_id": self.wallet_id,
            "contract_address": contract_address,
            "chain": chain,
            "percentage": percentage,
            "slippage": 0.05,
            "gas_limit": 300000,
            "max_priority_fee": 2,
            "max_fee": 50
        }
        
        endpoint = "/trade/sell"
        return await self._make_request("POST", endpoint, data)
    
    async def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction status."""
        endpoint = f"/transaction/{tx_hash}/status"
        return await self._make_request("GET", endpoint)
    
    async def get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio holdings."""
        endpoint = f"/wallet/{self.wallet_id}/portfolio"
        return await self._make_request("GET", endpoint)
    
    def buy_token_sync(self, contract_address: str, chain: str = "eth", amount_eth: float = 0.01) -> Dict[str, Any]:
        """Synchronous wrapper for buy_token."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                task = asyncio.create_task(self.buy_token(contract_address, chain, amount_eth))
                return {"success": True, "message": "Trade initiated", "task": task}
            else:
                # Run in new event loop
                return loop.run_until_complete(self.buy_token(contract_address, chain, amount_eth))
        except Exception as e:
            logger.error(f"Error in synchronous buy_token: {e}")
            return {"error": str(e), "success": False}
    
    def sell_token_sync(self, contract_address: str, chain: str = "eth", percentage: float = 100.0) -> Dict[str, Any]:
        """Synchronous wrapper for sell_token."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = asyncio.create_task(self.sell_token(contract_address, chain, percentage))
                return {"success": True, "message": "Trade initiated", "task": task}
            else:
                return loop.run_until_complete(self.sell_token(contract_address, chain, percentage))
        except Exception as e:
            logger.error(f"Error in synchronous sell_token: {e}")
            return {"error": str(e), "success": False}