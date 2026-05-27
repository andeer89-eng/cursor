# Configuration file for GMGN Memecoin Bot
import os
from typing import Dict, Any

# API Configuration
GMGN_API_KEY = os.getenv("GMGN_API_KEY", "your_api_key_here")
GMGN_WALLET_ID = os.getenv("GMGN_WALLET_ID", "your_wallet_id_here")

# Trading Parameters
TRADE_AMOUNT_ETH = float(os.getenv("TRADE_AMOUNT_ETH", "0.01"))

# Screening Criteria
MIN_MARKET_CAP = float(os.getenv("MIN_MARKET_CAP", "100000"))  # $100k minimum
MIN_VOLUME_1H = float(os.getenv("MIN_VOLUME_1H", "50000"))     # $50k minimum 1h volume
MIN_PAIR_AGE_HOURS = float(os.getenv("MIN_PAIR_AGE_HOURS", "1"))  # 1 hour minimum
MIN_HOLDERS = int(os.getenv("MIN_HOLDERS", "100"))             # 100 minimum holders

# Performance Settings
FETCH_INTERVAL_SECONDS = int(os.getenv("FETCH_INTERVAL_SECONDS", "60"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))  # Process pairs in batches
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Caching Configuration
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///trading_bot.db")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# API Rate Limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# Performance Monitoring
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))

# Error Handling
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# Memory Management
MAX_MEMORY_USAGE_MB = int(os.getenv("MAX_MEMORY_USAGE_MB", "512"))
GARBAGE_COLLECTION_INTERVAL = int(os.getenv("GARBAGE_COLLECTION_INTERVAL", "100"))

# Configuration validation
def validate_config() -> Dict[str, Any]:
    """Validate configuration and return any issues."""
    issues = {}
    
    if not GMGN_API_KEY or GMGN_API_KEY == "your_api_key_here":
        issues["GMGN_API_KEY"] = "API key not configured"
    
    if TRADE_AMOUNT_ETH <= 0:
        issues["TRADE_AMOUNT_ETH"] = "Trade amount must be positive"
    
    if MIN_MARKET_CAP <= 0:
        issues["MIN_MARKET_CAP"] = "Minimum market cap must be positive"
    
    return issues