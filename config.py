import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GMGN_API_KEY = os.getenv('GMGN_API_KEY', '')
GMGN_WALLET_ID = os.getenv('GMGN_WALLET_ID', '')

# Trading Parameters
TRADE_AMOUNT_ETH = float(os.getenv('TRADE_AMOUNT_ETH', '0.01'))
FETCH_INTERVAL_SECONDS = int(os.getenv('FETCH_INTERVAL_SECONDS', '60'))

# Filtering Criteria
MIN_MARKET_CAP = float(os.getenv('MIN_MARKET_CAP', '100000'))
MIN_VOLUME_1H = float(os.getenv('MIN_VOLUME_1H', '10000'))
MIN_PAIR_AGE_HOURS = int(os.getenv('MIN_PAIR_AGE_HOURS', '1'))
MIN_HOLDERS = int(os.getenv('MIN_HOLDERS', '100'))

# Performance Settings
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
CACHE_DURATION_SECONDS = int(os.getenv('CACHE_DURATION_SECONDS', '300'))

# Database Configuration
DATABASE_PATH = os.getenv('DATABASE_PATH', 'trading_bot.db')

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log')