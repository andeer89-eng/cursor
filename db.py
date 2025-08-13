import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from config import DATABASE_PATH
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._connection_pool = {}
        
    @contextmanager
    def get_connection(self):
        """Thread-safe database connection with connection pooling."""
        thread_id = threading.get_ident()
        
        with self._lock:
            if thread_id not in self._connection_pool:
                self._connection_pool[thread_id] = sqlite3.connect(
                    self.db_path, 
                    check_same_thread=False,
                    timeout=30.0
                )
                # Enable WAL mode for better concurrency
                self._connection_pool[thread_id].execute("PRAGMA journal_mode=WAL")
                self._connection_pool[thread_id].execute("PRAGMA synchronous=NORMAL")
                self._connection_pool[thread_id].execute("PRAGMA cache_size=10000")
                self._connection_pool[thread_id].execute("PRAGMA temp_store=MEMORY")
        
        try:
            yield self._connection_pool[thread_id]
        except Exception as e:
            logger.error(f"Database error: {e}")
            # Remove failed connection from pool
            if thread_id in self._connection_pool:
                del self._connection_pool[thread_id]
            raise

# Global database manager instance
db_manager = DatabaseManager()

def init_db():
    """Initialize database with optimized schema and indexes."""
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        
        # Create tokens table with optimized schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_address TEXT UNIQUE NOT NULL,
                base_symbol TEXT NOT NULL,
                quote_symbol TEXT NOT NULL,
                price_usd REAL,
                market_cap REAL,
                volume_1h REAL,
                tx_count_1h INTEGER,
                holders INTEGER,
                pair_created_at TIMESTAMP,
                fetched_at TIMESTAMP NOT NULL,
                rugcheck_url TEXT,
                bubblemaps_url TEXT,
                trend TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pair_address ON tokens(pair_address)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fetched_at ON tokens(fetched_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_cap ON tokens(market_cap)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_volume_1h ON tokens(volume_1h)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trend ON tokens(trend)')
        
        # Create cache table for API responses
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_expires ON api_cache(expires_at)')
        
        conn.commit()
        logger.info("Database initialized successfully")

def insert_token(**kwargs):
    """Insert token data with optimized batch processing."""
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        
        # Use parameterized query for security and performance
        cursor.execute('''
            INSERT OR REPLACE INTO tokens (
                pair_address, base_symbol, quote_symbol, price_usd, market_cap,
                volume_1h, tx_count_1h, holders, pair_created_at, fetched_at,
                rugcheck_url, bubblemaps_url, trend
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            kwargs.get('pair_address'),
            kwargs.get('base_symbol'),
            kwargs.get('quote_symbol'),
            kwargs.get('price_usd'),
            kwargs.get('market_cap'),
            kwargs.get('volume_1h'),
            kwargs.get('tx_count_1h'),
            kwargs.get('holders'),
            kwargs.get('pair_created_at'),
            kwargs.get('fetched_at'),
            kwargs.get('rugcheck_url'),
            kwargs.get('bubblemaps_url'),
            kwargs.get('trend')
        ))
        
        conn.commit()

def get_cached_data(key):
    """Get cached API response data."""
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT value FROM api_cache 
            WHERE key = ? AND expires_at > datetime('now')
        ''', (key,))
        result = cursor.fetchone()
        return result[0] if result else None

def set_cached_data(key, value, expires_in_seconds=300):
    """Cache API response data with expiration."""
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        expires_at = datetime.utcnow().timestamp() + expires_in_seconds
        cursor.execute('''
            INSERT OR REPLACE INTO api_cache (key, value, expires_at)
            VALUES (?, ?, datetime(?, 'unixepoch'))
        ''', (key, value, expires_at))
        conn.commit()

def cleanup_expired_cache():
    """Clean up expired cache entries."""
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM api_cache WHERE expires_at <= datetime("now")')
        conn.commit()