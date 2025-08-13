import sqlite3
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging
from config import DATABASE_URL, LOG_LEVEL

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Optimized database manager with connection pooling and caching."""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._connection_pool = []
        self._max_connections = 10
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cleanup = time.time()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables with optimized schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tokens table with optimized indexes
            cursor.execute("""
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
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tokens_fetched_at 
                ON tokens(fetched_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tokens_market_cap 
                ON tokens(market_cap)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tokens_volume_1h 
                ON tokens(volume_1h)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tokens_trend 
                ON tokens(trend)
            """)
            
            # Create performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection from the pool."""
        conn = None
        try:
            with self._lock:
                if self._connection_pool:
                    conn = self._connection_pool.pop()
                else:
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.execute("PRAGMA journal_mode=WAL")  # Better performance
                    conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
                    conn.execute("PRAGMA cache_size=10000")  # Larger cache
                    conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
            
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                with self._lock:
                    if len(self._connection_pool) < self._max_connections:
                        self._connection_pool.append(conn)
                    else:
                        conn.close()
    
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
    
    def insert_token(self, **kwargs) -> bool:
        """Insert a token with optimized batch processing."""
        try:
            self._cleanup_cache()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Use parameterized query for security and performance
                cursor.execute("""
                    INSERT OR REPLACE INTO tokens (
                        pair_address, base_symbol, quote_symbol, price_usd,
                        market_cap, volume_1h, tx_count_1h, holders,
                        pair_created_at, fetched_at, rugcheck_url,
                        bubblemaps_url, trend
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
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
                return True
                
        except Exception as e:
            logger.error(f"Error inserting token: {e}")
            return False
    
    def batch_insert_tokens(self, tokens: list) -> int:
        """Insert multiple tokens in a single transaction for better performance."""
        if not tokens:
            return 0
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Use executemany for batch insertion
                cursor.executemany("""
                    INSERT OR REPLACE INTO tokens (
                        pair_address, base_symbol, quote_symbol, price_usd,
                        market_cap, volume_1h, tx_count_1h, holders,
                        pair_created_at, fetched_at, rugcheck_url,
                        bubblemaps_url, trend
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        token.get('pair_address'),
                        token.get('base_symbol'),
                        token.get('quote_symbol'),
                        token.get('price_usd'),
                        token.get('market_cap'),
                        token.get('volume_1h'),
                        token.get('tx_count_1h'),
                        token.get('holders'),
                        token.get('pair_created_at'),
                        token.get('fetched_at'),
                        token.get('rugcheck_url'),
                        token.get('bubblemaps_url'),
                        token.get('trend')
                    ) for token in tokens
                ])
                
                conn.commit()
                return len(tokens)
                
        except Exception as e:
            logger.error(f"Error batch inserting tokens: {e}")
            return 0
    
    def get_recent_tokens(self, hours: int = 24, limit: int = 100) -> list:
        """Get recent tokens with caching."""
        cache_key = f"recent_tokens_{hours}_{limit}"
        self._cleanup_cache()
        
        if cache_key in self._cache:
            return self._cache[cache_key][0]
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM tokens 
                    WHERE fetched_at >= datetime('now', '-{} hours')
                    ORDER BY fetched_at DESC 
                    LIMIT ?
                """.format(hours), (limit,))
                
                columns = [description[0] for description in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                # Cache the results
                self._cache[cache_key] = (results, time.time())
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting recent tokens: {e}")
            return []
    
    def record_metric(self, metric_name: str, metric_value: float):
        """Record performance metrics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO performance_metrics (metric_name, metric_value)
                    VALUES (?, ?)
                """, (metric_name, metric_value))
                conn.commit()
        except Exception as e:
            logger.error(f"Error recording metric: {e}")

# Global database instance
_db_manager: Optional[DatabaseManager] = None

def init_db():
    """Initialize the database."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def insert_token(**kwargs) -> bool:
    """Insert a single token."""
    if _db_manager is None:
        init_db()
    return _db_manager.insert_token(**kwargs)

def batch_insert_tokens(tokens: list) -> int:
    """Insert multiple tokens in batch."""
    if _db_manager is None:
        init_db()
    return _db_manager.batch_insert_tokens(tokens)

def get_recent_tokens(hours: int = 24, limit: int = 100) -> list:
    """Get recent tokens."""
    if _db_manager is None:
        init_db()
    return _db_manager.get_recent_tokens(hours, limit)