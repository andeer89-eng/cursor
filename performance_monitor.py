import time
import psutil
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import sqlite3
from dataclasses import dataclass, asdict
from collections import deque
import threading

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float
    active_connections: int
    requests_per_second: float
    response_time_avg: float
    cache_hit_rate: float
    database_queries: int
    errors_count: int

class PerformanceMonitor:
    """Performance monitoring and optimization system."""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        self.db_path = db_path
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 metrics
        self.start_time = time.time()
        self.last_metrics = None
        self.monitoring = False
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_db()
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'response_time_avg': 5.0,  # seconds
            'cache_hit_rate': 0.7,  # 70%
            'errors_count': 10  # per hour
        }
    
    def _init_db(self):
        """Initialize performance metrics database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_mb REAL,
                    disk_io_read REAL,
                    disk_io_write REAL,
                    network_sent REAL,
                    network_recv REAL,
                    active_connections INTEGER,
                    requests_per_second REAL,
                    response_time_avg REAL,
                    cache_hit_rate REAL,
                    database_queries INTEGER,
                    errors_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better query performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_metrics(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cpu_percent ON performance_metrics(cpu_percent)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_percent ON performance_metrics(memory_percent)')
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used / (1024 * 1024)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_read = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_io_write = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_sent = network_io.bytes_sent / (1024 * 1024)
            network_recv = network_io.bytes_recv / (1024 * 1024)
            
            # Active connections
            active_connections = len(psutil.net_connections())
            
            # Calculate derived metrics
            requests_per_second = self._calculate_requests_per_second()
            response_time_avg = self._calculate_avg_response_time()
            cache_hit_rate = self._calculate_cache_hit_rate()
            database_queries = self._get_database_query_count()
            errors_count = self._get_error_count()
            
            metrics = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb,
                disk_io_read=disk_io_read,
                disk_io_write=disk_io_write,
                network_sent=network_sent,
                network_recv=network_recv,
                active_connections=active_connections,
                requests_per_second=requests_per_second,
                response_time_avg=response_time_avg,
                cache_hit_rate=cache_hit_rate,
                database_queries=database_queries,
                errors_count=errors_count
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return None
    
    def _calculate_requests_per_second(self) -> float:
        """Calculate requests per second (placeholder implementation)."""
        # In a real implementation, this would track actual HTTP requests
        return 0.0
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time (placeholder implementation)."""
        # In a real implementation, this would track actual response times
        return 0.0
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder implementation)."""
        # In a real implementation, this would track actual cache hits/misses
        return 0.0
    
    def _get_database_query_count(self) -> int:
        """Get database query count (placeholder implementation)."""
        # In a real implementation, this would track actual database queries
        return 0
    
    def _get_error_count(self) -> int:
        """Get error count (placeholder implementation)."""
        # In a real implementation, this would track actual errors
        return 0
    
    def store_metrics(self, metrics: PerformanceMetrics):
        """Store metrics in database and memory."""
        if not metrics:
            return
        
        with self.lock:
            # Store in memory
            self.metrics_history.append(metrics)
            self.last_metrics = metrics
            
            # Store in database
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO performance_metrics (
                            timestamp, cpu_percent, memory_percent, memory_mb,
                            disk_io_read, disk_io_write, network_sent, network_recv,
                            active_connections, requests_per_second, response_time_avg,
                            cache_hit_rate, database_queries, errors_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics.timestamp, metrics.cpu_percent, metrics.memory_percent,
                        metrics.memory_mb, metrics.disk_io_read, metrics.disk_io_write,
                        metrics.network_sent, metrics.network_recv, metrics.active_connections,
                        metrics.requests_per_second, metrics.response_time_avg,
                        metrics.cache_hit_rate, metrics.database_queries, metrics.errors_count
                    ))
            except Exception as e:
                logger.error(f"Error storing metrics: {e}")
    
    def check_performance_alerts(self, metrics: PerformanceMetrics) -> List[str]:
        """Check for performance alerts based on thresholds."""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.response_time_avg > self.thresholds['response_time_avg']:
            alerts.append(f"Slow response time: {metrics.response_time_avg:.2f}s")
        
        if metrics.cache_hit_rate < self.thresholds['cache_hit_rate']:
            alerts.append(f"Low cache hit rate: {metrics.cache_hit_rate:.2%}")
        
        if metrics.errors_count > self.thresholds['errors_count']:
            alerts.append(f"High error count: {metrics.errors_count}")
        
        return alerts
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                
                # Get metrics from database
                cursor = conn.execute('''
                    SELECT 
                        AVG(cpu_percent) as avg_cpu,
                        MAX(cpu_percent) as max_cpu,
                        AVG(memory_percent) as avg_memory,
                        MAX(memory_percent) as max_memory,
                        AVG(response_time_avg) as avg_response_time,
                        MAX(response_time_avg) as max_response_time,
                        AVG(cache_hit_rate) as avg_cache_hit_rate,
                        SUM(errors_count) as total_errors,
                        COUNT(*) as total_measurements
                    FROM performance_metrics 
                    WHERE timestamp >= ?
                ''', (cutoff_time,))
                
                row = cursor.fetchone()
                
                if row:
                    return {
                        'period_hours': hours,
                        'avg_cpu_percent': row[0] or 0,
                        'max_cpu_percent': row[1] or 0,
                        'avg_memory_percent': row[2] or 0,
                        'max_memory_percent': row[3] or 0,
                        'avg_response_time': row[4] or 0,
                        'max_response_time': row[5] or 0,
                        'avg_cache_hit_rate': row[6] or 0,
                        'total_errors': row[7] or 0,
                        'total_measurements': row[8] or 0
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        if not self.last_metrics:
            return recommendations
        
        # CPU optimization
        if self.last_metrics.cpu_percent > 70:
            recommendations.append("Consider implementing connection pooling to reduce CPU usage")
            recommendations.append("Optimize database queries and add indexes")
            recommendations.append("Implement caching for frequently accessed data")
        
        # Memory optimization
        if self.last_metrics.memory_percent > 80:
            recommendations.append("Implement memory-efficient data structures")
            recommendations.append("Add garbage collection optimization")
            recommendations.append("Consider using streaming for large datasets")
        
        # Response time optimization
        if self.last_metrics.response_time_avg > 2.0:
            recommendations.append("Implement async/await for I/O operations")
            recommendations.append("Add request caching and rate limiting")
            recommendations.append("Optimize database connection pooling")
        
        # Cache optimization
        if self.last_metrics.cache_hit_rate < 0.5:
            recommendations.append("Increase cache size and TTL")
            recommendations.append("Implement cache warming strategies")
            recommendations.append("Add cache invalidation policies")
        
        return recommendations
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous performance monitoring."""
        self.monitoring = True
        logger.info(f"Starting performance monitoring (interval: {interval_seconds}s)")
        
        while self.monitoring:
            try:
                # Collect and store metrics
                metrics = self.collect_metrics()
                if metrics:
                    self.store_metrics(metrics)
                    
                    # Check for alerts
                    alerts = self.check_performance_alerts(metrics)
                    if alerts:
                        logger.warning(f"Performance alerts: {', '.join(alerts)}")
                    
                    # Log current metrics
                    logger.info(f"CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%, "
                              f"Response Time: {metrics.response_time_avg:.2f}s")
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        logger.info("Performance monitoring stopped")
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics to JSON file."""
        if filename is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f'performance_metrics_{timestamp}.json'
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT * FROM performance_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                ''')
                
                columns = [description[0] for description in cursor.description]
                data = []
                
                for row in cursor.fetchall():
                    data.append(dict(zip(columns, row)))
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                
                logger.info(f"Metrics exported to {filename}")
                return filename
                
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return None

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

async def main():
    """Main function for performance monitoring."""
    try:
        # Start monitoring
        await performance_monitor.start_monitoring(interval_seconds=30)
    except KeyboardInterrupt:
        performance_monitor.stop_monitoring()
        print("\nPerformance monitoring stopped")

if __name__ == "__main__":
    asyncio.run(main())