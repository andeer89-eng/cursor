import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """Optimized trend analyzer with caching and efficient algorithms."""
    
    def __init__(self, price_series: pd.Series, window_size: int = 10):
        self.price_series = price_series
        self.window_size = window_size
        self._cache = {}
    
    @lru_cache(maxsize=128)
    def _calculate_sma(self, window: int) -> float:
        """Calculate Simple Moving Average with caching."""
        if len(self.price_series) < window:
            return np.nan
        return self.price_series.tail(window).mean()
    
    @lru_cache(maxsize=128)
    def _calculate_ema(self, window: int) -> float:
        """Calculate Exponential Moving Average with caching."""
        if len(self.price_series) < window:
            return np.nan
        return self.price_series.ewm(span=window).mean().iloc[-1]
    
    def _calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI efficiently."""
        if len(self.price_series) < period + 1:
            return 50.0  # Neutral RSI
        
        # Calculate price changes
        delta = self.price_series.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean().iloc[-1]
        avg_losses = losses.rolling(window=period).mean().iloc[-1]
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self) -> tuple:
        """Calculate MACD efficiently."""
        if len(self.price_series) < 26:
            return 0.0, 0.0, 0.0
        
        ema12 = self.price_series.ewm(span=12).mean()
        ema26 = self.price_series.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        return (
            macd_line.iloc[-1],
            signal_line.iloc[-1],
            histogram.iloc[-1]
        )
    
    def _calculate_bollinger_bands(self, window: int = 20, std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands efficiently."""
        if len(self.price_series) < window:
            return np.nan, np.nan, np.nan
        
        sma = self.price_series.rolling(window=window).mean().iloc[-1]
        std = self.price_series.rolling(window=window).std().iloc[-1]
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def detect_trend(self) -> str:
        """Detect trend using multiple indicators with optimized calculations."""
        try:
            if len(self.price_series) < 5:
                return "neutral"
            
            # Calculate indicators
            current_price = self.price_series.iloc[-1]
            sma_short = self._calculate_sma(5)
            sma_long = self._calculate_sma(20)
            ema_short = self._calculate_ema(12)
            ema_long = self._calculate_ema(26)
            rsi = self._calculate_rsi()
            macd_line, signal_line, histogram = self._calculate_macd()
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands()
            
            # Trend scoring system
            bullish_signals = 0
            bearish_signals = 0
            
            # Price vs Moving Averages
            if current_price > sma_short and sma_short > sma_long:
                bullish_signals += 2
            elif current_price < sma_short and sma_short < sma_long:
                bearish_signals += 2
            
            # EMA comparison
            if ema_short > ema_long:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # RSI analysis
            if rsi > 50 and rsi < 70:
                bullish_signals += 1
            elif rsi < 50 and rsi > 30:
                bearish_signals += 1
            elif rsi > 70:
                bearish_signals += 1  # Overbought
            elif rsi < 30:
                bullish_signals += 1  # Oversold
            
            # MACD analysis
            if macd_line > signal_line and histogram > 0:
                bullish_signals += 1
            elif macd_line < signal_line and histogram < 0:
                bearish_signals += 1
            
            # Bollinger Bands analysis
            if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                if current_price > bb_upper:
                    bearish_signals += 1  # Overbought
                elif current_price < bb_lower:
                    bullish_signals += 1  # Oversold
                elif current_price > bb_middle:
                    bullish_signals += 0.5
                else:
                    bearish_signals += 0.5
            
            # Volume trend (if available)
            if hasattr(self.price_series, 'volume'):
                recent_volume = self.price_series.volume.tail(5).mean()
                avg_volume = self.price_series.volume.tail(20).mean()
                if recent_volume > avg_volume * 1.5:
                    if bullish_signals > bearish_signals:
                        bullish_signals += 1
                    else:
                        bearish_signals += 1
            
            # Determine trend based on signal strength
            signal_threshold = 3
            if bullish_signals >= signal_threshold and bullish_signals > bearish_signals:
                return "bullish"
            elif bearish_signals >= signal_threshold and bearish_signals > bullish_signals:
                return "bearish"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error in trend detection: {e}")
            return "neutral"
    
    def get_trend_strength(self) -> float:
        """Get trend strength as a percentage (0-100)."""
        try:
            if len(self.price_series) < 10:
                return 50.0
            
            # Calculate price momentum
            recent_prices = self.price_series.tail(10)
            price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            # Calculate volatility
            returns = self.price_series.pct_change().dropna()
            volatility = returns.tail(20).std()
            
            # Normalize trend strength
            strength = abs(price_change) / (volatility + 1e-8) * 100
            return min(strength, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 50.0