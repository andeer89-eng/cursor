import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from functools import lru_cache
from config import LOG_LEVEL

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """Optimized trend analyzer with multiple algorithms and caching."""
    
    def __init__(self, price_series: pd.Series, window_size: int = 20):
        self.price_series = price_series
        self.window_size = window_size
        self._cache = {}
        
        # Validate input
        if len(price_series) < window_size:
            raise ValueError(f"Price series must have at least {window_size} data points")
    
    @lru_cache(maxsize=128)
    def _calculate_sma(self, window: int) -> float:
        """Calculate Simple Moving Average with caching."""
        return self.price_series.rolling(window=window).mean().iloc[-1]
    
    @lru_cache(maxsize=128)
    def _calculate_ema(self, window: int) -> float:
        """Calculate Exponential Moving Average with caching."""
        return self.price_series.ewm(span=window).mean().iloc[-1]
    
    def _calculate_rsi(self, window: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(self.price_series) < window + 1:
            return 50.0  # Neutral RSI if insufficient data
        
        delta = self.price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _calculate_macd(self) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(self.price_series) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        ema12 = self.price_series.ewm(span=12).mean()
        ema26 = self.price_series.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line.iloc[-1],
            "signal": signal_line.iloc[-1],
            "histogram": histogram.iloc[-1]
        }
    
    def _calculate_bollinger_bands(self, window: int = 20, std_dev: float = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        sma = self.price_series.rolling(window=window).mean()
        std = self.price_series.rolling(window=window).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            "upper": upper_band.iloc[-1],
            "middle": sma.iloc[-1],
            "lower": lower_band.iloc[-1]
        }
    
    def _calculate_volume_weighted_avg_price(self, volume_series: Optional[pd.Series] = None) -> float:
        """Calculate Volume Weighted Average Price."""
        if volume_series is None:
            # Use equal weights if volume data not available
            return self.price_series.mean()
        
        if len(volume_series) != len(self.price_series):
            return self.price_series.mean()
        
        vwap = (self.price_series * volume_series).sum() / volume_series.sum()
        return vwap
    
    def _calculate_momentum_indicators(self) -> Dict[str, float]:
        """Calculate momentum indicators."""
        if len(self.price_series) < 10:
            return {"momentum": 0, "rate_of_change": 0}
        
        # Price momentum (current price - price n periods ago)
        momentum = self.price_series.iloc[-1] - self.price_series.iloc[-10]
        
        # Rate of change
        rate_of_change = ((self.price_series.iloc[-1] - self.price_series.iloc[-10]) / 
                         self.price_series.iloc[-10]) * 100
        
        return {
            "momentum": momentum,
            "rate_of_change": rate_of_change
        }
    
    def detect_trend(self) -> str:
        """Detect trend using multiple indicators with weighted scoring."""
        try:
            # Calculate all indicators
            sma_short = self._calculate_sma(10)
            sma_long = self._calculate_sma(20)
            ema_short = self._calculate_ema(12)
            ema_long = self._calculate_ema(26)
            rsi = self._calculate_rsi()
            macd_data = self._calculate_macd()
            bb_data = self._calculate_bollinger_bands()
            momentum_data = self._calculate_momentum_indicators()
            
            current_price = self.price_series.iloc[-1]
            
            # Initialize scoring
            bullish_score = 0
            bearish_score = 0
            neutral_score = 0
            
            # Moving Average Analysis (30% weight)
            if current_price > sma_short > sma_long:
                bullish_score += 30
            elif current_price < sma_short < sma_long:
                bearish_score += 30
            else:
                neutral_score += 30
            
            # EMA Analysis (20% weight)
            if current_price > ema_short > ema_long:
                bullish_score += 20
            elif current_price < ema_short < ema_long:
                bearish_score += 20
            else:
                neutral_score += 20
            
            # RSI Analysis (15% weight)
            if rsi > 70:
                bearish_score += 15  # Overbought
            elif rsi < 30:
                bullish_score += 15  # Oversold
            elif 40 < rsi < 60:
                neutral_score += 15
            else:
                if rsi > 50:
                    bullish_score += 7.5
                else:
                    bearish_score += 7.5
            
            # MACD Analysis (15% weight)
            if macd_data["macd"] > macd_data["signal"] and macd_data["histogram"] > 0:
                bullish_score += 15
            elif macd_data["macd"] < macd_data["signal"] and macd_data["histogram"] < 0:
                bearish_score += 15
            else:
                neutral_score += 15
            
            # Bollinger Bands Analysis (10% weight)
            if current_price < bb_data["lower"]:
                bullish_score += 10  # Oversold
            elif current_price > bb_data["upper"]:
                bearish_score += 10  # Overbought
            else:
                neutral_score += 10
            
            # Momentum Analysis (10% weight)
            if momentum_data["momentum"] > 0 and momentum_data["rate_of_change"] > 0:
                bullish_score += 10
            elif momentum_data["momentum"] < 0 and momentum_data["rate_of_change"] < 0:
                bearish_score += 10
            else:
                neutral_score += 10
            
            # Determine trend based on highest score
            max_score = max(bullish_score, bearish_score, neutral_score)
            
            if max_score == bullish_score:
                return "bullish"
            elif max_score == bearish_score:
                return "bearish"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error in trend detection: {e}")
            return "neutral"
    
    def get_trend_strength(self) -> float:
        """Get trend strength as a percentage (0-100)."""
        try:
            # Calculate trend strength based on price movement consistency
            if len(self.price_series) < 10:
                return 50.0
            
            # Calculate price changes
            price_changes = self.price_series.pct_change().dropna()
            
            # Calculate consistency of direction
            positive_changes = (price_changes > 0).sum()
            total_changes = len(price_changes)
            
            if total_changes == 0:
                return 50.0
            
            consistency = abs(positive_changes / total_changes - 0.5) * 2  # 0-1 scale
            strength = consistency * 100
            
            return min(strength, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 50.0
    
    def get_support_resistance_levels(self) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        try:
            if len(self.price_series) < 20:
                return {"support": 0, "resistance": 0}
            
            # Use recent price range
            recent_prices = self.price_series.tail(20)
            support = recent_prices.min()
            resistance = recent_prices.max()
            
            return {
                "support": support,
                "resistance": resistance
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {"support": 0, "resistance": 0}
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary."""
        trend = self.detect_trend()
        strength = self.get_trend_strength()
        levels = self.get_support_resistance_levels()
        
        return {
            "trend": trend,
            "strength": strength,
            "support": levels["support"],
            "resistance": levels["resistance"],
            "current_price": self.price_series.iloc[-1],
            "price_change_24h": self._calculate_24h_change(),
            "volatility": self._calculate_volatility()
        }
    
    def _calculate_24h_change(self) -> float:
        """Calculate 24-hour price change percentage."""
        try:
            if len(self.price_series) < 2:
                return 0.0
            
            current = self.price_series.iloc[-1]
            previous = self.price_series.iloc[-2]
            
            if previous == 0:
                return 0.0
            
            return ((current - previous) / previous) * 100
            
        except Exception as e:
            logger.error(f"Error calculating 24h change: {e}")
            return 0.0
    
    def _calculate_volatility(self) -> float:
        """Calculate price volatility."""
        try:
            if len(self.price_series) < 10:
                return 0.0
            
            returns = self.price_series.pct_change().dropna()
            return returns.std() * 100  # Convert to percentage
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0