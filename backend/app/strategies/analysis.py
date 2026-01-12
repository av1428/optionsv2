import pandas as pd
import numpy as np

class TechnicalAnalysis:
    
    @staticmethod
    def calculate_pivots(high, low, close):
        """Calculate Standard Pivot Points."""
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        return {"pivot": pivot, "r1": r1, "s1": s1, "r2": r2, "s2": s2}

    @staticmethod
    def identify_trend(df: pd.DataFrame, period: int = 50):
        """
        Simple trend identification using Moving Averages.
        Bullish: Price > SMA50
        """
        if len(df) < period:
            return "Unknown"
        
        df['SMA'] = df['Close'].rolling(window=period).mean()
        current_price = df['Close'].iloc[-1]
        sma_value = df['SMA'].iloc[-1]
        
        if current_price > sma_value * 1.01:
            return "Uptrend"
        elif current_price < sma_value * 0.99:
            return "Downtrend"
        else:
            return "Ranging"

    @staticmethod
    def find_support_resistance(df: pd.DataFrame):
        """
        Identify key support and resistance levels based on local min/max
        and volume profiles (simplified).
        """
        # Simple local min/max over windows
        n = 20  # Window size
        df['min'] = df['Low'].iloc[(n-1):].rolling(window=n, center=True).min()
        df['max'] = df['High'].iloc[(n-1):].rolling(window=n, center=True).max()
        
        supports = df[df['Low'] == df['min']]['Low'].unique()
        resistances = df[df['High'] == df['max']]['High'].unique()
        
        # Filter levels close to current price for relevance
        current_price = df['Close'].iloc[-1]
        
        relevant_supports = [s for s in supports if s < current_price]
        relevant_resistances = [r for r in resistances if r > current_price]
        
        # Sort and take closest
        relevant_supports.sort()
        relevant_resistances.sort()
        
        return {
            "major_support": relevant_supports[-1] if relevant_supports else (current_price * 0.95),
            "major_resistance": relevant_resistances[0] if relevant_resistances else (current_price * 1.05)
        }

    @staticmethod
    def analyze_volume(df: pd.DataFrame):
        """
        Analyze Volume for Institutional Proxy.
        High Volume + Low Body = Potential Reversal/Churn (Institutional Activity)
        """
        recent = df.iloc[-5:] # Last 5 days
        avg_vol = df['Volume'].mean()
        
        signals = []
        for i, row in recent.iterrows():
            if row['Volume'] > avg_vol * 1.5:
                # High Volume
                body_size = abs(row['Close'] - row['Open'])
                range_size = row['High'] - row['Low']
                if range_size > 0 and (body_size / range_size) < 0.3:
                     # High Vol, Small Body (Doji/Spinning Top) -> Indecision/Absorption
                     signals.append(f"High Vol Absorption on {row.name}")
        
        return signals

technical_analysis = TechnicalAnalysis()
