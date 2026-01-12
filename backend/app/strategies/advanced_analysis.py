import pandas as pd
import numpy as np

class ConfluenceAI:
    """
    Institutional-Grade Analysis Engine.
    Uses 'Confluence' of multiple factors to generate high-probability trade signals.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate_indicators()

    def _calculate_indicators(self):
        """Pre-calculate all necessary technicals using standard Pandas."""
        close = self.df['Close']
        low = self.df['Low']
        high = self.df['High']
        
        # 1. Trend
        self.df['SMA_20'] = close.rolling(window=20).mean()
        self.df['SMA_50'] = close.rolling(window=50).mean()
        self.df['SMA_200'] = close.rolling(window=200).mean()

        # 2. Volatility - Conservative Tier (The Fortress)
        # 50 SMA +/- 3 STD (99.7% Probability)
        std_50 = close.rolling(window=50).std()
        self.df['BB_CONS_UPPER'] = self.df['SMA_50'] + (3.0 * std_50)
        self.df['BB_CONS_LOWER'] = self.df['SMA_50'] - (3.0 * std_50)

        # 3. Volatility - Aggressive Tier (The Sniper) - User Requested 21/2.1
        # 21 SMA +/- 2.1 STD (95%+ Probability)
        sma_21 = close.rolling(window=21).mean()
        std_21 = close.rolling(window=21).std()
        self.df['BB_AGGR_UPPER'] = sma_21 + (2.1 * std_21)
        self.df['BB_AGGR_LOWER'] = sma_21 - (2.1 * std_21)

        # 4. Momentum (RSI 14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # 5. ATR (14)
        # TR = Max(H-L, Abs(H-Cp), Abs(L-Cp))
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(window=14).mean()
        
        # 6. Volume Stats
        self.vol_avg = self.df['Volume'].rolling(20).mean().iloc[-1]

    def get_volume_profile_levels(self, bins=50):
        """
        Approximates Volume Profile using Price Bins.
        Returns Point of Control (POC) and High Volume Nodes (HVN).
        """
        # Create price bins
        price_min = self.df['Low'].min()
        price_max = self.df['High'].max()
        hist, bin_edges = np.histogram(self.df['Close'], bins=bins, weights=self.df['Volume'])
        
        # Find POC (Bin with max volume)
        max_idx = hist.argmax()
        poc_price = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
        
        # Find HVNs (Peaks in histogram)
        # Simple logic: Bins with > 75th percentile volume
        threshold = np.percentile(hist, 75)
        hvn_indices = np.where(hist > threshold)[0]
        hvns = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in hvn_indices]
        
        return {
            "POC": poc_price,
            "HVN": hvns
        }

    def analyze_confluence(self):
        """
        Generates Dual-Tier Strategy Recommendations.
        """
        row = self.df.iloc[-1]
        vp = self.get_volume_profile_levels()
        
        # Falling Knife / Panic Filter
        # Handle nan/inf in calc
        vol_check = row['Volume'] > 2.0 * self.vol_avg if not pd.isna(self.vol_avg) else False
        rsi_check = row['RSI'] < 25 if not pd.isna(row['RSI']) else False
        is_panic = vol_check or rsi_check
        
        strategies = {
            "Conservative": None,
            "Aggressive": None,
            "Notes": []
        }
        
        if is_panic:
            strategies['Notes'].append("⚠️ PANIC DETECTED: Volume/RSI Extreme. No Bullish Trades.")

        # --- Logic A: Trend Momentum (The 89% Win Rate Strategy) ---
        # Condition: Strong Uptrend (Price > 20SMA > 50SMA)
        # This is the "Growth Mode" strategy validated in Backtests.
        is_strong_trend = False
        if not pd.isna(row['SMA_20']) and not pd.isna(row['SMA_50']):
            is_strong_trend = row['Close'] > row['SMA_20'] and row['SMA_20'] > row['SMA_50']
            
            if is_strong_trend and not is_panic:
                 strategies['Aggressive'] = {
                    "Type": "Put Credit Spread (Bullish)",
                    "Reason": "Trend Momentum: Price surfing above 20SMA",
                    "Strike_Short": row['SMA_50'], # Support Level
                    "Strike_Long": row['SMA_50'] - 5 
                }
                 # Conservative also likes this trend, but wants deeper entry
                 strategies['Conservative'] = {
                    "Type": "Put Credit Spread (Bullish)",
                    "Reason": "Trend Backbone: Selling Puts at 50SMA Support",
                    "Strike_Short": row['SMA_50'] * 0.98, # 2% OTM Buffer
                    "Strike_Long": (row['SMA_50'] * 0.98) - 5
                }
        
        # --- Logic B: Dip Buy (Mean Reversion) ---
        # Condition: Uptrend (Price > 50SMA) + Pullback to Bands
        # Only triggers if "Trend Momentum" didn't already claim the slot, OR if it's a better signal
        if pd.isna(row['SMA_50']): return strategies

        is_uptrend = row['Close'] > row['SMA_50']
        
        if is_uptrend and not is_panic and not strategies['Aggressive']:
            # Check Aggressive Dip (21 SMA Band)
            if row['Low'] <= row['BB_AGGR_LOWER']:
                strategies['Aggressive'] = {
                    "Type": "Put Credit Spread (Bullish)",
                    "Reason": "Dip Buy: Uptrend + Touch of Lower 21-2.1 Bollinger Band",
                    "Strike_Short": row['BB_AGGR_LOWER'], 
                    "Strike_Long": row['BB_AGGR_LOWER'] - 5 
                }
        
        if is_uptrend and not is_panic and not strategies['Conservative']:
             # Check Conservative Dip (50 SMA Band)
             if row['Low'] <= row['BB_CONS_LOWER']:
                strategies['Conservative'] = {
                    "Type": "Put Credit Spread (Bullish)",
                    "Reason": "Deep Value: Selling below 50SMA-3SD Fortress",
                    "Strike_Short": row['BB_CONS_LOWER'], 
                    "Strike_Long": row['BB_CONS_LOWER'] - 5
                }
        
        # --- Logic C: Mean Reversion (Overbought) ---
        # Condition: Price > Upper Band + RSI > 70
        if not pd.isna(row['RSI']) and row['RSI'] > 70:
            if row['High'] >= row['BB_AGGR_UPPER']:
                 strategies['Aggressive'] = { # Overwrite Bullish signal if we are extremely overbought
                    "Type": "Call Credit Spread (Bearish)",
                    "Reason": "RSI > 70 + Touch of Upper 21-2.1 Bollinger Band",
                    "Strike_Short": row['BB_AGGR_UPPER'],
                    "Strike_Long": row['BB_AGGR_UPPER'] + 5 
                }
            
            # Conservative needs Extreme Extension
            if row['High'] >= row['BB_CONS_UPPER']:
                 strategies['Conservative'] = {
                    "Type": "Call Credit Spread (Bearish)",
                    "Reason": "Extreme Extension: RSI > 70 + Upper 50-3SD Band",
                    "Strike_Short": row['BB_CONS_UPPER'],
                    "Strike_Long": row['BB_CONS_UPPER'] + 5
                }
                
        # --- Logic C: Volume Support ---
        # If Price is near POC/HVN, it's a support zone
        if vp['HVN']:
            closest_hvn = min(vp['HVN'], key=lambda x: abs(x - row['Close']))
            if abs(closest_hvn - row['Close']) / row['Close'] < 0.01: # Within 1%
                 strategies['Notes'].append(f"ℹ️ Confluence: Price is at Major Volume Node (${closest_hvn:.2f})")
             
        return strategies
