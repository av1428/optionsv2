import pandas as pd
import numpy as np
from datetime import timedelta
from backend.app.strategies.advanced_analysis import ConfluenceAI

class Backtester:
    """
    Hedge Fund Simulator: Multi-Asset, Dynamic Sizing, Profit Taking.
    """
    
    def __init__(self, initial_capital=5000, max_positions=5):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.capital = initial_capital
        self.portfolio = [] # List of open trades
        self.trade_log = []
        self.equity_curve = []
        
        # Strategy Parameters
        self.risk_per_trade = 0.05 # Risk 5% of account per trade
        self.profit_target = 0.50 # Take profit at 50% of max credit
        self.stop_loss = 2.0 # Stop at 200% loss (2x credit)
        self.target_dte = 30
        self.spread_width = 5.0

    def run_portfolio(self, data_map, start_date=None, end_date=None):
        """
        Run simulation across multiple assets simultaneously.
        data_map: { 'AAPL': df, 'TSLA': df ... }
        start_date, end_date: datetime objects or strings
        """
        # Convert inputs to timestamps
        if start_date: start_date = pd.to_datetime(start_date)
        if end_date: end_date = pd.to_datetime(end_date)

        # Pre-calc indicators for speed
        analyzed_map = {}
        for ticker, df in data_map.items():
            # Ensure Date Index
            if not isinstance(df.index, pd.DatetimeIndex) and 'Date' in df.columns:
                 df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None) # Normalize
                 df = df.set_index('Date')
            
            # We need full history for indicators
            engine = ConfluenceAI(df)
            # Re-assign processed DF to map, preserving the index we just set
            analyzed_map[ticker] = engine.df

        # 1. Align Dates (Find common date range)
        # Now filtering from analyzed_map which has proper indexes
        all_dates = sorted(list(set(d for df in analyzed_map.values() for d in df.index)))

        # Filter Simulation Dates
        # We must ensure we don't start before we have indicators (200 days)
        # But if the user provides a start_date, we assume they (or the caller) ensured enough buffer exists.
        # We will iterate through requested window.
        
        sim_dates = [d for d in all_dates if pd.isna(d) == False]
        
        if start_date:
            sim_dates = [d for d in sim_dates if d >= start_date]
        if end_date:
            sim_dates = [d for d in sim_dates if d <= end_date]
            
        if not sim_dates:
            return {"error": "No trading days found in selected range."}

        # Time Loop
        for current_date in sim_dates:
            # 1. Manage Open Positions (Mark-to-Market)
            self._manage_portfolio(current_date, analyzed_map)
            
            # 2. Scan for New Opportunities
            if len(self.portfolio) < self.max_positions:
                self._scan_market(current_date, analyzed_map)
            
            # Track Equity
            locked = sum(p['margin_locked'] for p in self.portfolio)
            self.equity_curve.append({
                "date": current_date, 
                "equity": self.capital + locked,
                "cash": self.capital
            })
            
        return self._generate_report()

    def _scan_market(self, date, data_map):
        """Find best setup across all tickers for this date."""
        candidates = []
        
        for ticker, df in data_map.items():
            if date not in df.index: continue
            
            row = df.loc[date]
            if pd.isna(row['SMA_50']): continue
            
            score = 0
            signal_type = None
            strike_short = 0
            spot = row['Close']
            
            # --- STRATEGY 1: CONFLUENCE DIP (Mean Reversion) ---
            # Buying the dip in an uptrend.
            if spot > row['SMA_50'] and row['RSI'] < 40: # Oversold in Uptrend
                # Strike: Use Conservative Band or 3% OTM, whichever is safer
                # FORCE OTM: Strike must be < Spot
                safe_strike = min(row['BB_CONS_LOWER'], spot * 0.97) 
                
                score += 3 # High quality
                signal_type = "Put Credit Spread"
                strike_short = safe_strike

            # --- STRATEGY 2: TREND MOMENTUM (Growth) ---
            # Price surfing above SMA 20 (Strong Trend)
            elif spot > row['SMA_20'] and row['SMA_20'] > row['SMA_50']:
                # Sell Puts at SMA 50 (Strong Support)
                # This is usually deep OTM (Safe)
                support_strike = row['SMA_50']
                
                if support_strike < spot * 0.98: # Ensure at least 2% OTM
                    score += 2 # Good quality
                    signal_type = "Put Credit Spread"
                    strike_short = support_strike
            
            # --- STRATEGY 3: BEARISH (Call Spreads) ---
            # Only if downtrend
            elif spot < row['SMA_50'] and row['RSI'] > 70:
                safe_strike = max(row['BB_CONS_UPPER'], spot * 1.03)
                score += 2
                signal_type = "Call Credit Spread"
                strike_short = safe_strike

            if score > 0:
                candidates.append({
                    "ticker": ticker,
                    "score": score,
                    "type": signal_type,
                    "strike_short": strike_short,
                    "spot": spot
                })
        
        # Sort by Score (Quality)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Enter Top Picks
        for setup in candidates:
            if len(self.portfolio) >= self.max_positions: break
            
            # Diversify: Max 1 trade per ticker
            if any(p['ticker'] == setup['ticker'] for p in self.portfolio): continue

            self._open_trade(date, setup)

    def _open_trade(self, date, setup):
        # Dynamic Sizing: Risk 5% of CURRENT Equity
        current_equity = self.capital + sum(p['margin_locked'] for p in self.portfolio)
        risk_amt = current_equity * self.risk_per_trade
        
        max_loss_per_contract = self.spread_width * 100
        # Check capital
        if self.capital < max_loss_per_contract: return 

        contracts = max(1, int(risk_amt / max_loss_per_contract))
        
        # Synthetic Pricing: 20% Credit
        credit_per_contract = max_loss_per_contract * 0.20
        total_credit = credit_per_contract * contracts
        margin_req = (max_loss_per_contract * contracts) - total_credit
        
        if self.capital < margin_req: return # Safety check
        
        self.capital -= margin_req
        self.capital += total_credit # Collect premium
        
        self.portfolio.append({
            "ticker": setup['ticker'],
            "entry_date": date,
            "type": setup['type'],
            "strike_short": setup['strike_short'],
            "strike_long": setup['strike_short'] - 5 if "Put" in setup['type'] else setup['strike_short'] + 5,
            "margin_locked": max_loss_per_contract * contracts,
            "contracts": contracts,
            "credit": total_credit,
            "max_credit": total_credit,
            "entry_spot": setup['spot']
        })

    def _manage_portfolio(self, date, data_map):
        """Check Exits: Profit Target or Stop Loss."""
        for trade in self.portfolio[:]:
            ticker = trade['ticker']
            if date not in data_map[ticker].index: continue
            
            row = data_map[ticker].loc[date]
            curr_price = row['Close']
            
            # Days held
            days = (pd.to_datetime(date) - pd.to_datetime(trade['entry_date'])).days
            
            # 1. Calc Current P/L (Synthetic)
            # Theoretical Option Value DECAY
            # Simple decay model: Value = Initial_Credit * (1 - (days/30)) 
            # This is linear decay (Theta).
            # Delta impact: If price moves against us, value increases.
            
            # Simplified Management: 
            # Valid validation is ONLY possible at Expiration or Stop Loss logic based on stock price.
            # Using Stock Price Triggers:
            
            exit_signal = False
            pnl = None
            reason = ""
            
            # Expiration
            if days >= self.target_dte:
                exit_signal = True
                reason = "Expired"
                 # (Same exp logic as before)
            
            # Stop Loss (Price breached strike significantly)
            # If Put Spread, and price < long strike -> Max Loss mostly.
            if "Put" in trade['type'] and curr_price < trade['strike_long']:
                exit_signal = True
                reason = "Stop Loss (Max)"
                # Assume Max Loss
            
            # Profit Take (Early) - Hard to simul without option data
            # Proxy: If price > strike + 5% (safe) AND days > 15 (theta decay), take profit.
            if "Put" in trade['type'] and curr_price > trade['strike_short'] * 1.05 and days > 10:
                # Assume 50% profit
                exit_signal = True
                pnl = trade['credit'] * 0.50 # Keep 50%
                reason = "Take Profit (50%)"
                # Adjustment: We 'buy back' at 50% of credit.
                # So we pay back 50%.
                # Net PnL = Credit - Payback = 50%.
                
            if exit_signal:
                self._close_trade(trade, curr_price, reason, pnl)

    def _close_trade(self, trade, price, reason, forced_pnl=None):
        # Calculate PnL if not forced (Expiration)
        pnl = forced_pnl
        
        if pnl is None: # Expiration Logic
            if "Put" in trade['type']:
                if price >= trade['strike_short']:
                     pnl = trade['credit']
                elif price <= trade['strike_long']:
                     pnl = - (trade['margin_locked'] - trade['credit']) 
                     # Wait. Logic:
                     # Cash Flow: +Credit (Entry), -Margin (Entry).
                     # Exit: if win, +Margin. (Net = Credit).
                     # if loss (max), +0. (Net = Credit - Margin).
                     # Correct.
                else: 
                     # Partial
                     loss = (trade['strike_short'] - price) * 100 * trade['contracts']
                     pnl = trade['credit'] - loss
        
        # Accounting
        # If Win: Return Margin.
        # If Loss: Return Margin - Realized Loss.
        # But PnL is Net.
        # So Capital += Margin + PnL - Credit??
        # Simpler: Capital += Margin (unlock) + Realized_PnL_Adjustment?
        # Let's stick to Cash accounting.
        # Entry: Cap = Cap - Margin + Credit.
        # Exit: Cap = Cap + Margin - Cost_To_Close.
        # Cost to close = Credit - PnL.
        
        cost_to_close = trade['credit'] - pnl
        self.capital += trade['margin_locked'] # Unlock collateral
        self.capital -= cost_to_close # Pay to close
        
        self.trade_log.append({
            "ticker": trade['ticker'],
            "contracts": trade['contracts'],
            "entry": trade['entry_date'],
            "type": trade['type'],
            "strikes": f"{trade['strike_short']:.2f}/{trade['strike_long']:.2f}",
            "reason": reason,
            "pnl": pnl,
            "return_pct": pnl / trade['margin_locked'] if trade['margin_locked'] else 0
        })
        self.portfolio.remove(trade)

    def _generate_report(self):
        df_equity = pd.DataFrame(self.equity_curve)
        df_trades = pd.DataFrame(self.trade_log)

        return {
            "stats": self._calculate_advanced_stats(df_trades, df_equity),
            "trades": df_trades,
            "equity": df_equity
        }

    def _calculate_advanced_stats(self, trades, equity):
        stats = {
            "Final Capital": self.capital,
            "Total Return": (self.capital - self.initial_capital) / self.initial_capital,
            "Win Rate": 0,
            "Trades": len(trades),
            "Max Drawdown": 0,
            "Sharpe Ratio": 0,
            "Profit Factor": 0,
            "Avg Win": 0,
            "Avg Loss": 0,
            "Expectancy": 0
        }
        
        if trades.empty: return stats

        # 1. Win Rate & Averages
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] <= 0]
        stats['Win Rate'] = len(wins) / len(trades)
        stats['Avg Win'] = wins['pnl'].mean() if not wins.empty else 0
        stats['Avg Loss'] = losses['pnl'].mean() if not losses.empty else 0
        
        # 2. Profit Factor
        gross_win = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())
        stats['Profit Factor'] = gross_win / gross_loss if gross_loss > 0 else 999
        
        # 3. Expectancy (Avg Trade)
        stats['Expectancy'] = trades['pnl'].mean()

        # 4. Max Drawdown
        if not equity.empty:
            eq_curve = equity['equity']
            peak = eq_curve.cummax()
            drawdown = (eq_curve - peak) / peak
            stats['Max Drawdown'] = drawdown.min()

        # 5. Sharpe Ratio (Simplified daily)
        # Assuming Rfr = 0 for simplicity (excess return)
        if not equity.empty:
            equity['returns'] = equity['equity'].pct_change()
            mean_ret = equity['returns'].mean()
            std_ret = equity['returns'].std()
            if std_ret > 0:
                # Annualize (Root 252)
                stats['Sharpe Ratio'] = (mean_ret / std_ret) * np.sqrt(252)
        
        return stats
