from datetime import datetime, timedelta
import math
from backend.app.services.fmp_service import fmp_service
from backend.app.services.eodhd_service import eodhd_service
from backend.app.services.yfinance_service import yfinance_service
from backend.app.services.alphavantage_service import alphavantage_service
from backend.app.strategies.analysis import technical_analysis
from backend.app.strategies.risk import risk_manager
from backend.app.core.config import settings

class OptionsScanner:
    
    def __init__(self):
        self.risk_free_rate = 0.045

    def norm_cdf(self, x):
        """Cumulative distribution function for the standard normal distribution."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def norm_pdf(self, x):
        """Probability density function for the standard normal distribution."""
        return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)

    def calculate_greeks(self, S, K, T, r, sigma, type='call'):
        """
        Calculate Delta, Gamma, Theta, Vega using standard math.
        """
        if T <= 0 or sigma <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
            
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        pdf_d1 = self.norm_pdf(d1)
        cdf_d1 = self.norm_cdf(d1)
        cdf_d2 = self.norm_cdf(d2) # For Call rho/prob
        
        if type == 'call':
            delta = cdf_d1
            theta = (- (S * sigma * pdf_d1) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * self.norm_cdf(d2)) / 365
        else: # put
            delta = cdf_d1 - 1
            theta = (- (S * sigma * pdf_d1) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * self.norm_cdf(-d2)) / 365

        gamma = pdf_d1 / (S * sigma * math.sqrt(T))
        vega = S * math.sqrt(T) * pdf_d1 / 100 

        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


    def scan_opportunities(self):
        results = []
        
        for ticker in settings.WATCHLIST:
            # 1. Get Stock Data (YF)
            quote = yfinance_service.get_quote(ticker)
            if not quote:
                continue
            
            price = quote.get('price')
            
            # 2. Earnings Check (YF)
            earnings = yfinance_service.get_earnings_calendar(ticker)
            has_earnings_soon = False
            earnings_date = "None"
            limit_date = (datetime.now() + timedelta(days=45)).date()
            
            if earnings:
                # YF usually returns a list of datetimes or date strings depending on version
                # Simplified check assuming list of dicts from service
                for e in earnings:
                    try:
                        e_date = datetime.strptime(e['date'], "%Y-%m-%d").date()
                        if (datetime.now().date() <= e_date <= limit_date):
                            has_earnings_soon = True
                            earnings_date = e['date']
                            break
                    except:
                        pass
            
            # 3. Technical Analysis (YF History)
            hist_df = yfinance_service.get_historical_data(ticker, days=100)
            if hist_df.empty:
                 continue
                 
            trend = technical_analysis.identify_trend(hist_df)
            sr_levels = technical_analysis.find_support_resistance(hist_df)
            
            # 4. Sentiment
            sentiment = alphavantage_service.get_sentiment(ticker)
            
            # 5. Basic Logic (Fallback)
            strategy = "NEUTRAL"
            if trend == "Uptrend": 
                strategy = "BULLISH"
            elif trend == "Downtrend":
                strategy = "BEARISH"
            
            # Override if ranging
            if trend == "Ranging":
                strategy = "NEUTRAL"

            warning = "EARNINGS APPROACHING!" if has_earnings_soon else "Safe"

            # 6. Advanced Institutional Analysis (Confluence Engine)
            from backend.app.strategies.advanced_analysis import ConfluenceAI
            confluence_engine = ConfluenceAI(hist_df)
            advanced_signals = confluence_engine.analyze_confluence()
            
            # 6. Recommendation Construction
            # We need to pick strikes from the YF Option Chain
            chain = yfinance_service.get_option_chain(ticker)
            
            # If no chain, skip
            if not chain:
                continue
                
            # Logic: Use Advanced Signals if available, else Fallback to Basic Trend
            final_strategy = "Iron Condor (Neutral)"
            rec_strikes = {"put": sr_levels['major_support'], "call": sr_levels['major_resistance']}
            
            # Check Aggressive Signal first
            if advanced_signals['Aggressive']:
                sig = advanced_signals['Aggressive']
                final_strategy = f"⚡ {sig['Type']} (Aggressive)"
                rec_strikes['put'] = sig.get('Strike_Long', 0) # Simplified for display
                rec_strikes['call'] = sig.get('Strike_Short', 0)
                
            # Check Conservative Signal (Overrides Aggressive if present and safer?)
            # Actually, usually we want to show the strongest conviction.
            # Let's show Conservative if Aggressive is missing, or maybe simple UI text.
            if advanced_signals['Conservative']:
                 sig = advanced_signals['Conservative']
                 final_strategy = f"🛡️ {sig['Type']} (Conservative)"
                 rec_strikes['put'] = sig.get('Strike_Long', 0)
                 rec_strikes['call'] = sig.get('Strike_Short', 0)
            
            if advanced_signals['Notes']:
                 final_strategy += f" | {advanced_signals['Notes'][0]}"

            rec = {
                "ticker": ticker,
                "spot_price": price,
                "strategy_bias": strategy, # Basic Trend Bias
                "recommended_strategy": final_strategy,
                "trend": trend,
                "support": sr_levels['major_support'],
                "resistance": sr_levels['major_resistance'],
                "earnings_alert": warning,
                "earnings_date": earnings_date,
                "sentiment_score": sentiment.get('score', 0),
                "sentiment_label": sentiment.get('label', 'N/A'),
                "advanced_data": advanced_signals # Store full object for Deep Dive
            }
            results.append(rec)

        return results

options_scanner = OptionsScanner()
