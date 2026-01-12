import yfinance as yf
import pandas as pd
from datetime import datetime

class YFinanceService:
    def get_quote(self, symbol: str):
        """Fetch real-time quote."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            price = info.last_price
            prev_close = info.previous_close
            
            change_percent = 0.0
            if prev_close:
                change_percent = ((price - prev_close) / prev_close) * 100
                
            return {"price": price, "change_percent": change_percent}
        except Exception as e:
            print(f"YF Quote Error {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, days: int = 100):
        """Fetch daily candles for Trend/Support/Resistance analysis."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")
            return df.reset_index()
        except Exception as e:
            print(f"YF History Error {symbol}: {e}")
            return pd.DataFrame()

    def get_earnings_calendar(self, symbol: str):
        """Fetch upcoming earnings."""
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            # yfinance calendar structure varies. 
            # Usually returns a dict with 'Earnings Date' or 'Earnings High' etc
            if isinstance(calendar, dict):
                 if 'Earnings Date' in calendar:
                     dates = calendar['Earnings Date']
                     # Return as list of dicts to match FMP format partially
                     return [{"date": d.strftime("%Y-%m-%d")} for d in dates]
            return []
        except Exception as e:
            # print(f"YF Earnings Error {symbol}: {e}") # expected for some tickers
            return []

    def get_option_chain(self, symbol: str, expiration_date: str = None):
        """
        Fetch option chain from Yahoo Finance.
        Returns a list of contracts (calls and puts mixed or separate).
        """
        ticker = yf.Ticker(symbol)
        try:
            if not expiration_date:
                # Pick an expiry ~30-45 days out
                expirations = ticker.options
                if not expirations:
                    return []
                
                # Logic to find best expiry
                target_date = None
                for date in expirations:
                    # 'date' is string 'YYYY-MM-DD'
                    days_to_exp = (datetime.strptime(date, "%Y-%m-%d") - datetime.now()).days
                    if 25 <= days_to_exp <= 50:
                        target_date = date
                        break
                
                if not target_date:
                    target_date = expirations[0] # Fallback
                
                expiration_date = target_date

            # Fetch chain
            chain = ticker.option_chain(expiration_date)
            calls = chain.calls
            puts = chain.puts
            
            # Add type column
            calls['type'] = 'call'
            puts['type'] = 'put'
            
            # Combine
            full_chain = pd.concat([calls, puts])
            full_chain['expiration'] = expiration_date
            
            # Normalize columns to standard set
            # YF cols: contractSymbol, lastTradeDate, strike, lastPrice, bid, ask, change, percentChange, volume, openInterest, impliedVolatility, inTheMoney, contractSize, currency
            
            return full_chain.to_dict('records')

        except Exception as e:
            print(f"Error fetching YF options for {symbol}: {e}")
            return []

yfinance_service = YFinanceService()
