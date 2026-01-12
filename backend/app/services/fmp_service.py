import requests
import pandas as pd
from backend.app.core.config import settings
from datetime import datetime, timedelta

class FMPService:
    def __init__(self):
        self.api_key = settings.FMP_API_KEY
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_quote(self, symbol: str):
        """Fetch real-time quote."""
        url = f"{self.base_url}/quote/{symbol}?apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data[0] if data else None
        return None

    def get_historical_data(self, symbol: str, days: int = 100):
        """Fetch daily candles for Trend/Support/Resistance analysis."""
        url = f"{self.base_url}/historical-price-full/{symbol}?timeseries={days}&apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "historical" in data:
                df = pd.DataFrame(data["historical"])
                # Ensure correct sorting (oldest to newest) for analysis
                return df.iloc[::-1].reset_index(drop=True)
        return pd.DataFrame()

    def get_institutional_holders(self, symbol: str):
        """Fetch institutional holders to gauge 'Smart Money' interest."""
        url = f"{self.base_url}/institutional-holder/{symbol}?apikey={self.api_key}"
        response = requests.get(url)
        return response.json() if response.status_code == 200 else []

    def get_earnings_calendar(self, symbol: str):
        """Fetch upcoming earnings to avoid binary events."""
        # Provides earnings for the next few months
        url = f"{self.base_url}/historical/earning_calendar/{symbol}?limit=10&apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Filter for future earnings only
            today = datetime.now().date()
            future_earnings = [e for e in data if datetime.strptime(e['date'], "%Y-%m-%d").date() >= today]
            return future_earnings
        return []

fmp_service = FMPService()
