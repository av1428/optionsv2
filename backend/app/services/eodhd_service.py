import requests
import pandas as pd
from backend.app.core.config import settings

class EODHDService:
    def __init__(self):
        self.api_key = settings.EODHD_API_KEY
        # User specified endpoint structure for UnicornBay/EODHD options
        self.base_url = "https://eodhd.com/api/mp/unicornbay/options/contracts"

    def get_option_chain(self, symbol: str, expiration_date: str = None, contract_type: str = None):
        """
        Fetch option contracts. 
        Note: The user provided endpoint supports filtering. 
        Args:
            symbol: Underlying ticker (e.g. AAPL)
            expiration_date: Specific expiration (YYYY-MM-DD)
            contract_type: 'call' or 'put'
        """
        params = {
            "api_token": self.api_key,
            "filter[underlying_symbol]": symbol,
            "sort": "-"  # Default sort
        }
        
        if expiration_date:
            params["filter[exp_date_eq]"] = expiration_date
            
        if contract_type:
            params["filter[type]"] = contract_type

        # We want important fields for analysis (Greeks might need separate calculation if not provided here)
        # The user example showed: fields[options-contracts]=contract,bid_date,open,high,low,last
        # We likely need 'strike', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega' if available.
        # Assuming standard fields plus Greeks if this endpoint supports them.
        # For now, fetching broad data to see what we get.
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # The structure might be nested depending on the MP response
            # Returning raw list for now, Scanner will parse into DataFrame
            return data.get("data", []) # Usually API responses have a 'data' key or are a direct list
        except Exception as e:
            print(f"Error fetching EODHD options for {symbol}: {e}")
            return []

eodhd_service = EODHDService()
