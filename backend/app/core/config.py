import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Options Scanner AI"
    API_V1_STR: str = "/api/v1"
    
    # API KEYS
    FMP_API_KEY: str = os.getenv("FMP_API_KEY", "vuuZuDIkQjDIpFMp2jhw6KMcjBbAAsv2")
    EODHD_API_KEY: str = os.getenv("EODHD_API_KEY", "6959dd706eb1a2.14553323")
    ALPHAVANTAGE_API_KEY: str = os.getenv("ALPHAVANTAGE_API_KEY", "QEFCY51KJD83A034")

    # PRESET STOCKS (High Liquidity, High Interest)
    WATCHLIST: list = ["TSLA", "NVDA", "SPY", "QQQ", "IWM", "AMD", "AAPL", "MSFT", "PLTR", "AMZN"]
    
    class Config:
        case_sensitive = True

settings = Settings()
