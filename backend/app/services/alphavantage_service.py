import requests
from backend.app.core.config import settings

class AlphaVantageService:
    def __init__(self):
        self.api_key = settings.ALPHAVANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"

    def get_sentiment(self, symbol: str):
        """Fetch news sentiment analysis."""
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": self.api_key,
            "limit": 10
        }
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # Check for Rate Limit / API Error
            if "Information" in data:
                 return {"score": 0, "label": "API Error", "news_count": 0}
            
            # Basic sentiment aggregation
            if "feed" in data:
                sentiment_scores = [float(item.get("overall_sentiment_score", 0)) for item in data["feed"]]
                avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                
                # Honest Return
                if avg_score == 0:
                    label = "Neutral"
                else:
                    label = "Neutral"
                    if avg_score > 0.15: label = "Bullish"
                    if avg_score > 0.35: label = "Very Bullish"
                    if avg_score < -0.15: label = "Bearish"
                    if avg_score < -0.35: label = "Very Bearish"
                
                return {"score": avg_score, "label": label, "news_count": len(data["feed"])}
            
            # Fallback if no feed
            return {"score": 0, "label": "No News", "news_count": 0}
            
        except Exception as e:
            print(f"Error fetching sentiment for {symbol}: {e}")
            return {"score": 0, "label": "API Error", "news_count": 0}

alphavantage_service = AlphaVantageService()
