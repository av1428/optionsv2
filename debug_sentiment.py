
import requests
from backend.app.core.config import settings

def debug_response(symbol):
    api_key = settings.ALPHAVANTAGE_API_KEY
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "apikey": api_key,
        "limit": 10
    }
    
    print(f"Testing for {symbol} with Key Ending in: ...{api_key[-4:]}")
    try:
        response = requests.get(url, params=params)
        data = response.json()
        print("--- RAW RESPONSE ---")
        print(str(data)[:500]) # First 500 chars
        print("--------------------")
        
        if "feed" in data:
            print(f"Feed Count: {len(data['feed'])}")
            scores = [float(i.get("overall_sentiment_score", 0)) for i in data['feed']]
            print(f"Scores: {scores}")
            print(f"Average: {sum(scores)/len(scores) if scores else 0}")
        else:
            print("NO FEED FOUND.")
            
    except Exception as e:
        print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    debug_response("TSLA")
