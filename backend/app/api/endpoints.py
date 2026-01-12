from fastapi import APIRouter, HTTPException
from app.strategies.scanner import options_scanner

router = APIRouter()

@router.get("/scan")
def run_scan():
    """
    Run the full market scanner on the preset watchlist.
    Returns optimal strategy recommendations.
    """
    try:
        results = options_scanner.scan_opportunities()
        return {"status": "success", "count": len(results), "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
def get_config():
    """Return current configuration (Watchlist, etc)."""
    from app.core.config import settings
    return {"watchlist": settings.WATCHLIST}
