from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

# CORS (Allow all for development flexibility, restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Options Scanner AI Backend is Running", "market_status": "Live"}

from app.api.endpoints import router as scanner_router
app.include_router(scanner_router, prefix="/api/v1", tags=["scanner"])

