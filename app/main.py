"""
Juan365 Google Ads Dashboard
FastAPI Backend with T+1 Auto-Sync
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import router
from app.routers.api import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting Juan365 Dashboard...")
    logger.info(f"Sheet ID: {os.getenv('GOOGLE_SHEET_ID', 'Not set')}")
    logger.info(f"Timezone: {os.getenv('TIMEZONE', 'Asia/Manila')}")

    # Start the scheduler for T+1 daily sync
    from app.scheduler import start_scheduler, stop_scheduler, get_next_sync_time
    start_scheduler()
    logger.info(f"Next scheduled sync: {get_next_sync_time()}")

    yield

    # Stop scheduler on shutdown
    stop_scheduler()
    logger.info("Shutting down Juan365 Dashboard...")


# Create FastAPI app
app = FastAPI(
    title="Juan365 Google Ads Dashboard",
    description="Statistical Analysis Dashboard with T+1 Auto-Sync",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
async def root():
    """Serve the main dashboard"""
    index_path = os.path.join(static_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Juan365 Dashboard API",
        "docs": "/docs",
        "endpoints": {
            "metrics": "/api/metrics",
            "hourly": "/api/hourly",
            "statistics": "/api/statistics",
            "comparison": "/api/comparison",
            "tiers": "/api/tiers",
            "recommendations": "/api/recommendations",
            "sync_status": "/api/sync-status",
            "refresh": "/api/refresh",
            "all": "/api/all"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "juan365-dashboard"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
