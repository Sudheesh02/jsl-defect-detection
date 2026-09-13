"""
Main FastAPI Application for Jindal Stainless AI-Powered Surface Defect Detection.
"""
from pathlib import Path
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes_meta import router as meta_router
from backend.api.routes_detect import router as detect_router, get_detector
from backend.config import BASE_DIR, RAW_IMAGES_DIR

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("steel_inspector")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Jindal Stainless Defect Inspector engines...")
    detector = get_detector()
    if detector.verifier and not detector.verifier._loaded:
        logger.info("Pre-loading ResNet-50 verification model for zero-jitter dual-stage inference...")
        detector.verifier.load_model()
    logger.info("Inspection engine is online and ready for production line throughput.")
    yield

app = FastAPI(
    title="Jindal Stainless - AI Surface Defect Detection System",
    description="Real-time stainless steel surface inspection, root cause diagnosis, and coil quality disposition platform.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers (both /api and direct paths for dual local & Vercel serverless compatibility)
app.include_router(meta_router, prefix="/api")
app.include_router(detect_router, prefix="/api")
app.include_router(meta_router)
app.include_router(detect_router)

# Mount Static Files
STATIC_DIR = BASE_DIR / "backend" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if RAW_IMAGES_DIR.exists():
    app.mount("/raw_images", StaticFiles(directory=str(RAW_IMAGES_DIR)), name="raw_images")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the interactive industrial demo dashboard."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>Jindal Stainless AI Defect Detection Platform</h1><p>Static dashboard loading...</p>")
