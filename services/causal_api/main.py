"""Main FastAPI application for causal inference API."""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shared.config import CausalInferenceConfig, config_manager
from shared.observability import get_logger, get_metrics, setup_logging, setup_metrics
from shared.database import get_database_manager

from .routes import attribution, health

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    config = CausalInferenceConfig()
    config_manager.register_configuration("causal_inference", config)
    
    setup_logging(config)
    setup_metrics(config)
    
    logger.info("Starting Causal Inference API", version="0.1.0", environment=config.environment)
    
    yield
    
    # Shutdown
    db_manager = get_database_manager()
    await db_manager.close()
    logger.info("Causal Inference API shutdown complete")


app = FastAPI(
    title="Causal Inference API",
    description="API for causal inference computations in marketing applications",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect metrics for all requests."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Record metrics
        metrics = get_metrics()
        metrics.record_api_request(
            endpoint=request.url.path,
            method=request.method,
            status=str(response.status_code),
            duration=duration
        )
        
        return response
    except Exception as e:
        duration = time.time() - start_time
        
        # Record error metrics
        metrics = get_metrics() 
        metrics.record_api_request(
            endpoint=request.url.path,
            method=request.method,
            status="500",
            duration=duration
        )
        metrics.record_error(error_type=type(e).__name__, component="api")
        
        raise


# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(attribution.router, prefix="/api/v1/attribution", tags=["attribution"])


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    config = config_manager.get_configuration("causal_inference")
    return {
        "service": "Causal Inference API",
        "version": "0.1.0",
        "environment": config.environment.value if config else "unknown",
        "status": "healthy"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    metrics = get_metrics()
    metrics.record_error(error_type="HTTPException", component="api")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )