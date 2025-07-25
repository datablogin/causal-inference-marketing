"""Attribution analysis endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared.observability import get_logger, get_metrics

logger = get_logger(__name__)
router = APIRouter()


class AttributionRequest(BaseModel):
    """Request model for attribution analysis."""

    data: list[dict[str, Any]] = Field(description="Marketing touchpoint data")
    method: str = Field(
        default="doubly_robust", description="Attribution method to use"
    )
    treatment_col: str = Field(
        default="channel", description="Column name for treatment variable"
    )
    outcome_col: str = Field(
        default="conversion", description="Column name for outcome variable"
    )
    confidence_level: float = Field(
        default=0.95, description="Confidence level for estimates"
    )


class AttributionResponse(BaseModel):
    """Response model for attribution analysis."""

    method: str
    results: dict[str, Any]
    confidence_level: float
    sample_size: int
    computation_time: float


@router.post("/analyze", response_model=AttributionResponse)
async def analyze_attribution(request: AttributionRequest) -> AttributionResponse:
    """Perform attribution analysis."""
    import time

    start_time = time.time()

    try:
        logger.info(
            "Starting attribution analysis",
            method=request.method,
            sample_size=len(request.data),
        )

        # Placeholder implementation - replace with actual causal inference logic
        results = {
            "attribution_weights": {"email": 0.25, "social": 0.35, "search": 0.40},
            "confidence_intervals": {
                "email": [0.20, 0.30],
                "social": [0.30, 0.40],
                "search": [0.35, 0.45],
            },
            "p_values": {"email": 0.001, "social": 0.0001, "search": 0.00001},
        }

        computation_time = time.time() - start_time

        # Record metrics
        metrics = get_metrics()
        metrics.record_computation(
            method=request.method,
            duration=computation_time,
            status="success",
            sample_size=len(request.data),
        )

        logger.info(
            "Attribution analysis completed",
            method=request.method,
            computation_time=computation_time,
        )

        return AttributionResponse(
            method=request.method,
            results=results,
            confidence_level=request.confidence_level,
            sample_size=len(request.data),
            computation_time=computation_time,
        )

    except Exception as e:
        computation_time = time.time() - start_time

        # Record error metrics
        metrics = get_metrics()
        metrics.record_computation(
            method=request.method,
            duration=computation_time,
            status="error",
            sample_size=len(request.data),
        )
        metrics.record_error(error_type=type(e).__name__, component="attribution")

        logger.error("Attribution analysis failed", method=request.method, error=str(e))

        raise HTTPException(
            status_code=500, detail=f"Attribution analysis failed: {str(e)}"
        )


@router.get("/methods")
async def get_attribution_methods() -> dict[str, list[str]]:
    """Get available attribution methods."""
    return {
        "methods": [
            "first_touch",
            "last_touch",
            "linear",
            "time_decay",
            "position_based",
            "doubly_robust",
            "ipw",
            "g_computation",
        ]
    }
