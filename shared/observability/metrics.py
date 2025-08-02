"""Metrics collection compatible with analytics-backend-monorepo."""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from shared.config import CausalInferenceConfig


class CausalInferenceMetrics:
    """Metrics collection for causal inference operations."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()

        # Computation metrics
        self.computation_duration = Histogram(
            "causal_inference_computation_duration_seconds",
            "Duration of causal inference computations",
            ["method", "outcome"],
            registry=self.registry,
        )

        self.computation_count = Counter(
            "causal_inference_computations_total",
            "Total number of causal inference computations",
            ["method", "status"],
            registry=self.registry,
        )

        self.sample_size_gauge = Gauge(
            "causal_inference_sample_size",
            "Sample size of current computation",
            ["method"],
            registry=self.registry,
        )

        # API metrics
        self.api_requests = Counter(
            "causal_inference_api_requests_total",
            "Total API requests",
            ["endpoint", "method", "status"],
            registry=self.registry,
        )

        self.api_duration = Histogram(
            "causal_inference_api_duration_seconds",
            "API request duration",
            ["endpoint", "method"],
            registry=self.registry,
        )

        # Error metrics
        self.errors = Counter(
            "causal_inference_errors_total",
            "Total errors",
            ["error_type", "component"],
            registry=self.registry,
        )

    def record_computation(
        self, method: str, duration: float, status: str, sample_size: int
    ) -> None:
        """Record a causal inference computation."""
        self.computation_duration.labels(method=method, outcome=status).observe(
            duration
        )
        self.computation_count.labels(method=method, status=status).inc()
        self.sample_size_gauge.labels(method=method).set(sample_size)

    def record_api_request(
        self, endpoint: str, method: str, status: str, duration: float
    ) -> None:
        """Record an API request."""
        self.api_requests.labels(endpoint=endpoint, method=method, status=status).inc()
        self.api_duration.labels(endpoint=endpoint, method=method).observe(duration)

    def record_error(self, error_type: str, component: str) -> None:
        """Record an error."""
        self.errors.labels(error_type=error_type, component=component).inc()


# Global metrics instance
_metrics: CausalInferenceMetrics | None = None


def get_metrics() -> CausalInferenceMetrics:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = CausalInferenceMetrics()
    return _metrics


def setup_metrics(config: CausalInferenceConfig | None = None) -> None:
    """Set up metrics collection."""
    if config is None:
        config = CausalInferenceConfig()

    if config.enable_metrics:
        # Start Prometheus metrics server
        start_http_server(config.metrics_port)

        # Initialize metrics
        get_metrics()
