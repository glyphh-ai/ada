from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import Response

from ..core.config import Settings

logger = logging.getLogger("glyphh.runtime.monitoring")


def _setup_metrics(app, settings: Settings) -> None:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    request_count = Counter(
        "glyphh_runtime_http_requests_total",
        "HTTP request count",
        ["method", "path", "status"],
    )
    request_latency = Histogram(
        "glyphh_runtime_http_request_seconds",
        "HTTP request latency seconds",
        ["method", "path"],
        buckets=(0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        path = request.url.path
        request_count.labels(request.method, path, response.status_code).inc()
        request_latency.labels(request.method, path).observe(elapsed)
        return response

    @app.get(settings.metrics_path, include_in_schema=False)
    def metrics_endpoint() -> Response:
        payload = generate_latest()
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


def _setup_tracing(app, settings: Settings) -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("opentelemetry packages not installed; tracing disabled")
        return

    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)


def setup_monitoring(app, settings: Settings) -> None:
    if settings.metrics_enabled:
        _setup_metrics(app, settings)
    if settings.tracing_enabled:
        _setup_tracing(app, settings)
