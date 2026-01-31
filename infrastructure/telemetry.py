"""
Telemetry and Metrics for Glyphh Runtime.

Exports metrics to configured endpoint for monitoring.
Supports opt-out for privacy.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from infrastructure.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


class TelemetryCollector:
    """
    Collects and exports runtime telemetry.
    
    Metrics collected:
    - Model usage (queries per model)
    - Query patterns (search, fact_tree, predict)
    - Error rates
    - Response times
    - Resource usage
    """
    
    def __init__(self):
        self._enabled = settings.telemetry_enabled
        self._endpoint = settings.telemetry_endpoint
        self._metrics: List[MetricPoint] = []
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._export_task: Optional[asyncio.Task] = None
        self._export_interval = 60  # seconds
    
    def is_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self._enabled and self._endpoint is not None
    
    def increment(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        if not self._enabled:
            return
        
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value
    
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram observation."""
        if not self._enabled:
            return
        
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        if not self._enabled:
            return
        
        self._metrics.append(MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
        ))
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    async def start_export_loop(self) -> None:
        """Start the background export loop."""
        if not self.is_enabled():
            logger.info("Telemetry disabled, skipping export loop")
            return
        
        self._export_task = asyncio.create_task(self._export_loop())
        logger.info(f"Telemetry export started, interval={self._export_interval}s")
    
    async def stop_export_loop(self) -> None:
        """Stop the background export loop."""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
            self._export_task = None
    
    async def _export_loop(self) -> None:
        """Background loop for exporting metrics."""
        while True:
            try:
                await asyncio.sleep(self._export_interval)
                await self.export()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telemetry export failed: {e}")
    
    async def export(self) -> bool:
        """Export collected metrics to endpoint."""
        if not self.is_enabled():
            return False
        
        # Build metrics payload
        payload = self._build_payload()
        
        if not payload["metrics"]:
            return True  # Nothing to export
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._endpoint,
                    json=payload,
                    timeout=10.0,
                )
                response.raise_for_status()
            
            # Clear exported metrics
            self._metrics.clear()
            self._counters.clear()
            self._histograms.clear()
            
            logger.debug(f"Exported {len(payload['metrics'])} metrics")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to export telemetry: {e}")
            return False
    
    def _build_payload(self) -> Dict[str, Any]:
        """Build the metrics payload for export."""
        metrics = []
        
        # Add gauge metrics
        for point in self._metrics:
            metrics.append({
                "name": point.name,
                "type": "gauge",
                "value": point.value,
                "timestamp": point.timestamp.isoformat() + "Z",
                "labels": point.labels,
            })
        
        # Add counter metrics
        for key, value in self._counters.items():
            name, labels = self._parse_key(key)
            metrics.append({
                "name": name,
                "type": "counter",
                "value": value,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "labels": labels,
            })
        
        # Add histogram summaries
        for key, values in self._histograms.items():
            if not values:
                continue
            name, labels = self._parse_key(key)
            metrics.append({
                "name": f"{name}_count",
                "type": "counter",
                "value": len(values),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "labels": labels,
            })
            metrics.append({
                "name": f"{name}_sum",
                "type": "counter",
                "value": sum(values),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "labels": labels,
            })
        
        return {
            "runtime_version": "1.0.0",
            "deployment_mode": settings.deployment_mode,
            "metrics": metrics,
        }
    
    def _parse_key(self, key: str) -> tuple:
        """Parse a metric key back into name and labels."""
        if "{" not in key:
            return key, {}
        
        name = key[:key.index("{")]
        label_str = key[key.index("{") + 1:key.index("}")]
        labels = {}
        for pair in label_str.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
        return name, labels
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current telemetry statistics."""
        return {
            "enabled": self._enabled,
            "endpoint": self._endpoint,
            "pending_metrics": len(self._metrics),
            "pending_counters": len(self._counters),
            "pending_histograms": len(self._histograms),
        }


# Global telemetry instance
_telemetry: Optional[TelemetryCollector] = None


def get_telemetry() -> TelemetryCollector:
    """Get the global telemetry collector."""
    global _telemetry
    if _telemetry is None:
        _telemetry = TelemetryCollector()
    return _telemetry
