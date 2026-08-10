"""可观测性：结构化日志、request-id、Prometheus 指标。"""
from __future__ import annotations

import contextvars
import logging
import sys
import time

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] [req=%(request_id)s] %(message)s"
        )
    )
    handler.addFilter(_RequestIdFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


logger = logging.getLogger("to3d")


# --- Prometheus 指标（prometheus_client 缺失时降级为 no-op）------------------
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    GEN_TOTAL = Counter("to3d_generation_total", "生成任务总数", ["status"])
    GEN_DURATION = Histogram("to3d_generation_duration_seconds", "生成任务耗时")
    GEN_IN_FLIGHT = Gauge("to3d_generation_in_flight", "进行中的生成任务数")
    HTTP_REQUESTS = Counter("to3d_http_requests_total", "HTTP 请求数", ["method", "path", "code"])
    _PROM = True

    def metrics_response():
        return generate_latest(), CONTENT_TYPE_LATEST

except Exception:  # noqa: BLE001 - 无 prometheus_client 时降级
    _PROM = False

    class _Noop:
        def labels(self, *a, **k):
            return self

        def inc(self, *a, **k):
            pass

        def dec(self, *a, **k):
            pass

        def observe(self, *a, **k):
            pass

    GEN_TOTAL = GEN_DURATION = GEN_IN_FLIGHT = HTTP_REQUESTS = _Noop()

    def metrics_response():
        return b"# prometheus_client not installed\n", "text/plain"


class track_generation:
    """上下文管理器：统计生成任务耗时/在飞/结果。"""

    def __init__(self) -> None:
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.time()
        GEN_IN_FLIGHT.inc()
        return self

    def __exit__(self, exc_type, exc, tb):
        GEN_IN_FLIGHT.dec()
        GEN_DURATION.observe(time.time() - self._t0)
        GEN_TOTAL.labels(status="failed" if exc_type else "done").inc()
        return False
