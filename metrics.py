"""
metrics.py — lightweight telemetry hooks for zeroclaw-copilot-models.

Records per-model latency, error counts, and token usage.  Intentionally
dependency-free so it can run in any environment.  Plug in your own exporter
by passing a ``reporter`` callable to :class:`MetricsCollector`.

Usage
-----
    from metrics import MetricsCollector

    metrics = MetricsCollector()

    with metrics.record("gpt-4o") as rec:
        # call the model here …
        rec.tokens(input=512, output=128)

    print(metrics.summary())
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Dict, Generator, List, Optional


class ModelMetrics:
    """Accumulated statistics for a single model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.call_count: int = 0
        self.error_count: int = 0
        self.total_latency_ms: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_latency_ms / self.call_count

    @property
    def error_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.error_count / self.call_count

    def to_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "calls": self.call_count,
            "errors": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"ModelMetrics({self.to_dict()})"


class _CallRecorder:
    """Context-manager helper returned by :meth:`MetricsCollector.record`."""

    def __init__(self, metrics_obj: ModelMetrics) -> None:
        self._m = metrics_obj
        self._start: float = 0.0
        self._errored: bool = False
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    def tokens(self, *, input: int = 0, output: int = 0) -> None:  # noqa: A002
        """Record token counts for this call."""
        self._input_tokens += input
        self._output_tokens += output

    def mark_error(self) -> None:
        """Explicitly mark this call as an error (e.g. from a caught exception)."""
        self._errored = True

    # Internal hooks used by the context manager.
    def _begin(self) -> None:
        self._start = time.monotonic()

    def _end(self) -> None:
        elapsed_ms = (time.monotonic() - self._start) * 1000
        self._m.call_count += 1
        self._m.total_latency_ms += elapsed_ms
        self._m.total_input_tokens += self._input_tokens
        self._m.total_output_tokens += self._output_tokens
        if self._errored:
            self._m.error_count += 1


class MetricsCollector:
    """Collects call-level telemetry for all models.

    Parameters
    ----------
    reporter:
        Optional callable ``(metrics: ModelMetrics) -> None`` invoked after
        every recorded call.  Use this to forward metrics to an external
        system (Prometheus, Datadog, etc.).
    """

    def __init__(self, reporter: Optional[Callable[[ModelMetrics], None]] = None) -> None:
        self._data: Dict[str, ModelMetrics] = {}
        self._reporter = reporter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def record(self, model_name: str) -> Generator[_CallRecorder, None, None]:
        """Context manager that times a model call and accumulates stats.

        Example::

            with metrics.record("gpt-4o") as rec:
                response = call_model(...)
                rec.tokens(input=response.usage.input, output=response.usage.output)
        """
        entry = self._data.setdefault(model_name, ModelMetrics(model_name))
        recorder = _CallRecorder(entry)
        recorder._begin()
        try:
            yield recorder
        except Exception:
            recorder.mark_error()
            raise
        finally:
            recorder._end()
            if self._reporter is not None:
                self._reporter(entry)

    def get(self, model_name: str) -> Optional[ModelMetrics]:
        """Return accumulated metrics for *model_name*, or *None*."""
        return self._data.get(model_name)

    def summary(self) -> List[Dict]:
        """Return a list of per-model metric dicts sorted by model name."""
        return [m.to_dict() for m in sorted(self._data.values(), key=lambda m: m.model_name)]

    def reset(self) -> None:
        """Clear all accumulated metrics."""
        self._data.clear()
