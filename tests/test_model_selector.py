"""
tests/test_model_selector.py — unit tests for model_selector and metrics.
"""

import json
import os
import tempfile
import time

import pytest

from model_selector import ModelConfig, ModelSelector
from metrics import MetricsCollector, ModelMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(models: list, directory: str) -> str:
    path = os.path.join(directory, "models.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"models": models}, fh)
    return path


SAMPLE_MODELS = [
    {
        "name": "model-a",
        "provider": "openai",
        "priority": 1,
        "enabled": True,
        "cost_per_1k_input_tokens": 0.01,
        "cost_per_1k_output_tokens": 0.03,
        "description": "Top priority model",
    },
    {
        "name": "model-b",
        "provider": "anthropic",
        "priority": 2,
        "enabled": True,
        "cost_per_1k_input_tokens": 0.002,
        "cost_per_1k_output_tokens": 0.006,
        "description": "Second priority model",
    },
    {
        "name": "model-c",
        "provider": "google",
        "priority": 3,
        "enabled": False,
        "cost_per_1k_input_tokens": 0.001,
        "cost_per_1k_output_tokens": 0.002,
        "description": "Disabled model",
    },
]


# ---------------------------------------------------------------------------
# ModelConfig tests
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_basic_fields(self):
        cfg = ModelConfig(SAMPLE_MODELS[0])
        assert cfg.name == "model-a"
        assert cfg.provider == "openai"
        assert cfg.priority == 1
        assert cfg.enabled is True
        assert cfg.cost_per_1k_input_tokens == 0.01
        assert cfg.cost_per_1k_output_tokens == 0.03
        assert cfg.description == "Top priority model"

    def test_defaults(self):
        cfg = ModelConfig({"name": "x", "provider": "p", "priority": 99})
        assert cfg.enabled is True
        assert cfg.cost_per_1k_input_tokens == 0.0
        assert cfg.cost_per_1k_output_tokens == 0.0
        assert cfg.description == ""


# ---------------------------------------------------------------------------
# ModelSelector tests
# ---------------------------------------------------------------------------

class TestModelSelector:
    @pytest.fixture
    def config_path(self, tmp_path):
        return _write_config(SAMPLE_MODELS, str(tmp_path))

    def test_select_returns_highest_priority(self, config_path):
        sel = ModelSelector(config_path)
        model = sel.select()
        assert model.name == "model-a"

    def test_select_preferred_model(self, config_path):
        sel = ModelSelector(config_path)
        model = sel.select("model-b")
        assert model.name == "model-b"

    def test_select_preferred_falls_back_when_disabled(self, config_path):
        """Requesting a disabled model falls back to next enabled one."""
        sel = ModelSelector(config_path)
        model = sel.select("model-c")  # model-c is disabled
        assert model.name == "model-a"

    def test_select_falls_back_on_health_failure(self, config_path):
        """A failing health check causes fallback to the next model."""
        def health(name: str) -> bool:
            return name != "model-a"

        sel = ModelSelector(config_path, health_check=health)
        model = sel.select()
        assert model.name == "model-b"

    def test_on_fallback_called(self, config_path):
        skipped = []

        def health(name: str) -> bool:
            return name != "model-a"

        def on_fallback(name: str, reason: str) -> None:
            skipped.append((name, reason))

        sel = ModelSelector(config_path, health_check=health, on_fallback=on_fallback)
        sel.select()
        assert len(skipped) == 1
        assert skipped[0][0] == "model-a"
        assert "health check failed" in skipped[0][1]

    def test_raises_when_no_model_available(self, tmp_path):
        path = _write_config(
            [{"name": "x", "provider": "p", "priority": 1, "enabled": False}],
            str(tmp_path),
        )
        sel = ModelSelector(path)
        with pytest.raises(RuntimeError, match="No available model"):
            sel.select()

    def test_list_enabled(self, config_path):
        sel = ModelSelector(config_path)
        enabled = sel.list_enabled()
        names = [m.name for m in enabled]
        assert "model-c" not in names
        assert names == ["model-a", "model-b"]

    def test_list_all_includes_disabled(self, config_path):
        sel = ModelSelector(config_path)
        all_models = sel.list_all()
        assert len(all_models) == 3
        assert all_models[0].name == "model-a"

    def test_default_config_loads(self):
        """Smoke-test that the default models.json ships and is valid."""
        sel = ModelSelector()
        models = sel.list_all()
        assert len(models) > 0
        for m in models:
            assert m.name
            assert m.provider
            assert isinstance(m.priority, int)

    def test_priority_order_preserved(self, config_path):
        sel = ModelSelector(config_path)
        priorities = [m.priority for m in sel.list_all()]
        assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# MetricsCollector tests
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    def test_call_count_increments(self):
        mc = MetricsCollector()
        with mc.record("gpt-4o"):
            pass
        with mc.record("gpt-4o"):
            pass
        assert mc.get("gpt-4o").call_count == 2

    def test_token_accumulation(self):
        mc = MetricsCollector()
        with mc.record("gpt-4o") as rec:
            rec.tokens(input=100, output=50)
        m = mc.get("gpt-4o")
        assert m.total_input_tokens == 100
        assert m.total_output_tokens == 50

    def test_error_recorded_on_exception(self):
        mc = MetricsCollector()
        with pytest.raises(ValueError):
            with mc.record("gpt-4o"):
                raise ValueError("boom")
        m = mc.get("gpt-4o")
        assert m.error_count == 1
        assert m.call_count == 1

    def test_error_rate(self):
        mc = MetricsCollector()
        for _ in range(3):
            with mc.record("m"):
                pass
        with pytest.raises(RuntimeError):
            with mc.record("m"):
                raise RuntimeError("fail")
        assert mc.get("m").error_rate == pytest.approx(0.25)

    def test_latency_positive(self):
        mc = MetricsCollector()
        with mc.record("m"):
            time.sleep(0.01)
        assert mc.get("m").avg_latency_ms > 0

    def test_reporter_called(self):
        reports = []
        mc = MetricsCollector(reporter=lambda m: reports.append(m.model_name))
        with mc.record("gpt-4o"):
            pass
        assert reports == ["gpt-4o"]

    def test_summary_sorted_by_name(self):
        mc = MetricsCollector()
        with mc.record("z-model"):
            pass
        with mc.record("a-model"):
            pass
        names = [d["model"] for d in mc.summary()]
        assert names == ["a-model", "z-model"]

    def test_reset_clears_data(self):
        mc = MetricsCollector()
        with mc.record("m"):
            pass
        mc.reset()
        assert mc.get("m") is None
        assert mc.summary() == []

    def test_get_nonexistent_model_returns_none(self):
        mc = MetricsCollector()
        assert mc.get("nonexistent") is None

    def test_model_metrics_avg_latency_zero_calls(self):
        m = ModelMetrics("x")
        assert m.avg_latency_ms == 0.0
        assert m.error_rate == 0.0

    def test_to_dict_keys(self):
        mc = MetricsCollector()
        with mc.record("m") as rec:
            rec.tokens(input=10, output=5)
        d = mc.get("m").to_dict()
        expected_keys = {"model", "calls", "errors", "error_rate", "avg_latency_ms",
                         "total_input_tokens", "total_output_tokens"}
        assert set(d.keys()) == expected_keys
