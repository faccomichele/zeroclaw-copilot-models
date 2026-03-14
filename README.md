# zeroclaw-copilot-models

Handling GitHub Copilot models for ZeroClaw — static model registry, priority-ordered selection, automatic fallback, and lightweight telemetry.

---

## Overview

Live model discovery is not yet available for ZeroClaw.  
Instead, this repo ships:

| File | Purpose |
|------|---------|
| `models.json` | Static registry of all available models |
| `model_selector.py` | Picks the best model; falls back in priority order |
| `metrics.py` | Records latency, errors, and token usage per model |

---

## `models.json` — the model registry

Each entry in the `"models"` array describes one model:

```json
{
  "name": "gpt-4o",
  "provider": "openai",
  "priority": 1,
  "enabled": true,
  "cost_per_1k_input_tokens": 0.005,
  "cost_per_1k_output_tokens": 0.015,
  "description": "OpenAI GPT-4o — high capability, multimodal"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Unique model identifier |
| `provider` | string | ✅ | Provider name (e.g. `openai`, `anthropic`, `google`) |
| `priority` | integer | ✅ | Lower number = higher priority during selection |
| `enabled` | boolean | ✅ | Set to `false` to exclude from selection without deleting the entry |
| `cost_per_1k_input_tokens` | float | — | USD cost per 1 000 input tokens (informational) |
| `cost_per_1k_output_tokens` | float | — | USD cost per 1 000 output tokens (informational) |
| `description` | string | — | Human-readable notes |

### Adding a new model

1. Open `models.json`.
2. Append a new object to the `"models"` array using the schema above.
3. Set `"priority"` so it fits in the fallback order you want.
4. Set `"enabled": true` when the integration is ready; use `false` to pre-register it without activating it.

---

## `model_selector.py` — selection and fallback

```python
from model_selector import ModelSelector

selector = ModelSelector()

# Pick the highest-priority enabled model
model = selector.select()
print(model.name, model.provider)

# Try a specific model first, fall back automatically if unavailable
model = selector.select("claude-3-5-sonnet")

# List all enabled models in priority order
for m in selector.list_enabled():
    print(m.priority, m.name)
```

### Optional: runtime health checks

Pass a `health_check` callable that returns `True` when a model is reachable.  
The selector will skip any model whose health check fails and try the next one:

```python
import httpx

def is_healthy(model_name: str) -> bool:
    try:
        r = httpx.get(f"https://api.example.com/health/{model_name}", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

selector = ModelSelector(health_check=is_healthy)
model = selector.select()
```

### Optional: fallback notifications

```python
import logging

def log_fallback(skipped_model: str, reason: str) -> None:
    logging.warning("Skipping %s: %s", skipped_model, reason)

selector = ModelSelector(on_fallback=log_fallback)
```

---

## `metrics.py` — telemetry hooks

```python
from metrics import MetricsCollector

metrics = MetricsCollector()

with metrics.record("gpt-4o") as rec:
    response = call_model(...)                        # your model call here
    rec.tokens(input=response.usage.prompt_tokens,
               output=response.usage.completion_tokens)

# If the block raises, the error is automatically counted.

print(metrics.summary())
# [{'model': 'gpt-4o', 'calls': 1, 'errors': 0, 'error_rate': 0.0,
#   'avg_latency_ms': 312.4, 'total_input_tokens': 512, 'total_output_tokens': 128}]
```

### Forwarding metrics to an external system

```python
import datadog  # example

def send_to_datadog(m):
    datadog.statsd.gauge("model.latency_ms", m.avg_latency_ms, tags=[f"model:{m.model_name}"])
    datadog.statsd.gauge("model.error_rate", m.error_rate,    tags=[f"model:{m.model_name}"])

metrics = MetricsCollector(reporter=send_to_datadog)
```

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```
