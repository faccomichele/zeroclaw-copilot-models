"""
model_selector.py — static model selection and fallback for zeroclaw-copilot-models.

Usage
-----
    from model_selector import ModelSelector

    selector = ModelSelector()
    model = selector.select()          # best enabled model by priority
    model = selector.select("gpt-4o") # specific preferred model, falls back if unavailable
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "models.json")


class ModelConfig:
    """Represents a single model entry from the config file."""

    def __init__(self, data: Dict) -> None:
        self.name: str = data["name"]
        self.provider: str = data["provider"]
        self.priority: int = data["priority"]
        self.enabled: bool = data.get("enabled", True)
        self.cost_per_1k_input_tokens: float = data.get("cost_per_1k_input_tokens", 0.0)
        self.cost_per_1k_output_tokens: float = data.get("cost_per_1k_output_tokens", 0.0)
        self.description: str = data.get("description", "")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ModelConfig(name={self.name!r}, provider={self.provider!r}, "
            f"priority={self.priority}, enabled={self.enabled})"
        )


class ModelSelector:
    """Selects the best available model with priority-ordered fallback.

    Parameters
    ----------
    config_path:
        Path to the JSON config file.  Defaults to ``models.json`` in the
        same directory as this module.
    health_check:
        Optional callable ``(model_name: str) -> bool`` that verifies a model
        is reachable at runtime.  When *None* (the default) the selector only
        uses the ``enabled`` flag from the config.
    on_fallback:
        Optional callable ``(skipped: str, reason: str) -> None`` invoked each
        time a model is skipped during selection, useful for logging / metrics.
    """

    def __init__(
        self,
        config_path: str = _DEFAULT_CONFIG,
        health_check: Optional[Callable[[str], bool]] = None,
        on_fallback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._models: List[ModelConfig] = self._load(config_path)
        self._health_check = health_check
        self._on_fallback = on_fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, preferred: Optional[str] = None) -> ModelConfig:
        """Return the best available model.

        If *preferred* is given the selector first tries that model, then falls
        back through the remaining candidates in priority order.

        Raises
        ------
        RuntimeError
            When no enabled, healthy model can be found.
        """
        candidates = sorted(self._models, key=lambda m: m.priority)

        # Build the ordered list: preferred first, then the rest.
        if preferred:
            ordered = [m for m in candidates if m.name == preferred] + [
                m for m in candidates if m.name != preferred
            ]
        else:
            ordered = candidates

        for model in ordered:
            reason = self._unavailable_reason(model)
            if reason is None:
                return model
            self._emit_fallback(model.name, reason)

        raise RuntimeError(
            "No available model found. "
            "Check that at least one model is enabled in models.json."
        )

    def list_enabled(self) -> List[ModelConfig]:
        """Return all enabled models sorted by priority."""
        return sorted(
            [m for m in self._models if m.enabled], key=lambda m: m.priority
        )

    def list_all(self) -> List[ModelConfig]:
        """Return all models sorted by priority."""
        return sorted(self._models, key=lambda m: m.priority)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(config_path: str) -> List[ModelConfig]:
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return [ModelConfig(entry) for entry in data["models"]]

    def _unavailable_reason(self, model: ModelConfig) -> Optional[str]:
        """Return a reason string if the model is unavailable, else None."""
        if not model.enabled:
            return "disabled in config"
        if self._health_check is not None:
            start = time.monotonic()
            healthy = self._health_check(model.name)
            latency = time.monotonic() - start
            if not healthy:
                return f"health check failed (took {latency:.3f}s)"
        return None

    def _emit_fallback(self, model_name: str, reason: str) -> None:
        if self._on_fallback is not None:
            self._on_fallback(model_name, reason)
