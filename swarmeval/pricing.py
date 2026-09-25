"""Model pricing (USD per 1M tokens) used to estimate API cost.

Unknown models produce a cost of ``None`` rather than a silent zero so that
reports can say "cost unknown" instead of "cost $0.00".  Supply your own
table via ``PricingTable(custom={...})`` for other providers.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple


# input $/1M, output $/1M -- Anthropic first-party API rates (cached 2026-06).
DEFAULT_PRICES: Dict[str, Tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-mythos-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


class PricingTable:
    def __init__(self, custom: Optional[Dict[str, Tuple[float, float]]] = None,
                 include_defaults: bool = True) -> None:
        self.prices: Dict[str, Tuple[float, float]] = dict(DEFAULT_PRICES) if include_defaults else {}
        if custom:
            self.prices.update(custom)

    def lookup(self, model: Optional[str]) -> Optional[Tuple[float, float]]:
        if not model:
            return None
        if model in self.prices:
            return self.prices[model]
        # Longest-prefix match handles dated or provider-prefixed ids.
        best = None
        for key, val in self.prices.items():
            if model.startswith(key) or key in model:
                if best is None or len(key) > len(best[0]):
                    best = (key, val)
        return best[1] if best else None

    def cost(self, model: Optional[str], input_tokens: int, output_tokens: int) -> Optional[float]:
        p = self.lookup(model)
        if p is None:
            return None
        return (input_tokens * p[0] + output_tokens * p[1]) / 1_000_000.0


DEFAULT_PRICING = PricingTable()
