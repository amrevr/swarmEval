"""Model pricing (USD per 1M tokens) used to estimate API cost.

Unknown models produce a cost of ``None`` rather than a silent zero so that
reports can say "cost unknown" instead of "cost $0.00".  Supply your own
table via ``PricingTable(custom={...})`` or ``PricingTable.load(path)``.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Sequence, Tuple

# A price is (input, output) or (input, output, cache_read) in USD per 1M
# tokens.  Without an explicit cache-read rate, reads are billed at
# CACHE_READ_MULTIPLIER x input.
Price = Tuple[float, ...]

# Anthropic first-party API rates (cached 2026-09).
DEFAULT_PRICES: Dict[str, Price] = {
    "claude-fable-5-1": (10.00, 50.00, 0.25),
    "claude-mythos-5-1": (10.00, 50.00, 0.25),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Prompt-cache rates relative to the input rate: 5-minute cache writes cost
# 1.25x input; reads default to 0.1x input unless the price sets its own.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1

# Wrappers around a model id that do not change its price: a provider or
# region prefix ("anthropic/", "us.anthropic.") and a date or version suffix
# ("-20260101", "@20260101", "-v1:0").
_PREFIX = re.compile(r"^(?:.*/|(?:[a-z]{2}\.)?anthropic\.)")
_SUFFIX = re.compile(r"(?:[-@]\d{8}|-v\d+(?::\d+)?)+$")


def canonical_model(model: str) -> str:
    return _SUFFIX.sub("", _PREFIX.sub("", model.strip()))


class PricingTable:
    def __init__(self, custom: Optional[Dict[str, Sequence[float]]] = None,
                 include_defaults: bool = True, cache_write_multiplier: float = CACHE_WRITE_MULTIPLIER,
                 cache_read_multiplier: float = CACHE_READ_MULTIPLIER) -> None:
        self.prices: Dict[str, Price] = dict(DEFAULT_PRICES) if include_defaults else {}
        if custom:
            self.prices.update({k: tuple(float(x) for x in v) for k, v in custom.items()})
        self.cache_write_multiplier = cache_write_multiplier
        self.cache_read_multiplier = cache_read_multiplier

    def lookup(self, model: Optional[str]) -> Optional[Price]:
        """Price for ``model``, or ``None`` if it is not in the table.  Only
        dated or provider-prefixed forms of a listed id match; a different
        model that merely shares a prefix is unknown rather than mispriced."""
        if not model:
            return None
        if model in self.prices:
            return self.prices[model]
        return self.prices.get(canonical_model(model))

    def cost(self, model: Optional[str], input_tokens: int, output_tokens: int,
             cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> Optional[float]:
        p = self.lookup(model)
        if p is None:
            return None
        inp, out = p[0], p[1]
        read = p[2] if len(p) > 2 else inp * self.cache_read_multiplier
        total = (input_tokens * inp + output_tokens * out + cache_read_tokens * read
                 + cache_write_tokens * inp * self.cache_write_multiplier)
        return total / 1_000_000.0

    @classmethod
    def load(cls, path: str, include_defaults: bool = True) -> "PricingTable":
        """Read ``{"model": [input, output(, cache_read)], ...}`` (USD per 1M) from JSON."""
        import json
        with open(path) as f:
            return cls(json.load(f), include_defaults=include_defaults)


DEFAULT_PRICING = PricingTable()
