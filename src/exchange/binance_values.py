"""Small response-value helpers shared by Binance integrations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any


def field(value: object, *names: str) -> Any:
    """Read a field from SDK one-of wrappers, mappings, or model objects."""

    actual_instance = getattr(value, "actual_instance", None)
    if actual_instance is not None:
        return field(actual_instance, *names)
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def to_plain_value(value: Any) -> Any:
    """Convert Binance SDK response models into plain Python values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_plain_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_plain_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return to_plain_value(value.to_dict())
    if hasattr(value, "model_dump"):
        return to_plain_value(value.model_dump())
    return value


def as_decimal(
    name: str, value: object, *, error_type: type[RuntimeError]
) -> Decimal:
    """Parse a finite Binance value as Decimal using the caller's error type."""

    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise error_type(f"Binance returned an invalid {name} value.") from exc
    if not decimal_value.is_finite():
        raise error_type(f"Binance returned a non-finite {name} value.")
    return decimal_value
