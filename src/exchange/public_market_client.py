"""Unauthenticated Binance public Spot market-data REST integration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Any, TypeVar

from binance_common.configuration import ConfigurationRestAPI
from binance_sdk_spot.rest_api.models import KlinesIntervalEnum
from binance_sdk_spot.spot import Spot

from src.exchange.binance_values import as_decimal, field, to_plain_value
from src.models.market_data_source import MarketDataSource

T = TypeVar("T")


class PublicMarketDataError(RuntimeError):
    """Raised when public Binance market data is unavailable or malformed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class PublicMarketDataClient:
    """Expose only unauthenticated reads from Binance's public market-data host."""

    REST_BASE_URL = "https://data-api.binance.vision"

    def __init__(self, default_symbol: str = "BTCUSDC", *, rest_api: Any = None) -> None:
        self._default_symbol = default_symbol.strip().upper()
        if self._default_symbol != "BTCUSDC":
            raise ValueError("V1.1 supports BTCUSDC public market data only.")
        self._rest_api = rest_api if rest_api is not None else self._build_rest_api()

    @staticmethod
    def _build_rest_api() -> Any:
        # Deliberately omit all key, secret, private-key, and custom-header inputs.
        configuration = ConfigurationRestAPI(
            base_path=PublicMarketDataClient.REST_BASE_URL,
            timeout=5_000,
            retries=3,
            backoff=500,
        )
        # The SDK inserts an empty API-key header by default. Public requests must
        # not carry an authentication header at all, even when it is blank.
        configuration.base_headers.pop("X-MBX-APIKEY", None)
        return Spot(config_rest_api=configuration).rest_api

    @property
    def source(self) -> MarketDataSource:
        return MarketDataSource.BINANCE_PUBLIC

    @property
    def base_url(self) -> str:
        return self.REST_BASE_URL

    @property
    def uses_authentication(self) -> bool:
        return False

    def _request(self, operation_name: str, operation: Callable[[], T]) -> T:
        try:
            return operation()
        except Exception as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            headers = getattr(response, "headers", {}) or {}
            retry_after: float | None = None
            try:
                configured_retry_after = headers.get("Retry-After")
                if configured_retry_after is not None:
                    retry_after = max(0.0, float(configured_retry_after))
            except (TypeError, ValueError):
                retry_after = None
            raise PublicMarketDataError(
                f"Public Binance {operation_name} request failed ({type(exc).__name__}).",
                status_code=status_code,
                retry_after_seconds=retry_after,
            ) from exc

    def verify_connection(self) -> bool:
        """Verify the public market-data host with its read-only ping endpoint."""

        self._request("connectivity", lambda: self._rest_api.ping())
        return True

    def get_server_time(self) -> int:
        """Return public Binance server time as Unix milliseconds."""

        response = self._request("server time", lambda: self._rest_api.time())
        server_time = field(response.data(), "server_time", "serverTime")
        if server_time is None:
            raise PublicMarketDataError(
                "Public Binance server time response was incomplete."
            )
        try:
            return int(server_time)
        except (TypeError, ValueError) as exc:
            raise PublicMarketDataError(
                "Public Binance server time response was invalid."
            ) from exc

    def get_current_price(self, symbol: str | None = None) -> Decimal:
        """Return the real public BTC/USDC Spot price."""

        requested_symbol = (symbol or self._default_symbol).upper()
        if requested_symbol != "BTCUSDC":
            raise ValueError("V1.1 supports BTCUSDC price requests only.")
        response = self._request(
            "ticker price",
            lambda: self._rest_api.ticker_price(symbol=requested_symbol),
        )
        data = response.data()
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            data = next(
                (
                    item
                    for item in data
                    if field(item, "symbol") == requested_symbol
                ),
                None,
            )
        price = field(data, "price")
        if price is None:
            raise PublicMarketDataError(
                "Public Binance ticker price response was incomplete."
            )
        return as_decimal("price", price, error_type=PublicMarketDataError)

    def get_exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        """Return public BTC/USDC exchange metadata as plain Python data."""

        requested_symbol = (symbol or self._default_symbol).upper()
        if requested_symbol != "BTCUSDC":
            raise ValueError("V1.1 supports BTCUSDC exchange information only.")
        response = self._request(
            "exchange information",
            lambda: self._rest_api.exchange_info(symbol=requested_symbol),
        )
        data = to_plain_value(response.data())
        if not isinstance(data, dict):
            raise PublicMarketDataError(
                "Public Binance exchange information response was invalid."
            )
        return data

    def get_klines(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[list[Any]]:
        """Return real public Spot kline rows without transforming their values."""

        requested_symbol = symbol.upper()
        if requested_symbol != "BTCUSDC":
            raise ValueError("V1.1 supports BTCUSDC kline requests only.")
        if not 1 <= limit <= 1_000:
            raise ValueError("Binance kline limit must be between 1 and 1000.")
        if start_time_ms is not None and start_time_ms < 0:
            raise ValueError("Binance kline start time must not be negative.")
        if end_time_ms is not None and end_time_ms < 0:
            raise ValueError("Binance kline end time must not be negative.")
        if (
            start_time_ms is not None
            and end_time_ms is not None
            and start_time_ms > end_time_ms
        ):
            raise ValueError("Binance kline start time must not exceed end time.")
        try:
            sdk_interval = KlinesIntervalEnum(interval)
        except ValueError as exc:
            raise ValueError(f"Unsupported Binance kline interval: {interval}") from exc

        request_arguments: dict[str, int] = {"limit": limit}
        if start_time_ms is not None:
            request_arguments["start_time"] = start_time_ms
        if end_time_ms is not None:
            request_arguments["end_time"] = end_time_ms

        response = self._request(
            "historical klines",
            lambda: self._rest_api.klines(
                requested_symbol,
                sdk_interval,
                **request_arguments,
            ),
        )
        data = response.data()
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
            raise PublicMarketDataError(
                "Public Binance kline response was not a sequence."
            )

        rows: list[list[Any]] = []
        for row in data:
            if not isinstance(row, Sequence) or isinstance(
                row, (str, bytes, bytearray)
            ):
                raise PublicMarketDataError(
                    "Public Binance returned a malformed kline row."
                )
            rows.append(list(row))
        return rows
