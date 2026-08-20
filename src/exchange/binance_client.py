"""Authenticated Binance Spot Testnet client reserved for private reads."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import TypeVar

from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import SPOT_REST_API_TESTNET_URL
from binance_sdk_spot.spot import Spot

from src.config.settings import ExecutionEnvironment, Settings
from src.exchange.binance_values import as_decimal, field

T = TypeVar("T")


class BinanceClientError(RuntimeError):
    """Raised when an authenticated Testnet read cannot be completed safely."""


class BinanceClient:
    """Expose future/private Binance reads on Spot Testnet only.

    V1.1 does not instantiate this client during analytics startup and provides
    no order-placement methods.
    """

    REST_BASE_URL = SPOT_REST_API_TESTNET_URL

    def __init__(self, settings: Settings) -> None:
        if settings.execution_environment is not ExecutionEnvironment.TESTNET:
            raise ValueError("Authenticated Binance access is Testnet-only in V1.1.")
        self._settings = settings
        configuration = ConfigurationRestAPI(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_secret_key,
            base_path=self.REST_BASE_URL,
            timeout=5_000,
            retries=3,
            backoff=500,
        )
        self._rest_api = Spot(config_rest_api=configuration).rest_api

    @property
    def has_credentials(self) -> bool:
        return self._settings.has_credentials

    @property
    def environment_name(self) -> str:
        return self._settings.execution_environment_name

    def _request(self, operation_name: str, operation: Callable[[], T]) -> T:
        try:
            return operation()
        except Exception as exc:
            # Do not include SDK exception text; signed request details must never leak.
            raise BinanceClientError(
                f"Binance Testnet {operation_name} request failed ({type(exc).__name__})."
            ) from exc

    def get_account_balances(
        self, *, non_zero_only: bool = True
    ) -> dict[str, dict[str, Decimal]] | None:
        """Return Testnet balances when credentials exist, otherwise return None."""

        if not self.has_credentials:
            return None

        response = self._request("account balance", lambda: self._rest_api.get_account())
        balances = field(response.data(), "balances")
        if balances is None:
            raise BinanceClientError(
                "Binance Testnet account response did not contain balances."
            )

        result: dict[str, dict[str, Decimal]] = {}
        for balance in balances:
            asset = str(field(balance, "asset") or "")
            free = as_decimal(
                "free balance", field(balance, "free"), error_type=BinanceClientError
            )
            locked = as_decimal(
                "locked balance",
                field(balance, "locked"),
                error_type=BinanceClientError,
            )
            if not asset or (non_zero_only and free == 0 and locked == 0):
                continue
            result[asset] = {"free": free, "locked": locked}
        return result
