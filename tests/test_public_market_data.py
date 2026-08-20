from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.config.settings import ExecutionEnvironment, Settings, SettingsError
from src.exchange.binance_client import BinanceClient
from src.exchange.historical_data import HistoricalDataService
from src.exchange.market_stream import MarketStream
from src.exchange.public_market_client import (
    PublicMarketDataClient,
    PublicMarketDataError,
)
from src.models.market_data_source import MarketDataSource


class FakeResponse:
    def __init__(self, value: object) -> None:
        self._value = value

    def data(self) -> object:
        return self._value


def test_public_rest_configuration_has_no_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.exchange.public_market_client as public_module

    captured: dict[str, object] = {}

    class FakeConfiguration:
        def __init__(self, **kwargs: object) -> None:
            captured["configuration_kwargs"] = kwargs
            self.base_headers = {
                "Accept": "application/json",
                "X-MBX-APIKEY": "",
            }

    class FakeSpot:
        def __init__(self, *, config_rest_api: FakeConfiguration) -> None:
            captured["headers_after_build"] = dict(config_rest_api.base_headers)
            self.rest_api = object()

    monkeypatch.setattr(public_module, "ConfigurationRestAPI", FakeConfiguration)
    monkeypatch.setattr(public_module, "Spot", FakeSpot)

    client = PublicMarketDataClient()
    configuration_kwargs = captured["configuration_kwargs"]

    assert isinstance(configuration_kwargs, dict)
    assert configuration_kwargs["base_path"] == PublicMarketDataClient.REST_BASE_URL
    assert "api_key" not in configuration_kwargs
    assert "api_secret" not in configuration_kwargs
    assert "private_key" not in configuration_kwargs
    assert "custom_headers" not in configuration_kwargs
    assert "X-MBX-APIKEY" not in captured["headers_after_build"]
    assert all(
        "signature" not in str(header_name).lower()
        for header_name in captured["headers_after_build"]
    )
    assert client.uses_authentication is False


def test_market_data_source_cannot_be_redirected_by_execution_configuration() -> None:
    settings = Settings.from_env(
        env_file=None,
        environ={
            "MARKET_DATA_SOURCE": "binance_public",
            "EXECUTION_ENV": "testnet",
            "BINANCE_API_KEY": "testnet-key",
            "BINANCE_SECRET_KEY": "testnet-secret",
        },
    )
    client = PublicMarketDataClient(rest_api=object())
    history_service = HistoricalDataService(client)
    stream = MarketStream("BTCUSDC", "15m")

    assert settings.execution_environment is ExecutionEnvironment.TESTNET
    assert client.source is MarketDataSource.BINANCE_PUBLIC
    assert history_service.source is MarketDataSource.BINANCE_PUBLIC
    assert stream.source is MarketDataSource.BINANCE_PUBLIC
    assert client.base_url == "https://data-api.binance.vision"
    assert stream.stream_url == (
        "wss://data-stream.binance.vision/ws/btcusdc@kline_15m"
    )
    assert "testnet" not in client.base_url.lower()
    assert "testnet" not in stream.stream_url.lower()


def test_legacy_testnet_flag_cannot_redirect_public_market_data() -> None:
    settings = Settings.from_env(
        env_file=None,
        environ={"BINANCE_TESTNET": "false"},
    )
    client = PublicMarketDataClient(rest_api=object())
    stream = MarketStream("BTCUSDC", "15m")

    assert settings.execution_environment is ExecutionEnvironment.TESTNET
    assert client.base_url == PublicMarketDataClient.REST_BASE_URL
    assert stream.stream_url.startswith(MarketStream.WEBSOCKET_BASE_URL)


@pytest.mark.parametrize(
    ("environment", "message"),
    (
        ({"MARKET_DATA_SOURCE": "testnet"}, "MARKET_DATA_SOURCE"),
        ({"EXECUTION_ENV": "live"}, "live execution is disabled"),
    ),
)
def test_unsupported_data_source_or_live_execution_is_rejected(
    environment: dict[str, str], message: str
) -> None:
    with pytest.raises(SettingsError, match=message):
        Settings.from_env(env_file=None, environ=environment)


def test_public_ticker_price_parsing_uses_decimal() -> None:
    class FakeRestApi:
        def ticker_price(self, *, symbol: str) -> FakeResponse:
            assert symbol == "BTCUSDC"
            ticker = SimpleNamespace(symbol="BTCUSDC", price="64171.50000000")
            return FakeResponse(SimpleNamespace(actual_instance=ticker))

    price = PublicMarketDataClient(rest_api=FakeRestApi()).get_current_price()

    assert price == Decimal("64171.50000000")


def test_public_kline_rows_preserve_market_values_and_volume() -> None:
    expected_row = [
        1_786_998_600_000,
        "64000.10",
        "64100.25",
        "63950.00",
        "64050.25",
        "123.45678900",
        1_786_999_499_999,
        "0",
        10,
        "0",
        "0",
        "0",
    ]

    class FakeRestApi:
        def klines(self, symbol: str, interval: object, *, limit: int) -> FakeResponse:
            assert symbol == "BTCUSDC"
            assert str(interval.value) == "15m"
            assert limit == 1
            return FakeResponse([expected_row])

    rows = PublicMarketDataClient(rest_api=FakeRestApi()).get_klines(
        "BTCUSDC", "15m", limit=1
    )

    assert rows == [expected_row]
    assert rows[0][5] == "123.45678900"


@pytest.mark.parametrize("response_data", ({"not": "rows"}, [[0, "1"], "bad-row"]))
def test_public_kline_rejects_malformed_responses(response_data: object) -> None:
    class FakeRestApi:
        def klines(self, symbol: str, interval: object, *, limit: int) -> FakeResponse:
            return FakeResponse(response_data)

    client = PublicMarketDataClient(rest_api=FakeRestApi())

    with pytest.raises(PublicMarketDataError, match="kline"):
        client.get_klines("BTCUSDC", "15m", limit=1)


def test_public_network_failure_is_wrapped_without_fallback() -> None:
    class OfflineRestApi:
        def ticker_price(self, *, symbol: str) -> FakeResponse:
            raise ConnectionError("network unavailable")

    client = PublicMarketDataClient(rest_api=OfflineRestApi())

    with pytest.raises(PublicMarketDataError, match="Public Binance") as error:
        client.get_current_price()

    assert isinstance(error.value.__cause__, ConnectionError)


def test_public_exchange_info_is_read_from_public_client() -> None:
    class FakeRestApi:
        def exchange_info(self, *, symbol: str) -> FakeResponse:
            assert symbol == "BTCUSDC"
            return FakeResponse({"symbols": [{"symbol": symbol, "status": "TRADING"}]})

    info = PublicMarketDataClient(rest_api=FakeRestApi()).get_exchange_info()

    assert info["symbols"][0]["symbol"] == "BTCUSDC"


def test_private_client_is_testnet_only_and_has_no_order_methods() -> None:
    assert BinanceClient.REST_BASE_URL == "https://testnet.binance.vision"
    assert not hasattr(BinanceClient, "create_order")
    assert not hasattr(BinanceClient, "cancel_order")
    assert not hasattr(BinanceClient, "place_order")
