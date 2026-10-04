"""Causal V12 raw replay/certification and the unchanged Phase-1 bound adapter."""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, DecimalException, localcontext
from pathlib import Path

from src.microstructure.v10 import (
    MicrostructureIntegrityError,
    _canonical,
    _decimal,
    _from_text,
    sha256_file,
)
from src.microstructure.v12 import (
    DEFAULT_POLICY,
    SCHEMA,
    STREAMS,
    VERSION,
    Policy,
    digest,
    strict_json,
)
from src.orderbook.book import ApplyStatus, ReconstructedOrderBook
from src.orderbook.models import DepthDiffEvent, DepthSnapshot
from src.research.v12_maker_bound import (
    ARITHMETIC,
    Assumptions,
    ConditionalMakerBound,
    Order,
    PublicTrade,
    Receipt,
    State,
    Timing,
    _epoch_us,
)

D = Decimal
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def integer(value, label):
    if type(value) is not int or value < 0:
        raise MicrostructureIntegrityError(f"invalid integer {label}")
    return value


def nonnegative(value, label):
    if value == "0" or value == "0.00000000":
        return D(0)
    if not isinstance(value, str):
        raise MicrostructureIntegrityError(f"Decimal text required for {label}")
    result = D(value)
    if not result.is_finite() or result < 0:
        raise MicrostructureIntegrityError(f"invalid nonnegative {label}")
    return result


def levels(value):
    if not isinstance(value, list):
        raise MicrostructureIntegrityError("depth level array required")
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise MicrostructureIntegrityError("exact depth price/quantity pair required")
        result.append((_decimal(row[0], field="depth price"), nonnegative(row[1], "depth quantity")))
    if len({p for p, _ in result}) != len(result):
        raise MicrostructureIntegrityError("duplicate depth price")
    return tuple(result)


def verify_envelope(event):
    if event.get("schema_version") != SCHEMA or event.get("collector_version") != VERSION:
        raise MicrostructureIntegrityError("only new V12 public envelopes are accepted")
    if event.get("symbol") != "BTCUSDC" or event.get("venue") != "BINANCE_SPOT":
        raise MicrostructureIntegrityError("BTCUSDC Spot only")
    raw = base64.b64decode(event["raw_payload_base64"], validate=True)
    if digest(raw) != event["raw_payload_sha256"]:
        raise MicrostructureIntegrityError("raw payload hash mismatch")
    if event["record_type"] != "INVALID_PAYLOAD":
        decoded = strict_json(raw)
        if decoded != event["decoded_payload"]:
            raise MicrostructureIntegrityError("decoded original payload mismatch")
        body = decoded.get("data", decoded)
        if body != event["payload"]:
            raise MicrostructureIntegrityError("normalized payload mismatch")
        if event["record_type"] in STREAMS.values() and (
            decoded.get("stream") != event["stream_name"] or
            STREAMS.get(event["stream_name"]) != event["record_type"]
        ):
            raise MicrostructureIntegrityError("combined stream identity mismatch")
    if digest(_canonical(event["decoded_payload"]).encode()) != event["canonical_payload_sha256"]:
        raise MicrostructureIntegrityError("canonical payload hash mismatch")
    if _epoch_us(_from_text(event["received_at_utc"])) != event["receipt_utc_epoch_us"]:
        raise MicrostructureIntegrityError("UTC receipt encodings disagree")
    for field in ("ingestion_ordinal", "receipt_monotonic_ns", "dispatch_monotonic_ns",
                  "decode_start_monotonic_ns", "decode_end_monotonic_ns", "processing_delay_ns"):
        integer(event[field], field)
    if not (event["receipt_monotonic_ns"] <= event["decode_start_monotonic_ns"]
            <= event["decode_end_monotonic_ns"] <= event["dispatch_monotonic_ns"]):
        raise MicrostructureIntegrityError("processing timestamp order invalid")
    if event["processing_delay_ns"] != event["dispatch_monotonic_ns"] - event["receipt_monotonic_ns"]:
        raise MicrostructureIntegrityError("processing delay inconsistent")
    if event["exchange_timestamp_unit"] not in ("MICROSECOND", "MILLISECOND", "NONE"):
        raise MicrostructureIntegrityError("unknown timestamp unit")
    if _from_text(event["processing_available_utc"]) < _from_text(event["received_at_utc"]):
        raise MicrostructureIntegrityError("processing UTC precedes receipt")
    scale = {"MICROSECOND": 1, "MILLISECOND": 1000, "NONE": None}[event["exchange_timestamp_unit"]]
    for field in ("E", "T"):
        raw_value = event["payload"].get(field) if event["payload"] else None
        if event["exchange_timestamps_raw"][field] != raw_value:
            raise MicrostructureIntegrityError("exchange timestamp provenance mismatch")
        expected = ([raw_value * scale, raw_value * scale + scale - 1]
                    if type(raw_value) is int and raw_value >= 0 and scale else None)
        if event["exchange_time_intervals_us"][field] != expected:
            raise MicrostructureIntegrityError("timestamp units/interval mismatch")


class ReplayMachine:
    def __init__(self, policy=DEFAULT_POLICY):
        self.policy = policy
        self.book = ReconstructedOrderBook(max_levels_per_side=policy.max_levels)
        self.synced, self.resync_seen = False, False
        self.pending = []
        self.snapshot_id = None
        self.book_available_ns = None
        self.known_bid_floor = None
        self.book_states = {}
        self.tickers = []
        self.pending_tickers = {}
        self.reconcile_waiters = {}
        self.ready_aggregates = set()
        self.reconciled_trade_ids = set()
        self.trades, self.aggregates = {}, {}
        self.last_trade_id, self.last_agg_id, self.last_agg_end = None, None, None
        self.last_trade_time = None
        self.clock_mapping, self.metadata = None, None
        self.connections, self.connection_epochs, self.outages = {}, {}, []
        self.open_outages = {}
        self.clock_id, self.session_id = None, None
        self.last_ordinal, self.last_ns, self.last_utc = 0, None, None
        self.last_available_ns, self.last_available_utc = None, None
        self.stream_indices = {}
        self.failures, self.trace = [], []
        self.counts = {}
        self.clean_close = False
        self.bridges = 0
        self.duplicates = 0
        self.book_digest = None

    def fail(self, event, reason):
        item = {"ordinal": event["ingestion_ordinal"], "reason": reason}
        if item not in self.failures:
            self.failures.append(item)

    def source_time(self, event, field):
        interval = event["exchange_time_intervals_us"].get(field)
        if interval is None:
            raise MicrostructureIntegrityError(f"missing observed {field} timestamp/unit")
        return interval

    def receipt(self, event):
        names = {"TRADE": "TRADE", "AGGTRADE": "AGGTRADE", "DIFF_DEPTH": "DEPTH",
                 "BOOK_TICKER": "BOOK_TICKER"}
        stream = names.get(event["record_type"], "CONTROL")
        connection = event["connection_id"] if stream != "CONTROL" else "control:" + event["session_id"]
        return Receipt(event["session_id"], connection, stream, event["ingestion_ordinal"],
                       _from_text(event["received_at_utc"]), event["receipt_monotonic_ns"],
                       event["canonical_payload_sha256"])

    def individual(self, event):
        p = event["payload"]
        if p.get("e") != "trade" or p.get("s") != "BTCUSDC" or type(p.get("m")) is not bool:
            raise MicrostructureIntegrityError("ordinary BTCUSDC individual trade required")
        t, e = self.source_time(event, "T"), self.source_time(event, "E")
        return PublicTrade(integer(p.get("t"), "trade ID"), _decimal(p.get("p"), field="p"),
                           _decimal(p.get("q"), field="q"), p["m"], t[0], e[0], t[1] - t[0] + 1,
                           self.receipt(event))

    def feed(self, event):
        verify_envelope(event)
        before = len(self.failures)
        ordinal, ns = event["ingestion_ordinal"], event["receipt_monotonic_ns"]
        wall = event["receipt_utc_epoch_us"]
        if ordinal != self.last_ordinal + 1:
            raise MicrostructureIntegrityError("shared ordinal discontinuity: never hindsight-sort")
        if self.last_ns is not None and (ns < self.last_ns or wall < self.last_utc):
            self.fail(event, "RECEIPT_CLOCK_REGRESSION")
        if self.last_ns is not None and abs((wall - self.last_utc) - (ns - self.last_ns) // 1000) > self.policy.clock_jump_tolerance_us:
            self.fail(event, "CLOCK_DISCONTINUITY")
        if self.clock_id is None:
            self.clock_id, self.session_id = event["monotonic_clock_id"], event["session_id"]
        elif (self.clock_id, self.session_id) != (event["monotonic_clock_id"], event["session_id"]):
            self.fail(event, "CLOCK_OR_SESSION_CHANGED")
        self.last_ordinal, self.last_ns, self.last_utc = ordinal, ns, wall
        if self.last_available_ns is not None and ns < self.last_available_ns:
            self.fail(event, "OVERLAPPING_SINGLE_INGRESS_PROCESSING")
        self.last_available_ns = event["dispatch_monotonic_ns"]
        self.last_available_utc = _from_text(event["processing_available_utc"])
        stream_key = (event["connection_id"], event["stream_name"])
        expected_index = self.stream_indices.get(stream_key, 0)
        if event["stream_record_index"] != expected_index:
            self.fail(event, "STREAM_RECORD_INDEX_DISCONTINUITY")
        self.stream_indices[stream_key] = expected_index + 1
        kind, p = event["record_type"], event["payload"]
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if event["clock_discontinuity"]:
            self.fail(event, "CLOCK_DISCONTINUITY")
        if event["processing_delay_ns"] > self.policy.max_processing_delay_us * 1000:
            self.fail(event, "PROCESSING_BACKLOG_LIMIT")
        if event["observed_drop_count"]:
            self.fail(event, "APPLICATION_DROP")
        try:
            if kind in STREAMS.values() and (
                not self.connections.get(event["connection_id"]) or
                self.connection_epochs.get(event["connection_id"]) != event["reconnect_epoch"]
            ):
                self.fail(event, "MARKET_OUTSIDE_CONNECTED_EPOCH")
            if kind in ("CLOCK", "SYMBOL_RULES", "REST_SNAPSHOT"):
                request, response = event["details"]["request_started"], event["details"]["response_received"]
                for reading in (request, response):
                    if _epoch_us(_from_text(reading["utc"])) != reading["utc_epoch_us"]:
                        raise MicrostructureIntegrityError("HTTP UTC encodings disagree")
                if not (request["monotonic_ns"] <= response["monotonic_ns"] <= ns) or not (
                    request["utc_epoch_us"] <= response["utc_epoch_us"] <= wall
                ):
                    raise MicrostructureIntegrityError("HTTP availability provenance invalid")
            if kind == "CLOCK":
                request, response = event["details"]["request_started"], event["details"]["response_received"]
                scale = {"MICROSECOND": 1, "MILLISECOND": 1000}[event["details"]["timestamp_unit"]]
                server = integer(p.get("serverTime"), "serverTime") * scale
                if response["monotonic_ns"] < request["monotonic_ns"] or (
                    response["utc_epoch_us"] < request["utc_epoch_us"] or response["monotonic_ns"] > ns
                ):
                    raise MicrostructureIntegrityError("HTTP clock provenance invalid")
                self.clock_mapping = {
                    "id": f"clock-{ordinal}", "available_ns": event["dispatch_monotonic_ns"],
                    "offset_min_us": server - response["utc_epoch_us"],
                    "offset_max_us": server + scale - 1 - request["utc_epoch_us"],
                    "meaning": "public server-time RTT bracket, NOT order latency"}
            elif kind == "SYMBOL_RULES":
                symbols = p.get("symbols", [])
                if len(symbols) != 1 or symbols[0].get("symbol") != "BTCUSDC":
                    raise MicrostructureIntegrityError("dated BTCUSDC exchangeInfo required")
                info = symbols[0]
                if info.get("status") != "TRADING" or info.get("isSpotTradingAllowed") is not True or (
                    "LIMIT_MAKER" not in info.get("orderTypes", [])
                ):
                    raise MicrostructureIntegrityError("Spot LIMIT_MAKER trading status unavailable")
                filters = {f["filterType"]: f for f in info["filters"]}
                if len(filters) != len(info["filters"]):
                    raise MicrostructureIntegrityError("duplicate symbol filter")
                for ftype, fields in (("PRICE_FILTER", ("tickSize", "minPrice", "maxPrice")),
                                      ("LOT_SIZE", ("stepSize", "minQty", "maxQty"))):
                    for field in fields:
                        nonnegative(filters[ftype][field], field)
                if not ({"MIN_NOTIONAL", "NOTIONAL"} & filters.keys()):
                    raise MicrostructureIntegrityError("notional constraints unknown")
                for ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                    if ftype in filters:
                        nonnegative(filters[ftype]["minNotional"], "minNotional")
                        if ftype == "NOTIONAL":
                            nonnegative(filters[ftype]["maxNotional"], "maxNotional")
                self.metadata = {"available_ns": event["dispatch_monotonic_ns"], "info": info,
                                 "ordinal": ordinal, "sha256": event["canonical_payload_sha256"]}
            elif kind in ("CONNECT_START", "RECONNECT_START"):
                self.connection_epochs[event["connection_id"]] = event["reconnect_epoch"]
            elif kind in ("CONNECTED", "RECONNECTED"):
                if self.connection_epochs.get(event["connection_id"]) != event["reconnect_epoch"]:
                    self.fail(event, "CONNECTION_START_MISSING")
                self.connections[event["connection_id"]] = True
                if self.open_outages:
                    for connection, start in list(self.open_outages.items()):
                        self.outages.append({"connection_id": connection, "start": start,
                                             "end_ordinal": ordinal, "end_utc": event["received_at_utc"],
                                             "end_monotonic_ns": ns})
                    self.open_outages.clear()
            elif kind == "DISCONNECTED":
                self.connections[event["connection_id"]] = False
                if not p.get("expected", False):
                    self.open_outages[event["connection_id"]] = {
                        "ordinal": ordinal, "utc": event["received_at_utc"], "monotonic_ns": ns}
                    self.fail(event, "DISCONNECTED_INTERVAL")
                    self.synced = False
                    self.book.last_update_id = None
                    self.pending = []
            elif kind in ("GAP_DETECTED", "RESYNC_START", "PUBLIC_REQUEST_FAILED", "INVALID_PAYLOAD"):
                self.resync_seen = True
                self.fail(event, kind)
                if kind in ("RESYNC_START", "GAP_DETECTED"):
                    self.synced = False
                    self.book.last_update_id = None
                    self.pending = []
            elif kind == "REST_SNAPSHOT":
                self.snapshot_id = integer(p.get("lastUpdateId"), "snapshot ID")
                bids, asks = levels(p.get("bids")), levels(p.get("asks"))
                if not bids or not asks:
                    raise MicrostructureIntegrityError("snapshot sides missing")
                self.known_bid_floor = min(price for price, _ in bids)
                self.book.reset(DepthSnapshot(self.snapshot_id, bids, asks, _from_text(event["received_at_utc"])))
                self.synced = False
                if not self.book.is_valid:
                    raise MicrostructureIntegrityError("invalid snapshot book")
                buffered, self.pending = self.pending, []
                for diff, original in buffered:
                    if diff.final_update_id > self.snapshot_id:
                        self.apply_depth(diff, event)
                        if self.book.last_update_id is None:
                            break
                self.remember_book(event)
            elif kind == "DIFF_DEPTH":
                if p.get("e") != "depthUpdate" or p.get("s") != "BTCUSDC":
                    raise MicrostructureIntegrityError("wrong depth stream")
                first, final = integer(p.get("U"), "U"), integer(p.get("u"), "u")
                if first > final:
                    raise MicrostructureIntegrityError("invalid U/u interval")
                source = self.source_time(event, "E")[0]
                diff = DepthDiffEvent(first, final, EPOCH + timedelta(microseconds=source),
                                      levels(p.get("b")), levels(p.get("a")))
                if self.book.last_update_id is None:
                    self.pending.append((diff, event))
                    if len(self.pending) > self.policy.max_bridge_events:
                        raise MicrostructureIntegrityError("snapshot buffer overflow")
                else:
                    self.apply_depth(diff, event)
            elif kind == "TRADE":
                trade = self.individual(event)
                signature = event["canonical_payload_sha256"]
                if trade.trade_id in self.trades:
                    self.duplicates += 1
                    if self.trades[trade.trade_id][1] != signature:
                        self.fail(event, "CONFLICTING_DUPLICATE_TRADE")
                else:
                    if self.last_trade_id is not None and trade.trade_id != self.last_trade_id + 1:
                        self.fail(event, "TRADE_ID_GAP" if trade.trade_id > self.last_trade_id else "TRADE_ID_REGRESSION")
                    if self.last_trade_time is not None and trade.trade_time_us < self.last_trade_time:
                        self.fail(event, "EXCHANGE_TRADE_TIME_REGRESSION")
                    self.trades[trade.trade_id] = (trade, signature)
                    self.last_trade_id, self.last_trade_time = trade.trade_id, trade.trade_time_us
                    for aid in self.reconcile_waiters.pop(trade.trade_id, ()):
                        self.aggregates[aid]["missing"].remove(trade.trade_id)
                        if not self.aggregates[aid]["missing"]:
                            self.ready_aggregates.add(aid)
            elif kind == "AGGTRADE":
                if p.get("e") != "aggTrade" or p.get("s") != "BTCUSDC" or type(p.get("m")) is not bool:
                    raise MicrostructureIntegrityError("invalid BTCUSDC aggregate")
                aid, first, final = (integer(p.get(k), k) for k in ("a", "f", "l"))
                if final < first or final - first > 100_000:
                    raise MicrostructureIntegrityError("invalid/unsupported aggregate range")
                _decimal(p.get("p"), field="aggregate p")
                _decimal(p.get("q"), field="aggregate q")
                self.source_time(event, "T")
                self.source_time(event, "E")
                if self.source_time(event, "T")[0] > self.source_time(event, "E")[0]:
                    raise MicrostructureIntegrityError("aggregate event time precedes trade time")
                if aid in self.aggregates:
                    if self.aggregates[aid]["signature"] != event["canonical_payload_sha256"]:
                        self.fail(event, "CONFLICTING_DUPLICATE_AGGREGATE")
                else:
                    if self.last_agg_id is not None and aid != self.last_agg_id + 1:
                        self.fail(event, "AGGREGATE_ID_GAP_OR_REGRESSION")
                    if self.last_agg_end is not None and first != self.last_agg_end + 1:
                        self.fail(event, "AGGREGATE_RANGE_OVERLAP_OR_GAP")
                    self.aggregates[aid] = {"payload": p, "signature": event["canonical_payload_sha256"],
                                            "ordinal": ordinal, "status": "PENDING", "missing": []}
                    missing = [tid for tid in range(first, final + 1) if tid not in self.trades]
                    self.aggregates[aid]["missing"] = missing
                    for tid in missing:
                        self.reconcile_waiters.setdefault(tid, set()).add(aid)
                    if not missing:
                        self.ready_aggregates.add(aid)
                    self.last_agg_id, self.last_agg_end = aid, final
            elif kind == "BOOK_TICKER":
                if p.get("s") != "BTCUSDC":
                    raise MicrostructureIntegrityError("wrong bookTicker symbol")
                uid = integer(p.get("u"), "bookTicker u")
                bid, ask = _decimal(p.get("b"), field="b"), _decimal(p.get("a"), field="a")
                bq, aq = nonnegative(p.get("B"), "B"), nonnegative(p.get("A"), "A")
                if bid >= ask or bq == 0 or aq == 0:
                    raise MicrostructureIntegrityError("invalid bookTicker quotes")
                if self.tickers and uid < self.tickers[-1]["u"]:
                    self.fail(event, "BOOK_TICKER_ID_REGRESSION")
                self.tickers.append({"u": uid, "quote": (bid, bq, ask, aq),
                                     "ordinal": ordinal, "status": "UNVERIFIABLE_INTERMEDIATE"})
                self.pending_tickers.setdefault(uid, []).append(self.tickers[-1])
                self.check_tickers(event, uid)
            elif kind == "SESSION_CLOSE":
                self.clean_close = p.get("clean") is True
            if kind in ("TRADE", "AGGTRADE"):
                self.reconcile(event)
            if kind in ("TRADE", "DIFF_DEPTH", "AGGTRADE"):
                mapping = self.clock_mapping
                if mapping is None:
                    self.fail(event, "CLOCK_MAPPING_UNKNOWN")
                else:
                    observed_e = self.source_time(event, "E")[0]
                    if observed_e > event["receipt_utc_epoch_us"] + mapping["offset_max_us"]:
                        self.fail(event, "EXCHANGE_TIME_AFTER_POSSIBLE_RECEIPT")
        except (ValueError, KeyError, TypeError, OverflowError, DecimalException) as exc:
            self.fail(event, "INVALID_EVENT: " + str(exc))
            if kind in ("REST_SNAPSHOT", "DIFF_DEPTH"):
                self.synced = False
                self.book.last_update_id = None
        self.trace.append({"ordinal": ordinal, "kind": kind, "receipt_ns": ns,
                           "source_payload_sha256": event["canonical_payload_sha256"],
                           "book_synced": self.synced, "depth_id": self.book.last_update_id,
                           "book_sha256": self.book_digest,
                           "maker_input": self.maker_input(event),
                           "new_failures": self.failures[before:]})

    def maker_input(self, event):
        if event["record_type"] != "TRADE" or not event["payload"]:
            return None
        p = event["payload"]
        return {"trade_id": p.get("t"), "price": p.get("p"), "quantity": p.get("q"),
                "buyer_is_maker": p.get("m"), "exchange_intervals": event["exchange_time_intervals_us"],
                "available_monotonic_ns": event["dispatch_monotonic_ns"],
                "availability": "LOCAL_ONLY_NOT_MATCHING_ORDER"}

    def apply_depth(self, diff, available_event):
        status = self.book.apply(diff)
        if status in (ApplyStatus.SEQUENCE_GAP, ApplyStatus.INVALID_BOOK):
            self.fail(available_event, "DEPTH_GAP" if status == ApplyStatus.SEQUENCE_GAP else "INVALID_BOOK")
            self.synced = False
            self.book.last_update_id = None
            return
        if status == ApplyStatus.APPLIED:
            if not self.synced:
                self.bridges += 1
            self.synced = True
            self.remember_book(available_event)

    def remember_book(self, available_event):
        if not self.book.is_valid or self.book.last_update_id is None:
            return
        bids, asks = self.book.top_levels(self.policy.max_levels)
        self.book_digest = digest(_canonical({"u": self.book.last_update_id, "b": bids, "a": asks}).encode())
        bid, ask = self.book.best_bid, self.book.best_ask
        self.book_states[self.book.last_update_id] = (*bid, *ask)
        self.book_available_ns = available_event["dispatch_monotonic_ns"]
        self.check_tickers(available_event, self.book.last_update_id)
        # Trimming can only shrink known coverage. Never equate absent outside levels with zero.
        self.known_bid_floor = max(self.known_bid_floor, min(p for p, _ in bids))

    def check_tickers(self, event, uid):
        if uid not in self.book_states:
            return
        for ticker in self.pending_tickers.pop(uid, []):
            if ticker["status"] == "UNVERIFIABLE_INTERMEDIATE" and ticker["u"] in self.book_states:
                ticker["status"] = "MATCHED" if tuple(ticker["quote"]) == self.book_states[ticker["u"]] else "MISMATCH"
                ticker["checked_at_ordinal"] = event["ingestion_ordinal"]
                if ticker["status"] == "MISMATCH":
                    self.fail(event, "BOOK_TICKER_DEPTH_DISAGREEMENT")

    def reconcile(self, event):
        for aid in sorted(self.ready_aggregates):
            aggregate = self.aggregates[aid]
            p = aggregate["payload"]
            trades = [self.trades[tid][0] for tid in range(p["f"], p["l"] + 1)]
            with localcontext(ARITHMETIC):
                quantity = sum((t.quantity for t in trades), D(0))
            compatible = (quantity == D(p["q"]) and all(
                t.price == D(p["p"]) and t.buyer_is_maker == p["m"] for t in trades))
            aggregate["status"] = "MATCHED" if compatible else "MISMATCH"
            aggregate["checked_at_ordinal"] = event["ingestion_ordinal"]
            aggregate["individual_quantity"] = str(quantity)
            aggregate["individual_T_range_us"] = [min(t.trade_time_us for t in trades), max(t.trade_time_us for t in trades)]
            if compatible:
                self.reconciled_trade_ids.update(range(p["f"], p["l"] + 1))
            if not compatible:
                self.fail(event, "AGGREGATE_TRADE_RECONCILIATION_FAILURE")
        self.ready_aggregates.clear()

    def canonical_result(self):
        return _canonical({"events": self.trace, "failures": self.failures,
                           "book_sha256": self.book_digest, "counts": self.counts,
                           "aggregates": self.aggregates, "tickers": self.tickers,
                           "outages": self.outages, "open_outages": self.open_outages,
                           "clock": self.clock_mapping, "metadata": self.metadata,
                           "clean_close": self.clean_close, "snapshot_bridges": self.bridges,
                           "unreconciled_individual_ids": sorted(self.trades.keys() - self.reconciled_trade_ids),
                           "duplicate_trades": self.duplicates}).encode()


def load_session(directory: Path):
    if (directory / "closure.summary.json").exists():
        closure = strict_json((directory / "closure.summary.json").read_bytes())
        for name in ("session.manifest.json", "events.jsonl", "persistence.acks.jsonl"):
            if closure["artifact_sha256"][name] != sha256_file(directory / name):
                raise MicrostructureIntegrityError("closed session artifact hash mismatch")
    manifest = strict_json((directory / "session.manifest.json").read_bytes())
    if manifest.get("schema_version") != "V12_ENGINEERING_SESSION_1" or not manifest.get("engineering_only"):
        raise MicrostructureIntegrityError("only explicitly engineering V12 sessions accepted")
    lines = (directory / "events.jsonl").read_bytes().splitlines(keepends=True)
    acks = [strict_json(line) for line in (directory / "persistence.acks.jsonl").read_bytes().splitlines()]
    if len(lines) != len(acks):
        raise MicrostructureIntegrityError("unacknowledged persistence interval")
    events = []
    for ordinal, (line, ack) in enumerate(zip(lines, acks), 1):
        event = strict_json(line)
        if ack["ingestion_ordinal"] != ordinal or ack["event_sha256"] != digest(line) or (
            ack["persistence_complete_monotonic_ns"] < event["dispatch_monotonic_ns"]
        ):
            raise MicrostructureIntegrityError("persistence provenance/hash mismatch")
        if event["session_id"] != manifest["session_id"] or event["monotonic_clock_id"] != manifest["clock_id"]:
            raise MicrostructureIntegrityError("manifest/session/clock binding mismatch")
        events.append(event)
    return manifest, events


def replay(events, policy=DEFAULT_POLICY):
    machine = ReplayMachine(policy)
    for event in events:
        machine.feed(event)
    return machine


def certify_session(directory: Path):
    manifest, events = load_session(directory)
    policy = Policy(**manifest["policy"])
    first, second = replay(events, policy), replay(events, policy)
    deterministic = first.canonical_result() == second.canonical_result()
    required = ("REST_SNAPSHOT", "DIFF_DEPTH", "TRADE", "AGGTRADE", "BOOK_TICKER", "CLOCK", "SYMBOL_RULES")
    checks = {
        "clean_session_close": bool(events and events[-1]["record_type"] == "SESSION_CLOSE" and first.clean_close),
        "all_required_streams_observed": all(first.counts.get(k, 0) > 0 for k in required),
        "snapshot_bridge": first.bridges > 0,
        "no_integrity_failures_or_resync": not first.failures and not first.resync_seen,
        "no_open_outage": not first.open_outages,
        "trade_aggregate_reconciliation": bool(first.aggregates) and all(
            a["status"] == "MATCHED" for a in first.aggregates.values()) and
            not (first.trades.keys() - first.reconciled_trade_ids),
        "valid_clock_and_metadata": first.clock_mapping is not None and first.metadata is not None,
        "deterministic_replay": deterministic,
    }
    status = "ENGINEERING_REPLAY_CERTIFIED" if all(checks.values()) else "ENGINEERING_CAPTURE_NOT_ELIGIBLE"
    return {"schema_version": "V12_ENGINEERING_CLOSURE_1", "status": status,
            "checks": checks, "failures": first.failures, "counts": first.counts,
            "replay_sha256": digest(first.canonical_result()),
            "artifact_sha256": {name: sha256_file(directory / name) for name in
                                ("session.manifest.json", "events.jsonl", "persistence.acks.jsonl")},
            "phase1_binding": manifest.get("phase1_binding"),
            "acquisition_authorized": False, "research_eligible": False,
            "economic_evaluation": False, "matching_engine_order_claimed": False}


@dataclass(frozen=True)
class MakerPlan:
    order: Order
    timing: Timing
    assumptions: Assumptions
    decision_after_ordinal: int
    cancel_utc: datetime | None = None
    cancel_monotonic_ns: int | None = None


def validate_limit(metadata, order):
    if metadata is None:
        return False
    filters = {f["filterType"]: f for f in metadata["info"]["filters"]}
    for value, ftype, minimum, maximum, increment in (
        (order.limit, "PRICE_FILTER", "minPrice", "maxPrice", "tickSize"),
        (order.quantity, "LOT_SIZE", "minQty", "maxQty", "stepSize"),
    ):
        f = filters[ftype]
        low, high, step = (D(f[k]) for k in (minimum, maximum, increment))
        with localcontext(ARITHMETIC):
            if (low and value < low) or (high and value > high) or (step and value % step):
                return False
    with localcontext(ARITHMETIC):
        notional = order.limit * order.quantity
    for ftype in ("MIN_NOTIONAL", "NOTIONAL"):
        if ftype in filters:
            f = filters[ftype]
            if notional < D(f["minNotional"]) or (D(f.get("maxNotional", "0")) and notional > D(f["maxNotional"])):
                return False
    return True


def maker_replay(events, plan: MakerPlan, policy=DEFAULT_POLICY):
    """No strategy or PnL. Process receipt order; future diagnostics never backfill.

    A failed overall integrity audit withdraws certification of the provisional
    ledger; it does NOT rewrite Phase 1's historical engine credits.
    """
    if (plan.cancel_utc is None) != (plan.cancel_monotonic_ns is None):
        raise MicrostructureIntegrityError("both cancel clocks required")
    if plan.cancel_monotonic_ns is not None and (
        plan.cancel_monotonic_ns < plan.order.decision_monotonic_ns + plan.timing.submission_latency_us * 1000
        or plan.cancel_utc < plan.order.decision_utc + timedelta(microseconds=plan.timing.submission_latency_us)
    ):
        raise MicrostructureIntegrityError("cancel must follow hypothetical working start")
    machine = ReplayMachine(policy)
    engine = None
    cancelled = False
    engine_trace = []
    for event in events:
        if engine is None and event["ingestion_ordinal"] == plan.decision_after_ordinal + 1:
            if machine.last_available_ns is not None and (
                machine.last_available_ns > plan.order.decision_monotonic_ns or
                machine.last_available_utc > plan.order.decision_utc
            ):
                raise MicrostructureIntegrityError("decision precedes processed prefix availability")
            if event["receipt_monotonic_ns"] < plan.order.decision_monotonic_ns:
                raise MicrostructureIntegrityError("decision prefix excludes already received records")
            mapping = machine.clock_mapping
            coverage = (machine.synced and not machine.failures and machine.last_trade_id is not None
                        and validate_limit(machine.metadata, plan.order) and mapping is not None
                        and plan.order.limit >= machine.known_bid_floor
                        and plan.timing.exchange_offset_min_us <= mapping["offset_min_us"]
                        and plan.timing.exchange_offset_max_us >= mapping["offset_max_us"])
            coverage = coverage and plan.order.session_id == machine.session_id and all(
                data is not None and 0 <= plan.order.decision_monotonic_ns - data["available_ns"] <= age * 1000
                for data, age in ((machine.metadata, policy.metadata_max_age_us),
                                  (mapping, policy.clock_max_age_us))) and (
                machine.book_available_ns is not None and 0 <= plan.order.decision_monotonic_ns - machine.book_available_ns <= policy.book_max_age_us * 1000)
            engine = ConditionalMakerBound(plan.order, plan.timing, plan.assumptions,
                                           initial_trade_id=machine.last_trade_id or 0,
                                           initial_depth_id=machine.book.last_update_id or 0,
                                           initial_ordinal=plan.decision_after_ordinal,
                                           coverage_established=bool(coverage))
            if coverage:
                engine.submit()
        before = len(machine.failures)
        machine.feed(event)
        if engine is None:
            continue
        now = _from_text(event["processing_available_utc"])
        ns = event["dispatch_monotonic_ns"]
        if plan.cancel_monotonic_ns is not None and not cancelled and ns >= plan.cancel_monotonic_ns:
            if engine.state in (State.SUBMISSION_PENDING, State.CONDITIONALLY_WORKING, State.PARTIALLY_FILLED):
                engine.advance(as_of=plan.cancel_utc, monotonic_ns=plan.cancel_monotonic_ns)
                if engine.state in (State.CONDITIONALLY_WORKING, State.PARTIALLY_FILLED):
                    engine.request_cancel(as_of=plan.cancel_utc, monotonic_ns=plan.cancel_monotonic_ns)
            cancelled = True
        receipt = machine.receipt(event)
        kind = event["record_type"]
        stale = any(data is None or ns - data["available_ns"] > max_age * 1000 for data, max_age in (
            (machine.clock_mapping, policy.clock_max_age_us), (machine.metadata, policy.metadata_max_age_us)))
        mapping = machine.clock_mapping
        incompatible = (mapping is None or not validate_limit(machine.metadata, plan.order) or
                        machine.known_bid_floor is None or plan.order.limit < machine.known_bid_floor or
                        plan.timing.exchange_offset_min_us > mapping["offset_min_us"] or
                        plan.timing.exchange_offset_max_us < mapping["offset_max_us"])
        if len(machine.failures) > before or stale or incompatible or (machine.book_available_ns is None or
            ns - machine.book_available_ns > policy.book_max_age_us * 1000):
            engine.observe_boundary("public integrity/coverage/freshness unavailable", receipt,
                                    as_of=now, monotonic_ns=ns)
        elif kind == "REST_SNAPSHOT" or kind in ("RESYNC_START", "GAP_DETECTED"):
            engine.observe_boundary(kind, receipt, as_of=now, monotonic_ns=ns)
        elif kind == "TRADE":
            engine.observe_trade(machine.individual(event), as_of=now, monotonic_ns=ns)
        elif kind == "DIFF_DEPTH":
            p = event["payload"]
            engine.observe_depth(p["U"], p["u"], receipt, as_of=now, monotonic_ns=ns)
        else:
            engine.observe_auxiliary(receipt, as_of=now, monotonic_ns=ns)
        engine_trace.append({"ordinal": event["ingestion_ordinal"], "state": engine.state.value,
                             "provisional_credit": str(engine.credited), "reason": engine.reason})
    if engine is None:
        raise MicrostructureIntegrityError("decision prefix outside capture")
    reconciled = (bool(machine.aggregates) and all(a["status"] == "MATCHED" for a in machine.aggregates.values())
                  and not (machine.trades.keys() - machine.reconciled_trade_ids))
    valid = (not machine.failures and reconciled and machine.clean_close and
             engine.state != State.INDETERMINATE)
    return {"scope": "CONDITIONAL_QUANTITY_ONLY", "state": engine.state.value,
            "certified_conditional_credit": str(engine.credited) if valid else None,
            "unconditional_public_credit": "0", "provisional_ledger": [asdict(c) for c in engine.credits],
            "trace": engine_trace, "integrity_valid": valid, "failures": machine.failures,
            "economic_evaluation": False}
