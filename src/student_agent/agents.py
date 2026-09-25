"""Specialist agents. Each agent reads MCP evidence through its scoped gateway and returns a
deterministic finding; no agent invents data that is missing from evidence.

An order_id may carry several purchase *instances* (history rows). Specialists report their
evidence per instance; the coordinator later picks the instance the complaint is about."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any

from .state import CaseState, ResultMessage, ScopedGateway, TaskMessage

ORDER_TS = "order_purchase_timestamp"
HOUR = timedelta(hours=1)
# Offset windows (relative to the instance purchase time) used to link lifecycle records.
LINK_WINDOWS: dict[str, tuple[timedelta, timedelta]] = {
    "payment": (timedelta(0), 12 * HOUR),
    "refund": (240 * HOUR, 288 * HOUR),
    "limit": (48 * HOUR, 96 * HOUR),
}


def parse_ts(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def money(value: Any) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0


def dedupe(rows: list[Any]) -> list[dict[str, Any]]:
    """Drop exact duplicate records (identical in every field)."""
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            seen.setdefault(json.dumps(row, sort_keys=True, default=str), row)
    return list(seen.values())


@dataclass
class Instance:
    key: str
    start: datetime | None
    row: dict[str, Any]
    copies: int  # identical history rows collapsed into this instance


class Instances:
    """Purchase instances of one order_id and the rules linking records to them."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        signature = lambda row: json.dumps(row, sort_keys=True, default=str)  # noqa: E731
        copies: dict[str, int] = {}
        for row in rows:
            copies[signature(row)] = copies.get(signature(row), 0) + 1
        items = [Instance(str(row.get(ORDER_TS) or f"row-{index}"), parse_ts(row.get(ORDER_TS)),
                          row, copies[signature(row)])
                 for index, row in enumerate(dedupe(rows))]
        self.items = sorted(items, key=lambda i: (i.start is None, i.start or datetime.min))

    def by_key(self, key: str) -> Instance:
        return next(i for i in self.items if i.key == key)

    def owner(self, value: Any, kind: str) -> str | None:
        if len(self.items) == 1:
            return self.items[0].key
        ts = parse_ts(value)
        if ts is None:
            return None
        dated = [i for i in self.items if i.start is not None]
        if kind == "shipment_event":
            delivered = {i.key: parse_ts(i.row.get("order_delivered_customer_date"))
                         for i in dated}
            same_day = [key for key, day in delivered.items() if day and day.date() == ts.date()]
            if len(same_day) == 1:
                return same_day[0]
        window = LINK_WINDOWS.get(kind)
        if window:
            low, high = window
            hits = [i for i in dated if low <= ts - i.start <= high]  # type: ignore[operator]
            if len(hits) == 1:
                return hits[0].key
        before = [i for i in dated if i.start <= ts]  # type: ignore[operator]
        return (before[-1] if before else dated[0]).key if dated else None

    def group(self, rows: list[Any], field: str, kind: str) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {i.key: [] for i in self.items}
        for row in dedupe(rows):
            key = self.owner(row.get(field), kind)
            if key in grouped:
                grouped[key].append(row)
        return grouped


def _unique(values: Any) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _result(task: TaskMessage, status: str, finding: dict[str, Any], code: str,
            refs: list[str]) -> ResultMessage:
    return ResultMessage(task.case_id, task.correlation_id, task.recipient, status, finding,
                         code, refs)


# --------------------------------------------------------------------------- entity
async def entity_agent(task: TaskMessage, gw: ScopedGateway, state: CaseState) -> ResultMessage:
    case = state.case
    request = case.get("customer_request") or {}
    candidates: list[str] = list(dict.fromkeys(case.get("candidate_order_ids") or []))
    claimed = request.get("claimed_order_id")
    customer_hint = case.get("customer_unique_id_hint")
    refs: list[str] = []

    history_rows: list[dict[str, Any]] = []
    customer_id = None
    if customer_hint:
        history = await gw.try_call("get_customer_history", customer_unique_id=customer_hint)
        if history:
            refs.append(history["evidence_ref"])
            data = history.get("data") or {}
            customer_id = data.get("customer_unique_id") or customer_hint
            history_rows = [row for row in data.get("orders") or [] if isinstance(row, dict)]
    history_ids = _unique(row.get("order_id") for row in history_rows)

    ordered = sorted(candidates, key=lambda oid: oid != claimed)
    accepted = [oid for oid in ordered if oid in history_ids]
    order_row = None
    if not accepted:
        # Fall back to the authoritative order row, only for the claimed/first candidate.
        for oid in ordered[:1]:
            evidence = await gw.try_call("get_order", order_id=oid)
            if evidence and isinstance(evidence.get("data"), dict):
                refs.append(evidence["evidence_ref"])
                order_row = evidence["data"]
                accepted = [oid]
    if not accepted:
        finding = {"status": "not_found", "resolved": [], "rejected": candidates,
                   "customer_unique_id": customer_id, "related": history_ids, "confidence": 0.3}
        return _result(task, "insufficient", finding, "ENTITY_NOT_FOUND", refs)

    order_id = accepted[0]
    if order_row is None:
        evidence = await gw.try_call("get_order", order_id=order_id)
        if evidence and isinstance(evidence.get("data"), dict):
            refs.append(evidence["evidence_ref"])
            order_row = evidence["data"]

    rows = [row for row in history_rows if row.get("order_id") == order_id]
    if not rows and order_row:
        rows = [order_row]
    confidence = 0.95 if order_id == claimed else 0.85
    finding = {
        "status": "resolved" if rows else "ambiguous",
        "order_id": order_id,
        "resolved": [order_id],
        "rejected": [oid for oid in candidates if oid != order_id],
        "customer_unique_id": customer_id,
        "related": history_ids,
        "rows": rows,
        "order_row": order_row,
        "confidence": confidence,
    }
    code = "ENTITY_RESOLVED" if rows else "ENTITY_AMBIGUOUS"
    return _result(task, "ok" if rows else "insufficient", finding, code, refs)


# --------------------------------------------------------------------------- order/product
async def order_agent(task: TaskMessage, gw: ScopedGateway, state: CaseState) -> ResultMessage:
    order_id: str = task.payload["order_id"]
    instances: Instances = task.payload["instances"]
    refs: list[str] = []
    if task.task == "fetch_sellers":
        evidence = await gw.try_call("get_sellers", order_id=order_id)
        if not evidence:
            return _result(task, "insufficient", {}, "NO_RECORDS", refs)
        refs.append(evidence["evidence_ref"])
        ids = _unique(row.get("seller_id") for row in evidence.get("data") or []
                      if isinstance(row, dict))
        return _result(task, "ok", {"seller_records": ids}, "SELLERS_CONFIRMED", refs)

    items_ev = await gw.try_call("get_order_items", order_id=order_id)
    grouped: dict[str, list[dict[str, Any]]] = {i.key: [] for i in instances.items}
    if items_ev:
        refs.append(items_ev["evidence_ref"])
        grouped = instances.group(list(items_ev.get("data") or []), "shipping_limit_date",
                                  "limit")
    product_ev = await gw.try_call("get_product_context", order_id=order_id)
    categories: list[str] = []
    if product_ev:
        refs.append(product_ev["evidence_ref"])
        categories = _unique(row.get("category_name_english")
                             for row in product_ev.get("data") or [] if isinstance(row, dict))
    by_instance = {
        key: {
            "item_ids": _unique(i.get("order_item_id") for i in items),
            "seller_ids": _unique(i.get("seller_id") for i in items),
            "price_total": round(sum(money(i.get("price")) for i in items), 2),
            "freight_total": round(sum(money(i.get("freight_value")) for i in items), 2),
            "shipping_limits": {i.get("seller_id"): i.get("shipping_limit_date") for i in items},
        }
        for key, items in grouped.items()
    }
    found = any(v["item_ids"] for v in by_instance.values())
    return _result(task, "ok" if found else "insufficient",
                   {"by_instance": by_instance, "categories": categories},
                   "ITEMS_SCOPED" if found else "NO_RECORDS", refs)


# --------------------------------------------------------------------------- shipment
def shipment_verdict(row: dict[str, Any], limits: dict[str, datetime | None],
                     events: list[dict[str, Any]]) -> dict[str, Any]:
    purchase = parse_ts(row.get(ORDER_TS))
    carrier = parse_ts(row.get("order_delivered_carrier_date"))
    delivered = parse_ts(row.get("order_delivered_customer_date"))
    estimate = parse_ts(row.get("order_estimated_delivery_date"))
    confirmed = [e for e in events if e.get("status", "confirmed") == "confirmed"]
    late_sellers = [seller for seller, limit in limits.items()
                    if seller and limit and carrier and carrier > limit]
    event_types = {e.get("event_type") for e in confirmed}
    late_actors = {e.get("actor") for e in confirmed if e.get("event_type") == "delivered_late"}

    verdict = "insufficient_evidence"
    if "lost" in event_types:
        verdict = "lost"
    elif "returned" in event_types:
        verdict = "returned"
    elif delivered and estimate:
        if delivered <= estimate:
            verdict = "on_time"
        elif late_sellers:
            verdict = "seller_delay"
        else:
            verdict = "logistics_delay"

    disagreement = False
    if late_actors and verdict in {"on_time", "seller_delay", "logistics_delay"}:
        event_verdict = "seller_delay" if "seller" in late_actors else "logistics_delay"
        if event_verdict != verdict:
            disagreement = True
            verdict = event_verdict  # confirmed lifecycle event outranks derived dates
            if verdict == "seller_delay" and not late_sellers:
                late_sellers = [s for s in limits if s]
    return {
        "verdict": verdict,
        "late_seller_ids": late_sellers if verdict == "seller_delay" else [],
        "timeline_complete": all([purchase, carrier, delivered, estimate]),
        "rule_event_disagreement": disagreement,
    }


async def shipment_agent(task: TaskMessage, gw: ScopedGateway, state: CaseState) -> ResultMessage:
    order_id: str = task.payload["order_id"]
    instances: Instances = task.payload["instances"]
    item_limits: dict[str, dict[str, Any]] = task.payload.get("shipping_limits") or {}
    evidence = None
    if task.payload.get("fetch_summary", True):
        evidence = await gw.try_call("get_shipment_summary", order_id=order_id)
    # Without the summary, the verdict is derived from history timestamps and item limits.
    refs = [evidence["evidence_ref"]] if evidence else []
    data = (evidence or {}).get("data") or {}

    limit_rows = instances.group(list(data.get("shipping_limits") or []), "shipping_limit_at",
                                 "limit")
    event_rows = instances.group(list(data.get("events") or []), "event_at", "shipment_event")
    by_instance = {}
    for inst in instances.items:
        limits = {r.get("seller_id"): parse_ts(r.get("shipping_limit_at"))
                  for r in limit_rows.get(inst.key, [])}
        if not limits:
            limits = {s: parse_ts(v) for s, v in (item_limits.get(inst.key) or {}).items()}
        by_instance[inst.key] = shipment_verdict(inst.row, limits, event_rows.get(inst.key, []))
    finding = {"by_instance": by_instance,
               "summary_delivered_at": data.get("delivered_customer_at") if data else None,
               "has_summary": bool(data)}
    code = "TIMELINE_SCOPED" if data else "TIMELINE_FROM_HISTORY"
    return _result(task, "ok", finding, code, refs)


# --------------------------------------------------------------------------- payment/refund
async def payment_agent(task: TaskMessage, gw: ScopedGateway, state: CaseState) -> ResultMessage:
    order_id: str = task.payload["order_id"]
    instances: Instances = task.payload["instances"]
    refs: list[str] = []
    timeline = await gw.try_call("get_payment_timeline", order_id=order_id)
    events: dict[str, list[dict[str, Any]]] = {i.key: [] for i in instances.items}
    if timeline:
        refs.append(timeline["evidence_ref"])
        raw = (timeline.get("data") or {}).get("events") or []
        events = instances.group(list(raw), "event_at", "payment")
    refund_ev = None
    if task.payload.get("fetch_refunds", True):
        refund_ev = await gw.try_call("get_refund_timeline", order_id=order_id)
    refunds: dict[str, list[dict[str, Any]]] = {i.key: [] for i in instances.items}
    if refund_ev:
        refs.append(refund_ev["evidence_ref"])
        data = refund_ev.get("data") or {}
        raw = data.get("events") if isinstance(data, dict) else data
        refunds = instances.group(list(raw or []), "event_at", "refund")
    by_instance = {i.key: {"events": events.get(i.key, []), "refunds": refunds.get(i.key, [])}
                   for i in instances.items}
    finding = {"by_instance": by_instance, "has_timeline": bool(timeline)}
    return _result(task, "ok" if timeline else "insufficient", finding,
                   "PAYMENTS_SCOPED" if timeline else "NO_RECORDS", refs)


REFUND_ISSUES = {"refund_pending", "refund_failed"}


def payment_view(raw: dict[str, Any], expected: float | None, has_timeline: bool,
                 primary: str | None = None) -> dict[str, Any]:
    """Payment verdict and totals for one instance, optionally scoped to the primary scenario."""
    events, refunds = raw.get("events") or [], raw.get("refunds") or []
    captures = [money(e.get("amount_brl")) for e in events
                if e.get("event_type") == "captured" and e.get("status") == "confirmed"]
    refund_amounts = {money(e.get("amount_brl")) for e in refunds}
    if primary is not None and primary not in REFUND_ISSUES and refund_amounts:
        # Captures settled by a refund flow belong to that flow, not to this scenario.
        captures = [amt for amt in captures if amt not in refund_amounts]
        refunds = []
    elif primary in REFUND_ISSUES:
        captures = [amt for amt in captures if amt in refund_amounts] or captures
    mismatches = [money(e.get("amount_brl")) for e in events
                  if e.get("event_type") == "reconciliation_mismatch"
                  and e.get("status") != "resolved"]
    failed = [money(e.get("amount_brl")) for e in refunds if e.get("status") == "failed"]
    pending = [money(e.get("amount_brl")) for e in refunds if e.get("status") == "pending"]
    refunded = [money(e.get("amount_brl")) for e in refunds
                if e.get("status") in {"completed", "succeeded", "refunded", "confirmed"}]
    captured_total = round(float(sum(captures)), 2)
    refunded_total = round(float(sum(refunded)), 2)
    repeated = sorted({amt for amt in captures if captures.count(amt) > 1})
    duplicate = bool(repeated) and expected is not None and captured_total > expected + 0.01
    # A split is judged on captures that are not part of a refund flow.
    split_parts = _split_subset([amt for amt in captures if amt not in refund_amounts]
                                or captures, expected)
    split_valid = bool(split_parts)
    if primary == "valid_split_payment" and split_parts:
        captures = split_parts
        captured_total = round(float(sum(captures)), 2)
        repeated, duplicate = [], False

    if failed:
        verdict = "refund_failed"
    elif pending:
        verdict = "refund_pending"
    elif duplicate:
        verdict = "duplicate_capture"
    elif mismatches:
        verdict = "capture_mismatch"
    elif refunded:
        verdict = "refunded"
    elif captures:
        verdict = "reconciled"
    else:
        verdict = "insufficient_evidence"
    return {
        "verdict": verdict,
        "captured_total_brl": captured_total if has_timeline else None,
        "refunded_total_brl": refunded_total if has_timeline else None,
        "refundable_total_brl": round(max(0.0, captured_total - refunded_total), 2)
        if has_timeline else None,
        "failed_refund_brl": round(sum(failed), 2),
        "pending_refund_brl": round(sum(pending), 2),
        "mismatch_brl": round(sum(mismatches), 2),
        "duplicate_brl": repeated[0] if duplicate else 0.0,
        "split_valid": split_valid,
    }


def _split_subset(captures: list[float], expected: float | None) -> list[float]:
    """Smallest group of ≥2 captures that settles exactly the expected order total."""
    if expected is None or len(captures) < 2 or len(captures) > 8:
        return []
    for size in range(2, len(captures) + 1):
        for combo in combinations(captures, size):
            if abs(sum(combo) - expected) <= 0.01:
                return list(combo)
    return []


# --------------------------------------------------------------------------- policy
async def fetch_policy(task: TaskMessage, gw: ScopedGateway, state: CaseState) -> ResultMessage:
    evidence = await gw.try_call("get_policy", policy_version=task.payload["policy_version"])
    if not evidence:
        return _result(task, "insufficient", {"rules": {}}, "POLICY_UNAVAILABLE", [])
    rules = (evidence.get("data") or {}).get("rules") or {}
    return _result(task, "ok", {"rules": rules}, "POLICY_LOADED", [evidence["evidence_ref"]])


def classify(order_status: str | None, shipment: dict[str, Any],
             payment: dict[str, Any]) -> list[str]:
    """Issues supported by evidence, most specific first; valid_split_payment last."""
    issues: list[str] = []
    captured = payment.get("captured_total_brl") or 0.0
    if order_status == "canceled" and captured > 0:
        issues.append("canceled_order_paid")
    if order_status == "unavailable" and captured > 0:
        issues.append("unavailable_order_paid")
    verdict = payment.get("verdict")
    if verdict == "refund_failed":
        issues.append("refund_failed")
    if verdict == "refund_pending":
        issues.append("refund_pending")
    if verdict == "duplicate_capture":
        issues.append("duplicate_charge")
    if payment.get("mismatch_brl"):
        issues.append("payment_mismatch")
    if shipment.get("verdict") == "seller_delay":
        issues.append("late_delivery_seller")
    if shipment.get("verdict") == "logistics_delay":
        issues.append("late_delivery_logistics")
    if payment.get("split_valid"):
        issues.append("valid_split_payment")
    return issues


def evidence_amount(issue: str, order: dict[str, Any], payment: dict[str, Any]) -> float | None:
    refundable = payment.get("refundable_total_brl")
    if issue in {"canceled_order_paid", "unavailable_order_paid"}:
        return refundable
    if issue == "refund_failed":
        return payment.get("failed_refund_brl")
    if issue == "duplicate_charge":
        return payment.get("duplicate_brl")
    if issue == "payment_mismatch":
        return payment.get("mismatch_brl")
    if issue in {"late_delivery_seller", "late_delivery_logistics"}:
        freight = order.get("freight_total") or 0.0
        return round(min(freight, refundable), 2) if refundable is not None else freight
    return 0.0
