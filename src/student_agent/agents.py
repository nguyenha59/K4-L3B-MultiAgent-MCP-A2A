"""Specialist agents. Each agent reads MCP evidence through its scoped gateway and returns a
deterministic finding; no agent invents data that is missing from evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .state import CaseState, ResultMessage, ScopedGateway, TaskMessage

ORDER_TS = "order_purchase_timestamp"


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


class Instances:
    """Assigns timestamped rows of one order_id to the purchase instance they belong to."""

    def __init__(self, rows: list[dict[str, Any]], anchor: dict[str, Any] | None) -> None:
        dated = [(parse_ts(row.get(ORDER_TS)), row) for row in rows]
        self._starts = sorted((ts for ts, _ in dated if ts is not None))
        self.anchor = anchor
        self.anchor_start = parse_ts(anchor.get(ORDER_TS)) if anchor else None

    def owner(self, ts: datetime | None) -> datetime | None:
        if ts is None:
            return None
        owners = [start for start in self._starts if start <= ts]
        return owners[-1] if owners else None

    def in_anchor(self, value: Any) -> bool:
        if self.anchor_start is None:
            return True  # single unanchored instance: everything belongs to it
        if len(self._starts) <= 1:
            return True
        return self.owner(parse_ts(value)) == self.anchor_start


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
    opened_at = parse_ts(case.get("opened_at"))
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
    history_ids = list(dict.fromkeys(row.get("order_id") for row in history_rows))

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
    rejected = [oid for oid in candidates if oid != order_id]
    if order_row is None:
        evidence = await gw.try_call("get_order", order_id=order_id)
        if evidence and isinstance(evidence.get("data"), dict):
            refs.append(evidence["evidence_ref"])
            order_row = evidence["data"]

    rows = [row for row in history_rows if row.get("order_id") == order_id]
    if not rows and order_row:
        rows = [order_row]
    anchor, anchored_before_open = _choose_anchor(rows, opened_at)
    confidence = 0.95 if anchored_before_open else 0.7
    if order_id != claimed:
        confidence -= 0.1
    status = "resolved" if anchor else "ambiguous"
    finding = {
        "status": status,
        "order_id": order_id,
        "resolved": [order_id],
        "rejected": rejected,
        "customer_unique_id": customer_id,
        "related": history_ids,
        "rows": rows,
        "anchor": anchor,
        "order_row": order_row,
        "confidence": round(confidence, 2),
    }
    code = "ENTITY_RESOLVED" if status == "resolved" else "ENTITY_AMBIGUOUS"
    return _result(task, "ok" if anchor else "insufficient", finding, code, refs)


def _choose_anchor(rows: list[dict[str, Any]],
                   opened_at: datetime | None) -> tuple[dict[str, Any] | None, bool]:
    dated = [(parse_ts(row.get(ORDER_TS)), row) for row in rows]
    dated = [(ts, row) for ts, row in dated if ts is not None]
    if not dated:
        return (rows[0] if rows else None), False
    if opened_at is None:
        return max(dated, key=lambda item: item[0])[1], False
    before = [(ts, row) for ts, row in dated if ts <= opened_at]
    if before:
        return max(before, key=lambda item: item[0])[1], True
    return min(dated, key=lambda item: abs(item[0] - opened_at))[1], False


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
        ids = [row.get("seller_id") for row in evidence.get("data") or [] if isinstance(row, dict)]
        return _result(task, "ok", {"seller_records": ids}, "SELLERS_CONFIRMED", refs)

    items_ev = await gw.try_call("get_order_items", order_id=order_id)
    items: list[dict[str, Any]] = []
    if items_ev:
        refs.append(items_ev["evidence_ref"])
        rows = [row for row in items_ev.get("data") or [] if isinstance(row, dict)]
        items = [row for row in rows if instances.in_anchor(row.get("shipping_limit_date"))]
    product_ev = await gw.try_call("get_product_context", order_id=order_id)
    categories: list[str] = []
    if product_ev:
        refs.append(product_ev["evidence_ref"])
        for row in product_ev.get("data") or []:
            if isinstance(row, dict) and row.get("category_name_english"):
                categories.append(row["category_name_english"])
    finding = {
        "item_ids": _unique(i.get("order_item_id") for i in items),
        "seller_ids": _unique(i.get("seller_id") for i in items),
        "price_total": round(sum(money(i.get("price")) for i in items), 2),
        "freight_total": round(sum(money(i.get("freight_value")) for i in items), 2),
        "shipping_limits": {i.get("seller_id"): i.get("shipping_limit_date") for i in items},
        "categories": list(dict.fromkeys(categories)),
    }
    status = "ok" if items else "insufficient"
    return _result(task, status, finding, "ITEMS_SCOPED" if items else "NO_RECORDS", refs)


# --------------------------------------------------------------------------- shipment
async def shipment_agent(task: TaskMessage, gw: ScopedGateway, state: CaseState) -> ResultMessage:
    order_id: str = task.payload["order_id"]
    instances: Instances = task.payload["instances"]
    anchor = instances.anchor or {}
    evidence = await gw.try_call("get_shipment_summary", order_id=order_id)
    refs = [evidence["evidence_ref"]] if evidence else []
    data = (evidence or {}).get("data") or {}

    limits = {}
    for row in data.get("shipping_limits") or []:
        if isinstance(row, dict) and instances.in_anchor(row.get("shipping_limit_at")):
            limits[row.get("seller_id")] = parse_ts(row.get("shipping_limit_at"))
    if not limits:
        for seller, value in (task.payload.get("shipping_limits") or {}).items():
            limits[seller] = parse_ts(value)
    events = [e for e in data.get("events") or []
              if isinstance(e, dict) and instances.in_anchor(e.get("event_at"))
              and e.get("status", "confirmed") == "confirmed"]

    purchase = parse_ts(anchor.get(ORDER_TS))
    carrier = parse_ts(anchor.get("order_delivered_carrier_date"))
    delivered = parse_ts(anchor.get("order_delivered_customer_date"))
    estimate = parse_ts(anchor.get("order_estimated_delivery_date"))
    status = anchor.get("order_status")
    late_sellers = [seller for seller, limit in limits.items()
                    if seller and limit and carrier and carrier > limit]
    event_types = {e.get("event_type") for e in events}
    late_actors = {e.get("actor") for e in events if e.get("event_type") == "delivered_late"}

    rule_verdict = "insufficient_evidence"
    if "lost" in event_types:
        rule_verdict = "lost"
    elif "returned" in event_types:
        rule_verdict = "returned"
    elif delivered and estimate:
        if delivered <= estimate:
            rule_verdict = "on_time"
        elif late_sellers:
            rule_verdict = "seller_delay"
        else:
            rule_verdict = "logistics_delay"

    verdict = rule_verdict
    disagreement = False
    if late_actors and rule_verdict in {"on_time", "seller_delay", "logistics_delay"}:
        event_verdict = "seller_delay" if "seller" in late_actors else "logistics_delay"
        if event_verdict != rule_verdict:
            disagreement = True
            verdict = event_verdict  # confirmed lifecycle event outranks derived dates
            if event_verdict == "seller_delay" and not late_sellers:
                late_sellers = [s for s in limits if s]
            if event_verdict == "logistics_delay":
                late_sellers = []
    if verdict != "seller_delay":
        late_sellers = []

    summary_delivered = parse_ts(data.get("delivered_customer_at"))
    finding = {
        "verdict": verdict,
        "late_seller_ids": late_sellers,
        "timeline_complete": all([purchase, carrier, delivered, estimate]),
        "order_status": status,
        "rule_event_disagreement": disagreement,
        "summary_instance_mismatch": bool(data) and summary_delivered != delivered,
    }
    ok = verdict != "insufficient_evidence"
    return _result(task, "ok" if ok else "insufficient", finding,
                   verdict.upper(), refs)


# --------------------------------------------------------------------------- payment/refund
async def payment_agent(task: TaskMessage, gw: ScopedGateway, state: CaseState) -> ResultMessage:
    order_id: str = task.payload["order_id"]
    instances: Instances = task.payload["instances"]
    expected = task.payload.get("expected_total")
    refs: list[str] = []
    timeline = await gw.try_call("get_payment_timeline", order_id=order_id)
    events: list[dict[str, Any]] = []
    if timeline:
        refs.append(timeline["evidence_ref"])
        raw = (timeline.get("data") or {}).get("events") or []
        events = [e for e in raw if isinstance(e, dict) and instances.in_anchor(e.get("event_at"))]
    refund_ev = await gw.try_call("get_refund_timeline", order_id=order_id)
    refunds: list[dict[str, Any]] = []
    if refund_ev:
        refs.append(refund_ev["evidence_ref"])
        data = refund_ev.get("data") or {}
        raw = data.get("events") if isinstance(data, dict) else data
        refunds = [e for e in raw or []
                   if isinstance(e, dict) and instances.in_anchor(e.get("event_at"))]

    captures = [money(e.get("amount_brl")) for e in events
                if e.get("event_type") == "captured" and e.get("status") == "confirmed"]
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
    split_valid = (len(captures) > 1 and expected is not None
                   and abs(captured_total - expected) <= 0.01)

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
    finding = {
        "verdict": verdict,
        "captured_total_brl": captured_total if timeline else None,
        "refunded_total_brl": refunded_total if timeline else None,
        "refundable_total_brl": round(max(0.0, captured_total - refunded_total), 2)
        if timeline else None,
        "failed_refund_brl": round(sum(failed), 2),
        "pending_refund_brl": round(sum(pending), 2),
        "mismatch_brl": round(sum(mismatches), 2),
        "duplicate_brl": repeated[0] if duplicate else 0.0,
        "split_valid": split_valid,
        "capture_count": len(captures),
    }
    ok = verdict != "insufficient_evidence"
    return _result(task, "ok" if ok else "insufficient", finding, verdict.upper(), refs)


# --------------------------------------------------------------------------- policy
async def fetch_policy(task: TaskMessage, gw: ScopedGateway, state: CaseState) -> ResultMessage:
    evidence = await gw.try_call("get_policy", policy_version=task.payload["policy_version"])
    if not evidence:
        return _result(task, "insufficient", {"rules": {}}, "POLICY_UNAVAILABLE", [])
    rules = (evidence.get("data") or {}).get("rules") or {}
    return _result(task, "ok", {"rules": rules}, "POLICY_LOADED", [evidence["evidence_ref"]])


def classify(order_status: str | None, shipment: dict[str, Any],
             payment: dict[str, Any]) -> list[str]:
    """Issues supported by evidence, most specific first."""
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
    if not issues and payment.get("split_valid"):
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
