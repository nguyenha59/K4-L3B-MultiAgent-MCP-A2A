"""Offline workflow tests on synthetic evidence (no competition data, no network)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from student_agent.contracts import Contracts
from student_agent.state import CaseState
from student_agent.trace import TraceWriter
from student_agent.workflow import solve_case

ROOT = Path(__file__).resolve().parents[1]
ORDER = "order-synthetic-0001"
SELLER = "seller-synthetic"


def _ev(n: int, domain: str, data: Any) -> dict[str, Any]:
    return {"schema_version": "day09-mcp-evidence-v1", "evidence_ref": f"ev_synthetic_ref_{n:08d}",
            "result_hash": "sha256:" + "0" * 64, "domain": domain, "data": data, "warnings": []}


def _row(purchase: str, carrier: str, delivered: str, estimate: str) -> dict[str, Any]:
    return {"order_id": ORDER, "order_status": "delivered",
            "order_purchase_timestamp": purchase, "order_delivered_carrier_date": carrier,
            "order_delivered_customer_date": delivered, "order_estimated_delivery_date": estimate}


# The relevant instance (before opened_at) ships after the seller limit; the later one is noise.
RELEVANT = _row("2018-01-01T09:00:00-03:00", "2018-01-08T09:00:00-03:00",
                "2018-01-15T09:00:00-03:00", "2018-01-12T09:00:00-03:00")
NOISE = _row("2018-06-01T09:00:00-03:00", "2018-06-02T09:00:00-03:00",
             "2018-06-05T09:00:00-03:00", "2018-06-10T09:00:00-03:00")
RESPONSES = {
    "get_customer_history": _ev(1, "customer", {"customer_unique_id": "cust-1",
                                                "orders": [NOISE, RELEVANT]}),
    "get_order": _ev(2, "order", NOISE),
    "get_order_items": _ev(3, "item", [
        {"order_item_id": "item-1", "seller_id": SELLER, "price": "80.00",
         "freight_value": "20.00", "shipping_limit_date": "2018-01-04T09:00:00-03:00"},
        {"order_item_id": "item-1", "seller_id": SELLER, "price": "80.00",
         "freight_value": "12.00", "shipping_limit_date": "2018-06-04T09:00:00-03:00"}]),
    "get_product_context": _ev(4, "product", []),
    "get_shipment_summary": _ev(5, "shipment", {"shipping_limits": [], "events": []}),
    "get_payment_timeline": _ev(6, "payment", {"events": [
        {"event_at": "2018-01-01T10:00:00-03:00", "event_type": "captured",
         "amount_brl": "100.00", "status": "confirmed"}]}),
    "get_sellers": _ev(7, "seller", [{"seller_id": SELLER}]),
    "get_policy": _ev(8, "policy", {"rules": {"late_delivery_seller": {
        "case_status": "action_required", "recommended_action": "refund_freight",
        "refund_brl": 20.0, "responsible_parties": [{"party_type": "seller", "party_id": "x"}]}}}),
}


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call(self, tool: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        self.calls.append(tool)
        if tool == "get_order" and arguments.get("order_id") != ORDER:
            raise RuntimeError("unknown order")
        if tool not in RESPONSES:
            raise RuntimeError("no records")
        return RESPONSES[tool]


CASE = {
    "case_id": "SYN_CASE_001",
    "opened_at": "2018-01-20T09:00:00-03:00",
    "customer_request": {"claimed_order_id": ORDER, "claims": [
        {"claim_id": "c-a", "topic": "late_delivery_logistics"},
        {"claim_id": "c-b", "topic": "requested_full_refund"}]},
    "policy_version": "EC_POLICY_V2",
    "candidate_order_ids": [ORDER, "candidate-decoy"],
    "customer_unique_id_hint": "cust-1",
}


def test_workflow_anchors_instance_and_blames_seller(tmp_path: Path) -> None:
    contracts = Contracts(ROOT / "contracts" / "schemas")
    trace = TraceWriter(tmp_path / "trace.jsonl", contracts)
    gateway = FakeGateway()
    output = asyncio.run(solve_case(CASE, gateway, trace))  # type: ignore[arg-type]

    contracts.validate_output(output, "synthetic")
    assert output["assessment"]["primary_issue"] == "late_delivery_seller"
    assert output["shipment_analysis"]["late_seller_ids"] == [SELLER]
    assert output["entity_resolution"]["rejected_candidates"] == ["candidate-decoy"]
    assert output["financial_resolution"]["recommended_refund_brl"] == 20.0
    assert output["root_cause_analysis"]["responsible_parties"][0]["party_id"] == SELLER
    assert output["claim_assessments"][0]["verdict"] == "unsupported"
    assert len(gateway.calls) == len(set(gateway.calls)) <= 12

    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    types = [event["event_type"] for event in events]
    for required in ("task_assigned", "handoff", "tool_result_consumed", "policy_decided",
                     "verification_completed"):
        assert required in types
    consumed = {ref for e in events if e["event_type"] == "tool_result_consumed"
                for ref in e["evidence_refs"]}
    assert set(output["evidence_refs"]) <= consumed


def test_agents_cannot_call_tools_outside_their_scope(tmp_path: Path) -> None:
    contracts = Contracts(ROOT / "contracts" / "schemas")
    state = CaseState(CASE, FakeGateway(), TraceWriter(tmp_path / "t.jsonl", contracts))
    with pytest.raises(PermissionError):
        asyncio.run(state.scoped("shipment-agent").call("get_policy", policy_version="x"))
