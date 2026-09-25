"""L3B coordinator: hub-and-spoke multi-agent workflow (see ARCHITECTURE.md)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import cache
from pathlib import Path
from typing import Any

from . import OUTPUT_SCHEMA_VERSION
from .agents import (
    Instances,
    classify,
    entity_agent,
    evidence_amount,
    fetch_policy,
    order_agent,
    parse_ts,
    payment_agent,
    shipment_agent,
)
from .contracts import ContractError, Contracts
from .mcp_gateway import EvidenceGateway
from .state import TASK_TIMEOUT_S, CaseState, ResultMessage, TaskMessage
from .trace import TraceWriter

Agent = Callable[[TaskMessage, Any, CaseState], Awaitable[ResultMessage]]

SELLER_ISSUES = {"late_delivery_seller", "unavailable_order_paid"}
PAYMENT_ISSUES = {"canceled_order_paid", "unavailable_order_paid", "valid_split_payment",
                  "payment_mismatch", "duplicate_charge"}
REFUND_ISSUES = {"refund_pending", "refund_failed"}
SHIPMENT_ISSUES = {"late_delivery_seller", "late_delivery_logistics"}
DEFAULT_ACTIONS = {"action_required": "issue_refund", "no_action": "document_no_action",
                   "needs_investigation": "escalate_manual_review"}


@cache
def _contracts() -> Contracts:
    return Contracts(Path(__file__).resolve().parents[2] / "contracts" / "schemas")


async def _dispatch(state: CaseState, agent: Agent, recipient: str, task: str,
                    payload: dict[str, Any]) -> ResultMessage:
    """Coordinator → agent → coordinator, with observable task_assigned/handoff events."""
    message = TaskMessage(state.case_id, state.correlation_id(recipient), "coordinator",
                          recipient, task, payload)
    state.emit("task_assigned", "coordinator", target=recipient, decision_code=task.upper(),
               attributes={"correlation_id": message.correlation_id})
    try:
        result = await asyncio.wait_for(agent(message, state.scoped(recipient), state),
                                        TASK_TIMEOUT_S)
    except Exception as exc:  # transport failure or timeout: never invent a finding
        result = ResultMessage(state.case_id, message.correlation_id, recipient, "failed", {},
                               "TOOL_UNAVAILABLE")
        state.emit("handoff", recipient, target="coordinator", decision_code="TOOL_UNAVAILABLE",
                   attributes={"error": type(exc).__name__})
        return result
    if result.case_id != state.case_id or result.correlation_id != message.correlation_id:
        raise ValueError(f"{recipient} returned a result for another case/task")
    state.emit("handoff", recipient, target="coordinator", decision_code=result.decision_code,
               evidence_refs=result.evidence_refs[:20] or None,
               attributes={"status": result.status, "correlation_id": message.correlation_id})
    return result


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    state = CaseState(case, gateway, trace)

    # 1. Entity resolution gates every other specialist.
    entity = (await _dispatch(state, entity_agent, "entity-agent", "resolve_entity", {})).finding
    policy_task = _dispatch(state, fetch_policy, "policy-agent", "load_policy",
                            {"policy_version": case.get("policy_version") or "EC_POLICY_V2"})
    if entity.get("status") != "resolved":
        rules = (await policy_task).finding.get("rules", {})
        return _finalize(state, _insufficient_output(state, entity, rules))

    order_id = entity["order_id"]
    state.findings["anchor"] = entity["anchor"]
    instances = Instances(entity["rows"], entity["anchor"])

    # 2. Order/product first (shipping limits and expected totals feed the next specialists).
    order_result, policy_result = await asyncio.gather(
        _dispatch(state, order_agent, "order-agent", "scope_items",
                  {"order_id": order_id, "instances": instances}),
        policy_task,
    )
    order = order_result.finding
    expected = None
    if order_result.status == "ok":
        expected = round(order["price_total"] + order["freight_total"], 2)

    # 3. Shipment and payment specialists in parallel.
    shipment_result, payment_result = await asyncio.gather(
        _dispatch(state, shipment_agent, "shipment-agent", "analyze_shipment",
                  {"order_id": order_id, "instances": instances,
                   "shipping_limits": order.get("shipping_limits", {})}),
        _dispatch(state, payment_agent, "payment-agent", "analyze_payment",
                  {"order_id": order_id, "instances": instances, "expected_total": expected}),
    )
    shipment, payment = shipment_result.finding, payment_result.finding
    rules = policy_result.finding.get("rules", {})

    issues = classify((entity.get("anchor") or {}).get("order_status"), shipment, payment)
    primary = issues[0] if issues else "unsupported_claim"

    # 4. Dynamic follow-up: seller evidence only when a seller is held responsible.
    if primary in SELLER_ISSUES and order.get("seller_ids"):
        await _dispatch(state, order_agent, "order-agent", "fetch_sellers",
                        {"order_id": order_id, "instances": instances})

    conflicts = _resolve_conflicts(state, entity, shipment)
    draft = _decide(state, entity, order, shipment, payment, rules, issues, conflicts)
    return _finalize(state, draft)


# --------------------------------------------------------------------------- conflicts
def _resolve_conflicts(state: CaseState, entity: dict[str, Any],
                       shipment: dict[str, Any]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    anchor, order_row = entity.get("anchor") or {}, entity.get("order_row") or {}
    if order_row and anchor and (
        order_row.get("order_purchase_timestamp") != anchor.get("order_purchase_timestamp")
    ):
        conflicts.append({
            "field": "order_purchase_timestamp",
            "sources": ["get_order", "get_customer_history"],
            "selected_source": "get_customer_history",
            "resolution_code": "ANCHORED_TO_CASE_OPENED_AT",
        })
    if shipment.get("summary_instance_mismatch"):
        conflicts.append({
            "field": "delivered_customer_at",
            "sources": ["get_shipment_summary", "get_customer_history"],
            "selected_source": "get_customer_history",
            "resolution_code": "ANCHORED_TO_CASE_OPENED_AT",
        })
    if shipment.get("rule_event_disagreement"):
        conflicts.append({
            "field": "delay_responsibility",
            "sources": ["shipment_events", "shipping_limit_dates"],
            "selected_source": "shipment_events",
            "resolution_code": "CONFIRMED_EVENT_PRECEDENCE",
        })
    return conflicts[:5]


# --------------------------------------------------------------------------- policy
def _decide(state: CaseState, entity: dict[str, Any], order: dict[str, Any],
            shipment: dict[str, Any], payment: dict[str, Any], rules: dict[str, Any],
            issues: list[str], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
    order_id = entity["order_id"]
    primary = issues[0] if issues else "unsupported_claim"
    secondary = issues[1:]
    rule = rules.get(primary) or {}
    status = rule.get("case_status") or (
        "no_action" if primary in {"unsupported_claim", "valid_split_payment"}
        else "action_required")
    action = rule.get("recommended_action") or DEFAULT_ACTIONS[status]

    amount = evidence_amount(primary, order, payment)
    policy_amount = rule.get("refund_brl")
    amount_conflict = False
    if status == "no_action" or (policy_amount is not None and float(policy_amount) == 0.0):
        amount = 0.0
    elif amount is None:
        amount = float(policy_amount) if policy_amount is not None else 0.0
    elif policy_amount is not None and abs(float(policy_amount) - amount) > 0.01:
        amount_conflict = True
    amount = round(max(0.0, amount), 2)
    refundable = payment.get("refundable_total_brl")
    if refundable is not None:
        amount = min(amount, refundable)
    if amount_conflict and len(conflicts) < 5:
        conflicts.append({
            "field": "refund_amount_brl",
            "sources": ["get_policy", "get_payment_timeline"],
            "selected_source": "get_payment_timeline",
            "resolution_code": "CASE_EVIDENCE_OVER_POLICY_EXAMPLE",
        })

    sellers = order.get("seller_ids") or []
    parties = []
    for party in rule.get("responsible_parties") or [{"party_type": "unknown", "party_id": None}]:
        party_type = party.get("party_type", "unknown")
        party_id = (sellers[0] if sellers else None) if party_type == "seller" else None
        parties.append({"party_type": party_type, "party_id": party_id})

    late_sellers = shipment.get("late_seller_ids") or []
    if primary == "late_delivery_seller" and not late_sellers:
        late_sellers = sellers[:1]
    if primary != "late_delivery_seller":
        late_sellers = [] if shipment.get("verdict") != "seller_delay" else late_sellers

    state.emit("policy_decided", "policy-agent", decision_code=primary.upper(),
               evidence_refs=_output_refs(state, primary)[:20] or None,
               attributes={"case_status": status, "action": action,
                           "refund_brl": amount, "secondary": ",".join(secondary) or None})

    shipment_verdict = shipment.get("verdict") or "insufficient_evidence"
    payment_verdict = payment.get("verdict") or "insufficient_evidence"
    refs = _output_refs(state, primary)
    claims = _claim_assessments(state, primary, amount, payment, refs)
    output = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": state.case_id,
        "assessment": {
            "primary_issue": primary,
            "secondary_issues": secondary,
            "case_status": status,
            "confidence": 0.9,
        },
        "affected_entities": {
            "order_ids": [order_id],
            "item_ids": order.get("item_ids") or [],
            "seller_ids": sellers,
            "payment_references": [],
            "shipment_ids": [],
        },
        "claim_assessments": claims,
        "entity_resolution": {
            "status": "resolved",
            "resolved_order_ids": [order_id],
            "rejected_candidates": entity.get("rejected") or [],
            "confidence": entity.get("confidence", 0.9),
        },
        "customer_context": {
            "customer_unique_id": entity.get("customer_unique_id"),
            "related_order_ids": entity.get("related") or [order_id],
        },
        "shipment_analysis": {
            "verdict": shipment_verdict,
            "late_seller_ids": late_sellers,
            "timeline_complete": bool(shipment.get("timeline_complete")),
        },
        "payment_analysis": {
            "verdict": payment_verdict,
            "captured_total_brl": payment.get("captured_total_brl"),
            "refunded_total_brl": payment.get("refunded_total_brl"),
            "refundable_total_brl": payment.get("refundable_total_brl"),
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": code.upper(), "rank": rank}
                              for rank, code in enumerate([primary, *secondary][:5], 1)],
            "responsible_parties": parties,
        },
        "evidence_refs": refs,
        "data_conflicts": conflicts,
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": amount,
            "refund_lines": ([{"reason_code": action, "amount_brl": amount, "entity_id": order_id}]
                             if amount > 0 else []),
        },
        "resolution_actions": [action],
    }
    output["assessment"]["confidence"] = _confidence(entity, shipment, payment, primary,
                                                     conflicts, bool(issues))
    return output


def _output_refs(state: CaseState, primary: str) -> list[str]:
    tools = ["get_customer_history", "get_order", "get_order_items", "get_product_context"]
    if primary in SHIPMENT_ISSUES or primary == "unsupported_claim":
        tools.append("get_shipment_summary")
    if primary in PAYMENT_ISSUES or primary in REFUND_ISSUES or primary == "unsupported_claim":
        tools.append("get_payment_timeline")
    if primary in REFUND_ISSUES:
        tools.append("get_refund_timeline")
    if primary in SELLER_ISSUES:
        tools.append("get_sellers")
    tools.append("get_policy")
    return state.ledger.refs(tools)[:30]


def _claim_assessments(state: CaseState, primary: str, amount: float,
                       payment: dict[str, Any], refs: list[str]) -> list[dict[str, Any]]:
    claims = (state.case.get("customer_request") or {}).get("claims") or []
    result = []
    captured = payment.get("captured_total_brl") or 0.0
    for claim in claims[:5]:
        topic, claim_id = claim.get("topic"), claim.get("claim_id")
        if not claim_id:
            continue
        if topic == "requested_full_refund":
            if amount <= 0:
                verdict = "unsupported"
            elif captured and abs(amount - captured) <= 0.01:
                verdict = "supported"
            else:
                verdict = "partially_supported"
        elif topic == primary:
            verdict = "supported"
        else:
            verdict = "unsupported"
        result.append({"claim_id": claim_id[:64], "verdict": verdict, "confidence": 0.85,
                       "evidence_refs": refs[:30]})
    return result


def _confidence(entity: dict[str, Any], shipment: dict[str, Any], payment: dict[str, Any],
                primary: str, conflicts: list[dict[str, Any]], evidence_backed: bool) -> float:
    score = 0.9
    score -= 0.9 - min(0.9, entity.get("confidence", 0.9))
    if any(c.get("selected_source") is None for c in conflicts):
        score -= 0.15
    if shipment.get("rule_event_disagreement"):
        score -= 0.1
    if primary in SHIPMENT_ISSUES and not shipment.get("timeline_complete"):
        score -= 0.15
    if payment.get("verdict") == "insufficient_evidence" and primary not in SHIPMENT_ISSUES:
        score -= 0.2
    if not evidence_backed:
        score -= 0.1  # unsupported_claim is a negative finding
    return round(min(0.95, max(0.3, score)), 2)


def _insufficient_output(state: CaseState, entity: dict[str, Any],
                         rules: dict[str, Any]) -> dict[str, Any]:
    state.emit("policy_decided", "policy-agent", decision_code="INSUFFICIENT_EVIDENCE",
               attributes={"entity_status": entity.get("status")})
    status = "ambiguous" if entity.get("status") == "ambiguous" else "not_found"
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": state.case_id,
        "assessment": {"primary_issue": "insufficient_evidence", "secondary_issues": [],
                       "case_status": "needs_investigation", "confidence": 0.4},
        "affected_entities": {"order_ids": entity.get("resolved") or [], "item_ids": [],
                              "seller_ids": [], "payment_references": [], "shipment_ids": []},
        "entity_resolution": {"status": status, "resolved_order_ids": entity.get("resolved") or [],
                              "rejected_candidates": entity.get("rejected") or [],
                              "confidence": entity.get("confidence", 0.3)},
        "customer_context": {"customer_unique_id": entity.get("customer_unique_id"),
                             "related_order_ids": entity.get("related") or []},
        "shipment_analysis": {"verdict": "insufficient_evidence", "late_seller_ids": [],
                              "timeline_complete": False},
        "payment_analysis": {"verdict": "insufficient_evidence", "captured_total_brl": None,
                             "refunded_total_brl": None, "refundable_total_brl": None},
        "root_cause_analysis": {"ranked_causes": [], "responsible_parties": [
            {"party_type": "unknown", "party_id": None}]},
        "evidence_refs": state.ledger.refs(["get_customer_history", "get_order", "get_policy"]),
        "data_conflicts": [],
        "financial_resolution": {"currency": "BRL", "recommended_refund_brl": 0.0,
                                 "refund_lines": []},
        "resolution_actions": [DEFAULT_ACTIONS["needs_investigation"]],
    }


# --------------------------------------------------------------------------- verifier
def _finalize(state: CaseState, output: dict[str, Any]) -> dict[str, Any]:
    repairs = _verify(state, output)
    try:
        _contracts().validate_output(output, f"outputs/{state.case_id}.json")
        schema_ok = True
    except ContractError:
        schema_ok = False
    code = "PASS" if schema_ok and not repairs else ("REPAIRED" if schema_ok else "SCHEMA_FAIL")
    state.emit("verification_completed", "verifier", decision_code=code,
               evidence_refs=output["evidence_refs"][:20] or None,
               attributes={"repairs": len(repairs), "mcp_calls": state.calls,
                           "confidence": output["assessment"]["confidence"]})
    return output


def _verify(state: CaseState, output: dict[str, Any]) -> list[str]:
    repairs: list[str] = []
    # Evidence ownership: only refs consumed in this case's trace.
    refs = [ref for ref in output["evidence_refs"] if ref in state.consumed_refs]
    if refs != output["evidence_refs"]:
        repairs.append("evidence_scope")
        output["evidence_refs"] = refs
    for claim in output.get("claim_assessments", []):
        claim["evidence_refs"] = [r for r in claim["evidence_refs"] if r in state.consumed_refs]
    # Entity scope.
    resolution = output["entity_resolution"]
    rejected = [c for c in resolution["rejected_candidates"]
                if c not in resolution["resolved_order_ids"]]
    if rejected != resolution["rejected_candidates"]:
        repairs.append("rejected_overlap")
        resolution["rejected_candidates"] = rejected
    # Status / refund / action consistency.
    finance = output["financial_resolution"]
    status = output["assessment"]["case_status"]
    if status == "no_action" and finance["recommended_refund_brl"] > 0:
        repairs.append("no_action_refund")
        finance["recommended_refund_brl"] = 0.0
        finance["refund_lines"] = []
    line_total = round(sum(line["amount_brl"] for line in finance["refund_lines"]), 2)
    if abs(line_total - finance["recommended_refund_brl"]) > 0.01:
        repairs.append("refund_line_total")
        finance["recommended_refund_brl"] = line_total
    actions = list(dict.fromkeys(output["resolution_actions"]))[:8]
    if status == "action_required" and not actions:
        actions = [DEFAULT_ACTIONS[status]]
    output["resolution_actions"] = actions
    # Seller responsibility.
    primary = output["assessment"]["primary_issue"]
    shipment = output["shipment_analysis"]
    if primary == "late_delivery_logistics" and shipment["late_seller_ids"]:
        repairs.append("logistics_with_late_seller")
        shipment["late_seller_ids"] = []
    for party in output["root_cause_analysis"]["responsible_parties"]:
        seller_ids = output["affected_entities"]["seller_ids"]
        if party["party_type"] == "seller" and party["party_id"] not in seller_ids:
            repairs.append("seller_party_scope")
            party["party_id"] = seller_ids[0] if seller_ids else None
    # Timeline sanity: a carrier handoff before purchase means the anchor is unreliable.
    anchor = {}
    if output["entity_resolution"]["status"] == "resolved":
        anchor = next((f for f in [state.findings.get("anchor")] if f), {})
    purchase = parse_ts(anchor.get("order_purchase_timestamp")) if anchor else None
    carrier = parse_ts(anchor.get("order_delivered_carrier_date")) if anchor else None
    if purchase and carrier and carrier < purchase:
        repairs.append("timeline_order")
        output["assessment"]["confidence"] = min(output["assessment"]["confidence"], 0.5)
    if output["data_conflicts"] and output["assessment"]["confidence"] > 0.9:
        output["assessment"]["confidence"] = 0.9
    return repairs
