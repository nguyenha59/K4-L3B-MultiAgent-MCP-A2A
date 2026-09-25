"""The runner reconnects after a transport failure and re-runs only the interrupted case."""

from __future__ import annotations

import asyncio
import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from student_agent import OUTPUT_SCHEMA_VERSION, cli

ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ["SYN_CASE_001", "SYN_CASE_002", "SYN_CASE_003"]


def _output(case_id: str) -> dict[str, Any]:
    empty: list[str] = []
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION, "case_id": case_id,
        "assessment": {"primary_issue": "unsupported_claim", "secondary_issues": [],
                       "case_status": "no_action", "confidence": 0.5},
        "affected_entities": {"order_ids": empty, "item_ids": empty, "seller_ids": empty,
                              "payment_references": empty, "shipment_ids": empty},
        "entity_resolution": {"status": "not_found", "resolved_order_ids": empty,
                              "rejected_candidates": empty, "confidence": 0.3},
        "customer_context": {"customer_unique_id": None, "related_order_ids": empty},
        "shipment_analysis": {"verdict": "insufficient_evidence", "late_seller_ids": empty,
                              "timeline_complete": False},
        "payment_analysis": {"verdict": "insufficient_evidence", "captured_total_brl": None,
                             "refunded_total_brl": None, "refundable_total_brl": None},
        "root_cause_analysis": {"ranked_causes": [], "responsible_parties": []},
        "evidence_refs": [], "data_conflicts": [],
        "financial_resolution": {"currency": "BRL", "recommended_refund_brl": 0.0,
                                 "refund_lines": []},
        "resolution_actions": ["document_no_action"],
    }


class FakeGateway:
    async def list_tools(self) -> list[str]:
        return ["get_order"]


def test_run_reconnects_and_reruns_interrupted_case(tmp_path: Path, monkeypatch) -> None:
    shutil.copytree(ROOT / "contracts", tmp_path / "contracts")
    (tmp_path / "case-set.json").write_text(json.dumps(
        {"case_set_version": "test-v1", "variant_id": "l3b", "case_ids": CASE_IDS}))
    (tmp_path / "inputs").mkdir()
    for case_id in CASE_IDS:
        (tmp_path / "inputs" / f"{case_id}.json").write_text(json.dumps({"case_id": case_id}))
    (tmp_path / ".env").write_text(
        "COMPETITION_API_URL=http://localhost\n"
        "COMPETITION_TEAM_API_KEY=sk-team-test_key_0123456789\n"
        "MCP_ENDPOINT=http://localhost/mcp\n")

    connections: list[int] = []
    attempts: dict[str, int] = {}

    @asynccontextmanager
    async def fake_connect(*_args: Any):
        connections.append(1)
        yield FakeGateway()

    async def fake_solve(case: dict[str, Any], gateway: Any, trace: Any) -> dict[str, Any]:
        case_id = case["case_id"]
        attempts[case_id] = attempts.get(case_id, 0) + 1
        trace.emit(case_id=case_id, event_type="task_assigned", actor="coordinator")
        if case_id == "SYN_CASE_002" and attempts[case_id] == 1:
            raise ConnectionError("network dropped")
        return _output(case_id)

    monkeypatch.setattr(cli, "connect_gateway", fake_connect)
    monkeypatch.setattr(cli, "solve_case", fake_solve)
    monkeypatch.setattr(cli, "RECONNECT_DELAYS_S", (0, 0, 0, 0, 0))
    monkeypatch.setattr(cli, "load_case_set", _load)

    asyncio.run(cli._run(tmp_path))

    assert len(connections) == 2
    assert attempts == {"SYN_CASE_001": 1, "SYN_CASE_002": 2, "SYN_CASE_003": 1}
    assert sorted(p.stem for p in (tmp_path / "outputs").glob("*.json")) == CASE_IDS
    events = [json.loads(line) for line in
              (tmp_path / "traces" / "trace.jsonl").read_text().splitlines()]
    for case_id in CASE_IDS:
        types = [e["event_type"] for e in events if e["case_id"] == case_id]
        assert types == ["case_received", "task_assigned", "case_finalized"]


def test_logic_errors_are_not_retried() -> None:
    assert not cli._is_transient(ValueError("schema"))
    assert cli._is_transient(ConnectionError("dropped"))
    assert cli._is_transient(ExceptionGroup("tg", [ConnectionError("x"), TimeoutError()]))
    with pytest.raises(AssertionError):
        assert cli._is_transient(ExceptionGroup("tg", [ValueError("bad")]))


def _load(root: Path):
    from student_agent.cases import load_case_set

    return load_case_set(root, expected_count=3)
