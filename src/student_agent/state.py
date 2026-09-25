"""Per-case state, A2A message envelopes, evidence ledger and least-privilege gateway."""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Any

from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter

MAX_CALLS_PER_CASE = 12
TASK_TIMEOUT_S = 60.0

AGENT_TOOLS: dict[str, frozenset[str]] = {
    "entity-agent": frozenset({"get_customer_history", "get_order"}),
    "order-agent": frozenset({"get_order_items", "get_sellers", "get_product_context"}),
    "shipment-agent": frozenset({"get_shipment_summary"}),
    "payment-agent": frozenset({"get_payment_timeline", "get_refund_timeline"}),
    "policy-agent": frozenset({"get_policy"}),
}


class ToolFailure(RuntimeError):
    """The MCP tool answered with an error (e.g. no records); never retried."""


class TransportFailure(ConnectionError):
    """The MCP connection failed; the whole case must be re-run, never finalized degraded."""


@dataclass(frozen=True)
class TaskMessage:
    case_id: str
    correlation_id: str
    sender: str
    recipient: str
    task: str
    payload: dict[str, Any]


@dataclass
class ResultMessage:
    case_id: str
    correlation_id: str
    sender: str
    status: str  # ok | insufficient | failed
    finding: dict[str, Any]
    decision_code: str
    evidence_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceRecord:
    ref: str
    tool: str
    domain: str
    actor: str


class EvidenceLedger:
    """Evidence refs exactly as returned by MCP, scoped to one case."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self._by_tool: dict[str, EvidenceRecord] = {}

    def add(self, tool: str, actor: str, evidence: dict[str, Any]) -> str:
        record = EvidenceRecord(evidence["evidence_ref"], tool, evidence["domain"], actor)
        self._by_tool[tool] = record
        return record.ref

    def ref(self, tool: str) -> str | None:
        record = self._by_tool.get(tool)
        return record.ref if record else None

    def refs(self, tools: list[str]) -> list[str]:
        result: list[str] = []
        for tool in tools:
            ref = self.ref(tool)
            if ref and ref not in result:
                result.append(ref)
        return result

    def all_refs(self) -> set[str]:
        return {record.ref for record in self._by_tool.values()}


class CaseState:
    """Blackboard owned by the coordinator for a single case."""

    def __init__(self, case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter) -> None:
        self.case = case
        self.case_id: str = case["case_id"]
        self.trace = trace
        self.ledger = EvidenceLedger(self.case_id)
        self.findings: dict[str, dict[str, Any]] = {}
        self.consumed_refs: set[str] = set()
        self.calls = 0
        self._gateway = gateway
        self._cache: dict[tuple[str, tuple[tuple[str, str], ...]], Any] = {}
        self._lock = asyncio.Lock()
        self._ids = itertools.count(1)

    def correlation_id(self, recipient: str) -> str:
        return f"{self.case_id}:{recipient}:{next(self._ids)}"

    def emit(self, event_type: str, actor: str, **kwargs: Any) -> None:
        self.trace.emit(case_id=self.case_id, event_type=event_type, actor=actor, **kwargs)

    def scoped(self, actor: str) -> ScopedGateway:
        return ScopedGateway(self, actor, AGENT_TOOLS.get(actor, frozenset()))

    async def _call(self, actor: str, tool: str, arguments: dict[str, str]) -> dict[str, Any]:
        key = (tool, tuple(sorted(arguments.items())))
        async with self._lock:
            if key in self._cache:
                cached = self._cache[key]
                if isinstance(cached, ToolFailure):
                    raise cached
                return cached
            if self.calls >= MAX_CALLS_PER_CASE:
                raise ToolFailure(f"call budget exhausted before {tool}")
            self.calls += 1
        attempts = 2  # one retry for transport errors only
        for attempt in range(attempts):
            try:
                evidence = await self._gateway.call(tool, case_id=self.case_id, **arguments)
                break
            except RuntimeError as exc:  # MCP tool-level error: deterministic, do not retry
                failure = ToolFailure(str(exc))
                self._cache[key] = failure
                raise failure from exc
            except Exception as exc:
                if attempt + 1 == attempts:
                    raise TransportFailure(f"{tool}: {type(exc).__name__}") from exc
                async with self._lock:
                    self.calls += 1
        self._cache[key] = evidence
        ref = self.ledger.add(tool, actor, evidence)
        self.consumed_refs.add(ref)
        self.emit("tool_result_consumed", actor, tool_name=tool, evidence_refs=[ref])
        return evidence


class ScopedGateway:
    """Least-privilege view of the MCP gateway for one agent."""

    def __init__(self, state: CaseState, actor: str, allowed: frozenset[str]) -> None:
        self._state = state
        self.actor = actor
        self._allowed = allowed

    async def call(self, tool: str, **arguments: str) -> dict[str, Any]:
        if tool not in self._allowed:
            raise PermissionError(f"{self.actor} may not call {tool}")
        return await self._state._call(self.actor, tool, arguments)

    async def try_call(self, tool: str, **arguments: str) -> dict[str, Any] | None:
        try:
            return await self.call(tool, **arguments)
        except ToolFailure:
            return None
