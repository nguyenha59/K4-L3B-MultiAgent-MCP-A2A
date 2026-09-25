# L3B Architecture Record

Team phải cập nhật tài liệu này cùng source. Mục tiêu là mô tả quyết định có thể kiểm chứng, không ghi prompt bí mật hoặc chain-of-thought.

## 0. Quyết định nền tảng

- **Deterministic, rule-based, không dùng LLM.** Input có cấu trúc cố định (claim là enum,
  2 candidate, customer hint); evidence MCP là dữ liệu có cấu trúc và `get_policy` trả về
  luật máy đọc được. Mọi kết luận là phép so khớp ngày/tiền/ID nên code cho kết quả chính
  xác, tái lập được, không bịa `evidence_ref`, chạy được trên CPU. Ràng buộc model < 10B
  tham số được thỏa vì không có model nào trong vòng lặp.
- **Hub-and-spoke:** một Coordinator sở hữu state, lập kế hoạch động và điều phối các
  specialist; specialist không gọi nhau trực tiếp → không có vòng lặp, budget tập trung.
- **Thuần Python asyncio**, không framework agent.

## 1. System overview

```text
                         ┌───────────────────────────────┐
 input case ───────────► │ Coordinator (state, plan,     │ ── trace: case_received / case_finalized
                         │ budget, cache, handoff)       │
                         └──┬─────────────────────────┬──┘
          task_assigned     │                         │
                            ▼                         │
                  ┌───────────────────┐               │
                  │ Entity agent      │ customer_history, get_order
                  │ (candidate +      │ → resolved order, instance window, rejected
                  │  instance resolve)│
                  └─────────┬─────────┘
               handoff      │  (dừng sớm nếu not_found / ambiguous)
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
 ┌──────────────┐   ┌───────────────┐   ┌─────────────────┐
 │ Order/Product│   │ Shipment      │   │ Payment/Refund  │   (chạy song song)
 │ items,sellers│   │ shipment_summ.│   │ payment_timeline│
 │ product_ctx  │   │               │   │ refund_timeline │
 └──────┬───────┘   └──────┬────────┘   └────────┬────────┘
        └──────────────────┼─────────────────────┘ handoff
                           ▼
                 ┌───────────────────┐
                 │ Conflict resolver │  (không gọi tool) → data_conflicts
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │ Policy agent      │ get_policy → primary_issue, status, refund,
                 │                   │ responsible parties, actions   ── policy_decided
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │ Verifier          │ (không gọi tool) schema + invariants + confidence
                 └─────────┬─────────┘ ── verification_completed
                           ▼
                   outputs/<case_id>.json
```

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Tool permission | Output/handoff |
| --- | --- | --- | --- | --- |
| Coordinator (`coordinator`) | case input | Lập kế hoạch, giao task, giữ `CaseState`, budget/cache, dừng sớm | không gọi tool | `task_assigned` tới từng agent, nhận `handoff` |
| Entity/customer (`entity-agent`) | candidate_order_ids, claimed_order_id, customer hint, opened_at | Loại candidate không tồn tại/không thuộc khách; chọn **instance** liên quan của order (xem §3); customer context | `get_customer_history`, `get_order` | `EntityFinding` (status, order, instance anchor, rejected, related orders, confidence) |
| Order/product (`order-agent`) | order_id, instance anchor | Item, seller, giá, freight của instance; category | `get_order_items`, `get_sellers`, `get_product_context` | `OrderFinding` (item_ids, seller_ids, price, freight, shipping limits) |
| Shipment (`shipment-agent`) | order_id, instance anchor, shipping limits | Timeline carrier/customer/estimate; phân định seller vs logistics; completeness | `get_shipment_summary` | `ShipmentFinding` (verdict, late_seller_ids, timeline_complete) |
| Payment/refund (`payment-agent`) | order_id, instance anchor, expected total | Capture/refund/mismatch/duplicate của instance; tổng BRL | `get_payment_timeline`, `get_refund_timeline` | `PaymentFinding` (verdict, captured, refunded, refundable, payment refs) |
| Conflict resolver (`conflict-resolver`) | tất cả finding + raw evidence | Phát hiện nguồn mâu thuẫn (vd. `get_order` trả instance khác history), chọn nguồn theo precedence | không gọi tool | `data_conflicts[]` |
| Policy (`policy-agent`) | findings, conflicts, policy_version | Suy ra `primary_issue` từ evidence (không từ claim); áp luật `EC_POLICY_V2` | `get_policy` | assessment, root cause, financial resolution, actions — `policy_decided` |
| Verifier (`verifier`) | draft output + ledger + trace refs | Schema, invariants §6, sửa lỗi cơ học, hạ confidence | không gọi tool | output cuối — `verification_completed` |

Least privilege được **ép bằng code**: mỗi agent nhận một `ScopedGateway` chỉ cho phép các tool
trong cột "Tool permission"; gọi tool ngoài quyền → `PermissionError`.

## 3. Entity resolution và A2A protocol

**Candidate:** với mỗi `candidate_order_ids`, ưu tiên `claimed_order_id`. Candidate được chấp
nhận khi xuất hiện trong `get_customer_history` của `customer_unique_id_hint`. Candidate không có
trong history thì bị đưa vào `rejected_candidates` **mà không gọi thêm tool** (tiết kiệm call);
chỉ khi không candidate nào có trong history mới thử `get_order` từng candidate.

**Instance:** evidence của một `order_id` có thể chứa nhiều lần mua (history row) trộn lẫn;
history row trùng hệt nhau được gộp thành một instance. Specialist gán từng record vào instance
bằng **offset so với purchase** (không theo khoảng thời gian, vì flow của các instance đan xen):

- payment event: purchase + [0, 12h]; refund event: purchase + [240h, 288h];
- shipping limit / item: purchase + [48h, 96h]; shipment event: trùng ngày giao của instance;
- không khớp duy nhất → instance mua gần nhất trước record (fallback thời gian).
- event trùng hệt nhau (mọi field) bị loại trùng.

Specialist trả finding **theo từng instance**. Coordinator xét các instance mua ≤ `opened_at`,
tính issue của mỗi instance từ evidence, rồi chọn instance mới nhất **có evidence xác nhận một
claim**; nếu không instance nào xác nhận → instance mới nhất và issue theo evidence (claim bị bác
bỏ, confidence giảm). Khi chấm payment theo issue đã chọn, capture thuộc flow refund được tách
khỏi flow split/capture để không cộng lẫn hai kịch bản.

**Status:** `resolved` (1 order, anchor rõ), `ambiguous` (≥ 2 order hợp lệ hoặc anchor không
xác định), `not_found` (không candidate nào tồn tại). `ambiguous`/`not_found` → Coordinator
dừng nhánh specialist, output `insufficient_evidence` + `needs_investigation`.

**Message envelope (in-process A2A):**

```text
TaskMessage   {case_id, correlation_id, sender, recipient, task, payload, deadline_s}
ResultMessage {case_id, correlation_id, sender, status: ok|insufficient|failed,
               finding, evidence_refs}
```

`case_id` bắt buộc khớp ở mọi hop; Coordinator từ chối result có `case_id` khác. Mỗi task có
timeout (mặc định 60 s). Chỉ Coordinator được gửi task → đồ thị handoff là DAG cố định theo
kế hoạch, không thể lặp. Trace chỉ ghi `task_assigned`/`handoff` với `target`, `decision_code`
và số liệu tóm tắt, không ghi nội dung suy luận.

## 4. Evidence và conflict lifecycle

1. `EvidenceGateway.call` validate envelope với `mcp-evidence-response-v1`.
2. `EvidenceLedger` lưu `evidence_ref` nguyên văn cùng `tool`, `domain`, `actor`, `case_id`.
   Ref không bao giờ được tạo/sửa; ledger tạo mới cho mỗi case và bị hủy khi case kết thúc.
3. Agent dùng kết quả → emit `tool_result_consumed` (actor, tool_name, evidence_refs).
4. Output `evidence_refs` = các ref thật sự hỗ trợ kết luận (entity, domain của primary issue,
   policy); verifier bảo đảm mọi ref trong output đã xuất hiện trong trace của cùng case.
5. `claim_assessments` map từng claim của khách → verdict + ref của domain liên quan.

**Conflict:** so sánh field giữa các nguồn (vd. `order_status`/timestamps của `get_order`
so với anchor trong `get_customer_history`, tổng payment so với giá + freight). Precedence:
nguồn lifecycle chuyên biệt (payment/refund timeline, shipment events) > customer history đã
neo theo `opened_at` > order row tổng hợp. Không phân xử được → `selected_source: null`,
`resolution_code: UNRESOLVED`, verdict liên quan `conflicting`, hạ confidence.

## 5. Failure and efficiency policy

| Failure | Retry budget | Fallback | Trace event/code |
| --- | ---: | --- | --- |
| MCP timeout / lỗi mạng | 1 retry/call; 5 lần reconnect liên tiếp (5–60 s backoff) | hủy case (không ghi output thiếu dữ liệu), cắt trace của case dở, reconnect, chạy lại đúng case đó | stderr `WARN: MCP connection lost`; trace không chứa event của lần hỏng |
| Tool trả lỗi nghiệp vụ (vd. không có refund) | 0 | coi là "không có dữ liệu", không đoán | `tool_result_consumed` không có ref / `handoff` `NO_RECORDS` |
| Entity not found/ambiguous | 0 | dừng specialist, `needs_investigation` | `handoff` `ENTITY_NOT_FOUND` / `ENTITY_AMBIGUOUS` |
| Source conflict | 0 | precedence §4, không được thì `UNRESOLVED` | `policy_decided` + `data_conflicts` |
| Invalid specialist result | 0 | verifier sửa lỗi cơ học; lỗi logic → policy tính lại 1 lần | `verification_completed` `REPAIRED` / `DOWNGRADED` |

**Budget:** kế hoạch chuẩn 9 call/case (history, order, items, sellers, product, shipment,
payment timeline, refund timeline, policy); trần cứng 12. `get_order_payments` không gọi vì
`get_payment_timeline` đã chứa base payments. Cache key `(case_id, tool, args)`, phạm vi một
case. Không quét candidate đã bị loại. Chạy tuần tự từng case (concurrency 1 case) để trace
và audit dễ đối chiếu; trong case, 3 specialist chạy song song.

## 6. Verification invariants

- JSON Schema `day09-l3b-output-v2` pass; không field thừa; enum hợp lệ; `case_id` khớp.
- `entity_resolution.resolved_order_ids` ⊆ candidate; `rejected_candidates` ∩ resolved = ∅;
  `affected_entities.order_ids` = resolved.
- Mọi `evidence_ref` output thuộc ledger của case và đã có trong `tool_result_consumed`.
- Timeline: carrier ≥ purchase; verdict shipment khớp so sánh ngày.
- Payment: `refundable = max(0, captured − refunded)`; `recommended_refund_brl` =
  Σ `refund_lines.amount_brl` ≤ refundable (khi captured biết).
- Status/refund/action: `no_action` ⇒ refund 0 và action `document_no_action`;
  `action_required` ⇒ ≥ 1 action; không action trùng.
- Responsibility: `late_delivery_seller` ⇒ party seller với `party_id` ∈ `seller_ids` và
  `late_seller_ids` ≠ ∅; `late_delivery_logistics` ⇒ không có seller trong `late_seller_ids`.
- Confidence ∈ [0.3, 0.95]; trừ cho mỗi conflict unresolved, timeline/payment thiếu, anchor
  không chắc chắn.

## 7. Reproducibility

- Không LLM, không random seed; kết quả chỉ phụ thuộc input + evidence MCP.
- Python 3.11, dependency pin trong `pyproject.toml`; lệnh: `day09 run`, `day09 validate`,
  `day09 package --output dist/submission.zip`.
- Concurrency: 1 case một lúc, tối đa 3 specialist song song trong case; timeout call 300 s
  (gateway), task 60 s.
- Không ghi API key trong code, trace hay output.
