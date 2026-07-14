# T3.2 + T3.3 — runner message compaction + retry hardening (API backend)

**Roadmap:** T3.2 (M) + T3.3 (S), Tier 3. **Branch:** `feat/tier3-robustness`. **Spec refs:** §7 / §10.
**Scope: the API backend only** (`AgentRunner` + `AnthropicProvider`). The subscription backend drives
the `claude` CLI, which owns its own agent loop, context compaction, and retry/backoff — it never uses
`AgentRunner.messages`, so nothing here touches it.

## Problem

1. **T3.2 — the runner has no history compaction and dies at the context limit.** Only tool *results*
   are compacted, and only as they're appended (`_compact_tool_result`, 6000-char cap). The full
   `self.messages` history grows unbounded across a long chat; once the request exceeds the model's
   context window the provider raises `BadRequestError("prompt is too long…")`, which `turn()`'s
   `except Exception` turns into a one-line error and **stops the turn** (`runner.py:119-124`). No
   proactive shrink, no recovery.
2. **T3.3 — retry/backoff is unverified.** The roadmap says "verify vs SDK default retries first."
   **Verified (anthropic 0.116.0):** `Anthropic(...)` already retries 408/409/429/≥500 (overloaded is
   HTTP 529 → ≥500) with exponential backoff + jitter, honors the `x-should-retry` header, default
   `max_retries=2` — for `.messages.stream()` request establishment too. So a **hand-rolled backoff
   loop would be wrong/duplicative.** The only real gap: 2 retries is thin for a long agent run during a
   sustained overload, and neither `max_retries` nor `max_tokens` is configurable.

## Design

### T3.3 — lean on the SDK, just widen + expose (small)
`AnthropicProvider.__init__` passes `max_retries` to the client and reads `max_tokens` from config:
```python
Anthropic(api_key=api_key or None, max_retries=max_retries)   # SDK does exp backoff on 408/409/429/5xx
```
`build_runner` threads `max_retries` + `max_tokens` from config (`agent.max_retries` default **5**,
`agent.max_tokens` default **4096**). No custom retry code — a one-line comment records that the SDK
owns backoff. (The streaming *body* dropping mid-stream is not auto-retried by the SDK, but 429/overload
occur at request establishment, which IS retried; a mid-stream drop surfaces as today — out of scope.)

### T3.2 — compaction in `AgentRunner`
Two triggers, one pairing-safe compactor. **The Anthropic invariant that must never break:** every
assistant `tool_use` block is answered by a `tool_result` (same id) in the **immediately following**
user message. Compaction must preserve this.

**Sizing.** `_estimate_tokens(messages) = sum(len(json.dumps(m, default=str)) for m in messages) // 4`
(cheap char/4 heuristic; we only need order-of-magnitude).

**`_compact_messages(messages, *, aggressive) -> list[dict]` (pure, returns a new list):**
- **Always keep verbatim:** `messages[0]` (the first user turn) and the last `_KEEP_RECENT` messages
  (default 6). Compaction only rewrites the *middle*.
- **Soft (structure-preserving, lossless of shape):** in the middle messages, for each block:
  - `tool_result`: if its `content` string > `_STUB_OVER` (800) chars, replace with
    `"[earlier result elided to save context]"` — **id/structure untouched** so pairing holds.
  - `tool_use`: leave the block (id/name needed for pairing) but truncate oversized `input` values.
  - `text`: truncate to `_MAX_STRING_CHARS`.
  This alone can reclaim most tokens (tool results dominate) and can **never** invalidate pairing.
- **Aggressive (adds pair-dropping):** after soft, if still over `_HARD_TARGET`, drop **whole matched
  round-pairs** from the oldest middle region: a *pair* = an assistant message whose blocks are only
  `text`/`tool_use` **immediately followed** by a user message whose blocks are all `tool_result`.
  Dropping both members of such an adjacent pair keeps the sequence valid (…userText → assistant →
  userResult… becomes …userText → assistant → userResult… with the middle pair gone). Never drop a
  half-pair; never drop `messages[0]` or the last `_KEEP_RECENT`. Insert a single synthetic
  `{"role":"user","content":[{"type":"text","text":"[older tool rounds omitted to fit context]"}]}`
  marker where the run was removed **only if** the surrounding messages still form a valid
  user/assistant alternation (else omit the marker — correctness over cosmetics). Stop when under
  `_HARD_TARGET` or no more full pairs remain.

**Trigger 1 — proactive (proportionate):** at the top of each `for _ in range(max_loops)` iteration,
if `_estimate_tokens(self.messages) > _COMPACT_AT` (config `agent.compact_at_tokens`, default 150_000),
`self.messages = _compact_messages(self.messages, aggressive=False)`.

**Trigger 2 — reactive (the real recovery):** wrap the `provider.stream(...)` consumption so that if it
raises and `_is_context_limit_error(exc)` and we have **not** already force-compacted this iteration:
`self.messages = _compact_messages(self.messages, aggressive=True)` and **retry the stream once**. If it
still fails (or the error isn't a context-limit one), fall through to today's clean error path (drop the
just-appended user turn, yield the error, `TurnDone("error")`).
`_is_context_limit_error(exc)` is **provider-agnostic** — matches the message string case-insensitively
against `("prompt is too long", "context window", "maximum context", "too many tokens")` (and the class
name containing `BadRequest`), so the runner keeps zero hard dependency on `anthropic`.

Config additions (`agent` section): `max_retries: 5`, `max_tokens: 4096`, `compact_at_tokens: 150000`.
Module caps in runner: `_KEEP_RECENT = 6`, `_STUB_OVER = 800`, `_HARD_TARGET = 120_000`.

## Verification
- `tests/test_agent_runner.py` (extend), all with a **stub provider** (no network):
  - `_compact_messages` soft: a long middle `tool_result` is stubbed; `messages[0]` + last 6 kept
    verbatim; **every `tool_use` still has its matching `tool_result` id** afterward (assert pairing).
  - `_compact_messages` aggressive: oldest full round-pairs dropped, pairing still valid, size drops
    below target, last 6 preserved; a half-pair (assistant tool_use with no following result) is **not**
    orphaned.
  - reactive retry: a stub provider that raises a fake context-limit error on the **first** stream call
    then succeeds → `turn()` compacts and completes (one retry), and the transcript is sane; a provider
    that raises a **non**-context error → today's single-error path (no retry).
  - proactive: after synthetically inflating history over `_COMPACT_AT`, the next `turn()` shrinks it.
  - `_is_context_limit_error`: true for the known messages / BadRequest class-name, false for a generic
    `ValueError`.
- `tests/test_providers.py` or extend: `AnthropicProvider(max_retries=…, max_tokens=…)` forwards both to
  the client (monkeypatch `anthropic.Anthropic` to capture kwargs — no network); `build_runner` reads the
  config defaults.
- Packaging guard unaffected (runner change is pure-python, no new imports; `anthropic` stays lazy in the
  provider). Full suite green.

## Risks
- **R1 pairing bug** = a 400 from the API (orphan tool_use/result). Mitigated by soft compaction never
  touching structure, aggressive only dropping *matched adjacent* pairs, and explicit pairing asserts in
  tests. This is the one thing Codex must scrutinize.
- **R2 infinite retry** — reactive retry is **once per iteration** (a `compacted_retried` flag); a second
  context-limit error falls through to the error path.
- **R3 over-compaction losing needed context** — `messages[0]` (the task) + last 6 messages (the live
  work) are always kept; only older tool noise is shed. Acceptable and the point.
- **R4 char/4 token estimate is rough** — only gates *when* to compact; the reactive path is the real
  guarantee, so a loose estimate just means we compact a bit early/late, never incorrectly.

## Out of scope
Subscription-backend compaction (CLI owns it); mid-stream disconnect auto-resume; summarizing dropped
rounds via a model call (we elide, not summarize — no extra token cost/latency).
