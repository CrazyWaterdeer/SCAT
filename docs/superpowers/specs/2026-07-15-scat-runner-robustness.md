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

## Design (Codex-revised — reactive-only, strict pairing)

### Fix first — a pre-existing history-corruption bug (Codex #1)
`turn()`'s fatal-error handler pops the **last** user message (`runner.py:121-122`). On the first loop
iteration that's the just-appended user text (fine), but after a tool round the last user message is a
`tool_result` message — popping it **orphans the preceding assistant `tool_use`**, so the *next* turn
400s. This exists today, independent of T3.2. **Fix:** record `turn_start_len = len(self.messages)`
before appending the user text and, on any fatal error, `del self.messages[turn_start_len:]` — restoring
the exact pre-turn state (drops the whole failed turn, always valid). The existing
`test_stream_error_cleans_history` (expects `messages == []` after a 1st-iteration error) still holds.

### T3.3 — lean on the SDK, just widen + expose (small)
The SDK already does exponential backoff on 408/409/429/≥500 (529 overloaded ⊂ ≥500) at request
establishment. **No hand-rolled retry.** `AnthropicProvider.__init__(max_retries, max_tokens)` passes
`Anthropic(api_key=api_key or None, max_retries=max_retries)` and uses `max_tokens`. `build_runner`
threads both from config: `agent.max_retries` default **3** (a touch more resilient than the SDK's 2
without multiplying agent-loop latency the way 5 would — Codex #9), `agent.max_tokens` default **4096**.

### T3.2 — reactive compact-and-retry only
Proactive/threshold compaction is **dropped from v1** (Codex #8): with 1M-context models, char/4
thresholds like 150k would shed useful context at ~15% of the window for no benefit. The reactive path is
the real guarantee and can *never* over-compact — it only fires on an actual context-limit error.
**Invariant that must never break:** every assistant `tool_use` block is answered by a `tool_result`
(same id) in the **immediately following** user message.

**`_is_context_limit_error(exc) -> bool`** (provider-agnostic, Codex #6): true iff
`re.search(r"context|too many tokens|prompt is too long|maximum.*(context|token)", str(exc), re.I)` —
**not** the `BadRequest` class name alone (that false-positives on pairing/schema/model/`max_tokens`
errors, where compaction can't help). `status_code == 400` is required too **when the attribute exists**
(`getattr(exc, "status_code", 400) == 400`), else message-only.

**`_compact_history(messages) -> list[dict]`** (pure; the aggressive one-shot used on overflow):
- **Protected set** = index `0` (first user/task) ∪ the last `_KEEP_RECENT` (6) indices. Computed up
  front and **frozen** before any deletion (Codex #5).
- **Soft, structure-preserving (Codex #2):** for every *unprotected* message, rewrite **only**:
  `tool_result.content` (a string) → `"[earlier result elided to save context]"` when > `_STUB_OVER`
  (800) chars; `text` block text → truncated. **`tool_use.input` is left as-is (an object)** and all of
  `id/tool_use_id/name/type/is_error` are preserved — nothing that pairing depends on is touched.
- **Drop matched pairs (Codex #3/#4):** additionally drop every **droppable adjacent pair** in the
  unprotected middle, where a pair is `messages[i]` (assistant) + `messages[i+1]` (user) with:
  `messages[i]` blocks ⊆ {text, tool_use}; `messages[i+1]` blocks all `tool_result`;
  **`set(tool_use ids) == set(tool_result ids)`, non-empty, no dups, no extras on either side**; and
  both `i` and `i+1` unprotected. Drop both members. **No synthetic marker** — removing a valid pair from
  `…user, assistant, user, assistant…` leaves valid alternation; a `U(marker)` would create `U,U`
  (Codex #4). Build the result by copying protected/soft-rewritten messages and skipping dropped pairs;
  since we drop *both* sides, no `tool_use` is ever orphaned.

**Reactive trigger (Codex #7):** consume `provider.stream(...)` tracking `yielded_any` (did we emit any
TextDelta/ToolUse* to the caller this attempt?). If it raises **before yielding anything**,
`_is_context_limit_error(exc)` is true, and we haven't retried this iteration:
`self.messages = _compact_history(self.messages)` and retry the stream **once**. If it already yielded
(mid-stream failure) or isn't a context-limit error or already retried → fall through to the fixed fatal
path (`del self.messages[turn_start_len:]`, yield the error, `TurnDone("error")`). Retrying only when
`not yielded_any` prevents duplicate UI output.

Config additions (`agent`): `max_retries: 3`, `max_tokens: 4096`. Runner caps: `_KEEP_RECENT = 6`,
`_STUB_OVER = 800`.

## Verification
`tests/test_agent_runner.py` (extend), all with **stub providers** (no network):
- **Pre-existing-bug fix:** a provider that raises on the **2nd** iteration (after one real tool round)
  leaves history with **no orphaned `tool_use`** (every `tool_use` id has a matching `tool_result`);
  the 1st-iteration `test_stream_error_cleans_history` still yields `messages == []`.
- **`_compact_history` soft:** a long middle `tool_result.content` is stubbed; `messages[0]` + last 6
  kept verbatim; `tool_use.input` stays an object; all ids preserved.
- **`_compact_history` pair-drop — the nasty cases (Codex #3/#10):** `A(tu:a,tu:b),U(tr:a)` (partial) →
  **not** dropped; `A(tu:a),U(tr:a,tr:x)` (extra) → not dropped; `A(text-only),U(tr:a)` → not a pair;
  `A(text,tu:a),U(tr:a)` → dropped; a pair straddling the keep boundary (`A(tu:a)` middle, `U(tr:a)` in
  last 6) → **neither** side dropped; after any drop, **pairing across the whole list is still valid**;
  `len(messages) <= _KEEP_RECENT+1` → nothing dropped (all protected).
- **Reactive retry:** provider raises a fake context-limit error on the **1st** stream call (before any
  yield) then succeeds → `turn()` compacts + completes with exactly one retry; a provider that raises a
  **non**-context error → no retry, clean error path; a provider that yields a TextDelta **then** raises a
  context-limit error → **no** retry (Codex #7), clean error path, no duplicate text.
- **`_is_context_limit_error`:** true for the known messages, false for `ValueError("bad tool schema")`
  and a generic 400-shaped "invalid tool_use" message.
- **Provider/config:** `AnthropicProvider(max_retries=7, max_tokens=99)` forwards both to the client
  (monkeypatch `anthropic.Anthropic` to capture kwargs — no network); `build_runner` reads the config
  defaults.
- Packaging guard unaffected (`anthropic` stays lazy; runner adds only `re`). Full suite green.

## Risks
- **R1 pairing bug → a 400.** The one thing to get right: soft never touches structure; pair-drop uses
  exact id-set equality on *matched adjacent* pairs and drops *both* sides; protected set frozen before
  deletion; explicit whole-list pairing asserts in tests.
- **R2 infinite retry** — once per iteration (`compacted_retried` flag); a 2nd overflow falls through.
- **R3 duplicate UI output** — retry gated on `not yielded_any` (context-limit errors occur pre-yield).
- **R4 over-shed context** — only fires on a real overflow; keeping `messages[0]` + last 6 is the right
  recovery (continue with the task + live work rather than fail).

## Out of scope / deferred
Proactive/model-aware compaction (deferred — reactive suffices for 1M models; revisit if a smaller-window
model is added); subscription-backend compaction (CLI owns it); mid-stream disconnect auto-resume;
summarizing dropped rounds via a model call (we elide, not summarize); exposing request timeout.
