# T3.2 + T3.3 implementation plan (API-backend runner robustness)

**Spec:** `docs/superpowers/specs/2026-07-15-scat-runner-robustness.md` (Codex pass 1 folded in) ·
**Branch:** `feat/tier3-robustness`. Reactive-only compaction + a pre-existing-bug fix + SDK-backoff
config. Full suite green after.

## 1. `scat/agent/runner.py`

### 1a. Helpers (module level, near `_compact_tool_result`)
```python
import re
_KEEP_RECENT = 6
_STUB_OVER = 800
_CTX_RE = re.compile(r"context|too many tokens|prompt is too long|maximum.*(?:context|token)", re.I)

def _is_context_limit_error(exc) -> bool:
    if getattr(exc, "status_code", 400) != 400:      # if the exc exposes a status, require 400
        return False
    return bool(_CTX_RE.search(str(exc)))

def _tool_use_ids(msg) -> set[str]:            # ids from an assistant message's tool_use blocks
def _tool_result_ids(msg) -> set[str]:         # tool_use_ids from a user message's tool_result blocks
def _is_assistant_pairable(msg) -> bool:       # role assistant, blocks ⊆ {text, tool_use}, has ≥1 tool_use
def _is_all_tool_results(msg) -> bool:         # role user, blocks all tool_result, ≥1

def _soft_rewrite(msg) -> dict:
    # deep-ish copy; ONLY: tool_result.content(str) -> stub if len>_STUB_OVER; text.text -> _truncate_text.
    # tool_use.input untouched (object); id/tool_use_id/name/type/is_error preserved.

def _compact_history(messages: list[dict]) -> list[dict]:
    n = len(messages)
    protected = {0, *range(max(0, n - _KEEP_RECENT), n)}     # frozen BEFORE any deletion
    # 1) mark droppable matched adjacent pairs in the unprotected middle
    drop = set()
    i = 1
    while i < n - 1:
        if (i not in protected and (i + 1) not in protected
                and _is_assistant_pairable(messages[i]) and _is_all_tool_results(messages[i + 1])):
            ids_a, ids_r = _tool_use_ids(messages[i]), _tool_result_ids(messages[i + 1])
            if ids_a and ids_a == ids_r:               # exact set equality, non-empty, no extras/dups
                drop.add(i); drop.add(i + 1); i += 2; continue
        i += 1
    # 2) rebuild: keep protected verbatim, soft-rewrite other survivors, skip dropped
    out = []
    for idx, m in enumerate(messages):
        if idx in drop:
            continue
        out.append(m if idx in protected else _soft_rewrite(m))
    return out
```
Note `_tool_use_ids`/`_tool_result_ids` must dedupe-detect: if a message has duplicate ids, treat as
non-droppable (return a marker that fails equality) — simplest: build a `list`, and require
`len(list)==len(set(list))` on both sides before comparing sets.

### 1b. `turn()` restructure — wrap the stream in a one-retry loop
- Record `turn_start_len = len(self.messages)` **before** appending the user message.
- Replace the single `try: for event in stream …` with:
```python
retried = False
while True:
    assistant_blocks, current_text, stop_reason, yielded_any = [], "", "end_turn", False
    try:
        for event in self.provider.stream(self.messages, tools_spec, self.system_prompt):
            if self._cancelled:
                break
            if isinstance(event, TextDelta):
                current_text += event.text; yielded_any = True; yield event
            elif isinstance(event, ToolUseStart):
                if current_text:
                    assistant_blocks.append({"type": "text", "text": current_text}); current_text = ""
                yielded_any = True; yield event
            elif isinstance(event, ToolUse):
                assistant_blocks.append({"type": "tool_use", "id": event.id,
                                         "name": event.name, "input": event.input})
                yielded_any = True; yield event
            elif isinstance(event, Stop):
                stop_reason = event.reason
                for k, v in (event.usage or {}).items():
                    total_usage[k] = total_usage.get(k, 0) + int(v)
        break                                            # stream consumed cleanly
    except Exception as exc:
        if not yielded_any and not retried and _is_context_limit_error(exc):
            retried = True
            self.messages = _compact_history(self.messages)
            continue                                     # retry the stream once, post-compaction
        del self.messages[turn_start_len:]               # fixed cleanup (was: pop last user -> could orphan)
        yield TextDelta(f"\n[error: {exc}]\n")
        yield TurnDone("error", total_usage); return
```
Everything after (cancel check, append assistant_blocks, tool loop, tool_result backfill) is unchanged.
`retried` resets per outer `for _ in range(max_loops)` iteration (declare it at loop top).

## 2. `scat/agent/providers/anthropic_api.py`
- `__init__(self, api_key=None, model="claude-opus-4-8", max_tokens=4096, max_retries=3)`:
  `Anthropic(api_key=api_key, max_retries=max_retries) if api_key else Anthropic(max_retries=max_retries)`.
  One comment: SDK owns exp-backoff on 408/409/429/≥500. `max_tokens` used as today.

## 3. `scat/agent/backend.py` + `scat/config.py`
- `build_runner(..., max_tokens=None, max_retries=None)` — but cleaner: read config inside `build_runner`
  for the API branch: `AnthropicProvider(api_key=key, model=model, max_tokens=config.get("agent.max_tokens", 4096), max_retries=config.get("agent.max_retries", 3))`. (`config` import is core.)
- `config.py` `agent` section: add `"max_tokens": 4096, "max_retries": 3`.

## 4. Tests — `tests/test_agent_runner.py` (extend, stub providers only)
- **Pre-existing-bug fix:** a provider raising on the **2nd** iteration (after one real tool round via a
  real tool like scan_folder, or a stub `tool_caller`) → assert no orphaned `tool_use` in `r.messages`
  (build `{id: has_result}` and check all True). Keep `test_stream_error_cleans_history` (1st-iter → []).
- **`_compact_history` unit tests** (construct message lists directly, no provider):
  soft stub + verbatim keep of `messages[0]`/last 6 + `tool_use.input` intact; the five pair predicate
  cases (`tu:a,tu:b / tr:a` no-drop; `tu:a / tr:a,tr:x` no-drop; text-only+tr no-pair; `text,tu:a / tr:a`
  drop; boundary-straddle keep-both); duplicate-id no-drop; `n<=_KEEP_RECENT+1` → unchanged; **whole-list
  pairing valid after every case** (helper `_assert_paired(messages)`).
- **Reactive retry** (stub providers): 1st-call context-limit-then-succeed → one retry, completes;
  non-context error → no retry; yield-then-context-error → no retry + no duplicate text.
- **`_is_context_limit_error`:** true for known messages, false for `ValueError("bad tool schema")`,
  false for an exc with `status_code=400` but a non-context message, true for `status_code`-less + ctx msg.
- **Provider/config:** monkeypatch `anthropic.Anthropic` to capture kwargs → `AnthropicProvider(max_retries=7, max_tokens=99)` forwards both; `build_runner` API branch reads config defaults (monkeypatch build to API, capture provider).

## 5. Codex plan review — incorporated (pass 2, gpt-5.5 xhigh)
- **BLOCKER: rollback index invalidated by compaction.** Replace `turn_start_len` with a snapshot
  `pre_turn = list(self.messages)` taken before appending the user message; on the fatal give-up path
  `self.messages = pre_turn` (reverts the compaction too — always a valid pre-turn state). `_soft_rewrite`
  must **copy**, never mutate input dicts, so the snapshot stays intact.
- **BLOCKER: `yielded_any` ignored `Stop` → usage double-count.** Rename to `saw_event`, set it at the
  TOP of the event loop for **every** event (incl. `Stop`), before dispatch. Retry only when `not
  saw_event`.
- Cancel during a pre-event context error → don't retry: guard the except with `if self._cancelled:
  self.messages = pre_turn; yield TurnDone("cancelled", total_usage); self._cancelled=False; return`.
- Tighten `_CTX_RE` (drop bare `context`): `prompt is too long|input length and .*exceed|exceed\w*\s+
  context|context (?:window|length|limit)|too many tokens|maximum context`. Require `status_code==400`
  only when the attr exists; allow status-less message match.
- Duplicate ids → not droppable: build **lists**, require `len==len(set)` on both sides before comparing
  sets (no fake-marker set behind a `set[str]`).
- Tests: `_assert_paired` validates **adjacency** (each assistant tool_use answered by the *immediately
  following* user tool_result, exact id-set equality), not a global {id:has_result} map. Add:
  compact-then-retry-fails leaves the exact pre-turn prefix (no current user left behind);
  compact/retry-succeeds one round then next loop fails → no orphan + no turn-local remnant; Stop-only
  then exception → no retry, no usage double-count; `_is_context_limit_error` with `status_code=400`,
  `status_code=None`, and an incidental-"context" non-limit 400 → correct; duplicate ids on both sides.
- `build_runner`: read config **inside** the API branch; do **not** add unused params to its signature.

## Open confirmations
- `Stop`/`TextDelta`/`ToolUse`/`ToolUseStart` are already imported in runner.py (yes, line 6). `re` is new.
- `test_chat_widget.py:82` monkeypatches `backend.build_runner` — confirm the config reads don't break it
  (they're inside the real build_runner, which that test replaces, so unaffected).
