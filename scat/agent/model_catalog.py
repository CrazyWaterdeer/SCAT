"""Which Claude models the assistant offers, and how "Latest" resolves to a real one.

Two halves, both deliberately import-light (no ``anthropic`` at import time — the GUI
imports this module to build its picker):

* **The ``latest`` sentinel.** ``agent.model`` may hold :data:`AUTO` instead of a pinned id,
  which means "whatever the newest Claude is today". It resolves per backend at
  runner-build time: the subscription path hands the ``claude`` CLI its family *alias*
  (``opus``), which the CLI itself resolves to the newest Opus; the API path resolves it
  against the model catalog. Neither needs anyone to edit this file when Anthropic ships.
* **The catalog.** Live from the Models API (``client.models.list()``) when an API key is
  available, cached on disk for a day; :data:`FALLBACK_MODELS` when there is no key, no
  network, or no ``anthropic`` install.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time

# The sentinel stored in ``agent.model`` for "always the newest model". Not a real model id —
# always run it through :func:`resolve_model` before handing it to a backend.
AUTO = "latest"
AUTO_LABEL = "Latest (auto)"

# Offered when the live catalog is unavailable (no API key / offline / no SDK). Newest-first
# within each family. This is a *fallback*, not the source of truth: ``latest`` never depends
# on it while a subscription or an API key is in play.
FALLBACK_MODELS: list[tuple[str, str]] = [
    ("Opus 5", "claude-opus-5"),
    ("Opus 4.8", "claude-opus-4-8"),
    ("Fable 5.1", "claude-fable-5-1"),
    ("Sonnet 5", "claude-sonnet-5"),
    ("Haiku 4.5", "claude-haiku-4-5"),
]

# Picker/preference order of the model families. Opus leads because it is what ``latest``
# targets: the assistant's work is tool-driven analysis, where Opus is the right default.
_FAMILIES = ("opus", "fable", "mythos", "sonnet", "haiku")

# Family that ``latest`` resolves to, and the alias the `claude` CLI understands for it.
AUTO_FAMILY = "opus"

# Current-generation ids: claude-<family>-<major>[-<minor>][-<YYYYMMDD>]. The optional date is
# what some models ship as their canonical API id (claude-haiku-4-5-20251001), so it must be
# accepted — but the legacy shape (claude-3-5-sonnet-20241022, family in the middle) is still
# excluded, which is what keeps the picker to what is current.
_MODEL_RE = re.compile(r"^claude-(" + "|".join(_FAMILIES) + r")-(\d+)(?:-(\d+))?(?:-(\d{8}))?$")

_CACHE_FILE = "model_catalog.json"
_CACHE_TTL = 24 * 3600.0      # a day: new models are announced, not hourly events
_FETCH_TIMEOUT = 6.0          # bounded: the first send resolves the model on the GUI thread


# --------------------------------------------------------------------------- ordering
def _parse(model_id: str) -> tuple[str, int, int, int] | None:
    """(family, major, minor, snapshot date) for a current-generation id, else None."""
    m = _MODEL_RE.match(model_id or "")
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3) or 0), int(m.group(4) or 0)


def _sort_key(model_id: str) -> tuple[int, int, int, int]:
    """Family order first, then newest version — so index 0 of a family is its newest.

    Version, not the API's ``created_at`` order: "latest" here means the newest model *of the
    family we want*, and a newer release of another tier must never outrank it.
    """
    parsed = _parse(model_id)
    if parsed is None:
        return (len(_FAMILIES), 0, 0, 0)
    family, major, minor, date = parsed
    return (_FAMILIES.index(family), -major, -minor, -date)


def display_name(model_id: str) -> str:
    """"claude-opus-4-8" -> "Opus 4.8" (the id itself for anything unrecognized)."""
    parsed = _parse(model_id)
    if parsed is None:
        return model_id
    family, major, minor, _date = parsed
    version = f"{major}.{minor}" if minor else str(major)
    return f"{family.capitalize()} {version}"


# --------------------------------------------------------------------------- disk cache
def _cache_path():
    from scat.config import get_config_dir
    return get_config_dir() / _CACHE_FILE


def load_cache(ttl: float = _CACHE_TTL) -> list[tuple[str, str]] | None:
    """The cached catalog when it is younger than *ttl*, else None (never raises).

    Entries are re-validated on the way out: a cache file that is JSON but not a list of
    current-generation ids is treated as absent, never handed to a backend.
    """
    try:
        raw = json.loads(_cache_path().read_text(encoding="utf-8"))
        if time.time() - float(raw["fetched_at"]) > ttl:
            return None
        models = [(str(name), str(mid)) for name, mid in raw["models"] if _parse(str(mid))]
        return models or None
    except Exception:
        return None


def save_cache(models: list[tuple[str, str]]) -> None:
    """Best-effort, and atomic: a half-written cache would look like a valid short catalog to
    the next process, so write a temp file and rename it into place."""
    path = _cache_path()
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps({"fetched_at": time.time(), "models": [list(m) for m in models]}, indent=2),
            encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except Exception:
            pass


# --------------------------------------------------------------------------- live catalog
def fetch_live_models(api_key: str, timeout: float = _FETCH_TIMEOUT) -> list[tuple[str, str]]:
    """Ask the Models API what exists today. Raises on any failure — callers fall back."""
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key, timeout=timeout, max_retries=1)
    # One entry per released version: a model listed both undated and as dated snapshots
    # collapses to the undated alias (or the newest snapshot when that is all there is).
    best: dict[tuple[str, int, int], tuple[str, str, int]] = {}
    for info in client.models.list(limit=100).data:
        mid = getattr(info, "id", "")
        parsed = _parse(mid)
        if parsed is None:
            continue                       # legacy shape — keep the picker to what's current
        family, major, minor, date = parsed
        name = (getattr(info, "display_name", "") or display_name(mid)).replace("Claude ", "").strip()
        key = (family, major, minor)
        prev = best.get(key)
        if prev is None or date == 0 or (prev[2] != 0 and date > prev[2]):
            best[key] = (display_name(mid) if date else name, mid, date)
    if not best:
        raise RuntimeError("Models API returned no current-generation models")
    entries = [(name, mid) for name, mid, _date in best.values()]
    entries.sort(key=lambda e: _sort_key(e[1]))
    return entries


def _api_key(explicit: str | None = None) -> str:
    """The key to talk to the Models API with: explicit, else env, else Settings."""
    if explicit:
        return explicit.strip()
    import os
    from scat.config import config
    return (os.environ.get("ANTHROPIC_API_KEY") or "").strip() or (config.get("agent.api_key") or "").strip()


def catalog(api_key: str | None = None, allow_fetch: bool = True,
            ttl: float = _CACHE_TTL) -> list[tuple[str, str]]:
    """The model list to offer: fresh cache, else a live fetch, else :data:`FALLBACK_MODELS`.

    ``allow_fetch=False`` keeps this strictly offline (the GUI builds its picker that way, so
    opening the dock never waits on the network); the fetch happens on the API path in
    :func:`resolve_model`, which is already about to make a network call anyway.
    """
    cached = load_cache(ttl)
    if cached is not None:
        return cached
    key = _api_key(api_key)
    if allow_fetch and key:
        try:
            models = fetch_live_models(key)
        except Exception:
            pass                            # offline / API hiccup — fall through to what we know
        else:
            save_cache(models)
            return models
    # Last-known-good beats the static list: a stale real catalog was true once, the constant
    # below is only ever as fresh as the last SCAT release.
    stale = load_cache(ttl=float("inf"))
    return stale if stale is not None else list(FALLBACK_MODELS)


def available_models(api_key: str | None = None, allow_fetch: bool = False) -> list[tuple[str, str]]:
    """Picker entries: "Latest (auto)" first, then every concrete model in the catalog."""
    return [(AUTO_LABEL, AUTO)] + catalog(api_key, allow_fetch=allow_fetch)


# --------------------------------------------------------------------------- resolution
def newest(family: str = AUTO_FAMILY, api_key: str | None = None,
           allow_fetch: bool = True) -> str:
    """Newest concrete id in *family* — the catalog is family-ordered and newest-first."""
    models = catalog(api_key, allow_fetch=allow_fetch)
    for _name, mid in models:
        parsed = _parse(mid)
        if parsed and parsed[0] == family:
            return mid
    # Fail closed to a known model of the family we asked for. Falling through to whatever the
    # catalog happened to list first could hand the user a different (pricier) tier silently.
    for _name, mid in FALLBACK_MODELS:
        parsed = _parse(mid)
        if parsed and parsed[0] == family:
            return mid
    return FALLBACK_MODELS[0][1]


def prefetch(api_key: str | None = None) -> None:
    """Warm the catalog cache in a daemon thread (no-op without a key).

    The GUI calls this when the chat dock opens so the first send — which resolves the model on
    the GUI thread — normally finds a warm cache instead of waiting on the network.
    """
    if not _api_key(api_key):
        return

    def _warm():
        try:
            catalog(api_key, allow_fetch=True)
        except Exception:
            pass

    threading.Thread(target=_warm, name="scat-model-catalog", daemon=True).start()


def resolve_model(model: str | None, backend: str = "api", api_key: str | None = None) -> str:
    """Turn a stored ``agent.model`` into something a backend accepts.

    A pinned id passes through untouched. :data:`AUTO` becomes the CLI family alias on the
    subscription path (the CLI resolves it to the newest release, so SCAT never lags a launch)
    and the newest catalog id on the API path (the Messages API takes no ``-latest`` aliases).
    """
    if model and model != AUTO:
        return model
    if backend == "subscription":
        return AUTO_FAMILY          # `claude --model opus` == the latest Opus
    return newest(AUTO_FAMILY, api_key=api_key)


def describe_model(configured: str | None, resolved: str) -> str:
    """How the status line names the model — an auto pick shows what it landed on."""
    return resolved if (configured and configured != AUTO) else f"{resolved} (latest)"
