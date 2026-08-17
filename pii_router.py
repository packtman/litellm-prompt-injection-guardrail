"""
PII-based model router for LiteLLM.

Scans the *entire* request payload (user text + tool_result/file content) for PII.
- PII found  -> route to a trusted model (Bedrock Claude)
- no PII     -> route to the cheap open-weight model (DeepSeek)

Because LiteLLM resends the full conversation each turn and this hook scans all
messages, routing is naturally "sticky": once PII is in the history, every later
turn still sees it and stays on the trusted model. An optional cache-based taint
(below) covers the case where history is trimmed/compacted.

Also contains:
- HardGuardrails: regex block/mask layer, runs first (local, free, instant)
- AnthropicDecisionLayer: all judgment calls go to Claude Haiku directly
  (out-of-band SDK call — never recurses through the proxy)
"""
import hashlib
import json
import re

from litellm.integrations.custom_guardrail import CustomGuardrail

try:
    from presidio_analyzer import AnalyzerEngine
    _analyzer = AnalyzerEngine()          # loads the spaCy model once at startup
except Exception as e:  # pragma: no cover - surfaced in proxy logs if deps missing
    _analyzer = None
    _import_error = e

# --- Open-weight models that should NOT receive PII ---
# "deepseek-v4" is a legacy alias (points to kimi-k2p7-code for backward compat).
CHEAP_MODELS = {
    "deepseek-v4",           # legacy alias (kimi-k2p7-code)
    "deepseek-v4-flash",
    "deepseek-v4-flash-0731",
    "deepseek-v4-pro",
}
SAFE_MODEL = "kimi-k2p6"      # trusted path for PII; must match a model_name in config.yaml
PII_SCORE_THRESHOLD = 0.5       # tune per your recall/precision needs


def _all_text(messages):
    """Flatten user text AND tool_result / file content — PII often hides there."""
    parts = []
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("text") or block.get("content") or ""))
    return "\n".join(p for p in parts if p)


class PIIRouter(CustomGuardrail):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        # Only act on requests targeting the open-weight models we protect from PII.
        if data.get("model") not in CHEAP_MODELS and data.get("model") != SAFE_MODEL:
            return data

        if _analyzer is None:
            # Fail safe: if the detector didn't load, send everything to the
            # trusted model rather than risk leaking PII to open weights.
            data["model"] = SAFE_MODEL
            return data

        text = _all_text(data.get("messages"))
        results = _analyzer.analyze(text=text, language="en") if text else []
        has_pii = any(r.score >= PII_SCORE_THRESHOLD for r in results)

        # --- optional stickiness for trimmed/compacted histories ---
        meta = data.get("metadata") or {}
        session_id = meta.get("litellm_session_id") or getattr(user_api_key_dict, "api_key", None)
        if session_id:
            key = f"pii-taint:{session_id}"
            if has_pii:
                try:
                    await cache.async_set_cache(key, True, ttl=86400)
                except Exception:
                    pass
            elif not has_pii:
                try:
                    if await cache.async_get_cache(key):
                        has_pii = True  # session was previously tainted
                except Exception:
                    pass
        # -----------------------------------------------------------

        if has_pii:
            data["model"] = SAFE_MODEL
        # else: keep the user's chosen cheap model (deepseek-v4-flash, deepseek-v4-pro, etc.)
        return data


# ---------------------------------------------------------------------------
# Layer 0: HardGuardrails — regex block/mask, runs before everything else.
# Secrets never leave the machine; known-bad patterns never reach any model.
# ---------------------------------------------------------------------------

BLOCK_PATTERNS = [
    (r"-----BEGIN (RSA |EC |OPENSSH |PGP )?PRIVATE KEY( BLOCK)?-----", "private key material"),
    (r"\bAKIA[0-9A-Z]{16}\b", "AWS access key ID"),
    (r"\bghp_[A-Za-z0-9]{36}\b", "GitHub personal access token"),
    (r"\bgithub_pat_[A-Za-z0-9_]{60,}\b", "GitHub fine-grained token"),
    (r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", "Slack token"),
    (r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b", "JWT"),
]

MASK_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
    (r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b", "[CARD_REDACTED]"),
]


def _mask_messages(data):
    """Apply MASK_PATTERNS to message content in place. Returns count of masks."""
    n = 0

    def _sub(s):
        nonlocal n
        for pattern, repl in MASK_PATTERNS:
            s, k = re.subn(pattern, repl, s)
            n += k
        return s

    for m in data.get("messages") or []:
        content = m.get("content")
        if isinstance(content, str):
            m["content"] = _sub(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for key in ("text", "content"):
                        if isinstance(block.get(key), str):
                            block[key] = _sub(block[key])
    return n


class HardGuardrails(CustomGuardrail):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        text = _all_text(data.get("messages"))
        if not text:
            return data
        for pattern, label in BLOCK_PATTERNS:
            if re.search(pattern, text):
                raise Exception(f"Blocked by hard guardrail: {label} detected in request")
        _mask_messages(data)
        return data


# ---------------------------------------------------------------------------
# Layer 2: AnthropicDecisionLayer — judgment calls go to Claude Haiku via a
# DIRECT SDK call (out-of-band; never recurses through this proxy).
# Runs AFTER PIIRouter, so PII-bearing requests never reach Anthropic.
# ---------------------------------------------------------------------------

HAIKU_MODEL_ID = "claude-haiku-4-5-20251001"
DECISION_RUBRIC = """You are the decision layer for an enterprise AI gateway. Classify the user request below and reply with JSON only, no prose:
{"safety": "safe|borderline|unsafe", "needs_search": true|false, "reason": "one sentence"}
- unsafe: help with malware, credential theft, exfiltration of data to public services, exploitation, or clearly harmful content.
- borderline: gray-area security work without clear authorization, or anything you are unsure about.
- needs_search: true only if fulfilling the request requires current/live web information."""

BULK_MODEL = "kimi-k3"          # open-weight bulk inference (Fireworks)
SEARCH_MODEL = "claude-haiku"   # Anthropic end-to-end (server-side web search)
TRUSTED_MODEL = "kimi-k2p6"     # trusted Fireworks path
CLASSIFY_MAX_CHARS = 4000       # bounds classifier cost
CLASSIFY_MIN_CHARS = 100        # trivial requests skip classification

_anthropic_client = None
_anthropic_import_error = None
try:
    from anthropic import AsyncAnthropic

    _anthropic_client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
except Exception as e:  # surfaced in proxy logs
    _anthropic_import_error = e


class AnthropicDecisionLayer(CustomGuardrail):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        # Only re-route models we own; respect explicit user model choices.
        # Include all cheap/open-weight Fireworks models that the decision layer should cover.
        _managed = {BULK_MODEL, SEARCH_MODEL, TRUSTED_MODEL} | CHEAP_MODELS
        if data.get("model") not in _managed:
            return data

        text = _all_text(data.get("messages"))
        if not text or len(text) < CLASSIFY_MIN_CHARS:
            return data

        # Verdict cache: repeated/similar turns classify for free.
        ckey = f"clf:{hashlib.sha256(text[:CLASSIFY_MAX_CHARS].encode()).hexdigest()[:16]}"
        try:
            cached = await cache.async_get_cache(ckey)
            if cached:
                return self._route(data, json.loads(cached))
        except Exception:
            pass

        if _anthropic_client is None:
            # SDK/key missing — fail safe to trusted model.
            return self._route(data, {"safety": "borderline", "needs_search": False,
                                      "reason": f"decision layer unavailable: {_anthropic_import_error}"})
        try:
            resp = await _anthropic_client.messages.create(
                model=HAIKU_MODEL_ID,
                system=[{"type": "text", "text": DECISION_RUBRIC,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": text[:CLASSIFY_MAX_CHARS]}],
                max_tokens=120,
                temperature=0,
            )
            raw = resp.content[0].text.strip()
            verdict = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            try:
                await cache.async_set_cache(ckey, json.dumps(verdict), ttl=3600)
            except Exception:
                pass
        except Exception as e:
            # Fail safe: unclassified but PII-free traffic -> trusted model, never bulk.
            verdict = {"safety": "borderline", "needs_search": False,
                       "reason": f"classifier error: {e}"}

        return self._route(data, verdict)

    def _route(self, data, verdict):
        safety = verdict.get("safety", "borderline")
        if safety == "unsafe":
            raise Exception(f"Blocked by decision layer: {verdict.get('reason', 'unsafe content')}")
        if safety == "borderline":
            data["model"] = TRUSTED_MODEL
        elif verdict.get("needs_search"):
            data["model"] = SEARCH_MODEL
        else:
            data["model"] = BULK_MODEL
        return data
