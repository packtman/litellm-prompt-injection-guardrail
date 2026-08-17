"""
Prompt-injection LLM judge for LiteLLM (Layer 1.5).

Sits between HardGuardrails/PIIRouter and AnthropicDecisionLayer:
only the newest user/tool message is judged (prior turns were checked on
their own requests), via a DIRECT Anthropic SDK call — out-of-band, so it
never recurses through this proxy. Verdicts are cached by content hash.

The newest message is judged in segments (one per content block, further split
on Claude Code's interrupt marker). Only segments that judge as "injection" are
stripped from the payload — the rest of the message goes upstream unchanged, so
one bad turn doesn't take down the whole thread.

Two judging tiers: Haiku screens every segment, and anything it does not call
clearly "safe" is re-judged by Sonnet, whose verdict is final. A segment is only
stripped if both tiers agree it is an attack.

Policy (on the final verdict):
  - segment "injection"                      -> strip that segment; note it in the payload
  - every segment "injection"                -> block (400, nothing left to send)
  - segment "suspicious"                     -> allow, log it
  - Sonnet unreachable                       -> fall back to Haiku's verdict, don't cache
  - judge unavailable / error                -> allow, log loudly (fail-open)
  - own judge traffic (model == "injection-judge") is never re-judged
"""
import hashlib
import json
import os

from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail

_prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_prompt.txt")
with open(_prompt_path) as f:
    JUDGE_PROMPT = f.read()

JUDGE_MODEL_ID = "claude-haiku-4-5-20251001"   # direct Anthropic model id (not a proxy alias)
ESCALATION_MODEL_ID = os.environ.get("JUDGE_ESCALATION_MODEL", "claude-sonnet-5")
JUDGE_ALIAS = "injection-judge"                # must match model_name in config.yaml
JUDGE_MAX_CHARS = 4000                         # bounds judge cost per request
# Two tiers. Haiku screens every segment; anything it does not call clearly
# "safe" is re-judged by Sonnet, whose verdict is final. So a strip needs BOTH
# models to agree it is an attack (kills false positives), and Sonnet also gets
# a second look at what Haiku was merely unsure about (catches what Haiku
# missed). Clearly-safe segments never reach Sonnet, so the common path is
# Haiku-only.

_judge_client = None
_judge_import_error = None
try:
    from anthropic import AsyncAnthropic

    # IMPORTANT: never route the judge through this proxy. The shell env here sets
    # ANTHROPIC_BASE_URL -> the LiteLLM proxy, which would make the judge's own call
    # re-enter these guardrails recursively (and fail-open). Pin to the real API;
    # set JUDGE_BASE_URL in litellm.env only if you intentionally judge via another
    # gateway.
    _judge_client = AsyncAnthropic(
        base_url=os.environ.get("JUDGE_BASE_URL", "https://api.anthropic.com"),
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
except Exception as e:  # surfaced in proxy logs
    _judge_import_error = e


INTERRUPT_MARKER = "[Request interrupted by user]"
REMOVED_NOTE = "[content removed by prompt-injection guardrail]"


def _newest_user_message(messages):
    """The last user/tool message — the only untrusted content this request adds.

    Iterate FORWARD: the proxy appends its own system message after the user's,
    so a reverse scan that skips non-user roles can land on an older turn.
    """
    newest = None
    for m in messages or []:
        if m.get("role") in ("user", "tool"):
            newest = m
    return newest


def _block_text(block):
    """Text carried by a content block, or None for images/tool_result/etc."""
    if not isinstance(block, dict):
        return None
    text = block.get("text") or block.get("content")
    return text if isinstance(text, str) else None


def _set_block_text(block, text):
    key = "text" if "text" in block else "content"
    return {**block, key: text}


def _segments(text):
    """Split one text unit into independently-judgeable segments.

    Claude Code coalesces an interrupted turn and the turn the user retypes
    afterwards into ONE user message joined by INTERRUPT_MARKER. Judged as a
    single unit, an abandoned injection poisons every later turn in the thread.
    Each segment is still judged on its own, so a forged marker buys nothing.
    """
    return text.split(INTERRUPT_MARKER)


class InjectionJudge(CustomGuardrail):
    async def _judge(self, text, model_id):
        verbose_proxy_logger.info(
            f"InjectionJudge evaluating on {model_id} ({len(text)} chars): {text[:200]}")
        kwargs = dict(
            model=model_id,
            system=[{"type": "text", "text": JUDGE_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content":
                       f"<<<UNTRUSTED>>>\n{text[:JUDGE_MAX_CHARS]}\n<<<END_UNTRUSTED>>>"}],
            max_tokens=1000,
            timeout=20,
        )
        if model_id == JUDGE_MODEL_ID:
            # temperature is deprecated on the Sonnet tier and hard-400s there;
            # the Haiku screen still wants temperature=0 for stable verdicts.
            kwargs["temperature"] = 0
        resp = await _judge_client.messages.create(**kwargs)
        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        return json.loads(raw[raw.index("{"):raw.rindex("}") + 1])

    async def _verdict(self, cache, text):
        """Cached final verdict for one segment. None means "couldn't judge" (fail-open).

        Tier 1 Haiku screens; anything not clearly "safe" is re-judged by tier 2
        Sonnet, which overrules Haiku in both directions. Only the final verdict
        is cached, so a repeated segment costs nothing on either tier.
        """
        ckey = f"inj:{hashlib.sha256(text[:JUDGE_MAX_CHARS].encode()).hexdigest()[:16]}"
        try:
            cached = await cache.async_get_cache(ckey)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

        if _judge_client is None:
            verbose_proxy_logger.error(
                f"InjectionJudge unavailable (fail-open, allowing): {_judge_import_error}")
            return None
        try:
            verdict = await self._judge(text, JUDGE_MODEL_ID)
        except Exception as e:
            # Fail OPEN: availability over blocking — matches this gateway's other
            # layers being advisory on their non-PII paths.
            verbose_proxy_logger.error(f"InjectionJudge error (fail-open, allowing): {e}")
            return None

        if verdict.get("verdict") != "safe":
            try:
                escalated = await self._judge(text, ESCALATION_MODEL_ID)
                verbose_proxy_logger.info(
                    f"InjectionJudge escalated: haiku={verdict.get('verdict')} "
                    f"-> sonnet={escalated.get('verdict')}")
                escalated["escalated"] = True
                verdict = escalated
            except Exception as e:
                # Keep Haiku's call rather than dropping to unguarded: degraded,
                # still guarded. Not cached — retry the escalation next time.
                verbose_proxy_logger.error(
                    f"InjectionJudge escalation error (falling back to Haiku's verdict): {e}")
                verdict["escalation_failed"] = True
                return verdict

        try:
            await cache.async_set_cache(ckey, json.dumps(verdict), ttl=3600)
        except Exception:
            pass
        return verdict

    async def _clean(self, cache, text):
        """Judge each segment of `text`; return (surviving text, dropped verdicts)."""
        kept, dropped = [], []
        for seg in _segments(text):
            if not seg.strip():
                continue
            if len(seg) > JUDGE_MAX_CHARS * 4:
                # A judge only sees the first JUDGE_MAX_CHARS; past ~4x that, too
                # much of the segment is unjudged to act on the verdict.
                verbose_proxy_logger.info(
                    f"InjectionJudge skipped segment: {len(seg)} chars exceeds scan limit")
                kept.append(seg)
                continue
            verdict = await self._verdict(cache, seg)
            v = (verdict or {}).get("verdict", "safe")
            if v == "injection":
                dropped.append(verdict)
                verbose_proxy_logger.warning(f"InjectionJudge stripped segment: {verdict}")
                continue
            if v == "suspicious":
                verbose_proxy_logger.info(
                    f"InjectionJudge allowed suspicious segment: "
                    f"{str(verdict.get('evidence', ''))[:200]}")
            kept.append(seg)
        return "\n".join(s.strip() for s in kept).strip(), dropped

    @staticmethod
    def _blocked(verdicts):
        # 400, not a bare Exception: LiteLLM renders an uncaught Exception as a 500,
        # and clients retry 5xx (the Anthropic SDK burns all 10 attempts on it).
        # A guardrail rejection is deterministic — retrying can never help.
        v = verdicts[0] or {}
        return HTTPException(
            status_code=400,
            detail={"error": f"Blocked by prompt-injection guardrail: injection "
                             f"(confidence {v.get('confidence', 0)}, "
                             f"evidence: {str(v.get('evidence', ''))[:200]})"},
        )

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        # Never judge our own judge traffic.
        if data.get("model") == JUDGE_ALIAS:
            return data

        msg = _newest_user_message(data.get("messages"))
        if msg is None:
            return data
        content = msg.get("content")
        dropped = []

        if isinstance(content, str):
            cleaned, dropped = await self._clean(cache, content)
            if dropped:
                if not cleaned:
                    raise self._blocked(dropped)
                msg["content"] = f"{cleaned}\n{REMOVED_NOTE}"

        elif isinstance(content, list):
            blocks, kept_text = [], False
            for block in content:
                text = _block_text(block)
                if text is None or not text.strip():
                    blocks.append(block)          # images, tool_result, empties
                    continue
                cleaned, seg_dropped = await self._clean(cache, text)
                dropped.extend(seg_dropped)
                if not cleaned:
                    continue                      # whole block was injection — drop it
                kept_text = True
                blocks.append(_set_block_text(block, cleaned) if seg_dropped else block)
            if dropped:
                if not kept_text:
                    raise self._blocked(dropped)
                blocks.append({"type": "text", "text": REMOVED_NOTE})
                msg["content"] = blocks

        if dropped:
            verbose_proxy_logger.warning(
                f"InjectionJudge stripped {len(dropped)} injection segment(s), allowed the rest")
            data.setdefault("metadata", {})["injection_stripped"] = len(dropped)
        return data
