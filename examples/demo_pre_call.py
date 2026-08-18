#!/usr/bin/env python3
"""Show what the guardrail does to a request payload — offline, no API key.

`run_judge.py` answers "what verdict does a segment get?". This answers the
next question: "what does the request look like after the guardrail is done
with it?" It runs the real `InjectionJudge.async_pre_call_hook`, with only the
model call stubbed out — verdicts are read from examples/sample_cases.yaml
instead of Anthropic, so the whole flow is deterministic and free.

    python examples/demo_pre_call.py

Four scenarios, one per policy branch in the hook:
  1. partial strip  — an interrupted turn where only the abandoned half attacks
  2. block dropped  — indirect injection in one content block of several
  3. full block     — nothing survives, so the request is rejected 400
  4. suspicious     — allowed through untouched, logged only
"""
import asyncio
import json
import os
import sys

import yaml
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from injection_judge import INTERRUPT_MARKER, InjectionJudge  # noqa: E402

CASES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_cases.yaml")

BOLD, DIM, RED, GREEN, CYAN, RESET = (
    ("\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[36m", "\033[0m")
    if sys.stdout.isatty() else ("",) * 6
)

with open(CASES_PATH) as f:
    CASES = {c["id"]: c for c in yaml.safe_load(f)}

# Segment text -> the verdict the corpus says it deserves. Keyed on the same
# stripped text the hook passes to _verdict.
CORPUS_VERDICTS = {
    c["text"].strip(): {
        "verdict": c["expect"],
        "confidence": 0.95 if c["expect"] == "injection" else 0.6,
        "attack_types": [c["category"]] if c["expect"] == "injection" else [],
        "evidence": c["text"].strip().splitlines()[0][:80],
    }
    for c in CASES.values()
}


async def stub_verdict(self, cache, text):
    """Drop-in for InjectionJudge._verdict: corpus lookup instead of a model call."""
    return CORPUS_VERDICTS.get(text.strip(), {"verdict": "safe", "confidence": 0.99,
                                              "attack_types": [], "evidence": ""})


def text_of(case_id):
    return CASES[case_id]["text"].strip()


def show(title, why, before, after, error=None):
    print(f"\n{BOLD}{'─' * 78}\n{title}{RESET}\n{DIM}{why}{RESET}\n")
    print(f"{CYAN}before{RESET}")
    print(json.dumps(before, indent=2)[:1400])
    if error is not None:
        print(f"\n{RED}rejected{RESET}  HTTP {error.status_code}")
        print(json.dumps(error.detail, indent=2))
    else:
        print(f"\n{GREEN}after{RESET}")
        print(json.dumps(after, indent=2)[:1400])
        stripped = (after.get("metadata") or {}).get("injection_stripped", 0)
        print(f"\n{DIM}metadata.injection_stripped = {stripped}{RESET}")


async def run(judge, cache, title, why, payload):
    before = json.loads(json.dumps(payload))  # deep copy; the hook mutates in place
    try:
        after = await judge.async_pre_call_hook(None, cache, payload, "completion")
        show(title, why, before, after)
    except HTTPException as e:
        show(title, why, before, None, error=e)


async def main():
    InjectionJudge._verdict = stub_verdict  # the only thing faked
    judge, cache = InjectionJudge(), None   # cache is unused once _verdict is stubbed

    await run(
        judge, cache,
        "1. Partial strip — interrupted turn",
        "Claude Code joins an interrupted turn and the retyped one into a single "
        "user message.\n_segments() splits on the interrupt marker, so the abandoned "
        "attack is dropped\nand the honest question still reaches the model.",
        {"model": "kimi-k2", "messages": [
            {"role": "user", "content":
                f"{text_of('interrupt-marker-smuggle')}\n{INTERRUPT_MARKER}\n"
                f"{text_of('interrupt-marker-retype')}"}]},
    )

    await run(
        judge, cache,
        "2. Block dropped — indirect injection in fetched content",
        "The user's own question is fine; a README pulled in by a tool carries the "
        "payload.\nOnly that content block is dropped. The image block and the "
        "question go upstream.",
        {"model": "kimi-k2", "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": text_of("academic-question")},
                {"type": "text", "text": text_of("indirect-injection-in-tool-output")},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": "iVBORw0KGgo..."}}]}]},
    )

    await run(
        judge, cache,
        "3. Full block — nothing survives",
        "Every segment judged injection, so there is no request left to forward. "
        "Returned as\na 400, not a 500: a guardrail rejection is deterministic and "
        "must not be retried.",
        {"model": "kimi-k2", "messages": [
            {"role": "user", "content": text_of("classic-override")}]},
    )

    await run(
        judge, cache,
        "4. Suspicious — allowed, logged",
        "\"suspicious\" is uncertainty, not a block. Haiku calls this one injection, "
        "and when\nSonnet overrules it on the second tier the segment goes upstream "
        "untouched —\na strip needs both tiers to agree.",
        {"model": "kimi-k2", "messages": [
            {"role": "user", "content": text_of("persistent-format-override")}]},
    )

    print(f"\n{DIM}{'─' * 78}\nVerdicts came from sample_cases.yaml. To run the real "
          f"two-tier judge:\n  ANTHROPIC_API_KEY=sk-ant-... python examples/run_judge.py{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
