#!/usr/bin/env python3
"""Score the sample corpus against the real two-tier judge.

This calls the actual `InjectionJudge._verdict` from injection_judge.py — the
same Haiku-screen -> Sonnet-escalation path the proxy uses at request time — so
what you see here is what the guardrail would decide in production.

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/run_judge.py                       # score every case
    python examples/run_judge.py --id delimiter-escape # score one
    python examples/run_judge.py --json > results.json

Needs a real API key and network access; each case is 1-2 model calls.
For the offline version, see examples/demo_pre_call.py.

Exit code is 1 if any case lands on a verdict it does not accept.
"""
import argparse
import asyncio
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import injection_judge  # noqa: E402
from injection_judge import InjectionJudge  # noqa: E402

DEFAULT_CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_cases.yaml")

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    ("\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m")
    if sys.stdout.isatty() else ("",) * 6
)
COLOR = {"safe": GREEN, "suspicious": YELLOW, "injection": RED}


class MemoryCache:
    """Stands in for LiteLLM's DualCache — same two methods the judge calls."""

    def __init__(self):
        self.store = {}
        self.hits = 0

    async def async_get_cache(self, key):
        if key in self.store:
            self.hits += 1
        return self.store.get(key)

    async def async_set_cache(self, key, value, ttl=None):
        self.store[key] = value


async def score(judge, cache, case, sem):
    async with sem:
        verdict = await judge._verdict(cache, case["text"])
    accepted = {case["expect"], *case.get("accept", [])}
    actual = (verdict or {}).get("verdict", "unjudged")
    escalated = bool((verdict or {}).get("escalated"))
    # A borderline case asserts on the two-tier path firing, not just the label:
    # if Haiku screened it clearly-safe, tier 2 never got its say.
    passed = actual in accepted and (escalated or not case.get("escalates"))
    return {
        "id": case["id"],
        "category": case["category"],
        "expect": case["expect"],
        "accept": sorted(accepted),
        "actual": actual,
        "passed": passed,
        "escalates_expected": bool(case.get("escalates")),
        "confidence": (verdict or {}).get("confidence"),
        "attack_types": (verdict or {}).get("attack_types", []),
        "evidence": (verdict or {}).get("evidence", ""),
        "escalated": escalated,
        "escalation_failed": bool((verdict or {}).get("escalation_failed")),
    }


def render(rows, cache):
    width = max(len(r["id"]) for r in rows)
    print(f"\n{BOLD}{'case'.ljust(width)}  expect      actual      conf  tier{RESET}")
    print("-" * (width + 40))
    for r in rows:
        mark = f"{GREEN}PASS{RESET}" if r["passed"] else f"{RED}FAIL{RESET}"
        conf = f"{r['confidence']:.2f}" if isinstance(r["confidence"], (int, float)) else "  - "
        tier = "haiku+sonnet" if r["escalated"] else "haiku"
        if r["escalation_failed"]:
            tier = f"{YELLOW}haiku (tier 2 unreachable){RESET}{DIM}"
        elif r["escalates_expected"] and not r["escalated"]:
            tier = f"{RED}haiku (expected tier 2){RESET}{DIM}"
        color = COLOR.get(r["actual"], "")
        print(f"{r['id'].ljust(width)}  {r['expect']:<11} {color}{r['actual']:<11}{RESET} "
              f"{conf}  {DIM}{tier}{RESET}  {mark}")
        if not r["passed"] and r["evidence"]:
            print(f"{' ' * width}  {DIM}evidence: {r['evidence'][:90]}{RESET}")

    passed = sum(r["passed"] for r in rows)
    escalated = sum(r["escalated"] for r in rows)
    print(f"\n{BOLD}{passed}/{len(rows)} passed{RESET} · {escalated} escalated to Sonnet "
          f"· {cache.hits} cache hits")
    for r in rows:
        if r["passed"]:
            continue
        if r["actual"] not in r["accept"]:
            print(f"  {RED}FAIL{RESET} {r['id']}: expected one of "
                  f"{'/'.join(r['accept'])}, got {r['actual']}")
        else:
            print(f"  {RED}FAIL{RESET} {r['id']}: verdict {r['actual']} is fine, but the "
                  f"case should have escalated to tier 2 and did not")
    return passed == len(rows)


async def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=DEFAULT_CASES, help="corpus YAML (default: sample_cases.yaml)")
    ap.add_argument("--id", action="append", dest="ids", help="only score these case ids (repeatable)")
    ap.add_argument("--category", help="only score cases in this category")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel judge calls (default: 4)")
    ap.add_argument("--json", action="store_true", help="emit results as JSON instead of a table")
    args = ap.parse_args()

    if injection_judge._judge_client is None:
        sys.exit(f"judge unavailable: {injection_judge._judge_import_error}\n"
                 f"Set ANTHROPIC_API_KEY, or run examples/demo_pre_call.py for the offline demo.")

    with open(args.cases) as f:
        cases = yaml.safe_load(f)
    if args.ids:
        cases = [c for c in cases if c["id"] in set(args.ids)]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if not cases:
        sys.exit("no cases matched")

    if not args.json:
        print(f"{DIM}tier 1: {injection_judge.JUDGE_MODEL_ID}  ·  "
              f"tier 2: {injection_judge.ESCALATION_MODEL_ID}  ·  "
              f"{len(cases)} cases{RESET}")

    judge, cache = InjectionJudge(), MemoryCache()
    sem = asyncio.Semaphore(args.concurrency)
    rows = await asyncio.gather(*(score(judge, cache, c, sem) for c in cases))

    if args.json:
        json.dump({"cases": rows, "passed": sum(r["passed"] for r in rows), "total": len(rows)},
                  sys.stdout, indent=2)
        print()
        ok = all(r["passed"] for r in rows)
    else:
        ok = render(rows, cache)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
