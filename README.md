# litellm-prompt-injection-guardrail

A multi-layer security and routing guardrail for [LiteLLM](https://github.com/BerriAI/litellm) proxy — built to protect open-weight model endpoints from prompt injection, PII leakage, and sensitive content exposure.

## Architecture

The guardrail is organized in four layers that run in sequence before every request reaches a model:

| Layer | File | Guardrail | What it does |
|---|---|---|---|
| **0** | `pii_router.py` | `HardGuardrails` | Regex-based block/mask for secrets and PII patterns (local, zero-cost) |
| **1** | `pii_router.py` | `PIIRouter` | Presidio-based PII detection — routes PII-bearing requests to a trusted model |
| **1.5** | `injection_judge.py` | `InjectionJudge` | LLM judge (Haiku → Sonnet escalation) that strips prompt-injection segments from the newest user/tool message |
| **2** | `pii_router.py` | `AnthropicDecisionLayer` | Haiku classifier that decides safe vs borderline vs unsafe, routing to bulk/trusted/search models |

## Key Design Decisions

### Segment-level stripping (not full blocking)
The `InjectionJudge` only judges the **newest** user/tool message per request (earlier turns were already checked on their own requests). It splits messages into content-block segments and further on Claude Code's interrupt marker (`[Request interrupted by user]`). Only segments judged as "injection" are stripped — the rest of the thread goes upstream unchanged. One bad turn doesn't take down the whole conversation.

### Two-tier judging (Haiku → Sonnet)
- **Tier 1 (Haiku)** screens every segment. Anything not clearly "safe" is escalated.
- **Tier 2 (Sonnet)** re-judges escalated segments. Its verdict is final.
- A segment is only stripped if **both** tiers agree it's an attack — killing false positives while Haiku's uncertainty catches what Sonnet needs a second look at.
- Clearly-safe segments never reach Sonnet, keeping the common path fast and cheap.

### Out-of-band judge calls
The `InjectionJudge` uses a **direct Anthropic SDK call** (not the proxy) with a pinned `base_url`. This prevents the judge's own traffic from re-entering the guardrails recursively.

### PII routing
`PIIRouter` scans the **entire** request payload (user text + tool_result + file content) for PII using Microsoft Presidio. If found, the request is routed to a trusted model (`kimi-k2p6` by default). Routing is naturally sticky because LiteLLM resends the full conversation each turn — PII in history stays visible forever. An optional cache-based taint covers trimmed/compacted histories.

## Files

| File | Purpose |
|---|---|
| `injection_judge.py` | `InjectionJudge` guardrail class |
| `pii_router.py` | `PIIRouter`, `HardGuardrails`, `AnthropicDecisionLayer` classes |
| `judge_prompt.txt` | System prompt for the injection judge |
| `config.yaml` | LiteLLM proxy configuration (model list + guardrail wiring) |

## Setup

1. Copy `litellm.env.example` to `litellm.env` and fill in your keys:

```bash
cp litellm.env.example litellm.env
# edit litellm.env with your actual keys
```

2. Install dependencies:

```bash
pip install litellm presidio_analyzer anthropic
```

3. Start the proxy:

```bash
set -a && source litellm.env && set +a
litellm --config config.yaml --port 4000
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | For judge and decision layer |
| `FIREWORKS_API_KEY` | Yes | For open-weight Fireworks models |
| `LITELLM_MASTER_KEY` | Yes | LiteLLM proxy master key |
| `JUDGE_ESCALATION_MODEL` | No | Override the escalation model (default: `claude-sonnet-5`) |
| `JUDGE_BASE_URL` | No | Override the judge's API base URL (default: `https://api.anthropic.com`) |

## License

MIT
