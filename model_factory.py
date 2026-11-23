"""
model_factory.py — one model factory shared by the LangChain-based agents (ThinkDepth.ai, AIQ).

Every model is reached through one OpenAI-compatible endpoint: OPENAI_BASE_URL + OPENAI_API_KEY
(set in this directory's .env.qwen or .env.sonnet; see the .env.*.example files). Claude models go through the
ChatAnthropic SDK (the endpoint must then also serve the Anthropic Messages API); all other models
(qwen, doubao, gpt, ...) go through ChatOpenAI.

Usage:
    from model_factory import create_model
    model = create_model("claude-sonnet-4-6")       # -> ChatAnthropic
    model = create_model("qwen3.5-plus")            # -> ChatOpenAI
"""

import json
import os
from langchain_core.language_models import BaseChatModel


def create_model(model_name: str, *, max_tokens: int = 32768, purpose: str | None = None) -> BaseChatModel:
    """根据 model_name 创建 LangChain ChatModel。

    Claude -> ChatAnthropic; every other model -> ChatOpenAI (OpenAI-compatible endpoint).
    `purpose` (e.g. "agent", "synthesis") is sent as the X-Cairn-Purpose header, which Cairn's comparison gateway
    records with each call; other endpoints ignore it.
    """
    headers = {"X-Cairn-Purpose": purpose} if purpose else None
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ["OPENAI_BASE_URL"]   # endpoint from the agent's .env

    if model_name.startswith("claude") or os.environ.get("RCA_FORCE_ANTHROPIC") == "1":
        from langchain_anthropic import ChatAnthropic

        # the Anthropic SDK expects base_url without /v1
        anthropic_url = base_url.rstrip("/")
        if anthropic_url.endswith("/v1"):
            anthropic_url = anthropic_url[:-3]

        # Coding-plan endpoint rate-limits (429) under sustained load; the SDK
        # default of 2 retries is too few for a 30-60 call rollout, so a single
        # exhausted 429 kills the whole trajectory. Raise retries so calls ride
        # out the rate-limit window via Retry-After backoff. Anthropic-branch
        # only (the qwen/OpenAI branch never reaches here).
        return ChatAnthropic(
            model=model_name,
            api_key=api_key,
            base_url=anthropic_url,
            max_tokens=max_tokens,
            max_retries=8,
            **({"default_headers": headers} if headers else {}),
        )
    else:
        from langchain_openai import ChatOpenAI

        # Transport only: ride out short 429/5xx bursts under concurrent runs
        # (SDK default is 2 retries). RCA_MAX_RETRIES overrides. RCA_LLM_EXTRA_BODY (JSON, e.g.
        # {"enable_thinking": false}) is passed through when set, as _rca_shared/llm.py does.
        # Sampling: temperature 0 unless RCA_TEMPERATURE is set; RCA_TOP_P, RCA_PRESENCE_PENALTY and RCA_TOP_K
        # (vLLM's top_k; top_k and top_p go in the extra body) only when set, so a comparison can match another
        # framework's sampling on the same backbone.
        extra = json.loads(os.environ["RCA_LLM_EXTRA_BODY"]) if os.environ.get("RCA_LLM_EXTRA_BODY") else {}
        if os.environ.get("RCA_TOP_K"):
            extra["top_k"] = int(os.environ["RCA_TOP_K"])
        if os.environ.get("RCA_TOP_P"):
            # In the extra body: this langchain-openai leaves ChatOpenAI(top_p=...) out of the request.
            extra["top_p"] = float(os.environ["RCA_TOP_P"])
        sampling = {"temperature": float(os.environ.get("RCA_TEMPERATURE", "0"))}
        if os.environ.get("RCA_PRESENCE_PENALTY"):
            sampling["presence_penalty"] = float(os.environ["RCA_PRESENCE_PENALTY"])
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            max_retries=int(os.environ.get("RCA_MAX_RETRIES", "6")),
            **sampling,
            **({"extra_body": extra} if extra else {}),
            **({"default_headers": headers} if headers else {}),
        )
