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


def create_model(model_name: str, *, max_tokens: int = 32768) -> BaseChatModel:
    """根据 model_name 创建 LangChain ChatModel。

    Claude -> ChatAnthropic; every other model -> ChatOpenAI (OpenAI-compatible endpoint).
    """
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
        )
    else:
        from langchain_openai import ChatOpenAI

        # Transport only: ride out short 429/5xx bursts under concurrent runs
        # (SDK default is 2 retries). RCA_MAX_RETRIES overrides. RCA_LLM_EXTRA_BODY (JSON, e.g.
        # {"enable_thinking": false}) is passed through when set, as _rca_shared/llm.py does.
        extra = os.environ.get("RCA_LLM_EXTRA_BODY")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=0,
            max_retries=int(os.environ.get("RCA_MAX_RETRIES", "6")),
            **({"extra_body": json.loads(extra)} if extra else {}),
        )
