#!/usr/bin/env python
"""agent_runner.py — ThinkDepth.ai RCA adapter.

stdin:  JSON { question, system_prompt, user_prompt, compress_system_prompt, compress_user_prompt, data_dir }
        data_dir holds the case's telemetry tables: <table>.parquet or <table>/*.parquet.
stdout: JSON { output (CausalGraph JSON), graph_valid, trajectory (OpenAI format), usage,
                run_status, stop_reason, run_error }

Output caps: RCA_MAX_TOKENS for the Analyst (default 32768), RCA_COMPRESS_MAX_TOKENS for the synthesis (default
32000). A request the server refuses for its context length is not an infrastructure failure: the Analyst loop
ends there and the synthesis runs on what was gathered; if the synthesis is refused too, the run fails with
stop_reason context_length and an empty output (exit 0, so the harness stores it instead of retrying).
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

_AGENT_DIR = Path(__file__).resolve().parent
RCA_MODEL = os.environ.get("RCA_MODEL", "claude-sonnet-4-6")
_ENV_OVERRIDE = os.environ.get("RCA_ENV_FILE")
if _ENV_OVERRIDE:
    _ENV_PATH = Path(_ENV_OVERRIDE).expanduser()
    if not _ENV_PATH.is_absolute():
        _ENV_PATH = _AGENT_DIR / _ENV_PATH
else:
    _ENV_PATH = _AGENT_DIR / (".env.sonnet" if RCA_MODEL.startswith("claude") else ".env.qwen")
    if not _ENV_PATH.exists():
        _ENV_PATH = _AGENT_DIR / ".env"
load_dotenv(_ENV_PATH, override=False)

# _rca_shared is a sibling directory; RCA_SHARED_PARENT overrides its parent when this repo stands alone.
sys.path.insert(0, os.environ.get("RCA_SHARED_PARENT") or str(_AGENT_DIR.parent))
from _rca_shared.events import EV  # noqa: E402
from _rca_shared.usage_tracker import UsageTracker  # noqa: E402

_tracker = UsageTracker()
_tracker.install_openai_hooks()
if RCA_MODEL.startswith("claude"):
    _tracker.install_anthropic_hooks()
EV.install_openai_hooks()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from deep_research.prompts import rca_telemetry_prompt, rca_think_prompt  # noqa: E402
from deep_research.rca_tools import get_schema, list_tables_in_directory, query_parquet_files  # noqa: E402
from deep_research.state_research import ResearcherState  # noqa: E402
from deep_research.utils import think_tool  # noqa: E402
from model_factory import create_model  # noqa: E402

logger = logging.getLogger("agent_runner")

RCA_TOOLS = [think_tool, list_tables_in_directory, get_schema, query_parquet_files]
RECURSION_LIMIT = 300
ANALYST_MAX_TOKENS = int(os.environ.get("RCA_MAX_TOKENS", "32768"))
COMPRESS_MAX_TOKENS = int(os.environ.get("RCA_COMPRESS_MAX_TOKENS", "32000"))
CONTEXT_MARKERS = ("context length", "maximum context", "prompt is too long", "max_model_len")
# Why the run stopped other than by the Analyst finishing: set by the graph nodes, read by main().
RUN: dict = {"stop_reason": None, "error": None}


def _make_model(max_tokens: int = 32768, purpose: str | None = None):
    """Create the LLM through model_factory; the model name comes from RCA_MODEL."""
    return create_model(RCA_MODEL, max_tokens=max_tokens, purpose=purpose)


def is_context_length(exc: Exception) -> bool:
    """The server refused the request for its context window (HTTP 400 with vLLM's wording)."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return status == 400 and any(m in str(exc).lower() for m in CONTEXT_MARKERS)


def _content_to_text(content) -> str:
    """Plain text of an LLM message: a str unchanged; of Anthropic content blocks, only the text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def strip_markdown_json(text: str) -> str:
    """Take the JSON out of a ```json ... ``` block."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ── graph nodes ──────────────────────────────────────────────────────────────

def make_analyst_call(system_prompt: str, tools: list):
    model_with_tools = _make_model(ANALYST_MAX_TOKENS, purpose="agent").bind_tools(tools)

    def analyst_call(state: ResearcherState) -> dict:
        try:
            with EV.scope(role="analyst"):
                response = model_with_tools.invoke(
                    [SystemMessage(content=system_prompt)] + list(state["researcher_messages"]))
        except Exception as exc:
            if not is_context_length(exc):
                raise
            # The history no longer fits the server's window: stop investigating, synthesize what was gathered.
            RUN.update(stop_reason="context_length", error=f"analyst: {str(exc)[:500]}")
            return {}
        return {"researcher_messages": [response]}

    return analyst_call


def make_tool_node(tools: list):
    tools_by_name = {candidate.name: candidate for candidate in tools}

    def tool_node(state: ResearcherState) -> dict:
        outputs = []
        for call in state["researcher_messages"][-1].tool_calls:
            selected = tools_by_name.get(call["name"])
            if selected is None:
                result = f"ERROR: unknown tool {call['name']!r}; available tools: {', '.join(sorted(tools_by_name))}"
            else:
                error = None
                try:
                    with EV.scope(role="analyst"):
                        result = selected.invoke(call["args"])
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    result = f"ERROR calling {call['name']}: {type(exc).__name__}: {str(exc)[:500]}"
                EV.tool_call(call["name"], call["args"], result=str(result), error=error,
                             tool_call_id=call["id"], role="analyst")
            outputs.append(ToolMessage(content=str(result), name=call["name"], tool_call_id=call["id"]))
        return {"researcher_messages": outputs}

    return tool_node


def valid_graph(text):
    try:
        graph = json.loads(strip_markdown_json(text))
    except (TypeError, ValueError):
        return False
    return isinstance(graph, dict) and all(k in graph for k in ("nodes", "edges", "root_causes"))


def compact_tool_evidence(messages):
    # Keep tool IDs paired with calls and retain both ends of large table results.
    compact = []
    for message in messages:
        content = message.content
        if isinstance(message, ToolMessage) and isinstance(content, str) and len(content) > 2048:
            message = message.model_copy(update={"content": content[:1024] + "\n[rows omitted]\n" + content[-1024:]})
        compact.append(message)
    return compact


def make_compress_research(compress_sp: str, compress_up: str):
    compress_model = _make_model(COMPRESS_MAX_TOKENS, purpose="synthesis")

    def compress_research(state: ResearcherState) -> dict:
        messages = ([SystemMessage(content=compress_sp)] + list(state.get("researcher_messages", []))
                    + [HumanMessage(content=compress_up)])
        with EV.scope(role="compress"):
            try:
                response = compress_model.invoke(messages)
            except Exception as exc:
                if not is_context_length(exc):
                    raise
                try:
                    messages = compact_tool_evidence(messages)
                    response = compress_model.invoke(messages)
                except Exception as retry_exc:
                    if not is_context_length(retry_exc):
                        raise
                    RUN.update(stop_reason="context_length", error=f"synthesis: {str(retry_exc)[:500]}")
                    return {"compressed_research": ""}
            draft = _content_to_text(response.content)
            if draft.strip() and not valid_graph(draft):
                response = compress_model.invoke(messages)
        raw_notes = [str(m.content) for m in filter_messages(state["researcher_messages"], include_types=["tool", "ai"])]
        return {"compressed_research": _content_to_text(response.content), "raw_notes": ["\n".join(raw_notes)],
                "researcher_messages": [response]}

    return compress_research


def route_after_analyst(state: ResearcherState) -> str:
    if RUN["stop_reason"]:
        return "compress_research"
    if state["researcher_messages"][-1].tool_calls:
        return "tool_node"
    return "compress_research"


def build_agent(combined_sp: str, compress_sp: str, compress_up: str):
    """Analyst -> tools -> synthesis."""
    tools = list(RCA_TOOLS)

    builder = StateGraph(ResearcherState)
    builder.add_node("analyst_call", make_analyst_call(combined_sp, tools))
    builder.add_node("tool_node", make_tool_node(tools))
    builder.add_node("compress_research", make_compress_research(compress_sp, compress_up))
    routes = {"tool_node": "tool_node", "compress_research": "compress_research"}
    builder.add_edge(START, "analyst_call")
    builder.add_conditional_edges("analyst_call", route_after_analyst, routes)
    builder.add_edge("tool_node", "analyst_call")
    builder.add_edge("compress_research", END)
    return builder.compile()


# ── LangChain -> OpenAI format ───────────────────────────────────────────────

def to_openai_message(message) -> dict | None:
    if isinstance(message, HumanMessage):
        return {"role": "user", "agent": "analyst", "content": str(message.content)}
    if isinstance(message, AIMessage):
        entry: dict = {"role": "assistant", "agent": "analyst", "content": _content_to_text(message.content)}
        if message.tool_calls:
            entry["tool_calls"] = [
                {"id": call["id"], "type": "function",
                 "function": {"name": call["name"], "arguments": json.dumps(call["args"], ensure_ascii=False)}}
                for call in message.tool_calls]
        reasoning = (message.additional_kwargs or {}).get("reasoning_content")
        if reasoning:
            entry["reasoning_content"] = str(reasoning)
        return entry
    if isinstance(message, ToolMessage):
        return {"role": "tool", "agent": "analyst",
                "content": str(message.content), "name": message.name, "tool_call_id": message.tool_call_id}
    return None


def convert_trajectory(messages: list) -> list[dict]:
    return [converted for message in messages if (converted := to_openai_message(message)) is not None]


def _setup_logging(verbose: bool, log_file: str | None) -> None:
    handlers = []
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if verbose:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        handlers.append(stream)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    logging.basicConfig(level=logging.INFO if handlers else logging.WARNING, handlers=handlers or None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log-file", default=None)
    args, _ = parser.parse_known_args()
    _setup_logging(args.verbose, args.log_file)

    payload = json.loads(sys.stdin.read())
    os.environ["RCA_CASE_DATA_DIR"] = payload["data_dir"]
    user_prompt = payload["user_prompt"] + "\n\n" + payload.get("rca_telemetry_prompt", rca_telemetry_prompt)
    combined_sp = payload["system_prompt"] + "\n\n---\n\n" + payload.get("rca_think_prompt", rca_think_prompt)

    agent = build_agent(combined_sp, payload["compress_system_prompt"], payload["compress_user_prompt"])

    all_messages: list = []
    compressed_research = ""
    step = 0
    initial_state = {"researcher_messages": [HumanMessage(content=user_prompt)]}
    from langgraph.errors import GraphRecursionError
    try:
        for event in agent.stream(initial_state, config={"recursion_limit": RECURSION_LIMIT}):
            for node, value in event.items():
                if not isinstance(value, dict):
                    continue
                new_messages = list(value.get("researcher_messages") or [])
                if new_messages:
                    all_messages.extend(new_messages)
                    step += 1
                    for message in new_messages:
                        calls = [c["name"] for c in (getattr(message, "tool_calls", None) or [])]
                        logger.info("[step %s] %s %s tool_calls=%s", step, node, type(message).__name__, calls)
                if "compressed_research" in value:
                    compressed_research = value["compressed_research"]
    except GraphRecursionError:
        # Use the collected evidence for one final synthesis after the investigation budget.
        RUN.update(stop_reason="recursion_limit_synthesis", error=None)
        synthesize = make_compress_research(payload["compress_system_prompt"], payload["compress_user_prompt"])
        final = synthesize({"researcher_messages": [*initial_state["researcher_messages"], *all_messages]})
        compressed_research = final["compressed_research"]
        all_messages.extend(final.get("researcher_messages", []))

    output = strip_markdown_json(compressed_research)
    graph_valid = valid_graph(output)
    failed = not graph_valid
    if not graph_valid and not RUN["stop_reason"]:
        RUN.update(stop_reason="invalid_graph", error="Synthesis did not return a valid graph JSON")
    EV.artifact("final_graph", {"output": output, "graph_valid": graph_valid})
    # Single line: the harness parses the last stdout line.
    print(json.dumps({
        "output": output,  # kept verbatim when invalid: the judge can recover a graph from the trajectory
        "graph_valid": graph_valid,
        "trajectory": convert_trajectory(all_messages),
        "usage": _tracker.get_usage(),  # the SDK hook sees Analyst and synthesis calls
        "run_status": "failed" if failed else "completed",
        "stop_reason": RUN["stop_reason"] or "finish",
        "run_error": RUN["error"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
