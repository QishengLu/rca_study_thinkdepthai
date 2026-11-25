"""Offline smoke test of agent_runner.py with a fake model: no API key, no network.

Usage: .venv/bin/python tests/smoke_fake_llm.py [data_dir ...]

Without arguments the case is a small parquet directory written to a temporary folder. data_dir is a
directory of telemetry tables, <table>.parquet or <table>/*.parquet, such as a prepared sandbox's tables/.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "sk-offline-smoke")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.invalid/v1")
CASE_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

import agent_runner as runner  # noqa: E402
from deep_research.rca_tools import sandbox_tables  # noqa: E402


class FakeModel:
    def __init__(self, script):
        self.script = script
        self.seen = []
        self.bound = []

    def bind_tools(self, tools):
        self.bound.append([candidate.name for candidate in tools])
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class ContextLengthError(Exception):
    """What the OpenAI SDK raises for vLLM's refusal of an over-long request (HTTP 400)."""
    status_code = 400

    def __init__(self):
        super().__init__("Error code: 400 - This model's maximum context length is 32768 tokens. However, you "
                         "requested 4096 output tokens and your prompt contains at least 28700 input tokens.")


def call(name, call_id, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def write_case(root: Path) -> None:
    """One table per layout: a single file, and a directory of parts."""
    (root / "logs").mkdir()
    conn = duckdb.connect(":memory:")
    conn.execute(f"COPY (SELECT 1 AS time, 'payment' AS \"service.name\") TO '{root / 'abnormal_traces.parquet'}'")
    for part in (0, 1):
        conn.execute(f"COPY (SELECT {part} AS time, 'boom' AS message) TO '{root / 'logs' / f'{part}.parquet'}'")
    conn.close()
    (root / "ignored.txt").write_text("not a table")


def run_case(data_dir: str) -> None:
    os.environ["RCA_CASE_DATA_DIR"] = data_dir
    tables = sandbox_tables()
    assert tables, f"no telemetry tables in {data_dir}"
    first = next(iter(tables))
    graph = {"nodes": [{"component": "payment", "state": "UNAVAILABLE"}], "edges": [],
             "root_causes": [{"component": "payment", "state": "UNAVAILABLE"}], "component_to_service": {}}
    model = FakeModel([
        call("list_tables_in_directory", "c1", directory="."),
        call("get_schema", "c2", parquet_files=list(tables)),
        call("query_parquet_files", "c3", parquet_files=[first], query=f'SELECT COUNT(*) AS n FROM "{first}"'),
        call("think_tool", "c4", reflection="payment is the origin"),
        call("no_such_tool", "c5"),
        AIMessage(content="Root cause: payment."),
        AIMessage(content="```json\n" + json.dumps(graph) + "\n```"),
    ])
    runner._make_model = lambda max_tokens=32768, purpose=None: model
    payload = {"question": "q", "system_prompt": "SP", "user_prompt": "UP", "compress_system_prompt": "CSP",
               "compress_user_prompt": "CUP", "data_dir": data_dir}
    sys.stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        runner.main()
    result = json.loads(stdout.getvalue().strip().splitlines()[-1])

    assert set(result) == {"output", "graph_valid", "trajectory", "usage", "run_status", "stop_reason",
                           "run_error"}, set(result)
    assert result["graph_valid"] is True and json.loads(result["output"]) == graph
    assert result["run_status"] == "completed" and result["stop_reason"] == "finish"
    assert not model.script, "the fake model was not called to the end of its script"
    assert model.bound == [["think_tool", "list_tables_in_directory", "get_schema", "query_parquet_files"]]

    # The Analyst sees the combined system prompt, then the user prompt; synthesis comes last.
    first_call, last_call = model.seen[0], model.seen[-1]
    assert isinstance(first_call[0], SystemMessage) and first_call[0].content == "SP\n\n---\n\n" + runner.rca_think_prompt
    assert isinstance(first_call[1], HumanMessage) and first_call[1].content.startswith("UP\n\n## Telemetry Tables")
    assert last_call[0].content == "CSP" and last_call[-1].content == "CUP"

    # The trajectory holds what the graph nodes produced; the initial user message is not among them.
    trajectory = result["trajectory"]
    assert [m["role"] for m in trajectory] == ["assistant", "tool"] * 5 + ["assistant", "assistant"]
    assert {m["agent"] for m in trajectory} == {"analyst"}
    outputs = {m["name"]: m["content"] for m in trajectory if m["role"] == "tool"}
    assert [t["table"] for t in json.loads(outputs["list_tables_in_directory"])] == list(tables)
    assert '"columns"' in outputs["get_schema"] or "exceeds token budget" in outputs["get_schema"]
    assert json.loads(outputs["query_parquet_files"])[0]["n"] >= 0, outputs["query_parquet_files"]
    assert outputs["no_such_tool"].startswith("ERROR: unknown tool")
    assert data_dir not in json.dumps(trajectory), "the case path reached the model"
    print(f"case OK: {len(tables)} tables, first {first!r}: {data_dir}")


def run_overflow(data_dir: str, synthesis_fits: bool) -> None:
    """The server refuses an Analyst request for its context length: synthesis on what was gathered, or a failed
    run with no answer when the synthesis is refused too. Never an infrastructure failure (exit 0)."""
    os.environ["RCA_CASE_DATA_DIR"] = data_dir
    runner.RUN.update(stop_reason=None, error=None)
    graph = {"nodes": [], "edges": [], "root_causes": [{"component": "payment", "state": "UNAVAILABLE"}]}
    model = FakeModel([call("list_tables_in_directory", "c1", directory="."), ContextLengthError(),
                       AIMessage(content=json.dumps(graph)) if synthesis_fits else ContextLengthError()])
    if not synthesis_fits:
        model.script.append(ContextLengthError())  # The compacted synthesis request also exceeds the window.
    runner._make_model = lambda max_tokens=32768, purpose=None: model
    sys.stdin = io.StringIO(json.dumps({"question": "q", "system_prompt": "SP", "user_prompt": "UP",
                                        "compress_system_prompt": "CSP", "compress_user_prompt": "CUP",
                                        "data_dir": data_dir}))
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        runner.main()
    result = json.loads(stdout.getvalue().strip().splitlines()[-1])
    assert result["stop_reason"] == "context_length" and not model.script
    if synthesis_fits:
        assert result["run_status"] == "completed" and result["graph_valid"] is True
    else:
        assert result["run_status"] == "failed" and result["output"] == "" and "synthesis" in result["run_error"]
    assert [m["role"] for m in result["trajectory"]] == ["assistant", "tool"] + (["assistant"] if synthesis_fits else [])
    runner.RUN.update(stop_reason=None, error=None)


# The graph: start -> analyst -> (tools -> analyst)* -> synthesis -> end.
runner._make_model = lambda max_tokens=32768, purpose=None: FakeModel([])
assert set(runner.build_agent("SP", "CSP", "CUP").get_graph().nodes) == {
    "__start__", "analyst_call", "tool_node", "compress_research", "__end__"}
route = runner.route_after_analyst
assert route({"researcher_messages": [AIMessage(content="", tool_calls=[
    {"name": "get_schema", "args": {}, "id": "x", "type": "tool_call"}])]}) == "tool_node"
assert route({"researcher_messages": [AIMessage(content="done")]}) == "compress_research"
assert runner.to_openai_message(ToolMessage(content="r", name="get_schema", tool_call_id="x"))["agent"] == "analyst"

if CASE_ARGS:
    for case in CASE_ARGS:
        run_case(case)
else:
    with tempfile.TemporaryDirectory() as tmp:
        write_case(Path(tmp))
        os.environ["RCA_CASE_DATA_DIR"] = tmp
        assert list(sandbox_tables()) == ["abnormal_traces", "logs"]
        run_case(tmp)
        run_overflow(tmp, synthesis_fits=True)
        run_overflow(tmp, synthesis_fits=False)

print("offline smoke OK: Analyst tools, graph, routing, stdout contract, trajectory and context-length stops")
