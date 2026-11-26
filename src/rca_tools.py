import json
import os
from datetime import datetime
from pathlib import Path
from typing import Union, List
from langchain_core.tools import tool

TOKEN_LIMIT = 5000
# Native time columns a table's uniform `ts` column is derived from, most preferred first (lowercase). The rule is
# Aiops_Context's (aiops_context/backends/sql.py), so both frameworks see the same `ts` and the incident's ISO
# fault_window can be compared with it directly: nobody does epoch arithmetic in their head.
TIME_COLUMNS = ("time", "timestamp", "@timestamp", "_time_", "starttime", "start_time", "starttimemillis")
# Epoch magnitude -> unit and its factor to microseconds (negative: divide). Smaller values are not epochs.
EPOCH_UNITS = ((1e11, "s", 1_000_000), (1e14, "ms", 1000), (1e17, "us", 1), (1e20, "ns", -1000))
MIN_EPOCH = 1e8
_TS_CACHE: dict = {}

def _import_duckdb():
    """Import duckdb with helpful error message."""
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise ImportError(
            "duckdb is required. Install it with: pip install duckdb"
        )


def _connect():
    """An in-memory connection that shows and compares timestamps in UTC, the zone of fault_window."""
    conn = _import_duckdb().connect(":memory:")
    conn.execute("SET TimeZone='UTC'")
    return conn

def _serialize_datetime(obj):
    """Convert datetime objects to ISO format strings for JSON serialization"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    else:
        return obj

def _estimate_token_count(text: str) -> int:
    """Estimate token count using character-based approximation.
    
    Approximate for Chinese/English mixed text.
    Average: 3 characters per token.
    """
    average_chars_per_token = 3
    return (len(text) + average_chars_per_token - 1) // average_chars_per_token

def _enforce_token_limit(payload: str, context: str) -> str:
    """Ensure payload stays within the token budget before returning"""
    token_estimate = _estimate_token_count(payload)
    if token_estimate <= TOKEN_LIMIT:
        return payload

    current_size = len(json.loads(payload)) if payload.startswith("[") else None
    suggested_limit = None
    if current_size:
        ratio = TOKEN_LIMIT / token_estimate
        suggested_limit = max(1, int(current_size * ratio * 0.8))  # 80% safety margin

    suggestion_parts = [
        "The query result is too large. Please adjust your query:",
        "  • Reduce the LIMIT value" + (f" (try LIMIT {suggested_limit})" if suggested_limit else ""),
        "  • Filter rows with WHERE clauses to reduce result size",
        "  • Select only necessary columns instead of SELECT *",
        "  • Use aggregation (COUNT, SUM, AVG) instead of retrieving raw rows",
    ]

    warning = {
        "error": "Result exceeds token budget",
        "context": context,
        "estimated_tokens": token_estimate,
        "token_limit": TOKEN_LIMIT,
        "rows_returned": current_size,
        "suggested_limit": suggested_limit,
        "suggestion": "\n".join(suggestion_parts),
    }
    return json.dumps(warning, ensure_ascii=False, indent=2)


def _sandbox_root() -> Path:
    """The case's telemetry directory; the runner sets it from the payload's data_dir."""
    value = os.environ.get("RCA_CASE_DATA_DIR")
    if not value:
        raise RuntimeError("RCA_CASE_DATA_DIR is not set")
    return Path(value)


def sandbox_tables() -> dict[str, list[str]]:
    """Table name -> parquet files of this case: <name>.parquet, or <name>/*.parquet when split into parts."""
    tables: dict[str, list[str]] = {}
    for entry in sorted(_sandbox_root().iterdir()):
        name = entry.name if entry.is_dir() else entry.stem
        if entry.is_dir():
            parts = sorted(str(p) for p in entry.glob("*.parquet"))
            if parts:
                tables[name] = parts
        elif entry.suffix == ".parquet":
            tables[name] = [str(entry)]
    return tables


def _read_source(files: list[str]) -> str:
    """SQL source of one table; parts are merged by column name."""
    quoted = ", ".join("'" + f.replace("'", "''") + "'" for f in files)
    return f"read_parquet([{quoted}], union_by_name=true)"


def _resolve_table(table: str) -> list[str]:
    """Resolve a model-visible table name inside the case directory."""
    tables = sandbox_tables()
    if str(table) not in tables:
        raise ValueError("Use a table name returned by list_tables_in_directory")
    return tables[str(table)]


def _safe_tool_error(exc: Exception) -> str:
    return str(exc).replace(str(_sandbox_root()) + os.sep, "").replace(".parquet", "")


def _sanitize_column_name(name: str) -> str:
    """Replace dots in column names with underscores to avoid DuckDB dot-notation ambiguity."""
    return name.replace(".", "_")


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _time_expression(conn, source: str, column: str, kind: str):
    """SQL for a TIMESTAMPTZ (UTC) from one native time column, and how it was read; None if it is no time."""
    q = _quote_identifier(column)
    if kind.startswith("TIMESTAMP"):
        return f"CAST({q} AS TIMESTAMPTZ)", "timestamp"
    sample = [r[0] for r in conn.execute(f"SELECT {q} FROM {source} WHERE {q} IS NOT NULL LIMIT 200").fetchall()]
    if not sample:
        return None
    numbers = []
    for value in sample:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            break
    if len(numbers) == len(sample):
        unit, factor = next(((u, f) for bound, u, f in EPOCH_UNITS if max(numbers) < bound), (None, None))
        if unit is None or min(numbers) < MIN_EPOCH:
            return None
        # Integers (also digit strings such as nanosecond VARCHARs) stay exact; DOUBLE keeps its fraction.
        integral = "INT" in kind or (kind == "VARCHAR" and all(str(v).isdigit() for v in sample))
        value = f"TRY_CAST({q} AS {'BIGINT' if integral else 'DOUBLE'})"
        scaled = f"{value} * {factor}" if factor > 0 else f"{value} / {-factor}"
        micros = scaled if integral and factor > 0 else f"CAST({scaled} AS BIGINT)"
        return f"make_timestamp({micros})::TIMESTAMPTZ", f"epoch {unit}"
    if kind == "VARCHAR":
        parsed = conn.execute(f"SELECT count(TRY_CAST(v AS TIMESTAMPTZ)) FROM (VALUES "
                              + ", ".join(f"({_sql_text(str(v))})" for v in sample) + ") t(v)").fetchone()[0]
        if parsed == len(sample):
            return f"TRY_CAST({q} AS TIMESTAMPTZ)", "ISO text"
    return None


def _ts_column(conn, files: list[str], source: str, columns: list[tuple[str, str]]):
    """(SQL, description) of the table's uniform ts column, or None: the first native time column, by
    TIME_COLUMNS, that reads as a time. A table that already has a ts column keeps it."""
    if any(_sanitize_column_name(c).lower() == "ts" for c, _ in columns):
        return None
    key = tuple(files)
    if key not in _TS_CACHE:
        native = {c.lower(): (c, k) for c, k in columns}
        found = None
        for candidate in TIME_COLUMNS:
            if candidate in native:
                column, kind = native[candidate]
                derived = _time_expression(conn, source, column, kind)
                if derived:
                    found = (derived[0], f"{_sanitize_column_name(column)} ({derived[1]})")
                    break
        _TS_CACHE[key] = found
    return _TS_CACHE[key]


def _table_select(conn, files: list[str], source: str):
    """SELECT list of a table's view: ts first when derivable, then every column (dots renamed to underscores);
    and the ts description or None."""
    columns = [(r[0], str(r[1])) for r in conn.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]
    if any("." in c for c, _ in columns):
        body = ", ".join(f"{_quote_identifier(c)} AS {_quote_identifier(_sanitize_column_name(c))}" for c, _ in columns)
    else:
        body = "*"
    ts = _ts_column(conn, files, source, columns)
    return (f"{ts[0]} AS ts, {body}" if ts else body), (ts[1] if ts else None)


def _table_names(tables: Union[str, List[str]]) -> List[str]:
    """Some models (e.g. Qwen) serialize list arguments as JSON strings."""
    if isinstance(tables, str):
        stripped = tables.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                return parsed if isinstance(parsed, list) else [stripped]
            except (json.JSONDecodeError, ValueError):
                return [stripped]
        return [stripped.strip('"')]
    return list(tables)


@tool
def list_tables_in_directory(directory: str = ".") -> str:
    """
    List telemetry tables in the current incident with metadata.

    Args:
        directory: Use "." to list the current incident's telemetry tables

    Returns:
        JSON string containing table names and metadata
    """
    if directory not in (".", "", "tables"):
        return json.dumps({"error": "Use directory='.' for the current case"})
    files_info = []
    for name, files in sandbox_tables().items():
        conn = _connect()
        try:
            source = _read_source(files)
            row_count = conn.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0]
            select, _ = _table_select(conn, files, source)
            column_count = len(conn.execute(f"SELECT {select} FROM {source} LIMIT 0").description)
            files_info.append({"table": name, "row_count": row_count, "column_count": column_count})
        except Exception as e:
            files_info.append({"table": name, "error": _safe_tool_error(e)})
        finally:
            conn.close()
    return _enforce_token_limit(json.dumps(files_info, ensure_ascii=False, indent=2), "list_tables_in_directory")


def _get_schema_one(table: str) -> dict:
    """Schema of one table; with a ts column, where it comes from and the UTC span it covers."""
    try:
        files = _resolve_table(table)
    except ValueError as exc:
        return {"error": str(exc)}
    conn = _connect()
    try:
        source = _read_source(files)
        select, ts_from = _table_select(conn, files, source)
        view = f"(SELECT {select} FROM {source})"
        schema = [{"name": name, "type": str(kind)}
                  for name, kind, *_ in conn.execute(f"DESCRIBE SELECT * FROM {view}").fetchall()]
        result = {"table": table, "row_count": conn.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0],
                  "columns": schema}
        if ts_from:
            lo, hi = conn.execute(f"SELECT min(ts), max(ts) FROM {view}").fetchone()
            result["ts_from"] = ts_from
            result["ts_range"] = [_serialize_datetime(lo), _serialize_datetime(hi)]
        return result
    except Exception as e:
        return {"error": f"Failed to extract schema: {_safe_tool_error(e)}"}
    finally:
        conn.close()


@tool
def get_schema(parquet_files: Union[str, List[str]]) -> str:
    """
    Get schema information of a table, or a list of tables.

    A table with a time column also has `ts` (TIMESTAMP WITH TIME ZONE, UTC) as its first column, derived from
    the native column named in `ts_from`; `ts_range` is the earliest and latest `ts`.

    Args:
        parquet_files: Table name or list of table names for batch lookup

    Returns:
        JSON string containing table metadata — single object if one table, list if multiple
    """
    names = _table_names(parquet_files)
    payload = _get_schema_one(names[0]) if len(names) == 1 else [_get_schema_one(n) for n in names]
    return _enforce_token_limit(json.dumps(payload, ensure_ascii=False, indent=2), "get_schema")


@tool
def query_parquet_files(parquet_files: Union[str, List[str]], query: str, limit: int = 10) -> str:
    """
    Query telemetry tables using SQL syntax for data analysis and exploration.

    Filter time with the `ts` column (UTC) and ISO 8601 strings, e.g. WHERE ts >= '2021-03-04T06:30:00Z';
    timestamps in results are UTC.

    Args:
        parquet_files: Table name(s) returned by list_tables_in_directory
        query: SQL query to execute; each table is available under its own name
        limit: Maximum number of records to return

    Returns:
        JSON string of query results
    """
    names = _table_names(parquet_files)
    conn = _connect()
    try:
        for name in names:
            files = _resolve_table(name)
            source = _read_source(files)
            select, _ = _table_select(conn, files, source)
            conn.execute(f"CREATE VIEW {_quote_identifier(name)} AS SELECT {select} FROM {source}")
        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchmany(max(1, int(limit)))]
        return _enforce_token_limit(json.dumps(_serialize_datetime(rows), ensure_ascii=False, indent=2, default=str),
                                    "query_parquet_files")
    except Exception as e:
        error_msg = _safe_tool_error(e)
        # Provide contextual error messages
        if "syntax error" in error_msg.lower() or "parser error" in error_msg.lower():
            return json.dumps({
                "error": f"SQL syntax error in query: {error_msg}",
                "query": query,
                "available_tables": names
            })
        elif "catalog" in error_msg.lower() or "table" in error_msg.lower():
            return json.dumps({
                "error": f"Table reference error: {error_msg}",
                "query": query,
                "available_tables": names
            })
        else:
            return json.dumps({
                "error": f"Query execution failed: {error_msg}",
                "query": query,
                "available_tables": names
            })
    finally:
        conn.close()
