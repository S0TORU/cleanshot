#!/usr/bin/env python3
"""
CleanShot: Dataset cleaning CLI powered by DuckDB.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import sys
from decimal import Decimal
from pathlib import Path

import duckdb


def die(message: str, hint: str | None = None, code: int = 2) -> None:
    if hint:
        print(f"Error: {message}\nHint: {hint}", file=sys.stderr)
    else:
        print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(code)


def format_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}B"


def summarize_list(items, limit: int = 8) -> str:
    if not items:
        return "none"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", ... (+{len(items) - limit} more)"


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def is_safe_identifier(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


def ensure_columns_exist(columns: list[str], available: set[str], label: str) -> None:
    missing = [col for col in columns if col not in available]
    if missing:
        die(
            f"{label} columns not found: {', '.join(missing)}",
            "Check column names or run with --profile to inspect columns.",
        )


def write_json(path: str, payload: dict) -> None:
    def sanitize(value):
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (dt.date, dt.datetime)):
            return value.isoformat()
        return value

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=sanitize)


def parse_outlier_pct(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        die("Outlier percentiles must be two values", "Use --outlier-pct 0.01,0.99")
    try:
        lower = float(parts[0])
        upper = float(parts[1])
    except ValueError as exc:
        raise SystemExit("Outlier percentiles must be numeric") from exc
    if not (0 <= lower < upper <= 1):
        die("Outlier percentiles must be between 0 and 1")
    return lower, upper


def parse_split_ratios(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        die("Split ratios must be three values", "Use --split 0.8,0.1,0.1")
    try:
        ratios = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise SystemExit("Split ratios must be numeric") from exc
    return ratios


def safe_column_alias(prefix: str, column: str, used: set[str]) -> str:
    raw = f"{prefix}_{column}" if prefix else column
    alias = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_").lower()
    if not alias:
        alias = "col"
    if not is_safe_identifier(alias):
        alias = "col_" + alias
    candidate = alias
    counter = 1
    while candidate in used:
        candidate = f"{alias}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def table_to_string(headers, rows) -> str:
    if not rows:
        return "(no rows)"
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    lines = [fmt.format(*headers), "-+-".join("-" * w for w in widths)]
    for row in rows:
        lines.append(fmt.format(*[str(v) for v in row]))
    return "\n".join(lines)


def sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, Decimal)):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NULL"
        return repr(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return "'" + value.isoformat() + "'"
    return "'" + str(value).replace("'", "''") + "'"


def is_numeric_type(type_name: str) -> bool:
    upper = type_name.upper()
    return any(
        token in upper
        for token in (
            "INT",
            "DOUBLE",
            "DECIMAL",
            "REAL",
            "FLOAT",
            "HUGEINT",
        )
    )


def is_date_type(type_name: str) -> bool:
    upper = type_name.upper()
    return "DATE" in upper or "TIME" in upper or "TIMESTAMP" in upper


def is_string_type(type_name: str) -> bool:
    upper = type_name.upper()
    return any(token in upper for token in ("CHAR", "TEXT", "VARCHAR", "STRING"))


def describe_table(con: duckdb.DuckDBPyConnection, table: str):
    rows = con.execute(f"DESCRIBE {table}").fetchall()
    return [{"name": row[0], "type": row[1]} for row in rows]


def print_table(headers, rows):
    print(table_to_string(headers, rows))


def prompt_text(prompt: str, default: str | None = None) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{hint}]: ").strip().lower()
        if not value:
            return default
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False


def prompt_int(prompt: str, default: int) -> int:
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        if value.isdigit():
            return int(value)
        print("Enter a valid integer.")


def prompt_menu(title: str, options: list[str], default_index: int = 0) -> int:
    print(title)
    for idx, opt in enumerate(options, start=1):
        print(f"  {idx}) {opt}")
    while True:
        value = input(f"Choose [1-{len(options)}] (default {default_index + 1}): ").strip()
        if not value:
            return default_index
        if value.isdigit():
            choice = int(value) - 1
            if 0 <= choice < len(options):
                return choice
        print("Enter a valid choice.")


def detect_input(input_path: str, override_format: str | None):
    path = Path(input_path)
    glob_chars = any(ch in input_path for ch in "*?[]")
    detected_format = None
    load_target = input_path

    if override_format:
        detected_format = override_format
        if path.is_dir():
            load_target = str(path / f"*.{override_format}")
        return detected_format, load_target

    if glob_chars:
        matches = [Path(p) for p in glob.glob(input_path, recursive=True)]
        if not matches:
            die(
                f"No files match: {input_path}",
                "Check the path or wrap it in quotes if it has spaces.",
            )
        exts = {p.suffix.lower() for p in matches if p.is_file() and p.suffix}
        if len(exts) != 1:
            die(
                f"Mixed extensions in glob: {sorted(exts)}",
                "Use a single format or pass --format csv|parquet|json.",
            )
        detected_format = detect_format_from_ext(exts.pop())
        return detected_format, load_target

    if path.is_dir():
        files = [p for p in path.iterdir() if p.is_file()]
        if not files:
            die(f"No files in directory: {input_path}")
        exts = {p.suffix.lower() for p in files if p.suffix}
        if not exts:
            die(f"No files with extensions in directory: {input_path}")
        if len(exts) != 1:
            die(
                f"Mixed extensions in directory: {sorted(exts)}",
                "Use a single format or pass --format csv|parquet|json.",
            )
        ext = exts.pop()
        detected_format = detect_format_from_ext(ext)
        load_target = str(path / f"*{ext}")
        return detected_format, load_target

    if path.is_file():
        detected_format = detect_format_from_ext(path.suffix.lower())
        return detected_format, load_target

    die(
        f"Input not found: {input_path}",
        "Provide a file, directory, or glob (e.g., data/*.csv) or use --interactive.",
    )


def detect_format_from_ext(ext: str) -> str:
    if ext == ".csv":
        return "csv"
    if ext in {".parquet", ".pq"}:
        return "parquet"
    if ext in {".json", ".jsonl", ".ndjson"}:
        return "json"
    die(
        f"Unsupported extension: {ext}",
        "Supported: .csv, .parquet, .pq, .json, .jsonl, .ndjson.",
    )


def build_loader(input_format: str, load_target: str) -> str:
    target_literal = sql_string_literal(load_target)
    if input_format == "csv":
        return f"read_csv({target_literal}, auto_detect=true, sample_size=-1, parallel=true)"
    if input_format == "parquet":
        return f"read_parquet({target_literal})"
    if input_format == "json":
        return f"read_json_auto({target_literal})"
    die(f"Unsupported format: {input_format}", "Use --format csv|parquet|json.")


def resolve_input_size(load_target: str, fallback: str | None = None) -> tuple[int, int]:
    candidates = []
    if any(ch in load_target for ch in "*?[]"):
        candidates = [Path(p) for p in glob.glob(load_target, recursive=True)]
    else:
        candidates = [Path(load_target)]
    if not candidates and fallback:
        candidates = [Path(fallback)]
    total = 0
    count = 0
    for path in candidates:
        if path.exists() and path.is_file():
            total += path.stat().st_size
            count += 1
    return total, count


def collect_column_stats(con: duckdb.DuckDBPyConnection, table: str, cols: list[str]):
    stat_exprs = []
    for idx, col in enumerate(cols):
        qc = sql_ident(col)
        stat_exprs.append(
            f"SUM(CASE WHEN {qc} IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS null_pct_{idx}"
        )
        stat_exprs.append(f"COUNT(DISTINCT {qc}) AS unique_{idx}")
    stats = con.execute(f"SELECT {', '.join(stat_exprs)} FROM {table}").fetchone()
    results = []
    for idx, col in enumerate(cols):
        results.append(
            {
                "name": col,
                "null_pct": stats[idx * 2],
                "unique_count": stats[idx * 2 + 1],
            }
        )
    return results


def collect_numeric_bounds(con: duckdb.DuckDBPyConnection, table: str, cols: list[str]):
    if not cols:
        return {}
    exprs = []
    for idx, col in enumerate(cols):
        qc = sql_ident(col)
        exprs.append(f"min({qc}) AS min_{idx}")
        exprs.append(f"max({qc}) AS max_{idx}")
        exprs.append(f"avg({qc}) AS mean_{idx}")
        exprs.append(f"stddev_pop({qc}) AS std_{idx}")
    row = con.execute(f"SELECT {', '.join(exprs)} FROM {table}").fetchone()
    bounds = {}
    for idx, col in enumerate(cols):
        bounds[col] = {
            "min": row[idx * 4],
            "max": row[idx * 4 + 1],
            "mean": row[idx * 4 + 2],
            "std": row[idx * 4 + 3],
        }
    return bounds


def resolve_percentile_bounds(
    con: duckdb.DuckDBPyConnection,
    table: str,
    cols: list[str],
    lower: float,
    upper: float,
):
    if not cols:
        return {}
    exprs = []
    for idx, col in enumerate(cols):
        qc = sql_ident(col)
        exprs.append(f"quantile_cont({qc}, {lower}) AS pmin_{idx}")
        exprs.append(f"quantile_cont({qc}, {upper}) AS pmax_{idx}")
    row = con.execute(f"SELECT {', '.join(exprs)} FROM {table}").fetchone()
    bounds = {}
    for idx, col in enumerate(cols):
        bounds[col] = {"min": row[idx * 2], "max": row[idx * 2 + 1]}
    return bounds


def resolve_outlier_bounds(
    con: duckdb.DuckDBPyConnection,
    table: str,
    cols: list[str],
    mode: str,
    lower_pct: float,
    upper_pct: float,
    zscore: float,
):
    if not cols:
        return {}
    if mode == "percentile":
        return resolve_percentile_bounds(con, table, cols, lower_pct, upper_pct)
    stats = collect_numeric_bounds(con, table, cols)
    bounds = {}
    for col, entry in stats.items():
        mean = entry["mean"]
        std = entry["std"]
        if mean is None or std in (None, 0):
            bounds[col] = {"min": None, "max": None}
            continue
        bounds[col] = {"min": mean - zscore * std, "max": mean + zscore * std}
    return bounds


def apply_outlier_clipping(
    con: duckdb.DuckDBPyConnection,
    view: str,
    numeric_cols: list[str],
    bounds: dict,
):
    if not numeric_cols:
        return view, 0
    exprs = []
    clipped = 0
    for col in numeric_cols:
        qc = sql_ident(col)
        limit = bounds.get(col, {})
        min_val = limit.get("min")
        max_val = limit.get("max")
        if min_val is None or max_val is None:
            exprs.append(qc)
            continue
        clipped += 1
        exprs.append(
            f"CASE WHEN {qc} < {sql_literal(min_val)} THEN {sql_literal(min_val)} "
            f"WHEN {qc} > {sql_literal(max_val)} THEN {sql_literal(max_val)} "
            f"ELSE {qc} END AS {qc}"
        )
    if not clipped:
        return view, 0
    con.execute(f"CREATE VIEW stage_outliers AS SELECT {', '.join(exprs)} FROM {view};")
    return "stage_outliers", clipped


def add_standardized_columns(
    con: duckdb.DuckDBPyConnection,
    view: str,
    numeric_cols: list[str],
    target_cols: set[str],
    prefix: str,
):
    if not numeric_cols:
        return view, 0, []
    stats = collect_numeric_bounds(con, view, numeric_cols)
    base_cols = [c for c in describe_table(con, view)]
    exprs = [sql_ident(c["name"]) for c in base_cols]
    added = 0
    added_cols = []
    used = {c["name"].lower() for c in base_cols}
    prefix_clean = prefix.rstrip("_")
    for col in numeric_cols:
        if col in target_cols:
            continue
        entry = stats.get(col, {})
        mean = entry.get("mean")
        std = entry.get("std")
        if mean is None or std in (None, 0):
            continue
        alias = safe_column_alias(prefix_clean, col, used)
        added += 1
        added_cols.append(alias)
        exprs.append(
            f"({sql_ident(col)} - {sql_literal(mean)}) / NULLIF({sql_literal(std)}, 0) "
            f"AS {sql_ident(alias)}"
        )
    con.execute(f"CREATE VIEW stage_scale AS SELECT {', '.join(exprs)} FROM {view};")
    return "stage_scale", added, added_cols


def add_min_max_columns(
    con: duckdb.DuckDBPyConnection,
    view: str,
    numeric_cols: list[str],
    target_cols: set[str],
    prefix: str,
):
    if not numeric_cols:
        return view, 0, []
    stats = collect_numeric_bounds(con, view, numeric_cols)
    base_cols = [c for c in describe_table(con, view)]
    exprs = [sql_ident(c["name"]) for c in base_cols]
    added = 0
    added_cols = []
    used = {c["name"].lower() for c in base_cols}
    prefix_clean = prefix.rstrip("_")
    for col in numeric_cols:
        if col in target_cols:
            continue
        entry = stats.get(col, {})
        min_val = entry.get("min")
        max_val = entry.get("max")
        if min_val is None or max_val is None or min_val == max_val:
            continue
        alias = safe_column_alias(prefix_clean, col, used)
        added += 1
        added_cols.append(alias)
        exprs.append(
            f"({sql_ident(col)} - {sql_literal(min_val)}) / NULLIF({sql_literal(max_val - min_val)}, 0) "
            f"AS {sql_ident(alias)}"
        )
    con.execute(f"CREATE VIEW stage_scale AS SELECT {', '.join(exprs)} FROM {view};")
    return "stage_scale", added, added_cols


def resolve_split_counts(row_count: int, train_ratio: float, val_ratio: float, test_ratio: float):
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
        die("Split ratios must sum to 1.0")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        die("Split ratios must be > 0")
    train_count = int(row_count * train_ratio)
    val_count = int(row_count * val_ratio)
    test_count = row_count - train_count - val_count
    return train_count, val_count, test_count


def build_split_queries(view: str, seed: int, train_count: int, val_count: int):
    base = "SELECT *, row_number() OVER () AS cs_rownum FROM " + view
    shuffled = f"SELECT * FROM ({base}) ORDER BY hash(cs_rownum + {seed})"
    train_query = f"SELECT * EXCLUDE (cs_rownum) FROM ({shuffled}) LIMIT {train_count}"
    val_query = (
        f"SELECT * EXCLUDE (cs_rownum) FROM ({shuffled}) "
        f"LIMIT {val_count} OFFSET {train_count}"
    )
    test_query = (
        f"SELECT * EXCLUDE (cs_rownum) FROM ({shuffled}) "
        f"OFFSET {train_count + val_count}"
    )
    return train_query, val_query, test_query


def build_stratified_split_queries(
    view: str,
    target: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    base = f"SELECT *, row_number() OVER () AS cs_rownum FROM {view}"
    shuffled = (
        "SELECT *, "
        f"row_number() OVER (PARTITION BY {sql_ident(target)} "
        f"ORDER BY hash({sql_ident(target)} || '_' || cs_rownum::VARCHAR || '{seed}')) AS cs_rank, "
        f"count(*) OVER (PARTITION BY {sql_ident(target)}) AS cs_total "
        f"FROM ({base})"
    )
    train_cutoff = train_ratio
    val_cutoff = train_ratio + val_ratio
    train_query = (
        f"SELECT * EXCLUDE (cs_rownum, cs_rank, cs_total) FROM ({shuffled}) "
        f"WHERE cs_rank <= cs_total * {train_cutoff}"
    )
    val_query = (
        f"SELECT * EXCLUDE (cs_rownum, cs_rank, cs_total) FROM ({shuffled}) "
        f"WHERE cs_rank > cs_total * {train_cutoff} AND cs_rank <= cs_total * {val_cutoff}"
    )
    test_query = (
        f"SELECT * EXCLUDE (cs_rownum, cs_rank, cs_total) FROM ({shuffled}) "
        f"WHERE cs_rank > cs_total * {val_cutoff}"
    )
    return train_query, val_query, test_query


def expand_one_hot_columns(
    con: duckdb.DuckDBPyConnection,
    view: str,
    categorical_cols: list[str],
    target_cols: set[str],
    prefix: str,
    max_categories: int,
):
    if not categorical_cols:
        return view, 0, {}, []
    desc = describe_table(con, view)
    base_cols = [c["name"] for c in desc]
    exprs = [sql_ident(c) for c in base_cols]
    mapping = {}
    added_cols = []
    added = 0
    used = {name.lower() for name in base_cols}
    prefix_clean = prefix.rstrip("_")
    for col in categorical_cols:
        if col in target_cols:
            continue
        qc = sql_ident(col)
        values = [
            row[0]
            for row in con.execute(
                f"SELECT {qc} FROM {view} WHERE {qc} IS NOT NULL "
                f"GROUP BY {qc} ORDER BY COUNT(*) DESC LIMIT {max_categories}"
            ).fetchall()
            if row[0] is not None
        ]
        mapping[col] = []
        for value in values:
            safe_value = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
            if not safe_value:
                continue
            column_name = safe_column_alias(prefix_clean, f"{col}__{safe_value}", used)
            exprs.append(
                f"CASE WHEN {qc} = {sql_literal(value)} THEN 1 ELSE 0 END AS {sql_ident(column_name)}"
            )
            mapping[col].append({"value": str(value), "column": column_name})
            added_cols.append(column_name)
            added += 1
    con.execute(f"CREATE VIEW stage_ohe AS SELECT {', '.join(exprs)} FROM {view};")
    return "stage_ohe", added, mapping, added_cols


def add_frequency_encoded_columns(
    con: duckdb.DuckDBPyConnection,
    view: str,
    categorical_cols: list[str],
    target_cols: set[str],
    prefix: str,
):
    if not categorical_cols:
        return view, 0, {}, []
    desc = describe_table(con, view)
    base_cols = [c["name"] for c in desc]
    exprs = [sql_ident(c) for c in base_cols]
    mapping = {}
    added_cols = []
    added = 0
    used = {name.lower() for name in base_cols}
    prefix_clean = prefix.rstrip("_")
    for col in categorical_cols:
        if col in target_cols:
            continue
        qc = sql_ident(col)
        alias = safe_column_alias(prefix_clean, col, used)
        freq_rows = con.execute(
            f"SELECT {qc}, COUNT(*)::DOUBLE / (SELECT COUNT(*) FROM {view}) AS freq "
            f"FROM {view} WHERE {qc} IS NOT NULL GROUP BY {qc}"
        ).fetchall()
        mapping[col] = {str(row[0]): row[1] for row in freq_rows}
        exprs.append(
            f"COALESCE((SELECT freq FROM ("
            f"SELECT {qc} AS value, COUNT(*)::DOUBLE / (SELECT COUNT(*) FROM {view}) AS freq "
            f"FROM {view} WHERE {qc} IS NOT NULL GROUP BY {qc}) "
            f"WHERE value = {qc} LIMIT 1), 0.0) AS {sql_ident(alias)}"
        )
        added_cols.append(alias)
        added += 1
    con.execute(f"CREATE VIEW stage_freq AS SELECT {', '.join(exprs)} FROM {view};")
    return "stage_freq", added, mapping, added_cols


def build_schema_export(
    con: duckdb.DuckDBPyConnection,
    view: str,
    input_path: str,
    output_path: str | None,
    report: dict,
):
    desc = describe_table(con, view)
    row_count = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
    columns = []
    for col in desc:
        name = col["name"]
        col_type = col["type"]
        nulls = con.execute(
            f"SELECT COUNT(*) FROM {view} WHERE {sql_ident(name)} IS NULL"
        ).fetchone()[0]
        uniques = con.execute(
            f"SELECT COUNT(DISTINCT {sql_ident(name)}) FROM {view}"
        ).fetchone()[0]
        columns.append(
            {
                "name": name,
                "type": col_type,
                "nulls": nulls,
                "unique": uniques,
            }
        )
    payload = {
        "input": input_path,
        "output": output_path,
        "rows": row_count,
        "columns": columns,
        "report": report,
    }
    return payload


def normalize_path_target(path: str | None) -> str | None:
    if path is None:
        return None
    if path.endswith(".json"):
        return path
    return f"{path}.json"


def normalize_report_target(path: str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.suffix:
        return path
    return f"{path}.md"


def load_fusion_sources(sources: list[str]):
    fusion_sources = []
    used_names = set()
    for source in sources:
        if "=" in source:
            name, path = source.split("=", 1)
        else:
            name = Path(source).stem
            path = source
        safe_name = safe_column_alias("", name, used_names)
        fusion_sources.append({"name": safe_name, "path": path})
    if not fusion_sources:
        die("Fusion requires at least one source", "Use --fusion-source name=path")
    return fusion_sources


def parse_duration_to_seconds(value: str) -> float:
    match = re.match(r"^([0-9]*\.?[0-9]+)\s*(ms|s|m)?$", value.strip())
    if not match:
        die("Invalid time tolerance", "Use formats like 500ms, 2s, or 0.25s")
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    if unit == "ms":
        return amount / 1000
    if unit == "m":
        return amount * 60
    return amount


def build_fusion_query(
    con: duckdb.DuckDBPyConnection,
    sources: list[dict],
    id_col: str,
    time_col: str,
    tolerance_seconds: float,
):
    base_source = sources[0]
    base_name = base_source["name"]
    base_path = base_source["path"]
    con.execute(
        f"CREATE VIEW {sql_ident(base_name)} AS SELECT * FROM {build_loader('csv', base_path)};"
    )
    for source in sources[1:]:
        name = source["name"]
        path = source["path"]
        con.execute(
            f"CREATE VIEW {sql_ident(name)} AS SELECT * FROM {build_loader('csv', path)};"
        )

    select_cols = [f"{sql_ident(base_name)}.*"]
    join_clause = f"FROM {sql_ident(base_name)}"
    for source in sources[1:]:
        name = source["name"]
        select_cols.append(
            f"{sql_ident(name)}.* EXCLUDE ({sql_ident(id_col)}, {sql_ident(time_col)})"
        )
        join_clause += (
            f" LEFT JOIN {sql_ident(name)} ON {sql_ident(base_name)}.{sql_ident(id_col)} "
            f"= {sql_ident(name)}.{sql_ident(id_col)} "
            f"AND abs(epoch(try_cast({sql_ident(base_name)}.{sql_ident(time_col)} AS TIMESTAMP)) "
            f"- epoch(try_cast({sql_ident(name)}.{sql_ident(time_col)} AS TIMESTAMP))) <= {tolerance_seconds}"
        )
    query = f"SELECT {', '.join(select_cols)} {join_clause}"
    con.execute("CREATE VIEW fused_data AS " + query + ";")
    return "fused_data"


def default_output_for_input(input_path: str) -> str:
    if any(ch in input_path for ch in "*?[]"):
        return "cleaned.parquet"
    path = Path(input_path)
    if path.is_file():
        return f"{path.stem}_clean.parquet"
    return "cleaned.parquet"


def run_interactive(args: argparse.Namespace) -> argparse.Namespace:
    print("CleanShot Interactive")
    while not args.input:
        args.input = prompt_text("Input path or glob")
        try:
            detect_input(args.input, args.format)
        except SystemExit:
            args.input = None

    if args.output is None:
        default_output = default_output_for_input(args.input)
        args.output = prompt_text("Output parquet path", default_output)

    mode = prompt_menu(
        "Pick a mode",
        ["Quick Clean (fast defaults)", "ML-Ready (impute missing)", "Inspect First (profile + plan)"],
        default_index=1,
    )
    if mode == 0:
        args.preset = "basic"
        args.inspect_first = False
    elif mode == 1:
        args.preset = "ml_ready"
        args.inspect_first = False
    else:
        args.preset = None
        args.inspect_first = True

    args.threads = prompt_int("DuckDB threads", max(1, args.threads))
    args.dry_run = prompt_yes_no("Dry run (no output write)", default=args.dry_run)
    if not args.dry_run:
        compression_opts = ["zstd", "snappy", "gzip", "uncompressed"]
        default_idx = compression_opts.index(args.compression)
        comp_choice = prompt_menu("Parquet compression", compression_opts, default_index=default_idx)
        args.compression = compression_opts[comp_choice]
    args.verbose = prompt_yes_no("Verbose progress output", default=args.verbose)

    if prompt_yes_no("Enable ML feature pack?", default=False):
        if prompt_yes_no("Clip outliers?", default=False):
            outlier_mode = prompt_menu(
                "Outlier mode",
                ["percentile (p1/p99)", "z-score (3.0)", "skip"],
                default_index=0,
            )
            if outlier_mode == 0:
                args.outlier_mode = "percentile"
                args.outlier_pct = "0.01,0.99"
            elif outlier_mode == 1:
                args.outlier_mode = "zscore"
                args.outlier_zscore = 3.0
            else:
                args.outlier_mode = "none"

        if prompt_yes_no("Add scaled numeric columns?", default=False):
            scale_choice = prompt_menu(
                "Scaling",
                ["standardize", "minmax", "skip"],
                default_index=0,
            )
            if scale_choice == 0:
                args.scale = "standardize"
            elif scale_choice == 1:
                args.scale = "minmax"
            else:
                args.scale = "none"

        if prompt_yes_no("Encode categoricals?", default=False):
            encode_choice = prompt_menu(
                "Encoding",
                ["onehot", "frequency", "skip"],
                default_index=0,
            )
            if encode_choice == 0:
                args.encode = "onehot"
            elif encode_choice == 1:
                args.encode = "frequency"
            else:
                args.encode = "none"

        if prompt_yes_no("Export schema/stats JSON?", default=False):
            args.schema_out = prompt_text("Schema output path", "schema.json")

        if prompt_yes_no("Split train/val/test outputs?", default=False):
            args.split = "0.8,0.1,0.1"
            args.split_seed = prompt_int("Split seed", args.split_seed)
            if prompt_yes_no("Stratify split on target column?", default=False):
                args.split_stratify = prompt_text("Target column name")

    if prompt_yes_no("Sensor fusion merge?", default=False):
        args.fusion_sources = prompt_text(
            "Fusion sources (name=path,name2=path2)"
        )
        args.fusion_id_col = prompt_text("Fusion ID column")
        args.fusion_time_col = prompt_text("Fusion timestamp column")
        args.fusion_time_tolerance = prompt_text("Fusion time tolerance (e.g. 2s, 500ms)")

    return args


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CleanShot: dataset cleaning on DuckDB. "
            "Drops garbage columns, infers dates, cleans currencies, imputes, dedups."
        )
    )
    parser.add_argument("input", nargs="?", help="Input path or glob (e.g., data/*.csv)")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output Parquet file (default: cleaned.parquet)",
    )
    parser.add_argument(
        "--preset",
        choices=["basic", "ml_ready"],
        default="ml_ready",
        help="Preset: ml_ready enables imputation",
    )
    parser.add_argument("--interactive", action="store_true", help="Guided interactive setup")
    parser.add_argument("--profile", action="store_true", help="Show profile stats and exit")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline without writing output")
    parser.add_argument(
        "--compression",
        choices=["zstd", "snappy", "gzip", "uncompressed"],
        default="zstd",
        help="Parquet compression",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="DuckDB threads (default: CPU count)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "json"],
        help="Override input format detection",
    )
    parser.add_argument(
        "--currency-locale",
        choices=["auto", "us", "eu"],
        default="auto",
        help="Currency format for parsing (default: auto)",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=20000,
        help="Rows to sample for inference (0=full scan)",
    )
    parser.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable row deduplication",
    )
    parser.add_argument(
        "--dedup-keys",
        default=None,
        help="Comma-separated columns to dedup on",
    )
    parser.add_argument(
        "--outlier-mode",
        choices=["none", "percentile", "zscore"],
        default="none",
        help="Outlier clipping mode",
    )
    parser.add_argument(
        "--outlier-pct",
        default="0.01,0.99",
        help="Percentile bounds for clipping (e.g., 0.01,0.99)",
    )
    parser.add_argument(
        "--outlier-zscore",
        type=float,
        default=3.0,
        help="Z-score threshold for clipping",
    )
    parser.add_argument(
        "--scale",
        choices=["none", "standardize", "minmax"],
        default="none",
        help="Add scaled numeric columns",
    )
    parser.add_argument(
        "--scale-prefix",
        default="scale_",
        help="Prefix for scaled columns",
    )
    parser.add_argument(
        "--encode",
        choices=["none", "onehot", "frequency"],
        default="none",
        help="Encode categorical columns",
    )
    parser.add_argument(
        "--encode-prefix",
        default="enc_",
        help="Prefix for encoded columns",
    )
    parser.add_argument(
        "--encode-max-categories",
        type=int,
        default=50,
        help="Max categories for one-hot encoding",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column(s), comma-separated (excluded from encoding/scaling)",
    )
    parser.add_argument(
        "--schema-out",
        default=None,
        help="Write schema/stats JSON to path",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help="Write Markdown report to path",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Train/val/test ratios (e.g., 0.8,0.1,0.1)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for split shuffling",
    )
    parser.add_argument(
        "--split-stratify",
        default=None,
        help="Target column for stratified split",
    )
    parser.add_argument(
        "--fusion-source",
        action="append",
        default=[],
        help="Sensor fusion source (name=path or path). Repeatable.",
    )
    parser.add_argument(
        "--fusion-id-col",
        default=None,
        help="Sensor fusion ID column",
    )
    parser.add_argument(
        "--fusion-time-col",
        default=None,
        help="Sensor fusion timestamp column",
    )
    parser.add_argument(
        "--fusion-time-tolerance",
        default="2s",
        help="Sensor fusion time tolerance (e.g., 2s, 500ms)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit output Parquet file (same as positional output)",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose progress output",
    )

    args = parser.parse_args()
    args.inspect_first = False

    if args.interactive:
        args = run_interactive(args)
    elif args.fusion_source and args.input and args.output is None:
        # Fusion-only runs often want to specify just an output path.
        # Argparse assigns a single positional to `input`, so treat it as output.
        args.output = args.input
        args.input = None
    elif not args.input and not args.fusion_source:
        parser.error("input is required unless --interactive or --fusion-source is set")

    if args.output is None:
        args.output = "cleaned.parquet"

    threads = max(1, args.threads)
    verbose = args.verbose
    sample_rows = max(0, args.sample_rows)

    if args.fusion_source and not args.input:
        args.input = "fusion"

    def log(message: str):
        if verbose:
            print(message)

    con = duckdb.connect(":memory:")
    con.execute(f"SET threads={threads};")

    if args.fusion_source:
        fusion_sources = load_fusion_sources(args.fusion_source)
        if not args.fusion_id_col or not args.fusion_time_col:
            die("Fusion requires --fusion-id-col and --fusion-time-col")
        tolerance_seconds = parse_duration_to_seconds(args.fusion_time_tolerance)
        current_view = build_fusion_query(
            con,
            fusion_sources,
            args.fusion_id_col,
            args.fusion_time_col,
            tolerance_seconds,
        )
        input_format = "fusion"
        load_target = "fusion"
        row_count = con.execute(f"SELECT COUNT(*) FROM {current_view}").fetchone()[0]
        input_size = 0
        size_note = ""
        log(
            f"Loaded {row_count:,} rows from fusion sources "
            f"(sources={len(fusion_sources)}, threads={threads}{size_note})"
        )
    else:
        input_format, load_target = detect_input(args.input, args.format)
        load_query = build_loader(input_format, load_target)
        con.execute(f"CREATE VIEW raw_data AS SELECT * FROM {load_query};")
        current_view = "raw_data"
        row_count = con.execute("SELECT COUNT(*) FROM raw_data").fetchone()[0]
        input_size, input_files = resolve_input_size(load_target, args.input)
        size_note = f", size={format_bytes(input_size)}" if input_size else ""
        log(
            f"Loaded {row_count:,} rows from {args.input} "
            f"(format={input_format}, threads={threads}{size_note})"
        )

    if row_count == 0:
        log("No rows found. Exiting.")
        return 0

    if args.profile:
        print("\n=== Profile ===")
        cur = con.execute(f"SUMMARIZE {current_view}")
        print_table([d[0] for d in cur.description], cur.fetchall())
        return 0

    sample_view = current_view
    if sample_rows > 0 and row_count > sample_rows:
        sample_view = "sample_data"
        con.execute(
            "CREATE VIEW sample_data AS "
            f"SELECT * FROM {current_view} USING SAMPLE {sample_rows} ROWS;"
        )
        log(f"Using sample of {sample_rows:,} rows for inference.")

    if args.inspect_first:
        print("\n## Inspect")
        cur = con.execute(f"SUMMARIZE {current_view}")
        profile_table = table_to_string([d[0] for d in cur.description], cur.fetchall())
        print("```")
        print(profile_table)
        print("```")

    log("\nScanning garbage columns...")
    desc = describe_table(con, current_view)
    cols = [c["name"] for c in desc]
    original_col_count = len(cols)

    stats = collect_column_stats(con, current_view, cols)
    drop_cols = []
    max_null_pct = 0.0
    cols_with_nulls = 0
    for entry in stats:
        col = entry["name"]
        null_pct = entry["null_pct"] or 0.0
        unique_count = entry["unique_count"] or 0
        if null_pct > 0:
            cols_with_nulls += 1
        max_null_pct = max(max_null_pct, null_pct)
        if null_pct is None or unique_count is None:
            continue
        is_constant = row_count > 1 and unique_count <= 1
        if null_pct > 80 or is_constant:
            drop_cols.append(col)
            log(f"  Drop {col}: null={null_pct:.1f}%, unique={unique_count}")

    if args.inspect_first:
        recommended = "ml_ready" if cols_with_nulls > 0 else "basic"
        print("\n## Recommendation")
        print(f"- suggested_preset: `{recommended}`")
        print(f"- columns_with_nulls: {cols_with_nulls}")
        print(f"- max_null_pct: {max_null_pct:.1f}%")
        if drop_cols:
            print(f"- drop_candidates: {summarize_list(drop_cols)}")
        if input_size >= 1024**3:
            print("- note: large input detected; consider higher --threads")
        if not prompt_yes_no("Proceed with cleaning?", default=True):
            return 0
        if args.preset is None:
            while True:
                chosen = prompt_text("Preset (basic/ml_ready)", recommended).lower()
                if chosen in ("basic", "ml_ready"):
                    args.preset = chosen
                    break
                print("Enter basic or ml_ready.")

    if drop_cols:
        keep_cols = [c for c in cols if c not in drop_cols]
        if not keep_cols:
            log("All columns dropped; nothing to output.")
            return 1
        select_cols = ", ".join(sql_ident(c) for c in keep_cols)
        con.execute(f"CREATE VIEW stage1 AS SELECT {select_cols} FROM {current_view};")
        current_view = "stage1"
        desc = describe_table(con, current_view)
        cols = [c["name"] for c in desc]

    log("\nAuto-casting dates...")
    desc = describe_table(con, current_view)
    string_cols = [c["name"] for c in desc if is_string_type(c["type"])]
    date_cols = []
    if string_cols:
        date_exprs = []
        for idx, col in enumerate(string_cols):
            qc = sql_ident(col)
            date_exprs.append(
                f"AVG(CASE WHEN try_cast({qc} AS DATE) IS NOT NULL THEN 1 ELSE 0 END) * 100.0 AS pct_{idx}"
            )
        date_stats = con.execute(f"SELECT {', '.join(date_exprs)} FROM {sample_view}").fetchone()
        for idx, col in enumerate(string_cols):
            pct = date_stats[idx]
            if pct is not None and pct > 90:
                date_cols.append(col)
                log(f"  Cast {col} -> DATE ({pct:.1f}% valid)")

    if date_cols:
        cols = [c["name"] for c in desc]
        exprs = []
        for col in cols:
            qc = sql_ident(col)
            if col in date_cols:
                exprs.append(f"try_cast({qc} AS DATE) AS {qc}")
            else:
                exprs.append(qc)
        con.execute(f"CREATE VIEW stage2 AS SELECT {', '.join(exprs)} FROM {current_view};")
        current_view = "stage2"
        desc = describe_table(con, current_view)
        cols = [c["name"] for c in desc]

    log("\nCleaning currencies...")
    desc = describe_table(con, current_view)
    currency_candidates = []
    for col in desc:
        name = col["name"]
        lower = name.lower()
        if any(key in lower for key in ("price", "amount", "cost", "revenue", "salary", "fee")):
            if is_string_type(col["type"]):
                currency_candidates.append(name)

    if args.interactive and currency_candidates and args.currency_locale == "auto":
        locale_choice = prompt_menu(
            "Currency format",
            ["auto (detect)", "us (1,234.56)", "eu (1.234,56)"],
            default_index=0,
        )
        args.currency_locale = ["auto", "us", "eu"][locale_choice]

    currency_cols = []
    for col in currency_candidates:
        qc = sql_ident(col)
        sample_has_symbols = con.execute(
            "SELECT AVG(flag) FROM ("
            f"SELECT CASE WHEN CAST({qc} AS VARCHAR) LIKE '%$%' OR CAST({qc} AS VARCHAR) LIKE '%,%' "
            "THEN 1 ELSE 0 END AS flag "
            f"FROM {sample_view} WHERE {qc} IS NOT NULL LIMIT 100)"
        ).fetchone()[0]
        if sample_has_symbols and sample_has_symbols > 0.1:
            decimal_comma_pct = con.execute(
                "SELECT AVG(flag) FROM ("
                f"SELECT CASE WHEN strpos(reverse(val), ',') BETWEEN 3 AND 4 THEN 1 ELSE 0 END AS flag "
                f"FROM (SELECT CAST({qc} AS VARCHAR) AS val FROM {sample_view} "
                f"WHERE {qc} IS NOT NULL LIMIT 100))"
            ).fetchone()[0]
            if args.currency_locale == "us":
                decimal_comma = False
            elif args.currency_locale == "eu":
                decimal_comma = True
            else:
                decimal_comma = bool(decimal_comma_pct and decimal_comma_pct > 0.5)
            currency_cols.append((col, decimal_comma))
            log(f"  Clean {col} -> DOUBLE (decimal_comma={decimal_comma})")

    if currency_cols:
        cols = [c["name"] for c in desc]
        exprs = []
        for col in cols:
            qc = sql_ident(col)
            match = next((entry for entry in currency_cols if entry[0] == col), None)
            if match:
                _, decimal_comma = match
                if decimal_comma:
                    cleaned = f"CAST(regexp_replace(regexp_replace(regexp_replace(CAST({qc} AS VARCHAR), '[ $]', '', 'g'), ',', '.', 'g'), '[^0-9.\\-]', '', 'g') AS DOUBLE) AS {qc}"
                else:
                    cleaned = f"CAST(regexp_replace(regexp_replace(CAST({qc} AS VARCHAR), '[ $,]', '', 'g'), '[^0-9.\\-]', '', 'g') AS DOUBLE) AS {qc}"
                exprs.append(cleaned)
            else:
                exprs.append(qc)
        con.execute(f"CREATE VIEW stage3 AS SELECT {', '.join(exprs)} FROM {current_view};")
        current_view = "stage3"
        desc = describe_table(con, current_view)
        cols = [c["name"] for c in desc]

    num_cols = []
    cat_cols = []
    medians = {}
    modes = {}
    desc = describe_table(con, current_view)
    num_cols = [c["name"] for c in desc if is_numeric_type(c["type"])]
    cat_cols = [
        c["name"]
        for c in desc
        if not is_numeric_type(c["type"]) and not is_date_type(c["type"])
    ]

    if args.preset == "ml_ready":
        log("\nImputing...")

        medians = {}
        if num_cols:
            med_exprs = [f"median({sql_ident(c)}) AS med_{idx}" for idx, c in enumerate(num_cols)]
            med_row = con.execute(f"SELECT {', '.join(med_exprs)} FROM {current_view}").fetchone()
            for idx, col in enumerate(num_cols):
                medians[col] = med_row[idx]

        modes = {}
        if cat_cols:
            mode_exprs = []
            for idx, col in enumerate(cat_cols):
                qc = sql_ident(col)
                mode_exprs.append(
                    f"(SELECT {qc} FROM {current_view} WHERE {qc} IS NOT NULL "
                    f"GROUP BY {qc} ORDER BY COUNT(*) DESC LIMIT 1) AS mode_{idx}"
                )
            mode_row = con.execute(f"SELECT {', '.join(mode_exprs)}").fetchone()
            for idx, col in enumerate(cat_cols):
                modes[col] = mode_row[idx]

        exprs = []
        for col in [c["name"] for c in desc]:
            qc = sql_ident(col)
            if col in medians and medians[col] is not None:
                exprs.append(f"coalesce({qc}, {sql_literal(medians[col])}) AS {qc}")
            elif col in modes and modes[col] is not None:
                exprs.append(f"coalesce({qc}, {sql_literal(modes[col])}) AS {qc}")
            else:
                exprs.append(qc)

        con.execute(f"CREATE VIEW stage4 AS SELECT {', '.join(exprs)} FROM {current_view};")
        current_view = "stage4"
        desc = describe_table(con, current_view)
        num_cols = [c["name"] for c in desc if is_numeric_type(c["type"])]
        cat_cols = [
            c["name"]
            for c in desc
            if not is_numeric_type(c["type"]) and not is_date_type(c["type"])
        ]

    target_cols = set(parse_list(args.target))
    if target_cols:
        ensure_columns_exist(target_cols, {c["name"] for c in desc}, "Target")

    outlier_bounds = {}
    outlier_clipped_cols = 0
    if args.outlier_mode != "none" and num_cols:
        log("\nClipping outliers...")
        lower_pct, upper_pct = parse_outlier_pct(args.outlier_pct)
        outlier_bounds = resolve_outlier_bounds(
            con,
            current_view,
            num_cols,
            args.outlier_mode,
            lower_pct,
            upper_pct,
            args.outlier_zscore,
        )
        current_view, outlier_clipped_cols = apply_outlier_clipping(
            con, current_view, num_cols, outlier_bounds
        )
        desc = describe_table(con, current_view)
        num_cols = [c["name"] for c in desc if is_numeric_type(c["type"])]
        cat_cols = [
            c["name"]
            for c in desc
            if not is_numeric_type(c["type"]) and not is_date_type(c["type"])
        ]

    scale_added = 0
    scaled_cols = []
    if args.scale != "none" and num_cols:
        log("\nAdding scaled columns...")
        if args.scale == "standardize":
            current_view, scale_added, scaled_cols = add_standardized_columns(
                con,
                current_view,
                num_cols,
                target_cols,
                args.scale_prefix,
            )
        else:
            current_view, scale_added, scaled_cols = add_min_max_columns(
                con,
                current_view,
                num_cols,
                target_cols,
                args.scale_prefix,
            )
        desc = describe_table(con, current_view)
        num_cols = [c["name"] for c in desc if is_numeric_type(c["type"])]
        cat_cols = [
            c["name"]
            for c in desc
            if not is_numeric_type(c["type"]) and not is_date_type(c["type"])
        ]

    encode_added = 0
    encode_mapping = {}
    encoded_cols = []
    if args.encode != "none" and cat_cols:
        log("\nEncoding categoricals...")
        if args.encode == "onehot":
            current_view, encode_added, encode_mapping, encoded_cols = expand_one_hot_columns(
                con,
                current_view,
                cat_cols,
                target_cols,
                args.encode_prefix,
                args.encode_max_categories,
            )
        else:
            current_view, encode_added, encode_mapping, encoded_cols = add_frequency_encoded_columns(
                con,
                current_view,
                cat_cols,
                target_cols,
                args.encode_prefix,
            )
        desc = describe_table(con, current_view)
        num_cols = [c["name"] for c in desc if is_numeric_type(c["type"])]
        cat_cols = [
            c["name"]
            for c in desc
            if not is_numeric_type(c["type"]) and not is_date_type(c["type"])
        ]
    elif args.encode != "none":
        encode_mapping = {"warning": "no categorical columns found"}

    dedup_removed = 0

    if args.dedup:
        log("\nDeduplicating...")
        if args.dedup_keys:
            keys = [k.strip() for k in args.dedup_keys.split(",") if k.strip()]
            invalid_keys = [k for k in keys if k not in {c["name"] for c in desc}]
            if invalid_keys:
                die(
                    "Invalid dedup keys: " + ", ".join(invalid_keys),
                    "Use comma-separated column names from the dataset.",
                )
            if not keys:
                die("No valid dedup keys provided.")
            partition = ", ".join(sql_ident(k) for k in keys)
            con.execute(
                "CREATE VIEW cleaned AS "
                f"SELECT * EXCLUDE (cs_rownum) FROM ("
                f"SELECT *, row_number() OVER (PARTITION BY {partition}) AS cs_rownum "
                f"FROM {current_view}) WHERE cs_rownum = 1;"
            )
        else:
            con.execute(f"CREATE VIEW cleaned AS SELECT DISTINCT * FROM {current_view};")
    else:
        log("\nSkipping dedup...")
        con.execute(f"CREATE VIEW cleaned AS SELECT * FROM {current_view};")

    split_counts = None
    if args.split:
        train_ratio, val_ratio, test_ratio = parse_split_ratios(args.split)
        train_count, val_count, test_count = resolve_split_counts(
            row_count, train_ratio, val_ratio, test_ratio
        )
        if args.split_stratify:
            ensure_columns_exist(
                [args.split_stratify], {c["name"] for c in desc}, "Stratify"
            )
            train_query, val_query, test_query = build_stratified_split_queries(
                "cleaned",
                args.split_stratify,
                train_ratio,
                val_ratio,
                test_ratio,
                args.split_seed,
            )
        else:
            train_query, val_query, test_query = build_split_queries(
                "cleaned", args.split_seed, train_count, val_count
            )
        con.execute(f"CREATE VIEW split_train AS {train_query};")
        con.execute(f"CREATE VIEW split_val AS {val_query};")
        con.execute(f"CREATE VIEW split_test AS {test_query};")
        split_counts = {
            "train": con.execute("SELECT COUNT(*) FROM split_train").fetchone()[0],
            "val": con.execute("SELECT COUNT(*) FROM split_val").fetchone()[0],
            "test": con.execute("SELECT COUNT(*) FROM split_test").fetchone()[0],
        }
        final_count = split_counts["train"] + split_counts["val"] + split_counts["test"]
    else:
        final_count = con.execute("SELECT COUNT(*) FROM cleaned").fetchone()[0]
    dedup_removed = row_count - final_count
    log(f"Done! {final_count:,} rows (from {row_count:,}).")

    if args.dry_run:
        log("Dry run complete; no output written.")
    else:
        compression = args.compression.upper()
        if args.split:
            output_path = Path(args.output)
            train_path = output_path.with_name(output_path.stem + "_train" + output_path.suffix)
            val_path = output_path.with_name(output_path.stem + "_val" + output_path.suffix)
            test_path = output_path.with_name(output_path.stem + "_test" + output_path.suffix)
            con.execute(
                f"COPY (SELECT * FROM split_train) TO {sql_string_literal(str(train_path))} "
                f"(FORMAT PARQUET, COMPRESSION '{compression}');"
            )
            con.execute(
                f"COPY (SELECT * FROM split_val) TO {sql_string_literal(str(val_path))} "
                f"(FORMAT PARQUET, COMPRESSION '{compression}');"
            )
            con.execute(
                f"COPY (SELECT * FROM split_test) TO {sql_string_literal(str(test_path))} "
                f"(FORMAT PARQUET, COMPRESSION '{compression}');"
            )
            log(f"Wrote {train_path}.")
            log(f"Wrote {val_path}.")
            log(f"Wrote {test_path}.")
        else:
            output_path = sql_string_literal(args.output)
            con.execute(
                f"COPY (SELECT * FROM cleaned) TO {output_path} "
                f"(FORMAT PARQUET, COMPRESSION '{compression}');"
            )
            log(f"Wrote {args.output}.")

    final_desc = describe_table(con, "cleaned")
    final_col_count = len(final_desc)
    dropped_count = len(drop_cols)
    imputed_numeric = []
    imputed_categorical = []
    if args.preset == "ml_ready":
        imputed_numeric = [c for c in num_cols if medians.get(c) is not None]
        imputed_categorical = [c for c in cat_cols if modes.get(c) is not None]

    if args.split:
        final_desc = describe_table(con, "split_train")
        final_col_count = len(final_desc)

    output_size = None
    if not args.dry_run:
        if args.split:
            out_path = Path(args.output)
            for suffix in ("_train", "_val", "_test"):
                split_path = out_path.with_name(out_path.stem + suffix + out_path.suffix)
                if split_path.exists() and split_path.is_file():
                    output_size = (output_size or 0) + split_path.stat().st_size
        else:
            out_path = Path(args.output)
            if out_path.exists() and out_path.is_file():
                output_size = out_path.stat().st_size

    report_lines = []
    report_lines.append("# CleanShot Report")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append(f"- input: `{args.input}`")
    if args.dry_run:
        report_lines.append("- output: `dry-run`")
    elif args.split:
        output_path = Path(args.output)
        report_lines.append(
            f"- output: `{output_path.with_name(output_path.stem + '_train' + output_path.suffix)}`"
        )
        report_lines.append(
            f"- output_val: `{output_path.with_name(output_path.stem + '_val' + output_path.suffix)}`"
        )
        report_lines.append(
            f"- output_test: `{output_path.with_name(output_path.stem + '_test' + output_path.suffix)}`"
        )
    else:
        report_lines.append(f"- output: `{args.output}`")
    report_lines.append(f"- format: `{input_format}`")
    report_lines.append(f"- preset: `{args.preset}`")
    report_lines.append(f"- threads: {threads}")
    report_lines.append(f"- rows_in: {row_count:,}")
    report_lines.append(f"- rows_out: {final_count:,}")
    if split_counts:
        report_lines.append(
            f"- split_counts: train={split_counts['train']:,}, val={split_counts['val']:,}, test={split_counts['test']:,}"
        )
    report_lines.append(f"- columns_in: {original_col_count}")
    report_lines.append(f"- columns_out: {final_col_count}")
    report_lines.append(f"- dedup_removed: {dedup_removed:,}" if args.dedup else "- dedup_removed: skipped")
    if input_size:
        report_lines.append(f"- input_size: {format_bytes(input_size)}")
    if output_size is not None:
        report_lines.append(f"- output_size: {format_bytes(output_size)}")
    report_lines.append("")
    report_lines.append("## Steps")
    report_lines.append(f"- drop_garbage: {'done' if dropped_count else 'skipped'} ({dropped_count} columns)")
    report_lines.append(f"- date_cast: {'done' if date_cols else 'skipped'} ({len(date_cols)} columns)")
    report_lines.append(
        f"- currency_clean: {'done' if currency_cols else 'skipped'} ({len(currency_cols)} columns)"
    )
    if args.preset == "ml_ready":
        report_lines.append(
            f"- impute: done (numeric {len(imputed_numeric)}, categorical {len(imputed_categorical)})"
        )
    else:
        report_lines.append("- impute: skipped (preset basic)")
    if args.outlier_mode != "none":
        report_lines.append(
            f"- outlier_clip: {'done' if outlier_clipped_cols else 'skipped'} ({outlier_clipped_cols} columns)"
        )
    if args.scale != "none":
        report_lines.append(
            f"- scale: {'done' if scale_added else 'skipped'} ({scale_added} columns)"
        )
    if args.encode != "none":
        report_lines.append(
            f"- encode: {'done' if encode_added else 'skipped'} ({encode_added} columns)"
        )
    if args.fusion_source:
        report_lines.append(f"- fusion_merge: done ({len(args.fusion_source)} sources)")
    if args.dedup:
        report_lines.append(
            f"- dedup: done ({dedup_removed:,} rows removed)" if dedup_removed else "- dedup: done"
        )
    else:
        report_lines.append("- dedup: skipped")
    if args.split:
        report_lines.append(f"- split: done ({args.split})")
    report_lines.append("- write: skipped (dry-run)" if args.dry_run else "- write: done")
    report_lines.append("")
    report_lines.append("## Columns")
    report_lines.append(f"- dropped: {summarize_list(drop_cols)}")
    report_lines.append(f"- date_cast: {summarize_list(date_cols)}")
    report_lines.append(
        f"- currency_cleaned: {summarize_list([c for c, _ in currency_cols])}"
    )
    if args.preset == "ml_ready":
        report_lines.append(f"- imputed_numeric: {summarize_list(imputed_numeric)}")
        report_lines.append(f"- imputed_categorical: {summarize_list(imputed_categorical)}")
    if args.encode != "none" and encoded_cols:
        report_lines.append(f"- encoded_cols: {summarize_list(encoded_cols)}")
    if args.scale != "none" and scaled_cols:
        report_lines.append(f"- scaled_cols: {summarize_list(scaled_cols)}")

    print("\n" + "\n".join(report_lines))

    report_path = normalize_report_target(args.report_out)
    if report_path:
        report_file = Path(report_path)
        if report_file.parent != Path("."):
            report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        log(f"Wrote report to {report_file}.")

    if args.schema_out:
        schema_payload = build_schema_export(
            con,
            "cleaned",
            args.input,
            None if args.dry_run else args.output,
            {
                "dropped": drop_cols,
                "date_cast": date_cols,
                "currency_cleaned": [c for c, _ in currency_cols],
                "imputed_numeric": imputed_numeric,
                "imputed_categorical": imputed_categorical,
                "outlier_bounds": outlier_bounds,
                "encoded": encode_mapping,
                "encoded_columns": encoded_cols,
                "scaled_columns": scaled_cols,
                "split": args.split,
                "split_stratify": args.split_stratify,
                "fusion_sources": args.fusion_source,
            },
        )
        schema_path = normalize_path_target(args.schema_out)
        if schema_path:
            write_json(schema_path, schema_payload)
            log(f"Wrote schema to {schema_path}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
