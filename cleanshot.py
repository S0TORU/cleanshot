#!/usr/bin/env python3
"""
CleanShot: Dataset cleaning CLI powered by DuckDB.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import os
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
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose progress output",
    )

    args = parser.parse_args()
    args.inspect_first = False

    if args.interactive:
        args = run_interactive(args)
    elif not args.input:
        parser.error("input is required unless --interactive is set")

    if args.output is None:
        args.output = "cleaned.parquet"

    threads = max(1, args.threads)
    verbose = args.verbose

    def log(message: str):
        if verbose:
            print(message)

    con = duckdb.connect(":memory:")
    con.execute(f"SET threads={threads};")

    input_format, load_target = detect_input(args.input, args.format)
    load_query = build_loader(input_format, load_target)
    con.execute(f"CREATE VIEW raw_data AS SELECT * FROM {load_query};")

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
        cur = con.execute("SUMMARIZE raw_data")
        print_table([d[0] for d in cur.description], cur.fetchall())
        return 0

    current_view = "raw_data"

    if args.inspect_first:
        print("\n## Inspect")
        cur = con.execute("SUMMARIZE raw_data")
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
        if null_pct > 80 or unique_count <= 1:
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
        date_stats = con.execute(f"SELECT {', '.join(date_exprs)} FROM {current_view}").fetchone()
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
            f"FROM {current_view} WHERE {qc} IS NOT NULL LIMIT 100)"
        ).fetchone()[0]
        if sample_has_symbols and sample_has_symbols > 0.1:
            decimal_comma_pct = con.execute(
                "SELECT AVG(flag) FROM ("
                f"SELECT CASE WHEN strpos(reverse(val), ',') BETWEEN 3 AND 4 THEN 1 ELSE 0 END AS flag "
                f"FROM (SELECT CAST({qc} AS VARCHAR) AS val FROM {current_view} "
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
                    cleaned = (
                        f"CAST(regexp_replace(regexp_replace(regexp_replace(CAST({qc} AS VARCHAR), "
                        "'[ $]', '', 'g'), ',', '.', 'g'), '[^0-9.\\-]', '', 'g') AS DOUBLE) AS {qc}"
                    )
                else:
                    cleaned = (
                        f"CAST(regexp_replace(regexp_replace(CAST({qc} AS VARCHAR), '[ $,]', '', 'g'), "
                        "'[^0-9.\\-]', '', 'g') AS DOUBLE) AS {qc}"
                    )
                exprs.append(cleaned)
            else:
                exprs.append(qc)
        con.execute(f"CREATE VIEW stage3 AS SELECT {', '.join(exprs)} FROM {current_view};")
        current_view = "stage3"

    num_cols = []
    cat_cols = []
    medians = {}
    modes = {}
    if args.preset == "ml_ready":
        log("\nImputing...")
        desc = describe_table(con, current_view)
        num_cols = [c["name"] for c in desc if is_numeric_type(c["type"])]
        cat_cols = [
            c["name"]
            for c in desc
            if not is_numeric_type(c["type"]) and not is_date_type(c["type"])
        ]

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

    log("\nDeduplicating...")
    con.execute(f"CREATE VIEW cleaned AS SELECT DISTINCT * FROM {current_view};")
    final_count = con.execute("SELECT COUNT(*) FROM cleaned").fetchone()[0]
    log(f"Done! {final_count:,} rows (from {row_count:,}).")

    if args.dry_run:
        log("Dry run complete; no output written.")
    else:
        compression = args.compression.upper()
        output_path = sql_string_literal(args.output)
        con.execute(
            f"COPY (SELECT * FROM cleaned) TO {output_path} "
            f"(FORMAT PARQUET, COMPRESSION '{compression}');"
        )
        log(f"Wrote {args.output}.")

    final_desc = describe_table(con, "cleaned")
    final_col_count = len(final_desc)
    dropped_count = len(drop_cols)
    dedup_removed = row_count - final_count
    imputed_numeric = []
    imputed_categorical = []
    if args.preset == "ml_ready":
        imputed_numeric = [c for c in num_cols if medians.get(c) is not None]
        imputed_categorical = [c for c in cat_cols if modes.get(c) is not None]

    output_size = None
    if not args.dry_run:
        out_path = Path(args.output)
        if out_path.exists() and out_path.is_file():
            output_size = out_path.stat().st_size

    report_lines = []
    report_lines.append("# CleanShot Report")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append(f"- input: `{args.input}`")
    report_lines.append(f"- output: `{args.output}`" if not args.dry_run else "- output: `dry-run`")
    report_lines.append(f"- format: `{input_format}`")
    report_lines.append(f"- preset: `{args.preset}`")
    report_lines.append(f"- threads: {threads}")
    report_lines.append(f"- rows_in: {row_count:,}")
    report_lines.append(f"- rows_out: {final_count:,}")
    report_lines.append(f"- columns_in: {original_col_count}")
    report_lines.append(f"- columns_out: {final_col_count}")
    report_lines.append(f"- dedup_removed: {dedup_removed:,}")
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
    report_lines.append(
        f"- dedup: done ({dedup_removed:,} rows removed)" if dedup_removed else "- dedup: done"
    )
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

    print("\n" + "\n".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
