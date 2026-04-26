# ingestion/reader.py
# ══════════════════════════════════════════════════════════════════════
# FORMAT READER — Step 1 of the ingestion pipeline
#
# Detects the file format from the extension and reads it into a
# pandas DataFrame using the configuration from the mapping YAML.
#
# Supports: CSV (any delimiter), Excel (.xlsx), JSON (any envelope)
# ══════════════════════════════════════════════════════════════════════

import pandas as pd
import json
import os


def detect_format(file_path):
    """
    Detect file format from extension.
    Returns 'csv', 'excel', or 'json'.
    Raises ValueError for unsupported formats.
    """
    ext = os.path.splitext(file_path)[1].lower()

    format_map = {
        ".csv" : "csv",
        ".txt" : "csv",     # Pipe-delimited txt files treated as CSV
        ".xlsx": "excel",
        ".xls" : "excel",
        ".json": "json",
    }

    if ext not in format_map:
        raise ValueError(
            f"Unsupported file format: '{ext}'\n"
            f"Supported formats: {list(format_map.keys())}"
        )

    detected = format_map[ext]
    print(f"  File format detected: {detected.upper()} ({ext})")
    return detected


def read_file(file_path, config):
    """
    Read a policy data file into a raw DataFrame.

    Parameters
    ──────────
    file_path : Path to the source data file
    config    : Parsed YAML mapping configuration

    Returns
    ───────
    pd.DataFrame — raw data exactly as it appears in the source file
                   (no column renaming or value mapping yet)
    """
    fmt = detect_format(file_path)

    if fmt == "csv":
        df = _read_csv(file_path, config)
    elif fmt == "excel":
        df = _read_excel(file_path, config)
    elif fmt == "json":
        df = _read_json(file_path, config)

    print(f"  Rows read      : {len(df):,}")
    print(f"  Columns found  : {len(df.columns)}")
    print(f"  Column names   : {df.columns.tolist()}")

    return df


def _read_csv(file_path, config):
    """
    Read CSV with configuration-driven parameters.
    Handles any delimiter, any number of skip rows.
    """
    csv_cfg   = config.get("file_formats", {}).get("csv", {})
    delimiter = csv_cfg.get("delimiter", ",")
    skip_rows = csv_cfg.get("skip_rows", 0)
    encoding  = csv_cfg.get("encoding", "utf-8")

    # Build list of rows to skip
    # skip_rows=1 means skip the first row (index 0)
    skip = list(range(skip_rows)) if skip_rows > 0 else None

    df = pd.read_csv(
        file_path,
        sep        = delimiter,
        skiprows   = skip,
        encoding   = encoding,
        dtype      = str,          # Read everything as string first
                                   # Type conversion happens in deriver.py
        keep_default_na = False,   # Preserve empty strings as "" not NaN
    )

    # Strip whitespace from column names and string values
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    return df


def _read_excel(file_path, config):
    """
    Read Excel file from the configured sheet.
    Handles files with metadata sheets alongside the data sheet.
    """
    excel_cfg  = config.get("file_formats", {}).get("excel", {})
    sheet_name = excel_cfg.get("sheet_name", 0)    # 0 = first sheet
    header_row = excel_cfg.get("header_row", 0)

    df = pd.read_excel(
        file_path,
        sheet_name = sheet_name,
        header     = header_row,
        dtype      = str,
    )

    # Remove completely empty rows (common in Excel exports)
    df = df.dropna(how="all")

    # Strip whitespace
    df.columns = df.columns.astype(str).str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    return df


def _read_json(file_path, config):
    """
    Read JSON file — handles both direct arrays and API envelope format.

    Two supported structures:
        Structure A (direct array):
            [{"POL_NUM": "POL001", ...}, {"POL_NUM": "POL002", ...}]

        Structure B (API envelope — what our generator produces):
            {
                "status": "SUCCESS",
                "policies": [{"POL_NUM": "POL001", ...}, ...]
            }
    """
    json_cfg     = config.get("file_formats", {}).get("json", {})
    records_path = json_cfg.get("records_path", None)

    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # ── Structure A: JSON is already an array ──────────────────────────
    if isinstance(raw, list):
        print("  JSON structure : Direct array")
        records = raw

    # ── Structure B: JSON is an envelope object ────────────────────────
    elif isinstance(raw, dict):
        if records_path and records_path in raw:
            print(f"  JSON structure : Envelope (key='{records_path}')")
            records = raw[records_path]
        else:
            # Try common key names if records_path not specified or wrong
            for key in ["policies", "data", "records", "items", "results"]:
                if key in raw:
                    print(f"  JSON structure : Envelope (auto-detected key='{key}')")
                    records = raw[key]
                    break
            else:
                raise ValueError(
                    f"JSON file is an object but no records array found.\n"
                    f"Top-level keys: {list(raw.keys())}\n"
                    f"Set 'records_path' in your mapping YAML to the correct key."
                )
    else:
        raise ValueError(f"Unexpected JSON structure: {type(raw)}")

    df = pd.DataFrame(records)

    # Convert all columns to string for consistent downstream processing
    df = df.astype(str)
    df = df.replace("None", "")
    df = df.replace("nan", "")

    return df
