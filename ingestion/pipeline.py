# ingestion/pipeline.py
# ══════════════════════════════════════════════════════════════════════
# INGESTION PIPELINE — Orchestrates all 6 steps
#
# Usage:
#   from ingestion.pipeline import run_pipeline
#   df_valid, df_errors, report = run_pipeline(
#       file_path      = "data/synthetic/abc_life_export.csv",
#       mapping_path   = "mappings/synthetic_data.yaml",
#       valuation_date = date(2024, 12, 31)
#   )
# ══════════════════════════════════════════════════════════════════════

import yaml
import pandas as pd
import os
from datetime import date

from ingestion.reader    import read_file
from ingestion.mapper    import apply_mapping
from ingestion.deriver   import derive_all_fields
from ingestion.validator import validate


def load_config(mapping_path):
    """Load and parse the YAML mapping configuration."""
    with open(mapping_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"  Config loaded  : {config.get('source_system', 'Unknown')}")
    return config


def run_pipeline(file_path, mapping_path, valuation_date=None, save_outputs=True):
    """
    Run the full data ingestion pipeline.

    Steps:
        1. Read file (CSV, Excel, or JSON)
        2. Map columns to canonical names
        3. Derive computed fields
        4. Validate all records
        5. Save canonical output and data quality report

    Parameters
    ──────────
    file_path      : Path to source data file
    mapping_path   : Path to YAML mapping configuration
    valuation_date : Valuation date (date object). Defaults to config value.
    save_outputs   : Whether to save results to outputs/ folder

    Returns
    ───────
    df_valid  : Clean, validated DataFrame ready for valuation engine
    df_errors : Records that failed validation
    report    : Data quality report dictionary
    """
    print("=" * 55)
    print("  IFRS 17 / SAM — Data Ingestion Pipeline")
    print("=" * 55)
    print(f"\nInput file     : {file_path}")
    print(f"Mapping config : {mapping_path}")

    # ── Step 1: Read ───────────────────────────────────────────────────
    print(f"\nStep 1: Reading file...")
    config = load_config(mapping_path)
    df_raw = read_file(file_path, config)

    # ── Step 2: Map columns ────────────────────────────────────────────
    print(f"\nStep 2: Mapping columns...")
    df_mapped = apply_mapping(df_raw, config)

    # ── Step 3: Derive fields ──────────────────────────────────────────
    print(f"\nStep 3: Deriving fields...")
    df_derived = derive_all_fields(df_mapped, config, valuation_date)

    # ── Step 4: Validate ───────────────────────────────────────────────
    print(f"\nStep 4: Validating records...")
    df_valid, df_errors, report = validate(df_derived)

    # ── Step 5: Save outputs ───────────────────────────────────────────
    if save_outputs:
        print(f"\nStep 5: Saving outputs...")
        _save_outputs(df_valid, df_errors, report)

    print(f"\n{'='*55}")
    print(f"  Pipeline complete ✓")
    print(f"  {len(df_valid):,} records ready for valuation engine")
    print(f"{'='*55}")

    return df_valid, df_errors, report


def _save_outputs(df_valid, df_errors, report):
    """Save canonical data and quality report to outputs folder."""
    import json

    os.makedirs("outputs/canonical",            exist_ok=True)
    os.makedirs("outputs/data_quality_reports", exist_ok=True)

    # Save valid records as Parquet (fast, typed, preserves dtypes)
    valid_path = "outputs/canonical/policies_canonical.parquet"
    df_valid.to_parquet(valid_path, index=False)
    print(f"  Valid records → {valid_path}")

    # Save error records as CSV for manual review
    if len(df_errors) > 0:
        error_path = "outputs/data_quality_reports/validation_errors.csv"
        df_errors.to_csv(error_path, index=False)
        print(f"  Error records → {error_path}")

    # Save report as JSON
    report_path = "outputs/data_quality_reports/quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Quality report → {report_path}")
