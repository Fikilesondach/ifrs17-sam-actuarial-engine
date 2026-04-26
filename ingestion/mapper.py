# ingestion/mapper.py
# ══════════════════════════════════════════════════════════════════════
# COLUMN MAPPER — Step 2 of the ingestion pipeline
#
# Renames source columns to canonical names and converts coded values
# to standard values, using the YAML mapping configuration.
#
# After this step, every DataFrame has the same column names regardless
# of which source system it came from.
# ══════════════════════════════════════════════════════════════════════

import pandas as pd


def apply_mapping(df, config):
    """
    Apply column name mapping and value conversion to a raw DataFrame.

    Parameters
    ──────────
    df     : Raw DataFrame from reader.py (source column names)
    config : Parsed YAML mapping configuration

    Returns
    ───────
    pd.DataFrame with canonical column names and standardised values
    """
    df = df.copy()

    # ── Step 1: Rename columns ─────────────────────────────────────────
    df = _rename_columns(df, config)

    # ── Step 2: Convert coded values ──────────────────────────────────
    df = _convert_values(df, config)

    return df


def _rename_columns(df, config):
    """
    Rename source columns to canonical names using field_mappings.

    Only renames columns that exist in the source AND are mapped.
    Unmapped source columns are dropped — they are not needed downstream.
    Mapped fields that are missing from source are added as empty columns
    so downstream steps can handle them consistently.
    """
    field_mappings = config.get("field_mappings", {})

    # Invert: {source_name: canonical_name}
    rename_map = {v: k for k, v in field_mappings.items()}

    # Report what we found and what is missing
    found   = [src for src in rename_map if src in df.columns]
    missing = [src for src in rename_map if src not in df.columns]

    if missing:
        print(f"  Warning — mapped columns not found in source: {missing}")

    # Rename what we have
    df = df.rename(columns=rename_map)

    # Keep only canonical columns (drop all unmapped source columns)
    canonical_cols = list(field_mappings.keys())
    existing_canonical = [c for c in canonical_cols if c in df.columns]
    df = df[existing_canonical].copy()

    # Add missing canonical columns as empty strings
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = ""

    # Reorder to canonical order
    df = df[canonical_cols]

    print(f"  Columns mapped    : {len(found)}/{len(rename_map)}")
    print(f"  Canonical columns : {len(canonical_cols)}")

    return df


def _convert_values(df, config):
    """
    Convert coded source values to canonical standard values.

    Example: gender "1" → "M", "2" → "F"
    Uses value_mappings from the YAML configuration.
    Unmapped values are left as-is with a warning logged.
    """
    value_mappings = config.get("value_mappings", {})
    unmapped_log   = {}

    for field, mapping in value_mappings.items():
        if field not in df.columns:
            continue

        # Convert mapping keys to strings for consistent comparison
        str_mapping = {str(k): v for k, v in mapping.items()}

        def convert(val):
            val_str = str(val).strip()
            if val_str in str_mapping:
                return str_mapping[val_str]
            elif val_str in ("", "nan", "None"):
                return ""
            else:
                # Log unmapped value but do not crash
                if field not in unmapped_log:
                    unmapped_log[field] = set()
                unmapped_log[field].add(val_str)
                return val_str

        df[field] = df[field].apply(convert)

    if unmapped_log:
        print(f"  Warning — unmapped values found:")
        for field, values in unmapped_log.items():
            print(f"    {field}: {values}")

    print(f"  Value mappings applied: {len(value_mappings)} fields")
    return df
