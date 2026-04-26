# ingestion/validator.py
# ══════════════════════════════════════════════════════════════════════
# VALIDATION ENGINE — Step 4 of the ingestion pipeline
#
# Applies tiered validation rules to every policy record:
#   Tier 1 — Hard errors   : record excluded from valuation
#   Tier 2 — Warnings      : record included but flagged
#   Tier 3 — Anomalies     : statistical outliers logged
#
# Produces a data quality report showing counts and reasons.
# ══════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np


VALID_PRODUCTS  = {"TERM", "WL", "ENDOW", "DISAB", "ANNU"}
VALID_GENDERS   = {"M", "F"}
VALID_SMOKERS   = {"S", "NS"}
VALID_STATUSES  = {"IF", "PU", "LA"}
VALID_FREQS     = {"MONTHLY", "QUARTERLY", "ANNUAL"}
VALID_MODELS    = {"GMM", "PAA", "VFA"}
MAX_SA          = 50_000_000
MIN_AGE         = 0
MAX_AGE_ENTRY   = 80
MAX_TERM        = 50
MAX_ESCALATION  = 0.20


def validate(df):
    """
    Run all validation rules and return cleaned DataFrame plus report.

    Parameters
    ──────────
    df : Derived DataFrame from deriver.py

    Returns
    ───────
    df_valid  : Records that passed Tier 1 (may have Tier 2/3 flags)
    df_errors : Records that failed Tier 1 (excluded from valuation)
    report    : Dict summarising validation results
    """
    df = df.copy()

    # Initialise tracking columns
    df["_errors"]    = ""    # Tier 1 — fatal errors
    df["_warnings"]  = ""    # Tier 2 — warnings
    df["_anomalies"] = ""    # Tier 3 — statistical anomalies

    # ── Tier 1: Hard errors ────────────────────────────────────────────
    df = _check_tier1(df)

    # ── Tier 2: Warnings ──────────────────────────────────────────────
    df = _check_tier2(df)

    # ── Tier 3: Anomalies ─────────────────────────────────────────────
    df = _check_tier3(df)

    # ── Split into valid and error sets ───────────────────────────────
    has_error = df["_errors"] != ""
    df_errors = df[has_error].copy()
    df_valid  = df[~has_error].copy()

    # ── Build report ───────────────────────────────────────────────────
    report = _build_report(df, df_valid, df_errors)

    return df_valid, df_errors, report


def _add_flag(df, mask, col, message):
    """Append a message to the flag column for rows matching mask."""
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: f"{x}; {message}" if x else message
    )
    return df


def _check_tier1(df):
    """Hard errors — record is excluded from valuation."""

    # Null or empty policy ID
    df = _add_flag(df,
        df["policy_id"].isin(["", "nan", "None"]) | df["policy_id"].isnull(),
        "_errors", "MISSING_POLICY_ID"
    )

    # Duplicate policy IDs
    dupes = df["policy_id"].duplicated(keep=False)
    df = _add_flag(df, dupes, "_errors", "DUPLICATE_POLICY_ID")

    # Missing date of birth
    df = _add_flag(df,
        df["date_of_birth"].isnull(),
        "_errors", "MISSING_DATE_OF_BIRTH"
    )

    # Missing inception date
    df = _add_flag(df,
        df["inception_date"].isnull(),
        "_errors", "MISSING_INCEPTION_DATE"
    )

    # Sum assured is zero, negative, or missing
    sa = pd.to_numeric(df["sum_assured"], errors="coerce")
    df = _add_flag(df,
        sa.isnull() | (sa <= 0),
        "_errors", "INVALID_SUM_ASSURED"
    )

    # Premium is negative
    prem = pd.to_numeric(df["premium_amount"], errors="coerce")
    df = _add_flag(df,
        prem.isnull() | (prem < 0),
        "_errors", "INVALID_PREMIUM"
    )

    # Invalid product code
    df = _add_flag(df,
        ~df["product_code"].isin(VALID_PRODUCTS),
        "_errors", "INVALID_PRODUCT_CODE"
    )

    # Invalid policy status
    df = _add_flag(df,
        ~df["policy_status"].isin(VALID_STATUSES),
        "_errors", "INVALID_POLICY_STATUS"
    )

    # Age at entry out of range
    age = pd.to_numeric(df["age_at_entry"], errors="coerce")
    df = _add_flag(df,
        age.isnull() | (age < MIN_AGE) | (age > MAX_AGE_ENTRY),
        "_errors", "INVALID_AGE_AT_ENTRY"
    )

    # Expiry before inception (for non-whole-life)
    has_expiry = df["expiry_date"].notnull()
    bad_expiry = has_expiry & (df["expiry_date"] <= df["inception_date"])
    df = _add_flag(df, bad_expiry, "_errors", "EXPIRY_BEFORE_INCEPTION")

    return df


def _check_tier2(df):
    """Warnings — record is included but flagged for review."""

    # Only check records that passed Tier 1
    valid_mask = df["_errors"] == ""

    age = pd.to_numeric(df["age_at_entry"], errors="coerce")
    sa  = pd.to_numeric(df["sum_assured"],  errors="coerce")

    # Old age at entry
    df = _add_flag(df,
        valid_mask & (age > 65),
        "_warnings", "HIGH_AGE_AT_ENTRY"
    )

    # Very large sum assured
    df = _add_flag(df,
        valid_mask & (sa > MAX_SA),
        "_warnings", "LARGE_SUM_ASSURED"
    )

    # Reinsured but no treaty ID
    ri_flag = df["reinsurance_flag"].astype(str).isin(["True", "true", "Y", "1"])
    no_treaty = df["ri_treaty_id"].isin(["", "nan", "None"])
    df = _add_flag(df,
        valid_mask & ri_flag & no_treaty,
        "_warnings", "RI_FLAG_NO_TREATY"
    )

    # Invalid gender
    df = _add_flag(df,
        valid_mask & ~df["gender"].isin(VALID_GENDERS),
        "_warnings", "INVALID_GENDER"
    )

    # Invalid smoker status
    df = _add_flag(df,
        valid_mask & ~df["smoker_status"].isin(VALID_SMOKERS),
        "_warnings", "INVALID_SMOKER_STATUS"
    )

    # Very high escalation rate
    esc = pd.to_numeric(df["escalation_rate"], errors="coerce")
    df = _add_flag(df,
        valid_mask & (esc > MAX_ESCALATION),
        "_warnings", "HIGH_ESCALATION_RATE"
    )

    # Premium appears zero (but not flagged as error — could be paid-up)
    prem = pd.to_numeric(df["premium_amount"], errors="coerce")
    df = _add_flag(df,
        valid_mask & (prem == 0) & (df["policy_status"] == "IF"),
        "_warnings", "ZERO_PREMIUM_INFORCE"
    )

    return df


def _check_tier3(df):
    """Anomalies — statistical outliers flagged for information only."""

    valid_mask = df["_errors"] == ""
    valid_df   = df[valid_mask]

    if len(valid_df) < 10:
        return df

    for field in ["sum_assured", "premium_amount", "age_at_entry"]:
        vals = pd.to_numeric(valid_df[field], errors="coerce").dropna()
        if len(vals) < 2:
            continue
        mean  = vals.mean()
        std   = vals.std()
        if std == 0:
            continue
        z_scores = (pd.to_numeric(df[field], errors="coerce") - mean) / std
        outliers = valid_mask & (z_scores.abs() > 3)
        df = _add_flag(df, outliers, "_anomalies", f"OUTLIER_{field.upper()}")

    return df


def _build_report(df_full, df_valid, df_errors):
    """Build a structured data quality report."""

    total      = len(df_full)
    n_valid    = len(df_valid)
    n_errors   = len(df_errors)
    n_warnings = (df_valid["_warnings"] != "").sum()
    n_anomalies= (df_valid["_anomalies"] != "").sum()

    # Error breakdown
    error_counts = {}
    if n_errors > 0:
        all_errors = df_errors["_errors"].str.split("; ").explode()
        error_counts = all_errors.value_counts().to_dict()

    # Warning breakdown
    warning_counts = {}
    if n_warnings > 0:
        all_warnings = df_valid.loc[
            df_valid["_warnings"] != "", "_warnings"
        ].str.split("; ").explode()
        warning_counts = all_warnings.value_counts().to_dict()

    report = {
        "summary": {
            "total_records"          : total,
            "passed_validation"      : n_valid,
            "failed_validation"      : n_errors,
            "pass_rate"              : n_valid / total if total > 0 else 0,
            "records_with_warnings"  : int(n_warnings),
            "records_with_anomalies" : int(n_anomalies),
        },
        "error_breakdown"   : error_counts,
        "warning_breakdown" : warning_counts,
    }

    # Print summary to console
    print(f"\n  ── Validation Report ──────────────────────")
    print(f"  Total records    : {total:>6,}")
    print(f"  Passed (Tier 1)  : {n_valid:>6,}  ({n_valid/total:.1%})")
    print(f"  Failed (Tier 1)  : {n_errors:>6,}  ({n_errors/total:.1%})")
    print(f"  Warnings (Tier 2): {n_warnings:>6,}")
    print(f"  Anomalies (Tier 3): {n_anomalies:>5,}")

    if error_counts:
        print(f"\n  Error breakdown:")
        for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"    {err:<35}: {cnt}")

    return report
