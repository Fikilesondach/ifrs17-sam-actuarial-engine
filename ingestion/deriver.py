# ingestion/deriver.py
# ══════════════════════════════════════════════════════════════════════
# FIELD DERIVER — Step 3 of the ingestion pipeline
#
# Computes all derived fields that the valuation engine needs but that
# do not exist directly in the source data.
#
# Examples:
#   age_at_entry     = derived from date_of_birth + inception_date
#   annualised_premium = derived from premium_amount + frequency
#   cohort_id        = derived from inception_date (IFRS 17 annual cohort)
# ══════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta


# Frequency to annual multiplier
FREQ_MULTIPLIER = {
    "MONTHLY"  : 12,
    "QUARTERLY": 4,
    "ANNUAL"   : 1,
}


def derive_all_fields(df, config, valuation_date=None):
    """
    Compute all derived fields and add them to the DataFrame.

    Parameters
    ──────────
    df             : Mapped DataFrame from mapper.py
    config         : Parsed YAML mapping configuration
    valuation_date : Override valuation date (date object)
                     If None, uses config value

    Returns
    ───────
    pd.DataFrame with all derived fields added
    """
    df = df.copy()

    # ── Resolve valuation date ─────────────────────────────────────────
    if valuation_date is None:
        val_date_str  = config.get("valuation_date", str(date.today()))
        valuation_date = pd.to_datetime(val_date_str).date()

    date_fmt = config.get("date_format", "%Y-%m-%d")

    print(f"  Valuation date : {valuation_date}")

    # ── Step 1: Parse date columns ─────────────────────────────────────
    df = _parse_dates(df, date_fmt)

    # ── Step 2: Derive age fields ──────────────────────────────────────
    df = _derive_ages(df, valuation_date)

    # ── Step 3: Derive policy duration fields ──────────────────────────
    df = _derive_durations(df, valuation_date)

    # ── Step 4: Derive financial fields ───────────────────────────────
    df = _derive_financials(df)

    # ── Step 5: Derive IFRS 17 classification fields ───────────────────
    df = _derive_ifrs17_fields(df, valuation_date)

    # ── Step 6: Apply constant fields from config ──────────────────────
    df = _apply_constants(df, config)

    # ── Step 7: Add valuation date as a column ─────────────────────────
    df["valuation_date"] = valuation_date

    print(f"  Derived fields computed ✓")
    return df


def _parse_dates(df, date_fmt):
    """Convert date string columns to datetime objects."""
    date_cols = ["date_of_birth", "inception_date", "expiry_date"]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                format  = date_fmt,
                errors  = "coerce"    # Invalid dates become NaT
            )
    return df


def _derive_ages(df, valuation_date):
    """
    Compute age_at_entry and age_at_valuation.

    Age is computed as the number of complete years between
    the relevant date and the date of birth — using exact
    relativedelta to handle leap years correctly.
    """
    val_dt = pd.Timestamp(valuation_date)

    def exact_age(dob, ref_date):
        """Exact age in years (including decimal) at ref_date."""
        if pd.isnull(dob) or pd.isnull(ref_date):
            return np.nan
        try:
            rd    = relativedelta(ref_date.date(), dob.date())
            years = rd.years + rd.months/12 + rd.days/365.25
            return round(years, 4)
        except Exception:
            return np.nan

    if "date_of_birth" in df.columns and "inception_date" in df.columns:
        df["age_at_entry"] = df.apply(
            lambda r: exact_age(r["date_of_birth"], r["inception_date"]), axis=1
        )

    if "date_of_birth" in df.columns:
        df["age_at_valuation"] = df.apply(
            lambda r: exact_age(r["date_of_birth"], val_dt), axis=1
        )

    return df


def _derive_durations(df, valuation_date):
    """
    Compute policy_duration_years and remaining_term_years.

    policy_duration_years = time since inception (how long in force)
    remaining_term_years  = time until expiry (how long left)

    For whole life policies with no expiry_date, remaining_term is
    set to a large number (99 - current_age) to represent indefinite cover.
    """
    val_dt = pd.Timestamp(valuation_date)

    def years_between(d1, d2):
        if pd.isnull(d1) or pd.isnull(d2):
            return np.nan
        try:
            rd = relativedelta(d2.date(), d1.date())
            return round(rd.years + rd.months/12 + rd.days/365.25, 4)
        except Exception:
            return np.nan

    if "inception_date" in df.columns:
        df["policy_duration_years"] = df["inception_date"].apply(
            lambda d: years_between(d, val_dt)
        )

    if "expiry_date" in df.columns:
        df["remaining_term_years"] = df.apply(
            lambda r: years_between(val_dt, r["expiry_date"])
            if pd.notnull(r["expiry_date"])
            else max(0, 99 - (r.get("age_at_valuation") or 60)),
            axis=1
        )

    return df


def _derive_financials(df):
    """
    Compute annualised_premium from premium_amount and frequency.

    annualised_premium = premium_amount × frequency_multiplier
    e.g. R450/month × 12 = R5,400/year
    """
    if "premium_amount" in df.columns and "premium_frequency" in df.columns:

        df["premium_amount"] = pd.to_numeric(df["premium_amount"], errors="coerce")

        df["annualised_premium"] = df.apply(
            lambda r: (
                r["premium_amount"] * FREQ_MULTIPLIER.get(r["premium_frequency"], 1)
                if pd.notnull(r["premium_amount"])
                else np.nan
            ),
            axis=1
        ).round(2)

    # Convert sum_assured to numeric
    if "sum_assured" in df.columns:
        df["sum_assured"] = pd.to_numeric(df["sum_assured"], errors="coerce")

    # Convert escalation_rate to numeric
    if "escalation_rate" in df.columns:
        df["escalation_rate"] = pd.to_numeric(df["escalation_rate"], errors="coerce")

    return df


def _derive_ifrs17_fields(df, valuation_date):
    """
    Compute IFRS 17 portfolio and cohort identifiers.

    IFRS 17 requires grouping contracts into:
      - Portfolios  : contracts with similar risks managed together
                      We use product_code as the portfolio basis.
      - Cohorts     : annual groups of contracts issued in the same year
                      IFRS 17 prohibits mixing contracts from different
                      annual cohorts within the same group.
      - Risk groups : onerous / not onerous / no significant risk of becoming onerous

    cohort_id format: "{product_code}-{inception_year}"
    e.g. "TERM-2022", "WL-2019"
    """
    if "product_code" in df.columns and "inception_date" in df.columns:
        df["cohort_id"] = df.apply(
            lambda r: (
                f"{r['product_code']}-{r['inception_date'].year}"
                if pd.notnull(r.get("inception_date"))
                else "UNKNOWN"
            ),
            axis=1
        )

        df["portfolio_id"] = df.apply(
            lambda r: f"PORT-{r['product_code']}-{r.get('lob','LIFE')}",
            axis=1
        )

    return df


def _apply_constants(df, config):
    """Apply constant derived fields from YAML configuration."""
    derived = config.get("derived_fields", {})

    for field, spec in derived.items():
        if isinstance(spec, dict) and spec.get("method") == "constant":
            if field not in df.columns or df[field].eq("").all():
                df[field] = spec["value"]

    return df
