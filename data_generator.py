# data_generator.py
# ══════════════════════════════════════════════════════════════════════
# Synthetic Life Insurance Policy Generator
#
# Generates realistic but fictional policy portfolios in three formats:
# CSV, Excel, and JSON — each with deliberately messy column names,
# coded values, and inconsistent date formats that mirror real-world
# policy administration system exports.
#
# No real policyholder data is ever used in this project.
# All outputs are purely synthetic and statistically realistic.
# ══════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import json
import os
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import random
import warnings
warnings.filterwarnings("ignore")


# ── Product configuration ──────────────────────────────────────────────
# Each product has realistic parameter ranges drawn from South African
# retail life insurance market norms

PRODUCT_CONFIG = {
    "TERM": {
        "code"          : "TL01",
        "terms"         : [10, 15, 20, 25, 30],
        "min_age"       : 18,
        "max_age"       : 60,
        "min_sa"        : 100_000,
        "max_sa"        : 5_000_000,
        "prem_rate_range": (0.002, 0.008),   # Premium as fraction of SA
        "ri_threshold"  : 1_000_000,          # Reinsure above this SA
    },
    "WL": {
        "code"          : "WL01",
        "terms"         : [99],               # Whole life — no fixed expiry
        "min_age"       : 18,
        "max_age"       : 55,
        "min_sa"        : 50_000,
        "max_sa"        : 2_000_000,
        "prem_rate_range": (0.010, 0.025),
        "ri_threshold"  : 1_000_000,
    },
    "ENDOW": {
        "code"          : "EN01",
        "terms"         : [10, 15, 20, 25],
        "min_age"       : 18,
        "max_age"       : 55,
        "min_sa"        : 50_000,
        "max_sa"        : 3_000_000,
        "prem_rate_range": (0.030, 0.060),
        "ri_threshold"  : 2_000_000,
    },
}

# ── Value mappings that mimic real system codes ────────────────────────
# Real policy admin systems store coded values, not readable strings.
# Our ingestion layer must reverse these mappings.
GENDER_CODES    = {"M": "1", "F": "2"}
SMOKER_CODES    = {"S": "Y", "NS": "N"}
STATUS_CODES    = {"IF": "IF", "PU": "PU", "LA": "LA"}
FREQ_CODES      = {"MONTHLY": "M", "QUARTERLY": "Q", "ANNUAL": "A"}


def _random_date_of_birth(age_at_valuation, valuation_date):
    """
    Generate a realistic date of birth given an age.
    The birthday falls randomly within the year,
    making the exact age a decimal (more realistic).
    """
    approx_dob = valuation_date - relativedelta(years=age_at_valuation)
    day_offset  = random.randint(-182, 182)
    dob         = approx_dob + timedelta(days=day_offset)
    return dob


def _generate_single_policy(policy_num, valuation_date, product_mix, rng):
    """
    Generate one synthetic policy record.
    
    Returns a dictionary with deliberately messy field names and coded
    values — exactly what you would get from a LifePRO or FAST export.
    """
    # ── Select product ─────────────────────────────────────────────────
    product     = rng.choice(
        list(product_mix.keys()),
        p=list(product_mix.values())
    )
    cfg         = PRODUCT_CONFIG[product]

    # ── Demographics ───────────────────────────────────────────────────
    gender      = rng.choice(["M", "F"], p=[0.55, 0.45])
    smoker      = rng.choice(["S", "NS"], p=[0.20, 0.80])

    age_entry   = int(rng.normal(38, 9))
    age_entry   = max(cfg["min_age"], min(cfg["max_age"], age_entry))

    # ── Policy dates ───────────────────────────────────────────────────
    term        = int(rng.choice(cfg["terms"]))

    # Duration: how many years the policy has been in force
    max_duration = min(term - 1, 20) if term < 99 else 20
    duration    = random.randint(0, max_duration)

    inception   = valuation_date - relativedelta(years=duration)
    dob         = _random_date_of_birth(age_entry + duration, valuation_date)

    if term < 99:
        expiry  = inception + relativedelta(years=term)
    else:
        expiry  = None   # Whole life — no expiry

    # ── Financial terms ────────────────────────────────────────────────
    sa_raw      = rng.uniform(cfg["min_sa"], cfg["max_sa"])
    # Round to nearest R1,000 — realistic
    sa          = round(sa_raw / 1000) * 1000

    freq        = rng.choice(
        ["MONTHLY", "QUARTERLY", "ANNUAL"],
        p=[0.70, 0.10, 0.20]
    )
    freq_mult   = {"MONTHLY": 12, "QUARTERLY": 4, "ANNUAL": 1}[freq]

    prem_rate   = rng.uniform(*cfg["prem_rate_range"])
    annual_prem = sa * prem_rate
    prem_amount = round(annual_prem / freq_mult, 2)

    escalation  = rng.choice([0.0, 0.05, 0.10, 0.15], p=[0.20, 0.40, 0.30, 0.10])

    # ── Reinsurance ────────────────────────────────────────────────────
    reinsured   = sa > cfg["ri_threshold"]
    ri_retention = min(sa, cfg["ri_threshold"]) if reinsured else sa
    ri_ceded    = sa - ri_retention

    # ── Policy status ──────────────────────────────────────────────────
    # 85% in-force, 10% paid-up, 5% lapsed (but still in extract)
    status      = rng.choice(["IF", "PU", "LA"], p=[0.85, 0.10, 0.05])

    # ── Introduce deliberate data quality issues ───────────────────────
    # 2% of records will have issues — tests our validation engine
    has_issue   = rng.random() < 0.02
    if has_issue:
        issue_type = rng.choice(["neg_sa", "missing_dob", "bad_premium"])
        if issue_type == "neg_sa":
            sa = -abs(sa)
        elif issue_type == "missing_dob":
            dob = None
        elif issue_type == "bad_premium":
            prem_amount = 0.0

    # ── Build the raw record (messy format mimicking a real export) ────
    # Note: dates in DD/MM/YYYY, coded values, some nulls
    record = {
        "POL_NUM"       : f"POL{policy_num:07d}",
        "INSURER_CD"    : "ABCLIFE",
        "CLNT_DOB"      : dob.strftime("%d/%m/%Y") if dob else "",
        "CLNT_GENDER"   : GENDER_CODES[gender],
        "SMOKER_IND"    : SMOKER_CODES[smoker],
        "PROD_CD"       : cfg["code"],
        "POL_COMM_DT"   : inception.strftime("%d/%m/%Y"),
        "POL_EXPIRY_DT" : expiry.strftime("%d/%m/%Y") if expiry else "",
        "POL_TERM_YRS"  : term if term < 99 else "",
        "BASIC_SA"      : sa,
        "PREM_AMT"      : prem_amount,
        "PREM_FREQ_CD"  : FREQ_CODES[freq],
        "PREM_ESCL_RT"  : escalation,
        "PREM_PYMT_TERM": term if term < 99 else "",
        "REINSURED"     : "Y" if reinsured else "N",
        "RI_RETENTION"  : round(ri_retention, 2),
        "RI_CEDED"      : round(ri_ceded, 2),
        "RI_TREATY"     : f"TRT{rng.integers(1,5):02d}" if reinsured else "",
        "POL_STAT_CD"   : STATUS_CODES[status],
        "VAL_FLAG"      : "Y" if status == "IF" else "N",
    }

    return record


def generate_portfolio(
    n_policies      = 1000,
    product_mix     = None,
    valuation_date  = None,
    seed            = 42,
):
    """
    Generate a full synthetic policy portfolio.

    Parameters
    ──────────
    n_policies     : Number of policies to generate
    product_mix    : Dict of product weights e.g. {"TERM":0.5,"WL":0.3,"ENDOW":0.2}
    valuation_date : The valuation date for the portfolio
    seed           : Random seed for reproducibility

    Returns
    ───────
    pd.DataFrame with one row per policy in raw (messy) format
    """
    if product_mix is None:
        product_mix = {"TERM": 0.50, "WL": 0.30, "ENDOW": 0.20}

    if valuation_date is None:
        valuation_date = date(2024, 12, 31)

    # Validate product mix sums to 1
    total = sum(product_mix.values())
    assert abs(total - 1.0) < 1e-9, f"Product mix must sum to 1.0, got {total}"

    rng      = np.random.default_rng(seed)
    policies = []

    print(f"Generating {n_policies} synthetic policies...")
    print(f"  Product mix    : {product_mix}")
    print(f"  Valuation date : {valuation_date}")
    print(f"  Random seed    : {seed}")

    for i in range(n_policies):
        record = _generate_single_policy(i + 1, valuation_date, product_mix, rng)
        policies.append(record)

    df = pd.DataFrame(policies)
    print(f"  Generated      : {len(df)} records ✓")

    return df


def save_as_csv(df, path):
    """
    Save with pipe delimiter and header — mimics a LifePRO export.
    First row is a system timestamp line (deliberately awkward).
    """
    with open(path, "w") as f:
        f.write(f"# ABC Life LifePRO Export | Generated: 2024-12-31 | Records: {len(df)}\n")

    df.to_csv(path, sep="|", index=False, mode="a")
    print(f"  CSV saved  → {path}")


def save_as_excel(df, path):
    """
    Save as Excel with two header rows — common in actuarial extracts.
    Row 1: section labels (merged cells conceptually)
    Row 2: actual column names
    Row 3+: data
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Write a metadata sheet first
        meta = pd.DataFrame({
            "Parameter" : ["Source System", "Extract Date", "Policy Count", "Valuation Date"],
            "Value"     : ["ABC Life LifePRO", "2024-12-31", len(df), "2024-12-31"]
        })
        meta.to_excel(writer, sheet_name="Metadata", index=False)

        # Write the policy data
        df.to_excel(writer, sheet_name="Policies", index=False)

    print(f"  Excel saved → {path}")


def save_as_json(df, path):
    """
    Save as JSON records array — mimics a modern policy admin API response.
    Wraps the records in a realistic API response envelope.
    """
    payload = {
        "api_version"    : "v2.1",
        "source_system"  : "ABC Life Policy API",
        "extract_date"   : "2024-12-31",
        "valuation_date" : "2024-12-31",
        "record_count"   : len(df),
        "status"         : "SUCCESS",
        "policies"       : json.loads(df.to_json(orient="records"))
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"  JSON saved  → {path}")


if __name__ == "__main__":
    print("=" * 55)
    print("  IFRS 17 / SAM Engine — Synthetic Data Generator")
    print("=" * 55)

    # Generate the portfolio
    df = generate_portfolio(
        n_policies     = 1000,
        valuation_date = date(2024, 12, 31),
    )

    # Print summary statistics
    print(f"\nPortfolio Summary:")
    print(f"  Total policies    : {len(df)}")
    print(f"  Product breakdown :")
    prod_map = {"TL01": "Term", "WL01": "Whole Life", "EN01": "Endowment"}
    for code, name in prod_map.items():
        count = (df["PROD_CD"] == code).sum()
        print(f"    {name:15s}: {count:>5} ({count/len(df):.1%})")
    print(f"  Total sum assured : R{df['BASIC_SA'].sum():>20,.0f}")
    print(f"  Mean sum assured  : R{df['BASIC_SA'].mean():>20,.0f}")
    print(f"  In-force policies : {(df['POL_STAT_CD']=='IF').sum()}")

    # Save in all three formats
    print(f"\nSaving files...")
    os.makedirs("data/synthetic", exist_ok=True)

    save_as_csv(df,   "data/synthetic/abc_life_export.csv")
    save_as_excel(df, "data/synthetic/abc_life_export.xlsx")
    save_as_json(df,  "data/synthetic/abc_life_export.json")

    print(f"\n✓ All formats saved to data/synthetic/")
    print("  Run the ingestion pipeline next:")
    print("  python -c \"from ingestion.pipeline import run_pipeline; run_pipeline('data/synthetic/abc_life_export.csv', 'mappings/synthetic_data.yaml')\"")
