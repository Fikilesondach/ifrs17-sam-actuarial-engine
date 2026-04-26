# engine/projector.py
# ══════════════════════════════════════════════════════════════════════
# CASH FLOW PROJECTION ENGINE
#
# Projects the future cash flows of a single life insurance policy
# from the valuation date to policy expiry, using a year-by-year
# deterministic projection with decrements.
#
# Outputs:
#   1. Full projection table (one row per year)
#   2. Present value of future cash flows (net BEL)
#   3. Present value of premiums, claims, expenses separately
#
# This is the building block for:
#   - IFRS 17 Fulfilment Cash Flows (Phase 3)
#   - SAM Best Estimate Liability  (Phase 3)
#   - SAM SCR stress scenarios     (Phase 4)
# ══════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from dataclasses import dataclass
from engine.mortality   import MortalityTable
from engine.assumptions import AssumptionSet, build_default_assumptions


# ── Shared mortality table (loaded once, reused for all policies) ──────
_MORTALITY_TABLE = None

def get_mortality_table():
    global _MORTALITY_TABLE
    if _MORTALITY_TABLE is None:
        _MORTALITY_TABLE = MortalityTable()
    return _MORTALITY_TABLE


@dataclass
class PolicyInput:
    """
    Minimal policy data required for cash flow projection.

    This is extracted from the canonical DataFrame produced by
    the ingestion pipeline. One PolicyInput per policy per run.
    """
    policy_id:            str
    product_code:         str       # "TERM", "WL", "ENDOW"
    age_at_valuation:     float     # Exact age at valuation date
    gender:               str       # "M" or "F"
    smoker_status:        str       # "S" or "NS"
    sum_assured:          float     # Death benefit (ZAR)
    annualised_premium:   float     # Annual premium (ZAR)
    remaining_term_years: float     # Years of cover remaining
    premium_payment_term: float     # Years premiums still payable
    escalation_rate:      float     # Annual benefit/premium escalation
    policy_status:        str       # "IF", "PU" (paid-up)


@dataclass
class ProjectionResult:
    """
    Output of a single policy projection.

    Contains both the full year-by-year table and
    the summarised present values used in IFRS 17 and SAM.
    """
    policy_id:            str
    projection_table:     pd.DataFrame    # Full year-by-year cash flows
    pv_premiums:          float           # PV of future premium income
    pv_claims:            float           # PV of future death claims
    pv_expenses:          float           # PV of future expenses
    pv_net:               float           # PV net = claims + expenses - premiums
    bel:                  float           # Best Estimate Liability = PV net
    n_years_projected:    int
    age_at_valuation:     float
    assumptions_used:     str


def project_policy(
    policy:           PolicyInput,
    assumptions:      AssumptionSet = None,
    mortality_multiplier: float = 1.0,
) -> ProjectionResult:
    """
    Project cash flows for a single policy.

    Parameters
    ──────────
    policy               : PolicyInput dataclass
    assumptions          : AssumptionSet (defaults to best estimate)
    mortality_multiplier : Used for SAM mortality stress (1.0 = best estimate,
                           1.2 = 20% mortality stress)

    Returns
    ───────
    ProjectionResult containing full cash flow table and present values

    Projection mechanics:
    ─────────────────────
    We start with a unit cohort of 1.0 policy.

    At each year t:
      1. Look up qx for age (age_at_valuation + t)
      2. Apply mortality multiplier (for stress testing)
      3. Calculate deaths: d(t) = l(t) × qx × multiplier
      4. Calculate lapses: w(t) = [l(t) - d(t)] × lapse_rate(t)
      5. Survivors: l(t+1) = l(t) - d(t) - w(t)
      6. Cash flows:
           Premium   = l(t) × AP × (1+esc)^t   [start of year, so t-indexed]
           Claim     = d(t) × SA × (1+esc)^t   [end of year]
           Expense   = l(t) × per_policy + premium × per_premium_pct
      7. Discount each cash flow to time 0

    Paid-up policies:
      Premiums set to zero. Claims and survival benefits still project.
    """
    if assumptions is None:
        assumptions = build_default_assumptions()

    mt   = get_mortality_table()
    disc = assumptions.discount
    exp  = assumptions.expenses
    lap  = assumptions.lapse

    # ── Determine projection horizon ───────────────────────────────────
    # Number of future years to project (ceiling to integer)
    n_years = max(1, int(np.ceil(policy.remaining_term_years)))

    # For whole life, cap projection at age 100
    if policy.product_code == "WL":
        n_years = max(1, min(n_years, int(100 - policy.age_at_valuation)))

    # ── Initialise projection state ────────────────────────────────────
    l_t = 1.0    # In-force cohort — starts at 1.0 (unit policy)

    rows = []

    for t in range(n_years):
        # ── Current age this projection year ──────────────────────────
        age_t = policy.age_at_valuation + t

        # ── Mortality decrement ───────────────────────────────────────
        qx    = mt.get_qx(age_t, policy.gender, policy.smoker_status)
        qx   *= mortality_multiplier          # Apply stress if applicable
        qx    = min(qx, 1.0)

        deaths_t = l_t * qx

        # ── Lapse decrement ────────────────────────────────────────────
        # Lapses applied after deaths (UDD assumption)
        policy_year  = t + 1
        lapse_rate_t = lap.get_rate(policy_year)

        # Paid-up policies do not lapse (they have already stopped paying)
        if policy.policy_status == "PU":
            lapse_rate_t = 0.0

        lapses_t = (l_t - deaths_t) * lapse_rate_t

        # ── Survivors into next year ───────────────────────────────────
        l_next = l_t - deaths_t - lapses_t

        # ── Escalated benefit and premium ──────────────────────────────
        # Escalation applies from year 2 onwards (year 1 = base amounts)
        escalation_factor = (1 + policy.escalation_rate) ** t

        sa_t    = policy.sum_assured        * escalation_factor
        prem_t  = policy.annualised_premium * escalation_factor

        # Paid-up or past premium payment term → no premium income
        if policy.policy_status == "PU" or t >= policy.premium_payment_term:
            prem_t = 0.0

        # ── Cash flows (gross, before discounting) ─────────────────────
        # Premium: received at start of year (timing = t)
        cf_premium  = l_t * prem_t

        # Death claim: paid at end of year (timing = t + 1)
        cf_claim    = deaths_t * sa_t

        # Expenses: split into per-policy and per-premium components
        exp_per_pol = exp.inflated(exp.per_policy_annual, t)
        exp_per_prem= cf_premium * exp.per_premium_pct
        cf_expense  = l_t * exp_per_pol + exp_per_prem

        # ── Endowment maturity benefit ─────────────────────────────────
        # In the final year, survivors receive the maturity benefit
        cf_maturity = 0.0
        if policy.product_code == "ENDOW" and t == n_years - 1:
            cf_maturity = l_next * sa_t    # Survivors at end of final year

        # ── Discount factors ───────────────────────────────────────────
        # Premiums received start of year: discount by v(t)
        # Claims and expenses paid end of year: discount by v(t+1)
        v_start = disc.get_discount_factor(t)
        v_end   = disc.get_discount_factor(t + 1)

        # ── Present values ─────────────────────────────────────────────
        pv_prem    = cf_premium  * v_start
        pv_claim   = cf_claim    * v_end
        pv_expense = cf_expense  * v_end
        pv_maturity= cf_maturity * v_end

        # Net cash flow from insurer perspective:
        # Positive = outflow (claims + expenses)
        # Negative = inflow (premiums received)
        pv_net_t = pv_claim + pv_expense + pv_maturity - pv_prem

        # ── Store row ──────────────────────────────────────────────────
        rows.append({
            "year"            : t + 1,
            "age"             : round(age_t, 2),
            "qx"              : round(qx, 8),
            "l_t"             : round(l_t, 6),
            "deaths"          : round(deaths_t, 6),
            "lapses"          : round(lapses_t, 6),
            "l_next"          : round(l_next, 6),
            "cf_premium"      : round(cf_premium,  2),
            "cf_claim"        : round(cf_claim,    2),
            "cf_expense"      : round(cf_expense,  2),
            "cf_maturity"     : round(cf_maturity, 2),
            "cf_net"          : round(cf_claim + cf_expense + cf_maturity - cf_premium, 2),
            "discount_factor" : round(v_end, 6),
            "pv_premium"      : round(pv_prem,    2),
            "pv_claim"        : round(pv_claim,   2),
            "pv_expense"      : round(pv_expense, 2),
            "pv_maturity"     : round(pv_maturity,2),
            "pv_net"          : round(pv_net_t,   2),
        })

        # ── Advance cohort ─────────────────────────────────────────────
        l_t = l_next

        # Stop if cohort is effectively extinct
        if l_t < 1e-8:
            break

    # ── Assemble projection table ──────────────────────────────────────
    proj_df = pd.DataFrame(rows)

    # ── Aggregate present values ───────────────────────────────────────
    pv_premiums = proj_df["pv_premium"].sum()
    pv_claims   = proj_df["pv_claim"].sum()
    pv_expenses = proj_df["pv_expense"].sum()
    pv_maturity = proj_df["pv_maturity"].sum()
    pv_net      = proj_df["pv_net"].sum()

    # BEL = PV(outflows) - PV(inflows)
    # Positive BEL = insurer owes more than it receives → liability
    # Negative BEL = insurer receives more than it owes → asset (unusual)
    bel = pv_net

    return ProjectionResult(
        policy_id          = policy.policy_id,
        projection_table   = proj_df,
        pv_premiums        = round(pv_premiums, 2),
        pv_claims          = round(pv_claims,   2),
        pv_expenses        = round(pv_expenses, 2),
        pv_net             = round(pv_net,      2),
        bel                = round(bel,         2),
        n_years_projected  = len(proj_df),
        age_at_valuation   = policy.age_at_valuation,
        assumptions_used   = assumptions.name,
    )


def project_portfolio(canonical_df, assumptions=None, verbose=True):
    """
    Project cash flows for every in-force policy in the canonical DataFrame.

    Loops through all valid in-force policies and calls project_policy()
    for each one. Returns a summary DataFrame with BEL per policy
    and a full results list for detailed analysis.

    Parameters
    ──────────
    canonical_df : Output of ingestion pipeline (validated canonical records)
    assumptions  : AssumptionSet (defaults to best estimate)
    verbose      : Print progress

    Returns
    ───────
    summary_df   : One row per policy — BEL, PV components
    results_list : List of ProjectionResult objects (full detail)
    """
    if assumptions is None:
        assumptions = build_default_assumptions()

    # Only project in-force policies
    inforce = canonical_df[canonical_df["policy_status"] == "IF"].copy()

    if verbose:
        print(f"\nProjecting {len(inforce):,} in-force policies...")
        print(f"  Assumption set: {assumptions.name}")

    summary_rows = []
    results_list = []
    errors       = []

    for idx, row in inforce.iterrows():
        try:
            policy = PolicyInput(
                policy_id            = str(row["policy_id"]),
                product_code         = str(row["product_code"]),
                age_at_valuation     = float(row.get("age_at_valuation", 40)),
                gender               = str(row.get("gender", "M")),
                smoker_status        = str(row.get("smoker_status", "NS")),
                sum_assured          = float(row.get("sum_assured", 0)),
                annualised_premium   = float(row.get("annualised_premium", 0)),
                remaining_term_years = float(row.get("remaining_term_years", 10)),
                premium_payment_term = float(row.get("remaining_term_years", 10)),
                escalation_rate      = float(row.get("escalation_rate", 0)),
                policy_status        = str(row.get("policy_status", "IF")),
            )

            result = project_policy(policy, assumptions)
            results_list.append(result)

            summary_rows.append({
                "policy_id"        : result.policy_id,
                "product_code"     : policy.product_code,
                "age_at_valuation" : round(policy.age_at_valuation, 1),
                "gender"           : policy.gender,
                "smoker_status"    : policy.smoker_status,
                "sum_assured"      : policy.sum_assured,
                "annualised_premium": policy.annualised_premium,
                "remaining_term"   : round(policy.remaining_term_years, 1),
                "pv_premiums"      : result.pv_premiums,
                "pv_claims"        : result.pv_claims,
                "pv_expenses"      : result.pv_expenses,
                "bel"              : result.bel,
                "n_years"          : result.n_years_projected,
            })

        except Exception as e:
            errors.append({"policy_id": row.get("policy_id"), "error": str(e)})

    summary_df = pd.DataFrame(summary_rows)

    if verbose:
        print(f"\n  Projection complete:")
        print(f"  Policies projected   : {len(summary_rows):,}")
        print(f"  Projection errors    : {len(errors):,}")
        print(f"  Total BEL            : R{summary_df['bel'].sum():>20,.2f}")
        print(f"  Total PV Premiums    : R{summary_df['pv_premiums'].sum():>20,.2f}")
        print(f"  Total PV Claims      : R{summary_df['pv_claims'].sum():>20,.2f}")
        print(f"  Total PV Expenses    : R{summary_df['pv_expenses'].sum():>20,.2f}")
        print(f"  Mean BEL per policy  : R{summary_df['bel'].mean():>20,.2f}")

    return summary_df, results_list
