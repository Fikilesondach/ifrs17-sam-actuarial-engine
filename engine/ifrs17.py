# engine/ifrs17.py
# ══════════════════════════════════════════════════════════════════════
# IFRS 17 GENERAL MEASUREMENT MODEL (GMM)
#
# Implements the Building Block Approach for life insurance contracts:
#
#   Block 1: Fulfilment Cash Flows (FCF)
#            FCF = PV(Future Cash Flows) + Risk Adjustment
#
#   Block 2: Contractual Service Margin (CSM)
#            Unearned profit at inception, released over coverage period
#
#   Output:  Insurance Contract Liability (ICL) = FCF + CSM
#
# Also produces:
#   - CSM roll-forward schedule (how profit emerges year by year)
#   - P&L emergence (insurance service result per year)
#   - Liability for Remaining Coverage (LRC) breakdown
#   - Liability for Incurred Claims (LIC) — simplified for Phase 3
#
# References:
#   IFRS 17 Insurance Contracts (IASB, 2017, amended 2020)
#   IFRS 17 Implementation Guidance (IASB)
#   ASSA IFRS 17 Guidance Notes (Actuarial Society of SA)
# ══════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from engine.projector   import ProjectionResult, PolicyInput, project_policy
from engine.assumptions import AssumptionSet, build_default_assumptions


# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class RiskAdjustmentResult:
    """
    Risk Adjustment calculation output.

    The RA is the compensation the insurer requires for bearing
    the non-financial risk that actual experience differs from
    the best estimate (BEL).

    We implement the confidence interval approach:
    RA = VaR(q) - BEL
    where VaR(q) is the q-th percentile of the loss distribution.

    For a simplified implementation we approximate this using:
    RA = BEL × ra_loading_factor
    where the loading factor is calibrated to produce results
    consistent with a 75th percentile confidence interval.
    """
    bel:                float   # Best Estimate Liability
    ra_amount:          float   # Risk Adjustment amount
    confidence_level:   float   # Target confidence level (e.g. 0.75)
    ra_pct_of_bel:      float   # RA as percentage of BEL
    method:             str     # "confidence_interval" or "cost_of_capital"


@dataclass
class CSMResult:
    """
    Contractual Service Margin at inception and roll-forward.
    """
    policy_id:              str
    is_onerous:             bool        # True if contract is loss-making
    fcf_at_inception:       float       # FCF = BEL + RA at inception
    csm_at_inception:       float       # CSM = max(0, -FCF) at inception
    onerous_loss:           float       # Loss recognised if contract onerous
    locked_in_rate:         float       # Discount rate locked in at inception
    coverage_units_total:   float       # Total coverage units (denominator)
    roll_forward:           pd.DataFrame  # Year-by-year CSM movement


@dataclass
class IFRS17Result:
    """
    Complete IFRS 17 GMM valuation output for a single policy.

    This is the top-level result object combining all three blocks.
    """
    policy_id:          str
    product_code:       str
    valuation_date:     str

    # ── Block 1: Fulfilment Cash Flows ─────────────────────────────────
    bel:                float       # Best Estimate Liability
    risk_adjustment:    float       # Risk Adjustment
    fcf:                float       # FCF = BEL + RA

    # ── Block 2: CSM ───────────────────────────────────────────────────
    csm:                float       # CSM at valuation date
    is_onerous:         bool        # Contract is onerous?
    onerous_loss:       float       # Loss component if onerous

    # ── Total liability ────────────────────────────────────────────────
    icl:                float       # Insurance Contract Liability = FCF + CSM

    # ── Detailed sub-results ───────────────────────────────────────────
    ra_result:          RiskAdjustmentResult
    csm_result:         CSMResult
    projection:         ProjectionResult     # Underlying cash flows

    # ── Annual P&L emergence (CSM release schedule) ────────────────────
    pnl_schedule:       pd.DataFrame


# ══════════════════════════════════════════════════════════════════════
# RISK ADJUSTMENT CALCULATOR
# ══════════════════════════════════════════════════════════════════════

def calculate_risk_adjustment(
    projection:         ProjectionResult,
    assumptions:        AssumptionSet,
    product_code:       str = "TERM",
) -> RiskAdjustmentResult:
    """
    Calculate the IFRS 17 Risk Adjustment.

    The RA represents the compensation the insurer requires for
    bearing non-financial uncertainty. IFRS 17 does not prescribe
    the method — common approaches are:
      1. Confidence interval (Value at Risk)
      2. Cost of capital
      3. Conditional tail expectation (CTE)

    We implement a simplified confidence interval approach where
    the RA loading factor is product-specific and calibrated to
    approximately achieve the target confidence level.

    Loading factors by product (approximate, based on SA experience):
      TERM   : 12-18% of BEL  (pure mortality risk — moderately uncertain)
      WL     : 10-15% of BEL  (longer duration but more stable)
      ENDOW  : 8-12% of BEL   (lower mortality risk, higher lapse risk)

    In a full implementation this would involve stochastic simulation
    to derive the actual percentile of the loss distribution.
    """
    bel         = projection.bel
    conf_level  = assumptions.ra_confidence_level

    # ── Product-specific RA loading factors ────────────────────────────
    # These represent the RA as a percentage of the absolute BEL
    # Calibrated to approximately achieve a 75th percentile confidence
    base_loadings = {
        "TERM"  : 0.150,   # 15% of |BEL|
        "WL"    : 0.125,   # 12.5% of |BEL|
        "ENDOW" : 0.100,   # 10% of |BEL|
    }
    base_loading = base_loadings.get(product_code, 0.130)

    # ── Adjust for confidence level ────────────────────────────────────
    # Scale from 75th percentile base to actual target confidence
    # Using a simple linear interpolation
    # At 75th percentile: use base loading
    # At 90th percentile: use 2× base loading
    # At 50th percentile: use 0.5× base loading
    conf_scalar = (conf_level - 0.50) / (0.75 - 0.50)
    conf_scalar = max(0.5, min(2.5, conf_scalar))
    loading     = base_loading * conf_scalar

    # ── Duration adjustment ────────────────────────────────────────────
    # Longer duration contracts have proportionally more uncertainty
    # RA scales slightly with remaining projection years
    n_years       = projection.n_years_projected
    duration_adj  = 1.0 + 0.005 * max(0, n_years - 5)   # +0.5% per year beyond 5
    duration_adj  = min(duration_adj, 1.30)               # Cap at 30% uplift

    loading *= duration_adj

    # ── Compute RA ─────────────────────────────────────────────────────
    # RA is always positive (it is a compensation for risk)
    # We apply it to the absolute value of BEL to handle negative BEL cases
    bel_for_ra = abs(bel)
    ra_amount  = round(bel_for_ra * loading, 2)

    return RiskAdjustmentResult(
        bel              = bel,
        ra_amount        = ra_amount,
        confidence_level = conf_level,
        ra_pct_of_bel    = round(loading * 100, 2),
        method           = "confidence_interval_simplified",
    )


# ══════════════════════════════════════════════════════════════════════
# COVERAGE UNITS
# ══════════════════════════════════════════════════════════════════════

def calculate_coverage_units(projection: ProjectionResult) -> list:
    """
    Calculate coverage units for each projection year.

    Coverage units represent the quantum of service provided each year.
    They are used to determine the CSM release pattern — how much of
    the unearned profit is recognised as revenue each year.

    IFRS 17 paragraph B119: coverage units reflect the quantity of
    benefits provided and the expected duration of coverage.

    For life insurance we use the expected in-force sum assured
    weighted by survival probability:

        coverage_unit(t) = l(t) × SA(t)

    This means the CSM releases faster in early years (more policies
    in force, larger expected benefit exposure) and slower later
    (cohort has reduced through mortality and lapse).

    Returns list of coverage units, one per projection year.
    """
    proj = projection.projection_table

    if "l_t" in proj.columns and "cf_claim" in proj.columns:
        # Expected in-force exposure weighted by sum assured
        # cf_claim / deaths = SA per death, l_t = in-force cohort
        coverage_units = (proj["l_t"] * proj["cf_claim"].where(
            proj["deaths"] > 0,
            proj["l_t"]    # Fallback when no deaths
        )).fillna(proj["l_t"]).tolist()
    else:
        # Fallback: equal coverage units (straight-line release)
        n = len(proj)
        coverage_units = [1.0 / n] * n

    return coverage_units


# ══════════════════════════════════════════════════════════════════════
# CSM CALCULATOR
# ══════════════════════════════════════════════════════════════════════

def calculate_csm(
    projection:     ProjectionResult,
    ra_result:      RiskAdjustmentResult,
    assumptions:    AssumptionSet,
    valuation_date: str = "2024-12-31",
    inception_bel:  Optional[float] = None,
    inception_ra:   Optional[float] = None,
) -> CSMResult:
    """
    Calculate the Contractual Service Margin.

    The CSM is calculated at policy inception using LOCKED-IN assumptions.
    For policies already in force at the valuation date, we approximate
    the inception CSM by using current projection results.

    In a full production system:
      - CSM at inception is calculated once when the policy is issued
      - It is then rolled forward each quarter using the CSM roll-forward
      - Locked-in discount rate is the rate at policy inception date
      - Current economic assumptions are used for FCF updates (no CSM unlock)

    For Phase 3 we calculate CSM at the valuation date as a reasonable
    approximation. Phase 4 will introduce proper locked-in mechanics.

    Parameters
    ──────────
    inception_bel : BEL at inception (if known). If None, uses current BEL
                    as approximation for new business.
    inception_ra  : RA at inception (if known). If None, uses current RA.
    """
    # ── Locked-in discount rate ────────────────────────────────────────
    # For new business, this is the current risk-free rate
    # For in-force business, this was locked in at inception
    # We use the flat rate as approximation
    locked_in_rate = assumptions.discount.flat_rate

    # ── FCF at inception ───────────────────────────────────────────────
    bel_inception = inception_bel if inception_bel is not None else projection.bel
    ra_inception  = inception_ra  if inception_ra  is not None else ra_result.ra_amount
    fcf_inception = bel_inception + ra_inception

    # ── Determine if contract is onerous ──────────────────────────────
    is_onerous   = fcf_inception > 0

    if is_onerous:
        # Loss-making contract — CSM is zero, loss goes to P&L immediately
        csm_at_inception = 0.0
        onerous_loss     = fcf_inception    # Recognised immediately
    else:
        # Profitable contract — CSM captures the expected profit
        csm_at_inception = -fcf_inception   # Positive CSM = positive number
        onerous_loss     = 0.0

    # ── Coverage units ─────────────────────────────────────────────────
    coverage_units      = calculate_coverage_units(projection)
    total_coverage      = sum(coverage_units)

    # ── CSM Roll-Forward ───────────────────────────────────────────────
    roll_forward = _build_csm_roll_forward(
        csm_at_inception = csm_at_inception,
        coverage_units   = coverage_units,
        locked_in_rate   = locked_in_rate,
        projection       = projection,
    )

    return CSMResult(
        policy_id            = projection.policy_id,
        is_onerous           = is_onerous,
        fcf_at_inception     = round(fcf_inception,    2),
        csm_at_inception     = round(csm_at_inception, 2),
        onerous_loss         = round(onerous_loss,     2),
        locked_in_rate       = locked_in_rate,
        coverage_units_total = round(total_coverage,   4),
        roll_forward         = roll_forward,
    )


def _build_csm_roll_forward(
    csm_at_inception: float,
    coverage_units:   list,
    locked_in_rate:   float,
    projection:       ProjectionResult,
) -> pd.DataFrame:
    """
    Build the year-by-year CSM roll-forward table.

    The CSM roll-forward is one of the most important disclosures
    under IFRS 17. It shows how the unearned profit evolves each year.

    CSM Roll-Forward (per IFRS 17 paragraph 100):

    Opening CSM
    + Interest accretion        (CSM × locked-in rate)
    - CSM released to P&L      (service provided this year)
    ± Changes from assumptions  (non-economic assumption changes)
    = Closing CSM

    Simplified for Phase 3:
    We project the full roll-forward from inception to expiry.
    Assumption change effects are excluded (Phase 4 enhancement).
    """
    total_coverage = sum(coverage_units)
    n_years        = len(coverage_units)

    if total_coverage == 0:
        total_coverage = 1.0

    rows          = []
    csm_opening   = csm_at_inception
    proj_table    = projection.projection_table

    for t in range(n_years):
        # ── Interest accretion ─────────────────────────────────────────
        # CSM grows at the locked-in discount rate each year
        interest = csm_opening * locked_in_rate

        csm_before_release = csm_opening + interest

        # ── CSM release ────────────────────────────────────────────────
        # Release proportion = coverage units this year / remaining coverage
        remaining_coverage = sum(coverage_units[t:])

        if remaining_coverage > 0:
            release_pct = coverage_units[t] / remaining_coverage
        else:
            release_pct = 1.0

        csm_release = csm_before_release * release_pct

        # ── Closing CSM ────────────────────────────────────────────────
        csm_closing = csm_before_release - csm_release
        csm_closing = max(0.0, csm_closing)   # CSM cannot go below zero

        # ── Insurance Service Result (P&L this year) ───────────────────
        # ISR = Insurance Revenue - Insurance Service Expense
        # Simplified: Revenue ≈ expected claims + expenses + CSM release
        if t < len(proj_table):
            pv_row       = proj_table.iloc[t]
            exp_claims   = pv_row.get("cf_claim",   0)
            exp_expenses = pv_row.get("cf_expense", 0)
            exp_premium  = pv_row.get("cf_premium", 0)
        else:
            exp_claims = exp_expenses = exp_premium = 0

        # Insurance revenue = expected claims + expenses + CSM release
        # (the premium is not revenue under IFRS 17 — it is a deposit)
        insurance_revenue = exp_claims + exp_expenses + csm_release

        # Insurance service expense = expected claims + expenses
        ins_service_expense = exp_claims + exp_expenses

        # Insurance service result (profit this year from writing insurance)
        isr = csm_release    # Simplified — equals CSM released

        rows.append({
            "year"                  : t + 1,
            "csm_opening"           : round(csm_opening,          2),
            "interest_accretion"    : round(interest,             2),
            "csm_before_release"    : round(csm_before_release,   2),
            "coverage_units"        : round(coverage_units[t],    6),
            "release_percentage"    : round(release_pct * 100,    4),
            "csm_release"           : round(csm_release,          2),
            "csm_closing"           : round(csm_closing,          2),
            "insurance_revenue"     : round(insurance_revenue,    2),
            "ins_service_expense"   : round(ins_service_expense,  2),
            "insurance_service_result": round(isr,                2),
        })

        csm_opening = csm_closing

        if csm_closing < 0.01:
            break

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# MAIN IFRS 17 VALUATION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def value_policy_ifrs17(
    policy:         PolicyInput,
    assumptions:    AssumptionSet = None,
    valuation_date: str = "2024-12-31",
) -> IFRS17Result:
    """
    Full IFRS 17 GMM valuation for a single policy.

    Steps:
        1. Project cash flows (BEL)
        2. Calculate Risk Adjustment
        3. Compute FCF = BEL + RA
        4. Calculate CSM (and check for onerousness)
        5. Compute ICL = FCF + CSM
        6. Build P&L emergence schedule

    Parameters
    ──────────
    policy         : PolicyInput from ingestion pipeline
    assumptions    : AssumptionSet (defaults to best estimate)
    valuation_date : String date "YYYY-MM-DD"

    Returns
    ───────
    IFRS17Result with all components computed
    """
    if assumptions is None:
        assumptions = build_default_assumptions()

    # ── Step 1: Project cash flows ─────────────────────────────────────
    projection = project_policy(policy, assumptions)

    # ── Step 2: Risk Adjustment ────────────────────────────────────────
    ra_result  = calculate_risk_adjustment(projection, assumptions, policy.product_code)

    # ── Step 3: Fulfilment Cash Flows ──────────────────────────────────
    # FCF = PV of future net cash flows + Risk Adjustment
    # FCF > 0 → insurer has net obligation (liability)
    # FCF < 0 → insurer expects net profit (offset by CSM)
    fcf = projection.bel + ra_result.ra_amount

    # ── Step 4: Contractual Service Margin ────────────────────────────
    csm_result = calculate_csm(
        projection     = projection,
        ra_result      = ra_result,
        assumptions    = assumptions,
        valuation_date = valuation_date,
    )

    # ── Step 5: Insurance Contract Liability ───────────────────────────
    # ICL = FCF + CSM
    # For non-onerous contracts:  FCF < 0  and  CSM > 0
    #   ICL = negative FCF + positive CSM = small positive or zero
    # For onerous contracts:      FCF > 0  and  CSM = 0
    #   ICL = FCF (loss recognised immediately)
    icl = fcf + csm_result.csm_at_inception

    # ── Step 6: Build P&L schedule ────────────────────────────────────
    pnl_schedule = _build_pnl_schedule(
        projection  = projection,
        csm_result  = csm_result,
        ra_result   = ra_result,
    )

    return IFRS17Result(
        policy_id       = policy.policy_id,
        product_code    = policy.product_code,
        valuation_date  = valuation_date,
        bel             = projection.bel,
        risk_adjustment = ra_result.ra_amount,
        fcf             = round(fcf, 2),
        csm             = csm_result.csm_at_inception,
        is_onerous      = csm_result.is_onerous,
        onerous_loss    = csm_result.onerous_loss,
        icl             = round(icl, 2),
        ra_result       = ra_result,
        csm_result      = csm_result,
        projection      = projection,
        pnl_schedule    = pnl_schedule,
    )


def _build_pnl_schedule(
    projection:  ProjectionResult,
    csm_result:  CSMResult,
    ra_result:   RiskAdjustmentResult,
) -> pd.DataFrame:
    """
    Build the annual P&L emergence schedule.

    Shows how the insurance profit emerges over the policy lifetime.
    This is what the CFO presents to the board and analysts.

    Under IFRS 17 the income statement shows:
      Insurance Revenue          (not premiums — IFRS 17 is service-based)
      Insurance Service Expenses (claims + expenses)
      Insurance Service Result   (underwriting profit)
      Finance Income/Expense     (unwinding of discount)
    """
    proj_df = projection.projection_table
    rf_df   = csm_result.roll_forward

    rows = []
    n    = min(len(proj_df), len(rf_df))

    for t in range(n):
        proj_row = proj_df.iloc[t]
        rf_row   = rf_df.iloc[t]

        # Under IFRS 17, revenue ≠ premium collected
        # Revenue = expected claims + expenses + CSM release
        # This represents the service provided this period
        revenue   = rf_row["insurance_revenue"]
        expense   = rf_row["ins_service_expense"]
        isr       = rf_row["insurance_service_result"]

        # Finance expense = unwinding of discount on liabilities
        # Approximation: interest on opening BEL at locked-in rate
        finance_expense = abs(proj_row.get("pv_claim", 0)) * csm_result.locked_in_rate

        rows.append({
            "year"                      : int(proj_row["year"]),
            "insurance_revenue"         : round(revenue,         2),
            "insurance_service_expense" : round(expense,         2),
            "insurance_service_result"  : round(isr,             2),
            "finance_expense"           : round(finance_expense, 2),
            "total_comprehensive_income": round(isr - finance_expense, 2),
            "csm_closing"               : round(rf_row["csm_closing"], 2),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# PORTFOLIO VALUATION
# ══════════════════════════════════════════════════════════════════════

def value_portfolio_ifrs17(
    canonical_df:   pd.DataFrame,
    assumptions:    AssumptionSet = None,
    valuation_date: str = "2024-12-31",
    verbose:        bool = True,
) -> tuple:
    """
    Run full IFRS 17 GMM valuation for all in-force policies.

    Returns
    ───────
    summary_df   : One row per policy with all IFRS 17 components
    results_list : Full IFRS17Result objects for detailed analysis
    portfolio_summary : Aggregated portfolio-level results
    """
    if assumptions is None:
        assumptions = build_default_assumptions()

    inforce = canonical_df[canonical_df["policy_status"] == "IF"].copy()

    if verbose:
        print(f"\nRunning IFRS 17 GMM valuation...")
        print(f"  Policies in scope : {len(inforce):,}")
        print(f"  Valuation date    : {valuation_date}")
        print(f"  Assumption set    : {assumptions.name}")
        print(f"  Confidence level  : {assumptions.ra_confidence_level:.0%}")

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

            result = value_policy_ifrs17(policy, assumptions, valuation_date)
            results_list.append(result)

            summary_rows.append({
                "policy_id"         : result.policy_id,
                "product_code"      : result.product_code,
                "age_at_valuation"  : round(policy.age_at_valuation, 1),
                "gender"            : policy.gender,
                "sum_assured"       : policy.sum_assured,
                "remaining_term"    : round(policy.remaining_term_years, 1),
                # IFRS 17 Components
                "bel"               : result.bel,
                "risk_adjustment"   : result.risk_adjustment,
                "fcf"               : result.fcf,
                "csm"               : result.csm,
                "icl"               : result.icl,
                "is_onerous"        : result.is_onerous,
                "onerous_loss"      : result.onerous_loss,
                # Ratios
                "ra_pct_bel"        : result.ra_result.ra_pct_of_bel,
                "csm_pct_icl"       : (
                    result.csm / result.icl * 100
                    if result.icl != 0 else 0
                ),
            })

        except Exception as e:
            errors.append({
                "policy_id": row.get("policy_id", "UNKNOWN"),
                "error"    : str(e)
            })

    summary_df = pd.DataFrame(summary_rows)

    # ── Portfolio-level aggregates ──────────────────────────────────────
    portfolio_summary = _build_portfolio_summary(summary_df, errors)

    if verbose:
        _print_portfolio_summary(portfolio_summary)

    return summary_df, results_list, portfolio_summary


def _build_portfolio_summary(summary_df, errors):
    """Aggregate to portfolio level by product."""
    if summary_df.empty:
        return {}

    by_product = summary_df.groupby("product_code").agg(
        count         = ("policy_id",       "count"),
        total_bel     = ("bel",             "sum"),
        total_ra      = ("risk_adjustment", "sum"),
        total_fcf     = ("fcf",             "sum"),
        total_csm     = ("csm",             "sum"),
        total_icl     = ("icl",             "sum"),
        mean_icl      = ("icl",             "mean"),
        onerous_count = ("is_onerous",      "sum"),
        total_onerous_loss = ("onerous_loss","sum"),
    ).round(2)

    totals = {
        "total_policies"    : len(summary_df),
        "valuation_errors"  : len(errors),
        "total_bel"         : round(summary_df["bel"].sum(),             2),
        "total_ra"          : round(summary_df["risk_adjustment"].sum(), 2),
        "total_fcf"         : round(summary_df["fcf"].sum(),             2),
        "total_csm"         : round(summary_df["csm"].sum(),             2),
        "total_icl"         : round(summary_df["icl"].sum(),             2),
        "onerous_contracts" : int(summary_df["is_onerous"].sum()),
        "total_onerous_loss": round(summary_df["onerous_loss"].sum(),    2),
        "by_product"        : by_product,
    }
    return totals


def _print_portfolio_summary(ps):
    """Print formatted portfolio summary."""
    print(f"\n  {'═'*55}")
    print(f"  IFRS 17 GMM — PORTFOLIO VALUATION RESULTS")
    print(f"  {'═'*55}")
    print(f"  Total policies        : {ps['total_policies']:>10,}")
    print(f"  Valuation errors      : {ps['valuation_errors']:>10,}")
    print(f"  Onerous contracts     : {ps['onerous_contracts']:>10,}")
    print(f"  {'─'*55}")
    print(f"  Total BEL             : R{ps['total_bel']:>18,.2f}")
    print(f"  Total Risk Adjustment : R{ps['total_ra']:>18,.2f}")
    print(f"  Total FCF             : R{ps['total_fcf']:>18,.2f}")
    print(f"  Total CSM             : R{ps['total_csm']:>18,.2f}")
    print(f"  {'─'*55}")
    print(f"  Total ICL             : R{ps['total_icl']:>18,.2f}")
    print(f"  {'═'*55}")
    print(f"\n  By Product:")
    print(f"  {ps['by_product'][['count','total_icl','total_csm','onerous_count']].to_string()}")
