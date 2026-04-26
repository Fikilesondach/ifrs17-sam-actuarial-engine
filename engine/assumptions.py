# engine/assumptions.py
# ══════════════════════════════════════════════════════════════════════
# ASSUMPTION SET
#
# Encapsulates all non-mortality assumptions used in the projection:
#   - Lapse rates (probability of voluntary policy termination)
#   - Expense loadings (per-policy and per-premium costs)
#   - Discount rates (ZAR risk-free yield curve)
#   - Tax rates
#
# IFRS 17 requires best estimate assumptions — no margins, no prudence.
# SAM BEL uses the same best estimate assumptions.
# SAM SCR stresses each assumption in turn (Phase 4).
#
# The assumption set is a dataclass so it can be easily passed around,
# modified for sensitivity testing, and serialised to JSON for audit.
# ══════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import numpy as np
import json


@dataclass
class LapseAssumptions:
    """
    Lapse rate assumptions by policy year.

    Lapse rates are typically highest in years 1-3 (early lapse)
    and then settle to a lower long-term rate. This pattern is
    well-documented in South African life insurance experience studies.

    All rates are annual probabilities (0.0 to 1.0).
    """
    # Year-specific lapse rates (policy year → annual lapse rate)
    # After the last specified year, long_term_rate applies
    year_rates: Dict[int, float] = field(default_factory=lambda: {
        1: 0.120,    # 12% lapse in year 1 — very common to lapse early
        2: 0.090,    # 9%  in year 2
        3: 0.070,    # 7%  in year 3
        4: 0.055,    # 5.5% in year 4
        5: 0.045,    # 4.5% in year 5
    })
    long_term_rate: float = 0.035    # 3.5% per year after year 5

    def get_rate(self, policy_year: int) -> float:
        """Get lapse rate for a given policy year (1-indexed)."""
        return self.year_rates.get(policy_year, self.long_term_rate)

    def get_series(self, n_years: int) -> list:
        """Get lapse rates for all projection years."""
        return [self.get_rate(t + 1) for t in range(n_years)]


@dataclass
class ExpenseAssumptions:
    """
    Expense assumptions.

    Expenses are split into:
      1. Per-policy expenses — fixed cost per policy per year
         (administration, IT, regulatory reporting)
      2. Per-premium expenses — variable cost as % of premium
         (collection, commission-related costs)
      3. Claim expenses — cost of processing each claim
      4. Acquisition expenses — one-off cost at policy inception
         (distribution, underwriting, policy issuance)

    All monetary amounts in ZAR.
    Expense inflation is applied to per-policy and claim expenses.
    """
    # Annual recurring expenses
    per_policy_annual:   float = 350.00    # R350 per policy per year
    per_premium_pct:     float = 0.025     # 2.5% of premium received
    claim_processing:    float = 500.00    # R500 per claim paid

    # One-off acquisition expenses (at policy inception)
    acquisition_per_policy: float = 2_500.00   # R2,500 per new policy
    acquisition_pct_ap:     float = 0.150       # 15% of first year AP

    # Expense inflation rate (per annum)
    inflation_rate: float = 0.055    # 5.5% — SA long-term CPI proxy

    def inflated(self, base_amount: float, years: int) -> float:
        """Apply expense inflation to a base amount over n years."""
        return base_amount * (1 + self.inflation_rate) ** years


@dataclass
class DiscountAssumptions:
    """
    Discount rate assumptions for present value calculations.

    IFRS 17 requires discounting at rates that reflect the
    characteristics of the insurance contract liabilities —
    specifically the time value of money and the characteristics
    of the cash flows.

    Two approaches:
      Bottom-up  : Risk-free rate + illiquidity premium
      Top-down   : Asset earned rate − investment risk adjustment

    We use bottom-up (more common in SA).

    The ZAR risk-free curve is published weekly by the SARB.
    For the flat rate approximation we use the 10Y ZAR government bond yield.
    """
    # Flat rate approximation (used when no full yield curve is available)
    flat_rate: float = 0.0950    # 9.5% — approximate 10Y ZAR bond yield 2024

    # Illiquidity premium added to risk-free (for long-duration contracts)
    illiquidity_premium: float = 0.0025   # 25 basis points

    # Full term structure (optional — overrides flat_rate if provided)
    # Format: {term_years: annual_rate}
    term_structure: Optional[Dict[int, float]] = field(default_factory=lambda: {
        1:  0.0820,
        2:  0.0845,
        3:  0.0860,
        5:  0.0890,
        7:  0.0920,
        10: 0.0950,
        15: 0.0975,
        20: 0.0990,
        30: 0.1000,
    })

    def get_discount_factor(self, t: int) -> float:
        """
        Get the present value discount factor for time t years.

        v(t) = 1 / (1 + r(t))^t

        Uses the term structure if available, otherwise flat rate.
        """
        if self.term_structure and t in self.term_structure:
            rate = self.term_structure[t]
        elif self.term_structure:
            # Interpolate between known terms
            terms = sorted(self.term_structure.keys())
            rates = [self.term_structure[k] for k in terms]
            rate  = float(np.interp(t, terms, rates))
        else:
            rate = self.flat_rate + self.illiquidity_premium

        return 1.0 / (1.0 + rate) ** t

    def get_discount_series(self, n_years: int) -> list:
        """Get discount factors v(0), v(1), ..., v(n_years)."""
        return [self.get_discount_factor(t) for t in range(n_years + 1)]


@dataclass
class AssumptionSet:
    """
    Complete assumption set for IFRS 17 / SAM valuation.

    This is the single object passed to the projection engine.
    It contains all assumptions needed for a full valuation run.

    For IFRS 17 the locked-in assumptions (used for CSM) are set
    at inception and never updated. The current assumptions are
    updated at each valuation date.
    """
    name:        str = "Best Estimate — Q4 2024"
    basis:       str = "BEL"     # "BEL", "IFRS17_CURRENT", "IFRS17_LOCKED"
    description: str = "Best estimate assumptions for IFRS 17 and SAM BEL"

    mortality:   LapseAssumptions    = field(default_factory=LapseAssumptions)
    lapse:       LapseAssumptions    = field(default_factory=LapseAssumptions)
    expenses:    ExpenseAssumptions  = field(default_factory=ExpenseAssumptions)
    discount:    DiscountAssumptions = field(default_factory=DiscountAssumptions)

    # Tax rate on investment income
    tax_rate: float = 0.28    # 28% corporate tax rate (SA)

    # Risk Adjustment — confidence level for RA calculation (IFRS 17)
    ra_confidence_level: float = 0.750   # 75th percentile

    # Cost of capital rate for SAM Risk Margin
    cost_of_capital: float = 0.060       # 6% per SAM standard formula

    def to_dict(self):
        """Serialise assumption set to dictionary (for audit trail)."""
        return asdict(self)

    def to_json(self, path=None):
        """Export assumption set to JSON."""
        d = self.to_dict()
        if path:
            with open(path, "w") as f:
                json.dump(d, f, indent=2)
        return json.dumps(d, indent=2)

    def summary(self):
        """Print a human-readable summary of key assumptions."""
        print(f"\n  Assumption Set: {self.name}")
        print(f"  {'─'*45}")
        print(f"  Discount rate (flat)   : {self.discount.flat_rate:.2%}")
        print(f"  Illiquidity premium    : {self.discount.illiquidity_premium:.2%}")
        print(f"  Expense inflation      : {self.expenses.inflation_rate:.2%}")
        print(f"  Per-policy expense     : R{self.expenses.per_policy_annual:,.2f} p.a.")
        print(f"  Per-premium loading    : {self.expenses.per_premium_pct:.2%}")
        print(f"  Lapse rate (Year 1)    : {self.lapse.get_rate(1):.2%}")
        print(f"  Lapse rate (long-term) : {self.lapse.long_term_rate:.2%}")
        print(f"  Tax rate               : {self.tax_rate:.2%}")
        print(f"  RA confidence level    : {self.ra_confidence_level:.0%}")
        print(f"  Cost of capital (SAM)  : {self.cost_of_capital:.2%}")


def build_default_assumptions():
    """Return a standard best estimate assumption set."""
    return AssumptionSet(
        name        = "Best Estimate — Q4 2024",
        basis       = "BEL",
        lapse       = LapseAssumptions(),
        expenses    = ExpenseAssumptions(),
        discount    = DiscountAssumptions(),
    )


def build_stressed_assumptions(base: AssumptionSet, stress_type: str):
    """
    Apply a SAM Standard Formula stress to a base assumption set.
    Used in Phase 4 (SCR calculation).

    Stress types (per SAM Life SCR module):
        'mortality_up'   : 20% increase in qx rates
        'lapse_up'       : 50% increase in lapse rates
        'lapse_down'     : 50% decrease in lapse rates
        'expense_up'     : 10% increase in expenses + 1% extra inflation
    """
    import copy
    stressed = copy.deepcopy(base)
    stressed.basis = f"STRESSED_{stress_type}"

    if stress_type == "mortality_up":
        # Mortality stress: multiply all qx by 1.20
        # Implemented in projector by passing a mortality multiplier
        stressed.name = "Mortality Stress (+20% qx)"

    elif stress_type == "lapse_up":
        stressed.name = "Lapse Stress (+50% lapse rates)"
        for yr in stressed.lapse.year_rates:
            stressed.lapse.year_rates[yr] = min(
                stressed.lapse.year_rates[yr] * 1.50, 1.0
            )
        stressed.lapse.long_term_rate = min(
            stressed.lapse.long_term_rate * 1.50, 1.0
        )

    elif stress_type == "lapse_down":
        stressed.name = "Lapse Stress (-50% lapse rates)"
        for yr in stressed.lapse.year_rates:
            stressed.lapse.year_rates[yr] *= 0.50
        stressed.lapse.long_term_rate *= 0.50

    elif stress_type == "expense_up":
        stressed.name = "Expense Stress (+10% expenses)"
        stressed.expenses.per_policy_annual *= 1.10
        stressed.expenses.per_premium_pct   *= 1.10
        stressed.expenses.claim_processing  *= 1.10
        stressed.expenses.inflation_rate    += 0.01

    return stressed
