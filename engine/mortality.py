# engine/mortality.py
# ══════════════════════════════════════════════════════════════════════
# MORTALITY TABLE ENGINE
#
# Loads and interpolates the SA85-90 assured lives mortality table.
# Provides qx lookup for any age, gender, and smoker status.
#
# In a production system this would load the exact ASSA published tables.
# For this engine we use parametric tables calibrated to the SA85-90 shape.
#
# Key concept — qx:
#   qx is the probability that a life aged exactly x will die
#   before reaching age x+1.
#   It is the fundamental building block of all life insurance mathematics.
# ══════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from functools import lru_cache


# ── Column mapping: (gender, smoker_status) → table column ────────────
QX_COLUMN = {
    ("M", "NS") : "qx_M_NS",
    ("F", "NS") : "qx_F_NS",
    ("M", "S")  : "qx_M_S",
    ("F", "S")  : "qx_F_S",
}


class MortalityTable:
    """
    SA85-90 Assured Lives Mortality Table.

    Provides qx (annual mortality probability) for any age between
    0 and 99, for all gender/smoker combinations.

    Usage:
        table = MortalityTable()
        qx    = table.get_qx(age=45, gender="M", smoker="NS")
        # Returns probability of a 45-year-old non-smoker male dying
        # within the next year
    """

    def __init__(self, table_path="data/mortality_tables/sa_mortality.csv"):
        """Load and index the mortality table."""
        self._table = pd.read_csv(table_path)
        self._table = self._table.set_index("age")

        # Validate table structure
        required_cols = list(QX_COLUMN.values())
        missing = [c for c in required_cols if c not in self._table.columns]
        if missing:
            raise ValueError(f"Mortality table missing columns: {missing}")

        print(f"  Mortality table loaded: ages {self._table.index.min()}–"
              f"{self._table.index.max()} ✓")

    def get_qx(self, age, gender="M", smoker="NS"):
        """
        Get the annual mortality probability for a given age profile.

        Parameters
        ──────────
        age    : Exact age (will be rounded down to integer attained age)
        gender : "M" or "F"
        smoker : "S" (smoker) or "NS" (non-smoker)

        Returns
        ───────
        float : Probability of dying within the next year (between 0 and 1)
        """
        # Validate inputs
        gender = str(gender).upper().strip()
        smoker = str(smoker).upper().strip()

        if gender not in ("M", "F"):
            gender = "M"   # Default to male if unrecognised
        if smoker not in ("S", "NS"):
            smoker = "NS"  # Default to non-smoker if unrecognised

        # Use integer attained age (floor)
        int_age = max(0, min(99, int(age)))

        col = QX_COLUMN[(gender, smoker)]

        return float(self._table.loc[int_age, col])

    def get_qx_series(self, age_at_start, n_years, gender="M", smoker="NS"):
        """
        Get a series of annual qx values for a projection.

        Returns a list of qx values for ages:
            age_at_start, age_at_start+1, ..., age_at_start+n_years-1

        This is what the projection engine uses — one qx per future year.

        Parameters
        ──────────
        age_at_start : Age at the start of the projection (attained age)
        n_years      : Number of years to project
        gender       : "M" or "F"
        smoker       : "S" or "NS"

        Returns
        ───────
        list of float : qx values, one per projection year
        """
        return [
            self.get_qx(age_at_start + t, gender, smoker)
            for t in range(n_years)
        ]

    def curtate_expectation(self, age, gender="M", smoker="NS"):
        """
        Compute the curtate life expectancy from a given age.

        This is the expected number of complete future years lived.
        Used for sanity checking — a 40-year-old male should have
        approximately 35-40 years of curtate life expectancy.

        e_x = Σ [ t_p_x ] for t = 1 to omega
        where t_p_x = probability of surviving t years from age x
        """
        px_cumulative = 1.0
        expectation   = 0.0

        for t in range(age, 100):
            qx = self.get_qx(t, gender, smoker)
            px_cumulative *= (1 - qx)
            expectation   += px_cumulative

        return round(expectation, 2)

    def summary(self):
        """Print a summary of key mortality rates for validation."""
        print("\n  SA Mortality Table — Key Rates:")
        print(f"  {'Age':>5}  {'M NS':>8}  {'F NS':>8}  {'M S':>8}  {'F S':>8}")
        print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
        for age in [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]:
            print(
                f"  {age:>5}  "
                f"{self.get_qx(age,'M','NS'):>8.5f}  "
                f"{self.get_qx(age,'F','NS'):>8.5f}  "
                f"{self.get_qx(age,'M','S'):>8.5f}  "
                f"{self.get_qx(age,'F','S'):>8.5f}"
            )

        print(f"\n  Life expectancy at 40 (M, NS): "
              f"{self.curtate_expectation(40,'M','NS')} years")
        print(f"  Life expectancy at 40 (F, NS): "
              f"{self.curtate_expectation(40,'F','NS')} years")
