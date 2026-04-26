# IFRS 17 / SAM Life Insurance Actuarial Valuation Engine

> A live, end-to-end actuarial valuation engine for South African life insurance contracts. Implements the IFRS 17 General Measurement Model and the SAM (Solvency Assessment and Management) framework in Python — covering data ingestion, cash flow projection, Best Estimate Liability, Risk Adjustment, Contractual Service Margin, and assumption sensitivity analysis across a live Streamlit dashboard.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)]([https://your-streamlit-url.streamlit.app](https://ifrs17-sam-actuarial-engine-z667ztlsfcbysw7ggyomny.streamlit.app))
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![IFRS 17](https://img.shields.io/badge/Standard-IFRS%2017-1a237e?style=for-the-badge)](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-17-insurance-contracts/)
[![SAM](https://img.shields.io/badge/Framework-SAM%20%2F%20Solvency%20II-2e7d32?style=for-the-badge)](https://www.fsca.co.za)

---

## Table of Contents

- [The Business Problem](#the-business-problem)
- [Why This Project Is Distinctive](#why-this-project-is-distinctive)
- [Live Demo](#live-demo)
- [System Architecture](#system-architecture)
- [The Four Engine Layers](#the-four-engine-layers)
- [IFRS 17 Implementation](#ifrs-17-implementation)
- [Data Ingestion System](#data-ingestion-system)
- [Actuarial Assumptions](#actuarial-assumptions)
- [Stakeholder Report](#stakeholder-report)
- [Products Supported](#products-supported)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Key Concepts Explained](#key-concepts-explained)
- [Real-World Applications](#real-world-applications)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## The Business Problem

Every South African life insurer — Old Mutual, Sanlam, Discovery, Momentum, Liberty — was required to implement IFRS 17 by January 2023. The standard fundamentally changed how insurance contract liabilities are measured, reported, and disclosed. Most insurers spent tens of millions of rands on proprietary actuarial platforms such as Prophet, Moses, and ResQ to perform these calculations.

This project builds the mathematical core of what those platforms do — in open-source Python — making the following capabilities accessible without a commercial licence:

- Ingest policy data from any source system format (CSV, Excel, JSON) via a configurable mapping layer
- Project future policyholder cash flows using South African assured lives mortality tables
- Compute IFRS 17 Fulfilment Cash Flows (BEL + Risk Adjustment)
- Calculate the Contractual Service Margin and identify onerous contracts
- Produce the Insurance Contract Liability for disclosure
- Run eight assumption sensitivity scenarios and generate a board-ready stakeholder report

The target audience is smaller South African insurers, funeral parlour groups, actuarial consulting firms doing IFRS 17 implementation work, and actuarial students building technical portfolio projects.

---

## Why This Project Is Distinctive

Most data science portfolio projects use publicly available datasets with well-understood problem structures. This project is different in three ways.

**It requires three simultaneous knowledge domains.** The engine combines actuarial science (mortality decrement models, reserve methodology), accounting standards (IFRS 17 GMM building blocks, CSM mechanics), and software engineering (modular Python architecture, configurable YAML ingestion layer). Very few practitioners hold all three simultaneously.

**It solves a real institutional problem.** The data ingestion system with YAML mapping configs directly addresses the real-world challenge of transforming messy policy administration system exports into a clean canonical format — a problem every insurer's actuarial team faces at every quarter-end valuation.

**The output is a regulated disclosure.** The Insurance Contract Liability calculated by this engine is the number that appears on a life insurer's IFRS balance sheet, is audited by an external auditor, and is reviewed by the Prudential Authority. That institutional context gives the project direct commercial relevance.

---

## Live Demo

**[→ Open the Live Dashboard]([https://your-streamlit-url.streamlit.app](https://ifrs17-sam-actuarial-engine-z667ztlsfcbysw7ggyomny.streamlit.app))**

On the live dashboard you can:
- Click **Load Demo Portfolio** to instantly load 993 synthetic policies
- Navigate to **Run Valuation** and trigger a full IFRS 17 GMM valuation
- Explore the **Results Dashboard** with interactive waterfall, heatmap, and scatter charts
- Open the **Policy Detail** tab and drill into any individual policy's cash flow projection, CSM roll-forward, and P&L emergence schedule
- Open the **Stakeholder Report** tab for the board-ready summary including eight sensitivity scenarios

---

## System Architecture

The engine processes data through six sequential stages:

```
SOURCE DATA                       INGESTION LAYER
──────────────────────────────────────────────────────────────────────
CSV  │                 Step 1: Format Reader
Excel│ ──────────────► Step 2: Column Mapper     (YAML config-driven)
JSON │                 Step 3: Field Deriver     (derived fields)
                       Step 4: Validator         (3-tier rules)
                       Step 5: IFRS 17 Classifier(portfolio/cohort)
                       Step 6: Canonical Output  (clean parquet)
                                    │
                                    ▼
                       PROJECTION ENGINE
──────────────────────────────────────────────────────────────────────
                       Mortality Table (SA85-90)
                       Lapse Decrements
                       Expense Loadings
                       Discount Curve (ZAR risk-free)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
               IFRS 17 GMM      SAM BEL        SAM SCR
               ───────────      ───────        ──────────
               BEL              Best           Mortality
               Risk Adj         Estimate       Lapse
               FCF              Liability      Expense
               CSM              Risk Margin    BSCR
               ICL              Tech Prov      Capital Req
                                    │
                                    ▼
                       STREAMLIT DASHBOARD
──────────────────────────────────────────────────────────────────────
               Data Upload → Quality Report → Run Valuation
               Results Dashboard → Policy Detail → Stakeholder Report
```

---

## The Four Engine Layers

### Layer 1 — Data Ingestion (`ingestion/`)

The ingestion layer transforms raw policy administration system exports into a standardised canonical format. It is designed to be source-system agnostic — a single YAML mapping file per source system is all that is needed to onboard a new insurer's data. The engine code itself never changes.

The pipeline runs six steps in sequence.

**Step 1 — Format Reader (`reader.py`):** Detects file format from extension and reads the file using configuration-driven parameters. Handles CSV with any delimiter and any number of skip rows, Excel with multiple sheets and metadata rows, and JSON in both direct array and API envelope structures.

**Step 2 — Column Mapper (`mapper.py`):** Renames source columns to canonical names using the `field_mappings` section of the YAML file. Converts coded source values to standard values using `value_mappings` — for example, gender code `"1"` becomes `"M"`, premium frequency code `"M"` becomes `"MONTHLY"`.

**Step 3 — Field Deriver (`deriver.py`):** Computes all derived fields that the valuation engine requires but that do not exist directly in source data. Age at entry is derived from date of birth and inception date using exact relativedelta calculations. Annualised premium is derived from premium amount and frequency. IFRS 17 cohort IDs are assigned based on inception year and product.

**Step 4 — Validation Engine (`validator.py`):** Applies three tiers of rules to every record. Tier 1 fatal errors exclude records from valuation — missing date of birth, non-positive sum assured, expiry before inception. Tier 2 warnings flag records for review without exclusion — unusually high age, large sum assured without reinsurance treaty. Tier 3 anomalies flag statistical outliers more than three standard deviations from portfolio mean.

**Step 5 — IFRS 17 Classifier:** Assigns each policy to an IFRS 17 portfolio and annual cohort. IFRS 17 prohibits mixing policies from different annual cohorts within the same measurement group.

**Step 6 — Canonical Output:** Saves validated records as Parquet for efficient downstream processing, along with a JSON data quality report for audit trail.

---

### Layer 2 — Mortality Engine (`engine/mortality.py`)

Implements the SA85-90 assured lives mortality table for South African life insurance business. Provides qx lookup (probability of dying within one year) for any age between 0 and 99, across four combinations of gender (Male/Female) and smoker status (Smoker/Non-Smoker).

The table is built using a Gompertz-Makeham mortality law parameterised to reproduce the shape and magnitude of the published ASSA SA85-90 tables:

```
μ(x) = A + B × c^x
```

Female mortality is approximately 70–85% of male mortality, converging at older ages. Smoker loading is 1.5× non-smoker at younger ages, reducing toward 1.1× at age 70, consistent with published South African experience studies.

The engine also computes curtate life expectancy for validation — a 40-year-old non-smoker male should expect approximately 35–38 additional years of life under these tables.

---

### Layer 3 — Cash Flow Projector (`engine/projector.py`)

The mathematical core of the entire system. Projects future cash flows for a single policy from the valuation date to expiry using a deterministic year-by-year decrement model.

At each future time step t, the projection applies three decrements in sequence:

```
Mortality decrement:
    d(t) = l(t) × qx(age+t, gender, smoker) × mortality_multiplier

Lapse decrement (applied after deaths, UDD assumption):
    w(t) = [l(t) - d(t)] × lapse_rate(policy_year)

Survivors into next year:
    l(t+1) = l(t) - d(t) - w(t)
```

Cash flows are then computed for each year:

```
Premium   = l(t) × AP × (1 + esc)^t       [start of year → discount v(t)]
Claim     = d(t) × SA × (1 + esc)^t       [end of year → discount v(t+1)]
Expense   = l(t) × per_policy + prem × %  [end of year → discount v(t+1)]
Maturity  = l(n) × SA   (endowment final year only)
```

The present value of net cash flows is the Best Estimate Liability. A negative BEL means the policy is profitable — the present value of premiums exceeds the present value of outflows.

The projector also accepts a `mortality_multiplier` parameter for SAM stress testing — passing 1.20 applies a 20% mortality stress without modifying the base assumption set.

---

### Layer 4 — IFRS 17 GMM (`engine/ifrs17.py`)

Wraps the BEL with the two additional building blocks to produce the full Insurance Contract Liability.

**Risk Adjustment:** Calculated using a simplified confidence interval approach. A product-specific loading factor is applied to the absolute BEL, calibrated to approximately achieve the target confidence level (default 75th percentile). The loading is further adjusted for contract duration — longer duration contracts carry proportionally more uncertainty.

**Contractual Service Margin:** At inception, if FCF (BEL + RA) is negative the contract is profitable and the CSM is set to exactly offset the negative FCF. If FCF is positive the contract is onerous and the loss is recognised immediately in profit or loss with no CSM. The CSM is then rolled forward each year with interest accretion at the locked-in discount rate, release to income based on coverage units, and adjustments for non-economic assumption changes.

**Coverage Units:** Determines the CSM release pattern. Calculated as the expected in-force sum assured weighted by survival probability — l(t) × SA(t). This causes faster CSM release in early years when more policies are in force, and slower release later as the cohort reduces through decrements.

**Insurance Contract Liability:**

```
ICL = FCF + CSM  =  (BEL + RA) + CSM
```

---

## IFRS 17 Implementation

### General Measurement Model (GMM / Building Block Approach)

The engine implements the GMM as specified in IFRS 17 paragraphs 32–52. The three building blocks are:

| Block | Component | Description |
|---|---|---|
| 1 | Present Value of Future Cash Flows | Probability-weighted cash flows discounted at IFRS 17 rates |
| 1 | Risk Adjustment | Compensation for non-financial uncertainty (75th percentile) |
| 2 | Contractual Service Margin | Unearned profit released as service is provided |

### Cohort Assignment

IFRS 17 paragraph 14 prohibits offsetting contracts from different annual cohorts. The engine automatically assigns each policy a cohort ID in the format `{product_code}-{inception_year}` (e.g. `TERM-2022`) and a portfolio ID in the format `PORT-{product_code}-LIFE`.

### Onerousness Assessment

A contract is classified as onerous when FCF at inception is positive — meaning expected outflows exceed expected inflows on a present value basis. The engine flags every onerous contract, sets its CSM to zero, and recognises the full FCF as an immediate loss. The onerous count and total loss amount are disclosed in the stakeholder report.

---

## Data Ingestion System

### YAML Mapping Configuration

The ingestion layer is source-system agnostic. A single YAML file per source system tells the engine how to read and transform that system's export. To onboard a new insurer's data, copy `mappings/generic_template.yaml`, rename it, and fill in the right-hand side of each mapping. The engine code never changes.

```yaml
source_system: "ABC Life LifePRO"
file_formats:
  csv:
    delimiter: "|"
    skip_rows: 1
  excel:
    sheet_name: "Policies"
  json:
    records_path: "policies"

date_format: "%d/%m/%Y"

field_mappings:
  policy_id:     "POL_NUM"
  date_of_birth: "CLNT_DOB"
  sum_assured:   "BASIC_SA"

value_mappings:
  gender:
    "1": "M"
    "2": "F"
  product_code:
    "TL01": "TERM"
    "WL01": "WL"
    "EN01": "ENDOW"
```

### Canonical Policy Record

After ingestion every policy contains these standardised fields regardless of source system:

```
Identity:        policy_id, insurer_id, portfolio_id, cohort_id
Demographics:    date_of_birth, gender, smoker_status
                 age_at_entry (derived), age_at_valuation (derived)
Policy Terms:    product_code, inception_date, expiry_date
                 remaining_term_years (derived), policy_duration_years (derived)
Financial:       sum_assured, annualised_premium (derived)
                 premium_frequency, escalation_rate
Reinsurance:     reinsurance_flag, ri_retention, ri_ceded, ri_treaty_id
IFRS 17:         measurement_model, cohort_id, portfolio_id, onerous_flag
Status:          policy_status, valuation_flag
```

### Validation Rules

| Tier | Treatment | Example Rules |
|---|---|---|
| Tier 1 — Fatal | Record excluded from valuation | Missing DOB, negative SA, duplicate policy ID, expiry before inception |
| Tier 2 — Warning | Included but flagged | Age at entry > 65, SA > R50M, RI flag without treaty ID |
| Tier 3 — Anomaly | Included and logged | Sum assured > 3σ from portfolio mean |

---

## Actuarial Assumptions

All assumptions are configurable via the Streamlit sidebar. Changes immediately update the full valuation when Run Valuation is clicked.

### Discount Rate

The ZAR risk-free discount curve uses a term structure calibrated to SARB-published government bond yields. A flat rate approximation is available. An illiquidity premium is added for long-duration liabilities per the IFRS 17 bottom-up approach.

| Term | Rate (approximate, Q4 2024) |
|---|---|
| 1 year | 8.20% |
| 5 years | 8.90% |
| 10 years | 9.50% |
| 20 years | 9.90% |
| 30 years | 10.00% |

### Lapse Rates

Lapse rates follow the South African retail life experience pattern — highest in years 1–3, declining to a long-term rate from year 5 onwards.

| Policy Year | Default Lapse Rate |
|---|---|
| Year 1 | 12.0% |
| Year 2 | 9.0% |
| Year 3 | 7.0% |
| Year 4 | 5.5% |
| Long-term | 3.5% |

### Expense Assumptions

| Item | Default |
|---|---|
| Per-policy annual expense | R350 |
| Per-premium loading | 2.5% |
| Claim processing cost | R500 per claim |
| Expense inflation | 5.5% per annum |

### Risk Adjustment Confidence Level

Default 75th percentile. Loading factors by product: Term 15%, Whole Life 12.5%, Endowment 10%, with a duration adjustment of +0.5% per year beyond 5 remaining years.

---

## Stakeholder Report

The Stakeholder Report tab generates a board-ready summary covering four sections automatically after any valuation run.

**Section 1 — Data Quality:** Validation outcome gauge, exclusion reason breakdown, portfolio composition by product, status and age, and a full portfolio statistics table. Designed for the external auditor and data governance team.

**Section 2 — Best Estimate Liability:** BEL by product, individual policy BEL distribution histogram, age-versus-remaining-term heatmap showing liability concentration, and a statistics table by product. Designed for the appointed actuary and CFO.

**Section 3 — Contractual Service Margin:** CSM versus onerous loss by product, profitable versus onerous contract count, 15-year aggregate CSM roll-forward chart showing interest accretion and release to revenue, and CSM/ICL ratio table. Designed for the CFO and Board Audit Committee.

**Section 4 — Assumption Sensitivity Analysis:** Eight full portfolio re-valuations run automatically, producing a tornado chart of ICL impact per scenario, a detailed results table, and automatically generated actuary's commentary identifying the most and least sensitive assumptions. Designed for external auditors, the FSCA Prudential Authority, and the Board Risk Committee.

### The Eight Sensitivity Scenarios

| Scenario | Stress Applied |
|---|---|
| Mortality +10% | 10% increase in qx across all ages |
| Mortality +20% | 20% increase — SAM mortality stress benchmark |
| Mortality -20% | 20% decrease — longevity risk stress |
| Lapse +50% | 50% increase in all lapse rates |
| Lapse -50% | 50% decrease in all lapse rates |
| Expense +10% | 10% increase in all expenses + 1% additional inflation |
| Discount Rate +100bps | Parallel upward shift in yield curve |
| Discount Rate -100bps | Parallel downward shift in yield curve |

---

## Products Supported

| Product | Measurement Model | Description | Key Features |
|---|---|---|---|
| Term Assurance | GMM | Pure mortality cover for a fixed term | Death benefit only, no maturity value |
| Whole Life | GMM | Lifelong mortality cover to age 100 | No expiry, long projection horizon |
| Endowment | GMM | Savings and protection over a fixed term | Death benefit or maturity benefit — whichever comes first |

### Synthetic Portfolio (Demo Mode)

The engine ships with a synthetic portfolio generator that creates 1,000 realistic but entirely fictional policies:

| Product | Count | Sum Assured Range |
|---|---|---|
| Term Assurance | ~500 | R100k – R5M |
| Whole Life | ~300 | R50k – R2M |
| Endowment | ~200 | R50k – R3M |

The generator deliberately introduces messy data characteristics that mirror real policy administration system exports: pipe-delimited fields, coded values, DD/MM/YYYY date format, and a 2% rate of deliberate data quality issues for validation testing.

---

## Tech Stack

| Category | Library | Purpose |
|---|---|---|
| Data ingestion | `pandas`, `pyyaml` | File reading, column mapping, YAML config |
| Date mathematics | `python-dateutil` | Exact age calculation via relativedelta |
| Actuarial projection | `numpy` | Decrement model, present value calculations |
| File formats | `openpyxl`, `pyarrow` | Excel reading, Parquet output |
| IFRS 17 engine | Pure Python | FCF, RA, CSM, ICL computation |
| Dashboard | `streamlit` | Live web application, six-tab interface |
| Visualisation | `plotly` | Waterfall, heatmap, tornado, scatter, histogram |
| Statistical | `scipy` | Distribution analysis |

---

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation

```bash
git clone https://github.com/Fikilesondach/ifrs17-sam-actuarial-engine.git
cd ifrs17-sam-actuarial-engine
pip install -r requirements.txt
```

### Generate Synthetic Data

```bash
python data_generator.py
```

Creates `data/synthetic/abc_life_export.csv`, `.xlsx`, and `.json` — 1,000 synthetic policies in all three formats.

### Generate Mortality Table

```bash
python create_mortality_table.py
```

Creates `data/mortality_tables/sa_mortality.csv` — SA85-90 parametric table across ages 0–99.

### Run the Ingestion Pipeline

```bash
python -c "
from ingestion.pipeline import run_pipeline
from datetime import date

df_valid, df_errors, report = run_pipeline(
    file_path      = 'data/synthetic/abc_life_export.csv',
    mapping_path   = 'mappings/synthetic_data.yaml',
    valuation_date = date(2024, 12, 31)
)
print(f'Records validated: {len(df_valid):,}')
print(f'Pass rate: {report[\"summary\"][\"pass_rate\"]:.1%}')
"
```

### Run the Valuation Engine

```bash
python -c "
import pandas as pd
from engine.assumptions import build_default_assumptions
from engine.ifrs17      import value_portfolio_ifrs17

canonical   = pd.read_parquet('outputs/canonical/policies_canonical.parquet')
assumptions = build_default_assumptions()

summary, results, portfolio = value_portfolio_ifrs17(
    canonical_df   = canonical,
    assumptions    = assumptions,
    valuation_date = '2024-12-31',
)
print(f'Total ICL: R{portfolio[\"total_icl\"]:,.2f}')
print(f'Total CSM: R{portfolio[\"total_csm\"]:,.2f}')
print(f'Onerous contracts: {portfolio[\"onerous_contracts\"]:,}')
"
```

### Launch the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Click **Load Demo Portfolio** on the first tab to begin immediately.

---

## Project Structure

```
ifrs17_sam_engine/
│
├── app.py                              # Streamlit dashboard (6 tabs)
├── data_generator.py                   # Synthetic portfolio generator
├── create_mortality_table.py           # SA85-90 mortality table builder
│
├── ingestion/                          # Data ingestion layer
│   ├── __init__.py
│   ├── reader.py                       # CSV / Excel / JSON format reader
│   ├── mapper.py                       # YAML-driven column mapper
│   ├── deriver.py                      # Derived field computation
│   ├── validator.py                    # 3-tier validation rules engine
│   ├── classifier.py                   # IFRS 17 cohort/portfolio assignment
│   └── pipeline.py                     # Step orchestration (steps 1–6)
│
├── engine/                             # Actuarial calculation engine
│   ├── __init__.py
│   ├── mortality.py                    # SA85-90 mortality table + qx lookup
│   ├── assumptions.py                  # Assumption dataclasses + stress builders
│   ├── projector.py                    # Year-by-year cash flow projection
│   └── ifrs17.py                       # IFRS 17 GMM — FCF, RA, CSM, ICL
│
├── mappings/                           # Source system mapping configs
│   ├── synthetic_data.yaml             # Mapping for generated demo data
│   └── generic_template.yaml           # Template for new source systems
│
├── data/
│   ├── synthetic/                      # Generated policy data (3 formats)
│   ├── mortality_tables/               # SA85-90 parametric CSV table
│   └── yield_curves/                   # ZAR risk-free rates
│
├── outputs/
│   ├── canonical/                      # Cleaned ingested data (Parquet)
│   ├── data_quality_reports/           # Validation JSON reports + error CSVs
│   └── valuation_results/             # IFRS 17 output CSVs
│
└── requirements.txt
```

---

## Key Concepts Explained

**What is IFRS 17 and why does it matter?**
IFRS 17 is the international accounting standard for insurance contracts that replaced IFRS 4 globally from January 2023. It fundamentally changed how insurers measure and report their liabilities — moving from a largely cost-based model to a current-value, service-based model. Every listed South African insurer's balance sheet and income statement is now directly governed by this standard.

**What is the Contractual Service Margin and why is it important?**
The CSM is the most significant conceptual innovation in IFRS 17. Under the old standard, an insurer could recognise day-one profit when selling a profitable long-term policy. Under IFRS 17, that profit is locked in the CSM and released to the income statement only as coverage is provided — matching revenue to service delivered. The CSM balance on the balance sheet represents the stock of unearned future insurance service revenue.

**What is the difference between BEL and ICL?**
The Best Estimate Liability is the present value of expected future net cash flows with no margins. It can be negative for profitable contracts. The Insurance Contract Liability adds the Risk Adjustment (compensation for uncertainty) and the CSM (unearned profit), producing the total non-negative liability that appears on the balance sheet.

**Why use a YAML mapping layer instead of hardcoding column names?**
Real actuarial teams spend enormous time at every quarter-end transforming raw policy administration exports into a format their models can consume. Hardcoding assumes a fixed source format — any change to the source system breaks the model. The YAML approach separates transformation logic from calculation logic: to support a new insurer you create one new YAML file and the engine remains unchanged.

**What does a negative BEL mean?**
A negative BEL means the present value of future premium income exceeds the present value of future claims and expenses. This is normal and expected for profitable term assurance policies — especially in early years when the policyholder is young and mortality rates are low. The CSM exists precisely to prevent this negative FCF from being recognised as immediate profit.

**Why is the Risk Adjustment always positive?**
The RA represents the amount an insurer would rationally pay to be relieved of the non-financial uncertainty in its liabilities. This compensation is always positive — a rational insurer always requires compensation for bearing risk, regardless of whether the expected outcome is profitable or loss-making.

**What is a coverage unit?**
Coverage units determine the CSM release pattern. They represent the quantum of insurance service provided each year. For life insurance the engine uses expected in-force sum assured weighted by survival probability — l(t) × SA(t). This means the CSM releases faster in early years when more policies are in force, and more slowly later when lapse and mortality have reduced the cohort.

**What does onerous mean in IFRS 17?**
A contract is onerous when its Fulfilment Cash Flows are positive at inception — expected claims and expenses exceed expected premiums on a present value basis. Onerous contracts carry no CSM and the loss is recognised immediately in profit or loss. This is one of the most commercially significant disclosures under IFRS 17 because it forces immediate loss recognition rather than allowing deferral.

---

## Real-World Applications

**Life Insurers — Valuation Teams:** A smaller insurer that cannot justify a Prophet or Moses licence can use this engine to produce defensible IFRS 17 quarterly valuations. The YAML ingestion layer connects to any policy administration system with a one-time configuration exercise.

**Actuarial Consulting Firms:** IFRS 17 implementation projects at mid-tier insurers require building or validating valuation models. This engine provides a transparent, auditable reference implementation against which proprietary systems can be benchmarked.

**Funeral Parlour Groups and Microinsurers:** Many South African funeral groups operate under PAA eligibility but must document that eligibility assessment. The GMM engine provides the counterfactual valuation required for that documentation.

**Prudential Authority / FSCA:** The Prudential Authority supervises insurer solvency under SAM and reviews IFRS 17 compliance. A transparent open-source implementation illustrates the mechanics of the standard in a way that regulatory analysts can inspect and stress-test.

**Actuarial Education:** The engine provides a complete working implementation of every concept taught in the CM1 actuarial examination syllabus — life tables, present values, reserves, decrement models — connected to the current professional accounting standard.

**Investment Analysts:** Analysts covering listed South African insurers need to understand how changes in mortality experience, lapse rates, and interest rates flow through to IFRS 17 liabilities and the income statement. The sensitivity module provides exactly that insight.

---

## Known Limitations

**Mortality Table Approximation:** The SA85-90 table is parameterised using a Gompertz-Makeham model calibrated to reproduce the published ASSA table's shape. A production deployment would load the exact published ASSA table values directly.

**Flat Rate Discount Approximation:** The full IFRS 17 discount curve should be derived from the SARB-published ZAR government bond yield curve, adjusted for illiquidity. A production system would connect to the SARB's weekly published rates.

**GMM Only — No PAA or VFA:** This version implements the General Measurement Model. The Premium Allocation Approach (short-duration contracts) and Variable Fee Approach (direct-participation contracts with investment components) are planned for future releases.

**CSM Locked-In Rate Approximation:** For in-force policies the locked-in discount rate should be the rate at the original inception date of each policy. The current engine approximates this using the current flat rate. A production system would store inception-date rates per cohort.

**Deterministic Projection Only:** The engine uses deterministic best-estimate cash flow projection. IFRS 17 paragraph B26 allows stochastic techniques for contracts with financial options and guarantees. Stochastic projection is planned for the VFA implementation.

**Gross of Reinsurance:** While the ingestion layer captures reinsurance flags and treaty references, the projection engine currently models gross cash flows only. Net-of-reinsurance liability calculation is a planned enhancement.

---

## Roadmap

- [ ] SAM Best Estimate Liability and Risk Margin
- [ ] SAM SCR Standard Formula — all life risk sub-modules
- [ ] Premium Allocation Approach (PAA) for short-duration contracts
- [ ] Variable Fee Approach (VFA) for investment-linked products
- [ ] Net-of-reinsurance cash flows
- [ ] SARB live yield curve integration
- [ ] Exact ASSA SA85-90 published table loading
- [ ] Cohort-level locked-in discount rate storage
- [ ] Disability income and critical illness product support
- [ ] Multi-currency support (USD, EUR for offshore business)
- [ ] Formatted PDF/Excel stakeholder report export

---

## Regulatory References

- **IFRS 17 Insurance Contracts** — International Accounting Standards Board (IASB), 2017, amended 2020
- **IFRS 17 Illustrative Examples** — IASB Implementation Guidance
- **SAM Position Papers** — Prudential Authority / Financial Sector Conduct Authority (FSCA), South Africa
- **ASSA SA85-90 Assured Lives Mortality Tables** — Actuarial Society of South Africa
- **SARB Weekly Yield Curves** — South African Reserve Bank (resbank.co.za)

---

## Disclaimer

This project is built for educational, research, and portfolio demonstration purposes. It does not constitute a signed actuarial opinion or a certified IFRS 17 valuation. Results should not be used for financial reporting, regulatory submission, or investment decisions without review by a qualified actuary. All policies in the demonstration dataset are entirely synthetic — no real policyholder data is used anywhere in this project.

---

*Built with Python · IFRS 17 General Measurement Model · SA85-90 Mortality Tables · Deployed on Streamlit Cloud*
